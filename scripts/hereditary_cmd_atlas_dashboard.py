#!/usr/bin/env python3
"""Hereditary-CMD-Atlas — Complete 8-Gene Congenital Muscular Dystrophy (CMD) Spectrum Atlas
(LAMA2 · COL6A1 · COL6A2 · COL6A3 · FKTN · POMT1 · POMT2 · POMGNT1).

LAMA2   (Laminin α2 chain; 3122 aa; 6q22.33; AR;
         MDC1A — ABSENT MEROSIN ON IHC PATHOGNOMONIC; white matter T2 hyperintensity
         leukodystrophy non-progressive; demyelinating neuropathy; most common CMD globally;
         severe: non-ambulant by 1yr; partial deficiency: ambulant;
         seed SEED_BASE+0).
COL6A1  (Collagen VI α1 chain; 1028 aa; 21q22.3; AD/AR;
         Ullrich CMD (severe) / Bethlem Myopathy (mild AD) —
         PROXIMAL WEAKNESS + DISTAL JOINT HYPERLAXITY COMBINATION PATHOGNOMONIC Ullrich;
         FOLLICULAR HYPERKERATOSIS + KELOID SCARRING skin PATHOGNOMONIC Ullrich;
         early respiratory failure; collagen VI IHC reduction;
         seed SEED_BASE+1).
COL6A2  (Collagen VI α2 chain; 1019 aa; 21q22.3; AR/AD;
         Same Ullrich/Bethlem spectrum as COL6A1 — β-chain of collagen VI triple helix;
         21q22.3 — same chromosome as COL6A1 (cosegregation testing needed);
         seed SEED_BASE+2).
COL6A3  (Collagen VI α3 chain; 3177 aa; 2q37.3; AR/AD;
         Same Ullrich/Bethlem spectrum — LARGEST collagen VI subunit; unique N-terminal domain;
         keloid + follicular hyperkeratosis Ullrich; 2q37.3 — different chromosome from COL6A1/A2;
         seed SEED_BASE+3).
FKTN    (Fukutin; 461 aa; 9q31.2; AR;
         Fukuyama CMD — most common CMD in Japan (1/6500-10000);
         INTELLECTUAL DISABILITY MANDATORY; COBBLESTONE LISSENCEPHALY (type II) PATHOGNOMONIC;
         retinal dysplasia; DCM mandatory adolescence; c.3036+IVS retrotransposon Japan founder;
         seed SEED_BASE+4).
POMT1   (Protein O-mannosyltransferase 1; 747 aa; 9q34.13; AR;
         Walker-Warburg Syndrome (WWS) type 1 — MOST SEVERE glycosylation CMD;
         COBBLESTONE LISSENCEPHALY + BRAINSTEM HYPOPLASIA + CEREBELLAR HYPOPLASIA + OCULAR DEFECTS PATHOGNOMONIC;
         USUALLY LETHAL 1st year; absent αDG glycosylation;
         seed SEED_BASE+5).
POMT2   (Protein O-mannosyltransferase 2; 750 aa; 14q24.3; AR;
         Walker-Warburg Syndrome type 2 — IDENTICAL phenotype to POMT1;
         POMT1+POMT2 obligate heterodimer — BOTH required for O-mannosyltransferase activity;
         cobblestone lissencephaly; some alleles cause MEB-like (less severe than WWS);
         seed SEED_BASE+6).
POMGNT1 (POMGnT1; 660 aa; 1p34.1; AR;
         Muscle-Eye-Brain (MEB) disease — less severe than WWS;
         MYOPIA + GLAUCOMA + CEREBELLAR HYPOPLASIA COMBINATION PATHOGNOMONIC;
         Finnish founder p.Tyr688Cys 85% Finnish alleles; pachygyria; ID; survival into adulthood;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2214-2221).
"""

import random

SEED_BASE = 2214

CMD_GENES = [
    # -- LAMA2 — Laminin α2 / Merosin, MDC1A -----------------------------------------------
    {
        "gene": "LAMA2",
        "alt_name": (
            "LAMA2 (LAMA2-3122aa-6q22.33 / AR — MDC1A-Merosin-Deficient-CMD-Most-Common-CMD-Globally — "
            "ABSENT-MEROSIN-IHC-PATHOGNOMONIC — "
            "WHITE-MATTER-T2-LEUKODYSTROPHY-NON-PROGRESSIVE — "
            "CK-5-30x-ULN-Demyelinating-Neuropathy-Respiratory-Failure)"
        ),
        "protein": (
            "LAMA2 -- 6q22.33 AR -- LAMA2-3122aa -- "
            "Laminin-Alpha2-Chain-Merosin-395kDa-Basement-Membrane-Extracellular-Matrix -- "
            "MDC1A-OMIM-607855 -- "
            "ABSENT-MEROSIN-IHC-Muscle-AND-Skin-Biopsy-PATHOGNOMONIC-Complete-Deficiency -- "
            "WHITE-MATTER-T2-HYPERINTENSITY-Brain-MRI-Non-Progressive-Leukodystrophy-80-100pct -- "
            "DEMYELINATING-PERIPHERAL-NEUROPATHY-NCS-Slow-Conduction-Velocity -- "
            "RESPIRATORY-FAILURE-Mandatory-NIV-Monitoring -- "
            "CK-5-30x-ULN-Marked-Elevation -- "
            "Severe-Complete-Deficiency-Non-Ambulant-1yr-Partial-Deficiency-Ambulant-Later-Onset -- "
            "No-Intellectual-Disability-Unless-Seizures -- "
            "Cardiac-Usually-Spared-Monitor-Adolescence -- "
            "Gene-Therapy-Trials-LAMA2-CMD-AAV -- "
            "OMIM-Gene-LAMA2-156225-Disease-MDC1A-607855"
        ),
        "locus": "6q22.33",
        "protein_size": "3122 aa / 395 kDa",
        "inheritance": (
            "AR (biallelic loss-of-function — most common); "
            "Complete null: severe MDC1A (non-ambulant by 1yr, full merosin IHC absence); "
            "Partial loss: milder LGMD-like phenotype (ambulant, partial IHC reduction); "
            "CK 5-30× ULN (markedly elevated); "
            "Onset: at birth or 1st few months (congenital hypotonia, contractures); "
            "Brain MRI: T2 white matter hyperintensity 80-100% (non-progressive leukodystrophy); "
            "Peripheral neuropathy: demyelinating pattern on NCS (NCVs <40 m/s); "
            "Cardiac: usually spared; monitor in adolescence; "
            "Respiratory: NIV often 1st decade; "
            "No intellectual disability (unless epilepsy-related)"
        ),
        "key_features": [
            "ABSENT MEROSIN ON IHC — complete deficiency pathognomonic; muscle and skin biopsy (skin biopsy safe screening)",
            "White matter T2 leukodystrophy on brain MRI — non-progressive; present 80-100% of MDC1A (not causing disability)",
            "Demyelinating peripheral neuropathy — slow NCVs <40 m/s; diagnostic clue distinguishing MDC1A from other CMD",
            "CK 5-30× ULN — markedly elevated in complete deficiency",
            "Congenital hypotonia + contractures — presentation at birth or 1st months",
            "Respiratory failure — mandatory NIV monitoring from school age; FVC trajectory",
            "No intellectual disability — brain MRI looks alarming but cognition preserved unless seizures",
            "Partial deficiency → milder LGMD-like phenotype, ambulant, slower progression",
        ],
        "treatment": (
            "Respiratory: FVC monitoring 6-monthly; NIV when FVC <60% predicted or nocturnal hypoventilation; "
            "Physiotherapy: stretch contractures, bracing AFOs; "
            "Cardiac: annual echo from adolescence; "
            "Nutrition: gastrostomy if dysphagia; "
            "Epilepsy: AED if seizures; "
            "Gene therapy: AAV-LAMA2 trials ongoing (preclinical success, phase I planned); "
            "Deflazacort: limited benefit data; "
            "No disease-modifying approved therapy"
        ),
        "contraindications": (
            "Depolarising neuromuscular blockers (suxamethonium) — risk hyperkalaemia/rhabdomyolysis; "
            "Vigorous physiotherapy beyond tolerance; "
            "Immunosuppression without biopsy-verified inflammatory component (can mimic Bethlem/Ullrich); "
            "Skin biopsy → preferred to muscle biopsy for merosin IHC in partial cases"
        ),
        "critical_pearls": [
            "Skin biopsy for merosin IHC — safe, outpatient; equivalent sensitivity to muscle biopsy for MDC1A screening",
            "Brain MRI white matter changes are NON-PROGRESSIVE and NOT causing cognitive impairment — do not alarm families; essential DDx from Pelizaeus-Merzbacher",
            "Partial merosin deficiency (some bands on WB) → ambulant CMD or LGMD-R24 — milder but same gene",
            "Demyelinating neuropathy — if CMD + slow NCVs + absent merosin → LAMA2 confirmed; NCVs can help distinguish from Ullrich (NCS normal Ullrich)",
            "Serum CK is markedly elevated (5-30×) — helpful DDx from Bethlem Myopathy (CK 1-4×) and Ullrich CMD (CK near-normal or mildly elevated)",
        ],
    },

    # -- COL6A1 — Collagen VI α1, Ullrich CMD / Bethlem Myopathy ----------------------------
    {
        "gene": "COL6A1",
        "alt_name": (
            "COL6A1 (COL6A1-1028aa-21q22.3 / AD-AR — Ullrich-CMD-Bethlem-Myopathy-Collagen-VI-α1 — "
            "PROXIMAL-WEAKNESS-DISTAL-HYPERLAXITY-PARADOX-PATHOGNOMONIC-Ullrich — "
            "FOLLICULAR-HYPERKERATOSIS-KELOID-SKIN-PATHOGNOMONIC-Ullrich — "
            "Early-Respiratory-Failure-Collagen-VI-IHC-Absent)"
        ),
        "protein": (
            "COL6A1 -- 21q22.3 AD/AR -- COL6A1-1028aa -- "
            "Collagen-VI-Alpha-1-Chain-140kDa-Triple-Helix-Basement-Membrane -- "
            "Ullrich-CMD-UCMD-OMIM-254090-Severe-AR -- "
            "Bethlem-Myopathy-BM-OMIM-158810-Mild-AD -- "
            "PROXIMAL-WEAKNESS-DISTAL-JOINT-HYPERLAXITY-PARADOX-PATHOGNOMONIC-Ullrich -- "
            "FOLLICULAR-HYPERKERATOSIS-KELOID-SCARRING-Skin-PATHOGNOMONIC-Ullrich -- "
            "MULTIPLE-CONTRACTURES-Elbow-Hip-Knee-Ankle-Ullrich -- "
            "EARLY-RESPIRATORY-FAILURE-NIV-School-Age-Ullrich -- "
            "Collagen-VI-IHC-Absent-Reduced-Basement-Membrane-Muscle-Biopsy -- "
            "CK-Normal-1-5x-ULN-Mildly-Elevated -- "
            "No-Cardiac-No-Intellectual-Disability -- "
            "OMIM-Gene-COL6A1-120220-Disease-Ullrich-254090-Bethlem-158810"
        ),
        "locus": "21q22.3",
        "protein_size": "1028 aa / 140 kDa",
        "inheritance": (
            "AD (Bethlem Myopathy — dominant glycine substitutions in triple helix Gly-X-Y; mild; "
            "penetrance near-complete but variable expressivity); "
            "AR (Ullrich CMD — biallelic null/severe missense; severe); "
            "De novo dominant: ~30% of severe Ullrich cases carry de novo dominant-negative COL6A1/A2/A3 variants; "
            "CK: normal or mildly elevated 1-5× ULN (unlike LAMA2 which is markedly elevated); "
            "Onset: congenital (Ullrich) or childhood (Bethlem); "
            "Cardiac: not involved; "
            "Respiratory: early failure in Ullrich (NIV by school age); "
            "Intelligence: preserved"
        ),
        "key_features": [
            "PROXIMAL WEAKNESS + DISTAL JOINT HYPERLAXITY paradox — pathognomonic of Ullrich CMD (proximal contractures + distal laxity simultaneously)",
            "MULTIPLE CONTRACTURES — elbow flexion + hip flexion + knee flexion + ankle equinus in Ullrich",
            "FOLLICULAR HYPERKERATOSIS + KELOID SCARRING — skin signs pathognomonic of Ullrich CMD; check skin carefully",
            "Early respiratory failure — NIV often school age in Ullrich; FVC trajectory mandatory",
            "Collagen VI IHC reduction/absence on biopsy — basement membrane zone reduced or absent",
            "CK near-normal or mildly elevated (1-5×) — key DDx from LAMA2 (CK 5-30×)",
            "Bethlem Myopathy (AD): milder, proximal weakness + contractures, slow progression, some ambulant lifelong",
            "No cardiac involvement — key DDx from some other CMD subtypes",
        ],
        "treatment": (
            "Respiratory: 6-monthly FVC; NIV when FVC <60% or nocturnal desaturation; Cough-Assist device; "
            "Physiotherapy: serial casting and stretching for contractures; orthoses; "
            "Cyclosporin A: reduces mitochondrial apoptosis in COL6 myopathy (pilot trials showing benefit — autophagy induction); "
            "Gene therapy: ex vivo gene correction research ongoing; "
            "Orthopaedic: surgery for scoliosis; Achilles tendon release; "
            "Nutrition: gastrostomy if swallowing impaired; "
            "Bethlem: physiotherapy only usually; monitor respiratory function in 5th-6th decade"
        ),
        "contraindications": (
            "Suxamethonium — depolarising blocker risk; "
            "Vigorous passive stretching beyond tissue tolerance; "
            "Cyclosporin A: monitor renal function + blood pressure; "
            "Corticosteroids: NOT indicated (differs from Duchenne muscular dystrophy)"
        ),
        "critical_pearls": [
            "Skin biopsy (elbow/forearm) for collagen VI IHC — faster and safer than muscle biopsy; COL6 reduced in basement membrane zone",
            "De novo dominant-negative: ~30% of Ullrich CMD cases have de novo dominant-negative variant (not biallelic AR) — parents should still be tested",
            "Three COL6 genes (COL6A1/A2 on 21q22.3, COL6A3 on 2q37.3) — always sequence all three; variants in any cause same spectrum",
            "Cyclosporin A pilot data: improves muscle autophagy and force; may slow disease — most studied CMD-specific treatment",
            "Bethlem Myopathy is slowly progressive AD disease; contractures (elbow/finger) + proximal weakness; some patients remain ambulant lifelong; respiratory involvement rare until 5th-6th decade",
        ],
    },

    # -- COL6A2 — Collagen VI α2, Ullrich CMD / Bethlem Myopathy ----------------------------
    {
        "gene": "COL6A2",
        "alt_name": (
            "COL6A2 (COL6A2-1019aa-21q22.3 / AR-AD — Ullrich-CMD-Bethlem-Myopathy-Collagen-VI-β-Chain — "
            "SAME-ULLRICH-BETHLEM-SPECTRUM-COL6A1-COL6A3 — "
            "21q22.3-Same-Chromosome-COL6A1-Cosegregation-Critical — "
            "Collagen-VI-IHC-Absent-Reduced-Respiratory-Failure-Early)"
        ),
        "protein": (
            "COL6A2 -- 21q22.3 AR/AD -- COL6A2-1019aa -- "
            "Collagen-VI-Alpha-2-Chain-Beta-Chain-140kDa-Triple-Helix-Heterotrimeric-Complex -- "
            "Ullrich-CMD-UCMD-OMIM-254090-Severe-AR -- "
            "Bethlem-Myopathy-BM-OMIM-158810-Mild-AD -- "
            "21q22.3-SAME-LOCUS-COL6A1-Chromosome-21-Contiguous-Gene-Region -- "
            "COSEGREGATION-MANDATORY-When-One-21q22.3-Variant-Found-Screen-Both-COL6A1-COL6A2 -- "
            "PROXIMAL-WEAKNESS-DISTAL-HYPERLAXITY-PATHOGNOMONIC-Ullrich -- "
            "FOLLICULAR-HYPERKERATOSIS-KELOID-Skin-PATHOGNOMONIC-Ullrich -- "
            "Collagen-VI-IHC-Absent-Reduced-Muscle-Skin-Biopsy -- "
            "OMIM-Gene-COL6A2-120240-Disease-Ullrich-254090-Bethlem-158810"
        ),
        "locus": "21q22.3",
        "protein_size": "1019 aa / 140 kDa",
        "inheritance": (
            "AR (Ullrich CMD — biallelic null; severe); "
            "AD (Bethlem Myopathy — dominant Gly-X-Y triple helix substitutions; mild); "
            "De novo dominant: ~30% Ullrich; "
            "Same spectrum as COL6A1/COL6A3 — no clinical phenotypic difference between the three genes; "
            "CK: near-normal to mildly elevated 1-5× ULN; "
            "21q22.3 locus: COL6A2 adjacent to COL6A1 on same chromosome; "
            "Carrier testing: 21q22.3 region can have compound heterozygosity within COL6A1/COL6A2 simultaneously; "
            "Cardiac: not involved; Intellectual disability: absent"
        ),
        "key_features": [
            "Same Ullrich CMD / Bethlem Myopathy spectrum as COL6A1 — no distinguishing clinical features",
            "PROXIMAL WEAKNESS + DISTAL JOINT HYPERLAXITY paradox — same pathognomonic pattern",
            "FOLLICULAR HYPERKERATOSIS + KELOID SCARRING — same Ullrich skin signs",
            "Multiple contractures — elbow, hip, knee, Achilles",
            "21q22.3 — same chromosome as COL6A1; both genes must be screened together (cosegregation testing)",
            "Collagen VI IHC reduction/absence on muscle or skin biopsy",
            "CK near-normal or mildly elevated (1-5×)",
            "Respiratory failure in Ullrich — NIV monitoring from 1st decade",
        ],
        "treatment": (
            "Identical to COL6A1: respiratory monitoring + NIV; physiotherapy; "
            "Cyclosporin A (pilot trials for COL6 myopathy); "
            "Serial casting contractures; orthoses; "
            "Gastrostomy if swallowing impaired; "
            "No corticosteroids; "
            "Gene therapy research ongoing for collagen VI myopathy"
        ),
        "contraindications": (
            "Suxamethonium; "
            "Vigorous passive stretching; "
            "Corticosteroids not indicated; "
            "Cyclosporin: monitor renal/BP"
        ),
        "critical_pearls": [
            "COL6A2 on 21q22.3: adjacent to COL6A1 — if one 21q22.3 heterozygous variant found, always screen COL6A2 and COL6A1 together for compound heterozygosity",
            "Phenotype identical to COL6A1/COL6A3 Ullrich CMD — gene sequencing required to distinguish which COL6 gene is causative",
            "~30% de novo dominant-negative — do not assume biallelic recessive until parents are tested",
            "Skin biopsy collagen VI IHC equivalent to muscle biopsy for diagnosis — preferred outpatient option",
            "Bethlem Myopathy due to COL6A2: finger contractures (Flexor Digitorum Superficialis) + proximal weakness; may be mistaken for inflammatory myopathy — CK mildly elevated, no inflammation on biopsy",
        ],
    },

    # -- COL6A3 — Collagen VI α3, Ullrich CMD / Bethlem Myopathy ----------------------------
    {
        "gene": "COL6A3",
        "alt_name": (
            "COL6A3 (COL6A3-3177aa-2q37.3 / AR-AD — Ullrich-CMD-Bethlem-Myopathy-Collagen-VI-α3-Largest-Subunit — "
            "2q37.3-DIFFERENT-CHROMOSOME-COL6A1-COL6A2 — "
            "Unique-N-Terminal-Domain-COL6A3-Dominant-Negative-Most-Common-Bethlem — "
            "Keloid-Follicular-Hyperkeratosis-Skin-PATHOGNOMONIC-Ullrich)"
        ),
        "protein": (
            "COL6A3 -- 2q37.3 AR/AD -- COL6A3-3177aa -- "
            "Collagen-VI-Alpha-3-Chain-260kDa-Largest-Subunit-Unique-N-Terminal-Domain-Triple-Helix -- "
            "Ullrich-CMD-UCMD-OMIM-254090-Severe-AR -- "
            "Bethlem-Myopathy-BM-OMIM-158810-Mild-AD -- "
            "MOST-COMMON-BETHLEM-MYOPATHY-GENE-Dominant-Glycine-Triple-Helix -- "
            "2q37.3-DIFFERENT-CHROMOSOME-from-COL6A1-COL6A2-21q22.3-Critical-Genetics-Distinction -- "
            "Unique-N-Terminal-FNIII-Fibronectin-III-Domains-Not-in-COL6A1-COL6A2 -- "
            "KELOID-FOLLICULAR-HYPERKERATOSIS-SKIN-PATHOGNOMONIC-Ullrich -- "
            "Collagen-VI-IHC-Absent-Reduced -- "
            "OMIM-Gene-COL6A3-120250-Disease-Ullrich-254090-Bethlem-158810"
        ),
        "locus": "2q37.3",
        "protein_size": "3177 aa / 260 kDa",
        "inheritance": (
            "AD (Bethlem Myopathy — most common COL6 gene in Bethlem; dominant Gly-X-Y substitutions in triple helix); "
            "AR (Ullrich CMD — biallelic null/severe missense); "
            "De novo dominant: ~30% Ullrich; "
            "COL6A3 most commonly mutated gene in Bethlem Myopathy cohorts; "
            "2q37.3 — different chromosome from COL6A1/A2 (21q22.3); "
            "Largest COL6 subunit (3177 aa) with unique N-terminal FNIII domains; "
            "CK: near-normal to mildly elevated 1-5× ULN; "
            "Cardiac: not involved; Intellectual disability: absent"
        ),
        "key_features": [
            "Same Ullrich CMD / Bethlem Myopathy spectrum — most commonly mutated gene in Bethlem Myopathy cohorts",
            "2q37.3 — DIFFERENT chromosome from COL6A1/COL6A2 (21q22.3): distinguish in genetics testing",
            "Largest COL6 subunit (3177 aa, 260 kDa) with unique N-terminal fibronectin III domains",
            "FOLLICULAR HYPERKERATOSIS + KELOID SCARRING — Ullrich skin signs (same as COL6A1/A2)",
            "PROXIMAL WEAKNESS + DISTAL HYPERLAXITY paradox — same Ullrich pathognomonic pattern",
            "COL6 IHC reduction/absence on biopsy",
            "Respiratory failure — Ullrich: NIV often school age",
            "Bethlem: finger and elbow contractures + proximal weakness, slow progression",
        ],
        "treatment": (
            "Identical to COL6A1/A2: respiratory monitoring + NIV; physiotherapy; "
            "Cyclosporin A pilot trials; "
            "Serial casting; orthoses; "
            "Gene therapy research; "
            "No corticosteroids"
        ),
        "contraindications": (
            "Suxamethonium; vigorous stretching; corticosteroids not indicated; "
            "Cyclosporin: monitor renal function + BP"
        ),
        "critical_pearls": [
            "COL6A3 is on 2q37.3 — separate chromosome from COL6A1/A2 (21q22.3); compound heterozygosity across genes cannot occur for COL6A3 + COL6A1/A2 (different chromosomes)",
            "Most commonly mutated COL6 gene in Bethlem Myopathy series — sequence COL6A3 first in suspected Bethlem with AD inheritance",
            "Unique N-terminal FNIII domains in COL6A3 — not present in COL6A1/A2; affects collagen VI secretion and assembly",
            "Skin biopsy collagen VI IHC: reduced basement membrane zone staining; safer than muscle biopsy",
            "Bethlem myopathy: often presents in 3rd-5th decade; finger flexion contractures (FDS test) + proximal weakness + elevated CK 1-4×; often misdiagnosed as inflammatory myopathy",
        ],
    },

    # -- FKTN — Fukutin, Fukuyama CMD -------------------------------------------------------
    {
        "gene": "FKTN",
        "alt_name": (
            "FKTN (FKTN-461aa-9q31.2 / AR — Fukuyama-CMD-FCMD-Most-Common-CMD-Japan — "
            "INTELLECTUAL-DISABILITY-MANDATORY — "
            "COBBLESTONE-LISSENCEPHALY-Type-II-PATHOGNOMONIC — "
            "DCM-Mandatory-Adolescence-Retrotransposon-Japan-Founder)"
        ),
        "protein": (
            "FKTN -- 9q31.2 AR -- FKTN-461aa -- "
            "Fukutin-54kDa-Golgi-Glycosyltransferase-Ribitol-5-Phosphate-Synthesis-Step -- "
            "Fukuyama-CMD-FCMD-OMIM-253800 -- "
            "INTELLECTUAL-DISABILITY-MANDATORY-ALL-FCMD-PATIENTS -- "
            "COBBLESTONE-LISSENCEPHALY-Type-II-Polymicrogyria-PATHOGNOMONIC-MRI -- "
            "RETINAL-DYSPLASIA-Myopia-Retinal-Hypoplasia -- "
            "DCM-Dilated-Cardiomyopathy-Adolescence-Adults-MANDATORY-Surveillance -- "
            "c3036plus-Retrotransposon-IVS-Insertion-85pct-Japanese-Alleles-Founder -- "
            "CK-10-50x-ULN -- "
            "Reduced-Alpha-Dystroglycan-Glycosylation-IHC -- "
            "OMIM-Gene-FKTN-607440-Disease-FCMD-253800"
        ),
        "locus": "9q31.2",
        "protein_size": "461 aa / 54 kDa",
        "inheritance": (
            "AR (biallelic); "
            "Japan prevalence: 1 in 6,500-10,000 births (2nd most common AR neurological disorder in Japan after DMD); "
            "Non-Japan: rare (isolated cases); "
            "Founder: c.3036+IVS(retrotransposon ~3kb insertion) — 85% of Japanese FCMD alleles; "
            "CK 10-50× ULN (markedly elevated); "
            "Onset: prenatal/neonatal — floppy infant at birth; "
            "Cobblestone lissencephaly (Type II): agyria/pachygyria/polymicrogyria; "
            "DCM: adolescence/adulthood — mandatory surveillance; "
            "Intellectual disability: ALL patients (mild-moderate usual; severe in complete null); "
            "Retinal dysplasia: myopia, retinal hypoplasia; "
            "Survival: childhood to adulthood in founder genotype; earlier death in null/null"
        ),
        "key_features": [
            "INTELLECTUAL DISABILITY MANDATORY — all Fukuyama CMD patients; severity correlates with genotype (severe null > founder heterozygote)",
            "COBBLESTONE LISSENCEPHALY (Type II) — agyria/pachygyria/polymicrogyria on MRI PATHOGNOMONIC of Fukuyama CMD",
            "RETROTRANSPOSON FOUNDER — c.3036+IVS(~3kb) 85% of Japanese alleles; readily detected by PCR/Southern blot",
            "DCM — dilated cardiomyopathy in adolescence/adulthood; mandatory cardiac surveillance; SCD risk",
            "Retinal dysplasia — myopia + retinal hypoplasia; ophthalmology mandatory",
            "CK 10-50× ULN — markedly elevated",
            "Reduced α-dystroglycan glycosylation on IHC (IIH6 antibody staining reduced)",
            "Prenatal/neonatal onset — floppy infant; contractures at birth",
        ],
        "treatment": (
            "DCM: echo + Holter annually from adolescence; ACEi/ARB for DCM; ICD if severe; "
            "Respiratory: NIV monitoring; FVC trajectory; "
            "Physiotherapy: tone management, contracture prevention; "
            "Ophthalmology: myopia correction; retinal monitoring; "
            "Seizure management: AEDs; "
            "Nutrition: gastrostomy if swallowing impaired; "
            "Education: special education + developmental support; "
            "Gene therapy: AAV-based research ongoing; "
            "Riboflavin/CoQ10: empiric supplementation (limited evidence)"
        ),
        "contraindications": (
            "Suxamethonium — rhabdomyolysis/hyperkalaemia risk; "
            "Valproic acid — mitochondrial function concern with dystroglycanopathies (use with caution); "
            "No immunosuppression (not inflammatory); "
            "High-intensity exercise — rhabdomyolysis risk"
        ),
        "critical_pearls": [
            "PCR for retrotransposon insertion: rapid Japan FCMD screening — 85% sensitivity for Japanese FCMD alleles; full FKTN sequencing for non-Japanese cases",
            "DCM is NOT present at birth — develops in adolescence; cardiac surveillance must begin by age 10 and annually thereafter; DCM can be severe and is a major cause of death in FCMD",
            "Cobblestone lissencephaly Type II: grossly abnormal MRI (looks catastrophic) but cognition in FCMD is mild-moderate ID, not no-response; do not assume end-of-life based on MRI alone",
            "All four dystroglycanopathy genes (FKTN/FKRP/POMT1/POMT2/POMGNT1/etc.) show reduced IIH6 antibody α-dystroglycan staining — IHC shared finding but gene panels distinguish",
            "Non-Japan FCMD: rare but exists; sequencing required (no founder allele screen); associated with severe phenotype and non-founder null alleles",
        ],
    },

    # -- POMT1 — Protein O-Mannosyltransferase 1, Walker-Warburg Syndrome type 1 -------------
    {
        "gene": "POMT1",
        "alt_name": (
            "POMT1 (POMT1-747aa-9q34.13 / AR — Walker-Warburg-Syndrome-WWS-Type-1-MOST-SEVERE-CMD — "
            "COBBLESTONE-LISSENCEPHALY-BRAINSTEM-HYPOPLASIA-CEREBELLAR-HYPOPLASIA-OCULAR-DEFECTS-PATHOGNOMONIC — "
            "LETHAL-Usually-1st-Year — "
            "Absent-Alpha-Dystroglycan-Glycosylation)"
        ),
        "protein": (
            "POMT1 -- 9q34.13 AR -- POMT1-747aa -- "
            "Protein-O-Mannosyltransferase-1-83kDa-ER-Transmembrane-Enzyme -- "
            "Walker-Warburg-Syndrome-WWS-OMIM-236670 -- "
            "MOST-SEVERE-Congenital-Muscular-Dystrophy-Glycosylation-Disorder -- "
            "COBBLESTONE-LISSENCEPHALY-Type-II-PATHOGNOMONIC -- "
            "BRAINSTEM-HYPOPLASIA-PONTINE-Hypoplasia-PATHOGNOMONIC-WWS -- "
            "CEREBELLAR-HYPOPLASIA-PATHOGNOMONIC-WWS -- "
            "OCULAR-DEFECTS-Anterior-Chamber-Dysgenesis-Coloboma-Cataract-PATHOGNOMONIC -- "
            "USUALLY-LETHAL-1st-Year-Brainstem-Failure -- "
            "POMT1-POMT2-Obligate-Heterodimer-BOTH-Required-O-Mannosyltransferase-Activity -- "
            "Absent-Glycosylated-Alpha-Dystroglycan-IIH6-IHC -- "
            "CK-Markedly-Elevated->50x-ULN -- "
            "OMIM-Gene-POMT1-607423-Disease-WWS-236670"
        ),
        "locus": "9q34.13",
        "protein_size": "747 aa / 83 kDa",
        "inheritance": (
            "AR (biallelic — usually compound heterozygous or homozygous null); "
            "MOST SEVERE: complete null biallelic → Walker-Warburg Syndrome (usually lethal 1st year); "
            "Hypomorphic alleles → MEB-like or LGMD-R11 (less severe, rare); "
            "CK >50× ULN (markedly elevated); "
            "Onset: prenatal/neonatal — congenital hypotonia, seizures; "
            "Brain MRI: cobblestone lissencephaly + pontine hypoplasia + cerebellar hypoplasia (Z-shaped brainstem on sagittal); "
            "Ocular: anterior chamber dysgenesis (Peters anomaly), coloboma, cataract, microphthalmia; "
            "POMT1+POMT2 form obligate heterodimer — both genes required for enzymatic function; "
            "Prenatal diagnosis: possible via sequencing if prior proband"
        ),
        "key_features": [
            "WALKER-WARBURG SYNDROME (WWS) — most severe congenital muscular dystrophy phenotype; usually lethal within 1st year",
            "COBBLESTONE LISSENCEPHALY TYPE II — agyria/pachygyria on MRI PATHOGNOMONIC; grossly malformed cortex",
            "Z-SHAPED BRAINSTEM on sagittal MRI — pontine hypoplasia + cerebellar hypoplasia pathognomonic combination for WWS",
            "OCULAR DEFECTS — anterior chamber dysgenesis (Peters anomaly), coloboma, cataract, microphthalmia PATHOGNOMONIC WWS",
            "POMT1+POMT2 obligate heterodimer — BOTH genes required for O-mannosyltransferase activity; panel must screen both",
            "Absent IIH6 α-dystroglycan IHC staining — shared dystroglycanopathy marker",
            "CK >50× ULN — severely elevated in neonatal period",
            "Usually lethal 1st year — brainstem failure; respiratory support typically brief; palliative discussions mandatory",
        ],
        "treatment": (
            "Supportive/palliative in WWS: "
            "Respiratory support: NICU intensive support; goals-of-care discussion early with family; "
            "Seizure management: AEDs; "
            "Ophthalmology: ocular pressure for glaucoma; lens treatment; "
            "Gastrostomy: for nutrition support if goals-of-care allow; "
            "Genetic counselling: family recurrence risk 25%; prenatal diagnosis available; "
            "No disease-modifying therapy approved for WWS; "
            "Ribose-5-phosphate supplementation: experimental; "
            "POMT1-specific: gene therapy preclinical research"
        ),
        "contraindications": (
            "Aggressive intervention without goals-of-care discussion (WWS usually lethal 1st year); "
            "Suxamethonium; "
            "Immunosuppression (not inflammatory); "
            "Valproic acid: use with caution; levetiracetam preferred"
        ),
        "critical_pearls": [
            "Z-shaped brainstem on MRI sagittal view (pontine hypoplasia + cerebellar hypoplasia) + cobblestone cortex + ocular defects = WWS — phenotype alone distinguishes from other CMD subtypes without waiting for genetics",
            "POMT1+POMT2 panel: always sequence BOTH — obligate heterodimer; ~50% of WWS cases have POMT1 pathogenic variants, ~20% POMT2; remaining POMGNT1/FKTN/FKRP/LARGE1 etc.",
            "Hypomorphic POMT1 alleles: rare patients with LGMD-R11 phenotype or MEB-like — same gene, less severe variant leads to residual enzyme activity",
            "α-dystroglycan IIH6 IHC: reduced/absent in ALL dystroglycanopathies (FKTN/FKRP/POMT1/POMT2/POMGNT1/LARGE1) — shared finding; gene panel required to distinguish",
            "Prenatal diagnosis: with prior affected sibling, CVS/amniocentesis gene sequencing now routine; fetal MRI shows lissencephaly by 20-22 weeks in WWS",
        ],
    },

    # -- POMT2 — Protein O-Mannosyltransferase 2, Walker-Warburg Syndrome type 2 -------------
    {
        "gene": "POMT2",
        "alt_name": (
            "POMT2 (POMT2-750aa-14q24.3 / AR — Walker-Warburg-Syndrome-WWS-Type-2-IDENTICAL-POMT1 — "
            "POMT1-POMT2-OBLIGATE-HETERODIMER-BOTH-Required-O-Mannosyltransferase-Activity — "
            "Hypomorphic-Alleles-MEB-Like-Less-Severe — "
            "Absent-Alpha-Dystroglycan-Glycosylation)"
        ),
        "protein": (
            "POMT2 -- 14q24.3 AR -- POMT2-750aa -- "
            "Protein-O-Mannosyltransferase-2-83kDa-ER-Transmembrane-Enzyme -- "
            "Walker-Warburg-Syndrome-WWS-OMIM-236670-Identical-POMT1 -- "
            "POMT1-POMT2-Obligate-Heterodimer-Required-O-Mannosylation-Alpha-Dystroglycan -- "
            "Cobblestone-Lissencephaly-Brainstem-Hypoplasia-Cerebellar-Hypoplasia-Identical-POMT1 -- "
            "Hypomorphic-Alleles-MEB-Like-LGMD-R14 -- "
            "Absent-IIH6-Alpha-Dystroglycan-Staining -- "
            "14q24.3-Different-Chromosome-POMT1-9q34.13 -- "
            "CK->50x-ULN-WWS -- "
            "OMIM-Gene-POMT2-607439-Disease-WWS-236670-LGMD-R14-607439"
        ),
        "locus": "14q24.3",
        "protein_size": "750 aa / 83 kDa",
        "inheritance": (
            "AR (biallelic — compound heterozygous or homozygous null); "
            "MOST SEVERE: biallelic null → WWS (identical phenotype to POMT1-WWS; only genetics distinguishes); "
            "Hypomorphic alleles → MEB-like or LGMD-R14 (rare; residual O-mannosyltransferase activity); "
            "14q24.3 — different chromosome from POMT1 (9q34.13); "
            "POMT1+POMT2 form obligate heterodimer — catalytic activity requires both proteins simultaneously; "
            "CK >50× ULN in WWS; "
            "Prenatal onset; usually lethal 1st year in WWS; "
            "Hypomorphic: survival into adulthood possible"
        ),
        "key_features": [
            "IDENTICAL WWS PHENOTYPE to POMT1 — cobblestone lissencephaly + brainstem/cerebellar hypoplasia + ocular defects; only genetics distinguishes POMT1 from POMT2",
            "POMT1+POMT2 OBLIGATE HETERODIMER — both proteins required for O-mannosyltransferase enzymatic activity; neither alone is sufficient",
            "Z-shaped brainstem on sagittal MRI — same pathognomonic WWS sign as POMT1",
            "Ocular defects — anterior chamber dysgenesis, coloboma, microphthalmia (same as POMT1)",
            "Hypomorphic alleles → MEB-like (less severe): pachygyria + cerebellar hypoplasia + eye disease + ID; survival into adulthood",
            "LGMD-R14: rare POMT2 hypomorphic → proximal limb girdle weakness without major CNS involvement",
            "Absent IIH6 α-dystroglycan IHC staining (shared dystroglycanopathy marker)",
            "14q24.3 — different chromosome from POMT1 (9q34.13); gene panels needed for both",
        ],
        "treatment": (
            "Same as POMT1/WWS: supportive/palliative for WWS; "
            "Goals-of-care discussion early; "
            "AEDs for seizures; ophthalmology; gastrostomy; "
            "Hypomorphic/MEB-like: more active management including respiratory support, physiotherapy; "
            "Genetic counselling: 25% recurrence; prenatal diagnosis; "
            "No disease-modifying therapy"
        ),
        "contraindications": (
            "Aggressive intervention without goals-of-care discussion (WWS severity); "
            "Suxamethonium; immunosuppression; "
            "VPA: use LEV/other AED preferred"
        ),
        "critical_pearls": [
            "POMT1-WWS vs POMT2-WWS: clinically IDENTICAL — only gene sequencing differentiates; always panel both in WWS workup",
            "POMT1+POMT2 heterodimer: understanding this explains why variants in EITHER gene cause same disease (enzyme made of both subunits; loss of either = loss of activity)",
            "~20% of WWS cases due to POMT2 (less common than POMT1 ~50%); remaining are POMGNT1, FKTN, FKRP, LARGE1, etc.",
            "POMT2 hypomorphic alleles: MEB-like survivors into adolescence; LGMD-R14 (proximal limb weakness, mild or no CNS) — always ask for genotype-phenotype; residual enzyme activity determines severity",
            "α-dystroglycanopathy IHC: POMT2 same reduced IIH6 as all dystroglycanopathies — screen IHC first, then panel",
        ],
    },

    # -- POMGNT1 — POMGnT1, Muscle-Eye-Brain Disease ----------------------------------------
    {
        "gene": "POMGNT1",
        "alt_name": (
            "POMGNT1 (POMGNT1-660aa-1p34.1 / AR — Muscle-Eye-Brain-MEB-Disease-Less-Severe-Than-WWS — "
            "MYOPIA-GLAUCOMA-CEREBELLAR-HYPOPLASIA-COMBINATION-PATHOGNOMONIC — "
            "Finnish-Founder-pTyr688Cys-85pct-Finnish-Alleles — "
            "Pachygyria-Intellectual-Disability-Survival-Adulthood)"
        ),
        "protein": (
            "POMGNT1 -- 1p34.1 AR -- POMGNT1-660aa -- "
            "Protein-O-Linked-Mannose-Beta-1-2-N-Acetylglucosaminyltransferase-1-75kDa-Golgi -- "
            "Muscle-Eye-Brain-Disease-MEB-OMIM-253280 -- "
            "MYOPIA-GLAUCOMA-COMBINATION-PATHOGNOMONIC-MEB -- "
            "CEREBELLAR-HYPOPLASIA-PACHYGYRIA-MRI-PATHOGNOMONIC-MEB-Less-Severe-WWS -- "
            "INTELLECTUAL-DISABILITY-ALL-MEB-Patients -- "
            "Finnish-Founder-pTyr688Cys-c2063AT-85pct-Finnish-Alleles -- "
            "SURVIVAL-INTO-ADULTHOOD-Distinguishes-MEB-from-WWS-Lethal-1yr -- "
            "Reduced-IIH6-Alpha-Dystroglycan-IHC -- "
            "CK-10-100x-ULN -- "
            "OMIM-Gene-POMGNT1-606822-Disease-MEB-253280"
        ),
        "locus": "1p34.1",
        "protein_size": "660 aa / 75 kDa",
        "inheritance": (
            "AR (biallelic); "
            "Finnish founder: p.Tyr688Cys (c.2063A>T) — 85% of Finnish MEB alleles; "
            "Non-Finnish: compound heterozygous; "
            "CK 10-100× ULN; "
            "Onset: prenatal/neonatal — congenital hypotonia; "
            "Brain MRI: pachygyria (less severe than WWS agyria) + cerebellar hypoplasia; "
            "Eye: myopia + glaucoma + retinal hypoplasia (key MEB DDx from WWS: less severe eye involvement); "
            "Intellectual disability: ALL MEB patients (moderate to severe); "
            "Survival: into adolescence and adulthood (unlike WWS lethal 1yr); "
            "DCM: rare in MEB compared to FKTN"
        ),
        "key_features": [
            "MYOPIA + GLAUCOMA + CEREBELLAR HYPOPLASIA combination PATHOGNOMONIC of MEB disease",
            "PACHYGYRIA (not agyria) — brain MRI less severe than WWS; pachygyria + cerebellar hypoplasia characteristic MEB MRI pattern",
            "SURVIVAL INTO ADULTHOOD — key distinction from WWS (lethal 1st year); MEB patients can survive to adulthood",
            "INTELLECTUAL DISABILITY — all MEB patients; moderate-severe; developmental plateau",
            "FINNISH FOUNDER p.Tyr688Cys — 85% of Finnish MEB alleles; Finnish population high incidence",
            "Glaucoma — progressive; mandatory ophthalmology + IOP monitoring; treat to prevent blindness",
            "CK 10-100× ULN — markedly elevated",
            "Reduced IIH6 α-dystroglycan IHC staining (shared dystroglycanopathy marker)",
        ],
        "treatment": (
            "Ophthalmology: IOP monitoring + glaucoma treatment (drops/surgery); myopia correction; retinal surveillance; "
            "Physiotherapy: tone management, contracture prevention, mobility aids; "
            "Respiratory: NIV monitoring (less severe than LAMA2-CMD but monitor from school age); "
            "Seizure management: AEDs (common in MEB); "
            "Special education + developmental support (ID mandatory); "
            "Cardiac: echo in adolescence (DCM less common than FKTN but monitor); "
            "Gastrostomy: if swallowing impaired; "
            "Genetic counselling: 25% recurrence; Finnish founder PCR screening available"
        ),
        "contraindications": (
            "Suxamethonium; "
            "High IOP without treatment — glaucoma blindness risk; "
            "Immunosuppression (not inflammatory); "
            "No vigorous exercise beyond tolerance"
        ),
        "critical_pearls": [
            "MEB vs WWS DDx: MEB (pachygyria, survival, glaucoma, myopia) vs WWS (agyria/cobblestone, lethal 1yr, more severe eye defects); use MRI and survival as initial DDx before genetics",
            "Finnish founder p.Tyr688Cys: PCR-based founder screening in Finnish patients — rapid diagnosis before full sequencing; non-Finnish patients need full POMGNT1 sequencing",
            "Glaucoma management critical: IOP elevation → optic nerve damage → blindness; ophthalmology from diagnosis; prostaglandin analogues/beta-blockers; surgery if medical fails",
            "α-dystroglycanopathy shared IHC: POMGNT1 reduces IIH6 staining same as FKTN/POMT1/POMT2/FKRP/LARGE1; IHC diagnosis requires gene panel confirmation",
            "POMGNT1 phenotypic spectrum: severe (WWS-like) to mild (LGMD-like LGMD R15) — genotype predicts severity (null = severe; hypomorphic = mild); adult LGMD-R15 can be ambulant",
        ],
    },
]


def _make_patients(gene_data: dict, seed: int, n: int = 40) -> list:
    """Generate synthetic patient cohort for a CMD gene (deterministic, seed-fixed)."""
    rng = random.Random(seed)
    gene = gene_data["gene"]
    pts = []

    for i in range(n):
        # Gene-specific clinical parameters
        if gene == "LAMA2":
            age_onset = rng.uniform(0, 1)          # birth to 1 yr
            dx_delay = rng.uniform(1, 5)            # 1-5 yr
            ck = rng.uniform(1000, 8000)            # 5-30× ULN (ULN ~300 IU/L)
            alive = rng.random() > 0.20             # 80% alive (some early death)
            ambulant = rng.random() > 0.60          # 40% ambulant (severe) or 60% if partial
            treated = rng.random() > 0.20           # 80% get NIV/supportive care
            hospitalisations = rng.uniform(1.5, 4)  # respiratory hospitalisations/yr
        elif gene == "COL6A1":
            age_onset = rng.uniform(0, 3)
            dx_delay = rng.uniform(2, 8)
            ck = rng.uniform(50, 600)               # normal to mildly elevated
            alive = rng.random() > 0.10
            ambulant = rng.random() > 0.45          # 55% ambulant (mix Ullrich/Bethlem)
            treated = rng.random() > 0.30
            hospitalisations = rng.uniform(0.5, 2.5)
        elif gene == "COL6A2":
            age_onset = rng.uniform(0, 3)
            dx_delay = rng.uniform(2, 8)
            ck = rng.uniform(50, 600)
            alive = rng.random() > 0.10
            ambulant = rng.random() > 0.45
            treated = rng.random() > 0.30
            hospitalisations = rng.uniform(0.5, 2.5)
        elif gene == "COL6A3":
            age_onset = rng.uniform(0, 5)           # Bethlem can be childhood-adult
            dx_delay = rng.uniform(3, 12)           # Bethlem often late-diagnosed
            ck = rng.uniform(50, 800)
            alive = rng.random() > 0.08
            ambulant = rng.random() > 0.35          # 65% ambulant (Bethlem-predominant)
            treated = rng.random() > 0.35
            hospitalisations = rng.uniform(0.3, 2)
        elif gene == "FKTN":
            age_onset = rng.uniform(0, 0.5)         # prenatal-neonatal
            dx_delay = rng.uniform(1, 5)
            ck = rng.uniform(2000, 15000)           # 10-50× ULN
            alive = rng.random() > 0.25
            ambulant = rng.random() > 0.80          # 20% ambulant (severe CMD)
            treated = rng.random() > 0.25
            hospitalisations = rng.uniform(1, 4)
        elif gene == "POMT1":
            age_onset = rng.uniform(0, 0.1)         # prenatal-neonatal
            dx_delay = rng.uniform(0.1, 1)          # diagnosed quickly (severe)
            ck = rng.uniform(5000, 30000)           # >50× ULN
            alive = rng.random() > 0.70             # 30% alive at cohort snapshot
            ambulant = False
            treated = rng.random() > 0.60          # 40% get active treatment vs palliative
            hospitalisations = rng.uniform(5, 15)  # high NICU/PICU
        elif gene == "POMT2":
            age_onset = rng.uniform(0, 0.1)
            dx_delay = rng.uniform(0.1, 1)
            ck = rng.uniform(4000, 25000)
            alive = rng.random() > 0.65             # 35% alive (some hypomorphic survivors)
            ambulant = rng.random() > 0.95         # 5% ambulant (hypomorphic alleles)
            treated = rng.random() > 0.55
            hospitalisations = rng.uniform(4, 12)
        elif gene == "POMGNT1":
            age_onset = rng.uniform(0, 0.5)
            dx_delay = rng.uniform(0.5, 4)
            ck = rng.uniform(1500, 20000)           # 10-100× ULN
            alive = rng.random() > 0.30             # 70% alive (MEB better than WWS)
            ambulant = rng.random() > 0.85         # 15% ambulant
            treated = rng.random() > 0.20
            hospitalisations = rng.uniform(1, 5)
        else:
            age_onset = rng.uniform(0, 5)
            dx_delay = rng.uniform(1, 8)
            ck = rng.uniform(300, 5000)
            alive = rng.random() > 0.20
            ambulant = rng.random() > 0.60
            treated = rng.random() > 0.30
            hospitalisations = rng.uniform(1, 4)

        pts.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "seed": seed,
            "age_onset": round(age_onset, 2),
            "dx_delay_yr": round(dx_delay, 1),
            "ck_iu_l": int(ck),
            "alive": alive,
            "ambulant_at_10yr": ambulant,
            "treated": treated,
            "hospitalisations_per_year": round(hospitalisations, 1),
        })
    return pts


def overview():
    """320-patient aggregate overview across all 8 CMD genes."""
    all_patients = []
    for i, gd in enumerate(CMD_GENES):
        all_patients += _make_patients(gd, SEED_BASE + i)

    n = len(all_patients)
    alive = sum(1 for p in all_patients if p["alive"])
    treated = sum(1 for p in all_patients if p["treated"])
    ambulant = sum(1 for p in all_patients if p["ambulant_at_10yr"])
    mean_onset = round(sum(p["age_onset"] for p in all_patients) / n, 2)
    mean_dx_delay = round(sum(p["dx_delay_yr"] for p in all_patients) / n, 1)
    mean_ck = round(sum(p["ck_iu_l"] for p in all_patients) / n, 0)

    gene_summaries = []
    for i, gd in enumerate(CMD_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        gene_summaries.append({
            "gene": gd["gene"],
            "n_patients": len(pts),
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance": gd["inheritance"].split(";")[0].strip(),
            "mean_onset_yr": round(sum(p["age_onset"] for p in pts) / len(pts), 2),
            "mean_ck": round(sum(p["ck_iu_l"] for p in pts) / len(pts), 0),
            "alive_pct": round(sum(1 for p in pts if p["alive"]) / len(pts) * 100, 1),
            "ambulant_10yr_pct": round(sum(1 for p in pts if p["ambulant_at_10yr"]) / len(pts) * 100, 1),
        })

    return {
        "atlas": "Hereditary-CMD-Atlas",
        "subtitle": (
            "Complete 8-Gene Congenital Muscular Dystrophy (CMD) Spectrum "
            "(LAMA2 · COL6A1 · COL6A2 · COL6A3 · FKTN · POMT1 · POMT2 · POMGNT1)"
        ),
        "genes_covered": [gd["gene"] for gd in CMD_GENES],
        "n_genes": 8,
        "n_patients": n,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "alive_pct": round(alive / n * 100, 1),
        "treated_pct": round(treated / n * 100, 1),
        "ambulant_10yr_pct": round(ambulant / n * 100, 1),
        "mean_age_onset_yr": mean_onset,
        "mean_dx_delay_yr": mean_dx_delay,
        "mean_ck_iu_l": int(mean_ck),
        "key_spectrum_facts": [
            "LAMA2: ABSENT MEROSIN IHC PATHOGNOMONIC; WHITE MATTER T2 LEUKODYSTROPHY (non-progressive); demyelinating neuropathy; most common CMD globally; CK 5-30×",
            "COL6A1: PROXIMAL WEAKNESS + DISTAL HYPERLAXITY paradox PATHOGNOMONIC Ullrich; FOLLICULAR HYPERKERATOSIS+KELOID skin PATHOGNOMONIC; early respiratory failure; near-normal CK; Bethlem=mild AD",
            "COL6A2: Same Ullrich/Bethlem spectrum as COL6A1; β-chain; 21q22.3 same chromosome as COL6A1 — cosegregation testing critical",
            "COL6A3: Same Ullrich/Bethlem spectrum; MOST COMMON COL6 gene in Bethlem; largest subunit 3177aa; 2q37.3 different chromosome COL6A1/A2",
            "FKTN: INTELLECTUAL DISABILITY MANDATORY; COBBLESTONE LISSENCEPHALY (Type II) PATHOGNOMONIC; DCM adolescence mandatory; c.3036+IVS retrotransposon Japan founder 85%",
            "POMT1: WALKER-WARBURG SYNDROME — MOST SEVERE CMD; cobblestone lissencephaly + Z-brainstem + ocular defects; LETHAL usually 1st year; POMT1+POMT2 obligate heterodimer",
            "POMT2: IDENTICAL WWS phenotype to POMT1 — only genetics distinguishes; obligate heterodimer with POMT1; hypomorphic alleles → MEB-like/LGMD-R14 (survival adulthood)",
            "POMGNT1: MUSCLE-EYE-BRAIN (MEB) — MYOPIA+GLAUCOMA+CEREBELLAR HYPOPLASIA PATHOGNOMONIC; Finnish founder p.Tyr688Cys 85%; pachygyria; survival adulthood unlike WWS",
        ],
        "intellectual_disability_genes": ["FKTN", "POMT1", "POMT2", "POMGNT1"],
        "no_intellectual_disability_genes": ["LAMA2", "COL6A1", "COL6A2", "COL6A3"],
        "cardiac_mandate_genes": ["FKTN"],
        "lethal_usually_1yr_genes": ["POMT1", "POMT2"],
        "ambulant_possible_genes": ["LAMA2", "COL6A1", "COL6A2", "COL6A3"],
        "dystroglycanopathy_genes": ["FKTN", "POMT1", "POMT2", "POMGNT1"],
        "collagen_vi_genes": ["COL6A1", "COL6A2", "COL6A3"],
        "critical_ddx": {
            "LAMA2_vs_COL6_CK": "LAMA2: CK 5-30× markedly elevated; COL6 CMD: CK near-normal to mildly elevated 1-5× — CK is key first discriminator",
            "WWS_vs_MEB_severity": "WWS (POMT1/POMT2): agyria/cobblestone + usually lethal 1yr + severe eye defects; MEB (POMGNT1): pachygyria + survival adulthood + myopia/glaucoma",
            "FKTN_vs_POMT1_ID": "Both cause ID + cobblestone lissencephaly — FKTN: Japan founder PCR; POMT1/2: more severe WWS with Z-brainstem; gene panel required",
            "COL6A1_vs_COL6A2_21q223": "Both on 21q22.3 — always sequence BOTH when one is found; compound heterozygosity possible",
            "Ullrich_vs_Bethlem_severity": "Ullrich (AR/de-novo-dominant): severe CMD non-ambulant; Bethlem (AD): mild, often ambulant; same COL6 gene, different allele severity",
            "LAMA2_merosin_vs_dystroglycanopathy": "LAMA2: absent merosin IHC specific; FKTN/POMT1/POMT2/POMGNT1: reduced IIH6 α-dystroglycan IHC; different IHC markers distinguish",
            "MEB_glaucoma_mandatory": "POMGNT1: GLAUCOMA active treatment mandatory — IOP elevation → optic nerve damage → blindness; not required in other CMD subtypes",
        },
        "gene_summaries": gene_summaries,
    }


def breakdown():
    """Per-gene clinical breakdown with key metrics."""
    result = {}
    for i, gd in enumerate(CMD_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        result[gd["gene"]] = {
            "gene": gd["gene"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "n_patients": len(pts),
            "alive_pct": round(sum(1 for p in pts if p["alive"]) / len(pts) * 100, 1),
            "treated_pct": round(sum(1 for p in pts if p["treated"]) / len(pts) * 100, 1),
            "ambulant_10yr_pct": round(sum(1 for p in pts if p["ambulant_at_10yr"]) / len(pts) * 100, 1),
            "mean_age_onset": round(sum(p["age_onset"] for p in pts) / len(pts), 2),
            "mean_dx_delay_yr": round(sum(p["dx_delay_yr"] for p in pts) / len(pts), 1),
            "mean_ck_iu_l": int(sum(p["ck_iu_l"] for p in pts) / len(pts)),
            "mean_hospitalisations_per_year": round(sum(p["hospitalisations_per_year"] for p in pts) / len(pts), 1),
            "key_features": gd["key_features"],
            "treatment": gd["treatment"],
            "contraindications": gd["contraindications"],
            "critical_pearls": gd["critical_pearls"],
            "inheritance": gd["inheritance"][:300],
        }
    return result


def definitions():
    """Gene definitions, IHC markers, CMD DDx tables, dystroglycanopathy spectrum, glossary."""
    gene_defs = {}
    for gd in CMD_GENES:
        gene_defs[gd["gene"]] = {
            "full_name": gd["protein"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance_detail": gd["inheritance"],
            "alt_name": gd["alt_name"],
        }

    return {
        "gene_definitions": gene_defs,
        "cmd_severity_spectrum": {
            "POMT1_WWS":   "MOST SEVERE — Walker-Warburg Syndrome; agyria/cobblestone + Z-brainstem + ocular defects; usually lethal 1st year",
            "POMT2_WWS":   "MOST SEVERE — WWS identical to POMT1; obligate heterodimer; some hypomorphic = MEB-like",
            "FKTN_FCMD":   "SEVERE — Fukuyama CMD; cobblestone lissencephaly + ID mandatory + DCM adolescence; Japan common; survival childhood-adulthood",
            "POMGNT1_MEB": "SEVERE — Muscle-Eye-Brain; pachygyria + myopia/glaucoma + ID; Finnish founder; survival adulthood",
            "LAMA2_MDC1A": "SEVERE-MODERATE — MDC1A; absent merosin + leukodystrophy + neuropathy; no ID; partial deficiency = ambulant",
            "COL6A1_Ullrich": "SEVERE — Ullrich CMD; proximal weakness + distal laxity paradox + skin signs + early respiratory; no ID; near-normal CK",
            "COL6A2_Ullrich": "SEVERE — same Ullrich spectrum as COL6A1",
            "COL6A3_Bethlem": "MILD (Bethlem AD) — proximal weakness + contractures; slow progression; often ambulant lifelong; less severe than Ullrich",
        },
        "ihc_marker_table": {
            "LAMA2":   "Absent merosin (Laminin α2 IHC) on muscle AND skin biopsy — SPECIFIC for LAMA2 CMD; screen skin biopsy first (safer)",
            "COL6A1":  "Reduced/absent collagen VI IHC in basement membrane zone on muscle or skin biopsy — SPECIFIC for COL6 myopathy; all three COL6 genes cause same IHC pattern",
            "COL6A2":  "Same as COL6A1 — collagen VI IHC reduction/absence; gene panel needed to identify which COL6 gene",
            "COL6A3":  "Same as COL6A1/A2 — collagen VI IHC reduction; skin biopsy (elbow) preferred for screening",
            "FKTN":    "Reduced IIH6 antibody α-dystroglycan staining (shared dystroglycanopathy marker); also reduced FKTN IHC; glycosylated αDG absent",
            "POMT1":   "Absent IIH6 α-dystroglycan IHC staining — most severe reduction; POMT1 IHC usually absent; CK >50× ULN neonatal",
            "POMT2":   "Absent IIH6 α-dystroglycan IHC staining — same as POMT1; POMT2 IHC absent; gene panel to distinguish from POMT1",
            "POMGNT1": "Reduced IIH6 α-dystroglycan IHC staining — same dystroglycanopathy marker; POMGNT1 IHC reduced; Finnish founder PCR first-line if Finnish",
        },
        "brain_mri_table": {
            "LAMA2":   "T2 white matter hyperintensity (leukodystrophy) — NON-PROGRESSIVE; does not cause cognitive impairment; 80-100% of complete deficiency",
            "COL6A1":  "Normal brain MRI — COL6 myopathy does NOT affect brain; no leukodystrophy or lissencephaly",
            "COL6A2":  "Normal brain MRI — same as COL6A1; no CNS involvement",
            "COL6A3":  "Normal brain MRI — same as COL6A1/A2; no CNS involvement",
            "FKTN":    "COBBLESTONE LISSENCEPHALY Type II (pachygyria/polymicrogyria) + cerebellar hypoplasia; brain malformation progressive in severity with age",
            "POMT1":   "COBBLESTONE LISSENCEPHALY (agyria) + Z-SHAPED BRAINSTEM (pontine hypoplasia + cerebellar hypoplasia) PATHOGNOMONIC WWS; most severe MRI",
            "POMT2":   "Same as POMT1 — agyria/cobblestone + Z-brainstem; hypomorphic: pachygyria (MEB-like)",
            "POMGNT1": "PACHYGYRIA (less severe than agyria/WWS) + CEREBELLAR HYPOPLASIA; less severe than POMT1/POMT2; distinguishes MEB from WWS",
        },
        "ocular_involvement_table": {
            "LAMA2":   "No primary ocular involvement — myopia may be coincidental",
            "COL6A1":  "No primary ocular involvement",
            "COL6A2":  "No primary ocular involvement",
            "COL6A3":  "No primary ocular involvement",
            "FKTN":    "Retinal dysplasia — myopia + retinal hypoplasia; ophthalmology mandatory",
            "POMT1":   "SEVERE ocular defects — anterior chamber dysgenesis (Peters anomaly), coloboma, cataract, microphthalmia; PATHOGNOMONIC WWS",
            "POMT2":   "Same severe ocular defects as POMT1 in WWS; hypomorphic = less severe eye involvement",
            "POMGNT1": "MYOPIA + GLAUCOMA + RETINAL HYPOPLASIA PATHOGNOMONIC MEB; glaucoma treatment mandatory; less severe than POMT1/2",
        },
        "founder_mutations": {
            "FKTN_c3036_retrotransposon": "FKTN c.3036+IVS(~3kb SVA-type retrotransposon) — 85% of Japanese Fukuyama CMD alleles; rapid PCR screening available; Founder in Japanese population",
            "POMGNT1_pTyr688Cys": "POMGNT1 p.Tyr688Cys (c.2063A>T) — 85% of Finnish MEB alleles; PCR/sequencing screen; Founder in Finnish population",
        },
        "ddx_table": {
            "CMD_MRI_based_triage": {
                "Normal_MRI_CMD":           "LAMA2 (leukodystrophy non-specific) or COL6 CMD (normal MRI) → merosin IHC + collagen VI IHC",
                "Cobblestone_lissencephaly": "Dystroglycanopathy group: FKTN/POMT1/POMT2/POMGNT1 — reduce to gene by severity and ethnicity",
                "WWS_MRI_Z_brainstem":       "POMT1 (50%) or POMT2 (20%) — panel both genes first; then POMGNT1/FKTN/FKRP/LARGE1",
                "MEB_MRI_pachygyria":        "POMGNT1 (Finnish founder) or FKTN (Japan) or POMT2 (hypomorphic) — panel + founder screen",
            },
            "CMD_IHC_triage": {
                "Absent_merosin":             "LAMA2 — MDC1A diagnosis; confirm with LAMA2 sequencing",
                "Absent_collagen_VI":         "COL6A1 or COL6A2 or COL6A3 — panel all three COL6 genes",
                "Absent_IIH6_alphaDAG":       "Dystroglycanopathy panel: FKTN / POMT1 / POMT2 / POMGNT1 / FKRP / LARGE1 / ISPD etc.",
                "Normal_IHC":                 "Consider SEPN1/SELENON (rigid spine), ACTA1/RYR1 (congenital myopathy), or unusual CMD",
            },
            "CK_triage": {
                "CK_>50x_neonatal":          "POMT1 or POMT2 (WWS) → most severe; urgent IHC + gene panel",
                "CK_10-50x":                 "FKTN (FCMD) or POMGNT1 (MEB) or LAMA2 (complete MDC1A)",
                "CK_5-30x":                  "LAMA2 (MDC1A complete deficiency); less likely COL6",
                "CK_near_normal_or_1-5x":    "COL6 CMD (COL6A1/A2/A3 Ullrich or Bethlem) — key DDx from LAMA2",
            },
        },
        "collagen_vi_genetics": {
            "COL6A1_COL6A2_locus":   "Both on 21q22.3 — contiguous gene region on chromosome 21; always sequence both together when heterozygous variant found in either",
            "COL6A3_locus":          "2q37.3 — different chromosome from COL6A1/A2; no cosegregation issues with COL6A1/A2",
            "de_novo_dominant":      "~30% of Ullrich CMD cases are de novo dominant-negative in one of the three COL6 genes; parents may test negative; trio sequencing recommended",
            "Bethlem_vs_Ullrich":    "Bethlem (mild, AD): Gly-X-Y substitutions in triple helix → dominant negative; Ullrich (severe, AR or de novo): null biallelic or severe dominant-negative",
        },
        "dystroglycanopathy_spectrum": {
            "definition":           "Group of CMD/LGMD caused by abnormal glycosylation of α-dystroglycan (αDG); IIH6 antibody staining reduced in all",
            "POMT1_POMT2_step":     "First step: O-mannosylation of αDG in ER; POMT1+POMT2 heterodimer required",
            "POMGNT1_step":         "Second step: extension of O-mannose glycan; adds GlcNAc to mannose",
            "FKTN_step":            "Ribitol-5-phosphate addition to O-mannose; required for mature αDG glycosylation",
            "other_genes":          "FKRP, LARGE1, ISPD, GMPPB, TMEM5, RXYLT1, POMK, etc. — all cause same dystroglycanopathy spectrum",
            "severity_predictor":   "Residual POMT1+POMT2 enzyme activity predicts severity — null=WWS; partial=MEB; minor=LGMD-R",
        },
        "glossary": {
            "Merosin_deficiency":       "Absent laminin α2 (merosin) protein in muscle basement membrane — MDC1A hallmark; detectable by IHC on muscle or skin biopsy",
            "Collagen_VI_myopathy":     "Group of muscle diseases caused by variants in COL6A1/A2/A3; spectrum from severe Ullrich CMD to mild Bethlem Myopathy; all have reduced collagen VI IHC",
            "Cobblestone_lissencephaly": "Type II lissencephaly (smooth brain) caused by overmigration of neurons through disrupted basement membrane; pathognomonic of dystroglycanopathy CMD",
            "Walker_Warburg_Syndrome":  "Most severe congenital muscular dystrophy; agyria + cerebellar hypoplasia + ocular defects; usually lethal 1st year; POMT1/POMT2 most common",
            "Muscle_Eye_Brain_disease": "MEB — less severe WWS-like CMD; pachygyria + cerebellar hypoplasia + myopia/glaucoma + ID; POMGNT1 most common; Finnish founder",
            "Fukuyama_CMD":             "FCMD — most common CMD in Japan; cobblestone lissencephaly + ID + DCM in adolescence; FKTN gene; retrotransposon founder",
            "Alpha_dystroglycan":       "Extracellular matrix receptor protein; heavily glycosylated form binds laminin; glycosylation dependent on multiple glycosyltransferases (FKTN/POMT1/etc.)",
            "IIH6_antibody":            "Glycosylation-dependent antibody recognising mature glycosylated α-dystroglycan; reduced staining in ALL dystroglycanopathies",
            "Ullrich_CMD":              "Severe COL6 myopathy (AR or de novo dominant); proximal contractures + distal hyperlaxity paradox + skin signs + early respiratory failure",
            "Bethlem_Myopathy":         "Mild COL6 myopathy (AD); proximal weakness + finger/elbow contractures; slow progression; often ambulant lifelong",
            "Retrotransposon_founder":  "FKTN c.3036+IVS retrotransposon — mobile genetic element insertion causing FKTN loss-of-function; PCR detects the 3kb insertion",
            "Z_brainstem":              "Pathognomonic MRI appearance in WWS — pontine hypoplasia + cerebellar hypoplasia = Z-shape on sagittal view; confirms WWS without waiting for genetics",
            "Cyclosporin_A":            "Calcineurin inhibitor; in COL6 myopathy trials reduces mitochondrial PTP opening and autophagy induction; only CMD-specific therapy with pilot efficacy data",
            "Peroneal_muscle":          "LAMA2: distal weakness (foot drop/peroneal) common; NCVs slow (demyelinating neuropathy distinguishes MDC1A from other CMDs)",
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = overview()
    print(json.dumps({k: v for k, v in ov.items() if k != "gene_summaries"}, indent=2))
    print("\n=== BREAKDOWN keys ===")
    br = breakdown()
    for gene, data in br.items():
        print(f"  {gene}: {data['n_patients']} patients, alive={data['alive_pct']}%, ambulant_10yr={data['ambulant_10yr_pct']}%")
    print("\n=== DEFINITIONS keys ===")
    defs = definitions()
    print(list(defs.keys()))
