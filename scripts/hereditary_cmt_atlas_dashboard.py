#!/usr/bin/env python3
"""Hereditary-CMT-Atlas — Complete 8-Gene Charcot-Marie-Tooth Disease Atlas
(CMT1A/PMP22 · CMT1B/MPZ · CMTX1/GJB1 · CMT2A/MFN2 ·
 CMT4C/SH3TC2 · CMT4A/GDAP1 · CMT2E/NEFL · CMT4F/PRX).

PMP22    (Peripheral Myelin Protein 22; 160 aa; 17p12; AD;
          CMT1A (duplication) / HNPP (deletion) — most common CMT worldwide;
          UNIFORM NCV SLOWING <38 m/s ALL NERVES PATHOGNOMONIC for CMT1A;
          MLPA MANDATORY — NGS alone misses copy-number variants;
          HNPP: pressure palsies at entrapment sites;
          seed SEED_BASE+0).
MPZ      (Myelin Protein Zero; 248 aa; 1q23.3; AD;
          CMT1B — multiple phenotypes: early-severe / intermediate / late-mild;
          pThr124Met = late-onset mild, pSer44Phe = Dejerine-Sottas congenital severe;
          Nerve biopsy: onion bulbs; seed SEED_BASE+1).
GJB1     (Gap Junction Beta-1 / Connexin-32; 283 aa; Xq13.1; XLD;
          CMTX1 — X-LINKED DOMINANT: NO male-to-male transmission PATHOGNOMONIC;
          Males severely affected; females mildly affected (carrier);
          CNS: transient MRI white matter lesions with fever PATHOGNOMONIC;
          Intermediate NCV 25-45 m/s in males; seed SEED_BASE+2).
MFN2     (Mitofusin-2; 741 aa; 1p36.22; AD;
          CMT2A — OPTIC ATROPHY 20-30% PATHOGNOMONIC (CMT + vision loss = MFN2);
          Most severe CMT2 — early wheelchair common; de novo ~30-40%;
          Mitochondrial fusion gene; seed SEED_BASE+3).
SH3TC2   (SH3 Domain and Tetratricopeptide Repeats 2; 1288 aa; 5q32; AR;
          CMT4C — EARLY SCOLIOSIS PATHOGNOMONIC + cranial nerve involvement;
          Hearing loss 40-50%; most common AR CMT in Turkey/Pakistan/India;
          seed SEED_BASE+4).
GDAP1    (Ganglioside-Induced Differentiation-Associated Protein 1; 358 aa; 8q21.11; AR/AD;
          CMT4A (AR) / CMT2K (AD) — VOCAL CORD PARALYSIS 20-30% PATHOGNOMONIC (AR form);
          Mitochondrial outer membrane fission; North African/Spanish founders;
          seed SEED_BASE+5).
NEFL     (Neurofilament Light Chain; 543 aa; 8p21.2; AD;
          CMT2E (AD) / CMT1F (AR) — GIANT AXONS on nerve biopsy PATHOGNOMONIC;
          CSF NF-L elevated; early severe onset in AR form; p.Glu396Lys most common AD;
          seed SEED_BASE+6).
PRX      (Periaxin; 1461 aa; 19q13.13; AR;
          CMT4F — FOCALLY FOLDED MYELIN on nerve biopsy PATHOGNOMONIC;
          Severe early childhood onset; sensory > motor; Romani/Gypsy founders;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2142-2149).
"""

import random

SEED_BASE = 2142

CMT_GENES = [
    # -- PMP22 — CMT1A / HNPP --------------------------------------------------
    {
        "gene": "PMP22",
        "alt_name": (
            "PMP22 (PMP22-160aa-17p12 / AD — CMT1A-Charcot-Marie-Tooth-Disease-Type-1A — "
            "Gene-Duplication-17p12-70pct-All-CMT1 — HNPP-Hereditary-Neuropathy-Pressure-Palsies-Deletion — "
            "UNIFORM-NCV-SLOWING-<38-m/s-ALL-NERVES-PATHOGNOMONIC — MLPA-MANDATORY-NGS-Misses-CNV)"
        ),
        "protein": (
            "PMP22 -- 17p12 AD -- PMP22-160aa -- "
            "Peripheral-Myelin-Protein-22-Compact-Myelin-Structural-Protein-4-TM-Helices -- "
            "CMT1A-Charcot-Marie-Tooth-Type-1A-Most-Common-CMT-OMIM-118220 -- "
            "HNPP-Hereditary-Neuropathy-with-Liability-to-Pressure-Palsies-OMIM-162500 -- "
            "CMT1A-Tandem-Duplication-1.5Mb-17p12-PMP22-DOSAGE-EFFECT-70pct-All-CMT1 -- "
            "HNPP-Deletion-Same-Region-Tomaculous-Myelin-Sausage-Shaped -- "
            "UNIFORM-NCV-SLOWING-<38-m/s-ALL-NERVES-PATHOGNOMONIC-Even-Asymptomatic-Nerves -- "
            "MLPA-MANDATORY-CNV-Detection-NGS-Alone-MISSES-Duplication-Deletion -- "
            "Incidence-1-in-2500-Most-Common-Hereditary-Neuropathy -- "
            "Distal-Atrophy-Pes-Cavus-Hammertoes-Foot-Drop -- "
            "Ascorbic-Acid-RCT-FAILED-No-Disease-Modification -- "
            "17p12"
        ),
        "locus": "17p12",
        "protein_size": "160 aa",
        "inheritance": (
            "AD (autosomal dominant); gene dosage effect; "
            "CMT1A = tandem 1.5-Mb duplication of 17p12 containing PMP22 (3 copies); "
            "HNPP = deletion of same 1.5-Mb segment (1 copy); "
            "point mutations in PMP22 also cause CMT1A and CMT1E (rare); "
            "de novo duplication rate ~10%; "
            "GENETIC TESTING: MLPA or array CGH MANDATORY — NGS misses CNV; "
            "if MLPA negative but strong phenotype: PMP22 sequencing for point mutations"
        ),
        "age_of_onset": "First to second decade (childhood to young adult)",
        "pathognomonic": (
            "UNIFORM NCV SLOWING <38 m/s in ALL NERVES including asymptomatic ones = PATHOGNOMONIC CMT1A; "
            "NCV uniformity distinguishes from multifocal demyelination (CIDP); "
            "MLPA MANDATORY for diagnosis (NGS/WES misses duplication); "
            "Clinical: pes cavus + hammertoes + distal wasting + foot drop; "
            "HNPP: transient pressure palsy at fibular head/cubital tunnel/carpal tunnel; "
            "Nerve biopsy: CMT1A = onion bulbs; HNPP = sausage-shaped tomacula; "
            "Family history AD: parents/siblings often asymptomatic but NCS abnormal"
        ),
        "treatment": (
            "NO disease-modifying therapy (ascorbic acid RCT failed); "
            "ANKLE-FOOT ORTHOSES (AFOs): custom foot drop splints — reduces falls; "
            "PHYSIOTHERAPY: gait training, Achilles tendon stretching, strengthening; "
            "OCCUPATIONAL THERAPY: adaptive aids, grip splints; "
            "PODIATRY: pes cavus management, custom insoles; "
            "ORTHOPAEDIC: tendon transfer for severe foot drop (tibialis posterior transfer); "
            "HNPP: AVOID compression at entrapment sites — elbow pads, wrist rests; "
            "HNPP: avoid prolonged crouching / knee crossing; "
            "PAIN: gabapentin/pregabalin for neuropathic pain; "
            "GENETIC COUNSELLING: 50% offspring risk (AD); "
            "CASCADE TESTING: family members, siblings from age 16 yr; "
            "Exercise: moderate exercise beneficial; avoid extreme endurance sports"
        ),
        "key_biomarker": (
            "NCS: UNIFORM motor NCV <38 m/s all nerves (median, ulnar, peroneal) — PATHOGNOMONIC CMT1A; "
            "NCS: HNPP = focal slowing at entrapment sites + mild generalised slowing; "
            "MLPA: PMP22 duplication (CMT1A) or deletion (HNPP); "
            "Nerve biopsy: onion bulbs (CMT1A) or tomacula (HNPP) — rarely required; "
            "CK: normal; CSF: normal protein"
        ),
        "critical_flags": [
            "UNIFORM-NCV-<38-m/s-ALL-NERVES-PATHOGNOMONIC",
            "MLPA-MANDATORY-NGS-MISSES-CNV",
            "HNPP-PRESSURE-PALSIES-AVOID-COMPRESSION",
            "ASCORBIC-ACID-RCT-FAILED",
            "MOST-COMMON-CMT-1-IN-2500",
            "AFO-REDUCES-FALLS",
            "TENDON-TRANSFER-SEVERE-FOOT-DROP",
            "CASCADE-TESTING-50pct-RISK",
            "DISTINGUISH-FROM-CIDP-UNIFORM-SLOWING",
        ],
    },
    # -- MPZ — CMT1B -----------------------------------------------------------
    {
        "gene": "MPZ",
        "alt_name": (
            "MPZ (MPZ-248aa-1q23.3 / AD — CMT1B-Charcot-Marie-Tooth-Disease-Type-1B — "
            "Myelin-Protein-Zero-Major-Compact-Myelin-Protein-Ig-Domain — "
            "Multiple-Phenotypes-pThr124Met-Late-Onset-pSer44Phe-Dejerine-Sottas-Congenital-Severe — "
            "Onion-Bulbs-Nerve-Biopsy — NCV-Guided-Subtype-Stratification)"
        ),
        "protein": (
            "MPZ -- 1q23.3 AD -- MPZ-248aa -- "
            "Myelin-Protein-Zero-P0-Major-Compact-Myelin-Protein-Ig-Like-Domain-Single-TM-Helix -- "
            "CMT1B-Charcot-Marie-Tooth-Type-1B-OMIM-118200 -- "
            "Dejerine-Sottas-Syndrome-DSS-OMIM-145900-Severe-Congenital-Variant -- "
            "Congenital-Hypomyelination-CH-OMIM-605253-Neonatal-Severe -- "
            "pThr124Met-Late-Onset-Mild-Phenotype->40yr-Distinctive -- "
            "pSer44Phe-Congenital-Severe-Dejerine-Sottas-Respiratory-Risk -- "
            "NCV-Stratification-<10ms-Severe-10-25ms-Intermediate-25-38ms-Mild -- "
            "Onion-Bulbs-Nerve-Biopsy-Demyelinating -- "
            "Adhesion-Function-Myelin-Compaction-Ig-Domain-Interactions -- "
            "1q23.3"
        ),
        "locus": "1q23.3",
        "protein_size": "248 aa",
        "inheritance": (
            "AD (autosomal dominant) for most; AR (biallelic null) for severe congenital; "
            "dominant-negative (pSer44Phe, pGly68Asp) causes severe Dejerine-Sottas; "
            "haploinsufficiency (late truncating) causes late-mild phenotype; "
            "pThr124Met = distinctive late-onset (>40 yr) = mildest MPZ; "
            "genetic testing: MPZ sequencing; MLPA to exclude PMP22 duplication first; "
            "genotype-phenotype correlation CRITICAL for prognosis"
        ),
        "age_of_onset": "Variable by allele: neonatal (severe) → childhood (intermediate) → >40 yr (pThr124Met)",
        "pathognomonic": (
            "MULTIPLE PHENOTYPES defined by genotype: "
            "pThr124Met = LATE ONSET (>40 yr) DISTINCTIVE — mildest CMT1B; "
            "pSer44Phe = congenital Dejerine-Sottas — neonatal hypotonia, respiratory risk; "
            "NCS: variable slowing — <10 m/s (Dejerine-Sottas), 10-25 m/s (intermediate CMT1B), "
            "25-38 m/s (mild CMT1B); "
            "NERVE BIOPSY: onion bulbs (if diagnostic uncertainty); "
            "Pes cavus + distal wasting; scoliosis in severe forms; "
            "distinguish from CMT1A: MPZ sequencing after negative MLPA"
        ),
        "treatment": (
            "NO disease-modifying therapy; "
            "AFOs: foot drop management; "
            "RESPIRATORY: spirometry for severe (Dejerine-Sottas) cases — phrenic nerve risk; "
            "NIV if FVC < 70% (severe congenital forms); "
            "PHYSIOTHERAPY: gait, strengthening; "
            "SCOLIOSIS SURVEILLANCE: spinal X-ray 6-monthly in childhood (severe forms); "
            "ORTHOPAEDIC: scoliosis correction if Cobb angle >40°; "
            "PAIN: gabapentin/pregabalin; "
            "NEUROPATHIC PAIN: CMT1B often painful; "
            "GENETIC COUNSELLING: 50% offspring risk (AD); "
            "GENOTYPE-PHENOTYPE: discuss prognosis based on specific variant"
        ),
        "key_biomarker": (
            "NCS: motor NCV variable by genotype (<10, 10-25, 25-38 m/s); "
            "Nerve biopsy: onion bulbs (demyelinating); "
            "MPZ sequencing; "
            "MLPA: PMP22 normal (no duplication); "
            "CK: normal or mildly elevated"
        ),
        "critical_flags": [
            "MULTIPLE-PHENOTYPES-GENOTYPE-CRITICAL",
            "pThr124Met-LATE-ONSET->40yr-DISTINCTIVE",
            "pSer44Phe-DEJERINE-SOTTAS-CONGENITAL-SEVERE",
            "NCV-STRATIFICATION-PROGNOSIS",
            "ONION-BULBS-NERVE-BIOPSY",
            "RESPIRATORY-SURVEILLANCE-SEVERE-FORMS",
            "SCOLIOSIS-SURVEILLANCE-CHILDHOOD",
            "MLPA-FIRST-TO-EXCLUDE-PMP22-DUP",
            "NO-DISEASE-MODIFYING-RX",
        ],
    },
    # -- GJB1 — CMTX1 ----------------------------------------------------------
    {
        "gene": "GJB1",
        "alt_name": (
            "GJB1 (GJB1-283aa-Xq13.1 / XLD — CMTX1-Charcot-Marie-Tooth-Disease-X-Linked-Type-1 — "
            "Connexin-32-Gap-Junction-Protein-Myelin-Paranodal-Loops-Schmidt-Lantermann-Incisures — "
            "NO-Male-to-Male-Transmission-PATHOGNOMONIC — "
            "CNS-White-Matter-Lesions-Transient-Fever-PATHOGNOMONIC — "
            "Intermediate-NCV-25-45-m/s-Males)"
        ),
        "protein": (
            "GJB1 -- Xq13.1 XLD -- GJB1-283aa -- "
            "Gap-Junction-Beta-1-Connexin-32-CX32-Paranodal-Loop-Schmidt-Lantermann-Incisure-Gap-Junction -- "
            "CMTX1-Charcot-Marie-Tooth-Disease-X-Linked-Type-1-OMIM-302800 -- "
            "X-LINKED-DOMINANT-Males-Severely-Affected-Females-Mildly-Affected-Carriers -- "
            "NO-Male-to-Male-Transmission-PATHOGNOMONIC-X-Linked-DDx-from-AD -- "
            "INTERMEDIATE-NCV-25-45-m/s-MALES-Neither-Clearly-Demyelinating-Nor-Axonal -- "
            "CNS-White-Matter-Lesions-MRI-Transient-After-Fever-Illness-Altitude-PATHOGNOMONIC -- "
            "Connexin-32-Forms-Gap-Junctions-Reflexive-Loops-Myelin-Internodal-Communication -- "
            "Most-Common-X-Linked-Neuropathy-2nd-Most-Common-CMT-Overall -- "
            "Female-Carriers-Intermediate-Severity-Mild-NCV-Slowing -- "
            "Xq13.1"
        ),
        "locus": "Xq13.1",
        "protein_size": "283 aa",
        "inheritance": (
            "X-LINKED DOMINANT (XLD); "
            "NO male-to-male transmission — affected father transmits to ALL daughters, no sons; "
            "affected mother transmits to 50% sons (severely affected) and 50% daughters (mildly affected); "
            "males: severe; females: mild-moderate (carrier = lyonization); "
            "genetic testing: GJB1 sequencing + MLPA (exclude PMP22 dup first); "
            ">400 pathogenic variants in GJB1; missense most common"
        ),
        "age_of_onset": "First to second decade (males); later in carrier females",
        "pathognomonic": (
            "NO MALE-TO-MALE TRANSMISSION in pedigree = PATHOGNOMONIC X-linked; "
            "CNS: transient MRI white matter lesions with fever/illness/altitude = PATHOGNOMONIC CMTX1; "
            "lesions reversible — resolve over days-weeks without treatment; "
            "INTERMEDIATE NCV 25-45 m/s in affected males (not clearly demyelinating nor axonal); "
            "NCS asymmetry: males severe, carrier females mild; "
            "Distal weakness, pes cavus, areflexia in males; "
            "Most common X-linked CMT; 2nd most common CMT overall after CMT1A"
        ),
        "treatment": (
            "NO disease-modifying therapy; "
            "CNS LESION EPISODES: supportive — hospital admission only if seizures/stroke-like symptoms; "
            "avoid dehydration (exacerbates CNS episodes); maintain hydration during illness; "
            "AFOs: foot drop in males; "
            "PHYSIOTHERAPY: strengthening + gait; "
            "NEUROLOGIST + MRI: confirm CNS episodes are reversible (distinguish from true stroke); "
            "PAIN: gabapentin/pregabalin for neuropathic pain; "
            "GENETIC COUNSELLING: X-linked pattern — all daughters of affected males are carriers; "
            "female carriers: NCS often shows mild slowing — do not dismiss as normal; "
            "MALES: anticipate progressive course; wheelchair possible in 5th-6th decade"
        ),
        "key_biomarker": (
            "NCS: intermediate NCV 25-45 m/s in males (neither clearly demyelinating nor axonal); "
            "MRI brain: transient white matter T2 lesions (reversible); "
            "GJB1 sequencing (X-linked pattern in pedigree); "
            "CK: normal; "
            "Nerve biopsy: variable — myelin changes at paranodal loops (rarely needed)"
        ),
        "critical_flags": [
            "NO-MALE-TO-MALE-TRANSMISSION-PATHOGNOMONIC",
            "CNS-WM-LESIONS-TRANSIENT-FEVER-PATHOGNOMONIC",
            "INTERMEDIATE-NCV-25-45-m/s-MALES",
            "FEMALE-CARRIERS-MILDLY-AFFECTED",
            "ALL-DAUGHTERS-AFFECTED-FATHERS-ARE-CARRIERS",
            "LESIONS-REVERSIBLE-NO-SPECIFIC-RX",
            "HYDRATION-DURING-ILLNESS",
            "MRI-TO-CONFIRM-REVERSIBILITY-NOT-STROKE",
            "MOST-COMMON-X-LINKED-CMT",
        ],
    },
    # -- MFN2 — CMT2A ----------------------------------------------------------
    {
        "gene": "MFN2",
        "alt_name": (
            "MFN2 (MFN2-741aa-1p36.22 / AD — CMT2A-Charcot-Marie-Tooth-Disease-Type-2A — "
            "Mitofusin-2-Mitochondrial-Fusion-Outer-Membrane-GTPase — "
            "OPTIC-ATROPHY-20-30pct-PATHOGNOMONIC-CMT-Plus-Vision-Loss-Equals-MFN2 — "
            "Most-Severe-CMT2-Early-Wheelchair-Common — De-Novo-30-40pct)"
        ),
        "protein": (
            "MFN2 -- 1p36.22 AD -- MFN2-741aa -- "
            "Mitofusin-2-GTPase-Dynamin-Like-Coiled-Coil-Outer-Mitochondrial-Membrane-Fusion -- "
            "CMT2A-Charcot-Marie-Tooth-Type-2A-OMIM-609260 -- "
            "Hereditary-Motor-and-Sensory-Neuropathy-HMSN-Type-IIA -- "
            "OPTIC-ATROPHY-20-30pct-PATHOGNOMONIC-CMT-PLUS-VISION-LOSS-Equals-MFN2 -- "
            "Most-Severe-CMT2-Early-Wheelchair-First-Decade-in-Severe-Cases -- "
            "De-Novo-Mutations-30-40pct-Sporadic-Cases -- "
            "Mitochondrial-Fusion-Failure-Elongated-Fragmented-Mitochondria-Motor-Neurons -- "
            "Axonal-NCV-Normal-or-Mildly-Reduced-CMAP-Low -- "
            "MFN2-1p36.22-ALLELIC-Hereditary-Spastic-Paraplegia-SPG4-Like-Overlap -- "
            "1p36.22"
        ),
        "locus": "1p36.22",
        "protein_size": "741 aa",
        "inheritance": (
            "AD (autosomal dominant); GTPase domain mutations most pathogenic; "
            "de novo mutations ~30-40% of severe/early-onset cases; "
            "occasionally AR (homozygous null) in consanguineous families — severe; "
            "genetic testing: MFN2 sequencing; note de novo rate; "
            "optic atrophy co-segregation may suggest separate OPA1 mutation or MFN2 pleiotropy"
        ),
        "age_of_onset": "First decade (severe de novo) to 3rd decade (mild inherited); widely variable",
        "pathognomonic": (
            "OPTIC ATROPHY in 20-30% = PATHOGNOMONIC: CMT + visual loss → test MFN2 first; "
            "early wheelchair use (1st-2nd decade in severe cases); "
            "NCS: AXONAL pattern — normal/mildly reduced NCV, reduced CMAP amplitudes; "
            "upper limb > lower limb involvement in some; "
            "pyramidal signs possible; white matter changes on MRI (severe); "
            "de novo: sporadic presentation, unaffected parents, NCS parents normal; "
            "most severe CMT2 — contrast with milder MFN2 missenses"
        ),
        "treatment": (
            "NO disease-modifying therapy; "
            "OPHTHALMOLOGY: annual fundoscopy + OCT (optic atrophy monitoring); "
            "LOW VISION AIDS: magnification, high contrast; "
            "VISUAL FIELD TESTING: annual Humphrey visual fields; "
            "AFOs: foot drop — often early and severe; "
            "WHEELCHAIR: powerchair assessment early when indicated; "
            "PHYSIOTHERAPY: stretching, strengthening, contracture prevention; "
            "SCOLIOSIS SURVEILLANCE: spinal X-ray if truncal weakness; "
            "PAIN: neuropathic pain common — gabapentin/pregabalin; "
            "GENETIC COUNSELLING: 50% offspring risk; de novo rate means sporadic cases real; "
            "MRI BRAIN: white matter changes in severe de novo cases"
        ),
        "key_biomarker": (
            "NCS: axonal — low CMAP, normal/mildly reduced NCV (<38 m/s NOT expected); "
            "Fundoscopy/OCT: optic disc pallor + RNFL thinning; "
            "Visual fields: central scotoma in optic atrophy; "
            "MFN2 sequencing; "
            "CK: normal or mildly elevated"
        ),
        "critical_flags": [
            "OPTIC-ATROPHY-20-30pct-PATHOGNOMONIC",
            "CMT-PLUS-VISION-LOSS-EQUALS-MFN2",
            "MOST-SEVERE-CMT2-EARLY-WHEELCHAIR",
            "DE-NOVO-30-40pct-SPORADIC",
            "AXONAL-NCS-NORMAL-NCV-LOW-CMAP",
            "ANNUAL-FUNDOSCOPY-OCT-MANDATORY",
            "NO-DISEASE-MODIFYING-RX",
            "SCOLIOSIS-SURVEILLANCE",
            "DISTINGUISH-FROM-HEREDITARY-OPTIC-NEUROPATHY",
        ],
    },
    # -- SH3TC2 — CMT4C --------------------------------------------------------
    {
        "gene": "SH3TC2",
        "alt_name": (
            "SH3TC2 (SH3TC2-1288aa-5q32 / AR — CMT4C-Charcot-Marie-Tooth-Disease-Type-4C — "
            "SH3-Domain-Tetratricopeptide-Repeats-2-Schwann-Cell-Endosomal-Recycling — "
            "EARLY-SCOLIOSIS-PATHOGNOMONIC-50-70pct — "
            "Cranial-Nerve-Involvement-Hearing-Loss-40-50pct — "
            "Most-Common-AR-CMT-Turkey-Pakistan-India)"
        ),
        "protein": (
            "SH3TC2 -- 5q32 AR -- SH3TC2-1288aa -- "
            "SH3-Domain-Tetratricopeptide-Repeats-2-Recyclin11-Schwann-Cell-Endosomal-Recycling-Rab11 -- "
            "CMT4C-Charcot-Marie-Tooth-Type-4C-OMIM-601596 -- "
            "Autosomal-Recessive-Demyelinating-CMT -- "
            "EARLY-SCOLIOSIS-50-70pct-Before-Neuropathy-Symptomatic-PATHOGNOMONIC-CMT4C -- "
            "Cranial-Nerve-Involvement-CN7-Facial-Palsy-CN8-Hearing-Loss-40-50pct -- "
            "Most-Common-AR-CMT-Turkey-Pakistan-India-Gypsies-Roma -- "
            "Basal-Lamina-Onion-Bulbs-Schwann-Cell-Cytoplasm-Abnormal-Nerve-Biopsy -- "
            "NCV-Severe-Slowing-<15-m/s-Demyelinating -- "
            "Founder-Effect-Turkish-Pakistani-Indian-c.2860C>T-p.Arg954Stop -- "
            "5q32"
        ),
        "locus": "5q32",
        "protein_size": "1288 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "c.2860C>T (p.Arg954Stop) = Turkish/Pakistani/Indian founder; "
            "c.2860C>T homozygous in consanguineous Turkish pedigrees (common); "
            "genetic testing: SH3TC2 sequencing; "
            "25% sibling recurrence; "
            "carrier parents typically asymptomatic"
        ),
        "age_of_onset": "Childhood to early adolescence (5-15 years)",
        "pathognomonic": (
            "EARLY SCOLIOSIS (50-70%) preceding severe neuropathy symptoms = PATHOGNOMONIC CMT4C; "
            "scoliosis can be the presenting complaint (seen by orthopaedics before neurology); "
            "cranial nerve involvement: facial palsy (CN7) + sensorineural hearing loss (CN8, 40-50%); "
            "NCS: severe slowing <15 m/s (severe demyelinating); "
            "nerve biopsy: basal lamina onion bulbs + Schwann cell cytoplasmic inclusions; "
            "most common AR CMT in Turkey, Pakistan, India; "
            "wheelchair use 30-40% by 3rd decade"
        ),
        "treatment": (
            "NO disease-modifying therapy; "
            "SCOLIOSIS: spinal X-ray 6-monthly in childhood — early orthopaedic referral; "
            "bracing if Cobb angle 25-40°; spinal fusion if Cobb > 45°; "
            "AUDIOLOGY: pure tone audiogram annually — hearing aids when indicated; "
            "COCHLEAR IMPLANT: evaluate if severe SNHL; "
            "ENT: facial palsy assessment — eye protection if lagophthalmos; "
            "AFOs: severe foot drop — early orthotic intervention; "
            "PHYSIOTHERAPY: gait, scoliosis-related muscle strengthening; "
            "WHEELCHAIR: early assessment for powered mobility; "
            "GENETIC COUNSELLING: 25% sibling recurrence; "
            "CONSANGUINITY COUNSELLING: carrier frequency high in Turkish/Pakistani populations"
        ),
        "key_biomarker": (
            "NCS: severe motor NCV <15 m/s (severe demyelinating AR CMT); "
            "Audiogram: SNHL (CN8 involvement); "
            "Spinal X-ray: scoliosis (Cobb angle); "
            "Nerve biopsy: basal lamina onion bulbs + Schwann cell inclusions; "
            "SH3TC2 sequencing; "
            "CK: normal"
        ),
        "critical_flags": [
            "EARLY-SCOLIOSIS-PATHOGNOMONIC-50-70pct",
            "CRANIAL-NERVE-CN7-CN8-INVOLVEMENT",
            "HEARING-LOSS-40-50pct-AUDIOLOGY-MANDATORY",
            "MOST-COMMON-AR-CMT-TURKEY-PAKISTAN-INDIA",
            "SCOLIOSIS-MAY-PRECEDE-NEUROPATHY-SYMPTOMS",
            "SPINAL-XRAY-6-MONTHLY-CHILDHOOD",
            "WHEELCHAIR-30-40pct-3rd-DECADE",
            "25pct-SIBLING-RECURRENCE",
            "FOUNDER-c2860C>T-Turkish-Pakistani",
        ],
    },
    # -- GDAP1 — CMT4A ---------------------------------------------------------
    {
        "gene": "GDAP1",
        "alt_name": (
            "GDAP1 (GDAP1-358aa-8q21.11 / AR — CMT4A-Charcot-Marie-Tooth-Disease-Type-4A — "
            "Ganglioside-Induced-Differentiation-Associated-Protein-1-Mitochondrial-Fission — "
            "VOCAL-CORD-PARALYSIS-20-30pct-PATHOGNOMONIC-AR-Form — "
            "North-African-Spanish-Founders-pArg282His-pGln218His — "
            "Also-Dominant-CMT2K-AD-Milder)"
        ),
        "protein": (
            "GDAP1 -- 8q21.11 AR/AD -- GDAP1-358aa -- "
            "Ganglioside-Induced-Differentiation-Associated-Protein-1-Mitochondrial-Outer-Membrane-Fission -- "
            "CMT4A-Charcot-Marie-Tooth-Type-4A-AR-OMIM-214400 -- "
            "CMT2K-Charcot-Marie-Tooth-Type-2K-AD-OMIM-607831 -- "
            "VOCAL-CORD-PARALYSIS-20-30pct-PATHOGNOMONIC-AR-Biallelic-Form -- "
            "Diaphragm-Palsy-Respiratory-Failure-AR-Severe-Form -- "
            "North-African-Spanish-Founders-pArg282His-pGln218His-pLeu239Phe -- "
            "Mitochondrial-Fission-Failure-Elongated-Mitochondria-Schwann-Cells-Axons -- "
            "Early-Childhood-Onset-Severe-AR-Form-vs-Later-AD-Form -- "
            "8q21.11"
        ),
        "locus": "8q21.11",
        "protein_size": "358 aa",
        "inheritance": (
            "AR (autosomal recessive) for CMT4A (severe); "
            "AD (autosomal dominant) for CMT2K (milder, axonal); "
            "biallelic null/severe mutations → AR severe demyelinating + vocal cord palsy; "
            "heterozygous missense → AD axonal milder CMT2K; "
            "pArg282His (North African), pGln218His (Spanish/Tunisian), pLeu239Phe (Japanese) founders; "
            "genetic testing: GDAP1 sequencing; note inheritance pattern critical"
        ),
        "age_of_onset": "Infancy to early childhood (AR); young adult (AD CMT2K)",
        "pathognomonic": (
            "VOCAL CORD PARALYSIS in 20-30% of AR biallelic form = PATHOGNOMONIC (similar to DCTN1/dHMN7B); "
            "hoarse voice → stridor → respiratory failure risk; "
            "LARYNGOSCOPY MANDATORY at diagnosis in CMT4A; "
            "diaphragm palsy in severe AR cases; "
            "NCS AR form: mixed demyelinating-axonal (intermediate NCV) or demyelinating; "
            "NCS AD form (CMT2K): axonal; "
            "early severe childhood onset in AR biallelic null; "
            "North African/Spanish ancestry: GDAP1 most likely AR CMT cause"
        ),
        "treatment": (
            "NO disease-modifying therapy; "
            "LARYNGOSCOPY MANDATORY at diagnosis (vocal fold mobility); "
            "ENT 6-monthly surveillance — airway safety; "
            "TRACHEOSTOMY if bilateral vocal fold paralysis with stridor; "
            "RESPIRATORY: spirometry 6-monthly; NIV if FVC < 70%; "
            "GASTROSTOMY if dysphagia + bulbar involvement; "
            "AFOs: early foot drop intervention; "
            "PHYSIOTHERAPY: contracture prevention, gait; "
            "WHEELCHAIR: early assessment in severe AR cases; "
            "GENETIC COUNSELLING: AR = 25% sibling risk; AD = 50% offspring risk; "
            "NORTH AFRICAN/SPANISH ancestry: targeted founder mutation testing first"
        ),
        "key_biomarker": (
            "Laryngoscopy: vocal fold mobility (mandatory in CMT4A); "
            "Spirometry: FVC (diaphragm/respiratory monitoring); "
            "NCS: variable — demyelinating or intermediate (AR); axonal (AD); "
            "GDAP1 sequencing; "
            "Mitochondrial morphology: elongated/fragmented (research); "
            "CK: normal or mildly elevated"
        ),
        "critical_flags": [
            "VOCAL-CORD-PARALYSIS-20-30pct-PATHOGNOMONIC",
            "LARYNGOSCOPY-MANDATORY-CMT4A-DIAGNOSIS",
            "RESPIRATORY-FAILURE-AR-SEVERE-FORM",
            "TRACHEOSTOMY-BILATERAL-PALSY",
            "AR-CMT4A-vs-AD-CMT2K-SEVERITY-CRITICAL",
            "NORTH-AFRICAN-SPANISH-FOUNDERS",
            "FVC-SPIROMETRY-MANDATORY",
            "25pct-SIBLING-RISK-AR",
            "NO-DISEASE-MODIFYING-RX",
        ],
    },
    # -- NEFL — CMT2E / CMT1F --------------------------------------------------
    {
        "gene": "NEFL",
        "alt_name": (
            "NEFL (NEFL-543aa-8p21.2 / AD — CMT2E-Charcot-Marie-Tooth-Disease-Type-2E — "
            "Neurofilament-Light-Chain-NF-L-Cytoskeletal-Scaffold-Intermediate-Filament — "
            "GIANT-AXONS-Nerve-Biopsy-PATHOGNOMONIC — "
            "CSF-NF-L-Elevated-Disease-Severity-Biomarker — "
            "pGlu396Lys-Most-Common-AD-Mutation — AR-CMT1F-Severe-Infantile)"
        ),
        "protein": (
            "NEFL -- 8p21.2 AD/AR -- NEFL-543aa -- "
            "Neurofilament-Light-Chain-NF-L-68kDa-Type-IV-Intermediate-Filament-Cytoskeletal-Scaffold -- "
            "CMT2E-Charcot-Marie-Tooth-Type-2E-AD-OMIM-607684 -- "
            "CMT1F-Charcot-Marie-Tooth-Type-1F-AR-Severe-OMIM-607734 -- "
            "GIANT-AXONS-Abnormal-Neurofilament-Accumulation-Nerve-Biopsy-PATHOGNOMONIC -- "
            "CSF-NF-L-Elevated-Serum-NF-L-Elevated-Disease-Activity-Biomarker -- "
            "pGlu396Lys-Most-Common-AD-Missense-Early-Adult-Onset -- "
            "AR-CMT1F-Biallelic-Null-Infantile-Severe-Demyelinating -- "
            "Neurofilament-Triplet-NF-L-NF-M-NF-H-Assembly-Disruption -- "
            "8p21.2"
        ),
        "locus": "8p21.2",
        "protein_size": "543 aa",
        "inheritance": (
            "AD (autosomal dominant) for CMT2E — missense dominant-negative effect; "
            "AR (autosomal recessive) for CMT1F — biallelic null/severe mutations; "
            "pGlu396Lys = most common AD mutation (CMT2E); "
            "AD CMT2E: young adult onset, axonal; "
            "AR CMT1F: infantile severe demyelinating; "
            "genetic testing: NEFL sequencing; note inheritance pattern"
        ),
        "age_of_onset": "Young adult (AD CMT2E); infantile/early childhood (AR CMT1F)",
        "pathognomonic": (
            "GIANT AXONS on nerve biopsy = PATHOGNOMONIC neurofilament accumulation; "
            "CSF NF-L elevated (reflects neuroaxonal damage); "
            "serum NF-L elevated — disease activity monitoring; "
            "NCS AD CMT2E: axonal (low CMAP, near-normal NCV); "
            "NCS AR CMT1F: severe demyelinating (<10 m/s); "
            "AR infantile form: early onset, severe, wheelchair by 2nd decade; "
            "AD form: slow progression, mild-moderate disability; "
            "CSF NF-L is a monitoring biomarker (research/trials)"
        ),
        "treatment": (
            "NO disease-modifying therapy; "
            "NF-L BIOMARKER: serum/CSF NF-L monitoring in research/trial context; "
            "AFOs: foot drop management; "
            "PHYSIOTHERAPY: gait, contracture prevention; "
            "WHEELCHAIR: early assessment for AR CMT1F (severe infantile); "
            "RESPIRATORY: spirometry for severe infantile form; "
            "PAIN: neuropathic pain management — gabapentin/pregabalin; "
            "GENETIC COUNSELLING: AD CMT2E 50% offspring risk; AR CMT1F 25% sibling risk; "
            "NERVE BIOPSY: rarely needed but demonstrates giant axons if diagnostic uncertainty; "
            "RESEARCH TRIALS: NF-L as primary endpoint in therapeutic trials"
        ),
        "key_biomarker": (
            "Serum NF-L: elevated — disease activity correlate; "
            "CSF NF-L: elevated; "
            "Nerve biopsy: giant axons (neurofilament accumulation) — PATHOGNOMONIC; "
            "NCS: axonal (AD CMT2E) or severe demyelinating (AR CMT1F); "
            "NEFL sequencing; "
            "CK: normal"
        ),
        "critical_flags": [
            "GIANT-AXONS-NERVE-BIOPSY-PATHOGNOMONIC",
            "CSF-SERUM-NF-L-ELEVATED-BIOMARKER",
            "AD-CMT2E-vs-AR-CMT1F-PHENOTYPE-CRITICAL",
            "AR-CMT1F-INFANTILE-SEVERE",
            "NF-L-DISEASE-ACTIVITY-MONITORING",
            "NEUROFILAMENT-ACCUMULATION-MECHANISM",
            "AFO-PHYSIOTHERAPY",
            "AD-50pct-AR-25pct-RISK",
            "NO-DISEASE-MODIFYING-RX",
        ],
    },
    # -- PRX — CMT4F -----------------------------------------------------------
    {
        "gene": "PRX",
        "alt_name": (
            "PRX (PRX-1461aa-19q13.13 / AR — CMT4F-Charcot-Marie-Tooth-Disease-Type-4F — "
            "Periaxin-L-Periaxin-S-Dystrophin-Like-Scaffold-Schwann-Cell-Abaxonal-Cytoplasm — "
            "FOCALLY-FOLDED-MYELIN-Nerve-Biopsy-PATHOGNOMONIC — "
            "Sensory->Motor-Involvement-Distinctive — "
            "Romani-Gypsy-Spanish-Pakistani-Founders)"
        ),
        "protein": (
            "PRX -- 19q13.13 AR -- PRX-1461aa -- "
            "Periaxin-L-Periaxin-S-PDZ-Domain-Dystrophin-Related-Protein-Complex-Schwann-Cell-Abaxonal-Cytoplasm -- "
            "CMT4F-Charcot-Marie-Tooth-Type-4F-OMIM-614895 -- "
            "Dejerine-Sottas-Syndrome-DSS-Severe-Congenital-Biallelic-Null -- "
            "FOCALLY-FOLDED-MYELIN-Aberrant-Myelin-Folding-Nerve-Biopsy-PATHOGNOMONIC -- "
            "SENSORY-MORE-THAN-MOTOR-INVOLVEMENT-DISTINCTIVE -- "
            "Romani-Gypsy-Spanish-Pakistani-Founder-Mutations -- "
            "Severe-Early-Childhood-Onset-2-5yr -- "
            "Periaxin-Anchors-DRP2-Dystroglycan-Complex-Abaxonal-Schwann-Cell -- "
            "19q13.13"
        ),
        "locus": "19q13.13",
        "protein_size": "1461 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "Romani/Gypsy founder mutations (c.2122C>T, p.Arg708Stop); "
            "Spanish founder; Pakistani founder; "
            "biallelic null → severe CMT4F; "
            "genetic testing: PRX sequencing; "
            "25% sibling recurrence; consanguinity association in Pakistani families"
        ),
        "age_of_onset": "Early childhood (2-5 years); severe onset",
        "pathognomonic": (
            "FOCALLY FOLDED MYELIN on nerve biopsy = PATHOGNOMONIC CMT4F; "
            "myelin invaginates into axon creating focally folded (tomacula-like but folded not thickened) appearance; "
            "SENSORY > MOTOR involvement = distinctive (most CMT = motor > sensory); "
            "severe sensory loss: pain insensitivity, proprioception loss; "
            "early childhood onset; "
            "NCS: very slow NCV (< 10 m/s); "
            "scoliosis common; areflexia from very early; "
            "Romani/Pakistani ancestry → test PRX as priority AR CMT cause"
        ),
        "treatment": (
            "NO disease-modifying therapy; "
            "SENSORY NEUROPATHY PRECAUTIONS: skin inspection daily (pressure ulcers, burns, trauma); "
            "foot care: protective footwear, podiatry; "
            "PAIN INSENSITIVITY: injury prevention education for family; "
            "protective equipment; "
            "AFOs: foot drop + proprioceptive support; "
            "SCOLIOSIS SURVEILLANCE: spinal X-ray 6-monthly in childhood; "
            "ORTHOPAEDIC: bracing or fusion for severe scoliosis; "
            "PHYSIOTHERAPY: balance training (severe proprioception loss); "
            "WHEELCHAIR: early assessment — many need powered chair by 2nd decade; "
            "GENETIC COUNSELLING: 25% sibling recurrence; "
            "ROMANI/PAKISTANI ancestry: PRX founder testing as priority"
        ),
        "key_biomarker": (
            "Nerve biopsy: focally folded myelin (dedifferentiated Schwann cells) — PATHOGNOMONIC; "
            "NCS: very slow motor NCV (< 10 m/s), absent SNAPs (sensory > motor); "
            "Sural nerve biopsy preferred; "
            "PRX sequencing; "
            "Spinal X-ray: scoliosis; "
            "CK: normal"
        ),
        "critical_flags": [
            "FOCALLY-FOLDED-MYELIN-PATHOGNOMONIC",
            "SENSORY->MOTOR-DISTINCTIVE",
            "SEVERE-EARLY-CHILDHOOD-2-5yr",
            "ROMANI-GYPSY-PAKISTANI-FOUNDERS",
            "DAILY-SKIN-INSPECTION-INJURY-PREVENTION",
            "PROPRIOCEPTION-LOSS-BALANCE-TRAINING",
            "SCOLIOSIS-SURVEILLANCE-MANDATORY",
            "WHEELCHAIR-EARLY-2nd-DECADE",
            "25pct-SIBLING-RISK",
        ],
    },
]


def _build_cohort(gene_data: dict, seed: int, n: int = 40) -> list:
    """Build a synthetic 40-patient cohort for one CMT gene."""
    rng = random.Random(seed)
    gene = gene_data["gene"]
    cohort = []
    for pid in range(n):
        if gene == "PMP22":
            # CMT1A (dup) majority; HNPP (del) minority modelled
            hnpp = rng.random() < 0.25  # 25% HNPP vs 75% CMT1A
            onset = rng.randint(5, 20)
            demyelinating = True
            hnpp_pressure_palsy = hnpp
            foot_drop = not hnpp and rng.random() < 0.80
            hand_wasting = not hnpp and rng.random() < 0.55
            sensory_loss = rng.random() < 0.60
            optic_atrophy = False
            scoliosis = rng.random() < 0.20
            hearing_loss = False
            vocal_fold_palsy = False
            respiratory_failure = False
            cnv_related = True
            cnx_wm_lesion = False
            x_linked = False
            wheelchair = not hnpp and rng.random() < 0.10
            giant_axons = False
            sensory_predominant = False
            focal_myelin = False
        elif gene == "MPZ":
            # Variable phenotypes: 20% Dejerine-Sottas (severe), 60% intermediate, 20% late-mild
            subtype = rng.choices(["severe", "intermediate", "mild"], weights=[0.20, 0.60, 0.20])[0]
            if subtype == "severe":
                onset = rng.randint(0, 2)
            elif subtype == "intermediate":
                onset = rng.randint(5, 20)
            else:
                onset = rng.randint(40, 60)
            demyelinating = True
            hnpp_pressure_palsy = False
            foot_drop = rng.random() < 0.75
            hand_wasting = rng.random() < 0.60
            sensory_loss = rng.random() < 0.65
            optic_atrophy = False
            scoliosis = subtype == "severe" and rng.random() < 0.55
            hearing_loss = False
            vocal_fold_palsy = False
            respiratory_failure = subtype == "severe" and rng.random() < 0.20
            cnv_related = False
            cnx_wm_lesion = False
            x_linked = False
            wheelchair = subtype in ("severe", "intermediate") and rng.random() < 0.30
            giant_axons = False
            sensory_predominant = False
            focal_myelin = False
        elif gene == "GJB1":
            # Males ~60%, carrier females ~40%
            male = rng.random() < 0.60
            onset = rng.randint(10, 30)
            demyelinating = male and rng.random() < 0.75  # intermediate NCV
            hnpp_pressure_palsy = False
            foot_drop = male and rng.random() < 0.75
            hand_wasting = male and rng.random() < 0.55
            sensory_loss = rng.random() < 0.55
            optic_atrophy = False
            scoliosis = False
            hearing_loss = False
            vocal_fold_palsy = False
            respiratory_failure = False
            cnv_related = False
            cnx_wm_lesion = rng.random() < 0.30  # transient CNS WM lesion history
            x_linked = True
            wheelchair = male and rng.random() < 0.15
            giant_axons = False
            sensory_predominant = False
            focal_myelin = False
        elif gene == "MFN2":
            onset = rng.randint(5, 25)
            demyelinating = False  # axonal
            hnpp_pressure_palsy = False
            foot_drop = rng.random() < 0.85
            hand_wasting = rng.random() < 0.70
            sensory_loss = rng.random() < 0.70
            optic_atrophy = rng.random() < 0.25
            scoliosis = rng.random() < 0.25
            hearing_loss = False
            vocal_fold_palsy = False
            respiratory_failure = False
            cnv_related = False
            cnx_wm_lesion = False
            x_linked = False
            wheelchair = rng.random() < 0.35
            giant_axons = False
            sensory_predominant = False
            focal_myelin = False
        elif gene == "SH3TC2":
            onset = rng.randint(5, 15)
            demyelinating = True
            hnpp_pressure_palsy = False
            foot_drop = rng.random() < 0.90
            hand_wasting = rng.random() < 0.70
            sensory_loss = rng.random() < 0.75
            optic_atrophy = False
            scoliosis = rng.random() < 0.60
            hearing_loss = rng.random() < 0.45
            vocal_fold_palsy = False
            respiratory_failure = False
            cnv_related = False
            cnx_wm_lesion = False
            x_linked = False
            wheelchair = rng.random() < 0.35
            giant_axons = False
            sensory_predominant = False
            focal_myelin = False
        elif gene == "GDAP1":
            onset = rng.randint(1, 8)
            demyelinating = rng.random() < 0.55  # mixed or demyelinating AR
            hnpp_pressure_palsy = False
            foot_drop = rng.random() < 0.90
            hand_wasting = rng.random() < 0.80
            sensory_loss = rng.random() < 0.70
            optic_atrophy = False
            scoliosis = rng.random() < 0.20
            hearing_loss = False
            vocal_fold_palsy = rng.random() < 0.25
            respiratory_failure = rng.random() < 0.15
            cnv_related = False
            cnx_wm_lesion = False
            x_linked = False
            wheelchair = rng.random() < 0.40
            giant_axons = False
            sensory_predominant = False
            focal_myelin = False
        elif gene == "NEFL":
            # AD CMT2E (70%) vs AR CMT1F (30%)
            ar_form = rng.random() < 0.30
            onset = rng.randint(0, 5) if ar_form else rng.randint(15, 35)
            demyelinating = ar_form
            hnpp_pressure_palsy = False
            foot_drop = rng.random() < 0.85
            hand_wasting = rng.random() < 0.65
            sensory_loss = rng.random() < 0.70
            optic_atrophy = False
            scoliosis = ar_form and rng.random() < 0.30
            hearing_loss = False
            vocal_fold_palsy = False
            respiratory_failure = ar_form and rng.random() < 0.10
            cnv_related = False
            cnx_wm_lesion = False
            x_linked = False
            wheelchair = (ar_form and rng.random() < 0.50) or (not ar_form and rng.random() < 0.15)
            giant_axons = True  # pathognomonic
            sensory_predominant = False
            focal_myelin = False
        else:  # PRX — CMT4F
            onset = rng.randint(2, 6)
            demyelinating = True
            hnpp_pressure_palsy = False
            foot_drop = rng.random() < 0.90
            hand_wasting = rng.random() < 0.60
            sensory_loss = True  # sensory > motor
            optic_atrophy = False
            scoliosis = rng.random() < 0.45
            hearing_loss = False
            vocal_fold_palsy = False
            respiratory_failure = False
            cnv_related = False
            cnx_wm_lesion = False
            x_linked = False
            wheelchair = rng.random() < 0.50
            giant_axons = False
            sensory_predominant = True  # key distinguishing feature
            focal_myelin = True  # focally folded myelin

        cohort.append({
            "patient_id": f"{gene}-{pid+1:03d}",
            "onset_age": onset,
            "demyelinating": demyelinating,
            "hnpp_pressure_palsy": hnpp_pressure_palsy,
            "foot_drop": foot_drop,
            "hand_wasting": hand_wasting,
            "sensory_loss": sensory_loss,
            "optic_atrophy": optic_atrophy,
            "scoliosis": scoliosis,
            "hearing_loss": hearing_loss,
            "vocal_fold_palsy": vocal_fold_palsy,
            "respiratory_failure": respiratory_failure,
            "cnv_related": cnv_related,
            "cnx_wm_lesion": cnx_wm_lesion,
            "x_linked": x_linked,
            "wheelchair": wheelchair,
            "giant_axons": giant_axons,
            "sensory_predominant": sensory_predominant,
            "focal_myelin": focal_myelin,
        })
    return cohort


_ALL_COHORTS = {
    g["gene"]: _build_cohort(g, SEED_BASE + i)
    for i, g in enumerate(CMT_GENES)
}


def overview() -> dict:
    """Aggregate overview across all 8 CMT genes (320 patients, seeds 2142-2149)."""
    all_patients = [p for cohort in _ALL_COHORTS.values() for p in cohort]
    n = len(all_patients)

    pmp22_cohort  = _ALL_COHORTS["PMP22"]
    mpz_cohort    = _ALL_COHORTS["MPZ"]
    gjb1_cohort   = _ALL_COHORTS["GJB1"]
    mfn2_cohort   = _ALL_COHORTS["MFN2"]
    sh3tc2_cohort = _ALL_COHORTS["SH3TC2"]
    gdap1_cohort  = _ALL_COHORTS["GDAP1"]
    nefl_cohort   = _ALL_COHORTS["NEFL"]
    prx_cohort    = _ALL_COHORTS["PRX"]

    return {
        "atlas": "Hereditary CMT Atlas — Complete 8-Gene Charcot-Marie-Tooth Disease Reference",
        "genes": [g["gene"] for g in CMT_GENES],
        "total_patients": n,
        "seeds": "2142-2149",
        "seed_base": SEED_BASE,
        # PMP22
        "pmp22_demyelinating_patients": sum(1 for p in pmp22_cohort if p["demyelinating"]),
        "pmp22_hnpp_pressure_palsy_patients": sum(1 for p in pmp22_cohort if p["hnpp_pressure_palsy"]),
        "pmp22_foot_drop_patients": sum(1 for p in pmp22_cohort if p["foot_drop"]),
        # MPZ
        "mpz_demyelinating_patients": sum(1 for p in mpz_cohort if p["demyelinating"]),
        "mpz_scoliosis_patients": sum(1 for p in mpz_cohort if p["scoliosis"]),
        "mpz_respiratory_failure_patients": sum(1 for p in mpz_cohort if p["respiratory_failure"]),
        # GJB1
        "gjb1_cnx_wm_lesion_patients": sum(1 for p in gjb1_cohort if p["cnx_wm_lesion"]),
        "gjb1_x_linked_patients": sum(1 for p in gjb1_cohort if p["x_linked"]),
        # MFN2
        "mfn2_optic_atrophy_patients": sum(1 for p in mfn2_cohort if p["optic_atrophy"]),
        "mfn2_wheelchair_patients": sum(1 for p in mfn2_cohort if p["wheelchair"]),
        # SH3TC2
        "sh3tc2_scoliosis_patients": sum(1 for p in sh3tc2_cohort if p["scoliosis"]),
        "sh3tc2_hearing_loss_patients": sum(1 for p in sh3tc2_cohort if p["hearing_loss"]),
        # GDAP1
        "gdap1_vocal_fold_palsy_patients": sum(1 for p in gdap1_cohort if p["vocal_fold_palsy"]),
        "gdap1_respiratory_failure_patients": sum(1 for p in gdap1_cohort if p["respiratory_failure"]),
        # NEFL
        "nefl_giant_axons_patients": sum(1 for p in nefl_cohort if p["giant_axons"]),
        "nefl_wheelchair_patients": sum(1 for p in nefl_cohort if p["wheelchair"]),
        # PRX
        "prx_focal_myelin_patients": sum(1 for p in prx_cohort if p["focal_myelin"]),
        "prx_sensory_predominant_patients": sum(1 for p in prx_cohort if p["sensory_predominant"]),
        # Cross-atlas
        "all_foot_drop_patients": sum(1 for p in all_patients if p["foot_drop"]),
        "all_demyelinating_patients": sum(1 for p in all_patients if p["demyelinating"]),
        "all_wheelchair_patients": sum(1 for p in all_patients if p["wheelchair"]),
        "all_scoliosis_patients": sum(1 for p in all_patients if p["scoliosis"]),
        "all_sensory_loss_patients": sum(1 for p in all_patients if p["sensory_loss"]),
        "all_vocal_fold_palsy_patients": sum(1 for p in all_patients if p["vocal_fold_palsy"]),
        "all_optic_atrophy_patients": sum(1 for p in all_patients if p["optic_atrophy"]),
        "all_hearing_loss_patients": sum(1 for p in all_patients if p["hearing_loss"]),
    }


def breakdown() -> dict:
    """Per-gene clinical breakdown for all 8 CMT genes."""
    result = {}
    for g in CMT_GENES:
        gene = g["gene"]
        cohort = _ALL_COHORTS[gene]
        n = len(cohort)
        result[gene] = {
            "gene": gene,
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "key_biomarker": g["key_biomarker"],
            "critical_flags": g["critical_flags"],
            "patient_count": n,
            "mean_onset_age": round(sum(p["onset_age"] for p in cohort) / n, 1),
            "demyelinating_pct": round(100 * sum(1 for p in cohort if p["demyelinating"]) / n, 1),
            "hnpp_pressure_palsy_pct": round(100 * sum(1 for p in cohort if p["hnpp_pressure_palsy"]) / n, 1),
            "foot_drop_pct": round(100 * sum(1 for p in cohort if p["foot_drop"]) / n, 1),
            "hand_wasting_pct": round(100 * sum(1 for p in cohort if p["hand_wasting"]) / n, 1),
            "sensory_loss_pct": round(100 * sum(1 for p in cohort if p["sensory_loss"]) / n, 1),
            "optic_atrophy_pct": round(100 * sum(1 for p in cohort if p["optic_atrophy"]) / n, 1),
            "scoliosis_pct": round(100 * sum(1 for p in cohort if p["scoliosis"]) / n, 1),
            "hearing_loss_pct": round(100 * sum(1 for p in cohort if p["hearing_loss"]) / n, 1),
            "vocal_fold_palsy_pct": round(100 * sum(1 for p in cohort if p["vocal_fold_palsy"]) / n, 1),
            "respiratory_failure_pct": round(100 * sum(1 for p in cohort if p["respiratory_failure"]) / n, 1),
            "cnx_wm_lesion_pct": round(100 * sum(1 for p in cohort if p["cnx_wm_lesion"]) / n, 1),
            "wheelchair_pct": round(100 * sum(1 for p in cohort if p["wheelchair"]) / n, 1),
            "giant_axons_pct": round(100 * sum(1 for p in cohort if p["giant_axons"]) / n, 1),
            "sensory_predominant_pct": round(100 * sum(1 for p in cohort if p["sensory_predominant"]) / n, 1),
            "focal_myelin_pct": round(100 * sum(1 for p in cohort if p["focal_myelin"]) / n, 1),
        }
    return result


def definitions() -> dict:
    """Gene definitions, glossary, and surveillance protocols."""
    return {
        "genes": {
            g["gene"]: g["protein"]
            for g in CMT_GENES
        },
        "glossary": {
            "CMT (Charcot-Marie-Tooth Disease / Hereditary Motor Sensory Neuropathy)": (
                "The most common inherited peripheral neuropathy group; prevalence 1:2,500. "
                "Classified by NCS type and inheritance: "
                "CMT1 = demyelinating (NCV <38 m/s); "
                "CMT2 = axonal (normal NCV, reduced CMAP); "
                "CMTX = X-linked (intermediate NCV in males); "
                "CMT4 = AR demyelinating. "
                "Clinical hallmarks: distal muscle wasting, pes cavus, hammertoes, foot drop, "
                "areflexia, variable sensory loss. "
                "Diagnosis: NCS phenotyping → targeted gene panel (include PMP22 MLPA)."
            ),
            "PMP22 Duplication (CMT1A) vs Deletion (HNPP)": (
                "PMP22 gene dosage: 3 copies (1.5-Mb dup) → CMT1A demyelinating; "
                "1 copy (1.5-Mb del) → HNPP (pressure palsies at entrapment sites). "
                "MLPA or array CGH MANDATORY — NGS/WES alone misses copy-number variants. "
                "CMT1A: most common CMT worldwide; uniform NCV <38 m/s all nerves (PATHOGNOMONIC). "
                "HNPP: transient palsies at fibular head, cubital tunnel, carpal tunnel — avoid compression. "
                "Ascorbic acid RCT failed — no disease-modifying therapy."
            ),
            "NCS Classification for CMT Gene Prioritisation": (
                "Step 1 — NCS on index patient: "
                "NCV <38 m/s all nerves → CMT1 (demyelinating): test PMP22 MLPA first → MPZ → GJB1 → SH3TC2 → GDAP1 → NEFL; "
                "NCV normal/mildly reduced, CMAP low → CMT2 (axonal): test MFN2 → NEFL → GDAP1 (AD); "
                "NCV 25-45 m/s (intermediate): GJB1 first (X-linked pattern); "
                "Uniform vs patchy slowing: uniform = CMT1A; patchy = consider CIDP. "
                "Step 2 — inheritance pattern: "
                "AD (no consanguinity, male-to-male): CMT1A/PMP22, CMT1B/MPZ, CMT2A/MFN2; "
                "XLD (no male-to-male): GJB1 first; "
                "AR (consanguineous, severe childhood): SH3TC2, GDAP1, PRX, NEFL AR."
            ),
            "Demyelinating vs Axonal CMT — Key Clinical Differences": (
                "Demyelinating CMT (CMT1, CMT4): "
                "NCV severely reduced (<38 m/s CMT1; <10 m/s CMT4); "
                "onset childhood-adolescence; uniform NCS; pes cavus prominent; scoliosis; "
                "nerve biopsy: onion bulbs (CMT1A, MPZ) or abnormal myelin folding (PRX). "
                "Axonal CMT (CMT2): "
                "NCV near-normal, CMAP reduced; "
                "onset young adult; later onset; more variable; "
                "nerve biopsy: axon loss (rarely needed). "
                "Intermediate CMTX (GJB1): "
                "NCV 25-45 m/s in males; no male-to-male transmission; CNS white matter lesions."
            ),
            "Mandatory First Steps at CMT Diagnosis": (
                "1. NCS: quantify NCV to classify demyelinating/axonal/intermediate + inheritance clue. "
                "2. MLPA (PMP22 duplication/deletion): mandatory BEFORE NGS panel. "
                "3. Family history: male-to-male → excludes XLD; consanguinity → AR first. "
                "4. MLPA negative → gene panel: PMP22, MPZ, GJB1, MFN2, SH3TC2, GDAP1, NEFL, PRX (and others). "
                "5. Optic atrophy → MFN2 priority. "
                "6. Vocal cord palsy → GDAP1 AR, DCTN1 dHMN7B DDx. "
                "7. Scoliosis prominent → SH3TC2 CMT4C, PRX CMT4F. "
                "8. Hearing loss → SH3TC2, consanguineous AR CMT. "
                "9. Transient CNS WM lesions (fever) → GJB1 CMTX1. "
                "10. No diagnosis after panel: consider whole exome/genome."
            ),
            "CMT4 AR Demyelinating Subtypes — Key Distinguishers": (
                "CMT4A/GDAP1: vocal cord palsy 20-30% (LARYNGOSCOPY MANDATORY); mitochondrial fission; N Africa/Spain. "
                "CMT4C/SH3TC2: EARLY SCOLIOSIS 50-70% (PATHOGNOMONIC); hearing loss 40-50%; Turkey/Pakistan. "
                "CMT4F/PRX: FOCALLY FOLDED MYELIN (biopsy PATHOGNOMONIC); sensory > motor; Romani/Pakistani. "
                "All three: AR, severe demyelinating (NCV <15 m/s), childhood onset, no disease-modifying therapy. "
                "Founder mutations aid targeted testing in specific ethnicities."
            ),
            "Surveillance Protocols by Gene": (
                "ALL CMT: NCS baseline + 3-5 yr intervals; physiotherapy; orthotic assessment; genetic counselling. "
                "PMP22 (HNPP): avoid compression — elbow pads, wrist rests, no prolonged crouching. "
                "MPZ (severe congenital): spirometry + scoliosis X-ray 6-monthly in childhood. "
                "GJB1 (CMTX1): MRI brain protocol for CNS WM episodes; maintain hydration during illness. "
                "MFN2 (CMT2A): annual fundoscopy + OCT; visual fields; low vision aids. "
                "SH3TC2 (CMT4C): spinal X-ray 6-monthly; audiogram annually. "
                "GDAP1 (CMT4A): laryngoscopy MANDATORY; spirometry 6-monthly. "
                "NEFL: serum NF-L monitoring; nerve biopsy if diagnostic uncertainty. "
                "PRX (CMT4F): daily skin inspection; protective footwear; balance training."
            ),
        },
        "clinical_pearls": [
            "PMP22-MLPA-FIRST-ALWAYS — NGS alone misses 70% of CMT1A (copy-number variants)",
            "UNIFORM-NCV-<38-m/s-ALL-NERVES-PATHOGNOMONIC-CMT1A — patchy slowing → CIDP",
            "CMT-PLUS-OPTIC-ATROPHY-EQUALS-MFN2-FIRST",
            "NO-MALE-TO-MALE-TRANSMISSION-EQUALS-X-LINKED-GJB1",
            "CNS-WM-LESIONS-TRANSIENT-FEVER-EQUALS-GJB1-CMTX1",
            "EARLY-SCOLIOSIS-PROMINENT-CMT-EQUALS-SH3TC2-CMT4C",
            "VOCAL-CORD-PALSY-AR-CMT-EQUALS-GDAP1-CMT4A",
            "GIANT-AXONS-NERVE-BIOPSY-EQUALS-NEFL",
            "FOCALLY-FOLDED-MYELIN-EQUALS-PRX-CMT4F",
            "SENSORY->MOTOR-AR-DEMYELINATING-EQUALS-PRX-CMT4F",
            "HNPP-PRESSURE-PALSY-TRANSIENT-AT-ENTRAPMENT-EQUALS-PMP22-DELETION",
            "NORTH-AFRICAN-ANCESTRY-AR-CMT-EQUALS-GDAP1-FIRST",
            "ROMANI-GYPSY-ANCESTRY-AR-CMT-EQUALS-PRX-FIRST",
            "TURKISH-PAKISTANI-AR-CMT-EQUALS-SH3TC2-FIRST",
            "LARYNGOSCOPY-MANDATORY-GDAP1-CMT4A-AT-DIAGNOSIS",
            "AUDIOLOGY-MANDATORY-SH3TC2-CMT4C-ANNUALLY",
            "DAILY-SKIN-INSPECTION-PRX-CMT4F-SENSORY-LOSS",
            "ASCORBIC-ACID-FAILED-RCT-CMT1A-NO-DISEASE-MODIFYING-RX-AVAILABLE",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary CMT Atlas Overview ===")
    ov = overview()
    print(f"Total patients: {ov['total_patients']} | Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"All foot drop: {ov['all_foot_drop_patients']} | Demyelinating: {ov['all_demyelinating_patients']}")
    print(f"Wheelchair: {ov['all_wheelchair_patients']} | Scoliosis: {ov['all_scoliosis_patients']}")
    print(f"MFN2 optic atrophy: {ov['mfn2_optic_atrophy_patients']} / 40")
    print(f"SH3TC2 scoliosis: {ov['sh3tc2_scoliosis_patients']} / 40")
    print(f"GJB1 CNS WM lesions: {ov['gjb1_cnx_wm_lesion_patients']} / 40")
    print(f"GDAP1 vocal fold palsy: {ov['gdap1_vocal_fold_palsy_patients']} / 40")
    print(f"NEFL giant axons: {ov['nefl_giant_axons_patients']} / 40")
    print(f"PRX focal myelin: {ov['prx_focal_myelin_patients']} / 40")
    print("\n=== Per-Gene Breakdown (key stats) ===")
    bd = breakdown()
    for gene, data in bd.items():
        print(
            f"{gene}: onset={data['mean_onset_age']}yr | "
            f"demyelinating={data['demyelinating_pct']}% | "
            f"foot_drop={data['foot_drop_pct']}% | "
            f"wheelchair={data['wheelchair_pct']}%"
        )
