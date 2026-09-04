#!/usr/bin/env python3
"""CMT-Atlas — Complete 8-Gene Hereditary Neuropathy / Charcot-Marie-Tooth Atlas
PMP22  (CMT1A/HNPP; AD; gene duplication/deletion; 17p11.2; most common CMT ~70% of CMT1; NCV <15 m/s) ·
MPZ    (CMT1B; AD; myelin protein zero P0; 1q23.3; second most common demyelinating CMT; Dejerine-Sottas if severe) ·
GJB1   (CMTX1; X-linked; connexin 32 Cx32; Xq13.1; males severe intermediate NCV; females mild; CNS WML 50%) ·
MFN2   (CMT2A; AD; mitofusin 2; 1p36.22; most common CMT2 ~20%; axonal; optic atrophy 5-10%) ·
SH3TC2 (CMT4C; AR; SH3+TRP domains; 5q32; most common AR demyelinating CMT; scoliosis 60-70%; CrN VII) ·
GDAP1  (CMT4A; AR; ganglioside-induced differentiation-associated protein 1; 8q21.11; most common AR CMT overall; vocal cord paresis DISTINCTIVE) ·
HSPB1  (CMT2F/dHMN2B; AD; heat shock protein beta-1 HSP27; 7q11.23; axonal; predominantly motor) ·
NEFL   (CMT2E/CMT1F; AD; neurofilament light chain; 8p21.2; NF-L CSF/serum biomarker; variable demyelinating or axonal)
320-patient aggregate cohort (8 × 40, seeds 998–1005)

Hereditary Neuropathy — Key Neurological Principles:
  - CMT CLASSIFICATION: Charcot-Marie-Tooth disease is the most common inherited peripheral neuropathy
    (prevalence ~1:2,500). Classified primarily by nerve conduction velocity (NCV):
    CMT1 = demyelinating (median NCV <38 m/s); CMT2 = axonal (NCV >38 m/s but low CMAP amplitude);
    CMTX = X-linked intermediate; CMT4 = autosomal recessive (AR) forms.
  - PMP22 DUPLICATION: CMT1A is caused by ~1.4 Mb duplication at 17p11.2 containing PMP22 gene.
    Gene dosage effect: 3 copies of PMP22 → peripheral myelin overproduction → instability → demyelination.
    PMP22 DELETION: HNPP (Hereditary Neuropathy with Liability to Pressure Palsies) — episodic focal palsies.
    MLPA/aCGH is the mandatory first-line test for suspected CMT1 (not sequencing).
  - NERVE CONDUCTION VELOCITY CUTOFFS (MEDIAN NERVE):
    <38 m/s → demyelinating (CMT1 group); >38 m/s with low CMAP → axonal (CMT2 group);
    25-45 m/s → intermediate (CMTX1 / GJB1; GDAP1 axonal-intermediate).
  - X-LINKED INHERITANCE (GJB1/CMTX1): Gap junction beta-1 (connexin 32); expressed in myelin sheath
    of peripheral nerves. Males: intermediate NCV (25-40 m/s), significant weakness.
    Females: mild-moderate (NCV 30-45 m/s). CNS white matter lesions on MRI in ~50% (transient, stroke-like).
    Most common X-linked CMT (~10-15% of all CMT globally).
  - AUTOSOMAL RECESSIVE CMT (CMT4): Severe, early-onset (often childhood). Most common AR types:
    CMT4A (GDAP1) — most common AR CMT overall; vocal cord paresis DISTINCTIVE.
    CMT4C (SH3TC2) — most common AR demyelinating CMT; scoliosis 60-70%; cranial nerve VII palsy.
  - TREATMENT LANDSCAPE (2026):
    NO disease-modifying treatment approved for any CMT subtype (2026).
    ASCORBIC ACID (Vitamin C) for CMT1A: NEGATIVE — three large RCTs (CMT-TRIAAL, Austin, others) failed.
    PXT3003 (baclofen+naltrexone+D-sorbitol): Phase 3 RCT in CMT1A (PREMIER trial, ongoing 2026).
    MAINSTAY = AFO (ankle-foot orthoses) for foot drop; physiotherapy; orthopedic surgery for pes cavus.
    SH3TC2 (CMT4C): scoliosis surgery for Cobb angle >40°.
    GDAP1: respiratory monitoring (diaphragm involvement); ENT for vocal cord assessment.

COHORT: 8 × 40 = 320 patient slots (seeds 998–1005; gene-specific seeds)
"""

import random

SEED_BASE = 998

CMT_GENES = [
    # ── PMP22 — CMT1A / HNPP ────────────────────────────────────────────
    {
        "gene": "PMP22", "protein": "Peripheral Myelin Protein 22",
        "alias": "CMT1A/HNPP (OMIM CMT1A #118220, HNPP #162500); AD; gene duplication (CMT1A) or deletion (HNPP); most common CMT (~70% of CMT1); NCV <15 m/s; MLPA mandatory; 17p11.2",
        "aa": "160 aa", "kDa": "22 kDa",
        "gene_class": (
            "Peripheral myelin protein 22; tetraspan integral membrane protein expressed in "
            "Schwann cells during active myelination. PMP22 constitutes ~5% of total PNS myelin protein. "
            "Functions in myelin compaction, Schwann cell proliferation and apoptosis. "
            "GENE DOSAGE MECHANISM: "
            "CMT1A (~1.4 Mb duplication at 17p11.2 → 3 copies of PMP22) — overexpression → "
            "dysmyelination; uniformly slow NCV (5-15 m/s); length-dependent neuropathy. "
            "HNPP (deletion of same region → 1 copy) — haploinsufficiency → focally slow NCV at "
            "compression sites (median at carpal tunnel, peroneal at fibula head). "
            "Point mutations cause CMT1E or Dejerine-Sottas (severe). "
            "MLPA (Multiplex Ligation-dependent Probe Amplification) or aCGH mandatory for "
            "copy-number detection — sequencing misses duplications/deletions. "
            "17p11.2; OMIM gene 601097."
        ),
        "neuropathy_group": "CMT1 — Demyelinating AD CMT",
        "subtype": "CMT1A/HNPP — PMP22 Duplication/Deletion",
        "locus": "17p11.2", "omim_gene": 601097, "omim_disease": 118220,
        "inheritance": "Autosomal Dominant (AD). ~1.4 Mb duplication → CMT1A (3 copies PMP22). Deletion → HNPP. De novo duplication rate ~10%. Point mutations: CMT1E (rare).",
        "seed_offset": 0,
        "onset_range_y": (0.0, 30.0),
        "gender": "both",
        "severity_weights": [0.30, 0.45, 0.25],
        "nvc_range": (5, 30),
        "neuropathy_type": "demyelinating",
        "pes_cavus_prob": 0.75,
        "ankle_reflex_prob": 0.90,
        "scoliosis_prob": 0.10,
        "vocal_cord_prob": 0.0,
        "optic_atrophy_prob": 0.0,
        "cns_lesion_prob": 0.0,
        "phenotype": (
            "CMT1A (duplication): UNIFORM DEMYELINATING NEUROPATHY — NCV 5-15 m/s in all nerves "
            "(uniform slowing is key DDx from CIDP which is patchy). Onset: typically first-second decade. "
            "Foot deformity (pes cavus + hammer toes): most common presenting sign. "
            "Distal lower limb weakness → foot drop → steppage gait. "
            "Absent ankle reflexes (early); absent knee reflexes later. "
            "Intrinsic hand muscle wasting (later). Mild-moderate functional disability. "
            "Sensory: reduced vibration/proprioception in feet; pain usually absent. "
            "Onion bulb formation on nerve biopsy (Schwann cell proliferation around demyelinated axons). "
            "HNPP (deletion): EPISODIC FOCAL PALSIES at common compression sites — "
            "carpal tunnel syndrome, peroneal palsy (foot drop after leg crossing), "
            "ulnar neuropathy at elbow. Between episodes: mild generalised neuropathy. "
            "NCV: focal slowing at compression sites; background mild diffuse slowing. "
            "May be misdiagnosed as mononeuritis multiplex or multifocal neuropathy. "
            "ASCORBIC ACID TRIAL: NEGATIVE — three large RCTs showed no benefit (CMT-TRIAAL 2011, "
            "NACPMS 2011, others). Not recommended. PXT3003 Phase 3 ongoing (2026)."
        ),
        "disease": (
            "CMT1A: most common inherited peripheral neuropathy worldwide. Prevalence ~1:5,000. "
            "~50% of all CMT; ~70% of CMT1. Uniform NCV slowing distinguishes from acquired demyelinating neuropathies. "
            "HNPP: prevalence ~1:50,000 (underdiagnosed). "
            "Management: AFO for foot drop; occupational therapy; physiotherapy. "
            "Orthopedic surgery for severe pes cavus (plantar fasciotomy, calcaneal osteotomy). "
            "Genetic counselling: 50% risk to offspring (AD); de novo rate ~10% for duplications. "
            "DRUGS TO AVOID: vincristine (causes acute severe neuropathy in CMT1 patients); "
            "taxanes, cisplatin, amiodarone, statins (may worsen); nitrofurantoin."
        ),
        "treatment_options": [
            "AFO (ankle-foot orthoses) — mainstay for foot drop",
            "Physiotherapy (strength + balance + gait training)",
            "Occupational therapy + hand splints (intrinsic weakness)",
            "Orthopedic surgery (pes cavus — plantar fasciotomy / calcaneal osteotomy)",
            "Ascorbic acid trial (NOT recommended — negative RCTs)",
        ],
        "outcome_options": [
            "Stable — mild disability; AFO + physiotherapy; working into 5th decade",
            "Progressive — moderate weakness; foot drop; AFO dependent by 30s",
            "Progressive — pes cavus surgery + AFO; hand weakness late",
            "Mild — small pes cavus; NCV slow but minimal functional impact",
            "Progressive — severe foot drop + hand weakness; wheelchair 5th–6th decade",
        ],
    },

    # ── MPZ — CMT1B ─────────────────────────────────────────────────────
    {
        "gene": "MPZ", "protein": "Myelin Protein Zero (P0)",
        "alias": "CMT1B (OMIM #118200); AD; myelin protein zero P0; major structural protein of PNS myelin; NCV <38 m/s; Dejerine-Sottas if severe infant onset; 1q23.3",
        "aa": "248 aa", "kDa": "28 kDa",
        "gene_class": (
            "Myelin protein zero (MPZ/P0); type I transmembrane glycoprotein; largest single component "
            "of PNS compact myelin (~50% of total PNS myelin protein). "
            "Adhesion molecule: P0 homotypic interactions between adjacent myelin lamellae → "
            "compaction of myelin sheath. P0 extracellular Ig-like domain mediates homophilic binding. "
            "Heterozygous point mutations → dominant-negative effect OR haploinsufficiency → "
            "dysmyelination (demyelinating CMT1B) OR axonal loss (axonal CMT2I/CMT2J — late onset). "
            "Homozygous or compound het mutations → Dejerine-Sottas (severe childhood onset). "
            "Over 150 pathogenic mutations catalogued. "
            "Clinical spectrum: NCV <10 m/s (severe neonatal) to >38 m/s (late-onset axonal). "
            "1q23.3; OMIM gene 159440."
        ),
        "neuropathy_group": "CMT1 — Demyelinating AD CMT",
        "subtype": "CMT1B — Myelin Protein Zero",
        "locus": "1q23.3", "omim_gene": 159440, "omim_disease": 118200,
        "inheritance": "Autosomal Dominant (AD). Heterozygous point mutations (>150 variants). De novo mutations account for ~5-10%. Biallelic: Dejerine-Sottas or CHN (congenital hypomyelination neuropathy).",
        "seed_offset": 1,
        "onset_range_y": (0.0, 40.0),
        "gender": "both",
        "severity_weights": [0.20, 0.45, 0.35],
        "nvc_range": (15, 35),
        "neuropathy_type": "demyelinating",
        "pes_cavus_prob": 0.70,
        "ankle_reflex_prob": 0.85,
        "scoliosis_prob": 0.15,
        "vocal_cord_prob": 0.0,
        "optic_atrophy_prob": 0.0,
        "cns_lesion_prob": 0.0,
        "phenotype": (
            "WIDE PHENOTYPIC SPECTRUM depending on mutation class: "
            "EARLY ONSET (demyelinating CMT1B): childhood onset; NCV 10-30 m/s; severe disability earlier. "
            "LATE ONSET (axonal CMT2I/2J): onset 30-60 years; NCV >38 m/s but axonal loss; "
            "slower progression. "
            "DEJERINE-SOTTAS SYNDROME (DSS, biallelic or de novo dominant): severe infantile onset; "
            "NCV <10 m/s; profound weakness; scoliosis; may be wheelchair-bound in childhood; "
            "Onion bulbs on biopsy. "
            "CLASSIC CMT1B: gait ataxia, foot deformity (pes cavus), absent ankle reflexes, "
            "distal muscle wasting. Hands involved later. Sensory loss. "
            "Some mutations cause prominent PUPILLARY ABNORMALITIES (anisocoria) — distinctive for MPZ. "
            "NERVE BIOPSY: onion bulbs (repeated demyelination/remyelination) + hypomyelination "
            "in severe cases. "
            "DDx PMP22 duplication: MPZ mutations often cause more variable NCV (some late-onset); "
            "require sequencing (MLPA negative → sequence MPZ)."
        ),
        "disease": (
            "CMT1B: second most common demyelinating CMT after CMT1A. Prevalence ~1:50,000 (estimated). "
            "Responsible for 10-15% of dominant demyelinating CMT after PMP22 duplication excluded. "
            "Genetic testing: after MLPA negative (no PMP22 duplication), sequence MPZ next. "
            "Management: same as CMT1A — AFO, physiotherapy, occupational therapy. "
            "No disease-modifying treatment. "
            "Drugs to avoid: same as CMT1 list (vincristine, taxanes, amiodarone)."
        ),
        "treatment_options": [
            "AFO (ankle-foot orthoses) for foot drop — mainstay",
            "Physiotherapy (gait, balance, strength training)",
            "Occupational therapy + adaptive aids",
            "Orthopedic surgery (pes cavus if severe — Charcot joint risk)",
            "No disease-modifying Rx for CMT1B (2026)",
        ],
        "outcome_options": [
            "Progressive — moderate-severe; wheelchair 4th–5th decade (severe MPZ)",
            "Progressive — childhood onset DSS; severe disability; scoliosis surgery",
            "Moderate — classic CMT1B; AFO-dependent; hand involvement late",
            "Mild — late-onset axonal MPZ variant; walking maintained into 6th decade",
            "Progressive — intermediate severity; foot drop + hand weakness 5th decade",
        ],
    },

    # ── GJB1 — CMTX1 ─────────────────────────────────────────────────────
    {
        "gene": "GJB1", "protein": "Connexin 32 (Cx32)",
        "alias": "CMTX1 (X-linked Charcot-Marie-Tooth Type 1; OMIM #302800); X-linked; connexin 32 gap junction; Xq13.1; males severe intermediate NCV 25-40; females mild 30-45; CNS white matter lesions 50%; most common X-linked CMT (~10-15% of all CMT)",
        "aa": "283 aa", "kDa": "32 kDa",
        "gene_class": (
            "Connexin 32 (GJB1); gap junction protein beta-1; forms hexameric hemichannels (connexons) "
            "in paranodal loops and Schmidt-Lanterman incisures of Schwann cell myelin. "
            "GJB1 gap junctions allow rapid radial diffusion of small molecules (metabolites, ions) "
            "through the myelin sheath — essential for myelin maintenance and Schwann cell communication. "
            "X-LINKED INHERITANCE: hemizygous males → severe; heterozygous females → mild-moderate "
            "(lyonisation = variable X-inactivation → variable severity in females). "
            "CNS INVOLVEMENT: GJB1 also expressed in CNS oligodendrocytes → "
            "CNS white matter lesions (periventricular T2 hyperintensities) in ~50% of males; "
            "transient stroke-like episodes or encephalopathy have been reported (rare but important). "
            "INTERMEDIATE NCVs: CMTX1 males have median NCV 25-40 m/s (not as slow as CMT1A; "
            "not normal as CMT2) — intermediate range. Females: 30-45 m/s. "
            "Xq13.1; OMIM gene 304040."
        ),
        "neuropathy_group": "CMTX — X-Linked Intermediate CMT",
        "subtype": "CMTX1 — Connexin 32 Gap Junction Neuropathy",
        "locus": "Xq13.1", "omim_gene": 304040, "omim_disease": 302800,
        "inheritance": "X-Linked (XL). Hemizygous males: severe. Heterozygous females: mild-moderate (lyonisation). No male-to-male transmission. De novo mutations: ~10-15%.",
        "seed_offset": 2,
        "onset_range_y": (10.0, 50.0),
        "gender": "xlinkd",  # custom: males earlier/severe, females later/milder
        "severity_weights": [0.20, 0.45, 0.35],
        "nvc_range": (25, 45),
        "neuropathy_type": "intermediate",
        "pes_cavus_prob": 0.60,
        "ankle_reflex_prob": 0.75,
        "scoliosis_prob": 0.05,
        "vocal_cord_prob": 0.0,
        "optic_atrophy_prob": 0.0,
        "cns_lesion_prob": 0.50,  # 50% of males; females less
        "phenotype": (
            "X-LINKED INHERITANCE PATTERN: no male-to-male transmission (key clinical clue). "
            "MALES: onset 10-30 years; progressive distal weakness + wasting; intermediate NCV (25-40 m/s); "
            "absent ankle reflexes; sensory loss (especially vibration); significant functional disability. "
            "FEMALES: onset 20-50 years; mild-moderate neuropathy; NCV 30-45 m/s (intermediate); "
            "often diagnosed only during family screening; may be asymptomatic. "
            "CNS WHITE MATTER LESIONS: on MRI in ~50% of males; "
            "periventricular/subcortical T2/FLAIR hyperintensities; "
            "may present as transient encephalopathy (stroke-like episodes — rare); "
            "distinct from PML or MS (usually clinically silent). "
            "EMG/NCS KEY FEATURES: intermediate NCV (25-45 m/s); males not as slow as CMT1A; "
            "mixed features (some axonal + some demyelinating characteristics). "
            "DIFFERENTIAL: CMT1A (much slower NCV; no CNS lesions; AD pedigree); "
            "CMT2A (faster NCV; no sex difference; AD). "
            "Pes cavus, hammer toes, distal wasting — same classic triad as other CMT types."
        ),
        "disease": (
            "CMTX1: most common X-linked inherited neuropathy. ~10-15% of all CMT cases. "
            "Second most common cause of CMT after CMT1A (PMP22 duplication). "
            "Over 400 pathogenic GJB1 variants catalogued. "
            "Diagnosis: intermediate NCV in male + absent ankle reflexes + X-linked pedigree → "
            "immediate GJB1 sequencing. Females: MLPA negative → GJB1. "
            "CNS lesions: usually do not require treatment; rarely cause episodic encephalopathy. "
            "Management: AFO, physiotherapy, occupational therapy. "
            "No disease-modifying treatment. Genetic counselling: daughters are obligate carriers."
        ),
        "treatment_options": [
            "AFO (ankle-foot orthoses) — primary management",
            "Physiotherapy + occupational therapy",
            "MRI brain if encephalopathic episodes (CNS WML monitoring)",
            "Orthopedic surgery (pes cavus if structurally severe)",
            "No disease-modifying Rx for CMTX1 (2026)",
        ],
        "outcome_options": [
            "Progressive (male) — significant weakness; AFO; walking limited 4th–5th decade",
            "Moderate (male) — intermediate disability; working with aids",
            "Mild (female) — asymptomatic/mild; diagnosed on family screening",
            "Moderate (female) — symptomatic; AFO needed; mild weakness",
            "Progressive (male) — CNS lesions + neuropathy; episodic encephalopathy (rare)",
        ],
    },

    # ── MFN2 — CMT2A ─────────────────────────────────────────────────────
    {
        "gene": "MFN2", "protein": "Mitofusin 2",
        "alias": "CMT2A (OMIM #609260); AD; mitofusin 2; most common CMT2 (~20% of CMT2); axonal NCV >38 m/s low CMAP; optic atrophy 5-10%; mitochondrial fusion; 1p36.22",
        "aa": "741 aa", "kDa": "84 kDa",
        "gene_class": (
            "Mitofusin 2 (MFN2); integral outer mitochondrial membrane GTPase; mediates "
            "mitochondrial outer membrane fusion (with MFN1). "
            "Functions: mitochondrial network maintenance, distribution along axons, "
            "mitochondria-ER contact sites (MERC — calcium homeostasis, lipid transfer). "
            "MFN2 is critical in long peripheral axons (up to 1 m in large motor neurons) "
            "where mitochondrial transport and fusion are essential for energy supply. "
            "MFN2 loss → mitochondrial fragmentation → axonal degeneration → CMT2A. "
            "OPTIC ATROPHY: MFN2 variants cause optic nerve axonal degeneration in 5-10% "
            "(RGC axons are extremely long — similar vulnerability; overlap with CMT2A+DOA). "
            "EARLY SEVERE FORM: some compound het or de novo variants → severe childhood onset "
            "with rapid progression. "
            "1p36.22; OMIM gene 608507."
        ),
        "neuropathy_group": "CMT2 — Axonal AD CMT",
        "subtype": "CMT2A — Mitofusin 2 Axonal Neuropathy",
        "locus": "1p36.22", "omim_gene": 608507, "omim_disease": 609260,
        "inheritance": "Autosomal Dominant (AD). De novo mutations frequent (~50% in severe early-onset). Compound het rare (AR severe form). Wide phenotypic range: mild adult onset to severe childhood onset.",
        "seed_offset": 3,
        "onset_range_y": (5.0, 30.0),
        "gender": "both",
        "severity_weights": [0.15, 0.40, 0.45],
        "nvc_range": (38, 55),
        "neuropathy_type": "axonal",
        "pes_cavus_prob": 0.65,
        "ankle_reflex_prob": 0.80,
        "scoliosis_prob": 0.15,
        "vocal_cord_prob": 0.0,
        "optic_atrophy_prob": 0.08,  # 5-10%
        "cns_lesion_prob": 0.0,
        "phenotype": (
            "AXONAL NEUROPATHY: NCV >38 m/s (often normal or near-normal velocity); "
            "CMAP amplitude markedly reduced (axonal loss). "
            "EMG: denervation (fibrillations, positive sharp waves); reduced motor unit recruitment. "
            "ONSET: typically early childhood to young adult (often earlier and more severe than CMT1A). "
            "Progressive distal weakness → foot drop → proximal spread (hip girdle in severe cases). "
            "Absent ankle and knee reflexes. Distal wasting. Pes cavus (may be less pronounced than CMT1). "
            "OPTIC ATROPHY: 5-10% of MFN2 patients develop optic neuropathy "
            "(visual loss, pale discs, reduced visual acuity) — overlap with autosomal dominant "
            "optic atrophy (OPA1). CMT2A + optic atrophy = phenotype strongly suggests MFN2. "
            "EARLY SEVERE FORM (de novo / compound het): childhood-onset; pyramidal features; "
            "cerebellar involvement possible; rapid progression. "
            "NERVE BIOPSY: axonal loss (not onion bulbs as in demyelinating CMT). "
            "DDx other CMT2: MFN2 has early onset + optic atrophy + dominant; "
            "HSPB1/NEFL = later onset, motor-predominant."
        ),
        "disease": (
            "CMT2A: most common CMT2. ~20% of axonal CMT cases. "
            "Wide geographic distribution; no founder effect. "
            "Genetic diagnosis: after normal/near-normal NCV with low CMAP, sequence MFN2 first "
            "(most common axonal CMT). "
            "Ophthalmology: annual review recommended given optic atrophy risk. "
            "No disease-modifying treatment. "
            "Mitochondrial pathway: research target — mitochondrial fusion enhancers in preclinical stage."
        ),
        "treatment_options": [
            "AFO (ankle-foot orthoses) for foot drop",
            "Physiotherapy (progressive axonal neuropathy — early and sustained intervention)",
            "Ophthalmology follow-up (optic atrophy monitoring)",
            "Occupational therapy + hand splints (hand involvement)",
            "No disease-modifying Rx for CMT2A (2026)",
        ],
        "outcome_options": [
            "Progressive — severe early onset; wheelchair 2nd–3rd decade",
            "Progressive — childhood onset; foot drop + proximal weakness; AFO",
            "Moderate — adult onset; slower progression; walking with aids into 5th decade",
            "Progressive — optic atrophy + neuropathy; ophthalmology + AFO management",
            "Severe — de novo mutation; rapid progression; spinal cord involvement (rare)",
        ],
    },

    # ── SH3TC2 — CMT4C ───────────────────────────────────────────────────
    {
        "gene": "SH3TC2", "protein": "SH3 Domain and Tetratricopeptide Repeats 2",
        "alias": "CMT4C (OMIM #601596); AR; most common AR demyelinating CMT; severe scoliosis 60-70%; cranial nerve VII palsy; early childhood onset; Schwann cell lamellipodia defect; 5q32",
        "aa": "1288 aa", "kDa": "144 kDa",
        "gene_class": (
            "SH3TC2 (SH3 domain-containing protein 2); contains N-terminal SH3 domain and "
            "multiple tetratricopeptide repeat (TPR) motifs. Localises to Schwann cell plasma "
            "membrane, particularly lamellipodia at the tips of myelin sheaths. "
            "Functions in Schwann cell scaffolding and myelination maintenance. "
            "Interacts with Rab11 (endosomal recycling pathway). "
            "AR inheritance: biallelic loss-of-function (frameshift, nonsense, missense, splice). "
            "MOST COMMON AR DEMYELINATING CMT globally (CMT4C). "
            "SCOLIOSIS: 60-70% — more prevalent and severe than any other CMT type; "
            "likely due to trunk muscle involvement and early axial muscle weakness. "
            "CRANIAL NERVE VII PALSY: facial weakness in ~30% (distinguishes CMT4C from CMT1A). "
            "HEARING LOSS: sensorineural hearing loss (SNHL) in some patients. "
            "5q32; OMIM gene 608206."
        ),
        "neuropathy_group": "CMT4 — Demyelinating AR CMT",
        "subtype": "CMT4C — SH3TC2 Schwann Cell Scaffolding Neuropathy",
        "locus": "5q32", "omim_gene": 608206, "omim_disease": 601596,
        "inheritance": "Autosomal Recessive (AR). Biallelic loss-of-function mutations. Founder mutations in Roma (p.Arg954Stop) and other European populations. Carrier frequency higher in certain Eastern European communities.",
        "seed_offset": 4,
        "onset_range_y": (3.0, 15.0),
        "gender": "both",
        "severity_weights": [0.10, 0.35, 0.55],
        "nvc_range": (5, 25),
        "neuropathy_type": "demyelinating",
        "pes_cavus_prob": 0.80,
        "ankle_reflex_prob": 0.95,
        "scoliosis_prob": 0.65,  # 60-70%
        "vocal_cord_prob": 0.0,
        "optic_atrophy_prob": 0.0,
        "cns_lesion_prob": 0.0,
        "phenotype": (
            "MOST COMMON AR DEMYELINATING CMT. Early childhood onset (3-15 years). "
            "SEVERELY SLOW NCV: often <15 m/s (some of the slowest NCVs in all CMT types). "
            "Progressive lower limb weakness (foot drop early); distal wasting; "
            "absent ankle + knee reflexes; pes cavus (>80%). "
            "SCOLIOSIS (60-70%): progressive idiopathic-pattern scoliosis; often requires "
            "orthopedic surgery; may precede recognizable neuropathy. "
            "Cobb angle >40° → surgical indication. Respiratory restriction if severe. "
            "CRANIAL NERVE VII PALSY: facial weakness (~30%) — important clinical sign; "
            "not seen in most CMT subtypes; helps identify CMT4C. "
            "CRANIAL NERVE VIII: sensorineural hearing loss (SNHL) in some patients "
            "(hearing aids may be needed). "
            "NERVE BIOPSY: severe hypomyelination, onion bulbs, Schwann cell "
            "lamellipodia abnormalities. "
            "HAND INVOLVEMENT: later in course; upper limb weakness and wasting 2nd decade. "
            "Scoliosis + neuropathy in child → AR CMT → SH3TC2 sequencing mandatory."
        ),
        "disease": (
            "CMT4C: most common AR demyelinating CMT in European and Roma populations. "
            "Prevalence ~1:50,000 (estimated). "
            "Roma founder mutation p.Arg954Stop: highly prevalent in Romani people "
            "(heterozygous carrier rate 1-2% in some Roma communities). "
            "Diagnosis: biallelic mutations in SH3TC2; homozygous p.Arg954Stop in Roma. "
            "Scoliosis management: critical — early physiotherapy; surgery if Cobb >40°. "
            "Hearing: audiometry recommended; hearing aids if SNHL present. "
            "Respiratory: spirometry if severe scoliosis (risk of restrictive lung disease). "
            "No disease-modifying treatment."
        ),
        "treatment_options": [
            "Scoliosis surgery (Cobb angle >40° — posterior spinal fusion)",
            "Physiotherapy — intensive from early childhood",
            "AFO (ankle-foot orthoses) — foot drop; may need KAFO",
            "Hearing aids (if SNHL detected on audiometry)",
            "No disease-modifying Rx for CMT4C (2026)",
        ],
        "outcome_options": [
            "Progressive — severe scoliosis requiring surgery + wheelchair by 2nd decade",
            "Progressive — moderate scoliosis + profound foot drop; KAFO + AFO",
            "Progressive — facial palsy + neuropathy + scoliosis; multidisciplinary",
            "Severe — rapid progression; respiratory compromise from scoliosis",
            "Moderate — scoliosis managed surgically; ambulatory with aids into adulthood",
        ],
    },

    # ── GDAP1 — CMT4A ─────────────────────────────────────────────────────
    {
        "gene": "GDAP1", "protein": "Ganglioside-Induced Differentiation-Associated Protein 1",
        "alias": "CMT4A (OMIM #214400); AR; most common AR CMT overall; VOCAL CORD PARESIS DISTINCTIVE; diaphragm involvement; axonal OR intermediate; mitochondrial fission; 8q21.11",
        "aa": "358 aa", "kDa": "40 kDa",
        "gene_class": (
            "GDAP1; tail-anchored outer mitochondrial membrane protein; member of glutathione "
            "S-transferase (GST) superfamily (but lacks transferase activity). "
            "Critical role in mitochondrial fission: GDAP1 recruits dynamin-related proteins "
            "(DRP1/DRP2) → mitochondrial fragmentation → enables mitochondrial distribution along axons. "
            "Interacts with DRP1, FIS1 (fission 1). "
            "Expressed highly in Schwann cells AND neurons (dual expression — both demyelinating "
            "and axonal forms possible). "
            "AR inheritance: biallelic loss-of-function (nonsense, missense, splice site). "
            "DOMINANT form (AD GDAP1, rare): axonal CMT2K. "
            "MOST COMMON AR CMT overall — includes axonal, demyelinating, and intermediate forms. "
            "VOCAL CORD PARESIS: DISTINCTIVE feature of GDAP1 neuropathy "
            "(not seen in CMT1A/CMT1B/CMTX — strongly suggests GDAP1 when present). "
            "DIAPHRAGM: respiratory involvement (diaphragmatic weakness) — respiratory monitoring mandatory. "
            "8q21.11; OMIM gene 606598."
        ),
        "neuropathy_group": "CMT4 — AR CMT (Axonal/Intermediate)",
        "subtype": "CMT4A — GDAP1 Mitochondrial Fission Neuropathy",
        "locus": "8q21.11", "omim_gene": 606598, "omim_disease": 214400,
        "inheritance": "Autosomal Recessive (AR). Biallelic loss-of-function mutations. Axonal or intermediate NCV depending on mutation. AD form (CMT2K) — rare dominant negative.",
        "seed_offset": 5,
        "onset_range_y": (2.0, 20.0),
        "gender": "both",
        "severity_weights": [0.10, 0.35, 0.55],
        "nvc_range": (15, 45),
        "neuropathy_type": "intermediate",  # can be axonal or demyelinating
        "pes_cavus_prob": 0.70,
        "ankle_reflex_prob": 0.90,
        "scoliosis_prob": 0.20,
        "vocal_cord_prob": 0.30,  # distinctive feature
        "optic_atrophy_prob": 0.0,
        "cns_lesion_prob": 0.0,
        "phenotype": (
            "MOST COMMON AR CMT OVERALL (CMT4A). Very early onset (often 2-15 years). "
            "VARIABLE NEUROPATHY TYPE: axonal (NCV >38, low CMAP) OR demyelinating "
            "(NCV <38, low CMAP + NCV slowing) OR intermediate — depends on mutation. "
            "Progressive distal weakness from lower limbs; foot drop; proximal spread. "
            "Pes cavus (very common); absent ankle reflexes. "
            "VOCAL CORD PARESIS — DISTINCTIVE AND PATHOGNOMONIC for GDAP1: "
            "  - Recurrent laryngeal nerve involvement → hoarse voice, stridor, "
            "    dysphagia, aspiration risk. "
            "  - ENT assessment mandatory; laryngoscopy; tracheostomy in severe cases. "
            "DIAPHRAGM INVOLVEMENT: respiratory muscle weakness → reduced vital capacity; "
            "nocturnal hypoventilation; respiratory failure in severe cases. "
            "Pulmonary function tests annual. NIV (bilevel) if needed. "
            "SCOLIOSIS: ~20% (less than CMT4C but present). "
            "HAND INVOLVEMENT: early (upper limbs involved earlier than CMT1). "
            "Severe cases may be wheelchair dependent in childhood."
        ),
        "disease": (
            "CMT4A: most common AR inherited neuropathy overall. "
            "Especially prevalent in Mediterranean and Middle Eastern populations. "
            "North African founder mutations common. "
            "Diagnosis: AR neuropathy + vocal cord involvement → GDAP1 sequencing first. "
            "Respiratory monitoring: annual spirometry; polysomnography if NIV consideration. "
            "ENT: laryngoscopy (vocal cord); speech therapy (aspiration risk). "
            "No disease-modifying treatment. Mitochondrial fission pathway — research target."
        ),
        "treatment_options": [
            "Respiratory monitoring + NIV (if diaphragm/nocturnal hypoventilation)",
            "Vocal cord assessment — ENT laryngoscopy + speech therapy",
            "AFO (ankle-foot orthoses) + physiotherapy (early, intensive)",
            "Occupational therapy (early upper limb involvement)",
            "No disease-modifying Rx for CMT4A (2026)",
        ],
        "outcome_options": [
            "Progressive — vocal cord paresis + diaphragm weakness; NIV + AFO",
            "Severe — early onset; wheelchair 1st–2nd decade; respiratory support",
            "Progressive — foot drop + hand weakness; vocal hoarseness; ENT follow-up",
            "Moderate — intermediate NCV; ambulatory with aids; vocal cord stable",
            "Progressive — scoliosis + neuropathy + respiratory compromise; multidisciplinary",
        ],
    },

    # ── HSPB1 — CMT2F / dHMN2B ──────────────────────────────────────────
    {
        "gene": "HSPB1", "protein": "Heat Shock Protein Beta-1 (HSP27)",
        "alias": "CMT2F/dHMN2B (OMIM CMT2F #606595, dHMN2B #608634); AD; heat shock protein beta-1 HSP27; 7q11.23; axonal; predominantly motor (distal HMN); chaperone; intracytoplasmic inclusions",
        "aa": "205 aa", "kDa": "27 kDa",
        "gene_class": (
            "Heat shock protein beta-1 (HSPB1/HSP27); small heat shock protein; ATP-independent "
            "molecular chaperone. Oligomerises into large complexes (up to 24-mers). "
            "Functions: client protein stabilisation under stress, neurofilament assembly regulation, "
            "inhibition of apoptosis (interaction with cytochrome c), actin cytoskeleton stabilisation. "
            "HSP27 regulates neurofilament (NF-L/NF-M/NF-H) assembly — key for axonal integrity. "
            "GAIN-OF-FUNCTION (dominant-negative) mutations → disrupted neurofilament assembly → "
            "intracytoplasmic inclusions → axonal transport failure → distal motor axon degeneration. "
            "PHENOTYPE SPECTRUM: "
            "CMT2F: sensorimotor axonal neuropathy (both sensory and motor affected). "
            "dHMN2B: distal hereditary motor neuropathy — predominantly or purely MOTOR; "
            "sensory loss minimal/absent. "
            "ADULT ONSET: typically 20-50 years (later than most other CMT genes). "
            "7q11.23; OMIM gene 602195."
        ),
        "neuropathy_group": "CMT2 — Axonal AD CMT / Distal HMN",
        "subtype": "CMT2F/dHMN2B — HSP27 Axonal Neuropathy",
        "locus": "7q11.23", "omim_gene": 602195, "omim_disease": 606595,
        "inheritance": "Autosomal Dominant (AD). Gain-of-function (dominant-negative) point mutations. De novo mutations occur. Adult-onset (typically 20-50 years).",
        "seed_offset": 6,
        "onset_range_y": (20.0, 50.0),
        "gender": "both",
        "severity_weights": [0.30, 0.45, 0.25],
        "nvc_range": (38, 55),
        "neuropathy_type": "axonal",
        "pes_cavus_prob": 0.55,
        "ankle_reflex_prob": 0.70,
        "scoliosis_prob": 0.05,
        "vocal_cord_prob": 0.0,
        "optic_atrophy_prob": 0.0,
        "cns_lesion_prob": 0.0,
        "phenotype": (
            "ADULT-ONSET AXONAL NEUROPATHY (CMT2F/dHMN2B). Onset typically 20-50 years. "
            "PREDOMINANTLY MOTOR: in dHMN2B variant — distal lower limb weakness (foot drop); "
            "minimal sensory loss (distinguishes from most CMT2 subtypes). "
            "In CMT2F: sensorimotor axonal neuropathy (both motor + sensory). "
            "NCV: normal or near-normal (>38 m/s); CMAP amplitude reduced (axonal loss); "
            "SNAP normal or mildly reduced (dHMN) vs reduced (CMT2F). "
            "LOWER LIMBS: peroneal muscle wasting → foot drop → steppage gait; "
            "absent ankle reflexes; pes cavus (less pronounced than CMT1). "
            "UPPER LIMBS: intrinsic hand muscle wasting; grip weakness (later). "
            "INTRACYTOPLASMIC INCLUSIONS: on sural nerve biopsy — HSP27 aggregates in axons. "
            "PROGNOSIS: slowly progressive; most patients ambulatory into 5th-6th decade. "
            "DDx: progressive muscular atrophy (no sensory), ALS (UMN signs), "
            "Kennedy disease (androgen receptor — sensory + gynecomastia + bulbar). "
            "NF-L serum: may be elevated (axonal loss biomarker)."
        ),
        "disease": (
            "HSPB1 neuropathy: rare (exact prevalence unknown; estimated ~1-2% of CMT2). "
            "CMT2F and dHMN2B overlap in phenotype; same gene, different emphasis. "
            "Diagnosis: axonal neuropathy + predominantly motor features + AD family history "
            "in adult → include HSPB1 in panel. "
            "No disease-modifying treatment. "
            "HSP27 modulation: research interest (heat shock response upregulation — preclinical). "
            "Management: AFO for foot drop; physiotherapy; occupational therapy."
        ),
        "treatment_options": [
            "AFO (ankle-foot orthoses) for foot drop — mainstay",
            "Physiotherapy + occupational therapy (upper and lower limb)",
            "Hand splints + assistive devices (intrinsic weakness)",
            "Gait training + balance exercises",
            "No disease-modifying Rx for CMT2F/dHMN2B (2026)",
        ],
        "outcome_options": [
            "Slow progression — ambulatory into 5th-6th decade; mild AFO dependence",
            "Moderate — foot drop + hand weakness; AFO + occupational aids",
            "Progressive — significant distal wasting; AFO + gait aids in 4th decade",
            "Mild — late onset; minimal functional limitation; diagnosed incidentally",
            "Progressive — motor-predominant dHMN phenotype; hand + foot weakness",
        ],
    },

    # ── NEFL — CMT2E / CMT1F ─────────────────────────────────────────────
    {
        "gene": "NEFL", "protein": "Neurofilament Light Chain (NF-L)",
        "alias": "CMT2E/CMT1F (OMIM CMT2E #607684, CMT1F #607734); AD; neurofilament light chain NF-L; 8p21.2; variable demyelinating OR axonal; NF-L serum biomarker for axonal loss; severe juvenile onset possible",
        "aa": "543 aa", "kDa": "68 kDa",
        "gene_class": (
            "Neurofilament light chain (NEFL/NF-L); obligate component of neurofilament triplet "
            "(NF-L, NF-M, NF-H). Neurofilaments are the major intermediate filaments of neurons; "
            "they determine axonal calibre, conduction velocity, and structural integrity. "
            "NF-L forms the neurofilament backbone onto which NF-M and NF-H assemble. "
            "Mutations → dominant-negative → neurofilament misassembly → perikaryal aggregates "
            "→ axonal transport blockade → axon degeneration. "
            "DUAL PHENOTYPE depending on mutation class: "
            "CMT2E (axonal): NCV >38 m/s; CMAP low. "
            "CMT1F (demyelinating): NCV <38 m/s; uniform slowing. "
            "SEVERE JUVENILE ONSET: some mutations (e.g. p.Pro8Arg, p.Gln333Pro) → severe "
            "childhood disease; congenital hypomyelination in extreme cases. "
            "NF-L AS BIOMARKER: serum NF-L (SNfL) is now the leading CSF/serum biomarker "
            "of neuroaxonal damage across neurological diseases. "
            "In CMT: serum NF-L correlates with disease severity and rate of progression; "
            "emerging as primary endpoint in clinical trials (CMT-TRIAAL2, PREMIER trial). "
            "8p21.2; OMIM gene 162280."
        ),
        "neuropathy_group": "CMT2/CMT1 — Axonal or Demyelinating AD CMT",
        "subtype": "CMT2E/CMT1F — Neurofilament Light Chain Neuropathy",
        "locus": "8p21.2", "omim_gene": 162280, "omim_disease": 607684,
        "inheritance": "Autosomal Dominant (AD). Dominant-negative point mutations. Wide phenotypic range (axonal CMT2E to demyelinating CMT1F) depending on specific variant. Severe juvenile onset with some alleles.",
        "seed_offset": 7,
        "onset_range_y": (5.0, 40.0),
        "gender": "both",
        "severity_weights": [0.20, 0.40, 0.40],
        "nvc_range": (20, 50),
        "neuropathy_type": "variable",  # can be axonal or demyelinating
        "pes_cavus_prob": 0.65,
        "ankle_reflex_prob": 0.80,
        "scoliosis_prob": 0.15,
        "vocal_cord_prob": 0.0,
        "optic_atrophy_prob": 0.0,
        "cns_lesion_prob": 0.0,
        "phenotype": (
            "VARIABLE PHENOTYPE depending on mutation: "
            "CMT2E (axonal form): NCV >38 m/s; low CMAP; onset typically 2nd-4th decade. "
            "CMT1F (demyelinating form): NCV <38 m/s; uniform slowing; may mimic CMT1A. "
            "SEVERE JUVENILE ONSET (specific mutations): "
            "  - Childhood onset with rapid progression; pes cavus; foot drop early. "
            "  - Some develop congenital hypomyelination (CHN) — neonatal hypotonia. "
            "CLASSIC FEATURES: distal lower limb weakness → foot drop; pes cavus + hammer toes; "
            "absent ankle reflexes; distal sensory loss; hand weakness (later). "
            "NERVE BIOPSY: perikaryal neurofilament aggregates (pathognomonic for NEFL mutations); "
            "axonal loss (CMT2E) or demyelination/onion bulbs (CMT1F). "
            "NF-L SERUM BIOMARKER: serum NF-L elevated in proportion to axonal damage; "
            "now used as trial endpoint (correlates with disability, progression). "
            "Baseline serum NF-L recommended in all NEFL patients for longitudinal tracking. "
            "DDx CMT1A: if demyelinating NEFL (CMT1F) — MLPA negative → NEFL sequencing."
        ),
        "disease": (
            "NEFL neuropathy (CMT2E/CMT1F): rare, estimated <1% of CMT. "
            "Clinically important due to NF-L serum biomarker development — "
            "NEFL patients are ideal for biomarker validation studies. "
            "Serum NF-L in CMT: elevated vs controls; correlates with MRC score, "
            "CMTNS (CMT Neuropathy Score), and walking ability. "
            "Diagnosis: sequencing after MLPA negative (CMT1F) or axonal CMT2 panel. "
            "Management: AFO, physiotherapy. No disease-modifying Rx. "
            "NF-L biomarker monitoring: twice-yearly in trial settings; annually in clinic."
        ),
        "treatment_options": [
            "AFO (ankle-foot orthoses) — primary management",
            "Physiotherapy (early and sustained — strength + balance)",
            "Serum NF-L monitoring (biomarker — emerging trial endpoint)",
            "Occupational therapy + hand splints",
            "No disease-modifying Rx for CMT2E/CMT1F (2026)",
        ],
        "outcome_options": [
            "Progressive — moderate disability; AFO dependent; hand weakness 4th-5th decade",
            "Severe juvenile onset — rapid progression; wheelchair 2nd decade",
            "Moderate — adult onset; slow progression; ambulatory with aids",
            "Progressive — demyelinating CMT1F form; uniform NCV slowing + axonal loss",
            "Mild — late adult onset; minimal functional impact; serum NF-L monitoring",
        ],
    },
]


def _make_patients(gd):
    rng = random.Random(SEED_BASE + gd["seed_offset"])
    n = 40
    pts = []
    sev_labels = ["Mild", "Moderate", "Severe"]
    sev_weights = gd.get("severity_weights", [0.25, 0.45, 0.30])
    gender_bias = gd.get("gender", "both")

    for i in range(n):
        sid = f"{gd['gene']}-{SEED_BASE + gd['seed_offset']:03d}-{i+1:03d}"
        sev = rng.choices(sev_labels, weights=sev_weights, k=1)[0]
        lo, hi = gd["onset_range_y"]
        age_onset = round(rng.uniform(lo, hi), 1)

        # Diagnosis delay
        if gd["gene"] in ("SH3TC2", "GDAP1"):
            delay = round(rng.uniform(2.0, 10.0), 1)  # AR CMT — often delayed (rare)
        elif gd["gene"] == "GJB1":
            delay = round(rng.uniform(1.0, 8.0), 1)  # X-linked; females often misdiagnosed
        else:
            delay = round(rng.uniform(0.5, 6.0), 1)

        # Gender — X-linked logic for GJB1
        if gender_bias == "xlinkd":
            sex = rng.choice(["M", "F"])
            if sex == "M":
                age_onset = round(rng.uniform(10.0, 35.0), 1)  # males earlier
                sev = rng.choices(sev_labels, weights=[0.15, 0.40, 0.45], k=1)[0]  # males more severe
            else:
                age_onset = round(rng.uniform(20.0, 50.0), 1)  # females later
                sev = rng.choices(sev_labels, weights=[0.40, 0.45, 0.15], k=1)[0]  # females milder
        else:
            sex = rng.choice(["M", "F"])

        # NCV
        lo_ncv, hi_ncv = gd["nvc_range"]
        nvc_ms = round(rng.uniform(lo_ncv, hi_ncv), 1)

        # Neuropathy type
        ntype = gd["neuropathy_type"]
        if ntype == "variable":  # NEFL — can be axonal or demyelinating
            ntype = rng.choice(["demyelinating", "axonal"])

        # Clinical flags
        pes_cavus = rng.random() < gd.get("pes_cavus_prob", 0.60)
        ankle_reflex_absent = rng.random() < gd.get("ankle_reflex_prob", 0.80)
        scoliosis = rng.random() < gd.get("scoliosis_prob", 0.10)
        vocal_cord_paresis = rng.random() < gd.get("vocal_cord_prob", 0.0)
        optic_atrophy = rng.random() < gd.get("optic_atrophy_prob", 0.0)
        # CNS lesions for GJB1: males ~50%; females ~15%
        if gd["gene"] == "GJB1":
            cns_prob = 0.50 if sex == "M" else 0.15
        else:
            cns_prob = gd.get("cns_lesion_prob", 0.0)
        cns_lesions = rng.random() < cns_prob

        treatment = rng.choice(gd["treatment_options"])
        outcome = rng.choice(gd["outcome_options"])

        pts.append({
            "id": sid,
            "gene": gd["gene"],
            "sex": sex,
            "age_onset_y": age_onset,
            "dx_delay_y": delay,
            "severity": sev,
            "nvc_ms": nvc_ms,
            "neuropathy_type": ntype,
            "pes_cavus": pes_cavus,
            "ankle_reflex_absent": ankle_reflex_absent,
            "scoliosis": scoliosis,
            "vocal_cord_paresis": vocal_cord_paresis,
            "optic_atrophy": optic_atrophy,
            "cns_lesions": cns_lesions,
            "treatment": treatment,
            "outcome": outcome,
        })
    return pts


def get_overview():
    all_pts = []
    gene_summary = {}
    group_counts = {}
    for gd in CMT_GENES:
        pts = _make_patients(gd)
        all_pts.extend(pts)
        gene_summary[gd["gene"]] = len(pts)
        grp = gd["neuropathy_group"]
        group_counts[grp] = group_counts.get(grp, 0) + len(pts)

    n = len(all_pts)
    avg_onset = round(sum(p["age_onset_y"] for p in all_pts) / n, 1)
    avg_delay = round(sum(p["dx_delay_y"] for p in all_pts) / n, 1)

    sev_dist = {"Mild": 0, "Moderate": 0, "Severe": 0}
    pes_cavus_n = 0
    scoliosis_n = 0
    vocal_cord_n = 0
    optic_atrophy_n = 0
    cns_lesion_n = 0
    demyelinating_n = 0
    axonal_n = 0
    ar_cmt_n = 0

    for p in all_pts:
        sev_dist[p["severity"]] += 1
        if p["pes_cavus"]:
            pes_cavus_n += 1
        if p["scoliosis"]:
            scoliosis_n += 1
        if p["vocal_cord_paresis"]:
            vocal_cord_n += 1
        if p["optic_atrophy"]:
            optic_atrophy_n += 1
        if p["cns_lesions"]:
            cns_lesion_n += 1
        if p["neuropathy_type"] == "demyelinating":
            demyelinating_n += 1
        elif p["neuropathy_type"] == "axonal":
            axonal_n += 1
        if p["gene"] in ("SH3TC2", "GDAP1"):
            ar_cmt_n += 1

    return {
        "title": "CMT-Atlas — Complete 8-Gene Charcot-Marie-Tooth Hereditary Neuropathy Atlas",
        "subtitle": "PMP22/CMT1A-HNPP · MPZ/CMT1B · GJB1/CMTX1 · MFN2/CMT2A · SH3TC2/CMT4C · GDAP1/CMT4A · HSPB1/CMT2F · NEFL/CMT2E",
        "genes": [gd["gene"] for gd in CMT_GENES],
        "subtypes": [gd["subtype"] for gd in CMT_GENES],
        "total_patients": n,
        "avg_onset_y": avg_onset,
        "avg_dx_delay_y": avg_delay,
        "severity_distribution": sev_dist,
        "pes_cavus_n": pes_cavus_n,
        "pes_cavus_pct": round(100 * pes_cavus_n / n, 1),
        "scoliosis_n": scoliosis_n,
        "scoliosis_pct": round(100 * scoliosis_n / n, 1),
        "vocal_cord_n": vocal_cord_n,
        "vocal_cord_pct": round(100 * vocal_cord_n / n, 1),
        "optic_atrophy_n": optic_atrophy_n,
        "optic_atrophy_pct": round(100 * optic_atrophy_n / n, 1),
        "cns_lesion_n": cns_lesion_n,
        "cns_lesion_pct": round(100 * cns_lesion_n / n, 1),
        "demyelinating_n": demyelinating_n,
        "demyelinating_pct": round(100 * demyelinating_n / n, 1),
        "axonal_n": axonal_n,
        "axonal_pct": round(100 * axonal_n / n, 1),
        "ar_cmt_n": ar_cmt_n,
        "ar_cmt_pct": round(100 * ar_cmt_n / n, 1),
        "neuropathy_groups": group_counts,
        "gene_summary": gene_summary,
        "key_facts": [
            "PMP22 DUPLICATION = CMT1A (most common CMT; 70% of CMT1); PMP22 DELETION = HNPP (episodic pressure palsies); MLPA/aCGH mandatory first test",
            "CMT1 vs CMT2: median nerve NCV CUTOFF 38 m/s — demyelinating (CMT1) <38; axonal (CMT2) >38 with low CMAP amplitude; intermediate CMT (CMTX1): 25-45 m/s",
            "CMTX1 (GJB1): X-linked; males severe (NCV 25-40 m/s); females mild-moderate (30-45 m/s); CNS white matter lesions ~50% of males",
            "MFN2 (CMT2A): most common CMT2 (~20%); optic atrophy 5-10%; mitochondrial fusion; axonal; early/severe if de novo",
            "SH3TC2 (CMT4C): most common AR demyelinating CMT; severe scoliosis 60-70%; cranial nerve VII palsy (~30%); Roma founder p.Arg954Stop",
            "GDAP1 (CMT4A): most common AR CMT overall; vocal cord paresis DISTINCTIVE — ENT + respiratory review mandatory; mitochondrial fission",
            "ASCORBIC ACID (Vitamin C) for CMT1A: NEGATIVE — three large RCTs failed; NOT recommended; PXT3003 Phase 3 ongoing (2026)",
            "AFO (ankle-foot orthoses): mainstay treatment for foot drop in ALL CMT types; physiotherapy evidence-based for strength + function",
            "Pes cavus + hammer toes + absent ankle reflexes + distal wasting = classic CMT triad (all subtypes)",
            "NEFL (NF-L): serum NF-L is biomarker of CMT disease activity and axonal loss — emerging trial endpoint in CMT1A, CMT2 trials",
            "ONION BULB FORMATION on nerve biopsy: pathognomonic for demyelinating CMT1 (PMP22/MPZ) — repeated demyelination/remyelination cycles",
            "DRUGS TO AVOID in CMT: vincristine (severe acute neuropathy), taxanes, cisplatin, amiodarone, high-dose statins, nitrofurantoin",
        ],
        "critical_distinctions": {
            "CMT1A vs HNPP": "Same gene (PMP22), different mutation: duplication→CMT1A (slow NCV uniformly); deletion→HNPP (episodic pressure palsies, focal slowing at compression sites)",
            "CMT1 vs CMT2": "NCV-based: CMT1 = demyelinating (median NCV <38 m/s); CMT2 = axonal (NCV >38 m/s but low CMAP amplitude); intermediate CMT: 25-45 m/s (CMTX1)",
            "CMTX1 (GJB1) vs CMT1A": "X-linked; males severe; females mild-moderate; CNS white matter lesions 50%; INTERMEDIATE NCVs (not severely slow as CMT1A); connexin 32 gap junctions",
            "CMT4C vs CMT4A": "Both AR; CMT4C (SH3TC2) = severe scoliosis + cranial nerve VII; CMT4A (GDAP1) = vocal cord paresis + diaphragm + axonal type possible",
            "MFN2 (CMT2A) vs others": "Most common CMT2; optic atrophy (5-10%); mitochondrial fusion; axonal; early severe form in compound het; nerve biopsy shows axonal loss (no onion bulbs)",
        },
        "seeds_used": f"{SEED_BASE}–{SEED_BASE + len(CMT_GENES) - 1}",
    }


_gd_cache = {gd["gene"]: gd for gd in CMT_GENES}


def gd_by_gene(gene):
    return _gd_cache[gene]


def get_breakdown():
    result = []
    for gd in CMT_GENES:
        pts = _make_patients(gd)
        sev = {"Severe": 0, "Moderate": 0, "Mild": 0}
        for p in pts:
            sev[p["severity"]] += 1
        avg_delay = round(sum(p["dx_delay_y"] for p in pts) / len(pts), 1)
        avg_onset = round(sum(p["age_onset_y"] for p in pts) / len(pts), 1)
        avg_nvc = round(sum(p["nvc_ms"] for p in pts) / len(pts), 1)
        pes_cavus_n = sum(1 for p in pts if p["pes_cavus"])
        scoliosis_n = sum(1 for p in pts if p["scoliosis"])
        vocal_cord_n = sum(1 for p in pts if p["vocal_cord_paresis"])
        optic_atrophy_n = sum(1 for p in pts if p["optic_atrophy"])
        cns_lesion_n = sum(1 for p in pts if p["cns_lesions"])
        demyelinating_n = sum(1 for p in pts if p["neuropathy_type"] == "demyelinating")
        axonal_n = sum(1 for p in pts if p["neuropathy_type"] == "axonal")
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        top_tx = sorted(treatments.items(), key=lambda x: -x[1])[:3]
        outcomes = {}
        for p in pts:
            outcomes[p["outcome"]] = outcomes.get(p["outcome"], 0) + 1
        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "neuropathy_group": gd["neuropathy_group"],
            "subtype": gd["subtype"],
            "n_patients": len(pts),
            "severity_distribution": sev,
            "avg_onset_y": avg_onset,
            "avg_dx_delay_y": avg_delay,
            "avg_nvc_ms": avg_nvc,
            "nvc_range": f"{gd['nvc_range'][0]}–{gd['nvc_range'][1]} m/s",
            "neuropathy_type": gd["neuropathy_type"],
            "pes_cavus_n": pes_cavus_n,
            "pes_cavus_pct": round(100 * pes_cavus_n / len(pts), 1),
            "scoliosis_n": scoliosis_n,
            "scoliosis_pct": round(100 * scoliosis_n / len(pts), 1),
            "vocal_cord_n": vocal_cord_n,
            "vocal_cord_pct": round(100 * vocal_cord_n / len(pts), 1),
            "optic_atrophy_n": optic_atrophy_n,
            "optic_atrophy_pct": round(100 * optic_atrophy_n / len(pts), 1),
            "cns_lesion_n": cns_lesion_n,
            "cns_lesion_pct": round(100 * cns_lesion_n / len(pts), 1),
            "demyelinating_n": demyelinating_n,
            "demyelinating_pct": round(100 * demyelinating_n / len(pts), 1),
            "axonal_n": axonal_n,
            "axonal_pct": round(100 * axonal_n / len(pts), 1),
            "top_treatments": [{"tx": t, "n": c} for t, c in top_tx],
            "outcome_distribution": outcomes,
            "gene_class": gd["gene_class"],
            "phenotype": gd["phenotype"],
            "disease": gd["disease"],
        })
    return {
        "total_genes": len(CMT_GENES),
        "total_patients": sum(r["n_patients"] for r in result),
        "breakdown": result,
    }


def get_definitions():
    return {
        "definitions": [
            {
                "term": "CMT Classification — NCV-Based Scheme, Inheritance, and Genetic Approach",
                "definition": (
                    "Charcot-Marie-Tooth (CMT) disease: most common inherited peripheral neuropathy (prevalence ~1:2,500). "
                    "CLASSIFICATION BY NERVE CONDUCTION VELOCITY (MEDIAN NERVE): "
                    "  CMT1 = demyelinating (median NCV <38 m/s); uniform slowing; autosomal dominant. "
                    "  CMT2 = axonal (NCV >38 m/s but low CMAP amplitude); AD or AR. "
                    "  CMTX = X-linked (intermediate NCV 25-45 m/s; connexin-32 GJB1). "
                    "  CMT4 = autosomal recessive (demyelinating or axonal; early, severe). "
                    "  Intermediate CMT = NCV 25-45 m/s; overlap of demyelinating/axonal features. "
                    "GENETIC TESTING ALGORITHM: "
                    "(1) MLPA/aCGH for PMP22 copy number (duplication = CMT1A; deletion = HNPP). "
                    "(2) MPZ sequencing (if MLPA negative + demyelinating). "
                    "(3) GJB1 sequencing (if X-linked inheritance or intermediate NCV). "
                    "(4) MFN2 sequencing (axonal CMT2 — most common CMT2). "
                    "(5) Panel sequencing (SH3TC2, GDAP1, HSPB1, NEFL, and others). "
                    "(6) WES/WGS if panel negative. "
                    "KEY: NCV determines whether CMT1 or CMT2 → guides gene choice."
                )
            },
            {
                "term": "PMP22 Duplication Mechanism — Gene Dosage, MLPA, and Why Ascorbic Acid Failed",
                "definition": (
                    "CMT1A is caused by ~1.4 Mb segmental duplication at chromosome 17p11.2 → "
                    "3 copies of PMP22 (normal = 2). "
                    "GENE DOSAGE EFFECT: PMP22 overexpression in Schwann cells → "
                    "dysmyelination → uniformly slow NCV (5-15 m/s in all peripheral nerves). "
                    "Mechanism: excess PMP22 protein → retention in ER → UPR (unfolded protein response) → "
                    "Schwann cell dysfunction → demyelination. "
                    "MLPA (Multiplex Ligation-dependent Probe Amplification): detects copy number changes; "
                    "sequencing cannot detect duplications. aCGH (array CGH) also used. MLPA is mandatory "
                    "first-line test when CMT1 suspected. "
                    "ASCORBIC ACID HYPOTHESIS: Vitamin C was proposed to reduce PMP22 expression "
                    "via antioxidant mechanisms (promising in rodent CMT1A models). "
                    "FAILED RCTs: CMT-TRIAAL (European, 2011), NACPMS (North American, 2011), "
                    "third trial (2013) — all negative. No clinical benefit. "
                    "Not recommended in 2026 guidelines. "
                    "PXT3003 (baclofen + naltrexone + D-sorbitol) Phase 3 PREMIER trial: "
                    "targets multiple pathways (HDAC6, EGFR, neuroprotection); results awaited."
                )
            },
            {
                "term": "HNPP (PMP22 Deletion) — Episodic Pressure Palsies and Focal Slowing",
                "definition": (
                    "HNPP (Hereditary Neuropathy with Liability to Pressure Palsies): "
                    "caused by heterozygous ~1.4 Mb deletion at 17p11.2 → 1 copy of PMP22 "
                    "(haploinsufficiency → thinner myelin at paranodal regions). "
                    "CLINICAL PRESENTATION: "
                    "  - EPISODIC focal pressure palsies at common entrapment sites: "
                    "    carpal tunnel (wrist — median nerve), fibula head (peroneal nerve — foot drop), "
                    "    elbow (ulnar nerve — claw hand), spiral groove (radial nerve). "
                    "  - Palsies triggered by trivial compression (sitting cross-legged, sleeping on arm). "
                    "  - Usually recover over weeks-months (unlike CMT1A which is progressive). "
                    "  - Background mild sensorimotor neuropathy between episodes. "
                    "NCS PATTERN: focal conduction slowing at compression sites (above background); "
                    "mild diffuse generalised slowing across all nerves (background demyelination). "
                    "DIAGNOSIS: clinical suspicion + NCS (focal slowing at carpal tunnel + generalised mild slowing) "
                    "→ MLPA (PMP22 deletion detected). "
                    "DDx: mononeuritis multiplex (inflammatory), multifocal neuropathy (MMNCB), CIDP. "
                    "MANAGEMENT: avoidance of pressure (protective pads); ergonomic work adjustments; "
                    "physical therapy. Prognosis: generally good with avoidance."
                )
            },
            {
                "term": "CMTX1 (GJB1/Connexin 32) — Gap Junction Biology, Sex Differences, CNS Lesions",
                "definition": (
                    "Connexin 32 (GJB1) forms gap junctions (hemichannels) in paranodal loops and "
                    "Schmidt-Lanterman incisures of Schwann cell cytoplasm. "
                    "FUNCTION: radial diffusion of metabolites, ions, and small molecules through "
                    "myelin sheath — critical for maintenance of myelin layers far from Schwann cell body. "
                    "X-LINKED INHERITANCE: "
                    "  Males (hemizygous): severe; onset 10-35 years; NCV intermediate 25-40 m/s; "
                    "    significant weakness; may develop CNS lesions. "
                    "  Females (heterozygous): mild-moderate; onset 20-50 years; NCV 30-45 m/s; "
                    "    variable (lyonisation); often milder due to mosaic X-inactivation. "
                    "  NO MALE-TO-MALE TRANSMISSION: key X-linked diagnostic clue. "
                    "CNS WHITE MATTER LESIONS: "
                    "  GJB1 expressed in CNS oligodendrocytes → T2/FLAIR periventricular hyperintensities. "
                    "  Present in ~50% of male CMTX1 patients on MRI; clinically often silent. "
                    "  Rarely: transient encephalopathic episodes (acute confusion, ataxia) — "
                    "  usually self-limited; may be exacerbated by fever, exercise, altitude. "
                    "INTERMEDIATE NCV: distinguishes CMTX1 from CMT1A (severely slow <15) and CMT2A (normal). "
                    "Over 400 GJB1 pathogenic variants; frameshift/nonsense/missense all described."
                )
            },
            {
                "term": "CMT1 vs CMT2 Electrophysiology — Onion Bulbs vs Axonal Loss; Nerve Biopsy",
                "definition": (
                    "ELECTROPHYSIOLOGY — THE DEFINITIVE CMT CLASSIFICATION TOOL: "
                    "MOTOR NCV (median nerve preferred): "
                    "  CMT1 demyelinating: <38 m/s (often <15 m/s in CMT1A) — uniform across all nerves "
                    "    (uniform slowing = genetic demyelinating; patchy slowing = acquired CIDP). "
                    "  CMT2 axonal: >38 m/s (often normal); but CMAP amplitude markedly reduced "
                    "    (axonal dropout); SNAP amplitude reduced. "
                    "  Intermediate: 25-45 m/s — CMTX1, GDAP1, NEFL (some). "
                    "UNIFORMITY IS KEY: demyelinating CMT1 → same slow NCV in median, ulnar, peroneal. "
                    "ACQUIRED CIDP (DDx): patchy slowing; temporal dispersion; proximal > distal; "
                    "raised CSF protein; responds to immunotherapy. "
                    "NERVE BIOPSY — HISTOLOGY: "
                    "  CMT1 (demyelinating): ONION BULBS — concentric lamellae of Schwann cell "
                    "    processes around thinly myelinated or demyelinated axons; pathognomonic. "
                    "  CMT2 (axonal): axonal loss; no onion bulbs; may see clusters of regenerating axons. "
                    "  Biopsy not routinely required for diagnosis (genetic testing usually sufficient). "
                    "SURAL NERVE BIOPSY: reserved for diagnostically challenging cases."
                )
            },
            {
                "term": "GDAP1 Vocal Cord Paresis — Unique Feature of CMT4A; Respiratory Monitoring",
                "definition": (
                    "GDAP1 (CMT4A) is distinctive for VOCAL CORD PARESIS — not seen in CMT1A/CMT1B/CMTX1. "
                    "MECHANISM: GDAP1 expressed in Schwann cells and neurons; "
                    "recurrent laryngeal nerve (branch of vagus nerve) is long, thin myelinated nerve "
                    "→ vulnerable to GDAP1-related axonal/demyelinating degeneration → "
                    "vocal cord denervation → paresis. "
                    "CLINICAL FEATURES OF VOCAL CORD PARESIS: "
                    "  - Hoarse or breathy voice (dysphonia). "
                    "  - Stridor (inspiratory high-pitched sound — alarming when bilateral). "
                    "  - Dysphagia and aspiration risk (laryngeal incompetence). "
                    "  - Bilateral vocal cord paralysis in severe cases → respiratory failure. "
                    "MANAGEMENT: "
                    "  ENT (laryngoscopy): baseline + annual review in GDAP1 patients. "
                    "  Speech and language therapy: swallowing assessment; thickened fluids if aspiration. "
                    "  Tracheostomy: if severe bilateral paresis with stridor/hypoxia. "
                    "DIAPHRAGM INVOLVEMENT: respiratory muscle weakness in GDAP1 → "
                    "  Annual spirometry (FVC); nocturnal oximetry; bilevel NIV if FVC <50% or "
                    "  nocturnal desaturations. "
                    "Vocal cord paresis + childhood-onset neuropathy + AR inheritance → GDAP1 sequencing."
                )
            },
            {
                "term": "AFO and Orthopedic Management of CMT — Foot Deformity, Pes Cavus, Surgical Options",
                "definition": (
                    "FOOT DEFORMITY IN CMT: pes cavus (high-arched foot) + hammer toes + calluses "
                    "present in >70% of all CMT subtypes. "
                    "MECHANISM: imbalance between intrinsic foot muscles (denervated early) and extrinsic "
                    "(longer-fibred; relatively preserved initially) → progressive foot arch deformity. "
                    "PES CAVUS SEVERITY: graded by Meary angle (lateral weight-bearing X-ray); "
                    "Charcot-Marie-Tooth Neuropathy Score (CMTNS) includes functional foot assessment. "
                    "AFO (ANKLE-FOOT ORTHOSIS): "
                    "  - Primary management for foot drop (inability to dorsiflex). "
                    "  - Types: posterior leaf spring (mild drop); solid AFO (moderate-severe); "
                    "    KAFO (knee-ankle-foot) for severe proximal weakness. "
                    "  - Improves walking speed, energy expenditure, balance. "
                    "  - Should be provided early (on diagnosis, not when severe). "
                    "ORTHOPEDIC SURGERY: "
                    "  - Plantar fascia release + intrinsic muscle release for mild pes cavus. "
                    "  - Calcaneal osteotomy (Dwyer/lateral closing wedge): corrects heel varus. "
                    "  - First metatarsal osteotomy (dorsiflexion): corrects plantar flexion. "
                    "  - Tendon transfer (tibialis anterior, extensor hallucis): reduces drop. "
                    "PHYSIOTHERAPY: resistance training; aerobic exercise; balance; evidence-based. "
                    "Vitamin C trials: negative. PXT3003 Phase 3 for CMT1A ongoing."
                )
            },
            {
                "term": "NF-L (NEFL/Serum NF-L) — Biomarker for Axonal Damage and Trial Endpoint",
                "definition": (
                    "Neurofilament light chain (NF-L/NFL): structural protein of neuronal intermediate filaments. "
                    "NF-L is released into CSF and blood when axons are injured or degenerate. "
                    "SERUM NF-L IN CMT: "
                    "  - Elevated in CMT patients vs healthy controls across multiple subtypes (CMT1A, CMT2A, CMTX1). "
                    "  - Highest levels in severe or progressive forms. "
                    "  - Correlates with: CMTNS score (CMT Neuropathy Score), MRC motor scale, "
                    "    walking velocity, disease duration. "
                    "BIOMARKER VALUE: "
                    "  (1) Diagnostic: distinguishes active axonal loss from stable disease. "
                    "  (2) Prognostic: higher NF-L = faster progression. "
                    "  (3) TRIAL ENDPOINT: serum NF-L as primary endpoint in CMT1A and CMT2 trials "
                    "      (PREMIER trial for PXT3003 uses NF-L). "
                    "  (4) Treatment monitoring: change in NF-L may detect treatment response "
                    "      before clinical improvement visible. "
                    "MEASUREMENT: Simoa (single molecule array) or Ella platforms; "
                    "serum easier than CSF; reliable with proper sample handling. "
                    "NEFL MUTATIONS: NEFL gene mutations (CMT2E/CMT1F) → severe axonal disease; "
                    "these patients have highest NF-L levels; used as extreme model for biomarker studies. "
                    "AGE NORMALISATION: NF-L rises with age even in controls; age-matched reference ranges used."
                )
            },
        ]
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print("=== CMT ATLAS OVERVIEW ===")
    print(f"Genes: {ov['genes']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Avg onset: {ov['avg_onset_y']} y")
    print(f"Avg dx delay: {ov['avg_dx_delay_y']} y")
    print(f"Pes cavus: {ov['pes_cavus_n']} ({ov['pes_cavus_pct']}%)")
    print(f"Scoliosis: {ov['scoliosis_n']} ({ov['scoliosis_pct']}%)")
    print(f"Vocal cord paresis: {ov['vocal_cord_n']} ({ov['vocal_cord_pct']}%)")
    print(f"Optic atrophy: {ov['optic_atrophy_n']} ({ov['optic_atrophy_pct']}%)")
    print(f"CNS lesions: {ov['cns_lesion_n']} ({ov['cns_lesion_pct']}%)")
    print(f"Demyelinating: {ov['demyelinating_n']} ({ov['demyelinating_pct']}%)")
    print(f"Axonal: {ov['axonal_n']} ({ov['axonal_pct']}%)")
    print(f"Neuropathy groups: {ov['neuropathy_groups']}")
    bd = get_breakdown()
    print(f"\n=== BREAKDOWN ({bd['total_genes']} genes) ===")
    for g in bd["breakdown"]:
        print(f"  {g['gene']}: {g['n_patients']} pts, onset {g['avg_onset_y']}y, delay {g['avg_dx_delay_y']}y, NVC {g['avg_nvc_ms']} m/s")
    defs = get_definitions()
    print(f"\n=== DEFINITIONS: {len(defs['definitions'])} terms ===")
