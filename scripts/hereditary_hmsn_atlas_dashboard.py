#!/usr/bin/env python3
"""Hereditary-HMSN-Atlas — Complete 8-Gene Hereditary Motor and Sensory Neuropathy (CMT/HMSN) Atlas
PMP22   (Peripheral Myelin Protein 22; 160 aa; 17p12; AD/AR;
          CMT1A — duplication 17p12 — most common CMT 70-80% of CMT1;
          HNPP — deletion 17p12 — pressure-induced episodic neuropathy;
          CMT1E — point mutations — variable severity;
          Uniformly slowed NCV <38 m/s median; pes cavus; onion bulbs; MLPA MANDATORY) ·
MPZ     (Myelin Protein Zero; 248 aa; 1q23.3; AD;
          CMT1B — severe demyelinating, NCV <20 m/s — P0 adhesion molecule;
          CMT2I/J — axonal adult-onset — same gene, different mechanism;
          Early NCV testing — childhood onset — NCV determines CMT1B vs CMT2I/J) ·
GJB1    (Connexin 32; 283 aa; Xq13.1; XLD;
          CMTX1 — most common X-linked CMT 10-15% of all CMT;
          Males moderate-severe NCV 30-40 m/s intermediate; Females mild/asymptomatic;
          CNS white matter lesions — transient stroke-like episodes) ·
MFN2    (Mitofusin 2; 741 aa; 1p36.22; AD;
          CMT2A2 — most common axonal CMT 20% of CMT2;
          Early onset severe — upper limb > lower limb — optic atrophy 15%;
          NCV normal/mildly reduced — mitochondrial fusion defect) ·
GDAP1   (GDAP1; 358 aa; 8q21.11; AR;
          CMT4A — most severe AR CMT — early childhood onset;
          Vocal cord paresis 30% — diaphragm involvement — wheelchair risk;
          CMT2K — milder AR allelic form) ·
SH3TC2  (SH3TC2; 1288 aa; 5q32; AR;
          CMT4C — most common AR CMT worldwide — Romani/Gypsy founder;
          Scoliosis 60% MANDATORY surveillance — cranial nerve palsies;
          R954W Mediterranean founder — Schwann cell endosomal recycling) ·
NEFL    (Neurofilament Light Chain; 543 aa; 8p21.2; AD/AR;
          CMT2E — AD axonal — giant axon neurofilament aggregation;
          CMT1F — AD demyelinating intermediate NCV;
          Facial/bulbar involvement possible — proximal weakness uncommon) ·
EGR2    (Early Growth Response 2 / Krox20; 472 aa; 10q21.2; AD/AR;
          CMT1D — AD severe demyelinating hypomyelination — master myelin TF;
          CMT4E — AR — congenital onset — NCV near 0 m/s;
          R359W / R381H hotspot — zinc finger DNA-binding domain)
320-patient aggregate cohort (8 × 40, seeds 1462–1469)
"""

import random

SEED_BASE = 1462

HMSN_GENES = [
    # ── PMP22 — CMT1A / HNPP / CMT1E ──
    {
        "gene": "PMP22",
        "protein": "Peripheral Myelin Protein 22 — Myelin Compaction Structural Glycoprotein",
        "alias": (
            "PMP22; OMIM gene 601097; CMT1A OMIM 118220; HNPP OMIM 162500; CMT1E OMIM 118300; 17p12; 160 aa; ~22 kDa; "
            "Peripheral Myelin Protein 22 — compact myelin structural protein expressed in Schwann cells; "
            "17p12 tandem duplication (1.4 Mb CMT1A-REP mediated) → 3 copies → CMT1A most common CMT; "
            "17p12 deletion → HNPP: hereditary neuropathy with liability to pressure palsies — episodic; "
            "Point mutations (CMT1E) — severe hypomyelinating; some de novo; "
            "Uniform slowing NCV <38 m/s (median) ALL fibres — electrophysiology hallmark; "
            "Pes cavus; distal weakness/wasting; areflexia; onion bulb formation on nerve biopsy; "
            "MLPA MANDATORY for CMT1A/HNPP — standard sequencing misses deletion/duplication; "
            "No curative therapy; orthotics; MDT; ankle-foot orthoses (AFO); do NOT prescribe vincristine"
        ),
        "aa": "160 aa",
        "kDa": "~22 kDa",
        "locus": "17p12",
        "omim_gene": 601097,
        "omim_disease": 118220,
        "inheritance": "AD — duplication (CMT1A) / deletion (HNPP) / point mutation (CMT1E); MLPA mandatory",
        "gene_class": (
            "PMP22 encodes a compact myelin glycoprotein expressed almost exclusively in myelinating Schwann "
            "cells of the peripheral nervous system. The protein constitutes ~5% of total PNS myelin by mass "
            "and is critical for myelin compaction, Schwann cell differentiation, and apoptosis regulation. "
            "Gene dosage is the key pathomechanism: a 1.4 Mb tandem duplication of chromosome 17p12 (mediated "
            "by CMT1A-REP low-copy repeats), yielding 3 PMP22 copies, causes CMT1A — the most prevalent "
            "inherited neuropathy globally (~1:2500). The reciprocal deletion causes HNPP, characterised by "
            "episodic pressure-induced focal demyelination at compression sites (fibular head, carpal tunnel, "
            "elbow). Electrophysiology in CMT1A shows uniform diffuse conduction slowing of ALL fibres "
            "(NCV <38 m/s median), distinguishing it from acquired inflammatory neuropathy (which is "
            "non-uniform). PMP22 point mutations (CMT1E) cause a more severe or hypomyelinating phenotype. "
            "MLPA is the diagnostic standard — sequencing alone misses 99% of CMT1A/HNPP cases."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("17p12 tandem duplication — 3 copies PMP22 — CMT1A, most common CMT", 0.70),
            ("17p12 deletion — 1 copy PMP22 — HNPP episodic pressure neuropathy", 0.18),
            ("PMP22 point mutation exon 3/5 — CMT1E, severe or hypomyelinating", 0.08),
            ("PMP22 frameshift/nonsense — hypomyelination, severe", 0.04),
        ],
        "age_onset_years_range": (5, 20),
        "sex_ratio_M": 0.50,
        "rates": {
            "pes_cavus":                               0.85,
            "distal_lower_limb_weakness":              0.90,
            "distal_upper_limb_weakness":              0.55,
            "areflexia_all_tendons":                   0.80,
            "ncv_less_than_38_ms_median":              0.95,
            "onion_bulb_nerve_biopsy":                 0.75,
            "scoliosis":                               0.25,
            "sensory_loss_glove_stocking":             0.75,
            "episodic_pressure_palsy_hnpp":            0.20,
            "wheelchair_by_age_40":                    0.12,
        },
        "critical_alerts": [
            "MLPA MANDATORY — standard sequencing misses CMT1A duplication and HNPP deletion — order MLPA first",
            "VINCRISTINE ABSOLUTE CI — demyelinating CMT + vincristine = fatal motor neuropathy — flag on all records",
            "NCV <38 m/s ALL fibres = uniform slowing = demyelinating CMT — distinguishes from acquired neuropathy",
            "HNPP: avoid prolonged pressure at fibular head, elbow, wrist — occupational therapy guidance mandatory",
        ],
        "key_ddx_rules": [
            "Uniform NCV slowing ALL fibres → CMT1 (genetic); Non-uniform focal slowing → acquired (CIDP, MMN)",
            "HNPP: episodic palsies + family history → MLPA deletion; NOT CIDP — steroids contraindicated",
            "CMT1A confirmed by duplication; do NOT sequencePMP22 first — MLPA before gene panel",
            "Intermediate NCV 30–40 m/s + male + X-linked pattern → order GJB1/CMTX1 not PMP22",
        ],
    },

    # ── MPZ (P0) — CMT1B / CMT2I / CMT2J ──
    {
        "gene": "MPZ",
        "protein": "Myelin Protein Zero (P0) — PNS Myelin Adhesion Molecule Homotypic Compaction",
        "alias": (
            "MPZ; P0; OMIM gene 159440; CMT1B OMIM 118200; CMT2I OMIM 607677; CMT2J OMIM 607736; 1q23.3; 248 aa; ~28 kDa; "
            "Myelin Protein Zero — most abundant PNS myelin protein (~50-70% of myelin protein); "
            "Homotypic adhesion molecule — extracellular Ig-like domain compacts myelin lamellae (P0-P0 trans); "
            "CMT1B: demyelinating severe, NCV <20 m/s, childhood onset, onion bulbs, deafness in some; "
            "CMT2I: axonal adult-onset (>40 yr), NCV normal, late-onset sensory dominant; "
            "CMT2J: axonal + pupillary abnormalities + deafness + adult onset; "
            ">200 pathogenic variants; genotype-phenotype correlation guides prognosis; "
            "D75V/D75N hotspot; T124M adult-onset axonal; NCV measurement child guides type"
        ),
        "aa": "248 aa",
        "kDa": "~28 kDa",
        "locus": "1q23.3",
        "omim_gene": 159440,
        "omim_disease": 118200,
        "inheritance": "AD — dominant negative or haploinsufficiency; CMT1B early severe, CMT2I/J adult onset",
        "gene_class": (
            "MPZ encodes Myelin Protein Zero (P0), the most abundant structural protein of peripheral nerve "
            "myelin, constituting 50–70% of total PNS myelin protein. P0 functions as a homotypic adhesion "
            "molecule: its extracellular immunoglobulin-like domain bridges opposing myelin membrane leaflets "
            "(trans interaction), maintaining the tight 'intraperiod' line of compact myelin. Pathogenic "
            "MPZ variants cause two distinct clinical syndromes depending on variant type and position: "
            "(1) CMT1B — childhood-onset severe demyelinating neuropathy with NCV <20 m/s, thick onion "
            "bulbs on nerve biopsy, progressive weakness, and sometimes hearing loss; "
            "(2) CMT2I/J — adult-onset (>40 years) predominantly axonal neuropathy with normal or "
            "near-normal NCV; CMT2J additionally features pupillary abnormalities and deafness. "
            "Key genotype-phenotype landmarks: Asp75Val/Asn (D75V/D75N) typically causes CMT1B; "
            "Thr124Met (T124M) causes adult CMT2I. NCV measured in childhood is the first branch-point "
            "in differentiating CMT1B from CMT2I/J. >200 MPZ pathogenic variants are reported."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("MPZ missense exon 3/4 Ig-domain — CMT1B demyelinating childhood severe", 0.55),
            ("MPZ T124M or equivalent — CMT2I adult axonal sensory dominant", 0.22),
            ("MPZ D75V/D75N hotspot — CMT1B with deafness/pupillary changes", 0.13),
            ("MPZ frameshift/nonsense — haploinsufficiency, intermediate NCV", 0.06),
            ("MPZ splice site — exon skipping partial protein, variable", 0.04),
        ],
        "age_onset_years_range": (2, 50),
        "sex_ratio_M": 0.50,
        "rates": {
            "demyelinating_ncv_less_20ms_cmtib":       0.55,
            "axonal_ncv_normal_cmt2ij":                0.43,
            "hearing_loss_sensorineural":              0.25,
            "pupillary_abnormalities_cmt2j":           0.15,
            "pes_cavus":                               0.70,
            "distal_weakness_lower":                   0.85,
            "areflexia":                               0.75,
            "onion_bulb_nerve_biopsy":                 0.55,
            "scoliosis":                               0.20,
            "wheelchair_by_age_40":                    0.18,
        },
        "critical_alerts": [
            "NCV <20 m/s in child → CMT1B MPZ — most severe demyelinating CMT — audiogram mandatory",
            "Adult onset sensory neuropathy + family history → check MPZ T124M (CMT2I/J) — often missed",
            "CMT2J: pupils irregular/light-near dissociation + deafness + neuropathy — ophthalmology mandatory",
            "VINCRISTINE ABSOLUTE CI in all MPZ neuropathy regardless of CMT subtype",
        ],
        "key_ddx_rules": [
            "NCV <20 m/s childhood = CMT1B > CMTX (X-linked pattern distinguishes) > EGR2",
            "Adult-onset sensory neuropathy + slow progression → MPZ T124M (CMT2I) vs NEFL vs MFN2",
            "CMT1B vs CMT1A: MPZ panel + PMP22 MLPA together; CMT1B NCV more variable, MPZ more severe",
            "Deafness + neuropathy → MPZ CMT2J OR GJB1 CMTX1 CNS — check X-linkage first",
        ],
    },

    # ── GJB1 (Cx32) — CMTX1 ──
    {
        "gene": "GJB1",
        "protein": "Connexin 32 (Cx32) — Gap Junction Protein Beta 1 — Schwann Cell Paranodal Gap Junctions",
        "alias": (
            "GJB1; CX32; OMIM gene 304040; CMTX1 OMIM 302800; Xq13.1; 283 aa; ~32 kDa; "
            "Connexin 32 — gap junction channel protein expressed in Schwann cell paranodal loops and incisures; "
            "CMTX1: most common X-linked CMT, ~10-15% of all CMT, second most common CMT type after CMT1A; "
            "X-linked dominant (XLD) — males moderate-severe (NCV 30-40 m/s intermediate); females mild/asymptomatic; "
            "Intermediate NCV 30-40 m/s males — does NOT fit demyelinating (<38) or axonal (>40) pure categories; "
            "CNS white matter lesions on MRI — transient stroke-like episodes especially during fever/infection; "
            "Hearing loss sensorineural males 20%; >400 GJB1 pathogenic variants; "
            "NO male-to-male transmission (X-linked) — key pedigree rule"
        ),
        "aa": "283 aa",
        "kDa": "~32 kDa",
        "locus": "Xq13.1",
        "omim_gene": 304040,
        "omim_disease": 302800,
        "inheritance": "XLD — males moderate-severe; females mild/asymptomatic; NO male-to-male transmission",
        "gene_class": (
            "GJB1 encodes Connexin 32 (Cx32), a gap junction protein highly expressed in the paranodal loops "
            "and Schmidt-Lantermann incisures of myelinating Schwann cells. Cx32 forms gap junction channels "
            "that allow rapid ionic and metabolic communication across the compact myelin sheath (reflexive "
            "or 'reflexive junctions') — a pathway ~1000× shorter than the outer Schwann cell surface path. "
            "Loss of Cx32 impairs the radial diffusion pathway for small molecules within myelin, "
            "compromising Schwann cell metabolic support of the axon. CMTX1 is X-linked dominant: "
            "hemizygous males (one X) are moderately to severely affected with NCV 30–40 m/s (intermediate "
            "— neither purely demyelinating nor axonal), while heterozygous females are typically mildly "
            "affected or subclinical. A critical diagnostic clue: intermediate NCV in a male with apparent "
            "sporadic or X-linked pedigree should trigger GJB1 sequencing before CMT panel. CNS white "
            "matter involvement (T2 hyperintensities on MRI) causes transient stroke-like episodes, "
            "particularly with fever; recovery is complete but episodes alarm clinicians. >400 GJB1 "
            "pathogenic variants reported; no male-to-male transmission confirms X-linkage."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("GJB1 missense — connexin-forming domain — channel dysfunction, most common", 0.65),
            ("GJB1 frameshift/nonsense — Cx32 absent — severe male phenotype", 0.15),
            ("GJB1 splice site — exon skipping partial Cx32 truncation", 0.12),
            ("GJB1 large deletion Xq13.1 — MLPA required, rare", 0.05),
            ("GJB1 promoter region — reduced expression, mild phenotype females", 0.03),
        ],
        "age_onset_years_range": (10, 35),
        "sex_ratio_M": 0.60,
        "rates": {
            "intermediate_ncv_30_40ms_males":          0.90,
            "females_mild_or_asymptomatic":            0.70,
            "cns_white_matter_lesions_mri":            0.35,
            "transient_strokelike_episodes":           0.20,
            "hearing_loss_sensorineural_males":        0.20,
            "pes_cavus":                               0.65,
            "distal_weakness_lower":                   0.80,
            "areflexia":                               0.70,
            "no_maleto_male_transmission":             1.00,
            "wheelchair_by_age_40":                    0.08,
        },
        "critical_alerts": [
            "INTERMEDIATE NCV 30-40 m/s in male → GJB1/CMTX1 first — NO male-to-male transmission confirms X-linkage",
            "CNS episodes: transient hemiplegia/ataxia + white matter MRI → CMTX1 CNS — NOT stroke — supportive only",
            "Female carriers: mild or asymptomatic — do NOT dismiss — nerve conduction may be mildly abnormal",
            "VINCRISTINE ABSOLUTE CI — all CMTX1 patients — flag prominently in records",
        ],
        "key_ddx_rules": [
            "NCV 30-40 m/s intermediate + male + no father-son → GJB1/CMTX1 before CMT panel",
            "Stroke-like episodes + neuropathy → CMTX1 CNS (GJB1 sequence) vs MELAS (mitochondrial)",
            "CMTX1 vs CMT1A: X-linkage distinguishes; CMTX1 NCV intermediate not uniformly <38 m/s",
            "Asymptomatic female + affected son → obligate carrier GJB1 — screen all sons",
        ],
    },

    # ── MFN2 — CMT2A2 ──
    {
        "gene": "MFN2",
        "protein": "Mitofusin 2 — Mitochondrial Outer Membrane Fusion GTPase Axonal Transport Regulator",
        "alias": (
            "MFN2; OMIM gene 608507; CMT2A2 OMIM 609260; 1p36.22; 741 aa; ~86 kDa; "
            "Mitofusin 2 — GTPase mediating mitochondrial outer membrane fusion (with MFN1); "
            "CMT2A2: most common axonal CMT, ~20% of CMT2; early onset childhood to adolescence; "
            "Upper limb > lower limb involvement — unusual for CMT; wheelchair risk early; "
            "Optic atrophy 15% — Kjer-like, visual acuity monitoring mandatory; "
            "Pyramidal signs in some; NCV normal or mildly reduced (axonal); "
            "Mitochondrial transport defect in axons — axon length-dependent; "
            "De novo mutations common in sporadic severe cases; proband sequencing + parents"
        ),
        "aa": "741 aa",
        "kDa": "~86 kDa",
        "locus": "1p36.22",
        "omim_gene": 608507,
        "omim_disease": 609260,
        "inheritance": "AD — haploinsufficiency / dominant negative; de novo common in severe early-onset",
        "gene_class": (
            "MFN2 encodes Mitofusin 2, a dynamin-related GTPase anchored in the mitochondrial outer membrane "
            "that mediates fusion of mitochondria. MFN2 heterodimerises with MFN1 to tether and fuse "
            "adjacent outer membranes. Beyond organelle morphology, MFN2 regulates axonal mitochondrial "
            "transport via its interaction with the Miro-TRAK trafficking complex, positioning mitochondria "
            "at high energy-demand sites in axons. CMT2A2 pathogenic variants disrupt mitochondrial fusion "
            "and axonal distribution, causing length-dependent axonal degeneration predominantly in the "
            "longest motor and sensory axons. Key clinical features distinguishing CMT2A from other CMT2: "
            "early childhood onset (age 5–10 yr in many), upper limb involvement disproportionate to lower "
            "limb (highly unusual for CMT), optic atrophy in ~15% (mandating annual visual acuity and "
            "fundoscopy), and pyramidal features in some. Wheelchair dependence by the third decade is "
            "reported in up to 30% of severely affected individuals. De novo variants are common in "
            "sporadic early-onset severe cases. NCV is normal or mildly reduced (axonal pattern)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("MFN2 missense GTPase domain (R94Q, R274Q hotspots) — dominant negative, most severe", 0.60),
            ("MFN2 missense coiled-coil/HR2 domain — fusion interface disruption, moderate", 0.22),
            ("MFN2 frameshift/nonsense — haploinsufficiency, moderate", 0.10),
            ("MFN2 de novo missense — severe early childhood, sporadic presentation", 0.08),
        ],
        "age_onset_years_range": (5, 25),
        "sex_ratio_M": 0.50,
        "rates": {
            "upper_limb_involvement_prominent":        0.65,
            "lower_limb_weakness":                     0.90,
            "optic_atrophy_annual_check":              0.15,
            "pyramidal_signs_brisk_reflexes":          0.20,
            "axonal_ncv_normal_or_mild":               0.90,
            "pes_cavus":                               0.55,
            "wheelchair_by_age_30":                    0.30,
            "de_novo_mutation_sporadic":               0.25,
            "scoliosis":                               0.30,
            "hearing_loss":                            0.05,
        },
        "critical_alerts": [
            "OPTIC ATROPHY 15% — annual ophthalmology + visual acuity + fundoscopy MANDATORY — Kjer-like",
            "UPPER LIMB > LOWER LIMB — unusual for CMT — MFN2 CMT2A2 — ensure MFN2 in axonal CMT panel",
            "DE NOVO common in severe early-onset — trio sequencing parent-proband recommended",
            "WHEELCHAIR RISK high — physiotherapy + AFO + standing frame programme from diagnosis",
        ],
        "key_ddx_rules": [
            "CMT2 early onset + upper limb > lower + optic atrophy → MFN2 CMT2A first",
            "CMT2 + de novo + sporadic → MFN2 > NEFL > GDAP1 AR (check consanguinity)",
            "MFN2 vs GDAP1 AR: consanguinity → GDAP1 AR; de novo sporadic → MFN2",
            "Optic atrophy + CMT2 → MFN2 OR Hereditary Optic Neuropathy panel (OPA1, LHON) — sequence both",
        ],
    },

    # ── GDAP1 — CMT4A / CMT2K ──
    {
        "gene": "GDAP1",
        "protein": "Ganglioside-Induced Differentiation-Associated Protein 1 — Mitochondrial Fission/Apoptosis Regulator",
        "alias": (
            "GDAP1; OMIM gene 606598; CMT4A OMIM 214400; CMT2K OMIM 607831; 8q21.11; 358 aa; ~42 kDa; "
            "GDAP1 — mitochondrial outer/inner membrane protein regulating mitochondrial fission (with DRP1); "
            "CMT4A: most severe AR CMT, early childhood onset 1-3 yr, often wheelchair by 10-20 yr; "
            "Vocal cord paresis 30% PATHOGNOMONIC for CMT4A — hoarseness, weak cry, stridor; "
            "Diaphragm involvement — respiratory monitoring mandatory — sleep study + PFTS; "
            "CMT2K: milder AR allelic (homozygous mild alleles or compound heterozygous with mild); "
            "Consanguinity common — homozygous in Mediterranean, Turkish, Spanish, Romani populations; "
            "AD GDAP1 (CMT2K AD) rare — heterozygous dominant; usually milder adult onset"
        ),
        "aa": "358 aa",
        "kDa": "~42 kDa",
        "locus": "8q21.11",
        "omim_gene": 606598,
        "omim_disease": 214400,
        "inheritance": "AR biallelic (CMT4A severe / CMT2K milder); AD rare heterozygous (CMT2K AD)",
        "gene_class": (
            "GDAP1 encodes Ganglioside-Induced Differentiation-Associated Protein 1, a member of the "
            "glutathione S-transferase superfamily localised to the mitochondrial outer and inner membranes. "
            "GDAP1 promotes mitochondrial fragmentation (fission) by interacting with the dynamin-related "
            "fission machinery (DRP1/FIS1 complex). Loss of GDAP1 disrupts axonal mitochondrial dynamics, "
            "impairing energy delivery to distal motor and sensory axons. CMT4A (biallelic LOF) is the "
            "most severe autosomal recessive CMT, with onset at age 1–3 years, rapid progression, "
            "wheelchair dependence frequently by the second decade. The cardinal distinguishing feature is "
            "vocal cord paresis, present in ~30% of CMT4A patients and virtually pathognomonic among "
            "inherited neuropathies. Diaphragmatic involvement necessitates regular pulmonary function "
            "testing and sleep studies. CMT2K represents milder allelic disease. GDAP1 variants cluster "
            "in Mediterranean, Turkish, Romani, and Spanish populations due to founder effects. "
            "Consanguinity is present in >60% of AR CMT4A pedigrees."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("GDAP1 homozygous frameshift/nonsense — CMT4A severe, consanguinity typical", 0.50),
            ("GDAP1 compound heterozygous missense + LOF — CMT4A moderately severe", 0.25),
            ("GDAP1 homozygous missense mild allele — CMT2K, later onset milder course", 0.15),
            ("GDAP1 heterozygous dominant missense — CMT2K AD rare, adult onset", 0.07),
            ("GDAP1 large deletion — MLPA required, rare", 0.03),
        ],
        "age_onset_years_range": (1, 10),
        "sex_ratio_M": 0.50,
        "rates": {
            "vocal_cord_paresis_hoarseness":           0.30,
            "diaphragm_involvement_dyspnoea":          0.20,
            "wheelchair_by_age_20":                    0.45,
            "consanguinity_in_family":                 0.65,
            "demyelinating_or_axonal_pattern":         0.80,
            "distal_weakness_severe":                  0.90,
            "pes_cavus":                               0.70,
            "areflexia":                               0.85,
            "respiratory_insufficiency_sleep_study":   0.20,
            "scoliosis":                               0.35,
        },
        "critical_alerts": [
            "VOCAL CORD PARESIS 30% — pathognomonic CMT4A — ENT laryngoscopy MANDATORY at diagnosis",
            "RESPIRATORY MONITORING MANDATORY — PFTs 6-monthly + sleep study — diaphragm involvement 20%",
            "VINCRISTINE ABSOLUTE CI — flag prominently",
            "CONSANGUINITY: both parents obligate carriers — 25% recurrence — genetic counselling mandatory",
        ],
        "key_ddx_rules": [
            "AR CMT + vocal cord paresis → GDAP1 CMT4A first — pathognomonic combination",
            "Severe AR CMT childhood onset + consanguinity → GDAP1 + SH3TC2 simultaneously",
            "CMT4A vs CMT4C (SH3TC2): scoliosis>60% favours SH3TC2; vocal cord palsy favours GDAP1",
            "Diaphragm + neuropathy → GDAP1 OR TTR amyloidosis (adult) — check family history and age",
        ],
    },

    # ── SH3TC2 — CMT4C ──
    {
        "gene": "SH3TC2",
        "protein": "SH3 Domain and Tetratricopeptide Repeats 2 — Schwann Cell Endosomal Recycling Scaffold",
        "alias": (
            "SH3TC2; OMIM gene 608206; CMT4C OMIM 601596; 5q32; 1288 aa; ~143 kDa; "
            "SH3TC2 — Schwann cell perinuclear endosomal recycling scaffold (RAB11-positive compartment); "
            "CMT4C: most common AR CMT worldwide — especially frequent in Romani/Gypsy (R954W founder), "
            "Mediterranean, Turkish, Eastern European populations; "
            "Moderately severe demyelinating — NCV <30 m/s; onset childhood-adolescence; "
            "Scoliosis 60% HIGHLY CHARACTERISTIC — spine surveillance MANDATORY from diagnosis; "
            "Cranial nerve involvement: facial palsy, sensorineural hearing loss, tongue atrophy; "
            "R954W homozygous — most common Romani founder; compound heterozygous also common"
        ),
        "aa": "1288 aa",
        "kDa": "~143 kDa",
        "locus": "5q32",
        "omim_gene": 608206,
        "omim_disease": 601596,
        "inheritance": "AR biallelic — homozygous (R954W Romani founder) or compound heterozygous",
        "gene_class": (
            "SH3TC2 encodes a large scaffold protein localised to the perinuclear endosomal recycling "
            "compartment (RAB11-positive) in myelinating Schwann cells. SH3TC2 contains an N-terminal "
            "SH3 domain, multiple tetratricopeptide repeats (TPRs), and a C-terminal SH3 domain. "
            "It interacts with NDRG1 and the RAB11/recycling endosome machinery to regulate membrane "
            "recycling in Schwann cells during myelination. Biallelic LOF causes CMT4C, the most "
            "prevalent autosomal recessive CMT worldwide. The Romani/Gypsy founder variant R954W "
            "(c.2860C>T) accounts for the majority of CMT4C cases in Romani populations and is a "
            "targeted first test in at-risk communities. CMT4C causes a moderately severe demyelinating "
            "neuropathy with childhood-to-adolescent onset, onion bulbs and basal lamina onion bulbs "
            "on nerve biopsy. Two clinical features highly characteristic of CMT4C and distinguishing it "
            "from other AR CMTs: (1) scoliosis in >60% of patients, often requiring surgical intervention; "
            "(2) cranial nerve involvement — facial palsy, tongue wasting, sensorineural deafness. "
            "These features mandate spine surveillance from diagnosis and audiological assessment."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("SH3TC2 R954W homozygous — Romani founder, most common CMT4C allele", 0.40),
            ("SH3TC2 compound heterozygous R954W + second allele — common in mixed ancestry", 0.30),
            ("SH3TC2 compound heterozygous novel + LOF — non-Romani populations", 0.20),
            ("SH3TC2 homozygous frameshift/nonsense — consanguinity non-Romani", 0.10),
        ],
        "age_onset_years_range": (5, 20),
        "sex_ratio_M": 0.50,
        "rates": {
            "scoliosis_highly_characteristic":         0.60,
            "cranial_nerve_involvement":               0.35,
            "facial_palsy":                            0.20,
            "sensorineural_hearing_loss":              0.25,
            "tongue_atrophy_wasting":                  0.15,
            "demyelinating_ncv_less_30ms":             0.80,
            "onion_bulb_nerve_biopsy":                 0.70,
            "distal_weakness_lower":                   0.85,
            "pes_cavus":                               0.70,
            "wheelchair_by_age_40":                    0.25,
        },
        "critical_alerts": [
            "SCOLIOSIS 60% — spine X-ray at diagnosis + 6-monthly surveillance — surgical referral if Cobb >40°",
            "CRANIAL NERVE PALSIES — facial/tongue — ENT + audiology at diagnosis; annual follow-up",
            "R954W Romani founder: targeted single-site test first before full panel if Romani ancestry",
            "VINCRISTINE ABSOLUTE CI — all AR CMT patients including CMT4C",
        ],
        "key_ddx_rules": [
            "AR CMT + scoliosis >60% + cranial nerve → SH3TC2 CMT4C first",
            "CMT4C vs CMT4A (GDAP1): vocal cord palsy → GDAP1; scoliosis + cranial → SH3TC2",
            "Romani ancestry + AR neuropathy → R954W targeted test before full panel",
            "Severe scoliosis + AR CMT → SH3TC2 + spine surgery referral simultaneously",
        ],
    },

    # ── NEFL — CMT2E / CMT1F ──
    {
        "gene": "NEFL",
        "protein": "Neurofilament Light Chain (NF-L) — Axonal Cytoskeleton Triplet Intermediate Filament",
        "alias": (
            "NEFL; NF-L; OMIM gene 162280; CMT2E OMIM 607684; CMT1F OMIM 607734; 8p21.2; 543 aa; ~68 kDa; "
            "Neurofilament Light Chain — most abundant neurofilament subunit; axonal cytoskeleton triplet (NF-L/NF-M/NF-H); "
            "CMT2E: AD axonal — neurofilament aggregation in axons — giant axon changes on biopsy; "
            "CMT1F: AD demyelinating intermediate NCV — same gene, different variants/mechanism; "
            "Onset childhood to early adult; variable severity including proximal weakness; "
            "Facial and bulbar involvement possible (unusual for typical CMT); "
            "Serum NF-L elevated — biomarker of axonal damage; "
            "P8R/Q333P dominant missense — inhibit NF-L polymerisation into filament triplet"
        ),
        "aa": "543 aa",
        "kDa": "~68 kDa",
        "locus": "8p21.2",
        "omim_gene": 162280,
        "omim_disease": 607684,
        "inheritance": "AD dominant negative (CMT2E axonal) / AD demyelinating intermediate (CMT1F)",
        "gene_class": (
            "NEFL encodes Neurofilament Light Chain (NF-L), the obligate backbone subunit of the axonal "
            "neurofilament (NF) triplet. NF-L co-assembles with NF-M and NF-H to form 10 nm intermediate "
            "filaments running the length of axons, providing structural support and regulating axon "
            "diameter (and thus conduction velocity). Pathogenic NEFL missense variants act as dominant "
            "negatives, disrupting NF-L polymerisation and causing aggregation of NF-L protein within "
            "axons. The resulting 'giant axon' changes (abnormally large neurofilament-packed axons) are "
            "visible on nerve biopsy. Clinically, NEFL causes two overlapping syndromes: CMT2E (axonal, "
            "NCV normal or mildly reduced) and CMT1F (demyelinating with intermediate NCV). Onset is "
            "typically childhood or early adulthood. Unlike most CMT, NEFL neuropathy may involve facial "
            "and bulbar muscles, and proximal weakness is occasionally present. Serum NF-L is an "
            "established biomarker of axonal damage and is elevated in NEFL CMT2E, potentially useful "
            "for monitoring disease progression and treatment response in future trials."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("NEFL dominant missense (P8R, Q333P, E396K) — NF-L polymerisation disruption, CMT2E", 0.60),
            ("NEFL dominant missense intermediate NCV domain — CMT1F demyelinating-like", 0.20),
            ("NEFL AR biallelic — homozygous or compound het — severe early-onset", 0.12),
            ("NEFL nonsense dominant — haploinsufficiency — milder phenotype", 0.08),
        ],
        "age_onset_years_range": (5, 30),
        "sex_ratio_M": 0.50,
        "rates": {
            "axonal_ncv_cmt2e":                        0.60,
            "intermediate_ncv_cmt1f":                  0.30,
            "giant_axon_nerve_biopsy":                 0.55,
            "serum_nfl_elevated_biomarker":            0.80,
            "facial_bulbar_involvement":               0.20,
            "proximal_weakness":                       0.15,
            "distal_weakness_lower":                   0.85,
            "pes_cavus":                               0.65,
            "areflexia":                               0.70,
            "wheelchair_by_age_40":                    0.15,
        },
        "critical_alerts": [
            "FACIAL/BULBAR INVOLVEMENT — unusual for CMT — NEFL CMT2E/CMT1F — bulbar screen at each visit",
            "SERUM NF-L elevated — biomarker — useful for monitoring progression and treatment trials",
            "GIANT AXON on nerve biopsy → NEFL top differential (also GAN); order NEFL + GAN sequencing",
            "VINCRISTINE ABSOLUTE CI",
        ],
        "key_ddx_rules": [
            "Giant axon nerve biopsy + AD neuropathy → NEFL CMT2E vs GAN (giant axon neuropathy AR)",
            "CMT + facial palsy + proximal weakness → NEFL vs SH3TC2 (AR) — check inheritance pattern",
            "Elevated serum NF-L + CMT → NEFL CMT2E biomarker confirmation — useful for trial enrolment",
            "CMT1F intermediate NCV + AD → NEFL vs GJB1 CMTX1 — check sex-linkage pattern",
        ],
    },

    # ── EGR2 — CMT1D / CMT4E ──
    {
        "gene": "EGR2",
        "protein": "Early Growth Response 2 (Krox20) — Zinc Finger Transcription Factor Master Peripheral Myelination",
        "alias": (
            "EGR2; KROX20; OMIM gene 129010; CMT1D OMIM 607678; CMT4E OMIM 605253; 10q21.2; 472 aa; ~52 kDa; "
            "EGR2/Krox20 — zinc finger transcription factor essential for peripheral nervous system myelination; "
            "Master regulator of Schwann cell myelination program — activates PMP22, MPZ, MBP, PRX; "
            "CMT1D: AD severe demyelinating or hypomyelinating — NCV near 0-5 m/s — congenital/infantile onset; "
            "CMT4E: AR — most severe, congenital hypomyelinating neuropathy (CHN) — NCV 0-10 m/s — arthrogryposis; "
            "R359W / R381H zinc finger hotspot — dominant negative on myelin gene activation; "
            "Congenital onset: hypotonia, respiratory failure, arthrogryposis — NICU presentation; "
            "CSF protein elevated; nerve biopsy shows hypomyelination or onion bulbs"
        ),
        "aa": "472 aa",
        "kDa": "~52 kDa",
        "locus": "10q21.2",
        "omim_gene": 129010,
        "omim_disease": 607678,
        "inheritance": "AD dominant negative zinc finger (CMT1D); AR biallelic (CMT4E CHN severe); congenital onset common",
        "gene_class": (
            "EGR2 (Early Growth Response 2, also known as Krox20) encodes a zinc finger transcription "
            "factor that is the master regulator of peripheral nervous system myelination. During Schwann "
            "cell development, EGR2 is upregulated as promyelinating Schwann cells transition to "
            "myelinating Schwann cells, activating the entire myelin gene programme including PMP22, "
            "MPZ (P0), MBP, PRX (periaxin), and CDKN1C. EGR2 loss therefore causes global failure of "
            "myelin gene transcription. CMT1D pathogenic variants (AD) cluster in the zinc finger "
            "DNA-binding domain (R359W, R381H hotspots) and act as dominant negatives by competing "
            "with wild-type EGR2 for DNA binding without activating transcription. The resulting "
            "phenotype is severe hypomyelinating or demyelinating neuropathy with NCV often <5 m/s, "
            "presenting in infancy or even at birth with hypotonia, respiratory failure, and arthrogryposis. "
            "AR CMT4E (biallelic EGR2 LOF) represents the most severe congenital hypomyelinating "
            "neuropathy (CHN), sometimes requiring ventilatory support from birth. Nerve biopsy shows "
            "virtually absent myelin sheaths (hypomyelination) or severe onion bulb formation. CSF "
            "protein is markedly elevated. EGR2 sequencing is essential in any CMT with NCV <10 m/s."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("EGR2 R359W dominant negative zinc finger — CMT1D severe hypomyelination, most common AD", 0.40),
            ("EGR2 R381H dominant negative zinc finger — CMT1D demyelinating, second hotspot", 0.25),
            ("EGR2 biallelic LOF — CMT4E AR congenital hypomyelinating — most severe", 0.20),
            ("EGR2 other zinc finger missense — dominant negative, variable severity", 0.10),
            ("EGR2 frameshift — haploinsufficiency, milder demyelinating CMT1D", 0.05),
        ],
        "age_onset_years_range": (0, 5),
        "sex_ratio_M": 0.50,
        "rates": {
            "congenital_or_infantile_onset":           0.70,
            "ncv_less_10ms_hypomyelination":           0.80,
            "arthrogryposis_congenital":               0.30,
            "respiratory_failure_neonatal":            0.25,
            "csf_protein_elevated":                    0.75,
            "hypomyelination_nerve_biopsy":            0.70,
            "onion_bulb_formation_severe":             0.55,
            "distal_weakness_severe":                  0.90,
            "scoliosis":                               0.40,
            "wheelchair_dependent":                    0.55,
        },
        "critical_alerts": [
            "NCV <10 m/s in infant/child → EGR2 and MPZ MANDATORY — congenital hypomyelinating neuropathy",
            "RESPIRATORY FAILURE — neonatal/infantile — ventilatory support may be needed from birth — NICU alert",
            "ARTHROGRYPOSIS + neonatal hypotonia + neuropathy → EGR2 CMT4E AR — genetic emergency",
            "R359W / R381H dominant negative hotspots — targeted sequencing confirms in >60% of CMT1D",
        ],
        "key_ddx_rules": [
            "NCV <10 m/s infant + hypomyelination + congenital → EGR2 (CMT1D/4E) + MPZ (CMT1B) + PRX simultaneously",
            "CMT1D vs CMT4E: AD family history → CMT1D; consanguinity + AR → CMT4E; de novo → CMT1D possible",
            "Arthrogryposis + neuropathy → EGR2 CMT4E vs LMNA AR vs PIEZO2 — multi-gene panel",
            "CSF protein markedly elevated + childhood neuropathy → EGR2 / MPZ / CIDP exclusion",
        ],
    },
]


def _make_patients(gene_data: dict) -> list:
    rng = random.Random(gene_data["seed"])
    patients = []
    rates = gene_data["rates"]
    for i in range(gene_data["n_patients"]):
        pid = f"HMSN-{gene_data['gene']}-{gene_data['seed']:04d}-{i+1:02d}"
        age_lo, age_hi = gene_data["age_onset_years_range"]
        onset = round(rng.uniform(age_lo, age_hi), 1)
        male = rng.random() < gene_data["sex_ratio_M"]
        etiology, _ = rng.choices(
            gene_data["etiologies"],
            weights=[w for _, w in gene_data["etiologies"]],
        )[0]
        features = {k: (rng.random() < v) for k, v in rates.items()}
        patients.append({
            "patient_id": pid,
            "gene": gene_data["gene"],
            "age_onset_years": onset,
            "sex": "M" if male else "F",
            "etiology": etiology,
            **features,
        })
    return patients


def _aggregate(gene_data: dict, patients: list) -> dict:
    n = len(patients)
    rates_pct = {}
    for k in gene_data["rates"]:
        rates_pct[k] = round(sum(1 for p in patients if p.get(k)) / n * 100, 1)
    dominant_etio = gene_data["etiologies"][0][0]
    return {
        "gene": gene_data["gene"],
        "protein": gene_data["protein"],
        "locus": gene_data["locus"],
        "aa": gene_data["aa"],
        "inheritance": gene_data["inheritance"],
        "omim_gene": gene_data["omim_gene"],
        "omim_disease": gene_data["omim_disease"],
        "n_patients": n,
        "gene_class": gene_data["gene_class"],
        "critical_alerts": gene_data["critical_alerts"],
        "key_ddx_rules": gene_data["key_ddx_rules"],
        "phenotype_rates": rates_pct,
        "etiologies": [
            {"label": lbl, "pct": round(w * 100, 1)}
            for lbl, w in gene_data["etiologies"]
        ],
        "dominant_etiology": dominant_etio,
    }


def get_overview() -> dict:
    all_patients = []
    gene_summaries = []
    for g in HMSN_GENES:
        pts = _make_patients(g)
        all_patients.extend(pts)
        gene_summaries.append(_aggregate(g, pts))

    n = len(all_patients)

    def pct(key):
        return round(sum(1 for p in all_patients if p.get(key)) / n * 100, 1)

    # Cross-gene aggregate stats
    agg = {
        "pes_cavus_any_gene":               pct("pes_cavus"),
        "distal_weakness_lower":            pct("distal_lower_limb_weakness") if any(p.get("distal_lower_limb_weakness") is not None for p in all_patients) else pct("distal_weakness_lower"),
        "demyelinating_ncv":                pct("demyelinating_ncv_less_20ms_cmtib") + pct("ncv_less_than_38_ms_median") if any(p.get("ncv_less_than_38_ms_median") is not None for p in all_patients) else pct("demyelinating_ncv_less_30ms"),
        "axonal_ncv_pattern":               pct("axonal_ncv_cmt2e") if any(p.get("axonal_ncv_cmt2e") is not None for p in all_patients) else pct("axonal_ncv_normal_or_mild"),
        "intermediate_ncv_cmtx1":           pct("intermediate_ncv_30_40ms_males"),
        "optic_atrophy_mfn2":               pct("optic_atrophy_annual_check"),
        "vocal_cord_paresis_gdap1":         pct("vocal_cord_paresis_hoarseness"),
        "scoliosis_any_gene":               pct("scoliosis"),
        "cranial_nerve_involvement":        pct("cranial_nerve_involvement"),
        "congenital_onset_egr2":            pct("congenital_or_infantile_onset"),
        "wheelchair_risk_combined":         max(pct("wheelchair_by_age_40"), pct("wheelchair_by_age_20"), pct("wheelchair_by_age_30"), pct("wheelchair_dependent")),
    }

    # Fix: some keys might return 0 if key doesn't exist in patients
    # Safer aggregate computation
    def safe_pct(keys_list):
        """Return pct of patients where ANY of the keys is True."""
        cnt = sum(1 for p in all_patients if any(p.get(k) for k in keys_list))
        return round(cnt / n * 100, 1)

    agg_safe = {
        "pes_cavus_any_gene":               safe_pct(["pes_cavus"]),
        "distal_weakness_lower_any_gene":   safe_pct(["distal_lower_limb_weakness", "distal_weakness_lower", "distal_weakness_severe"]),
        "demyelinating_ncv_any_gene":       safe_pct(["ncv_less_than_38_ms_median", "demyelinating_ncv_less_20ms_cmtib", "demyelinating_ncv_less_30ms", "ncv_less_10ms_hypomyelination"]),
        "axonal_ncv_any_gene":              safe_pct(["axonal_ncv_cmt2e", "axonal_ncv_normal_or_mild", "axonal_ncv_normal_cmt2ij"]),
        "intermediate_ncv_cmtx1":           safe_pct(["intermediate_ncv_30_40ms_males"]),
        "optic_atrophy_mfn2":               safe_pct(["optic_atrophy_annual_check"]),
        "vocal_cord_paresis_gdap1":         safe_pct(["vocal_cord_paresis_hoarseness"]),
        "scoliosis_any_gene":               safe_pct(["scoliosis", "scoliosis_highly_characteristic"]),
        "cranial_nerve_involvement":        safe_pct(["cranial_nerve_involvement"]),
        "congenital_onset_egr2_gdap1":      safe_pct(["congenital_or_infantile_onset"]),
        "wheelchair_risk_combined":         safe_pct(["wheelchair_by_age_40", "wheelchair_by_age_20", "wheelchair_by_age_30", "wheelchair_dependent"]),
    }

    top_alerts = [
        "MLPA MANDATORY for PMP22 (CMT1A duplication / HNPP deletion) — sequencing alone misses >99%",
        "VINCRISTINE ABSOLUTE CI in ALL CMT patients regardless of subtype — document on every clinical record",
        "INTERMEDIATE NCV 30-40 m/s in male → GJB1/CMTX1 first — NO male-to-male transmission = X-linked",
        "VOCAL CORD PARESIS → GDAP1 CMT4A — pathognomonic — ENT laryngoscopy mandatory at diagnosis",
        "NCV <10 m/s infant → EGR2 + MPZ STAT — congenital hypomyelinating neuropathy — respiratory alert",
        "OPTIC ATROPHY 15% MFN2 CMT2A — annual ophthalmology mandatory — Kjer-like progressive",
        "SCOLIOSIS 60% SH3TC2 CMT4C — spine surveillance mandatory — surgical referral Cobb >40°",
        "RESPIRATORY MONITORING GDAP1/EGR2 — PFTs + sleep study — diaphragm + intercostal involvement",
        "CMTX1 CNS white matter lesions — transient stroke-like episodes — NOT ischaemic stroke — supportive",
        "CASCADE TESTING: all first-degree relatives of AD CMT — NCV is the most sensitive presymptomatic test",
    ]

    return {
        "title": "Hereditary HMSN Atlas — Complete 8-Gene CMT/HMSN Reference",
        "subtitle": "320-Patient Aggregate (8×40, seeds 1462–1469) — PMP22 · MPZ · GJB1 · MFN2 · GDAP1 · SH3TC2 · NEFL · EGR2",
        "n_total": n,
        "genes": [g["gene"] for g in HMSN_GENES],
        "aggregate_stats": agg_safe,
        "top_alerts": top_alerts,
        "gene_summaries": gene_summaries,
    }


def get_breakdown() -> dict:
    result = {}
    for g in HMSN_GENES:
        pts = _make_patients(g)
        result[g["gene"]] = _aggregate(g, pts)
    return result


def get_definitions() -> dict:
    return {
        "definitions": [
            {
                "term": "Charcot-Marie-Tooth Disease (CMT) / Hereditary Motor and Sensory Neuropathy (HMSN)",
                "definition": (
                    "Charcot-Marie-Tooth disease (CMT), also termed Hereditary Motor and Sensory Neuropathy "
                    "(HMSN), is the most common inherited peripheral neuropathy, with a prevalence of ~1:2500. "
                    "CMT encompasses a clinically and genetically heterogeneous group of disorders sharing a "
                    "core phenotype: progressive distal limb weakness and wasting, sensory loss in a 'glove "
                    "and stocking' distribution, reduced or absent deep tendon reflexes, pes cavus foot "
                    "deformity, and slowed nerve conduction velocities. Classification is electrophysiological: "
                    "CMT1 (demyelinating, NCV <38 m/s median nerve), CMT2 (axonal, NCV >38 m/s with reduced "
                    "amplitude), CMTX (X-linked, intermediate NCV 30–40 m/s in males), and CMT4 (AR "
                    "demyelinating). >100 causative genes have been identified, but PMP22 duplication (CMT1A) "
                    "accounts for ~50% of all CMT. HMSN classification (Harding-Thomas) uses Roman numerals "
                    "I–VII based on electrophysiology and additional features."
                ),
            },
            {
                "term": "Demyelinating vs Axonal CMT — Electrophysiology Branch-Point",
                "definition": (
                    "Nerve conduction velocity (NCV) of the median motor nerve is the critical branch-point "
                    "in CMT classification: NCV <38 m/s = demyelinating CMT1 (PMP22, MPZ, EGR2, GJB1 CN "
                    "intermediate, SH3TC2); NCV >38 m/s with reduced CMAP amplitude = axonal CMT2 (MFN2, "
                    "NEFL, GDAP1 CMT2K); NCV 30–40 m/s in males with X-linked pattern = CMTX1 (GJB1). "
                    "In demyelinating CMT, slowing is UNIFORM across all fibres (all nerves, both sides) "
                    "— this distinguishes genetic from acquired inflammatory neuropathy (CIDP, MMN) which "
                    "shows focal, non-uniform, asymmetric slowing with conduction block. This electrophysio- "
                    "logical distinction is critical: CIDP is treatable with immunosuppression; CMT is not, "
                    "and steroids are ineffective and potentially harmful."
                ),
            },
            {
                "term": "MLPA — Multiplex Ligation-Dependent Probe Amplification for PMP22",
                "definition": (
                    "MLPA (Multiplex Ligation-Dependent Probe Amplification) is the gold-standard diagnostic "
                    "test for PMP22 copy number variants: duplication (CMT1A, 3 copies) and deletion (HNPP, "
                    "1 copy). Standard Sanger sequencing and next-generation sequencing (NGS) panels cannot "
                    "detect copy number changes reliably; MLPA directly quantifies relative dosage of each "
                    "exon. Since PMP22 duplication accounts for ~70–80% of CMT1 and deletion causes HNPP, "
                    "MLPA must be ordered BEFORE or simultaneously with gene panel sequencing in any patient "
                    "with suspected CMT1 (NCV <38 m/s). Failure to order MLPA is the most common diagnostic "
                    "delay in CMT clinical practice. Arrays (chromosomal microarray, SNP arrays) can also "
                    "detect the 1.4 Mb CMT1A duplication and are increasingly used in diagnostic pipelines."
                ),
            },
            {
                "term": "Vincristine Absolute Contraindication — All CMT Subtypes",
                "definition": (
                    "Vincristine (a vinca alkaloid used in cancer chemotherapy) is ABSOLUTELY CONTRAINDICATED "
                    "in ALL patients with CMT regardless of subtype, including asymptomatic gene carriers. "
                    "Vincristine disrupts microtubule polymerisation by binding tubulin, critically impairing "
                    "axonal transport. In a peripheral nervous system already compromised by genetic CMT, "
                    "vincristine causes catastrophic, rapidly progressive, often irreversible motor neuropathy — "
                    "multiple deaths and severe permanent paralysis have been documented. Every CMT patient's "
                    "electronic record must carry a permanent allergy/contraindication flag for vincristine. "
                    "The oncology team must be notified at the time of cancer diagnosis. Alternative "
                    "vincristine-free regimens should be substituted (consult haematology/oncology). "
                    "Other neurotoxic agents requiring caution: taxanes, bortezomib, thalidomide, amiodarone, "
                    "nitrofurantoin, high-dose pyridoxine — flag and discuss risk-benefit with neurology."
                ),
            },
            {
                "term": "CMTX1 — Intermediate NCV and CNS Involvement (GJB1/Connexin 32)",
                "definition": (
                    "CMTX1 caused by GJB1 (Connexin 32) mutations is the second most common CMT (~10–15% "
                    "of all CMT), X-linked dominant. Hemizygous males show moderate-severe neuropathy with "
                    "nerve conduction velocities in the 'intermediate' range (30–40 m/s) — too slow for "
                    "purely axonal CMT2 but too fast for typical demyelinating CMT1 — making NCV "
                    "electrophysiology the pivotal clue. No male-to-male transmission confirms X-linkage. "
                    "Heterozygous females are typically mildly affected or asymptomatic. A critical "
                    "distinguishing feature: CNS white matter lesions on brain MRI, causing transient "
                    "stroke-like episodes (acute hemiplegia, ataxia, confusion, dysarthria) lasting hours "
                    "to days, with full recovery. These episodes occur in ~20% of affected males, often "
                    "triggered by fever or intercurrent illness. They mimic stroke or demyelinating CNS "
                    "disease; the correct diagnosis avoids inappropriate thrombolysis or immunosuppression."
                ),
            },
            {
                "term": "CMT2A2 — MFN2 Upper Limb Predominance and Optic Atrophy",
                "definition": (
                    "CMT2A2 (MFN2) is the most common axonal CMT, representing ~20% of CMT2 cases. "
                    "Unlike most CMT subtypes where lower limb weakness dominates, CMT2A2 frequently "
                    "involves the upper limbs to a degree disproportionate to lower limb disease — a "
                    "distinguishing clinical signature. Onset is typically childhood to early adolescence. "
                    "Two features demand specific monitoring: (1) optic atrophy in ~15% of patients, "
                    "resembling Kjer disease (dominant optic atrophy), requiring annual visual acuity "
                    "assessment and fundoscopy; (2) pyramidal signs (brisk reflexes, spasticity) in a "
                    "subset. De novo MFN2 variants account for a significant proportion of sporadic "
                    "severe early-onset cases, justifying parental sequencing (trio analysis) in "
                    "apparently sporadic presentations. Wheelchair dependence by the third decade occurs "
                    "in approximately 30% of severely affected individuals."
                ),
            },
            {
                "term": "CMT4A — GDAP1 Vocal Cord Paresis Pathognomonic Feature",
                "definition": (
                    "CMT4A (biallelic GDAP1) is the most severe autosomal recessive CMT, with onset at "
                    "age 1–3 years and frequent wheelchair dependence by the second decade. The pathognomonic "
                    "distinguishing feature among all inherited neuropathies is vocal cord paresis, present "
                    "in ~30% of CMT4A patients. This manifests as hoarseness, weak or breathy voice, weak "
                    "cry in infants, and — critically — risk of aspiration and airway compromise. ENT "
                    "laryngoscopy is mandatory at CMT4A diagnosis to document cord mobility. Diaphragmatic "
                    "involvement (intercostal nerve involvement) causes respiratory insufficiency requiring "
                    "monitoring by pulmonary function tests (PFTs) and overnight sleep studies (OSA/CSA "
                    "risk). GDAP1 pathogenic variants are particularly prevalent in Mediterranean, Turkish, "
                    "and Romani populations due to founder effects. Consanguinity is present in >60% of "
                    "confirmed CMT4A pedigrees."
                ),
            },
            {
                "term": "CMT4C — SH3TC2 Scoliosis and Cranial Nerve Involvement",
                "definition": (
                    "CMT4C (biallelic SH3TC2) is the most globally prevalent autosomal recessive CMT. "
                    "The R954W (c.2860C>T) variant is a Romani/Gypsy founder mutation accounting for the "
                    "majority of CMT4C in populations with Romani ancestry; targeted R954W testing precedes "
                    "full gene panel in at-risk communities. Two features distinguish CMT4C from other AR "
                    "CMTs: (1) Scoliosis in >60% of patients — the highest scoliosis rate of any CMT "
                    "subtype — often severe, requiring orthopaedic spine surveillance from diagnosis and "
                    "spinal instrumentation when Cobb angle exceeds 40°; (2) Cranial nerve involvement — "
                    "facial palsy, tongue atrophy, and sensorineural hearing loss — attributable to "
                    "Schwann cell dysfunction in cranial nerve territories. Audiological assessment at "
                    "diagnosis and ENT follow-up are mandatory. These two features (scoliosis + cranial "
                    "nerve palsies) in AR neuropathy should immediately trigger SH3TC2 sequencing."
                ),
            },
            {
                "term": "Congenital Hypomyelinating Neuropathy (CHN) — EGR2 Spectrum",
                "definition": (
                    "Congenital hypomyelinating neuropathy (CHN) represents the most severe end of the "
                    "demyelinating CMT spectrum, presenting at birth or in infancy with profound hypotonia, "
                    "respiratory failure (often requiring mechanical ventilation from birth), arthrogryposis, "
                    "absent deep tendon reflexes, and minimal or absent NCV (<10 m/s, sometimes unmeasurable). "
                    "EGR2/Krox20 pathogenic variants — dominant (CMT1D, R359W/R381H hotspots) or biallelic "
                    "AR (CMT4E CHN) — account for a significant proportion of CHN cases. Other CHN genes "
                    "include MPZ, PMP22, PRX (periaxin), and CNTNAP1. In CHN, nerve biopsy shows virtual "
                    "absence of myelin sheaths (hypomyelination) or severe onion bulb formation. CSF "
                    "protein is markedly elevated. EGR2 is a priority in any infant with NCV <10 m/s and "
                    "should be sequenced simultaneously with MPZ and PRX. Respiratory management — including "
                    "non-invasive ventilation (NIV) or tracheostomy — is the immediate priority; "
                    "multidisciplinary NICU involvement from birth is required."
                ),
            },
            {
                "term": "Cascade Genetic Testing — Hereditary HMSN / CMT Families",
                "definition": (
                    "Once an index case carries a confirmed pathogenic CMT variant, cascade testing of all "
                    "first-degree relatives is recommended regardless of clinical symptoms, because: (1) "
                    "pes cavus and mild NCV slowing may be the only presymptomatic finding — clinical "
                    "examination misses early disease; (2) asymptomatic variant carriers still carry the "
                    "vincristine CI and may face career/insurance implications requiring informed decision; "
                    "(3) reproductive counselling (recurrence risk 50% AD, 25% AR, variable XLD) enables "
                    "informed family planning. NCV measurement is the most sensitive presymptomatic test "
                    "for demyelinating CMT (uniformly slowed even before weakness develops). For X-linked "
                    "CMTX1 (GJB1): all sons of affected males are unaffected (no male-to-male transmission); "
                    "all daughters are obligate carriers. For CMT4A/C (AR): both parents are obligate "
                    "carriers — sibling recurrence risk 25%. Prenatal diagnosis and preimplantation genetic "
                    "testing (PGT-M) are available for all CMT subtypes in this atlas."
                ),
            },
        ]
    }
