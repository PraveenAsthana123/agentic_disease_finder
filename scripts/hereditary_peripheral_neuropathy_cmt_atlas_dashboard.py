#!/usr/bin/env python3
"""Hereditary-Peripheral-Neuropathy-CMT-Atlas -- Complete 8-Gene Reference
Charcot-Marie-Tooth disease and related hereditary peripheral neuropathies
(PMP22/CMT1A-most-common-60-70pct-duplication-HNPP-deletion,
 MPZ/CMT1B-mutation-class-determines-severity-onion-bulb,
 GJB1/CMT1X-intermediate-NCV-PATHOGNOMONIC-males-severe,
 MFN2/CMT2A-most-common-axonal-optic-atrophy-PATHOGNOMONIC,
 GDAP1/CMT4A-vocal-cord-phrenic-PATHOGNOMONIC,
 SH3TC2/CMT4C-scoliosis-cranial-nerve-PATHOGNOMONIC-R954W,
 PRX/CMT4F-onion-bulb-severe-NCV-lt10-PATHOGNOMONIC,
 NEFL/CMT2E-CMT1F-same-gene-two-phenotypes):

PMP22  (Peripheral myelin protein 22; 160aa; 17p12; AD;
         CMT1A: 17p12 1.4Mb duplication — most common CMT (60-70% of all CMT1);
         HNPP: 17p12 deletion — focal conduction blocks, episodic neuropathy;
         Pes cavus + hammertoe + hyporeflexia = PATHOGNOMONIC triad;
         NO curative therapy; physical therapy + AFOs; avoid nerve compression;
         seed SEED_BASE+0).
MPZ    (Myelin protein zero / P0; 248aa; 1q23.3; AD;
         CMT1B: 2nd most common demyelinating CMT; P0 major myelin structural protein;
         Thr124Met: late-onset (>30yr) mild; Ile107Met: severe early-onset Dejerine-Sottas;
         Mutation class determines severity — TEST MUTATION BEFORE PREDICTING PROGNOSIS;
         Onion bulb formation on nerve biopsy PATHOGNOMONIC for severe mutations;
         seed SEED_BASE+1).
GJB1   (Connexin 32 / Cx32; 283aa; Xq13.1; X-linked dominant;
         CMT1X: most common X-linked CMT; gap junction protein at paranodal regions;
         Males severely affected (NCV 30-40 m/s, intermediate); females mild-moderate (lyonization);
         INTERMEDIATE NCV (25-45 m/s) does NOT fit pure demyelinating OR pure axonal — PATHOGNOMONIC for GJB1;
         Central white matter lesions (MRI) with acute neurological episodes in severe cases;
         seed SEED_BASE+2).
MFN2   (Mitofusin 2; 757aa; 1p36.22; AD;
         CMT2A: most common axonal CMT2; mitochondrial fusion protein;
         Axonal NCV (>38 m/s) with reduced CMAP amplitude;
         Optic atrophy in 20-30% — CMT2A WITH optic atrophy = PATHOGNOMONIC for MFN2;
         Lower limb >> upper limb involvement initially; severe early-onset worse prognosis;
         seed SEED_BASE+3).
GDAP1  (Ganglioside-induced differentiation-associated protein 1; 358aa; 8q21.11; AR;
         CMT4A: most common AR CMT in some populations; axonal or demyelinating;
         Vocal cord paresis and diaphragm/phrenic nerve involvement = PATHOGNOMONIC for GDAP1;
         Early onset (childhood); rapid progression; wheelchair by 2nd-3rd decade AR-null mutations;
         AD form (GOF) milder;
         seed SEED_BASE+4).
SH3TC2 (SH3 domain and tetratricopeptide repeats 2; 1288aa; 5q32; AR;
         CMT4C: most common AR-CMT in Europeans; R954W founder mutation (Roma/European);
         Scoliosis + cranial nerve involvement = PATHOGNOMONIC for SH3TC2-CMT4C;
         Large-caliber myelinated fiber loss; onion bulb formation; progressive hearing loss;
         seed SEED_BASE+5).
PRX    (Periaxin; 1461aa; 19q13.2; AR;
         CMT4F: severe hypomyelinating neuropathy; periaxin maintains myelin compaction;
         Onion bulb formation severe = PATHOGNOMONIC; very low NCV (<10 m/s);
         Early onset walking difficulties; sensory loss predominant;
         Dejerine-Sottas phenotype in severe truncating mutations;
         seed SEED_BASE+6).
NEFL   (Neurofilament light chain; 543aa; 8p21.2; AD;
         CMT2E (axonal) and CMT1F (demyelinating) — SAME GENE, different phenotype classes;
         AD mutations; neurofilament accumulation in axons = PATHOGNOMONIC on nerve biopsy;
         Early onset childhood weakness; tremor common; intermediate NCV in some;
         CSF protein elevated in demyelinating form (CMT1F);
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3822-3829)
"""
import random

SEED_BASE = 3822

ATLAS_GENES = [
    {
        "gene": "PMP22",
        "protein": (
            "PMP22 -- 17p12 AD LOF/GOF -- 160aa -- "
            "Peripheral-myelin-protein-22-major-myelin-component-"
            "CMT1A-duplication-most-common-CMT-60-70pct-"
            "HNPP-deletion-episodic-focal-conduction-blocks-"
            "pes-cavus-hammertoe-hyporeflexia-PATHOGNOMONIC-triad-"
            "NO-curative-therapy-physical-therapy-AFOs"
        ),
        "locus": "17p12",
        "protein_size": (
            "160 aa / ~17.8 kDa / 17p12 PMP22 peripheral myelin protein 22 molecular context: "
            "STRUCTURE: "
            "  160 aa; 17.8 kDa tetraspan membrane protein; major component of PNS myelin; "
            "  Constitutes 2-5% of myelin protein; critical for myelin stability and compaction; "
            "  CMT1A: 1.4 Mb duplication at 17p12 → 3 copies PMP22 → overexpression → dysmyelination; "
            "  HNPP: 17p12 deletion → 1 copy PMP22 → underexpression → focal thickening 'tomaculous'; "
            "DISEASE MECHANISMS: "
            "  CMT1A (duplication): most common CMT worldwide (60-70% of CMT1); "
            "  Slowly progressive distal muscle weakness, beginning in lower limbs; "
            "  NCV demyelinating: <38 m/s (typically 15-30 m/s for CMT1A); "
            "  HNPP (deletion): episodic focal neuropathies at pressure points; "
            "  HNPP: hereditary neuropathy with liability to pressure palsies; "
            "PATHOGNOMONIC FEATURES: "
            "  Pes cavus (high arched foot) + hammertoe + hyporeflexia TRIAD PATHOGNOMONIC; "
            "  Onset: typically 1st-2nd decade, but may be subclinical initially; "
            "  Foot deformity often precedes weakness by years; "
            "  Stork leg appearance (distal muscle wasting) in advanced disease; "
            "TREATMENT: "
            "  NO curative therapy; no disease-modifying treatment approved; "
            "  Physical therapy + AFOs (ankle-foot orthoses) standard; "
            "  Avoid nerve compression (HNPP especially); "
            "  Avoid neurotoxic drugs (vincristine, cisplatin, amiodarone, isoniazid); "
            "  Opicinumab and other remyelination trials ongoing; "
            "DIAGNOSIS: "
            "  MLPA or FISH for 17p12 duplication/deletion; "
            "  Nerve conduction studies: uniform slow NCV"
        ),
        "syndrome": "PMP22-CMT1A-duplication-most-common-60-70pct / HNPP-deletion-episodic / pes-cavus-hammertoe-triad-PATHOGNOMONIC",
        "inheritance": "AD",
        "pathognomonic": "Pes cavus + hammertoe + hyporeflexia TRIAD PATHOGNOMONIC; NCV <38 m/s uniform slowing",
        "key_therapy": "NO curative therapy; physical therapy + AFOs; avoid nerve compression and neurotoxic drugs",
        "critical_rules": [
            "PMP22-DUPLICATION-CMT1A-MOST-COMMON-60-70pct",
            "HNPP-DELETION-EPISODIC-CONDUCTION-BLOCK",
            "PES-CAVUS-HAMMERTOE-HYPOREFLEXIA-TRIAD-PATHOGNOMONIC",
            "NO-CURATIVE-THERAPY-PHYSICAL-THERAPY-AFO",
            "NEUROTOXIC-DRUGS-VINCRISTINE-CISPLATIN-AVOID",
        ],
    },
    {
        "gene": "MPZ",
        "protein": (
            "MPZ -- 1q23.3 AD LOF/GOF -- 248aa -- "
            "Myelin-protein-zero-P0-major-myelin-structural-protein-"
            "CMT1B-2nd-most-common-demyelinating-CMT-"
            "mutation-class-determines-severity-"
            "Thr124Met-late-onset-mild-vs-Ile107Met-severe-Dejerine-Sottas-"
            "onion-bulb-severe-mutations-PATHOGNOMONIC"
        ),
        "locus": "1q23.3",
        "protein_size": (
            "248 aa / ~28 kDa / 1q23.3 MPZ myelin protein zero molecular context: "
            "STRUCTURE: "
            "  248 aa; 28 kDa single-pass transmembrane glycoprotein; "
            "  Major structural protein of PNS myelin (~50% of peripheral myelin protein); "
            "  Mediates myelin membrane compaction via homophilic adhesion; "
            "  MPZ LOF or dominant-negative → impaired compaction → demyelination; "
            "MUTATION-CLASS-DETERMINES-SEVERITY (CRITICAL): "
            "  Early-onset severe: p.Ile107Met, p.Asp61Asn → Dejerine-Sottas phenotype; "
            "  NCV: <10 m/s; congenital hypotonia; wheelchair early; "
            "  Late-onset mild: p.Thr124Met → onset >30 yr; NCV 20-35 m/s; "
            "  Adult-onset: p.Ser44Phe, p.Arg98His → relatively preserved function; "
            "  Connexin 32-like MPZ mutations: intermediate severity; "
            "  MANDATORY: genotype before predicting prognosis; "
            "PATHOGNOMONIC FEATURES: "
            "  Onion bulb formation on nerve biopsy: PATHOGNOMONIC for severe MPZ mutations; "
            "  Redundant Schwann cell processes forming concentric 'onion bulb' layers; "
            "  Reflects repeated cycles of demyelination/remyelination; "
            "DIAGNOSIS: "
            "  NCS: demyelinating CMT1B (NCV <38 m/s); "
            "  Nerve biopsy if severe — onion bulbs confirm demyelinating CMT; "
            "  Sequence MPZ after PMP22 excluded"
        ),
        "syndrome": "MPZ-CMT1B-demyelinating / mutation-class-determines-severity / Thr124Met-late-mild / onion-bulb-severe-PATHOGNOMONIC",
        "inheritance": "AD",
        "pathognomonic": "Onion bulb formation on nerve biopsy PATHOGNOMONIC for severe MPZ mutations; Thr124Met = late-onset mild",
        "key_therapy": "NO curative therapy; mutation genotype MANDATORY before prognosis; physical therapy + AFOs",
        "critical_rules": [
            "MPZ-MUTATION-CLASS-DETERMINES-SEVERITY",
            "THR124MET-LATE-ONSET-MILD",
            "ONION-BULB-SEVERE-MUTATIONS-PATHOGNOMONIC",
            "ILE107MET-SEVERE-EARLY-ONSET-DEJERINE-SOTTAS",
            "GENOTYPE-MANDATORY-BEFORE-PREDICTING-PROGNOSIS",
        ],
    },
    {
        "gene": "GJB1",
        "protein": (
            "GJB1 -- Xq13.1 X-linked-dominant LOF -- 283aa -- "
            "Connexin-32-Cx32-gap-junction-protein-paranodal-regions-"
            "CMT1X-most-common-X-linked-CMT-"
            "intermediate-NCV-25-45ms-PATHOGNOMONIC-NOT-pure-demyelinating-NOT-pure-axonal-"
            "males-severely-affected-females-mild-lyonization-"
            "central-white-matter-lesions-acute-neurological-episodes"
        ),
        "locus": "Xq13.1",
        "protein_size": (
            "283 aa / ~31 kDa / Xq13.1 GJB1 connexin 32 molecular context: "
            "STRUCTURE: "
            "  283 aa; 31 kDa gap junction protein (connexin family); "
            "  Forms intercellular channels at paranodal regions of peripheral myelin; "
            "  Allows diffusion of ions, metabolites within Schwann cell layers; "
            "  GJB1 LOF → impaired Schwann cell communication → axonopathy + demyelination; "
            "PATHOGNOMONIC NCV PATTERN: "
            "  INTERMEDIATE NCV (25-45 m/s): does NOT fit pure demyelinating (<38 m/s) or pure axonal; "
            "  This intermediate pattern is PATHOGNOMONIC for GJB1/CMT1X; "
            "  Results from combined axonal + demyelinating pathology; "
            "  CRITICAL: if NCV is intermediate and male → test GJB1 FIRST; "
            "X-LINKED SEVERITY DIFFERENCE: "
            "  Males (hemizygous): severely affected; NCV 30-40 m/s; wheelchair in some; "
            "  Females (heterozygous): mild-moderate due to lyonization (X-inactivation); "
            "  Females: may have NCV in normal range despite mutation; "
            "CENTRAL NERVOUS SYSTEM INVOLVEMENT: "
            "  MRI: central white matter lesions in severe cases; "
            "  Acute neurological episodes: stroke-like episodes, rare; "
            "  CNS involvement distinguishes GJB1 from other peripheral CMT genes; "
            "DIAGNOSIS: "
            "  GJB1 sequencing (Xq13.1 — no MLPA needed, point mutations); "
            "  Check all family members; males clinically affected, females variable"
        ),
        "syndrome": "GJB1-CMT1X-X-linked / intermediate-NCV-25-45ms-PATHOGNOMONIC / males-severe-females-mild-lyonization",
        "inheritance": "X-linked dominant",
        "pathognomonic": "Intermediate NCV (25-45 m/s) does NOT fit pure demyelinating or axonal = PATHOGNOMONIC for GJB1",
        "key_therapy": "NO curative therapy; physical therapy + AFOs; genetic counseling critical (X-linked)",
        "critical_rules": [
            "INTERMEDIATE-NCV-25-45-ms-PATHOGNOMONIC-GJB1",
            "MALES-SEVERELY-AFFECTED-FEMALES-MILD-LYONIZATION",
            "CENTRAL-WHITE-MATTER-LESIONS-ACUTE-EPISODES",
            "GJB1-MOST-COMMON-X-LINKED-CMT",
            "INTERMEDIATE-NCV-NOT-PURE-DEMYELINATING-NOT-PURE-AXONAL",
        ],
    },
    {
        "gene": "MFN2",
        "protein": (
            "MFN2 -- 1p36.22 AD LOF -- 757aa -- "
            "Mitofusin-2-mitochondrial-fusion-protein-GTPase-"
            "CMT2A-most-common-axonal-CMT2-"
            "axonal-NCV-gt38ms-reduced-CMAP-amplitude-"
            "optic-atrophy-20-30pct-CMT2A-WITH-optic-atrophy-PATHOGNOMONIC-"
            "lower-limb-predominant-initially-pyramidal-signs-some-families"
        ),
        "locus": "1p36.22",
        "protein_size": (
            "757 aa / ~84 kDa / 1p36.22 MFN2 mitofusin 2 molecular context: "
            "STRUCTURE: "
            "  757 aa; 84 kDa outer mitochondrial membrane GTPase; "
            "  Mediates mitochondrial outer membrane fusion; "
            "  MFN2 LOF → fragmented mitochondria → impaired axonal transport; "
            "  Motor neurons highly dependent on mitochondrial dynamics (long axons); "
            "CMT2A — AXONAL NEUROPATHY: "
            "  Most common CMT2 (axonal subtype); "
            "  NCS: axonal pattern — NCV >38 m/s but reduced CMAP amplitude; "
            "  EMG: denervation in distal lower limbs; "
            "  Onset: typically childhood to young adult; "
            "  Lower limb >> upper limb initially; "
            "  Some families: severe early-onset with rapid progression to wheelchair; "
            "PATHOGNOMONIC FEATURE: "
            "  Optic atrophy: 20-30% of MFN2 patients; "
            "  CMT2A WITH optic atrophy = PATHOGNOMONIC for MFN2; "
            "  Bilateral optic atrophy with neuropathy: MFN2 first gene to test; "
            "  Pyramidal signs in some families; "
            "  Hearing loss rare; "
            "SEVERITY: "
            "  Severe early-onset (childhood): poor prognosis, wheelchair early; "
            "  Moderate: onset 2nd decade, walking aids by 3rd-4th decade; "
            "  Mild: onset 3rd-4th decade, slowly progressive"
        ),
        "syndrome": "MFN2-CMT2A-most-common-axonal / optic-atrophy-20-30pct-PATHOGNOMONIC / lower-limb-predominant",
        "inheritance": "AD",
        "pathognomonic": "CMT2A WITH optic atrophy = PATHOGNOMONIC for MFN2; axonal NCV pattern (>38 m/s, reduced CMAP)",
        "key_therapy": "NO curative therapy; ophthalmology annual (optic atrophy); physical therapy + AFOs",
        "critical_rules": [
            "MFN2-OPTIC-ATROPHY-20-30pct-PATHOGNOMONIC",
            "AXONAL-CMT2A-MOST-COMMON-CMT2",
            "LOWER-LIMB-PREDOMINANT-INITIALLY",
            "CMT2A-WITH-OPTIC-ATROPHY-TEST-MFN2-FIRST",
            "AXONAL-NCV-GT38ms-REDUCED-CMAP-AMPLITUDE",
        ],
    },
    {
        "gene": "GDAP1",
        "protein": (
            "GDAP1 -- 8q21.11 AR LOF -- 358aa -- "
            "Ganglioside-induced-differentiation-associated-protein-1-"
            "CMT4A-most-common-AR-CMT-some-populations-"
            "axonal-or-demyelinating-mutation-dependent-"
            "vocal-cord-paresis-diaphragm-phrenic-nerve-PATHOGNOMONIC-"
            "early-onset-rapid-progression-wheelchair-2nd-3rd-decade"
        ),
        "locus": "8q21.11",
        "protein_size": (
            "358 aa / ~41 kDa / 8q21.11 GDAP1 ganglioside-induced differentiation-associated protein 1 molecular context: "
            "STRUCTURE: "
            "  358 aa; 41 kDa; mitochondrial outer membrane protein; "
            "  Involved in mitochondrial fission; highly expressed in neurons/Schwann cells; "
            "  GDAP1 LOF → impaired mitochondrial fission → axonal transport failure; "
            "  Mutation type determines neuropathy subtype (axonal vs demyelinating); "
            "CMT4A — AR CMT: "
            "  Most common AR CMT in some populations (Spanish, Arab, Roma); "
            "  Null/truncating mutations: severe axonal CMT — very early onset; "
            "  Missense mutations: may give demyelinating or axonal subtype; "
            "  AR-null mutations: wheelchair confinement by 2nd-3rd decade; "
            "  AD form (GOF mutations): milder, later onset; "
            "PATHOGNOMONIC FEATURES: "
            "  Vocal cord paresis: a hallmark — due to recurrent laryngeal nerve involvement; "
            "  Diaphragm/phrenic nerve involvement: respiratory compromise; "
            "  GDAP1 is the ONLY common CMT gene with significant vocal cord + phrenic nerve involvement; "
            "  This combination = PATHOGNOMONIC for GDAP1; "
            "  Rapid progression distinguishes from other AR-CMT genes; "
            "DIAGNOSIS: "
            "  GDAP1 sequencing + copy number analysis; "
            "  NCS: axonal or demyelinating depending on mutation"
        ),
        "syndrome": "GDAP1-CMT4A-AR / vocal-cord-phrenic-PATHOGNOMONIC / wheelchair-2nd-3rd-decade / axonal-or-demyelinating",
        "inheritance": "AR",
        "pathognomonic": "Vocal cord paresis + phrenic nerve involvement = PATHOGNOMONIC for GDAP1; rapid early-onset progression",
        "key_therapy": "NO curative therapy; ENT + pulmonary monitoring (vocal cord/phrenic); physical therapy + AFOs",
        "critical_rules": [
            "GDAP1-VOCAL-CORD-PHRENIC-PATHOGNOMONIC",
            "AR-NULL-WHEELCHAIR-2ND-3RD-DECADE",
            "AXONAL-OR-DEMYELINATING-MUTATION-DEPENDENT",
            "EARLY-ONSET-RAPID-PROGRESSION-GDAP1",
            "ENT-PULMONARY-MONITORING-VOCAL-CORD-PHRENIC",
        ],
    },
    {
        "gene": "SH3TC2",
        "protein": (
            "SH3TC2 -- 5q32 AR LOF -- 1288aa -- "
            "SH3-domain-tetratricopeptide-repeats-2-"
            "CMT4C-most-common-AR-CMT-Europeans-"
            "R954W-founder-mutation-Roma-European-"
            "scoliosis-cranial-nerve-involvement-PATHOGNOMONIC-"
            "large-caliber-myelinated-fiber-loss-onion-bulbs"
        ),
        "locus": "5q32",
        "protein_size": (
            "1288 aa / ~142 kDa / 5q32 SH3TC2 SH3 domain and tetratricopeptide repeats 2 molecular context: "
            "STRUCTURE: "
            "  1288 aa; 142 kDa scaffolding protein; expressed in Schwann cells; "
            "  Contains SH3 domain (protein-protein interaction) and TPR repeats (scaffold); "
            "  Localizes to early endosomes; involved in Schwann cell lipid metabolism; "
            "  SH3TC2 LOF → impaired Schwann cell endosomal recycling → demyelination; "
            "CMT4C — MOST COMMON AR-CMT IN EUROPEANS: "
            "  R954W (p.Arg954Trp): founder mutation found in Roma (Romani) and European populations; "
            "  One of most common AR-CMT alleles in European cohorts; "
            "  Demyelinating neuropathy: NCV typically 15-25 m/s; "
            "PATHOGNOMONIC FEATURES: "
            "  Scoliosis: in ~50-60% — more severe than other CMT; "
            "  Cranial nerve involvement: facial weakness, hearing loss; "
            "  Scoliosis + cranial nerve = PATHOGNOMONIC for SH3TC2-CMT4C (distinguishes from other AR-CMT); "
            "  Hearing loss: progressive sensorineural in subset; "
            "  Nerve biopsy: large-caliber myelinated fiber loss + onion bulbs; "
            "DIAGNOSIS: "
            "  R954W founder screen first in European populations; "
            "  Full SH3TC2 sequencing + copy number if R954W negative"
        ),
        "syndrome": "SH3TC2-CMT4C-most-common-AR-CMT-Europeans / scoliosis-cranial-nerve-PATHOGNOMONIC / R954W-founder",
        "inheritance": "AR",
        "pathognomonic": "Scoliosis + cranial nerve involvement = PATHOGNOMONIC for SH3TC2-CMT4C; R954W European founder",
        "key_therapy": "NO curative therapy; orthopedic management (scoliosis); audiological follow-up; physical therapy + AFOs",
        "critical_rules": [
            "SH3TC2-SCOLIOSIS-CRANIAL-NERVE-PATHOGNOMONIC-CMT4C",
            "R954W-EUROPEAN-FOUNDER-ROMA",
            "MOST-COMMON-AR-CMT-EUROPEANS",
            "SCOLIOSIS-50-60pct-SEVERE-SH3TC2",
            "HEARING-LOSS-PROGRESSIVE-SENSORINEURAL-SUBSET",
        ],
    },
    {
        "gene": "PRX",
        "protein": (
            "PRX -- 19q13.2 AR LOF -- 1461aa -- "
            "Periaxin-163kDa-abaxonal-cytoskeletal-scaffold-"
            "CMT4F-severe-hypomyelinating-neuropathy-"
            "periaxin-maintains-myelin-compaction-"
            "onion-bulb-severe-PATHOGNOMONIC-NCV-lt10ms-"
            "early-onset-sensory-loss-predominant-Dejerine-Sottas"
        ),
        "locus": "19q13.2",
        "protein_size": (
            "1461 aa / ~163 kDa / 19q13.2 PRX periaxin molecular context: "
            "STRUCTURE: "
            "  1461 aa; 163 kDa; largest CMT gene product (peripheral); "
            "  Abaxonal cytoskeletal scaffold in mature compact myelin; "
            "  L-periaxin: links Schwann cell dystroglycan complex to myelin; "
            "  PRX LOF → severe destabilization of compact myelin → profound hypomyelination; "
            "CMT4F — SEVERE HYPOMYELINATING: "
            "  NCV: very low, <10 m/s in most patients (severely hypomyelinated); "
            "  Onset: early childhood, often first years of life; "
            "  Walking difficulties from early age; "
            "  Sensory loss: often predominant (pain insensitivity, foot ulcers); "
            "  Dejerine-Sottas phenotype: severe truncating mutations → congenital onset; "
            "PATHOGNOMONIC FEATURES: "
            "  Onion bulb formation: very severe, prominent on biopsy = PATHOGNOMONIC; "
            "  More severe and abundant onion bulbs than MPZ or SH3TC2; "
            "  NCV <10 m/s: lower threshold than other demyelinating CMT; "
            "  Combined NCV <10 m/s + early onset + sensory predominance = PATHOGNOMONIC; "
            "DIAGNOSIS: "
            "  PRX sequencing; nerve biopsy confirms severe onion bulbs; "
            "  Consider in any CMT4 with NCV <10 m/s"
        ),
        "syndrome": "PRX-CMT4F-severe-hypomyelinating / onion-bulb-severe-PATHOGNOMONIC / NCV-lt10-ms / early-onset-sensory-predominant",
        "inheritance": "AR",
        "pathognomonic": "Onion bulb severe + NCV <10 m/s + early onset sensory predominant = PRX CMT4F PATHOGNOMONIC",
        "key_therapy": "NO curative therapy; foot care (sensory loss/ulcers); physical therapy + AFOs; scoliosis monitoring",
        "critical_rules": [
            "PRX-ONION-BULB-SEVERE-NCV-LT10-PATHOGNOMONIC",
            "EARLY-ONSET-SENSORY-PREDOMINANT",
            "DEJERINE-SOTTAS-TRUNCATING-MUTATIONS",
            "NCV-LT10-ms-MOST-SEVERELY-HYPOMYELINATED",
            "FOOT-ULCERS-SENSORY-LOSS-PREVENTION-MANDATORY",
        ],
    },
    {
        "gene": "NEFL",
        "protein": (
            "NEFL -- 8p21.2 AD LOF -- 543aa -- "
            "Neurofilament-light-chain-NF-L-intermediate-filament-axonal-cytoskeleton-"
            "CMT2E-axonal-AND-CMT1F-demyelinating-SAME-GENE-different-phenotype-classes-"
            "neurofilament-accumulation-axons-PATHOGNOMONIC-nerve-biopsy-"
            "early-onset-childhood-weakness-tremor-common-"
            "CSF-protein-elevated-demyelinating-form-CMT1F"
        ),
        "locus": "8p21.2",
        "protein_size": (
            "543 aa / ~61 kDa / 8p21.2 NEFL neurofilament light chain molecular context: "
            "STRUCTURE: "
            "  543 aa; 61 kDa; forms NF-L intermediate filament in axons; "
            "  Component of neurofilament triplet (NF-L, NF-M, NF-H); "
            "  Critical for axonal caliber maintenance and axonal transport scaffold; "
            "  NEFL mutations → neurofilament accumulation in cell body and axons; "
            "SAME GENE — TWO PHENOTYPES (UNIQUE FEATURE): "
            "  CMT2E: axonal neuropathy; NCV >38 m/s, reduced CMAP; "
            "  CMT1F: demyelinating neuropathy; NCV <38 m/s; "
            "  SAME GENE can produce axonal OR demyelinating CMT: UNIQUE; "
            "  Mutation position may partially explain phenotype class; "
            "PATHOGNOMONIC FEATURES: "
            "  Neurofilament accumulation in axons = PATHOGNOMONIC on nerve biopsy; "
            "  Giant axonal changes with neurofilament aggregates; "
            "  Early onset: childhood weakness, sometimes congenital; "
            "  Tremor: present in many families; "
            "CSF PROTEIN: "
            "  Elevated CSF protein in demyelinating form (CMT1F); "
            "  May prompt LP if inflammatory neuropathy suspected; "
            "  NEFL NCV may be intermediate in some families; "
            "  Serum NfL: elevated in CMT (neurofilament light chain biomarker); "
            "DIAGNOSIS: "
            "  NEFL sequencing; NCS distinguishes CMT2E vs CMT1F subtype"
        ),
        "syndrome": "NEFL-CMT2E-axonal-CMT1F-demyelinating-SAME-GENE / neurofilament-accumulation-PATHOGNOMONIC / early-onset-tremor",
        "inheritance": "AD",
        "pathognomonic": "Neurofilament accumulation in axons on nerve biopsy PATHOGNOMONIC; same gene gives axonal OR demyelinating CMT",
        "key_therapy": "NO curative therapy; physical therapy + AFOs; serum NfL as disease biomarker",
        "critical_rules": [
            "NEFL-SAME-GENE-CMT2E-AXONAL-CMT1F-DEMYELINATING",
            "NEUROFILAMENT-ACCUMULATION-BIOPSY-PATHOGNOMONIC",
            "CSF-PROTEIN-ELEVATED-DEMYELINATING-FORM",
            "EARLY-ONSET-CHILDHOOD-TREMOR-COMMON",
            "SERUM-NFL-BIOMARKER-ELEVATED-CMT",
        ],
    },
]


def _make_cohort(gene_data: dict, seed: int) -> dict:
    """Generate a synthetic 40-patient cohort for one CMT gene."""
    rng = random.Random(seed)
    n = 40
    gene = gene_data["gene"]

    # Age at diagnosis (years) — gene-specific
    age_ranges = {
        "PMP22": (5.0, 30.0),   # childhood to adult; pes cavus may precede weakness
        "MPZ":   (1.0, 35.0),   # bimodal: early severe or late mild (Thr124Met)
        "GJB1":  (10.0, 35.0),  # childhood to young adult; males earlier
        "MFN2":  (1.0, 20.0),   # childhood to young adult; severe early-onset possible
        "GDAP1": (2.0, 10.0),   # early childhood; rapid progression
        "SH3TC2":(3.0, 15.0),   # childhood; European founder
        "PRX":   (1.0, 8.0),    # early childhood; congenital in severe truncating
        "NEFL":  (2.0, 15.0),   # childhood; sometimes congenital
    }
    lo, hi = age_ranges.get(gene, (5.0, 20.0))
    ages = [round(rng.uniform(lo, hi), 2) for _ in range(n)]
    mean_age_dx = round(sum(ages) / n, 1)

    # Pes cavus frequency
    pes_cavus_rates = {
        "PMP22": 0.80, "MPZ": 0.72, "GJB1": 0.65,
        "MFN2": 0.55, "GDAP1": 0.60, "SH3TC2": 0.58,
        "PRX": 0.65, "NEFL": 0.52,
    }
    pes_cavus_n = sum(1 for _ in range(n) if rng.random() < pes_cavus_rates.get(gene, 0.6))

    # Optic atrophy (only MFN2 high)
    optic_rates = {
        "PMP22": 0.03, "MPZ": 0.04, "GJB1": 0.05,
        "MFN2": 0.25, "GDAP1": 0.05, "SH3TC2": 0.05,
        "PRX": 0.03, "NEFL": 0.06,
    }
    optic_atrophy_n = sum(1 for _ in range(n) if rng.random() < optic_rates.get(gene, 0.04))

    # Vocal cord (only GDAP1 high)
    vocal_cord_rates = {
        "PMP22": 0.02, "MPZ": 0.02, "GJB1": 0.03,
        "MFN2": 0.03, "GDAP1": 0.35, "SH3TC2": 0.04,
        "PRX": 0.03, "NEFL": 0.03,
    }
    vocal_cord_n = sum(1 for _ in range(n) if rng.random() < vocal_cord_rates.get(gene, 0.03))

    # Scoliosis (SH3TC2 highest)
    scoliosis_rates = {
        "PMP22": 0.12, "MPZ": 0.15, "GJB1": 0.10,
        "MFN2": 0.14, "GDAP1": 0.18, "SH3TC2": 0.50,
        "PRX": 0.22, "NEFL": 0.16,
    }
    scoliosis_n = sum(1 for _ in range(n) if rng.random() < scoliosis_rates.get(gene, 0.15))

    # Cranial nerve involvement (SH3TC2 highest)
    cranial_nerve_rates = {
        "PMP22": 0.04, "MPZ": 0.06, "GJB1": 0.08,
        "MFN2": 0.04, "GDAP1": 0.08, "SH3TC2": 0.45,
        "PRX": 0.05, "NEFL": 0.06,
    }
    cranial_nerve_n = sum(1 for _ in range(n) if rng.random() < cranial_nerve_rates.get(gene, 0.05))

    # Intermediate NCV (GJB1 highest)
    intermediate_ncv_rates = {
        "PMP22": 0.05, "MPZ": 0.08, "GJB1": 0.75,
        "MFN2": 0.04, "GDAP1": 0.10, "SH3TC2": 0.06,
        "PRX": 0.03, "NEFL": 0.18,
    }
    intermediate_ncv_n = sum(1 for _ in range(n) if rng.random() < intermediate_ncv_rates.get(gene, 0.06))

    # Onion bulb on biopsy (PRX highest, MPZ high, SH3TC2 high)
    onion_bulb_rates = {
        "PMP22": 0.18, "MPZ": 0.50, "GJB1": 0.08,
        "MFN2": 0.06, "GDAP1": 0.15, "SH3TC2": 0.55,
        "PRX": 0.88, "NEFL": 0.10,
    }
    onion_bulb_n = sum(1 for _ in range(n) if rng.random() < onion_bulb_rates.get(gene, 0.12))

    # Wheelchair use
    wheelchair_rates = {
        "PMP22": 0.08, "MPZ": 0.20, "GJB1": 0.12,
        "MFN2": 0.22, "GDAP1": 0.45, "SH3TC2": 0.15,
        "PRX": 0.50, "NEFL": 0.18,
    }
    wheelchair_n = sum(1 for _ in range(n) if rng.random() < wheelchair_rates.get(gene, 0.15))

    # Mean NCV (m/s) — gene-specific range
    ncv_ranges = {
        "PMP22": (14.0, 28.0),   # demyelinating, uniform
        "MPZ":   (8.0, 30.0),    # demyelinating, variable by mutation
        "GJB1":  (26.0, 44.0),   # intermediate
        "MFN2":  (40.0, 55.0),   # axonal (preserved NCV, reduced amplitude)
        "GDAP1": (12.0, 28.0),   # axonal or demyelinating
        "SH3TC2":(10.0, 22.0),   # demyelinating
        "PRX":   (4.0, 12.0),    # severely hypomyelinated
        "NEFL":  (22.0, 42.0),   # axonal or demyelinating
    }
    ncv_lo, ncv_hi = ncv_ranges.get(gene, (20.0, 38.0))
    ncv_vals = [round(rng.uniform(ncv_lo, ncv_hi), 1) for _ in range(n)]
    mean_ncv = round(sum(ncv_vals) / n, 1)

    return {
        "gene": gene,
        "n": n,
        "mean_age_dx": mean_age_dx,
        "pes_cavus_n": pes_cavus_n,
        "optic_atrophy_n": optic_atrophy_n,
        "vocal_cord_n": vocal_cord_n,
        "scoliosis_n": scoliosis_n,
        "cranial_nerve_n": cranial_nerve_n,
        "intermediate_ncv_n": intermediate_ncv_n,
        "onion_bulb_n": onion_bulb_n,
        "wheelchair_n": wheelchair_n,
        "mean_ncv": mean_ncv,
        "syndrome": gene_data["syndrome"],
        "pathognomonic": gene_data["pathognomonic"],
        "key_therapy": gene_data["key_therapy"],
        "inheritance": gene_data["inheritance"],
        "locus": gene_data["locus"],
        "protein_size": gene_data["protein_size"],
        "critical_rules": gene_data["critical_rules"],
    }


def generate_overview():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    total = sum(c["n"] for c in cohorts)
    return {
        "atlas": "Hereditary-Peripheral-Neuropathy-CMT-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "description": (
            "Complete 8-gene Hereditary Peripheral Neuropathy (CMT Disease) reference atlas. "
            "PMP22 (CMT1A duplication most common CMT 60-70%; HNPP deletion episodic; "
            "pes cavus + hammertoe + hyporeflexia TRIAD PATHOGNOMONIC; "
            "no curative therapy; avoid neurotoxic drugs). "
            "MPZ (CMT1B 2nd most common demyelinating; mutation class determines severity; "
            "Thr124Met late-onset mild; onion bulb severe mutations PATHOGNOMONIC). "
            "GJB1 (CMT1X most common X-linked; intermediate NCV 25-45 m/s PATHOGNOMONIC; "
            "males severely affected, females mild via lyonization; central white matter lesions). "
            "MFN2 (CMT2A most common axonal; optic atrophy 20-30% PATHOGNOMONIC; "
            "axonal NCV pattern with reduced CMAP; lower limb predominant). "
            "GDAP1 (CMT4A; vocal cord paresis + phrenic nerve PATHOGNOMONIC; "
            "early onset rapid progression; wheelchair 2nd-3rd decade AR-null). "
            "SH3TC2 (CMT4C most common AR-CMT Europeans; R954W founder mutation; "
            "scoliosis + cranial nerve PATHOGNOMONIC; progressive hearing loss). "
            "PRX (CMT4F severe hypomyelinating; onion bulb severe + NCV <10 m/s PATHOGNOMONIC; "
            "early onset sensory predominant; Dejerine-Sottas phenotype). "
            "NEFL (CMT2E axonal AND CMT1F demyelinating — SAME GENE two phenotypes; "
            "neurofilament accumulation PATHOGNOMONIC on biopsy; CSF protein elevated in CMT1F). "
            "320-patient aggregate, 8 x 40 patients, seeds 3822-3829."
        ),
        "gene_summaries": [
            {
                "gene": g["gene"],
                "syndrome": g["syndrome"],
                "key_therapy": g["key_therapy"],
                "pathognomonic": g["pathognomonic"],
                "inheritance": g["inheritance"],
                "locus": g["locus"],
                "protein_size": g["protein_size"].split("molecular context")[0].strip(),
            }
            for g in ATLAS_GENES
        ],
        "cohort_summary": {
            c["gene"]: {
                "n": c["n"],
                "mean_age_dx_yr": c["mean_age_dx"],
                "pes_cavus_n": c["pes_cavus_n"],
                "optic_atrophy_n": c["optic_atrophy_n"],
                "vocal_cord_n": c["vocal_cord_n"],
                "scoliosis_n": c["scoliosis_n"],
                "cranial_nerve_n": c["cranial_nerve_n"],
                "intermediate_ncv_n": c["intermediate_ncv_n"],
                "onion_bulb_n": c["onion_bulb_n"],
                "wheelchair_n": c["wheelchair_n"],
                "mean_ncv": c["mean_ncv"],
            }
            for c in cohorts
        },
    }


def generate_breakdown():
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(ATLAS_GENES)]
    return {
        "atlas": "Hereditary-Peripheral-Neuropathy-CMT-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_data": {c["gene"]: c for c in cohorts},
        "atlas_genes": ATLAS_GENES,
        "total_patients": sum(c["n"] for c in cohorts),
    }


def generate_definitions():
    return {
        "atlas": "Hereditary-Peripheral-Neuropathy-CMT-Atlas",
        "definitions": {
            "Charcot-Marie-Tooth Disease (CMT)": (
                "Most common inherited peripheral neuropathy; prevalence 1 in 2500. "
                "Heterogeneous group classified by NCV: demyelinating CMT1 (NCV <38 m/s), "
                "axonal CMT2 (NCV ≥38 m/s, reduced CMAP), and intermediate CMT (NCV 25-45 m/s). "
                "Over 100 causative genes identified."
            ),
            "CMT1A (PMP22 duplication)": (
                "Most common CMT subtype (60-70% of CMT1). "
                "Caused by 1.4 Mb duplication of 17p12 containing PMP22. "
                "Slowly progressive; pes cavus triad pathognomonic. "
                "Diagnosis by MLPA or FISH; gene sequencing not required for duplication."
            ),
            "HNPP (PMP22 deletion)": (
                "Hereditary neuropathy with liability to pressure palsies. "
                "17p12 deletion (1 copy PMP22). Episodic focal neuropathies at pressure points. "
                "Tomaculous (sausage-shaped) myelin on biopsy pathognomonic. "
                "Avoid nerve compression; prognosis generally good."
            ),
            "Intermediate NCV — GJB1 PATHOGNOMONIC": (
                "NCV 25-45 m/s: does not fit pure demyelinating (<38 m/s) or pure axonal. "
                "Intermediate NCV in a male patient = test GJB1 (CMT1X) FIRST. "
                "Result of combined axonal + demyelinating pathology from Cx32 loss. "
                "GJB1 females may have NCV in normal range despite carrying mutation."
            ),
            "Onion bulbs on nerve biopsy": (
                "Concentric Schwann cell processes surrounding axon, resembling onion. "
                "Formed by repeated cycles of demyelination/remyelination. "
                "Present in: CMT1 (PMP22, MPZ, SH3TC2, PRX). "
                "Most severe/abundant in PRX; severe in MPZ severe mutations. "
                "Finding onion bulbs confirms demyelinating CMT subtype."
            ),
            "Demyelinating vs Axonal CMT": (
                "Demyelinating CMT1: NCV <38 m/s (motor); uniform slowing; onion bulbs on biopsy. "
                "Axonal CMT2: NCV ≥38 m/s but reduced CMAP amplitude; denervation on EMG. "
                "Intermediate: 25-45 m/s — GJB1/CMT1X most common cause. "
                "NEFL unique: same gene can produce CMT2E (axonal) or CMT1F (demyelinating)."
            ),
            "Pes Cavus": (
                "High-arched foot deformity caused by intrinsic foot muscle wasting. "
                "Pathognomonic triad in CMT: pes cavus + hammertoe + hyporeflexia. "
                "Often precedes weakness by years (subclinical CMT marker). "
                "Unexplained pes cavus: order CMT gene panel."
            ),
            "MFN2 Optic Atrophy": (
                "Optic atrophy in 20-30% of MFN2 patients = PATHOGNOMONIC for MFN2/CMT2A. "
                "Bilateral optic atrophy with peripheral neuropathy: test MFN2 first. "
                "Mitochondrial fusion protein dysfunction affects both peripheral nerve and optic nerve. "
                "Annual ophthalmology evaluation recommended for all MFN2 patients."
            ),
            "GDAP1 Vocal Cord/Phrenic Involvement": (
                "Vocal cord paresis (hoarseness, stridor) + phrenic nerve (diaphragm) involvement "
                "= PATHOGNOMONIC for GDAP1/CMT4A. "
                "Recurrent laryngeal nerve particularly vulnerable. "
                "Respiratory monitoring mandatory; may require ventilatory support. "
                "Distinguishes GDAP1 from all other AR-CMT genes."
            ),
            "CMT4C (SH3TC2) — Scoliosis + Cranial Nerve": (
                "Scoliosis (~50-60%) + cranial nerve involvement (facial weakness, hearing loss) "
                "= PATHOGNOMONIC combination for SH3TC2/CMT4C. "
                "R954W: founder mutation in Roma/European populations — screen first. "
                "Most common AR-CMT in European populations."
            ),
            "PRX — Severe Hypomyelination": (
                "CMT4F: most severe demyelinating CMT subtype. "
                "NCV typically <10 m/s; early onset walking difficulties; sensory loss predominant. "
                "Severe abundant onion bulbs on biopsy PATHOGNOMONIC. "
                "Foot care essential: sensory loss → ulcers → infection. "
                "Dejerine-Sottas phenotype in truncating mutations."
            ),
            "NEFL — Same Gene Two Phenotypes": (
                "NEFL is unique: same gene produces CMT2E (axonal, NCV ≥38 m/s) "
                "OR CMT1F (demyelinating, NCV <38 m/s). "
                "Neurofilament light chain accumulation in axons PATHOGNOMONIC on biopsy. "
                "CSF protein elevated in demyelinating form (CMT1F) — may mimic CIDP. "
                "Serum NfL: elevated biomarker correlating with disease activity."
            ),
        },
        "critical_clinical_rules": {
            rule: True
            for gene_data in ATLAS_GENES
            for rule in gene_data["critical_rules"]
        },
        "ncv_classification": {
            "demyelinating_CMT1": {
                "threshold": "<38 m/s motor NCV",
                "genes": ["PMP22", "MPZ", "SH3TC2", "PRX", "NEFL-CMT1F"],
                "approach": "MLPA/FISH PMP22 first; if negative sequence MPZ, GJB1, SH3TC2, PRX, NEFL",
            },
            "intermediate_CMT": {
                "threshold": "25-45 m/s motor NCV",
                "genes": ["GJB1", "NEFL-some"],
                "approach": "Male with intermediate NCV: test GJB1 FIRST (CMT1X most common X-linked)",
            },
            "axonal_CMT2": {
                "threshold": "≥38 m/s NCV with reduced CMAP amplitude",
                "genes": ["MFN2", "GDAP1-axonal", "NEFL-CMT2E"],
                "approach": "Test MFN2 first (most common CMT2); if optic atrophy MFN2 PATHOGNOMONIC",
            },
        },
        "cascade_testing": {
            "PMP22": ["MLPA_17p12_duplication_deletion_first", "PMP22_sequencing_if_MLPA_negative"],
            "MPZ": ["MPZ_sequencing", "note_mutation_class_determines_severity"],
            "GJB1": ["GJB1_sequencing_Xq13_males_first_if_intermediate_NCV"],
            "MFN2": ["MFN2_sequencing", "ophthalmology_optic_atrophy_screening"],
            "GDAP1": ["GDAP1_sequencing", "ENT_vocal_cord_assessment", "pulmonary_phrenic"],
            "SH3TC2": ["R954W_founder_screen_European_first", "SH3TC2_full_sequencing_if_negative"],
            "PRX": ["PRX_sequencing", "nerve_biopsy_severe_onion_bulbs"],
            "NEFL": ["NEFL_sequencing", "LP_CSF_protein_if_demyelinating_CMT1F"],
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary Peripheral Neuropathy CMT Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(
            f"  {gene}: n={d['n']}, age_dx={d['mean_age_dx']}yr, "
            f"pes_cavus={d['pes_cavus_n']}/40, "
            f"onion_bulb={d['onion_bulb_n']}/40, "
            f"wheelchair={d['wheelchair_n']}/40, "
            f"mean_ncv={d['mean_ncv']}m/s"
        )
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
    print("SCRIPT OK")
