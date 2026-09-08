#!/usr/bin/env python3
"""Hereditary-AR-SCA-Atlas — Complete 8-Gene Autosomal-Recessive Cerebellar Ataxia Atlas
(FRDA / ARSACS / AOA1 / AOA2 / AVED / POLG-Ataxia / ARCA2 / ARCA1).

FXN      (Frataxin; 210 aa; 9q21.11; AR;
          Friedreich Ataxia — MOST COMMON hereditary ataxia (1:50 000);
          GAA triplet repeat >66 copies in intron 1 — >400 severe/early;
          ABSENT LOWER LIMB REFLEXES PATHOGNOMONIC — hallmark exam finding;
          Hypertrophic cardiomyopathy → ECG/echo at diagnosis MANDATORY, causes most deaths;
          Diabetes mellitus 30%; scoliosis 60%; hearing loss 10%;
          Omaveloxolone (Skyclarys) FDA-approved Feb 2023 — Nrf2 activator;
          seed SEED_BASE+0).
SACS     (Sacsin; 4579 aa; 13q12.12; AR;
          ARSACS — Spastic Ataxia of Charlevoix-Saguenay;
          SPASTICITY PATHOGNOMONIC — lower limb spasticity before or concurrent with ataxia;
          Retinal hypermyelination (horizontal yellow-white stripes on fundoscopy/OCT) PATHOGNOMONIC;
          Enlarged pons + superior cerebellar peduncle on MRI;
          French-Canadian founder (Charlevoix-Saguenay region, Quebec); Turkish, Italian, North African also;
          No disease-modifying therapy; baclofen/tizanidine for spasticity; seed SEED_BASE+1).
APTX     (Aprataxin; 342 aa; 9p21.1; AR;
          AOA1 — Ataxia with Oculomotor Apraxia type 1;
          OCULOMOTOR APRAXIA PATHOGNOMONIC — horizontal gaze initiation failure;
          HYPOALBUMINAEMIA + HYPERCHOLESTEROLAEMIA PATHOGNOMONIC biochemical signature;
          Early severe peripheral neuropathy (sensorimotor axonal);
          Japan + Portugal highest prevalence; compound het in others;
          Coenzyme Q10 supplement anecdotal; no disease-modifying therapy; seed SEED_BASE+2).
SETX     (Senataxin; 2667 aa; 9q34.13; AR;
          AOA2 — Ataxia with Oculomotor Apraxia type 2;
          ELEVATED AFP (alpha-fetoprotein) PATHOGNOMONIC — key DDx from AOA1 (normal AFP in AOA1);
          Oculomotor apraxia (less severe than AOA1);
          Peripheral neuropathy; cerebellar atrophy; onset teens-20s;
          NORMAL ALBUMIN distinguishes from AOA1 (hypoalbuminaemia in AOA1);
          RNA/DNA helicase — resolves transcription-replication conflicts;
          No disease-modifying therapy; seed SEED_BASE+3).
TTPA     (alpha-tocopherol transfer protein; 278 aa; 8q12.3; AR;
          AVED — Ataxia with Vitamin E Deficiency;
          VERY LOW SERUM VITAMIN E PATHOGNOMONIC (< 3 mg/L);
          TREATABLE — vitamin E 800-1200 mg/day halts or reverses progression;
          Resembles Friedreich ataxia: absent lower limb reflexes, sensory neuropathy;
          Key DDx: absent GAA repeat, normal cardiac; test vitamin E level in all ataxia;
          North African / Mediterranean founder; also Tunisian, Moroccan;
          seed SEED_BASE+4).
POLG     (DNA Polymerase Gamma catalytic subunit; 1239 aa; 15q26.1; AR;
          POLG-related ataxia (MIRAS / MEMSA / SANDO);
          EPILEPSY PROMINENT — focal/status epilepticus early feature;
          VALPROATE ABSOLUTE CI — fatal hepatotoxicity (mitochondrial toxicity, liver failure);
          Multisystem: sensory neuropathy, ophthalmoplegia, cognitive decline;
          mtDNA depletion or multiple deletions in muscle;
          Levetiracetam + lamotrigine preferred AEDs; avoid valproate in any patient with ataxia+epilepsy until POLG excluded;
          seed SEED_BASE+5).
ADCK3    (aarF domain-containing kinase 3; 454 aa; 1q42.13; AR;
          ARCA2 — Autosomal Recessive Cerebellar Ataxia type 2;
          CoQ10 (ubiquinone) biosynthesis deficiency — ADCK3 = CABC1;
          Elevated lactate (blood and CSF) common;
          CoQ10 supplementation 300-2400 mg/day may improve or stabilise;
          Childhood onset (2-20 yr); cerebellar atrophy; exercise intolerance;
          Pyramidal signs 50%; cognitive impairment in severe cases;
          seed SEED_BASE+6).
SYNE1    (Spectrin repeat containing nuclear envelope protein 1 / Nesprin-1; 8797 aa; 6q25.2; AR;
          ARCA1 — Autosomal Recessive Cerebellar Ataxia type 1;
          PURE CEREBELLAR — no extracerebellar features, no peripheral neuropathy;
          Slow progression; normal cognitive function; normal life expectancy;
          French-Canadian founder (Quebec); also North African;
          No specific treatment; physiotherapy; cerebellar cortical atrophy on MRI;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2118-2125).
"""

import random

SEED_BASE = 2118

AR_SCA_GENES = [
    # -- FXN — Friedreich Ataxia --------------------------------------------
    {
        "gene": "FXN",
        "alt_name": (
            "FXN (FXN-210aa-9q21.11 / AR — Friedreich-Ataxia-Most-Common-Hereditary-Ataxia — "
            "GAA-Repeat->66-Copies-Intron1-Triplet-Expansion — "
            "Absent-Lower-Limb-Reflexes-PATHOGNOMONIC — "
            "Hypertrophic-Cardiomyopathy-Most-Common-Death-Cause — "
            "Omaveloxolone-Skyclarys-FDA-Approved-2023-Nrf2-Activator)"
        ),
        "protein": (
            "FXN -- 9q21.11 AR -- FXN-210aa -- "
            "Frataxin-Mitochondrial-Iron-Chaperone-Iron-Sulfur-Cluster-Assembly -- "
            "GAA-Triplet-Repeat-Expansion-Intron1->66-Pathogenic->400-Severe-Early-Onset -- "
            "Frataxin-Deficiency-Iron-Accumulation-Mitochondria-Dorsal-Root-Ganglia-Purkinje-Cells -- "
            "Friedreich-Ataxia-FRDA-OMIM-229300-Most-Common-Hereditary-Ataxia-1:50000 -- "
            "Absent-Lower-Limb-Reflexes-PATHOGNOMONIC-Areflexia-Sensory-Neuropathy -- "
            "Hypertrophic-Cardiomyopathy-ECG-Echo-Mandatory-Diagnosis-Leading-Cause-Death -- "
            "Diabetes-Mellitus-30pct-Annual-HbA1c-Scoliosis-60pct-Spine-XR-Hearing-Loss-10pct -- "
            "Omaveloxolone-Skyclarys-150mg-Daily-FDA-2023-Nrf2-Pathway-Antioxidant-Response -- "
            "GAA-PCR-Triplet-Primed-Both-Alleles-Mandatory-Standard-WES-Misses-Repeat-Expansion"
        ),
        "locus": "9q21.11",
        "protein_size": "210 aa",
        "inheritance": (
            "AR (autosomal recessive); GAA triplet repeat expansion in intron 1 of FXN; "
            ">66 repeats = pathogenic; >400 repeats = severe/early onset; "
            "compound het (expansion + point mutation) in 2-5%; "
            "carriers (1/100) unaffected; "
            "repeat-primed PCR MANDATORY — standard exome panels MISS repeat expansions"
        ),
        "age_of_onset": "5-25 years (mean 15); occasionally adult >25 yr (late-onset LOFA)",
        "pathognomonic": (
            "ABSENT LOWER LIMB REFLEXES PATHOGNOMONIC — knee + ankle jerks absent from early disease; "
            "Progressive gait + limb ataxia + dysarthria; "
            "Hypertrophic cardiomyopathy (ECG: T-wave inversion leads V1-V6 + short PR); "
            "Positive Romberg; loss of vibration/proprioception (posterior column); "
            "MRI: spinal cord atrophy (predominantly) + mild cerebellar vermis atrophy; "
            "Late-onset LOFA (>25 yr): retained reflexes possible — atypical presentation"
        ),
        "treatment": (
            "OMAVELOXOLONE (Skyclarys) 150 mg/day — FDA-approved Feb 2023; Nrf2 activator; "
            "slows decline in MOXIe trial (primary endpoint SARA); monitor LFTs (hepatotoxicity); "
            "IDEBENONE 900 mg/day — off-label cardiac benefit, modest neuroprotection; "
            "CARDIAC: echo + ECG annually; bisoprolol/ACE-I for dilated phase; "
            "DIABETES: HbA1c annually; insulin if needed; "
            "SCOLIOSIS: spinal brace if Cobb <40°; surgical fixation if >40°; "
            "PHYSIOTHERAPY: gait, balance, core strengthening; "
            "SLT: dysarthria management; "
            "Genetic counselling: 25% sibling risk (AR); carrier testing for parents"
        ),
        "key_biomarker": (
            "Absent lower limb reflexes + progressive ataxia → test FXN GAA repeat; "
            "GAA repeat size (both alleles) predicts severity; "
            "ECG T-wave inversion V1-V6 = cardiomyopathy signature"
        ),
        "critical_flags": [
            "FXN-GAA-REPEAT->66-PATHOGNOMONIC-MOLECULAR",
            "ABSENT-LOWER-LIMB-REFLEXES-PATHOGNOMONIC",
            "HYPERTROPHIC-CARDIOMYOPATHY-ANNUAL-ECHO-ECG-MANDATORY",
            "OMAVELOXOLONE-SKYCLARYS-FDA-2023-LEVEL-A",
            "REPEAT-PRIMED-PCR-MANDATORY-WES-MISSES",
            "DIABETES-30pct-ANNUAL-HBA1C",
            "SCOLIOSIS-60pct-SPINE-XR",
            "LOFA->25yr-RETAINED-REFLEXES-ATYPICAL",
            "CASCADE-TESTING-AR-25pct-SIBLING-RISK",
        ],
    },
    # -- SACS — ARSACS -------------------------------------------------------
    {
        "gene": "SACS",
        "alt_name": (
            "SACS (SACS-4579aa-13q12.12 / AR — ARSACS-Spastic-Ataxia-Charlevoix-Saguenay — "
            "Spasticity-PATHOGNOMONIC-Lower-Limb — "
            "Retinal-Hypermyelination-Yellow-Stripes-Fundoscopy-OCT-PATHOGNOMONIC — "
            "French-Canadian-Founder-Quebec)"
        ),
        "protein": (
            "SACS -- 13q12.12 AR -- SACS-4579aa -- "
            "Sacsin-Giant-Modular-Protein-Hsp90-Like-DNAJ-UBL-XPCB-Domains -- "
            "Mitochondrial-Anchoring-Purkinje-Cell-Specific-High-Expression -- "
            "ARSACS-OMIM-270550-Spastic-Ataxia-Charlevoix-Saguenay -- "
            "Lower-Limb-Spasticity-PATHOGNOMONIC-Before-Or-Concurrent-With-Ataxia -- "
            "Retinal-Hypermyelination-Horizontal-Yellow-White-Stripes-PATHOGNOMONIC-Fundoscopy-OCT -- "
            "MRI-Enlarged-Pons-Superior-Cerebellar-Peduncle-Hypointense-Lines-T2 -- "
            "French-Canadian-Charlevoix-Saguenay-Founder-p.Val3261Alafs-c.9737del -- "
            "Also-Turkish-Italian-North-African-Japanese-Worldwide-400-Families -- "
            "No-Disease-Modifying-Therapy-Baclofen-Tizanidine-Spasticity-Management"
        ),
        "locus": "13q12.12",
        "protein_size": "4579 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic SACS mutations; "
            "French-Canadian founder: p.Val3261Alafs (c.9737del) homozygous in Charlevoix-Saguenay; "
            "Turkish, Italian, North African, Japanese: distinct pathogenic variants; "
            "Heterozygous carriers unaffected; "
            "Standard exome panels detect SACS point mutations; founder allele by targeted PCR"
        ),
        "age_of_onset": "12-18 months (first signs: walking delay, spastic gait); ataxia onset 10-40 yr",
        "pathognomonic": (
            "LOWER LIMB SPASTICITY PATHOGNOMONIC — hyperreflexia, clonus, extensor plantar; "
            "occurs in all ARSACS patients (distinguishes from most other AR-SCAs); "
            "RETINAL HYPERMYELINATION PATHOGNOMONIC — horizontal yellow-white myelinated fibres on fundoscopy/OCT; "
            "MRI: enlarged pons + superior cerebellar peduncle + linear T2 hypointense pontine stripes; "
            "Peripheral neuropathy (axonal sensorimotor); "
            "Cerebellar atrophy (progressive); nystagmus; dysarthria; dysphagia"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "SPASTICITY: BACLOFEN 10-80 mg/day oral, or intrathecal baclofen pump if severe; "
            "TIZANIDINE 4-36 mg/day (muscle relaxant); PHYSIOTHERAPY: stretching, gait; "
            "ORTHOTICS: ankle-foot orthosis (AFO) for foot drop and gait; "
            "Wheelchair when ambulation fails (usually 3rd-4th decade); "
            "OPHTHALMOLOGY: retinal surveillance (annual fundoscopy/OCT); "
            "DYSPHAGIA: SLT assessment; PEG if severe; "
            "Genetic counselling: 25% sibling risk; cascade testing in Quebec community"
        ),
        "key_biomarker": (
            "Spastic ataxia + retinal stripes on fundoscopy = SACS until proven otherwise; "
            "MRI: enlarged pons + pontine linear hypointensities (T2) = ARSACS signature; "
            "French-Canadian ancestry + spastic ataxia → test SACS founder allele first"
        ),
        "critical_flags": [
            "SACS-SPASTICITY-PATHOGNOMONIC-ALL-PATIENTS",
            "RETINAL-HYPERMYELINATION-YELLOW-STRIPES-PATHOGNOMONIC",
            "MRI-ENLARGED-PONS-SCP-HYPOINTENSE-LINES",
            "FRENCH-CANADIAN-CHARLEVOIX-SAGUENAY-FOUNDER",
            "BACLOFEN-TIZANIDINE-SPASTICITY-FIRST-LINE",
            "INTRATHECAL-BACLOFEN-PUMP-SEVERE-SPASTICITY",
            "ANNUAL-FUNDOSCOPY-OCT-RETINAL-SURVEILLANCE",
            "PERIPHERAL-NEUROPATHY-AXONAL-SENSORIMOTOR",
            "NO-DISEASE-MODIFYING-THERAPY",
        ],
    },
    # -- APTX — AOA1 ---------------------------------------------------------
    {
        "gene": "APTX",
        "alt_name": (
            "APTX (APTX-342aa-9p21.1 / AR — AOA1-Ataxia-Oculomotor-Apraxia-Type1 — "
            "Oculomotor-Apraxia-PATHOGNOMONIC — "
            "Hypoalbuminaemia-Hypercholesterolaemia-PATHOGNOMONIC-Biochemical-Signature — "
            "Japan-Portugal-Prevalence)"
        ),
        "protein": (
            "APTX -- 9p21.1 AR -- APTX-342aa -- "
            "Aprataxin-Histidine-Triad-HIT-Superfamily-DNA-Repair-Deadenylase -- "
            "Resolves-Abortive-DNA-Ligation-Events-Single-Strand-Break-Repair -- "
            "AOA1-OMIM-208920-Ataxia-Oculomotor-Apraxia-Type-1 -- "
            "Oculomotor-Apraxia-PATHOGNOMONIC-Horizontal-Gaze-Initiation-Failure -- "
            "Hypoalbuminaemia-<35g/L-PATHOGNOMONIC-Biochemical-Hallmark-AOA1 -- "
            "Hypercholesterolaemia-Elevated-LDL-Cholesterol-PATHOGNOMONIC-AOA1 -- "
            "Early-Severe-Peripheral-Neuropathy-Sensorimotor-Axonal -- "
            "Japan-Highest-Prevalence-Portugal-Second-Compound-Het-Other-Populations -- "
            "Aprataxin-Foci-DNA-Damage-Response-XRCC1-PCNA-Interaction"
        ),
        "locus": "9p21.1",
        "protein_size": "342 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic APTX loss-of-function mutations; "
            "Japan: p.His201Gln (c.602A>G) and p.Trp279Ter (c.837G>A) common; "
            "Portugal: p.Trp279Ter; "
            "Compound heterozygous in non-founder populations; "
            "Standard exome panels detect pathogenic variants"
        ),
        "age_of_onset": "2-18 years (mean 4-10 yr); early childhood onset typical",
        "pathognomonic": (
            "OCULOMOTOR APRAXIA PATHOGNOMONIC — failure to initiate voluntary horizontal saccades; "
            "head thrust precedes eye movement (compensatory); "
            "HYPOALBUMINAEMIA (<35 g/L) PATHOGNOMONIC biochemical — check albumin in all childhood ataxia; "
            "HYPERCHOLESTEROLAEMIA (elevated LDL) PATHOGNOMONIC biochemical; "
            "Cerebellar ataxia; severe early sensorimotor peripheral neuropathy; "
            "Cerebellar cortical atrophy on MRI; "
            "Normal AFP (key DDx from AOA2: AFP elevated); "
            "Normal immunoglobulins (DDx from Ataxia-Telangiectasia: low Ig in AT)"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "STATIN + dietary intervention for hypercholesterolaemia; "
            "ALBUMIN: nutritional support if severely hypoalbuminaemic; "
            "PHYSIOTHERAPY: cerebellar rehabilitation, gait training, wheelchair when needed; "
            "SLT: dysarthria and dysphagia; "
            "ORTHOTICS: AFO for neuropathy foot drop; "
            "Coenzyme Q10 anecdotal (no evidence); "
            "Genetic counselling: 25% sibling risk; "
            "Ophthalmology: oculomotor apraxia management (head-turn strategies); "
            "Annual: albumin, cholesterol, neurophysiology (neuropathy progression)"
        ),
        "key_biomarker": (
            "Childhood ataxia + oculomotor apraxia + low albumin + high cholesterol → APTX/AOA1; "
            "Normal AFP distinguishes AOA1 from AOA2; "
            "Severe peripheral neuropathy early = important DDx from FRDA (absent reflexes but AFP normal)"
        ),
        "critical_flags": [
            "APTX-OCULOMOTOR-APRAXIA-PATHOGNOMONIC",
            "HYPOALBUMINAEMIA-<35g/L-PATHOGNOMONIC-BIOCHEMICAL",
            "HYPERCHOLESTEROLAEMIA-ELEVATED-LDL-PATHOGNOMONIC",
            "NORMAL-AFP-DDx-AOA2-ELEVATED-AFP",
            "JAPAN-PORTUGAL-HIGHEST-PREVALENCE",
            "SEVERE-EARLY-PERIPHERAL-NEUROPATHY",
            "STATIN-DIETARY-HYPERCHOLESTEROLAEMIA",
            "CHECK-ALBUMIN-CHOLESTEROL-ALL-CHILDHOOD-ATAXIA",
            "NO-DISEASE-MODIFYING-THERAPY",
        ],
    },
    # -- SETX — AOA2 ---------------------------------------------------------
    {
        "gene": "SETX",
        "alt_name": (
            "SETX (SETX-2667aa-9q34.13 / AR — AOA2-Ataxia-Oculomotor-Apraxia-Type2 — "
            "Elevated-AFP-PATHOGNOMONIC-Key-DDx-AOA1-Normal-AFP — "
            "Normal-Albumin-DDx-AOA1-Hypoalbuminaemia)"
        ),
        "protein": (
            "SETX -- 9q34.13 AR -- SETX-2667aa -- "
            "Senataxin-RNA-DNA-Helicase-Resolves-R-Loops-Transcription-Replication-Conflicts -- "
            "Upf1-Like-Helicase-Domain-C-Terminal-DNA-Damage-Response-Foci -- "
            "AOA2-OMIM-606002-Ataxia-Oculomotor-Apraxia-Type-2-Most-Common-AR-Ataxia-Europe -- "
            "Elevated-AFP-Alpha-Fetoprotein-PATHOGNOMONIC-Key-Differentiator-All-Other-AR-Ataxias -- "
            "Normal-Serum-Albumin-Distinguishes-AOA2-From-AOA1-Hypoalbuminaemia -- "
            "Oculomotor-Apraxia-Less-Severe-Than-AOA1-Horizontal-Gaze-Initiation -- "
            "Peripheral-Neuropathy-Sensorimotor-Axonal-Less-Severe-Than-AOA1 -- "
            "Cerebellar-Atrophy-Progressive-Onset-Teens-Twenties -- "
            "Japan-France-Europe-North-Africa-Founder-Variants"
        ),
        "locus": "9q34.13",
        "protein_size": "2667 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic SETX loss-of-function mutations; "
            "Most common AR ataxia in Southern Europe (France, Italy, Portugal); "
            "Japanese SETX mutations described; "
            "Compound heterozygous common; "
            "Standard exome panels detect pathogenic SETX variants; "
            "Note: heterozygous SETX dominant mutations → ALS4 (separate disease)"
        ),
        "age_of_onset": "10-22 years (teens to early adult; range 2-35 yr)",
        "pathognomonic": (
            "ELEVATED AFP (alpha-fetoprotein) PATHOGNOMONIC — present in >95% AOA2; "
            "KEY DDx: AFP elevated in AOA2, normal in AOA1 (APTX) and FRDA; "
            "NORMAL SERUM ALBUMIN — distinguishes from AOA1 (hypoalbuminaemia); "
            "Oculomotor apraxia (less severe than AOA1, may be intermittent); "
            "Peripheral neuropathy (sensorimotor, less severe than AOA1); "
            "Cerebellar atrophy (progressive); "
            "Normal immunoglobulins (DDx from Ataxia-Telangiectasia)"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "PHYSIOTHERAPY: gait training, cerebellar rehabilitation; "
            "SLT: dysarthria management; "
            "ORTHOTICS: AFO for peripheral neuropathy foot drop; "
            "Annual: AFP (disease marker/progression), neurophysiology; "
            "OPHTHALMOLOGY: oculomotor apraxia management; "
            "Genetic counselling: 25% sibling risk; "
            "Note: elevated AFP raises teratoma concern → distinguish by SETX genotyping; "
            "EUROSCA / AOA2 registry enrolment"
        ),
        "key_biomarker": (
            "AFP elevated in ataxia = AOA2 until proven otherwise; "
            "Normal albumin (not low) distinguishes AOA2 from AOA1; "
            "Most common AR ataxia in Southern Europe — test SETX first in French/Italian/Portuguese patients"
        ),
        "critical_flags": [
            "SETX-AFP-ELEVATED-PATHOGNOMONIC",
            "AFP-ELEVATED-AOA2-NORMAL-AOA1-FRDA-KEY-DDx",
            "NORMAL-ALBUMIN-DDx-AOA1-HYPOALBUMINAEMIA",
            "MOST-COMMON-AR-ATAXIA-SOUTHERN-EUROPE",
            "OCULOMOTOR-APRAXIA-LESS-SEVERE-THAN-AOA1",
            "PERIPHERAL-NEUROPATHY-LESS-SEVERE-THAN-AOA1",
            "ANNUAL-AFP-PROGRESSION-MARKER",
            "SETX-DOMINANT-HETEROZYGOUS-ALS4-SEPARATE-DISEASE",
            "NO-DISEASE-MODIFYING-THERAPY",
        ],
    },
    # -- TTPA — AVED ---------------------------------------------------------
    {
        "gene": "TTPA",
        "alt_name": (
            "TTPA (TTPA-278aa-8q12.3 / AR — AVED-Ataxia-Vitamin-E-Deficiency — "
            "Very-Low-Serum-Vitamin-E-PATHOGNOMONIC — "
            "TREATABLE-Vitamin-E-800-1200mg-Day-Halts-Reverses-Progression — "
            "North-African-Mediterranean-Founder)"
        ),
        "protein": (
            "TTPA -- 8q12.3 AR -- TTPA-278aa -- "
            "Alpha-Tocopherol-Transfer-Protein-Hepatic-Secretion-Vitamin-E-Into-VLDL -- "
            "Selective-Incorporation-Alpha-Tocopherol-VLDL-Over-Other-Tocopherols -- "
            "AVED-OMIM-277460-Ataxia-With-Vitamin-E-Deficiency -- "
            "Very-Low-Serum-Vitamin-E-<3mg/L-PATHOGNOMONIC-TREATABLE-CONDITION -- "
            "TTPA-Loss-Hepatic-Vitamin-E-Retention-Cannot-Secrete-Causes-Systemic-Deficiency -- "
            "Clinically-Resembles-Friedreich-Ataxia-Absent-Reflexes-Posterior-Column -- "
            "Key-DDx-FRDA-Absent-GAA-Repeat-Normal-Cardiac-Vitamin-E-Low -- "
            "North-African-Tunisian-Moroccan-Algerian-Founder-p.His101Gln -- "
            "Vitamin-E-800-1200mg-Daily-Tocopherol-Halts-Reverses-Ataxia-Level-B-Evidence"
        ),
        "locus": "8q12.3",
        "protein_size": "278 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic TTPA loss-of-function mutations; "
            "North African founder: p.His101Gln most common (Tunisia, Morocco, Algeria); "
            "Other: p.Arg59Trp (European); p.Glu141Lys; compound het in non-founder populations; "
            "Standard exome panels detect TTPA point mutations"
        ),
        "age_of_onset": "2-20 years (range; North African founder cohorts often 5-15 yr)",
        "pathognomonic": (
            "VERY LOW SERUM VITAMIN E (<3 mg/L) PATHOGNOMONIC and treatable; "
            "CHECK VITAMIN E IN EVERY ATAXIA PATIENT — simple blood test, treatable cause; "
            "Resembles Friedreich ataxia: absent/reduced lower limb reflexes, posterior column loss, gait ataxia; "
            "KEY DDx from FRDA: no GAA repeat, normal cardiac (no cardiomyopathy), low vitamin E; "
            "No retinitis pigmentosa (DDx from abetalipoproteinaemia); "
            "With treatment: stabilisation or improvement in 70% if started early"
        ),
        "treatment": (
            "VITAMIN E 800-1200 mg/day (alpha-tocopherol) — Level B evidence; "
            "LIFE-LONG TREATMENT — must not stop (ataxia rapidly worsens on stopping); "
            "Monitor serum vitamin E 3-6 monthly until stable, then annually; "
            "Response: stabilisation 60%, improvement 25%, continued decline 15%; "
            "Physiotherapy and SLT as supportive; "
            "Annual neurological assessment (SARA/ICARS); "
            "Genetic counselling: 25% sibling risk; "
            "Test vitamin E in ALL siblings at diagnosis — presymptomatic treatment prevents onset; "
            "Avoid high-dose vitamin E in anticoagulated patients (potentiates warfarin)"
        ),
        "key_biomarker": (
            "Serum vitamin E <3 mg/L in ataxia = AVED until proven otherwise; "
            "Test vitamin E in every ataxia workup — treatable cause must not be missed; "
            "North African ancestry + ataxia + absent reflexes → test TTPA and vitamin E first"
        ),
        "critical_flags": [
            "TTPA-LOW-VITAMIN-E-<3mg/L-PATHOGNOMONIC",
            "TREATABLE-VITAMIN-E-800-1200mg-DAY",
            "TEST-VITAMIN-E-ALL-ATAXIA-PATIENTS-MANDATORY",
            "LIFELONG-TREATMENT-STOP-WORSENS-RAPIDLY",
            "NORTH-AFRICAN-TUNISIAN-MOROCCAN-FOUNDER",
            "DDx-FRDA-NO-GAA-REPEAT-NORMAL-CARDIAC",
            "PRESYMPTOMATIC-SIBLING-TREATMENT-PREVENTS-ONSET",
            "MONITOR-VITAMIN-E-3-6-MONTHLY",
            "AVOID-HIGH-DOSE-VITAMIN-E-ON-WARFARIN",
        ],
    },
    # -- POLG — POLG-Ataxia --------------------------------------------------
    {
        "gene": "POLG",
        "alt_name": (
            "POLG (POLG-1239aa-15q26.1 / AR — POLG-Related-Ataxia-MIRAS-MEMSA-SANDO — "
            "Epilepsy-PROMINENT-Early-Feature — "
            "VALPROATE-ABSOLUTE-CI-Fatal-Hepatotoxicity-Mitochondrial-Toxicity)"
        ),
        "protein": (
            "POLG -- 15q26.1 AR -- POLG-1239aa -- "
            "DNA-Polymerase-Gamma-Catalytic-Subunit-Mitochondrial-DNA-Replication-Repair -- "
            "ExoI-ExoII-ExoIII-Proofreading-Domains-Linker-Polymerase-Domain -- "
            "mtDNA-Depletion-Or-Multiple-Deletions-Muscle-Biopsy-Diagnostic -- "
            "POLG-Related-Ataxia-MIRAS-SANDO-MEMSA-Alpers-CPEO-Phenotype-Spectrum -- "
            "MIRAS-OMIM-607459-Mitochondrial-Recessive-Ataxia-Syndrome-Norwegian-Finnish -- "
            "Epilepsy-Focal-Status-Epilepticus-PROMINENT-Early-Multi-Drug-Refractory -- "
            "VALPROATE-ABSOLUTE-CONTRAINDICATED-Fatal-Hepatotoxicity-Alpers-Syndrome -- "
            "Sensory-Neuropathy-Ophthalmoplegia-Cognitive-Decline-Multisystem -- "
            "Levetiracetam-Lamotrigine-Preferred-AEDs-Avoid-Valproate-Any-Ataxia-Epilepsy"
        ),
        "locus": "15q26.1",
        "protein_size": "1239 aa",
        "inheritance": (
            "AR (autosomal recessive) for ataxia syndromes; "
            "AD for CPEO/POLG1-related CPEO (dominant negative mutations); "
            "Compound het most common in ataxia phenotypes; "
            "p.Ala467Thr + p.Trp748Ser = MIRAS common alleles (Norwegian/Finnish/Scandinavian); "
            "Standard exome panels detect POLG point mutations; mtDNA analysis in muscle MANDATORY"
        ),
        "age_of_onset": "MIRAS: 5-41 yr (adult more typical); Alpers: infantile-early childhood",
        "pathognomonic": (
            "EPILEPSY PROMINENT — focal seizures, status epilepticus, occipital epilepsy; "
            "multi-drug refractory; may precede ataxia; "
            "VALPROATE ABSOLUTE CI — can trigger fatal hepatic failure (Alpers) even in MIRAS adults; "
            "Sensory ataxic neuropathy; ophthalmoplegia/ptosis; cognitive impairment; "
            "MRI: cerebellar + thalamic T2 signal change (acute decompensation); "
            "Elevated lactate (blood/CSF); mtDNA depletion in muscle"
        ),
        "treatment": (
            "AVOID VALPROATE IN ALL ATAXIA + EPILEPSY — until POLG excluded; "
            "LEVETIRACETAM: first-line AED (safe, no mitochondrial toxicity); "
            "LAMOTRIGINE: adjunct AED (safe); LACOSAMIDE: adjunct AED (safe); "
            "AVOID: valproate, phenobarbitone (ETC complex I toxicity), linezolid (ETC toxicity); "
            "COFACTOR SUPPLEMENTATION: CoQ10 + riboflavin + thiamine (anecdotal/supportive); "
            "PHYSIOTHERAPY: ataxia, neuropathy management; "
            "OPHTHALMOLOGY: ptosis/ophthalmoplegia — prism glasses, ptosis surgery; "
            "GENETIC TESTING: maternal relatives (mitochondrial disease team); "
            "Annual review: seizure control, neurophysiology (neuropathy), cognitive"
        ),
        "key_biomarker": (
            "Ataxia + epilepsy → exclude POLG BEFORE prescribing valproate; "
            "p.Ala467Thr + p.Trp748Ser = MIRAS alleles in Scandinavian patients; "
            "Muscle biopsy: mtDNA depletion or multiple deletions = POLG confirmation"
        ),
        "critical_flags": [
            "POLG-VALPROATE-ABSOLUTE-CI-FATAL-HEPATOTOXICITY",
            "EPILEPSY-PROMINENT-FOCAL-STATUS-EPILEPTICUS",
            "EXCLUDE-POLG-BEFORE-VALPROATE-IN-ATAXIA-EPILEPSY",
            "LEVETIRACETAM-LAMOTRIGINE-SAFE-PREFERRED-AEDs",
            "MIRAS-pAla467Thr-pTrp748Ser-SCANDINAVIAN",
            "MTDNA-DEPLETION-MUSCLE-BIOPSY-DIAGNOSTIC",
            "AVOID-PHENOBARB-LINEZOLID-MITOCHONDRIAL-TOXIC",
            "MULTISYSTEM-NEUROPATHY-OPHTHALMOPLEGIA-COGNITIVE",
            "COFACTOR-CoQ10-RIBOFLAVIN-THIAMINE-SUPPORTIVE",
        ],
    },
    # -- ADCK3 — ARCA2 -------------------------------------------------------
    {
        "gene": "ADCK3",
        "alt_name": (
            "ADCK3 (ADCK3-454aa-1q42.13 / AR — ARCA2-Autosomal-Recessive-Cerebellar-Ataxia-Type2 — "
            "CoQ10-Deficiency-Ubiquinone-Biosynthesis — "
            "CoQ10-Supplementation-300-2400mg-Day-May-Improve)"
        ),
        "protein": (
            "ADCK3 -- 1q42.13 AR -- ADCK3-454aa -- "
            "aarF-Domain-Containing-Kinase-3-CABC1-ABC1-Mitochondrial-Atypical-Kinase -- "
            "CoQ10-Ubiquinone-Biosynthesis-Regulatory-Kinase-Regulates-PDSS2-COQ-Enzymes -- "
            "ARCA2-OMIM-612016-Autosomal-Recessive-Cerebellar-Ataxia-Type-2 -- "
            "CoQ10-Primary-Deficiency-Reduced-Muscle-CoQ10-Level-Diagnostic -- "
            "Elevated-Lactate-Blood-CSF-Mitochondrial-Dysfunction-Marker -- "
            "Childhood-Onset-2-20yr-Cerebellar-Atrophy-Exercise-Intolerance -- "
            "Pyramidal-Signs-50pct-Cognitive-Impairment-Severe-Cases -- "
            "CoQ10-Supplementation-300-2400mg-Daily-May-Improve-Or-Stabilise -- "
            "Also-SCAR9-Secondary-CoQ10-Deficiency-Mutations-PDSS1-PDSS2-COQ2-COQ4-COQ6-COQ8B"
        ),
        "locus": "1q42.13",
        "protein_size": "454 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic ADCK3 loss-of-function mutations; "
            "Multiple ethnicities; no single founder; "
            "Standard exome panels detect ADCK3 mutations; "
            "Muscle CoQ10 assay MANDATORY to confirm deficiency; "
            "Plasma CoQ10 unreliable (normal plasma CoQ10 does not exclude muscle deficiency)"
        ),
        "age_of_onset": "2-20 years (mean 10-12 yr); exercise intolerance may precede ataxia",
        "pathognomonic": (
            "Cerebellar ataxia + childhood onset + elevated lactate = CoQ10 deficiency workup; "
            "MUSCLE CoQ10 ASSAY — reduced (<0.2 nmol/mg protein) confirms deficiency; "
            "PLASMA LACTATE elevated (blood and CSF); "
            "Exercise intolerance (myopathy component); "
            "Pyramidal signs (brisk reflexes, extensor plantar) 50%; "
            "MRI: cerebellar cortical atrophy; "
            "Muscle biopsy: ragged red fibres, COX-negative fibres (if severe)"
        ),
        "treatment": (
            "COENZYME Q10 (ubiquinol preferred) 300-2400 mg/day — trial for 6-12 months; "
            "Response variable: improvement (30-40%), stabilisation (40%), no benefit (20%); "
            "IDEBENONE 5-10 mg/kg/day — short-chain CoQ10 analogue, alternative; "
            "RIBOFLAVIN 200 mg/day — cofactor support; "
            "PHYSIOTHERAPY: gait training, balance, exercise tolerance; "
            "Avoid prolonged starvation (worsens lactic acidosis); "
            "Neuropsychology if cognitive impairment; "
            "Annual: lactate, CoQ10 (if on treatment), neurophysiology; "
            "Genetic counselling: 25% sibling risk; "
            "Muscle biopsy at diagnosis (CoQ10 assay + histology)"
        ),
        "key_biomarker": (
            "Childhood ataxia + elevated lactate → CoQ10 deficiency workup; "
            "Muscle CoQ10 assay (not plasma) is the diagnostic test; "
            "CoQ10 supplementation is the only disease-modifying trial available — start early"
        ),
        "critical_flags": [
            "ADCK3-CoQ10-DEFICIENCY-MUSCLE-ASSAY-MANDATORY",
            "PLASMA-CoQ10-UNRELIABLE-USE-MUSCLE-ASSAY",
            "CoQ10-SUPPLEMENTATION-300-2400mg-MAY-IMPROVE",
            "ELEVATED-LACTATE-BLOOD-CSF",
            "CHILDHOOD-ONSET-EXERCISE-INTOLERANCE",
            "PYRAMIDAL-SIGNS-50pct",
            "UBIQUINOL-PREFERRED-FORM-CoQ10",
            "IDEBENONE-SHORT-CHAIN-CoQ10-ALTERNATIVE",
            "AVOID-PROLONGED-STARVATION-LACTIC-ACIDOSIS",
        ],
    },
    # -- SYNE1 — ARCA1 -------------------------------------------------------
    {
        "gene": "SYNE1",
        "alt_name": (
            "SYNE1 (SYNE1-8797aa-6q25.2 / AR — ARCA1-Autosomal-Recessive-Cerebellar-Ataxia-Type1 — "
            "Pure-Cerebellar-No-Extracerebellar-Features — "
            "French-Canadian-Founder-Quebec-Slow-Progression)"
        ),
        "protein": (
            "SYNE1 -- 6q25.2 AR -- SYNE1-8797aa -- "
            "Nesprin-1-Spectrin-Repeat-Nuclear-Envelope-Protein-LINC-Complex -- "
            "Actin-Cytoskeleton-Nuclear-Membrane-Connection-Purkinje-Cell-Nuclear-Positioning -- "
            "ARCA1-OMIM-610743-Autosomal-Recessive-Cerebellar-Ataxia-Type-1 -- "
            "Pure-Cerebellar-Syndrome-No-Peripheral-Neuropathy-No-Extracerebellar-Features -- "
            "Slow-Progression-Normal-Cognitive-Function-Normal-Life-Expectancy -- "
            "French-Canadian-Quebec-Founder-Most-Cases-Worldwide -- "
            "Also-North-African-Portuguese-Turkish-Populations -- "
            "Cerebellar-Cortical-Atrophy-Vermal-Predominance-MRI -- "
            "No-Specific-Treatment-Physiotherapy-SLT"
        ),
        "locus": "6q25.2",
        "protein_size": "8797 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic SYNE1 truncating mutations most common; "
            "French-Canadian founder: multiple truncating alleles in Quebec population; "
            "North African, Portuguese, Turkish: distinct pathogenic variants; "
            "Standard exome panels detect SYNE1 truncating variants; "
            "Note: SYNE1 heterozygous variants do not cause AR ataxia — biallelic required"
        ),
        "age_of_onset": "20-40 years (adult onset, range 7-55); slower onset than many AR-SCAs",
        "pathognomonic": (
            "PURE CEREBELLAR SYNDROME — gait ataxia, limb ataxia, dysarthria; "
            "NO peripheral neuropathy (distinguishes from FRDA, AOA1, AOA2, POLG); "
            "NO extracerebellar features (cognitive, psychiatric, seizures all absent); "
            "Normal deep tendon reflexes (distinguishes from FRDA: absent reflexes); "
            "MRI: cerebellar cortical atrophy (vermal + hemispheric); brainstem largely spared; "
            "Slow progression — remains ambulant for decades; "
            "Normal AFP, normal albumin, normal lactate, normal vitamin E (DDx panel)"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "PHYSIOTHERAPY: gait training, balance exercises, fall prevention; "
            "SLT: dysarthria management (baseline + if worsening); "
            "OCCUPATIONAL THERAPY: adaptive equipment for ADL; "
            "Wheelchair usually not needed for 20-30 yr from onset (slow progression); "
            "Annual SARA assessment; "
            "Genetic counselling: 25% sibling risk; carrier testing for partner of affected; "
            "EUROSCA / RISCA registry enrolment; "
            "Communicate positive prognosis: pure cerebellar, slow, normal cognition, normal life expectancy"
        ),
        "key_biomarker": (
            "Pure cerebellar ataxia + normal reflexes + no neuropathy + slow progression → SYNE1/ARCA1; "
            "French-Canadian ancestry = ARCA1 most common pure AR cerebellar ataxia; "
            "Normal vitamin E, AFP, albumin, lactate — distinguishes from all other AR-SCAs"
        ),
        "critical_flags": [
            "SYNE1-PURE-CEREBELLAR-NO-EXTRACEREBELLAR",
            "NORMAL-REFLEXES-DDx-FRDA-ABSENT-REFLEXES",
            "NO-PERIPHERAL-NEUROPATHY-DDx-APTX-SETX-POLG",
            "FRENCH-CANADIAN-QUEBEC-FOUNDER",
            "SLOW-PROGRESSION-AMBULANT-DECADES",
            "NORMAL-COGNITION-NORMAL-LIFE-EXPECTANCY",
            "NORMAL-AFP-ALBUMIN-LACTATE-VITAMIN-E",
            "PURE-CEREBELLAR-CORTICAL-ATROPHY-MRI",
            "NO-DISEASE-MODIFYING-THERAPY",
        ],
    },
]


def _build_cohort(gene_entry: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_entry["gene"]
    cohort = []
    for pid in range(40):
        if gene == "FXN":
            onset = rng.randint(5, 35)
            sara = round(rng.uniform(8.0, 38.0), 1)
            absent_reflexes = True
            cardiomyopathy = rng.random() < 0.80
            diabetes = rng.random() < 0.30
            scoliosis = rng.random() < 0.60
            hearing_loss = rng.random() < 0.10
            omaveloxolone_eligible = True
            spasticity = False
            oculomotor_apraxia = False
            low_albumin = False
            elevated_afp = False
            low_vitamin_e = False
            epilepsy = False
            elevated_lactate = False
            coq10_deficient = False
            pure_cerebellar = False
        elif gene == "SACS":
            onset = rng.randint(0, 5)
            sara = round(rng.uniform(6.0, 30.0), 1)
            absent_reflexes = False
            cardiomyopathy = False
            diabetes = False
            scoliosis = False
            hearing_loss = False
            omaveloxolone_eligible = False
            spasticity = True
            oculomotor_apraxia = rng.random() < 0.60
            low_albumin = False
            elevated_afp = False
            low_vitamin_e = False
            epilepsy = False
            elevated_lactate = False
            coq10_deficient = False
            pure_cerebellar = False
        elif gene == "APTX":
            onset = rng.randint(2, 18)
            sara = round(rng.uniform(10.0, 35.0), 1)
            absent_reflexes = rng.random() < 0.70
            cardiomyopathy = False
            diabetes = False
            scoliosis = False
            hearing_loss = False
            omaveloxolone_eligible = False
            spasticity = False
            oculomotor_apraxia = True
            low_albumin = True
            elevated_afp = False
            low_vitamin_e = False
            epilepsy = False
            elevated_lactate = False
            coq10_deficient = False
            pure_cerebellar = False
        elif gene == "SETX":
            onset = rng.randint(10, 25)
            sara = round(rng.uniform(8.0, 28.0), 1)
            absent_reflexes = rng.random() < 0.50
            cardiomyopathy = False
            diabetes = False
            scoliosis = False
            hearing_loss = False
            omaveloxolone_eligible = False
            spasticity = False
            oculomotor_apraxia = rng.random() < 0.75
            low_albumin = False
            elevated_afp = True
            low_vitamin_e = False
            epilepsy = False
            elevated_lactate = False
            coq10_deficient = False
            pure_cerebellar = False
        elif gene == "TTPA":
            onset = rng.randint(2, 20)
            sara = round(rng.uniform(6.0, 32.0), 1)
            absent_reflexes = rng.random() < 0.75
            cardiomyopathy = rng.random() < 0.10
            diabetes = False
            scoliosis = False
            hearing_loss = False
            omaveloxolone_eligible = False
            spasticity = False
            oculomotor_apraxia = False
            low_albumin = False
            elevated_afp = False
            low_vitamin_e = True
            epilepsy = False
            elevated_lactate = False
            coq10_deficient = False
            pure_cerebellar = False
        elif gene == "POLG":
            onset = rng.randint(5, 45)
            sara = round(rng.uniform(8.0, 34.0), 1)
            absent_reflexes = rng.random() < 0.40
            cardiomyopathy = False
            diabetes = False
            scoliosis = False
            hearing_loss = rng.random() < 0.30
            omaveloxolone_eligible = False
            spasticity = rng.random() < 0.20
            oculomotor_apraxia = rng.random() < 0.40
            low_albumin = False
            elevated_afp = False
            low_vitamin_e = False
            epilepsy = True
            elevated_lactate = rng.random() < 0.70
            coq10_deficient = False
            pure_cerebellar = False
        elif gene == "ADCK3":
            onset = rng.randint(2, 22)
            sara = round(rng.uniform(6.0, 26.0), 1)
            absent_reflexes = False
            cardiomyopathy = False
            diabetes = False
            scoliosis = False
            hearing_loss = False
            omaveloxolone_eligible = False
            spasticity = rng.random() < 0.50
            oculomotor_apraxia = False
            low_albumin = False
            elevated_afp = False
            low_vitamin_e = False
            epilepsy = rng.random() < 0.25
            elevated_lactate = rng.random() < 0.75
            coq10_deficient = True
            pure_cerebellar = False
        else:  # SYNE1
            onset = rng.randint(15, 50)
            sara = round(rng.uniform(4.0, 22.0), 1)
            absent_reflexes = False
            cardiomyopathy = False
            diabetes = False
            scoliosis = False
            hearing_loss = False
            omaveloxolone_eligible = False
            spasticity = False
            oculomotor_apraxia = False
            low_albumin = False
            elevated_afp = False
            low_vitamin_e = False
            epilepsy = False
            elevated_lactate = False
            coq10_deficient = False
            pure_cerebellar = True

        cohort.append({
            "patient_id": f"{gene}-{pid+1:03d}",
            "onset_age": onset,
            "sara_score": sara,
            "absent_reflexes": absent_reflexes,
            "cardiomyopathy": cardiomyopathy,
            "diabetes": diabetes,
            "scoliosis": scoliosis,
            "hearing_loss": hearing_loss,
            "omaveloxolone_eligible": omaveloxolone_eligible,
            "spasticity": spasticity,
            "oculomotor_apraxia": oculomotor_apraxia,
            "low_albumin": low_albumin,
            "elevated_afp": elevated_afp,
            "low_vitamin_e": low_vitamin_e,
            "epilepsy": epilepsy,
            "elevated_lactate": elevated_lactate,
            "coq10_deficient": coq10_deficient,
            "pure_cerebellar": pure_cerebellar,
        })
    return cohort


_ALL_COHORTS = {
    g["gene"]: _build_cohort(g, SEED_BASE + i)
    for i, g in enumerate(AR_SCA_GENES)
}


def overview() -> dict:
    """Aggregate overview across all 8 AR-SCA genes (320 patients, seeds 2118-2125)."""
    all_patients = [p for cohort in _ALL_COHORTS.values() for p in cohort]
    n = len(all_patients)

    frda_cohort = _ALL_COHORTS["FXN"]
    sacs_cohort = _ALL_COHORTS["SACS"]
    aptx_cohort = _ALL_COHORTS["APTX"]
    setx_cohort = _ALL_COHORTS["SETX"]
    ttpa_cohort = _ALL_COHORTS["TTPA"]
    polg_cohort = _ALL_COHORTS["POLG"]
    adck3_cohort = _ALL_COHORTS["ADCK3"]
    syne1_cohort = _ALL_COHORTS["SYNE1"]

    return {
        "atlas": "Hereditary AR-SCA Atlas — Complete 8-Gene Autosomal-Recessive Cerebellar Ataxia Reference",
        "genes": [g["gene"] for g in AR_SCA_GENES],
        "total_patients": n,
        "seeds": f"2118-2125",
        "seed_base": SEED_BASE,
        # FRDA
        "frda_absent_reflexes_patients": sum(1 for p in frda_cohort if p["absent_reflexes"]),
        "frda_cardiomyopathy_patients": sum(1 for p in frda_cohort if p["cardiomyopathy"]),
        "frda_diabetes_patients": sum(1 for p in frda_cohort if p["diabetes"]),
        "frda_scoliosis_patients": sum(1 for p in frda_cohort if p["scoliosis"]),
        "frda_omaveloxolone_eligible_patients": 40,
        # SACS
        "sacs_spasticity_patients": sum(1 for p in sacs_cohort if p["spasticity"]),
        "sacs_oculomotor_apraxia_patients": sum(1 for p in sacs_cohort if p["oculomotor_apraxia"]),
        # AOA1
        "aoa1_oculomotor_apraxia_patients": sum(1 for p in aptx_cohort if p["oculomotor_apraxia"]),
        "aoa1_low_albumin_patients": sum(1 for p in aptx_cohort if p["low_albumin"]),
        # AOA2
        "aoa2_elevated_afp_patients": sum(1 for p in setx_cohort if p["elevated_afp"]),
        "aoa2_oculomotor_apraxia_patients": sum(1 for p in setx_cohort if p["oculomotor_apraxia"]),
        # AVED
        "aved_low_vitamin_e_patients": sum(1 for p in ttpa_cohort if p["low_vitamin_e"]),
        "aved_absent_reflexes_patients": sum(1 for p in ttpa_cohort if p["absent_reflexes"]),
        # POLG
        "polg_epilepsy_patients": sum(1 for p in polg_cohort if p["epilepsy"]),
        "polg_elevated_lactate_patients": sum(1 for p in polg_cohort if p["elevated_lactate"]),
        # ADCK3
        "adck3_coq10_deficient_patients": sum(1 for p in adck3_cohort if p["coq10_deficient"]),
        "adck3_elevated_lactate_patients": sum(1 for p in adck3_cohort if p["elevated_lactate"]),
        # SYNE1
        "syne1_pure_cerebellar_patients": sum(1 for p in syne1_cohort if p["pure_cerebellar"]),
        # Cross-atlas
        "valproate_ci_patients": sum(1 for p in polg_cohort if p["epilepsy"]),
        "treatable_patients": (
            sum(1 for p in ttpa_cohort if p["low_vitamin_e"]) +
            sum(1 for p in adck3_cohort if p["coq10_deficient"]) +
            sum(1 for p in frda_cohort if p["omaveloxolone_eligible"])
        ),
        "epilepsy_any_gene_patients": (
            sum(1 for p in polg_cohort if p["epilepsy"]) +
            sum(1 for p in adck3_cohort if p["epilepsy"])
        ),
        "repeat_expansion_missed_by_wes_patients": 40,  # All FRDA (GAA repeat)
    }


def breakdown() -> dict:
    """Per-gene clinical breakdown for all 8 AR-SCA genes."""
    result = {}
    for g in AR_SCA_GENES:
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
            "mean_sara": round(sum(p["sara_score"] for p in cohort) / n, 1),
            "mean_onset_age": round(sum(p["onset_age"] for p in cohort) / n, 1),
            "absent_reflexes_pct": round(100 * sum(1 for p in cohort if p["absent_reflexes"]) / n, 1),
            "cardiomyopathy_pct": round(100 * sum(1 for p in cohort if p["cardiomyopathy"]) / n, 1),
            "spasticity_pct": round(100 * sum(1 for p in cohort if p["spasticity"]) / n, 1),
            "oculomotor_apraxia_pct": round(100 * sum(1 for p in cohort if p["oculomotor_apraxia"]) / n, 1),
            "epilepsy_pct": round(100 * sum(1 for p in cohort if p["epilepsy"]) / n, 1),
            "elevated_lactate_pct": round(100 * sum(1 for p in cohort if p["elevated_lactate"]) / n, 1),
        }
    return result


def definitions() -> dict:
    """Gene definitions, glossary, and surveillance protocols."""
    return {
        "genes": {
            g["gene"]: g["protein"]
            for g in AR_SCA_GENES
        },
        "glossary": {
            "AR-SCA (Autosomal Recessive Cerebellar Ataxia)": (
                "Hereditary cerebellar ataxias caused by biallelic (homozygous or compound heterozygous) mutations. "
                "AR-SCAs differ from AD-SCAs in: earlier onset (typically childhood/adolescence), "
                "higher diagnostic yield from biochemical workup (vitamin E, AFP, albumin, lactate), "
                "and the presence of treatable subtypes (AVED, ARCA2). "
                "Friedreich ataxia (GAA repeat) is the most common, comprising ~50% of all hereditary ataxia."
            ),
            "FRDA / Friedreich Ataxia": (
                "GAA triplet repeat expansion (>66 copies) in intron 1 of FXN — biallelic. "
                "Most common hereditary ataxia (1:50,000). "
                "Absent lower limb reflexes PATHOGNOMONIC. "
                "Omaveloxolone (Skyclarys) FDA-approved 2023. "
                "Cardiomyopathy causes most deaths — annual echo + ECG MANDATORY."
            ),
            "ARSACS / Sacsin": (
                "Spastic ataxia with SPASTICITY PATHOGNOMONIC (distinguishes from all other pure ataxias). "
                "Retinal hypermyelination on OCT/fundoscopy PATHOGNOMONIC. "
                "French-Canadian founder (Charlevoix-Saguenay, Quebec). "
                "Enlarged pons + SCP on MRI = imaging signature. "
                "Baclofen/tizanidine for spasticity management."
            ),
            "AOA1 vs AOA2 Distinguishing Features": (
                "AOA1 (APTX): oculomotor apraxia + LOW ALBUMIN + HIGH CHOLESTEROL + NORMAL AFP; Japan/Portugal. "
                "AOA2 (SETX): oculomotor apraxia + ELEVATED AFP + NORMAL ALBUMIN; Southern Europe. "
                "Key: albumin and AFP separate AOA1 from AOA2. "
                "Both: peripheral neuropathy, cerebellar atrophy. "
                "Neither: cardiomyopathy, GAA repeat, low vitamin E."
            ),
            "AVED / Vitamin E Deficiency": (
                "VERY LOW SERUM VITAMIN E (<3 mg/L) — pathognomonic and treatable. "
                "Alpha-TTP liver protein secretes vitamin E into VLDL; TTPA loss → systemic deficiency. "
                "Clinical resemblance to FRDA: absent reflexes, posterior column, gait ataxia. "
                "KEY DDx from FRDA: no GAA repeat, normal cardiac, LOW vitamin E. "
                "VITAMIN E 800-1200 mg/day = level B treatment — must not miss AVED."
            ),
            "POLG Valproate Contraindication": (
                "VALPROATE ABSOLUTE CONTRAINDICATED in POLG-related ataxia. "
                "Mechanism: valproate inhibits mitochondrial beta-oxidation + depletes CoA → "
                "triggers Alpers-type fulminant hepatic failure (often fatal). "
                "RULE: in any patient with ataxia + epilepsy, exclude POLG BEFORE prescribing valproate. "
                "Safe AEDs: levetiracetam, lamotrigine, lacosamide."
            ),
            "CoQ10 Deficiency / ARCA2": (
                "ADCK3 (CABC1) kinase regulates ubiquinone (CoQ10) biosynthesis. "
                "Muscle CoQ10 assay (not plasma) is the diagnostic test. "
                "Plasma CoQ10 levels can be normal even with muscle deficiency. "
                "CoQ10 300-2400 mg/day trial — variable response; ubiquinol preferred form. "
                "Elevated lactate common (mitochondrial chain dysfunction)."
            ),
            "ARCA1 / SYNE1": (
                "Pure cerebellar ataxia with NO extracerebellar features. "
                "Normal reflexes, cognition, AFP, albumin, vitamin E, lactate. "
                "French-Canadian Quebec population most common. "
                "Slow progression — distinguishes from most other AR-SCAs. "
                "Normal life expectancy — communicate positive prognosis."
            ),
            "Repeat-Primed PCR for FXN": (
                "Standard WES (whole exome sequencing) MISSES GAA repeat expansions in FXN intron 1. "
                "Triplet-repeat primed PCR is mandatory to detect FRDA. "
                "If WES is negative in a patient with classic FRDA phenotype, "
                "request specific GAA repeat analysis SEPARATELY. "
                "This is the most common reason FRDA is missed on next-generation sequencing panels."
            ),
        },
        "surveillance_protocols": {
            "FXN (Friedreich Ataxia)": (
                "FXN GAA repeat analysis (both alleles — triplet-primed PCR); "
                "ECHO at diagnosis + annually — hypertrophic cardiomyopathy; "
                "12-lead ECG at diagnosis + annually — T-wave inversion V1-V6; "
                "HbA1c annually — diabetes mellitus 30%; "
                "Spine X-ray at diagnosis — scoliosis 60% (Cobb angle); "
                "Audiometry if hearing symptoms; ophthalmology annually (optic atrophy); "
                "SARA biannually; "
                "OMAVELOXOLONE eligibility: age ≥16 yr, ambulant; LFT baseline + 3-monthly; "
                "Physiotherapy: gait, balance, core; SLT: dysarthria; OT when driving/work affected; "
                "Genetic counselling: AR 25% sibling risk; carrier testing for siblings/partners; "
                "FRDA registry (EFACTS / FARA) enrolment"
            ),
            "SACS (ARSACS)": (
                "SACS sequencing (full coding + founder allele by PCR); "
                "OCT + dilated fundoscopy at diagnosis — retinal hypermyelination (mandatory); "
                "Annual ophthalmology (retinal surveillance); "
                "MRI brain at diagnosis — enlarged pons + SCP + T2 pontine lines; "
                "SPASTICITY management: oral baclofen (titrate 10-80 mg/day), tizanidine; "
                "Intrathecal baclofen pump referral if spasticity severe/refractory; "
                "Physiotherapy: spasticity stretching + gait + transfer training; "
                "AFO (ankle-foot orthosis) for foot drop; "
                "SLT: dysarthria + dysphagia (PEG if aspiration); "
                "SARA biannually; "
                "Genetic counselling: 25% sibling risk; Quebec community cascade testing"
            ),
            "APTX (AOA1)": (
                "APTX sequencing; "
                "SERUM ALBUMIN at diagnosis + annually — hypoalbuminaemia <35 g/L; "
                "FASTING LIPID PROFILE at diagnosis + annually — hypercholesterolaemia; "
                "Statin + dietary intervention for LDL >5 mmol/L; "
                "Neurophysiology (EMG/NCS) at diagnosis — sensorimotor axonal neuropathy; "
                "Ophthalmology: oculomotor apraxia assessment + head-turn strategy training; "
                "SARA biannually; "
                "Physiotherapy: gait + AFO for neuropathic foot drop; "
                "SLT: dysarthria; OT for ADL; "
                "Genetic counselling: 25% sibling risk; Japan/Portugal community testing; "
                "Annual review: albumin, cholesterol, neurophysiology"
            ),
            "SETX (AOA2)": (
                "SETX sequencing; "
                "SERUM AFP at diagnosis + annually — elevated AFP (disease activity marker); "
                "SERUM ALBUMIN at diagnosis — normal (distinguishes from AOA1); "
                "Neurophysiology at diagnosis — peripheral neuropathy; "
                "Ophthalmology: oculomotor apraxia assessment; "
                "Note: elevated AFP may raise teratoma concern → distinguish by genotype + imaging; "
                "SARA biannually; AFP as progression marker; "
                "Physiotherapy: gait + balance; SLT; OT; "
                "Genetic counselling: 25% sibling risk; Southern European community; "
                "EUROSCA / AOA2 registry enrolment"
            ),
            "TTPA (AVED)": (
                "TTPA sequencing; "
                "SERUM VITAMIN E at diagnosis — very low (<3 mg/L) PATHOGNOMONIC; "
                "VITAMIN E 800-1200 mg/day START IMMEDIATELY after diagnosis; "
                "Monitor serum vitamin E 3-monthly until stable (target 12-20 mg/L), then 6-monthly; "
                "NEVER STOP VITAMIN E — ataxia rapidly worsens on cessation; "
                "SARA biannually (response to treatment monitored); "
                "ECG + echo at diagnosis (cardiac involvement <10%, but check); "
                "TEST ALL SIBLINGS for serum vitamin E — presymptomatic treatment prevents onset; "
                "Physiotherapy; SLT if dysarthria; "
                "Genetic counselling: 25% sibling risk; North African community cascade"
            ),
            "POLG (POLG-Ataxia)": (
                "POLG sequencing (both alleles — compound het); "
                "Muscle biopsy: CoQ10 + mtDNA copy number + OXPHOS enzymes + histology; "
                "AVOID VALPROATE ABSOLUTELY — document CI in notes + medication alert; "
                "LEVETIRACETAM 1000-3000 mg/day as first-line AED; "
                "Lamotrigine / lacosamide as adjuncts; "
                "LFT at diagnosis + 3-monthly if on any AED; "
                "Plasma lactate at diagnosis + 6-monthly; "
                "Neurophysiology: sensory neuropathy; "
                "Ophthalmology: ptosis, ophthalmoplegia (prism glasses); "
                "Neuropsychology if cognitive decline; "
                "SARA biannually; "
                "Genetic counselling: AR 25% sibling risk; "
                "Mitochondrial specialist co-management"
            ),
            "ADCK3 (ARCA2)": (
                "ADCK3 sequencing; "
                "MUSCLE COENZYME Q10 ASSAY — not plasma (plasma CoQ10 unreliable); "
                "Plasma lactate at diagnosis + 6-monthly; "
                "COENZYME Q10 (ubiquinol) 300-2400 mg/day trial — reassess at 12 months; "
                "Muscle biopsy if CoQ10 assay unavailable (ragged red fibres, COX-negative); "
                "Riboflavin 200 mg/day (cofactor support); "
                "Physiotherapy: balance + exercise tolerance; "
                "Neuropsychology if cognitive impairment; "
                "Annual: lactate, CoQ10 level (if on treatment), SARA; "
                "Avoid prolonged fasting (lactic acidosis risk); "
                "Genetic counselling: 25% sibling risk"
            ),
            "SYNE1 (ARCA1)": (
                "SYNE1 sequencing (full gene — many truncating variants); "
                "Exclude other AR-SCAs first: serum vitamin E, AFP, albumin, lactate — all normal in ARCA1; "
                "SARA biannually; "
                "MRI brain at diagnosis (cerebellar cortical atrophy); "
                "Physiotherapy: gait, balance, fall prevention; "
                "SLT: dysarthria (baseline + if worsening); "
                "Occupational therapy for ADL; "
                "COMMUNICATE POSITIVE PROGNOSIS: pure cerebellar, slow, normal cognition, normal life expectancy; "
                "Driving assessment: slow progression — often ambulant for 20+ yr; "
                "Genetic counselling: 25% sibling risk; Quebec/North African community; "
                "EUROSCA / RISCA registry enrolment"
            ),
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"FRDA absent reflexes: {ov['frda_absent_reflexes_patients']}")
    print(f"FRDA cardiomyopathy: {ov['frda_cardiomyopathy_patients']}")
    print(f"FRDA omaveloxolone eligible: {ov['frda_omaveloxolone_eligible_patients']}")
    print(f"SACS spasticity: {ov['sacs_spasticity_patients']}")
    print(f"AOA1 low albumin: {ov['aoa1_low_albumin_patients']}")
    print(f"AOA2 elevated AFP: {ov['aoa2_elevated_afp_patients']}")
    print(f"AVED low vitamin E: {ov['aved_low_vitamin_e_patients']}")
    print(f"POLG epilepsy: {ov['polg_epilepsy_patients']}")
    print(f"ADCK3 CoQ10 deficient: {ov['adck3_coq10_deficient_patients']}")
    print(f"SYNE1 pure cerebellar: {ov['syne1_pure_cerebellar_patients']}")
    print(f"Valproate CI patients (POLG): {ov['valproate_ci_patients']}")
    print(f"Treatable patients: {ov['treatable_patients']}")
