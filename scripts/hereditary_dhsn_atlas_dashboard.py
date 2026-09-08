#!/usr/bin/env python3
"""Hereditary-dHMN-Atlas — Complete 8-Gene Distal Hereditary Motor Neuropathy Atlas
(dHMN2B/HSPB1 · dHMN2A/HSPB8 · dHMN5A/GARS1 · dHMN7B/DCTN1 ·
 SMALED1/BICD2 · SMALED1/DYNC1H1 · SMARD1/IGHMBP2 · SMALED2/TRPV4).

HSPB1   (Heat Shock Protein 27 / HSP27; 640 aa; 7q11.23; AD;
          dHMN2B / CMT2F (with sensory) — adult-onset distal lower-limb weakness → foot drop → steppage gait;
          Distal upper-limb wasting follows (wrist drop); NO sensory loss distinguishes from CMT;
          Small heat-shock chaperone — mutant HSPB1 misfolds and forms cytoplasmic aggregates;
          pSer135Phe and pArg136Trp most common; NCS: motor axonal, SNAPs preserved;
          seed SEED_BASE+0).
HSPB8   (Heat Shock Protein 22 / HSP22; 196 aa; 12q24.23; AD;
          dHMN2A / CMT2L (with sensory) — juvenile onset; distal lower > upper weakness;
          pLys141Asn / pLys141Glu European founders; allelic with CMT2L;
          NCS: motor-axonal; slow progression; wheelchair rare; seed SEED_BASE+1).
GARS1   (Glycyl-tRNA Synthetase; 685 aa; 7p14.3; AD;
          dHMN5A / CMT2D — UPPER LIMB PREDOMINANT: thenar + interosseous wasting PATHOGNOMONIC;
          Juvenile onset 10-20 yr; phrenic nerve 20-30% → respiratory monitoring mandatory;
          Aminoacyl-tRNA synthetase — dominant-negative inhibition of neuronal tRNA charging;
          seed SEED_BASE+2).
DCTN1   (Dynactin p150glued; 1278 aa; 2p13.1; AD;
          dHMN7B — VOCAL FOLD PARALYSIS PATHOGNOMONIC: hoarse voice → stridor → respiratory failure;
          Laryngoscopy MANDATORY at diagnosis; pGly59Ser European founder for dHMN7B;
          Retrograde axonal transport adaptor; severe cases overlap with ALS14;
          seed SEED_BASE+3).
BICD2   (Bicaudal D Cargo Adaptor 2; 820 aa; 9q22.31; AD (de novo ~40%);
          SMALED1 — CONGENITAL LOWER-LIMB CONTRACTURES + HIP DISLOCATION at birth PATHOGNOMONIC;
          Lower-extremity predominant; intelligence normal; non-progressive or slowly progressive;
          Golgi-dynein cargo adaptor; pSer107Leu most common; seed SEED_BASE+4).
DYNC1H1 (Dynein Cytoplasmic 1 Heavy Chain 1; 4646 aa; 14q32.31; AD (de novo ~60%);
          SMALED1 — LOWER EXTREMITY SMA + INTELLECTUAL DISABILITY 30% PATHOGNOMONIC combination;
          Pachygyria/lissencephaly in severe de novo; hip subluxation from proximal weakness;
          pHis306Arg and pLys671Glu common variants; seed SEED_BASE+5).
IGHMBP2 (Immunoglobulin Mu DNA Binding Protein 2; 993 aa; 11q13.3; AR;
          SMARD1 / dHMN6 — INFANTILE RESPIRATORY FAILURE (3-6 months) PATHOGNOMONIC;
          Diaphragmatic palsy + phrenic nerve → NIV/tracheostomy MANDATORY at diagnosis;
          Paradoxical breathing (chest paradox) at rest; death <2 yr without ventilation;
          RNA helicase; pArg318Gln most common; seed SEED_BASE+6).
TRPV4   (Transient Receptor Potential Vanilloid 4; 871 aa; 12q24.11; AD;
          SMALED2 / CMT2C — SKELETAL DYSPLASIA + MOTOR NEUROPATHY PATHOGNOMONIC combination;
          Vocal cord paralysis + hearing loss; brachyolmia / metatropic dysplasia spectrum;
          Mechano-sensitive Ca2+ channel; pArg269His commonest neuropathy variant;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2134-2141).
"""

import random

SEED_BASE = 2134

DHSN_GENES = [
    # -- HSPB1 — dHMN2B --------------------------------------------------------
    {
        "gene": "HSPB1",
        "alt_name": (
            "HSPB1 (HSPB1-640aa-7q11.23 / AD — dHMN2B-Distal-Hereditary-Motor-Neuropathy-Type-2B — "
            "Adult-Onset-Foot-Drop-Steppage-Gait — Intrinsic-Hand-Wasting — "
            "NO-Sensory-Loss-DDx-CMT2F — Small-Heat-Shock-Chaperone-Aggregates)"
        ),
        "protein": (
            "HSPB1 -- 7q11.23 AD -- HSPB1-640aa -- "
            "Heat-Shock-Protein-27-HSP27-Small-Heat-Shock-Protein-Chaperone-Family -- "
            "dHMN2B-Distal-Hereditary-Motor-Neuropathy-Type-2B-OMIM-158590 -- "
            "CMT2F-When-Sensory-Involvement-OMIM-606595 -- "
            "Dominant-Mutations-pSer135Phe-pArg136Trp-pPro182Leu-Cause-Misfolding-Aggregation -- "
            "Adult-Onset-20-40yr-Distal-Lower-Limb-Weakness-First-Foot-Drop-Steppage-Gait -- "
            "Intrinsic-Hand-Muscle-Wasting-Wrist-Drop-Later -- "
            "NO-SENSORY-LOSS-Key-DDx-from-CMT2F-Allelic-Condition -- "
            "NCS-Motor-Axonal-SNAPs-Preserved-or-Mildly-Reduced -- "
            "Normal-CK-Normal-Cognitive-Function -- "
            "Slow-Progression-Ambulatory-for-Decades -- "
            "No-Disease-Modifying-Therapy -- "
            "pSer135Phe-Most-Common-7q11.23"
        ),
        "locus": "7q11.23",
        "protein_size": "640 aa",
        "inheritance": (
            "AD (autosomal dominant); missense gain-of-toxic-function; "
            "mutations cluster in alpha-crystallin domain; "
            "reduced penetrance reported; "
            "genetic testing: NGS panel covering HSPB1 + HSPB8 simultaneously; "
            "key alleles: pSer135Phe (most common), pArg136Trp, pPro182Leu"
        ),
        "age_of_onset": "20-40 years (adult onset); rarely juvenile",
        "pathognomonic": (
            "DISTAL LOWER LIMB WEAKNESS WITH FOOT DROP — slow progressive steppage gait; "
            "intrinsic hand muscle wasting later; "
            "NO sensory loss distinguishes dHMN2B from CMT2F (allelic); "
            "NCS: motor axonal (CMAP reduced); SNAPs preserved; "
            "EMG: active denervation distal muscles; "
            "normal CK; no tongue fasciculations (excludes ALS)"
        ),
        "treatment": (
            "NO disease-modifying treatment available; "
            "ANKLE-FOOT ORTHOSES (AFOs): custom foot drop splints — reduce trip/fall risk; "
            "PHYSIOTHERAPY: gait training, muscle strengthening proximal muscles; "
            "OCCUPATIONAL THERAPY: hand function assessment + adaptive equipment; "
            "RESPIRATORY: spirometry 2-yearly (phrenic nerve rare); "
            "FALLS PREVENTION: home assessment, handrails, non-slip flooring; "
            "GENETIC COUNSELLING: 50% offspring risk; testing from age 18 yr; "
            "No need for wheelchair in first 20 yr in most; "
            "Avoid statins (may worsen CK elevation in motor neuropathy)"
        ),
        "key_biomarker": (
            "NCS: reduced CMAP amplitudes distal muscles; SNAPs normal/near-normal; "
            "EMG: active denervation + chronic reinnervation distal lower limb; "
            "CK: normal or mildly elevated (<3× ULN); "
            "HSPB1 sequencing: panel NGS"
        ),
        "critical_flags": [
            "NO-SENSORY-LOSS-DDx-CMT2F",
            "FOOT-DROP-STEPPAGE-GAIT-ADULT-ONSET",
            "SNAPS-PRESERVED-KEY-DDx",
            "ALLELIC-CMT2F-SENSORY-INVOLVED",
            "SLOW-PROGRESSION-AMBULATORY-20yr",
            "AVOID-STATINS-CK-RISK",
            "NO-DISEASE-MODIFYING-RX",
            "AFO-REDUCES-FALLS",
            "CASCADE-TESTING-50pct-RISK",
        ],
    },
    # -- HSPB8 — dHMN2A --------------------------------------------------------
    {
        "gene": "HSPB8",
        "alt_name": (
            "HSPB8 (HSPB8-196aa-12q24.23 / AD — dHMN2A-Distal-Hereditary-Motor-Neuropathy-Type-2A — "
            "Juvenile-Onset-Distal-Leg-Weakness — pLys141Asn-European-Founder — "
            "Allelic-CMT2L-Sensory-Variant)"
        ),
        "protein": (
            "HSPB8 -- 12q24.23 AD -- HSPB8-196aa -- "
            "Heat-Shock-Protein-22-HSP22-Small-Heat-Shock-Protein-Chaperone-BAG3-Partner -- "
            "dHMN2A-Distal-Hereditary-Motor-Neuropathy-Type-2A-OMIM-158590 -- "
            "CMT2L-When-Sensory-Involvement-OMIM-608673 -- "
            "pLys141Asn-pLys141Glu-European-Founder-Mutations -- "
            "Juvenile-Onset-10-25yr-Distal-Lower-Limb-Weakness-First -- "
            "Upper-Limb-Wasting-Follows-in-2nd-Decade -- "
            "NO-SENSORY-LOSS-dHMN2A-Allelic-CMT2L-Has-Sensory -- "
            "NCS-Motor-Axonal-Normal-SNAPs -- "
            "Slow-Progression-Rare-Wheelchair -- "
            "Co-aggregates-with-HSPB1-in-Neuronal-Inclusions -- "
            "12q24.23"
        ),
        "locus": "12q24.23",
        "protein_size": "196 aa",
        "inheritance": (
            "AD (autosomal dominant); hotspot mutations Lys141; "
            "pLys141Asn and pLys141Glu are European founders; "
            "allelic with CMT2L (dHMN2A + sensory = CMT2L); "
            "genetic testing: NGS panel; Sanger for Lys141 hotspot"
        ),
        "age_of_onset": "10-25 years (juvenile onset)",
        "pathognomonic": (
            "JUVENILE-ONSET distal lower limb weakness (dHMN2A) beginning 10-25 yr; "
            "upper limb involved later; "
            "NCS: motor axonal, SNAPs normal; "
            "EMG: distal denervation lower + upper limb; "
            "pLys141Asn / pLys141Glu founder mutations in European ancestry; "
            "allelic condition CMT2L has same variants + sensory involvement"
        ),
        "treatment": (
            "NO disease-modifying therapy; "
            "AFOs for foot drop when present; "
            "PHYSIOTHERAPY: strengthening + gait; "
            "HAND THERAPY: writing aids + grip assistance; "
            "SCHOOL/OCCUPATIONAL: support with fine motor tasks early; "
            "GENETIC COUNSELLING: 50% offspring risk; "
            "Prognosis: slow progression, majority ambulant into 5th-6th decade"
        ),
        "key_biomarker": (
            "NCS: motor axonal (CMAP reduction distal); SNAPs normal; "
            "EMG: distal denervation + reinnervation; "
            "CK: normal; "
            "HSPB8 Lys141 hotspot Sanger or panel NGS"
        ),
        "critical_flags": [
            "JUVENILE-ONSET-10-25yr",
            "pLys141Asn-pLys141Glu-EUROPEAN-FOUNDERS",
            "ALLELIC-CMT2L-IF-SENSORY-PRESENT",
            "NO-SENSORY-dHMN2A",
            "SLOW-PROGRESSION-AMBULANT-DECADES",
            "NO-DISEASE-MODIFYING-RX",
            "AFO-FOOT-DROP",
            "CASCADE-TESTING-50pct",
        ],
    },
    # -- GARS1 — dHMN5A --------------------------------------------------------
    {
        "gene": "GARS1",
        "alt_name": (
            "GARS1 (GARS1-685aa-7p14.3 / AD — dHMN5A-Distal-Hereditary-Motor-Neuropathy-Type-5A — "
            "UPPER-LIMB-PREDOMINANT-Thenar-Interosseous-Wasting-PATHOGNOMONIC — "
            "Phrenic-Nerve-20-30pct-Respiratory-Monitoring-Mandatory — "
            "Aminoacyl-tRNA-Synthetase-Dominant-Negative)"
        ),
        "protein": (
            "GARS1 -- 7p14.3 AD -- GARS1-685aa -- "
            "Glycyl-tRNA-Synthetase-GlyRS-Aminoacyl-tRNA-Synthetase-ARS-Family -- "
            "dHMN5A-Distal-Hereditary-Motor-Neuropathy-Type-5A-OMIM-600287 -- "
            "CMT2D-When-Sensory-Involvement -- "
            "UPPER-LIMB-PREDOMINANT-THENAR-HYPOTHENAR-INTEROSSEOUS-WASTING-PATHOGNOMONIC -- "
            "Juvenile-Onset-10-20yr-Hand-Weakness-Before-Foot-Drop -- "
            "Dominant-Negative-Mechanism-Inhibits-Neuronal-tRNA-Charging -- "
            "Phrenic-Nerve-Involvement-20-30pct-Spirometry-Mandatory -- "
            "NCS-Predominantly-Motor-Axonal-Upper-Limb -- "
            "ARS-Gene-Family-Multiple-Genes-Cause-Similar-Neuropathy -- "
            "7p14.3"
        ),
        "locus": "7p14.3",
        "protein_size": "685 aa",
        "inheritance": (
            "AD (autosomal dominant); dominant-negative mechanism; "
            "mutations inhibit neuronal tRNA aminoacylation; "
            "part of the ARS (aminoacyl-tRNA synthetase) gene family causing dHMN; "
            "genetic testing: GARS1 sequencing; consider full ARS panel (AARS1, YARS1, MARS1) "
            "when upper-limb dHMN phenotype"
        ),
        "age_of_onset": "10-20 years (juvenile to young adult onset)",
        "pathognomonic": (
            "UPPER LIMB PREDOMINANT: thenar eminence + first dorsal interosseous wasting first = PATHOGNOMONIC; "
            "hand weakness precedes foot drop (reverse of most dHMNs); "
            "wrist drop + finger extension weakness; "
            "phrenic nerve involvement 20-30% → paradoxical breathing → FVC < 70%; "
            "NCS: predominantly upper limb motor axonal; "
            "key ARS gene: when upper-limb-first dHMN → test GARS1 + full ARS panel"
        ),
        "treatment": (
            "NO disease-modifying therapy; "
            "RESPIRATORY: spirometry 6-monthly (phrenic nerve risk); "
            "if FVC < 70%: sleep study (nocturnal hypoventilation); NIV if indicated; "
            "HAND THERAPY: splints + adaptive aids for grip; "
            "AFOs if foot drop develops; "
            "PHYSIOTHERAPY: upper and lower limb strengthening; "
            "OCCUPATIONAL THERAPY: computer access, writing aids, kitchen aids; "
            "ARS GENE PANEL if upper-limb-first phenotype (AARS1, YARS1, MARS1 also cause dHMN); "
            "GENETIC COUNSELLING: 50% offspring risk"
        ),
        "key_biomarker": (
            "NCS: upper > lower limb motor axonal; SNAPs may be reduced (dHMN5A vs CMT2D); "
            "Spirometry/FVC: phrenic nerve monitoring; "
            "Chest X-ray: elevated hemidiaphragm (phrenic palsy); "
            "GARS1 sequencing + ARS panel"
        ),
        "critical_flags": [
            "UPPER-LIMB-PREDOMINANT-PATHOGNOMONIC",
            "THENAR-INTEROSSEOUS-WASTING-FIRST",
            "PHRENIC-NERVE-20-30pct-FVC-MANDATORY",
            "NIV-IF-FVC-BELOW-70pct",
            "ARS-GENE-FAMILY-PANEL-UPPER-LIMB-dHMN",
            "JUVENILE-ONSET-10-20yr",
            "NO-DISEASE-MODIFYING-RX",
            "RESPIRATORY-SURVEILLANCE-PRIORITY",
        ],
    },
    # -- DCTN1 — dHMN7B --------------------------------------------------------
    {
        "gene": "DCTN1",
        "alt_name": (
            "DCTN1 (DCTN1-1278aa-2p13.1 / AD — dHMN7B-Distal-Hereditary-Motor-Neuropathy-Type-7B — "
            "VOCAL-FOLD-PARALYSIS-PATHOGNOMONIC-Hoarse-Voice-Stridor-Respiratory-Failure — "
            "Laryngoscopy-MANDATORY-at-Diagnosis — pGly59Ser-European-Founder)"
        ),
        "protein": (
            "DCTN1 -- 2p13.1 AD -- DCTN1-1278aa -- "
            "Dynactin-1-p150glued-Retrograde-Axonal-Transport-Adaptor-Dynein-Activator -- "
            "dHMN7B-Distal-Hereditary-Motor-Neuropathy-Type-7B-OMIM-607641 -- "
            "ALS14-When-UMN-Involvement-OMIM-613954 -- "
            "VOCAL-FOLD-PARALYSIS-HOARSE-VOICE-STRIDOR-RESPIRATORY-FAILURE-PATHOGNOMONIC -- "
            "Laryngoscopy-MANDATORY-at-Diagnosis-to-assess-arytenoid-mobility -- "
            "pGly59Ser-European-Founder-Mutation-dHMN7B -- "
            "Retrograde-Dynein-Motor-Axonal-Transport-Failure-Motor-Neurons -- "
            "Adult-Onset-20-30yr -- "
            "Severe-Cases-Overlap-ALS14-UMN-LMN -- "
            "2p13.1"
        ),
        "locus": "2p13.1",
        "protein_size": "1278 aa",
        "inheritance": (
            "AD (autosomal dominant); pGly59Ser is the predominant dHMN7B founder mutation; "
            "different mutations cause Perry syndrome (parkinsonism + hypoventilation); "
            "severe variants may cause ALS14 (UMN + LMN); "
            "genetic testing: DCTN1 sequencing; note mutation-phenotype map"
        ),
        "age_of_onset": "20-35 years (adult onset)",
        "pathognomonic": (
            "VOCAL FOLD PARALYSIS: hoarse voice → stridor → respiratory failure = PATHOGNOMONIC for dHMN7B; "
            "laryngoscopy shows bilateral arytenoid/vocal fold immobility; "
            "ENT referral MANDATORY at diagnosis; "
            "respiratory failure may precede limb weakness; "
            "NCS: motor axonal; EMG: denervation; "
            "distal upper + lower limb weakness 2nd-3rd decade"
        ),
        "treatment": (
            "LARYNGOSCOPY MANDATORY at diagnosis — vocal fold mobility assessment; "
            "ENT surveillance 6-monthly: airway safety; "
            "TRACHEOSTOMY if bilateral vocal fold paralysis causes stridor or severe airway compromise; "
            "RESPIRATORY: spirometry 6-monthly; NIV early (dyspnoea on exertion); "
            "SPEECH THERAPY: dysphagia assessment + safe swallowing; "
            "VIDEO FLUOROSCOPY: swallowing study if dysphagia; "
            "AFOs for foot drop; "
            "PHYSIOTHERAPY: limb strengthening; "
            "GENETIC COUNSELLING: 50% offspring risk; "
            "Distinguish pGly59Ser (dHMN7B) from Perry syndrome DCTN1 mutations (parkinsonism)"
        ),
        "key_biomarker": (
            "LARYNGOSCOPY: vocal fold / arytenoid mobility — abnormal in dHMN7B; "
            "FVC: respiratory monitoring; "
            "NCS: motor axonal; "
            "DCTN1 sequencing (pGly59Ser = dHMN7B founder)"
        ),
        "critical_flags": [
            "VOCAL-FOLD-PARALYSIS-PATHOGNOMONIC",
            "LARYNGOSCOPY-MANDATORY-AT-DIAGNOSIS",
            "TRACHEOSTOMY-IF-BILATERAL-PALSY",
            "RESPIRATORY-FAILURE-MAY-PRECEDE-LIMB",
            "pGly59Ser-EUROPEAN-FOUNDER-dHMN7B",
            "PERRY-SYNDROME-DDx-DCTN1-PARKINSONISM",
            "ALS14-SEVERE-OVERLAP-UMN-LMN",
            "ENT-6-MONTHLY",
        ],
    },
    # -- BICD2 — SMALED1 --------------------------------------------------------
    {
        "gene": "BICD2",
        "alt_name": (
            "BICD2 (BICD2-820aa-9q22.31 / AD-de-novo-40pct — SMALED1-Spinal-Muscular-Atrophy-Lower-Extremity-Dominant-1 — "
            "CONGENITAL-LOWER-LIMB-CONTRACTURES-HIP-DISLOCATION-PATHOGNOMONIC — "
            "Normal-Intelligence — Non-Progressive)"
        ),
        "protein": (
            "BICD2 -- 9q22.31 AD (de novo ~40%) -- BICD2-820aa -- "
            "Bicaudal-D-Cargo-Adaptor-2-Golgi-Dynein-Motor-Linker -- "
            "SMALED1-Spinal-Muscular-Atrophy-Lower-Extremity-Dominant-Type-1-OMIM-158600 -- "
            "CONGENITAL-LOWER-LIMB-CONTRACTURES-ARTHROGRYPOSIS-HIP-DISLOCATION-AT-BIRTH-PATHOGNOMONIC -- "
            "Lower-Extremity-Predominant-SMA-from-Birth -- "
            "Normal-Intelligence-Cognitive-Function-Preserved -- "
            "Non-Progressive-or-Slowly-Progressive -- "
            "De-Novo-Mutations-~40pct-No-Family-History-Does-Not-Exclude -- "
            "pSer107Leu-pGlu774Gly-Most-Common-Variants -- "
            "Golgi-Fragmentation-Impaired-Dynein-Cargo-Adaption -- "
            "9q22.31"
        ),
        "locus": "9q22.31",
        "protein_size": "820 aa",
        "inheritance": (
            "AD (autosomal dominant); de novo mutations account for ~40%; "
            "negative family history does NOT exclude BICD2; "
            "pSer107Leu and pGlu774Gly are recurrent; "
            "genetic testing: NGS panel or trio WES (de novo detection)"
        ),
        "age_of_onset": "Congenital / neonatal (birth)",
        "pathognomonic": (
            "CONGENITAL LOWER LIMB CONTRACTURES (arthrogryposis) + hip dislocation at birth = PATHOGNOMONIC; "
            "purely lower extremity dominant SMA from birth; "
            "upper limbs spared; "
            "intelligence normal (important for counselling); "
            "non-progressive or only slowly progressive course; "
            "NCS: motor axonal lower limb; EMG: chronic neurogenic changes; "
            "de novo mutation ~40% — negative family history does not exclude diagnosis"
        ),
        "treatment": (
            "ORTHOPAEDIC: hip surveillance — Pavlik harness / surgical reduction for hip dislocation; "
            "serial casting for lower limb contractures neonatally; "
            "PHYSIOTHERAPY: passive + active lower limb mobilisation; "
            "ORTHOTICS: AFOs, knee-ankle-foot orthoses; "
            "DEVELOPMENTAL: normal intelligence — mainstream school appropriate; "
            "SEATING AND MOBILITY: power wheelchair if non-ambulant; "
            "RESPIRATORY: spirometry 2-yearly (usually normal); "
            "GENETIC COUNSELLING: trio WES for de novo; 50% offspring risk for confirmed AD parent; "
            "PROGNOSIS: non-progressive in most; many ambulant with aids"
        ),
        "key_biomarker": (
            "Congenital hip dislocation + arthrogryposis lower limbs on neonatal imaging; "
            "EMG: chronic neurogenic lower limb; NCS: motor axonal; "
            "Muscle MRI: lower limb selective denervation; "
            "BICD2 sequencing (trio WES for de novo)"
        ),
        "critical_flags": [
            "CONGENITAL-HIP-DISLOCATION-PATHOGNOMONIC",
            "LOWER-LIMB-ONLY-NEVER-UPPER-LIMB-DOMINANT",
            "NORMAL-INTELLIGENCE",
            "DE-NOVO-40pct-NO-FAMILY-HISTORY-DOES-NOT-EXCLUDE",
            "NON-PROGRESSIVE-GOOD-PROGNOSIS",
            "ORTHOPAEDIC-HIP-SURVEILLANCE-MANDATORY",
            "TRIO-WES-DE-NOVO-DETECTION",
        ],
    },
    # -- DYNC1H1 — SMALED1 -------------------------------------------------------
    {
        "gene": "DYNC1H1",
        "alt_name": (
            "DYNC1H1 (DYNC1H1-4646aa-14q32.31 / AD-de-novo-60pct — SMALED1-Spinal-Muscular-Atrophy-Lower-Extremity-Dominant — "
            "LOWER-EXTREMITY-SMA-PLUS-INTELLECTUAL-DISABILITY-30pct-PATHOGNOMONIC — "
            "Hip-Subluxation — Pachygyria-Severe-De-Novo)"
        ),
        "protein": (
            "DYNC1H1 -- 14q32.31 AD (de novo ~60%) -- DYNC1H1-4646aa -- "
            "Dynein-Cytoplasmic-1-Heavy-Chain-1-Retrograde-Axonal-Transport-Motor -- "
            "SMALED1-Spinal-Muscular-Atrophy-Lower-Extremity-Dominant-OMIM-158600 -- "
            "Mental-Retardation-AD13-OMIM-614563-Overlapping-Phenotype -- "
            "LOWER-EXTREMITY-PREDOMINANT-SMA-FROM-CHILDHOOD -- "
            "INTELLECTUAL-DISABILITY-LEARNING-DIFFICULTIES-30pct-PATHOGNOMONIC-COMBINATION -- "
            "Pachygyria-Lissencephaly-in-Severe-De-Novo-Cases -- "
            "Hip-Subluxation-from-Proximal-Weakness -- "
            "pHis306Arg-pLys671Glu-Recurrent-Variants -- "
            "Largest-Gene-in-This-Atlas-4646aa -- "
            "Retrograde-Dynein-Fails-Sensorimotor-Neuron-Survival -- "
            "14q32.31"
        ),
        "locus": "14q32.31",
        "protein_size": "4646 aa",
        "inheritance": (
            "AD (autosomal dominant); de novo mutations ~60%; "
            "trio WES preferred for sporadic cases; "
            "pHis306Arg and pLys671Glu are recurrent pathogenic variants; "
            "severe CNS phenotype (pachygyria/lissencephaly) = de novo; "
            "genetic testing: trio WES; note DYNC1H1 is large gene"
        ),
        "age_of_onset": "Childhood (1-10 years)",
        "pathognomonic": (
            "LOWER EXTREMITY SMA + INTELLECTUAL DISABILITY/LEARNING DIFFICULTIES 30% = PATHOGNOMONIC combination; "
            "proximal lower > distal lower weakness; "
            "hip subluxation/dislocation from proximal weakness; "
            "pachygyria or lissencephaly in severe de novo cases; "
            "NCS: motor axonal lower limb; "
            "EMG: chronic neurogenic lower limb; "
            "de novo mutation ~60% — no family history common"
        ),
        "treatment": (
            "ORTHOPAEDIC: hip surveillance (subluxation risk); "
            "serial orthopaedic review + hip X-ray 6-monthly; "
            "surgical hip stabilisation if progressive subluxation; "
            "PHYSIOTHERAPY: lower limb strengthening + mobility; "
            "EDUCATIONAL: intellectual support, EHCP (education, health, care plan); "
            "NEUROLOGY: MRI brain if cognitive delay (pachygyria); "
            "EPILEPSY: EEG if suspected (pachygyria → seizures 30%); "
            "SEATING: power wheelchair for proximal weakness; "
            "RESPIRATORY: spirometry 2-yearly; "
            "GENETIC COUNSELLING: trio WES; 50% offspring risk from confirmed AD parent"
        ),
        "key_biomarker": (
            "MRI brain: pachygyria/lissencephaly (severe de novo); "
            "Hip X-ray: subluxation; "
            "EMG/NCS: lower limb motor neurogenic; "
            "DYNC1H1 sequencing (trio WES preferred)"
        ),
        "critical_flags": [
            "INTELLECTUAL-DISABILITY-30pct-PATHOGNOMONIC",
            "LOWER-EXTREMITY-PREDOMINANT",
            "HIP-SUBLUXATION-ORTHOPAEDIC-SURVEILLANCE",
            "PACHYGYRIA-LISSENCEPHALY-SEVERE-DE-NOVO",
            "DE-NOVO-60pct-NO-FAMILY-HISTORY",
            "TRIO-WES-PREFERRED",
            "EPILEPSY-30pct-IF-PACHYGYRIA",
            "EDUCATIONAL-SUPPORT-MANDATORY",
        ],
    },
    # -- IGHMBP2 — SMARD1 -------------------------------------------------------
    {
        "gene": "IGHMBP2",
        "alt_name": (
            "IGHMBP2 (IGHMBP2-993aa-11q13.3 / AR — SMARD1-Spinal-Muscular-Atrophy-Respiratory-Distress-1 — "
            "INFANTILE-RESPIRATORY-FAILURE-3-6-MONTHS-PATHOGNOMONIC — "
            "Diaphragmatic-Palsy-NIV-Tracheostomy-MANDATORY — "
            "Death-Before-2yr-Without-Ventilation)"
        ),
        "protein": (
            "IGHMBP2 -- 11q13.3 AR -- IGHMBP2-993aa -- "
            "Immunoglobulin-Mu-Binding-Protein-2-RNA-Helicase-Mitochondrial-tRNA-Processing -- "
            "SMARD1-Spinal-Muscular-Atrophy-with-Respiratory-Distress-Type-1-OMIM-604320 -- "
            "dHMN6-Milder-Juvenile-Onset-CMT2S-OMIM-616155 -- "
            "INFANTILE-RESPIRATORY-FAILURE-3-6-MONTHS-PATHOGNOMONIC -- "
            "Diaphragmatic-Palsy-Phrenic-Nerve-Paralysis -- "
            "NIV-TRACHEOSTOMY-MANDATORY-AT-DIAGNOSIS -- "
            "Paradoxical-Breathing-Chest-Paradox-at-Rest -- "
            "Death-Before-2yr-Without-Ventilation -- "
            "pArg318Gln-Most-Common-AR-Mutation -- "
            "Milder-Alleles-Present-as-dHMN6-CMT2S-Juvenile-Onset -- "
            "11q13.3"
        ),
        "locus": "11q13.3",
        "protein_size": "993 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "pArg318Gln most common pathogenic variant; "
            "milder alleles cause dHMN6 / CMT2S (juvenile onset); "
            "SMARD1 = severe alleles; "
            "genetic testing: IGHMBP2 sequencing + deletion/duplication analysis"
        ),
        "age_of_onset": "3-6 months (infantile) for SMARD1; 1-20 yr for dHMN6/CMT2S",
        "pathognomonic": (
            "INFANTILE RESPIRATORY FAILURE at 3-6 months = PATHOGNOMONIC for SMARD1; "
            "diaphragmatic palsy → chest paradox (chest caves in on inspiration); "
            "paradoxical breathing (abdomen rises, chest falls) at rest; "
            "phrenic nerve electromyography: absent phrenic CMAP; "
            "distal motor neuropathy at 6-12 months follows respiratory onset; "
            "chest X-ray: elevated hemidiaphragm(s); "
            "death by 2 years WITHOUT ventilatory support"
        ),
        "treatment": (
            "NIV/TRACHEOSTOMY MANDATORY at diagnosis: "
            "respiratory support — CPAP/BiPAP in mild cases; tracheostomy in severe; "
            "CHEST PHYSIOTHERAPY: airway clearance; cough assist; "
            "DIAPHRAGM PACING: phrenic nerve stimulation if candidates; "
            "GASTROSTOMY: feeding support (NG initially, PEG at 4-6 months); "
            "PHYSIOTHERAPY: limb mobilisation; prevent contractures; "
            "MULTIDISCIPLINARY CARE: respiratory, neurology, gastroenterology, orthopaedics, genetics; "
            "NUSINERSEN/GENE THERAPY: NOT SMA1 — IGHMBP2, not SMN1; no SMN therapies apply; "
            "GENETIC COUNSELLING: 25% sibling risk; carrier testing parents; "
            "PROGNOSIS: home ventilation → survival into school age and beyond"
        ),
        "key_biomarker": (
            "Chest X-ray: elevated hemidiaphragm; "
            "Phrenic nerve NCS: absent CMAP bilaterally; "
            "EMG: distal + phrenic denervation; "
            "Chest ultrasound: paradoxical diaphragm movement; "
            "IGHMBP2 sequencing"
        ),
        "critical_flags": [
            "INFANTILE-RESPIRATORY-FAILURE-PATHOGNOMONIC",
            "NIV-TRACHEOSTOMY-MANDATORY-AT-DIAGNOSIS",
            "DIAPHRAGM-PHRENIC-NERVE-PALSY",
            "PARADOXICAL-BREATHING-CHEST-PARADOX",
            "DEATH-BEFORE-2yr-WITHOUT-VENTILATION",
            "NOT-SMA1-NUSINERSEN-DOES-NOT-APPLY",
            "CHEST-XRAY-ELEVATED-HEMIDIAPHRAGM",
            "SMARD1-vs-dHMN6-ALLELE-SEVERITY",
            "25pct-SIBLING-RISK-AR",
        ],
    },
    # -- TRPV4 — SMALED2 -------------------------------------------------------
    {
        "gene": "TRPV4",
        "alt_name": (
            "TRPV4 (TRPV4-871aa-12q24.11 / AD — SMALED2-Spinal-Muscular-Atrophy-Lower-Extremity-Dominant-2 — "
            "SKELETAL-DYSPLASIA-PLUS-MOTOR-NEUROPATHY-PATHOGNOMONIC — "
            "Vocal-Cord-Paralysis-Hearing-Loss — "
            "pArg269His-Commonest-Neuropathy-Variant)"
        ),
        "protein": (
            "TRPV4 -- 12q24.11 AD -- TRPV4-871aa -- "
            "Transient-Receptor-Potential-Vanilloid-4-Mechano-Sensing-Calcium-Channel -- "
            "SMALED2-Spinal-Muscular-Atrophy-Lower-Extremity-Dominant-Type-2-OMIM-615290 -- "
            "CMT2C-When-Sensory-Involvement-OMIM-606071 -- "
            "SKELETAL-DYSPLASIA-BRACHYOLMIA-METATROPIC-SED-PLUS-MOTOR-NEUROPATHY-PATHOGNOMONIC -- "
            "Vocal-Cord-Paralysis-Sensorineural-Hearing-Loss -- "
            "pArg269His-Most-Common-Neuropathy-Variant-Lower-Extremity-SMA -- "
            "Severe-Variants-Metatropic-Dysplasia-Lethal-Skeletal -- "
            "NCS-Motor-Axonal-EMG-Selective-Denervation -- "
            "Mechano-Sensitive-Ca2plus-Channel-Gain-of-Function -- "
            "12q24.11"
        ),
        "locus": "12q24.11",
        "protein_size": "871 aa",
        "inheritance": (
            "AD (autosomal dominant); gain-of-function calcium channel; "
            "pArg269His most common neuropathy variant (SMALED2/CMT2C); "
            "severe variants (pArg315Trp, pArg594His) cause metatropic dysplasia (lethal skeletal); "
            "genetic testing: TRPV4 sequencing; note genotype-phenotype correlation"
        ),
        "age_of_onset": "Childhood (1-10 years) for SMALED2; neonatal for severe skeletal variants",
        "pathognomonic": (
            "SKELETAL DYSPLASIA + MOTOR NEUROPATHY in same patient = PATHOGNOMONIC combination; "
            "brachyolmia (short trunk, short stature) + lower extremity SMA; "
            "vocal cord paralysis (similar to DCTN1) → hoarse voice → respiratory issues; "
            "sensorineural hearing loss; "
            "NCS: motor axonal lower > upper; "
            "muscle MRI: selective lower limb denervation; "
            "pArg269His = most common neuropathy allele; "
            "severe mutations → metatropic dysplasia (lethal) — genotype-phenotype CRITICAL"
        ),
        "treatment": (
            "ORTHOPAEDIC: spine surveillance (scoliosis, kyphoscoliosis from skeletal dysplasia); "
            "RHEUMATOLOGY/PAEDIATRIC ORTHOPAEDICS: joint surveillance; "
            "LARYNGOSCOPY: vocal fold mobility assessment; "
            "ENT 6-monthly: airway safety monitoring; "
            "TRACHEOSTOMY if bilateral vocal fold palsy; "
            "RESPIRATORY: spirometry 6-monthly; NIV if FVC < 70%; "
            "AUDIOLOGY: pure tone audiogram annually; hearing aids; "
            "AFOs for foot drop; "
            "PHYSIOTHERAPY + SEATING; "
            "GENETIC COUNSELLING: 50% offspring risk; "
            "GENOTYPE-PHENOTYPE: confirm variant class before prognosis discussion"
        ),
        "key_biomarker": (
            "Skeletal survey: brachyolmia/SED changes; "
            "Laryngoscopy: vocal fold mobility; "
            "Audiogram: SNHL; "
            "NCS: motor axonal lower limb; "
            "Muscle MRI: lower limb selective denervation; "
            "TRPV4 sequencing + genotype-phenotype correlation"
        ),
        "critical_flags": [
            "SKELETAL-DYSPLASIA-PLUS-NEUROPATHY-PATHOGNOMONIC",
            "VOCAL-CORD-PARALYSIS-LARYNGOSCOPY-MANDATORY",
            "SENSORINEURAL-HEARING-LOSS",
            "pArg269His-NEUROPATHY-VARIANT",
            "SEVERE-VARIANTS-METATROPIC-DYSPLASIA-LETHAL",
            "GENOTYPE-PHENOTYPE-CRITICAL",
            "RESPIRATORY-NIV-IF-FVC-BELOW-70pct",
            "CASCADE-TESTING-50pct",
        ],
    },
]


def _build_cohort(gene_data: dict, seed: int, n: int = 40) -> list:
    """Build a synthetic 40-patient cohort for one gene."""
    rng = random.Random(seed)
    gene = gene_data["gene"]
    cohort = []
    for pid in range(n):
        if gene == "HSPB1":
            onset = rng.randint(20, 45)
            foot_drop = True
            hand_wasting = rng.random() < 0.55
            sensory_loss = False
            upper_limb_predominant = False
            vocal_fold_palsy = False
            respiratory_failure = False
            congenital_contractures = False
            intellectual_disability = False
            skeletal_dysplasia = False
            hearing_loss = False
            respiratory_fvc_low = rng.random() < 0.05
            wheelchair = rng.random() < 0.15
            hip_dislocation = False
            diaphragm_palsy = False
        elif gene == "HSPB8":
            onset = rng.randint(10, 25)
            foot_drop = rng.random() < 0.85
            hand_wasting = rng.random() < 0.45
            sensory_loss = False
            upper_limb_predominant = False
            vocal_fold_palsy = False
            respiratory_failure = False
            congenital_contractures = False
            intellectual_disability = False
            skeletal_dysplasia = False
            hearing_loss = False
            respiratory_fvc_low = rng.random() < 0.05
            wheelchair = rng.random() < 0.10
            hip_dislocation = False
            diaphragm_palsy = False
        elif gene == "GARS1":
            onset = rng.randint(10, 22)
            foot_drop = rng.random() < 0.45
            hand_wasting = True
            sensory_loss = rng.random() < 0.30
            upper_limb_predominant = True
            vocal_fold_palsy = False
            respiratory_failure = rng.random() < 0.25
            congenital_contractures = False
            intellectual_disability = False
            skeletal_dysplasia = False
            hearing_loss = False
            respiratory_fvc_low = rng.random() < 0.25
            wheelchair = rng.random() < 0.20
            hip_dislocation = False
            diaphragm_palsy = rng.random() < 0.20
        elif gene == "DCTN1":
            onset = rng.randint(20, 38)
            foot_drop = rng.random() < 0.75
            hand_wasting = rng.random() < 0.50
            sensory_loss = False
            upper_limb_predominant = False
            vocal_fold_palsy = True
            respiratory_failure = rng.random() < 0.40
            congenital_contractures = False
            intellectual_disability = False
            skeletal_dysplasia = False
            hearing_loss = False
            respiratory_fvc_low = rng.random() < 0.35
            wheelchair = rng.random() < 0.25
            hip_dislocation = False
            diaphragm_palsy = False
        elif gene == "BICD2":
            onset = 0  # congenital
            foot_drop = rng.random() < 0.70
            hand_wasting = False
            sensory_loss = False
            upper_limb_predominant = False
            vocal_fold_palsy = False
            respiratory_failure = False
            congenital_contractures = True
            intellectual_disability = False
            skeletal_dysplasia = False
            hearing_loss = False
            respiratory_fvc_low = rng.random() < 0.05
            wheelchair = rng.random() < 0.40
            hip_dislocation = True
            diaphragm_palsy = False
        elif gene == "DYNC1H1":
            onset = rng.randint(1, 10)
            foot_drop = rng.random() < 0.60
            hand_wasting = False
            sensory_loss = False
            upper_limb_predominant = False
            vocal_fold_palsy = False
            respiratory_failure = False
            congenital_contractures = False
            intellectual_disability = rng.random() < 0.30
            skeletal_dysplasia = False
            hearing_loss = False
            respiratory_fvc_low = rng.random() < 0.10
            wheelchair = rng.random() < 0.35
            hip_dislocation = rng.random() < 0.25
            diaphragm_palsy = False
        elif gene == "IGHMBP2":
            onset = rng.randint(0, 1)  # 0-1 year (infantile)
            foot_drop = rng.random() < 0.80
            hand_wasting = rng.random() < 0.70
            sensory_loss = False
            upper_limb_predominant = False
            vocal_fold_palsy = False
            respiratory_failure = True
            congenital_contractures = False
            intellectual_disability = False
            skeletal_dysplasia = False
            hearing_loss = False
            respiratory_fvc_low = True
            wheelchair = True
            hip_dislocation = False
            diaphragm_palsy = True
        else:  # TRPV4
            onset = rng.randint(1, 10)
            foot_drop = rng.random() < 0.70
            hand_wasting = rng.random() < 0.30
            sensory_loss = rng.random() < 0.35
            upper_limb_predominant = False
            vocal_fold_palsy = rng.random() < 0.60
            respiratory_failure = rng.random() < 0.20
            congenital_contractures = False
            intellectual_disability = False
            skeletal_dysplasia = True
            hearing_loss = rng.random() < 0.55
            respiratory_fvc_low = rng.random() < 0.20
            wheelchair = rng.random() < 0.30
            hip_dislocation = False
            diaphragm_palsy = False

        cohort.append({
            "patient_id": f"{gene}-{pid+1:03d}",
            "onset_age": onset,
            "foot_drop": foot_drop,
            "hand_wasting": hand_wasting,
            "sensory_loss": sensory_loss,
            "upper_limb_predominant": upper_limb_predominant,
            "vocal_fold_palsy": vocal_fold_palsy,
            "respiratory_failure": respiratory_failure,
            "congenital_contractures": congenital_contractures,
            "intellectual_disability": intellectual_disability,
            "skeletal_dysplasia": skeletal_dysplasia,
            "hearing_loss": hearing_loss,
            "respiratory_fvc_low": respiratory_fvc_low,
            "wheelchair": wheelchair,
            "hip_dislocation": hip_dislocation,
            "diaphragm_palsy": diaphragm_palsy,
        })
    return cohort


_ALL_COHORTS = {
    g["gene"]: _build_cohort(g, SEED_BASE + i)
    for i, g in enumerate(DHSN_GENES)
}


def overview() -> dict:
    """Aggregate overview across all 8 dHMN genes (320 patients, seeds 2134-2141)."""
    all_patients = [p for cohort in _ALL_COHORTS.values() for p in cohort]
    n = len(all_patients)

    hspb1_cohort   = _ALL_COHORTS["HSPB1"]
    hspb8_cohort   = _ALL_COHORTS["HSPB8"]
    gars1_cohort   = _ALL_COHORTS["GARS1"]
    dctn1_cohort   = _ALL_COHORTS["DCTN1"]
    bicd2_cohort   = _ALL_COHORTS["BICD2"]
    dync1h1_cohort = _ALL_COHORTS["DYNC1H1"]
    ighmbp2_cohort = _ALL_COHORTS["IGHMBP2"]
    trpv4_cohort   = _ALL_COHORTS["TRPV4"]

    return {
        "atlas": "Hereditary dHMN Atlas — Complete 8-Gene Distal Hereditary Motor Neuropathy Reference",
        "genes": [g["gene"] for g in DHSN_GENES],
        "total_patients": n,
        "seeds": "2134-2141",
        "seed_base": SEED_BASE,
        # HSPB1
        "hspb1_foot_drop_patients": sum(1 for p in hspb1_cohort if p["foot_drop"]),
        "hspb1_hand_wasting_patients": sum(1 for p in hspb1_cohort if p["hand_wasting"]),
        # HSPB8
        "hspb8_foot_drop_patients": sum(1 for p in hspb8_cohort if p["foot_drop"]),
        # GARS1
        "gars1_upper_limb_predominant_patients": sum(1 for p in gars1_cohort if p["upper_limb_predominant"]),
        "gars1_respiratory_failure_patients": sum(1 for p in gars1_cohort if p["respiratory_failure"]),
        "gars1_diaphragm_palsy_patients": sum(1 for p in gars1_cohort if p["diaphragm_palsy"]),
        # DCTN1
        "dctn1_vocal_fold_palsy_patients": sum(1 for p in dctn1_cohort if p["vocal_fold_palsy"]),
        "dctn1_respiratory_failure_patients": sum(1 for p in dctn1_cohort if p["respiratory_failure"]),
        # BICD2
        "bicd2_congenital_contractures_patients": sum(1 for p in bicd2_cohort if p["congenital_contractures"]),
        "bicd2_hip_dislocation_patients": sum(1 for p in bicd2_cohort if p["hip_dislocation"]),
        # DYNC1H1
        "dync1h1_intellectual_disability_patients": sum(1 for p in dync1h1_cohort if p["intellectual_disability"]),
        "dync1h1_hip_dislocation_patients": sum(1 for p in dync1h1_cohort if p["hip_dislocation"]),
        # IGHMBP2
        "ighmbp2_respiratory_failure_patients": sum(1 for p in ighmbp2_cohort if p["respiratory_failure"]),
        "ighmbp2_diaphragm_palsy_patients": sum(1 for p in ighmbp2_cohort if p["diaphragm_palsy"]),
        "ighmbp2_wheelchair_patients": sum(1 for p in ighmbp2_cohort if p["wheelchair"]),
        # TRPV4
        "trpv4_skeletal_dysplasia_patients": sum(1 for p in trpv4_cohort if p["skeletal_dysplasia"]),
        "trpv4_vocal_fold_palsy_patients": sum(1 for p in trpv4_cohort if p["vocal_fold_palsy"]),
        "trpv4_hearing_loss_patients": sum(1 for p in trpv4_cohort if p["hearing_loss"]),
        # Cross-atlas
        "all_foot_drop_patients": sum(1 for p in all_patients if p["foot_drop"]),
        "all_respiratory_failure_patients": sum(1 for p in all_patients if p["respiratory_failure"]),
        "all_vocal_fold_palsy_patients": sum(1 for p in all_patients if p["vocal_fold_palsy"]),
        "all_wheelchair_patients": sum(1 for p in all_patients if p["wheelchair"]),
        "all_congenital_patients": sum(1 for p in all_patients if p["onset_age"] == 0),
    }


def breakdown() -> dict:
    """Per-gene clinical breakdown for all 8 dHMN genes."""
    result = {}
    for g in DHSN_GENES:
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
            "foot_drop_pct": round(100 * sum(1 for p in cohort if p["foot_drop"]) / n, 1),
            "hand_wasting_pct": round(100 * sum(1 for p in cohort if p["hand_wasting"]) / n, 1),
            "sensory_loss_pct": round(100 * sum(1 for p in cohort if p["sensory_loss"]) / n, 1),
            "upper_limb_predominant_pct": round(100 * sum(1 for p in cohort if p["upper_limb_predominant"]) / n, 1),
            "vocal_fold_palsy_pct": round(100 * sum(1 for p in cohort if p["vocal_fold_palsy"]) / n, 1),
            "respiratory_failure_pct": round(100 * sum(1 for p in cohort if p["respiratory_failure"]) / n, 1),
            "congenital_contractures_pct": round(100 * sum(1 for p in cohort if p["congenital_contractures"]) / n, 1),
            "intellectual_disability_pct": round(100 * sum(1 for p in cohort if p["intellectual_disability"]) / n, 1),
            "skeletal_dysplasia_pct": round(100 * sum(1 for p in cohort if p["skeletal_dysplasia"]) / n, 1),
            "hearing_loss_pct": round(100 * sum(1 for p in cohort if p["hearing_loss"]) / n, 1),
            "wheelchair_pct": round(100 * sum(1 for p in cohort if p["wheelchair"]) / n, 1),
            "hip_dislocation_pct": round(100 * sum(1 for p in cohort if p["hip_dislocation"]) / n, 1),
            "diaphragm_palsy_pct": round(100 * sum(1 for p in cohort if p["diaphragm_palsy"]) / n, 1),
        }
    return result


def definitions() -> dict:
    """Gene definitions, glossary, and surveillance protocols."""
    return {
        "genes": {
            g["gene"]: g["protein"]
            for g in DHSN_GENES
        },
        "glossary": {
            "dHMN (Distal Hereditary Motor Neuropathy)": (
                "A group of hereditary neuropathies primarily affecting distal motor neurons. "
                "Key distinction from CMT/HMSN: motor >> sensory (CMT = motor + sensory equally). "
                "Key distinction from SMA: distal limb distribution (SMA = proximal). "
                "Classified by gene; over 30 genes identified. "
                "Phenotypic spectrum ranges from pure distal weakness to complex syndromes "
                "(respiratory failure, vocal fold palsy, cognitive impairment, skeletal dysplasia). "
                "NCS: motor axonal; SNAPs preserved or near-normal distinguishes from CMT."
            ),
            "Small Heat-Shock Proteins (HSPB1/HSP27 and HSPB8/HSP22)": (
                "HSPB1 and HSPB8 are small heat-shock proteins that act as molecular chaperones. "
                "Both cause dHMN2 (AD) and CMT2 (AD, with sensory). "
                "Pathogenic mutations cause misfolding and cytoplasmic aggregate formation in motor neurons. "
                "HSPB1 (640aa, 7q11.23) and HSPB8 (196aa, 12q24.23) interact — co-aggregate. "
                "Adult and juvenile onset respectively; slow progression; excellent long-term ambulation. "
                "No disease-modifying therapy; supportive AFOs + physiotherapy."
            ),
            "Aminoacyl-tRNA Synthetase Genes (GARS1 and ARS family)": (
                "GARS1 (glycyl-tRNA synthetase) is one of >15 aminoacyl-tRNA synthetase genes causing dHMN. "
                "ALL ARS genes have the same dominant-negative mechanism: "
                "mutant enzyme inhibits neuronal aminoacyl-tRNA charging → axon degeneration. "
                "UPPER LIMB PREDOMINANT phenotype distinguishes GARS1/ARS dHMN from most others. "
                "Key: test full ARS panel when dHMN phenotype is upper-limb-first. "
                "Phrenic nerve involvement (20-30%) means respiratory monitoring is mandatory."
            ),
            "Retrograde Axonal Transport Genes (DCTN1/BICD2/DYNC1H1)": (
                "Three genes in this atlas encode components of the dynein/dynactin retrograde transport machinery: "
                "DCTN1 (p150glued, 2p13.1): activator of dynein; dHMN7B with vocal fold palsy. "
                "BICD2 (Bicaudal-D2, 9q22.31): cargo adaptor; SMALED1 with congenital lower-limb. "
                "DYNC1H1 (Dynein heavy chain, 14q32.31): motor; SMALED1 + intellectual disability. "
                "All three share de novo mutations, retrograde failure, lower motor neuron loss. "
                "Clinical separation: DCTN1 = vocal fold; BICD2 = congenital lower limb + normal IQ; "
                "DYNC1H1 = lower limb + intellectual disability + pachygyria."
            ),
            "IGHMBP2 (SMARD1) — NOT SMA1": (
                "SMARD1 (Spinal Muscular Atrophy with Respiratory Distress) caused by IGHMBP2 mutations "
                "is clinically similar to SMA1 (SMN1) but is a completely different gene and mechanism. "
                "CRITICAL DISTINCTION: nusinersen/risdiplam/zolgensma target SMN1/SMN2 pathway — "
                "they do NOT treat IGHMBP2-SMARD1. Giving SMN therapy to SMARD1 = no benefit. "
                "PATHOGNOMONIC: respiratory failure + diaphragm paralysis before limb weakness. "
                "Test IGHMBP2 in infantile onset SMA with diaphragmatic palsy. "
                "Treatment: ventilatory support (NIV → tracheostomy) + physiotherapy + multidisciplinary."
            ),
            "TRPV4 Spectrum (SMALED2/CMT2C/Skeletal)": (
                "TRPV4 mutations cause a spectrum of conditions depending on variant severity: "
                "Mild/neuropathy alleles (pArg269His): SMALED2 + CMT2C (motor + sensory + vocal fold). "
                "Intermediate: brachyolmia + neuropathy. "
                "Severe: metatropic dysplasia / SED (lethal skeletal dysplasia). "
                "PATHOGNOMONIC combination: skeletal dysplasia + distal motor neuropathy in same patient. "
                "Genotype-phenotype correlation is essential before prognostic counselling. "
                "Features shared with DCTN1: vocal fold palsy (both require laryngoscopy)."
            ),
            "Laryngoscopy in dHMN — When and Why": (
                "Two genes in this atlas cause vocal fold paralysis: DCTN1 (dHMN7B) and TRPV4 (SMALED2). "
                "DCTN1: bilateral vocal fold immobility → hoarse voice → stridor → respiratory failure. "
                "TRPV4: vocal fold palsy in up to 60%; may precede or follow limb weakness. "
                "ACTION: laryngoscopy at diagnosis for DCTN1 and TRPV4; repeat 6-monthly. "
                "If bilateral vocal fold palsy: immediate ENT + anaesthetics review; "
                "secure airway before any sedation/anaesthesia; "
                "tracheostomy if airway compromised at rest."
            ),
            "Cascade Testing and De Novo Burden in dHMN": (
                "Three genes in this atlas have significant de novo mutation rates: "
                "BICD2: ~40% de novo; DYNC1H1: ~60% de novo. "
                "Consequence: negative family history does NOT exclude a hereditary cause. "
                "BICD2/DYNC1H1: trio whole-exome sequencing (proband + parents) is the preferred test. "
                "AD genes (HSPB1, HSPB8, GARS1, DCTN1, TRPV4): 50% offspring risk → cascade testing. "
                "AR gene (IGHMBP2): 25% sibling risk → carrier testing parents; prenatal available. "
                "Genetic counselling is mandatory for all dHMN families."
            ),
        },
        "surveillance_protocols": {
            "HSPB1 (dHMN2B)": (
                "HSPB1 sequencing (NGS panel); "
                "NCS annually: CMAP amplitudes + SNAPs as progression markers; "
                "EMG: distal motor denervation; "
                "PODIATRY: AFO fitting at foot drop; 6-monthly review; "
                "PHYSIOTHERAPY: gait training + falls prevention annually; "
                "OT: hand function 2-yearly; "
                "RESPIRATORY: spirometry 2-yearly (phrenic rare); "
                "FALLS: home assessment + grab rails; "
                "GENETIC COUNSELLING: 50% offspring risk; "
                "CASCADE TESTING: first-degree relatives; presymptomatic testing ≥18 yr"
            ),
            "HSPB8 (dHMN2A)": (
                "HSPB8 Lys141 hotspot Sanger or panel NGS; "
                "NCS annually: CMAP + SNAP progression; "
                "EMG: distal motor denervation; "
                "PHYSIOTHERAPY: gait + upper limb strengthening; "
                "SCHOOL SUPPORT: fine motor difficulties early; "
                "AFO: foot drop splints; "
                "RESPIRATORY: spirometry 2-yearly; "
                "GENETIC COUNSELLING: 50% offspring risk; "
                "PROGNOSIS COUNSELLING: slow progression, ambulant decades"
            ),
            "GARS1 (dHMN5A)": (
                "GARS1 sequencing + ARS panel (AARS1, YARS1, MARS1); "
                "RESPIRATORY PRIORITY: spirometry 6-monthly; "
                "if FVC < 70%: sleep study + NIV assessment; "
                "Phrenic nerve NCS: CMAP amplitude monitoring; "
                "NCS + EMG: upper > lower limb; "
                "HAND THERAPY: OT assessment 6-monthly; "
                "AFO if foot drop develops; "
                "GENETIC COUNSELLING: 50% offspring risk; "
                "UPPER-LIMB-FIRST PHENOTYPE: test full ARS panel"
            ),
            "DCTN1 (dHMN7B)": (
                "DCTN1 pGly59Ser sequencing; "
                "LARYNGOSCOPY MANDATORY at diagnosis; "
                "ENT 6-monthly: vocal fold mobility; "
                "RESPIRATORY: spirometry 6-monthly; "
                "if FVC < 70% or bilateral palsy: NIV or tracheostomy; "
                "VIDEO FLUOROSCOPY: swallowing annually (dysphagia risk); "
                "NCS + EMG annually: motor axonal progression; "
                "AFO for foot drop; "
                "ANAESTHETIC ALERT: difficult airway (vocal fold palsy); "
                "GENETIC COUNSELLING: 50% offspring risk"
            ),
            "BICD2 (SMALED1)": (
                "BICD2 sequencing + trio WES (de novo ~40%); "
                "ORTHOPAEDIC: hip X-ray at diagnosis; 6-monthly in infancy; "
                "Pavlik harness or surgical hip reduction if dislocated; "
                "Serial casting for lower limb contractures; "
                "PHYSIOTHERAPY: passive + active lower limb mobilisation; "
                "ORTHOTICS: AFO + KAFO; "
                "DEVELOPMENTAL REVIEW: confirm normal IQ; mainstream school; "
                "SEATING: power wheelchair assessment if non-ambulant; "
                "RESPIRATORY: spirometry 2-yearly; "
                "GENETIC COUNSELLING: trio WES; 50% offspring from AD parent"
            ),
            "DYNC1H1 (SMALED1 + ID)": (
                "DYNC1H1 sequencing + trio WES (de novo ~60%); "
                "MRI BRAIN: pachygyria/lissencephaly workup; "
                "COGNITIVE ASSESSMENT: neuropsychology at diagnosis; "
                "EDUCATIONAL: EHCP; educational psychologist input; "
                "EPILEPSY: EEG if seizures (pachygyria → 30% seizures); "
                "AED if epilepsy; "
                "HIP X-RAY: 6-monthly (subluxation risk); "
                "PHYSIOTHERAPY: lower limb strengthening; "
                "SEATING: power wheelchair; "
                "RESPIRATORY: spirometry 2-yearly; "
                "GENETIC COUNSELLING: trio WES; 50% offspring from AD parent"
            ),
            "IGHMBP2 (SMARD1)": (
                "IGHMBP2 sequencing; "
                "NIV/TRACHEOSTOMY MANDATORY at diagnosis; "
                "RESPIRATORY: continuous monitoring; RespTech team; "
                "Diaphragm function: phrenic NCS + ultrasound; "
                "PHRENIC NERVE PACING: referral if candidate; "
                "GASTROSTOMY (PEG): 4-6 months; "
                "CHEST PHYSIOTHERAPY: airway clearance twice daily; cough assist; "
                "PHYSIOTHERAPY: limb mobilisation; contracture prevention; "
                "NOT SMN THERAPY: IGHMBP2 ≠ SMN1; nusinersen/risdiplam NOT indicated; "
                "MULTIDISCIPLINARY: respiratory + neurology + gastro + genetics + OT + PT; "
                "GENETIC COUNSELLING: 25% sibling risk; carrier testing parents"
            ),
            "TRPV4 (SMALED2)": (
                "TRPV4 sequencing + genotype-phenotype correlation; "
                "SKELETAL SURVEY: spine + long bones at diagnosis; "
                "RHEUMATOLOGY/ORTHOPAEDICS: joint surveillance 6-monthly; "
                "LARYNGOSCOPY MANDATORY at diagnosis: vocal fold mobility; "
                "ENT 6-monthly: airway monitoring; "
                "RESPIRATORY: spirometry 6-monthly; "
                "AUDIOLOGY: pure tone audiogram annually; hearing aids early; "
                "NCS + EMG 6-monthly: progression; "
                "AFO for foot drop; "
                "ANAESTHETIC ALERT: vocal fold palsy + skeletal → difficult airway; "
                "GENETIC COUNSELLING: 50% offspring risk; genotype-phenotype before prognosis"
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
    print(f"HSPB1 foot drop: {ov['hspb1_foot_drop_patients']}")
    print(f"GARS1 upper limb predominant: {ov['gars1_upper_limb_predominant_patients']}")
    print(f"GARS1 respiratory failure: {ov['gars1_respiratory_failure_patients']}")
    print(f"DCTN1 vocal fold palsy: {ov['dctn1_vocal_fold_palsy_patients']}")
    print(f"BICD2 congenital contractures: {ov['bicd2_congenital_contractures_patients']}")
    print(f"BICD2 hip dislocation: {ov['bicd2_hip_dislocation_patients']}")
    print(f"DYNC1H1 intellectual disability: {ov['dync1h1_intellectual_disability_patients']}")
    print(f"IGHMBP2 respiratory failure: {ov['ighmbp2_respiratory_failure_patients']}")
    print(f"IGHMBP2 diaphragm palsy: {ov['ighmbp2_diaphragm_palsy_patients']}")
    print(f"TRPV4 skeletal dysplasia: {ov['trpv4_skeletal_dysplasia_patients']}")
    print(f"All vocal fold palsy: {ov['all_vocal_fold_palsy_patients']}")
    print(f"All respiratory failure: {ov['all_respiratory_failure_patients']}")
    print(f"All foot drop: {ov['all_foot_drop_patients']}")
