#!/usr/bin/env python3
"""Hereditary-HSAN-Atlas — Complete 8-Gene Hereditary Sensory and Autonomic Neuropathy Atlas
(HSAN1A/SPTLC1 · HSAN3/FD/IKBKAP · HSAN4/CIPA/NTRK1 · HSAN2D-IE/SCN9A ·
 HSAN2B/RETREG1 · HSAN2A/WNK1 · HSAN1E/DNMT1 · HSAN1D/ATL1).

SPTLC1  (Serine Palmitoyltransferase Long-Chain Base Subunit 1; 479 aa; 9q22.31; AD;
          HSAN1A — SHOOTING/LANCINATING PAINS PATHOGNOMONIC — onset 2nd-3rd decade;
          Loss of pain/temperature sensation → foot ulcers + dorsal scars;
          Deoxy-sphingolipid metabolites (dSL = atypical SPTLC1 uses L-alanine not serine);
          L-SERINE SUPPLEMENTATION 400 mg/kg/day reduces dSL → slows neuropathy — TREATABLE;
          seed SEED_BASE+0).
IKBKAP  (Inhibitor of κB kinase complex-associated protein / Elongator complex protein 1;
          1332 aa; 9q31.3; AR;
          HSAN3 / Riley-Day Syndrome / FAMILIAL DYSAUTONOMIA (FD);
          ABSENT FUNGIFORM PAPILLAE (tongue inspection first → smooth dorsum) PATHOGNOMONIC;
          AUTONOMIC CRISES (episodic vomiting + hyperhidrosis + labile BP) PATHOGNOMONIC;
          Ashkenazi Jewish founder: IVS20+6T>C splicing mutation 99.5% of FD alleles;
          Ataluren (PTC124) read-through therapy compassionate use;
          seed SEED_BASE+1).
NTRK1   (Neurotrophic Tyrosine Receptor Kinase 1 / TRKA; 796 aa; 1q23.1; AR;
          HSAN4 / CIPA — Congenital Insensitivity to Pain with Anhidrosis;
          CONGENITAL PAIN INSENSITIVITY + ANHIDROSIS → SELF-MUTILATION PATHOGNOMONIC;
          HYPERTHERMIA DEATHS — anhidrotic heat stroke; fever protocol MANDATORY;
          AVOID AMPUTATION — fractures heal; self-biting of tongue/lip/fingers;
          No TRKA = no NGF signalling → nociceptors/autonomic neurons fail to survive;
          seed SEED_BASE+2).
SCN9A   (Sodium Channel protein Nav1.7 / Alpha subunit; 1988 aa; 2q24.3; AR/AD;
          PARADOX GENE: LOF (AR) = HSAN2D / Congenital Insensitivity to Pain (CIP);
                         GOF (AD) = INHERITED ERYTHROMELALGIA (IEM) — burning feet + redness PATHOGNOMONIC;
          LOF: complete pain insensitivity from birth; anosmia (50%); otherwise normal intellect;
          GOF (Inherited Erythromelalgia): bilateral red/hot feet + BURNING PAIN — warmth triggers;
          CARBAMAZEPINE may reduce pain in GOF Nav1.7 mutations (Na-channel blocker);
          seed SEED_BASE+3).
RETREG1 (Reticulophagy Regulator 1 / FAM134B; 460 aa; 5p15.1; AR;
          HSAN2B — congenital onset, severe pan-sensory loss (pain/temperature/touch);
          CORNEAL ANAESTHESIA → NEUROTROPHIC CORNEAL ULCERATION PATHOGNOMONIC;
          Mutilating arthropathy of hands and feet; skeletal deformity from unrecognised fractures;
          ER-phagy (selective autophagy of ER) regulator — loss → ER failure in sensory neurons;
          Turkish/Sudanese/Israeli Arab founders; consanguineous families;
          seed SEED_BASE+4).
WNK1    (With-no-lysine kinase 1 / HSN2 splice isoform; 2382 aa; 12p13.33; AR (HSN2 exon);
          HSAN2A — congenital onset <5 yr; pan-sensory loss (all modalities); mutilating neuropathy;
          SEVERE MUTILATION OF EXTREMITIES — toe/finger loss from ignored trauma PATHOGNOMONIC;
          Sudanese/Nova Scotian (Acadian)/French-Canadian founders;
          Mutations ONLY in HSN2 neuronal-specific exon (standard WES may miss if HSN2 not annotated);
          seed SEED_BASE+5).
DNMT1   (DNA (Cytosine-5)-Methyltransferase 1; 1616 aa; 19p13.2; AD;
          HSAN1E / ADCA-DN (Autosomal Dominant Cerebellar Ataxia-Deafness-Narcolepsy);
          NARCOLEPSY-CATAPLEXY + SENSORINEURAL HEARING LOSS + DEMENTIA + SENSORY NEUROPATHY
          — 4-feature syndrome PATHOGNOMONIC — onset 20-40 yr;
          REMD (Replication foci targeting sequence - Methyltransferase - DNMT1 Orc1) domain mutations;
          No effective disease-modifying therapy; modafinil for narcolepsy;
          seed SEED_BASE+6).
ATL1    (Atlastin-1 GTPase; 558 aa; 14q22.1; AD;
          HSAN1D — late-onset (40-60 yr);
          CHRONIC DRY/HOARSE COUGH + GASTRO-OESOPHAGEAL REFLUX PATHOGNOMONIC;
          Sensorineural hearing loss; sensory neuropathy (distal lower > upper limb);
          Atlastin-1 = ER tubule fusion GTPase; dominant variants impair ER network formation;
          No disease-modifying therapy; PPI for GERD; hearing aids for SNHL;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2126-2133).
"""

import random

SEED_BASE = 2126

HSAN_GENES = [
    # -- SPTLC1 — HSAN1A --------------------------------------------------------
    {
        "gene": "SPTLC1",
        "alt_name": (
            "SPTLC1 (SPTLC1-479aa-9q22.31 / AD — HSAN1A-Hereditary-Sensory-Autonomic-Neuropathy-Type-1A — "
            "Shooting-Lancinating-Pains-PATHOGNOMONIC — Foot-Ulcers-Dorsal-Scars — "
            "Deoxy-Sphingolipid-Metabolites-Alanine-Not-Serine — "
            "L-Serine-Supplementation-400mg-kg-day-TREATABLE)"
        ),
        "protein": (
            "SPTLC1 -- 9q22.31 AD -- SPTLC1-479aa -- "
            "Serine-Palmitoyltransferase-Long-Chain-Base-Subunit-1-Enzyme-De-Novo-Sphingolipid-Synthesis -- "
            "HSAN1A-Hereditary-Sensory-Autonomic-Neuropathy-Type-1A-OMIM-256800 -- "
            "Gain-of-Function-Dominant-Variants-Allow-L-Alanine-Substrate-Instead-of-L-Serine -- "
            "Deoxy-Sphingolipid-Metabolites-1-Deoxy-Sphinganine-Accumulation-Toxic-to-Sensory-Neurons -- "
            "Shooting-Lancinating-Pains-Onset-2nd-3rd-Decade-PATHOGNOMONIC -- "
            "Progressive-Sensory-Loss-Pain-Temperature-First-Then-Touch-Proprioception -- "
            "Foot-Dorsal-Scars-Plantar-Ulcers-Osteomyelitis-from-Unrecognised-Trauma -- "
            "L-Serine-Supplementation-400mg-kg-day-Reduces-dSL-Plasma-Levels-Slows-Neuropathy-TREATABLE -- "
            "pCys133Trp-pVal144Asp-Most-Common-Variants-Most-Penetrant"
        ),
        "locus": "9q22.31",
        "protein_size": "479 aa",
        "inheritance": (
            "AD (autosomal dominant); gain-of-function; "
            "missense mutations enabling L-alanine substrate incorporation; "
            "most variants in transmembrane domain of SPTLC1; "
            "reduced penetrance ~75%; late-onset variants also reported; "
            "genetic testing: NGS panel or WES detects coding variants; "
            "key alleles: pCys133Trp (most common), pVal144Asp"
        ),
        "age_of_onset": "15-40 years (2nd-4th decade); rarely >50 yr",
        "pathognomonic": (
            "SHOOTING/LANCINATING PAINS PATHOGNOMONIC — positive sensory symptoms early in dominant HSAN (rare in other HSANs); "
            "foot ulcers with dorsal scars (not plantar as in diabetic neuropathy — DDx key); "
            "progressive loss of pain/temperature → touch/proprioception; "
            "motor involvement mild/late; sensorimotor axonal on NCS; "
            "deoxy-sphingolipid plasma assay confirms: 1-deoxy-sphinganine + 1-deoxy-sphingosine elevated"
        ),
        "treatment": (
            "L-SERINE SUPPLEMENTATION 400 mg/kg/day (up to 3-5 g/day) — reduces deoxy-SL plasma levels; "
            "phase 2 trial showed stabilisation of neurological deficits (SNAPS amplitude); "
            "WOUND CARE: podiatry review quarterly; orthopaedic shoes + offloading insoles; "
            "osteomyelitis: antibiotics + bone debridement if infected ulcer; "
            "NEUROPATHIC PAIN: pregabalin/gabapentin for positive sensory symptoms; "
            "PHYSIOTHERAPY: gait training + balance; "
            "GENETIC COUNSELLING: 50% offspring risk; presymptomatic testing at 18 yr; "
            "plasma deoxy-SL monitoring on treatment"
        ),
        "key_biomarker": (
            "Plasma deoxy-sphingolipids elevated (1-DSa + 1-DS); "
            "NCS: sensorimotor axonal neuropathy (SNAP reduced/absent); "
            "Skin biopsy: reduced intraepidermal nerve fibre density (IENFD); "
            "EMG: normal strength initially"
        ),
        "critical_flags": [
            "SPTLC1-SHOOTING-LANCINATING-PAINS-PATHOGNOMONIC",
            "SPTLC1-FOOT-ULCERS-DORSAL-SCARS",
            "SPTLC1-L-SERINE-SUPPLEMENTATION-TREATABLE",
            "SPTLC1-DEOXY-SPHINGOLIPID-METABOLITES",
            "SPTLC1-GAIN-OF-FUNCTION-ALANINE-SUBSTRATE",
            "SPTLC1-PCDOS133TRP-MOST-COMMON-ALLELE",
            "HSAN1A-AD-ADULT-ONSET",
        ],
    },

    # -- IKBKAP / ELP1 — HSAN3 / Familial Dysautonomia -----------------------
    {
        "gene": "IKBKAP",
        "alt_name": (
            "IKBKAP/ELP1 (IKBKAP-1332aa-9q31.3 / AR — HSAN3-Familial-Dysautonomia-Riley-Day-Syndrome — "
            "Absent-Fungiform-Papillae-PATHOGNOMONIC-Tongue-Inspection-First — "
            "Autonomic-Crises-Episodic-Vomiting-Hyperhidrosis-Labile-BP-PATHOGNOMONIC — "
            "Ashkenazi-Jewish-IVS20+6T>C-99.5pct-Founder)"
        ),
        "protein": (
            "IKBKAP/ELP1 -- 9q31.3 AR -- IKBKAP-1332aa -- "
            "Elongator-Complex-Protein-1-tRNA-Modification-Wobble-Uridine-U34 -- "
            "HSAN3-Familial-Dysautonomia-Riley-Day-OMIM-223900-Ashkenazi-Jewish-1:3700-Births -- "
            "Splicing-Mutation-IVS20+6T>C-99.5pct-FD-Alleles-Tissue-Specific-Skipping-Exon-20 -- "
            "Neural-Crest-Cell-Migration-Defect-Sensory-Autonomic-Ganglia-Underpopulated -- "
            "Absent-Fungiform-Papillae-Smooth-Tongue-Dorsum-PATHOGNOMONIC-First-Look-Sign -- "
            "Autonomic-Crises-Episodic-Vomiting-Diaphoresis-Hypertension-Tachycardia-PATHOGNOMONIC -- "
            "No-Overflow-Tears-Alacrima-Anhidrosis-Absent-Deep-Tendon-Reflexes -- "
            "Ataluren-PTC124-Read-Through-Therapy-Compassionate-Use -- "
            "Carrier-Frequency-1:30-Ashkenazi-Carrier-Testing-Mandatory-Ashkenazi-Couples"
        ),
        "locus": "9q31.3",
        "protein_size": "1332 aa",
        "inheritance": (
            "AR (autosomal recessive); "
            "founder mutation IVS20+6T>C in >99.5% of FD alleles — Ashkenazi Jewish; "
            "splicing mutation causes tissue-specific skipping of exon 20 (more in nervous system); "
            "carrier frequency 1/30 Ashkenazi; incidence 1/3700 Ashkenazi births; "
            "carrier testing panel includes this variant — mandatory for Ashkenazi couples; "
            "genetic diagnosis by targeted mutation analysis (IVS20+6T>C) sufficient in Ashkenazi"
        ),
        "age_of_onset": "Congenital (birth); diagnosed in neonatal period by absent fungiform papillae + alacrima",
        "pathognomonic": (
            "ABSENT FUNGIFORM PAPILLAE — inspect tongue dorsum; smooth surface (no pink bumps) = FD; "
            "INSPECT AT EVERY CLINIC VISIT as first test; fungiform papillae palpable in normal subjects; "
            "AUTONOMIC CRISES PATHOGNOMONIC — episodic vomiting 1-4 hr + profuse sweating + labile BP (hypertension or hypotension); "
            "triggered by stress/excitement/illness; resolved by diazepam IV + ondansetron; "
            "ALACRIMA (absent overflow tears during emotional crying) from birth; "
            "absent deep tendon reflexes; relative indifference to pain; "
            "absent corneal reflexes (Schirmer test dry); "
            "spinal curvature (scoliosis 95%); dysautonomic symptoms: postural hypotension + aspiration"
        ),
        "treatment": (
            "AUTONOMIC CRISIS: IV diazepam 0.1 mg/kg + ondansetron 0.15 mg/kg; NG tube feeding during crisis; "
            "ATALUREN (PTC124) 10/10/20 mg/kg/day — read-through therapy; compassionate use programme; "
            "SCOLIOSIS: early surgery if Cobb >40°; "
            "EYE: methylcellulose drops QID + night ointment; moisture chamber glasses; "
            "AUTONOMIC: fludrocortisone + thigh/abdominal compression for postural hypotension; "
            "DYSPHAGIA: fundoplication for severe reflux + aspiration; G-tube if oral feeds unsafe; "
            "CRISIS DIARY: patient/family tracks triggers + frequency; "
            "PHYSIOTHERAPY: scoliosis prevention, chest physio; "
            "MULTIDISCIPLINARY: neurology + pulmonology + orthopaedics + ophthalmology"
        ),
        "key_biomarker": (
            "Absent fungiform papillae on tongue inspection (clinical PATHOGNOMONIC); "
            "IVS20+6T>C targeted mutation analysis (Ashkenazi); "
            "Schirmer test: <5 mm in 5 min (alacrima); "
            "Histamine intradermal test: no axon flare response; "
            "Postural BP measurement: >30 mmHg systolic drop orthostatic"
        ),
        "critical_flags": [
            "IKBKAP-ABSENT-FUNGIFORM-PAPILLAE-TONGUE-INSPECTION-PATHOGNOMONIC",
            "IKBKAP-AUTONOMIC-CRISES-PATHOGNOMONIC",
            "IKBKAP-ALACRIMA-NO-OVERFLOW-TEARS",
            "IKBKAP-ASHKENAZI-IVS20+6T>C-99.5pct-FOUNDER",
            "IKBKAP-CARRIER-1:30-ASHKENAZI",
            "IKBKAP-ATALUREN-READ-THROUGH-COMPASSIONATE",
            "HSAN3-FAMILIAL-DYSAUTONOMIA-AR-CONGENITAL",
        ],
    },

    # -- NTRK1 — HSAN4 / CIPA -------------------------------------------------
    {
        "gene": "NTRK1",
        "alt_name": (
            "NTRK1 (NTRK1-796aa-1q23.1 / AR — HSAN4-CIPA-Congenital-Insensitivity-to-Pain-with-Anhidrosis — "
            "Pain-Insensitivity-Self-Mutilation-PATHOGNOMONIC — "
            "Hyperthermia-Anhidrotic-Heat-Stroke-FEVER-PROTOCOL-MANDATORY — "
            "AVOID-AMPUTATION-Fractures-Heal)"
        ),
        "protein": (
            "NTRK1/TRKA -- 1q23.1 AR -- NTRK1-796aa -- "
            "Neurotrophic-Tyrosine-Receptor-Kinase-1-TrkA-High-Affinity-NGF-Receptor -- "
            "HSAN4-CIPA-OMIM-256800-Most-Common-HSAN-in-Japan -- "
            "No-TRKA-No-NGF-Signalling-Nociceptors-Sympathetic-Neurons-Fail-Developmental-Apoptosis -- "
            "Congenital-Pain-Insensitivity-Complete-from-Birth-Pathognomonic -- "
            "Anhidrosis-No-Sweating-Anhidrotic-Heat-Stroke-Leading-Cause-Death-Early-Childhood -- "
            "Self-Mutilation-Tongue-Biting-Lip-Biting-Finger-Chewing-from-Teething-Onset -- "
            "Intellectual-Disability-Variable-50pct-Mild-Moderate -- "
            "Avoid-Amputation-Unrecognised-Fractures-Do-Heal-Joint-Destruction-Charcot -- "
            "Fever-Protocol-Cool-Environment-No-Exercise-Heated-Spaces-Mandatory"
        ),
        "locus": "1q23.1",
        "protein_size": "796 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function mutations; "
            "most common in Japan (consanguinity); also Arab/Turkish/Israeli Arab families; "
            "point mutations, deletions, splice site variants; "
            "no TRKA protein → no NGF receptor → nociceptive and sympathetic neurons die during development; "
            "complete penetrance; recurrence risk 25% siblings"
        ),
        "age_of_onset": "Congenital — recognised in infancy by self-mutilation at teething",
        "pathognomonic": (
            "CONGENITAL PAIN INSENSITIVITY — no protective pain sensation from birth; "
            "SELF-MUTILATION PATHOGNOMONIC — biting of tongue, lips, fingertips at teething onset; "
            "marks for gloves, mouth guards, nail care from infancy; "
            "ANHIDROSIS — no sweating in any stimulus; heat intolerance → anhidrotic hyperthermia; "
            "HYPERTHERMIA DEATHS in early childhood — heat stroke without fever illness; "
            "recurrent unexplained fevers from heat retention (not infection); "
            "intellectual disability mild-moderate 50%; "
            "Charcot joints (neuropathic arthropathy) from ignored joint trauma; "
            "absent corneal reflex; absent pain response to venepuncture/injection; "
            "bone: Charcot joints, osteomyelitis from unrecognised fractures"
        ),
        "treatment": (
            "FEVER MANAGEMENT: cool environment always; no vigorous exercise/hot baths; "
            "fan + tepid sponging + antipyretics (paracetamol) for any temp >38°C; "
            "SELF-INJURY PREVENTION: soft helmet, gloves, elbow guards from teething; "
            "custom mouth guard; regular dental/oral review; "
            "WOUND/ORTHOPEDIC: regular podiatry; X-ray any swollen joint (no pain = no complaint); "
            "AVOID AMPUTATION — fractures/osteomyelitis heal with antibiotics + offloading; "
            "INTELLECTUAL DISABILITY: educational support; special school; "
            "OPHTHALMOLOGY: annual slit-lamp (corneal ulcers from absent reflex); "
            "PHYSIOTHERAPY: joint protection; orthotics; "
            "GENETIC COUNSELLING: 25% recurrence; consanguinity counselling"
        ),
        "key_biomarker": (
            "Quantitative Sudomotor Axon Reflex Test (QSART): absent sweat response; "
            "Skin biopsy: absent intraepidermal nerve fibres + absent mast cells; "
            "Histamine flare test: absent axon flare; "
            "NCS: absent SNAPs (sensory); motor NCS normal; "
            "Nerve biopsy: absent unmyelinated fibres + absent small myelinated fibres"
        ),
        "critical_flags": [
            "NTRK1-CIPA-CONGENITAL-PAIN-INSENSITIVITY-PATHOGNOMONIC",
            "NTRK1-SELF-MUTILATION-PATHOGNOMONIC",
            "NTRK1-ANHIDROSIS-HYPERTHERMIA-DEATHS",
            "NTRK1-FEVER-PROTOCOL-MANDATORY",
            "NTRK1-AVOID-AMPUTATION-FRACTURES-HEAL",
            "NTRK1-CHARCOT-JOINTS-FROM-IGNORED-TRAUMA",
            "HSAN4-AR-CONGENITAL-NO-PAIN-NO-SWEAT",
        ],
    },

    # -- SCN9A — HSAN2D / CIP (LOF) + Inherited Erythromelalgia (GOF) ---------
    {
        "gene": "SCN9A",
        "alt_name": (
            "SCN9A (SCN9A-1988aa-2q24.3 / PARADOX-GENE — "
            "LOF-AR-HSAN2D-CIP-Complete-Pain-Insensitivity + "
            "GOF-AD-Inherited-Erythromelalgia-Burning-Feet-Redness-PATHOGNOMONIC — "
            "Carbamazepine-Nav1.7-GOF-Pain-Reduction)"
        ),
        "protein": (
            "SCN9A/Nav1.7 -- 2q24.3 -- SCN9A-1988aa -- "
            "Voltage-Gated-Sodium-Channel-Nav1.7-DRG-Nociceptors-Sympathetic-Neurons -- "
            "PARADOX-GENE-Two-Opposite-Phenotypes-Same-Channel -- "
            "LOF-AR-HSAN2D-CIP-Congenital-Insensitivity-to-Pain-Normal-Intellect-Anosmia-50pct -- "
            "LOF-Biallelic-Nav1.7-No-Nociceptor-Firing-No-Pain-Signal-Normal-Touch-Proprioception -- "
            "GOF-AD-Inherited-Erythromelalgia-IEM-Bilateral-Red-Hot-Feet-Burning-Pain-PATHOGNOMONIC -- "
            "GOF-Nav1.7-Hyperpolarised-Threshold-Enhanced-Nociceptor-Firing-Warmth-Triggers-Episodes -- "
            "Paroxysmal-Extreme-Pain-Disorder-PEPD-Also-GOF-SCN9A-Rectal-Ocular-Submandibular-Pain -- "
            "Carbamazepine-Na-Channel-Blocker-Reduces-Pain-in-IEM-GOF-Variants-pArg185His-pGly616Arg -- "
            "Anosmia-50pct-LOF-Olfactory-Nav1.7-Required"
        ),
        "locus": "2q24.3",
        "protein_size": "1988 aa",
        "inheritance": (
            "BIDIRECTIONAL: LOF = AR (biallelic loss-of-function) → HSAN2D/CIP; "
            "GOF = AD (gain-of-function heterozygous) → Inherited Erythromelalgia (IEM) or PEPD; "
            "LOF: truncating + missense abolishing channel function; "
            "GOF IEM: missense in voltage sensor/pore — Nav1.7 activates at hyperpolarised threshold; "
            "recurrence risk: LOF 25% siblings (AR); GOF 50% offspring (AD)"
        ),
        "age_of_onset": (
            "LOF (HSAN2D/CIP): birth; "
            "GOF IEM: childhood-adolescence (often before 10 yr)"
        ),
        "pathognomonic": (
            "LOF HSAN2D/CIP: complete pain insensitivity from birth + ANOSMIA 50% (distinguishes from NTRK1/HSAN4); "
            "normal intellect + normal autonomic function (distinguishes from NTRK1 = ID+anhidrosis); "
            "normal sweating (unlike NTRK1 anhidrosis); normal temperature sensation (distinguishes from partial HSANs); "
            "GOF IEM PATHOGNOMONIC: bilateral red/hot/swollen feet + BURNING PAIN episodes; "
            "triggered by warmth (walking, hot weather, exercise) → improved by ice water immersion; "
            "redness and heat = physical findings during episode; "
            "PEPD (PEPD variant): rectal/periorbital/mandibular pain episodes → triggered by defecation/eating"
        ),
        "treatment": (
            "LOF CIP: injury surveillance — regular podiatry, skin checks, dental review; "
            "avoid contact sports; medical alert bracelet (will not report pain — anaesthetic awareness check); "
            "GOF IEM: CARBAMAZEPINE 200-400 mg/day — reduces pain episodes in channel-blocking-sensitive GOF alleles; "
            "lidocaine IV for severe crises; cooling strategies (cool water foot baths); "
            "avoid triggers: warm rooms, exercise, hot showers; "
            "mexiletine (Na channel blocker) may help; "
            "gene-specific: SCN9A p.Arg185His responds to carbamazepine — genotype before choosing drug; "
            "PEPD: carbamazepine first-line"
        ),
        "key_biomarker": (
            "LOF: absent Nav1.7 protein on skin biopsy; absent SNAPs; genetic confirmation; "
            "GOF IEM: erythromelalgia clinical + genetic; provocation test (warm water) triggers redness/pain; "
            "SCN9A sequencing with functional annotation; "
            "Skin biopsy IENFD: normal in LOF CIP (unlike other HSANs)"
        ),
        "critical_flags": [
            "SCN9A-PARADOX-GENE-LOF-NO-PAIN-GOF-BURNING-PAIN",
            "SCN9A-LOF-CIP-ANOSMIA-50pct-NORMAL-INTELLECT",
            "SCN9A-GOF-ERYTHROMELALGIA-BURNING-FEET-PATHOGNOMONIC",
            "SCN9A-CARBAMAZEPINE-GOF-IEM-PAIN-REDUCTION",
            "SCN9A-IEM-WARMTH-TRIGGERS-ICE-WATER-RELIEVES",
            "SCN9A-CIP-NORMAL-AUTONOMIC-DDx-NTRK1-ANHIDROSIS",
            "HSAN2D-AR-LOF-IEM-AD-GOF-SAME-GENE",
        ],
    },

    # -- RETREG1 / FAM134B — HSAN2B -------------------------------------------
    {
        "gene": "RETREG1",
        "alt_name": (
            "RETREG1/FAM134B (RETREG1-460aa-5p15.1 / AR — HSAN2B-Hereditary-Sensory-Autonomic-Neuropathy-Type-2B — "
            "Corneal-Anaesthesia-Neurotrophic-Corneal-Ulceration-PATHOGNOMONIC — "
            "Congenital-Pan-Sensory-Loss-Mutilating-Arthropathy)"
        ),
        "protein": (
            "RETREG1/FAM134B -- 5p15.1 AR -- RETREG1-460aa -- "
            "Reticulophagy-Regulator-1-ER-Phagy-LIR-Motif-Selective-ER-Autophagy-Receptor -- "
            "HSAN2B-Hereditary-Sensory-Autonomic-Neuropathy-Type-2B-OMIM-613115 -- "
            "Loss-ER-Phagy-Sensory-Neuron-ER-Stress-Accumulation-Neuron-Death -- "
            "Congenital-Onset-Pan-Sensory-Loss-Pain-Temperature-Touch-Proprioception-All-Absent -- "
            "CORNEAL-ANAESTHESIA-PATHOGNOMONIC-Neurotrophic-Keratitis-Corneal-Ulceration -- "
            "Mutilating-Arthropathy-Hands-Feet-Skeletal-Deformity-Unrecognised-Fractures -- "
            "Autoamputation-Toes-Fingers-from-Repeated-Ignored-Trauma-Infection -- "
            "Turkish-Sudanese-Israeli-Arab-Consanguineous-Families-Founder-Mutations"
        ),
        "locus": "5p15.1",
        "protein_size": "460 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic truncating or missense mutations; "
            "Turkish/Sudanese/Israeli Arab founders; consanguineous families; "
            "FAM134B: LIR (LC3-interacting region) motif mediates ER phagy; "
            "loss → ER sheet expansion + ER stress → DRG/TG neuron apoptosis; "
            "complete penetrance; recurrence risk 25%"
        ),
        "age_of_onset": "Congenital; self-mutilation from teething; corneal ulcers from infancy",
        "pathognomonic": (
            "CORNEAL ANAESTHESIA → NEUROTROPHIC CORNEAL ULCERATION PATHOGNOMONIC — "
            "absent corneal reflex → minor trauma → corneal epithelial defects → ulceration → scarring → vision loss; "
            "inspect cornea at every visit; fluorescein staining mandatory; "
            "pan-sensory loss: pain, temperature, touch, vibration, proprioception all lost; "
            "mutilating arthropathy — Charcot joints, autoamputation of extremities; "
            "skeletal deformities from repeated unrecognised fractures; "
            "autonomic involvement mild (unlike HSAN3/FD); intellect usually normal"
        ),
        "treatment": (
            "CORNEA: preservative-free artificial tears QID; moisture chamber glasses; scleral contact lenses; "
            "corneal perforation risk → ophthalmology urgently if any redness/discharge; "
            "corneal tarsorrhaphy (partial lid closure) for severe neurotrophic keratitis; "
            "WOUND CARE: regular podiatry + dermatology; padding/orthotics; "
            "INFECTION: prompt antibiotic treatment of skin/bone infection; "
            "AVOID AMPUTATION where possible — infection control + offloading; "
            "ORTHOPAEDICS: serial casting for Charcot foot; "
            "PROTECTIVE EQUIPMENT: gloves, padding from infancy; "
            "GENETIC COUNSELLING: 25% recurrence; consanguinity counselling"
        ),
        "key_biomarker": (
            "Corneal anaesthesia (Cochet-Bonnet aesthesiometry: absent/reduced); "
            "Skin biopsy: absent IENFD (intraepidermal nerve fibres); "
            "NCS: absent SNAPs (all sensory); "
            "Nerve biopsy: absent unmyelinated + small myelinated fibres; "
            "FAM134B/RETREG1 gene sequencing"
        ),
        "critical_flags": [
            "RETREG1-CORNEAL-ANAESTHESIA-NEUROTROPHIC-ULCERATION-PATHOGNOMONIC",
            "RETREG1-PAN-SENSORY-LOSS-CONGENITAL",
            "RETREG1-MUTILATING-ARTHROPATHY-AUTOAMPUTATION",
            "RETREG1-ER-PHAGY-MECHANISM",
            "RETREG1-TURKISH-SUDANESE-ARAB-CONSANGUINEOUS",
            "RETREG1-OPHTHALMOLOGY-MANDATORY-CORNEA",
            "HSAN2B-AR-CONGENITAL-ALL-MODALITIES",
        ],
    },

    # -- WNK1 / HSN2 — HSAN2A -------------------------------------------------
    {
        "gene": "WNK1",
        "alt_name": (
            "WNK1-HSN2 (WNK1-2382aa-12p13.33 / AR-HSN2-Exon-Only — HSAN2A-Hereditary-Sensory-Autonomic-Neuropathy-Type-2A — "
            "Severe-Mutilation-Extremities-Toe-Finger-Loss-PATHOGNOMONIC — "
            "Congenital-Onset-<5yr-Pan-Sensory-Loss — "
            "Sudanese-Nova-Scotian-French-Canadian-Founders — "
            "HSN2-Exon-Standard-WES-May-Miss)"
        ),
        "protein": (
            "WNK1/HSN2 -- 12p13.33 AR -- WNK1-2382aa -- "
            "With-No-Lysine-Kinase-1-Serine-Threonine-Kinase-Ion-Cotransporter-Regulation -- "
            "HSAN2A-Hereditary-Sensory-Autonomic-Neuropathy-Type-2A-OMIM-201300 -- "
            "HSN2-Neuronal-Specific-Exon-Between-Exons-8-9-Mutations-Cause-HSAN2A -- "
            "WNK1-Kinase-SPAK-OSR1-Axis-KCC2-NKCC1-Cl-Cotransporter-Sensory-Neuron-Survival -- "
            "Congenital-Onset-<5-Years-Pan-Sensory-Loss-Pain-Temperature-Touch -- "
            "SEVERE-MUTILATION-EXTREMITIES-PATHOGNOMONIC-Toe-Finger-Amputation-Repeated-Trauma -- "
            "Sudanese-Founder-Mutations-Nova-Scotian-Acadian-French-Canadian-Populations -- "
            "HSN2-Exon-Not-Standard-WES-Annotation-Request-HSN2-Specific-Amplification"
        ),
        "locus": "12p13.33",
        "protein_size": "2382 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic mutations ONLY in HSN2 exon (neuronal isoform); "
            "WNK1 kinase main isoform mutations cause a separate condition (hypertension/pseudohypoaldosteronism); "
            "IMPORTANT: standard WES exome capture may not annotate HSN2 exon → request HSN2-specific sequencing; "
            "Sudanese/Nova Scotian (Acadian)/French-Canadian founders; "
            "complete penetrance; recurrence risk 25%"
        ),
        "age_of_onset": "Congenital to <5 years; severe from infancy",
        "pathognomonic": (
            "SEVERE MUTILATION OF EXTREMITIES PATHOGNOMONIC — self-inflicted biting + traumatic autoamputation; "
            "repeated fractures/dislocations → joint deformity → loss of toes, fingers, digits; "
            "PAN-SENSORY LOSS — pain, temperature, touch all severely impaired from birth; "
            "proprioception impaired; deep reflexes absent; "
            "autonomic involvement mild (no severe dysautonomia unlike HSAN3/FD); "
            "intellectual function normal; "
            "unlike HSAN4/CIPA: no anhidrosis, no ID; "
            "unlike HSAN3/FD: no autonomic crises, no fungiform papillae loss"
        ),
        "treatment": (
            "INJURY PREVENTION: padded gloves + helmets from infancy; "
            "dental occlusal guard to prevent tongue/lip biting; "
            "WOUND CARE: daily skin inspection; regular podiatry; orthopaedic shoes; "
            "osteomyelitis: prompt antibiotics + surgical debridement; "
            "AVOID AMPUTATION where clinically feasible — conservative management; "
            "PHYSIOTHERAPY: joint protection + mobility; "
            "OPHTHALMOLOGY: annual slit-lamp corneal check; "
            "GENETIC COUNSELLING: 25% recurrence; founder mutation testing for Sudanese/Acadian families; "
            "MEDICAL ALERT: patient will not report pain → anticipate undetected trauma"
        ),
        "key_biomarker": (
            "NCS: absent SNAPs all nerves (severe axonal sensory neuropathy); "
            "Skin biopsy: absent/severely reduced IENFD; "
            "Nerve biopsy: absent unmyelinated fibres; "
            "WNK1 HSN2 exon-specific sequencing (not standard WES); "
            "No biochemical biomarker"
        ),
        "critical_flags": [
            "WNK1-HSN2-EXON-MUTATIONS-ONLY",
            "WNK1-SEVERE-MUTILATION-EXTREMITIES-PATHOGNOMONIC",
            "WNK1-PAN-SENSORY-LOSS-CONGENITAL",
            "WNK1-STANDARD-WES-MAY-MISS-HSN2-EXON",
            "WNK1-SUDANESE-NOVA-SCOTIAN-ACADIAN-FOUNDERS",
            "WNK1-NO-ANHIDROSIS-DDx-NTRK1",
            "HSAN2A-AR-CONGENITAL-MUTILATING",
        ],
    },

    # -- DNMT1 — HSAN1E / ADCA-DN ---------------------------------------------
    {
        "gene": "DNMT1",
        "alt_name": (
            "DNMT1 (DNMT1-1616aa-19p13.2 / AD — HSAN1E-ADCA-DN-Autosomal-Dominant-Cerebellar-Ataxia-Deafness-Narcolepsy — "
            "Narcolepsy-Cataplexy-PATHOGNOMONIC + Sensorineural-Hearing-Loss + Dementia + Sensory-Neuropathy — "
            "4-Feature-Syndrome-PATHOGNOMONIC)"
        ),
        "protein": (
            "DNMT1 -- 19p13.2 AD -- DNMT1-1616aa -- "
            "DNA-Cytosine-5-Methyltransferase-1-Maintenance-Methylation-CpG-Methylation -- "
            "HSAN1E-ADCA-DN-Autosomal-Dominant-Cerebellar-Ataxia-Deafness-Narcolepsy-OMIM-614116 -- "
            "REMD-Domain-Mutations-Replication-Foci-Targeting-Sequence-Methyltransferase-Domain -- "
            "Mutant-DNMT1-Premature-Degradation-Global-Hypomethylation-Progressive-Neuron-Loss -- "
            "NARCOLEPSY-CATAPLEXY-PATHOGNOMONIC-Hypocretin-Producing-Neuron-Loss-Hypothalamus -- "
            "Sensorineural-Hearing-Loss-Progressive-Cochlear-Neuron-Involvement -- "
            "Dementia-Cognitive-Decline-Hippocampal-Atrophy-Progressive -- "
            "Sensory-Neuropathy-HSAN1-Shooting-Pains-Foot-Ulcers-Dorsal-Scars -- "
            "4-Feature-Syndrome-Any-Two-Prompt-DNMT1-Testing-No-Disease-Modifying-Therapy"
        ),
        "locus": "19p13.2",
        "protein_size": "1616 aa",
        "inheritance": (
            "AD (autosomal dominant); heterozygous missense mutations in REMD domain of DNMT1; "
            "mutations affect replication foci targeting → premature proteolytic degradation; "
            "global DNA hypomethylation → progressive neurodegeneration; "
            "onset 20-40 yr; full penetrance; 50% offspring risk; "
            "no founder mutation — sporadic de novo mutations also reported"
        ),
        "age_of_onset": "20-40 years; narcolepsy typically first, neuropathy/hearing/dementia follow",
        "pathognomonic": (
            "4-FEATURE SYNDROME PATHOGNOMONIC: "
            "① NARCOLEPSY-CATAPLEXY (irresistible sleep + muscle tone loss with laughter/emotion) — usually first feature; "
            "② SENSORINEURAL HEARING LOSS — progressive, bilateral, high-frequency first; "
            "③ DEMENTIA — executive dysfunction → global cognitive decline → late-stage dependence; "
            "④ SENSORY NEUROPATHY — distal pain/temperature loss, foot ulcers (HSAN1-pattern); "
            "ANY 2 OF 4 FEATURES → INVESTIGATE DNMT1; "
            "narcolepsy diagnosed by PSG + MSLT (mean sleep latency <8 min, ≥2 SOREMPs); "
            "low CSF hypocretin-1 (<110 pg/mL); "
            "MRI: cerebellar + hippocampal atrophy; white matter signal"
        ),
        "treatment": (
            "NARCOLEPSY: modafinil 200-400 mg/day (wakefulness); "
            "sodium oxybate (Xyrem) for cataplexy; "
            "methylphenidate (alternative stimulant); "
            "HEARING LOSS: hearing aids; cochlear implant if severe-profound; "
            "DEMENTIA: acetylcholinesterase inhibitors (donepezil/rivastigmine) — modest benefit; "
            "NEUROPATHY: pregabalin/gabapentin (neuropathic pain); wound care; "
            "NO DISEASE-MODIFYING THERAPY — no treatment reverses neurodegeneration; "
            "GENETIC COUNSELLING: 50% offspring risk; predictive testing from age 18; "
            "MULTIDISCIPLINARY: neurology + ENT + sleep specialist + psychiatry"
        ),
        "key_biomarker": (
            "PSG + MSLT: narcolepsy with cataplexy (confirmed); "
            "CSF hypocretin-1 <110 pg/mL; "
            "MRI: cerebellar + hippocampal atrophy; "
            "Pure tone audiogram: sensorineural; "
            "NCS: sensory axonal neuropathy; "
            "DNMT1 sequencing (REMD domain)"
        ),
        "critical_flags": [
            "DNMT1-NARCOLEPSY-CATAPLEXY-PATHOGNOMONIC-4-FEATURE",
            "DNMT1-4-FEATURE-ANY-2-PROMPT-TESTING",
            "DNMT1-HEARING-LOSS-DEMENTIA-NEUROPATHY-NARCOLEPSY",
            "DNMT1-NO-DISEASE-MODIFYING-THERAPY",
            "DNMT1-REMD-DOMAIN-MUTATIONS",
            "DNMT1-MODAFINIL-NARCOLEPSY-SODIUM-OXYBATE",
            "HSAN1E-ADCA-DN-AD-ADULT-ONSET",
        ],
    },

    # -- ATL1 — HSAN1D ---------------------------------------------------------
    {
        "gene": "ATL1",
        "alt_name": (
            "ATL1 (ATL1-558aa-14q22.1 / AD — HSAN1D-Hereditary-Sensory-Autonomic-Neuropathy-Type-1D — "
            "Chronic-Dry-Hoarse-Cough-PATHOGNOMONIC + GERD + Sensorineural-Hearing-Loss + Sensory-Neuropathy — "
            "Late-Onset-40-60yr)"
        ),
        "protein": (
            "ATL1/Atlastin-1 -- 14q22.1 AD -- ATL1-558aa -- "
            "Atlastin-1-ER-Tubule-Fusion-GTPase-Dynamin-Related-Protein-ER-Network-Formation -- "
            "HSAN1D-Hereditary-Sensory-Autonomic-Neuropathy-Type-1D-OMIM-613455 -- "
            "Also-Causes-Hereditary-Spastic-Paraplegia-SPG3A-Different-Variants -- "
            "Dominant-Variants-HSAN1D-Impair-ER-Tubule-Formation-Sensory-Axon-ER-Stress -- "
            "CHRONIC-DRY-HOARSE-COUGH-PATHOGNOMONIC-Autonomic-Airway-Neuropathy -- "
            "GERD-Gastro-Oesophageal-Reflux-Autonomic-Innervation-Loss-Lower-Oesophageal-Sphincter -- "
            "Sensorineural-Hearing-Loss-Progressive-Bilateral -- "
            "Sensory-Neuropathy-Distal-Lower-Limb-Late-Adult-Onset-40-60yr -- "
            "Note-SPG3A-ATL1-Variants-Different-Domain-Motor-Dominant-HSAN1D-Variants-Sensory-Autonomic"
        ),
        "locus": "14q22.1",
        "protein_size": "558 aa",
        "inheritance": (
            "AD (autosomal dominant); heterozygous missense mutations — HSAN1D-causing alleles differ from SPG3A-causing alleles in ATL1; "
            "HSAN1D variants: typically in GTPase domain (different from SPG3A middle domain); "
            "incomplete penetrance; rare condition; "
            "50% offspring risk if confirmed pathogenic HSAN1D variant; "
            "genotype-phenotype correlation important: report variant + phenotype to expert centre"
        ),
        "age_of_onset": "40-60 years (late-onset dominant neuropathy)",
        "pathognomonic": (
            "CHRONIC DRY/HOARSE COUGH PATHOGNOMONIC — autonomic neuropathy of airway → "
            "impaired cough reflex + hypersensitive sensory C-fibres; "
            "persistent cough preceding neuropathy diagnosis by years; "
            "often misdiagnosed as asthma/GERD/idiopathic cough; "
            "GERD PATHOGNOMONIC — autonomic loss of lower oesophageal sphincter tone → reflux; "
            "SENSORINEURAL HEARING LOSS — bilateral, progressive; "
            "SENSORY NEUROPATHY — distal lower > upper limb; "
            "pain/temperature loss → foot ulcers; "
            "Any 3 of 4 features (cough + GERD + SNHL + neuropathy) → INVESTIGATE ATL1"
        ),
        "treatment": (
            "COUGH: speech therapy for cough suppression; low-dose codeine (short-term); "
            "neuromodulators (pregabalin/amitriptyline) for hypersensitive airway; "
            "GERD: PPI (omeprazole/pantoprazole) daily; head-of-bed elevation; dietary modification; "
            "HEARING LOSS: hearing aids; ENT review; cochlear implant if severe-profound; "
            "NEUROPATHY: wound care + podiatry; pregabalin/gabapentin neuropathic pain; "
            "GENETIC COUNSELLING: 50% offspring risk; predictive testing; "
            "NO DISEASE-MODIFYING THERAPY"
        ),
        "key_biomarker": (
            "NCS: sensory axonal neuropathy distal lower limbs; "
            "Skin biopsy: reduced IENFD; "
            "Pure tone audiogram: SNHL; "
            "Laryngoscopy/bronchoscopy: normal (functional cough); "
            "ATL1 gene sequencing with HSAN1D-phenotype correlation"
        ),
        "critical_flags": [
            "ATL1-CHRONIC-COUGH-GERD-PATHOGNOMONIC",
            "ATL1-4-FEATURE-COUGH-GERD-SNHL-NEUROPATHY",
            "ATL1-LATE-ONSET-40-60yr",
            "ATL1-DIFFERENT-VARIANTS-FROM-SPG3A",
            "ATL1-OFTEN-MISDIAGNOSED-ASTHMA",
            "ATL1-PPI-FOR-GERD",
            "HSAN1D-AD-LATE-ONSET-AUTONOMIC-COUGH",
        ],
    },
]


def _build_cohort(gene_entry: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_entry["gene"]
    cohort = []
    for pid in range(40):
        if gene == "SPTLC1":
            onset = rng.randint(15, 42)
            pain_insensitivity = False
            burning_pain = False
            shooting_pains = True
            autonomic_crises = False
            self_mutilation = False
            anhidrosis = False
            corneal_ulcers = rng.random() < 0.15
            foot_ulcers = rng.random() < 0.80
            narcolepsy = False
            dementia = False
            hearing_loss = False
            anosmia = False
            serine_treatable = True
            l_serine_started = rng.random() < 0.60
        elif gene == "IKBKAP":
            onset = 0
            pain_insensitivity = rng.random() < 0.70
            burning_pain = False
            shooting_pains = False
            autonomic_crises = True
            self_mutilation = rng.random() < 0.30
            anhidrosis = rng.random() < 0.80
            corneal_ulcers = rng.random() < 0.40
            foot_ulcers = rng.random() < 0.20
            narcolepsy = False
            dementia = False
            hearing_loss = rng.random() < 0.20
            anosmia = False
            serine_treatable = False
            l_serine_started = False
        elif gene == "NTRK1":
            onset = 0
            pain_insensitivity = True
            burning_pain = False
            shooting_pains = False
            autonomic_crises = False
            self_mutilation = True
            anhidrosis = True
            corneal_ulcers = rng.random() < 0.30
            foot_ulcers = rng.random() < 0.60
            narcolepsy = False
            dementia = False
            hearing_loss = False
            anosmia = False
            serine_treatable = False
            l_serine_started = False
        elif gene == "SCN9A":
            # Mix: ~50% LOF CIP, ~50% GOF IEM
            is_lof = rng.random() < 0.50
            onset = 0 if is_lof else rng.randint(3, 15)
            pain_insensitivity = is_lof
            burning_pain = not is_lof
            shooting_pains = False
            autonomic_crises = False
            self_mutilation = rng.random() < 0.25 if is_lof else False
            anhidrosis = False
            corneal_ulcers = False
            foot_ulcers = rng.random() < 0.30 if is_lof else False
            narcolepsy = False
            dementia = False
            hearing_loss = False
            anosmia = rng.random() < 0.50 if is_lof else False
            serine_treatable = False
            l_serine_started = False
        elif gene == "RETREG1":
            onset = 0
            pain_insensitivity = True
            burning_pain = False
            shooting_pains = False
            autonomic_crises = False
            self_mutilation = rng.random() < 0.60
            anhidrosis = False
            corneal_ulcers = rng.random() < 0.70
            foot_ulcers = rng.random() < 0.70
            narcolepsy = False
            dementia = False
            hearing_loss = False
            anosmia = False
            serine_treatable = False
            l_serine_started = False
        elif gene == "WNK1":
            onset = rng.randint(0, 5)
            pain_insensitivity = True
            burning_pain = False
            shooting_pains = False
            autonomic_crises = False
            self_mutilation = rng.random() < 0.75
            anhidrosis = False
            corneal_ulcers = rng.random() < 0.20
            foot_ulcers = rng.random() < 0.80
            narcolepsy = False
            dementia = False
            hearing_loss = False
            anosmia = False
            serine_treatable = False
            l_serine_started = False
        elif gene == "DNMT1":
            onset = rng.randint(20, 45)
            pain_insensitivity = False
            burning_pain = False
            shooting_pains = rng.random() < 0.70
            autonomic_crises = False
            self_mutilation = False
            anhidrosis = False
            corneal_ulcers = False
            foot_ulcers = rng.random() < 0.40
            narcolepsy = True
            dementia = rng.random() < 0.85
            hearing_loss = True
            anosmia = False
            serine_treatable = False
            l_serine_started = False
        else:  # ATL1
            onset = rng.randint(38, 62)
            pain_insensitivity = False
            burning_pain = False
            shooting_pains = rng.random() < 0.40
            autonomic_crises = False
            self_mutilation = False
            anhidrosis = False
            corneal_ulcers = False
            foot_ulcers = rng.random() < 0.25
            narcolepsy = False
            dementia = False
            hearing_loss = True
            anosmia = False
            serine_treatable = False
            l_serine_started = False

        cohort.append({
            "patient_id": f"{gene}-{pid+1:03d}",
            "onset_age": onset,
            "pain_insensitivity": pain_insensitivity,
            "burning_pain": burning_pain,
            "shooting_pains": shooting_pains,
            "autonomic_crises": autonomic_crises,
            "self_mutilation": self_mutilation,
            "anhidrosis": anhidrosis,
            "corneal_ulcers": corneal_ulcers,
            "foot_ulcers": foot_ulcers,
            "narcolepsy": narcolepsy,
            "dementia": dementia,
            "hearing_loss": hearing_loss,
            "anosmia": anosmia,
            "serine_treatable": serine_treatable,
            "l_serine_started": l_serine_started,
        })
    return cohort


_ALL_COHORTS = {
    g["gene"]: _build_cohort(g, SEED_BASE + i)
    for i, g in enumerate(HSAN_GENES)
}


def overview() -> dict:
    """Aggregate overview across all 8 HSAN genes (320 patients, seeds 2126-2133)."""
    all_patients = [p for cohort in _ALL_COHORTS.values() for p in cohort]
    n = len(all_patients)

    sptlc1_cohort = _ALL_COHORTS["SPTLC1"]
    ikbkap_cohort = _ALL_COHORTS["IKBKAP"]
    ntrk1_cohort  = _ALL_COHORTS["NTRK1"]
    scn9a_cohort  = _ALL_COHORTS["SCN9A"]
    retreg1_cohort = _ALL_COHORTS["RETREG1"]
    wnk1_cohort   = _ALL_COHORTS["WNK1"]
    dnmt1_cohort  = _ALL_COHORTS["DNMT1"]
    atl1_cohort   = _ALL_COHORTS["ATL1"]

    return {
        "atlas": "Hereditary HSAN Atlas — Complete 8-Gene Hereditary Sensory and Autonomic Neuropathy Reference",
        "genes": [g["gene"] for g in HSAN_GENES],
        "total_patients": n,
        "seeds": "2126-2133",
        "seed_base": SEED_BASE,
        # SPTLC1
        "sptlc1_shooting_pains_patients": sum(1 for p in sptlc1_cohort if p["shooting_pains"]),
        "sptlc1_foot_ulcers_patients": sum(1 for p in sptlc1_cohort if p["foot_ulcers"]),
        "sptlc1_l_serine_started_patients": sum(1 for p in sptlc1_cohort if p["l_serine_started"]),
        # IKBKAP
        "ikbkap_autonomic_crises_patients": sum(1 for p in ikbkap_cohort if p["autonomic_crises"]),
        "ikbkap_anhidrosis_patients": sum(1 for p in ikbkap_cohort if p["anhidrosis"]),
        "ikbkap_corneal_ulcers_patients": sum(1 for p in ikbkap_cohort if p["corneal_ulcers"]),
        # NTRK1
        "ntrk1_pain_insensitivity_patients": sum(1 for p in ntrk1_cohort if p["pain_insensitivity"]),
        "ntrk1_self_mutilation_patients": sum(1 for p in ntrk1_cohort if p["self_mutilation"]),
        "ntrk1_anhidrosis_patients": sum(1 for p in ntrk1_cohort if p["anhidrosis"]),
        # SCN9A
        "scn9a_cip_lof_patients": sum(1 for p in scn9a_cohort if p["pain_insensitivity"]),
        "scn9a_iem_gof_patients": sum(1 for p in scn9a_cohort if p["burning_pain"]),
        "scn9a_anosmia_patients": sum(1 for p in scn9a_cohort if p["anosmia"]),
        # RETREG1
        "retreg1_corneal_ulcers_patients": sum(1 for p in retreg1_cohort if p["corneal_ulcers"]),
        "retreg1_self_mutilation_patients": sum(1 for p in retreg1_cohort if p["self_mutilation"]),
        # WNK1
        "wnk1_self_mutilation_patients": sum(1 for p in wnk1_cohort if p["self_mutilation"]),
        "wnk1_foot_ulcers_patients": sum(1 for p in wnk1_cohort if p["foot_ulcers"]),
        # DNMT1
        "dnmt1_narcolepsy_patients": sum(1 for p in dnmt1_cohort if p["narcolepsy"]),
        "dnmt1_dementia_patients": sum(1 for p in dnmt1_cohort if p["dementia"]),
        "dnmt1_hearing_loss_patients": sum(1 for p in dnmt1_cohort if p["hearing_loss"]),
        # ATL1
        "atl1_hearing_loss_patients": sum(1 for p in atl1_cohort if p["hearing_loss"]),
        # Cross-atlas
        "treatable_patients": (
            sum(1 for p in sptlc1_cohort if p["serine_treatable"])
        ),
        "pain_insensitivity_all_patients": sum(
            1 for p in all_patients if p["pain_insensitivity"]
        ),
        "self_mutilation_all_patients": sum(
            1 for p in all_patients if p["self_mutilation"]
        ),
        "corneal_ulcer_risk_patients": sum(
            1 for p in all_patients if p["corneal_ulcers"]
        ),
    }


def breakdown() -> dict:
    """Per-gene clinical breakdown for all 8 HSAN genes."""
    result = {}
    for g in HSAN_GENES:
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
            "pain_insensitivity_pct": round(100 * sum(1 for p in cohort if p["pain_insensitivity"]) / n, 1),
            "burning_pain_pct": round(100 * sum(1 for p in cohort if p["burning_pain"]) / n, 1),
            "shooting_pains_pct": round(100 * sum(1 for p in cohort if p["shooting_pains"]) / n, 1),
            "autonomic_crises_pct": round(100 * sum(1 for p in cohort if p["autonomic_crises"]) / n, 1),
            "self_mutilation_pct": round(100 * sum(1 for p in cohort if p["self_mutilation"]) / n, 1),
            "anhidrosis_pct": round(100 * sum(1 for p in cohort if p["anhidrosis"]) / n, 1),
            "corneal_ulcers_pct": round(100 * sum(1 for p in cohort if p["corneal_ulcers"]) / n, 1),
            "foot_ulcers_pct": round(100 * sum(1 for p in cohort if p["foot_ulcers"]) / n, 1),
            "narcolepsy_pct": round(100 * sum(1 for p in cohort if p["narcolepsy"]) / n, 1),
            "dementia_pct": round(100 * sum(1 for p in cohort if p["dementia"]) / n, 1),
            "hearing_loss_pct": round(100 * sum(1 for p in cohort if p["hearing_loss"]) / n, 1),
        }
    return result


def definitions() -> dict:
    """Gene definitions, glossary, and surveillance protocols."""
    return {
        "genes": {
            g["gene"]: g["protein"]
            for g in HSAN_GENES
        },
        "glossary": {
            "HSAN (Hereditary Sensory and Autonomic Neuropathy)": (
                "A group of hereditary neuropathies primarily affecting sensory and autonomic neurons. "
                "Classified HSAN1-5 historically by phenotype; now classified by gene. "
                "Key distinction from CMT/HMSN: sensory >> motor involvement (CMT = motor + sensory equally). "
                "Clinical hallmark: loss of pain/temperature ± autonomic features ± self-mutilation. "
                "8 genes covered here span the major subtypes encountered in genetics clinics."
            ),
            "Deoxy-sphingolipids (SPTLC1/HSAN1A)": (
                "SPTLC1 gain-of-function mutations allow L-alanine (instead of L-serine) as substrate, "
                "producing 1-deoxy-sphinganine (1-DSa) and 1-deoxysphingosine (1-DS). "
                "These atypical metabolites cannot be degraded normally and accumulate in sensory neurons. "
                "Plasma deoxy-SL measurement is the diagnostic biomarker. "
                "L-serine supplementation 400 mg/kg/day competitively reduces alanine incorporation → "
                "reduces deoxy-SL → slows neuropathy. This is the first evidence-based treatment for HSAN1."
            ),
            "Familial Dysautonomia (HSAN3 / Riley-Day / IKBKAP)": (
                "Most common HSAN worldwide (in Ashkenazi Jewish population — 1:3700 births, carrier 1:30). "
                "ABSENT FUNGIFORM PAPILLAE — inspect tongue; smooth dorsum = immediate bedside diagnosis. "
                "Autonomic crises = episodic vomiting + hyperhidrosis + hypertension → diazepam IV + ondansetron. "
                "IVS20+6T>C splicing mutation in >99.5% FD alleles — Ashkenazi panel covers this. "
                "Ataluren (PTC124): read-through for premature stop codons — compassionate use ongoing."
            ),
            "CIPA (Congenital Insensitivity to Pain with Anhidrosis / HSAN4 / NTRK1)": (
                "No TRKA (NGF receptor) → nociceptive + sympathetic neurons fail to survive development. "
                "Clinical consequence: ① pain insensitivity (self-mutilation from teething) + "
                "② anhidrosis (heat stroke risk — most common early cause of death). "
                "AVOID AMPUTATION: fractures heal; Charcot joints develop but function preserves. "
                "Fever management is the primary intervention — cool environment always."
            ),
            "SCN9A Nav1.7 Paradox Gene": (
                "Same gene (Nav1.7/SCN9A) → opposite pain phenotypes by mutation class: "
                "LOF (AR biallelic) = CIP (no pain at all; anosmia 50%; normal intellect); "
                "GOF (AD heterozygous) = Inherited Erythromelalgia (burning pain + redness bilateral feet). "
                "Nav1.7 is gatekeeper for nociceptor excitability. "
                "Therapeutic target: SCN9A GOF IEM → carbamazepine (Na-channel blocker) reduces episodes. "
                "Genotype before choosing drug — variant-specific pharmacogenomics applies."
            ),
            "Neurotrophic Keratitis (RETREG1/WNK1 corneal anaesthesia)": (
                "Absent corneal sensation → minor trauma goes undetected → epithelial defect → "
                "bacterial superinfection → corneal ulcer → perforation → blindness. "
                "Prevention: preservative-free artificial tears + moisture chambers + scleral lenses. "
                "Detect: Cochet-Bonnet aesthesiometry at each visit; fluorescein staining of cornea. "
                "Management: ophthalmology urgently at first sign of redness; tarsorrhaphy for severe cases."
            ),
            "WNK1 HSN2 Exon — Why Standard WES Misses It": (
                "WNK1 has a neuronal-specific exon (HSN2) inserted between exons 8-9 of the ubiquitous isoform. "
                "HSAN2A mutations occur exclusively in this HSN2 exon. "
                "Standard WES exome capture designs may not include this exon or may not annotate it. "
                "Clinical consequence: a WES-negative HSAN2A patient could be false-negative. "
                "Request: WNK1 HSN2 exon-specific amplification + Sanger sequencing if HSAN2 phenotype."
            ),
            "DNMT1 4-Feature Syndrome (HSAN1E/ADCA-DN)": (
                "Narcolepsy-cataplexy + sensorineural hearing loss + dementia + sensory neuropathy. "
                "Narcolepsy (irresistible sleep + cataplexy) is typically the first feature (20-30s). "
                "Any 2 of 4 features in a young adult → investigate DNMT1 REMD domain. "
                "CSF hypocretin-1 <110 pg/mL confirms narcolepsy type 1 (hypocretin neuron loss). "
                "No disease-modifying therapy; modafinil/oxybate symptomatically manage narcolepsy."
            ),
            "ATL1 HSAN1D vs SPG3A": (
                "ATL1 (Atlastin-1) mutations cause two distinct conditions depending on domain: "
                "SPG3A (Hereditary Spastic Paraplegia type 3A) — dominant motor > sensory; "
                "HSAN1D — dominant sensory + autonomic (cough + GERD + hearing loss + neuropathy). "
                "Never conflate these — genotype-phenotype correlation is essential. "
                "HSAN1D variants cluster differently from SPG3A variants in ATL1 structure. "
                "Chronic cough misdiagnosed as asthma/GERD alone for years before neuropathy identified."
            ),
        },
        "surveillance_protocols": {
            "SPTLC1 (HSAN1A)": (
                "SPTLC1 sequencing (NGS panel or WES); "
                "Plasma deoxy-sphingolipids (1-DSa + 1-DS) at diagnosis + 6-monthly on L-serine; "
                "NCS annually: sensorimotor axonal neuropathy — SNAP amplitudes as progression markers; "
                "Skin biopsy IENFD at diagnosis; "
                "L-SERINE 400 mg/kg/day: start at genetic confirmation; target plasma dSL reduction; "
                "PODIATRY: quarterly; wound clinic if ulcer; orthopaedic shoes; "
                "INFECTION: X-ray + MRI if suspected osteomyelitis; "
                "PAIN: pregabalin 75-300 mg BD for neuropathic pain; "
                "GENETIC COUNSELLING: 50% offspring risk; presymptomatic testing ≥18 yr; "
                "CASCADE TESTING: first-degree relatives; asymptomatic carriers may benefit from L-serine"
            ),
            "IKBKAP (HSAN3/Familial Dysautonomia)": (
                "IKBKAP IVS20+6T>C targeted mutation (Ashkenazi) or full sequencing; "
                "CRISIS MANAGEMENT PROTOCOL: diazepam 0.1 mg/kg IV + ondansetron 0.15 mg/kg IV; "
                "CRISIS DIARY: frequency, duration, triggers; "
                "Schirmer test 3-monthly — alacrima severity; "
                "Slit-lamp cornea 3-monthly — neurotrophic keratitis; "
                "Scoliosis: spine X-ray annually — surgical if Cobb >40°; "
                "Swallowing: video fluoroscopy/MBS annually — aspiration risk; "
                "Postural BP: 3-monthly; fludrocortisone + compression garments; "
                "Pulmonary function: spirometry annually — aspiration pneumonia risk; "
                "ATALUREN: refer to FD compassionate use programme (foundation/expert centre); "
                "GENETIC COUNSELLING: AR 25% risk; Ashkenazi carrier testing panel"
            ),
            "NTRK1 (HSAN4/CIPA)": (
                "NTRK1 sequencing; "
                "HEAT MANAGEMENT: home thermometer; fans; cool environment mandatory; "
                "temperature monitoring: check BT daily in hot weather; "
                "SELF-INJURY: mouth guard at teething onset; soft helmet; padded gloves; "
                "DENTAL: 3-monthly dental review; "
                "SKIN/ORTHOPAEDIC: monthly skin check; X-ray any swollen joint; "
                "OPHTHALMOLOGY: 6-monthly corneal slit-lamp; "
                "Bone health: DEXA scan — Charcot joint workup; "
                "DEVELOPMENT: IQ assessment + educational planning; "
                "PHYSIOTHERAPY: joint protection; orthotics; "
                "GENETIC COUNSELLING: 25% risk; consanguinity"
            ),
            "SCN9A (HSAN2D/CIP + IEM)": (
                "SCN9A sequencing with functional annotation (LOF vs GOF); "
                "LOF/CIP: injury surveillance — podiatry, skin checks, dental; "
                "LOF/CIP: medical alert bracelet; "
                "NCS: confirm absent SNAPs; "
                "GOF/IEM: carbamazepine 200-400 mg/day — trial 3 months; assess pain diary; "
                "GOF/IEM: pain diary (frequency + severity + triggers); "
                "GOF/IEM: mexiletine 150-300 mg BD (if CBZ fails); "
                "COOLING STRATEGIES: cool water foot basin during episodes; "
                "GOF/IEM: avoid triggers (warm rooms, exercise, hot water); "
                "Olfactory testing: scratch-and-sniff (LOF anosmia 50%); "
                "GENETIC COUNSELLING: LOF AR 25%; GOF AD 50% offspring"
            ),
            "RETREG1 (HSAN2B)": (
                "RETREG1/FAM134B sequencing; "
                "OPHTHALMOLOGY: 3-monthly — Cochet-Bonnet aesthesiometry + fluorescein staining; "
                "Scleral lenses + preservative-free tears + moisture chambers; "
                "Tarsorrhaphy referral if neurotrophic keratitis progressive; "
                "SKIN/ORTHOPAEDIC: monthly skin check; X-ray any swollen/deformed extremity; "
                "Podiatry quarterly; custom offloading footwear; "
                "INFECTION: prompt antibiotics + bone MRI for osteomyelitis; "
                "PROTECTIVE EQUIPMENT: padded gloves, helmet; "
                "NCS annually: axonal sensory neuropathy progression; "
                "GENETIC COUNSELLING: AR 25%; consanguinity counselling"
            ),
            "WNK1 (HSAN2A)": (
                "WNK1 HSN2 exon-specific sequencing (NOT standard WES alone — may miss); "
                "INJURY PREVENTION: padded equipment; mouth guard from teething; "
                "PODIATRY: quarterly; wound care; orthopaedic assessment; "
                "ORTHOPAEDIC: X-ray any swollen joint/deformity; serial casting for Charcot foot; "
                "OPHTHALMOLOGY: annual slit-lamp cornea; "
                "NCS annually: absent SNAPs; "
                "MEDICAL ALERT: will not report pain — anticipate undetected trauma in emergency; "
                "INFECTION: low threshold for antibiotics + imaging; "
                "GENETIC COUNSELLING: AR 25%; Sudanese/Acadian communities — cascade testing"
            ),
            "DNMT1 (HSAN1E/ADCA-DN)": (
                "DNMT1 REMD domain sequencing; "
                "SLEEP: PSG + MSLT — diagnosis + severity; "
                "CSF hypocretin-1 measurement; "
                "MODAFINIL 200-400 mg/day: wakefulness; "
                "SODIUM OXYBATE: cataplexy; "
                "COGNITIVE: neuropsychology annually; "
                "AUDIOLOGY: pure tone audiogram 6-monthly; hearing aids early; "
                "cochlear implant referral if severe-profound loss; "
                "NCS 6-monthly: sensory axonal neuropathy; "
                "MRI brain annual: cerebellar + hippocampal atrophy; "
                "WOUND CARE: podiatry; neuropathic pain: pregabalin; "
                "GENETIC COUNSELLING: AD 50% offspring risk; predictive testing ≥18 yr"
            ),
            "ATL1 (HSAN1D)": (
                "ATL1 sequencing — confirm HSAN1D variant (not SPG3A allele); "
                "COUGH: speech therapy; low-dose codeine; pregabalin/amitriptyline; "
                "laryngoscopy to exclude structural cause; "
                "GERD: PPI daily; 24-hr pH-metry if needed; head-of-bed elevation; "
                "AUDIOLOGY: pure tone audiogram annually; hearing aids; "
                "cochlear implant if severe-profound SNHL; "
                "NCS annually: distal sensory axonal neuropathy; "
                "Skin biopsy IENFD at diagnosis; "
                "PODIATRY: quarterly foot check; wound care; "
                "GENETIC COUNSELLING: AD 50% offspring; predictive testing from age 18; "
                "DISTINGUISH from SPG3A: report variant to genotype-phenotype registry"
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
    print(f"SPTLC1 shooting pains: {ov['sptlc1_shooting_pains_patients']}")
    print(f"IKBKAP autonomic crises: {ov['ikbkap_autonomic_crises_patients']}")
    print(f"NTRK1 pain insensitivity: {ov['ntrk1_pain_insensitivity_patients']}")
    print(f"NTRK1 self-mutilation: {ov['ntrk1_self_mutilation_patients']}")
    print(f"SCN9A CIP (LOF): {ov['scn9a_cip_lof_patients']}")
    print(f"SCN9A IEM (GOF): {ov['scn9a_iem_gof_patients']}")
    print(f"RETREG1 corneal ulcers: {ov['retreg1_corneal_ulcers_patients']}")
    print(f"WNK1 self-mutilation: {ov['wnk1_self_mutilation_patients']}")
    print(f"DNMT1 narcolepsy: {ov['dnmt1_narcolepsy_patients']}")
    print(f"DNMT1 dementia: {ov['dnmt1_dementia_patients']}")
    print(f"ATL1 hearing loss: {ov['atl1_hearing_loss_patients']}")
    print(f"Treatable patients (SPTLC1 L-serine): {ov['treatable_patients']}")
    print(f"Pain insensitivity all: {ov['pain_insensitivity_all_patients']}")
    print(f"Self-mutilation all: {ov['self_mutilation_all_patients']}")
