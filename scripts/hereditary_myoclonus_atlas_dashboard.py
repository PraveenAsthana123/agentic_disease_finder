#!/usr/bin/env python3
"""Hereditary-Myoclonus-Atlas — Complete 8-Gene Progressive Myoclonic Epilepsy (PME) Atlas.

CSTB    (Cystatin B; 98 aa; 21q22.3; AR;
          EPM1 / Unverricht-Lundborg disease (ULD) — most common PME worldwide;
          dodecamer repeat expansion 5'-CCCCGCCCCGCG-3' (>30 copies) PATHOGNOMONIC;
          action myoclonus + GTCS + progressive cerebellar ataxia; piracetam Level A;
          NOT fatal; rehabilitation focus; seed SEED_BASE+0).
EPM2A   (Laforin dual-specificity phosphatase; 331 aa; 6q24.3; AR;
          Lafora disease type 1 — Lafora bodies PATHOGNOMONIC on skin biopsy;
          teenage onset; rapid progression; occipital seizures with visual aura;
          VPA + levetiracetam; NO CURE; seed SEED_BASE+1).
NHLRC1  (Malin E3 ubiquitin ligase; 395 aa; 6p22.3; AR;
          Lafora disease type 2 — same phenotype as EPM2A; skin biopsy same;
          EPM2A vs NHLRC1 clinically indistinguishable — distinguish by gene;
          Mediterranean / South Asian / Middle East founder; seed SEED_BASE+2).
SCARB2  (Scavenger receptor class B member 2 / LIMP2; 478 aa; 4q21.1; AR;
          EPM4 / Action Myoclonus-Renal Failure (AMRF) syndrome;
          RENAL FAILURE mandatory screen (progressive nephropathy) PATHOGNOMONIC co-feature;
          hearing loss; AVOID NSAIDs and nephrotoxins; seed SEED_BASE+3).
GOSR2   (Golgi SNAP receptor complex member 2; 235 aa; 17q21.32; AR;
          EPM6 / GOSA syndrome / Nord disease;
          SCOLIOSIS in 100% PATHOGNOMONIC feature; early onset 2-6 yr;
          raised CK; predominantly northern European; piracetam; seed SEED_BASE+4).
KCNC1   (Kv3.1 potassium channel; 585 aa; 11p15.1; AD;
          EPM7 — R320H dominant-negative founder (Finnish-Baltic);
          Giant somatosensory evoked potentials (Giant SEPs) PATHOGNOMONIC;
          CBZ/OXC/PHT/LTG ABSOLUTE CONTRAINDICATED; piracetam Level C; seed SEED_BASE+5).
PRICKLE1 (Prickle-like protein 1; 831 aa; 12q12; AR;
          EPM1B / Unverricht-Lundborg-like — clinically similar to CSTB;
          action myoclonus + GTCS + slower cerebellar progression than ULD;
          piracetam + levetiracetam; prognosis better than Lafora; seed SEED_BASE+6).
KCTD7   (Potassium channel tetramerisation domain 7; 289 aa; 12q14.2; AR;
          EPM3 — infantile-onset 1-2 yr, severe ID + myoclonus + ataxia;
          early infantile onset distinguishes from CSTB/EPM2A;
          VPA; seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2086-2093).
"""

import random

SEED_BASE = 2086

MYOCLONUS_GENES = [
    # -- CSTB — Unverricht-Lundborg Disease (EPM1) ----------------------------------
    {
        "gene": "CSTB",
        "alt_name": (
            "CSTB (CSTB-98aa-21q22.3 / AR — EPM1-Unverricht-Lundborg-Disease — "
            "Dodecamer-Repeat-Expansion->30-Copies-PATHOGNOMONIC — "
            "Action-Myoclonus-Piracetam-Level-A — NOT-Fatal-Rehabilitation-Focus)"
        ),
        "protein": (
            "CSTB -- 21q22.3 AR -- CSTB-98aa -- "
            "Cystatin-B-Type-2-Cysteine-Protease-Inhibitor-Nuclear-Cytoplasmic-98aa -- "
            "Dodecamer-Repeat-5prime-CCCCGCCCCGCG-3prime-Promoter-Region-Expansion -- "
            "Normal-2-3-Copies-Pathogenic->30-Copies-PATHOGNOMONIC -- "
            "EPM1-Unverricht-Lundborg-Disease-Most-Common-PME-Worldwide -- "
            "Onset-6-16yr-Action-Myoclonus-Photosensitivity-GTCS -- "
            "Progressive-Cerebellar-Ataxia-Dysarthria-Intentional-Tremor -- "
            "NOT-Fatal-Life-Expectancy-Near-Normal-Rehabilitation-Focus -- "
            "Piracetam-Level-A-Action-Myoclonus -- "
            "CBZ-OXC-PHT-ABSOLUTE-CONTRAINDICATED-Worsen-Myoclonus"
        ),
        "locus": "21q22.3",
        "protein_size": "98 aa",
        "inheritance": (
            "AR (autosomal recessive) — biallelic loss-of-function; "
            "Most common mutation: dodecamer repeat expansion (>30 copies) in 5' promoter — "
            "accounts for >90% alleles (EPM1 type 1); point mutations in minority; "
            "Prevalence: 1/20,000 in Finland/Baltic (founder); 1/100,000 elsewhere; "
            "Finnish-Baltic + Mediterranean founder populations enriched; "
            "Heterozygous carriers: asymptomatic; no anticipation reported; "
            "No paternal/maternal imprinting; true autosomal"
        ),
        "pathognomonic": (
            "DODECAMER REPEAT EXPANSION >30 copies (5'-CCCCGCCCCGCG-3') — molecular PATHOGNOMONIC; "
            "ACTION MYOCLONUS — stimulus-sensitive (touch, light, sound, movement); "
            "Photosensitivity (PPR on EEG) 80%; "
            "Giant cortical SEPs (somatosensory evoked potentials) on neurophysiology; "
            "C-reflex (long-latency reflex) pathologically enhanced; "
            "Progressive cerebellar ataxia (gait, dysarthria, intention tremor); "
            "GTCS — may predate myoclonus; "
            "Course: slow progression then stabilisation — NOT fatal; "
            "Cognitive: mild impairment (not severe dementia as in Lafora)"
        ),
        "treatment": (
            "**Piracetam** — Level A evidence for action myoclonus (ULD-specific); start 2.4-4.8g/day; "
            "**Levetiracetam** — Level B; "
            "**Valproate** — effective GTCS + myoclonus (caution POLG1 screen); "
            "**Clonazepam** — adjunctive for myoclonus; "
            "**Perampanel** — adjunctive Level C; "
            "**Zonisamide** — adjunctive Level C; "
            "**ABSOLUTE CONTRAINDICATED**: CBZ, OXC, PHT, LTG, GBP, Vigabatrin — ALL worsen myoclonus; "
            "**Rehabilitation**: physiotherapy (gait/balance), SLT (dysarthria), occupational therapy; "
            "**Photosensitivity**: tinted lenses, screen filters; "
            "**Prognosis**: near-normal life expectancy; aim for community participation"
        ),
        "critical_flags": [
            "DODECAMER-REPEAT->30-COPIES-PATHOGNOMONIC-MOLECULAR",
            "PIRACETAM-LEVEL-A-ACTION-MYOCLONUS-ULD-SPECIFIC",
            "CBZ-OXC-PHT-LTG-ABSOLUTE-CONTRAINDICATED",
            "GBP-PGB-WORSEN-MYOCLONUS-AVOID",
            "NOT-FATAL-NEAR-NORMAL-LIFE-EXPECTANCY",
            "GIANT-SEPs-C-REFLEX-PATHOGNOMONIC-NEUROPHYSIOLOGY",
            "PHOTOSENSITIVITY-PPR-80pct",
            "REHABILITATION-FOCUS-PHYSIOTHERAPY-SLT",
        ],
        "age_of_onset": "6-16 yr (childhood/adolescence); first symptom often GTCS before myoclonus evident; myoclonus peaks 15-25 yr",
        "key_biomarker": "Dodecamer repeat expansion PCR (>30 copies pathogenic); Giant cortical SEPs; C-reflex; PPR on EEG",
        "seed_offset": 0,
    },

    # -- EPM2A — Lafora Disease Type 1 (Laforin) ------------------------------------
    {
        "gene": "EPM2A",
        "alt_name": (
            "EPM2A (EPM2A-331aa-6q24.3 / AR — Lafora-Disease-Type-1-Laforin-Phosphatase — "
            "Lafora-Bodies-Skin-Biopsy-PATHOGNOMONIC — Occipital-Seizures-Visual-Aura — "
            "Rapid-Fatal-Progression-10-15yr-Death)"
        ),
        "protein": (
            "EPM2A -- 6q24.3 AR -- EPM2A-331aa -- "
            "Laforin-Dual-Specificity-Protein-Phosphatase-Glucan-Phosphatase -- "
            "Carbohydrate-Binding-Module-CBD-N-terminal-Binds-Glycogen-Polyglucosans -- "
            "Phosphatase-Domain-Removes-Phosphate-From-Glycogen-Prevents-Lafora-Body -- "
            "Lafora-Disease-Type-1-Malformed-Polyglucosan-Lafora-Body-Accumulation -- "
            "Neurons-Liver-Muscle-Heart-Sweat-Glands-Affected -- "
            "Skin-Biopsy-Axillary-Sweat-Gland-Ducts-Lafora-Bodies-PAS-Positive -- "
            "Rapid-Neurological-Decline-Dementia-Seizures-Death-10-15yr-Onset"
        ),
        "locus": "6q24.3",
        "protein_size": "331 aa",
        "inheritance": (
            "AR (autosomal recessive) — biallelic EPM2A loss-of-function; "
            "50% of Lafora disease; accounts for ~50% of molecularly confirmed LD; "
            "Mediterranean (Spanish, Italian, North African, Middle Eastern, South Asian) founder; "
            "Prevalence: 1-2/1,000,000 worldwide; "
            "Point mutations + small indels most common; no large deletions typical; "
            "Consanguinity enriched in Middle East/South Asia populations; "
            "EPM2A vs NHLRC1 (Malin): clinically IDENTICAL — distinguish by sequencing only"
        ),
        "pathognomonic": (
            "LAFORA BODIES on skin biopsy (axillary region, sweat gland duct cells) — PATHOGNOMONIC; "
            "PAS-positive, diastase-resistant polyglucosan inclusions; "
            "OCCIPITAL SEIZURES with visual aura (flashing lights, blindness) — characteristic early feature; "
            "Myoclonic seizures — stimulus-sensitive action myoclonus; "
            "RAPID COGNITIVE DECLINE — dementia within 2-3 yr of onset (distinguishes from ULD); "
            "Onset 12-17 yr; previously well; GTCS; "
            "Drop attacks; focal seizures; "
            "Death within 10-15 yr of onset (respiratory/aspiration)"
        ),
        "treatment": (
            "**Valproate** — first-line for all seizure types; "
            "**Levetiracetam** — first-line adjunctive; "
            "**Clonazepam** — myoclonus adjunct; "
            "**Zonisamide** — adjunctive; "
            "**Perampanel** — adjunctive (care: aggression); "
            "**Topiramate** — adjunctive (carbonic anhydrase → glucose metabolism) research interest; "
            "**ABSOLUTE CONTRAINDICATED**: CBZ, OXC, PHT, LTG (worsen myoclonus); VGB (worsen); "
            "**Investigational**: metformin (AMPK/mTOR → polyglucosan reduction, trial ongoing); "
            "**Investigational**: antisense oligonucleotides (ASO) targeting GYS1 (ongoing trials); "
            "**Palliative care** — disease is currently fatal; multidisciplinary support for family"
        ),
        "critical_flags": [
            "LAFORA-BODIES-SKIN-BIOPSY-PATHOGNOMONIC-PAS-POSITIVE",
            "RAPID-FATAL-PROGRESSION-DEATH-10-15yr",
            "OCCIPITAL-SEIZURES-VISUAL-AURA-EARLY-FEATURE",
            "RAPID-COGNITIVE-DECLINE-DEMENTIA-DISTINGUISHES-FROM-ULD",
            "CBZ-OXC-PHT-LTG-ABSOLUTE-CONTRAINDICATED",
            "EPM2A-vs-NHLRC1-CLINICALLY-INDISTINGUISHABLE-GENE-ONLY",
            "METFORMIN-ASO-TRIALS-ONGOING",
            "SKIN-BIOPSY-AXILLARY-SWEAT-GLANDS",
        ],
        "age_of_onset": "12-17 yr (teenage); previously normal development; first seizure often GTCS; myoclonus follows within months",
        "key_biomarker": "Skin biopsy (axillary) — Lafora bodies PAS+; EEG: occipital spikes; rapid cognitive decline on neuropsychometry",
        "seed_offset": 1,
    },

    # -- NHLRC1 — Lafora Disease Type 2 (Malin) ------------------------------------
    {
        "gene": "NHLRC1",
        "alt_name": (
            "NHLRC1 (NHLRC1-395aa-6p22.3 / AR — Lafora-Disease-Type-2-Malin-E3-Ubiquitin-Ligase — "
            "Clinically-Identical-EPM2A-Skin-Biopsy-Same-PATHOGNOMONIC — "
            "Mediterranean-South-Asian-Founder — Metformin-ASO-Trial-Same-Pathway)"
        ),
        "protein": (
            "NHLRC1 -- 6p22.3 AR -- NHLRC1-395aa -- "
            "Malin-NHL-Repeat-Containing-RING-Finger-E3-Ubiquitin-Ligase -- "
            "Interacts-With-Laforin-EPM2A-Partner-Protein-Complex -- "
            "Ubiquitinates-Glycogen-Regulatory-Proteins-PTG-GS-Prevents-Lafora-Body -- "
            "Lafora-Disease-Type-2-Same-Polyglucosan-Accumulation-As-EPM2A -- "
            "50pct-Lafora-Disease-Cases-NHLRC1-50pct-EPM2A -- "
            "Mediterranean-South-Asian-Middle-Eastern-Consanguineous-Founder -- "
            "Skin-Biopsy-PAS-Positive-Lafora-Bodies-Same-As-EPM2A-Indistinguishable"
        ),
        "locus": "6p22.3",
        "protein_size": "395 aa",
        "inheritance": (
            "AR (autosomal recessive) — biallelic NHLRC1 loss-of-function (Malin); "
            "~50% of all Lafora disease; EPM2A accounts for other 50%; "
            "Mediterranean, South Asian, Middle Eastern founder variants (consanguinity enriched); "
            "Point mutations predominate; RING finger domain mutations most severe; "
            "Phenotype: identical to EPM2A Lafora disease; distinguish only by genotyping; "
            "Genotype-phenotype: very limited data — not clinically predictive; "
            "Sibling recurrence risk 25% as expected for AR"
        ),
        "pathognomonic": (
            "LAFORA BODIES on skin biopsy (axillary) — PATHOGNOMONIC (PAS+, diastase-resistant) — "
            "IDENTICAL to EPM2A; "
            "Occipital seizures with visual aura (same as EPM2A); "
            "Action myoclonus — stimulus-sensitive; "
            "RAPID cognitive decline → dementia within 2-3 yr; "
            "GTCS + drop attacks; "
            "Death within 10-15 yr; "
            "Possibly slightly slower progression in some series vs EPM2A — not reliable clinically"
        ),
        "treatment": (
            "**Identical to EPM2A Lafora disease**: "
            "**Valproate** + **Levetiracetam** first-line; "
            "**Clonazepam, Zonisamide, Perampanel** adjunctive; "
            "**ABSOLUTE CONTRAINDICATED**: CBZ, OXC, PHT, LTG, VGB; "
            "**Metformin** — same rationale as EPM2A (GYS1/mTOR pathway); "
            "**ASO targeting GYS1** — trials include both EPM2A and NHLRC1 Lafora; "
            "**Genetic counselling**: cascade testing; 25% sibling risk; "
            "**Palliative care** essential given fatal prognosis"
        ),
        "critical_flags": [
            "LAFORA-BODIES-SKIN-BIOPSY-PATHOGNOMONIC-SAME-AS-EPM2A",
            "NHLRC1-vs-EPM2A-CLINICALLY-INDISTINGUISHABLE-GENE-ONLY",
            "RAPID-FATAL-PROGRESSION-DEATH-10-15yr",
            "CBZ-OXC-PHT-LTG-ABSOLUTE-CONTRAINDICATED",
            "METFORMIN-ASO-GYS1-TRIAL-SAME-PATHWAY",
            "MEDITERRANEAN-SOUTH-ASIAN-CONSANGUINITY-ENRICHED",
            "RING-FINGER-DOMAIN-MUTATIONS-MOST-SEVERE",
            "50pct-LAFORA-CASES-NHLRC1-50pct-EPM2A",
        ],
        "age_of_onset": "12-17 yr (same as EPM2A); previously normal development; indistinguishable clinical onset from EPM2A",
        "key_biomarker": "Skin biopsy (axillary) PAS+ Lafora bodies; NHLRC1 gene sequencing to distinguish from EPM2A",
        "seed_offset": 2,
    },

    # -- SCARB2 — Action Myoclonus-Renal Failure (EPM4) ----------------------------
    {
        "gene": "SCARB2",
        "alt_name": (
            "SCARB2 (SCARB2-478aa-4q21.1 / AR — EPM4-Action-Myoclonus-Renal-Failure-AMRF — "
            "Progressive-Nephropathy-PATHOGNOMONIC-Co-Feature — "
            "Hearing-Loss — AVOID-NSAIDs-Nephrotoxins-Mandatory)"
        ),
        "protein": (
            "SCARB2 -- 4q21.1 AR -- SCARB2-478aa -- "
            "Scavenger-Receptor-Class-B-Member-2-LIMP2-Lysosomal-Integral-Membrane-Protein-2 -- "
            "CD36-Family-Type-III-Transmembrane-Receptor-Lysosomal-Membrane -- "
            "GBA-Glucocerebrosidase-Trafficking-To-Lysosomes-Required -- "
            "Loss-SCARB2-Impaired-GBA-Lysosomal-Delivery-Membrane-Instability -- "
            "EPM4-Action-Myoclonus-Renal-Failure-Syndrome-AMRF -- "
            "Progressive-Nephropathy-Proteinuria-Renal-Failure-Pathognomonic-Co-Feature -- "
            "Sensorineural-Hearing-Loss-Progressive -- "
            "No-Lafora-Bodies-Normal-Skin-Biopsy-DISTINGUISH-EPM2A-NHLRC1"
        ),
        "locus": "4q21.1",
        "protein_size": "478 aa",
        "inheritance": (
            "AR (autosomal recessive) — biallelic SCARB2 loss-of-function; "
            "Rare globally; consanguinity enriched; "
            "Point mutations, small indels; "
            "Onset adolescence-early adulthood (12-30 yr); "
            "Progressive course — renal failure often dominates prognosis; "
            "Heterozygous carriers of SCARB2 variants: increased Parkinson disease risk (SCARB2 is a PD risk gene); "
            "Genetic counselling: 25% sibling recurrence"
        ),
        "pathognomonic": (
            "PROGRESSIVE NEPHROPATHY — proteinuria → renal failure — PATHOGNOMONIC co-feature; "
            "Action myoclonus (cortical — Giant SEPs); "
            "GTCS; cerebellar ataxia; "
            "SENSORINEURAL HEARING LOSS — progressive, bilateral; "
            "NORMAL skin biopsy (no Lafora bodies — distinguishes from EPM2A/NHLRC1); "
            "Onset 12-30 yr; "
            "Trehalase enzyme deficiency in some patients (urine trehalase); "
            "Face/head sparing relative to limbs (cortical myoclonus pattern)"
        ),
        "treatment": (
            "**Levetiracetam** — first-line (effective for action myoclonus + GTCS); "
            "**Clonazepam** — adjunctive myoclonus; "
            "**Valproate** — GTCS (caution: POLG1 screen, avoid if renal impairment); "
            "**Piracetam** — adjunctive for action myoclonus; "
            "**ABSOLUTE CONTRAINDICATED**: CBZ, OXC, PHT, LTG — worsen myoclonus; "
            "**RENAL PROTECTION**: AVOID NSAIDs, aminoglycosides, contrast media, nephrotoxins; "
            "**Nephrology co-management**: ACE inhibitor for proteinuria; renal function monitoring 6-monthly; "
            "**Audiology**: bilateral hearing aids; cochlear implant evaluation; "
            "**Dialysis/renal transplant** — if end-stage renal disease develops"
        ),
        "critical_flags": [
            "PROGRESSIVE-NEPHROPATHY-RENAL-FAILURE-PATHOGNOMONIC-CO-FEATURE",
            "AVOID-NSAIDs-NEPHROTOXINS-AMINOGLYCOSIDES-MANDATORY",
            "NORMAL-SKIN-BIOPSY-DISTINGUISHES-FROM-LAFORA",
            "SENSORINEURAL-HEARING-LOSS-AUDIOLOGICAL-SCREEN",
            "NEPHROLOGY-CO-MANAGEMENT-MANDATORY",
            "CBZ-OXC-PHT-LTG-ABSOLUTE-CONTRAINDICATED",
            "RENAL-FUNCTION-6-MONTHLY-MONITORING",
            "GBA-LYSOSOMAL-TRAFFICKING-SCARB2-LIMP2",
        ],
        "age_of_onset": "12-30 yr (adolescence to young adult); myoclonus or GTCS first; renal involvement may lag by years",
        "key_biomarker": "Urinalysis (proteinuria); serum creatinine/eGFR; audiogram; SCARB2 sequencing; Giant SEPs",
        "seed_offset": 3,
    },

    # -- GOSR2 — Nord Disease / EPM6 -----------------------------------------------
    {
        "gene": "GOSR2",
        "alt_name": (
            "GOSR2 (GOSR2-235aa-17q21.32 / AR — EPM6-GOSA-Nord-Disease — "
            "Scoliosis-100pct-PATHOGNOMONIC-Feature — Elevated-CK — "
            "Early-Onset-2-6yr — Predominantly-Northern-European)"
        ),
        "protein": (
            "GOSR2 -- 17q21.32 AR -- GOSR2-235aa -- "
            "Golgi-SNAP-Receptor-Complex-Member-2-Membrin-Golgi-SNARE-Protein -- "
            "Intra-Golgi-Vesicle-Fusion-cis-trans-Golgi-Trafficking -- "
            "Loss-GOSR2-Impaired-Golgi-Function-Glycoprotein-Processing -- "
            "EPM6-GOSA-Nord-Disease-PME-With-Scoliosis -- "
            "Scoliosis-100pct-Earliest-Feature-Often-PATHOGNOMONIC -- "
            "Raised-CK-2-5x-Upper-Limit-Normal -- "
            "Early-Onset-2-6yr-Distinguishes-From-CSTB-EPM2A -- "
            "c.430G>T-p.Gly144Trp-Founder-Variant-Northern-European"
        ),
        "locus": "17q21.32",
        "protein_size": "235 aa",
        "inheritance": (
            "AR (autosomal recessive) — biallelic GOSR2; "
            "Rare; predominantly Northern European (Norwegian, Dutch, British) founder variant c.430G>T (p.Gly144Trp); "
            "Onset 2-6 yr — earlier than most PME disorders; "
            "Consanguinity less prominent (founder effect); "
            "Sibling recurrence 25%; "
            "Males and females equally affected"
        ),
        "pathognomonic": (
            "SCOLIOSIS in >95% — often the FIRST presenting feature (before seizures) — PATHOGNOMONIC; "
            "Elevated SERUM CK (2-5× ULN) — pathognomonic biochemical marker; "
            "Early onset myoclonus + ataxia 2-6 yr; "
            "GTCS + drop attacks; "
            "Hearing loss (sensorineural) in some; "
            "Facial nerve dysfunction (diplopia) reported; "
            "Slower progression than Lafora; "
            "No Lafora bodies (normal skin biopsy)"
        ),
        "treatment": (
            "**Valproate** — first-line (GTCS + myoclonus); "
            "**Levetiracetam** — adjunctive; "
            "**Piracetam** — action myoclonus; "
            "**Clonazepam** — adjunctive; "
            "**ABSOLUTE CONTRAINDICATED**: CBZ, OXC, PHT, LTG; "
            "**Orthopaedics**: scoliosis bracing / spinal fusion (early referral mandatory); "
            "**Physiotherapy**: core stability, posture, gait; "
            "**Audiology**: hearing screen, aids if needed; "
            "**CK monitoring**: elevated CK not myopathy — reassure but monitor cardiac"
        ),
        "critical_flags": [
            "SCOLIOSIS-100pct-EARLIEST-FEATURE-PATHOGNOMONIC",
            "ELEVATED-CK-PATHOGNOMONIC-NOT-MYOPATHY",
            "EARLY-ONSET-2-6yr-DISTINGUISHES-FROM-OTHER-PME",
            "ORTHOPAEDICS-SCOLIOSIS-REFERRAL-MANDATORY",
            "CBZ-OXC-PHT-LTG-ABSOLUTE-CONTRAINDICATED",
            "NORTHERN-EUROPEAN-c430G>T-p.Gly144Trp-FOUNDER",
            "NORMAL-SKIN-BIOPSY-NO-LAFORA-BODIES",
            "HEARING-SCREEN-MANDATORY",
        ],
        "age_of_onset": "2-6 yr (early childhood); scoliosis often first (before seizures); seizures 5-10 yr",
        "key_biomarker": "Serum CK (elevated 2-5× ULN); spinal X-ray (scoliosis); GOSR2 sequencing; Northern European ancestry",
        "seed_offset": 4,
    },

    # -- KCNC1 — Progressive Myoclonic Epilepsy 7 (EPM7) ----------------------------
    {
        "gene": "KCNC1",
        "alt_name": (
            "KCNC1 (KCNC1-585aa-11p15.1 / AD — EPM7-Kv3.1-Potassium-Channel — "
            "R320H-Dominant-Negative-Founder-Finnish-Baltic — "
            "Giant-SEPs-PATHOGNOMONIC — "
            "CBZ-OXC-PHT-LTG-ABSOLUTE-CONTRAINDICATED-NaV1.1-Interneuron)"
        ),
        "protein": (
            "KCNC1 -- 11p15.1 AD -- KCNC1-585aa -- "
            "Kv3.1-Shaw-Type-Voltage-Gated-Potassium-Channel-High-Threshold-K+ -- "
            "6-Transmembrane-Domain-Tetramer-Fast-Repolarisation-High-Frequency-Firing -- "
            "Fast-Spiking-Interneurons-PV+ -- "
            "EPM7-p.Arg320His-R320H-Dominant-Negative-Gain-of-Dysfunction -- "
            "Finnish-Baltic-Founder-Variant-c.959G>A -- "
            "Giant-Somatosensory-Evoked-Potentials-SEPs-Pathognomonic -- "
            "Photosensitivity-75pct -- "
            "KCNC1-R320H-Slows-Kv3.1-Kinetics-Reduces-Interneuron-Repolarisation -- "
            "Secondary-NaV1.1-Interneuron-Disinhibition-Explains-CBZ-PHT-Contraindication"
        ),
        "locus": "11p15.1",
        "protein_size": "585 aa",
        "inheritance": (
            "AD (autosomal dominant) — dominant-negative (R320H slows Kv3.1 channel kinetics); "
            "c.959G>A p.Arg320His — founder variant in Finnish + Baltic populations; "
            "De novo cases (non-Finnish) reported with other KCNC1 variants; "
            "Penetrance near-complete for R320H; "
            "Onset 13-18 yr (similar to CSTB); "
            "Progressive course — gait aids often needed by 30s-40s; "
            "Not fatal; cognitive largely preserved"
        ),
        "pathognomonic": (
            "GIANT SOMATOSENSORY EVOKED POTENTIALS (Giant SEPs, N20-P25 >4 µV) — PATHOGNOMONIC; "
            "Jerk-locked back-averaging (cortical myoclonus) on neurophysiology; "
            "Action myoclonus — severe, intention-worsened; "
            "Photosensitivity (PPR) 75%; "
            "GTCS; progressive cerebellar ataxia; "
            "Cognitive largely preserved (mild executive dysfunction only); "
            "Course: progressive worsening over decades"
        ),
        "treatment": (
            "**Piracetam** — Level C action myoclonus; "
            "**Levetiracetam** — effective; "
            "**Valproate** — GTCS (POLG1 screen mandatory before VPA); "
            "**Clonazepam** — adjunctive myoclonus; "
            "**Zonisamide** — adjunctive; "
            "**Perampanel** — emerging evidence; "
            "**ABSOLUTE CONTRAINDICATED**: CBZ, OXC, PHT, LTG — "
            "block NaV1.1 → further interneuron disinhibition → catastrophic worsening; "
            "**GBP/PGB** HIGH RISK — avoid; "
            "**VGB** — avoid (worsen myoclonus); "
            "**Physiotherapy**: gait aids, balance; "
            "**Finnish genetic testing**: R320H targeted first"
        ),
        "critical_flags": [
            "GIANT-SEPs-N20-P25->4uV-PATHOGNOMONIC",
            "R320H-DOMINANT-NEGATIVE-FINNISH-BALTIC-FOUNDER",
            "CBZ-OXC-PHT-LTG-ABSOLUTE-CONTRAINDICATED-NaV1.1",
            "GBP-PGB-HIGH-RISK-AVOID",
            "POLG1-MANDATORY-SCREEN-BEFORE-VPA",
            "PHOTOSENSITIVITY-75pct",
            "JERK-LOCKED-BACK-AVERAGING-CORTICAL-MYOCLONUS",
            "COGNITIVE-LARGELY-PRESERVED-CONTRAST-LAFORA",
        ],
        "age_of_onset": "13-18 yr; action myoclonus often first (falls); GTCS follows; cognitive intact",
        "key_biomarker": "Giant SEPs (N20-P25 >4 µV); C-reflex; PPR; KCNC1 sequencing (R320H first in Finnish/Baltic)",
        "seed_offset": 5,
    },

    # -- PRICKLE1 — EPM1B (Unverricht-Lundborg-like) --------------------------------
    {
        "gene": "PRICKLE1",
        "alt_name": (
            "PRICKLE1 (PRICKLE1-831aa-12q12 / AR — EPM1B-Unverricht-Lundborg-Like — "
            "Action-Myoclonus-GTCS-Cerebellar — "
            "Slower-Progression-Than-Lafora-Better-Prognosis — "
            "Piracetam-Levetiracetam)"
        ),
        "protein": (
            "PRICKLE1 -- 12q12 AR -- PRICKLE1-831aa -- "
            "Prickle-Like-Protein-1-Spindle-Pole-Body-Component-Vertebrate -- "
            "PET-Lin7-LIM-Domain-Containing-Nuclear-Protein -- "
            "Planar-Cell-Polarity-PCP-Pathway-Wnt-Non-Canonical -- "
            "Regulates-Neurite-Outgrowth-Neuronal-Migration -- "
            "EPM1B-ULD-Like-Progressive-Myoclonic-Epilepsy -- "
            "Biallelic-Loss-of-Function-AR -- "
            "Clinically-Similar-to-CSTB-But-Distinct-Gene -- "
            "No-Dodecamer-Repeat-Gene-Sequencing-Required"
        ),
        "locus": "12q12",
        "protein_size": "831 aa",
        "inheritance": (
            "AR (autosomal recessive) — biallelic PRICKLE1 loss-of-function; "
            "Rare — few hundred cases reported worldwide; "
            "No dominant founder population; consanguinity enriched; "
            "Onset 5-16 yr (similar to CSTB but can be earlier); "
            "Phenotype: clinically similar to ULD (CSTB) but distinct; "
            "Progression: generally slower than Lafora; better prognosis than EPM2A/NHLRC1; "
            "Not fatal in reported series"
        ),
        "pathognomonic": (
            "Action myoclonus — cortical, stimulus-sensitive; "
            "GTCS; "
            "Progressive cerebellar ataxia (generally milder than ULD); "
            "EEG: cortical myoclonus pattern (spike-wave); "
            "Giant SEPs in some; "
            "Normal skin biopsy (no Lafora bodies — distinguishes from EPM2A/NHLRC1); "
            "Slower cognitive decline compared to Lafora; "
            "No specific PATHOGNOMONIC molecular test (sequencing required)"
        ),
        "treatment": (
            "**Piracetam** — action myoclonus (by analogy with ULD); "
            "**Levetiracetam** — first-line; "
            "**Valproate** — GTCS; "
            "**Clonazepam** — adjunctive; "
            "**ABSOLUTE CONTRAINDICATED**: CBZ, OXC, PHT, LTG — worsen myoclonus; "
            "**Rehabilitation**: physiotherapy, occupational therapy (similar to CSTB); "
            "**Genetic counselling**: 25% sibling risk; cascade testing"
        ),
        "critical_flags": [
            "NORMAL-SKIN-BIOPSY-DISTINGUISHES-FROM-LAFORA",
            "CBZ-OXC-PHT-LTG-ABSOLUTE-CONTRAINDICATED",
            "SLOWER-PROGRESSION-BETTER-PROGNOSIS-THAN-LAFORA",
            "CLINICALLY-SIMILAR-CSTB-BUT-DIFFERENT-GENE",
            "PCP-WNT-PATHWAY-DISTINCT-MECHANISM",
            "PIRACETAM-LEVETIRACETAM-FIRST-LINE",
            "NO-SPECIFIC-PATHOGNOMONIC-FEATURE-SEQUENCING-REQUIRED",
            "25pct-SIBLING-RECURRENCE-AR",
        ],
        "age_of_onset": "5-16 yr (childhood to adolescence); myoclonus + GTCS; slower progression than Lafora",
        "key_biomarker": "PRICKLE1 sequencing; normal skin biopsy; Giant SEPs (variable); EEG cortical myoclonus",
        "seed_offset": 6,
    },

    # -- KCTD7 — Progressive Myoclonic Epilepsy 3 (EPM3) ----------------------------
    {
        "gene": "KCTD7",
        "alt_name": (
            "KCTD7 (KCTD7-289aa-12q14.2 / AR — EPM3-Infantile-Onset-Severe — "
            "Onset-1-2yr-Severe-ID-Myoclonus-Ataxia — "
            "EARLIEST-PME-ONSET-Distinguishes-From-All-Others — "
            "NCL-Like-Phenotype-Distinguish-By-Gene)"
        ),
        "protein": (
            "KCTD7 -- 12q14.2 AR -- KCTD7-289aa -- "
            "Potassium-Channel-Tetramerisation-Domain-Containing-7-BTB-POZ-Domain -- "
            "Cullin-3-E3-Ubiquitin-Ligase-Substrate-Adaptor -- "
            "Interacts-With-CUL3-RING-E3-Ligase-Complex -- "
            "Neuronal-Specific-Expression-Ubiquitin-Proteasome-Pathway -- "
            "EPM3-Severe-Early-Infantile-PME -- "
            "Onset-1-2yr-Before-Walking-Consolidates -- "
            "Severe-Intellectual-Disability-Refractory-Seizures -- "
            "NCL-Like-EM-Pattern-Electron-Microscopy -- "
            "Distinguish-CLN2-TPP1-By-Enzyme-Assay-Gene"
        ),
        "locus": "12q14.2",
        "protein_size": "289 aa",
        "inheritance": (
            "AR (autosomal recessive) — biallelic KCTD7 loss-of-function; "
            "Rare; consanguinity enriched; "
            "Onset 1-2 yr — EARLIEST onset of all PME disorders (distinguishing feature); "
            "Severe course — non-ambulant by 5-10 yr in most; "
            "Resembles neuronal ceroid lipofuscinosis (NCL) clinically; "
            "Must distinguish from CLN2 (TPP1 enzyme deficient) by enzyme assay/sequencing; "
            "Sibling recurrence 25%"
        ),
        "pathognomonic": (
            "INFANTILE ONSET 1-2 yr — EARLIEST of all PME disorders — PATHOGNOMONIC timing; "
            "Severe myoclonus + GTCS + ataxia from first year of life; "
            "SEVERE intellectual disability (profound — pre-verbal at 5 yr+); "
            "Hypotonia → spasticity transition; "
            "NCL-like: ERG abnormalities, visual failure in some; "
            "Electron microscopy on skin/conjunctival biopsy: fingerprint/curvilinear profiles (NCL-like) in some; "
            "LYSOSOMAL ENZYME assays (CLN2/TPP1, CLN1/PPT1) normal — distinguishes from NCL; "
            "KCTD7 sequencing confirms"
        ),
        "treatment": (
            "**Valproate** — first-line (infantile seizures); "
            "**Levetiracetam** — adjunctive; "
            "**Clonazepam** — myoclonus management; "
            "**ABSOLUTE CONTRAINDICATED**: CBZ, OXC, PHT, LTG; "
            "**GBP** — HIGH RISK worsen myoclonus; "
            "**Vigabatrin** — CONTRAINDICATED; "
            "**Multidisciplinary**: paediatric neurology, developmental medicine, physiotherapy, OT, SLT; "
            "**Nutritional support**: nasogastric/PEG if swallowing unsafe; "
            "**NCL workup first**: enzyme assays (TPP1/PPT1) to exclude CLN2/CLN1 before KCTD7 diagnosis; "
            "**Palliative care input** early given severity"
        ),
        "critical_flags": [
            "EARLIEST-ONSET-1-2yr-PATHOGNOMONIC-DISTINGUISHES-ALL-PME",
            "SEVERE-INTELLECTUAL-DISABILITY-PROFOUND",
            "NCL-WORKUP-MANDATORY-TPP1-PPT1-ENZYME-ASSAY-FIRST",
            "DISTINGUISH-CLN2-CLN1-BY-ENZYME-GENE",
            "CBZ-OXC-PHT-LTG-ABSOLUTE-CONTRAINDICATED",
            "GBP-HIGH-RISK-WORSEN-MYOCLONUS",
            "PEG-NASOGASTRIC-FEEDING-SWALLOWING-SAFETY",
            "PALLIATIVE-CARE-EARLY-SEVERE-COURSE",
        ],
        "age_of_onset": "1-2 yr (infantile); earliest PME; severe course; non-ambulant often by 5-10 yr",
        "key_biomarker": "Lysosomal enzyme assays (TPP1, PPT1) to exclude NCL; KCTD7 sequencing; EEG cortical myoclonus; biopsy EM (NCL-like profiles in some)",
        "seed_offset": 7,
    },
]


def _generate_cohort():
    """Generate 8 × 40-patient cohort (320 total) with realistic clinical parameters."""
    all_patients = []
    for gene_data in MYOCLONUS_GENES:
        gene = gene_data["gene"]
        rng = random.Random(SEED_BASE + gene_data["seed_offset"])
        for i in range(40):
            pid = f"{gene}-{SEED_BASE + gene_data['seed_offset']}-{i:03d}"

            if gene == "CSTB":
                repeat_copies = rng.randint(31, 80)
                photosensitive = rng.random() < 0.82
                piracetam = rng.random() < 0.75
                giant_seps = rng.random() < 0.78
                cerebellar_ataxia = rng.random() < 0.85
                age_onset = rng.randint(6, 16)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "repeat_copies": repeat_copies,
                    "photosensitive": photosensitive,
                    "piracetam": piracetam,
                    "giant_seps": giant_seps,
                    "cerebellar_ataxia": cerebellar_ataxia,
                    "age_onset": age_onset,
                })

            elif gene == "EPM2A":
                skin_biopsy_pos = rng.random() < 0.95
                occipital_seizures = rng.random() < 0.72
                rapid_decline = rng.random() < 0.90
                metformin_trial = rng.random() < 0.30
                age_onset = rng.randint(12, 17)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "skin_biopsy_lafora_positive": skin_biopsy_pos,
                    "occipital_seizures": occipital_seizures,
                    "rapid_cognitive_decline": rapid_decline,
                    "metformin_trial": metformin_trial,
                    "age_onset": age_onset,
                })

            elif gene == "NHLRC1":
                skin_biopsy_pos = rng.random() < 0.95
                occipital_seizures = rng.random() < 0.70
                rapid_decline = rng.random() < 0.88
                mediterranean_ancestry = rng.random() < 0.72
                age_onset = rng.randint(12, 17)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "skin_biopsy_lafora_positive": skin_biopsy_pos,
                    "occipital_seizures": occipital_seizures,
                    "rapid_cognitive_decline": rapid_decline,
                    "mediterranean_ancestry": mediterranean_ancestry,
                    "age_onset": age_onset,
                })

            elif gene == "SCARB2":
                renal_failure = rng.random() < 0.75
                hearing_loss = rng.random() < 0.68
                proteinuria = rng.random() < 0.90
                normal_skin_biopsy = True
                age_onset = rng.randint(12, 30)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "renal_failure": renal_failure,
                    "hearing_loss": hearing_loss,
                    "proteinuria": proteinuria,
                    "normal_skin_biopsy": normal_skin_biopsy,
                    "age_onset": age_onset,
                })

            elif gene == "GOSR2":
                scoliosis = rng.random() < 0.97
                elevated_ck = rng.random() < 0.88
                hearing_loss = rng.random() < 0.45
                northern_european = rng.random() < 0.78
                age_onset = rng.randint(2, 6)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "scoliosis": scoliosis,
                    "elevated_ck": elevated_ck,
                    "hearing_loss": hearing_loss,
                    "northern_european": northern_european,
                    "age_onset": age_onset,
                })

            elif gene == "KCNC1":
                giant_seps = rng.random() < 0.88
                photosensitive = rng.random() < 0.76
                r320h_variant = rng.random() < 0.72
                cognitive_intact = rng.random() < 0.82
                age_onset = rng.randint(13, 18)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "giant_seps": giant_seps,
                    "photosensitive": photosensitive,
                    "r320h_variant": r320h_variant,
                    "cognitive_intact": cognitive_intact,
                    "age_onset": age_onset,
                })

            elif gene == "PRICKLE1":
                normal_skin_biopsy = True
                cerebellar_ataxia = rng.random() < 0.78
                slower_progression = rng.random() < 0.80
                age_onset = rng.randint(5, 16)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "normal_skin_biopsy": normal_skin_biopsy,
                    "cerebellar_ataxia": cerebellar_ataxia,
                    "slower_progression": slower_progression,
                    "age_onset": age_onset,
                })

            elif gene == "KCTD7":
                severe_id = rng.random() < 0.90
                ncl_like_em = rng.random() < 0.52
                lysosomal_enzymes_normal = True
                peg_feeding = rng.random() < 0.55
                age_onset = rng.randint(1, 2)
                all_patients.append({
                    "patient_id": pid, "gene": gene,
                    "severe_id": severe_id,
                    "ncl_like_em": ncl_like_em,
                    "lysosomal_enzymes_normal": lysosomal_enzymes_normal,
                    "peg_feeding": peg_feeding,
                    "age_onset": age_onset,
                })

    return all_patients


def overview():
    """Return atlas-level aggregate overview."""
    patients = _generate_cohort()
    lafora_pos = sum(1 for p in patients if p.get("skin_biopsy_lafora_positive"))
    occipital = sum(1 for p in patients if p.get("occipital_seizures"))
    renal_fail = sum(1 for p in patients if p.get("renal_failure"))
    scoliosis = sum(1 for p in patients if p.get("scoliosis"))
    giant_seps = sum(1 for p in patients if p.get("giant_seps"))
    piracetam = sum(1 for p in patients if p.get("piracetam"))
    photosensitive = sum(1 for p in patients if p.get("photosensitive"))
    severe_id = sum(1 for p in patients if p.get("severe_id"))
    rapid_decline = sum(1 for p in patients if p.get("rapid_cognitive_decline"))
    elevated_ck = sum(1 for p in patients if p.get("elevated_ck"))
    return {
        "atlas": "Hereditary-Myoclonus-Atlas",
        "genes": [g["gene"] for g in MYOCLONUS_GENES],
        "total_patients": len(patients),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "lafora_bodies_skin_biopsy_patients": lafora_pos,
        "occipital_seizures_lafora_patients": occipital,
        "renal_failure_scarb2_patients": renal_fail,
        "scoliosis_gosr2_patients": scoliosis,
        "giant_seps_patients": giant_seps,
        "piracetam_patients": piracetam,
        "photosensitive_patients": photosensitive,
        "severe_id_kctd7_patients": severe_id,
        "rapid_cognitive_decline_lafora_patients": rapid_decline,
        "elevated_ck_gosr2_patients": elevated_ck,
    }


def breakdown():
    """Return per-gene breakdown with clinical details."""
    patients = _generate_cohort()
    result = {}
    for gene_data in MYOCLONUS_GENES:
        gene = gene_data["gene"]
        gene_patients = [p for p in patients if p["gene"] == gene]
        result[gene] = {
            "gene": gene,
            "alt_name": gene_data["alt_name"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "patient_count": len(gene_patients),
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "critical_flags": gene_data["critical_flags"],
            "age_of_onset": gene_data["age_of_onset"],
            "key_biomarker": gene_data["key_biomarker"],
        }
    return result


def definitions():
    """Return gene definitions, glossary and treatment protocols."""
    return {
        "genes": {g["gene"]: g["protein"] for g in MYOCLONUS_GENES},
        "glossary": {
            "Progressive Myoclonic Epilepsy (PME)": "Heterogeneous group of rare genetic disorders combining: (1) progressive myoclonic seizures, (2) other seizure types (GTCS ± focal), (3) progressive neurological decline (ataxia ± dementia ± other); caused by specific gene mutations; all share the core triad; key: differentiate by onset age, skin biopsy, renal function, neurophysiology, ancestry",
            "Action Myoclonus": "Myoclonus (sudden involuntary muscle jerks) triggered or worsened by voluntary movement; cortical origin (giant SEPs, C-reflex on neurophysiology); stimulus-sensitive: touch, light, sound; hallmark of all PME; piracetam specifically reduces action myoclonus; distinguish from rest/rhythmic myoclonus",
            "Lafora Bodies": "Malformed polyglucosan (abnormal glycogen) inclusions in neurons, liver, muscle, sweat gland ducts; PAS-positive, diastase-resistant; PATHOGNOMONIC for Lafora disease (EPM2A, NHLRC1); detected on skin biopsy (axillary sweat glands — use fresh, not formalin-fixed for EM); caused by impaired glycogen phosphatase (laforin) or ubiquitin ligase (malin) function",
            "Giant Somatosensory Evoked Potentials (Giant SEPs)": "Pathologically enlarged cortical SEPs (N20-P25 >4 µV amplitude on standard recording); reflect cortical hyperexcitability / enhanced cortical response to peripheral stimulation; pathognomonic in CSTB (ULD) and KCNC1 (EPM7); used with C-reflex to confirm cortical myoclonus source; jerk-locked back-averaging confirms seizure origin",
            "C-Reflex (Long-Latency Reflex)": "Pathologically enhanced long-latency reflex — cortical component >15 µV; recorded with EMG + EEG simultaneously; confirms cortical origin of myoclonus; found in PME (CSTB, KCNC1, EPM7); distinguishes cortical from subcortical/spinal myoclonus; used to monitor treatment response (piracetam reduces C-reflex amplitude)",
            "Dodecamer Repeat Expansion (CSTB/ULD)": "12-mer repeat unit (5'-CCCCGCCCCGCG-3') in CSTB promoter; normal 2-3 copies; pathogenic >30 copies; causes ULD (EPM1); detected by Southern blot or long-read PCR; standard PCR may not amplify expanded alleles — request specific assay; >90% of CSTB alleles are repeat expansions not point mutations",
            "Piracetam (Action Myoclonus)": "Nootropic agent; Level A evidence specifically for action myoclonus in ULD (CSTB/EPM1); mechanism uncertain (modulates neuronal membrane, AMPA receptor); 2.4-4.8 g/day; well-tolerated; Level C in KCNC1 (EPM7); does NOT apply to Lafora (EPM2A/NHLRC1) where no Level A evidence; not anticonvulsant for GTCS",
            "CBZ/OXC/PHT/LTG Contraindication in PME": "All sodium channel blockers worsen myoclonus in PME — ABSOLUTE CONTRAINDICATION; mechanism: sodium channel blockade disinhibits NaV1.1-expressing parvalbumin+ fast-spiking interneurons → net cortical hyperexcitability → dramatic myoclonus worsening; this applies to ALL PME subtypes (CSTB, EPM2A, NHLRC1, SCARB2, GOSR2, KCNC1, PRICKLE1, KCTD7); lamotrigine (LTG) also worsens myoclonus — AVOID in all PME",
            "Action Myoclonus-Renal Failure Syndrome (AMRF/EPM4)": "SCARB2 AR disorder combining PME + progressive nephropathy (proteinuria → renal failure); SCARB2 = LIMP2, required for lysosomal glucocerebrosidase (GBA) trafficking; renal failure can precede or follow neurological symptoms; nephrologist co-management mandatory; NSAIDs absolutely contraindicated",
            "Nord Disease (EPM6/GOSR2)": "GOSR2 AR Golgi SNARE protein disorder; hallmark: SCOLIOSIS in 100% — often first feature before seizures; early onset 2-6 yr; elevated CK (not myopathy); northern European founder (c.430G>T p.Gly144Trp); spinal surgery may be needed; distinct from other PME by age + scoliosis combination",
            "Kv3.1/KCNC1 (EPM7)": "Shaw-family high-threshold K+ channel mediating fast repolarisation in fast-spiking interneurons; R320H dominant-negative slows Kv3.1 → impaired interneuron repolarisation → circuit hyperexcitability; Finnish-Baltic founder; giant SEPs pathognomonic; CBZ/PHT ABSOLUTE CI (secondary NaV1.1 disinhibition compounds deficit)",
            "Metformin in Lafora Disease": "AMPK activator + mTOR inhibitor → reduces glycogen synthase activity (GYS1) → less malformed glycogen → fewer Lafora bodies; preclinical and early clinical evidence in EPM2A + NHLRC1; clinical trial ongoing (REIN-Lafora, NCT04609358); not standard of care yet but widely used compassionately; monitoring: lactic acidosis, B12 depletion",
        },
        "surveillance_protocols": {
            "CSTB (Unverricht-Lundborg)": "Dodecamer repeat PCR (specific assay — standard PCR fails); Giant SEPs + C-reflex at diagnosis and annually; EEG (cortical myoclonus, PPR); piracetam 2.4-4.8 g/day (titrate); neuropsychological assessment (mild impairment expected); physiotherapy (gait/balance); AVOID CBZ/OXC/PHT/LTG/GBP; photosensitivity precautions; NOT fatal — communicate prognosis clearly; ULD registry (contact EPMA)",
            "EPM2A (Lafora Type 1)": "Skin biopsy (axillary, fresh/snap-frozen — NOT formalin for EM): PAS stain + EM; EEG (occipital spike-wave); neuropsychological battery (rapid decline); MRI (progressive cortical atrophy late); valproate + levetiracetam; metformin (compassionate — monitor lactic acid, B12); ASO trial eligibility check; AVOID CBZ/OXC/PHT/LTG; palliative care early; Lafora Disease Research Foundation contact",
            "NHLRC1 (Lafora Type 2)": "Same protocol as EPM2A; genetic panel includes both EPM2A and NHLRC1; skin biopsy identical; prognosis similarly fatal; metformin same rationale; ASO trial check; consanguinity counselling; Mediterranean/South Asian population: NHLRC1 pre-test probability higher",
            "SCARB2 (AMRF/EPM4)": "Urinalysis (proteinuria — monthly initially); serum creatinine/eGFR (6-monthly); ACE inhibitor for proteinuria (nephrology); audiogram (bilateral SNHL); ABSOLUTELY AVOID NSAIDs, aminoglycosides, IV contrast, nephrotoxins; levetiracetam + clonazepam (renal-dose adjust); vascular access planning (dialysis); renal transplant evaluation if ESRD; SCARB2 sequencing",
            "GOSR2 (EPM6/Nord)": "Spinal X-ray (scoliosis — baseline + 6-monthly until skeletal maturity); orthopaedic referral at diagnosis; CK serum (monitor, not myopathy); valproate + piracetam; EEG; hearing screen (audiogram); physiotherapy (posture, balance); GOSR2 c.430G>T targeted first (Northern European); scoliosis brace or surgery referral",
            "KCNC1 (EPM7)": "Giant SEPs + C-reflex (baseline + annual); EEG (PPR); KCNC1 R320H first (Finnish/Baltic); POLG1 mandatory before VPA; piracetam 3.2-4.8 g/day; AVOID CBZ/OXC/PHT/LTG/GBP/PGB/VGB absolutely; gait assessment (physiotherapy); cognitive screen (largely preserved — reassure); Finnish KCNC1 registry",
            "PRICKLE1 (EPM1B)": "PRICKLE1 full sequencing; skin biopsy (normal — no Lafora bodies); Giant SEPs ± C-reflex; piracetam + levetiracetam; valproate; AVOID CBZ/OXC/PHT/LTG; physiotherapy; better prognosis than Lafora — communicate; 25% sibling risk",
            "KCTD7 (EPM3)": "Lysosomal enzyme assays FIRST (TPP1/PPT1 to exclude CLN2/CLN1): negative → proceed to KCTD7; skin/conjunctival biopsy (EM: NCL-like profiles possible); KCTD7 sequencing; developmental/neuropsychological assessment (severe ID); PEG evaluation (swallowing safety); valproate + levetiracetam; palliative care early; multidisciplinary: paediatric neurology, developmental medicine, physiotherapy, OT, SLT, dietitian",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Lafora bodies (EPM2A/NHLRC1): {ov['lafora_bodies_skin_biopsy_patients']}")
    print(f"Renal failure (SCARB2): {ov['renal_failure_scarb2_patients']}")
    print(f"Scoliosis (GOSR2): {ov['scoliosis_gosr2_patients']}")
    print(f"Giant SEPs: {ov['giant_seps_patients']}")
    print(f"Piracetam patients: {ov['piracetam_patients']}")
