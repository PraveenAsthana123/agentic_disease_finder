#!/usr/bin/env python3
"""Hereditary-Polyglutamine-SCA-Atlas — Complete 8-Gene Polyglutamine & Repeat-Expansion
Spinocerebellar Ataxia Atlas (SCA1/2/3/7/8/10/12/13).

ATXN1    (Ataxin-1; 816 aa; 6p22.3; AD;
          SCA1 — CAG ≥39 pathogenic (normal ≤35);
          HYPERREFLEXIA EARLY PATHOGNOMONIC — distinguishes from most SCAs (hyporeflexia);
          Fastest progression of common SCAs — wheelchair ~10yr from onset;
          Brainstem atrophy (pons + medulla) + cerebellar on MRI;
          Dysphagia and respiratory failure cause premature death;
          No disease-modifying therapy; seed SEED_BASE+0).
ATXN2    (Ataxin-2; 1312 aa; 12q24.12; AD;
          SCA2 — CAG ≥33 pathogenic (normal ≤31);
          SLOW SACCADES PATHOGNOMONIC — hypometric + slowed horizontal saccades on oculomotor exam;
          ALS MODIFIER at 27-33 intermediate repeats — SCA2 alleles modulate TDP-43 aggregation;
          Hyporeflexia/areflexia (peripheral neuropathy) — opposite of SCA1;
          Cuban founder (Holguin province — highest SCA2 prevalence globally);
          seed SEED_BASE+1).
ATXN3    (Ataxin-3 / Machado-Joseph Disease; 361 aa; 14q32.12; AD;
          SCA3/MJD — MOST COMMON SCA WORLDWIDE (28% of all SCAs);
          CAG ≥60 pathogenic (normal ≤44); anticipation strong;
          EXOPHTHALMOS (bulging eyes / eyelid retraction) PATHOGNOMONIC;
          Facial fasciculations and facial muscle rigidity;
          Three phenotypes: Type 1 (pyramidal), Type 2 (cerebellar), Type 3 (neuropathy);
          Azorean/Portuguese founder — very high prevalence Azores islands;
          seed SEED_BASE+2).
ATXN7    (Ataxin-7; 892 aa; 3p14.1; AD;
          SCA7 — CAG ≥37 pathogenic (normal ≤17);
          RETINAL DEGENERATION (macular dystrophy) PATHOGNOMONIC — annual fundoscopy mandatory;
          Blue-yellow colour vision loss EARLIEST feature (before ataxia);
          Extreme anticipation — infantile/severe form in children of SCA7 parents;
          ERG (electroretinogram) for subclinical retinal damage;
          Visual loss → central blindness precedes or accompanies cerebellar signs;
          seed SEED_BASE+3).
ATXN10   (Ataxin-10; 475 aa; 22q13.31; AD;
          SCA10 — ATTCT pentanucleotide repeat expansion (intron 9) — NOT polyQ;
          SEIZURES 50% — complex partial/secondarily generalized — PATHOGNOMONIC co-feature;
          Standard sequencing MISSES this — request ATTCT-specific PCR;
          Mexican / Brazilian ancestry (Amerindian origin) — ethnic clue;
          Pure cerebellar ataxia + seizures; minimal cognitive decline;
          AED mandatory (LEV/VPA); seed SEED_BASE+4).
ATXN8OS  (Ataxin-8 OS / ATXN8; CTG/CAG bidirectional; 13q21.33; AD;
          SCA8 — INCOMPLETE PENETRANCE (~30%) — genetic counselling essential;
          CTG ≥71 repeats on ATXN8OS / CAG on antisense ATXN8 both contribute;
          Tremor prominent (action tremor + rest tremor) — atypical for pure SCA;
          POSITIVE FAMILY HISTORY NOT REQUIRED — reduced penetrance;
          Slow progression; spastic component (UMN signs);
          Repeat-primed PCR for ATXN8/ATXN8OS bidirectional assay;
          seed SEED_BASE+5).
PPP2R2B  (Protein Phosphatase 2 Regulatory Subunit B; 443 aa; 5q32; AD;
          SCA12 — CAG expansion in 5-prime UTR/promoter region of PPP2R2B;
          TREMOR DOMINANT (action + head tremor) BEFORE ataxia — misdiagnosed as Essential Tremor;
          Indian subcontinent founder — most common SCA in India (Punjab predominant);
          Early-onset action tremor (hands + head) misdiagnosed ET — test PPP2R2B in Indian ET;
          Cognitive decline late; cerebellar atrophy on MRI;
          seed SEED_BASE+6).
KCNC3    (Potassium Channel Kv3.3; 735 aa; 19q13.33; AD;
          SCA13 — loss-of-function Kv3.3 fast-spiking neuron channel;
          CHILDHOOD ONSET with INTELLECTUAL DISABILITY (R420H variant) PATHOGNOMONIC;
          Cerebellar hypoplasia on MRI (not atrophy) — structural malformation not degeneration;
          R420H (p.Arg420His) — Western European founder (French/Luxembourg) — childhood, non-progressive;
          F448L (p.Phe448Leu) — adult onset, progressive;
          Two phenotypes same gene: map genotype to expected course;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2102-2109).
"""

import random

SEED_BASE = 2102

POLYQ_SCA_GENES = [
    # -- ATXN1 — SCA1 ---------------------------------------------------------
    {
        "gene": "ATXN1",
        "alt_name": (
            "ATXN1 (ATXN1-816aa-6p22.3 / AD — SCA1-CAG≥39-Hyperreflexia-PATHOGNOMONIC-Early — "
            "Fastest-Progression-Common-SCAs-Wheelchair-10yr — "
            "Brainstem-Atrophy-Pons-Medulla-MRI — Dysphagia-Respiratory-Premature-Death)"
        ),
        "protein": (
            "ATXN1 -- 6p22.3 AD -- ATXN1-816aa -- "
            "Ataxin-1-AXH-Domain-RNA-Binding-Transcriptional-Regulator -- "
            "SCA1-CAG-Repeat-Exon-8-Pathogenic≥39-Normal≤35-PolyQ-Nuclear-Aggregation -- "
            "AXH-Domain-Interactions-LANP-CIC-RBM17-Transcription-Factor-Complex -- "
            "Mutant-Ataxin-1-Misfolds-PolyQ-Nuclear-Inclusions-Purkinje-Cell-Nuclei -- "
            "ATXN1-Phosphorylation-Ser776-Required-Toxicity-14-3-3-Binding -- "
            "PolyQ-Protein-Aggregation-UPS-Impairment-Spinocerebellar-Tract -- "
            "Purkinje-Cell-Loss-Inferior-Olivary-Nucleus-Degeneration -- "
            "HYPERREFLEXIA-BRISK-EARLY-SCA1-Unlike-Most-SCAs-Hyporeflexia -- "
            "Brainstem-Atrophy-Pons-Medulla-T2-MRI-Hot-Cross-Bun-Sign-Sometimes -- "
            "Dysphagia-Dysarthria-Respiratory-Failure-Progressive -- "
            "Cognitive-Impairment-Executive-Dysfunction-Psychiatric-Features"
        ),
        "locus": "6p22.3",
        "protein_size": "816 aa",
        "inheritance": (
            "AD (autosomal dominant); high penetrance (>99%); "
            "CAG ≥39 repeats pathogenic; normal ≤35; 36-38 reduced penetrance; "
            "CAG range correlates with severity: longer repeat → earlier onset/faster; "
            "Anticipation: CAG repeats expand on paternal transmission (preference); "
            "De novo mutations rare; "
            "Family history present in ~90% of cases; "
            "Homozygotes (two expanded alleles) — more severe phenotype; "
            "Repeat-primed PCR + sizing mandatory for accurate count"
        ),
        "pathognomonic": (
            "HYPERREFLEXIA BRISK TENDON REFLEXES EARLY — pathognomonic among SCAs (most others have hyporeflexia); "
            "Brainstem atrophy (pons + medulla) on MRI alongside cerebellar atrophy; "
            "SCA1 CAG ≥39 molecular confirmation; "
            "Fast progression — mean time to wheelchair ~10yr (fastest among SCA1/2/3/6); "
            "Pyramidal signs (spasticity, Babinski) + cerebellar signs; "
            "Dysphagia requiring PEG consideration within 10-15yr; "
            "Cognitive impairment (executive function) + psychiatric (bipolar-like) in subset; "
            "Facial hypomimia + amyotrophy late (unlike SCA2 which has early fasciculations)"
        ),
        "treatment": (
            "NO disease-modifying therapy (no approved agent); "
            "Riluzole 50 mg BD (off-label, modest cerebellar benefit, limited evidence); "
            "Physiotherapy: gait training, falls prevention, balance; "
            "SLT: dysphagia management, PEG assessment; "
            "Respiratory monitoring: spirometry annually when dysphagia present; "
            "Nutritional support; "
            "Antidepressants for psychiatric symptoms; "
            "EUROSCA and RISCA registries — enrol for natural history; "
            "SARA score biannually; "
            "ATXN1 silencing RNA/ASO research active (no patient-ready trial)"
        ),
        "critical_flags": [
            "HYPERREFLEXIA-EARLY-PATHOGNOMONIC-SCA1",
            "FASTEST-PROGRESSION-COMMON-SCAS",
            "BRAINSTEM-ATROPHY-MRI-PONS-MEDULLA",
            "DYSPHAGIA-PEG-EARLY-CONSIDERATION",
            "RESPIRATORY-MONITORING-MANDATORY",
            "ANTICIPATION-PATERNAL-TRANSMISSION",
            "NO-DISEASE-MODIFYING-RX",
        ],
        "age_of_onset": "Adult onset typical 30-40yr; range 10-70yr; longer repeats → earlier/severe",
        "key_biomarker": "ATXN1 CAG repeat ≥39 (repeat-primed PCR; note 36-38 = reduced penetrance)",
        "seed": SEED_BASE + 0,
    },
    # -- ATXN2 — SCA2 ---------------------------------------------------------
    {
        "gene": "ATXN2",
        "alt_name": (
            "ATXN2 (ATXN2-1312aa-12q24.12 / AD — SCA2-CAG≥33-Slow-Saccades-PATHOGNOMONIC — "
            "ALS-Modifier-Intermediate-27-33-Repeats-TDP43 — "
            "Hyporeflexia-Peripheral-Neuropathy-Unlike-SCA1 — "
            "Cuban-Holguin-Province-Founder-Most-Prevalent-SCA2-Region)"
        ),
        "protein": (
            "ATXN2 -- 12q24.12 AD -- ATXN2-1312aa -- "
            "Ataxin-2-PAM2-Motif-PABP-Interaction-RNA-Granule-Assembly -- "
            "SCA2-CAG-Repeat-Exon-1-Pathogenic≥33-Normal≤31-CAG-32-Intermediate-Variable -- "
            "Intermediate-27-33-CAG-ALS-Risk-Modifier-TDP-43-Phase-Separation-Enhancer -- "
            "RNA-Granule-Stress-Response-Ataxin-2-P-Body-Component -- "
            "PolyQ-Expansion-Toxic-Nuclear-Cytoplasmic-Mislocalisation-Purkinje-Cells -- "
            "HYPOREFLEXIA-AREFLEXIA-PERIPHERAL-NEUROPATHY-PROMINENT -- "
            "Slow-Saccades-Hypometric-VOR-Impairment-Early-EOM-Finding -- "
            "Parkinsonism-REM-Behaviour-Disorder-Late-Features -- "
            "ALS-Intermediate-Allele-Tanaka-2003-NatGenet-Discovery -- "
            "REST-TREMOR-FASCICULATIONS-Lower-Motor-Neuron-Signs"
        ),
        "locus": "12q24.12",
        "protein_size": "1312 aa",
        "inheritance": (
            "AD (autosomal dominant); high penetrance (>95% at ≥38 repeats); "
            "CAG ≥33 pathogenic; normal ≤31; 32 = intermediate (uncertain significance); "
            "ALS modifier: 27-33 CAG intermediate alleles significantly increase ALS risk; "
            "Anticipation present (maternal > paternal expansion tendency); "
            "Cuban founder — very high allele frequency in Holguin province Cuba; "
            "Also high frequency in Italy, China, India (SCA2 2nd/3rd most common globally); "
            "Repeat-primed PCR + sizing; request specific ATXN2 CAG assay"
        ),
        "pathognomonic": (
            "SLOW SACCADES (hypometric, reduced velocity horizontal saccades) PATHOGNOMONIC — "
            "oculomotor exam mandatory; quantitative oculomotor testing (video-oculography preferred); "
            "Hyporeflexia/areflexia — distinguishes SCA2 from SCA1 (hyperreflexia); "
            "ATXN2 CAG ≥33 molecular confirmation; "
            "ALS modifier check: family members with ALS + ATXN2 27-33 CAG? (intermediate alleles); "
            "Parkinsonism component (bradykinesia, rigidity) in subset — levodopa trial; "
            "Peripheral neuropathy (axonal, NCS/EMG); "
            "MRI: cerebellar + pontine atrophy; olivo-ponto-cerebellar pattern"
        ),
        "treatment": (
            "NO disease-modifying therapy for SCA2 (no approved agent); "
            "Riluzole 50 mg BD (off-label cerebellar benefit — modest); "
            "Levodopa trial if parkinsonism features (30-40% partial response); "
            "Physiotherapy: balance, gait, falls prevention; "
            "SLT: dysarthria management; "
            "Oculomotor rehabilitation (vestibular-ocular reflex training); "
            "ALS-risk counselling for family with intermediate alleles (27-33 CAG); "
            "EUROSCA registry; RISCA natural history; "
            "ATXN2-specific ASO research (pre-clinical); "
            "SARA biannually"
        ),
        "critical_flags": [
            "SLOW-SACCADES-PATHOGNOMONIC-SCA2",
            "ALS-MODIFIER-27-33-INTERMEDIATE-REPEATS",
            "HYPOREFLEXIA-DDx-SCA1-HYPERREFLEXIA",
            "LEVODOPA-TRIAL-PARKINSONISM-COMPONENT",
            "CUBAN-HOLGUIN-FOUNDER-HIGH-PREVALENCE",
            "OCULOMOTOR-EXAM-MANDATORY",
            "NO-DISEASE-MODIFYING-RX",
        ],
        "age_of_onset": "Adult 20-50yr (mean 30yr); earlier with longer repeats; Cuban variants earlier",
        "key_biomarker": "ATXN2 CAG ≥33 (repeat-primed PCR; 27-33 = ALS modifier intermediate allele)",
        "seed": SEED_BASE + 1,
    },
    # -- ATXN3 — SCA3 / MJD ---------------------------------------------------
    {
        "gene": "ATXN3",
        "alt_name": (
            "ATXN3 (ATXN3-361aa-14q32.12 / AD — SCA3-MJD-Most-Common-SCA-Worldwide-28pct — "
            "CAG≥60-Pathognomonic-Molecular — "
            "Exophthalmos-Eyelid-Retraction-PATHOGNOMONIC-Facial-Fasciculations — "
            "Azorean-Portuguese-Founder-Three-Phenotype-Types-1-2-3)"
        ),
        "protein": (
            "ATXN3 -- 14q32.12 AD -- ATXN3-361aa -- "
            "Ataxin-3-Josephin-Domain-DUB-Deubiquitinase-PolyUb-Chain-Editing -- "
            "SCA3-MJD-CAG-Repeat-Exon-10-Pathogenic≥60-Normal≤44-PolyQ-Nuclear-Aggregation -- "
            "Most-Common-SCA-Worldwide-28pct-All-SCA-Families-All-Continents -- "
            "Azorean-Islands-Portugal-Haplotype-D1B-Founder-Atlantic-Slave-Trade-Spread -- "
            "Josephin-Domain-Ubiquitin-Editing-LOF-PolyQ-GOF-Dual-Mechanism -- "
            "THREE-PHENOTYPE-TYPES-Repeat-Length-Dependent-Clinical-Variability -- "
            "EXOPHTHALMOS-Eyelid-Retraction-Fasciculations-Perioral-Face-PATHOGNOMONIC -- "
            "Cerebellar-Ataxia-Pyramidal-Extrapyramidal-Peripheral-Neuropathy-Restless-Legs -- "
            "Dystonia-Childhood-Onset-Type-1 -- "
            "Autonomic-Dysfunction-Orthostatic-Hypotension-Bladder"
        ),
        "locus": "14q32.12",
        "protein_size": "361 aa",
        "inheritance": (
            "AD (autosomal dominant); high penetrance; "
            "CAG ≥60 clearly pathogenic; 45-59 = reduced penetrance (complex); normal ≤44; "
            "Strong anticipation — paternal transmission preferential expansion; "
            "Inverse correlation: shorter repeats → later onset/milder; "
            "Azorean founder c.CAG-repeat — emigrant spread worldwide (Brazil, USA, Japan); "
            "Most common SCA in Asia, Brazil, and globally; "
            "Molecular: ATXN3 repeat-primed PCR; estimate repeat size both alleles"
        ),
        "pathognomonic": (
            "EXOPHTHALMOS (bulging eyes / eyelid retraction) PATHOGNOMONIC clinical sign; "
            "Facial fasciculations and perioral facial fasciculations; "
            "ATXN3 CAG ≥60 molecular confirmation; "
            "Type 1 (CAG >74): early onset, pyramidal + cerebellar + dystonia; "
            "Type 2 (CAG 60-74): adult onset, pure cerebellar (most common); "
            "Type 3 (CAG ~60): late onset, peripheral neuropathy prominent; "
            "Restless legs syndrome and sleep disorders; "
            "Autonomic neuropathy (orthostatic hypotension); "
            "MRI: cerebellar + pontine atrophy; substantia nigra signal change"
        ),
        "treatment": (
            "NO disease-modifying therapy (no approved agent); "
            "Riluzole 50 mg BD (off-label cerebellar benefit); "
            "Restless legs: dopamine agonists (pramipexole, rotigotine); "
            "Dystonia (Type 1): botulinum toxin, trihexyphenidyl, DBS consideration; "
            "Autonomic: midodrine/fludrocortisone for orthostatic hypotension; "
            "Physiotherapy + OT; SLT (dysphagia); "
            "EUROSCA + RISCA + SCA3 MJD natural history registries; "
            "ASO targeting ATXN3 in pre-clinical phase (IONIS, Alnylam); "
            "SARA biannually; SCAFI (SCA Functional Index)"
        ),
        "critical_flags": [
            "MOST-COMMON-SCA-WORLDWIDE-28pct",
            "EXOPHTHALMOS-EYELID-RETRACTION-PATHOGNOMONIC",
            "THREE-PHENOTYPES-REPEAT-LENGTH-DEPENDENT",
            "AZOREAN-PORTUGUESE-FOUNDER",
            "RESTLESS-LEGS-DOPAMINE-AGONIST",
            "DYSTONIA-TYPE1-DBS-CANDIDATE",
            "ANTICIPATION-STRONG-PATERNAL",
        ],
        "age_of_onset": "Type 1 (>74 CAG): teens-30s; Type 2 (60-74): 30-50yr; Type 3 (~60): 40-60yr",
        "key_biomarker": "ATXN3 CAG ≥60 (repeat-primed PCR; both alleles; ≥45 reduced penetrance)",
        "seed": SEED_BASE + 2,
    },
    # -- ATXN7 — SCA7 ---------------------------------------------------------
    {
        "gene": "ATXN7",
        "alt_name": (
            "ATXN7 (ATXN7-892aa-3p14.1 / AD — SCA7-CAG≥37-Retinal-Degeneration-PATHOGNOMONIC — "
            "Annual-Fundoscopy-ERG-Mandatory — "
            "Blue-Yellow-Colour-Vision-Loss-EARLIEST-Feature — "
            "Extreme-Anticipation-Infantile-Form-Children-SCA7-Parents)"
        ),
        "protein": (
            "ATXN7 -- 3p14.1 AD -- ATXN7-892aa -- "
            "Ataxin-7-SAGA-Transcriptional-Coactivator-DUB-Module-Component -- "
            "SCA7-CAG-Repeat-Exon-3-Pathogenic≥37-Normal≤17-Most-Variable-Window-SCAs -- "
            "SAGA-Complex-ATXN7-Component-H2B-Deubiquitination-Transcription-Activation -- "
            "CONE-ROD-DYSTROPHY-Macular-Degeneration-PATHOGNOMONIC-Retina-BEFORE-Cerebellum -- "
            "BLUE-YELLOW-TRITAN-COLOUR-VISION-DEFECT-EARLIEST-Photoreceptor-Cone-Loss -- "
            "Extreme-Anticipation-SCA7-Very-Unstable-Repeat-Infantile-Grandfathers-Mild -- "
            "Central-Scotoma-Progressive-Peripheral-Visual-Field-Loss-Total-Blindness -- "
            "SAGA-Aberrant-Transcription-Purkinje-Cells-AND-Retinal-Cells-Dual-Target -- "
            "ERG-Electroretinogram-Subclinical-Detection-Before-Ophthalmoscopy-Changes -- "
            "Visual-Loss-May-PRECEDE-Ataxia-Different-Order-Presentation"
        ),
        "locus": "3p14.1",
        "protein_size": "892 aa",
        "inheritance": (
            "AD (autosomal dominant); high penetrance; "
            "CAG ≥37 pathogenic; normal ≤17; 18-36 intermediate (variable); "
            "EXTREME anticipation — SCA7 has the most unstable repeat of all PolyQ SCAs; "
            "Grandfathers with mild ataxia can have grandchildren with infantile-onset disease; "
            "Paternal transmission: massive expansion possible (>100 → >300 repeats in one generation); "
            "Genetic counselling mandatory before reproduction; "
            "Repeat-primed PCR + Southern blot for large expansions"
        ),
        "pathognomonic": (
            "RETINAL DEGENERATION (cone-rod macular dystrophy) PATHOGNOMONIC — annual fundoscopy mandatory; "
            "Blue-yellow (tritan) colour vision defect EARLIEST feature — Farnsworth D-15 test; "
            "ERG (electroretinogram): photoreceptor dysfunction precedes ophthalmoscopic changes; "
            "Central scotoma on Amsler grid — patient reports blurring/distortion central vision; "
            "ATXN7 CAG ≥37 molecular; "
            "Visual loss may PRECEDE cerebellar ataxia (SCA7 unique feature among polyQ SCAs); "
            "Progressive central → peripheral visual field loss → near-total blindness; "
            "Cerebellar ataxia + dysarthria follow retinal degeneration; "
            "Infantile form: hypotonia, cardiac defects, severe CNS involvement"
        ),
        "treatment": (
            "NO disease-modifying therapy (no approved agent); "
            "Annual ophthalmology: fundoscopy + OCT + ERG + colour vision; "
            "Low vision aids + rehabilitation (magnification, orientation and mobility); "
            "Genetic counselling: extreme anticipation risk to offspring; "
            "Physiotherapy (ataxia component); "
            "Driving cessation when visual acuity <6/12 or significant field loss; "
            "SCA7 natural history consortium registration; "
            "ATXN7 gene silencing (AAV/ASO) pre-clinical research; "
            "Nutritional supplements (antioxidants) no proven efficacy but low risk"
        ),
        "critical_flags": [
            "RETINAL-DEGENERATION-PATHOGNOMONIC-SCA7",
            "ANNUAL-FUNDOSCOPY-ERG-MANDATORY",
            "BLUE-YELLOW-COLOUR-VISION-EARLIEST",
            "EXTREME-ANTICIPATION-INFANTILE-FORM",
            "VISUAL-LOSS-MAY-PRECEDE-ATAXIA",
            "GENETIC-COUNSELLING-MANDATORY-REPRODUCTION",
            "DRIVING-CESSATION-VISUAL-CRITERIA",
        ],
        "age_of_onset": "Adult typical 30-40yr; infantile form (very large repeats) neonatal; range 0-70yr",
        "key_biomarker": "ATXN7 CAG ≥37 + ERG/fundoscopy retinal dystrophy (earliest change)",
        "seed": SEED_BASE + 3,
    },
    # -- ATXN10 — SCA10 -------------------------------------------------------
    {
        "gene": "ATXN10",
        "alt_name": (
            "ATXN10 (ATXN10-475aa-22q13.31 / AD — SCA10-ATTCT-Pentanucleotide-Repeat-NOT-PolyQ — "
            "Seizures-50pct-PATHOGNOMONIC-Co-Feature — "
            "Standard-Sequencing-Misses-ATTCT-Request-Specific-PCR — "
            "Mexican-Brazilian-Amerindian-Ancestry-Ethnic-Clue)"
        ),
        "protein": (
            "ATXN10 -- 22q13.31 AD -- ATXN10-475aa -- "
            "Ataxin-10-Armadillo-Repeat-Scaffold-Protein-PKC-Epsilon-Interactor -- "
            "SCA10-ATTCT-Pentanucleotide-Repeat-Intron-9-NOT-PolyQ-Repeat-Expansion -- "
            "Pathogenic->800-ATTCT-Repeats-Normal-<33-Repeats-Wide-Gap -- "
            "Largest-Repeat-Expansions-4500-Units-Extreme-Cases -- "
            "Standard-Short-Read-Sequencing-Completely-Misses-ATTCT-Expansion -- "
            "Standard-Microarray-and-WES-MISS-Requires-ATTCT-Specific-PCR-Assay -- "
            "Amerindian-Origin-Haplotype-Mexico-Brazil-Japan-Distinct-Founders -- "
            "SEIZURES-Complex-Partial-Secondarily-Generalized-50pct-Coexist-Ataxia -- "
            "Pure-Cerebellar-Ataxia-Otherwise-Minimal-Other-Features -- "
            "Epilepsy-Present-Misdiagnosed-Epilepsy-Plus-Ataxia-if-ATTCT-not-tested"
        ),
        "locus": "22q13.31",
        "protein_size": "475 aa",
        "inheritance": (
            "AD (autosomal dominant); "
            "ATTCT pentanucleotide repeat expansion intron 9 of ATXN10; "
            "Pathogenic: >800 ATTCT repeats; normal: <33 repeats; "
            "Large gap between normal and pathogenic (no grey zone); "
            "Standard sequencing, microarray, WES ALL MISS this repeat; "
            "Request ATTCT-specific PCR (specialized neuro genetics lab); "
            "Anticipation variably present; "
            "Mexican founder (Nuevo León/Jalisco haplotype) distinct from Brazilian; "
            "Japanese SCA10 described with distinct haplotype"
        ),
        "pathognomonic": (
            "SEIZURES ~50% of SCA10 patients — complex partial or secondarily generalized; "
            "Ataxia + seizures combination in patient of Amerindian ancestry = SCA10 until proven otherwise; "
            "ATTCT repeat >800 on specialized PCR — molecular confirmation; "
            "Pure cerebellar syndrome (otherwise) — no pyramidal/extrapyramidal/retinal signs; "
            "MRI: cerebellar atrophy (mild to moderate) without brainstem signal change; "
            "EEG: epileptiform discharges temporal/multiregional; "
            "Minimal cognitive decline (unlike other epileptic SCAs); "
            "Onset typically 20-45yr"
        ),
        "treatment": (
            "AED for seizures MANDATORY: "
            "Levetiracetam (preferred — renal excretion, minimal interactions, well-tolerated) or "
            "Valproate (VPA — effective but avoid in women of childbearing age); "
            "Carbamazepine second-line; "
            "Physiotherapy for ataxia; "
            "SCA10 natural history — limited data, enrol in registries; "
            "Genetic counselling (paternal/maternal); "
            "Driving assessment (seizure-free interval per jurisdiction before driving)"
        ),
        "critical_flags": [
            "SEIZURES-50pct-PATHOGNOMONIC-CO-FEATURE-SCA10",
            "STANDARD-SEQUENCING-MISSES-ATTCT-REPEAT",
            "REQUEST-ATTCT-SPECIFIC-PCR-MANDATORY",
            "MEXICAN-BRAZILIAN-AMERINDIAN-ANCESTRY",
            "AED-MANDATORY-LEVETIRACETAM-PREFERRED",
            "NOT-POLYQ-PENTANUCLEOTIDE-ATTCT",
            "DRIVING-SEIZURE-FREE-INTERVAL",
        ],
        "age_of_onset": "20-45yr typical; range varies; seizures may predate ataxia by years",
        "key_biomarker": "ATXN10 ATTCT repeat >800 on specialized PCR (NOT detected by standard WES/panel)",
        "seed": SEED_BASE + 4,
    },
    # -- ATXN8OS — SCA8 -------------------------------------------------------
    {
        "gene": "ATXN8OS",
        "alt_name": (
            "ATXN8OS (ATXN8OS/ATXN8-CTG/CAG-Bidirectional-13q21.33 / AD — "
            "SCA8-Incomplete-Penetrance-30pct-Genetic-Counselling-Essential — "
            "Tremor-Prominent-Action-Rest-Atypical-SCA — "
            "Positive-Family-History-NOT-Required-Reduced-Penetrance)"
        ),
        "protein": (
            "ATXN8OS -- 13q21.33 AD-Incomplete-Penetrance -- ATXN8OS/ATXN8-CTG/CAG-bidirectional -- "
            "Bidirectional-Gene-Locus-CTG-Repeat-Sense-Strand-ATXN8OS-Non-Coding-RNA -- "
            "CAG-Antisense-Strand-ATXN8-PolyQ-Protein -- "
            "DUAL-MECHANISM-RNA-Toxic-Gain-Function-MBNL1-Sequestration-PLUS-PolyQ-Toxicity -- "
            "CTG-Repeat-Pathogenic≥71-Variable-Penetrance-30pct-Population -- "
            "INCOMPLETE-PENETRANCE-Major-Challenge-Genetic-Counselling -- "
            "MBNL1-RNA-Binding-Protein-Sequestration-Splicing-Dysregulation-Similar-DM1 -- "
            "ATXN8-PolyQ-Nuclear-Inclusions-Purkinje-Bergmann-Glia -- "
            "Tremor-Prominent-Action-Rest-Tremor-ATYPICAL-Most-SCAs-Pure-Cerebellar -- "
            "Spastic-Gait-UMN-Signs-Cerebellar-Ataxia-Combination -- "
            "Variable-Expression-Some-Carriers-Totally-Asymptomatic-Lifetime"
        ),
        "locus": "13q21.33",
        "protein_size": "CTG/CAG repeat (bidirectional)",
        "inheritance": (
            "AD with INCOMPLETE PENETRANCE (~30% of repeat carriers develop disease); "
            "CTG ≥71 on ATXN8OS associated with SCA8; "
            "Pathogenic range complex: 71-400+ repeats; normal <50; "
            "Penetrance estimate only 30% — most carriers asymptomatic; "
            "Genetic counselling essential: positive test ≠ certain diagnosis; "
            "Family history NOT required (low penetrance means many carriers undiagnosed); "
            "Bidirectional assay: test CTG (sense) AND CAG (antisense) strands; "
            "Repeat-primed PCR for ATXN8/ATXN8OS bidirectional"
        ),
        "pathognomonic": (
            "TREMOR PROMINENT (action tremor + rest tremor) — atypical for most SCAs (normally pure cerebellar); "
            "Spastic-ataxic gait (UMN signs + cerebellar — combination); "
            "ATXN8OS CTG ≥71 molecular + clinical evidence (low penetrance requires both); "
            "INCOMPLETE PENETRANCE: positive family member with repeat does NOT confirm diagnosis alone; "
            "MRI: cerebellar atrophy + sometimes mild white matter signal; "
            "Dysarthria (cerebellar + pyramidal component); "
            "Slow progression — many patients not wheelchair-bound for decades"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "Propranolol / primidone for tremor; "
            "Baclofen / tizanidine for spasticity; "
            "Physiotherapy (ataxia + spasticity); "
            "SLT (dysarthria); "
            "Genetic counselling: LOW PENETRANCE — test result interpretation complex; "
            "Avoid unnecessary anxiety in asymptomatic carriers — may never develop disease; "
            "SARA score + TUG (Timed Up and Go) biannually; "
            "SCA8 natural history — limited controlled data"
        ),
        "critical_flags": [
            "INCOMPLETE-PENETRANCE-30pct-SCA8",
            "POSITIVE-TEST-DOES-NOT-CONFIRM-DIAGNOSIS",
            "TREMOR-PROMINENT-ATYPICAL-SCA",
            "BIDIRECTIONAL-CTG-CAG-ASSAY-REQUIRED",
            "GENETIC-COUNSELLING-MANDATORY-LOW-PENETRANCE",
            "SPASTIC-ATAXIA-UMN-CEREBELLAR",
            "MBNL1-RNA-SEQUESTRATION-MECHANISM",
        ],
        "age_of_onset": "Wide range 20-70yr; slow progression; some carriers remain asymptomatic",
        "key_biomarker": "ATXN8OS CTG ≥71 + ATXN8 CAG (bidirectional PCR) + clinical syndrome (penetrance ~30%)",
        "seed": SEED_BASE + 5,
    },
    # -- PPP2R2B — SCA12 ------------------------------------------------------
    {
        "gene": "PPP2R2B",
        "alt_name": (
            "PPP2R2B (PPP2R2B-443aa-5q32 / AD — SCA12-CAG-5prime-UTR-Tremor-DOMINANT — "
            "Action-Head-Tremor-MISDIAGNOSED-Essential-Tremor-TEST-PPP2R2B-Indian-Patients — "
            "Indian-Subcontinent-Founder-Most-Common-SCA-India-Punjab — "
            "Late-Cerebellar-Ataxia-After-Years-of-Tremor)"
        ),
        "protein": (
            "PPP2R2B -- 5q32 AD -- PPP2R2B-443aa -- "
            "Protein-Phosphatase-2A-Regulatory-B-Subunit-Beta2-PPP2R2B-Brain-Enriched -- "
            "SCA12-CAG-Repeat-5prime-UTR-Promoter-Region-Pathogenic>40-Normal<29 -- "
            "NOT-Coding-PolyQ-BUT-5-UTR-Expansion-Affects-PPP2R2B-Transcription-Level -- "
            "PP2A-Phosphatase-Tau-Dephosphorylation-Role-Neurodegeneration-Pathway -- "
            "TREMOR-DOMINANT-ACTION-TREMOR-HEAD-TREMOR-YEARS-BEFORE-ATAXIA -- "
            "Indian-Subcontinent-Founder-Haplotype-Most-Common-SCA-Subtype-India -- "
            "Misdiagnosed-Essential-Tremor-Decades-Before-Ataxia-Develops -- "
            "Late-Onset-Cerebellar-Ataxia-After-Tremor-Phase -- "
            "Cognitive-Decline-Dementia-Late-Features-SCA12 -- "
            "Cerebellar-Atrophy-MRI-Also-Cerebral-Cortical-Atrophy-Late"
        ),
        "locus": "5q32",
        "protein_size": "443 aa",
        "inheritance": (
            "AD (autosomal dominant); high penetrance; "
            "CAG repeat in 5' UTR of PPP2R2B; pathogenic >40 repeats; normal <29; "
            "Indian subcontinent founder haplotype — Punjab, Haryana high prevalence; "
            "Most common SCA subtype in India; "
            "Anticipation present; "
            "Test: PPP2R2B CAG repeat assay (5' UTR PCR — NOT standard exome panel); "
            "Often missed because it's non-coding and not in standard panels"
        ),
        "pathognomonic": (
            "ACTION TREMOR OF HANDS AND HEAD PATHOGNOMONIC EARLY (10-20yr before ataxia); "
            "Indian subcontinent ancestry + tremor misdiagnosed as essential tremor = SCA12 until proven otherwise; "
            "PPP2R2B CAG >40 molecular confirmation; "
            "Tremor phase: 3-10 Hz postural/action tremor; "
            "Cerebellar phase: gait ataxia develops years after tremor onset; "
            "Cognitive decline / dementia late (affects cerebral cortex); "
            "MRI: cerebellar atrophy + cerebral cortical atrophy (more diffuse than other SCAs)"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "Propranolol 40-160 mg/day for tremor (Level B essential tremor protocol); "
            "Primidone 50-250 mg/day (alternative/addition); "
            "DBS VIM (ventral intermediate thalamus) if refractory disabling tremor (off-label SCA12); "
            "Physiotherapy (ataxia when develops); "
            "Cognitive support / dementia management late; "
            "SCA12 — limited natural history data; "
            "Indian SCA12 consortium enrolment; "
            "SARA biannually once ataxia develops"
        ),
        "critical_flags": [
            "TREMOR-DOMINANT-MISDIAGNOSED-ESSENTIAL-TREMOR",
            "TEST-PPP2R2B-IN-INDIAN-ET-PATIENTS",
            "5-PRIME-UTR-REPEAT-NOT-POLYQ-CODING",
            "STANDARD-PANELS-MISS-NON-CODING",
            "INDIAN-SUBCONTINENT-FOUNDER",
            "PROPRANOLOL-PRIMIDONE-TREMOR",
            "DBS-VIM-REFRACTORY-TREMOR",
        ],
        "age_of_onset": "Tremor onset 30-50yr; ataxia follows 10-20yr later; cognitive decline very late",
        "key_biomarker": "PPP2R2B CAG >40 in 5' UTR (non-coding PCR assay; not standard exome)",
        "seed": SEED_BASE + 6,
    },
    # -- KCNC3 — SCA13 --------------------------------------------------------
    {
        "gene": "KCNC3",
        "alt_name": (
            "KCNC3 (KCNC3-735aa-19q13.33 / AD — SCA13-Kv3.3-Fast-Spiking-Channel — "
            "Childhood-Onset-Intellectual-Disability-R420H-PATHOGNOMONIC — "
            "Cerebellar-Hypoplasia-NOT-Atrophy-MRI-Structural-NOT-Degenerative — "
            "Two-Alleles-Two-Phenotypes-R420H-Childhood-F448L-Adult-Progressive)"
        ),
        "protein": (
            "KCNC3 -- 19q13.33 AD -- KCNC3-735aa -- "
            "Potassium-Channel-Kv3.3-Shaw-Family-Fast-Activating-Deactivating-High-Threshold -- "
            "SCA13-Loss-of-Function-Kv3.3-Fast-Spiking-Neuron-Channel-Purkinje-Cells-Interneurons -- "
            "TWO-PHENOTYPES-SAME-GENE-Genotype-Phenotype-Mapping-Essential -- "
            "R420H-p.Arg420His-Western-European-French-Luxembourg-Founder-Childhood-Onset -- "
            "F448L-p.Phe448Leu-Adult-Onset-Progressive-Ataxia -- "
            "R420H-CHILDHOOD-NON-PROGRESSIVE-Cerebellar-Hypoplasia-Mild-ID -- "
            "Cerebellar-Hypoplasia-MRI-Structural-MALFORMATION-Not-Degenerative-Atrophy -- "
            "Fast-Spiking-Inhibitory-Interneuron-Basket-Cells-Stellate-Cells-Affected -- "
            "Slow-Kinetics-LOF-Reduces-Purkinje-Cell-Inhibitory-Input-Dysrhythmia -- "
            "KCNC3-R420H-vs-F448L-Order-Genetic-Report-Critical-For-Prognosis"
        ),
        "locus": "19q13.33",
        "protein_size": "735 aa",
        "inheritance": (
            "AD (autosomal dominant); loss-of-function variants; "
            "Two major alleles with distinct phenotypes: "
            "R420H (p.Arg420His, c.1259G>A) — Western European (French/Luxembourg) founder; "
            "F448L (p.Phe448Leu, c.1342T>C) — adult onset progressive; "
            "Other LOF variants: truncations, splice site; "
            "Genetic counselling: specify WHICH variant — determines prognosis; "
            "De novo variants described for severe forms"
        ),
        "pathognomonic": (
            "CHILDHOOD ONSET WITH INTELLECTUAL DISABILITY (R420H) PATHOGNOMONIC — "
            "cerebellar ataxia with mild-moderate ID from early childhood; "
            "CEREBELLAR HYPOPLASIA on MRI (not atrophy — structural, not degenerative) — R420H; "
            "F448L: adult-onset progressive pure cerebellar ataxia (atrophy not hypoplasia); "
            "Genotype-phenotype: MUST report which specific variant found; "
            "R420H: non-progressive or slowly progressive cerebellar syndrome; "
            "F448L: progressive cerebellar ataxia onset 40-60yr; "
            "Pontine hypoplasia alongside cerebellar in R420H; "
            "Dysarthria in both phenotypes"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "R420H childhood: developmental support, special education, physiotherapy; "
            "Intellectual disability: cognitive rehabilitation, school support; "
            "Physiotherapy for ataxia (both phenotypes); "
            "SLT for dysarthria; "
            "F448L adult: SARA biannually; falls prevention; "
            "KCNC3 natural history: limited studies; "
            "Genetic counselling: prognosis depends on exact variant (R420H vs F448L); "
            "Research: Kv3.3 channel enhancers in pre-clinical phase"
        ),
        "critical_flags": [
            "CHILDHOOD-ONSET-ID-R420H-PATHOGNOMONIC",
            "CEREBELLAR-HYPOPLASIA-NOT-ATROPHY-MRI",
            "TWO-ALLELES-TWO-PHENOTYPES-GENOTYPE-ESSENTIAL",
            "R420H-NON-PROGRESSIVE-CHILDHOOD",
            "F448L-ADULT-PROGRESSIVE",
            "FAST-SPIKING-PURKINJE-INTERNEURON",
            "DEVELOPMENTAL-SUPPORT-R420H",
        ],
        "age_of_onset": "R420H: childhood (birth-5yr); F448L: adult 40-60yr",
        "key_biomarker": "KCNC3 variant (R420H vs F448L — MUST distinguish; phenotype/prognosis differ)",
        "seed": SEED_BASE + 7,
    },
]


def _generate_cohort():
    """Generate 320 synthetic patients (40 per gene, seeds 2102-2109)."""
    patients = []
    for gene_data in POLYQ_SCA_GENES:
        rng = random.Random(gene_data["seed"])
        gene = gene_data["gene"]
        for i in range(40):
            p: dict = {"gene": gene, "patient_id": f"{gene}-{i+1:03d}"}

            if gene == "ATXN1":
                repeat = rng.randint(39, 72)
                p["cag_repeat"] = repeat
                p["hyperreflexia"] = True  # pathognomonic
                p["brainstem_atrophy"] = repeat > 55
                p["dysphagia"] = repeat > 60 or rng.random() < 0.45
                p["cognitive_impairment"] = rng.random() < 0.50
                p["age_onset"] = max(15, 70 - repeat // 2 + rng.randint(-5, 5))
                p["years_to_wheelchair"] = max(5, 15 - (repeat - 39) // 5 + rng.randint(-2, 2))
                p["standard_panel_missed"] = rng.random() < 0.05  # rare miss

            elif gene == "ATXN2":
                repeat = rng.randint(33, 62)
                p["cag_repeat"] = repeat
                p["slow_saccades"] = True  # pathognomonic
                p["hyporeflexia"] = True
                p["peripheral_neuropathy"] = rng.random() < 0.75
                p["parkinsonism"] = rng.random() < 0.30
                p["als_risk_modifier"] = (27 <= repeat <= 33)
                p["age_onset"] = max(20, 55 - repeat // 3 + rng.randint(-5, 5))
                p["levodopa_trial"] = p["parkinsonism"]

            elif gene == "ATXN3":
                repeat = rng.randint(60, 82)
                p["cag_repeat"] = repeat
                p["exophthalmos"] = True  # pathognomonic
                p["facial_fasciculations"] = rng.random() < 0.80
                p["restless_legs"] = rng.random() < 0.55
                p["autonomic_dysfunction"] = rng.random() < 0.40
                phenotype_score = repeat
                p["phenotype_type"] = "Type1" if repeat > 74 else "Type2" if repeat >= 60 else "Type3"
                p["age_onset"] = max(10, 65 - (repeat - 60) + rng.randint(-5, 5))
                p["dystonia"] = (repeat > 74) and (rng.random() < 0.60)

            elif gene == "ATXN7":
                repeat = rng.randint(37, 75)
                p["cag_repeat"] = repeat
                p["retinal_degeneration"] = True  # pathognomonic
                p["colour_vision_loss"] = True
                p["ergs_abnormal"] = True
                p["visual_loss_before_ataxia"] = rng.random() < 0.55
                p["infantile_severe"] = (repeat > 200) and (rng.random() < 0.10)
                p["age_onset"] = max(5, 55 - (repeat - 37) // 2 + rng.randint(-5, 5))
                p["fundoscopy_abnormal"] = True

            elif gene == "ATXN10":
                repeats = rng.randint(800, 4000)
                p["attct_repeats"] = repeats
                p["seizures"] = rng.random() < 0.50  # pathognomonic co-feature
                p["pure_cerebellar"] = True
                p["standard_panel_missed"] = True  # always missed by standard
                p["standard_sequencing_missed"] = True
                p["mexican_brazilian_ancestry"] = rng.random() < 0.70
                p["aed_prescribed"] = p["seizures"]
                p["age_onset"] = rng.randint(20, 45)

            elif gene == "ATXN8OS":
                ctg_repeat = rng.randint(71, 300)
                p["ctg_repeat"] = ctg_repeat
                p["action_tremor_prominent"] = True
                p["spastic_ataxia"] = rng.random() < 0.65
                p["incomplete_penetrance_noted"] = True
                p["family_history_positive"] = rng.random() < 0.50  # only half (low penetrance)
                p["slow_progression"] = True
                p["age_onset"] = rng.randint(20, 65)

            elif gene == "PPP2R2B":
                repeat = rng.randint(41, 65)
                p["cag_repeat_5utr"] = repeat
                p["action_tremor_hands_head"] = True  # pathognomonic early
                p["misdiagnosed_et"] = rng.random() < 0.70  # misdiagnosed ET before SCA12 dx
                p["indian_subcontinent_ancestry"] = rng.random() < 0.75
                p["ataxia_present"] = rng.random() < 0.65
                p["cognitive_decline"] = rng.random() < 0.35
                p["standard_panel_missed"] = True  # non-coding, always missed
                p["standard_sequencing_missed"] = True
                p["age_onset_tremor"] = rng.randint(30, 50)
                p["years_tremor_before_ataxia"] = rng.randint(5, 20)

            elif gene == "KCNC3":
                variant = rng.choice(["R420H"] * 3 + ["F448L"])
                p["variant"] = variant
                p["childhood_onset"] = (variant == "R420H")
                p["intellectual_disability"] = (variant == "R420H")
                p["cerebellar_hypoplasia"] = (variant == "R420H")
                p["cerebellar_atrophy"] = (variant == "F448L")
                p["progressive"] = (variant == "F448L")
                p["western_european"] = (variant == "R420H") and (rng.random() < 0.70)
                p["age_onset"] = rng.randint(0, 5) if variant == "R420H" else rng.randint(40, 60)

            patients.append(p)
    return patients


def overview():
    """Return atlas-level aggregate overview."""
    patients = _generate_cohort()
    # Gene-specific pathognomonic counts
    hyperreflexia_atxn1 = sum(1 for p in patients if p.get("hyperreflexia"))
    slow_saccades_atxn2 = sum(1 for p in patients if p.get("slow_saccades"))
    exophthalmos_atxn3 = sum(1 for p in patients if p.get("exophthalmos"))
    retinal_degen_atxn7 = sum(1 for p in patients if p.get("retinal_degeneration"))
    seizures_atxn10 = sum(1 for p in patients if p.get("seizures"))
    tremor_ppp2r2b = sum(1 for p in patients if p.get("action_tremor_hands_head"))
    childhood_kcnc3 = sum(1 for p in patients if p.get("childhood_onset"))
    incomplete_pen_sca8 = sum(1 for p in patients if p.get("incomplete_penetrance_noted"))
    standard_missed = sum(1 for p in patients if p.get("standard_panel_missed") or p.get("standard_sequencing_missed"))
    misdiagnosed_et = sum(1 for p in patients if p.get("misdiagnosed_et"))
    return {
        "atlas": "Hereditary-Polyglutamine-SCA-Atlas",
        "genes": [g["gene"] for g in POLYQ_SCA_GENES],
        "total_patients": len(patients),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "atxn1_hyperreflexia_patients": hyperreflexia_atxn1,
        "atxn2_slow_saccades_patients": slow_saccades_atxn2,
        "atxn3_exophthalmos_patients": exophthalmos_atxn3,
        "atxn7_retinal_degeneration_patients": retinal_degen_atxn7,
        "atxn10_seizure_patients": seizures_atxn10,
        "sca8_incomplete_penetrance_patients": incomplete_pen_sca8,
        "ppp2r2b_action_tremor_patients": tremor_ppp2r2b,
        "kcnc3_childhood_onset_patients": childhood_kcnc3,
        "standard_panels_missed_diagnosis_patients": standard_missed,
        "ppp2r2b_misdiagnosed_et_patients": misdiagnosed_et,
    }


def breakdown():
    """Return per-gene breakdown with clinical details."""
    patients = _generate_cohort()
    result = {}
    for gene_data in POLYQ_SCA_GENES:
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
        "genes": {g["gene"]: g["protein"] for g in POLYQ_SCA_GENES},
        "glossary": {
            "Polyglutamine (PolyQ) SCA": (
                "Spinocerebellar ataxias caused by CAG trinucleotide repeat expansions in coding regions "
                "that produce expanded polyglutamine tracts; "
                "PolyQ SCAs: SCA1 (ATXN1), SCA2 (ATXN2), SCA3/MJD (ATXN3), SCA6 (CACNA1A), SCA7 (ATXN7), "
                "SCA17 (TBP); "
                "Note: SCA8, SCA10, SCA12 are REPEAT EXPANSION SCAs but NOT traditional polyQ coding expansions; "
                "All share anticipation (repeat lengthening across generations) and dominant inheritance; "
                "Mechanism: PolyQ misfolding → nuclear inclusions → transcriptional dysregulation → neurodegeneration"
            ),
            "SCA1 vs SCA2 Hyperreflexia/Hyporeflexia Rule": (
                "SCA1 (ATXN1): HYPERREFLEXIA early — brisk tendon reflexes due to pyramidal tract involvement; "
                "SCA2 (ATXN2): HYPOREFLEXIA/AREFLEXIA — peripheral neuropathy and dorsal column degeneration; "
                "This is a key clinical DDx: if early brisk reflexes → SCA1 first; if early absent reflexes → SCA2; "
                "Both cause cerebellar ataxia but reflex exam is the fastest bedside distinguisher"
            ),
            "Anticipation in Repeat SCAs": (
                "Most repeat SCAs show anticipation: repeat length increases generation-to-generation; "
                "SCA7 has EXTREME anticipation — grandfather mild ataxia, father moderate, child infantile fatal; "
                "Paternal transmission often expands more (SCA1, SCA3); "
                "SCA2 shows anticipation predominantly maternal-side; "
                "SCA8 has variable/incomplete penetrance — low penetrance makes anticipation less predictable; "
                "Clinical implication: earlier onset in children of affected parents = expanded repeats"
            ),
            "ATXN2 ALS Modifier (Intermediate Alleles)": (
                "ATXN2 CAG 27-33 repeats = INTERMEDIATE alleles; "
                "These do NOT cause SCA2 but significantly increase risk of ALS (~4-fold); "
                "TDP-43 co-aggregates with ATXN2 via liquid-liquid phase separation (LLPS); "
                "Tanaka 2003 (Nature Genetics) first discovered ALS-ATXN2 connection; "
                "ALS genetics now routinely includes ATXN2 intermediate repeat testing; "
                "Family members of ATXN2-27-33 allele carriers who develop ALS should be counselled; "
                "SCA2 family members may have ALS relatives — intermediate alleles in pedigree"
            ),
            "SCA7 Retinal Degeneration Protocol": (
                "SCA7 is UNIQUE among SCAs in causing progressive retinal degeneration (cone-rod dystrophy); "
                "Retinal degeneration may PRECEDE cerebellar ataxia — ophthalmology before neurology; "
                "Annual screening: ophthalmoscopy + OCT (central retinal thickness) + ERG + colour vision; "
                "Blue-yellow (tritan) axis affected first — Farnsworth D-15 panel test most sensitive early; "
                "Central scotoma on Amsler grid (patient self-monitoring); "
                "Infantile SCA7 (very large repeats): life-threatening, cardiac involvement; "
                "ERG: reduced amplitude photoreceptor response (cone then rod) before fundoscopic changes"
            ),
            "SCA10 ATTCT Repeat Testing": (
                "SCA10 is caused by ATTCT pentanucleotide repeat expansion in intron 9 of ATXN10; "
                "This is NOT a coding polyglutamine — it is an intronic repeat (not detected by exome); "
                "Standard WES, gene panels, SNP microarrays ALL MISS this expansion; "
                "Request: 'ATXN10 ATTCT repeat-primed PCR' explicitly in genetics referral; "
                "Normal: <33 ATTCT repeats; Pathogenic: >800 repeats (large gap, no grey zone); "
                "Ethnic context: Mexican, Brazilian, Japanese ancestry strongly predicted; "
                "Seizures in 50% — if cerebellar ataxia + seizures + Amerindian ancestry → test SCA10 first"
            ),
            "SCA8 Low Penetrance Counselling": (
                "SCA8 (ATXN8OS/ATXN8) has only ~30% penetrance in repeat carriers; "
                "This means 70% of people with the CTG expansion NEVER develop disease; "
                "A positive ATXN8OS CTG test does NOT confirm SCA8 diagnosis alone; "
                "Diagnosis requires BOTH the molecular finding AND a compatible clinical syndrome; "
                "Genetic counselling: positive result in asymptomatic family member = uncertain prognosis; "
                "Advise against unnecessary catastrophising — most carriers will not develop disease; "
                "Bidirectional assay needed: test both CTG (sense) and CAG (antisense) strands"
            ),
            "SCA12 Essential Tremor Misdiagnosis": (
                "SCA12 (PPP2R2B) presents with ACTION TREMOR of hands and head for 10-20yr BEFORE ataxia; "
                "This leads to misdiagnosis as Essential Tremor (ET) in most patients initially; "
                "Key clue: Indian subcontinent ancestry + ET + family history cerebellar features; "
                "PPP2R2B CAG repeat is in the 5' UTR (non-coding) — NOT detected by standard exome or panel; "
                "Request: 'PPP2R2B 5-prime UTR CAG repeat assay' explicitly; "
                "When ET in Indian patient does not fully respond to propranolol — consider SCA12; "
                "Treament gap: propranolol/primidone for tremor (same as ET), DBS VIM if refractory"
            ),
            "SCA13 Two Allele Clinical Rule": (
                "SCA13 (KCNC3) has TWO main phenotypes determined by WHICH variant: "
                "R420H (Western European/French-Luxembourg) → CHILDHOOD onset, non-progressive, mild ID, cerebellar HYPOPLASIA; "
                "F448L → ADULT onset 40-60yr, progressive pure cerebellar ataxia, cerebellar ATROPHY; "
                "Clinical mandate: ALWAYS report which specific variant (R420H or F448L); "
                "Never combine prognoses — childhood non-progressive ≠ adult progressive; "
                "Cerebellar HYPOPLASIA (structural malformation) distinguishes from ATROPHY (degeneration); "
                "R420H school-age children need cognitive support + physiotherapy not just 'observation'"
            ),
            "MJD/SCA3 Exophthalmos Sign": (
                "Machado-Joseph Disease (SCA3/ATXN3) is the MOST COMMON SCA worldwide (28%); "
                "EXOPHTHALMOS (bulging/staring eyes, eyelid retraction) is a pathognomonic bedside sign; "
                "Mechanism: eyelid levator hyperactivity (not orbital disease — eye movements intact until late); "
                "Facial fasciculations (perioral, mandibular) — look for subtle twitching; "
                "Restless legs syndrome affects ~50% — treatable with dopamine agonists; "
                "Azorean Portuguese founder: entire islands with high SCA3 prevalence; "
                "Three phenotypes: Type 1 (CAG>74, pyramidal+), Type 2 (adult cerebellar), Type 3 (neuropathy)"
            ),
        },
        "surveillance_protocols": {
            "ATXN1 (SCA1)": (
                "ATXN1 CAG repeat PCR + sizing (request ATXN1-specific assay); "
                "Baseline: MRI brain (cerebellar + brainstem atrophy pattern); "
                "SARA score biannually (SCA Ataxia Rating Scale); "
                "Swallowing assessment (SLT): clinical swallow eval + videofluoroscopy when dysphagia; "
                "PEG gastrostomy: consider early in fast progression; "
                "Spirometry annually (forced vital capacity — respiratory failure risk); "
                "Physiotherapy: balance, gait, falls prevention; "
                "Cognitive/psychiatric screen annually (executive dysfunction, bipolar features); "
                "EUROSCA registry; RISCA study"
            ),
            "ATXN2 (SCA2)": (
                "ATXN2 CAG repeat PCR (request ATXN2-specific assay); report full size both alleles; "
                "Intermediate alleles 27-33: counsel about ALS risk; screen family members; "
                "Oculomotor exam: quantitative video-oculography (saccade velocity/accuracy) biannually; "
                "NCS/EMG: baseline peripheral neuropathy characterisation; "
                "Levodopa trial if parkinsonism; "
                "MRI brain biannually (ponto-cerebellar atrophy); "
                "SARA biannually; "
                "EUROSCA + Cuban SCA2 network"
            ),
            "ATXN3 (SCA3/MJD)": (
                "ATXN3 CAG repeat PCR + sizing; both alleles; "
                "Clinical exam: look for exophthalmos + facial fasciculations; "
                "Restless legs: Pittsburgh Sleep Quality Index + dopamine agonist trial; "
                "SARA biannually; SCAFI annually; "
                "MRI brain (ponto-cerebellar atrophy + substantia nigra); "
                "Autonomic screen (tilt table, QSART, bladder diary) if symptomatic; "
                "Dystonia component (Type 1): botulinum toxin + trihexyphenidyl + DBS; "
                "EUROSCA + RISCA + Brazilian MJD Foundation"
            ),
            "ATXN7 (SCA7)": (
                "ATXN7 CAG repeat PCR + sizing (Southern blot for very large repeats); "
                "ANNUAL OPHTHALMOLOGY — mandatory: fundoscopy + OCT macular thickness + ERG + colour vision (Farnsworth D-15); "
                "ERG before symptom onset in at-risk family members; "
                "Amsler grid for self-monitoring central vision; "
                "Low vision rehabilitation when acuity <6/18; "
                "Driving cessation assessment; "
                "SARA biannually; "
                "Genetic counselling: extreme anticipation — offspring at high risk of more severe/earlier disease"
            ),
            "ATXN10 (SCA10)": (
                "ATXN10 ATTCT repeat PCR (MUST request specifically — NOT in standard panels/WES); "
                "EEG: baseline if seizures (complex partial monitoring); "
                "AED initiation: LEV 500 mg BD → titrate to 1000-1500 mg BD (preferred); "
                "VPA alternative (not in women of childbearing age); "
                "SARA biannually (mild ataxia); "
                "Seizure diary; driving — seizure-free interval per jurisdiction; "
                "MRI brain (mild cerebellar atrophy); "
                "Ethnic ancestry documentation (Mexican/Brazilian/Japanese founder)"
            ),
            "ATXN8OS (SCA8)": (
                "ATXN8OS CTG + ATXN8 CAG BIDIRECTIONAL assay (both strands; repeat-primed PCR); "
                "INTERPRET WITH CAUTION: 30% penetrance — positive test alone insufficient for diagnosis; "
                "Combine with clinical syndrome before diagnosing; "
                "Spasticity: baclofen 10-30 mg/day; tizanidine; "
                "Tremor: propranolol/primidone; "
                "SARA + TUG biannually; "
                "Genetic counselling: inform asymptomatic carriers of low penetrance; "
                "Annual clinical review; neuropsychological evaluation"
            ),
            "PPP2R2B (SCA12)": (
                "PPP2R2B 5-prime UTR CAG repeat assay (NON-CODING — specify to lab; NOT in standard panels); "
                "Indian subcontinent ancestry + ET → test PPP2R2B; "
                "Propranolol 40-120 mg/day for tremor; primidone if inadequate; "
                "DBS VIM if refractory disabling tremor; "
                "SARA when ataxia develops; "
                "Cognitive screen annually (MoCA) — late cognitive decline risk; "
                "MRI brain (cerebellar + cerebral atrophy late); "
                "Indian SCA12 research consortium registration"
            ),
            "KCNC3 (SCA13)": (
                "KCNC3 gene sequencing; REPORT SPECIFIC VARIANT (R420H vs F448L — CRITICAL for prognosis); "
                "R420H childhood: developmental assessment + IQ testing; school support plan; "
                "Physiotherapy (both phenotypes): ataxia management, falls; "
                "R420H: cerebellar hypoplasia MRI at diagnosis (not repeat); "
                "F448L: SARA biannually; cerebellar atrophy MRI annually; "
                "SLT: dysarthria both phenotypes; "
                "Genetic counselling: R420H vs F448L phenotype prognosis clearly communicated"
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
    print(f"ATXN1 hyperreflexia: {ov['atxn1_hyperreflexia_patients']}")
    print(f"ATXN2 slow saccades: {ov['atxn2_slow_saccades_patients']}")
    print(f"ATXN3 exophthalmos: {ov['atxn3_exophthalmos_patients']}")
    print(f"ATXN7 retinal degeneration: {ov['atxn7_retinal_degeneration_patients']}")
    print(f"ATXN10 seizures: {ov['atxn10_seizure_patients']}")
    print(f"SCA8 incomplete penetrance: {ov['sca8_incomplete_penetrance_patients']}")
    print(f"PPP2R2B action tremor: {ov['ppp2r2b_action_tremor_patients']}")
    print(f"KCNC3 childhood onset: {ov['kcnc3_childhood_onset_patients']}")
    print(f"Standard panels missed: {ov['standard_panels_missed_diagnosis_patients']}")
    print(f"PPP2R2B misdiagnosed ET: {ov['ppp2r2b_misdiagnosed_et_patients']}")
