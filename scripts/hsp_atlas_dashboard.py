#!/usr/bin/env python3
"""HSP-Atlas — Complete 8-Gene Hereditary Spastic Paraplegia Atlas
SPAST   (SPG4;  AD;  616 aa;  2q22.3; most common HSP ~40%; microtubule-severing spastin) ·
ATL1    (SPG3A; AD;  558 aa;  14q22.1; 2nd most common AD; early-onset childhood; atlastin-1 GTPase) ·
REEP1   (SPG31; AD;  201 aa;  2p11.2; 3rd most common AD; ER-shaping; dHMN overlap) ·
SPG11   (SPG11; AR;  2443 aa; 15q14;  most common AR complex HSP; thin corpus callosum PATHOGNOMONIC) ·
SPG7    (SPG7;  AR;  795 aa;  16q24.3; paraplegin; mitochondrial AAA metalloprotease; cerebellar) ·
CYP7B1  (SPG5A; AR;  506 aa;  8q12.3; oxysterol 7α-hydroxylase; elevated 25/27-OH-cholesterol; lovastatin Rx) ·
ZFYVE26 (SPG15; AR;  2539 aa; 14q24.1; spastizin; Kjellin syndrome overlap; thin corpus callosum) ·
KIF1A   (SPG30; AR/de-novo-AD; 1756 aa; 2q37.3; kinesin-3 motor; severe complex HSP; axonal transport)
320-patient aggregate cohort (8 × 40, seeds 1006–1013)

Hereditary Spastic Paraplegia — Key Neurological Principles:
  - HSP CLASSIFICATION: Characterised by progressive lower-limb spastic weakness (corticospinal tracts).
    PURE (uncomplicated) HSP: spasticity ± mild urinary urgency ± mild vibration loss in feet.
    COMPLEX (complicated) HSP: additional features — cognitive impairment, cerebellar ataxia,
    peripheral neuropathy, thin corpus callosum (TCC), optic neuropathy, epilepsy, pigmentary maculopathy.
  - INHERITANCE PATTERNS: AD pure (SPAST, ATL1, REEP1); AR complex (SPG11, SPG7, CYP7B1, ZFYVE26);
    AR/de-novo AD (KIF1A — de novo mutations disproportionately common; severe complex HSP).
  - SPAST/SPG4 (MOST COMMON ~40%): Spastin — microtubule-severing AAA-ATPase; haploinsufficiency.
    Pure HSP; variable onset (child to adult); slowly progressive; highly variable within families.
    Most common AD HSP globally. Penetrance incomplete (~75%).
  - ATL1/SPG3A: Atlastin-1 GTPase — ER tubule membrane fusion. Childhood / infancy onset (<10 y).
    DISTINCTIVE: early onset AD HSP with slow or non-progressive course (contrast SPAST which is progressive).
    2nd most common AD HSP (~10%).
  - REEP1/SPG31: ER-shaping protein; tubular ER; microtubule-binding. 3rd most common AD (~5%).
    Overlap with distal hereditary motor neuropathy (dHMN). Variable age of onset.
  - SPG11/Spatacsin: MOST COMMON AR COMPLEX HSP globally. THIN CORPUS CALLOSUM (TCC) on MRI —
    PATHOGNOMONIC for SPG11 context. Cognitive impairment. Amyotrophy. Cerebellar signs. Onset <20 y.
    Mutations in 40 exons; nonsense/frameshift predominate (LOF). Common in Turkish, North-African consanguineous.
  - SPG7/Paraplegin: Mitochondrial m-AAA metalloprotease (with AFG3L2). Progressive spastic ataxia.
    CEREBELLAR ATAXIA ≥ spasticity. Optic neuropathy (30%). Nystagmus. Dysphagia. Muscle biopsy: COX-negative fibres.
    Ragged-red fibres (RRF). Hearing loss. Mtochondrial disorder masquerade.
  - CYP7B1/SPG5A: Oxysterol 7α-hydroxylase — bile acid synthesis pathway. AR PURE HSP.
    BIOMARKER: elevated plasma 25-hydroxycholesterol AND 27-hydroxycholesterol (PATHOGNOMONIC ELEVATION).
    Oxysterol measurement MANDATORY for diagnosis confirmation and treatment monitoring.
    TREATMENT: Lovastatin (reduces 27-OHC synthesis via HMG-CoA reductase) — Level B evidence;
    pravastatin / atorvastatin alternatives. Sole treatable biochemical HSP.
  - ZFYVE26/SPG15: Spastizin — FYVE domain PI3P-binding; autophagosome maturation.
    AR COMPLEX HSP — TCC (not as consistent as SPG11); Kjellin syndrome (TCC + pigmentary maculopathy).
    Amyotrophy. Mild cognitive impairment. Onset 10–20 y. North-African/Middle-Eastern founders.
  - KIF1A/SPG30: Kinesin-3 motor — anterograde axonal transport of synaptic vesicle precursors.
    Severe COMPLEX HSP with intellectual disability, cerebellar atrophy, optic atrophy, epilepsy.
    De-novo dominant mutations DISPROPORTIONATELY COMMON (heterozygous); severe phenotype.
    Biallelic AR mutations: milder SPG30. Motor domain missense clustered at microtubule-binding surface.
  - MANAGEMENT (2026): No disease-modifying treatment for most HSP types.
    CYP7B1/SPG5A exception: lovastatin/atorvastatin (oxysterol reduction — Level B).
    Spasticity: oral baclofen (first-line); intrathecal baclofen pump (severe); tizanidine.
    Botulinum toxin A: lower-limb spasticity (hip adductors, gastrocnemius).
    Physiotherapy + orthoses mandatory. No fasting restriction. No mitochondrial drug CI (except SPG7 — mild).

COHORT: 8 × 40 = 320 patient slots (seeds 1006–1013; gene-specific seeds)
"""

import random

SEED_BASE = 1006

HSP_GENES = [
    # ── SPAST — SPG4 ────────────────────────────────────────────────────────
    {
        "gene": "SPAST", "protein": "Spastin",
        "alias": "SPG4 (OMIM #182601); AD; most common HSP (~40%); microtubule-severing AAA-ATPase; pure HSP; variable onset; 2q22.3",
        "aa": "616 aa", "kDa": "67 kDa",
        "gene_class": (
            "Spastin — ATP-dependent microtubule-severing enzyme; AAA (ATPases Associated with diverse "
            "Activities) superfamily; expressed in corticospinal neurons, hippocampus, spinal cord. "
            "MECHANISM: spastin hexamers assemble around microtubule lattice; ATP hydrolysis drives "
            "mechanical severing of tubulin subunits. Haploinsufficiency → reduced microtubule "
            "turnover → impaired axonal transport in distal corticospinal tract axons "
            "(the longest axons in the CNS, >1 m). "
            "Two isoforms: M1 (full-length, ER-associated) and M87 (cytosolic). "
            "MUTATION SPECTRUM: truncating mutations (LOF) ~60%; missense ~35%; CNVs ~5%. "
            "Penetrance ~75% (incomplete — explains phenotypic variability in families). "
            "2q22.3; OMIM gene 604277."
        ),
        "hsp_group": "AD Pure HSP — SPG4 (most common)",
        "subtype": "SPG4 — Pure HSP; SPAST Haploinsufficiency",
        "locus": "2q22.3", "omim_gene": 604277, "omim_disease": 182601,
        "inheritance": "Autosomal Dominant (AD). Haploinsufficiency. Penetrance ~75% (incomplete). De novo ~10-15%. High intrafamilial variability.",
        "seed_offset": 0,
        "onset_range_y": (2.0, 70.0),
        "gender": "both",
        "severity_weights": [0.30, 0.45, 0.25],
        "hsp_type": "pure",
        "tcc_prob": 0.0,
        "cognitive_prob": 0.05,
        "cerebellar_prob": 0.0,
        "optic_prob": 0.0,
        "maculopathy_prob": 0.0,
        "epilepsy_prob": 0.03,
        "peripheral_neuropathy_prob": 0.10,
        "phenotype": (
            "PURE SPASTIC PARAPLEGIA: Progressive lower-limb spastic weakness and gait disorder. "
            "Onset highly variable: infancy to 7th decade; typically 2nd–4th decade most common. "
            "Gait: scissoring, stiff-legged, circumduction. Lower-limb hyperreflexia + Babinski signs. "
            "Bladder urgency/frequency: 50–70% (corticospinal + autonomic tracts). "
            "MILD vibration reduction in feet (posterior columns — distal involvement). "
            "Upper limbs: largely spared in pure form. No cognitive impairment (pure SPG4). "
            "HIGH INTRAFAMILIAL VARIABILITY: asymptomatic carriers (25% penetrance gap) to "
            "wheelchair-requiring severe spasticity — all from same family mutation. "
            "Progression: slow; majority remain ambulant into 6th–7th decade. "
            "DRUGS: Baclofen oral (first-line spasticity); tizanidine; "
            "botulinum toxin A (hip adductors, gastrocnemius); intrathecal baclofen pump (severe). "
            "Physiotherapy mandatory. NO vincristine (worsens neuropathy). "
            "GENETIC: 50% risk to offspring; de-novo ~10-15%."
        ),
        "disease": (
            "SPG4/SPAST: most common hereditary spastic paraplegia worldwide. Prevalence ~2-3:100,000. "
            "~40% of all AD HSP; ~20% of all HSP combined. "
            "MRI brain/spine: often normal; may show mild corpus callosum thinning (not diagnostic). "
            "EMG: usually normal (pure form); mild sensory changes in some. "
            "Management: multidisciplinary — neurology, physiotherapy, urology (bladder). "
            "NO disease-modifying treatment (2026). Symptomatic spasticity management mainstay."
        ),
        "treatment_options": [
            "Oral baclofen — first-line for lower-limb spasticity",
            "Tizanidine — alternative antispasticity (caution hepatotoxicity)",
            "Botulinum toxin A — hip adductors + gastrocnemius (every 3 months)",
            "Intrathecal baclofen pump — severe refractory spasticity",
            "Physiotherapy — stretching, strengthening, gait training (mandatory)",
            "Ankle-foot orthoses (AFO) — gait support",
            "Urological management — anticholinergics for bladder urgency",
        ],
        "outcome_options": [
            "Stable — mild spasticity; ambulant with aids into 7th decade; physiotherapy maintained",
            "Progressive — moderate spasticity; scissoring gait; AFO dependent; bladder urgency",
            "Progressive — severe spasticity; intrathecal baclofen pump; limited community ambulation",
            "Slowly progressive — late onset (>40 y); minor disability at 20-year follow-up",
        ],
        "key_ddx": [
            "SPG3A/ATL1 — early onset AD HSP (childhood); non-progressive vs SPG4 progressive",
            "Primary lateral sclerosis (PLS) — adult onset; no family history; upper motor neuron only",
            "Hereditary motor neuropathy (dHMN) — REEP1 overlap; no true spasticity",
            "Tropical spastic paraparesis (HTLV-1) — acquired; serology mandatory in endemic areas",
            "Vitamin B12 deficiency — subacute combined degeneration; MRI cord signal; reversible",
            "SPG11 — AR complex; TCC on MRI; cognitive impairment; onset <20 y",
            "Multiple sclerosis — episodic; MRI plaques; CSF oligoclonal bands",
        ],
    },
    # ── ATL1 — SPG3A ────────────────────────────────────────────────────────
    {
        "gene": "ATL1", "protein": "Atlastin-1",
        "alias": "SPG3A (OMIM #182600); AD; 2nd most common AD HSP (~10%); GTPase; EARLY ONSET childhood; ER membrane fusion; 14q22.1",
        "aa": "558 aa", "kDa": "63 kDa",
        "gene_class": (
            "Atlastin-1 — dynamin-like GTPase; ER membrane fusion; critical for tubular ER network "
            "formation. Three-way ER junction formation via trans-dimerisation and GTP hydrolysis. "
            "Highly expressed in corticospinal neurons. Interacts with spastin (SPAST/SPG4) — "
            "ATL1 recruits spastin to ER via REEP1 for coordinated microtubule-ER remodelling. "
            "MUTATION SPECTRUM: missense dominant-negative mutations ~80%; truncating ~20%. "
            "Most common: p.Arg239Cys, p.Pro342Ser, p.Arg495Trp. "
            "14q22.1; OMIM gene 606439."
        ),
        "hsp_group": "AD Pure HSP — SPG3A (2nd most common AD)",
        "subtype": "SPG3A — Early-Onset Pure HSP; Atlastin-1 GTPase",
        "locus": "14q22.1", "omim_gene": 606439, "omim_disease": 182600,
        "inheritance": "Autosomal Dominant (AD). Dominant-negative (most) or haploinsufficiency. ~10% de novo.",
        "seed_offset": 1,
        "onset_range_y": (0.5, 12.0),
        "gender": "both",
        "severity_weights": [0.45, 0.40, 0.15],
        "hsp_type": "pure",
        "tcc_prob": 0.0,
        "cognitive_prob": 0.0,
        "cerebellar_prob": 0.0,
        "optic_prob": 0.0,
        "maculopathy_prob": 0.0,
        "epilepsy_prob": 0.02,
        "peripheral_neuropathy_prob": 0.05,
        "phenotype": (
            "EARLY-ONSET PURE SPASTIC PARAPLEGIA — CHILDHOOD / INFANCY onset (<10 y; often 1-5 y). "
            "KEY DISTINCTION: SLOW PROGRESSION or NON-PROGRESSIVE course after initial stabilisation. "
            "Children present with toe-walking, scissoring gait, delayed motor milestones. "
            "Lower-limb spasticity: Babinski+, hyperreflexia, clonus. "
            "May plateau in childhood — less progressive than SPAST/SPG4. "
            "Pure form: no cognitive impairment, no corpus callosum abnormality, no cerebellar signs. "
            "Upper limbs: generally spared. "
            "CLINICAL DISTINCTION FROM SPG4: ATL1 = early onset + slower progression; "
            "SPG4 = variable onset + progressive. "
            "Management: physiotherapy from infancy; AFO for toe-walking; baclofen if severe. "
            "NO vincristine."
        ),
        "disease": (
            "SPG3A/ATL1: 2nd most common AD HSP; ~10% of AD forms. Onset typically <10 y (sometimes infancy). "
            "MRI: usually normal. Prognosis better than SPG4 — significant proportion non-progressive. "
            "Management: early physiotherapy; AFO; baclofen if needed. Genetic counselling: 50% risk (AD)."
        ),
        "treatment_options": [
            "Early physiotherapy (from infancy/toddler age) — stretching, gait training",
            "Ankle-foot orthoses (AFO) — toe-walking correction",
            "Oral baclofen — if spasticity causes functional impairment",
            "Botulinum toxin A — gastrocnemius (toe-walking, young children)",
            "Serial casting — if equinus deformity progressive",
            "Orthopaedic input — for severe equinus/hip adductor tightness",
        ],
        "outcome_options": [
            "Stable — childhood onset; plateau by adolescence; independently ambulant as adult",
            "Mild progression — gait difficulty persists; AFO long-term; independent ambulation maintained",
            "Non-progressive — minimal disability; toe-walking corrected with physiotherapy + AFO",
            "Slowly progressive — moderate spasticity; baclofen; independent ambulation into adulthood",
        ],
        "key_ddx": [
            "SPG4/SPAST — later onset; progressive; adults common; ATL1 = childhood onset",
            "Diplegia / cerebral palsy — acquired perinatal; non-hereditary; neuroimaging abnormal",
            "Hereditary spastic diplegia (general) — exclude AR forms if consanguineous",
            "REEP1/SPG31 — AD HSP; overlap with dHMN; variable onset",
            "Dopa-responsive dystonia (DRD/GCH1) — dramatic l-DOPA response; diurnal variation",
            "Primary dystonia — DYT1/THAP1; absence of true corticospinal signs",
        ],
    },
    # ── REEP1 — SPG31 ────────────────────────────────────────────────────────
    {
        "gene": "REEP1", "protein": "Receptor Expression-Enhancing Protein 1",
        "alias": "SPG31 (OMIM #610250); AD; 3rd most common AD HSP (~5%); ER-shaping; microtubule-binding; dHMN overlap; 2p11.2",
        "aa": "201 aa", "kDa": "22 kDa",
        "gene_class": (
            "REEP1 (Receptor Expression-Enhancing Protein 1) — ER membrane-shaping protein; "
            "hair-pin loop insertions into ER membrane; induces membrane curvature of tubular ER. "
            "Interacts directly with spastin (SPAST) and atlastin-1 (ATL1) to form the 'SPG trinity' "
            "complex coordinating tubular ER + microtubule dynamics in distal axons. "
            "REEP1 also functions in distal motor axons → overlap with distal hereditary motor neuropathy (dHMN). "
            "MUTATION SPECTRUM: truncating (LOF) ~70%; missense ~30%. "
            "Haploinsufficiency mechanism. "
            "2p11.2; OMIM gene 609139."
        ),
        "hsp_group": "AD Pure HSP — SPG31 (3rd most common AD)",
        "subtype": "SPG31 — Pure HSP/dHMN Overlap; REEP1 ER-Shaping Protein",
        "locus": "2p11.2", "omim_gene": 609139, "omim_disease": 610250,
        "inheritance": "Autosomal Dominant (AD). Haploinsufficiency. De novo: ~20% (higher than SPAST/ATL1).",
        "seed_offset": 2,
        "onset_range_y": (1.0, 50.0),
        "gender": "both",
        "severity_weights": [0.35, 0.45, 0.20],
        "hsp_type": "pure_dhmn_overlap",
        "tcc_prob": 0.0,
        "cognitive_prob": 0.02,
        "cerebellar_prob": 0.0,
        "optic_prob": 0.0,
        "maculopathy_prob": 0.0,
        "epilepsy_prob": 0.02,
        "peripheral_neuropathy_prob": 0.30,
        "phenotype": (
            "PURE SPASTIC PARAPLEGIA WITH DISTAL MOTOR NEUROPATHY OVERLAP. "
            "Onset variable (childhood to adult). Spastic gait + lower-limb spasticity. "
            "DISTINCTIVE: significant distal lower-limb amyotrophy (weakness/wasting) in many patients — "
            "reflects dHMN overlap (pure motor neuropathy without sensory involvement). "
            "Pes cavus foot deformity common (due to distal motor involvement). "
            "EMG: may show chronic partial denervation in distal muscles (axonal motor pattern). "
            "Upper limbs: distal weakness in some (dHMN overlap). "
            "'SPG trinity' ER network dysfunction — shared pathway with SPAST and ATL1. "
            "Higher de-novo rate (~20%) — consider in sporadic cases with onset <20 y. "
            "Management: baclofen; AFO; physiotherapy; orthopaedics (pes cavus)."
        ),
        "disease": (
            "SPG31/REEP1: ~5% of AD HSP; 3rd most common AD form. "
            "MRI: usually normal. EMG may show distal motor axonopathy. "
            "Prognosis: slowly progressive; majority ambulant long-term. "
            "De-novo mutations: consider in apparent sporadic HSP under 20 y with normal parental examination."
        ),
        "treatment_options": [
            "Oral baclofen — spasticity management",
            "Physiotherapy — gait training + stretching (mandatory)",
            "Ankle-foot orthoses (AFO) — foot drop + pes cavus",
            "Orthopaedic surgery — pes cavus (plantar fasciotomy / calcaneal osteotomy) if severe",
            "Botulinum toxin A — gastrocnemius / hip adductors",
            "Neurology + orthopaedics multidisciplinary care",
        ],
        "outcome_options": [
            "Slowly progressive — spastic gait + mild amyotrophy; AFO; independently ambulant",
            "Moderate — pes cavus surgery + AFO; spasticity managed with baclofen",
            "Mild — childhood onset; plateau; gait aid in later decades",
            "Progressive — distal weakness + spasticity; wheelchair part-time >60 y",
        ],
        "key_ddx": [
            "SPG4/SPAST — most common AD HSP; less amyotrophy; no dHMN overlap",
            "ATL1/SPG3A — early onset childhood; less amyotrophy; non-progressive",
            "Distal hereditary motor neuropathy (dHMN) — motor only; no corticospinal signs; HSPB1/BSCL2",
            "CMT2/axonal CMT — sensory involvement; NEFL/MFN2/MPZ; NCS abnormal sensory",
            "ALS (amyotrophic lateral sclerosis) — adult onset; rapid progression; UMN+LMN; SOD1/FUS",
        ],
    },
    # ── SPG11 — Spatacsin ────────────────────────────────────────────────────
    {
        "gene": "SPG11", "protein": "Spatacsin",
        "alias": "SPG11 (OMIM #604360); AR; 2443 aa; most common AR complex HSP globally; thin corpus callosum PATHOGNOMONIC; 15q14",
        "aa": "2443 aa", "kDa": "270 kDa",
        "gene_class": (
            "Spatacsin — large scaffolding protein; localises to lysosomes and autolysosomes; "
            "forms complex with ZFYVE26 (spastizin/SPG15) and AP-5 adaptor protein complex. "
            "Role in lysosomal tubule reformation and autophagosome recycling. "
            "LOF → lysosomal storage dysfunction → glycolipid accumulation → axonal degeneration "
            "with length-dependent corticospinal + peripheral axon loss. "
            "MUTATION SPECTRUM: truncating (frameshift/nonsense) predominate (~85%); "
            "40 coding exons; large gene → many private mutations. "
            "High prevalence in Turkish, North-African, and other consanguineous populations. "
            "15q14; OMIM gene 610844."
        ),
        "hsp_group": "AR Complex HSP — SPG11 (most common AR complex HSP)",
        "subtype": "SPG11 — AR Complex HSP; Spatacsin; Thin Corpus Callosum PATHOGNOMONIC",
        "locus": "15q14", "omim_gene": 610844, "omim_disease": 604360,
        "inheritance": "Autosomal Recessive (AR). Biallelic LOF. Parents: obligate carriers. Consanguinity common.",
        "seed_offset": 3,
        "onset_range_y": (5.0, 25.0),
        "gender": "both",
        "severity_weights": [0.15, 0.45, 0.40],
        "hsp_type": "complex",
        "tcc_prob": 0.85,
        "cognitive_prob": 0.80,
        "cerebellar_prob": 0.40,
        "optic_prob": 0.10,
        "maculopathy_prob": 0.05,
        "epilepsy_prob": 0.20,
        "peripheral_neuropathy_prob": 0.50,
        "phenotype": (
            "COMPLEX AR HSP: spastic paraplegia + thin corpus callosum (TCC) + cognitive impairment + amyotrophy. "
            "Onset: typically <20 y (mean ~15 y). "
            "MRI BRAIN: THIN CORPUS CALLOSUM (TCC) — PATHOGNOMONIC FINDING IN CONTEXT. "
            "White matter changes (periventricular) in ~50%. Cortical atrophy in advanced disease. "
            "COGNITIVE IMPAIRMENT: present in ~80%; mild-moderate learning disability; dysarthria; "
            "frontal lobe dysfunction. "
            "AMYOTROPHY: distal lower-limb wasting; peripheral neuropathy (mixed axonal) in ~50%. "
            "CEREBELLAR SIGNS: cerebellar ataxia in ~40%. "
            "EPILEPSY: ~20%. "
            "PROGRESSION: Rapid — most patients require wheelchair by 30s. "
            "DISTINCT FROM SPG15: SPG11 TCC more consistent; SPG15 Kjellin = maculopathy additional. "
            "MANAGEMENT: early physiotherapy; baclofen; antiepileptic drugs if seizures; "
            "cognitive support; multidisciplinary. No disease-modifying treatment."
        ),
        "disease": (
            "SPG11: most common AR complex HSP globally; prevalence especially high in North Africa, Turkey. "
            "Natural history: progressive course; most wheelchair-dependent by 3rd–4th decade. "
            "Diagnosis: MRI (TCC), WES (large gene — NGS mandatory). "
            "Management: multidisciplinary; physiotherapy; baclofen; cognitive-behavioural support; "
            "urology (bladder); epilepsy management if present."
        ),
        "treatment_options": [
            "Oral baclofen — spasticity (high doses often required)",
            "Intrathecal baclofen pump — severe spasticity (earlier consideration than AD forms)",
            "Physiotherapy — maintain ambulation as long as possible",
            "Antiepileptic drugs (LEV/LTG) — if seizures present",
            "Cognitive support / educational assistance",
            "Botulinum toxin A — lower-limb spasticity",
            "Wheelchair assessment — early; most require by 3rd decade",
            "Urology — bladder management (anticholinergics / CIC)",
        ],
        "outcome_options": [
            "Progressive — wheelchair by 3rd decade; cognitive impairment; baclofen + antiepileptics",
            "Severe — early wheelchair; moderate cognitive deficit; dysarthria; baclofen pump",
            "Moderate — ambulant with aids into 4th decade; cognitive support; baclofen",
            "Progressive — TCC on MRI; onset 15 y; wheelchair early 30s; seizures controlled LEV",
        ],
        "key_ddx": [
            "SPG15/ZFYVE26 — AR complex; TCC (less consistent); Kjellin maculopathy DISTINCTIVE",
            "AP-4-related HSP (SPG47/50/51/52) — AR complex; TCC; intellectual disability; similar phenotype",
            "SPG4/SPAST — AD pure; no TCC; no cognitive impairment; later onset usually",
            "Juvenile ALS — UMN+LMN; rapid; no TCC; SOD1/FUS/TDP43",
            "Krabbe disease (juvenile) — lysosomal; CSF protein elevated; ARSA-MLD overlap",
        ],
    },
    # ── SPG7 — Paraplegin ────────────────────────────────────────────────────
    {
        "gene": "SPG7", "protein": "Paraplegin",
        "alias": "SPG7 (OMIM #607259); AR; 795 aa; mitochondrial m-AAA metalloprotease; cerebellar ataxia PROMINENT; COX-negative fibres; 16q24.3",
        "aa": "795 aa", "kDa": "88 kDa",
        "gene_class": (
            "Paraplegin — mitochondrial m-AAA metalloprotease; inner mitochondrial membrane (IMS face); "
            "forms heteromeric complex with AFG3L2 (m-AAA protease subunit, also mutated in SCA28). "
            "Functions: proteolysis of misfolded mitochondrial proteins; ribosomal biogenesis "
            "(mitochondrial ribosomes); processing of key substrates including MrpL32, OPA1. "
            "LOF → impaired mitochondrial quality control → mitochondrial dysfunction → "
            "energy failure in long corticospinal and cerebellar axons. "
            "MUTATION SPECTRUM: missense + truncating (mixed); p.Ala510Val Mediterranean founder. "
            "16q24.3; OMIM gene 602783."
        ),
        "hsp_group": "AR Complex/Mitochondrial HSP — SPG7",
        "subtype": "SPG7 — Spastic Ataxia; Paraplegin m-AAA Protease; Mitochondrial; COX-negative Fibres",
        "locus": "16q24.3", "omim_gene": 602783, "omim_disease": 607259,
        "inheritance": "Autosomal Recessive (AR). Biallelic. Compound heterozygous common. Mediterranean founder p.Ala510Val.",
        "seed_offset": 4,
        "onset_range_y": (20.0, 60.0),
        "gender": "both",
        "severity_weights": [0.20, 0.45, 0.35],
        "hsp_type": "complex_mitochondrial",
        "tcc_prob": 0.15,
        "cognitive_prob": 0.15,
        "cerebellar_prob": 0.85,
        "optic_prob": 0.30,
        "maculopathy_prob": 0.0,
        "epilepsy_prob": 0.08,
        "peripheral_neuropathy_prob": 0.40,
        "phenotype": (
            "SPASTIC-ATAXIA SYNDROME — CEREBELLAR ATAXIA OFTEN EQUALS OR EXCEEDS SPASTICITY. "
            "Onset: typically adult (20-60 y); mean 30-40 y. "
            "GAIT: ataxic-spastic gait (tandem gait severely impaired; positive Romberg in some). "
            "CEREBELLAR SIGNS: dysmetria, dysdiadochokinesia, scanning speech, nystagmus (80-85%). "
            "SPASTICITY: lower limb; Babinski+; hyperreflexia. "
            "OPTIC NEUROPATHY: 30% (optic atrophy, reduced visual acuity); mandatory ophthalmology. "
            "DYSPHAGIA: 30% (medullary/cerebellar involvement). "
            "HEARING LOSS: ~20%. "
            "MUSCLE BIOPSY: COX-negative fibres + ragged-red fibres (RRF) — mitochondrial myopathy. "
            "PERIPHERAL NEUROPATHY: axonal, predominantly motor, ~40%. "
            "MEDITERRANEAN FOUNDER: p.Ala510Val — Spain, Italy, North Africa; heterozygous carriers "
            "occasionally show mild late-onset spasticity (pseudo-dominant). "
            "MANAGEMENT: mitochondrial supportive — avoid valproate (use caution); CoQ10 (Level C); "
            "riboflavin; baclofen; physiotherapy. Annual ophthalmology + audiology. "
            "MRI brain: cerebellar atrophy; possible brainstem atrophy."
        ),
        "disease": (
            "SPG7: ~1-2% of all HSP; more common in southern Europe (Mediterranean founder). "
            "Mitochondrial basis differentiates from pure cytoskeletal HSPs. "
            "Muscle biopsy: COX-negative fibres + RRF in ~50-70%. "
            "Mitochondrial respiratory chain: combined OXPHOS defects (CI+IV) in some. "
            "Diagnosis: WES; muscle biopsy; ophthalmology; audiology; cardiac echo (cardiomyopathy rare). "
            "AFG3L2 mutations (SCA28) — allelic heterogeneity at same complex; important DDx."
        ),
        "treatment_options": [
            "Oral baclofen — spasticity",
            "Physiotherapy — ataxia rehabilitation + gait training",
            "CoQ10 (ubiquinol) — mitochondrial support (Level C; empiric)",
            "Riboflavin B2 — mitochondrial support (empiric)",
            "Ophthalmology follow-up — optic neuropathy; low-vision aids",
            "Audiology — hearing aids if sensorineural hearing loss",
            "Dysphagia: speech-language pathology (SLP) assessment; PEG if severe",
            "Valproate: use CAUTION (mitochondrial disease — not absolute CI but avoid if possible)",
        ],
        "outcome_options": [
            "Progressive — ataxia > spasticity; cerebellar atrophy; optic neuropathy; gait aid by 5th decade",
            "Moderate — spastic-ataxic gait; physiotherapy; baclofen; ophthalmology monitoring",
            "Slow — adult onset 40s; ambulant with rollator; CoQ10 empiric; stable 10 y follow-up",
            "Progressive — Mediterranean founder p.Ala510Val; cerebellar atrophy; optic atrophy; wheelchair 6th decade",
        ],
        "key_ddx": [
            "SCA28/AFG3L2 — allelic m-AAA complex; AD cerebellar ataxia; SCA28 = pure ataxia; SPG7 = spastic-ataxia",
            "Friedreich ataxia (FXN/FRDA) — AR; GAA repeat; HCM 80%; loss of reflexes (areflexia) vs SPG7 hyperreflexia",
            "SCA3/ATXN3 — AD; CAG repeat; RLS 60-80%; bulging eyes; no COX-negative fibres",
            "MSA-C (multiple system atrophy cerebellar) — acquired; adult onset; autonomic failure; no family history",
            "Chronic progressive external ophthalmoplegia (CPEO) — mtDNA disorder; ptosis; ophthalmoplegia",
            "SCA7/ATXN7 — AD; retinal dystrophy DISTINCTIVE (not in SPG7); CAG repeat; extreme anticipation",
        ],
    },
    # ── CYP7B1 — SPG5A ───────────────────────────────────────────────────────
    {
        "gene": "CYP7B1", "protein": "Oxysterol 7α-hydroxylase",
        "alias": "SPG5A (OMIM #270800); AR; 506 aa; oxysterol 7α-hydroxylase; elevated 25-OHC + 27-OHC PATHOGNOMONIC; LOVASTATIN TREATABLE; 8q12.3",
        "aa": "506 aa", "kDa": "57 kDa",
        "gene_class": (
            "CYP7B1 — cytochrome P450 family 7 subfamily B member 1; oxysterol 7α-hydroxylase; "
            "microsomal enzyme; ER-resident; catalyses 7α-hydroxylation of oxysterols and neurosteroids: "
            "25-hydroxycholesterol (25-OHC), 27-hydroxycholesterol (27-OHC), dehydroepiandrosterone (DHEA). "
            "LOF → toxic accumulation of 25-OHC and 27-OHC in CNS → corticospinal tract axon degeneration. "
            "BIOMARKER: plasma 25-OHC AND 27-OHC BOTH ELEVATED — PATHOGNOMONIC. "
            "TREATMENT TARGET: lovastatin (inhibits HMG-CoA reductase → reduces 27-OHC precursor cholesterol synthesis) "
            "→ reduces circulating 27-OHC → clinical stabilisation (Level B evidence). "
            "Also: atorvastatin / pravastatin. Oxysterol monitoring guides treatment. "
            "8q12.3; OMIM gene 603711."
        ),
        "hsp_group": "AR Pure HSP — SPG5A (treatable oxysterol disorder)",
        "subtype": "SPG5A — Pure HSP; CYP7B1; Elevated Oxysterols; LOVASTATIN TREATABLE",
        "locus": "8q12.3", "omim_gene": 603711, "omim_disease": 270800,
        "inheritance": "Autosomal Recessive (AR). Biallelic LOF. Consanguineous families common.",
        "seed_offset": 5,
        "onset_range_y": (5.0, 50.0),
        "gender": "both",
        "severity_weights": [0.25, 0.50, 0.25],
        "hsp_type": "pure_treatable",
        "tcc_prob": 0.05,
        "cognitive_prob": 0.05,
        "cerebellar_prob": 0.15,
        "optic_prob": 0.05,
        "maculopathy_prob": 0.0,
        "epilepsy_prob": 0.03,
        "peripheral_neuropathy_prob": 0.20,
        "phenotype": (
            "PURE (OCCASIONALLY COMPLEX) SPASTIC PARAPLEGIA WITH TREATABLE BIOCHEMISTRY. "
            "Onset: variable (5-50 y); mean ~20 y; slowly progressive. "
            "Spastic paraplegia: progressive lower-limb spasticity; Babinski+; hyperreflexia; "
            "bladder urgency. Generally pure form (no cognitive impairment, no TCC). "
            "Mild cerebellar signs in some cases (~15%). "
            "BIOMARKER DIAGNOSTIC: plasma 25-hydroxycholesterol AND 27-hydroxycholesterol BOTH ELEVATED — "
            "specific and sensitive; oxysterol panel MANDATORY in AR pure HSP workup. "
            "TREATMENT: LOVASTATIN 20-40 mg/day (or atorvastatin/pravastatin) reduces 27-OHC synthesis; "
            "plasma oxysterols normalise or decrease; clinical stabilisation/improvement reported (Level B). "
            "Monitor: CK (myopathy risk); LFTs (hepatotoxicity); fasting lipid panel. "
            "AVOID: high-dose statins (rhabdomyolysis risk in mitochondrial overlap). "
            "Physiotherapy; baclofen; AFO as adjuncts. "
            "CRITICAL TEACHING POINT: CYP7B1/SPG5A is the ONLY biochemically treatable pure HSP (2026)."
        ),
        "disease": (
            "SPG5A/CYP7B1: AR pure HSP; ~1-3% of AR HSP. "
            "Prevalence: under-recognised globally. "
            "Diagnosis: plasma oxysterol panel (25-OHC + 27-OHC both elevated) + WES. "
            "Treatment: statin therapy (Level B) — possibly the most actionable genetic HSP diagnosis. "
            "Monitoring: oxysterols, CK, LFTs every 6 months on statin. "
            "Natural history without treatment: progressive over decades."
        ),
        "treatment_options": [
            "LOVASTATIN 20-40 mg/day — FIRST-LINE (reduces 27-OHC synthesis; Level B)",
            "Atorvastatin / pravastatin — alternatives if lovastatin not tolerated",
            "Plasma oxysterol monitoring — 25-OHC + 27-OHC every 6 months",
            "CK + LFT monitoring — statin safety (every 6 months)",
            "Oral baclofen — spasticity symptom management",
            "Physiotherapy — gait training, stretching",
            "AFO — foot-drop support",
        ],
        "outcome_options": [
            "Treated — lovastatin started; oxysterols normalised; spasticity stabilised; ambulation maintained",
            "Progressive (untreated) — slowly worsening spasticity; AFO + baclofen; oxysterol elevation confirmed late",
            "Treated — early lovastatin; clinical improvement (rare but reported); monitoring oxysterols 6-monthly",
            "Diagnosed late — moderate spasticity; lovastatin stabilises; physiotherapy + AFO + baclofen",
        ],
        "key_ddx": [
            "CTX (cerebrotendinous xanthomatosis / CYP27A1) — xanthomas; cataracts; diarrhoea; elevated cholestanol (not oxysterols)",
            "SPG4/SPAST — AD pure HSP; normal oxysterols; most common",
            "Other AR pure HSP — normal oxysterols; WES needed",
            "Niemann-Pick type C (NPC1) — NPC1 gene; vertical gaze palsy; cataplexy; filipin staining; normal oxysterols (NPC pattern differs)",
            "SPG11 — AR complex; TCC; cognitive; oxysterols normal",
            "Vitamin B12 / copper deficiency myelopathy — acquired; reversible; screen metabolites",
        ],
    },
    # ── ZFYVE26 — SPG15 ──────────────────────────────────────────────────────
    {
        "gene": "ZFYVE26", "protein": "Spastizin (ZFYVE26)",
        "alias": "SPG15 (OMIM #270700); AR; 2539 aa; FYVE-domain; autophagosome maturation; Kjellin syndrome; thin corpus callosum; 14q24.1",
        "aa": "2539 aa", "kDa": "285 kDa",
        "gene_class": (
            "Spastizin (ZFYVE26) — FYVE zinc-finger domain protein; binds phosphatidylinositol-3-phosphate (PI3P) "
            "on endosomal/autophagosomal membranes. "
            "Forms complex with spatacsin (SPG11) and AP-5 adaptor protein complex. "
            "Role: autophagosome maturation, lysosomal tubule reformation, endo-lysosomal trafficking. "
            "LOF → lysosomal storage dysfunction → axonal degeneration (shared pathway with SPG11). "
            "MUTATION SPECTRUM: truncating LOF; large gene. Consanguineous North-African, Middle-Eastern. "
            "KJELLIN SYNDROME: SPG15 + pigmentary maculopathy (retinal degeneration) — DISTINCTIVE. "
            "14q24.1; OMIM gene 610033."
        ),
        "hsp_group": "AR Complex HSP — SPG15 (Kjellin syndrome; spastizin)",
        "subtype": "SPG15 — AR Complex HSP; Kjellin Syndrome (TCC + Pigmentary Maculopathy DISTINCTIVE); Spastizin",
        "locus": "14q24.1", "omim_gene": 610033, "omim_disease": 270700,
        "inheritance": "Autosomal Recessive (AR). Biallelic LOF. Consanguineous common.",
        "seed_offset": 6,
        "onset_range_y": (8.0, 25.0),
        "gender": "both",
        "severity_weights": [0.20, 0.45, 0.35],
        "hsp_type": "complex",
        "tcc_prob": 0.65,
        "cognitive_prob": 0.70,
        "cerebellar_prob": 0.35,
        "optic_prob": 0.0,
        "maculopathy_prob": 0.55,
        "epilepsy_prob": 0.18,
        "peripheral_neuropathy_prob": 0.45,
        "phenotype": (
            "AR COMPLEX HSP WITH KJELLIN SYNDROME OVERLAP. "
            "Onset: childhood to early adulthood (8-25 y); mean ~15 y. "
            "SPASTIC PARAPLEGIA: progressive lower-limb spasticity + amyotrophy. "
            "THIN CORPUS CALLOSUM (TCC): present in ~65% (less consistent than SPG11). "
            "COGNITIVE IMPAIRMENT: mild-moderate in ~70%; dysarthria; frontal lobe dysfunction. "
            "KJELLIN SYNDROME = SPG15 + PIGMENTARY MACULOPATHY: "
            "  Macular degeneration / pigmentary changes on fundoscopy in ~55% — DISTINCTIVE. "
            "  Visual acuity reduction; ERG (electroretinography) abnormalities. "
            "  Mandatory ophthalmology with ERG at diagnosis and annually. "
            "CEREBELLAR SIGNS: ~35%. "
            "AMYOTROPHY + PERIPHERAL NEUROPATHY: ~45% (overlap with SPG11). "
            "EPILEPSY: ~18%. "
            "PROGRESSION: progressive; often wheelchair 3rd–4th decade. "
            "DDx SPG11: SPG15 has maculopathy (Kjellin) as DISTINCTIVE extra feature. "
            "MANAGEMENT: ophthalmology (ERG) mandatory; baclofen; antiepileptics; physiotherapy."
        ),
        "disease": (
            "SPG15/ZFYVE26 (Kjellin syndrome): rare AR complex HSP. "
            "Originally described as 'Kjellin syndrome' (spastic paraplegia + macular degeneration). "
            "Genetic basis: SPG15 biallelic mutations. "
            "Diagnosis: MRI (TCC), ophthalmology + ERG (maculopathy), WES. "
            "Management: ophthalmology follow-up (low-vision aids if visual loss); "
            "baclofen; antiepileptics; physiotherapy; multidisciplinary."
        ),
        "treatment_options": [
            "Ophthalmology + ERG — at diagnosis and annually (maculopathy monitoring)",
            "Low-vision aids / visual rehabilitation — if macular degeneration progresses",
            "Oral baclofen — spasticity",
            "Antiepileptic drugs (LEV/LTG) — if seizures",
            "Physiotherapy — gait; maintain ambulation",
            "Intrathecal baclofen pump — severe refractory spasticity",
            "Cognitive / educational support",
            "Urology — bladder management",
        ],
        "outcome_options": [
            "Progressive — Kjellin phenotype; maculopathy + TCC; wheelchair 3rd decade; LEV seizure control",
            "Moderate — spasticity + mild maculopathy; ambulation with aids; ophthalmology follow-up",
            "Progressive — TCC; cognitive impairment; ERG maculopathy; physiotherapy + baclofen",
            "Severe — early onset; wheelchair 30s; low vision; baclofen pump; LEV",
        ],
        "key_ddx": [
            "SPG11/Spatacsin — most common AR complex HSP; TCC (more consistent); NO maculopathy",
            "Batten disease (CLN3/NCL3) — AR; juvenile; maculopathy + seizures + cognitive; vacuolated lymphocytes",
            "SCA7/ATXN7 — AD; retinal dystrophy + ataxia; CAG repeat; extreme anticipation",
            "Refsum disease (PHYH) — AR; ichthyosis + retinitis pigmentosa + neuropathy + elevated phytanic acid; treatable",
            "NBIA (PANK2) — AR; eye-of-tiger sign MRI; iron accumulation; no maculopathy",
            "SPG4/SPAST — AD; pure; no maculopathy; no TCC; most common HSP",
        ],
    },
    # ── KIF1A — SPG30 ────────────────────────────────────────────────────────
    {
        "gene": "KIF1A", "protein": "Kinesin-like Protein KIF1A (Kinesin-3)",
        "alias": "SPG30 (OMIM #610357); AR/de-novo-AD; 1756 aa; kinesin-3 motor; anterograde axonal transport; severe complex HSP; intellectual disability; 2q37.3",
        "aa": "1756 aa", "kDa": "202 kDa",
        "gene_class": (
            "KIF1A — kinesin-3 (KIF1 subfamily) motor protein; homodimeric anterograde axonal transport; "
            "transports synaptic vesicle precursors (SVPs) and dense-core vesicles (DCVs) from soma "
            "to distal axon along microtubules (plus-end-directed, ATP-dependent). "
            "Critical in long sensory + motor + corticospinal axons. "
            "MUTATION SPECTRUM: "
            "  AR biallelic — LOF; milder SPG30 (hereditary spastic paraplegia with cognitive impairment). "
            "  De-novo heterozygous — dominant-negative motor domain missense; "
            "    clustered at microtubule-binding surface (loop L8, switch regions); "
            "    SEVERE COMPLEX NEURODEVELOPMENTAL DISORDER. "
            "De-novo dominant mutations disproportionately common — most severe clinical presentations. "
            "2q37.3; OMIM gene 601255."
        ),
        "hsp_group": "AR/De-novo Complex HSP — SPG30 (KIF1A; severe; axonal transport)",
        "subtype": "SPG30 — Complex HSP/NAND; KIF1A Kinesin Motor; De-novo Dominant Most Severe; Intellectual Disability",
        "locus": "2q37.3", "omim_gene": 601255, "omim_disease": 610357,
        "inheritance": "AR biallelic (SPG30 — milder) OR de-novo heterozygous dominant (KIF1A-associated neurological disorder / KAND — severe).",
        "seed_offset": 7,
        "onset_range_y": (0.0, 5.0),
        "gender": "both",
        "severity_weights": [0.10, 0.30, 0.60],
        "hsp_type": "complex_severe",
        "tcc_prob": 0.50,
        "cognitive_prob": 0.90,
        "cerebellar_prob": 0.65,
        "optic_prob": 0.40,
        "maculopathy_prob": 0.0,
        "epilepsy_prob": 0.45,
        "peripheral_neuropathy_prob": 0.30,
        "phenotype": (
            "SEVERE COMPLEX HSP / KIF1A-ASSOCIATED NEUROLOGICAL DISORDER (KAND). "
            "Onset: INFANCY to early childhood (0-5 y); global developmental delay. "
            "SPASTIC PARAPLEGIA: severe lower-limb spasticity; Babinski+; hyperreflexia. "
            "INTELLECTUAL DISABILITY: SEVERE in de-novo cases (90%); moderate in AR biallelic. "
            "CEREBELLAR ATROPHY: 65%; ataxia + cerebellar hypoplasia on MRI. "
            "OPTIC ATROPHY: 40%; visual impairment. "
            "EPILEPSY: 45%; infantile spasms, myoclonic, focal seizures. "
            "THIN CORPUS CALLOSUM (TCC): 50%. "
            "PERIPHERAL NEUROPATHY: hereditary sensorimotor neuropathy component in AR form. "
            "MRI BRAIN: cerebellar atrophy; TCC; white matter changes; brain atrophy. "
            "PROGRESSIVE: relentless deterioration in de-novo dominant; slower in AR biallelic. "
            "DE-NOVO RATE: HIGH — most severe cases sporadic; mandatory parental testing. "
            "MANAGEMENT: antiepileptic drugs (LEV, VPA if tolerated, ACTH for spasms); "
            "baclofen; physiotherapy; vision assessment; multidisciplinary neurodevelopmental care. "
            "No disease-modifying treatment (2026); trials ongoing (axonal transport modulators)."
        ),
        "disease": (
            "KIF1A/SPG30 (KAND): rare but increasingly recognised; de-novo mutations underrecognised. "
            "Natural history: severe progressive neurological disease; most non-ambulant by teen years. "
            "Diagnosis: WES/WGS mandatory (especially de-novo); parental testing essential. "
            "Management: multidisciplinary — epilepsy, ophthalmology, physiotherapy, neurodevelopment, "
            "feeding (PEG in severe), spasticity management."
        ),
        "treatment_options": [
            "Antiepileptic drugs — LEV first-line; ACTH for infantile spasms; VPA (if mitochondrial excluded)",
            "Oral baclofen — severe spasticity",
            "Intrathecal baclofen pump — refractory lower-limb spasticity",
            "Physiotherapy — early intensive; maintain function",
            "Botulinum toxin A — lower-limb spasticity (hip adductors, gastrocnemius)",
            "Ophthalmology — optic atrophy; low-vision aids",
            "Feeding support — PEG if severe dysphagia/aspiration",
            "Augmentative and alternative communication (AAC) — non-verbal patients",
        ],
        "outcome_options": [
            "Severe de-novo — non-ambulant by 5 y; severe ID; epilepsy; cerebellar atrophy; baclofen pump + LEV",
            "Progressive de-novo — ambulant with aids; moderate-severe ID; optic atrophy; baclofen + antiepileptics",
            "AR biallelic (milder SPG30) — ambulant; moderate ID; slower progression; physiotherapy",
            "Severe de-novo — infantile spasms → severe epilepsy; global DD; cerebellar hypoplasia; PEG + LEV",
        ],
        "key_ddx": [
            "Pelizaeus-Merzbacher disease (PLP1) — X-linked; nystagmus from birth; hypomyelination MRI",
            "SPG11 — AR complex; TCC; cognitive; no cerebellar hypoplasia; onset slightly later",
            "West syndrome (isolated) — infantile spasms; ACTH response; no progressive spastic-ataxia",
            "Ataxic cerebral palsy — perinatal; non-progressive; no family history",
            "NALD/ZSD (PEX1) — peroxisomal; VLCFA elevated; different MRI pattern",
            "Pitt-Hopkins syndrome (TCF4) — ID; breathing abnormality; distinctive facies; no motor neuron disease",
        ],
    },
]

# ── Patient Cohort Generator ──────────────────────────────────────────────────

def _gen_cohort(n_per_gene: int = 40) -> list:
    """Generate realistic synthetic patient cohort for the HSP Atlas (8 genes × 40 = 320 patients)."""
    patients = []
    for gene_data in HSP_GENES:
        rng = random.Random(SEED_BASE + gene_data["seed_offset"])
        for i in range(n_per_gene):
            onset = round(rng.uniform(*gene_data["onset_range_y"]), 1)
            if gene_data["gender"] == "both":
                sex = rng.choice(["Male", "Female"])
            else:
                sex = gene_data["gender"].title()
            sev_label = rng.choices(
                ["Mild", "Moderate", "Severe"],
                weights=gene_data["severity_weights"], k=1
            )[0]
            # Clinical features
            tcc = rng.random() < gene_data["tcc_prob"]
            cognitive = rng.random() < gene_data["cognitive_prob"]
            cerebellar = rng.random() < gene_data["cerebellar_prob"]
            optic = rng.random() < gene_data["optic_prob"]
            maculopathy = rng.random() < gene_data["maculopathy_prob"]
            epilepsy = rng.random() < gene_data["epilepsy_prob"]
            periph_n = rng.random() < gene_data["peripheral_neuropathy_prob"]
            outcome = rng.choice(gene_data["outcome_options"])
            treatment = rng.choice(gene_data["treatment_options"])
            patients.append({
                "patient_id": f"{gene_data['gene']}-{i+1:03d}",
                "gene": gene_data["gene"],
                "protein": gene_data["protein"],
                "disease_subtype": gene_data["subtype"],
                "onset_age_y": onset,
                "sex": sex,
                "severity": sev_label,
                "hsp_type": gene_data["hsp_type"],
                "thin_corpus_callosum": tcc,
                "cognitive_impairment": cognitive,
                "cerebellar_signs": cerebellar,
                "optic_atrophy": optic,
                "pigmentary_maculopathy": maculopathy,
                "epilepsy": epilepsy,
                "peripheral_neuropathy": periph_n,
                "outcome_note": outcome,
                "primary_treatment": treatment,
                "locus": gene_data["locus"],
            })
    return patients

# ── Public API ────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)
    gene_counts = {}
    for p in patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1
    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    tcc_n = sum(1 for p in patients if p["thin_corpus_callosum"])
    cog_n = sum(1 for p in patients if p["cognitive_impairment"])
    cer_n = sum(1 for p in patients if p["cerebellar_signs"])
    optic_n = sum(1 for p in patients if p["optic_atrophy"])
    mac_n = sum(1 for p in patients if p["pigmentary_maculopathy"])
    epi_n = sum(1 for p in patients if p["epilepsy"])
    pn_n = sum(1 for p in patients if p["peripheral_neuropathy"])
    for p in patients:
        sev[p["severity"]] += 1
    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 1)
    return {
        "atlas": "HSP-Atlas",
        "full_name": "Complete 8-Gene Hereditary Spastic Paraplegia (HSP) Atlas",
        "subtitle": "SPAST·ATL1·REEP1·SPG11·SPG7·CYP7B1·ZFYVE26·KIF1A — 320 patients (8×40, seeds 1006–1013)",
        "description": (
            "Comprehensive atlas of the 8 most clinically important Hereditary Spastic Paraplegia (HSP) genes. "
            "Covers AD pure HSP (SPAST/ATL1/REEP1), AR complex HSP (SPG11/ZFYVE26/KIF1A), "
            "mitochondrial spastic ataxia (SPG7), and the sole biochemically treatable HSP (CYP7B1/SPG5A). "
            "HSP = progressive corticospinal tract degeneration → lower-limb spastic weakness. "
            "Pure (uncomplicated) vs Complex (complicated) classification. "
            "CYP7B1/SPG5A: oxysterol biomarker + lovastatin treatment — the only actionable biochemical HSP."
        ),
        "total_patients": n,
        "genes_covered": len(HSP_GENES),
        "patients_per_gene": 40,
        "seed_range": "1006–1013",
        "gene_list": [g["gene"] for g in HSP_GENES],
        "inheritance_breakdown": {
            "AD_pure": ["SPAST", "ATL1", "REEP1"],
            "AR_complex": ["SPG11", "SPG7", "CYP7B1", "ZFYVE26"],
            "AR_or_de_novo_AD": ["KIF1A"],
        },
        "hsp_type_breakdown": {
            "Pure HSP (AD)": 120,  # SPAST + ATL1 + REEP1
            "Pure HSP (AR treatable)": 40,  # CYP7B1
            "Complex HSP (AR)": 120,  # SPG11 + ZFYVE26 + KIF1A
            "Spastic-Ataxia (AR/mitochondrial)": 40,  # SPG7
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "complex_features_prevalence": {
            "thin_corpus_callosum_pct": round(100 * tcc_n / n, 1),
            "cognitive_impairment_pct": round(100 * cog_n / n, 1),
            "cerebellar_signs_pct": round(100 * cer_n / n, 1),
            "optic_atrophy_pct": round(100 * optic_n / n, 1),
            "pigmentary_maculopathy_pct": round(100 * mac_n / n, 1),
            "epilepsy_pct": round(100 * epi_n / n, 1),
            "peripheral_neuropathy_pct": round(100 * pn_n / n, 1),
        },
        "key_teaching_points": [
            "SPAST/SPG4: most common HSP (40%); AD; pure; incomplete penetrance (~75%); highly variable",
            "ATL1/SPG3A: 2nd most common AD HSP; CHILDHOOD ONSET; SLOW/NON-PROGRESSIVE — key DDx SPG4",
            "SPG11/Spatacsin: most common AR COMPLEX HSP; THIN CORPUS CALLOSUM PATHOGNOMONIC; cognitive impairment",
            "CYP7B1/SPG5A: SOLE TREATABLE HSP — plasma oxysterols (25-OHC + 27-OHC) elevated; LOVASTATIN Rx Level B",
            "SPG7/Paraplegin: SPASTIC ATAXIA; cerebellar > spasticity; COX-negative fibres; mitochondrial m-AAA protease",
            "ZFYVE26/SPG15: KJELLIN SYNDROME = TCC + PIGMENTARY MACULOPATHY (DISTINCTIVE); ERG mandatory",
            "KIF1A/SPG30: SEVERE COMPLEX HSP; de-novo dominant disproportionately common; kinesin-3 axonal transport",
            "REEP1/SPG31: 3rd most common AD HSP; dHMN overlap; de-novo rate 20%; ER-shaping protein",
        ],
        "drug_alerts": [
            "Valproate (VPA): CAUTION in SPG7 (mitochondrial basis); generally SAFE in pure AD HSP forms",
            "Baclofen: FIRST-LINE for all HSP spasticity; consider intrathecal pump for severe cases",
            "Lovastatin/statins: MANDATORY trial in CYP7B1/SPG5A (oxysterol reduction); monitor CK + LFTs",
            "Botulinum toxin A: effective for focal spasticity (hip adductors, gastrocnemius) all HSP types",
            "Vincristine: AVOID in HSP patients with peripheral neuropathy component (SPG7, KIF1A, REEP1)",
        ],
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    gene_profiles = []
    for gene_data in HSP_GENES:
        gene_pts = [p for p in patients if p["gene"] == gene_data["gene"]]
        n = len(gene_pts)
        tcc = sum(1 for p in gene_pts if p["thin_corpus_callosum"])
        cog = sum(1 for p in gene_pts if p["cognitive_impairment"])
        cer = sum(1 for p in gene_pts if p["cerebellar_signs"])
        optic = sum(1 for p in gene_pts if p["optic_atrophy"])
        mac = sum(1 for p in gene_pts if p["pigmentary_maculopathy"])
        epi = sum(1 for p in gene_pts if p["epilepsy"])
        pn = sum(1 for p in gene_pts if p["peripheral_neuropathy"])
        sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for p in gene_pts:
            sev[p["severity"]] += 1
        gene_profiles.append({
            "gene": gene_data["gene"],
            "protein": gene_data["protein"],
            "alias": gene_data["alias"],
            "locus": gene_data["locus"],
            "omim_gene": gene_data["omim_gene"],
            "omim_disease": gene_data["omim_disease"],
            "inheritance": gene_data["inheritance"],
            "hsp_group": gene_data["hsp_group"],
            "hsp_type": gene_data["hsp_type"],
            "aa": gene_data["aa"],
            "kDa": gene_data["kDa"],
            "gene_class": gene_data["gene_class"],
            "phenotype": gene_data["phenotype"],
            "disease": gene_data["disease"],
            "treatment_options": gene_data["treatment_options"],
            "key_ddx": gene_data["key_ddx"],
            "onset_range_y": list(gene_data["onset_range_y"]),
            "n_patients": n,
            "severity_distribution": {
                "mild_pct": round(100 * sev["Mild"] / n, 1),
                "moderate_pct": round(100 * sev["Moderate"] / n, 1),
                "severe_pct": round(100 * sev["Severe"] / n, 1),
            },
            "complex_features": {
                "thin_corpus_callosum_pct": round(100 * tcc / n, 1),
                "cognitive_impairment_pct": round(100 * cog / n, 1),
                "cerebellar_signs_pct": round(100 * cer / n, 1),
                "optic_atrophy_pct": round(100 * optic / n, 1),
                "pigmentary_maculopathy_pct": round(100 * mac / n, 1),
                "epilepsy_pct": round(100 * epi / n, 1),
                "peripheral_neuropathy_pct": round(100 * pn / n, 1),
            },
            "sample_patients": gene_pts[:3],
        })
    return {
        "atlas": "HSP-Atlas",
        "genes": gene_profiles,
        "total_patients": len(patients),
    }


def get_definitions() -> dict:
    return {
        "atlas": "HSP-Atlas",
        "definitions": [
            {
                "term": "Hereditary Spastic Paraplegia (HSP)",
                "definition": (
                    "Clinically and genetically heterogeneous group of inherited neurological disorders "
                    "characterised by progressive lower-limb spastic weakness due to degeneration of "
                    "corticospinal tract axons. Classified as PURE (uncomplicated) — spasticity ± "
                    "mild urinary urgency ± mild vibration loss — or COMPLEX (complicated) — "
                    "additional features (cognitive impairment, cerebellar ataxia, peripheral neuropathy, "
                    "thin corpus callosum, optic neuropathy, epilepsy, pigmentary maculopathy). "
                    ">70 SPG loci identified. Prevalence: 1.8-9.6 per 100,000."
                )
            },
            {
                "term": "Spastin (SPAST) / SPG4",
                "definition": (
                    "Most common HSP gene (AD, ~40% of AD HSP). ATP-dependent microtubule-severing "
                    "AAA-ATPase. Haploinsufficiency → reduced microtubule dynamics in distal corticospinal "
                    "tract axons (>1 m length). Incomplete penetrance (~75%). Pure HSP; highly variable "
                    "within families. No disease-modifying treatment (2026)."
                )
            },
            {
                "term": "Atlastin-1 (ATL1) / SPG3A",
                "definition": (
                    "2nd most common AD HSP (~10%). Dynamin-like GTPase; ER membrane fusion; tubular ER "
                    "network formation. EARLY ONSET (typically <10 y; often childhood/infancy). "
                    "KEY DISTINCTION: slow or non-progressive course after initial presentation — "
                    "contrast with SPAST/SPG4 which is progressive. Dominant-negative or haploinsufficiency."
                )
            },
            {
                "term": "Thin Corpus Callosum (TCC)",
                "definition": (
                    "MRI finding (midsagittal view): thin CC body/genu. PATHOGNOMONIC in context of "
                    "AR complex HSP with cognitive impairment and amyotrophy → SPG11 (spatacsin). "
                    "Also seen in SPG15 (ZFYVE26/spastizin) — less consistently. "
                    "TCC + maculopathy = Kjellin syndrome (SPG15). "
                    "TCC + cognitive impairment + young-onset AR HSP → SPG11 diagnosis strongly suggested."
                )
            },
            {
                "term": "Kjellin Syndrome",
                "definition": (
                    "Phenotypic variant of SPG15 (ZFYVE26 biallelic mutations): AR complex HSP with "
                    "PIGMENTARY MACULOPATHY (retinal degeneration) — DISTINCTIVE. "
                    "Originally described clinically before genetic basis known. "
                    "ERG (electroretinography) abnormal in macular involvement. "
                    "Ophthalmology + ERG mandatory at diagnosis and annually. "
                    "TCC ± cognitive impairment also present in majority."
                )
            },
            {
                "term": "CYP7B1 / SPG5A — Treatable HSP",
                "definition": (
                    "Only biochemically treatable pure HSP (2026). CYP7B1 = oxysterol 7α-hydroxylase; "
                    "bile acid synthesis pathway. LOF → accumulation of 25-OHC and 27-OHC (neurotoxic oxysterols). "
                    "DIAGNOSIS: plasma oxysterol panel (25-OH-cholesterol + 27-OH-cholesterol BOTH elevated). "
                    "TREATMENT: lovastatin (or atorvastatin/pravastatin) reduces 27-OHC synthesis via HMG-CoA "
                    "reductase inhibition → clinical stabilisation. Level B evidence. Oxysterol monitoring mandatory."
                )
            },
            {
                "term": "Paraplegin (SPG7) / Spastic Ataxia",
                "definition": (
                    "AR mitochondrial HSP. Paraplegin = m-AAA metalloprotease (inner mitochondrial membrane); "
                    "forms complex with AFG3L2. SPASTIC ATAXIA phenotype — cerebellar ataxia prominent "
                    "(often equals or exceeds spasticity). COX-negative fibres + RRF on muscle biopsy "
                    "(mitochondrial myopathy). Optic neuropathy 30%. Mediterranean founder: p.Ala510Val. "
                    "DDx AFG3L2 (SCA28 — AD ataxia, same complex)."
                )
            },
            {
                "term": "KIF1A / SPG30 / KAND",
                "definition": (
                    "Kinesin-3 motor; anterograde axonal transport of synaptic vesicle precursors. "
                    "AR biallelic = SPG30 (milder); de-novo heterozygous dominant-negative = KIF1A-Associated "
                    "Neurological Disorder (KAND) — severe complex HSP with intellectual disability, "
                    "cerebellar atrophy, optic atrophy, epilepsy. De-novo rate HIGH — most severe cases sporadic. "
                    "Motor domain missense mutations cluster at microtubule-binding surface."
                )
            },
            {
                "term": "Oxysterols (25-OHC / 27-OHC)",
                "definition": (
                    "Oxidised forms of cholesterol: 25-hydroxycholesterol (25-OHC) and "
                    "27-hydroxycholesterol (27-OHC). Normally metabolised by CYP7B1 (7α-hydroxylation). "
                    "CYP7B1/SPG5A LOF → toxic accumulation in CNS. "
                    "Plasma oxysterol panel: BOTH 25-OHC and 27-OHC elevated = PATHOGNOMONIC for SPG5A. "
                    "(Distinguish from CTX = cerebrotendinous xanthomatosis: elevated cholestanol, different pathway.)"
                )
            },
            {
                "term": "Pure vs Complex HSP Classification",
                "definition": (
                    "PURE (uncomplicated) HSP: lower-limb spastic weakness; ± mild urinary urgency; "
                    "± mild vibration reduction in feet. Brain MRI often normal. "
                    "Examples: SPG4 (SPAST), SPG3A (ATL1), SPG31 (REEP1), SPG5A (CYP7B1). "
                    "COMPLEX (complicated) HSP: additional neurological or systemic features. "
                    "Examples: SPG11 (TCC + cognitive), SPG15/Kjellin (TCC + maculopathy), "
                    "SPG7 (cerebellar ataxia), SPG30/KAND (severe ID + epilepsy + cerebellar)."
                )
            },
            {
                "term": "Corticospinal Tract Degeneration",
                "definition": (
                    "Pathological hallmark of HSP. Dying-back axonopathy of the longest neurons in the CNS "
                    "(corticospinal tract axons: >1 m to lumbar/sacral spinal cord). "
                    "Length-dependent vulnerability: distal axons degenerate first → lower-limb symptoms precede "
                    "upper-limb involvement. MRI spine: may show cord atrophy (late). "
                    "Pathogenic mechanisms: microtubule dynamics (SPAST, ATL1, REEP1), lysosomal/autophagy "
                    "(SPG11, ZFYVE26), mitochondrial (SPG7), axonal transport (KIF1A), oxysterols (CYP7B1)."
                )
            },
            {
                "term": "Spasticity Management in HSP",
                "definition": (
                    "No disease-modifying treatment for most HSP forms (2026 — except CYP7B1/SPG5A with statins). "
                    "Symptomatic spasticity management: "
                    "(1) Oral baclofen — FIRST-LINE GABA-B agonist; start low, titrate; monitor sedation/weakness. "
                    "(2) Tizanidine — alpha-2 agonist; caution hepatotoxicity (monitor LFTs). "
                    "(3) Botulinum toxin A — focal lower-limb spasticity (hip adductors, gastrocnemius); every 3 months. "
                    "(4) Intrathecal baclofen (ITB) pump — refractory severe spasticity; surgical implant. "
                    "Physiotherapy MANDATORY for all HSP; AFO for foot-drop/equinus."
                )
            },
        ]
    }


if __name__ == "__main__":
    import json
    print("=== HSP-Atlas Overview ===")
    ov = get_overview()
    print(f"Total patients: {ov['total_patients']}")
    print(f"Genes: {ov['gene_list']}")
    print(f"Mean onset: {ov['mean_onset_age_y']} y")
    print(f"TCC prevalence: {ov['complex_features_prevalence']['thin_corpus_callosum_pct']}%")
    print(f"Cognitive impairment: {ov['complex_features_prevalence']['cognitive_impairment_pct']}%")
    print("\n=== Breakdown (gene count) ===")
    bd = get_breakdown()
    for g in bd["genes"]:
        print(f"  {g['gene']:10s} n={g['n_patients']}  severe={g['severity_distribution']['severe_pct']}%  TCC={g['complex_features']['thin_corpus_callosum_pct']}%  epi={g['complex_features']['epilepsy_pct']}%")
    print("\n=== Definitions (count) ===")
    df = get_definitions()
    print(f"  {len(df['definitions'])} definitions")
