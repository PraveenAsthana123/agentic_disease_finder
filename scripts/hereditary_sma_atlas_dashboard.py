#!/usr/bin/env python3
"""Hereditary-SMA-Atlas — Complete 8-Gene Hereditary Spinal Muscular Atrophy Atlas
SMN1   (survival motor neuron 1; 294 aa; 5q13.2; AR;
         SMA 5q — MOST COMMON hereditary motor neuron disease;
         3 FDA-approved therapies (nusinersen 2016, onasemnogene 2019, risdiplam 2020);
         newborn screening mandatory 40+ countries; seed SEED_BASE+0) ·
AR     (androgen receptor; 919 aa; Xq11.2-q12; X-linked recessive;
         Kennedy SBMA — spinal and bulbar muscular atrophy;
         CAG repeat >36; polyQ disease; males predominantly; slowly progressive;
         leuprolide trial NEGATIVE; seed SEED_BASE+1) ·
BICD2  (bicaudal D2; 820 aa; 9q22.31; AD;
         SMALED2 — SMA with lower extremity predominance;
         congenital hypotonia; DDH screen mandatory; seed SEED_BASE+2) ·
DYNC1H1 (dynein cytoplasmic 1 heavy chain 1; 4646 aa; 14q32.31; AD;
          SMALED1 — SMA lower extremity dominant + CNS malformations subgroup;
          foot deformities; cortical dysplasia subgroup; seed SEED_BASE+3) ·
VAPB   (VAMP-associated protein B/C; 243 aa; 20q13.32; AD;
         ALS8 / SMA proximal — P56S Brazil founder;
         late onset slowly progressive; fasciculations prominent; seed SEED_BASE+4) ·
SETX   (senataxin; 2677 aa; 9q34.13; AD;
         ALS4 juvenile — onset 10-25y; slowly progressive; NO bulbar involvement KEY;
         decades survival; RNA-DNA hybrid helicase; seed SEED_BASE+5) ·
SIGMAR1 (sigma non-opioid intracellular receptor 1; 291 aa; 9p13.3; AR;
          Distal SMA / ALS16 — ER-mitochondria MAM junction;
          juvenile onset; motor + sensory; seed SEED_BASE+6) ·
PLEKHG5 (pleckstrin homology and RhoGEF domain protein G5; 1007 aa; 1p36.31; AR;
          DSMA4 — distal SMA type 4; autophagy-NF-κB pathway;
          slowly progressive distal weakness; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1718–1725)
"""

import random

SEED_BASE = 1718

SMA_GENES = [
    # ── SMN1 — SMA 5q ─────────────────────────────────────────────────────────
    {
        "gene": "SMN1",
        "protein": "SMN1 — 5q13.2 AR — Survival-Motor-Neuron-1-294aa — SMA-5q-MOST-COMMON-HeredMND — Nusinersen-FDA2016 / Onasemnogene-FDA2019 / Risdiplam-FDA2020 — NBS-Mandatory-40+Countries",
        "alias": (
            "SMN1 (survival motor neuron 1, telomeric); OMIM gene 600354; "
            "SMA (spinal muscular atrophy) OMIM 253300/253550/253400/271150. "
            "5q13.2; 294 aa; ~38 kDa; autosomal recessive. "
            "FUNCTION: SMN1 is the primary gene encoding survival motor neuron (SMN) protein; "
            "SMN is a ubiquitous protein essential for spliceosomal snRNP biogenesis, "
            "pre-mRNA splicing, axonal RNA transport, and localised translation. "
            "Motor neurons are uniquely sensitive to reduced SMN due to their extreme length "
            "and high translational demands. SMN2 (centromeric copy) can partially compensate "
            "via exon 7 inclusion (~10-20% full-length SMN protein). "
            "SMA MECHANISM: homozygous SMN1 deletion/conversion in ~95%; "
            "compound heterozygous (deletion + point mutation) ~5%. "
            "SMN2 copy number is the primary phenotypic modifier: "
            "1 copy → Type 1 (Werdnig-Hoffmann, most severe); "
            "2 copies → Type 1-2; 3 copies → Type 2-3; "
            "4 copies → Type 3-4 (mild/adult). "
            "TYPES: SMA Type 1 (onset 0-6 months; never sit unsupported; "
            "median survival <2y without treatment; die of respiratory failure); "
            "Type 2 (onset 6-18 months; sit but never walk; scoliosis, respiratory decline); "
            "Type 3 (onset >18 months; walk; proximal weakness; normal lifespan); "
            "Type 4 (adult onset >21y; mild proximal weakness; normal lifespan). "
            "TREATMENTS: "
            "(1) Nusinersen (Spinraza, Biogen): intrathecal ASO promoting SMN2 exon 7 inclusion; "
            "FDA approved December 2016; ENDEAR trial (Type 1) showed 41% motor response; "
            "CHERISH trial (Type 2) showed improvement; dosing 4 loading doses then every 4 months; "
            "(2) Onasemnogene abeparvovec (Zolgensma, Novartis/AveXis): IV gene therapy "
            "(AAV9-SMN1); FDA approved May 2019; single infusion; indicated <2y age (label); "
            "STR1VE trial: 92% survived event-free at 14 months; "
            "STRONG-SMA trial shows benefit older patients off-label; "
            "(3) Risdiplam (Evrysdi, Roche): oral SMN2 exon 7 splicing modifier; "
            "FDA approved August 2020; FIREFISH (Type 1) and SUNFISH (Type 2-3) trials; "
            "convenient oral administration; no intrathecal required; used across all types. "
            "NEWBORN SCREENING: mandatory or recommended in 40+ countries; "
            "pre-symptomatic treatment dramatically improves outcomes — Type 1 babies treated "
            "at birth can achieve motor milestones; NBS by SMN1 deletion detection from DBS. "
            "GENETIC TESTING: MLPA (multiplex ligation-dependent probe amplification) is gold standard "
            "for deletion detection; SMN2 copy number MANDATORY for prognosis and treatment decisions; "
            "comprehensive panel needed for compound heterozygotes (point mutations). "
            "KEY BIOMARKER: plasma/CSF neurofilament (NfL, pNfH) measures treatment response; "
            "phospho-neurofilament heavy (pNfH) is most sensitive SMA progression biomarker."
        ),
        "locus": "5q13.2",
        "aa": 294,
        "kDa": 38,
        "omim_gene": "600354",
        "omim_disease": "253300",
        "inheritance": "AR — homozygous deletion SMN1 exon 7-8 in 95%; compound het deletion+point mutation 5%; SMN2 copy number is phenotypic modifier; 1 in 40-50 carrier frequency general population",
        "gene_class": "RNA-binding protein; snRNP biogenesis; spliceosome assembly; axonal mRNA transport; SMN complex with Gemins 2-8 and Unrip; Cajal body gems",
        "key_alerts": [
            "SMN1-NEWBORN-SCREENING-MANDATORY: SMA is on newborn screening panel in 40+ countries — pre-symptomatic treatment dramatically improves outcomes; Type 1 babies treated before symptoms can achieve sitting/standing; advocate for NBS in any unscreened infant with hypotonia",
            "SMN1-3-TREATMENTS-AVAILABLE-2020: three FDA-approved disease-modifying therapies exist (nusinersen intrathecal, onasemnogene IV gene therapy, risdiplam oral) — all SMA patients should be referred to SMA specialist centre for treatment initiation; no head-to-head trials exist",
            "SMN1-SMN2-COPY-NUMBER-MANDATORY: SMN2 copy number MUST be tested at diagnosis — determines phenotype prognosis and guides treatment urgency; 1 SMN2 copy = Type 1 most severe; 4 copies = milder; never skip SMN2 testing",
            "SMN1-TYPE1-RESPIRATORY-EMERGENCY: SMA Type 1 babies develop respiratory failure typically by 12-18 months untreated; aggressive NIV (non-invasive ventilation) is life-saving; gastrostomy for nutritional support; palliative care discussion at diagnosis",
            "SMN1-SCOLIOSIS-SURVEILLANCE-TYPE2: Type 2 SMA patients who sit but never walk develop progressive scoliosis nearly universally — annual spine X-ray from age 3; surgical fusion often needed; scoliosis worsens respiratory function",
            "SMN1-ONASEMNOGENE-AGE-WEIGHT-LIMITS: onasemnogene (Zolgensma) label is <2y; weight limit 13.5 kg due to AAV9 dose calculation; liver toxicity monitoring mandatory post-infusion; one-time administration (no re-dosing possible due to neutralising antibodies)",
        ],
        "etiologies": [
            "SMA Type 1 (Werdnig-Hoffmann) — onset 0-6 months; 1-2 SMN2 copies; never achieves sitting; respiratory failure by 2y untreated; most common and severe",
            "SMA Type 2 (Dubowitz) — onset 6-18 months; 3 SMN2 copies; achieves sitting never walking; progressive respiratory decline; scoliosis universal",
            "SMA Type 3 (Kugelberg-Welander) — onset >18 months; 3-4 SMN2 copies; achieves walking; proximal weakness; normal lifespan; sub-divided 3a (<3y) and 3b (>3y)",
            "SMA Type 4 — adult onset >21y; 4 SMN2 copies; mild proximal weakness; tremor; very slow progression; normal lifespan",
            "SMA-PME (SMA with progressive myoclonic epilepsy) — biallelic SMN1 deletions + ASAH1 variants; very rare combined phenotype",
            "Compound heterozygous — deletion one allele + point mutation other allele (intron 6 c.822-3C>T most common intragenic); MLPA only detects deletions — sequencing required for full gene",
        ],
        "stats": {
            "typical_onset_age_type1_months": 3,
            "typical_onset_age_type2_months": 12,
            "typical_onset_age_type3_years": 3,
            "deletion_rate_pct": 95,
            "smn2_copy_range": "1-4",
            "nbs_countries": 40,
        },
    },
    # ── AR — Kennedy SBMA ─────────────────────────────────────────────────────
    {
        "gene": "AR",
        "protein": "AR — Xq11.2-q12 XLR — Androgen-Receptor-919aa — Kennedy-SBMA-SpinalBulbarMuscularAtrophy — CAG-Repeat->36 — PolyQ-Disease — Leuprolide-FUTILE-Trial-Negative",
        "alias": (
            "AR (androgen receptor); OMIM gene 313700; "
            "Kennedy disease / SBMA (spinal and bulbar muscular atrophy) OMIM 313200. "
            "Xq11.2-q12; 919 aa; ~99 kDa; X-linked recessive (males predominantly affected; "
            "homozygous or highly skewed X-inactivation females rarely symptomatic). "
            "FUNCTION: AR is a nuclear receptor transcription factor activated by androgens "
            "(testosterone, dihydrotestosterone); regulates gene expression controlling "
            "male secondary sexual characteristics, spermatogenesis, muscle mass, and bone density. "
            "The N-terminal transactivation domain contains a CAG repeat (normal 9-36 repeats) "
            "encoding a polyglutamine (polyQ) tract. "
            "KENNEDY DISEASE MECHANISM: CAG expansion >36 repeats causes a "
            "toxic gain-of-function — the polyQ-expanded AR misfolds, accumulates nuclear "
            "inclusions in motor neurons and dorsal root ganglia, impairs transcription, "
            "and causes selective motor and sensory neuron degeneration. "
            "Critically: androgen binding to expanded AR is REQUIRED for toxicity — "
            "castrate male mice with SBMA AR are protected from motor neuron disease. "
            "CLINICAL FEATURES: Males; onset typically 40-60 years (range 20-80); "
            "slowly progressive proximal and bulbar muscle weakness (bulbar prominent — "
            "tongue fasciculations, facial weakness, dysphagia, dysarthria); "
            "prominent fasciculations (often mis-diagnosed as ALS); "
            "sensory neuropathy (large fibre — reduced reflexes, sensory ataxia); "
            "endocrine features: gynaecomastia (>70%), testicular atrophy, reduced fertility/infertility; "
            "diabetes mellitus increased; CAG repeat length inversely correlates with age of onset. "
            "BIOMARKER: elevated serum creatine kinase; reduced testosterone levels sometimes; "
            "EMG shows denervation + sensory neuropathy (differentiates from pure ALS). "
            "LEUPROLIDE (GnRH agonist): Phase 3 trial (JASMINS) was NEGATIVE — "
            "androgen suppression did NOT significantly slow progression clinically; "
            "no approved disease-modifying therapy; IGF-1 (mecasermin) Phase 2 also negative. "
            "PROGNOSIS: slowly progressive; normal or near-normal lifespan; "
            "wheelchair dependency 10-20 years after symptom onset in many; "
            "respiratory complications in later stages but much slower than ALS. "
            "GENETIC COUNSELLING: X-linked; carrier females have 50% chance of affected sons; "
            "prenatal/pre-implantation diagnosis possible; daughters of affected males are obligate carriers."
        ),
        "locus": "Xq11.2-q12",
        "aa": 919,
        "kDa": 99,
        "omim_gene": "313700",
        "omim_disease": "313200",
        "inheritance": "X-linked recessive; males predominantly affected; females carriers (rarely symptomatic if highly skewed X-inactivation); CAG >36 repeats (full penetrance >40 repeats); inverse correlation repeat length with age of onset",
        "gene_class": "Nuclear receptor transcription factor; ligand-activated (androgen); polyQ toxic gain-of-function mechanism; androgen binding required for toxicity (castration protective in mouse models); nuclear inclusions ubiquitinated polyQ-AR",
        "key_alerts": [
            "AR-KENNEDY-NOT-ALS: Kennedy SBMA is commonly misdiagnosed as ALS — key differentiating features: slowly progressive (years-decades NOT months); sensory neuropathy on EMG/NCS (ALS is purely motor); gynaecomastia (>70%); endocrine features; bulbar involvement prominent with fasciculations; correct diagnosis critical as prognosis very different",
            "AR-LEUPROLIDE-FUTILE-TRIAL-NEGATIVE: leuprolide (GnRH agonist) Phase 3 JASMINS trial was NEGATIVE — do NOT offer androgen suppression as SMA treatment; no FDA-approved disease-modifying therapy exists for Kennedy disease; supportive care and physiotherapy are mainstays",
            "AR-GYNECOMASTIA-ENDOCRINE-SCREEN: gynaecomastia present in >70% of Kennedy disease males — screen all males with unexplained gynecomastia + weakness + fasciculations; also check testosterone, LH/FSH (elevated), semen analysis (azoospermia/oligospermia common)",
            "AR-SENSORY-NEUROPATHY-DISTINGUISHES-ALS: Kennedy disease has sensory neuropathy (large fibre — reduced vibration, proprioception, sensory ataxia) — ALS is a pure motor neuron disease; sensory abnormalities on NCS/EMG should prompt AR repeat testing before ALS diagnosis",
            "AR-CARRIERS-REPRODUCTIVE-COUNSELLING: female carriers have no/minimal symptoms but 50% sons affected, 50% daughters carriers; pre-conception genetic counselling mandatory; PGT (preimplantation genetic testing) available; daughters of affected men are all obligate carriers",
            "AR-RESPIRATORY-LATE-NOT-EARLY: respiratory failure occurs much later than in ALS; annual pulmonary function testing from diagnosis; NIV rarely needed in first decade; distinguish from ALS where respiratory failure is often early",
        ],
        "etiologies": [
            "Classical Kennedy SBMA — CAG repeat 38-60; onset 40-60y; slow progression; gynaecomastia + sensory neuropathy + proximal/bulbar weakness triad",
            "Juvenile/early-onset — CAG >60 repeats; onset 20-30y; more rapid course; oligospermia/azoospermia invariable",
            "Late-onset mild — CAG 36-40 repeats; onset 60-80y; subtle proximal weakness; frequent misdiagnosis as age-related sarcopenia",
            "Symptomatic female carriers — extreme X-inactivation skewing favouring expanded allele; cramping, mild weakness, gynecomastia absent; rare",
            "Overlap with primary hypogonadism — low testosterone, high LH/FSH, azoospermia, Leydig cell failure — endocrinology referral mandatory",
            "SBMA plus metabolic syndrome — increased diabetes, dyslipidaemia, non-alcoholic fatty liver disease in CAG >42 cohort",
        ],
        "stats": {
            "cag_repeat_cutoff_pathogenic": 36,
            "cag_repeat_full_penetrance": 40,
            "gynecomastia_pct": 72,
            "sensory_neuropathy_pct": 95,
            "typical_onset_decade": "40-60y",
        },
    },
    # ── BICD2 — SMALED2 ───────────────────────────────────────────────────────
    {
        "gene": "BICD2",
        "protein": "BICD2 — 9q22.31 AD — Bicaudal-D2-820aa — SMALED2-CongenitalHypotonia-LowerExtremityPredominance — DDH-Screen-Mandatory — Dynein-Cargo-Adaptor",
        "alias": (
            "BICD2 (bicaudal D cargo adaptor 2); OMIM gene 609797; "
            "SMALED2 (spinal muscular atrophy, lower extremity predominant 2) OMIM 615290. "
            "9q22.31; 820 aa; ~93 kDa; autosomal dominant (gain-of-function). "
            "FUNCTION: BICD2 is a dynein/dynactin cargo adaptor protein; "
            "it links cellular cargo (Golgi vesicles, secretory vesicles, mRNA) to the "
            "retrograde microtubule motor dynein-dynactin complex. "
            "BICD2 has an elongated coiled-coil structure with cargo-binding C-terminus "
            "and dynein-dynactin-binding N-terminus. "
            "SMALED2 MECHANISM: gain-of-function mutations (mainly in coiled-coil domains) "
            "constitutively activate dynein-dynactin binding — this disrupts Golgi function, "
            "vesicular trafficking, and nuclear migration in developing neurons, "
            "leading to selective lower motor neuron degeneration. "
            "CLINICAL FEATURES: typically congenital onset (hypotonia at birth or early infancy); "
            "LOWER EXTREMITY PREDOMINANT weakness — legs much weaker than arms; "
            "foot deformities (club foot, vertical talus, pes cavus) common; "
            "developmental dysplasia of the hip (DDH) — screen ALL BICD2 patients; "
            "proximal > distal lower limb weakness; upper limbs relatively spared; "
            "bulbar not involved; intelligence NORMAL; "
            "some patients never walk; others walk with aids; "
            "slowly progressive or relatively stable after initial period; "
            "fasciculations may be present. "
            "IMAGING: brain and spine MRI normal; phrenic nerve conduction normal (no respiratory involvement typically). "
            "DIFFERENTIAL: SMA 5q (will not have normal SMN1 or upper extremity sparing); "
            "DYNC1H1 (SMALED1) — similar phenotype, distinguish by gene. "
            "TREATMENT: no disease-modifying treatment; physiotherapy; bracing/orthotics; "
            "orthopaedic surgery for scoliosis/foot deformities/DDH. "
            "GENETIC COUNSELLING: autosomal dominant; de novo mutations common; "
            "penetrance high; offspring risk 50%. "
            "BIOMARKER: EMG shows denervation pattern; lower extremity > upper; "
            "CK mildly elevated or normal."
        ),
        "locus": "9q22.31",
        "aa": 820,
        "kDa": 93,
        "omim_gene": "609797",
        "omim_disease": "615290",
        "inheritance": "AD — gain-of-function; de novo mutations common; penetrance high; offspring risk 50%; germline mosaicism reported",
        "gene_class": "Dynein-dynactin cargo adaptor; coiled-coil protein; Golgi vesicle trafficking; nuclear migration; constitutive dynein activation (GOF mechanism); microtubule minus-end directed transport",
        "key_alerts": [
            "BICD2-DDH-SCREEN-MANDATORY: developmental dysplasia of the hip (DDH) is significantly increased in BICD2 SMALED2 — hip ultrasound in all infants, pelvis X-ray in older children; undiagnosed DDH causes long-term joint damage and pain; orthopaedic referral at diagnosis",
            "BICD2-LOWER-EXTREMITY-PREDOMINANCE: weakness is dramatically lower extremity predominant — legs much weaker than arms; this pattern is highly characteristic and distinguishes from SMA 5q (proximal limb girdle) and Kennedy (also proximal but bulbar); always document upper vs lower extremity strength comparison",
            "BICD2-CONGENITAL-PRESENTATION: many patients present at birth or early infancy with hypotonia and foot deformities — neonatal hypotonia workup should include BICD2 testing when lower extremity weakness is prominent; clubfoot + hypotonia is a red flag",
            "BICD2-INTELLIGENCE-NORMAL: cognitive development is NORMAL in SMALED2 — reassure families; school performance should be normal; separate from conditions with CNS involvement like DYNC1H1 which can have cortical dysplasia subgroup",
            "BICD2-STABLE-OR-SLOWLY-PROGRESSIVE: after initial congenital/early childhood presentation, the course is relatively stable or very slowly progressive — important for prognostication; many patients maintain independent ambulation well into adulthood",
            "BICD2-DE-NOVO-GENETIC-COUNSELLING: de novo mutations are common — negative family history does NOT exclude diagnosis; test proband by sequencing panel; if de novo confirmed, recurrence risk for parents is low but germline mosaicism risk requires counselling",
        ],
        "etiologies": [
            "BICD2 coiled-coil domain mutations — most common; S107L, N188T, R694C hotspots; gain-of-function constitutive dynein activation; congenital onset typical",
            "Congenital severe — fetal akinesia features; extreme hypotonia at birth; arthrogryposis; DDH; may never achieve standing",
            "Childhood onset mild — later presentation (2-5y); gait abnormality; lower extremity weakness; pes cavus; near-normal walking ability into adulthood",
            "Sporadic (de novo) — negative family history; confirmed de novo on trio sequencing; most BICD2 cases are de novo",
            "Familial autosomal dominant — multiple generations affected; variable expressivity within family; parent often mild while proband severe",
            "BICD2-related with respiratory involvement — rare; diaphragm involvement in a subset; respiratory surveillance recommended in all",
        ],
        "stats": {
            "ddh_rate_pct": 35,
            "foot_deformity_pct": 70,
            "de_novo_rate_pct": 60,
            "upper_extremity_relative_sparing_pct": 80,
        },
    },
    # ── DYNC1H1 — SMALED1 ────────────────────────────────────────────────────
    {
        "gene": "DYNC1H1",
        "protein": "DYNC1H1 — 14q32.31 AD — Dynein-CytoplasmaticHeavyChain1-4646aa — SMALED1-LowerExtremityDominant — CNS-MalformationSubgroup-Cortical-Dysplasia — Foot-Deformities-DISTINCTIVE",
        "alias": (
            "DYNC1H1 (dynein cytoplasmic 1 heavy chain 1); OMIM gene 600112; "
            "SMALED1 (spinal muscular atrophy, lower extremity dominant 1) OMIM 158600; "
            "also causes CMT2O (Charcot-Marie-Tooth type 2O), intellectual disability with cortical malformations. "
            "14q32.31; 4646 aa; ~532 kDa; autosomal dominant (predominantly haploinsufficiency or dominant negative). "
            "FUNCTION: DYNC1H1 encodes the heavy chain of cytoplasmic dynein-1 — the main "
            "minus-end directed microtubule motor. Cytoplasmic dynein-1 is critical for: "
            "(1) retrograde axonal transport (organelles, vesicles, mRNAs from axon tip to cell body); "
            "(2) nuclear positioning and migration during cortical neurogenesis; "
            "(3) mitotic spindle positioning and chromosome segregation. "
            "SMALED1 MECHANISM: dominant mutations in motor domain, stem, or HEAT repeats "
            "impair retrograde axonal transport and nuclear migration, causing selective "
            "lower motor neuron degeneration; some mutations additionally disrupt cortical "
            "neuronal migration → lissencephaly/pachygyria subgroup. "
            "CLINICAL FEATURES: "
            "Motor — lower extremity predominant weakness (proximal > distal); "
            "foot deformities (pes cavus, talipes, vertical talus) highly characteristic; "
            "some never walk independently; proximal hip girdle weakness; "
            "fasciculations; CK mildly elevated; "
            "IMPORTANT: intelligence can be AFFECTED in the CNS subgroup — "
            "cortical malformations (pachygyria, lissencephaly, cortical dysplasia) on brain MRI "
            "in approximately 20-30% of DYNC1H1 mutation carriers; "
            "intellectual disability ranges from mild to severe in CNS subgroup; "
            "seizures in CNS malformation subgroup. "
            "Peripheral neuropathy (CMT2O) — axonal sensorimotor neuropathy; "
            "distal weakness; demyelination not prominent. "
            "DIFFERENTIAL: BICD2/SMALED2 (similar but cognition almost always normal); "
            "SMA 5q (proximal symmetric, test SMN1 first). "
            "TREATMENT: no disease-modifying therapy; physiotherapy; orthotics; "
            "antiepileptics for CNS subgroup; MRI brain mandatory in all DYNC1H1 patients. "
            "GENETIC COUNSELLING: autosomal dominant; de novo mutations very common; "
            "germline mosaicism documented; 50% offspring risk."
        ),
        "locus": "14q32.31",
        "aa": 4646,
        "kDa": 532,
        "omim_gene": "600112",
        "omim_disease": "158600",
        "inheritance": "AD — haploinsufficiency or dominant negative; de novo mutations very common (majority sporadic); 50% offspring risk; germline mosaicism documented",
        "gene_class": "Cytoplasmic dynein-1 heavy chain; microtubule minus-end directed motor; retrograde axonal transport; nuclear migration during cortical neurogenesis; largest known motor protein (4646 aa); ATPase motor domain (AAA+ ring)",
        "key_alerts": [
            "DYNC1H1-BRAIN-MRI-MANDATORY: ALL DYNC1H1 patients require brain MRI — cortical malformations (pachygyria, lissencephaly, subcortical band heterotopia) in 20-30%; MRI-negative patients have normal cognition; MRI-positive subgroup has intellectual disability and seizures; this distinguishes DYNC1H1 from BICD2",
            "DYNC1H1-LOWER-EXTREMITY-FOOT-DEFORMITIES: foot deformities (pes cavus, vertical talus, clubfoot) are highly characteristic of SMALED1 — include in differential of any infant/child with hypotonia + foot deformities + lower extremity predominant weakness; radiograph feet in all cases",
            "DYNC1H1-COGNITION-VARIABLE: intelligence is NOT always normal in DYNC1H1 — unlike BICD2; always assess cognition formally; the CNS malformation subgroup has intellectual disability (mild to severe) and may need special educational support and epilepsy management",
            "DYNC1H1-DE-NOVO-COMMON: de novo mutations are very common — do not exclude diagnosis based on negative family history; trio sequencing (proband + both parents) is most efficient diagnostic approach; if de novo, parental recurrence risk is low (germline mosaicism counselling still needed)",
            "DYNC1H1-LARGE-GENE-SEQUENCING: DYNC1H1 is one of the largest human genes (4646 aa) — standard panel sequencing may miss deep intronic variants; WES or WGS may be required for complex cases; CNV analysis with MLPA or array-CGH for deletions",
            "DYNC1H1-CMT2O-OVERLAP: some DYNC1H1 mutations cause CMT2O (axonal Charcot-Marie-Tooth type 2O) rather than SMALED1 — the distinguishing feature is distal vs proximal weakness and NCS pattern; comprehensive DYNC1H1 testing should be on CMT panels",
        ],
        "etiologies": [
            "SMALED1 motor predominant — lower extremity weakness, foot deformities, fasciculations; cognition normal; no cortical malformations on MRI",
            "SMALED1 with cortical malformations — pachygyria/lissencephaly on MRI; intellectual disability mild-severe; seizures; motor weakness lower extremity",
            "CMT2O — axonal sensorimotor neuropathy; distal weakness hands and feet; demyelination absent; overlaps with SMALED1 in some families",
            "Severe congenital — profound hypotonia at birth; arthrogryposis; bilateral foot deformities; poor motor prognosis; often de novo large effect variants",
            "Mild late-onset — subtle gait abnormality; pes cavus only; detected incidentally on family cascade testing",
            "DYNC1H1 with epileptic encephalopathy — CNS malformation subgroup; drug-resistant seizures; severe intellectual disability; motor weakness may be overshadowed by CNS symptoms",
        ],
        "stats": {
            "cns_malformation_pct": 25,
            "foot_deformity_pct": 75,
            "de_novo_rate_pct": 65,
            "intellectual_disability_if_mri_positive_pct": 90,
        },
    },
    # ── VAPB — ALS8/SMA-Proximal ──────────────────────────────────────────────
    {
        "gene": "VAPB",
        "protein": "VAPB — 20q13.32 AD — VAMP-Associated-Protein-B-243aa — ALS8-SMAProximal — P56S-Brazil-Founder — LateOnset-SlowlyProgressive — Fasciculations-PROMINENT",
        "alias": (
            "VAPB (VAMP associated protein B and C); OMIM gene 605704; "
            "ALS8 (amyotrophic lateral sclerosis type 8) / SMA-proximal OMIM 608627. "
            "20q13.32; 243 aa; ~27 kDa; autosomal dominant. "
            "FUNCTION: VAPB is a tail-anchored ER membrane protein that is a major "
            "component of the endoplasmic reticulum-mitochondria contact sites "
            "(mitochondria-associated membranes, MAM). "
            "VAPB interacts with PTPIP51 (RMDN3) to tether ER to mitochondria, "
            "regulating calcium transfer between ER and mitochondria, "
            "lipid metabolism, and ER-phagy. "
            "VAPB also interacts with FFAT (two phenylalanines in an acidic tract) motif-containing "
            "proteins including lipid transfer proteins at membrane contact sites. "
            "ALS8/SMA MECHANISM: P56S mutation creates a dominant toxic aggregate — "
            "P56S VAPB misfolds and forms large cytoplasmic inclusions that "
            "sequester wild-type VAPB and binding partners, disrupting ER-mitochondria tethering, "
            "calcium homeostasis, and ER protein quality control → motor neuron degeneration. "
            "CLINICAL FEATURES: "
            "P56S founder mutation: Brazilian/Portuguese founder mutation — first described in "
            "nine Brazilian families of Portuguese descent; subsequently found in other ethnicities. "
            "Onset: late (typically 40-60 years); very slowly progressive (decades). "
            "Phenotypic spectrum: "
            "(1) ALS-like (LMN+UMN signs, fasciculations, proximal weakness, bulbar involvement late); "
            "(2) SMA-like proximal weakness (LMN predominant, fasciculations, no UMN signs); "
            "(3) Late-onset distal SMA variant. "
            "FASCICULATIONS are very prominent — often first symptom; "
            "widespread fasciculations should prompt VAPB testing especially in patients of "
            "Brazilian/Portuguese descent with positive family history. "
            "PROGNOSIS: much slower than classical ALS; many patients survive 10-20+ years. "
            "TREATMENT: no approved disease-modifying therapy; supportive care; "
            "clinical trials targeting ER-mitochondria dysfunction underway. "
            "GENETIC COUNSELLING: autosomal dominant; P56S penetrance appears complete; "
            "first-degree relatives should undergo testing."
        ),
        "locus": "20q13.32",
        "aa": 243,
        "kDa": 27,
        "omim_gene": "605704",
        "omim_disease": "608627",
        "inheritance": "AD — P56S gain-of-toxic-function; dominant aggregate trapping WT VAPB; full penetrance for P56S; other rare variants also pathogenic; offspring risk 50%",
        "gene_class": "ER membrane protein; ER-mitochondria contact site (MAM) tethering; FFAT motif ligand; calcium transfer ER→mitochondria; lipid metabolism; tail-anchored type II integral membrane protein",
        "key_alerts": [
            "VAPB-FASCICULATIONS-PROMINENT-EARLY: widespread prominent fasciculations are often the first symptom of VAPB ALS8 — sometimes preceding weakness by years; test VAPB in any patient with prominent fasciculations + positive family history, especially of Brazilian/Portuguese descent",
            "VAPB-P56S-BRAZILIAN-PORTUGUESE-FOUNDER: P56S is a founder mutation in Brazilian/Portuguese populations — always ask about family ancestry; cascade testing of first-degree relatives is essential given autosomal dominant inheritance with apparent full penetrance",
            "VAPB-SLOWLY-PROGRESSIVE-NOT-ALS: VAPB ALS8 is much more slowly progressive than classical ALS (survival often >10 years) — do NOT apply standard ALS prognosis to VAPB patients; the slowly progressive course requires different palliative care timing and NIV/PEG planning",
            "VAPB-PHENOTYPIC-SPECTRUM: VAPB mutations cause a spectrum from ALS-like (UMN+LMN) to pure SMA-like proximal weakness (LMN only) — phenotypic heterogeneity within the same family is common; do NOT exclude VAPB because phenotype is 'too mild' for ALS",
            "VAPB-ER-MITOCHONDRIA-BIOLOGY: VAPB disrupts ER-mitochondria contact sites (MAM) — a pathway shared with other neurodegenerative diseases; ongoing research into ER stress modulators and mitochondrial bioenergetics as therapeutic targets",
            "VAPB-CASCADE-TESTING: autosomal dominant with full penetrance — first-degree relatives (children and siblings of affected) should all be offered genetic testing; pre-symptomatic testing enables earlier diagnosis and potential trial participation",
        ],
        "etiologies": [
            "P56S founder mutation — Brazilian/Portuguese descent; classic late-onset slowly progressive; fasciculations prominent; most reported cases",
            "ALS8 phenotype — UMN + LMN features; bulbar involvement; fasciculations; but slower than classical ALS",
            "SMA-proximal phenotype — pure LMN; proximal weakness; no UMN signs; slowly progressive; sometimes walked decades",
            "Late-onset distal SMA — distal predominant weakness; slowly progressive; rare",
            "Non-P56S variants — other VAPB mutations; similar slowly progressive phenotype; non-Brazilian families",
            "Familial ALS8 with dementia — rare cases with frontotemporal involvement; less common than in C9ORF72 or TBK1",
        ],
        "stats": {
            "founder_mutation_pct": 85,
            "fasciculations_prominent_pct": 90,
            "upc_involvement_pct": 40,
            "typical_onset_decade": "40-60y",
        },
    },
    # ── SETX — ALS4 Juvenile ──────────────────────────────────────────────────
    {
        "gene": "SETX",
        "protein": "SETX — 9q34.13 AD — Senataxin-2677aa — ALS4-Juvenile-10-25y — NO-Bulbar-KEY-Distinguishing-Feature — RNA-DNA-Hybrid-Helicase — Slowly-Progressive-Decades-Survival",
        "alias": (
            "SETX (senataxin); OMIM gene 608465; "
            "ALS4 (amyotrophic lateral sclerosis type 4, juvenile) OMIM 602433; "
            "also causes AOA2 (ataxia with oculomotor apraxia type 2) when biallelic AR mutations. "
            "9q34.13; 2677 aa; ~303 kDa; autosomal dominant (ALS4) / autosomal recessive (AOA2). "
            "FUNCTION: Senataxin is a DNA/RNA helicase that resolves R-loops "
            "(RNA-DNA hybrids formed during transcription), "
            "facilitates transcription-replication conflict resolution, "
            "coordinates RNA processing (3' end processing, termination), "
            "and is involved in DNA damage response (ATM pathway). "
            "Senataxin localises to sites of R-loop formation, especially at transcription "
            "termination sites, ensuring genomic stability and proper RNA metabolism. "
            "ALS4 MECHANISM: AD gain-of-function (or dominant negative) mutations in the "
            "helicase domain cause accumulation of R-loops, transcription stress, "
            "and DNA damage in motor neurons → selective degeneration. "
            "CLINICAL FEATURES (ALS4 AD): "
            "Onset: juvenile/young adult (10-25 years); "
            "HALLMARK: NO BULBAR INVOLVEMENT — this is THE key distinguishing feature from "
            "other MND; speech and swallowing remain preserved throughout course; "
            "normal intelligence; "
            "Progressive proximal + distal limb weakness; UMN signs often present; "
            "respiratory involvement is very late or absent; "
            "VERY SLOWLY PROGRESSIVE — survival for decades; many patients ambulate 20+ years; "
            "pyramidal signs (hyperreflexia, extensor plantar) common; "
            "subtle sensory abnormalities in some. "
            "AOA2 (biallelic AR): cerebellar ataxia + oculomotor apraxia + sensory neuropathy + "
            "elevated alpha-fetoprotein (AFP) — completely different from ALS4; "
            "same gene, different inheritance, different phenotype. "
            "TREATMENT: no approved disease-modifying therapy; physiotherapy; "
            "reassurance that prognosis is much better than ALS. "
            "GENETIC COUNSELLING: ALS4 AD — 50% offspring risk; "
            "variable expressivity within families; AOA2 AR — carrier testing for siblings."
        ),
        "locus": "9q34.13",
        "aa": 2677,
        "kDa": 303,
        "omim_gene": "608465",
        "omim_disease": "602433",
        "inheritance": "AD (ALS4 gain-of-function/dominant negative in helicase domain); AR (AOA2 — completely different phenotype); penetrance variable for ALS4; juvenile onset typical; offspring risk 50% for AD",
        "gene_class": "DNA/RNA helicase (SFI superfamily); R-loop resolution; transcription termination; DNA damage response; ATM pathway; transcription-replication conflict resolution; genomic stability maintenance",
        "key_alerts": [
            "SETX-NO-BULBAR-KEY-DIFFERENTIATOR: ALS4 (SETX) characteristically has NO bulbar involvement (speech and swallowing preserved throughout course) — this is THE most important distinguishing feature from classical ALS and other MND; if bulbar symptoms present, reconsider SETX diagnosis",
            "SETX-JUVENILE-ONSET-10-25Y: ALS4 presents in children and young adults (10-25 years) — juvenile-onset MND in this age group should prompt SETX testing; juvenile MND differential also includes Kennedy SBMA (but XLR), BICD2/DYNC1H1 (lower extremity), and hexosaminidase deficiency",
            "SETX-DECADES-SURVIVAL: ALS4 patients survive for decades with very slow progression — dramatically different prognosis from ALS (median 2-5y); families must be clearly counselled on prognosis to avoid unnecessary despair; many ALS4 patients remain ambulatory 20+ years after diagnosis",
            "SETX-AOA2-DIFFERENT-DISEASE-SAME-GENE: biallelic AR SETX mutations cause AOA2 (cerebellar ataxia + oculomotor apraxia + elevated AFP) — completely different from ALS4 (AD); do not confuse the two conditions; testing of siblings for AR carrier status if AOA2 proband",
            "SETX-AFP-ELEVATED-AOA2: alpha-fetoprotein (AFP) is markedly elevated in AOA2 — this is a diagnostic biomarker for AOA2; check AFP in any young patient with cerebellar ataxia + oculomotor apraxia + sensory neuropathy before expensive gene panels",
            "SETX-R-LOOP-BIOLOGY: SETX resolves RNA-DNA hybrids (R-loops) — a mechanism increasingly linked to multiple neurodegenerative diseases; ongoing research into R-loop pathway modulators as therapeutic targets; ALS4 patients suitable for longitudinal research cohorts",
        ],
        "etiologies": [
            "ALS4 juvenile — onset 10-25y; AD; proximal + distal weakness; NO bulbar; slowly progressive; UMN signs; normal cognition",
            "ALS4 with pyramidal features — hyperreflexia, Babinski, spastic gait; can mimic hereditary spastic paraplegia (HSP) in early stages",
            "ALS4 family history — multiple generations affected; variable age of onset and rate of progression within family",
            "De novo SETX helicase domain — sporadic juvenile-onset MND without family history; de novo verified on trio sequencing",
            "AOA2 (ataxia-oculomotor apraxia type 2) — biallelic AR mutations; cerebellar ataxia; oculomotor apraxia; sensory neuropathy; elevated AFP; distinct from ALS4",
            "ALS4 early severe — rare severe early-onset variant; childhood onset <10y; more rapid progression than typical; large helicase domain deletions/truncations",
        ],
        "stats": {
            "no_bulbar_involvement_pct": 98,
            "umc_signs_pct": 75,
            "typical_onset_age_years": "10-25",
            "afp_elevated_aoa2_pct": 95,
        },
    },
    # ── SIGMAR1 — Distal SMA / ALS16 ─────────────────────────────────────────
    {
        "gene": "SIGMAR1",
        "protein": "SIGMAR1 — 9p13.3 AR — Sigma1-Receptor-291aa — DistalSMA-ALS16-JuvenileALS — ER-Mitochondria-MAM-Junction — Motor-Sensory-Peripheral-Neuropathy",
        "alias": (
            "SIGMAR1 (sigma non-opioid intracellular receptor 1); OMIM gene 601978; "
            "Distal SMA / ALS16 (amyotrophic lateral sclerosis type 16, juvenile) OMIM 614373; "
            "also causes distal hereditary motor neuropathy (dHMN). "
            "9p13.3; 291 aa; ~26 kDa; autosomal recessive (biallelic). "
            "FUNCTION: SIGMAR1 (Sigma-1 receptor; σ1R) is a ligand-operated chaperone "
            "located primarily at the ER-mitochondria contact sites (MAM — mitochondria-associated membranes). "
            "σ1R acts as a dynamic scaffold that: "
            "(1) regulates calcium transfer from ER to mitochondria via IP3 receptor stabilisation; "
            "(2) chaperones misfolded proteins to prevent ER stress; "
            "(3) modulates ion channels (voltage-gated K+ channels); "
            "(4) facilitates neurotrophin signalling (BDNF receptor TrkB); "
            "(5) regulates lipid raft formation. "
            "σ1R ligands include progesterone (endogenous), cocaine, and various drugs "
            "(donepezil, haloperidol, fluvoxamine). "
            "ALS16 MECHANISM: biallelic loss-of-function SIGMAR1 mutations → ER stress, "
            "impaired calcium homeostasis, mitochondrial dysfunction → motor and sensory neuron loss. "
            "CLINICAL FEATURES: "
            "Onset: juvenile (typically 10-25 years); "
            "Distal weakness hands + feet (distal predominant unlike most SMA); "
            "sensory involvement (peripheral neuropathy pattern); "
            "slowly progressive; can be SMA-like (LMN predominant) or CMT-like (sensorimotor); "
            "EMG: axonal neuropathy; distal denervation; "
            "vocal cord paresis reported in some kindreds; "
            "cognitive function normal. "
            "TREATMENT: no approved therapy; "
            "σ1R agonists (fluvoxamine, SA4503) are being investigated for ER stress neuroprotection. "
            "GENETIC COUNSELLING: autosomal recessive; "
            "sibling carrier testing important; offspring risk 25% if both parents carriers."
        ),
        "locus": "9p13.3",
        "aa": 291,
        "kDa": 26,
        "omim_gene": "601978",
        "omim_disease": "614373",
        "inheritance": "AR (biallelic LOF mutations); carrier frequency varies by population; sibling risk 25%; parents obligate carriers; heterozygous carriers typically unaffected",
        "gene_class": "ER-mitochondria contact site (MAM) chaperone; sigma receptor ligand-binding; calcium homeostasis ER→mitochondria; ER stress regulation; IP3 receptor stabilisation; neurotrophin signalling modulator",
        "key_alerts": [
            "SIGMAR1-DISTAL-PREDOMINANT-SMA: SIGMAR1/ALS16 causes distal predominant weakness (hands and feet) unlike typical proximal SMA — EMG shows distal denervation pattern; include in differential of juvenile-onset distal SMA (dHMN) and CMT-like presentations",
            "SIGMAR1-SENSORY-INVOLVEMENT: peripheral sensory neuropathy is present in many SIGMAR1 cases — this helps distinguish from pure motor SMA; NCS should show sensory abnormalities; axonal pattern",
            "SIGMAR1-AR-SIBLINGS-AT-RISK: autosomal recessive — both parents are obligate carriers; 25% of siblings are affected; 50% of siblings are carriers; offer testing to ALL siblings of affected proband; family counselling critical",
            "SIGMAR1-ER-STRESS-BIOLOGY: SIGMAR1 dysfunction leads to ER stress — a shared mechanism in multiple neurodegenerative diseases; σ1R agonists (fluvoxamine) are being investigated as neuroprotective agents in ALS trials; SIGMAR1 patients may be eligible for mechanistically rational trials",
            "SIGMAR1-VOCAL-CORD-PARESIS: vocal cord paresis has been reported in some SIGMAR1 families — assess voice quality and consider laryngoscopy in patients with voice changes; distinguishes from some other distal SMA subtypes",
            "SIGMAR1-JUVENILE-ONSET-DIFFERENTIAL: juvenile-onset distal SMA differential includes SETX (ALS4, but no sensory involvement), CMT subtypes (GJB1, MFN2), DCTN1 (Perry syndrome), and SIGMAR1 — comprehensive gene panel required",
        ],
        "etiologies": [
            "Classic distal SMA / ALS16 — biallelic SIGMAR1 LOF; onset 10-25y; distal weakness + sensory neuropathy; slowly progressive",
            "Juvenile ALS phenotype — early-onset slowly progressive ALS with distal predominance; AR SIGMAR1; some UMN features in advanced disease",
            "Distal hereditary motor neuropathy (dHMN) — purely motor variant; distal denervation; no sensory loss; rare",
            "SIGMAR1 with vocal cord paresis — specific kindreds with vocal cord involvement; dysphonia; laryngoscopy diagnostic",
            "Compound heterozygous — two different pathogenic SIGMAR1 alleles; phenotype similar to homozygous LOF",
            "Late-onset AR SIGMAR1 — adult onset (30-40y); slowly progressive distal neuropathy; later presentation; diagnosed on panel sequencing",
        ],
        "stats": {
            "distal_predominance_pct": 85,
            "sensory_involvement_pct": 70,
            "typical_onset_age_years": "10-25",
            "vocal_cord_paresis_pct": 20,
        },
    },
    # ── PLEKHG5 — DSMA4 ───────────────────────────────────────────────────────
    {
        "gene": "PLEKHG5",
        "protein": "PLEKHG5 — 1p36.31 AR — PH-RhoGEF-1007aa — DSMA4-DistalSMA-Type4 — Autophagy-NF-kB-Pathway — LowerMotorNeuron-ONLY — Slowly-Progressive-Distal",
        "alias": (
            "PLEKHG5 (pleckstrin homology and RhoGEF domain containing G5); OMIM gene 611101; "
            "DSMA4 (distal spinal muscular atrophy type 4) OMIM 611067. "
            "1p36.31; 1007 aa; ~114 kDa; autosomal recessive. "
            "FUNCTION: PLEKHG5 is a guanine nucleotide exchange factor (GEF) for RhoA/RhoB "
            "small GTPases, activated via its PH (pleckstrin homology) domain "
            "by phosphoinositides at cellular membranes. "
            "PLEKHG5 activates RhoA GTPase signalling downstream in the NF-κB pathway "
            "and regulates macroautophagy — specifically autophagic vesicle formation "
            "at the axon terminal, which is critical for clearance of damaged organelles "
            "and proteins in long motor neuron axons. "
            "DSMA4 MECHANISM: biallelic PLEKHG5 LOF mutations → impaired NF-κB activation "
            "→ reduced autophagic clearance in motor axon terminals → accumulation of "
            "damaged mitochondria and aggregated proteins → distal motor neuron degeneration "
            "(length-dependent, longest axons affected first). "
            "CLINICAL FEATURES: "
            "Onset: congenital to early childhood; "
            "DISTAL lower motor neuron weakness — hands and feet predominantly; "
            "foot drop, wasting of intrinsic foot muscles, weakness grip; "
            "LOWER MOTOR NEURON ONLY — no UMN signs (normal reflexes or absent distally); "
            "voice hoarseness / vocal cord weakness in some; "
            "very slowly progressive — many patients maintain function into adulthood; "
            "EMG: chronic neurogenic changes distal; reduced amplitudes distally; "
            "NO sensory involvement (distinguishes from CMT). "
            "DIFFERENTIAL: SETX-ALS4 (but AD; no sensory; proximal > distal), "
            "SIGMAR1 (sensory involvement), DCTN1, HSPB1/HSPB8 dHMN. "
            "TREATMENT: no approved therapy; supportive; "
            "NF-κB pathway and autophagy modulation under investigation. "
            "GENETIC COUNSELLING: autosomal recessive; carrier frequency variable; "
            "sibling risk 25%; parents obligate carriers."
        ),
        "locus": "1p36.31",
        "aa": 1007,
        "kDa": 114,
        "omim_gene": "611101",
        "omim_disease": "611067",
        "inheritance": "AR (biallelic LOF); both parents obligate carriers; sibling risk 25%; carrier frequency varies by population; no known founder mutations",
        "gene_class": "RhoGEF guanine nucleotide exchange factor; RhoA/RhoB GTPase activator; NF-κB pathway; macroautophagy regulation (autophagic vesicle formation at axon terminal); PH domain phosphoinositide binding; length-dependent motor axon degeneration mechanism",
        "key_alerts": [
            "PLEKHG5-PURE-LMN-NO-UMN: DSMA4 is a pure lower motor neuron disease — absent or reduced reflexes distally; NO upper motor neuron signs (no spasticity, no Babinski, no hyperreflexia); presence of UMN signs should prompt reconsideration of diagnosis",
            "PLEKHG5-DISTAL-MOTOR-ONLY-NO-SENSORY: PLEKHG5/DSMA4 is distal motor only — NO sensory involvement on NCS; this distinguishes it from CMT subtypes and SIGMAR1; if sensory loss is present, consider CMT gene panel instead",
            "PLEKHG5-AR-SIBLING-RISK: autosomal recessive — 25% sibling risk; parents are obligate carriers; test all siblings; carrier testing for reproductive planning of parents' future children",
            "PLEKHG5-VOCAL-CORD-INVOLVEMENT: vocal cord weakness/hoarseness may be an early feature — assess voice and consider laryngoscopy in patients with dysphonia; neurogenic vocal cord paresis is a clue to this diagnosis",
            "PLEKHG5-SLOWLY-PROGRESSIVE-PROGNOSIS: DSMA4 is very slowly progressive — reassure families; most patients retain independent function for many years; wheelchair dependency is late if it occurs; normal lifespan expected",
            "PLEKHG5-AUTOPHAGY-NF-KB-BIOLOGY: PLEKHG5 dysfunction impairs autophagy and NF-κB signalling in motor axon terminals — emerging therapeutic target; patients suitable for longitudinal natural history studies and autophagy-modulating drug trials",
        ],
        "etiologies": [
            "Classic DSMA4 — biallelic PLEKHG5 LOF; congenital or early childhood onset; distal weakness hands + feet; slowly progressive; no sensory involvement",
            "Mild late-onset — later presentation (childhood to adolescent); subtle foot drop; gait abnormality; identified on comprehensive dHMN panel",
            "Compound heterozygous — two different PLEKHG5 pathogenic alleles; phenotype similar to homozygous LOF",
            "DSMA4 with vocal cord paresis — vocal cord weakness prominent; dysphonia early symptom; laryngoscopy confirms neurogenic paresis",
            "Severe congenital — pronounced hypotonia and distal weakness at birth; respiratory involvement in first year; rare",
            "DSMA4 with late respiratory involvement — diaphragm weakness in advanced disease; annual pulmonary function testing recommended",
        ],
        "stats": {
            "pure_lmn_pct": 95,
            "distal_predominance_pct": 95,
            "sensory_involvement_pct": 5,
            "vocal_cord_pct": 30,
        },
    },
]


def _make_cohort(gene_dict, seed):
    """Generate 40 synthetic patients for a single SMA gene."""
    rng = random.Random(seed)
    gene = gene_dict["gene"]
    patients = []

    if gene == "SMN1":
        for i in range(40):
            sma_type = rng.choices(["Type1", "Type2", "Type3", "Type4"],
                                   weights=[25, 30, 35, 10])[0]
            onset_months = {"Type1": rng.randint(0, 6),
                            "Type2": rng.randint(6, 18),
                            "Type3": rng.randint(18, 60),
                            "Type4": rng.randint(252, 480)}[sma_type]
            onset_age = round(onset_months / 12, 1)
            smn2_copies = {"Type1": rng.randint(1, 2),
                           "Type2": rng.choice([2, 3]),
                           "Type3": rng.choice([3, 4]),
                           "Type4": rng.randint(4, 5)}[sma_type]
            dx_delay = max(1, round(rng.gauss(8, 6))) if sma_type == "Type1" else \
                       max(2, round(rng.gauss(14, 8)))
            treatment = rng.choice(["Nusinersen", "Risdiplam", "Onasemnogene", "None"])
            patients.append({
                "patient_id": f"SMN1-{seed}-{i+1:03d}",
                "onset_age": onset_age,
                "dx_delay_months": dx_delay,
                "sma_type": sma_type,
                "smn2_copies": smn2_copies,
                "treatment": treatment,
                "bulbar_onset": False,
                "ftd": False,
                "respiratory_support": sma_type in ("Type1", "Type2"),
            })
    elif gene == "AR":
        for i in range(40):
            onset_age = round(rng.gauss(47, 9), 1)
            onset_age = max(20.0, onset_age)
            cag = rng.randint(37, 62)
            dx_delay = max(6, round(rng.gauss(72, 30)))
            gynecomastia = rng.random() < 0.72
            infertility = rng.random() < 0.60
            sensory = rng.random() < 0.95
            patients.append({
                "patient_id": f"AR-{seed}-{i+1:03d}",
                "onset_age": onset_age,
                "dx_delay_months": dx_delay,
                "cag_repeat": cag,
                "gynecomastia": gynecomastia,
                "infertility": infertility,
                "sensory_neuropathy": sensory,
                "bulbar_onset": rng.random() < 0.60,
                "ftd": False,
            })
    elif gene == "BICD2":
        for i in range(40):
            congenital = rng.random() < 0.60
            onset_age = round(rng.gauss(0.2, 0.1), 1) if congenital else round(rng.gauss(2.5, 1.5), 1)
            onset_age = max(0.0, onset_age)
            dx_delay = max(3, round(rng.gauss(30, 15)))
            ddh = rng.random() < 0.35
            foot_deformity = rng.random() < 0.70
            de_novo = rng.random() < 0.60
            patients.append({
                "patient_id": f"BICD2-{seed}-{i+1:03d}",
                "onset_age": onset_age,
                "dx_delay_months": dx_delay,
                "congenital": congenital,
                "ddh": ddh,
                "foot_deformity": foot_deformity,
                "de_novo": de_novo,
                "bulbar_onset": False,
                "ftd": False,
            })
    elif gene == "DYNC1H1":
        for i in range(40):
            congenital = rng.random() < 0.55
            onset_age = round(rng.gauss(0.3, 0.2), 1) if congenital else round(rng.gauss(3.0, 2.0), 1)
            onset_age = max(0.0, onset_age)
            dx_delay = max(3, round(rng.gauss(36, 18)))
            cns_malformation = rng.random() < 0.25
            foot_deformity = rng.random() < 0.75
            id_if_cns = rng.random() < 0.90 if cns_malformation else False
            patients.append({
                "patient_id": f"DYNC1H1-{seed}-{i+1:03d}",
                "onset_age": onset_age,
                "dx_delay_months": dx_delay,
                "cns_malformation": cns_malformation,
                "foot_deformity": foot_deformity,
                "intellectual_disability": id_if_cns,
                "bulbar_onset": False,
                "ftd": False,
            })
    elif gene == "VAPB":
        for i in range(40):
            onset_age = round(rng.gauss(50, 8), 1)
            onset_age = max(30.0, onset_age)
            dx_delay = max(12, round(rng.gauss(48, 24)))
            p56s = rng.random() < 0.85
            fasciculations = rng.random() < 0.90
            umc = rng.random() < 0.40
            patients.append({
                "patient_id": f"VAPB-{seed}-{i+1:03d}",
                "onset_age": onset_age,
                "dx_delay_months": dx_delay,
                "p56s_mutation": p56s,
                "fasciculations": fasciculations,
                "umc_signs": umc,
                "bulbar_onset": rng.random() < 0.20,
                "ftd": False,
            })
    elif gene == "SETX":
        for i in range(40):
            onset_age = round(rng.gauss(17, 5), 1)
            onset_age = max(8.0, min(30.0, onset_age))
            dx_delay = max(6, round(rng.gauss(30, 15)))
            umc = rng.random() < 0.75
            patients.append({
                "patient_id": f"SETX-{seed}-{i+1:03d}",
                "onset_age": onset_age,
                "dx_delay_months": dx_delay,
                "umc_signs": umc,
                "bulbar_involvement": False,
                "bulbar_onset": False,
                "ftd": False,
                "aoa2": False,
            })
    elif gene == "SIGMAR1":
        for i in range(40):
            onset_age = round(rng.gauss(16, 5), 1)
            onset_age = max(8.0, min(35.0, onset_age))
            dx_delay = max(6, round(rng.gauss(40, 18)))
            distal = rng.random() < 0.85
            sensory = rng.random() < 0.70
            vocal_cord = rng.random() < 0.20
            patients.append({
                "patient_id": f"SIGMAR1-{seed}-{i+1:03d}",
                "onset_age": onset_age,
                "dx_delay_months": dx_delay,
                "distal_predominant": distal,
                "sensory_neuropathy": sensory,
                "vocal_cord_paresis": vocal_cord,
                "bulbar_onset": False,
                "ftd": False,
            })
    else:  # PLEKHG5
        for i in range(40):
            congenital = rng.random() < 0.40
            onset_age = round(rng.gauss(0.5, 0.3), 1) if congenital else round(rng.gauss(3.5, 2.0), 1)
            onset_age = max(0.0, onset_age)
            dx_delay = max(6, round(rng.gauss(36, 18)))
            vocal_cord = rng.random() < 0.30
            patients.append({
                "patient_id": f"PLEKHG5-{seed}-{i+1:03d}",
                "onset_age": onset_age,
                "dx_delay_months": dx_delay,
                "congenital": congenital,
                "vocal_cord_paresis": vocal_cord,
                "pure_lmn": True,
                "sensory_involvement": rng.random() < 0.05,
                "bulbar_onset": False,
                "ftd": False,
            })

    ages = [p["onset_age"] for p in patients]
    delays = [p["dx_delay_months"] for p in patients]
    return {
        "gene": gene_dict["gene"],
        "protein": gene_dict["protein"],
        "alias": gene_dict["alias"],
        "locus": gene_dict["locus"],
        "aa": gene_dict["aa"],
        "kDa": gene_dict["kDa"],
        "omim_gene": gene_dict["omim_gene"],
        "omim_disease": gene_dict.get("omim_disease", ""),
        "inheritance": gene_dict["inheritance"],
        "gene_class": gene_dict["gene_class"],
        "key_alerts": gene_dict["key_alerts"],
        "etiologies": gene_dict["etiologies"],
        "stats": gene_dict["stats"],
        "sample_patients": patients[:10],
        "computed": {
            "n_patients": len(patients),
            "mean_dx_age": round(sum(ages) / len(ages), 1),
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
        },
    }


def _build_all():
    cohorts = []
    for i, gene in enumerate(SMA_GENES):
        cohorts.append(_make_cohort(gene, SEED_BASE + i))
    return cohorts


def get_overview():
    cohorts = _build_all()
    all_ages = [p["onset_age"] for c in cohorts for p in c["sample_patients"]]
    all_delays = [p["dx_delay_months"] for c in cohorts for p in c["sample_patients"]]
    top_alerts = []
    for c in cohorts:
        if c["key_alerts"]:
            top_alerts.append(c["key_alerts"][0])
    return {
        "atlas": "Hereditary-SMA-Atlas — Complete 8-Gene Hereditary Spinal Muscular Atrophy Atlas",
        "subtitle": (
            "SMN1 (SMA 5q / NBS / 3 FDA treatments) · AR (Kennedy SBMA / CAG repeat) · "
            "BICD2 (SMALED2 / DDH) · DYNC1H1 (SMALED1 / CNS malformation) · "
            "VAPB (ALS8 / P56S founder) · SETX (ALS4 / juvenile / NO bulbar) · "
            "SIGMAR1 (ALS16 / distal) · PLEKHG5 (DSMA4 / NF-κB autophagy) — "
            "320 Patients (8×40, Seeds 1718–1725)"
        ),
        "total_patients": 320,
        "seed_range": "1718–1725",
        "aggregate_stats": {
            "genes_covered": 8,
            "patients_per_gene": 40,
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
        },
        "genes": [
            {
                "gene": c["gene"],
                "locus": c["locus"],
                "aa": c["aa"],
                "kDa": c["kDa"],
                "mean_dx_age": c["computed"]["mean_dx_age"],
                "mean_dx_delay_months": c["computed"]["mean_dx_delay_months"],
                "n_patients": c["computed"]["n_patients"],
            }
            for c in cohorts
        ],
        "top_alerts": top_alerts,
    }


def get_breakdown():
    cohorts = _build_all()
    return [
        {
            "gene": c["gene"],
            "protein": c["protein"],
            "locus": c["locus"],
            "aa": c["aa"],
            "kDa": c["kDa"],
            "omim_gene": c["omim_gene"],
            "omim_disease": c["omim_disease"],
            "inheritance": c["inheritance"],
            "gene_class": c["gene_class"],
            "key_alerts": c["key_alerts"],
            "etiologies": c["etiologies"],
            "alias": c["alias"],
            "stats": c["stats"],
            "sample_patients": c["sample_patients"],
            "computed": c["computed"],
        }
        for c in cohorts
    ]


def get_definitions():
    return {
        "concepts": {
            "SMA 5q (SMN1) vs Other SMA — Critical Distinction": (
                "SMA 5q (SMN1-related) is the most common form and has three approved disease-modifying "
                "treatments (nusinersen, onasemnogene, risdiplam). All other SMA subtypes (BICD2, DYNC1H1, "
                "VAPB, SETX, SIGMAR1, PLEKHG5) have NO approved treatments. When a patient is labelled "
                "'SMA', confirm whether SMN1 gene is confirmed — only SMN1-related SMA has treatment options."
            ),
            "SMN2 Copy Number — Pharmacogenomics of SMA": (
                "SMN2 copy number is the single most important modifier of SMA 5q severity: "
                "1 SMN2 copy = SMA Type 1 (most severe, never sit); "
                "2 SMN2 copies = Type 1-2 (borderline); "
                "3 SMN2 copies = Type 2-3 (sit not walk); "
                "4 SMN2 copies = Type 3-4 (mild, walk). "
                "SMN2 copy number MUST be tested at diagnosis for prognosis and to guide treatment urgency. "
                "All three approved treatments target SMN2 exon 7 splicing (nusinersen, risdiplam) or "
                "replace SMN1 function (onasemnogene). Higher SMN2 copy numbers mean more residual SMN protein "
                "and better response to splicing modifiers."
            ),
            "Kennedy Disease SBMA vs ALS — Key Distinguishing Features": (
                "Kennedy SBMA (AR gene) is commonly misdiagnosed as ALS. Critical differences: "
                "(1) Sensory neuropathy — Kennedy has large fibre sensory loss; ALS is purely motor; "
                "(2) Progression — Kennedy very slowly progressive (years-decades); ALS rapid (months-years); "
                "(3) Endocrine — gynaecomastia in >70% Kennedy; absent in ALS; "
                "(4) X-linked recessive — only males affected (typically); "
                "(5) Bulbar — prominent in Kennedy but very slowly progressive; "
                "(6) No approved treatment for Kennedy (leuprolide trial was NEGATIVE); "
                "(7) Normal or near-normal lifespan in Kennedy vs median 2-5y in ALS. "
                "Confusing Kennedy for ALS causes unnecessary prognostic despair and incorrect trial enrollment."
            ),
            "SMALED1 (DYNC1H1) vs SMALED2 (BICD2) — Lower Extremity Dominant SMA": (
                "Both DYNC1H1 and BICD2 cause lower extremity predominant SMA with congenital or childhood onset. "
                "Key distinguishing features: "
                "(1) CNS malformations — DYNC1H1 has cortical malformations (pachygyria/lissencephaly) in ~25%; "
                "BICD2 brain MRI is NORMAL; "
                "(2) Cognition — DYNC1H1 CNS subgroup has intellectual disability; BICD2 cognition ALWAYS NORMAL; "
                "(3) MRI is mandatory for all DYNC1H1 patients to classify CNS vs pure motor subgroup; "
                "(4) DDH (hip dysplasia) occurs in both but more common in BICD2; "
                "(5) Both are autosomal dominant with high de novo rates."
            ),
            "ALS4 (SETX) — Juvenile ALS with Excellent Prognosis": (
                "ALS4 caused by SETX mutations is fundamentally different from sporadic ALS: "
                "(1) Juvenile onset (10-25 years); "
                "(2) NO bulbar involvement — speech and swallowing preserved throughout course; "
                "(3) Very slowly progressive — survival for decades, many walk 20+ years; "
                "(4) UMN signs are common (hyperreflexia, Babinski); "
                "(5) Normal intelligence; "
                "(6) Autosomal dominant. "
                "Prognosis must be clearly distinguished from classical ALS to avoid unnecessary despair. "
                "Note: biallelic AR SETX mutations cause an entirely different disease — AOA2 "
                "(ataxia with oculomotor apraxia type 2), characterised by cerebellar ataxia, elevated AFP."
            ),
            "ER-Mitochondria Contact Sites (MAM) — Shared Mechanism in SMA Subtypes": (
                "Three SMA/MND genes in this atlas — VAPB, SIGMAR1, PLEKHG5 — converge on ER-mitochondria "
                "contact site biology (MAM = mitochondria-associated membranes): "
                "VAPB tethers ER to mitochondria via PTPIP51 interaction; "
                "SIGMAR1 is a MAM-resident chaperone regulating calcium transfer; "
                "PLEKHG5 regulates autophagy (autophagic vesicle formation at axon terminals). "
                "This mechanistic convergence suggests potential for shared therapeutic targeting of "
                "ER stress, mitochondrial calcium, and autophagy pathways across these subtypes."
            ),
            "Newborn Screening for SMA — Global Standard": (
                "SMA 5q (SMN1) is now on the newborn screening panel in 40+ countries. "
                "Standard: DBS (dried blood spot) testing for SMN1 exon 7 homozygous deletion. "
                "Pre-symptomatic treatment in NBS-detected SMA Type 1 infants with nusinersen or "
                "onasemnogene results in outcomes indistinguishable from healthy controls in many cases. "
                "Advocate for NBS in any country/region where it is not yet implemented. "
                "Do NOT wait for symptoms to begin treatment in NBS-positive infants — immediate referral to "
                "SMA specialist centre is mandatory. The treatment window is before motor neuron loss occurs."
            ),
        },
        "pharmacological_distinctions": [
            "SMN1/SMA 5q ONLY — Nusinersen (Spinraza): intrathecal ASO (splice-switching) promoting SMN2 exon 7 inclusion; FDA 2016; dosing 4 loading doses then every 4 months intrathecal; used ALL ages including adults; CHERISH and ENDEAR trial evidence",
            "SMN1/SMA 5q ONLY — Onasemnogene abeparvovec (Zolgensma): IV gene therapy (AAV9-SMN1); FDA May 2019; single IV infusion; label <2 years; weight limit 13.5 kg; liver toxicity monitoring mandatory; ONE ADMINISTRATION ONLY (anti-AAV9 antibodies preclude re-dosing)",
            "SMN1/SMA 5q ONLY — Risdiplam (Evrysdi): oral small molecule splicing modifier (SMN2 exon 7); FDA August 2020; daily oral liquid; used ≥2 months age; FIREFISH/SUNFISH trials; convenient (no intrathecal); used across Types 1-4",
            "AR/Kennedy SBMA — Leuprolide (GnRH agonist): Phase 3 JASMINS trial NEGATIVE — do NOT use; no disease-modifying therapy approved for Kennedy disease; supportive care and physiotherapy only",
            "VAPB/ALS8 — No approved therapy; σ1R agonists (fluvoxamine, SA4503) under investigation for ER stress neuroprotection; supportive care as in ALS but prognosis much better",
            "SETX/ALS4 — No approved therapy; physiotherapy; very slowly progressive; reassurance on prognosis critical; RNA helicase/R-loop pathway drugs under investigation",
            "SIGMAR1/ALS16 — No approved therapy; σ1R agonists (fluvoxamine) biologically rational but no approved indication; clinical trials needed; supportive care",
            "BICD2/SMALED2, DYNC1H1/SMALED1, PLEKHG5/DSMA4 — No disease-modifying treatment; orthopaedic management (DDH, foot deformities, scoliosis); physiotherapy; respiratory monitoring; antiepileptics for DYNC1H1 CNS subgroup",
        ],
        "key_standards": [
            "SMA CARE STANDARDS: International Standard of Care for SMA (Mercuri et al. Lancet Neurology 2018) — multidisciplinary team mandatory (neurology, pulmonology, orthopaedics, nutrition, rehab, genetics); updated 2022 consensus guidelines",
            "NBS SMN1: SMA included in RUSP (Recommended Uniform Screening Panel) USA 2018; EUNENBS European NBS recommendations; DBS testing for SMN1 exon 7 deletion (qPCR or MLPA); immediate referral on positive screen before symptoms",
            "SBMA/KENNEDY STANDARDS: European Neuromuscular Centre (ENMC) 2013 consensus; annual monitoring CK, testosterone, LH, FSH, semen analysis; respiratory and swallowing assessment; avoid testosterone supplementation paradoxically can worsen",
            "DYNC1H1/BICD2 STANDARDS: brain MRI mandatory for all DYNC1H1 patients (cortical malformation subgroup); hip ultrasound all BICD2 infants (DDH); comprehensive multidisciplinary assessment; scoliosis surveillance in all non-ambulant SMA patients",
            "ALS4/SETX: no formal international guidelines; manage as juvenile-onset MND; physiotherapy-led; prognosis counselling critical; research cohort enrolment recommended (natural history data sparse)",
            "RESPIRATORY MONITORING ALL SMA SUBTYPES: annual spirometry (FVC, FVC lying); early NIV discussion when FVC <50% or symptomatic nocturnal hypoventilation; sleep study (polysomnography) when symptomatic",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = get_breakdown()
    print(json.dumps(bd[0], indent=2)[:2000])
    print("\n=== DEFINITIONS ===")
    print(json.dumps(get_definitions(), indent=2)[:1000])
