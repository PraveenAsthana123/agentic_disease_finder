#!/usr/bin/env python3
"""Hereditary-Brain-Malformation-Atlas — Complete 8-Gene Brain Malformation (MCD) Atlas.

PAFAH1B1  (platelet-activating-factor acetylhydrolase-1b regulatory-subunit-1 / LIS1;
           380 aa; 17p13.3; AD/de-novo; Lissencephaly Type 1 — Grade 1-2 agyria/pachygyria;
           Miller-Dieker Syndrome when 17p13.3 deletion includes YWHAE;
           seed SEED_BASE+0).
DCX        (Doublecortin; 360 aa; Xq22.3; XL;
           Lissencephaly (hemizygous males) / Subcortical Band Heterotopia (heterozygous females);
           X-linked inheritance — male/female OPPOSITE phenotype rule;
           seed SEED_BASE+1).
TUBA1A     (Tubulin alpha-1A; 451 aa; 12q13.12; AD de-novo;
           Pachygyria + cerebellar hypoplasia + corpus callosum dysgenesis;
           Tubulinopathy spectrum — most common after LIS1/DCX;
           seed SEED_BASE+2).
FLNA       (Filamin A; 2647 aa; Xq28; XL dominant;
           Bilateral periventricular nodular heterotopia (BPNH);
           Epilepsy with NORMAL IQ in females; lethal in hemizygous males;
           seed SEED_BASE+3).
ASPM       (abnormal spindle microtubule assembly; 3477 aa; 1q31.3; AR;
           Primary Microcephaly MCPH5 — OFC -7 to -10 SD at birth;
           No structural malformation; simplified gyral pattern; mild ID;
           seed SEED_BASE+4).
CDK5RAP2   (CDK5 regulatory-subunit-associated protein 2; 1893 aa; 9q33.2; AR;
           Primary Microcephaly MCPH3 — less severe than ASPM;
           seed SEED_BASE+5).
ADGRG1     (adhesion G-protein-coupled receptor G1 / GPR56; 693 aa; 16q21; AR;
           Bilateral Frontoparietal Polymicrogyria (BFPP) — strabismus + hypotonia + dysarthria;
           seed SEED_BASE+6).
RELN       (Reelin; 3461 aa; 7q22.1; AR/AD;
           AR homozygous: Lissencephaly with Cerebellar Hypoplasia (LCH) — most severe;
           AD heterozygous: LQTS-associated autism / mild cortical migration delay;
           seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2046-2053).
"""

import random

SEED_BASE = 2046

BM_GENES = [
    # -- PAFAH1B1 / LIS1 — Lissencephaly Type 1 / Miller-Dieker Syndrome (AD) -----------
    {
        "gene": "PAFAH1B1",
        "alt_name": "PAFAH1B1 / LIS1 (LIS1-380aa-17p13.3 / AD — Lissencephaly-Type-1-Grade-1-2-Agyria — Miller-Dieker-Syndrome-17p13.3-Deletion-YWHAE — Seizures-Onset-<3m-Hypotonia-Posturing)",
        "protein": (
            "PAFAH1B1/LIS1 -- 17p13.3 AD -- LIS1-380aa -- "
            "Lissencephaly-Type-1-Classic-Agyria-Grade-1-2-Posterior>Anterior-Gradient -- "
            "Miller-Dieker-Syndrome-17p13.3-Contiguous-Deletion-YWHAE-Facial-Dysmorphism -- "
            "Seizures-Infantile-Spasms-<3-Months-Hypotonia-Posturing -- "
            "MOST-COMMON-GENE-Lissencephaly-Type-1"
        ),
        "locus": "17p13.3",
        "protein_size": "380 aa",
        "inheritance": (
            "AD (autosomal dominant) — PAFAH1B1 haploinsufficiency; "
            "Most cases de novo; 17p13.3 deletion (Miller-Dieker) includes PAFAH1B1 + YWHAE; "
            "Intragenic mutations: isolated lissencephaly sequence (ILS); "
            "Parental balanced translocation: rare — recurrence risk up to 25-50%; "
            "Point mutations: isolated lissencephaly without MDS facial features"
        ),
        "age_of_onset": (
            "Prenatal: reduced gyration on fetal MRI from 20-22 weeks — smooth brain posterior > anterior; "
            "Neonatal: hypotonia, poor feeding, dysmorphic features (MDS); "
            "Seizures: infantile spasms 3-6 months — West syndrome; later mixed epilepsy (Lennox-Gastaut-like); "
            "Developmental delay: profound (MDS) to severe-moderate (ILS); "
            "Posturing: cortical thumbs, fisting; opisthotonus; "
            "Prognosis: MDS very severe — most die by 10 yr; ILS variable, some into adulthood; "
            "MDS facial features: bitemporal hollowing, upturned nose, thin vermilion, prominent occiput; "
            "Feeding difficulties: gastrostomy often required"
        ),
        "key_biomarker": (
            "Brain MRI: agyria (Grade 1 — complete smooth brain) or pachygyria (Grade 2 — thick cortex 10-20 mm); "
            "Posterior > anterior gradient: occipital most severe — DISTINGUISHES from DCX (anterior > posterior); "
            "Cortical thickness: 10-20 mm (normal 3-4 mm); "
            "Absent or simplified gyral pattern; absent corpus callosum (MDS); "
            "Chromosomal microarray (CMA): 17p13.3 deletion — includes PAFAH1B1 ± YWHAE (MDS); "
            "Intragenic PAFAH1B1 sequencing: point mutations, small deletions; "
            "EEG: hypsarrhythmia (West syndrome); multifocal spikes; "
            "Fetal MRI: reduced gyrification at 22-24 wks (normally 26-28 wks for primary sulci)"
        ),
        "pathognomonic": (
            "SMOOTH BRAIN (agyria) with THICK CORTEX (10-20 mm) on MRI + POSTERIOR > ANTERIOR gradient = PAFAH1B1 lissencephaly; "
            "MILLER-DIEKER SYNDROME: lissencephaly + bitemporal hollowing + cardiac + growth retardation + 17p13.3 deletion; "
            "DISTINGUISH DCX lissencephaly: anterior > posterior gradient (males) or band heterotopia (females); "
            "DISTINGUISH TUBA1A: pachygyria + cerebellar hypoplasia + agenesis CC (tubulinopathy); "
            "SEIZURE ONSET <3 MONTHS in smooth brain = PAFAH1B1 / DCX until proven otherwise; "
            "FETAL MRI at 22 wks: smooth brain = referral for urgent chromosomal microarray + PAFAH1B1/DCX panel"
        ),
        "treatment": (
            "Anti-seizure medications: vigabatrin first-line for infantile spasms; ACTH second-line; "
            "Later epilepsy: valproate, clonazepam (polytherapy common); ketogenic diet — evidence limited; "
            "AVOID: carbamazepine / oxcarbazepine if generalised — may worsen; "
            "Gastrostomy: for severe dysphagia — early placement improves nutrition; "
            "Physiotherapy + occupational therapy + speech therapy: early intervention; "
            "Pulmonary: chest physiotherapy; recurrent aspiration pneumonia management; "
            "Palliative care planning: MDS prognosis; GOC discussion early; "
            "Prenatal: fetal MRI at 20 wks if family history; CMA if fetal brain anomaly detected"
        ),
        "critical_flags": [
            "PAFAH1B1-POSTERIOR-ANTERIOR-GRADIENT-PATHOGNOMONIC",
            "PAFAH1B1-MILLER-DIEKER-17p13.3-DELETION-YWHAE",
            "PAFAH1B1-INFANTILE-SPASMS-<3-MONTHS",
            "PAFAH1B1-CMA-FIRST-LINE-THEN-SEQUENCE",
            "PAFAH1B1-FETAL-MRI-22-WKS-SMOOTH-BRAIN",
            "PAFAH1B1-CORTEX-THICKNESS-10-20mm",
            "PAFAH1B1-MDS-PROGNOSIS-DEATH-<10yr",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- DCX — Doublecortin X-Linked Lissencephaly / Band Heterotopia (XL) ---------------
    {
        "gene": "DCX",
        "alt_name": "DCX (Doublecortin-360aa-Xq22.3 / XL — Males-Lissencephaly-ANTERIOR>POSTERIOR — Females-Band-Heterotopia-DOUBLE-CORTEX — OPPOSITE-PHENOTYPE-RULE-X-Inactivation)",
        "protein": (
            "DCX -- Xq22.3 XL -- DCX-360aa -- "
            "Males-Hemizygous-Lissencephaly-Anterior-Greater-Posterior-Gradient -- "
            "Females-Heterozygous-Subcortical-Band-Heterotopia-Double-Cortex-Sign -- "
            "OPPOSITE-PHENOTYPE-MALE-FEMALE-X-INACTIVATION-MOSAICISM -- "
            "Microtubule-Associated-Neuronal-Migration"
        ),
        "locus": "Xq22.3",
        "protein_size": "360 aa",
        "inheritance": (
            "X-linked — DCX is X-linked; "
            "Hemizygous males: classic lissencephaly (full mutation effect — no normal allele); "
            "Heterozygous females: subcortical band heterotopia due to random X-inactivation mosaicism; "
            "De novo in ~90% of males with lissencephaly; "
            "Carrier females (band heterotopia): may have mild epilepsy and near-normal IQ — often diagnosed incidentally on MRI; "
            "X-inactivation skewing: females with skewed inactivation (95:5) may have more severe phenotype"
        ),
        "age_of_onset": (
            "Males (lissencephaly): neonatal hypotonia, seizure onset 3-6 months (infantile spasms); "
            "Cognitive delay: severe-profound in males; "
            "Anterior > posterior gradient: DISTINGUISHES from PAFAH1B1 (which is posterior > anterior); "
            "Females (band heterotopia / SBH): epilepsy onset variable — 2nd-3rd decade most common; "
            "SBH females: IQ usually low-normal to borderline; functional seizures; "
            "Double cortex: inner band of grey matter separated from cortex by white matter layer; "
            "Focal cortical dysplasia: some female carriers have focal anomalies rather than diffuse band; "
            "Mosaic males (somatic mosaicism): intermediate phenotype — band heterotopia"
        ),
        "key_biomarker": (
            "Brain MRI males: lissencephaly ANTERIOR > POSTERIOR gradient (opposite of PAFAH1B1); "
            "Brain MRI females: bilateral subcortical band heterotopia — 'double cortex sign' on axial cuts; "
            "White matter cleft separating outer cortex from inner grey matter band; "
            "DCX gene sequencing: exons 1-9 (missense/truncation); X-linked panel; "
            "MLPA: exon deletions/duplications; "
            "EEG: hypsarrhythmia (males); multifocal spikes/GSWD (females); "
            "Karyotype + CMA: exclude chromosomal anomaly first; "
            "Maternal testing: carrier female (SBH) may be clinically silent; brain MRI mother recommended"
        ),
        "pathognomonic": (
            "ANTERIOR > POSTERIOR lissencephaly in MALE infant = DCX until proven otherwise (vs PAFAH1B1 posterior > anterior); "
            "BILATERAL SUBCORTICAL BAND HETEROTOPIA ('double cortex sign') in FEMALE = DCX heterozygous carrier; "
            "MALE WITH LISSENCEPHALY + MOTHER WITH BAND HETEROTOPIA = X-linked DCX confirmed; "
            "OPPOSITE PHENOTYPE RULE: hemizygous male → lissencephaly; heterozygous female → band heterotopia; "
            "MOSAIC MALE: somatic DCX mosaicism → band heterotopia in male (diagnose by sequencing blood + saliva + buccal); "
            "MATERNAL BRAIN MRI: always recommended when DCX lissencephaly found in male proband"
        ),
        "treatment": (
            "Males (lissencephaly): vigabatrin for infantile spasms; ACTH; polytherapy (valproate, clonazepam); "
            "Ketogenic diet: moderate evidence for drug-resistant epilepsy; "
            "Females (SBH): ASMs tailored to seizure type — lamotrigine, levetiracetam for focal; "
            "Surgery: lesionectomy or corpus callosotomy if drug-resistant in SBH females; "
            "Gastrostomy: dysphagia in severely affected males; "
            "Early intervention: physio, OT, speech therapy; "
            "Genetic counselling: X-linked — mothers of affected males should have MRI; "
            "Prenatal: fetal MRI + DCX sequencing in at-risk pregnancies"
        ),
        "critical_flags": [
            "DCX-ANTERIOR-POSTERIOR-GRADIENT-MALES-DISTINGUISH-PAFAH1B1",
            "DCX-DOUBLE-CORTEX-FEMALES-BAND-HETEROTOPIA",
            "DCX-OPPOSITE-PHENOTYPE-MALE-FEMALE-X-INACTIVATION",
            "DCX-MATERNAL-MRI-MANDATORY-WHEN-MALE-PROBAND",
            "DCX-MOSAIC-MALE-SEQUENCING-BLOOD-SALIVA-BUCCAL",
            "DCX-SBH-FEMALES-NEAR-NORMAL-IQ-EPILEPSY",
            "DCX-X-LINKED-90pct-DE-NOVO",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- TUBA1A — Tubulinopathy Pachygyria + Cerebellar Hypoplasia (AD de novo) ----------
    {
        "gene": "TUBA1A",
        "alt_name": "TUBA1A (Tubulin-Alpha-1A-451aa-12q13.12 / AD-de-novo — Pachygyria-Cerebellar-Hypoplasia-Corpus-Callosum-Dysgenesis — Tubulinopathy-MOST-COMMON-After-LIS1-DCX — Basal-Ganglia-Dysplasia-PATHOGNOMONIC)",
        "protein": (
            "TUBA1A -- 12q13.12 AD-de-novo -- TUBA1A-451aa -- "
            "Pachygyria-Cerebellar-Hypoplasia-Agenesis-Corpus-Callosum-Tubulinopathy -- "
            "Basal-Ganglia-Dysplasia-Dysmorphic-Caudate-Putamen-PATHOGNOMONIC-Tubulinopathy -- "
            "MOST-COMMON-Tubulinopathy-Gene-After-LIS1-DCX -- "
            "Alpha-Tubulin-Microtubule-Protofilament-Neuronal-Migration"
        ),
        "locus": "12q13.12",
        "protein_size": "451 aa",
        "inheritance": (
            "AD — almost exclusively de novo; "
            "TUBA1A encodes alpha-tubulin 1A — essential component of microtubule protofilaments; "
            "Dominant negative mechanism: mutant alpha-tubulin disrupts microtubule polymerisation; "
            "Recurrence risk: <1% (de novo) unless parental gonadal mosaicism; "
            "Severity correlates with mutation position (GTP-binding domain vs polymerisation interface)"
        ),
        "age_of_onset": (
            "Prenatal: abnormal fetal MRI from 20-22 weeks — simplified gyration, thin CC, small cerebellum; "
            "Neonatal: hypotonia, poor feeding, microcephaly (OFC -2 to -4 SD typical); "
            "Seizures: onset 1st year — focal or generalised; variable severity; "
            "Development: severe-profound ID in most; some milder cases with dysarthria + ataxia; "
            "Cerebellar hypoplasia: hypotonia, truncal ataxia, intention tremor; "
            "Corpus callosum: dysgenesis (partial agenesis to complete agenesis) — common; "
            "Basal ganglia: dysmorphic/fused caudate-putamen on MRI — DIAGNOSTIC for tubulinopathy spectrum; "
            "Motor: spastic diplegia / quadriplegia in severe cases"
        ),
        "key_biomarker": (
            "Brain MRI: pachygyria (4-6 mm thick cortex, simplified gyration); "
            "Cerebellar hypoplasia: small vermis ± hemispheres; "
            "Corpus callosum: thin, partial or complete agenesis (hypogenesis); "
            "BASAL GANGLIA DYSPLASIA: dysmorphic fused/undivided putamen-caudate on axial MRI — PATHOGNOMONIC for tubulinopathy; "
            "Brain stem: hypoplastic pons; "
            "TUBA1A sequencing: missense mutations dominant; "
            "Chromosomal microarray: exclude 12q13.12 deletion (rare); "
            "EEG: focal cortical irritability; "
            "Developmental assessment: Griffiths / Bayley scales to track trajectory"
        ),
        "pathognomonic": (
            "PACHYGYRIA + CEREBELLAR HYPOPLASIA + CORPUS CALLOSUM DYSGENESIS + DYSMORPHIC BASAL GANGLIA = Tubulinopathy (TUBA1A most common); "
            "BASAL GANGLIA DYSPLASIA (fused/undivided caudate-putamen) on MRI = TUBULINOPATHY SPECTRUM — test TUBA1A, TUBB2B, TUBB3, TUBB; "
            "DISTINGUISH PAFAH1B1: no cerebellar hypoplasia; posterior > anterior gradient; normal basal ganglia; "
            "DISTINGUISH RELN: cerebellar hypoplasia present but AR inheritance + hypoplastic brainstem; "
            "TUBA1A SPECTRUM: ranges from complete lissencephaly to mild pachygyria — mutation position predicts severity; "
            "PRENATAL MRI at 22 WKS: abnormal — thin CC + simplified sulcation + small cerebellum = urgent TUBA1A panel"
        ),
        "treatment": (
            "Anti-seizure medications: levetiracetam, valproate first-line; "
            "Refractory epilepsy: ketogenic diet, VNS; "
            "Feeding: nasogastric → gastrostomy for dysphagia; "
            "Physiotherapy: for spasticity, motor development; "
            "Occupational therapy + speech and language therapy: early; "
            "Intrathecal baclofen pump: for severe spasticity in quadriplegia; "
            "Ophthalmology: strabismus correction; "
            "Genetic counselling: de novo — recurrence <1%; prenatal diagnosis available; "
            "Research: no approved targeted therapy; microtubule stabilisers investigational"
        ),
        "critical_flags": [
            "TUBA1A-BASAL-GANGLIA-DYSPLASIA-PATHOGNOMONIC-TUBULINOPATHY",
            "TUBA1A-PACHYGYRIA-CEREBELLAR-HYPOPLASIA-CC-DYSGENESIS",
            "TUBA1A-DE-NOVO-ALMOST-EXCLUSIVELY",
            "TUBA1A-SPECTRUM-TUBA1A-TUBB2B-TUBB3",
            "TUBA1A-PRENATAL-MRI-ABNORMAL-22-WKS",
            "TUBA1A-DISTINGUISH-PAFAH1B1-CEREBELLAR-HYPOPLASIA",
            "TUBA1A-DOMINANT-NEGATIVE-MICROTUBULE",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- FLNA — Filamin A Bilateral Periventricular Nodular Heterotopia (XL dominant) ----
    {
        "gene": "FLNA",
        "alt_name": "FLNA (Filamin-A-2647aa-Xq28 / XL-dominant — Bilateral-Periventricular-Nodular-Heterotopia-BPNH — Epilepsy-NORMAL-IQ-Females — LETHAL-Hemizygous-Males — Cardiac-Vascular-Features)",
        "protein": (
            "FLNA -- Xq28 XL-dominant -- FLNA-2647aa -- "
            "Bilateral-Periventricular-Nodular-Heterotopia-BPNH-Lining-Lateral-Ventricles -- "
            "Epilepsy-Focal-Normal-IQ-Typical-Females -- "
            "LETHAL-Hemizygous-Males-Cardiovascular-PDA-Aortic-Dilation -- "
            "Actin-Cross-Linking-Cytoskeletal-Scaffold-Neuronal-Migration-Arrest"
        ),
        "locus": "Xq28",
        "protein_size": "2647 aa",
        "inheritance": (
            "X-linked dominant — FLNA is X-linked; "
            "Heterozygous females: classic BPNH phenotype — epilepsy + normal IQ; "
            "Hemizygous males: usually lethal (cardiac + vascular abnormalities; absent heterotopia nodules); "
            "Male livebirths: rare — typically hypomorphic alleles or mosaicism; "
            "De novo in ~50% of index cases; familial AD-like pedigree through females; "
            "Somatic mosaicism: in some males with milder presentations"
        ),
        "age_of_onset": (
            "Females: epilepsy onset typically 2nd-3rd decade (focal seizures, secondarily generalised); "
            "IQ: usually NORMAL or low-normal — females often diagnosed in adulthood after first seizure; "
            "Periventricular nodules: subependymal grey matter nodules lining lateral ventricles — bilateral symmetrical; "
            "Cardiovascular: PDA, aortic root dilation, MVP, coarctation — cardiac screen at diagnosis; "
            "Joint hypermobility: connective tissue features; "
            "Males (livebirth survivors): severe cardiac malformations; bowel malrotation; "
            "Ehlers-Danlos-like features: skin extensibility, easy bruising; "
            "PVNH: heterotopic neurons form epileptogenic foci"
        ),
        "key_biomarker": (
            "Brain MRI T1: bilateral nodules lining lateral ventricle walls — isointense to grey matter; "
            "Best seen on coronal T1 with inversion recovery (IR) sequences; "
            "SYMMETRICAL bilateral > unilateral (unilateral — consider FLNA or other genes); "
            "FLNA sequencing: truncating mutations most common in BPNH; "
            "X-linked panel: FLNA coding region; "
            "Echocardiography: MVP, aortic root measurement; PDA screen; "
            "MRA aorta: dilation; "
            "EEG: focal temporal / occipital; "
            "Chromosomal microarray: exclude Xq28 deletion; "
            "Skin biopsy: fibroblast FLNA protein analysis if sequencing uncertain"
        ),
        "pathognomonic": (
            "BILATERAL PERIVENTRICULAR NODULAR HETEROTOPIA + NORMAL IQ + EPILEPSY IN FEMALE = FLNA until proven otherwise; "
            "BPNH NODULES: grey matter isointense, lining lateral ventricle — do NOT enhance; "
            "FEMALE WITH BPNH + FAMILY HISTORY THROUGH FEMALES ONLY (male lethality) = X-linked FLNA; "
            "CARDIAC SCREEN MANDATORY: PDA, MVP, aortic dilation occur with FLNA even without overt cardiac symptoms; "
            "DISTINGUISH TSC (tuberose sclerosis): subependymal nodules calcify and enhance — FLNA nodules do NOT; "
            "DISTINGUISH sporadic BPNH: bilateral BPNH without family history — still test FLNA; "
            "MALE FETUS with BPNH on fetal MRI + FLNA maternal mutation = male carrier prognosis poor"
        ),
        "treatment": (
            "Anti-seizure medications: lamotrigine, levetiracetam, carbamazepine for focal seizures; "
            "Epilepsy surgery: selective amygdalohippocampectomy + lesionectomy (resection of PVNH nodule) — case series positive results; "
            "Laser interstitial thermal therapy (LITT): minimally invasive nodule ablation — emerging; "
            "Cardiac: PDA closure; aortic root monitoring — prophylactic surgery if >45 mm; "
            "Joint hypermobility: physiotherapy; avoid hyperextension; "
            "Genetic counselling: X-linked dominant — 50% risk daughters; male lethality; "
            "Pregnancy: higher risk of vascular complications — cardiology co-management; "
            "Cardiac monitoring: annual echocardiogram from diagnosis"
        ),
        "critical_flags": [
            "FLNA-BPNH-NORMAL-IQ-EPILEPSY-FEMALES",
            "FLNA-LETHAL-HEMIZYGOUS-MALES",
            "FLNA-CARDIAC-PDA-AORTIC-DILATION-MANDATORY-SCREEN",
            "FLNA-NODULES-DO-NOT-CALCIFY-DO-NOT-ENHANCE-DISTINGUISH-TSC",
            "FLNA-X-LINKED-DOMINANT-MALE-LETHALITY",
            "FLNA-SURGERY-LITT-PVNH-ABLATION-EMERGING",
            "FLNA-BILATERAL-SYMMETRICAL-PVNH-CLASSIC",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- ASPM — Primary Microcephaly MCPH5 (AR) ------------------------------------------
    {
        "gene": "ASPM",
        "alt_name": "ASPM (ASPM-3477aa-1q31.3 / AR — Primary-Microcephaly-MCPH5-OFC-MINUS-7-TO-10-SD — MOST-COMMON-AR-Microcephaly-Gene — Simplified-Gyral-Pattern — Normal-Cortical-Layering)",
        "protein": (
            "ASPM -- 1q31.3 AR -- ASPM-3477aa -- "
            "Primary-Microcephaly-MCPH5-Most-Common-AR-Primary-Microcephaly-Gene -- "
            "OFC-Minus-7-to-10-SD-At-Birth-Progressive-Simplified-Gyral-Pattern -- "
            "Normal-Cortical-Lamination-No-Heterotopia-Distinguishes-MCD -- "
            "Centriolar-Mitotic-Spindle-Orientation-Apical-Progenitor"
        ),
        "locus": "1q31.3",
        "protein_size": "3477 aa",
        "inheritance": (
            "AR (autosomal recessive) — ASPM biallelic LOF mutations; "
            "ASPM is the most commonly mutated gene in primary microcephaly (MCPH5); "
            "Founder mutations in consanguineous populations (Middle East, South Asia, Pakistan); "
            "Carrier frequency in consanguineous populations: ~1 in 40-50; "
            "Genotype-phenotype: truncating mutations more severe; missense near IQ motifs — variable"
        ),
        "age_of_onset": (
            "Prenatal: small head circumference on fetal ultrasound from 20 weeks; "
            "Birth: OFC -7 to -10 SD (often -4 to -7 SD at birth; progresses to -10+ SD postnatally); "
            "No structural brain malformation: cortex layered normally — NO heterotopia, NO lissencephaly; "
            "Simplified gyral pattern: reduced complexity of sulcation (not frank pachygyria); "
            "Intelligence: mild-moderate ID (often functions better than imaging suggests); "
            "Motor: mild delay — ambulate independently in most; "
            "Speech: delayed but present; "
            "Epilepsy: 10-20% — usually mild and drug-responsive; "
            "Behaviour: autistic traits common; "
            "Growth: proportionate — not associated with dwarfism"
        ),
        "key_biomarker": (
            "Head circumference: OFC ≤ -3 SD at birth (diagnostic for primary microcephaly); "
            "Serial OFC: progressive — microcephaly worsens in first year; "
            "Brain MRI: simplified gyral pattern; thin cortex; normal layering; reduced white matter; "
            "No pachygyria (distinguishes from PAFAH1B1/DCX/TUBA1A); "
            "Chromosomal microarray: exclude 1q31.3 deletion; "
            "ASPM gene sequencing: biallelic mutations — founder mutations in consanguineous families; "
            "Developmental testing: Bayley-III / WPPSI; "
            "EEG: only if seizures; "
            "Metabolic screen: exclude secondary microcephaly (TORCH, metabolic)"
        ),
        "pathognomonic": (
            "SEVERE MICROCEPHALY (OFC ≤ -3 SD) + NORMAL CORTICAL STRUCTURE (no heterotopia, no lissencephaly) + AR INHERITANCE = Primary Microcephaly — ASPM first gene to test (MCPH5); "
            "SIMPLIFIED GYRAL PATTERN without frank pachygyria = ASPM/CDK5RAP2 (primary microcephaly) not tubulinopathy; "
            "DISTINGUISH lissencephaly: primary microcephaly = simplified but NOT smooth brain — cortex thickness NORMAL; "
            "CONSANGUINEOUS FAMILY + MICROCEPHALY = AR primary microcephaly panel (ASPM, CENPJ, CDK5RAP2, WDR62, CEP63); "
            "ASPM vs CDK5RAP2: ASPM more severe OFC; CDK5RAP2 milder; "
            "SECONDARY MICROCEPHALY: TORCH, fetal alcohol, metabolic — exclude before labelling primary"
        ),
        "treatment": (
            "No approved targeted therapy; "
            "Early intervention: speech therapy, OT, physiotherapy from diagnosis; "
            "Educational support: special educational needs — most attend supported schooling; "
            "Anti-seizure medications: if epilepsy — valproate, levetiracetam (usually drug-responsive); "
            "Behavioural support: autism spectrum features; structured environment; "
            "Ophthalmology: strabismus screening; "
            "Nutritional support: if feeding difficulties; "
            "Genetic counselling: AR — 25% recurrence; PGT available; "
            "Prenatal: fetal OFC monitoring from 20 wks in at-risk pregnancies; "
            "Research: restoration of neuronal progenitor proliferation — investigational"
        ),
        "critical_flags": [
            "ASPM-MCPH5-MOST-COMMON-AR-PRIMARY-MICROCEPHALY",
            "ASPM-OFC-MINUS-7-TO-10-SD-BIRTH",
            "ASPM-NORMAL-CORTEX-LAYERING-NO-HETEROTOPIA",
            "ASPM-SIMPLIFIED-GYRAL-NOT-LISSENCEPHALY",
            "ASPM-CONSANGUINITY-FOUNDER-MUTATIONS",
            "ASPM-MILD-ID-FUNCTIONS-BETTER-THAN-IMAGING",
            "ASPM-AR-25pct-RECURRENCE",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- CDK5RAP2 — Primary Microcephaly MCPH3 (AR) --------------------------------------
    {
        "gene": "CDK5RAP2",
        "alt_name": "CDK5RAP2 (CDK5RAP2-1893aa-9q33.2 / AR — Primary-Microcephaly-MCPH3 — Milder-Than-ASPM-OFC-Minus-3-to-6-SD — Centrosomal-Pericentriolar-Material — Spindle-Checkpoint)",
        "protein": (
            "CDK5RAP2 -- 9q33.2 AR -- CDK5RAP2-1893aa -- "
            "Primary-Microcephaly-MCPH3-Milder-Than-ASPM-MCPH5 -- "
            "Centrosomal-Pericentriolar-Material-Protein-Mitotic-Spindle-Orientation -- "
            "OFC-Minus-3-to-6-SD-Mild-Moderate-ID -- "
            "Normal-Cortical-Lamination-Simplified-Gyral-Pattern"
        ),
        "locus": "9q33.2",
        "protein_size": "1893 aa",
        "inheritance": (
            "AR (autosomal recessive) — CDK5RAP2 biallelic LOF; "
            "CDK5RAP2 encodes CDK5 regulatory subunit-associated protein 2 — centrosomal pericentriolar material protein; "
            "Rarer than ASPM but well-established MCPH gene; "
            "Consanguineous pedigrees most common; "
            "Founder mutations in Pakistani, Arab, and South Asian families"
        ),
        "age_of_onset": (
            "Prenatal: OFC reduced from 2nd trimester; "
            "Birth: OFC -3 to -6 SD (milder than ASPM); "
            "Development: mild-moderate ID; most achieve independent ambulation; "
            "Speech: delayed but develops — better prognosis than ASPM; "
            "Behaviour: autistic traits in some; hyperactivity; "
            "Epilepsy: less common than ASPM (<10%); "
            "Brain MRI: simplified gyral pattern; thin corpus callosum possible; "
            "Prognosis: generally better functional outcome than ASPM"
        ),
        "key_biomarker": (
            "Head circumference: OFC -3 to -6 SD (milder than ASPM); "
            "Brain MRI: simplified gyral pattern; reduced sulcation; normal lamination; "
            "Thin or short corpus callosum in some; "
            "CDK5RAP2 sequencing: biallelic truncating/missense mutations; "
            "Chromosomal microarray: exclude 9q33.2 deletion; "
            "Primary microcephaly gene panel: ASPM, CDK5RAP2, CENPJ, WDR62, CEP63, CASC5; "
            "Developmental testing: Griffiths / WISC; "
            "Metabolic: exclude secondary causes"
        ),
        "pathognomonic": (
            "PRIMARY MICROCEPHALY + MILDER DEGREE + AR INHERITANCE = CDK5RAP2 (MCPH3) vs ASPM (MCPH5 — more severe); "
            "CDK5RAP2 vs ASPM: CDK5RAP2 OFC milder (-3 to -6 SD vs -7 to -10 SD), better functional outcome; "
            "CENTROSOMAL GENE PANEL: CDK5RAP2, CENPJ, CEP63, CEP152 — all primary microcephaly, AR; "
            "DISTINGUISH TUBA1A: CDK5RAP2 has NO basal ganglia dysplasia, NO cerebellar hypoplasia, NO pachygyria; "
            "DISTINGUISH MCPH from MCD: primary microcephaly = reduced head, normal cortex structure; MCD = abnormal cortex; "
            "CONSANGUINITY is the key red flag: always run AR primary microcephaly panel in consanguineous family with microcephaly"
        ),
        "treatment": (
            "No approved targeted therapy; "
            "Early intervention: speech, OT, physiotherapy; "
            "Educational support: special educational needs — outcome better than ASPM; "
            "Behavioural support: autism and hyperactivity management; "
            "Anti-seizure medications: if epilepsy; "
            "Genetic counselling: AR 25% recurrence; PGT available; "
            "Prenatal monitoring: fetal OFC from 20 wks; "
            "Research: centrosome biology — microcephaly drug targets investigational"
        ),
        "critical_flags": [
            "CDK5RAP2-MCPH3-MILDER-THAN-ASPM",
            "CDK5RAP2-OFC-MINUS-3-TO-6-SD",
            "CDK5RAP2-CENTROSOMAL-GENE-AR",
            "CDK5RAP2-NORMAL-CORTEX-NO-LISSENCEPHALY",
            "CDK5RAP2-CONSANGUINITY-FOUNDER-MUTATIONS",
            "CDK5RAP2-BETTER-PROGNOSIS-THAN-ASPM",
            "CDK5RAP2-PANEL-ASPM-CDK5RAP2-CENPJ-WDR62",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- ADGRG1 / GPR56 — Bilateral Frontoparietal Polymicrogyria (AR) -------------------
    {
        "gene": "ADGRG1",
        "alt_name": "ADGRG1 / GPR56 (ADGRG1-GPR56-693aa-16q21 / AR — Bilateral-Frontoparietal-Polymicrogyria-BFPP — Strabismus-Hypotonia-Dysarthria-CLASSIC-TRIAD — 16q21-Deletion-Contiguous — Laminin-Receptor-Cortical-Adhesion)",
        "protein": (
            "ADGRG1/GPR56 -- 16q21 AR -- ADGRG1-693aa -- "
            "Bilateral-Frontoparietal-Polymicrogyria-BFPP-Most-Common-AR-PMG -- "
            "Strabismus-Hypotonia-Dysarthria-Classic-TRIAD-BFPP -- "
            "Adhesion-GPCR-Extracellular-Matrix-Laminin-Cortical-Adhesion-Migration -- "
            "BFPP-Distinctive-MRI-Pebbled-Cobblestone-Frontoparietal-Pattern"
        ),
        "locus": "16q21",
        "protein_size": "693 aa",
        "inheritance": (
            "AR (autosomal recessive) — ADGRG1 biallelic LOF; "
            "ADGRG1 (previously GPR56) — adhesion G-protein coupled receptor G1; "
            "Mediates neuronal-ECM (extracellular matrix/collagen III) interaction for cortical lamination; "
            "Consanguineous families most commonly; "
            "Founder mutations in populations from Turkey, Egypt, Saudi Arabia"
        ),
        "age_of_onset": (
            "Prenatal: abnormal fetal MRI (polymicrogyria, simplified pattern) from 24-28 weeks; "
            "Neonatal: hypotonia, poor visual fixation (strabismus evident early); "
            "Strabismus: EXOTROPIA most common — ophthalmology referral at diagnosis; "
            "Hypotonia: truncal + limb — delayed motor milestones; "
            "Dysarthria: severe; expressive language delayed but receptive better; "
            "Epilepsy: 60-80%; focal with secondary generalisation; "
            "Cognitive: moderate-severe ID; "
            "Brain: BFPP — bilateral frontoparietal polymicrogyria — excessive gyration with fused small gyri; "
            "Cerebellar hypoplasia: mild; white matter abnormality"
        ),
        "key_biomarker": (
            "Brain MRI: bilateral frontoparietal polymicrogyria — pebbled/cobblestone cortex frontoparietal distribution; "
            "Abnormal myelination pattern: delayed/abnormal posterior white matter; "
            "Cerebellar hypoplasia: mild vermian; "
            "Corpus callosum: thin or hypoplastic; "
            "Ophthalmology: exotropic strabismus on clinical exam; ERG/VEP; "
            "ADGRG1 sequencing: biallelic mutations; "
            "Chromosomal microarray: 16q21 deletion (contiguous gene syndrome); "
            "EEG: focal temporal/frontoparietal epileptiform; "
            "EMG/NCS: if neuropathy suspected (usually normal)"
        ),
        "pathognomonic": (
            "BILATERAL FRONTOPARIETAL POLYMICROGYRIA + STRABISMUS (exotropia) + HYPOTONIA + DYSARTHRIA = ADGRG1/GPR56 BFPP — CLASSIC TRIAD; "
            "BFPP MRI: pebbled/cobblestone cortex involving FRONTAL and PARIETAL lobes bilaterally (NOT perisylvian); "
            "DISTINGUISH Perisylvian PMG (bilateral): COL4A1, SRPX2, TMTC3, PIK3R2 — different distribution; "
            "DISTINGUISH cobblestone cortex (lissencephaly type 2): POMT1/POMT2 (Walker-Warburg) — elevated CK, muscular dystrophy; "
            "STRABISMUS in PMG: always screen for ADGRG1 — strabismus + frontoparietal PMG = pathognomonic combination; "
            "CONSANGUINITY + FRONTOPARIETAL PMG: test ADGRG1 first"
        ),
        "treatment": (
            "Anti-seizure medications: lamotrigine, levetiracetam, valproate; "
            "Refractory epilepsy: ketogenic diet, VNS — resection usually not feasible (bilateral); "
            "Strabismus: corrective surgery (strabismus surgery) + patching for amblyopia; "
            "Physiotherapy: for hypotonia, motor development; "
            "Occupational therapy: fine motor, daily living skills; "
            "Speech and language: augmentative/alternative communication (AAC) for severe dysarthria; "
            "Genetic counselling: AR 25% recurrence; "
            "Prenatal: fetal MRI + ADGRG1 testing in at-risk pregnancies; "
            "No approved targeted therapy"
        ),
        "critical_flags": [
            "ADGRG1-BFPP-STRABISMUS-HYPOTONIA-DYSARTHRIA-CLASSIC-TRIAD",
            "ADGRG1-FRONTOPARIETAL-DISTRIBUTION-NOT-PERISYLVIAN",
            "ADGRG1-COBBLESTONE-DISTINGUISH-WALKER-WARBURG-CK",
            "ADGRG1-EXOTROPIA-OPHTHALMOLOGY-MANDATORY",
            "ADGRG1-AR-CONSANGUINITY",
            "ADGRG1-BFPP-MRI-PEBBLED-CORTEX",
            "ADGRG1-RESECTION-NOT-FEASIBLE-BILATERAL",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- RELN — Lissencephaly with Cerebellar Hypoplasia (AR lethal / AD mild) -----------
    {
        "gene": "RELN",
        "alt_name": "RELN (Reelin-3461aa-7q22.1 / AR-homozygous-LCH-LETHAL-Cerebellar-Agenesis — AD-heterozygous-Autism-LQTS-Mild-Cortical — Reelin-Glycoprotein-Dab1-ApoER2-VLDLR — Cerebellar-Agenesis-PATHOGNOMONIC-AR)",
        "protein": (
            "RELN -- 7q22.1 AR/AD -- RELN-3461aa -- "
            "AR-Homozygous-Lissencephaly-Cerebellar-Hypoplasia-LCH-Most-Severe-Brain-Malformation -- "
            "Cerebellar-Agenesis-Nearly-Complete-PATHOGNOMONIC-AR-RELN -- "
            "AD-Heterozygous-Autism-LQTS-Mild-Cortical-Migration-Delay -- "
            "Reelin-Extracellular-Matrix-Glycoprotein-ApoER2-VLDLR-Receptor-Dab1-Adapter"
        ),
        "locus": "7q22.1",
        "protein_size": "3461 aa",
        "inheritance": (
            "AR homozygous/compound heterozygous: severe LCH phenotype; "
            "AD heterozygous: milder autism spectrum + QT prolongation association; "
            "AR RELN: rarest lissencephaly gene; "
            "AD RELN: identified in autism/SCZ cohort studies; penetrance incomplete; "
            "Largest gene among MCD genes (3461 aa protein); "
            "Reelin signals through ApoER2/VLDLR receptors → Dab1 phosphorylation → inside-out cortical layering"
        ),
        "age_of_onset": (
            "AR homozygous: prenatal — severe brain malformation on fetal MRI; "
            "LCH: lissencephaly (pachygyria) with nearly complete cerebellar agenesis — pathognomonic; "
            "Cerebellar agenesis: essentially absent cerebellum and vermis; "
            "Neonatal: profound hypotonia, seizures from first days; severe feeding difficulties; "
            "Development: profound ID; no independent ambulation; "
            "Prognosis AR: very severe; high mortality in infancy; "
            "AD heterozygous: autism spectrum features; QT prolongation on ECG; "
            "AD: much milder — no frank brain malformation on standard MRI; "
            "Temporal lobe: focal dysplasia reported in some AD cases"
        ),
        "key_biomarker": (
            "Brain MRI (AR): lissencephaly/pachygyria + nearly complete CEREBELLAR AGENESIS — unique combination; "
            "Brainstem: hypoplastic pons; "
            "Corpus callosum: absent/hypoplastic; "
            "CEREBELLAR AGENESIS on MRI distinguishes RELN from PAFAH1B1/DCX/TUBA1A; "
            "RELN sequencing: biallelic mutations (AR) or heterozygous (AD association); "
            "Chromosomal microarray: 7q22.1 deletion; "
            "ECG: QTc measurement (AD cases — LQTS association); "
            "EEG: multifocal, hypsarrhythmia (AR); "
            "Reelin protein assay (CSF): very low in AR cases; "
            "ApoER2/VLDLR receptor pathway: research context"
        ),
        "pathognomonic": (
            "LISSENCEPHALY + NEARLY COMPLETE CEREBELLAR AGENESIS = AR RELN — the most distinctive MRI in lissencephaly spectrum; "
            "NO OTHER LISSENCEPHALY GENE (PAFAH1B1, DCX, TUBA1A) causes near-complete cerebellar agenesis this severe; "
            "DISTINGUISH TUBA1A: cerebellar hypoplasia present but NOT agenesis; TUBA1A has basal ganglia dysplasia; "
            "DISTINGUISH Walker-Warburg (POMT1/2): also severe cerebellar changes but cobblestone cortex + muscular dystrophy; "
            "AD RELN heterozygous: QT PROLONGATION on ECG in autism families — screen ECG in RELN-positive individuals; "
            "REELIN as biomarker: very low reelin in CSF of AR patients — research utility"
        ),
        "treatment": (
            "AR severe LCH: supportive/palliative care in most; "
            "Anti-seizure medications: phenobarbitone, levetiracetam — refractory common; "
            "Feeding: gastrostomy early given profound dysphagia; "
            "Respiratory: secretion management; tracheostomy in some; "
            "Palliative care: early goals-of-care discussion; "
            "AD heterozygous: QT monitoring — AVOID QT-prolonging drugs; beta-blockers if QTc >500 ms; "
            "Autism treatment: ABA, EIBI, AAC; "
            "Genetic counselling: AR — 25% recurrence; AD — 50% risk of mild phenotype; "
            "Prenatal: fetal MRI + RELN sequencing in at-risk AR pregnancies; "
            "Research: reelin augmentation — preclinical investigational"
        ),
        "critical_flags": [
            "RELN-AR-CEREBELLAR-AGENESIS-PATHOGNOMONIC-LISSENCEPHALY",
            "RELN-DISTINGUISH-NO-OTHER-LIS-GENE-HAS-CEREBELLAR-AGENESIS",
            "RELN-AD-QT-PROLONGATION-AVOID-QT-DRUGS",
            "RELN-AR-PROFOUND-SEVERE-PROGNOSIS-PALLIATIVE",
            "RELN-AD-AUTISM-LQTS-ASSOCIATION",
            "RELN-LARGEST-MCD-GENE-3461aa",
            "RELN-REELIN-CSF-VERY-LOW-AR",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _generate_cohort():
    """Generate 320-patient cohort (8 genes × 40, seeds 2046-2053)."""
    all_patients = []
    for gene_data in BM_GENES:
        rng = random.Random(gene_data["seed"])
        gene = gene_data["gene"]
        for i in range(40):
            age = rng.randint(0, 18)
            sex = rng.choice(["M", "F"])
            # Gene-specific phenotype simulation
            if gene == "PAFAH1B1":
                seizure_onset = rng.choice([True, True, True, False])
                iq_severe = rng.choice([True, True, False])
                mds = rng.choice([True, False, False])
                all_patients.append({
                    "gene": gene, "age": age, "sex": sex,
                    "seizure_onset_3mo": seizure_onset, "severe_id": iq_severe,
                    "miller_dieker": mds, "lissencephaly_grade": rng.choice([1, 1, 2, 2]),
                    "malformation_type": "agyria/pachygyria",
                })
            elif gene == "DCX":
                male = sex == "M"
                bh = not male and rng.choice([True, True, False])
                all_patients.append({
                    "gene": gene, "age": age, "sex": sex,
                    "lissencephaly_male": male,
                    "band_heterotopia_female": bh,
                    "normal_iq": not male and bh,
                    "malformation_type": "lissencephaly" if male else "band heterotopia",
                })
            elif gene == "TUBA1A":
                cb_hypo = rng.choice([True, True, False])
                cc_agenesis = rng.choice([True, False])
                all_patients.append({
                    "gene": gene, "age": age, "sex": sex,
                    "cerebellar_hypoplasia": cb_hypo,
                    "cc_dysgenesis": cc_agenesis,
                    "basal_ganglia_dysplasia": rng.choice([True, True, False]),
                    "malformation_type": "pachygyria+cerebellar-hypoplasia",
                })
            elif gene == "FLNA":
                female = sex == "F"
                all_patients.append({
                    "gene": gene, "age": age, "sex": sex,
                    "bpnh": female,
                    "normal_iq": female and rng.choice([True, True, False]),
                    "cardiac_feature": rng.choice([True, False, False]),
                    "malformation_type": "bilateral-PVNH" if female else "lethal-male",
                })
            elif gene == "ASPM":
                all_patients.append({
                    "gene": gene, "age": age, "sex": sex,
                    "ofc_sd": rng.uniform(-10, -7),
                    "simplified_gyral_pattern": True,
                    "epilepsy": rng.choice([True, False, False, False]),
                    "malformation_type": "primary-microcephaly",
                })
            elif gene == "CDK5RAP2":
                all_patients.append({
                    "gene": gene, "age": age, "sex": sex,
                    "ofc_sd": rng.uniform(-6, -3),
                    "simplified_gyral_pattern": True,
                    "epilepsy": rng.choice([True, False, False, False, False]),
                    "malformation_type": "primary-microcephaly",
                })
            elif gene == "ADGRG1":
                all_patients.append({
                    "gene": gene, "age": age, "sex": sex,
                    "strabismus": rng.choice([True, True, False]),
                    "hypotonia": True,
                    "dysarthria": rng.choice([True, True, False]),
                    "epilepsy": rng.choice([True, True, False, False]),
                    "malformation_type": "bilateral-frontoparietal-PMG",
                })
            else:  # RELN
                ar = rng.choice([True, False, False])
                all_patients.append({
                    "gene": gene, "age": age, "sex": sex,
                    "ar_lch": ar,
                    "cerebellar_agenesis": ar,
                    "qt_prolongation": not ar and rng.choice([True, False]),
                    "autism_ad": not ar,
                    "malformation_type": "lissencephaly+cerebellar-agenesis" if ar else "mild-cortical-AD",
                })
    return all_patients


def overview():
    """Return atlas-level aggregate overview."""
    patients = _generate_cohort()
    lissencephaly = sum(1 for p in patients
                        if "lissencephaly" in p.get("malformation_type", ""))
    pmg = sum(1 for p in patients
              if "PMG" in p.get("malformation_type", ""))
    microcephaly = sum(1 for p in patients
                       if "microcephaly" in p.get("malformation_type", ""))
    pvnh = sum(1 for p in patients
               if "PVNH" in p.get("malformation_type", ""))
    cerebellar = sum(1 for p in patients
                     if "cerebellar" in p.get("malformation_type", ""))
    seizure_patients = sum(1 for p in patients
                           if p.get("epilepsy") or p.get("seizure_onset_3mo"))
    cardiac_patients = sum(1 for p in patients if p.get("cardiac_feature"))
    strabismus_patients = sum(1 for p in patients if p.get("strabismus"))

    return {
        "atlas": "Hereditary-Brain-Malformation-Atlas",
        "genes": [g["gene"] for g in BM_GENES],
        "total_patients": len(patients),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "lissencephaly_patients": lissencephaly,
        "polymicrogyria_patients": pmg,
        "primary_microcephaly_patients": microcephaly,
        "periventricular_heterotopia_patients": pvnh,
        "cerebellar_patients": cerebellar,
        "seizure_patients": seizure_patients,
        "cardiac_feature_patients": cardiac_patients,
        "strabismus_patients": strabismus_patients,
    }


def breakdown():
    """Return per-gene breakdown with clinical details."""
    patients = _generate_cohort()
    result = {}
    for gene_data in BM_GENES:
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
    """Return gene definitions, glossary and surveillance protocols."""
    return {
        "genes": {g["gene"]: g["protein"] for g in BM_GENES},
        "glossary": {
            "Lissencephaly": "Smooth brain — absent or markedly reduced gyration; Types 1 (LIS1/DCX) and 2 (cobblestone); brain appears smooth due to failed neuronal migration",
            "Pachygyria": "Broad, shallow gyri with fewer sulci than normal; intermediate between normal and agyria; cortex thickened (4-10 mm)",
            "Band Heterotopia (SBH)": "Subcortical band of grey matter separated from cortex by white matter — 'double cortex sign'; DCX females; epilepsy with preserved IQ",
            "Periventricular Nodular Heterotopia (PVNH)": "Nodules of grey matter lining lateral ventricle walls — failed migration; FLNA most common; epilepsy + normal IQ in females",
            "Polymicrogyria (PMG)": "Excessive small gyri fused to each other; multiple aetiologies; bilateral frontoparietal (ADGRG1) vs perisylvian patterns",
            "Primary Microcephaly": "OFC ≤ -3 SD at birth; normal cortical lamination but reduced brain size; ASPM and CDK5RAP2 most common AR genes",
            "Cerebellar Hypoplasia": "Underdeveloped cerebellum ± vermis; feature of TUBA1A and RELN; distinguish from ASPM/CDK5RAP2 (no cerebellar change)",
            "Malformation of Cortical Development (MCD)": "Umbrella term: lissencephaly, heterotopia, PMG, microcephaly — all failures of neuronal proliferation, migration, or organisation",
            "Tubulinopathy": "Spectrum of MCD caused by mutations in tubulin genes (TUBA1A, TUBB2B, TUBB3, TUBB); basal ganglia dysplasia pathognomonic",
            "Agyria": "Complete absence of gyri (smoothest lissencephaly); cortex 10-20 mm thick; Grade 1; PAFAH1B1/LIS1",
            "Cobblestone Cortex": "Type 2 lissencephaly — irregular bumpy surface due to overmigration; different from Type 1; caused by POMT1/POMT2/FKRP (muscular dystrophy genes)",
            "Reelin": "Extracellular glycoprotein secreted by Cajal-Retzius cells; signals through ApoER2/VLDLR → Dab1; controls inside-out cortical layering",
            "Inside-Out Cortical Layering": "Normal cortex: later-born neurons migrate past earlier-born to form outer layers; disrupted in lissencephaly — outside-in instead",
            "Miller-Dieker Syndrome (MDS)": "Contiguous 17p13.3 deletion including PAFAH1B1 and YWHAE; lissencephaly + facial dysmorphism + severe prognosis",
        },
        "surveillance_protocols": {
            "PAFAH1B1": "Brain MRI at diagnosis (thickness + gradient); EEG at seizure onset; CMA + PAFAH1B1 sequencing; chest X-ray (aspiration); dietitian (gastrostomy planning); neurology 3-monthly; palliative care referral (MDS)",
            "DCX": "Brain MRI (gradient direction — anterior vs posterior); maternal brain MRI mandatory; DCX sequencing (blood + saliva if male proband); EEG; multi-disciplinary: neurology, physiotherapy, SLT, dietitian",
            "TUBA1A": "Brain MRI (basal ganglia + CC + cerebellum); tubulinopathy gene panel; EEG; ophthalmology (strabismus); physiotherapy; feeding assessment; annual neurodevelopmental review",
            "FLNA": "Brain MRI (PVNH protocol — coronal IR); FLNA sequencing; ECHO (MVP, aortic root — annual); MRA aorta (baseline + 2-yearly); EEG; ophthalmology; epilepsy surgery assessment (LITT available)",
            "ASPM": "Head circumference monthly (first year); brain MRI at diagnosis; ASPM sequencing; developmental testing 6-monthly (Bayley/Griffiths); EEG if seizures; educational psychology at school age",
            "CDK5RAP2": "Head circumference monthly; brain MRI; CDK5RAP2 sequencing (primary microcephaly panel); developmental testing; EEG if seizures; school-age educational support",
            "ADGRG1": "Brain MRI (frontoparietal PMG pattern); ADGRG1 sequencing; ophthalmology + strabismus surgery assessment; EEG; physiotherapy (hypotonia); SLT (dysarthria + AAC); anti-seizure medication review 6-monthly",
            "RELN": "Brain MRI (cerebellar + cortical assessment); RELN sequencing; ECG (QTc — AD cases); EEG (AR cases); pulmonary (secretion management AR); palliative care (AR severe); autism assessment (AD)",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Lissencephaly: {ov['lissencephaly_patients']}")
    print(f"Polymicrogyria: {ov['polymicrogyria_patients']}")
    print(f"Primary Microcephaly: {ov['primary_microcephaly_patients']}")
    print(f"Periventricular Heterotopia: {ov['periventricular_heterotopia_patients']}")
    print(f"Seizure patients: {ov['seizure_patients']}")
    print(f"Cardiac feature: {ov['cardiac_feature_patients']}")
