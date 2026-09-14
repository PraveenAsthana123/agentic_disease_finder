"""Hereditary NBIA Atlas — 8-Gene Reference
PANK2-PLA2G6-WDR45-C19orf12-FA2H-ATP13A2-COASY-DCAF17
320 patients (8 x 40), seeds 2606-2613.
Endpoints: /api/hereditary-nbia-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "PANK2",
        "protein": (
            "PANK2 -- 2p13.3 AR -- 570aa -- Pantothenate-Kinase-2-63kDa-"
            "Mitochondrial-CoA-Biosynthesis-Enzyme-PKAN-NBIA1-AR -- OMIM-Gene-606157-Disease-PKAN-234200"
        ),
        "locus": "2p13.3",
        "protein_size": "570 aa / 63 kDa",
        "inheritance": (
            "AR (biallelic loss of function); "
            "PKAN (Pantothenate Kinase-Associated Neurodegeneration); NBIA1; "
            "Most common NBIA — ~50% of all NBIA cases; "
            "Onset: childhood classic (3-6yr) or atypical adult (>18yr); "
            "Biallelic PANK2 loss of function — truncating variants more severe; "
            "de novo mutations rare; consanguinity enriched in classic PKAN; "
            "Carrier frequency ~1:90 in general population"
        ),
        "disease_category": (
            "Pantothenate Kinase-Associated Neurodegeneration (PKAN); "
            "Progressive extra-pyramidal disorder: dystonia + rigidity + dysarthria; "
            "Globus pallidus (GP) iron deposition with central T2-hyperintensity = EYE-OF-TIGER PATHOGNOMONIC; "
            "No approved disease-modifying therapy; Deferiprone (DFP) used off-label; "
            "PANK2 encodes the rate-limiting enzyme in mitochondrial CoA biosynthesis; "
            "CoA deficiency → ↓ lipid metabolism + ↑ cysteine accumulation → iron chelation + neurodegeneration"
        ),
        "disease_pathway": (
            "PANK2 encodes pantothenate kinase 2, the rate-limiting mitochondrial enzyme converting "
            "pantothenate (vitamin B5) to 4'-phosphopantothenate in coenzyme A (CoA) biosynthesis. "
            "Loss of PANK2 → CoA deficiency in mitochondria → impaired fatty acid β-oxidation + TCA cycle. "
            "Cysteine (a CoA intermediate substrate) accumulates → chelates ferrous iron → "
            "iron-cysteine complexes deposit in globus pallidus (GP) inner segment → "
            "T2*/SWI hypointensity on MRI (ring of iron). Central T2-hyperintensity in GP = "
            "'Eye-of-Tiger' sign: central focus of oedema/gliosis surrounded by iron ring. "
            "Progressive nigral and GP degeneration → extra-pyramidal syndrome dominating the phenotype."
        ),
        "pathognomonic": (
            "EYE-OF-TIGER SIGN on T2-weighted/SWI MRI PATHOGNOMONIC FOR PKAN: "
            "Bilateral GP hypointensity (iron, T2-dark ring) with CENTRAL T2-HYPERINTENSITY (gliosis/oedema) — "
            "seen in >90% of PKAN; not specific to PKAN if eye-of-tiger absent (variants without central hyperintensity exist); "
            "GP iron WITHOUT central hyperintensity seen in other NBIA subtypes (PLA2G6, C19orf12); "
            "SWI more sensitive than T2 for iron deposition; "
            "Retinal degeneration on fundoscopy (50%); "
            "OPTIC DISC PALLOR (not lenticonus — key DDx from Alport); "
            "Classic PKAN: severe dystonia + early course vs atypical PKAN: dysarthria + pyramidal + slower"
        ),
        "treatment": (
            "NO FDA/EMA-APPROVED DISEASE-MODIFYING THERAPY; "
            "DEFERIPRONE (DFP) — oral iron chelator; B-PKAN trial (2022): slowed iron accumulation on MRI "
            "but did NOT improve neurological outcomes; off-label use continues; "
            "Monitoring: neutrophil count (agranulocytosis risk with DFP); "
            "PANTETHINE (pantothenate salvage pathway): small trials, limited evidence; "
            "BOTULINUM TOXIN — focal dystonia palliation; "
            "BACLOFEN/TRIHEXYPHENIDYL — spasticity and dystonia; "
            "DEEP BRAIN STIMULATION (DBS) — GPi DBS palliates dystonia in selected patients; "
            "RILUZOLE — glutamate antagonist; investigational neuroprotection; "
            "MULTIDISCIPLINARY: physiotherapy, speech, swallowing, nutrition; "
            "GENETIC COUNSELLING: AR inheritance; 25% recurrence; prenatal testing available"
        ),
        "key_features": [
            "PANK2: most common NBIA (~50%); AR; childhood onset 3-6yr classic / adult >18yr atypical",
            "Eye-of-Tiger sign on T2/SWI MRI = PATHOGNOMONIC for PKAN (central T2-bright in GP T2-dark ring)",
            "Dystonia dominant (focal oromandibular → generalised); pigmentary retinopathy in 50%",
            "Deferiprone (DFP) slows iron accumulation but no confirmed neurological benefit (B-PKAN 2022)",
            "GPi deep brain stimulation (DBS) palliates dystonia — does not halt progression",
            "CoA biosynthesis enzyme — pantothenate (B5) supplementation: limited evidence",
            "Atypical PKAN (>18yr onset): dysarthria + pyramidal signs + slower progression; eye-of-tiger present",
            "Retinal degeneration 50% — annual fundoscopy mandatory; electroretinogram (ERG) baseline",
        ],
        "key_ddx": [
            "PLA2G6 PLAN: GP iron WITHOUT eye-of-tiger; optic atrophy early; axonal spheroids EM",
            "WDR45 BPAN: females; biphasic (childhood seizures → adult Parkinsonism); T1 halo sign not eye-of-tiger",
            "C19orf12 MPAN: GP+SN iron confluent; optic atrophy + motor neuropathy; slower progression",
            "Wilson disease (ATP7B): liver + neurological; Kayser-Fleischer rings; low ceruloplasmin; treatable",
        ],
        "onset_age": 5.0,
        "gp_iron_pct": 98,
        "eye_of_tiger_pct": 92,
        "dystonia_pct": 96,
        "retinopathy_pct": 50,
        "dbs_response_pct": 58,
        "seed": 2606,
    },
    {
        "gene": "PLA2G6",
        "protein": (
            "PLA2G6 -- 22q13.1 AR -- 806aa -- Phospholipase-A2-Group-VI-iPLA2-VIA-88kDa-"
            "Membrane-Phospholipid-Remodelling-PLAN-NBIA2-PARK14-AR -- OMIM-Gene-603604-Disease-PLAN-256600"
        ),
        "locus": "22q13.1",
        "protein_size": "806 aa / 88 kDa",
        "inheritance": (
            "AR (biallelic loss of function); "
            "PLA2G6-Associated Neurodegeneration (PLAN); NBIA2; PARK14; "
            "THREE phenotypic subtypes — same gene, different severity/age: "
            "(1) Infantile Neuroaxonal Dystrophy (INAD): most severe, onset 6m-3yr; "
            "(2) Atypical NAD/NBIA2b: later onset, slower; "
            "(3) PARK14 adult-onset Parkinson: levodopa-responsive Parkinsonism; "
            "iPLA2-VIA = calcium-independent phospholipase A2 group VI"
        ),
        "disease_category": (
            "PLA2G6-Associated Neurodegeneration (PLAN) — three phenotypes from same gene: "
            "INAD (most common): regression + hypotonia + cerebellar ataxia + optic atrophy onset 6m-3yr; "
            "Axonal spheroids on electron microscopy = PATHOGNOMONIC for PLAN (neuroaxonal dystrophy); "
            "GP iron variable (less prominent than PANK2); Cerebellar + optic nerve atrophy on MRI; "
            "PARK14: adult levodopa-responsive Parkinsonism + cognitive impairment; "
            "iPLA2-VIA critical for phospholipid membrane remodelling in axons"
        ),
        "disease_pathway": (
            "PLA2G6 encodes iPLA2-VIA (cytosolic calcium-independent phospholipase A2), "
            "an enzyme that cleaves the sn-2 acyl chain from membrane glycerophospholipids "
            "generating lysophospholipid + free fatty acid (typically arachidonic acid). "
            "iPLA2-VIA is essential for phospholipid remodelling ('Lands cycle') in axonal membranes. "
            "Loss of iPLA2-VIA → impaired membrane phospholipid turnover → "
            "abnormal accumulation of peroxidised phospholipids → axonal degeneration. "
            "Hallmark pathology: NEUROAXONAL SPHEROIDS — focal swellings of distal axons filled "
            "with accumulating mitochondria, tubular material, and lipids. "
            "Iron deposition (GP/SN) is secondary to neurodegeneration; less prominent than in PANK2."
        ),
        "pathognomonic": (
            "NEUROAXONAL SPHEROIDS on electron microscopy (EM) PATHOGNOMONIC for PLAN/INAD: "
            "Eosinophilic swellings in distal axons + synaptic terminals — skin or conjunctival biopsy accessible; "
            "OPTIC ATROPHY early feature — differs from PANK2 (pigmentary retinopathy); "
            "CEREBELLAR ATROPHY on MRI — early progressive; "
            "GP IRON variable — T2*/SWI hypointensity, but NO eye-of-tiger (no central hyperintensity); "
            "T2 cerebellar cortex hyperintensity (Purkinje cell/granule cell degeneration); "
            "PARK14: adult-onset; brain MRI normal or mild atrophy; levodopa-responsive Parkinsonism; "
            "ERG: rod-cone dystrophy early in INAD (like PANK2 retinopathy)"
        ),
        "treatment": (
            "NO DISEASE-MODIFYING THERAPY; supportive management: "
            "INAD: early supportive care; feeding tube for dysphagia; antiepileptic drugs (AEDs) for seizures; "
            "VITAMINS E + C: antioxidant neuroprotection — small open-label series; no RCT evidence; "
            "PARK14: LEVODOPA — responsive initially (carbidopa-levodopa standard doses); "
            "dopamine agonists as adjunct; response wanes as Parkinsonism advances; "
            "IRON CHELATION (deferiprone): theoretical for iron component; limited clinical evidence; "
            "SINEMET/LEVODOPA: first-line for PARK14 Parkinsonism; "
            "PHYSIOTHERAPY + OCCUPATIONAL THERAPY: functional maintenance; "
            "GENETIC COUNSELLING: AR; 25% recurrence risk; PGD available"
        ),
        "key_features": [
            "PLA2G6: AR; three phenotypes INAD (severe neonatal) / Atypical NAD / PARK14 (adult Parkinson's)",
            "INAD: hypotonia + regression + optic atrophy + cerebellar ataxia onset 6m-3yr; rapid deterioration",
            "Neuroaxonal spheroids on EM PATHOGNOMONIC — accessible via skin/conjunctival biopsy",
            "Optic ATROPHY (nerve degeneration) not pigmentary retinopathy — DDx from PANK2",
            "GP iron: variable, less prominent; NO eye-of-tiger sign (unlike PANK2)",
            "PARK14: levodopa-responsive juvenile/adult Parkinsonism; cognitive impairment prominent",
            "Cerebellar atrophy + T2 hyperintensity early on MRI — progresses throughout disease",
            "iPLA2-VIA membrane phospholipid remodelling (Lands cycle) — axonal membrane integrity",
        ],
        "key_ddx": [
            "PANK2 PKAN: eye-of-tiger GP iron; pigmentary retinopathy not optic atrophy; CoA pathway",
            "WDR45 BPAN: females; biphasic; T1 halo; no spheroids",
            "Infantile NCL (CLN1/CLN2): similar regression + optic atrophy but EM shows granular osmiophilic deposits not spheroids",
            "Idiopathic Parkinson (PARK14 DDx): alpha-synuclein/LRRK2 negative; biallelic PLA2G6; young onset",
        ],
        "onset_age": 1.5,
        "gp_iron_pct": 65,
        "eye_of_tiger_pct": 2,
        "dystonia_pct": 72,
        "retinopathy_pct": 78,
        "optic_atrophy_pct": 85,
        "cerebellar_atrophy_pct": 90,
        "seed": 2607,
    },
    {
        "gene": "WDR45",
        "protein": (
            "WDR45 -- Xp11.23 XLD-de-novo -- 361aa -- WD-Repeat-Domain-45-WIPI4-40kDa-"
            "Autophagy-PI3P-Binding-PROPELLER-BPAN-NBIA5-XLD -- OMIM-Gene-300526-Disease-BPAN-300894"
        ),
        "locus": "Xp11.23",
        "protein_size": "361 aa / 40 kDa",
        "inheritance": (
            "X-linked dominant (XLD), >95% de novo; "
            "BPAN (Beta-Propeller Protein-Associated Neurodegeneration); NBIA5; "
            "Females predominantly affected (males often non-viable/severely affected); "
            "Males: embryonic or early lethal in severe cases; rare surviving males with severe encephalopathy; "
            "WDR45 is on Xp11.23 — hemizygous males largely non-viable; "
            "SOMATIC MOSAICISM documented in mild female cases; "
            "de novo mutations cause essentially all cases — not inherited from parents"
        ),
        "disease_category": (
            "Beta-Propeller Protein-Associated Neurodegeneration (BPAN); "
            "PATHOGNOMONIC BIPHASIC COURSE: "
            "Phase 1 (childhood): seizures (100%) + global intellectual disability + autistic features; "
            "Phase 2 (late adolescence/adulthood): ABRUPT TRANSITION to Parkinson-like syndrome + rapidly progressive dementia; "
            "UNIQUE: static childhood encephalopathy → dementia transition distinguishes BPAN from all other NBIA; "
            "WDR45/WIPI4 is an autophagy regulator (PI3P-binding β-propeller); "
            "Iron accumulation: GP, SN, cerebral white matter — T1 HALO SIGN"
        ),
        "disease_pathway": (
            "WDR45 encodes WIPI4 (WD Repeat Domain Phosphoinositide-Interacting protein 4), "
            "a β-propeller protein that binds PI3P (phosphatidylinositol-3-phosphate) at the "
            "isolation membrane/phagophore during early autophagy. "
            "WIPI4 is ortholog of yeast Atg18 — essential for autophagosome membrane extension. "
            "Loss of WIPI4 → stalled autophagosome formation → failure of autophagic cargo clearance → "
            "accumulation of iron-containing proteins (ferritin complexes, mitochondria) + "
            "ubiquitinated protein aggregates in neurons. "
            "Iron deposits in GP, SN, and cerebral white matter on T2*/SWI MRI. "
            "Unique biphasic: childhood phase may represent compensated autophagy dysfunction; "
            "adult transition may represent threshold where remaining neuronal population cannot compensate."
        ),
        "pathognomonic": (
            "BIPHASIC COURSE = PATHOGNOMONIC CLINICAL SIGNATURE OF BPAN: "
            "childhood (seizures + ID + autism) → adulthood (Parkinsonism + dementia) transition; "
            "T1 HALO SIGN on MRI: hyperintense ring around GP/SN on T1-weighted images "
            "(iron-containing neuromelanin accumulation creates T1-shortening halo); "
            "T2*/SWI: GP + SN + cerebral white matter iron hypointensity; "
            "FEMALES EXCLUSIVELY (or nearly): any NBIA in female with biphasic = WDR45 first; "
            "EEG: West syndrome pattern in infancy → multiple seizure types; "
            "ABRUPT DEMENTIA IN 20s-30s: rapid cognitive decline after years of static encephalopathy"
        ),
        "treatment": (
            "NO DISEASE-MODIFYING THERAPY; "
            "SEIZURE MANAGEMENT: multi-drug refractory; ketogenic diet beneficial in selected cases; "
            "Valproate: effective for some seizure types but monitor LFTs + ammonia (mitochondrial concern); "
            "Levetiracetam, clobazam, topiramate as adjuncts; "
            "AUTOPHAGY ENHANCEMENT (experimental): mTOR inhibitors (rapamycin/everolimus) — theoretical; "
            "small human cases of everolimus use; no RCT; "
            "PARKINSON PHASE: levodopa (carbidopa-levodopa) — variable response; often poor; "
            "dopamine agonists adjunct; "
            "IRON CHELATION: deferiprone off-label; no RCT evidence in BPAN specifically; "
            "PALLIATIVE CARE early planning given biphasic deterioration; "
            "COMMUNICATION AIDS for adults in late phase (AAC devices); "
            "MULTIDISCIPLINARY: early seizure + developmental support then neurological decline management"
        ),
        "key_features": [
            "WDR45: X-linked dominant, >95% de novo; FEMALES exclusively/predominantly affected",
            "BIPHASIC PATHOGNOMONIC: childhood seizures+ID → adulthood Parkinsonism+dementia (abrupt transition)",
            "T1 HALO SIGN on MRI: hyperintense ring around GP/SN on T1-weighted images",
            "WDR45 de novo → genetic testing essential; recurrence risk <1% for parents (de novo)",
            "Seizures 100%: infantile spasms → multiple types; often drug-refractory",
            "Dementia transition: 2nd-3rd decade; rapid progressive; Parkinsonism + cognitive decline",
            "WIPI4/ATG18 autophagy PI3P-binding β-propeller — autophagosome formation defect",
            "Males: largely non-viable; hemizygous WDR45 → severe embryonic phenotype",
        ],
        "key_ddx": [
            "Rett syndrome (MECP2): females; regression; hand stereotypies; NO iron on MRI; no Parkinsonism",
            "CDKL5 deficiency (CDD): X-linked; early-onset seizures+ID; NO Parkinsonism transition; NO iron accumulation",
            "PANK2 PKAN: eye-of-tiger not T1-halo; both sexes; no clear biphasic",
            "Idiopathic Parkinson (young-onset): WDR45 de novo panel; no prior childhood encephalopathy",
        ],
        "onset_age": 2.0,
        "gp_iron_pct": 95,
        "eye_of_tiger_pct": 0,
        "t1_halo_pct": 88,
        "dystonia_pct": 60,
        "seizure_pct": 100,
        "female_pct": 92,
        "seed": 2608,
    },
    {
        "gene": "C19orf12",
        "protein": (
            "C19orf12 -- 19q12 AR -- 141aa -- Mitochondrial-Membrane-Protein-16kDa-"
            "Coiled-Coil-Mitochondrial-Lipid-MPAN-NBIA4-AR -- OMIM-Gene-614297-Disease-MPAN-614298"
        ),
        "locus": "19q12",
        "protein_size": "141 aa / 16 kDa",
        "inheritance": (
            "AR (biallelic loss of function); "
            "MPAN (Mitochondria-Membrane Protein-Associated Neurodegeneration); NBIA4; "
            "Second most common NBIA after PANK2 (~20-35% of non-PKAN NBIA); "
            "Polish founder variant: c.204_214del11 (p.Gly69ArgfsX10) — enriched in Central/Eastern Europe; "
            "C19orf12 localised to mitochondrial membrane; "
            "Function incompletely characterised — lipid metabolism + CoA homeostasis pathway overlap"
        ),
        "disease_category": (
            "Mitochondria-Membrane Protein-Associated Neurodegeneration (MPAN); "
            "Clinical triad: optic atrophy + motor neuropathy + neuropsychiatric features; "
            "Dystonia + spasticity + Parkinsonism in combination; "
            "SLOWLY PROGRESSIVE: later onset (~10yr) + slower course than PANK2; "
            "Psychiatric features: depression, obsessive-compulsive behavior, impulse dyscontrol; "
            "GP + SN iron on T2*/SWI (confluent, less specific than eye-of-tiger); "
            "Axonal neuropathy on nerve conduction studies; "
            "C19orf12 interacts with COASY (CoPAN) — overlapping CoA pathway"
        ),
        "disease_pathway": (
            "C19orf12 encodes a small mitochondrial inner membrane protein (141aa). "
            "The precise function is debated but evidence suggests roles in: "
            "(1) CoA homeostasis — C19orf12 interacts with COASY (CoA synthase) and PANK2, "
            "suggesting a shared CoA biosynthesis pathway; "
            "(2) Lipid metabolism — C19orf12 has a coiled-coil lipid transfer domain; "
            "(3) Mitochondrial autophagy (mitophagy) — evidence for role in clearing damaged mitochondria. "
            "Loss → mitochondrial dysfunction + impaired CoA availability → "
            "secondary iron accumulation in GP/SN (less pronounced than PANK2). "
            "Motor axon vulnerability: mitochondrial dysfunction → dying-back axonal neuropathy. "
            "Optic nerve: axonal loss → optic atrophy (different from PLA2G6 EM spheroids)."
        ),
        "pathognomonic": (
            "CLINICAL TRIAD: optic atrophy + motor/sensorimotor axonal neuropathy + neuropsychiatric features; "
            "GP + SN CONFLUENT IRON on T2*/SWI MRI — bilateral symmetrical; no eye-of-tiger; "
            "SLOWLY PROGRESSIVE extra-pyramidal syndrome (dystonia + Parkinsonism + spasticity); "
            "AXONAL NEUROPATHY on nerve conduction (reduced CMAP/SNAP amplitudes); "
            "OPTIC ATROPHY on fundoscopy and OCT (optic disc pallor, thinned RNFL); "
            "PSYCHIATRIC FEATURES: depression (60%), OCD (40%), impulsivity — distinguish from PANK2; "
            "POLISH FOUNDER: c.204_214del11 — Eastern European ancestry raises pre-test probability"
        ),
        "treatment": (
            "NO DISEASE-MODIFYING THERAPY; "
            "PSYCHIATRIC FEATURES: SSRIs for depression/OCD (essential — markedly improves quality of life); "
            "Antipsychotics cautiously for impulse dyscontrol (avoid high-dose anticholinergics); "
            "DYSTONIA/SPASTICITY: trihexyphenidyl + baclofen + botulinum toxin; "
            "IRON CHELATION: deferiprone off-label (same rational as PANK2 but no dedicated RCT); "
            "NEUROPATHY: gabapentin/pregabalin for neuropathic pain; physiotherapy; orthoses; "
            "OPTIC ATROPHY: low vision rehabilitation; contrast sensitivity aids; "
            "PARKINSONISM: levodopa (carbidopa-levodopa) — partial benefit; "
            "MULTIDISCIPLINARY: neuropsychiatry involvement mandatory given psychiatric burden; "
            "GENETIC COUNSELLING: AR; 25% recurrence; founder variant testing rapid in Eastern European"
        ),
        "key_features": [
            "C19orf12: AR; MPAN; second most common NBIA after PANK2 (~20-35% of non-PKAN NBIA)",
            "Triad: optic atrophy + axonal motor neuropathy + neuropsychiatric (depression/OCD/impulsivity)",
            "Polish/Eastern European founder: c.204_214del11 (p.Gly69ArgfsX10)",
            "SLOWLY PROGRESSIVE — later onset than PANK2; mean symptom onset ~10yr; ESRD-equivalent ~30-40yr",
            "GP+SN confluent iron on T2*/SWI — no eye-of-tiger (key DDx from PANK2)",
            "Psychiatric features (SSRIs essential) — often overlooked; depression 60%; OCD 40%",
            "Axonal neuropathy on nerve conduction studies — reduced amplitudes",
            "C19orf12 interacts with COASY/PANK2 — shared CoA pathway; overlapping MPAN/CoPAN phenotypes",
        ],
        "key_ddx": [
            "PANK2 PKAN: eye-of-tiger; childhood onset; no motor neuropathy; CoA biosynthesis same pathway",
            "COASY CoPAN: similar CoA pathway overlap; spastic-dystonia; GP iron milder; no optic atrophy",
            "Hereditary spastic paraplegia (multiple genes): axonal neuropathy overlap; no iron on MRI; no psychiatric",
            "Friedreich ataxia (FXN): peripheral neuropathy + ataxia; cardiac (hypertrophic CMP); no MRI iron",
        ],
        "onset_age": 10.0,
        "gp_iron_pct": 90,
        "eye_of_tiger_pct": 0,
        "dystonia_pct": 78,
        "optic_atrophy_pct": 72,
        "neuropathy_pct": 68,
        "psychiatric_pct": 65,
        "seed": 2609,
    },
    {
        "gene": "FA2H",
        "protein": (
            "FA2H -- 16q23.1 AR -- 480aa -- Fatty-Acid-2-Hydroxylase-55kDa-"
            "2-Hydroxysphingolipid-Myelin-Synthesis-SPG35-FAHN-AR -- OMIM-Gene-611026-Disease-SPG35-612319"
        ),
        "locus": "16q23.1",
        "protein_size": "480 aa / 55 kDa",
        "inheritance": (
            "AR (biallelic loss of function); "
            "FA2H-Associated Neurodegeneration (FAHN) / SPG35 (Hereditary Spastic Paraplegia type 35); "
            "Rare NBIA subtype — <5% of NBIA; "
            "Same gene causes SPG35 and FAHN (phenotypic spectrum); "
            "Onset: childhood 3-15yr; progressive course; "
            "FA2H encodes the enzyme hydroxylating fatty acids in 2-hydroxysphingolipid synthesis for myelin"
        ),
        "disease_category": (
            "FA2H-Associated Neurodegeneration (FAHN) / SPG35; "
            "LEUKODYSTROPHY IS THE EARLIEST AND MOST PROMINENT MRI FEATURE — distinguishes from other NBIA; "
            "Spastic paraplegia dominant feature (hence SPG35 designation); "
            "Cerebellar ataxia + pyramidal signs + periventricular/subcortical white matter changes; "
            "Thin corpus callosum on MRI; "
            "GP iron MILD relative to leukodystrophy — brain iron is secondary, not primary; "
            "2-Hydroxysphingolipids critical for myelin sheath compaction and stability; "
            "Loss → progressive dysmyelination/demyelination"
        ),
        "disease_pathway": (
            "FA2H encodes fatty acid 2-hydroxylase, an ER membrane enzyme that converts "
            "long-chain fatty acids to their 2-hydroxy forms using NADPH and molecular oxygen. "
            "2-Hydroxy fatty acids are essential components of 2-hydroxylated galactosylceramide (HGC) "
            "and 2-hydroxylated sulfatide — major myelin lipids in the CNS and PNS. "
            "Loss of FA2H → deficiency of 2-hydroxysphingolipids → "
            "impaired myelin sheath compaction and stability → primary leukodystrophy. "
            "GP iron deposition is secondary (less prominent than PANK2): "
            "may reflect dying neuronal population in basal ganglia secondary to white matter degeneration. "
            "Thin corpus callosum: callosal axons are among the most myelination-dependent white matter tracts."
        ),
        "pathognomonic": (
            "LEUKODYSTROPHY ON MRI = EARLIEST/MOST PROMINENT FEATURE: "
            "Bilateral symmetrical periventricular + subcortical white matter T2-hyperintensity; "
            "THIN CORPUS CALLOSUM (callosal thinning/atrophy on sagittal MRI); "
            "GP iron PRESENT but MILD and LATE — brain iron is NOT the primary finding (unlike PANK2); "
            "SPASTIC PARAPLEGIA clinically dominant — lower limb spasticity, hyperreflexia, extensor plantar; "
            "CEREBELLAR ATROPHY on MRI (superior cerebellar vermis); "
            "PERIVENTRICULAR ABNORMALITIES early — can mimic other leukodystrophies; "
            "MRI DISTINCTION: brain iron in FAHN is mild/late; leukodystrophy pattern is the clue"
        ),
        "treatment": (
            "NO DISEASE-MODIFYING THERAPY; "
            "SPASTICITY: baclofen (oral or intrathecal pump for severe), tizanidine, botulinum toxin; "
            "PHYSIOTHERAPY: essential to maintain ambulation and prevent contractures; "
            "ORTHOSES: ankle-foot orthoses (AFO) for foot drop; "
            "EPILEPSY (if present): standard AEDs; "
            "ATAXIA: coordination exercises; occupational therapy; "
            "WHEELCHAIR: powered mobility as disease advances; "
            "NUTRITIONAL: dysphagia management (thickened fluids, NGT, PEG if severe); "
            "FATTY ACID SUPPLEMENTATION (lauric acid/dietary): theoretical; no human RCT; "
            "GENETIC COUNSELLING: AR; 25% recurrence; spastic paraplegia gene panel essential"
        ),
        "key_features": [
            "FA2H (FAHN/SPG35): AR; LEUKODYSTROPHY earliest + most prominent MRI finding — unlike other NBIA",
            "Spastic paraplegia dominant clinically (SPG35) + cerebellar ataxia + pyramidal signs",
            "Thin corpus callosum on sagittal MRI — progressive callosal atrophy",
            "GP iron MILD and LATE — brain iron secondary to white matter degeneration (DDx from PANK2/C19orf12)",
            "2-Hydroxysphingolipid deficiency → dysmyelination: myelin sheath compaction defect",
            "Same FA2H gene → SPG35 (spastic paraplegia emphasis) and FAHN (full neurodegeneration spectrum)",
            "Childhood onset 3-15yr; progressive; mean ambulation loss ~10yr after onset",
            "Leukodystrophy pattern can mimic CADASIL, PKAN atypical, or other hereditary leukodystrophies",
        ],
        "key_ddx": [
            "PANK2 PKAN: eye-of-tiger prominent iron; dystonia dominant; minimal leukodystrophy",
            "Metachromatic leukodystrophy (ARSA): similar leukodystrophy pattern; peripheral neuropathy; arylsulfatase A low; no iron",
            "X-linked adrenoleukodystrophy (ABCD1): males; adrenal involvement; VLCFA elevated; no iron accumulation",
            "Hereditary spastic paraplegia (HSP other genes): many without iron; MLPA/NGS panel for SPG35 distinction",
        ],
        "onset_age": 8.0,
        "gp_iron_pct": 55,
        "eye_of_tiger_pct": 0,
        "leukodystrophy_pct": 95,
        "dystonia_pct": 40,
        "spasticity_pct": 92,
        "thin_cc_pct": 88,
        "seed": 2610,
    },
    {
        "gene": "ATP13A2",
        "protein": (
            "ATP13A2 -- 1p36.13 AR -- 1180aa -- P5-Type-ATPase-Lysosomal-128kDa-"
            "Zinc-Polyamine-Transporter-KRS-PARK9-AR -- OMIM-Gene-610513-Disease-KRS-606693"
        ),
        "locus": "1p36.13",
        "protein_size": "1180 aa / 128 kDa",
        "inheritance": (
            "AR (biallelic loss of function); "
            "Kufor-Rakeb Syndrome (KRS) / PARK9; "
            "Juvenile Parkinsonism with pyramidal and supranuclear features; "
            "Rare NBIA subtype — Middle East, specifically Jordan (Kufor-Rakeb) founder; "
            "Consanguinity common in original Jordanian family; "
            "ATP13A2 encodes a lysosomal P5-type ATPase (zinc/polyamine transporter); "
            "Also implicated in neuronal ceroid lipofuscinosis (NCL4B) at same locus"
        ),
        "disease_category": (
            "Kufor-Rakeb Syndrome (KRS) — juvenile Parkinson + supranuclear gaze palsy + pyramidal signs; "
            "Onset: 11-16yr; levodopa-responsive initially (unlike most NBIA); "
            "Extra pyramidal (PARKINSONISM dominant): bradykinesia + rigidity + tremor; "
            "SUPRANUCLEAR GAZE PALSY: upward gaze limitation (vertical supranuclear gaze palsy); "
            "Pyramidal signs: spasticity, hyperreflexia, extensor plantar; "
            "Dementia + psychiatric features with disease progression; "
            "Pallido-pyramidal degeneration + SN/GP iron deposition"
        ),
        "disease_pathway": (
            "ATP13A2 encodes a lysosomal P-type ATPase (P5 subfamily) involved in transporting "
            "zinc (Zn2+) and polyamines across the lysosomal membrane. "
            "Loss of ATP13A2 → lysosomal zinc accumulation + polyamine transport disruption → "
            "lysosomal alkalisation + impaired lysosomal hydrolase function → "
            "failure of autophagy-lysosomal pathway (ALP) → "
            "accumulation of α-synuclein, neuromelanin, and damaged organelles in dopaminergic neurons. "
            "Zinc toxicity and polyamine accumulation directly impair mitochondrial function. "
            "Iron accumulation in GP/SN is secondary to lysosomal dysfunction and neuronal death. "
            "Levodopa responsiveness: surviving dopaminergic neurons retain partial function early."
        ),
        "pathognomonic": (
            "JUVENILE PARKINSONISM + SUPRANUCLEAR UPWARD GAZE PALSY + PYRAMIDAL SIGNS = KRS TRIAD; "
            "LEVODOPA RESPONSE (initially): improvement of Parkinsonism distinguishes from most NBIA; "
            "PALLIDO-PYRAMIDAL DEGENERATION on MRI: GP + putamen iron + cortical atrophy; "
            "SN IRON on T2*/SWI MRI; "
            "FACIAL FINGER TREMOR (fine); "
            "OCULOGYRIC CRISES: paroxysmal eye deviation (upward); "
            "PSYCHIATRIC: visual hallucinations + dementia progression; "
            "FOUNDER: Middle East (Jordan specifically) consanguineous families; "
            "NERVE CONDUCTION: occasionally axonal neuropathy"
        ),
        "treatment": (
            "LEVODOPA (carbidopa-levodopa): FIRST-LINE; initially responsive; "
            "doses titrate as in Parkinson's; benefit wanes as disease advances; "
            "DOPAMINE AGONISTS: pramipexole, ropinirole — adjunct to levodopa; "
            "ANTICHOLINERGICS (trihexyphenidyl): for tremor; use cautiously (cognitive side effects); "
            "SPASTICITY: baclofen, tizanidine; "
            "PSYCHIATRIC: antipsychotics with caution (worsen motor features); "
            "quetiapine preferred (least dopamine antagonism); "
            "ZINC CHELATION: D-penicillamine theoretical; no clinical evidence; "
            "DBS: GPi DBS consideration in advanced motor complications; "
            "GENETIC COUNSELLING: AR; founder testing rapid in Jordanian/consanguineous families; "
            "LYSOSOMAL FUNCTION SUPPORT: investigational"
        ),
        "key_features": [
            "ATP13A2 (KRS/PARK9): AR; juvenile Parkinsonism onset 11-16yr; levodopa-responsive initially",
            "KRS TRIAD: Parkinsonism + vertical supranuclear gaze palsy + pyramidal signs",
            "LEVODOPA RESPONSE distinguishes KRS from most other NBIA subtypes (important diagnostic clue)",
            "Lysosomal P5-type ATPase zinc/polyamine transporter — lysosomal dysfunction pathway",
            "Middle East/Jordanian consanguineous families (Kufor-Rakeb village origin)",
            "Oculogyric crises — paroxysmal upward gaze deviation; psychiatric features with progression",
            "SN+GP iron on MRI; pallido-pyramidal degeneration; cortical atrophy over time",
            "Also allelic with NCL type 4B (neuronal ceroid lipofuscinosis) — same ATP13A2 locus",
        ],
        "key_ddx": [
            "PANK2 PKAN: dystonia > Parkinsonism; no levodopa response; eye-of-tiger; no gaze palsy",
            "Wilson disease (ATP7B): liver; KF rings; low ceruloplasmin; treatable with chelation; juvenile onset",
            "SCA (spinocerebellar ataxia): cerebellar dominant; gaze palsy (cerebellar type); no iron; different ataxia types",
            "Idiopathic juvenile Parkinson (LRRK2/GBA): no supranuclear gaze palsy; no pyramidal; no iron MRI",
        ],
        "onset_age": 13.0,
        "gp_iron_pct": 80,
        "eye_of_tiger_pct": 0,
        "dystonia_pct": 50,
        "levodopa_response_pct": 85,
        "gaze_palsy_pct": 78,
        "pyramidal_pct": 82,
        "seed": 2611,
    },
    {
        "gene": "COASY",
        "protein": (
            "COASY -- 17q21.2 AR -- 579aa -- CoA-Synthase-Bifunctional-64kDa-"
            "PPAT-DPCK-CoA-Biosynthesis-CoPAN-NBIA6-AR -- OMIM-Gene-609855-Disease-CoPAN-615643"
        ),
        "locus": "17q21.2",
        "protein_size": "579 aa / 64 kDa",
        "inheritance": (
            "AR (biallelic loss of function); "
            "CoPAN (CoA Synthase Protein-Associated Neurodegeneration); NBIA6; "
            "EXTREMELY RARE — only a handful of families worldwide; "
            "COASY encodes bifunctional CoA synthase: "
            "  - PPAT domain: 4'-phosphopantetheine adenylyltransferase; "
            "  - DPCK domain: dephospho-CoA kinase; "
            "CoA biosynthesis: Pantothenate → (PANK2) → 4'-Phosphopantothenate → ... → (COASY) → CoA; "
            "Same CoA biosynthesis pathway as PANK2 (downstream enzyme)"
        ),
        "disease_category": (
            "CoA Synthase Protein-Associated Neurodegeneration (CoPAN); "
            "PHENOTYPICALLY OVERLAPS PKAN (same CoA pathway): spastic-dystonia + mild cognitive impairment; "
            "GP iron on MRI — MILD (no eye-of-tiger; less prominent than PANK2); "
            "SPASTICITY DOMINANT (unlike PKAN where dystonia dominant); "
            "Later onset than PKAN classic (childhood/adolescent vs PKAN 3-6yr); "
            "Cognitive impairment present but milder than other NBIA subtypes; "
            "Psychiatric features: obsessional traits; "
            "COASY downstream of PANK2 in CoA biosynthesis — CoA deficiency mechanism shared"
        ),
        "disease_pathway": (
            "COASY encodes bifunctional CoA synthase (PPAT-DPCK) that catalyses the "
            "final two steps of CoA biosynthesis: "
            "Step 1 (PPAT domain): 4'-phosphopantetheine + ATP → dephospho-CoA + PPi; "
            "Step 2 (DPCK domain): dephospho-CoA + ATP → CoA + ADP. "
            "Loss of COASY → CoA deficiency (same downstream consequence as PANK2 loss, "
            "but affecting a different step of the same biosynthetic pathway). "
            "CoA deficiency → impaired fatty acid oxidation + acylation reactions + TCA cycle intermediates → "
            "mitochondrial dysfunction → cysteine/iron accumulation in GP (mechanism shared with PANK2 but less severe). "
            "Why milder than PANK2? PANK2 is rate-limiting; COASY step has partial redundancy "
            "and COASY is expressed more widely with partial compensation in some tissues."
        ),
        "pathognomonic": (
            "GP IRON ON T2*/SWI — present but MILD: less than PANK2; no eye-of-tiger central hyperintensity; "
            "SPASTIC-DYSTONIA phenotype: lower limb spasticity + dystonia combined; "
            "CoA BIOSYNTHESIS PATHWAY GENE panel: COASY detected alongside PANK2 testing; "
            "COGNITIVE IMPAIRMENT: mild; frontally predominant; "
            "PSYCHIATRIC: obsessional traits, anxiety; "
            "CHILDHOOD/ADOLESCENT ONSET: distinguishes from MPAN (later adult) but overlaps PKAN atypical; "
            "NERVE CONDUCTION: occasionally mild axonal changes; "
            "CLINICAL OVERLAP WITH PKAN: spastic-dystonia + GP iron = same clinical lane; "
            "COASY genetic diagnosis often by exome/genome after PANK2 negative"
        ),
        "treatment": (
            "NO DISEASE-MODIFYING THERAPY; management analogous to PKAN: "
            "SPASTICITY: baclofen + tizanidine; intrathecal baclofen pump for severe; "
            "DYSTONIA: trihexyphenidyl + botulinum toxin + GPi DBS consideration; "
            "CoA SUPPLEMENTATION: pantothenate supplementation theoretical (limited evidence); "
            "IRON CHELATION: deferiprone off-label (same rational as PANK2); "
            "PSYCHIATRIC: SSRIs for obsessional traits; "
            "COGNITIVE SUPPORT: neuropsychological assessment + adaptive aids; "
            "PHYSIOTHERAPY + OCCUPATIONAL THERAPY: functional maintenance; "
            "GENETIC COUNSELLING: AR; 25% recurrence; ultra-rare — genetic counsellor specialist referral"
        ),
        "key_features": [
            "COASY (CoPAN): AR; SAME CoA biosynthesis pathway as PANK2 — downstream enzyme; extremely rare",
            "GP iron on MRI: MILD, no eye-of-tiger (key DDx from PANK2 despite same pathway)",
            "SPASTICITY DOMINANT (unlike PKAN where dystonia dominant) — phenotypic difference despite shared pathway",
            "Later onset than classic PKAN; milder progression; cognitive impairment present",
            "Bifunctional enzyme (PPAT + DPCK): two CoA biosynthesis steps affected simultaneously",
            "Detected by exome/genome after PANK2 negative on targeted panel — rare variant",
            "Overlaps ClinGen/ClinVar with C19orf12 (MPAN) phenotype — shared CoA/mitochondrial pathway",
            "PPAT = 4'-phosphopantetheine adenylyltransferase; DPCK = dephospho-CoA kinase — both domains critical",
        ],
        "key_ddx": [
            "PANK2 PKAN: same CoA pathway but upstream; eye-of-tiger present; more severe; younger onset classic",
            "C19orf12 MPAN: overlapping spastic-dystonia + psychiatric; optic atrophy; motor neuropathy",
            "HSP (hereditary spastic paraplegia): spasticity dominant; no iron; SPG gene panel (SPG11/SPG7/SPG35)",
            "Dopa-responsive dystonia (GCH1): dramatic levodopa response; diurnal fluctuation; no iron MRI",
        ],
        "onset_age": 12.0,
        "gp_iron_pct": 72,
        "eye_of_tiger_pct": 0,
        "dystonia_pct": 68,
        "spasticity_pct": 80,
        "cognitive_impairment_pct": 65,
        "psychiatric_pct": 55,
        "seed": 2612,
    },
    {
        "gene": "DCAF17",
        "protein": (
            "DCAF17 -- 2q31.1 AR -- 520aa -- DDB1-CUL4-Associated-Factor-17-58kDa-"
            "E3-Ubiquitin-Ligase-Substrate-Receptor-WSS-NBIA-AR -- OMIM-Gene-612515-Disease-WSS-241080"
        ),
        "locus": "2q31.1",
        "protein_size": "520 aa / 58 kDa",
        "inheritance": (
            "AR (biallelic loss of function); "
            "Woodhouse-Sakati Syndrome (WSS); "
            "UNIQUE MULTISYSTEM NBIA — endocrine + neurological + sensorineural; "
            "Gulf Arab founder: p.Cys44Tyr (c.131G>A) enriched in Saudi Arabia + Gulf region; "
            "DCAF17 is a substrate receptor for the DDB1-CUL4 E3 ubiquitin ligase complex; "
            "Ubiquitin-mediated proteasomal degradation of specific cellular proteins"
        ),
        "disease_category": (
            "Woodhouse-Sakati Syndrome (WSS) — UNIQUE multisystem NBIA with endocrine features: "
            "HYPOGONADISM (primary; males + females) + ALOPECIA (generalised) + TYPE 2 DM + "
            "SENSORINEURAL DEAFNESS + NBIA (basal ganglia iron) + NEURODEGENERATION; "
            "This combination IS PATHOGNOMONIC for WSS; "
            "Neurological: dystonia + dementia + pyramidal signs; "
            "Endocrine: hypogonadism evident at puberty (pubertal failure); "
            "Middle East founder population; consanguinity common; "
            "DCAF17 ubiquitin ligase pathway → accumulation of specific substrates"
        ),
        "disease_pathway": (
            "DCAF17 is a substrate receptor (adaptor) for the CRL4-DDB1 E3 ubiquitin ligase complex, "
            "which attaches polyubiquitin tags to specific substrate proteins for proteasomal degradation. "
            "DCAF17 directs the CRL4 complex to specific cellular targets — the identity of its key "
            "neurological and endocrine substrates remains incompletely characterised. "
            "Loss of DCAF17 → failure to ubiquitinate and degrade specific substrate proteins → "
            "accumulation of these substrates in neurons (basal ganglia, frontotemporal cortex) and "
            "endocrine cells (gonads, pancreatic β-cells, cochlear hair cells). "
            "Iron accumulation in basal ganglia: secondary to neuronal death and lysosomal release of "
            "iron from ferritin. The unique multi-organ endocrine phenotype distinguishes DCAF17 from "
            "all other NBIA genes — no other NBIA gene causes hypogonadism + DM + alopecia."
        ),
        "pathognomonic": (
            "UNIQUE WSS PENTAD = PATHOGNOMONIC: "
            "1. HYPOGONADISM (primary; pubertal failure; elevated FSH/LH; low oestradiol/testosterone); "
            "2. ALOPECIA (generalised, diffuse scalp hair loss from early adolescence); "
            "3. TYPE 2 DIABETES MELLITUS (often insulin-requiring); "
            "4. SENSORINEURAL HEARING LOSS (bilateral, progressive); "
            "5. NEURODEGENERATION + BASAL GANGLIA IRON (dystonia + dementia + GP/caudate iron T2*/SWI); "
            "MIDDLE EAST ancestry raises pre-test probability; "
            "Gulf Arab founder p.Cys44Tyr — rapid targeted testing if suspected; "
            "NEUROLOGICAL: dystonia + cognitive impairment + pyramidal signs; "
            "MRI: caudate + GP iron + frontal white matter changes"
        ),
        "treatment": (
            "MULTISYSTEM MANAGEMENT: "
            "HYPOGONADISM: oestrogen replacement (females — HRT); testosterone replacement (males); "
            "Aim: puberty induction, bone mineral density preservation, cardiovascular protection; "
            "DIABETES: insulin ± metformin; monitor HbA1c; ophthalmology/podiatry screening; "
            "HEARING LOSS: hearing aids; cochlear implants for severe-profound deafness; "
            "ALOPECIA: cosmetic management; wigs; counselling (significant psychosocial impact); "
            "DYSTONIA: trihexyphenidyl + botulinum toxin + baclofen; "
            "GPi DBS consideration for severe disabling dystonia; "
            "DEMENTIA: cognitive support, orientation, safety planning; "
            "IRON CHELATION: deferiprone off-label; no RCT; "
            "GENETIC COUNSELLING: AR; Gulf Arab founder testing; 25% recurrence; "
            "ENDOCRINOLOGY ANNUAL REVIEW: hormone levels, bone density, glucose, hearing"
        ),
        "key_features": [
            "DCAF17 (WSS): AR; UNIQUE multisystem NBIA — ONLY NBIA with hypogonadism + alopecia + DM + deafness",
            "WSS PENTAD PATHOGNOMONIC: hypogonadism + alopecia + DM + SNHL + NBIA neurodegeneration",
            "Gulf Arab founder variant p.Cys44Tyr — Saudi Arabia + Gulf region enrichment",
            "DDB1-CUL4 E3 ubiquitin ligase substrate receptor — ubiquitin-proteasomal pathway",
            "Pubertal failure: primary hypogonadism; HRT mandatory for bone + cardiovascular health",
            "Neurological: dystonia + progressive dementia + pyramidal signs + basal ganglia iron",
            "Any NBIA + endocrine (hypogonadism/alopecia/DM/SNHL) = WSS until proven otherwise",
            "Multidisciplinary mandatory: endocrinology + neurology + audiology + ophthalmology + genetics",
        ],
        "key_ddx": [
            "Other NBIA subtypes: NO other NBIA gene causes hypogonadism + alopecia + DM — combination unique to WSS",
            "Kallmann syndrome (ANOS1/FGFR1): hypogonadism + anosmia; no alopecia, DM, hearing loss, or brain iron",
            "Turner syndrome (45,X): hypergonadotrophic hypogonadism; no neurodegeneration; no alopecia/DM",
            "MIDD (mitochondrial DM + deafness, MT-TL1/MT-ND4): DM + SNHL but no alopecia/hypogonadism/iron",
        ],
        "onset_age": 14.0,
        "gp_iron_pct": 82,
        "eye_of_tiger_pct": 0,
        "dystonia_pct": 72,
        "hypogonadism_pct": 95,
        "alopecia_pct": 90,
        "diabetes_pct": 82,
        "snhl_pct": 88,
        "seed": 2613,
    },
]

SEEDS = [g["seed"] for g in ATLAS_GENES]


def _simulate_cohort(gene: dict, seed: int) -> list:
    rng = random.Random(seed)
    pts = []
    n = 40
    for i in range(n):
        age_onset = gene.get("onset_age", 8.0) + rng.gauss(0, 2.5)
        age_onset = max(0.5, age_onset)
        gp_iron = int(rng.random() < gene.get("gp_iron_pct", 70) / 100)
        eye_of_tiger = int(rng.random() < gene.get("eye_of_tiger_pct", 5) / 100)
        dystonia = int(rng.random() < gene.get("dystonia_pct", 60) / 100)
        pts.append({
            "gene": gene["gene"],
            "patient_id": f"{gene['gene']}-{seed}-{i+1:03d}",
            "age_onset": round(age_onset, 1),
            "gp_iron": gp_iron,
            "eye_of_tiger": eye_of_tiger,
            "dystonia": dystonia,
            "levodopa_response": int(gene["gene"] == "ATP13A2" and rng.random() < 0.85),
            "deferiprone_use": int(rng.random() < 0.35),
            "dbs_implanted": int(rng.random() < gene.get("dbs_response_pct", 10) / 100),
            "seizures": int(gene["gene"] == "WDR45" or rng.random() < 0.25),
            "optic_atrophy": int(rng.random() < gene.get("optic_atrophy_pct", 20) / 100),
            "cognitive_impairment": int(rng.random() < 0.60),
            "seed": seed,
        })
    return pts


def generate_overview() -> dict:
    summary_by_gene = []
    all_pts = []
    for gene in ATLAS_GENES:
        pts = _simulate_cohort(gene, gene["seed"])
        all_pts.extend(pts)
        n = len(pts)
        summary_by_gene.append({
            "gene": gene["gene"],
            "locus": gene["locus"],
            "n_patients": n,
            "avg_onset_age": round(sum(p["age_onset"] for p in pts) / n, 1),
            "gp_iron_pct": round(sum(p["gp_iron"] for p in pts) / n * 100, 1),
            "eye_of_tiger_pct": round(sum(p["eye_of_tiger"] for p in pts) / n * 100, 1),
            "dystonia_pct": round(sum(p["dystonia"] for p in pts) / n * 100, 1),
        })

    total = len(all_pts)
    return {
        "atlas": "Hereditary-NBIA-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": len(ATLAS_GENES),
        "total_patients": total,
        "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "gene_summaries": summary_by_gene,
        "aggregate_stats": {
            "overall_gp_iron_pct": round(sum(p["gp_iron"] for p in all_pts) / total * 100, 1),
            "overall_dystonia_pct": round(sum(p["dystonia"] for p in all_pts) / total * 100, 1),
            "overall_eye_of_tiger_pct": round(sum(p["eye_of_tiger"] for p in all_pts) / total * 100, 1),
        },
        "disease_classes": [
            f"{g['gene']} — {g['disease_category'].split(';')[0].strip()}"
            for g in ATLAS_GENES
        ],
        "key_clinical_distinctions": [
            "PANK2 PKAN: Eye-of-Tiger sign PATHOGNOMONIC (>90%) — bilateral GP iron + central T2-hyperintensity; ~50% of all NBIA",
            "PLA2G6 PLAN: Neuroaxonal spheroids on EM PATHOGNOMONIC; optic ATROPHY (not retinopathy); cerebellar atrophy; NO eye-of-tiger",
            "WDR45 BPAN: FEMALES; BIPHASIC (childhood seizures → adult Parkinsonism/dementia) PATHOGNOMONIC; T1 HALO not eye-of-tiger",
            "C19orf12 MPAN: Triad optic atrophy + axonal neuropathy + neuropsychiatric; Polish founder; slowly progressive",
            "FA2H FAHN/SPG35: LEUKODYSTROPHY earliest/most prominent MRI; thin corpus callosum; spasticity dominant; iron MILD",
            "ATP13A2 KRS/PARK9: Juvenile Parkinsonism + supranuclear gaze palsy + pyramidal = TRIAD; LEVODOPA-RESPONSIVE (unique)",
            "COASY CoPAN: CoA pathway same as PANK2 (downstream); GP iron MILD; spasticity dominant; extremely rare",
            "DCAF17 WSS: ONLY NBIA with hypogonadism + alopecia + DM + SNHL — UNIQUE multisystem; Gulf Arab founder",
        ],
    }


def generate_breakdown() -> dict:
    gene_breakdowns = []
    for gene in ATLAS_GENES:
        pts = _simulate_cohort(gene, gene["seed"])
        n = len(pts)
        entry = {
            "gene": gene["gene"],
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"],
            "disease_category": gene["disease_category"],
            "pathognomonic": gene["pathognomonic"],
            "treatment": gene["treatment"],
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
            "n_patients": n,
            "avg_onset_age": round(sum(p["age_onset"] for p in pts) / n, 1),
            "gp_iron_pct": round(sum(p["gp_iron"] for p in pts) / n * 100, 1),
            "eye_of_tiger_pct": round(sum(p["eye_of_tiger"] for p in pts) / n * 100, 1),
            "dystonia_pct": round(sum(p["dystonia"] for p in pts) / n * 100, 1),
            "levodopa_response_pct": round(sum(p["levodopa_response"] for p in pts) / n * 100, 1),
            "cognitive_impairment_pct": round(sum(p["cognitive_impairment"] for p in pts) / n * 100, 1),
        }
        gene_breakdowns.append(entry)
    return {"gene_breakdowns": gene_breakdowns}


def generate_definitions() -> dict:
    gene_entries = {}
    for gene in ATLAS_GENES:
        gene_entries[gene["gene"]] = {
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"].split(";")[0].strip(),
            "disease_name": gene["disease_category"].split(";")[0].strip(),
            "disease_pathway": gene["disease_pathway"],
            "pathognomonic": gene["pathognomonic"],
            "treatment_summary": gene["treatment"].split(";")[0].strip() + "...",
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
        }
    return {
        "gene_entries": gene_entries,
        "nbia_glossary": {
            "NBIA — Neurodegeneration with Brain Iron Accumulation": (
                "NBIA is an umbrella term for a clinically and genetically heterogeneous group of "
                "progressive neurodegenerative disorders characterised by iron deposition in basal ganglia "
                "(especially globus pallidus and substantia nigra) on MRI T2*/SWI. "
                "Prevalence: ~1-3 per million. Combined, PANK2 (PKAN) accounts for ~50% of NBIA; "
                "the remaining 50% split among PLA2G6 (PLAN), WDR45 (BPAN), C19orf12 (MPAN), "
                "FA2H (FAHN/SPG35), ATP13A2 (KRS), COASY (CoPAN), DCAF17 (WSS), and others. "
                "Clinical features: extra-pyramidal syndrome (dystonia/Parkinsonism/spasticity) + "
                "progressive neurodegeneration ± cognitive impairment. "
                "Key MRI: T2*/SWI iron hypointensity in GP/SN; Eye-of-tiger (PANK2 only); T1 halo (WDR45); leukodystrophy (FA2H)."
            ),
            "Eye-of-Tiger Sign (PKAN PATHOGNOMONIC)": (
                "The eye-of-tiger sign is a T2-MRI finding pathognomonic for PKAN (PANK2): "
                "Bilateral globus pallidus (inner segment predominantly) shows T2 hypointensity (iron deposition) "
                "with a CENTRAL AREA OF T2 HYPERINTENSITY (gliosis + oedema — the 'pupil' of the tiger's eye). "
                "The central hyperintensity distinguishes PKAN from other NBIA subtypes which show "
                "GP iron without central hyperintensity. "
                "SWI sequence is more sensitive than T2 for iron detection. "
                "Seen in >90% of PKAN; its ABSENCE should prompt consideration of other NBIA diagnoses."
            ),
            "Deferiprone (DFP) in NBIA": (
                "Deferiprone is an oral iron chelator (3-hydroxypyridin-4-one) used off-label in NBIA "
                "based on the hypothesis that reducing local brain iron would slow neurodegeneration. "
                "B-PKAN trial (2022, Lancet Neurology): deferiprone reduced iron accumulation on MRI "
                "but did NOT improve neurological outcomes (UDRS motor score) over 18 months. "
                "Off-label use continues given progressive/fatal nature of NBIA with no alternatives. "
                "Monitoring: neutrophil count weekly × 6m then every 2 weeks (agranulocytosis risk); "
                "LFTs; renal function. "
                "No RCT evidence for PLA2G6, WDR45, C19orf12, or other NBIA subtypes."
            ),
            "CoA Biosynthesis Pathway (PANK2-COASY Overlap)": (
                "Coenzyme A (CoA) biosynthesis pathway in mitochondria: "
                "Pantothenate (B5) → [PANK2] → 4'-phosphopantothenate → "
                "→ 4'-phosphopantothenoyl-cysteine → 4'-phosphopantetheine → "
                "[COASY-PPAT domain] → dephospho-CoA → [COASY-DPCK domain] → CoA. "
                "PANK2 (PKAN) = step 1 (rate-limiting); COASY (CoPAN) = steps 5-6. "
                "Both deficiencies → CoA deficiency → impaired fatty acid β-oxidation + iron/cysteine accumulation in GP. "
                "PKAN is more severe: PANK2 is rate-limiting with no backup. "
                "CoPAN (COASY) has partial redundancy → milder phenotype with less prominent iron."
            ),
        },
    }
