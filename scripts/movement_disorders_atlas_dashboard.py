#!/usr/bin/env python3
"""Movement Disorders Atlas — Complete 8-Gene Hereditary Movement Disorders Atlas
HTT    (Huntingtin; 3144 aa; 4p16.3; Huntington disease; CAG repeat >36; AD; striatal neurodegeneration; no disease-modifying therapy approved; tetrabenazine/deutetrabenazine for chorea) ·
ATP7B  (Wilson ATPase; 1465 aa; 13q14.3; Wilson disease; AR; copper accumulation; liver+brain+KF rings; penicillamine/trientine chelation; zinc maintenance; TREATABLE-CURABLE if caught early) ·
TOR1A  (Torsin A; 332 aa; 9q34.11; DYT-TOR1A / DYT1 early-onset generalized dystonia; AD de novo; 3-bp GAG deletion common; limb onset; levodopa TRIAL mandatory; GPi-DBS highly effective) ·
PANK2  (Pantothenate kinase 2; 570 aa; 20p13; PKAN — Neurodegeneration with Brain Iron Accumulation type 1; AR; eye-of-the-tiger MRI sign PATHOGNOMONIC; pigmentary retinopathy; early-onset) ·
VPS13A (Vacuolar protein sorting 13A; 3174 aa; 9q21.2; Chorea-Acanthocytosis; AR; orofacial dystonia+self-mutilation PATHOGNOMONIC; acanthocytes on blood film; elevated CK; caudate atrophy) ·
GCH1   (GTP cyclohydrolase I; 250 aa; 14q22.2; Dopa-responsive dystonia DRD / Segawa syndrome; AD; DIURNAL VARIATION PATHOGNOMONIC; dramatic levodopa response; female predominance) ·
PRRT2  (Proline-rich transmembrane protein 2; 340 aa; 16p11.2; PKD — Paroxysmal Kinesigenic Dyskinesia; AD; brief attacks triggered by MOVEMENT; carbamazepine first-line; NOT a channelopathy by EEG) ·
ATP1A3 (Na+/K+-ATPase alpha3; 1013 aa; 19q13.2; AHC — Alternating Hemiplegia of Childhood; AD de novo; hemiplegic attacks + oculomotor crises + fixed dystonia; flunarizine first-line; avoid triggers)
320-patient aggregate cohort (8 × 40, seeds 1078–1085)
"""

import random

SEED_BASE = 1078

MD_GENES = [
    # ── HTT — Huntington Disease ─────────────────────────────────────────────
    {
        "gene": "HTT", "protein": "Huntingtin (HTT)",
        "alias": "HTT; OMIM gene 613004; 4p16.3; ~3144 aa; Huntington disease (OMIM #143100); CAG repeat >36; autosomal dominant; striatal neurodegeneration; trinucleotide repeat expansion",
        "aa": "~3144 aa", "kDa": "~348 kDa",
        "mechanism": (
            "HTT encodes huntingtin, a large multifunctional scaffold protein ubiquitously expressed "
            "throughout the nervous system, with highest levels in medium spiny neurons (MSNs) of the striatum. "
            "NORMAL FUNCTION: huntingtin participates in vesicular transport, transcriptional regulation, "
            "synaptic function, and anti-apoptotic signalling. CAG repeat in exon 1 codes for a polyglutamine "
            "(polyQ) tract — normal alleles: ≤26 CAG; intermediate: 27–35 (not HD but may expand in next generation); "
            "reduced penetrance: 36–39 CAG; full penetrance: ≥40 CAG. "
            "PATHOMECHANISM: expanded polyQ tract causes mutant huntingtin (mHTT) to misfold → "
            "gain-of-toxic function → mHTT aggregates disrupt: (1) transcription (CREB, CBP sequestration); "
            "(2) axonal transport (disrupts dynein/kinesin machinery); (3) mitochondrial function "
            "(impairs complex II/III — striatum uniquely vulnerable); (4) synaptic transmission "
            "(disrupts BDNF release/transport from cortex → striatum dependence on cortical BDNF); "
            "(5) protein clearance (overloads ubiquitin-proteasome system and autophagy). "
            "STRIATAL SELECTIVITY: MSNs (indirect pathway — expressing D2R, enkephalin) are preferentially lost early, "
            "causing CHOREA (indirect pathway loss → thalamic disinhibition → cortical hyperactivation); "
            "direct pathway (D1R, SP) lost later → rigidity/bradykinesia in later stages. "
            "CAG REPEAT LENGTH AND ONSET: each additional CAG repeat shortens onset by ~3–4 years; "
            "juvenile HD (<20 years onset) has >55 CAG repeats and presents with RIGIDITY (Westphal variant), "
            "not chorea. ANTICIPATION: repeat expansions more common in paternal transmission (sperm selection)."
        ),
        "disease_type": "Huntington Disease (AD CAG trinucleotide repeat expansion — full penetrance ≥40 CAG)",
        "locus": "4p16.3", "omim_gene": 613004, "omim_disease": 143100,
        "inheritance": (
            "AUTOSOMAL DOMINANT — FULLY PENETRANT for ≥40 CAG repeats (no obligate carrier state). "
            "Intermediate repeats (27–35 CAG): unaffected carrier, risk of expansion in next generation. "
            "Reduced penetrance (36–39 CAG): may or may not develop HD (age-dependent incomplete penetrance). "
            "ANTICIPATION: CAG repeat can expand across generations (especially paternal); "
            "a father with 42 CAG may have a child with 55+ CAG → juvenile HD. "
            "GENETIC TESTING: PCR-based CAG repeat sizing (Southern blot for very large expansions). "
            "PREDICTIVE TESTING: requires genetic counselling protocol (HD is 100% penetrant — "
            "positive result has profound psychological implications). At-risk individuals must consent; "
            "children should NOT be tested (cannot make autonomous decision). "
            "PRENATAL/PREIMPLANTATION: PGT-M available for at-risk pregnancies."
        ),
        "phenotype": (
            "HUNTINGTON DISEASE (HD): "
            "ONSET: Mean 30–50 years (adult form); range 2–80 years. "
            "MOTOR: CHOREA is the hallmark (involuntary, writhing, dancing movements) — initially subtle "
            "(restlessness, clumsiness) → orofacial dyskinesia → generalised; "
            "later stages: chorea decreases, replaced by rigidity + dystonia + bradykinesia. "
            "PSYCHIATRIC (often FIRST): depression (40%), anxiety, irritability, disinhibition, apathy, "
            "obsessive-compulsive symptoms — may precede motor by years. "
            "COGNITIVE: subcortical dementia — executive dysfunction, processing speed, memory retrieval "
            "(recall worse than recognition); progresses to severe dementia. "
            "DYSPHAGIA + WEIGHT LOSS: aspiration risk; caloric supplementation needed. "
            "JUVENILE HD (<20y, >55 CAG — Westphal variant): rigidity > chorea; seizures; "
            "rapid progression; Parkinson-like phenotype. "
            "PROGRESSION: TFC (total functional capacity) scale 0–13; mean survival 15–20 years from motor onset; "
            "cause of death: pneumonia (aspiration) > falls > suicide (psychiatric phase). "
            "BIOMARKER: CSF/blood neurofilament light (NfL) rises years before symptom onset — "
            "research use; mutant huntingtin (mHTT) assay in CSF now available."
        ),
        "treatment_options": [
            "Tetrabenazine (TBZ): VMAT2 inhibitor — monoamine depleter; FDA-approved for HD chorea; "
            "reduces chorea; side effects: depression (SCREEN BEFORE USE — risk worsening in HD psychiatry), "
            "Parkinsonism, sedation, akathisia; start low 12.5mg/day titrate; max 100mg/day; "
            "CYP2D6 pharmacogenomics affects dose",
            "Deutetrabenazine (AUSTEDO): deuterium-modified TBZ; FDA approved for HD chorea 2017; "
            "longer half-life → twice-daily dosing; fewer psychiatric side effects than TBZ; "
            "preferred over TBZ in patients at psychiatric risk; must screen for depression/suicidality",
            "Valbenazine (INGREZZA): VMAT2 inhibitor; FDA approved for tardive dyskinesia; "
            "off-label for HD chorea; once-daily; slower titration; less clinical HD data vs TBZ/DTBZ",
            "Antidepressants: SSRIs/SNRIs (sertraline, fluoxetine) for HD depression and irritability; "
            "SCREEN BEFORE starting tetrabenazine/deutetrabenazine — combined depression risk; "
            "mirtazapine for anxiety + weight loss promotion",
            "Antipsychotics: quetiapine, aripiprazole, olanzapine — for psychiatric symptoms (psychosis, "
            "aggression, irritability); avoid haloperidol if possible (worsens motor, older data); "
            "atypicals preferred",
            "Riluzole: modulates glutamate; some evidence for slowing HD progression (modest); "
            "well-tolerated; not standard of care but reasonable adjunct",
            "Multidisciplinary care: MANDATORY — neurologist + psychiatrist + speech-language pathologist "
            "(dysphagia assessment REQUIRED), dietician (weight monitoring, caloric support), "
            "physical + occupational therapy; falls prevention; advance directives early",
            "DISEASE-MODIFYING (INVESTIGATIONAL): HTT-lowering strategies — antisense oligonucleotides (ASOs, "
            "tominersen — Phase 3 GENERATION HD1 paused 2021 due to safety signal at high doses); "
            "RNA interference (siRNA/shRNA); small molecule allele-selective approaches; "
            "trial participation encouraged given no approved disease-modifying therapy",
        ],
        "key_ddx": [
            "Dentatorubral-pallidoluysian atrophy (DRPLA — ATN1) — CAG repeat; ataxia + choreoathetosis + myoclonus; East Asian predominance",
            "Benign hereditary chorea (NKX2-1) — childhood-onset; non-progressive; normal cognition; thyroid involvement",
            "McLeod syndrome (XK gene) — X-linked; chorea + neuropsychiatric; acanthocytes; Kell blood group anomaly; muscle disease",
            "Sydenham's chorea — post-streptococcal; self-limiting; no family history; ASTR titre",
            "Drug-induced chorea (metoclopramide, levodopa, antipsychotics) — medication history; no HD family history",
        ],
        "onset_range_y": (30.0, 55.0),
        "sex_female_prob": 0.50,
        "eeg_pattern": "Normal (HD) / Diffuse slowing in advanced stage",
        "movement_type": "Chorea (involuntary writhing) / Rigidity (late / juvenile) / Oculomotor dysfunction",
        "severity_dist": {"Severe": 0.30, "Moderate": 0.50, "Mild": 0.20},
        "seizure_free_rate": 0.90,
        "progression_rate": 0.90,
        "drug_error_rate": 0.10,
        "targeted_therapy_available": False,
        "first_line_drug": "Tetrabenazine / Deutetrabenazine (for chorea; SCREEN depression first)",
        "critical_avoid": "SCREEN for depression BEFORE tetrabenazine (suicidality risk); NO disease-modifying therapy approved (ASOs failed Phase 3 2021); MANDATORY: genetic counselling before predictive testing; AVOID testing children",
    },
    # ── ATP7B — Wilson Disease ────────────────────────────────────────────────
    {
        "gene": "ATP7B", "protein": "Copper-Transporting ATPase Beta (ATP7B / Wilson ATPase)",
        "alias": "ATP7B; OMIM gene 606882; 13q14.3; ~1465 aa; Wilson disease (OMIM #277900); AR; copper accumulation liver/brain/cornea; TREATABLE — early diagnosis essential",
        "aa": "~1465 aa", "kDa": "~165 kDa",
        "mechanism": (
            "ATP7B encodes a copper-transporting P-type ATPase expressed predominantly in hepatocytes and "
            "neurons. NORMAL FUNCTION: ATP7B transports copper from hepatocytes into bile (biliary excretion "
            "= the primary route of copper elimination from the body) and also incorporates copper into "
            "ceruloplasmin (the major plasma copper carrier) in the trans-Golgi network. "
            "WILSON PATHOMECHANISM: biallelic loss-of-function mutations → failure of biliary copper excretion "
            "→ copper accumulates progressively in: (1) LIVER — hepatocellular copper deposition → "
            "lipid peroxidation → Fenton reaction → reactive oxygen species → hepatitis → cirrhosis → "
            "acute liver failure (Wilson's presenting as ALF requires urgent LT referral); "
            "(2) BRAIN — copper deposits preferentially in BASAL GANGLIA (putamen > caudate > globus pallidus) "
            "and CEREBELLUM → neuronal death → movement disorder; also FRONTAL LOBE → psychiatric symptoms; "
            "(3) CORNEA — Kayser-Fleischer (KF) rings: copper in Descemet membrane (posterior corneal ring "
            "= PATHOGNOMONIC with neurological Wilson); (4) KIDNEY — renal tubular acidosis (Fanconi syndrome), "
            "aminoaciduria, haematuria; (5) JOINTS — early-onset arthropathy; "
            "(6) HAEMOLYSIS — acute Coombs-negative haemolytic anaemia (copper-induced RBC damage). "
            "CERULOPLASMIN: low serum ceruloplasmin (<200 mg/L) is the classical screening finding "
            "(ATP7B is required for ceruloplasmin copper incorporation); "
            "DIAGNOSTIC TRIAD: low ceruloplasmin + high 24h urine copper + KF rings (in neurological Wilson)."
        ),
        "disease_type": "Wilson Disease — hepatolenticular degeneration (AR biallelic ATP7B LOF; copper accumulation; TREATABLE)",
        "locus": "13q14.3", "omim_gene": 606882, "omim_disease": 277900,
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic pathogenic variants. "
            "Prevalence: ~1:30,000 (carrier frequency ~1:90). "
            "Most patients are compound heterozygotes (two different mutations). "
            "MOST COMMON MUTATIONS: p.His1069Gln (European, ~40% of alleles in European populations); "
            "p.Arg778Leu (East Asian). Many private/rare variants exist (>500 reported). "
            "GENETIC TESTING: sequencing identifies >95% of pathogenic variants; "
            "gene panel recommended. Siblings of index cases: 25% risk — test ALL siblings. "
            "FAMILY SCREENING: parents are obligate carriers (test to confirm); "
            "offspring of patients are obligate carriers; offspring risk depends on partner status. "
            "PRENATAL DIAGNOSIS: available if both parental mutations known."
        ),
        "phenotype": (
            "WILSON DISEASE — TREATABLE METABOLIC NEURODEGENERATION: "
            "ONSET: 5–35 years (mean ~12–15y hepatic; ~20y neurological; rarely infantile or >40y). "
            "HEPATIC (commoner in children/adolescents): asymptomatic transaminitis → "
            "chronic hepatitis → cirrhosis → ACUTE LIVER FAILURE (ALF with Coombs-negative haemolysis = "
            "Wilson's crisis — EMERGENCY; very high urine copper; immediate LT evaluation). "
            "NEUROLOGICAL (commoner in young adults): "
            "WING-BEATING TREMOR (proximal arm tremor with elbows abducted = PATHOGNOMONIC); "
            "dysarthria (scanning/slurred); dysphagia; dystonia (orofacial + limb); "
            "ataxia; Parkinsonism; chorea; "
            "MRI: T2 hyperintensity basal ganglia (putamen > caudate > globus pallidus) + "
            "PANDA SIGN (midbrain T2 changes giving panda face) = highly suggestive Wilson. "
            "PSYCHIATRIC (often first neurological sign): personality change, depression, psychosis, "
            "impulsivity — misdiagnosed as primary psychiatric disorder for years. "
            "KAYSER-FLEISCHER RINGS: golden-brown corneal ring visible on slit-lamp examination — "
            "PRESENT in >95% of neurological Wilson; only 50–60% of hepatic Wilson. "
            "SUNFLOWER CATARACTS: greenish lens opacities — another copper deposit."
        ),
        "treatment_options": [
            "D-Penicillamine (DPA): copper chelator — first available treatment; "
            "induces cupriuresis (increased urine copper excretion); "
            "SIDE EFFECTS: neurological worsening paradox (20–50% worsen neurologically on starting DPA — "
            "mobilises brain copper initially); lupus-like reaction; nephrotoxicity; thrombocytopenia; "
            "pyridoxine supplementation required (DPA is anti-pyridoxine); "
            "less favoured in neurological Wilson due to worsening risk",
            "Trientine (TETA — triethylenetetramine): copper chelator; fewer side effects than DPA; "
            "PREFERRED first-line for symptomatic Wilson (hepatic + neurological); "
            "less neurological worsening than DPA; monitor urine copper + CBC + LFT; "
            "given 3–4 times daily away from food and zinc",
            "Zinc acetate/gluconate: ZINC THERAPY — not a chelator but inhibits intestinal copper absorption "
            "(induces metallothionein in enterocytes → copper sequestered and lost in shed cells); "
            "FIRST-LINE for PRESYMPTOMATIC patients (siblings screened after index case); "
            "used as MAINTENANCE after initial chelation-induced decoppering; "
            "safe in pregnancy; no neurological worsening risk; 50mg elemental zinc 3x/day",
            "Liver transplantation (LT): for Wilson's acute liver failure (ALF) or end-stage cirrhosis; "
            "CURATIVE for liver disease — transplanted liver normalises copper metabolism; "
            "neurological symptoms can improve post-LT; LT does not cure neurological Wilson alone; "
            "MELD score guides urgency; LT evaluation in ALF-Wilson is EMERGENCY (hours matter)",
            "Tetrathiomolybdate (TM): rapidly reduces free copper (forms stable complex with albumin + "
            "copper + TM); less neurological worsening than DPA; clinical trials ongoing (WTR-001); "
            "not yet widely available",
            "Monitoring: 24h urine copper (goal 200–500 mcg/day on chelation); serum non-ceruloplasmin-bound "
            "copper (free copper — goal <10 mcg/dL); serum ceruloplasmin (rises with treatment); "
            "LFT; CBC; urinalysis — EVERY 3–6 months; neurological exam + brain MRI annually in neuro Wilson",
        ],
        "key_ddx": [
            "Autoimmune hepatitis — ANA/ASMA positive; responds to steroids; no KF rings; copper normal",
            "Juvenile Parkinson's disease (PARK2/PINK1) — no hepatic disease; no KF rings; low ceruloplasmin absent",
            "Dystonia (TOR1A/DYT1) — no hepatic involvement; no KF rings; ceruloplasmin normal",
            "Aceruloplasminaemia — absent ceruloplasmin due to CP gene mutation; anaemia; diabetes; no KF rings; different MRI",
            "Drug-induced hepatitis/neurological syndrome — medication history; no copper accumulation",
        ],
        "onset_range_y": (8.0, 30.0),
        "sex_female_prob": 0.45,
        "eeg_pattern": "Diffuse slowing (advanced) / Normal in early neurological Wilson",
        "movement_type": "Wing-beating tremor (PATHOGNOMONIC) / Dystonia / Ataxia / Parkinsonism / Chorea",
        "severity_dist": {"Severe": 0.20, "Moderate": 0.45, "Mild": 0.35},
        "seizure_free_rate": 0.80,
        "progression_rate": 0.65,
        "drug_error_rate": 0.15,
        "targeted_therapy_available": True,
        "first_line_drug": "Trientine (symptomatic) / Zinc (presymptomatic/maintenance) / Liver transplant (ALF/cirrhosis)",
        "critical_avoid": "AVOID penicillamine first-line in neurological Wilson (20-50% paradoxical worsening — mobilises brain copper); SCREEN ALL SIBLINGS with slit-lamp + ceruloplasmin + urine copper; Wilson's ALF is a TRANSPLANT EMERGENCY; KF rings on slit-lamp MANDATORY in neurological workup",
    },
    # ── TOR1A — DYT-TOR1A / Early-Onset Generalized Dystonia ─────────────────
    {
        "gene": "TOR1A", "protein": "Torsin Family 1 Member A (TOR1A / Torsin-1A)",
        "alias": "TOR1A; OMIM gene 605204; 9q34.11; ~332 aa; DYT-TOR1A / DYT1 early-onset generalized dystonia (OMIM #128100); AD; 3-bp GAG deletion (p.Glu302del) most common; GPi-DBS highly effective",
        "aa": "~332 aa", "kDa": "~37.8 kDa",
        "mechanism": (
            "TOR1A encodes torsin-1A, an AAA+ ATPase residing in the endoplasmic reticulum (ER) lumen "
            "and perinuclear space (nuclear envelope). NORMAL FUNCTION: torsin-1A acts as a chaperone "
            "maintaining nuclear envelope integrity, facilitating nuclear pore complex biogenesis, "
            "nuclear membrane protein clearance, and cytoskeletal interactions (linker of nucleoskeleton "
            "to cytoskeleton — LINC complex). "
            "DYT1 PATHOMECHANISM: the GAG deletion (most common: p.Glu302del, removing one glutamic acid "
            "from a glutamate repeat in the C-terminal domain) acts via DOMINANT NEGATIVE mechanism — "
            "mutant torsin-1A loses ATPase activity → forms non-functional complexes with WT torsin-1A "
            "→ traps WT in non-functional state → nuclear envelope dysfunction → "
            "impaired nucleocytoplasmic transport → abnormal perinuclear blebs (morphological hallmark in "
            "neurons). REDUCED PENETRANCE: DYT1 is only 30% penetrant (despite being autosomal dominant) — "
            "only 30% of carriers develop dystonia; genetic modifiers, epigenetics, and stochastic "
            "developmental factors determine expressivity. "
            "CIRCUIT DYSFUNCTION: torsin-1A loss impairs D1-pathway striatal output → "
            "basal ganglia-thalamo-cortical circuit dysfunction → loss of motor inhibition → dystonic postures. "
            "DOPAMINE HYPOTHESIS: dopamine turnover disrupted in DYT1 striatum — explains levodopa trial "
            "rationale and variable response."
        ),
        "disease_type": "DYT-TOR1A (DYT1) Early-Onset Generalised Dystonia (AD; 3-bp GAG deletion; 30% penetrance; GPi-DBS transformative)",
        "locus": "9q34.11", "omim_gene": 605204, "omim_disease": 128100,
        "inheritance": (
            "AUTOSOMAL DOMINANT with REDUCED PENETRANCE (~30%). "
            "Most common cause: 3-bp in-frame GAG deletion in exon 5 (p.Glu302del) — "
            "present in ~80% of ASHKENAZI JEWISH early-onset dystonia families; "
            "also found in non-Jewish populations worldwide (founder effect). "
            "Other TOR1A variants (missense, truncating) are rare and may have different penetrance. "
            "PENETRANCE MODIFIERS: TOR polymorphism (D216H) reduces penetrance; environmental triggers "
            "may influence onset; sex (females slightly more penetrant in some studies). "
            "GENETIC COUNSELLING: 30% penetrance means most carriers (~70%) are unaffected; "
            "offspring of affected individual: 50% carry variant, of those ~30% will develop dystonia "
            "= ~15% risk per offspring. "
            "TESTING STRATEGY: molecular testing for GAG deletion (PCR/Sanger); gene panel for atypical cases."
        ),
        "phenotype": (
            "DYT-TOR1A (DYT1) EARLY-ONSET GENERALIZED DYSTONIA: "
            "ONSET: Childhood/adolescence — mean 12–13 years; 70% onset ≤26 years. "
            "TYPICAL PRESENTATION: involuntary sustained muscle contractions → abnormal postures/movements; "
            "LIMB-ONSET most common (often one leg: foot inversion, tip-toeing, gait dystonia); "
            "then SPREADS to other limbs and trunk (generalises in 60% within 5 years). "
            "CERVICAL DYSTONIA: torticollis/anterocollis (in some cases). "
            "CRANIAL INVOLVEMENT: less prominent than secondary dystonias. "
            "DIURNAL VARIATION: typically ABSENT in DYT1 (distinguishes from dopa-responsive GCH1). "
            "LEVODOPA TRIAL: MANDATORY in all early-onset dystonia — DYT-GCH1 has dramatic response; "
            "DYT1 has partial/no response; trial guides diagnosis. "
            "FUNCTIONAL IMPACT: significant disability from sustained abnormal postures; "
            "writing, walking, self-care impaired; no cognitive impairment; normal life expectancy. "
            "GPi-DBS OUTCOME: EXCELLENT — globus pallidus internus DBS produces 50–80% improvement in "
            "dystonia scales (BFMDRS); consider when ≥2 medications failed and disability significant; "
            "younger age at surgery predicts better outcome."
        ),
        "treatment_options": [
            "Levodopa/carbidopa TRIAL: MANDATORY in all early-onset dystonia — must rule out GCH1/DYT-GCH1 "
            "(dopa-responsive dystonia has dramatic response); DYT1 has minimal/no levodopa response; "
            "trial: 3–6 months, titrate to at least 300mg/day levodopa equivalent before declaring non-response",
            "Trihexyphenidyl (anticholinergic): first-line medical therapy in DYT1; "
            "high-dose trihexyphenidyl (20–120mg/day) — most effective oral treatment; "
            "side effects: cognitive (memory, confusion — dose-limiting in adults, better tolerated in children), "
            "anticholinergic (dry mouth, urinary retention, blurred vision, constipation); "
            "titrate slowly; children tolerate higher doses",
            "Baclofen (oral + intrathecal): oral baclofen adjunctive; intrathecal baclofen (ITB) pump "
            "for axial/lower limb predominant dystonia; direct spinal delivery → less systemic side effects; "
            "requires implanted pump + regular refill",
            "Clonazepam / diazepam: adjunctive; reduces muscle spasm; sedation limits dose; "
            "useful short-term in dystonic crisis or as adjunctive agent",
            "GPi-DBS (Globus Pallidus Internus Deep Brain Stimulation): "
            "GOLD STANDARD for medically refractory generalised DYT1 dystonia; "
            "bilateral GPi stimulation → 50–80% improvement in BFMDRS (Burke-Fahn-Marsden Dystonia Rating Scale); "
            "DELAYED RESPONSE (unlike Parkinson's) — improvement over 6–12 months post-implant; "
            "DYT1 responds BETTER than secondary/symptomatic dystonias; "
            "refer when ≥2 medications failed + significant functional disability",
            "Botulinum toxin type A or B: for focal/segmental dystonia (cervical, cranial); "
            "less useful for generalised DYT1 (too widespread to inject all muscles); "
            "injections every 3 months; EMG guidance improves accuracy; "
            "may complement DBS for residual focal involvement",
        ],
        "key_ddx": [
            "GCH1 dopa-responsive dystonia (DRD) — DIURNAL VARIATION (worse at day end); dramatic levodopa response; MANDATORY trial before DYT1 label",
            "Wilson disease (ATP7B) — ALWAYS exclude in early-onset dystonia: ceruloplasmin, slit-lamp, urine copper",
            "Secondary dystonia (structural, metabolic, drug-induced) — MRI brain mandatory; drug history",
            "PANK2/PKAN — eye-of-the-tiger MRI; pigmentary retinopathy; AR; different MRI pattern",
            "KMT2B dystonia (DYT28) — early childhood-onset generalised + later oromandibular; KMT2B mutations; responds to GPi-DBS",
        ],
        "onset_range_y": (8.0, 26.0),
        "sex_female_prob": 0.55,
        "eeg_pattern": "Normal (primary dystonia — no EEG abnormality)",
        "movement_type": "Sustained muscle contractions → abnormal postures (generalised dystonia); limb-onset; torsional",
        "severity_dist": {"Severe": 0.30, "Moderate": 0.50, "Mild": 0.20},
        "seizure_free_rate": 0.95,
        "progression_rate": 0.60,
        "drug_error_rate": 0.20,
        "targeted_therapy_available": True,
        "first_line_drug": "Levodopa TRIAL (mandatory to exclude GCH1) then Trihexyphenidyl / GPi-DBS (refractory)",
        "critical_avoid": "NEVER label early-onset dystonia as DYT1 without levodopa TRIAL (missing dopa-responsive GCH1 is a major error); ALWAYS exclude Wilson (ATP7B) with ceruloplasmin + slit-lamp; GPi-DBS HIGHLY effective — early referral if medications failing",
    },
    # ── PANK2 — PKAN (Pantothenate Kinase-Associated Neurodegeneration) ───────
    {
        "gene": "PANK2", "protein": "Pantothenate Kinase 2 (PANK2)",
        "alias": "PANK2; OMIM gene 606157; 20p13; ~570 aa; PKAN — Neurodegeneration with Brain Iron Accumulation type 1 NBIA1 (OMIM #234200); AR; eye-of-the-tiger MRI sign PATHOGNOMONIC",
        "aa": "~570 aa", "kDa": "~63 kDa",
        "mechanism": (
            "PANK2 encodes pantothenate kinase 2, the first and rate-limiting enzyme of the coenzyme A (CoA) "
            "biosynthesis pathway. PANK2 is mitochondrially targeted and specifically expressed at high levels "
            "in neurons of the globus pallidus interna (GPi) and substantia nigra pars reticulata (SNr). "
            "NORMAL FUNCTION: phosphorylates pantothenate (vitamin B5) → 4'-phosphopantothenate → "
            "eventual synthesis of CoA → essential for fatty acid metabolism, TCA cycle, "
            "acetylcholine synthesis, and mitochondrial energy production in neurons. "
            "PKAN PATHOMECHANISM: biallelic PANK2 mutations → CoA deficiency in neurons → "
            "accumulation of cysteine-containing substrates (cysteinyl-pantothenate, cysteine) → "
            "cysteine reacts with iron (Fenton chemistry) → iron-catalysed free radical generation → "
            "neuronal death in GPi and SNr (iron-rich areas). "
            "EYE-OF-THE-TIGER SIGN: T2-weighted MRI shows hypointense (iron) areas in globus pallidus "
            "surrounding a central T2-hyperintense (gliosis/vacuolization) core — "
            "PATHOGNOMONIC for PKAN (>95% specificity). "
            "CLASSIC vs ATYPICAL PKAN: classic (null mutations, no PANK2 activity) = early-onset (<10y), "
            "rapid progression; atypical (residual PANK2 activity) = later onset (10–25y), slower, "
            "speech/psychiatric prominent. "
            "ACANTHOCYTES: spiky RBCs (acanthocytosis) occur in some PKAN patients — "
            "indicates neuroacanthocytosis overlap or separate diagnosis; not universal in PKAN."
        ),
        "disease_type": "PKAN — NBIA Type 1 (AR biallelic PANK2; iron accumulation GPi; eye-of-the-tiger T2 MRI PATHOGNOMONIC)",
        "locus": "20p13", "omim_gene": 606157, "omim_disease": 234200,
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic pathogenic variants. "
            "Prevalence: ~1–3 per million (rare). "
            "CLASSIC PKAN: typically homozygous or compound heterozygous null variants (frameshift, nonsense, "
            "splice-site) — complete loss of PANK2 activity. "
            "ATYPICAL PKAN: missense variants with residual PANK2 function. "
            "NO FOUNDER MUTATION: broad allelic heterogeneity (>100 variants reported). "
            "GENETIC TESTING: PANK2 gene sequencing; if negative with classical phenotype, consider: "
            "COASY (NBIA5-CoA synthase), FA2H (FAHN), MPAN (C19orf12), BPAN (WDR45 — X-linked). "
            "FAMILY TESTING: siblings at 25% risk; parents are obligate carriers."
        ),
        "phenotype": (
            "PKAN — NEURODEGENERATION WITH BRAIN IRON ACCUMULATION TYPE 1: "
            "CLASSIC PKAN (onset 3–6 years): "
            "GAIT DISTURBANCE — initial feature (foot dystonia, tip-toeing); "
            "GENERALISED DYSTONIA — progressive; axial > limb; oromandibular involvement; "
            "SPASTICITY — pyramidal signs (hyperreflexia, extensor plantar); "
            "DYSARTHRIA — severe, unintelligible by adolescence; "
            "PIGMENTARY RETINOPATHY — visual impairment in ~65% (salt-and-pepper retinopathy, "
            "optic atrophy); MANDATORY ophthalmological assessment; "
            "COGNITIVE DECLINE — intellectual deterioration, behavioural disturbance; "
            "PALLIDAL ATROPHY on MRI — diminishing globus pallidus volume with disease progression. "
            "EYE-OF-THE-TIGER SIGN (MRI HALLMARK): T2 hypointense GP (iron) + central T2 hyperintense "
            "area (gliosis) — seen in BOTH classic and atypical PKAN; diagnostic. "
            "ATYPICAL PKAN (onset 10–25 years): slower progression; speech/language prominent; "
            "psychiatric symptoms; parkinsonism; may have long plateau phases. "
            "COURSE: relentless progression; wheelchair-bound by teens in classic; "
            "death typically 3rd–4th decade classic, later atypical."
        ),
        "treatment_options": [
            "Pantethine / pantothenol (high-dose): bypasses PANK2 step → increases CoA substrate availability; "
            "anecdotal benefit; some case reports of stabilisation; not proven in RCT; "
            "reasonable empirical trial given benign safety profile; dose: several grams/day",
            "Deferiprone (iron chelator): removes brain iron; clinical trials (TIRCON consortium — phase 2); "
            "reduces MRI iron signal in globus pallidus; some stabilisation reported; "
            "side effects: agranulocytosis (MANDATORY weekly CBC for first 6 months), nausea; "
            "most data support slowing rather than reversing progression",
            "Trihexyphenidyl / anticholinergics: for dystonia management; high-dose; "
            "similar to DYT1 approach but PKAN dystonia is more refractory; "
            "partial benefit in some patients",
            "Intrathecal baclofen (ITB) pump: for severe spasticity + axial dystonia; "
            "direct spinal delivery; reduces tone; may help functional ability and comfort; "
            "requires maintenance and refill",
            "GPi-DBS: some cases show improvement; less predictable than DYT1; "
            "case series suggest modest benefit in dystonia reduction; discuss at specialist centre; "
            "MRI safety of DBS with iron accumulation — special programming protocols needed",
            "Botulinum toxin: for focal high-impact dystonia (oromandibular, cervical, limb); "
            "EMG-guided; improves focal function; every 3 months",
            "Symptomatic: PEG (percutaneous endoscopic gastrostomy) for dysphagia/aspiration; "
            "communication devices (AAC); physiotherapy; wheelchair; ophthalmology follow-up; "
            "low-vision aids; advance care planning",
        ],
        "key_ddx": [
            "Other NBIA disorders — MPAN (C19orf12), BPAN (WDR45 X-linked), COASY (NBIA5), FA2H (FAHN) — different MRI patterns; genetic panel",
            "Chorea-Acanthocytosis (VPS13A) — eye-of-the-tiger absent; acanthocytes; CK elevated; orofacial self-mutilation",
            "Huntington disease — no eye-of-the-tiger; chorea not dystonia; adult onset; HTT CAG repeat",
            "DYT-TOR1A — normal MRI (no iron); no retinopathy; responds to GPi-DBS; levodopa trial",
            "Neuronal ceroid lipofuscinosis (NCL) — visual failure + seizures + dementia; NCL on EM; different MRI",
        ],
        "onset_range_y": (3.0, 15.0),
        "sex_female_prob": 0.50,
        "eeg_pattern": "Diffuse slowing (advanced) / Normal early / May have epileptiform discharges in advanced disease",
        "movement_type": "Generalised dystonia (axial+limb) / Spasticity / Dysarthria / Oculomotor (retinopathy)",
        "severity_dist": {"Severe": 0.60, "Moderate": 0.30, "Mild": 0.10},
        "seizure_free_rate": 0.70,
        "progression_rate": 0.85,
        "drug_error_rate": 0.08,
        "targeted_therapy_available": False,
        "first_line_drug": "Trihexyphenidyl (dystonia) / Deferiprone (iron chelation — WEEKLY CBC) / Pantethine (substrate bypass)",
        "critical_avoid": "EYE-OF-THE-TIGER MRI sign is PATHOGNOMONIC — if present, PANK2 testing MANDATORY; MANDATORY ophthalmology (pigmentary retinopathy ~65%); Deferiprone: WEEKLY CBC for agranulocytosis risk; SCREEN siblings (25% risk); GPi-DBS specialist centre evaluation",
    },
    # ── VPS13A — Chorea-Acanthocytosis ───────────────────────────────────────
    {
        "gene": "VPS13A", "protein": "Vacuolar Protein Sorting 13 Homologue A (VPS13A / CHAC)",
        "alias": "VPS13A; OMIM gene 605978; 9q21.2; ~3174 aa; Chorea-Acanthocytosis ChAc (OMIM #200150); AR; orofacial dystonia + self-mutilation PATHOGNOMONIC; acanthocytes on fresh blood film; elevated CK",
        "aa": "~3174 aa", "kDa": "~360 kDa",
        "mechanism": (
            "VPS13A encodes vacuolar protein sorting-associated protein 13A (CHAC protein), a large lipid-transfer "
            "protein at membrane contact sites — specifically at the intersection of mitochondria with ER, "
            "lipid droplets, and endosomes. NORMAL FUNCTION: VPS13A transfers phospholipids between "
            "organellar membranes, maintaining lipid homeostasis, mitochondrial membrane integrity, "
            "and membrane tethering at contact sites — critical for autophagy and mitochondrial dynamics. "
            "ChAc PATHOMECHANISM: biallelic VPS13A null mutations → loss of membrane lipid transfer → "
            "defective membrane composition → aberrant red blood cell (RBC) membrane deformability → "
            "ACANTHOCYTOSIS (spiky RBC morphology — dense projections on cell surface); "
            "in neurons: progressive neurodegeneration of caudate nucleus and putamen "
            "(striatal neurodegeneration — MRI: caudate atrophy > putamen), and frontal cortex; "
            "also substantia nigra involvement. "
            "PATHOGNOMONIC FEATURES: (1) OROFACIAL DYSTONIA — involuntary lip/tongue biting "
            "(self-mutilation of lips, tongue, cheeks) — results from dystonia of orofacial muscles "
            "driving the patient to bite; UNIQUE to ChAc in severity; "
            "(2) FEEDING DYSTONIA — involuntary tongue protrusion during eating (pushes food out); "
            "(3) VOCALISATIONS — grunting, barking. "
            "ELEVATED CK: chronic skeletal myopathy component → elevated creatine kinase in most ChAc patients."
        ),
        "disease_type": "Chorea-Acanthocytosis (AR biallelic VPS13A; caudate neurodegeneration; orofacial self-mutilation + acanthocytes PATHOGNOMONIC)",
        "locus": "9q21.2", "omim_gene": 605978, "omim_disease": 200150,
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic pathogenic variants. "
            "Prevalence: ~1:1–5 million (very rare). "
            "Most mutations: truncating (nonsense, frameshift, splice-site) → null alleles. "
            "No common founder mutation; broad allelic heterogeneity (pan-ethnic, rare globally). "
            "GENETIC TESTING: VPS13A sequencing (large gene — 73 exons; NGS gene panel); "
            "consider whole-exome if panel negative. "
            "DIFFERENTIAL NGS: McLeod syndrome (XK — X-linked male; Kell blood group abnormality; "
            "less severe self-mutilation), PKAN (PANK2), HD (HTT). "
            "FAMILY TESTING: siblings at 25% risk; VPS13A protein Western blot (chorein — near-absent in ChAc) "
            "can confirm diagnosis before genetic result; acanthocytes on fresh blood film (not EDTA)."
        ),
        "phenotype": (
            "CHOREA-ACANTHOCYTOSIS (ChAc): "
            "ONSET: 20–40 years (young adult onset; range 8–62 years). "
            "MOVEMENT DISORDER: CHOREA — involuntary writhing movements (choreiform); "
            "DYSTONIA — orofacial dystonia is HALLMARK (involuntary tongue protrusion, "
            "lip + cheek biting, grunting VOCALISATIONS); "
            "SELF-MUTILATION OF LIPS AND TONGUE — biting driven by dystonic contraction of masticatory "
            "muscles (NOT compulsive); severe cases require dental moulding/mouth guards; "
            "FEEDING DYSTONIA — tongue pushes food out of mouth involuntarily; weight loss; "
            "PARKINSONISM features in later stages (bradykinesia, rigidity). "
            "PSYCHIATRIC: personality change, OCD, frontal lobe syndrome, impulse control disorder, "
            "depression, anxiety — common early features. "
            "COGNITIVE: executive dysfunction, memory problems (frontal-subcortical pattern). "
            "EPILEPSY: seizures in 30–50% of patients (partial > generalised); often prominent. "
            "PERIPHERAL NEUROPATHY: sensorimotor; areflexia; elevated CK (myopathy component). "
            "NEUROIMAGING: caudate atrophy (MRI) + putaminal atrophy; no iron accumulation (unlike PKAN). "
            "BLOOD FILM: acanthocytes (fresh smear — EDTA sample causes pseudo-acanthocytosis) in >80%. "
            "CHOREIN WESTERN BLOT: near-absent chorein (VPS13A protein) in RBCs = confirmatory. "
            "COURSE: relentless neurodegeneration; death 10–30 years after onset."
        ),
        "treatment_options": [
            "Antiepileptics for seizures: levetiracetam, valproate, lamotrigine — standard management; "
            "ChAc epilepsy is often focal; EEG monitoring recommended; status epilepticus risk",
            "Tetrabenazine / deutetrabenazine: for chorea — modest benefit; "
            "may worsen depression/parkinsonism in ChAc (psychiatric monitoring mandatory); "
            "lower doses than HD; screen for depression before starting",
            "Botulinum toxin for orofacial dystonia: injections into orbicularis oris, masseter, "
            "and tongue muscles can reduce severity of self-mutilation; "
            "require EMG guidance; repeat every 3 months; multidisciplinary (dentistry, speech pathology)",
            "Mouth guards / dental mouldings: MANDATORY for lip/tongue self-mutilation; "
            "custom-fitted oral appliance protects mucosa; co-managed with dentistry; "
            "may need replacement as disease progresses",
            "Antipsychotics / mood stabilisers: for psychiatric symptoms (frontal lobe syndrome, "
            "OCD, aggression, impulsivity); quetiapine preferred (less extrapyramidal); "
            "monitor carefully as ChAc patients may worsen with antipsychotics",
            "Antidepressants: SSRIs for depression + OCD features; sertraline or fluvoxamine (OCD); "
            "monitor for drug interactions",
            "GPi-DBS: case reports of benefit for dystonia in ChAc; "
            "less evidence than DYT1; specialist centre evaluation",
            "Multidisciplinary: speech-language pathology (feeding assessment, dysphagia); "
            "dietician (weight monitoring); physiotherapy; wheelchair; PEG if severe dysphagia; "
            "neuropsychiatry for behavioural symptoms; community care coordination",
        ],
        "key_ddx": [
            "McLeod syndrome (XK gene) — X-linked males; weaker expression females; Kell antigen absent/weak; cardiac (cardiomyopathy); less severe lip biting",
            "Huntington disease (HTT) — chorea + dementia; no acanthocytes; no self-mutilation; positive CAG repeat",
            "PKAN (PANK2) — eye-of-the-tiger MRI; retinopathy; no acanthocytes; AR",
            "Tardive dyskinesia (drug-induced) — orofacial movements post-antipsychotics; medication history; acanthocytes absent",
            "Wilson disease (ATP7B) — KF rings; liver disease; ceruloplasmin low; copper elevated; treatable",
        ],
        "onset_range_y": (20.0, 40.0),
        "sex_female_prob": 0.50,
        "eeg_pattern": "Focal or generalised epileptiform discharges (30-50% have epilepsy) / Diffuse slowing",
        "movement_type": "Chorea + Orofacial dystonia (self-mutilation PATHOGNOMONIC) + Parkinsonism (late)",
        "severity_dist": {"Severe": 0.40, "Moderate": 0.45, "Mild": 0.15},
        "seizure_free_rate": 0.55,
        "progression_rate": 0.80,
        "drug_error_rate": 0.12,
        "targeted_therapy_available": False,
        "first_line_drug": "Tetrabenazine (chorea) / Botulinum toxin (orofacial dystonia) / AEDs (seizures) / Mouth guards (self-mutilation)",
        "critical_avoid": "FRESH BLOOD SMEAR for acanthocytes (EDTA sample causes pseudoacanthocytes — false negative); CHOREIN WESTERN BLOT is rapid confirmatory test; MANDATORY mouth guards for self-mutilation; Screen for seizures (30-50%); tetrabenazine — SCREEN depression first; VPS13A is the largest gene in the neuroacanthocytosis group",
    },
    # ── GCH1 — Dopa-Responsive Dystonia / Segawa Syndrome ────────────────────
    {
        "gene": "GCH1", "protein": "GTP Cyclohydrolase I (GCH1)",
        "alias": "GCH1; OMIM gene 600225; 14q22.2; ~250 aa; Dopa-Responsive Dystonia DRD / Segawa syndrome (OMIM #128230); AD; DIURNAL VARIATION PATHOGNOMONIC; dramatic levodopa response; female predominance 3:1",
        "aa": "~250 aa", "kDa": "~28 kDa",
        "mechanism": (
            "GCH1 encodes GTP cyclohydrolase I, the rate-limiting enzyme of the tetrahydrobiopterin (BH4) "
            "biosynthesis pathway. BH4 is the essential cofactor for ALL three aromatic amino acid hydroxylases: "
            "tyrosine hydroxylase (TH — dopamine synthesis), tryptophan hydroxylase (TpH — serotonin synthesis), "
            "and phenylalanine hydroxylase (PAH — phenylalanine metabolism). "
            "GCH1 PATHOMECHANISM in DRD: heterozygous GCH1 LOF → haploinsufficiency → "
            "reduced BH4 production → reduced TH cofactor → reduced dopamine synthesis in nigrostriatal neurons → "
            "dopamine deficiency in striatum (caudate/putamen) → dopaminergic pathway hypofunction → "
            "dystonia AND Parkinsonism. "
            "DIURNAL VARIATION MECHANISM: dopamine is DEPLETED by normal neuronal activity throughout the day; "
            "re-synthesised during rest/sleep; morning = replete dopamine → BEST function; "
            "evening = depleted dopamine → WORST function. This diurnal cycle creates the PATHOGNOMONIC "
            "diurnal variation of DRD symptoms (worse in afternoon/evening, better after rest/sleep). "
            "LEVODOPA RESPONSE: exogenous levodopa is converted to dopamine → replenishes depleted "
            "striatal dopamine → DRAMATIC and SUSTAINED response to LOW-DOSE levodopa — "
            "distinguishing from other dystonias and parkinsonism syndromes. "
            "CSF BIOPTERIN: reduced CSF neopterin and biopterin in GCH1-DRD; "
            "BH4 loading test shows subnormal response — diagnostic in equivocal cases."
        ),
        "disease_type": "Dopa-Responsive Dystonia DRD / Segawa Syndrome (AD GCH1 haploinsufficiency; DIURNAL VARIATION; dramatic low-dose levodopa response; female 3:1)",
        "locus": "14q22.2", "omim_gene": 600225, "omim_disease": 128230,
        "inheritance": (
            "AUTOSOMAL DOMINANT with reduced penetrance (30% penetrance overall; "
            "FEMALES: ~60% penetrance; MALES: ~15% penetrance — sex-specific penetrance difference). "
            "Female-to-male ratio: 3:1 in clinical presentations. "
            "SEX DIFFERENCE MECHANISM: females have higher estrogen → estrogen inhibits TH activity → "
            "females require full BH4 for adequate dopamine → more sensitive to GCH1 haploinsufficiency. "
            "AUTOSOMAL RECESSIVE DRD: biallelic GCH1 mutations cause severe hyperphenylalaninaemia + "
            "severe DRD + serotonin deficiency — different phenotype (HPA); also TH gene mutations (AR-DRD). "
            "GENETIC TESTING: GCH1 sequencing (>100 variants; some promoter mutations missed by exon-only sequencing); "
            "BH4 loading test (non-invasive — phenylalanine loading); "
            "CSF neurotransmitter analysis (low HVA, low biopterin) at specialised centre."
        ),
        "phenotype": (
            "DOPA-RESPONSIVE DYSTONIA (DRD) / SEGAWA SYNDROME: "
            "ONSET: 6–16 years (range 1–40 years; occasionally adult-onset). "
            "CARDINAL FEATURES — THREE PATHOGNOMONIC: "
            "(1) DIURNAL VARIATION: worse in the afternoon/evening (dopamine depleted by activity); "
            "dramatically BETTER after sleep/rest (replenished overnight); "
            "CHILD who is NORMAL IN THE MORNING but limps/falls/has difficulty walking by evening → DRD. "
            "(2) GAIT DYSTONIA: foot dystonia → equinus posture → toe-walking; "
            "lower limb > upper limb onset; young child may appear to have cerebral palsy. "
            "(3) DRAMATIC LEVODOPA RESPONSE: sustained complete or near-complete resolution of dystonia "
            "with LOW DOSE levodopa (levodopa 1–3 mg/kg/day); no wearing off; no dyskinesias (unlike PD); "
            "response maintained for decades. "
            "PARKINSONISM: may accompany or predominate (especially in males with mild presentation — "
            "tremor, bradykinesia, rigidity); OFTEN MISDIAGNOSED AS YOUNG-ONSET PARKINSON'S. "
            "PSYCHOMOTOR RETARDATION: subtle in some cases; academic difficulty. "
            "DIAGNOSIS IS OFTEN DELAYED 5–10 YEARS: common misdiagnoses include spastic diplegia (CP), "
            "hereditary spastic paraplegia, or functional neurological disorder. "
            "LEVODOPA TRIAL IS THE DIAGNOSTIC KEY: all early-onset dystonia MUST have levodopa trial."
        ),
        "treatment_options": [
            "Levodopa/carbidopa: CURATIVE TREATMENT for GCH1-DRD; "
            "low doses sufficient (1–3 mg/kg/day levodopa + 1/4 carbidopa ratio; "
            "typical adult dose 50–200mg/day levodopa in 3 divided doses); "
            "DRAMATIC RESPONSE expected within days to 2 weeks; "
            "SUSTAINED benefit without wearing-off or dyskinesia (unlike Parkinson's); "
            "titrate up slowly to minimise nausea; maintain LIFELONG (stopping causes relapse); "
            "DO NOT withhold if diagnosis suspected — response confirms diagnosis",
            "Carbidopa alone: reduces peripheral levodopa breakdown — always combine with levodopa; "
            "standard combination tablets (Sinemet/Madopar); avoid carbidopa alone therapy",
            "BH4 (sapropterin dihydrochloride): direct BH4 supplementation; "
            "may benefit DRD especially in BH4-deficiency conditions; "
            "less used in GCH1-DRD where levodopa is more directly curative; "
            "useful in AR-GCH1 with HPA (hyperphenylalaninaemia) — lowers phenylalanine",
            "5-hydroxytryptophan (5-HTP) + carbidopa: for AR GCH1 or TH mutations with serotonin deficiency; "
            "serotonin precursor supplementation alongside levodopa",
            "Anticholinergics (trihexyphenidyl): adjunctive if levodopa partially incomplete; "
            "rarely needed given excellent levodopa response in DRD",
        ],
        "key_ddx": [
            "Cerebral palsy (spastic diplegia) — NO diurnal variation; perinatal history; MRI brain usually abnormal; no levodopa response",
            "DYT-TOR1A (DYT1) — no diurnal variation; no levodopa response; GAG deletion; GPi-DBS consideration",
            "Young-onset Parkinson's disease (PARK2/PINK1) — prominent parkinsonism; levodopa response but with dyskinesias; no diurnal variation; different genetic cause",
            "Hereditary spastic paraplegia (SPG4/SPAST) — hyperreflexia; no dystonia; no diurnal variation; levodopa does not help",
            "Functional neurological disorder (FND) — variable; no diurnal variation; no genetic cause; Hoover sign",
        ],
        "onset_range_y": (6.0, 18.0),
        "sex_female_prob": 0.75,
        "eeg_pattern": "Normal (primary dystonia/parkinsonism — no epileptiform activity)",
        "movement_type": "Foot dystonia (gait) → DIURNAL VARIATION (PATHOGNOMONIC) / Parkinsonism / Levodopa-responsive",
        "severity_dist": {"Severe": 0.15, "Moderate": 0.45, "Mild": 0.40},
        "seizure_free_rate": 0.95,
        "progression_rate": 0.20,
        "drug_error_rate": 0.30,
        "targeted_therapy_available": True,
        "first_line_drug": "Levodopa/carbidopa (CURATIVE — low dose; sustained lifelong response; no wearing-off)",
        "critical_avoid": "NEVER skip levodopa TRIAL in early-onset dystonia (missing DRD = preventable years of disability); DIURNAL VARIATION is PATHOGNOMONIC — child who improves after sleep has DRD until proven otherwise; STOP only if no response after adequate dose/duration trial; Female predominance reflects incomplete penetrance sex difference",
    },
    # ── PRRT2 — Paroxysmal Kinesigenic Dyskinesia (PKD) ──────────────────────
    {
        "gene": "PRRT2", "protein": "Proline-Rich Transmembrane Protein 2 (PRRT2)",
        "alias": "PRRT2; OMIM gene 614386; 16p11.2; ~340 aa; PKD — Paroxysmal Kinesigenic Dyskinesia (OMIM #128200); AD; attacks triggered by SUDDEN MOVEMENT; carbamazepine FIRST-LINE; excellent prognosis",
        "aa": "~340 aa", "kDa": "~38 kDa",
        "mechanism": (
            "PRRT2 encodes proline-rich transmembrane protein 2, a neuronal presynaptic protein expressed "
            "predominantly in the cerebellum, striatum, and cerebral cortex. "
            "NORMAL FUNCTION: PRRT2 interacts with SNAP25 (synaptosomal-associated protein 25 kDa), "
            "a core SNARE complex protein critical for synaptic vesicle fusion. "
            "PRRT2 modulates voltage-gated sodium channel Nav1.2 and Nav1.6 surface expression by "
            "interacting with SNAP25 → dampens neuronal hyperexcitability in motor circuits. "
            "PKD PATHOMECHANISM: PRRT2 haploinsufficiency (frameshift/truncating common: c.649dupC — "
            "hotspot mutation in Chinese populations especially) → "
            "loss of PRRT2-SNAP25 interaction → disinhibition of sodium channel surface expression → "
            "motor circuit hyperexcitability → paroxysmal attacks triggered by SUDDEN MOVEMENT. "
            "KINESIGENIC TRIGGER: proprioceptive/sudden movement stimuli trigger the paroxysmal attacks — "
            "corticospinal-cerebellar circuit → abnormal oscillation → dyskinesia/athetosis/chorea "
            "lasting seconds (not epileptic — EEG normal during attacks). "
            "ALLELIC CONDITIONS: PRRT2 mutations also cause: "
            "BFIS (benign familial infantile seizures) — neonatal/infantile onset; self-limiting seizures; "
            "ICCA syndrome (infantile convulsions + choreoathetosis) — overlap of BFIS + PKD; "
            "Hemiplegic migraine type 4 (HM4) — migraine with aura + hemiplegic attacks."
        ),
        "disease_type": "PKD — Paroxysmal Kinesigenic Dyskinesia (AD PRRT2 haploinsufficiency; attacks triggered by sudden movement; carbamazepine dramatically effective; allelic BFIS, ICCA)",
        "locus": "16p11.2", "omim_gene": 614386, "omim_disease": 128200,
        "inheritance": (
            "AUTOSOMAL DOMINANT — variable penetrance (~60–80% penetrance). "
            "MOST COMMON MUTATION: c.649dupC (p.Arg217Profs*8) — frameshift duplication in cytosine repeat; "
            "HOTSPOT in East Asian populations (found in >80% of Chinese PKD families); "
            "also common in European families. "
            "DE NOVO: new mutations account for ~30% of cases (sporadic PKD). "
            "ALLELIC SPECTRUM: same PRRT2 gene mutations cause different phenotypes depending on family: "
            "BFIS (benign familial infantile seizures), PKD, ICCA (infantile convulsions and choreoathetosis), "
            "hemiplegic migraine — intrafamilial phenotypic variability common. "
            "GENETIC TESTING: PRRT2 sequencing (look for c.649dupC specifically); "
            "family history may show mix of phenotypes."
        ),
        "phenotype": (
            "PAROXYSMAL KINESIGENIC DYSKINESIA (PKD): "
            "ONSET: 5–15 years (childhood/adolescence; male predominance 3.5:1). "
            "ATTACK CHARACTERISTICS — ALL FOUR ARE CARDINAL: "
            "(1) KINESIGENIC TRIGGER: sudden movement (standing up quickly, starting to run, "
            "reaching suddenly) → IMMEDIATELY triggers attack; stimulus-response within 1 second; "
            "(2) DYSTONIA/CHOREA/ATHETOSIS during attack: involuntary movements (dystonic posturing, "
            "choreiform writhing) — usually unilateral or alternating; not epileptic; "
            "(3) DURATION: BRIEF — seconds to 1–2 minutes (usually <1 minute); "
            "(4) FREQUENCY: can be VERY FREQUENT (up to 100 attacks/day — most disabling period is adolescence); "
            "CONSCIOUSNESS: PRESERVED throughout attacks (distinguishes from epileptic seizures); "
            "NO POSTICTAL STATE (distinguishes from epilepsy). "
            "INTERICTAL: completely NORMAL neurological examination between attacks. "
            "EEG: NORMAL during attacks (NO epileptiform discharges — PKD is NOT epilepsy despite 'kinesigenic'). "
            "PROGNOSIS: EXCELLENT — spontaneous improvement in early adulthood; "
            "most patients are attack-free by their 30s without medication; "
            "carbamazepine virtually eliminates attacks. "
            "FAMILY HISTORY: BFIS in infancy (family members may report infantile seizures), "
            "or migraine with aura — ask specifically."
        ),
        "treatment_options": [
            "Carbamazepine / oxcarbazepine: FIRST-LINE — DRAMATIC RESPONSE; "
            "even very low doses (100–200mg carbamazepine or 150–300mg oxcarbazepine twice daily) "
            "virtually eliminate attacks in most PKD patients; "
            "rapid response (within days); well-tolerated; "
            "monitor CBC (rare agranulocytosis) and LFT; HLA-B*15:02 screen before CBZ in Asian patients "
            "(SJS risk); "
            "oxcarbazepine preferred if HLA-B*15:02 positive (lower SJS risk though caution still advised)",
            "Phenytoin / lamotrigine: second-line alternatives if CBZ/OXC not tolerated; "
            "some evidence for PKD; lamotrigine particularly useful if migraine comorbidity",
            "Valproate: second-line; some efficacy; less than CBZ; "
            "avoid in females of childbearing potential",
            "Observation ± reassurance: for very mild/infrequent attacks or before medication decision; "
            "PKD spontaneously remits in early adulthood — discuss natural history with patient/family; "
            "avoid triggers (get up slowly, warn before sudden movements) as a non-pharmacological strategy",
        ],
        "key_ddx": [
            "Paroxysmal non-kinesigenic dyskinesia (PNKD — MR1 gene) — attacks longer (minutes-hours); triggered by caffeine/alcohol NOT movement; valproate > CBZ",
            "Epileptic seizures — EEG abnormal during attacks; consciousness may be impaired; postictal state; triggered by movement is rare for epilepsy",
            "Transient ischemic attacks — older onset; focal neurological deficits; MRI/vascular imaging",
            "Functional movement disorder — inconsistent; responsive to distraction; no family history of BFIS; PRRT2 negative",
            "Hyperekplexia (GLRA1) — exaggerated startle (not movement-triggered dyskinesia); tonic attacks; respond to clonazepam",
        ],
        "onset_range_y": (5.0, 15.0),
        "sex_female_prob": 0.22,
        "eeg_pattern": "Normal (EEG normal during attacks — distinguishes from epilepsy)",
        "movement_type": "Paroxysmal dystonia/chorea/athetosis (seconds; triggered by SUDDEN MOVEMENT; consciousness preserved)",
        "severity_dist": {"Severe": 0.10, "Moderate": 0.50, "Mild": 0.40},
        "seizure_free_rate": 0.75,
        "progression_rate": 0.10,
        "drug_error_rate": 0.20,
        "targeted_therapy_available": True,
        "first_line_drug": "Carbamazepine / Oxcarbazepine (low dose; DRAMATIC response; attacks virtually eliminated)",
        "critical_avoid": "EEG NORMAL during attacks — DO NOT diagnose as epilepsy (PKD is not epilepsy; AEDs work via different mechanism); HLA-B*15:02 screen MANDATORY before carbamazepine in EAST ASIAN patients (Stevens-Johnson syndrome risk); EXCELLENT PROGNOSIS — reassure family; spontaneous remission in 3rd decade; low-dose CBZ is usually sufficient",
    },
    # ── ATP1A3 — Alternating Hemiplegia of Childhood (AHC) ───────────────────
    {
        "gene": "ATP1A3", "protein": "Sodium/Potassium-Transporting ATPase Alpha-3 Subunit (ATP1A3)",
        "alias": "ATP1A3; OMIM gene 182350; 19q13.2; ~1013 aa; AHC — Alternating Hemiplegia of Childhood (OMIM #614820); AD de novo; hemiplegic attacks + oculomotor crises + fixed dystonia; flunarizine first-line; CRISIS PROTOCOL essential",
        "aa": "~1013 aa", "kDa": "~113 kDa",
        "mechanism": (
            "ATP1A3 encodes the alpha-3 subunit of the Na+/K+-ATPase pump, expressed predominantly in neurons "
            "(the neuronal isoform, as opposed to alpha-1 which is ubiquitous). "
            "Na+/K+-ATPase pumps 3 Na+ out and 2 K+ in per ATP cycle → maintains the resting membrane potential "
            "and ionic gradients essential for action potential recovery. "
            "NORMAL FUNCTION: ATP1A3 (alpha-3) is the dominant Na+/K+-ATPase in GABAergic interneurons "
            "and cerebellar Purkinje cells — critical for maintaining ionic homeostasis during sustained "
            "neuronal activity. "
            "AHC PATHOMECHANISM: de novo missense mutations in ATP1A3 (dominant negative mechanism) → "
            "reduced Na+/K+-ATPase pump function → failure to maintain ionic gradients after sustained firing → "
            "membrane depolarisation block ('spreading depression' equivalent) → "
            "alternating hemiplegic attacks (one side then the other — 'alternating') and other paroxysmal events. "
            "HOTSPOT MUTATIONS: p.Asp801Asn (most common AHC, ~30%), p.Glu815Lys (severe), "
            "p.Gly947Arg (severe with cardiac arrhythmia risk). "
            "TRIGGERS: emotional stress, physical exercise, water immersion, fever, anesthesia — "
            "all increase neuronal firing → exceeds pump capacity → attack. "
            "FIXED DEFICITS: repeated attacks cause progressive cerebellar/cortical injury → "
            "cumulative fixed neurological deficits over time. "
            "ALLELIC CONDITIONS: CAPOS syndrome (cerebellar ataxia, areflexia, pes cavus, optic atrophy, "
            "sensorineural hearing loss — ATP1A3 p.Glu818Lys); RDP (rapid-onset dystonia-parkinsonism — "
            "adult onset; different hotspots); HINE (hemiplegic infants, neonatal onset)."
        ),
        "disease_type": "AHC — Alternating Hemiplegia of Childhood (AD de novo ATP1A3; hemiplegic attacks + oculomotor crises + fixed dystonia; flunarizine first-line; CRISIS PROTOCOL)",
        "locus": "19q13.2", "omim_gene": 182350, "omim_disease": 614820,
        "inheritance": (
            "AUTOSOMAL DOMINANT — predominantly DE NOVO mutations (~95% of AHC are de novo). "
            "Familial AHC (inherited) is rare (<5%); parent may be mildly affected. "
            "HOTSPOT MUTATIONS (for AHC specifically): "
            "p.Asp801Asn (most common, ~30–35%); p.Glu815Lys (~20–25%, often severe); "
            "p.Gly947Arg (~5–10%, severe including cardiac involvement). "
            "OTHER ATP1A3 PHENOTYPES (different hotspots): "
            "RDP (p.Arg756His, p.Ile758Ser): adult-onset rapid dystonia-parkinsonism; "
            "CAPOS (p.Glu818Lys): cerebellar ataxia + hearing loss + optic atrophy; "
            "HINE: severe neonatal onset. "
            "GENETIC TESTING: ATP1A3 targeted sequencing; panel if unsure (includes CACNA1A — hemiplegic migraine)."
        ),
        "phenotype": (
            "ALTERNATING HEMIPLEGIA OF CHILDHOOD (AHC): "
            "ONSET: First 18 months of life (usually <6 months). "
            "PAROXYSMAL EVENTS (ATTACKS): "
            "(1) HEMIPLEGIC ATTACKS: unilateral flaccid or spastic weakness of one side (arm + leg ± face); "
            "ALTERNATES between left and right side (and bilateral/quadriplegic attacks also occur) — "
            "ALTERNATING is PATHOGNOMONIC; duration hours to days; "
            "(2) OCULOMOTOR CRISES: tonic deviation of eyes ± nystagmus — "
            "often precede or accompany hemiplegic attacks; "
            "(3) DYSTONIC ATTACKS: sustained abnormal posturing — hemidystonia or generalised; "
            "(4) AUTONOMIC FEATURES during attack: skin colour change (pallor/flushing), pupillary change, "
            "hyperhidrosis; "
            "(5) EPILEPTIC SEIZURES: generalised/focal seizures occur in 50–60% of AHC patients. "
            "SLEEP RESOLVES ATTACKS (KEY): all hemiplegic attacks RESOLVE WITH SLEEP (may return on waking); "
            "this is PATHOGNOMONIC — sleep terminates AHC attacks. "
            "TRIGGERS: water immersion (bathing), emotional stress, exercise, fever — "
            "specific triggers identified for each patient (family diary). "
            "FIXED DEFICITS (PROGRESSIVE): "
            "cerebellar ataxia — progressive; dysarthria; mild-to-moderate intellectual disability (most); "
            "behavioural problems; cumulative cognitive decline with frequent severe attacks. "
            "CRISIS: prolonged bilateral attack — EMERGENCY; may need hospital admission."
        ),
        "treatment_options": [
            "Flunarizine: FIRST-LINE prophylactic treatment for AHC; "
            "calcium channel blocker (F-type and T-type); reduces attack frequency and severity; "
            "modest effect size (no cure); dose: 5–10mg/day in children; side effects: weight gain, sedation, "
            "extrapyramidal effects (rare); evidence from open-label series (no large RCT for AHC); "
            "start at lowest dose; assess response over 3 months",
            "Benzodiazepines (diazepam, midazolam) DURING ATTACKS: "
            "for acute management of prolonged hemiplegic or dystonic attacks; "
            "buccal/intranasal midazolam at home — family-administered rescue for prolonged attacks; "
            "reduces attack duration; CRISIS PROTOCOL: family trained in rescue administration",
            "ATP (adenosine triphosphate) — anecdotal: some families report attacks abort with exertion/ATP; "
            "theoretical basis: intracellular ATP supports Na+/K+-ATPase; not proven in trials",
            "Antiepileptics for seizures: valproate, levetiracetam, clobazam — standard management of "
            "co-existing epilepsy in AHC; note epilepsy is separate from hemiplegic attacks (both need treatment)",
            "Trigger AVOIDANCE (critical): identify patient-specific triggers (water immersion — "
            "avoid baths; use shower only; emotional stress management; fever management with antipyretics); "
            "family diary essential to identify triggers; water trigger avoidance is MANDATORY "
            "(bathing is a major and consistent trigger — use shower, never immerse child in bath unsupported)",
            "Supportive / multidisciplinary: physiotherapy (during and between attacks); "
            "speech-language pathology; educational support; neuropsychology; "
            "AHC parent support group (AHC Association); "
            "medical alert ID band ('AHC — avoid water immersion'); "
            "crisis hospital protocol card (kept with family); anaesthesia alert (trigger risk)",
        ],
        "key_ddx": [
            "Hemiplegic migraine type 1/2/3 (CACNA1A/ATP1A2/SCN1A) — older onset; severe headache; MRI changes (TIA-like); usually no oculomotor crises; ATP1A3 negative",
            "Epileptic hemiplegia (Todd's paresis post-seizure) — follows ictal event; EEG changes; unilateral; resolves within hours; no alternating",
            "Focal status epilepticus (Rasmussen) — progressive unilateral; MRI cerebral hemiatrophy; continuous EEG changes",
            "Transient ischemic attacks (TIA) — rare in children; vascular risk factors; DWI-MRI positive; not alternating",
            "Alternating hemiplegia (other causes: metabolic, vascular) — exclude MELAS, PDH deficiency (lactate), structural lesions; ATP1A3 mutation differentiates",
        ],
        "onset_range_y": (0.0, 1.5),
        "sex_female_prob": 0.50,
        "eeg_pattern": "Normal during hemiplegic attacks (unlike epileptic hemiplegia) / Epileptiform in 50% (co-existing epilepsy)",
        "movement_type": "Hemiplegic attacks (ALTERNATING sides — PATHOGNOMONIC) + Oculomotor crises + Fixed dystonia",
        "severity_dist": {"Severe": 0.45, "Moderate": 0.40, "Mild": 0.15},
        "seizure_free_rate": 0.40,
        "progression_rate": 0.70,
        "drug_error_rate": 0.15,
        "targeted_therapy_available": True,
        "first_line_drug": "Flunarizine (prophylaxis) / Buccal midazolam (acute crisis) / Trigger avoidance (WATER IMMERSION — MANDATORY)",
        "critical_avoid": "WATER IMMERSION (bathing) is a MAJOR TRIGGER — shower ONLY, never immerse; SLEEP TERMINATES ATTACKS (PATHOGNOMONIC) — observe; hemiplegic attacks resolve with sleep (unlike epileptic hemiplegia); CRISIS PROTOCOL with buccal midazolam prescribed to ALL families; anaesthesia alert (triggers attacks); AVOID diagnosing as epilepsy alone (AHC attacks are NOT ictal)",
    },
]


# ── Patient generator ───────────────────────────────────────────────────────

def _gen_patients(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_data["gene"]
    patients = []
    onset_lo, onset_hi = gene_data["onset_range_y"]

    for i in range(40):
        onset = round(rng.uniform(onset_lo, onset_hi), 2)

        # Severity
        r = rng.random()
        cumulative = 0.0
        sev = "Severe"
        for label, prob in gene_data["severity_dist"].items():
            cumulative += prob
            if r < cumulative:
                sev = label
                break

        # Sex
        sex = "F" if rng.random() < gene_data["sex_female_prob"] else "M"

        # Clinical booleans
        seizure_free  = rng.random() < gene_data["seizure_free_rate"]
        drug_error    = rng.random() < gene_data["drug_error_rate"]
        on_targeted   = gene_data["targeted_therapy_available"] and rng.random() < 0.55
        progressing   = rng.random() < gene_data["progression_rate"]
        cognitive_imp = sev in ("Severe", "Moderate") and rng.random() < (
            0.95 if gene in ("HTT", "PANK2") else
            0.80 if gene in ("VPS13A", "ATP7B") else
            0.60 if gene in ("ATP1A3",) else
            0.30 if gene in ("TOR1A", "GCH1") else 0.20
        )

        # Movement type (primary descriptor)
        mv = gene_data["movement_type"].split(" / ")[0] if sev == "Severe" else gene_data["movement_type"].split(" / ")[-1]

        # Treatment
        fl = gene_data["first_line_drug"].split(" / ")[0]
        if on_targeted:
            tx = fl + " (targeted/disease-specific)"
        elif drug_error:
            tx = "CONTRAINDICATED or incorrect drug prescribed (error detected)"
        else:
            tx = fl + (" + adjunctive" if rng.random() < 0.55 else "")

        age_at_dx = round(min(onset + rng.uniform(0.3, 3.0), onset_hi + 5.0), 2)

        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "onset_age_y": onset,
            "diagnosis_age_y": age_at_dx,
            "sex": sex,
            "severity": sev,
            "movement_type": mv,
            "seizure_free": seizure_free,
            "drug_avoid_prescribed_error": drug_error,
            "on_targeted_therapy": on_targeted,
            "disease_progression": progressing,
            "cognitive_impairment": cognitive_imp,
            "treatment": tx,
            "first_line_drug": gene_data["first_line_drug"],
            "critical_avoid": gene_data["critical_avoid"],
        })
    return patients


def _gen_cohort() -> list:
    all_pts = []
    for idx, gene_data in enumerate(MD_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients(gene_data, seed))
    return all_pts


# ── Public API ──────────────────────────────────────────────────────────────

def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] += 1

    seizfree_n  = sum(1 for p in patients if p["seizure_free"])
    prog_n      = sum(1 for p in patients if p["disease_progression"])
    cog_n       = sum(1 for p in patients if p["cognitive_impairment"])
    targeted_n  = sum(1 for p in patients if p["on_targeted_therapy"])
    drug_err_n  = sum(1 for p in patients if p["drug_avoid_prescribed_error"])

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 2)
    mean_dx_age = round(sum(p["diagnosis_age_y"] for p in patients) / n, 2)

    return {
        "atlas": "Movement-Disorders-Atlas",
        "full_name": "Complete 8-Gene Hereditary Movement Disorders Atlas",
        "subtitle": (
            "HTT·ATP7B·TOR1A·PANK2·VPS13A·GCH1·PRRT2·ATP1A3 — "
            "320 patients (8×40, seeds 1078–1085)"
        ),
        "description": (
            "Comprehensive atlas of 8 major genetic hereditary movement disorders encompassing: "
            "HUNTINGTON DISEASE (HTT — CAG repeat ≥40; AD; striatal neurodegeneration; chorea + dementia + psychiatric; "
            "tetrabenazine/deutetrabenazine for chorea; NO disease-modifying therapy approved; "
            "SCREEN depression before TBZ; predictive testing requires genetic counselling); "
            "WILSON DISEASE (ATP7B — AR biallelic; copper accumulation liver+brain+cornea; "
            "KF RINGS PATHOGNOMONIC in neurological Wilson; wing-beating tremor; "
            "TREATABLE: trientine chelation + zinc; liver transplant for ALF — EMERGENCY; "
            "SCREEN ALL SIBLINGS — 25% risk; penicillamine WORSENS neurological Wilson in 20-50%); "
            "DYT-TOR1A (TOR1A — GAG deletion; AD 30% penetrance; generalised dystonia limb-onset; "
            "MANDATORY levodopa trial to exclude GCH1; GPi-DBS highly effective — 50-80% improvement); "
            "PKAN/NBIA1 (PANK2 — AR; eye-of-the-tiger MRI PATHOGNOMONIC; pigmentary retinopathy; "
            "deferiprone iron chelation — WEEKLY CBC; rapidly progressive); "
            "CHOREA-ACANTHOCYTOSIS (VPS13A — AR; orofacial dystonia + SELF-MUTILATION PATHOGNOMONIC; "
            "acanthocytes FRESH SMEAR; chorein Western blot confirmatory; seizures 30-50%); "
            "DOPA-RESPONSIVE DYSTONIA (GCH1 — AD; DIURNAL VARIATION PATHOGNOMONIC; "
            "levodopa CURATIVE at low dose — lifelong; female predominance 3:1; "
            "MOST COMMONLY MISDIAGNOSED AS CEREBRAL PALSY); "
            "PAROXYSMAL KINESIGENIC DYSKINESIA (PRRT2 — AD; attacks triggered by SUDDEN MOVEMENT; "
            "EEG NORMAL during attacks; carbamazepine DRAMATICALLY EFFECTIVE low dose; excellent prognosis; "
            "HLA-B*15:02 screen before CBZ in East Asian patients); "
            "ALTERNATING HEMIPLEGIA OF CHILDHOOD (ATP1A3 — AD de novo; hemiplegic attacks ALTERNATE SIDES; "
            "SLEEP TERMINATES ATTACKS; water immersion TRIGGER — SHOWER ONLY; "
            "flunarizine prophylaxis; buccal midazolam rescue — CRISIS PROTOCOL)."
        ),
        "total_patients": n,
        "genes_covered": len(MD_GENES),
        "patients_per_gene": 40,
        "seed_range": "1078–1085",
        "gene_list": [g["gene"] for g in MD_GENES],
        "disease_category_breakdown": {
            "Huntington Disease (AD CAG ≥40 repeat; HTT; chorea+dementia+psychiatric; no DMT approved; TBZ/DTBZ for chorea)": ["HTT"],
            "Wilson Disease (AR ATP7B; copper accumulation; KF rings; wing-beating tremor; TREATABLE — trientine+zinc)": ["ATP7B"],
            "DYT-TOR1A Generalised Dystonia (AD 30% penetrance; GAG deletion; limb-onset; GPi-DBS highly effective)": ["TOR1A"],
            "PKAN/NBIA1 (AR PANK2; eye-of-the-tiger MRI PATHOGNOMONIC; pigmentary retinopathy; iron accumulation)": ["PANK2"],
            "Chorea-Acanthocytosis (AR VPS13A; orofacial self-mutilation PATHOGNOMONIC; acanthocytes; caudate atrophy)": ["VPS13A"],
            "Dopa-Responsive Dystonia DRD (AD GCH1; DIURNAL VARIATION PATHOGNOMONIC; levodopa CURATIVE; female 3:1)": ["GCH1"],
            "Paroxysmal Kinesigenic Dyskinesia PKD (AD PRRT2; movement-triggered attacks; CBZ dramatically effective; PRRT2)": ["PRRT2"],
            "Alternating Hemiplegia of Childhood AHC (AD de novo ATP1A3; alternating hemiplegia; sleep terminates; water trigger)": ["ATP1A3"],
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "mean_diagnosis_age_y": mean_dx_age,
        "kpis": [
            {"label": "Total Patients", "value": n, "color": "#1a237e"},
            {"label": "Genes Covered", "value": len(MD_GENES), "color": "#2e7d32"},
            {"label": "Patients/Gene", "value": 40, "color": "#6a1b9a"},
            {"label": "Progression Rate", "value": f"{round(100 * prog_n / n, 1)}%", "color": "#b71c1c"},
            {"label": "Mean Onset (y)", "value": mean_onset, "color": "#e65100"},
            {"label": "Seeds", "value": "1078–1085", "color": "#37474f"},
        ],
        "clinical_features_prevalence": {
            "Disease Progression": round(100 * prog_n / n, 1),
            "Cognitive Impairment": round(100 * cog_n / n, 1),
            "On Targeted Therapy": round(100 * targeted_n / n, 1),
            "Drug-Prescribing Error Detected": round(100 * drug_err_n / n, 1),
            "Seizure-Free (current)": round(100 * seizfree_n / n, 1),
        },
        "drug_alerts": [
            "HUNTINGTON (HTT): SCREEN for depression/suicidality BEFORE starting tetrabenazine or deutetrabenazine "
            "(VMAT2 inhibitors worsen depression — HD already has high psychiatric morbidity); "
            "NO disease-modifying therapy approved (ASOs failed Phase 3 2021); "
            "predictive testing REQUIRES genetic counselling protocol — never test minors",
            "WILSON DISEASE (ATP7B): D-PENICILLAMINE causes paradoxical NEUROLOGICAL WORSENING in 20-50% of "
            "neurological Wilson (mobilises brain copper) — use TRIENTINE as first-line; "
            "Wilson's ALF is a TRANSPLANT EMERGENCY — MELD >25 → urgent liver transplant evaluation; "
            "KF rings on slit-lamp MANDATORY in any young-onset movement disorder workup; "
            "SCREEN ALL SIBLINGS (25% risk — ceruloplasmin + slit-lamp + urine copper)",
            "DYT-TOR1A / GCH1: LEVODOPA TRIAL IS MANDATORY in ALL early-onset dystonia — "
            "GCH1 dopa-responsive dystonia (DRD) has DRAMATIC levodopa response and is CURATIVE; "
            "missing GCH1 = preventable years of disability; "
            "DIURNAL VARIATION (worse in afternoon, better after sleep) = GCH1 until excluded",
            "PKAN (PANK2): EYE-OF-THE-TIGER MRI sign is PATHOGNOMONIC — bilateral GPi T2 hypointensity + "
            "central hyperintense core; MANDATORY OPHTHALMOLOGY (pigmentary retinopathy in ~65%); "
            "DEFERIPRONE: WEEKLY CBC mandatory (agranulocytosis risk); "
            "SCREEN SIBLINGS (25% risk — AR disorder)",
            "CHOREA-ACANTHOCYTOSIS (VPS13A): FRESH BLOOD SMEAR for acanthocytes (EDTA causes pseudoacanthocytes "
            "→ false negative); CHOREIN Western blot from RBCs is rapid confirmatory test; "
            "MOUTH GUARDS MANDATORY for lip/tongue self-mutilation; "
            "SEIZURES in 30-50% — evaluate and treat",
            "PRRT2 (PKD): HLA-B*15:02 SCREEN MANDATORY before carbamazepine in EAST ASIAN patients "
            "(Stevens-Johnson syndrome risk); EEG NORMAL during attacks — NOT epilepsy; "
            "low-dose CBZ is sufficient; EXCELLENT PROGNOSIS — spontaneous remission by 3rd decade",
            "ATP1A3 (AHC): WATER IMMERSION is MAJOR TRIGGER — shower only, NEVER immerse child in bath; "
            "SLEEP TERMINATES ATTACKS (diagnostic); CRISIS PROTOCOL — buccal midazolam prescribed to ALL families; "
            "ANAESTHESIA ALERT card (surgical triggers); AHC attacks are NOT ictal — do not diagnose as epilepsy alone",
        ],
        "diagnostic_pearls": [
            "HUNTINGTON (HTT): choreic movements + psychiatric symptoms + family history of chorea/dementia → "
            "CAG repeat testing; JUVENILE HD (>55 CAG) presents as RIGIDITY not chorea; "
            "anticipation in paternal transmission",
            "WILSON DISEASE (ATP7B): any young-onset movement disorder (dystonia/tremor/parkinsonism) → "
            "ALWAYS check: slit-lamp (KF rings), ceruloplasmin, 24h urine copper; "
            "WING-BEATING TREMOR (proximal arm, elbows abducted) is PATHOGNOMONIC; "
            "PANDA SIGN on MRI (midbrain T2 — face of panda) highly suggestive",
            "TOR1A (DYT1): CHILDHOOD LIMB-ONSET DYSTONIA (gait dystonia, foot inversion) → "
            "levodopa trial MANDATORY; if no response → DYT1 testing; "
            "30% penetrance — carrier parent may be unaffected",
            "PANK2 (PKAN): CHILDHOOD DYSTONIA + eye-of-the-tiger on MRI → PANK2 immediately; "
            "pigmentary retinopathy present in most — ophthalmology referral",
            "VPS13A (ChAc): OROFACIAL DYSTONIA + SELF-MUTILATION (lip/tongue biting) + CHOREA → "
            "fresh blood smear (acanthocytes) + VPS13A/chorein; NOT EDTA sample; "
            "CK elevated (myopathy component)",
            "GCH1 (DRD): CHILD WHO IS NORMAL IN THE MORNING BUT LIMPS BY EVENING → DRD until excluded; "
            "levodopa trial first (dramatic response = diagnostic); "
            "female 3:1 (sex-penetrance); don't label as cerebral palsy without levodopa trial",
            "PRRT2 (PKD): ATTACKS TRIGGERED BY SUDDEN MOVEMENT (getting up, starting to run) → PRRT2; "
            "EEG normal during attack; carbamazepine low dose eliminates attacks; "
            "family history of infantile seizures (BFIS) or hemiplegic migraine → PRRT2",
            "ATP1A3 (AHC): INFANT with HEMIPLEGIC ATTACKS alternating sides + resolves with SLEEP → "
            "ATP1A3; oculomotor crises co-occurring; water bath triggers; "
            "flunarizine prophylaxis; crisis plan with buccal midazolam",
        ],
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    breakdown = {}
    for gene_data in MD_GENES:
        gene = gene_data["gene"]
        gene_pts = [p for p in patients if p["gene"] == gene]
        n = len(gene_pts)
        sev = {s: sum(1 for p in gene_pts if p["severity"] == s) for s in ("Mild", "Moderate", "Severe")}
        f_n = sum(1 for p in gene_pts if p["sex"] == "F")

        breakdown[gene] = {
            "gene": gene,
            "protein": gene_data["protein"],
            "alias": gene_data["alias"],
            "n_patients": n,
            "disease_type": gene_data["disease_type"],
            "locus": gene_data["locus"],
            "omim_gene": gene_data["omim_gene"],
            "omim_disease": gene_data["omim_disease"],
            "inheritance": gene_data["inheritance"],
            "phenotype": gene_data["phenotype"],
            "treatment_options": gene_data["treatment_options"],
            "key_ddx": gene_data["key_ddx"],
            "mechanism": gene_data["mechanism"],
            "movement_type": gene_data["movement_type"],
            "first_line_drug": gene_data["first_line_drug"],
            "critical_avoid": gene_data["critical_avoid"],
            "severity_distribution": {
                "mild_pct": round(100 * sev["Mild"] / n, 1),
                "moderate_pct": round(100 * sev["Moderate"] / n, 1),
                "severe_pct": round(100 * sev["Severe"] / n, 1),
            },
            "mean_onset_age_y": round(sum(p["onset_age_y"] for p in gene_pts) / n, 2),
            "mean_diagnosis_age_y": round(sum(p["diagnosis_age_y"] for p in gene_pts) / n, 2),
            "sex_pct_female": round(100 * f_n / n, 1),
            "seizure_free_pct": round(100 * sum(1 for p in gene_pts if p["seizure_free"]) / n, 1),
            "drug_error_pct": round(100 * sum(1 for p in gene_pts if p["drug_avoid_prescribed_error"]) / n, 1),
            "cognitive_impairment_pct": round(100 * sum(1 for p in gene_pts if p["cognitive_impairment"]) / n, 1),
            "on_targeted_therapy_pct": round(100 * sum(1 for p in gene_pts if p["on_targeted_therapy"]) / n, 1),
            "progression_pct": round(100 * sum(1 for p in gene_pts if p["disease_progression"]) / n, 1),
        }
    return {
        "atlas": "Movement-Disorders-Atlas",
        "subtitle": "Per-gene clinical breakdown — 320 patients (8×40, seeds 1078–1085)",
        "genes": breakdown,
        "gene_order": [g["gene"] for g in MD_GENES],
    }


def get_definitions() -> dict:
    return {
        "atlas": "Movement-Disorders-Atlas",
        "subtitle": "Clinical and genetic terminology definitions for Movement Disorders Atlas",
        "definitions": {
            "Chorea": (
                "Involuntary, irregular, unpredictable, brief muscle contractions producing a flowing, "
                "dance-like movement pattern (from Greek 'choreia' = dance). "
                "Characteristic of Huntington disease (HTT) and Chorea-Acanthocytosis (VPS13A). "
                "Mechanism: loss of indirect pathway MSNs (D2R-expressing) in striatum → "
                "thalamic disinhibition → cortical hyperactivation → involuntary movements. "
                "Treatment: VMAT2 inhibitors (tetrabenazine, deutetrabenazine)."
            ),
            "Dystonia": (
                "Sustained or intermittent muscle contractions causing abnormal, often repetitive, "
                "movements or postures. Classified by: age of onset, body distribution (focal/segmental/"
                "generalised), aetiology (primary/secondary). "
                "Genes causing dystonia: TOR1A (DYT1 generalised), GCH1 (DRD), ATP7B (Wilson's), "
                "PANK2 (PKAN), ATP1A3 (AHC). "
                "Generalised dystonia treatment: levodopa trial FIRST (excludes GCH1), then "
                "trihexyphenidyl, then GPi-DBS."
            ),
            "Paroxysmal Movement Disorder": (
                "Movement disorder occurring in discrete episodic attacks with return to normal between attacks. "
                "Types: kinesigenic (PRRT2-PKD — triggered by sudden movement), non-kinesigenic "
                "(PNKD — MR1 gene; triggered by caffeine/alcohol), exertional (PED), hypnogenic. "
                "PKD (PRRT2): seconds, frequent, carbamazepine curative. "
                "EEG normal during PKD attacks — not epilepsy."
            ),
            "CAG Trinucleotide Repeat Expansion": (
                "DNA repeat where cytosine-adenine-guanine is expanded beyond normal range. "
                "HTT (Huntington): normal ≤26 CAG; intermediate 27–35 (risk of expansion); "
                "reduced penetrance 36–39; full penetrance ≥40. "
                "Repeat length inversely correlates with age of onset (~3-4 years per additional repeat). "
                "Anticipation: expansions more common in paternal transmission (CAG instability in sperm)."
            ),
            "Kayser-Fleischer (KF) Rings": (
                "Golden-brown rings visible at the outer margin of the cornea (Descemet membrane) on slit-lamp examination. "
                "PATHOGNOMONIC of Wilson disease (ATP7B) in the context of neurological symptoms. "
                "Present: >95% of neurological Wilson; 50-60% hepatic Wilson. "
                "Mechanism: copper deposition in posterior corneal stroma (Descemet membrane). "
                "Visualisation: REQUIRES slit-lamp examination — invisible to naked eye in early stages. "
                "Disappear with successful copper chelation therapy (confirms treatment response)."
            ),
            "Eye-of-the-Tiger Sign": (
                "MRI (T2-weighted) finding in globus pallidus: bilateral central T2-HYPERINTENSITY "
                "(gliosis/vacuolisation) surrounded by T2-HYPOINTENSITY (iron deposition). "
                "PATHOGNOMONIC for PKAN (PANK2 mutations) — >95% specificity. "
                "Iron deposits cause signal void on T2; gliotic core gives central bright signal. "
                "Seen in BOTH classic and atypical PKAN. "
                "MRI must include T2-weighted or SWI (susceptibility-weighted) for detection."
            ),
            "Acanthocytes": (
                "Spiky red blood cells with irregular, variably-spaced projections (from Greek 'akantha' = thorn). "
                "Seen in: Chorea-Acanthocytosis (VPS13A), McLeod syndrome (XK), "
                "abetalipoproteinaemia, and other lipid disorders. "
                "IMPORTANT: FRESH BLOOD SMEAR required (EDTA anticoagulant causes pseudoacanthocytes). "
                "CHOREIN (VPS13A protein) Western blot from RBC lysate: near-absent in ChAc = confirmatory test. "
                "Mechanism: VPS13A membrane lipid transfer deficiency → abnormal RBC membrane phospholipid composition."
            ),
            "Diurnal Variation": (
                "Cyclical change in symptoms related to time of day. "
                "PATHOGNOMONIC of GCH1 dopa-responsive dystonia (DRD): "
                "symptoms WORSE in the afternoon/evening (dopamine depleted by activity); "
                "BETTER after rest/sleep (dopamine replenished overnight). "
                "Child who walks normally in the morning but limps by evening = DRD until proven otherwise. "
                "Mechanism: GCH1 haploinsufficiency → reduced BH4 → reduced dopamine synthesis → "
                "dopamine depot depleted faster than it is resynthesised during waking hours."
            ),
            "Kinesigenic Trigger (PKD)": (
                "Sudden movement as the specific trigger for paroxysmal dyskinesia attacks in PRRT2-PKD. "
                "Onset: within 1 second of the trigger movement (standing up quickly, starting to run, "
                "sudden reach). Attack: dystonia/chorea lasting seconds; consciousness PRESERVED. "
                "EEG: NORMAL during attacks (NOT epileptic). "
                "Treatment: carbamazepine dramatically eliminates attacks even at low doses. "
                "Distinguishes from: non-kinesigenic dyskinesia (caffeine/alcohol trigger, minutes-hours duration), "
                "exertional dyskinesia (prolonged exercise trigger)."
            ),
            "Wing-Beating Tremor": (
                "Coarse, low-frequency tremor of the proximal arms held with elbows abducted (outstretched) "
                "— resembling wings of a bird. "
                "PATHOGNOMONIC of Wilson disease (ATP7B). "
                "Mechanism: copper-induced degeneration of basal ganglia and cerebellar outflow tracts → "
                "combined postural + intention tremor predominantly proximal. "
                "Confirmed by: KF rings (slit-lamp), low ceruloplasmin, elevated 24h urine copper."
            ),
            "Alternating Hemiplegia (AHC)": (
                "Episodic unilateral weakness (arm + leg ± face) that ALTERNATES between left and right sides "
                "within a single patient. PATHOGNOMONIC of ATP1A3 mutations (AHC). "
                "SLEEP TERMINATES ATTACKS — all AHC hemiplegic attacks resolve with sleep "
                "(distinguishes from epileptic hemiplegia/Todd's paresis where sleep is not reliably curative). "
                "WATER IMMERSION is a MAJOR TRIGGER — shower only, never immerse child in bath. "
                "EEG: NORMAL during hemiplegic attacks (attacks are not ictal)."
            ),
            "GPi-DBS (Globus Pallidus Internus Deep Brain Stimulation)": (
                "Neurosurgical procedure placing bilateral stimulating electrodes in the globus pallidus internus (GPi). "
                "MECHANISM: high-frequency stimulation modulates (inhibits/disrupts) the overactive GPi → "
                "reduces dystonic output to thalamus and cortex. "
                "BEST RESULTS in: DYT-TOR1A (DYT1) generalised dystonia — 50-80% BFMDRS improvement; "
                "DYT-KMT2B. "
                "DELAYED EFFECT in dystonia: improvement continues over 6-12 months (unlike Parkinson's DBS). "
                "INDICATION: generalised dystonia with significant disability, ≥2 medications failed. "
                "MRI-conditional: specific programming required if MRI brain needed post-implant."
            ),
            "Tetrabenazine (TBZ) and Deutetrabenazine": (
                "VMAT2 (vesicular monoamine transporter 2) inhibitors — deplete presynaptic monoamines "
                "(dopamine, serotonin, norepinephrine) by blocking storage in synaptic vesicles. "
                "TBZ: FDA approved for HD chorea; RISK: depression/suicidality (screen before prescribing — "
                "HD has high baseline psychiatric morbidity); Parkinsonism; sedation; "
                "CYP2D6 pharmacogenomics affects dosing. "
                "DEUTETRABENAZINE (AUSTEDO): deuterium-modified TBZ; slower metabolism; twice daily; "
                "possibly fewer psychiatric side effects; FDA approved 2017 for HD chorea."
            ),
            "Chorein (VPS13A protein)": (
                "The protein product of VPS13A — a large lipid-transfer protein. "
                "CHOREIN WESTERN BLOT: RBC lysate shows near-absent chorein in Chorea-Acanthocytosis (VPS13A). "
                "RAPID CONFIRMATORY TEST — available before genetic result; "
                "positive result (absent chorein) confirms ChAc diagnosis. "
                "Commercially available antibodies; requires fresh/frozen RBC pellet from blood. "
                "Negative (absent) chorein + acanthocytes on fresh smear = ChAc diagnosis confirmed."
            ),
            "Pantothenate Kinase (PANK2)": (
                "Rate-limiting mitochondrial enzyme of coenzyme A (CoA) biosynthesis. "
                "Phosphorylates pantothenate (vitamin B5) → 4'-phosphopantothenate → CoA. "
                "PKAN: biallelic PANK2 loss → CoA deficiency in GPi neurons → cysteine accumulation → "
                "iron-catalysed oxidative damage → GPi neurodegeneration + iron accumulation. "
                "Deferiprone (iron chelator) and pantethine (CoA pathway bypass) are current therapeutic approaches. "
                "No approved disease-modifying therapy; management is symptomatic."
            ),
            "BH4 (Tetrahydrobiopterin)": (
                "Essential cofactor for: tyrosine hydroxylase (TH — dopamine synthesis), "
                "tryptophan hydroxylase (TpH — serotonin synthesis), phenylalanine hydroxylase (PAH). "
                "GCH1 produces BH4 (rate-limiting step). GCH1 haploinsufficiency → reduced BH4 → "
                "reduced TH activity → reduced dopamine in striatum → dystonia/parkinsonism. "
                "GCH1-DRD TREATMENT: levodopa (bypasses TH step — provides exogenous dopamine precursor) "
                "is curative at low doses. "
                "AR GCH1: hyperphenylalaninaemia + severe DRD + serotonin deficiency → sapropterin + levodopa + 5-HTP."
            ),
        },
    }
