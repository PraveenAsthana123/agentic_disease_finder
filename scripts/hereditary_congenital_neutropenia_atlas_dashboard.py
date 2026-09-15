"""Hereditary Congenital Neutropenia Atlas — 8-Gene Reference
ELANE-HAX1-G6PC3-WAS-CXCR4-GFI1-VPS45-JAGN1
Severe Congenital Neutropenia / Cyclic Neutropenia / WHIM / Kostmann / WAS Spectrum
320 patients (8 x 40), seeds 2822-2829.
Endpoints: /api/hereditary-congenital-neutropenia-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "ELANE",
        "protein": (
            "ELANE -- 19p13.3 AD -- 267aa -- Neutrophil-Elastase-29kDa-"
            "Serine-Protease-Azurophil-Granules-Misfolded-UPR-Apoptosis-"
            "OMIM-Gene-130130-Disease-SCN1-202700-CN1-162800"
        ),
        "locus": "19p13.3",
        "protein_size": "267 aa / 29 kDa (serine protease; azurophil (primary) granule component; degrades ECM proteins; mutations cause misfolding → ER stress → unfolded protein response → myeloid apoptosis; most common SCN gene; also cyclic neutropenia when oscillator disrupted)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (gain-of-toxic-function / dominant negative) — most common SCN/CN gene (~50% of SCN, ~90% of cyclic neutropenia); "
            "MECHANISM — SCN vs CN distinction critical: "
            "  SCN (severe congenital neutropenia): missense/truncation → misfolded NE protein → "
            "    ER stress → unfolded protein response (UPR) → CHOP/DDIT3 induction → myeloid progenitor apoptosis → "
            "    maturation arrest at promyelocyte/myelocyte stage; "
            "  CYCLIC NEUTROPENIA (CN1): different ELANE mutations → dysregulation of haematopoietic oscillator → "
            "    periodic ANC nadir every 21 days (±3 days) → neutropenia lasting 3-6 days per cycle; "
            "  Both caused by ELANE mutations but DIFFERENT mutation classes → completely different clinical phenotypes; "
            "G-CSF RESPONSE: "
            "  SCN: usually responds to G-CSF (5-20 mcg/kg/day) — ANC normalisation in 90-95%; "
            "  CN: also responds but cyclicity persists (nadir reduced, not eliminated); "
            "AML/MDS TRANSFORMATION RISK: "
            "  SCN 15-25% lifetime (higher with high G-CSF doses, lower with bone marrow surveillance); "
            "  CN 1-3% lifetime (much lower — CN not considered pre-malignant syndrome); "
            "MUTATION TYPES: missense >50%, nonsense, frameshift; de novo ~75% of SCN-ELANE; "
            "INHERITANCE: AD but de novo rate high; familial recurrence in siblings counselled as 50% risk if parental ELANE identified; "
            "CURATIVE TREATMENT: allogeneic HSCT (SCN non-responders to G-CSF / AML transformation / high-risk features)"
        ),
        "disease_category": (
            "SEVERE CONGENITAL NEUTROPENIA TYPE 1 (SCN1) — OMIM 202700; "
            "CYCLIC NEUTROPENIA TYPE 1 (CN1) — OMIM 162800; "
            "HAEMATOLOGICAL FEATURES: "
            "  ANC at nadir: <0.2 × 10⁹/L (SCN); 0.0-0.5 × 10⁹/L at nadir (CN, every 21 days); "
            "  Maturation arrest: promyelocyte stage (BM) in SCN — DIAGNOSTIC on aspirate; "
            "  CBC otherwise relatively preserved (Hb/platelets usually normal); "
            "CLINICAL PRESENTATION (SCN): "
            "  Onset: birth / first weeks of life; "
            "  Recurrent bacterial infections: skin/soft tissue infections, pneumonia, perirectal abscess; "
            "  Omphalitis (neonatal umbilical infection) — should trigger workup; "
            "  Oral ulcers, gingivitis, severe periodontal disease (gingival/alveolar bone destruction); "
            "  Absent pus formation (no neutrophils to form pus) — DISTINCTIVE; "
            "CYCLIC NEUTROPENIA (CN1): "
            "  Predictable 21-day cycle ± 3 days; "
            "  Oral ulcers + fever + cervical lymphadenopathy at each nadir (oral aphthae classic); "
            "  Generally milder than SCN; AML risk much lower; "
            "  BONE MARROW: oscillates with blood — promyelocyte arrest AT nadir only; "
            "DIAGNOSIS: "
            "  3× weekly CBC for 6-8 weeks to document periodicity (CN) or persistent neutropenia (SCN); "
            "  BM biopsy + aspirate: maturation arrest at promyelocyte (SCN) or cycling (CN); "
            "  ELANE sequencing (gene panel); G-CSF stimulation test"
        ),
        "disease_pathway": (
            "MISFOLDED NEUTROPHIL ELASTASE / ER STRESS / UPR PATHWAY (SCN): "
            "NORMAL NE TRAFFICKING: "
            "  ELANE mRNA → NE pre-protein → ER → correct folding → Golgi → azurophil granule packaging → "
            "  → mature neutrophil granule release; "
            "MUTANT NE MISFOLDING: "
            "  Missense mutation → misfolded NE protein accumulates in ER → "
            "  → ER STRESS → UNFOLDED PROTEIN RESPONSE (UPR) activation: "
            "    • IRE1α branch → XBP1 splicing → ERAD upregulation; "
            "    • PERK branch → eIF2α phosphorylation → ATF4 → CHOP (DDIT3); "
            "    • ATF6 branch → BiP/GRP78 upregulation; "
            "  CHOP INDUCTION → pro-apoptotic transcription → myeloid progenitor apoptosis; "
            "  MATURATION ARREST: promyelocytes (stage where NE first expressed) → arrested, cannot mature to bands/segs; "
            "HAEMATOPOIETIC OSCILLATOR (CN): "
            "  Different ELANE mutations → destabilise myeloid/haematopoietic oscillator (G-CSF feedback loop); "
            "  Period = 21 days (same as normal haematopoietic cell cycle periodicity, amplified); "
            "  Mechanisms not fully understood — likely ELANE-NE regulates oscillator signalling; "
            "CSF3R (G-CSF RECEPTOR) SIGNALLING: "
            "  G-CSF → CSF3R → JAK2/STAT3/ERK → myeloid proliferation + differentiation; "
            "  G-CSF treatment rescues ELANE-SCN by driving committed progenitors past arrest point; "
            "AML TRANSFORMATION: "
            "  Acquired CSF3R truncation mutations (T617I / d715 region) → G-CSF hypersensitivity → "
            "  → clonal expansion → AML/MDS transformation signal; "
            "  Monitor: annual BM aspirate + cytogenetics + CSF3R mutation screening in SCN-ELANE"
        ),
        "pathognomonic": (
            "ABSENT PUS FORMATION — PATHOGNOMONIC for severe neutropenia: "
            "  Bacterial infections without visible pus (no neutrophils to form abscess); "
            "  'Cold abscess' (no warmth/fluctuance) from concurrent immunoglobulin-competent but neutrophil-absent response; "
            "MATURATION ARREST AT PROMYELOCYTE — BM DIAGNOSTIC: "
            "  Bone marrow shows abundant promyelocytes but very few myelocytes/metamyelocytes/bands; "
            "  Primary (azurophil) granules present; secondary (specific) granule-stage absent; "
            "  ELANE first expressed at promyelocyte stage — toxic misfolded NE kills progenitors at this stage; "
            "21-DAY CYCLE PERIODICITY (CN) — PATHOGNOMONIC for cyclic neutropenia: "
            "  Serial 3×/week CBC for 6-8 weeks documents predictable 21-day ANC nadir cycle; "
            "  Oral aphthae + fever at each nadir = cyclic pattern; "
            "OMPHALITIS IN NEONATE with NEUTROPENIA: "
            "  Omphalitis (umbilical stump infection) + neutropenia at birth → SCN until excluded; "
            "PERIODONTAL DISEASE SEVERITY: "
            "  SCN patients without G-CSF develop severe alveolar bone loss / tooth loss by age 5-10 — "
            "  indicates chronically severe neutropenia; dental surveillance mandatory"
        ),
        "treatment": (
            "G-CSF (FILGRASTIM / LENOGRASTIM) — FIRST-LINE SCN: "
            "  Starting dose: 5 mcg/kg/day SC; titrate to ANC target >1.0 × 10⁹/L (ideally 1.0-10.0); "
            "  Response: 90-95% of ELANE-SCN achieve ANC >1.0 × 10⁹/L on G-CSF; "
            "  Frequency: daily (some patients alternate day after stabilisation); "
            "  HIGH-DOSE G-CSF (>20 mcg/kg/day): increased AML risk — minimise dose; "
            "  CYCLIC NEUTROPENIA: G-CSF shortens nadir duration and reduces infection frequency but does NOT abolish cyclicity; "
            "BONE MARROW SURVEILLANCE (mandatory SCN on G-CSF): "
            "  Annual BM aspirate + cytogenetics + CSF3R mutation analysis; "
            "  Early CSF3R truncation: intensify surveillance ± HSCT evaluation; "
            "  MDS/AML criteria → HSCT; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): "
            "  INDICATIONS: G-CSF non-response (ANC <0.5 despite G-CSF >20 mcg/kg/day); AML/MDS transformation; "
            "  CURATIVE: eliminates transformation risk; 5-year OS ~80%; "
            "ANTIBIOTIC PROPHYLAXIS: "
            "  Trimethoprim-sulfamethoxazole prophylaxis for PCP when ANC <0.5; "
            "  Antifungal prophylaxis when ANC <0.1 for prolonged periods; "
            "DENTAL CARE: "
            "  Intensive dental hygiene protocol; biannual dental review; "
            "  G-CSF before dental procedures (elective) to boost ANC; "
            "  Chlorhexidine mouthwash reduces gingivitis progression"
        ),
        "seed": 2822,
        "pt_vars": {
            "anc_nadir": (0.01, 0.18),
            "infection_per_year": (3.0, 12.0),
            "gcsf_dose_mcg_kg": (3, 22),
            "gcsf_response_pct": 92,
            "aml_risk_pct": 18,
        }
    },
    {
        "gene": "HAX1",
        "protein": (
            "HAX1 -- 1q21.3 AR -- 279aa -- HCLS1-Associated-Protein-X-1-32kDa-"
            "Pro-Apoptotic-Mitochondrial-ER-Adapter-Kostmann-Syndrome-"
            "OMIM-Gene-605998-Disease-Kostmann-SCN3-610738"
        ),
        "locus": "1q21.3",
        "protein_size": "279 aa / 32 kDa (HCLS1-associated adaptor protein; mitochondrial and ER membrane; anti-apoptotic function in myeloid cells; regulates mitochondrial membrane potential; two isoforms — HAX1-I and HAX1-II — truncating mutations in HAX1-II affect neurological isoform; AR biallelic required)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic loss-of-function) — original Kostmann disease gene (~5-10% of SCN); "
            "HISTORY: Rolf Kostmann described autosomal recessive severe neutropenia in Swedish families (1956); "
            "  Original Kostmann families harbour HAX1 mutations; now called SCN3 (OMIM 610738); "
            "  HAX1 gene identified 2007 (Klein et al., Nature Genetics); "
            "MECHANISM: "
            "  HAX1 = anti-apoptotic regulator at mitochondrial outer membrane and ER; "
            "  HAX1 LOF → increased myeloid progenitor apoptosis via: "
            "    1. Mitochondrial membrane potential loss → cytochrome c release → caspase cascade; "
            "    2. ER stress amplification (HAX1 modulates SERCA2/ER calcium); "
            "    3. Procaspase-9 activation (direct inhibition lost); "
            "  HAPLOINSUFFICIENCY NOT SUFFICIENT: both alleles must be disrupted; "
            "  ISOFORM SPECIFICITY (critical clinical distinction): "
            "    HAX1-I isoform: ubiquitous — essential for myeloid survival; "
            "    HAX1-II isoform: brain-expressed — neurological function; "
            "    p.Trp44X truncation: disrupts BOTH isoforms → SCN + intellectual disability + epilepsy; "
            "    p.Gln190X truncation: disrupts only HAX1-I → SCN WITHOUT neurology; "
            "    Mutation type determines whether neurological phenotype co-exists; "
            "G-CSF RESPONSE: 85-95% respond (similar to ELANE-SCN); "
            "AML/MDS RISK: 15-20% (comparable to ELANE-SCN); "
            "CONSANGUINITY: increased frequency in consanguineous families (1q21 large region allows homozygous run)"
        ),
        "disease_category": (
            "KOSTMANN SYNDROME / SEVERE CONGENITAL NEUTROPENIA TYPE 3 (SCN3) — OMIM 610738; "
            "HAEMATOLOGICAL FEATURES: "
            "  ANC: <0.2 × 10⁹/L (severe/persistent); "
            "  BM: maturation arrest at promyelocyte (similar to ELANE-SCN); "
            "  Eosinophilia (relative): common background finding; monocytosis often present; "
            "DISTINCTIVE FEATURES vs ELANE-SCN: "
            "  1. AUTOSOMAL RECESSIVE — affected siblings possible; parental carrier testing mandatory; "
            "  2. NEUROLOGICAL PHENOTYPE IN SUBSET: "
            "     HAX1-II isoform disruption (e.g., p.Trp44X) → intellectual disability + epilepsy + developmental delay; "
            "     HAX1-I only disruption → pure haematological phenotype; "
            "     MRI: progressive white matter changes in neurological subset; "
            "  3. CONSANGUINITY HISTORY: common finding — must document pedigree; "
            "  4. SWEDISH ORIGIN: original Kostmann families from northern Sweden (founder effect p.Gln190X); "
            "PRESENTATION: "
            "  Onset: birth / first months of life; "
            "  Recurrent bacterial infections (skin/soft tissue/pneumonia/septicaemia); "
            "  Mouth ulcers, severe gingivitis; "
            "DIAGNOSIS: "
            "  Serial CBC (persistent ANC <0.2); BM aspirate (promyelocyte arrest); "
            "  HAX1 gene sequencing (identify isoform-specific mutation for neurological risk stratification); "
            "  Neurological assessment + MRI (all HAX1-SCN patients at diagnosis)"
        ),
        "disease_pathway": (
            "HAX1 ANTI-APOPTOTIC MITOCHONDRIAL-ER PATHWAY: "
            "NORMAL HAX1 FUNCTION: "
            "  HAX1 expressed on mitochondrial outer membrane and ER membrane; "
            "  Interacts with: HCLS1 (SH3 domain), procaspase-9, SERCA2 (ER Ca²⁺ pump); "
            "  Maintains mitochondrial membrane potential (ΔΨm) in myeloid progenitors; "
            "  Inhibits caspase-9 activation under baseline stress conditions; "
            "HAX1 LOF MECHANISM: "
            "  Loss of HAX1 → mitochondrial ΔΨm collapse → cytochrome c release → "
            "  → apoptosome formation (Apaf-1 + cyt c + procaspase-9) → caspase-9 activation → "
            "  → executioner caspase-3/7 → apoptosis of myeloid progenitors; "
            "ER COMPONENT: "
            "  HAX1 regulates SERCA2 (ER Ca²⁺-ATPase) → loss → ER Ca²⁺ depletion → ER stress → "
            "  → UPR amplification (synergises with misfolded-NE mechanism in compound phenotypes); "
            "ISOFORM BIOLOGY: "
            "  HAX1-II extends 28aa at C-terminus via alternative splicing; "
            "  HAX1-II: expressed in neurons; HAX1-I: ubiquitous (myeloid cells); "
            "  Truncations eliminating HAX1-II = neurological phenotype risk; "
            "  p53-DEPENDENT APOPTOSIS: HAX1 also modulates p53 stability in myeloid cells → "
            "    HAX1 LOF → p53 upregulation → amplified myeloid apoptosis (same p53 node as ELANE-UPR); "
            "G-CSF RESCUE: "
            "  G-CSF → CSF3R → JAK2/STAT3 → BCL-XL/BCL-2 upregulation → anti-apoptotic signals "
            "  partially compensate for HAX1 loss → progenitors survive to mature to bands/segs"
        ),
        "pathognomonic": (
            "AUTOSOMAL RECESSIVE SCN IN CONSANGUINEOUS FAMILY — DIAGNOSTIC FLAG FOR HAX1: "
            "  AR-SCN with parental consanguinity → HAX1 sequencing first; "
            "  Sibling recurrence risk 25% (both parents carriers); "
            "NEUROLOGICAL PHENOTYPE IN SCN — HAX1 HALLMARK: "
            "  Intellectual disability + epilepsy in child with SCN → HAX1 isoform-disrupting mutation; "
            "  EEG abnormalities / seizures in neonatal period with concurrent neutropenia → test HAX1; "
            "  MRI white matter changes in subset; "
            "SWEDISH/SCANDINAVIAN ANCESTRY with SCN: "
            "  Original Kostmann families from Nordmaling, Sweden; "
            "  p.Gln190X and p.Trp44X founder mutations (documented in northern Swedish pedigrees); "
            "EOSINOPHILIA BACKGROUND: "
            "  Mild eosinophilia (relative) common in HAX1-SCN on BM / peripheral blood; "
            "MONOCYTOSIS: "
            "  Relative monocytosis compensates for absent neutrophils; "
            "  Monocytes engulf but less effectively than neutrophils → infections still occur"
        ),
        "treatment": (
            "G-CSF (FILGRASTIM) — FIRST-LINE (same as ELANE-SCN): "
            "  Starting dose 5 mcg/kg/day → titrate to ANC >1.0 × 10⁹/L; "
            "  Response: 85-95% achieve ANC >1.0 × 10⁹/L; "
            "  NEUROLOGICAL MONITORING: serial neurodevelopmental assessments every 6 months; "
            "  Neurological phenotype does NOT respond to G-CSF (G-CSF for haematological only); "
            "NEUROLOGY CO-MANAGEMENT (HAX1-II-affected mutations): "
            "  Antiepileptic therapy if seizures present (LEV first-line in paediatrics); "
            "  Neuropsychology input + special educational support; "
            "  MRI surveillance: baseline + every 2-3 years; "
            "ANNUAL BONE MARROW SURVEILLANCE: "
            "  Cytogenetics + CSF3R truncation screening; "
            "  MDS/AML: trigger HSCT evaluation; "
            "HSCT (CURATIVE): "
            "  Eliminates neutropenia (haematological); "
            "  Does NOT correct neurological phenotype (HAX1-II expressed in neurons — "
            "    HSCT replaces haematopoietic but not neural HAX1); "
            "  IMPORTANT COUNSELLING: neurological progression may continue despite HSCT cure of neutropenia; "
            "GENETIC COUNSELLING: "
            "  Carrier testing of both parents (mandatory); sibling testing (25% risk); "
            "  Prenatal diagnosis available; preimplantation genetic testing (PGT) option"
        ),
        "seed": 2823,
        "pt_vars": {
            "anc_nadir": (0.01, 0.15),
            "infection_per_year": (4.0, 14.0),
            "gcsf_dose_mcg_kg": (3, 20),
            "gcsf_response_pct": 90,
            "aml_risk_pct": 17,
        }
    },
    {
        "gene": "G6PC3",
        "protein": (
            "G6PC3 -- 17q21.31 AR -- 346aa -- Glucose-6-Phosphatase-Catalytic-Subunit-3-"
            "37kDa-ER-Membrane-9-TM-Helices-Dursun-Syndrome-SCN4-"
            "OMIM-Gene-611045-Disease-Dursun-SCN4-612541"
        ),
        "locus": "17q21.31",
        "protein_size": "346 aa / 37 kDa (glucose-6-phosphatase catalytic subunit-3; ER membrane protein with 9 TM helices; hydrolyses glucose-6-phosphate → glucose in ER; maintains glucose supply in ER during ER stress; G6PC3 is the ubiquitous isoform — G6PC (liver) and G6PC2 (pancreas) are tissue-specific paralogues; biallelic LOF → Dursun syndrome = SCN4)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic loss-of-function) — Dursun syndrome / SCN4 (~2-5% of SCN); "
            "MECHANISM: "
            "  G6PC3 hydrolyses glucose-6-phosphate → free glucose in ER lumen; "
            "  Maintains ER glucose supply → critical for ER protein folding (glycosylation requires UDP-glucose); "
            "  G6PC3 LOF → ER glucose depletion → ER stress → UPR → myeloid progenitor apoptosis; "
            "  NEUTROPHIL SPECIFICITY: neutrophil progenitors highly dependent on G6PC3 for ER glucose; "
            "  PARALLEL WITH G6PC (liver): G6PC (GSD-Ia) mutations cause hepatic GSD; G6PC3 mutations = different syndrome (Dursun) — no hypoglycaemia because liver G6PC intact; "
            "  KEY DISTINCTION: G6PC3-Dursun has NO hepatomegaly, NO hypoglycaemia (liver G6PC unaffected); "
            "DISTINCTIVE MULTI-ORGAN PHENOTYPE (Dursun syndrome): "
            "  The ONLY SCN gene with major cardiac and urogenital anomalies as a defining feature; "
            "  Prominent superficial venous pattern (visible veins on skin) — DISTINCTIVE cosmetic finding; "
            "G-CSF RESPONSE: 80-90% respond; dose often higher than ELANE-SCN; "
            "AML RISK: 10-15% (slightly lower than ELANE-HAX1); "
            "GENOTYPE: homozygous or compound heterozygous; de novo rare; family history of consanguinity common"
        ),
        "disease_category": (
            "DURSUN SYNDROME / SEVERE CONGENITAL NEUTROPENIA TYPE 4 (SCN4) — OMIM 612541; "
            "CARDINAL FEATURES (ALL G6PC3-SCN) — TRIAD BEYOND NEUTROPENIA: "
            "  1. CARDIAC DEFECTS (55-75%): "
            "     Structural anomalies: VSD most common; ASD; PDA; complex CHD in severe cases; "
            "     Cardiac surveillance: echocardiogram at diagnosis + annually; "
            "  2. UROGENITAL ANOMALIES (40-60%): "
            "     Cryptorchidism (males 70%); vesicoureteral reflux; duplex collecting system; "
            "     Renal anomaly screening: renal ultrasound at diagnosis + monitoring; "
            "  3. INNER EAR MALFORMATIONS (20-40%): "
            "     Sensorineural hearing loss; cochlear anomalies on MRI/CT; "
            "     Audiology assessment mandatory at diagnosis; "
            "  4. PROMINENT SUPERFICIAL VEINS: "
            "     Visible cutaneous venous pattern (thorax/extremities) — COSMETIC but DISTINCTIVE CLINICAL CLUE; "
            "     Not a haemodynamic abnormality; "
            "HAEMATOLOGICAL FEATURES: "
            "  ANC <0.2 × 10⁹/L; maturation arrest (promyelocyte/myelocyte); "
            "  Anaemia (mild, normocytic) and intermittent thrombocytopenia in some; "
            "  BM: hypercellular with myeloid maturation arrest; relative eosinophilia; "
            "CLINICAL ONSET: first weeks of life with bacterial infections"
        ),
        "disease_pathway": (
            "G6PC3 / ER GLUCOSE HOMEOSTASIS / ER STRESS PATHWAY: "
            "NORMAL G6PC3 FUNCTION: "
            "  G6-phosphate generated in cytoplasm (hexokinase + glycolysis) → translocated to ER via SLC37A4 (G6P translocase); "
            "  G6PC3 → hydrolyses G6P → free glucose in ER lumen; "
            "  ER glucose → converted to UDP-glucose → substrate for ER protein glycosylation (N-glycosylation); "
            "  G6PC3 also: regulates ER redox via GSH transport; maintains NADPH (via pentose phosphate shunt in ER); "
            "G6PC3 LOF CONSEQUENCE: "
            "  G6P accumulates in ER → depletes free glucose → "
            "  → UDP-glucose supply reduced → N-glycosylation of ER proteins impaired → "
            "  → misfolded/unglycosylated proteins accumulate → ER STRESS → UPR → apoptosis; "
            "  Neutrophil progenitors express high G6PC3 → most vulnerable; "
            "PARALLEL WITH GSD-1B (SLC37A4 DEFICIENCY): "
            "  SLC37A4 transports G6P into ER (translocase); SLC37A4 mutations → GSD-1b (neutropenia + liver); "
            "  G6PC3 is DOWNSTREAM of SLC37A4; G6PC3-deficiency = ER-isolated form without liver disease; "
            "  Empagliflozin (SGLT2 inhibitor) being investigated in GSD-1b and G6PC3 for neutropenia "
            "  (reduces G6P accumulation by reducing renal tubular G6P reabsorption — investigational); "
            "MULTI-ORGAN MANIFESTATION: "
            "  Cardiac defects: G6PC3 expressed in cardiac development → LOF disrupts cardiac progenitor metabolism; "
            "  Urogenital: G6PC3 expressed in mesonephros/gonadal ridge → renal/gonadal anomalies; "
            "  Cochlea: G6PC3 expressed in inner ear sensory epithelium development"
        ),
        "pathognomonic": (
            "SCN + CARDIAC ANOMALIES — PATHOGNOMONIC COMBINATION FOR G6PC3/Dursun: "
            "  Neutropenia at birth + structural cardiac defect (especially VSD) → G6PC3 testing first; "
            "  No other SCN gene causes structural heart defects at this frequency (55-75%); "
            "PROMINENT SUPERFICIAL VEINS IN CHILD WITH NEUTROPENIA — G6PC3 CLUE: "
            "  Visible superficial venous network (thorax + extremities) in infant with SCN → distinctive G6PC3 finding; "
            "  Not explained by malnutrition alone; present from infancy; "
            "CRYPTORCHIDISM + SCN IN MALE INFANT: "
            "  Cryptorchidism (70% of G6PC3 males) + neutropenia → G6PC3 primary consideration; "
            "  SOHH (structural heart + undescended testes + neutropenia) = virtually pathognomonic for G6PC3; "
            "SENSORINEURAL HEARING LOSS + SCN: "
            "  Inner ear anomaly / SNHL in child with SCN → G6PC3 panel; "
            "NO HYPOGLYCAEMIA (critical negative finding): "
            "  G6PC3-Dursun has NO fasting hypoglycaemia (liver G6PC intact) — "
            "  distinguishes from GSD type 1 (G6PC or SLC37A4) which causes severe hypoglycaemia; "
            "RENAL ULTRASOUND ANOMALY + SCN: "
            "  Duplex collecting system, hydronephrosis, or VUR detected on routine neonatal US + neutropenia = G6PC3"
        ),
        "treatment": (
            "G-CSF (FILGRASTIM) — FIRST-LINE: "
            "  Starting 5-10 mcg/kg/day; often requires higher doses than ELANE-SCN; "
            "  Target ANC >1.0 × 10⁹/L; 80-90% respond; "
            "CARDIAC MANAGEMENT: "
            "  Cardiology co-management from diagnosis; echocardiogram at diagnosis; "
            "  Surgical repair of VSD/ASD per standard cardiac criteria; "
            "  Cardiac review annually even after repair; "
            "UROLOGICAL MANAGEMENT: "
            "  Renal ultrasound at diagnosis + annually; urology referral for VUR (prophylactic antibiotics); "
            "  Orchidopexy: standard of care for cryptorchidism (within first 18 months); "
            "AUDIOLOGY: "
            "  Formal audiological assessment at diagnosis; cochlear implant consideration for SNHL >70 dB; "
            "  Annual audiology follow-up; early hearing aids to support speech development; "
            "ANNUAL BM SURVEILLANCE (as per all SCN on G-CSF): "
            "  Cytogenetics + CSF3R mutation; "
            "HSCT: "
            "  Indicated for G-CSF non-response, MDS/AML; "
            "  IMPORTANT: HSCT cures the HAEMATOLOGICAL phenotype only; "
            "  Cardiac/urogenital/audiology anomalies persist post-HSCT and require independent management; "
            "EMPAGLIFLOZIN (INVESTIGATIONAL): "
            "  SGLT2 inhibitor reduces ER G6P accumulation; early case series in SCN4/GSD-1b showing ANC improvement; "
            "  Not yet standard of care; enrol in registries"
        ),
        "seed": 2824,
        "pt_vars": {
            "anc_nadir": (0.01, 0.20),
            "infection_per_year": (3.0, 11.0),
            "gcsf_dose_mcg_kg": (5, 25),
            "gcsf_response_pct": 85,
            "aml_risk_pct": 13,
        }
    },
    {
        "gene": "WAS",
        "protein": (
            "WAS -- Xp11.22 XLR -- 502aa -- Wiskott-Aldrich-Syndrome-Protein-WASP-"
            "53kDa-Arp2-3-Activator-PH-WH1-GBD-PRD-VCA-Domains-"
            "OMIM-Gene-300392-Disease-WAS-300100-XLN-300299"
        ),
        "locus": "Xp11.22",
        "protein_size": "502 aa / 53 kDa (Wiskott-Aldrich syndrome protein; Arp2/3 complex activator → actin polymerisation; haematopoietic-specific expression; N-WASP ubiquitous; WASp = haematopoietic only; domains: WH1 (N-terminal, WIP-binding), GBD (Cdc42-binding, autoinhibition), PRD (proline-rich, SH3 binders), VCA (verprolin-cofilin-acidic, Arp2/3 activation)); XLR = hemizygous males affected; carrier females mosaic/protected",
        "inheritance": (
            "X-LINKED RECESSIVE — hemizygous males affected; carrier females generally unaffected (UNLESS extreme lyonisation); "
            "SPECTRUM OF WAS MUTATIONS (genotype-phenotype critical): "
            "  NULL MUTATIONS (frameshift/nonsense/large deletion) → classic WAS (severe triad): "
            "    Thrombocytopenia + Eczema + Immunodeficiency — XLT/WAS triad; "
            "    WASp absent on Western blot / flow cytometry (WASp level diagnostic); "
            "  MISSENSE IN GBD/VCA → partial WASp function → attenuated WAS (XLT = X-linked thrombocytopenia); "
            "  GAIN-OF-FUNCTION (GOF) MISSENSE (L270P/I294T/S272P/A134T) → X-LINKED SEVERE CONGENITAL NEUTROPENIA (XLN): "
            "    Constitutively active WASp → actin hyperpolymerisation → neutrophil differentiation arrest; "
            "    Unique: SCN WITHOUT thrombocytopenia; WASp protein PRESENT; "
            "    AML transformation risk VERY HIGH in XLN (>30% lifetime); "
            "    HSCT URGENTLY recommended once XLN confirmed — highest malignant risk among SCN genes; "
            "CLASSIC WAS TRIAD: "
            "  Microthromobocytopenia (small platelets + low count, typically 20-50 × 10⁹/L); "
            "  Eczema (atopic, severe); "
            "  Combined B+T cell immunodeficiency; "
            "WASp EXPRESSION: test by flow cytometry (WASp+/−); critical for WAS vs XLN classification"
        ),
        "disease_category": (
            "WISKOTT-ALDRICH SYNDROME (WAS) — OMIM 300100; "
            "X-LINKED CONGENITAL NEUTROPENIA (XLN) — OMIM 300299; "
            "CLASSIC WAS — TRIAD: "
            "  1. MICROTHROMBOCYTOPENIA: "
            "     Platelet count 20-50 × 10⁹/L (often <50); "
            "     SMALL PLATELETS (mean platelet volume MPV <7 fL) — key distinguishing feature from ITP; "
            "     ITP: platelets small in WAS vs ITP (large platelets in ITP) → check MPV; "
            "     Bleeding: petechiae, bloody diarrhoea, ICH risk (lifelong); "
            "  2. ECZEMA: "
            "     Present in 80%; severity variable; often severe atopic dermatitis; "
            "     Can be the presenting feature in infancy; "
            "  3. IMMUNODEFICIENCY: "
            "     Combined humoral + cellular; elevated IgA/IgE; low IgM; variable IgG; "
            "     Poor vaccine responses; recurrent bacterial (encapsulated organisms) + viral (HSV/CMV) infections; "
            "     Autoimmune complications: haemolytic anaemia, vasculitis, IBD (30-40%); "
            "     Lymphoma risk (B-cell NHL): 13-22% lifetime; "
            "XLN (GOF WAS): "
            "  Isolated severe neutropenia (ANC <0.2) WITHOUT thrombocytopenia/eczema — confusing phenotype; "
            "  WASp present (GOF mutation, not null); "
            "  Myelodysplasia / AML transformation risk highest of all SCN genes; "
            "DIAGNOSIS: "
            "  WASp protein by flow cytometry (CD3+ T cells); absent in WAS, present/elevated in XLN; "
            "  WAS gene sequencing; classify as LOF vs GOF mutation"
        ),
        "disease_pathway": (
            "WASp / Arp2/3 / ACTIN NUCLEATION PATHWAY: "
            "NORMAL WASP FUNCTION: "
            "  WASp expressed EXCLUSIVELY in haematopoietic cells; "
            "  Autoinhibited conformation: WH1 + GBD fold → VCA domain inaccessible; "
            "  ACTIVATION: Cdc42-GTP binds GBD → conformational change → VCA domain exposed → "
            "  → VCA activates Arp2/3 complex → branches actin polymerisation; "
            "  FUNCTIONS in haematopoietic cells: "
            "    Immunological synapse formation (T cell activation); "
            "    Phagocytic cup formation (neutrophil killing); "
            "    Platelet production (proplatelet extension by megakaryocytes); "
            "    Lymphocyte migration (CXCR4 downstream); "
            "CLASSIC WAS (LOF): "
            "  Absent WASp → impaired Arp2/3 activation → actin polymerisation defect → "
            "    → T-cell IS formation failure → T cell activation impaired → immunodeficiency; "
            "    → Megakaryocyte proplatelet extension defective → microthrombocytopenia; "
            "    → Neutrophil phagocytic cup defective (neutrophil count often borderline low in WAS); "
            "XLN (GOF WAS): "
            "  Constitutively active WASp (GBD mutation removes autoinhibition) → "
            "  → persistent Arp2/3 activation → hyperactive actin polymerisation → "
            "  → cytoskeletal dysfunction in myeloid progenitors → neutrophil differentiation arrest → "
            "  → neutropenia; "
            "  Myeloid progenitors: hyperpolymerised actin → ectopic lamellipodia → impaired division → arrest; "
            "WASp AUTOIMMUNITY MECHANISM: "
            "  WASp participates in Treg function and peripheral tolerance; "
            "  WAS LOF → impaired Treg suppressive capacity → autoimmunity (30-40%)"
        ),
        "pathognomonic": (
            "SMALL PLATELETS (MPV <7 fL) + THROMBOCYTOPENIA IN MALE INFANT — PATHOGNOMONIC FOR WAS: "
            "  MPV distinguishes WAS microthrombocytes from ITP (large platelets in ITP); "
            "  Male infant + thrombocytopenia + small platelets → WAS until excluded; "
            "  Also: bloody diarrhoea in male infant with thrombocytopenia = classic WAS presentation; "
            "WASP ABSENCE ON FLOW CYTOMETRY — DIAGNOSTIC: "
            "  WASp expression by flow cytometry on CD3+ T cells: "
            "    Absent (LOF/null) = classic WAS; "
            "    Present/elevated (GOF) = XLN (neutropenia without thrombocytopenia); "
            "  RAPID TEST: can be done within 24 hours; single most useful diagnostic step after CBC; "
            "BLOODY DIARRHOEA + ECZEMA + THROMBOCYTOPENIA IN INFANT BOY — CLASSIC WAS TRIAD: "
            "  Any 2 of 3 in male infant → test WASp immediately; "
            "GOF WAS (XLN): MALE WITH ISOLATED SCN + AML RISK >30%: "
            "  Male patient with severe isolated neutropenia (no thrombocytopenia/eczema) + "
            "  strong family history of X-linked SCN/AML in males → GOF WAS (XLN) first; "
            "  URGENT HSCT evaluation in XLN — highest AML risk of any SCN gene"
        ),
        "treatment": (
            "CLASSIC WAS (LOF null mutations): "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT) — CURATIVE: "
            "  Standard of care for all WAS patients; should not be delayed; "
            "  Best results: HLA-matched sibling or matched unrelated donor, age < 5 years (OS >90%); "
            "  HSCT cures: neutropenia + thrombocytopenia + immunodeficiency + eczema + autoimmunity + lymphoma risk; "
            "  Supportive therapy: IVIG (monthly), PCP prophylaxis, antifungal; "
            "GENE THERAPY (investigational): "
            "  Ex vivo lentiviral WAS gene correction of autologous HSCs; "
            "  Clinical trials: significant immune reconstitution; thrombocytopenia corrected; "
            "  Insertional mutagenesis risk (γ-retroviral) overcome by SIN-lentiviral design; "
            "SYMPTOMATIC: "
            "  Thrombocytopenia: avoid aspirin/NSAIDs; platelet transfusion for bleeding; "
            "  Eczema: topical steroids; tacrolimus; avoid scratching triggers; "
            "  IVIG: monthly to prevent bacterial infections; "
            "XLN (GOF mutations): "
            "  G-CSF: limited benefit (GOF mutation — constitutive activation); "
            "  HSCT: URGENTLY recommended (>30% lifetime AML risk); "
            "  AML monitoring: annual BM aspirate from diagnosis in XLN; "
            "  No targeted therapy for XLN currently standard"
        ),
        "seed": 2825,
        "pt_vars": {
            "anc_nadir": (0.01, 0.25),
            "infection_per_year": (3.0, 10.0),
            "gcsf_dose_mcg_kg": (5, 15),
            "gcsf_response_pct": 58,
            "aml_risk_pct": 27,
        }
    },
    {
        "gene": "CXCR4",
        "protein": (
            "CXCR4 -- 2q22.1 AD -- 360aa -- C-X-C-Motif-Chemokine-Receptor-4-"
            "39kDa-GPCR-7TM-SDF1-CXCL12-Receptor-Gain-of-Function-C-Terminal-Truncation-"
            "OMIM-Gene-162643-Disease-WHIM-193670"
        ),
        "locus": "2q22.1",
        "protein_size": "360 aa / 39 kDa (C-X-C chemokine receptor type 4; 7-TM GPCR; primary receptor for CXCL12/SDF-1; expressed on leucocytes, HSCs, endothelium; normal C-terminus contains WHIM mutations (truncation of last 10-19 aa) → increased CXCL12/CXCR4 signalling retention in BM → myelokathexis; GOF mechanism unique among SCN genes)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (gain-of-function via C-terminal truncation) — WHIM syndrome; "
            "UNIQUE MECHANISM — unlike all other SCN genes: "
            "  Normal CXCR4: CXCL12 → CXCR4 → signalling → receptor desensitisation/internalisation → "
            "    C-terminal serine/threonine residues phosphorylated → β-arrestin recruitment → receptor internalised; "
            "  WHIM mutations: truncation of C-terminus (last 10-19 aa) → phosphorylatable serines lost → "
            "    → β-arrestin cannot bind → receptor NOT internalised → HYPERRESPONSIVE to CXCL12; "
            "  CONSEQUENCE: CXCL12 (SDF-1) in bone marrow stroma → CXCR4-GOF → "
            "    → mature neutrophils UNABLE to leave bone marrow (hyperactivated retention signal); "
            "    → myelokathexis (hypersegmented pyknotic neutrophils in BM, unable to exit); "
            "    → peripheral blood neutropenia despite FULL NORMAL MATURATION in BM; "
            "  KEY DISTINCTION: BM maturation is COMPLETE (unlike all other SCN) but neutrophil EGRESS blocked; "
            "  ALSO: CXCR4-GOF in lymphocytes → lymphocyte retention in BM/nodes → lymphopenia; "
            "    B-cell retention → hypogammaglobulinaemia; "
            "    HPV-specific immunity impaired → warts (HPV); "
            "PREVALENCE: WHIM syndrome: very rare (<50 families worldwide; 1 large Newfoundland kindred p.R334X); "
            "DOMINANT: one allele mutated; 50% transmission risk"
        ),
        "disease_category": (
            "WHIM SYNDROME — OMIM 193670; "
            "WHIM ACRONYM: Warts + Hypogammaglobulinaemia + Infections + Myelokathexis; "
            "FOUR CARDINAL FEATURES: "
            "  1. WARTS (W) — HPV-DRIVEN: "
            "     Multiple cutaneous warts (HPV types 2, 6, 11, 16, 18); "
            "     Anogenital HPV; cervical/oropharyngeal HPV-associated malignancy risk elevated; "
            "     HPV immunity requires CXCR4-mediated lymphocyte trafficking — impaired in WHIM; "
            "  2. HYPOGAMMAGLOBULINAEMIA (H): "
            "     Low IgG (all subclasses); low IgA/IgM; poor vaccine responses; "
            "     B cells present but sequestered in BM/nodes (CXCL13 retention); "
            "  3. INFECTIONS (I): "
            "     Recurrent bacterial infections (encapsulated organisms: Streptococcus pneumoniae, H. influenzae); "
            "     Sinopulmonary disease; recurrent otitis media; pneumonia; "
            "  4. MYELOKATHEXIS (M): "
            "     Hypersegmented pyknotic neutrophils in bone marrow + peripheral neutropenia; "
            "     BM: FULL neutrophil maturation (bands + segmented neutrophils present) but pyknotic/hypersegmented; "
            "     CRITICAL DDx: myelokathexis = neutrophils ARE mature but appear degenerate on smear; "
            "     This distinguishes WHIM from all other SCN (all others have maturation arrest); "
            "ADDITIONAL FEATURES: "
            "  Lymphopenia (B > T cell); "
            "  Occasional monocytopenia; "
            "  HPV-related malignancies if not prevented (cervical carcinoma, oropharyngeal SCC)"
        ),
        "disease_pathway": (
            "CXCL12/CXCR4 AXIS / NEUTROPHIL BONE MARROW EGRESS PATHWAY: "
            "NORMAL CXCL12/CXCR4 NEUTROPHIL EGRESS: "
            "  CXCL12 (SDF-1): produced by BM stromal cells (CXCL12-abundant reticular cells, CAR cells); "
            "  CXCR4: expressed on HSCs, neutrophil progenitors → retains cells in BM niche; "
            "  G-CSF → CXCR4 downregulation + CXCL12 degradation → neutrophil release from BM → circulation; "
            "  Mature neutrophils: CXCR4 low → can leave BM → peripheral blood; "
            "  Old/senescent neutrophils: CXCR4 re-expression → return to BM → clearance by macrophages; "
            "WHIM CXCR4-GOF MECHANISM: "
            "  Truncated CXCR4 → no β-arrestin desensitisation → "
            "  → sustained, hyperactivated CXCL12 signalling in BM → "
            "  → PI3Kγ/Akt/ERK persistent activation → "
            "  → neutrophil chemotaxis signals lock cells in BM; "
            "  → EVEN MATURE SEGMENTED NEUTROPHILS cannot exit BM; "
            "MYELOKATHEXIS MORPHOLOGY: "
            "  Prolonged neutrophil BM retention → "
            "  → nuclear hypersegmentation + karyorrhexis (pyknotic) + cytoplasmic vacuolation → "
            "  → apoptotic-appearing but not truly apoptotic (chromatin fragmentation pattern different); "
            "AMD3100 (PLERIXAFOR) MECHANISM: "
            "  Plerixafor = CXCR4 antagonist; blocks CXCL12 binding; "
            "  In WHIM: plerixafor mobilises sequestered neutrophils from BM → rapid ANC normalisation; "
            "  PATHOGNOMONIC RESPONSE TO PLERIXAFOR: ANC normalises within hours; "
            "  Confirms WHIM diagnosis by mechanism"
        ),
        "pathognomonic": (
            "MYELOKATHEXIS ON BM SMEAR — PATHOGNOMONIC FOR CXCR4-WHIM: "
            "  BM smear: hypersegmented (5-10 lobes), pyknotic, vacuolated neutrophils; "
            "  FULL MATURATION PRESENT (bands + segs abundant in BM) — unique among SCN genes; "
            "  Contrast: ALL other SCN genes have maturation arrest; WHIM has complete maturation + egress failure; "
            "WARTS + NEUTROPENIA COMBINATION: "
            "  Multiple HPV warts in child/young adult with neutropenia → WHIM syndrome first; "
            "  No other SCN syndrome has HPV wart susceptibility as a cardinal feature; "
            "PLERIXAFOR (CXCR4 ANTAGONIST) RESPONSE — DIAGNOSTIC AND THERAPEUTIC: "
            "  Subcutaneous plerixafor (0.24 mg/kg) → ANC rises from <0.2 to >5.0 × 10⁹/L within 4-6 hours; "
            "  This immediate dramatic ANC rise is pathognomonic for WHIM / CXCR4-GOF; "
            "NEWFOUNDLAND KINDRED / p.R334X FOUNDER: "
            "  Large Newfoundland, Canada pedigree with WHIM; p.Arg334X truncation founder mutation; "
            "  Historically documented large family allows easy molecular diagnosis; "
            "HYPOGAMMAGLOBULINAEMIA + LYMPHOPENIA + NEUTROPENIA TRIPLE HAEMATOLOGICAL DEFICIENCY: "
            "  Triad of pan-leucopenia + hypogammaglobulinaemia + warts = WHIM essentially confirmed"
        ),
        "treatment": (
            "PLERIXAFOR (AMD3100, MOZOBIL) — SPECIFIC TARGETED THERAPY FOR WHIM: "
            "  Mechanism: CXCR4 antagonist → mobilises BM-retained neutrophils → ANC normalisation; "
            "  Dose: 0.24 mg/kg SC daily or alternate-day; "
            "  APPROVAL: FDA approved for SCN/WHIM (2024); "
            "  Most specific and targeted therapy for any SCN syndrome; "
            "  RESPONSE: ANC rises from <0.2 to 1.5-10 × 10⁹/L; infection frequency decreases dramatically; "
            "G-CSF (ADJUNCT, SECOND-LINE): "
            "  G-CSF + WHIM: modest ANC response (CXCR4-GOF limits neutrophil egress despite increased production); "
            "  Used before plerixafor availability; now second-line; "
            "IVIG: "
            "  Monthly IVIG replacement for hypogammaglobulinaemia; "
            "  Prevents encapsulated organism infections; "
            "  Continue regardless of plerixafor response; "
            "HPV PREVENTION: "
            "  HPV VACCINATION STRONGLY RECOMMENDED (all WHIM patients + family carriers): "
            "  Gardasil 9 (9-valent): before sexual debut; "
            "  Annual cervical screening (HPV-related malignancy surveillance); "
            "  Wart treatment: cryotherapy, topical cidofovir, immunotherapy; "
            "BM SURVEILLANCE: "
            "  Annual BM aspirate + cytogenetics (CXCR4-WHIM has MDS risk); "
            "GENETIC COUNSELLING: 50% transmission risk (AD)"
        ),
        "seed": 2826,
        "pt_vars": {
            "anc_nadir": (0.02, 0.30),
            "infection_per_year": (2.0, 8.0),
            "gcsf_dose_mcg_kg": (3, 12),
            "gcsf_response_pct": 65,
            "aml_risk_pct": 10,
        }
    },
    {
        "gene": "GFI1",
        "protein": (
            "GFI1 -- 1p22.1 AD -- 422aa -- Growth-Factor-Independence-1-"
            "46kDa-Zinc-Finger-Transcriptional-Repressor-SNAG-Domain-"
            "SCN2-Dominant-Negative-ELANE-Repressor-"
            "OMIM-Gene-600871-Disease-SCN2-613107"
        ),
        "locus": "1p22.1",
        "protein_size": "422 aa / 46 kDa (Growth Factor Independence 1; zinc finger transcriptional repressor with N-terminal SNAG domain (SNAIL/GFI1-family repressor domain); 6 C2H2 zinc fingers (DNA binding); represses ELANE transcription during neutrophil maturation; GFI1-null: de-repression of ELANE → neutrophil toxicity; AD dominant-negative mutations cause SCN2)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (dominant-negative mechanism) — SCN2 (~3-5% SCN); "
            "GFI1 FUNCTION: "
            "  GFI1 = master transcriptional repressor of myeloid differentiation; "
            "  Directly represses ELANE transcription (occupies ELANE promoter → recruits LSD1/CoREST → histone H3K4 demethylation → ELANE silencing); "
            "  Also represses: PBEF1 (nicotinamide phosphoribosyltransferase), CCND2, CEBPE; "
            "  Drives transition: granulocyte-monocyte progenitor (GMP) → granulocyte fate; "
            "DOMINANT-NEGATIVE MECHANISM: "
            "  GFI1 missense mutations (N382S most common; K403R; R412X zinc finger mutations): "
            "    Mutant GFI1 expressed but cannot bind DNA or recruit co-repressors; "
            "    BUT mutant GFI1 still dimerises with WT GFI1 (heterodimerisation via SNAG domain); "
            "    → WT GFI1 + mutant GFI1 heterodimer → LOSS OF WT REPRESSOR ACTIVITY (dominant negative); "
            "    → ELANE de-repressed → excess NE protein → ER stress (same downstream as ELANE mutations); "
            "  NOT same mechanism as GFI1-null (which causes autoimmune disease, not SCN); "
            "PHENOTYPE vs ELANE-SCN: "
            "  Generally MILDER than ELANE-SCN (partial dominant-negative effect); "
            "  ANC nadirs often 0.1-0.5 × 10⁹/L (not always <0.1); "
            "  G-CSF responsiveness high (90-95%); "
            "  AML risk present but lower than ELANE-HAX1 (~8-12%); "
            "  No extra-haematopoietic features (unlike G6PC3 or HAX1)"
        ),
        "disease_category": (
            "SEVERE CONGENITAL NEUTROPENIA TYPE 2 (SCN2) — OMIM 613107; "
            "HAEMATOLOGICAL FEATURES: "
            "  ANC: variable — can be <0.1 (classic SCN) or 0.1-0.5 (moderate range — 'atypical SCN'); "
            "  BM maturation arrest: promyelocyte/myelocyte (mild-moderate); "
            "  BM: often less complete arrest than ELANE-SCN (some bands present); "
            "  Mild anaemia and thrombocytopenia in minority; "
            "CLINICAL PRESENTATION: "
            "  Onset: infancy, but can be later than ELANE (first year of life typical); "
            "  Recurrent bacterial infections (usually milder frequency than ELANE-SCN on average); "
            "  Oral ulcers, gingivitis; "
            "  No extra-haematopoietic anomalies; "
            "DISTINGUISHING FROM ELANE-SCN: "
            "  Both AD; both involve NE pathway (GFI1 de-represses ELANE); "
            "  GFI1-SCN2 generally milder severity; "
            "  Gene sequencing distinguishes (mutation in GFI1 vs ELANE); "
            "  BM: GFI1-SCN2 may show slightly more mature forms; "
            "DIAGNOSIS: "
            "  Clinical SCN criteria; BM aspirate; gene panel (include GFI1 alongside ELANE/HAX1/G6PC3)"
        ),
        "disease_pathway": (
            "GFI1 DOMINANT-NEGATIVE / ELANE DE-REPRESSION / SHARED NE-ER STRESS PATHWAY: "
            "NORMAL GFI1 FUNCTION IN MYELOID DIFFERENTIATION: "
            "  GFI1 occupies ELANE promoter → recruits: "
            "    LSD1 (KDM1A): histone H3K4me2 demethylase → silences ELANE at myelocyte stage; "
            "    CoREST (RCOR1) complex; HDACs → chromatin compaction; "
            "  TIMING: ELANE high in promyelocyte (azurophil granule biogenesis) → "
            "    → GFI1 represses ELANE at myelocyte stage → NE protein falls → mature neutrophil has low NE; "
            "  GFI1 also maintains myeloid proliferation vs differentiation balance; "
            "DOMINANT-NEGATIVE GFI1 MECHANISM: "
            "  DN-GFI1 (N382S, etc.) expressed from mutant allele → cannot bind ELANE promoter; "
            "  DN-GFI1 + WT-GFI1 → non-functional heterodimer → "
            "  → ELANE promoter UNOCCUPIED by functional GFI1 → LSD1/CoREST not recruited → "
            "  → ELANE de-repressed at myelocyte stage → excess NE protein → misfolding → "
            "  → ER STRESS → UPR → CHOP → apoptosis (IDENTICAL downstream as ELANE-SCN); "
            "TRANSCRIPTOMIC OVERLAP: "
            "  RNA-seq: GFI1-SCN2 and ELANE-SCN1 share overlapping ER stress gene signatures; "
            "  Confirms: both converge on NE-ER stress apoptosis axis; "
            "GFI1-NULL vs GFI1-DN: "
            "  GFI1-null (biallelic): severe combined immunodeficiency + monocytosis (autoimmune) NOT SCN; "
            "  GFI1-DN (dominant-negative missense): SCN via ELANE de-repression; "
            "  Mechanistically opposite: null loses myeloid repression globally; DN specifically disrupts ELANE suppression"
        ),
        "pathognomonic": (
            "MILDER SCN + AUTOSOMAL DOMINANT INHERITANCE + NO EXTRA-HAEMATOPOIETIC FEATURES: "
            "  AD SCN with relatively preserved ANC (0.1-0.5 range) and no cardiac/neurological features → "
            "  GFI1 in differential (ELANE first, then GFI1); "
            "ELANE-PATHWAY SCN WITHOUT ELANE MUTATION: "
            "  SCN with ER stress signature (UPR genes elevated) but ELANE sequencing normal → test GFI1; "
            "  GFI1-SCN de-represses ELANE → same pathway; ELANE mRNA may be elevated despite no ELANE mutation; "
            "N382S MOST COMMON GFI1 MUTATION: "
            "  Asparagine-382-Serine in zinc finger 5 → most characterised dominant-negative allele; "
            "  Found in multiple unrelated SCN2 families; "
            "FAMILY HISTORY AD SCN MILDER PHENOTYPE: "
            "  Multi-generation AD SCN with generally moderate phenotype (G-CSF responsive, low AML risk) → "
            "  GFI1 sequencing warranted alongside ELANE"
        ),
        "treatment": (
            "G-CSF (FIRST-LINE — SAME AS ELANE-SCN): "
            "  Starting 5 mcg/kg/day; 90-95% respond; "
            "  GFI1-SCN2 often requires LOWER G-CSF doses than ELANE-SCN1 (milder arrest); "
            "  Target ANC >1.0 × 10⁹/L; minimise dose to lowest effective; "
            "  G-CSF sparing in milder GFI1-SCN2 patients (alternate-day dosing feasible); "
            "ANNUAL BM SURVEILLANCE: "
            "  AML risk lower (~8-12%) but present; annual BM + cytogenetics mandatory; "
            "  CSF3R truncation screening; "
            "HSCT: "
            "  Indicated for MDS/AML transformation or G-CSF non-response; "
            "  Less frequently needed than ELANE-SCN (milder phenotype generally); "
            "DENTAL/ORAL CARE: "
            "  Standard SCN dental protocol (biannual dental review, chlorhexidine); "
            "GENETIC COUNSELLING: "
            "  AD: 50% transmission; test first-degree relatives; "
            "  De novo rate: ~30-40% (significant de novo component); "
            "  GFI1 is a proto-oncogene; ELANE de-repression explains malignant risk; "
            "RESEARCH: "
            "  LSD1 inhibitors (iadademstat/seclidemstat): LSD1 is recruited by GFI1 to silence ELANE; "
            "  In GFI1-DN: LSD1 inhibition rationale less clear (LSD1 already not recruited); "
            "  Research ongoing in GFI1-null model (not DN)"
        ),
        "seed": 2827,
        "pt_vars": {
            "anc_nadir": (0.05, 0.40),
            "infection_per_year": (2.0, 8.0),
            "gcsf_dose_mcg_kg": (2, 15),
            "gcsf_response_pct": 93,
            "aml_risk_pct": 10,
        }
    },
    {
        "gene": "VPS45",
        "protein": (
            "VPS45 -- 1q21.2 AR -- 578aa -- Vacuolar-Protein-Sorting-45-"
            "65kDa-Sec1-Munc18-SM-Family-Lysosomal-Vesicle-Fusion-"
            "Bone-Marrow-Fibrosis-Nephromegaly-SCN5-"
            "OMIM-Gene-610035-Disease-SCN5-615285"
        ),
        "locus": "1q21.2",
        "protein_size": "578 aa / 65 kDa (Vacuolar protein sorting-45; Sec1/Munc18 (SM) family; regulates SNARE-mediated vesicle fusion in endosomal/lysosomal pathway; VPS45 promotes fusion of early endosomes and trans-Golgi network vesicles with late endosomes → lysosomal trafficking; VPS45 LOF → impaired lysosomal degradation → vacuolar pathology; very rare — ultra-rare SCN5; <30 reported cases worldwide)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF) — ultra-rare SCN5 (OMIM 615285); <30 families reported worldwide; "
            "MECHANISM (unique among SCN genes): "
            "  VPS45 regulates SNARE-mediated vesicle fusion: SYNTAXIN-16 (STX16) is key VPS45 partner; "
            "  VPS45 LOF → impaired STX16-mediated endosomal vesicle fusion → "
            "    → lysosomal trafficking defect → vacuolar accumulation → cell death; "
            "  Myeloid progenitors: very high lysosomal activity (granule biogenesis) → vulnerable to VPS45 LOF; "
            "  Fibroblast involvement: VPS45 also expressed in fibroblasts → "
            "    → fibroblast lysosomal dysfunction → extracellular matrix protein accumulation → FIBROSIS; "
            "UNIQUE MULTI-ORGAN FEATURES: "
            "  BONE MARROW FIBROSIS: progressive reticulin fibrosis on BM biopsy (not seen in other SCN genes); "
            "  NEPHROMEGALY: bilateral renal enlargement (vacuolar pathology in tubular cells); "
            "  EXTRA-HAEMATOPOIETIC MANIFESTATIONS UNIQUE TO VPS45: makes this a systemic lysosomal trafficking disorder, not just a neutropenia syndrome; "
            "SEVERITY: typically fatal in infancy/early childhood without HSCT (most severe SCN); "
            "G-CSF RESPONSE: POOR (BM fibrosis limits haematopoietic expansion — no space for progenitors); "
            "HSCT URGENTLY INDICATED: only effective treatment; fibrosis resolves post-HSCT"
        ),
        "disease_category": (
            "SEVERE CONGENITAL NEUTROPENIA TYPE 5 (SCN5) — OMIM 615285; "
            "ULTRA-RARE: <30 reported cases worldwide; "
            "MULTI-ORGAN DISEASE — UNIQUE AMONG SCN: "
            "  1. SEVERE NEUTROPENIA: ANC <0.1 × 10⁹/L; early onset (birth); life-threatening infections; "
            "  2. BONE MARROW FIBROSIS (PATHOGNOMONIC for VPS45-SCN): "
            "     Reticulin fibrosis on BM biopsy (MF-1 to MF-3 fibrosis); "
            "     NOT seen in ELANE/HAX1/G6PC3/WAS/CXCR4/GFI1 — specific to VPS45; "
            "     Fibrosis → failure of haematopoiesis → pancytopenia progresses; "
            "     BM aspirate often a dry tap (fibrosis); must do trephine biopsy; "
            "  3. NEPHROMEGALY: "
            "     Bilateral renal enlargement (echogenic on ultrasound); "
            "     Tubular vacuolation on biopsy; can progress to renal impairment; "
            "     Renal ultrasound mandatory at diagnosis + follow-up; "
            "  4. PROGRESSIVE CYTOPENIA: "
            "     Thrombocytopenia develops as BM fibrosis progresses; anaemia; pancytopenia; "
            "  5. FAILURE TO THRIVE: common (progressive multi-organ disease); "
            "PROGNOSIS WITHOUT HSCT: death in infancy/early childhood from infections + BM failure; "
            "DIAGNOSIS: BM biopsy (trephine — fibrosis) + VPS45 sequencing + renal ultrasound"
        ),
        "disease_pathway": (
            "VPS45 / SM-PROTEIN / SNARE VESICLE FUSION / LYSOSOMAL TRAFFICKING PATHWAY: "
            "NORMAL VESICLE FUSION (SM PROTEIN FAMILY): "
            "  SNARE proteins mediate membrane fusion: v-SNARE (vesicle) + t-SNARE (target membrane); "
            "  SM proteins (Sec1/Munc18 family) regulate SNARE assembly: "
            "    VPS45 specifically regulates: SYNTAXIN-16 (STX16) — early-to-late endosome fusion; "
            "    Also regulates: VPS21/RAB5 endosome maturation; TGN-to-endosome trafficking; "
            "VPS45 LOF CONSEQUENCE: "
            "  STX16 function impaired → "
            "  → Early endosome → late endosome fusion blocked → "
            "  → Lysosomal delivery of cargo (degradation substrates) impaired → "
            "  → Lysosomal enzyme delivery impaired → "
            "  → Vacuolar accumulation in myeloid progenitors → toxicity → apoptosis; "
            "FIBROSIS MECHANISM: "
            "  VPS45 LOF in fibroblasts → "
            "  → Impaired lysosomal degradation of extracellular matrix proteins → "
            "  → Collagen/fibronectin accumulation → BM stroma fibrosis; "
            "  Reticulin deposition replaces haematopoietic space; "
            "NEPHROMEGALY MECHANISM: "
            "  VPS45 LOF in renal tubular cells → "
            "  → Vacuolar accumulation in tubular epithelium → tubular enlargement → nephromegaly; "
            "CONTRAST WITH OTHER STORAGE DISEASES: "
            "  VPS45-SCN resembles lysosomal storage disorder (LSD) in mechanism but not in classification; "
            "  No enzyme substrate accumulation detectable (trafficking defect, not enzyme deficiency); "
            "POST-HSCT FIBROSIS RESOLUTION: "
            "  Donor HSC-derived myeloid cells (functional VPS45) populate BM → "
            "  → Fibrosis partially resolves over 6-12 months post-HSCT (if done early)"
        ),
        "pathognomonic": (
            "BONE MARROW FIBROSIS IN AN INFANT WITH SCN — PATHOGNOMONIC FOR VPS45: "
            "  BM biopsy showing reticulin fibrosis (MF-1 to MF-3) in infant with SCN → VPS45 first; "
            "  NO other SCN gene causes progressive BM fibrosis; "
            "  'Dry tap' on BM aspirate (fibrosis) in infant with neutropenia → always do trephine + VPS45 testing; "
            "NEPHROMEGALY ON RENAL ULTRASOUND + SCN: "
            "  Bilateral renal enlargement (nephromegaly) in infant with SCN → VPS45 virtually specific; "
            "  No other SCN gene causes renal enlargement; "
            "PROGRESSIVE PANCYTOPENIA + SCN IN FIRST YEAR: "
            "  SCN progressing to pancytopenia (thrombocytopenia + anaemia + neutropenia) → "
            "  BM fibrosis from VPS45 can explain progressive multi-lineage failure; "
            "G-CSF POOR RESPONSE WITH BM FIBROSIS: "
            "  SCN with expected G-CSF non-response on initial trial + BM fibrosis → VPS45 URGENT; "
            "  Most SCN respond to G-CSF (90-95%); failure + fibrosis = VPS45 until excluded; "
            "EARLY HSCT CRITICAL: "
            "  VPS45-SCN has highest mortality without HSCT; "
            "  Any infant with BM fibrosis + SCN: URGENT HSCT referral while donor search initiated"
        ),
        "treatment": (
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT) — URGENTLY INDICATED: "
            "  VPS45-SCN = most severe SCN; ONLY effective treatment; "
            "  URGENCY: BM fibrosis progresses rapidly → progressive BM failure → fatal without HSCT; "
            "  DONOR: matched sibling or unrelated donor (unrelated acceptable given urgency); "
            "  CONDITIONING: myeloablative (fibrotic BM requires ablation for engraftment); "
            "  POST-HSCT: BM fibrosis resolves over 6-12 months as donor cells repopulate stroma; "
            "  HSCT DOES NOT CURE RENAL PATHOLOGY: nephromegaly may persist / progress independently; "
            "  Renal monitoring mandatory even post-HSCT (nephrology co-management); "
            "G-CSF (BRIDGE TO HSCT ONLY): "
            "  Usually poor response (BM fibrosis limits response); "
            "  Use only as bridge to HSCT to reduce infection risk during donor search; "
            "  Do NOT delay HSCT for G-CSF trial in suspected VPS45-SCN; "
            "INFECTION PROPHYLAXIS AND MANAGEMENT: "
            "  Aggressive antibiotic prophylaxis (TMP-SMX, antifungal); "
            "  Isolation precautions; "
            "  Broad-spectrum antibiotics at first fever (high mortality from sepsis); "
            "RENAL MANAGEMENT: "
            "  Nephrology co-management from diagnosis; "
            "  Avoid nephrotoxic agents (aminoglycosides: HIGH CAUTION; NSAIDs: AVOID); "
            "  Hydration protocols; "
            "GENETIC COUNSELLING: AR; 25% sibling risk; consanguinity common in reported cases"
        ),
        "seed": 2828,
        "pt_vars": {
            "anc_nadir": (0.00, 0.10),
            "infection_per_year": (6.0, 18.0),
            "gcsf_dose_mcg_kg": (10, 30),
            "gcsf_response_pct": 35,
            "aml_risk_pct": 12,
        }
    },
    {
        "gene": "JAGN1",
        "protein": (
            "JAGN1 -- 3p25.3 AR -- 183aa -- Jagunal-Homolog-1-"
            "21kDa-ER-Membrane-Protein-N-Glycoprotein-Processing-"
            "Glycosylation-Quality-Control-SCN7-"
            "OMIM-Gene-616012-Disease-SCN7-617014"
        ),
        "locus": "3p25.3",
        "protein_size": "183 aa / 21 kDa (Jagunal homolog 1; ER membrane protein; regulates N-glycosylation quality control of neutrophil granule proteins; required for correct ER-Golgi trafficking of G-CSFR (CSF3R); JAGN1 LOF → G-CSFR glycosylation defect → impaired G-CSF signalling → myeloid differentiation arrest; also named DROSOPHILA jagunal ortholog; very rare — <25 families reported; AR biallelic required)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF) — ultra-rare SCN7 (OMIM 617014); <25 families worldwide; "
            "MECHANISM (glycosylation-dependent — distinct from all other SCN genes): "
            "  JAGN1 localises to ER membranes adjacent to ER exit sites; "
            "  Required for N-glycosylation and ER quality control of specific glycoproteins; "
            "  KEY TARGET: G-CSF receptor (CSF3R) N-glycosylation at ER requires JAGN1; "
            "    → JAGN1 LOF → CSF3R misglycosylated → impaired folding → ER retention → degradation; "
            "    → Cell surface G-CSFR reduced/absent → impaired G-CSF signalling → myeloid differentiation arrest; "
            "  ALSO: Neutrophil granule proteins (NE, MPO, lactoferrin) require JAGN1 for proper N-glycosylation; "
            "    → Granule protein misfolding → ER stress → apoptosis; "
            "  UNIQUE COMBINATION: both G-CSFR signalling impairment AND ER stress contribute; "
            "DISTINCTIVE FEATURE: "
            "  G-CSF POOR RESPONSE (mechanism: G-CSFR itself misglycosylated → impaired signalling even with G-CSF); "
            "  Contrast: ELANE/HAX1/GFI1 respond well to G-CSF (their G-CSFR is normal); "
            "  JAGN1 G-CSF resistance is mechanistically expected and diagnostically useful; "
            "CONSANGUINITY: high rate (AR); founder mutations described in specific populations; "
            "AML RISK: lower (~5-8%) given different pathobiology; "
            "SEVERITY: usually severe from birth; without HSCT → progressive infections, early death"
        ),
        "disease_category": (
            "SEVERE CONGENITAL NEUTROPENIA TYPE 7 (SCN7) — OMIM 617014; "
            "ULTRA-RARE: <25 families reported worldwide; identified 2014 (Boztug et al., Nature Genetics); "
            "HAEMATOLOGICAL FEATURES: "
            "  ANC: <0.1 × 10⁹/L (severe from birth); "
            "  BM: promyelocyte/myelocyte maturation arrest (similar to ELANE-SCN on BM smear); "
            "  Relative eosinophilia; monocytosis; "
            "  Thrombocytopenia: mild, intermittent; "
            "  Anaemia: mild normocytic; "
            "CLINICAL PRESENTATION: "
            "  Onset: birth / neonatal period; "
            "  Very severe infections from first days/weeks of life; "
            "  Omphalitis, umbilical sepsis, skin infections, pneumonia, septicaemia; "
            "  Oral ulcers; profound gingivitis; "
            "DISTINCTIVE FEATURES vs OTHER SCN: "
            "  G-CSF POOR RESPONSE (mechanism: G-CSFR misglycosylated); "
            "  No extra-haematopoietic anomalies (unlike G6PC3/HAX1/VPS45); "
            "  AR genetics → consanguinity common; "
            "DIAGNOSIS: "
            "  Persistent ANC <0.1; BM aspirate (maturation arrest); "
            "  G-CSF trial → poor response (clinical clue for JAGN1); "
            "  Broad SCN gene panel including JAGN1; "
            "  CSF3R surface expression by flow cytometry (reduced in JAGN1-SCN)"
        ),
        "disease_pathway": (
            "JAGN1 / N-GLYCOSYLATION QUALITY CONTROL / G-CSFR ER PROCESSING PATHWAY: "
            "N-GLYCOSYLATION IN ER: "
            "  N-glycosylation: attachment of oligosaccharide to asparagine (NxS/T sequon) in ER; "
            "  Required for: protein folding, stability, trafficking, receptor signalling; "
            "  G-CSFR (CSF3R): glycoprotein with multiple N-glycosylation sites → requires correct glycosylation "
            "    for proper folding in ER → transport to cell surface → G-CSF signalling; "
            "JAGN1 FUNCTION: "
            "  JAGN1 = ER-localised protein at ER exit sites; "
            "  Mechanism not fully elucidated but: "
            "    JAGN1 associates with OST (oligosaccharyltransferase) complex or CNX/CRT cycle; "
            "    Facilitates N-glycosylation quality control (correctly glycosylated proteins exit; misfolded retained); "
            "  Specific JAGN1 targets in neutrophil progenitors: CSF3R, NE, MPO, lactoferrin; "
            "JAGN1 LOF CONSEQUENCE: "
            "  CSF3R N-glycosylation defective → misfolded CSF3R → ER retention → degradation (ERAD); "
            "  → Reduced cell-surface G-CSFR → impaired G-CSF signalling → "
            "    → JAK2/STAT3/ERK activation reduced → myeloid differentiation arrest; "
            "  Granule protein misfolding (NE, MPO) → ER stress → UPR → apoptosis; "
            "  DUAL MECHANISM: G-CSFR signalling defect + ER stress both contribute to neutropenia; "
            "CSF3R FLOW CYTOMETRY DIAGNOSTIC USE: "
            "  Neutrophil/progenitor surface CSF3R expression by flow cytometry; "
            "  JAGN1-SCN: reduced surface CSF3R → suggests glycosylation trafficking defect; "
            "  May guide to JAGN1 sequencing before full panel"
        ),
        "pathognomonic": (
            "SEVERE SCN WITH G-CSF POOR RESPONSE AND NO ELANE/HAX1/G6PC3/WAS/CXCR4/GFI1 MUTATION: "
            "  G-CSF non-response in SCN infant → test VPS45 and JAGN1 (both resist G-CSF by mechanism); "
            "  JAGN1: G-CSFR itself misglycosylated → G-CSF has impaired target → poor ANC response; "
            "REDUCED CSF3R SURFACE EXPRESSION BY FLOW CYTOMETRY: "
            "  Low surface G-CSFR in SCN patient → JAGN1-SCN specific finding; "
            "  Confirms G-CSFR glycosylation trafficking defect; "
            "CONSANGUINEOUS AR-SCN WITH NO OTHER GENE: "
            "  Consanguineous family with SCN, AR inheritance, negative for ELANE/HAX1/G6PC3/VPS45 → "
            "  JAGN1 and other ultra-rare SCN genes warranted; "
            "SEVERE NEONATAL SEPSIS + ABSENT NEUTROPHILS FROM DAY 1: "
            "  Omphalitis + complete absence of neutrophils in first week of life → "
            "  ELANE most common; if ELANE negative and consanguineous → HAX1/JAGN1/VPS45; "
            "NORMAL BM MATURATION ARREST PATTERN (PROMYELOCYTE) with G-CSF RESISTANCE: "
            "  BM arrest same as ELANE but G-CSF does not rescue → JAGN1 or VPS45; "
            "  VPS45 has fibrosis; JAGN1 has no fibrosis → BM biopsy distinguishes them"
        ),
        "treatment": (
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT) — TREATMENT OF CHOICE: "
            "  Given G-CSF poor response, HSCT is the most effective long-term treatment; "
            "  INDICATION: SCN7 with G-CSF non-response → proceed to HSCT evaluation; "
            "  Outcomes: improving with modern transplant protocols; OS ~70-80% in recent series; "
            "  TIMING: early HSCT (first 2 years of life) preferred; avoid prolonged severe neutropenia; "
            "G-CSF (BRIDGE THERAPY): "
            "  Try G-CSF despite expected poor response — some partial responses reported; "
            "  Use highest tolerated dose (up to 50-100 mcg/kg/day explored in ultra-severe cases); "
            "  HSCT NOT delayed waiting for G-CSF response in JAGN1-confirmed cases; "
            "  G-CSF reduces infection frequency even with partial ANC improvement; "
            "INFECTION MANAGEMENT: "
            "  Aggressive prophylaxis: TMP-SMX (PCP + bacterial); azole antifungal; "
            "  Hospitalise and treat with IV antibiotics at first fever; "
            "  Consider granulocyte transfusions during life-threatening infections as bridge; "
            "ANNUAL BM SURVEILLANCE: "
            "  Even with HSCT, MDS risk monitoring; "
            "  For G-CSF-managed patients: annual cytogenetics + CSF3R truncation; "
            "GENETIC COUNSELLING: AR; 25% sibling risk; consanguinity common; prenatal diagnosis available"
        ),
        "seed": 2829,
        "pt_vars": {
            "anc_nadir": (0.00, 0.12),
            "infection_per_year": (5.0, 16.0),
            "gcsf_dose_mcg_kg": (10, 50),
            "gcsf_response_pct": 40,
            "aml_risk_pct": 7,
        }
    },
]

DEFINITIONS = {
    "definitions": [
        {
            "term": "Severe Congenital Neutropenia (SCN)",
            "definition": (
                "Persistent ANC <0.5 × 10⁹/L (usually <0.2) from birth caused by hereditary mutations; "
                "Maturation arrest in bone marrow (promyelocyte/myelocyte stage — most SCN genes); "
                "Presenting symptoms: recurrent bacterial infections, omphalitis, oral ulcers, absent pus formation; "
                "Major genes: ELANE (SCN1), HAX1 (SCN3), G6PC3 (SCN4), WAS/XLN, VPS45 (SCN5), JAGN1 (SCN7); "
                "ELANE cyclic neutropenia (CN1): periodic ANC nadir every 21 days; "
                "G-CSF response: 90-95% for most SCN genes (exception: VPS45 and JAGN1); "
                "AML/MDS transformation: 15-25% lifetime (highest in ELANE/HAX1/XLN-WAS)"
            )
        },
        {
            "term": "Absolute Neutrophil Count (ANC)",
            "definition": (
                "ANC = WBC × (% neutrophils + % bands) / 100; "
                "Normal range: 1.5-8.0 × 10⁹/L (adults); 1.0-8.5 × 10⁹/L (children); "
                "Mild neutropenia: 1.0-1.5; Moderate: 0.5-1.0; Severe: <0.5; Very severe: <0.1; "
                "SCN: ANC consistently <0.5 from birth; CN: ANC <0.5 at nadir (every 21 days); "
                "Infection risk: substantially increased below 0.5; very high below 0.1"
            )
        },
        {
            "term": "Myelokathexis",
            "definition": (
                "Pathological BM finding specific to WHIM syndrome (CXCR4-GOF); "
                "Hypersegmented (4-10 lobed), pyknotic, vacuolated neutrophils in bone marrow; "
                "FULL maturation present — neutrophils ARE made but cannot exit BM (egress failure); "
                "Key distinction from SCN maturation arrest: myelokathexis = mature cells retained; "
                "arrest = immature cells accumulating; both cause peripheral neutropenia by different mechanisms; "
                "Plerixafor (CXCR4 antagonist) mobilises myelokathexis neutrophils within hours"
            )
        },
        {
            "term": "Maturation Arrest (Promyelocyte Stage)",
            "definition": (
                "Bone marrow finding in ELANE/HAX1/G6PC3/GFI1/JAGN1 SCN: "
                "Abundant promyelocytes + early myelocytes in BM; virtually absent later stages (metamyelocytes, bands, segs); "
                "Mechanism: ER stress/apoptosis occurs at promyelocyte stage (when NE/granule proteins first expressed); "
                "Eosinophilia: relative eosinophilia common (eosinophil maturation less affected); "
                "Monocytosis: compensatory; "
                "Contrast with myelokathexis (CXCR4): all stages present including mature segments"
            )
        },
        {
            "term": "G-CSF / Filgrastim",
            "definition": (
                "Granulocyte colony-stimulating factor; recombinant human G-CSF; "
                "Mechanism: binds G-CSFR (CSF3R) → JAK2/STAT3/ERK → myeloid proliferation + differentiation; "
                "Drives committed progenitors past maturation arrest; "
                "SCN dosing: 5-20 mcg/kg/day SC; titrate to ANC target >1.0 × 10⁹/L; "
                "AML risk: high-dose G-CSF (>20 mcg/kg/day) associated with higher AML/MDS risk; minimise dose; "
                "G-CSF poor response: VPS45 (BM fibrosis), JAGN1 (G-CSFR misglycosylated) → proceed to HSCT"
            )
        },
        {
            "term": "Cyclic Neutropenia (CN)",
            "definition": (
                "Periodic ANC nadir every 21 days (±3 days); nadir lasts 3-6 days; "
                "Caused by ELANE mutations that dysregulate haematopoietic oscillator (different mutations from SCN-ELANE); "
                "Diagnosis: 3×/week CBC for 6-8 weeks to document periodicity; "
                "Nadir symptoms: oral aphthae, fever, cervical lymphadenopathy; "
                "G-CSF: shortens nadir duration; does NOT abolish cyclicity; "
                "AML risk: 1-3% (much lower than SCN); cyclic neutropenia is NOT a pre-malignant condition"
            )
        },
        {
            "term": "AML/MDS Transformation Risk in SCN",
            "definition": (
                "Cumulative lifetime AML/MDS risk: ELANE-SCN ~15-25%; HAX1-SCN ~15-20%; XLN-WAS >30%; "
                "CXCR4-WHIM ~10%; GFI1-SCN2 ~8-12%; G6PC3 ~10-15%; VPS45 ~12%; JAGN1 ~5-8%; "
                "Mechanism: acquired CSF3R truncating mutations (clonal myeloid evolution) → AML; "
                "Prevention: minimise G-CSF dose; annual BM surveillance with cytogenetics; "
                "HSCT: curative and eliminates AML risk; indication for HSCT includes early CSF3R truncation"
            )
        },
        {
            "term": "WHIM Syndrome (CXCR4)",
            "definition": (
                "Warts + Hypogammaglobulinaemia + Infections + Myelokathexis; "
                "GOF C-terminal truncations of CXCR4 → impaired desensitisation → BM neutrophil retention; "
                "Unique: full BM maturation (myelokathexis, not arrest); HPV susceptibility; "
                "Plerixafor (CXCR4 antagonist) = targeted therapy; FDA-approved 2024; "
                "Newfoundland kindred (p.R334X): largest described family with WHIM syndrome"
            )
        },
        {
            "term": "Kostmann Syndrome (HAX1)",
            "definition": (
                "Original autosomal recessive SCN described by Rolf Kostmann (1956) in Swedish families; "
                "HAX1 gene identified 2007 (Klein et al.); Kostmann's original families have HAX1 mutations; "
                "HAX1 = anti-apoptotic mitochondrial/ER adaptor; LOF → myeloid progenitor apoptosis; "
                "HAX1-II isoform disruption → neurological phenotype (intellectual disability + epilepsy); "
                "p.Trp44X: disrupts both isoforms → neutropenia + neurology; "
                "p.Gln190X: disrupts HAX1-I only → neutropenia alone; "
                "HSCT cures haematological but not neurological phenotype"
            )
        },
        {
            "term": "Wiskott-Aldrich Syndrome Protein (WASp)",
            "definition": (
                "Haematopoietic-specific Arp2/3 activator → actin nucleation; "
                "WAS (LOF null): microthrombocytopenia + eczema + combined immunodeficiency; "
                "XLN (GOF): isolated severe congenital neutropenia (actin hyperpolymerisation); "
                "Small platelets (MPV <7 fL) + thrombocytopenia in male infant → test WASp flow cytometry; "
                "WASp expression by flow: absent in WAS; present/elevated in XLN (GOF); "
                "HSCT curative for WAS; gene therapy clinical trials for WAS; "
                "XLN: urgent HSCT (highest AML risk of all SCN genes >30%)"
            )
        },
        {
            "term": "Bone Marrow Surveillance in SCN",
            "definition": (
                "Annual BM aspirate + cytogenetics + CSF3R truncation mutation analysis; "
                "Indication: all SCN patients on G-CSF; "
                "Purpose: detect early AML/MDS transformation; "
                "CSF3R truncation (d715, T617I): early clonal evolution → intensify surveillance or HSCT; "
                "Monosomy 7 / del(7q): high AML risk → urgent HSCT evaluation; "
                "Exception: VPS45-SCN should have HSCT urgently — BM aspirate may be 'dry tap'"
            )
        },
        {
            "term": "Plerixafor (AMD3100, Mozobil)",
            "definition": (
                "Bicyclam CXCR4 antagonist; blocks CXCL12-CXCR4 interaction; "
                "Normal use: HSC mobilisation for stem cell collection (combined with G-CSF); "
                "WHIM syndrome: specifically reverses BM neutrophil retention → ANC normalises within hours; "
                "Dose for WHIM: 0.24 mg/kg SC; daily or alternate-day; "
                "FDA approved for SCN/WHIM 2024; first targeted therapy for CXCR4 gain-of-function neutropenia; "
                "Dramatic ANC response is pathognomonic for WHIM when seen"
            )
        },
    ],
    "standards": [
        "Severe Chronic Neutropenia International Registry (SCNIR) — University of Washington",
        "European Bone Marrow Transplantation (EBMT) — SCN Guidelines",
        "BSH Guidelines: Investigation and Management of Heritable Neutropenias (2020)",
        "OMIM: SCN1 202700 (ELANE), Kostmann/SCN3 610738 (HAX1), Dursun/SCN4 612541 (G6PC3), WAS 300100, XLN 300299 (WAS-GOF), WHIM 193670 (CXCR4), SCN2 613107 (GFI1), SCN5 615285 (VPS45), SCN7 617014 (JAGN1)",
        "Dale DC et al. Severe Chronic Neutropenia: Treatment and Follow-up of Patients in the US and Canada. J Pediatr. 2003",
        "Boztug K et al. Stem-Cell Gene Therapy for the Wiskott-Aldrich Syndrome. NEJM. 2010",
        "McDermott DH et al. Plerixafor for WHIM Syndrome. NEJM. 2019",
        "Boztug K et al. JAGN1 deficiency causes aberrant myeloid cell homeostasis and congenital neutropenia. Nature Genetics. 2014",
        "ESID (European Society for Immunodeficiencies) — PID Registry SCN module",
        "Masre SF et al. Hereditary Severe Congenital Neutropenia — Comprehensive Review. Front Immunol. 2022",
    ]
}


def _make_patients(gene_data):
    rng = random.Random(gene_data["seed"])
    pts = []
    anc_lo, anc_hi = gene_data["pt_vars"]["anc_nadir"]
    inf_lo, inf_hi = gene_data["pt_vars"]["infection_per_year"]
    dose_lo, dose_hi = gene_data["pt_vars"]["gcsf_dose_mcg_kg"]
    gcsf_resp_pct = gene_data["pt_vars"]["gcsf_response_pct"]
    aml_risk = gene_data["pt_vars"]["aml_risk_pct"]

    for i in range(40):
        anc = round(rng.uniform(anc_lo, anc_hi), 3)
        inf_rate = round(rng.uniform(inf_lo, inf_hi), 1)
        dose = round(rng.uniform(dose_lo, dose_hi), 1)
        age_diag = round(rng.uniform(0.0, 1.5), 2)
        gcsf_responds = rng.random() < gcsf_resp_pct / 100
        aml_event = rng.random() < aml_risk / 100
        hsct = rng.random() < 0.22

        # XLN (WAS GOF): male-dominant; WHIM: warts present
        if gene_data["gene"] == "WAS":
            sex = "M"  # XLN/WAS hemizygous males in this cohort
        elif gene_data["gene"] == "CXCR4":
            sex = rng.choice(["M", "F"])  # WHIM AD — both sexes
        else:
            sex = rng.choice(["M", "F"])

        warts_present = gene_data["gene"] == "CXCR4" and rng.random() < 0.75
        bm_fibrosis = gene_data["gene"] == "VPS45" and rng.random() < 0.85
        neuro_phenotype = gene_data["gene"] == "HAX1" and rng.random() < 0.40

        pts.append({
            "patient_id": f"{gene_data['gene']}-{i+1:03d}",
            "gene": gene_data["gene"],
            "sex": sex,
            "age_at_diagnosis_years": age_diag,
            "anc_nadir_per_uL": anc,
            "infections_per_year": inf_rate,
            "gcsf_dose_mcg_kg": dose,
            "gcsf_response": gcsf_responds,
            "aml_mds_event": aml_event,
            "hsct_performed": hsct,
            "warts_present": warts_present,
            "bm_fibrosis": bm_fibrosis,
            "neurological_phenotype": neuro_phenotype,
            "seed": gene_data["seed"],
        })
    return pts


def generate_overview():
    total_patients = 0
    all_genes = []

    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        total_patients += len(patients)
        gcsf_resp = sum(1 for p in patients if p["gcsf_response"])
        aml_ev = sum(1 for p in patients if p["aml_mds_event"])
        hsct = sum(1 for p in patients if p["hsct_performed"])
        warts = sum(1 for p in patients if p["warts_present"])
        fibrosis = sum(1 for p in patients if p["bm_fibrosis"])
        neuro = sum(1 for p in patients if p["neurological_phenotype"])

        all_genes.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein": gene_data["protein"],
            "disease_category": gene_data["disease_category"],
            "inheritance": gene_data["inheritance"],
            "n_patients": len(patients),
            "median_anc_nadir": round(
                sorted(p["anc_nadir_per_uL"] for p in patients)[len(patients) // 2], 3
            ),
            "mean_infections_per_year": round(
                sum(p["infections_per_year"] for p in patients) / len(patients), 1
            ),
            "pct_gcsf_response": round(gcsf_resp / len(patients) * 100, 1),
            "pct_aml_mds": round(aml_ev / len(patients) * 100, 1),
            "pct_hsct": round(hsct / len(patients) * 100, 1),
            "pct_warts": round(warts / len(patients) * 100, 1),
            "pct_bm_fibrosis": round(fibrosis / len(patients) * 100, 1),
            "pct_neuro_phenotype": round(neuro / len(patients) * 100, 1),
            "seed": gene_data["seed"],
        })

    return {
        "atlas": "Hereditary Congenital Neutropenia Atlas",
        "subtitle": (
            "Complete 8-Gene SCN / WHIM / Kostmann / WAS-XLN Reference — "
            "ELANE·HAX1·G6PC3·WAS·CXCR4·GFI1·VPS45·JAGN1"
        ),
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "gene_summaries": all_genes,
        "seeds": "2822-2829",
        "pathway_categories": [
            {
                "pathway": "ER Stress / UPR / Neutrophil Elastase Axis",
                "genes": ["ELANE", "HAX1", "GFI1"],
                "note": (
                    "ELANE: misfolded NE → UPR → CHOP → apoptosis at promyelocyte stage (direct); "
                    "HAX1: anti-apoptotic mitochondrial/ER adaptor; LOF → caspase-9 → apoptosis; "
                    "GFI1: dominant-negative → ELANE de-repressed → same ER stress as ELANE mutations; "
                    "All three converge on promyelocyte ER stress apoptosis; G-CSF responsive (90-95%)"
                ),
            },
            {
                "pathway": "ER Protein Glycosylation / Trafficking Defects",
                "genes": ["G6PC3", "JAGN1"],
                "note": (
                    "G6PC3: ER glucose depletion → N-glycosylation substrate reduced → misfolded glycoproteins → ER stress; "
                    "JAGN1: N-glycosylation quality control defect → CSF3R misglycosylated → G-CSFR impaired; "
                    "Both: EXTRA-HAEMATOPOIETIC manifestations (G6PC3: cardiac+urogenital+ear; JAGN1: G-CSF resistance); "
                    "G6PC3: multi-organ anomalies (Dursun syndrome); JAGN1: ultra-rare, G-CSF poor response"
                ),
            },
            {
                "pathway": "CXCL12/CXCR4 Chemokine Axis — Neutrophil Egress Failure",
                "genes": ["CXCR4"],
                "note": (
                    "UNIQUE MECHANISM: NOT maturation arrest; FULL neutrophil maturation but BM egress blocked; "
                    "CXCR4-GOF (C-terminal truncation) → no receptor desensitisation → hyperactivated CXCL12 retention; "
                    "Myelokathexis on BM (hypersegmented pyknotic neutrophils, NOT arrest); "
                    "Plerixafor (CXCR4 antagonist) = specific targeted therapy; FDA-approved 2024"
                ),
            },
            {
                "pathway": "Actin Cytoskeleton / WASp-Arp2/3 Pathway (Haematopoietic)",
                "genes": ["WAS"],
                "note": (
                    "WAS-LOF (null): microthrombocytopenia + eczema + combined immunodeficiency (Wiskott-Aldrich); "
                    "WAS-GOF (XLN): constitutive Arp2/3 activation → actin hyperpolymerisation → neutrophil differentiation arrest; "
                    "Only SCN gene with X-linked inheritance; Only gene with THROMBOCYTOPENIA as concurrent feature; "
                    "XLN = highest AML risk (>30%); urgent HSCT; gene therapy available for classic WAS"
                ),
            },
            {
                "pathway": "Lysosomal Vesicle Trafficking (SM-Protein Pathway)",
                "genes": ["VPS45"],
                "note": (
                    "UNIQUE SYSTEMIC DISEASE: lysosomal trafficking defect → "
                    "BM fibrosis (PATHOGNOMONIC) + nephromegaly + progressive pancytopenia; "
                    "G-CSF POOR RESPONSE (BM fibrosis limits expansion space); "
                    "Most severe SCN — fatal without HSCT; urgent HSCT is only effective treatment; "
                    "VPS45 is the only SCN gene causing BM fibrosis"
                ),
            },
        ],
        "critical_distinctions": [
            "ELANE (SCN1) vs ELANE (CN1): SAME gene — DIFFERENT mutation types → SCN (persistent <0.2) vs Cyclic (21-day cycle); maturation arrest same BM; CN has lower AML risk (1-3%)",
            "MYELOKATHEXIS (WHIM/CXCR4) vs MATURATION ARREST (all other SCN): CXCR4 BM = full maturation with pyknotic retained neutrophils; ALL other SCN = promyelocyte arrest; CXCR4 = egress failure, not arrest",
            "WAS (LOF null) vs XLN (WAS-GOF): LOF = classic triad (thrombocytopenia + eczema + immunodeficiency); GOF = isolated neutropenia ONLY (no thrombocytopenia!); WASp flow cytometry: absent (LOF) vs present (GOF)",
            "SMALL PLATELETS (MPV <7fL) + THROMBOCYTOPENIA in male infant: WAS until excluded; ITP = large platelets (high MPV); the MPV distinguishes WAS from ITP immediately",
            "G6PC3 vs GSD-1 (G6PC): G6PC3 = neutropenia + cardiac/urogenital/ear anomalies + NO hypoglycaemia; G6PC (GSD-1a) = hepatomegaly + severe hypoglycaemia + NO neutropenia phenotype; completely different syndromes",
            "VPS45 vs ELANE BM: both promyelocyte arrest on aspirate; VPS45 ALSO has reticulin fibrosis on biopsy (dry tap); G-CSF fails in VPS45 (fibrosis); ELANE responds well",
            "JAGN1 vs VPS45 (both G-CSF poor responders): JAGN1 = no BM fibrosis (just arrest); VPS45 = BM fibrosis; BM biopsy fibrosis = VPS45; no fibrosis + G-CSF resistance = JAGN1 or other ultra-rare",
            "HAX1 ISOFORM CRITICAL: HAX1-p.Trp44X disrupts both isoforms → SCN + intellectual disability + epilepsy; HAX1-p.Gln190X disrupts HAX1-I only → SCN without neurology; ALWAYS classify HAX1 mutation by isoform impact at diagnosis",
            "AML RISK RANKING: XLN-WAS (>30%) > ELANE-SCN (~20%) ≈ HAX1-SCN (~18%) > G6PC3 (~13%) > VPS45 (~12%) > CXCR4-WHIM (~10%) ≈ GFI1 (~10%) > JAGN1 (~7%); cyclic neutropenia (ELANE-CN) = 1-3%",
            "PLERIXAFOR ANC RISE PATHOGNOMONIC FOR WHIM: ANC 0.1 → 5.0+ within hours of plerixafor = CXCR4-GOF WHIM until excluded; no other SCN syndrome shows this immediate dramatic response",
        ],
    }


def generate_breakdown():
    result = []
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        result.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein": gene_data["protein"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "n_patients": len(patients),
            "patients": patients[:5],
        })
    return {"genes": result, "total": len(ATLAS_GENES), "seeds": "2822-2829"}


def generate_definitions():
    return {
        "atlas": "Hereditary Congenital Neutropenia Atlas",
        "definitions": DEFINITIONS["definitions"],
        "standards": DEFINITIONS["standards"],
        "gene_count": len(ATLAS_GENES),
        "seeds": "2822-2829",
    }
