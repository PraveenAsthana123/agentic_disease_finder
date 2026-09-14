"""Hereditary Biopterin (BH4) Metabolism Atlas — 8-Gene Reference
GCH1-PTS-QDPR-PCBD1-SPR-DNAJC12-TH-DDC
320 patients (8 x 40), seeds 2670-2677.
Endpoints: /api/hereditary-biopterin-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "GCH1",
        "protein": (
            "GCH1 -- 14q22.2 AR/AD -- 252aa -- GTP-Cyclohydrolase-I-GTPCH-I-28kDa-Homodecamer-Mitochondrial-"
            "Rate-Limiting-BH4-Synthesis-Enzyme-AR-Severe-BH4-Deficiency-HPA-AD-DRD-Segawa-Disease-GOF-Feedback-"
            "OMIM-Gene-600225-AR-Disease-233910-AD-Disease-128230"
        ),
        "locus": "14q22.2",
        "protein_size": "252 aa / 28 kDa (homodecamer)",
        "inheritance": (
            "DUAL INHERITANCE — same gene, two completely different diseases: "
            "AR (biallelic LOF): severe BH4 deficiency with hyperphenylalaninaemia (HPA); "
            "  GCH1 loss → cannot synthesise BH4 → PAH inactive → phenylalanine accumulates + "
            "  TH/TPH inactive → dopamine + serotonin deficient; "
            "  COMBINED defect: elevated Phe + low CSF neurotransmitters; "
            "  treatment: BH4 (sapropterin) + neurotransmitters (L-DOPA + 5-HTP) + phenylalanine restriction; "
            "AD (monoallelic LOF): DOPA-responsive dystonia (DRD, Segawa disease); "
            "  ONE functional allele sufficient for BH4 synthesis BUT insufficient for TH under high neuronal demand; "
            "  CRITICAL MECHANISM: AR-LOF completely ablates GTPCH-I; AD-LOF uses GTPCH-I feedback regulation: "
            "  GTPCH-I is inhibited by BH4 via GTPCH-I Feedback Regulatory Protein (GFRP); "
            "  AD-LOF → haploinsufficiency → reduced GTPCH-I → borderline BH4 → TH activity marginal; "
            "  DIURNAL FLUCTUATION PATHOGNOMONIC: dopamine-dependent striatal circuits worse in evening "
            "  (BH4 reserves depleted by daytime neuronal activity) → dystonia worse PM, better AM; "
            "  L-DOPA DRAMATIC RESPONSE: near-complete remission at low doses (hallmark); "
            "  Phe NORMAL in AD-DRD (BH4 borderline but sufficient for hepatic PAH); "
            "  AUTOSOMAL DOMINANT — family history in ~50%; de novo AD-DRD cases also occur"
        ),
        "disease_category": (
            "AR: Severe BH4 deficiency type I (GTPCH-I deficiency) — OMIM 233910; "
            "  COMBINED HPA + neurotransmitter deficiency: elevated Phe 600-1800 µmol/L; "
            "  CSF HVA low + CSF 5-HIAA low; biopterin VERY LOW in urine; "
            "  neopterin/biopterin ratio ELEVATED (upstream metabolite accumulates); "
            "  NBS: detected by elevated phenylalanine on first-tier NBS; "
            "  BH4 loading test: sapropterin 20 mg/kg → Phe drops >30% at 4-8h (differentiates BH4-responsive PA from PKU); "
            "  DISTINGUISH from PKU: biopterin + neopterin urine pterin profile mandatory for ALL NBS-positive HPA; "
            "AD: DOPA-Responsive Dystonia, Segawa disease — OMIM 128230; "
            "  typically presents in childhood (2-12 years) with lower-limb dystonia; "
            "  classic diurnal fluctuation: worst in afternoon/evening, improved after sleep; "
            "  may mimic cerebral palsy — DRD IS TREATABLE; must exclude in all atypical CP; "
            "  Phe NORMAL (NBS negative); pterin profile may be borderline; "
            "  CSF pterins: neopterin and biopterin both mildly reduced (in contrast to AR: very low biopterin); "
            "  CSF HVA mildly reduced (dopamine metabolite)"
        ),
        "disease_pathway": (
            "GCH1/GTPCH-I BH4 SYNTHESIS PATHWAY: "
            "GTP → (GCH1/GTPCH-I) → 7,8-dihydroneopterin triphosphate → (PTS/PTPS) → 6-pyruvoyltetrahydropterin → "
            "(SPR/sepiapterin reductase) → BH4 (tetrahydrobiopterin); "
            "BH4 is an ESSENTIAL COFACTOR for: "
            "  PAH (phenylalanine hydroxylase) — hepatic Phe catabolism; "
            "  TH (tyrosine hydroxylase) — rate-limiting step dopamine synthesis in brain; "
            "  TPH1/TPH2 (tryptophan hydroxylase) — rate-limiting step serotonin synthesis; "
            "  NOS1/2/3 (nitric oxide synthases) — NO synthesis; "
            "GCH1 FEEDBACK REGULATION: "
            "  GFRP (GCH1 Feedback Regulatory Protein, encoded by GCHFR): "
            "  BH4 binds GFRP → GFRP binds GTPCH-I homodecamer → inhibits GTPCH-I activity; "
            "  Phe binds GFRP → GFRP activates GTPCH-I (opposite signal — stimulates BH4 when Phe high); "
            "  AD-LOF: haploinsufficiency → reduced GTPCH-I dimer partners → "
            "  even modest GFRP inhibition (from baseline BH4) brings TH activity below threshold; "
            "  DIURNAL: daytime dopamine turnover depletes BH4 → worsened TH activity → more dystonia by evening; "
            "  sleep restores BH4 → improved AM; "
            "AR-LOF: complete GTPCH-I absence → GTP cannot enter pterin pathway → BH4 = 0 → "
            "  ALL BH4-dependent enzymes fail → combined HPA + dopamine/serotonin deficiency"
        ),
        "pathognomonic": (
            "GCH1 DIAGNOSTIC CLUSTER: "
            "AR-FORM: "
            "1) HPA: elevated Phe ≥600 µmol/L on NBS + urine pterins: VERY LOW biopterin + HIGH neopterin/biopterin ratio; "
            "2) CSF NEUROTRANSMITTERS: HVA low + 5-HIAA low (combined dopamine + serotonin deficiency) — MANDATORY in all BH4 deficiency; "
            "3) BH4 LOADING TEST: sapropterin 20 mg/kg → >30% Phe reduction at 4-8h; "
            "4) MOVEMENT DISORDER: parkinsonism, dystonia, oculomotor abnormalities in first year; "
            "AD-FORM (DRD): "
            "1) DIURNAL FLUCTUATION OF DYSTONIA: worse evening/afternoon, better after sleep — PATHOGNOMONIC; "
            "2) L-DOPA DRAMATIC RESPONSE: near-complete remission at low doses (0.5-2 mg/kg/day); "
            "3) ONSET CHILDHOOD: lower-limb dystonia, foot deformity, gait abnormality; "
            "4) Phe NORMAL: NBS negative — not detected by standard NBS; "
            "5) CSF pterins: neopterin + biopterin mildly reduced; CSF HVA mildly reduced; "
            "6) FAMILY HISTORY: autosomal dominant — 50% of parents affected; "
            "CRITICAL: L-DOPA trial MANDATORY in any child with dystonia of unknown cause — "
            "DRD is fully treatable; misdiagnosis as cerebral palsy = preventable tragedy; "
            "OCULOGYRIC CRISES: 30% of AR-GCH1 — upward eye deviation, forced sustained; "
            "MOLECULAR: GCH1 sequencing (gene panel with PTS, QDPR, PCBD1, SPR)"
        ),
        "treatment": (
            "GCH1/GTPCH-I TREATMENT: "
            "AR-FORM: "
            "SAPROPTERIN (BH4): 5-20 mg/kg/day — replaces deficient BH4; activates residual PAH + TH + TPH; "
            "L-DOPA/CARBIDOPA: 1-5 mg/kg/day — replaces deficient dopamine; carbidopa 25% of L-DOPA dose; "
            "5-HYDROXYTRYPTOPHAN (5-HTP): 2-8 mg/kg/day — replaces deficient serotonin (AADC substrate); "
            "CARBIDOPA given WITH 5-HTP to prevent peripheral decarboxylation; "
            "PHENYLALANINE RESTRICTION: low-Phe formula diet — adjunct to BH4; "
            "FOLINIC ACID: 5-15 mg/day — secondary folate deficiency possible; "
            "AD-FORM (DRD): "
            "L-DOPA/CARBIDOPA: DRAMATIC RESPONSE at low doses (hallmark); "
            "start 0.5-1 mg/kg/day L-DOPA → titrate to response; "
            "lifelong maintenance — near-complete remission in most; "
            "5-HTP: not routinely required for AD-DRD (serotonin preserved in AD haploinsufficiency); "
            "SAPROPTERIN: 5-10 mg/kg/day — may supplement for BH4 marginal cases; "
            "MONITORING: CSF neurotransmitters annually; Phe monitoring (AR only); "
            "OCULOGYRIC CRISES: benzodiazepines for acute; adjust neurotransmitter therapy long-term"
        ),
    },
    {
        "gene": "PTS",
        "protein": (
            "PTS -- 11q23.1 AR -- 145aa -- 6-Pyruvoyltetrahydropterin-Synthase-PTPS-16kDa-Homotrimer-"
            "Second-Step-BH4-Synthesis-Most-Common-BH4-Deficiency-60-75pct-Central-vs-Peripheral-Subtype-"
            "OMIM-Gene-612719-Disease-261640"
        ),
        "locus": "11q23.1",
        "protein_size": "145 aa / 16 kDa (homotrimer)",
        "inheritance": (
            "AR (biallelic LOF); PTS encodes 6-pyruvoyltetrahydropterin synthase (PTPS) — step 2 of BH4 synthesis; "
            "MOST COMMON BH4 deficiency: 60-75% of all BH4-deficient HPA; "
            "CRITICAL SUBTYPE DISTINCTION — determines treatment: "
            "PERIPHERAL (mild) PTS deficiency: ~30% of PTS cases; "
            "  urine pterins: LOW biopterin + HIGH neopterin (upstream accumulation) — typical BH4 deficiency pattern; "
            "  CSF neurotransmitters: NORMAL HVA and 5-HIAA (residual BH4 sufficient for neuronal TH/TPH); "
            "  treatment: sapropterin + Phe restriction ONLY (no neurotransmitters needed); "
            "CENTRAL (severe) PTS deficiency: ~70% of PTS cases; "
            "  SAME urine pterin profile as peripheral; "
            "  CSF neurotransmitters: LOW HVA + LOW 5-HIAA (CNS BH4 insufficient for TH/TPH); "
            "  treatment: sapropterin + L-DOPA + 5-HTP + Phe restriction; "
            "CRITICAL: CSF neurotransmitters are MANDATORY to distinguish — "
            "urine pterins CANNOT distinguish central from peripheral PTS deficiency; "
            "p.Arg16Cys most common in Asian populations; p.Pro87Ser common in European"
        ),
        "disease_category": (
            "BH4 deficiency type IV (6-pyruvoyltetrahydropterin synthase deficiency) — OMIM 261640; "
            "ALL PTS patients: elevated Phe on NBS (HPA); "
            "urine/blood pterins: VERY LOW biopterin + HIGH neopterin; "
            "BH4 loading test: >30% Phe reduction — confirms BH4 responsiveness; "
            "PERIPHERAL SUBTYPE: HPA controlled by BH4/diet alone; no neurological deterioration if treated; "
            "CENTRAL SUBTYPE: if neurotransmitters not replaced → progressive neurological impairment; "
            "seizures, movement disorder, hypotonia, intellectual disability; "
            "OUTCOME: peripheral PTS — excellent prognosis with BH4/diet; "
            "central PTS — good prognosis if neurotransmitter replacement started early; "
            "NBS DETECTION: ALL PTS patients detected by elevated Phe; however, subtype only known after CSF; "
            "INCIDENCE: most common BH4 deficiency (~1:500,000-1,000,000 overall)"
        ),
        "disease_pathway": (
            "PTS/PTPS BH4 SYNTHESIS — STEP 2: "
            "7,8-dihydroneopterin triphosphate (from GCH1) → (PTPS/PTS) → 6-pyruvoyltetrahydropterin (6-PTP); "
            "PTPS mechanism: Zn2+-dependent enzyme; eliminates triphosphate + isomerises side chain; "
            "6-PTP → (SPR/sepiapterin reductase) → BH4 in two reduction steps; "
            "PTS LOF: 7,8-dihydroneopterin triphosphate → dephosphorylated to dihydroneopterin → "
            "excreted in urine as HIGH NEOPTERIN (explains elevated urine neopterin); "
            "PERIPHERAL vs CENTRAL mechanism: "
            "  Brain TH/TPH are rate-limited by neuronal BH4 availability; "
            "  Hepatic PAH: HIGH Phe stimulates residual GTPCH-I via GFRP → more BH4; "
            "  In peripheral PTS: residual PTPS activity sufficient for neuronal BH4; "
            "  In central PTS: no significant residual PTPS → CNS BH4 critically low → "
            "    TH/TPH cannot function → dopamine + serotonin deficient; "
            "SAPROPTERIN BYPASS: exogenous BH4 bypasses the PTPS step → restores PAH + TH + TPH; "
            "explains why all PTS patients respond to sapropterin"
        ),
        "pathognomonic": (
            "PTS/PTPS DIAGNOSTIC CLUSTER: "
            "1) ELEVATED PHE ON NBS: all PTS patients detected — identical to PKU on NBS; "
            "2) URINE PTERINS: VERY LOW biopterin + VERY HIGH neopterin → neopterin/biopterin ratio markedly elevated; "
            "   neopterin dominant because PTPS step blocked → upstream metabolite accumulates; "
            "3) BH4 LOADING TEST: >30% Phe reduction confirms BH4 deficiency; "
            "4) MANDATORY CSF NEUROTRANSMITTERS (ALL cases): "
            "   HVA NORMAL + 5-HIAA NORMAL = PERIPHERAL subtype → sapropterin + diet only; "
            "   HVA LOW + 5-HIAA LOW = CENTRAL subtype → neurotransmitter replacement mandatory; "
            "5) DISTINGUISH FROM GCH1-AR: "
            "   GCH1-AR: BOTH neopterin AND biopterin LOW (cannot synthesise any pterins); "
            "   PTS: neopterin HIGH + biopterin LOW (upstream accumulates); "
            "6) MRI BRAIN: may show basal ganglia signal changes in untreated/late-treated central PTS; "
            "7) SEIZURES: ~60% of central PTS if undertreated; "
            "MANAGEMENT: immediate sapropterin on diagnosis; defer neurotransmitter decision pending CSF result; "
            "CSF must be obtained BEFORE starting L-DOPA (L-DOPA alters CSF HVA)"
        ),
        "treatment": (
            "PTS/PTPS TREATMENT: "
            "SAPROPTERIN (BH4): 5-20 mg/kg/day — ALL PTS patients; replaces deficient BH4; "
            "PHENYLALANINE RESTRICTION: low-Phe formula in early treatment phase; "
            "relaxed once Phe controlled with sapropterin; "
            "PERIPHERAL SUBTYPE (CSF normal): "
            "Sapropterin + Phe restriction ONLY; no neurotransmitters; "
            "excellent prognosis; neurodevelopment normal with early treatment; "
            "CENTRAL SUBTYPE (CSF HVA + 5-HIAA low): "
            "L-DOPA/CARBIDOPA: 1-5 mg/kg/day + carbidopa; "
            "5-HTP: 2-8 mg/kg/day + carbidopa; "
            "TIMING: obtain CSF BEFORE starting L-DOPA; start L-DOPA + 5-HTP immediately after CSF; "
            "CAUTION — OVERSUPPRESSION: excessive L-DOPA → hypotonia, irritability, vomiting; "
            "titrate carefully; monitor CSF HVA target range; "
            "MONITORING: Phe levels; CSF neurotransmitters annually; developmental assessment; "
            "EEG monitoring for seizures (central subtype); "
            "FOLINIC ACID: 5-15 mg/day — adjunct for some cases; "
            "TRANSITION: sapropterin and neurotransmitters lifelong"
        ),
    },
    {
        "gene": "QDPR",
        "protein": (
            "QDPR -- 4p15.32 AR -- 244aa -- Dihydropteridine-Reductase-DHPR-25.7kDa-Homotetramer-Cytoplasmic-"
            "BH4-Recycling-Enzyme-NAD+-Dependent-SECONDARY-FOLATE-DEFICIENCY-Folinic-Acid-MANDATORY-"
            "Basal-Ganglia-Calcification-OMIM-Gene-612676-Disease-261630"
        ),
        "locus": "4p15.32",
        "protein_size": "244 aa / 25.7 kDa (homotetramer)",
        "inheritance": (
            "AR (biallelic LOF); QDPR encodes dihydropteridine reductase (DHPR) — the BH4 RECYCLING enzyme; "
            "BH4 is consumed in PAH/TH/TPH reactions → oxidised to quinonoid-dihydrobiopterin (qBH2); "
            "DHPR regenerates BH4 from qBH2 using NADH (NAD+-dependent reduction); "
            "QDPR LOF: qBH2 ACCUMULATES → BH4 cannot be recycled → "
            "  progressive BH4 depletion despite INTACT synthesis; "
            "CRITICAL SECONDARY EFFECT: FOLATE DEFICIENCY — qBH2 inhibits sepiapterin reductase and "
            "competes with BH4 at dihydrofolate reductase (DHFR) → secondary DHFR inhibition → "
            "5-methylTHF cannot be regenerated → FUNCTIONAL FOLATE DEFICIENCY; "
            "FOLINIC ACID MANDATORY alongside BH4 — without it: progressive neurodegeneration occurs "
            "even if BH4 is replaced; "
            "BASAL GANGLIA CALCIFICATION: progressive, seen in ~50% of QDPR patients on brain imaging; "
            "pathognomonic feature of QDPR deficiency among BH4 disorders; "
            "DHPR BLOOD SPOT ASSAY: DHPR activity measurable on dried blood spot — "
            "the only BH4 deficiency gene with a DIRECT ENZYME ASSAY on DBS; "
            "DIAGNOSTIC SHORTCUT: DHPR DBS assay → zero/near-zero activity = QDPR confirmed"
        ),
        "disease_category": (
            "BH4 deficiency type II (dihydropteridine reductase deficiency) — OMIM 261630; "
            "ELEVATED PHE ON NBS: detected by HPA on standard NBS; "
            "URINE PTERINS: NORMAL biopterin + NORMAL neopterin (synthesis intact) — "
            "DISTINCT from synthesis defects (GCH1/PTS/SPR); "
            "DHFR INHIBITION → folate deficiency: "
            "  brain 5-MTHF (5-methyltetrahydrofolate) critically depleted; "
            "  CSF 5-MTHF low; "
            "  functionally similar to methotrexate toxicity in brain; "
            "BASAL GANGLIA CALCIFICATION: "
            "  CT scan: bilateral calcifications in basal ganglia and dentate nucleus; "
            "  in untreated/undertreated patients; correlates with folinic acid deficiency; "
            "NEUROTRANSMITTER DEFICIENCY: "
            "  CSF HVA low + 5-HIAA low (same as central defects); "
            "  requires L-DOPA + 5-HTP alongside BH4 + folinic acid; "
            "INCIDENCE: ~1:1,000,000 — second most common BH4 deficiency after PTS"
        ),
        "disease_pathway": (
            "QDPR/DHPR BH4 RECYCLING PATHWAY: "
            "BH4 + PAH (or TH, TPH, NOS) catalytic cycle: "
            "  BH4 (tetrahydrobiopterin) → oxidised to 4a-hydroxyBH4 → "
            "  non-enzymatic dehydration → qBH2 (quinonoid-dihydrobiopterin); "
            "DHPR regeneration: qBH2 + NADH → BH4 + NAD+ (QDPR/DHPR catalyses); "
            "QDPR LOF: qBH2 accumulates: "
            "  (1) BH4 cannot be recycled → progressive BH4 depletion; "
            "  (2) qBH2 inhibits DHFR (dihydrofolate reductase): "
            "    DHFR normally reduces DHF → THF; "
            "    qBH2 is a competitive inhibitor of DHFR → 5-methylTHF trap; "
            "    CSF 5-MTHF severely depleted; "
            "    methionine synthase (MTR) requires 5-MTHF → methionine synthesis impaired; "
            "    neuronal SAM depletion → impaired methylation (myelin, neurotransmitter metabolism); "
            "FOLINIC ACID (5-formylTHF) BYPASS: "
            "  folinic acid is a reduced folate → enters folate cycle directly, bypasses DHFR; "
            "  does NOT require DHFR activation → restores 5-MTHF in brain despite DHFR inhibition; "
            "  MANDATORY: cannot be omitted even if BH4 normalises Phe; "
            "BASAL GANGLIA CALCIFICATION mechanism: "
            "  qBH2 accumulation → calcium-pteridine complex deposition; "
            "  or vascular damage from nitric oxide synthetase impairment"
        ),
        "pathognomonic": (
            "QDPR/DHPR DIAGNOSTIC CLUSTER: "
            "1) ELEVATED PHE ON NBS: detected by HPA; "
            "2) URINE PTERINS: NORMAL biopterin + NORMAL neopterin — "
            "   KEY DISTINCTION from GCH1-AR (both low) and PTS (neopterin high, biopterin low); "
            "   biopterin pattern in QDPR: NORMAL to slightly HIGH (qBH2 → biopterin via pterin oxidation); "
            "3) DHPR DBS ASSAY: ZERO ACTIVITY — PATHOGNOMONIC; the ONLY BH4 gene with direct DBS enzyme assay; "
            "   confirms diagnosis without CSF or pterins; "
            "4) CSF NEUROTRANSMITTERS: HVA low + 5-HIAA low (central neurotransmitter deficiency); "
            "5) CSF 5-MTHF: CRITICALLY LOW — marker of secondary folate deficiency; "
            "   low CSF 5-MTHF is PATHOGNOMONIC for QDPR among BH4 disorders; "
            "6) BRAIN MRI/CT: BASAL GANGLIA CALCIFICATION (bilateral, symmetric, ~50% of cases); "
            "   calcifications in dentate nucleus (cerebellum) also seen; "
            "7) FOLINIC ACID TRIAL: rapid clinical improvement with folinic acid if neurodegeneration early; "
            "CRITICAL WARNING: "
            "  patients treated with BH4 + L-DOPA + 5-HTP but WITHOUT folinic acid → "
            "  progressive neurological deterioration despite apparent metabolic control of HPA; "
            "  FOLINIC ACID IS NOT OPTIONAL — calcifications and neurodegeneration continue without it"
        ),
        "treatment": (
            "QDPR/DHPR TREATMENT: "
            "SAPROPTERIN (BH4): 5-20 mg/kg/day — replaces depleted BH4 (NORMAL synthesis but deficient recycling); "
            "FOLINIC ACID: 10-20 mg/day — MANDATORY; bypasses DHFR inhibition; "
            "  critical: start folinic acid AT DIAGNOSIS; never omit even if Phe controlled; "
            "  monitor CSF 5-MTHF — target: normal range for age; "
            "L-DOPA/CARBIDOPA: 1-5 mg/kg/day — replaces dopamine (neurotransmitter replacement); "
            "5-HTP: 2-8 mg/kg/day + carbidopa — replaces serotonin; "
            "PHENYLALANINE RESTRICTION: adjunct low-Phe formula; "
            "BH4 DOSE: typically higher than PTS (recycling defect harder to compensate); "
            "MONITORING: "
            "  CSF 5-MTHF (annually — key marker of folate status in CNS); "
            "  CSF neurotransmitters (HVA + 5-HIAA annually); "
            "  Brain MRI/CT (basal ganglia calcification surveillance); "
            "  Phe levels (monthly); "
            "  Developmental + neurological assessment; "
            "METHOTREXATE ABSOLUTE CI: MTX inhibits DHFR → catastrophic worsening of already-impaired DHFR; "
            "TRIMETHOPRIM: avoid if possible — DHFR inhibitor; "
            "TRANSITION: all treatments lifelong; folinic acid dose may increase with age"
        ),
    },
    {
        "gene": "PCBD1",
        "protein": (
            "PCBD1 -- 10q22.1 AR -- 104aa -- Pterin-4-Carbinolamine-Dehydratase-1-PCD-DCoH-12kDa-Homotetramer-"
            "Cytoplasmic-BH4-Synthesis-Cofactor-Dual-Role-PCD-Enzyme-HNF1-Transcription-Factor-Cofactor-"
            "Transient-Benign-HPA-MODY-like-OMIM-Gene-126090-Disease-264070"
        ),
        "locus": "10q22.1",
        "protein_size": "104 aa / 12 kDa (homotetramer)",
        "inheritance": (
            "AR (biallelic LOF); PCBD1 encodes pterin-4-carbinolamine dehydratase 1 (PCD) — also known as DCoH; "
            "DUAL ROLE — two completely different functions in the same protein: "
            "(1) PCD ENZYME FUNCTION: in BH4 synthesis — "
            "  converts pterin-4-carbinolamine back to dihydrobiopterin (qBH2) for QDPR recycling; "
            "  without PCD: pterin-4-carbinolamine → 7-biopterin (inactive isomer) instead of qBH2; "
            "  LOF → more 7-biopterin excreted → mildly less efficient BH4 recycling; "
            "  HOWEVER: alternative pathways partially compensate → BENIGN HPA; "
            "(2) HNF1 COFACTOR FUNCTION: "
            "  DCoH (dimerisation cofactor of HNF1): PCBD1 tetramer stabilises HNF1α/HNF1β dimerisation; "
            "  HNF1α controls PAH transcription in liver; "
            "  PCBD1 LOF → reduced HNF1 transcriptional activity → reduced PAH expression → "
            "  mild HPA in addition to pterin pathway effect; "
            "  SEVERE PCBD1 mutations → HNF1α-like MODY (maturity-onset diabetes of the young) phenotype; "
            "CLINICAL HALLMARK: TRANSIENT HPA — Phe elevated on NBS but normalises in weeks-months WITHOUT treatment; "
            "NO NEUROTRANSMITTER DEFICIENCY: CSF HVA + 5-HIAA NORMAL — benign condition; "
            "MOST IMPORTANT CLINICAL FACT: PCBD1 deficiency is BENIGN — no neurological sequelae if recognised"
        ),
        "disease_category": (
            "BH4 deficiency type V / Primapterinuria — OMIM 264070; "
            "TRANSIENT MILD HPA: Phe mildly elevated 180-400 µmol/L on NBS; normalises spontaneously; "
            "PRIMAPTERINURIA: urine pterins show elevated primapterin (7-biopterin) — "
            "PATHOGNOMONIC for PCBD1 deficiency among BH4 disorders; "
            "biopterin: mildly elevated (7-biopterin accumulates, measured as total biopterin); "
            "neopterin: NORMAL (GCH1 and PTS intact); "
            "DHPR ACTIVITY: NORMAL (QDPR intact — recycling enzyme is fine); "
            "CSF NEUROTRANSMITTERS: NORMAL — NO treatment needed for neurological function; "
            "MANAGEMENT: "
            "  mild Phe elevation: observe; some patients need short-term low-Phe formula; "
            "  NEVER start L-DOPA or 5-HTP — neurotransmitters are normal; "
            "  HNF1 connection: monitor for MODY if severe mutations; "
            "IMPORTANCE: distinguishing PCBD1 from PTS/QDPR/GCH1 is CRITICAL "
            "to avoid unnecessary neurotransmitter replacement in a benign condition"
        ),
        "disease_pathway": (
            "PCBD1/PCD BH4 RECYCLING — MINOR BRANCH: "
            "BH4 catalytic cycle: BH4 → 4a-hydroxyl-BH4 → pterin-4-carbinolamine; "
            "PCBD1 STEP: pterin-4-carbinolamine → (PCD/PCBD1 dehydratase) → quinonoid-BH2 (qBH2); "
            "WITHOUT PCBD1: pterin-4-carbinolamine spontaneously → 7-biopterin (primapterin, inactive isomer); "
            "7-biopterin: cannot be recycled to BH4 by DHPR → excreted in urine as primapterin; "
            "PARTIAL ALTERNATIVE: pterin-4-carbinolamine also spontaneously converts to qBH2 at a low rate → "
            "some BH4 recycling occurs even without PCD → explains mild (not severe) phenotype; "
            "DCoH NUCLEAR FUNCTION: "
            "PCBD1 homodimer exists in cytoplasm (pterin recycling) and as tetramer in nucleus (DCoH); "
            "nuclear DCoH tetramer: 2 PCBD1 homodimers + 2 HNF1α (or HNF1β) monomers → "
            "HNF1 dimerisation stabilised → HNF1 target gene transcription (including PAH, TTR, albumin); "
            "PCBD1 LOF → HNF1 complex unstable → reduced PAH transcription → "
            "mildly reduced hepatic PAH protein → mild HPA (additive to pterin branch loss); "
            "TRANSIENT HPA mechanism: compensatory upregulation of alternative pterin recycling; "
            "HPA resolves as other BH4 pathway components upregulate over weeks to months"
        ),
        "pathognomonic": (
            "PCBD1/PCD DIAGNOSTIC CLUSTER: "
            "1) TRANSIENT MILD HPA: NBS Phe 180-400 µmol/L → resolves spontaneously WITHOUT treatment; "
            "   KEY: Phe returns to normal within weeks-months; distinguishes from PKU and other BH4 deficiencies; "
            "2) PRIMAPTERINURIA: elevated urine 7-biopterin (primapterin) — "
            "   PATHOGNOMONIC FOR PCBD1 DEFICIENCY; "
            "   7-biopterin measured as part of urine pterin profile; "
            "   normal neopterin; mildly elevated biopterin (total); "
            "3) DHPR ACTIVITY NORMAL: DBS DHPR assay normal — distinguishes from QDPR deficiency; "
            "4) CSF NEUROTRANSMITTERS: NORMAL HVA + NORMAL 5-HIAA — "
            "   NO neurotransmitter replacement needed; "
            "5) BH4 LOADING TEST: Phe drops with BH4 (confirms BH4 etiology); "
            "6) HNF1 CONNECTION: consider MODY monitoring in severe mutations (HNF1α-like phenotype); "
            "CRITICAL DISTINCTION TABLE: "
            "  GCH1-AR: neopterin LOW + biopterin LOW; "
            "  PTS: neopterin HIGH + biopterin LOW; "
            "  QDPR: neopterin NORMAL + biopterin NORMAL/slightly high; DHPR DBS ZERO; "
            "  PCBD1: primapterin (7-biopterin) HIGH + neopterin NORMAL; DHPR NORMAL; "
            "MANAGEMENT: most PCBD1 patients require NO long-term treatment — "
            "observe Phe; confirm normalisation; genetic counselling; no neurotransmitters"
        ),
        "treatment": (
            "PCBD1/PCD TREATMENT: "
            "GENERALLY NO TREATMENT REQUIRED — BENIGN CONDITION; "
            "TRANSIENT PHE MANAGEMENT: "
            "  mild Phe elevation (180-400 µmol/L): observe; standard formula acceptable; "
            "  if Phe persistently >360 µmol/L in infancy: short-term low-Phe formula; "
            "  do NOT use phenylalanine-free formula long-term — HPA resolves; "
            "SAPROPTERIN: not routinely indicated; may be considered for persistent mild HPA; "
            "L-DOPA AND 5-HTP: CONTRAINDICATED — neurotransmitters normal; "
            "  starting L-DOPA in PCBD1 causes IATROGENIC neurotransmitter excess; "
            "  risk of dyskinesias, irritability, vomiting; "
            "FOLINIC ACID: NOT required — CSF 5-MTHF normal (unlike QDPR); "
            "HNF1/MODY MONITORING: "
            "  if severe PCBD1 mutations: annual glucose, HbA1c monitoring; "
            "  diabetes onset possible in adult life (HNF1α haploinsufficiency effect); "
            "  if MODY develops: sulfonylureas responsive (same as HNF1A-MODY); "
            "GENETIC COUNSELLING: AR condition; 25% recurrence risk for siblings; "
            "LONG-TERM: no neurological follow-up required unless specific clinical concern; "
            "REASSURANCE: excellent prognosis; no cognitive impairment from PCBD1 deficiency"
        ),
    },
    {
        "gene": "SPR",
        "protein": (
            "SPR -- 2p13.2 AR -- 261aa -- Sepiapterin-Reductase-28kDa-Homodimer-Cytoplasmic-"
            "Final-Step-BH4-Synthesis-COMBINED-BH4-Monoamine-Defect-NORMAL-PHE-PATHOGNOMONIC-"
            "NBS-ALWAYS-MISSED-CSF-Neurotransmitters-Profoundly-Low-OMIM-Gene-182125-Disease-612716"
        ),
        "locus": "2p13.2",
        "protein_size": "261 aa / 28 kDa (homodimer)",
        "inheritance": (
            "AR (biallelic LOF); SPR encodes sepiapterin reductase — the FINAL STEP of BH4 de novo synthesis; "
            "SPR catalyses TWO sequential reductions: "
            "  (1) 6-pyruvoyltetrahydropterin → 1-oxo-2-hydroxy-propyl-BH4 (first reduction); "
            "  (2) 1-oxo-2-hydroxy-propyl-BH4 → BH4 (second reduction); "
            "SPR is ALSO the final step of the SALVAGE PATHWAY (sepiapterin → BH4); "
            "SPR LOF: "
            "  (1) BH4 synthesis BLOCKED at final step → BH4 deficient; "
            "  (2) SEPIAPTERIN ACCUMULATES → converted to 7-biopterin by alternative enzymes; "
            "  (3) DOPAMINE DEFICIENCY + SEROTONIN DEFICIENCY (BH4 required for TH + TPH); "
            "CRITICAL UNIQUE FEATURE — NORMAL PHENYLALANINE: "
            "  PAH in liver uses sepiapterin → BH4 VIA CARBONYL REDUCTASE 1 (alternative enzyme); "
            "  carbonyl reductase 1 is expressed in liver but NOT in brain; "
            "  hepatic BH4 partially restored via alternative route → PAH active → Phe NORMAL; "
            "  brain TH/TPH: NO alternative route → BH4 deficient → dopamine + serotonin severely low; "
            "CONSEQUENCE: SPR deficiency ALWAYS MISSED BY NBS (Phe normal); "
            "presents late with progressive severe neurological disease; "
            "severe progressive dystonia + parkinsonism + oculomotor abnormalities"
        ),
        "disease_category": (
            "Dopa-responsive dystonia, sepiapterin reductase type — OMIM 612716; "
            "NORMAL PHENYLALANINE — the key NBS-missing feature; "
            "NBS always normal → diagnosis often delayed by years; "
            "CLINICAL PRESENTATION: "
            "  severe progressive dystonia (often starting in infancy/early childhood); "
            "  parkinsonism features (rigidity, bradykinesia); "
            "  oculomotor crisis (oculogyric crises, 25%); "
            "  intellectual disability (progressive if untreated); "
            "  hypotonia in infancy; "
            "CSF NEUROTRANSMITTERS: profoundly low HVA (dopamine metabolite) + profoundly low 5-HIAA (serotonin metabolite); "
            "CSF PTERINS: low BH4 + elevated sepiapterin + elevated 7-biopterin + NORMAL neopterin; "
            "URINE PTERINS: NORMAL/minimal changes (sepiapterin concentrated in CSF, not well reflected in urine); "
            "PARTIAL L-DOPA RESPONSE: responds to L-DOPA but NOT as dramatically as GCH1-AD-DRD; "
            "FULL TREATMENT: L-DOPA + 5-HTP + BH4 (sapropterin) all required; "
            "INCIDENCE: rare; likely underdiagnosed due to NBS miss"
        ),
        "disease_pathway": (
            "SPR BH4 SYNTHESIS — FINAL STEP: "
            "6-pyruvoyltetrahydropterin (from PTS) → (SPR, 2 NADPH reductions) → BH4; "
            "SPR LOF: 6-pyruvoyltetrahydropterin → (alternative routes): "
            "  → 6-lactoyltetrahydropterin → 7-biopterin (via non-enzymatic); "
            "  → sepiapterin (via carbonyl reductase → partial BH4 in LIVER but NOT brain); "
            "LIVER vs BRAIN ASYMMETRY: "
            "LIVER (hepatic PAH): "
            "  carbonyl reductase 1 (CBR1) expressed → converts sepiapterin to BH4 via salvage; "
            "  hepatic BH4 partially restored → PAH active → Phe catabolised → PHE NORMAL; "
            "BRAIN (TH/TPH): "
            "  carbonyl reductase 1 NOT significantly expressed in neurons; "
            "  no alternative BH4 synthesis → brain BH4 critically depleted; "
            "  TH BLOCKED: dopamine synthesis fails → HVA profoundly low in CSF; "
            "  TPH BLOCKED: serotonin synthesis fails → 5-HIAA profoundly low in CSF; "
            "SEPIAPTERIN ACCUMULATION: CSF sepiapterin measurable (specific to SPR deficiency); "
            "TREATMENT RATIONALE: "
            "  L-DOPA bypasses blocked TH; 5-HTP bypasses blocked TPH; "
            "  sapropterin provides exogenous BH4 to neurons (bypasses SPR defect); "
            "  residual SPR activity may be enhanced by high BH4 substrate relief"
        ),
        "pathognomonic": (
            "SPR DIAGNOSTIC CLUSTER: "
            "1) NORMAL PHENYLALANINE — PATHOGNOMONIC: NBS ALWAYS NEGATIVE; "
            "   this is the KEY diagnostic trap: SPR deficiency is NEVER detected by standard NBS; "
            "   suspect in any child with progressive dystonia/parkinsonism + normal Phe + no metabolic cause; "
            "2) CSF NEUROTRANSMITTERS PROFOUNDLY LOW: HVA very low + 5-HIAA very low; "
            "   MORE PROFOUND depletion than peripheral PTS (HVA/5-HIAA near zero vs mildly low); "
            "3) CSF PTERINS: LOW BH4 + ELEVATED SEPIAPTERIN + elevated 7-biopterin; "
            "   SEPIAPTERIN IN CSF = PATHOGNOMONIC for SPR deficiency; "
            "4) URINE PTERINS: may show primapterin/7-biopterin; neopterin NORMAL; "
            "   DOES NOT SHOW HPA because liver uses alternative pathway; "
            "5) SEVERE PROGRESSIVE MOVEMENT DISORDER: dystonia + parkinsonism + oculomotor; "
            "   onset often in infancy/early childhood; "
            "   oculogyric crises (25%); "
            "6) L-DOPA PARTIAL RESPONSE: improvement but NOT dramatic as GCH1-AD-DRD; "
            "   full response requires L-DOPA + 5-HTP + BH4; "
            "7) GENE PANEL: SPR sequencing (included in BH4 movement disorder panel); "
            "CRITICAL WARNING: "
            "  patients with SPR are often initially diagnosed as cerebral palsy or DRD; "
            "  KEY QUESTION: is the phenylalanine NORMAL? If YES and movement disorder severe → SUSPECT SPR"
        ),
        "treatment": (
            "SPR TREATMENT: "
            "L-DOPA/CARBIDOPA: 3-10 mg/kg/day — bypasses blocked TH → dopamine replacement; "
            "  start low, titrate carefully; SPR responds but NOT as dramatically as GCH1-AD; "
            "5-HTP (5-hydroxytryptophan): 3-10 mg/kg/day — bypasses blocked TPH → serotonin replacement; "
            "  always co-administer with carbidopa (prevents peripheral decarboxylation); "
            "SAPROPTERIN (BH4): 5-20 mg/kg/day — provides exogenous BH4 to neurons; "
            "  crosses blood-brain barrier (partial) → reduces BH4 deficiency in brain; "
            "  additive to L-DOPA + 5-HTP; "
            "COMBINATION THERAPY IS ESSENTIAL: all three drugs required for optimal response; "
            "MONITORING: "
            "  CSF HVA + 5-HIAA + BH4 + sepiapterin (treatment targets); "
            "  neurodevelopmental assessment; "
            "  movement disorder scoring (AIMS, BFMDRS); "
            "  oculogyric crises frequency; "
            "OVERSUPPRESSION RISK: excessive L-DOPA → dyskinesias, stereotypies, sleep disturbance; "
            "PHENYLALANINE: no restriction needed (Phe normal); "
            "PROGNOSIS: significant improvement with treatment; some residual motor disability; "
            "  early treatment → better neurodevelopmental outcome; "
            "GENETIC COUNSELLING: AR; 25% recurrence; siblings must be tested urgently"
        ),
    },
    {
        "gene": "DNAJC12",
        "protein": (
            "DNAJC12 -- 10q21.3 AR -- 198aa -- DnaJ-Heat-Shock-Protein-Cochaperone-HSP70-PAH-Cochaperone-"
            "Hyperphenylalaninaemia-Neurotransmitter-Deficiency-BH4-NORMAL-Distinguishes-From-Other-BH4-Deficiencies-"
            "Novel-Described-2017-OMIM-Gene-606060-Disease-617384"
        ),
        "locus": "10q21.3",
        "protein_size": "198 aa / 23 kDa",
        "inheritance": (
            "AR (biallelic LOF); DNAJC12 encodes a DnaJ (HSP40) co-chaperone of HSP70; "
            "MECHANISM: "
            "  DNAJC12 is required for proper folding and stability of PAH (phenylalanine hydroxylase); "
            "  DNAJC12 LOF → PAH misfolding/premature degradation → reduced PAH activity → HPA; "
            "  ALSO: DNAJC12 is a co-chaperone for AADC (aromatic amino acid decarboxylase, DDC gene); "
            "  AADC misfolding → reduced AADC activity → impaired dopamine + serotonin synthesis; "
            "  mechanism differs from BH4 deficiencies: BH4 itself is NORMAL; "
            "CRITICAL DISTINGUISHING FEATURE: BH4 NORMAL (not a BH4 synthesis or recycling defect); "
            "  urine pterins: NORMAL; DHPR assay: NORMAL; BH4 loading test result: VARIABLE; "
            "  neurotransmitter deficiency despite normal BH4 = chaperone defect; "
            "PHENOTYPIC SPECTRUM: "
            "  mild: simple HPA without neurological features; "
            "  moderate: HPA + mild intellectual disability; "
            "  severe: HPA + DRD-like severe movement disorder + intellectual disability; "
            "DISCOVERY: first described 2017 (Blau et al.) — classified as novel cause of HPA with NT deficiency; "
            "INCIDENCE: uncertain; likely underdiagnosed; not on standard BH4 gene panels before 2017"
        ),
        "disease_category": (
            "Hyperphenylalaninaemia with neurotransmitter deficiency (DNAJC12 cochaperone deficiency) — OMIM 617384; "
            "NBS DETECTION: detected by elevated Phe (HPA) on standard NBS; "
            "BH4 LOADING TEST: variable — some patients show partial Phe response; "
            "  CRITICAL TRAP: partial BH4 response leads to misdiagnosis as BH4-responsive PAH/mild PKU; "
            "  BH4 response may reflect PAH stabilisation by pharmacological chaperone effect; "
            "URINE PTERINS: NORMAL (synthesis intact; recycling intact); "
            "CSF NEUROTRANSMITTERS: HVA low + 5-HIAA low (AADC impairment); "
            "CLINICAL SPECTRUM: "
            "  intellectual disability; movement disorder (DRD-like, can be severe); "
            "  may present as simple HPA → treatment with sapropterin → "
            "  Phe control but progressive neurological deterioration if NT deficiency missed; "
            "MANAGEMENT: requires both BH4/dietary treatment for HPA AND NT supplementation (L-DOPA + 5-HTP); "
            "KEY: if BH4 loading test gives partial response, always measure CSF neurotransmitters; "
            "DISTINCTION FROM PKU: normal BH4 + NT deficiency + chaperone gene = DNAJC12"
        ),
        "disease_pathway": (
            "DNAJC12 CHAPERONE PATHWAY: "
            "HSP70 chaperone cycle: "
            "  client protein (PAH or AADC) → HSP70 (heat shock protein 70 kDa) → "
            "  DNAJC12 (DnaJ/HSP40 co-chaperone) stimulates HSP70 ATPase → "
            "  ATP hydrolysis → HSP70 conformational change → client protein folding/stabilisation; "
            "DNAJC12 LOF: "
            "  (1) PAH client: HSP70 ATPase stimulation reduced → PAH folding impaired → "
            "      PAH misfolded/aggregated/degraded → reduced PAH activity → Phe accumulates → HPA; "
            "  (2) AADC client: HSP70 stimulation reduced → AADC stability decreased → "
            "      AADC activity reduced → impaired decarboxylation of L-DOPA → dopamine deficit; "
            "      AADC also converts 5-HTP → serotonin; reduced AADC → serotonin deficit; "
            "BH4 COMPLETELY NORMAL in DNAJC12: "
            "  GCH1, PTS, SPR, QDPR, PCBD1 all intact → BH4 synthesis and recycling normal; "
            "  TH uses BH4 normally; but L-DOPA produced by TH cannot be converted to dopamine efficiently "
            "  (AADC impaired) → functional dopamine deficiency despite normal TH/BH4; "
            "PHARMACOLOGICAL CHAPERONE EFFECT OF SAPROPTERIN: "
            "  sapropterin may stabilise residual PAH protein → partial HPA correction; "
            "  explains partial BH4 loading test response"
        ),
        "pathognomonic": (
            "DNAJC12 DIAGNOSTIC CLUSTER: "
            "1) ELEVATED PHE ON NBS: HPA detected; "
            "2) BH4 LOADING TEST: VARIABLE PARTIAL RESPONSE — "
            "   differs from true BH4 deficiencies (which respond robustly); "
            "   unlike simple PKU/PAH deficiency: CSF neurotransmitters abnormal; "
            "3) URINE PTERINS: COMPLETELY NORMAL — "
            "   distinguishes DNAJC12 from all BH4 synthesis/recycling defects; "
            "4) DHPR ASSAY: NORMAL — excludes QDPR; "
            "5) CSF NEUROTRANSMITTERS: HVA LOW + 5-HIAA LOW — "
            "   despite NORMAL BH4 and normal urine pterins; "
            "   this combination (normal pterins + NT deficiency) is KEY for DNAJC12; "
            "6) PHENOTYPE: ranges from simple HPA to severe DRD-like movement disorder; "
            "   progressive neurological deterioration if NT deficiency not treated; "
            "7) MOLECULAR: DNAJC12 sequencing (must be on extended HPA gene panel); "
            "   gene first described 2017 — not on older panels; "
            "CRITICAL CLINICAL ALERT: "
            "  child with HPA + partial BH4 response + normal pterins + L-DOPA responsive movement disorder: "
            "  MUST check DNAJC12 + CSF neurotransmitters before concluding BH4-responsive PKU; "
            "  misdiagnosis as PKU with missing NT treatment = preventable neurological injury"
        ),
        "treatment": (
            "DNAJC12 TREATMENT: "
            "PHE MANAGEMENT: "
            "  SAPROPTERIN (BH4): 5-20 mg/kg/day — partial Phe response via PAH stabilisation; "
            "  LOW-PHE FORMULA: phenylalanine restriction — adjunct to BH4; "
            "  target Phe <360 µmol/L in infancy/childhood; "
            "NEUROTRANSMITTER REPLACEMENT (MANDATORY when CSF abnormal): "
            "L-DOPA/CARBIDOPA: 1-5 mg/kg/day — replaces dopamine (bypasses impaired AADC); "
            "5-HTP: 1-5 mg/kg/day + carbidopa — replaces serotonin; "
            "AADC FUNCTION NOTE: AADC is impaired but NOT absent → some enzymatic conversion; "
            "  lower NT supplement doses may suffice vs SPR or DDC; titrate by CSF response; "
            "MONITORING: "
            "  CSF HVA + 5-HIAA annually (treatment target: normal range for age); "
            "  Phe levels monthly; "
            "  neurodevelopmental + movement assessment; "
            "  brain MRI periodically; "
            "GENETIC COUNSELLING: AR condition; 25% recurrence; "
            "NEWLY DESCRIBED GENE: ensure genetic panel includes DNAJC12 (may be absent on older panels); "
            "PROGNOSIS: with early treatment, good neurodevelopmental outcome; "
            "  delayed diagnosis → progressive intellectual and motor disability; "
            "FUTURE THERAPY: chaperone therapy target (HSP70 system manipulation) — research stage"
        ),
    },
    {
        "gene": "TH",
        "protein": (
            "TH -- 11p15.5 AR -- 498aa -- Tyrosine-Hydroxylase-TH-59kDa-Homotetramer-Cytoplasmic-Mitochondria-Associated-"
            "BH4-Dependent-Rate-Limiting-Dopamine-Synthesis-NOT-a-BH4-Gene-Infantile-Parkinsonism-DRD-"
            "L-DOPA-Highly-Responsive-Near-Complete-Remission-OMIM-Gene-191290-Disease-191290-605407"
        ),
        "locus": "11p15.5",
        "protein_size": "498 aa / 59 kDa (homotetramer)",
        "inheritance": (
            "AR (biallelic LOF); TH encodes tyrosine hydroxylase — the RATE-LIMITING ENZYME for dopamine synthesis; "
            "CRITICAL CLASSIFICATION: TH is NOT a BH4 gene — it is a BH4-DEPENDENT enzyme; "
            "  BH4 is a cofactor for TH; TH uses BH4 to hydroxylate tyrosine → L-DOPA; "
            "  TH LOF: L-DOPA synthesis FAILS (regardless of BH4 status); "
            "  BH4 is NORMAL in TH deficiency (synthesis and recycling intact); "
            "TWO CLINICAL PHENOTYPES (allele severity determines phenotype): "
            "TYPE A (DRD-B / DOPA-RESPONSIVE DYSTONIA TYPE B): "
            "  milder TH mutations with residual activity; "
            "  onset childhood; diurnal dystonia (less dramatic than GCH1-AD-DRD); "
            "  L-DOPA HIGHLY RESPONSIVE; "
            "  OMIM 191290; "
            "TYPE B (INFANTILE PARKINSONISM / PROGRESSIVE ENCEPHALOPATHY): "
            "  severe null TH mutations; onset neonatal/infancy; "
            "  severe hypokinetic-rigid syndrome (infantile parkinsonism); "
            "  progressive encephalopathy, ptosis, eye-movement abnormalities; "
            "  L-DOPA HIGHLY RESPONSIVE — near-complete remission possible even in severe form; "
            "  OMIM 605407; "
            "KEY DISTINCTION FROM GCH1-AD-DRD: "
            "  GCH1-AD: BH4 deficiency → ALL BH4-dependent enzymes affected → diurnal fluctuation pathognomonic; "
            "  TH: BH4 NORMAL → only dopamine deficient (serotonin pathway intact via TPH + normal BH4); "
            "  Phe NORMAL (PAH uses normal BH4); CSF 5-HIAA NORMAL (TPH intact)"
        ),
        "disease_category": (
            "Tyrosine hydroxylase deficiency — OMIM 191290 (Type A, DRD-B) / 605407 (Type B, infantile parkinsonism); "
            "BH4 NORMAL — not a BH4 deficiency; pterin profile normal; "
            "NBS: Phe NORMAL — not detected by standard NBS; "
            "CSF BIOCHEMISTRY FINGERPRINT: "
            "  HVA (homovanillic acid) LOW — dopamine metabolite; "
            "  5-HIAA (5-hydroxyindolacetic acid) NORMAL — serotonin metabolite (TPH uses normal BH4); "
            "  BH4 NORMAL; neopterin NORMAL; biopterin NORMAL; "
            "  PATHOGNOMONIC: HVA low + 5-HIAA NORMAL + BH4 normal = TH deficiency; "
            "  (contrasts with GCH1-AR and SPR where BOTH HVA and 5-HIAA low); "
            "L-DOPA: HIGHLY RESPONSIVE — often dramatic response with near-complete remission; "
            "  TH deficiency may be the MOST L-DOPA RESPONSIVE condition among movement disorders; "
            "  even severe infantile parkinsonism (type B) can achieve near-normal function with L-DOPA; "
            "INCIDENCE: rare but likely underdiagnosed; "
            "  mild type A may be misdiagnosed as spastic diplegia, DRD, or benign essential tremor"
        ),
        "disease_pathway": (
            "TH/DOPAMINE SYNTHESIS PATHWAY: "
            "Tyrosine → (TH + BH4 cofactor + Fe2+ + O2) → L-DOPA; "
            "L-DOPA → (AADC/DDC + PLP cofactor) → DOPAMINE; "
            "DOPAMINE → (DBH) → NORADRENALINE → (PNMT) → ADRENALINE; "
            "CATECHOLAMINE METABOLITES: dopamine → HVA (HVA measured in CSF as dopamine proxy); "
            "TH CATALYTIC MECHANISM: "
            "  TH active site: Fe2+ + BH4 (cofactors); O2 substrate; tyrosine aromatic substrate; "
            "  Fe2+ + O2 + BH4 → Fe3+ + OH radical → hydroxylation of tyrosine ring → L-DOPA; "
            "  BH4 oxidised to qBH2 in process → regenerated by DHPR (QDPR); "
            "TH LOF: DOPAMINE SYNTHESIS BLOCKED at first step; "
            "  L-DOPA absent → dopamine absent → HVA in CSF = VERY LOW; "
            "  SEROTONIN UNAFFECTED: TPH (tryptophan hydroxylase) uses BH4 + L-tryptophan → 5-HTP; "
            "  TH does not affect serotonin pathway → 5-HIAA NORMAL; "
            "L-DOPA BYPASS: oral L-DOPA crosses BBB → AADC converts to dopamine in brain → "
            "  bypasses the blocked TH step completely; "
            "  explains DRAMATIC RESPONSE to L-DOPA in TH deficiency; "
            "ALLELE SEVERITY: severe null mutations → infantile parkinsonism (type B); "
            "  missense mutations with partial activity → DRD-B (type A, milder)"
        ),
        "pathognomonic": (
            "TH/TYROSINE HYDROXYLASE DIAGNOSTIC CLUSTER: "
            "1) NORMAL PHE: NBS negative — Phe not elevated (PAH uses normal BH4); "
            "2) CSF HVA VERY LOW: dopamine deficiency fingerprint; "
            "3) CSF 5-HIAA NORMAL — PATHOGNOMONIC CONTRAST: "
            "   serotonin synthesis intact (TPH + BH4 both normal); "
            "   HVA low + 5-HIAA NORMAL = ISOLATED DOPAMINE DEFICIENCY = TH deficiency or DDC partial; "
            "   compare: GCH1-AR + SPR: BOTH HVA and 5-HIAA low; "
            "4) BH4 NORMAL: CSF BH4, urine pterins, neopterin — all NORMAL; "
            "   excludes all BH4 synthesis/recycling defects; "
            "5) L-DOPA HIGHLY RESPONSIVE: near-complete remission possible; "
            "   diagnostic AND therapeutic; always trial L-DOPA in movement disorder child with HVA low; "
            "6) PHENOTYPE: "
            "   TYPE A: diurnal dystonia (less fluctuation than GCH1-AD); childhood onset; "
            "   TYPE B: neonatal/infantile hypokinetic-rigid syndrome; ptosis; oculomotor; "
            "   progressive encephalopathy if untreated; "
            "7) DO NOT CONFUSE WITH GCH1-AD-DRD: "
            "   GCH1-AD: BH4 borderline + diurnal fluctuation + 5-HIAA mildly low + family history AD; "
            "   TH: BH4 NORMAL + 5-HIAA NORMAL + AR inheritance; "
            "GENE PANEL: TH must be included in DRD/movement disorder panels; "
            "CSF is mandatory diagnostic tool"
        ),
        "treatment": (
            "TH/TYROSINE HYDROXYLASE TREATMENT: "
            "L-DOPA/CARBIDOPA: CORNERSTONE — bypasses blocked TH; "
            "TYPE A (DRD-B): "
            "  L-DOPA: 2-5 mg/kg/day; start low (0.5 mg/kg/day) and titrate; "
            "  DRAMATIC RESPONSE: near-complete resolution of dystonia; "
            "  carbidopa: 25% of L-DOPA dose (prevents peripheral conversion); "
            "TYPE B (infantile parkinsonism): "
            "  L-DOPA: 5-15 mg/kg/day (higher doses may be needed); "
            "  remarkable recovery of motor function possible even in severe cases; "
            "  lifelong L-DOPA required; "
            "5-HTP: NOT required — serotonin pathway intact (5-HIAA normal); "
            "BH4 (sapropterin): NOT required — BH4 normal; "
            "DO NOT confuse treatment protocol with GCH1-AR or SPR (those need 5-HTP + BH4 in addition); "
            "MONITORING: "
            "  CSF HVA (treatment target: normal or near-normal); "
            "  CSF 5-HIAA (should remain normal — if falls, consider co-existing defect); "
            "  motor function scoring; "
            "  L-DOPA side effects: dyskinesias (dose-dependent), sleep disturbance; "
            "CARBIDOPA RATIO: maintain L-DOPA:carbidopa 4:1; "
            "POTENTIAL COMPLICATIONS: "
            "  L-DOPA-induced dyskinesias (reduce dose, add adjuncts); "
            "  psychiatric symptoms (rare at therapeutic doses); "
            "PROGNOSIS: with early L-DOPA treatment: excellent — most type A and many type B patients achieve "
            "near-normal motor milestones and neurodevelopment"
        ),
    },
    {
        "gene": "DDC",
        "protein": (
            "DDC -- 7p12.3 AR -- 480aa -- DOPA-Decarboxylase-Aromatic-L-Amino-Acid-Decarboxylase-AADC-54kDa-Homodimer-"
            "PLP-Dependent-Final-Step-Dopamine-AND-Serotonin-Synthesis-Oculogyric-Crises-PATHOGNOMONIC-"
            "Gene-Therapy-Upstaza-EU-UK-2022-OMIM-Gene-107930-Disease-608643"
        ),
        "locus": "7p12.3",
        "protein_size": "480 aa / 54 kDa (homodimer)",
        "inheritance": (
            "AR (biallelic LOF); DDC encodes DOPA decarboxylase (also called aromatic L-amino acid decarboxylase, AADC); "
            "AADC catalyses THE FINAL STEP OF BOTH dopamine AND serotonin synthesis: "
            "  L-DOPA → (AADC + PLP) → DOPAMINE (in catecholamine pathway); "
            "  5-HTP → (AADC + PLP) → SEROTONIN (in indolamine pathway); "
            "PYRIDOXAL-5'-PHOSPHATE (PLP) DEPENDENT: PLP is the essential cofactor; "
            "  B6 (pyridoxine) → PLP by pyridoxal kinase; "
            "  AADC apoenzyme requires PLP for catalytic activity; "
            "DDC LOF: "
            "  BOTH dopamine and serotonin synthesis blocked at final step; "
            "  L-DOPA accumulates → 3-OMD (3-O-methyldopa) by COMT → elevated 3-OMD in plasma/CSF; "
            "  BOTH HVA and 5-HIAA profoundly low in CSF; "
            "OCULOGYRIC CRISES: PATHOGNOMONIC for DDC/AADC deficiency (55-85%); "
            "  episodes of forced sustained upward eye deviation; can last minutes to hours; "
            "  triggered by illness, fatigue, excitement; "
            "GENE THERAPY UPSTAZA (eladocagene exuparvovec): "
            "  EMA approved July 2022; MHRA UK approved 2022; "
            "  intracranial injection into putamen; viral vector (AAV2-hAADC); "
            "  restores AADC in striatum; dramatic improvement in severe infantile cases"
        ),
        "disease_category": (
            "Aromatic L-amino acid decarboxylase (AADC) deficiency — OMIM 608643; "
            "COMBINED dopamine + serotonin deficiency (BOTH pathways blocked at AADC step); "
            "CSF BIOCHEMISTRY FINGERPRINT: "
            "  HVA VERY LOW (dopamine metabolite absent); "
            "  5-HIAA VERY LOW (serotonin metabolite absent); "
            "  3-OMD (3-O-methyldopa) ELEVATED — L-DOPA methylated by COMT; "
            "    3-OMD elevation is PATHOGNOMONIC for AADC deficiency; "
            "  AADC PLASMA ENZYME ACTIVITY: ZERO — confirmatory; "
            "  PLP-responsive AADC variants: some missense mutations respond to pyridoxine; "
            "NBS: Phe NORMAL — missed by standard NBS; "
            "CLINICAL FEATURES: "
            "  OCULOGYRIC CRISES: 55-85% — MOST CHARACTERISTIC FEATURE; "
            "  hypotonia (severe); "
            "  movement disorder (dystonia, choreoathetosis); "
            "  autonomic dysfunction (sweating, temperature instability, nasal congestion); "
            "  ptosis; "
            "  developmental delay; "
            "  feeding difficulties; "
            "INCIDENCE: rare; concentrated in Taiwan/East Asian populations (c.IVS6+4A>T founder mutation)"
        ),
        "disease_pathway": (
            "DDC/AADC COMBINED CATECHOLAMINE + INDOLAMINE PATHWAY: "
            "CATECHOLAMINE ARM: "
            "  Tyrosine → (TH + BH4) → L-DOPA → (AADC/DDC + PLP) → DOPAMINE → "
            "  (DBH) → NORADRENALINE → (PNMT) → ADRENALINE; "
            "INDOLAMINE ARM: "
            "  Tryptophan → (TPH + BH4) → 5-HTP → (AADC/DDC + PLP) → SEROTONIN; "
            "AADC is the CONVERGENCE POINT of both pathways; "
            "DDC LOF: "
            "  CATECHOLAMINES: L-DOPA cannot be converted → dopamine = 0 → "
            "    L-DOPA accumulates → COMT methylates L-DOPA → 3-OMD (3-O-methyldopa) ELEVATED; "
            "    HVA (dopamine → COMT/MAO → HVA) = ZERO; "
            "  SEROTONIN: 5-HTP cannot be converted → serotonin = 0 → "
            "    5-HIAA (serotonin metabolite) = ZERO; "
            "  BOTH HVA and 5-HIAA ABSENT = AADC fingerprint; "
            "  3-OMD in CSF: elevated because L-DOPA (accumulating) → methylated to 3-OMD; "
            "PLP DEPENDENCY: "
            "  PLP forms Schiff base with Lys303 (AADC active site) → external aldimine with substrate; "
            "  PLP LOF or depletion → AADC apoenzyme inactive; "
            "  SOME missense variants: reduce PLP affinity → respond to pyridoxine (B6 supplementation); "
            "GENE THERAPY MECHANISM: AAV2 vector → bilateral putamen injection → "
            "  AADC transgene expression in striatal neurons → dopamine synthesis restored locally"
        ),
        "pathognomonic": (
            "DDC/AADC DIAGNOSTIC CLUSTER: "
            "1) OCULOGYRIC CRISES (55-85%) — MOST PATHOGNOMONIC FEATURE: "
            "   episodic forced sustained upward eye deviation; can last hours; "
            "   worsened by illness/fatigue; relieved by sleep; "
            "   virtually DIAGNOSTIC for AADC deficiency in context of movement disorder + hypotonia; "
            "2) CSF BIOCHEMISTRY FINGERPRINT: "
            "   HVA VERY LOW/ABSENT (dopamine depleted); "
            "   5-HIAA VERY LOW/ABSENT (serotonin depleted); "
            "   3-OMD (3-O-methyldopa) ELEVATED — PATHOGNOMONIC: L-DOPA methylated by COMT; "
            "   3-OMD + HVA absent + 5-HIAA absent = AADC fingerprint; "
            "3) PLASMA AADC ENZYME ACTIVITY: ZERO — confirmatory assay on blood; "
            "4) URINE CATECHOLAMINES: L-DOPA elevated; dopamine absent; VMA low; "
            "5) NORMAL PHE: NBS negative; "
            "6) NORMAL BH4: pterins normal (BH4 synthesis and recycling intact); "
            "7) PLP-RESPONSIVE SUBGROUP: pyridoxine 10-20 mg/kg/day → partial improvement "
            "   in missense mutations affecting PLP binding; trial B6 in all new diagnoses; "
            "8) GENE THERAPY ELIGIBILITY: age 18 months to 6 years; severe phenotype; "
            "   Upstaza (eladocagene exuparvovec) — EMA approved 2022; "
            "CONTRAST WITH TH DEFICIENCY: TH: HVA low + 5-HIAA NORMAL; DDC: BOTH HVA + 5-HIAA absent + 3-OMD high; "
            "AUTONOMIC FEATURES: profuse sweating, temperature instability — reflects noradrenaline/adrenaline deficiency"
        ),
        "treatment": (
            "DDC/AADC TREATMENT: "
            "GENE THERAPY — UPSTAZA (eladocagene exuparvovec): "
            "  EMA approved July 2022; MHRA UK 2022; bilateral putamen injection; AAV2-hAADC vector; "
            "  eligibility: age 18 months to 6 years; severe phenotype; "
            "  results: dramatic motor improvement; oculogyric crises resolved in most; "
            "  long-term data: 5-10 year follow-up showing sustained benefit; "
            "PHARMACOLOGICAL (pre-gene therapy or non-eligible patients): "
            "PYRIDOXINE (B6): 10-30 mg/kg/day — trial in all patients; "
            "  PLP is AADC cofactor; some missense variants respond; "
            "  safe to trial; response rate ~30-40% partial improvement; "
            "MAO-B INHIBITOR (SELEGILINE): 0.1-0.2 mg/kg/day; "
            "  reduces dopamine metabolism → extends effect of residual AADC; "
            "  used as first pharmacological adjunct; "
            "DOPAMINE AGONISTS: bromocriptine 0.1-0.3 mg/kg/day OR pramipexole; "
            "  directly stimulate dopamine receptors (bypasses AADC-dependent synthesis); "
            "  useful adjunct to selegiline; "
            "L-DOPA: GENERALLY NOT HELPFUL (cannot be converted without AADC); "
            "  may exacerbate oculogyric crises (L-DOPA substrate accumulates); "
            "  use cautiously if at all; "
            "FOLINIC ACID: adjunct for some patients; "
            "BENZODIAZEPINES: for acute oculogyric crises; "
            "MONITORING: CSF HVA + 5-HIAA + 3-OMD (treatment targets); "
            "AUTONOMIC SUPPORT: midodrine for postural hypotension; "
            "PROGNOSIS: without gene therapy — poor; severe disability; "
            "  with gene therapy: transformative improvement in majority"
        ),
    },
]

# Per-gene clinical feature probabilities
_PHE_RANGES = {
    "GCH1": (600, 1800),   # AR form: HPA; AD-DRD normal (mixed)
    "PTS":  (400, 2000),   # HPA (most common BH4 deficiency)
    "QDPR": (400, 1600),   # HPA
    "PCBD1": (120, 400),   # Transient mild HPA
    "SPR":  (5, 60),       # NORMAL Phe — the diagnostic trap
    "DNAJC12": (200, 800), # HPA (moderate)
    "TH":   (5, 60),       # NORMAL Phe — dopamine deficiency not HPA
    "DDC":  (5, 60),       # NORMAL Phe — AADC deficiency not HPA
}

_NBS_DETECTED = {
    "GCH1": 0.60,   # AR cases detected; AD-DRD not (mixed population)
    "PTS":  0.98,
    "QDPR": 0.95,
    "PCBD1": 0.90,
    "SPR":  0.00,   # ALWAYS MISSED — Phe normal
    "DNAJC12": 0.90,
    "TH":   0.00,   # Phe normal — never detected by NBS
    "DDC":  0.00,   # Phe normal — never detected by NBS
}

_BH4_RESPONSIVE = {
    "GCH1": 0.65,   # AR: BH4 responsive; AD: L-DOPA primary
    "PTS":  0.90,   # Most respond to sapropterin
    "QDPR": 0.55,   # Partial response
    "PCBD1": 0.70,  # Transient; sapropterin not usually needed
    "SPR":  0.20,   # Partial (BH4 helps brain TH/TPH)
    "DNAJC12": 0.45, # Variable partial response
    "TH":   0.10,   # Not a BH4 deficiency; BH4 normal
    "DDC":  0.05,   # BH4 normal; pyridoxine may help
}

_CSF_HVA_LOW = {
    "GCH1": 0.75,   # AR: yes; AD-DRD: mildly low
    "PTS":  0.65,   # Central form: yes; peripheral: no
    "QDPR": 0.90,   # Yes (neurotransmitter deficiency)
    "PCBD1": 0.05,  # BENIGN — no neurotransmitter deficiency
    "SPR":  0.98,   # Profoundly low
    "DNAJC12": 0.80, # Low (AADC impaired)
    "TH":   0.95,   # VERY LOW — isolated dopamine deficiency
    "DDC":  0.98,   # VERY LOW (AADC absent — no dopamine)
}

_CSF_5HIAA_LOW = {
    "GCH1": 0.70,   # AR: yes; AD-DRD: mildly low
    "PTS":  0.60,   # Central form: yes
    "QDPR": 0.85,
    "PCBD1": 0.03,  # BENIGN
    "SPR":  0.97,   # Profoundly low
    "DNAJC12": 0.75, # Low (AADC impaired for serotonin too)
    "TH":   0.10,   # NORMAL — TPH uses normal BH4; serotonin intact
    "DDC":  0.97,   # VERY LOW (AADC absent — no serotonin)
}

_DYSTONIA_MOVEMENT_DISORDER = {
    "GCH1": 0.85,   # DRD/dystonia hallmark
    "PTS":  0.65,   # Central form: yes
    "QDPR": 0.80,
    "PCBD1": 0.05,  # BENIGN
    "SPR":  0.95,   # Severe progressive
    "DNAJC12": 0.70,
    "TH":   0.90,   # Infantile parkinsonism / DRD
    "DDC":  0.80,   # Movement disorder
}

_OCULOGYRIC_CRISES = {
    "GCH1": 0.30,   # AR form
    "PTS":  0.25,   # Central form
    "QDPR": 0.20,
    "PCBD1": 0.02,
    "SPR":  0.25,
    "DNAJC12": 0.15,
    "TH":   0.20,   # Oculomotor abnormalities
    "DDC":  0.70,   # PATHOGNOMONIC (55-85%)
}

_SEIZURES = {
    "GCH1": 0.30,   # AR form
    "PTS":  0.60,   # Central form
    "QDPR": 0.55,
    "PCBD1": 0.03,
    "SPR":  0.45,
    "DNAJC12": 0.40,
    "TH":   0.25,
    "DDC":  0.35,
}

_LDOPA_RESPONSIVE = {
    "GCH1": 0.92,   # DRD: near-complete remission
    "PTS":  0.55,   # Central form: yes
    "QDPR": 0.70,
    "PCBD1": 0.10,  # Not indicated
    "SPR":  0.60,   # Partial response
    "DNAJC12": 0.60,
    "TH":   0.95,   # HIGHLY RESPONSIVE — near-complete remission
    "DDC":  0.15,   # Generally NOT helpful (cannot convert L-DOPA)
}

_DIURNAL_FLUCTUATION = {
    "GCH1": 0.75,   # DRD hallmark
    "PTS":  0.30,
    "QDPR": 0.25,
    "PCBD1": 0.05,
    "SPR":  0.15,
    "DNAJC12": 0.20,
    "TH":   0.50,   # Type A DRD-B
    "DDC":  0.20,
}


def _generate_patients(gene_idx: int, n: int = 40, seed: int = 0):
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_idx]
    g = gene["gene"]
    patients = []
    onset_params = {
        "GCH1":    (0.7, 0.6),
        "PTS":     (0.3, 0.3),
        "QDPR":    (0.2, 0.2),
        "PCBD1":   (0.0, 0.05),  # NBS — detected at birth
        "SPR":     (1.0, 0.8),
        "DNAJC12": (0.6, 0.5),
        "TH":      (0.8, 0.7),
        "DDC":     (0.2, 0.2),
    }
    mu, sigma = onset_params.get(g, (1.0, 0.5))
    phe_lo, phe_hi = _PHE_RANGES.get(g, (60, 400))
    for i in range(n):
        onset_yrs = max(0.01, rng.gauss(mu, sigma))
        phe_peak = rng.uniform(phe_lo, phe_hi)
        patients.append({
            "patient_id": f"{g}-{seed}-{i+1:03d}",
            "gene": g,
            "onset_years": round(onset_yrs, 2),
            "phe_peak_umol_L": round(phe_peak, 1),
            "nbs_detected": rng.random() < _NBS_DETECTED.get(g, 0.0),
            "bh4_responsive": rng.random() < _BH4_RESPONSIVE.get(g, 0.0),
            "csf_hva_low": rng.random() < _CSF_HVA_LOW.get(g, 0.0),
            "csf_5hiaa_low": rng.random() < _CSF_5HIAA_LOW.get(g, 0.0),
            "dystonia_movement_disorder": rng.random() < _DYSTONIA_MOVEMENT_DISORDER.get(g, 0.0),
            "oculogyric_crises": rng.random() < _OCULOGYRIC_CRISES.get(g, 0.0),
            "seizures": rng.random() < _SEIZURES.get(g, 0.0),
            "ldopa_responsive": rng.random() < _LDOPA_RESPONSIVE.get(g, 0.0),
            "diurnal_fluctuation": rng.random() < _DIURNAL_FLUCTUATION.get(g, 0.0),
        })
    return patients


def generate_overview():
    all_patients = []
    for idx in range(len(ATLAS_GENES)):
        all_patients.extend(_generate_patients(idx, n=40, seed=2670 + idx))

    summary = {}
    for p in all_patients:
        g = p["gene"]
        if g not in summary:
            summary[g] = {
                "gene": g, "n": 0,
                "nbs_detected_pct": 0,
                "bh4_responsive_pct": 0,
                "csf_hva_low_pct": 0,
                "csf_5hiaa_low_pct": 0,
                "dystonia_pct": 0,
                "oculogyric_pct": 0,
                "seizures_pct": 0,
                "ldopa_responsive_pct": 0,
                "diurnal_fluctuation_pct": 0,
                "mean_onset_yrs": 0.0,
                "mean_phe": 0.0,
            }
        s = summary[g]
        s["n"] += 1
        s["nbs_detected_pct"] += int(p["nbs_detected"])
        s["bh4_responsive_pct"] += int(p["bh4_responsive"])
        s["csf_hva_low_pct"] += int(p["csf_hva_low"])
        s["csf_5hiaa_low_pct"] += int(p["csf_5hiaa_low"])
        s["dystonia_pct"] += int(p["dystonia_movement_disorder"])
        s["oculogyric_pct"] += int(p["oculogyric_crises"])
        s["seizures_pct"] += int(p["seizures"])
        s["ldopa_responsive_pct"] += int(p["ldopa_responsive"])
        s["diurnal_fluctuation_pct"] += int(p["diurnal_fluctuation"])
        s["mean_onset_yrs"] += p["onset_years"]
        s["mean_phe"] += p["phe_peak_umol_L"]

    gene_summaries = []
    for g, s in summary.items():
        n = s["n"]
        gene_summaries.append({
            "gene": g,
            "n": n,
            "nbs_detected_pct": round(100 * s["nbs_detected_pct"] / n, 1),
            "bh4_responsive_pct": round(100 * s["bh4_responsive_pct"] / n, 1),
            "csf_hva_low_pct": round(100 * s["csf_hva_low_pct"] / n, 1),
            "csf_5hiaa_low_pct": round(100 * s["csf_5hiaa_low_pct"] / n, 1),
            "dystonia_movement_disorder_pct": round(100 * s["dystonia_pct"] / n, 1),
            "oculogyric_crises_pct": round(100 * s["oculogyric_pct"] / n, 1),
            "seizures_pct": round(100 * s["seizures_pct"] / n, 1),
            "ldopa_responsive_pct": round(100 * s["ldopa_responsive_pct"] / n, 1),
            "diurnal_fluctuation_pct": round(100 * s["diurnal_fluctuation_pct"] / n, 1),
            "mean_onset_years": round(s["mean_onset_yrs"] / n, 2),
            "mean_phe_umol_L": round(s["mean_phe"] / n, 1),
        })

    return {
        "atlas": "Hereditary Biopterin (BH4) Metabolism Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": len(all_patients),
        "seeds": list(range(2670, 2678)),
        "gene_summaries": gene_summaries,
        "bh4_pathway_classification": [
            "BH4 Synthesis Defects (HPA + NT deficiency): GCH1-AR (GTPCH-I; AR-severe), PTS (PTPS; most common 60-75%)",
            "BH4 Recycling Defects (HPA + NT deficiency + secondary folate): QDPR (DHPR; folinic acid MANDATORY; basal ganglia calcification)",
            "BH4 Minor Pathway/HNF1: PCBD1 (PCD; BENIGN transient HPA; no NT deficiency; primapterinuria PATHOGNOMONIC)",
            "BH4 Synthesis NBS-MISS (normal Phe + severe NT deficiency): SPR (sepiapterin reductase; liver-brain asymmetry; always missed by NBS)",
            "BH4-Chaperone/Cochaperone: DNAJC12 (PAH + AADC cochaperone; BH4 NORMAL; HPA + NT deficiency; novel 2017)",
            "BH4-DEPENDENT Enzyme Defects (NOT BH4 genes; BH4 normal): TH (infantile parkinsonism/DRD-B; HVA low + 5-HIAA NORMAL), DDC (AADC; oculogyric crises PATHOGNOMONIC; BOTH HVA + 5-HIAA absent; gene therapy Upstaza EU 2022)",
            "GCH1-AD: Dopa-Responsive Dystonia (DRD/Segawa) — AD form; diurnal fluctuation PATHOGNOMONIC; L-DOPA dramatic response; NBS negative",
        ],
        "nbs_gap_genes": ["SPR", "TH", "DDC"],
        "normal_phe_genes": ["SPR", "TH", "DDC"],
        "ldopa_highly_responsive": ["GCH1 (AD-DRD)", "TH (near-complete remission)"],
        "gene_therapy_approved": ["DDC (Upstaza EU/UK 2022 — eladocagene exuparvovec)"],
        "folinic_acid_mandatory": ["QDPR (secondary folate deficiency via qBH2-DHFR inhibition)"],
        "csf_mandatory_genes": ["GCH1", "PTS", "QDPR", "SPR", "DNAJC12", "TH", "DDC"],
    }


def generate_breakdown():
    all_entries = []
    for idx, gene_data in enumerate(ATLAS_GENES):
        patients = _generate_patients(idx, n=40, seed=2670 + idx)
        g = gene_data["gene"]
        n = len(patients)
        all_entries.append({
            "gene": g,
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "inheritance_summary": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "patients_n": n,
            "mean_onset_years": round(sum(p["onset_years"] for p in patients) / n, 2),
            "mean_phe_umol_L": round(sum(p["phe_peak_umol_L"] for p in patients) / n, 1),
            "nbs_detected_pct": round(100 * sum(p["nbs_detected"] for p in patients) / n, 1),
            "bh4_responsive_pct": round(100 * sum(p["bh4_responsive"] for p in patients) / n, 1),
            "csf_hva_low_pct": round(100 * sum(p["csf_hva_low"] for p in patients) / n, 1),
            "csf_5hiaa_low_pct": round(100 * sum(p["csf_5hiaa_low"] for p in patients) / n, 1),
            "dystonia_movement_disorder_pct": round(100 * sum(p["dystonia_movement_disorder"] for p in patients) / n, 1),
            "oculogyric_crises_pct": round(100 * sum(p["oculogyric_crises"] for p in patients) / n, 1),
            "seizures_pct": round(100 * sum(p["seizures"] for p in patients) / n, 1),
            "ldopa_responsive_pct": round(100 * sum(p["ldopa_responsive"] for p in patients) / n, 1),
            "diurnal_fluctuation_pct": round(100 * sum(p["diurnal_fluctuation"] for p in patients) / n, 1),
        })
    return {"atlas": "Hereditary Biopterin (BH4) Metabolism Atlas", "gene_entries": all_entries}


def generate_definitions():
    gene_entries = {g["gene"]: g["pathognomonic"] for g in ATLAS_GENES}

    glossary = {
        "BH4 (Tetrahydrobiopterin) — Synthesis Pathway Overview": (
            "TETRAHYDROBIOPTERIN (BH4) SYNTHESIS — 3 ENZYMATIC STEPS: "
            "Step 1: GTP → (GCH1/GTPCH-I) → 7,8-dihydroneopterin triphosphate; "
            "Step 2: 7,8-dihydroneopterin triphosphate → (PTS/PTPS) → 6-pyruvoyltetrahydropterin; "
            "Step 3: 6-pyruvoyltetrahydropterin → (SPR/sepiapterin reductase, 2 NADPH reductions) → BH4; "
            "BH4 RECYCLING: BH4 oxidised in catalytic cycle → qBH2 → (QDPR/DHPR + NADH) → BH4; "
            "MINOR BRANCH: pterin-4-carbinolamine → (PCBD1/PCD) → qBH2 (fed into QDPR); "
            "WITHOUT PCBD1: pterin-4-carbinolamine → 7-biopterin (primapterin, inactive); "
            "BH4-DEPENDENT ENZYMES (critical): "
            "  PAH (phenylalanine hydroxylase) — liver; Phe → Tyr; deficiency → HPA/PKU; "
            "  TH (tyrosine hydroxylase) — brain; Tyr → L-DOPA; rate-limiting dopamine synthesis; "
            "  TPH1/TPH2 (tryptophan hydroxylase) — gut/brain; Trp → 5-HTP; rate-limiting serotonin synthesis; "
            "  NOS1/2/3 (nitric oxide synthases) — endothelial, neuronal, inducible; "
            "CLINICAL RULE: BH4 deficiency → BOTH HPA + neurotransmitter deficiency (unless peripheral subtype PTS/PCBD1)"
        ),
        "Urine Pterins — Diagnostic Pattern (MANDATORY for all NBS-positive HPA)": (
            "URINE PTERIN PROFILE — diagnostic pattern for each BH4 deficiency: "
            "GCH1-AR: BOTH neopterin LOW + biopterin VERY LOW (cannot synthesise any pterins → neopterin also absent); "
            "PTS/PTPS: neopterin VERY HIGH + biopterin VERY LOW (step 2 blocked; neopterin accumulates upstream); "
            "QDPR/DHPR: neopterin NORMAL + biopterin NORMAL (synthesis intact; recycling defect — qBH2 not regenerated); "
            "PCBD1/PCD: primapterin (7-biopterin) ELEVATED + neopterin NORMAL + biopterin mildly elevated; "
            "SPR: 7-biopterin + sepiapterin detectable; NORMAL neopterin; Phe NORMAL (see liver-brain asymmetry); "
            "DNAJC12: pterins COMPLETELY NORMAL (not a pterin gene — chaperone defect); "
            "TH: pterins COMPLETELY NORMAL (BH4 normal; TH uses BH4 as cofactor); "
            "DDC: pterins COMPLETELY NORMAL (BH4 normal; DDC converts L-DOPA → dopamine); "
            "PROTOCOL: ALL NBS-positive HPA (elevated Phe) → MANDATORY urine pterin profile BEFORE starting any treatment; "
            "NEVER diagnose classic PKU (PAH deficiency) without excluding BH4 deficiencies first; "
            "MINIMUM PROFILE: neopterin, biopterin, primapterin, DHPR DBS assay"
        ),
        "CSF Neurotransmitters — Mandatory Investigation in BH4 Disorders": (
            "CSF NEUROTRANSMITTER ANALYSIS — essential for treatment stratification: "
            "METABOLITES MEASURED: HVA (homovanillic acid = dopamine metabolite), 5-HIAA (5-hydroxyindolacetic acid = serotonin metabolite); "
            "ALSO: BH4, neopterin, biopterin, 5-MTHF (folate); "
            "PATTERN INTERPRETATION: "
            "  HVA low + 5-HIAA low + BH4 low: GCH1-AR, PTS-central, QDPR, SPR; "
            "  HVA low + 5-HIAA NORMAL + BH4 normal: TH deficiency (isolated dopamine); "
            "  HVA absent + 5-HIAA absent + 3-OMD elevated + BH4 normal: DDC/AADC deficiency; "
            "  HVA normal + 5-HIAA normal + BH4 normal: PCBD1 (BENIGN), PTS-peripheral; "
            "  CSF 5-MTHF low + BH4 low: QDPR (secondary folate — folinic acid mandatory); "
            "  HVA low + 5-HIAA low + BH4 low + sepiapterin elevated: SPR deficiency; "
            "CRITICAL RULE: CSF MUST BE OBTAINED BEFORE STARTING L-DOPA "
            "(L-DOPA treatment raises HVA → masks the diagnostic low HVA); "
            "TIMING: obtain CSF within first week of life if NBS positive; immediately in any symptomatic child; "
            "MANDATORY GENES: PTS (peripheral vs central), GCH1, QDPR, SPR, DNAJC12, TH, DDC; "
            "not required for PCBD1 (benign, CSF normal) but may be obtained to confirm"
        ),
        "NBS Gaps in BH4 Disorders — SPR, TH, DDC Always Missed": (
            "NBS (NEWBORN SCREENING) GAPS IN BH4/DOPAMINE-SEROTONIN METABOLISM: "
            "DETECTED BY STANDARD NBS (elevated Phe): GCH1-AR, PTS, QDPR, PCBD1, DNAJC12; "
            "MISSED BY STANDARD NBS (Phe NORMAL): "
            "SPR (sepiapterin reductase): ALWAYS MISSED — Phe NORMAL because liver uses alternative carbonyl reductase 1 pathway; "
            "  brain has NO alternative → severe neurological disease develops before diagnosis; "
            "  typical delay: 1-5+ years; misdiagnosed as cerebral palsy, DRD; "
            "TH (tyrosine hydroxylase): ALWAYS MISSED — Phe NORMAL; dopamine deficiency only; "
            "  CSF HVA very low, 5-HIAA normal; L-DOPA trial in any child with movement disorder; "
            "DDC (AADC): ALWAYS MISSED — Phe NORMAL; "
            "  oculogyric crises in first year may not reach metabolic team promptly; "
            "GCH1-AD (Segawa DRD): MISSED — Phe NORMAL; AD-DRD often diagnosed 2-12 years of age; "
            "  diurnal fluctuation pattern is KEY clinical clue; "
            "NBS IMPROVEMENT PROPOSALS: "
            "  CSF neurotransmitter-based second-tier testing in movement disorder; "
            "  AADC enzyme activity in blood spots (detects DDC); "
            "  plasma 3-OMD for DDC; "
            "CLINICAL RULE: ANY child with dystonia/parkinsonism/movement disorder of unknown cause → "
            "BH4 + pterin profile + CSF neurotransmitters + TH + DDC gene panel"
        ),
        "QDPR Secondary Folate Deficiency — Folinic Acid Mandatory": (
            "QDPR (DHPR DEFICIENCY) — UNIQUE SECONDARY FOLATE DEFICIENCY MECHANISM: "
            "MECHANISM: QDPR LOF → qBH2 accumulates → qBH2 inhibits DHFR (dihydrofolate reductase); "
            "  DHFR normally: DHF → THF (dihydrofolate → tetrahydrofolate); "
            "  qBH2 competitive inhibition → THF deficiency → 5-methylTHF trap → CSF 5-MTHF critically low; "
            "  functional folate deficiency = analogous to methotrexate toxicity in brain; "
            "CONSEQUENCES: "
            "  neuronal methionine synthase (MTR) requires 5-MTHF — impaired; "
            "  neuronal SAM depleted — impaired methylation; "
            "  demyelination risk; "
            "BASAL GANGLIA CALCIFICATION: calcium-pteridine complexes (qBH2 + Ca2+) deposit; "
            "  bilateral symmetric calcifications on CT/MRI; "
            "FOLINIC ACID (5-formylTHF) BYPASS: "
            "  folinic acid enters folate cycle directly — BYPASSES DHFR (the inhibited enzyme); "
            "  does NOT require DHFR; directly metabolised to THF and 5-MTHF; "
            "  restores brain folate despite qBH2-mediated DHFR inhibition; "
            "DOSE: folinic acid 10-20 mg/day (not folic acid — folic acid REQUIRES DHFR); "
            "FOLIC ACID IS WRONG: folic acid → DHF → THF requires DHFR → blocked by qBH2; "
            "  always use folinic acid (leucovorin, 5-formylTHF) NOT folic acid in QDPR; "
            "MONITORING: CSF 5-MTHF target normal range; CT for calcification progression; "
            "METHOTREXATE ABSOLUTE CI: direct DHFR inhibitor — catastrophic in QDPR"
        ),
        "L-DOPA Trial — Movement Disorder Diagnostic Principle": (
            "L-DOPA DIAGNOSTIC AND THERAPEUTIC TRIAL IN CHILDHOOD MOVEMENT DISORDERS: "
            "RATIONALE: multiple hereditary movement disorders are HIGHLY RESPONSIVE to L-DOPA; "
            "  a dramatic L-DOPA response is DIAGNOSTIC for dopamine-deficiency states; "
            "HIGHLY RESPONSIVE (near-complete remission): GCH1-AD (DRD/Segawa), TH deficiency; "
            "MODERATELY RESPONSIVE: GCH1-AR, QDPR, SPR; "
            "PARTIALLY RESPONSIVE: PTS-central, DNAJC12; "
            "NOT RESPONSIVE: DDC/AADC (cannot convert L-DOPA to dopamine); "
            "PROTOCOL: L-DOPA/carbidopa 0.5-1 mg/kg/day → titrate weekly; "
            "  GCH1-AD DRD: typically 80-100% improvement at 1-2 mg/kg/day; "
            "  TH deficiency: dramatic improvement, may need 5-10 mg/kg/day for type B; "
            "DIURNAL FLUCTUATION TEST: give first L-DOPA dose in morning → observe evening improvement; "
            "BEFORE L-DOPA: obtain CSF (L-DOPA raises HVA → masks diagnostic pattern); "
            "AFTER L-DOPA TRIAL: if no response at adequate dose → consider DDC/AADC, "
            "DYT genes, other causes; "
            "CEREBRAL PALSY DDx: ALL atypical CP, especially with diurnal fluctuation → "
            "L-DOPA trial MANDATORY; DRD is fully treatable; CP is not; "
            "GENE THERAPY NOTE: DDC/AADC gene therapy (Upstaza) is now available — "
            "L-DOPA non-response + oculogyric crises → AADC gene therapy pathway"
        ),
        "Oculogyric Crises — AADC Deficiency and Differential Diagnosis": (
            "OCULOGYRIC CRISES (OGC) — FORCED SUSTAINED UPWARD EYE DEVIATION: "
            "DEFINITION: episodic involuntary sustained upward deviation of both eyes; "
            "  duration: minutes to hours; may recur multiple times daily; "
            "  associated features: neck extension, facial grimacing, anxiety, salivation; "
            "DDC/AADC DEFICIENCY: 55-85% prevalence — MOST PATHOGNOMONIC FEATURE; "
            "  often first neurological feature recognised (onset infancy); "
            "  high-amplitude OGC triggered by fatigue, illness, excitement; "
            "GCH1-AR: OGC in ~30% (dopamine deficiency drives basal ganglia dysfunction); "
            "SPR: OGC in ~25%; "
            "PTS-central, QDPR: OGC in ~20-25%; "
            "TH: oculomotor abnormalities (ptosis, gaze palsy) rather than classic OGC; "
            "DRUG-INDUCED OGC (DDx): metoclopramide, antipsychotics (dopamine receptor blockers) → "
            "  acutely reversible; distinguish by medication history; "
            "ACUTE OGC MANAGEMENT: benzodiazepines (diazepam IV) for acute episode; "
            "LONG-TERM MANAGEMENT: "
            "  DDC: gene therapy (Upstaza) + pyridoxine + selegiline + dopamine agonists; "
            "  GCH1/PTS/QDPR/SPR: L-DOPA + 5-HTP + BH4 (neurotransmitter restoration); "
            "DIAGNOSTIC CLUE: OGC + hypotonia + developmental delay in infancy → "
            "  CSF neurotransmitters IMMEDIATELY → DDC is gene therapy eligible; "
            "UPSTAZA ELIGIBILITY: OGC + DDC confirmed + age 18m-6y → refer to gene therapy centre"
        ),
        "Upstaza (Eladocagene Exuparvovec) — Gene Therapy for AADC Deficiency": (
            "UPSTAZA — FIRST APPROVED GENE THERAPY FOR AADC (DDC) DEFICIENCY: "
            "APPROVAL: EMA (European Medicines Agency) — July 2022; MHRA (UK) — 2022; "
            "PRODUCT: eladocagene exuparvovec (AAV2-hAADC); "
            "VECTOR: adeno-associated virus serotype 2 (AAV2); transgene: human AADC cDNA; "
            "ROUTE: stereotactic neurosurgical bilateral putamen injection; "
            "MECHANISM: AAV2 transduces striatal neurons → permanent AADC expression → "
            "  L-DOPA (from TH, which is intact) → dopamine (now converted by transgene AADC); "
            "ELIGIBILITY: "
            "  age 18 months to 6 years; "
            "  confirmed biallelic DDC mutations; "
            "  severe phenotype (classic infantile form); "
            "CLINICAL RESULTS: "
            "  oculogyric crises: resolved in >80% of patients; "
            "  motor milestones: most treated patients gained sitting/standing/walking ability; "
            "  durability: 5-10 year follow-up data shows sustained benefit; "
            "  some patients: near-normal motor function; "
            "PRE-GENE THERAPY PREPARATION: "
            "  selegiline to reduce dopamine catabolism prior to surgery; "
            "  stop dopamine agonists before procedure; "
            "POST-GENE THERAPY: "
            "  dopamine dysregulation syndrome (transient): first weeks post-injection; "
            "  gradually add L-DOPA as needed (AADC now present to convert it); "
            "CENTRES: specialist neurosurgical centres in EU/UK; "
            "PRE-APPROVAL OUTCOMES: 26 patients in key clinical trials — statistically significant improvement"
        ),
    }

    return {
        "atlas": "Hereditary Biopterin (BH4) Metabolism Atlas",
        "gene_entries": gene_entries,
        "bh4_glossary": glossary,
    }
