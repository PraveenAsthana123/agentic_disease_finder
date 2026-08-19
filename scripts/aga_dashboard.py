#!/usr/bin/env python3
"""AGA / Aspartylglucosaminuria Epilepsy Dashboard — seed data module.

Aspartylglucosaminuria (AGU): aspartylglucosaminidase (AGA, also called
glycosylasparaginase) deficiency (AGA, 4q34.3, AR).
Oligosaccharidosis group (not MPS): aspartylglucosamine accumulates in lysosomes
→ progressive neurodegeneration after a period of apparently normal development.

KEY DISTINGUISHING FEATURES:
  1. BORN NORMAL → PROGRESSIVE — infants appear normal (only recurrent infections
     and mild coarse facies early); intellectual regression typically starts age 5-10yr;
     UNIQUE among LSDs — early childhood seems near-normal, then regression.
  2. FINNISH FOUNDER MUTATION — p.Cys163Ser (c.488G>C) homozygous accounts for ~98%
     of Finnish patients; most common LSD in Finland (1:18,000 births); near-total
     allele frequency in Finnish population; non-Finnish patients often compound-het.
  3. BEHAVIORAL PHENOTYPE DOMINANT — aggression, hyperactivity, self-injury PROMINENT;
     psychiatric misdiagnosis (ADHD, autism, conduct disorder) precedes correct diagnosis
     by 5-10yr in many cases; behavioral problems worsen with neurodegeneration.
  4. EPILEPSY 60-70% — NOT the presenting feature (behavior/ID precede); tonic-clonic
     and myoclonic; onset typically age 10-30yr; seizures follow behavioral regression.
  5. NO ERT APPROVED (2026) — gene therapy (AAV-AGA) in early Phase I only (2025 data);
     bone marrow transplant (BMT) historically attempted; limited evidence for CNS benefit.
  6. URINE DIAGNOSIS: aspartylglucosamine (2-acetamido-1-(β-L-aspartamido)-1,2-dideoxy-
     β-D-glucose) elevated — NOT GAG screen (not MPS); confirmed by AGA enzyme assay.

Enzyme biology:
  AGA (aspartylglucosaminidase, also glycosylasparaginase) cleaves the bond between
  aspartylglucosamine and asparagine during N-linked oligosaccharide catabolism.
  When AGA is deficient, aspartylglucosamine (GlcNAc-Asn) accumulates in lysosomes
  throughout the body, particularly in neurons, hepatocytes, and connective tissue.
  AGA is a serine N-terminal nucleophile (Ntn) hydrolase; autoactivated by autocatalytic
  processing (pro-enzyme → active heterodimer α2β2); the p.Cys163Ser Finnish founder
  mutation disrupts the catalytic site, abolishing processing and activity.

Pharmacology:
  Seizure management is the primary therapeutic target in AGA.
  POLG1 exclusion mandatory before VPA (CPIC Grade A — mitochondrial risk).
  NO sodium-channel CI (unlike MPS-III Sanfilippo myoclonus); CBZ/OXC can be used
  IF no myoclonic component (check EEG carefully).
  VGB: no cherry red spot, no corneal clouding → VGB not CI for visual reasons,
    BUT behavioral side effects (agitation) especially problematic given behavioral phenotype.
  PHT: not absolute CI (no myoclonus dominant seizures usually), but avoid in
    patients with myoclonic component (check EEG).
  Typical antipsychotics: HIGH RISK for behavioral features — basal ganglia
    oligosaccharide accumulation → EPS sensitivity; use atypicals (quetiapine preferred).
  Atypical antipsychotics: PREFERRED for behavioral management (quetiapine, risperidone
    low-dose with caution); behavioral phenotype management is major quality-of-life issue.
"""
import random

GENE = "AGA"
LOCUS = "4q34.3"
OMIM_GENE = "613228"
OMIM_DISEASE = "208400"
INHERITANCE = "Autosomal Recessive (AR) — biallelic AGA LOF; both males AND females equally affected"
COHORT_SIZE = 40
DISEASE_MECHANISM = (
    "Aspartylglucosaminidase (AGA) deficiency → lysosomal accumulation of "
    "aspartylglucosamine (2-acetamido-1-(β-L-aspartamido)-1,2-dideoxy-β-D-glucose, "
    "GlcNAc-Asn) throughout the body. AGA normally cleaves the N-glycosidic bond "
    "between GlcNAc and asparagine during catabolism of N-linked glycoproteins after "
    "endocytosis and lysosomal protein degradation. Without AGA, GlcNAc-Asn (aspartylglucosamine) "
    "accumulates progressively in lysosomes — especially in neurons, hepatocytes, "
    "macrophages, and connective tissue. "
    "DEVELOPMENTAL TRAJECTORY: The 'born normal → progressive' pattern is characteristic: "
    "infants appear nearly normal (only recurrent otitis media, upper respiratory infections, "
    "and subtle coarsening of features as early clues); intellectual regression typically "
    "begins age 5-10yr when lysosomal burden exceeds neuronal compensation; behavioral "
    "problems (aggression, hyperactivity, self-injury) often PRECEDE seizures; seizures "
    "emerge at age 10-30yr after behavioral regression is established. "
    "Finnish founder: p.Cys163Ser (c.488G>C) homozygous — AGA proenzyme cannot autocatalytically "
    "activate (autocatalytic processing blocked → no active α2β2 heterodimer → zero enzyme "
    "activity). Finland prevalence 1:18,000; near-complete allele fixation in Finnish population. "
    "Seizure mechanism: progressive lysosomal GlcNAc-Asn storage in cortical neurons → "
    "neuronal dysfunction → cortical hyperexcitability → tonic-clonic and myoclonic seizures. "
    "Behavioral mechanism: accumulation in prefrontal cortex and limbic structures → "
    "disinhibition, impulse control failure, aggression; dopamine pathway disruption "
    "in striatum → hyperactivity; basal ganglia storage → EPS sensitivity to D2 blockers. "
    "No ERT approved 2026: recombinant AGA enzyme replacement has been investigated but "
    "CNS penetration is the primary limiting factor; gene therapy (AAV-AGA intrathecal) "
    "in Phase I (Ultragenyx UX143, 2025 data). BMT/HSCT historically used (Finnish experience "
    "largest); limited CNS disease-modifying effect; somatic benefits better documented "
    "than cognitive outcomes."
)

ETIOLOGIES = [
    {
        "name": "Finnish Founder Homozygous (p.Cys163Ser / Finnish AGU)",
        "pct": 35,
        "n": 14,
        "seizure_risk": "65-75% (progressive; onset age 10-30yr; after behavioral regression)",
        "eeg": "Generalized slow spike-wave complexes; multifocal discharges; progressive background "
               "slowing paralleling cognitive decline; polyspike-wave with myoclonic component "
               "in 30-40%; EEG often near-normal early (age 5-10yr before seizure onset); "
               "sleep EEG: increased interictal discharges; normal EEG background early "
               "in disease course does not rule out AGA; photoparoxysmal response 20-30%",
        "variant_detail": (
            "Homozygous p.Cys163Ser (c.488G>C) — Finnish founder; enzyme absent (<1% control); "
            "CLASSIC Finnish AGU phenotype: born normal, recurrent otitis media age 0-3yr, "
            "subtle coarse facies, regression starts age 5-8yr (first signs: social withdrawal, "
            "speech regression, school failure), then progressive ID (moderate-severe by age 20), "
            "behavioral problems DOMINANT (aggression, hyperactivity, self-injury, autistic features), "
            "macroorchidism in adult males, seizures emerge age 10-25yr after behavioral regression; "
            "survival typically 4th-5th decade; diagnosis often delayed 5-10yr (initially misdiagnosed "
            "as ADHD, autism, conduct disorder); urine aspartylglucosamine + AGA enzyme assay confirms"
        ),
        "hsct_eligible": True,   # BMT historically attempted in Finnish cohort
        "ert_eligible": False,
    },
    {
        "name": "Non-Finnish Compound-Het (Missense + Splice / European)",
        "pct": 28,
        "n": 11,
        "seizure_risk": "60-70% (onset variable age 8-25yr; seizures follow behavioral/cognitive regression)",
        "eeg": "Focal temporal or frontal discharges; generalized slow spike-wave; "
               "moderate background slowing; less severe than Finnish homozygous at same age; "
               "polyspike complexes in myoclonic subset; EEG-guided seizure classification "
               "essential — myoclonic component changes pharmacological approach (PHT CI); "
               "absence seizures in juvenile onset (3Hz S-W in subset)",
        "variant_detail": (
            "Compound-het: one missense (partial residual enzyme 5-15%) + one splice/frameshift; "
            "variable phenotype based on residual activity; attenuated relative to homozygous null; "
            "onset age 2-15yr; milder intellectual disability in some; behavioral features still "
            "prominent (aggression, hyperactivity); seizures often focal before secondary "
            "generalization; diagnosis commonly delayed due to variable phenotype; non-Finnish "
            "European cases scattered (UK, Italy, Germany); urine aspartylglucosamine + AGA enzyme assay"
        ),
        "hsct_eligible": True,
        "ert_eligible": False,
    },
    {
        "name": "Biallelic Null — Severe Early Onset (Non-Finnish)",
        "pct": 20,
        "n": 8,
        "seizure_risk": "70-80% (more severe; earlier seizure onset age 5-15yr; higher DRE rate)",
        "eeg": "Multifocal discharges; generalized slow spike-wave; progressive deterioration; "
               "high-amplitude irregular slow background (theta/delta predominant); "
               "EEG deterioration faster than Finnish founder group at same age; "
               "polyspike-wave 40-50%; status epilepticus risk in severe null group; "
               "cerebellar-cortical loop involved; EEG mandatory to guide PHT decision "
               "(myoclonic component = PHT CI even if not expected in AGU)",
        "variant_detail": (
            "Biallelic null/frameshift — non-Finnish: enzyme absent (<1% control); severe "
            "early-onset phenotype; regression starts earlier (age 2-6yr); severe intellectual "
            "disability by age 15; profound behavioral phenotype; macroorchidism in males; "
            "coarse facies prominent; hepatomegaly mild; seizures earlier and harder to control; "
            "DRE 35-45%; status epilepticus risk higher; urine aspartylglucosamine marked; "
            "vacuolated lymphocytes on peripheral smear (less dramatic than Sialidosis Type II "
            "but detectable in null cases); BMT considered under age 6 if suitable donor"
        ),
        "hsct_eligible": True,
        "ert_eligible": False,
    },
    {
        "name": "p.Cys163Ser Compound-Het (Finnish + Severe Allele)",
        "pct": 12,
        "n": 5,
        "seizure_risk": "65-75% (intermediate; mildly attenuated vs homozygous Finnish)",
        "eeg": "Similar to Finnish homozygous but slightly earlier onset; generalized slow "
               "spike-wave; focal temporal component possible; EEG mirrors cognitive trajectory; "
               "background alpha preserved longer than biallelic null; "
               "photo-sensitivity 15-25%; EEG essential before PHT consideration",
        "variant_detail": (
            "Compound-het: p.Cys163Ser (Finnish founder allele) + severe second allele (null or "
            "splice); residual enzyme 2-8%; intermediate phenotype between homozygous Finnish "
            "and homozygous null; onset age 3-12yr; moderate-severe intellectual disability; "
            "behavioral problems moderate; Finnish ethnic background but one allele non-Finnish; "
            "compound-het detected by full AGA sequencing; parents each carry one allele "
            "(not both carriers of Finnish founder)"
        ),
        "hsct_eligible": True,
        "ert_eligible": False,
    },
    {
        "name": "Rare Missense (Residual Activity / Attenuated Phenotype)",
        "pct": 5,
        "n": 2,
        "seizure_risk": "40-55% (attenuated; mild ID; seizures late and mild)",
        "eeg": "Near-normal background with mild slowing; rare focal discharges; "
               "seizures may be first clinical presentation in attenuated cases; "
               "EEG may be essentially normal between seizures; "
               "MRI: normal or mild white matter changes; "
               "enzyme assay: 15-30% residual activity (compared to <1% in null-null)",
        "variant_detail": (
            "Biallelic hypomorphic missense: residual enzyme 15-30% control; attenuated phenotype; "
            "mild-moderate intellectual disability; behavioral features present but less severe; "
            "seizures mild (occasional GTCS); late onset (age 15-40yr); misdiagnosed as "
            "non-specific intellectual disability without genetic workup; urine aspartylglucosamine "
            "mildly elevated (borderline on routine screen); enzyme assay essential; "
            "gene sequencing confirms; may be enriched in isolated populations with founder effects"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
]

SEIZURE_TYPES = [
    {"type": "Generalized Tonic-Clonic Seizures (GTCS)", "pct": 75,
     "note": "Most common seizure type in AGA; onset age 10-30yr after behavioral regression; "
             "VPA Level B (POLG1 mandatory first); LEV Level B; CBZ/OXC acceptable if NO myoclonus "
             "(check EEG carefully — myoclonic component changes CI status)"},
    {"type": "Myoclonic Seizures", "pct": 35,
     "note": "Myoclonic component in subset; EEG-confirmed polyspike-wave; "
             "PHT AVOID if myoclonus present (worsens cortical myoclonus); "
             "VPA + LEV combination preferred; Clonazepam adjunct for myoclonus control"},
    {"type": "Focal Onset Seizures", "pct": 30,
     "note": "Temporal or frontal focal onset; secondary generalization common; "
             "CBZ/OXC appropriate IF EEG confirms no myoclonic component; "
             "lacosamide or oxcarbazepine for focal without myoclonus"},
    {"type": "Absence Seizures (3Hz S-W)", "pct": 20,
     "note": "Juvenile subset (age 5-15yr); EEG-confirmed 3Hz spike-wave; "
             "Ethosuximide or VPA (POLG1 mandatory); avoid CBZ/OXC (worsens absence)"},
    {"type": "Myoclonic-Atonic Seizures", "pct": 12,
     "note": "Severe null group; fall risk; VPA + CLB combination; protective headgear; "
             "PHT absolute CI if myoclonic-atonic component confirmed on EEG"},
]

TRIGGERS = [
    {"trigger": "Sleep Deprivation / Irregular Sleep", "pct": 65,
     "note": "DOMINANT trigger — behavioral dysregulation disrupts sleep; irregular sleep "
             "→ seizures; sleep hygiene education (caregiver training essential); "
             "melatonin Level B (2-10mg HS) for sleep-seizure nexus; "
             "caregivers often sleep-deprived themselves"},
    {"trigger": "Behavioral Crisis / Agitation", "pct": 55,
     "note": "Behavioral outbursts (aggression, self-injury) may immediately precede seizures; "
             "behavioral management = seizure prevention; atypical antipsychotic "
             "(quetiapine preferred) for behavioral crises; NOT typical AP (EPS risk)"},
    {"trigger": "Febrile Illness", "pct": 48,
     "note": "Recurrent infections (especially otitis media, RTI) are part of AGU phenotype "
             "from early childhood; febrile seizures in subset; rescue benzodiazepine "
             "protocol; parenteral AED if oral impossible during febrile illness"},
    {"trigger": "Missed AED Dose", "pct": 40,
     "note": "Behavioral dysregulation may cause non-adherence; caregiver-administered AEDs "
             "essential (patients cannot self-manage reliably); liquid formulations preferred; "
             "depot/long-acting formulations where available"},
    {"trigger": "Photic Stimulation", "pct": 28,
     "note": "Photoparoxysmal response 20-30%; avoid strobe environments; television seizures "
             "possible; photosensitivity usually mild vs NCL or Sialidosis"},
    {"trigger": "Iatrogenic — Typical Antipsychotics (D2 blockade)", "pct": 20,
     "note": "Typical AP prescribed for behavioral phenotype → EPS (basal ganglia storage), "
             "seizure threshold lowering, metabolic side effects; AVOID typical AP; "
             "use atypical AP (quetiapine, risperidone low-dose); clozapine for refractory "
             "behavioral aggression (weekly WBC monitoring)"},
]

TREATMENTS = [
    {"drug": "Levetiracetam (LEV)", "level": "Level B",
     "indication": "FIRST-LINE broad-spectrum: GTCS + focal + myoclonic; SV2A modulation; "
                   "1000-3000mg/day; IV formulation for status/NPO; preferred over PHT for "
                   "acute management; behavioral side effects (irritability) require monitoring — "
                   "important given pre-existing behavioral phenotype",
     "ci": "Behavioral worsening (irritability, aggression) especially problematic in AGA "
           "behavioral phenotype — monitor closely; supplement with B6 (pyridoxine) if "
           "behavioral SE develops (empirically used)"},
    {"drug": "Valproate / VPA (Depakote / Epilim)", "level": "Level B",
     "indication": "GTCS + myoclonus + absence; broad-spectrum Level B; 500-2500mg/day; "
                   "MANDATORY POLG1 exclusion BEFORE prescribing (CPIC Grade A); "
                   "IDH1/2 optional; effective for behavioral stabilization in some patients "
                   "(mood stabilization as secondary benefit)",
     "ci": "POLG1/POLG2 carriers: ABSOLUTE CI (Alpers-Huttenlocher; fulminant hepatic failure); "
           "pregnancy (teratogenic — avoid in women of childbearing age); pancreatitis; weight gain"},
    {"drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)", "level": "Level B",
     "indication": "Focal seizures WITHOUT myoclonic component; CBZ 400-1600mg/day; "
                   "OXC 600-2400mg/day; HLA-B1502 test before CBZ/OXC in SE-Asian heritage; "
                   "IMPORTANT: CHECK EEG before prescribing — if myoclonic component confirmed → "
                   "CBZ/OXC RELATIVE CI (worsens myoclonus); safe for pure focal/GTCS",
     "ci": "RELATIVE CI if myoclonus present on EEG; HLA-B1502 SJS risk in SE-Asian; "
           "hyponatremia (OXC more common); drug interactions via CYP3A4"},
    {"drug": "Clonazepam (CLZ)", "level": "Level B",
     "indication": "Adjunct for myoclonic component; 0.5-2mg TID; GABAergic modulation; "
                   "useful in combination with LEV for myoclonus+GTCS burden; behavioral "
                   "anxiolysis as secondary benefit",
     "ci": "Sedation (adds to cognitive burden); tolerance; withdrawal rebound; "
           "never abrupt stop; behavioral disinhibition possible in AGA"},
    {"drug": "Quetiapine (Seroquel) — atypical AP for behavior", "level": "Level B",
     "indication": "BEHAVIORAL PHENOTYPE MANAGEMENT — aggression, self-injury, hyperactivity; "
                   "quetiapine preferred atypical AP: lowest EPS risk, D2 partial agonism, "
                   "some anticonvulsant properties; 25-300mg/day; improves quality of life "
                   "for caregivers; behavioral management = seizure prevention",
     "ci": "QTc monitoring (baseline ECG); metabolic monitoring (glucose, lipids, weight); "
           "sedation (timing dosing to behavioral needs); NOT typical AP (see CI below)"},
    {"drug": "Melatonin", "level": "Level B",
     "indication": "Sleep-seizure nexus; 2-10mg HS; reduces nocturnal seizure frequency; "
                   "safe in all ages; improves sleep in AGA caregivers too; "
                   "behavioral benefit (improved circadian rhythm) secondary to sleep improvement",
     "ci": "None clinically significant"},
    {"drug": "Lacosamide (Vimpat)", "level": "Level C",
     "indication": "Focal seizures refractory to CBZ/OXC; slow-inactivation sodium-channel "
                   "modulation; 200-400mg/day; IV formulation available; particularly useful "
                   "if CBZ/OXC causing hyponatremia or drug interactions; no myoclonus CI",
     "ci": "PR interval prolongation (ECG monitoring); dizziness; not first-line"},
    {"drug": "BMT / HSCT (allogeneic)", "level": "Level C",
     "indication": "Finnish BMT experience (largest cohort); best outcomes if performed under "
                   "age 6 before severe neurodegeneration; somatic benefits (stabilization) "
                   "better than CNS cognitive reversal; limited CNS disease-modification; "
                   "considered if suitable HLA-matched donor available; "
                   "not standard therapy (limited evidence + significant procedural risk)",
     "ci": "Procedural risk; CNS benefits limited; not standard; investigational"},
]

CONTRAINDICATIONS = [
    {"drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine, Fluphenazine)",
     "severity": "HIGH RISK",
     "reason": "BEHAVIORAL PHENOTYPE TRAP: typical APs are prescribed for behavioral "
               "features (aggression, hyperactivity) BUT cause severe EPS in AGA due to "
               "oligosaccharide accumulation in basal ganglia (globus pallidus, substantia nigra); "
               "risk of tardive dyskinesia, acute dystonic reactions, Parkinsonism; "
               "ALSO lower seizure threshold; use ATYPICAL AP instead (quetiapine preferred)"},
    {"drug": "Phenytoin (PHT) / Fosphenytoin — if myoclonus present",
     "severity": "AVOID if myoclonus on EEG",
     "reason": "PHT worsens myoclonic component if present (sodium-channel blockade "
               "paradoxically worsens cortical myoclonus); AGA typically NOT a classic "
               "progressive myoclonic epilepsy, but subset (35%) has myoclonic component; "
               "MANDATORY EEG before PHT — if polyspike-wave or myoclonus → AVOID; "
               "IV LEV is the safe acute alternative; PHT is NOT absolute CI in AGA "
               "unlike NEU1/FUCA1 but RELATIVE CI when myoclonus documented"},
    {"drug": "Carbamazepine / Oxcarbazepine — if myoclonus present",
     "severity": "RELATIVE CI if myoclonus on EEG",
     "reason": "CBZ/OXC worsen myoclonic component (similar to PHT mechanism); "
               "SAFE for pure focal or GTCS without myoclonus; "
               "CHECK EEG — if no myoclonic discharges → CBZ/OXC appropriate; "
               "if polyspike-wave → avoid; EEG mandatory before prescribing"},
    {"drug": "POLG1 carriers — Valproate",
     "severity": "ABSOLUTE CI (CPIC Grade A)",
     "reason": "MANDATORY POLG1/POLG2 sequencing before VPA; POLG1 carriers → "
               "VPA → acute hepatic failure (Alpers-Huttenlocher); CPIC Grade A; "
               "if POLG1+ → avoid VPA → use LEV ± CLZ ± CBZ/OXC (EEG-guided)"},
    {"drug": "Vigabatrin (VGB) — behavioral risk in AGA",
     "severity": "RELATIVE CI (behavioral)",
     "reason": "VGB causes agitation, behavioral worsening, irritability (especially "
               "in intellectual disability patients with pre-existing behavioral phenotype); "
               "AGA has PROMINENT behavioral phenotype (aggression, self-injury, hyperactivity); "
               "VGB adds to behavioral burden; NOT CI for visual reasons (no cherry red spot, "
               "no corneal clouding in AGA) but behavioral risk is the AGA-specific concern; "
               "avoid unless behavioral monitoring available"},
]


def get_overview():
    random.seed(42)
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "inheritance": INHERITANCE,
        "cohort_size": COHORT_SIZE,
        "disease_mechanism": DISEASE_MECHANISM,
        "kpis": {
            "epilepsy_prevalence": "60-70% (onset typically age 10-30yr; follows behavioral regression)",
            "drug_resistance_rate": "25-40% (moderate DRE; behavioral non-adherence confounds assessment)",
            "born_normal_window": "Age 0-5yr near-normal development — UNIQUE among oligosaccharidoses",
            "behavioral_phenotype": "Aggression/hyperactivity/self-injury DOMINANT (100% of severe cases)",
            "finnish_prevalence": "1:18,000 (most common LSD in Finland; p.Cys163Ser Finnish founder)",
            "psychiatric_misdiagnosis": "5-10yr diagnosis delay — ADHD/autism/conduct disorder precede AGU dx",
            "macroorchidism_males": "50-70% of adult males — diagnostic clue in male AGU patients",
            "vacuolated_lymphocytes": "40-60% positive — less prominent than Sialidosis Type II",
            "hsct_evidence": "Level C — Finnish BMT experience; somatic > cognitive CNS benefit",
            "ert_approved": "None approved (2026) — AAV-AGA gene therapy Phase I (UX143, 2025)",
            "polg1_mandatory": "MANDATORY before VPA (CPIC Grade A)",
            "urine_screen": "Aspartylglucosamine (GlcNAc-Asn) — NOT GAG screen (not MPS)",
            "cbz_oxc_status": "ACCEPTABLE if EEG confirms no myoclonic component (AGA-specific EEG rule)",
            "typical_ap_risk": "Typical antipsychotics HIGH RISK (behavioral phenotype trap — use atypicals)",
            "quetiapine_preferred": "Quetiapine Level B for behavioral management (lowest EPS risk atypical)",
        },
        "etiologies": ETIOLOGIES,
    }


def get_breakdown():
    random.seed(42)
    patients = []
    for eth in ETIOLOGIES:
        n = eth["n"]
        for i in range(n):
            is_null = "Null" in eth["name"] or "Severe" in eth["name"]
            is_finnish = "Finnish" in eth["name"]
            age_onset = (
                random.randint(10, 28) if is_finnish
                else (random.randint(5, 20) if is_null else random.randint(8, 25))
            )
            pts = {
                "id": f"AGA-{len(patients)+1:03d}",
                "etiology": eth["name"],
                "age_onset_yr": age_onset,
                "seizure_type": random.choice(
                    ["GTCS", "GTCS + Myoclonic", "Focal + GTCS", "Absence + GTCS",
                     "Myoclonic-Atonic"]
                ),
                "seizure_risk": eth["seizure_risk"],
                "eeg_pattern": eth["eeg"][:100] + "…",
                "treatment_1": random.choice(["Levetiracetam", "Valproate", "Carbamazepine"]),
                "treatment_2": random.choice(["Clonazepam", "Quetiapine", "Melatonin", "Lacosamide"]),
                "trigger": random.choice([t["trigger"] for t in TRIGGERS]),
                "ci_avoided": random.choice([c["drug"][:40] for c in CONTRAINDICATIONS]),
                "cherry_red_spot": False,   # AGA does NOT have cherry red spot
                "vacuolated_lymphocytes": random.random() < (0.55 if is_null else 0.35),
                "behavioral_phenotype": True,  # Universal in AGA
                "polg1_tested": True,
                "urine_aspartylglucosamine": "ELEVATED (GlcNAc-Asn accumulation — not GAG)",
                "aga_enzyme": f"{random.randint(1, 10) if not is_null else random.randint(0, 3)}% control",
                "hsct": eth.get("hsct_eligible", False) and random.random() < 0.15,
                "ert": False,
                "psychiatric_dx_before_agu": random.choice(["ADHD", "Autism", "Conduct Disorder", "None"]),
            }
            patients.append(pts)

    return {
        "cohort_size": COHORT_SIZE,
        "patients": patients,
        "seizure_summary": SEIZURE_TYPES,
        "trigger_summary": TRIGGERS,
        "treatment_summary": TREATMENTS,
        "contraindication_summary": CONTRAINDICATIONS,
        "etiology_breakdown": [
            {"name": e["name"], "pct": e["pct"], "n": e["n"]} for e in ETIOLOGIES
        ],
    }


def get_definitions():
    return {
        "glossary": [
            {"term": "AGA (Aspartylglucosaminidase)", "definition":
             "Lysosomal enzyme encoded by AGA gene (4q34.3); serine N-terminal nucleophile (Ntn) "
             "hydrolase; cleaves the N-glycosidic bond between GlcNAc and asparagine "
             "(aspartylglucosamine → GlcNAc + aspartate) during N-linked glycoprotein catabolism. "
             "Autoactivated by autocatalytic processing: proenzyme → active α2β2 heterodimer."},
            {"term": "Aspartylglucosaminuria (AGU)", "definition":
             "Oligosaccharidosis (NOT MPS) caused by biallelic AGA deficiency; accumulation of "
             "aspartylglucosamine (GlcNAc-Asn) in lysosomes; progressive neurodegeneration after "
             "near-normal early development; most common LSD in Finland (1:18,000 births)."},
            {"term": "Born Normal → Progressive (AGU developmental trajectory)", "definition":
             "UNIQUE in LSD spectrum: AGU infants appear nearly normal (only recurrent infections "
             "and mild coarse facies); intellectual regression starts age 5-10yr; behavioral problems "
             "precede seizures; full regression to severe ID by age 20-30. 'Born normal' period "
             "often delays diagnosis — family/clinician may not seek metabolic workup initially."},
            {"term": "Finnish Founder p.Cys163Ser (c.488G>C)", "definition":
             "Homozygous missense mutation in catalytic site of AGA; disrupts autocatalytic proenzyme "
             "activation → zero AGA activity; accounts for ~98% of Finnish AGU alleles; "
             "Finnish prevalence 1:18,000 (most common LSD in Finland); non-Finnish patients "
             "typically compound-het with different second allele; confirmed by gene sequencing."},
            {"term": "Aspartylglucosamine (GlcNAc-Asn)", "definition":
             "The accumulating substrate in AGU: 2-acetamido-1-(β-L-aspartamido)-1,2-dideoxy-β-D-glucose; "
             "elevated in urine on TLC/HPLC/mass spectrometry; "
             "NOT a GAG (not heparan sulfate, dermatan sulfate, keratan sulfate) — "
             "do NOT order urine GAG screen (will be NORMAL in AGU)."},
            {"term": "Behavioral Phenotype (AGU)", "definition":
             "Aggression, hyperactivity, self-injurious behavior (SIB), autistic features, impulse "
             "control failure DOMINANT in AGU; often leads to misdiagnosis (ADHD, autism, conduct "
             "disorder) before metabolic diagnosis; behavioral management is primary quality-of-life "
             "issue; atypical antipsychotics (quetiapine preferred) for management; behavioral "
             "episodes may trigger or precede seizures in some patients."},
            {"term": "Psychiatric Misdiagnosis Delay", "definition":
             "Average 5-10yr delay from symptom onset to AGU diagnosis; initial diagnoses include "
             "ADHD, autism spectrum disorder, conduct disorder, intellectual disability of unknown cause; "
             "delayed because 'born normal' and behavioral features precede clear metabolic phenotype; "
             "urine oligosaccharide screen + AGA enzyme assay resolves diagnosis."},
            {"term": "Macroorchidism (Males, AGU)", "definition":
             "Testicular enlargement (>25mL volume) in 50-70% of adult male AGU patients; "
             "due to glycoprotein accumulation in Sertoli cells and interstitium; "
             "diagnostic clue in adult males with ID + macroorchidism (also seen in Fragile X "
             "but FMR1 molecular testing distinguishes); AGA enzyme assay needed."},
            {"term": "EEG-Guided CBZ/OXC Decision (AGA-specific)", "definition":
             "CBZ/OXC are ACCEPTABLE in AGA IF EEG confirms NO myoclonic component; "
             "AGA seizures are often pure focal or GTCS (no myoclonus) — in this case "
             "CBZ/OXC are first-line appropriate for focal seizures; "
             "BUT if EEG shows polyspike-wave or myoclonus → CBZ/OXC RELATIVE CI; "
             "This EEG-guided approach is AGA-specific (unlike NEU1/FUCA1 where CBZ CI is broader)."},
            {"term": "Typical AP HIGH RISK (AGA behavioral trap)", "definition":
             "Typical antipsychotics (haloperidol, chlorpromazine) prescribed for behavioral "
             "aggression in AGU → severe EPS (basal ganglia oligosaccharide accumulation "
             "→ D2 hypersensitivity → acute dystonia, Parkinsonism, tardive dyskinesia); "
             "ALSO lower seizure threshold; use ATYPICAL AP (quetiapine, low-dose risperidone) instead."},
            {"term": "POLG1 Mandatory Exclusion", "definition":
             "POLG1/POLG2 sequencing MANDATORY before valproate (CPIC Grade A); "
             "POLG1 carriers → VPA → Alpers-Huttenlocher (acute hepatic failure); "
             "test first → if POLG1+ → avoid VPA → LEV ± CLZ ± CBZ/OXC (EEG-guided)."},
            {"term": "AGA Enzyme Assay (Leukocytes/Fibroblasts)", "definition":
             "Gold-standard biochemical diagnosis: AGA activity in leukocytes or skin fibroblasts "
             "markedly reduced (<5% control in null-null; 5-25% in compound-het); "
             "confirms AGU after positive urine screen; precedes gene sequencing; "
             "DBS (dried blood spot) enzyme assay available in some centers."},
            {"term": "Urine Aspartylglucosamine Screen (NOT GAG)", "definition":
             "Urine TLC/HPLC/mass spectrometry shows elevated aspartylglucosamine spot; "
             "CRITICAL: this is NOT the GAG screen used for MPS (heparan/dermatan/keratan sulfate); "
             "ORDER: urine oligosaccharide screen (not urine GAG fractionation); "
             "abnormal oligosaccharide spot → enzyme assay → gene sequencing."},
            {"term": "BMT/HSCT (Finnish AGU experience)", "definition":
             "Finnish AGU cohort has largest BMT experience in any oligosaccharidosis; "
             "performed in 1990s-2000s; somatic benefits documented (reduced infections, "
             "stabilization of peripheral disease); CNS cognitive benefit LIMITED; "
             "neurodegeneration not reliably halted; not standard therapy; Level C evidence; "
             "considered only under age 6 with suitable HLA-matched donor."},
            {"term": "AAV-AGA Gene Therapy (UX143, Ultragenyx)", "definition":
             "Phase I intrathecal AAV-AGA gene therapy (Ultragenyx UX143) — early trial data "
             "2025; aims to restore CNS AGA activity via CSF delivery; "
             "the most promising disease-modifying approach (2026); "
             "not yet approved; investigational; enroll clinical trial if available."},
            {"term": "VGB Relative CI (Behavioral, not Visual)", "definition":
             "Vigabatrin RELATIVE CI in AGA for BEHAVIORAL reasons (not visual): "
             "VGB causes agitation, irritability, behavioral worsening in ID patients; "
             "AGA has prominent behavioral phenotype (aggression, SIB) → VGB adds burden; "
             "UNLIKE NEU1/FUCA1: AGA has NO cherry red spot, NO corneal clouding → "
             "VGB visual CI does NOT apply; the CI in AGA is behavioral only."},
            {"term": "Recurrent Infections (Early AGU)", "definition":
             "Recurrent otitis media, upper respiratory infections, sinusitis in AGU children "
             "(age 0-6yr) before cognitive regression begins; due to IgG subclass deficiency "
             "(similar to MAN2B1 alpha-mannosidosis); often the FIRST clinical clue; "
             "IgG replacement (IVIG/SCIG) Level C if severe immunodeficiency documented."},
            {"term": "Quetiapine (Preferred Atypical AP in AGU)", "definition":
             "First-choice atypical antipsychotic for behavioral management in AGU; "
             "low D2 affinity (highest safety margin for EPS in basal ganglia storage disease); "
             "25-300mg/day (start low, titrate); some mild anticonvulsant properties; "
             "QTc monitoring; metabolic monitoring; timing dosing to behavioral episodes."},
            {"term": "Differential Diagnosis (AGA vs Other Oligosaccharidoses)", "definition":
             "Alpha-Mannosidosis (MAN2B1): mannose-rich oligosaccharides; Velmanase-alfa ERT; "
             "no macroorchidism; MAN2B1 enzyme low (not AGA); "
             "Fucosidosis (FUCA1): angiokeratoma type 2; Italian founder; no macroorchidism; "
             "Sialidosis (NEU1): cherry red spot; action myoclonus; CLZ first-line; "
             "Galactosialidosis (CTSA): both NEU1 and GLB1 low; "
             "Sanfilippo A-D (MPS-III): heparan sulfate GAG elevated; severe behavioral; "
             "different enzyme panel."},
            {"term": "Lacosamide (AGA focal CI-free option)", "definition":
             "Slow-inactivation sodium-channel modulator; useful for focal seizures in AGA "
             "when CBZ/OXC causing hyponatremia, drug interactions, or EEG shows borderline "
             "myoclonus making CBZ questionable; 200-400mg/day; IV formulation available; "
             "no known myoclonus-worsening; QTc monitoring."},
        ],
        "diagnostic_algorithm": [
            "Step 1: Suspect AGA — behavioral regression (aggression, hyperactivity, SIB) + "
            "progressive intellectual disability in a child with near-normal early development",
            "Step 2: Look for clues — coarse facies (mild), recurrent otitis media, "
            "macroorchidism in males, vacuolated lymphocytes on peripheral smear (40-60% positive)",
            "Step 3: Urine oligosaccharide screen — aspartylglucosamine (GlcNAc-Asn) elevated "
            "(TLC/HPLC/mass spectrometry); NOT urine GAG screen (AGU is not MPS)",
            "Step 4: AGA enzyme assay — leukocytes or skin fibroblasts; markedly reduced (<5% "
            "control in null; 5-25% in compound-het); DBS assay where available",
            "Step 5: Confirmatory AGA gene sequencing — biallelic pathogenic variants; "
            "identify Finnish p.Cys163Ser vs non-Finnish alleles",
            "Step 6: EEG — MANDATORY before prescribing any AED; classify seizure type "
            "(GTCS vs focal vs myoclonic); presence of myoclonus changes CBZ/OXC and PHT decisions",
            "Step 7: POLG1/POLG2 sequencing — MANDATORY before any valproate prescription",
            "Step 8: Neuropsychological assessment — cognitive level (guides behavioral management); "
            "behavioral profile (aggression severity, SIB, hyperactivity score)",
            "Step 9: MRI brain — cerebral atrophy, white matter changes in advanced cases; "
            "may be normal early in disease; correlates with cognitive decline stage",
            "Step 10: IgG subclasses — assess for immunodeficiency (recurrent infection risk); "
            "IVIG Level C if documented severe IgG deficiency",
            "Step 11: Ophthalmology — rule out cherry red spot (AGA does NOT have it; "
            "if present → reconsider Sialidosis NEU1, Galactosialidosis CTSA, GM1 GLB1)",
            "Step 12: BMT candidacy assessment (if under age 6, severe phenotype, "
            "HLA-matched donor available) and/or gene therapy trial enrollment",
        ],
        "pharmacological_distinctions": [
            "LEV vs VPA: LEV preferred if POLG1 not yet tested; VPA Level B but POLG1 mandatory "
            "first; if POLG1+ → LEV is the backbone; behavioral SE of LEV (irritability) requires "
            "monitoring given AGA behavioral phenotype",
            "CBZ/OXC vs Lacosamide: CBZ/OXC acceptable IF EEG no myoclonus — first-line for "
            "focal seizures; lacosamide preferred if hyponatremia, drug interactions, or EEG "
            "borderline (avoiding sodium-channel worsening of potential myoclonus)",
            "Quetiapine vs Haloperidol: quetiapine PREFERRED atypical (behavioral management, "
            "lowest EPS); haloperidol AVOID (HIGH RISK EPS in basal ganglia storage, "
            "seizure threshold lowering) — this is the 'behavioral trap' in AGU",
            "PHT vs LEV: PHT AVOID IF myoclonus on EEG; LEV is safe alternative for all AGA "
            "seizure types including acute management; IV LEV 20mg/kg loading dose",
            "VPA vs LEV+CLZ: VPA broad-spectrum (GTCS+myoclonus+absence) but POLG1 mandatory; "
            "LEV+CLZ (for myoclonus) is POLG1-safe alternative if VPA contraindicated",
            "VGB: RELATIVE CI for behavioral reasons in AGA (agitation worsens behavioral phenotype); "
            "NOT CI for visual reasons (no cherry red spot, no corneal clouding — unlike NEU1, FUCA1)",
            "AGA vs NEU1 treatment differences: NEU1 needs Clonazepam+Piracetam (action myoclonus); "
            "AGA typically needs LEV+VPA (GTCS-dominant); CBZ acceptable in AGA (not NEU1); "
            "the behavioral antipsychotic management is AGA-specific and not needed in NEU1 Type I",
        ],
        "differential_diagnosis": [
            "Alpha-Mannosidosis (MAN2B1): oligosaccharidosis, behavioral phenotype, vacuolated "
            "lymphocytes, recurrent infections SIMILAR to AGA; BUT mannose-rich oligosaccharides "
            "in urine (different TLC spot); MAN2B1 enzyme low; Velmanase-alfa ERT available; "
            "no macroorchidism; p.His72Tyr Scandinavian founder",
            "Fucosidosis (FUCA1): oligosaccharidosis, behavioral/ID SIMILAR; BUT angiokeratoma "
            "corporis diffusum type 2 (NOT in AGA); Italian founder p.Arg178Ter; "
            "urine fucosylated oligosaccharides (different TLC); FUCA1 enzyme low",
            "MPS-IIIA-D (Sanfilippo): behavioral phenotype VERY SIMILAR to AGU (aggression, "
            "hyperactivity, SIB, progressive ID); BUT heparan sulfate GAG elevated in urine "
            "(urine GAG NORMAL in AGA); Sanfilippo enzyme panel (SGSH, NAGLU, HGSNAT, GNS)",
            "Sialidosis NEU1 Type I: progressive myoclonic epilepsy, cherry red spot; "
            "ACTION MYOCLONUS dominant (not in AGA); NEU1 enzyme low; piracetam+clonazepam "
            "first-line (not quetiapine for behavioral); no macroorchidism",
            "Fragile X (FMR1): macroorchidism + ID similar to AGA in males; BUT FMR1 trinucleotide "
            "repeat expansion (not enzyme defect); urine oligosaccharide screen NORMAL; "
            "no LSD features; FMR1 PCR/Southern blot distinguishes",
            "Galactosialidosis (CTSA): NEU1 low + GLB1 low (both); cherry red spot; "
            "no macroorchidism; AGA enzyme NORMAL in galactosialidosis; different urine pattern",
            "Non-specific ID/ADHD: initial psychiatric/developmental misdiagnosis in AGA; "
            "urine oligosaccharide screen resolves; metabolic workup in any ID + behavioral "
            "regression + coarse facies is MANDATORY (LSD panel)",
        ],
    }
