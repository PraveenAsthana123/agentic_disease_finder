#!/usr/bin/env python3
"""MANBA / Beta-mannosidosis Epilepsy Dashboard — seed data module.

Beta-mannosidosis: lysosomal beta-D-mannosidase (MANBA gene, 4q24) deficiency.
Oligosaccharidosis group (NOT MPS): Man-β-GlcNAc disaccharide accumulates in lysosomes
→ variable neurodegeneration; rarest oligosaccharidosis (<100 cases reported worldwide).

KEY DISTINGUISHING FEATURES:
  1. RAREST OLIGOSACCHARIDOSIS — fewer than 100 confirmed cases globally; extreme allelic
     heterogeneity; NO predominant founder mutation (unlike AGA Finnish p.Cys163Ser or
     MAN2B1 Scandinavian p.His72Tyr); mutations scattered worldwide without ethnic clustering.
  2. BEHAVIORAL PHENOTYPE DOMINANT — intellectual disability + aggression + hyperactivity +
     self-injurious behavior; INDISTINGUISHABLE from AGA and Sanfilippo early in course;
     behavioral features often lead to psychiatric misdiagnosis (ADHD, autism, conduct disorder)
     before metabolic workup; delays of 5-15yr are common given extreme rarity.
  3. HEARING LOSS (SENSORINEURAL) — 30-60% of reported cases have sensorineural hearing
     loss; this is MORE PROMINENT in MANBA than in AGA or MAN2B1; audiological evaluation
     mandatory at diagnosis and annually; hearing aids or cochlear implants as needed.
  4. ANGIOKERATOMA (MILD/FOCAL) — present in some cases; less prominent than Fucosidosis
     FUCA1 (type 2 angiokeratoma corporis diffusum) or Fabry GLA; focal skin lesions;
     important clinical differentiator from AGA (no angiokeratoma).
  5. EPILEPSY 30-50% — LOWER than AGA (60-70%) and FUCA1 (70-80%); tonic-clonic most
     common; myoclonic component in subset; onset variable (age 5-30yr); DRE 15-25%.
  6. URINE DIAGNOSIS: Man-β-GlcNAc disaccharide (dimannose) elevated on TLC/HPLC/mass
     spec; NOT GAG screen (not MPS); MANBA enzyme assay confirms; gene sequencing finalizes.
  7. NO ERT APPROVED (2026) — no HSCT evidence in MANBA; management is purely symptomatic;
     extreme rarity makes clinical trial enrollment the primary disease-modifying option.

Enzyme biology:
  MANBA (lysosomal beta-D-mannosidase, also called mannosidase beta) cleaves the terminal
  beta-mannose residue from Man-β-GlcNAc (the Man-beta-1,4-GlcNAc disaccharide) during
  catabolism of N-linked glycoprotein oligosaccharide cores. This is the FINAL exoglycosidase
  step in N-linked glycoprotein breakdown (acts after alpha-mannosidases have removed
  alpha-linked mannose residues). When MANBA is deficient, Man-β-GlcNAc accumulates
  intralysosomally. MANBA is distinct from alpha-mannosidase (MAN2B1): alpha-man removes
  alpha-linked mannose from the outer branches first; MANBA removes the penultimate core
  beta-mannose; both are required for complete glycoprotein catabolism.
  Genetics: MANBA gene at 4q24; >40 pathogenic variants reported; most are private (family-
  specific) missense, nonsense, or splice mutations; no single founder allele accounts for
  >5% of cases; MANBA protein is a 120kDa homodimer processed in the lysosomal compartment.

Pharmacology:
  Epilepsy management is symptomatic (no disease-modifying ERT available).
  POLG1 exclusion mandatory before VPA (CPIC Grade A — same rationale as AGA/MAN2B1).
  EEG-guided CBZ/OXC decisions: acceptable if NO myoclonic component (similar to AGA).
  PHT: not absolute CI in MANBA (unlike FUCA1), but AVOID if myoclonus present on EEG.
  VGB: RELATIVE CI for BEHAVIORAL reasons (agitation, behavioral worsening) — NOT visual;
    MANBA has no cherry red spot, no corneal clouding → visual CI does NOT apply.
  Hearing loss complicates AED choice: avoid ototoxic combinations; audiological monitoring.
  Typical antipsychotics: HIGH RISK — basal ganglia oligosaccharide accumulation → EPS;
    behavioral phenotype trap: clinicians prescribe typical AP for aggression → worsens EPS;
    use atypicals (quetiapine preferred; lowest EPS risk among atypicals).
"""
import random

GENE = "MANBA"
LOCUS = "4q24"
OMIM_GENE = "609489"
OMIM_DISEASE = "248510"
INHERITANCE = "Autosomal Recessive (AR) — biallelic MANBA LOF; both males AND females equally affected"
COHORT_SIZE = 40
DISEASE_MECHANISM = (
    "Lysosomal beta-D-mannosidase (MANBA) deficiency → accumulation of Man-β-GlcNAc "
    "(the Man-beta-1,4-GlcNAc disaccharide) in lysosomes throughout the body. MANBA "
    "normally cleaves the terminal beta-mannose residue from Man-β-GlcNAc — the FINAL "
    "exoglycosidase step in N-linked oligosaccharide catabolism after alpha-mannosidases "
    "have removed outer alpha-mannose branches. Without MANBA, Man-β-GlcNAc accumulates "
    "progressively in neurons, hepatocytes, macrophages, and connective tissue. "
    "RARITY: Beta-mannosidosis is the rarest oligosaccharidosis (<100 confirmed cases "
    "globally as of 2026); extreme allelic heterogeneity (>40 private mutations); "
    "no predominant founder mutation distinguishes MANBA from AGA (Finnish), MAN2B1 "
    "(Scandinavian p.His72Tyr), and FUCA1 (Italian p.Arg178Ter). "
    "BEHAVIORAL TRAJECTORY: Like AGA, behavioral phenotype (intellectual disability, "
    "aggression, hyperactivity, self-injury) dominates the clinical picture; onset "
    "typically age 1-10yr; behavioral problems precede or coincide with seizures; "
    "psychiatric misdiagnosis (ADHD, autism, conduct disorder) delays correct diagnosis "
    "for years or decades given extreme rarity. "
    "HEARING LOSS: Sensorineural hearing loss in 30-60% — more prominent than in AGA "
    "or MAN2B1; due to Man-β-GlcNAc accumulation in cochlear hair cells and spiral "
    "ganglion neurons; mandatory audiological assessment; hearing aids reduce isolation "
    "and behavioral burden. "
    "SEIZURE MECHANISM: Progressive lysosomal Man-β-GlcNAc storage in cortical neurons "
    "→ neuronal dysfunction → cortical hyperexcitability → tonic-clonic and myoclonic "
    "seizures; LOWER seizure prevalence (30-50%) than AGA (60-70%) or FUCA1 (70-80%) "
    "reflecting less severe cortical storage burden; DRE 15-25% (lower than AGA 25-40%). "
    "EEG PATTERN: Variable; generalized slow spike-wave; multifocal discharges; background "
    "slowing paralleling cognitive decline; myoclonic component in subset (changes PHT/CBZ "
    "decisions); EEG mandatory before any AED prescription. "
    "NO ERT (2026): No approved enzyme replacement therapy; no HSCT evidence in MANBA "
    "(unlike AGA Finnish BMT experience or MAN2B1 HSCT). Extreme rarity limits trial "
    "recruitment; patients should be referred to natural history registries."
)

ETIOLOGIES = [
    {
        "name": "Biallelic Null/Frameshift — Severe Early Onset",
        "pct": 30,
        "n": 12,
        "seizure_risk": "45-55% (early onset; moderate DRE; severe ID; behavioral phenotype dominant)",
        "eeg": "Multifocal discharges; generalized slow spike-wave; moderate background slowing; "
               "polyspike-wave in 25-40% (myoclonic component guides PHT/CBZ decision); "
               "progressive EEG deterioration paralleling cognitive decline; "
               "absent photoparoxysmal response (unlike AGA where 20-30% photosensitive); "
               "sleep EEG: increased interictal discharges; EEG mandatory before all AED starts",
        "variant_detail": (
            "Biallelic null/frameshift (nonsense + frameshift): MANBA enzyme absent (<3% control); "
            "severe early-onset phenotype; regression starts age 1-5yr; severe-profound intellectual "
            "disability by age 15; behavioral phenotype dominant (aggression, hyperactivity, SIB); "
            "hearing loss 50-65% (most common in null-null); angiokeratoma in some (focal); "
            "seizures earlier and harder to control than missense cases; DRE 20-30%; "
            "coarse facies moderate; vacuolated lymphocytes variable (less consistent than AGA); "
            "urine Man-β-GlcNAc markedly elevated; MANBA enzyme assay confirms"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "Compound-Het Missense + Null (Intermediate Phenotype)",
        "pct": 35,
        "n": 14,
        "seizure_risk": "30-45% (variable onset; intermediate DRE; moderate-severe ID)",
        "eeg": "Generalized slow spike-wave with moderate background slowing; "
               "focal temporal or frontal discharges in some; EEG-guided classification "
               "essential (myoclonic component changes CBZ/OXC CI status, same rule as AGA); "
               "absence seizures in juvenile onset subset (3Hz S-W); "
               "background alpha preserved longer than null-null cases; "
               "progressive slow-wave increase with disease progression",
        "variant_detail": (
            "Compound-het: missense (partial residual enzyme 5-20%) + null/frameshift; "
            "intermediate phenotype; onset age 2-10yr; moderate-severe intellectual disability; "
            "behavioral phenotype prominent (aggression, hyperactivity, self-injury); "
            "hearing loss 35-50% (sensorineural); angiokeratoma less prominent than null-null; "
            "seizures variable (focal or generalized); DRE 15-25%; "
            "urine Man-β-GlcNAc elevated (may be less marked than null-null on routine screen); "
            "MANBA enzyme 5-20% control; full MANBA sequencing required for variant identification"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "Biallelic Missense — Attenuated (Residual Activity)",
        "pct": 20,
        "n": 8,
        "seizure_risk": "20-35% (late onset; mild DRE; mild-moderate ID; hearing loss key clue)",
        "eeg": "Near-normal or mildly slow background; rare focal discharges; "
               "seizures may be first or only neurological presentation in attenuated cases; "
               "EEG may be essentially normal between seizures in mild cases; "
               "absence of myoclonic component in most attenuated cases (CBZ/OXC safe); "
               "MRI: normal or mild white matter changes; enzyme 20-35% residual activity",
        "variant_detail": (
            "Biallelic hypomorphic missense: residual MANBA enzyme 20-40% control; attenuated "
            "phenotype; mild-moderate intellectual disability; behavioral features present but "
            "milder (less aggression, less SIB); hearing loss 25-40% (sensorineural); "
            "angiokeratoma minimal or absent; seizures mild (occasional GTCS); late onset "
            "(age 10-35yr); misdiagnosed as non-specific intellectual disability without metabolic "
            "workup; urine Man-β-GlcNAc borderline elevated (may require quantitative HPLC); "
            "enzyme assay essential; MANBA gene sequencing confirms biallelic hypomorphic alleles"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "Splice-Site Variants — Variable Phenotype",
        "pct": 10,
        "n": 4,
        "seizure_risk": "35-50% (variable; depends on splice consequence; moderate-severe phenotype)",
        "eeg": "Variable: severe splice → multifocal discharges (like null-null); "
               "partial splice → generalized slow spike-wave; "
               "EEG-guided approach same as all MANBA cases; "
               "status epilepticus risk intermediate; "
               "background slowing severity correlates with cognitive decline stage",
        "variant_detail": (
            "Biallelic or compound-het splice-site variants; phenotype depends on degree of "
            "exon skipping/intron retention → residual splicing determines severity; "
            "cryptic splice site activation can produce partial residual enzyme; "
            "RNA analysis (RT-PCR) required to quantify splice consequence; "
            "phenotype ranges from severe-null to intermediate; hearing loss 40-55%; "
            "urine oligosaccharide elevated; enzyme assay confirms; gene sequencing for variant "
            "identification; RNA study predicts functional severity; counseling complex"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "Adult-Diagnosed / Ultra-Attenuated (Mild Alleles)",
        "pct": 5,
        "n": 2,
        "seizure_risk": "15-25% (rare; late onset; possibly first seizure in 3rd-4th decade)",
        "eeg": "Near-normal background; sporadic focal discharges; first seizure EEG may "
               "show only minimal abnormality; seizure may be the first referral trigger "
               "leading to metabolic workup; enzyme assay and urine screen resolve diagnosis; "
               "MRI: typically normal in ultra-attenuated adult cases at time of diagnosis",
        "variant_detail": (
            "Biallelic ultra-hypomorphic alleles: MANBA enzyme 30-50% residual; adult-diagnosed "
            "at age 20-50yr; mild intellectual disability or borderline cognition; "
            "behavioral features minimal (anxiety, mood dysregulation); "
            "hearing loss variable (25-35%); angiokeratoma absent or minimal; "
            "seizure (GTCS) may be first referral to neurology; urine Man-β-GlcNAc "
            "borderline elevated (quantitative HPLC/mass spec required); enzyme assay confirms; "
            "gene sequencing identifies biallelic hypomorphic variants; "
            "metabolic workup triggered by adult seizure + intellectual disability combination"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
]

SEIZURE_TYPES = [
    {"type": "Generalized Tonic-Clonic Seizures (GTCS)", "pct": 70,
     "note": "Most common seizure type in MANBA; variable onset (age 5-30yr); "
             "LEV Level B first-line; VPA Level B (POLG1 mandatory first); "
             "CBZ/OXC acceptable if EEG confirms no myoclonic component (EEG-guided — same rule as AGA)"},
    {"type": "Myoclonic Seizures", "pct": 25,
     "note": "Myoclonic component in 25-40% of MANBA cases with epilepsy; EEG-confirmed polyspike-wave; "
             "PHT AVOID if myoclonus present; VPA + LEV preferred; CLZ adjunct for myoclonus; "
             "CBZ/OXC RELATIVE CI if myoclonic component on EEG"},
    {"type": "Focal Onset Seizures", "pct": 25,
     "note": "Temporal or frontal focal onset; secondary generalization common; "
             "CBZ/OXC appropriate IF no myoclonic component on EEG (same EEG-guided rule as AGA); "
             "lacosamide or OXC for focal without myoclonus"},
    {"type": "Absence Seizures (3Hz S-W)", "pct": 15,
     "note": "Juvenile onset subset (age 5-15yr); EEG-confirmed 3Hz spike-wave; "
             "ethosuximide or VPA (POLG1 mandatory); avoid CBZ/OXC (worsens absence)"},
    {"type": "Status Epilepticus (SE)", "pct": 10,
     "note": "Febrile or provoked SE in severe null cases; rescue benzodiazepine protocol; "
             "IV LEV or VPA for SE management; fever (infection) is a key SE trigger"},
]

TRIGGERS = [
    {"trigger": "Febrile Illness / Infection", "pct": 60,
     "note": "Infections trigger seizures — possibly via immune activation and CNS cytokine release; "
             "fever also lowers seizure threshold directly; "
             "aggressive antipyretic management; rescue BDZ protocol for febrile SE; "
             "immunodeficiency screen at baseline (IgG subclasses — similar to MAN2B1)"},
    {"trigger": "Sleep Deprivation / Irregular Sleep", "pct": 55,
     "note": "Behavioral dysregulation disrupts sleep; irregular sleep → seizures; "
             "melatonin Level B (2-10mg HS) for sleep-seizure nexus; "
             "caregivers often sleep-deprived; sleep hygiene education essential"},
    {"trigger": "Behavioral Crisis / Agitation", "pct": 50,
     "note": "Behavioral outbursts (aggression, self-injury) may precede or precipitate seizures; "
             "behavioral management = seizure prevention; atypical AP (quetiapine preferred) "
             "for behavioral crises; typical AP AVOID (EPS risk — behavioral phenotype trap)"},
    {"trigger": "Missed AED Dose", "pct": 40,
     "note": "Behavioral dysregulation → non-adherence; caregiver-administered AEDs essential; "
             "liquid formulations preferred; long-acting/depot formulations where available"},
    {"trigger": "Acoustic / Auditory Stimulation", "pct": 20,
     "note": "MANBA-SPECIFIC: hearing loss patients may have audiogenic seizure precipitation "
             "when unexpected sounds penetrate residual hearing; hearing aids paradoxically "
             "may reduce audiogenic triggers by normalizing auditory input; "
             "audiological management is part of seizure management in MANBA"},
    {"trigger": "Photic / Flickering Light", "pct": 15,
     "note": "Photoparoxysmal response in minority (less common than AGA 20-30%); "
             "avoid strobe environments; television/video-game seizures possible in photosensitive subset"},
]

TREATMENTS = [
    {"drug": "Levetiracetam (LEV)", "level": "Level B",
     "indication": "FIRST-LINE broad-spectrum: GTCS + focal + myoclonic; SV2A modulation; "
                   "1000-3000mg/day; IV formulation for status/NPO; "
                   "preferred over PHT for acute management; "
                   "behavioral side effects (irritability) require monitoring given pre-existing "
                   "behavioral phenotype; B6 (pyridoxine) supplement if behavioral SE develops",
     "ci": "Behavioral worsening (irritability, aggression) — monitor closely given AGA-like "
           "behavioral phenotype; supplement with B6 if behavioral SE develops"},
    {"drug": "Valproate / VPA (Depakote / Epilim)", "level": "Level B",
     "indication": "GTCS + myoclonus + absence; broad-spectrum Level B; 500-2500mg/day; "
                   "MANDATORY POLG1 exclusion BEFORE prescribing (CPIC Grade A); "
                   "effective for behavioral stabilization as secondary benefit",
     "ci": "POLG1/POLG2 carriers: ABSOLUTE CI (Alpers-Huttenlocher; fulminant hepatic failure); "
           "pregnancy (teratogenic); pancreatitis; weight gain"},
    {"drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)", "level": "Level B",
     "indication": "Focal seizures WITHOUT myoclonic component; CBZ 400-1600mg/day; "
                   "OXC 600-2400mg/day; MANDATORY EEG before prescribing — if myoclonus → "
                   "RELATIVE CI (worsens myoclonus); safe for pure focal/GTCS; "
                   "HLA-B1502 test before CBZ in SE-Asian patients",
     "ci": "RELATIVE CI if myoclonus present on EEG; HLA-B1502 SJS risk; "
           "hyponatremia (OXC); CAUTION with hearing-loss patients (avoid ototoxic combinations)"},
    {"drug": "Clonazepam (CLZ)", "level": "Level B",
     "indication": "Adjunct for myoclonic component; 0.5-2mg TID; GABAergic modulation; "
                   "useful in combination with LEV or VPA for myoclonus+GTCS burden; "
                   "behavioral anxiolysis as secondary benefit",
     "ci": "Sedation (adds to cognitive burden); tolerance; withdrawal rebound; "
           "behavioral disinhibition possible; never abrupt discontinuation"},
    {"drug": "Quetiapine (Seroquel) — atypical AP for behavior", "level": "Level B",
     "indication": "BEHAVIORAL PHENOTYPE MANAGEMENT — aggression, self-injury, hyperactivity; "
                   "quetiapine PREFERRED atypical AP: lowest EPS risk; D2 partial agonism; "
                   "25-300mg/day; behavioral management = seizure prevention; "
                   "improves quality of life for caregivers",
     "ci": "QTc monitoring (baseline ECG); metabolic monitoring; sedation; NOT typical AP"},
    {"drug": "Melatonin", "level": "Level B",
     "indication": "Sleep-seizure nexus; 2-10mg HS; reduces nocturnal seizure frequency; "
                   "safe in all ages; behavioral benefit (improved circadian rhythm)",
     "ci": "None clinically significant"},
    {"drug": "Lacosamide (Vimpat)", "level": "Level C",
     "indication": "Focal seizures refractory to CBZ/OXC; slow-inactivation sodium-channel "
                   "modulator; 200-400mg/day; useful if CBZ causing hyponatremia or EEG "
                   "shows borderline myoclonus; IV formulation available",
     "ci": "PR interval prolongation (ECG monitoring); dizziness; not first-line"},
    {"drug": "Hearing Aids / Cochlear Implant (non-AED)", "level": "Level B",
     "indication": "MANBA-SPECIFIC: sensorineural hearing loss management; hearing loss "
                   "amplifies behavioral burden and seizure triggers (audiogenic seizures); "
                   "audiological rehabilitation reduces behavioral outbursts secondary to "
                   "communication frustration; cochlear implant if profound hearing loss",
     "ci": "Cochlear implant: general surgical risk; compatibility with MRI follow-up "
           "(MRI-conditional CI planning required for MANBA patients needing brain MRI)"},
]

CONTRAINDICATIONS = [
    {"drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine, Fluphenazine)",
     "severity": "HIGH RISK",
     "reason": "BEHAVIORAL PHENOTYPE TRAP: typical APs prescribed for behavioral features "
               "(aggression, hyperactivity) in MANBA → severe EPS due to Man-β-GlcNAc "
               "accumulation in basal ganglia (globus pallidus, substantia nigra → D2 "
               "hypersensitivity); risk of tardive dyskinesia, acute dystonic reactions, "
               "Parkinsonism; ALSO lowers seizure threshold; use ATYPICAL AP instead "
               "(quetiapine preferred — same principle as AGA)"},
    {"drug": "Phenytoin (PHT) / Fosphenytoin — if myoclonus present",
     "severity": "AVOID if myoclonus on EEG",
     "reason": "PHT worsens myoclonic component if present (sodium-channel blockade "
               "paradoxically worsens cortical myoclonus); MANBA has myoclonic component "
               "in 25-40% of cases with epilepsy; MANDATORY EEG before PHT; "
               "if polyspike-wave or myoclonus → AVOID; IV LEV is the safe alternative; "
               "PHT is NOT absolute CI (unlike FUCA1) but RELATIVE CI when myoclonus documented"},
    {"drug": "Carbamazepine / Oxcarbazepine — if myoclonus present",
     "severity": "RELATIVE CI if myoclonus on EEG",
     "reason": "CBZ/OXC worsen myoclonic component; SAFE for pure focal or GTCS without "
               "myoclonus; CHECK EEG — if no myoclonic discharges → CBZ/OXC appropriate; "
               "if polyspike-wave → avoid; EEG mandatory before prescribing; "
               "same EEG-guided rule as AGA (not the absolute CI of FUCA1)"},
    {"drug": "POLG1 carriers — Valproate",
     "severity": "ABSOLUTE CI (CPIC Grade A)",
     "reason": "MANDATORY POLG1/POLG2 sequencing before VPA in ALL oligosaccharidoses; "
               "POLG1 carriers → VPA → Alpers-Huttenlocher (acute hepatic failure); "
               "CPIC Grade A; if POLG1+ → use LEV ± CLZ ± CBZ/OXC (EEG-guided) instead"},
    {"drug": "Vigabatrin (VGB) — behavioral risk in MANBA",
     "severity": "RELATIVE CI (behavioral)",
     "reason": "VGB causes agitation, behavioral worsening, irritability — especially harmful "
               "in patients with behavioral phenotype (aggression, SIB, hyperactivity); "
               "MANBA has AGA-like behavioral phenotype → VGB adds behavioral burden; "
               "NOT CI for visual reasons (no cherry red spot, no corneal clouding in MANBA); "
               "the CI is BEHAVIORAL in MANBA (same as AGA, unlike NEU1 where visual CI applies)"},
    {"drug": "Ototoxic Drug Combinations (Aminoglycosides, Loop Diuretics)",
     "severity": "CAUTION — MANBA-SPECIFIC hearing loss risk",
     "reason": "MANBA-SPECIFIC: 30-60% already have sensorineural hearing loss; "
               "ototoxic drugs (gentamicin, furosemide) can worsen pre-existing cochlear damage; "
               "AVOID aminoglycoside antibiotics when alternatives exist; "
               "use furosemide cautiously with audiological monitoring; "
               "document baseline audiogram before any ototoxic drug exposure"},
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
            "epilepsy_prevalence": "30-50% (LOWER than AGA 60-70% and FUCA1 70-80%)",
            "drug_resistance_rate": "15-25% (moderate DRE; lower than AGA 25-40%)",
            "rarity": "Rarest oligosaccharidosis — <100 cases confirmed worldwide as of 2026",
            "hearing_loss": "Sensorineural hearing loss 30-60% — MOST PROMINENT in this oligosaccharidosis group",
            "behavioral_phenotype": "Aggression/hyperactivity/SIB dominant — indistinguishable from AGA early",
            "no_founder_mutation": "No predominant founder allele; >40 private mutations worldwide",
            "angiokeratoma": "Mild/focal angiokeratoma in subset — less prominent than FUCA1 type 2",
            "psychiatric_misdiagnosis": "ADHD/autism/conduct disorder — 5-15yr delay common given extreme rarity",
            "ert_approved": "None approved (2026) — extreme rarity limits clinical trials",
            "hsct_evidence": "None (no HSCT data in MANBA; unlike AGA Finnish BMT experience)",
            "polg1_mandatory": "MANDATORY before VPA (CPIC Grade A)",
            "urine_screen": "Man-β-GlcNAc disaccharide (dimannose) — NOT GAG screen (not MPS)",
            "cbz_oxc_status": "Acceptable if EEG confirms no myoclonic component (EEG-guided, same as AGA)",
            "typical_ap_risk": "Typical antipsychotics HIGH RISK (behavioral trap — EPS in basal ganglia storage)",
            "audiogenic_trigger": "MANBA-SPECIFIC: audiogenic seizure triggers in hearing-loss patients",
        },
        "etiologies": ETIOLOGIES,
    }


def get_breakdown():
    random.seed(42)
    patients = []
    for eth in ETIOLOGIES:
        n = eth["n"]
        for i in range(n):
            is_null = "Null" in eth["name"] or "Frameshift" in eth["name"] or "Severe" in eth["name"]
            is_adult = "Adult" in eth["name"]
            age_onset = (
                random.randint(1, 10) if is_null
                else (random.randint(20, 45) if is_adult else random.randint(5, 20))
            )
            has_hearing_loss = random.random() < (0.60 if is_null else 0.35)
            pts = {
                "id": f"MANBA-{len(patients)+1:03d}",
                "etiology": eth["name"],
                "age_onset_yr": age_onset,
                "seizure_type": random.choice(
                    ["GTCS", "GTCS + Myoclonic", "Focal + GTCS", "Absence + GTCS",
                     "Focal Onset"]
                ),
                "seizure_risk": eth["seizure_risk"],
                "eeg_pattern": eth["eeg"][:100] + "…",
                "treatment_1": random.choice(["Levetiracetam", "Valproate", "Carbamazepine"]),
                "treatment_2": random.choice(["Clonazepam", "Quetiapine", "Melatonin", "Lacosamide"]),
                "trigger": random.choice([t["trigger"] for t in TRIGGERS]),
                "ci_avoided": random.choice([c["drug"][:40] for c in CONTRAINDICATIONS]),
                "cherry_red_spot": False,   # MANBA does NOT have cherry red spot
                "corneal_clouding": False,  # MANBA does NOT have corneal clouding
                "angiokeratoma": random.random() < 0.25,   # ~25% have mild angiokeratoma
                "hearing_loss_sensorineural": has_hearing_loss,
                "behavioral_phenotype": True,  # Dominant in MANBA
                "polg1_tested": True,
                "urine_dimannose": "ELEVATED (Man-β-GlcNAc disaccharide — not GAG)",
                "manba_enzyme": f"{random.randint(1, 12) if not is_null else random.randint(0, 3)}% control",
                "hsct": False,   # No HSCT data in MANBA
                "ert": False,
                "hearing_aid": has_hearing_loss and random.random() < 0.75,
                "psychiatric_dx_before_manba": random.choice(["ADHD", "Autism", "Conduct Disorder", "None", "None"]),
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
            {"term": "MANBA (Lysosomal Beta-D-Mannosidase)", "definition":
             "Lysosomal enzyme encoded by MANBA gene (4q24); cleaves the terminal beta-mannose "
             "residue from Man-β-GlcNAc (Man-beta-1,4-GlcNAc disaccharide) — the FINAL "
             "exoglycosidase step in N-linked glycoprotein catabolism; MANBA acts after "
             "alpha-mannosidases (MAN2B1) have removed outer alpha-mannose branches; "
             "deficiency causes Man-β-GlcNAc accumulation in lysosomes throughout the body."},
            {"term": "Beta-Mannosidosis (MANBA Deficiency)", "definition":
             "Oligosaccharidosis (NOT MPS) caused by biallelic MANBA LOF; accumulation of "
             "Man-β-GlcNAc (dimannose disaccharide) in lysosomes; rarest oligosaccharidosis "
             "(<100 confirmed cases worldwide as of 2026); AR, both sexes equally affected; "
             "no predominant founder mutation; variable phenotype (severe to attenuated)."},
            {"term": "Man-β-GlcNAc (Dimannose Disaccharide)", "definition":
             "The accumulating substrate in MANBA deficiency: Man-beta-1,4-GlcNAc "
             "(mannose linked beta-1,4 to N-acetylglucosamine); elevated in urine "
             "detected by TLC/HPLC/mass spectrometry; NOT a GAG — do NOT order "
             "urine GAG screen (will be NORMAL in MANBA); order urine oligosaccharide screen."},
            {"term": "Rarest Oligosaccharidosis", "definition":
             "Beta-mannosidosis is the rarest recognized oligosaccharidosis group (<100 confirmed "
             "cases globally, 2026); extreme allelic heterogeneity (>40 private pathogenic variants); "
             "no predominant founder allele distinguishes MANBA from AGA (Finnish p.Cys163Ser), "
             "MAN2B1 (Scandinavian p.His72Tyr), and FUCA1 (Italian p.Arg178Ter); "
             "diagnosis frequently missed or delayed by 5-15 years; natural history registries are essential."},
            {"term": "Sensorineural Hearing Loss (MANBA)", "definition":
             "30-60% of MANBA patients have sensorineural hearing loss — more prominent than "
             "in AGA or MAN2B1; Man-β-GlcNAc accumulation in cochlear outer hair cells and "
             "spiral ganglion neurons disrupts auditory transduction; audiological assessment "
             "mandatory at diagnosis and annually; hearing aids reduce behavioral burden "
             "(communication frustration → aggression); cochlear implant for profound loss."},
            {"term": "Behavioral Phenotype (MANBA)", "definition":
             "Intellectual disability (mild-severe) + aggression + hyperactivity + self-injurious "
             "behavior (SIB) — indistinguishable from AGA behavioral phenotype early in course; "
             "hearing loss AMPLIFIES behavioral burden (communication frustration → aggression); "
             "psychiatric misdiagnosis (ADHD, autism, conduct disorder) common given extreme rarity; "
             "atypical antipsychotics (quetiapine preferred) for management."},
            {"term": "Angiokeratoma (MANBA — Mild/Focal)", "definition":
             "Mild or focal angiokeratoma in a subset of MANBA patients (~25%); "
             "LESS PROMINENT than Fucosidosis FUCA1 (angiokeratoma corporis diffusum type 2 "
             "in 50% — confluent widespread lesions) or Fabry GLA (buttocks, genitalia); "
             "MANBA angiokeratoma: scattered isolated lesions; not pathognomonic alone; "
             "distinguishes MANBA from AGA (which has NO angiokeratoma)."},
            {"term": "No Founder Mutation (MANBA)", "definition":
             "Unlike AGA (Finnish p.Cys163Ser ~98% Finnish alleles), MAN2B1 (Scandinavian "
             "p.His72Tyr ~24%), or FUCA1 (Italian p.Arg178Ter ~25%), MANBA has NO "
             "predominant founder mutation; >40 private pathogenic variants reported; "
             "most families have unique compound-het combinations; full MANBA sequencing "
             "required (no targeted founder testing possible)."},
            {"term": "EEG-Guided CBZ/OXC Decision (MANBA — same as AGA)", "definition":
             "CBZ/OXC are ACCEPTABLE in MANBA IF EEG confirms NO myoclonic component; "
             "myoclonic component in 25-40% of MANBA epilepsy cases — EEG mandatory before "
             "prescribing any sodium-channel AED; if polyspike-wave or myoclonus → "
             "CBZ/OXC RELATIVE CI; same EEG-guided rule as AGA (not the absolute CI of FUCA1)."},
            {"term": "Typical AP HIGH RISK (MANBA behavioral trap)", "definition":
             "Typical antipsychotics (haloperidol, chlorpromazine) prescribed for behavioral "
             "aggression in MANBA → severe EPS (Man-β-GlcNAc accumulation in basal ganglia "
             "→ D2 hypersensitivity → acute dystonia, Parkinsonism, tardive dyskinesia); "
             "ALSO lowers seizure threshold; use ATYPICAL AP (quetiapine preferred — same as AGA)."},
            {"term": "POLG1 Mandatory (MANBA, CPIC Grade A)", "definition":
             "POLG1/POLG2 sequencing MANDATORY before valproate in MANBA (as in all "
             "oligosaccharidoses); POLG1 carriers → VPA → Alpers-Huttenlocher (fulminant "
             "hepatic failure); CPIC Grade A; if POLG1+ → LEV ± CLZ ± CBZ/OXC (EEG-guided)."},
            {"term": "VGB Relative CI (Behavioral, NOT Visual — MANBA)", "definition":
             "Vigabatrin RELATIVE CI in MANBA for BEHAVIORAL reasons only: VGB causes "
             "agitation, irritability, behavioral worsening in ID patients with behavioral "
             "phenotype; UNLIKE NEU1 (visual CI from cherry red spot) or FUCA1 (visual CI "
             "from monitoring impossibility) — MANBA has NO cherry red spot, NO corneal "
             "clouding → VGB visual CI does NOT apply; behavioral risk is the MANBA-specific concern."},
            {"term": "MANBA Enzyme Assay", "definition":
             "Gold-standard biochemical diagnosis: beta-mannosidase activity in leukocytes "
             "or skin fibroblasts using synthetic substrate (4-methylumbelliferyl-β-D-mannopyranoside); "
             "markedly reduced (<5% control in null; 5-30% in compound-het missense); "
             "confirms MANBA deficiency after urine screen; precedes gene sequencing; "
             "NOTE: synthetic substrate pH optimization important (MANBA active at pH 4.0-4.5)."},
            {"term": "Urine Oligosaccharide Screen (NOT GAG — MANBA)", "definition":
             "Urine TLC/HPLC/mass spectrometry shows Man-β-GlcNAc disaccharide; "
             "CRITICAL: NOT the MPS GAG screen; order URINE OLIGOSACCHARIDE SCREEN "
             "(not urine GAG fractionation — which will be NORMAL in MANBA); "
             "oligosaccharide TLC spot may be subtle in attenuated cases — quantitative HPLC/MS required."},
            {"term": "Audiogenic Seizure Trigger (MANBA-Specific)", "definition":
             "MANBA-SPECIFIC trigger: patients with sensorineural hearing loss may have "
             "audiogenic seizure precipitation when unexpected loud sounds penetrate residual "
             "hearing; hearing aids paradoxically may REDUCE audiogenic triggers by normalizing "
             "auditory input and reducing startle response; document audiogenic triggers in "
             "seizure diary; hearing rehabilitation is part of epilepsy management."},
            {"term": "No HSCT Evidence (MANBA vs AGA)", "definition":
             "Unlike AGA (Finnish BMT experience — largest HSCT experience in oligosaccharidosis) "
             "and MAN2B1 (HSCT data Level C for CNS benefit), MANBA has NO systematic HSCT "
             "data; extreme rarity prevents organized HSCT trials; management is purely "
             "symptomatic; refer to MANBA natural history registry; clinical trial if available."},
            {"term": "PHT Avoid-If-Myoclonus (MANBA — NOT Absolute CI unlike FUCA1)", "definition":
             "PHT is NOT absolute CI in MANBA (unlike FUCA1 where PHT/fosphenytoin absolute CI); "
             "however, AVOID if myoclonus documented on EEG; check EEG first; "
             "if no myoclonus → PHT acceptable for acute GTCS management; "
             "IV LEV is the safer acute alternative regardless."},
            {"term": "Alpha- vs Beta-Mannosidosis Distinction", "definition":
             "Alpha-mannosidosis (MAN2B1, 19p13.2): removes ALPHA-mannose from outer branches; "
             "Velmanase-alfa (Lamzede) ERT available; p.His72Tyr Scandinavian founder; "
             "Beta-mannosidosis (MANBA, 4q24): removes BETA-mannose from the core disaccharide; "
             "NO ERT available; no founder; rarer; hearing loss more prominent in MANBA; "
             "enzyme assays use different substrates (alpha vs beta selectivity); "
             "urine oligosaccharide profiles differ (mannose vs dimannose predominant spot)."},
            {"term": "Differential Diagnosis (MANBA vs Other Oligosaccharidoses)", "definition":
             "AGA (Aspartylglucosaminuria): behavioral phenotype VERY SIMILAR; NO angiokeratoma; "
             "Finnish founder p.Cys163Ser; urine GlcNAc-Asn (not dimannose); AGA enzyme low; "
             "MAN2B1 (alpha-mannosidosis): mannose-rich oligosaccharides; Velmanase-alfa ERT; "
             "Scandinavian founder; Vacuolated lymphocytes 98% (vs less consistent MANBA); "
             "FUCA1 (fucosidosis): angiokeratoma type 2 confluent (vs MANBA mild/focal); "
             "Italian founder; fucosylated oligosaccharide in urine; "
             "MPS-III Sanfilippo: behavioral phenotype SIMILAR; but heparan sulfate GAG elevated; "
             "NEU1 (Sialidosis): cherry red spot; action myoclonus; no angiokeratoma."},
            {"term": "Lacosamide (MANBA focal CI-free option)", "definition":
             "Slow-inactivation sodium-channel modulator; 200-400mg/day; IV formulation; "
             "useful for focal seizures when CBZ/OXC causing hyponatremia, drug interactions, "
             "or EEG borderline myoclonus; no known myoclonus-worsening; "
             "QTc monitoring; advantageous in MANBA given need to avoid ototoxic combinations."},
        ],
        "diagnostic_algorithm": [
            "Step 1: Suspect MANBA — behavioral regression (aggression, hyperactivity, SIB) + "
            "progressive intellectual disability + sensorineural hearing loss in a child; "
            "or adult with ID + hearing loss + seizures of unknown etiology",
            "Step 2: Clinical clues — sensorineural hearing loss (30-60%), mild coarse facies, "
            "focal angiokeratoma (25%), behavioral phenotype indistinguishable from AGA early; "
            "absence of cherry red spot (rules out NEU1/GM1) and corneal clouding (rules out GUSB/MPS)",
            "Step 3: Urine OLIGOSACCHARIDE screen — Man-β-GlcNAc (dimannose) elevated on "
            "TLC/HPLC/mass spec; NOT urine GAG screen (MANBA is not MPS — GAG will be NORMAL); "
            "quantitative HPLC/MS required for attenuated cases where TLC may be subtle",
            "Step 4: MANBA enzyme assay — leukocytes or skin fibroblasts; beta-mannosidase "
            "activity markedly reduced; use 4-MU-β-D-mannopyranoside substrate at pH 4.0-4.5; "
            "confirm assay distinguishes MANBA (beta) from MAN2B1 (alpha) activity",
            "Step 5: Confirmatory MANBA gene sequencing — biallelic pathogenic variants; "
            "full gene sequencing + deletion/duplication analysis (no targeted founder test)",
            "Step 6: Audiological evaluation — pure-tone audiogram; MANBA has highest "
            "sensorineural hearing loss rate among oligosaccharidoses (30-60%); "
            "baseline audiogram before any ototoxic drug exposure",
            "Step 7: EEG — MANDATORY before prescribing any AED; classify seizure type "
            "(GTCS vs focal vs myoclonic); presence of myoclonus changes CBZ/OXC and PHT decisions; "
            "same EEG-guided rule as AGA",
            "Step 8: POLG1/POLG2 sequencing — MANDATORY before any valproate prescription",
            "Step 9: Neuropsychological assessment — cognitive level; behavioral profile "
            "(aggression severity, SIB score, hyperactivity); hearing loss impact on behavior",
            "Step 10: MRI brain — cerebral atrophy, white matter changes in advanced cases; "
            "may be normal early; plan MRI-compatible cochlear implant if cochlear implant planned",
            "Step 11: Ophthalmology — rule out cherry red spot (MANBA does NOT have it; "
            "presence → reconsider NEU1, GM1 GLB1, HEXA/HEXB, Galactosialidosis CTSA)",
            "Step 12: Register in MANBA natural history database; refer to clinical trial registry "
            "for any emerging disease-modifying therapies; IgG subclasses for immunodeficiency screen",
        ],
        "pharmacological_distinctions": [
            "LEV vs VPA: LEV preferred if POLG1 not yet tested; VPA Level B but POLG1 mandatory "
            "first; if POLG1+ → LEV is backbone; behavioral SE of LEV (irritability) requires "
            "monitoring given MANBA behavioral phenotype; B6 supplement if behavioral SE develops",
            "CBZ/OXC vs Lacosamide: CBZ/OXC acceptable IF EEG no myoclonus (same EEG-guided "
            "rule as AGA — NOT the absolute CI of FUCA1); lacosamide preferred if hyponatremia, "
            "ototoxic combination concerns, or EEG borderline myoclonus",
            "Quetiapine vs Haloperidol: quetiapine PREFERRED (lowest EPS risk atypical); "
            "haloperidol AVOID (HIGH RISK EPS in basal ganglia storage — behavioral trap); "
            "same principle as AGA; hearing loss makes behavioral management more complex "
            "(communication frustration amplifies aggression)",
            "PHT vs LEV: PHT AVOID IF myoclonus on EEG (RELATIVE CI in MANBA, NOT absolute "
            "as in FUCA1); IV LEV safer acute alternative; check EEG before any PHT decision",
            "MANBA vs AGA hearing loss distinction: AGA has NO hearing loss; MANBA has "
            "30-60% sensorineural hearing loss → hearing aids/cochlear implant part of treatment; "
            "audiogenic triggers unique to MANBA among this oligosaccharidosis group",
            "MANBA vs AGA HSCT distinction: AGA Finnish BMT experience (Level C somatic benefit); "
            "MANBA has NO HSCT data; do not extrapolate AGA HSCT evidence to MANBA",
            "VGB: RELATIVE CI for behavioral reasons only — no visual CI (no cherry red spot, "
            "no corneal clouding); same behavioral CI rationale as AGA but NOT as FUCA1/NEU1",
        ],
        "differential_diagnosis": [
            "AGA (Aspartylglucosaminuria): behavioral phenotype most similar to MANBA; "
            "DISTINGUISHES: Finnish founder p.Cys163Ser (no founder in MANBA); AGA has NO "
            "hearing loss; AGA has macroorchidism males 50-70% (MANBA: absent); "
            "urine: AGA = GlcNAc-Asn spot; MANBA = dimannose spot; different enzyme assays",
            "MAN2B1 (Alpha-mannosidosis): behavioral phenotype + recurrent infections similar; "
            "DISTINGUISHES: MAN2B1 = alpha-mannose substrate; Velmanase-alfa ERT available; "
            "Scandinavian founder p.His72Tyr; vacuolated lymphocytes 98% PAS+ (vs MANBA variable); "
            "hearing loss less prominent in MAN2B1 than MANBA",
            "FUCA1 (Fucosidosis): DISTINGUISHES — angiokeratoma corporis diffusum TYPE 2 "
            "(widespread confluent, 50%) vs MANBA (mild/focal, 25%); Italian founder p.Arg178Ter; "
            "FUCA1 has PHT ABSOLUTE CI (vs MANBA RELATIVE CI); FUCA1 urine = fucosylated oligos",
            "MPS-IIIA-D (Sanfilippo): behavioral phenotype VERY SIMILAR (aggression, SIB, progressive ID); "
            "DISTINGUISHES: Sanfilippo urine GAG = heparan sulfate ELEVATED (MANBA: GAG NORMAL); "
            "Sanfilippo enzyme panel (SGSH, NAGLU, HGSNAT, GNS); no angiokeratoma in MPS-III",
            "NEU1 (Sialidosis Type I): progressive myoclonic epilepsy; cherry red spot; "
            "DISTINGUISHES: NEU1 has cherry red spot (MANBA NONE); action myoclonus dominant "
            "(MANBA: GTCS dominant); NEU1 enzyme: neuraminidase low; piracetam+CLZ first-line",
            "CTNS (Cystinosis): hearing loss + ID in nephropathic cystinosis; "
            "DISTINGUISHES: cystinosis has RENAL involvement (Fanconi syndrome); "
            "slit-lamp corneal cystine crystals PATHOGNOMONIC; cystine accumulation (not oligosaccharide); "
            "MANBA has no renal involvement, no corneal crystals; urine oligosaccharide positive (MANBA)",
        ],
    }
