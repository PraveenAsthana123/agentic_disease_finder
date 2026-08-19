#!/usr/bin/env python3
"""IDUA / MPS-I (Hurler / Hurler-Scheie / Scheie) Epilepsy Dashboard — seed data module.

MPS-I: alpha-L-Iduronidase deficiency (IDUA, 4p16.3, AR).
HS + DS both elevated (identical to MPS-II fingerprint but AR and corneal clouding present).
Three phenotypic subtypes: Hurler (null/null — severe), Hurler-Scheie (intermediate),
Scheie (missense — attenuated, normal intelligence).
HSCT = GOLD STANDARD for Hurler if done <2.5 yr (cognitive protection — contrasts MPS-II
where HSCT is controversial).
ERT: Laronidase (Aldurazyme, FDA 2003) — does NOT cross BBB (contrasts Pabinafusp-alfa for MPS-II).
Epilepsy 25-40% Hurler; dominant seizure triggers: communicating hydrocephalus (35-45%),
OSA (50-60%), cervical cord compression (atlantoaxial instability 50-60%).
Corneal clouding: PATHOGNOMONIC (earliest physical sign) — contrasts MPS-II (NO corneal clouding).
Founder mutations: p.W402X (Caucasian 37%), p.Q70X (Scandinavian 10%), p.P533R (Canavan/AJ).
"""
import random

GENE = "IDUA"
LOCUS = "4p16.3"
OMIM = "607014"
INHERITANCE = "Autosomal Recessive (AR) — biallelic IDUA LOF; both males AND females equally affected (contrasts MPS-II X-linked)"
COHORT_SIZE = 40
DISEASE_MECHANISM = (
    "alpha-L-Iduronidase deficiency → lysosomal accumulation of heparan sulfate (HS) + "
    "dermatan sulfate (DS) in ALL tissues (CNS, cornea, cardiac valves, airways, bone, liver). "
    "HS/DS burden correlates with phenotypic severity. CNS HS accumulation drives seizures, "
    "communicating hydrocephalus (GAG in arachnoid villi), and cognitive decline. "
    "DS accumulation drives cardiomyopathy, valvulopathy, and airway GAG deposition. "
    "Corneal stromal DS storage produces pathognomonic corneal clouding (earliest physical sign). "
    "Contrast MPS-II (IDS — X-linked): corneal stroma spared → NO corneal clouding in MPS-II."
)

# 5 variant classes (etiologies) — deterministic percentages
ETIOLOGIES = [
    {
        "name": "Hurler (Null/Null — homozygous or compound-het p.W402X)",
        "pct": 30,
        "n": 12,
        "seizure_risk": "High (35-40%)",
        "eeg": "Diffuse slowing + multifocal spikes; hypsarrhythmia in infants",
        "variant_detail": "p.W402X (Caucasian founder 37%) or p.Q70X (Scandinavian 10%) — null alleles; enzyme <0.3% control",
        "hsct_eligible": True,
        "ert_alone": False,
    },
    {
        "name": "Hurler (Compound-het — null + missense)",
        "pct": 25,
        "n": 10,
        "seizure_risk": "High (30-38%)",
        "eeg": "Focal temporal/occipital spikes; diffuse slowing",
        "variant_detail": "Null allele (p.W402X/p.Q70X) + missense — residual enzyme <1%; severe phenotype",
        "hsct_eligible": True,
        "ert_alone": False,
    },
    {
        "name": "Hurler-Scheie (Intermediate — compound-het missense/missense)",
        "pct": 22,
        "n": 9,
        "seizure_risk": "Moderate (20-28%)",
        "eeg": "Focal spikes; mild diffuse slowing",
        "variant_detail": "Biallelic missense — residual enzyme 1-5%; intermediate cognitive/somatic phenotype; HSCT controversial",
        "hsct_eligible": False,
        "ert_alone": True,
    },
    {
        "name": "Scheie (Attenuated — biallelic missense, >5% residual activity)",
        "pct": 15,
        "n": 6,
        "seizure_risk": "Low (10-15%)",
        "eeg": "Focal spikes (carpal tunnel / cervical cord pressure); normal background",
        "variant_detail": "p.A327P or p.P533R (AJ/Canavan attenuated) — >5% residual; normal intelligence; corneal clouding later onset",
        "hsct_eligible": False,
        "ert_alone": True,
    },
    {
        "name": "Severe — rare biallelic nonsense (non-W402X/Q70X)",
        "pct": 8,
        "n": 3,
        "seizure_risk": "High (35-42%)",
        "eeg": "Diffuse slowing + multifocal sharp waves; IS in infants",
        "variant_detail": "Private nonsense/frameshift biallelic — enzyme <0.3%; severe phenotype; often no founder mutation detected on targeted panel",
        "hsct_eligible": True,
        "ert_alone": False,
    },
]

# Seizure types
SEIZURE_TYPES = [
    {"type": "GTCS", "pct": 45, "eeg": "Bisynchronous ictal pattern; post-ictal slowing"},
    {"type": "Focal aware / impaired awareness", "pct": 35, "eeg": "Focal temporal/occipital onset; GAG-driven cortical irritability"},
    {"type": "Myoclonic", "pct": 20, "eeg": "Polyspike-wave 3-4 Hz; prominent in advanced disease"},
    {"type": "Atypical absence", "pct": 15, "eeg": "Slow spike-wave <2.5 Hz; diffuse GAG cortical involvement"},
    {"type": "Infantile spasms (IS)", "pct": 10, "eeg": "Hypsarrhythmia; modified-IS in older infants with rapid GAG accumulation"},
    {"type": "Tonic", "pct": 10, "eeg": "Diffuse fast recruitment; brainstem GAG involvement"},
]

# Seizure triggers (MPS-I specific)
TRIGGERS = [
    {"trigger": "Communicating hydrocephalus (GAG in arachnoid villi)", "pct": 42, "note": "Most common structural trigger; ICP elevation → GAG block of CSF resorption"},
    {"trigger": "Obstructive sleep apnea (airway GAG — adenotonsillar + tongue)", "pct": 58, "note": "Dominant functional trigger; OSA → hypoxia → lowered seizure threshold"},
    {"trigger": "Intercurrent febrile illness", "pct": 65, "note": "Universal trigger; fever + GAG-laden cortex = maximum vulnerability"},
    {"trigger": "Missed ERT (Laronidase) dose / ERT gap", "pct": 30, "note": "Somatic GAG rebound; CNS penetration poor but systemic improvement lost"},
    {"trigger": "Cervical cord compression (atlantoaxial instability)", "pct": 22, "note": "Hypoxic-ischemic trigger; cord GAG + atlantoaxial instability → myelopathy"},
    {"trigger": "Cardiac arrhythmia (valvulopathy + cardiomyopathy)", "pct": 18, "note": "Cerebral hypoperfusion; DS valvulopathy present in ALL MPS-I patients"},
    {"trigger": "Sleep deprivation (OSA + caregiver burden)", "pct": 55, "note": "Secondary; OSA fragments sleep → caregiver fatigue → missed AED doses"},
    {"trigger": "Emotional / physical stress", "pct": 40, "note": "General trigger; exacerbates baseline cortical hyperexcitability"},
]

# Treatments (AEDs + ERT + HSCT)
TREATMENTS = [
    {"name": "Levetiracetam (LEV)", "level": "Level B", "role": "First-line AED — no CYP induction (ERT pharmacokinetics preserved); safe in cardiomyopathy; IV formulation for SE", "ci": None},
    {"name": "Valproate (VPA)", "level": "Level B", "role": "Broad-spectrum; POLG1 exclusion MANDATORY (CPIC A); MPS-I GAG liver storage increases hepatic fragility → VPA hepatotoxicity risk amplified", "ci": "POLG1 mutation (mandatory exclusion); hepatic dysfunction"},
    {"name": "Clobazam (CLB)", "level": "Level B", "role": "Adjunctive for focal + GTCS; sedation caution in OSA (respiratory depression; CPAP-dependent patients)", "ci": "Severe OSA without CPAP"},
    {"name": "ACTH / Vigabatrin (IS regimen)", "level": "Level A (IS)", "role": "ACTH preferred for infantile spasms in MPS-I; VGB RELATIVE CI (corneal clouding makes visual field monitoring impossible)", "ci": "VGB alone if corneal clouding present"},
    {"name": "Clonazepam", "level": "Level B", "role": "Myoclonic + atypical absence; sedation caution in OSA", "ci": "Uncontrolled OSA"},
    {"name": "Zonisamide (ZNS)", "level": "Level C", "role": "Focal + GTCS adjunct; heat intolerance (GAG anhidrosis); adequate hydration mandatory", "ci": "Sulfonamide allergy"},
    {"name": "Laronidase (Aldurazyme, ERT)", "level": "Level A (somatic)", "role": "FDA 2003; weekly IV; corrects somatic HS/DS storage; does NOT cross BBB — CNS seizure burden not directly addressed; pre-treatment antibody screen mandatory", "ci": "Severe infusion reaction (switch to desensitization protocol)"},
    {"name": "HSCT (hematopoietic stem cell transplantation)", "level": "Level A (Hurler)", "role": "GOLD STANDARD for MPS-IH if done before 2.5 yr; donor enzyme corrects CNS GAG via microglial engraftment; PREVENTS cognitive decline (does not reverse established deficits); ERT bridge pre-HSCT", "ci": "MPS-IH >2.5yr (benefit diminished); attenuated Hurler-Scheie/Scheie (risk > benefit)"},
    {"name": "Melatonin", "level": "Level B", "role": "OSA + circadian disruption (GAG CNS involvement); sleep study first; CPAP first-line for OSA; melatonin adjunct for circadian resync", "ci": None},
]

# Contraindications
CONTRAINDICATIONS = [
    {"drug": "Typical antipsychotics (haloperidol, chlorpromazine)", "risk": "ABSOLUTE AVOID", "reason": "HS/DS-laden basal ganglia → extreme EPS / NMS risk; behavioral phase MPS-I generates maximum prescribing pressure from non-specialist teams", "alternative": "Atypical: risperidone, aripiprazole, quetiapine (atypicals — with EPS monitoring)"},
    {"drug": "Phenytoin / Fosphenytoin", "risk": "AVOID", "reason": "Cardiac conduction toxicity in MPS-I cardiomyopathy + valvulopathy (present in ALL patients); CYP2C9/3A4 induction disrupts ERT; IV fosphenytoin → cardiac arrhythmia risk in valvulopathy", "alternative": "IV Levetiracetam for status epilepticus"},
    {"drug": "Vigabatrin (VGB)", "risk": "RELATIVE CI", "reason": "Corneal clouding (pathognomonic MPS-I) makes visual field monitoring mandatory but impossible in severe ID; VGB retinal toxicity additive to corneal GAG storage", "alternative": "ACTH (Level A for IS); LEV for focal seizures"},
    {"drug": "Carbamazepine / Oxcarbazepine", "risk": "CAUTION (not absolute CI)", "reason": "Axonal neuropathy risk (HS/DS peripheral nerve GAG); cardiac sodium-channel toxicity in MPS-I cardiomyopathy; HLA-B*15:02 mandatory in SE Asian ancestry", "alternative": "LEV or VPA (POLG1-cleared); HLA screen first"},
    {"drug": "General anaesthesia (any agent)", "risk": "EXTREME HAZARD", "reason": "Airway GAG (adenotonsillar + tongue + tracheal) → difficult intubation; atlantoaxial instability → C-spine XR pre-GA mandatory; cardiac valvulopathy → haemodynamic instability; RSI mandatory; video laryngoscopy", "alternative": "Pre-anaesthesia alert; ENT/cardiac/neurosurgery joint planning; awake fibreoptic intubation if severe"},
    {"drug": "Serum alpha-L-iduronidase for diagnosis", "risk": "DIAGNOSTIC PITFALL", "reason": "Serum/plasma enzyme assay has pseudo-deficiency alleles (p.H82Q common) → false positives; LEUKOCYTE enzyme assay mandatory for diagnosis; concurrent ARSB measurement to exclude MSD (SUMF1)", "alternative": "Leukocyte alpha-L-iduronidase + urine HS/DS tandem-MS + IDUA gene sequencing"},
]

# Monitoring parameters
MONITORING = [
    "Alpha-L-iduronidase leukocyte enzyme activity (diagnosis + ERT response)",
    "Urine GAG quantification — HS + DS tandem-MS (treatment monitoring; both decrease with ERT)",
    "Brain MRI: periventricular white matter changes + communicating hydrocephalus (CSF resorption block) — q12M",
    "C-spine XR / MRI: atlantoaxial instability (odontoid hypoplasia + GAG ligament laxity) — pre-GA + q12M",
    "Cardiac echo: valvulopathy (mitral > aortic in ALL patients) + cardiomyopathy — q12M",
    "Slit-lamp: corneal clouding grade — every 6M; corneal transplant (PKP) evaluation if severe",
    "Polysomnography: OSA — q12M (CPAP titration); CPAP compliance monitoring",
    "Neuropsychology: cognitive trajectory (HSCT candidates pre- and post-procedure); Griffiths/Bayley",
    "POLG1 genotyping — before VPA initiation (CPIC Level A mandatory)",
    "HLA-B*15:02 — SE Asian ancestry before CBZ/OXC/LTG",
    "Anti-laronidase IgE / IgG — infusion reaction risk stratification",
    "Audiometry: conductive + sensorineural hearing loss (HS/DS middle ear + cochlear GAG)",
]

# Clinical thresholds
THRESHOLDS = [
    {"parameter": "HSCT optimal window", "threshold": "<2.5 years of age", "action": "Refer for HSCT evaluation immediately if Hurler phenotype confirmed (DQ >70 or strong family history of cognitive preservation)"},
    {"parameter": "ERT initiation", "threshold": "At diagnosis — no lower age limit", "action": "Start Laronidase (0.58 mg/kg weekly IV) as bridge to HSCT or as sole therapy (attenuated)"},
    {"parameter": "Hydrocephalus ICP", "threshold": "Opening pressure >20 cmH₂O on LP", "action": "Ventriculoperitoneal shunt or serial LP; ICP-lowering reduces seizure burden"},
    {"parameter": "Atlantoaxial instability", "threshold": "Atlanto-dens interval >5 mm or cord signal on MRI", "action": "Cervical fusion pre-HSCT; no neck manipulation; C-spine XR pre-GA mandatory"},
    {"parameter": "OSA AHI", "threshold": "AHI ≥5 events/hr (any severity)", "action": "CPAP titration; adenotonsillectomy if adenotonsillar hypertrophy (caution — airway GAG makes adenoid tissue enlarged and surgery high-risk)"},
    {"parameter": "Cardiac valvulopathy", "threshold": "Moderate–severe regurgitation or gradient >40 mmHg", "action": "Cardiac surgery consultation (valve repair/replacement); minimize PHT/fosphenytoin exposure"},
    {"parameter": "Urine GAG on ERT", "threshold": ">3x ULN despite 6M ERT", "action": "Review compliance; assess for anti-laronidase antibodies; consider dose escalation or intrathecal ERT trial"},
    {"parameter": "DRE threshold (MPS-I)", "threshold": "Failure of ≥2 appropriate AEDs", "action": "Video-EEG for presurgical evaluation; exclude hydrocephalus as reversible cause; consider ketogenic diet (KD — Level B)"},
]

# Lifecycle stages
LIFECYCLE = [
    {"stage": "Infantile (0–2 yr)", "features": "Hepatosplenomegaly, coarse facies, recurrent otitis media, corneal clouding (earliest sign by 6M), hernia. EEG: diffuse slowing. IS in subset.", "action": "Urgent enzyme assay; HSCT window assessment; ERT initiation"},
    {"stage": "Behavioral/Regression Phase (2–4 yr — Hurler)", "features": "Cognitive plateau then regression; language loss; behavioral deterioration; seizures emerge (25-40%); OSA dominates sleep phenotype. EEG: multifocal spikes + slowing.", "action": "AED + CPAP + hydrocephalus monitoring; HSCT benefit diminishes after 2.5yr"},
    {"stage": "Progressive (4–10 yr — Hurler post-HSCT)", "features": "Post-HSCT: cognitive stabilization; continued somatic HS/DS (valve/bone); seizure burden reduced. Without HSCT: continued regression, spasticity, death by 10yr.", "action": "ERT continuation post-HSCT; cardiac monitoring; AED maintenance"},
    {"stage": "Intermediate (Hurler-Scheie — school age)", "features": "Normal or borderline intelligence; significant somatic disease (cornea, valve, spine); focal seizures; peripheral nerve entrapment (carpal tunnel)", "action": "ERT (somatic); corneal PKP; valve monitoring; AED for seizures"},
    {"stage": "Attenuated Adult (Scheie)", "features": "Normal intelligence; severe somatic: corneal blindness, valve disease, joint stiffness; low seizure risk (10-15%); carpal tunnel syndrome; cervical cord compression", "action": "ERT lifelong; cardiac surgery; corneal transplant; cervical spine monitoring"},
    {"stage": "Palliative / End-Stage (Hurler without HSCT)", "features": "Spastic quadriplegia; loss of swallow; cardiac failure; apneic episodes. Seizures refractory. Death by 6–12yr without HSCT.", "action": "Palliative care + seizure comfort management; anticipatory guidance for families"},
]

# Differential diagnosis
DIFFERENTIAL_DIAGNOSIS = [
    {"condition": "MPS-II (Hunter — IDS, X-linked)", "distinction": "NO corneal clouding (MPS-II spares corneal stroma); only hemizygous males (females = rare carriers); HS+DS same fingerprint; IDS enzyme deficient"},
    {"condition": "MPS-VI (Maroteaux-Lamy — ARSB)", "distinction": "DS only elevated (not HS); normal intelligence; severe skeletal + corneal + cardiac; epilepsy rare; ARSB enzyme deficient"},
    {"condition": "MPS-VII (Sly — GUSB)", "distinction": "HS + DS + CS all elevated; extremely rare; hydrops fetalis common; GUSB enzyme deficient; epilepsy present"},
    {"condition": "Multiple Sulfatase Deficiency (MSD — SUMF1)", "distinction": "Multiple lysosomal sulfatases deficient (including IDUA + ARSB + IDS); measure co-enzyme alongside IDUA to exclude; ichthyosis present"},
    {"condition": "GM1-Gangliosidosis (GLB1)", "distinction": "Cherry-red spot (90%); no GAG in urine; galactosidase deficient; no corneal clouding initially; cherry-red vs corneal clouding is key early differentiator"},
    {"condition": "Mucolipidosis II/III (ML-II/III — GNPTA)", "distinction": "I-cell inclusions on bone marrow; multiple lysosomal enzyme elevations in SERUM (not urine); gingival hypertrophy; normal urine GAG"},
    {"condition": "Fucosidosis (FUCA1)", "distinction": "Angiokeratoma (skin); cardiomegaly; sea-blue histiocytes; fucose-containing glycans not GAG; fucosidase deficient"},
    {"condition": "POLG-related epilepsy (POLG1 mutations)", "distinction": "Mitochondrial NOT lysosomal; Alpers syndrome phenocopy; VPA ABSOLUTE CI; muscle biopsy ragged-red fibres; exclude before VPA in any severe pediatric epilepsy"},
]

# Definitions glossary
DEFINITIONS = [
    {"term": "IDUA", "definition": "Alpha-L-iduronidase — lysosomal enzyme that cleaves alpha-L-iduroniduronic acid residues from the non-reducing terminus of heparan sulfate (HS) and dermatan sulfate (DS). Deficiency causes MPS-I."},
    {"term": "MPS-I", "definition": "Mucopolysaccharidosis type I — autosomal recessive IDUA deficiency. Three subtypes by severity: Hurler (MPS-IH, null/null), Hurler-Scheie (intermediate), Scheie (MPS-IS, attenuated)."},
    {"term": "Laronidase (Aldurazyme)", "definition": "Recombinant human alpha-L-iduronidase. FDA-approved 2003. Weekly IV infusion (0.58 mg/kg). Corrects somatic HS/DS storage. DOES NOT cross the blood-brain barrier — CNS seizure burden not directly addressed by ERT alone."},
    {"term": "HSCT (Hematopoietic Stem Cell Transplantation)", "definition": "Gold standard treatment for Hurler (MPS-IH) if performed before 2.5 years of age. Donor-derived microglia produce alpha-L-iduronidase in CNS, correcting GAG accumulation and preventing cognitive decline. Does not reverse established deficits. ERT bridge pre-HSCT recommended."},
    {"term": "Communicating Hydrocephalus", "definition": "GAG accumulation in arachnoid villi blocks CSF resorption → elevated ICP → communicating (non-obstructive) hydrocephalus. Major seizure trigger in MPS-I. Treated with VP shunting. Distinct from obstructive hydrocephalus."},
    {"term": "Corneal Clouding", "definition": "Pathognomonic early sign of MPS-I (and MPS-IV, VI, VII). Dermatan sulfate accumulation in corneal stromal keratocytes → progressive opacification. ABSENT in MPS-II (Hunter) — critical differentiating feature. Visual monitoring mandatory for VGB REMS."},
    {"term": "Atlantoaxial Instability", "definition": "Odontoid hypoplasia + GAG-laden transverse ligament laxity → unstable C1-C2 articulation. Risk of cervical cord compression with neck flexion (anaesthesia hazard). Present in 50-60% of severe MPS-I. C-spine XR pre-GA mandatory."},
    {"term": "HS (Heparan Sulfate)", "definition": "Glycosaminoglycan substrate of IDUA. Accumulates in CNS → seizures, cognitive decline, communicating hydrocephalus. Urine HS measured by tandem-MS (DMMB quantification). Both HS and DS elevated in MPS-I (and MPS-II — fingerprint distinction from MPS-III HS-only and MPS-VI DS-only)."},
    {"term": "DS (Dermatan Sulfate)", "definition": "Glycosaminoglycan substrate of IDUA. Accumulates in connective tissue → corneal clouding, valvulopathy, cardiomyopathy, joint stiffness. Both HS and DS elevated in MPS-I — distinguishes from MPS-III (HS only) and MPS-VI (DS only)."},
    {"term": "Pseudo-deficiency (IDUA p.H82Q)", "definition": "Common IDUA variant causing reduced enzyme activity in plasma/serum WITHOUT causing MPS-I. Serum/plasma enzyme assay → false positive. LEUKOCYTE enzyme assay mandatory for definitive diagnosis. Affects ~1-2% of population."},
    {"term": "DQ (Developmental Quotient)", "definition": "Cognitive measure used to identify HSCT candidates. DQ >70 at HSCT referral associated with better post-HSCT cognitive outcomes. DQ-based eligibility guides transplant decision in intermediate phenotypes."},
    {"term": "p.W402X", "definition": "Most common Caucasian MPS-I founder mutation (37%). Nonsense variant — null allele — causes severe Hurler phenotype (MPS-IH) when homozygous or compound-het with another null. Essential in targeted first-tier panel."},
    {"term": "Alpers-Huttenlocher Syndrome", "definition": "POLG1-related mitochondrial hepatoencephalopathy phenotypically overlapping with severe MPS-I epilepsy. VPA causes fatal hepatotoxicity in POLG1 mutations. POLG1 genotyping MANDATORY before VPA in any severe pediatric epilepsy. Exclusion criterion for VPA prescription."},
    {"term": "DRE (Drug-Resistant Epilepsy)", "definition": "Failure of ≥2 appropriate AEDs at adequate doses. Affects 15-25% of MPS-I epilepsy patients (mostly Hurler). Hydrocephalus as a reversible structural cause must be excluded before DRE label. KD Level B evidence."},
]

KEY_CONCEPTS = [
    "Corneal clouding: PATHOGNOMONIC early sign in MPS-I — corneal DS storage in stromal keratocytes; present by 6 months in Hurler; ABSENT in MPS-II (Hunter) — single most useful bedside differentiator",
    "HSCT GOLD STANDARD for Hurler (MPS-IH): <2.5 yr age window — donor microglial enzyme corrects CNS GAG, preventing cognitive decline; CONTRASTS MPS-II where HSCT is controversial",
    "ERT (Laronidase): corrects somatic HS/DS storage but does NOT cross BBB — CNS seizure burden requires separate AED management; ERT + HSCT combination optimal for Hurler",
    "Communicating hydrocephalus: GAG blocks arachnoid villi → ICP elevation → dominant structural seizure trigger; reversible with VP shunting; must be excluded before DRE label",
    "OSA: dominant functional trigger (50-60%); airway GAG (adenotonsillar + tongue + subglottic) → hypoxia → lowered seizure threshold; CPAP = seizure management",
    "Cardiac: valvulopathy (mitral > aortic) in ALL MPS-I patients; PHT/fosphenytoin AVOID (cardiac conduction toxicity); pre-anaesthesia cardiac echo mandatory",
    "Typical antipsychotics ABSOLUTE AVOID: HS/DS basal ganglia → extreme EPS risk; behavioral phase generates prescribing pressure — coordinate with psychiatry",
    "VGB RELATIVE CI: corneal clouding makes visual field monitoring impossible (mandatory for VGB REMS); ACTH preferred for IS in MPS-I",
    "PHT/Fosphenytoin AVOID: cardiac toxicity in MPS-I cardiomyopathy; CYP induction disrupts ERT pharmacokinetics; use IV LEV for SE",
    "CBZ/OXC CAUTION: axonal neuropathy (HS/DS peripheral nerve); cardiac sodium-channel risk; HLA-B*15:02 mandatory in SE Asian ancestry",
    "POLG1 exclusion mandatory before VPA: severe MPS-I epilepsy phenocopies Alpers; VPA fatal hepatotoxicity in POLG1; leukocyte enzyme assay also differentiates",
    "Pseudo-deficiency trap: serum/plasma IDUA assay → false positives (p.H82Q common); LEUKOCYTE assay mandatory for diagnosis",
    "Atlantoaxial instability: C-spine XR mandatory pre-GA; 50-60% of severe MPS-I; cervical cord compression = hypoxic seizure trigger",
    "AR inheritance: both males AND females equally affected (contrast MPS-II X-linked — only hemizygous males; female carriers rare and mostly unaffected)",
    "Gene therapy: AAV-IDUA intrathecal/intracranial phase I/II trials underway (2024-2026); investigational — HSCT remains standard",
    "Anesthesia EXTREME HAZARD: airway GAG + atlantoaxial instability + valvulopathy — RSI + video laryngoscopy + C-spine immobilization mandatory",
    "Most common MPS with HSCT data: MPS-I Hurler has largest evidence base for HSCT cognitive benefit among all MPS types",
]

STANDARDS = [
    "Clarke et al. 2017 — International consensus guidelines for management of MPS-I (JIMD Reports)",
    "de Ru et al. 2011 — European MPS-I patient registry guidelines",
    "Wraith et al. 2004 — ERT guidelines for MPS-I (Laronidase/Aldurazyme FDA approval basis)",
    "Aldenhoven et al. 2015 — HSCT outcomes in MPS-IH (long-term cognitive follow-up)",
    "Muenzer et al. 2009 — ERT international consensus guidelines (MPS-I and II)",
    "CPIC 2023 — POLG1 genotyping Level A guideline before valproate initiation",
    "REMS VGB (Vigabatrin) — SHARE program: visual field perimetry q3M mandatory; corneal clouding in MPS-I makes compliance impossible",
    "Escolar et al. 2005 — HSCT age <2.5 yr cognitive benefit threshold (NEJM)",
    "Baldo et al. 2023 — AAV-based gene therapy for MPS-I: Phase I/II CNS outcomes",
    "ICH E6(R2) GCP — Lysosomal Storage Disease clinical trial standards",
]


def get_overview():
    """Return IDUA/MPS-I overview for /api/idua/overview."""
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "disease_mechanism": DISEASE_MECHANISM,
        "cohort_size": COHORT_SIZE,
        "epilepsy_prevalence_pct": {"hurler": "25-40", "hurler_scheie": "20-28", "scheie": "10-15"},
        "drug_resistance_pct": {"hurler": "15-25", "overall": "12-18"},
        "osa_pct": 58,
        "hydrocephalus_pct": 38,
        "atlantoaxial_instability_pct": 52,
        "corneal_clouding_pct": {"hurler": 100, "hurler_scheie": 90, "scheie": 75},
        "cardiac_valvulopathy_pct": 100,
        "hsct_eligible_pct": 55,
        "on_ert_pct": 88,
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "key_concepts": KEY_CONCEPTS[:6],
        "standards": STANDARDS,
    }


def get_breakdown():
    """Return IDUA/MPS-I 40-patient cohort for /api/idua/breakdown."""
    rng = random.Random(42)

    phenotype_weights = {
        "Hurler (MPS-IH)": 0.55,
        "Hurler-Scheie": 0.22,
        "Scheie (MPS-IS)": 0.15,
        "Severe non-founder": 0.08,
    }
    phenotype_seizure_rates = {
        "Hurler (MPS-IH)": 0.37,
        "Hurler-Scheie": 0.24,
        "Scheie (MPS-IS)": 0.12,
        "Severe non-founder": 0.40,
    }
    phenotype_dre_rates = {
        "Hurler (MPS-IH)": 0.22,
        "Hurler-Scheie": 0.10,
        "Scheie (MPS-IS)": 0.05,
        "Severe non-founder": 0.25,
    }

    etiology_by_phenotype = {
        "Hurler (MPS-IH)": "Hurler (Null/Null — homozygous or compound-het p.W402X)",
        "Hurler-Scheie": "Hurler-Scheie (Intermediate — compound-het missense/missense)",
        "Scheie (MPS-IS)": "Scheie (Attenuated — biallelic missense, >5% residual activity)",
        "Severe non-founder": "Severe — rare biallelic nonsense (non-W402X/Q70X)",
    }

    aed_first_line = ["Levetiracetam (LEV)", "Valproate (VPA — POLG1-cleared)", "Clobazam (CLB)"]
    seizure_type_pool = [st["type"] for st in SEIZURE_TYPES]
    trigger_pool = [t["trigger"][:50] for t in TRIGGERS]
    lifecycle_stages = [l["stage"].split(" (")[0] for l in LIFECYCLE]

    phenotype_pool = rng.choices(
        list(phenotype_weights.keys()),
        weights=list(phenotype_weights.values()),
        k=COHORT_SIZE,
    )

    patients = []
    drug_resistant_n = 0
    osa_n = 0
    on_ert_n = 0
    on_hsct_n = 0
    hydrocephalus_n = 0

    for i in range(COHORT_SIZE):
        pid = i + 1
        pheno = phenotype_pool[i]
        etiol = etiology_by_phenotype[pheno]
        has_seizures = rng.random() < phenotype_seizure_rates[pheno]
        has_dre = has_seizures and rng.random() < phenotype_dre_rates[pheno]
        osa = rng.random() < 0.58
        hydrocephalus = (pheno in ["Hurler (MPS-IH)", "Severe non-founder"]) and rng.random() < 0.38
        atlantoaxial = rng.random() < 0.52
        on_ert = rng.random() < 0.88
        on_hsct = (pheno == "Hurler (MPS-IH)") and rng.random() < 0.65

        if has_dre:
            drug_resistant_n += 1
        if osa:
            osa_n += 1
        if on_ert:
            on_ert_n += 1
        if on_hsct:
            on_hsct_n += 1
        if hydrocephalus:
            hydrocephalus_n += 1

        n_seizure_types = rng.randint(1, 3) if has_seizures else 0
        sz_types = rng.sample(seizure_type_pool, n_seizure_types) if has_seizures else []
        primary_aed = rng.choice(aed_first_line) if has_seizures else None
        if has_seizures:
            drug_response = "Drug-resistant" if has_dre else rng.choice(["Controlled", "Controlled", "Partially controlled"])
        else:
            drug_response = None
        top_trigger = rng.choice(trigger_pool) if has_seizures else None
        lc_idx = min(pid % len(lifecycle_stages), len(lifecycle_stages) - 1)
        lc_stage = lifecycle_stages[lc_idx]

        age_onset = rng.randint(1, 8) if has_seizures else None

        patients.append({
            "patient_id": f"IDUA-{pid:02d}",
            "phenotype": pheno,
            "etiology": etiol,
            "has_seizures": has_seizures,
            "age_onset_seizures_yrs": age_onset,
            "seizure_types": sz_types,
            "primary_aed": primary_aed,
            "drug_response": drug_response,
            "drug_resistant": has_dre,
            "osa": osa,
            "on_cpap": osa,
            "on_ert": on_ert,
            "post_hsct": on_hsct,
            "hydrocephalus": hydrocephalus,
            "atlantoaxial_instability": atlantoaxial,
            "corneal_clouding": pheno != "Scheie (MPS-IS)" or rng.random() < 0.75,
            "cardiac_valvulopathy": True,
            "lifecycle_stage": lc_stage,
            "top_trigger": top_trigger,
        })

    seizure_n = sum(1 for p in patients if p["has_seizures"])
    return {
        "gene": GENE,
        "cohort_size": COHORT_SIZE,
        "seizure_n": seizure_n,
        "seizure_pct": round(seizure_n / COHORT_SIZE * 100, 1),
        "drug_resistant_n": drug_resistant_n,
        "drug_resistant_pct": round(drug_resistant_n / COHORT_SIZE * 100, 1),
        "osa_n": osa_n,
        "osa_pct": round(osa_n / COHORT_SIZE * 100, 1),
        "on_ert_n": on_ert_n,
        "on_ert_pct": round(on_ert_n / COHORT_SIZE * 100, 1),
        "on_hsct_n": on_hsct_n,
        "on_hsct_pct": round(on_hsct_n / COHORT_SIZE * 100, 1),
        "hydrocephalus_n": hydrocephalus_n,
        "hydrocephalus_pct": round(hydrocephalus_n / COHORT_SIZE * 100, 1),
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "thresholds": THRESHOLDS,
        "lifecycle": LIFECYCLE,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "patients": patients,
    }


def get_definitions():
    """Return IDUA/MPS-I definitions for /api/idua/definitions."""
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "definitions": DEFINITIONS,
        "key_concepts": KEY_CONCEPTS,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "standards": STANDARDS,
        "diagnostic_algorithm": [
            "Step 1: Urine GAG screen — DMMB quantification + HS + DS tandem-MS; both HS AND DS elevated (MPS-I fingerprint; identical to MPS-II — differentiate by corneal slit-lamp + enzyme assay)",
            "Step 2: Leukocyte alpha-L-iduronidase enzyme activity (<5% control confirms MPS-I); co-measure ARSB (or other sulfatase) to exclude Multiple Sulfatase Deficiency (SUMF1)",
            "Step 3: Slit-lamp examination — corneal clouding PRESENT in MPS-I; ABSENT in MPS-II (IDS, X-linked Hunter); corneal clouding distinguishes MPS-I from MPS-II with same GAG fingerprint",
            "Step 4: IDUA gene sequencing (biallelic — AR) — targeted panel (p.W402X, p.Q70X first); if heterozygous or negative → full gene sequencing + deletion/duplication analysis (MLPA)",
            "Step 5: Phenotype subtyping — neurocognitive assessment (DQ/IQ), corneal grade, somatic score → Hurler vs Hurler-Scheie vs Scheie; determines HSCT eligibility",
            "Step 6: HSCT eligibility window assessment — age ≤2.5 yr? DQ >70? HLA matching? Refer transplant centre immediately if Hurler phenotype",
            "Step 7: POLG1 genotyping — CPIC Level A mandatory before VPA; Alpers phenocopy exclusion; VPA fatal in POLG1",
            "Step 8: HLA-B*15:02 — mandatory before CBZ/OXC/LTG in SE Asian ancestry",
            "Step 9: Multidisciplinary workup: cardiac echo + C-spine XR + polysomnography + brain MRI (hydrocephalus) + neuropsychology + audiometry + ophthalmology",
            "Step 10: ERT initiation (Laronidase 0.58 mg/kg weekly IV) at diagnosis; HSCT bridge ERT if <2.5 yr; anti-laronidase antibody screen pre-treatment; infusion reaction protocol ready",
        ],
        "pharmacological_distinctions": [
            "LEV FIRST-LINE: no CYP induction (ERT pharmacokinetics preserved); safe in cardiomyopathy (no cardiac conduction effect); IV formulation for SE; renal excretion — adjust for renal involvement",
            "VPA SECOND-LINE: POLG1 exclusion MANDATORY (CPIC A); MPS-I GAG liver storage amplifies hepatic fragility → VPA hepatotoxicity risk elevated; monitor LFTs closely",
            "PHT/Fosphenytoin AVOID: cardiac conduction toxicity (valvulopathy in ALL MPS-I patients); CYP2C9/3A4 induction disrupts ERT; fosphenytoin IV → arrhythmia in cardiomyopathy; use IV LEV instead",
            "Typical antipsychotics ABSOLUTE AVOID: HS/DS basal ganglia → extreme EPS/NMS; behavioral phase Hurler generates maximum prescribing pressure from non-neurology teams; use atypicals",
            "VGB RELATIVE CI: corneal clouding (pathognomonic MPS-I) makes Goldman visual field perimetry impossible — SHARE REMS compliance unachievable; ACTH preferred for IS",
            "CBZ/OXC CAUTION (not absolute CI): axonal neuropathy (HS/DS peripheral nerve GAG); cardiac sodium-channel risk in cardiomyopathy; HLA-B*15:02 mandatory in SE Asian; contrast MLD/GALC (demyelinating — CBZ absolute CI)",
            "Anesthesia EXTREME HAZARD: airway GAG (adenotonsillar + tongue + subglottic) + atlantoaxial instability + valvulopathy; RSI mandatory; video laryngoscopy; C-spine immobilization; cardiac monitoring",
            "CPAP treats OSA = seizure management: OSA is the dominant functional trigger in MPS-I; CPAP compliance directly reduces seizure frequency; prescribe CPAP before AED escalation for OSA-triggered seizures",
            "ERT (Laronidase) CNS limitation: does NOT cross BBB; somatic HS/DS improves but CNS seizures persist; ERT + AED combination mandatory; HSCT provides CNS enzyme via microglial engraftment",
            "Melatonin Level B: OSA + GAG CNS circadian disruption; sleep study + CPAP first; melatonin for circadian resync after OSA management",
            "HSCT-ERT interaction: ERT recommended as bridge pre-HSCT (reduces somatic burden, optimizes HSCT tolerance); continue ERT post-HSCT for somatic disease (CNS now covered by donor microglia)",
            "Anti-laronidase antibodies: IgG/IgE screen pre-infusion; infusion reactions in 5-10% of patients; desensitization protocol available; check compliance if GAG fails to fall on ERT",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== IDUA OVERVIEW ===")
    ov = get_overview()
    print(f"Gene: {ov['gene']} | Cohort: {ov['cohort_size']} | Epilepsy Hurler: {ov['epilepsy_prevalence_pct']['hurler']}%")
    print(f"OSA: {ov['osa_pct']}% | Hydrocephalus: {ov['hydrocephalus_pct']}% | AAI: {ov['atlantoaxial_instability_pct']}%")
    print("\n=== IDUA BREAKDOWN ===")
    bk = get_breakdown()
    print(f"Seizures: {bk['seizure_n']}/{COHORT_SIZE} ({bk['seizure_pct']}%)")
    print(f"DRE: {bk['drug_resistant_n']} ({bk['drug_resistant_pct']}%)")
    print(f"OSA: {bk['osa_n']} ({bk['osa_pct']}%) | ERT: {bk['on_ert_n']} ({bk['on_ert_pct']}%)")
    print(f"HSCT: {bk['on_hsct_n']} ({bk['on_hsct_pct']}%) | Hydrocephalus: {bk['hydrocephalus_n']} ({bk['hydrocephalus_pct']}%)")
    print("\n=== IDUA DEFINITIONS ===")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])} terms | Key concepts: {len(df['key_concepts'])}")
    print(f"Diagnostic algorithm: {len(df['diagnostic_algorithm'])} steps")
