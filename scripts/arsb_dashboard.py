#!/usr/bin/env python3
"""ARSB / MPS-VI (Maroteaux-Lamy Syndrome) Epilepsy Dashboard — seed data module.

MPS-VI: Arylsulfatase B (N-Acetylgalactosamine-4-Sulfatase) deficiency (ARSB, 5q14.1, AR).
Dermatan sulfate (DS) + chondroitin-4-sulfate (C4S) elevated — NOT heparan sulfate, NOT keratan sulfate.
DS fingerprint distinguishes MPS-VI from MPS-I/II (HS+DS), MPS-III (HS only), MPS-IVA (KS+C6S).
KEY DISTINGUISHING FEATURE: NORMAL INTELLIGENCE — DS does NOT drive cortical neuronal damage at
disease-causing levels (contrasts MPS-I/II/III which have severe cognitive decline).
DOMINANT SOMATIC DISEASE: short stature, hepatosplenomegaly, coarse facies, corneal clouding UNIVERSAL
(MOST SEVERE among non-CNS MPS), OSA (tongue + tonsil + larynx DS accumulation → SEVERE airway
obstruction), cardiac valvulopathy UNIVERSAL (aortic + mitral regurgitation ALL patients).
ERT: Galsulfase (Naglazyme, FDA 2005) — 1 mg/kg IV WEEKLY (contrast elosulfase every other week).
HSCT: YES for severe MPS VI (< 6-8 years) — primarily somatic benefit (cognition already normal);
contrast MPS-IVA (no HSCT — normal IQ, skeletal-only) and MPS-I (HSCT for cognitive preservation).
Epilepsy 15-25% (higher than MPS-IVA — communicating hydrocephalus + OSA + cord compression).
Corneal clouding UNIVERSAL and MOST SEVERE among non-CNS MPS — Goldman visual field IMPOSSIBLE →
VGB ABSOLUTE/RELATIVE CI stronger than MPS-IVA.
Cardiac valvulopathy UNIVERSAL → PHT/fosphenytoin ABSOLUTE AVOID.
Communicating hydrocephalus 20-30% (DS arachnoid deposition → CSF obstruction → seizure risk).
Founder mutation: p.R152W (Portuguese-Brazilian ~15%).
"""
import random

GENE = "ARSB"
LOCUS = "5q14.1"
OMIM = "253200"
INHERITANCE = "Autosomal Recessive (AR) — biallelic ARSB LOF; both males AND females equally affected"
COHORT_SIZE = 40
DISEASE_MECHANISM = (
    "Arylsulfatase B (N-acetylgalactosamine-4-sulfatase) deficiency → lysosomal accumulation of dermatan "
    "sulfate (DS) + chondroitin-4-sulfate (C4S) in ALL somatic tissues (cartilage, bone, cornea, cardiac "
    "valves, airway soft tissue, liver/spleen, arachnoid villi). DS/C4S accumulation causes PRIMARY "
    "SOMATIC DISEASE: short stature, coarse facies, hepatosplenomegaly, corneal clouding (UNIVERSAL and "
    "MOST SEVERE among non-CNS MPS types), cardiac valvulopathy (UNIVERSAL — aortic + mitral "
    "regurgitation ALL patients), OSA (tongue + tonsil + larynx DS accumulation → SEVERE airway "
    "obstruction dominant seizure trigger). INTELLIGENCE IS NORMAL — DS/C4S do NOT drive cortical "
    "neuronal damage at disease-causing levels (unlike HS in MPS-I/II/III). Seizures arise from "
    "STRUCTURAL mechanisms: communicating hydrocephalus (DS arachnoid infiltration → CSF obstruction "
    "20-30%), OSA-driven hypoxia (65%), and atlantoaxial instability (40-50%). DS ONLY elevated in "
    "urine/plasma (HS/KS NORMAL) — this GAG fingerprint distinguishes MPS-VI from MPS-I/II (HS+DS "
    "both elevated), MPS-III (HS only), MPS-IVA (KS+C6S). Galsulfase ERT reduces somatic DS burden "
    "but does NOT cross the BBB and does NOT treat the airway — BiPAP + surgical interventions required."
)

# 5 variant classes (etiologies) — deterministic percentages
ETIOLOGIES = [
    {
        "name": "Classic/Severe (Null/Null — biallelic truncating)",
        "pct": 35,
        "n": 14,
        "seizure_risk": "20-25% (communicating hydrocephalus + severe OSA dominant)",
        "eeg": "Normal background (normal intelligence); focal/generalized slowing if hydrocephalus; bilateral slow waves with OSA",
        "variant_detail": "Biallelic nonsense/frameshift — null alleles; enzyme <1% control; severe phenotype: extreme short stature, severe corneal clouding, universal valvulopathy, OSA requiring BiPAP, hepatosplenomegaly; HSCT considered if < 6-8yr",
        "hsct_eligible": True,
        "ert_alone": True,
    },
    {
        "name": "Attenuated (Biallelic missense — residual enzyme 2-10%)",
        "pct": 25,
        "n": 10,
        "seizure_risk": "10-12% (milder somatic; less severe OSA and AAI)",
        "eeg": "Normal background; focal slowing only if hydrocephalus confirmed on MRI; sleep EEG: periodic arousal pattern from OSA",
        "variant_detail": "Biallelic missense — residual enzyme 2-10% control; milder phenotype: shorter stature still present but less severe; corneal clouding milder; valvulopathy present but slower progression; OSA milder; galsulfase ERT primary",
        "hsct_eligible": False,
        "ert_alone": True,
    },
    {
        "name": "Portuguese-Brazilian Founder (p.R152W biallelic)",
        "pct": 15,
        "n": 6,
        "seizure_risk": "18-22% (intermediate severity; OSA + hydrocephalus common)",
        "eeg": "Normal background; mild slowing if hydrocephalus; cardiac arrhythmia must be distinguished from ictal events on EEG",
        "variant_detail": "p.R152W biallelic (Portuguese-Brazilian founder, ~15% MPS-VI alleles in Portugal/Brazil) — intermediate to severe phenotype; consanguineous or founder-enriched families; enzyme severely reduced; coarse facies + valvulopathy prominent; ERT + BiPAP mandatory",
        "hsct_eligible": True,
        "ert_alone": True,
    },
    {
        "name": "Intermediate Compound-het (Null + missense)",
        "pct": 15,
        "n": 6,
        "seizure_risk": "15% (intermediate somatic severity)",
        "eeg": "Normal background; focally abnormal if hydrocephalus or cord myelopathy; EEG mandatory to exclude non-epileptiform events (cardiac arrhythmia, OSA-related events)",
        "variant_detail": "Null allele (nonsense/frameshift) + missense compound-het — intermediate enzyme residual 1-5%; intermediate-to-severe phenotype; corneal clouding significant; cardiac valvulopathy progressing; OSA requiring management",
        "hsct_eligible": True,
        "ert_alone": True,
    },
    {
        "name": "Rare/Private (Deep intronic or novel biallelic)",
        "pct": 10,
        "n": 4,
        "seizure_risk": "20% (diagnostic delay → later OSA/hydrocephalus surveillance)",
        "eeg": "EEG mandatory to distinguish seizures from cardiac arrhythmia events and OSA-related events in MPS-VI; normal background between events",
        "variant_detail": "Deep intronic variants or novel biallelic private mutations — WGS + RNA-seq required if panel negative; diagnostic delay common; ARSB enzyme low but panel-negative; high urine DS confirms lysosomal storage; ARSB activity differentiates from other LSDs",
        "hsct_eligible": False,
        "ert_alone": True,
    },
]

# Seizure types (MPS-VI specific — structural dominant)
SEIZURE_TYPES = [
    {"type": "GTCS", "pct": 55, "eeg": "Bisynchronous ictal; post-ictal slowing; communicating hydrocephalus + OSA-hypoxia bilateral → GTCS dominant pattern"},
    {"type": "Focal aware", "pct": 20, "eeg": "Focal onset; preserved awareness (normal intelligence); focal cortical irritation from hydrocephalus/raised ICP"},
    {"type": "Tonic", "pct": 15, "eeg": "Diffuse fast recruitment; brainstem/cord compression from AAI; raised ICP tonic posturing"},
    {"type": "Myoclonic", "pct": 8, "eeg": "Polyspike-wave; rare; late disease; cumulative hypoxia from OSA; cardiac valve arrhythmia must be excluded"},
    {"type": "Atypical absence", "pct": 5, "eeg": "Slow spike-wave; rare; intelligence preserved so atypical absences noticed and reported; ICP elevation mimics absence spells — EEG mandatory"},
]

# Seizure triggers (MPS-VI specific)
TRIGGERS = [
    {"trigger": "OSA / airway obstruction (tongue + tonsil + larynx DS accumulation)", "pct": 65, "note": "DOMINANT trigger in MPS-VI — more severe airway obstruction than any other MPS type except MPS-II; tongue enlargement (macroglossia) + tonsillar hypertrophy + laryngeal DS infiltration → severe obstructive sleep apnea; BiPAP mandatory; tonsillectomy/adenoidectomy in pediatric cases"},
    {"trigger": "Communicating hydrocephalus (DS arachnoid villi infiltration)", "pct": 30, "note": "DS deposition in arachnoid villi → impaired CSF reabsorption → communicating hydrocephalus → raised ICP → seizures; VP shunt if ICP confirmed; brain MRI periventricular changes; LP opening pressure measurement mandatory in any MPS-VI patient with headache or seizure onset"},
    {"trigger": "Atlantoaxial instability (odontoid hypoplasia — less severe than MPS-IVA)", "pct": 45, "note": "40-50% prevalence; less severe than MPS-IVA (odontoid hypoplasia present but not as extreme); C-spine XR flexion/extension + MRI pre-GA mandatory; cord signal on MRI → surgical fusion"},
    {"trigger": "Intercurrent febrile illness", "pct": 55, "note": "Universal trigger; fever worsens OSA severity + increases metabolic demand; DS accumulation in airway worsens with fever-induced mucosal edema; hypoxia threshold lowered"},
    {"trigger": "Cardiac decompensation / arrhythmia (universal valvulopathy)", "pct": 20, "note": "Universal aortic + mitral regurgitation — cardiac arrhythmia can mimic or trigger seizures; Holter monitor mandatory if palpitations or cardiac-origin events suspected; PHT/fosphenytoin ABSOLUTE AVOID — lethal arrhythmia risk in severe valvulopathy"},
    {"trigger": "General anaesthesia / sedation", "pct": 35, "note": "EXTREME HAZARD — severe tongue enlargement + macroglossia + laryngeal DS → catastrophic airway obstruction at induction; valvulopathy → hemodynamic instability; video laryngoscopy + awake fibreoptic intubation mandatory; tracheotomy may be required in severe airway obstruction; C-spine XR + cardiac echo pre-GA"},
    {"trigger": "Missed ERT (Galsulfase) dose", "pct": 18, "note": "DS somatic rebound; airway and cardiac burden worsen with ERT interruption; OSA severity increases; weekly dosing compliance critical; infusion reaction monitoring (anti-galsulfase IgG/IgE)"},
]

# Treatments
TREATMENTS = [
    {"name": "Levetiracetam (LEV)", "level": "Level B", "role": "First-line AED — no CYP induction (galsulfase ERT pharmacokinetics preserved); no cardiac effect (safe in universal aortic/mitral valvulopathy); IV formulation for SE; renal excretion (adjust if renal involvement from DS infiltration)", "ci": None},
    {"name": "Valproate (VPA)", "level": "Level B", "role": "Broad-spectrum; POLG1 exclusion MANDATORY (CPIC A); hepatic DS accumulation in MPS-VI → LFT monitoring; less hepatic burden than MPS-I HS/DS but DS does accumulate; safe if POLG1-cleared and LFTs normal; effective for GTCS", "ci": "POLG1 mutation (mandatory exclusion); hepatic dysfunction; elevated LFTs"},
    {"name": "Clobazam (CLB)", "level": "Level B", "role": "Adjunctive for focal + GTCS; SEVERE OSA caution in MPS-VI — tongue + airway DS accumulation is more severe than MPS-IVA; benzodiazepine respiratory depression in BiPAP-dependent patients; use with caution; dose-reduce in severe OSA", "ci": "Severe OSA without non-invasive ventilation; severe airway compromise"},
    {"name": "Galsulfase (Naglazyme, ERT)", "level": "Level A (somatic)", "role": "FDA 2005; 1 mg/kg IV WEEKLY (contrast elosulfase alfa every other week); reduces urine DS; improves 6-minute walk test and pulmonary function; DOES NOT cross BBB; pre-infusion anti-galsulfase antibody screen mandatory; continue peri-operatively (do NOT hold for surgery); anaphylaxis risk — alert anaesthesia team", "ci": "Severe infusion reaction (switch to desensitization); anti-galsulfase IgE (pre-screen mandatory)"},
    {"name": "HSCT (Haematopoietic Stem Cell Transplantation)", "level": "Level A (severe, early)", "role": "Recommended for SEVERE MPS VI if age < 6-8 years — primarily somatic benefit (cognition already normal); HSCT stabilizes somatic disease progression (DS accumulation in somatic tissues); survival and 6MW improvement documented; timing critical: before severe organ involvement; contrast MPS-IVA (HSCT not indicated) and MPS-I (HSCT for cognitive preservation)", "ci": "Attenuated MPS-VI (risk-benefit unfavorable); age > 8-10yr (diminishing benefit); severe pre-existing organ failure"},
    {"name": "CPAP / BiPAP (non-invasive ventilation)", "level": "Level A (OSA dominant trigger)", "role": "OSA management in MPS-VI — MOST IMPORTANT non-pharmacological seizure management; tongue + airway DS accumulation causes SEVERE OSA; BiPAP almost always required (macroglossia prevents CPAP response); sleep study mandatory for titration; non-invasive ventilation reduces hypoxic seizure threshold; tonsillectomy/adenoidectomy in pediatric cases", "ci": None},
    {"name": "VP Shunt / ETV (hydrocephalus)", "level": "Level A (communicating hydrocephalus)", "role": "Ventriculoperitoneal shunt or endoscopic third ventriculostomy for communicating hydrocephalus (20-30% MPS-VI) — DS arachnoid infiltration → raised ICP → seizures; LP opening pressure measurement mandatory if headache/seizure onset; shunt → ICP normalization → seizure reduction; ETV less effective in communicating hydrocephalus (consider CSF diversion first)", "ci": "Uncorrected coagulopathy; active CNS infection"},
    {"name": "Cervical decompression/fusion (AAI surgical)", "level": "Level A (if cord signal or ADI >5mm)", "role": "Surgical treatment of atlantoaxial instability (40-50% MPS-VI) — indicated when ADI >5mm or T2 cord signal on MRI; less frequent than MPS-IVA but still important; EEG mandatory to distinguish epileptiform seizures from cord compression events before AED escalation; video laryngoscopy mandatory (severe airway disease)", "ci": "Anaesthesia risk extreme — pre-operative airway planning: awake fibreoptic intubation in severe macroglossia"},
    {"name": "Melatonin", "level": "Level B", "role": "Circadian disruption from OSA + nocturnal events; BiPAP/NIV first (OSA must be treated before melatonin); melatonin adjunct for circadian resync after OSA management; no interactions with ERT", "ci": None},
]

# Contraindications (MPS-VI specific)
CONTRAINDICATIONS = [
    {"drug": "Typical antipsychotics (haloperidol, chlorpromazine)", "risk": "HIGH RISK", "reason": "DS-laden basal ganglia → EPS/NMS risk (real in MPS-VI — behavioral issues uncommon in normal-IQ patients but skeletal pain management drives prescribing; haloperidol for agitation in hospital → EPS risk from DS basal ganglia infiltration)", "alternative": "Atypicals (risperidone, aripiprazole, quetiapine) with EPS monitoring; non-opioid analgesia for skeletal pain; melatonin for behavioral dysregulation"},
    {"drug": "Phenytoin / Fosphenytoin", "risk": "ABSOLUTE AVOID (stronger than MPS-IVA)", "reason": "UNIVERSAL aortic + mitral valve regurgitation ALL MPS-VI patients → cardiac conduction disease → PHT cardiac conduction toxicity → lethal arrhythmia; IV fosphenytoin in severe aortic regurgitation = ABSOLUTE AVOID (lethal arrhythmia risk); CYP2C9/3A4 induction disrupts galsulfase ERT pharmacokinetics; NO safe alternative use — replace with IV LEV for SE", "alternative": "IV Levetiracetam for status epilepticus — NO cardiac conduction effect; safe in universal valvulopathy"},
    {"drug": "Vigabatrin (VGB)", "risk": "ABSOLUTE/RELATIVE CI (STRONGEST MPS-VI contraindication)", "reason": "Corneal clouding UNIVERSAL and MOST SEVERE in MPS-VI → Goldman visual field perimetry IMPOSSIBLE in most patients; VGB retinal toxicity additive with corneal DS storage + severe corneal clouding; REMS SHARE program mandatory VF monitoring not achievable in MPS-VI with severe corneal clouding; stronger contraindication than MPS-IVA (relative CI) — consider ABSOLUTE CI in severe corneal disease", "alternative": "LEV or VPA (POLG1-cleared) for focal seizures; ACTH (Level A) if infantile spasms present; vigabatrin REMS program requires documented slit-lamp + VF — impossible in severe MPS-VI corneal clouding"},
    {"drug": "Carbamazepine / Oxcarbazepine", "risk": "CAUTION (not absolute CI)", "reason": "DS peripheral nerve axonal neuropathy risk (similar to MPS-IVA KS neuropathy); cardiac sodium-channel toxicity in UNIVERSAL valvulopathy (aortic + mitral regurgitation); HLA-B*15:02 mandatory in SE Asian ancestry before CBZ/OXC/LTG; CYP induction risk — disrupts galsulfase ERT in high-dose CBZ", "alternative": "LEV or VPA (POLG1-cleared); HLA-B*15:02 screen first in SE Asian ancestry; cardiac clearance required"},
    {"drug": "General anaesthesia (any agent)", "risk": "EXTREME HAZARD", "reason": "SEVERE macroglossia (tongue DS enlargement) + tonsillar hypertrophy + laryngeal DS infiltration → catastrophic airway obstruction at induction of anaesthesia; DIFFERENT mechanism from MPS-IVA (cord) but EQUALLY DANGEROUS; valvulopathy (universal aortic + mitral regurgitation) → hemodynamic instability; cardiac arrhythmia from haemodynamic compromise; galsulfase IgE risk intraoperatively; tracheotomy may be required if airway cannot be secured", "alternative": "Pre-GA: slit-lamp (corneal assessment) + C-spine XR/MRI + cardiac echo + pulmonary function + anti-galsulfase IgE + ENT airway assessment MANDATORY; awake fibreoptic intubation in severe macroglossia; video laryngoscopy; anaesthesia alert bracelet; ENT + cardiac + neuroanesthesia joint pre-op planning; tracheotomy set standby"},
]

# Monitoring parameters (12 items)
MONITORING = [
    "Urine DS quantification (and C4S): baseline + q3-6M on ERT (DS correlates with disease burden and ERT response; HS/KS NORMAL confirms MPS-VI; DS normalization on ERT is goal but rarely complete)",
    "Brain MRI: hydrocephalus surveillance — periventricular DS changes; communicating hydrocephalus (20-30%); T2 periventricular signal; brainstem/cord signal at foramen magnum — annual; URGENT if headache/new neurological signs; LP opening pressure if headache",
    "C-spine XR (flexion/extension views) + MRI: atlantoaxial instability (ADI) — every 6-12M AND pre-GA MANDATORY (ABSOLUTE REQUIREMENT); cord signal = urgent surgical referral",
    "Cardiac echo: aortic regurgitation + mitral valve (UNIVERSAL in ALL MPS-VI patients) — annual; gradient >40 mmHg → cardiac surgery consultation; PHT/fosphenytoin ABSOLUTE AVOID in ANY MPS-VI patient",
    "Sleep study (polysomnography): OSA dominant trigger — baseline and annual; AHI and oxygen nadir; BiPAP titration; ENT referral (tonsillectomy in pediatric patients if hypertrophy confirmed)",
    "Slit-lamp: corneal clouding grade (UNIVERSAL and MOST SEVERE) — annual; required for VGB REMS SHARE documentation (typically IMPOSSIBLE in severe MPS-VI); visual acuity assessment; corneal transplant consideration in adult severe disease",
    "Respiratory function tests: FVC + flow-volume loop — q6-12M; thoracic cage restriction; FVC <60% → NIV referral; spirometry limited by effort dependence (cooperation possible — normal intelligence)",
    "Audiometry: sensorineural + conductive hearing loss (60-70%) — annual; conductive from DS middle ear infiltration; sensorineural from cochlear DS storage; hearing aids as needed",
    "ARSB leukocyte enzyme activity: diagnosis + carrier testing; co-measure GALNS (to distinguish MPS-IVA KS pattern), IDUA, IDS for multi-enzyme panel confirmation",
    "Anti-galsulfase IgG/IgE: infusion reaction monitoring; pre-treatment baseline + q6M on ERT; IgE elevation = anaphylaxis risk (alert anaesthesia team); desensitization protocol if IgE-positive",
    "POLG1 genotyping: before VPA initiation (CPIC Level A mandatory); MPS-VI with hepatic DS accumulation can phenocopy mitochondrial hepatopathy",
    "HLA-B*15:02: SE Asian ancestry before CBZ/OXC/LTG; hypersensitivity syndrome risk in SE Asian populations",
]

# Clinical thresholds (8 items)
THRESHOLDS = [
    {"parameter": "OSA (AHI — dominant seizure trigger)", "threshold": "AHI ≥5 events/hr (any severity); BiPAP-dependent MPS-VI patients with severe macroglossia", "action": "Non-invasive ventilation: BiPAP (not CPAP — macroglossia prevents CPAP response in severe MPS-VI); sleep study titration mandatory; ENT (tonsillectomy in pediatric); BiPAP reduces hypoxic seizure threshold; PRESCRIBE BEFORE AED escalation for OSA-triggered seizures"},
    {"parameter": "Communicating hydrocephalus (ICP)", "threshold": "LP opening pressure >25 cmH2O OR periventricular T2 signal on MRI + symptoms", "action": "VP shunt referral (CSF diversion for communicating hydrocephalus from DS arachnoid infiltration); ICP normalization → seizure reduction; Diamox (acetazolamide) temporary bridging to shunt; EEG mandatory to distinguish ICP-related events from epileptiform seizures"},
    {"parameter": "Cervical cord compression (ADI + cord signal)", "threshold": "ADI >5mm on flexion/extension XR OR T2 cord signal on MRI", "action": "Urgent neurosurgical referral; cervical decompression/fusion; NO neck manipulation; anaesthesia alert (airway already extreme hazard in MPS-VI); pre-operative airway planning mandatory"},
    {"parameter": "ERT initiation", "threshold": "At diagnosis, any age", "action": "Start galsulfase 1 mg/kg IV WEEKLY; pre-infusion antibody screen (IgG/IgE); infusion reaction protocol ready; continue peri-operatively (do NOT hold for surgery); HSCT consideration if severe and < 6-8yr"},
    {"parameter": "HSCT eligibility", "threshold": "Severe MPS-VI + age < 6-8 years at transplant", "action": "HSCT referral for severe classic phenotype (biallelic null) before severe organ involvement; cognition normal so HSCT primarily somatic; ERT bridge to HSCT; outcome better with earlier transplant; attenuated disease: ERT alone"},
    {"parameter": "Aortic/mitral valve gradient", "threshold": "Aortic gradient >40 mmHg OR mitral gradient >10 mmHg", "action": "Cardiac surgery consultation (valve repair/replacement); minimize exposure to PHT/fosphenytoin (ABSOLUTE AVOID); pre-GA cardiac clearance MANDATORY; cardiac anaesthesia team involvement"},
    {"parameter": "FVC (forced vital capacity)", "threshold": "FVC <60% predicted", "action": "NIV (BiPAP) referral + respiratory team; anaesthesia risk extreme in combination with airway disease; thoracic cage restriction from skeletal DS involvement; respiratory team co-management"},
    {"parameter": "Pre-anaesthesia (any MPS-VI patient)", "threshold": "ANY MPS-VI patient requiring GA or sedation", "action": "Slit-lamp + C-spine XR/MRI + cardiac echo + pulmonary function + anti-galsulfase IgE + ENT airway assessment + anaesthesia alert MANDATORY; awake fibreoptic intubation in severe macroglossia; video laryngoscopy + tracheotomy standby; joint ENT-cardiac-neuroanesthesia planning; cardiac arrhythmia prophylaxis"},
]

# Lifecycle stages (5 stages)
LIFECYCLE = [
    {"stage": "Infantile/Toddler (0-2 yr)", "features": "Coarse facies emerging; hepatosplenomegaly (usually first sign); inguinal hernia common; normal developmental milestones INITIALLY (normal intelligence — key feature); corneal clouding beginning; cardiac murmur audible; recurrent otitis media (conductive hearing loss)", "action": "Urine DS confirmation; ARSB enzyme assay; ERT initiation (galsulfase weekly); HSCT evaluation for severe biallelic null (< 6-8yr); C-spine surveillance begins; cardiac echo baseline; ophthalmology slit-lamp"},
    {"stage": "Childhood (2-8 yr)", "features": "Progressive skeletal involvement: short stature, pectus carinatum; corneal clouding worsening (slit-lamp annual); OSA onset (macroglossia + tonsillar hypertrophy); hearing aids often needed; hepatosplenomegaly prominent; cardiac valve disease progressing; SCHOOL PERFORMANCE NORMAL (normal intelligence); atlantoaxial instability emerging", "action": "ERT continuation (galsulfase weekly); HSCT if severe and < 6-8yr (last window); OSA/BiPAP management (tonsillectomy in pediatric); slit-lamp annual; cardiac echo annual; C-spine XR q6-12M; audiometry"},
    {"stage": "School age / Adolescent (8-15 yr)", "features": "Skeletal disease maximum progression; corneal clouding severe (vision compromise); cardiac valve disease significant (regurgitation progressing); OSA requiring BiPAP; hearing loss significant; first seizures (hydrocephalus + OSA + AAI); atlantoaxial instability most dangerous; FULL SCHOOL PARTICIPATION with accommodations (normal IQ)", "action": "Galsulfase ERT lifelong; BiPAP (not CPAP); cardiac echo + cardiac surgery consultation if gradient rising; C-spine MRI intensified; corneal specialist; AED if epileptiform seizures confirmed on EEG; VP shunt if hydrocephalus + ICP elevated"},
    {"stage": "Young adult (15-30 yr)", "features": "Progressive cardiac burden (aortic/mitral regurgitation dominant); ambulatory decline (many require wheelchair by 20s-30s); significant corneal clouding (vision limited); hearing loss requiring aids; respiratory restriction; seizures ongoing (OSA + hydrocephalus + AAI); INTELLIGENCE PRESERVED — employment/independence possible with accommodations; pain from joint disease", "action": "Cardiac echo annual; cardiac surgery for valve disease; respiratory function q6M; pain management (non-opioid preferred — normal cognition allows medication compliance); vocational planning; ERT lifelong; AED maintenance; VP shunt monitoring"},
    {"stage": "Adult/Late (30+ yr)", "features": "Severe cardiac/respiratory burden; wheelchair-dependent in many; corneal clouding may require corneal transplant; significant hearing loss; seizure risk ongoing from cumulative OSA + hydrocephalus; ERT continues; longevity reduced vs general population but survival to 4th-5th decade possible with treatment", "action": "Palliative/supportive care planning; cardiac valve management (replacement often needed); respiratory support (BiPAP + NIV); ERT continues; AED maintenance; corneal transplant consideration; anaesthesia EXTREME CAUTION for any procedure — airway + cardiac dual hazard"},
]

# Differential diagnosis (8 conditions)
DIFFERENTIAL_DIAGNOSIS = [
    {"condition": "MPS-I (IDUA — Hurler/Hurler-Scheie/Scheie)", "distinction": "HS + DS BOTH elevated (NOT DS alone) — different GAG fingerprint; cognitive decline in Hurler/Hurler-Scheie (UNLIKE MPS-VI where intelligence NORMAL); corneal clouding present in MPS-I too; HSCT GOLD STANDARD for Hurler (cognition preservation — different from MPS-VI where HSCT somatic only); laronidase ERT; IDUA enzyme deficient"},
    {"condition": "MPS-II (IDS — Hunter Syndrome)", "distinction": "HS + DS BOTH elevated; X-LINKED (hemizygous males — UNLIKE MPS-VI AR); ABSENT corneal clouding (pathognomonic difference — MPS-VI corneal clouding UNIVERSAL); pebbly ivory skin papules pathognomonic in MPS-II; severe cognitive decline in severe MPS-II; IDS enzyme deficient"},
    {"condition": "MPS-IVA (GALNS — Morquio A)", "distinction": "KS + C6S elevated (NOT DS) — entirely different GAG fingerprint; skeletal dominant (Morquio phenotype — pectus carinatum, genu valgum, odontoid hypoplasia WORST ALL MPS); normal intelligence (shared with MPS-VI); elosulfase alfa ERT (every other week vs galsulfase weekly); NO corneal clouding severity MPS-IVA (milder); GALNS enzyme deficient"},
    {"condition": "MPS-VII (GUSB — Sly Syndrome)", "distinction": "HS + DS + CS ALL elevated (more substrates than MPS-VI); EXTREMELY RARE; hydrops fetalis common; intelligence variable (contrast MPS-VI NORMAL); GUSB (beta-glucuronidase) enzyme deficient; vestronidase alfa ERT (FDA 2017); very different multi-substrate phenotype"},
    {"condition": "GM1-Gangliosidosis (GLB1)", "distinction": "NOT a DS storage disorder — ganglioside GM1 and keratan sulfate accumulation (NOT dermatan sulfate); GLB1 (beta-galactosidase) deficient; SEVERE cognitive decline (unlike MPS-VI normal intelligence); cherry-red spot on ophthalmology (unlike MPS-VI); hepatosplenomegaly and coarse facies shared but GAG fingerprint entirely different"},
    {"condition": "MPS-III (SGSH/NAGLU/HGSNAT/GNS — Sanfilippo)", "distinction": "HS ONLY elevated (NOT DS); SEVERE COGNITIVE DECLINE — behavioral phase + dementia (UNLIKE MPS-VI where intelligence NORMAL); milder somatic features; MPS-III is PRIMARY CNS disease; corneal clouding ABSENT (unlike MPS-VI UNIVERSAL); no approved ERT; HSCT NOT recommended MPS-III"},
    {"condition": "MPS-I Attenuated (Scheie/Hurler-Scheie)", "distinction": "HS + DS BOTH elevated (NOT DS alone); ATTENUATED cognitive involvement (partial overlap with MPS-VI normal intelligence in Scheie); corneal clouding present in MPS-I too; IDUA enzyme deficient; laronidase ERT; HSCT benefit if young and severe — cognitive preservation rationale UNLIKE MPS-VI somatic-only HSCT"},
    {"condition": "POLG-related epilepsy (POLG1 mutations)", "distinction": "Mitochondrial NOT lysosomal; Alpers syndrome phenocopy; VPA ABSOLUTE CI (fatal hepatotoxicity); muscle biopsy ragged-red fibres; NO DS in urine; no ARSB enzyme deficiency; exclude before VPA in any pediatric epilepsy with hepatic involvement or abnormal LFTs"},
]

# Definitions glossary (12 terms)
DEFINITIONS = [
    {"term": "ARSB", "definition": "Arylsulfatase B (N-acetylgalactosamine-4-sulfatase) — lysosomal enzyme that cleaves sulfate from N-acetylgalactosamine-4-sulfate residues in dermatan sulfate (DS) and chondroitin-4-sulfate (C4S). ARSB deficiency causes MPS-VI (Maroteaux-Lamy syndrome)."},
    {"term": "MPS-VI (Maroteaux-Lamy)", "definition": "Mucopolysaccharidosis type VI — autosomal recessive ARSB deficiency. DS + C4S accumulate in ALL somatic tissues. KEY FEATURES: NORMAL INTELLIGENCE (contrasts MPS-I/II/III), universal cardiac valvulopathy, corneal clouding UNIVERSAL and MOST SEVERE, OSA dominant trigger. Epilepsy 15-25% from structural mechanisms (hydrocephalus + OSA + AAI)."},
    {"term": "Galsulfase (Naglazyme)", "definition": "Recombinant human N-acetylgalactosamine-4-sulfatase. FDA-approved 2005. Dose: 1 mg/kg IV WEEKLY (contrast elosulfase alfa MPS-IVA every other week). Reduces urine DS; improves 6-minute walk test; DOES NOT cross BBB. Continue peri-operatively — do NOT hold for surgery. Alert anaesthesia team to galsulfase IgE anaphylaxis risk."},
    {"term": "DS (Dermatan Sulfate)", "definition": "Primary lysosomal substrate in MPS-VI. DS accumulates in connective tissue, cardiac valves, cornea, airway soft tissue, arachnoid villi, bone. Urine DS elevated (HS/KS NORMAL) — the diagnostic GAG fingerprint of MPS-VI. DS in arachnoid villi → communicating hydrocephalus. DS in cardiac valves → universal valvulopathy. DS in tongue/larynx → severe OSA."},
    {"term": "Communicating Hydrocephalus", "definition": "CSF accumulation from impaired reabsorption at arachnoid villi (DS infiltration → arachnoid villi dysfunction). 20-30% of MPS-VI patients. Manifest as periventricular T2 signal on MRI + raised LP opening pressure. Primary seizure mechanism in MPS-VI alongside OSA. VP shunt or ETV for ICP control. LP opening pressure mandatory in any MPS-VI patient with headache or seizure onset."},
    {"term": "OSA in MPS-VI (dominant seizure trigger)", "definition": "Obstructive sleep apnea from DS accumulation in tongue (macroglossia), tonsils, larynx → severe airway obstruction. DOMINANT seizure trigger in MPS-VI — more severe airway disease than any other MPS type except MPS-II. BiPAP (not CPAP) required in most patients (macroglossia prevents CPAP response). Tonsillectomy in pediatric MPS-VI. BiPAP reduces hypoxic seizure threshold."},
    {"term": "Cardiac Valvulopathy (MPS-VI Universal)", "definition": "Aortic regurgitation + mitral regurgitation in ALL MPS-VI patients — DS infiltration of valve leaflets and chordae tendineae. Prevalence: ~100% with varying severity. Mechanism: DS accumulation in valve leaflets → thickening + regurgitation. PHT/fosphenytoin ABSOLUTE AVOID in any MPS-VI patient — lethal arrhythmia risk in aortic regurgitation + cardiac conduction disease."},
    {"term": "Corneal Clouding (MPS-VI — Universal and Most Severe)", "definition": "Corneal DS accumulation causing clouding in ALL MPS-VI patients — MOST SEVERE among non-CNS MPS types (contrast MPS-II where corneal clouding ABSENT). Goldman visual field perimetry IMPOSSIBLE in severe clouding → VGB contraindication strongest among MPS diseases. Slit-lamp annual. Corneal transplant in adult severe disease. Contrast MPS-IVA where corneal clouding milder (late onset)."},
    {"term": "DRE (Drug-Resistant Epilepsy) in MPS-VI", "definition": "Failure of ≥2 appropriate AEDs at adequate doses. 8-12% in MPS-VI. Seizures are structural (OSA + hydrocephalus + AAI) NOT epileptiform cortical hyperexcitability. Video-EEG mandatory to exclude: (1) OSA-related events, (2) cardiac arrhythmia events (universal valvulopathy), (3) ICP spikes from hydrocephalus — before DRE label. Treat underlying structural cause first (BiPAP, VP shunt, cervical fusion)."},
    {"term": "Normal Intelligence (MPS-VI key feature)", "definition": "MPS-VI patients have NORMAL intelligence — the KEY distinguishing feature from MPS-I/II/III. DS/C4S do NOT accumulate in neurons at disease-causing levels. Unlike HS (MPS-I/II/III) which drives cortical neuronal damage and dementia, DS accumulates in connective tissue, valves, cornea, and airway. Children attend school with normal cognitive function. This distinguishes MPS-VI most importantly from Hurler (MPS-IH) and Hunter (MPS-II) syndromes."},
    {"term": "HSCT in MPS-VI (somatic benefit only)", "definition": "Haematopoietic stem cell transplantation in SEVERE MPS-VI (< 6-8 years) — primarily somatic benefit (stabilizes DS accumulation in somatic tissues). IMPORTANT DISTINCTION: MPS-VI HSCT is NOT for cognitive preservation (intelligence already normal) — contrast MPS-I (Hurler) where HSCT IS for cognitive preservation. MPS-VI HSCT primarily improves somatic outcomes (6MW, respiratory, cardiac). Timing critical: before severe organ involvement."},
    {"term": "Macroglossia (tongue enlargement — MPS-VI airway dominant)", "definition": "DS accumulation in tongue → progressive macroglossia (tongue enlargement). Primary contributor to severe OSA in MPS-VI. Also contributes to catastrophic airway obstruction at anaesthesia induction (EXTREME HAZARD). More severe than MPS-IVA airway disease (which is tracheomalacia/thoracic restriction dominant). BiPAP (not CPAP) required; tonsillectomy in pediatric; tracheotomy standby for GA."},
]

KEY_CONCEPTS = [
    "DS ONLY elevated in MPS-VI (HS/KS NORMAL): diagnostic GAG fingerprint — distinguishes from MPS-I/II (HS+DS both elevated), MPS-III (HS only), MPS-IVA (KS+C6S); urine DS quantification is the primary screening test; HS/KS NORMAL in MPS-VI confirms the diagnosis",
    "Normal intelligence: KEY distinguishing feature of MPS-VI from MPS-I/II/III — cognitive decline ABSENT; DS/C4S do NOT drive neuronal damage; seizures are NOT from GAG-driven cortical hyperexcitability but from structural mechanisms (OSA hypoxia + communicating hydrocephalus + atlantoaxial instability); children attend school normally",
    "OSA DOMINANT seizure trigger (strongest in MPS-VI of all MPS types): macroglossia (tongue DS enlargement) + tonsillar hypertrophy + laryngeal DS → SEVERE airway obstruction; BiPAP mandatory (CPAP insufficient in severe macroglossia); tonsillectomy in pediatric; BiPAP BEFORE AED escalation for OSA-triggered seizures",
    "Corneal clouding UNIVERSAL and MOST SEVERE (non-CNS MPS): ALL MPS-VI patients affected; Goldman visual field perimetry IMPOSSIBLE in severe disease → VGB ABSOLUTE/RELATIVE CI (STRONGEST contraindication among all MPS diseases); slit-lamp annual; corneal transplant in adults; contrast MPS-II (corneal clouding ABSENT)",
    "Cardiac valvulopathy UNIVERSAL (ALL MPS-VI patients): aortic + mitral regurgitation ALL patients → PHT/fosphenytoin ABSOLUTE AVOID (lethal arrhythmia risk); cardiac echo annual mandatory; cardiac surgery when gradient >40 mmHg; cardiac arrhythmia can mimic or trigger seizures — Holter monitor if suspected",
    "Galsulfase (Naglazyme) ERT: 1 mg/kg IV WEEKLY (not every other week); FDA 2005; reduces urine DS; improves 6-minute walk test; DOES NOT cross BBB; continue peri-operatively (do NOT hold for surgery); anti-galsulfase IgE pre-screen mandatory (anaphylaxis risk — alert anaesthesia team)",
    "HSCT for severe MPS-VI (< 6-8yr): different from MPS-IVA (no HSCT) and MPS-I (HSCT for cognitive preservation); MPS-VI HSCT primarily SOMATIC benefit (cognition already normal); improves 6MW, respiratory function, cardiac; timing critical — before severe organ involvement; attenuated: ERT alone",
    "Communicating hydrocephalus 20-30%: DS arachnoid villi infiltration → CSF reabsorption failure → raised ICP → seizures; LP opening pressure mandatory in any MPS-VI patient with headache or seizure onset; VP shunt for ICP control; periventricular T2 MRI signal as surveillance",
    "PHT/Fosphenytoin ABSOLUTE AVOID (strongest contraindication — MPS-VI): UNIVERSAL aortic + mitral valvulopathy → cardiac conduction disease + arrhythmia; IV fosphenytoin in severe aortic regurgitation → lethal arrhythmia; CYP induction disrupts galsulfase ERT; use IV LEV for SE in all MPS-VI patients",
    "VGB ABSOLUTE/RELATIVE CI (STRONGEST among MPS): corneal clouding UNIVERSAL and SEVERE → Goldman visual field perimetry IMPOSSIBLE; VGB retinal toxicity additive with corneal DS storage; REMS SHARE VF monitoring not achievable — stronger than MPS-IVA (relative CI); ACTH for infantile spasms preferred",
    "Anesthesia EXTREME HAZARD (different mechanism from MPS-IVA): SEVERE macroglossia + tonsillar hypertrophy + laryngeal DS → catastrophic airway obstruction at induction; universal valvulopathy → hemodynamic instability + arrhythmia; tracheotomy standby; awake fibreoptic intubation; video laryngoscopy; joint ENT-cardiac-neuroanesthesia team",
    "CBZ/OXC CAUTION: DS peripheral nerve axonal neuropathy; cardiac sodium-channel risk in UNIVERSAL valvulopathy; HLA-B*15:02 mandatory SE Asian ancestry before CBZ/OXC/LTG; CYP induction risk disrupts galsulfase",
    "POLG1 exclusion mandatory before VPA: hepatic DS accumulation in MPS-VI → monitor LFTs; POLG1 mutations → fatal hepatotoxicity with VPA; CPIC Level A mandatory screening; if hepatic dysfunction present, VPA contraindicated regardless of POLG1 status",
    "AR inheritance: both sexes equally affected (contrast IDS/MPS-II which is X-linked); p.R152W Portuguese-Brazilian founder (~15% MPS-VI alleles in Portugal/Brazil); consanguinity enriches Middle East/SA/Mediterranean populations; homozygous null = severe phenotype; ARSB pseudogene absent (unlike IDS/IDSP1)",
    "ATL instability 40-50% (less severe than MPS-IVA but significant): odontoid hypoplasia present; C-spine XR flexion/extension + MRI pre-GA mandatory; cord signal → surgical fusion; EEG mandatory to distinguish epileptiform seizures from cord compression events before DRE label",
]

STANDARDS = [
    "Harmatz et al. 2006 — Galsulfase (Naglazyme) Phase III MPS-VI trial GINS: 39-patient RCT; primary endpoint 12-minute walk test; FDA approval basis (Annals of Internal Medicine)",
    "Hendriksz et al. 2013 — International MPS-VI management guidelines: ERT, HSCT, monitoring, surgical interventions (J Inherit Metab Dis)",
    "Brands et al. 2021 — HSCT outcomes in MPS-VI: timing, somatic benefit, comparison with MPS-I (J Inherit Metab Dis / Orphanet)",
    "Swiedler et al. 2005 — Phase 2/3 MPS-VI galsulfase open-label extension: long-term safety and efficacy; urine DS normalization",
    "CPIC 2023 — POLG1 genotyping Level A guideline before valproate initiation in any pediatric epilepsy",
    "REMS Vigabatrin (SHARE program) — mandatory visual field perimetry q3M; MPS-VI corneal clouding makes VGB SHARE compliance impossible in most patients; document contraindication",
    "Wraith 2004 — MPS-VI (Maroteaux-Lamy): disease review, natural history, clinical spectrum, diagnosis (J Inherit Metab Dis)",
    "ICH E6(R2) GCP — Lysosomal Storage Disease clinical trial standards; lysosomal enzyme replacement therapy pharmacokinetics",
]


def get_overview():
    """Return ARSB/MPS-VI overview for /api/arsb/overview."""
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "disease_mechanism": DISEASE_MECHANISM,
        "cohort_size": COHORT_SIZE,
        "epilepsy_prevalence_pct": {"overall": "15-25", "classic_severe": "20-25", "attenuated": "10-12"},
        "drug_resistance_pct": {"overall": "8-12"},
        "osa_pct": 62,
        "communicating_hydrocephalus_pct": 25,
        "atlantoaxial_instability_pct": 45,
        "corneal_clouding_pct": 100,
        "cardiac_valvulopathy_pct": 100,
        "hearing_loss_pct": 65,
        "on_ert_pct": 84,
        "on_hsct_pct": 18,
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "key_concepts": KEY_CONCEPTS[:6],
        "standards": STANDARDS,
    }


def get_breakdown():
    """Return ARSB/MPS-VI 40-patient cohort for /api/arsb/breakdown."""
    rng = random.Random(42)

    phenotype_weights = {
        "Classic/Severe": 0.35,
        "Attenuated": 0.25,
        "Portuguese-Brazilian-Founder": 0.15,
        "Intermediate": 0.15,
        "Rare-Private": 0.10,
    }
    phenotype_seizure_rates = {
        "Classic/Severe": 0.22,
        "Attenuated": 0.11,
        "Portuguese-Brazilian-Founder": 0.20,
        "Intermediate": 0.15,
        "Rare-Private": 0.20,
    }
    phenotype_dre_rates = {
        "Classic/Severe": 0.10,
        "Attenuated": 0.05,
        "Portuguese-Brazilian-Founder": 0.09,
        "Intermediate": 0.07,
        "Rare-Private": 0.10,
    }

    etiology_by_phenotype = {
        "Classic/Severe": "Classic/Severe (Null/Null — biallelic truncating)",
        "Attenuated": "Attenuated (Biallelic missense — residual enzyme 2-10%)",
        "Portuguese-Brazilian-Founder": "Portuguese-Brazilian Founder (p.R152W biallelic)",
        "Intermediate": "Intermediate Compound-het (Null + missense)",
        "Rare-Private": "Rare/Private (Deep intronic or novel biallelic)",
    }

    aed_first_line = ["Levetiracetam (LEV)", "Valproate (VPA — POLG1-cleared)", "Clobazam (CLB)"]
    seizure_type_pool = [st["type"] for st in SEIZURE_TYPES]
    trigger_pool = [t["trigger"][:60] for t in TRIGGERS]
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
    atlantoaxial_n = 0
    vp_shunt_n = 0

    for i in range(COHORT_SIZE):
        pid = i + 1
        pheno = phenotype_pool[i]
        etiol = etiology_by_phenotype[pheno]
        has_seizures = rng.random() < phenotype_seizure_rates[pheno]
        has_dre = has_seizures and rng.random() < phenotype_dre_rates[pheno]
        osa = rng.random() < 0.62
        hydrocephalus = rng.random() < 0.25
        atlantoaxial = rng.random() < 0.45
        vp_shunt = hydrocephalus and rng.random() < 0.55
        on_ert = rng.random() < 0.84
        on_hsct = pheno in ("Classic/Severe", "Portuguese-Brazilian-Founder") and rng.random() < 0.22
        hearing_loss = rng.random() < 0.65
        cardiac_valvulopathy = True  # Universal in MPS-VI
        corneal_clouding = True  # Universal in MPS-VI

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
        if atlantoaxial:
            atlantoaxial_n += 1
        if vp_shunt:
            vp_shunt_n += 1

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

        age_onset = rng.randint(6, 20) if has_seizures else None

        patients.append({
            "patient_id": f"ARSB-{pid:02d}",
            "phenotype": pheno,
            "etiology": etiol,
            "has_seizures": has_seizures,
            "age_onset_seizures_yrs": age_onset,
            "seizure_types": sz_types,
            "primary_aed": primary_aed,
            "drug_response": drug_response,
            "drug_resistant": has_dre,
            "osa": osa,
            "on_bipap": osa,
            "hydrocephalus": hydrocephalus,
            "vp_shunt": vp_shunt,
            "on_ert": on_ert,
            "post_hsct": on_hsct,
            "atlantoaxial_instability": atlantoaxial,
            "corneal_clouding": corneal_clouding,
            "cardiac_valvulopathy": cardiac_valvulopathy,
            "hearing_loss": hearing_loss,
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
        "atlantoaxial_n": atlantoaxial_n,
        "atlantoaxial_pct": round(atlantoaxial_n / COHORT_SIZE * 100, 1),
        "vp_shunt_n": vp_shunt_n,
        "vp_shunt_pct": round(vp_shunt_n / COHORT_SIZE * 100, 1),
        "corneal_clouding_n": COHORT_SIZE,
        "corneal_clouding_pct": 100.0,
        "cardiac_valvulopathy_n": COHORT_SIZE,
        "cardiac_valvulopathy_pct": 100.0,
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
    """Return ARSB/MPS-VI definitions for /api/arsb/definitions."""
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "definitions": DEFINITIONS,
        "key_concepts": KEY_CONCEPTS,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "standards": STANDARDS,
        "diagnostic_algorithm": [
            "Step 1: Urine GAG quantification — DS + C4S elevated; HS/KS NORMAL (MPS-VI fingerprint); distinguishes from MPS-I/II (HS+DS both elevated), MPS-III (HS only), MPS-IVA (KS+C6S); DS quantification by LC-MS/MS or DMB dye assay",
            "Step 2: Leukocyte ARSB enzyme activity — <5% control confirms MPS-VI; co-measure IDUA/IDS/GALNS/NAGLU on multi-enzyme LSD panel to exclude other MPS types; ARSB pseudogene absent (unlike IDS/IDSP1 — no pseudogene complication)",
            "Step 3: ARSB gene sequencing — biallelic variants (AR); targeted panel first (p.R152W Portuguese-Brazilian founder ~15%); if heterozygous or panel-negative → full gene sequencing + MLPA for exon deletions/duplications; consanguineous pedigrees may be homozygous for founder or private variants",
            "Step 4: Phenotype assessment — corneal clouding (universal, slit-lamp); coarse facies; hepatosplenomegaly; skeletal X-rays (dysostosis multiplex); echocardiogram (aortic + mitral regurgitation — universal at any age); cognitive assessment (NORMAL — no intellectual disability in MPS-VI)",
            "Step 5: C-spine XR flexion/extension + MRI — atlantoaxial instability (ADI measurement); T2 cord signal; 40-50% prevalence; pre-GA MANDATORY in all MPS-VI patients; urgent surgical referral if ADI >5mm or cord signal",
            "Step 6: Brain MRI — communicating hydrocephalus (periventricular T2 signal); CSF spaces; 20-30% prevalence; LP opening pressure if headache or seizure onset; white matter changes suggest DS arachnoid accumulation",
            "Step 7: Cardiac echo — aortic regurgitation + mitral valve (UNIVERSAL); gradient measurement; arrhythmia risk assessment; Holter monitor if palpitations; baseline at diagnosis + annual",
            "Step 8: Sleep study (polysomnography) — OSA severity (AHI) — dominant seizure trigger; BiPAP titration; ENT airway assessment (tonsillectomy in pediatric); oxygen nadir measurement; OSA must be treated BEFORE AED escalation for seizures",
            "Step 9: Ophthalmology — slit-lamp corneal clouding grade (UNIVERSAL); visual acuity; Goldman visual field (often IMPOSSIBLE in severe corneal disease — document for VGB contraindication); corneal specialist referral for adult severe disease",
            "Step 10: POLG1 genotyping before VPA (CPIC Level A mandatory); HLA-B*15:02 before CBZ/OXC/LTG in SE Asian ancestry; anti-galsulfase IgG/IgE baseline before ERT initiation; audiometry (60-70% hearing loss); multidisciplinary planning: HSCT evaluation if severe + age < 6-8yr",
        ],
        "pharmacological_distinctions": [
            "LEV FIRST-LINE: no CYP induction (galsulfase ERT pharmacokinetics preserved); no cardiac effect (safe in UNIVERSAL aortic/mitral valvulopathy); IV formulation for SE — preferred over PHT/fosphenytoin in ALL MPS-VI patients; renal excretion; no hepatic DS interaction",
            "VPA SECOND-LINE: POLG1 exclusion MANDATORY (CPIC A); hepatic DS accumulation in MPS-VI → LFT monitoring; if LFTs elevated → VPA contraindicated; effective for GTCS; DS liver infiltration less hepatotoxic than HS/DS in MPS-I but monitoring required",
            "PHT/Fosphenytoin ABSOLUTE AVOID (strongest contraindication in MPS-VI): UNIVERSAL aortic + mitral valvulopathy → cardiac conduction disease → lethal arrhythmia with PHT; IV fosphenytoin in aortic regurgitation = ABSOLUTE AVOID; CYP2C9/3A4 induction disrupts galsulfase ERT; replace with IV LEV for ALL SE in MPS-VI",
            "Typical antipsychotics HIGH RISK: DS-laden basal ganglia → EPS/NMS risk; behavioral issues uncommon (normal intelligence) but skeletal pain management drives opioid prescribing (consider non-opioid alternatives); use atypicals with EPS monitoring if antipsychotic required",
            "VGB ABSOLUTE/RELATIVE CI (STRONGEST among all MPS diseases): corneal clouding UNIVERSAL and MOST SEVERE → Goldman visual field perimetry IMPOSSIBLE in most MPS-VI patients; VGB retinal toxicity additive with corneal DS storage; REMS SHARE VF monitoring not achievable; document ABSOLUTE CI in severe corneal disease; ACTH for infantile spasms instead",
            "CBZ/OXC CAUTION (not absolute CI): DS peripheral nerve axonal neuropathy; cardiac sodium-channel risk in UNIVERSAL valvulopathy; HLA-B*15:02 mandatory in SE Asian ancestry; CYP induction disrupts galsulfase ERT at high CBZ doses; monitor cardiac conduction carefully if used",
            "Clobazam CAUTION in severe OSA: benzodiazepine respiratory depression in BiPAP-dependent patients with severe macroglossia; dose-reduce; ensure NIV in place before initiating; adjunctive use only; avoid in non-invasive ventilation-naive patients with severe airway disease",
            "Anesthesia EXTREME HAZARD (different from MPS-IVA — airway dominant): SEVERE macroglossia + tonsillar hypertrophy + laryngeal DS → airway obstruction at induction (different mechanism from MPS-IVA cord); UNIVERSAL valvulopathy → haemodynamic instability; awake fibreoptic intubation mandatory; tracheotomy standby; video laryngoscopy; cardiac anaesthesia required; alert to galsulfase IgE anaphylaxis risk",
            "Galsulfase (Naglazyme) 1 mg/kg WEEKLY: contrast elosulfase alfa (MPS-IVA every other week); anti-galsulfase IgE pre-screen mandatory; anaphylaxis risk — inform anaesthesia team; continue peri-operatively (do NOT hold for surgery); reduces urine DS; improves walk test (somatic benefit only — does NOT cross BBB)",
            "BiPAP (not CPAP) for OSA — primary seizure management: macroglossia prevents CPAP response in severe MPS-VI; BiPAP mandatory; prescribe BEFORE AED escalation for OSA-triggered seizures; tonsillectomy in pediatric MPS-VI for tonsillar OSA component; ENT referral mandatory at OSA diagnosis",
            "VP Shunt for communicating hydrocephalus: LP opening pressure mandatory if headache/seizure onset; EEG to distinguish ICP-related from epileptiform events; VP shunt ICP normalization → seizure reduction; Diamox bridging if surgical delay; ETV less effective in communicating hydrocephalus pattern",
            "HSCT somatic (NOT cognitive) rationale in MPS-VI: cognition already normal → HSCT NOT for cognitive preservation (unlike Hurler); MPS-VI HSCT primarily stabilizes somatic DS accumulation (cardiac, respiratory, 6MW); severe phenotype + age < 6-8yr → HSCT referral; ERT bridge to HSCT; attenuated disease → ERT alone; contrast MPS-IVA where HSCT not indicated at all",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== ARSB OVERVIEW ===")
    ov = get_overview()
    print(f"Gene: {ov['gene']} | Locus: {ov['locus']} | OMIM: {ov['omim']} | Cohort: {ov['cohort_size']}")
    print(f"Epilepsy overall: {ov['epilepsy_prevalence_pct']['overall']}% | DRE: {ov['drug_resistance_pct']['overall']}%")
    print(f"OSA: {ov['osa_pct']}% | Hydrocephalus: {ov['communicating_hydrocephalus_pct']}% | AAI: {ov['atlantoaxial_instability_pct']}%")
    print(f"ERT: {ov['on_ert_pct']}% | HSCT: {ov['on_hsct_pct']}% | Corneal clouding: 100% | Cardiac: 100%")
    print("\n=== ARSB BREAKDOWN ===")
    bk = get_breakdown()
    print(f"Seizures: {bk['seizure_n']}/{COHORT_SIZE} ({bk['seizure_pct']}%)")
    print(f"DRE: {bk['drug_resistant_n']} ({bk['drug_resistant_pct']}%)")
    print(f"OSA: {bk['osa_n']} ({bk['osa_pct']}%) | Hydrocephalus: {bk['hydrocephalus_n']} ({bk['hydrocephalus_pct']}%)")
    print(f"ERT: {bk['on_ert_n']} ({bk['on_ert_pct']}%) | HSCT: {bk['on_hsct_n']} ({bk['on_hsct_pct']}%)")
    print(f"Corneal clouding: {bk['corneal_clouding_n']} (100%) | Cardiac: {bk['cardiac_valvulopathy_n']} (100%)")
    print("\n=== ARSB DEFINITIONS ===")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])} terms | Key concepts: {len(df['key_concepts'])}")
    print(f"Diagnostic algorithm: {len(df['diagnostic_algorithm'])} steps")
