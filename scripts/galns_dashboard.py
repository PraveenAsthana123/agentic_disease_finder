#!/usr/bin/env python3
"""GALNS / MPS-IVA (Morquio A Syndrome) Epilepsy Dashboard — seed data module.

MPS-IVA: N-acetylgalactosamine-6-sulfate sulfatase deficiency (GALNS, 16q24.3, AR).
KS (Keratan Sulfate) + C6S (Chondroitin-6-Sulfate) elevated — NOT heparan sulfate/dermatan sulfate.
This KS+C6S fingerprint distinguishes MPS-IVA from MPS-I/II (HS+DS), MPS-III (HS only), MPS-VI (DS only).
PRIMARY SKELETAL DISEASE: short stature (universal), pectus carinatum, genu valgum, scoliosis,
atlantoaxial instability (MOST SEVERE among ALL MPS types — odontoid hypoplasia).
KEY DISTINGUISHING FEATURE: NORMAL OR NEAR-NORMAL INTELLIGENCE — cognitive decline ABSENT
(contrasts MPS-I/II/III which have cognitive decline). Seizures NOT from CNS GAG but from
structural cord compression (atlantoaxial instability → myelopathy).
ERT: Elosulfase alfa (Vimizim, FDA 2014) — 2 mg/kg every OTHER week IV (not weekly like Laronidase).
No standard HSCT: cognition normal → HSCT offers NO cognitive benefit; experimental only.
Epilepsy 10-15% (lowest among progressive MPS); primary trigger: atlantoaxial instability.
Corneal clouding: present (late onset, milder than MPS-I; slit-lamp required).
Founder mutations: p.R386C (European ~15%), p.I113F (Saudi), p.T291N (Middle Eastern), p.G340D.
Anesthesia EXTREME HAZARD: odontoid hypoplasia = WORST AAI among all MPS types; neck hyperextension
during laryngoscopy can cause DEATH; video laryngoscopy + C-spine immobilization mandatory.
"""
import random

GENE = "GALNS"
LOCUS = "16q24.3"
OMIM = "612222"
INHERITANCE = "Autosomal Recessive (AR) — biallelic GALNS LOF; both males AND females equally affected"
COHORT_SIZE = 40
DISEASE_MECHANISM = (
    "N-acetylgalactosamine-6-sulfate sulfatase deficiency → lysosomal accumulation of keratan sulfate (KS) "
    "+ chondroitin-6-sulfate (C6S) in ALL somatic tissues (cartilage, bone, cornea, cardiac valves, airways). "
    "KS/C6S accumulation causes PRIMARY SKELETAL DISEASE (short stature, pectus carinatum, genu valgum, "
    "scoliosis, odontoid hypoplasia). INTELLIGENCE IS NORMAL OR NEAR-NORMAL — KS/C6S do NOT drive the "
    "same CNS neuronal damage as HS in MPS-I/II/III. Seizures arise from STRUCTURAL cord compression "
    "(atlantoaxial instability → cervical myelopathy → hypoxic-ischemic mechanism) NOT from CNS GAG "
    "accumulation. KS/C6S fingerprint distinguishes MPS-IVA from MPS-I/II (HS+DS both elevated) and "
    "MPS-III (HS only) and MPS-VI (DS only). HS/DS NORMAL or mildly elevated in MPS-IVA — key diagnostic "
    "distinction. Odontoid (dens) hypoplasia + transverse ligament KS deposition = WORST atlantoaxial "
    "instability among all MPS types → life-threatening cord compression with neck hyperextension."
)

# 5 variant classes (etiologies) — deterministic percentages
ETIOLOGIES = [
    {
        "name": "Classic/Severe (Null/Null — biallelic truncating)",
        "pct": 35,
        "n": 14,
        "seizure_risk": "12-15% (structural cord compression dominant)",
        "eeg": "Normal background (normal intelligence); focal slowing if myelopathy present",
        "variant_detail": "Biallelic nonsense/frameshift — null alleles; enzyme <1% control; severe skeletal; odontoid hypoplasia severe",
        "hsct_eligible": False,
        "ert_alone": True,
    },
    {
        "name": "Attenuated (Biallelic missense — p.R386C compound or homozygous)",
        "pct": 25,
        "n": 10,
        "seizure_risk": "8-10% (milder skeletal, less severe AAI)",
        "eeg": "Normal background; focal slowing only if myelopathy confirmed on MRI",
        "variant_detail": "p.R386C (European founder ~15%) homozygous or compound-het missense/missense — residual enzyme 2-8%; milder skeletal phenotype",
        "hsct_eligible": False,
        "ert_alone": True,
    },
    {
        "name": "Saudi Founder (p.I113F biallelic — consanguineous Middle East)",
        "pct": 20,
        "n": 8,
        "seizure_risk": "10-12% (consanguineous — severe skeletal common)",
        "eeg": "Normal background; focally abnormal if cord signal present",
        "variant_detail": "p.I113F biallelic (Saudi founder) — consanguineous Middle Eastern families; enzyme severely reduced; pectus + odontoid hypoplasia prominent",
        "hsct_eligible": False,
        "ert_alone": True,
    },
    {
        "name": "Intermediate (Null + missense — compound-het)",
        "pct": 12,
        "n": 5,
        "seizure_risk": "12% (intermediate skeletal severity)",
        "eeg": "Normal background; focal changes if myelopathy; no diffuse slowing",
        "variant_detail": "Null allele (nonsense/frameshift) + missense compound-het — intermediate enzyme residual; moderate skeletal + cardiac + corneal phenotype",
        "hsct_eligible": False,
        "ert_alone": True,
    },
    {
        "name": "Rare/Private (Deep intronic or novel biallelic)",
        "pct": 8,
        "n": 3,
        "seizure_risk": "15% (diagnostic delay → later cervical surveillance)",
        "eeg": "Normal background; EEG mandatory to distinguish cord-compression events from epileptiform seizures",
        "variant_detail": "Deep intronic variants or novel biallelic private mutations — WGS + RNA-seq required if panel negative; diagnostic delay common; GALNS enzyme low but panel-negative",
        "hsct_eligible": False,
        "ert_alone": True,
    },
]

# Seizure types (MPS-IVA specific — structural/cord compression dominant)
SEIZURE_TYPES = [
    {"type": "GTCS", "pct": 55, "eeg": "Bisynchronous ictal; post-ictal slowing; cervical myelopathy → hypoxic-ischemic bilateral seizures"},
    {"type": "Focal aware", "pct": 25, "eeg": "Focal onset; focal cord or root compression; preserved awareness (normal intelligence)"},
    {"type": "Tonic", "pct": 15, "eeg": "Diffuse fast recruitment; brainstem compression from atlantoaxial dislocation"},
    {"type": "Myoclonic", "pct": 8, "eeg": "Polyspike-wave; rare; late disease; cord ischemia cumulative"},
    {"type": "Atypical absence", "pct": 5, "eeg": "Slow spike-wave; rare; intelligence preserved so atypical absences noticed and reported (unlike MPS-III)"},
]

# Seizure triggers (MPS-IVA specific — structural dominant)
TRIGGERS = [
    {"trigger": "Atlantoaxial instability / cervical cord compression (odontoid hypoplasia)", "pct": 62, "note": "DOMINANT trigger — odontoid hypoplasia WORST among all MPS types; cord signal on MRI = urgent surgical referral"},
    {"trigger": "Sleep apnea (thoracic cage restriction + tracheomalacia)", "pct": 48, "note": "Pectus carinatum restricts thoracic expansion; tracheomalacia worsens airway collapse; BiPAP often required not just CPAP"},
    {"trigger": "Intercurrent febrile illness", "pct": 55, "note": "Universal trigger; fever lowers cord ischemia threshold in AAI; cervical cord at risk if fever causes systemic hypotension"},
    {"trigger": "General anaesthesia / sedation", "pct": 35, "note": "EXTREME HAZARD — neck hyperextension during laryngoscopy risks C1/C2 dislocation → cord transection; video laryngoscopy + C-spine immobilization mandatory"},
    {"trigger": "Atlantoaxial subluxation (trauma/minor fall)", "pct": 28, "note": "Even minor trauma can cause atlantoaxial subluxation in MPS-IVA — odontoid hypoplasia leaves C1-C2 inherently unstable"},
    {"trigger": "Missed ERT dose (Elosulfase alfa gap)", "pct": 18, "note": "Somatic KS/C6S rebound; skeletal/respiratory burden worsens; indirect seizure threshold effect via OSA/hypoxia"},
    {"trigger": "Cervical MRI-confirmed myelopathy", "pct": 22, "note": "Cord signal change on MRI indicates cumulative ischemia; seizure risk elevated; surgical fusion indicated"},
]

# Treatments (AEDs + ERT + surgery + OSA management)
TREATMENTS = [
    {"name": "Levetiracetam (LEV)", "level": "Level B", "role": "First-line AED — no CYP induction (ERT pharmacokinetics preserved); no cardiac effect; IV formulation for SE; renal excretion", "ci": None},
    {"name": "Valproate (VPA)", "level": "Level B", "role": "Broad-spectrum; POLG1 exclusion MANDATORY (CPIC A); MPS-IVA liver has KS/C6S storage but less hepatic burden than MPS-I HS/DS — caution warranted; monitor LFTs; safe in normal-IQ MPS-IVA if POLG1-cleared", "ci": "POLG1 mutation (mandatory exclusion); hepatic dysfunction"},
    {"name": "Clobazam (CLB)", "level": "Level B", "role": "Adjunctive for focal + GTCS; sedation caution in OSA from thoracic cage restriction (respiratory depression; BiPAP-dependent patients)", "ci": "Severe OSA without non-invasive ventilation"},
    {"name": "Elosulfase alfa (Vimizim, ERT)", "level": "Level A (somatic)", "role": "FDA 2014; 2 mg/kg IV every OTHER week (not weekly); reduces urine KS/C6S; improves 6-minute walk test; DOES NOT cross BBB; pre-infusion anti-elosulfase antibody screen mandatory; continue peri-operatively (do NOT hold for surgery)", "ci": "Severe infusion reaction (switch to desensitization protocol)"},
    {"name": "Cervical decompression/fusion (surgical)", "level": "Level A (non-pharmacological)", "role": "PRIMARY seizure prevention in MPS-IVA — surgical treatment of dominant trigger (atlantoaxial instability); most important intervention; indicated when cord signal on MRI or ADI >5mm; EEG mandatory to distinguish epileptiform seizures from cord compression events", "ci": "Anaesthesia risk must be mitigated: video laryngoscopy + C-spine immobilization pre-fusion mandatory"},
    {"name": "CPAP / BiPAP (non-invasive ventilation)", "level": "Level B", "role": "OSA management — thoracic cage restriction (pectus carinatum) limits CPAP response; BiPAP often required; non-invasive ventilation reduces hypoxic seizure threshold; sleep study mandatory for titration", "ci": None},
    {"name": "Melatonin", "level": "Level B", "role": "Circadian disruption + OSA; sleep study + BiPAP first; melatonin adjunct for circadian resync after OSA management", "ci": None},
]

# Contraindications (MPS-IVA specific)
CONTRAINDICATIONS = [
    {"drug": "Typical antipsychotics (haloperidol, chlorpromazine)", "risk": "HIGH RISK", "reason": "KS-laden basal ganglia → EPS/NMS risk (lower than MPS-II/III but real; behavioral issues uncommon in normal-IQ MPS-IVA but pain management drives prescribing; opioid-sparing approaches preferred)", "alternative": "Atypicals (risperidone, aripiprazole, quetiapine) with EPS monitoring; non-opioid analgesia for skeletal pain"},
    {"drug": "Phenytoin / Fosphenytoin", "risk": "AVOID", "reason": "Aortic regurgitation + cardiac conduction toxicity (KS/C6S valve infiltration in 30-40% of patients); CYP2C9/3A4 induction disrupts elosulfase alfa ERT pharmacokinetics; IV fosphenytoin → arrhythmia risk in aortic valve disease", "alternative": "IV Levetiracetam for status epilepticus"},
    {"drug": "Vigabatrin (VGB)", "risk": "RELATIVE CI", "reason": "Corneal clouding present in MPS-IVA (milder than MPS-I but slit-lamp monitoring required); Goldman visual field perimetry challenging in young MPS-IVA patients; VGB retinal toxicity additive to corneal KS storage", "alternative": "LEV or VPA (POLG1-cleared) for focal seizures; ACTH if infantile spasms present"},
    {"drug": "Carbamazepine / Oxcarbazepine", "risk": "CAUTION (not absolute CI)", "reason": "KS/C6S peripheral nerve axonal neuropathy risk; cardiac sodium-channel toxicity in aortic valve disease; HLA-B*15:02 mandatory in SE Asian ancestry before CBZ/OXC/LTG", "alternative": "LEV or VPA (POLG1-cleared); HLA-B*15:02 screen first in SE Asian"},
    {"drug": "General anaesthesia (any agent)", "risk": "EXTREME HAZARD (most severe among all MPS)", "reason": "Odontoid hypoplasia = WORST atlantoaxial instability among all MPS types; neck hyperextension during laryngoscopy → C1/C2 dislocation → cord transection → DEATH; tracheomalacia compounds airway hazard; aortic regurgitation → haemodynamic instability; elosulfase IgE risk intraoperatively", "alternative": "C-spine XR + MRI pre-GA mandatory; video laryngoscopy + C-spine immobilization; awake fibreoptic intubation in severe AAI; anaesthesia alert bracelet; joint neurosurgery-cardiac-anaesthesia pre-op planning"},
]

# Monitoring parameters (12 items)
MONITORING = [
    "Urine KS + C6S quantification: baseline + q3-6M on ERT (KS levels correlate with disease burden — key monitoring biomarker; HS/DS NORMAL confirms MPS-IVA diagnosis)",
    "Brain MRI: periventricular changes minimal (normal intelligence); brainstem/cord signal at foramen magnum — annual; urgent if new neurological signs",
    "C-spine XR (flexion/extension views) + MRI: atlantoaxial instability (ADI measurement) — every 6-12M AND pre-GA MANDATORY (ABSOLUTE REQUIREMENT for any procedure)",
    "Respiratory function tests: forced vital capacity (FVC) — q6-12M; thoracic cage restriction (pectus carinatum); FVC <60% → NIV referral",
    "Cardiac echo: aortic regurgitation + mitral valve (KS/C6S valve infiltration) — annual; gradient >40 mmHg → cardiac surgery consultation",
    "Slit-lamp: corneal clouding grade — q12M; visual field assessment (for VGB REMS consideration)",
    "Audiometry: sensorineural + conductive hearing loss (70-85% prevalence) — annual; hearing aids as needed",
    "GALNS leukocyte enzyme activity: diagnosis confirmation + carrier testing (co-measure GLB1 to distinguish MPS-IVB)",
    "Anti-elosulfase IgG/IgE: infusion reaction monitoring; pre-treatment baseline + q6M on ERT",
    "POLG1 genotyping: before VPA initiation (CPIC Level A mandatory)",
    "HLA-B*15:02: SE Asian ancestry before CBZ/OXC/LTG",
    "Growth chart + bone age: annual (short stature monitoring; ERT impact on growth velocity)",
]

# Clinical thresholds (8 items)
THRESHOLDS = [
    {"parameter": "Cervical cord compression (MRI signal or ADI)", "threshold": "Cord signal on MRI or atlanto-dens interval (ADI) >5 mm", "action": "Urgent cervical decompression/fusion referral; no neck manipulation; anaesthesia alert; seizure risk elevated — add AED if cord compression confirmed"},
    {"parameter": "ERT initiation", "threshold": "At diagnosis, any age", "action": "Start elosulfase alfa 2 mg/kg IV every other week; pre-infusion antibody screen; infusion reaction protocol ready; continue peri-operatively (do NOT hold for surgery)"},
    {"parameter": "OSA severity (AHI)", "threshold": "AHI ≥5 events/hr (any severity)", "action": "Non-invasive ventilation (CPAP or BiPAP); BiPAP often required for thoracic cage restriction; sleep study titration; NIV reduces hypoxic seizure threshold"},
    {"parameter": "FVC (forced vital capacity)", "threshold": "FVC <60% predicted", "action": "NIV (BiPAP) referral + respiratory team; anaesthesia risk extreme; thoracic cage restriction limits inspiratory reserve"},
    {"parameter": "Aortic gradient (cardiac echo)", "threshold": "Aortic gradient >40 mmHg", "action": "Cardiac surgery consultation (valve repair/replacement); minimize PHT/fosphenytoin exposure; pre-GA cardiac clearance mandatory"},
    {"parameter": "Urine KS on ERT", "threshold": "KS >30 μmol/mmol creatinine on ERT", "action": "Review compliance; assess for anti-elosulfase antibodies (IgG/IgE); dose review; compliance counseling (every OTHER week dosing — confirm schedule adherence)"},
    {"parameter": "DRE threshold (MPS-IVA)", "threshold": "Failure of ≥2 appropriate AEDs at adequate doses", "action": "Video-EEG mandatory to exclude cord compression events mimicking epileptiform seizures; cervical MRI; surgical fusion as primary intervention before AED escalation"},
    {"parameter": "Pre-anaesthesia (any MPS-IVA patient)", "threshold": "ANY MPS-IVA patient requiring GA or sedation", "action": "C-spine XR + MRI + cardiac echo + pulmonary function + anti-elosulfase IgE assessment + anaesthesia alert mandatory; video laryngoscopy + C-spine immobilization; awake fibreoptic intubation if severe AAI"},
]

# Lifecycle stages (5 stages)
LIFECYCLE = [
    {"stage": "Infantile/Toddler (0-2 yr)", "features": "Short stature emerging; hypermobility; recurrent otitis media (conductive hearing loss); no cognitive delay (NORMAL development — key distinguishing feature from Hurler/Hunter); kyphosis emerging; normal milestones for language and cognition", "action": "Urine KS/C6S confirmation; GALNS enzyme assay; ERT initiation; no HSCT (intelligence normal); C-spine surveillance begins"},
    {"stage": "Childhood (2-8 yr)", "features": "Skeletal progression accelerating; corneal clouding appears (late onset, mild); genu valgum; pectus carinatum evident; OSA onset (thoracic restriction + tracheomalacia); hearing aids often needed; school performance NORMAL", "action": "ERT continuation; OSA/NIV management; slit-lamp annual; audiometry; C-spine flexion/extension XR every 6-12M"},
    {"stage": "School age / Adolescent (8-15 yr)", "features": "Atlantoaxial instability most dangerous phase — spinal cord at greatest risk; hearing loss significant; first seizures from cord compression; respiratory restriction limits exercise; pectus carinatum and genu valgum max severity; normal IQ allows full school participation with accommodations", "action": "C-spine MRI surveillance intensified; cervical fusion if ADI >5mm or cord signal; EEG if seizures to distinguish cord events; AED if epileptiform seizures confirmed"},
    {"stage": "Young adult (15-30 yr)", "features": "Progressive skeletal and respiratory restriction; cardiac valve disease (aortic regurgitation in 30-40%); pain from joint disease; ambulatory decline (many require wheelchair by 20s-30s); intelligence preserved; hearing loss significant; ERT ongoing", "action": "Cardiac echo annual; respiratory function q6M; pain management (non-opioid preferred); vocational planning (normal cognition = employment possible with accommodations); ERT lifelong"},
    {"stage": "Adult/Late (30+ yr)", "features": "Wheelchair-dependent in many; significant cardiac/respiratory burden; seizure risk from cumulative cord ischemia; hearing aids; corneal clouding may limit vision; continued ERT; longevity reduced vs general population but survival to 4th-5th decade possible with treatment", "action": "Palliative/supportive care planning; cardiac valve management; respiratory support; ERT benefit continues (somatic); AED maintenance if seizures; anaesthesia EXTREME caution for any procedure"},
]

# Differential diagnosis (8 conditions)
DIFFERENTIAL_DIAGNOSIS = [
    {"condition": "MPS-IVB (GLB1 — beta-galactosidase / Morquio B)", "distinction": "Same skeletal Morquio phenotype (short stature, pectus, genu valgum); urine KS elevated (same fingerprint); corneal clouding present; normal intelligence (same as MPS-IVA); distinguish ONLY by leukocyte enzyme assay: GALNS deficient (IVA) vs GLB1 deficient (IVB); GLB1 also causes GM1-gangliosidosis (severe CNS) — different allele"},
    {"condition": "MPS-I (IDUA — Hurler/Hurler-Scheie/Scheie)", "distinction": "HS + DS BOTH elevated (NOT KS) — different GAG fingerprint; cognitive decline in Hurler and Hurler-Scheie (contrasts MPS-IVA where intelligence NORMAL); corneal clouding present in MPS-I too; HSCT GOLD STANDARD for Hurler (contrasts MPS-IVA where HSCT not indicated); IDUA enzyme deficient"},
    {"condition": "MPS-VI (ARSB — Maroteaux-Lamy)", "distinction": "DS ONLY elevated (not KS or HS); normal intelligence (similar to MPS-IVA — both spare CNS); corneal clouding present; skeletal pattern different (coarse facies more prominent in MPS-VI); ARSB (arylsulfatase B) enzyme deficient; Galsulfase ERT available"},
    {"condition": "MPS-VII (GUSB — Sly syndrome)", "distinction": "HS + DS + CS ALL elevated; intelligence variable; extremely rare; hydrops fetalis common; GUSB (beta-glucuronidase) enzyme deficient; epilepsy present; very different phenotype"},
    {"condition": "Spondyloepiphyseal Dysplasia (SED)", "distinction": "Skeletal phenotype similar to MPS-IVA (short stature, odontoid hypoplasia); NO GAG in urine (NOT a storage disorder); normal enzyme activity; COL2A1 mutations; no corneal clouding from GAG; no ERT available or needed"},
    {"condition": "Dyggve-Melchior-Clausen (DMC)", "distinction": "KS elevated in urine (similar to MPS-IVA); however INTELLECTUAL DISABILITY present (unlike MPS-IVA where intelligence NORMAL — key differentiator); DYM gene; platyspondyly + irregular epiphyses; immune system not involved; enzyme panel normal"},
    {"condition": "Pseudoachondroplasia", "distinction": "Short stature (similar to MPS-IVA); NO GAG in urine; no lysosomal enzyme deficiency; COMP gene mutation; normal face (unlike MPS-IVA pectus/genu valgum); no corneal clouding; no ERT; management orthopedic only"},
    {"condition": "POLG-related epilepsy (POLG1 mutations)", "distinction": "Mitochondrial NOT lysosomal; Alpers syndrome phenocopy; VPA ABSOLUTE CI (fatal hepatotoxicity); muscle biopsy ragged-red fibres; no GAG in urine; no enzyme deficiency; exclude before VPA in any pediatric epilepsy with liver involvement"},
]

# Definitions glossary (12 terms)
DEFINITIONS = [
    {"term": "GALNS", "definition": "N-acetylgalactosamine-6-sulfate sulfatase — lysosomal enzyme that cleaves sulfate from N-acetylgalactosamine-6-sulfate residues in keratan sulfate (KS) and chondroitin-6-sulfate (C6S). Deficiency causes MPS-IVA (Morquio A syndrome)."},
    {"term": "MPS-IVA (Morquio A)", "definition": "Mucopolysaccharidosis type IVA — autosomal recessive GALNS deficiency. Primary skeletal disease with KS + C6S accumulation. KEY FEATURE: NORMAL or near-normal intelligence (contrasts MPS-I/II/III). Epilepsy 10-15% from structural cord compression, NOT from CNS GAG accumulation."},
    {"term": "Elosulfase alfa (Vimizim)", "definition": "Recombinant human N-acetylgalactosamine-6-sulfate sulfatase. FDA-approved 2014. Dose: 2 mg/kg IV every OTHER week (contrast Laronidase/Idursulfase — weekly). Reduces urine KS and improves 6-minute walk test. DOES NOT cross BBB. Cervical spine decompression still required for AAI."},
    {"term": "Atlantoaxial Instability (MPS-IVA — most severe)", "definition": "Odontoid (dens) hypoplasia + KS-laden transverse ligament laxity → unstable C1-C2 articulation. MOST SEVERE among ALL MPS types in MPS-IVA. Neck hyperextension during anaesthesia → C1/C2 dislocation → cord transection → death. C-spine XR pre-GA ABSOLUTE MANDATORY."},
    {"term": "Odontoid Hypoplasia", "definition": "Underdevelopment of the odontoid process (dens) of C2 vertebra — pathognomonic finding in MPS-IVA. The dens normally stabilizes the atlantoaxial joint; hypoplasia leaves C1-C2 inherently unstable. Present universally in classic MPS-IVA. Severity correlates with seizure risk from cord compression."},
    {"term": "KS (Keratan Sulfate)", "definition": "Primary lysosomal substrate in MPS-IVA. Elevated in urine (and serum) — MPS-IVA biomarker. KS accumulates in cartilage, bone, cornea. Urine KS levels correlate with disease burden and ERT response. Key diagnostic fingerprint: KS elevated but HS/DS NORMAL or mildly elevated (distinguishes MPS-IVA from MPS-I/II/VI)."},
    {"term": "C6S (Chondroitin-6-Sulfate)", "definition": "Secondary lysosomal substrate in MPS-IVA. Co-elevated with KS in urine. C6S accumulates in cartilage and bone matrix. Contributes to skeletal pathology alongside KS. C6S elevation (with KS) + HS/DS normal = MPS-IVA fingerprint."},
    {"term": "Tracheomalacia", "definition": "Cartilaginous softening of the trachea due to KS/C6S accumulation in tracheal cartilage rings → dynamic airway collapse on expiration. Contributes to OSA in MPS-IVA (alongside thoracic cage restriction from pectus carinatum). Worsens anaesthesia airway management risk."},
    {"term": "DRE (Drug-Resistant Epilepsy)", "definition": "Failure of ≥2 appropriate AEDs at adequate doses. Uncommon in MPS-IVA (5-8%) because seizures are structural (cord compression), not epileptiform cortical. Video-EEG mandatory to exclude cord compression events mimicking seizures before DRE label. Cervical fusion is primary 'anti-seizure' intervention."},
    {"term": "Normal Intelligence (MPS-IVA key feature)", "definition": "MPS-IVA patients have NORMAL or near-normal intelligence — the KEY distinguishing feature from MPS-I/II/III. KS/C6S do NOT accumulate in neurons at disease-causing levels. Seizures are NOT from GAG-driven cortical hyperexcitability but from mechanical cord compression (AAI). Children attend school with normal cognitive function."},
    {"term": "Cord Myelopathy", "definition": "Spinal cord injury/ischemia from chronic or acute cervical cord compression due to atlantoaxial instability (odontoid hypoplasia). Manifest as T2 signal change on cervical MRI. Primary cause of GTCS and tonic seizures in MPS-IVA. Surgical decompression/fusion = primary seizure prevention, more effective than AED escalation alone."},
    {"term": "AAI-surgical-threshold", "definition": "Atlantoaxial interval (ADI) >5mm on flexion/extension XR OR any T2 cord signal change on MRI in MPS-IVA patient → indication for urgent cervical decompression/fusion referral. This threshold triggers surgical intervention which is the primary seizure prevention strategy in MPS-IVA."},
]

KEY_CONCEPTS = [
    "Normal intelligence: KEY distinguishing feature of MPS-IVA from MPS-I/II/III — cognitive decline ABSENT; KS/C6S do NOT drive neuronal damage; seizures NOT from GAG-driven cortical hyperexcitability but from structural cord compression (atlantoaxial instability → myelopathy)",
    "Atlantoaxial instability: MOST SEVERE among all MPS types — odontoid (dens) hypoplasia + transverse ligament KS deposition → life-threatening cord compression with any neck hyperextension; C-spine XR pre-GA ABSOLUTE MANDATORY; ADI >5mm or cord signal = urgent surgical fusion",
    "Anesthesia EXTREME HAZARD (most severe among all MPS): neck hyperextension during laryngoscopy → C1/C2 dislocation → cord transection → DEATH; tracheomalacia compounds airway difficulty; video laryngoscopy + C-spine immobilization + awake fibreoptic intubation in severe AAI; elosulfase IgE risk intraoperatively",
    "KS + C6S in urine (NOT heparan sulfate or dermatan sulfate): MPS-IVA fingerprint — distinguishes from MPS-I/II (HS+DS both elevated), MPS-III (HS only), MPS-VI (DS only); HS/DS NORMAL or mildly elevated in MPS-IVA is the KEY diagnostic distinction",
    "ERT (Elosulfase alfa/Vimizim): every OTHER week (not weekly like Laronidase/Idursulfase); 2 mg/kg IV; FDA 2014; reduces urine KS; improves walk test; DOES NOT cross BBB; cervical spine decompression still independently required; continue peri-operatively (do NOT hold for surgery)",
    "No HSCT in MPS-IVA: cognitive preservation is NORMAL → HSCT offers NO cognitive benefit; experimental only; contrasts Hurler (MPS-IH) where HSCT is gold standard; HSCT risk-benefit unfavorable in MPS-IVA",
    "Primary seizure cause: mechanical/structural (cord compression from AAI) NOT metabolic CNS GAG accumulation — SURGERY (cervical fusion) is primary seizure prevention; EEG mandatory to distinguish epileptiform seizures from cord compression events; AED escalation alone is insufficient",
    "PHT AVOID: aortic regurgitation + cardiomyopathy (KS/C6S valve disease, 30-40%) + CYP induction disrupts ERT (elosulfase alfa pharmacokinetics); use IV LEV for SE",
    "VGB RELATIVE CI: corneal clouding present in MPS-IVA (milder than MPS-I but slit-lamp monitoring required; Goldman visual field perimetry challenging in young patients and impossible in severe disease)",
    "CBZ/OXC CAUTION: KS/C6S axonal neuropathy (peripheral nerve infiltration); HLA-B*15:02 mandatory SE Asian ancestry before CBZ/OXC/LTG; cardiac sodium-channel risk in aortic valve disease",
    "POLG1 exclusion mandatory before VPA: severe MPS-IVA epilepsy can phenocopy mitochondrial disease; VPA hepatotoxicity fatal in POLG1; CPIC Level A guideline",
    "Hearing loss 70-85%: conductive + sensorineural; hearing aids required; impacts quality of life and AED compliance counseling; conductive from Eustachian tube KS deposition, sensorineural from cochlear KS",
    "AR inheritance: both sexes equally affected; parental consanguinity increases risk (Saudi, Middle Eastern, South Asian populations — p.I113F Saudi founder); consanguineous pedigrees have higher frequency of severe null/null genotypes",
    "Pectus carinatum: universal skeletal marker in MPS-IVA; restricts thoracic expansion → FVC reduction → hypoxia → lowered seizure threshold; contributes to OSA; BiPAP often required",
    "MPS-IVB distinction: GLB1 gene (beta-galactosidase) causes MPS-IVB (Morquio B — same skeletal phenotype, same urine KS) AND GM1-gangliosidosis (different alleles, severe CNS); distinguish MPS-IVA vs IVB ONLY by leukocyte enzyme panel (GALNS vs GLB1 activity); treatment approach differs",
]

STANDARDS = [
    "Hendriksz et al. 2014 — International guidelines for management of MPS-IVA (J Inherit Metab Dis)",
    "Harmatz et al. 2014 — Elosulfase alfa Phase III trial ADVANCE: MPS-IVA 6-minute walk test outcomes (NEJM)",
    "Chiaro et al. 2021 — Anaesthetic management MPS-IVA: cervical spine considerations and airway protocol",
    "Valayannopoulos et al. 2010 — MPS-IVA natural history and skeletal disease burden (Orphanet J Rare Dis)",
    "CPIC 2023 — POLG1 genotyping Level A guideline before valproate initiation",
    "REMS VGB (Vigabatrin) — SHARE program: visual field perimetry q3M mandatory; corneal clouding in MPS-IVA requires slit-lamp documentation",
    "Dung et al. 2013 — Atlantoaxial instability in MPS-IVA: surgical indications and outcomes",
    "ICH E6(R2) GCP — Lysosomal Storage Disease clinical trial standards",
]


def get_overview():
    """Return GALNS/MPS-IVA overview for /api/galns/overview."""
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "disease_mechanism": DISEASE_MECHANISM,
        "cohort_size": COHORT_SIZE,
        "epilepsy_prevalence_pct": {"overall": "10-15", "classic_severe": "12-15", "attenuated": "8-10"},
        "drug_resistance_pct": {"overall": "5-8"},
        "osa_pct": 50,
        "atlantoaxial_instability_pct": 85,
        "corneal_clouding_pct": 70,
        "cardiac_regurgitation_pct": 35,
        "hearing_loss_pct": 78,
        "on_ert_pct": 82,
        "on_hsct_pct": 0,
        "cervical_fusion_pct": 45,
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "key_concepts": KEY_CONCEPTS[:6],
        "standards": STANDARDS,
    }


def get_breakdown():
    """Return GALNS/MPS-IVA 40-patient cohort for /api/galns/breakdown."""
    rng = random.Random(42)

    phenotype_weights = {
        "Classic/Severe": 0.35,
        "Attenuated": 0.25,
        "Saudi-Founder": 0.20,
        "Intermediate": 0.12,
        "Rare-Private": 0.08,
    }
    phenotype_seizure_rates = {
        "Classic/Severe": 0.13,
        "Attenuated": 0.09,
        "Saudi-Founder": 0.11,
        "Intermediate": 0.12,
        "Rare-Private": 0.15,
    }
    phenotype_dre_rates = {
        "Classic/Severe": 0.06,
        "Attenuated": 0.04,
        "Saudi-Founder": 0.05,
        "Intermediate": 0.06,
        "Rare-Private": 0.08,
    }

    etiology_by_phenotype = {
        "Classic/Severe": "Classic/Severe (Null/Null — biallelic truncating)",
        "Attenuated": "Attenuated (Biallelic missense — p.R386C compound or homozygous)",
        "Saudi-Founder": "Saudi Founder (p.I113F biallelic — consanguineous Middle East)",
        "Intermediate": "Intermediate (Null + missense — compound-het)",
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
    atlantoaxial_n = 0
    cervical_fusion_n = 0

    for i in range(COHORT_SIZE):
        pid = i + 1
        pheno = phenotype_pool[i]
        etiol = etiology_by_phenotype[pheno]
        has_seizures = rng.random() < phenotype_seizure_rates[pheno]
        has_dre = has_seizures and rng.random() < phenotype_dre_rates[pheno]
        osa = rng.random() < 0.50
        atlantoaxial = rng.random() < 0.85
        cervical_fusion = atlantoaxial and rng.random() < 0.45
        on_ert = rng.random() < 0.82
        hearing_loss = rng.random() < 0.78
        cardiac_regurgitation = rng.random() < 0.35
        corneal_clouding = rng.random() < 0.70

        if has_dre:
            drug_resistant_n += 1
        if osa:
            osa_n += 1
        if on_ert:
            on_ert_n += 1
        if atlantoaxial:
            atlantoaxial_n += 1
        if cervical_fusion:
            cervical_fusion_n += 1

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

        age_onset = rng.randint(8, 18) if has_seizures else None

        patients.append({
            "patient_id": f"GALNS-{pid:02d}",
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
            "post_hsct": False,
            "atlantoaxial_instability": atlantoaxial,
            "corneal_clouding": corneal_clouding,
            "cardiac_regurgitation": cardiac_regurgitation,
            "hearing_loss": hearing_loss,
            "cervical_fusion": cervical_fusion,
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
        "on_hsct_n": 0,
        "on_hsct_pct": 0.0,
        "atlantoaxial_n": atlantoaxial_n,
        "atlantoaxial_pct": round(atlantoaxial_n / COHORT_SIZE * 100, 1),
        "cervical_fusion_n": cervical_fusion_n,
        "cervical_fusion_pct": round(cervical_fusion_n / COHORT_SIZE * 100, 1),
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
    """Return GALNS/MPS-IVA definitions for /api/galns/definitions."""
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "definitions": DEFINITIONS,
        "key_concepts": KEY_CONCEPTS,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "standards": STANDARDS,
        "diagnostic_algorithm": [
            "Step 1: Urine GAG quantification — KS + C6S elevated; HS/DS NORMAL or mildly elevated (MPS-IVA fingerprint distinguishes from MPS-I/II where HS+DS both elevated, and MPS-III where HS only, and MPS-VI where DS only)",
            "Step 2: Leukocyte GALNS enzyme activity — <5% control confirms MPS-IVA; co-measure GLB1 (beta-galactosidase) to distinguish MPS-IVB (same KS elevation, same skeletal phenotype — different enzyme deficiency)",
            "Step 3: GALNS gene sequencing — biallelic variants (AR); targeted panel first (p.R386C European founder, p.I113F Saudi founder, p.T291N Middle Eastern, p.G340D); if heterozygous or negative → full gene sequencing + MLPA; no GALNS pseudogene (unlike IDS/IDSP1)",
            "Step 4: Phenotype assessment — skeletal score (short stature + pectus carinatum + genu valgum + odontoid hypoplasia) + cognitive assessment (NORMAL intelligence confirms MPS-IVA; intellectual disability suggests MPS-IVB/DMC/MPS-I/II/III)",
            "Step 5: C-spine XR flexion/extension + MRI cervical — odontoid hypoplasia + atlantoaxial interval (ADI) measurement; immediate surgical referral if ADI >5mm or cord signal on MRI; MANDATORY at diagnosis and every 6-12M",
            "Step 6: Cardiac echo — aortic regurgitation + mitral valve (KS/C6S valve infiltration) — baseline at diagnosis; annual surveillance",
            "Step 7: Pulmonary function tests — FVC + flow-volume loop — baseline + annual; thoracic cage restriction (pectus carinatum); FVC <60% → NIV referral; polysomnography for OSA assessment",
            "Step 8: Ophthalmology — slit-lamp corneal clouding grade + visual acuity (milder than MPS-I but present; required for VGB REMS compliance assessment)",
            "Step 9: POLG1 genotyping — before VPA initiation (CPIC Level A mandatory); HLA-B*15:02 screening in SE Asian ancestry before CBZ/OXC/LTG",
            "Step 10: Multidisciplinary workup — orthopedics (cervical fusion timing + genu valgum osteotomy) + cardiac + respiratory + ENT (hearing aids) + ophthalmology + neurology (AED only if epileptiform seizures confirmed on EEG; EEG mandatory to distinguish cord compression events from epileptiform discharges)",
        ],
        "pharmacological_distinctions": [
            "LEV FIRST-LINE: no CYP induction (elosulfase alfa ERT pharmacokinetics preserved); no cardiac effect (safe in aortic regurgitation); IV formulation for SE; renal excretion — adjust for renal involvement",
            "VPA SECOND-LINE: POLG1 exclusion MANDATORY (CPIC A); MPS-IVA liver has KS/C6S storage but less hepatic burden than MPS-I HS/DS — caution warranted; monitor LFTs; effective in normal-IQ MPS-IVA if POLG1-cleared",
            "PHT/Fosphenytoin AVOID: aortic regurgitation + cardiac conduction toxicity (KS/C6S valve disease 30-40%); CYP2C9/3A4 induction disrupts elosulfase alfa ERT; IV fosphenytoin → arrhythmia in aortic valve disease; use IV LEV instead",
            "Typical antipsychotics HIGH RISK: KS-laden basal ganglia → EPS/NMS risk; behavioral issues uncommon in normal-IQ MPS-IVA but skeletal pain management drives opioid prescribing (consider non-opioid); use atypicals with EPS monitoring",
            "VGB RELATIVE CI: corneal clouding present in MPS-IVA (milder than MPS-I) → Goldman visual field perimetry still challenging; slit-lamp documentation required for REMS compliance; ACTH preferred if infantile spasms present",
            "CBZ/OXC CAUTION (not absolute CI): KS/C6S peripheral nerve axonal neuropathy; cardiac sodium-channel risk in aortic valve disease; HLA-B*15:02 mandatory in SE Asian ancestry before CBZ/OXC/LTG",
            "Anesthesia EXTREME HAZARD (most severe among all MPS): odontoid hypoplasia = WORST AAI among all MPS types; tracheomalacia; RSI + video laryngoscopy + C-spine immobilization mandatory; awake fibreoptic intubation in severe AAI; neck hyperextension MUST BE AVOIDED ABSOLUTELY; elosulfase IgE risk",
            "CPAP/BiPAP for OSA = seizure management: thoracic cage restriction (pectus carinatum) + tracheomalacia → OSA; BiPAP often required (not just CPAP); non-invasive ventilation reduces hypoxic seizure threshold; prescribe NIV before AED escalation for OSA-triggered seizures",
            "Elosulfase alfa (Vimizim) every OTHER week: 2 mg/kg IV q2 weeks — contrast Laronidase (0.58 mg/kg weekly) and Idursulfase (0.5 mg/kg weekly); pre-infusion antibody screen; anaphylaxis risk (inform anaesthesia team); continue peri-operatively — do NOT hold for surgery",
            "Surgery = primary seizure prevention: cervical decompression/fusion is the most important anti-seizure intervention in MPS-IVA — NOT AED escalation; EEG mandatory to distinguish cord compression events from epileptiform seizures before labeling as DRE",
            "Melatonin Level B: OSA + circadian disruption; sleep study first; BiPAP before melatonin for thoracic OSA; melatonin adjunct for circadian resync",
            "ERT-surgery interaction: continue elosulfase alfa peri-operatively (do NOT hold for surgery in MPS-IVA); elosulfase reduces somatic KS/C6S burden but cervical decompression is independently indicated and not replaced by ERT; alert anaesthesia team to elosulfase IgE risk for intraoperative anaphylaxis",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== GALNS OVERVIEW ===")
    ov = get_overview()
    print(f"Gene: {ov['gene']} | Locus: {ov['locus']} | OMIM: {ov['omim']} | Cohort: {ov['cohort_size']}")
    print(f"Epilepsy overall: {ov['epilepsy_prevalence_pct']['overall']}% | DRE: {ov['drug_resistance_pct']['overall']}%")
    print(f"OSA: {ov['osa_pct']}% | AAI: {ov['atlantoaxial_instability_pct']}% | Cervical fusion: {ov['cervical_fusion_pct']}%")
    print(f"ERT: {ov['on_ert_pct']}% | HSCT: {ov['on_hsct_pct']}% (none — intelligence normal)")
    print("\n=== GALNS BREAKDOWN ===")
    bk = get_breakdown()
    print(f"Seizures: {bk['seizure_n']}/{COHORT_SIZE} ({bk['seizure_pct']}%)")
    print(f"DRE: {bk['drug_resistant_n']} ({bk['drug_resistant_pct']}%)")
    print(f"OSA: {bk['osa_n']} ({bk['osa_pct']}%) | ERT: {bk['on_ert_n']} ({bk['on_ert_pct']}%)")
    print(f"HSCT: {bk['on_hsct_n']} ({bk['on_hsct_pct']}%) | AAI: {bk['atlantoaxial_n']} ({bk['atlantoaxial_pct']}%)")
    print(f"Cervical fusion: {bk['cervical_fusion_n']} ({bk['cervical_fusion_pct']}%)")
    print("\n=== GALNS DEFINITIONS ===")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])} terms | Key concepts: {len(df['key_concepts'])}")
    print(f"Diagnostic algorithm: {len(df['diagnostic_algorithm'])} steps")
    print(f"Pharmacological distinctions: {len(df['pharmacological_distinctions'])} points")
