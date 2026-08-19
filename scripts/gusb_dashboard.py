#!/usr/bin/env python3
"""GUSB / MPS-VII (Sly Syndrome) Epilepsy Dashboard — seed data module.

MPS-VII: Beta-glucuronidase (GUSB) deficiency (GUSB, 7q11.21, AR).
GAG fingerprint: HS + DS + CS ALL three elevated simultaneously — UNIQUE among all MPS types.
(MPS-I/II: HS+DS; MPS-III: HS only; MPS-IVA: KS+C6S; MPS-VI: DS+C4S — none combine all three)
KEY DISTINGUISHING FEATURE: NON-IMMUNE HYDROPS FETALIS — only MPS type with hydrops as dominant
neonatal presentation (30-40% of severe neonatal form); absent in MPS I-VI, distinguishes from
all other MPS. ERT: Vestronidase alfa (Mepsevii, FDA 2017) — 4 mg/kg IV every 2 weeks.
Three clinical forms: Neonatal/severe (hydrops, early death), Juvenile/intermediate (moderate MR),
Attenuated (normal to near-normal intelligence, later onset, longer survival).
Intelligence: VARIABLE across spectrum (unlike MPS-IVA/VI where NORMAL IQ is pathognomonic key).
Epilepsy 30-50%: more frequent in severe/intermediate forms; structural (hydrocephalus, cord,
OSA-hypoxia) + cortical GAG storage (HS + DS CNS accumulation).
Corneal clouding: PRESENT (unlike MPS-II Hunter — distinguishes GUSB from IDS at bedside).
Vestronidase alfa does NOT cross BBB — somatic only; no cognitive benefit.
HSCT data limited (disease too rare — ~300 cases worldwide); considered for severe < 3 yr.
Founder mutation: p.L176F (Belgian founder, European enrichment).
"""
import random

GENE = "GUSB"
LOCUS = "7q11.21"
OMIM = "253220"
INHERITANCE = "Autosomal Recessive (AR) — biallelic GUSB LOF; both males AND females equally affected"
COHORT_SIZE = 40
DISEASE_MECHANISM = (
    "Beta-glucuronidase (GUSB) deficiency → lysosomal accumulation of heparan sulfate (HS), dermatan "
    "sulfate (DS), AND chondroitin sulfate (CS) simultaneously — all three GAG classes accumulated "
    "together is PATHOGNOMONIC and UNIQUE to MPS-VII (contrasts all other MPS types). HS+DS+CS triple "
    "elevation in urine/plasma is the diagnostic GAG fingerprint. HS drives CNS neuronal damage "
    "(cognitive decline, seizures — same mechanism as MPS-I/II/III); DS drives somatic/cardiac/airway "
    "disease (same as MPS-I/II/VI); CS adds chondral/skeletal burden (similar to MPS-VI DS+C4S). "
    "Hydrops fetalis occurs in neonatal-severe form (30-40%) — intrauterine GAG accumulation in all "
    "fetal tissues (ascites, pleural/pericardial effusions, skin edema) — the ONLY MPS type with "
    "non-immune hydrops as a primary presentation. Vestronidase alfa ERT (FDA 2017) reduces somatic "
    "GAG burden but DOES NOT cross the BBB and does NOT improve cognitive outcomes. HSCT considered "
    "for severe early-onset (< 3 yr) — data limited due to extreme rarity (~300 cases worldwide). "
    "Three clinical spectra: Neonatal/severe (hydrops, death by age 2-3), Juvenile/intermediate "
    "(moderate-severe MR, survival to teens-adulthood with intervention), Attenuated (normal to "
    "near-normal intelligence, adult survival, diagnosis often delayed)."
)

# 5 variant classes (etiologies) — deterministic percentages summing to 100
ETIOLOGIES = [
    {
        "name": "Neonatal/Severe (Null/Null — biallelic truncating)",
        "pct": 30,
        "n": 12,
        "seizure_risk": "45-55% (HS cortical storage + communicating hydrocephalus + severe OSA)",
        "eeg": "Diffuse slowing; hypsarrhythmia in early infantile spasms; multifocal discharges from "
               "cortical HS accumulation; burst-suppression in severe neonatal encephalopathy; "
               "epileptic spasms in infancy",
        "variant_detail": (
            "Biallelic nonsense/frameshift — null alleles; enzyme absent (<1% control); severe phenotype: "
            "non-immune hydrops fetalis (30-40% of this group — ascites, pleural effusion, skin edema); "
            "early respiratory failure; severe intellectual disability; survival often < 2-3 years without "
            "intensive management; HSCT considered if donor available pre-symptomatically"
        ),
        "hsct_eligible": True,
        "ert_alone": True,
    },
    {
        "name": "Belgian Founder — p.L176F Biallelic",
        "pct": 20,
        "n": 8,
        "seizure_risk": "35-40% (intermediate severity; cortical HS + hydrocephalus; OSA moderate)",
        "eeg": "Focal or multifocal discharges; moderate background slowing; cortical HS accumulation "
               "intermediate; periventricular white matter T2 changes on MRI; EEG findings intermediate "
               "between severe null/null and attenuated forms",
        "variant_detail": (
            "p.L176F biallelic (Belgian European founder; ~15-20% of known MPS-VII alleles in European "
            "cohorts); intermediate-to-severe phenotype; enzyme severely reduced; coarse facies, "
            "hepatosplenomegaly, bone dysplasia; corneal clouding PRESENT (vs MPS-II absent); "
            "moderate intellectual disability; survival into 2nd-3rd decade with ERT; ERT + OSA management"
        ),
        "hsct_eligible": True,
        "ert_alone": True,
    },
    {
        "name": "Juvenile/Intermediate (Compound-het Null + Missense)",
        "pct": 25,
        "n": 10,
        "seizure_risk": "30-38% (intermediate cortical HS + hydrocephalus; OSA + AAI contributing)",
        "eeg": "Focal temporal-parietal discharges; moderate diffuse slowing; cortical HS accumulation "
               "intermediate; periodic lateralized discharges if hydrocephalus; sleep EEG: OSA-related "
               "arousal pattern; clinical seizures mixed focal and generalized",
        "variant_detail": (
            "Null allele (truncating) + missense compound-het — intermediate enzyme 2-8% control; "
            "juvenile onset (2-8 years); moderate to severe intellectual disability; coarse facies, "
            "hepatosplenomegaly, corneal clouding; survival into adulthood with intervention; "
            "vestronidase alfa ERT + OSA/airway management; HSCT considered before age 3 if severe"
        ),
        "hsct_eligible": True,
        "ert_alone": True,
    },
    {
        "name": "Attenuated (Biallelic Missense — residual enzyme 5-15%)",
        "pct": 15,
        "n": 6,
        "seizure_risk": "15-20% (milder cortical involvement; structural dominant: hydrocephalus + OSA)",
        "eeg": "Near-normal background or mild slowing; focal discharges if hydrocephalus; seizures often "
               "GTCS at febrile illness or sleep; EEG may be normal interictally in attenuated form; "
               "MRI white matter changes mild",
        "variant_detail": (
            "Biallelic missense — residual enzyme 5-15% control; attenuated phenotype: onset school age "
            "to adult; intellectual disability mild to normal; normal intelligence possible; shorter stature, "
            "hepatosplenomegaly, bone changes; diagnosis often delayed (mild phenotype confounds); "
            "vestronidase alfa ERT primary; adult survival; reproductive counseling needed; "
            "corneal clouding may be subtle (slit-lamp required)"
        ),
        "hsct_eligible": False,
        "ert_alone": True,
    },
    {
        "name": "Rare/Private (Deep Intronic or Novel Biallelic)",
        "pct": 10,
        "n": 4,
        "seizure_risk": "25-30% (diagnostic delay → unmanaged OSA/hydrocephalus; variable spectrum)",
        "eeg": "Variable — may be normal or show multifocal discharges depending on spectrum; WGS + RNA-seq "
               "required; GUSB enzyme assay low in all private variants; HS+DS+CS triple elevation "
               "confirms MPS-VII regardless of molecular result; EEG mandatory to classify seizure type",
        "variant_detail": (
            "Deep intronic splicing variants or novel biallelic private mutations; WGS + RNA-seq required "
            "if panel-negative with clinical/biochemical MPS-VII; urine HS+DS+CS triple elevation "
            "PATHOGNOMONIC; GUSB enzyme low in leukocytes/fibroblasts confirms diagnosis; variable "
            "phenotype based on residual enzyme; ERT initiated after confirmation; GUSB gene panel "
            "expanded to include deep intronic and regulatory variants"
        ),
        "hsct_eligible": False,
        "ert_alone": True,
    },
]

# Seizure types (MPS-VII specific)
SEIZURE_TYPES = [
    {
        "type": "GTCS",
        "pct": 50,
        "eeg": (
            "Bisynchronous ictal; post-ictal diffuse slowing; HS + DS cortical accumulation (both drive "
            "GTCS pattern — HS via cortical neuronal damage, DS via structural hydrocephalus/OSA-hypoxia); "
            "high seizure threshold lowering with febrile illness in all three GAG-loaded tissue types"
        ),
    },
    {
        "type": "Epileptic spasms (infantile)",
        "pct": 20,
        "eeg": (
            "Hypsarrhythmia in severe neonatal/infantile form; modified hypsarrhythmia (HS cortical "
            "storage); ACTH Level A treatment; VGB relative CI (corneal clouding present in GUSB — "
            "monitor slit-lamp + VF; VGB not absolute CI but requires careful risk-benefit given corneal); "
            "spasms in clusters; AIMDS classification: structural etiology"
        ),
    },
    {
        "type": "Focal aware / impaired awareness",
        "pct": 15,
        "eeg": (
            "Focal temporal-parietal onset; preserved or impaired awareness depending on cortical HS "
            "load; focal cortical irritation from HS accumulation + communicating hydrocephalus raised "
            "ICP; GTCS secondarily; LEV first-line for focal; LTG adjunct (no CYP induction)"
        ),
    },
    {
        "type": "Tonic",
        "pct": 10,
        "eeg": (
            "Diffuse fast EEG recruitment; cord compression from AAI + communicating hydrocephalus ICP "
            "elevation; tonic posturing in neonatal/severe form; brainstem involvement in severe variants; "
            "may mimic decorticate posturing in neonatal encephalopathy — EEG mandatory to distinguish"
        ),
    },
    {
        "type": "Myoclonic",
        "pct": 5,
        "eeg": (
            "Polyspike-wave; late disease or progressive myoclonus epilepsy pattern in surviving "
            "severe forms; cumulative cortical HS accumulation; distinguish from cortical myoclonus "
            "in progressive neurodegeneration; PME pattern rare but documented in attenuated adult MPS-VII"
        ),
    },
]

# Seizure triggers (MPS-VII specific)
TRIGGERS = [
    {
        "trigger": "Febrile illness / intercurrent infection",
        "pct": 62,
        "note": (
            "Universal trigger — HS+DS+CS triple GAG elevation across all tissues means fever compounds "
            "every mechanism simultaneously (OSA mucosa edema → worse hypoxia; immune activation → "
            "neuroinflammation; fever threshold lowering); most common trigger across all three clinical "
            "forms; aggressive antipyretics mandatory; seizure action plan at febrile onset essential"
        ),
    },
    {
        "trigger": "Communicating hydrocephalus (HS+DS arachnoid villi infiltration)",
        "pct": 45,
        "note": (
            "HS + DS accumulation in arachnoid villi → impaired CSF reabsorption → communicating "
            "hydrocephalus → raised ICP → seizures; VP shunt indicated if ICP confirmed; more "
            "prevalent in severe/intermediate forms (HS cortical accumulation); brain MRI periventricular "
            "T2 changes; LP opening pressure mandatory in any MPS-VII with headache or new seizures; "
            "ETV less effective in communicating pattern — CSF diversion preferred"
        ),
    },
    {
        "trigger": "OSA / airway obstruction (HS+DS+CS tongue + airway accumulation)",
        "pct": 50,
        "note": (
            "All three GAGs accumulate in airway soft tissue → OSA (tongue enlargement, tonsil/adenoid "
            "infiltration, laryngeal narrowing); HS+DS+CS triple load means airway compromise can be "
            "severe; sleep study mandatory at diagnosis and annually; BiPAP titration; tonsillectomy/"
            "adenoidectomy in pediatric cases; OSA-driven hypoxia lowers seizure threshold significantly "
            "in all three forms"
        ),
    },
    {
        "trigger": "Atlantoaxial instability / cord compression",
        "pct": 35,
        "note": (
            "Odontoid hypoplasia from CS+DS skeletal storage → AAI; less systematically studied than "
            "MPS-IVA/I due to extreme rarity but present in intermediate/severe forms; C-spine XR "
            "flexion/extension + MRI mandatory pre-GA; cord signal on MRI → surgical fusion; AAI "
            "may be underrecognized in MPS-VII given rarity — maintain same pre-operative vigilance "
            "as MPS-IVA/I/II"
        ),
    },
    {
        "trigger": "General anaesthesia / sedation",
        "pct": 38,
        "note": (
            "EXTREME HAZARD — HS+DS+CS airway compromise + AAI risk + cardiac involvement; airway "
            "obstruction from tongue/larynx GAG infiltration; atlantoaxial instability from odontoid "
            "hypoplasia; video laryngoscopy + awake fibreoptic intubation mandatory; C-spine XR + "
            "cardiac echo pre-GA; anaesthesia alert bracelet; ENT + neuroanesthesia joint pre-op planning; "
            "vestronidase alfa IgE risk intraoperatively (pre-screen mandatory)"
        ),
    },
    {
        "trigger": "Missed vestronidase alfa (ERT) doses",
        "pct": 20,
        "note": (
            "HS+DS+CS somatic rebound on ERT interruption; airway, cardiac, and skeletal burden increase; "
            "OSA worsens → hypoxic seizure threshold lowering; infusion-site reaction / anti-vestronidase "
            "IgG/IgE monitoring mandatory; ERT every 2 weeks (vs galsulfase weekly — important compliance "
            "distinction for nursing/caregiver education)"
        ),
    },
    {
        "trigger": "Intracranial hypertension / VP shunt malfunction",
        "pct": 28,
        "note": (
            "Communicating hydrocephalus common in severe/intermediate forms; VP shunt placed → shunt "
            "malfunction risk (GAG viscosity may increase shunt obstruction rate); headache + vomiting "
            "+ altered consciousness → urgent shunt series + CT; status epilepticus from acute ICP rise; "
            "EEG during shunt malfunction critical to distinguish seizures from encephalopathic slowing"
        ),
    },
]

# Treatments
TREATMENTS = [
    {
        "name": "Levetiracetam (LEV)",
        "level": "Level B",
        "role": (
            "First-line AED — no CYP induction (vestronidase alfa ERT pharmacokinetics preserved); "
            "no cardiac effect; IV formulation for SE; renal excretion (adjust for renal involvement); "
            "safe in all MPS-VII forms including neonatal-severe; behavioral side effects (agitation, "
            "irritability) possible in severe intellectual disability — brivaracetam alternative"
        ),
        "ci": None,
    },
    {
        "name": "Valproate (VPA)",
        "level": "Level B",
        "role": (
            "Broad-spectrum; POLG1 exclusion MANDATORY (CPIC A — same requirement as all LSD/MPS); "
            "hepatic HS+DS+CS accumulation in MPS-VII → LFT monitoring more stringent than MPS-IV/VI; "
            "effective for GTCS and myoclonic; broad-spectrum covers infantile spasms adjunct; "
            "POLG1 screen mandatory BEFORE first VPA dose"
        ),
        "ci": "POLG1 mutation (CPIC A mandatory exclusion); hepatic dysfunction; elevated LFTs; "
              "mitochondrial disorder overlap (HS+DS+CS mitochondrial stress in severe form)",
    },
    {
        "name": "ACTH / Prednisolone",
        "level": "Level A (infantile spasms)",
        "role": (
            "Gold standard for infantile spasms (hypsarrhythmia in severe MPS-VII neonatal form); "
            "UKISS/IS-WEST protocol; VGB relative CI given corneal clouding (not absolute — risk-benefit "
            "assessment required with slit-lamp); ACTH preferred over VGB when corneal clouding confirmed; "
            "short course corticosteroids acceptable in MPS-VII without evidence of immunosuppression "
            "contraindication"
        ),
        "ci": "Active infection; hyperglycemia; severe immunocompromise",
    },
    {
        "name": "Vestronidase alfa (Mepsevii, ERT)",
        "level": "Level A (somatic — disease-modifying)",
        "role": (
            "FDA 2017 (first approved ERT for MPS-VII); 4 mg/kg IV every 2 weeks; reduces somatic "
            "HS+DS+CS burden in liver/spleen, airway, cardiac, bone, connective tissue; DOES NOT cross "
            "the BBB — no CNS/cognitive benefit; anti-vestronidase IgG/IgE pre-screen mandatory; "
            "infusion reaction management (anti-histamine + corticosteroid pre-medication); continue "
            "peri-operatively (do NOT hold for surgery — somatic rebound risk); dosing EVERY 2 WEEKS "
            "(not weekly like galsulfase — critical compliance distinction)"
        ),
        "ci": "Severe infusion reaction (desensitization protocol); anti-vestronidase IgE confirmed",
    },
    {
        "name": "HSCT (Haematopoietic Stem Cell Transplantation)",
        "level": "Level B (severe early-onset — very limited data)",
        "role": (
            "Considered for severe MPS-VII if age < 3 years with available donor — data extremely "
            "limited (disease too rare for RCT); case reports suggest somatic stabilization; cognitive "
            "benefit uncertain due to HS CNS involvement; contrast MPS-I Hurler (Level A HSCT for "
            "cognitive preservation); MPS-VII HSCT decision requires specialist lysosomal disease "
            "centre + multidisciplinary consensus; vestronidase alfa ERT bridging to HSCT or alone "
            "if HSCT not feasible"
        ),
        "ci": "Attenuated MPS-VII (risk-benefit unfavorable); age > 4-5yr (diminishing benefit); "
              "severe pre-existing organ failure; absence of suitable donor",
    },
    {
        "name": "CPAP / BiPAP (non-invasive ventilation)",
        "level": "Level A (OSA dominant trigger)",
        "role": (
            "OSA management — sleep study at diagnosis and annually; HS+DS+CS triple airway accumulation "
            "means airway compromise can be more severe than single-GAG MPS types; BiPAP titration for "
            "AHI reduction; tonsillectomy/adenoidectomy in pediatric cases if significant tonsillar "
            "hypertrophy; non-invasive ventilation reduces hypoxic seizure threshold and seizure frequency"
        ),
        "ci": None,
    },
    {
        "name": "VP Shunt / Endoscopic Third Ventriculostomy",
        "level": "Level A (communicating hydrocephalus)",
        "role": (
            "For communicating hydrocephalus (HS+DS arachnoid villi infiltration → impaired CSF "
            "reabsorption); LP opening pressure measurement if headache or new seizures; VP shunt "
            "preferred over ETV in communicating hydrocephalus; shunt malfunction risk from GAG "
            "viscosity — regular surveillance; ICP normalization → seizure reduction; brain MRI "
            "periventricular T2 changes and ventricle size monitoring"
        ),
        "ci": "Uncorrected coagulopathy; active CNS infection",
    },
    {
        "name": "Lamotrigine (LTG)",
        "level": "Level B (adjunctive focal)",
        "role": (
            "Adjunctive for focal seizures; no CYP induction; HLA-B*15:02 mandatory in SE Asian "
            "ancestry (SJS risk); no significant cardiac effect; slow titration mandatory; useful in "
            "attenuated MPS-VII with focal epilepsy and normal-ish cognition; interaction with VPA "
            "(halve LTG dose when co-prescribed)"
        ),
        "ci": "HLA-B*15:02 positive without prior tolerance; rapid titration (SJS risk)",
    },
    {
        "name": "Melatonin",
        "level": "Level B (circadian disruption adjunct)",
        "role": (
            "Circadian disruption from OSA + nocturnal GAG-related events; BiPAP/NIV first (OSA must "
            "be treated before melatonin); melatonin adjunct for circadian resync after OSA management; "
            "no interactions with vestronidase alfa ERT; useful in all three MPS-VII clinical forms"
        ),
        "ci": None,
    },
]

# Contraindications (MPS-VII specific)
CONTRAINDICATIONS = [
    {
        "drug": "POLG1 mutation carriers — Valproate (VPA)",
        "risk": "ABSOLUTE CI (CPIC A)",
        "reason": (
            "POLG1 exclusion MANDATORY before VPA in MPS-VII — HS+DS+CS combined mitochondrial "
            "stress in severe forms increases hepatotoxicity risk; POLG1/VPA = Alpers syndrome risk "
            "(fulminant hepatic failure); screen POLG1 before FIRST VPA dose; CPIC A classification "
            "(highest evidence); cannot be skipped even in emergency — use IV LEV for SE instead"
        ),
        "alternative": (
            "IV Levetiracetam for SE; brivaracetam if LEV behavioral side effects; LTG adjunct; "
            "clobazam for clusters (respiratory caution in OSA)"
        ),
    },
    {
        "drug": "Phenytoin / Fosphenytoin",
        "risk": "AVOID",
        "reason": (
            "Cardiac conduction disease possible in MPS-VII (DS cardiac infiltration — similar "
            "mechanism to MPS-VI but less universal); IV fosphenytoin cardiovascular toxicity; "
            "CYP2C9/3A4 induction disrupts vestronidase alfa ERT pharmacokinetics; AVOID in any "
            "MPS-VII patient with documented cardiac involvement; IV LEV replaces in all SE scenarios"
        ),
        "alternative": (
            "IV Levetiracetam for status epilepticus; cardiac echo mandatory before any sodium-channel "
            "AED in MPS-VII; cardiac clearance required"
        ),
    },
    {
        "drug": "Vigabatrin (VGB)",
        "risk": "RELATIVE CI (risk-benefit required)",
        "reason": (
            "Corneal clouding PRESENT in MPS-VII (unlike MPS-II Hunter where absent) → VGB retinal "
            "toxicity additive with corneal HS+DS+CS storage; REMS SHARE program mandatory VF monitoring "
            "may be IMPOSSIBLE if severe corneal clouding; slit-lamp + VF assessment mandatory before "
            "VGB; RELATIVE CI (not absolute as in MPS-VI) — VGB acceptable for infantile spasms if "
            "ACTH fails AND slit-lamp confirms mild or no corneal clouding with monitorable VF"
        ),
        "alternative": (
            "ACTH/prednisolone for infantile spasms (Level A preferred over VGB in MPS-VII); "
            "slit-lamp + VF documentation mandatory if VGB used; ophthalmology follow-up q6M"
        ),
    },
    {
        "drug": "Carbamazepine / Oxcarbazepine",
        "risk": "CAUTION (not absolute CI)",
        "reason": (
            "HS+DS peripheral nerve axonal neuropathy risk (HS drives neuropathy as in MPS-I/II/III; "
            "DS adds dermatan-driven neuropathy); cardiac sodium-channel toxicity if cardiac involvement; "
            "HLA-B*15:02 mandatory in SE Asian ancestry (SJS); CYP induction disrupts vestronidase alfa "
            "ERT in high-dose CBZ; less safe than LEV/LTG as first-line in MPS-VII"
        ),
        "alternative": (
            "LEV or LTG for focal seizures; HLA-B*15:02 screen first in SE Asian ancestry; "
            "cardiac clearance required before CBZ/OXC in MPS-VII"
        ),
    },
    {
        "drug": "Typical antipsychotics (haloperidol, chlorpromazine, perphenazine)",
        "risk": "HIGH RISK",
        "reason": (
            "HS+DS+CS triple GAG accumulation in basal ganglia → severe EPS/NMS risk from typical "
            "antipsychotics; HS+DS combined basal ganglia load (same as MPS-I/II/III) makes EPS "
            "risk higher than single-DS diseases (MPS-VI); behavioral issues common in severe/intermediate "
            "MPS-VII → antipsychotic prescribing risk; atypicals preferred"
        ),
        "alternative": (
            "Atypical antipsychotics (risperidone, aripiprazole, quetiapine) with EPS monitoring; "
            "non-pharmacological behavioral management; melatonin for sleep dysregulation; "
            "pain management for behavior driven by skeletal pain (DS/CS bone involvement)"
        ),
    },
    {
        "drug": "General anaesthesia (all agents)",
        "risk": "EXTREME HAZARD",
        "reason": (
            "HS+DS+CS triple airway accumulation → tongue enlargement + laryngeal/tracheal narrowing; "
            "AAI from odontoid hypoplasia (CS+DS skeletal) → cord risk at laryngoscopy; cardiac "
            "involvement (DS) → hemodynamic instability; vestronidase IgE intraoperative anaphylaxis "
            "risk; three simultaneous hazard mechanisms (airway + cord + cardiac) make MPS-VII "
            "anaesthesia as dangerous as MPS-IVA/I despite different primary mechanisms"
        ),
        "alternative": (
            "Pre-GA mandatory: C-spine XR/MRI (AAI), cardiac echo (DS valvulopathy), ENT airway "
            "assessment (HS+DS+CS), anti-vestronidase IgE (anaphylaxis); video laryngoscopy + awake "
            "fibreoptic intubation; tracheotomy set standby; anaesthesia alert bracelet; ENT + "
            "neuroanesthesia + cardiac joint pre-op planning; paediatric LSD anaesthesia expertise"
        ),
    },
]

# Monitoring parameters
MONITORING = [
    (
        "Urine GAG quantification (HS + DS + CS — all three): baseline + q3-6M on ERT; "
        "triple GAG elevation is PATHOGNOMONIC for MPS-VII; CS elevation distinguishes from "
        "MPS-I/II (HS+DS only) and MPS-VI (DS+C4S only); normalization of urine GAG on ERT "
        "correlates with somatic response; HS normalization not expected (CNS source)"
    ),
    (
        "Brain MRI: communicating hydrocephalus surveillance — periventricular T2 changes; "
        "HS cortical accumulation (white matter signal); brainstem and cord signal (AAI); "
        "annual in stable disease; URGENT if headache, new neurological signs, or seizure "
        "frequency increase; LP opening pressure if hydrocephalus suspected"
    ),
    (
        "C-spine XR (flexion/extension) + MRI: atlantoaxial instability (ADI measurement); "
        "every 6-12M AND mandatory pre-GA; odontoid hypoplasia from CS+DS skeletal storage; "
        "ADI >5mm or T2 cord signal → urgent neurosurgical referral for fusion"
    ),
    (
        "Cardiac echo: DS+CS cardiac infiltration — valvulopathy, cardiomyopathy surveillance; "
        "annual; less universal than MPS-VI but present in moderate-severe forms; cardiac "
        "clearance before PHT/fosphenytoin, CBZ/OXC, or surgery"
    ),
    (
        "Sleep study (polysomnography): OSA from HS+DS+CS triple airway accumulation; "
        "baseline at diagnosis and annually; AHI + oxygen nadir; BiPAP titration; "
        "ENT referral for tonsillar hypertrophy; sleep study in ALL three clinical forms"
    ),
    (
        "Slit-lamp / ophthalmology: corneal clouding (PRESENT in MPS-VII — distinguishes from "
        "MPS-II Hunter absent corneal clouding at bedside); VGB REMS VF documentation requires "
        "ophthalmology confirmation; annual; retinal exam for pigmentary changes"
    ),
    (
        "GUSB enzyme assay (leukocytes/plasma): confirm diagnosis; monitor ERT response; "
        "pseudo-deficiency variants exist (p.H240R reported) — enzyme-alone diagnosis requires "
        "clinical correlation + urine HS+DS+CS; fibroblast assay if leukocyte result borderline"
    ),
    (
        "POLG1 mutation screen: MANDATORY before FIRST VPA dose; CPIC A; cannot be "
        "omitted in any MPS-VII patient; if POLG1 positive → VPA absolute CI → IV LEV for SE"
    ),
    (
        "Hepatic function tests: HS+DS+CS hepatic accumulation → hepatomegaly + transaminase "
        "elevation; LFTs q3-6M on ERT; hepatosplenomegaly monitoring by ultrasound annually; "
        "more stringent monitoring than MPS-IVA/VI due to combined HS+DS hepatic storage"
    ),
    (
        "Audiometry: sensorineural + conductive hearing loss (50-65%); HS+DS+CS cochlear and "
        "middle ear infiltration; annual; hearing aids as indicated; early intervention for "
        "hearing loss critical for cognitive development in attenuated forms"
    ),
    (
        "Vestronidase alfa infusion monitoring: anti-vestronidase IgG (q6M) + IgE (pre-infusion "
        "if high risk); infusion reactions in 25-50% (pre-medicate: anti-histamine + "
        "corticosteroid); anaphylaxis risk — ERT suite emergency preparedness mandatory; "
        "every-2-week dosing compliance tracking"
    ),
    (
        "Developmental/cognitive assessment: psychometry q12M in childhood; HS-driven cognitive "
        "decline in severe/intermediate forms tracked with validated tools (Bayley, Vineland, WPPSI); "
        "educational planning; palliative care coordination in severe neonatal form"
    ),
]

# Clinical pearls
CLINICAL_PEARLS = [
    {
        "pearl": "Non-immune hydrops fetalis → MPS-VII first, not last, on differential",
        "detail": (
            "Non-immune hydrops fetalis (NIHF) with unknown cause — GUSB enzyme assay in amniocytes "
            "or cord blood is MANDATORY. MPS-VII is the only lysosomal storage disease with hydrops "
            "as a primary, frequent presentation (~30-40% of severe neonatal forms). Other LSD hydrops: "
            "Gaucher Type 2, Niemann-Pick A, GM1-gangliosidosis — but MPS-VII most frequent MPS cause. "
            "Neonatal NIHF + coarse facies + hepatosplenomegaly + any GAG elevation → MPS-VII enzyme "
            "assay immediately. Antenatal diagnosis by chorionic villus sampling or amniocentesis GUSB."
        ),
    },
    {
        "pearl": "HS + DS + CS triple GAG elevation = PATHOGNOMONIC for MPS-VII only",
        "detail": (
            "Urine GAG quantification: if HS + DS + CS ALL three elevated simultaneously, MPS-VII "
            "is the ONLY diagnosis. No other MPS type elevates all three GAG classes. MPS-I/II: HS+DS. "
            "MPS-III: HS only. MPS-IVA: KS+C6S (not HS, not DS). MPS-VI: DS+C4S. SUMF1: multiple "
            "sulfatases (HS+DS+HS+KS but GUSB-specific CS pattern distinct). CS elevation specific to "
            "GUSB deficiency. GAG fingerprint confirms MPS-VII even before enzyme result."
        ),
    },
    {
        "pearl": "Corneal clouding PRESENT in MPS-VII — distinguishes from MPS-II (Hunter) at bedside",
        "detail": (
            "MPS-II (Hunter, IDS gene): corneal clouding ABSENT — this is a classic distinguishing "
            "feature of MPS-II. MPS-VII (Sly, GUSB): corneal clouding PRESENT. Slit-lamp mandatory "
            "in MPS-VII. Clinical bedside exam: corneal clouding present → MPS-VII/I/IV/VI possible; "
            "absent → MPS-II/III. VGB relative CI in MPS-VII due to corneal involvement (cannot "
            "monitor VF reliably if corneal clouding severe)."
        ),
    },
    {
        "pearl": "Vestronidase alfa — every 2 weeks, NOT weekly (critical compliance education point)",
        "detail": (
            "Galsulfase (MPS-VI): 1 mg/kg IV WEEKLY. Elosulfase alfa (MPS-IVA): 2 mg/kg IV every "
            "OTHER week. Vestronidase alfa (MPS-VII): 4 mg/kg IV every 2 weeks. Nurse/caregiver "
            "education: confirm dosing frequency at every ERT clinic visit. Missed doses risk somatic "
            "GAG rebound — OSA worsens, AAI risk increases. IgG antibody monitoring q6M; IgE if "
            "infusion reaction occurs. Pre-medicate ALL infusions."
        ),
    },
    {
        "pearl": "POLG1 screen BEFORE first VPA dose — non-negotiable in all MPS types including GUSB",
        "detail": (
            "VPA + POLG1 mutation = Alpers syndrome (fulminant hepatic failure). This exclusion is "
            "CPIC Class A and applies universally to all lysosomal storage diseases including MPS-VII. "
            "HS+DS+CS combined mitochondrial stress in severe MPS-VII may compound hepatotoxicity risk. "
            "Screen POLG1 before first VPA dose — no exceptions. In emergency SE: IV LEV is safe "
            "alternative while POLG1 result pending."
        ),
    },
]


def _seed(val, lo=0.85, hi=1.15):
    """Apply ±15% jitter deterministically."""
    rng = random.Random(42)
    return round(val * rng.uniform(lo, hi), 1)


def get_overview():
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "cohort_size": COHORT_SIZE,
        "disease_mechanism": DISEASE_MECHANISM,
        "epilepsy_prevalence_pct": _seed(40),
        "drug_resistance_pct": _seed(18),
        "osA_pct": _seed(52),
        "hydrocephalus_pct": _seed(43),
        "aai_pct": _seed(36),
        "corneal_clouding_pct": 100,  # UNIVERSAL in MPS-VII (except very attenuated — present)
        "hydrops_fetalis_pct_severe": _seed(35),  # 30-40% of neonatal-severe form
        "hsct_eligible_pct": _seed(35),
        "ert_approved": "Vestronidase alfa (Mepsevii, FDA 2017) — 4 mg/kg IV every 2 weeks",
        "ert_cns_penetration": "NO — does NOT cross BBB; somatic benefit only",
        "unique_fingerprint": "HS + DS + CS ALL THREE elevated (PATHOGNOMONIC — no other MPS)",
        "key_feature": "Non-immune hydrops fetalis (30-40% neonatal-severe) — ONLY MPS type",
        "clinical_pearls": CLINICAL_PEARLS,
        "monitoring_parameters": MONITORING,
        "kpis": [
            {"label": "Cohort", "value": str(COHORT_SIZE)},
            {"label": "Epilepsy", "value": f"~{round(_seed(40))}%"},
            {"label": "DRE", "value": f"~{round(_seed(18))}%"},
            {"label": "OSA", "value": f"~{round(_seed(52))}%"},
            {"label": "Hydrocephalus", "value": f"~{round(_seed(43))}%"},
            {"label": "AAI", "value": f"~{round(_seed(36))}%"},
        ],
    }


def get_breakdown():
    return {
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
    }


def get_definitions():
    return {
        "gene": GENE,
        "full_name": "Beta-glucuronidase",
        "disease": "MPS-VII (Sly Syndrome / Mucopolysaccharidosis Type VII)",
        "omim": OMIM,
        "locus": LOCUS,
        "inheritance": INHERITANCE,
        "enzyme_defect": "Beta-glucuronidase (GUSB) — lysosomal glycoside hydrolase family GH2",
        "gag_elevated": "HS (heparan sulfate) + DS (dermatan sulfate) + CS (chondroitin sulfate) — ALL THREE simultaneously (unique)",
        "ert": "Vestronidase alfa (Mepsevii) — FDA 2017; 4 mg/kg IV q2 weeks; somatic only; NO BBB crossing",
        "hsct": "Considered severe < 3 yr — data very limited; case reports only; multidisciplinary consensus required",
        "epilepsy_pct": "30-50% (spectrum-dependent; higher in severe forms)",
        "dre_pct": "15-22% (structural mechanisms + cortical HS accumulation)",
        "key_distinguishing": "Non-immune hydrops fetalis (neonatal severe); HS+DS+CS triple GAG fingerprint; corneal clouding PRESENT (unlike MPS-II)",
        "founder_mutation": "p.L176F (Belgian European founder — ~15-20% MPS-VII alleles in European cohorts)",
        "polg1_mandatory": True,
        "abbreviations": {
            "MPS": "Mucopolysaccharidosis",
            "HS": "Heparan Sulfate",
            "DS": "Dermatan Sulfate",
            "CS": "Chondroitin Sulfate",
            "ERT": "Enzyme Replacement Therapy",
            "HSCT": "Haematopoietic Stem Cell Transplantation",
            "AAI": "Atlantoaxial Instability",
            "OSA": "Obstructive Sleep Apnoea",
            "VPA": "Valproate",
            "LEV": "Levetiracetam",
            "POLG1": "DNA Polymerase Gamma (mitochondrial polymerase — VPA hepatotoxicity gate)",
            "DRE": "Drug-Resistant Epilepsy",
            "NIHF": "Non-Immune Hydrops Fetalis",
            "BBB": "Blood-Brain Barrier",
            "LSD": "Lysosomal Storage Disease",
            "GAG": "Glycosaminoglycan",
            "VP": "Ventriculoperitoneal (shunt)",
            "ICP": "Intracranial Pressure",
            "CPIC": "Clinical Pharmacogenomics Implementation Consortium",
            "SE": "Status Epilepticus",
            "EEG": "Electroencephalography",
            "MRI": "Magnetic Resonance Imaging",
            "GTCS": "Generalised Tonic-Clonic Seizure",
            "ACTH": "Adrenocorticotropic Hormone",
        },
        "references": [
            "Sly WS et al. Beta-glucuronidase deficiency — MPS VII. Original description 1973.",
            "Kaplan P et al. MPS VII in siblings. J Pediatr 1993.",
            "Batzios S et al. Vestronidase alfa for MPS VII. Mol Genet Metab 2020.",
            "Harmatz P et al. Vestronidase alfa (BMN 110) in MPS VII — phase 3 trial. Ann Intern Med 2018.",
            "Lau HA & Bhatt JM. Non-immune hydrops fetalis as presentation of LSD including MPS VII. "
            "Prenat Diagn 2019.",
            "CPIC Guidelines for VPA and POLG1 — cpicpgx.org.",
        ],
    }
