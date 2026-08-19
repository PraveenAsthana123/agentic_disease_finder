"""
NAGLU Epilepsy — Mucopolysaccharidosis Type IIIB (Sanfilippo Syndrome B)
=========================================================================
40-patient cohort · NAGLU (17q21.2) · Autosomal Recessive (AR) biallelic LOF
NAGLU encodes N-Acetyl-Alpha-Glucosaminidase (NAGLU, 743 aa, ~83 kDa):
  NAGLU cleaves terminal α-N-acetylglucosamine (GlcNAc) residues from the
  non-reducing end of N-sulfated heparan sulfate (HS) chains in lysosomes.
  NAGLU LOF → HS accumulation → MPS IIIB / Sanfilippo syndrome type B.
  HS storage predominantly affects neurons (CNS-dominant LSD).

DISEASE — MPS IIIB / SANFILIPPO SYNDROME TYPE B (OMIM 252920):
  Incidence ~1:211,000 live births (European); MPS III is collectively the most
  prevalent MPS; MPS IIIB is 2nd most common (after IIIA/SGSH).
  Pan-ethnic but higher in some European (NW European, Greek), Middle Eastern
  populations; Netherlands: highest incidence of IIIB (NAGLU p.Arg643Cys founder).
  Severe progressive neurodegeneration: behavioral regression (3–5 years) →
  cognitive decline → motor deterioration → death (early–mid adulthood).
  Epilepsy: 80–90% lifetime prevalence — one of the most epilepsy-prone LSDs.

PATHOGNOMONIC FEATURES:
  (1) URINE HEPARAN SULFATE (HS) — QUANTITATIVE (PATHOGNOMONIC FIRST-LINE):
      Urine GAG quantification (dimethylmethylene blue, DMMB) + HS-specific HPLC/
      tandem-MS → elevated HS with normal/low CS, DS, KS; sensitivity >95%.
      MPS IIIB: isolated HS elevation (unlike MPS I/II/VI where multiple GAGs↑).
      MPS IIIB HS pattern overlaps with MPS IIIA/C/D → enzyme panel mandatory.
  (2) LEUKOCYTE NAGLU ENZYME ACTIVITY <1–5% CONTROL (CONFIRMATORY):
      4-methylumbelliferyl-α-D-glucosaminide substrate; DBS newborn screening
      feasible (multiplex enzyme assay MPS IIIB panel); <5% residual = diagnostic.
      Pseudodeficiency NOT described for NAGLU (unlike MPS I, Fabry, MLD).
  (3) BIALLELIC NAGLU VARIANTS (MOLECULAR CONFIRMATION):
      WES/WGS; >170 pathogenic variants; Dutch founder: p.Arg643Cys (~23%
      Dutch alleles); Greek founder: p.Arg626stop; compound het most common.
      Genotype–phenotype: null/null → severe rapid neurodegeneration; some
      missense alleles confer slower decline.
  (4) BRAIN MRI — WHITE MATTER CHANGES (SUPPORTIVE):
      Periventricular and subcortical white matter T2 hyperintensity (HS
      accumulation in oligodendrocytes/astrocytes); progressive atrophy;
      no cherry-red spot (unlike sphingolipidoses); no corpus callosum thinning
      as early feature (unlike MLD/GALC). Neuroimaging cannot distinguish MPS
      IIIA from IIIB — enzyme/gene confirmation essential.

EPILEPSY — NAGLU-SPECIFIC:
  Prevalence: 80–90% lifetime; onset typically late childhood (6–12 years);
  SECOND most epilepsy-prone MPS after MPS III overall; seizures emerge as
  disease progresses (behavioral phase → cognitive decline → seizure phase).
  Seizure types: GTCS (75%), myoclonic (55%), tonic (40%), absence-like (35%),
  epileptic spasms (late, 20%).
  Drug resistance: 50–60% (HIGH — heparan sulfate accumulation disrupts
  synaptic function broadly; multiple mechanism impairment).
  PME-like pattern: action myoclonus + cognitive decline + EEG generalized SW
  mimics PME but HS accumulation mechanism differs from sphingolipids.
  EEG: generalized slow spike-wave, background slowing, multifocal sharp waves;
  photosensitivity 20–30% (less prominent than NCL/sphingolipidoses).

DISEASE-MODIFYING THERAPY: NO APPROVED THERAPY (2026):
  No ERT approved (NAGLU M6P-tagging recombinant enzyme produced; Phase I/II
  trials showed peripheral reduction but CNS penetration minimal — BBB barrier);
  Intrathecal / intracerebroventricular enzyme delivery: under investigation.
  Gene therapy: AAV9-NAGLU intrathecal/ICV Phase I (NCT03315232) — promising
  CSF HS reduction; CNS vectored approach circumvents BBB limitation.
  Substrate reduction therapy (SRT): miglustat (off-label) — some centres use
  but RCT evidence absent; genistein — negative RCT.
  HSCT: NOT RECOMMENDED for MPS IIIB (unlike MPS I/II) — neurological decline
  continues post-HSCT in MPS III (CNS not protected by donor macrophages in
  established disease); occasional pre-symptomatic NBS-detected cases only.

DRUG SAFETY (AED-SPECIFIC):
  CBZ/OXC: CAUTION (not absolute CI) — peripheral neuropathy less prominent than
    MLD/GALC/ARSA but subclinical axonal neuropathy described in late-stage MPS
    IIIB; hyponatraemia risk; aggravates drowsiness in degenerating CNS.
  VPA: SAFE (HS pathway, not mitochondrial); POLG1 exclusion MANDATORY
    (CPIC Level A — MPS III clinical phenotype overlaps POLG/MERRF superficially);
    hepatic monitoring; VPPP in females ≥12 years.
  VGB: RELATIVE CI — no retinal NCL ganglion cell toxicity but VGB visual
    field testing impossible in severe intellectual disability (ID); use only if
    benefits outweigh monitoring limitation; functional vision assessment quarterly.
  Typical Antipsychotics: HIGH RISK — CNS HS accumulation lowers seizure threshold;
    haloperidol/risperidone → EPS + increased seizures; atypical antipsychotics
    preferred for behavioral symptoms (most prominent feature: aggression, hyperactivity,
    sleep disorder — often misdiagnosed as ADHD or autism spectrum disorder).
  PHT/Phenobarbital: CAUTION — sedation compounds cognitive decline; enzyme
    induction → reduces drug levels unpredictably in progressive CNS disease.
  Clobazam (CLB): USEFUL adjunct for clusters/acute behavioral-seizure overlap;
    sedation monitored (aggravates sleep disorder — prominent in MPS IIIB).
  Melatonin: STRONGLY INDICATED (not AED) — severe sleep disorder (90%+ of MPS
    IIIB patients); sleep fragmentation triggers seizure clusters; melatonin 2–10 mg
    at bedtime; reduces seizure clusters secondary to sleep improvement.
"""

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

GENE        = "NAGLU (N-Acetyl-Alpha-Glucosaminidase)"
LOCUS       = "17q21.2"
OMIM        = "609701 (NAGLU gene); 252920 (MPS IIIB / Sanfilippo Syndrome B)"
INHERITANCE = "Autosomal Recessive (AR) — biallelic LOF; null/null → severe rapid course"
COHORT_SIZE = 40

ETIOLOGIES = [
    {
        "name": "Compound Heterozygous — Null + Missense",
        "pct": 38,
        "onset": "Early childhood — rapid (3–6 years behavioral onset)",
        "notes": (
            "One truncating allele (frameshift/nonsense) + one missense allele; most common genotype "
            "in non-consanguineous European populations; NAGLU activity <2%; "
            "rapid behavioral regression (hyperactivity, aggression, sleep disorder) by age 3–5 years; "
            "language loss by 6–8 years; seizure onset 7–12 years in 85%; "
            "motor decline mid-childhood; death early–mid adulthood (10–30 years)."
        ),
        "key_finding": "Null + missense compound het; NAGLU <2%; rapid neurodegeneration; seizures 85%",
    },
    {
        "name": "Homozygous Dutch Founder — p.Arg643Cys",
        "pct": 23,
        "onset": "Childhood (4–8 years behavioral onset); slightly slower course",
        "notes": (
            "p.Arg643Cys (c.1927C>T) — most prevalent NAGLU variant in Dutch/NW European population "
            "(~23% Dutch alleles); homozygous course moderately severe; "
            "some residual NAGLU activity (1–5%) → slightly attenuated progression vs null/null; "
            "seizures develop in 80%; cognitive plateau phase may extend 1–2 additional years "
            "vs biallelic null; NBS detection increasingly captures this founder variant."
        ),
        "key_finding": "p.Arg643Cys founder (Dutch/NW European); NAGLU 1–5%; moderately severe course",
    },
    {
        "name": "Biallelic Null / Homozygous Truncating",
        "pct": 18,
        "onset": "Severe infantile-childhood (2–4 years behavioral onset); fastest decline",
        "notes": (
            "Biallelic frameshift, nonsense, or large deletion variants; NAGLU activity undetectable "
            "(<1%); most severe and rapidly progressive phenotype; behavioral regression by age 2–4 years; "
            "seizure onset 5–9 years in 90%; drug-resistant epilepsy 65%; "
            "non-ambulatory by age 10–15 years; earliest death in MPS IIIB spectrum; "
            "NBS or family cascade (index case) — pre-symptomatic detection opportunity."
        ),
        "key_finding": "Biallelic null; NAGLU undetectable; fastest decline; seizures 90%; DRE 65%",
    },
    {
        "name": "Greek Founder — p.Arg626stop (Homozygous / Compound Het)",
        "pct": 12,
        "onset": "Childhood severe (3–6 years)",
        "notes": (
            "p.Arg626stop (c.1876C>T) — founder variant enriched in Greek/Eastern Mediterranean populations; "
            "nonsense truncation → null NAGLU; phenotype similar to biallelic null (severe); "
            "isolated from other Mediterranean LSD founders (GBA p.N370S/L444P not NAGLU-associated); "
            "epilepsy onset 6–11 years; seizures 88%; drug resistance 55%; "
            "compound heterozygous with other pathogenic variants also common."
        ),
        "key_finding": "p.Arg626stop founder (Greek/E. Mediterranean); null phenotype; epilepsy 88%",
    },
    {
        "name": "Attenuated — Missense/Missense with Residual Activity",
        "pct": 9,
        "onset": "Late childhood / adolescence (6–12 years behavioral onset); slower course",
        "notes": (
            "Biallelic missense variants preserving 5–15% residual NAGLU activity; attenuated phenotype "
            "(rare — <10% of MPS IIIB); extended plateau phase; seizures present in 60%; "
            "intellectual disability mild-moderate initially; may survive to 4th decade; "
            "psychiatric misdiagnosis (ADHD, autism, schizophrenia) common in attenuated forms; "
            "most vulnerable to late/missed diagnosis; potentially highest benefit from future gene therapy."
        ),
        "key_finding": "Residual NAGLU 5–15%; attenuated slow course; misdiagnosis risk high; seizures 60%",
    },
]

SEIZURE_TYPES = [
    {
        "type": "Generalised Tonic-Clonic (GTCS)",
        "pct": 75,
        "subtype": (
            "Most common seizure type; onset late childhood (7–12 years) as HS accumulation progresses; "
            "LEV and VPA first-line; drug resistance 50–60% overall; "
            "EEG: generalized irregular spike-wave, background slow delta; "
            "cluster pattern common (3–5 GTCS in 24 hours) during febrile illness"
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "pct": 55,
        "subtype": (
            "Action myoclonus (cortical) — triggered by voluntary movement, startle, and light; "
            "PME-like syndrome in severe cases (action myoclonus + cognitive decline + ↓EEG); "
            "LEV most effective; piracetam Level C adjunct; "
            "distinguished from MERRF/sphingolipid PME by GAG accumulation mechanism"
        ),
    },
    {
        "type": "Tonic Seizures",
        "pct": 40,
        "subtype": (
            "Tonic posturing; nocturnal predominance; drug-resistant in 60% of tonic-type; "
            "clobazam PRN nocturnal; rufinamide (Level C) for tonic-atonic clusters; "
            "EEG: desynchronisation + high-amplitude tonic discharge; "
            "falls and injury risk prominent — must be addressed in care planning"
        ),
    },
    {
        "type": "Absence-like (Atypical Absence / Staring Spells)",
        "pct": 35,
        "subtype": (
            "Atypical absence with slow spike-wave (<3 Hz); difficult to distinguish from "
            "cognitive fluctuations of advancing dementia without video-EEG; "
            "ethosuximide POOR choice (no data; myoclonic + tonic component contraindicated); "
            "VPA adjunct; clinical-EEG correlation mandatory"
        ),
    },
    {
        "type": "Epileptic Spasms (Late-onset)",
        "pct": 20,
        "subtype": (
            "Infantile spasm-like clusters emerging late (age 8–15 years) in severe MPS IIIB; "
            "HS accumulation disrupts cortical maturation maintaining spasm susceptibility; "
            "ACTH trial Level C (late-onset); vigabatrin — relative CI (vision monitoring impossible "
            "in severe ID); clobazam + VPA adjunct"
        ),
    },
]

TRIGGERS = [
    {
        "trigger": "Sleep deprivation / disrupted sleep (circadian dysfunction)",
        "pct": 88,
        "notes": (
            "CARDINAL TRIGGER unique to MPS IIIB — severe sleep disorder (90%+ patients): "
            "circadian rhythm disruption from HS accumulation in hypothalamus/SCN → "
            "sleep fragmentation, night waking, reversed sleep-wake cycle; "
            "secondary sleep deprivation → seizure clusters; "
            "melatonin 2–10 mg at bedtime STRONGLY INDICATED (Level B); "
            "sleep disorder often the presenting symptom before seizures; "
            "EEG sleep architecture abnormal (↓slow-wave sleep, ↓REM); "
            "carer burnout risk — sleep disorder is primary caregiver burden"
        ),
    },
    {
        "trigger": "Febrile illness / intercurrent infection",
        "pct": 72,
        "notes": (
            "Fever lowers seizure threshold on top of HS-laden CNS; "
            "MPS IIIB patients less immunodeficient than MPS I/II/VI (no organomegaly-related "
            "immunosuppression) but recurrent otitis media + sinusitis common (GAG-filled macrophages "
            "in mucous membranes impair barrier function); "
            "fever management: paracetamol first-line; rescue benzodiazepine for cluster prevention; "
            "febrile seizure-like clusters may occur even at modest temperatures (38°C)"
        ),
    },
    {
        "trigger": "Behavioral agitation / emotional stress",
        "pct": 65,
        "notes": (
            "MPS IIIB behavioral phase: severe hyperactivity, aggression, impulsivity → "
            "autonomic arousal → seizure threshold lowering; "
            "behavioral and seizure management interlinked (treat sleep disorder → reduces behavioral "
            "agitation → reduces seizure frequency); "
            "behavioral medications (antipsychotics) must not worsen seizures — "
            "atypical antipsychotics preferred (risperidone low-dose, aripiprazole)"
        ),
    },
    {
        "trigger": "Missed AED dose",
        "pct": 60,
        "notes": (
            "Complex medication regimen (AEDs + melatonin + behavioral medications + supplements); "
            "severe intellectual disability + behavioral disorder → adherence impossible without carer; "
            "simplest possible AED regimen essential (once-daily preferred); "
            "liquid formulations for dysphagia in late-stage disease; "
            "caregiver education and pill reminder systems critical"
        ),
    },
    {
        "trigger": "Physical exertion / voluntary movement",
        "pct": 50,
        "notes": (
            "Action myoclonus triggered by voluntary movement → worsened by exertion; "
            "as disease progresses, rest tremor + action myoclonus merge; "
            "physiotherapy modification: balance + gentle movement; "
            "protective equipment for falls; "
            "seizure activity may appear to rise with physiotherapy — distinguish ictal from movement artifact"
        ),
    },
    {
        "trigger": "Photosensitivity (light stimulation)",
        "pct": 25,
        "notes": (
            "Photosensitivity in 20–30% (less than NCL/HEXB/HEXA but more than MAN2B1/GLA); "
            "photo-paroxysmal response on EEG (IPS testing); "
            "standard photosensitivity precautions (polarised glasses, screen filter); "
            "VPA is most effective photosensitivity AED — also addresses GTCS/myoclonus; "
            "EEG testing for PPR before any photic-environment decisions"
        ),
    },
    {
        "trigger": "Antipsychotic medication (typical) initiation/change",
        "pct": 40,
        "notes": (
            "Behavioral symptoms (hyperactivity, aggression, sleep disorder) commonly trigger "
            "antipsychotic prescription; typical antipsychotics (haloperidol, chlorpromazine) → "
            "markedly lower seizure threshold + EPS worsening motor function; "
            "atypical antipsychotics preferred (risperidone ≤0.5–1 mg/day, aripiprazole); "
            "any antipsychotic initiation → EEG monitoring recommended for 4–6 weeks"
        ),
    },
    {
        "trigger": "Sedative/hypnotic drug changes",
        "pct": 35,
        "notes": (
            "Sleep medications (benzodiazepines, antihistamines, clonidine) often prescribed for "
            "sleep disorder; abrupt cessation of benzodiazepines → seizure cluster; "
            "paradoxical agitation with benzodiazepines in MPS IIIB (documented); "
            "clonidine 25–75 mcg at bedtime: useful non-benzodiazepine sleep aid in MPS IIIB; "
            "melatonin 5–10 mg preferred first-line (no withdrawal risk, no seizure threshold effect)"
        ),
    },
]

TREATMENTS = [
    {
        "treatment": "Levetiracetam (LEV)",
        "level": "B",
        "indication": (
            "GTCS, myoclonic, focal seizures; broad-spectrum first-line; IV formulation for SE; "
            "renal excretion (safe in hepatic disease); all MPS IIIB phenotypes; "
            "preferred in advanced disease (no hepatic metabolism, no enzyme interactions)"
        ),
        "mechanism": (
            "SV2A synaptic vesicle protein modulation; reduces cortical myoclonus + generalised bursts; "
            "no CYP enzyme interactions (critical in polypharmacy patients with antipsychotics + melatonin)"
        ),
        "monitoring": (
            "Behavioural side-effects (irritability 10–20% — CRITICAL in MPS IIIB where behavioral "
            "disorder is primary symptom; misattributed as disease progression); "
            "renal dose adjustment; CBC annually; psychiatric monitoring at each visit"
        ),
        "caution": (
            "Behavioral side-effects may worsen agitation and aggression in behavioral phase; "
            "if severe: add clobazam 5 mg at bedtime or switch to ZNS/VPA; "
            "IV LEV preferred over IV PHT/fosphenytoin in SE (avoids Na-channel caution in degenerating CNS)"
        ),
    },
    {
        "treatment": "Valproic Acid (VPA)",
        "level": "B",
        "indication": (
            "GTCS + myoclonic + tonic seizures; photosensitivity; broad-spectrum; "
            "POLG1 exclusion MANDATORY (CPIC Level A) — MPS III phenotype overlaps MERRF superficially; "
            "preferred in photosensitivity pattern"
        ),
        "mechanism": (
            "Sodium channel + GABA augmentation + T-type calcium channel; anti-myoclonic; "
            "HS pathway — no mechanistic conflict with VPA (lysosomal, not mitochondrial); "
            "VPA also has anti-agitation effect (mood stabiliser) — dual benefit in MPS IIIB behavioral phase"
        ),
        "monitoring": (
            "LFT every 3 months (hepatic involvement uncommon in MPS III but mandatory); "
            "ammonia if encephalopathy suspected (must not misattribute HS encephalopathy to VPA); "
            "POLG1 exclusion MANDATORY before initiation; "
            "VPPP monitoring in females ≥12 years; weight gain monitoring (compound with behavioural "
            "medication weight effect)"
        ),
        "caution": (
            "POLG1 exclusion mandatory — MERRF is key phenocopy (PME + mitochondrial disease); "
            "VPA hyperammonaemia may be misdiagnosed as HS encephalopathy; "
            "check ammonia if acute encephalopathy; L-carnitine for VPA-hyperammonaemia if confirmed"
        ),
    },
    {
        "treatment": "Clobazam (CLB)",
        "level": "B",
        "indication": (
            "ADJUNCT for clusters, tonic seizures, rescue; sleep-associated seizure clusters "
            "(nocturnal clobazam 5–10 mg); behavioral agitation overlap; "
            "clobazam 5 mg TID in DRE; STATUS EPILEPTICUS: buccal midazolam first-line rescue"
        ),
        "mechanism": (
            "1,5-benzodiazepine; GABA-A positive allosteric modulator; "
            "longer half-life than clonazepam; less sedating than diazepam; "
            "N-desmethylclobazam (active metabolite) — CYP2C19 polymorphism affects levels"
        ),
        "monitoring": (
            "Sedation (additive with antipsychotics, melatonin, clonidine — all commonly prescribed); "
            "paradoxical agitation in MPS IIIB (documented — reduce dose if aggression worsens); "
            "tolerance (reduce to lowest effective dose after seizure control achieved); "
            "CYP2C19 polymorphism (poor metabolisers have 5× higher N-desmethyl-CLB levels)"
        ),
        "caution": (
            "Paradoxical agitation documented in MPS IIIB — can worsen behavioral phase; "
            "sedation additive with sleep medications; "
            "withdrawal seizures on abrupt cessation; taper over 4–6 weeks; "
            "respiratory depression in severe stage with bulbar weakness"
        ),
    },
    {
        "treatment": "Zonisamide (ZNS)",
        "level": "C",
        "indication": (
            "Adjunct in DRE; myoclonic component; once-daily dosing (adherence advantage); "
            "useful if LEV behavioural side-effects preclude use; "
            "carbonic anhydrase activity → metabolic acidosis monitoring"
        ),
        "mechanism": (
            "Sodium + T-type calcium channel blockade; carbonic anhydrase inhibition; "
            "modulates dopaminergic/serotonergic systems (relevant to behavioral symptoms); "
            "once-daily dosing improves adherence in intellectual disability"
        ),
        "monitoring": (
            "Metabolic acidosis (bicarbonate level 3-monthly); renal stones (hydration counselling); "
            "oligohidrosis/hyperthermia (especially in non-verbal patients — temperature monitoring); "
            "weight loss (additive with disease-related weight loss in late stage)"
        ),
        "caution": (
            "Oligohidrosis risk (reduced sweating → hyperthermia) — non-verbal MPS IIIB patients "
            "cannot report heat intolerance; caregiver temperature monitoring mandatory; "
            "anorexia worsens disease-related dysphagia in late stage"
        ),
    },
    {
        "treatment": "Rufinamide",
        "level": "C",
        "indication": (
            "Tonic-atonic seizure clusters (drop attacks); Lennox–Gastaut-like phenotype in late-stage "
            "MPS IIIB; adjunct to clobazam for tonic spells; "
            "not first-line but useful in DRE tonic-predominant pattern"
        ),
        "mechanism": (
            "Sodium channel modulation (prolonged inactivation); reduces tonic and atonic seizures; "
            "QTc shortening (ECG monitoring mandatory); CYP3A4 weak inhibitor"
        ),
        "monitoring": (
            "QTc (ECG at baseline + 4 weeks — QTc shortening, stop if <300 ms); "
            "appetite suppression (weight loss monitoring in late-stage disease); "
            "drug interactions: VPA increases rufinamide levels (reduce rufinamide dose 30%)"
        ),
        "caution": (
            "QTc shortening may be dangerous in underlying cardiac disease; "
            "interactions with VPA (increase rufinamide by 30–40% → monitor levels); "
            "sedation additive; not for primary myoclonic epilepsy (may worsen)"
        ),
    },
    {
        "treatment": "Melatonin (sleep-seizure cluster prevention)",
        "level": "B",
        "indication": (
            "STRONGLY INDICATED in ALL MPS IIIB patients — not an AED but seizure cluster prevention "
            "via sleep normalisation; 2–10 mg at bedtime; improves sleep onset + duration; "
            "reduces nocturnal seizure clusters secondary to sleep fragmentation"
        ),
        "mechanism": (
            "MT1/MT2 receptor agonism in SCN (hypothalamic circadian pacemaker); "
            "resynchronises disrupted circadian rhythm from hypothalamic HS accumulation; "
            "no dependency, no withdrawal seizures, no behavioural side-effects; "
            "anti-inflammatory effect (minor) — may attenuate HS-driven neuroinflammation"
        ),
        "monitoring": (
            "Sleep diary (nights per week of waking, estimated total sleep hours); "
            "dose titration from 2 mg → 10 mg if needed; "
            "prolonged-release preferred (matches the sleep-maintenance problem pattern of MPS III); "
            "no hepatic/renal monitoring required"
        ),
        "caution": (
            "Not a substitute for AED optimisation; "
            "high-dose melatonin (>20 mg) — no harm signal but no evidence; "
            "daytime sedation at >10 mg — split dosing if needed; "
            "drug interactions minimal (mild CYP1A2 interaction with fluvoxamine — rare co-prescription)"
        ),
    },
    {
        "treatment": "AAV9-NAGLU Gene Therapy (investigational — ICV/IT delivery)",
        "level": "Investigational",
        "indication": (
            "Phase I/II (NCT03315232 — intrathecal; Nationwide Children's Hospital/Lysogene); "
            "ICV/IT delivery bypasses BBB limitation; CSF HS reduction achieved in early cohorts; "
            "pre-symptomatic / early symptomatic preferred; "
            "not approved 2026 — patient access via compassionate use / expanded access"
        ),
        "mechanism": (
            "AAV9 serotype with strong CNS tropism; NAGLU transgene delivered to CNS cells; "
            "intrathecal / intracerebroventricular (ICV) injection → widespread transduction of "
            "neurons + astrocytes; cross-correction (secreted NAGLU taken up by neighbouring cells); "
            "peripheral enzyme deficiency may persist (peripheral vector delivery may be needed separately)"
        ),
        "monitoring": (
            "CSF HS levels (primary biomarker — target normalisation); "
            "CSF/serum NAGLU activity; neurocognitive assessments (Vineland, Bayley); "
            "AAV9 antibody titre (pre-dose screen — pre-existing immunity may exclude); "
            "MRI brain (vector spread, inflammation); "
            "hepatotoxicity (high-dose systemic AAV — minimised by ICV route)"
        ),
        "caution": (
            "NOT APPROVED — no standard dosing; enrol in clinical trial only; "
            "ICV procedure: neurosurgical risk (infection, haemorrhage); "
            "immune response to AAV9 capsid (corticosteroid prophylaxis protocol); "
            "CNS inflammatory response (vector delivery — monitor CSF cells, protein); "
            "efficacy highest pre-symptomatic — advocate NBS detection for maximal gene therapy window"
        ),
    },
    {
        "treatment": "HSCT (NOT recommended in established MPS IIIB)",
        "level": "D (not recommended)",
        "indication": (
            "HSCT NOT recommended for MPS IIIB after symptoms appear — neurological decline "
            "continues post-HSCT in MPS III (unlike MPS I/II where HSCT is beneficial); "
            "occasional consideration in pre-symptomatic NBS-detected infants only (case series, "
            "no RCT); discuss with specialist MPS team; do NOT offer based on MPS I/II data"
        ),
        "mechanism": (
            "Donor macrophages → CNS microglia replacement → cross-correction; "
            "in MPS IIIB: CNS HS burden overwhelming for donor macrophage cross-correction "
            "(insufficient enzyme cross-correction rate for CNS HS accumulation rate); "
            "unlike MPS I (brain HS moderate) — MPS IIIB CNS HS far exceeds cross-correction capacity"
        ),
        "monitoring": "N/A (not recommended)",
        "caution": (
            "Transplant morbidity/mortality NOT justified in MPS IIIB (no neurological benefit demonstrated); "
            "families may request based on MPS I success — clear counselling essential; "
            "refer to gene therapy trial instead; "
            "pre-symptomatic NBS cases: discuss individually with transplant + metabolic specialists"
        ),
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine, Pimozide)",
        "level": "HIGH RISK",
        "reason": (
            "HS accumulation in striatum/basal ganglia → dopaminergic dysfunction → "
            "extreme EPS sensitivity (tardive dyskinesia, dystonia, rigidity); "
            "significant seizure threshold lowering (documented case series); "
            "QTc prolongation (cardiac risk in progressive cardiomyopathy); "
            "atypical antipsychotics preferred (risperidone ≤1 mg/day, aripiprazole, "
            "quetiapine) for behavioral symptoms"
        ),
        "safe_alternative": "Risperidone ≤1 mg/day, aripiprazole, quetiapine; melatonin for sleep",
    },
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin",
        "level": "AVOID / ABSOLUTE CI in SE for MPS IIIB",
        "reason": (
            "CNS HS accumulation + progressive neurodegeneration → narrow therapeutic window; "
            "enzyme induction (CYP2C9/3A4) reduces levels of antipsychotics, melatonin; "
            "IV PHT infusion risk (cardiac arrhythmia, hypotension) in progressive cardiomyopathy; "
            "IV LEV preferred over IV PHT in SE; "
            "chronic PHT → cerebellar atrophy (worsens pre-existing ataxia + motor decline)"
        ),
        "safe_alternative": "IV Levetiracetam for SE; LEV/VPA/CLB for chronic treatment",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "RELATIVE CI",
        "reason": (
            "Visual field monitoring (mandatory for VGB): IMPOSSIBLE to perform reliably in severe "
            "intellectual disability (MPS IIIB patients cannot cooperate with Humphrey perimetry); "
            "VGB retinal toxicity risk cannot be monitored → relative CI; "
            "if used: functional vision assessment (optokinetic drum, preferential looking) quarterly; "
            "no retinal NCL pathology (unlike NCL/CLN diseases where VGB is absolute CI)"
        ),
        "safe_alternative": "LEV, VPA, CLB; ACTH for late spasms (Level C)",
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "level": "CAUTION (not absolute CI)",
        "reason": (
            "Subclinical axonal neuropathy described in late-stage MPS IIIB; "
            "CBZ/OXC aggravate neuropathy in advanced disease; "
            "severe sedation compounds cognitive decline; "
            "hyponatraemia risk (CBZ/OXC SIADH) — difficult to detect in non-verbal patients; "
            "enzyme induction reduces antipsychotic + VPA levels; "
            "HLA-B*15:02 MANDATORY before use in SE Asian ancestry"
        ),
        "safe_alternative": "LEV first-line; VPA if myoclonic; CLB adjunct",
    },
    {
        "drug": "Ethosuximide",
        "level": "NOT INDICATED",
        "reason": (
            "Pure absence epilepsy agent; absence-like spells in MPS IIIB have mixed mechanism "
            "(atypical absence + myoclonic + tonic components); "
            "ethosuximide monotherapy → worsening of GTCS/tonic/myoclonic components; "
            "VPA preferable (covers absence + myoclonus + GTCS spectrum)"
        ),
        "safe_alternative": "VPA (covers full spectrum); CLB adjunct",
    },
    {
        "drug": "Sedating Antihistamines (Diphenhydramine, Promethazine)",
        "level": "AVOID for sleep disorder",
        "reason": (
            "Paradoxical agitation in MPS IIIB (documented — CNS HS disrupts histamine "
            "signalling); anticholinergic effects (urinary retention, constipation in already "
            "autonomic-impaired patients); seizure threshold lowering (antihistamine effect); "
            "melatonin + clonidine are safer alternatives for sleep disorder in MPS IIIB"
        ),
        "safe_alternative": "Melatonin 5–10 mg; clonidine 25–75 mcg; low-dose sedating antidepressants (off-label)",
    },
]

MONITORING = [
    "Urine GAG quantification (DMMB + HS-specific HPLC-MS) — annual biomarker tracking",
    "Leukocyte NAGLU enzyme activity — baseline + response to investigational therapy",
    "NAGLU gene panel / WES — biallelic variants confirmed at diagnosis; family cascade",
    "Video-EEG — 2-hourly minimum at diagnosis; annually or after seizure type change",
    "Sleep diary + actigraphy (caregiver-reported) — monthly (sleep disorder is #1 trigger)",
    "Behavioural assessment (ABAS-3, Vineland) — 6-monthly (disease staging + AED impact)",
    "Brain MRI — baseline + every 2 years (white matter progression, cortical atrophy)",
    "ECG — baseline + on rufinamide (QTc shortening), on antipsychotics (QTc prolongation)",
    "LFT + ammonia — 3-monthly on VPA (hepatic monitoring + hyperammonaemia screen)",
    "VPA level (trough) — 3-monthly; VPPP compliance annually in females ≥12 years",
    "POLG1 sequencing — MANDATORY before VPA initiation",
    "HLA-B*15:02 — MANDATORY before CBZ/OXC in SE Asian ancestry",
    "Ophthalmology — annual (fundoscopy; no cherry-red spot expected but monitor for retinal HS deposits)",
    "Dysphagia screen (SALT referral) — 6-monthly in late stage (aspiration pneumonia risk)",
]

THRESHOLDS = [
    {"parameter": "NAGLU enzyme activity (diagnostic)", "threshold": "<5% control activity", "action": "Diagnostic; confirm biallelic NAGLU variants"},
    {"parameter": "Urine HS (DMMB)", "threshold": ">2× upper normal for age", "action": "Send MPS III enzyme panel (NAGLU, SGSH, GNS, HGSNAT)"},
    {"parameter": "VPA trough level", "threshold": "50–100 mg/L (350–700 µmol/L)", "action": "Check if below: increase dose; if above 120 mg/L: reduce (toxicity)"},
    {"parameter": "VPA ALT", "threshold": ">3× ULN on VPA", "action": "Reduce/hold VPA; ammonia; discuss alternative (LEV, CLB)"},
    {"parameter": "CBZ sodium (SIADH)", "threshold": "<130 mmol/L on CBZ/OXC", "action": "Hold CBZ/OXC; fluid restriction; switch to LEV"},
    {"parameter": "Rufinamide QTc", "threshold": "QTc <310 ms on rufinamide", "action": "Stop rufinamide immediately; cardiology review"},
    {"parameter": "Antipsychotic QTc", "threshold": "QTc >500 ms on antipsychotic", "action": "Reduce/stop antipsychotic; switch to aripiprazole (minimal QTc effect)"},
    {"parameter": "Melatonin dose", "threshold": "Start 2 mg; titrate to 10 mg", "action": "If no sleep improvement at 10 mg: add low-dose clonidine 25 mcg"},
    {"parameter": "Seizure cluster (emergency)", "threshold": "≥3 seizures / 24 hours OR any seizure >5 min", "action": "Buccal midazolam; call emergency services if not responsive"},
    {"parameter": "Drug resistance definition", "threshold": "Failure of 2 adequate AED trials (ILAE 2010)", "action": "MPS specialist review; gene therapy trial referral; presurgical evaluation (rarely applicable)"},
    {"parameter": "NAGLU NBS screen", "threshold": "Enzyme activity <10% on DBS newborn screen", "action": "Confirmatory leukocyte assay + urine HS + NAGLU gene panel; early gene therapy referral"},
    {"parameter": "ZNS bicarbonate", "threshold": "<18 mmol/L on ZNS", "action": "Reduce ZNS dose; bicarbonate supplement if symptomatic acidosis"},
]

STANDARDS = [
    {"code": "ILAE-2022", "title": "ILAE Operational Classification of Seizure Types 2022"},
    {"code": "NICE-NG17", "title": "NICE NG17: Diagnosis and Management of MPS (UK 2016, updated 2022)"},
    {"code": "ACMG-AMP-2015", "title": "ACMG/AMP Variant Classification Standards (NAGLU VUS interpretation)"},
    {"code": "CPIC-POLG-2023", "title": "CPIC Guideline: VPA and POLG1 (Level A — mandatory exclusion before VPA)"},
    {"code": "CPIC-HLA-B1502-CBZ", "title": "CPIC Guideline: CBZ and HLA-B*15:02 (Level A — SE Asian ancestry)"},
    {"code": "MHRA-VPPP-2021", "title": "MHRA Valproate Pregnancy Prevention Programme (females ≥12 years)"},
    {"code": "Fecarotta-2020", "title": "Fecarotta S et al. MPS IIIB European natural history (Orphanet J Rare Dis 2020)"},
    {"code": "Tardieu-2017", "title": "Tardieu M et al. AAV9-NAGLU intrathecal gene therapy Phase I (EMBO Mol Med 2017)"},
    {"code": "NCT03315232", "title": "NCT03315232: AAV9-NAGLU IT gene therapy Phase I/II (Nationwide Children's + Lysogene)"},
    {"code": "Lavery-2017", "title": "Lavery C et al. MPS IIIB natural history UK cohort (J Inherit Metab Dis 2017)"},
    {"code": "WHO-ICF-2019", "title": "WHO ICF Framework for MPS III disability classification"},
    {"code": "ILAE-Genetic-2018", "title": "ILAE Report: Genetic epilepsies — classification and diagnosis"},
]

DEFINITIONS = [
    {"term": "NAGLU", "definition": "N-Acetyl-Alpha-Glucosaminidase; lysosomal exoglycosidase cleaving α-N-acetylglucosamine from N-sulfated heparan sulfate chains; 743 aa; NAGLU gene at 17q21.2; biallelic LOF → MPS IIIB."},
    {"term": "MPS IIIB (Sanfilippo B)", "definition": "Mucopolysaccharidosis type IIIB; NAGLU deficiency; HS accumulation predominantly in CNS neurons; most severe CNS-dominant LSD; progressive neurodegeneration from childhood; epilepsy 80–90%; no approved disease-modifying therapy (2026)."},
    {"term": "Heparan Sulfate (HS)", "definition": "N-sulfated glycosaminoglycan attached to proteoglycans; abundant on neuronal cell surfaces; NAGLU LOF → incomplete degradation → lysosomal HS accumulation → neuronal dysfunction + death; CSF HS is biomarker."},
    {"term": "Sanfilippo Syndrome", "definition": "Collective term for MPS IIIA (SGSH), IIIB (NAGLU), IIIC (HGSNAT), IIID (GNS); all cause isolated HS accumulation + severe progressive dementia with behavioral phase; enzyme panel required to distinguish subtypes."},
    {"term": "Dutch Founder p.Arg643Cys", "definition": "Most common NAGLU pathogenic variant in NW European population (~23% Dutch alleles); c.1927C>T; missense preserving limited NAGLU activity (residual 1–5%); moderately severe MPS IIIB phenotype; NBS increasingly identifies."},
    {"term": "Blood-Brain Barrier (BBB) challenge in MPS III", "definition": "IV ERT cannot adequately penetrate BBB in MPS IIIB; M6P receptor expression on CNS endothelium insufficient; intrathecal/ICV gene therapy circumvents this barrier — the rationale for AAV9 IT delivery in NCT03315232."},
    {"term": "AAV9-NAGLU (IT/ICV gene therapy)", "definition": "AAV serotype 9 vector encoding human NAGLU; intrathecal (IT) or intracerebroventricular (ICV) delivery; broad CNS tropism; Phase I/II NCT03315232 (Lysogene/Nationwide Children's Hospital); CSF HS reduction demonstrated; not yet approved."},
    {"term": "Melatonin (sleep-seizure nexus)", "definition": "Endogenous SCN hormone for circadian entrainment; MPS IIIB disrupts hypothalamic SCN function (HS accumulation) → circadian disorder → severe sleep fragmentation → seizure cluster trigger; melatonin 2–10 mg at bedtime is Level B intervention."},
    {"term": "POLG1 exclusion (mandatory before VPA)", "definition": "POLG1 encodes mitochondrial DNA polymerase gamma; POLG1 pathogenic variants → POLG disease (MERRF, Alpers) → VPA → fatal hepatotoxicity; MPS IIIB is phenocopy of PME/POLG superficially; POLG1 sequencing mandatory before VPA."},
    {"term": "Paradoxical agitation (benzodiazepine)", "definition": "Documented in MPS IIIB: benzodiazepines (diazepam, clonazepam, clobazam) → paradoxical increase in agitation + aggression in some MPS IIIB patients (HS disrupts GABA-A receptor function in striatum); reduce dose/switch if paradoxical effect observed."},
    {"term": "HLA-B*15:02", "definition": "Pharmacogenomic allele (SE Asian ancestry — Han Chinese, Thai, Malaysian, Filipino, Vietnamese); associated with CBZ/OXC-induced SJS/TEN (fatal); CPIC Level A recommendation: test before CBZ/OXC in SE Asian patients; LEV is safe alternative."},
    {"term": "Drug Resistance in MPS IIIB", "definition": "Failure of ≥2 adequate AED trials (ILAE 2010 definition); 50–60% DRE rate in MPS IIIB (higher than most genetic epilepsies); driven by broad HS-mediated synaptic dysfunction (multiple mechanisms simultaneously impaired); gene therapy may modify seizure biology."},
]

LIFECYCLE = [
    {
        "stage": "Pre-symptomatic (0–2 years)",
        "description": (
            "Newborn screening (NBS) increasingly detects NAGLU deficiency via multiplex enzyme assay; "
            "confirmatory leukocyte NAGLU + urine HS + gene panel; "
            "no clinical symptoms at this stage; earliest gene therapy window (highest benefit); "
            "family cascade genetic counselling; HSCT discussion (evidence very limited for MPS IIIB)."
        ),
    },
    {
        "stage": "Behavioral Phase (2–6 years)",
        "description": (
            "First symptoms: hyperactivity, aggression, sleep disorder, language regression; "
            "typically misdiagnosed as ADHD, autism, or speech delay at this stage; "
            "seizures absent in 90% during early behavioral phase; "
            "sleep disorder onset (circadian disruption) — melatonin initiated; "
            "behavioral medications initiated (risperidone, aripiprazole — NOT typical antipsychotics)."
        ),
    },
    {
        "stage": "Cognitive Decline Phase (6–12 years)",
        "description": (
            "Progressive intellectual regression; language lost; comprehension deteriorates; "
            "seizure onset in majority (75–85%) during this phase — GTCS, myoclonic; "
            "EEG: generalised slow spike-wave; AED initiation; "
            "VPA/LEV first-line; clobazam adjunct; gene therapy window narrowing."
        ),
    },
    {
        "stage": "Motor Deterioration Phase (10–18 years)",
        "description": (
            "Loss of ambulation (mean age 12–15 years); dysphagia develops; "
            "epilepsy peaks in frequency and drug resistance (DRE 50–60%); "
            "tonic seizures + drop attacks emerge; rufinamide adjunct; "
            "SALT referral for dysphagia (NG/PEG tube consideration); "
            "seizure rescue plan essential; non-verbal communication strategies."
        ),
    },
    {
        "stage": "Palliative Phase (15+ years)",
        "description": (
            "Bed-bound; fully dependent care; dysphagia + aspiration pneumonia risk; "
            "epilepsy frequency may stabilise or reduce (cortical atrophy reduces excitable tissue); "
            "palliative AED approach (comfort, not seizure freedom target); "
            "hospice planning discussion with family; advance care directive; SUDEP counselling."
        ),
    },
]

KEY_CONCEPTS = [
    "MPS IIIB (Sanfilippo B): NO approved disease-modifying therapy (2026) — highest unmet need in MPS series",
    "NAGLU at 17q21.2: biallelic LOF → heparan sulfate accumulation → severe CNS-dominant neurodegeneration",
    "Epilepsy prevalence 80–90%: highest among MPS III subtypes; onset late childhood (7–12 years)",
    "Drug resistance 50–60%: driven by broad HS-mediated synaptic dysfunction (multiple mechanisms impaired simultaneously)",
    "Sleep disorder (90%+): cardinal feature; circadian disruption → seizure cluster trigger; melatonin Level B",
    "Typical antipsychotics: HIGH RISK — EPS + seizure threshold lowering on HS-laden basal ganglia; use atypicals",
    "PHT/Fosphenytoin: avoid in SE; use IV LEV instead; PHT enzyme induction disrupts antipsychotic levels",
    "VGB: relative CI — visual field monitoring impossible in severe ID; functional vision assessment only",
    "CBZ/OXC: caution (not absolute CI) — late-stage axonal neuropathy + SIADH risk; not same as MLD/ARSA absolute CI",
    "POLG1 exclusion MANDATORY before VPA — MPS IIIB phenotype overlaps MERRF/PME superficially",
    "HLA-B*15:02 MANDATORY before CBZ/OXC in SE Asian ancestry — SJS/TEN fatal risk",
    "AAV9-NAGLU gene therapy (NCT03315232): intrathecal/ICV delivery bypasses BBB; Phase I/II; enrol in trial",
    "HSCT NOT recommended in established MPS IIIB — unlike MPS I/II, neurological decline continues post-HSCT",
    "Clobazam paradoxical agitation: documented in MPS IIIB — reduce dose if agitation worsens on CLB",
    "NBS detection is critical: pre-symptomatic detection → earliest gene therapy window; best long-term outcome",
]

DIFFERENTIAL_DIAGNOSIS = [
    {
        "condition": "MPS IIIA (SGSH / Sanfilippo A)",
        "distinguishing": "Urine HS identical pattern; SGSH enzyme (not NAGLU) low; SGSH gene; clinically indistinguishable from MPS IIIB — ENZYME PANEL mandatory",
    },
    {
        "condition": "MPS IIIC (HGSNAT / Sanfilippo C)",
        "distinguishing": "HGSNAT enzyme low (not NAGLU); HS elevated; slower clinical course than IIIA/B; adult-onset attenuated forms; HGSNAT gene on 8p11.21",
    },
    {
        "condition": "MPS IIID (GNS / Sanfilippo D)",
        "distinguishing": "GNS enzyme low; rarest subtype; clinical similar to other Sanfilippo; enzyme panel distinguishes",
    },
    {
        "condition": "MPS II (Iduronate-2-Sulfatase / IDS) — X-linked",
        "distinguishing": "HS + DS elevated (not HS alone); X-linked recessive (males predominantly affected); somatic features prominent (coarse facies, hepatosplenomegaly, cardiac valve); IDS gene; ERT available (idursulfase)",
    },
    {
        "condition": "MERRF / POLG disease (PME, mitochondrial)",
        "distinguishing": "Superficial phenocopy in early behavioral phase; lactate elevated; POLG1 gene; mitochondrial DNA deletion (muscle biopsy); VPA CI in POLG; ragged red fibres; key exclusion before VPA",
    },
    {
        "condition": "Autism Spectrum Disorder (ASD) / ADHD",
        "distinguishing": "MPS IIIB behavioral phase misdiagnosed as ASD/ADHD in 60–70%; urine GAG screen distinguishes; NAGLU enzyme assay; progressive regression (not static) is red flag for metabolic disease",
    },
    {
        "condition": "NCL (Neuronal Ceroid Lipofuscinosis — CLN2, CLN3)",
        "distinguishing": "HS normal; ceroid/lipofuscin on EM (curvilinear bodies, fingerprint profiles); CLN enzyme panel; VGB ABSOLUTE CI in NCL (not relative CI as in MPS IIIB); cherry-red spot rare (unlike HEXA/HEXB)",
    },
]


# ---------------------------------------------------------------------------
# API FUNCTIONS
# ---------------------------------------------------------------------------

def get_overview():
    """Return NAGLU/MPS IIIB overview data for /api/naglu/overview."""
    by_etiology = {e["name"]: e["pct"] for e in ETIOLOGIES}
    by_seizure   = {s["type"]: s["pct"] for s in SEIZURE_TYPES}
    by_trigger   = {t["trigger"]: t["pct"] for t in TRIGGERS}

    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "cohort_size": COHORT_SIZE,
        "disease_name": "MPS IIIB — Sanfilippo Syndrome Type B",
        "disease_mechanism": (
            "NAGLU biallelic LOF → N-Acetyl-Alpha-Glucosaminidase deficiency → "
            "lysosomal heparan sulfate (HS) accumulation → CNS-predominant progressive "
            "neurodegeneration; behavioral regression → cognitive decline → motor failure."
        ),
        "epilepsy_prevalence_pct": 85,
        "epilepsy_onset": "Late childhood — 7–12 years (peaks in cognitive decline phase)",
        "drug_resistance_pct": 55,
        "no_approved_disease_therapy": True,
        "gene_therapy_phase": "Phase I/II (NCT03315232 — AAV9-NAGLU intrathecal/ICV)",
        "pme_risk": "Moderate — action myoclonus + cognitive decline pattern; HS mechanism (not sphingolipid)",
        "cardinal_seizure_types": ["GTCS", "Myoclonic", "Tonic", "Absence-like", "Late spasms"],
        "cardinal_trigger": "Sleep deprivation (circadian disruption — SCN HS accumulation)",
        "absolute_ci_drugs": ["Typical antipsychotics (EPS + seizure threshold)", "PHT/Fosphenytoin (SE — use IV LEV)"],
        "key_safe_aeds": ["Levetiracetam (LEV)", "Valproic Acid (VPA — POLG1 excluded)", "Clobazam (CLB adjunct)", "Zonisamide (ZNS)"],
        "key_warning": (
            "NO approved disease-modifying therapy (2026). Enrol in AAV9-NAGLU gene therapy trial "
            "(NCT03315232). POLG1 exclusion MANDATORY before VPA. HSCT NOT recommended in "
            "established disease. Melatonin strongly indicated (sleep-seizure nexus)."
        ),
        "by_etiology": by_etiology,
        "by_seizure_type": by_seizure,
        "top_triggers": dict(list(by_trigger.items())[:5]),
        "etiologies": ETIOLOGIES,
        "lifecycle": LIFECYCLE,
        "key_concepts": KEY_CONCEPTS,
    }


def get_breakdown():
    """Return NAGLU/MPS IIIB detailed breakdown for /api/naglu/breakdown."""
    patients = []
    import random
    rng = random.Random(42)
    etiology_pool = ETIOLOGIES
    seizure_pool  = SEIZURE_TYPES
    treatment_pool= TREATMENTS
    drug_resp = ["Controlled", "Partially controlled", "Drug-resistant"]
    drug_resp_w = [0.15, 0.30, 0.55]
    stages = [l["stage"] for l in LIFECYCLE]

    for i in range(1, COHORT_SIZE + 1):
        etiology = rng.choices(etiology_pool, weights=[e["pct"] for e in etiology_pool])[0]
        sz_types = [s["type"] for s in rng.choices(seizure_pool, k=rng.randint(1, 3), weights=[s["pct"] for s in seizure_pool])]
        sz_types = list(dict.fromkeys(sz_types))  # unique preserve order
        aed = rng.choice(treatment_pool[:6])
        response = rng.choices(drug_resp, weights=drug_resp_w)[0]
        age_onset = rng.randint(6, 14)
        stage_idx = min(int((i - 1) / 8), len(stages) - 1)
        trigger = rng.choice(TRIGGERS)
        sleep_disorder = rng.random() < 0.90
        on_melatonin   = sleep_disorder and rng.random() < 0.70

        patients.append({
            "patient_id": f"NAGLU-{i:02d}",
            "etiology": etiology["name"],
            "age_onset_seizures_yrs": age_onset,
            "seizure_types": sz_types,
            "primary_aed": aed["treatment"],
            "drug_response": response,
            "drug_resistant": response == "Drug-resistant",
            "sleep_disorder": sleep_disorder,
            "on_melatonin": on_melatonin,
            "lifecycle_stage": stages[stage_idx],
            "top_trigger": trigger["trigger"],
        })

    drug_resistant_n = sum(1 for p in patients if p["drug_resistant"])
    sleep_disorder_n = sum(1 for p in patients if p["sleep_disorder"])
    on_melatonin_n   = sum(1 for p in patients if p["on_melatonin"])

    return {
        "gene": GENE,
        "cohort_size": COHORT_SIZE,
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "thresholds": THRESHOLDS,
        "lifecycle": LIFECYCLE,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "cohort_summary": {
            "total_patients": COHORT_SIZE,
            "drug_resistant_n": drug_resistant_n,
            "drug_resistant_pct": round(drug_resistant_n / COHORT_SIZE * 100, 1),
            "sleep_disorder_n": sleep_disorder_n,
            "sleep_disorder_pct": round(sleep_disorder_n / COHORT_SIZE * 100, 1),
            "on_melatonin_n": on_melatonin_n,
            "on_melatonin_pct": round(on_melatonin_n / COHORT_SIZE * 100, 1),
        },
        "patients": patients,
    }


def get_definitions():
    """Return NAGLU/MPS IIIB definitions for /api/naglu/definitions."""
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "definitions": DEFINITIONS,
        "key_concepts": KEY_CONCEPTS,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "standards": STANDARDS,
        "diagnostic_algorithm": [
            "Step 1: Urine GAG screen (DMMB quantification + HS-specific HPLC-MS) — elevated isolated HS",
            "Step 2: MPS III enzyme panel (NAGLU + SGSH + HGSNAT + GNS leukocyte assays) — identify NAGLU deficiency",
            "Step 3: Confirmatory leukocyte NAGLU enzyme activity (<5% of control)",
            "Step 4: NAGLU gene sequencing (WES/WGS or targeted panel) — biallelic pathogenic variants",
            "Step 5: POLG1 sequencing — mandatory exclusion before VPA initiation",
            "Step 6: HLA-B*15:02 genotyping — mandatory before CBZ/OXC in SE Asian ancestry",
            "Step 7: Brain MRI (white matter T2 hyperintensity, progressive atrophy) — disease staging",
            "Step 8: Video-EEG with sleep — seizure type characterisation; drug selection",
        ],
        "pharmacological_distinctions": [
            "POLG1 mandatory exclusion before VPA — MPS IIIB phenocopy of PME/MERRF (not mitochondrial but superficially similar)",
            "IV LEV preferred over IV PHT in status epilepticus — PHT enzyme induction; cardiac risk in progressive disease",
            "Typical antipsychotics HIGH RISK — HS-laden basal ganglia → extreme EPS + seizure threshold lowering",
            "VGB relative CI (not absolute) — visual monitoring impossible in severe ID; functional vision only",
            "CBZ/OXC: CAUTION (not absolute CI) — different from MLD/ARSA/GALC where ABSOLUTE CI; late neuropathy risk",
            "Melatonin Level B: sleep-seizure nexus is unique to MPS IIIB among MPS subtypes",
            "HSCT NOT recommended for established MPS IIIB — unique exception among MPS diseases (unlike MPS I/II)",
            "Gene therapy (AAV9-NAGLU IT/ICV) — enrol in NCT03315232; ICV bypasses BBB limitation of IV ERT",
            "Clobazam paradoxical agitation — documented in MPS IIIB; reduce dose if agitation worsens on CLB",
            "No approved ERT — contrast with MPS I (laronidase), MPS II (idursulfase), MPS IVA (elosulfase), MPS VI (galsulfase)",
        ],
    }
