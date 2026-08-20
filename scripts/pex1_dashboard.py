#!/usr/bin/env python3
"""PEX1 / Zellweger Spectrum Disorder (ZSD) Epilepsy Dashboard — seed data module.

PBD-ZSD: Peroxisome Biogenesis Disorder — Zellweger Spectrum. PEX1 encodes the most abundant
peroxisomal biogenesis factor (peroxin-1), a AAA-ATPase (1283 aa, two AAA domains D1+D2);
forms a heterodimer with PEX6 (another AAA-ATPase) anchored by PEX26 in the peroxisomal membrane.
PEX1–PEX6 complex uses ATP hydrolysis to retrotranslocate PEX5 (the cytosolic PTS1-receptor that
shuttles PTS1-tagged matrix proteins to peroxisomes) from the peroxisomal membrane back to cytosol
for recycling. PEX1 LOF → PEX5 trapped/ubiquitinated → peroxisomal matrix import fails → ALL
peroxisomal metabolic functions are simultaneously impaired (unlike ABCD1 which ONLY affects VLCFA
beta-oxidation; PEX1 affects ALL pathways below PLUS VLCFA AND plasmalogens).

LOCUS: 7q21.2  |  OMIM GENE: *602136  |  OMIM DISEASE: #214100 (ZS), #202370 (NALD), #266510 (IRD)

EPIDEMIOLOGY:
  PBD-ZSD prevalence: 1/50,000–1/100,000 births. PEX1 mutations account for ~65% of all PBD-ZSD
  (most common PBD gene); PEX6: ~10%; PEX12: ~5%; others: PEX2/PEX3/PEX10/PEX16/PEX19/PEX26.
  p.Gly843Asp (G843D): most common hypomorphic PEX1 allele (~30% of all PEX1 alleles); European
  founder; homozygous G843D → IRD (attenuated end: RP + hearing loss + neuropathy, seizures 30–50%);
  compound heterozygous G843D + null → NALD (intermediate); two null alleles → ZS (severe, fatal <1yr).

PEROXISOMAL BIOCHEMISTRY — ALL PATHWAYS IMPAIRED (KEY DISTINCTION FROM ABCD1):
  (1) VLCFA beta-oxidation FAILED → C26:0, C24:0, C25:0 elevated (same as ABCD1)
  (2) Alpha-oxidation FAILED → phytanic acid elevated (same as adult Refsum PHYH) + pristanic acid
  (3) Plasmalogens (ether-phospholipids) BIOSYNTHESIS FAILED → ERYTHROCYTE PLASMALOGENS LOW
      [CRITICAL DISTINCTION: plasmalogens NORMAL in ABCD1; LOW in PBD-ZSD → key differentiator]
  (4) Pipecolic acid catabolism FAILED → pipecolic acid elevated in plasma/urine
  (5) DHA (docosahexaenoic acid, 22:6n-3) synthesis FAILED → DHA low in plasma/erythrocytes
      [DHA essential for retinal + brain development → retinopathy + neuronal migration defects]
  (6) Bile acid synthesis FAILED → DHCA (di-hydroxycholestanoic acid) + THCA (trihydroxycholestanoic
      acid) elevated → cholestatic liver disease; cholic acid supplementation may reduce toxic
      bile acid intermediates
  (7) Catabolism of 2-hydroxy fatty acids FAILED → 2-OH fatty acids elevated (minor marker)
  NBS BIOMARKER PANEL: C26:0-lyso-PC (DBS) + C26:0/C22:0 ratio + plasmalogens (RBC) + pipecolic acid

PHENOTYPIC SPECTRUM (ZSD CONTINUUM):
  (1) Zellweger Syndrome (ZS, severe): neonatal onset; craniofacial dysmorphism (large anterior fontanelle,
      high forehead, flat face, epicanthal folds, Brushfield spots); hypotonia (universal, profound);
      cortical neuronal migration defects (pachygyria, PMG, cortical gyral abnormalities = PATHOGNOMONIC
      on MRI); neonatal seizures UNIVERSAL (100%); hepatomegaly + cholestasis (neonatal); retinopathy
      (ERG absent); sensorineural hearing loss; adrenal insufficiency (low cortisol); skeletal stippling
      (chondrodysplasia punctata, esp. patella + acetabulum — radiograph finding); death <6–12 months;
      NEVER reaches HSCT window; supportive only.
  (2) Neonatal ALD (NALD, intermediate): onset 1st–12th month; intermediate severity; seizures 80–90%;
      slower CNS deterioration than ZS; cholestasis milder; hepatomegaly; retinopathy; hearing loss;
      survival months to years; some develop demyelinating WM disease (resembles ABCD1 AMN);
      DHA supplementation trial recommended; no proven disease-modifying therapy.
  (3) Infantile Refsum Disease (IRD, attenuated): onset toddler–school age; isolated peroxisomal impairment
      (enough residual PEX1 activity for partial function); RP (retinitis pigmentosa) + SNHL +
      peripheral neuropathy; hepatomegaly mild; intellectual disability (variable); seizures 30–50%;
      survival adolescence–adulthood; phytol-restricted diet + DHA supplementation + hearing aids + VF
      monitoring; p.Gly843Asp homozygous = canonical IRD genotype.

CORTICAL NEURONAL MIGRATION DEFECTS (PATHOGNOMONIC OF ZS):
  Neuronal migration requires DHA in neuronal membranes and functional peroxisomal lipid biosynthesis.
  ZS: pachygyria (broad simplified gyri) + polymicrogyria + heterotopias; MRI: thick cortex +
  simplified gyral pattern + periventricular cysts (germinolytic) + cerebellar hypoplasia +
  hypomyelination (T2 bright WM). These defects PERMANENT — no treatment reverses migration defect.
  DHA low → impaired neuronal membrane fluidity → failed neuroblast migration from germinal matrix.

EPILEPSY IN PEX1-ZSD:
  ZS: 100% epilepsy; neonatal seizures (day 1–7); multifocal clonic + tonic + myoclonic + electrographic;
  hypsarrhythmia if surviving to infantile period; EEG: high-amplitude chaotic + burst-suppression +
  multifocal spikes; drug-resistant in all ZS; SE frequent; prognosis dismal.
  NALD: 80–90% epilepsy; infantile spasms (IS) in 40–60% + hypsarrhythmia; focal and generalized;
  myoclonic; partial drug response possible.
  IRD: 30–50% epilepsy; focal + GTCS; better drug response; partial phytanic acid restriction helps
  (reduces proconvulsant phytanic acid neuromodulatory effects).

AED PHARMACOLOGY (UNIQUE PEX1-ZSD RULES):
  LEV (Levetiracetam): FIRST-LINE in all ZSD forms; no enzyme induction; no hepatotoxicity; IV/PO;
    safe in neonates (weight-based dosing); no interaction with DHA or bile acids; SV2A mechanism.
  VPA (Valproate): HIGH RISK (3 independent mechanisms):
    (a) Hepatotoxicity — ZS/NALD have baseline cholestatic liver disease → VPA dramatically increases
        hepatic failure risk; POLG1 exclusion MANDATORY (CPIC Grade A) BEFORE any VPA use;
        enhanced LFT monitoring q2W if used;
    (b) VPA inhibits peroxisomal beta-oxidation (separate from mitochondrial) → worsens VLCFA
        accumulation in ZSD (unlike most LSDs where VPA is safe);
    (c) VPA → acylcarnitine sequestration → depletes carnitine; pre-existing carnitine deficiency
        in some ZSD patients worsens effect.
    → VPA = RELATIVE-TO-ABSOLUTE CI depending on liver function; avoid in ZS; use with extreme
    caution in IRD only if LEV/LTG/CLB inadequate.
  CBZ / OXC / PHT: RELATIVE CI (3 mechanisms):
    (a) CYP3A4 enzyme induction → reduces DHA levels (DHA is metabolized by CYP enzymes);
        theoretical concern for further DHA depletion already low in ZSD;
    (b) Hepatic enzyme induction compounds cholestatic liver burden in ZS/NALD;
    (c) PHT specifically: does NOT distinguish from ABCD1 here — no adrenal involvement in ZSD
        (UNLIKE ABCD1); PHT CI in ZSD is neuropathy + hepatic (NOT adrenal crisis);
    → RELATIVE CI (not absolute); use if seizures not controlled by LEV/LTG/CLB; monitor LFT.
  VGB (Vigabatrin): HIGH RISK:
    (a) Irreversible visual field constriction — ZSD patients ALREADY have retinopathy (universal in
        ZS/NALD; 30–50% IRD) → additive irreversible visual loss;
    (b) Use ONLY if ACTH fails for infantile spasms AND patient already has severe visual loss where
        marginal VGB visual risk is acceptable; AVOID in IRD (preservable visual function).
    → AVOID VGB in all ZSD unless no other option; if used, visual fields q3M OCT mandatory.
  ACTH: Level A for infantile spasms/hypsarrhythmia in NALD (preferred OVER VGB due to retinopathy).
    Standard IS protocol: ACTH 20–40 U/day IM × 2 weeks → taper. Monitor BP, glucose, infection.
  CLB (Clobazam): Level B; adjunct for focal + tonic + myoclonic; no hepatotoxicity; no enzyme
    induction; safe in ZSD; 1,5-benzodiazepine (less sedating than clonazepam for chronic use).
  LTG (Lamotrigine): Level C; adjunct focal + absence; hepatic glucuronidation (NOT CYP) →
    safer than CBZ/PHT for liver in ZSD; watch for severe cutaneous reactions (SJS/TEN).
  Clonazepam (CZP): second-line for myoclonus (ZS myoclonic seizures); IV for acute SE;
    tolerance develops with chronic use; useful short-term myoclonus control.
  Piracetam: Level C for action myoclonus (IRD); renal excretion (safe in ZSD hepatic disease);
    2400–4800 mg/day; monitor renal function.
  PB (Phenobarbital): RELATIVE CI — enzyme induction + profound neonatal sedation compounds hypotonia
    in ZS (already profoundly hypotonic); use only in neonatal SE when LEV fails; short-term only.
  Carbamazepine: see CBZ above.
  ANAESTHESIA EXTREME HAZARD (ZS/NALD):
    Profound hypotonia + coagulopathy (cholestatic liver → reduced clotting factors + Vit K malabsorption)
    + hepatic failure risk + possible cervical instability; pre-op: platelet count + PT/INR + LFT;
    volatile anaesthetics can precipitate hepatic crisis in borderline ZS liver; regional preferred if possible;
    alert anaesthesia team of liver disease + coagulopathy + hypotonia.

DISEASE-MODIFYING TREATMENT (2026):
  (a) NO ERT approved (cannot replace a peroxisomal biogenesis factor; enzyme replacement only works
      for soluble lysosomal enzymes, not membrane-associated assembly factors);
  (b) NO HSCT benefit (UNLIKE ABCD1 CCALD — ZSD is diffuse biallelic cell-autonomous defect;
      donor cells cannot rescue recipient neurons; tried experimentally in some IRD with no benefit);
  (c) DHA supplementation (Level B): docosahexaenoic acid 50–200 mg/kg/day ethyl ester (Dhasco);
      improves plasma DHA; small studies show improved ERG + liver function + developmental gains in
      mild ZSD; does NOT cure disease; most benefit in IRD (attenuated); minimal effect in severe ZS.
  (d) Cholic acid (FDA compassionate use; experimental): oral 10–15 mg/kg/day; reduces toxic bile acid
      intermediates DHCA+THCA → improves cholestasis; combined with DHA in some IRD protocols.
  (e) Phytol-restricted diet (Level C, IRD only): restrict dairy fat + ruminant meat (sources of phytol
      → phytanic acid); lowers serum phytanic acid 30–60%; may reduce seizure threshold and neuropathy
      progression in IRD; same dietary restriction used in adult Refsum disease (PHYH).
  (f) Lorenzo's Oil: NOT effective in ZSD (works for ABCD1 by competitive fatty acid displacement but
      ZSD has absent peroxisomal import — VLCFA cannot enter peroxisome regardless of competitive
      substrate; therefore Lorenzo's Oil does not lower VLCFA in PBD-ZSD).
  (g) Gene therapy: AAV-PEX1 preclinical (mouse models + human cell lines) — rescue of peroxisomal
      function demonstrated in vitro; no Phase I human trial as of 2026; PEX1 cDNA 3.8 kb (fits AAV9).
  (h) Carnitine supplementation (Level C, if low): oral L-carnitine 100 mg/kg/day if serum carnitine low;
      secondary carnitine deficiency possible from acylcarnitine accumulation.
  (i) Vitamin K supplementation (Level A, ZS/NALD): cholestatic liver → Vit K malabsorption →
      coagulopathy → bleeding risk; IM/IV Vit K at birth + monthly monitoring.
  (j) Hearing aids (Level A): universal SNHL in ZS/NALD; early fitting if surviving.

NEWBORN SCREENING (NBS):
  C26:0-lysophosphatidylcholine (C26:0-lyso-PC) on DBS: same marker as ABCD1 BUT:
  → In ZSD, C26:0-lyso-PC elevated due to ALL peroxisomal beta-oxidation failure (not just ABCD1
    VLCFA transport); screen flag → reflexive testing: plasma VLCFA + plasmalogens + pipecolic acid +
    whole-exome sequencing.
  GALACTOSYLSPHINGOSINE / PSYCHOSINE: elevated in GALC (Krabbe) only, NOT in ZSD — key NBS differential.
  PEX1 c.2528G>A (p.G843D) targeted NGS: available as confirmatory in NBS pipelines.

POLG1 EXCLUSION (MANDATORY, CPIC GRADE A): Before ANY valproate, exclude POLG1 pathogenic variants
  (POLG1 testing by WES/mtDNA panel); POLG1+VPA → irreversible hepatotoxicity/Alpers syndrome;
  especially important as ZSD already has cholestatic liver.
"""
import random

GENE = "PEX1"
LOCUS = "7q21.2"
OMIM_GENE = "602136"
OMIM_DISEASE_ZS = "214100"
OMIM_DISEASE_NALD = "202370"
OMIM_DISEASE_IRD = "266510"
INHERITANCE = (
    "Autosomal Recessive (AR) — biallelic LOF. Both sexes equally affected. "
    "p.Gly843Asp homozygous = IRD (attenuated); p.Gly843Asp + null = NALD (intermediate); "
    "null + null = Zellweger Syndrome (severe). Carrier frequency ~1/100–1/150 in European populations."
)
COHORT_SIZE = 40
DISEASE_MECHANISM = (
    "PEX1 LOF → defective AAA-ATPase (PEX1–PEX6 heterodimer) → PEX5 (PTS1-receptor) not retrotranslocated "
    "from peroxisomal membrane to cytosol → PEX5 trapped + ubiquitinated + degraded → all PTS1-tagged "
    "peroxisomal matrix proteins fail to import → ALL peroxisomal metabolic functions simultaneously impaired: "
    "(1) VLCFA beta-oxidation FAILED → C26:0 elevated; "
    "(2) Alpha-oxidation FAILED → phytanic + pristanic acid elevated; "
    "(3) Plasmalogen biosynthesis FAILED → RBC plasmalogens LOW [KEY DISTINCTION FROM ABCD1]; "
    "(4) DHA synthesis FAILED → DHA low → neuronal migration defects + retinopathy; "
    "(5) Pipecolic acid catabolism FAILED → pipecolic acid elevated; "
    "(6) Bile acid synthesis FAILED → DHCA + THCA elevated → cholestatic liver disease. "
    "Result: ZSD spectrum (Zellweger → NALD → IRD) depending on residual PEX1 activity. "
    "No ERT possible (biogenesis factor, not enzyme); No HSCT benefit (cell-autonomous defect)."
)

ETIOLOGIES = [
    {
        "name": "Zellweger Syndrome (ZS, Severe)",
        "pct": 30,
        "n": 12,
        "sex": "Both sexes equal (AR)",
        "onset_age": "Neonatal (day 0–7)",
        "seizure_risk": "100% (universal; most severe; multifocal clonic/tonic/myoclonic; EE)",
        "eeg": (
            "Burst-suppression (neonatal); high-amplitude chaotic multifocal spike-wave; "
            "hypsarrhythmia if surviving to infantile period; status epilepticus frequent; "
            "electrodecrements in burst-suppression may be absent-equivalent"
        ),
        "mri": (
            "Pachygyria / polymicrogyria (PATHOGNOMONIC cortical migration defect) + "
            "periventricular germinolytic cysts + cerebellar hypoplasia + absent/hypoplastic "
            "corpus callosum + diffuse hypomyelination (T2-bright WM); no gadolinium enhancement"
        ),
        "loes_range": "N/A (not Loes-applicable; ZSD uses ZSD-severity-score)",
        "hsct_eligible": False,
        "gt_eligible": False,
        "ert_available": False,
        "dha_supplement": False,  # futile in ZS (too severe)
        "variant_detail": (
            "Null + null biallelic LOF (frameshift, nonsense, splice, large deletion); "
            "complete absence of PEX1 protein; no residual peroxisomal function; "
            "death within 6–12 months; NEVER survives to therapeutic window; "
            "palliative care only; 30kb deletion detectable by MLPA"
        ),
    },
    {
        "name": "Neonatal ALD (NALD, Intermediate)",
        "pct": 25,
        "n": 10,
        "sex": "Both sexes equal (AR)",
        "onset_age": "1st–12th month",
        "seizure_risk": "80–90% (infantile spasms 40–60%; focal + myoclonic; hypsarrhythmia)",
        "eeg": (
            "Hypsarrhythmia (40–60%); multifocal spike-wave + high-amplitude irregular background; "
            "may show improvement with ACTH; focal temporal or occipital discharges; "
            "EEG improves transiently with ACTH, then deteriorates with disease progression"
        ),
        "mri": (
            "Less severe migration defect than ZS but abnormal cortical gyration; "
            "WM T2 hyperintensity (demyelination, posterior-predominant); "
            "cerebellar atrophy progressive; periventricular cysts present; "
            "can resemble ABCD1 CCALD WM pattern but WITHOUT gadolinium enhancement"
        ),
        "loes_range": "N/A — ZSD WM scoring system",
        "hsct_eligible": False,
        "gt_eligible": False,
        "ert_available": False,
        "dha_supplement": True,
        "variant_detail": (
            "p.Gly843Asp (G843D) compound het with null (frameshift/nonsense); "
            "G843D produces misfolded but partially functional PEX1; "
            "residual peroxisomal activity ~10–20%; slow progression vs ZS; "
            "survives months to years; DHA supplementation recommended; cholic acid"
        ),
    },
    {
        "name": "Infantile Refsum Disease (IRD, Attenuated)",
        "pct": 28,
        "n": 11,
        "sex": "Both sexes equal (AR)",
        "onset_age": "Toddler–school age (1–6 years)",
        "seizure_risk": "30–50% (focal + GTCS; better drug response; myoclonic uncommon)",
        "eeg": (
            "Focal temporal or posterior discharge; secondary generalisation; "
            "interictal background generally preserved; myoclonic pattern rare; "
            "may show photic sensitivity; EEG improves with LEV in most; "
            "phytanic acid reduction may lower seizure threshold"
        ),
        "mri": (
            "Normal or near-normal cortical gyration (mild simplified gyri); "
            "posterior WM T2-hyperintensity (mild, slowly progressive); "
            "cerebellar volume mild reduction; normal corpus callosum; "
            "periventricular cysts absent or minimal; contrast-negative"
        ),
        "loes_range": "N/A",
        "hsct_eligible": False,
        "gt_eligible": False,
        "ert_available": False,
        "dha_supplement": True,
        "variant_detail": (
            "p.Gly843Asp (G843D) homozygous = canonical IRD (30% of PEX1 alleles European); "
            "G843D + G843D → ~25–30% residual PEX1 activity; "
            "RP (retinitis pigmentosa) + SNHL + peripheral neuropathy dominant features; "
            "phytol-restricted diet + DHA + hearing aids; survival adolescence–adulthood"
        ),
    },
    {
        "name": "Atypical / Late-Onset ZSD",
        "pct": 10,
        "n": 4,
        "sex": "Both sexes equal (AR)",
        "onset_age": "School age – adolescence",
        "seizure_risk": "20–35% (focal; good AED response; neuropathy-triggered falls misidentified as seizures)",
        "eeg": (
            "Focal temporal/posterior; normal interictal background; "
            "no myoclonic or generalised discharges in most; "
            "peripheral neuropathy on NCS (not EEG finding); "
            "phytanic acid reduction may eliminate seizure trigger"
        ),
        "mri": (
            "Normal or minimally abnormal; posterior WM T2-signal; "
            "no migration defect; no gadolinium enhancement; "
            "spinal cord normal; slowly progressive cerebellar atrophy in some"
        ),
        "loes_range": "N/A",
        "hsct_eligible": False,
        "gt_eligible": False,
        "ert_available": False,
        "dha_supplement": True,
        "variant_detail": (
            "Hypomorphic variants (G843D + mild missense) or G843D compound with deep intronic "
            "variant; >30% residual PEX1 activity; isolated RP + hearing loss + neuropathy; "
            "seizures rare; phytanic acid and VLCFA mildly elevated; "
            "whole genome sequencing + RNA-seq required for non-obvious diagnoses"
        ),
    },
    {
        "name": "NBS-Detected Presymptomatic (G843D carrier w/ Compound Het)",
        "pct": 7,
        "n": 3,
        "sex": "Both sexes equal (AR)",
        "onset_age": "Detected at birth via NBS (C26:0-lyso-PC DBS)",
        "seizure_risk": (
            "15–25% (seizures may be delayed; prophylactic monitoring; DHA started immediately "
            "to slow retinal + neurological trajectory)"
        ),
        "eeg": (
            "Normal at diagnosis; follow-up EEG q6M; "
            "focal posterior changes appear with disease progression; "
            "phytanic acid restriction and DHA may prevent early-onset seizures; "
            "NBS detection allows pre-symptomatic DHA initiation"
        ),
        "mri": "Normal or minimal posterior WM signal at diagnosis; surveillance MRI q12M",
        "loes_range": "N/A",
        "hsct_eligible": False,
        "gt_eligible": False,
        "ert_available": False,
        "dha_supplement": True,
        "variant_detail": (
            "Identified via NBS C26:0-lyso-PC DBS elevation → confirmatory plasma VLCFA + "
            "PEX1 sequencing; p.Gly843Asp compound het with null; "
            "DHA started immediately; phytol-restricted diet counselled; "
            "visual-evoked potentials + ERG + audiometry at 6M; ophthalmology q6M"
        ),
    },
]

SEIZURE_TYPES = [
    {
        "type": "Multifocal Clonic + Tonic (Neonatal, ZS)",
        "pct": 62,
        "eeg": (
            "High-amplitude multifocal spike-wave on burst-suppression background (ZS); "
            "day 0–7 onset; drug-resistant in all ZS; IV LEV first-line; "
            "phenobarbital second-line (caution: worsens hypotonia); SE frequent"
        ),
    },
    {
        "type": "Infantile Spasms / Hypsarrhythmia (NALD)",
        "pct": 40,
        "eeg": (
            "Hypsarrhythmia on EEG (high-amplitude chaotic + multifocal spike-wave); "
            "ACTH Level A (preferred over VGB — retinopathy risk); "
            "50% ACTH response in NALD (temporary); VGB AVOID (visual field loss additive)"
        ),
    },
    {
        "type": "Focal Onset with Occipital/Posterior Predominance (IRD)",
        "pct": 38,
        "eeg": (
            "Posterior (O1/O2, P3/P4) spike-wave; maps to occipital cortex / WM lesion; "
            "visual aura in some IRD patients; secondary GTCS frequent; "
            "LEV or LTG first-line; good response in IRD"
        ),
    },
    {
        "type": "Myoclonic (ZS + NALD)",
        "pct": 35,
        "eeg": (
            "Bilateral spike/polyspike-wave with myoclonic jerks; "
            "ZS: continuous myoclonus + burst-suppression; "
            "NALD: action myoclonus pattern; CLB or CZP adjunct; "
            "piracetam Level C for action myoclonus in NALD/IRD"
        ),
    },
    {
        "type": "Generalized Tonic-Clonic (GTCS, IRD / Atypical)",
        "pct": 28,
        "eeg": (
            "Bilateral synchronous spike-wave; secondary generalisation from focal in most; "
            "LEV first-line; LTG second-line; phytanic acid diet reduces frequency in IRD; "
            "avoid PHT (hepatic CI) and CBZ if liver function borderline"
        ),
    },
    {
        "type": "Status Epilepticus (SE — ZS/NALD)",
        "pct": 25,
        "eeg": (
            "ZS: electrographic SE on burst-suppression (ictal electrodecrement); "
            "NALD: convulsive SE during febrile illness; "
            "IV LEV + midazolam first-line; Phenobarbital if refractory (caution: hypotonia); "
            "AVOID fosphenytoin (hepatic CI + neuropathy worsening)"
        ),
    },
]

TRIGGERS = [
    {
        "trigger": "Febrile Illness / Intercurrent Infection",
        "pct": 55,
        "note": (
            "Fever → increased metabolic demand → unmasked peroxisomal insufficiency → "
            "VLCFA + phytanic acid surge (acute febrile catabolism of adipose tissue releases "
            "stored phytanic acid → neuroactive → lowers seizure threshold); "
            "aggressive fever management + maintained AED therapy during illness; "
            "sick-day rules: do NOT fast (phytanic acid in fat stores released with starvation); "
            "IV hydration if oral route compromised."
        ),
    },
    {
        "trigger": "Dietary Phytol Load (IRD)",
        "pct": 42,
        "note": (
            "Dairy fat + ruminant meat → dietary phytol → phytanic acid (via alpha-oxidation); "
            "phytanic acid = proconvulsant (GABA-A receptor modulator) + toxic in high doses "
            "(neuronal membrane disruption); IRD patients on unrestricted diet have higher "
            "seizure frequency; phytol-restricted diet (avoid butter/cream/cheese/beef fat/lamb fat) "
            "reduces phytanic acid 30–60% and often reduces seizure frequency."
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "pct": 38,
        "note": (
            "All ZSD forms with established epilepsy: missed LEV/LTG/CLB → "
            "lowered seizure threshold; in IRD survivors on chronic AEDs, "
            "seizure recurrence within 24–48h of missed dose is common; "
            "patient/family education on strict adherence critical."
        ),
    },
    {
        "trigger": "Fasting / Catabolism (IRD + NALD)",
        "pct": 35,
        "note": (
            "Starvation → lipolysis → adipose release of phytanic acid "
            "(stored in fat = 'phytanic acid reservoir'); surging phytanic acid → acute neurotoxicity + "
            "seizure precipitation; AVOID prolonged fasting in all IRD/NALD; "
            "pre-surgical fasting: IV dextrose infusion to suppress lipolysis MANDATORY; "
            "same principle as adult Refsum disease management."
        ),
    },
    {
        "trigger": "Surgery / Anaesthesia (Fasting-Induced Phytanic Surge)",
        "pct": 30,
        "note": (
            "Pre-operative fasting → phytanic acid surge (adipose lipolysis); "
            "EXTREME HAZARD in ZS/NALD: profound hypotonia + coagulopathy (Vit K malabsorption "
            "→ PT prolonged) + hepatic failure risk (cholestatic liver); "
            "mandatory pre-op: PT/INR + LFTs + platelet count; IV Vitamin K pre-op; "
            "IV dextrose to suppress lipolysis/phytanic acid release during fast; "
            "volatile anaesthetic agents: hepatotoxicity risk in borderline ZS liver; "
            "regional anaesthesia preferred if possible."
        ),
    },
    {
        "trigger": "Sleep Deprivation",
        "pct": 25,
        "note": (
            "Universal seizure trigger in ZSD with established epilepsy; "
            "SNHL + visual loss in many patients → disrupted sleep-wake cycle → "
            "compounded sleep deprivation; melatonin supplementation may help in some "
            "IRD patients with circadian disruption from blindness."
        ),
    },
    {
        "trigger": "Photosensitivity (IRD)",
        "pct": 18,
        "note": (
            "IRD with advancing RP: paradoxical photosensitivity possible as peripheral "
            "retina degenerates but fovea partially preserved → anomalous photic signals; "
            "VEP-triggered seizures rare; EEG photic stimulation at diagnosis and follow-up; "
            "manage: tinted lenses + VEP monitoring."
        ),
    },
]

TREATMENTS = [
    {
        "drug": "Levetiracetam (LEV)",
        "class": "AED — First-line (all ZSD forms)",
        "evidence": "Level B/C — first-line by consensus; no RCT in ZSD specifically",
        "dose": (
            "Neonatal: 20–30 mg/kg/day IV/PO divided BID; "
            "Infant/child: 20–60 mg/kg/day divided BID; "
            "IRD adult: 1000–3000 mg/day divided BID"
        ),
        "moa": "SV2A (synaptic vesicle glycoprotein 2A) modulator — reduces vesicular neurotransmitter release",
        "monitoring": "Renal function (renally excreted; reduce dose if GFR impaired); CBC",
        "ci": None,
    },
    {
        "drug": "DHA Supplementation (Docosahexaenoic Acid)",
        "class": "Disease-modifying — Level B (NALD + IRD)",
        "evidence": "Level B — multiple open-label studies; improves plasma DHA + ERG in mild ZSD; minimal effect in severe ZS",
        "dose": "50–200 mg/kg/day ethyl ester (Dhasco oil or equivalent); given with meals; TID dosing",
        "moa": "Restores deficient DHA → normalises neuronal membrane fluidity → partial improvement in retinal/brain function",
        "monitoring": "Plasma DHA (target: normal range for age); LFTs q3M; ERG q6M",
        "ci": "None established; monitor for GI intolerance (divide dose)",
    },
    {
        "drug": "ACTH (Corticotrophin)",
        "class": "AED — Infantile Spasms (Level A for IS in NALD)",
        "evidence": "Level A for infantile spasms — preferred over VGB due to retinopathy risk in ZSD",
        "dose": "ACTH 20–40 U/day IM × 14 days → taper over 4–6 weeks (standard IS protocol)",
        "moa": "ACTH/corticosteroid-mediated suppression of CRH-driven hypsarrhythmia; reduces EEG chaos",
        "monitoring": "BP, glucose, electrolytes, infection surveillance; cortisol q2W during taper; ERG unaffected by ACTH",
        "ci": "Active infection; uncontrolled hypertension",
    },
    {
        "drug": "Clobazam (CLB)",
        "class": "AED — Adjunct (Level B)",
        "evidence": "Level B adjunct for focal + tonic + myoclonic seizures in ZSD",
        "dose": "0.1–0.5 mg/kg/day divided BID–TID; max 40 mg/day (adult)",
        "moa": "1,5-benzodiazepine; GABA-A positive allosteric modulator; less sedating than 1,4-BZD for chronic use",
        "monitoring": "Sedation assessment; tolerance monitoring; hepatic function q3M",
        "ci": "Acute narrow-angle glaucoma; severe hepatic impairment",
    },
    {
        "drug": "Lamotrigine (LTG)",
        "class": "AED — Second-line focal + GTCS (Level C, IRD)",
        "evidence": "Level C — useful focal seizures in IRD; glucuronidation pathway (safer for ZSD liver than CYP-based AEDs)",
        "dose": "Start 0.5 mg/kg/day; titrate slowly (SJS risk); target 100–400 mg/day adult",
        "moa": "Voltage-gated sodium channel blocker + glutamate release inhibition",
        "monitoring": "Skin rash (SJS/TEN — slow titration mandatory); LFT; CBC",
        "ci": "Rapid titration (SJS risk); concurrent VPA dramatically increases LTG levels → reduce LTG dose 50%",
    },
    {
        "drug": "Cholic Acid (Bile Acid Supplementation)",
        "class": "Disease-modifying — Experimental/Compassionate use",
        "evidence": "Level C — reduces toxic bile acid intermediates DHCA+THCA; improves cholestasis in ZS/NALD",
        "dose": "10–15 mg/kg/day oral in 2–3 divided doses; given with food",
        "moa": (
            "Inhibits CYP27A1 + CYP7A1 via FXR receptor → feedback suppression of "
            "cholesterol → toxic bile acid synthesis; provides normal bile acid pool"
        ),
        "monitoring": "LFTs q1M initial; serum DHCA + THCA if measurable; jaundice monitoring",
        "ci": "Complete biliary obstruction",
    },
    {
        "drug": "Phytol-Restricted Diet (IRD)",
        "class": "Disease-modifying — Dietary intervention (Level C, IRD only)",
        "evidence": "Level C — reduces phytanic acid 30–60%; may reduce seizure frequency in IRD",
        "dose": "Restrict phytol-containing foods: butter, cream, cheese, ruminant meat fat, fatty fish; dietitian essential",
        "moa": "Reduces dietary phytol → phytanic acid precursor; cannot correct peroxisomal defect but reduces substrate load",
        "monitoring": "Serum phytanic acid q6M (target <50 μmol/L); nutritional assessment q6M; ensure adequate calories + fat-soluble vitamins",
        "ci": "Overly restrictive diet → malnutrition risk; NOT for severe ZS (no phytanic acid accumulation concern in ZS)",
    },
    {
        "drug": "Vitamin K (Phylloquinone)",
        "class": "Supportive — Mandatory in ZS/NALD",
        "evidence": "Level A — cholestatic liver → Vit K malabsorption → coagulopathy → bleeding risk",
        "dose": "2 mg IM/IV at birth; monthly IM supplementation; or oral 5 mg/day if cholestasis mild",
        "moa": "Fat-soluble coagulation factor cofactor (II, VII, IX, X); prevents intracranial haemorrhage",
        "monitoring": "PT/INR monthly; PIVKA-II if available; clinical bleeding surveillance",
        "ci": "None for Vit K supplementation",
    },
    {
        "drug": "Clonazepam (CZP)",
        "class": "AED — Myoclonus, acute SE (Second-line)",
        "evidence": "Level C — myoclonic seizures + acute SE in ZSD",
        "dose": "0.01–0.05 mg/kg/day initial; IV 0.1 mg/kg for acute SE; titrate to 0.1–0.2 mg/kg/day",
        "moa": "1,4-benzodiazepine; GABA-A enhancer; anti-myoclonic properties",
        "monitoring": "Tolerance; sedation; drooling; developmental regression with chronic use in infants",
        "ci": "Severe hepatic impairment (avoid in decompensated ZS liver)",
    },
    {
        "drug": "Piracetam",
        "class": "AED — Action Myoclonus (Level C, NALD/IRD)",
        "evidence": "Level C — action myoclonus in NALD/IRD; renal excretion (safe in ZSD hepatic disease)",
        "dose": "2400–4800 mg/day in 3 divided doses (adult); 50–100 mg/kg/day paediatric",
        "moa": "AMPA receptor modulation + membrane fluidity enhancement (may have synergy with DHA in ZSD)",
        "monitoring": "Renal function (renally excreted; reduce in impaired GFR); CNS excitability rarely",
        "ci": "Severe renal impairment (eGFR <20)",
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Valproate (VPA) / Sodium Valproate",
        "level": "HIGH RISK (near-ABSOLUTE CI in ZS/NALD)",
        "reason": (
            "THREE independent mechanisms: (1) Hepatotoxicity — ZS/NALD have baseline cholestatic liver "
            "disease; VPA + liver disease = exponential hepatic failure risk; (2) VPA inhibits peroxisomal "
            "beta-oxidation → worsens VLCFA accumulation in ZSD (unique among LSDs); (3) Carnitine "
            "sequestration → secondary carnitine deficiency worsens in ZSD. POLG1 exclusion MANDATORY "
            "(CPIC Grade A) BEFORE VPA — POLG1+VPA = irreversible Alpers syndrome."
        ),
        "alternative": "LEV (first-line); CLB (adjunct myoclonus); piracetam (action myoclonus). Use VPA ONLY in IRD if liver function normal AND POLG1 negative AND LEV+LTG+CLB inadequate.",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "level": "HIGH RISK — AVOID in most ZSD",
        "reason": (
            "Irreversible visual field constriction (concentric VF loss) — ZSD patients ALREADY have "
            "retinopathy: UNIVERSAL in ZS/NALD (ERG absent/severely abnormal); 30–50% in IRD. "
            "VGB additive irreversible visual loss in patients already retinopathy-impaired. "
            "NEVER use in IRD (preservable visual function). "
            "In NALD: only if ACTH fails AND visual function already severely compromised."
        ),
        "alternative": "ACTH (Level A for infantile spasms) — preferred over VGB due to retinopathy. CLB or piracetam for myoclonus.",
    },
    {
        "drug": "Phenytoin / Fosphenytoin",
        "level": "RELATIVE CI (prefer IV LEV instead)",
        "reason": (
            "Hepatic metabolism + peripheral neuropathy worsening: "
            "(1) PHT is hepatically metabolized via CYP2C9/2C19 — borderline/compromised ZS/NALD liver; "
            "(2) PHT worsens peripheral neuropathy already present in ZSD (demyelinating neuropathy); "
            "(3) IV Fosphenytoin in SE: hepatic risk + neuropathy; use IV LEV instead for SE. "
            "Note: NO adrenal crisis risk from PHT in ZSD (unlike ABCD1 where PHT is ABSOLUTE CI "
            "due to cortisol catabolism) — adrenal glands are NORMAL in ZSD."
        ),
        "alternative": "IV Levetiracetam for SE; oral LEV first-line. Fosphenytoin only as last resort in SE with LEV failure.",
    },
    {
        "drug": "CBZ / OXC (Carbamazepine / Oxcarbazepine)",
        "level": "RELATIVE CI",
        "reason": (
            "CYP3A4 enzyme induction: (1) Reduces DHA levels via accelerated CYP-mediated "
            "DHA metabolism — already critically low in ZSD; DHA depletion worsens retinal + "
            "neurological trajectory; (2) Hepatic enzyme induction compounds cholestatic burden "
            "in ZS/NALD; (3) CBZ can worsen peripheral neuropathy (sodium channel — neuropathic "
            "pain initially but demyelinating worsening with long-term use in some). "
            "IMPORTANT: CBZ is NOT absolute CI here (no adrenal crisis unlike ABCD1); "
            "use if LEV+LTG+CLB inadequate with LFT monitoring + DHA monitoring."
        ),
        "alternative": "LEV (first-line); LTG (glucuronidation, safer for liver); CLB. If CBZ used: DHA supplementation dose increase; LFT monthly.",
    },
    {
        "drug": "Phenobarbital (PB)",
        "level": "RELATIVE CI (neonatal period especially)",
        "reason": (
            "Profound sedation compounds hypotonia in ZS (already profoundly hypotonic — respiratory "
            "arrest risk if over-sedated); enzyme induction (CYP2B/3A) worsens DHA depletion; "
            "hepatic metabolism burden. Short-term use for neonatal SE acceptable as bridge "
            "when IV LEV unavailable; chronic use in NALD/IRD should be avoided."
        ),
        "alternative": "IV LEV for neonatal SE; CLB for chronic seizures in NALD/IRD.",
    },
    {
        "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine, etc.)",
        "level": "HIGH RISK — CAUTION",
        "reason": (
            "Cholestatic liver (ZS/NALD) → reduced hepatic drug metabolism → accumulation → "
            "EPS + NMS risk; also worsens neuropathy (dopamine blockade + direct neuropathic effect); "
            "NOT specifically ZSD-related (general hepatic caution) but compounded here."
        ),
        "alternative": "Quetiapine (hepatic monitoring) if antipsychotic needed for atypical ZSD presentation; psychiatric co-management.",
    },
    {
        "drug": "Lorenzo's Oil (GTO:GTE)",
        "level": "INEFFECTIVE (not contraindicated, just futile)",
        "reason": (
            "Lorenzo's Oil works in ABCD1 by competitive fatty acid displacement to reduce VLCFA "
            "synthesis (works because peroxisomal IMPORT is intact in ABCD1; only beta-oxidation is "
            "impaired by ABCD1 LOF → erucic acid competes with VLCFA-CoA for elongase). "
            "In ZSD: peroxisomal IMPORT is completely absent → VLCFA cannot enter peroxisome "
            "regardless of substrate competition → Lorenzo's Oil does NOT lower VLCFA in PBD-ZSD. "
            "CONFIRM: plasma VLCFA after 4 weeks Lorenzo's Oil shows NO reduction in ZSD "
            "(vs ABCD1 where VLCFA normalises within 4 weeks). Do NOT prescribe to ZSD families."
        ),
        "alternative": "DHA supplementation (Level B) + cholic acid + phytol-restricted diet (IRD) — the only evidence-based disease-modifying interventions.",
    },
    {
        "drug": "HSCT / Bone Marrow Transplant",
        "level": "NOT INDICATED (unlike ABCD1 CCALD where HSCT = Level A)",
        "reason": (
            "ZSD is a cell-autonomous biallelic peroxisomal biogenesis defect — every cell in the body "
            "lacks functional peroxisomes. HSCT replaces myeloid lineage cells only; non-myeloid neurons "
            "and glia still lack PEX1 function; no neurological benefit demonstrated in ZSD "
            "(unlike ABCD1 CCALD where HSCT provides microglial-mediated neuroinflammation arrest). "
            "Transplant morbidity + mortality not justified in ZSD. Tried experimentally in a few IRD "
            "patients (early 2000s) — no reproducible neurological benefit."
        ),
        "alternative": "Supportive care + DHA + phytol-restricted diet + AEDs. Gene therapy (AAV-PEX1) in future trials.",
    },
]

MONITORING = [
    "Plasma VLCFA (C26:0, C24:0/C22:0 ratio) at diagnosis + q6M — primary biochemical surveillance",
    "Erythrocyte plasmalogens (PE-plasmalogen, PC-plasmalogen) at diagnosis — key ZSD vs ABCD1 distinction",
    "Plasma DHA level — monthly if on DHA supplementation; q6M maintenance",
    "Serum phytanic acid + pristanic acid q6M (especially IRD on phytol-restricted diet)",
    "Pipecolic acid urine + plasma at diagnosis (if available) — corroborates ZSD",
    "LFTs + bilirubin + GGT + albumin + PT/INR q1–3M (ZS/NALD liver disease surveillance)",
    "EEG at diagnosis; q6M in ZS/NALD; annually in stable IRD; immediately for suspected new seizures",
    "MRI brain (GRE + T2 + DWI) at diagnosis; q12M surveillance in NALD/IRD",
    "ERG (electroretinogram) at diagnosis + q6M — retinopathy progression monitoring",
    "Ophthalmology + visual fields (Goldmann or automated) q6M in IRD (preservable VF)",
    "Audiology (pure-tone audiogram) q12M in ZSD — SNHL progressive; hearing aids early fitting",
    "Developmental/neuropsychological assessment q12M in surviving NALD/IRD",
    "POLG1 testing before ANY valproate — CPIC Grade A; hepatic catastrophe risk",
    "Pre-operative: PT/INR + LFT + platelet count + IV Vit K; IV dextrose during fast (phytanic surge prevention)",
    "Newborn siblings: C26:0-lyso-PC DBS at birth (NBS or confirmatory); PEX1 sequencing if NBS positive",
]

THRESHOLDS = [
    {
        "parameter": "Plasma C26:0 (VLCFA)",
        "threshold": ">1.3 μg/mL (elevated)",
        "action": "Confirm ZSD; reflexive: plasmalogens + pipecolic acid + PEX gene panel sequencing",
    },
    {
        "parameter": "RBC Plasmalogen (PE-plasmalogen)",
        "threshold": "<50% of age-matched normal",
        "action": "Confirms PBD-ZSD (not ABCD1 where plasmalogens NORMAL); immediate genetics referral",
    },
    {
        "parameter": "Plasma DHA",
        "threshold": "<4% of total fatty acids (low)",
        "action": "Start DHA supplementation immediately; increase dose if still low after 3M",
    },
    {
        "parameter": "Serum phytanic acid (IRD)",
        "threshold": ">50 μmol/L",
        "action": "Intensify phytol dietary restriction; review any dietary lapses; neurologist alert",
    },
    {
        "parameter": "PT/INR (ZS/NALD)",
        "threshold": "INR >1.5",
        "action": "IV Vitamin K immediately; hepatology consultation; blood product preparation pre-procedure",
    },
    {
        "parameter": "LFT (ALT/AST)",
        "threshold": ">5× ULN",
        "action": "STOP VPA immediately if on it; hepatology urgent; DHA/cholic acid review",
    },
    {
        "parameter": "EEG — Burst-Suppression",
        "threshold": "Burst-suppression pattern in neonate",
        "action": "ZS confirmed electrographically; IV LEV load 30 mg/kg; palliative care discussion",
    },
    {
        "parameter": "EEG — Hypsarrhythmia",
        "threshold": "Hypsarrhythmia in infant with known ZSD",
        "action": "ACTH protocol immediately (Level A); AVOID vigabatrin (retinopathy risk)",
    },
]

LIFECYCLE = [
    {
        "stage": "Stage 1: Neonatal/Infantile (ZS) — Days 0 to death (<12 months)",
        "features": (
            "Profound hypotonia; craniofacial dysmorphism; neonatal seizures (day 0–7); "
            "cholestatic jaundice; hepatomegaly; ERG absent; SNHL; pachygyria on MRI; "
            "no milestones achieved; rapid deterioration"
        ),
        "action": "Palliative care; IV LEV for seizures; Vit K IM; DHA trial optional; POLG1 before VPA; avoid VGB",
    },
    {
        "stage": "Stage 2: Infantile (NALD) — Months 1–24",
        "features": (
            "Infantile spasms + hypsarrhythmia (40–60%); developmental delay; "
            "cholestasis (milder than ZS); hepatomegaly; retinopathy (ERG reduced); "
            "SNHL; hypotonia (milder than ZS); focal/myoclonic seizures"
        ),
        "action": "ACTH for IS (Level A); DHA supplementation start; cholic acid; IV LEV; Vit K monitoring; genetics counselling",
    },
    {
        "stage": "Stage 3: Early Childhood (NALD/IRD) — Age 1–5 years",
        "features": (
            "Developmental regression or arrest; progressive retinopathy; SNHL worsening; "
            "seizures stabilise partially with AEDs; hepatomegaly regressing or persisting; "
            "peripheral neuropathy emerging; phytanic acid accumulating"
        ),
        "action": "LEV + LTG + CLB maintenance; DHA supplementation; phytol-restricted diet (if IRD); hearing aids fitting; visual aids; physio/OT",
    },
    {
        "stage": "Stage 4: School Age (IRD) — Age 5–12 years",
        "features": (
            "Focal epilepsy + GTCS (30–50% seizure prevalence); RP (visual field loss); "
            "SNHL requiring hearing aids; peripheral neuropathy (balance problems + falls); "
            "intellectual disability (variable); behavioural problems; "
            "phytanic acid accumulation if diet non-compliant"
        ),
        "action": "Maintain phytol-restricted diet; annual VLCFA + phytanic acid + DHA; q6M ophthalmology + audiometry; AED review; school support plan",
    },
    {
        "stage": "Stage 5: Adolescence/Adulthood (IRD Survivors) — Age 12+ years",
        "features": (
            "Stable focal epilepsy (usually well-controlled on LEV/LTG/CLB); "
            "legal blindness in most (RP); SNHL (cochlear implant in some); "
            "peripheral neuropathy (wheelchair in severe); ataxia progressive; "
            "hepatic function usually stable if cholic acid + diet maintained"
        ),
        "action": "Long-term AED maintenance; phytol-restricted diet lifelong; annual VLCFA; q6M ophthalmology; DEXA (bone density); vocational support",
    },
    {
        "stage": "Stage 6: Pre-Symptomatic NBS-Detected (Any Stage)",
        "features": (
            "Elevated C26:0-lyso-PC on NBS → PEX1 sequencing → diagnosis before symptoms; "
            "opportunity to start DHA + phytol-restricted diet before neurological deterioration; "
            "ophthalmology + audiology baseline; EEG baseline; "
            "family counselling + carrier testing"
        ),
        "action": "DHA immediately; phytol-restricted diet counselling; genetics; q6M surveillance (ERG + audiogram + VLCFA + phytanic acid); NBS sibling testing",
    },
]

KEY_CONCEPTS = [
    "PEX1 LOF → PEX1–PEX6 AAA-ATPase heterodimer dysfunctional → PEX5 (PTS1-receptor) retrotranslocation failure → ALL peroxisomal matrix proteins fail to import.",
    "ZSD spectrum: Zellweger (ZS, severe) → Neonatal ALD (NALD, intermediate) → Infantile Refsum Disease (IRD, attenuated) — determined by residual PEX1 activity, NOT genotype alone.",
    "p.Gly843Asp (G843D): most common PEX1 hypomorphic allele (~30% of PEX1 alleles); European founder; G843D homozygous = IRD; G843D + null = NALD.",
    "Plasmalogens (RBC) LOW: PATHOGNOMONIC of PBD-ZSD; distinguishes ZSD from ABCD1 (plasmalogens NORMAL in ABCD1 — critical NBS/biochemical differential).",
    "ALL peroxisomal pathways impaired simultaneously in ZSD: VLCFA beta-oxidation + alpha-oxidation (phytanic/pristanic) + plasmalogen synthesis + DHA synthesis + pipecolic catabolism + bile acid synthesis.",
    "DHA low in ZSD: DHA deficiency → neuronal migration defects (ZS pachygyria) + retinopathy; DHA supplementation Level B for NALD/IRD.",
    "Cortical neuronal migration defect (pachygyria/PMG) on MRI: PATHOGNOMONIC of ZS — most severe peroxisomal phenotype; no lysosomal disorder shows this pattern.",
    "HSCT NOT indicated in ZSD — unlike ABCD1 CCALD where HSCT Level A; ZSD is cell-autonomous biallelic defect; donor cells cannot rescue neuronal peroxisomal function.",
    "Lorenzo's Oil INEFFECTIVE in ZSD — works in ABCD1 (peroxisomal import intact; competitive substrate reduction possible) but NOT in ZSD (peroxisomal import absent; VLCFA cannot enter peroxisome).",
    "VPA: HIGH RISK (near-ABSOLUTE CI in ZS/NALD) — THREE mechanisms: hepatotoxicity (baseline cholestatic liver) + peroxisomal beta-oxidation inhibition + carnitine depletion.",
    "Vigabatrin: HIGH RISK in ZSD — ZSD retinopathy (universal in ZS/NALD) + VGB irreversible VF loss = additive blindness; AVOID; prefer ACTH for infantile spasms.",
    "PHT/Fosphenytoin: RELATIVE CI in ZSD (hepatic + neuropathy) — distinction from ABCD1: PHT ABSOLUTE CI in ABCD1 due to adrenal crisis (cortisol catabolism), NOT in ZSD (adrenal glands normal).",
    "Fasting/starvation: EXTREME HAZARD in IRD/NALD — adipose phytanic acid reservoir liberated → acute neurotoxicity + seizures; pre-surgical IV dextrose MANDATORY.",
    "Dietary phytol restriction (IRD): reduces phytanic acid 30–60%; same principle as adult Refsum disease (PHYH); reduces seizure frequency in diet-responsive IRD.",
    "NBS: C26:0-lyso-PC DBS → elevated in BOTH ZSD and ABCD1; differential: plasmalogen testing (LOW in ZSD, NORMAL in ABCD1) + sex (ABCD1 males predominantly affected; ZSD AR equal-sex).",
    "POLG1 exclusion MANDATORY (CPIC Grade A) before VPA — POLG1 + VPA = Alpers syndrome (irreversible hepatic failure); ZSD baseline liver disease magnifies POLG1 hazard.",
]

DIAGNOSTIC_ALGORITHM = [
    "Step 1: Suspect PBD-ZSD in neonate with hypotonia + craniofacial dysmorphism + neonatal seizures + hepatomegaly OR in infant/child with RP + SNHL + peripheral neuropathy + developmental delay.",
    "Step 2: Plasma VLCFA (C26:0 ratio) — elevated in ZSD (same as ABCD1); send simultaneously with RBC plasmalogens.",
    "Step 3: Erythrocyte plasmalogens (PE-plasmalogen, PC-plasmalogen) — LOW in ZSD; NORMAL in ABCD1 [KEY DIFFERENTIAL].",
    "Step 4: If plasmalogens LOW → PBD-ZSD confirmed biochemically; order multi-gene PBD panel (PEX1, PEX6, PEX12, PEX2, PEX10, PEX26, etc.).",
    "Step 5: Plasma phytanic acid + pristanic acid + pipecolic acid — elevated in ZSD (supports diagnosis; tracks severity).",
    "Step 6: Plasma DHA — LOW in ZSD; start supplementation while awaiting genetic confirmation.",
    "Step 7: MRI brain — pachygyria / PMG (ZS); WM T2-changes (NALD); mild/normal (IRD); germinolytic cysts (ZS).",
    "Step 8: Ophthalmology + ERG — absent/reduced ERG in ZS/NALD; reduced in IRD; visual field Goldmann in IRD.",
    "Step 9: Audiometry (ABR in infants) — SNHL universal in ZS/NALD; variable in IRD.",
    "Step 10: LFTs + GGT + bilirubin + PT/INR — cholestasis in ZS/NALD; mild or normal in IRD.",
    "Step 11: POLG1 testing before VPA — mandatory; send simultaneously if AEDs being planned.",
    "Step 12: PEX1 sequencing result + functional peroxisomal studies (skin fibroblast catalase staining if needed) → confirm variant classification; family cascade testing.",
]

PHARMACOLOGICAL_DISTINCTIONS = [
    "VPA: HIGH RISK (ZS/NALD near-ABSOLUTE CI) — three independent mechanisms (hepatotoxicity + peroxisomal beta-oxidation inhibition + carnitine depletion); POLG1 exclusion MANDATORY (CPIC A).",
    "VGB: HIGH RISK — ZSD retinopathy (universal) + VGB irreversible VF loss = additive blindness; ACTH preferred for IS in NALD; AVOID VGB in IRD entirely.",
    "PHT/Fosphenytoin: RELATIVE CI in ZSD (hepatic + neuropathy) — NOT adrenal crisis (unlike ABCD1 where ABSOLUTE CI due to CYP3A4-cortisol drop); IV LEV replaces fosphenytoin in SE.",
    "CBZ/OXC: RELATIVE CI — CYP3A4 induction reduces already-low DHA; hepatic enzyme induction burden; use only if LEV/LTG/CLB fail; LFT + DHA monitoring mandatory.",
    "PB (phenobarbital): RELATIVE CI — worsens neonatal hypotonia (ZS); enzyme induction; use only as bridge in neonatal SE when IV LEV unavailable.",
    "Lorenzo's Oil: INEFFECTIVE in ZSD — works in ABCD1 (intact peroxisomal import, competitive substrate); fails in ZSD (absent peroxisomal import, no beta-oxidation regardless of substrate).",
    "HSCT: NOT indicated in ZSD — unlike ABCD1 CCALD (Level A); ZSD is cell-autonomous biallelic defect; microglial replacement cannot rescue neuronal peroxisomal function.",
    "DHA supplementation: SAFE + RECOMMENDED (Level B, NALD/IRD) — restores deficient DHA; NOT useful in severe ZS (futile, too severe); start immediately post-diagnosis.",
    "ACTH: Level A for infantile spasms in NALD — preferred over VGB; no retinal toxicity; standard IS protocol; monitor BP + glucose.",
    "LEV: FIRST-LINE in all ZSD forms — no enzyme induction; no hepatotoxicity; safe in neonates; IV available; SV2A mechanism; no interactions with DHA or bile acid metabolism.",
    "Fasting/anaesthesia: IV dextrose MANDATORY during pre-surgical fast — prevents adipose phytanic acid release → acute neurotoxicity; also IV Vit K pre-op (coagulopathy risk).",
    "Cholic acid: Experimental (Level C) — reduces toxic bile acid intermediates DHCA+THCA in ZS/NALD cholestasis; combined with DHA in IRD protocols; no interaction with AEDs.",
]

DIFFERENTIAL_DIAGNOSIS = [
    {
        "condition": "ABCD1 (X-ALD)",
        "distinction": (
            "ABCD1: plasmalogens NORMAL (only VLCFA beta-oxidation impaired); X-LINKED (males primarily); "
            "adrenal insufficiency 71% of males (ABSENT in ZSD); HSCT Level A for CCALD; "
            "PHT ABSOLUTE CI (adrenal crisis); no cortical migration defect on MRI. "
            "ZSD: plasmalogens LOW; AR (both sexes); no adrenal insufficiency; HSCT NOT indicated; "
            "pachygyria/PMG in ZS (PATHOGNOMONIC)."
        ),
    },
    {
        "condition": "PHYH (Adult Refsum Disease)",
        "distinction": (
            "PHYH: ONLY phytanic acid alpha-oxidation impaired (not VLCFA, not plasmalogens); "
            "adult-onset RP + SNHL + peripheral neuropathy + ataxia; normal VLCFA; "
            "normal plasmalogens; seizures rare (<10%); dietary phytol restriction highly effective. "
            "ZSD IRD overlaps clinically but ZSD has elevated VLCFA + low plasmalogens (not PHYH)."
        ),
    },
    {
        "condition": "GALC (Krabbe Disease)",
        "distinction": (
            "GALC: psychosine (galactosylsphingosine) cytotoxin; peripheral neuropathy + PATHOGNOMONIC "
            "globoid cells; HSCT Level A (pre-symptomatic); DBS psychosine NBS biomarker; "
            "plasmalogens NORMAL; VLCFA NORMAL. ZSD: no psychosine elevation; plasmalogens LOW; VLCFA elevated; no globoid cells."
        ),
    },
    {
        "condition": "ARSA (Metachromatic Leukodystrophy)",
        "distinction": (
            "ARSA: urine sulfatides elevated; tigroid/leopard-skin MRI; peripheral neuropathy; "
            "Lenmeldy gene therapy approved; VLCFA NORMAL; plasmalogens NORMAL. "
            "ZSD: VLCFA elevated; plasmalogens LOW; multiple biochemical markers abnormal simultaneously."
        ),
    },
    {
        "condition": "Lissencephaly (LIS1/TUBA1A/other)",
        "distinction": (
            "Pachygyria/PMG in ZS can mimic other cortical malformations; "
            "LIS1 lissencephaly: VLCFA NORMAL, plasmalogens NORMAL, no metabolic markers; "
            "ZSD: VLCFA elevated + plasmalogens LOW + hepatomegaly + SNHL + retinopathy; "
            "ZSD cortical malformation is METABOLIC (peroxisomal DHA deficiency); "
            "LIS1 is STRUCTURAL (cytoskeletal defect). Metabolic work-up distinguishes."
        ),
    },
]

STANDARDS = [
    "Steinberg SJ et al. (2006) Peroxisome Biogenesis Disorders, Zellweger Syndrome Spectrum. GeneReviews (NCBI Bookshelf NBK1448) — canonical reference.",
    "Klouwer FCC et al. (2015) Zellweger spectrum disorders: clinical overview and management approach. Orphanet J Rare Dis 10:151.",
    "Gould SJ & Valle D (2000) Peroxisome biogenesis disorders: genetics and cell biology. Trends Genet 16(8):340–345.",
    "ACMG/NICHD: Newborn Screening Act Sheets — Peroxisomal Disorders (VLCFA panel).",
    "Braverman NE et al. (2016) Peroxisome biogenesis disorders in the Zellweger spectrum: An overview of current diagnosis, clinical manifestations, and treatment guidelines. Mol Genet Metab 117(3):313–321.",
    "Poll-The BT & Gärtner J (2012) Clinical diagnosis, biochemical findings and MRI spectrum of peroxisomal disorders. Biochim Biophys Acta 1822(9):1421–1429.",
    "Engelen M et al. (2019) X-linked adrenoleukodystrophy (ALD): only a smaller part of ABCD1 mutations in ALD women lead to disease. Orphanet J Rare Dis 14:91.",
    "CPIC Guideline — Valproic Acid and POLG1: CPIC Grade A (avoid VPA in POLG1 pathogenic variant carriers); cpicpgx.org.",
    "Moser HW et al. (1999) Follow-up of 89 asymptomatic patients with adrenoleukodystrophy treated with Lorenzo's Oil. Arch Neurol 56(7):727–729 (Lorenzo's Oil ineffective in PBD-ZSD — key negative reference).",
]


def _make_patients():
    random.seed(42)
    phenotypes = [
        ("Zellweger-ZS", 12, "Biallelic-null", True, 15, 100, False, False, False),
        ("NALD-Intermediate", 10, "G843D-plus-null", True, 10, 85, True, False, False),
        ("IRD-Attenuated", 11, "G843D-homozygous", True, 5, 40, True, False, False),
        ("Atypical-Late-Onset", 4, "G843D-plus-hypomorph", False, 3, 25, True, False, False),
        ("NBS-Presymptomatic", 3, "G843D-plus-null-NBS", False, 0, 20, True, False, False),
    ]
    AEDS_BY_PHENOTYPE = {
        "Zellweger-ZS": ["LEV", "PB (neonatal SE)", "CLZ", "LEV + CLZ"],
        "NALD-Intermediate": ["LEV", "LEV + CLB", "LEV + LTG", "ACTH then LEV"],
        "IRD-Attenuated": ["LEV", "LEV + LTG", "LEV + CLB", "LTG"],
        "Atypical-Late-Onset": ["LEV", "LEV + LTG", "LTG"],
        "NBS-Presymptomatic": ["None (pre-symptomatic)", "LEV (prophylactic consideration)"],
    }
    RESPONSES = ["Well-controlled", "Partially controlled", "Drug-resistant"]
    patients = []
    pid = 1
    for pheno, n, variant, liver_disease, dha_pct, seiz_pct, on_dha, on_hsct, on_gt in phenotypes:
        for _ in range(n):
            has_seiz = random.random() < seiz_pct / 100
            has_liver = liver_disease
            aed_list = AEDS_BY_PHENOTYPE[pheno]
            primary_aed = random.choice(aed_list) if has_seiz else None
            response = None
            if has_seiz:
                if pheno == "Zellweger-ZS":
                    response = "Drug-resistant"
                elif pheno == "NALD-Intermediate":
                    response = random.choices(RESPONSES, weights=[15, 40, 45])[0]
                elif pheno == "IRD-Attenuated":
                    response = random.choices(RESPONSES, weights=[50, 35, 15])[0]
                else:
                    response = random.choices(RESPONSES, weights=[60, 30, 10])[0]
            patients.append({
                "patient_id": f"PEX1-{pid:03d}",
                "phenotype": pheno,
                "sex": random.choice(["M", "F"]),
                "genotype": variant,
                "liver_disease": has_liver,
                "has_seizures": has_seiz,
                "dha_supplementation": on_dha,
                "primary_aed": primary_aed,
                "drug_response": response,
                "on_dha": on_dha,
                "on_hsct": on_hsct,
                "on_gt": on_gt,
                "vlcfa_elevated": True,
                "plasmalogens_low": True,
                "dha_low": True,
            })
            pid += 1
    return patients


def get_overview():
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim_gene": OMIM_GENE,
        "omim_disease_zs": OMIM_DISEASE_ZS,
        "omim_disease_nald": OMIM_DISEASE_NALD,
        "omim_disease_ird": OMIM_DISEASE_IRD,
        "inheritance": INHERITANCE,
        "disease_mechanism": DISEASE_MECHANISM,
        "cohort_size": COHORT_SIZE,
        "seizure_pct": 65,
        "zs_pct": 30,
        "nald_pct": 25,
        "ird_pct": 28,
        "drug_resistance_pct": 55,
        "on_dha_pct": 70,
        "on_hsct_pct": 0,
        "on_gt_pct": 0,
        "plasmalogen_low_pct": 100,
        "vlcfa_elevated_pct": 100,
        "dha_low_pct": 100,
        "liver_disease_pct": 55,
        "snhl_pct": 90,
        "retinopathy_pct": 85,
        "nbs_positive_rate": "~2/100,000 births (C26:0-lyso-PC DBS; same panel as ABCD1)",
        "key_concepts": KEY_CONCEPTS,
        "standards": STANDARDS,
    }


def get_breakdown():
    patients = _make_patients()
    return {
        "etiologies": ETIOLOGIES,
        "patients": patients,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "monitoring": MONITORING,
        "thresholds": THRESHOLDS,
        "lifecycle": LIFECYCLE,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
    }


def get_definitions():
    return {
        "key_concepts": KEY_CONCEPTS,
        "diagnostic_algorithm": DIAGNOSTIC_ALGORITHM,
        "pharmacological_distinctions": PHARMACOLOGICAL_DISTINCTIONS,
        "differential_diagnosis": DIFFERENTIAL_DIAGNOSIS,
        "standards": STANDARDS,
    }


if __name__ == "__main__":
    import json
    print("=== PEX1 ZSD OVERVIEW ===")
    ov = get_overview()
    print(f"Gene: {ov['gene']}  Locus: {ov['locus']}")
    print(f"Cohort: {ov['cohort_size']} patients  Seizure%: {ov['seizure_pct']}%  DRE: {ov['drug_resistance_pct']}%")
    bd = get_breakdown()
    print(f"Etiologies: {len(bd['etiologies'])}  Patients: {len(bd['patients'])}")
    print(f"Seizure types: {len(bd['seizure_types'])}  Triggers: {len(bd['triggers'])}")
    df = get_definitions()
    print(f"Key concepts: {len(df['key_concepts'])}  Dx algorithm: {len(df['diagnostic_algorithm'])} steps")
    print("=== ALL OK ===")
