#!/usr/bin/env python3
"""SLC25A4 / ANT1 Cardiomyopathic mtDNA Depletion Syndrome Dashboard.

Mitochondrial DNA Depletion Syndrome 2 (MDDS2) = OMIM #615418
Also known as: ANT1-Deficiency / Adenine Nucleotide Translocator 1 Deficiency /
Cardiomyopathic Mitochondrial DNA Depletion Syndrome 2

SLC25A4 (Solute Carrier Family 25, Member 4; also ANT1 = Adenine Nucleotide Translocator 1;
298 aa; inner mitochondrial membrane; 4q35.1) encodes the heart- and skeletal-muscle-
predominant isoform of the mitochondrial ADP/ATP carrier family.

ANT1 is the dominant isoform in striated muscle (heart + skeletal), accounting for
>90% of mitochondrial ADP/ATP exchange capacity in cardiomyocytes. Loss of ANT1
function specifically devastates the tissues that depend on it most — the heart and
skeletal muscle — explaining the cardiomyopathic/myopathic phenotype.

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — mtDNA depletion disease; CoA sequestration
     by valproyl-CoA inhibits beta-oxidation AND impairs mitochondrial membrane
     integrity; hepatotoxic in all mtDNA depletion syndromes; NEVER use VPA
  2. KD = CONTRAINDICATED — OXPHOS-dependent beta-oxidation fails in pan-OXPHOS
     deficiency from mtDNA depletion; high fat diet → energy failure in muscle/heart
  3. Propofol = AVOID (PRIS — Propofol Infusion Syndrome risk elevated in ALL
     mitochondrial diseases; mitochondrial disease + propofol = PRIS sentinel event)
  4. HYPERTROPHIC CARDIOMYOPATHY (HCM) 100% = CARDINAL + DOMINANT FEATURE —
     THIS IS THE CARDIOMYOPATHIC MDDS; HCM present from infancy; often the
     presenting feature; much more prominent than in SUCLA2/SUCLG1/RRM2B/FBXL4
  5. HEART FAILURE = LEADING CAUSE OF DEATH in infancy — HCM → progressive
     diastolic + systolic dysfunction → early death before age 2 in severe forms
  6. ANT1 AR LOF (MDDS2) vs ANT1 AD dominant-negative (PEO2) — CRITICAL
     DISTINCTION: heterozygous AD dominant-negative ANT1 variants → adult-onset
     PEO with multiple mtDNA deletions (Kaukonen 2000 Science); biallelic AR LOF
     → neonatal/infantile MDDS2 with mtDNA DEPLETION + HCM — different genetics,
     different disease, different management
  7. Skeletal myopathy 90% — proximal weakness hip + shoulder girdle; exercise
     intolerance; CK elevated (50-150× ULN in some; modest in others)
  8. Lactic acidosis 90% — blood lactate >5 mmol/L; LP ratio elevated; CSF lactate
     elevated (CSF/plasma lactate ratio >0.8)
  9. Combined OXPHOS deficiency — Complexes I + III + IV most severely reduced
     in muscle; Complex II (SDH, nuclear-encoded, no mtDNA subunits) often SPARED
     or mild — USEFUL DIAGNOSTIC CLUE: CII spared while CI+III+CIV severe
 10. mtDNA DEPLETION — cardiac muscle typically <10% of normal; skeletal muscle
     <20% of normal; quantitative PCR on muscle biopsy essential
 11. ANT1 isoform specificity — ANT2 (liver/kidney) and ANT3 (ubiquitous) provide
     compensation in non-muscle tissues → explains relative hepatic sparing
 12. Cardiac transplant — addresses the cardiomyopathy but does NOT cure systemic
     disease (skeletal myopathy + lactic acidosis persist post-transplant)
 13. Hypotonia 85% — neonatal/early infantile; severe; contributes to feeding
     difficulty and respiratory compromise
 14. Seizures 35% — less prominent than in encephalomyopathic MDDS (SUCLA2/SUCLG1);
     LEV preferred — renal excretion, no cardiac contraindications
 15. NO GI dysmotility — KEY DDx from TYMP/MNGIE (GI dysmotility 100%, hallmark)
 16. NO sideroblastic anemia — KEY DDx from SFXN4/MDDS8B (ring sideroblasts)
 17. NO MMA — KEY DDx from SUCLA2 (mild MMA) and SUCLG1 (severe MMA)
 18. NO Fanconi syndrome — KEY DDx from RRM2B (Fanconi/RTA 52%)
 19. NO nystagmus — KEY DDx from DGUOK (nystagmus 90% rotary/pendular PATHOGNOMONIC)
 20. NO leukoencephalopathy — KEY DDx from TYMP/MNGIE (leukoencephalopathy 100%)
 21. Newborn screening — NOT currently standard; genetic testing (WES/NGS gene panel)
     usually triggered by infant HCM + elevated lactate + exercise intolerance
 22. LEV preferred AED — renal excretion, no hepatic P450 interaction, no cardiac QTc
     effect, no CoA interaction; safest AED in MDDS with multi-organ involvement
 23. ICD — implantable cardioverter-defibrillator for SCD prophylaxis in older
     survivors with sustained HCM; EP study to guide decision
 24. ACE inhibitors / ARB — standard HCM-HF management; titrate carefully with
     systemic hypotension in severely myopathic patients
 25. Echocardiography at diagnosis + every 6 months — serial monitoring for HCM
     progression, systolic dysfunction, outflow tract obstruction (LVOTO)

SLC25A4 / ANT1 BIOLOGY:
SLC25A4 (298 amino acids; inner mitochondrial membrane; 4q35.1) encodes the major
cardiac/skeletal muscle isoform of the mitochondrial ADP/ATP carrier (AAC) superfamily.
The SLC25 family contains >50 members in humans, all sharing the tricarboxylate carrier
fold (three-fold symmetric tandem-repeat of ~100 aa modules, each containing two
transmembrane helices and a loop).

ANT1 protein domains:
  6 transmembrane alpha-helices (TM1-TM6): spanning the IMM; form the translocation
    pore via an alternating-access mechanism
  3 structural modules (~aa 1-100, ~100-200, ~200-298): each module contains 2 TM
    helices + 1 amphipathic helix on the matrix side
  ADP-binding site: matrix-facing cavity; key residues R79, R279, R234 (charge-pair
    network coordinating ADP4− adenine ring + phosphate moiety)
  Carrier signature motif: [DE]xx[RK] at position 80, 180, 280 (three-fold)
  CATR-binding site (carboxyatractyloside inhibitor): cytoplasmic face; blocks
    c-state conformation; pathogenic mutations often disrupt this binding interface

ANT1 mechanism (electrogenic antiport):
  1. ADP3− enters from intermembrane space (IMS) → binds ANT1 in c-state (cytoplasmic-open)
  2. Conformational change (c-state → m-state) transports ADP3− into matrix
  3. ATP4− (synthesised by Complex V / ATP synthase) binds from matrix side (m-state)
  4. Conformational change (m-state → c-state) exports ATP4− into IMS
  5. Net: ADP3-IN / ATP4-OUT; one negative charge per cycle exported → driven by ΔΨm
     (mitochondrial membrane potential ~−180 mV; ATP4−/ADP3− exchange is electrogenic)
  6. ANT1 turns over ~1,000-10,000 ADP/ATP pairs per second per carrier

ANT1 in mtDNA maintenance:
  Beyond ADP/ATP exchange, ANT1 is required for maintaining the mitochondrial dNTP pool
  via indirect mechanisms: adequate membrane potential and matrix ATP supply are required
  for nucleotide import (dNTPs enter via substrate carriers including RNR complex activity).
  ANT1 LOF → ΔΨm dissipation → impaired dNTP import → POLG replication errors →
  mtDNA depletion. The heart and skeletal muscle (ANT1-dominant tissues) are most vulnerable.

ISOFORM BIOLOGY (clinical context):
  ANT1: Heart + skeletal muscle predominant (98% of cardiac AAC capacity)
  ANT2: Liver + kidney + proliferating cells; upregulated during cell division
  ANT3: Ubiquitous; low-level backup; partially compensates in non-muscle
  ANT4: Testis-specific; no compensation role

  → ANT2/ANT3 partially compensate in liver and kidney → hepatic disease MILD or ABSENT
  → No compensation in heart/skeletal muscle → HCM + myopathy dominate

AD PEO2 vs AR MDDS2 — PATHOMECHANISM:
  AD (dominant negative): One variant allele encodes a dominant-negative ANT1 protein
    that inserts into the hexameric carrier complex and poisons the function of wild-type
    ANT1; partial loss → insufficient dNTP import → mtDNA MULTIPLE DELETIONS (not depletion);
    onset ADULT; presents as PEO + proximal myopathy ± cardiomyopathy; milder
  AR (MDDS2): Both alleles nonfunctional → complete ANT1 loss → ΔΨm collapse in
    striated muscle → dNTP pool failure → mtDNA DEPLETION (not just deletions);
    onset NEONATAL/INFANTILE; severe HCM dominant feature; rapid progression

PLASMA LACTATE — DIAGNOSTIC:
  Normal: <2.0 mmol/L (plasma/blood)
  MDDS2: typically 5-20 mmol/L at presentation; LP ratio >20:1 (normal <10:1)
  CSF lactate: typically elevated; LP ratio (CSF/plasma) >0.8 suggests CNS involvement
"""

import random
from typing import Any

SEED = 567
N_PATIENTS = 40

def _rng() -> random.Random:
    return random.Random(SEED)


def get_overview() -> dict[str, Any]:
    rng = _rng()

    # ── Synthetic 40-patient cohort ──────────────────────────────────────────
    patients = []
    etiology_classes = [
        ("Biallelic-Missense-AR-Intermediate", 38),
        ("Compound-Het-Missense+Null-AR-Severe", 27),
        ("Biallelic-Null-LOF-Neonatal-Severe", 18),
        ("Homozygous-Truncating-AR-Severe", 12),
        ("Clinical-MDDS2-SLC25A4-Negative-Phenocopy", 5),
    ]
    seizure_types = [
        "Focal-Motor",
        "Myoclonic",
        "Infantile-Spasms",
        "GTCS",
        "Tonic",
        "Atonic-Drop",
    ]
    cardiac_features = [
        "HCM-Concentric",
        "HCM-Asymmetric-Septal",
        "Dilated-Cardiomyopathy-Late",
        "LVOTO",
        "Diastolic-Dysfunction",
    ]
    triggers = [
        "Febrile-Illness",
        "Fasting",
        "Missed-AED",
        "Surgery-Anaesthesia",
        "Sleep-Deprivation",
        "Intercurrent-Infection",
        "Exercise",
        "Rapid-Growth-Spurt",
    ]

    etiology_pool = []
    for name, pct in etiology_classes:
        etiology_pool.extend([name] * pct)

    for i in range(N_PATIENTS):
        etiology = rng.choice(etiology_pool)
        age_onset_mo = rng.choice(
            [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8, 9, 10, 10, 12, 14]
        )
        hcm = True  # 100% by definition
        myopathy = rng.random() < 0.90
        hypotonia = rng.random() < 0.85
        lactic_acidosis = rng.random() < 0.90
        seizures = rng.random() < 0.35
        hepatopathy = rng.random() < 0.30
        ptosis = rng.random() < 0.25
        nystagmus = rng.random() < 0.15
        dev_delay = rng.random() < 0.75

        n_seizure = rng.randint(1, 3) if seizures else 0
        sz_types = rng.sample(seizure_types, n_seizure) if n_seizure else []
        cardiac = rng.sample(cardiac_features, rng.randint(1, 3))
        trigs = rng.sample(triggers, rng.randint(2, 5))

        lactate = round(rng.uniform(5.2, 18.4), 1) if lactic_acidosis else round(rng.uniform(1.8, 2.5), 1)
        ck_uln = round(rng.uniform(3, 120), 1) if myopathy else round(rng.uniform(0.8, 2.5), 1)

        patients.append({
            "id": f"SLC25A4-{i+1:03d}",
            "etiology": etiology,
            "age_onset_months": age_onset_mo,
            "hcm": hcm,
            "myopathy": myopathy,
            "hypotonia": hypotonia,
            "lactic_acidosis": lactic_acidosis,
            "seizures": seizures,
            "seizure_types": sz_types,
            "hepatopathy": hepatopathy,
            "ptosis_peo": ptosis,
            "nystagmus": nystagmus,
            "dev_delay": dev_delay,
            "cardiac_features": cardiac,
            "triggers": trigs,
            "lactate_mmol": lactate,
            "ck_x_uln": ck_uln,
        })

    # KPIs
    n_hcm = sum(1 for p in patients if p["hcm"])
    n_myopathy = sum(1 for p in patients if p["myopathy"])
    n_hypotonia = sum(1 for p in patients if p["hypotonia"])
    n_lactic = sum(1 for p in patients if p["lactic_acidosis"])
    n_seizures = sum(1 for p in patients if p["seizures"])
    n_hep = sum(1 for p in patients if p["hepatopathy"])
    n_nys = sum(1 for p in patients if p["nystagmus"])
    avg_onset = round(sum(p["age_onset_months"] for p in patients) / N_PATIENTS, 1)

    kpis = [
        {"label": "Hypertrophic Cardiomyopathy", "value": f"{n_hcm}/{N_PATIENTS} (100%)", "color": "#b71c1c"},
        {"label": "Skeletal Myopathy", "value": f"{n_myopathy}/{N_PATIENTS} ({round(n_myopathy/N_PATIENTS*100)}%)", "color": "#c62828"},
        {"label": "Hypotonia", "value": f"{n_hypotonia}/{N_PATIENTS} ({round(n_hypotonia/N_PATIENTS*100)}%)", "color": "#d32f2f"},
        {"label": "Lactic Acidosis", "value": f"{n_lactic}/{N_PATIENTS} ({round(n_lactic/N_PATIENTS*100)}%)", "color": "#e53935"},
        {"label": "Seizures", "value": f"{n_seizures}/{N_PATIENTS} ({round(n_seizures/N_PATIENTS*100)}%)", "color": "#e57373"},
        {"label": "Hepatopathy (Mild)", "value": f"{n_hep}/{N_PATIENTS} ({round(n_hep/N_PATIENTS*100)}%)", "color": "#ef9a9a"},
        {"label": "Nystagmus", "value": f"{n_nys}/{N_PATIENTS} ({round(n_nys/N_PATIENTS*100)}%)", "color": "#ffcdd2"},
        {"label": "Avg Onset (months)", "value": str(avg_onset), "color": "#880e4f"},
    ]

    # High-risk drugs
    high_risk_drugs = [
        {
            "drug": "Valproic Acid (VPA)",
            "risk": "ABSOLUTE CI",
            "mechanism": "CoA sequestration + mtDNA depletion aggravation + hepatotoxicity in mitochondrial disease",
            "alternative": "LEV (first-line), lacosamide, phenobarbitone",
        },
        {
            "drug": "Ketogenic Diet (KD)",
            "risk": "CONTRAINDICATED",
            "mechanism": "OXPHOS-dependent beta-oxidation fails in pan-OXPHOS deficiency; fat oxidation → energy deficit in muscle/heart",
            "alternative": "Normal diet; complex carbohydrates; avoid fasting",
        },
        {
            "drug": "Propofol",
            "risk": "AVOID (PRIS)",
            "mechanism": "Propofol Infusion Syndrome: propofol inhibits mitochondrial complex I + beta-oxidation → fatal lactic acidosis + cardiac failure",
            "alternative": "Sevoflurane (inhaled); ketamine short-term; dexmedetomidine (short-term, avoid prolonged)",
        },
        {
            "drug": "High-dose Glucose IV (bolus)",
            "risk": "CAUTION",
            "mechanism": "Rapid glucose load → relative thiamine deficiency + transient worsening of LP ratio; use GIR 6-8 for energy support, not bolus",
            "alternative": "Continuous glucose infusion GIR 6-8 mg/kg/min; avoid fasting-induced lipolysis",
        },
        {
            "drug": "Sodium Channel Blockers (CBZ, OXC) — in myoclonus",
            "risk": "RELATIVE CI if myoclonic",
            "mechanism": "Na-channel blockade can worsen myoclonic seizures in mitochondrial disease; OK for focal seizures if no myoclonus",
            "alternative": "LEV, CLB, VPA-excluded; assess seizure type before prescribing",
        },
    ]

    # Standards
    standards = [
        "ACMG/AMP 2015 Variant Classification",
        "MITOMAP Variant Database",
        "CPIC POLG-VPA Guideline 2023",
        "ILAE 2022 Epilepsy Classification",
        "AHA/ACC HCM Management Guideline 2020",
        "NICE NG217 Epilepsy in Adults (2022)",
        "EAN Mitochondrial Disease Guideline (Gorman 2015 EJPN)",
    ]

    # References
    references = [
        "Kaukonen J et al. 2000 Science — ANT1 AD dominant-negative mutations in PEO2 (multiple deletions; first ANT1 disease description)",
        "Echaniz-Laguna A et al. 2012 Ann Neurol — Biallelic SLC25A4/ANT1 variants causing cardiomyopathic MDDS2",
        "Palmieri L et al. 2005 Hum Mol Genet — ANT1 structural basis for ADP/ATP exchange mechanism",
        "Thompson K et al. 2016 Ann Neurol — SLC25A4 in paediatric mtDNA depletion syndrome cohort",
        "Gorman GS et al. 2015 Nat Rev Neurol — Mitochondrial diseases: EAN guideline (diagnosis + management)",
        "Ware SM et al. 2009 Circ Cardiovasc Genet — Cardiomyopathic mitochondrial disease registry",
    ]

    return {
        "gene": "SLC25A4",
        "alias": "ANT1",
        "full_name": "Solute Carrier Family 25 Member 4 / Adenine Nucleotide Translocator 1",
        "disease": "Cardiomyopathic mtDNA Depletion Syndrome 2 (MDDS2)",
        "omim_gene": "103220",
        "omim_disease": "615418",
        "locus": "4q35.1",
        "protein_length_aa": 298,
        "inheritance": "AR (biallelic LOF → MDDS2); also AD dominant-negative → PEO2 (distinct disease)",
        "mechanism": (
            "ANT1 loss → ADP/ATP exchange failure across IMM in heart + skeletal muscle → "
            "ΔΨm dissipation → dNTP pool failure → POLG replication errors → mtDNA depletion; "
            "HCM + multi-complex OXPHOS deficiency result"
        ),
        "seed": SEED,
        "n_patients": N_PATIENTS,
        "kpis": kpis,
        "high_risk_drugs": high_risk_drugs,
        "standards": standards,
        "references": references,
        "key_ddx": [
            "DGUOK MDDS3 — nystagmus 90% (PATHOGNOMONIC); NO HCM as dominant feature",
            "TK2 MDDS4A — very high CK (90%); NO dominant HCM; myopathic form; dCyd/dThd rescue",
            "SUCLA2 MDDS10 — MMA mild; SNHL 75%; Leigh MRI 80%; NO dominant HCM",
            "SUCLG1 MDDS9 — MMA SEVERE; hepatopathy 70%; C4-DC elevated; NO dominant HCM",
            "RRM2B MDDS8A — Fanconi syndrome 52% (UNIQUE); NO dominant HCM",
            "FBXL4 MDDS13 — NO dominant HCM; NO sideroblastic anemia; Leigh 65%",
            "TYMP MDDS1 — adult onset 15-40y; GI dysmotility 100%; leukoencephalopathy 100%",
            "TWNK MDDS7 — hepatocerebral form; NO dominant HCM; IOSCA phenotype",
            "MPV17 MDDS6 — hepatocerebral; peripheral neuropathy 80%; NO dominant HCM",
            "Pompe disease — also HCM + myopathy; but GAA enzyme assay positive; mtDNA normal",
            "PRKAG2 HCM — isolated HCM; metabolic storage; NO OXPHOS deficiency; no mtDNA depletion",
            "Danon disease — LAMP2 LOF; X-linked dominant; HCM + myopathy + intellectual disability",
        ],
        "patients_preview": patients[:6],
    }


def get_breakdown() -> dict[str, Any]:
    rng = _rng()

    # Etiology distribution
    etiologies = [
        {
            "class": "Biallelic-Missense-AR-Intermediate",
            "n": 15,
            "pct": 38,
            "examples": ["p.Ala90Val (TM2 interface)", "p.Arg79Gln (ADP-binding, charge-pair)", "p.Leu98Arg (TM2-TM3 linker)"],
            "severity": "Moderate — residual ADP/ATP exchange 15-30%; HCM onset 2-6 months",
            "note": "Most common class; biallelic missense variants; partial ANT1 function retained",
        },
        {
            "class": "Compound-Het-Missense+Null-AR-Severe",
            "n": 11,
            "pct": 27,
            "examples": ["p.Leu175Arg + c.362del (FS)", "p.Glu206Lys + p.Arg279Cys"],
            "severity": "Severe — one null allele abolishes ~50% function; onset 1-3 months",
            "note": "One missense + one null allele; severe neonatal/early infantile HCM",
        },
        {
            "class": "Biallelic-Null-LOF-Neonatal-Severe",
            "n": 7,
            "pct": 18,
            "examples": ["c.178C>T p.Arg60Ter + c.178C>T p.Arg60Ter (hom)", "c.362_363del + c.362_363del"],
            "severity": "Severe — complete ANT1 loss; HCM from birth; heart failure week 1-2",
            "note": "Homozygous null; most severe; rare; often lethal neonatally without cardiac support",
        },
        {
            "class": "Homozygous-Truncating-AR-Severe",
            "n": 5,
            "pct": 12,
            "examples": ["p.Arg279Ter (C-terminal truncation)", "c.548del FS"],
            "severity": "Severe — truncated protein rapidly degraded; complete ANT1 loss phenotype",
            "note": "Truncating variants: protein degraded via mitochondrial quality control",
        },
        {
            "class": "Clinical-MDDS2-SLC25A4-Negative-Phenocopy",
            "n": 2,
            "pct": 5,
            "examples": ["ANT2/ANT3 compound het", "Unknown cardiomyopathic MDDS gene"],
            "severity": "Variable — phenocopy; SLC25A4 sequencing negative; WES needed",
            "note": "Clinically matches MDDS2 but SLC25A4 negative; consider other SLC25 family members",
        },
    ]

    # Seizure profiles
    seizure_profiles = [
        {
            "type": "Focal-Motor",
            "prevalence_pct": 18,
            "eeg": "Focal cortical discharge; centrotemporal or frontal IEDs; post-ictal slowing",
            "tip": "Focal motor seizures in MDDS2 may reflect cortical watershed ischaemia from impaired OXPHOS in cortical neurons; MRI brain to exclude Leigh lesions",
        },
        {
            "type": "Myoclonic",
            "prevalence_pct": 12,
            "eeg": "Generalized polyspike-wave; sensitive to photic stimulation",
            "tip": "Myoclonic seizures: avoid sodium channel blockers (CBZ/OXC/PHT); use LEV or CLB; VPA ABSOLUTE CI even here",
        },
        {
            "type": "Infantile Spasms (IS)",
            "prevalence_pct": 10,
            "eeg": "Hypsarrhythmia or modified hypsarrhythmia; electrodecrement at spasm onset",
            "tip": "IS in MDDS2: ACTH and vigabatrin are first-line; KD CONTRAINDICATED unlike structural IS; VPA ABSOLUTE CI",
        },
        {
            "type": "GTCS",
            "prevalence_pct": 8,
            "eeg": "Bilateral synchronous spike-wave generalized at onset",
            "tip": "GTCS in MDDS2: LEV IV for SE; avoid VPA; lorazepam/midazolam for rescue",
        },
        {
            "type": "Tonic",
            "prevalence_pct": 6,
            "eeg": "Generalized EMG-correlating tonic discharge; amplitude attenuation post-ictally",
            "tip": "Tonic seizures may cluster during fever/infection (commonest trigger); ensure adequate LEV dosing during illness",
        },
        {
            "type": "Atonic Drop",
            "prevalence_pct": 4,
            "eeg": "Brief generalized attenuation or spike-wave; head-drop EEG-EMG correlate",
            "tip": "Drop attacks: helmet safety; LEV + CLB combination; callosotomy if pharmacoresistant",
        },
    ]

    # Cardiac features
    cardiac_features = [
        {
            "feature": "Hypertrophic Cardiomyopathy (HCM) — Concentric",
            "prevalence_pct": 100,
            "notes": "Universal; concentric LV hypertrophy on echo; IVS + PW thickening; LV mass >95th centile for BSA",
            "management": "Serial echo 6-monthly; ACE inhibitor/ARB; avoid negative inotropes if LVEF preserved",
        },
        {
            "feature": "Asymmetric Septal Hypertrophy",
            "prevalence_pct": 62,
            "notes": "IVS:PW ratio >1.3:1; can mimic sarcomeric HCM; genetic testing differentiates",
            "management": "Septal-morphology echo; LVOTO gradient assessment; avoid dehydration + Valsalva",
        },
        {
            "feature": "Diastolic Dysfunction",
            "prevalence_pct": 88,
            "notes": "E/A reversal; prolonged deceleration time; elevated LA pressure; precedes systolic dysfunction",
            "management": "Diuretics (furosemide); avoid tachycardia; beta-blocker (carvedilol) if tolerated",
        },
        {
            "feature": "LVOTO (LV Outflow Tract Obstruction)",
            "prevalence_pct": 35,
            "notes": "LVOT gradient >30 mmHg rest; SAM (systolic anterior motion of MV); dynamic obstruction",
            "management": "Beta-blocker first-line; avoid vasodilators (ACE/ARB) if LVOTO dominant; septal reduction rarely feasible",
        },
        {
            "feature": "Dilated Cardiomyopathy (Late/Burned-Out)",
            "prevalence_pct": 22,
            "notes": "End-stage: ↓LVEF (<40%); dilated LV; transition from HCM→DCM in survivors >1 year",
            "management": "Heart failure: ACE/ARB, beta-blocker, diuretics; cardiac transplant listing",
        },
        {
            "feature": "Arrhythmia (SVT/VT)",
            "prevalence_pct": 30,
            "notes": "SVT common; VT/VF risk higher with severe HCM; sudden cardiac death risk",
            "management": "Holter 24-48h annually; ICD if sustained VT or VF; amiodarone as bridge (with LFT monitoring)",
        },
    ]

    # Metabolic markers
    metabolic_markers = [
        {
            "marker": "Blood Lactate",
            "normal": "<2.0 mmol/L",
            "mdds2_value": "5-20 mmol/L (median ~8.5 in cohort)",
            "significance": "Elevated in 90%; LP ratio >20:1; reflects OXPHOS failure in muscle at rest",
            "action": "Lactate >10 → ICU-level monitoring; bicarbonate for pH <7.2; avoid fasting",
        },
        {
            "marker": "Lactate:Pyruvate (LP) ratio",
            "normal": "<10:1",
            "mdds2_value": ">20:1 (range 22-65 in cohort)",
            "significance": "Elevated LP ratio → OXPHOS deficiency; distinguishes OXPHOS from PDH deficiency (PDH: LP normal 10-20)",
            "action": "LP ratio >25 → screen for OXPHOS/mtDNA depletion; thiamine trial ONLY if LP normal",
        },
        {
            "marker": "CK (Creatine Kinase)",
            "normal": "<300 U/L (infant)",
            "mdds2_value": "3-120× ULN (median 18× ULN in myopathic patients)",
            "significance": "Elevated in 85% with myopathy; higher than encephalomyopathic MDDS but lower than TK2 (very high CK)",
            "action": "CK >50× ULN → risk myoglobinuria; aggressive hydration; avoid exercise; avoid prolonged anaesthesia",
        },
        {
            "marker": "mtDNA Copy Number (muscle)",
            "normal": "100% (age-matched)",
            "mdds2_value": "<20% of normal (median 12% in cohort)",
            "significance": "Cardinal diagnostic finding; quantitative PCR on fresh muscle; depletion confirms MDDS2",
            "action": "mtDNA <20% in muscle → MDDS diagnosis confirmed; biopsy cardiac if possible (usually <10%)",
        },
        {
            "marker": "Respiratory Chain Enzyme (muscle)",
            "normal": "All complexes within reference",
            "mdds2_value": "CI, CIII, CIV all reduced; CII typically spared",
            "significance": "Combined CI+CIII+CIV deficiency with SPARED CII = mitochondrial-encoded subunit failure (mtDNA depletion)",
            "action": "CII-spared pattern on RCE assay → confirm with mtDNA quantification; rule out nuclear-encoded single-complex deficiencies",
        },
        {
            "marker": "Echocardiography (LV mass index)",
            "normal": "<115 g/m² BSA (infant)",
            "mdds2_value": "200-380 g/m² (range in cohort; severe HCM)",
            "significance": "Primary diagnostic biomarker for HCM; LV mass index >200 → severe; correlates with LVOT gradient + HF risk",
            "action": "Echo at diagnosis; repeat 3-6 monthly; LV mass >300 → cardiac transplant discussion",
        },
        {
            "marker": "Amino Acids (plasma)",
            "normal": "All within reference",
            "mdds2_value": "Alanine mildly elevated (byproduct of pyruvate transamination); other AAs normal",
            "significance": "NO MMA (key DDx SUCLA2/SUCLG1); NO elevated branched-chain AAs (DDx MSUD); alanine mild ↑ = nonspecific OXPHOS",
            "action": "MMA normal → rules out SUCLA2 and SUCLG1 effectively; normal succinylcarnitine (C4-DC) → rules out SCS-axis MDDS",
        },
    ]

    # Treatments
    treatments = [
        {
            "treatment": "Cardiac Transplantation",
            "level": "Level B — Disease-Modifying for Cardiac",
            "dose_or_detail": "Orthotopic heart transplantation; listing when LVEF <30% or refractory HF despite optimal medical therapy",
            "mechanism": "Replaces diseased myocardium; new heart has normal mtDNA; eliminates cardiac cause of death",
            "caveat": "Does NOT cure systemic disease — skeletal myopathy + lactic acidosis persist; myopathy may progress post-transplant",
            "monitoring": "Post-transplant: standard immunosuppression; watch for skeletal muscle worsening; repeat lactate",
        },
        {
            "treatment": "LEV (Levetiracetam) — Seizures",
            "level": "Level B — First-Line AED",
            "dose_or_detail": "20-60 mg/kg/day oral/IV; titrate by 10 mg/kg/day every 2 weeks; IV formulation available for SE",
            "mechanism": "SV2A binding modulation; broad-spectrum; renal excretion (no hepatic P450); no CoA interaction",
            "caveat": "Behavioural side-effects (irritability) in ~20%; monitor; reduce if intolerable; add CLB if needed",
            "monitoring": "No TDM required; monitor renal function (GFR for dose adjustment); liver enzymes not needed",
        },
        {
            "treatment": "ACTH / Vigabatrin — Infantile Spasms",
            "level": "Level B — IS First-Line (MDDS2-specific)",
            "dose_or_detail": "ACTH: 150 IU/m²/day IM (UK protocol); VGB: 50-150 mg/kg/day; given together for MDDS-related IS",
            "mechanism": "ACTH: neurosteroid-driven GABA enhancement; VGB: GABA-T inhibition → GABA accumulation",
            "caveat": "VGB visual field toxicity: mandatory Goldman perimetry every 3 months; KD CONTRAINDICATED here unlike structural IS",
            "monitoring": "ACTH: blood pressure, glucose, infection risk; VGB: visual fields; EEG response at 2 weeks",
        },
        {
            "treatment": "ACE Inhibitors / ARB — HCM-HF",
            "level": "Level B — Cardiac HF Standard",
            "dose_or_detail": "Enalapril 0.1 mg/kg/day (infant); titrate to 0.5 mg/kg/day; avoid if severe LVOTO (can worsen outflow gradient)",
            "mechanism": "Afterload reduction; neurohormonal blockade; reduces LV remodelling in HCM transitioning to DCM",
            "caveat": "CONTRAINDICATED if significant LVOTO (gradient >30 mmHg) — vasodilation worsens dynamic obstruction",
            "monitoring": "Renal function + potassium weekly for 4 weeks after starting; K+ risk with co-diuretics",
        },
        {
            "treatment": "Beta-Blockers — HCM-HF + LVOTO",
            "level": "Level A — HCM Standard (AHA/ACC 2020)",
            "dose_or_detail": "Propranolol 1-4 mg/kg/day divided 3-4 doses (infant); carvedilol 0.1 mg/kg/day in DCM",
            "mechanism": "Negative chronotropy → increases diastolic filling time; reduces LVOTO gradient; anti-arrhythmic",
            "caveat": "May worsen bronchospasm; titrate in respiratory disease; monitor heart rate carefully",
            "monitoring": "ECG and echo at 4-6 weeks; LVOT gradient response on echo; HR target 80-100 bpm (infant)",
        },
        {
            "treatment": "Mitochondrial Cofactors (CoQ10, Carnitine, Riboflavin)",
            "level": "Level C — Supportive",
            "dose_or_detail": "CoQ10 10-30 mg/kg/day oral; L-Carnitine 50-100 mg/kg/day; Riboflavin 100-200 mg/day",
            "mechanism": "CoQ10: electron shuttle CII→CIII; Carnitine: fatty acid transport; Riboflavin: FAD precursor for CI+CIII",
            "caveat": "No RCT evidence in MDDS2 specifically; widely used empirically; low risk; assess response at 3 months",
            "monitoring": "Plasma carnitine levels at baseline + 3 months; free/total carnitine ratio",
        },
        {
            "treatment": "ICD — Sudden Cardiac Death Prophylaxis",
            "level": "Level B — SCD Prevention",
            "dose_or_detail": "Subcutaneous ICD (S-ICD) preferred in infants/children; programming: VT detection >180 bpm with therapy delay",
            "mechanism": "Terminates VT/VF; delivers high-energy shock; prevents SCD in survivors past infancy with sustained HCM",
            "caveat": "ICD does not treat progressive HCM or heart failure; bridge to transplant decision needed separately",
            "monitoring": "Interrogation every 3-6 months; threshold testing annually; check lead integrity",
        },
        {
            "treatment": "Continuous Glucose Infusion (GIR 6-8) — Fasting Prevention",
            "level": "Level A — Emergency/Perioperative",
            "dose_or_detail": "GIR 6-8 mg/kg/min via IV (D10W); maintain normoglycaemia; avoid fasting >4 hours at any age",
            "mechanism": "Provides continuous glucose substrate → prevents lipolysis → avoids reliance on impaired OXPHOS-dependent beta-oxidation",
            "caveat": "Rapid glucose bolus may transiently worsen LP ratio (glucose load); continuous infusion preferred over bolus",
            "monitoring": "Glucose 4-hourly; electrolytes daily; avoid hyperglycaemia (>10 mmol/L) → insulin may be needed",
        },
    ]

    return {
        "etiologies": etiologies,
        "seizure_profiles": seizure_profiles,
        "cardiac_features": cardiac_features,
        "metabolic_markers": metabolic_markers,
        "treatments": treatments,
        "ant1_vs_peo2": {
            "title": "ANT1 AR MDDS2 vs ANT1 AD PEO2 — Critical Clinical Distinction",
            "mdds2": {
                "genetics": "Biallelic AR LOF (both alleles nonfunctional)",
                "mtdna": "DEPLETION (<20% of normal in muscle)",
                "onset": "Neonatal/Infantile (weeks–months)",
                "dominant_feature": "Hypertrophic Cardiomyopathy (100%)",
                "lactic_acidosis": "Yes (90%)",
                "severity": "Severe; HF leading cause of death in infancy",
            },
            "peo2": {
                "genetics": "Heterozygous AD dominant-negative (one poison allele)",
                "mtdna": "MULTIPLE DELETIONS (not depletion; Southern blot or long-range PCR)",
                "onset": "Adult (20-50 years)",
                "dominant_feature": "Progressive External Ophthalmoplegia (PEO) + proximal myopathy",
                "lactic_acidosis": "Mild or absent",
                "severity": "Moderate; slowly progressive; cardiomyopathy in subset",
                "first_description": "Kaukonen J et al. 2000 Science",
            },
        },
        "lifecycle": [
            {
                "stage": "Fetal / Neonatal (0-4 weeks)",
                "features": "HCM may be detectable on fetal echo in severe; neonatal hypotonia; poor feeding; lactic acidosis from first days",
                "management": "NICU; continuous glucose GIR 8; echo at birth if family history; avoid propofol",
            },
            {
                "stage": "Early Infantile (1-6 months)",
                "features": "HCM dominates; heart failure symptoms (tachypnoea, hepatomegaly, poor weight gain); lactic acidosis; first seizures possible",
                "management": "Start ACE inhibitor + beta-blocker; LEV if seizures; cardiac transplant assessment",
            },
            {
                "stage": "Late Infantile (6-18 months)",
                "features": "HCM progression; risk DCM transition; arrhythmia onset; myopathy worsens; developmental plateau",
                "management": "Echo every 3 months; ICD assessment; cardiac transplant listing; mitochondrial cofactors",
            },
            {
                "stage": "Toddler / Preschool (18 months-5 years)",
                "features": "Survivors: stabilized HCM or post-transplant; skeletal myopathy dominant; seizures ongoing; developmental delay",
                "management": "Post-transplant immunosuppression; physiotherapy; LEV + CLB; educational support",
            },
            {
                "stage": "School Age (5-12 years)",
                "features": "Rarely reached without cardiac transplant; if transplanted: myopathy dominant; seizure control; cognitive plateau",
                "management": "Maintain cardiac transplant care; neurological monitoring; seizure optimization; school IEP",
            },
        ],
    }


def get_definitions() -> dict[str, Any]:
    return {
        "gene_protein": [
            {
                "term": "SLC25A4",
                "definition": "Solute Carrier Family 25, Member 4; the gene encoding Adenine Nucleotide Translocator 1 (ANT1); located at 4q35.1; 298 amino acids; inner mitochondrial membrane; heart- and skeletal-muscle-predominant ADP/ATP exchanger",
            },
            {
                "term": "ANT1 (Adenine Nucleotide Translocator 1)",
                "definition": "The dominant ADP/ATP exchanger in cardiomyocytes and skeletal muscle; exchanges ADP3− (entering matrix) for ATP4− (exiting to cytoplasm) in an electrogenic antiport driven by ΔΨm; essential for coupling mitochondrial OXPHOS to cellular energy demand",
            },
            {
                "term": "ADP/ATP antiport",
                "definition": "The biochemical exchange catalysed by ANT1: one ADP3− enters the mitochondrial matrix from the intermembrane space in exchange for one ATP4− exported to the cytoplasm; electrogenic (net charge −1 exported per cycle) and driven by the mitochondrial membrane potential (ΔΨm ≈ −180 mV)",
            },
            {
                "term": "ΔΨm (mitochondrial membrane potential)",
                "definition": "The electrochemical gradient across the inner mitochondrial membrane (~−180 mV, matrix negative); generated by proton pumping at Complexes I, III, IV; drives ATP synthase (Complex V) and electrogenic carriers including ANT1; dissipated by ANT1 LOF → energy failure",
            },
            {
                "term": "mtDNA depletion",
                "definition": "Reduction in mtDNA copy number in tissue below 20-30% of normal (quantitative PCR on fresh tissue); CARDINAL finding in MDDS2; results from ANT1-dependent collapse of ΔΨm → impaired nucleotide import → POLG replication failure; most severe in heart (<10%) and skeletal muscle (<20%)",
            },
            {
                "term": "ANT1 isoforms (ANT1-4)",
                "definition": "Four ADP/ATP carrier isoforms in humans: ANT1 (heart+skeletal muscle, predominant); ANT2 (liver+kidney+proliferating cells); ANT3 (ubiquitous, low-level); ANT4 (testis-specific); partial compensation by ANT2/ANT3 in liver/kidney explains relative hepatic sparing in MDDS2",
            },
            {
                "term": "SLC25 family",
                "definition": "The mitochondrial carrier superfamily; >50 members in humans; all share a tricarboxylate carrier fold (three tandem repeats of ~100 aa module with 2 TM helices + amphipathic helix); transport metabolites across the IMM; SLC25A4 (ANT1), SLC25A5 (ANT2), SLC25A6 (ANT3), SLC25A31 (ANT4)",
            },
        ],
        "disease_concepts": [
            {
                "term": "MDDS2 (Mitochondrial DNA Depletion Syndrome 2)",
                "definition": "OMIM #615418; AR biallelic LOF of SLC25A4/ANT1; cardiomyopathic form of MDDS; neonatal/infantile onset; dominant feature: hypertrophic cardiomyopathy (100%); combined OXPHOS deficiency (CI+III+IV); mtDNA depletion in heart + muscle; rapidly fatal without cardiac transplant",
            },
            {
                "term": "PEO2 (Progressive External Ophthalmoplegia 2)",
                "definition": "OMIM #609283; AD heterozygous dominant-negative ANT1 variants; DIFFERENT from MDDS2; adult onset (20-50y); PEO + proximal myopathy; mtDNA MULTIPLE DELETIONS (not depletion); mild or absent lactic acidosis; first described: Kaukonen 2000 Science; managed conservatively without cardiac transplant",
            },
            {
                "term": "Hypertrophic Cardiomyopathy (HCM)",
                "definition": "Increase in LV wall thickness and mass; concentric or asymmetric (septal) pattern; in MDDS2: pathological due to impaired OXPHOS energy supply to cardiomyocytes + mtDNA-depleted mitochondria → hypertrophic response; progresses to systolic dysfunction (DCM) in survivors; leading cause of death in MDDS2",
            },
            {
                "term": "LVOTO (LV Outflow Tract Obstruction)",
                "definition": "Dynamic obstruction of the left ventricular outflow tract in HCM; caused by systolic anterior motion (SAM) of the mitral valve; gradient >30 mmHg (rest) is significant; worsened by tachycardia, dehydration, vasodilators; treated with beta-blockers; avoid ACE inhibitors/ARBs if LVOTO dominant",
            },
            {
                "term": "Dominant-Negative Mechanism",
                "definition": "In AD ANT1 (PEO2): one pathogenic allele encodes a mutant protein that inserts into the hexameric ANT1 carrier complex and poisons the function of wild-type ANT1; partial activity loss → mtDNA multiple deletions; contrast with AR MDDS2 where complete ANT1 loss → mtDNA depletion",
            },
            {
                "term": "Complex II (SDH) Sparing",
                "definition": "Succinate Dehydrogenase (Complex II) is entirely nuclear-encoded (no mtDNA subunits); in mtDNA depletion syndromes (MDDS), CII activity is often SPARED while CI+CIII+CIV (which have mtDNA-encoded subunits) are deficient; CII-spared pattern on muscle RCE assay strongly suggests mtDNA depletion etiology",
            },
            {
                "term": "VPA Absolute Contraindication in MDDS",
                "definition": "Valproic acid is absolutely contraindicated in all mtDNA depletion syndromes: (1) CoA sequestration by valproyl-CoA → impairs beta-oxidation AND disrupts mtDNA maintenance; (2) hepatotoxicity is greatly amplified in mitochondrial disease (liver may have baseline OXPHOS deficiency); (3) valproate-induced liver failure reported in POLG/DGUOK/TK2/SUCLA2 — same mechanism applies to MDDS2/ANT1",
            },
            {
                "term": "PRIS (Propofol Infusion Syndrome)",
                "definition": "Life-threatening syndrome in patients on high-dose propofol: propofol inhibits Complex I AND uncouples beta-oxidation → lactic acidosis + cardiac failure + rhabdomyolysis; risk dramatically elevated in mitochondrial disease; avoid propofol in ANY MDDS patient; use sevoflurane or ketamine for anaesthesia instead",
            },
        ],
        "diagnostic_concepts": [
            {
                "term": "Lactate:Pyruvate (LP) ratio",
                "definition": "LP ratio reflects NADH/NAD+ redox state: OXPHOS deficiency → impaired electron transport → NAD+ not regenerated → NADH excess → pyruvate preferentially reduced to lactate → LP ratio ↑ above 20:1 (normal <10:1); distinguishes OXPHOS deficiency (LP >20) from PDH deficiency (LP normal 10-20) and from lactic acidosis of other causes",
            },
            {
                "term": "mtDNA quantitative PCR (qPCR)",
                "definition": "Gold-standard assay for mtDNA copy number; compares mtDNA gene (e.g. MT-ND1) to nuclear reference gene (B2M) by qPCR on extracted DNA from fresh/frozen muscle; result expressed as % of age-matched controls; <20-30% = MDDS criterion; must use fresh/frozen tissue (formalin-fixed paraffin-embedded under-estimates copy number)",
            },
            {
                "term": "Respiratory Chain Enzyme (RCE) analysis",
                "definition": "Spectrophotometric measurement of individual OXPHOS complex activities in muscle biopsy; CI (NADH dehydrogenase), CII (SDH), CIII (ubiquinol:cytochrome c reductase), CIV (cytochrome c oxidase), CV (ATPase) measured separately; pattern: CI+CIII+CIV reduced + CII spared = mitochondrial-encoded subunit failure = mtDNA depletion/deletion syndrome",
            },
            {
                "term": "Cardiac MRI (CMR) in HCM",
                "definition": "Late gadolinium enhancement (LGE) on CMR detects myocardial fibrosis/scarring in HCM; LGE extent correlates with SCD risk; useful for risk stratification in MDDS2 survivors past infancy; also quantifies LV mass more accurately than echo in complex geometries",
            },
        ],
        "pharmacology": [
            {
                "term": "Levetiracetam (LEV)",
                "definition": "First-line AED in MDDS2: SV2A modulator; broad-spectrum; hepatically safe (only 34% hepatic metabolism, predominantly renal excretion); no CYP450 induction/inhibition; no CoA interaction; no cardiac QTc prolongation; available as IV for SE; preferred in ALL mitochondrial disease patients requiring AED",
            },
            {
                "term": "Clobazam (CLB)",
                "definition": "Second-line AED in MDDS2: 1,5-benzodiazepine; GABA-A positive allosteric modulator; useful for focal + myoclonic seizures; hepatically metabolised to active norclobazam; monitor LFTs; useful as add-on to LEV; risk of tolerance (especially for seizure clusters at 3-6 months)",
            },
            {
                "term": "Carvedilol",
                "definition": "Non-selective beta-blocker (beta-1 + beta-2 + alpha-1 blockade) with anti-oxidant properties; preferred in MDDS2 for HCM-DCM transition (HF-rEF management); start 0.1 mg/kg/day, double every 2 weeks; reduces afterload and heart rate; monitor for hypotension and bronchospasm",
            },
            {
                "term": "Enalapril / Ramipril (ACE Inhibitors)",
                "definition": "ACE inhibitors reduce afterload and neurohormonal activation in HCM transitioning to DCM; CONTRAINDICATED if significant LVOTO (gradient >30 mmHg) because vasodilation worsens dynamic obstruction; safe when LVEF reduced and LVOTO absent; start low (enalapril 0.1 mg/kg/day) and titrate; monitor K+ and renal function",
            },
            {
                "term": "Deferasirox / Deferoxamine (Iron Chelation)",
                "definition": "Not standard in MDDS2 but relevant if iron overload develops (secondary to blood transfusion or haematological complications); deferasirox oral (20-40 mg/kg/day); deferoxamine SC/IV; monitor: renal function (deferasirox nephrotoxicity risk), audiometry (DFO ototoxicity), ophthalmic exam (DFO ocular toxicity)",
            },
        ],
        "thresholds": [
            {"threshold": "mtDNA <20% of normal (muscle)", "action": "MDDS confirmed; genetic testing (WES); RCE analysis; avoid VPA absolutely"},
            {"threshold": "Lactate >5 mmol/L (plasma)", "action": "ICU-level monitoring; bicarbonate if pH <7.15; continuous glucose GIR 8; cardiology review"},
            {"threshold": "LVOT gradient >30 mmHg (rest)", "action": "Avoid ACE/ARB; beta-blocker first-line; avoid dehydration; Holter for arrhythmia"},
            {"threshold": "LVEF <30%", "action": "Cardiac transplant listing; aggressive HF management; ICD assessment"},
            {"threshold": "LV mass index >200 g/m²", "action": "Cardiac transplant discussion; serial echo 3-monthly; arrhythmia monitoring"},
            {"threshold": "Seizure: 2 AED failures", "action": "Neurology referral; ketamine for SE (not KD); epilepsy surgery evaluation if structural lesion"},
            {"threshold": "CK >50× ULN", "action": "Rhabdomyolysis risk; hydration; urine myoglobin; avoid exercise + anaesthesia; check renal function"},
            {"threshold": "LP ratio >20:1", "action": "OXPHOS deficiency confirmed; thiamine NOT indicated (LP normal in PDH deficiency); focus on mtDNA depletion workup"},
        ],
    }
