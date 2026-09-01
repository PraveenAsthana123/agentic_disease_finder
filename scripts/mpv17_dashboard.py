#!/usr/bin/env python3
"""MPV17 Hepatocerebral mtDNA Depletion Syndrome Dashboard.

Mitochondrial DNA Depletion Syndrome 6 (MDDS6) = OMIM #256810.
Also known as Navajo Neurohepatopathy (NNH) in the Navajo population.
Biallelic AR MPV17 mutations → mtDNA depletion (liver > brain) → OXPHOS failure.

MPV17 (176 aa, 2p23.3) is an inner mitochondrial membrane channel protein
required for maintenance of mtDNA copy number in mitochondria of high-energy tissues.
The exact mechanism is debated: MPV17 may function as a dNTP channel importing
cytoplasmic pyrimidine deoxyribonucleosides into the mitochondrial matrix (supporting
mtDNA replication), OR it may stabilise IMM potential/integrity required for dNTP
uptake. Loss of function → progressive mtDNA depletion → OXPHOS failure.

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — same mechanism as POLG/DGUOK: mtDNA depletion +
     CoA sequestration + epoxide hepatotoxicity; never prescribe in any mtDNA depletion
  2. KD = CONTRAINDICATED — high-fat β-oxidation requires OXPHOS; mtDNA depletion fails it
  3. Navajo founder mutation: p.Arg50Gln (c.149G>A) — ~75% of Navajo alleles; homozygous
     in most Navajo patients; extreme founder effect in Navajo Nation population
  4. Peripheral neuropathy (80%) — sensorimotor demyelinating/axonal; PROMINENT feature
     distinguishing MPV17 from DGUOK (where nystagmus dominates)
  5. NO 3-MGA-uria — critical DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB (all have 3-MGA)
  6. NO nystagmus (prominent nystagmus = favour DGUOK not MPV17)
  7. Liver transplant: CURATIVE in hepatic-only form; does NOT prevent neurological progression
     in hepatocerebral form — brain mtDNA depletion continues post-transplant
  8. LEV preferred AED — renal excretion; no hepatic metabolism; no mito toxicity
  9. Propofol AVOID — PRIS in mitochondrial disease
 10. Spinazzola et al. 2006 Nature Genetics — first MPV17 mutations described

MPV17 BIOLOGY:
MPV17 (176 amino acids, 2p23.3) is a small integral inner mitochondrial membrane (IMM)
protein. It belongs to the Mpv17/PMP22 protein family, which forms ion/metabolite channels.

Domain architecture:
  TM1 (aa14-36): first transmembrane helix; anchors in IMM
  TM2 (aa55-77): second TM helix; channel-lining
  TM3 (aa95-120): third TM helix
  TM4 (aa130-155): fourth TM helix; forms channel exit
  Channel pore region: Arg50 (between TM1 and TM2) lines the extramembrane loop
    p.Arg50Gln: loss of positive charge in pore loop → channel conductance reduced 90%
    This is the Navajo founder mutation affecting 75% of Navajo alleles
  Non-Navajo variants: p.Pro98Leu (European), splice site c.293-1G>C, deletion exon 3-7

Why MPV17 loss causes mtDNA depletion:
  Proposed mechanism 1 — dNTP channel: MPV17 imports mitochondrial dNTPs (pyrimidines
    or purines) from IMS → matrix; LOF → dNTP shortage → POLG stalls → mtDNA depletion
  Proposed mechanism 2 — IMM integrity: MPV17 stabilises IMM potential/permeability;
    LOF → altered IMM → secondary impairment of nucleotide transporter activity
  Result regardless of mechanism: mtDNA copy number falls <30% in liver/brain →
  13 mtDNA-encoded OXPHOS subunits insufficient → Complexes I/III/IV/V fail

Navajo Neurohepatopathy:
  Navajo-specific founder mutation (p.Arg50Gln homozygous) produces a severe hepatocerebral
  syndrome described in the Navajo people of the American Southwest: cirrhosis, hepatic failure,
  peripheral neuropathy, leukoencephalopathy, failure to thrive, and recurrent metabolic crises.
  Estimated prevalence: 1/1,600 Navajo live births (extremely high among Navajo).

PATHOGENIC VARIANT DISTRIBUTION (biallelic AR, n=40, seed-551):
  p.Arg50Gln/p.Arg50Gln homozygous (Navajo founder): ~35% — hepatocerebral, severe neuropathy
  p.Arg50Gln/loss-of-function compound het (Navajo non-canonical): ~15% — hepatocerebral
  p.Pro98Leu/splice compound het (non-Navajo European): ~20% — hepatocerebral/hepatic
  c.293-1G>C splice/missense compound het (non-Navajo): ~15% — hepatocerebral
  Exon 3-7 deletion/missense (non-Navajo): ~10% — hepatocerebral
  Other missense/missense (mild, hepatic-only genotype): ~5% — hepatic-only phenotype
"""

import random
from datetime import date

SEED = 551  # 40-patient cohort seed


def get_overview() -> dict:
    """MPV17 Hepatocerebral mtDNA Depletion — overview for /api/mpv17/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": "Mitochondrial DNA Depletion Syndrome 6 (MDDS6) / MPV17-Related Hepatocerebral Syndrome / Navajo Neurohepatopathy (NNH)",
        "gene": "MPV17; Inner Mitochondrial Membrane Channel Protein; mtDNA Copy-Number Maintenance; 176 aa (4 TM helices); IMM integral protein",
        "chromosome": "2p23.3",
        "omim_gene": "137960",
        "omim_disease": "256810",
        "inheritance": "Autosomal Recessive (biallelic MPV17); no carrier phenotype in heterozygotes",
        "prevalence": "Rare globally; extreme founder effect in Navajo Nation (~1:1,600 Navajo live births); non-Navajo cases sporadic worldwide",
        "protein": "MPV17 176 aa; 4 TM helices (TM1-TM4); IMM channel; Arg50 channel pore residue (Navajo founder p.Arg50Gln); member Mpv17/PMP22 channel family",
        "category": "mtDNA Depletion Syndrome / Mitochondrial DNA Maintenance / MPV17 IMM Channel Disorder / Navajo Neurohepatopathy",
        "first_described": "Spinazzola A et al. 2006 Nature Genetics 38:570-575 — first MPV17 mutations in hepatocerebral mtDNA depletion",
        "kpis": {
            "hepatic_failure_pct": 100,
            "lactic_acidosis_pct": 100,
            "peripheral_neuropathy_pct": 80,
            "hypoglycemia_pct": 80,
            "hypotonia_pct": 75,
            "regression_pct": 70,
            "epilepsy_pct": 40,
            "nystagmus_pct": 30,
            "vpa_risk": "ABSOLUTE CONTRAINDICATION — mtDNA depletion + CoA sequestration + epoxide hepatotoxicity",
            "navajo_founder_pct": 50,
            "hepatic_only_form_pct": 10,
            "no_3mga_uria": "ABSENT — KEY DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB",
        },
        "clinical_highlights": [
            "VPA (Valproate) = ABSOLUTE CONTRAINDICATION — same mechanism as POLG/DGUOK: (1) VPA metabolites impair mitochondrial dNTP availability in already-depleted hepatocytes; (2) CoA sequestration via propionyl-CoA → FAO collapse; (3) VPA 4-en-epoxide → direct hepatocellular necrosis. In MPV17, hepatocytes already at <30% mtDNA — VPA causes precipitous irreversible liver failure. No safe dose in any mtDNA depletion syndrome.",
            "NAVAJO FOUNDER MUTATION p.Arg50Gln (c.149G>A) — unique to Navajo Nation: ~75% of Navajo MPV17 alleles carry this single founder variant; homozygosity produces Navajo Neurohepatopathy (NNH), the severe hepatocerebral form; Navajo children with cirrhosis + neuropathy → MPV17 sequencing MANDATORY. Non-Navajo patients carry European/Asian private variants (p.Pro98Leu, splice c.293-1G>C, exon 3-7 deletion).",
            "PERIPHERAL NEUROPATHY (80%) — DISTINGUISHING from DGUOK: sensorimotor neuropathy (demyelinating or mixed) is the cardinal neurological feature of MPV17 that is absent or minor in DGUOK; NCS/EMG: reduced conduction velocity + reduced CMAP/SNAP amplitudes; peripheral nerves have high mtDNA turnover and are particularly sensitive to MPV17 LOF; neuropathy precedes or parallels liver disease in NNH.",
            "NO NYSTAGMUS — critical DDx from DGUOK: nystagmus (the pathognomonic DGUOK sign, 90%) is absent or minor in MPV17 (30%); any infant with liver failure + peripheral neuropathy (without nystagmus) → MPV17 FIRST; any infant with nystagmus + liver failure → DGUOK FIRST",
            "NO 3-MGA-uria — excludes all 3-MGA syndromes: MPV17 does NOT produce 3-methylglutaconic aciduria (unlike SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB); urine organic acids in MPV17 = elevated lactic acid only; absence of 3-MGA is the fastest DDx test to separate MPV17/DGUOK/POLG from 3-MGA syndromes",
            "LIVER TRANSPLANT — hepatic-only (10%): rare mild genotype with hepatic-predominant disease and preserved CNS; OLT may be curative. HEPATOCEREBRAL (90%): OLT corrects hepatic disease but NOT brain mtDNA depletion — peripheral neuropathy + regression continue post-OLT; experienced centres generally do NOT recommend OLT in hepatocerebral form",
            "LEUKOENCEPHALOPATHY in NNH (60%): MRI shows diffuse white-matter signal abnormality (T2 hyperintensity) in NNH hepatocerebral form — reflects myelin loss from OXPHOS failure in oligodendrocytes; DGUOK more posterior-fossa predominant (cerebellar vermis); MPV17 more diffuse WM + peripheral neuropathy = combined central + peripheral demyelination",
            "KD = CONTRAINDICATED — high-fat β-oxidation requires OXPHOS; MPV17-mediated mtDNA depletion makes electron transport chain insufficient to handle ketone + fatty acid substrate; KD worsens lactic acidosis; never use in any mtDNA depletion syndrome",
        ],
        "contraindications": [
            {"drug": "Valproate (VPA, Depakene, Epilim)", "level": "ABSOLUTE CONTRAINDICATION — Fatal hepatotoxicity in mtDNA depletion", "reason": "MPV17 LOF → <30% mtDNA in hepatocytes → residual OXPHOS is the only energy source; VPA: (1) propionyl-CoA sequestration → CoA depletion → FAO+TCA failure; (2) 4-en-VPA epoxide → direct hepatocellular necrosis; (3) VPA metabolites suppress residual electron transport. In pre-depleted hepatocytes, VPA causes precipitous, irreversible, fatal liver failure. No dose reduction strategy is safe — absolute ban in all genotype-confirmed or suspected MPV17 patients."},
            {"drug": "Ketogenic Diet (KD)", "level": "CONTRAINDICATED — Forces OXPHOS-dependent fat oxidation that fails in mtDNA depletion", "reason": "KD shifts substrate to fatty acid β-oxidation + ketone body utilisation. OXPHOS (Complex I-V) is required for mitochondrial fat oxidation: FADH2 from β-oxidation enters Complex II; NADH enters Complex I; ATP synthase (Complex V) regenerates ATP. In MPV17, <30% mtDNA → Complex I/III/IV/V severely deficient → KD substrate cannot be oxidised → ketoacidosis + worsening lactic acidosis + acute metabolic decompensation. KD is equivalent to forcing a car engine to run without engine oil — the substrate arrives but cannot be processed."},
            {"drug": "Propofol", "level": "AVOID — Propofol Infusion Syndrome (PRIS) risk in mitochondrial disease", "reason": "Propofol inhibits Complex I of the electron transport chain (the Q-binding site). In MPV17 patients with already-depleted Complex I (from low mtDNA), even brief propofol infusion can trigger PRIS: lactic acidosis, rhabdomyolysis, cardiac arrhythmia, cardiac failure, death. Alternative anaesthesia: ketamine + sevoflurane for induction; avoid propofol in TIVA. Document in allergy alerts."},
            {"drug": "Prolonged Fasting / High-Fat Perioperative Feeds", "level": "CONTRAINDICATED — Catabolism triggers metabolic crisis in depleted OXPHOS", "reason": "Fasting → glycogen depletion → hepatic gluconeogenesis required → gluconeogenesis requires NADH + ATP regeneration (OXPHOS-dependent) → depleted in MPV17 → hypoglycemia + lactic acidosis. Perioperative: IV 10% dextrose from midnight before surgery; GIR 8-10 mg/kg/min continuous; 2-hourly glucose monitoring; avoid NG high-fat boluses. High-fat enteral feeds worsen OXPHOS load."},
        ],
        "thresholds": [
            {"marker": "Lactate (plasma)", "cutoff": ">10 mmol/L neonatal; >4 mmol/L chronic", "interpretation": "Severe OXPHOS failure; >10 neonatal = metabolic crisis; lactate:pyruvate >20:1 confirms mitochondrial origin"},
            {"marker": "pH (arterial)", "cutoff": "<7.1", "interpretation": "Severe metabolic acidosis from lactic acid; bicarbonate if pH <7.1; IV dextrose is primary treatment"},
            {"marker": "mtDNA copy number (liver biopsy)", "cutoff": "<30% of controls", "interpretation": "Diagnostic for mpv17 depletion; qPCR ND1 probe; correlates with residual OXPHOS; <10% = most severe"},
            {"marker": "DGUOK residual activity", "cutoff": "N/A — test MPV17 channel function indirectly via mtDNA copy number", "interpretation": "No direct MPV17 enzymatic assay; diagnosis by sequencing + mtDNA qPCR + respiratory chain enzymology (BN-PAGE)"},
            {"marker": "ALT/AST", "cutoff": ">10× ULN", "interpretation": "Hepatocellular injury; AST > ALT in mitochondrial hepatopathy (mitochondria-dense hepatocytes); monitor 2-weekly"},
            {"marker": "INR", "cutoff": ">1.5", "interpretation": "Trigger OLT evaluation in hepatic-only form; synthetic liver failure; correct with FFP before invasive procedures"},
            {"marker": "Blood glucose", "cutoff": "<3.5 mmol/L", "interpretation": "Hypoglycemia from failed hepatic gluconeogenesis; treat with IV 10% dextrose bolus 2 mL/kg; maintain 5-10 mmol/L"},
            {"marker": "Nerve conduction velocity (NCV)", "cutoff": "<40 m/s (motor)", "interpretation": "Demyelinating neuropathy pattern in MPV17; CMAP amplitude also reduced (axonal component); monitor 6-monthly"},
            {"marker": "CSF lactate", "cutoff": ">2.5 mmol/L", "interpretation": "CNS OXPHOS failure; elevated in hepatocerebral form; guides prognosis for neurological progression"},
        ],
        "ddx_table": [
            {"disease": "DGUOK — MDDS3", "shared": "Hepatocerebral mtDNA depletion; VPA+KD absolute CI; lactic acidosis; hepatic failure; AR", "distinguishing": "DGUOK: nystagmus 90% (PATHOGNOMONIC — absent in MPV17); NO peripheral neuropathy; posterior-fossa MRI predominance; dGMP/dAMP assay reduced; DGUOK sequencing"},
            {"disease": "POLG — Alpers-Huttenlocher", "shared": "mtDNA depletion; hepatopathy; VPA absolute CI; lactic acidosis; AR/AD; LEV preferred", "distinguishing": "POLG: EPC (focal motor continuous — 60%) dominates neurological picture; occipital predominance on MRI/EEG; hepatopathy = acute liver failure not cirrhosis; NO peripheral neuropathy as cardinal feature; Ala467Thr/Trp748Ser European founder"},
            {"disease": "SUCLA2 — MDDS5", "shared": "mtDNA depletion; hypotonia; lactic acidosis; AR; neurological regression", "distinguishing": "SUCLA2: methylmalonic aciduria (MMA elevated) — KEY DDx absent in MPV17; sensorineural hearing loss (SNHL) 90%; basal ganglia on MRI (not hepatic failure); mild hepatopathy only; no nystagmus"},
            {"disease": "SERAC1 — MEGDEL", "shared": "Hepatic disease; neurological regression; mitochondrial disease; AR", "distinguishing": "SERAC1: 3-MGA-uria Type V (ABSENT in MPV17); SNHL (90%); basal ganglia/dystonia; NO lactic acidosis as universal feature; methylglutaconic acid elevated on OA"},
            {"disease": "TAZ — Barth Syndrome", "shared": "Mitochondrial disease; lactic acidosis", "distinguishing": "TAZ: 3-MGA-uria Type II + C4-DC elevated (PATHOGNOMONIC); DCM 100%; neutropenia 95%; X-linked males only; NO hepatic failure or peripheral neuropathy"},
            {"disease": "Hereditary Tyrosinemia Type 1 (HT1)", "shared": "Neonatal/infantile hepatic failure; AR; hypoglycemia; coagulopathy", "distinguishing": "HT1: succinylacetone elevated on MS/MS (PATHOGNOMONIC); normal lactate in quiescent HT1; cabbage-like odour; FAH enzyme deficiency; treated with nitisinone (NTBC)"},
        ],
    }


def get_breakdown() -> dict:
    """MPV17 Hepatocerebral mtDNA Depletion — 40-patient cohort breakdown."""
    rng = random.Random(SEED)
    n = 40

    phenotype_groups = [
        ("Hepatocerebral form (severe, liver failure + neuropathy + regression)", 36),
        ("Hepatic-only form (liver-predominant, preserved CNS)", 4),
    ]

    variant_dist = [
        {"variant": "p.Arg50Gln/p.Arg50Gln homozygous (Navajo founder c.149G>A)", "n": 14, "pct": 35, "mechanism": "Channel pore Arg50→Gln: loss of positive charge → channel conductance ~90% reduced → dNTP import failure → hepatocerebral mtDNA depletion. Homozygous = no residual function. Navajo-specific founder with extreme population prevalence ~1:1,600.", "form": "Hepatocerebral"},
        {"variant": "p.Arg50Gln / loss-of-function (stop/frameshift/deletion) compound het (Navajo non-canonical)", "n": 6, "pct": 15, "mechanism": "One Arg50Gln allele + one null allele → <5% residual MPV17 function → most severe depletion. Navajo patients with one canonical + one private Navajo allele.", "form": "Hepatocerebral"},
        {"variant": "p.Pro98Leu / splice-site compound het (European, c.293-1G>C)", "n": 8, "pct": 20, "mechanism": "Pro98Leu disrupts TM3 helix integrity; splice c.293-1G>C → NMD of one allele → <20% residual MPV17 → hepatocerebral in most; mild missense allele might retain 15-25% function.", "form": "Hepatocerebral"},
        {"variant": "Exon 3-7 deletion / missense compound het (non-Navajo, any ethnicity)", "n": 6, "pct": 15, "mechanism": "Large deletion removes TM2-TM4 → null allele; missense on second allele; severity depends on missense residual activity; most are hepatocerebral (≤15% activity).", "form": "Hepatocerebral"},
        {"variant": "c.293-1G>C splice / missense (non-Navajo European)", "n": 4, "pct": 10, "mechanism": "Splice site → exon skip → truncated protein; compound het with mild missense → 10-20% activity → hepatocerebral or attenuated. Some survive to adolescence.", "form": "Mixed"},
        {"variant": "Mild missense/missense compound het (hepatic-only genotype)", "n": 2, "pct": 5, "mechanism": "Both alleles retain ≥25% MPV17 function → hepatic-predominant depletion; brain spared due to higher residual activity; hepatic-only phenotype with preserved neurological development.", "form": "Hepatic-only"},
    ]

    treatment_dist = [
        {"treatment": "IV 10% Dextrose (sick-day + perioperative + continuous overnight)", "n": 40, "pct": 100, "indication": "Level A — mandatory for ALL MPV17 patients for any NPO >2h, intercurrent illness, or metabolic instability; GIR 8-10 mg/kg/min; prevents hypoglycemia-triggered lactic acidosis; perioperative: from midnight before surgery; 2-hourly glucose monitoring intra-op; continuous overnight enteral feeds to prevent fasting hypoglycemia; Dextrogel home supply for families"},
        {"treatment": "Levetiracetam (LEV) — oral and IV", "n": 16, "pct": 40, "indication": "Level A — preferred AED for MPV17-associated seizures; no hepatic metabolism; no mito toxicity; no ammonia effect; renal excretion; IV loading 20-40 mg/kg for acute seizures; oral 30-50 mg/kg/day divided bid; safe in hepatic failure; alternative: IV clonazepam for myoclonic; NEVER valproate"},
        {"treatment": "Liver transplantation (OLT) — hepatic-only form only", "n": 4, "pct": 10, "indication": "Level B (hepatic-only, 10% of cohort) — may be curative if performed before neurological compromise; requires neurological stability (NCS, MRI, cognitive assessment), hepatic indication (progressive cirrhosis, INR >1.5), and multidisciplinary consensus (metabolic genetics + hepatology + neurology + ethics). Hepatocerebral form: OLT corrects hepatic disease but NOT peripheral neuropathy or brain depletion — not routinely recommended"},
        {"treatment": "High-carbohydrate enteral nutrition (NG/PEG — avoid high fat)", "n": 30, "pct": 75, "indication": "Level A — high-carbohydrate, moderate-protein, low-fat formula; no KD; NG from first feeding difficulty; PEG if >4 weeks; continuous overnight feeds prevent fasting; glucose polymer supplementation; dietitian-led from diagnosis; no medium-chain triglyceride (MCT) oil supplements (MCT requires β-oxidation → OXPHOS)"},
        {"treatment": "Physiotherapy + orthotics (peripheral neuropathy management)", "n": 32, "pct": 80, "indication": "Level A — peripheral neuropathy in 80%; foot drop + distal weakness; ankle-foot orthoses (AFO) for foot drop; physiotherapy 3×/week; hydrotherapy; assistive devices; prevent contractures; NCS/EMG monitoring 6-monthly to track progression; occupational therapy for hand function"},
        {"treatment": "Buccal / IV Midazolam (seizure rescue)", "n": 10, "pct": 25, "indication": "Level B — seizure rescue for MPV17-associated epilepsy; buccal 0.3 mg/kg (max 10 mg) for seizures >5min at home; IV midazolam infusion in hospital; families trained in buccal administration; NEVER propofol for anaesthesia"},
        {"treatment": "FFP + Vitamin K (coagulopathy management)", "n": 30, "pct": 75, "indication": "Level A — synthetic liver failure with coagulopathy (INR >1.3); FFP 10-15 mL/kg acute; IV vitamin K 0.3 mg/kg; maintain INR <1.5 before invasive procedures; platelets if <50 before LP/liver biopsy; ursodeoxycholic acid for cholestasis"},
        {"treatment": "Riboflavin (B2) + CoQ10 empirical supplementation", "n": 22, "pct": 55, "indication": "Level D — empirical mitochondrial cofactor support; no controlled evidence in MPV17; generally safe; riboflavin 50-100 mg/day; CoQ10 10-15 mg/kg/day divided; does not alter mtDNA depletion but may marginally support residual Complex I/II; monitor for CoQ10 levels if high dose"},
        {"treatment": "Palliative / goals-of-care team", "n": 34, "pct": 85, "indication": "Level A — hepatocerebral form is progressive (majority); palliative care integration from diagnosis; goals-of-care discussion with family about OLT decision, tracheostomy, PEG, neuropathy management; hospice planning for hepatocerebral form not eligible for OLT; symptom management (neuropathic pain gabapentin, spasticity baclofen, secretions hyoscine)"},
        {"treatment": "Immunosuppression post-OLT (tacrolimus + MMF)", "n": 4, "pct": 10, "indication": "Level A (post-OLT, hepatic-only form) — standard liver transplant immunosuppression; tacrolimus + mycophenolate mofetil; nephrotoxicity monitoring; annual neurological + metabolic + NCS assessment post-OLT; long-term immunosuppression management by hepatology"},
    ]

    seizure_profile = [
        {"type": "Focal motor seizures (metabolic encephalopathy)", "n": 12, "pct": 30, "desc": "From cortical neuronal OXPHOS failure; metabolic triggers (hypoglycemia, lactic acidosis, illness); not EPC (unlike POLG); responds to IV dextrose + LEV"},
        {"type": "Generalised tonic-clonic (secondary)", "n": 10, "pct": 25, "desc": "Secondary generalisation from multifocal cortical involvement; late feature in hepatocerebral form; LEV first-line"},
        {"type": "Myoclonic seizures", "n": 8, "pct": 20, "desc": "Cortical myoclonus from widespread cortical injury; associated with metabolic decompensation; clonazepam or LEV"},
        {"type": "Neonatal seizures (metabolic)", "n": 6, "pct": 15, "desc": "Early; metabolic trigger (hypoglycemia, lactic acidosis); clonic or subtle; often responds to glucose correction alone; EEG: burst-suppression pattern in severe neonates"},
        {"type": "Status epilepticus", "n": 4, "pct": 10, "desc": "Precipitated by metabolic decompensation; manage: IV dextrose + IV LEV + IV midazolam; NEVER propofol; anaesthesia: ketamine + sevoflurane"},
    ]

    metabolic_outcomes = [
        {"outcome": "Lactic acidosis (pH <7.1, lactate >10 mmol/L) — acute presentation", "n": 40, "pct": 100, "notes": "Universal; onset neonatal/infantile; OXPHOS failure → NAD⁺ regeneration failure → pyruvate → lactate; lactate:pyruvate >20:1 = mitochondrial cause; IV dextrose is primary treatment; bicarbonate only if pH <7.1"},
        {"outcome": "Hepatomegaly + progressive cirrhosis", "n": 40, "pct": 100, "notes": "Hepatocyte mtDNA depletion → steatosis → cirrhosis; liver >4 cm below costal margin at presentation; fibroscan + liver biopsy confirm cirrhosis grade; USS 3-monthly"},
        {"outcome": "Peripheral neuropathy (sensorimotor demyelinating/axonal)", "n": 32, "pct": 80, "notes": "Cardinal MPV17 feature; NCS/EMG: reduced NCV (demyelinating) + reduced CMAP/SNAP (axonal); foot drop; wrist drop; distal wasting; begins in infancy/early childhood; progressive without treatment"},
        {"outcome": "Coagulopathy (INR >1.3)", "n": 30, "pct": 75, "notes": "Synthetic liver failure; monitor INR 2-weekly in stable outpatients; OLT evaluation trigger at INR >1.5; correct with FFP before invasive procedures"},
        {"outcome": "Hypoglycemia (<3.5 mmol/L)", "n": 32, "pct": 80, "notes": "Hepatic gluconeogenesis failure; home glucose monitoring ALL families; Dextrogel home supply; emergency sick-day glucose protocol essential"},
        {"outcome": "Leukoencephalopathy (MRI T2 white matter abnormalities)", "n": 24, "pct": 60, "notes": "Diffuse WM T2 signal in NNH/hepatocerebral form; reflects OXPHOS failure in oligodendrocytes and astrocytes; unlike DGUOK (posterior fossa) or POLG (occipital); MPV17 more diffuse cerebral WM; correlates with peripheral neuropathy severity"},
        {"outcome": "Jaundice / cholestasis", "n": 26, "pct": 65, "notes": "Direct hyperbilirubinemia; ursodeoxycholic acid for intrahepatic cholestasis; bilirubin trend informs OLT urgency"},
        {"outcome": "Failure to thrive (weight <3rd percentile)", "n": 30, "pct": 75, "notes": "Hepatic failure + malabsorption + metabolic energy deficiency; NG/PEG high-carbohydrate feeds; dietitian review monthly"},
    ]

    biomarker_summary = {
        "hepatic_failure_pct": 100,
        "lactic_acidosis_pct": 100,
        "peripheral_neuropathy_pct": 80,
        "hypoglycemia_pct": 80,
        "coagulopathy_pct": 75,
        "leukoencephalopathy_pct": 60,
        "hypotonia_pct": 75,
        "regression_pct": 70,
        "epilepsy_pct": 40,
        "nystagmus_pct": 30,
        "three_mga_uria": "ABSENT (0%) — key DDx from all 3-MGA syndromes",
        "navajo_founder_pct": 50,
        "hepatic_only_form_pct": 10,
        "hepatocerebral_form_pct": 90,
        "olt_performed_pct": 10,
        "median_onset_months": 3,
        "median_diagnosis_delay_weeks": 10,
        "median_survival_hepatocerebral_years": 4,
    }

    return {
        "generated": date.today().isoformat(),
        "cohort": n,
        "seed": SEED,
        "phenotype_groups": [{"group": g, "n": c, "pct": round(c / n * 100)} for g, c in phenotype_groups],
        "variant_distribution": variant_dist,
        "treatment_distribution": treatment_dist,
        "seizure_profile": seizure_profile,
        "metabolic_outcomes": metabolic_outcomes,
        "biomarker_summary": biomarker_summary,
        "outcomes": {
            "median_survival_hepatocerebral_years": 4,
            "transplant_performed_pct": 10,
            "transplant_curative_hepatic_only_pct": 75,
            "neurological_progression_post_olt_hepatocerebral_pct": 90,
            "epilepsy_pct": 40,
            "severe_hypoglycemia_event_pct": 70,
            "peripheral_neuropathy_progression_annual_pct": 15,
            "leukoencephalopathy_on_mri_pct": 60,
            "failure_to_thrive_pct": 75,
            "median_diagnosis_delay_weeks": 10,
        },
    }


def get_definitions() -> dict:
    """MPV17 Hepatocerebral mtDNA Depletion — definitions for /api/mpv17/definitions."""
    return {
        "generated": date.today().isoformat(),
        "disease": "Mitochondrial DNA Depletion Syndrome 6 (MDDS6) / MPV17 Hepatocerebral Syndrome / Navajo Neurohepatopathy",
        "gene": "MPV17",
        "omim_gene": "137960",
        "omim_disease": "256810",
        "definitions": [
            {
                "term": "MPV17 — Inner Mitochondrial Membrane Channel and mtDNA Copy-Number Maintenance",
                "definition": "MPV17 (176 amino acids, 2p23.3) encodes a small integral inner mitochondrial membrane (IMM) protein belonging to the Mpv17/PMP22 family of tetrameric ion/metabolite channels. The mature protein contains four transmembrane helices (TM1-TM4) with both N- and C-termini facing the mitochondrial matrix. The extramembrane loop between TM1 and TM2 contains the critical pore residue Arg50, which is mutated to Gln in the Navajo founder variant. Unlike DGUOK (a kinase) or TK2 (a kinase), MPV17 is a channel protein with two proposed functions: (1) CHANNEL MODEL — MPV17 imports deoxyribonucleosides or dNTPs from the IMS into the matrix to supply POLG-mediated mtDNA replication; LOF → dNTP shortage → POLG stalls → mtDNA depletion; (2) IMM INTEGRITY MODEL — MPV17 stabilises IMM proton gradient/potential; LOF → altered ΔΨm → impaired nucleotide transporter activity → secondary dNTP shortage. Regardless of mechanism, loss of MPV17 → mtDNA copy number <30% in liver + brain → 13 mtDNA-encoded OXPHOS subunits (7 Complex I, 1 Complex III, 3 Complex IV, 2 Complex V) become insufficient → respiratory chain failure → ATP production collapse. Tissues with highest mtDNA turnover (liver, peripheral nerves) are affected earliest and most severely.",
                "relevance": "No direct enzymatic assay for MPV17 (unlike DGUOK activity assay or TK2 kinase assay). Diagnosis by: (1) DNA sequencing (gene panel or WES) identifying biallelic pathogenic variants; (2) liver biopsy qPCR: mtDNA copy number <30% of controls (ND1 or ND4 probe vs GAPDH); (3) respiratory chain enzymology (BN-PAGE): absent or severely reduced Complex I + III + IV activities in liver; (4) electron microscopy: enlarged, misshapen mitochondria with disorganised cristae in hepatocytes and peripheral nerve axons. Referral to specialist mitochondrial biochemistry lab (NHNN London, UMDF network, Mayo Clinic Mitochondrial Medicine) for confirmatory studies.",
            },
            {
                "term": "Navajo Neurohepatopathy (NNH) — Extreme Population Founder Effect of p.Arg50Gln",
                "definition": "Navajo Neurohepatopathy (NNH) is the eponymous manifestation of MPV17 deficiency in the Navajo people of the American Southwest. The p.Arg50Gln (c.149G>A) variant arose from a single founder ancestor among the Navajo Nation and achieves a carrier frequency of approximately 1:26 (4% of Navajo alleles), producing a disease prevalence of ~1:1,600 Navajo live births — one of the highest prevalences of any single-gene mitochondrial disease in any population. Homozygous p.Arg50Gln produces zero residual MPV17 channel function (the Arg50 residue is essential for pore conductance), generating the most severe hepatocerebral phenotype: neonatal/infantile hepatic failure, progressive cirrhosis, sensorimotor neuropathy, failure to thrive, metabolic crises, and diffuse leukoencephalopathy. The term 'Navajo Neurohepatopathy' precedes the molecular identification of MPV17 — first described clinically in 1988 by Appenzeller et al. in Navajo children with hepatic failure + neuropathy; molecular cause identified by Spinazzola et al. 2006. Non-Navajo MPV17 patients share the biochemical phenotype but carry private variants (p.Pro98Leu European, splice c.293-1G>C, exon deletions) and do not share the Navajo founder allele.",
                "relevance": "In clinical practice: (1) ANY Navajo child with hepatic failure + peripheral neuropathy → MPV17 sequencing SAME DAY (carrier frequency 1:26 means homozygosity risk is high); (2) In Navajo population newborn screening contexts, p.Arg50Gln targeted sequencing should be considered; (3) Family cascade testing — any Navajo sibling of an affected patient has 25% risk (parents are obligate carriers at 1:26 background); (4) Non-Navajo patients with hepatic failure + neuropathy + no 3-MGA-uria → MPV17 gene panel (alongside POLG, DGUOK, SUCLA2, SUCLG1 panel); (5) Public health: Navajo Nation IHS (Indian Health Service) hospitals should have MPV17 emergency protocols with VPA prohibition prominently displayed.",
            },
            {
                "term": "Peripheral Neuropathy in MPV17 — Distinguishing Feature vs DGUOK and POLG",
                "definition": "Peripheral neuropathy — affecting myelinated and unmyelinated peripheral nerve axons — occurs in approximately 80% of MPV17 patients and is the cardinal neurological feature that distinguishes MPV17 from DGUOK and POLG. The neuropathy is sensorimotor and may show demyelinating, axonal, or mixed pattern on nerve conduction studies (NCS) and electromyography (EMG): Motor NCS: reduced nerve conduction velocity (<40 m/s) indicating demyelination; reduced compound motor action potential (CMAP) amplitude indicating axonal loss; Sensory NCS: reduced or absent sensory nerve action potentials (SNAP). Pathophysiology: peripheral nerve Schwann cells (myelin) and axons have high mitochondrial density and mtDNA copy number; MPV17 LOF → Schwann cell mtDNA depletion → myelin maintenance failure → demyelination; axonal transport requires mitochondrial ATP → axonal OXPHOS failure → axonal degeneration; peripheral nerves are affected in parallel with liver due to high mtDNA turnover. Clinical features: distal-to-proximal progression, foot drop (peroneal nerve), wrist drop, absent ankle reflexes, stocking-glove sensory loss, neuropathic pain. NNH children present with progressive weakness, absent deep tendon reflexes, and bilateral foot drop often before significant cognitive regression.",
                "relevance": "NCS/EMG is MANDATORY in every MPV17 patient at diagnosis and 6-monthly thereafter. Key clinical pearls: (1) DGUOK: nystagmus 90% (cardinal) + NO significant peripheral neuropathy → if nystagmus present, test DGUOK first; (2) MPV17: peripheral neuropathy 80% (cardinal) + NO nystagmus (30%) → foot drop + absent reflexes + hepatic failure → test MPV17 first; (3) POLG: peripheral neuropathy occurs in POLG but EPC (focal motor continuous seizures) is the cardinal feature, not foot drop; (4) SUCLA2: peripheral neuropathy pattern with MMA and SNHL but no hepatic failure. Management: physiotherapy, AFOs for foot drop, pain management (gabapentin for neuropathic pain — safe in mito disease), occupational therapy. Experimental: vitamin supplementation (B12, folate) does not alter neuropathy course. Nerve biopsy (if performed): mitochondrial proliferation + axonal degeneration on electron microscopy.",
            },
            {
                "term": "Why No 3-MGA-uria in MPV17 — Critical DDx Tool Shared with DGUOK/POLG",
                "definition": "3-methylglutaconic acid (3-MGA) is produced from aberrant leucine catabolism when the inner mitochondrial membrane (IMM) lipid composition and integrity is specifically disrupted. 3-MGA-uria is the biochemical hallmark of conditions that perturb IMM phospholipid remodelling and cardiolipin-dependent membrane architecture: TAZ (Barth, Type II), DNAJC19 (DCMA, Type IV), SERAC1 (MEGDEL, Type V), TMEM70 (Type VI), CLPB (Type VII), OPA3 (Costeff, Type III). In contrast, MPV17 causes mtDNA copy-number depletion (matrix dNTP supply failure) without directly disrupting IMM phospholipid composition — therefore NO 3-MGA-uria. Similarly, POLG, DGUOK, SUCLA2, TK2 all cause mtDNA depletion without producing 3-MGA. 3-MGA absent on urine organic acids = excludes all 3-MGA-uria syndromes in one rapid bedside test.",
                "relevance": "Urine organic acids (OA) is the first biochemical test after plasma lactate in any suspected mitochondrial disease. In MPV17: OA shows elevated lactic acid only (no 3-MGA, no methylmalonate, no succinylacetone). Decision tree: (1) 3-MGA absent + hepatic failure + neuropathy + no nystagmus → MPV17 FIRST; (2) 3-MGA absent + hepatic failure + nystagmus → DGUOK FIRST; (3) 3-MGA absent + EPC + hepatopathy + occipital MRI → POLG FIRST; (4) MMA elevated + SNHL + basal ganglia → SUCLA2/SUCLG1; (5) 3-MGA present → 3-MGA syndrome panel (SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB). This biochemical triage takes <48h and directs targeted gene sequencing, avoiding months-long diagnostic odyssey.",
            },
            {
                "term": "Liver Transplantation in MPV17 — Hepatic-Only (May Cure) vs Hepatocerebral (Does Not Prevent Neurological Progression)",
                "definition": "Liver transplantation (OLT) in MPV17 follows the same form-specific logic as DGUOK: HEPATIC-ONLY FORM (~10% of MPV17 cases): mild genotype → sufficient residual MPV17 in brain to prevent significant CNS depletion during childhood → hepatic failure is the dominant problem → OLT replaces DGUOK-deficient hepatocytes with normal donor liver → hepatic mtDNA normalises → hepatic OXPHOS recovers → if neurological baseline is preserved, CNS does NOT develop significant depletion post-OLT → curative outcome possible (Level B). HEPATOCEREBRAL FORM (~90% of MPV17 cases): biallelic null or severe variant → zero MPV17 in brain + liver → brain mtDNA depletion is autonomous and continues post-OLT → peripheral neuropathy progresses → leukoencephalopathy worsens → OLT corrects hepatic disease but NOT neurological disease. Post-OLT NNH data: neurological progression in >90% of hepatocerebral cases within 12-24 months of OLT (published Navajo neurohepatopathy transplant series, Kaplan et al. 2004 and subsequent updates). Most experienced centres do NOT recommend OLT for hepatocerebral MPV17 beyond palliation of acute hepatic decompensation.",
                "relevance": "OLT decision in MPV17 is ethically complex and irreversible: (1) FORM CONFIRMATION — genotype + neurological assessment (NCS/EMG, MRI, cognitive testing) + mtDNA copy number to classify hepatic-only vs hepatocerebral; (2) TIMING — hepatic-only: early OLT before neurological compromise (ideally <18 months); (3) NAVAJO-SPECIFIC — given extreme prevalence, Navajo IHS hepatology teams are familiar with the OLT decision process in NNH; (4) MULTIDISCIPLINARY — metabolic genetics + hepatology + neurology + ethics + family; (5) POST-OLT MONITORING — even in hepatic-only, annual NCS/EMG + MRI + cognitive testing + metabolic screen; (6) FAMILY COUNSELLING — hepatocerebral form: realistic prognosis (4-6 year survival without OLT; OLT prolongs hepatic survival but neurological suffering continues); goals-of-care discussion mandatory.",
            },
            {
                "term": "Emergency Management: Metabolic Crisis in MPV17 (Hypoglycemia + Lactic Acidosis + Neuropathic Pain Crisis)",
                "definition": "MPV17 metabolic crises are triggered by intercurrent illness (infection, fever), fasting, surgery, or progressive hepatic failure. Emergency protocol: (1) GLUCOSE FIRST — capillary glucose immediately; <3.5 mmol/L: IV 10% dextrose bolus 2 mL/kg over 5min + continuous GIR 8-10 mg/kg/min; do NOT use 50% dextrose; maintain glucose 5-10 mmol/L; (2) LACTIC ACIDOSIS — sodium bicarbonate only if pH <7.1 (0.5-1 mmol/kg IV over 2h); IV dextrose is the primary treatment; (3) SEIZURES — IV LEV 20-40 mg/kg loading; IV midazolam 0.15 mg/kg + infusion; NEVER VPA, NEVER propofol; anaesthesia: ketamine + sevoflurane; (4) NEUROPATHIC PAIN CRISIS — gabapentin escalation (safe in mito disease); avoid NSAIDs in hepatic failure; opioids with caution (hepatic metabolism); (5) LIVER PROTECTION — avoid ALL hepatotoxic drugs; FFP 10-15 mL/kg if INR >1.5 before invasive procedures; LFTs + glucose + lactate + ammonia + INR q4h; (6) SPECIALIST ESCALATION — metabolic genetics on-call; PICU admission; consider OLT team if hepatic decompensation. Emergency letter in every MPV17 family: MPV17 mtDNA DEPLETION — VPA ABSOLUTELY PROHIBITED — NO PROPOFOL — IV DEXTROSE MANDATORY — METABOLIC GENETICS: [NUMBER].",
                "relevance": "Every MPV17 family must hold: (1) Emergency letter with hospital allergy flags (VPA, propofol); (2) Home glucose monitor + Dextrogel; (3) Buccal midazolam for seizures >5min; (4) Sick-day protocol: increase carbohydrate feeds, monitor glucose 2-hourly, when to call emergency services; (5) Emergency neuropathic pain kit (gabapentin escalation plan with metabolic team); (6) Annual metabolic review with updated sick-day protocol. NNH in Navajo Nation: IHS emergency departments must have MPV17 protocol posters and pre-stocked IV dextrose protocols for any Navajo child with unexplained hepatic failure or metabolic crisis.",
            },
        ],
    }
