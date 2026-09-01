#!/usr/bin/env python3
"""DGUOK Hepatocerebral mtDNA Depletion Syndrome Dashboard.

Mitochondrial DNA Depletion Syndrome 3 (MDDS3) = OMIM #251880.
Neonatal hepatic failure + nystagmus + progressive neurological disease.
Biallelic AR DGUOK mutations → dGTP/dATP pool depletion → mtDNA depletion
→ OXPHOS failure (liver + brain).

DGUOK (Deoxyguanosine Kinase, 277 aa, 2p13.1) is the mitochondrial matrix
enzyme that phosphorylates purine deoxyribonucleosides (deoxyguanosine → dGMP;
deoxyadenosine → dAMP), supplying the dNTP pool essential for mtDNA replication.

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — identical mechanism to POLG: mtDNA depletion +
     CoA sequestration + epoxide hepatotoxicity; fatal in patients with pre-existing depletion
  2. Two clinical forms: Hepatocerebral (75%, neonatal, fatal) vs Hepatic-only (25%, nystagmus
     + liver; transplant curative — neurological development preserved)
  3. Nystagmus (90%) — rotary/pendular — KEY early sign; first feature before overt liver failure
  4. NO 3-MGA-uria — critical DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19 (all have 3-MGA)
  5. Liver transplant: CURATIVE in hepatic-only form; does NOT prevent neurological progression
     in hepatocerebral form — brain mtDNA depletion continues post-transplant
  6. KD = CONTRAINDICATED — high-fat β-oxidation requires intact OXPHOS; mtDNA depletion
     → ketone body metabolism fails → worsens lactic acidosis; never use in DGUOK
  7. LEV preferred AED — renal excretion; no hepatic metabolism; no mito toxicity
  8. Propofol AVOID — PRIS in any mitochondrial disease
  9. IV 10% dextrose (GIR 8-10 mg/kg/min) mandatory for ANY illness/NPO in DGUOK
 10. Mandel et al. 2001 Nature Genetics — first DGUOK mutations described

DGUOK BIOLOGY:
DGUOK (277 amino acids, 2p13.1) carries a 16 aa N-terminal mitochondrial
targeting sequence (MTS, aa1-16) cleaved on import into the mitochondrial matrix.

Domain architecture:
  MTS (aa1-16): cleavage site; deletion = cytoplasmic retention + LOF
  Catalytic core (aa17-277): P-loop NTPase fold; Asp158-Phe-Gly (DFG) catalytic motif;
    nucleoside-binding pocket; Arg47, His86, Asn165 line the active site
    Arg47Gly: disrupts arginine-guanine base stacking → 95% activity loss
    Asn165Ser: disrupts ribose OH hydrogen bond → 80% activity loss
    Tyr166His: partial loss (~40%); associated with hepatic-only form

Why dGTP/dATP depletion is unique:
  DGUOK LOF → mitochondrial dG and dA salvage fails → dGTP + dATP below threshold
  for POLG-mediated mtDNA strand displacement synthesis → replication stalls →
  copy number falls <30% → 13 mtDNA-encoded OXPHOS subunits become insufficient →
  Complex I (7 subunits), Complex III (1), Complex IV (3), Complex V (2) all fail.
  The purine depletion is hepatocyte-selective (high mtDNA turnover in liver) explaining
  the hepatic-predominant phenotype vs brain (lower mtDNA turnover = slower depletion).

TWO CLINICAL FORMS (genotype-phenotype correlation):
  Hepatocerebral form (~75%): Null/null or Null/severe missense → <10% residual activity
    → rapid mtDNA depletion in liver AND brain → neonatal liver failure + nystagmus +
    progressive encephalopathy + epilepsy + regression → fatal by 1-4 years without OLT;
    OLT corrects hepatic disease but NOT brain depletion → neurological progression continues.
  Hepatic-only form (~25%): Missense/missense with ≥20% residual activity (e.g. Tyr166His/
    mild missense) → slower depletion → liver preferentially affected → nystagmus + liver
    disease + preserved CNS development → OLT can be curative if done before neurological
    compromise; ongoing monitoring required post-transplant.

PATHOGENIC VARIANT DISTRIBUTION (biallelic AR, n=40, seed-549):
  p.Asn165Ser/null compound het or homozygous: ~30% — hepatocerebral; <10% activity
  p.Arg47Gly/null compound (South Asian, Middle Eastern): ~20% — hepatocerebral; <5% activity
  Splice variants (c.494+1G>A, c.763-2A>G) compound het: ~20% — hepatocerebral; NMD
  p.Tyr166His/mild missense compound (hepatic-only form): ~15% — 20-35% residual activity
  Other (p.His86Asn, p.Arg107Cys, p.Gly227Asp): ~15% — mixed hepatocerebral/hepatic
"""

import random
from datetime import date

SEED = 549  # 40-patient cohort seed


def get_overview() -> dict:
    """DGUOK Hepatocerebral mtDNA Depletion — overview for /api/dguok/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": "Mitochondrial DNA Depletion Syndrome 3 (MDDS3) / DGUOK Hepatocerebral Syndrome",
        "gene": "DGUOK; Deoxyguanosine Kinase; Mitochondrial Purine Salvage dNTPase; 277 aa (MTS aa1-16 cleaved); mitochondrial matrix",
        "chromosome": "2p13.1",
        "omim_gene": "601465",
        "omim_disease": "251880",
        "inheritance": "Autosomal Recessive (biallelic DGUOK); no carrier phenotype in heterozygotes",
        "prevalence": "Rare; estimated <1:100,000 livebirths; exact population prevalence unknown; South Asian and Middle Eastern populations overrepresented; no single European founder variant unlike POLG",
        "protein": "DGUOK 277 aa; MTS (aa1-16)-Catalytic-deoxyguanosine-kinase-core (aa17-277); DFG catalytic motif; P-loop NTPase fold; phosphorylates dG→dGMP and dA→dAMP; mitochondrial matrix",
        "category": "mtDNA Depletion Syndrome / Mitochondrial DNA Maintenance / DGUOK Purine Salvage Disorder",
        "first_described": "Mandel H et al. 2001 Nature Genetics 29:337-341 — first DGUOK mutations in hepatocerebral mtDNA depletion",
        "kpis": {
            "nystagmus_pct": 90,
            "hepatic_failure_pct": 85,
            "lactic_acidosis_pct": 100,
            "hypoglycemia_pct": 70,
            "hypotonia_pct": 85,
            "regression_pct": 80,
            "epilepsy_pct": 45,
            "vpa_risk": "ABSOLUTE CONTRAINDICATION — mtDNA depletion + CoA sequestration + epoxide hepatotoxicity",
            "hepatic_only_form_pct": 25,
            "transplant_curative_hepatic_only": "CURATIVE in hepatic form only — does NOT prevent brain depletion in hepatocerebral form",
            "no_3mga_uria": "ABSENT — KEY DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB",
        },
        "clinical_highlights": [
            "VPA (Valproate) = ABSOLUTE CONTRAINDICATION — mtDNA depletion mechanism identical to POLG: (1) VPA metabolites inhibit residual DGUOK dNTP supply → complete mtDNA depletion in already-depleted hepatocytes; (2) CoA sequestration via propionyl-CoA → FAO collapse; (3) VPA 4-en-epoxide → direct hepatotoxicity. In DGUOK, hepatocytes already at <30% mtDNA — VPA causes precipitous irreversible failure. No safe dose in any mtDNA depletion syndrome.",
            "TWO FORMS — HEPATOCEREBRAL (75%) vs HEPATIC-ONLY (25%): Hepatocerebral = null/severe genotype → <10% DGUOK activity → neonatal liver failure + nystagmus + encephalopathy + epilepsy + regression → fatal 1-4yr; liver transplant corrects hepatic disease but NOT neurological progression. Hepatic-only = milder genotype → ≥20% activity → liver + nystagmus + preserved CNS; transplant can be CURATIVE.",
            "NYSTAGMUS (90%) — early and PATHOGNOMONIC for DGUOK: rotary or pendular nystagmus appearing in the first weeks of life, often before overt liver failure is recognised; any neonate with unexplained nystagmus + lactic acidosis + hepatomegaly → DGUOK sequencing MANDATORY; nystagmus persists even with successful liver transplant",
            "NO 3-MGA-uria — CRITICAL DDx: DGUOK does NOT produce 3-methylglutaconic acid (3-MGA) unlike SERAC1 (3-MGA Type V), TAZ/Barth (3-MGA Type II), TMEM70 (3-MGA Type VI), OPA3 (3-MGA Type III), DNAJC19 (3-MGA Type IV), CLPB (3-MGA Type VII). Urine organic acids: NO 3-MGA in DGUOK — excludes all 3-MGA-uria syndromes; lactic acid ELEVATED instead.",
            "Neonatal lactic acidosis (100%) — often severe pH <7.1; lactate >10 mmol/L; lactic:pyruvate ratio >20:1 = mitochondrial cause confirmed; CSF lactate elevated in hepatocerebral form; IV dextrose GIR 8-10 mg/kg/min mandatory to prevent catabolism",
            "Liver transplant — CURATIVE for hepatic-only form (Level B evidence): selected patients with preserved neurological function and mild/moderate genotype — OLT corrects the hepatic OXPHOS deficit; post-transplant CNS monitoring essential. DOES NOT prevent neurological progression in hepatocerebral form — brain mtDNA depletion is autonomous and continues post-transplant regardless of liver recovery.",
            "KD = CONTRAINDICATED — in any mtDNA depletion syndrome: high-fat ketogenic metabolism requires intact mitochondrial β-oxidation and ketone body oxidation, both OXPHOS-dependent; KD forces catabolism through a pathway that fails in mtDNA-depleted neurons → worsens lactic acidosis and neurological decline; never use KD in DGUOK (unlike channelopathies where KD is first-line)",
            "Hypoglycemia (70%) — hepatic gluconeogenesis failure from OXPHOS dysfunction; glucose 2-hourly during acute illness; perioperative IV dextrose; families have written sick-day glucose protocol; point-of-care glucose at every medical contact",
            "Propofol AVOID (PRIS risk) — any anaesthetic in mitochondrial disease: use ketamine + sevoflurane; alert anaesthesia team; document DGUOK at every procedural pre-assessment",
            "Nucleotide supplementation (investigational) — oral deoxyguanosine (dG) + deoxyadenosine (dA) nucleosides to replenish the depleted dNTP pool; compassionate use; animal model benefit; no RCT in humans; does not reverse established mtDNA depletion",
        ],
        "contraindications": [
            {
                "drug": "VPA (Valproate / Valproic Acid / Sodium Valproate / Divalproex)",
                "level": "ABSOLUTE CONTRAINDICATION — DO NOT USE UNDER ANY CIRCUMSTANCES IN ANY MTDNA DEPLETION SYNDROME",
                "reason": "VPA causes fatal hepatotoxicity in DGUOK by three synergistic mechanisms identical to POLG: (1) VPA competes with dNTPs at the residual DGUOK/POLG active site, suppressing the already-depleted dGTP/dATP pool further → complete mtDNA depletion in hepatocytes already at <30% normal → necrosis within days to weeks; (2) VPA is metabolised to propionyl-CoA → mitochondrial CoA sequestration → FAO collapse → microvesicular steatosis → acute hepatic failure; (3) VPA 4-en-VPA epoxide is hepatotoxic through covalent protein modification — glucuronidation detoxification fails in OXPHOS-impaired liver. Emergency action: if DGUOK diagnosed while on VPA, switch IMMEDIATELY to IV levetiracetam 20 mg/kg loading; document VPA as permanently contraindicated in ALL medical records and allergy systems.",
            },
            {
                "drug": "Ketogenic Diet (KD) / high-fat dietary interventions",
                "level": "CONTRAINDICATED — forces OXPHOS-dependent pathway that fails in mtDNA depletion",
                "reason": "KD induces ketosis requiring: (1) hepatic β-oxidation of long-chain fatty acids → acetyl-CoA → ketone bodies (OXPHOS-dependent; Complex I/II/III generate NADH/FADH2 for re-oxidation); (2) neuronal ketone body oxidation via succinyl-CoA:acetoacetate-CoA transferase (OXPHOS-dependent). In DGUOK, Complex I/III/IV are rate-limiting. KD forces substrate through collapsed OXPHOS → worsens lactic acidosis (NAD⁺ regeneration fails → pyruvate → lactate); documented worsening in mtDNA depletion; NEVER use in DGUOK, POLG, TMEM70, or any Complex I/III/IV primary deficiency.",
            },
            {
                "drug": "Propofol (anaesthesia / sedation)",
                "level": "AVOID — PRIS (Propofol Infusion Syndrome) risk in mitochondrial disease",
                "reason": "Propofol inhibits Complex I (NADH:ubiquinone oxidoreductase) and uncouples the mitochondrial inner membrane. In DGUOK-depleted mitochondria with pre-existing OXPHOS failure, propofol infusion → PRIS (lactic acidosis + rhabdomyolysis + cardiac failure + renal failure). Use ketamine for induction, sevoflurane/desflurane for maintenance. Alert anaesthesia team to DGUOK diagnosis before every procedure.",
            },
            {
                "drug": "Prolonged fasting / NPO without IV glucose",
                "level": "DANGER — mandatory IV 10% dextrose (GIR 8-10 mg/kg/min) for ANY NPO >2h",
                "reason": "Hepatic gluconeogenesis is OXPHOS-dependent and fails in DGUOK. Any prolonged fast → hypoglycemia + catabolism → forces fat oxidation (worsens lactic acidosis) → acute metabolic decompensation. Protocol: glucose-containing drinks during mild illness; if vomiting or NPO: IV 10% dextrose GIR 8-10 mg/kg/min immediately; perioperative: IV dextrose from midnight before surgery; glucose checked 2-hourly intra-op and post-op; target glucose 5-10 mmol/L.",
            },
            {
                "drug": "Liver transplantation (as cure in hepatocerebral form)",
                "level": "CLINICAL CAUTION — curative hepatic-only; does NOT prevent neurological progression in hepatocerebral form",
                "reason": "In hepatocerebral DGUOK: liver transplant corrects the hepatic mtDNA depletion (transplanted liver has normal DGUOK and normal mtDNA) but does not supply dGTP/dATP to neurons — brain mtDNA depletion is autonomous and continues post-transplant; neurological decline continues and may accelerate perioperatively (surgical stress, immunosuppressants, anaesthesia). In hepatic-only form with preserved neurological function and mild genotype: OLT prevents hepatic failure and may preserve CNS before brain depletion is established — Level B evidence; early transplant decision requires specialist metabolic + hepatology + neurology consensus.",
            },
        ],
        "thresholds": [
            {"marker": "Serum lactate (resting)", "cutoff": ">3.0 mmol/L persistent; acute illness >5 mmol/L", "interpretation": "Mitochondrial OXPHOS failure marker. Persistent >3 = metabolic decompensation → admit + IV 10% dextrose GIR 8-10 mg/kg/min + metabolic team. Lactate:pyruvate ratio >20:1 confirms mitochondrial (not hypoxic) aetiology. Measure pre-prandial and fasting. Neonatal lactate >10 mmol/L = EMERGENCY."},
            {"marker": "ALT / AST (serum transaminases)", "cutoff": ">3× ULN at any point", "interpretation": "STOP all hepatotoxic medications immediately including VPA if somehow prescribed. Weekly LFTs at 3× ULN. Liver team input at 10× ULN. INR + bilirubin + ammonia + glucose in acute decompensation. Quarterly LFT monitoring in stable DGUOK."},
            {"marker": "Blood glucose (capillary/plasma)", "cutoff": "<3.5 mmol/L at any time; target 5-10 mmol/L during illness/NPO", "interpretation": "Hypoglycemia = hepatic gluconeogenesis failure. TREAT IMMEDIATELY: IV 10% dextrose bolus 2 mL/kg + infusion GIR 8-10 mg/kg/min. Families carry glucose gel (Dextrogel) for home use during intercurrent illness. POC glucose at every medical contact in known DGUOK."},
            {"marker": "mtDNA copy number (liver biopsy qPCR)", "cutoff": "<30% of age-matched controls", "interpretation": "Diagnostic for mtDNA depletion syndrome. ND1 or ND4 probe vs nuclear housekeeping (GAPDH). Liver <30% = significant depletion; DGUOK mutation analysis if not already done. Liver mtDNA more sensitive than muscle in DGUOK (liver is primary tissue). Repeat after OLT to confirm graft mtDNA normalisation."},
            {"marker": "Urine organic acids — 3-methylglutaconic acid (3-MGA)", "cutoff": "ABSENT in DGUOK (key DDx)", "interpretation": "3-MGA ABSENT = excludes all 3-MGA-uria syndromes (SERAC1, TAZ, TMEM70, OPA3, DNAJC19, CLPB). DGUOK urine organics show elevated lactic acid + pyruvate only. If 3-MGA present in any quantity → reconsider diagnosis, rule out 3-MGA syndromes first. This is the most rapid bedside DDx tool."},
            {"marker": "INR (coagulation)", "cutoff": ">1.3 (early); >1.5 (liver transplant evaluation trigger)", "interpretation": "Synthetic liver function failure. INR >1.3: FFP + vitamin K; coagulation team. INR >1.5: liver transplant evaluation URGENTLY in hepatic-only form. Avoid invasive procedures if INR >1.5 without correction. Neonatal coagulopathy is often the presenting feature before transaminase elevation is recognised."},
        ],
        "ddx_table": [
            {
                "disease": "POLG Alpers-Huttenlocher Syndrome",
                "shared": "mtDNA depletion; hepatopathy; neurological regression; VPA absolute CI; liver transplant does not cure brain",
                "distinguishing": "POLG: onset 2mo-4yr (not neonatal); EPC (epilepsia partialis continua) HALLMARK; occipital DWI restriction; Ala467Thr/Trp748Ser European founders; NO nystagmus as early feature. DGUOK: NEONATAL onset; NYSTAGMUS is the hallmark (rotary/pendular, day 1-4 weeks); no EPC; no European founder; NO 3-MGA in either. Confirm by gene panel: DGUOK chromosome 2p13.1; POLG chromosome 15q25.1.",
            },
            {
                "disease": "TMEM70 Complex V Deficiency (3-MGA Type VI)",
                "shared": "Mitochondrial disease; neonatal presentation; VPA absolute CI; lactic acidosis",
                "distinguishing": "TMEM70: 3-MGA-uria (100%) + hyperammonemia (90%, NH3 50-500 μmol/L) + DCM (85%) = CARDINAL TRIAD. DGUOK: NO 3-MGA; NO hyperammonemia; NO DCM. Nystagmus prominent in DGUOK; not in TMEM70. IV dextrose + ammonia scavengers in TMEM70; IV dextrose alone in DGUOK. Urine organic acids + plasma ammonia are the fastest bedside DDx.",
            },
            {
                "disease": "SERAC1 MEGDEL Syndrome (3-MGA Type V)",
                "shared": "Mitochondrial disease; neonatal liver involvement; lactic acidosis",
                "distinguishing": "SERAC1: 3-MGA-uria (100%) + SNHL (sensorineural hearing loss, 100%, PATHOGNOMONIC); Leigh-like MRI bilateral putamen (not thalami); transient neonatal liver (resolves by 3 months). DGUOK: NO 3-MGA; NO SNHL; liver does NOT resolve spontaneously — progresses to failure; nystagmus prominent in DGUOK; hearing normal.",
            },
            {
                "disease": "TAZ Barth Syndrome (3-MGA Type II)",
                "shared": "Mitochondrial disease; lactic acidosis; liver involvement possible",
                "distinguishing": "TAZ: 3-MGA-uria Type II; C4-DC elevated (PATHOGNOMONIC); DCM 100%; neutropenia 95%; NORMAL COGNITION; X-linked (males only). DGUOK: NO 3-MGA; NO C4-DC; NO DCM as presenting feature; NO neutropenia; affects both sexes (AR); liver failure primary.",
            },
            {
                "disease": "Galactosaemia (GALT deficiency)",
                "shared": "Neonatal jaundice; hepatic failure; E. coli sepsis; lactic acidosis; hypoglycemia",
                "distinguishing": "Galactosaemia: neonatal presentation with breast/formula milk → jaundice + cataracts + E. coli sepsis; urine reducing substances positive (galactose); galactose-1-phosphate uridyltransferase assay diagnostic; no mtDNA depletion; no nystagmus as primary feature; treatable with galactose-free diet. DGUOK: mtDNA depletion confirmed on liver biopsy; nystagmus; no galactose metabolite elevation; not diet-treatable.",
            },
            {
                "disease": "Neonatal Haemochromatosis (Gestational Alloimmune Liver Disease — GALD)",
                "shared": "Neonatal hepatic failure; coagulopathy; elevated ferritin; liver histology abnormal",
                "distinguishing": "GALD/Neonatal haemochromatosis: extrahepatic siderosis (buccal mucosa biopsy positive); very high ferritin + transferrin saturation; MRI liver + extrahepatic iron deposition; responds to IVIG + exchange transfusion; NO mtDNA depletion; NO nystagmus; NO lactic acidosis. DGUOK: normal ferritin unless secondary to liver failure; lactic acidosis; nystagmus; mtDNA depletion on liver biopsy.",
            },
        ],
    }


def get_breakdown() -> dict:
    """DGUOK Hepatocerebral mtDNA Depletion — patient breakdown for /api/dguok/breakdown."""
    rng = random.Random(SEED)
    n = 40

    phenotype_groups = [
        ("Hepatocerebral form: neonatal liver failure + nystagmus + progressive encephalopathy + epilepsy", 30),
        ("Hepatic-only form: neonatal liver + nystagmus; preserved neurological development; OLT considered", 10),
    ]
    assert sum(c for _, c in phenotype_groups) == n

    variant_dist = [
        {"variant": "p.Asn165Ser / null compound heterozygous or homozygous", "n_alleles": 24, "pct": 30, "effect": "Most common hepatocerebral genotype globally; Asn165 lines the ribose OH-binding pocket; Ser165 disrupts hydrogen bond → 80% activity loss; homozygous or with null second allele → <10% total; severe neonatal liver failure; nystagmus day 1-3 weeks; rapidly fatal without OLT; brain depletion progresses even after liver transplant"},
        {"variant": "p.Arg47Gly / null compound het (South Asian, Middle Eastern)", "n_alleles": 16, "pct": 20, "effect": "Second most common hepatocerebral genotype; Arg47 lines the guanine-binding pocket; Gly47 abolishes guanine stacking → >95% activity loss with null second allele; neonatal presentation within first week; median survival without OLT 3 months; common in South Asian and Middle Eastern consanguineous families"},
        {"variant": "Splice variants: c.494+1G>A / c.763-2A>G compound het", "n_alleles": 16, "pct": 20, "effect": "Splice-site variants → exon skipping → frameshift → NMD → null alleles; compound het with each other or with missense; hepatocerebral phenotype; early neonatal onset; lactic acidosis pH <7.0 at birth; OXPHOS complexes I+III+IV absent on BN-PAGE"},
        {"variant": "p.Tyr166His / mild missense compound (hepatic-only form)", "n_alleles": 12, "pct": 15, "effect": "Tyr166 is adjacent to Asn165 in the catalytic pocket; His166 reduces substrate affinity moderately → ~25-35% residual DGUOK activity → hepatic-predominant depletion (liver higher mtDNA turnover than brain); liver failure neonatal but neurological development preserved; OLT curative in these patients; ongoing monitoring required post-transplant"},
        {"variant": "Other: p.His86Asn, p.Arg107Cys, p.Gly227Asp (mixed forms)", "n_alleles": 12, "pct": 15, "effect": "Heterogeneous; >50 reported pathogenic DGUOK variants; His86Asn disrupts a conserved histidine in the P-loop; Arg107Cys creates aberrant disulfide; Gly227Asp destabilises C-terminal helix; residual activity 10-30% determines hepatocerebral vs hepatic-only; functional assay in fibroblasts or yeast complementation required for VUS classification"},
    ]

    treatment_dist = [
        {"treatment": "IV 10% Dextrose (sick-day + perioperative + acute)", "n": 40, "pct": 100, "indication": "Level A — mandatory for any NPO >2h or intercurrent illness; GIR 8-10 mg/kg/min; prevents hypoglycemia + catabolism → lactic acidosis worsening; perioperative: start IV dextrose from midnight before surgery; glucose 2-hourly intra-op; families have Dextrogel home supply + written sick-day protocol"},
        {"treatment": "Levetiracetam (LEV) — oral and IV", "n": 18, "pct": 45, "indication": "Level A — preferred AED for DGUOK-associated seizures; no hepatic metabolism; no mito toxicity; no ammonia effect; renal excretion; IV loading 20-40 mg/kg for acute SE; oral 30-50 mg/kg/day divided; safe in liver failure"},
        {"treatment": "Liver transplantation (OLT)", "n": 10, "pct": 25, "indication": "Level B (hepatic-only form) — CURATIVE in selected hepatic-only patients with preserved neurological function and mild genotype; requires specialist consensus (metabolic + hepatology + neurology + ethics + family); performed before neurological compromise to maximise benefit; DOES NOT help hepatocerebral form — brain depletion continues post-OLT"},
        {"treatment": "NG / PEG tube feeding (enteral nutrition support)", "n": 24, "pct": 60, "indication": "Level A — high-carbohydrate low-fat formula; no KD or high-fat feeds; NG from first sign of feeding difficulty; PEG if >4-week requirement; minimise fasting duration; continuous overnight feeds to prevent hypoglycemia; dietitian-led from diagnosis"},
        {"treatment": "Buccal / IV Midazolam (seizure rescue)", "n": 12, "pct": 30, "indication": "Level B — first-line acute seizure rescue for DGUOK-associated epilepsy; buccal 0.3 mg/kg (max 10 mg) for seizures >5min at home; IV midazolam infusion in hospital; families trained in buccal administration"},
        {"treatment": "FFP + Vitamin K (coagulopathy management)", "n": 32, "pct": 80, "indication": "Level A — synthetic liver failure with coagulopathy (INR >1.3); FFP 10-15 mL/kg acute correction; IV vitamin K 0.3 mg/kg; maintain INR <1.5 before invasive procedures; platelets if <50 before LP/liver biopsy"},
        {"treatment": "Riboflavin (B2) + CoQ10 empirical supplementation", "n": 20, "pct": 50, "indication": "Level D — empirical mitochondrial cofactor support; no controlled evidence in DGUOK; generally safe; does not alter mtDNA depletion course but may marginally support residual Complex I/II; riboflavin 50-100 mg/day; CoQ10 10-15 mg/kg/day divided"},
        {"treatment": "Deoxyguanosine + deoxyadenosine nucleoside supplementation (investigational)", "n": 6, "pct": 15, "indication": "Level D (investigational) — oral dG + dA to replenish depleted mitochondrial dNTP pool; animal model (DGUOK-knockout mouse) showed benefit; compassionate use only; does not reverse established depletion; families must understand fully experimental status; no RCT data"},
        {"treatment": "Palliative / goals-of-care team", "n": 30, "pct": 75, "indication": "Level A — hepatocerebral form is progressive and fatal (median survival 1-4yr from onset without OLT); palliative care integration from diagnosis; goals-of-care discussion with family including OLT decision; hospice planning for hepatocerebral form not eligible for OLT; symptom management (anti-emetics, sedation, nutritional support)"},
        {"treatment": "Physiotherapy + occupational therapy", "n": 24, "pct": 60, "indication": "Level A — hypotonia + motor regression; positioning; tone management; adaptive equipment; upper limb function preservation; from diagnosis; intensity increases with regression"},
        {"treatment": "Ophthalmology (nystagmus management)", "n": 36, "pct": 90, "indication": "Level B — ophthalmology review for nystagmus type and severity; spectacle correction for refractive error; patching if amblyopia risk; prism correction for null zone; visual acuity monitoring 3-monthly; brain MRI for nystagmus-related supranuclear pathways"},
        {"treatment": "Immunosuppression post-OLT (tacrolimus + MMF)", "n": 10, "pct": 25, "indication": "Level A (post-OLT) — standard liver transplant immunosuppression; tacrolimus + mycophenolate mofetil; nephrotoxicity monitoring; target tacrolimus trough 5-10 ng/mL at 3 months; steroids wean by 3 months; annual metabolic + neurological review post-OLT"},
    ]

    seizure_profile = [
        {"type": "Focal motor seizures (secondary to metabolic encephalopathy)", "n": 16, "pct": 40, "desc": "Focal jerking or tonic posturing from neuronal OXPHOS failure; not EPC (unlike POLG); metabolic trigger: hypoglycemia, lactic acidosis worsening, intercurrent illness; responds to IV dextrose + LEV"},
        {"type": "Generalised tonic-clonic (secondary)", "n": 12, "pct": 30, "desc": "Secondary generalisation from multifocal cortical involvement; late feature; reflects advanced cortical neuronal loss from mtDNA depletion; manage with LEV + midazolam rescue"},
        {"type": "Myoclonic seizures", "n": 10, "pct": 25, "desc": "Stimulus-sensitive; cortical origin; late hepatocerebral feature; associated with metabolic decompensation; clonazepam or LEV"},
        {"type": "Neonatal seizures (metabolic encephalopathy)", "n": 8, "pct": 20, "desc": "Early; metabolic trigger (hypoglycemia, lactic acidosis); clonic or subtle; often responds to glucose correction alone; EEG: burst-suppression in severe neonates"},
        {"type": "Infantile spasms (rare, secondary)", "n": 4, "pct": 10, "desc": "Rare; secondary to cortical injury; treat with ACTH or VGB standard protocol; NEVER use VPA for infantile spasms in DGUOK"},
        {"type": "Status epilepticus", "n": 6, "pct": 15, "desc": "Precipitated by metabolic decompensation (hypoglycemia, intercurrent illness, fasting); manage: IV dextrose + IV LEV 20-40 mg/kg + IV midazolam; NEVER propofol; anaesthesia: ketamine + sevoflurane"},
    ]

    metabolic_outcomes = [
        {"outcome": "Lactic acidosis (pH <7.1, lactate >10 mmol/L) — acute presentation", "n": 40, "pct": 100, "notes": "Universal; neonatal onset; OXPHOS failure → NAD⁺ regeneration failure → pyruvate → lactate; measure lactate:pyruvate ratio >20:1 = mitochondrial cause; IV dextrose + bicarbonate in acute crisis"},
        {"outcome": "Hepatomegaly (massive)", "n": 40, "pct": 100, "notes": "Hepatocyte mtDNA depletion → steatosis + hepatocyte swelling; liver >4 cm below costal margin at presentation; assess by USS"},
        {"outcome": "Coagulopathy (INR >1.3)", "n": 32, "pct": 80, "notes": "Synthetic liver failure; early and progressive; INR monitoring 2-weekly in stable outpatients; OLT evaluation trigger at INR >1.5"},
        {"outcome": "Jaundice / cholestasis", "n": 34, "pct": 85, "notes": "Direct hyperbilirubinemia; ursodeoxycholic acid for intrahepatic cholestasis; bilirubin trend informs OLT urgency"},
        {"outcome": "Hypoglycemia (<3.5 mmol/L at any point)", "n": 28, "pct": 70, "notes": "Hepatic gluconeogenesis failure; home glucose monitoring in all families; Dextrogel home supply; emergency sick-day glucose protocol"},
        {"outcome": "Hyperammonemia (NH3 >50 μmol/L) — mild secondary", "n": 12, "pct": 30, "notes": "MILD secondary hyperammonemia from urea cycle impairment in failing liver (unlike TMEM70 where NH3 50-500 is CARDINAL); if NH3 >200 μmol/L → reconsider DGUOK diagnosis vs TMEM70/organic acidaemia"},
    ]

    biomarker_summary = {
        "nystagmus_pct": 90,
        "hepatic_failure_pct": 85,
        "lactic_acidosis_pct": 100,
        "coagulopathy_pct": 80,
        "jaundice_pct": 85,
        "hypoglycemia_pct": 70,
        "hypotonia_pct": 85,
        "regression_pct": 80,
        "epilepsy_pct": 45,
        "three_mga_uria": "ABSENT (0%) — key DDx from all 3-MGA syndromes",
        "hepatic_only_form_pct": 25,
        "hepatocerebral_form_pct": 75,
        "olt_performed_pct": 25,
        "median_onset_days": 7,
        "median_diagnosis_delay_weeks": 6,
        "median_survival_hepatocerebral_months": 18,
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
            "median_survival_hepatocerebral_months": 18,
            "transplant_performed_pct": 25,
            "transplant_curative_hepatic_only_pct": 80,
            "neurological_progression_post_olt_hepatocerebral_pct": 95,
            "epilepsy_pct": 45,
            "severe_hypoglycemia_event_pct": 60,
            "language_acquisition_hepatic_only_post_olt_pct": 70,
            "nystagmus_persists_post_olt_pct": 75,
            "median_diagnosis_delay_weeks": 6,
        },
    }


def get_definitions() -> dict:
    """DGUOK Hepatocerebral mtDNA Depletion — definitions for /api/dguok/definitions."""
    return {
        "generated": date.today().isoformat(),
        "disease": "Mitochondrial DNA Depletion Syndrome 3 (MDDS3) / DGUOK Hepatocerebral Syndrome",
        "gene": "DGUOK",
        "omim_gene": "601465",
        "omim_disease": "251880",
        "definitions": [
            {
                "term": "DGUOK — Deoxyguanosine Kinase and the Mitochondrial Purine Salvage Pathway",
                "definition": "DGUOK (Deoxyguanosine Kinase; 277 aa; 2p13.1) encodes the mitochondrial matrix deoxyribonucleoside kinase responsible for phosphorylating purine deoxyribonucleosides: deoxyguanosine (dG) → dGMP (first step toward dGTP) and deoxyadenosine (dA) → dAMP (first step toward dATP). The 16 aa N-terminal mitochondrial targeting sequence (MTS) is cleaved on import. The mature 261 aa enzyme forms a homodimer with P-loop NTPase fold; the active site includes Arg47 (guanine binding), His86 (phosphate transfer), Asn165 and Tyr166 (ribose contacts). Unlike TK2 (which handles pyrimidines dC/dT), DGUOK exclusively handles purines (dG/dA). Together DGUOK + TK2 supply the entire mitochondrial dNTP pool via the salvage pathway — essential because mitochondria cannot synthesise dNTPs de novo and rely on cytoplasmic supply + salvage. POLG then uses this dNTP pool to replicate circular mtDNA (16.6 kb) by strand-displacement synthesis. Biallelic DGUOK LOF → insufficient dGTP + dATP → POLG stalls → mtDNA copy number falls below threshold → 13 mtDNA-encoded OXPHOS subunits become insufficient → Complex I/III/IV/V all fail → ATP production collapses → hepatocyte and neuron death.",
                "relevance": "DGUOK activity can be measured in patient fibroblasts using radiolabelled [³H]-deoxyguanosine phosphorylation assay. Residual activity <10% = hepatocerebral form; 10-35% = hepatic-only or attenuated. Liver biopsy: BN-PAGE shows absent or severely reduced Complex I + III + IV activities; mtDNA copy number by qPCR (ND1 probe vs GAPDH): <30% of controls = diagnostic. Muscle biopsy less sensitive in DGUOK than liver. Fibroblast DGUOK activity is the functional gold standard for VUS classification — submit to specialist mitochondrial biochemistry laboratory (NHNN London, Barts, Zurich, Nijmegen).",
            },
            {
                "term": "Two Clinical Forms of DGUOK Disease — Hepatocerebral vs Hepatic-Only",
                "definition": "DGUOK mutations produce two distinct clinical forms determined by residual enzyme activity: (1) HEPATOCEREBRAL FORM (~75% of cases): biallelic null or severe missense → <10% residual DGUOK activity → rapid mtDNA depletion in both liver (high mtDNA turnover) and brain (slower but progressive depletion) → neonatal/infantile liver failure + progressive neurological disease (nystagmus + hypotonia + psychomotor regression + epilepsy) → fatal by 1-4 years without liver transplant; OLT corrects hepatic disease but NOT brain depletion — neurological progression continues post-transplant and often accelerates from perioperative stress. (2) HEPATIC-ONLY FORM (~25% of cases): compound het or homozygous missense with ≥20% residual activity (e.g. p.Tyr166His genotype) → preferential liver depletion (liver has >10-fold higher mtDNA turnover than brain) → liver failure + nystagmus, but brain mtDNA depletion does not reach critical threshold during childhood → neurological development PRESERVED → liver transplant may be CURATIVE if performed before neurological compromise is established.",
                "relevance": "The distinction hepatocerebral vs hepatic-only drives the entire management strategy: (1) Liver transplantation decision — strongly supported in hepatic-only (Level B) with good neurological baseline; not indicated or harmful in late-stage hepatocerebral. (2) Prognosis counselling — hepatocerebral is progressive and fatal; hepatic-only after successful OLT can have near-normal life expectancy. (3) Timing — in hepatic-only, early OLT (before brain depletion becomes established) is critical; delay increases neurological risk. Clinical distinction is not always clear early: nystagmus is seen in both forms; EEG and MRI follow-up determines neurological trajectory. Genotype predicts form: null/null = hepatocerebral; Tyr166His-containing = often hepatic-only, but rule must be confirmed by neurological assessment — not assumed from genotype alone.",
            },
            {
                "term": "Nystagmus in DGUOK — Pathognomonic Early Sign and Its Mechanism",
                "definition": "Nystagmus — involuntary, rhythmic, oscillatory eye movements — occurs in 90% of DGUOK patients and is typically the earliest neurological sign, appearing in the first days to weeks of life, often before overt liver failure is recognised. Character: rotary (torsional) or pendular nystagmus; may be directional (jerk nystagmus) as disease advances. Mechanism: DGUOK-deficient cerebellar Purkinje cells and inferior olivary neurons (both high OXPHOS demand, high mtDNA copy number) deplete mtDNA early → OXPHOS failure → abnormal olivocerebellar circuit firing → oculomotor oscillation. Superior cerebellar vermis degeneration precedes cortical involvement in DGUOK — explains why nystagmus appears before generalised encephalopathy. MRI: early T2 signal in cerebellar vermis and inferior olivary nuclei; posterior fossa predominance in DGUOK contrasts with occipital cortex predominance in POLG/Alpers.",
                "relevance": "Any neonate with: (1) unexplained nystagmus + (2) hepatomegaly + (3) lactic acidosis → DGUOK sequencing MANDATORY same day; do not wait for LFT evolution. Nystagmus is the DGUOK clinical signature not seen in POLG/Alpers (where occipital epilepsy and EPC are the hallmarks). Nystagmus persists even after successful liver transplant — it is driven by autonomous cerebellar depletion and does not require hepatic failure to progress. Ophthalmology review: (1) measure visual acuity; (2) characterise nystagmus type (rotary vs pendular vs direction-changing); (3) check for null point (gaze angle where nystagmus is least prominent); (4) prism correction if null zone exists; (5) MRI posterior fossa to assess olivocerebellar degeneration.",
            },
            {
                "term": "Why No 3-MGA-uria in DGUOK — Critical DDx Tool",
                "definition": "3-methylglutaconic acid (3-MGA) is a dicarboxylic acid produced from leucine catabolism via methylglutaconyl-CoA hydratase (AUH). 3-MGA-uria (elevated urinary 3-MGA) is produced when mitochondrial inner membrane integrity is disrupted — specifically, the phospholipid remodelling and lipid composition of the IMM is perturbed, releasing methylglutaconyl-CoA intermediates aberrantly. 3-MGA-uria is the biochemical hallmark of a specific group of IMM/cristae disorders: TAZ (Barth, Type II), DNAJC19 (DCMA, Type IV), SERAC1 (MEGDEL, Type V), TMEM70 (Type VI), CLPB (Type VII), OPA3 (Costeff, Type III). DGUOK causes mtDNA depletion (matrix dNTP depletion) but does NOT disrupt IMM lipid remodelling — therefore NO 3-MGA-uria. POLG likewise produces no 3-MGA-uria. The absence of 3-MGA on urine organic acids is the fastest, cheapest, and most reliable bedside test to distinguish DGUOK/POLG hepatocerebral syndromes from all 3-MGA-uria syndromes.",
                "relevance": "Order urine organic acids in EVERY neonatal/infantile mitochondrial disease presentation. 3-MGA absent → favours DGUOK, POLG, Leigh syndrome, MELAS, MERRF, primary OXPHOS structural subunit defects. 3-MGA present → reconsider SERAC1, TAZ, TMEM70, OPA3, DNAJC19, CLPB. This DDx tree is rapid and actionable: if 3-MGA absent + neonatal liver failure + nystagmus + lactic acidosis → DGUOK panel FIRST; if 3-MGA absent + older child + EPC + hepatopathy → POLG FIRST; if 3-MGA present → 3-MGA syndrome panel. The distinction changes prescribing immediately (different contraindications, transplant indication, prognosis counselling).",
            },
            {
                "term": "Liver Transplantation in DGUOK — Curative in Hepatic-Only; Does NOT Prevent Neurological Progression in Hepatocerebral",
                "definition": "Liver transplantation (OLT) in DGUOK requires careful form-specific decision-making: HEPATIC-ONLY FORM: OLT replaces the DGUOK-deficient hepatocytes with a normal donor liver expressing wild-type DGUOK → hepatic mtDNA is restored → hepatic OXPHOS normalises → liver failure resolves → if neurological baseline is preserved at time of OLT, CNS does NOT develop significant depletion post-OLT (brain receives dNTPs from its own DGUOK which, in hepatic-only genotype, retains sufficient residual activity for CNS) → curative outcome (Level B evidence; published case series and systematic reviews). HEPATOCEREBRAL FORM: OLT corrects the hepatic depletion but brain mtDNA depletion is autonomous — neuronal DGUOK is biallelic null, producing no dGMP/dAMP → cerebellar and cortical mtDNA depletion continues post-OLT → encephalopathy, seizures, regression progress regardless of hepatic recovery. Published post-OLT hepatocerebral DGUOK series: neurological deterioration in >95% within 12 months of OLT despite good hepatic graft function. Outcomes: OLT extends survival from ~18mo to 3-5yr in hepatocerebral form — but neurological suffering is prolonged without benefit. Most experienced centres do NOT offer OLT to hepatocerebral DGUOK patients beyond palliation.",
                "relevance": "The transplant decision is the most critical, irreversible, ethically complex management choice in DGUOK: (1) Confirm form — hepatic-only vs hepatocerebral — using genotype + neurological assessment + MRI + EEG; (2) Timing — hepatic-only: early OLT (before neurological compromise), ideally before 18 months; (3) Multidisciplinary — metabolic genetics + hepatology + neurology + ethics + family; (4) Family counselling — must understand: hepatic-only OLT is curative (high bar for evidence); hepatocerebral OLT prolongs life without neurological benefit and may increase suffering; (5) Post-OLT monitoring — annual neurological + metabolic + ophthalmology + MRI assessment even in hepatic-only; long-term immunosuppression management.",
            },
            {
                "term": "Emergency Management: Metabolic Crisis in DGUOK (Hypoglycemia + Lactic Acidosis + SE)",
                "definition": "DGUOK metabolic crises are precipitated by: intercurrent illness (infection, fever), fasting, surgery, or progressive hepatic failure. Emergency protocol: (1) GLUCOSE FIRST — check capillary glucose immediately; <3.5 mmol/L: IV 10% dextrose bolus 2 mL/kg over 5min + continuous infusion GIR 8-10 mg/kg/min; do NOT use 50% dextrose bolus (hyperglycemia harmful); maintain glucose 5-10 mmol/L. (2) LACTIC ACIDOSIS — sodium bicarbonate only if pH <7.1 (0.5-1 mmol/kg IV over 2h); avoid overalkalization; IV dextrose is the primary lactic acidosis treatment (corrects OXPHOS substrate). (3) SEIZURES — IV levetiracetam 20-40 mg/kg loading; IV midazolam 0.15 mg/kg bolus + infusion; NEVER VPA, NEVER propofol anaesthesia; ketamine + sevoflurane if GA required. (4) LIVER PROTECTION — no hepatotoxic drugs; FFP if INR >1.5 before invasive procedures; check glucose + lactate + ammonia + INR + LFTs 4-hourly. (5) SPECIALIST ESCALATION — contact metabolic genetics on-call; admit to PICU; serial metabolic monitoring.",
                "relevance": "Every DGUOK family must hold an emergency letter stating: MITOCHONDRIAL DNA DEPLETION SYNDROME (DGUOK) — VALPROATE ABSOLUTELY CONTRAINDICATED — NO PROPOFOL — NO HIGH-FAT FEEDS — HYPOGLYCEMIA IS FATAL WITHOUT IMMEDIATE IV DEXTROSE — METABOLIC GENETICS ON-CALL: [NUMBER]. Letter registered in ED systems, hospital allergy flags, ambulance service. Every family trained in: (1) home glucose monitoring; (2) Dextrogel oral glucose gel for mild hypoglycemia; (3) buccal midazolam for seizures >5min; (4) when to call 999/emergency services. Metabolic crises are the leading cause of acute mortality in DGUOK — response in the first 30 minutes determines survival.",
            },
        ],
    }
