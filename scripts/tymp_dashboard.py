#!/usr/bin/env python3
"""TYMP MNGIE Dashboard — Mitochondrial Neurogastrointestinal Encephalomyopathy.

Mitochondrial DNA Depletion Syndrome 1 (MDDS1) = OMIM #603041 (MNGIE)
Also known as: TYMP-Deficiency / Thymidine Phosphorylase Deficiency / POLIP Syndrome

TYMP (Thymidine Phosphorylase; also PD-ECGF: Platelet-Derived Endothelial Cell Growth Factor;
482 aa; cytoplasmic homodimer; 22q13.32) encodes the enzyme that catalyses the reversible
phosphorolysis of thymidine and deoxyuridine to their respective bases + deoxyribose-1-phosphate.
TYMP is the ONLY human thymidine phosphorylase expressed ubiquitously outside the CNS.

Without TYMP:
  thymidine (dThd) accumulates in plasma: >3 µmol/L (normal <0.5 µmol/L) — PATHOGNOMONIC
  deoxyuridine (dU) accumulates in plasma: >5 µmol/L (normal <0.5 µmol/L) — ALSO PATHOGNOMONIC
  Nucleoside imbalance → dNTP pool distortion in post-mitotic cells (brain, muscle, GI neurons)
  Thymidine excess → dTTP pool EXCESS → allosteric feedback inhibition of ribonucleotide reductase
  → dCTP pool ↓ (unbalanced) → replication errors → mtDNA accumulates MULTIPLE DELETIONS
  (not primary depletion as in TK2/DGUOK; but depletion also occurs late)

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = CONTRAINDICATED — thymidine pool imbalance worsens mtDNA damage; hepatotoxicity
     risk in mitochondrial disease; do NOT use VPA in MNGIE/TYMP
  2. KD = CONTRAINDICATED — OXPHOS-dependent beta-oxidation impaired in mtDNA multiple deletions
  3. ADULT / ADOLESCENT ONSET — DISTINCTIVE vs paediatric MDDS; onset typically 15-40 years;
     KEY DDx from DGUOK/MPV17/SUCLA2/SUCLG1 (neonatal/infantile onset)
  4. GI DYSMOTILITY = HALLMARK (100%) — gastroparesis, intestinal pseudo-obstruction,
     malabsorption, episodic nausea/vomiting/diarrhoea/constipation; UNIQUE among MDDS
  5. LEUKOENCEPHALOPATHY (100%) — diffuse symmetric WM T2 hyperintensity on MRI;
     NOT Leigh-pattern (NOT basal ganglia T2); asymptomatic in most (detected by MRI)
  6. PERIPHERAL NEUROPATHY (80-90%) — sensorimotor, predominantly axonal; EMG: reduced
     amplitude motor/sensory; weakness + distal sensory loss
  7. CACHEXIA (80%) — profound weight loss; BMI <18.5; GI malabsorption is primary driver;
     KEY exam trap: clinicians attribute to eating disorder — test plasma thymidine first
  8. PTOSIS + PEO (50-60%) — progressive external ophthalmoplegia; bilateral ptosis +
     limited upgaze; KEY DDx from KSS (KSS has retinopathy; MNGIE does NOT)
  9. NO RETINOPATHY — KEY DDx from KSS (Kearns-Sayre: retinopathy 100% + PEO)
 10. NO STROKE-LIKE EPISODES — KEY DDx from MELAS (MELAS: SLEs + elevated CSF lactate)
 11. NO CARDIOMYOPATHY — KEY DDx from Kearns-Sayre (KSS: complete heart block 30%)
 12. Plasma thymidine >3 µmol/L: PATHOGNOMONIC — simplest, most accessible diagnostic test;
     order when: GI dysmotility + neuropathy + PEO + cachexia in adolescent/young adult
 13. Plasma deoxyuridine >5 µmol/L: ALSO ELEVATED — confirm both thymidine AND dU
 14. mtDNA from buffy coat: MULTIPLE DELETIONS (Southern blot / long-range PCR) — NOT
     depletion as primary finding (unlike SUCLA2/DGUOK where depletion is primary)
 15. HSCT (Haematopoietic Stem Cell Transplant) = ONLY PROVEN CURATIVE THERAPY —
     donor TYMP replaces enzyme in circulation → normalises plasma thymidine/dU; best
     outcomes if done before severe neurological disease (within 2-3 years of diagnosis)
 16. Peritoneal dialysis — removes thymidine/dU from circulation as BRIDGE to HSCT
 17. Propofol = AVOID (PRIS risk — mitochondrial disease universal rule)
 18. LEV preferred AED — renal excretion, no hepatic P450, no CoA/mtDNA interaction

TYMP BIOLOGY:
TYMP (482 amino acids; cytoplasmic homodimer; no MTS; 22q13.32) encodes thymidine
phosphorylase. The protein is identical to PD-ECGF (Platelet-Derived Endothelial Cell
Growth Factor), which was initially described as an angiogenic factor.

TYMP catalytic mechanism (reversible):
  Thymidine + phosphate → thymine + 2-deoxyribose-1-phosphate
  Deoxyuridine + phosphate → uracil + 2-deoxyribose-1-phosphate

TYMP protein domains:
  N-terminal alpha/beta barrel (phosphate-binding, substrate-binding): aa 1-~220
  C-terminal alpha-helical domain: aa ~220-482 (dimer interface; homodimer formation)
  Active site: Arg202 (phosphate coordination), Lys221, His116 (substrate binding)

MNGIE pathogenic mechanism:
  1. TYMP LOF → thymidine + dU accumulate in all tissues (plasma thymidine >3 µmol/L)
  2. Thymidine enters cells via equilibrative nucleoside transporters (ENT1, ENT2)
  3. Intracellular thymidine → phosphorylated to dTMP/dTDP/dTTP by TK1 (cytoplasmic) and
     TK2 (mitochondrial) → dTTP POOL EXCESS in post-mitotic cells
  4. dTTP excess → allosteric inhibition of ribonucleotide reductase (RNR) → dCTP pool ↓
  5. dTTP/dCTP imbalance → POLG replication errors → mtDNA multiple deletions ± depletion
  6. GI smooth muscle and ICC (interstitial cells of Cajal) are particularly vulnerable →
     severe enteric neuropathy → GI dysmotility (MNGIE hallmark)
  7. PNS axons accumulate deletions → peripheral neuropathy (axonal > demyelinating)
  8. CNS white matter: leukoencephalopathy (mechanism partially: WM metabolic vulnerability
     + thymidine toxicity to oligodendrocytes)

PLASMA THYMIDINE — DIAGNOSTIC GOLD STANDARD:
  Normal: <0.5 µmol/L (in plasma/serum)
  MNGIE: typically 5-20 µmol/L (some >40 µmol/L in null variants)
  Pre-analytical: stable in EDTA plasma on ice for 4h; freeze immediately for transport
  Method: HPLC (gold standard) or LC-MS/MS (more sensitive; some labs also measure dU)
  Rapid screen: any reference lab; critical result >3 µmol/L = diagnostic of TYMP deficiency
  Note: plasma thymidine is NORMAL in all other MDDS (TK2, DGUOK, SUCLA2, SUCLG1, FBXL4)
  — this single test distinguishes MNGIE biochemically from ALL other encephalomyopathies

GI MANIFESTATIONS (UNIQUE among all MDDS):
  Gastroparesis: delayed gastric emptying → postprandial nausea, vomiting, early satiety
  Intestinal pseudo-obstruction (IPO): failure of intestinal peristalsis → abdominal pain,
    distension, obstipation; may mimic surgical emergency (mechanical obstruction)
  Malabsorption: bacterial overgrowth (SIBO) secondary to slow motility → diarrhoea,
    steatorrhea, fat-soluble vitamin deficiencies (A, D, E, K)
  Cachexia: caloric insufficiency from malabsorption + vomiting → progressive weight loss;
    can reach BMI 12-14; muscle wasting + fat depletion
  Diverticulosis: small bowel diverticulae develop in chronic disease (~50%)
  Histology: smooth muscle atrophy + mitochondrial abnormalities in enteric neurons;
    absence of inflammation (DDx: IBD, vasculitis)

LEUKOENCEPHALOPATHY (DIFFUSE SYMMETRIC):
  MRI: T2/FLAIR hyperintensity — DIFFUSE, SYMMETRIC, BILATERAL white matter;
    predominantly periventricular and subcortical WM; NOT basal ganglia/brainstem (NOT Leigh)
  Clinically SILENT in most patients (detected incidentally on diagnostic MRI)
  Occasionally: mild cognitive slowing, executive dysfunction in advanced disease
  DDx: MELAS (WM + stroke-like cortical; asymmetric), KSS (WM + basal ganglia), MS (plaques)
  MNGIE leukoencephalopathy is characterised by its diffuse, symmetric, non-gadolinium-enhancing
  pattern that parallels the severity of thymidine accumulation

GENOTYPE-PHENOTYPE CORRELATION:
  Biallelic null (nonsense/frameshift, <1% residual TYMP activity):
    → severe; early onset (15-25 yrs); very high thymidine (>15 µmol/L); rapid GI
    progression; early cachexia; median survival 37-40 years without HSCT
  Splice-site / missense (5-20% residual TYMP activity):
    → intermediate; onset 20-35 yrs; moderate thymidine (5-15 µmol/L); slower progression
  Founder variants:
    p.Arg152Ter (c.454C>T): Japanese founder; null; severe/typical
    p.Tyr94Cys (c.281A>G): German/European; missense; variable
    p.Glu289Lys (c.865G>A): Palestinian/Middle Eastern
    p.Ala352Thr (c.1054G>A): Spanish; moderate

TREATMENT HIERARCHY:
  HSCT (allogeneic): CURATIVE — donor neutrophils provide continuous circulating TYMP;
    thymidine normalises within weeks; GI + neurological function stabilises/improves;
    best outcomes: myeloablative conditioning; matched sibling > MUD; pre-HSCT ECOG <3
  ERT (E. coli thymidine phosphorylase): INVESTIGATIONAL — reduces plasma thymidine;
    phase I/II trials; limited by immunogenicity and half-life; not approved
  Peritoneal dialysis: BRIDGE — removes thymidine/dU; stabilises during HSCT workup;
    continuous ambulatory or automated PD; partial correction only
  Enteral nutrition: ESSENTIAL — NG/PEG feeds for caloric support; anti-emetics;
    partial parenteral nutrition (TPN) if enteral not tolerated
  Prophylactic antibiotics: SIBO treatment — rifaximin, metronidazole, rotating courses;
    reduces bacterial overgrowth-related malabsorption
  Prokinetics: domperidone, metoclopramide (caution: EPS risk); erythromycin (motilin
    agonist); octreotide (paradoxically improves some); gastric electrical stimulation
  Riboflavin + CoQ10: supportive mitochondrial co-factors (low evidence in MNGIE)

KEY DDx PEARLS:
  vs KSS (Kearns-Sayre): KSS = PEO + retinopathy (100%) ± heart block; NO GI dysmotility;
    MNGIE = PEO + GI dysmotility; NO retinopathy; plasma thymidine NORMAL in KSS
  vs MELAS: MELAS = stroke-like episodes + elevated CSF lactate + RRF on biopsy;
    NO GI dysmotility; plasma thymidine NORMAL; MELAS maternal inheritance (mtDNA)
  vs POLG (Alpers/PEO-related): POLG = EPC 60% / hepatopathy 80% / VPA ABSOLUTE CI;
    NO GI dysmotility; mtDNA depletion (not multiple deletions); childhood onset
  vs IBD / Hirschsprung: IBD has mucosal inflammation, normal MRI WM; MNGIE has
    leukoencephalopathy + neuropathy + PEO absent in IBD; plasma thymidine distinguishes
  vs Eating disorder / anorexia nervosa: MNGIE cachexia = involuntary, GI mechanical;
    EMG shows neuropathy; MRI shows leukoencephalopathy; plasma thymidine diagnostic
  vs DGUOK (MDDS3): DGUOK = neonatal hepatocerebral; MNGIE = adolescent/adult GI-dominant
  vs SUCLA2/SUCLG1: both paediatric encephalomyopathic; MNGIE = adult + GI + no MMA/C4-DC

GENETICS:
  TYMP gene: 22q13.32; 10 exons; ~4.5 kb mRNA; 482 aa cytoplasmic homodimer protein
  OMIM Gene: 131222 | Disease OMIM: 603041 (MNGIE = MDDS1)
  Inheritance: Autosomal Recessive (AR); biallelic LOF
  Prevalence: ~1 in 1,000,000 (very rare; ~100-200 cases worldwide reported)
  TYMP activity: measured in buffy coat leukocytes (<10% normal = MNGIE; carrier = 20-50%)

REFERENCE: Hirano M et al. (1994) — first description of MNGIE;
  Nishino I et al. (1999) TYMP mutations in MNGIE, Ann Neurol;
  Garone C et al. (2011) HSCT outcomes in MNGIE, Ann Neurol;
  Martí R et al. (2004) — plasma thymidine diagnostic standard.

COHORT: 40-patient cohort, seed-565, simulated from published MNGIE literature.
"""

import random
from datetime import date

SEED = 565
TOTAL = 40


def _make_patients(rng: random.Random) -> list[dict]:
    """Generate 40 MNGIE (TYMP-deficiency) synthetic patients."""
    ethnicities = [
        "Japanese", "Japanese", "European", "European", "European",
        "Middle-Eastern", "Middle-Eastern", "Spanish", "Spanish",
        "North-African", "Turkish", "Italian", "Greek", "Pakistani",
        "Korean", "Chinese", "Argentinian", "Brazilian", "Palestinian",
    ]

    # Null variants cause severe/early; missense moderate; splice variable
    genotype_classes = [
        ("p.Arg152Ter / p.Arg152Ter (null/null)", "null", "Japanese founder"),
        ("p.Arg152Ter / p.Tyr94Cys (null/missense)", "mixed", "Japanese/European"),
        ("p.Tyr94Cys / p.Tyr94Cys (missense/missense)", "missense", "European"),
        ("p.Glu289Lys / p.Glu289Lys (null/null)", "null", "Middle-Eastern"),
        ("p.Ala352Thr / c.866-2A>G (missense/splice)", "mixed", "Spanish"),
        ("c.454C>T / c.866-2A>G (null/splice)", "mixed", "European"),
        ("Novel null / Novel null (biallelic null)", "null", "Turkish/consanguineous"),
        ("p.Tyr94Cys / p.Glu289Lys (missense/missense)", "missense", "compound het"),
    ]

    patients = []
    for i in range(TOTAL):
        gt_data = rng.choice(genotype_classes)
        genotype, gt_class, population = gt_data

        # Onset age: null earlier (15-25), missense later (25-40)
        if gt_class == "null":
            onset_age = rng.randint(14, 26)
            thymidine = round(rng.uniform(12.0, 42.0), 1)
            du = round(rng.uniform(8.0, 30.0), 1)
            tymp_activity = rng.uniform(0.1, 2.0)
        elif gt_class == "missense":
            onset_age = rng.randint(22, 40)
            thymidine = round(rng.uniform(4.5, 14.0), 1)
            du = round(rng.uniform(3.5, 10.0), 1)
            tymp_activity = rng.uniform(3.0, 10.0)
        else:  # mixed/splice
            onset_age = rng.randint(17, 35)
            thymidine = round(rng.uniform(6.0, 20.0), 1)
            du = round(rng.uniform(4.0, 14.0), 1)
            tymp_activity = rng.uniform(1.5, 6.0)

        current_age = onset_age + rng.randint(2, 25)
        bmi = round(rng.uniform(10.5, 21.0), 1)
        height = rng.randint(152, 182)

        # GI features (all have some form)
        gastroparesis = rng.random() < 0.92
        pseudo_obstruction = rng.random() < 0.75
        malabsorption = rng.random() < 0.88
        sibo = rng.random() < 0.65

        # Neurological features
        peripheral_neuropathy = rng.random() < 0.87
        peo = rng.random() < 0.58
        ptosis = peo or (rng.random() < 0.15)  # ptosis usually with PEO
        hearing_loss = rng.random() < 0.53
        leukoencephalopathy = True  # 100% on MRI
        cognitive_slowing = rng.random() < 0.30

        # Seizures: relatively rare in MNGIE
        seizures = rng.random() < 0.18

        # Treatment
        hsct_done = rng.random() < 0.30
        pd_therapy = (not hsct_done) and rng.random() < 0.25
        peg_tube = rng.random() < 0.60
        tpn = (not peg_tube) and rng.random() < 0.30

        patients.append({
            "id": f"MNGIE-{i+1:03d}",
            "sex": rng.choice(["M", "F"]),
            "ethnicity": rng.choice(ethnicities),
            "population_background": population,
            "genotype": genotype,
            "genotype_class": gt_class,
            "onset_age_yrs": onset_age,
            "current_age_yrs": min(current_age, 62),
            "plasma_thymidine_umolL": thymidine,
            "plasma_dU_umolL": du,
            "tymp_activity_pct_control": round(tymp_activity, 1),
            "bmi": bmi,
            "height_cm": height,
            # GI
            "gastroparesis": gastroparesis,
            "intestinal_pseudo_obstruction": pseudo_obstruction,
            "malabsorption": malabsorption,
            "sibo": sibo,
            # Neurological
            "leukoencephalopathy_mri": leukoencephalopathy,
            "peripheral_neuropathy": peripheral_neuropathy,
            "peo": peo,
            "ptosis": ptosis,
            "hearing_loss": hearing_loss,
            "cognitive_slowing": cognitive_slowing,
            "seizures": seizures,
            # Treatment
            "hsct_completed": hsct_done,
            "peritoneal_dialysis": pd_therapy,
            "peg_tube": peg_tube,
            "parenteral_nutrition": tpn,
        })

    return patients


def get_overview() -> dict:
    """TYMP MNGIE — summary KPIs for /api/tymp/overview."""
    rng = random.Random(SEED)
    patients = _make_patients(rng)
    total = len(patients)

    def pct(fn):
        return round(sum(1 for p in patients if fn(p)) / total * 100)

    # Plasma thymidine stats
    thymidines = [p["plasma_thymidine_umolL"] for p in patients]
    mean_thymidine = round(sum(thymidines) / len(thymidines), 1)
    median_thymidine = round(sorted(thymidines)[len(thymidines) // 2], 1)

    onset_ages = [p["onset_age_yrs"] for p in patients]
    mean_onset = round(sum(onset_ages) / len(onset_ages), 1)

    bmis = [p["bmi"] for p in patients]
    mean_bmi = round(sum(bmis) / len(bmis), 1)

    return {
        "generated": date.today().isoformat(),
        "disease": "TYMP MNGIE (MDDS1)",
        "gene": "TYMP",
        "protein": "Thymidine Phosphorylase (PD-ECGF)",
        "protein_length_aa": 482,
        "chromosomal_location": "22q13.32",
        "omim_gene": "131222",
        "omim_disease": "603041",
        "inheritance": "Autosomal Recessive (AR)",
        "disease_full": "MNGIE — Mitochondrial Neurogastrointestinal Encephalomyopathy",
        "mdds_number": "MDDS1",
        "prevalence_per_million": 1,
        "cohort_n": total,
        "seed": SEED,
        "mean_onset_age_yrs": mean_onset,
        "mean_plasma_thymidine_umolL": mean_thymidine,
        "median_plasma_thymidine_umolL": median_thymidine,
        "mean_bmi": mean_bmi,
        "kpis": [
            {
                "label": "GI Dysmotility",
                "value": "100%",
                "note": "Universal — gastroparesis + pseudo-obstruction; hallmark of MNGIE",
            },
            {
                "label": "Leukoencephalopathy",
                "value": "100%",
                "note": "Diffuse symmetric WM T2 hyperintensity; NOT Leigh pattern",
            },
            {
                "label": "Peripheral Neuropathy",
                "value": f"{pct(lambda p: p['peripheral_neuropathy'])}%",
                "note": "Sensorimotor axonal; EMG: reduced amplitude; distal weakness+sensory loss",
            },
            {
                "label": "Cachexia (BMI<18.5)",
                "value": f"{pct(lambda p: p['bmi'] < 18.5)}%",
                "note": "GI malabsorption + vomiting → profound weight loss; KEY presenting feature",
            },
            {
                "label": "PEO / Ptosis",
                "value": f"{pct(lambda p: p['peo'] or p['ptosis'])}%",
                "note": "Progressive external ophthalmoplegia ± ptosis; NO retinopathy (DDx KSS)",
            },
            {
                "label": "SNHL",
                "value": f"{pct(lambda p: p['hearing_loss'])}%",
                "note": "Sensorineural hearing loss; audiogram at diagnosis",
            },
            {
                "label": "Plasma dThd >3µmol/L",
                "value": "100%",
                "note": "PATHOGNOMONIC — plasma thymidine diagnostic test; normal <0.5 µmol/L",
            },
            {
                "label": "HSCT Completed",
                "value": f"{pct(lambda p: p['hsct_completed'])}%",
                "note": "Only curative therapy; best outcomes pre-severe neurological disease",
            },
        ],
        "prescribing_summary": {
            "vpa": "CONTRAINDICATED — thymidine pool worsens mtDNA multiple deletions; hepatotoxicity risk",
            "kd": "CONTRAINDICATED — OXPHOS-dependent beta-oxidation impaired in mtDNA deletion disease",
            "propofol": "AVOID — PRIS risk in mitochondrial disease; use sevoflurane or ketamine",
            "aed_of_choice": "LEV (Levetiracetam) — renal excretion; no hepatic P450; no CoA interaction",
            "curative": "HSCT (allogeneic) — corrects systemic TYMP; normalises plasma thymidine",
            "bridge_therapy": "Peritoneal dialysis — removes thymidine/dU; bridge to HSCT",
            "gi_support": "Enteral/parenteral nutrition + prokinetics + SIBO antibiotics + PEG tube",
        },
        "key_diagnostic_test": {
            "test": "Plasma thymidine (HPLC / LC-MS/MS)",
            "normal": "<0.5 µmol/L",
            "mngie_range": ">3 µmol/L (typically 5-40 µmol/L)",
            "also_order": "Plasma deoxyuridine (dU) — also elevated >5 µmol/L",
            "confirmation": "TYMP enzyme activity in buffy coat (<10% of normal = diagnostic)",
            "mtdna": "Buffy coat mtDNA: multiple deletions on Southern blot / long-range PCR",
            "note": (
                "This single plasma test distinguishes MNGIE biochemically from ALL other MDDS. "
                "Thymidine is NORMAL in TK2, DGUOK, SUCLA2, SUCLG1, FBXL4, POLG, MPV17. "
                "Order when: GI dysmotility + neuropathy + PEO/ptosis + cachexia in adolescent/adult."
            ),
        },
    }


def get_breakdown() -> dict:
    """TYMP MNGIE — detailed breakdown for /api/tymp/breakdown."""
    rng = random.Random(SEED)
    patients = _make_patients(rng)
    total = len(patients)

    def pct(fn):
        return round(sum(1 for p in patients if fn(p)) / total * 100)

    # Genotype class distribution
    from collections import Counter
    gt_class_count = Counter(p["genotype_class"] for p in patients)
    gt_dist = [
        {
            "class": cls,
            "n": cnt,
            "pct": round(cnt / total * 100),
            "onset_note": (
                "null/null: earliest onset 14-26 yrs; highest thymidine >12 µmol/L; most severe"
                if cls == "null" else
                "missense/missense: latest onset 22-40 yrs; lowest thymidine 4.5-14 µmol/L; slowest"
                if cls == "missense" else
                "mixed (null/splice, null/missense): intermediate onset 17-35 yrs; moderate thymidine"
            ),
        }
        for cls, cnt in gt_class_count.most_common()
    ]

    # Feature prevalence
    feature_prevalence = [
        {
            "feature": "GI Dysmotility — Gastroparesis",
            "pct": pct(lambda p: p["gastroparesis"]),
            "note": (
                "Delayed gastric emptying → postprandial nausea, early satiety, vomiting. "
                "Gastric scintigraphy (4h test): >10% retention at 4h diagnostic. "
                "Treat: domperidone / erythromycin; gastric electrical stimulation in refractory. "
                "KEY: gastroparesis in a young adult with neuropathy + PEO = test plasma thymidine."
            ),
        },
        {
            "feature": "Intestinal Pseudo-Obstruction",
            "pct": pct(lambda p: p["intestinal_pseudo_obstruction"]),
            "note": (
                "Failure of intestinal peristalsis → abdominal distension, pain, obstipation. "
                "X-ray/CT: dilated bowel loops; air-fluid levels; NO mechanical transition point. "
                "CRITICAL EXAM TRAP: MNGIE IPO mimics surgical abdomen — laparotomy contraindicated "
                "(no mechanical obstruction; surgical stress risks metabolic decompensation). "
                "Manage medically: NG decompression, octreotide, bowel rest, bridge to HSCT."
            ),
        },
        {
            "feature": "Malabsorption",
            "pct": pct(lambda p: p["malabsorption"]),
            "note": (
                "SIBO (small intestinal bacterial overgrowth) from slow motility → malabsorption. "
                "Steatorrhoea, fat-soluble vitamin deficiencies (A, D, E, K), diarrhoea. "
                "Hydrogen breath test: often positive. Treat: rifaximin / rotating antibiotics. "
                "Supplement fat-soluble vitamins. Elemental feeds may improve absorption."
            ),
        },
        {
            "feature": "Leukoencephalopathy (MRI)",
            "pct": 100,
            "note": (
                "Diffuse symmetric bilateral WM T2/FLAIR hyperintensity — hallmark, 100% on MRI. "
                "Pattern: periventricular + subcortical; NO basal ganglia/brainstem T2 (NOT Leigh). "
                "Clinically silent in most; some show cognitive slowing in advanced disease. "
                "Non-gadolinium-enhancing; does NOT respond to steroids (DDx: MS, ADEM). "
                "Severity tracks plasma thymidine levels; improves with HSCT in some."
            ),
        },
        {
            "feature": "Peripheral Neuropathy",
            "pct": pct(lambda p: p["peripheral_neuropathy"]),
            "note": (
                "Sensorimotor, predominantly axonal (reduced amplitude EMG/NCS); some demyelinating. "
                "Distal greater than proximal; lower > upper limbs; loss of ankle reflexes. "
                "NCS: reduced CMAP/SNAP amplitude; nerve conduction velocity mildly-moderately reduced. "
                "Progressive; may lead to foot drop; AFOs for ambulatory patients. "
                "Neuropathic pain: gabapentin / pregabalin; avoid opioids if GI dysmotility severe."
            ),
        },
        {
            "feature": "Cachexia (BMI <18.5)",
            "pct": pct(lambda p: p["bmi"] < 18.5),
            "note": (
                "Profound weight loss from GI malabsorption + vomiting + anorexia. "
                "BMI often 12-16 at nadir. Muscle wasting + fat depletion. "
                "PEG tube placement recommended when BMI <18 or oral intake <75% requirements. "
                "Nocturnal enteral feeds often needed in addition to oral intake. "
                "KEY DDx from eating disorder: MNGIE cachexia is involuntary + GI mechanical."
            ),
        },
        {
            "feature": "PEO / Ptosis",
            "pct": pct(lambda p: p["peo"] or p["ptosis"]),
            "note": (
                "Progressive external ophthalmoplegia (PEO): bilateral limited upgaze + horizontal. "
                "Ptosis: bilateral, progressive; ptosis precedes PEO in some. "
                "NO retinopathy — KEY DDx from KSS (KSS: retinopathy 100%). "
                "NO cardiac involvement — KEY DDx from KSS (KSS: complete heart block 30%). "
                "Diplopia rare (bilateral symmetric PEO rarely causes diplopia). "
                "Strabismus surgery generally not required."
            ),
        },
        {
            "feature": "Sensorineural Hearing Loss",
            "pct": pct(lambda p: p["hearing_loss"]),
            "note": (
                "SNHL detected on pure-tone audiogram; often bilateral and symmetric. "
                "Onset usually after GI symptoms. Hearing aids; cochlear implant if severe. "
                "Annual audiometry recommended from diagnosis. Monitoring critical as SNHL "
                "worsens with systemic disease progression."
            ),
        },
        {
            "feature": "Cognitive Slowing",
            "pct": pct(lambda p: p["cognitive_slowing"]),
            "note": (
                "Mild executive dysfunction and cognitive slowing in advanced disease. "
                "Related to leukoencephalopathy burden. Usually NOT dementia in early stages. "
                "Neuropsychological testing: frontal/executive deficit pattern. "
                "MNGIE patients remain socially aware + emotionally intact longer than WM suggests."
            ),
        },
        {
            "feature": "Seizures",
            "pct": pct(lambda p: p["seizures"]),
            "note": (
                "Seizures are NOT a primary feature of MNGIE (~18%); occur in advanced disease. "
                "Focal seizures more common than generalised. LEV preferred AED. "
                "VPA CONTRAINDICATED — worsens mtDNA damage via thymidine + CoA sequestration. "
                "EEG: generally diffuse slowing rather than epileptiform; MRI leukoencephalopathy."
            ),
        },
        {
            "feature": "SIBO (Small Intestinal Bacterial Overgrowth)",
            "pct": pct(lambda p: p["sibo"]),
            "note": (
                "Secondary to slow intestinal motility → bacterial stasis → overgrowth. "
                "Symptoms: bloating, diarrhoea, abdominal cramps, worsening malabsorption. "
                "Diagnose: hydrogen/methane breath test; empiric treatment acceptable in MNGIE. "
                "Treat: rifaximin 200-400 mg TID x 7-14d; rotate antibiotics to prevent resistance."
            ),
        },
        {
            "feature": "PEG Tube",
            "pct": pct(lambda p: p["peg_tube"]),
            "note": (
                "Percutaneous endoscopic gastrostomy for supplemental or full enteral nutrition. "
                "Indicated when BMI <18 or oral intake insufficient. "
                "Continuous nocturnal feeds improve caloric intake; formula: polymeric or semi-elemental. "
                "Anti-emetics required for gastrostomy feeds in gastroparesis. "
                "KD CONTRAINDICATED — high-fat formula absolutely avoided in MNGIE."
            ),
        },
        {
            "feature": "HSCT Completed",
            "pct": pct(lambda p: p["hsct_completed"]),
            "note": (
                "Allogeneic haematopoietic stem cell transplantation — only proven curative therapy. "
                "Donor neutrophils provide circulating TYMP → thymidine normalises within weeks. "
                "Best outcomes: ECOG performance status <3 at time of HSCT; early treatment. "
                "Matched sibling donor preferred; MUD (matched unrelated donor) acceptable. "
                "Post-HSCT: GI function improves in 6-24 months; neuropathy stabilises (partial). "
                "Mortality risk: myeloablative conditioning in cachectic/malnourished patient is HIGH — "
                "nutritional optimisation pre-conditioning is mandatory."
            ),
        },
    ]

    treatments = [
        {
            "tx": "HSCT (Allogeneic Haematopoietic Stem Cell Transplant)",
            "level": "A — Only Proven Curative Therapy",
            "note": (
                "Allogeneic HSCT corrects TYMP deficiency systemically: donor neutrophils/monocytes "
                "provide TYMP activity in circulation → plasma thymidine normalises → dNTP rebalance "
                "→ stops ongoing mtDNA damage. "
                "Recommended conditioning: myeloablative (busulfan/cyclophosphamide or fludarabine/busulfan). "
                "Pre-HSCT optimisation: nutritional support (TPN if needed), treat active infections, "
                "PFTs (pulmonary reserve), cardiac echo (mitochondrial cardiomyopathy exclusion). "
                "Early HSCT (within 2-3 years of symptom onset, before severe neurological involvement) "
                "gives best outcomes. Garone 2011: 24 HSCT patients, 50% 3-year survival; mortality "
                "primarily from conditioning toxicity in malnourished patients."
            ),
        },
        {
            "tx": "Peritoneal Dialysis (PD) — Bridge to HSCT",
            "level": "B — Bridge / Not Curative",
            "note": (
                "Continuous ambulatory or automated PD removes thymidine and deoxyuridine from blood "
                "by equilibration with dialysate → partial reduction in plasma thymidine (typically "
                "50-70% reduction; does NOT reach normal). "
                "Indicated as bridge to HSCT or for patients not HSCT-eligible. "
                "Reduces metabolic burden + allows partial nutritional improvement pre-HSCT. "
                "Complications: peritonitis, mechanical complications of PD catheter."
            ),
        },
        {
            "tx": "ERT — E. coli Thymidine Phosphorylase Enzyme",
            "level": "C — Investigational; Not Approved",
            "note": (
                "E. coli TP enzyme replacement reduces plasma thymidine in phase I/II studies. "
                "Limitations: rapid immunogenicity, short half-life, repeated dosing needed. "
                "PEGylation strategies reduce immunogenicity (ongoing research). "
                "Not currently available outside clinical trials. Consult MitoAction/UMDF for trials."
            ),
        },
        {
            "tx": "Enteral Nutrition (PEG Tube) + Anti-emetics",
            "level": "A — Mandatory Nutritional Support",
            "note": (
                "PEG tube for supplemental or full enteral nutrition. Target BMI ≥18.5 pre-HSCT. "
                "Continuous nocturnal feeds; polymeric formula preferred. "
                "Fat content: moderate (NOT high fat — KD absolutely contraindicated). "
                "Anti-emetics: ondansetron (5-HT3 antagonist) first-line; avoid metoclopramide "
                "(tardive dyskinesia risk with long-term use); domperidone where available. "
                "Prokinetics: erythromycin (motilin agonist) 3mg/kg/dose TID (short-term); "
                "octreotide (somatostatin analogue, paradoxically improves motility in some MNGIE). "
                "Parenteral nutrition (TPN) if enteral route not tolerated or intestinal failure."
            ),
        },
        {
            "tx": "SIBO Treatment — Rotating Antibiotics",
            "level": "A — Standard of Care",
            "note": (
                "Rifaximin 200-400 mg TID x 7-14 days; non-absorbed gut antibiotic; first-line. "
                "Rotate: metronidazole 500 mg TID x 7 days (second course). "
                "Rotate: amoxicillin-clavulanate 875/125 mg BID x 7 days (third). "
                "Ongoing rotating monthly courses often required in severe SIBO. "
                "Lactobacillus-based probiotics adjunct (limited evidence in MNGIE). "
                "Monitor: folate, B12, fat-soluble vitamins; supplement as needed."
            ),
        },
        {
            "tx": "LEV (Levetiracetam) — AED if Needed",
            "level": "A — Preferred if Seizures Occur",
            "note": (
                "Seizures in MNGIE: focal > generalised; LEV 10-20 mg/kg/day → 40-60 mg/kg/day. "
                "Renal excretion — safe in MNGIE patients with poor nutritional status. "
                "VPA ABSOLUTELY CONTRAINDICATED in MNGIE. "
                "LZP/CZP IV for status epilepticus. "
                "EEG to characterise seizure type before initiating AED. "
                "Seizures may improve with HSCT as metabolic burden reduces."
            ),
        },
        {
            "tx": "Riboflavin + CoQ10 — Mitochondrial Cofactors",
            "level": "C — Supportive",
            "note": (
                "Standard mitochondrial supplement co-factors. Low risk. "
                "Riboflavin (B2) 100 mg/day; CoQ10 (ubiquinol preferred) 10-30 mg/kg/day. "
                "No MNGIE-specific controlled data. Include in empirical mitochondrial support "
                "pending HSCT. Does not substitute HSCT."
            ),
        },
        {
            "tx": "Fat-Soluble Vitamin Supplementation",
            "level": "A — Mandatory in Malabsorption",
            "note": (
                "Malabsorption leads to deficiencies of vitamins A, D, E, K. "
                "Supplement: vitamin D3 2000-4000 IU/day (target >75 nmol/L); "
                "vitamin K1 10 mg/day or vitamin K2 menaquinone; vitamin E 400-800 IU/day; "
                "vitamin A 5000-10000 IU/day (monitor for toxicity). "
                "Monitor: 25-OH-D, PT/INR (vitamin K), serum retinol, alpha-tocopherol annually."
            ),
        },
        {
            "tx": "Genetic Counselling",
            "level": "A — Mandatory",
            "note": (
                "AR inheritance: 25% recurrence. Prenatal diagnosis via CVS or amniocentesis "
                "for known TYMP family variants. Preimplantation genetic testing (PGT-M) available. "
                "Carrier testing: TYMP enzyme activity in buffy coat (carriers: 20-50% normal). "
                "Siblings of probands: test TYMP activity regardless of symptoms — "
                "early diagnosis enables earlier HSCT before disease progression."
            ),
        },
        {
            "tx": "Gastric Electrical Stimulation (GES)",
            "level": "C — Refractory Gastroparesis",
            "note": (
                "For refractory gastroparesis unresponsive to pharmacological prokinetics. "
                "Surgically implanted device (Enterra, Medtronic); reduces nausea/vomiting. "
                "Case reports positive in MNGIE; not in routine use. "
                "Refer to specialist GI motility centre for assessment."
            ),
        },
    ]

    disease_timeline = [
        {
            "phase": "Onset (typically 15-25 yrs in null; 22-40 yrs in missense)",
            "events": (
                "First symptom usually GI: unexplained nausea, episodic vomiting, early satiety, "
                "weight loss (initially attributed to eating disorder, functional dyspepsia, IBD). "
                "Leukoencephalopathy already present asymptomatically on MRI (if done). "
                "Peripheral neuropathy may be subclinical at onset (EMG picks up early changes). "
                "Plasma thymidine first-line investigation — >3 µmol/L confirms MNGIE; refer metabolics."
            ),
        },
        {
            "phase": "Early Disease (2-5 years post-onset)",
            "events": (
                "Progressive GI dysmotility: gastroparesis worsening → PEG tube insertion. "
                "Weight loss accelerating (BMI 18 → 15); cachexia prominent. "
                "SIBO treated; rotating antibiotics begin. "
                "PEO/ptosis developing; ophthalmology referral. "
                "Neuropathy symptomatic: distal sensory loss, ankle areflexia, EMG confirms axonal. "
                "HSCT workup initiated: HLA typing patient + family, bone marrow biopsy, "
                "organ function assessment (cardiac echo, PFTs, GFR, LFTs). "
                "Peritoneal dialysis as bridge if thymidine very high while awaiting HSCT."
            ),
        },
        {
            "phase": "HSCT (if performed, ideally <5 years onset)",
            "events": (
                "Myeloablative conditioning (busulfan/fludarabine-based). "
                "Pre-conditioning: TPN for 4-6 weeks to optimise nutritional status. "
                "Engraftment: D+14 to D+28; donor TYMP activity in buffy coat confirms engraftment. "
                "Post-HSCT: plasma thymidine normalises within weeks of full engraftment. "
                "GI improvement: gastroparesis/IPO begins to resolve over 6-24 months post-HSCT. "
                "Weight gain: gradual; BMI improves over 12-24 months. "
                "Neuropathy: stabilises; some improvement in 1-2 years; not fully reversible. "
                "Leukoencephalopathy: MRI lesions may improve partially over 2-5 years. "
                "Lifelong follow-up: TYMP activity, plasma thymidine, annual MRI, neuro review."
            ),
        },
        {
            "phase": "Advanced Disease (without HSCT, 10-20 yrs onset)",
            "events": (
                "Severe cachexia (BMI <14); TPN dependence. "
                "Recurrent IPO episodes requiring hospitalisation, NG decompression. "
                "Wheelchair dependence from neuropathy + deconditioning. "
                "Progressive cognitive slowing; MMSE may decline in 3rd/4th decade of disease. "
                "Seizures develop (focal; LEV treatment). "
                "Hearing aids for SNHL. Vocational disability. "
                "Median survival without HSCT: literature suggests 5th decade (37-42 years); "
                "actual range highly variable (early 3rd decade to late 5th decade). "
                "Palliative care planning: TPN, symptom management, quality of life focus."
            ),
        },
    ]

    gi_profile = {
        "gastroparesis": {
            "prevalence": "~90-95%",
            "mechanism": "Mitochondrial dysfunction in ICC (interstitial cells of Cajal) + GI smooth muscle",
            "diagnosis": "4-hour gastric scintigraphy; gastric manometry; breath test (13C-octanoate)",
            "treatment": "Domperidone / erythromycin / octreotide / GES; PEG bypass; TPN",
        },
        "intestinal_pseudo_obstruction": {
            "prevalence": "~70-80%",
            "mechanism": "Enteric neuropathy + smooth muscle atrophy → failure of peristaltic reflex",
            "diagnosis": "Plain X-ray / CT abdomen: dilated bowel, no mechanical transition point",
            "critical_note": "NEVER operate for pseudo-obstruction — no mechanical obstruction present",
            "treatment": "NG decompression; bowel rest; octreotide 50-100 µg SC TID; neostigmine IV (acute)",
        },
        "sibo": {
            "prevalence": "~60-70%",
            "mechanism": "Slow intestinal transit → bacterial stasis → overgrowth",
            "diagnosis": "Hydrogen/methane breath test; jejunal aspirate culture (>10^5 CFU/mL)",
            "treatment": "Rifaximin; rotating antibiotics; probiotics adjunct",
        },
        "diverticulosis": {
            "prevalence": "~50% in chronic disease",
            "type": "Small bowel diverticulae (atypical location — duodenal/jejunal)",
            "note": "Rarely complicated; distinguishing from Meckel's diverticulum important",
        },
    }

    mri_profile = {
        "pattern": "Diffuse symmetric bilateral WM T2/FLAIR hyperintensity",
        "location": "Periventricular + subcortical white matter; centrifugal distribution",
        "spared": "Cortical grey matter, basal ganglia, brainstem (NOT Leigh-pattern)",
        "gadolinium": "Non-enhancing (distinguishes from inflammatory/demyelinating disease)",
        "clinical_correlation": "Often clinically silent; mild cognitive slowing in advanced",
        "post_hsct": "May partially improve over 2-5 years after successful HSCT",
        "ddx_mri": {
            "vs_leigh": "Leigh: BG + brainstem T2; MNGIE: WM only; different distributions",
            "vs_kss": "KSS: cerebellum + BG + WM; MNGIE: diffuse WM only; NO BG",
            "vs_melas": "MELAS: cortical SLE lesions; MNGIE: WM only; NO cortical stroke",
            "vs_ms": "MS: periventricular plaques; enhancing; MNGIE: non-enhancing diffuse",
        },
    }

    plasma_thymidine = {
        "diagnostic_threshold": ">3.0 µmol/L (normal <0.5 µmol/L)",
        "typical_mngie_range": "5-40 µmol/L",
        "deoxyuridine": ">5.0 µmol/L (normal <0.5 µmol/L); ALSO elevated in MNGIE",
        "method": "HPLC (gold standard) or LC-MS/MS (more sensitive)",
        "pre_analytical": "EDTA plasma; process within 4h or freeze; avoid haemolysis",
        "interpretation": {
            "mild": "3-8 µmol/L: usually missense / some residual activity; moderate phenotype",
            "moderate": "8-15 µmol/L: mixed genotype; intermediate severity",
            "severe": ">15 µmol/L: null/null; severe phenotype; early onset; rapid progression",
        },
        "post_hsct_target": "<0.5 µmol/L (complete normalisation = full engraftment)",
        "pd_effect": "50-70% reduction; does NOT fully normalise to <0.5 µmol/L",
        "key_note": (
            "Plasma thymidine is NORMAL in all other MDDS (TK2, DGUOK, SUCLA2, SUCLG1, "
            "FBXL4, POLG, MPV17, TWNK, RRM2B). A single plasma thymidine test biochemically "
            "distinguishes MNGIE from the entire MDDS differential. Order it first."
        ),
    }

    return {
        "generated": date.today().isoformat(),
        "disease": "TYMP MNGIE (MDDS1)",
        "cohort_n": total,
        "seed": SEED,
        "patients_sample": patients[:8],
        "genotype_distribution": gt_dist,
        "feature_prevalence": feature_prevalence,
        "treatments": treatments,
        "disease_timeline": disease_timeline,
        "gi_profile": gi_profile,
        "mri_leukoencephalopathy_profile": mri_profile,
        "plasma_thymidine_diagnostic": plasma_thymidine,
        "ddx_summary": {
            "vs_kss": (
                "KSS: PEO + RETINOPATHY (100%) ± heart block; NO GI dysmotility; "
                "mtDNA large single deletion; plasma thymidine NORMAL in KSS"
            ),
            "vs_melas": (
                "MELAS: stroke-like episodes + elevated CSF lactate + RRF; "
                "maternal inheritance (m.3243A>G); NO GI dysmotility; thymidine NORMAL"
            ),
            "vs_polg_peo": (
                "POLG-PEO: EPC 60% / hepatopathy 80%; childhood/adult; NO GI dysmotility hallmark; "
                "mtDNA depletion NOT deletion; thymidine NORMAL"
            ),
            "vs_ibd": (
                "IBD (Crohn's, UC): mucosal inflammation on endoscopy; normal WM MRI; "
                "normal EMG; normal thymidine; no PEO/ptosis"
            ),
            "vs_anorexia": (
                "Anorexia nervosa: normal EMG, normal MRI WM, normal thymidine; "
                "MNGIE cachexia involuntary — GI mechanical drive, not behavioural"
            ),
            "mngie_unique_triad": (
                "GI dysmotility + leukoencephalopathy + peripheral neuropathy = "
                "MNGIE until proven otherwise → plasma thymidine STAT"
            ),
        },
    }


def get_definitions() -> dict:
    """TYMP MNGIE — clinical definitions for /api/tymp/definitions."""
    return {
        "generated": date.today().isoformat(),
        "disease": "TYMP MNGIE (MDDS1)",
        "terms": [
            {
                "term": "TYMP (Thymidine Phosphorylase / PD-ECGF)",
                "definition": (
                    "TYMP (482 aa; cytoplasmic homodimer; 22q13.32) encodes thymidine phosphorylase, "
                    "the enzyme that phosphorolyses thymidine and deoxyuridine to their respective bases "
                    "plus deoxyribose-1-phosphate. TYMP is also known as PD-ECGF (Platelet-Derived "
                    "Endothelial Cell Growth Factor) due to its angiogenic activity, though enzymatic "
                    "function (not angiogenic signalling) is primary in MNGIE pathogenesis. "
                    "TYMP is ubiquitously expressed outside the CNS; the only human cytoplasmic "
                    "thymidine phosphorylase. Biallelic LOF causes systemic thymidine accumulation."
                ),
            },
            {
                "term": "MNGIE (Mitochondrial Neurogastrointestinal Encephalomyopathy)",
                "definition": (
                    "OMIM #603041. Autosomal recessive disease caused by biallelic TYMP mutations "
                    "leading to thymidine phosphorylase deficiency. MNGIE = MDDS1. "
                    "Classic pentad: (1) GI dysmotility (gastroparesis, pseudo-obstruction), "
                    "(2) cachexia, (3) peripheral neuropathy, (4) ptosis/PEO, "
                    "(5) diffuse leukoencephalopathy. "
                    "Onset: typically adolescence to young adulthood (15-40 years). "
                    "DISTINCTIVE among MDDS: adult onset, GI dominant, plasma thymidine diagnostic. "
                    "First described: Hirano M et al. (1994); TYMP mutations: Nishino et al. (1999)."
                ),
            },
            {
                "term": "Plasma Thymidine — Pathognomonic Biomarker",
                "definition": (
                    "Plasma thymidine >3 µmol/L (normal <0.5 µmol/L) is PATHOGNOMONIC for MNGIE. "
                    "Simultaneously: deoxyuridine (dU) also elevated >5 µmol/L (normal <0.5). "
                    "Measured by HPLC or LC-MS/MS from EDTA plasma. "
                    "This single test biochemically distinguishes MNGIE from all other MDDS — "
                    "thymidine is NORMAL in TK2, DGUOK, MPV17, TWNK, SUCLA2, SUCLG1, FBXL4, POLG. "
                    "Severity: >15 µmol/L correlates with null/null genotype + severe phenotype. "
                    "Post-HSCT: normalises to <0.5 µmol/L confirming cure."
                ),
            },
            {
                "term": "Leukoencephalopathy",
                "definition": (
                    "Diffuse symmetric bilateral white matter (WM) T2/FLAIR hyperintensity on MRI. "
                    "In MNGIE: periventricular + subcortical WM; NOT basal ganglia / brainstem "
                    "(differentiates from Leigh syndrome). Non-gadolinium-enhancing. Clinically "
                    "silent in most patients at diagnosis. Mechanism: thymidine toxicity to "
                    "oligodendrocytes + WM metabolic vulnerability from mtDNA deletion-mediated "
                    "OXPHOS impairment. Improves partially after successful HSCT over 2-5 years."
                ),
            },
            {
                "term": "Intestinal Pseudo-Obstruction (IPO)",
                "definition": (
                    "Failure of intestinal peristalsis without mechanical obstruction, resulting in "
                    "abdominal distension, pain, and obstipation mimicking surgical obstruction. "
                    "In MNGIE: caused by enteric neuropathy + smooth muscle atrophy from mtDNA "
                    "deletions in GI neurons. Radiology: dilated bowel loops, air-fluid levels, "
                    "NO transition point (mechanical obstruction). "
                    "CRITICAL: do NOT operate — no mechanical cause; surgical stress can precipitate "
                    "metabolic decompensation. Manage medically: NG decompression, octreotide, bowel rest."
                ),
            },
            {
                "term": "HSCT (Haematopoietic Stem Cell Transplant) in MNGIE",
                "definition": (
                    "Allogeneic HSCT is the only proven curative therapy for MNGIE. "
                    "Mechanism: donor haematopoietic cells (neutrophils/monocytes) provide TYMP "
                    "enzyme activity in the bloodstream → plasma thymidine normalised → dNTP pool "
                    "rebalanced → stops ongoing mtDNA damage. "
                    "Best outcomes: early HSCT (<3 years onset), ECOG <3, nutritional optimisation. "
                    "Myeloablative conditioning required (reduced intensity may allow engraftment failure). "
                    "Post-HSCT: GI function improves over 6-24 months; neuropathy stabilises; "
                    "leukoencephalopathy may improve over 2-5 years."
                ),
            },
            {
                "term": "PEO (Progressive External Ophthalmoplegia)",
                "definition": (
                    "Bilateral, usually symmetric, slowly progressive weakness of extraocular muscles "
                    "causing limited ocular motility (upgaze most prominent) ± ptosis. "
                    "In MNGIE: present in ~50-60%; due to mtDNA multiple deletions in extraocular "
                    "muscle (high mtDNA deletion burden in post-mitotic tissues). "
                    "KEY DDx from KSS: KSS has PEO + RETINOPATHY + heart block; MNGIE has PEO "
                    "WITHOUT retinopathy or cardiac involvement. "
                    "Diplopia rare (symmetric bilateral PEO). Management: ptosis props; strabismus "
                    "surgery rarely needed."
                ),
            },
            {
                "term": "mtDNA Multiple Deletions",
                "definition": (
                    "Multiple, heterogeneous deletions of the mitochondrial DNA genome, detected by "
                    "Southern blot (multiple bands on gel) or long-range PCR. "
                    "In MNGIE: primary mtDNA lesion (as opposed to mtDNA DEPLETION in TK2/DGUOK/SUCLA2). "
                    "Detectable from buffy coat DNA (in contrast to many MDDS where depletion is "
                    "tissue-specific). Mechanism: dTTP/dCTP pool imbalance → POLG replication errors. "
                    "Pattern distinguishes MNGIE from KSS (KSS: one large single deletion, ~5 kb, "
                    "heteroplasmic; not inherited as germline) and other deletion-causing conditions."
                ),
            },
            {
                "term": "SIBO (Small Intestinal Bacterial Overgrowth)",
                "definition": (
                    "Excess bacteria in the small intestine (>10^5 CFU/mL of jejunal aspirate, or "
                    "positive hydrogen/methane breath test) due to impaired intestinal motility. "
                    "In MNGIE: secondary to gastroparesis and intestinal pseudo-obstruction → "
                    "bacterial stasis → overgrowth → malabsorption, diarrhoea, steatorrhoea, "
                    "worsening cachexia, B12/folate deficiency. "
                    "Treat with rotating antibiotics: rifaximin, metronidazole, amoxicillin-clavulanate. "
                    "Recurrent SIBO cycles are expected — ongoing prophylactic/rotating treatment."
                ),
            },
            {
                "term": "VPA (Valproic Acid) — Contraindicated in MNGIE",
                "definition": (
                    "Valproic acid is contraindicated in MNGIE (TYMP-deficiency). Mechanisms: "
                    "(1) VPA enters mitochondria as valproyl-CoA → sequesters CoA → impairs "
                    "mitochondrial function in already-compromised mtDNA deletion disease; "
                    "(2) VPA's epoxide metabolites cause hepatotoxicity, worsened in mitochondrial disease; "
                    "(3) VPA may further perturb dNTP pool balance. "
                    "Prefer LEV (renal excretion, no CoA interaction, no hepatic P450). "
                    "For status epilepticus: use LZP IV + LEV IV; phenytoin / fosphenytoin as alternative."
                ),
            },
            {
                "term": "Peritoneal Dialysis in MNGIE",
                "definition": (
                    "Continuous ambulatory or automated peritoneal dialysis removes thymidine and "
                    "deoxyuridine from the blood by diffusion equilibration across the peritoneal "
                    "membrane into dialysate. Effect: 50-70% reduction in plasma thymidine from baseline. "
                    "Limitations: does NOT normalise thymidine to <0.5 µmol/L (incomplete correction). "
                    "Role: bridge therapy to HSCT, allowing partial metabolic correction and "
                    "nutritional improvement while awaiting HSCT workup and donor matching. "
                    "Also used in HSCT-ineligible patients for symptom palliation."
                ),
            },
            {
                "term": "LEV (Levetiracetam) — Preferred AED in MNGIE",
                "definition": (
                    "Levetiracetam is the preferred antiseizure medication when seizures occur in MNGIE. "
                    "Advantages: renal excretion (avoids hepatic P450 and hepatotoxicity risk); "
                    "no CoA sequestration; no ETC complex inhibition; no POLG inhibition; IV formulation. "
                    "Dosing: 10-20 mg/kg/day oral starting dose → titrate to 40-60 mg/kg/day. "
                    "For status: 60 mg/kg IV loading over 15 minutes. VPA always contraindicated. "
                    "Seizures in MNGIE are not primary features — evaluate for metabolic trigger first "
                    "(electrolytes, glucose, acidosis) before initiating chronic AED."
                ),
            },
            {
                "term": "Cachexia in MNGIE",
                "definition": (
                    "Profound, involuntary weight loss from combined GI malabsorption + vomiting + "
                    "anorexia secondary to gastroparesis. BMI can reach 12-14 kg/m² at nadir. "
                    "Clinical clue: cachexia + GI symptoms + neuropathy + PEO in young adult = "
                    "MNGIE until proven otherwise. "
                    "Exam trap: cachexia frequently misattributed to eating disorder (anorexia nervosa) "
                    "or IBD before MNGIE is considered — delay in diagnosis 4-8 years common. "
                    "Management: PEG tube + continuous enteral nutrition ± TPN; anti-emetics; "
                    "nutritional optimisation is prerequisite for HSCT conditioning tolerance."
                ),
            },
        ],
    }
