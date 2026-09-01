#!/usr/bin/env python3
"""GRACILE Syndrome — BCS1L Complex III Assembly Factor Deficiency.

GRACILE = Growth Restriction, Aminoaciduria, Cholestasis, Iron overload,
          Lactic acidosis, Early death
  BCS1L (BCS1 Like DEAD-Box ATPase / Complex III assembly factor)
  GRACILE Syndrome   OMIM #603839
  Björnstad Syndrome OMIM #262000 (same gene, different allele: pili torti + SNHL)
  Complex III Deficiency (Multiple types) OMIM #124000

PATHOPHYSIOLOGY (BCS1L / Complex III assembly / Rieske FeS insertion):
BCS1L is an AAA-ATPase that functions as a chaperone for the insertion of the Rieske
iron-sulfur protein (UQCRFS1/RISP) into the cytochrome bc1 complex (Complex III)
assembly intermediate. Without BCS1L activity:
  • RISP (the Rieske 2Fe-2S cluster protein, the last subunit added to Complex III)
    CANNOT be inserted into the pre-Complex III (MT-CYB core) assembly intermediate
  • Complex III remains in its late assembly intermediate form (CIII*) with LYRM7-bound
    RISP precursor that cannot be released and incorporated
  • Functional Complex III (with bc1 turnover, Q-cycle for proton pumping) is absent
  • Electrons cannot pass Ubiquinol→FeS→cytochrome c1 → oxidation of ubiquinol blocked
  • ENTIRE respiratory chain backs up: Complex I and II-generated NADH and FADH2
    cannot be re-oxidised → ALL oxidative phosphorylation fails
  • Lactic acidosis: pyruvate cannot enter Krebs cycle via PDH; accumulates → lactate
  • Iron-sulfur protein accumulation: BCS1L failure → iron regulation disrupted in
    mitochondrial matrix → paradoxical iron overload despite severe metabolic illness

MOLECULAR: Two discrete clinical phenotypes from BCS1L mutations:
  (A) GRACILE Syndrome — severe LOF alleles (Finnish founder p.Ser78Gly; severe
      truncating or missense alleles eliminating ATPase/chaperone function):
      - Profound neonatal lactic acidosis; hepatic + renal involvement
      - Essentially 100% neonatal lethality within months (Finnish cohort median ~3 months)
  (B) Björnstad Syndrome (BJNB) — milder hypomorphic alleles (p.Arg45Cys):
      - Pili torti (kinky/twisted hair) + sensorineural hearing loss
      - NO lactic acidosis; NO hepatic disease; normal lifespan possible
      - Cochlear outer hair cells dependent on Complex III for OXPHOS → SNHL
  (C) Complex III deficiency (intermediate) — other biallelic missense:
      - Variable; some hepatomuscular; some encephalomyopathic; tissue-dependent

FINNISH FOUNDER ALLELE (p.Ser78Gly / c.232A>G):
  • Located in the N-terminal transmembrane anchor of BCS1L (residues 37–94)
  • Disrupts protein stability in the mitochondrial inner membrane
  • Carrier frequency: ~1/36 in Finland (enriched in Salla region)
  • Birth prevalence GRACILE in Finland: ~1/50,000–1/70,000
  • Accounting for ~80% of GRACILE cases worldwide
  • Homozygous Ser78Gly → severe GRACILE (near-100% neonatal mortality)
  • Compound heterozygous Ser78Gly / other severe allele → same GRACILE phenotype

GRACILE CLINICAL PRESENTATION — ALL FEATURES UNIVERSAL (100% or near):
  1. IUGR (Intrauterine Growth Restriction) — 100%: severe at birth; birth weight
     typically <5th percentile (mean ~2.0 kg at term); head circumference relatively
     spared (brain somewhat protected by lactate as fuel in neonates); polyhydramnios
     in some pregnancies; no specific prenatal marker

  2. Lactic Acidosis — 100%: severe from birth; blood lactate 10–20 mmol/L at
     presentation (severe acidosis pH <7.15); lactate:pyruvate ratio >20 (confirms
     OXPHOS block, not PDHC deficiency where ratio <20); worsens with ANY fasting,
     infection, or physiological stress; CSF lactate also elevated

  3. Aminoaciduria — 100%: GENERALISED (all essential + non-essential amino acids
     in urine); reflects proximal tubular dysfunction (Fanconi syndrome of renal
     tubule); NOT selective (rules out specific enzyme deficiencies); glucosuria
     (glycosuria without hyperglycaemia), phosphaturia, bicarbonuria also present;
     urine amino acid screen mandatory at diagnosis

  4. Cholestasis — 100%: neonatal onset; conjugated (direct) bilirubin elevated;
     GGT elevated (>10× upper normal limit); alkaline phosphatase elevated; pale
     stools in some; bile canalicular plugging on liver biopsy; cholestasis is
     hepatocellular in origin (mitochondrial hepatocyte failure) NOT biliary obstructive

  5. Iron Overload — 100%: PATHOGNOMONIC feature unique among mitochondrial diseases;
     serum ferritin markedly elevated (often >2000 ng/mL despite no transfusions);
     transferrin saturation >90%; hepatic iron quantification (Perls stain) shows
     hepatocellular pattern; mechanism: Complex III failure → impaired mitochondrial
     iron-sulfur cluster biosynthesis and ABCB10-mediated haem synthesis → iron
     redistributes into mitochondria and hepatocytes; secondary hepcidin suppression;
     NOT haemochromatosis (different gene/mechanism); NOT neonatal haemochromatosis (GTD)

  6. Liver Failure — 85%: progressive hepatocellular dysfunction; coagulopathy
     (INR >2); hypoalbuminaemia; hypoglycaemia from gluconeogenesis failure;
     ascites; jaundice; transaminases elevated (ALT/AST 200–1000 IU/L);
     liver biopsy: mitochondrial cristae abnormalities on EM; steatosis; fibrosis

  7. Renal Tubulopathy — 80%: proximal tubular dysfunction (Fanconi-like);
     aminoaciduria + glycosuria + phosphaturia + bicarbonuria; hypophosphataemia;
     renal tubular acidosis (Type II); renal failure uncommon acutely but tubulopathy
     worsens metabolic acidosis

  8. Hypoglycaemia — 70%: DANGEROUS; due to liver failure + OXPHOS blockade of
     gluconeogenesis; blood glucose <2.5 mmol/L; requires CONTINUOUS IV dextrose
     infusion (GIR 8–10 mg/kg/min); NEVER withhold glucose (fasting → lactic
     crisis + hypoglycaemia simultaneously)

  9. Encephalopathy — 60%: cerebral OXPHOS failure; hypotonia; poor suck; seizures;
     cerebral oedema on MRI (T2 signal in basal ganglia in severe cases — Leigh-like
     pattern in some but NOT classic bilateral BG symmetric as in pure Leigh);
     EEG may show burst-suppression or hypsarrhythmia-like; seizures are metabolic
     (hypoglycaemia + lactic acidosis) NOT primarily epileptic

  10. Haemolytic Anaemia — 40%: microangiopathic features; complex III in red cell
      mitochondria (reticulocytes and early RBCs require OXPHOS); Heinz body-like
      haemolysis; combined with iron overload → refractory haemolytic anaemia
      despite high ferritin

  11. Cardiac Involvement — 20%: HCM (hypertrophic cardiomyopathy) in some cases;
      arrhythmias; cardiac OXPHOS failure; less prominent than in other Complex
      deficiency syndromes but present in severe cases

  12. Death — 95% within first year: Finnish cohort median survival 2–4 months;
      cause: multi-organ failure (liver + metabolic); some survive to 6–12 months
      with aggressive supportive care; liver transplant has been attempted in a few
      with mixed results (does not cure brain/renal disease)

ALLELIC DIFFERENTIAL — BJÖRNSTAD SYNDROME (p.Arg45Cys / c.133C>T):
  — Located in the N-terminal TM anchor (residues 37–94) like Ser78Gly but milder impact
  — BCS1L retains partial ATPase/chaperone function → Complex III partially assembled
  — Clinical: pili torti (twisted hair cortex, brittle, breaks easily) + SNHL (high-frequency)
  — NO lactic acidosis; NO hepatic disease; NO IUGR; normal life expectancy possible
  — Cochlear outer hair cells highly dependent on Complex III OXPHOS → SNHL dominant
  — Hair follicle outer root sheath: Complex III-dependent; pili torti = hair shaft defect
  — KEY DDx from GRACILE: completely different prognosis and management

HISTOPATHOLOGY:
  Liver biopsy (if performed before death):
  • Mitochondrial cristae: disordered, abnormal on electron microscopy
  • Hepatocyte iron: grade 2–4 (Perls stain) hepatocellular pattern
  • Steatosis (macro/microvesicular) present
  • Bile canalicular plugging; cholestatic features
  • Fibrosis progressing to cirrhosis
  Muscle biopsy:
  • Ragged red fibres (RRF): variable; may be absent in neonates
  • COX negative fibres: present (Complex IV downstream of Complex III failure)
  • SDH positive: Complex II nuclear-encoded, unaffected
  Biochemistry (muscle homogenate enzyme assay):
  • Complex III activity: severely reduced (<20% of normal)
  • Complex I/IV/V also reduced secondary (downstream cascade from blocked QH2 oxidation)
  • Complex II normal or near-normal (nuclear-encoded, bypasses bc1)

KEY DIFFERENTIAL DIAGNOSIS:
  1. Neonatal haemochromatosis (gestational alloimmune liver disease, GALD): iron overload
     + liver failure + IUGR — but NO lactic acidosis, NO aminoaciduria; treat with IVIg
     + IVIG antepartum; NOT Complex III deficiency
  2. DGUOK (MDDS3): hepatocerebral mtDNA depletion; hepatic + encephalopathy; NO iron
     overload; nystagmus 90%; different amino acid pattern; mtDNA depletion on quantification
  3. MPV17 (MDDS6): hepatocerebral depletion; hepatic + neurological; NO iron overload
     as cardinal feature; mtDNA depletion; Navajo neurohepatopathy (NNH) type
  4. POLG Alpers-Huttenlocher: hepatocellular + VPA CI; EPC seizures cardinal; NO iron
     overload; VPA hepatotoxicity hallmark; mtDNA depletion; different mutation pattern
  5. Tyrosinemia type I (FAH deficiency): liver failure + renal tubulopathy; but
     plasma tyrosine/succinylacetone positive; NTBC treatment; NO lactic acidosis severity
  6. Wilson disease (ATP7B): hepatic + neurological; copper accumulation NOT iron;
     KF rings; ceruloplasmin low; different biochemistry entirely; NOT neonatal
  7. MCAD/VLCAD: fatty acid oxidation; plasma acylcarnitines diagnostic; different
     metabolite profile; NO iron overload; treat with carnitine + avoid fasting

ABSOLUTE DRUG CONTRAINDICATIONS:
  VPA (valproate) — ABSOLUTE CI ALL BCS1L/GRACILE:
    (a) CoA sequestration → worsens already-failed OXPHOS in every tissue
    (b) POLG1 inhibition → accelerates mitochondrial dysfunction
    (c) Hepatotoxicity ADDITIVE with existing liver failure (often fatal)
    Alternative for seizures: LEV (renal excretion, NO mito toxicity); phenobarbital
    (CAUTION — CI in Complex I deficiency but tolerated better than VPA in GRACILE);
    benzodiazepines for acute seizure control

  Metformin — ABSOLUTE CI ALL BCS1L/GRACILE:
    Complex I inhibition → pyruvate/lactate cannot clear via electron transport;
    exacerbates lactic acidosis that is already life-threatening (lactate 10–20 mmol/L);
    NEVER use for hypoglycaemia; use IV dextrose/glucagon only

  Iron Supplementation — ABSOLUTE CI — ALL GRACILE:
    Patients already have severe iron overload (ferritin >2000, transferrin saturation >90%);
    ANY additional iron → hepatocellular iron toxicity (Fenton reaction → hydroxyl radical
    → lipid peroxidation); worsens liver failure; haemolysis; NEVER supplement iron;
    monitor ferritin; chelation considered only if Fe critically high and patient stable

  Linezolid — ABSOLUTE CI — ALL mitochondrial disease:
    Mitochondrial 23S rRNA inhibitor → blocks mtDNA-encoded Complex III/IV synthesis;
    in GRACILE (already zero Complex III) → completely blocks any residual OXPHOS;
    lactic acidosis + DION + pancytopenia; alternative: daptomycin / β-lactams

  Ketogenic Diet — ABSOLUTE CI — ALL BCS1L/GRACILE:
    Beta-oxidation of fatty acids requires Complex III for FADH2 re-oxidation;
    without Complex III → FAO-derived electrons cannot pass → FAO fails completely;
    KD would cause metabolic collapse; NO therapeutic role whatsoever

  Propofol (any dose) — ABSOLUTE CI in neonates:
    Propofol infusion syndrome (PRIS) in Complex III deficiency: propofol impairs Complex
    IV via uncoupling; in GRACILE with pre-existing OXPHOS failure → cardiac arrhythmia,
    metabolic acidosis, renal failure; use inhalational agents (sevoflurane) for anaesthesia

  Phenobarbital — HIGH CAUTION (not absolute CI but HIGH RISK):
    Complex I inhibitor → worsens lactic acidosis; USE ONLY if LEV unavailable and
    seizures uncontrolled; monitor lactate closely; NOT first-line

TREATMENT (Supportive — No Disease-Modifying Therapy):
  1. CONTINUOUS IV Dextrose (GIR 8–10 mg/kg/min) — MANDATORY, Level A:
     NEVER fast; even brief fasting precipitates hypoglycaemic + lactic crisis;
     maintain blood glucose 4–7 mmol/L; may require 20% dextrose centrally
  2. Sodium Bicarbonate (NaHCO3) — for acute lactic acidosis correction:
     Target pH >7.2; partial correction; CAUTIOUS volume management (liver oedema);
     bicarbonate deficit calculation: 0.3 × weight(kg) × base deficit; slow IV infusion
  3. CoQ10 / Ubiquinol — Level C — 300–600 mg/day (oral/NG):
     Short-chain quinone analogues may partially bypass Complex III block via
     electron transfer to cytochrome c; modest at best; ubiquinol preferred bioavailability
  4. Riboflavin B2 — Level C — 50–300 mg/day:
     FAD/FMN cofactor for Complex II and Complex III electron transfer proteins;
     may partially support residual Complex III function
  5. UDCA (Ursodeoxycholic acid) — Level C — 15–20 mg/kg/day:
     Improves cholestasis; reduces bile acid toxicity to hepatocytes; standard cholestasis
     management; does NOT treat underlying Complex III deficiency
  6. L-Carnitine — Level C — 50–100 mg/kg/day:
     Secondary carnitine deficiency common in liver failure + OXPHOS blockade;
     supports FAO partial function; may improve acidosis marginally
  7. Vitamin K (phytomenadione) — for coagulopathy from hepatic failure:
     INR target <2; fresh frozen plasma if acute bleeding; avoid IM if INR >2
  8. Parenteral Nutrition (TPN) — if oral/NG feeds not tolerated:
     Glucose-predominant (NOT high fat — FAO impaired); protein restricted if
     hyperammonaemia develops; careful electrolyte management (phosphate, potassium)
  9. LEV (levetiracetam) — preferred AED for seizures — Level B:
     Renal excretion; no hepatic metabolism; no Complex I inhibition; safe in liver failure;
     IV formulation available for neonatal use; load 20 mg/kg then maintenance 20–40 mg/kg/day
  10. Liver Transplantation — controversial / not standard of care:
      Corrects hepatic disease ONLY; does NOT cure renal/brain/cardiac OXPHOS failure;
      few case reports; most patients die before transplant; NOT recommended unless
      patient has isolated hepatic phenotype (very rare BCS1L variant); discuss with
      metabolic + transplant team only

GENETIC COUNSELLING:
  • AR inheritance: 25% recurrence risk per pregnancy
  • Finnish founder p.Ser78Gly: carrier screening available; 1/36 carrier rate in Finland
  • Prenatal testing: chorionic villus sampling (CVS) for known familial mutation;
    DNA testing for pathogenic BCS1L variant if parents are known carriers
  • Finnish carrier screening programs are available; genetic counselling essential
  • No genotype-phenotype correlation within GRACILE phenotype (all severe alleles → GRACILE)
  • BJNB allele (p.Arg45Cys) → different syndrome; prognostically very different

KEY REFERENCES:
  Fellman V, et al. 1998. A GRACILE Syndrome, a Defined Lethal Disease with Iron Overload,
  Lactic Acidosis, and Liver Disease. Ann Med 30:260–266. FIRST GRACILE DESCRIPTION.
  Visapää I, et al. 2002. GRACILE Syndrome, a Lethal Metabolic Disorder with Iron
  Overload, Is Caused by a Point Mutation in BCS1L. Am J Hum Genet 71:863–876.
  BCS1L mutations IDENTIFIED as cause of GRACILE.
  de Lonlay P, et al. 2001. A Mutant Mitochondrial Respiratory Chain Assembly Protein
  Causes Complex III Deficiency in Patients with Tubulopathy, Encephalopathy and Liver
  Failure. Nat Genet 29:57–60. BCS1L Complex III mechanism.
  Barel O, et al. 2008. Maternally Inherited Björnstad Syndrome Caused by a Mutation in
  BCS1L. Brain 131:2032–2040. BJNB vs GRACILE BCS1L phenotypic spectrum.
  Hinson JT, et al. 2007. Missense Mutations in the BCS1L Gene as a Cause of the
  GRACILE Syndrome. J Clin Invest 117:2149–2158. Crystal structure context.
"""

from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED         = 591
DISEASE_ID   = "gracile"
DISEASE_NAME = "GRACILE Syndrome"
GENE         = "BCS1L"
OMIM_GENE    = "#603358"
OMIM_DISEASE = "#603839"
CHROMOSOME   = "2q35"
INHERITANCE  = "AR (autosomal recessive biallelic)"
ONSET        = "Neonatal / in utero (IUGR prenatal; lactic acidosis at birth)"
COHORT_SIZE  = 40
COLOR        = "#4e342e"   # deep brown — iron overload / hepatic / metabolic crisis
LIGHT        = "#efebe9"

# Genotype pool
GENO_FIN_HOM = "p.Ser78Gly (c.232A>G) homozygous — Finnish founder"
GENO_FIN_CPX = "p.Ser78Gly / other severe allele — compound heterozygous"
GENO_OTHER   = "Other biallelic BCS1L severe allele (non-Finnish)"

GENO_POOL    = [GENO_FIN_HOM, GENO_FIN_CPX, GENO_OTHER]
GENO_WEIGHTS = [0.65,          0.15,          0.20]


# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient GRACILE cohort (seed-591)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient GRACILE cohort (seed-591).

    All patients meet GRACILE definition: IUGR + lactic acidosis +
    aminoaciduria + cholestasis + iron overload (all 100%).
    Additional features stochastic per published frequencies.
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        geno    = rng.choices(GENO_POOL, weights=GENO_WEIGHTS)[0]
        sex     = "F" if rng.random() < 0.50 else "M"
        # Birth weight (kg) — IUGR: mean 2.0 kg, SD 0.3, min 1.4 max 2.8
        bwt     = round(min(2.8, max(1.4, rng.gauss(2.0, 0.3))), 2)
        # Lactate at presentation (mmol/L)
        lactate = round(rng.uniform(10.0, 22.0), 1)
        # pH
        pH      = round(rng.uniform(7.00, 7.20), 2)
        # Ferritin (ng/mL)
        ferritin = round(rng.uniform(800, 5000))

        # Features (all 100% first 5; additional stochastic)
        liver_fail  = rng.random() < 0.85
        renal_tub   = rng.random() < 0.80
        hypoglycemia= rng.random() < 0.70
        encephalo   = rng.random() < 0.60
        hemolysis   = rng.random() < 0.40
        cardiac     = rng.random() < 0.20
        seizures    = rng.random() < 0.25
        # Survival (95% die <12 months in full GRACILE)
        died_yr1    = rng.random() < 0.90  # 90% in cohort for modelling purposes

        # Age at death (months) if died; or age at last follow-up
        if died_yr1:
            age_outcome = round(rng.uniform(1.0, 10.0), 1)
            outcome = f"Died at {age_outcome}mo"
        else:
            age_outcome = round(rng.uniform(8.0, 24.0), 1)
            outcome = f"Alive {age_outcome}mo (LT or partial phenotype)"

        # Treatments
        txs = ["IV Dextrose GIR 8-10", "NaHCO3 (lactic acidosis)", "UDCA (cholestasis)"]
        if not died_yr1 or rng.random() < 0.7: txs.append("CoQ10/Ubiquinol")
        if not died_yr1 or rng.random() < 0.6: txs.append("Riboflavin B2")
        txs.append("Carnitine L")
        if seizures: txs.append("LEV (preferred AED)")
        if liver_fail: txs.append("Vitamin K (coagulopathy)")

        # Alerts
        alerts_list = []
        alerts_list.append("🚫 VPA ABSOLUTE CI — liver failure risk lethal")
        alerts_list.append("🚫 Fe supplements CI — iron overload")
        if seizures: alerts_list.append("⚠ Seizures: LEV only — NOT VPA/phenobarbital")
        if hypoglycemia: alerts_list.append("🚨 Hypoglycaemia: continuous dextrose mandatory")
        alerts = "; ".join(alerts_list)

        # Feature string
        feats = ["IUGR", "Lactic acidosis", "Aminoaciduria", "Cholestasis", "Iron overload"]
        if liver_fail:   feats.append("Liver failure")
        if renal_tub:    feats.append("Renal tubulopathy")
        if hypoglycemia: feats.append("Hypoglycaemia")
        if encephalo:    feats.append("Encephalopathy")
        if hemolysis:    feats.append("Haemolysis")
        if cardiac:      feats.append("Cardiomyopathy")
        if seizures:     feats.append("Seizures")

        patients.append({
            "id":         f"GRACILE-{i:03d}",
            "geno":       geno,
            "sex":        sex,
            "bwt":        bwt,
            "lactate":    lactate,
            "pH":         pH,
            "ferritin":   ferritin,
            "features":   ", ".join(feats),
            "treatments": ", ".join(txs),
            "alerts":     alerts,
            "outcome":    outcome,
        })
    return patients


# ── Public API functions ─────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    """GRACILE overview — gene, disease identity, KPIs, contraindications."""
    rng = _rng()
    cohort = _build_cohort(rng)

    n = len(cohort)
    n_liver   = sum(1 for p in cohort if "Liver failure" in p["features"])
    n_renal   = sum(1 for p in cohort if "Renal" in p["features"])
    n_hypogly = sum(1 for p in cohort if "Hypoglycaemia" in p["features"])
    n_enceph  = sum(1 for p in cohort if "Encephalopathy" in p["features"])
    n_hemol   = sum(1 for p in cohort if "Haemolysis" in p["features"])
    n_cardiac = sum(1 for p in cohort if "Cardiomyopathy" in p["features"])
    n_seiz    = sum(1 for p in cohort if "Seizures" in p["features"])
    n_died    = sum(1 for p in cohort if "Died" in p["outcome"])
    mean_lact = round(sum(p["lactate"] for p in cohort) / n, 1)
    mean_ferr = round(sum(p["ferritin"] for p in cohort) / n)
    mean_bwt  = round(sum(p["bwt"] for p in cohort) / n, 2)
    n_finn_hom = sum(1 for p in cohort if "homozygous" in p["geno"])

    return {
        "gene":         "BCS1L (BCS1 Like DEAD-Box ATPase / Complex III assembly factor)",
        "protein":      "BCS1L-478aa — AAA-ATPase chaperone for Rieske FeS protein (UQCRFS1/RISP) insertion into Complex III (cytochrome bc1)",
        "disease":      "GRACILE Syndrome (Growth Restriction, Aminoaciduria, Cholestasis, Iron overload, Lactic acidosis, Early death)",
        "omim_gene":    "#603358 (BCS1L)",
        "omim_disease": "#603839 (GRACILE) · Allelic: #262000 (Björnstad / BJNB — pili torti + SNHL)",
        "chromosome":   "2q35",
        "inheritance":  "AR (autosomal recessive biallelic) — 25% recurrence per pregnancy",
        "onset":        "Neonatal / prenatal IUGR; lactic acidosis manifest at birth",
        "cohort":       f"{n} patients · seed-591 · GRACILE BCS1L biallelic",
        "mechanism": (
            "BCS1L is an AAA-ATPase that inserts the Rieske iron-sulfur protein (UQCRFS1/RISP) "
            "into the pre-Complex III assembly intermediate (CIII*, the MT-CYB-containing "
            "pre-dimer). Without BCS1L chaperone activity, RISP (the last subunit added, "
            "containing the catalytic 2Fe-2S cluster) cannot be released from LYRM7 and "
            "incorporated into Complex III. The result is ZERO functional cytochrome bc1 complex: "
            "ubiquinol oxidation is blocked → ALL Complex I/II-derived electrons accumulate as "
            "QH2 → entire respiratory chain backs up → ATP synthesis fails → pyruvate/lactate "
            "accumulate → severe multi-organ lactic acidosis. Iron-sulfur cluster dysregulation "
            "from RISP accumulation causes paradoxical hepatic iron overload despite metabolic crisis."
        ),
        "bjnb_contrast": (
            "Björnstad Syndrome (BJNB) — same BCS1L gene, DIFFERENT allele (p.Arg45Cys / c.133C>T): "
            "BCS1L partially functional → cochlear outer hair cells and hair follicle outer root "
            "sheath (both Complex III-dependent) fail selectively. Phenotype: pili torti (twisted "
            "hair shaft, brittle, breaks) + SNHL (high-frequency bilateral). "
            "NO lactic acidosis. NO hepatic disease. NO IUGR. Normal lifespan possible. "
            "CRITICAL DDx: GRACILE (lethal within months) vs BJNB (compatible with life). "
            "Genotype determines outcome absolutely — molecular testing mandatory."
        ),
        "kpis": [
            {"label": "IUGR (100%)", "value": "100%", "color": COLOR},
            {"label": "Lactic Acidosis", "value": "100%", "color": "#c62828"},
            {"label": "Aminoaciduria", "value": "100%", "color": COLOR},
            {"label": "Cholestasis", "value": "100%", "color": COLOR},
            {"label": "Iron Overload", "value": "100%", "color": "#bf360c"},
            {"label": "Liver Failure", "value": f"{n_liver/n*100:.0f}%", "color": "#c62828"},
            {"label": "Renal Tubulop.", "value": f"{n_renal/n*100:.0f}%", "color": COLOR},
            {"label": "Hypoglycaemia", "value": f"{n_hypogly/n*100:.0f}%", "color": "#e65100"},
            {"label": "Encephalopathy", "value": f"{n_enceph/n*100:.0f}%", "color": "#6a1b9a"},
            {"label": "Haemolysis", "value": f"{n_hemol/n*100:.0f}%", "color": COLOR},
            {"label": "Mean Lactate", "value": f"{mean_lact} mmol/L", "color": "#c62828"},
            {"label": "Mean Ferritin", "value": f"{mean_ferr} ng/mL", "color": "#bf360c"},
            {"label": "Mean BWT", "value": f"{mean_bwt} kg", "color": COLOR},
            {"label": "Mortality <1yr", "value": f"{n_died/n*100:.0f}%", "color": "#b71c1c"},
            {"label": "Finnish Hom.", "value": f"{n_finn_hom/n*100:.0f}%", "color": COLOR},
            {"label": "Seizures", "value": f"{n_seiz/n*100:.0f}%", "color": "#6a1b9a"},
        ],
        "contraindications": [
            {
                "drug":      "VPA / Valproate",
                "severity":  "ABSOLUTE CI — ALL BCS1L/GRACILE",
                "mechanism": (
                    "Triple mechanism: (a) CoA sequestration → worsens already-zero OXPHOS in every "
                    "organ; (b) POLG1 inhibition → accelerates mitochondrial genome damage; "
                    "(c) Hepatotoxicity ADDITIVE with existing hepatic failure — can precipitate "
                    "acute liver failure and death. Use LEV (IV, renal excretion, no mito toxicity) "
                    "for seizures; benzodiazepine for acute control."
                ),
            },
            {
                "drug":      "Iron supplementation (any form)",
                "severity":  "ABSOLUTE CI — ALL GRACILE",
                "mechanism": (
                    "Patients have SEVERE iron overload (ferritin >2000 ng/mL, transferrin "
                    "saturation >90%) from Complex III failure → iron dysregulation. Additional iron "
                    "drives Fenton reaction → hydroxyl radical → lipid peroxidation → accelerates "
                    "hepatocellular necrosis, haemolysis, and multi-organ failure. "
                    "NEVER supplement iron. Monitor ferritin; chelation if ferritin >5000 and patient stable."
                ),
            },
            {
                "drug":      "Metformin",
                "severity":  "ABSOLUTE CI — ALL BCS1L/GRACILE",
                "mechanism": (
                    "Complex I inhibitor → exacerbates lactic acidosis already at 10–20 mmol/L. "
                    "In GRACILE, lactic acidosis is the immediate cause of metabolic death; "
                    "ANY additional Complex I inhibition → fatal lactic crisis. "
                    "Use IV dextrose + glucagon for hypoglycaemia; NEVER metformin."
                ),
            },
            {
                "drug":      "Linezolid",
                "severity":  "ABSOLUTE CI — ALL mitochondrial disease",
                "mechanism": (
                    "Mitochondrial 23S rRNA inhibitor → blocks mtDNA-encoded Complex III/IV "
                    "protein synthesis. In GRACILE (already zero Complex III assembly) → "
                    "eliminates any residual Complex III/IV activity → accelerates "
                    "pan-OXPHOS collapse. Alternative: daptomycin, beta-lactams, carbapenems."
                ),
            },
            {
                "drug":      "Ketogenic Diet",
                "severity":  "ABSOLUTE CI — ALL BCS1L/GRACILE",
                "mechanism": (
                    "Beta-oxidation of fatty acids generates FADH2 (via ACAD/ETF/ETFDH chain) "
                    "that requires Complex III (bc1) for re-oxidation. Without Complex III, "
                    "FADH2 cannot be oxidised → FAO completely blocked → metabolic collapse + "
                    "lactic acidosis + hypoglycaemia. No therapeutic role."
                ),
            },
            {
                "drug":      "Propofol (neonatal anaesthesia)",
                "severity":  "ABSOLUTE CI — neonates with Complex III deficiency",
                "mechanism": (
                    "Propofol infusion syndrome (PRIS): propofol impairs Complex IV via "
                    "uncoupling + direct mitochondrial membrane disruption. In GRACILE with "
                    "pre-existing OXPHOS failure → cardiac arrhythmia, severe lactic acidosis, "
                    "rhabdomyolysis, acute renal failure. Use sevoflurane inhalational anaesthesia "
                    "if anaesthesia required; AVOID propofol even for brief induction."
                ),
            },
            {
                "drug":      "Phenobarbital",
                "severity":  "HIGH CAUTION — not first-line (Complex I inhibitor)",
                "mechanism": (
                    "Complex I inhibitor → exacerbates lactic acidosis; but less dangerous than "
                    "VPA in GRACILE. USE ONLY if LEV unavailable and seizures uncontrolled; "
                    "monitor lactate closely with use. NOT preferred — LEV always first."
                ),
            },
        ],
    }


def get_breakdown() -> dict[str, Any]:
    """GRACILE patient cohort table + clinical feature frequencies."""
    rng = _rng()
    cohort = _build_cohort(rng)

    n = len(cohort)

    def pct(feat: str) -> int:
        return round(sum(1 for p in cohort if feat in p["features"]) / n * 100)

    feature_frequencies = {
        "IUGR (birth weight <5th percentile)": 100,
        "Lactic Acidosis (lactate ≥10 mmol/L)": 100,
        "Aminoaciduria (generalised, Fanconi-like)": 100,
        "Cholestasis (conjugated bilirubin elevated)": 100,
        "Iron Overload (ferritin >2000 ng/mL)": 100,
        "Liver Failure (coagulopathy/ascites)": pct("Liver failure"),
        "Renal Tubulopathy (Fanconi-like)": pct("Renal"),
        "Hypoglycaemia (<2.5 mmol/L)": pct("Hypoglycaemia"),
        "Encephalopathy (hypotonia/Leigh-like MRI)": pct("Encephalopathy"),
        "Haemolytic Anaemia": pct("Haemolysis"),
        "Cardiomyopathy (HCM)": pct("Cardiomyopathy"),
        "Seizures (metabolic/epileptic)": pct("Seizures"),
        "Death within first year": round(sum(1 for p in cohort if "Died" in p["outcome"]) / n * 100),
    }

    return {
        "patients": cohort,
        "feature_frequencies": feature_frequencies,
    }


def get_definitions() -> dict[str, Any]:
    """GRACILE clinical, pharmacological and molecular definitions."""
    return {
        "pharmacology": [
            {
                "term": "BCS1L (BCS1 Like DEAD-Box ATPase)",
                "definition": (
                    "Nuclear-encoded mitochondrial AAA-ATPase (478 amino acids; 2q35). "
                    "Imported into the mitochondrial inner membrane (IMM) via N-terminal "
                    "matrix-targeting sequence (MTS). Forms a homo-oligomeric ring (likely "
                    "hexameric) in the IMM. Unique function: chaperone for the insertion of "
                    "the Rieske iron-sulfur protein (UQCRFS1/RISP — containing the critical "
                    "2Fe-2S cluster) from the pre-Complex III assembly intermediate (CIII*) "
                    "into the mature Complex III (cytochrome bc1 dimer). ATP hydrolysis by "
                    "BCS1L drives conformational change that releases RISP from its chaperone "
                    "LYRM7 (LYR domain-containing protein 7) and inserts it into CIII*. "
                    "Without BCS1L: Complex III assembly arrests at the CIII* stage; RISP "
                    "accumulates in mitochondria as LYRM7-bound precursor; no mature bc1 "
                    "complex is formed; Complex III activity = 0."
                ),
            },
            {
                "term": "Complex III (Cytochrome bc1 Complex / Ubiquinol:Cytochrome c Oxidoreductase)",
                "definition": (
                    "The third complex of the mitochondrial respiratory chain. Catalyses the "
                    "oxidation of ubiquinol (QH2) and the reduction of cytochrome c (Cyt c): "
                    "  QH2 + 2 Cyt c (ox) → Q + 2 Cyt c (red) + 2 H+ (pumped into IMS)\n"
                    "The Q-cycle: Q-cycle mechanism pumps 4 protons per 2 electrons, "
                    "contributing to the proton motive force for ATP synthesis. "
                    "Key subunits: MT-CYB (mtDNA-encoded, core), CYC1 (cytochrome c1), "
                    "UQCRC1/2 (Core1/Core2), UQCRFS1/RISP (Rieske 2Fe-2S), UQCRB, "
                    "UQCRQ, UQCR10, UQCR11 (all nuclear-encoded). "
                    "Assembly: complex hierarchical process requiring BCS1L for final RISP "
                    "insertion step. Complete absence (as in GRACILE) → zero bc1 activity."
                ),
            },
            {
                "term": "GRACILE Acronym — Clinical Diagnostic Criteria",
                "definition": (
                    "G — Growth Restriction: IUGR (prenatal) + postnatal growth failure; "
                    "birth weight typically <5th centile; birth length also affected\n"
                    "R — Aminoaciduria (Renal aminoaciduria): generalised aminoaciduria "
                    "reflecting proximal tubular Fanconi syndrome; ALL amino acids elevated "
                    "in urine (not selective — rules out specific enzyme deficiencies)\n"
                    "A — Cholestasis (Anicteric or icteric): neonatal onset; conjugated "
                    "hyperbilirubinaemia; GGT and ALP markedly elevated; hepatocellular origin\n"
                    "C — Iron Overload: serum ferritin markedly elevated (>2000 ng/mL); "
                    "hepatic iron deposition (Perls stain positive); transferrin saturation >90%\n"
                    "I — Lactic Acidosis: blood lactate 10–20 mmol/L at birth; pH <7.15; "
                    "L:P ratio >20 (OXPHOS block confirmed)\n"
                    "L — Early death (Lethal in infancy): median survival 2–4 months in Finnish "
                    "cohort; nearly 100% lethality within first year without heroic intervention\n"
                    "E — (E included for acronym; Early death)\n"
                    "ALL SIX criteria must be present for definitive GRACILE diagnosis."
                ),
            },
            {
                "term": "Iron Overload Mechanism in GRACILE — Paradoxical Siderosis",
                "definition": (
                    "GRACILE iron overload is mechanistically distinct from haemochromatosis "
                    "(HFE/TfR2/HJV mutations) or neonatal haemochromatosis (GALD). "
                    "Pathophysiology: BCS1L failure → RISP (iron-sulfur cluster protein) cannot "
                    "be incorporated into Complex III → mitochondrial iron-sulfur cluster "
                    "synthesis pathway (ISCA1/ISCA2/GLRX5 scaffold) is dysregulated → excess "
                    "mitochondrial iron cannot be exported or utilised → accumulates in "
                    "hepatocytes (post-mitotic cells with high mitochondrial density) → "
                    "secondary hepcidin suppression (liver failure impairs hepcidin synthesis) "
                    "→ increased intestinal iron absorption further adds to overload. "
                    "Clinical: high ferritin, high transferrin saturation, hepatocellular "
                    "iron pattern on Perls stain, normal serum ceruloplasmin. "
                    "NOT a primary haemochromatosis — it is SECONDARY to Complex III failure. "
                    "Management: NEVER supplement iron; chelation only if critically elevated "
                    "and patient stable (most die before chelation is feasible)."
                ),
            },
            {
                "term": "VPA Absolute Contraindication — Triple Mechanism",
                "definition": (
                    "Valproate is ABSOLUTELY CONTRAINDICATED in ALL BCS1L/GRACILE patients "
                    "via three independent mechanisms:\n"
                    "1. CoA Sequestration: valproate → valproyl-CoA → depletes free CoA pool → "
                    "impairs TCA cycle (succinyl-CoA, acetyl-CoA steps) already stressed by "
                    "OXPHOS failure; worsens energy crisis in every cell\n"
                    "2. POLG1 Inhibition: valproate (via valproic acid-CoA) directly inhibits "
                    "POLG1 (mtDNA polymerase gamma) → slows mtDNA replication → reduces "
                    "mtDNA copy number → accelerates mitochondrial dysfunction\n"
                    "3. Hepatotoxicity ADDITIVE: VPA causes dose-dependent and idiosyncratic "
                    "hepatotoxicity; in GRACILE with established hepatic failure, VPA "
                    "hepatotoxicity is amplified and can precipitate acute fulminant liver "
                    "failure and death within days of initiation\n"
                    "Alternative: LEV (levetiracetam) — IV formulation, renal excretion, no "
                    "hepatic metabolism, no mito toxicity, safe in liver failure."
                ),
            },
        ],
        "gene_concepts": [
            {
                "term": "GRACILE vs Björnstad Syndrome — Allele-Phenotype Correlation",
                "definition": (
                    "Same gene (BCS1L, 2q35) causes two completely different syndromes "
                    "depending on the allele severity:\n\n"
                    "GRACILE (p.Ser78Gly or other severe alleles):\n"
                    "• Located in N-terminal TM anchor (aa 37–94) — disrupts IMM insertion\n"
                    "• Complete loss of BCS1L function → zero Complex III assembly\n"
                    "• Multi-organ failure: liver + kidney + IUGR + lactic acidosis + Fe overload\n"
                    "• Near-100% mortality within 12 months\n\n"
                    "Björnstad Syndrome (BJNB, p.Arg45Cys or other hypomorphic alleles):\n"
                    "• Located in N-terminal TM anchor but MILDER impact\n"
                    "• Partial BCS1L function → incomplete Complex III assembly\n"
                    "• Only tissues with HIGHEST Complex III dependency affected:\n"
                    "  — Cochlear outer hair cells → SNHL\n"
                    "  — Hair follicle outer root sheath → pili torti (twisted/kinky hair)\n"
                    "• NO lactic acidosis. NO hepatic disease. NO IUGR. Normal lifespan.\n\n"
                    "Molecular testing MANDATORY to distinguish: prognosis is the extreme "
                    "opposite between these two allele classes."
                ),
            },
            {
                "term": "Finnish Founder Effect — p.Ser78Gly (c.232A>G)",
                "definition": (
                    "The p.Ser78Gly BCS1L variant originated from a single ancestral founder "
                    "in the Finnish population, likely during the population bottleneck of "
                    "the Finnish settlement period (1500s–1700s). Key epidemiology:\n"
                    "• Carrier frequency in Finland: ~1/36 (2.8%) — one of the highest "
                    "carrier rates for any lethal metabolic disease\n"
                    "• Particularly enriched in the Salla region of Northern Finland\n"
                    "• Carrier frequency in other European populations: ~1/200–400\n"
                    "• Birth prevalence GRACILE in Finland: ~1/50,000–70,000 live births\n"
                    "• Homozygous Ser78Gly = classic GRACILE (lethal in months)\n"
                    "• Compound heterozygous Ser78Gly + other severe allele = GRACILE phenotype\n"
                    "• Finnish newborn screening does NOT yet include BCS1L; prenatal "
                    "diagnosis available for known carrier couples via CVS/amniocentesis\n"
                    "• All GRACILE infants in Finnish register died within 12 months (1980s–2000s data)"
                ),
            },
        ],
        "disease_concepts": [
            {
                "term": "Lactic Acidosis — Severity and Lactate:Pyruvate Ratio",
                "definition": (
                    "GRACILE lactic acidosis is among the most severe of all metabolic diseases:\n"
                    "• Blood lactate: typically 10–22 mmol/L at birth (normal <2.0 mmol/L)\n"
                    "• pH: 7.00–7.20 (life-threatening metabolic acidosis)\n"
                    "• Base excess: -15 to -25 mEq/L\n"
                    "• Lactate:Pyruvate (L:P) ratio: >20 (normal <10)\n"
                    "  — L:P ratio >20 indicates OXPHOS block (electron chain failure)\n"
                    "  — L:P ratio <20 with high lactate indicates PDHC deficiency\n"
                    "  — In GRACILE, L:P >20 confirms Complex III block\n"
                    "• CSF lactate also elevated (brain OXPHOS failure)\n"
                    "• Worsened by: fasting (even 2–4 hours), any intercurrent illness, "
                    "fever, physiological stress, surgery, anaesthesia\n"
                    "• Management: continuous IV dextrose (GIR 8–10 mg/kg/min) "
                    "+ NaHCO3 for acute correction (target pH >7.2) + treat triggers"
                ),
            },
            {
                "term": "Neonatal Haemochromatosis — Critical DDx from GRACILE",
                "definition": (
                    "Neonatal Haemochromatosis (NH; gestational alloimmune liver disease, GALD) "
                    "is the most important DDx when GRACILE presents with iron overload + liver failure:\n\n"
                    "NEONATAL HAEMOCHROMATOSIS (GALD):\n"
                    "• Mechanism: maternal alloimmune IgG → neonatal liver injury → haemochromatosis\n"
                    "• Iron overload: YES — hepatic + extrahepatic (salivary glands, pancreas)\n"
                    "• Lactic acidosis: NO (or mild secondary)\n"
                    "• Aminoaciduria: NO\n"
                    "• IUGR: variable\n"
                    "• Treatment: IV immunoglobulin (IVIg 1 g/kg) + N-acetylcysteine; "
                    "double-volume exchange transfusion; liver transplant curative in survivors\n"
                    "• Recurrence in subsequent pregnancies: 90% — IVIG antepartum from 14 weeks\n\n"
                    "GRACILE (BCS1L):\n"
                    "• Lactic acidosis SEVERE (10–22 mmol/L) — THIS IS THE KEY DISTINGUISHER\n"
                    "• Aminoaciduria generalised\n"
                    "• No maternal IgG involvement\n"
                    "• No IVIg response\n"
                    "• AR genetics — 25% recurrence\n"
                    "→ Plasma/urine amino acids + blood lactate + L:P ratio "
                    "diagnose GRACILE definitively; BCS1L sequencing confirms."
                ),
            },
        ],
        "prescribing_safety": [
            {
                "term": "Continuous IV Dextrose — 'Never Fast' Protocol",
                "definition": (
                    "In GRACILE (and all Complex III / OXPHOS diseases), fasting precipitates "
                    "simultaneous lactic crisis + hypoglycaemia — both immediately life-threatening.\n\n"
                    "PROTOCOL (mandatory in all hospitalised GRACILE patients):\n"
                    "• Glucose Infusion Rate (GIR) target: 8–10 mg/kg/min continuously\n"
                    "• Concentration: 10% dextrose (peripheral) or 20% dextrose (central) "
                    "to achieve GIR without volume overload\n"
                    "• Blood glucose target: 4.0–7.0 mmol/L (avoid hypoglycaemia AND "
                    "hyperglycaemia — both worsen metabolic state)\n"
                    "• PRE-PROCEDURE: if any procedure requires NPO (e.g., imaging under "
                    "sedation), IV dextrose MUST be running before and throughout\n"
                    "• EMERGENCY: if IV access lost → place IO (intraosseous) immediately "
                    "and restart dextrose; do not wait for new IV access\n"
                    "• Monitoring: blood glucose every 2–4 hours; lactate every 4–6 hours\n"
                    "• Discharge: families instructed on sick-day rule — if vomiting/unable "
                    "to feed → immediate hospital admission for IV dextrose"
                ),
            },
            {
                "term": "Levetiracetam (LEV) — Preferred AED in Liver Failure / GRACILE",
                "definition": (
                    "LEV (levetiracetam) is the ONLY first-line AED safe in GRACILE:\n"
                    "• Renal excretion (66% unchanged in urine) → no hepatic metabolism\n"
                    "• No Complex I/II/III inhibition\n"
                    "• Available as IV solution (Keppra IV) for neonatal use\n"
                    "• Loading dose (neonatal): 20 mg/kg IV over 15 minutes\n"
                    "• Maintenance: 10–20 mg/kg/dose IV/oral every 12 hours\n"
                    "• Protein binding: <10% → not displaced by acidosis or hypoalbuminaemia\n"
                    "• NOT metabolised by CYP enzymes → no drug interactions in critically ill\n"
                    "• Compared with alternatives:\n"
                    "  — VPA: ABSOLUTE CI (see above — lethal)\n"
                    "  — Phenobarbital: Complex I inhibitor; HIGH CAUTION (not first-line)\n"
                    "  — Phenytoin/fosphenytoin: CYP-metabolised; caution in liver failure; "
                    "not CI in GRACILE but not preferred\n"
                    "  — Benzodiazepine (midazolam/lorazepam): safe for acute seizure control; "
                    "not for maintenance (GABA tolerance, accumulation)"
                ),
            },
        ],
    }
