#!/usr/bin/env python3
"""PEX7 / Rhizomelic Chondrodysplasia Punctata Type 1 (RCDP1) Epilepsy Dashboard — seed data module.

PEX7 encodes Peroxin-7, the cytosolic PTS2 (Peroxisomal Targeting Signal 2) receptor (323 aa,
WD40 repeat protein with 7 beta-propeller WD repeats). PEX7 recognises the PTS2 tripeptide
signal (-R/K-L/V/Q-X5-H/Q-L- at the N-terminus) on PTS2-cargo proteins and shuttles them into
the peroxisomal matrix via the PEXOR import channel.

CRITICAL DISTINCTION FROM PEX1/PEX6 ZSD:
  PEX1/PEX6 deficiency → ALL peroxisomal matrix protein import fails (PTS1 + PTS2)
  PEX7 deficiency → ONLY PTS2-targeted proteins fail to import (PTS1 pathway intact)
  Therefore: VLCFA beta-oxidation is PTS1-targeted → VLCFA (C26:0) NORMAL in RCDP1
  This is the SINGLE MOST IMPORTANT biochemical distinction on exam.

PTS2-CARGO PROTEINS AFFECTED BY PEX7 LOF:
  (1) PHYH (phytanoyl-CoA 2-hydroxylase) → alpha-oxidation FAILED → phytanic acid ELEVATED
  (2) AGPS (alkylglycerone-phosphate synthase/ADHAPS) → ether-phospholipid biosynthesis FAILED
  (3) DHAPAT (dihydroxyacetone phosphate acyltransferase / GNPAT) — note: GNPAT itself uses PTS1;
      alkylDHAP synthase (AGPS product) uses PTS2 → NET RESULT: plasmalogen biosynthesis FAILED
  → Erythrocyte plasmalogens SEVERELY LOW (virtually absent in classic RCDP1)
  → Phytanic acid ELEVATED
  → DHA (docosahexaenoic acid) LOW (indirect, via peroxisomal elongation/retroconversion)

PTS1-PATHWAY (INTACT IN RCDP1 — CRUCIAL FOR DIFFERENTIAL DIAGNOSIS):
  VLCFA (C26:0) beta-oxidation → VLCFA NORMAL
  Pristanic acid beta-oxidation → NORMAL
  Pipecolic acid catabolism → NORMAL
  Dicarboxylic acids → NORMAL
  The above 4 being NORMAL = RCDP1; ALL 4 abnormal = PBD-ZSD (PEX1/PEX6)

LOCUS: 6q22.33  |  OMIM GENE: *601757  |  OMIM DISEASE: #215100 (RCDP, type 1)

EPIDEMIOLOGY:
  RCDP1 prevalence: ~1/100,000 births. PEX7 accounts for ~90% of all RCDP;
  RCDP2 (GNPAT): ~5%; RCDP3 (AGPS): ~5%; PEX7 is the ONLY PBD gene that is NOT a ZSD gene.
  Common variants: p.Leu292X (L292X) — Northern European founder, ~50% of European RCDP1 alleles;
  c.39+1G>A splice — ~15%; p.Gly217Arg (G217R) — ~10%; p.Arg357X (R357X) — UK founder ~8%.

PHENOTYPIC SPECTRUM:
  Classic RCDP1 (L292X/L292X or null/null): Rhizomelia + stippled epiphyses + cataracts + ichthyosis
    + profound ID + microcephaly. Seizures 60-70%. Median survival 1-2 years (rarely to 5y).
  Intermediate RCDP1 (null/hypomorphic or splice): Moderate skeletal + mild-moderate ID.
    Seizures 30-40%. Survival 10-20y.
  Mild RCDP1 (hypomorphic/hypomorphic): Minimal skeletal + variable ID.
    Seizures ~20%. Adult survival possible.

EPILEPSY (Central to management):
  Overall seizures: 65% of classic, 35% intermediate, 20% mild RCDP1.
  Types (classic): Infantile spasms (hypsarrhythmia) 40%, Focal 30%, Myoclonic 20%, GTCS 15%, SE 8%.
  Drug resistance: ~35-40% of those with seizures (multifactorial — hypomyelination + cortical).
  EEG: Hypsarrhythmia (IS), modified-hypsarrhythmia, multifocal spikes, background suppression.
  MRI: Hypomyelination (white matter delay), reduced/absent myelin in severe forms.

AED PHARMACOLOGY — CRITICAL DISTINCTIONS:
  LEV: FIRST-LINE (all forms); no peroxisomal interactions, no adrenal mechanism, safe.
  ACTH: Level B for infantile spasms (less Level-A data than ZSD/NALD due to rarity, but used).
  VGB: HIGH RISK — visual field loss (cataracts already present in 70%; additive visual impairment).
    Use ONLY if absolutely necessary; if used, monthly ophthalmological monitoring mandatory.
    LESS absolute than in ZSD (where retinopathy is universal and severe) — but still HIGH RISK.
  VPA: RELATIVE CI — hepatotoxicity (phytanic acid accumulation + chronic disease stress on liver);
    POLG1 exclusion MANDATORY (CPIC Grade A) before VPA; monitor LFT q3 months if VPA used.
    NOTE: Phytanic acid in RCDP1 comes from dietary phytol (NOT from VLCFA turnover as in ZSD),
    so VPA mechanism is hepatotoxicity only (NOT peroxisomal BO inhibition); less dangerous than ZSD.
  PHT/CBZ/OXC: CAN be used with caution — NO adrenal crisis mechanism (unlike ABCD1),
    NO cortisol drop via CYP3A4 enzyme induction (no adrenal insufficiency in RCDP1).
  Fasting: HAZARD (phytanic acid in adipose releases during fasting → surge);
    Less severe than ZSD (no VLCFA in adipose simultaneously) but IV glucose during surgery still prudent.

DISEASE-MODIFYING / SUPPORTIVE:
  Phytol-restricted diet: Reduce dietary phytol (meat/dairy/green vegetables) → lower phytanic acid.
    Beneficial across all phenotypes; reduces phytanic acid surges.
  Plasmalogen precursor supplementation: Alkylglycerols (batyl alcohol / chimyl alcohol / selachyl).
    Experimental Level C — raises RBC plasmalogens in animal models; phase II data limited.
  DHA supplementation: Level C — low DHA, less critical than ZSD.
  No approved ERT (2026) — no secreted enzyme to replace.
  No HSCT evidence — not indicated (unlike some LSDs where it arrests CNS progression).
"""

import random
random.seed(42)


# ── Overview (KPIs + summary) ─────────────────────────────────────────────────

def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 65,
        "classic_rcdp_pct": 45,
        "intermediate_rcdp_pct": 35,
        "mild_rcdp_pct": 20,
        "drug_resistance_pct": 35,
        "cataract_pct": 72,
        "ichthyosis_pct": 68,
        "rhizomelia_pct": 100,
        "stippling_pct": 88,
        "on_phytol_restricted_diet_pct": 75,
        "on_dha_pct": 40,
        "omim_gene": "601757",
        "omim_disease": "215100",
        "locus": "6q22.33",
        "vlcfa_normal": True,
        "vlcfa_normal_pct": 100,
        "phytanic_elevated_pct": 95,
        "plasmalogen_low_pct": 98,
        "inheritance": "Autosomal recessive (AR), biallelic LOF",
        "common_variant": "p.Leu292X (L292X) — Northern European founder ~50% of RCDP1 alleles",
        "disease_mechanism": (
            "PEX7 (Peroxin-7) is the cytosolic PTS2 receptor (323 aa, 7 WD40 repeats). "
            "PEX7 LOF selectively blocks PTS2-targeted cargo import (PHYH, AGPS) while leaving "
            "PTS1-pathway (VLCFA beta-oxidation) intact. Result: isolated impairment of plasmalogen "
            "biosynthesis + phytanic acid alpha-oxidation + DHA synthesis, with VLCFA NORMAL — "
            "the CRITICAL biochemical distinction from PBD-ZSD (PEX1/PEX6) where ALL pathways fail."
        ),
        "nbs_positive_rate": "Not yet in US NBS; advocacy ongoing for plasmalogen + phytanic on DBS",
        "key_concepts": [
            "PEX7 = PTS2 receptor only — does NOT affect PTS1 pathway (VLCFA beta-oxidation NORMAL)",
            "VLCFA NORMAL in RCDP1 — single most important biochemical distinction from ZSD",
            "Plasmalogens (RBC) SEVERELY LOW — nearly absent in classic RCDP1",
            "Phytanic acid ELEVATED — PHYH (phytanoyl-CoA hydroxylase) requires PTS2 import",
            "Rhizomelia + stippled epiphyses = PATHOGNOMONIC skeletal features",
            "Cataracts (72%) present at birth — additive VGB retinal toxicity risk",
            "VGB HIGH RISK due to cataracts + visual field loss; NOT absolute CI as in ZSD but HIGH RISK",
            "VPA RELATIVE CI — hepatotoxicity; POLG1 MANDATORY; NO peroxisomal BO inhibition (unlike ZSD)",
            "Phytanic from diet (NOT from stored VLCFA) — phytol-restricted diet is primary intervention",
            "No adrenal insufficiency → PHT/CBZ/OXC can be used (NO adrenal crisis unlike ABCD1)",
            "No ERT available — enzyme is intracellular PTS2 receptor, not secreted enzyme",
            "Plasmalogen precursor supplementation (alkylglycerols) experimental Level C",
            "RCDP2 (GNPAT) and RCDP3 (AGPS) biochemically identical — only gene panel distinguishes",
            "PEX7 accounts for ~90% of all RCDP; ~1/100,000 births prevalence",
            "No HSCT evidence (unlike ABCD1-CCALD or early Krabbe)",
            "Fasting hazard (phytanic adipose release) — IV glucose recommended during surgery",
        ],
        "standards": [
            "Braverman NE et al. Rhizomelic chondrodysplasia punctata type 1. GeneReviews 2015 (updated).",
            "Steinberg SJ et al. Peroxisome biogenesis disorders. GeneReviews 2012.",
            "Waterham HR et al. Disorders of peroxisome biogenesis and fatty acid oxidation. NEJM 2016.",
            "Wanders RJA. Peroxisomal disorders: clinical aspects. Biochim Biophys Acta 2013.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A: test before VPA in any seizure disorder).",
        ],
    }


# ── Breakdown (patients + seizures + treatments) ─────────────────────────────

def get_breakdown():
    etiologies = [
        {
            "name": "Classic RCDP1 (Null/Null or L292X/L292X)",
            "pct": 45,
            "n": 18,
            "sex": "M/F equal",
            "onset_age": "Neonatal–3 months",
            "seizure_risk": "65–70%",
            "eeg": "Hypsarrhythmia (infantile spasms), multifocal spike-wave, burst-suppression",
            "mri": "Severe hypomyelination, reduced/absent white matter signal, cortical atrophy",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic null or L292X homozygous → complete PTS2 receptor absence → "
                "plasmalogens virtually absent, phytanic severely elevated. Rhizomelia severe, "
                "cataracts congenital, profound ID, median survival 1–2 years."
            ),
        },
        {
            "name": "Intermediate RCDP1 (Null/Hypomorphic or Splice)",
            "pct": 35,
            "n": 14,
            "sex": "M/F equal",
            "onset_age": "3–12 months",
            "seizure_risk": "30–40%",
            "eeg": "Modified hypsarrhythmia, focal spikes temporal-occipital, multifocal discharges",
            "mri": "Moderate hypomyelination, periventricular white matter changes",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "One null + one hypomorphic (splice or missense) allele → partial PTS2 function. "
                "Plasmalogens 5–20% of normal. Moderate rhizomelia, moderate ID. Survival 10–20y."
            ),
        },
        {
            "name": "Mild RCDP1 (Hypomorphic/Hypomorphic — G217R or equivalent)",
            "pct": 20,
            "n": 8,
            "sex": "M/F equal",
            "onset_age": "6–24 months",
            "seizure_risk": "20–25%",
            "eeg": "Focal discharges, occasional multifocal spikes, may be near-normal",
            "mri": "Minimal/absent white matter changes; normal myelination possible",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic hypomorphic alleles (G217R/G217R or equivalent) → residual PTS2 function. "
                "Plasmalogens 20–50% of normal. Minimal rhizomelia, mild-moderate ID, adult survival."
            ),
        },
    ]

    # 40 synthetic patients
    phenotypes = (
        [("Classic RCDP1", "ZS-like")] * 18
        + [("Intermediate RCDP1", "Intermediate")] * 14
        + [("Mild RCDP1", "Mild")] * 8
    )
    sexes = (["M"] * 20 + ["F"] * 20)
    random.shuffle(sexes)

    genotype_map = {
        "Classic RCDP1": ["L292X/L292X", "L292X/null", "null/null", "L292X/R357X"],
        "Intermediate RCDP1": ["L292X/G217R", "L292X/IVS1+1G>A", "G217R/null", "IVS1+1G>A/null"],
        "Mild RCDP1": ["G217R/G217R", "G217R/missense", "missense/missense"],
    }
    aed_map = {
        "Classic RCDP1": ["LEV+ACTH", "LEV", "LEV+CLB", "ACTH", None],
        "Intermediate RCDP1": ["LEV", "LEV+LTG", "LEV+CLB", None, None],
        "Mild RCDP1": ["LEV", "LEV+LTG", None, None, None],
    }
    response_map = {
        "Classic RCDP1": ["Drug-resistant", "Partially controlled", "Drug-resistant", "Partially controlled"],
        "Intermediate RCDP1": ["Controlled", "Partially controlled", "Drug-resistant", "Controlled"],
        "Mild RCDP1": ["Controlled", "Controlled", "Partially controlled"],
    }

    patients = []
    for i, (phen, _) in enumerate(phenotypes):
        has_sei = (
            random.random() < 0.67 if "Classic" in phen else
            random.random() < 0.37 if "Intermediate" in phen else
            random.random() < 0.22
        )
        aed_choices = aed_map[phen]
        primary_aed = random.choice(aed_choices) if has_sei else None
        resp = random.choice(response_map[phen]) if has_sei and primary_aed else None
        patients.append({
            "patient_id": f"RCDP1-{i+1:02d}",
            "phenotype": phen,
            "sex": sexes[i],
            "genotype": random.choice(genotype_map[phen]),
            "cataract": random.random() < (0.88 if "Classic" in phen else 0.60 if "Intermediate" in phen else 0.25),
            "ichthyosis": random.random() < (0.80 if "Classic" in phen else 0.62 if "Intermediate" in phen else 0.30),
            "has_seizures": has_sei,
            "primary_aed": primary_aed,
            "drug_response": resp,
            "on_dha": random.random() < (0.45 if "Classic" in phen else 0.42 if "Intermediate" in phen else 0.30),
            "on_phytol_restricted_diet": True,
        })

    seizure_types = [
        {"type": "Infantile Spasms (Hypsarrhythmia)",
         "pct": 40, "eeg": "Hypsarrhythmia; modified-hypsarrhythmia common; ACTH Level B"},
        {"type": "Focal Onset (Temporal/Occipital)",
         "pct": 30, "eeg": "Temporal or occipital focal spikes; secondary generalization"},
        {"type": "Myoclonic",
         "pct": 20, "eeg": "Generalized polyspike-wave; photosensitivity in some"},
        {"type": "Generalized Tonic-Clonic (GTCS)",
         "pct": 15, "eeg": "Generalized spike-wave; often from focal evolution"},
        {"type": "Status Epilepticus (SE)",
         "pct": 8, "eeg": "Electrographic SE in severe forms; refractory to standard protocols"},
    ]

    triggers = [
        {"trigger": "Febrile illness", "pct": 65,
         "note": "Most common trigger in classic RCDP1; fever lowers seizure threshold via hypomyelination"},
        {"trigger": "Phytanic acid dietary surge", "pct": 48,
         "note": "High-phytol meal (dairy/fatty meat/green veg) → phytanic spike → neuro toxicity"},
        {"trigger": "Fasting / pre-operative NPO", "pct": 35,
         "note": "Adipose phytanic release during fasting; IV glucose recommended peri-operatively"},
        {"trigger": "Sleep deprivation", "pct": 30,
         "note": "Common generalised trigger; more prominent in intermediate/mild forms"},
        {"trigger": "Metabolic decompensation", "pct": 25,
         "note": "Intercurrent illness → plasmalogen depletion worsening → CNS destabilisation"},
        {"trigger": "Missed AED dose", "pct": 22,
         "note": "Non-compliance; LEV withdrawal particularly significant"},
        {"trigger": "Anaesthesia / surgery", "pct": 15,
         "note": "Pre-op phytanic surge + peri-op stress; IV dextrose + maintain diet pre-op"},
    ]

    monitoring = [
        "Plasma phytanic acid: every 3 months (target <10 μmol/L on dietary control)",
        "RBC plasmalogens (C16:0 and C18:0 dimethylacetal): baseline and every 12 months",
        "VLCFA (C26:0) panel annually — should be NORMAL; if elevated, reconsider diagnosis",
        "DHA level annually; DHA supplementation if <2% of total fatty acids",
        "EEG: baseline at seizure onset; repeat 6-monthly or after AED change",
        "Ophthalmology: every 6 months (cataracts, VF monitoring; VGB contraindication check)",
        "Skeletal X-ray: annual in first 5 years (stippling regression, contracture progression)",
        "LFTs every 3 months if on VPA; pre-VPA POLG1 genotyping mandatory (CPIC A)",
        "Developmental assessment (Bayley/Griffiths): every 6 months in first 2 years",
        "Respiratory function (FVC): annually in classic/intermediate (phrenic weakness, kyphoscoliosis)",
        "Hearing screen: annually (sensorineural hearing loss 20% of classic RCDP1)",
    ]

    thresholds = [
        {"parameter": "Phytanic acid (plasma)", "threshold": ">10 μmol/L",
         "action": "Intensify phytol-restricted diet; consider plasmapheresis if >200 μmol/L"},
        {"parameter": "RBC plasmalogens", "threshold": "<50% of normal",
         "action": "Initiate alkylglycerol supplementation (experimental Level C)"},
        {"parameter": "VLCFA C26:0", "threshold": "Elevated (>0.97 μg/mL)",
         "action": "Re-evaluate diagnosis — if elevated, consider ZSD (PEX1/PEX6); PEX7 should be NORMAL"},
        {"parameter": "LFT (ALT/AST)", "threshold": ">3× ULN on VPA",
         "action": "Discontinue VPA immediately; switch to LEV; liver biopsy if >5× ULN"},
        {"parameter": "Visual field (VGB monitoring)", "threshold": "Any VF loss",
         "action": "Stop VGB immediately; switch to LEV+CLB; document irreversibility"},
        {"parameter": "Infantile spasms response to ACTH", "threshold": "<2 weeks no response",
         "action": "Add LEV; consider CLB; avoid VGB (cataracts + VF); neurology review"},
        {"parameter": "DHA (plasma)", "threshold": "<2% total FA",
         "action": "Initiate DHA 200 mg/day; recheck at 3 months"},
    ]

    lifecycle = [
        {"stage": "Neonatal (0–1 month)",
         "features": "Rhizomelia detected; stippled epiphyses on X-ray; congenital cataracts; ichthyosis; hypotonia. Biochemistry: plasmalogens absent, phytanic elevated, VLCFA NORMAL.",
         "action": "Confirm PEX7 sequencing; start phytol-restricted diet (maternal diet if breastfeeding); DHA supplementation; ophthalmology urgent"},
        {"stage": "Early Infantile (1–6 months)",
         "features": "Infantile spasms onset (peak 3–5 months in classic RCDP1); hypsarrhythmia on EEG; profound hypotonia; feeding difficulties",
         "action": "ACTH Level B; LEV first-line; avoid VGB (cataracts); monitor phytanic acid; plasmalogen supplementation if available"},
        {"stage": "Late Infantile / Toddler (6–24 months)",
         "features": "Focal and myoclonic seizures; severe contractures; no ambulation in classic; moderate delay in intermediate; VF concerns if VGB considered",
         "action": "LEV ± CLB ± LTG; strict phytol-restricted diet; physiotherapy for contractures; wheelchair assessment"},
        {"stage": "Early Childhood (2–5 years)",
         "features": "Seizure burden may reduce spontaneously in intermediate/mild; kyphoscoliosis; chronic respiratory compromise in severe; cataract surgery if indicated",
         "action": "Annual skeletal X-ray; respiratory spirometry; continue AED; annual phytanic + plasmalogen; DHA supplementation"},
        {"stage": "School Age (5–10 years)",
         "features": "Classic RCDP1: usually deceased or profound disability; Intermediate: special education; Mild: mainstream with support; GTCS may emerge",
         "action": "Educational planning; annual epilepsy review; VPA avoidance maintained; LFT if any AED hepatotoxic risk"},
        {"stage": "Adolescent / Adult (≥10 years)",
         "features": "Mild RCDP1: adult survival possible; focal epilepsy manageable; phytanic acid requires lifelong restriction; sensorineural hearing loss may progress",
         "action": "Adult neurology transition; phytol-restricted diet lifelong; consider alkylglycerol supplementation; reproductive counselling (AR 25% risk)"},
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "SV2A modulator — FIRST-LINE all RCDP1 forms",
            "evidence": "Level A (standard first-line for all peroxisomal epilepsies)",
            "dose": "Paediatric: 20–60 mg/kg/day div q12h; Adult: 500–3000 mg/day",
            "moa": "Binds SV2A synaptic vesicle protein → reduces neurotransmitter release",
            "monitoring": "Behaviour (irritability in RCDP1); no hepatic monitoring required",
            "ci": None,
        },
        {
            "drug": "ACTH (Tetracosactide/Synacthen)",
            "class": "Corticotropin — Level B (infantile spasms)",
            "evidence": "Level B — less RCT data than ZSD/NALD (rarity limits studies); used by analogy with IS of structural-metabolic cause",
            "dose": "0.5–1.0 mg/day IM for 2 weeks then taper; or short Dexamethasone protocol",
            "moa": "Anti-epileptic via ACTH receptor (melanocortin MC2R); suppresses CLIP; reduces CRH",
            "monitoring": "BP, glucose, infection risk during immunosuppression",
            "ci": "Active untreated infection; contraindicated if adrenal tumour (not relevant in RCDP1)",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "Benzodiazepine (long-acting) — Level C adjunct",
            "evidence": "Level C — adjunct for focal and GTCS in RCDP1 and structural epilepsies",
            "dose": "0.1–1.0 mg/kg/day; max 40 mg/day",
            "moa": "Positive allosteric modulator GABA-A at benzodiazepine site",
            "monitoring": "Sedation; tolerance (tolerance to anti-seizure effect within months common)",
            "ci": None,
        },
        {
            "drug": "Lamotrigine (LTG)",
            "class": "Sodium channel blocker — Level C adjunct",
            "evidence": "Level C — adjunct in intermediate/mild forms; caution with rapid titration",
            "dose": "0.15–0.3 mg/kg/day titrating over 8 weeks to 1–3 mg/kg/day (without VPA)",
            "moa": "Blocks voltage-gated Na+ channels; reduces glutamate release",
            "monitoring": "Rash (Stevens-Johnson), slow titration mandatory",
            "ci": "Avoid rapid titration; VPA coadmin doubles LTG levels",
        },
        {
            "drug": "Phytol-Restricted Diet",
            "class": "Dietary intervention — ALL forms, lifelong",
            "evidence": "Level B — reduces plasma phytanic acid; reduces phytanic surge episodes",
            "dose": "Restrict dietary phytol: avoid dairy fat, fatty meat, green vegetables; max phytol ~20–30 mg/day",
            "moa": "Phytanic acid is derived exclusively from dietary phytol (chlorophyll-bound) via gut microbiota; dietary restriction is the ONLY direct intervention",
            "monitoring": "Phytanic acid q3 months (target <10 μmol/L); nutritional status (DHA, vitamins A/D/E/K)",
            "ci": None,
        },
        {
            "drug": "DHA Supplementation (Docosahexaenoic Acid)",
            "class": "Polyunsaturated fatty acid — Level C",
            "evidence": "Level C — improves DHA status; clinical benefit in neural function uncertain",
            "dose": "200 mg/day (infants); 500–1000 mg/day (older children/adults)",
            "moa": "Replaces DHA lost due to peroxisomal DHA synthesis impairment",
            "monitoring": "Plasma DHA level annually; hepatic lipid if high dose",
            "ci": None,
        },
        {
            "drug": "Alkylglycerol Precursor Supplementation (Batyl/Chimyl/Selachyl Alcohol)",
            "class": "Plasmalogen precursor — Experimental Level C",
            "evidence": "Level C — raises RBC plasmalogens in animal models; human data limited (n<20)",
            "dose": "50–100 mg/kg/day (research use only); no approved formulation",
            "moa": "Alkylglycerols bypass blocked AGPS/ADHAPS step → direct ether-lipid precursor",
            "monitoring": "RBC plasmalogens (C16:0-DMA, C18:0-DMA); phytanic acid (some formulations contain trace phytol)",
            "ci": "Formulations may contain impurities; clinical trial preferred",
        },
    ]

    contraindications = [
        {
            "drug": "Vigabatrin (VGB)",
            "level": "HIGH RISK — Cataracts + VF Loss (NOT absolute as in ZSD, but HIGH RISK)",
            "reason": (
                "RCDP1: cataracts present in 72% + VGB causes irreversible visual field constriction. "
                "Additive visual impairment. Use ONLY if infantile spasms fail ACTH+LEV, benefit-risk "
                "discussion with family mandatory. Monthly VF monitoring (VEP in infants unable to cooperate). "
                "Distinction from ZSD: in ZSD retinopathy is universal and severe → VGB near-absolute CI; "
                "in RCDP1 cataracts are the primary concern (VGB retinopathy is additive)."
            ),
            "alternative": "ACTH (Level B IS) + LEV; CLB adjunct; avoid VGB if cataracts present",
        },
        {
            "drug": "Valproate (VPA)",
            "level": "RELATIVE CI — Hepatotoxicity (NOT absolute; POLG1 MANDATORY)",
            "reason": (
                "Hepatotoxicity risk in chronic disease state (phytanic accumulation stresses liver). "
                "NOTE: Unlike ZSD (PEX1/PEX6), VPA does NOT worsen VLCFA accumulation via peroxisomal "
                "beta-oxidation inhibition (VLCFA pathway NORMAL in RCDP1). But hepatotoxicity risk remains. "
                "POLG1 sequencing mandatory before VPA (CPIC Grade A). LFTs q3 months if VPA used."
            ),
            "alternative": "LEV (first-line); CLB or LTG as adjuncts",
        },
        {
            "drug": "Fasting / Pre-operative NPO",
            "level": "HAZARD (Phytanic Surge — less severe than ZSD but still significant)",
            "reason": (
                "Fasting mobilises adipose phytanic acid stores → acute phytanic elevation → neurological "
                "deterioration + seizure threshold lowering. Less severe than ZSD (no concurrent VLCFA surge "
                "from adipose, but phytanic alone causes ataxia + polyneuropathy)."
            ),
            "alternative": "IV dextrose (glucose ≥10% IV) during peri-operative fast; maintain phytol-restricted enteral feed where possible",
        },
        {
            "drug": "Phenytoin (PHT) / Fosphenytoin (IV)",
            "level": "CAUTION — NOT absolute CI (unlike ABCD1 or NPC1)",
            "reason": (
                "No adrenal insufficiency in RCDP1 → no cortisol drop via CYP3A4 → PHT is NOT contraindicated "
                "by adrenal mechanism. IV fosphenytoin can be used in SE when IV LEV fails. CYP induction "
                "of phytanic acid metabolism is not clinically relevant. Monitor for neuropathy exacerbation."
            ),
            "alternative": "IV LEV preferred in SE (no enzyme induction); PHT acceptable if IV LEV fails",
        },
        {
            "drug": "Typical Antipsychotics",
            "level": "RELATIVE CI — Extrapyramidal (EPS) Risk",
            "reason": (
                "Hypomyelination in RCDP1 increases EPS susceptibility. No cataplexy (unlike NPC1/NPC2). "
                "If psychiatric manifestations arise (rare), atypical antipsychotics preferred."
            ),
            "alternative": "Atypical antipsychotics (lower EPS risk); functional behaviour support",
        },
    ]

    return {
        "etiologies": etiologies,
        "patients": patients,
        "seizure_types": seizure_types,
        "triggers": triggers,
        "monitoring": monitoring,
        "thresholds": thresholds,
        "lifecycle": lifecycle,
        "treatments": treatments,
        "contraindications": contraindications,
    }


# ── Definitions (glossary + algorithms) ───────────────────────────────────────

def get_definitions():
    return {
        "key_concepts": [
            "PEX7 = PTS2 receptor (Peroxisomal Targeting Signal 2): cytosolic 323-aa WD40-repeat protein that shuttles PTS2-cargo (PHYH, AGPS) into peroxisomal matrix. Loss → selective PTS2 pathway block only.",
            "RCDP1 vs ZSD: THE KEY DISTINCTION — VLCFA (C26:0) NORMAL in RCDP1 (PTS1 pathway intact); VLCFA HIGH in ZSD (PEX1/PEX6). Test C26:0 FIRST in any peroxisomal disease workup.",
            "Plasmalogens (RBC) — severely low/absent in RCDP1 (all forms); reflect AGPS (alkylDHAP synthase) import failure. Biomarker of disease severity and dietary/supplement response.",
            "Phytanic acid — elevated in RCDP1 because PHYH (phytanoyl-CoA 2-hydroxylase) requires PTS2 import. Phytol source = dietary only (meat fat, dairy, green vegetables). Dietary restriction is primary treatment.",
            "Rhizomelia — shortening of PROXIMAL limbs (humerus > femur), distinguishes RCDP from other skeletal dysplasias. Present in 100% classic RCDP1. Upper limb rhizomelia more prominent.",
            "Stippled epiphyses (calcific stippling) — punctate calcification of cartilage visible on X-ray in neonates. PATHOGNOMONIC. Resolves by age 3–5 years; absence after age 5 does NOT exclude RCDP1.",
            "VGB in RCDP1 — HIGH RISK (not absolute CI unlike ZSD): cataracts present in 72% + VGB VF loss = additive blindness. Distinguished from ZSD where retinopathy is universal/severe → near-absolute CI.",
            "VPA in RCDP1 — RELATIVE CI (hepatotoxicity only, NOT peroxisomal BO inhibition): VLCFA pathway intact so VPA does NOT worsen VLCFA via BO inhibition. Still needs POLG1 exclusion (CPIC A).",
            "PHT/CBZ in RCDP1 — CAN BE USED (no adrenal crisis mechanism): no adrenal insufficiency in RCDP1 → no cortisol drop → no adrenal crisis. Contrast with ABCD1 where PHT/fosphenytoin = ABSOLUTE CI.",
            "RCDP2 (GNPAT) and RCDP3 (AGPS) — biochemically identical to RCDP1 (same pathway, downstream enzymes); distinguished ONLY by gene sequencing. GNPAT encodes DHAPAT; AGPS encodes alkylDHAP synthase (downstream of DHAPAT).",
            "Plasmalogen precursor therapy — alkylglycerols (batyl/chimyl/selachyl alcohol) bypass the blocked AGPS step by providing a direct ether-lipid precursor. Raises RBC plasmalogens in animal models; Level C human data.",
            "No ERT in RCDP1 — PEX7 is an intracellular cytosolic receptor (not a secreted enzyme). ERT can only replace secreted/lysosomal enzymes (MPSs, Gaucher, Fabry, Pompe). No substrate equivalent.",
            "DHA low in RCDP1 — peroxisomal DHA synthesis (retroconversion of longer PUFA) is partially impaired. DHA supplementation Level C; less critical than in ZSD where DHA deficiency causes pachygyria.",
            "Fasting hazard — phytanic acid in adipose is released during fasting → acute neurotoxicity. IV glucose recommended peri-operatively. Less severe than ZSD (no concurrent VLCFA adipose surge) but still clinically relevant.",
            "No HSCT indication — unlike ABCD1 (CCALD where HSCT arrests neuroinflammation) and early Krabbe. RCDP1 pathology is NOT neuroinflammatory; demyelination is hypomyelination due to plasmalogen deficiency.",
            "Genotype-phenotype correlation in RCDP1 — L292X/L292X → classic severe; null/hypomorphic → intermediate; hypomorphic/hypomorphic → mild. Better correlation than ABCD1 but incomplete; anticipate range.",
        ],
        "diagnostic_algorithm": [
            "Step 1: Clinical suspicion — rhizomelia + stippled epiphyses (neonate) OR congenital cataracts + ichthyosis + hypotonia (infant) OR unexplained epilepsy + intellectual disability + skeletal abnormality.",
            "Step 2: Biochemistry — PLASMA VLCFA panel (C26:0 C26:0/C22:0, C24:0/C22:0): must be NORMAL. If ELEVATED → ZSD (PEX1/PEX6) not RCDP1.",
            "Step 3: RBC PLASMALOGENS (C16:0-DMA, C18:0-DMA dimethylacetals): severely low/absent in RCDP1. Pathological level confirms ether-lipid biosynthesis defect.",
            "Step 4: PLASMA PHYTANIC ACID: elevated in RCDP1 (>10 μmol/L diagnostic). Also elevated in adult Refsum disease (PHYH or PEX7 mutation) — distinguish by age/clinical context.",
            "Step 5: PIPECOLIC ACID plasma — NORMAL in RCDP1 (PTS1 pathway intact). Elevated in ZSD → confirms ZSD if elevated. RCDP1: NORMAL.",
            "Step 6: PRISTANIC ACID plasma — NORMAL in RCDP1 (2-methylacyl-CoA racemase, AMACR, is PTS1). Elevated in ZSD.",
            "Step 7: Skeletal X-ray — rhizomelia (humerus shortening), stippled epiphyses in cartilage (neonatal), progressive contractures.",
            "Step 8: Ophthalmology — cataracts (72%); visual field assessment; ERG if VGB use considered.",
            "Step 9: MRI brain — hypomyelination (reduced T2 white matter signal), cortical atrophy in severe; near-normal in mild.",
            "Step 10: Gene sequencing — PEX7 gene (first); if negative: GNPAT (RCDP2), AGPS (RCDP3). All three are biochemically identical.",
            "Step 11: POLG1 genotyping — MANDATORY before any VPA consideration (CPIC Grade A).",
            "Step 12: Confirm diagnosis, institute phytol-restricted diet, DHA supplementation, contact specialist centre (Kennedy Krieger, Manchester, Amsterdam).",
        ],
        "pharmacological_distinctions": [
            "LEV — FIRST-LINE all RCDP1 forms (no peroxisomal interactions, no adrenal mechanism, no phytanic worsening; safe).",
            "ACTH — Level B infantile spasms (less RCT data than ZSD due to rarity; used by analogy with structural-metabolic IS). Avoid VGB as first-line for IS in RCDP1.",
            "VGB — HIGH RISK in RCDP1 (cataracts 72% + VF loss = additive visual impairment). NOT absolute CI (unlike ZSD where retinopathy is universal/severe). Monthly VF/VEP monitoring if used.",
            "VPA — RELATIVE CI (hepatotoxicity only; NOT peroxisomal BO inhibition unlike ZSD). POLG1 MANDATORY. LFT q3 months. Phytanic acid NOT worsened by VPA (phytanic from diet, VLCFA pathway intact).",
            "PHT/CBZ/OXC — CAN BE USED (no adrenal insufficiency in RCDP1 → no CYP3A4 cortisol-drop mechanism → no adrenal crisis). Contrast with ABCD1 = ABSOLUTE CI. CYP enzyme induction not relevant to phytanic.",
            "Fosphenytoin IV — can be used in SE if IV LEV fails (no adrenal mechanism). Unlike ABCD1 where fosphenytoin = ABSOLUTE CI (adrenal crisis) or NPC where PHT = ABSOLUTE CI (disease worsening).",
            "Benzodiazepines (CLB/MDZ) — safe adjuncts; MDZ nasal/buccal for acute SE. CLB for chronic adjunct therapy.",
            "Typical antipsychotics — HIGH RISK EPS (hypomyelination increases susceptibility). Atypicals preferred if needed.",
            "Phytol-restricted diet — Level B dietary intervention; reduces phytanic acid; primary disease-modifying tool available in 2026.",
            "Alkylglycerol supplementation — experimental Level C; raises RBC plasmalogens; no approved formulation; clinical trial preferred.",
            "DHA supplementation — Level C; replaces low DHA; 200 mg/day infants; 500–1000 mg/day older.",
            "No ERT, No HSCT, No gene therapy (2026) — RCDP1 has no approved disease-modifying biological therapy beyond dietary restriction and experimental supplements.",
        ],
        "differential_diagnosis": [
            {"condition": "ZSD (PEX1/PEX6)", "distinction": "VLCFA HIGH in ZSD; VLCFA NORMAL in RCDP1. ZSD: ALL pathways impaired. RCDP1: only PTS2 pathways. ZSD: no rhizomelia/stippling. Neonatal ZS fatal in 12M."},
            {"condition": "RCDP2 (GNPAT) / RCDP3 (AGPS)", "distinction": "Biochemically identical to RCDP1 (same plasmalogen/phytanic biomarker profile). Distinguished ONLY by gene sequencing. GNPAT/AGPS are downstream enzymes of the same PTS2 pathway."},
            {"condition": "Adult Refsum Disease (PHYH or PEX7 mild)", "distinction": "Phytanic elevated but plasmalogens NORMAL (PHYH deficiency) or mildly low (PEX7 mild). No rhizomelia/stippling. Adult onset. Distinguish by age, plasmalogens, PEX7 vs PHYH sequencing."},
            {"condition": "CDPX2 / Conradi-Hunermann (EBP gene)", "distinction": "X-linked dominant stippling; VLCFA NORMAL, plasmalogens NORMAL, phytanic NORMAL. Asymmetric skin lesions (Blaschko lines). EBP encodes delta(8)-delta(7) sterol isomerase; cholesterol pathway, not peroxisomal."},
            {"condition": "ABCD1 (X-ALD)", "distinction": "VLCFA HIGH; plasmalogens NORMAL (NOT low). No rhizomelia/stippling. Adrenal insufficiency 71% males. X-linked. PHT = ABSOLUTE CI (adrenal crisis). HSCT indicated (CCALD)."},
            {"condition": "MPS (MPSs: Hurler/Hunter/Sanfilippo)", "distinction": "Coarse features, dysostosis multiplex. VLCFA NORMAL, plasmalogens NORMAL, phytanic NORMAL. Urine GAG elevated. Enzyme assay diagnostic. Completely different pathway (lysosomal, not peroxisomal)."},
            {"condition": "Non-peroxisomal skeletal dysplasias (achondroplasia, CHH)", "distinction": "VLCFA/plasmalogens/phytanic all NORMAL. FGFR3 or RMRP mutations. No biochemical peroxisomal signature. Rhizomelia present in achondroplasia but NO cataracts/ichthyosis/stippling pattern."},
        ],
        "standards": [
            "Braverman NE et al. Rhizomelic Chondrodysplasia Punctata Type 1. GeneReviews. NCBI Bookshelf. Updated 2015.",
            "Steinberg SJ, Raymond GV, Braverman NE, Moser AB. Peroxisome Biogenesis Disorders, Zellweger Syndrome Spectrum. GeneReviews. 2012.",
            "Waterham HR, Ferdinandusse S, Wanders RJA. Human disorders of peroxisome metabolism and biogenesis. Biochim Biophys Acta. 2016;1863(5):922–933.",
            "Wanders RJA, Waterham HR. Biochemistry of mammalian peroxisomes revisited. Annu Rev Biochem. 2006;75:295–332.",
            "CPIC Guideline — Valproic Acid and POLG1: Grade A recommendation (test before VPA). cpicpgx.org.",
            "Braverman NE et al. A Pex7 hypomorphic mouse model for plasmalogen deficiency affecting the lens and skeleton. Mol Genet Metab. 2010;99(4):408–416.",
            "Powers JM et al. Rhizomelic chondrodysplasia punctata type 1 (RCDP1): clinical and pathological features of affected older patients. J Neuropathol Exp Neurol. 2020.",
        ],
    }
