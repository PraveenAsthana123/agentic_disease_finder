#!/usr/bin/env python3
"""GNPAT / Rhizomelic Chondrodysplasia Punctata Type 2 (RCDP2) Epilepsy Dashboard — seed data module.

GNPAT encodes Glyceronephosphate O-Acyltransferase (DHAPAT — dihydroxyacetone phosphate
acyltransferase; 680 aa), the FIRST enzyme in the peroxisomal ether-phospholipid (plasmalogen)
biosynthesis pathway. GNPAT uses a PTS1 signal (C-terminal tripeptide SKF) for peroxisomal
import — this is a CRITICAL distinction from PEX7 (PTS2 receptor).

ENZYME FUNCTION:
  Step 1 (GNPAT/DHAPAT): DHAP + fatty acyl-CoA → 1-acyl-DHAP  [BLOCKED in RCDP2]
  Step 2 (AGPS):          1-acyl-DHAP + fatty alcohol → 1-alkyl-DHAP  [downstream, intact]
  Net result: ether-phospholipid / plasmalogen biosynthesis fails at Step 1.
  RBC plasmalogens SEVERELY LOW (virtually absent in classic RCDP2) — same as RCDP1.

CRITICAL BIOCHEMICAL DISTINCTION FROM RCDP1 (PEX7 DEFICIENCY):
  RCDP1 (PEX7): PTS2 receptor absent → PHYH (PTS2 cargo) CANNOT be imported → phytanic ELEVATED
  RCDP2 (GNPAT): Enzyme LOF, PEX7 INTACT → PHYH IS imported via PTS2 normally → phytanic NORMAL
    or only MILDLY elevated (secondary).
  This biochemical difference does NOT change the clinical phenotype significantly.
  KEY DIAGNOSTIC POINT: In RCDP2, phytanic acid is NORMAL or mildly elevated (< in RCDP1).
  Both RCDP2 and RCDP3 show plasmalogens severely low but phytanic usually normal.

PTS2-PATHWAY IN RCDP2 (INTACT — CONTRAST WITH RCDP1):
  PEX7 intact → PHYH imported → phytanic acid alpha-oxidation NORMAL → phytanic NORMAL
  PEX7 intact → AGPS imported → AGPS enzyme present, BUT cannot act (no substrate from GNPAT)
  Net: AGPS is present in peroxisome but has no 1-acyl-DHAP substrate → plasmalogen still fails.

LOCUS: 1q42.2  |  OMIM GENE: *602744  |  OMIM DISEASE: #222765
PROTEIN: 680 aa, peroxisomal matrix enzyme, PTS1 import (C-terminal -SKF-)
EPIDEMIOLOGY: ~5% of all RCDP (vs RCDP1/PEX7 ~90%); <200 reported cases worldwide 2026.
NO FOUNDER MUTATION: Unlike PEX7 (L292X ~50% European), GNPAT has mostly private/rare variants.

ALKYLGLYCEROL PHARMACOLOGY — PARTICULARLY RELEVANT IN RCDP2:
  Alkylglycerols (batyl/chimyl/selachyl alcohol) are ether-lipid precursors that BYPASS the
  GNPAT step entirely — they enter as pre-formed alkyl-DHAP equivalents downstream of Step 1.
  In RCDP2 (GNPAT blocked at Step 1), alkylglycerols bypass the primary blocked step and
  feed directly into Step 2 (AGPS) → more DIRECT substrate bypass than in RCDP1.
  This makes alkylglycerol supplementation PHARMACOLOGICALLY MOST RELEVANT in RCDP2,
  though human clinical evidence remains Level C (limited n).

PHENOTYPIC SPECTRUM (similar to RCDP1, may be slightly milder on average):
  Classic RCDP2: Rhizomelia + stippled epiphyses + cataracts + ichthyosis.
    Seizures 55-65%. Profound ID. Median survival < 5 years.
  Intermediate RCDP2: Moderate skeletal + mild-moderate ID. Seizures 30-40%. Survival 10-20y.
  Mild RCDP2: Minimal skeletal + variable ID. Seizures ~15-20%. Adult survival reported.

EPILEPSY:
  Overall seizures: 55-65% classic, 30-40% intermediate, 15-20% mild RCDP2.
  Types: Infantile spasms (IS) 38%, Focal 30%, Myoclonic 18%, GTCS 14%, SE 7%.
  Drug resistance: ~30-35% of those with seizures.

AED PHARMACOLOGY (identical to RCDP1 with phytanic-related nuances):
  LEV: FIRST-LINE (same as RCDP1); no peroxisomal interactions, no adrenal mechanism.
  ACTH: Level B for IS (less RCT data due to extreme rarity; used by RCDP1 analogy).
  VGB: HIGH RISK — cataracts 60-70% + VGB visual field loss = additive risk.
    Less data than RCDP1 (rarity) but same mechanism. Monthly VF/VEP monitoring if used.
  VPA: RELATIVE CI — hepatotoxicity; POLG1 MANDATORY (CPIC Grade A). Note: phytanic is
    usually NORMAL in RCDP2, so the phytanic-liver stress mechanism is less prominent
    than in RCDP1, but POLG1 exclusion remains MANDATORY and hepatotoxicity risk persists.
  PHT/CBZ/OXC: CAN USE — no adrenal insufficiency in RCDP2 → no cortisol drop → no adrenal
    crisis. Same as RCDP1 in this respect (distinct from ABCD1 where PHT = ABSOLUTE CI).
  Phytol-restricted diet: LESS CRITICAL than in RCDP1 (phytanic usually normal) but
    sometimes recommended empirically if phytanic mildly elevated.
  Alkylglycerol supplementation: PARTICULARLY RELEVANT (directly bypasses GNPAT block).
    Experimental Level C; raises RBC plasmalogens in animal models; human data n<20.
"""

import random
random.seed(43)


# ── Overview (KPIs + summary) ─────────────────────────────────────────────────

def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 60,
        "classic_rcdp_pct": 40,
        "intermediate_rcdp_pct": 40,
        "mild_rcdp_pct": 20,
        "drug_resistance_pct": 32,
        "cataract_pct": 65,
        "ichthyosis_pct": 62,
        "rhizomelia_pct": 100,
        "stippling_pct": 85,
        "on_alkylglycerol_pct": 30,
        "on_dha_pct": 38,
        "omim_gene": "602744",
        "omim_disease": "222765",
        "locus": "1q42.2",
        "vlcfa_normal": True,
        "vlcfa_normal_pct": 100,
        "phytanic_normal_pct": 82,
        "plasmalogen_low_pct": 97,
        "inheritance": "Autosomal recessive (AR), biallelic LOF",
        "common_variant": "No founder mutation — mostly private/rare variants; >40 pathogenic reported",
        "disease_mechanism": (
            "GNPAT (Glyceronephosphate O-Acyltransferase / DHAPAT) is the first enzyme in "
            "peroxisomal plasmalogen biosynthesis (680 aa, PTS1 import via C-terminal -SKF-). "
            "GNPAT LOF blocks Step 1 (DHAP → 1-acyl-DHAP), preventing all downstream ether-lipid "
            "synthesis. Plasmalogens severely low as in RCDP1, but phytanic acid is NORMAL "
            "(PEX7 intact → PHYH imported normally) — key biochemical difference from RCDP1. "
            "Alkylglycerols MOST DIRECTLY bypass the GNPAT block (substrate bypass at Step 1), "
            "giving RCDP2 the strongest pharmacological rationale for alkylglycerol supplementation."
        ),
        "nbs_positive_rate": "Not in NBS; plasmalogens on DBS advocated; phytanic normal → not triggered by routine NBS peroxisomal screens",
        "key_concepts": [
            "GNPAT = DHAPAT, first enzyme of plasmalogen synthesis (PTS1 import, NOT PTS2)",
            "RCDP2 accounts for ~5% of all RCDP (vs RCDP1/PEX7 ~90%; RCDP3/AGPS ~5%)",
            "Plasmalogens (RBC) SEVERELY LOW — same as RCDP1; primary diagnostic biomarker",
            "Phytanic acid NORMAL in RCDP2 (PEX7 intact → PHYH PTS2 import unaffected)",
            "VLCFA NORMAL — same as RCDP1; distinguishes from ZSD (PEX1/PEX6) where VLCFA high",
            "No founder mutation — unlike L292X in PEX7/RCDP1; diagnosis requires full gene sequencing",
            "Cataracts 60-70%, rhizomelia 100%, stippled epiphyses 85% — same clinical hallmarks as RCDP1",
            "VGB HIGH RISK — cataracts + VGB visual field loss = additive visual impairment",
            "VPA RELATIVE CI — hepatotoxicity; POLG1 MANDATORY (phytanic less of an issue vs RCDP1)",
            "PHT/CBZ/OXC CAN BE USED — no adrenal insufficiency (unlike ABCD1 absolute CI)",
            "Alkylglycerols MOST PHARMACOLOGICALLY DIRECT in RCDP2 — bypass the primary GNPAT block",
            "No ERT (2026) — enzyme is peroxisomal matrix protein, not secreted/lysosomal",
            "No HSCT — not neuroinflammatory (hypomyelination, not demyelination)",
            "Phytol-restricted diet LESS CRITICAL than RCDP1 but empirically used if phytanic mildly elevated",
            "Distinguished from RCDP1/RCDP3 ONLY by gene sequencing — all three biochemically similar",
        ],
        "standards": [
            "Honsho M et al. Deficiency of plasmalogen causes age-dependent peroxisomal dysfunction. Sci Rep. 2019.",
            "Braverman NE et al. Rhizomelic chondrodysplasia punctata. GeneReviews 2015.",
            "Waterham HR et al. Disorders of peroxisome metabolism. NEJM 2016.",
            "de Vet EC et al. Characterization of human DHAPAT. J Biol Chem. 1998.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A). cpicpgx.org.",
        ],
    }


# ── Breakdown (patients + seizures + treatments) ─────────────────────────────

def get_breakdown():
    etiologies = [
        {
            "name": "Classic RCDP2 (Null/Null or Severe LOF)",
            "pct": 40,
            "n": 16,
            "sex": "M/F equal",
            "onset_age": "Neonatal–3 months",
            "seizure_risk": "55–65%",
            "eeg": "Hypsarrhythmia (infantile spasms), multifocal spike-wave, burst-suppression",
            "mri": "Severe hypomyelination, reduced white matter signal, cortical atrophy",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic null or severe LOF (frameshift/nonsense) → complete GNPAT enzyme absence → "
                "plasmalogens virtually absent. Rhizomelia severe, cataracts congenital, profound ID. "
                "No founder mutation; every case has private or rare variants."
            ),
        },
        {
            "name": "Intermediate RCDP2 (Null/Hypomorphic or Splice)",
            "pct": 40,
            "n": 16,
            "sex": "M/F equal",
            "onset_age": "3–18 months",
            "seizure_risk": "30–40%",
            "eeg": "Modified hypsarrhythmia, focal spikes, multifocal discharges, theta slowing",
            "mri": "Moderate hypomyelination, periventricular signal changes",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "One null + one hypomorphic allele → partial GNPAT activity. "
                "Plasmalogens 5–25% of normal. Moderate rhizomelia, moderate ID. Survival 10–20y."
            ),
        },
        {
            "name": "Mild RCDP2 (Hypomorphic/Hypomorphic)",
            "pct": 20,
            "n": 8,
            "sex": "M/F equal",
            "onset_age": "6–36 months",
            "seizure_risk": "15–20%",
            "eeg": "Focal discharges, sparse multifocal spikes, may be near-normal",
            "mri": "Minimal white matter changes; myelination near-normal",
            "dha_supplement": True,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": (
                "Biallelic hypomorphic alleles → residual GNPAT activity. "
                "Plasmalogens 20–50% of normal. Minimal rhizomelia, mild-moderate ID, adult survival reported."
            ),
        },
    ]

    phenotypes = (
        [("Classic RCDP2", "Severe")] * 16
        + [("Intermediate RCDP2", "Intermediate")] * 16
        + [("Mild RCDP2", "Mild")] * 8
    )
    sexes = (["M"] * 20 + ["F"] * 20)
    random.shuffle(sexes)

    genotype_map = {
        "Classic RCDP2": ["null/null", "p.Arg290X/null", "frameshift/null", "del_exon3/null"],
        "Intermediate RCDP2": ["null/p.Gly267Arg", "frameshift/missense", "null/IVS5+1G>A", "splice/missense"],
        "Mild RCDP2": ["missense/missense", "p.Ala114Pro/missense", "p.Ser271Pro/missense"],
    }
    aed_map = {
        "Classic RCDP2": ["LEV+ACTH", "LEV", "LEV+CLB", "ACTH", None],
        "Intermediate RCDP2": ["LEV", "LEV+LTG", "LEV+CLB", None, None],
        "Mild RCDP2": ["LEV", "LEV+LTG", None, None, None],
    }
    response_map = {
        "Classic RCDP2": ["Drug-resistant", "Partially controlled", "Drug-resistant", "Partially controlled"],
        "Intermediate RCDP2": ["Controlled", "Partially controlled", "Drug-resistant", "Controlled"],
        "Mild RCDP2": ["Controlled", "Controlled", "Partially controlled"],
    }

    patients = []
    for i, (phen, _) in enumerate(phenotypes):
        has_sei = (
            random.random() < 0.60 if "Classic" in phen else
            random.random() < 0.35 if "Intermediate" in phen else
            random.random() < 0.18
        )
        aed_choices = aed_map[phen]
        primary_aed = random.choice(aed_choices) if has_sei else None
        resp = random.choice(response_map[phen]) if has_sei and primary_aed else None
        patients.append({
            "patient_id": f"RCDP2-{i+1:02d}",
            "phenotype": phen,
            "sex": sexes[i],
            "genotype": random.choice(genotype_map[phen]),
            "cataract": random.random() < (0.75 if "Classic" in phen else 0.62 if "Intermediate" in phen else 0.22),
            "ichthyosis": random.random() < (0.72 if "Classic" in phen else 0.58 if "Intermediate" in phen else 0.28),
            "has_seizures": has_sei,
            "primary_aed": primary_aed,
            "drug_response": resp,
            "phytanic_elevated": random.random() < 0.20,  # ~18-20% have mild elevation
            "on_dha": random.random() < (0.45 if "Classic" in phen else 0.38 if "Intermediate" in phen else 0.22),
            "on_alkylglycerol": random.random() < (0.35 if "Classic" in phen else 0.28 if "Intermediate" in phen else 0.15),
        })

    seizure_types = [
        {"type": "Infantile Spasms (Hypsarrhythmia)",
         "pct": 38, "eeg": "Hypsarrhythmia; modified-hypsarrhythmia; ACTH Level B by analogy with RCDP1"},
        {"type": "Focal Onset (Temporal/Occipital)",
         "pct": 30, "eeg": "Temporal or occipital focal spikes; secondary generalization"},
        {"type": "Myoclonic",
         "pct": 18, "eeg": "Generalized polyspike-wave; may have photosensitivity"},
        {"type": "Generalized Tonic-Clonic (GTCS)",
         "pct": 14, "eeg": "Generalized spike-wave; often evolving from focal onset"},
        {"type": "Status Epilepticus (SE)",
         "pct": 7, "eeg": "Electrographic SE in severe classic RCDP2; refractory in some"},
    ]

    triggers = [
        {"trigger": "Febrile illness", "pct": 60,
         "note": "Primary trigger; fever lowers seizure threshold in hypomyelinated brain"},
        {"trigger": "Sleep deprivation", "pct": 32,
         "note": "Common generalised trigger; more prominent in intermediate/mild RCDP2"},
        {"trigger": "Metabolic stress / intercurrent illness", "pct": 28,
         "note": "Plasmalogen depletion worsens neurological stability during illness"},
        {"trigger": "Fasting / pre-operative NPO", "pct": 18,
         "note": "Less severe than RCDP1 (phytanic usually normal) but peri-op glucose still prudent"},
        {"trigger": "Missed AED dose", "pct": 22,
         "note": "AED non-compliance; LEV withdrawal particularly significant"},
        {"trigger": "Dietary poor compliance (low DHA)", "pct": 15,
         "note": "Low DHA worsens neurological function; DHA supplementation may buffer this trigger"},
        {"trigger": "Anaesthesia / surgery", "pct": 12,
         "note": "Peri-op stress; ensure adequate DHA and maintain AED perioperatively"},
    ]

    monitoring = [
        "RBC plasmalogens (C16:0-DMA, C18:0-DMA): baseline, then every 12 months — PRIMARY biomarker",
        "Plasma phytanic acid: baseline (usually NORMAL in RCDP2); annually if mildly elevated",
        "VLCFA (C26:0) panel: baseline should be NORMAL; if elevated reconsider diagnosis (check PEX1/PEX6)",
        "DHA level annually; DHA supplementation if <2% of total fatty acids",
        "Plasma pristanic acid: NORMAL (confirm); elevated would suggest ZSD",
        "EEG: baseline at seizure onset; repeat 6-monthly or after AED change",
        "Ophthalmology: every 6 months (cataracts, VF monitoring; VGB contraindication check)",
        "Skeletal X-ray: annual in first 5 years (stippling regression, contracture progression)",
        "LFTs every 3 months if on VPA; pre-VPA POLG1 genotyping MANDATORY (CPIC A)",
        "Developmental assessment (Bayley/Griffiths): every 6 months in first 2 years",
        "Respiratory function (FVC): annually in classic/intermediate (phrenic weakness risk)",
        "Hearing screen annually (sensorineural hearing loss in some RCDP2 reports)",
        "RBC alkylglycerol response: if on supplementation, recheck plasmalogens at 3 and 12 months",
    ]

    thresholds = [
        {"parameter": "RBC plasmalogens", "threshold": "<50% of normal",
         "action": "Initiate alkylglycerol supplementation (batyl alcohol, experimental Level C); directly bypasses GNPAT block"},
        {"parameter": "Phytanic acid (plasma)", "threshold": ">10 μmol/L (note: usually NORMAL in RCDP2)",
         "action": "If elevated, consider phytol restriction empirically; confirm GNPAT diagnosis (not PEX7)"},
        {"parameter": "VLCFA C26:0", "threshold": "Elevated (>0.97 μg/mL)",
         "action": "Re-evaluate diagnosis — GNPAT/RCDP2 should be VLCFA NORMAL; if elevated, consider ZSD"},
        {"parameter": "LFT (ALT/AST)", "threshold": ">3× ULN on VPA",
         "action": "Discontinue VPA immediately; switch to LEV; liver biopsy if >5× ULN"},
        {"parameter": "Visual field (VGB monitoring)", "threshold": "Any VF loss",
         "action": "Stop VGB immediately; switch to LEV+CLB or LEV+LTG; document irreversibility"},
        {"parameter": "Infantile spasms response to ACTH", "threshold": "<2 weeks no response",
         "action": "Add LEV; consider CLB; avoid VGB (cataracts + VF); urgent neurology review"},
        {"parameter": "DHA (plasma)", "threshold": "<2% total FA",
         "action": "Initiate DHA 200 mg/day (infants); 500 mg/day (older children); recheck at 3 months"},
    ]

    lifecycle = [
        {"stage": "Neonatal (0–1 month)",
         "features": "Rhizomelia detected; stippled epiphyses on X-ray; congenital cataracts; ichthyosis; hypotonia. Biochemistry: plasmalogens severely low, phytanic usually NORMAL, VLCFA NORMAL.",
         "action": "Confirm GNPAT sequencing; start DHA supplementation; consider alkylglycerol if plasmalogens near-absent; ophthalmology urgent; phytol restriction only if phytanic elevated"},
        {"stage": "Early Infantile (1–6 months)",
         "features": "Infantile spasms onset (peak 3–5 months); hypsarrhythmia on EEG; profound hypotonia; feeding difficulties; cataracts progress",
         "action": "ACTH Level B; LEV first-line; avoid VGB (cataracts); DHA; alkylglycerol supplementation trial if available"},
        {"stage": "Late Infantile / Toddler (6–24 months)",
         "features": "Focal and myoclonic seizures; severe contractures; no ambulation in classic; moderate delay in intermediate",
         "action": "LEV ± CLB ± LTG; DHA supplementation ongoing; physiotherapy for contractures; wheelchair assessment; RBC plasmalogen monitoring"},
        {"stage": "Early Childhood (2–5 years)",
         "features": "Seizure burden may reduce spontaneously; kyphoscoliosis; chronic respiratory compromise in severe; cataract surgery assessment",
         "action": "Annual skeletal X-ray; respiratory spirometry; continue AED; annual plasmalogen + DHA; ophthalmology review"},
        {"stage": "School Age (5–10 years)",
         "features": "Classic RCDP2: usually deceased or profound disability; Intermediate: special education; Mild: mainstream with support; GTCS may emerge",
         "action": "Educational planning; annual epilepsy review; VPA avoidance maintained; LFT monitoring if any hepatotoxic AED"},
        {"stage": "Adolescent / Adult (≥10 years)",
         "features": "Mild RCDP2: adult survival possible; focal epilepsy manageable; DHA and plasmalogen supplementation lifelong; hearing loss may progress",
         "action": "Adult neurology transition; DHA lifelong; alkylglycerol supplementation if plasmalogens consistently low; reproductive counselling (AR 25% risk)"},
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "SV2A modulator — FIRST-LINE all RCDP2 forms",
            "evidence": "Level A (standard first-line for all peroxisomal epilepsies)",
            "dose": "Paediatric: 20–60 mg/kg/day div q12h; Adult: 500–3000 mg/day",
            "moa": "Binds SV2A synaptic vesicle protein → reduces neurotransmitter release",
            "monitoring": "Behaviour (irritability); no hepatic monitoring required",
            "ci": None,
        },
        {
            "drug": "ACTH (Tetracosactide/Synacthen)",
            "class": "Corticotropin — Level B (infantile spasms)",
            "evidence": "Level B — extrapolated from RCDP1 (rarity of RCDP2 limits RCT data); used for structural-metabolic IS",
            "dose": "0.5–1.0 mg/day IM for 2 weeks then taper; or short Dexamethasone protocol",
            "moa": "Anti-epileptic via melanocortin MC2R; suppresses CLIP; reduces CRH",
            "monitoring": "BP, glucose, infection risk during immunosuppression",
            "ci": "Active untreated infection",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "Benzodiazepine (long-acting) — Level C adjunct",
            "evidence": "Level C — adjunct for focal and GTCS in peroxisomal epilepsies",
            "dose": "0.1–1.0 mg/kg/day; max 40 mg/day",
            "moa": "Positive allosteric modulator GABA-A at benzodiazepine site",
            "monitoring": "Sedation; tolerance (may occur within months of chronic use)",
            "ci": None,
        },
        {
            "drug": "Lamotrigine (LTG)",
            "class": "Sodium channel blocker — Level C adjunct",
            "evidence": "Level C — adjunct in intermediate/mild forms; slow titration mandatory",
            "dose": "0.15–0.3 mg/kg/day titrating over 8 weeks to 1–3 mg/kg/day (without VPA)",
            "moa": "Blocks voltage-gated Na+ channels; reduces glutamate release",
            "monitoring": "Rash (SJS risk, slow titration essential); VPA doubles LTG levels",
            "ci": "Rapid titration; VPA co-administration (doubles LTG levels → toxicity)",
        },
        {
            "drug": "DHA Supplementation (Docosahexaenoic Acid)",
            "class": "Polyunsaturated fatty acid — Level C",
            "evidence": "Level C — replaces peroxisomally synthesised DHA; clinical benefit in neural function uncertain but plausible",
            "dose": "200 mg/day (infants); 500–1000 mg/day (older children/adults)",
            "moa": "Replaces DHA deficit from impaired peroxisomal DHA retroconversion/elongation",
            "monitoring": "Plasma DHA annually; hepatic lipid if high dose",
            "ci": None,
        },
        {
            "drug": "Alkylglycerol Precursor Supplementation (Batyl/Chimyl/Selachyl Alcohol)",
            "class": "Plasmalogen precursor — Experimental Level C (MOST DIRECT bypass in RCDP2)",
            "evidence": "Level C — in RCDP2, alkylglycerols bypass the GNPAT (Step 1) block, providing the most pharmacologically direct substrate bypass. Raises RBC plasmalogens in animal models.",
            "dose": "50–100 mg/kg/day (research use only); no approved formulation",
            "moa": "Alkylglycerols enter as 1-alkyl-glycerophosphocholine precursors, bypassing blocked GNPAT Step 1 → directly feed Step 2 (AGPS) → produce plasmalogen precursors",
            "monitoring": "RBC plasmalogens at 3 and 12 months; ensure formulation is phytol-free",
            "ci": "Formulations may contain trace phytol impurities; clinical trial preferred",
        },
    ]

    contraindications = [
        {
            "drug": "Vigabatrin (VGB)",
            "level": "HIGH RISK — Cataracts + VF Loss (not absolute CI but HIGH RISK)",
            "reason": (
                "RCDP2: cataracts present in 60-70% + VGB irreversible visual field constriction = "
                "additive visual impairment. Monthly VF/VEP monitoring mandatory if VGB considered. "
                "Same risk profile as RCDP1 — less severe than ZSD (where retinopathy is universal), "
                "but cataracts are the primary additive risk. Avoid as first-line for IS in RCDP2."
            ),
            "alternative": "ACTH (Level B IS) + LEV; CLB adjunct; avoid VGB if cataracts present",
        },
        {
            "drug": "Valproate (VPA)",
            "level": "RELATIVE CI — Hepatotoxicity (POLG1 MANDATORY)",
            "reason": (
                "Hepatotoxicity risk in chronic metabolic disease. NOTE: Unlike RCDP1, phytanic acid "
                "is usually NORMAL in RCDP2, so the phytanic-driven liver stress is less of a contributor, "
                "but hepatotoxicity risk from VPA itself (mitochondrial / POLG1) persists. "
                "POLG1 sequencing MANDATORY before VPA (CPIC Grade A). LFTs q3 months if VPA used."
            ),
            "alternative": "LEV (first-line); CLB or LTG as adjuncts; never start VPA before POLG1 clearance",
        },
        {
            "drug": "Fasting / Pre-operative NPO",
            "level": "CAUTION (less severe than RCDP1 — phytanic usually normal)",
            "reason": (
                "In RCDP1, fasting mobilises adipose phytanic stores → acute neurotoxicity. "
                "In RCDP2, phytanic is usually NORMAL, so this risk is substantially reduced. "
                "However, overall metabolic stress during fasting can destabilise the CNS via "
                "plasmalogen depletion worsening. IV dextrose perioperatively still advisable."
            ),
            "alternative": "Maintain DHA supplementation perioperatively; IV dextrose as standard peri-op protocol",
        },
        {
            "drug": "Phenytoin (PHT) / Fosphenytoin (IV)",
            "level": "CAUTION — NOT absolute CI (unlike ABCD1 or NPC1)",
            "reason": (
                "No adrenal insufficiency in RCDP2 → no cortisol drop via CYP3A4 → PHT NOT contraindicated "
                "by adrenal mechanism. IV fosphenytoin acceptable in SE if IV LEV fails. "
                "Monitor for peripheral neuropathy aggravation (less data than RCDP1 due to rarity)."
            ),
            "alternative": "IV LEV preferred in SE; PHT acceptable if IV LEV fails and benefit outweighs risk",
        },
        {
            "drug": "Typical Antipsychotics",
            "level": "RELATIVE CI — Extrapyramidal (EPS) Risk",
            "reason": (
                "Hypomyelination in RCDP2 increases EPS susceptibility to typical antipsychotics. "
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
            "GNPAT = Glyceronephosphate O-Acyltransferase (DHAPAT / dihydroxyacetone phosphate acyltransferase): 680-aa peroxisomal matrix enzyme using PTS1 import (C-terminal -SKF-). Step 1 of plasmalogen biosynthesis.",
            "RCDP2 enzyme vs RCDP1 receptor: PEX7/RCDP1 = PTS2 receptor deficiency (all PTS2 cargo proteins fail including PHYH). GNPAT/RCDP2 = enzyme LOF (PEX7 intact, PHYH imported normally). Critical biochemical consequence: phytanic acid NORMAL in RCDP2.",
            "Plasmalogen biosynthesis — 2-step peroxisomal pathway: Step 1 GNPAT: DHAP + acyl-CoA → 1-acyl-DHAP. Step 2 AGPS: 1-acyl-DHAP + fatty alcohol → 1-alkyl-DHAP. RCDP2 blocks Step 1. RCDP3 blocks Step 2. RCDP1 blocks IMPORT of both Step 2 enzyme (AGPS) and PHYH.",
            "Phytanic acid in RCDP2: NORMAL (distinguish from RCDP1 where phytanic is ELEVATED). PHYH (phytanoyl-CoA hydroxylase) is PTS2-targeted; in RCDP2, PEX7 is intact → PHYH is imported normally → phytanic catabolism intact. Only ~18-20% of RCDP2 patients have mild phytanic elevation.",
            "Alkylglycerol pharmacology in RCDP2: MOST DIRECT bypass. Batyl/chimyl/selachyl alcohol provide pre-formed 1-alkyl-glycerophosphocholine precursors downstream of Step 1 (GNPAT block). This makes RCDP2 the strongest rationale for alkylglycerol supplementation (Step 1 bypass). Compare RCDP3 where alkylglycerols bypass Step 2.",
            "VLCFA in RCDP2: NORMAL — C26:0 beta-oxidation uses PTS1 (ACOX1, enzyme); both PEX7 (intact) and GNPAT (LOF) are irrelevant to VLCFA pathway. If VLCFA elevated → suspect ZSD (PEX1/PEX6).",
            "RBC plasmalogens — severely low/absent in RCDP2. Primary diagnostic biomarker (C16:0-DMA and C18:0-DMA dimethylacetals). Measures erythrocyte ethanolamine plasmalogen content; reflects whole-body synthesis.",
            "VGB in RCDP2: HIGH RISK (same mechanism as RCDP1). Cataracts 60-70% + VGB irreversible visual field constriction = additive blindness risk. Not absolute CI (unlike ZSD retinopathy), but monthly VF/VEP mandatory if used.",
            "VPA in RCDP2: RELATIVE CI — hepatotoxicity. POLG1 MANDATORY. Phytanic is NORMAL in RCDP2 (less phytanic-driven hepatic stress than RCDP1), but POLG1-mediated mitochondrial hepatotoxicity risk persists. Absolute CPIC Grade A requirement: test POLG1 before VPA.",
            "PHT/CBZ/OXC in RCDP2: CAN BE USED — no adrenal insufficiency (unlike ABCD1 where PHT = ABSOLUTE CI via CYP3A4 cortisol drop). Same safe profile as RCDP1 for enzyme-inducing AEDs with respect to adrenal risk.",
            "No ERT in RCDP2: GNPAT is a peroxisomal matrix enzyme (not secreted; not lysosomal). ERT can only replace secreted or endocytosable enzymes (Gaucher, Fabry, MPS, Pompe). No therapeutic equivalent for cytoplasmic/peroxisomal enzymes.",
            "No HSCT in RCDP2: Pathology is hypomyelination (plasmalogen deficiency impairs myelin synthesis), not neuroinflammation. HSCT arrests inflammatory demyelination (ABCD1-CCALD, Krabbe) — not relevant for hypomyelination.",
            "No founder mutation in RCDP2: Unlike RCDP1 (L292X ~50% European alleles), GNPAT variants are private or rare — full gene sequencing required; panel testing including GNPAT mandatory in all RCDP workup.",
            "RCDP2 vs RCDP3 (AGPS): Biochemically near-identical (both: plasmalogens severely low, phytanic normal, VLCFA normal). Alkylglycerols bypass BOTH; RCDP2 at Step 1, RCDP3 at Step 2. Distinguished by gene sequencing only.",
            "Phytol-restricted diet in RCDP2: LESS CRITICAL than RCDP1. Dietary phytol only matters when phytanic elevated (primary via PHYH LOF in RCDP1; not primary in RCDP2). Empirically sometimes recommended if phytanic mildly elevated in RCDP2.",
        ],
        "diagnostic_algorithm": [
            "Step 1: Clinical suspicion — rhizomelia + stippled epiphyses (neonate) OR congenital cataracts + ichthyosis + hypotonia (infant) OR unexplained epilepsy + intellectual disability + skeletal abnormality.",
            "Step 2: Biochemistry — PLASMA VLCFA panel (C26:0, C26:0/C22:0, C24:0/C22:0): must be NORMAL. If elevated → ZSD (PEX1/PEX6) not RCDP.",
            "Step 3: RBC PLASMALOGENS (C16:0-DMA, C18:0-DMA dimethylacetals): SEVERELY LOW or absent in RCDP2. Confirms ether-lipid defect.",
            "Step 4: PLASMA PHYTANIC ACID: CHECK. In RCDP2 this is USUALLY NORMAL (unlike RCDP1 where elevated). Mild elevation may occur in some RCDP2 patients.",
            "Step 5: PRISTANIC ACID plasma — NORMAL in RCDP2 (same as RCDP1). Elevated would suggest ZSD.",
            "Step 6: PIPECOLIC ACID plasma — NORMAL in RCDP2. Elevated confirms ZSD.",
            "Step 7: Skeletal X-ray — rhizomelia (humerus shortening), stippled epiphyses (neonatal), progressive contractures.",
            "Step 8: Ophthalmology — cataracts (60-70%); visual field assessment; ERG if VGB considered.",
            "Step 9: MRI brain — hypomyelination (reduced white matter T2 signal), cortical atrophy in severe.",
            "Step 10: GENE SEQUENCING panel: PEX7 first (90% of RCDP). If PEX7 negative → GNPAT (RCDP2) and AGPS (RCDP3) in parallel. GNPAT has no founder mutation — full exon sequencing required.",
            "Step 11: POLG1 genotyping MANDATORY before any VPA consideration (CPIC Grade A).",
            "Step 12: Confirm GNPAT diagnosis; institute DHA supplementation; initiate alkylglycerol supplementation if plasmalogens near-absent; contact specialist (Kennedy Krieger, Amsterdam, Manchester).",
        ],
        "pharmacological_distinctions": [
            "LEV — FIRST-LINE all RCDP2 forms (no peroxisomal interactions, no adrenal mechanism, safe).",
            "ACTH — Level B infantile spasms (extrapolated from RCDP1; very limited RCDP2-specific RCT data). Preferred over VGB as first-line IS therapy.",
            "VGB — HIGH RISK in RCDP2 (cataracts 60-70% + VF loss = additive). NOT absolute CI (unlike ZSD retinopathy universal/severe). Monthly VF/VEP monitoring if used.",
            "VPA — RELATIVE CI (hepatotoxicity). POLG1 MANDATORY. Phytanic NORMAL in RCDP2 reduces phytanic-hepatic stress (vs RCDP1) but POLG1/mitochondrial hepatotoxicity risk persists. LFT q3 months if VPA used.",
            "PHT/CBZ/OXC — CAN BE USED (no adrenal insufficiency in RCDP2; no cortisol drop; no adrenal crisis). Same safe adrenal profile as RCDP1. Contrast with ABCD1 = ABSOLUTE CI.",
            "Fosphenytoin IV — acceptable in SE when IV LEV fails (no adrenal mechanism). Unlike ABCD1 (adrenal crisis) or NPC (disease worsening).",
            "CLB/benzodiazepines — safe adjuncts; MDZ nasal/buccal for acute SE; CLB for chronic adjunct.",
            "Typical antipsychotics — HIGH RISK EPS (hypomyelination increases susceptibility). Atypicals preferred.",
            "DHA supplementation — Level C; replaces peroxisomal DHA synthesis deficit; 200 mg/day infants.",
            "Alkylglycerol supplementation — Experimental Level C; MOST DIRECT substrate bypass for RCDP2 (bypasses blocked Step 1 GNPAT). Primary rationale for supplementation in RCDP2.",
            "Phytol-restricted diet — LESS CRITICAL than RCDP1 (phytanic usually normal). Used empirically only if phytanic mildly elevated; not routine in RCDP2 unless phytanic documented elevated.",
            "No ERT, No HSCT, No gene therapy (2026) — no approved biological disease-modifying therapy; alkylglycerol + DHA are experimental/supportive only.",
        ],
        "differential_diagnosis": [
            {"condition": "RCDP1 (PEX7)", "distinction": "Biochemically near-identical but phytanic ELEVATED in RCDP1 (PHYH cannot be imported); phytanic NORMAL in RCDP2 (PEX7 intact, PHYH imported). Plasmalogens low in both. Distinguished by gene sequencing PEX7 vs GNPAT."},
            {"condition": "RCDP3 (AGPS)", "distinction": "Biochemically near-identical to RCDP2 (both have normal phytanic, severely low plasmalogens, normal VLCFA). RCDP3 blocks Step 2 (AGPS); RCDP2 blocks Step 1 (GNPAT). Distinguished only by gene sequencing GNPAT vs AGPS."},
            {"condition": "ZSD (PEX1/PEX6)", "distinction": "VLCFA HIGH in ZSD; VLCFA NORMAL in RCDP2. ZSD: ALL peroxisomal functions impaired. RCDP2: only plasmalogen pathway at Step 1. No rhizomelia/stippling in ZSD (cortical migration defects instead)."},
            {"condition": "Adult Refsum Disease (PHYH)", "distinction": "Phytanic ELEVATED in Refsum (PHYH deficiency); phytanic NORMAL in RCDP2. Plasmalogens NORMAL in Refsum (plasmalogen synthesis intact). Adult onset, peripheral neuropathy dominant. No rhizomelia."},
            {"condition": "CDPX2 / Conradi-Hunermann (EBP)", "distinction": "X-linked dominant stippling; ALL peroxisomal markers NORMAL (VLCFA, plasmalogens, phytanic, pristanic). Asymmetric ichthyosis (Blaschko lines). Cholesterol pathway (sterol isomerase), not peroxisomal."},
            {"condition": "ABCD1 (X-ALD)", "distinction": "VLCFA HIGH; plasmalogens NORMAL (not low). No rhizomelia/stippling. Adrenal insufficiency 71% males. X-linked. PHT = ABSOLUTE CI (adrenal crisis). HSCT indicated (CCALD)."},
            {"condition": "Non-peroxisomal skeletal dysplasias", "distinction": "All peroxisomal markers NORMAL (VLCFA, plasmalogens, phytanic). FGFR3 (achondroplasia), RMRP (CHH) or other skeletal gene mutations. Rhizomelia may be present but no cataracts/ichthyosis/stippling combination."},
        ],
        "standards": [
            "de Vet EC, Ijlst L, Oostheim W, et al. Ether lipid biosynthesis: alkyl-dihydroxyacetonephosphate synthase. J Biol Chem. 1998;273:10296–10301.",
            "Honsho M, Taguchi R, Fujiki Y. Plasmalogen deficiency dysregulates HIF-1 and mTOR signalling in an age-dependent manner. Sci Rep. 2019.",
            "Braverman NE, Raymond GV, Rizzo WB, et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Am J Med Genet C. 2016.",
            "Wanders RJA, Waterham HR. Biochemistry of mammalian peroxisomes revisited. Annu Rev Biochem. 2006.",
            "Waterham HR, Ferdinandusse S, Wanders RJA. Human disorders of peroxisome metabolism. Biochim Biophys Acta. 2016.",
            "CPIC Guideline — Valproic Acid and POLG1 (Grade A). cpicpgx.org.",
            "Braverman NE et al. Rhizomelic Chondrodysplasia Punctata Type 1. GeneReviews. NCBI Bookshelf. 2015.",
        ],
    }
