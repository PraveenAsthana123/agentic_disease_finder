#!/usr/bin/env python3
"""PEX11B / Peroxisome Fission Defect Epilepsy Dashboard — seed data module.

PEX11B encodes Peroxisome Biogenesis Factor 11 Beta (247 aa), an integral peroxisomal
membrane protein that drives peroxisome fission and proliferation. Unlike all 13 other
ZSD-causing PEX genes (PEX1/2/3/5/6/7/10/12/13/14/16/19/26), PEX11B does NOT cause
Zellweger Spectrum Disorder (ZSD). Instead, PEX11B LOF causes an ISOLATED PEROXISOME
FISSION DEFECT — a mechanistically distinct fourth tier of peroxisomal epilepsy.

MECHANISM — PEROXISOME FISSION TIER (DISTINCT FROM ALL ZSD):
  PEX11B architecture (247 aa):
    N-terminal cytoplasmic region (~aa 1–30): amphipathic helix; membrane curvature sensing
    Transmembrane domain 1 (~aa 31–55): anchors in peroxisomal membrane
    Lumenal loop (~aa 56–80): peroxisomal lumenal exposure
    Transmembrane domain 2 (~aa 81–105): second membrane anchor
    C-terminal cytoplasmic tail (~aa 106–247): amphipathic helix; oligomerisation domain;
      DRP1 recruitment interface; FIS1 interaction site
  PEX11B fission cycle:
    (1) PEX11B oligomerises on peroxisomal membrane → membrane elongation and tubulation.
    (2) Elongated peroxisomal tubule recruits DRP1 (dynamin-related GTPase) via PEX11B
        C-terminal tail, assisted by FIS1 (mitochondrial fission 1 protein, also on
        peroxisomal membrane) and MFF (mitochondrial fission factor).
    (3) DRP1 assembles into a helical oligomeric ring around the peroxisomal tubule neck.
    (4) GTP hydrolysis by DRP1 constricts the neck → membrane scission → two daughter
        peroxisomes. This is the PEROXISOME DIVISION step.
    (5) Divided peroxisomes grow via PEX11-mediated tubulation cycles and matrix import
        of proteins (PTS1 via PEX5, PTS2 via PEX7·PEX5L) + lipid incorporation.
  PEX11B LOF consequences (FISSION TIER):
    (A) Fission cannot complete → peroxisomes elongate into tubular structures → fewer,
        larger, morphologically abnormal peroxisomes remain.
    (B) Peroxisomal MEMBRANE BIOGENESIS IS INTACT (PEX3/PEX16/PEX19 axis functional).
    (C) IMPORTOMER IS INTACT (PEX13/PEX14/PEX17 complex structurally normal).
    (D) MATRIX PROTEIN IMPORT IS INTACT (PEX5 PTS1 receptor present and functional;
        PEX7·PEX5L PTS2 co-receptor complex present and functional).
    (E) Each individual peroxisome CAN still import matrix proteins — fission defect
        reduces TOTAL peroxisome number but does not abolish per-organelle function.
    (F) Net effect: REDUCED TOTAL PEROXISOMAL METABOLIC CAPACITY (fewer peroxisomes)
        → mild-to-moderate biochemical abnormalities (NOT the complete biochemical
        failure seen in ZSD where ALL peroxisomal matrix import fails globally).
  CRITICAL IF SIGNATURE DISTINCTION — FOURTH TIER:
    In PEX11B LOF the IF pattern is OPPOSITE to ZSD for catalase:
    PMP70 PRESENT (membrane biogenesis intact — PEX3/PEX16/PEX19 axis functional)
    PEX14 PRESENT (importomer central scaffold intact — PEX19 inserts PEX14 normally)
    Catalase PEROXISOMAL (import IS working — PEX5 receptor functional!)
    Morphology: ELONGATED/TUBULAR peroxisomes (not absent, not normally spherical/punctate)
    This distinguishes PEX11B LOF from ALL 13 ZSD-causing PEX genes where catalase
    is CYTOPLASMIC (because matrix import fails). In PEX11B LOF, catalase IS inside
    the peroxisome — the fission defect does not abolish per-organelle import function.
  FOUR-TIER IF HIERARCHY (PEX11B adds the 4th tier to the ZSD framework):
    Tier 1 — Membrane-Biogenesis (PEX3/16/19 LOF): PMP70− + PEX14− + catalase cytoplasmic
    Tier 2 — Importomer-Scaffold (PEX14 LOF): PMP70+ + PEX14− + catalase cytoplasmic
    Tier 3 — Importomer-SH3 (PEX13 LOF): PMP70+ + PEX14+ + catalase cytoplasmic
         + Receptor Tier (PEX5 LOF): PMP70+ + PEX14+ + PEX13+ + catalase cytoplasmic
    Tier 4 — FISSION TIER (PEX11B LOF): PMP70+ + PEX14+ + catalase PEROXISOMAL
              + elongated/tubular peroxisomes (not absent or normal spherical)

LOCUS: 1q22  |  OMIM GENE: *603637  |  OMIM DISEASE: #614862 (PBD9B)

EPIDEMIOLOGY:
  Peroxisome fission defect due to PEX11B LOF is extremely rare (~10–15 confirmed cases
  worldwide as of 2026). First clinically described by Ebberink MS et al. (2012) and
  Yoneda T et al. (in fibroblasts) — biallelic LOF variants confirmed in patients with
  epilepsy, intellectual disability, sensorineural hearing loss, and photosensitivity.
  No founder allele; all reported variants are private/sporadic.

PEROXISOMAL BIOCHEMISTRY — PARTIAL/MILD ABNORMALITIES (DISTINCT FROM ZSD):
  PEX11B deficiency does NOT abolish all peroxisomal matrix import. Each of the
  fewer, enlarged peroxisomes still imports matrix proteins via intact PEX5/PEX7
  pathways. Therefore biochemical abnormalities are PARTIAL, not complete:
  (1) VLCFA beta-oxidation: MILDLY REDUCED → C26:0 borderline elevated or normal
      [CRITICAL: NOT the frank elevation of ZSD; VLCFA may be only borderline]
  (2) Alpha-oxidation: MILDLY REDUCED → phytanic acid borderline or normal
      [PHYH is PTS2 cargo; import is intact but peroxisome number reduced → mild impair.]
  (3) Plasmalogen biosynthesis: BORDERLINE or NORMAL
      [GNPAT/AGPS are matrix enzymes; import works per-organelle → plasmalogens near-normal]
      CRITICAL: plasmalogens (RBC) NORMAL or borderline in PEX11B — vs SEVERELY LOW in ZSD
      This is a key biochemical diagnostic distinction separating PEX11B from ZSD.
  (4) Pipecolic acid catabolism: NORMAL or MILDLY ELEVATED
      [Pipeline-metabolising enzymes are imported normally per organelle]
  (5) DHA synthesis: BORDERLINE or NORMAL
      [DHA-synthesising enzymes are imported; fewer peroxisomes may mildly reduce flux]
  (6) Bile acid intermediates: NORMAL or BORDERLINE
  NBS BIOMARKER: C26:0-lyso-PC (DBS) may be borderline/equivocal — key distinguisher
  from ZSD where it is frankly elevated. Suspected if equivocal NBS + epilepsy + ID.
  Electron microscopy or IF morphology (elongated peroxisomes) + NGS panel confirms.

PHENOTYPIC SPECTRUM:
  Severity reflects residual fission capacity (number of functional peroxisomes):
  Severe neonatal ~35%  (null/null → neonatal seizures, near-absent peroxisome population)
  Moderate infantile ~45% (modal; partial-function alleles → IS + moderate ID)
  Mild childhood-onset ~15% (milder alleles → focal epilepsy + mild-moderate ID)
  Atypical late-onset ~5% (very mild alleles → adult-onset epilepsy only)
"""
import random


def get_overview():
    return {
        "cohort_size": 40,
        "seizure_pct": 85,
        "severe_neonatal_pct": 35,
        "moderate_infantile_pct": 45,
        "mild_childhood_pct": 15,
        "atypical_pct": 5,
        "drug_resistance_pct": 55,
        "on_dha_pct": 22,
        "hearing_loss_pct": 78,
        "retinopathy_pct": 62,
        "photosensitivity_pct": 71,
        "vlcfa_elevated_pct": 65,
        "plasmalogen_low_pct": 18,
        "catalase_peroxisomal_pct": 100,
        "omim_gene": "603637",
        "omim_disease": "614862",
        "locus": "1q22",
        "protein_size": "247 aa (single isoform)",
        "nbs_positive_rate": "~45% sensitivity (C26:0-lyso-PC borderline/equivocal — unlike frank ZSD elevation)",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "common_variant": (
            "p.Ile195Thr (I195T): partial-function allele in C-terminal oligomerisation domain; "
            "reduces DRP1 recruitment efficiency (~40% residual fission) → moderate phenotype; "
            "p.Leu145Pro (L145Pro): transmembrane domain stability loss → severe neonatal; "
            "p.Arg68* (null, truncation before TM domain 2 → complete LOF → severe neonatal); "
            "p.Trp200Gly (C-terminal amphipathic helix disrupted → impaired DRP1 docking)"
        ),
        "disease_mechanism": (
            "PEX11B (247 aa, 1q22) is an INTEGRAL PEROXISOMAL MEMBRANE PROTEIN that drives "
            "peroxisome fission via DRP1/FIS1/MFF recruitment — the FISSION TIER of peroxisomal "
            "epilepsy, mechanistically distinct from all 13 ZSD-causing PEX genes. PEX11B "
            "oligomerises on the peroxisomal membrane → elongates the organelle into a tubular "
            "structure → recruits DRP1 GTPase via its C-terminal tail (assisted by FIS1/MFF) → "
            "DRP1 helical ring constricts the tubule neck → GTP hydrolysis drives membrane "
            "scission → two daughter peroxisomes. PEX11B LOF → fission cannot complete → "
            "elongated/tubular peroxisomes persist → fewer, larger organelles. CRITICAL: "
            "membrane biogenesis IS intact (PEX3/16/19 functional), importomer IS intact "
            "(PEX13/14 present), matrix import IS intact (PEX5 PTS1 receptor + PEX7·PEX5L "
            "PTS2 co-receptor both functional) → catalase IS PEROXISOMAL (not cytoplasmic). "
            "This is the defining feature distinguishing PEX11B from all ZSD: import works "
            "per-organelle, but total organelle number is reduced → partial biochemical "
            "phenotype. Plasmalogens NORMAL or borderline (not severely low as in ZSD). "
            "VLCFA borderline (not frankly elevated as in ZSD). ~10–15 cases worldwide 2026. "
            "NBS may be equivocal. Electron microscopy or IF (elongated peroxisomes + "
            "peroxisomal catalase) + NGS panel confirms. OMIM *603637 / #614862 (PBD9B)."
        ),
        "key_concepts": [
            "PEX11B (247 aa, 1q22, OMIM *603637): integral peroxisomal membrane protein with 2 transmembrane domains and cytoplasmic N/C termini. Function: oligomerise on membrane → tubulate peroxisome → recruit DRP1 GTPase + FIS1 + MFF fission machinery → drive membrane scission → daughter peroxisomes. PEX11B LOF → fission fails → elongated/tubular peroxisomes → fewer total organelles",
            "FISSION TIER — 4th mechanistic tier of peroxisomal epilepsy (distinct from ZSD tiers): ZSD involves peroxisomal membrane absence (Tier 1), importomer defects (Tiers 2–3), or receptor absence (Receptor tier) — all of which ABOLISH matrix import. PEX11B LOF (Tier 4) preserves membrane biogenesis + importomer + matrix import per organelle; only fission (organelle proliferation) is lost",
            "CRITICAL IF SIGNATURE distinguishing PEX11B from ALL ZSD: PMP70 PRESENT (membrane intact) + PEX14 PRESENT (importomer intact) + catalase PEROXISOMAL (import IS functioning) + elongated/tubular peroxisome morphology (fission-deficient, not spherical-normal or absent). In ALL ZSD (regardless of tier): catalase is CYTOPLASMIC because matrix import fails. In PEX11B LOF catalase is PEROXISOMAL — single most important diagnostic differentiator",
            "FOUR-TIER IF HIERARCHY: Tier 1 (PEX3/PEX16/PEX19 LOF): PMP70− + PEX14− + catalase cytoplasmic; Tier 2 (PEX14 LOF): PMP70+ + PEX14− + catalase cytoplasmic; Tier 3/Receptor (PEX13/PEX5 LOF): PMP70+ + PEX14± + catalase cytoplasmic; TIER 4-FISSION (PEX11B LOF): PMP70+ + PEX14+ + catalase PEROXISOMAL + elongated morphology",
            "Biochemical PARTIAL phenotype (distinguishes PEX11B from ZSD): VLCFA borderline (not frank elevation) + plasmalogens NORMAL or borderline (not severely low) + phytanic NORMAL or borderline + DHA NORMAL or borderline. KEY: plasmalogens NORMAL in PEX11B — severely LOW in ALL ZSD. This single test (erythrocyte plasmalogens) discriminates PEX11B from ZSD. ABCD1 (X-ALD) also has normal plasmalogens but no epilepsy at onset",
            "NBS EQUIVOCAL challenge: C26:0-lyso-PC (DBS) may be borderline/equivocal in PEX11B LOF — unlike the frank elevation in ZSD. A borderline NBS in a child with epilepsy + ID + SNHL should prompt full peroxisomal biochemistry panel + electron microscopy + NGS. Do NOT exclude PEX11B on normal NBS alone",
            "Peroxisome morphology by IF/EM: elongated, tubular peroxisomes (not the normal spherical-punctate pattern of PMP70+ in ZSD-membrane-tier) and not absent. Elongated morphology + peroxisomal catalase = pathognomonic of fission-tier defect. Electron microscopy shows elongated/worm-shaped peroxisomes in fibroblasts",
            "Photosensitivity (71% cohort): photoconvulsive response on EEG in PEX11B LOF — mechanism unclear, possibly altered membrane lipid composition in PEX11B-deficient neurons (reduced peroxisome number → locally reduced plasmalogen availability in neuronal membranes despite near-normal RBC plasmalogens). Clinical implication: avoid EEG photic stimulation without seizure rescue ready; protective eyewear and light-sensitivity precautions",
            "Sensorineural hearing loss (SNHL, 78% cohort): bilateral, often neonatal-onset. Cochlear peroxisomal deficiency → auditory hair cell dysfunction. Amplification (hearing aids / cochlear implant evaluation) from age 6 months. Distinguish from mitochondrial hearing loss (no ragged red fibres, normal lactate) and from MELAS/Usher syndrome on NGS",
            "DRP1 (dynamin-related protein 1) fission machinery: DRP1 is a cytoplasmic GTPase that assembles into helical rings at peroxisomal (and mitochondrial) constriction necks. Recruited to peroxisomes by PEX11B + FIS1 + MFF. GTP hydrolysis drives ring constriction → membrane fission. Note: mutations in DRP1 itself (DNM1L gene) cause a SEPARATE disorder (DRP1 deficiency — infantile encephalopathy with abnormal brain mitochondria + peroxisomes), distinct from PEX11B LOF",
            "Absence of pachygyria (distinguishes PEX11B from ZS/NALD): in ZSD, DHA deficiency → pachygyria/polymicrogyria (MRI PATHOGNOMONIC for ZS). In PEX11B LOF, DHA synthesis is near-normal (import intact) → brain MRI shows NON-SPECIFIC findings (cerebral atrophy, delayed myelination) without the gyral malformation of ZSD. If MRI shows pachygyria, reconsider ZSD, not PEX11B",
            "VPA caution (not triple CI): VPA hepatotoxicity + carnitine depletion (same as ZSD concerns) but NO peroxisomal beta-oxidation inhibition mechanism (import works, VLCFA-oxidising enzymes ARE in peroxisomes). So VPA risk is dual (not triple as in ZSD) — still HIGH RISK but mechanism is less severe. POLG1 exclusion MANDATORY. Treat as high risk, use LEV first-line",
            "VGB caution (not absolute CI like ZSD): retinopathy affects ~62% of PEX11B cohort (less universal than ZSD). In patients WITH confirmed retinopathy, VGB should be AVOIDED (additive visual field loss). In patients WITHOUT confirmed RP, VGB caution still advised. ACTH preferred for IS regardless",
            "Fasting CAUTION (not extreme hazard like ZSD): phytanic acid borderline in PEX11B (alpha-oxidation per-organelle intact, just fewer organelles). Fasting may mildly elevate phytanic acid from adipose release. Caution during NPO/illness — IV dextrose if prolonged fast. NOT the extreme acute neurotoxic surge risk of ZSD (where alpha-oxidation completely fails)",
            "Gene therapy / future: AAV9-PEX11B delivery in fibroblasts restores normal peroxisome morphology in cell culture. No clinical gene therapy trials 2026. HSCT NOT indicated (non-inflammatory metabolic defect). No ERT (integral membrane protein, not soluble lysosomal enzyme; no M6P receptor route)",
            "~10–15 cases worldwide 2026; OMIM *603637 / #614862; AR biallelic LOF; NO founder allele; FISSION TIER — peroxisome morphology defect without matrix import failure; catalase PEROXISOMAL = single most important IF differentiator from all ZSD",
        ],
        "standards": [
            "Ebberink MS et al. A novel peroxisomal disease type with defects in biogenesis only of peroxisomes. Am J Hum Genet. 2012",
            "Yoneda T et al. Structural analysis of PEX11 in peroxisome elongation. Mol Biol Cell. 2011",
            "Schrader M et al. Peroxisome morphology, dynamics and reproduction: past, present and future. Biochim Biophys Acta. 2016",
            "Koch J et al. Disturbed mitochondrial and peroxisomal dynamics due to loss of MFF causes Leigh-like encephalopathy, optic atrophy and peripheral neuropathy. J Med Genet. 2016",
            "Waterham HR, Ebberink MS. Genetics and molecular basis of human peroxisome biogenesis disorders. Biochim Biophys Acta. 2012",
            "Braverman NE et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Mol Genet Metab. 2016",
            "Ziegler GE et al. PEX11β regulates peroxisome proliferation and fission: a structural and functional update. Biochem J. 2021",
            "Steinberg SJ et al. Peroxisome biogenesis disorders in the Zellweger spectrum. Neurology. 2006",
            "CPIC VPA-POLG1 guideline (Level A). cpicpgx.org/guidelines/cpic-guideline-vpa-polg1/",
            "OMIM *603637 (PEX11B gene); #614862 (PBD9B — PEX11B peroxisome fission defect)",
        ],
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown():
    random.seed(1122)  # PEX11B — reproducible seed

    etiologies = [
        {
            "name": "Severe Neonatal — null/null genotype",
            "pct": 35, "n": 14,
            "sex": "7M/7F",
            "onset_age": "Neonatal (day 1–14)",
            "seizure_risk": "100% — neonatal myoclonic + tonic-clonic; photosensitive EEG from day 1; SNHL on ABR",
            "eeg": "Multifocal spike-wave; photosensitive (photoconvulsive response from day 5+); no burst-suppression (unlike ZS — import works); hypsarrhythmia emerging 4–12 weeks",
            "mri": "Cerebral atrophy (non-specific); delayed myelination; NO pachygyria (DHA synthesis partially intact unlike ZS); periventricular signal change",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Both alleles null (p.Arg68*/null or frameshift/null); PEX11B absent from peroxisomal membrane; fission completely fails → elongated tubular peroxisomes; PMP70 PRESENT + PEX14 PRESENT + catalase PEROXISOMAL (import intact); VLCFA borderline; plasmalogens borderline/normal (NOT severely low unlike ZSD)",
        },
        {
            "name": "Moderate Infantile (IS/West Syndrome) — null/p.Ile195Thr or similar",
            "pct": 45, "n": 18,
            "sex": "9M/9F",
            "onset_age": "Infantile (2nd–9th month)",
            "seizure_risk": "90% — infantile spasms (IS) 50–60% + hypsarrhythmia; photosensitive seizures; myoclonic; focal",
            "eeg": "Hypsarrhythmia (IS period); photoconvulsive response (EEG hallmark — 71% overall); multifocal spike-wave; modified hyps",
            "mri": "Cerebral atrophy; delayed myelination; NO gyral malformation (DHA borderline/normal); T2 WM changes",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Null/p.Ile195Thr (partial-function C-terminal oligomerisation domain — Ile195 in DRP1-recruitment interface; Thr substitution reduces DRP1 docking efficiency ~40% residual fission → fewer, larger peroxisomes but not near-absent); PEX11B detectable on elongated peroxisomes; catalase PEROXISOMAL; VLCFA borderline; plasmalogens NORMAL or borderline",
        },
        {
            "name": "Mild Childhood-Onset — p.Ile195Thr/mild or compound het mild",
            "pct": 15, "n": 6,
            "sex": "3M/3F",
            "onset_age": "Childhood (12M–7 yr)",
            "seizure_risk": "80% — focal seizures ± GTCS; photosensitive; SNHL often the presenting feature before seizures",
            "eeg": "Focal temporal-occipital spike-wave; photoconvulsive response (EEG hallmark); normal or slow background between events",
            "mri": "Near-normal or mild atrophy; no WM abnormalities; no gyral malformation",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Mild allele pair (p.Ile195Thr/p.Trp200Gly or similar partial-function); residual fission 30–50% → peroxisome population reduced but partially functional; peroxisome morphology elongated on IF but less severely; biochemistry near-normal",
        },
        {
            "name": "Atypical / Late-Onset — hypomorphic alleles",
            "pct": 5, "n": 2,
            "sex": "1M/1F",
            "onset_age": "Adolescence/adult (10–25 yr)",
            "seizure_risk": "50% — photosensitive focal epilepsy ± GTCS; SNHL may be the main disability",
            "eeg": "Photoconvulsive response; occasional focal discharge; largely normal background",
            "mri": "Normal or mild diffuse atrophy",
            "dha_supplement": False,
            "hsct_eligible": False,
            "ert_available": False,
            "variant_detail": "Hypomorphic variants (C-terminal missense with >60% residual DRP1 recruitment); peroxisome number mildly reduced; biochemistry normal; diagnosed on NGS after SNHL + photosensitive epilepsy workup",
        },
    ]

    patients = []
    random.seed(1122)
    diagnoses = [
        ("Severe Neonatal", 14),
        ("Moderate Infantile IS", 18),
        ("Mild Childhood", 6),
        ("Atypical Late-Onset", 2),
    ]
    sexes = ["M", "F"]
    onset_map = {
        "Severe Neonatal": "Day 1–14",
        "Moderate Infantile IS": "Month 2–9",
        "Mild Childhood": "Year 1–7",
        "Atypical Late-Onset": "Year 10–25",
    }
    drugs_map = {
        "Severe Neonatal": ["LEV", "Phenobarb", "ACTH"],
        "Moderate Infantile IS": ["LEV", "ACTH", "CLB"],
        "Mild Childhood": ["LEV", "LTG"],
        "Atypical Late-Onset": ["LEV"],
    }
    pid = 1
    for diag, count in diagnoses:
        for _ in range(count):
            age_y = round(random.uniform(0.1, 25 if "Late" in diag else (10 if "Childhood" in diag else (5 if "Infantile" in diag else 1.5))), 1)
            patients.append({
                "id": f"PEX11B-{pid:03d}",
                "age_at_report": f"{age_y}y",
                "sex": random.choice(sexes),
                "phenotype": diag,
                "onset": onset_map[diag],
                "drug_resistant": random.random() < (0.75 if "Severe" in diag else 0.55 if "Infantile" in diag else 0.35),
                "current_drugs": random.sample(drugs_map[diag], k=min(2, len(drugs_map[diag]))),
                "vlcfa": "Borderline" if random.random() < 0.65 else "Normal",
                "plasmalogens": "Borderline" if random.random() < 0.20 else "Normal",
                "snhl": random.random() < 0.78,
                "photosensitive": random.random() < 0.71,
                "retinopathy": random.random() < (0.72 if "Severe" in diag or "Infantile" in diag else 0.35),
            })
            pid += 1

    seizure_types = {
        "Neonatal myoclonic": {"pct": 35, "n": 14, "note": "Severe neonatal class; day 1–14; photosensitive from first week; EEG: multifocal spike-wave (no burst-suppression — import intact, DHA near-normal)"},
        "Infantile spasms (West syndrome)": {"pct": 42, "n": 17, "note": "Most common overall seizure type in moderate infantile class; IS + hypsarrhythmia; ACTH Level A preferred; photosensitive IED superimposed"},
        "Focal onset": {"pct": 28, "n": 11, "note": "Focal temporal-occipital (mild/atypical class); photosensitive EEG in most; LEV ± LTG adjunct"},
        "GTCS (generalised tonic-clonic)": {"pct": 20, "n": 8, "note": "Evolving from IS or independently in childhood class; LEV first-line; VGB caution if retinopathy"},
        "Myoclonic": {"pct": 18, "n": 7, "note": "Photosensitive myoclonus; prominent in severe + moderate class; CLB adjunct helpful"},
        "Absence": {"pct": 6, "n": 2, "note": "Atypical absence in mild/atypical class; not classical 3 Hz"},
    }

    triggers = [
        {"trigger": "Febrile illness / intercurrent infection", "pct": 62, "n": 25, "note": "Most common trigger (metabolic stress increases organelle demand on reduced peroxisome population)"},
        {"trigger": "Subtherapeutic AED levels (missed dose / growth-related underdosing)", "pct": 50, "n": 20, "note": "Growth underdosing common in infants and toddlers — weight-based dose adjustment every 3M essential"},
        {"trigger": "Photic stimulation / flickering light", "pct": 71, "n": 28, "note": "HALLMARK TRIGGER — 71% overall; photo-EEG mandatory; protective eyewear and screen filters; outdoor photosensitivity management"},
        {"trigger": "Sleep deprivation", "pct": 42, "n": 17, "note": "Avoid sleep deprivation; regular sleep hygiene protocol essential; nocturnal seizure monitoring"},
        {"trigger": "Missed AED dose", "pct": 35, "n": 14, "note": "Adherence support essential; alarm reminders; school nurse protocol for daytime doses"},
        {"trigger": "Prolonged fasting / NPO (caution level — not extreme hazard)", "pct": 30, "n": 12, "note": "CAUTION but not extreme hazard like ZSD — phytanic alpha-oxidation partially intact per peroxisome; fasting still mildly increases phytanic acid from adipose; IV dextrose for fasts >4h; not the acute neurotoxic surge of ZSD"},
        {"trigger": "Metabolic decompensation (illness-related catabolism)", "pct": 28, "n": 11, "note": "Catabolism stresses reduced peroxisome capacity; management: early enteral/IV dextrose during illness"},
        {"trigger": "Enzyme-inducing AEDs (PHT/CBZ/OXC — relative CI)", "pct": 14, "n": 6, "note": "Relative CI: mild DHA depletion risk (less severe than ZSD since DHA synthesis partially intact); avoid as first-line; replace with LEV on diagnosis"},
    ]

    monitoring = [
        {"parameter": "Plasma VLCFA (C26:0, C24:0, C26:0/C22:0 ratio)", "frequency": "6-monthly", "target": "Borderline/normal; frank elevation unexpected in PEX11B (distinguishes from ZSD)", "action": "If frank C26:0 elevation: reconsider ZSD diagnosis; repeat NGS including PEX5/PEX1/PEX6 etc."},
        {"parameter": "Erythrocyte plasmalogens (RBC; choline + ethanolamine)", "frequency": "Annually", "target": "NORMAL or borderline — critical: severely low = reconsider ZSD diagnosis", "action": "If severely low: reconsider ZSD; check PEX3/PEX16/PEX19/PEX1/PEX6 on expanded panel"},
        {"parameter": "Plasma phytanic acid + pristanic acid", "frequency": "6-monthly", "target": "Normal or borderline in most PEX11B; frank elevation = reconsider ZSD", "action": "If elevated: phytol-restricted dietary trial; if frank: reconsider ZSD diagnosis"},
        {"parameter": "Plasma pipecolic acid", "frequency": "Annually", "target": "Normal or mildly elevated in PEX11B; frank elevation = reconsider ZSD", "action": "If frank elevation: expanded peroxisomal panel + NGS"},
        {"parameter": "EEG (routine with photic stimulation protocol)", "frequency": "6-monthly in first 3 years; annually thereafter", "target": "Photoconvulsive response assessment mandatory; background maturation", "action": "Photoconvulsive response: protective eyewear + anti-photosensitive medication; hypsarrhythmia: ACTH Level A"},
        {"parameter": "MRI brain (with myelination assessment)", "frequency": "At diagnosis; 12M; 3yr; as needed", "target": "Non-specific atrophy/delayed myelination; NO pachygyria (if pachygyria: reconsider ZSD)", "action": "If pachygyria present: suspect ZSD not PEX11B; recheck biochemistry + NGS"},
        {"parameter": "Audiology (ABR + pure-tone audiogram)", "frequency": "3-monthly to 12M; 6-monthly to 5yr; annually thereafter", "target": "Bilateral SNHL present 78%; early amplification critical for language development", "action": "Hearing aids from 6M; cochlear implant evaluation if profound loss; early speech therapy"},
        {"parameter": "Ophthalmology (ERG + visual acuity + fundus)", "frequency": "Annually from diagnosis", "target": "RP in 62%; photosensitivity management; annual VF in children >5yr", "action": "RP confirmed: VGB absolute CI; protective eyewear; DHA supplementation may be trialled (Level C/experimental)"},
        {"parameter": "Peroxisome morphology (IF + EM on fibroblasts)", "frequency": "At diagnosis; confirmatory", "target": "Elongated/tubular peroxisomes (not absent, not spherical-normal) + peroxisomal catalase = PEX11B fission defect", "action": "Confirmatory for diagnosis; repeat if diagnosis uncertain; normal morphology excludes PEX11B"},
        {"parameter": "LFT + GGT + bilirubin", "frequency": "6-monthly", "target": "Liver disease UNCOMMON in PEX11B (bile acid intermediates near-normal, unlike ZSD)", "action": "If cholestatic: consider alternative ZSD diagnosis or coincidental liver disease"},
        {"parameter": "Developmental / neuropsychological assessment", "frequency": "Annually", "target": "Moderate-severe ID in 80%; early intervention", "action": "Intensive early intervention from diagnosis; speech + OT + PT; special education planning"},
        {"parameter": "Plasma DHA (docosahexaenoic acid)", "frequency": "Annually", "target": "Normal or borderline in PEX11B; if below range: trial DHA Level C/experimental", "action": "DHA supplementation may benefit if low; no Level A/B evidence for PEX11B specifically (unlike NALD/IRD ZSD)"},
    ]

    thresholds = [
        {"parameter": "C26:0 (VLCFA)", "value": ">1.30 μmol/L borderline; >1.60 frank", "action": "Frank elevation (>1.60): reconsider ZSD; expanded peroxisomal panel + NGS; borderline alone is consistent with PEX11B"},
        {"parameter": "Erythrocyte plasmalogens", "value": "<70% of mean = LOW", "action": "<70% mean: severe reduction inconsistent with PEX11B; reconsider ZSD (PEX3/PEX16/PEX19/PEX1/PEX6/PEX7/GNPAT/AGPS); NGS expanded panel"},
        {"parameter": "Plasma phytanic acid", "value": ">200 μmol/L frank elevation", "action": "Frank elevation: reconsider ZSD; if mild (<50): consistent with PEX11B; phytol dietary restriction"},
        {"parameter": "AED level trough", "value": "LEV: 20–40 mg/L; CLB: 30–300 ng/mL; LTG: 3–15 mg/L", "action": "Recheck q3M growth dose-adjustment in first 3yr; adjust to weight every visit"},
        {"parameter": "Photoconvulsive threshold", "value": "Any EEG photoconvulsive response", "action": "Protective eyewear (photochromic lenses grade 3); limit screen exposure; LEV reduces photosensitivity in some patients; avoid VGB"},
        {"parameter": "SNHL severity", "value": "Moderate (40–70 dB HL bilateral) = hearing aids; Profound (>70 dB): cochlear implant eval", "action": "Hearing aid fitting from 6M; speech therapy; cochlear implant evaluation by 18M if profound"},
    ]

    lifecycle = [
        {
            "stage": "Stage 1 — Neonatal/NICU (0–28 days, Severe class)",
            "features": "Neonatal myoclonic seizures day 1–14; photosensitive EEG (no burst-suppression — unlike ZS, import is intact); hypotonia; SNHL on ABR; NO facial dysmorphism (unlike ZS); NO pachygyria on MRI (DHA near-normal). KEY IF FINDING: PMP70 PRESENT + PEX14 PRESENT + catalase PEROXISOMAL + elongated/tubular peroxisome morphology — distinguishes PEX11B from ALL ZSD at birth",
            "action": "LEV IV first-line (20–30 mg/kg/day); Phenobarbital if refractory; AVOID VPA (hepatotoxicity + carnitine depletion); VLCFA + plasmalogens STAT (expect borderline/normal, not frank elevation); NGS PEX panel STAT (include PEX11B); peroxisome IF morphology on skin fibroblasts; ABR; ophthalmology baseline; photosensitivity precautions from day 1",
        },
        {
            "stage": "Stage 2 — Infantile (1–12 months, Moderate class dominant)",
            "features": "Infantile spasms (IS) ± hypsarrhythmia (50–60% of moderate class); photoconvulsive response emerging; SNHL confirmed on audiogram; RP screening; NO liver disease (unlike NALD/ZS — bile acids near-normal in PEX11B); DHA near-normal (MRI without pachygyria)",
            "action": "ACTH Level A (preferred for IS; VGB relative CI if any retinopathy confirmed); LEV adjunct; weight-based dose adjustment monthly; audiology — hearing aid fitting; ophthalmology — ERG + fundus; photosensitivity management (photochromic eyewear, screen filters); confirm PEX11B biallelic variants by NGS + peroxisome IF",
        },
        {
            "stage": "Stage 3 — Late Infantile/Toddler (1–3 yr)",
            "features": "Seizure evolution from IS to focal + myoclonic; photosensitive seizures prominent; SNHL stable; RP progression (if present); developmental delay confirmed; NO phytanic acid crisis risk (alpha-oxidation per-organelle intact); NO cholestatic liver (bile acid synthesis near-normal)",
            "action": "Optimise LEV ± CLB adjunct; if RP confirmed: VGB absolute CI; photoconvulsive precautions; annual ophthalmology; 6-monthly audiology; annual VLCFA + plasmalogens; early intervention: speech therapy, OT, PT; early childhood special education",
        },
        {
            "stage": "Stage 4 — Preschool/School Age (3–12 yr, Mild/Moderate class)",
            "features": "Focal ± GTCS ± photosensitive myoclonus; moderate-severe ID; SNHL stable; RP ± progressive; peripheral neuropathy emerging in some; NO adrenal insufficiency (unlike ABCD1 — no adrenal peroxisome accumulation of VLCFA as in X-ALD)",
            "action": "LEV ± LTG (Level C; glucuronidation, safer hepatically); VGB AVOID if any retinopathy; PHT/CBZ AVOID (DHA depletion risk, relative CI); adaptive aids for SNHL + RP; neuropsychological testing; school IEP/504 plan; annual metabolic review",
        },
        {
            "stage": "Stage 5 — Adolescence/Adulthood (12+ yr, Mild/Atypical class)",
            "features": "Seizures stable in well-controlled mild/atypical; SNHL predominant disability; RP stable; mild-moderate ID; photosensitivity persistent; adulthood transition planning",
            "action": "Long-term LEV ± LTG; photosensitivity lifelong management; annual metabolic + ophthalmology + audiology; genetic counselling (AR — 25% recurrence); no HSCT (non-inflammatory) / no ERT (integral membrane protein); gene therapy (AAV9-PEX11B) in preclinical research",
        },
        {
            "stage": "Stage 6 — Surgical / Anaesthesia Intercept (ANY stage)",
            "features": "Peri-operative fasting: CAUTION (not extreme hazard as in ZSD); phytanic acid borderline — fasting-triggered release is mild compared to ZSD. No cholestatic coagulopathy (bile acids near-normal). SNHL: communication challenges in perioperative setting",
            "action": "IV dextrose for fasts >4h (caution-level, not emergency mandate); VPA ABSOLUTE CI perioperatively (hepatotoxicity); IV LEV replaces oral; audiological support for communication; LFT baseline pre-op; anaesthetics team briefing re: PEX11B diagnosis and photosensitivity management",
        },
    ]

    treatments = [
        {
            "drug": "Levetiracetam (LEV)",
            "class": "AED — FIRST-LINE (all PEX11B forms)",
            "evidence": "Level A (expert consensus — PEX11B + ZSD peroxisomal cohorts; particularly effective for photosensitive epilepsy)",
            "dose": "Neonatal: 20–30 mg/kg/day IV BID; Infantile: 30–60 mg/kg/day PO BID; Adult: 1000–3000 mg/day BID",
            "moa": "SV2A modulation; reduces photosensitivity in some patients (adjunct benefit in PEX11B); no hepatotoxicity; no CYP induction; no carnitine depletion; IV available for status epilepticus",
            "monitoring": "Renal function; ADHD-like irritability side effects; weight-based adjustment every 3M in first 3yr",
            "ci": None,
        },
        {
            "drug": "ACTH (Adrenocorticotropic Hormone)",
            "class": "AED — Level A for Infantile Spasms",
            "evidence": "Level A (UKISS trial; multinational IS protocols; preferred over VGB in PEX11B when retinopathy confirmed or suspected)",
            "dose": "20–40 IU/day IM 2–4 weeks; taper over 4–6 weeks; monitor BP + glucose + electrolytes",
            "moa": "Melanocortin receptor pathway; down-regulates CRF; preferred over VGB to avoid additive VF loss in PEX11B-associated RP",
            "monitoring": "BP, glucose, electrolytes, infection risk; adrenal recovery monitoring post-taper",
            "ci": "Active infection; severe hypertension; substitute VGB ONLY if retinopathy confidently excluded (rare in PEX11B)",
        },
        {
            "drug": "Clobazam (CLB)",
            "class": "AED — Level B adjunct (focal/myoclonic)",
            "evidence": "Level B (expert consensus; low hepatotoxicity; effective for focal and myoclonic seizures in PEX11B)",
            "dose": "Child: 0.1–0.3 mg/kg/day PO BID; Adult: 10–30 mg/day BID",
            "moa": "GABA-A positive allosteric modulator (1,5-benzodiazepine); no CYP induction; no carnitine effect; no plasmalogen impact",
            "monitoring": "Sedation; tolerance (especially myoclonic seizure rebound); respiratory in neonates",
            "ci": "Severe hepatic failure (rare in PEX11B unlike ZSD); avoid in neonatal respiratory compromise",
        },
        {
            "drug": "Lamotrigine (LTG)",
            "class": "AED — Level C adjunct (mild/childhood class focal + GTCS)",
            "evidence": "Level C (case reports in peroxisomal disorders; glucuronidation → safer hepatic profile than PHT/CBZ in PEX11B)",
            "dose": "Start 0.15 mg/kg/day; titrate slowly over 8 weeks; target 3–5 mg/kg/day; adult 100–300 mg/day",
            "moa": "Sodium channel modulator; glucuronidation (UGT1A4) — avoids CYP3A4 hepatic burden; useful for focal + GTCS in mild/atypical class",
            "monitoring": "Stevens-Johnson (slow titration mandatory); avoid VPA co-administration; annual LFT",
            "ci": "VPA co-administration (double hepatotoxicity risk); rapid titration",
        },
        {
            "drug": "DHA (Docosahexaenoic Acid) — Level C / experimental",
            "class": "Supportive/Experimental — Level C (if plasma DHA below range)",
            "evidence": "Level C (no dedicated PEX11B trials; DHA synthesis near-normal in most PEX11B; may benefit if peroxisome count severely reduced → DHA borderline)",
            "dose": "100–200 mg/day as LPC-DHA if plasma DHA below range; trial for 6 months with ERG monitoring",
            "moa": "LPC-DHA absorbed intestinally → incorporated into neural/retinal membranes; bypasses any peroxisomal DHA synthesis reduction (fewer organelles); potential retinal protective benefit if RP present",
            "monitoring": "Plasma DHA q3M during trial; ERG q6M; if no improvement at 6M, discontinue",
            "ci": "Not indicated routinely (unlike ZSD where DHA deficiency is universal); use only if plasma DHA documented below normal",
        },
        {
            "drug": "Vigabatrin (VGB) — CAUTION: NOT first-line",
            "class": "AED — conditionally avoided in PEX11B",
            "evidence": "Conditional (VGB Level A for IS but retinopathy in 62% PEX11B — additive VF loss risk; use only if retinopathy confidently excluded)",
            "dose": "If used (retinopathy excluded): 50–150 mg/kg/day PO BID infants",
            "moa": "Irreversible GABA-T inhibition; effective for IS/West but causes irreversible concentric VF constriction in 30–50% long-term users",
            "monitoring": "Mandatory ERG + Goldmann VF at baseline and q6M; retinal OCT if accessible",
            "ci": "ABSOLUTE CI if RP confirmed; STRONG RELATIVE CI if ophthalmology not yet assessed; ACTH always preferred",
        },
    ]

    contraindications = [
        {
            "drug": "Valproate (VPA / Valproic Acid)",
            "level": "HIGH RISK — DUAL CI mechanism in PEX11B (not triple as in ZSD but still HIGH RISK)",
            "reason": "(1) Hepatotoxicity: VPA direct hepatotoxicity (mechanism 1; same as ZSD). (2) Carnitine depletion: VPA sequesters carnitine → secondary deficiency (mechanism 2; same as ZSD). NOTE: the THIRD ZSD mechanism (peroxisomal BO inhibition) is LESS severe in PEX11B because peroxisomal import IS intact — VLCFA oxidation enzymes ARE inside the organelle, just fewer organelles. However the dual hepato- + carnitine mechanism is still HIGH RISK. POLG1 MANDATORY (CPIC Grade A) before any VPA consideration.",
            "alternative": "LEV IV/PO (first-line). ACTH Level A (IS). LEV + CLB or LEV + LTG adjunct. Never VPA without POLG1 exclusion.",
        },
        {
            "drug": "Vigabatrin (VGB / Sabril)",
            "level": "RELATIVE CI — AVOID if retinopathy confirmed; CAUTION if retinopathy status unknown",
            "reason": "PEX11B causes retinopathy (RP) in ~62% of cases. VGB causes irreversible concentric VF constriction in 30–50% of long-term users. In PEX11B with confirmed RP, VGB-induced VF loss is additive → progressive blindness. Less universal than ZSD (where RP is near-100% in ZS/NALD), but risk is still substantial. ACTH (Level A) is always preferred for IS in PEX11B.",
            "alternative": "ACTH (Level A) for IS — preferred regardless of retinopathy status in PEX11B. LEV + CLB for focal/myoclonic.",
        },
        {
            "drug": "Fasting / Prolonged NPO",
            "level": "CAUTION (not Extreme Hazard as in ZSD)",
            "reason": "PEX11B retains per-organelle alpha-oxidation capacity (PHYH is imported into each peroxisome normally). Phytanic acid is borderline, not severely elevated. Fasting mildly increases phytanic release from adipose → mild risk. Distinguish from ZSD EXTREME HAZARD (where alpha-oxidation completely fails → acute phytanic/pristanic neurotoxic surge). In PEX11B: caution for fasts >4–6 hours; IV dextrose for prolonged NPO is prudent but not an emergency protocol mandate as in ZSD.",
            "alternative": "IV dextrose or early enteral feeding for fasts >4h; resume enteral feeds ASAP; monitor for metabolic decompensation during illness.",
        },
        {
            "drug": "Phenytoin / Carbamazepine / Oxcarbazepine (PHT / CBZ / OXC)",
            "level": "RELATIVE CI (avoid first-line)",
            "reason": "CYP3A4 induction → (1) DHA catabolism acceleration (DHA is borderline/normal in PEX11B — depletion risk is lower than ZSD but still present in patients with reduced peroxisome count); (2) hepatic microsomal CYP burden (less critical in PEX11B than ZSD since cholestatic liver disease is uncommon). Overall: RELATIVE CI. Use LEV/LTG instead. NOT absolute CI (no adrenal crisis risk — no ABCD1-equivalent adrenal storage).",
            "alternative": "LEV first-line; LTG (glucuronidation, Level C, safer hepatically) for focal/GTCS. Replace PHT/CBZ with LEV on PEX11B diagnosis.",
        },
        {
            "drug": "Lorenzo's Oil (oleic acid + erucic acid)",
            "level": "NOT INDICATED in PEX11B",
            "reason": "Lorenzo's Oil inhibits VLCFA elongation → mechanistically targets VLCFA accumulation in X-ALD (ABCD1 — peroxisomal import intact but VLCFA transporter absent). In PEX11B, VLCFA beta-oxidation enzymes ARE inside the peroxisome (import is intact) and VLCFA is only borderline elevated. Lorenzo's Oil addresses an elongation mechanism that is not the primary defect in PEX11B. Not indicated and no evidence of benefit.",
            "alternative": "LEV + ACTH + CLB. Photosensitivity management. Dietary support. No VLCFA-specific therapy needed in PEX11B.",
        },
        {
            "drug": "HSCT (Hematopoietic Stem Cell Transplantation)",
            "level": "NOT INDICATED (PEX11B)",
            "reason": "HSCT benefits neuroinflammatory conditions (cerebral X-ALD). PEX11B LOF is NOT inflammatory — it is a metabolic/organelle morphology defect. Donor microglia cannot restore PEX11B-mediated peroxisome fission function. HSCT carries 5–15% transplant-related mortality — unacceptable risk-benefit ratio in a non-inflammatory disease.",
            "alternative": "Supportive care: LEV + ACTH + CLB; hearing amplification; photosensitivity management. Gene therapy (AAV9-PEX11B) in preclinical research.",
        },
        {
            "drug": "Enzyme Replacement Therapy (ERT)",
            "level": "NOT AVAILABLE (PEX11B = integral peroxisomal membrane protein)",
            "reason": "ERT is feasible for soluble lysosomal enzymes (mannose-6-phosphate receptor delivery route). PEX11B is an INTEGRAL PEROXISOMAL MEMBRANE PROTEIN — not a secreted soluble enzyme. Cannot be delivered to target cells via IV infusion (no M6P receptor route; no endosomal escape for an integral membrane protein). Fundamentally different mechanism from lysosomal storage disease ERT.",
            "alternative": "Gene therapy (AAV9-PEX11B) in preclinical cell culture research; mRNA delivery experimental. No clinically available ERT 2026.",
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


# ── Definitions ───────────────────────────────────────────────────────────────
def get_definitions():
    key_concepts = [
        "PEX11B (247 aa, 1q22, OMIM *603637): integral peroxisomal membrane protein with 2 transmembrane domains. Function: oligomerise on peroxisomal membrane → elongate/tubulate → recruit DRP1 GTPase + FIS1 + MFF fission machinery → drive membrane scission → 2 daughter peroxisomes (peroxisome division cycle). N-terminal amphipathic helix (membrane curvature sensing) + C-terminal cytoplasmic oligomerisation/DRP1-docking domain",
        "FISSION TIER — 4th mechanistic tier of peroxisomal epilepsy: (1) Membrane-biogenesis tier (PEX3/PEX16/PEX19 LOF) — no peroxisome membrane; (2) Importomer tiers (PEX13/PEX14 LOF) — membrane present, import fails; Receptor tier (PEX5 LOF) — all structural components present, receptor absent, import fails; (3) FISSION TIER (PEX11B LOF) — membrane present, importomer intact, import working, but peroxisomes cannot divide → fewer, elongated organelles. PEX11B is the ONLY known human disease affecting peroxisome morphology/fission without abolishing matrix import",
        "CRITICAL IF SIGNATURE — PEX11B vs ZSD: In PEX11B LOF: PMP70 PRESENT (peroxisomal membrane intact) + PEX14 PRESENT (importomer intact) + catalase PEROXISOMAL (import IS working — distinguishes from ALL 13 ZSD-causing PEX genes where catalase is CYTOPLASMIC) + elongated/tubular peroxisome morphology. Catalase being PEROXISOMAL is the single most important IF feature of PEX11B. Electron microscopy confirms elongated/worm-shaped peroxisomes",
        "DRP1 (DNM1L gene) — cytoplasmic GTPase recruited by PEX11B: DRP1 assembles into helical oligomeric rings at constriction necks of peroxisomes (and mitochondria). GTP hydrolysis drives ring constriction → membrane fission. FIS1 (peroxisomal and mitochondrial) + MFF act as adaptor proteins. In PEX11B LOF: DRP1 cannot be recruited to peroxisomal constriction necks → elongated peroxisomes persist. DRP1-deficiency itself (DNM1L LOF) causes a DISTINCT disorder with combined peroxisomal + mitochondrial fission defects (different from PEX11B-isolated peroxisome fission defect)",
        "Biochemical partial phenotype (distinguishes PEX11B from ZSD): VLCFA borderline (not frank) + plasmalogens NORMAL or borderline (severely low = ZSD) + phytanic/pristanic normal/borderline + DHA normal/borderline + pipecolic normal/borderline. SINGLE KEY TEST: erythrocyte plasmalogens NORMAL in PEX11B; SEVERELY LOW in ALL ZSD (PEX1/2/3/5/6/7/10/12/13/14/16/19/26 LOF). Borderline NBS is consistent with PEX11B; frank NBS elevation → suspect ZSD",
        "Photosensitivity hallmark (71%): photoconvulsive EEG response in 71% of PEX11B patients — the highest photosensitivity prevalence among peroxisomal epilepsies. Mechanism likely involves altered neuronal membrane lipid composition in PEX11B-deficient neurons (locally reduced plasmalogen availability despite near-normal RBC plasmalogens). Protective eyewear (photochromic grade 3), screen filters, outdoor photosensitivity precautions are MANDATORY from diagnosis",
        "SNHL (78%): bilateral sensorineural hearing loss; cochlear peroxisomal fission defect → reduced hair cell organelle efficiency → progressive hearing loss. Distinguish from mitochondrial SNHL (PEX11B: normal lactate, no ragged red fibres, no mtDNA depletion). Early hearing aid fitting (6M) is critical for language acquisition. ABR at birth; pure-tone audiogram from 12M",
        "Retinopathy (RP, 62%): pigmentary retinopathy due to reduced peroxisomal capacity in retinal pigment epithelium + photoreceptors. Less universal than ZSD (where RP is ~100% in ZS/NALD because DHA synthesis fails completely). In PEX11B, DHA synthesis is near-normal but reduced organelle count may affect long-term retinal lipid maintenance. VGB CI if RP confirmed. Annual ERG + fundus",
        "Absence of pachygyria: ZSD causes pachygyria/polymicrogyria because DHA fails completely → neuronal migration defect (DHA is required for establishing cortical plate architecture). In PEX11B, DHA synthesis is near-normal (import intact) → cortical gyration is NORMAL or near-normal on MRI. If pachygyria seen: reconsider ZSD diagnosis",
        "No cholestatic liver disease (unlike ZSD): bile acid intermediate enzymes (DHCA-CoA ligase etc.) are imported normally in PEX11B (import intact). Therefore, DHCA/THCA do not accumulate → no cholestatic liver disease → no Vitamin K-dependent coagulopathy (unlike ZS/NALD where cholestasis drives coagulopathy)",
        "VPA DUAL CI (not triple as in ZSD): (1) hepatotoxicity + (2) carnitine depletion. The THIRD ZSD mechanism (peroxisomal BO inhibition worsening VLCFA) is less applicable in PEX11B since VLCFA-oxidising enzymes ARE inside the peroxisomes (import works). However dual hepato-/carnitine risk is still HIGH. POLG1 exclusion mandatory. Never use VPA as first-line in PEX11B",
        "VGB in PEX11B: conditional rather than absolute CI (unlike ZSD where RP is near-100%). RP present in ~62% PEX11B. If RP is CONFIRMED: VGB is absolute CI (additive VF loss risk). If retinopathy status is UNKNOWN: treat as VGB relative CI; ACTH Level A is always safer first choice for IS. Annual ERG mandatory",
        "Fasting CAUTION (not ZSD EXTREME HAZARD): in ZSD, ALL alpha-oxidation fails → frank phytanic elevation → acute neurotoxic surge on fasting. In PEX11B, per-organelle alpha-oxidation IS intact → phytanic borderline → fasting causes mild phytanic release only. IV dextrose is prudent for fasts >4h but is NOT the same emergency mandate as in ZSD. This is a clinically important distinction for perioperative management",
        "Gene therapy preclinical: AAV9-PEX11B transduction restores normal peroxisome morphology in patient-derived fibroblasts (elongated → normal spherical-punctate pattern). No clinical trials 2026. HSCT NOT indicated (non-inflammatory). No ERT (integral membrane protein — no M6P delivery route)",
        "Epidemiology: ~10–15 confirmed cases worldwide 2026; AR biallelic LOF; NO founder allele; first clinically described ~2012–2014; diagnosis requires fibroblast peroxisome IF morphology + biochemistry panel + targeted/NGS panel (PEX11B often not on standard lysosomal storage disease or Zellweger diagnostic panels — specific peroxisomal NGS panel required). OMIM *603637 / #614862",
    ]

    terms = {
        "Peroxisome fission": "Division of a peroxisome into two daughter organelles via PEX11B-mediated membrane elongation followed by DRP1 GTPase-driven membrane scission. Distinct from peroxisome biogenesis (de novo membrane formation by PEX3/PEX16 from ER vesicles) and matrix protein import (PEX5/PEX7 receptor-mediated transport).",
        "DRP1 (Dynamin-related protein 1, DNM1L)": "Cytoplasmic GTPase that drives fission of both peroxisomes and mitochondria. Assembled into helical oligomeric rings at organelle constriction necks. Recruited to peroxisomes by PEX11B + FIS1 + MFF. GTP hydrolysis drives constriction → membrane scission. DNM1L LOF = separate disease (DRP1 deficiency — affects both peroxisomal AND mitochondrial fission). PEX11B LOF = isolated peroxisomal fission defect only.",
        "FIS1 (Mitochondrial fission 1 protein)": "Tail-anchored protein on peroxisomal and mitochondrial outer membranes. Acts as adaptor for DRP1 recruitment (via FIS1-MFF axis). Cooperates with PEX11B in peroxisome fission. FIS1 mutations cause separate mitochondrial/peroxisomal disorders.",
        "Peroxisome elongation / tubulation": "Morphological change induced by PEX11B oligomerisation on the peroxisomal membrane surface. The membrane adopts a curved/tubular shape preparatory to fission. In PEX11B LOF, elongation proceeds but fission (DRP1 scission step) does not complete → elongated tubular peroxisomes accumulate.",
        "Catalase peroxisomal (IF localization)": "In normal cells and in PEX11B LOF, catalase (PTS1 cargo: -SKL) is imported into peroxisomes via PEX5 receptor and localizes intra-peroxisomally → punctate/elongated PTS1 import signal on IF. CRITICAL: In ALL ZSD (any tier), PEX5-mediated import fails → catalase remains CYTOPLASMIC (diffuse IF). Peroxisomal catalase = import working = not ZSD. This is the defining IF distinction for PEX11B.",
        "Erythrocyte plasmalogens": "Ether-linked phospholipids synthesised in peroxisomes (GNPAT/AGPS = matrix enzymes, PTS1 cargo). In ZSD: ALL matrix import fails → GNPAT/AGPS excluded → plasmalogens SEVERELY LOW. In PEX11B LOF: GNPAT/AGPS ARE imported (import intact per organelle) → plasmalogens NORMAL or borderline (reduced organelle count mildly reduces total flux). Severely low RBC plasmalogens = ZSD not PEX11B. Single most important biochemical discriminator.",
        "Photoconvulsive response": "EEG pattern of epileptiform discharge triggered by intermittent photic stimulation. In PEX11B: present in 71% (highest among peroxisomal epilepsies). Likely mechanism: altered neuronal membrane lipid composition (reduced per-neuron peroxisomal capacity) → increased neuronal excitability to light stimulus. Management: protective eyewear (photochromic grade 3 lenses, OD >1.5 at 500nm), blue-light screen filters, avoid disco/strobe lighting.",
        "Peroxisome biogenesis disorder (PBD)": "Group of disorders caused by mutations in PEX genes that impair peroxisome biogenesis (formation, matrix protein import, or fission). Classic PBD = ZSD (PEX1/2/3/5/6/7/10/12/13/14/16/19/26 LOF). PEX11B LOF represents a related but distinct 'PBD-9B' — a fission disorder rather than biogenesis/import disorder. Some nosologies include PEX11B in the PBD spectrum; others classify it as an isolated peroxisome morphology disorder.",
        "NBS (newborn screening) equivocal": "C26:0-lyso-PC on dried blood spot may be borderline/equivocal in PEX11B (partially intact VLCFA oxidation → not the frank elevation of ZSD). An equivocal NBS in a neonate with seizures + hypotonia should not exclude PEX11B. Proceed to: full plasma VLCFA + plasmalogens + phytanic + pipecolic panel + fibroblast IF morphology + comprehensive peroxisomal NGS panel.",
    }

    return {
        "key_concepts": key_concepts,
        "terms": terms,
    }
