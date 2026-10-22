#!/usr/bin/env python3
"""Hereditary-Nonketotic-Hyperglycinemia-Glycine-Encephalopathy-Hyperekplexia-Atlas -- Complete 8-Gene Reference
Glycine Cleavage System (NKH) and Glycine Receptor/Transporter (Hyperekplexia) Disorders
(GLDC/Classical-NKH-GCS-P-protein-~80pct-CSF-Plasma-Ratio->0.08-PATHOGNOMONIC-Burst-Suppression-Hiccupping-Sodium-Benzoate-Dextromethorphan,
 AMT/T-protein-~10pct-NKH-Biochemically-IDENTICAL-GLDC-AMT-GCSH-Molecular-Panel-Required-Attenuated-Variants,
 GCSH/H-protein-<1pct-NKH-EXTREMELY-RARE-<50-Cases-Worldwide-Molecular-Diagnosis-Only,
 DLD/L-protein-E3-COMBINED-GCS-PDH-KGDH-BCKDH-Deficiency-Lactic-Acidosis-PATHOGNOMONIC-DDx-GLDC-AMT-GCSH-Ashkenazi-G229C,
 GLRA1/GlyR-alpha1-Hyperekplexia-Type1-DOMINANT-p.R271L-Nose-Tap-PATHOGNOMONIC-Hypertonia-NOT-Hypotonia-Clonazepam,
 GLRB/GlyR-beta-Hyperekplexia-Type2-AR-ONLY-Gephyrin-Anchoring-Clonazepam-Effective,
 SLC6A5/GlyT2-Presynaptic-Transporter-Hyperekplexia-Type3-AD-AR-Clonazepam-CSF-Glycine-NORMAL,
 SLC6A9/GlyT1-Astrocytic-Transporter-GlyT1-Encephalopathy-CSF-Glycine-ELEVATED-GCS-Activity-NORMAL-NOT-Hyperekplexia):

GLDC (Glycine decarboxylase / P-protein; 1020aa; 9p24.1; AR-LOF;
     ~80% of classical NKH (non-ketotic hyperglycinemia);
     P-protein decarboxylases glycine using PLP (pyridoxal-5'-phosphate) — FIRST step of GCS;
     GCS = P-protein (GLDC) + H-protein (GCSH) + T-protein (AMT) + L-protein (DLD)
       in inner mitochondrial membrane;
     GLDC LOF → GCS unable to decarboxylase glycine → glycine accumulates systemically AND in CSF;
     BIOCHEMISTRY: plasma glycine VERY HIGH (600-2000 µmol/L; normal <450);
       CSF glycine VERY HIGH (>100 µmol/L; normal <10);
       CSF:plasma glycine ratio >0.08 PATHOGNOMONIC (normal <0.02);
     NO KETOSIS, NO ORGANIC ACIDEMIA, NO ACIDOSIS — distinguishes from KETOTIC hyperglycinemia
       (PA/MMA/IVA where glycine elevation is secondary to organic acid accumulation);
     CLINICAL PHENOTYPE: neonatal onset within hours to days;
       burst-suppression on EEG PATHOGNOMONIC at birth;
       HICCUPPING DISTINCTIVE (persistent/intractable in neonates — caused by glycine action
         on brainstem respiratory centres);
       severe hypotonia; apnea (respiratory failure requiring ventilation);
       myoclonic seizures; absent seizure semiology;
     PROGNOSIS: neonatal lethal (aggressive form) OR severe intellectual disability +
       drug-resistant epilepsy if survives; 'attenuated' NKH: later onset (2-6 months),
       milder ID, treatment-responsive seizures;
     NBS: NOT detected by standard acylcarnitine newborn screen (glycine not elevated to
       detected level on MS/MS; may show on amino acid profile if included);
     TREATMENT: sodium benzoate 250-750 mg/kg/day (conjugates glycine → hippurate, diverts
       glycine, reduces plasma/CSF glycine); dextromethorphan 5-20 mg/kg/day (NMDA antagonist,
       reduces NMDA receptor hyperstimulation from excess glycine); high-dose pyridoxine
       (GCS is PLP-dependent; response rate ~20%); ketamine (NMDA antagonist, acute crisis);
       high-dose folate (may help in some);
     AVOID: vigabatrin (GABA-T inhibitor — worsens by reducing GABA relative to glycine);
       sodium valproate (reduces glycine further catabolism);
     PATHOGNOMONIC: CSF:plasma glycine ratio >0.08 + normal organic acids + normal acylcarnitines
       (no PA/MMA/IVA) = NKH until proven otherwise;
     FOUNDER: p.S564I (Finnish); p.R515S (Mediterranean/Middle Eastern);
     seed SEED_BASE+0).

AMT  (Aminomethyltransferase / T-protein; 403aa; 3p21.31; AR-LOF;
     ~10% of classical NKH;
     T-protein transfers aminomethyl group from H-protein-lipoamide to THF → releasing NH3 +
       methyleneTHF; T-protein regenerates reduced H-protein; this is the SECOND step of GCS;
     AMT LOF → second step of GCS blocked → glycine accumulates identically to GLDC deficiency;
     BIOCHEMICALLY IDENTICAL to GLDC: same CSF:plasma glycine ratio elevation;
       CANNOT DISTINGUISH AMT vs GLDC vs GCSH by biochemistry alone;
     MOLECULAR DIAGNOSIS MANDATORY: gene panel (GLDC/AMT/GCSH) required to distinguish;
     CLINICAL PHENOTYPE: identical to GLDC NKH — neonatal onset, burst-suppression,
       hiccupping, hypotonia, apnea;
       slightly higher rate of attenuated variants than GLDC (AMT has more attenuated phenotype);
     PROGNOSIS: variable — AMT has more attenuated variants (later onset, milder);
       p.G269D Japanese patients often intermediate phenotype;
     TREATMENT: identical to GLDC — sodium benzoate + dextromethorphan; pyridoxine trial;
     FOUNDER: p.G269D (Japanese founder, attenuated NKH phenotype); p.W135R (Mediterranean);
     seed SEED_BASE+1).

GCSH (Glycine cleavage system protein H / H-protein; 173aa; 16q23.2; AR-LOF;
     GCSH is the lipoamide-bearing carrier: smallest GCS protein (173 aa / 20 kDa);
     H-protein oscillates between GCS components, shuttling the aminomethyl group
       between P-protein (GLDC) and T-protein (AMT);
     <1% of NKH; EXTREMELY RARE: <50 confirmed cases worldwide as of 2026;
     BIOCHEMISTRY: identical to GLDC/AMT — plasma glycine elevated, CSF glycine elevated,
       CSF:plasma ratio >0.08;
     NO BIOCHEMICAL WAY TO DISTINGUISH from GLDC/AMT: molecular panel required;
     CLINICAL: some attenuated variants described; neonatal presentation possible;
       molecular diagnosis via comprehensive panel;
     Very few pathogenic variants catalogued; MLPA required to detect large deletions;
     TREATMENT: identical to GLDC/AMT — sodium benzoate + dextromethorphan;
     No founder mutations established (too rare for population-level founder effects);
     seed SEED_BASE+2).

DLD  (Dihydrolipoamide dehydrogenase / L-protein / E3 subunit; 509aa; 7q31.1; AR-LOF;
     DLD = shared L-protein component of FOUR multi-enzyme complexes:
       GCS (glycine cleavage), PDH (pyruvate dehydrogenase), KGDH (alpha-ketoglutarate
       dehydrogenase), BCKDH (branched-chain keto-acid dehydrogenase);
     DLD LOF affects ALL FOUR complexes simultaneously = COMBINED deficiency;
     PATHOGNOMONIC: NKH-like glycine elevation (GCS failure) SIMULTANEOUSLY WITH lactic acidosis
       (PDH/KGDH failure) — SPECIFIC DDx: GLDC/AMT/GCSH have NKH ONLY without lactic acidosis;
     BIOCHEMISTRY: elevated plasma glycine (moderate, often less severe than GLDC);
       CSF glycine elevated; elevated plasma lactate/pyruvate (PDH); elevated 2-oxoglutarate/
       glutarate (KGDH); elevated BCAA/alloisoleucine (BCKDH component, as in MSUD type III);
       COMBINED organic/amino acid profile;
     Liver disease (prominent — KGDH role in Krebs cycle); cardiomyopathy in some;
     ASHKENAZI FOUNDER: p.G229C (most common Ashkenazi variant); Ashkenazi DLD deficiency
       often presents with LIVER-DOMINANT phenotype (not NKH-dominant);
     CLINICAL: hepatic presentation common (hepatomegaly, cholestasis, liver failure);
       Leigh-like brain (basal ganglia lesions on MRI); lactic acidosis crises triggered
       by illness; BCAA elevation/alloisoleucine; multi-system failure;
     TREATMENT: thiamine (for PDH + BCKDH components); riboflavin (FAD-dependent DLD);
       BCAA-reduced diet (for MSUD component); dichloroacetate (DCA) for lactic acidosis;
       sodium benzoate for glycine component; liver transplant considered for hepatic phenotype;
     NOTE: LIVER TRANSPLANT NOT CURATIVE (DLD expressed in brain/heart/muscle, not just
       liver — unlike MSUD type I/II where liver Tx cures the BCKDH deficiency);
     seed SEED_BASE+3).

GLRA1 (Glycine receptor alpha-1 subunit; 464aa; 5q33.1; AD-dominant-negative or AR-LOF;
     GLRA1 encodes glycine receptor alpha-1 subunit — primary inhibitory ligand-gated chloride
       channel at spinal cord and brainstem glycinergic synapses;
     HYPEREKPLEXIA TYPE 1 (startle disease; OMIM #149400);
     FUNDAMENTALLY DIFFERENT from NKH: glycine SIGNALLING DEFICIENT (receptor absent/
       dysfunctional), NOT glycine accumulation; plasma/CSF glycine NORMAL;
     PATHOGNOMONIC: HYPEREKPLEXIA = exaggerated non-habituating startle response to sudden
       acoustic or tactile stimuli → brief tonic spasm → rigidity → falling (WITHOUT loss
       of consciousness);
     NOSE TAP TEST: firm tap to nasal bridge → sustained tonic contraction + head retraction
       = PATHOGNOMONIC for hyperekplexia;
     NEONATAL PRESENTATION: hypertonia (board-like rigidity); apneas triggered by startle
       (life-threatening neonatal apneas); bradycardia;
       HYPERTONIA (NOT hypotonia — OPPOSITE of NKH);
       UMBILICUS MANOEUVRE: forced flexion of head/limbs terminates spasm;
     AD DOMINANT NEGATIVE (most common): one pathogenic allele disrupts pentamer assembly;
       p.R271L most common worldwide; p.R271Q, p.R252H;
     AR RECESSIVE FORM: rare; null alleles; more severe; may have peripheral neuropathy;
     TREATMENT: clonazepam (benzodiazepine — potentiates GABA-A to compensate for glycine
       receptor loss); 0.1-0.2 mg/kg/day; highly effective; lifelong required;
       some adults can taper; CLONAZEPAM HIGHLY EFFECTIVE = PATHOGNOMONIC TREATMENT RESPONSE;
     Key DDx from NKH: glycine levels NORMAL; no burst-suppression; hypertonia not hypotonia;
       startle-triggered (not spontaneous myoclonus); nose-tap positive;
       clonazepam effective (NKH refractory to usual AEDs);
     seed SEED_BASE+4).

GLRB (Glycine receptor beta subunit; 474aa; 4q32.1; AR-LOF;
     GLRB encodes glycine receptor beta subunit — required for surface expression and anchoring
       of heteromeric alpha-1-beta receptors at synapses via GEPHYRIN binding;
       GLRB essential for synaptic clustering of GlyR at inhibitory synapses;
     HYPEREKPLEXIA TYPE 2 (OMIM #149401) — AR ONLY (no dominant variants described for GLRB);
     IDENTICAL CLINICAL PHENOTYPE to GLRA1 hyperekplexia — cannot distinguish clinically;
     Biochemically: glycine levels NORMAL (receptor dysfunction, not accumulation);
     RARER than GLRA1 (AR only, less founder effect);
     TREATMENT: clonazepam (same as GLRA1); highly effective;
     Molecular panel (GLRA1 + GLRB + SLC6A5) required for genetic diagnosis;
     seed SEED_BASE+5).

SLC6A5 (Solute carrier family 6 member 5 / GlyT2 / Glycine transporter 2; 799aa; 11p15.1; AD or AR;
     GlyT2 is a presynaptic Na+/Cl- dependent glycine transporter — RECYCLES glycine back into
       presynaptic glycinergic interneurons at spinal cord and brainstem;
     GlyT2 dysfunction → DEPLETION of presynaptic glycine stores → glycinergic inhibitory
       transmission severely reduced → HYPEREKPLEXIA;
     HYPEREKPLEXIA TYPE 3 (OMIM #614618; also called GlyT2-related hyperekplexia);
     AD FORM (HAPLOINSUFFICIENCY): heterozygous LoF → 50% GlyT2 → insufficient presynaptic
       glycine recycling; p.Y705S, p.N509S, common AD variants;
     AR FORM: more severe; biallelic null; may be lethal in severe cases;
     CLINICAL: identical to GLRA1/GLRB hyperekplexia — startle, neonatal apnea, hypertonia,
       nose-tap positive;
     GLYCINE LEVELS: may be mildly elevated in plasma/CSF (transporter dysfunction → less
       efficient glycine removal), OR normal — DIFFERENT from NKH where glycine is very high;
       CSF:plasma glycine ratio: NORMAL (<0.08) — distinguishes from NKH;
     TREATMENT: clonazepam; same as GLRA1; highly effective;
     seed SEED_BASE+6).

SLC6A9 (Solute carrier family 6 member 9 / GlyT1 / Glycine transporter 1; 706aa; 1p34.1; AR-LOF;
     GlyT1 is an astrocytic Na+/Cl- cotransporter — CLEARS glycine from the synapse
       (glycine is also co-agonist at NMDA receptors); predominantly expressed in astrocytes
       surrounding synapses;
     GlyT1 deficiency = inability to clear glycine from synapses → elevated CSF glycine +
       glycine encephalopathy;
     GlyT1 ENCEPHALOPATHY / GlyT1 SYNDROME (rare, described ~2016; very few cases);
     BIOCHEMISTRY: CSF glycine ELEVATED (>50 µmol/L); CSF:plasma glycine ratio ELEVATED
       (can mimic NKH); HOWEVER glycine cleavage system enzyme activity NORMAL in
       liver/fibroblasts — KEY DDx from true NKH (GLDC/AMT/GCSH/DLD);
       GCS enzyme activity NORMAL = excludes GLDC/AMT/GCSH/DLD;
     CLINICAL: severe neonatal encephalopathy; burst-suppression EEG; respiratory failure;
       overlaps NKH phenotype;
     TREATMENT: SYMPTOMATIC — sodium benzoate may help (similar to NKH);
       sarcosine (N-methylglycine) as GlyT1 substrate competitor to reduce glycine transport
       dysfunction; NMDA modulators; glycine-restricted diet;
       DO NOT confuse with NKH treatment: because GCS is intact, the glycine surplus is
       synaptic not systemic;
     VERY RARE: <20 cases worldwide as of 2026;
     NOT hyperekplexia (GlyT1 deficiency causes NKH-like encephalopathy, NOT startle disease);
     seed SEED_BASE+7).

SEED_BASE = 4198
"""

import json
import random
from typing import Any


ATLAS_ID   = "hereditary-nonketotic-hyperglycinemia-glycine-encephalopathy-hyperekplexia-atlas"
ATLAS_NAME = "Hereditary-Nonketotic-Hyperglycinemia-Glycine-Encephalopathy-Hyperekplexia-Atlas"
SEED_BASE  = 4198
N_PER_GENE = 40

GENES = ["GLDC", "AMT", "GCSH", "DLD", "GLRA1", "GLRB", "SLC6A5", "SLC6A9"]

GENE_META = {
    "GLDC":   {"aa": 1020, "locus": "9p24.1",  "kda": 114, "inh": "AR",    "disease": "Classical NKH (GLDC deficiency, ~80% of NKH)",              "omim_gene": 238300, "omim_disease": 605899},
    "AMT":    {"aa": 403,  "locus": "3p21.31",  "kda": 45,  "inh": "AR",    "disease": "Classical NKH (AMT deficiency, T-protein, ~10% of NKH)",    "omim_gene": 238310, "omim_disease": 606580},
    "GCSH":   {"aa": 173,  "locus": "16q23.2",  "kda": 20,  "inh": "AR",    "disease": "Classical NKH (GCSH deficiency, H-protein, <1% NKH)",       "omim_gene": 238330, "omim_disease": 238330},
    "DLD":    {"aa": 509,  "locus": "7q31.1",   "kda": 56,  "inh": "AR",    "disease": "DLD Deficiency (Combined GCS+PDH+KGDH+BCKDH, E3 subunit)", "omim_gene": 238331, "omim_disease": 246900},
    "GLRA1":  {"aa": 464,  "locus": "5q33.1",   "kda": 52,  "inh": "AD/AR", "disease": "Hyperekplexia Type 1 (Startle Disease, GlyR alpha-1)",      "omim_gene": 138491, "omim_disease": 149400},
    "GLRB":   {"aa": 474,  "locus": "4q32.1",   "kda": 53,  "inh": "AR",    "disease": "Hyperekplexia Type 2 (GlyR beta subunit, AR only)",         "omim_gene": 138492, "omim_disease": 149401},
    "SLC6A5": {"aa": 799,  "locus": "11p15.1",  "kda": 87,  "inh": "AD/AR", "disease": "Hyperekplexia Type 3 (GlyT2, presynaptic glycine recycling)","omim_gene": 604159, "omim_disease": 614618},
    "SLC6A9": {"aa": 706,  "locus": "1p34.1",   "kda": 78,  "inh": "AR",    "disease": "GlyT1 Encephalopathy (astrocytic glycine clearance defect)", "omim_gene": 601019, "omim_disease": 617301},
}

GENE_PATHOGNOMONIC = {
    "GLDC":   "CSF:plasma glycine ratio >0.08 PATHOGNOMONIC + burst-suppression EEG + hiccupping neonatal + NO ketosis/organic acidemia",
    "AMT":    "Biochemically IDENTICAL to GLDC; CSF:plasma glycine ratio >0.08; molecular panel (GLDC/AMT/GCSH) required to distinguish",
    "GCSH":   "Biochemically IDENTICAL to GLDC/AMT; CSF:plasma glycine ratio >0.08; EXTREMELY RARE (<50 cases); molecular panel mandatory",
    "DLD":    "NKH-like glycine elevation SIMULTANEOUSLY WITH lactic acidosis PATHOGNOMONIC; combined GCS+PDH+KGDH+BCKDH failure; Ashkenazi p.G229C",
    "GLRA1":  "Nose-tap test PATHOGNOMONIC (tonic contraction on nasal bridge tap); hypertonia (NOT hypotonia); glycine NORMAL; AD p.R271L most common",
    "GLRB":   "Hyperekplexia identical to GLRA1; AR only; glycine NORMAL; gephyrin binding lost; molecular panel required to distinguish from GLRA1",
    "SLC6A5": "Hyperekplexia (startle, neonatal apnea, nose-tap positive); CSF:plasma glycine NORMAL or mildly elevated; GlyT2 presynaptic recycling defect",
    "SLC6A9": "CSF glycine ELEVATED mimicking NKH BUT GCS enzyme activity NORMAL in fibroblasts — KEY DDx from true NKH; GlyT1 encephalopathy; NOT hyperekplexia",
}

GENE_KEY_DRUG = {
    "GLDC":   "Sodium benzoate 250-750 mg/kg/day + dextromethorphan 5-20 mg/kg/day; avoid vigabatrin/valproate; pyridoxine trial (~20% response)",
    "AMT":    "Sodium benzoate + dextromethorphan (identical to GLDC); pyridoxine trial; attenuated variants may respond better",
    "GCSH":   "Sodium benzoate + dextromethorphan (identical to GLDC/AMT); symptomatic AED management",
    "DLD":    "Thiamine + riboflavin + BCAA-reduced diet + DCA for lactic acidosis + sodium benzoate for glycine; liver Tx NOT curative (brain/heart affected)",
    "GLRA1":  "Clonazepam 0.1-0.2 mg/kg/day (highly effective, lifelong); umbilicus manoeuvre for acute neonatal spasm; NOT dextromethorphan/benzoate",
    "GLRB":   "Clonazepam (identical to GLRA1); highly effective",
    "SLC6A5": "Clonazepam (same as GLRA1/GLRB); highly effective; molecular panel to confirm GlyT2 vs GlyR mutation",
    "SLC6A9": "Symptomatic: sodium benzoate ± sarcosine; NMDA modulators; glycine-restricted diet; NOT clonazepam first-line (encephalopathy not startle disease)",
}

VITAMIN_COFACTOR = {
    "GLDC":   "PLP (pyridoxal-5'-phosphate) — GCS is PLP-dependent; pyridoxine trial; ~20% response rate",
    "AMT":    "PLP — T-protein uses PLP; pyridoxine trial recommended",
    "GCSH":   "None specific (carrier/shuttle protein; no direct cofactor use)",
    "DLD":    "FAD + NAD+ (dihydrolipoamide dehydrogenase is FAD-dependent); thiamine for PDH/BCKDH components; riboflavin",
    "GLRA1":  "Cl- channel (no vitamin cofactor; clonazepam potentiates Cl- influx via GABA-A)",
    "GLRB":   "Cl- channel (no vitamin cofactor)",
    "SLC6A5": "Na+/Cl- cotransporter (no vitamin; ionic gradient driven)",
    "SLC6A9": "Na+/Cl- cotransporter (no vitamin; ionic gradient driven)",
}

GLYCINE_CSF_PLASMA_RATIO_STATUS = {
    "GLDC":   ">0.08 PATHOGNOMONIC",
    "AMT":    ">0.08 PATHOGNOMONIC",
    "GCSH":   ">0.08 PATHOGNOMONIC",
    "DLD":    ">0.08 plus lactic acidosis PATHOGNOMONIC",
    "GLRA1":  "NORMAL (<0.04)",
    "GLRB":   "NORMAL (<0.04)",
    "SLC6A5": "NORMAL or mildly elevated",
    "SLC6A9": "ELEVATED (mimics NKH but GCS activity NORMAL)",
}

NBS_DETECTS = {
    "GLDC":   False,
    "AMT":    False,
    "GCSH":   False,
    "DLD":    False,  # partially (lactic acidosis/BCAA may trigger, unreliable)
    "GLRA1":  False,
    "GLRB":   False,
    "SLC6A5": False,
    "SLC6A9": False,
}

CLONAZEPAM_RESPONSIVE = {
    "GLDC":   False,
    "AMT":    False,
    "GCSH":   False,
    "DLD":    False,
    "GLRA1":  True,
    "GLRB":   True,
    "SLC6A5": True,
    "SLC6A9": False,
}

FOUNDER_MUTATIONS = {
    "GLDC":   "p.S564I (Finnish founder, attenuated NKH); p.R515S (Mediterranean/Middle Eastern); IVS7-1G>A",
    "AMT":    "p.G269D (Japanese founder, often intermediate/attenuated phenotype); p.W135R (Mediterranean)",
    "GCSH":   "No established common founder (too rare; <50 cases worldwide; each family often private variants)",
    "DLD":    "p.G229C (Ashkenazi Jewish founder, liver-dominant phenotype); p.I12T (Palestinian Arab founder)",
    "GLRA1":  "p.R271L (most common worldwide, AD dominant-negative); p.R271Q; p.R252H; several founder effects",
    "GLRB":   "Rare individual variants; no dominant founder; AR recessive only",
    "SLC6A5": "p.Y705S (reported European cases); p.N509S (AD form); biallelic null (severe AR form)",
    "SLC6A9": "Very rare (<20 cases 2026); private family variants; no founder established",
}

NKH_GROUP = ["GLDC", "AMT", "GCSH", "DLD"]
HYPEREKPLEXIA_GROUP = ["GLRA1", "GLRB", "SLC6A5", "SLC6A9"]


def _make_cohort(gene: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    meta = GENE_META[gene]
    n = N_PER_GENE

    male_n = rng.randint(n // 3, 2 * n // 3)
    female_n = n - male_n

    # Onset in months
    if gene in ("GLDC", "GCSH"):
        median_onset = rng.uniform(0, 0.5)    # neonatal within hours-days
    elif gene == "AMT":
        median_onset = rng.uniform(0, 2)      # neonatal; attenuated variants later
    elif gene == "DLD":
        median_onset = rng.uniform(0, 3)      # neonatal/early infantile
    elif gene in ("GLRA1", "GLRB", "SLC6A5"):
        median_onset = rng.uniform(0, 0.5)    # neonatal hyperekplexia
    else:  # SLC6A9
        median_onset = rng.uniform(0, 1)      # neonatal encephalopathy

    # CSF glycine elevated (NKH group) vs normal (hyperekplexia group)
    csf_glycine_elevated_pct = (
        rng.uniform(0.88, 0.97) if gene in NKH_GROUP else
        (rng.uniform(0.20, 0.40) if gene == "SLC6A5" else  # mildly elevated possible
         (rng.uniform(0.60, 0.80) if gene == "SLC6A9" else  # elevated, mimics NKH
          rng.uniform(0.02, 0.06)))  # GLRA1/GLRB: normal
    )

    # Lactic acidosis (DLD only)
    lactic_acidosis_pct = rng.uniform(0.75, 0.92) if gene == "DLD" else rng.uniform(0.0, 0.04)

    # Burst-suppression EEG on neonatal recording
    burst_suppression_pct = (
        rng.uniform(0.80, 0.95) if gene in ("GLDC", "AMT", "GCSH") else
        (rng.uniform(0.55, 0.75) if gene in ("DLD", "SLC6A9") else
         rng.uniform(0.02, 0.06))  # hyperekplexia: no burst-suppression
    )

    # Hypertonia (hyperekplexia) vs hypotonia (NKH)
    hypertonia_pct = (
        rng.uniform(0.82, 0.95) if gene in ("GLRA1", "GLRB", "SLC6A5") else
        rng.uniform(0.02, 0.08)
    )

    # Nose-tap positive (GLRA1/GLRB/SLC6A5 only)
    nose_tap_pct = (
        rng.uniform(0.85, 0.97) if gene in ("GLRA1", "GLRB", "SLC6A5") else
        rng.uniform(0.0, 0.04)
    )

    # Clonazepam responsive
    clonaz_resp_pct = (
        rng.uniform(0.82, 0.95) if CLONAZEPAM_RESPONSIVE[gene] else
        rng.uniform(0.02, 0.08)
    )

    # NBS detection
    nbs_det_pct = 0.03 if not NBS_DETECTS[gene] else 0.40  # DLD partial

    # Hiccupping (GLDC/AMT/GCSH neonatal)
    hiccupping_pct = (
        rng.uniform(0.55, 0.80) if gene in ("GLDC", "AMT", "GCSH") else
        rng.uniform(0.0, 0.05)
    )

    return {
        "gene":                        gene,
        "n":                           n,
        "seed":                        seed,
        "male_n":                      male_n,
        "female_n":                    female_n,
        "disease":                     meta["disease"],
        "locus":                       meta["locus"],
        "aa":                          meta["aa"],
        "kda":                         meta["kda"],
        "inh":                         meta["inh"],
        "glycine_csf_plasma_ratio":    GLYCINE_CSF_PLASMA_RATIO_STATUS[gene],
        "vitamin_cofactor":            VITAMIN_COFACTOR[gene],
        "nbs_detected_pct":            round(nbs_det_pct, 2),
        "clonazepam_responsive":       CLONAZEPAM_RESPONSIVE[gene],
        "clonazepam_responsive_pct":   round(clonaz_resp_pct, 2),
        "median_onset_months":         round(median_onset, 2),
        "burst_suppression_pct":       round(burst_suppression_pct, 2),
        "csf_glycine_elevated_pct":    round(csf_glycine_elevated_pct, 2),
        "lactic_acidosis_pct":         round(lactic_acidosis_pct, 2),
        "hypertonia_pct":              round(hypertonia_pct, 2),
        "nose_tap_pct":                round(nose_tap_pct, 2),
        "hiccupping_pct":              round(hiccupping_pct, 2),
        "pathognomonic":               GENE_PATHOGNOMONIC[gene],
        "key_drug":                    GENE_KEY_DRUG[gene],
        "founder_mutation":            FOUNDER_MUTATIONS[gene],
    }


def generate_overview() -> dict[str, Any]:
    cohorts = [_make_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    total = sum(c["n"] for c in cohorts)

    return {
        "atlas_id":       ATLAS_ID,
        "atlas_name":     ATLAS_NAME,
        "total_patients": total,
        "genes_count":    len(GENES),
        "seeds":          f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "gene_list":      GENES,
        "gene_cohorts":   cohorts,
        "atlas_summary": {
            "pathway_covered":              "Glycine Cleavage System / NKH (GLDC/AMT/GCSH/DLD) + Glycine Receptor Inhibition (GLRA1/GLRB) + Glycine Transporter (SLC6A5 GlyT2 / SLC6A9 GlyT1)",
            "nkh_group_genes":              NKH_GROUP,
            "hyperekplexia_group_genes":    ["GLRA1", "GLRB", "SLC6A5"],
            "gcs_enzyme_normal_mimics_nkh": ["SLC6A9"],
            "clonazepam_responsive_genes":  [g for g in GENES if CLONAZEPAM_RESPONSIVE[g]],
            "nbs_detected_genes":           [g for g in GENES if NBS_DETECTS[g]],
            "csf_glycine_elevated_genes":   [g for g in GENES if ">0.08" in GLYCINE_CSF_PLASMA_RATIO_STATUS[g] or "ELEVATED" in GLYCINE_CSF_PLASMA_RATIO_STATUS[g]],
            "glycine_normal_genes":         [g for g in GENES if "NORMAL" in GLYCINE_CSF_PLASMA_RATIO_STATUS[g]],
            "critical_ddx_rule":            "PRIMARY SPLIT: CSF glycine elevated (NKH group + SLC6A9) vs CSF glycine NORMAL (GLRA1/GLRB/SLC6A5 hyperekplexia); WITHIN NKH: DLD has LACTIC ACIDOSIS PATHOGNOMONIC; GCSH/AMT/GLDC pure NKH — molecular only; SLC6A9 mimics NKH BUT GCS enzyme activity NORMAL in fibroblasts",
        },
    }


def generate_breakdown() -> dict[str, Any]:
    pathway_groups = {
        "nkh_glycine_cleavage_system_GLDC_AMT_GCSH_DLD": {
            "genes":         ["GLDC", "AMT", "GCSH", "DLD"],
            "mechanism":     "GCS is a four-protein complex in the inner mitochondrial membrane: P-protein (GLDC) decarboxylases glycine using PLP; H-protein (GCSH) shuttles the aminomethyl intermediate via lipoamide arm; T-protein (AMT) transfers aminomethyl to THF releasing NH3+5,10-methyleneTHF; L-protein (DLD) reoxidises H-protein lipoamide using FAD/NAD+. LOF in any component → glycine accumulates systemically and in CSF.",
            "shared_feature": "All four: CSF:plasma glycine ratio >0.08 PATHOGNOMONIC; plasma glycine 600-2000 µmol/L; CSF glycine >100 µmol/L; burst-suppression neonatal EEG; neonatal onset; hiccupping (GLDC/AMT/GCSH more than DLD). NBS typically misses all four on standard acylcarnitine screen.",
            "key_ddx":        "DLD: lactic acidosis + alloisoleucine + BCAA elevation + hepatic disease = PATHOGNOMONIC combined deficiency (GCS+PDH+KGDH+BCKDH). GLDC/AMT/GCSH: pure NKH, no lactic acidosis, no BCAA elevation. GLDC vs AMT vs GCSH: biochemically IDENTICAL — molecular gene panel mandatory. SLC6A9 mimics NKH but GCS enzyme activity is NORMAL in fibroblasts.",
        },
        "hyperekplexia_glycine_receptor_transporter_GLRA1_GLRB_SLC6A5_SLC6A9": {
            "genes":         ["GLRA1", "GLRB", "SLC6A5", "SLC6A9"],
            "mechanism":     "GLRA1/GLRB: glycine receptor subunits; GlyR is a heteropentameric ligand-gated Cl- channel at inhibitory glycinergic synapses in spinal cord and brainstem; LOF → glycinergic inhibition lost → hyperekplexia. SLC6A5 (GlyT2): presynaptic Na+/Cl- transporter that recycles glycine into presynaptic terminals; LOF → presynaptic glycine depletion → reduced glycinergic transmission → hyperekplexia. SLC6A9 (GlyT1): astrocytic Na+/Cl- transporter clears glycine from synapse; LOF → synaptic glycine accumulation → NMDA co-agonist excess → encephalopathy (NOT hyperekplexia).",
            "shared_feature": "GLRA1/GLRB/SLC6A5: glycine signalling deficient (not accumulation); plasma/CSF glycine NORMAL; hypertonia (board-like rigidity) NOT hypotonia; neonatal apneas triggered by startle; nose-tap test PATHOGNOMONIC; clonazepam HIGHLY EFFECTIVE. SLC6A9: different — glycine elevated in CSF; encephalopathy not startle; NOT clonazepam first-line.",
            "key_ddx":        "GLRA1 vs GLRB vs SLC6A5: clinically identical hyperekplexia; molecular panel required. GLRA1: AD (dominant negative) most common (p.R271L) — consider AD pattern. GLRB: AR only. SLC6A5: AD or AR. SLC6A9: CSF glycine elevated, GCS normal — unique position, resembles NKH biochemically but is a transporter defect. All hyperekplexia group: glycine NORMAL distinguishes from NKH group.",
        },
    }

    glycine_diagnostic_cascade = [
        "STEP 1 — PLASMA AMINO ACIDS (quantitative): glycine markedly elevated (>600 µmol/L; normal <450 µmol/L) → immediately raises NKH vs ketotic hyperglycinemia differential; hypertonia at birth with startle → consider hyperekplexia (glycine may be normal)",
        "STEP 2 — SIMULTANEOUS CSF AMINO ACIDS (must be done at SAME time as plasma): CSF glycine >100 µmol/L PLUS calculate CSF:plasma glycine ratio; ratio >0.08 = PATHOGNOMONIC for NKH group (GLDC/AMT/GCSH/DLD/SLC6A9); ratio <0.04 = hyperekplexia group (GLRA1/GLRB/SLC6A5)",
        "STEP 3 — PLASMA LACTATE AND PYRUVATE: if lactate elevated (>2.5 mmol/L) with L:P ratio >20 → DLD deficiency (combined PDH/KGDH/GCS failure); GLDC/AMT/GCSH have NORMAL lactate — CRITICAL DLD vs pure NKH DDx",
        "STEP 4 — URINE ORGANIC ACIDS: in DLD: 2-oxoglutarate elevated + 2-hydroxyglutarate + lactate; in pure NKH (GLDC/AMT/GCSH): organic acids NORMAL; in ketotic hyperglycinemia (PA/MMA/IVA): propionate/methylmalonate/isovalerate elevated with ketosis",
        "STEP 5 — PLASMA AMINO ACIDS FOR BCAA/ALLOISOLEUCINE: alloisoleucine elevated → DLD (BCKDH component) or MSUD; absent in GLDC/AMT/GCSH pure NKH",
        "STEP 6 — NEONATAL EEG: burst-suppression pattern PATHOGNOMONIC for NKH group (GLDC/AMT/GCSH/DLD/SLC6A9); hyperekplexia group (GLRA1/GLRB/SLC6A5): NO burst-suppression; may have normal EEG or non-specific changes",
        "STEP 7 — NOSE-TAP TEST (if hyperekplexia suspected): firm tap to nasal bridge → sustained tonic contraction + head retraction = POSITIVE = PATHOGNOMONIC HYPEREKPLEXIA; absent in NKH (hypotonic, not hypertonically reactive); perform when hypertonia + startle apneas present",
        "STEP 8 — GCS ENZYME ACTIVITY IN FIBROBLASTS/LIVER (if NKH confirmed biochemically): glycine cleavage activity measured using [1-14C]-glycine or [2-14C]-glycine assay; ABSENT/severely reduced = true GCS defect (GLDC/AMT/GCSH/DLD); NORMAL GCS activity + elevated CSF glycine ratio = SLC6A9 (GlyT1 encephalopathy) — KEY DDx",
        "STEP 9 — MOLECULAR GENE PANEL (all confirmed NKH and hyperekplexia): GLDC + AMT + GCSH (for NKH group); DLD (if lactic acidosis/combined); GLRA1 + GLRB + SLC6A5 (for hyperekplexia); SLC6A9 (if GCS enzyme normal + CSF glycine elevated); panel distinguishes GLDC/AMT/GCSH which are biochemically identical",
        "STEP 10 — PYRIDOXINE TRIAL AND TREATMENT INITIATION: pyridoxine 50 mg/day trial (if GCS defect confirmed) for ~20% response; start sodium benzoate + dextromethorphan immediately in NKH; start clonazepam in hyperekplexia group; avoid vigabatrin/valproate in NKH; repeat CSF glycine ratio after 4 weeks to assess treatment response",
    ]

    treatment_pathways = [
        {
            "gene": "GLDC",
            "disease": "Classical NKH (~80%)",
            "first_line": "Sodium benzoate 250-750 mg/kg/day (in 3-4 divided doses) to reduce plasma/CSF glycine; dextromethorphan 5-20 mg/kg/day as NMDA antagonist to block glycine hyperstimulation of NMDA receptors",
            "precision_therapy": "Pyridoxine trial (~20% respond); ketamine for acute crises (NMDA block); high-dose folate may help some; attenuated NKH: lower benzoate doses often sufficient",
            "avoid": "Vigabatrin (GABA-T inhibitor worsens glycine-to-GABA ratio); sodium valproate (impairs glycine catabolism); high-glycine formula feeds",
            "monitoring": "Plasma glycine (target <300 µmol/L); CSF glycine ratio at 3 and 12 months; EEG for burst-suppression resolution; developmental assessments; benzoate levels (hepatotoxic if overdosed)",
        },
        {
            "gene": "AMT",
            "disease": "Classical NKH (~10%)",
            "first_line": "Sodium benzoate + dextromethorphan (identical to GLDC); dose adjustment in attenuated variants (may need lower doses)",
            "precision_therapy": "Pyridoxine trial; attenuated AMT (e.g., p.G269D homozygous) may have better treatment response and milder seizure burden",
            "avoid": "Vigabatrin; valproate",
            "monitoring": "Identical to GLDC; molecular subtyping important for prognostic counselling (attenuated vs classic AMT variants)",
        },
        {
            "gene": "GCSH",
            "disease": "Classical NKH (<1%, extremely rare)",
            "first_line": "Sodium benzoate + dextromethorphan (identical to GLDC/AMT); symptomatic AED management for seizures",
            "precision_therapy": "Pyridoxine trial; very limited case data due to rarity; treatment extrapolated from GLDC/AMT experience",
            "avoid": "Vigabatrin; valproate",
            "monitoring": "As per GLDC/AMT; MLPA or full gene sequencing for large deletions (few pathogenic variants known)",
        },
        {
            "gene": "DLD",
            "disease": "DLD Deficiency (Combined E3 deficiency)",
            "first_line": "Thiamine 100-600 mg/day (cofactor for PDH and BCKDH components); riboflavin 100-300 mg/day (FAD-dependent DLD); BCAA-restricted diet for MSUD component; sodium benzoate for glycine component",
            "precision_therapy": "Dichloroacetate (DCA) for lactic acidosis crisis management (activates PDH complex); liver transplant considered for liver-dominant Ashkenazi phenotype (NOT curative for neurological/cardiac disease)",
            "avoid": "Fasting (triggers lactic acidosis crisis); high-protein meals (raises BCAA); N2O (further compromises multi-complex enzymes); hepatotoxic drugs",
            "monitoring": "Lactate + pyruvate monthly; alloisoleucine and BCAA levels; liver function (LFTs + imaging); cardiac echo (cardiomyopathy); brain MRI for Leigh-like basal ganglia lesions; DXA for osteoporosis from low-BCAA diet",
        },
        {
            "gene": "GLRA1",
            "disease": "Hyperekplexia Type 1",
            "first_line": "Clonazepam 0.1-0.2 mg/kg/day (potentiates GABA-A Cl- channel to compensate for GlyR alpha-1 loss); highly effective; response within days; lifelong treatment in most",
            "precision_therapy": "Umbilicus manoeuvre (flexion of head + limbs = Vigevano manoeuvre) for acute neonatal spasm termination; adults may taper clonazepam after childhood with monitoring",
            "avoid": "Do NOT use sodium benzoate or dextromethorphan (these are for NKH glycine accumulation, not hyperekplexia receptor loss); sedation without airway support in neonates (apnea risk)",
            "monitoring": "Startle frequency; apnea monitoring in neonates; developmental assessment; EEG to exclude epilepsy (hyperekplexia is non-epileptic); clonazepam levels and tolerance",
        },
        {
            "gene": "GLRB",
            "disease": "Hyperekplexia Type 2",
            "first_line": "Clonazepam (identical to GLRA1); highly effective",
            "precision_therapy": "Same acute management as GLRA1; molecular panel confirms GLRB vs GLRA1",
            "avoid": "NKH treatments (benzoate/dextromethorphan — not relevant)",
            "monitoring": "Identical to GLRA1 monitoring",
        },
        {
            "gene": "SLC6A5",
            "disease": "Hyperekplexia Type 3 (GlyT2)",
            "first_line": "Clonazepam (same as GLRA1/GLRB); highly effective; presynaptic glycine recycling defect compensated by GABA-A potentiation",
            "precision_therapy": "AD form (haploinsufficiency) typically milder; AR biallelic form more severe; clonazepam dose titrated to response",
            "avoid": "NKH treatments; high-glycine feeding not indicated (transporter defect not accumulation)",
            "monitoring": "CSF glycine (usually normal or mildly elevated — helps exclude NKH DDx); startle frequency; developmental tracking",
        },
        {
            "gene": "SLC6A9",
            "disease": "GlyT1 Encephalopathy",
            "first_line": "Symptomatic: sodium benzoate (may reduce CSF glycine); NMDA receptor modulators (dextromethorphan); glycine-restricted diet; sarcosine (N-methylglycine — GlyT1 substrate competitor)",
            "precision_therapy": "GCS enzyme assay in fibroblasts is KEY DDx step (GCS normal = SLC6A9, not true NKH); treatment experience very limited (<20 cases worldwide)",
            "avoid": "Clonazepam alone (encephalopathy, not hyperekplexia — different mechanism); vigabatrin",
            "monitoring": "CSF glycine ratio (goal <0.08); EEG for burst-suppression; GCS enzyme activity confirmation; NMDA receptor biomarkers if available; developmental outcomes",
        },
    ]

    return {
        "atlas_id":                 ATLAS_ID,
        "pathway_groups":           pathway_groups,
        "glycine_diagnostic_cascade": glycine_diagnostic_cascade,
        "treatment_pathways":       treatment_pathways,
        "seeds":                    f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "vigabatrin_valproate_warning": "Vigabatrin CONTRAINDICATED in NKH (GABA-T inhibition worsens glycine-to-GABA imbalance); valproate reduces glycine catabolism — both worsen glycine encephalopathy in GLDC/AMT/GCSH/DLD",
        "clonazepam_nkh_distinction": "Clonazepam is NOT first-line for NKH (GLDC/AMT/GCSH/DLD) — glycine NMDA receptor hyperstimulation is addressed by dextromethorphan; clonazepam is PATHOGNOMONIC TREATMENT in hyperekplexia (GLRA1/GLRB/SLC6A5)",
    }


def generate_definitions() -> dict[str, Any]:
    key_distinguishers = [
        {
            "name":        "Primary DDx Split: CSF glycine ratio",
            "description": "CSF:plasma glycine ratio >0.08 = NKH group (GLDC/AMT/GCSH/DLD) or SLC6A9 GlyT1 encephalopathy; ratio <0.04 = hyperekplexia group (GLRA1/GLRB/SLC6A5); this single ratio is the FIRST major branch point in glycine disorder diagnosis",
        },
        {
            "name":        "DLD lactic acidosis PATHOGNOMONIC within NKH",
            "description": "NKH + lactic acidosis simultaneously = DLD deficiency (L-protein, E3 subunit); GLDC/AMT/GCSH have NKH WITHOUT lactic acidosis; DLD affects GCS+PDH+KGDH+BCKDH simultaneously — combined organic/amino acid profile; alloisoleucine elevated in DLD (BCKDH component)",
        },
        {
            "name":        "GLDC vs AMT vs GCSH: molecular panel mandatory",
            "description": "Three GCS genes (GLDC, AMT, GCSH) are biochemically INDISTINGUISHABLE from each other; all show CSF:plasma glycine ratio >0.08 with normal organic acids; molecular gene panel is the ONLY way to assign gene-specific diagnosis; relevant for accurate carrier testing, recurrence risk, and potential gene-specific prognosis (AMT has more attenuated variants)",
        },
        {
            "name":        "SLC6A9 mimics NKH: GCS enzyme activity is the KEY DDx",
            "description": "SLC6A9 (GlyT1) deficiency elevates CSF glycine (mimics NKH) but GCS enzyme activity is NORMAL in fibroblasts/liver — this is the critical discriminator; if CSF glycine elevated + organic acids normal + GCS activity normal → SLC6A9; true NKH (GLDC/AMT/GCSH/DLD) has severely reduced GCS enzyme activity",
        },
        {
            "name":        "Hypertonia vs hypotonia: reverse tone DDx",
            "description": "NKH (GLDC/AMT/GCSH/DLD): HYPOTONIA (neonatal glycine causes inhibitory tone excess at brainstem/spinal cord via glycine receptors — paradoxically floppy); Hyperekplexia (GLRA1/GLRB/SLC6A5): HYPERTONIA (board-like rigidity — glycine receptor loss removes inhibitory brake on motor neurons); tone QUALITY is a bedside DDx clue",
        },
        {
            "name":        "Nose-tap test PATHOGNOMONIC for hyperekplexia",
            "description": "Firm tap to nasal bridge in hyperekplexia (GLRA1/GLRB/SLC6A5): sustained tonic contraction + head retraction = POSITIVE = PATHOGNOMONIC; does NOT habituate with repeated stimuli (normal startle habituates); NEGATIVE in NKH, benign neonatal sleep myoclonus, epileptic spasms; bedside test, no equipment required",
        },
        {
            "name":        "Burst-suppression EEG: NKH vs hyperekplexia vs benign neonatal",
            "description": "Burst-suppression on neonatal EEG = PATHOGNOMONIC for NKH (GLDC/AMT/GCSH) and SLC6A9; also seen in DLD; ABSENT in hyperekplexia group (GLRA1/GLRB/SLC6A5) — normal or non-specific EEG; burst-suppression must be documented in neonatal period as it may evolve to hypsarrhythmia or multifocal pattern later",
        },
        {
            "name":        "Clonazepam response = hyperekplexia; not NKH",
            "description": "GLRA1/GLRB/SLC6A5: clonazepam (BZD — GABA-A potentiator) is HIGHLY EFFECTIVE, often dramatically within days; NKH (GLDC/AMT/GCSH/DLD): clonazepam NOT effective as primary treatment (NMDA hyperstimulation by glycine requires NMDA antagonism, not GABAergic potentiation); excellent clonazepam response in a neonatal startle disorder = hyperekplexia until proven otherwise",
        },
        {
            "name":        "DLD liver transplant NOT curative",
            "description": "Unlike MSUD type I/II (where liver Tx cures BCKDH deficiency by replacing the liver enzyme), DLD is expressed in brain, heart, and muscle — liver Tx replaces the hepatic DLD but does not address neurological or cardiac DLD; liver Tx may stabilise hepatic phenotype (Ashkenazi G229C) but neurological progression continues; contrast with CBS (Tx NOT reported) and MSUD type I/II (Tx curative)",
        },
        {
            "name":        "Umbilicus manoeuvre for acute hyperekplexia neonatal spasm",
            "description": "VIGEVANO MANOEUVRE: firm flexion of the neonate's head and legs toward the trunk immediately terminates the hyperekplexia tonic spasm; mechanism: proprioceptive input via flexion inhibits the sustained alpha-motor neuron discharge; LIFE-SAVING in neonatal apnea from GLRA1/GLRB/SLC6A5 (apnea terminates with spasm termination); all parents/carers should be taught this manoeuvre",
        },
    ]

    treatment_matrix = [
        {
            "intervention":    "Sodium benzoate",
            "indications":     "NKH group (GLDC/AMT/GCSH/DLD); SLC6A9 GlyT1 encephalopathy",
            "mechanism":       "Conjugates glycine + benzoate → hippurate (excreted in urine); removes ~250 mg glycine per gram benzoate; reduces plasma and CSF glycine load",
            "dose":            "250-750 mg/kg/day in 3-4 divided doses; adjust to plasma glycine <300 µmol/L",
            "contraindicated": "NOT required in hyperekplexia (GLRA1/GLRB/SLC6A5) where glycine levels NORMAL; monitor for hepatotoxicity at high doses",
        },
        {
            "intervention":    "Dextromethorphan",
            "indications":     "NKH group (GLDC/AMT/GCSH/DLD); SLC6A9",
            "mechanism":       "Non-competitive NMDA receptor antagonist; blocks glycine co-agonist site hyperstimulation; reduces excitatory neurotoxicity from excess glycine at NMDA receptors in brain",
            "dose":            "5-20 mg/kg/day; titrate to seizure frequency and EEG response",
            "contraindicated": "NOT for hyperekplexia (receptor loss requires GABA-A support, not NMDA blockade)",
        },
        {
            "intervention":    "Clonazepam",
            "indications":     "Hyperekplexia group (GLRA1/GLRB/SLC6A5); highly effective, often lifelong",
            "mechanism":       "Benzodiazepine; potentiates GABA-A receptor (Cl- channel); compensates for lost GlyR-mediated inhibitory Cl- influx; restores inhibitory tone to spinal/brainstem motor circuits",
            "dose":            "0.1-0.2 mg/kg/day in neonates; titrate upward in older children; adults may taper",
            "contraindicated": "NOT primary treatment for NKH (different mechanism); respiratory depression risk in neonates — titrate carefully",
        },
        {
            "intervention":    "Pyridoxine (B6) trial",
            "indications":     "NKH group (GLDC/AMT/GCSH) — PLP-dependent GCS; ~20% response rate",
            "mechanism":       "GCS requires PLP as cofactor; pharmacological doses may partially restore residual enzyme activity; response measured by reduction in plasma/CSF glycine after 4-6 weeks",
            "dose":            "50-200 mg/day in neonates; MANDATORY trial before classifying as non-responsive",
            "contraindicated": "Not indicated in DLD (FAD/NAD+ dependent, not PLP); not indicated in hyperekplexia group",
        },
        {
            "intervention":    "Thiamine + riboflavin (DLD only)",
            "indications":     "DLD deficiency only (PDH + BCKDH components are thiamine-dependent)",
            "mechanism":       "Thiamine (B1) is cofactor for PDH E1 and BCKDH E1; riboflavin (B2) provides FAD for DLD L-protein; pharmacological doses may partially support residual complex activity",
            "dose":            "Thiamine 100-600 mg/day; riboflavin 100-300 mg/day",
            "contraindicated": "Not useful for GLDC/AMT/GCSH pure NKH",
        },
    ]

    definitions = [
        {
            "term": "Non-Ketotic Hyperglycinemia (NKH)",
            "definition": "Inborn error of glycine metabolism caused by defective glycine cleavage system (GCS) → massive glycine accumulation in plasma and CSF; 'non-ketotic' distinguishes from KETOTIC hyperglycinemia (PA/MMA/IVA) where glycine elevation is secondary; primary NKH: GLDC (~80%), AMT (~10%), GCSH (<1%), DLD (combined); CSF:plasma glycine ratio >0.08 PATHOGNOMONIC; neonatal presentation with burst-suppression, hiccupping, hypotonia, apnea",
        },
        {
            "term": "CSF:plasma glycine ratio",
            "definition": "The diagnostic ratio: (CSF glycine µmol/L) / (plasma glycine µmol/L); normal <0.02; NKH >0.08 (PATHOGNOMONIC — CSF glycine is disproportionately elevated because GCS normally maintains low CSF glycine via catabolism in brain mitochondria); ratio >0.08 with normal organic acids = NKH until proven otherwise; MUST be measured simultaneously (paired samples); false negatives if patient on sodium benzoate treatment",
        },
        {
            "term": "Glycine Cleavage System (GCS)",
            "definition": "Multi-enzyme complex in the inner mitochondrial membrane that degrades glycine: P-protein (GLDC, PLP-dependent decarboxylase) → H-protein (GCSH, lipoamide shuttle) → T-protein (AMT, THF-dependent aminomethyltransferase) → L-protein (DLD, FAD-dependent dihydrolipoamide dehydrogenase); end products: CO2 + NH3 + 5,10-methyleneTHF; deficiency of any component → glycine accumulates; major route of glycine catabolism in brain and liver",
        },
        {
            "term": "Hyperekplexia (Startle Disease)",
            "definition": "Disorder of glycinergic inhibitory neurotransmission at spinal cord/brainstem causing exaggerated, non-habituating startle response to acoustic or tactile stimuli → tonic spasm → rigidity; neonatal form: board-like hypertonia + apneas triggered by startle (life-threatening); NOSE-TAP TEST PATHOGNOMONIC; glycine levels NORMAL (signalling defect, not accumulation); mutations: GLRA1 (alpha-1 subunit, AD), GLRB (beta subunit, AR), SLC6A5 (GlyT2, presynaptic transporter); clonazepam highly effective",
        },
        {
            "term": "Nose-tap test (Vigevano test)",
            "definition": "Bedside diagnostic test for hyperekplexia: firm percussion to nasal bridge (or glabella) elicits sustained non-habituating tonic contraction with head retraction = POSITIVE = PATHOGNOMONIC for hyperekplexia; normal startle habituates after 2-3 taps; hyperekplexia startle does NOT habituate; absent in NKH (hypotonic, not hypertonically reactive); immediate family education: umbilicus manoeuvre terminates the spasm (sustained flexion of head + limbs)",
        },
        {
            "term": "Burst-suppression EEG",
            "definition": "Neonatal EEG pattern: alternating high-amplitude bursts of mixed frequency activity interspersed with periods of near-electrocerebral silence (suppression); PATHOGNOMONIC in NKH neonates (GLDC/AMT/GCSH and DLD/SLC6A9) — glycine hyperstimulation of NMDA receptors in immature brain causes oscillatory excitation; ABSENT in hyperekplexia group (GLRA1/GLRB/SLC6A5); also seen in hypoxic-ischaemic encephalopathy, drug-induced coma, severe metabolic encephalopathy (must exclude via organic acids/ammonia/glucose)",
        },
        {
            "term": "Sodium benzoate mechanism in NKH",
            "definition": "Sodium benzoate + glycine → hippurate (via glycine N-acyltransferase in liver/kidney mitochondria); hippurate excreted in urine, effectively removing glycine from the body; each mole of benzoate removes one mole of glycine; this glycine conjugation pathway is an alternative to the blocked GCS; dose 250-750 mg/kg/day reduces plasma glycine (target <300 µmol/L) and CSF glycine ratio; monitor benzoate levels (hepatotoxic if accumulated); must be given in multiple divided doses (short half-life)",
        },
        {
            "term": "Dextromethorphan as NMDA antagonist in NKH",
            "definition": "Glycine is a co-agonist at the NMDA receptor (binds glycine-B site, mandatory for NMDA channel opening); in NKH, excess CSF glycine hyperstimulates NMDA receptors → excitotoxicity → seizures + neuronal damage; dextromethorphan (DXM) is an uncompetitive NMDA receptor antagonist that reduces NMDA hyperstimulation; dose 5-20 mg/kg/day; combined with sodium benzoate (reduces glycine load); reduces seizure frequency and may improve developmental outcomes in attenuated NKH; NOT NMDA antagonist for hyperekplexia (different receptor, different mechanism)",
        },
        {
            "term": "Gephyrin and GlyR synaptic clustering",
            "definition": "Gephyrin is a scaffolding protein at inhibitory post-synaptic densities that anchors glycine receptor (GlyR) and GABA-A receptors to the cytoskeleton; GLRB (beta subunit) is required for gephyrin binding — without GLRB, GlyR heteromeric complexes cannot be stabilised at synapses; surface expression and clustering of GlyR alpha-1 depends on beta subunit + gephyrin interaction; GLRB LOF → mislocated, unstable GlyRs → reduced glycinergic inhibition → hyperekplexia; distinct from GLRA1 (alpha-1 channel gate) but identical clinical outcome",
        },
        {
            "term": "GlyT2 (SLC6A5) vs GlyT1 (SLC6A9): opposite functions",
            "definition": "GlyT2 (SLC6A5): presynaptic glycine transporter — REPLENISHES glycine into presynaptic glycinergic interneurons; uses Na+/Cl- gradient (3 Na+ : 1 Cl- : 1 glycine); GlyT2 LOF → presynaptic glycine store DEPLETED → glycinergic transmission fails → hyperekplexia (insufficient inhibitory signalling). GlyT1 (SLC6A9): astrocytic glycine transporter — CLEARS glycine from the synapse; GlyT1 LOF → glycine accumulates at synapse → NMDA co-agonist hyperstimulation → encephalopathy (NOT hyperekplexia — opposite phenotype to GlyT2 deficiency). KEY: GlyT2 deficiency = too little presynaptic glycine = startle; GlyT1 deficiency = too much synaptic glycine = encephalopathy",
        },
        {
            "term": "DLD combined deficiency: four complexes",
            "definition": "DLD (L-protein / E3 subunit / dihydrolipoamide dehydrogenase) is shared by four mitochondrial multi-enzyme complexes: (1) GCS (glycine cleavage), (2) PDH (pyruvate dehydrogenase complex, E3 is the third component), (3) KGDH (alpha-ketoglutarate dehydrogenase / oxoglutarate dehydrogenase), (4) BCKDH (branched-chain alpha-keto acid dehydrogenase). DLD LOF simultaneously impairs all four: glycine accumulation (GCS) + lactic acidosis (PDH) + 2-oxoglutarate accumulation (KGDH) + BCAA/alloisoleucine elevation (BCKDH). This COMBINED biochemical profile is PATHOGNOMONIC for DLD; GLDC/AMT/GCSH affect GCS only",
        },
        {
            "term": "Attenuated NKH",
            "definition": "Milder NKH phenotype with later onset (2-6 months, not neonatal), less severe intellectual disability, potentially treatment-responsive seizures; associated with partial residual GCS activity; more common in AMT than GLDC; may not have neonatal burst-suppression; cognitive outcomes better but not normal; sodium benzoate + dextromethorphan treatment often maintains better glycine control; prognosis more favourable than classic neonatal NKH; Japanese p.G269D (AMT) is a well-characterised attenuated variant",
        },
        {
            "term": "Ketotic hyperglycinemia (DDx NKH)",
            "definition": "Hyperglycinemia seen SECONDARY to organic acidemias: propionic acidemia (PA), methylmalonic acidemia (MMA), isovaleric acidemia (IVA) — these cause secondary glycine elevation via unclear mechanisms (propionyl-CoA competes with glycine conjugation, or mitochondrial toxicity impairs GCS). KEY DDx from NKH: KETOSIS present (NKH = NO ketosis); urine organic acids elevated (PA: propionylglycine, 3-OH-propionate; MMA: methylmalonate; IVA: isovalerylglycine); CSF:plasma glycine ratio typically <0.08 in ketotic hyperglycinemia vs >0.08 in true NKH; acylcarnitines elevated on NBS in PA/MMA/IVA",
        },
    ]

    return {
        "atlas_id":           ATLAS_ID,
        "key_distinguishers": key_distinguishers,
        "treatment_matrix":   treatment_matrix,
        "definitions":        definitions,
        "seeds":              f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
    }


if __name__ == "__main__":
    overview    = generate_overview()
    breakdown   = generate_breakdown()
    definitions = generate_definitions()

    print(json.dumps({
        "overview":    overview,
        "breakdown":   breakdown,
        "definitions": definitions,
    }, indent=2, default=str))
