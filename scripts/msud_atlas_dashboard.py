#!/usr/bin/env python3
"""MSUD-Atlas — Complete 8-Gene Maple Syrup Urine Disease & Branched-Chain Amino Acid Disorders Atlas
BCKDHA (E1α) · BCKDHB (E1β) · DBT (E2) · DLD (E3 / MSUD-III) ·
BCKDK (kinase, autism+epilepsy) · PPM1K (phosphatase, MSUD-V) ·
BCAT2 (transaminase) · SLC7A5/LAT1 (brain BCAA transporter)
320-patient aggregate cohort (8 × 40, seeds 942–949)

Maple Syrup Urine Disease (MSUD) facts:
  - MSUD = autosomal recessive inborn error of branched-chain amino acid (BCAA) catabolism.
    The BCKDH (branched-chain alpha-ketoacid dehydrogenase) complex is deficient in classic MSUD.
    BCAAs (leucine, isoleucine, valine) and their alpha-ketoacid derivatives accumulate.
  - Global incidence: ~1/185,000 (classic); ~1/26,000 in Mennonite populations.
  - KEY TEACHING POINTS:
      ALLOISOLEUCINE: allo-isoleucine is PATHOGNOMONIC — not a normal human amino acid;
        its presence on amino acid quantification confirms MSUD without molecular testing.
      LEUCINE NEUROTOXIC: leucine (not isoleucine or valine) drives cerebral edema and
        excitotoxicity; leucine monitoring is the key therapeutic target (goal <200 µmol/L).
      LIVER TRANSPLANT: curative for BCKDHA/B/DBT classic MSUD; the liver provides >95%
        systemic BCKDH activity; post-transplant, normal diet, leucine normalises.
      THIAMINE TRIAL: all BCKDH complex deficiencies require empiric thiamine (10–20 mg/kg/day)
        because TPP (thiamine pyrophosphate) is the cofactor for E1; ~5% of patients
        are thiamine-responsive (especially DBT and some BCKDHA variants).
      BCKDK INVERSE PHENOTYPE: BCKDK is the KINASE that INACTIVATES the BCKDH complex.
        Loss-of-function BCKDK → BCKDH hyperactive → BCAAs abnormally LOW.
        Phenotype: autism + intellectual disability + seizures + epilepsy.
        Treatment: LEUCINE SUPPLEMENTATION (raises BCAA levels to normal).
      DLD MULTI-COMPLEX: DLD encodes E3, shared by PDC, BCKDH, and 2-OGDC complexes.
        E3 deficiency → MSUD (branched-chain) + lactic acidosis (PDC) + 2-OGA (2-OGDC)
        simultaneously; MMA mild secondary; multiple-complex fingerprint on organic acids.
      KD CONTRAINDICATED: ketogenic diet is CONTRAINDICATED in BCKDHA/B/DBT/DLD classic MSUD
        because fat catabolism + muscle BCAA release during ketosis raises leucine levels.
      NBS: elevated leucine (+ valine + isoleucine) on MS/MS newborn screen;
        NBS DOES NOT detect BCKDK deficiency (BCAAlow, not elevated).

COHORT: 8 × 40 = 320 patient slots (seeds 942–949; gene-specific seeds)
"""

import random

SEED_BASE = 942

MSUD_GENES = [
    # ── BCKDHA — E1α subunit — MSUD type Ia ─────────────────────────────────────────
    {
        "gene": "BCKDHA", "protein": "BCKDHase E1α",
        "alias": "BCKDHA/E1α — MSUD type Ia, most common classic MSUD (OMIM #248600)",
        "aa": "445 aa", "kDa": "50 kDa",
        "gene_class": "BCKDHase E1 alpha-subunit: decarboxylase component of the branched-chain alpha-ketoacid dehydrogenase (BCKDH) multi-enzyme complex; thiamine pyrophosphate (TPP)-dependent; catalyses oxidative decarboxylation of alpha-ketoisovalerate (Val), alpha-ketoisocaproate (Leu), and alpha-keto-beta-methylvalerate (Ile)",
        "msud_subgroup": "Classic MSUD (MSUD type Ia)",
        "locus": "19q13.2", "omim_gene": 608348,
        "inheritance": "AR. 19q13.2. Both sexes equally. Mennonite founder p.Tyr438Asn (>95% of Mennonite MSUD alleles; incidence 1/358 newborns in Lancaster County PA).",
        "onset_range_y": (0.0, 0.02),
        "phenotype": (
            "Classic MSUD neonatal encephalopathy: normal at birth → progressive lethargy, poor feeding, "
            "maple syrup odor (urine, cerumen, sweat) by days 3-5 → cerebral edema → coma → "
            "death if untreated by 2-4 weeks; survivors with intellectual disability proportional "
            "to leucine exposure duration and peak."
        ),
        "disease": (
            "BCKDHA encodes the E1α subunit (445aa, 50kDa) of the BCKDH multi-enzyme complex. "
            "The BCKDH complex catalyses the second step of BCAA catabolism: oxidative decarboxylation "
            "of the three branched-chain alpha-ketoacids (BCKAs) — alpha-ketoisocaproate (KIC, from Leu), "
            "alpha-keto-beta-methylvalerate (KMV, from Ile), alpha-ketoisovalerate (KIV, from Val) — "
            "to the corresponding acyl-CoA derivatives. TPP (thiamine pyrophosphate) is the required "
            "cofactor for E1. Loss of E1α → complete BCKDH block → plasma Leu, Ile, Val and their "
            "BCKAs accumulate. Alloisoleucine (allo-Ile) appears: formed spontaneously from KMV "
            "accumulation in plasma; PATHOGNOMONIC — no other condition generates allo-Ile.\n\n"
            "BCKDHA is the most commonly mutated gene in classic MSUD, accounting for ~45-50% of cases. "
            "Over 130 pathogenic variants reported (missense, nonsense, frameshift, splicing). "
            "Mennonite founder p.Tyr438Asn (c.1312T>A): ~95-99% of Old Order Mennonite MSUD alleles. "
            "Incidence in Lancaster County PA Mennonite community: 1/358 newborns. "
            "NBS: elevated leucine (with isoleucine + valine) on tandem mass spectrometry (MS/MS); "
            "BCKDHA NBS median leucine ~3,200 µmol/L at day 4-5 in untreated newborns "
            "(normal <150 µmol/L). BCAA ratio (Leu/Ala+Glu) enhances specificity. "
            "Emergency treatment: IV dextrose (GIR 8-12 mg/kg/min anti-catabolic) + BCAA-free amino "
            "acid formula (enteral or PN) + leucine-specific target <200 µmol/L rapidly. "
            "Thiamine (50 mg/day × 3 weeks trial): ~5% of BCKDHA cases partially thiamine-responsive "
            "(usually mild variants with some residual BCKDH activity). "
            "Liver transplant: curative — normalises plasma leucine; allows unrestricted diet; "
            "does not reverse pre-existing neurological damage. Best outcomes: transplant before crisis."
        ),
        "hallmark": (
            "BCKDHA HALLMARKS: "
            "(1) MSUD TYPE Ia — most common cause (~45%) of classic MSUD. "
            "(2) ALLOISOLEUCINE PATHOGNOMONIC — appears in ALL MSUD types (1a-V), confirms diagnosis. "
            "(3) MENNONITE FOUNDER — p.Tyr438Asn accounts for >95% Mennonite MSUD alleles. "
            "(4) LEUCINE NEUROTOXIC — leucine specifically causes cerebral edema (not Ile or Val). "
            "(5) LIVER TRANSPLANT CURATIVE — liver provides >95% systemic BCKDH. "
            "(6) KD CONTRAINDICATED — fat catabolism + BCAA release worsens leucine levels. "
            "(7) VPA HIGH RISK — organic acid accumulation; prefer LEV. "
            "(8) THIAMINE TRIAL MANDATORY — 5% response rate; empiric 50 mg/day × 3 weeks."
        ),
        "diet_treatment": "BCAA-free amino acid formula (Leu/Ile/Val titration to normal plasma); leucine <200 µmol/L target; no protein restriction if all BCAA titrated. Liver transplant eliminates dietary restriction.",
        "nbs_marker": "Elevated leucine (MS/MS Day 2-4); allo-isoleucine on amino acid chromatography; leucine/alanine ratio",
        "key_biomarker": "Alloisoleucine (allo-Ile, C6H13NO2) — PATHOGNOMONIC; plasma amino acid quantification; urine organic acids (alpha-ketoisocaproate, alpha-hydroxyisocaproate = sotolone precursor)",
        "severity_spectrum": "Neonatal encephalopathy if untreated (all), cerebral oedema, coma → death within 2-4 weeks; with treatment: normal IQ possible if leucine controlled",
        "founder_variant": "p.Tyr438Asn — Old Order Mennonite founder; incidence 1/358 Mennonite newborns",
        "thiamine_responsive": False,
        "liver_transplant_curative": True,
        "kd_contraindicated": True,
        "alloisoleucine_positive": True,
        "patients": [],
        "n_patients": 40,
    },
    # ── BCKDHB — E1β subunit — MSUD type Ib ─────────────────────────────────────────
    {
        "gene": "BCKDHB", "protein": "BCKDHase E1β",
        "alias": "BCKDHB/E1β — MSUD type Ib (OMIM #248600)",
        "aa": "342 aa", "kDa": "37 kDa",
        "gene_class": "BCKDHase E1 beta-subunit: second subunit of the decarboxylase E1 component; heterotetrameric alpha2-beta2 complex; structural-regulatory role in E1 assembly and TPP binding stabilisation",
        "msud_subgroup": "Classic MSUD (MSUD type Ib)",
        "locus": "6q14.1", "omim_gene": 248611,
        "inheritance": "AR. 6q14.1. Both sexes equally. No single dominant founder (multiple ethnic-specific alleles).",
        "onset_range_y": (0.0, 0.02),
        "phenotype": (
            "Classic MSUD: phenotypically identical to type Ia — neonatal encephalopathy, maple syrup "
            "odor, cerebral edema, alloisoleucine elevation; distinguished by genotyping only."
        ),
        "disease": (
            "BCKDHB encodes the E1β subunit (342aa, 37kDa) of the BCKDH complex. "
            "E1 is a heterotetrameric enzyme (α2β2): both subunits required for activity. "
            "E1β stabilises the alpha2beta2 tetramer and TPP (thiamine pyrophosphate) cofactor binding "
            "site — mutations in E1β prevent TPP engagement, abolishing decarboxylase function. "
            "Accounts for ~35% of classic MSUD cases — the second most common MSUD gene. "
            "Clinical phenotype is IDENTICAL to type Ia at presentation; distinction requires genotyping. "
            "Over 80 pathogenic variants known. No dominant founder in most populations, but "
            "p.Gln108Arg enriched in Middle Eastern populations; p.Arg183Trp in Japanese patients. "
            "Thiamine responsiveness: similar to Ia (~5-10%); trial mandatory. "
            "Liver transplant: equally curative as for type Ia (all hepatic BCKDH restored). "
            "Enzyme activity: BCKDH complex activity in cultured fibroblasts, lymphocytes; "
            "molecular confirmation by gene sequencing. "
            "Metabolic monitoring: plasma leucine + isoleucine + valine + allo-isoleucine; "
            "24-h urine BCKAs (alpha-ketoisocaproate, alpha-keto-beta-methylvalerate, alpha-ketoisovalerate). "
            "EEG: burst-suppression in neonatal MSUD encephalopathy; normalises with metabolic control."
        ),
        "hallmark": (
            "BCKDHB HALLMARKS: "
            "(1) MSUD TYPE Ib — second most common MSUD cause (~35%). "
            "(2) PHENOTYPICALLY IDENTICAL TO Ia — distinguish by genotyping. "
            "(3) E1β STABILISES α2β2 TETRAMER — structural + TPP-binding support. "
            "(4) ALLOISOLEUCINE PATHOGNOMONIC — present in all classic MSUD. "
            "(5) LIVER TRANSPLANT EQUALLY CURATIVE. "
            "(6) KD CONTRAINDICATED. "
            "(7) THIAMINE TRIAL MANDATORY — empiric 50 mg/day × 3 weeks."
        ),
        "diet_treatment": "BCAA-free amino acid formula; leucine target <200 µmol/L; liver transplant eliminates dietary restriction.",
        "nbs_marker": "Elevated leucine (MS/MS); allo-isoleucine; elevated Leu/Ala ratio",
        "key_biomarker": "Alloisoleucine; plasma Leu >200 µmol/L (crisis >1000 µmol/L); urine BCKAs",
        "severity_spectrum": "Classic MSUD neonatal encephalopathy; identical prognosis to type Ia",
        "founder_variant": "p.Gln108Arg — Middle Eastern; p.Arg183Trp — Japanese",
        "thiamine_responsive": False,
        "liver_transplant_curative": True,
        "kd_contraindicated": True,
        "alloisoleucine_positive": True,
        "patients": [],
        "n_patients": 40,
    },
    # ── DBT — E2 acetyltransferase — MSUD type II ────────────────────────────────────
    {
        "gene": "DBT", "protein": "BCKDH E2",
        "alias": "DBT/E2 — MSUD type II, dihydrolipoamide branched-chain transacylase (OMIM #248610)",
        "aa": "482 aa", "kDa": "53 kDa",
        "gene_class": "BCKDHase E2 subunit: dihydrolipoamide S-acyltransferase; forms 24-subunit cube core of BCKDH complex; anchors E1 and E3; lipoic acid scaffold for acyl group transfer from E1 to CoA",
        "msud_subgroup": "MSUD type II",
        "locus": "1p21.2", "omim_gene": 248610,
        "inheritance": "AR. 1p21.2. Both sexes equally. Various ethnic-specific variants.",
        "onset_range_y": (0.0, 0.02),
        "phenotype": (
            "Primarily classic MSUD (phenotypically similar to types Ia/Ib); some DBT variants "
            "retain partial E2 activity producing intermediate or mild MSUD phenotypes. "
            "Thiamine-responsive subgroup: ~5-10% of DBT cases."
        ),
        "disease": (
            "DBT encodes the E2 subunit (482aa, 53kDa; dihydrolipoamide S-acyltransferase). "
            "E2 forms the icosahedral 24-mer cube core of the BCKDH multi-enzyme complex. "
            "The E2 core simultaneously: (1) anchors E1 (decarboxylase) and E3 (dihydrolipoamide "
            "dehydrogenase) via protein-protein interaction domains; (2) carries the lipoyl domain "
            "(lipoic acid scaffold) that swings between E1, E2 active site, and E3 during catalysis. "
            "Loss of E2 → entire BCKDH complex disassembly → complete BCKA block. "
            "DBT accounts for ~10-15% of classic MSUD cases; third most common gene. "
            "Key distinguishing feature: some DBT missense variants retain partial E2 activity "
            "→ INTERMEDIATE MSUD (onset in infancy 3-6 months) or MILD MSUD (episodic with "
            "leucine elevation during catabolic illness). "
            "Thiamine responsiveness: slightly higher than Ia/Ib (~10-15%); TPP stabilises "
            "the E1-E2 interface when E2 folds sub-optimally; empiric thiamine trial recommended. "
            "E2 missense variants in the lipoyl-binding region: most severe (no cofactor loading). "
            "E2 variants in peripheral C-terminal domain (E1/E3 binding): variable residual function. "
            "Liver transplant: curative for all DBT MSUD types (hepatic BCKDH fully restored). "
            "Enzyme assay: BCKDH complex activity + E2-specific lipoic acid content (immunoblot lipoic acid). "
            "Maple syrup odor: sotolone (3-hydroxy-4,5-dimethyl-2(5H)-furanone) — same in all MSUD types."
        ),
        "hallmark": (
            "DBT HALLMARKS: "
            "(1) E2 CORE SCAFFOLD — 24-mer icosahedral cube that positions E1 and E3. "
            "(2) MSUD TYPE II — 10-15% of classic MSUD. "
            "(3) INTERMEDIATE/MILD VARIANTS — partial E2 activity → later-onset episodic MSUD. "
            "(4) THIAMINE RESPONSIVE SUBSET (~10-15%) — TPP stabilises E1-E2 interface. "
            "(5) LIPOYL DOMAIN CRITICAL — lipoic acid scaffold on E2 carries acyl group. "
            "(6) LIVER TRANSPLANT CURATIVE. "
            "(7) KD CONTRAINDICATED."
        ),
        "diet_treatment": "BCAA-free amino acid formula; leucine target <200 µmol/L; thiamine trial 50-100 mg/day × 3 weeks for all; liver transplant curative.",
        "nbs_marker": "Elevated leucine (MS/MS); allo-isoleucine; some mild variants NBS borderline",
        "key_biomarker": "Allo-isoleucine; plasma BCAA; urine BCKAs; E2 lipoic acid immunoblot",
        "severity_spectrum": "Severe classic (null) → Intermediate (missense) → Mild/episodic (partial E2 function)",
        "founder_variant": "No single dominant founder; multiple ethnic-specific alleles",
        "thiamine_responsive": True,
        "liver_transplant_curative": True,
        "kd_contraindicated": True,
        "alloisoleucine_positive": True,
        "patients": [],
        "n_patients": 40,
    },
    # ── DLD — E3 subunit — MSUD type III / E3 deficiency ────────────────────────────
    {
        "gene": "DLD", "protein": "Dihydrolipoamide Dehydrogenase (E3)",
        "alias": "DLD/E3 — MSUD type III / E3 deficiency: TRIPLE COMPLEX (PDC + BCKDH + 2-OGDC) (OMIM #246900)",
        "aa": "509 aa", "kDa": "55 kDa",
        "gene_class": "Dihydrolipoamide dehydrogenase (E3): FAD-dependent oxidoreductase; shared E3 component of THREE mitochondrial alpha-ketoacid dehydrogenase complexes: PDC (pyruvate), BCKDH (branched-chain), and 2-OGDC (2-oxoglutarate); also a component of the glycine cleavage system (GCS)",
        "msud_subgroup": "MSUD type III (E3 deficiency) — multi-complex disorder",
        "locus": "7q31.1", "omim_gene": 238331,
        "inheritance": "AR. 7q31.1. Both sexes equally. Ashkenazi Jewish founder p.Gly136Asp (pGly136Asp, c.407G>A): incidence ~1/35,000 to 1/94,000 in Ashkenazi Jewish population.",
        "onset_range_y": (0.0, 0.5),
        "phenotype": (
            "TRIPLE-COMPLEX phenotype: MSUD features (BCAA elevation) + lactic acidosis (PDC defect) + "
            "2-oxoglutaricaciduria (2-OGDC defect) SIMULTANEOUSLY. Unique multi-organic acid pattern. "
            "Hepatopathy (55% of cases, especially Ashkenazi p.Gly136Asp) — KEY DDx: other MSUD types "
            "have NO hepatopathy. Neonatal to infantile onset; severity inversely proportional to "
            "residual E3 activity."
        ),
        "disease": (
            "DLD encodes dihydrolipoamide dehydrogenase (E3, 509aa, 55kDa, FAD-dependent). "
            "E3 is the SHARED final oxidation component of THREE mitochondrial alpha-ketoacid "
            "dehydrogenase complexes:\n"
            "  (1) PDC (pyruvate dehydrogenase complex) — E3 reoxidises PDC-E2 lipoyl domain\n"
            "  (2) BCKDH (branched-chain alpha-ketoacid dehydrogenase) — E3 reoxidises BCKDH-E2\n"
            "  (3) 2-OGDC (2-oxoglutarate dehydrogenase complex) — E3 reoxidises 2-OGDC-E2\n"
            "E3 is also the L-protein component of the Glycine Cleavage System (GCS).\n\n"
            "E3 deficiency uniquely impairs ALL FOUR systems simultaneously → biochemical fingerprint:\n"
            "  - Elevated plasma BCAA + allo-isoleucine (BCKDH block)\n"
            "  - Elevated plasma lactate + pyruvate (PDC block)\n"
            "  - Elevated urine 2-oxoglutarate + 2-hydroxyglutarate (2-OGDC block)\n"
            "  - Mildly elevated plasma glycine (GCS impairment)\n\n"
            "Ashkenazi Jewish founder p.Gly136Asp (c.407G>A): the most common E3 mutation globally; "
            "incidence 1:35,000-94,000 Ashkenazi Jewish births. This variant causes severe E3 "
            "deficiency with hepatopathy (55%) — liver involvement DISTINGUISHES E3 deficiency from "
            "other MSUD types which are hepatopathy-free. "
            "Additional features unique to E3 deficiency vs other MSUD types:\n"
            "  - Hepatopathy/liver disease (55%) — not seen in MSUD Ia, Ib, II, V\n"
            "  - Lactic acidosis: primary feature (PDC component)\n"
            "  - Mild MMA elevation (secondary to propionyl-CoA accumulation from BCAA catabolism backup)\n"
            "  - 2-oxoglutaric aciduria on urine organic acids\n\n"
            "Thiamine: partial benefit for E3 deficiency (stabilises BCKDH E1 component); "
            "riboflavin (FAD precursor) trial also warranted (E3 is FAD-dependent). "
            "Liver transplant: DOES NOT CURE E3 deficiency (unlike MSUD Ia/Ib/II) because E3 is "
            "ubiquitously expressed; hepatic correction insufficient for systemic PDC/2-OGDC function. "
            "KD: contraindicated (worsens BCAA and ketoacid load)."
        ),
        "hallmark": (
            "DLD HALLMARKS: "
            "(1) TRIPLE COMPLEX — E3 shared by PDC+BCKDH+2-OGDC; triple organic acid fingerprint. "
            "(2) ASHKENAZI FOUNDER p.Gly136Asp — most common DLD variant worldwide. "
            "(3) HEPATOPATHY 55% — DISTINGUISHES DLD from all other MSUD types (MSUD Ia/Ib/II have NO hepatopathy). "
            "(4) LACTIC ACIDOSIS PRIMARY — PDC block; leucine elevated secondarily. "
            "(5) 2-OXOGLUTARICACIDURIA — 2-OGDC block; urine organic acids pathognomonic triad. "
            "(6) LIVER TRANSPLANT NOT CURATIVE — unlike MSUD Ia/Ib/II; E3 ubiquitous. "
            "(7) GCS IMPAIRMENT — mild glycine elevation (not NKH). "
            "(8) RIBOFLAVIN TRIAL — FAD-dependent enzyme; riboflavin + thiamine both warranted."
        ),
        "diet_treatment": "Low-BCAA diet (less restrictive than classic MSUD); low-protein diet; thiamine 50 mg/day + riboflavin 100 mg/day trials; avoid fasting; high GIR during illness.",
        "nbs_marker": "Elevated leucine + lactic acidosis combination; urine 2-OGA; allo-isoleucine",
        "key_biomarker": "Allo-isoleucine + lactic acidosis + 2-oxoglutaric aciduria TRIAD; Ashkenazi Jewish ancestry; hepatopathy",
        "severity_spectrum": "Severe neonatal (null alleles) → moderate infantile → mild (partial residual E3 activity; Ashkenazi p.Gly136Asp intermediate severity)",
        "founder_variant": "p.Gly136Asp (c.407G>A) — Ashkenazi Jewish; 1/35,000-94,000 births",
        "thiamine_responsive": True,
        "liver_transplant_curative": False,
        "kd_contraindicated": True,
        "alloisoleucine_positive": True,
        "patients": [],
        "n_patients": 40,
    },
    # ── BCKDK — BCKDHase kinase — INVERSE: autism + epilepsy + BCAA-low ────────────
    {
        "gene": "BCKDK", "protein": "BCKDHase Kinase",
        "alias": "BCKDK — Branched-chain keto acid dehydrogenase kinase deficiency: autism + epilepsy + LOW BCAA (OMIM #614923)",
        "aa": "412 aa", "kDa": "46 kDa",
        "gene_class": "BCKDH regulatory kinase: phosphorylates E1α-Ser293 → INACTIVATES BCKDH complex; controls BCAA catabolism rate by ON/OFF switching; constitutively active under nutrient-replete conditions",
        "msud_subgroup": "BCKDK deficiency — INVERSE phenotype (LOW BCAA), not classic MSUD",
        "locus": "16q24.3", "omim_gene": 614920,
        "inheritance": "AR. 16q24.3. Both sexes equally. Rare; <50 families reported worldwide.",
        "onset_range_y": (0.0, 3.0),
        "phenotype": (
            "INVERSE MSUD phenotype: autism spectrum disorder (ASD) + intellectual disability (ID) + "
            "epilepsy + EEG abnormalities PLUS LOW/LOW-NORMAL plasma BCAAs (leucine, isoleucine, valine). "
            "NO maple syrup odor. NO allo-isoleucine. NBS DOES NOT detect. "
            "Treatment: LEUCINE SUPPLEMENTATION (raises BCAA to normal range; may improve ASD + seizures)."
        ),
        "disease": (
            "BCKDK encodes branched-chain alpha-keto acid dehydrogenase kinase (412aa, 46kDa). "
            "BCKDK is the REGULATORY KINASE of the BCKDH complex:\n"
            "  - BCKDK ON → phosphorylates E1α at Ser293 → INACTIVATES BCKDH → BCAAs not catabolised → plasma BCAAs NORMAL\n"
            "  - BCKDK OFF → E1α dephosphorylated (by PPM1K phosphatase) → BCKDH ACTIVE → BCAAs catabolised\n\n"
            "BCKDK loss-of-function → BCKDH constitutively HYPERACTIVE → BCAAs catastrophically "
            "over-catabolised → plasma BCAAs ABNORMALLY LOW → brain BCAA deprivation.\n\n"
            "Brain BCAA deprivation mechanism: leucine and isoleucine are precursors for brain "
            "mTOR activation and neurotrophic factor synthesis. Severely low brain leucine:\n"
            "  - Impairs mTOR signalling → reduced synaptic protein synthesis\n"
            "  - Depletes neurotransmitter precursors (glutamate from alpha-KG via BCAT)\n"
            "  - Causes epilepsy via GABAergic/glutamatergic imbalance\n\n"
            "Clinical features: autism spectrum disorder (ASD) in most; intellectual disability (ID); "
            "epilepsy (focal or generalised); EEG: focal spikes, generalised discharges, hypsarrhythmia "
            "in some; structural MRI: variable cortical atrophy, simplified gyri in severe cases. "
            "Plasma amino acids: leucine and isoleucine notably LOW (may be within normal reference "
            "range in mild cases but LOW compared to an individual's own baseline). "
            "No allo-isoleucine. No maple syrup odor. NOT detected by NBS (no elevation).\n\n"
            "Treatment: LEUCINE SUPPLEMENTATION (0.5-1 g/kg/day elemental leucine) → raises plasma "
            "BCAA → may dramatically improve ASD behaviours, seizure frequency, and cognitive function. "
            "Protein-enriched diet also rationale. EEG should be monitored for improvement with "
            "leucine supplementation. mTOR agonist (leucine acts via mTOR): this is the mechanism "
            "for leucine supplementation benefit. "
            "Ketogenic diet: effect uncertain in BCKDK (different mechanism than classic MSUD)."
        ),
        "hallmark": (
            "BCKDK HALLMARKS: "
            "(1) KINASE NOT COMPLEX — loss of kinase → hyperactive BCKDH → BCAAs OVER-catabolised. "
            "(2) INVERSE PHENOTYPE — LOW BCAA + autism + epilepsy (OPPOSITE of classic MSUD). "
            "(3) NBS MISSES — no elevation; borderline-low leucine may not trigger alert. "
            "(4) NO ALLOISOLEUCINE — no BCKA accumulation (BCKDH hyperactive, not blocked). "
            "(5) LEUCINE SUPPLEMENTATION TREATMENT — raises BCAAs; may reverse ASD+seizures. "
            "(6) mTOR LINK — leucine is key mTOR activator; brain mTOR impaired when leucine LOW. "
            "(7) EEG IMPROVEMENT WITH TREATMENT — monitor EEG response to leucine supplementation. "
            "(8) EPILEPSY RELEVANT — a treatable metabolic epilepsy; crucial to diagnose early."
        ),
        "diet_treatment": "LEUCINE SUPPLEMENTATION (not restriction): 0.5-1 g/kg/day elemental leucine; protein-enriched diet; avoid protein restriction. Monitor plasma leucine to 150-300 µmol/L target.",
        "nbs_marker": "NOT detected by NBS (low or low-normal leucine does not trigger standard cutoffs); often diagnosed by WES/WGS after ASD + epilepsy workup",
        "key_biomarker": "Low/low-normal plasma leucine + isoleucine + valine; NO allo-isoleucine; ASD + epilepsy phenotype",
        "severity_spectrum": "Moderate-severe ASD + ID + epilepsy in most reported cases; leucine supplementation response variable",
        "founder_variant": "Multiple private variants; no single founder (very rare disease)",
        "thiamine_responsive": False,
        "liver_transplant_curative": False,
        "kd_contraindicated": False,
        "alloisoleucine_positive": False,
        "patients": [],
        "n_patients": 40,
    },
    # ── PPM1K — BCKDH phosphatase — MSUD type V ──────────────────────────────────────
    {
        "gene": "PPM1K", "protein": "PP2Cm / BCKDH Phosphatase",
        "alias": "PPM1K/PP2Cm — MSUD type V, BCKDH phosphatase deficiency (OMIM #615135)",
        "aa": "372 aa", "kDa": "42 kDa",
        "gene_class": "Protein phosphatase Mg2+/Mn2+-dependent 1K (PP2Cm): mitochondrial protein phosphatase; dephosphorylates E1α-pSer293 → ACTIVATES BCKDH complex; the counterpart phosphatase to BCKDK kinase",
        "msud_subgroup": "MSUD type V (phosphatase deficiency, rare)",
        "locus": "4q22.1", "omim_gene": 611015,
        "inheritance": "AR. 4q22.1. Both sexes equally. Extremely rare; <25 reported families worldwide.",
        "onset_range_y": (0.0, 0.1),
        "phenotype": (
            "MSUD-like phenotype but MILDER than classic MSUD; biochemical similarity to intermediate MSUD. "
            "Elevated BCAAs + allo-isoleucine (confirms BCKDH block). PPM1K deficiency prevents BCKDH "
            "activation → BCKDH remains phosphorylated (inactive) → BCAAs accumulate. "
            "Mild-moderate phenotype in most reported cases."
        ),
        "disease": (
            "PPM1K encodes PP2Cm (protein phosphatase 2C mitochondrial, 372aa, 42kDa), the ACTIVATING "
            "phosphatase of the BCKDH complex. PP2Cm dephosphorylates E1α-pSer293 → BCKDH becomes "
            "ACTIVE (opposite of BCKDK kinase action). PPM1K and BCKDK form a kinase-phosphatase "
            "switch that controls BCAA catabolism in response to nutrient state:\n"
            "  - Fed state: BCKDK active → E1α phosphorylated → BCKDH inactive → BCAAs preserved\n"
            "  - Fasting/exercise: PPM1K active → E1α dephosphorylated → BCKDH active → BCAAs catabolised\n\n"
            "PPM1K loss-of-function → BCKDH CANNOT be activated → BCKDH constitutively INACTIVE → "
            "BCAAs ACCUMULATE even in catabolic states. Mechanism differs from MSUD Ia/Ib/II/III "
            "(those have structural BCKDH complex defects); PPM1K MSUD has an INTACT BCKDH complex "
            "that simply cannot be unphosphorylated (switched on).\n\n"
            "Clinical features: MSUD-like biochemistry (elevated Leu + Ile + Val + allo-Ile) but "
            "generally MILDER than classic MSUD because some residual PPM1K-independent dephosphorylation "
            "occurs (PP2A can partially substitute). Mild hyperleucineaemia in most; episodic metabolic "
            "crises during catabolic illness (infection, fasting). Thiamine partially effective "
            "(TPP cofactor; BCKDH complex structure intact). "
            "Liver transplant: rationale exists (hepatic BCKDH control restored) but limited data. "
            "Allo-isoleucine: PRESENT (BCKDH block persists). "
            "KD: contraindicated (BCAA release from fat catabolism worsens leucine). "
            "NBS: elevated leucine — may be below classic MSUD cutoffs in mild cases; re-test protocol needed."
        ),
        "hallmark": (
            "PPM1K HALLMARKS: "
            "(1) PHOSPHATASE NOT COMPLEX — BCKDH complex structurally intact but cannot activate. "
            "(2) MSUD TYPE V — rarest classic MSUD subtype; intact complex, impaired activation. "
            "(3) MILDER THAN Ia/Ib/II — some residual dephosphorylation (PP2A substitution). "
            "(4) KINASE-PHOSPHATASE SWITCH — PPM1K activates BCKDH; BCKDK inactivates. "
            "(5) ALLOISOLEUCINE POSITIVE — confirms BCKDH block (even partial). "
            "(6) THIAMINE RATIONAL — TPP cofactor; BCKDH complex structurally intact. "
            "(7) KD CONTRAINDICATED. "
            "(8) EPISODIC CRISES — catabolic illness precipitates leucine spikes."
        ),
        "diet_treatment": "BCAA-restricted diet (leucine target <200 µmol/L); less restrictive than classic MSUD in mild cases; thiamine trial 50 mg/day.",
        "nbs_marker": "Elevated leucine (MS/MS); may be borderline; allo-isoleucine confirms",
        "key_biomarker": "Allo-isoleucine; plasma leucine >200 µmol/L (milder than classic); kinase-phosphatase ratio studies",
        "severity_spectrum": "Mild-moderate (most); severe neonatal crisis possible in null alleles",
        "founder_variant": "No single founder; very rare disease",
        "thiamine_responsive": True,
        "liver_transplant_curative": True,
        "kd_contraindicated": True,
        "alloisoleucine_positive": True,
        "patients": [],
        "n_patients": 40,
    },
    # ── BCAT2 — BCAA transaminase 2 — hypervalinemia / hyperBCAA ────────────────────
    {
        "gene": "BCAT2", "protein": "BCAA Aminotransferase 2 (Mitochondrial)",
        "alias": "BCAT2 — Branched-chain amino acid aminotransferase 2 (mitochondrial) deficiency — hypervalinemia-hyperleucine-isoleucinemia (OMIM #113530 / BCAT2)",
        "aa": "392 aa", "kDa": "44 kDa",
        "gene_class": "BCAA aminotransferase 2 (mitochondrial): PLP (pyridoxal-5'-phosphate)-dependent transaminase; catalyses the first step of BCAA catabolism — transamination of leucine, isoleucine, and valine to their respective BCKAs (keto acids) — upstream of the BCKDH complex; provides the substrate for BCKDH",
        "msud_subgroup": "Hypervalinemia-hyperleucine-isoleucinemia (BCAT2 deficiency, pre-BCKDH block)",
        "locus": "19q13.2", "omim_gene": 113530,
        "inheritance": "AR. 19q13.2. Both sexes equally. Extremely rare; fewer than 10 families reported.",
        "onset_range_y": (0.0, 0.5),
        "phenotype": (
            "Elevated plasma BCAA (leucine + isoleucine + valine) WITHOUT elevated BCKAs and WITHOUT "
            "allo-isoleucine. Intellectual disability, growth failure, feeding difficulties. "
            "BCAT2 deficiency is PRE-BCKDH: BCAAs elevated because they cannot be transaminated "
            "to BCKAs → BCKAs NOT elevated → BCKDH not overloaded → NO allo-isoleucine."
        ),
        "disease": (
            "BCAT2 encodes mitochondrial BCAA aminotransferase 2 (392aa, 44kDa; PLP-dependent). "
            "BCAT2 catalyses the FIRST step of BCAA catabolism:\n"
            "  Leucine + alpha-KG → KIC (alpha-ketoisocaproate) + glutamate\n"
            "  Isoleucine + alpha-KG → KMV (alpha-keto-beta-methylvalerate) + glutamate\n"
            "  Valine + alpha-KG → KIV (alpha-ketoisovalerate) + glutamate\n\n"
            "BCAT2 is UPSTREAM of the BCKDH complex. BCAT2 deficiency → BCAAs accumulate, "
            "but BCKAs (the BCKDH substrates) are NOT elevated → BCKDH is NOT overloaded → "
            "NO allo-isoleucine (allo-Ile requires KMV accumulation from excess isoleucine → "
            "equilibration through ketimine → allo-Ile; with BCAT2 block, KMV is NOT elevated).\n\n"
            "KEY BIOCHEMICAL DISTINCTION from classic MSUD:\n"
            "  BCAT2 deficiency: BCAAs HIGH, BCKAs LOW/NORMAL, allo-Ile ABSENT → pre-BCKDH\n"
            "  Classic MSUD (Ia/Ib/II/V): BCAAs HIGH, BCKAs HIGH, allo-Ile PRESENT → BCKDH block\n\n"
            "Plasma amino acid profile: BCAA elevation (leucine often most elevated). "
            "Urine organic acids: BCKAs (alpha-ketoisocaproate, KMV, KIV) NOT elevated — crucial DDx. "
            "PLP (pyridoxal-5'-phosphate) trial: BCAT2 requires PLP as cofactor; pyridoxine B6 "
            "supplementation trial warranted (as with other PLP-dependent enzymopathies). "
            "Treatment: BCAA-restricted diet; no leucine neurotoxicity pathway same as MSUD (BCKAs absent), "
            "but BCAA excess itself contributes to ID and feeding problems. "
            "KD: likely contraindicated (BCAAs from fat/protein catabolism). "
            "Liver transplant: limited data; BCAT2 widely expressed, hepatic transplant unlikely fully curative."
        ),
        "hallmark": (
            "BCAT2 HALLMARKS: "
            "(1) PRE-BCKDH BLOCK — upstream aminotransferase; BCKAs NOT elevated. "
            "(2) NO ALLOISOLEUCINE — allo-Ile absent (requires KMV accumulation; KMV not elevated). "
            "(3) KEY DDx FROM CLASSIC MSUD — BCKAs normal distinguishes BCAT2 from Ia/Ib/II/V. "
            "(4) PLP-DEPENDENT — pyridoxine/PLP trial warranted. "
            "(5) BCAA ELEVATED — leucine/Ile/Val high but by different mechanism. "
            "(6) NO MAPLE SYRUP ODOR (sotolone is from BCKAs not BCAAs). "
            "(7) EXTREMELY RARE — fewer than 10 families worldwide."
        ),
        "diet_treatment": "BCAA-restricted diet (titrate all three); pyridoxine/PLP trial; protein intake monitoring.",
        "nbs_marker": "Elevated leucine on MS/MS; allo-isoleucine ABSENT on amino acid quantification — distinguishes from classic MSUD",
        "key_biomarker": "Elevated BCAAs WITHOUT elevated BCKAs on urine organic acids; allo-Ile ABSENT; PLP-responsive possible",
        "severity_spectrum": "Moderate ID + feeding difficulty; severity proportional to BCAA elevation degree",
        "founder_variant": "No founder; all private variants; extremely rare",
        "thiamine_responsive": False,
        "liver_transplant_curative": False,
        "kd_contraindicated": True,
        "alloisoleucine_positive": False,
        "patients": [],
        "n_patients": 40,
    },
    # ── SLC7A5/LAT1 — Brain BCAA transporter — autism + epilepsy ───────────────────
    {
        "gene": "SLC7A5", "protein": "LAT1 (Large Neutral Amino Acid Transporter 1)",
        "alias": "SLC7A5/LAT1 — LAT1 deficiency: brain BCAA transport defect; autism + epilepsy + movement disorder (OMIM #618865)",
        "aa": "507 aa", "kDa": "42 kDa",
        "gene_class": "Large neutral amino acid transporter 1 (LAT1): solute carrier family 7 member 5; forms heterodimer with 4F2hc (SLC3A2); antiporter at blood-brain barrier importing leucine, isoleucine, valine, and other large neutral amino acids (phenylalanine, tyrosine, methionine, tryptophan) into brain in exchange for glutamine",
        "msud_subgroup": "LAT1 deficiency — BCAA brain transport defect (brain BCAA deprivation despite normal plasma BCAA)",
        "locus": "16q24.2", "omim_gene": 600182,
        "inheritance": "AR. 16q24.2. Both sexes equally. Rare; ~100+ families reported as of 2026.",
        "onset_range_y": (0.0, 2.0),
        "phenotype": (
            "Autism spectrum disorder (ASD) + intellectual disability + epilepsy + movement disorder "
            "DESPITE NORMAL plasma BCAA. Plasma amino acids are NORMAL (transport failure is at the "
            "BBB, not in systemic metabolism). Brain BCAA deprivation causes ASD + epilepsy via "
            "mTOR and neurotransmitter synthesis impairment. "
            "Treatment: leucine supplementation DOES NOT help (systemic; cannot enter brain via LAT1)."
        ),
        "disease": (
            "SLC7A5 encodes LAT1 (large neutral amino acid transporter 1, 507aa, 42kDa). "
            "LAT1 is the primary transporter of large neutral amino acids (LNAAs) at the "
            "BLOOD-BRAIN BARRIER (BBB). LAT1 forms a disulfide-linked heterodimer with 4F2hc "
            "(SLC3A2) on the luminal membrane of brain endothelial cells. "
            "LAT1 is an ANTIPORTER: imports LNAAs (Leu, Ile, Val, Phe, Tyr, Trp, Met) INTO "
            "brain in exchange for glutamine (Gln) exiting brain. "
            "This is the rate-limiting step for brain leucine, isoleucine, and valine delivery. "
            "LAT1 deficiency → brain deprivation of ALL LNAAs (especially BCAAs + aromatic AA) despite "
            "NORMAL PLASMA LEVELS:\n"
            "  - Brain leucine LOW → mTOR impaired → reduced synaptic protein synthesis → ASD, ID\n"
            "  - Brain phenylalanine + tyrosine + tryptophan LOW → dopamine + serotonin + noradrenaline synthesis impaired → behavioural/movement disorder\n"
            "  - Brain GABA/glutamate imbalance → epilepsy\n\n"
            "Clinical features: ASD/ID in most; epilepsy (often drug-resistant); movement disorder "
            "(ataxia, dystonia); normal plasma BCAA and LNAAs (BBB is the problem, not plasma). "
            "EEG: focal spikes, generalised epileptiform discharges, sometimes Lennox-Gastaut-like. "
            "MRI: often normal; some cases show cortical atrophy, basal ganglia T2 changes. "
            "NBS: NOT detected (plasma amino acids normal). Diagnosis: clinical features + WES/WGS. "
            "LAT1 CSF amino acids: low leucine, low phenylalanine, low tyrosine in CSF compared to plasma. "
            "Treatment challenges: leucine supplementation INEFFECTIVE (LAT1 needed to carry it into brain). "
            "Investigational: large neutral amino acid mixtures; modified aromatic AA precursors; gene therapy. "
            "PKU analogy: in PKU, excess phenylalanine competitively blocks LAT1 → all LNAAs drop in brain "
            "(same transporter). LAT1-deficient patients have permanently compromised BBB LNAA transport "
            "regardless of plasma levels."
        ),
        "hallmark": (
            "SLC7A5/LAT1 HALLMARKS: "
            "(1) BBB TRANSPORT DEFECT — LAT1 is the blood-brain barrier BCAA importer. "
            "(2) NORMAL PLASMA AMINO ACIDS — BBB block, not systemic catabolism. "
            "(3) ASD + EPILEPSY + MOVEMENT DISORDER — brain LNAA deprivation phenotype. "
            "(4) mTOR LINK — brain leucine via LAT1 activates mTOR-dependent synaptic growth. "
            "(5) PKU ANALOGY — in PKU, phe FLOODS LAT1, blocks all LNAA entry; in LAT1 deficiency, LAT1 absent. "
            "(6) NO ALLOISOLEUCINE — no BCKDH block; BCAA catabolism normal systemically. "
            "(7) LEUCINE SUPPLEMENTATION NOT EFFECTIVE — cannot cross impaired BBB via LAT1. "
            "(8) DRUG-RESISTANT EPILEPSY — often refractory; novel mechanisms require novel approaches."
        ),
        "diet_treatment": "No established effective dietary treatment; investigational LNAA supplementation; protein-enriched diet studied (inconclusive); primarily symptomatic AED management.",
        "nbs_marker": "NOT detected by NBS (plasma amino acids normal)",
        "key_biomarker": "Normal plasma amino acids; low CSF BCAA/LNAA relative to plasma; clinical ASD+epilepsy+movement triad; WES diagnosis",
        "severity_spectrum": "Moderate-severe ASD + ID + epilepsy; variable movement disorder; drug-resistant epilepsy common",
        "founder_variant": "Multiple alleles; some enrichment in specific populations; no single dominant founder",
        "thiamine_responsive": False,
        "liver_transplant_curative": False,
        "kd_contraindicated": False,
        "alloisoleucine_positive": False,
        "patients": [],
        "n_patients": 40,
    },
]


# ─── Synthetic patient cohort generation ─────────────────────────────────────────
def _make_patients(gene_idx: int) -> list:
    """Generate 40 synthetic MSUD/BCAA-disorder patient records for a given gene."""
    g = MSUD_GENES[gene_idx]
    rng = random.Random(SEED_BASE + gene_idx)
    gene = g["gene"]
    is_classic_msud = g["alloisoleucine_positive"]
    kd_ci = g["kd_contraindicated"]
    liver_tx_curative = g["liver_transplant_curative"]

    # Age-at-dx distribution per gene
    if gene in ("BCKDHA", "BCKDHB", "DBT"):
        age_dx = [(rng.uniform(0.0, 0.08)) for _ in range(40)]   # neonatal: 0-4 weeks
    elif gene == "DLD":
        age_dx = [(rng.uniform(0.0, 0.5)) for _ in range(40)]    # 0-6 months, varied onset
    elif gene == "BCKDK":
        age_dx = [(rng.uniform(0.5, 6.0)) for _ in range(40)]    # ASD diagnosis 6m-6y
    elif gene == "PPM1K":
        age_dx = [(rng.uniform(0.0, 0.3)) for _ in range(40)]    # neonatal to 3 months
    elif gene == "BCAT2":
        age_dx = [(rng.uniform(0.0, 0.5)) for _ in range(40)]    # infantile
    else:  # SLC7A5
        age_dx = [(rng.uniform(0.5, 3.0)) for _ in range(40)]    # ASD onset 6m-3y

    patients = []
    for i in range(40):
        pt_id = f"{gene}-{i+1:03d}"
        # Leucine at diagnosis (µmol/L)
        if gene in ("BCKDHA", "BCKDHB", "DBT"):
            leu = rng.randint(800, 4200)
        elif gene == "DLD":
            leu = rng.randint(300, 1800)
        elif gene == "PPM1K":
            leu = rng.randint(250, 900)
        elif gene == "BCAT2":
            leu = rng.randint(250, 700)
        elif gene == "BCKDK":
            leu = rng.randint(30, 110)     # LOW — INVERSE
        else:  # SLC7A5
            leu = rng.randint(90, 170)     # NORMAL

        # Alloisoleucine
        allo_ile_positive = is_classic_msud and (leu > 200)
        # Thiamine response (only classic types, partial)
        thiamine_resp = False
        if g["thiamine_responsive"]:
            thiamine_resp = rng.random() < 0.08   # ~8%

        # Liver transplant status
        liver_tx = False
        if liver_tx_curative and age_dx[i] < 0.5:
            liver_tx = rng.random() < 0.35    # 35% of classic infants get LTx

        # Metabolic crisis at presentation
        crisis = (gene in ("BCKDHA", "BCKDHB", "DBT") and leu > 1500) or \
                 (gene == "DLD" and leu > 600) or (gene == "PPM1K" and leu > 500)

        # MRI abnormal (classic MSUD cerebral edema)
        mri_abnormal = (gene in ("BCKDHA", "BCKDHB", "DBT") and leu > 600) or \
                       (gene == "DLD") or (gene == "PPM1K" and leu > 400)

        # EEG abnormalities
        eeg_abnormal = gene in ("BCKDHA", "BCKDHB", "DBT", "DLD", "BCKDK", "PPM1K", "SLC7A5")

        # Hepatopathy (DLD specific)
        hepatopathy = (gene == "DLD") and (rng.random() < 0.55)

        # Lactic acidosis (DLD)
        lactic_acidosis = (gene == "DLD")

        # Autism / neurodevelopmental
        asd = gene in ("BCKDK", "SLC7A5")

        # Drug resistant epilepsy
        drug_resistant_epilepsy = (gene in ("BCKDK", "SLC7A5") and rng.random() < 0.45)

        # AED used
        if eeg_abnormal:
            aed = rng.choice(["LEV", "VPA (*HIGH RISK)", "PB", "OXC", "LCM", "ZNS", "CLB", "CZP"])
            if gene not in ("BCKDHA", "BCKDHB", "DBT", "DLD", "PPM1K"):
                aed = rng.choice(["LEV", "LCM", "ZNS", "OXC", "CLB"])  # avoid VPA for VPA-risk genes
            vpa_used = "VPA" in aed
        else:
            aed = "None"
            vpa_used = False

        patients.append({
            "id": pt_id,
            "gene": gene,
            "age_dx_y": round(age_dx[i], 3),
            "sex": rng.choice(["M", "F"]),
            "leucine_dx_umolL": leu,
            "allo_isoleucine_positive": allo_ile_positive,
            "thiamine_responsive": thiamine_resp,
            "metabolic_crisis_at_dx": crisis,
            "mri_abnormal": mri_abnormal,
            "eeg_abnormal": eeg_abnormal,
            "hepatopathy": hepatopathy,
            "lactic_acidosis": lactic_acidosis,
            "asd_diagnosis": asd,
            "drug_resistant_epilepsy": drug_resistant_epilepsy,
            "liver_transplant": liver_tx,
            "vpa_used": vpa_used,
            "aed": aed,
        })
    return patients


# Populate all gene patient cohorts
ALL_PATIENTS = []
for idx in range(len(MSUD_GENES)):
    pts = _make_patients(idx)
    MSUD_GENES[idx]["patients"] = pts
    ALL_PATIENTS.extend(pts)


# ─── API: get_overview ─────────────────────────────────────────────────────────────
def get_overview():
    """Return high-level MSUD-Atlas summary."""
    total = len(ALL_PATIENTS)

    gene_summary = []
    for g in MSUD_GENES:
        pts = g["patients"]
        gene_summary.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "gene_class": g["gene_class"],
            "msud_subgroup": g["msud_subgroup"],
            "n_patients": g["n_patients"],
            "inheritance": g["inheritance"],
            "alloisoleucine_positive": g["alloisoleucine_positive"],
            "liver_transplant_curative": g["liver_transplant_curative"],
            "kd_contraindicated": g["kd_contraindicated"],
            "thiamine_responsive": g["thiamine_responsive"],
            "phenotype": g["phenotype"],
            "diet_treatment": g["diet_treatment"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 2),
        })

    n_allo_positive = sum(1 for p in ALL_PATIENTS if p.get("allo_isoleucine_positive", False))
    n_liver_tx      = sum(1 for p in ALL_PATIENTS if p.get("liver_transplant", False))
    n_crisis        = sum(1 for p in ALL_PATIENTS if p.get("metabolic_crisis_at_dx", False))
    n_mri_abnormal  = sum(1 for p in ALL_PATIENTS if p.get("mri_abnormal", False))
    n_eeg_abnormal  = sum(1 for p in ALL_PATIENTS if p.get("eeg_abnormal", False))
    n_hepatopathy   = sum(1 for p in ALL_PATIENTS if p.get("hepatopathy", False))
    n_asd           = sum(1 for p in ALL_PATIENTS if p.get("asd_diagnosis", False))
    n_thiamine_resp = sum(1 for p in ALL_PATIENTS if p.get("thiamine_responsive", False))
    n_vpa_used      = sum(1 for p in ALL_PATIENTS if p.get("vpa_used", False))
    n_drug_resistant = sum(1 for p in ALL_PATIENTS if p.get("drug_resistant_epilepsy", False))

    return {
        "atlas": "MSUD-Atlas",
        "title": "Complete 8-Gene Maple Syrup Urine Disease & BCAA Disorders Atlas",
        "n_genes": len(MSUD_GENES),
        "n_patients": total,
        "seeds": f"{SEED_BASE}–{SEED_BASE + len(MSUD_GENES) - 1}",
        "genes": gene_summary,
        "cohort_stats": {
            "n_allo_isoleucine_positive": n_allo_positive,
            "pct_allo_positive": round(100 * n_allo_positive / total, 1),
            "n_liver_transplant": n_liver_tx,
            "pct_liver_transplant": round(100 * n_liver_tx / total, 1),
            "n_metabolic_crisis_at_dx": n_crisis,
            "pct_crisis": round(100 * n_crisis / total, 1),
            "n_mri_abnormal": n_mri_abnormal,
            "pct_mri_abnormal": round(100 * n_mri_abnormal / total, 1),
            "n_eeg_abnormal": n_eeg_abnormal,
            "pct_eeg_abnormal": round(100 * n_eeg_abnormal / total, 1),
            "n_hepatopathy": n_hepatopathy,
            "pct_hepatopathy": round(100 * n_hepatopathy / total, 1),
            "n_asd": n_asd,
            "pct_asd": round(100 * n_asd / total, 1),
            "n_thiamine_responsive": n_thiamine_resp,
            "n_vpa_used": n_vpa_used,
            "n_drug_resistant_epilepsy": n_drug_resistant,
        },
        "key_teaching": {
            "allo_isoleucine_pathognomonic": "Allo-isoleucine: the ONLY amino acid unique to MSUD — not found normally in humans. Confirms BCKDH complex deficiency without molecular testing. Present in MSUD types Ia/Ib/II/III/V (BCKDHA/B/DBT/DLD/PPM1K) but NOT in BCKDK deficiency or BCAT2 deficiency.",
            "leucine_neurotoxic": "LEUCINE specifically (not isoleucine, not valine) drives cerebral edema and excitotoxicity in MSUD crisis. Monitor leucine selectively; target <200 µmol/L in chronic management; emergency target <300 µmol/L within 24-48h of crisis.",
            "liver_transplant_curative": "Liver transplant CURATIVE for BCKDHA/BCKDHB/DBT/PPM1K: the liver provides >95% of systemic BCKDH activity. Post-transplant: leucine normalises; unrestricted diet; alloisoleucine disappears. NOT curative for DLD (ubiquitous E3) or BCAT2 or LAT1.",
            "bckdk_inverse": "BCKDK INVERSE PHENOTYPE: LOF kinase → hyperactive BCKDH → BCAAs pathologically LOW → autism + epilepsy + ID. Treatment: LEUCINE SUPPLEMENTATION (opposite of classic MSUD). NBS misses BCKDK (low leucine does not trigger standard cutoff).",
            "dld_triple_complex": "DLD E3 DEFICIENCY: triple complex block (PDC + BCKDH + 2-OGDC) → lactic acidosis + BCAAs elevated + 2-oxoglutaricaciduria SIMULTANEOUSLY. Hepatopathy 55% = KEY distinguishing feature from other MSUD types. Liver transplant NOT curative.",
            "kd_contraindicated": "KD CONTRAINDICATED in BCKDHA/B/DBT/DLD/PPM1K: fat catabolism + protein breakdown during ketosis mobilises BCAA from muscle → worsens leucine. Leucine 'starvation diet' + ketosis paradoxically increases leucine toxicity. Prefer LEV for seizures.",
            "nbs_and_bckdk_miss": "NBS detects leucine elevation for classic MSUD (Ia/Ib/II/III/V). BCKDK deficiency and SLC7A5 deficiency are NOT detected by NBS (low/normal leucine). Diagnosis requires WES/WGS for BCKDK and LAT1 deficiency.",
            "thiamine_trial_mandatory": "THIAMINE TRIAL MANDATORY for ALL BCKDH complex deficiencies (BCKDHA/B/DBT/PPM1K): ~5-10% are thiamine-responsive. TPP is the E1 cofactor; thiamine stabilises BCKDH-E1 in some variants. Empiric thiamine 50-100 mg/day × 3 weeks. No harm in trial.",
        },
    }


# ─── API: get_breakdown ───────────────────────────────────────────────────────────
def get_breakdown():
    """Return per-gene patient cohort breakdown."""
    genes_out = []
    for g in MSUD_GENES:
        pts = g["patients"]
        genes_out.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "n_patients": g["n_patients"],
            "alloisoleucine_positive": g["alloisoleucine_positive"],
            "liver_transplant_curative": g["liver_transplant_curative"],
            "kd_contraindicated": g["kd_contraindicated"],
            "thiamine_responsive": g["thiamine_responsive"],
            "founder_variant": g["founder_variant"],
            "mean_leucine_dx": round(sum(p["leucine_dx_umolL"] for p in pts) / len(pts), 0),
            "n_crisis": sum(1 for p in pts if p["metabolic_crisis_at_dx"]),
            "n_liver_tx": sum(1 for p in pts if p["liver_transplant"]),
            "n_eeg_abnormal": sum(1 for p in pts if p["eeg_abnormal"]),
            "n_mri_abnormal": sum(1 for p in pts if p["mri_abnormal"]),
            "n_hepatopathy": sum(1 for p in pts if p["hepatopathy"]),
            "n_asd": sum(1 for p in pts if p["asd_diagnosis"]),
            "n_thiamine_responsive": sum(1 for p in pts if p["thiamine_responsive"]),
            "n_vpa_used": sum(1 for p in pts if p["vpa_used"]),
            "n_drug_resistant_epilepsy": sum(1 for p in pts if p["drug_resistant_epilepsy"]),
            "hallmark": g["hallmark"],
            "disease": g["disease"],
            "patients": pts[:10],  # first 10 for preview
        })
    return {"genes": genes_out, "n_total": len(ALL_PATIENTS)}


# ─── API: get_definitions ─────────────────────────────────────────────────────────────
def get_definitions():
    """Return MSUD clinical term definitions."""
    return {
        "atlas": "MSUD-Atlas — Complete 8-Gene Maple Syrup Urine Disease & Branched-Chain Amino Acid Disorders Atlas",
        "msud_overview": {
            "full_name": "Maple Syrup Urine Disease (MSUD) and Branched-Chain Amino Acid (BCAA) Disorders — an autosomal recessive inborn error of BCAA catabolism; the BCKDH multi-enzyme complex (E1α/BCKDHA, E1β/BCKDHB, E2/DBT, E3/DLD) is deficient in classic MSUD; related disorders include BCKDK (kinase), PPM1K (phosphatase), BCAT2 (aminotransferase), SLC7A5/LAT1 (BBB transport)",
            "genes_in_atlas": 8,
            "pathognomonic_biomarker": "Alloisoleucine (allo-Ile) — PATHOGNOMONIC for BCKDH complex deficiency; not found in normal humans; confirms MSUD types Ia/Ib/II/III/V",
            "curative_treatment": "Liver transplant: curative for BCKDHA/BCKDHB/DBT/PPM1K (liver provides >95% BCKDH activity)",
            "kd_rule": "KD CONTRAINDICATED in all BCKDH complex deficiencies (BCKDHA/B/DBT/DLD/PPM1K)",
        },
        "definitions": [
            {
                "term": "MSUD Pathophysiology: BCKDH Complex and BCAA Catabolism Block",
                "definition": (
                    "Branched-chain amino acids (BCAAs) — leucine, isoleucine, and valine — undergo a "
                    "two-step catabolic pathway in mitochondria:\n\n"
                    "Step 1 (BCAT2): Transamination by BCAA aminotransferase 2 (BCAT2) → BCKAs\n"
                    "  Leucine → alpha-ketoisocaproate (KIC)\n"
                    "  Isoleucine → alpha-keto-beta-methylvalerate (KMV)\n"
                    "  Valine → alpha-ketoisovalerate (KIV)\n\n"
                    "Step 2 (BCKDH complex): Oxidative decarboxylation by BCKDH → acyl-CoAs\n"
                    "  KIC → isovaleryl-CoA (→ leucine catabolism)\n"
                    "  KMV → 2-methylbutyryl-CoA (→ isoleucine catabolism)\n"
                    "  KIV → isobutyryl-CoA (→ valine catabolism)\n\n"
                    "BCKDH complex architecture (like PDC):\n"
                    "  E1 (BCKDHA + BCKDHB, α2β2 tetramer): thiamine pyrophosphate (TPP)-dependent decarboxylase\n"
                    "  E2 (DBT): 24-mer lipoamide acetyltransferase cube core\n"
                    "  E3 (DLD): FAD-dependent dihydrolipoamide dehydrogenase (SHARED with PDC and 2-OGDC)\n"
                    "  Regulatory kinase: BCKDK (phosphorylates E1α-Ser293 → INACTIVATES BCKDH)\n"
                    "  Regulatory phosphatase: PPM1K/PP2Cm (dephosphorylates E1α-pSer293 → ACTIVATES BCKDH)\n\n"
                    "BCKDH block → BCAAs + BCKAs accumulate → allo-isoleucine forms → cerebral oedema.\n\n"
                    "Allo-isoleucine formation: accumulated KMV equilibrates to allo-KMV via ketimine → "
                    "allo-KMV transaminated back to allo-isoleucine by BCAT2; this isomeric pathway is "
                    "BLOCKED by BCAT2 deficiency (explaining why BCAT2 deficiency has NO allo-Ile)."
                ),
            },
            {
                "term": "Alloisoleucine — PATHOGNOMONIC Biomarker for MSUD (Classic Types)",
                "definition": (
                    "Allo-isoleucine (2S,3R-isoleucine; allo-Ile) is the PATHOGNOMONIC amino acid for MSUD. "
                    "It does NOT occur in normal human metabolism.\n\n"
                    "Formation mechanism: KMV (alpha-keto-beta-methylvalerate, the isoleucine-derived BCKA) "
                    "accumulates in BCKDH-deficient MSUD → KMV equilibrates via ketimine intermediate → "
                    "allo-KMV formed → BCAT2 transaminates allo-KMV back to allo-isoleucine (the epimer).\n\n"
                    "Interpretation:\n"
                    "  - Allo-Ile PRESENT → BCKDH complex block (MSUD types Ia/Ib/II/III/V)\n"
                    "  - Allo-Ile ABSENT + high BCAAs → BCAT2 deficiency (pre-BCKDH block; no KMV)\n"
                    "  - Allo-Ile ABSENT + low BCAAs → BCKDK deficiency (BCKDH hyperactive; no KMV accumulation)\n"
                    "  - Allo-Ile ABSENT + normal BCAAs → LAT1 deficiency (systemic BCAA normal)\n\n"
                    "Detection: standard plasma amino acid quantification (ion-exchange chromatography or "
                    "targeted MS); appears between isoleucine and leucine peaks. "
                    "Even small amounts (>5 µmol/L) are significant. "
                    "NBS DOES detect allo-Ile on amino acid panels but standard leucine screening is the "
                    "primary NBS trigger; allo-Ile confirmation follows elevated leucine result."
                ),
            },
            {
                "term": "Leucine Neurotoxicity — Why Leucine (Not Ile or Val) is the Target",
                "definition": (
                    "In MSUD, leucine is specifically neurotoxic; isoleucine and valine are NOT primarily "
                    "responsible for brain injury.\n\n"
                    "Mechanisms of leucine neurotoxicity:\n"
                    "  (1) BBB transport competition: Leucine and other LNAAs share the LAT1 (SLC7A5) "
                    "transporter at the BBB. Elevated leucine COMPETITIVELY INHIBITS entry of other LNAAs "
                    "(phenylalanine, tyrosine, tryptophan, isoleucine, valine) into brain → brain LNAA "
                    "deprivation (similar to mechanism in phenylketonuria). Loss of phenylalanine + tyrosine "
                    "→ dopamine, serotonin synthesis drops → seizures, abnormal tone.\n"
                    "  (2) Cerebral oedema: KIC (alpha-ketoisocaproate from leucine) directly inhibits "
                    "mitochondrial respiratory chain (Complex I and alpha-KGDH) in neurons → energy failure "
                    "→ cytotoxic oedema. KIC is MORE neurotoxic than the BCKAs from Ile and Val.\n"
                    "  (3) Excitotoxicity: Leucine displaces glutamate uptake transporters; KIC acts as "
                    "a partial GABA-A agonist and NMDA disruptor.\n"
                    "  (4) Astrocyte swelling: Leucine enters astrocytes, competes with glutamine export.\n\n"
                    "PRACTICAL CONSEQUENCE: In crisis management, the leucine level is the key monitoring "
                    "target. Isoleucine and valine levels serve as metabolic monitors but cerebral oedema "
                    "risk correlates with LEUCINE > 1000 µmol/L specifically. "
                    "MRI in MSUD encephalopathy: symmetric DWI restriction in WM, brainstem, BG, cerebellum; "
                    "normalises with metabolic control."
                ),
            },
            {
                "term": "Liver Transplant in MSUD — Curative for Classic Types; Not for DLD",
                "definition": (
                    "Liver transplant is CURATIVE for MSUD types Ia (BCKDHA), Ib (BCKDHB), II (DBT), and V (PPM1K). "
                    "This is because the liver provides >95% of systemic BCKDH activity. "
                    "After liver transplant:\n"
                    "  - Plasma leucine normalises rapidly (hours to days)\n"
                    "  - Allo-isoleucine disappears from plasma\n"
                    "  - Unrestricted protein diet becomes possible\n"
                    "  - MSUD crises eliminated\n\n"
                    "Mechanism: the BCKDH complex in hepatocytes accounts for the overwhelming majority of "
                    "BCAA catabolism. Even with complete absence of BCKDH in other tissues (muscle, brain), "
                    "a normal liver provides sufficient BCAA catabolism to maintain normal plasma levels. "
                    "Transplanted liver with functioning BCKDH 'clears' BCAAs system-wide.\n\n"
                    "Optimal timing: early (before severe neurological damage); pre-transplant IQ is the "
                    "strongest predictor of post-transplant cognitive outcome.\n\n"
                    "NOT curative for:\n"
                    "  - DLD (E3): E3 is UBIQUITOUSLY expressed; hepatic correction insufficient for "
                    "PDC and 2-OGDC in brain and muscle; lactic acidosis and 2-OGA persist post-transplant.\n"
                    "  - BCAT2: BCAT2 also widely expressed.\n"
                    "  - BCKDK/LAT1: kinase + transporter defects; not hepatic-specific."
                ),
            },
            {
                "term": "BCKDK Deficiency — Inverse MSUD Phenotype: LOW BCAA + ASD + Epilepsy",
                "definition": (
                    "BCKDK deficiency is an INVERSE MSUD phenotype — opposite of classic MSUD in every biochemical way.\n\n"
                    "Mechanism:\n"
                    "  BCKDK normally PHOSPHORYLATES E1α-Ser293 → INACTIVATES BCKDH (BCAAs preserved)\n"
                    "  BCKDK LOF → E1α NOT phosphorylated → BCKDH CONSTITUTIVELY ACTIVE → BCAAs "
                    "OVER-catabolised → plasma BCAAs pathologically LOW\n\n"
                    "Clinical consequence: brain leucine deprivation:\n"
                    "  - Leucine activates mTORC1 → synaptic protein synthesis; low leucine → mTOR "
                    "impaired → ASD, intellectual disability\n"
                    "  - Low brain BCAAs → reduced glutamate synthesis (BCAAs are nitrogen donors for "
                    "glutamate via BCAT1/2) → GABA/glutamate imbalance → epilepsy\n\n"
                    "Plasma amino acids: leucine LOW (often 30-80 µmol/L; normal 100-200 µmol/L). "
                    "May be within low-normal reference range, easily missed.\n"
                    "NBS: NOT detected (leucine low, not elevated).\n"
                    "No allo-isoleucine, no maple syrup odor, no metabolic crisis.\n\n"
                    "Treatment: LEUCINE SUPPLEMENTATION (0.5-1 g/kg/day) to restore plasma leucine "
                    "to normal range → mTOR activation → may improve ASD, seizures, cognition.\n"
                    "Monitor EEG improvement on leucine supplementation.\n\n"
                    "DDx from MSUD: 'LOW' vs 'HIGH' plasma leucine is the immediate distinguishing feature."
                ),
            },
            {
                "term": "DLD / E3 Deficiency — Triple Complex Disorder with Hepatopathy",
                "definition": (
                    "DLD (dihydrolipoamide dehydrogenase, E3) is the SHARED E3 subunit of three "
                    "mitochondrial alpha-ketoacid dehydrogenase complexes:\n"
                    "  PDC (pyruvate dehydrogenase complex) — pyruvate → acetyl-CoA\n"
                    "  BCKDH (branched-chain alpha-ketoacid dehydrogenase complex) — BCKAs → acyl-CoAs\n"
                    "  2-OGDC (2-oxoglutarate dehydrogenase complex) — alpha-KG → succinyl-CoA\n"
                    "  GCS (glycine cleavage system) — L-protein component\n\n"
                    "E3 deficiency simultaneously disrupts ALL FOUR pathways:\n"
                    "  PDC block → LACTIC ACIDOSIS + elevated lactate:pyruvate ratio\n"
                    "  BCKDH block → BCAA elevation + allo-isoleucine\n"
                    "  2-OGDC block → 2-OXOGLUTARICACIDURIA (urine organic acids pathognomonic)\n"
                    "  GCS impairment → mild glycine elevation (distinguish from NKH by other markers)\n\n"
                    "KEY DISTINGUISHING FEATURE: HEPATOPATHY in 55% of E3 deficiency patients "
                    "(especially Ashkenazi pGly136Asp). Other MSUD types (Ia/Ib/II/V) have NO hepatopathy.\n\n"
                    "Ashkenazi Jewish founder p.Gly136Asp (c.407G>A):\n"
                    "  - Located at the interface between E3 dimer → impairs dimerization\n"
                    "  - Most common E3 deficiency variant worldwide\n"
                    "  - Incidence ~1/35,000-94,000 Ashkenazi Jewish newborns\n\n"
                    "Treatment: avoid fasting; high glucose; thiamine + riboflavin trials; "
                    "low-BCAA diet (less severe restriction than classic MSUD). "
                    "Liver transplant NOT curative (E3 expressed everywhere)."
                ),
            },
            {
                "term": "MSUD Emergency Management — Crisis Protocol",
                "definition": (
                    "MSUD metabolic crisis (leucine >1000 µmol/L) = NEUROLOGICAL EMERGENCY.\n\n"
                    "Acute management priorities:\n"
                    "(1) STOP BCAA INTAKE: discontinue all protein for 24-48h; BCAA-free formula only\n"
                    "(2) ANABOLISM: high glucose infusion rate (GIR 8-12 mg/kg/min) = suppress catabolism "
                    "→ reduce BCAA release from muscle protein breakdown; prevent 'leucine starvation' crisis "
                    "where catabolism paradoxically raises leucine further\n"
                    "(3) ENERGY: insulin 0.05-0.1 IU/kg/h if hyperglycaemia; otherwise continue high GIR\n"
                    "(4) BCAA-FREE AMINO ACID FORMULA: essential to prevent negative nitrogen balance "
                    "(which worsens catabolism → more leucine); isoleucine + valine supplementation helps "
                    "counter leucine competition at BBB and BCKDH reconstitution\n"
                    "(5) DIALYSIS: if leucine >1500 µmol/L or encephalopathy; haemodialysis > peritoneal; "
                    "leucine is small molecule, highly dialysable\n"
                    "(6) MONITOR: leucine every 4-6h during crisis; target <300 µmol/L within 24h, "
                    "<200 µmol/L by 48h; also monitor glucose, lactate, BGA\n"
                    "(7) MRI: DWI sequence critical — confirms cerebral oedema extent; "
                    "guide prognostication\n"
                    "(8) EEG: burst-suppression in severe neonatal crisis; continuous monitoring warranted\n\n"
                    "CRITICAL ERRORS to avoid:\n"
                    "  - Protein restriction WITHOUT caloric support → INCREASES catabolism → worsens crisis\n"
                    "  - KD at any point: absolutely contraindicated\n"
                    "  - VPA: HIGH RISK; use LEV, LCM for seizures\n"
                    "  - Failure to re-introduce Ile + Val during crisis: leucine alone drops Ile+Val further"
                ),
            },
            {
                "term": "LAT1 (SLC7A5) Deficiency — Blood-Brain Barrier BCAA Transport Defect",
                "definition": (
                    "LAT1 (large neutral amino acid transporter 1, SLC7A5) is the PRIMARY importer of "
                    "leucine, isoleucine, valine, phenylalanine, tyrosine, and tryptophan at the "
                    "blood-brain barrier (BBB). LAT1 forms a disulfide heterodimer with 4F2hc (SLC3A2).\n\n"
                    "LAT1 deficiency → brain BCAA AND aromatic amino acid deprivation DESPITE NORMAL PLASMA:\n"
                    "  - Normal plasma amino acids (systemic catabolism intact)\n"
                    "  - CSF amino acids low relative to plasma (BBB importer blocked)\n"
                    "  - Brain mTOR reduced (leucine-dependent)\n"
                    "  - Brain catecholamines reduced (phenylalanine + tyrosine entry blocked)\n"
                    "  - Brain serotonin reduced (tryptophan entry blocked)\n\n"
                    "Clinical: ASD + ID + epilepsy + movement disorder.\n"
                    "NBS: NOT detected (plasma normal).\n\n"
                    "PKU connection: In classic PKU, excessive phenylalanine FLOODS LAT1 at the BBB, "
                    "competitively blocking entry of all other LNAAs → brain LNAA deprivation despite "
                    "elevated plasma LNAA. LAT1 deficiency = permanent structural failure of the same step.\n\n"
                    "Treatment: no effective established therapy; investigational LNAA supplementation; "
                    "leucine supplementation ineffective (cannot enter brain without LAT1). "
                    "Standard AEDs for seizure control; note drug-resistant epilepsy common.\n\n"
                    "Distinguishing from BCKDK: BCKDK → low plasma BCAAs; LAT1 → normal plasma BCAAs; "
                    "both have ASD + epilepsy but BCKDK responds to leucine supplementation while LAT1 does not."
                ),
            },
            {
                "term": "Thiamine Trial in MSUD — Who Responds and Why",
                "definition": (
                    "Thiamine pyrophosphate (TPP) is the cofactor for the E1 (decarboxylase) component "
                    "of the BCKDH complex. A subset of MSUD patients (~5-10% of BCKDHA, ~10-15% of DBT) "
                    "are thiamine-responsive:\n\n"
                    "Mechanism of responsiveness: certain missense variants in E1α (BCKDHA) or E2 (DBT) "
                    "reduce TPP binding affinity. High-dose thiamine supplementation increases available "
                    "TPP → shifts equilibrium → partially restores TPP occupancy → residual BCKDH activity "
                    "sufficient to prevent crisis levels of leucine accumulation.\n\n"
                    "Who to trial: ALL patients with BCKDH complex deficiency (BCKDHA, BCKDHB, DBT, PPM1K) "
                    "should receive empiric thiamine trial — harmless, potentially life-changing.\n\n"
                    "Protocol: thiamine 10-20 mg/kg/day (max 200 mg/day) × 3 weeks. "
                    "Monitor plasma leucine and allo-isoleucine before and after. "
                    "Response: >50% reduction in leucine + improved BCKDH enzyme activity in fibroblasts.\n\n"
                    "Non-responders: still receive standard BCAA-free formula + dietary management.\n\n"
                    "DLD: riboflavin (FAD precursor) trial is also warranted in addition to thiamine "
                    "because E3 is FAD-dependent. Riboflavin 100-200 mg/day trial × 3 weeks."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== MSUD-Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
    # Quick sanity check
    for g in ov["genes"]:
        print(f"  {g['gene']}: {g['n_patients']} pts, allo-Ile={g['alloisoleucine_positive']}, LTx={g['liver_transplant_curative']}")
    print("OK")
