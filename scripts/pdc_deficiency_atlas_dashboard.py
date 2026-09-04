#!/usr/bin/env python3
"""PDC-Deficiency-Atlas — Complete 7-Gene Pyruvate Dehydrogenase Complex Deficiency Atlas
PDHA1 · PDHB · DLAT · DLD · PDHX · PDP1 · PDP2
280-patient aggregate cohort (7 × 40, seeds 800–806)

Pyruvate Dehydrogenase Complex (PDC) facts:
  - PDC converts pyruvate → acetyl-CoA + CO₂ + NADH (irreversible gateway from glycolysis to TCA cycle)
  - PDC deficiency = lactic acidosis with NORMAL or LOW lactate/pyruvate (L/P) ratio (<25)
    → distinguishes from OXPHOS lactic acidosis (L/P >25, citric-acid type)
  - PDC structure: E1 (alpha+beta subunits, PDHA1+PDHB) + E2 (DLAT) + E3 (DLD) + E3BP (PDHX)
    + regulatory kinases (PDK1-4) + phosphatases (PDP1+PDP2)
  - Inheritance: PDHA1 = X-linked (most common); all others AR
  - CRITICAL TREATMENT: Ketogenic Diet (KD) = PRIMARY treatment (unlike OXPHOS where KD is often CI)
  - Thiamine (B1) = obligate cofactor for E1-alpha; trial mandatory in all new PDC deficiency cases
  - DCA (Dichloroacetate) = PDH kinase inhibitor → promotes active (dephosphorylated) PDC state → used off-label

ATLAS SCOPE (7 nuclear-encoded PDC genes):
  E1 alpha subunit (rate-limiting, X-linked):
    PDHA1 (50-70% of all PDC deficiency)
  E1 beta subunit (AR):
    PDHB
  E2 dihydrolipoamide acetyltransferase (AR):
    DLAT
  E3 dihydrolipoamide dehydrogenase — shared with 2-OGDC and GCS (AR):
    DLD (triple-complex deficiency: PDC + 2-OGDC + GCS)
  E3-binding protein / E3BP (PDC-specific, AR):
    PDHX
  PDH phosphatase regulatory subunit (AR):
    PDP1
  PDH phosphatase catalytic subunit (AR, very rare):
    PDP2

CRITICAL CLINICAL RULES:
  1. KD (Ketogenic Diet) = PRIMARY TREATMENT in ALL 7 genes — shifts energy metabolism from pyruvate to fat;
     THIS IS THE OPPOSITE OF OXPHOS DISORDERS where KD is often CI (CV, CoQ10) or CAUTION (CI, MDDS)
  2. Thiamine (Vitamin B1) = MANDATORY TRIAL in ALL PDC deficiency — obligate cofactor E1-alpha;
     thiamine-responsive PDHA1 variants (thiamine-binding pocket): 10-25% of PDHA1 cases respond
  3. DCA (Dichloroacetate) = inhibits PDH kinase → keeps PDC dephosphorylated (active form) → lowers lactate;
     used off-label; neurotoxicity (peripheral neuropathy) limits long-term use; DLD patients: DCA not effective
  4. Metformin ABSOLUTE CI in ALL 7 — Complex I inhibitor → pyruvate cannot be oxidised → catastrophic
     lactate accumulation on top of existing PDC deficiency
  5. VPA HIGH RISK in ALL 7 — inhibits fatty acid beta-oxidation (critical pathway in KD-treated patients);
     also carnitine depletion + worsens metabolic acidosis; prefer LEV/LCM/ZNS
  6. Fasting AVOID in ALL 7 — any illness with reduced oral intake → accelerated catabolism → lactate crisis;
     glucose-containing IV fluids mandatory during any acute illness
  7. PDHA1 X-linked — males severely affected; females VARIABLE (X-inactivation skewing determines severity);
     maternal carrier females may range from asymptomatic to severely affected
  8. DLD triple-complex deficiency — lactic acidosis + elevated 2-oxoglutaric acid (urine) + elevated glycine:
     resembles BOTH MSUD (branched-chain amino acids) AND NKH (glycine) AND PDC deficiency simultaneously
  9. L/P ratio <25 (pyruvate-type) = KEY BIOCHEMICAL FINGERPRINT of PDC deficiency;
     distinguishes from respiratory chain / OXPHOS lactic acidosis (L/P >25, citric-acid type)
  10. BTBGD (SLC19A3) MANDATORY EXCLUSION before PDC workup — treatable mimic with Leigh-like MRI;
      give biotin+thiamine empirically in any Leigh-like presentation before invasive studies
  11. Propofol AVOID — propofol infusion syndrome (PRIS) involves mitochondrial fatty acid oxidation disruption;
      also interferes directly with respiratory chain; conflicts with KD metabolic state
  12. WES detects ALL 7 nuclear PDC genes; enzyme activity in fibroblasts/muscle confirms diagnosis

COHORT: 7 × 40 = 280 patient slots (seeds 800–806; gene-specific seeds)
"""

import random

SEED_BASE = 800

# ── All 7 nuclear-encoded Pyruvate Dehydrogenase Complex genes ───────────────
PDC_GENES = [
    # ── E1 alpha subunit — X-linked, most common ─────────────────────────────
    {
        "gene": "PDHA1", "alias": "E1α / Pyruvate dehydrogenase E1-alpha subunit",
        "aa": "390 aa", "kDa": "43.2 kDa",
        "gene_class": "e1_alpha_xlinked",
        "locus": "Xp22.12", "omim_gene": 300502,
        "phenotype": "PDC Deficiency — MOST COMMON (50-70%) · Leigh syndrome MRI · X-linked · KD = treatment",
        "disease": "PDC Deficiency type E1-alpha (OMIM #312170) — PDHA1; pyruvate dehydrogenase E1-alpha subunit; contains thiamine pyrophosphate (TPP) binding pocket; most common PDC gene (50-70% of PDC deficiency); X-linked dominant; males severely affected; females highly variable (X-inactivation skewing); Leigh syndrome pattern on MRI (basal ganglia + brainstem) most common presentation; lactic acidosis with normal/low L/P ratio (<25) distinguishes from OXPHOS; thiamine trial MANDATORY at diagnosis; KD = primary treatment; Lissauer 1971 first PDC description; Brown 1987 first molecular characterization",
        "inheritance": "X-linked dominant (PDHA1; Xp22.12); males severely affected (hemizygous); females variable (heterozygous, X-inactivation-dependent severity)",
        "hallmark": "PDHA1 = MOST COMMON PDC GENE (50-70% of PDC deficiency); X-linked — requires sex-specific approach; males: neonatal/infantile severe lactic acidosis + encephalopathy + Leigh MRI (BG + brainstem); females: range from asymptomatic (skewed X-inactivation) to as severe as males; Leigh syndrome MRI is hallmark — bilateral basal ganglia + brainstem T2 signal abnormalities; THIAMINE TRIAL MANDATORY (10-25% PDHA1 variants are thiamine-responsive — binding pocket variants p.Arg349 region most responsive); NORMAL OR LOW L/P RATIO (<25) is the biochemical fingerprint of PDC deficiency — critical DDx from respiratory chain lactic acidosis (L/P >25); KD must be started early (before irreversible neurological damage); elevated pyruvate in blood and CSF",
        "key_ddx": "vs PDHB/DLAT/PDHX (AR PDC): PDHA1 is X-linked; vs Leigh syndrome from OXPHOS (SURF1/ND genes): OXPHOS L/P >25 vs PDC L/P <25; vs BTBGD (SLC19A3): treatable mimic — biotin+thiamine; vs pyruvate carboxylase deficiency (PC): PC → neonatal hyperammonemia + citrulline; vs DLD: also elevated 2-oxoglutarate + glycine in DLD; vs NKH (glycine encephalopathy): no lactic acidosis in NKH; vs MSUD: no organic aciduria in PDHA1",
        "founder_variant": "p.Arg349His (thiamine-responsive, European), p.Ala199Val (common null), p.Arg234Cys (recurrent de novo); many private variants given X-linked mechanism",
        "onset_pattern": "Males: neonatal/infantile severe (days to 6 months); females: range from neonatal to adult (ataxia); episodic: exacerbated by carbohydrate load, illness, fever",
        "mri_pattern": "Leigh syndrome: bilateral BG (putamen/caudate) + brainstem (periaqueductal grey) T2 signal; cortical atrophy; dysgenesis corpus callosum in severe neonatal; MRS elevated lactate in BG+WM",
        "hepatopathy_rate": 0.10, "myopathy_rate": 0.40, "encephalopathy_rate": 0.92,
        "cpeo_rate": 0.05, "epilepsy_rate": 0.80, "ataxia_rate": 0.65,
        "lactic_ac_rate": 0.96, "hcm_rate": 0.05, "snhl_rate": 0.08,
        "renal_rate": 0.05, "gi_rate": 0.05, "cognitive_rate": 0.88, "respiratory_rate": 0.50,
        "seed": 800,
        "vpa_ci": "HIGH RISK — VPA inhibits fatty acid beta-oxidation (critical for KD-treated PDC patients); carnitine depletion; worsens metabolic acidosis; AVOID VPA; prefer LEV, LCM, ZNS",
        "thiamine_responsive": True,
        "kd_treatment": True,
        "dca_used": True,
        "dld_triple_complex": False,
        "xlinked": True,
    },
    # ── E1 beta subunit (AR) ──────────────────────────────────────────────────
    {
        "gene": "PDHB", "alias": "E1β / Pyruvate dehydrogenase E1-beta subunit",
        "aa": "359 aa", "kDa": "39.2 kDa",
        "gene_class": "e1_beta_ar",
        "locus": "3p14.3", "omim_gene": 179060,
        "phenotype": "PDC Deficiency — E1-beta subunit · AR · Leigh syndrome · overlaps PDHA1 clinically",
        "disease": "PDC Deficiency type E1-beta (OMIM #614111) — PDHB; E1-beta subunit of E1 decarboxylase component; E1 is a tetramer (α2β2) with two active sites at α-β interfaces; PDHB loss → E1 cannot assemble → complete PDC deficiency; AR inheritance (autosomal recessive); similar phenotype to PDHA1 but autosomal, equal sex incidence; lactic acidosis + Leigh-like encephalopathy; elevated CSF lactate + pyruvate; rare compared to PDHA1; thiamine trial still warranted (E1-beta forms the interface stabilizing TPP in E1-alpha); KD primary treatment; Lissauer/Brown PDC framework applies",
        "inheritance": "Autosomal recessive (PDHB; 3p14.3); equal sex incidence; consanguinity enriches carrier frequency",
        "hallmark": "PDHB = E1-beta; AR PDC deficiency; E1 tetramer (α2β2): PDHB loss → E1 subunit assembly failure → complete PDC deficiency; clinical overlap with PDHA1 (Leigh MRI, lactic acidosis, encephalopathy) but AR (equal sex); fewer variants in literature (PDHA1 >6x more prevalent); L/P ratio <25; CSF lactate elevated; normal or elevated pyruvate; thiamine trial mandatory; KD primary treatment; DCA off-label option; propofol AVOID",
        "key_ddx": "vs PDHA1: X-linked vs PDHB AR (equal sex — key discriminator for genetic counselling); vs DLAT/PDHX: enzyme subunit localisation on activity assay; vs BTBGD: thiamine+biotin responsive; vs Leigh from SURF1/NDUFAF (OXPHOS): L/P >25; vs DLD: no 2-oxoglutaric aciduria in PDHB",
        "founder_variant": "No common founder known; private variants; missense + splice variants described across the coding sequence",
        "onset_pattern": "Infancy (1–12 months): lactic acidosis + hypotonia + encephalopathy; occasional later presentation with episodic ataxia",
        "mri_pattern": "Leigh-like bilateral BG + brainstem; white matter changes; corpus callosum hypoplasia in severe; lactate on MRS",
        "hepatopathy_rate": 0.08, "myopathy_rate": 0.38, "encephalopathy_rate": 0.88,
        "cpeo_rate": 0.04, "epilepsy_rate": 0.72, "ataxia_rate": 0.62,
        "lactic_ac_rate": 0.92, "hcm_rate": 0.05, "snhl_rate": 0.07,
        "renal_rate": 0.04, "gi_rate": 0.04, "cognitive_rate": 0.84, "respiratory_rate": 0.42,
        "seed": 801,
        "vpa_ci": "HIGH RISK — VPA inhibits FA beta-oxidation (KD dependency); carnitine depletion; prefer LEV/LCM",
        "thiamine_responsive": False,
        "kd_treatment": True,
        "dca_used": True,
        "dld_triple_complex": False,
        "xlinked": False,
    },
    # ── E2 dihydrolipoamide acetyltransferase (AR) ────────────────────────────
    {
        "gene": "DLAT", "alias": "E2 / Dihydrolipoamide S-acetyltransferase",
        "aa": "647 aa", "kDa": "69.0 kDa",
        "gene_class": "e2_ar",
        "locus": "11q23.1", "omim_gene": 608770,
        "phenotype": "PDC Deficiency — E2 acetyltransferase · AR · Ataxia prominent · Episodic",
        "disease": "PDC Deficiency type E2 (OMIM #245348) — DLAT; E2 dihydrolipoamide acetyltransferase; inner core of PDC (60-mer icosahedral scaffold); transfers acetyl group from lipoic acid to CoA; contains three domains: N-terminal lipoyl domain (×2) + peripheral subunit-binding domain + C-terminal acetyltransferase; lipoic acid covalently bound to specific lysine in lipoyl domain — DLAT deficiency disrupts lipoic acid attachment AND structural scaffold of entire PDC complex; ataxia is prominent feature (>other PDC genes); episodic metabolic crises; AR",
        "inheritance": "Autosomal recessive (DLAT; 11q23.1)",
        "hallmark": "DLAT = E2 STRUCTURAL SCAFFOLD of PDC (60-mer icosahedron); lipoic acid bound covalently to DLAT lipoyl domain — DLAT deficiency impairs both acetyl-transfer AND lipoylation of E3BP (PDHX); ataxia more prominent relative to other PDC genes; episodic crises more common than continuous severe encephalopathy; periodic vomiting + ataxia + lactic acidosis triggered by carbohydrate load or illness; lipoid granules seen on electron microscopy of liver (abnormal lipoylation); fewer cases in literature vs PDHA1/PDHB; KD primary treatment; DCA reduces lactate but does not restore structural E2; thiamine trial warranted",
        "key_ddx": "vs PDHA1: DLAT AR + more ataxia; vs LIPT1/LIPT2 (lipoic acid biosynthesis): affects E2 of PDC AND 2-OGDC AND KGDH (broader); vs DLD: DLD has triple complex deficiency + glycine elevation; vs episodic ataxia type 1 (EA1, KCNA1): no lactic acidosis; vs EA2 (CACNA1A): no lactic acidosis + acetazolamide-responsive; vs Friedreich ataxia: no lactic acidosis + FRDA GAA repeat",
        "founder_variant": "No well-established founder; rare; multiple private LOF variants",
        "onset_pattern": "Variable: infancy to childhood; episodic exacerbations with carbohydrate loading; can appear as Reye-like syndrome with vomiting + encephalopathy",
        "mri_pattern": "Cerebellar atrophy (ataxia-dominant); ± BG signal changes; less classic Leigh pattern than PDHA1/PDHB; corpus callosum thinning",
        "hepatopathy_rate": 0.18, "myopathy_rate": 0.35, "encephalopathy_rate": 0.78,
        "cpeo_rate": 0.04, "epilepsy_rate": 0.62, "ataxia_rate": 0.78,
        "lactic_ac_rate": 0.88, "hcm_rate": 0.04, "snhl_rate": 0.06,
        "renal_rate": 0.05, "gi_rate": 0.12, "cognitive_rate": 0.75, "respiratory_rate": 0.35,
        "seed": 802,
        "vpa_ci": "HIGH RISK — VPA inhibits FA beta-oxidation (KD-treated); prefer LEV/LCM; also worsens hepatic risk with baseline hepatopathy",
        "thiamine_responsive": False,
        "kd_treatment": True,
        "dca_used": True,
        "dld_triple_complex": False,
        "xlinked": False,
    },
    # ── E3 dihydrolipoamide dehydrogenase — shared triple complex (AR) ────────
    {
        "gene": "DLD", "alias": "E3 / Dihydrolipoamide dehydrogenase (LAD) — shared by PDC, 2-OGDC, GCS",
        "aa": "509 aa", "kDa": "55.2 kDa",
        "gene_class": "e3_triple_complex_ar",
        "locus": "7q31.1", "omim_gene": 238331,
        "phenotype": "Triple-Complex Deficiency (PDC + 2-OGDC + GCS) — Ashkenazi Founder · Hepatopathy · MSUD+NKH+Lactic Acidosis Overlap",
        "disease": "Dihydrolipoamide dehydrogenase (LAD) deficiency / DLD deficiency (OMIM #246900) — DLD; E3 subunit is shared by THREE metabolic complexes: (1) PDC (pyruvate dehydrogenase) + (2) 2-OGDC (2-oxoglutarate / alpha-ketoglutarate dehydrogenase = KGDH) + (3) GCS (glycine cleavage system); DLD deficiency therefore causes COMBINED deficiency of ALL THREE; clinical: combined metabolic pattern = lactic acidosis (PDC) + elevated 2-oxoglutaric acid in urine (2-OGDC) + elevated glycine/sarcosinuria (GCS) + elevated BCAA (MSUD-like from 2-OGDC); Ashkenazi Jewish founder p.Gly136Ser (~1:94,000 births Ashkenazi); hepatopathy prominent (liver disease more common than other PDC genes); DLD is a homodimer (two FAD-binding subunits) regenerating lipoic acid on E2/E3BP of all three complexes",
        "inheritance": "Autosomal recessive (DLD; 7q31.1); Ashkenazi Jewish founder p.Gly136Ser highly prevalent",
        "hallmark": "DLD = TRIPLE-COMPLEX DEFICIENCY — the only PDC gene causing combined PDC+2-OGDC+GCS enzyme deficiency simultaneously; UNIQUE METABOLIC FINGERPRINT: lactic acidosis + 2-oxoglutaric aciduria + elevated glycine + MSUD-like amino acid pattern — NO OTHER single-gene defect mimics this combination; Ashkenazi Jewish founder (p.Gly136Ser — c.683GA; ~1/94,000 births; ~1/100,000 births outside Ashkenazi Gly136Ser compound with other allele); hepatopathy 55% (much higher than other PDC genes — shared GCS pathway affects liver significantly); KD use with CAUTION in DLD (2-OGDC component affects TCA cycle — KD still first-line but monitor TCA intermediates); DCA NOT effective in DLD (DCA's target is PDH kinase which phosphorylates E1-alpha — not relevant when E3 is the defect since E3 dysfunction impairs PDC even when dephosphorylated)",
        "key_ddx": "vs PDHA1/PDHB/DLAT/PDHX (pure PDC): DLD has 2-oxoglutaric aciduria + elevated glycine; vs MSUD (BCKDHA/B/DBT): pure BCAA elevation, no lactic acidosis; vs NKH (GLDC/AMT): pure hyperglycinemia, no lactic acidosis; vs methylmalonic acidemia: no 2-OGA + no glycine elevation; vs BTBGD (SLC19A3): responsive to B1+biotin; vs urea cycle defects: DLD hepatopathy + ammonia possible",
        "founder_variant": "p.Gly136Ser (c.683G>A) — Ashkenazi Jewish founder; p.Ile12Thr; p.Phe354Leu; compound heterozygotes common in non-Ashkenazi",
        "onset_pattern": "Neonatal/infantile: metabolic crisis with combined acidosis; episodic exacerbations; neonatal liver failure possible; hepatopathy can precede neurological features",
        "mri_pattern": "Cortical atrophy; basal ganglia changes; white matter; less classic Leigh than PDHA1; hepatic MRI changes if liver disease",
        "hepatopathy_rate": 0.55, "myopathy_rate": 0.42, "encephalopathy_rate": 0.88,
        "cpeo_rate": 0.05, "epilepsy_rate": 0.70, "ataxia_rate": 0.60,
        "lactic_ac_rate": 0.95, "hcm_rate": 0.08, "snhl_rate": 0.08,
        "renal_rate": 0.10, "gi_rate": 0.18, "cognitive_rate": 0.85, "respiratory_rate": 0.45,
        "seed": 803,
        "vpa_ci": "HIGH RISK — VPA hepatotoxicity (DLD has significant hepatopathy baseline 55%); FA oxidation inhibition; DLD triple-complex: VPA worsens metabolic acidosis via multiple mechanisms; AVOID VPA; prefer LEV/LCM",
        "thiamine_responsive": False,
        "kd_treatment": True,  # KD still first-line but CAUTION note for DLD
        "dca_used": False,  # DCA not effective in DLD (targets PDH kinase, not E3 function)
        "dld_triple_complex": True,
        "xlinked": False,
    },
    # ── E3-binding protein / E3BP — PDC-specific (AR) ─────────────────────────
    {
        "gene": "PDHX", "alias": "E3BP / PDC-E3BP / Protein X — PDC-specific E3-binding protein",
        "aa": "501 aa", "kDa": "54.3 kDa",
        "gene_class": "e3bp_ar",
        "locus": "11p13", "omim_gene": 608769,
        "phenotype": "PDC Deficiency — E3BP subunit · AR · Generally milder than PDHA1 · Romani founder",
        "disease": "PDC Deficiency type E3BP (OMIM #245349) — PDHX; E3-binding protein (formerly called Protein X or dihydrolipoamide dehydrogenase-binding protein, DLDBP); PDC-SPECIFIC (unlike E3/DLD which is shared); binds E3 (DLD) to E2 (DLAT) scaffold via its peripheral subunit-binding domain; contains N-terminal lipoyl domain important for E3 tethering; PDHX deficiency → E3 cannot bind PDC core → PDC activity severely reduced; PDC-specific (2-OGDC and GCS use a different mechanism) → NO triple-complex deficiency (distinguishes from DLD); AR; Romani/Gypsy community founder variant described (Spain/Portugal); generally less severe presentation than PDHA1 — episodic ataxia prominent; KD and thiamine remain first-line",
        "inheritance": "Autosomal recessive (PDHX; 11p13)",
        "hallmark": "PDHX = PDC-SPECIFIC E3BP — unlike DLD (shared E3), PDHX is ONLY in PDC; therefore PDHX deficiency causes PURE PDC deficiency (no 2-oxoglutaric aciduria, no glycine elevation — distinguishes from DLD); peripheral subunit-binding domain targets E3 to the E2 icosahedral core; lipoyl domain of PDHX anchors E3 position; Romani/Gypsy founder variant (p.Arg 206fs or similar splice); tends milder than severe PDHA1 hemizygous males; episodic presentation; CSF lactate elevated; L/P <25 (pure PDC type); DCA therapy applicable; KD primary treatment; thiamine trial warranted",
        "key_ddx": "vs DLD (E3): DLD has triple complex + 2-OGA + glycine — PDHX does NOT; vs PDHA1/PDHB: PDHX AR, generally milder; vs episodic ataxia (EA1/EA2): no lactic acidosis; vs Friedreich ataxia: GAA repeat + absent tendon reflexes; vs BTBGD: biotin+thiamine responsive",
        "founder_variant": "Romani/Romanichal Gypsy founder (exact variant varies by population); p.Asp440Val (European); various splice variants",
        "onset_pattern": "Infancy to childhood; episodic ataxia + lactic acidosis during intercurrent illness; may be milder than PDHA1 hemizygous",
        "mri_pattern": "Variable: cerebellar atrophy (episodic ataxia pattern); ± Leigh-like BG; ± corpus callosum thinning; generally less severe than PDHA1/PDHB Leigh",
        "hepatopathy_rate": 0.08, "myopathy_rate": 0.32, "encephalopathy_rate": 0.75,
        "cpeo_rate": 0.04, "epilepsy_rate": 0.60, "ataxia_rate": 0.72,
        "lactic_ac_rate": 0.88, "hcm_rate": 0.04, "snhl_rate": 0.06,
        "renal_rate": 0.04, "gi_rate": 0.06, "cognitive_rate": 0.72, "respiratory_rate": 0.32,
        "seed": 804,
        "vpa_ci": "HIGH RISK — VPA inhibits FA beta-oxidation (KD-treated patients); AVOID VPA; prefer LEV/LCM",
        "thiamine_responsive": False,
        "kd_treatment": True,
        "dca_used": True,
        "dld_triple_complex": False,
        "xlinked": False,
    },
    # ── PDH phosphatase regulatory subunit (AR) ───────────────────────────────
    {
        "gene": "PDP1", "alias": "PDP1 / PDPR / PDH phosphatase regulatory subunit 1",
        "aa": "537 aa", "kDa": "60.4 kDa",
        "gene_class": "phosphatase_regulatory_ar",
        "locus": "8q22.1", "omim_gene": 605993,
        "phenotype": "PDC Deficiency — PDH phosphatase regulatory subunit · AR · Episodic lactic acidosis · Prominent ataxia",
        "disease": "PDC Deficiency type PDP1 (OMIM #608782) — PDP1 (also PDPR); PDH phosphatase regulatory subunit 1; PDC activity is reversibly regulated by phosphorylation (PDH kinase 1-4, PDKs) and dephosphorylation (PDH phosphatase PDP1/PDP2); phosphorylated E1-alpha = INACTIVE; dephosphorylated E1-alpha = ACTIVE; PDP1 recruits and activates the catalytic phosphatase subunit (PDP2 or PPM2C); PDP1 loss → E1-alpha cannot be dephosphorylated → PDC permanently inactivated even when substrate (pyruvate) is present; DCA (dichloroacetate) = PDH kinase inhibitor — in PDP1 deficiency DCA cannot fully compensate (phosphatase still absent) but reduces kinase-driven phosphorylation; Thiamine important as cofactor; episodic > continuous phenotype; Rahner 2001 first description",
        "inheritance": "Autosomal recessive (PDP1; 8q22.1)",
        "hallmark": "PDP1 = PDH PHOSPHATASE REGULATORY SUBUNIT; phosphorylation/dephosphorylation cycle regulates PDC: ACTIVE (dephosphorylated) ↔ INACTIVE (phosphorylated); PDKs phosphorylate, PDPs dephosphorylate; PDP1 loss → cannot activate PDC even when pyruvate accumulates; EPISODIC PRESENTATION common — exacerbated by carbohydrate load (triggers PDK), illness, fever; DCA partially helpful (inhibits PDKs → reduces hyperphosphorylation); KD remains primary treatment (reduces pyruvate flux reducing PDK activation); thiamine cofactor; L/P <25 (pyruvate type); ataxia and movement disorder prominent; Rahner 2001 original description; PDP1 also binds SIRT3/SIRT5 (mitochondrial sirtuins) — linking acetylation to PDC activity",
        "key_ddx": "vs PDK deficiency (no OMIM disease established): PDK excess → similar permanent inactivation; vs PDHA1/PDHB/PDHX (structural subunits): phosphatase regulation vs substrate handling; vs DLD (triple complex): no 2-OGA/glycine in PDP1; vs BTBGD: biotin+thiamine responsive Leigh mimic; vs episodic ataxia EA2: CACNA1A, acetazolamide-responsive, no lactic acidosis",
        "founder_variant": "Rare; limited variants described; p.Arg350Trp (Rahner 2001 discovery allele); splice variants described",
        "onset_pattern": "Infancy to childhood; episodic more than continuous; exacerbations with carbohydrate loads; can present as acute metabolic crises",
        "mri_pattern": "BG involvement ± brainstem; episodic white matter changes; less constant Leigh pattern than PDHA1; cerebellar atrophy in long-standing ataxia",
        "hepatopathy_rate": 0.10, "myopathy_rate": 0.30, "encephalopathy_rate": 0.72,
        "cpeo_rate": 0.03, "epilepsy_rate": 0.58, "ataxia_rate": 0.75,
        "lactic_ac_rate": 0.85, "hcm_rate": 0.04, "snhl_rate": 0.05,
        "renal_rate": 0.04, "gi_rate": 0.06, "cognitive_rate": 0.68, "respiratory_rate": 0.28,
        "seed": 805,
        "vpa_ci": "HIGH RISK — VPA inhibits FA beta-oxidation in KD-treated patients; also worsens carnitine status; prefer LEV/LCM/ZNS",
        "thiamine_responsive": False,
        "kd_treatment": True,
        "dca_used": True,  # partial benefit — reduces kinase-driven phosphorylation
        "dld_triple_complex": False,
        "xlinked": False,
    },
    # ── PDH phosphatase catalytic subunit (AR, very rare) ────────────────────
    {
        "gene": "PDP2", "alias": "PDP2 / PPM2C / PDH phosphatase catalytic subunit 2",
        "aa": "490 aa", "kDa": "55.8 kDa",
        "gene_class": "phosphatase_catalytic_ar",
        "locus": "16q22.1", "omim_gene": 615105,
        "phenotype": "PDC Deficiency — PDH phosphatase catalytic subunit · AR · Very rare · Episodic lactic acidosis",
        "disease": "PDC Deficiency type PDP2 / PPM2C deficiency — PDP2; catalytic subunit of PDH phosphatase; PDP1 is the regulatory/targeting subunit, PDP2 (PPM2C) is the catalytic metal-dependent (Mg²⁺/Mn²⁺) phosphatase; PP2C-type Ser/Thr phosphatase; dephosphorylates phospho-Ser264-E1-alpha → activates PDC; PDP2 deficiency → PDC permanently inactive; very rare — only a handful of reported cases; clinical overlap with PDP1 (episodic lactic acidosis + encephalopathy); PDP2 also dephosphorylates ACC (acetyl-CoA carboxylase) — secondary effects on lipid metabolism may modify phenotype; de Lonlay 2000 family with similar phosphatase deficiency; thiamine + KD + DCA applied as in PDP1",
        "inheritance": "Autosomal recessive (PDP2/PPM2C; 16q22.1); very rare — <10 reported families",
        "hallmark": "PDP2 = CATALYTIC phosphatase (PPM2C/PP2C-type); Mg²⁺/Mn²⁺-dependent; activates PDC by dephosphorylating E1-alpha phospho-Ser264; PDP1 recruits PDP2 to the PDC complex — both subunits required; VERY RARE: phenotype overlaps PDP1 deficiency (episodic lactic acidosis, ataxia, encephalopathy); PDC enzyme activity in fibroblasts required for diagnosis alongside gene panel; DCA benefits (reduces PDK activity, partially compensatory); KD primary treatment; thiamine mandatory trial; propofol AVOID (PRIS); rare enough that individual case reports define the phenotype rather than a cohort; relationship between PDP1 and PDP2 analogous to regulatory vs catalytic subunit of PP2A",
        "key_ddx": "vs PDP1 (regulatory subunit): same functional pathway — phosphatase activity; enzyme activity overlap; vs structural PDC subunits (PDHA1-PDHX): identify by residual activity + WES; vs BTBGD: biotin+thiamine responsive; vs DLD: triple complex deficiency (not in PDP2); vs Leigh from OXPHOS: L/P >25 in OXPHOS",
        "founder_variant": "No founder known — very rare; individual missense and truncating variants; Rett-like cases with PDP2 variants reported anecdotally",
        "onset_pattern": "Variable — infancy to childhood; episodic exacerbations; some cases detected on expanded NBS (elevated pyruvate)",
        "mri_pattern": "Variable: BG ± brainstem ± cerebellar atrophy; episodic leukoencephalopathy pattern possible; clinical MRI not fully characterised given rarity",
        "hepatopathy_rate": 0.08, "myopathy_rate": 0.28, "encephalopathy_rate": 0.70,
        "cpeo_rate": 0.03, "epilepsy_rate": 0.55, "ataxia_rate": 0.68,
        "lactic_ac_rate": 0.82, "hcm_rate": 0.04, "snhl_rate": 0.05,
        "renal_rate": 0.04, "gi_rate": 0.05, "cognitive_rate": 0.65, "respiratory_rate": 0.25,
        "seed": 806,
        "vpa_ci": "HIGH RISK — VPA inhibits FA beta-oxidation (KD-treated patients); AVOID VPA; prefer LEV/LCM",
        "thiamine_responsive": False,
        "kd_treatment": True,
        "dca_used": True,
        "dld_triple_complex": False,
        "xlinked": False,
    },
]

# ── Cohort generation ─────────────────────────────────────────────────────────
def _generate_cohort(g: dict, n: int = 40) -> list:
    rng = random.Random(g["seed"])
    cohort = []
    for _ in range(n):
        cohort.append({
            "hepatopathy":       rng.random() < g["hepatopathy_rate"],
            "myopathy":          rng.random() < g["myopathy_rate"],
            "encephalopathy":    rng.random() < g["encephalopathy_rate"],
            "cpeo":              rng.random() < g["cpeo_rate"],
            "epilepsy":          rng.random() < g["epilepsy_rate"],
            "ataxia":            rng.random() < g["ataxia_rate"],
            "lactic_acidosis":   rng.random() < g["lactic_ac_rate"],
            "hcm":               rng.random() < g["hcm_rate"],
            "snhl":              rng.random() < g["snhl_rate"],
            "renal":             rng.random() < g["renal_rate"],
            "gi_dysmotility":    rng.random() < g["gi_rate"],
            "cognitive":         rng.random() < g["cognitive_rate"],
            "respiratory_failure": rng.random() < g["respiratory_rate"],
        })
    return cohort


def _pct(cohort: list, key: str) -> float:
    return round(sum(1 for p in cohort if p.get(key)) / len(cohort) * 100, 1)


# ── Public API functions ──────────────────────────────────────────────────────
def get_overview():
    """Return atlas-wide overview and aggregate clinical statistics."""
    all_patients = []
    for g in PDC_GENES:
        all_patients.extend(_generate_cohort(g))

    total = len(all_patients)

    def agg_pct(key):
        return round(sum(1 for p in all_patients if p.get(key)) / total * 100, 1)

    return {
        "atlas_name": "PDC-Deficiency-Atlas",
        "atlas_subtitle": "Complete 7-Gene Pyruvate Dehydrogenase Complex Deficiency Reference",
        "n_genes": len(PDC_GENES),
        "n_patients": total,
        "seeds": "800–806",
        "description": (
            "The Pyruvate Dehydrogenase Complex (PDC) catalyses the irreversible oxidative decarboxylation "
            "of pyruvate to acetyl-CoA — the gateway from glycolysis and lactate into the TCA cycle and "
            "mitochondrial OXPHOS. PDC deficiency causes lactic acidosis with a characteristically NORMAL "
            "or LOW lactate/pyruvate (L/P) ratio (<25), distinguishing it from respiratory chain disorders "
            "(L/P >25). PDHA1 (E1-alpha, X-linked) accounts for 50-70% of all PDC deficiency. "
            "CRITICAL: Ketogenic Diet (KD) IS the primary treatment — the opposite of most OXPHOS disorders "
            "where KD is often contraindicated. Thiamine (B1) is the obligate cofactor and must be trialled "
            "in every new case."
        ),
        "complex_architecture": {
            "e1_component": "E1 decarboxylase tetramer (α2β2): PDHA1 (E1-alpha, TPP-binding) + PDHB (E1-beta, interface)",
            "e2_component": "E2 acetyltransferase (60-mer icosahedral core scaffold): DLAT — anchors lipoic acid; accepts acetyl from E1",
            "e3_component": "E3 dihydrolipoamide dehydrogenase homodimer: DLD — reoxidises lipoate using FAD+NAD+; SHARED with 2-OGDC and GCS",
            "e3bp_component": "E3-binding protein (PDC-specific, 12-mer inner core): PDHX — tethers E3 to E2 scaffold",
            "phosphatases": "PDH phosphatases activate PDC by dephosphorylating E1-alpha pSer264: PDP1 (regulatory) + PDP2/PPM2C (catalytic)",
            "kinases": "PDH kinases 1-4 (PDK1-4) inactivate PDC by phosphorylating E1-alpha; DCA inhibits PDKs (off-label therapy)",
            "coenzymes": "TPP (thiamine pyrophosphate) for E1 reaction; Lipoic acid for E2/E3BP; FAD for E3; CoA + NAD+ as substrates",
        },
        "xlinked_gene": "PDHA1 (Xp22.12) — the only X-linked gene in the PDC complex; all others AR",
        "lp_ratio_rule": "L/P <25 = pyruvate-type lactic acidosis (PDC deficiency); L/P >25 = citric-acid type (OXPHOS/respiratory chain); measure paired lactate+pyruvate simultaneously on ice",
        "aggregate_clinical": {
            "encephalopathy_pct": agg_pct("encephalopathy"),
            "lactic_acidosis_pct": agg_pct("lactic_acidosis"),
            "epilepsy_pct": agg_pct("epilepsy"),
            "cognitive_pct": agg_pct("cognitive"),
            "ataxia_pct": agg_pct("ataxia"),
            "myopathy_pct": agg_pct("myopathy"),
            "respiratory_failure_pct": agg_pct("respiratory_failure"),
            "hepatopathy_pct": agg_pct("hepatopathy"),
            "hcm_pct": agg_pct("hcm"),
            "snhl_pct": agg_pct("snhl"),
            "renal_pct": agg_pct("renal"),
            "cpeo_pct": agg_pct("cpeo"),
            "gi_dysmotility_pct": agg_pct("gi_dysmotility"),
        },
        "drug_contraindications": {
            "kd_primary_treatment": {
                "rule": "KD (Ketogenic Diet) = PRIMARY TREATMENT in ALL 7 PDC genes — OPPOSITE of OXPHOS disorders; shifts energy source from glucose (→ pyruvate → PDC block) to fat (→ acetyl-CoA bypassing PDC); start early, before irreversible neurological damage",
                "note": "In OXPHOS disorders: KD often CI (Complex V/CoQ10) or CAUTION (CI, MDDS/POLG); in PDC: KD IS THE TREATMENT — fundamental distinction for clinical team",
                "dld_caution": "DLD: KD still primary treatment but monitor TCA intermediates (2-OGDC deficiency affects TCA flux); KD still beneficial",
            },
            "thiamine_mandatory_trial": {
                "rule": "Thiamine (Vitamin B1) = MANDATORY TRIAL in ALL new PDC deficiency diagnoses — obligate cofactor for E1-alpha TPP-binding pocket; 10-25% of PDHA1 variants are thiamine-responsive (binding pocket variants); non-responsive variants still receive thiamine as supportive cofactor; dose 10-20mg/kg/day",
                "responsive_gene": "PDHA1 (thiamine-binding pocket variants, especially p.Arg349 region)",
            },
            "dca_off_label": {
                "rule": "DCA (Dichloroacetate) = PDH kinase inhibitor → promotes dephosphorylated (active) PDC state → lowers plasma lactate; off-label; peripheral neuropathy limits long-term use; annual nerve conduction monitoring required",
                "not_effective": "DCA NOT effective in DLD deficiency (DCA targets PDH kinase / E1-alpha phosphorylation; in DLD, E3 function is absent — PDC is inactive regardless of phosphorylation state)",
            },
            "metformin_absolute_ci": {
                "rule": "Metformin ABSOLUTE CI in ALL 7 PDC genes — inhibits Complex I → pyruvate cannot be oxidised via OXPHOS → catastrophic lactate accumulation compounds existing PDC block → fatal lactic acidosis",
                "alternative": "Insulin for diabetes management; GLP-1 agonists with extreme caution; no biguanides",
            },
            "vpa_high_risk_all": {
                "rule": "VPA HIGH RISK in ALL 7 PDC genes — VPA inhibits fatty acid beta-oxidation (the critical energy pathway in KD-treated PDC patients); also causes carnitine depletion; worsens metabolic acidosis; VPA use may precipitate acute metabolic crises in KD-dependent patients",
                "alternative": "Levetiracetam (LEV), Lacosamide (LCM), Zonisamide (ZNS) — preferred AEDs in PDC; avoid carbamazepine in high-dose acute situations (also some mitochondrial toxicity concerns)",
                "extra_caution_dld": "DLD: VPA additionally risks hepatotoxicity given baseline hepatopathy (55%)",
            },
            "propofol_avoid": {
                "rule": "Propofol AVOID in ALL 7 PDC genes — propofol infusion syndrome (PRIS) involves disruption of fatty acid oxidation + direct mitochondrial respiratory chain impairment; conflicts directly with KD metabolic state where FA oxidation is the primary energy source",
                "alternative": "Ketamine, isoflurane, or sevoflurane preferred; liaise with metabolic team before any general anaesthesia",
            },
            "fasting_absolutely_forbidden": {
                "rule": "FASTING ABSOLUTELY FORBIDDEN — any illness/surgery with reduced oral intake → catabolism → accelerated pyruvate production → crisis; glucose-containing IV fluids MANDATORY during any acute illness, surgery, procedure; never use nil-by-mouth without IV glucose supplementation; emergency protocol required",
                "crisis_management": "Acute PDC crisis: IV glucose 10% at 1.5x maintenance + thiamine IV + bicarbonate if severe acidosis + DCA if available; AVOID lactate-containing fluids",
            },
            "chloramphenicol_avoid": "Chloramphenicol AVOID — mt ribosome inhibitor → impairs OXPHOS → synergistic lactic acidosis in PDC (where OXPHOS is already under higher demand)",
            "linezolid_avoid": "Linezolid AVOID — mt ribosome inhibitor → reduced OXPHOS capacity → worsens lactic acidosis in PDC",
            "aminoglycosides_caution": "Aminoglycosides CAUTION — mt ribosome inhibitors (prokaryotic origin); renal proximal tubular toxicity may complicate metabolic management; not absolute CI but use alternatives when possible",
        },
        "special_therapies": {
            "ketogenic_diet": {
                "genes": "ALL 7 PDC genes",
                "therapy": "Ketogenic diet (4:1 or modified Atkins) — primary energy substrate shift from glucose to fat; bypasses PDC block; start as early as possible after diagnosis; maintain continuously; carbohydrate loading triggers crises",
                "mechanism": "Fat → fatty acid beta-oxidation → acetyl-CoA (enters TCA directly, bypassing PDC); reduces pyruvate burden; reduces PDH kinase activation; reduces net lactate production",
                "status": "Standard of care; established evidence; monitored by metabolic dietitian",
            },
            "thiamine_supplementation": {
                "gene_primary": "PDHA1 (10-25% thiamine-responsive — TPP binding pocket variants)",
                "gene_all": "ALL 7 — trial in all newly diagnosed PDC; thiamine as cofactor support even in non-responsive cases",
                "dose": "10-20 mg/kg/day orally (high dose); IV thiamine 50-100mg in acute crisis",
                "mechanism": "Increases TPP availability → maximises residual E1 activity in those with thiamine-responsive variants; may stabilize E1-alpha conformation",
                "status": "Trial mandatory at diagnosis — safe, low-cost, high upside; identify responders",
            },
            "dca_dichloroacetate": {
                "genes": "PDHA1, PDHB, DLAT, PDHX, PDP1, PDP2 (NOT DLD — DCA not effective in E3 deficiency)",
                "therapy": "DCA (Dichloroacetate) 12.5-25 mg/kg/day orally",
                "mechanism": "Inhibits PDH kinase 1-4 (PDKs) → prevents phosphorylation of E1-alpha → more PDC in dephosphorylated (active) form; reduces plasma lactate 30-50%",
                "limitation": "Peripheral neuropathy with long-term use (dose-limiting); thiamine co-administration may be protective; annual nerve conduction studies; not disease-modifying — only reduces lactate load",
                "status": "Off-label; used in clinical practice; long-term safety data limited",
            },
        },
        "key_rules": {
            "kd_is_treatment_not_ci": "KD (Ketogenic Diet) = PRIMARY TREATMENT for PDC deficiency — fundamental teaching point: in OXPHOS/respiratory chain disorders KD is often CONTRAINDICATED or CAUTIONED; in PDC deficiency, KD is the THERAPY. Never withhold KD citing 'mitochondrial disease = KD risk' without confirming which mitochondrial disease.",
            "lp_ratio_fingerprint": "L/P ratio <25 = PYRUVATE TYPE lactic acidosis (PDC/pyruvate metabolism defect); L/P >25 = CITRIC-ACID TYPE (respiratory chain / OXPHOS). Sample blood and CSF lactate+pyruvate SIMULTANEOUSLY, on ice, directly to lab — delayed samples lose pyruvate validity.",
            "pdha1_xlinked_sex_specific": "PDHA1 X-linked: DO NOT dismiss a severely affected female — X-inactivation skewing towards PDHA1-deficient allele produces female phenotype as severe as hemizygous male; likewise, carrier females may be mildly affected or symptomatic during illness.",
            "thiamine_trial_mandatory": "Thiamine trial is MANDATORY in every newly diagnosed PDC deficiency case — 10-25% of PDHA1 variants are thiamine-responsive; safe; cheap; potentially life-changing; never delay for confirmatory enzyme studies.",
            "dld_triple_complex_fingerprint": "DLD deficiency = TRIPLE COMPLEX DEFICIENCY — PDC + 2-OGDC + GCS simultaneously; biochemical fingerprint: lactic acidosis + 2-oxoglutaric aciduria + elevated glycine + MSUD-like amino acids; NO other single nuclear gene defect produces this combination; Ashkenazi Jewish founder p.Gly136Ser.",
            "fasting_crisis_iv_glucose": "ACUTE PDC CRISIS: IV glucose 10% at 1.5× maintenance — do NOT use lactate-containing Ringer's/Hartmann's (adds lactate); IV thiamine; bicarbonate for pH <7.2; ICU if respiratory failure; DCA may be added; avoid propofol anaesthesia.",
            "dca_not_dld": "DCA is NOT effective in DLD deficiency — DCA's mechanism is inhibition of PDH kinases (E1-alpha phosphorylation), but in DLD deficiency, the E3 function is absent: even if E1-alpha is dephosphorylated (active), the overall complex is non-functional without E3; don't apply DCA to DLD patients.",
            "btbgd_mandatory_exclusion": "BTBGD (SLC19A3 — Biotin-Thiamine-Responsive Basal Ganglia Disease) MANDATORY EXCLUSION before PDC workup — treatable mimic of Leigh syndrome (BG + brainstem MRI); empirically give biotin + thiamine to any Leigh-like encephalopathy BEFORE invasive metabolic investigations.",
            "carbohydrate_crisis_trigger": "Carbohydrate loading is the major acute crisis trigger — surgery with IV glucose without KD, illness with glucose drinks, transition off KD; educate family/EMS: never give glucose boluses without metabolic team guidance unless in extremis.",
            "enzyme_activity_fibroblasts": "PDC enzyme activity in fibroblasts or muscle biopsy is required to confirm PDC deficiency (complement WES); WES identifies the gene variant; enzyme activity confirms functional impact and distinguishes subunit defects (fractionated activity).",
        },
        "wes_utility": {
            "nuclear_genes_detected": "ALL 7 nuclear PDC genes detected by WES/gene panel",
            "pdha1_special_note": "PDHA1 on X chromosome — confirm hemizygosity in males; heterozygosity in females; X-inactivation studies available if phenotype discordant with carrier status",
            "enzyme_activity_mandatory": "PDC enzyme activity in fibroblasts/muscle MANDATORY alongside WES to confirm functional deficiency and subunit localization",
            "lp_ratio_simultaneously": "L/P ratio measurement: simultaneous blood and CSF lactate + pyruvate — separated pre-analytical handling causes false results; paired measurement critical",
            "dld_metabolic_fingerprint": "DLD diagnosis: urine organic acids (2-oxoglutaric acid elevation) + plasma amino acids (glycine + BCAA elevation) alongside WES — biochemistry often diagnostic before gene result",
        },
        "pdha1_note": "PDHA1 is the MOST COMMON PDC gene (50-70%). X-linked. Males severely affected; females range from asymptomatic to severe (X-inactivation). Leigh syndrome MRI. Thiamine-responsive variants: 10-25%. KD is primary treatment.",
        "dld_note": "DLD causes TRIPLE-COMPLEX DEFICIENCY (PDC + 2-OGDC + GCS). Ashkenazi Jewish founder p.Gly136Ser. Unique combined biochemical fingerprint: lactic acidosis + 2-OGA (urine) + glycine elevation + MSUD-like amino acids. Hepatopathy 55% — highest in PDC atlas. DCA NOT effective.",
        "kd_oxphos_distinction_note": "CRITICAL TEACHING: KD = TREATMENT for PDC deficiency; KD = CONTRAINDICATED/CAUTION in many OXPHOS disorders (CV subunit defects, CoQ10 deficiency, POLG-MDDS). Know the metabolic block before prescribing KD in any mitochondrial disease.",
    }


def get_breakdown():
    """Return per-gene detailed breakdown for all 7 PDC genes."""
    results = []
    for g in PDC_GENES:
        cohort = _generate_cohort(g)
        results.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "gene_class": g["gene_class"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "phenotype": g["phenotype"],
            "disease": g["disease"],
            "inheritance": g["inheritance"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "founder_variant": g["founder_variant"],
            "onset_pattern": g["onset_pattern"],
            "mri_pattern": g["mri_pattern"],
            "seed": g["seed"],
            "cohort_n": len(cohort),
            # Aggregate phenotype rates from cohort
            "hepatopathy_pct": _pct(cohort, "hepatopathy"),
            "myopathy_pct": _pct(cohort, "myopathy"),
            "encephalopathy_pct": _pct(cohort, "encephalopathy"),
            "cpeo_pct": _pct(cohort, "cpeo"),
            "epilepsy_pct": _pct(cohort, "epilepsy"),
            "ataxia_pct": _pct(cohort, "ataxia"),
            "lactic_ac_pct": _pct(cohort, "lactic_acidosis"),
            "hcm_pct": _pct(cohort, "hcm"),
            "snhl_pct": _pct(cohort, "snhl"),
            "renal_pct": _pct(cohort, "renal"),
            "gi_pct": _pct(cohort, "gi_dysmotility"),
            "cognitive_pct": _pct(cohort, "cognitive"),
            "respiratory_pct": _pct(cohort, "respiratory_failure"),
            # PDC-specific flags
            "vpa_ci": g["vpa_ci"],
            "thiamine_responsive": g["thiamine_responsive"],
            "kd_treatment": g["kd_treatment"],
            "dca_used": g["dca_used"],
            "dld_triple_complex": g["dld_triple_complex"],
            "xlinked": g["xlinked"],
        })
    return {"genes": results}


def get_definitions():
    """Return list of key PDC-specific clinical terms and definitions."""
    return [
        {
            "term": "Pyruvate Dehydrogenase Complex (PDC)",
            "definition": "Multi-enzyme complex (molecular weight ~9.5 MDa) in the mitochondrial matrix that irreversibly converts pyruvate → acetyl-CoA + CO₂ + NADH. This is the obligate entry point from glycolysis/lactate into the TCA cycle. Structure: E1 tetramer (α2β2; PDHA1+PDHB) + E2 60-mer icosahedral core (DLAT) + E3 homodimer (DLD) + E3-binding protein/E3BP (PDHX) + regulatory kinases (PDK1-4) + phosphatases (PDP1+PDP2). Cofactors: TPP (E1), lipoic acid (E2/E3BP), FAD (E3), CoA+NAD+."
        },
        {
            "term": "Lactate/Pyruvate (L/P) Ratio",
            "definition": "Critical diagnostic ratio distinguishing two forms of lactic acidosis. PYRUVATE TYPE (L/P <25): PDC deficiency and other pyruvate metabolism defects — pyruvate accumulates equally with lactate → L/P normal/low. CITRIC-ACID TYPE (L/P >25): respiratory chain / OXPHOS deficiency — NADH accumulates driving pyruvate→lactate → L/P elevated. Measurement requires SIMULTANEOUS blood (and ideally CSF) lactate+pyruvate, collected on ice, processed immediately — delayed or improperly handled samples lose pyruvate, falsely elevating the L/P ratio."
        },
        {
            "term": "Ketogenic Diet (KD) in PDC Deficiency — PRIMARY TREATMENT",
            "definition": "KD provides fat as the predominant fuel, generating acetyl-CoA via fatty acid beta-oxidation that enters the TCA cycle DIRECTLY — completely bypassing the PDC block. KD reduces pyruvate load, reduces PDH kinase activation, and reduces net lactate production. KD is the standard primary treatment for all 7 PDC genes. IMPORTANT DISTINCTION: KD is CONTRAINDICATED or CAUTIONED in many OXPHOS disorders (Complex V/CoQ10/POLG-MDDS). Prescribers must confirm the specific mitochondrial disease before applying or withholding KD."
        },
        {
            "term": "Thiamine (Vitamin B1) and Thiamine Pyrophosphate (TPP)",
            "definition": "Thiamine is the obligate cofactor for PDHA1 (E1-alpha). In the active site, TPP (the phosphorylated active form) participates in the decarboxylation of pyruvate. PDHA1 variants in the TPP-binding pocket (especially near p.Arg349) may retain residual activity that is augmented by high thiamine concentrations — 'thiamine-responsive' variants (10-25% of PDHA1 cases). Thiamine trial is mandatory at diagnosis of any PDC deficiency: dose 10-20 mg/kg/day. Response assessment: serial plasma lactate + CSF lactate during supplementation."
        },
        {
            "term": "DCA (Dichloroacetate)",
            "definition": "Pyruvate analogue that inhibits PDH kinases (PDK1-4). PDKs inactivate PDC by phosphorylating E1-alpha at Ser264/Ser271/Ser293; DCA blocks this → more PDC in the dephosphorylated active state. DCA reduces plasma lactate by 30-50% in PDC deficiency. LIMITATIONS: peripheral sensorimotor neuropathy with long-term use (dose-dependent); monitor with annual nerve conduction studies; not disease-modifying. DCA is NOT effective in DLD deficiency (target is E1-alpha phosphorylation — irrelevant when E3 function is absent)."
        },
        {
            "term": "DLD (E3) Triple-Complex Deficiency",
            "definition": "DLD (dihydrolipoamide dehydrogenase) is the E3 subunit shared by THREE mitochondrial multi-enzyme complexes: (1) PDC — pyruvate → acetyl-CoA; (2) 2-OGDC/KGDH — 2-oxoglutarate → succinyl-CoA (TCA cycle step 4); (3) GCS — glycine cleavage system. DLD deficiency therefore impairs ALL THREE simultaneously. Biochemical fingerprint: lactic acidosis (PDC) + 2-oxoglutaric aciduria (2-OGDC) + elevated plasma glycine (GCS) + BCAA elevation (MSUD-like from OGDH). Ashkenazi Jewish founder: p.Gly136Ser (~1:94,000 births). Hepatopathy 55% (highest in PDC atlas)."
        },
        {
            "term": "PDHA1 X-linked Inheritance",
            "definition": "PDHA1 at Xp22.12 is the only X-linked gene in the PDC complex. Hemizygous males: uniformly severely affected; cannot compensate. Heterozygous females: severity determined by X-inactivation pattern (lyonization) — if the normal allele is preferentially inactivated (skewed X-inactivation toward PDHA1-mutant), females may be as severely affected as males. Carrier females: range from mildly symptomatic (episodic ataxia, fatigue) to asymptomatic. IMPORTANT: a severely affected female with lactic acidosis should NOT exclude PDHA1 diagnosis."
        },
        {
            "term": "Fasting and Acute Crisis Prevention",
            "definition": "PDC-deficient patients cannot oxidize pyruvate efficiently. Fasting or illness with reduced oral intake → catabolism → increased pyruvate production → acute crisis. Emergency rule: GLUCOSE-CONTAINING IV FLUIDS MANDATORY during any acute illness, surgical fasting, or procedure requiring nil-by-mouth. Do NOT use lactate-containing fluids (Ringer's/Hartmann's — adds exogenous lactate). Emergency protocol must be documented in the patient's record, accessible to EMS and emergency departments."
        },
        {
            "term": "PDHX (E3-binding Protein) — PDC-Specific",
            "definition": "PDHX encodes the E3-binding protein (E3BP), a structural component unique to PDC (unlike DLD/E3 which is shared). E3BP bridges E3 (DLD) to the E2 (DLAT) icosahedral core via its peripheral subunit-binding domain; its N-terminal lipoyl domain positions E3 for catalysis. PDHX deficiency = pure PDC deficiency without 2-OGDC or GCS involvement (distinguishing feature from DLD). Romani/Gypsy populations may have founder variants. Generally milder than severe PDHA1."
        },
        {
            "term": "PDH Phosphatase Regulation (PDP1/PDP2)",
            "definition": "PDC activity is dynamically regulated by reversible phosphorylation: PDH kinases (PDK1-4) inactivate PDC by phosphorylating E1-alpha → INACTIVE; PDH phosphatases (PDP1 regulatory + PDP2/PPM2C catalytic) activate PDC by dephosphorylation → ACTIVE. PDP1 recruits PDP2 to the PDC complex and is activated by pyruvate/Ca²⁺. PDP1 or PDP2 deficiency → PDC permanently inactivated even when substrate is present. DCA therapy (PDK inhibitor) has partial benefit in PDP1/PDP2 deficiency (reduces kinase-driven phosphorylation but cannot fully compensate for absent phosphatase)."
        },
        {
            "term": "BTBGD (Biotin-Thiamine-Responsive Basal Ganglia Disease) — Mandatory Exclusion",
            "definition": "SLC19A3 deficiency causes BTBGD — a treatable mimic of Leigh syndrome (basal ganglia + brainstem MRI signal abnormalities) that must be excluded BEFORE PDC workup. Empiric biotin (5-10 mg/day) + thiamine (300 mg/day) can produce dramatic clinical improvement. Never delay biotin+thiamine trial waiting for genetic confirmation in any acute Leigh-like presentation. Missing BTBGD = missed cure. BTBGD must be excluded with SLC19A3 sequencing and empiric treatment in any Leigh-like encephalopathy regardless of suspected PDC vs OXPHOS aetiology."
        },
        {
            "term": "Pyruvate Carboxylase (PC) vs PDC",
            "definition": "Both PDC and pyruvate carboxylase (PC gene) deficiency cause lactic acidosis in neonates/infants but have distinct biochemical signatures: PDC deficiency: L/P ratio normal/low (<25); elevated pyruvate; no hyperammonemia; no citrullinemia. PC deficiency: L/P ratio >25 (citric-acid type); hyperammonemia (PC required for N-acetylglutamate synthesis via anaplerosis); elevated citrulline; also elevated 3-hydroxybutyrate. Distinguishing these guides treatment (KD beneficial in PDC; anaplerotic therapy used in PC)."
        },
        {
            "term": "Carbohydrate Loading as Crisis Trigger",
            "definition": "Glucose administration activates PDH kinase (PDK4 especially) → inactivates PDC by phosphorylating E1-alpha → pyruvate cannot be oxidised → acute lactic acidosis crisis. High-carbohydrate meals, IV dextrose boluses, glucose tolerance tests, post-surgical glucose drips, or transition off KD are common iatrogenic triggers. Educate family and emergency teams: never give glucose boluses or high-carbohydrate foods without metabolic team guidance. During acute illness use 10% dextrose at 1.5× maintenance rate, not bolus dextrose."
        },
        {
            "term": "PDC Enzyme Activity Assay",
            "definition": "Functional enzyme activity measurement in cultured skin fibroblasts or muscle biopsy complements WES gene diagnosis. Measures overall PDC activity (CO₂ release from ¹⁴C-pyruvate), and fractionated activity for each subunit. Confirms pathogenicity of WES variants of uncertain significance. Distinguishes PDC deficiency subtype (E1, E2, E3, E3BP, phosphatase component) before WES result is available. Required in all PDC suspected cases alongside molecular diagnosis."
        },
    ]


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(f"Atlas: {ov['atlas_name']}")
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}")
    print(f"Aggregate encephalopathy: {ov['aggregate_clinical']['encephalopathy_pct']}%")
    print(f"Aggregate lactic acidosis: {ov['aggregate_clinical']['lactic_acidosis_pct']}%")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    defs = get_definitions()
    print(f"Definitions: {len(defs)}")
    print("OK")
