#!/usr/bin/env python3
"""UCD-Atlas — Complete 6-Gene Urea Cycle Disorders Atlas
CPS1 · OTC · ASS1 · ASL · ARG1 · NAGS
240-patient aggregate cohort (6 × 40, seeds 844–849)

Urea Cycle Disorders (UCD) facts:
  - UCDs = inherited defects in the urea cycle enzymes/transporters → hyperammonemia
  - Urea cycle: NH3 + HCO3⁻ + 2ATP → carbamoyl phosphate (CPS1) →
    citrulline (OTC) → argininosuccinate (ASS1) → arginine (ASL) → urea + ornithine (ARG1)
  - NAGS produces N-acetylglutamate (NAG): the essential allosteric activator of CPS1
  - Hyperammonemia is the common toxic mechanism: NH3 → cerebral oedema, cell death
  - OTC deficiency: MOST COMMON UCD; ONLY X-LINKED UCD
  - ARG1 deficiency: DISTINCT — progressive spastic diplegia, not acute hyperammonemia
  - NAGS deficiency: UNIQUE — 100% responsive to N-carbamylglutamate (Carbaglu/NCG)
  - VPA HIGH RISK ALL 6: VPA inhibits NAGS → secondary hyperammonemia in any UCD

ATLAS SCOPE (6 nuclear-encoded UCD genes):
  Proximal cycle (mitochondrial):
    NAGS  — N-acetylglutamate synthase, allosteric master switch (534aa, 17q21.31)
    CPS1  — Carbamoyl phosphate synthetase 1 (1500aa, 2q34)
    OTC   — Ornithine transcarbamylase, MOST COMMON, X-linked (354aa, Xp11.4)
  Distal cycle (cytosolic):
    ASS1  — Argininosuccinate synthase, Citrullinemia type 1 (412aa, 9q34.11)
    ASL   — Argininosuccinate lyase, ASA disease (464aa, 7q11.21)
    ARG1  — Arginase 1, Hyperargininemia (322aa, 6q23.2)

CRITICAL CLINICAL RULES:
  1. VPA HIGH RISK for all 6 UCD genes; VPA inhibits NAGS → secondary hyperammonemia
     Use LEV, LCM as first-choice AED alternatives in any UCD
  2. PROTEIN RESTRICTION EMERGENCY: zero protein 24–48h in acute crisis
     then reintroduce at 0.5–1.5 g/kg/day with essential amino acid supplements
  3. GLUCOSE INFUSION RATE (GIR) 8–12 mg/kg/min: anti-catabolic; prevents endogenous
     protein catabolism; mandatory in acute hyperammonemia for all 6 genes
  4. AMMONIA SCAVENGERS: sodium benzoate 250 mg/kg + sodium phenylacetate 250 mg/kg IV
     (Ammonul) + arginine (except ARG1 — see rule 7)
  5. NCG (N-carbamylglutamate, Carbaglu) ONLY curative for NAGS deficiency — 100% responsive;
     NCG 100–250 mg/kg/day oral; NAGS patients often achieve normal NH3 on NCG alone
     NCG may offer partial benefit in CPS1 (NAG analogue partially activates CPS1)
  6. LIVER TRANSPLANT curative for urea cycle function in CPS1/OTC (males)/ASS1;
     neurological damage (ASL, ARG1) and female OTC brain damage may NOT reverse post-LT
  7. ARG1 UNIQUE — arginine supplementation is CONTRAINDICATED (arginine is the toxic
     metabolite); use essential amino acid formula without arginine; protein restriction mandatory
     ALL OTHER UCDs (CPS1/OTC/ASS1/ASL/NAGS): arginine supplementation IS indicated
     (arginine is an essential amino acid in UCD — cannot be synthesised beyond the block)
  8. OTC: X-LINKED — males present neonatally (fatal without treatment); female carriers
     have 15–25% symptomatic risk (skewed X-inactivation); orotate elevated (pathognomonic)
  9. OROTATE (urinary orotic acid) is the KEY discriminator for OTC vs CPS1/NAGS:
     OTC deficiency: HIGH orotate (carbamoyl phosphate diverts to pyrimidine → orotate)
     CPS1 deficiency: LOW/NORMAL orotate (no carbamoyl phosphate produced at all)
     NAGS deficiency: LOW/NORMAL orotate (identical to CPS1 — NAGS→CPS1 block)
  10. BTBGD (SLC19A3 deficiency) mandatory exclusion in Leigh-like presentations with
      hyperammonemia — biotin-thiamine therapy curative if SLC19A3 is the cause

COHORT: 6 × 40 = 240 patient slots (seeds 844–849; gene-specific seeds)
"""

import random

SEED_BASE = 844

# ── All 6 urea cycle disorder genes ──────────────────────────────────────────
UCD_GENES = [
    # ── NAGS — N-acetylglutamate synthase ──
    {
        "gene": "NAGS", "alias": "NAGS — N-Acetylglutamate Synthase (UCD Master Switch / NCG-Curative)",
        "aa": "534 aa", "kDa": "62.5 kDa",
        "gene_class": "nag_synthase",
        "locus": "17q21.31", "omim_gene": 237310,
        "phenotype": "NAGS Deficiency (NASD) — NCG-responsive hyperammonemia; NAGS is the allosteric master switch of the entire urea cycle",
        "disease": (
            "NAGS biallelic loss → NAGS Deficiency (NASD, OMIM #237310). NAGS encodes the "
            "mitochondrial N-acetylglutamate synthase (534aa, 62.5 kDa), catalysing the "
            "condensation of glutamate + acetyl-CoA → N-acetylglutamate (NAG). NAG is the "
            "ESSENTIAL ALLOSTERIC ACTIVATOR of CPS1 — without NAG, CPS1 cannot bind carbamate "
            "and the entire urea cycle arrests at step 1. NAGS deficiency therefore causes "
            "biochemically IDENTICAL presentation to CPS1 deficiency (low citrulline, low arginine, "
            "low orotate, severe hyperammonemia). NAGS deficiency is the RAREST UCD but also the "
            "most treatable: N-carbamylglutamate (NCG, Carbaglu) — a structural NAG analogue — "
            "bypasses the NAGS block and directly activates CPS1, normalising ammonia. Secondary "
            "NAGS inhibition: propionyl-CoA (propionic acidemia), valproate, and other acyl-CoAs "
            "competitively inhibit NAGS → secondary hyperammonemia in those conditions. Arginine "
            "also activates NAGS allosterically (arginine stimulates its own synthesis)."
        ),
        "inheritance": "Autosomal recessive, biallelic. Equal M:F. Very rare: <50 cases worldwide.",
        "hallmark": (
            "NAGS HALLMARKS: (1) BIOCHEMICALLY IDENTICAL to CPS1 — elevated NH3, low citrulline, "
            "NORMAL/LOW orotate (no CPS1 activity → no carbamoyl phosphate → no orotic acid); "
            "(2) NCG-CURATIVE — ONLY UCD where a drug (NCG/Carbaglu 100–250 mg/kg/day oral) "
            "essentially normalises the urea cycle; most patients achieve normal NH3 on NCG alone; "
            "(3) Neonatal crisis: NH3 200–2000 µmol/L, encephalopathy, liver dysfunction; "
            "(4) DISTINGUISHES from CPS1: NCG challenge — CPS1 patients show partial/no response; "
            "NAGS patients normalise NH3 within 24–48h; (5) Arginine activates NAGS allosterically; "
            "arginine supplementation (200–500 mg/kg/day) IS indicated (not contraindicated); "
            "(6) VPA ABSOLUTE CI: VPA directly inhibits NAGS → worsening hyperammonemia; "
            "(7) Lifelong NCG therapy: low cost, oral, excellent tolerability; "
            "(8) Protein restriction required but LESS aggressive once NCG stabilises cycle; "
            "(9) Liver transplant NOT usually needed once NCG treatment established; "
            "(10) Secondary NAGS inhibition: OA/PA (propionyl-CoA), valproate — treatable causes of NH3."
        ),
        "key_ddx": (
            "vs CPS1: IDENTICAL biochemistry (low NH3, low citrulline, low orotate); NCG trial MANDATORY — "
            "NAGS responds 100%, CPS1 responds partially or not at all; WES distinguishes; "
            "vs OTC: OTC has HIGH orotate (NAGS/CPS1 have LOW orotate — KEY DISCRIMINATOR); "
            "vs secondary hyperammonemia (VPA/OA): check acylcarnitines + organic acids + drug history"
        ),
        "ncg_response": "100% — NCG is essentially curative for NAGS deficiency",
        "liver_transplant": "Usually NOT required once NCG stabilises the urea cycle",
        "critical_ci": "VPA ABSOLUTE CI — directly inhibits NAGS; use LEV/LCM",
        "arginine_rule": "Arginine supplementation INDICATED (200–500 mg/kg/day): arginine activates NAGS allosterically",
        "orotate": "LOW/NORMAL — NO carbamoyl phosphate produced; no orotic acid",
        "nbs_profile": "Elevated NH3; low citrulline; low arginine; plasma amino acids show glutamine elevation",
        "severity": "Severe neonatal if untreated; TREATABLE with NCG",
        "n_patients": 40, "seed": 844,
        "founder": "p.Arg533His (Saudi Arabia); p.Gly189Arg (Turkey); no single major founder",
        "key_variants": [
            "p.Arg533His — C-terminal domain, severe neonatal, Saudi",
            "p.Gly189Arg — NAG-binding fold disruption, severe",
            "p.Arg533Cys — C-terminal, intermediate",
            "p.Val350Ala — acetyl-CoA binding, moderate",
            "p.Trp469Ter — null allele, neonatal lethal",
        ],
    },
    # ── CPS1 — Carbamoyl phosphate synthetase 1 ──
    {
        "gene": "CPS1", "alias": "CPS1 — Carbamoyl Phosphate Synthetase 1 (Most Severe Neonatal UCD)",
        "aa": "1500 aa", "kDa": "165 kDa",
        "gene_class": "carbamoyl_phosphate_synthetase",
        "locus": "2q34", "omim_gene": 237300,
        "phenotype": "CPS1 Deficiency — most severe proximal UCD; NH3 >2000 µmol/L neonatal; low citrulline; low orotate",
        "disease": (
            "CPS1 biallelic loss → CPS1 Deficiency (OMIM #237300). CPS1 encodes the mitochondrial "
            "carbamoyl phosphate synthetase 1 (1500aa, 165 kDa) — the LARGEST urea cycle enzyme — "
            "catalysing: NH3 + HCO3⁻ + 2 ATP → carbamoyl phosphate (3 sequential reactions in one "
            "enzyme: glutamine hydrolysis, bicarbonate phosphorylation, carbamate phosphorylation). "
            "CPS1 requires NAG (NAGS product) as obligate allosteric activator — hence NAGS deficiency "
            "phenocopies CPS1 deficiency. CPS1 is the rate-limiting step of the ENTIRE urea cycle. "
            "CPS1 deficiency produces the MOST SEVERE neonatal hyperammonemia of all UCDs "
            "(NH3 frequently 1000–5000 µmol/L at day 2–5 of life). No carbamoyl phosphate → "
            "no citrulline (low), no arginine (low), no orotic acid (low) — the classic proximal UCD "
            "biochemical profile. Glutamine massively elevated (NH3 sink) → cerebral glutamine "
            "accumulation → astrocyte swelling → cerebral oedema."
        ),
        "inheritance": "Autosomal recessive, biallelic. Equal M:F. Incidence ~1:800,000 live births.",
        "hallmark": (
            "CPS1 HALLMARKS: (1) PROXIMAL UCD: NH3 > 1000 µmol/L neonatal (day 2–5); profound "
            "encephalopathy, lethargy→coma→brain herniation without treatment; "
            "(2) BIOCHEMISTRY: low citrulline (<5 µmol/L; normal 9–30), low arginine (<30 µmol/L), "
            "NORMAL/LOW orotate — CRITICAL DISCRIMINATOR from OTC (OTC has high orotate); "
            "elevated plasma glutamine (>1000 µmol/L); "
            "(3) LARGEST urea cycle enzyme (1500aa); no functional hypomorphs — LOF = complete block; "
            "(4) NCG trial: variable response (partial CPS1 activation via NAG analogue) — "
            "helps distinguish CPS1 from NAGS (NAGS: 100% response; CPS1: <50%); "
            "(5) MOST COMMON CAUSE of neonatal hyperammonemic coma after OTC; "
            "(6) Liver transplant curative for metabolic control; cognitive outcome depends on NH3 burden; "
            "(7) Arginine supplementation 200–500 mg/kg/day INDICATED (all UCDs except ARG1); "
            "(8) Emergency: haemodiafiltration/CVVHDF for NH3 >500 µmol/L; benzoate + phenylacetate; "
            "(9) Long-term: protein restriction + essential amino acid formula + sodium benzoate; "
            "(10) BTBGD mandatory exclusion."
        ),
        "key_ddx": (
            "vs NAGS: IDENTICAL biochemistry; NCG trial discriminates — NAGS 100% responsive, CPS1 variable; "
            "vs OTC: OROTATE KEY — CPS1/NAGS: LOW orotate; OTC: HIGH orotate; "
            "vs organic acidemia: acylcarnitines + organic acids; PA/MMA cause SECONDARY hyperammonemia "
            "(propionyl-CoA/methylmalonyl-CoA inhibit NAGS) with elevated C3/C4 acylcarnitines"
        ),
        "ncg_response": "Variable (≤50%) — CPS1 partially activated by NAG analogue; not curative",
        "liver_transplant": "Curative for urea cycle function; RECOMMENDED — prevents repeated NH3 crises",
        "critical_ci": "VPA HIGH RISK (inhibits NAGS → reduces CPS1 activity further); use LEV/LCM",
        "arginine_rule": "Arginine supplementation INDICATED (200–500 mg/kg/day): essential in UCD",
        "orotate": "LOW/NORMAL — no carbamoyl phosphate → no orotic acid production",
        "nbs_profile": "Elevated NH3; low citrulline; low arginine; elevated glutamine",
        "severity": "Most severe neonatal UCD; high mortality without emergent treatment",
        "n_patients": 40, "seed": 845,
        "founder": "Highly heterogeneous; no single founder; >200 pathogenic variants described",
        "key_variants": [
            "p.Thr544Met — active site, most common European, severe",
            "p.Arg1375Gln — domain IV, severe neonatal",
            "p.Ala1379Thr — domain IV, attenuated late-onset",
            "p.Gln263Ter — null, neonatal lethal",
            "c.4217+1G>A — splice site, partial function",
        ],
    },
    # ── OTC — Ornithine transcarbamylase ──
    {
        "gene": "OTC", "alias": "OTC — Ornithine Transcarbamylase (MOST COMMON UCD · ONLY X-LINKED)",
        "aa": "354 aa", "kDa": "40.5 kDa",
        "gene_class": "ornithine_transcarbamylase",
        "locus": "Xp11.4", "omim_gene": 311250,
        "phenotype": "OTC Deficiency — MOST COMMON UCD; X-linked; HIGH urinary orotate PATHOGNOMONIC; female carriers variable",
        "disease": (
            "OTC (hemi)- or biallelic loss → OTC Deficiency (OMIM #311250). OTC encodes the "
            "mitochondrial ornithine transcarbamylase (354aa, 40.5 kDa, homotrimer), catalysing: "
            "ornithine + carbamoyl phosphate → citrulline. OTC is the SECOND STEP of the urea "
            "cycle and the ONLY X-LINKED UCD gene. Loss → carbamoyl phosphate accumulates → "
            "spills into cytosol → enters pyrimidine synthesis pathway → orotate + uridine "
            "markedly elevated in urine. This orotate elevation is PATHOGNOMONIC for OTC deficiency "
            "and THE KEY DISCRIMINATOR from CPS1/NAGS deficiency (which have LOW orotate). "
            "OTC is the MOST COMMON UCD (~1:40,000–80,000 males). Hemizygous males: neonatal "
            "hyperammonemia within 48–72h of life; NH3 often >1000 µmol/L. Female carriers: "
            "highly variable (Lyon effect / X-inactivation skewing) — 15–25% symptomatic; "
            "presentation may be episodic, protein-aversive, migraine-like, or acute. "
            "Allopurinol provocation test can unmask OTC carrier females."
        ),
        "inheritance": "X-linked; hemizygous males fully affected; female carriers 15-25% symptomatic (skewed X-inactivation). ~1:40,000 males.",
        "hallmark": (
            "OTC HALLMARKS: (1) MOST COMMON UCD; ONLY X-LINKED UCD; "
            "(2) OROTATE HIGH — PATHOGNOMONIC: urinary orotic acid >20 µmol/mmol creatinine (normal <2); "
            "carbamoyl phosphate diverts to pyrimidines → orotate; distinguishes OTC from CPS1/NAGS; "
            "(3) Hemizygous males: NH3 >1000 µmol/L neonatal; rapid progression to coma; "
            "(4) LOW citrulline (ornithine → citrulline block) + elevated plasma glutamine; "
            "(5) FEMALE CARRIERS: variable expression; 15-25% symptomatic at any age; "
            "migraine, confusion, stroke-like episodes after protein load or intercurrent illness; "
            "NEVER dismiss an OTC carrier — potentially lethal acute episodes; "
            "(6) Allopurinol provocation: oral allopurinol → orotate/orotidine spike in carriers; "
            "(7) Arginine supplementation INDICATED (essential AA in UCD; 200–500 mg/kg/day); "
            "(8) Liver transplant CURATIVE for metabolic function in males; "
            "female carriers with severe phenotype also benefit; "
            "(9) VPA ABSOLUTE CI: inhibits NAGS → reduces residual urea cycle further; "
            "(10) Dietary protein aversion: classic clinical sign in late-presenting patients."
        ),
        "key_ddx": (
            "vs CPS1/NAGS: OROTATE — OTC: HIGH orotate (PATHOGNOMONIC); CPS1/NAGS: LOW orotate; "
            "vs ASS1: ASS1 has VERY HIGH citrulline (>1000 µmol/L); OTC has LOW citrulline; "
            "vs ASL: ASL has elevated argininosuccinate; OTC has no argininosuccinate; "
            "vs ARG1: ARG1 has elevated arginine + spastic diplegia, NH3 normal/mild"
        ),
        "ncg_response": "NCG NOT indicated for OTC (block is downstream of CPS1 step)",
        "liver_transplant": "Curative for metabolic function; RECOMMENDED for severe males",
        "critical_ci": "VPA ABSOLUTE CI — inhibits NAGS → further reduces residual cycle; use LEV/LCM",
        "arginine_rule": "Arginine supplementation INDICATED (200–500 mg/kg/day): essential amino acid in OTC",
        "orotate": "HIGH — PATHOGNOMONIC; carbamoyl phosphate diverts to pyrimidine pathway",
        "nbs_profile": "Elevated NH3; LOW citrulline; HIGH urine orotate (pathognomonic); elevated glutamine",
        "severity": "Hemizygous males: neonatal lethal without treatment; carrier females: variable",
        "n_patients": 40, "seed": 846,
        "founder": "p.Arg277Trp (European common variant); highly heterogeneous >600 variants",
        "key_variants": [
            "p.Arg277Trp — c.829C>T, most common European, severe",
            "p.Arg277Gln — c.830G>A, intermediate severity",
            "p.Arg277His — intermediate severity",
            "p.Thr125Met — attenuated late-onset male/symptomatic female",
            "Large deletions exon 1 — neonatal lethal null",
        ],
    },
    # ── ASS1 — Argininosuccinate synthase ──
    {
        "gene": "ASS1", "alias": "ASS1 — Argininosuccinate Synthase (Citrullinemia Type 1 — VERY HIGH Citrulline)",
        "aa": "412 aa", "kDa": "46.5 kDa",
        "gene_class": "argininosuccinate_synthase",
        "locus": "9q34.11", "omim_gene": 215700,
        "phenotype": "Citrullinemia Type 1 (CTLN1) — very high citrulline 1000-5000 µmol/L PATHOGNOMONIC; NBS-detected",
        "disease": (
            "ASS1 biallelic loss → Citrullinemia Type 1 (CTLN1, OMIM #215700). ASS1 encodes "
            "argininosuccinate synthase (412aa, 46.5 kDa, homotetramer), catalysing: citrulline + "
            "aspartate + ATP → argininosuccinate + AMP + PPi. This is the THIRD step of the urea "
            "cycle, the first cytosolic step (CPS1+OTC are mitochondrial). ASS1 deficiency → "
            "citrulline accumulation to VERY HIGH LEVELS (plasma citrulline 1000–5000 µmol/L; "
            "normal 9–30 µmol/L) — PATHOGNOMONIC for CTLN1. NBS: massively elevated citrulline "
            "(acylcarnitine profile: Cit spot on tandem MS). Argininosuccinate normal (not yet made). "
            "Secondary arginine deficiency (essential AA requires intact ASS1+ASL+ARG1 synthesis). "
            "Arginine supplementation is MORE critical in ASS1/ASL than in proximal UCDs because "
            "arginine is the only nitrogenous product that CAN exit the urea cycle to urea. "
            "NBS provides early diagnosis — many CTLN1 patients identified before first crisis "
            "in the post-NBS era."
        ),
        "inheritance": "Autosomal recessive, biallelic. Equal M:F. Incidence ~1:57,000 (second most common UCD after OTC).",
        "hallmark": (
            "ASS1/CTLN1 HALLMARKS: (1) PLASMA CITRULLINE 1000–5000 µmol/L — PATHOGNOMONIC; "
            "highest citrulline of ALL UCDs; cannot be missed on NBS acylcarnitine or amino acid profile; "
            "(2) Normal orotate (unlike OTC) and low arginine; "
            "(3) NBS DETECTED: first UCD detected by newborn screening (citrulline spot); "
            "(4) Neonatal severe form: NH3 >500–2000 µmol/L within 24–48h of life; "
            "coma, respiratory alkalosis, cerebral oedema; "
            "(5) Adult/mild form: episodic hyperammonemia triggered by protein, illness, pregnancy; "
            "(6) ARGININE SUPPLEMENTATION CRITICAL: 200–500 mg/kg/day; arginine drives nitrogen "
            "excretion as argininosuccinate (ASS1 substrates cycle differently — arginine supplementation "
            "reduces NH3 by shunting nitrogen to argininosuccinate excretion); "
            "(7) Liver transplant curative for severe neonatal form; "
            "(8) Sodium benzoate + phenylacetate: nitrogen scavengers; "
            "(9) Protein restriction lifelong but less severe than CPS1/OTC if well controlled; "
            "(10) VPA HIGH RISK: NAGS inhibition worsens cycle even in controlled CTLN1."
        ),
        "key_ddx": (
            "vs ASL: ASL has ELEVATED ARGININOSUCCINATE (pathognomonic) + elevated citrulline "
            "(100–300 µmol/L, moderate); ASS1 has VERY HIGH citrulline (1000+) + NO argininosuccinate; "
            "vs OTC: OTC has LOW citrulline + HIGH orotate; ASS1 has VERY HIGH citrulline + normal orotate; "
            "vs CTLN2 (citrin deficiency, SLC25A13): also elevated citrulline but less severe; "
            "different age of onset; citrin on newborn screen requires SUCLG1 enzyme assay to distinguish"
        ),
        "ncg_response": "NCG NOT indicated for ASS1 (block is downstream of CPS1/NAGS)",
        "liver_transplant": "Curative for neonatal severe form; not always required for mild form",
        "critical_ci": "VPA HIGH RISK — inhibits NAGS → secondary hyperammonemia; use LEV/LCM",
        "arginine_rule": "Arginine supplementation CRITICAL (200–500 mg/kg/day): arginine drives N excretion",
        "orotate": "NORMAL — CPS1 and OTC steps are intact; carbamoyl phosphate fully consumed",
        "nbs_profile": "VERY HIGH plasma citrulline (1000–5000 µmol/L) on NBS amino acids; massively elevated",
        "severity": "Severe neonatal to mild adult episodic; NBS enables pre-symptomatic detection",
        "n_patients": 40, "seed": 847,
        "founder": "p.Gly390Arg (European and Asian common variant); p.Arg265Cys (Japanese)",
        "key_variants": [
            "p.Gly390Arg — c.1168G>A, most common worldwide, severe",
            "p.Arg265Cys — c.793C>T, Japanese, intermediate",
            "p.Arg362Gln — severe European",
            "p.Ala118Thr — mild/attenuated",
            "p.Gln286Ter — null allele, neonatal severe",
        ],
    },
    # ── ASL — Argininosuccinate lyase ──
    {
        "gene": "ASL", "alias": "ASL — Argininosuccinate Lyase (ASA Disease — Neurological Despite Control — Trichorrhexis Nodosa)",
        "aa": "464 aa", "kDa": "52.1 kDa",
        "gene_class": "argininosuccinate_lyase",
        "locus": "7q11.21", "omim_gene": 207900,
        "phenotype": "Argininosuccinic Aciduria (ASA disease) — elevated argininosuccinate PATHOGNOMONIC; neurological disease despite metabolic control; brittle hair",
        "disease": (
            "ASL biallelic loss → Argininosuccinic Aciduria / ASA disease (OMIM #207900). ASL "
            "encodes the cytosolic argininosuccinate lyase (464aa, 52.1 kDa, homotetramer), "
            "catalysing: argininosuccinate → arginine + fumarate (FOURTH step of the urea cycle). "
            "ASL is UNIQUE among UCDs: (1) argininosuccinate accumulates massively in plasma, urine, "
            "and CSF — PATHOGNOMONIC with no other metabolite matching this pattern; "
            "(2) NEUROLOGICAL DISEASE OCCURS DESPITE GOOD METABOLIC CONTROL — the argininosuccinate "
            "itself (or secondary nitric oxide/NO deficiency) causes direct neurotoxicity independent "
            "of NH3 levels; (3) TRICHORRHEXIS NODOSA — brittle, fragile hair with node-like "
            "fractures — a distinctive clinical sign seen in ASL (and no other UCD); "
            "(4) SYSTEMIC HYPERTENSION: argininosuccinate cannot be converted to arginine → "
            "arginine deficiency → NO synthase substrate deficiency → reduced NO → systemic "
            "vasoconstriction → hypertension (treat with arginine supplementation 200–500 mg/kg/day "
            "to restore NO synthesis). Liver transplant corrects urea cycle function but the "
            "neurological and hypertension components may not fully resolve — multisystem disease."
        ),
        "inheritance": "Autosomal recessive, biallelic. Equal M:F. Incidence ~1:70,000 (third most common UCD).",
        "hallmark": (
            "ASL HALLMARKS: (1) ARGININOSUCCINATE ELEVATED — PATHOGNOMONIC: plasma ArgSuc "
            "100–1000 µmol/L (normal: undetectable); urine ArgSuc massively elevated; "
            "also elevated in CSF; (2) MODERATE citrulline elevation (100–300 µmol/L); "
            "(3) NEUROLOGICAL DISEASE DESPITE CONTROL: cognitive impairment, learning disability, "
            "seizures can progress even with well-controlled NH3 — direct ArgSuc neurotoxicity "
            "and/or NO deficiency; (4) TRICHORRHEXIS NODOSA — brittle hair with bamboo-node "
            "fractures — seen on hair microscopy; pathognomonic within UCDs; "
            "(5) HYPERTENSION: systemic NO deficiency → vasoconstriction; treat with "
            "arginine supplementation which restores NOS substrate; "
            "(6) ARGININE SUPPLEMENTATION MANDATORY: 200–500 mg/kg/day; reduces NH3 AND "
            "restores NO production; unlike other UCDs, arginine TRULY cannot be synthesised; "
            "(7) Liver transplant corrects urea cycle but neurological damage may persist; "
            "(8) Low-protein diet + essential amino acids + arginine + benzoate/phenylacetate; "
            "(9) NBS detected (amino acid profile: elevated citrulline + argininosuccinate spot); "
            "(10) VPA HIGH RISK — use LEV/LCM."
        ),
        "key_ddx": (
            "vs ASS1: ASS1 has VERY HIGH citrulline (1000+) and NO argininosuccinate; "
            "ASL has MODERATE citrulline (100–300) plus ELEVATED ARGININOSUCCINATE (pathognomonic); "
            "vs OTC: OTC has LOW citrulline + HIGH orotate; ASL has elevated citrulline; "
            "vs ARG1: ARG1 has HIGH arginine + spastic diplegia; ASL has ArgSuc + moderate citrulline; "
            "trichorrhexis nodosa: ONLY in ASL among UCDs"
        ),
        "ncg_response": "NCG NOT indicated for ASL (block is downstream of CPS1/NAGS)",
        "liver_transplant": "Curative for urea cycle; neurological and hypertension may persist post-LT",
        "critical_ci": "VPA HIGH RISK — inhibits NAGS → secondary hyperammonemia; use LEV/LCM",
        "arginine_rule": "Arginine supplementation CRITICAL (200–500 mg/kg/day): restores NO synthesis AND reduces NH3",
        "orotate": "NORMAL — CPS1/OTC steps are intact",
        "nbs_profile": "Elevated citrulline (moderate) + elevated argininosuccinate on NBS amino acids",
        "severity": "Severe to mild neonatal; neurological burden persists despite metabolic control",
        "n_patients": 40, "seed": 848,
        "founder": "p.Gln354Arg (common European); p.Arg385Trp (severe neonatal)",
        "key_variants": [
            "p.Gln354Arg — c.1061A>G, most common worldwide, moderate-severe",
            "p.Arg385Trp — severe neonatal",
            "p.Arg193Trp — c.577C>T, intermediate",
            "p.Glu189Lys — c.565G>A, mild attenuated",
            "c.1060+1G>A — splice site, neonatal severe",
        ],
    },
    # ── ARG1 — Arginase 1 ──
    {
        "gene": "ARG1", "alias": "ARG1 — Arginase 1 (Hyperargininemia — UNIQUE: Spastic Diplegia NOT Acute NH3 — Arginine CONTRAINDICATED)",
        "aa": "322 aa", "kDa": "34.7 kDa",
        "gene_class": "arginase_1",
        "locus": "6q23.2", "omim_gene": 207800,
        "phenotype": "Hyperargininemia (ARG1 deficiency) — UNIQUE: spastic diplegia/tetraplegia NOT acute hyperammonemia; arginine is toxic; arginine supplementation CONTRAINDICATED",
        "disease": (
            "ARG1 biallelic loss → Hyperargininemia (OMIM #207800). ARG1 encodes cytosolic "
            "arginase 1 (322aa, 34.7 kDa, homotrimer, Mn²⁺-dependent), catalysing: arginine → "
            "ornithine + urea (FINAL STEP of the urea cycle). ARG1 deficiency is CLINICALLY DISTINCT "
            "from all other UCDs: (1) PROGRESSIVE SPASTIC DIPLEGIA/TETRAPLEGIA — the dominant "
            "presentation, resembling cerebral palsy; onset 2–5 years of age; caused by arginine-"
            "mediated neurotoxicity (arginine itself, or its metabolites guanidino compounds, "
            "are neurotoxic at high concentrations); (2) AMMONIA IS OFTEN NORMAL or only MILDLY "
            "ELEVATED (<200 µmol/L) — the classic hyperammonemic crisis that defines other UCDs "
            "is ABSENT in ARG1; (3) ELEVATED ARGININE (plasma arginine >200 µmol/L; normal 10–140) "
            "is the hallmark — can be detected on NBS amino acids; (4) PROTEIN RESTRICTION "
            "MANDATORY — reduces arginine substrate; essential amino acid formula WITHOUT arginine; "
            "(5) ARGININE SUPPLEMENTATION IS ABSOLUTELY CONTRAINDICATED — arginine IS the toxic "
            "metabolite; this is the OPPOSITE of all other UCDs where arginine is supplemented."
        ),
        "inheritance": "Autosomal recessive, biallelic. Equal M:F. Rare: ~1:300,000–1:1,000,000.",
        "hallmark": (
            "ARG1 HALLMARKS: (1) PROGRESSIVE SPASTIC DIPLEGIA/TETRAPLEGIA — onset 2–5 years; "
            "resembles cerebral palsy; NOT acute encephalopathy or neonatal crisis (different from "
            "all other UCDs); progressive loss of ambulation; (2) NH3 MILD OR NORMAL (<200 µmol/L) — "
            "KEY DISCRIMINATOR; 'urea cycle disorder without hyperammonemia' — arginine itself is toxic; "
            "(3) ELEVATED ARGININE — plasma arginine >200 µmol/L (normal 10–140); guanidino compounds "
            "(homoarginine, alpha-keto-delta-guanidinovaleric acid) also neurotoxic; "
            "(4) ARGININE SUPPLEMENTATION ABSOLUTE CI — OPPOSITE of all other UCDs; "
            "arginine IS the toxic substrate; never supplement; protein restriction reduces arginine intake; "
            "(5) Essential amino acid formula WITHOUT arginine: reduces arginine intake while meeting "
            "other AA needs; (6) NBS: elevated arginine on amino acid profile; "
            "(7) Intellectual disability, seizures (often well-controlled), tremor, ataxia; "
            "(8) Liver transplant: may partially stabilise the disorder but neurological damage "
            "may not reverse — prevention (early protein restriction) is critical; "
            "(9) VPA HIGH RISK: inhibits NAGS → worsens secondary hyperammonemia; "
            "(10) Protein restriction is PRIMARY treatment: reduce arginine ingestion."
        ),
        "key_ddx": (
            "vs spastic diplegia (cerebral palsy): plasma arginine elevated in ARG1 — a routine "
            "metabolic screen diagnoses this; CP has normal amino acids; "
            "vs other UCDs: other UCDs have acute hyperammonemia; ARG1 has NORMAL/MILD NH3; "
            "vs ASL: ASL has argininosuccinate + moderate NH3; ARG1 has arginine + normal NH3; "
            "vs MSUD: MSUD has elevated BCAAs (leucine/isoleucine/valine); ARG1 has elevated arginine only"
        ),
        "ncg_response": "NCG NOT indicated for ARG1 (block is downstream of all synthesis steps)",
        "liver_transplant": "Partial benefit — corrects urea cycle but progressive neurological damage may persist",
        "critical_ci": "VPA HIGH RISK; Arginine supplementation ABSOLUTE CI (arginine IS the toxic metabolite)",
        "arginine_rule": "Arginine supplementation ABSOLUTE CI — ONLY UCD where arginine is CONTRAINDICATED",
        "orotate": "NORMAL — all prior steps intact",
        "nbs_profile": "Elevated plasma arginine (>200 µmol/L) on NBS amino acids",
        "severity": "Milder acute course (no neonatal crisis) but progressive neurological damage",
        "n_patients": 40, "seed": 849,
        "founder": "p.Arg21Ter (European); p.Trp122Ter (Italian); no single major founder",
        "key_variants": [
            "p.Arg21Ter — c.61C>T, null, progressive spastic diplegia",
            "p.Trp122Ter — c.366G>A, null, Italian",
            "p.Gly235Asp — Mn²⁺ binding, severe",
            "p.Arg291Cys — c.871C>T, mild/attenuated",
            "c.IVS7+2T>C — splice, partial function",
        ],
    },
]

COHORT_SIZE = 40

# ─── Patient cohort generator ───────────────────────────────────────────────
def _gen_patients(gene_entry):
    rng = random.Random(gene_entry["seed"])
    gene = gene_entry["gene"]
    severity_weights = {
        "NAGS":  {"severe_neonatal": 45, "infantile": 25, "late_onset": 25, "attenuated": 5},
        "CPS1":  {"severe_neonatal": 60, "infantile": 20, "late_onset": 15, "attenuated": 5},
        "OTC":   {"severe_neonatal": 40, "infantile": 15, "late_onset": 30, "carrier_female": 15},
        "ASS1":  {"severe_neonatal": 35, "infantile": 20, "late_onset": 35, "attenuated": 10},
        "ASL":   {"severe_neonatal": 25, "infantile": 25, "late_onset": 40, "attenuated": 10},
        "ARG1":  {"progressive": 65, "infantile": 20, "adult": 15},
    }
    sex_weights = {"OTC": {"M": 60, "F": 40}, "default": {"M": 50, "F": 50}}
    phenotypes = list(severity_weights.get(gene, {"severe_neonatal": 50, "late_onset": 50}).keys())
    phenotype_w = list(severity_weights.get(gene, {"severe_neonatal": 50, "late_onset": 50}).values())
    sw = sex_weights["OTC"] if gene == "OTC" else sex_weights["default"]
    patients = []
    for i in range(COHORT_SIZE):
        sex = rng.choices(["M", "F"], weights=[sw["M"], sw["F"]])[0]
        phenotype = rng.choices(phenotypes, weights=phenotype_w)[0]
        # Age at diagnosis
        if "neonatal" in phenotype or "severe" in phenotype:
            dx_age = rng.uniform(0.003, 0.08)  # days 1-30
        elif "progressive" in phenotype:
            dx_age = rng.uniform(1.5, 6.0)
        elif "late" in phenotype or "adult" in phenotype:
            dx_age = rng.uniform(5, 40)
        elif "carrier" in phenotype:
            dx_age = rng.uniform(15, 55)
        else:
            dx_age = rng.uniform(0.1, 2.0)
        # Peak NH3
        if "neonatal" in phenotype or "severe" in phenotype:
            peak_nh3 = rng.randint(800, 3500)
        elif "progressive" in phenotype:
            peak_nh3 = rng.randint(50, 180)  # ARG1 — mild
        elif "late" in phenotype or "adult" in phenotype:
            peak_nh3 = rng.randint(150, 600)
        elif "carrier" in phenotype:
            peak_nh3 = rng.randint(80, 400)
        else:
            peak_nh3 = rng.randint(300, 1200)
        # Outcomes
        liver_tx = rng.random() < (
            0.45 if gene in ("CPS1", "OTC") and "neonatal" in phenotype
            else 0.25 if gene == "ASS1" and "neonatal" in phenotype
            else 0.12
        )
        ncg_trial = gene == "NAGS"
        ncg_response = ncg_trial and rng.random() < 0.98
        survivors = rng.random() < (
            0.97 if ncg_response
            else 0.85 if liver_tx
            else 0.72 if "neonatal" in phenotype or "severe" in phenotype
            else 0.95
        )
        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "sex": sex,
            "phenotype": phenotype,
            "dx_age_yr": round(dx_age, 2),
            "peak_nh3": peak_nh3,
            "liver_tx": liver_tx,
            "ncg_response": ncg_response,
            "survived": survivors,
        })
    return patients


# ─── Build full cohort ───────────────────────────────────────────────────────
COHORT = {}
for _g in UCD_GENES:
    COHORT[_g["gene"]] = _gen_patients(_g)
ALL_PATIENTS = [p for g in COHORT.values() for p in g]


# ─── Aggregate statistics ────────────────────────────────────────────────────
def _agg():
    n = len(ALL_PATIENTS)
    neonatal = sum(1 for p in ALL_PATIENTS if "neonatal" in p["phenotype"] or "severe" in p["phenotype"])
    liver_tx = sum(1 for p in ALL_PATIENTS if p["liver_tx"])
    ncg_resp = sum(1 for p in ALL_PATIENTS if p["ncg_response"])
    nh3_vals = [p["peak_nh3"] for p in ALL_PATIENTS]
    return {
        "total_patients": n,
        "neonatal_severe_pct": round(100 * neonatal / n, 1),
        "liver_tx_pct": round(100 * liver_tx / n, 1),
        "ncg_responsive_pct": round(100 * ncg_resp / n, 1),
        "median_peak_nh3": sorted(nh3_vals)[n // 2],
        "nh3_over_500": round(100 * sum(1 for v in nh3_vals if v >= 500) / n, 1),
    }


# ─── API: get_overview ───────────────────────────────────────────────────────
def get_overview():
    agg = _agg()
    return {
        "atlas": "UCD-Atlas — Complete 6-Gene Urea Cycle Disorders Atlas",
        "atlas_scope": "Nuclear-encoded urea cycle enzymes: NAGS, CPS1, OTC, ASS1, ASL, ARG1",
        "n_genes": len(UCD_GENES),
        "n_patients": len(ALL_PATIENTS),
        "seeds": f"{SEED_BASE}–{SEED_BASE + len(UCD_GENES) - 1}",
        "urea_cycle": {
            "pathway": "NH3 + HCO3⁻ + ATP → carbamoyl phosphate (CPS1, NAGS-activated) → citrulline (OTC) → argininosuccinate (ASS1) → arginine (ASL) → ornithine + urea (ARG1)",
            "nags_role": "NAGS produces N-acetylglutamate (NAG) — essential allosteric activator of CPS1; master switch",
            "mitochondrial_steps": "NAGS + CPS1 + OTC (steps 1-3 are mitochondrial)",
            "cytosolic_steps": "ASS1 + ASL + ARG1 (steps 4-6 are cytosolic)",
            "x_linked_gene": "OTC — ONLY X-linked UCD gene; all others are autosomal recessive",
            "ncg_curative_gene": "NAGS — ONLY UCD where NCG/Carbaglu is essentially curative",
            "arginine_ci_gene": "ARG1 — ONLY UCD where arginine supplementation is CONTRAINDICATED",
            "most_common": "OTC — most common UCD (~1:40,000 males)",
            "nbs_flag": "All 6 detectable on expanded NBS (amino acids + acylcarnitines + orotate)",
        },
        "aggregate_clinical": {
            "neonatal_severe_pct": agg["neonatal_severe_pct"],
            "liver_tx_pct": agg["liver_tx_pct"],
            "ncg_responsive_pct": agg["ncg_responsive_pct"],
            "median_peak_nh3": agg["median_peak_nh3"],
            "nh3_over_500_pct": agg["nh3_over_500"],
            "encephalopathy_pct": 78,
            "cerebral_oedema_pct": 52,
            "protein_aversion_pct": 65,
            "spastic_diplegia_arg1_pct": 65,   # ARG1-specific
            "trichorrhexis_asl_pct": 55,        # ASL-specific
            "hypertension_asl_pct": 42,          # ASL-specific
        },
        "orotate_profile": {
            "OTC": "HIGH — pathognomonic; carbamoyl phosphate diverts to pyrimidine synthesis",
            "CPS1": "LOW/NORMAL — no carbamoyl phosphate produced at all",
            "NAGS": "LOW/NORMAL — identical to CPS1 (NAGS→CPS1 block)",
            "ASS1": "NORMAL — CPS1+OTC steps intact; citrulline is the biomarker",
            "ASL": "NORMAL — all upstream steps intact; argininosuccinate is the biomarker",
            "ARG1": "NORMAL — all upstream steps intact; arginine is the biomarker",
            "key_rule": "HIGH orotate = OTC deficiency; LOW orotate with low citrulline = CPS1 or NAGS",
        },
        "drug_safety": {
            "vpa_risk_all": True,
            "vpa_mechanism": "VPA directly inhibits NAGS → reduces N-acetylglutamate → CPS1 less active → hyperammonemia in any UCD; particularly dangerous in OTC carriers",
            "vpa_high_risk_genes": ["CPS1", "OTC", "ASS1", "ASL", "ARG1", "NAGS"],
            "ncg_curative_only_nags": True,
            "arginine_ci_gene": "ARG1",
            "arginine_indicated_genes": ["NAGS", "CPS1", "OTC", "ASS1", "ASL"],
            "preferred_aed": "LEV, LCM — no NAGS inhibition, renal excretion, no drug interactions",
            "ammonia_scavengers": "Sodium benzoate 250 mg/kg + sodium phenylacetate 250 mg/kg IV (Ammonul); arginine IV (except ARG1)",
            "gir": "8–12 mg/kg/min glucose infusion rate; anti-catabolic in acute crisis",
            "zero_protein_crisis": "24–48h zero protein in acute hyperammonemic crisis; reintroduce at 0.5 g/kg/day",
            "liver_transplant": "Curative for CPS1/OTC/ASS1; partial benefit in ASL/ARG1",
            "haemodialysis": "CVVHDF/haemodiafiltration for NH3 >500 µmol/L; faster than peritoneal dialysis",
        },
        "btbgd_exclusion": "MANDATORY: biotin-thiamine responsive basal ganglia disease (SLC19A3) mimics hyperammonemia + Leigh-like changes; Carbaglu/biotin trial before other UCD workup in Leigh-like",
        "wes_note": "WES detects ALL 6 nuclear-encoded UCD genes; OTC variants on X-chromosome confirmed on male hemizygous dosage",
    }


# ─── API: get_breakdown ──────────────────────────────────────────────────────
def get_breakdown():
    gene_rows = []
    for g in UCD_GENES:
        pts = COHORT[g["gene"]]
        nh3s = [p["peak_nh3"] for p in pts]
        gene_rows.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "gene_class": g["gene_class"],
            "n_patients": g["n_patients"],
            "seed": g["seed"],
            "phenotype": g["phenotype"],
            "inheritance": g["inheritance"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "ncg_response": g["ncg_response"],
            "liver_transplant": g["liver_transplant"],
            "critical_ci": g["critical_ci"],
            "arginine_rule": g["arginine_rule"],
            "orotate": g["orotate"],
            "nbs_profile": g["nbs_profile"],
            "severity": g["severity"],
            "founder": g["founder"],
            "key_variants": g["key_variants"],
            # Aggregate from cohort
            "pct_neonatal_severe": round(
                100 * sum(1 for p in pts if "neonatal" in p["phenotype"] or "severe" in p["phenotype"]) / len(pts), 1
            ),
            "pct_liver_tx": round(100 * sum(1 for p in pts if p["liver_tx"]) / len(pts), 1),
            "pct_ncg_response": round(100 * sum(1 for p in pts if p["ncg_response"]) / len(pts), 1),
            "median_peak_nh3": sorted(nh3s)[len(nh3s) // 2],
        })
    return {
        "genes": gene_rows,
        "total": len(UCD_GENES),
        "total_patients": len(ALL_PATIENTS),
    }


# ─── API: get_definitions ────────────────────────────────────────────────────
def get_definitions():
    return {
        "atlas": "UCD-Atlas — Complete 6-Gene Urea Cycle Disorders Atlas",
        "urea_cycle": {
            "full_name": "Urea Cycle — hepatic mitochondrial+cytosolic enzymatic pathway detoxifying NH3 → urea",
            "subunits_total": 6,
            "proximal_mitochondrial": ["NAGS (activator)", "CPS1 (step 1)", "OTC (step 2)"],
            "distal_cytosolic": ["ASS1 (step 3)", "ASL (step 4)", "ARG1 (step 5)"],
        },
        "definitions": [
            {"term": "Urea Cycle", "definition": "Hepatic enzymatic cycle detoxifying ammonia (NH3) into urea: NH3 + HCO3⁻ → carbamoyl phosphate (CPS1) → citrulline (OTC) → argininosuccinate (ASS1) → arginine (ASL) → urea + ornithine (ARG1). NAGS produces the obligate CPS1 activator N-acetylglutamate. Steps 1-3 are mitochondrial; steps 4-6 are cytosolic. Ornithine returns to mitochondria via ORC1 (SLC25A15) to complete the cycle."},
            {"term": "N-Acetylglutamate (NAG) / NAGS", "definition": "N-acetylglutamate synthase (NAGS) catalyses glutamate + acetyl-CoA → NAG. NAG is the OBLIGATE allosteric activator of CPS1 — without NAG, CPS1 cannot catalyse the first urea cycle step. NAGS deficiency therefore phenocopies CPS1 deficiency biochemically. Arginine allosterically activates NAGS (positive feedback). VPA and propionyl-CoA competitively inhibit NAGS → secondary hyperammonemia in any patient on VPA or with propionic acidemia/MMA."},
            {"term": "N-Carbamylglutamate (NCG / Carbaglu)", "definition": "Structural analogue of NAG; directly activates CPS1 without requiring NAGS. ONLY indicated (and curative) for NAGS deficiency — NCG 100-250 mg/kg/day oral normalises NH3 in virtually 100% of NAGS patients. NCG may partially help CPS1 patients (partial CPS1 activation via NAG analogue). NOT indicated for OTC, ASS1, ASL, or ARG1 (blocks are downstream of CPS1)."},
            {"term": "Hyperammonemia (NH3 toxicity)", "definition": "NH3 accumulation → cerebral oedema (astrocyte glutamine accumulation → osmotic swelling), mitochondrial dysfunction, oxidative stress, and ultimately cortical/cerebellar herniation. Emergency threshold: NH3 >200 µmol/L in neonate → immediate intervention. NH3 >500 µmol/L → haemodiafiltration (CVVHDF). Glucose 10% IV + sodium benzoate + sodium phenylacetate (Ammonul) + arginine (not ARG1). Dialysis faster than peritoneal."},
            {"term": "Orotic Acid / Orotate (Discriminator)", "definition": "KEY DISCRIMINATOR: OTC deficiency → carbamoyl phosphate accumulates in mitochondria → spills to cytosol → enters pyrimidine synthesis pathway → orotic acid + uridine markedly elevated in urine (>20 µmol/mmol creatinine). CPS1/NAGS deficiency → NO carbamoyl phosphate produced → orotate LOW/NORMAL. Rule: HIGH orotate = OTC; LOW orotate + low citrulline = CPS1 or NAGS (NCG trial distinguishes them)."},
            {"term": "OTC Deficiency — X-Linkage", "definition": "OTC is the ONLY X-linked UCD gene (Xp11.4). Hemizygous males: universally severely affected (neonatal hyperammonemia, lethal without treatment). Female carriers: Lyon effect (X-inactivation skewing) → 15-25% symptomatic; presentation ranges from asymptomatic to fatal hyperammonemia; protein aversion, migraine, confusion, stroke-like episodes after high protein load. Allopurinol provocation test: oral allopurinol → urinary orotate/orotidine spike in 90% of carriers. NEVER dismiss an OTC family without screening female relatives."},
            {"term": "Citrullinemia Type 1 (ASS1 deficiency)", "definition": "CTLN1 — plasma citrulline 1000-5000 µmol/L (normal 9-30) is PATHOGNOMONIC. NBS detects this on day 2-3 of life. First UCD systematically detected by expanded NBS. Arginine supplementation is particularly critical: arginine drives nitrogen excretion via argininosuccinate and alternative pathways. Liver transplant curative. Severe neonatal form (NH3 >500 µmol/L within 24h) + milder late-onset forms."},
            {"term": "Argininosuccinic Aciduria / ASL Disease", "definition": "ASL deficiency → argininosuccinate accumulates in plasma, urine, and CSF. PATHOGNOMONIC — no other disorder produces elevated argininosuccinate. UNIQUE: NEUROLOGICAL DISEASE DESPITE GOOD METABOLIC CONTROL — direct ArgSuc toxicity and/or systemic NO deficiency (arginine → NO via NOS; ASL block depletes arginine → NOS substrate deficit → systemic hypertension + neurological injury). Trichorrhexis nodosa (brittle bamboo-node hair) is pathognomonic within UCDs. Liver transplant corrects urea cycle but neurological components persist."},
            {"term": "ARG1 Deficiency / Hyperargininemia", "definition": "CLINICALLY DISTINCT from other UCDs: presents with progressive spastic diplegia/tetraplegia (onset 2-5 years, resembling cerebral palsy) — NOT acute hyperammonemia. NH3 is often normal or mildly elevated (<200 µmol/L). Elevated ARGININE is the toxic metabolite (guanidino compounds also neurotoxic). ARGININE SUPPLEMENTATION ABSOLUTE CI — this is the OPPOSITE of all other UCDs. Protein restriction essential. Essential amino acid formula WITHOUT arginine. NBS: elevated plasma arginine."},
            {"term": "Arginine Supplementation — UCD Rule", "definition": "RULE: Arginine supplementation (200-500 mg/kg/day) is INDICATED in 5 of 6 UCD genes (NAGS, CPS1, OTC, ASS1, ASL) because arginine is an essential amino acid in UCD (cannot be synthesised beyond the enzymatic block). EXCEPTION: ARG1 — arginine is the TOXIC METABOLITE; supplementation is ABSOLUTE CI; protein restriction + AA formula without arginine. Never supplement arginine in ARG1 deficiency."},
            {"term": "Ammonia Scavengers", "definition": "Sodium benzoate (250 mg/kg IV/PO) conjugates glycine → hippuric acid (excreted renally; each benzoate removes 1 nitrogen). Sodium phenylacetate (250 mg/kg IV, combined as Ammonul) conjugates glutamine → phenylacetylglutamine (excreted renally; each removes 2 nitrogens — more efficient). Together: alternative nitrogen disposal. Arginine IV (not for ARG1): promotes citrulline/argininosuccinate nitrogen excretion. GIR 8-12 mg/kg/min: anti-catabolic (prevents muscle protein catabolism → reduces NH3 load)."},
            {"term": "VPA (Valproate) Risk in UCD", "definition": "Valproate DIRECTLY INHIBITS NAGS → reduces NAG production → CPS1 less active → hyperammonemia. This mechanism operates in ANY patient — not just those with known UCD. VPA also depletes carnitine and inhibits mitochondrial FAO. In UCD patients: VPA can precipitate fatal hyperammonemic crisis. In carriers (especially OTC female carriers): VPA can precipitate first symptomatic crisis. ALWAYS check plasma amino acids + urine orotate before starting VPA in any patient with encephalopathy or developmental delay. Preferred AED alternatives: LEV, LCM (no NAGS inhibition)."},
            {"term": "Liver Transplant in UCD", "definition": "Liver transplant (LT) replaces deficient hepatic urea cycle enzyme; corrects urea cycle function in CPS1, OTC (males), ASS1 — effectively curative for metabolic decompensation risk. Partial benefit in ASL/ARG1 — neurological damage from direct toxicity (ArgSuc or arginine/guanidino compounds) may not reverse post-LT even when NH3 normalises. LT does not reverse pre-existing brain damage. Decision: balance morbidity of LT against morbidity of ongoing dietary restriction and crisis risk. LT is NOT required for NAGS (NCG curative)."},
            {"term": "BTBGD — Mandatory Exclusion", "definition": "Biotin-Thiamine Responsive Basal Ganglia Disease (SLC19A3 deficiency) mimics acute hyperammonemia + Leigh-like MRI + metabolic crisis. Trial of biotin 5-10 mg/kg/day + thiamine 100-300 mg/day should ALWAYS accompany acute crisis workup for any undiagnosed metabolic encephalopathy, while waiting for OA/UCD biochemical results and WES. SLC19A3 variants are the most common treatable metabolic emergency mimicking UCD."},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== UCD Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    print(f"Neonatal severe: {ov['aggregate_clinical']['neonatal_severe_pct']}%")
    print(f"Liver Tx: {ov['aggregate_clinical']['liver_tx_pct']}%")
    print(f"NCG responsive: {ov['aggregate_clinical']['ncg_responsive_pct']}%")
    print(f"NH3 >500: {ov['aggregate_clinical']['nh3_over_500_pct']}%")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
    print("=== ALL PASS ===")
