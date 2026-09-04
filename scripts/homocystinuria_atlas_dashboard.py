#!/usr/bin/env python3
"""Homocystinuria-Atlas — Complete 8-Gene Homocystinuria & Remethylation Disorders Atlas
CBS (Classical HCU type I) · MTHFR (MTHFR Deficiency) · MTR (cblG) · MTRR (cblE) ·
MMACHC (cblC — most common combined MMA+HC) · MMADHC (cblD) ·
AHCY (Adenosylhomocysteinase Deficiency) · MAT1A (MAT I/III Deficiency)
320-patient aggregate cohort (8 × 40, seeds 950–957)

Homocystinuria & Remethylation Disorders facts:
  - All disorders share ELEVATED TOTAL PLASMA HOMOCYSTEINE (tHcy) as the unifying biomarker.
    Severe: tHcy >100 µmol/L; moderate: 30-100; mild: 15-30 (normal <15 µmol/L).
  - The methionine cycle: Met → SAM → SAH → Homocysteine → (remethylation or transsulfuration).
    Remethylation: Hcy + 5-MTHF → Met (via MTR/MTRR + methylcobalamin); OR Hcy + betaine → Met (BHMT).
    Transsulfuration: Hcy → cystathionine → cysteine (via CBS + B6/PLP).
  - KEY TEACHING POINTS:
      CBS LENS DISLOCATION PATHOGNOMONIC: ectopia lentis (inferior/nasal) is PATHOGNOMONIC
        for CBS-type HCU; opposite direction from Marfan (FBN1) where lens subluxes superior/temporal.
      PYRIDOXINE (B6) TRIAL MANDATORY: ~50% of CBS patients are B6-responsive; empiric B6
        (200-1000 mg/day) trial is mandatory at diagnosis; responders have 50% reduction in tHcy.
      THROMBOEMBOLISM #1 CAUSE OF DEATH: venous and arterial thromboembolism is the
        main morbidity and mortality driver in untreated CBS; risk by age 30: >50% without treatment.
      BETAINE CENTRAL: betaine (trimethylglycine) remethylates homocysteine via BHMT;
        reduces tHcy in ALL forms of homocystinuria (CBS, MTHFR, cblC, cblD, cblE, cblG, AHCY).
      MTHFR vs CBS: MTHFR deficiency has LOW methionine (block in remethylation → Hcy accumulates,
        Met cannot be made); CBS has HIGH methionine (block in transsulfuration → Hcy + Met both rise).
      cblC (MMACHC): MOST COMMON combined MMA+Homocystinuria; retinal disease
        (macular degeneration + nystagmus) is PATHOGNOMONIC; ONLY cobalamin disorder
        with major ophthalmological disease. Hydroxocobalamin NOT cyanocobalamin.
      NBS DETECTION: CBS: elevated methionine on MS/MS. cblC: elevated C3-carnitine.
        MTHFR/MTR/MTRR: often MISSED by NBS (methionine normal or low).
      MARFANOID HABITUS: CBS HCU mimics Marfan (tall, arachnodactyly, dolichostenomelia)
        but MARFAN has superior lens subluxation + aortic root dilation; CBS has inferior lens +
        thromboembolism (no aortic root); the KEY clinical DDx.

COHORT: 8 × 40 = 320 patient slots (seeds 950–957; gene-specific seeds)
"""

import random

SEED_BASE = 950

HCU_GENES = [
    # ── CBS — Classical Homocystinuria Type I ─────────────────────────────────────────
    {
        "gene": "CBS", "protein": "Cystathionine Beta-Synthase",
        "alias": "CBS — Classical Homocystinuria type I; most common form 50-70% (OMIM #236200)",
        "aa": "551 aa", "kDa": "63 kDa",
        "gene_class": (
            "Cystathionine beta-synthase: PLP (pyridoxal-5'-phosphate / vitamin B6)-dependent enzyme; "
            "catalyses the first step of the transsulfuration pathway — condensation of homocysteine + serine "
            "→ cystathionine; located in the cytoplasm; homotetrameric; regulated by SAM (allosteric activator); "
            "CBS is the ONLY enzyme that removes homocysteine via transsulfuration"
        ),
        "hcu_subgroup": "Classical Homocystinuria (HCU type I) — transsulfuration block",
        "locus": "21q22.3", "omim_gene": 613381,
        "inheritance": "AR. 21q22.3. Both sexes equally. Incidence ~1/200,000-335,000 worldwide; 1/65,000 Ireland (founder).",
        "onset_range_y": (0.0, 10.0),
        "phenotype": (
            "Classical HCU triad: (1) OPHTHALMOLOGICAL — ectopia lentis (inferior/nasal lens dislocation, "
            "by age 10 in ~90%); myopia; glaucoma; (2) SKELETAL — Marfanoid habitus (tall, arachnodactyly, "
            "dolichostenomelia, pectus, scoliosis, genu valgum); osteoporosis; (3) VASCULAR — "
            "thromboembolism (venous and arterial, #1 cause of early death, risk >50% by age 30 untreated); "
            "stroke; myocardial infarction. NEUROLOGICAL: intellectual disability (IQ 10-20 points below "
            "expected in B6-non-responsive); psychiatric features; seizures (less common)."
        ),
        "disease": (
            "CBS encodes cystathionine beta-synthase (551aa, 63kDa). CBS is a PLP-dependent "
            "homotetrameric enzyme that catalyses homocysteine + serine → cystathionine "
            "(first step of transsulfuration pathway). CBS deficiency → homocysteine accumulates → "
            "methionine also accumulates (upstream) → cystathionine absent → cysteine depleted.\n\n"
            "Biochemical hallmarks: elevated plasma total homocysteine (tHcy often >100 µmol/L, "
            "normal <15); elevated methionine (>100 µmol/L, normal 20-50); cystine LOW; "
            "urine homocystine strongly positive; urine nitroprusside test positive (screening).\n\n"
            "B6-RESPONSIVENESS: ~50% of CBS patients are pyridoxine (B6)-responsive — B6 stabilises "
            "the PLP cofactor-binding site, partially restoring CBS activity. Defined as >50% reduction "
            "in tHcy on supplemental B6 (200-1000 mg/day). B6-responsive patients have milder phenotype, "
            "later onset ectopia lentis, lower thrombosis risk. B6 trial MANDATORY at diagnosis (3 months).\n\n"
            "TREATMENT: (a) B6-responsive: pyridoxine + folate + B12 (to support remethylation); "
            "betaine 6-9 g/day (alternative remethylation via BHMT); moderate methionine restriction. "
            "(b) B6-non-responsive: methionine-restricted diet (low-Met formula + low natural protein) + "
            "betaine 6-9 g/day + folate + B12 + aspirin/anticoagulation for thromboprophylaxis.\n\n"
            "NBS: elevated methionine on MS/MS (≥50 µmol/L; varies by laboratory). False-negative risk "
            "in B6-responsive or mild variants. Confirmatory: plasma amino acids + urine organic acids + "
            "CBS enzyme assay + CBS sequencing.\n\n"
            "p.Ile278Thr — most common B6-non-responsive allele worldwide; associated with severe phenotype. "
            "p.Gly307Ser — common in certain European populations. Irish founder allele (c.833T>C; p.Ile278Thr)."
        ),
        "hallmark": (
            "CBS HALLMARKS: "
            "(1) ECTOPIA LENTIS INFERIOR/NASAL — PATHOGNOMONIC (vs Marfan: superior/temporal). "
            "(2) METHIONINE HIGH — transsulfuration block backs up both Hcy AND Met (vs MTHFR/cblG/cblE: Met LOW). "
            "(3) THROMBOEMBOLISM #1 CAUSE MORBIDITY-MORTALITY — anticoagulation life-saving. "
            "(4) PYRIDOXINE (B6) TRIAL MANDATORY — 50% respond; B6 stabilises PLP cofactor. "
            "(5) BETAINE KEY TREATMENT — remethylates Hcy via BHMT pathway independent of B12/folate. "
            "(6) MARFANOID BUT NO AORTIC ROOT — mimics Marfan, FBN1 negative, aorta normal. "
            "(7) NBS ELEVATED METHIONINE — detected by MS/MS if methionine >50 µmol/L. "
            "(8) CYSTATHIONINE ABSENT IN PLASMA — transsulfuration block: cystathionine cannot be formed."
        ),
        "nbs_marker": "Elevated methionine on MS/MS; urine homocystine positive",
        "key_biomarker": "Plasma tHcy >100 µmol/L + methionine HIGH + cystine LOW; urine nitroprusside positive; homocystinuria",
        "severity_spectrum": "B6-non-responsive: severe early lens dislocation, high thrombosis risk, ID; B6-responsive: milder, later onset",
        "founder_variant": "p.Ile278Thr (B6-non-responsive, widespread); Irish: c.833T>C (1/65,000); Portuguese, Japanese founders exist",
        "b6_responsive_pct": 50,
        "methionine_high": True,
        "lens_dislocation": True,
        "thromboembolism_risk": "HIGH",
        "retinal_disease": False,
        "combined_mma": False,
        "diet_treatment": "Methionine-restricted diet + betaine + B6 (if responsive) + folate/B12; thromboprophylaxis",
        "patients": [],
        "n_patients": 40,
    },
    # ── MTHFR — MTHFR Deficiency ──────────────────────────────────────────────────────
    {
        "gene": "MTHFR", "protein": "Methylenetetrahydrofolate Reductase",
        "alias": "MTHFR — MTHFR Deficiency; severe form rare; common 677C>T usually benign (OMIM #236250)",
        "aa": "656 aa", "kDa": "74 kDa",
        "gene_class": (
            "Methylenetetrahydrofolate reductase: FAD-dependent enzyme; catalyses the irreversible "
            "reduction of 5,10-methylenetetrahydrofolate → 5-methyltetrahydrofolate (5-MTHF); "
            "5-MTHF is the methyl donor for MTR (methionine synthase) to remethylate Hcy → Met; "
            "MTHFR is the rate-limiting enzyme of folate-dependent remethylation; inhibited by SAM"
        ),
        "hcu_subgroup": "Remethylation defect — MTHFR enzyme deficiency (5-MTHF synthesis block)",
        "locus": "1p36.22", "omim_gene": 607093,
        "inheritance": "AR. 1p36.22. Both sexes equally. Severe deficiency rare (~200 patients reported). 677C>T homozygous common (10% Northern European) but usually benign with adequate folate.",
        "onset_range_y": (0.0, 5.0),
        "phenotype": (
            "Severe MTHFR deficiency: neurological crisis in infancy or childhood — apnoea, microcephaly, "
            "hypotonia, progressive encephalopathy, developmental regression; seizures common; "
            "peripheral neuropathy (demyelinating); psychiatric features (schizophreniform); "
            "NO lens dislocation; NO skeletal features; THROMBOEMBOLISM can occur (less common than CBS). "
            "MRI: leukoencephalopathy (periventricular, corpus callosum); cerebral atrophy."
        ),
        "disease": (
            "MTHFR encodes methylenetetrahydrofolate reductase (656aa, 74kDa). MTHFR catalyses the "
            "irreversible reduction of 5,10-methylene-THF → 5-methyl-THF (5-MTHF). 5-MTHF is the "
            "ONLY methyl donor for MTR (methionine synthase) to convert homocysteine → methionine. "
            "MTHFR deficiency → 5-MTHF depleted → MTR cannot remethylate Hcy → Hcy accumulates; "
            "methionine FALLS (cannot be made from Hcy) → SAM depleted → hypomethylation of DNA, "
            "myelin, neurotransmitters; cerebral folate deficiency (CSF 5-MTHF low).\n\n"
            "KEY DDx from CBS: Methionine is LOW (not high) in MTHFR deficiency — blocked at "
            "remethylation step, so Hcy rises but Met cannot be regenerated. Methionine LOW is the "
            "critical DDx from CBS (Met HIGH). No lens dislocation. No marfanoid features.\n\n"
            "TREATMENT: 5-methyltetrahydrofolate (5-MTHF/methylfolate/folinic acid) — NOT standard "
            "folic acid (which requires MTHFR to convert to 5-MTHF; MTHFR deficient patients cannot "
            "process folic acid to its active form); betaine 6-9 g/day; riboflavin (FAD cofactor "
            "for MTHFR — riboflavin 2-5 mg/day; riboflavin supplementation modestly improves "
            "MTHFR activity especially for thermolabile 677C>T variant); methionine supplementation "
            "(to correct low Met if severe deficiency); hydroxocobalamin (cofactor for MTR).\n\n"
            "COMMON 677C>T VARIANT: p.Ala222Val; very common (homozygous in 10-15% Northern Europeans); "
            "causes thermolabile MTHFR enzyme with reduced activity (~30-50% residual); homozygous "
            "677TT with folate deficiency → modestly elevated Hcy (20-30 µmol/L); usually benign but "
            "associated with neural tube defect risk. DISTINCT from rare severe MTHFR deficiency "
            "(biallelic severe variants, Hcy often >100 µmol/L, neurological disease).\n\n"
            "NBS: often MISSED — methionine is NORMAL or LOW; no standard NBS marker for MTHFR deficiency. "
            "Diagnosis: plasma amino acids (low Met) + tHcy (high) + plasma folate analysis + MTHFR enzyme "
            "assay + MTHFR sequencing. CSF: low 5-MTHF (cerebral folate deficiency)."
        ),
        "hallmark": (
            "MTHFR HALLMARKS: "
            "(1) METHIONINE LOW — remethylation block; Hcy rises, Met cannot be formed; OPPOSITE to CBS. "
            "(2) NO LENS DISLOCATION — no skeletal/marfanoid; neurological predominant. "
            "(3) NBS MISSED — methionine normal/low; no standard NBS marker. "
            "(4) METHYLFOLATE (5-MTHF) NOT FOLIC ACID — MTHFR cannot process folic acid to its active form. "
            "(5) RIBOFLAVIN (FAD COFACTOR) — improves MTHFR activity; mandatory supplement. "
            "(6) LEUKOENCEPHALOPATHY ON MRI — periventricular white matter disease; cerebral folate deficiency. "
            "(7) BETAINE BYPASSES MTHFR — BHMT pathway (liver) can remethylate Hcy independently. "
            "(8) COMMON 677C>T BENIGN vs RARE SEVERE DEFICIENCY — distinguish biallelic severe variants."
        ),
        "nbs_marker": "Often NOT detected by NBS (methionine normal/low); diagnosis usually clinical/biochemical",
        "key_biomarker": "tHcy HIGH + methionine LOW + plasma folate low + CSF 5-MTHF low; leukoencephalopathy MRI",
        "severity_spectrum": "Spectrum: benign thermolabile 677TT → severe biallelic null (encephalopathy, early death)",
        "founder_variant": "677C>T (p.Ala222Val) very common but usually benign; severe biallelic: private variants",
        "b6_responsive_pct": 0,
        "methionine_high": False,
        "lens_dislocation": False,
        "thromboembolism_risk": "MODERATE",
        "retinal_disease": False,
        "combined_mma": False,
        "diet_treatment": "Methylfolate (NOT folic acid) + betaine + riboflavin + hydroxocobalamin + methionine supplementation",
        "patients": [],
        "n_patients": 40,
    },
    # ── MTR — cblG (Methionine Synthase Deficiency) ──────────────────────────────────
    {
        "gene": "MTR", "protein": "Methionine Synthase (MS)",
        "alias": "MTR — cblG: cobalamin G complementation group; methionine synthase deficiency (OMIM #250940)",
        "aa": "1265 aa", "kDa": "140 kDa",
        "gene_class": (
            "Methionine synthase (MS): the key remethylation enzyme; methylcobalamin (MeCbl)-dependent; "
            "transfers methyl group from 5-MTHF to cobalamin cofactor → methylcobalamin; "
            "methylcobalamin then donates its methyl group to homocysteine → methionine; "
            "MTR requires periodic reactivation by MTRR (methionine synthase reductase) when "
            "the cobalamin becomes oxidised to cob(II)alamin"
        ),
        "hcu_subgroup": "Remethylation defect — methionine synthase (cblG); cobalamin-dependent",
        "locus": "1q43", "omim_gene": 156570,
        "inheritance": "AR. 1q43. Both sexes equally. Rare; ~50 patients reported (cblG complementation group).",
        "onset_range_y": (0.0, 2.0),
        "phenotype": (
            "cblG = methionine synthase deficiency: megaloblastic anemia + neurological features — "
            "developmental regression, hypotonia, encephalopathy; seizures; psychiatric features; "
            "MRI: leukoencephalopathy, cortical atrophy; NO lens dislocation; methionine LOW; "
            "tHcy HIGH. Similar to cblE (MTRR); distinguished by complementation studies/sequencing. "
            "Hydroxocobalamin i.m. is the mainstay treatment."
        ),
        "disease": (
            "MTR encodes methionine synthase (1265aa, 140kDa), the central enzyme of "
            "folate-dependent remethylation. MTR uses methylcobalamin (MeCbl) as its cobalamin "
            "cofactor — the methyl group from 5-MTHF is first transferred to cob(I)alamin "
            "→ methylcobalamin; then the methyl group from MeCbl is donated to homocysteine → "
            "methionine, regenerating cob(I)alamin.\n\n"
            "MTR can become inactivated when cobalamin is oxidised to cob(II)alamin. "
            "MTRR (methionine synthase reductase) reactivates MTR by regenerating MeCbl "
            "from cob(II)alamin using SAM as methyl donor. MTR deficiency (cblG) and MTRR "
            "deficiency (cblE) have essentially identical biochemical and clinical phenotypes "
            "— distinguished only by complementation assay or sequencing.\n\n"
            "MTR deficiency → Hcy cannot be remethylated → tHcy HIGH; Met LOW (remethylation "
            "blocked); folate 'trapped' as 5-MTHF (folate trap — cannot recycle to THF for "
            "DNA synthesis) → megaloblastic anemia; neural tube formation impaired; "
            "myelin methylation impaired → leukoencephalopathy.\n\n"
            "TREATMENT: hydroxocobalamin i.m. (NOT cyanocobalamin — the CN group must be removed "
            "and OHCbl is directly incorporated into the MeCbl pool more efficiently); "
            "methylfolate (to bypass folate trap); betaine (remethylates Hcy via BHMT); "
            "methionine supplementation if severe depletion. Monitor tHcy + methionine + CBC.\n\n"
            "NBS: often MISSED — methionine normal/low; no specific NBS marker. "
            "Diagnosis: plasma amino acids + tHcy + B12/folate + plasma MMA (normal in cblG/E "
            "unlike cblC/D) + complementation studies or MTR sequencing."
        ),
        "hallmark": (
            "MTR/cblG HALLMARKS: "
            "(1) METHIONINE LOW + tHcy HIGH — remethylation block; same as MTHFR, cblE. "
            "(2) MEGALOBLASTIC ANEMIA — folate trap; THF depleted; impaired DNA synthesis. "
            "(3) HYDROXOCOBALAMIN I.M. TREATMENT — NOT cyanocobalamin; OHCbl → MeCbl efficiently. "
            "(4) MMA NORMAL — distinguishes cblG from cblC/D where MMA is also elevated. "
            "(5) IDENTICAL PHENOTYPE TO cblE (MTRR) — complementation or sequencing required for DDx. "
            "(6) FOLATE TRAP — 5-MTHF cannot be converted to THF without MTR activity; folate supplements needed. "
            "(7) NBS MISSED — methionine low/normal; no standard marker. "
            "(8) BETAINE BYPASSES MTR — BHMT remethylation pathway is independent of B12/cobalamin."
        ),
        "nbs_marker": "Often NOT detected (methionine normal/low; MMA normal); clinical/biochemical diagnosis",
        "key_biomarker": "tHcy HIGH + methionine LOW + MMA NORMAL + megaloblastic anemia; leukoencephalopathy",
        "severity_spectrum": "Severe: neonatal encephalopathy; moderate: infantile regression; mild: adult psychiatric",
        "founder_variant": "No founder; private variants; fewer than 50 patients reported worldwide",
        "b6_responsive_pct": 0,
        "methionine_high": False,
        "lens_dislocation": False,
        "thromboembolism_risk": "MODERATE",
        "retinal_disease": False,
        "combined_mma": False,
        "diet_treatment": "Hydroxocobalamin i.m. + methylfolate + betaine + methionine supplementation; CBC monitoring",
        "patients": [],
        "n_patients": 40,
    },
    # ── MTRR — cblE (Methionine Synthase Reductase) ──────────────────────────────────
    {
        "gene": "MTRR", "protein": "Methionine Synthase Reductase (MSR)",
        "alias": "MTRR — cblE: cobalamin E complementation group; methionine synthase reductase deficiency (OMIM #236270)",
        "aa": "698 aa", "kDa": "78 kDa",
        "gene_class": (
            "Methionine synthase reductase: flavoprotein (FMN + FAD); reactivates MTR by "
            "reducing cob(II)alamin back to cob(I)alamin using SAM as methyl donor → "
            "regenerates methylcobalamin for MTR catalysis; without MTRR, MTR becomes "
            "permanently inactivated as its cobalamin is oxidised; MTRR is essential for "
            "sustained MTR activity"
        ),
        "hcu_subgroup": "Remethylation defect — methionine synthase reductase (cblE); cobalamin-dependent",
        "locus": "5p15.31", "omim_gene": 602568,
        "inheritance": "AR. 5p15.31. Both sexes equally. Rare; ~100 patients reported (cblE complementation group, more common than cblG).",
        "onset_range_y": (0.0, 2.0),
        "phenotype": (
            "cblE = MTRR deficiency: CLINICALLY AND BIOCHEMICALLY IDENTICAL TO cblG (MTR deficiency). "
            "Megaloblastic anemia + neurological features (encephalopathy, hypotonia, seizures, "
            "regression, psychiatric); MRI leukoencephalopathy; methionine LOW; tHcy HIGH; "
            "MMA normal. Distinguished from cblG only by complementation studies or gene sequencing. "
            "Hydroxocobalamin i.m. + betaine + methylfolate."
        ),
        "disease": (
            "MTRR encodes methionine synthase reductase (698aa, 78kDa), a diflavin oxidoreductase "
            "(FMN + FAD domains). MTRR reactivates MTR by reducing oxidised cob(II)alamin back "
            "to the active cob(I)alamin form, with SAM providing the methyl group for regenerating "
            "methylcobalamin. Without MTRR, MTR becomes irreversibly inactivated as cobalamin "
            "oxidises over time.\n\n"
            "MTRR deficiency (cblE) → MTR cannot be reactivated → effective MTR deficiency → "
            "same downstream biochemistry as MTR/cblG deficiency: tHcy HIGH, methionine LOW, "
            "folate trap, megaloblastic anemia, leukoencephalopathy, neurological disease.\n\n"
            "MTRR p.Ile22Met (c.66A>G): very common MTRR variant (20-40% allele frequency in "
            "Europeans); modest reduction in MTRR activity; NOT a disease-causing variant in isolation "
            "but a modifying polymorphism for folate-related phenotypes.\n\n"
            "TREATMENT: identical to cblG — hydroxocobalamin i.m. (OHCbl → methylcobalamin, "
            "partially bypasses need for MTRR reactivation by providing fresh MeCbl supply); "
            "methylfolate (to bypass folate trap); betaine 6-9 g/day (BHMT remethylation).\n\n"
            "DISTINGUISHING cblE vs cblG: Cell complementation assay (cblE cells rescue cblG cells "
            "and vice versa — different genes). Modern approach: sequence both MTRR and MTR. "
            "MMA is normal in both (unlike cblC/D). Both respond to hydroxocobalamin."
        ),
        "hallmark": (
            "MTRR/cblE HALLMARKS: "
            "(1) IDENTICAL TO cblG BIOCHEMICALLY — MTR vs MTRR deficiency; same downstream effect. "
            "(2) MMA NORMAL — distinguishes cblE from cblC (MMA+HC combined). "
            "(3) HYDROXOCOBALAMIN BYPASSES MTRR — fresh MeCbl supply partially circumvents need for MTRR reactivation. "
            "(4) MEGALOBLASTIC ANEMIA + NEUROLOGICAL — folate trap + hypomethylation. "
            "(5) p.Ile22Met (c.66A>G) COMMON VARIANT — modifier only, NOT disease-causing alone; high allele freq. "
            "(6) METHIONINE LOW — remethylation blocked; betaine (BHMT) bypasses. "
            "(7) NBS MISSED — methionine normal/low; no standard marker. "
            "(8) COMPLEMENTATION ASSAY OR SEQUENCING TO DISTINGUISH cblE vs cblG."
        ),
        "nbs_marker": "Often NOT detected (methionine normal/low; MMA normal); clinical/biochemical diagnosis",
        "key_biomarker": "tHcy HIGH + methionine LOW + MMA NORMAL + megaloblastic anemia; OHCbl-responsive",
        "severity_spectrum": "Severe neonatal to mild adult; typically presents in infancy with encephalopathy",
        "founder_variant": "p.Ile22Met common modifier (20-40% Europeans); severe biallelic: private pathogenic variants",
        "b6_responsive_pct": 0,
        "methionine_high": False,
        "lens_dislocation": False,
        "thromboembolism_risk": "MODERATE",
        "retinal_disease": False,
        "combined_mma": False,
        "diet_treatment": "Hydroxocobalamin i.m. + methylfolate + betaine + methionine supplementation",
        "patients": [],
        "n_patients": 40,
    },
    # ── MMACHC — cblC (Most Common Combined MMA+Homocystinuria) ──────────────────────
    {
        "gene": "MMACHC", "protein": "cblC protein (MMACHC — methylmalonyl-CoA/homocysteine metabolism)",
        "alias": "MMACHC — cblC: most common combined methylmalonic acidemia + homocystinuria; retinal disease PATHOGNOMONIC (OMIM #277400)",
        "aa": "282 aa", "kDa": "31 kDa",
        "gene_class": (
            "MMACHC protein: cytoplasmic flavoprotein; catalyses decyanation of cyanocobalamin "
            "(removes CN from CNCbl → forms cob(II)alamin) AND dealkylation of MeCbl/AdoCbl; "
            "processes all forms of cobalamin for downstream synthesis of AdoCbl (for MUT) and "
            "MeCbl (for MTR); first step in intracellular cobalamin metabolism shared by BOTH "
            "the MUT pathway (mitochondrial AdoCbl) and the MTR pathway (cytoplasmic MeCbl)"
        ),
        "hcu_subgroup": "Combined MMA + Homocystinuria (cblC) — defective intracellular cobalamin processing",
        "locus": "1p34.1", "omim_gene": 609831,
        "inheritance": "AR. 1p34.1. Both sexes equally. Most common inborn error of cobalamin metabolism (~1/100,000 live births); most common combined MMA+HC disorder.",
        "onset_range_y": (0.0, 1.0),
        "phenotype": (
            "cblC early-onset (most common): neonatal/infantile presentation — feeding difficulties, "
            "lethargy, hypotonia, encephalopathy, failure to thrive; megaloblastic anemia; "
            "RETINAL DISEASE — macular degeneration + nystagmus (PATHOGNOMONIC — no other cobalamin "
            "disorder has major retinopathy); hydrocephalus possible; pulmonary hypertension in severe cases. "
            "BIOCHEMISTRY: both MMA AND homocysteine elevated (combined MMA+HC — the diagnostic signature). "
            "Late-onset cblC (rare): adult-onset dementia + myelopathy + psychiatric features."
        ),
        "disease": (
            "MMACHC encodes the cblC protein (282aa, 31kDa), a cytoplasmic flavoprotein that is the "
            "FIRST step in intracellular cobalamin metabolism, required for BOTH AdoCbl and MeCbl synthesis. "
            "MMACHC processes ingested cobalamin forms — removing CN from CNCbl (decyanation) or "
            "alkyl groups from MeCbl/AdoCbl (dealkylation) — generating free cob(I/II)alamin for "
            "downstream pathway enzymes:\n"
            "  → AdoCbl synthesis: needed by methylmalonyl-CoA mutase (MUT) → propionate disposal\n"
            "  → MeCbl synthesis: needed by methionine synthase (MTR) → Hcy remethylation\n\n"
            "MMACHC deficiency → BOTH pathways fail simultaneously → combined MMA+HC:\n"
            "  MMA elevated: propionyl-CoA cannot be fully disposed → propionate accumulates\n"
            "  Hcy elevated: MTR cannot function → remethylation blocked → met LOW\n\n"
            "RETINAL DISEASE (PATHOGNOMONIC): macular degeneration (pigmentary retinopathy) + "
            "nystagmus present in >50% of cblC patients by 1-2 years; NO other cobalamin disorder "
            "causes significant retinopathy; mechanism: AdoCbl deficiency in retinal pigment epithelium + "
            "hyperhomocysteinaemia → retinal toxicity.\n\n"
            "TREATMENT: hydroxocobalamin i.m. (NOT cyanocobalamin — MMACHC decyanates CNCbl, so without "
            "MMACHC the CN cannot be removed; OHCbl is the preferred form); betaine 6-9 g/day "
            "(BHMT remethylation, reduces Hcy independently); carnitine supplementation "
            "(propionylcarnitine accumulates, depletes free carnitine); methylfolate (folate trap).\n\n"
            "NBS: ELEVATED C3-CARNITINE (propionylcarnitine) on MS/MS — the best NBS marker for cblC; "
            "also: elevated MMA on urine organic acids at confirmation. Methionine normal or low. "
            "Plasma amino acid + urine organic acid + plasma acylcarnitine + tHcy = full work-up.\n\n"
            "MOST COMMON VARIANT: c.271dupA (p.Arg91Lysfs*14) — most common pathogenic allele; "
            "truncating; associated with early-onset severe phenotype; very high frequency in cblC patients."
        ),
        "hallmark": (
            "MMACHC/cblC HALLMARKS: "
            "(1) COMBINED MMA + HOMOCYSTINURIA — BOTH elevated; no other cobalamin disorder has this exact combination plus retinal disease. "
            "(2) RETINAL DISEASE PATHOGNOMONIC — macular degeneration + nystagmus; ONLY cobalamin disorder with major retinopathy. "
            "(3) C3-CARNITINE ELEVATED ON NBS — propionylcarnitine; best NBS marker; DETECTED by standard NBS. "
            "(4) HYDROXOCOBALAMIN NOT CYANOCOBALAMIN — MMACHC decyanates CNCbl; without MMACHC, CNCbl cannot be processed. "
            "(5) BETAINE MANDATORYALONGSIDE OHCbl — BHMT reduces Hcy independently of B12. "
            "(6) METHIONINE LOW — remethylation blocked. "
            "(7) c.271dupA MOST COMMON VARIANT — truncating; early-onset severe phenotype. "
            "(8) PULMONARY HYPERTENSION POSSIBLE — severe cases; monitor cardiology."
        ),
        "nbs_marker": "Elevated C3-carnitine (propionylcarnitine) on MS/MS — DETECTED by standard NBS",
        "key_biomarker": "MMA elevated + tHcy elevated + C3 elevated + retinal disease + methionine LOW; combined MMA+HC signature",
        "severity_spectrum": "Early-onset (most common): neonatal encephalopathy + retinopathy; late-onset: adult dementia/myelopathy",
        "founder_variant": "c.271dupA (p.Arg91Lysfs*14) most common worldwide; c.394C>T (p.Arg132*) second most common",
        "b6_responsive_pct": 0,
        "methionine_high": False,
        "lens_dislocation": False,
        "thromboembolism_risk": "MODERATE",
        "retinal_disease": True,
        "combined_mma": True,
        "diet_treatment": "Hydroxocobalamin i.m. + betaine + carnitine + methylfolate + protein restriction (modest); ophthalmology surveillance",
        "patients": [],
        "n_patients": 40,
    },
    # ── MMADHC — cblD (Variable MMA and/or Homocystinuria) ───────────────────────────
    {
        "gene": "MMADHC", "protein": "MMADHC (cblD protein — methylmalonic aciduria and homocystinuria type D)",
        "alias": "MMADHC — cblD: variable phenotype — MMA only / HC only / combined MMA+HC (OMIM #277410)",
        "aa": "296 aa", "kDa": "34 kDa",
        "gene_class": (
            "MMADHC protein: cytoplasmic protein involved in intracellular cobalamin trafficking; "
            "functions DOWNSTREAM of MMACHC; routes reduced cobalamin to either the cytoplasm "
            "(→ MeCbl for MTR) or mitochondria (→ AdoCbl for MUT); has a C-terminal cobalamin-binding "
            "domain and N-terminal mitochondrial targeting sequence; variant location determines "
            "which pathway is preferentially disrupted"
        ),
        "hcu_subgroup": "Combined or isolated MMA/HC (cblD) — variable intracellular cobalamin routing defect",
        "locus": "2q23.2", "omim_gene": 611935,
        "inheritance": "AR. 2q23.2. Both sexes equally. Rare; fewer than 50 patients reported. Three distinct biochemical subtypes based on variant location.",
        "onset_range_y": (0.0, 5.0),
        "phenotype": (
            "cblD — THREE PHENOTYPIC SUBTYPES based on variant location in MMADHC protein:\n"
            "(1) cblD-MMA only: isolated methylmalonic acidemia without homocystinuria — "
            "variants in N-terminal region affect mitochondrial AdoCbl only;\n"
            "(2) cblD-HC only: isolated homocystinuria without MMA — variants in C-terminal "
            "cobalamin-binding domain affect cytoplasmic MeCbl only;\n"
            "(3) cblD-combined: both MMA and HC elevated — null alleles or midprotein variants "
            "affecting both pathways. Phenotype: developmental delay, encephalopathy, anemia, "
            "variable onset; no retinal disease (distinguishes from cblC)."
        ),
        "disease": (
            "MMADHC encodes a 296aa protein that functions downstream of MMACHC to route "
            "free cob(I/II)alamin to the correct intracellular compartment:\n"
            "  → C-terminal domain: routes cob to cytoplasm → MeCbl synthesis → MTR activity → Hcy remethylation\n"
            "  → N-terminal mitochondrial targeting sequence: routes cob to mitochondria → AdoCbl synthesis → MUT activity → propionate disposal\n\n"
            "GENOTYPE-PHENOTYPE CORRELATION (UNIQUE TO cblD):\n"
            "  Variants in N-terminal half → preferential disruption of mitochondrial routing → AdoCbl deficiency → MMA isolated (no HC)\n"
            "  Variants in C-terminal cobalamin-binding domain → preferential disruption of cytoplasmic routing → MeCbl deficiency → HC isolated (no MMA)\n"
            "  Null alleles / early truncations → both pathways → combined MMA+HC (like cblC)\n\n"
            "KEY DDx from cblC: cblD has NO retinal disease (retinopathy absent); cblC has "
            "pathognomonic retinal disease. Combined MMA+HC without retinal disease = cblD "
            "complementation group or later-onset cblC. Cell complementation assay or sequencing "
            "required for definitive diagnosis.\n\n"
            "TREATMENT: hydroxocobalamin i.m.; betaine; carnitine; methylfolate; treatment "
            "tailored to which pathway is affected (MMA-only → carnitine + protein restriction; "
            "HC-only → betaine + OHCbl; combined → full regimen).\n\n"
            "NBS: C3 elevated if MMA component present; methionine normal/low if HC component present. "
            "HC-only cblD may be MISSED by NBS (no C3 elevation, methionine not elevated)."
        ),
        "hallmark": (
            "MMADHC/cblD HALLMARKS: "
            "(1) THREE PHENOTYPIC SUBTYPES — MMA-only / HC-only / combined; determined by variant location in protein. "
            "(2) NO RETINAL DISEASE — key DDx from cblC (MMACHC); combined MMA+HC WITHOUT retinopathy = cblD. "
            "(3) GENOTYPE-PHENOTYPE CORRELATION — N-terminal variants → MMA-only; C-terminal → HC-only; null → combined. "
            "(4) HC-ONLY cblD MISSED BY NBS — no C3 elevation if no MMA component. "
            "(5) DOWNSTREAM OF MMACHC — cblD routes cobalamin after MMACHC decyanation step. "
            "(6) HYDROXOCOBALAMIN + BETAINE — treatment approach matches phenotypic subtype. "
            "(7) COMPLEMENTATION ASSAY DISTINGUISHES cblD vs cblC vs cblE/cblG. "
            "(8) RARER THAN cblC — fewer than 50 patients reported; highly heterogeneous."
        ),
        "nbs_marker": "Variable: C3 elevated if MMA present; HC-only subtype often NOT detected by NBS",
        "key_biomarker": "MMA +/- tHcy elevated (depends on subtype); no retinal disease (distinguishes from cblC); MMADHC sequencing",
        "severity_spectrum": "Variable; MMA-only often milder; combined or HC-only can be severe neurological",
        "founder_variant": "No founder; private variants; genotype-phenotype correlation by variant location",
        "b6_responsive_pct": 0,
        "methionine_high": False,
        "lens_dislocation": False,
        "thromboembolism_risk": "LOW-MODERATE",
        "retinal_disease": False,
        "combined_mma": True,
        "diet_treatment": "Hydroxocobalamin + betaine + carnitine tailored to phenotypic subtype (MMA-only vs HC-only vs combined)",
        "patients": [],
        "n_patients": 40,
    },
    # ── AHCY — Adenosylhomocysteinase Deficiency ─────────────────────────────────────
    {
        "gene": "AHCY", "protein": "Adenosylhomocysteine Hydrolase (SAHH)",
        "alias": "AHCY — Adenosylhomocysteinase deficiency; hypermethioninemia + hyperhomocysteinemia + myopathy (OMIM #613752)",
        "aa": "432 aa", "kDa": "47 kDa",
        "gene_class": (
            "Adenosylhomocysteine hydrolase (SAHH): bidirectional enzyme that hydrolyses "
            "S-adenosylhomocysteine (SAH) → adenosine + homocysteine; NAD-dependent; "
            "AHCY is the ONLY enzyme that can remove SAH; SAH is a potent product inhibitor "
            "of ALL SAM-dependent methyltransferases; AHCY deficiency → SAH accumulates → "
            "global methylation inhibition (DNA, RNA, protein, lipid methylation)"
        ),
        "hcu_subgroup": "SAH hydrolase deficiency — global methylation inhibition; unique mechanism among HCU types",
        "locus": "20q11.22", "omim_gene": 180960,
        "inheritance": "AR. 20q11.22. Both sexes equally. Extremely rare — fewer than 30 patients reported worldwide (one of the rarest metabolic diseases).",
        "onset_range_y": (0.0, 2.0),
        "phenotype": (
            "AHCY deficiency: hypermethioninemia (elevated methionine — unique among remethylation "
            "disorders; AHCY deficiency causes Met HIGH because SAH accumulation inhibits methionine "
            "adenosyltransferase reverse flux) + elevated SAH + elevated tHcy; "
            "MYOPATHY (muscle creatine kinase HIGH — distinctive feature); hypotonia; "
            "CARDIOMYOPATHY; developmental delay; liver disease (elevated transaminases); "
            "sometimes hepatomegaly. Unlike CBS HCU, thromboembolism rare. "
            "MRI: leukoencephalopathy possible. Diagnosis often delayed (rare; unusual biochemistry)."
        ),
        "disease": (
            "AHCY encodes adenosylhomocysteine hydrolase (432aa, 47kDa), a tetrameric NAD-dependent "
            "enzyme. AHCY is the ONLY enzyme that removes SAH (S-adenosylhomocysteine) from cells. "
            "SAH is generated in every SAM (S-adenosylmethionine)-dependent methylation reaction:\n"
            "  SAM + substrate → SAH + methylated-substrate\n\n"
            "SAH is a potent PRODUCT INHIBITOR of essentially all SAM-dependent methyltransferases "
            "(DNMT, HNMT, PNMT, COMT, histone methyltransferases, etc.). AHCY deficiency → "
            "SAH cannot be cleared → SAH accumulates → GLOBAL HYPOMETHYLATION (DNA, histones, "
            "catecholamines, creatine, myelin, phospholipids, RNA cap-methylation).\n\n"
            "UNIQUE BIOCHEMISTRY DISTINGUISHING AHCY FROM OTHER HCU TYPES:\n"
            "  Methionine HIGH (not low) — SAH accumulation inhibits MAT (methionine → SAM conversion) "
            "  → Met cannot be used → met backed up; also SAM/SAH ratio falls → feedback on remethylation\n"
            "  SAH dramatically elevated (key biomarker)\n"
            "  SAM/SAH ratio severely depressed\n"
            "  Homocysteine elevated (AHCY reaction bidirectional; SAH equilibrium favours SAH accumulation)\n\n"
            "CK ELEVATION (MYOPATHY): creatine requires methylation (GAMT → creatine); "
            "methylation impaired → creatine synthesis disrupted + muscle SAH toxicity → myopathy.\n\n"
            "TREATMENT: largely supportive; methionine-restricted diet (reduce SAM generation → "
            "reduce SAH production); no specific vitamin responsive. SAH-lowering strategies "
            "experimental (adenosine deaminase inhibitors, nucleoside transport inhibitors). "
            "Betaine may worsen (reduces Hcy but cannot address SAH accumulation).\n\n"
            "NBS: elevated methionine on MS/MS (UNIQUE: Met HIGH, not low — important diagnostic clue). "
            "CBS also has Met HIGH — distinguished by: CBS has lens dislocation + thromboembolism, "
            "AHCY has myopathy + CK + SAH elevated. CBS: cystathionine absent; AHCY: cystathionine normal."
        ),
        "hallmark": (
            "AHCY HALLMARKS: "
            "(1) SAH ACCUMULATION — global methylation inhibitor; SAH is the ONLY substrate for AHCY; AHCY = ONLY enzyme for SAH removal. "
            "(2) METHIONINE HIGH (unique among remethylation disorders) — met backs up; SAH inhibits MAT; similar to CBS but by different mechanism. "
            "(3) MYOPATHY + CK ELEVATED — creatine methylation impaired; muscle SAH toxicity; key DDx from CBS. "
            "(4) CARDIOMYOPATHY — myocardial methylation impaired; cardiac surveillance mandatory. "
            "(5) NBS ELEVATED METHIONINE — detected; but must distinguish from CBS (CK normal in CBS, CK HIGH in AHCY). "
            "(6) BETAINE MAY WORSEN — reduces Hcy but cannot address SAH; use with caution. "
            "(7) GLOBAL HYPOMETHYLATION — DNA, histone, RNA cap, catecholamines, creatine all affected. "
            "(8) EXTREMELY RARE — fewer than 30 patients worldwide; underdiagnosed."
        ),
        "nbs_marker": "Elevated methionine on MS/MS — DETECTED; must distinguish from CBS (similar NBS marker)",
        "key_biomarker": "SAH dramatically elevated + methionine HIGH + CK elevated + homocysteine elevated; SAM/SAH ratio depressed",
        "severity_spectrum": "Severe: early hepatopathy + cardiomyopathy; variable; often progressive without treatment",
        "founder_variant": "No founder; private variants; fewer than 30 patients worldwide",
        "b6_responsive_pct": 0,
        "methionine_high": True,
        "lens_dislocation": False,
        "thromboembolism_risk": "LOW",
        "retinal_disease": False,
        "combined_mma": False,
        "diet_treatment": "Methionine-restricted diet (reduce SAH generation); no specific vitamin responsive; SAH-lowering experimental",
        "patients": [],
        "n_patients": 40,
    },
    # ── MAT1A — Methionine Adenosyltransferase I/III Deficiency ───────────────────────
    {
        "gene": "MAT1A", "protein": "Methionine Adenosyltransferase Alpha-1 (MAT I/III)",
        "alias": "MAT1A — MAT I/III deficiency: hepatic isolated hypermethioninemia; usually benign; demyelination in severe cases (OMIM #250850)",
        "aa": "395 aa", "kDa": "44 kDa",
        "gene_class": (
            "Methionine adenosyltransferase alpha-1: catalytic subunit of hepatic MAT I "
            "(homotetramer) and MAT III (homodimer); synthesises SAM (S-adenosylmethionine) "
            "from methionine + ATP; MAT1A is expressed exclusively in the liver (adult); "
            "liver-specific SAM synthesis; MAT2A isozyme (ubiquitous) is not affected; "
            "hence MAT1A deficiency causes hepatic SAM deficiency with compensatory methionine rise"
        ),
        "hcu_subgroup": "Hepatic methionine metabolism — MAT I/III deficiency; isolated hypermethioninemia; usually benign",
        "locus": "10q22.3", "omim_gene": 610550,
        "inheritance": "AR (severe) or AD (dominant-negative); 10q22.3. Incidence: 1/27,000 (isolated hypermethioninemia). Usually benign; severe AR form rare.",
        "onset_range_y": (0.0, 1.0),
        "phenotype": (
            "MAT1A deficiency: isolated hypermethioninemia (methionine elevated, homocysteine NORMAL "
            "or only mildly elevated); usually ASYMPTOMATIC or very mild in most patients (detected "
            "by NBS). SEVERE AR form: demyelinating neurological disease — white matter disease "
            "(leukoencephalopathy on MRI), cognitive impairment, abnormal brain MRI. "
            "Homocysteine usually NOT elevated (remethylation intact; only transsulfuration "
            "reduced from lack of hepatic SAM substrate). Liver usually normal. "
            "Breath odour of methionine (dimethyl sulfide — 'boiled cabbage' smell)."
        ),
        "disease": (
            "MAT1A encodes the alpha-1 subunit of hepatic methionine adenosyltransferase (395aa, 44kDa). "
            "MAT1A forms MAT I (homotetramer, high-Km) and MAT III (homodimer, low-Km) — "
            "LIVER-SPECIFIC isoforms that produce >85% of total body SAM. "
            "MAT2A (ubiquitous isozyme) is unaffected in MAT1A deficiency.\n\n"
            "MAT1A deficiency → hepatic SAM synthesis impaired → methionine cannot be converted "
            "to SAM efficiently → methionine accumulates (isolated hypermethioninemia) → "
            "SAM depleted in liver → hepatic methylation reactions impaired.\n\n"
            "UNIQUE FEATURES:\n"
            "  Homocysteine NORMAL (or mildly elevated) — remethylation (MTR/MTHFR) intact; "
            "  CBS transsulfuration intact; only SAM synthesis step impaired in liver\n"
            "  MAT2A compensates systemically — extrahepatic tissues have normal SAM\n"
            "  Liver histology usually normal (unlike AHCY which causes hepatopathy)\n"
            "  Most patients asymptomatic — MAT1A is NOT required for most metabolic functions\n\n"
            "NEUROLOGICAL FORM (SEVERE AR): SAM depleted in liver → systemic SAM transport "
            "reduced → brain methylation (myelin maintenance, catecholamines) impaired in severe cases "
            "→ leukoencephalopathy. Brain MRI: symmetric white matter changes. \n\n"
            "DOMINANT-NEGATIVE FORM (AD inheritance): some heterozygotes with dominant-negative "
            "variants show isolated hypermethioninemia with milder phenotype.\n\n"
            "TREATMENT: Methionine-restricted diet (reduce substrate) + SAM supplementation "
            "(exogenous SAM crosses blood-brain barrier) for neurological forms. Most patients "
            "require no treatment. Betaine NOT the mainstay (remethylation intact). "
            "Monitoring: brain MRI annually in confirmed cases; methionine levels.\n\n"
            "NBS: elevated methionine — DETECTED by MS/MS. Must distinguish from CBS "
            "(CBS: tHcy HIGH, lens dislocation; MAT1A: tHcy NORMAL, no lens dislocation) "
            "and AHCY (AHCY: SAH HIGH, CK HIGH, myopathy; MAT1A: SAH normal, CK normal).\n\n"
            "INCIDENCE: ~1/27,000 newborns identified by NBS with isolated hypermethioninemia "
            "(all causes combined); MAT1A is the most common; most have benign course."
        ),
        "hallmark": (
            "MAT1A HALLMARKS: "
            "(1) ISOLATED HYPERMETHIONINEMIA — methionine HIGH, homocysteine NORMAL; SAM synthesis block only. "
            "(2) LIVER-SPECIFIC ENZYME — MAT1A hepatic only; MAT2A (ubiquitous) unaffected; extrahepatic SAM normal. "
            "(3) USUALLY BENIGN — most patients asymptomatic; NBS-detected; reassure family. "
            "(4) DEMYELINATION IN SEVERE AR — white matter disease; SAM supplement crosses BBB. "
            "(5) NBS ELEVATED METHIONINE — DETECTED; must DDx CBS (tHcy high + CBS has lens dislocation), AHCY (SAH high + CK high). "
            "(6) BREATH ODOUR — dimethyl sulfide ('boiled cabbage') from methionine metabolism. "
            "(7) AD DOMINANT-NEGATIVE FORM — heterozygous dominant-negative variants; milder hypermethioninemia. "
            "(8) BETAINE NOT MAINSTAY — remethylation intact; SAM supplementation for neurological forms."
        ),
        "nbs_marker": "Elevated methionine on MS/MS — DETECTED; most common cause of isolated hypermethioninemia on NBS",
        "key_biomarker": "Methionine HIGH + homocysteine NORMAL/mildly elevated + SAH normal + CK normal; MAT1A sequencing",
        "severity_spectrum": "Usually benign (NBS-detected); severe AR: leukoencephalopathy; AD: mild hypermethioninemia",
        "founder_variant": "p.Arg264His (c.791G>A) — common pathogenic variant; various founder alleles in different populations",
        "b6_responsive_pct": 0,
        "methionine_high": True,
        "lens_dislocation": False,
        "thromboembolism_risk": "VERY LOW",
        "retinal_disease": False,
        "combined_mma": False,
        "diet_treatment": "Methionine-restricted diet (if symptomatic or severe) + SAM supplementation for neurological; most need no treatment",
        "patients": [],
        "n_patients": 40,
    },
]


def _make_patients(gene_idx: int):
    """Generate realistic 40-patient cohort for a given gene."""
    g = HCU_GENES[gene_idx]
    gene = g["gene"]
    seed = SEED_BASE + gene_idx
    rng = random.Random(seed)

    # Age at diagnosis distribution per gene
    if gene == "CBS":
        age_dx = [rng.uniform(0.5, 15.0) for _ in range(40)]   # often later; childhood-adolescent
    elif gene == "MTHFR":
        age_dx = [rng.uniform(0.0, 3.0) for _ in range(40)]    # infantile-toddler encephalopathy
    elif gene == "MTR":
        age_dx = [rng.uniform(0.0, 1.5) for _ in range(40)]    # infantile
    elif gene == "MTRR":
        age_dx = [rng.uniform(0.0, 1.5) for _ in range(40)]    # infantile
    elif gene == "MMACHC":
        age_dx = [rng.uniform(0.0, 0.5) for _ in range(40)]    # neonatal-early infantile (NBS)
    elif gene == "MMADHC":
        age_dx = [rng.uniform(0.0, 5.0) for _ in range(40)]    # variable
    elif gene == "AHCY":
        age_dx = [rng.uniform(0.0, 2.0) for _ in range(40)]    # infantile
    else:  # MAT1A
        age_dx = [rng.uniform(0.0, 0.1) for _ in range(40)]    # NBS-detected (neonatal)

    patients = []
    for i in range(40):
        pt_id = f"{gene}-{i+1:03d}"

        # Plasma total homocysteine at diagnosis (µmol/L)
        if gene == "CBS":
            thcy = rng.randint(50, 350)
        elif gene == "MTHFR":
            thcy = rng.randint(40, 200)
        elif gene == "MTR":
            thcy = rng.randint(30, 150)
        elif gene == "MTRR":
            thcy = rng.randint(30, 150)
        elif gene == "MMACHC":
            thcy = rng.randint(50, 250)
        elif gene == "MMADHC":
            thcy = rng.randint(20, 180)
        elif gene == "AHCY":
            thcy = rng.randint(30, 120)
        else:  # MAT1A
            thcy = rng.randint(5, 30)   # often NORMAL/mildly elevated

        # Plasma methionine (µmol/L)
        if g["methionine_high"]:
            methionine = rng.randint(100, 800)  # CBS, AHCY, MAT1A
        else:
            methionine = rng.randint(5, 35)     # MTHFR, MTR, MTRR, MMACHC, MMADHC — LOW

        # Plasma MMA (µmol/L — only elevated in cblC and some cblD)
        if gene == "MMACHC":
            mma_umol = rng.randint(200, 2000)   # elevated
        elif gene == "MMADHC":
            mma_umol = rng.randint(0, 1500)     # variable (MMA-only subtype can be high)
        else:
            mma_umol = rng.randint(0, 20)       # normal

        # B6-responsive (CBS only, ~50%)
        b6_responsive = False
        if gene == "CBS":
            b6_responsive = rng.random() < 0.50

        # Lens dislocation (CBS only)
        lens_disloc = False
        if gene == "CBS":
            lens_disloc = (age_dx[i] > 3.0) or (rng.random() < 0.70)

        # Thromboembolism (CBS mainly)
        thromboembolism = False
        if gene == "CBS":
            thromboembolism = (age_dx[i] > 5.0) and (rng.random() < 0.45)
        elif gene in ("MTHFR", "MTR", "MTRR", "MMACHC"):
            thromboembolism = rng.random() < 0.10

        # Retinal disease (MMACHC only)
        retinal_disease = False
        if gene == "MMACHC":
            retinal_disease = rng.random() < 0.60

        # Megaloblastic anemia (remethylation disorders)
        megaloblastic_anemia = False
        if gene in ("MTR", "MTRR", "MMACHC"):
            megaloblastic_anemia = rng.random() < 0.70
        elif gene == "MMADHC":
            megaloblastic_anemia = rng.random() < 0.30

        # Myopathy / CK elevated (AHCY mainly)
        myopathy = False
        ck_elevated = False
        if gene == "AHCY":
            myopathy = rng.random() < 0.75
            ck_elevated = rng.random() < 0.80

        # MRI leukoencephalopathy
        mri_leuko = False
        if gene in ("MTHFR", "MTR", "MTRR", "MMACHC", "AHCY"):
            mri_leuko = rng.random() < 0.65
        elif gene == "MAT1A":
            mri_leuko = rng.random() < 0.20   # mild/late

        # Encephalopathy at presentation
        encephalopathy = False
        if gene in ("MTHFR", "MTR", "MTRR", "MMACHC", "AHCY"):
            encephalopathy = rng.random() < 0.60

        # Seizures
        seizures = False
        if gene in ("MTHFR", "MTR", "MTRR", "MMACHC"):
            seizures = rng.random() < 0.40
        elif gene == "CBS":
            seizures = rng.random() < 0.20

        # NBS detected
        nbs_detected = False
        if gene == "CBS":
            nbs_detected = rng.random() < 0.70   # ~70% detected (Met elevation)
        elif gene == "MMACHC":
            nbs_detected = rng.random() < 0.90   # ~90% (C3 elevated)
        elif gene == "MAT1A":
            nbs_detected = rng.random() < 0.95   # ~95% (Met clearly elevated)
        elif gene == "AHCY":
            nbs_detected = rng.random() < 0.75   # Met elevated
        elif gene in ("MTHFR", "MTR", "MTRR"):
            nbs_detected = rng.random() < 0.20   # often MISSED
        elif gene == "MMADHC":
            nbs_detected = rng.random() < 0.50   # C3 only if MMA component

        patients.append({
            "id": pt_id,
            "gene": gene,
            "age_dx_y": round(age_dx[i], 3),
            "sex": rng.choice(["M", "F"]),
            "thcy_umolL": thcy,
            "methionine_umolL": methionine,
            "mma_umolL": mma_umol,
            "methionine_high": g["methionine_high"],
            "b6_responsive": b6_responsive,
            "lens_dislocation": lens_disloc,
            "thromboembolism": thromboembolism,
            "retinal_disease": retinal_disease,
            "megaloblastic_anemia": megaloblastic_anemia,
            "myopathy": myopathy,
            "ck_elevated": ck_elevated,
            "mri_leukoencephalopathy": mri_leuko,
            "encephalopathy_at_dx": encephalopathy,
            "seizures": seizures,
            "nbs_detected": nbs_detected,
        })
    return patients


# Populate all gene patient cohorts
ALL_PATIENTS = []
for _idx in range(len(HCU_GENES)):
    _pts = _make_patients(_idx)
    HCU_GENES[_idx]["patients"] = _pts
    ALL_PATIENTS.extend(_pts)


# ─── API: get_overview ─────────────────────────────────────────────────────────────
def get_overview():
    """Return high-level Homocystinuria-Atlas summary."""
    total = len(ALL_PATIENTS)

    gene_summary = []
    for g in HCU_GENES:
        pts = g["patients"]
        gene_summary.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "gene_class": g["gene_class"],
            "hcu_subgroup": g["hcu_subgroup"],
            "n_patients": g["n_patients"],
            "inheritance": g["inheritance"],
            "b6_responsive_pct": g["b6_responsive_pct"],
            "methionine_high": g["methionine_high"],
            "lens_dislocation": g["lens_dislocation"],
            "thromboembolism_risk": g["thromboembolism_risk"],
            "retinal_disease": g["retinal_disease"],
            "combined_mma": g["combined_mma"],
            "phenotype": g["phenotype"],
            "diet_treatment": g["diet_treatment"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 2),
            "mean_thcy_umolL": round(sum(p["thcy_umolL"] for p in pts) / len(pts), 1),
        })

    n_lens = sum(1 for p in ALL_PATIENTS if p.get("lens_dislocation"))
    n_thrombosis = sum(1 for p in ALL_PATIENTS if p.get("thromboembolism"))
    n_retinal = sum(1 for p in ALL_PATIENTS if p.get("retinal_disease"))
    n_megaloblastic = sum(1 for p in ALL_PATIENTS if p.get("megaloblastic_anemia"))
    n_b6_resp = sum(1 for p in ALL_PATIENTS if p.get("b6_responsive"))
    n_nbs = sum(1 for p in ALL_PATIENTS if p.get("nbs_detected"))
    n_mma = sum(1 for p in ALL_PATIENTS if p.get("mma_umolL", 0) > 50)
    n_leuko = sum(1 for p in ALL_PATIENTS if p.get("mri_leukoencephalopathy"))
    n_enceph = sum(1 for p in ALL_PATIENTS if p.get("encephalopathy_at_dx"))
    n_seizures = sum(1 for p in ALL_PATIENTS if p.get("seizures"))
    n_myopathy = sum(1 for p in ALL_PATIENTS if p.get("myopathy"))

    return {
        "atlas": "Homocystinuria-Atlas",
        "title": "Complete 8-Gene Homocystinuria & Remethylation Disorders Atlas",
        "n_genes": len(HCU_GENES),
        "n_patients": total,
        "seeds": f"{SEED_BASE}–{SEED_BASE + len(HCU_GENES) - 1}",
        "genes": gene_summary,
        "cohort_stats": {
            "n_lens_dislocation": n_lens,
            "pct_lens_dislocation": round(100 * n_lens / total, 1),
            "n_thromboembolism": n_thrombosis,
            "pct_thromboembolism": round(100 * n_thrombosis / total, 1),
            "n_retinal_disease": n_retinal,
            "pct_retinal_disease": round(100 * n_retinal / total, 1),
            "n_megaloblastic_anemia": n_megaloblastic,
            "pct_megaloblastic_anemia": round(100 * n_megaloblastic / total, 1),
            "n_b6_responsive": n_b6_resp,
            "pct_b6_responsive": round(100 * n_b6_resp / total, 1),
            "n_nbs_detected": n_nbs,
            "pct_nbs_detected": round(100 * n_nbs / total, 1),
            "n_elevated_mma": n_mma,
            "pct_elevated_mma": round(100 * n_mma / total, 1),
            "n_mri_leukoencephalopathy": n_leuko,
            "pct_mri_leuko": round(100 * n_leuko / total, 1),
            "n_encephalopathy_at_dx": n_enceph,
            "n_seizures": n_seizures,
            "n_myopathy": n_myopathy,
        },
        "key_teaching": {
            "lens_dislocation_cbs_pathognomonic": (
                "Ectopia lentis (inferior/nasal direction) is PATHOGNOMONIC for CBS HCU type I. "
                "Marfan syndrome (FBN1): lens subluxes superior/temporal direction. "
                "CBS: lens subluxes inferior/nasal. The direction of lens dislocation clinically "
                "distinguishes CBS from Marfan."
            ),
            "thromboembolism_#1_cause_death_cbs": (
                "Thromboembolism is the #1 cause of morbidity and mortality in untreated CBS HCU. "
                "By age 30: >50% risk of venous or arterial thromboembolic event without treatment. "
                "Mechanism: homocysteine is directly endotheliotoxic; also activates platelets and "
                "coagulation. Anticoagulation and betaine are mandatory."
            ),
            "pyridoxine_B6_trial_mandatory_CBS": (
                "B6 (pyridoxine) trial is MANDATORY for all CBS patients at diagnosis: ~50% are "
                "B6-responsive (>50% reduction in tHcy on B6 200-1000 mg/day). B6-responsive patients "
                "have milder phenotype, later onset, lower thrombosis risk. B6 stabilises PLP cofactor "
                "binding. 3-month trial required to determine responsiveness."
            ),
            "methionine_high_vs_low": (
                "METHIONINE LEVEL IS THE KEY DDx BETWEEN HCU TYPES: "
                "Met HIGH: CBS (transsulfuration block — both Hcy and Met rise) and AHCY (SAH inhibits MAT) and MAT1A. "
                "Met LOW: MTHFR, MTR/cblG, MTRR/cblE, MMACHC/cblC (remethylation block — Hcy rises, Met cannot be made). "
                "Met NORMAL or mildly elevated: MMADHC (variable). "
                "This is the most clinically useful first-pass biochemical DDx."
            ),
            "cblC_retinal_disease_pathognomonic": (
                "Retinal disease (macular degeneration + nystagmus) is PATHOGNOMONIC for cblC (MMACHC). "
                "ONLY cobalamin disorder with major ophthalmological involvement. Combined MMA+HC + "
                "retinopathy = cblC until proven otherwise. Ophthalmology surveillance mandatory from diagnosis."
            ),
            "betaine_central_all_HCU": (
                "Betaine (trimethylglycine) is a central treatment for all forms of homocystinuria. "
                "Betaine remethylates homocysteine → methionine via BHMT (betaine-homocysteine methyltransferase) "
                "in the liver, independently of B12, folate, or MTR. Reduces tHcy in CBS, MTHFR, cblC, cblD, cblE, cblG, AHCY. "
                "Standard dose: 6-9 g/day in adults; 100-250 mg/kg/day in children."
            ),
            "hydroxocobalamin_not_cyanocobalamin": (
                "For all cobalamin disorders (cblC/MMACHC, cblD/MMADHC, cblE/MTRR, cblG/MTR): "
                "USE HYDROXOCOBALAMIN (OHCbl), NOT cyanocobalamin (CNCbl). "
                "cblC/MMACHC: decyanation of CNCbl requires MMACHC enzyme; without MMACHC, CNCbl cannot be processed. "
                "OHCbl is directly incorporated into the cobalamin pool without requiring decyanation. "
                "OHCbl also has a longer half-life and higher retention."
            ),
            "nbs_detection_gaps": (
                "NBS DETECTION: CBS (Met elevated) and MMACHC/cblC (C3 elevated) and MAT1A (Met elevated) and AHCY (Met elevated) "
                "are DETECTED by standard MS/MS NBS. "
                "MISSED by standard NBS: MTHFR (Met normal/low), MTR/cblG (Met low), MTRR/cblE (Met low), "
                "cblD-HC-only subtype (no C3 elevation, Met normal). These require clinical recognition and targeted testing."
            ),
        },
    }


# ─── API: get_breakdown ───────────────────────────────────────────────────────────
def get_breakdown():
    """Return per-gene patient cohort breakdown."""
    genes_out = []
    for g in HCU_GENES:
        pts = g["patients"]
        genes_out.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "n_patients": g["n_patients"],
            "b6_responsive_pct": g["b6_responsive_pct"],
            "methionine_high": g["methionine_high"],
            "lens_dislocation": g["lens_dislocation"],
            "thromboembolism_risk": g["thromboembolism_risk"],
            "retinal_disease": g["retinal_disease"],
            "combined_mma": g["combined_mma"],
            "founder_variant": g["founder_variant"],
            "mean_thcy_dx": round(sum(p["thcy_umolL"] for p in pts) / len(pts), 1),
            "mean_methionine_dx": round(sum(p["methionine_umolL"] for p in pts) / len(pts), 1),
            "n_lens_dislocation": sum(1 for p in pts if p["lens_dislocation"]),
            "n_thromboembolism": sum(1 for p in pts if p["thromboembolism"]),
            "n_retinal_disease": sum(1 for p in pts if p["retinal_disease"]),
            "n_megaloblastic_anemia": sum(1 for p in pts if p["megaloblastic_anemia"]),
            "n_b6_responsive": sum(1 for p in pts if p["b6_responsive"]),
            "n_nbs_detected": sum(1 for p in pts if p["nbs_detected"]),
            "n_mma_elevated": sum(1 for p in pts if p.get("mma_umolL", 0) > 50),
            "n_mri_leuko": sum(1 for p in pts if p["mri_leukoencephalopathy"]),
            "n_encephalopathy": sum(1 for p in pts if p["encephalopathy_at_dx"]),
            "n_seizures": sum(1 for p in pts if p["seizures"]),
            "n_myopathy": sum(1 for p in pts if p["myopathy"]),
            "hallmark": g["hallmark"],
            "disease": g["disease"],
            "patients": pts[:10],
        })
    return {"genes": genes_out, "n_total": len(ALL_PATIENTS)}


# ─── API: get_definitions ─────────────────────────────────────────────────────────────
def get_definitions():
    """Return Homocystinuria-Atlas clinical term definitions."""
    return {
        "atlas": "Homocystinuria-Atlas — Complete 8-Gene Homocystinuria & Remethylation Disorders Atlas",
        "hcu_overview": {
            "full_name": (
                "Homocystinuria & Remethylation Disorders — a group of inborn errors of methionine "
                "cycle metabolism sharing elevated total plasma homocysteine (tHcy >15 µmol/L; severe >100) "
                "as the unifying biomarker. Causes: transsulfuration defects (CBS), remethylation defects "
                "(MTHFR, MTR/cblG, MTRR/cblE, MMACHC/cblC, MMADHC/cblD), SAH hydrolase deficiency (AHCY), "
                "and hepatic SAM synthesis defects (MAT1A)."
            ),
            "genes_in_atlas": 8,
            "unifying_biomarker": "Elevated total plasma homocysteine (tHcy) — present in all forms except MAT1A (mild/normal tHcy)",
            "key_ddx_point": "Methionine level: HIGH in CBS/AHCY/MAT1A (upstream accumulation); LOW in MTHFR/MTR/MTRR/MMACHC (remethylation block)",
            "central_treatment": "Betaine 6-9 g/day: reduces tHcy in ALL forms via BHMT remethylation pathway (liver)",
        },
        "definitions": [
            {
                "term": "Methionine Cycle and Homocysteine Metabolism: Pathway Architecture",
                "definition": (
                    "The methionine cycle is the central hub of one-carbon metabolism:\n\n"
                    "METHIONINE → SAM (methionine adenosyltransferase, MAT1A/MAT2A + ATP)\n"
                    "SAM → SAH (SAM-dependent methyltransferases: DNMT, HNMT, COMT, etc. — methylate DNA/RNA/proteins/lipids)\n"
                    "SAH → Hcy + Adenosine (adenosylhomocysteine hydrolase, AHCY)\n\n"
                    "Homocysteine fate — two competing pathways:\n"
                    "  (A) TRANSSULFURATION: Hcy + Ser → Cystathionine (CBS + PLP/B6) → Cysteine (CTH + PLP/B6) → Taurine/GSH\n"
                    "  (B) REMETHYLATION:\n"
                    "    Folate-dependent: Hcy + 5-MTHF → Met (MTR + methylcobalamin); 5-MTHF generated by MTHFR\n"
                    "    Betaine-dependent (liver only): Hcy + betaine → Met (BHMT)\n\n"
                    "Disease location in pathway:\n"
                    "  CBS: transsulfuration block → Hcy + Met both accumulate\n"
                    "  MTHFR/MTR/MTRR/cblC/cblD: remethylation block → Hcy accumulates, Met falls\n"
                    "  AHCY: SAH removal block → SAH accumulates, Hcy rises, global methylation inhibited\n"
                    "  MAT1A: SAM synthesis block (liver only) → Met accumulates, hepatic SAM low"
                ),
            },
            {
                "term": "Lens Dislocation in CBS HCU: Inferior/Nasal vs Marfan Superior/Temporal",
                "definition": (
                    "Ectopia lentis is the hallmark ophthalmological finding of CBS HCU.\n\n"
                    "CBS HOMOCYSTINURIA:\n"
                    "  Direction: inferior and nasal (downward and inward)\n"
                    "  Mechanism: homocysteine disrupts fibrillin-1 cross-linking in zonular fibres "
                    "(different from Marfan — direct fibrillin mutation vs Hcy-mediated disulfide "
                    "cross-linking impairment); progressive worsening\n"
                    "  Onset: typically 2-10 years without treatment; can be the presenting feature\n"
                    "  Prevalence: >90% of untreated CBS patients by age 10\n"
                    "  Associated: myopia (severe), secondary glaucoma, retinal detachment risk\n\n"
                    "MARFAN SYNDROME (FBN1):\n"
                    "  Direction: superior and temporal (upward and outward)\n"
                    "  Mechanism: fibrillin-1 mutation → zonular fibre structural weakness\n"
                    "  Associated: aortic root dilation (life-threatening), tall stature\n\n"
                    "DDx summary: Marfanoid body + ectopia lentis → check tHcy (CBS) + echocardiogram (Marfan). "
                    "CBS: tHcy >100 µmol/L, homocystinuria, no aortic root dilation. "
                    "Marfan: aortic root Z-score >2, FBN1 pathogenic variant, tHcy normal."
                ),
            },
            {
                "term": "B6 (Pyridoxine) Responsiveness in CBS HCU: Mechanism and Clinical Impact",
                "definition": (
                    "~50% of CBS patients demonstrate pyridoxine (vitamin B6) responsiveness:\n"
                    "  Defined: >50% reduction in tHcy after 3 months of B6 supplementation (200-1000 mg/day)\n"
                    "  Mechanism: PLP (pyridoxal-5'-phosphate = active B6) is the cofactor for CBS; "
                    "B6 administration → higher PLP levels → stabilises mutant CBS protein → "
                    "residual enzyme activity restored in those with PLP-binding site variants\n\n"
                    "VARIANT CORRELATION:\n"
                    "  B6-responsive: variants that preserve some PLP-binding capacity; residual activity present\n"
                    "  B6-non-responsive: variants that abolish catalytic site; B6 cannot rescue\n"
                    "  p.Ile278Thr: most common B6-NON-responsive allele; severe phenotype\n"
                    "  p.Gly307Ser: often B6-responsive\n\n"
                    "CLINICAL IMPACT:\n"
                    "  B6-responsive: DRAMATICALLY better prognosis — later onset ectopia lentis, "
                    "lower thrombosis risk, often normal IQ with treatment\n"
                    "  B6-non-responsive: methionine-restricted diet + betaine mandatory\n"
                    "  TRIAL MANDATORY: all CBS patients must receive 3-month B6 trial at diagnosis; "
                    "never assume non-responsiveness without trial"
                ),
            },
            {
                "term": "cblC (MMACHC) Retinal Disease: Macular Degeneration + Nystagmus",
                "definition": (
                    "Retinal disease in cblC is PATHOGNOMONIC — no other cobalamin disorder causes major retinopathy:\n\n"
                    "CLINICAL PRESENTATION:\n"
                    "  Macular degeneration: pigmentary changes, bull's-eye maculopathy on fundoscopy; "
                    "ERG (electroretinogram) abnormal (reduced cone and rod responses)\n"
                    "  Nystagmus: can be presenting feature in infancy; sensory nystagmus from macular dysfunction\n"
                    "  Onset: often within first year of life (early-onset cblC)\n"
                    "  Prevalence: >50% of cblC patients; higher in early-onset form\n\n"
                    "MECHANISM:\n"
                    "  AdoCbl deficiency: retinal pigment epithelium (RPE) cells require AdoCbl for "
                    "methylmalonyl-CoA mutase (MUT) activity; without AdoCbl, propionate accumulates → RPE toxicity\n"
                    "  Hyperhomocysteinaemia: direct retinal endothelial toxicity; VEGF dysregulation\n\n"
                    "MONITORING:\n"
                    "  Ophthalmology review at diagnosis; annual ERG and fundoscopy; OCT for macular thickness\n"
                    "  OHCbl treatment slows but may not fully halt progression\n\n"
                    "DDx: Combined MMA+HC + retinal disease = cblC until proven otherwise; "
                    "cblD (MMADHC) may have combined MMA+HC but NO retinal disease."
                ),
            },
            {
                "term": "Betaine (Trimethylglycine): Mechanism and Use in Homocystinuria",
                "definition": (
                    "Betaine is S-adenosyl-independent methyl donor for homocysteine remethylation:\n\n"
                    "MECHANISM:\n"
                    "  Betaine + Hcy → Met + Dimethylglycine (catalysed by BHMT — betaine-homocysteine methyltransferase)\n"
                    "  BHMT is expressed in liver (and kidney, lower level); betaine remethylation is "
                    "INDEPENDENT of B12, folate, MTR, MTRR, or MTHFR — functions even when all these are deficient\n\n"
                    "CLINICAL APPLICATIONS:\n"
                    "  CBS: B6-non-responsive → betaine + methionine-restricted diet; standard of care\n"
                    "  MTHFR, cblG, cblE: betaine reduces tHcy when primary remethylation (MTR) impaired\n"
                    "  cblC, cblD: betaine reduces Hcy component independently of OHCbl\n"
                    "  AHCY: betaine may paradoxically worsen SAH accumulation (generates more dimethylglycine "
                    "which is SAM-dependent to metabolise further); use with caution in AHCY\n\n"
                    "DOSE: 6-9 g/day in adults (divided 2-3 times); 100-250 mg/kg/day in children\n"
                    "MONITORING: plasma tHcy + methionine (betaine raises methionine — watch for "
                    "hypermethioninemia especially in CBS where Met already elevated; cerebral oedema "
                    "reported in rare cases of extreme hypermethioninemia on betaine in CBS)"
                ),
            },
            {
                "term": "NBS Detection in Homocystinuria Disorders: What Is and Is NOT Caught",
                "definition": (
                    "DETECTED BY STANDARD MS/MS NEWBORN SCREEN:\n"
                    "  CBS: elevated methionine (Met >50 µmol/L typical; varies by programme); "
                    "detected in ~70-80% if methionine-based screening used\n"
                    "  MMACHC/cblC: elevated C3-carnitine (propionylcarnitine) — most reliable NBS marker; "
                    "detected in >90% of early-onset cblC\n"
                    "  MAT1A: elevated methionine — most common cause of isolated hypermethioninemia on NBS\n"
                    "  AHCY: elevated methionine — may be detected\n\n"
                    "MISSED BY STANDARD NBS:\n"
                    "  MTHFR: methionine NORMAL or LOW; no C3 elevation; no standard NBS marker → "
                    "clinical diagnosis on encephalopathy/MRI findings\n"
                    "  MTR/cblG: methionine LOW; MMA NORMAL; often missed\n"
                    "  MTRR/cblE: methionine LOW; MMA NORMAL; often missed\n"
                    "  MMADHC/cblD-HC-only: no C3 elevation; methionine normal → missed\n\n"
                    "IMPLICATIONS:\n"
                    "  Negative NBS does NOT exclude remethylation disorders\n"
                    "  Any infant/child with unexplained encephalopathy, leukoencephalopathy, "
                    "or megaloblastic anemia: measure tHcy, plasma amino acids, acylcarnitine profile, and MMA\n"
                    "  Targeted homocysteine measurement at NBS reduces missed cases"
                ),
            },
            {
                "term": "AHCY Deficiency: SAH Accumulation and Global Hypomethylation",
                "definition": (
                    "AHCY deficiency has a unique mechanism distinct from all other HCU types:\n\n"
                    "SAH IS THE INHIBITOR:\n"
                    "  SAH (S-adenosylhomocysteine) is generated by every SAM-dependent methylation reaction\n"
                    "  SAH is a potent product inhibitor of ALL SAM-dependent methyltransferases "
                    "(Ki often in the nanomolar range — tighter than SAM Km in some cases)\n"
                    "  AHCY is the ONLY enzyme to remove SAH; without AHCY → SAH builds up → "
                    "all methylation reactions globally inhibited regardless of SAM levels\n\n"
                    "CONSEQUENCES OF GLOBAL HYPOMETHYLATION:\n"
                    "  DNA methylation: epigenetic deregulation → developmental abnormalities\n"
                    "  Myelin methylation (TPMT → phosphatidylcholine, myelin): demyelination\n"
                    "  Creatine synthesis (GAMT): muscle creatine depleted → myopathy + CK elevation\n"
                    "  Catecholamines (PNMT: noradrenaline → adrenaline): neurological + autonomic effects\n"
                    "  RNA cap methylation: protein synthesis impaired\n\n"
                    "DISTINGUISHING FROM CBS: both have HIGH methionine + tHcy elevated, but:\n"
                    "  AHCY: SAH dramatically elevated (key diagnostic biomarker); CK elevated; myopathy; "
                    "no lens dislocation; no thromboembolism\n"
                    "  CBS: SAH normal; CK normal; lens dislocation; thromboembolism; cystathionine absent"
                ),
            },
        ],
    }
