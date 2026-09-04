#!/usr/bin/env python3
"""AA-Disorders-Atlas — Complete 10-Gene Amino Acid Disorders Atlas
PAH · BCKDHA · CBS · FAH · TAT · GLDC · OAT · SLC7A7 · PHGDH · SLC6A19
400-patient aggregate cohort (10 × 40, seeds 858–867)

Amino Acid Disorders (AAD) facts:
  - Inherited defects in enzymes or transporters of amino acid catabolism / biosynthesis
  - Detected by NBS (newborn screening) via tandem MS/MS amino acid and acylcarnitine profiles
  - Wide spectrum: from asymptomatic (Hartnup) to neonatal lethal (classic MSUD, GLDC)
  - PKU (PAH) is the most common IEM in European populations: 1 in 10,000
  - NTBC (nitisinone) for FAH/Tyrosinemia type 1 is one of the most impactful treatments in IEM
  - B6 (pyridoxine) responsiveness in CBS is a critical treatment-determining phenotype
  - BH4 (sapropterin) responsiveness in PAH defines a pharmacological subset of PKU
  - Dietary amino acid restriction is the cornerstone for PAH, CBS, BCKDHA, TAT, OAT

ATLAS SCOPE (10 classic AA disorder genes):
  Phenylalanine pathway:
    PAH   — phenylalanine hydroxylase → PKU / HPA (12q23.2)
  Branched-chain amino acid catabolism:
    BCKDHA — BCKA dehydrogenase E1α → MSUD type Ia (19q13.2)
  Sulfur amino acid metabolism:
    CBS   — cystathionine β-synthase → Homocystinuria type 1 (21q22.3)
  Tyrosine catabolism:
    FAH   — fumarylacetoacetate hydrolase → Tyrosinemia type 1 (15q25.1)
    TAT   — tyrosine aminotransferase → Tyrosinemia type 2 / Richner-Hanhart (16q22.2)
  Glycine cleavage system:
    GLDC  — glycine decarboxylase P-protein → NKH (9p24.1)
  Ornithine catabolism:
    OAT   — ornithine δ-aminotransferase → Gyrate atrophy (10q26.13)
  Cationic amino acid transport:
    SLC7A7 — y+LAT1 → Lysinuric protein intolerance / LPI (14q11.2)
  Serine biosynthesis:
    PHGDH — 3-phosphoglycerate dehydrogenase → Serine deficiency (1p12)
  Neutral amino acid transport:
    SLC6A19 — B0AT1 → Hartnup disease (5p15.33)

CRITICAL CLINICAL RULES:
  1. PAH: BH4-responsiveness (sapropterin) test MANDATORY before prescribing — only 25-50% respond
     (mild PKU predominantly); Phe restriction + large neutral AA for ALL classic PKU.
     Maternal PKU: Phe >360 µmol/L in pregnancy → fetal teratogenesis (microcephaly, CHD).
  2. BCKDHA (MSUD): Leucine is the PRIMARY neurotoxin — not isoleucine or valine;
     maple syrup odor in urine/cerumen by day 4-5; allo-Ile is pathognomonic;
     liver transplant is CURATIVE (metabolic) but does NOT reverse neurological damage.
  3. CBS: B6-responsiveness testing MANDATORY — 50% respond; non-responders need
     methionine-restricted diet + betaine. Lens subluxation is DOWNWARD (vs Marfan: upward).
     MTHFR is NOT CBS; different enzyme, different biochemistry, different treatment.
  4. FAH (Tyrosinemia type 1): succinylacetone (urine/DBS) is pathognomonic;
     NTBC (nitisinone) started BEFORE liver biopsy — biopsy risk extreme with coagulopathy;
     NTBC is essentially curative if started early; HCC risk persists in late-diagnosed.
  5. TAT (Tyrosinemia type 2): low-Tyr/Phe diet resolves keratitis + keratoderma COMPLETELY —
     dietary control is the definitive treatment; NTBC NOT indicated (no succinylacetone/HT1).
  6. GLDC (NKH): NO curative treatment; sodium benzoate + dextromethorphan is palliative;
     CSF:plasma glycine ratio >0.08 confirms NKH (normal <0.03); VPA CONTRAINDICATED (increases glycine).
  7. OAT (Gyrate atrophy): arginine-restricted diet + pyridoxine trial (B6-responsive ~5-10%);
     progressive chorioretinal atrophy → legal blindness by 40-50y; creatine supplementation controversial.
  8. SLC7A7 (LPI): citrulline supplementation BYPASSES transport defect (citrulline absorbed via
     neutral transporter); pulmonary alveolar proteinosis (PAP) in 65% → monitor lung function;
     protein restriction ALONE is insufficient (citrulline essential).
  9. PHGDH (Serine deficiency): L-serine + glycine supplementation; CSF serine monitoring mandatory;
     Neu-Laxova syndrome (lethal) is severe biallelic PHGDH loss; supplement EARLY before irreversible brain damage.
  10. SLC6A19 (Hartnup): most patients ASYMPTOMATIC (peptide absorption intact via PepT1);
      symptomatic: nicotinamide 40-200 mg/day + high-protein diet; rash + ataxia in childhood = Hartnup.

COHORT: 10 × 40 = 400 patient slots (seeds 858–867; gene-specific seeds)
"""

import random

SEED_BASE = 858

# ── All 10 Amino Acid Disorder genes ──────────────────────────────────────────
AA_GENES = [
    # ── PAH — Phenylalanine Hydroxylase ──
    {
        "gene": "PAH", "alias": "PAH — Phenylalanine Hydroxylase (PKU / HPA)",
        "aa": "451 aa", "kDa": "51.9 kDa",
        "gene_class": "hydroxylase",
        "locus": "12q23.2", "omim_gene": 612349,
        "phenotype": "PKU / Hyperphenylalaninemia — Phe accumulation → ID, seizures, mousy odor",
        "disease": (
            "PAH biallelic loss → Phenylketonuria (PKU, OMIM #261600) or Hyperphenylalaninemia (HPA). "
            "PAH encodes phenylalanine-4-hydroxylase (451aa, 51.9 kDa), which hydroxylates Phe → Tyr "
            "using tetrahydrobiopterin (BH4) as essential cofactor. Deficiency → Phe accumulates in "
            "blood and brain. Classic PKU: plasma Phe >1200 µmol/L (>1200 = classic; 600-1200 = "
            "moderate; 360-600 = mild PKU; <360 = mild HPA). Untreated classic PKU: severe intellectual "
            "disability (IQ <50), seizures, autistic features, mousy/musty urine odor (phenylacetate, "
            "phenyllactate), fair skin and hair (Tyr/melanin deficiency), eczema. PKU is the most "
            "common IEM in European populations: 1 in 10,000 (range 1 in 4,500 Turkey to 1 in 50,000 "
            "Japan). NBS has essentially eliminated classic PKU complications in screened populations. "
            "Maternal PKU: plasma Phe >360 µmol/L during pregnancy → fetal teratogenesis independent "
            "of fetal genotype (heterozygous fetus): microcephaly, CHD, intrauterine growth restriction. "
            "Maternal PKU is the major current cause of PKU-related harm globally in countries with NBS."
        ),
        "inheritance": "Autosomal recessive. PAH locus 12q23.2. >1,000 pathogenic variants. Prevalence 1 in 10,000 European.",
        "hallmark": (
            "PAH HALLMARKS: (1) MOUSY/MUSTY ODOR — phenylacetate and phenyllactate in urine/sweat of "
            "untreated PKU (pathognomonic in classical era pre-NBS); "
            "(2) FAIR PHENOTYPE — hypopigmented skin/hair/eyes due to Phe competitive inhibition of "
            "tyrosinase (Phe blocks Tyr → melanin); "
            "(3) BH4 (SAPROPTERIN) RESPONSIVENESS — 25-50% of PAH-deficient patients (mild/moderate PKU "
            "predominantly) respond to high-dose BH4 loading test (20 mg/kg × 24-48h) with >30% "
            "reduction in plasma Phe; pharmacological chaperone mechanism for some; cofactor supplementation "
            "for BH4-synthesis defects (PTPS, DHPR, GCH1) must be excluded first; "
            "(4) PHE-RESTRICTED DIET: all PAH-deficient patients need Phe-restricted diet (Phe <360 "
            "in classic PKU, <600 in mild); large neutral AA (LNAA) supplement competes with Phe for "
            "BBB transport → reduces CNS Phe; Phe-free amino acid formula provides protein; "
            "(5) BRAIN WHITE MATTER CHANGES — periventricular T2-hyperintensity on MRI correlates with "
            "metabolic control (plasma Phe); reversible with strict diet; "
            "(6) PAXgene (pegvaliase) — PAH-substituting enzyme (phenylalanine ammonia lyase, "
            "subcutaneous, immunogenic); FDA-approved 2018 for adult PKU uncontrolled on other therapy; "
            "(7) MATERNAL PKU SYNDROME — even heterozygous fetus damaged by maternal Phe >360 µmol/L; "
            "requires STRICT maternal Phe control BEFORE conception; "
            "(8) EXECUTIVE FUNCTION DEFICITS — even well-controlled PKU adults have attention/working "
            "memory deficits with Phe in normal range (residual neurotoxicity); "
            "(9) PTPS/DHPR/GCH1: BH4 metabolism defects → also HPA with NORMAL PAH — must exclude by "
            "urine pterins + DHPR enzyme in all NBS-detected HPA cases"
        ),
        "key_ddx": (
            "PAH DDx: (1) BH4-cofactor defects (PTPS, DHPR, GCH1, PTS, QDPR) — also HPA, but with "
            "progressive neurology despite Phe control; diagnose by urine pterins + DHPR in DBS; "
            "(2) DNAJC12 deficiency — BH4 cofactor delivery; (3) Maternal PKU — high Phe from "
            "heterozygous/homozygous mother (check maternal plasma Phe if infant has microcephaly/CHD); "
            "(4) Transient HPA of newborn — resolves spontaneously; low Phe diet not needed."
        ),
        "diet_treatment": "Low-Phe diet + LNAA + Phe-free AA formula (ALL PKU); BH4/sapropterin if responsive (25-50%); pegvaliase (adult refractory)",
        "gene_therapy_status": "Phase 1-2 trials (AAV-PAH intramuscular); IND-approved pegvaliase SC (PAL enzyme)",
        "critical_ci": (
            "CRITICAL CI: (1) BH4 (sapropterin) WITHOUT formal loading test — only 25-50% respond; "
            "prescribing blindly wastes resources and may delay dietary control; "
            "(2) Relaxing maternal Phe control during pregnancy — maternal Phe >360 µmol/L teratogenic "
            "regardless of fetal genotype; "
            "(3) Stopping Phe-restricted diet in adolescence — causes white matter changes + executive "
            "function deficits; diet is LIFELONG; "
            "(4) Treating PTPS/DHPR (BH4 metabolism defects) with Phe restriction alone — neurological "
            "deterioration despite Phe control; requires neurotransmitter precursors (DOPA/5-HTP)"
        ),
        "nbs_marker": "Phe DBS (MS/MS) — plasma Phe elevation; Phe:Tyr ratio >3 in PKU",
        "key_biomarker": "Plasma Phe (>1200 µmol/L = classic PKU; 360-600 = mild PKU); urine phenylpyruvate/phenyllactate; plasma Tyr (low)",
        "severity_spectrum": "Classic PKU (Phe >1200) → Moderate PKU (600-1200) → Mild PKU (360-600) → Mild HPA (<360)",
        "founder_variant": "p.Arg408Trp (Northern European, severe/classic PKU, 30-50% of alleles); p.IVS10nt-11G→A (Eastern European)",
        "key_variants": [
            "p.Arg408Trp (R408W) — Northern European null, classic PKU",
            "p.IVS10nt-11G→A — Eastern European splice, severe",
            "p.Tyr414Cys — Mediterranean, moderate",
            "p.Ile65Thr — mild PKU, BH4-responsive",
            "p.Ala300Ser — BH4-responsive, mild HPA",
        ],
        "seed": SEED_BASE + 0,
    },
    # ── BCKDHA — MSUD E1α ──
    {
        "gene": "BCKDHA", "alias": "BCKDHA — BCKA Dehydrogenase E1α (MSUD type Ia / Maple Syrup Urine Disease)",
        "aa": "445 aa", "kDa": "46.3 kDa",
        "gene_class": "dehydrogenase",
        "locus": "19q13.2", "omim_gene": 608348,
        "phenotype": "MSUD — Leu/Ile/Val + BCKA accumulation; neonatal encephalopathy; maple syrup odor",
        "disease": (
            "BCKDHA (BCKD E1α) biallelic loss → Maple Syrup Urine Disease (MSUD, OMIM #248600). "
            "BCKDHA encodes the E1α subunit (445aa, 46.3 kDa) of branched-chain α-ketoacid "
            "dehydrogenase (BCKD), the mitochondrial rate-limiting enzyme for leucine, isoleucine, "
            "and valine catabolism. Also encoded by BCKDHB (E1β, 342aa, 6q14.1) and DBT (E2, 421aa, "
            "1p21.2) — all three can cause MSUD. MSUD: BCAA (Leu, Ile, Val) + BCKA (branched-chain "
            "keto-acids) accumulate → neonatal encephalopathy in classic form. Classic MSUD: "
            "normal at birth, then poor feeding, maple syrup odor in urine/cerumen (by day 4-5), "
            "progressive encephalopathy, opisthotonus, cerebral edema, apnea. Leucine is the PRIMARY "
            "neurotoxin (competes with other LNAA for BBB transport, depletes Tyr/Phe/Trp). "
            "Incidence: 1 in 185,000 globally; much higher in Old Order Mennonite (1 in 380) due to "
            "p.Tyr393Asn BCKDHA founder. Thiamine-responsive MSUD: ~5% of cases — high-dose thiamine "
            "(10-1000 mg/day) improves BCKD activity; try in ALL patients."
        ),
        "inheritance": "Autosomal recessive. BCKDHA (19q13.2), BCKDHB (6q14.1), DBT (1p21.2) — all cause MSUD. Incidence 1 in 185,000.",
        "hallmark": (
            "BCKDHA/MSUD HALLMARKS: (1) MAPLE SYRUP ODOR — in urine and cerumen by day 4-5; "
            "pathognomonic for MSUD; caused by sotolone (branched-chain keto-acid fermentation product); "
            "(2) ALLO-ISOLEUCINE — pathognomonic biomarker; allo-Ile normally absent in plasma; "
            "its presence on plasma amino acid chromatography is diagnostic of MSUD; "
            "(3) LEUCINE NEUROTOXICITY — Leu is PRIMARY neurotoxin (>1000 µmol/L = crisis level); "
            "competes with Tyr/Phe/Trp/Lys for BBB transport via LAT1 → depletes brain monoamine "
            "precursors; cerebral edema, myelinopathy; goal: plasma Leu 75-200 µmol/L; "
            "(4) BCAA-RESTRICTED DIET: lifelong; Leu-free/BCAA-free infant formula during crisis; "
            "isoleucine + valine supplementation MANDATORY during acute Leu restriction to prevent "
            "deficiency (all three compete for absorption); "
            "(5) INSULIN THERAPY — drives Leu into cells; used IV during hyperaminoacidaemic crisis; "
            "0.05-0.1 units/kg/h with dextrose; "
            "(6) LIVER TRANSPLANTATION — curative for metabolic phenotype (transplanted liver "
            "provides sufficient BCKD activity); DOES NOT reverse pre-existing neurological damage; "
            "patients may liberalize diet post-Tx; "
            "(7) THIAMINE TRIAL — high-dose thiamine (ThPP is BCKD cofactor) in all new diagnoses; "
            "~5% respond with significant Leu reduction; "
            "(8) HEMODIALYSIS/ECMO — for severe neonatal crisis with Leu >1500 µmol/L and "
            "cerebral edema; rapidly clears Leu; "
            "(9) Old Order Mennonite founder p.Tyr393Asn BCKDHA: 1 in 380 in this community; "
            "universal NBS in community recommended"
        ),
        "key_ddx": (
            "BCKDHA DDx: (1) Other MSUD-causing genes (BCKDHB E1β, DBT E2, DLD E3) — same biochemistry, "
            "distinguish by sequencing; DLD (E3) deficiency also causes PDC/OGD deficiency with lactic acidosis; "
            "(2) Sepsis/meningitis — neonatal encephalopathy differential; allo-Ile rules in MSUD; "
            "(3) BCAA elevation from rhabdomyolysis or hepatic failure — allo-Ile absent; "
            "(4) IVA (isovaleric acidemia) — maple syrup smell variant, but elevated C5 acylcarnitine not Leu/Ile/Val"
        ),
        "diet_treatment": "Lifelong BCAA-restricted diet; Leu-free/BCAA-free formula; Ile+Val supplementation during Leu restriction; thiamine trial; insulin IV for crisis; liver transplant",
        "gene_therapy_status": "Phase 1 trials (AAV-BCKDHA intrahepatic); liver transplant currently definitive",
        "critical_ci": (
            "CRITICAL CI: (1) Restricting only leucine without Ile/Val supplementation — Ile/Val "
            "deficiency causes metabolic crisis and rash (pellagra-like); "
            "(2) Delaying thiamine trial — all newly diagnosed MSUD must receive thiamine trial; "
            "(3) Liver Tx without pre-transplant metabolic control — active encephalopathy increases "
            "surgical risk; aim Leu <300 before Tx; "
            "(4) Standard total parenteral nutrition without BCAA-free formula — triggers crisis "
            "due to BCAA load; use MSUD-specific PN formulations"
        ),
        "nbs_marker": "Leu+Ile+Val elevated (MS/MS DBS); Leu:Phe ratio; allo-Ile (second-tier); urine organic acids (BCKA)",
        "key_biomarker": "Plasma allo-Ile (pathognomonic, absent normally); plasma Leu (>1000 µmol/L = crisis); urine BCKA; plasma Ile/Val",
        "severity_spectrum": "Classic MSUD (neonatal, <2% BCKD activity) → Intermediate (5-20%) → Mild (<40%) → Thiamine-responsive (~5%)",
        "founder_variant": "p.Tyr393Asn (BCKDHA, Old Order Mennonite community); p.Gly278Asp (German founder)",
        "key_variants": [
            "p.Tyr393Asn (Y393N) — Old Order Mennonite, BCKDHA, classic",
            "p.Gly278Asp — German/Central European, BCKDHA",
            "p.Phe364Cys — BCKDHA null, classic",
            "p.Arg183Gln — BCKDHB, intermediate",
            "p.Glu368Ter — DBT E2, classic",
        ],
        "seed": SEED_BASE + 1,
    },
    # ── CBS — Cystathionine β-synthase (Homocystinuria type 1) ──
    {
        "gene": "CBS", "alias": "CBS — Cystathionine β-Synthase (Homocystinuria / HCU Type 1)",
        "aa": "551 aa", "kDa": "63.2 kDa",
        "gene_class": "lyase",
        "locus": "21q22.3", "omim_gene": 613381,
        "phenotype": "HCU type 1 — homocysteine + Met accumulation; ectopia lentis DOWNWARD, thromboembolism, Marfanoid",
        "disease": (
            "CBS biallelic loss → Homocystinuria type 1 (HCU, OMIM #236200), the most common disorder "
            "of sulfur amino acid metabolism. CBS encodes cystathionine β-synthase (551aa, 63.2 kDa, "
            "homotetrameric), which catalyses condensation of serine + homocysteine → cystathionine "
            "using PLP (pyridoxal-5'-phosphate/B6) as cofactor; cystathionine is then cleaved by "
            "CGL to cysteine. CBS deficiency → homocysteine and methionine accumulate; cysteine and "
            "downstream thioethers depleted. Incidence: 1 in 200,000-300,000 (NBS; higher in Ireland: "
            "1 in 65,000 due to p.Gly307Ser founder). Clinical tetrad: (1) ectopia lentis (80%), "
            "(2) skeletal (tall stature, Marfanoid, scoliosis, osteoporosis), (3) vascular "
            "(thromboembolic disease — DVT/PE/stroke, LEADING CAUSE OF DEATH), (4) CNS "
            "(ID/developmental delay in 50% untreated). Distinguish HCU type 1 from "
            "remethylation defects (MTHFR, MTR, MTRR, MMACHC): CBS → HIGH Met + HIGH Hcy; "
            "remethylation defects → LOW Met + HIGH Hcy."
        ),
        "inheritance": "Autosomal recessive. CBS 21q22.3. Incidence ~1 in 200,000-300,000; Irish founder 1 in 65,000.",
        "hallmark": (
            "CBS HALLMARKS: (1) ECTOPIA LENTIS DOWNWARD — lens subluxation downward and nasally; "
            "vs Marfan syndrome (FBN1): upward/superotemporal; both have Marfanoid habitus but "
            "HCU lens goes DOWN, Marfan goes UP; slit-lamp examination essential; "
            "(2) B6-RESPONSIVENESS — 50% of CBS patients respond to pharmacological B6 (pyridoxine, "
            "100-500 mg/day) → normalize plasma total homocysteine (tHcy); B6-responsive patients "
            "have better outcomes and milder phenotype; non-responders need full dietary + betaine therapy; "
            "B6 trial MANDATORY in ALL newly diagnosed CBS; "
            "(3) BETAINE (trimethylglycine) — re-methylates Hcy → Met via BHMT (betaine-homocysteine "
            "methyltransferase) pathway, bypassing CBS; used for B6-non-responders; dose 100-200 mg/kg/day; "
            "risk of hypermethioninaemia (monitor Met — cerebral oedema reported at very high Met >1000 µmol/L); "
            "(4) METHIONINE-RESTRICTED DIET — for B6-non-responders; cystine supplementation needed "
            "(cysteine essential, CBS blocks its synthesis pathway); "
            "(5) THROMBOEMBOLIC DISEASE — arterial AND venous; DVT in young patients (<30y), "
            "Hcy-induced endothelial damage + platelet activation + coagulation; "
            "consider anticoagulation + antiplatelet; elective surgery/immobilisation → "
            "thromboprophylaxis mandatory; "
            "(6) MTHFR C677T — NOT CBS deficiency; mild HHcy (up to 30 µmol/L); low/normal Met; "
            "folic acid treatment; does NOT cause ectopia lentis or severe clinical HCU; "
            "(7) OSTEOPOROSIS — Hcy inhibits collagen crosslinking; DEXA mandatory; "
            "(8) TOTAL HOMOCYSTEINE (tHcy) goal <50 µmol/L (B6-responsive aim <30; normal <15)"
        ),
        "key_ddx": (
            "CBS DDx: (1) Marfan syndrome (FBN1) — Marfanoid + ectopia lentis; CBS → DOWNWARD lens, "
            "HIGH Met, HIGH tHcy; Marfan → upward lens, normal amino acids; "
            "(2) Remethylation defects (MTHFR, MTR, MTRR, MMACHC/cblC) — HIGH tHcy but LOW Met; "
            "no ectopia lentis; different treatment (folate/cobalamin, NOT Met restriction); "
            "(3) Sulfite oxidase deficiency (SUOX) — dislocation + ID; urine sulfite positive; "
            "(4) Beals syndrome (FBN2) — Marfanoid + camptodactyly; no Hcy; "
            "(5) Nutritional B12/folate deficiency — mildly elevated tHcy; normal Met; no lens"
        ),
        "diet_treatment": "B6 trial ALL patients; B6-responsive: pyridoxine lifelong; B6-non-responsive: Met restriction + betaine + cystine; thromboprophylaxis",
        "gene_therapy_status": "Phase 2 trial (mRNA therapy, MRT5201/OX2003); AAV hepatic early studies",
        "critical_ci": (
            "CRITICAL CI: (1) Not trialling B6 — 50% of patients manageable with pyridoxine alone; "
            "(2) Betaine without monitoring plasma Met — hypermethioninaemia (>1000 µmol/L) → "
            "cerebral oedema and encephalopathy; "
            "(3) Diagnosing as Marfan syndrome without tHcy/Met measurement — lens direction is "
            "the key bedside clue; "
            "(4) Not providing thromboprophylaxis for surgery/immobilisation — DVT/PE leading cause "
            "of death in untreated/poorly controlled HCU; "
            "(5) Treating MTHFR with Met restriction — wrong diagnosis/wrong treatment"
        ),
        "nbs_marker": "Met elevated DBS (MS/MS) — moderate sensitivity; tHcy/Met ratio; homocystine on urine amino acids",
        "key_biomarker": "Plasma total homocysteine (tHcy >50 µmol/L = HCU; normal <15); plasma Met (HIGH in CBS, LOW in remethylation defects); urine homocystine (sodium nitroprusside test positive)",
        "severity_spectrum": "B6-responsive (milder, 50%) → B6-non-responsive (severe, 50%) — lens/thrombus/ID risk higher non-responsive",
        "founder_variant": "p.Gly307Ser (Ireland/Celtic, B6-responsive, 50-70% Irish HCU alleles); p.Ile278Thr (mild/B6-responsive)",
        "key_variants": [
            "p.Gly307Ser — Ireland/Celtic founder, B6-responsive",
            "p.Ile278Thr — B6-responsive, mild",
            "p.Trp323Ter — B6-non-responsive, severe null",
            "p.Arg125Gln — B6-responsive intermediate",
            "p.Glu302Lys — B6-non-responsive",
        ],
        "seed": SEED_BASE + 2,
    },
    # ── FAH — Fumarylacetoacetate Hydrolase (Tyrosinemia type 1) ──
    {
        "gene": "FAH", "alias": "FAH — Fumarylacetoacetate Hydrolase (Tyrosinemia Type 1 / HT1)",
        "aa": "419 aa", "kDa": "46.3 kDa",
        "gene_class": "hydrolase",
        "locus": "15q25.1", "omim_gene": 613871,
        "phenotype": "HT1 — succinylacetone (pathognomonic); liver failure, HCC, tubular nephropathy, porphyria-like crisis",
        "disease": (
            "FAH biallelic loss → Tyrosinemia type 1 (HT1, OMIM #276700), the most severe amino acid "
            "disorder affecting the liver. FAH encodes fumarylacetoacetate hydrolase (419aa, 46.3 kDa), "
            "the last enzyme in tyrosine catabolism: fumarylacetoacetate → fumarate + acetoacetate. "
            "Deficiency → fumarylacetoacetate (FAA) and maleylacetoacetate (MAA) accumulate; "
            "FAA is spontaneously converted to SUCCINYLACETONE (SA, 4,6-dioxoheptanoic acid), "
            "which is the pathognomonic biomarker and primary toxin. Succinylacetone inhibits "
            "delta-aminolevulinic acid (ALA) dehydratase → secondary porphyria-like crises; "
            "also alkylates DNA → hepatocellular carcinoma risk. Clinical: acute (infantile) form: "
            "liver failure <6 months (coagulopathy, jaundice, ascites, hypoglycaemia), tubular "
            "nephropathy (Fanconi syndrome with phosphaturia, glucosuria, aminoaciduria), "
            "porphyria-like painful neurological crises (ALA dehydratase inhibition). "
            "Chronic form: cirrhosis, HCC (10-15% by age 5 without NTBC), Fanconi nephropathy. "
            "Incidence: 1 in 100,000-120,000; French-Canadian Saguenay-Lac-Saint-Jean: 1 in 1,846."
        ),
        "inheritance": "Autosomal recessive. FAH 15q25.1. French-Canadian founder p.IVS12+5G→A (Quebec). 1 in 100,000 globally.",
        "hallmark": (
            "FAH HALLMARKS: (1) SUCCINYLACETONE (SA) — pathognomonic; detectable in urine (GCMS) "
            "and DBS (MS/MS) in ALL HT1 patients regardless of NTBC status; "
            "SA is now the GOLD STANDARD for NBS (DBS succinylacetone: near 100% sensitivity + "
            "specificity for HT1, unlike Tyr elevation which is non-specific); "
            "(2) NTBC (NITISINONE / ORFADIN) — inhibits 4-hydroxyphenylpyruvate dioxygenase "
            "(4-HPPD), two steps upstream of FAH → stops SA + FAA production; "
            "essentially curative if started before liver damage; "
            "start IMMEDIATELY on clinical suspicion — do NOT wait for biopsy; "
            "dose 1 mg/kg/day → reduces to 0.5 mg/kg maintenance; "
            "(3) NTBC + LOW-TYR/PHE DIET — NTBC causes Tyr to rise (upstream block); "
            "diet restricts Tyr/Phe to prevent keratitis/keratoderma (secondary HT2-like phenotype "
            "from elevated Tyr on NTBC); target plasma Tyr <500 µmol/L on NTBC; "
            "(4) HCC RISK persists even on NTBC — especially if NTBC started after age 2y or "
            "with pre-existing liver nodules; liver AFP monitoring every 6 months mandatory; "
            "alpha-fetoprotein (AFP) is biomarker for active HT1 (elevated) AND HCC (rising on NTBC = alarm); "
            "(5) PORPHYRIA-LIKE CRISIS — acute neurological crisis: painful ascending neuropathy, "
            "hypertension, autonomic instability; SA inhibits ALA dehydratase; treated with hemin "
            "and NTBC; NOT classic porphyria (ALAD inhibited by SA, not by enzyme defect); "
            "(6) FANCONI NEPHROPATHY — proximal tubular dysfunction (phosphaturia → rickets, "
            "glucosuria, aminoaciduria); resolves on NTBC; "
            "(7) LIVER BIOPSY CONTRAINDICATED before SA result — profound coagulopathy (HT1 liver failure); "
            "(8) LIVER TRANSPLANT — was the only treatment pre-NTBC; still used for liver failure "
            "before NTBC diagnosis, HCC, or NTBC non-compliance; post-Tx: corrects metabolic disease "
            "but SA may persist from extrahepatic FAH expression; "
            "(9) French-Canadian IVS12+5G→A founder: 0.25 allele freq Saguenay-Lac-Saint-Jean "
            "(carrier frequency 1 in 15!)"
        ),
        "key_ddx": (
            "FAH DDx: (1) Galactosemia (GALT) — infant liver failure + E. coli sepsis; "
            "galactitol in urine not SA; (2) Hepatitis/TORCH — viral serology; "
            "(3) NKH/GLDC — liver less affected; CSF glycine; (4) Other organic acidurias — "
            "SA absent; urine organic acids distinguish; (5) Classic porphyrias (AIP/VP) — "
            "urine ALA + PBG elevated; SA absent; different enzyme in heme pathway; "
            "(6) Tyrosinemia type 2 (TAT) — elevated Tyr but NO SA, no liver disease, keratitis"
        ),
        "diet_treatment": "NTBC (nitisinone) 1 mg/kg/day IMMEDIATELY on suspicion + low-Tyr/Phe diet; liver transplant for failure/HCC; AFP + SA monitoring",
        "gene_therapy_status": "NTBC is so effective that gene therapy not a priority; hepatocyte transplant investigated",
        "critical_ci": (
            "CRITICAL CI: (1) Liver biopsy before SA/NTBC — coagulopathy; risk of fatal haemorrhage; "
            "(2) Delaying NTBC for confirmatory genetics — start on clinical + SA basis; "
            "(3) Starting NTBC without low-Tyr/Phe diet — NTBC causes secondary hypertyrosinaemia "
            "→ keratitis (corneal crystals); "
            "(4) Not monitoring AFP every 6m — HCC can arise even on NTBC; "
            "(5) Treating porphyria-like crisis with heme infusion alone without NTBC — "
            "recurrence until SA production stopped"
        ),
        "nbs_marker": "Succinylacetone DBS (MS/MS) — gold standard; Tyr elevated (non-specific, not used alone); AFP elevated in acute HT1",
        "key_biomarker": "Urine succinylacetone (GCMS, pathognomonic); plasma AFP (high in acute HT1/HCC marker); plasma Tyr (elevated); coagulation (INR markedly prolonged)",
        "severity_spectrum": "Acute infantile (<6m, liver failure) → Subacute (6-24m) → Chronic (>2y, cirrhosis/HCC risk)",
        "founder_variant": "p.IVS12+5G→A (splice, Quebec French-Canadian, ~80% Quebec HT1 alleles); p.Gln64His (Mediterranean)",
        "key_variants": [
            "p.IVS12+5G→A — French-Canadian founder, 80% Quebec alleles",
            "p.Gln64His (Q64H) — Mediterranean, severe",
            "p.Asp233Val — Northern European",
            "p.Pro261Leu — moderate phenotype",
            "p.Trp234Ter — null, severe",
        ],
        "seed": SEED_BASE + 3,
    },
    # ── TAT — Tyrosine Aminotransferase (Tyrosinemia type 2) ──
    {
        "gene": "TAT", "alias": "TAT — Tyrosine Aminotransferase (Tyrosinemia Type 2 / Richner-Hanhart Syndrome)",
        "aa": "454 aa", "kDa": "50.3 kDa",
        "gene_class": "transferase",
        "locus": "16q22.2", "omim_gene": 613018,
        "phenotype": "HT2 / Richner-Hanhart — plasma Tyr 500-3500 µmol/L; keratitis + palmoplantar keratoderma; NO liver disease",
        "disease": (
            "TAT biallelic loss → Tyrosinemia type 2 (HT2, Richner-Hanhart syndrome, OMIM #276600). "
            "TAT encodes the mitochondrial matrix tyrosine aminotransferase (454aa, 50.3 kDa), "
            "the FIRST enzyme in tyrosine catabolism in the liver (converts Tyr → 4-hydroxyphenylpyruvate). "
            "Deficiency → tyrosine accumulates markedly (plasma Tyr 500-3500 µmol/L; normal 40-100). "
            "HT2 is DISTINCT from HT1 (FAH): NO succinylacetone, NO liver disease, NO porphyria-like "
            "crises. Tyrosine crystals deposit extracellularly in cornea and skin. Clinical triad: "
            "(1) OCULAR — bilateral punctate dendritiform keratitis (photophobia, lacrimation, red eyes, "
            "pain) from Tyr crystal deposition; may be presenting sign in infancy; "
            "(2) CUTANEOUS — palmoplantar hyperkeratosis/keratoderma (thickening/pustules/erosions of "
            "palms and soles) from Tyr crystals in skin keratinocytes; "
            "(3) INTELLECTUAL DISABILITY — variable, up to 50% of patients; not obligate. "
            "TREATMENT: low-tyrosine + low-phenylalanine diet → ophthalmologic AND skin lesions "
            "RESOLVE COMPLETELY on dietary control. Diet is curative for end-organ manifestations. "
            "Incidence: rare, <1 in 250,000."
        ),
        "inheritance": "Autosomal recessive. TAT 16q22.2. Very rare (<1 in 250,000). Saudi/Mediterranean enrichment.",
        "hallmark": (
            "TAT HALLMARKS: (1) DENDRITIFORM KERATITIS — bilateral, punctate, pseudo-herpetic "
            "corneal lesions; Tyr crystals deposit in corneal epithelium → ulceration-like appearance; "
            "distinguished from HSV keratitis by bilateral presentation + amino acid profile; "
            "(2) PALMOPLANTAR KERATODERMA — hyperkeratosis of palms and soles; painful in severe cases; "
            "not unique but highly characteristic combined with keratitis; "
            "(3) DIET IS THE TREATMENT — low-Tyr + low-Phe diet (Phe restricted because Phe → Tyr via PAH); "
            "within weeks of dietary control, keratitis and keratoderma resolve; "
            "(4) NO NTBC — HT2 does not involve succinylacetone or 4-HPPD pathway; NTBC irrelevant; "
            "(5) NO LIVER DISEASE — critical distinction from HT1; ALT/AST normal; no AFP elevation; "
            "(6) PLASMA TYR VERY HIGH — 500-3500 µmol/L (HT1: Tyr 200-600 on NTBC, lower before NTBC); "
            "urine organic acids show elevated 4-hydroxyphenylpyruvate, 4-hydroxyphenyllactate, "
            "4-hydroxyphenylacetate (SAME metabolites as HT1 but WITHOUT succinylacetone); "
            "(7) INTELLECTUAL DISABILITY PREVENTION — early dietary control likely prevents ID; "
            "neonatal diagnosis via NBS (Tyr elevation) + treatment: key outcome driver; "
            "(8) RICHNER-HANHART SYNDROME — eponym; described 1938 (Richner) + 1947 (Hanhart)"
        ),
        "key_ddx": (
            "TAT DDx: (1) HT1 (FAH) — same Tyr elevation on NBS; DISTINGUISH by SA (urine/DBS): "
            "present in HT1, ABSENT in HT2; liver disease in HT1, absent in HT2; "
            "(2) Neonatal transient Tyr (premature/high-protein) — Tyr <400, no SA, resolves; "
            "(3) HT3 (HPD deficiency) — also no SA, no eye/skin; mild Tyr; "
            "(4) Herpetic keratitis (HSV1) — unilateral usually; corneal culture; normal plasma Tyr; "
            "(5) NTBC-treated HT1 — Tyr elevated on NTBC; SA absent while on NTBC (look at pre-NTBC SA)"
        ),
        "diet_treatment": "Low-Tyr + low-Phe diet (curative for eye and skin manifestations); NO NTBC (not indicated); ophthalmology monitoring",
        "gene_therapy_status": "None pursued (diet so effective); gene correction investigational",
        "critical_ci": (
            "CRITICAL CI: (1) Prescribing NTBC — NOT indicated in HT2; no 4-HPPD involvement; "
            "(2) Treating keratitis as herpetic without plasma amino acids — delays diagnosis, "
            "patient loses vision; (3) Not restricting Phe along with Tyr — Phe → Tyr via PAH "
            "negates Tyr restriction; (4) Stopping diet when symptoms resolve — Tyr will re-elevate "
            "and manifestations return; diet is LIFELONG"
        ),
        "nbs_marker": "Tyr elevated DBS (MS/MS) — non-specific; SA ABSENT (distinguishes from HT1); urine organic acids for 4-OH phenyl metabolites",
        "key_biomarker": "Plasma Tyr (500-3500 µmol/L); urine 4-hydroxyphenylpyruvate + 4-hydroxyphenyllactate + 4-hydroxyphenylacetate (ABSENT succinylacetone); SA ABSENT",
        "severity_spectrum": "Mild (keratitis/keratoderma without ID) → Moderate (keratitis + keratoderma + ID) — no lethal form",
        "founder_variant": "No single dominant founder; Arabic/Saudi and Italian variants described; missense predominant",
        "key_variants": [
            "p.Arg57Ter — Saudi/Arab, null severe",
            "p.Gly362Arg — Italian/Southern European",
            "p.Val455del — splice region",
            "p.Leu89Pro — moderate",
            "p.Arg181Gly — mis-splicing",
        ],
        "seed": SEED_BASE + 4,
    },
    # ── GLDC — Glycine Decarboxylase P-protein (NKH) ──
    {
        "gene": "GLDC", "alias": "GLDC — Glycine Decarboxylase P-protein (NKH / Glycine Encephalopathy)",
        "aa": "1020 aa", "kDa": "114.1 kDa",
        "gene_class": "decarboxylase",
        "locus": "9p24.1", "omim_gene": 238300,
        "phenotype": "NKH — glycine accumulation; neonatal hypotonia, apnea, hiccups (pathognomonic), burst-suppression EEG",
        "disease": (
            "GLDC biallelic loss → Non-ketotic hyperglycinemia (NKH, glycine encephalopathy, "
            "OMIM #605899). GLDC encodes the P-protein (glycine decarboxylase, 1020aa, 114.1 kDa), "
            "the largest subunit of the mitochondrial glycine cleavage system (GCS). GCS complex: "
            "P-protein (GLDC) + T-protein (AMT, aminomethyltransferase) + H-protein (GCSH) + "
            "L-protein (dihydrolipoamide dehydrogenase, DLD/shared with PDC/OGD). "
            "GLDC accounts for ~80% of NKH cases; AMT ~15%; GCSH rare. Deficiency → glycine "
            "accumulates in blood, urine, and especially CSF. CSF glycine >30 µmol/L (normal <10); "
            "CSF:plasma Gly ratio >0.08 (normal <0.03) — PATHOGNOMONIC for NKH. "
            "Clinical neonatal-onset: hypotonia (severe, generalized), apnea requiring ventilation, "
            "HICCUPS (highly characteristic — glycine-mediated GlyR activation in brainstem), "
            "myoclonic seizures refractory to standard AEDs, progressive encephalopathy. "
            "EEG: BURST-SUPPRESSION or hypsarrhythmia. NO CURATIVE TREATMENT available. "
            "Prognosis: neonatal-onset NKH = profound intellectual disability + refractory epilepsy; "
            "atypical late-onset: milder, spinocerebellar phenotype."
        ),
        "inheritance": "Autosomal recessive. GLDC 9p24.1 (80%), AMT 3p21.2 (15%), GCSH 16q23.2 (rare). Incidence 1 in 60,000.",
        "hallmark": (
            "GLDC/NKH HALLMARKS: (1) HICCUPS — intractable hiccups in neonate = NKH until proven; "
            "glycine activates inhibitory GlyR (glycine receptors) in brainstem → respiratory rhythm "
            "disruption; also causes burst-suppression EEG; "
            "(2) CSF:PLASMA GLYCINE RATIO >0.08 — pathognomonic; normal ratio <0.03; "
            "plasma Gly alone insufficient (non-specific elevation); CSF/plasma SIMULTANEOUSLY; "
            "(3) BURST-SUPPRESSION EEG — alternating bursts of spike-wave activity with suppressed "
            "background; correlates with severe neonatal NKH; evolves to hypsarrhythmia with age; "
            "(4) SODIUM BENZOATE — standard treatment; benzoate conjugates glycine → hippurate "
            "(excreted renally), reducing glycine pool; dose 250-750 mg/kg/day; reduces plasma "
            "Gly but NOT CSF Gly adequately; partial seizure benefit; "
            "(5) DEXTROMETHORPHAN (DXM) — NMDA receptor antagonist; reduces glycinergic "
            "hyperactivation of NMDA receptors; 5-10 mg/kg/day; used with benzoate; "
            "(6) GlyT1 INHIBITORS (bitopertin) — reduce CNS glycine via glycine transporter "
            "inhibition; clinical trials; "
            "(7) GLYCINE RESTRICTION DIET — ineffective for CNS (glycine synthesized endogenously); "
            "(8) VALPROATE CONTRAINDICATED — increases glycine levels; use LEV/CLB/CLN for epilepsy; "
            "(9) ATYPICAL NKH — late-onset (spinocerebellar + optic atrophy + chorea); milder course; "
            "GLDC hypomorphic or AMT; "
            "(10) CORPUS CALLOSUM AGENESIS on MRI — associated in ~50% of NKH"
        ),
        "key_ddx": (
            "GLDC DDx: (1) Ketotic hyperglycinemias (PA, MMA, IVA) — plasma Gly elevated but "
            "CSF:plasma ratio NORMAL (<0.03); high anion gap metabolic acidosis; organic acids abnormal; "
            "(2) Pyridoxine-dependent epilepsy (ALDH7A1) — neonatal seizures; trial pyridoxine; "
            "pipecolic acid + α-AASA in urine; (3) GABA-T deficiency — high GABA/β-alanine; "
            "(4) DLD deficiency — combined GCS + PDC + OGDH deficiency; elevated lactate + Gly + "
            "BCKA; (5) NKH-like drug effect (valproate, sodium benzoate overdose) — history"
        ),
        "diet_treatment": "Sodium benzoate 250-750 mg/kg/day + dextromethorphan; palliative only (NO CURE); GlyT1 inhibitors investigational; avoid VPA",
        "gene_therapy_status": "No approved gene therapy; GlyT1 inhibitors (bitopertin) in trials; prenatal diagnosis by sequencing",
        "critical_ci": (
            "CRITICAL CI: (1) Valproate (VPA) — increases glycine; CONTRAINDICATED in NKH; "
            "use LEV/CLB/clobazam/clonazepam; (2) Measuring plasma glycine alone — must measure "
            "CSF:plasma ratio simultaneously; plasma Gly elevation non-specific (PA, MMA, IVA); "
            "(3) Giving benzoate without dose titration — neonatal benzoate toxicity "
            "(encephalopathy) at high doses; monitor ammonia (benzoate metabolized via UCD pathway); "
            "(4) Not counselling palliative/hospice early — neonatal NKH has no cure; "
            "family must understand prognosis realistically"
        ),
        "nbs_marker": "Gly elevated DBS (non-specific); CSF:plasma Gly ratio >0.08 is diagnostic (NOT NBS panel item); plasma Gly on amino acids",
        "key_biomarker": "CSF glycine >30 µmol/L + CSF:plasma Gly ratio >0.08 (pathognomonic); plasma Gly >600 µmol/L; EEG burst-suppression",
        "severity_spectrum": "Neonatal (most, severe, burst-suppression, apnea) → Atypical late-onset (milder, spinocerebellar) → Transient neonatal (rare, resolves)",
        "founder_variant": "p.Ser564Ile (GLDC, Israeli Arab founder); p.Gly761Arg (Finnish); p.Arg515His (Northern European)",
        "key_variants": [
            "p.Ser564Ile — Israeli Arab, GLDC null",
            "p.Gly761Arg — Finnish founder, GLDC",
            "p.Arg515His — Northern European, severe",
            "p.Ser867Leu — AMT mild/atypical NKH",
            "p.Asn246Ser — AMT hypomorphic, late-onset",
        ],
        "seed": SEED_BASE + 5,
    },
    # ── OAT — Ornithine δ-Aminotransferase (Gyrate Atrophy) ──
    {
        "gene": "OAT", "alias": "OAT — Ornithine δ-Aminotransferase (Gyrate Atrophy of Choroid and Retina)",
        "aa": "439 aa", "kDa": "48.9 kDa",
        "gene_class": "transferase",
        "locus": "10q26.13", "omim_gene": 613349,
        "phenotype": "Gyrate atrophy — ornithine 300-1000 µmol/L; progressive chorioretinal atrophy → legal blindness by 40-50y",
        "disease": (
            "OAT biallelic loss → Gyrate atrophy of the choroid and retina (OMIM #258870). "
            "OAT encodes ornithine δ-aminotransferase (439aa, 48.9 kDa), a mitochondrial "
            "PLP-requiring enzyme that converts ornithine → glutamate-5-semialdehyde (interconverts "
            "with pyrroline-5-carboxylate, P5C) using α-ketoglutarate as amine acceptor. "
            "OAT also participates in arginine/proline metabolism and creatine biosynthesis "
            "(OAT provides arginine-derived ornithine in the urea cycle / arginine pathway). "
            "Deficiency → ornithine accumulates (plasma Orn 300-1000 µmol/L; normal 50-150). "
            "Progressive chorioretinal atrophy: Orn-mediated toxicity to RPE (retinal pigment "
            "epithelium) and choriocapillaris; tubular, ring-shaped atrophic areas beginning "
            "peripherally → converge centrally → legal blindness by 40-50 years. "
            "Creatine deficiency accompanies OAT deficiency (reduced P5C → proline → Pro pathway "
            "for creatine synthesis). Incidence: 1 in 1,000,000 globally; 1 in 50,000 in Finland "
            "due to p.Arg180Ter founder."
        ),
        "inheritance": "Autosomal recessive. OAT 10q26.13. Finnish founder p.Arg180Ter. Very rare globally (1 in 1,000,000); 1 in 50,000 Finland.",
        "hallmark": (
            "OAT HALLMARKS: (1) GYRATE ATROPHIC PATTERN — ophthalmoscopy: sharply demarcated, "
            "circular/oval choroidal atrophic patches in mid-periphery; 'gyrate' (= convoluted/round) "
            "atrophy; peripheral fields lost first → tubular vision → total blindness; "
            "(2) PLASMA ORNITHINE 300-1000 µmol/L — markedly elevated; plasma Arg, Pro, Glu, "
            "Lys all reduced (OAT interconversions); "
            "(3) ARGININE-RESTRICTED DIET — reduces ornithine production (Arg → Orn via arginase); "
            "primary dietary intervention; low-protein diet with Arg-free supplement; "
            "target plasma Orn <200 µmol/L; "
            "(4) PYRIDOXINE (B6) TRIAL — ~5-10% of OAT patients respond (B6-responsive OAT): "
            "pharmacological PLP doses (500-1000 mg/day) stabilize OAT → reduce plasma Orn; "
            "trial mandatory in all newly diagnosed; "
            "(5) CREATINE SUPPLEMENTATION — OAT involved in arginine/creatine pathway; "
            "muscle creatine deficiency in OAT; creatine supplementation (1-2 g/day) controversial "
            "but used in some centres; "
            "(6) LYSINE SUPPLEMENTATION — lysine competes with ornithine for transport; "
            "L-lysine supplementation may reduce ornithine in some patients; "
            "(7) NO APPROVED THERAPY beyond diet + B6 trial; "
            "(8) HISTOLOGICAL: OAT accumulation → RPE degeneration + choriocapillaris dropout; "
            "ERG progressively abnormal; "
            "(9) Finnish founder p.Arg180Ter: 80% of Finnish OAT alleles"
        ),
        "key_ddx": (
            "OAT DDx: (1) Retinitis pigmentosa (RP) — also progressive chorioretinal; "
            "plasma Orn NORMAL in RP; rod-cone pattern vs gyrate pattern; "
            "(2) Choroideremia (CHM) — X-linked, males predominantly; plasma Orn normal; "
            "(3) PRSS56 high myopia with retinal atrophy; (4) OTC deficiency — also high Orn "
            "in plasma (from ornithine transport defect) BUT UCD hyperammonemia; "
            "check ammonia + urine orotate; (5) HHH syndrome (SLC25A15) — hyperammonaemia + "
            "hyperornithinaemia + homocitrullinuria; UCD phenotype; Orn high from mitochondrial "
            "transport defect, not OAT"
        ),
        "diet_treatment": "Arginine-restricted diet; B6 trial (all new diagnoses); creatine supplement (controversial); annual ophthalmology ERG + visual field",
        "gene_therapy_status": "Phase 1 AAV-OAT (subretinal) trials; systemic AAV explored; no approved therapy",
        "critical_ci": (
            "CRITICAL CI: (1) Missing B6 trial — 5-10% of patients manageable with pyridoxine; "
            "(2) Arginine-free diet without protein complement — arginine-free AA formula needed "
            "to prevent arginine deficiency (Arg is semi-essential); "
            "(3) Diagnosing as RP without plasma amino acids — OAT is treatable; "
            "ophthalmology without metabolic workup is an incomplete approach; "
            "(4) Not monitoring ERG/VF annually — atrophy progression quantification guides "
            "therapy escalation"
        ),
        "nbs_marker": "Orn elevated (MS/MS DBS) — not in all NBS panels; plasma AA chromatography; urine amino acids (ornithinuria)",
        "key_biomarker": "Plasma ornithine (300-1000 µmol/L; normal 50-150); urine ornithine; fundus + ERG; OAT enzyme in fibroblasts",
        "severity_spectrum": "Progressive (all patients): peripheral atrophy → central involvement; B6-responsive (5-10%): slower progression",
        "founder_variant": "p.Arg180Ter (R180X, Finland, 80% Finnish alleles, 1 in 50,000 Finland); p.Glu318Lys (European)",
        "key_variants": [
            "p.Arg180Ter (R180X) — Finnish founder, null, 80% Finnish alleles",
            "p.Glu318Lys — Northern European",
            "p.Leu402Pro — severe null",
            "p.Ala226Val — B6-responsive",
            "p.Thr181Met — B6-responsive partial",
        ],
        "seed": SEED_BASE + 6,
    },
    # ── SLC7A7 — LPI (Lysinuric Protein Intolerance) ──
    {
        "gene": "SLC7A7", "alias": "SLC7A7 — y+LAT1 Cationic Amino Acid Transporter (LPI / Lysinuric Protein Intolerance)",
        "aa": "511 aa", "kDa": "57.4 kDa",
        "gene_class": "transporter",
        "locus": "14q11.2", "omim_gene": 222700,
        "phenotype": "LPI — post-prandial hyperammonaemia; protein aversion; PAP in 65%; urine dibasic aminoaciduria",
        "disease": (
            "SLC7A7 biallelic loss → Lysinuric Protein Intolerance (LPI, OMIM #222700). "
            "SLC7A7 encodes the y+LAT1 light chain (511aa, 57.4 kDa) of a heteromeric cationic "
            "amino acid transporter (y+L system); partners with SLC3A2 (4F2hc heavy chain) at the "
            "basolateral membrane of intestinal epithelium and kidney proximal tubule. "
            "Defect: impaired basolateral export of DIBASIC AMINO ACIDS (Lys, Arg, Orn) from "
            "intestinal enterocytes and renal tubular cells → malabsorption + urine losses of "
            "dibasic AA; plasma Lys/Arg/Orn low; urine Lys/Arg/Orn massively elevated. "
            "Pathophysiology: Arg + Orn deficiency → urea cycle dysfunction → HYPERAMMONAEMIA "
            "after protein meals → protein aversion/vomiting. "
            "Secondary urea cycle insufficiency. NOT a primary UCD — urea cycle enzymes intact. "
            "CLINICAL: vomiting/diarrhoea after protein meals (since weaning off breast milk), "
            "protein aversion, failure to thrive, hepatosplenomegaly, osteoporosis, "
            "PULMONARY ALVEOLAR PROTEINOSIS (PAP, 65% of LPI), immune dysfunction (lymphopenia, "
            "recurrent infections, haemophagocytic lymphohistiocytosis/HLH). "
            "Incidence: Finnish 1 in 60,000; Japan ~1 in 60,000; 1 in 300,000 globally."
        ),
        "inheritance": "Autosomal recessive. SLC7A7 14q11.2. Finnish founder p.Arg468Ter. 1 in 60,000 Finland/Japan; 1 in 300,000 globally.",
        "hallmark": (
            "SLC7A7/LPI HALLMARKS: (1) PROTEIN AVERSION — infant vomits protein-containing meals "
            "since weaning (breast milk OK: low protein, adequate Lys/Arg); self-selected low-protein "
            "diet; failure to thrive; "
            "(2) DIBASIC AMINOACIDURIA — massive urinary losses of lysine, arginine, ornithine "
            "(pattern: high urine Lys+Arg+Orn with LOW plasma Lys+Arg+Orn); PATHOGNOMONIC; "
            "(3) POST-PRANDIAL HYPERAMMONAEMIA — NH3 peak 1-3h after protein meal; "
            "urea cycle dysfunction from Arg + Orn deficiency; "
            "(4) CITRULLINE SUPPLEMENTATION — citrulline is a NEUTRAL amino acid absorbed via "
            "neutral transporters (not y+LAT1); intracellular citrulline → Arg + Orn via urea "
            "cycle enzymes (intact in LPI); bypasses transport defect; dose 2-8 g/day; "
            "CORNERSTONE of LPI treatment (not just diet); "
            "(5) PULMONARY ALVEOLAR PROTEINOSIS (PAP, 65%) — macrophage LPI dysfunction → "
            "surfactant accumulation in alveoli; restrictive lung disease; CT: ground-glass +  "
            "crazy-paving pattern; treated by whole-lung lavage + GM-CSF; LPI-PAP is macrophage "
            "driven (not autoimmune anti-GM-CSF as in secondary PAP); "
            "(6) HAEMOPHAGOCYTIC LYMPHOHISTIOCYTOSIS (HLH) — life-threatening immune dysregulation; "
            "LPI macrophage dysfunction; triggered by infection; treat underlying LPI + HLH protocol; "
            "(7) GLOMERULONEPHRITIS / RENAL DISEASE — immune-complex in some LPI; monitor creatinine; "
            "(8) OSTEOPOROSIS — amino acid deficiency impairs bone matrix; DEXA monitoring; "
            "(9) Finnish founder c.1402C→T (p.Arg468Ter): all Finnish LPI patients carry this allele"
        ),
        "key_ddx": (
            "SLC7A7 DDx: (1) UCD (OTC, CPS1, ASS1) — also hyperammonaemia; but plasma Arg/Orn NOT low "
            "in UCD (except Arg1); no dibasic aminoaciduria in UCD; "
            "(2) Cystinuria (SLC3A1, SLC7A9) — dibasic aminoaciduria + cystinuria; plasma AA normal; "
            "no hyperammonaemia; renal stones; "
            "(3) Protein allergy / eosinophilic GI disease — histology; no aminoaciduria; "
            "(4) PAP from anti-GM-CSF autoimmune — check anti-GM-CSF antibodies; LPI-PAP: "
            "anti-GM-CSF NEGATIVE; "
            "(5) HHH syndrome (SLC25A15) — hyperornithinaemia + hyperammonaemia + homocitrullinuria; "
            "Orn HIGH in plasma (mitochondrial transport defect, not absorptive)"
        ),
        "diet_treatment": "Protein restriction + citrulline supplementation (2-8 g/day, essential NOT optional); Lys supplement; whole-lung lavage for PAP; HLH protocol if needed",
        "gene_therapy_status": "No approved gene therapy; investigational; HSC transplant for severe HLH considered",
        "critical_ci": (
            "CRITICAL CI: (1) Protein restriction WITHOUT citrulline — diet alone insufficient; "
            "Arg/Orn depletion worsens without citrulline bypass; "
            "(2) Missing PAP in LPI — 65% have subclinical PAP on CT; lung function + HRCT mandatory "
            "at diagnosis and follow-up; "
            "(3) High-protein load (sport nutrition, Arg supplements) — triggers hyperammonaemia crisis; "
            "(4) Treating hyperammonaemia without recognizing LPI underlying cause — "
            "ammonia scavengers alone without citrulline will fail"
        ),
        "nbs_marker": "Not standard NBS; plasma Lys/Arg/Orn low with urine dibasic AA high; post-prandial NH3 elevated",
        "key_biomarker": "Urine Lys/Arg/Orn markedly elevated; plasma Lys/Arg/Orn low; plasma ammonia (post-prandial); HRCT chest (PAP ground-glass); LDH (PAP activity)",
        "severity_spectrum": "Variable: mild protein aversion only → severe HLH + PAP + immune dysfunction (life-threatening)",
        "founder_variant": "p.Arg468Ter (Finland, all Finnish LPI; c.1402C→T); p.Tyr199Cys (Italy); p.Ile232Val (Japanese enriched)",
        "key_variants": [
            "p.Arg468Ter — Finnish founder, null, ALL Finnish LPI",
            "p.Tyr199Cys — Italian enriched",
            "p.Ile232Val — Japanese enriched",
            "p.Leu334Arg — pan-ethnic severe",
            "p.Val128Ala — mild phenotype",
        ],
        "seed": SEED_BASE + 7,
    },
    # ── PHGDH — 3-Phosphoglycerate Dehydrogenase (Serine Deficiency) ──
    {
        "gene": "PHGDH", "alias": "PHGDH — 3-Phosphoglycerate Dehydrogenase (Serine Biosynthesis Defect / Neu-Laxova)",
        "aa": "533 aa", "kDa": "56.6 kDa",
        "gene_class": "dehydrogenase",
        "locus": "1p12", "omim_gene": 601815,
        "phenotype": "Serine deficiency — microcephaly, epilepsy, ID, ichthyosis; low CSF serine; Neu-Laxova (lethal) if complete loss",
        "disease": (
            "PHGDH biallelic loss → Serine deficiency syndrome (OMIM #601815) or, with complete loss, "
            "Neu-Laxova syndrome (NLS, lethal). PHGDH encodes 3-phosphoglycerate dehydrogenase "
            "(533aa, 56.6 kDa), which catalyses the first committed step of the phosphoserine pathway "
            "(de novo serine biosynthesis from 3-phosphoglycerate). Serine is conditionally essential "
            "for brain development (myelination, sphingolipid synthesis, nucleotide synthesis via "
            "one-carbon metabolism). Downstream: PSAT1 (phosphoserine aminotransferase) and PSPH "
            "(phosphoserine phosphatase) also cause serine deficiency when mutated. PHGDH → de novo "
            "Ser; impaired → CNS serine deficiency despite normal dietary Ser supply (dietary Ser "
            "poorly crosses BBB; brain serine is ~70% de novo synthesized). "
            "CLINICAL: microcephaly (congenital/progressive), refractory epilepsy (infantile spasms, "
            "myoclonic), profound intellectual disability, ichthyosis, hypogonadism. "
            "Severe (Neu-Laxova): complete PHGDH loss → fetal defect: severe microcephaly, skin "
            "congenital ichthyosis (tight, shiny skin), limb contractures, exophthalmos → "
            "incompatible with life (stillbirth or neonatal death). "
            "Incidence: rare (<1 in 500,000)."
        ),
        "inheritance": "Autosomal recessive. PHGDH 1p12. Very rare. Also PSAT1 (9q21.2) and PSPH (7p11.2) — same phenotypic spectrum.",
        "hallmark": (
            "PHGDH HALLMARKS: (1) LOW CSF SERINE — plasma Ser may be low-normal or low; "
            "CSF serine <30 µmol/L (normal 90-170) is the diagnostic hallmark; "
            "plasma serine alone insufficient (transport into CSF regulated separately); "
            "ALWAYS measure CSF amino acids in microcephaly + epilepsy; "
            "(2) L-SERINE SUPPLEMENTATION — brain-available serine; dose 400-600 mg/kg/day "
            "divided 3-4 doses daily (maximises CNS availability); "
            "seizures respond dramatically if treatment started EARLY before neuronal damage; "
            "(3) GLYCINE SUPPLEMENTATION — added to serine supplementation; glycine is a "
            "downstream metabolite (Ser → Gly via serine hydroxymethyltransferase); "
            "typical dose 200-300 mg/kg/day; "
            "(4) OUTCOME DEPENDS ON TIMING — early treatment (neonatal, ideally intrauterine "
            "via maternal supplementation) → seizure-free + better cognitive outcome; "
            "late treatment → seizure reduction but neurological damage irreversible; "
            "(5) NEU-LAXOVA SYNDROME — severe PHGDH biallelic null; prenatal features: "
            "IUGR, microcephaly, lissencephaly, ichthyosis (tight translucent skin), "
            "flexion contractures, exophthalmos; lethal at birth or stillbirth; "
            "prenatal diagnosis (cvs/amnio + sequencing) for recurrence risk; "
            "(6) MATERNAL SERINE SUPPLEMENTATION — for pregnancies of known PHGDH-affected "
            "fetuses: maternal L-serine 400-600 mg/kg/day may improve fetal serine delivery; "
            "(7) CSF SERINE monitoring guides dose adequacy; aim CSF Ser >45 µmol/L"
        ),
        "key_ddx": (
            "PHGDH DDx: (1) PSAT1 (phosphoserine aminotransferase) deficiency — identical CSF "
            "serine deficiency + microcephaly; distinguish by sequencing; (2) PSPH deficiency — "
            "rarer, similar; (3) Angelman syndrome — epilepsy + microcephaly; Gly/Ser normal; "
            "methylation testing; (4) West syndrome — infantile spasms; ACTH/vigabatrin response "
            "but serine normal; (5) Congenital disorder of glycosylation — transferrin IEF; "
            "serine normal; (6) Zika embryopathy — exposure history; viral serology; serine normal"
        ),
        "diet_treatment": "L-serine 400-600 mg/kg/day + glycine 200-300 mg/kg/day (start immediately at diagnosis); monitor CSF serine; early treatment critical",
        "gene_therapy_status": "No approved gene therapy; investigational; dietary supplementation highly effective if early",
        "critical_ci": (
            "CRITICAL CI: (1) Not measuring CSF serine — plasma Ser can be borderline normal; "
            "CSF is the diagnostic fluid; (2) Delaying serine supplementation — every week of "
            "delay = irreversible neuronal loss; (3) Not supplementing in pregnancy for known "
            "affected fetus — maternal serine can improve fetal brain serine; "
            "(4) Misdiagnosing Neu-Laxova as Trisomy 18 without sequencing — genetic counselling "
            "for recurrence risk requires molecular diagnosis"
        ),
        "nbs_marker": "Not in standard NBS; plasma Ser low-normal; CSF serine <30 µmol/L (requires LP); clinical suspicion needed",
        "key_biomarker": "CSF serine <30 µmol/L (pathognomonic; normal 90-170); plasma serine (low or borderline); CSF glycine (low); MRI microcephaly + delayed myelination",
        "severity_spectrum": "Neu-Laxova (lethal, complete loss) → Severe serine deficiency (profound ID + epilepsy) → Moderate (residual activity, epilepsy + mild ID)",
        "founder_variant": "No dominant founder worldwide; cluster in Middle East and Ashkenazi Jewish for PHGDH",
        "key_variants": [
            "p.Val425Met — hypomorphic, milder serine deficiency",
            "p.Leu364Pro — severe null, Neu-Laxova",
            "p.Glu283Lys — intermediate",
            "p.Asp359His — moderate phenotype",
            "p.Val425Glu — null, severe",
        ],
        "seed": SEED_BASE + 8,
    },
    # ── SLC6A19 — B0AT1 Neutral Amino Acid Transporter (Hartnup) ──
    {
        "gene": "SLC6A19", "alias": "SLC6A19 — B0AT1 Neutral Amino Acid Transporter (Hartnup Disease)",
        "aa": "819 aa", "kDa": "92.1 kDa",
        "gene_class": "transporter",
        "locus": "5p15.33", "omim_gene": 234500,
        "phenotype": "Hartnup — neutral aminoaciduria; intermittent photosensitive rash + cerebellar ataxia; MOSTLY ASYMPTOMATIC",
        "disease": (
            "SLC6A19 biallelic loss → Hartnup disease (OMIM #234500). SLC6A19 encodes B0AT1 "
            "(819aa, 92.1 kDa), a sodium-dependent neutral amino acid transporter expressed "
            "in intestinal brush border and kidney proximal tubule; requires collectrin "
            "(TMEM27) for membrane targeting in intestine, and angiotensin-converting enzyme "
            "(ACE) in kidney. Defect: impaired intestinal absorption + renal reabsorption of "
            "NEUTRAL AMINO ACIDS (alanine, serine, threonine, valine, leucine, isoleucine, "
            "phenylalanine, tyrosine, histidine, tryptophan) → neutral aminoaciduria + "
            "reduced intestinal uptake. Tryptophan malabsorption → nicotinamide (vitamin B3) "
            "deficiency → pellagra-like manifestations (nicotinate is synthesized from Trp). "
            "HOWEVER: intact small-peptide absorption (PepT1 transporter) provides sufficient "
            "Trp via dipeptides/tripeptides → MOST SLC6A19 patients ASYMPTOMATIC. "
            "Symptomatic phenotype (especially with low-protein diet, malnutrition, fever, "
            "antibiotics destroying gut flora): intermittent photosensitive dermatitis "
            "(pellagra-like: erythematous, scaly rash on sun-exposed skin), cerebellar "
            "ataxia, psychiatric symptoms (anxiety, emotional instability). "
            "Incidence: 1 in 30,000; named after Hartnup family in London (1956)."
        ),
        "inheritance": "Autosomal recessive. SLC6A19 5p15.33. Common (1 in 30,000). MOSTLY ASYMPTOMATIC. High penetrance of neutral aminoaciduria; low penetrance of clinical phenotype.",
        "hallmark": (
            "SLC6A19/HARTNUP HALLMARKS: (1) NEUTRAL AMINOACIDURIA — the defining biochemical "
            "phenotype; 24h urine shows massive loss of ALL neutral amino acids (Ala, Ser, Thr, "
            "Val, Leu, Ile, Phe, Tyr, His, Trp) SIMULTANEOUSLY; amino acid pattern is distinctive "
            "and diagnostic; cystine and dibasic AA (Lys, Arg, Orn) are NORMAL (transported via "
            "different system — SLC7A9/cystinuria genes); "
            "(2) PELLAGRA-LIKE RASH — photosensitive, erythematous, scaly rash on sun-exposed "
            "areas (face, dorsa of hands/feet, neck V-area); 4 Ds of pellagra (niacin deficiency): "
            "Dermatitis, Diarrhoea, Dementia, Death; Hartnup rash is intermittent (worsened by "
            "fever, sunlight, low-protein diet, antibiotic-induced gut flora loss); "
            "(3) CEREBELLAR ATAXIA — episodic in symptomatic patients; reversible with nicotinamide; "
            "(4) NICOTINAMIDE SUPPLEMENTATION — 40-200 mg/day; CURATIVE for symptomatic episodes; "
            "prophylactic in patients with recurrent symptoms; "
            "(5) HIGH-PROTEIN DIET — increases PepT1-mediated Trp delivery → compensates for "
            "B0AT1 loss; dietary protein is the main therapeutic tool; "
            "(6) MOSTLY ASYMPTOMATIC — clinical penetrance <20%; diagnostic NBS incidentaloma "
            "in many; reassure family; "
            "(7) SLC6A19 is a DRUG TARGET — inhibitors studied for metabolic syndrome / obesity "
            "(blocking intestinal Trp/Phe absorption); "
            "(8) DDx: distinguish neutral aminoaciduria from dibasic aminoaciduria (LPI, cystinuria) "
            "by amino acid pattern"
        ),
        "key_ddx": (
            "SLC6A19 DDx: (1) Pellagra (dietary niacin deficiency) — no aminoaciduria; "
            "dietary history; responds to niacin; (2) Cystinuria (SLC7A9/SLC3A1) — "
            "dibasic + cystine aminoaciduria (not neutral); renal stones; "
            "(3) LPI (SLC7A7) — dibasic AA (Lys/Arg/Orn) in urine; protein aversion; "
            "PAP; hyperammonaemia; (4) Maple syrup urine disease (BCKDHA) — branched-chain AA "
            "elevated; allo-Ile; (5) Carcinoid syndrome — Trp depletion but no neutral aminoaciduria"
        ),
        "diet_treatment": "Nicotinamide 40-200 mg/day (symptomatic/prophylactic); high-protein diet; sun protection; most patients need no treatment",
        "gene_therapy_status": "None pursued (most asymptomatic; nutritional treatment effective); B0AT1 inhibitors under study for metabolic disease (inverse indication)",
        "critical_ci": (
            "CRITICAL CI: (1) Over-treating asymptomatic patients with nicotinamide — most do not "
            "need treatment; reassurance is appropriate; (2) Diagnosing as pellagra without "
            "urine amino acids — nutritional pellagra has no aminoaciduria; Hartnup does; "
            "(3) Restricting protein (low-protein diet) — INCREASES symptom risk; high-protein "
            "preferred; (4) Antibiotic courses without extra nicotinamide — gut flora loss "
            "reduces Trp production; symptomatic flare risk"
        ),
        "nbs_marker": "Urine amino acids (neutral aminoaciduria pattern); not in standard DBS NBS panel; plasma neutral AA may be mildly low",
        "key_biomarker": "Urine neutral aminoaciduria (Ala/Ser/Thr/Val/Leu/Ile/Phe/Tyr/His/Trp ALL elevated simultaneously); plasma Trp (low if symptomatic); urine indican (indole derivatives from gut Trp fermentation)",
        "severity_spectrum": "Asymptomatic (>80%) → Episodic symptomatic (rash + ataxia, triggered by fever/low-protein/antibiotics) → Rare persistent (chronic low-protein diet)",
        "founder_variant": "p.Asp173Asn (pan-European, most common); p.Pro265Leu (European); p.Trp234Ter (null, severe)",
        "key_variants": [
            "p.Asp173Asn (D173N) — most common European allele",
            "p.Pro265Leu — European, moderate",
            "p.Arg240Gln — Asian enriched",
            "p.Trp234Ter — null, symptomatic",
            "p.Val304Ile — hypomorphic, often asymptomatic",
        ],
        "seed": SEED_BASE + 9,
    },
]


# ── Patient cohort generator ───────────────────────────────────────────────────
def _make_patients_for_gene(g):
    rng = random.Random(g["seed"])
    gene = g["gene"]
    patients = []
    n = 40
    for i in range(n):
        pid = f"AA-{gene}-{i+1:03d}"
        # Gene-specific simulation
        if gene == "PAH":
            age_dx = rng.choice(
                [rng.uniform(0, 0.1)] * 30 +   # NBS detection neonatal
                [rng.uniform(0.5, 5)] * 6 +     # late clinical (pre-NBS era)
                [rng.uniform(5, 30)] * 4        # adult/incidental
            )
            subtype = rng.choice(["classic_pku"] * 15 + ["moderate_pku"] * 10 +
                                  ["mild_pku"] * 10 + ["mild_hpa"] * 5)
            diet_controlled = rng.random() < 0.75
            bh4_responsive = subtype in ("mild_pku", "mild_hpa") and rng.random() < 0.50
            neurological = not diet_controlled and rng.random() < 0.65
            deceased = False
        elif gene == "BCKDHA":
            age_dx = rng.choice(
                [rng.uniform(0, 0.05)] * 28 +  # classic neonatal NBS/clinical
                [rng.uniform(0.2, 5)] * 8 +    # intermediate/mild
                [rng.uniform(5, 20)] * 4       # thiamine-responsive, mild
            )
            subtype = rng.choice(["classic"] * 20 + ["intermediate"] * 10 +
                                  ["mild"] * 6 + ["thiamine_responsive"] * 4)
            diet_controlled = rng.random() < 0.60
            bh4_responsive = False  # thiamine responsive tracked separately
            neurological = subtype == "classic" and rng.random() < 0.70
            deceased = subtype == "classic" and not diet_controlled and rng.random() < 0.25
        elif gene == "CBS":
            age_dx = rng.choice(
                [rng.uniform(0, 0.1)] * 12 +   # NBS Met elevation
                [rng.uniform(1, 15)] * 18 +    # ectopia lentis / thromboembolism
                [rng.uniform(15, 40)] * 10     # adult DVT/stroke
            )
            subtype = rng.choice(["b6_responsive"] * 20 + ["b6_nonresponsive"] * 20)
            diet_controlled = subtype == "b6_responsive" and rng.random() < 0.80
            bh4_responsive = False
            neurological = not diet_controlled and rng.random() < 0.50
            deceased = rng.random() < 0.08  # thromboembolism
        elif gene == "FAH":
            age_dx = rng.choice(
                [rng.uniform(0, 0.1)] * 25 +   # NBS succinylacetone
                [rng.uniform(0.1, 1)] * 10 +   # acute liver failure
                [rng.uniform(1, 5)] * 5        # chronic cirrhosis
            )
            subtype = rng.choice(["acute_infantile"] * 20 + ["subacute"] * 12 + ["chronic"] * 8)
            diet_controlled = rng.random() < 0.85  # NTBC efficacy
            bh4_responsive = False
            neurological = rng.random() < 0.20  # porphyria-like crisis
            deceased = (not diet_controlled and rng.random() < 0.35) or (
                subtype == "acute_infantile" and rng.random() < 0.12)
        elif gene == "TAT":
            age_dx = rng.choice(
                [rng.uniform(0, 2)] * 25 +    # infant keratitis + keratoderma
                [rng.uniform(2, 15)] * 12 +   # childhood
                [rng.uniform(15, 40)] * 3     # adult
            )
            subtype = rng.choice(["keratitis_keratoderma"] * 25 +
                                  ["keratitis_id"] * 10 + ["keratoderma_only"] * 5)
            diet_controlled = rng.random() < 0.80
            bh4_responsive = False
            neurological = "id" in subtype and rng.random() < 0.50
            deceased = False
        elif gene == "GLDC":
            age_dx = rng.choice(
                [rng.uniform(0, 0.05)] * 30 +  # neonatal burst-suppression
                [rng.uniform(1, 10)] * 7 +     # atypical late-onset
                [rng.uniform(0.1, 1)] * 3      # transient neonatal
            )
            subtype = rng.choice(["neonatal_classic"] * 28 + ["atypical_late"] * 7 +
                                  ["transient_neonatal"] * 5)
            diet_controlled = rng.random() < 0.15  # palliative only
            bh4_responsive = False
            neurological = subtype == "neonatal_classic" and rng.random() < 0.95
            deceased = subtype == "neonatal_classic" and rng.random() < 0.40
        elif gene == "OAT":
            age_dx = rng.choice(
                [rng.uniform(5, 20)] * 20 +    # visual symptoms, childhood
                [rng.uniform(20, 40)] * 15 +   # adult fundus incidental
                [rng.uniform(0.5, 5)] * 5      # amino acids NBS incidental
            )
            subtype = rng.choice(["progressive_standard"] * 30 + ["b6_responsive"] * 5 +
                                  ["early_severe"] * 5)
            diet_controlled = rng.random() < 0.55  # Arg restriction partial
            bh4_responsive = False  # B6 responsiveness (tracked as diet_controlled variant)
            neurological = False  # primarily retinal
            deceased = False
        elif gene == "SLC7A7":
            age_dx = rng.choice(
                [rng.uniform(0.1, 2)] * 25 +   # weaning protein aversion
                [rng.uniform(2, 15)] * 10 +    # childhood PAP / HLH
                [rng.uniform(15, 40)] * 5      # adult lung/immune
            )
            subtype = rng.choice(["protein_aversion_mild"] * 15 +
                                  ["pap_predominant"] * 15 +
                                  ["hlh_episode"] * 10)
            diet_controlled = rng.random() < 0.65
            bh4_responsive = False
            neurological = rng.random() < 0.20
            deceased = subtype == "hlh_episode" and rng.random() < 0.20
        elif gene == "PHGDH":
            age_dx = rng.choice(
                [rng.uniform(0, 0.5)] * 20 +   # neonatal severe
                [rng.uniform(0, 0)] * 10 +     # Neu-Laxova stillbirth (birth)
                [rng.uniform(0.5, 5)] * 10     # moderate, early childhood
            )
            subtype = rng.choice(["severe_microcephaly"] * 20 + ["neu_laxova"] * 8 +
                                  ["moderate"] * 12)
            diet_controlled = subtype != "neu_laxova" and rng.random() < 0.55
            bh4_responsive = False
            neurological = subtype != "neu_laxova" and rng.random() < 0.80
            deceased = (subtype == "neu_laxova") or (
                subtype == "severe_microcephaly" and rng.random() < 0.20)
        else:  # SLC6A19 — Hartnup
            age_dx = rng.choice(
                [rng.uniform(0.05, 0.1)] * 15 +  # NBS urine amino acids
                [rng.uniform(1, 15)] * 20 +       # childhood rash + ataxia
                [rng.uniform(15, 40)] * 5         # adult incidental
            )
            subtype = rng.choice(["asymptomatic"] * 25 + ["episodic_rash_ataxia"] * 12 +
                                  ["mild_rash_only"] * 3)
            diet_controlled = rng.random() < 0.90  # nicotinamide highly effective
            bh4_responsive = False
            neurological = subtype == "episodic_rash_ataxia" and rng.random() < 0.40
            deceased = False

        patients.append({
            "pid": pid,
            "gene": gene,
            "subtype": subtype,
            "age_dx_y": round(max(0.0, age_dx), 2),
            "diet_controlled": diet_controlled,
            "bh4_responsive": bh4_responsive,
            "neurological": neurological,
            "deceased": deceased,
        })
    return patients


ALL_PATIENTS = []
for _g in AA_GENES:
    _pts = _make_patients_for_gene(_g)
    _g["n_patients"] = len(_pts)
    _g["patients"] = _pts
    ALL_PATIENTS.extend(_pts)


# ─── API: get_overview ──────────────────────────────────────────────────────────
def get_overview():
    total = len(ALL_PATIENTS)
    n_diet = sum(1 for p in ALL_PATIENTS if p["diet_controlled"])
    n_bh4 = sum(1 for p in ALL_PATIENTS if p["bh4_responsive"])
    n_neuro = sum(1 for p in ALL_PATIENTS if p["neurological"])
    n_deceased = sum(1 for p in ALL_PATIENTS if p["deceased"])

    gene_summary = []
    for g in AA_GENES:
        pts = g["patients"]
        gene_summary.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "locus": g["locus"],
            "gene_class": g["gene_class"],
            "n_patients": g["n_patients"],
            "diet_treatment": g["diet_treatment"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "pct_diet": round(100 * sum(1 for p in pts if p["diet_controlled"]) / len(pts), 1),
            "pct_bh4": round(100 * sum(1 for p in pts if p["bh4_responsive"]) / len(pts), 1),
            "pct_neuro": round(100 * sum(1 for p in pts if p["neurological"]) / len(pts), 1),
            "pct_deceased": round(100 * sum(1 for p in pts if p["deceased"]) / len(pts), 1),
        })

    return {
        "atlas": "AA-Disorders-Atlas — Complete 10-Gene Amino Acid Disorders Atlas",
        "n_genes": len(AA_GENES),
        "n_patients": total,
        "seeds": [g["seed"] for g in AA_GENES],
        "genes_covered": [g["gene"] for g in AA_GENES],
        "gene_classes": {
            "phenylalanine_pathway":        ["PAH"],
            "branched_chain_catabolism":    ["BCKDHA"],
            "sulfur_amino_acid":            ["CBS"],
            "tyrosine_catabolism":          ["FAH", "TAT"],
            "glycine_cleavage":             ["GLDC"],
            "ornithine_catabolism":         ["OAT"],
            "cationic_aa_transport":        ["SLC7A7"],
            "serine_biosynthesis":          ["PHGDH"],
            "neutral_aa_transport":         ["SLC6A19"],
        },
        "aggregate_clinical": {
            "pct_diet_controlled":  round(100 * n_diet / total, 1),
            "pct_bh4_responsive":   round(100 * n_bh4 / total, 1),
            "pct_neurological":     round(100 * n_neuro / total, 1),
            "pct_deceased":         round(100 * n_deceased / total, 1),
        },
        "gene_summary": gene_summary,
        "critical_clinical_rules": [
            "PAH (PKU): BH4-loading test MANDATORY before prescribing sapropterin — only 25-50% respond; Phe restriction + LNAA for ALL; maternal Phe >360 µmol/L in pregnancy → fetal teratogenesis (Maternal PKU)",
            "BCKDHA (MSUD): Leucine is PRIMARY neurotoxin (not Ile/Val); allo-Ile is pathognomonic; liver Tx is metabolic cure but does NOT reverse neurological damage; thiamine trial ALL new patients",
            "CBS (HCU type 1): B6-responsiveness test MANDATORY (50% respond); betaine without Met monitoring → hypermethioninaemia + cerebral oedema; lens subluxation DOWNWARD (Marfan = UPWARD); MTHFR ≠ CBS",
            "FAH (Tyrosinemia type 1): Succinylacetone pathognomonic; liver biopsy CONTRAINDICATED before SA result (coagulopathy); start NTBC on clinical suspicion — do NOT wait for genetics; AFP every 6m (HCC risk persists on NTBC)",
            "TAT (Tyrosinemia type 2): NTBC NOT INDICATED — no succinylacetone, no 4-HPPD involvement; low-Tyr/Phe diet RESOLVES keratitis + keratoderma COMPLETELY; do not restrict only Tyr (Phe → Tyr via PAH)",
            "GLDC (NKH): NO CURATIVE TREATMENT; measure CSF:plasma Gly ratio (>0.08 pathognomonic) — plasma Gly alone insufficient; VPA CONTRAINDICATED (raises glycine); sodium benzoate + dextromethorphan palliative",
            "OAT (Gyrate atrophy): B6 trial ALL new diagnoses (5-10% respond); arginine-restricted diet primary; progressive chorioretinal atrophy → blindness by 40-50y; annual ERG + visual field mandatory",
            "SLC7A7 (LPI): Citrulline supplementation ESSENTIAL (not optional) — bypasses transport defect; protein restriction alone insufficient; PAP in 65% — HRCT chest at diagnosis; HLH risk = life-threatening",
            "PHGDH (Serine deficiency): Measure CSF serine NOT plasma serine alone (<30 µmol/L diagnostic); L-serine + glycine supplementation started IMMEDIATELY; timing critical (early treatment prevents neuronal damage); Neu-Laxova = lethal biallelic null",
            "SLC6A19 (Hartnup): >80% ASYMPTOMATIC — neutral aminoaciduria without pellagra; nicotinamide 40-200 mg/day for symptomatic; HIGH-protein diet NOT restriction; antibiotics → Trp loss → symptom flare → extra nicotinamide",
        ],
    }


# ─── API: get_breakdown ─────────────────────────────────────────────────────────
def get_breakdown():
    gene_rows = []
    for g in AA_GENES:
        pts = g["patients"]
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
            "diet_treatment": g["diet_treatment"],
            "gene_therapy_status": g["gene_therapy_status"],
            "critical_ci": g["critical_ci"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "key_variants": g["key_variants"],
            # Aggregate from cohort
            "pct_diet": round(100 * sum(1 for p in pts if p["diet_controlled"]) / len(pts), 1),
            "pct_bh4":  round(100 * sum(1 for p in pts if p["bh4_responsive"]) / len(pts), 1),
            "pct_neuro": round(100 * sum(1 for p in pts if p["neurological"]) / len(pts), 1),
            "pct_deceased": round(100 * sum(1 for p in pts if p["deceased"]) / len(pts), 1),
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
        })
    return {
        "genes": gene_rows,
        "total": len(AA_GENES),
        "total_patients": len(ALL_PATIENTS),
    }


# ─── API: get_definitions ─────────────────────────────────────────────────────
def get_definitions():
    return {
        "atlas": "AA-Disorders-Atlas — Complete 10-Gene Amino Acid Disorders Atlas",
        "aa_overview": {
            "full_name": "Amino Acid Disorders — inherited defects in amino acid catabolism, biosynthesis, or transport → substrate accumulation or product deficiency in blood/CSF/tissues",
            "genes_in_atlas": 10,
            "total_known_aa_disorders": "~50+ defined amino acid disorders",
            "collective_incidence": "PAH/PKU alone ~1 in 10,000; others 1 in 60,000–500,000 individually",
            "inheritance_note": "All 10 genes: autosomal recessive; no X-linked in this atlas",
            "nbs_note": "Most detected by expanded NBS (MS/MS plasma amino acids) except GLDC/NKH (requires CSF), OAT, PHGDH (CSF serine); Hartnup (urine amino acids)",
        },
        "definitions": [
            {
                "term": "Phenylketonuria (PKU) — PAH Deficiency",
                "definition": "Most common IEM in European populations (1 in 10,000). PAH deficiency → phenylalanine accumulation → brain toxicity. Classic PKU: plasma Phe >1200 µmol/L → severe ID, seizures, mousy odor untreated. NBS detected, diet-managed, and BH4-responsive (~25-50% mild/moderate). Sapropterin (Kuvan) is FDA/EMA-approved BH4 supplementation. Lifelong low-Phe diet with LNAA + amino acid formula. Maternal PKU: Phe >360 in pregnancy → fetal microcephaly/CHD regardless of fetal genotype.",
            },
            {
                "term": "Maple Syrup Urine Disease (MSUD) — BCKDHA/B/DBT Deficiency",
                "definition": "Branched-chain amino acid (Leu/Ile/Val) catabolism defect → BCKA accumulation. Maple syrup odor in urine/cerumen by day 4-5 of life. Allo-isoleucine is pathognomonic (absent in normal plasma). Leucine is the PRIMARY neurotoxin causing cerebral edema and myelinopathy. Treatment: BCAA-restricted diet; Leu-free formula in crisis; insulin IV; thiamine trial (5% respond); liver transplant is metabolic cure but does not reverse neurological damage.",
            },
            {
                "term": "Homocystinuria Type 1 (HCU) — CBS Deficiency",
                "definition": "Sulfur amino acid metabolism defect. Elevated total homocysteine (tHcy >50 µmol/L) + elevated methionine. Clinical tetrad: ectopia lentis (DOWNWARD), Marfanoid habitus, thromboembolic disease, intellectual disability. B6-responsive (50%): pyridoxine normalizes Hcy. Non-responsive: methionine restriction + betaine. MTHFR is NOT CBS — MTHFR: low Met + HHcy; CBS: HIGH Met + HHcy + lens subluxation. Thrombosis is leading cause of death — anticoagulation and surgical prophylaxis essential.",
            },
            {
                "term": "Tyrosinemia Type 1 (HT1) — FAH Deficiency",
                "definition": "Most severe amino acid disorder of the liver. Succinylacetone (SA) is pathognomonic (DBS/urine MS/MS). FAA → SA accumulation → hepatotoxicity + ALA-dehydratase inhibition (porphyria-like crisis) + DNA alkylation (HCC). NTBC (nitisinone) blocks 4-HPPD upstream → stops SA production → essentially curative if started early. Liver biopsy CONTRAINDICATED before SA result. AFP monitoring every 6 months mandatory even on NTBC (HCC risk persists, especially if NTBC started late).",
            },
            {
                "term": "Tyrosinemia Type 2 (HT2) — TAT Deficiency / Richner-Hanhart Syndrome",
                "definition": "TAT deficiency → plasma Tyr 500-3500 µmol/L. Clinical triad: bilateral keratitis (dendritiform, pseudo-herpetic), palmoplantar keratoderma, variable ID. NO liver disease, NO succinylacetone (HT2 ≠ HT1). NTBC NOT indicated. Low-Tyr + low-Phe diet RESOLVES keratitis and keratoderma completely. Dietary control is curative for end-organ manifestations. Phe must be restricted alongside Tyr (PAH converts Phe → Tyr, negating Tyr restriction alone).",
            },
            {
                "term": "Non-Ketotic Hyperglycinemia (NKH) — GLDC/AMT Deficiency",
                "definition": "Glycine cleavage system defect → CSF glycine accumulation. CSF:plasma Gly ratio >0.08 (normal <0.03) is diagnostic — NOT plasma Gly alone (ketotic hyperglycinemias from PA/MMA/IVA also elevate plasma Gly but CSF:plasma ratio NORMAL). Neonatal triad: hypotonia + apnea + hiccups + burst-suppression EEG. No curative treatment. Sodium benzoate + dextromethorphan (NMDA antagonist) = palliative standard. VPA CONTRAINDICATED (raises glycine). GLDC = 80% of NKH; AMT = 15%; GCSH rare.",
            },
            {
                "term": "Gyrate Atrophy — OAT Deficiency",
                "definition": "Progressive chorioretinal atrophy from ornithine accumulation (300-1000 µmol/L). Circular atrophic patches beginning peripherally → legal blindness by 40-50y. Treatment: arginine-restricted diet (Arg → Orn via arginase); B6 trial all patients (~5-10% responsive). No curative therapy; gene therapy (AAV-OAT subretinal) in trials. Annual ERG + visual field monitoring essential. Finnish founder p.Arg180Ter (80% Finnish alleles). OAT is also involved in creatine synthesis; creatine supplementation used in some centers.",
            },
            {
                "term": "Lysinuric Protein Intolerance (LPI) — SLC7A7 Deficiency",
                "definition": "y+LAT1 cationic AA transporter defect → impaired intestinal absorption + renal reabsorption of Lys/Arg/Orn → dibasic aminoaciduria + plasma Lys/Arg/Orn low → secondary urea cycle dysfunction → hyperammonaemia after protein meals. CITRULLINE supplementation (neutral transporter, bypasses defect) is the cornerstone of treatment — not just dietary restriction. Pulmonary alveolar proteinosis (PAP) in 65% — macrophage driven, not anti-GM-CSF antibody. HLH = life-threatening complication (immune dysregulation). Finnish founder p.Arg468Ter.",
            },
            {
                "term": "Serine Deficiency — PHGDH Deficiency",
                "definition": "Impaired de novo serine biosynthesis → CNS serine deficiency. Brain synthesizes ~70% of its serine de novo (dietary serine crosses BBB poorly). Plasma serine may be borderline; CSF serine <30 µmol/L (normal 90-170) is diagnostic — always measure CSF. Clinical: microcephaly, refractory epilepsy, ID, ichthyosis. Neu-Laxova syndrome = complete PHGDH biallelic null → lethal (stillbirth/neonatal). Treatment: L-serine 400-600 mg/kg/day + glycine. TIMING CRITICAL — early (even prenatal) treatment can prevent neurological damage. PSAT1 and PSPH cause the same spectrum.",
            },
            {
                "term": "Hartnup Disease — SLC6A19 Deficiency",
                "definition": "B0AT1 neutral amino acid transporter defect → neutral aminoaciduria (all neutral AA: Ala/Ser/Thr/Val/Leu/Ile/Phe/Tyr/His/Trp). MOSTLY ASYMPTOMATIC: PepT1 (peptide transporter) absorbs Trp via di/tripeptides, compensating for B0AT1 loss. Symptomatic (triggered by low-protein diet, fever, antibiotics): photosensitive pellagra-like rash + cerebellar ataxia from tryptophan/nicotinamide deficiency. Treatment: nicotinamide 40-200 mg/day + high-protein diet. Cystine and dibasic AA are NORMAL (cystinuria ≠ Hartnup). Over 80% of SLC6A19-deficient individuals never develop symptoms.",
            },
            {
                "term": "BH4 (Sapropterin) Responsiveness in PAH",
                "definition": "Tetrahydrobiopterin (BH4) is the essential cofactor for PAH (phenylalanine hydroxylase). Sapropterin dihydrochloride (Kuvan/Biopten) is the pharmacological form approved for BH4-responsive PAH deficiency. Mechanism: acts as pharmacological chaperone (stabilises residual PAH protein) + cofactor supplementation. RESPONSIVENESS CRITERIA: >30% reduction in plasma Phe after loading test (20 mg/kg × 24-48h). ~25-50% of PAH-deficient patients respond (predominantly mild/moderate PKU with some residual PAH activity). Non-responders: dietary management + LNAA/pegvaliase. BH4 cofactor defects (PTPS/DHPR/GCH1) cause HPA independently of PAH — must be excluded first in all NBS-detected HPA.",
            },
            {
                "term": "Succinylacetone (SA) — FAH Biomarker",
                "definition": "4,6-Dioxoheptanoic acid; formed spontaneously from fumarylacetoacetate (FAA) in FAH deficiency. SA is the PATHOGNOMONIC biomarker for HT1 (Tyrosinemia type 1); absent in HT2, HT3, and all other amino acid disorders. SA inhibits ALA dehydratase → 5-aminolevulinic acid (ALA) accumulates → porphyria-like neurological crisis. SA also alkylates DNA → hepatocellular carcinoma risk. Detection: DBS (MS/MS, NBS) or urine (GCMS). SA remains elevated in untreated HT1 and normalises rapidly on NTBC (but trace SA may persist from extrahepatic FAH expression). Rising urine SA on NTBC = non-compliance or medication error.",
            },
            {
                "term": "Citrulline Bypass Strategy — LPI",
                "definition": "In LPI (SLC7A7 deficiency), the defective y+LAT1 transporter handles cationic AA (Lys/Arg/Orn) at the basolateral membrane. Citrulline is a NEUTRAL amino acid — absorbed via neutral transporters (SLC6A19, SLC1A5) unaffected by LPI. Once inside cells, citrulline enters the urea cycle: citrulline → Arg → Orn → citrulline (cycle), thus providing intracellular Arg and Orn despite the defective basolateral export. Dose: 2-8 g/day divided with meals. This bypass strategy is ESSENTIAL in LPI — protein restriction alone does not adequately correct urea cycle function or ammonia handling. Citrulline should be used alongside — not instead of — protein restriction.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== AA-Disorders Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    print(f"Diet controlled: {ov['aggregate_clinical']['pct_diet_controlled']}%")
    print(f"Neurological: {ov['aggregate_clinical']['pct_neurological']}%")
    print(f"Deceased: {ov['aggregate_clinical']['pct_deceased']}%")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
