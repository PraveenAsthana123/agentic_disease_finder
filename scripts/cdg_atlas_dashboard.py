#!/usr/bin/env python3
"""CDG-Atlas — Complete 10-Gene Congenital Disorders of Glycosylation Atlas
PMM2 · MPI · ALG6 · DPAGT1 · SRD5A3 · PGM1 · COG7 · SLC35C1 · MGAT2 · TMEM165
400-patient aggregate cohort (10 × 40, seeds 878–887)

Congenital Disorders of Glycosylation (CDG) facts:
  - Inherited defects in glycan biosynthesis or processing; affect virtually every organ system
  - Collectively estimated ~1 in 50,000–100,000 (likely underdiagnosed; whole-exome expanding diagnosis)
  - PMM2-CDG is the most common CDG (~60% of all CDG), followed by ALG6-CDG
  - Transferrin isoelectric focusing (IEF) screens for N-glycosylation CDGs:
      Type I pattern (ER-level LLO or transfer defects): PMM2, MPI, ALG6, DPAGT1, SRD5A3
      Type II pattern (Golgi processing defects): COG7, SLC35C1, MGAT2, TMEM165
      Mixed pattern: PGM1 (mixed type I + II features on IEF)
  - MPI-CDG and PGM1-CDG are TREATABLE: mannose (MPI) and galactose (PGM1) supplementation
  - PMM2-CDG p.Phe119Leu LETHAL in homozygous — compound heterozygous only in survivors
  - SLC35C1-CDG (LAD-II): Bombay blood group (h/h) is PATHOGNOMONIC for fucosylation defect
  - PGM1-CDG: bifid uvula + cleft palate combination is PATHOGNOMONIC — look for it
  - COG7-CDG: wrinkled/loose skin at birth PATHOGNOMONIC among Golgi-CDG (most severe)
  - DPAGT1-CDG: myasthenic syndrome overlap — AChR-antibody NEGATIVE, pyridostigmine partial benefit

ATLAS SCOPE (10 landmark CDG genes):
  N-glycosylation pre-ER / LLO pathway:
    PMM2      — Phosphomannomutase 2 → CDG-Ia (16p13.2)    [MOST COMMON CDG, ~60%]
    MPI       — Mannose phosphate isomerase → CDG-Ib (15q24.1)  [TREATABLE, NO neuro]
    ALG6      — alpha-1,3-glucosyltransferase → CDG-Ic (1p31.3) [2nd most common N-CDG]
    DPAGT1    — UDP-GlcNAc-dolichol-P transferase → CDG-Ij (11q23.3)  [myasthenic overlap]
    SRD5A3    — Steroid 5-alpha-reductase type 3 (dolichol synthesis) → CDG-Iq (4q12)
  N-glycosylation processing / Golgi:
    COG7      — Component of Oligomeric Golgi complex 7 → CDG-IIe (16p12.2) [most severe Golgi-CDG]
    SLC35C1   — GDP-fucose transporter → CDG-IIc / LAD-II (11p11.2) [Bombay blood group]
    MGAT2     — GlcNAc-transferase II → CDG-IIa (14q21.3) [bleeding + dysmorphism]
    TMEM165   — Golgi Ca2+/Mn2+ transporter → CDG-IIk (4q12) [hepatopathy + coagulopathy]
  Mixed N+O glycosylation:
    PGM1      — Phosphoglucomutase 1 → CDG-It (1p31.3)    [TREATABLE galactose; bifid uvula]

CRITICAL CLINICAL RULES:
  1. PMM2 (CDG-Ia): MOST COMMON CDG (~60% of all CDGs); p.Arg141His most common European allele;
     p.Phe119Leu LETHAL in homozygous — NEVER seen as homozygote in living patients;
     cerebellar hypoplasia on MRI + ataxia + peripheral neuropathy + intellectual disability;
     coagulopathy (low factors XI + antithrombin) → bleeding AND clotting risk (paradoxical);
     inverted nipples + abnormal fat pads in infancy PATHOGNOMONIC;
     NO specific disease-modifying treatment; IEF type I pattern; NBS not in standard panels.
  2. MPI (CDG-Ib): ONLY CDG without neurological involvement — brain MRI NORMAL;
     hepatopathy + coagulopathy + protein-losing enteropathy + hypoglycaemia;
     TREATABLE: D-mannose 2g/kg/day (or 150-200mg/kg 6×/day) reverses coagulopathy and hepatopathy;
     serum transferrin IEF may be NORMAL or minimally abnormal — do erythrocyte MPI enzyme assay;
     IEF type I pattern when abnormal; do NOT miss — mannose is curative.
  3. ALG6 (CDG-Ic): 2nd most common N-glycosylation CDG; seizures + intellectual disability + hypotonia;
     coagulopathy; strabismus; no specific treatment;
     IEF type I pattern; milder than PMM2 on average but significant cognitive impairment.
  4. DPAGT1 (CDG-Ij): initiates N-glycosylation (dolichol-PP-GlcNAc, FIRST committed step);
     congenital myasthenic syndrome (CMS) overlap — fatigable weakness, AChR-antibody NEGATIVE;
     pyridostigmine (acetylcholinesterase inhibitor) gives partial benefit;
     intellectual disability + seizures + microcephaly; IEF type I pattern.
  5. SRD5A3 (CDG-Iq): SRD5A3 converts polyprenol → dolichol (ESSENTIAL for N-glycosylation initiation);
     ichthyosis-like skin (congenital ichthyosis overlap); ophthalmological (Leber-like retinal dystrophy,
     nystagmus, coloboma, cataract) — eye involvement PROMINENT;
     cerebellar ataxia + intellectual disability; Kabuki-like facial features;
     IEF type I pattern; dolichol phosphate supplementation experimental.
  6. PGM1 (CDG-It): BIFID UVULA + CLEFT PALATE — PATHOGNOMONIC combination;
     exercise intolerance + post-exercise rhabdomyolysis + elevated CK (myopathy component);
     coagulopathy (mixed type I + II IEF pattern — hepatic glycosylation defect);
     TREATABLE: D-galactose 1–1.5g/kg/day normalises transferrin glycoforms and CK;
     fasting hypoglycaemia (hepatic glycogenolysis impaired);
     distinct from GSD — bifid uvula + CDG IEF mixed pattern, NOT standard GSD profile.
  7. COG7 (CDG-IIe): MOST SEVERE Golgi-CDG; multi-organ failure common in infancy;
     wrinkled/loose skin at birth PATHOGNOMONIC among Golgi-CDG phenotypes;
     severe intellectual disability + brain malformations + hepatopathy + cardiac anomalies + jaundice;
     IEF type II pattern; very poor prognosis; most die in first years of life;
     Golgi COG complex tethers vesicles → loss → Golgi morphology disrupted → multi-glycosylation defect.
  8. SLC35C1 (CDG-IIc, LAD-II): GDP-fucose transporter deficiency → fucosylation failure →
     Bombay blood group (h/h — completely absent H-antigen, absent ABO antigens) PATHOGNOMONIC;
     Leukocyte Adhesion Deficiency type II (LAD-II): no sialyl-Lewis X on neutrophils →
     repeated severe bacterial infections + markedly elevated WBC (30,000-100,000/µL) between infections;
     intellectual disability + short stature; L-fucose supplementation (25mg/kg/day) partially corrects
     neutrophil migration but less effective on neurological/blood group phenotype;
     IEF type II pattern.
  9. MGAT2 (CDG-IIa): GlcNAc-transferase II — required for biantennary complex N-glycans;
     dysmorphic features + intellectual disability + behavioural problems + stereotypies;
     BLEEDING TENDENCY (elevated prothrombin time, low factor XI) — mimics coagulation defect;
     seizures variable; IEF type II pattern; all bi-antennary complex N-glycans absent.
  10. TMEM165 (CDG-IIk): Golgi Ca2+/Mn2+ transporter — manganese-dependent glycosyltransferases fail;
      hepatomegaly + coagulopathy + neurological variable (mild to moderate intellectual disability);
      IEF type I or mixed (variable on transferrin IEF);
      manganese supplementation experimental (small case series); hepatopathy may improve with age.

COHORT: 10 × 40 = 400 patient slots (seeds 878–887; gene-specific seeds)
"""

import random

SEED_BASE = 878

# ── All 10 CDG Genes ────────────────────────────────────────────────────────────
CDG_GENES = [
    # ── PMM2 — Phosphomannomutase 2 (CDG-Ia) ──────────────────────────────────
    {
        "gene": "PMM2", "alias": "PMM2 — Phosphomannomutase 2 (CDG-Ia, most common CDG)",
        "aa": "246 aa", "kDa": "28.4 kDa",
        "gene_class": "phosphomutase",
        "cdg_type": "Type I (LLO defect)",
        "locus": "16p13.2", "omim_gene": 601785,
        "phenotype": "CDG-Ia — cerebellar hypoplasia, ataxia, intellectual disability, coagulopathy, inverted nipples; NO specific treatment; ~60% of all CDG",
        "disease": (
            "PMM2 biallelic loss → Congenital Disorder of Glycosylation type Ia (CDG-Ia, PMM2-CDG, OMIM #212065). "
            "PMM2 encodes phosphomannomutase 2 (246aa, 28.4kDa), which converts mannose-6-phosphate → mannose-1-phosphate "
            "(rate-limiting step for GDP-mannose synthesis, which provides mannose for N-glycosylation lipid-linked "
            "oligosaccharide (LLO) assembly on the ER membrane). Deficiency → insufficient GDP-mannose → "
            "incomplete LLO → hypoglycosylated glycoproteins exported from ER (type I transferrin IEF pattern). "
            "Most common CDG worldwide (~60% of all CDG diagnoses). Phenotype: cerebellar hypoplasia "
            "(profound, on MRI), ataxia, moderate-to-severe intellectual disability, peripheral neuropathy, "
            "coagulopathy (low clotting factors XI, XII, antithrombin, protein C → paradoxical thrombotic AND "
            "bleeding risk), inverted nipples + abnormal fat distribution in infancy (PATHOGNOMONIC), "
            "strabismus, retinitis pigmentosa (variable). Hepatopathy mild. NO stroke. Incidence: ~1 in 20,000 "
            "(most common CDG). p.Arg141His is the most common European pathogenic variant (~70% alleles); "
            "p.Phe119Leu is LETHAL in homozygous — all survivors are compound heterozygous with milder second allele. "
            "Mortality: 20-30% in first years from multi-organ involvement."
        ),
        "inheritance": "Autosomal recessive. PMM2 16p13.2. Most common CDG (~60%). Pan-ethnic; enriched in Northern European populations.",
        "hallmark": (
            "PMM2/CDG-Ia HALLMARKS: (1) INVERTED NIPPLES + ABNORMAL SUBCUTANEOUS FAT PADS in infancy — "
            "PATHOGNOMONIC clinical constellation (abnormal glycosylation of adipose tissue/skin); "
            "(2) CEREBELLAR HYPOPLASIA on MRI — one of the most severe cerebellar developmental abnormalities "
            "in metabolic disease; typically non-progressive but profound; "
            "(3) PARADOXICAL COAGULOPATHY — both bleeding AND clotting risk; low factors XI, XII, antithrombin, "
            "protein C (predisposes to thrombosis) while PT/APTT may be elevated; "
            "manage clots with caution (antithrombin reduced → heparin less effective); "
            "(4) p.Phe119Leu NEVER HOMOZYGOUS in living patients — biologically lethal; "
            "if sequencing shows homozygous Phe119Leu, verify — likely error or compound het; "
            "(5) TRANSFERRIN IEF TYPE I PATTERN — increased disialo- and asialotransferrin, "
            "decreased tetrasialotransferrin; gold standard screening for PMM2 and all type I CDG; "
            "(6) NO DISEASE-MODIFYING TREATMENT — symptomatic management only; "
            "mannose supplementation does NOT work in PMM2-CDG (enzyme blocked UPSTREAM of mannose conversion); "
            "(7) GENETIC DIAGNOSIS mandatory — p.Arg141His common European allele; "
            "PMM2 enzyme assay in leukocytes or fibroblasts confirms (residual activity 0-15%); "
            "(8) LIVER DISEASE mild in most — transaminases modestly elevated; NOT primary problem; "
            "(9) HYPOGONADISM — ovarian insufficiency in females; male hypogonadism less prominent; "
            "(10) STROKE-LIKE episodes NOT a feature — distinguish from MELAS/mitochondrial"
        ),
        "key_ddx": (
            "PMM2 DDx: (1) MPI-CDG — type I IEF but NO neurological involvement; "
            "normal brain MRI; mannose-treatable — CRITICAL to distinguish; "
            "(2) ALG6-CDG — type I IEF; seizures + ID but less severe cerebellar; "
            "(3) SRD5A3-CDG — type I IEF; ichthyosis + eye involvement prominent; "
            "(4) PGM1-CDG — mixed IEF; bifid uvula + exercise intolerance — distinct pattern; "
            "(5) Cerebellar hypoplasia (other causes) — RELN, COL4A1, VLDLR; "
            "check transferrin IEF first in any unexplained cerebellar hypoplasia + coagulopathy; "
            "(6) Joubert syndrome — cerebellar hypoplasia + retinal dystrophy; "
            "molar tooth sign on MRI; ciliopathy, not glycosylation"
        ),
        "diet_treatment": "No disease-modifying treatment; symptomatic: physiotherapy, AED for seizures, vitamins (coenzyme Q10 experimental). Mannose is INEFFECTIVE in PMM2 (enzyme is UPSTREAM). Anticoagulation with caution (reduced antithrombin → heparin less effective; use LMWH with anti-Xa monitoring).",
        "gene_therapy_status": "No approved gene therapy. mRNA therapy (RESTORE-CDG) in preclinical development. PMM2 protein replacement studied in cell models. No clinical trial reached Phase 2 as of 2025.",
        "critical_ci": (
            "CRITICAL CI: (1) Mannose supplementation for PMM2-CDG — ineffective (enzyme is UPSTREAM of mannose conversion; "
            "mannose → M6P → PMM2 blocked → mannose cannot proceed); "
            "(2) Missing the p.Phe119Leu homozygous impossibility — if found, assume compound het or sequencing error; "
            "(3) Anticoagulation without anti-Xa monitoring — reduced antithrombin reduces heparin efficacy; "
            "(4) Assuming no clotting risk — paradoxical thrombosis from low protein C/antithrombin; "
            "(5) Normal transferrin IEF in infant — IEF can be falsely normal in severe galactosaemia and in neonates; repeat after 2 weeks"
        ),
        "nbs_marker": "Not in standard NBS; clinical + transferrin IEF (type I) + PMM2 enzyme assay in leukocytes/fibroblasts + gene panel/WES. Transferrin IEF may be borderline in neonates; repeat after 2-3 weeks of age.",
        "key_biomarker": "Transferrin isoelectric focusing (IEF): type I pattern (increased disialo-, asialotransferrin; decreased tetrasialo-). PMM2 enzyme activity in leukocytes or fibroblasts. Plasma coagulation studies (low factors XI, XII, antithrombin, protein C).",
        "severity_spectrum": "Classic severe (neonatal multi-organ; 20-30% early mortality) → Moderate (ataxia + intellectual disability + coagulopathy) → Mild (late-onset ataxia only; rare p.Val231Met homozygous in Scandinavian hypomorph)",
        "founder_variant": "p.Arg141His (R141H) — most common European allele (~70% European CDG-Ia); p.Phe119Leu — severe null; p.Val231Met — Scandinavian hypomorph (mild adult ataxia); p.Arg301Gln — pan-ethnic",
        "key_variants": [
            "p.Arg141His (R141H) — most common European allele, ~70% European CDG-Ia",
            "p.Phe119Leu — severe null; LETHAL in homozygous (never seen in living pt)",
            "p.Val231Met — Scandinavian mild hypomorph (residual ~30%); mild adult ataxia",
            "p.Arg301Gln — pan-ethnic moderate",
            "p.Asp65Tyr — East Asian enriched",
        ],
        "seed": SEED_BASE + 0,
    },
    # ── MPI — Mannose Phosphate Isomerase (CDG-Ib) ──────────────────────────────
    {
        "gene": "MPI", "alias": "MPI — Mannose Phosphate Isomerase (CDG-Ib, mannose-TREATABLE)",
        "aa": "432 aa", "kDa": "47.3 kDa",
        "gene_class": "isomerase",
        "cdg_type": "Type I (LLO defect)",
        "locus": "15q24.1", "omim_gene": 154550,
        "phenotype": "CDG-Ib — hepatopathy, coagulopathy, protein-losing enteropathy, hypoglycaemia; NO neurological involvement (brain MRI NORMAL); TREATABLE with mannose",
        "disease": (
            "MPI biallelic loss → Congenital Disorder of Glycosylation type Ib (CDG-Ib, MPI-CDG, OMIM #602579). "
            "MPI encodes mannose phosphate isomerase (432aa, 47.3kDa), which interconverts "
            "fructose-6-phosphate ↔ mannose-6-phosphate (M6P). Deficiency → reduced M6P → reduced GDP-mannose "
            "→ incomplete N-glycosylation LLO → type I transferrin IEF pattern. "
            "Unique among CDG: MPI-CDG has NO neurological involvement (brain MRI normal, intellect preserved). "
            "Clinical: predominantly hepatic — hepatomegaly, elevated transaminases; coagulopathy "
            "(factor XI, antithrombin low — same as PMM2 but more severe bleeding episodes); "
            "protein-losing enteropathy (diarrhoea, hypoalbuminaemia, oedema); "
            "hypoglycaemia (impaired glycogenolysis); cyclic vomiting episodes. "
            "TREATABLE: oral D-mannose (2g/kg/day in 6 divided doses, or 150-200mg/kg 6×/day) "
            "bypasses MPI block → mannose enters as mannose-6-phosphate (via hexokinase) → "
            "restores GDP-mannose → normalises glycosylation. Coagulopathy, hepatopathy, and "
            "enteropathy improve significantly with mannose supplementation. "
            "Transferrin IEF normalises on therapy. Note: serum transferrin IEF may be NORMAL or "
            "near-normal in some MPI-CDG patients — must use erythrocyte MPI enzyme assay for confirmation."
        ),
        "inheritance": "Autosomal recessive. MPI 15q24.1. Rare but treatable. Pan-ethnic.",
        "hallmark": (
            "MPI/CDG-Ib HALLMARKS: (1) ONLY CDG WITHOUT NEUROLOGICAL INVOLVEMENT — brain MRI NORMAL; "
            "cognitive development normal or near-normal; distinguishes from ALL other CDGs; "
            "(2) TREATABLE WITH MANNOSE — oral D-mannose 2g/kg/day; bypasses MPI block via hexokinase; "
            "reverses coagulopathy + hepatopathy + enteropathy; transferrin IEF normalises on therapy; "
            "(3) TRANSFERRIN IEF MAY BE NORMAL — erythrocyte MPI enzyme assay is GOLD STANDARD; "
            "do NOT exclude MPI-CDG on normal transferrin IEF if clinical picture fits; "
            "(4) SEVERE COAGULOPATHY relative to PMM2-CDG — epistaxis, easy bruising, GI bleeding; "
            "factor XI + antithrombin + protein C low; "
            "(5) PROTEIN-LOSING ENTEROPATHY — cyclic vomiting, diarrhoea, hypoalbuminaemia, oedema; "
            "albumin infusions for acute episodes; mannose treats root cause; "
            "(6) HYPOGLYCAEMIA — impaired hepatic glycogen/glucose metabolism; fasting intolerance; "
            "continuous feeds or frequent meals; mannose supplementation improves; "
            "(7) HEPATOMEGALY + ELEVATED TRANSAMINASES — not progressive to cirrhosis if treated early; "
            "(8) CYCLIC VOMITING — recurrent episodes can mimic metabolic crises; "
            "mannose supplementation reduces frequency markedly; "
            "(9) NO CEREBELLAR HYPOPLASIA, NO INVERTED NIPPLES, NO ATAXIA — key distinguishing features from PMM2"
        ),
        "key_ddx": (
            "MPI DDx: (1) PMM2-CDG — also type I IEF but HAS neurological (ataxia, cerebellar hypoplasia, ID); "
            "mannose does NOT work in PMM2; most common CDG; check MRI; "
            "(2) GSD Ia/Ib — also hepatomegaly + hypoglycaemia; "
            "BUT no coagulopathy pattern + transferrin IEF normal in GSD I; "
            "(3) Fanconi-Bickel syndrome (SLC2A2) — hepatomegaly + hypoglycaemia; "
            "Fanconi syndrome glucosuria; normal IEF; "
            "(4) Protein-losing enteropathy (other causes) — intestinal lymphangiectasia, "
            "inflammatory bowel disease; CDG screen (IEF) distinguishes; "
            "(5) FPIES / congenital diarrhoea syndromes — cyclic vomiting overlap; "
            "check transferrin IEF + MPI enzyme assay"
        ),
        "diet_treatment": "D-mannose 2g/kg/day (150-200mg/kg 6×/day) oral supplementation — CURATIVE for systemic manifestations. Maintains normoglycaemia (continuous feeds). Protein + albumin supplementation for enteropathy. Normal diet otherwise.",
        "gene_therapy_status": "Not required — mannose supplementation is effective long-term. No active gene therapy programme (condition is manageable with dietary supplementation).",
        "critical_ci": (
            "CRITICAL CI: (1) Missing MPI-CDG on normal transferrin IEF — IEF may be near-normal; "
            "use erythrocyte MPI enzyme assay for any unexplained hepatopathy + coagulopathy + enteropathy; "
            "(2) Not treating with mannose — MPI-CDG is completely treatable; "
            "undiagnosed/untreated leads to severe coagulopathy and liver disease; "
            "(3) Prescribing mannose in PMM2-CDG — does not work (wrong enzyme); "
            "(4) Missing fasting hypoglycaemia protocol — fasting dangerous; continuous feeds mandatory in infancy"
        ),
        "nbs_marker": "Not in standard NBS. Transferrin IEF type I (may be near-normal). Erythrocyte MPI enzyme assay is the diagnostic gold standard. Gene panel/WES. D-mannose therapy trial both diagnostic and therapeutic.",
        "key_biomarker": "Erythrocyte MPI enzyme activity (gold standard; transferrin IEF may be normal). Serum albumin (low in protein-losing enteropathy). Coagulation studies (factor XI, antithrombin low). Blood glucose (fasting hypoglycaemia).",
        "severity_spectrum": "All MPI-CDG is clinically significant at diagnosis (hepatopathy + coagulopathy + enteropathy); severity varies by residual MPI activity; all respond substantially to mannose; NO neurological spectrum (neuro preserved in all)",
        "founder_variant": "No single founder allele; pan-ethnic. p.Arg219Gln, p.Val231Met (NOT same as PMM2 Val231Met), various splice variants reported.",
        "key_variants": [
            "p.Arg219Gln — European moderate",
            "Various splice site variants across 15q24.1",
            "p.Ala43Thr — functionally reduced activity",
            "Null alleles rare (consistent with lethality if combined)",
            "Residual activity 5-30% in most reported patients",
        ],
        "seed": SEED_BASE + 1,
    },
    # ── ALG6 — alpha-1,3-glucosyltransferase (CDG-Ic) ──────────────────────────
    {
        "gene": "ALG6", "alias": "ALG6 — alpha-1,3-glucosyltransferase (CDG-Ic, 2nd most common N-CDG)",
        "aa": "507 aa", "kDa": "57.2 kDa",
        "gene_class": "glucosyltransferase",
        "cdg_type": "Type I (LLO defect)",
        "locus": "1p31.3", "omim_gene": 604566,
        "phenotype": "CDG-Ic — seizures, intellectual disability, hypotonia, coagulopathy, strabismus; type I IEF; no specific treatment; 2nd most common N-glycosylation CDG",
        "disease": (
            "ALG6 biallelic loss → Congenital Disorder of Glycosylation type Ic (CDG-Ic, ALG6-CDG, OMIM #603147). "
            "ALG6 encodes dolichyl-P-Glc:Man9GlcNAc2-PP-dolichol alpha-1,3-glucosyltransferase (507aa, 57.2kDa), "
            "which adds the first of three terminal glucose residues to the growing LLO on the ER membrane. "
            "Deficiency → Man9GlcNAc2-PP-dolichol accumulates (cannot add glucose) → "
            "incomplete LLO transferred to protein → hypoglycosylated glycoproteins (type I IEF pattern). "
            "Second most common N-glycosylation CDG (~5% of CDG). "
            "Phenotype: seizures (epileptic encephalopathy in some), intellectual disability (mild to moderate), "
            "hypotonia, coagulopathy, strabismus, feeding difficulties, occasional hepatopathy. "
            "Cerebellar hypoplasia mild to absent (less severe than PMM2-CDG). "
            "Inverted nipples can occur (less consistent than PMM2). "
            "NO reliable treatment; symptom management."
        ),
        "inheritance": "Autosomal recessive. ALG6 1p31.3. ~5% of CDG. Pan-ethnic.",
        "hallmark": (
            "ALG6/CDG-Ic HALLMARKS: (1) 2nd MOST COMMON N-GLYCOSYLATION CDG (~5%); "
            "milder average phenotype than PMM2-CDG but still significant intellectual disability; "
            "(2) SEIZURES prominent — often focal or multifocal; epileptic encephalopathy in severe cases; "
            "standard AED management; "
            "(3) COAGULOPATHY — similar to PMM2 (low factor XI, antithrombin); "
            "(4) STRABISMUS common — early ophthalmological referral; "
            "(5) TYPE I TRANSFERRIN IEF — identical pattern to PMM2-CDG; "
            "distinguish by PMM2 enzyme assay + ALG6 sequencing; "
            "(6) CEREBELLAR HYPOPLASIA mild or absent — key DDx from PMM2 (severe cerebellar); "
            "(7) NO specific treatment; anticonvulsants + supportive care; "
            "(8) HYPOTONIA in infancy — feeding difficulties, PEG in severe; "
            "(9) HEPATOPATHY mild, transient in most — transaminases modestly elevated in infancy"
        ),
        "key_ddx": (
            "ALG6 DDx: (1) PMM2-CDG — both type I IEF; PMM2 has SEVERE cerebellar hypoplasia + inverted nipples + "
            "peripheral neuropathy; PMM2 enzyme assay + gene sequencing distinguishes; "
            "(2) MPI-CDG — type I IEF but NO neurological involvement; normal brain MRI; "
            "(3) DPAGT1-CDG — type I IEF; myasthenic syndrome overlap; "
            "(4) ALG8-CDG — also glucosyltransferase deficiency; very similar to ALG6; "
            "severe hepatopathy more prominent in ALG8; "
            "(5) Angelman syndrome — hypotonia + seizures + ID; but NO CDG IEF abnormality; "
            "(6) Other epileptic encephalopathies (KCNQ2, SCN2A, CDKL5) — CDG screen first"
        ),
        "diet_treatment": "No disease-modifying treatment. Standard anti-epileptic management. Physiotherapy for hypotonia. Coagulation monitoring. G-tube if severe feeding difficulties.",
        "gene_therapy_status": "No approved gene therapy. ALG6 mRNA therapy in research stage. LLO-targeted therapies under study.",
        "critical_ci": (
            "CRITICAL CI: (1) Assuming type I IEF = PMM2 without PMM2 enzyme assay — 40% of type I CDGs are NOT PMM2; "
            "ALG6 is the 2nd most common; always do enzyme assay; "
            "(2) Using mannose in ALG6-CDG — ineffective (defect is at glucose addition step, not mannose supply); "
            "(3) Missing anticonvulsant need — seizures can be the dominant and most harmful feature; "
            "(4) Underestimating coagulopathy — surgical and procedural bleeding risk"
        ),
        "nbs_marker": "Not in standard NBS. Transferrin IEF type I. PMM2 enzyme assay + ALG6 gene sequencing for distinction. WES/NGS panel.",
        "key_biomarker": "Transferrin IEF type I pattern. PMM2 enzyme (normal in ALG6-CDG). ALG6 sequencing. Coagulation profile (low factor XI, antithrombin).",
        "severity_spectrum": "Severe (epileptic encephalopathy, severe ID) → Moderate (seizures + moderate ID) → Mild (near-normal cognition, late-onset ataxia only)",
        "founder_variant": "p.Ala333Val — most common pathogenic variant (European). Various missense and splice variants.",
        "key_variants": [
            "p.Ala333Val — most common, European enriched",
            "p.Tyr234Cys — moderate severity",
            "p.Gly227Glu — severe, near-null",
            "IVS7+5G>A — splice variant, moderate",
            "p.Phe304Ser — reported North African",
        ],
        "seed": SEED_BASE + 2,
    },
    # ── DPAGT1 — UDP-GlcNAc-dolichol-P GlcNAc-1-P transferase (CDG-Ij) ─────────
    {
        "gene": "DPAGT1", "alias": "DPAGT1 — UDP-GlcNAc-Dolichol-P Transferase (CDG-Ij, myasthenic CMS overlap)",
        "aa": "408 aa", "kDa": "46.6 kDa",
        "gene_class": "transferase",
        "cdg_type": "Type I (LLO initiation defect)",
        "locus": "11q23.3", "omim_gene": 191350,
        "phenotype": "CDG-Ij — intellectual disability, seizures, microcephaly, hypotonia; myasthenic syndrome overlap (AChR-Ab NEGATIVE); pyridostigmine partially beneficial",
        "disease": (
            "DPAGT1 biallelic loss → Congenital Disorder of Glycosylation type Ij (CDG-Ij, DPAGT1-CDG, OMIM #608093). "
            "DPAGT1 (dolichyl-phosphate N-acetylglucosaminephosphotransferase 1, 408aa, 46.6kDa) catalyses "
            "the FIRST committed step of N-glycosylation LLO assembly: transfers GlcNAc-1-phosphate from "
            "UDP-GlcNAc to dolichol-phosphate → dolichyl-PP-GlcNAc. Deficiency → entire LLO pathway fails "
            "at initiation → severely hypoglycosylated proteins → type I transferrin IEF. "
            "Phenotype: intellectual disability (moderate to severe), seizures (often drug-resistant), "
            "microcephaly, severe hypotonia, feeding difficulties. "
            "Overlap with congenital myasthenic syndrome (CMS): DPAGT1 is the initiating enzyme of "
            "N-glycosylation of neuromuscular junction proteins (AChR, MuSK, rapsyn) — "
            "hypoglycosylation impairs NMJ structure → fatigable weakness + ptosis + ophthalmoplegia. "
            "AChR antibodies NEGATIVE (structural NMJ defect, not autoimmune). "
            "Pyridostigmine (acetylcholinesterase inhibitor) partially improves myasthenic symptoms. "
            "Albuterol (beta-2 agonist) may also benefit NMJ."
        ),
        "inheritance": "Autosomal recessive. DPAGT1 11q23.3. Rare. Pan-ethnic.",
        "hallmark": (
            "DPAGT1/CDG-Ij HALLMARKS: (1) MYASTHENIC SYNDROME OVERLAP — fatigable ptosis, ophthalmoplegia, "
            "limb weakness; AChR antibodies NEGATIVE (seronegative myasthenia) — structural NMJ glycosylation defect; "
            "(2) PYRIDOSTIGMINE PARTIALLY EFFECTIVE — acetylcholinesterase inhibition partially compensates "
            "for impaired NMJ; do NOT withhold for seronegative myasthenia without CDG workup; "
            "(3) FIRST COMMITTED N-GLYCOSYLATION STEP defect — most severe type I CDG biochemically; "
            "(4) DRUG-RESISTANT SEIZURES — especially infantile spasms; vigabatrin + ACTH for IS; "
            "difficult to control; "
            "(5) MICROCEPHALY — congenital or postnatal; brain MRI abnormalities (simplified gyri, "
            "hypomyelination, thin corpus callosum); "
            "(6) TYPE I TRANSFERRIN IEF — same as PMM2/ALG6; distinguish by sequencing; "
            "(7) SEVERE HYPOTONIA — may require PEG + NIV/ventilatory support; "
            "(8) ALBUTEROL (BETA-2 AGONIST) — may improve NMJ transmission (VGCC upregulation); "
            "trial before pyridostigmine or in combination"
        ),
        "key_ddx": (
            "DPAGT1 DDx: (1) PMM2-CDG — type I IEF; NO myasthenic features; cerebellar hypoplasia; "
            "(2) ALG6-CDG — type I IEF; seizures; NO myasthenic CMS; "
            "(3) Seronegative autoimmune myasthenia (MuSK Ab) — AChR+MuSK Ab, not CDG; "
            "responds to plasma exchange + immunosuppression (DPAGT1 does NOT); "
            "(4) Other CMS (RAPSN, COLQ, CHRNE, GFPT1) — all seronegative; GFPT1-CMS has "
            "abnormal muscle N-glycosylation; gene panel distinguishes; "
            "(5) Rett syndrome — progressive regression; MECP2 mutation; "
            "no CDG IEF abnormality; no myasthenic features"
        ),
        "diet_treatment": "No disease-modifying treatment. Pyridostigmine 3-5mg/kg/day (start low, titrate) for myasthenic symptoms. Albuterol 0.1mg/kg/day (beta-2 agonist). Standard AED for seizures. PEG + physiotherapy + NIV as needed.",
        "gene_therapy_status": "No approved gene therapy. DPAGT1 mRNA therapy in preclinical studies. No clinical trials as of 2025.",
        "critical_ci": (
            "CRITICAL CI: (1) Starting immunosuppression for DPAGT1 myasthenia (not autoimmune — immunosuppression "
            "ineffective and harmful); use pyridostigmine/albuterol instead; "
            "(2) Plasma exchange / IVIG for seronegative myasthenic crisis in DPAGT1 — ineffective; "
            "(3) Missing CDG screen in seronegative myasthenia + intellectual disability; "
            "(4) Withholding pyridostigmine because CMS label not established — trial it in NMJ dysfunction + CDG"
        ),
        "nbs_marker": "Not in standard NBS. Transferrin IEF type I. CDG gene panel / WES for DPAGT1 sequencing. EMG/NCS + repetitive nerve stimulation (decremental response) for NMJ documentation.",
        "key_biomarker": "Transferrin IEF type I pattern. EMG decremental response to repetitive stimulation (myasthenic NMJ). Brain MRI (microcephaly, simplified gyri). AChR + MuSK Ab (negative — rules out autoimmune MG).",
        "severity_spectrum": "Severe (infantile spasms + severe ID + myasthenic crisis) → Moderate (drug-resistant seizures + moderate ID + fatigable weakness) → Mild (myasthenic-predominant + mild ID; rare with high residual activity)",
        "founder_variant": "p.Asp234Asn — reported in multiple families. Various compound heterozygous. No single dominant founder.",
        "key_variants": [
            "p.Asp234Asn — moderate functional impairment, multiple families",
            "p.Ala200Thr — severe, LLO initiation nearly abolished",
            "p.Gly328Glu — myasthenic-predominant phenotype",
            "Various compound heterozygous missense + splice",
            "p.Leu155Pro — helix-breaking, severe",
        ],
        "seed": SEED_BASE + 3,
    },
    # ── SRD5A3 — Steroid 5-alpha-Reductase Type 3 (CDG-Iq) ──────────────────────
    {
        "gene": "SRD5A3", "alias": "SRD5A3 — Steroid 5-alpha-Reductase Type 3 (CDG-Iq, dolichol synthesis)",
        "aa": "318 aa", "kDa": "36.0 kDa",
        "gene_class": "reductase",
        "cdg_type": "Type I (dolichol synthesis defect)",
        "locus": "4q12", "omim_gene": 611715,
        "phenotype": "CDG-Iq — ophthalmological (Leber-like retinal dystrophy, nystagmus, coloboma), ichthyosis-like skin, cerebellar ataxia, intellectual disability; dolichol pathway",
        "disease": (
            "SRD5A3 biallelic loss → Congenital Disorder of Glycosylation type Iq (CDG-Iq, SRD5A3-CDG, OMIM #612379). "
            "SRD5A3 (steroid 5-alpha-reductase type 3, 318aa, 36.0kDa) was classically known for steroid metabolism "
            "but is ESSENTIAL for dolichol biosynthesis: converts polyprenol → dihydropolyprenol (dolichol precursor) "
            "via the saturating double bond reduction on the alpha-omega isoprene unit. "
            "Without SRD5A3, polyprenol accumulates; dolichol (which carries the LLO for N-glycosylation) "
            "is depleted → N-glycosylation initiation fails → type I transferrin IEF. "
            "Phenotype: OPHTHALMOLOGICAL FEATURES PROMINENT — Leber congenital amaurosis-like retinal dystrophy, "
            "nystagmus, coloboma (iris, choroid), cataract, optic nerve hypoplasia — eye involvement in virtually all; "
            "ichthyosis-like skin (congenital ichthyosis overlap, lamellar pattern in some); "
            "cerebellar ataxia + intellectual disability (moderate to severe); "
            "Kabuki-like facial features (epicanthal folds, broad nose, thick lips). "
            "SKIN AND EYE combination makes it PHENOTYPICALLY DISTINCT among CDGs."
        ),
        "inheritance": "Autosomal recessive. SRD5A3 4q12. Rare. Middle Eastern, Moroccan enrichment (founder variants reported).",
        "hallmark": (
            "SRD5A3/CDG-Iq HALLMARKS: (1) OPHTHALMOLOGICAL FEATURES PROMINENT — Leber-like retinal dystrophy "
            "in virtually all patients; ERG abnormal (severely reduced rod/cone response); "
            "nystagmus, coloboma, cataract; ophthalmology referral MANDATORY in type I CDG; "
            "(2) ICHTHYOSIS-LIKE SKIN — congenital ichthyosis component (lamellar, palmoplantar keratoderma); "
            "skin + eye + CDG = SRD5A3 until proven otherwise; "
            "(3) DOLICHOL PATHWAY — not mannose supply or LLO assembly enzyme; "
            "dolichol-phosphate supplementation theoretically logical but unproven in humans; "
            "(4) TYPE I TRANSFERRIN IEF — same pattern as other type I CDG; "
            "(5) KABUKI-LIKE FACE — broad nasal tip, epicanthal folds, thin vermilion; "
            "KMT2D mutation (true Kabuki) has no CDG IEF abnormality; "
            "(6) CEREBELLAR ATAXIA + INTELLECTUAL DISABILITY — moderate to severe; "
            "cerebellar hypoplasia on MRI; "
            "(7) NO neurological involvement in skin-only polyprenol reductase defects — "
            "if only skin without neuro, consider other ichthyosis causes (ALOX12B, CYP4F22)"
        ),
        "key_ddx": (
            "SRD5A3 DDx: (1) Leber congenital amaurosis (LCA — GUCY2D, RPE65, AIPL1 etc.) — "
            "retinal dystrophy but NO ichthyosis, NO CDG IEF abnormality; "
            "(2) PMM2-CDG — type I IEF; cerebellar hypoplasia; NO ichthyosis/ophthalmological predominance; "
            "(3) Kabuki syndrome (KMT2D, KDM6A) — Kabuki face but NO CDG IEF abnormality, no ichthyosis-retina; "
            "(4) ALDH3A2 (Sjögren-Larsson syndrome) — ichthyosis + neuro (spasticity); "
            "no CDG IEF; urine fatty aldehyde; "
            "(5) Refsum disease (PHYH/PEX7) — ichthyosis + ataxia + neuropathy; "
            "plasma phytanic acid elevated; no CDG IEF; "
            "(6) Cohen syndrome (VPS13B) — intellectual disability + facial; no CDG IEF"
        ),
        "diet_treatment": "No established treatment. Dolichol-phosphate supplementation experimental (no robust human data). Skin: emollients + keratolytics for ichthyosis. Eye: low-vision aids; coloboma surgery if indicated. Standard AED.",
        "gene_therapy_status": "No approved therapy. Dolichol phosphate oral supplementation studied in cell/mouse models; not proven in humans. AAV-SRD5A3 preclinical.",
        "critical_ci": (
            "CRITICAL CI: (1) Diagnosing LCA without CDG screen — check transferrin IEF in retinal dystrophy + "
            "ichthyosis + ID (triple combination strongly suggests SRD5A3-CDG); "
            "(2) Steroid 5-alpha-reductase inhibitors (finasteride) — SRD5A3 is the TYPE 3 isozyme; "
            "finasteride (type 2 inhibitor) not relevant; do not confuse isoenzymes; "
            "(3) Missing the dolichol pathway concept — mannose supplementation is ineffective (defect is dolichol, not mannose); "
            "(4) Missing ophthalmological examination — retinal dystrophy is universal and progressive"
        ),
        "nbs_marker": "Not in standard NBS. Transferrin IEF type I. Plasma dolichol quantification (elevated polyprenol). Electroretinography (ERG — severely reduced). SRD5A3 gene sequencing / WES.",
        "key_biomarker": "Transferrin IEF type I. Plasma polyprenol elevation (dolichol reduced). ERG (reduced/extinguished rod-cone responses). Urinary dolichol analysis. Brain MRI (cerebellar hypoplasia).",
        "severity_spectrum": "Severe (congenital blindness + profound ID + ichthyosis) → Moderate (retinal dystrophy + moderate ID + ataxia) → Mild (retinal-predominant; rare high residual activity)",
        "founder_variant": "p.Arg 101Trp — Moroccan/North African founder. p.Ile134Asn — Middle Eastern. Various compound heterozygous.",
        "key_variants": [
            "p.Arg101Trp — North African/Moroccan founder",
            "p.Ile134Asn — Middle Eastern enriched",
            "p.Gly175Arg — severe",
            "p.Leu119Pro — helix-breaking, severe",
            "p.Gly234Val — retinal-predominant moderate",
        ],
        "seed": SEED_BASE + 4,
    },
    # ── PGM1 — Phosphoglucomutase 1 (CDG-It) ────────────────────────────────────
    {
        "gene": "PGM1", "alias": "PGM1 — Phosphoglucomutase 1 (CDG-It, TREATABLE galactose; bifid uvula)",
        "aa": "562 aa", "kDa": "61.5 kDa",
        "gene_class": "phosphomutase",
        "cdg_type": "Mixed (Type I + II features)",
        "locus": "1p31.3", "omim_gene": 171900,
        "phenotype": "CDG-It — BIFID UVULA + cleft palate (PATHOGNOMONIC), exercise intolerance + rhabdomyolysis, coagulopathy, hepatopathy; TREATABLE with galactose",
        "disease": (
            "PGM1 biallelic loss → Congenital Disorder of Glycosylation type It (CDG-It, PGM1-CDG, OMIM #614921). "
            "PGM1 (phosphoglucomutase 1, 562aa, 61.5kDa) interconverts glucose-1-phosphate ↔ glucose-6-phosphate "
            "and glucose-1,6-bisphosphate (central glycolytic/glycogenic hub). "
            "In the CDG context: PGM1 also produces UDP-glucose and UDP-galactose "
            "from glucose-1-phosphate → deficiency → impaired N-glycosylation (mixed type I+II IEF pattern) "
            "AND impaired O-glycosylation AND impaired glycogen synthesis. "
            "BIFID UVULA + CLEFT PALATE: PATHOGNOMONIC combination; structural defect present at birth "
            "(cleft of the secondary palate); bifid uvula in many without full cleft. "
            "EXERCISE INTOLERANCE + RHABDOMYOLYSIS: PGM1 is expressed in muscle; "
            "G1P ↔ G6P conversion impaired → glycolytic flux reduced during exercise → myalgia, "
            "CK elevation (often >10× ULN after exercise), myoglobinuria in severe episodes. "
            "COAGULOPATHY: hepatic N-glycosylation impaired → low clotting factors. "
            "TREATABLE: oral D-galactose 1-1.5g/kg/day → Gal1P → UDPGal → restores glycosylation; "
            "coagulopathy and rhabdomyolysis frequency markedly reduce; transferrin IEF normalises."
        ),
        "inheritance": "Autosomal recessive. PGM1 1p31.3. Pan-ethnic; enriched in some European populations.",
        "hallmark": (
            "PGM1/CDG-It HALLMARKS: (1) BIFID UVULA + CLEFT PALATE — PATHOGNOMONIC; "
            "examine hard/soft palate and uvula in any unexplained rhabdomyolysis + coagulopathy; "
            "bifid uvula alone (without full cleft) occurs; midline palatal defect in virtually all severe; "
            "(2) TREATABLE WITH GALACTOSE — D-galactose 1-1.5g/kg/day; "
            "reverses coagulopathy, normalises transferrin IEF, reduces rhabdomyolysis; "
            "start promptly after diagnosis; "
            "(3) EXERCISE INTOLERANCE + RHABDOMYOLYSIS — CK >10× ULN post-exercise; "
            "acute renal failure risk from myoglobinuria; avoid intense exercise; adequate hydration; "
            "(4) MIXED TRANSFERRIN IEF PATTERN — type I + type II features (biphasic or mixed on IEF); "
            "distinguishes from pure type I CDG (PMM2, ALG6); "
            "(5) HEPATOPATHY — elevated transaminases; hepatomegaly; coagulopathy from liver; "
            "(6) HYPOGLYCAEMIA — fasting intolerance from impaired glycogenolysis (G1P-G6P conversion); "
            "(7) DILATED CARDIOMYOPATHY — reported in subset; cardiac monitoring; "
            "(8) SHORT STATURE + INTELLECTUAL DISABILITY — variable; mild ID in most; "
            "(9) GALACTOSE DIFFERENTIATES FROM GSD — no GSD has bifid uvula + CDG IEF pattern; "
            "misdiagnosed as GSD V/VII (exercise rhabdomyolysis) or coagulation defect"
        ),
        "key_ddx": (
            "PGM1 DDx: (1) GSD V (McArdle/PYGM) — exercise rhabdomyolysis + CK elevation; "
            "BUT no bifid uvula, NO CDG IEF, forearm ischemic test (lactate fails); "
            "(2) GSD VII (Tarui/PFKM) — exercise intolerance + hemolytic anemia; "
            "no bifid uvula, no CDG IEF; "
            "(3) PMM2-CDG — type I IEF; no bifid uvula, no exercise rhabdomyolysis; "
            "(4) MPI-CDG — type I IEF; no bifid uvula, no exercise component; no neurological; "
            "(5) Coagulation factor deficiencies — normal transferrin IEF; no bifid uvula; no rhabdomyolysis; "
            "(6) VLCAD deficiency — exercise myopathy + rhabdomyolysis; acylcarnitine panel; no CDG IEF; "
            "no bifid uvula"
        ),
        "diet_treatment": "D-galactose 1-1.5g/kg/day oral (in divided doses); reverses coagulopathy and normalises IEF; reduces rhabdomyolysis frequency. Fasting avoidance (frequent meals, uncooked cornstarch at night). Hydration during exercise. Avoid intense anaerobic exercise.",
        "gene_therapy_status": "Not currently developed — condition is manageable with galactose supplementation. No clinical trials as of 2025.",
        "critical_ci": (
            "CRITICAL CI: (1) Missing bifid uvula on exam — examine palate in any unexplained CK elevation + coagulopathy; "
            "(2) Diagnosing GSD V or VII in rhabdomyolysis without CDG screen — check transferrin IEF; "
            "(3) Not treating with galactose — coagulopathy and hepatopathy preventable; "
            "(4) Intense exercise without prior diagnosis — myoglobinuria → acute renal failure; "
            "(5) Using mannose instead of galactose — wrong sugar for PGM1 pathway; galactose only"
        ),
        "nbs_marker": "Not in standard NBS. Mixed transferrin IEF (type I + II). PGM1 enzyme assay. Bifid uvula/cleft palate on clinical exam. CK (elevated baseline and post-exercise). Gene panel/WES.",
        "key_biomarker": "Transferrin IEF mixed (type I + type II). CK (elevated, especially post-exercise). PT/APTT + factor XI (low). PGM1 enzyme in erythrocytes. D-galactose challenge (normalises IEF — both diagnostic and therapeutic).",
        "severity_spectrum": "Severe (neonatal hepatopathy + cleft palate + significant ID) → Moderate (exercise rhabdomyolysis + coagulopathy + mild ID) → Mild (exercise intolerance only; bifid uvula; near-normal IEF with high residual PGM1)",
        "founder_variant": "p.Gln66Ter, p.Pro117Leu, p.Arg493Trp — European. Various compound heterozygous.",
        "key_variants": [
            "p.Gln66Ter — European null, severe",
            "p.Pro117Leu — moderate, exercise-predominant",
            "p.Arg493Trp — European, mixed severity",
            "p.Asp418His — hepatic-predominant",
            "IVS17-2A>G — splice defect, variable",
        ],
        "seed": SEED_BASE + 5,
    },
    # ── COG7 — Component of Oligomeric Golgi Complex 7 (CDG-IIe) ────────────────
    {
        "gene": "COG7", "alias": "COG7 — Component of Oligomeric Golgi Complex 7 (CDG-IIe, most severe Golgi-CDG)",
        "aa": "828 aa", "kDa": "93.2 kDa",
        "gene_class": "golgi_tethering",
        "cdg_type": "Type II (Golgi processing defect)",
        "locus": "16p12.2", "omim_gene": 606978,
        "phenotype": "CDG-IIe — wrinkled/loose skin at birth (PATHOGNOMONIC), severe multi-organ (cardiac, hepatic, brain), type II IEF; very poor prognosis",
        "disease": (
            "COG7 biallelic loss → Congenital Disorder of Glycosylation type IIe (CDG-IIe, COG7-CDG, OMIM #608779). "
            "COG7 (component of oligomeric Golgi complex 7, 828aa, 93.2kDa) is a subunit of the COG complex "
            "(8-subunit COG1-8), which tethers retrograde vesicles in the Golgi to maintain Golgi stack "
            "structure and return glycosyltransferases/glycosidases to appropriate Golgi cisternae. "
            "COG7 loss → intra-Golgi trafficking disrupted → mislocalized or degraded Golgi enzymes → "
            "defective N- and O-glycan processing → type II transferrin IEF pattern. "
            "WRINKLED/LOOSE SKIN at birth is PATHOGNOMONIC among Golgi-CDG phenotypes — "
            "excessive, wrinkled skin resembling a prematurely aged infant (cutis laxa-like). "
            "Severe multi-organ involvement: facial dysmorphism (short nose, small chin, large ears), "
            "cardiac anomalies (ASD, VSD, PDA), hepatosplenomegaly + cholestasis + coagulopathy, "
            "brain malformations (pachygyria, hypomyelination), severe intellectual disability. "
            "VERY POOR PROGNOSIS: most patients with COG7-CDG die in the first 1-2 years of life. "
            "COG7-CDG is the most severe and best-described COG-subunit CDG."
        ),
        "inheritance": "Autosomal recessive. COG7 16p12.2. Very rare. Enriched in consanguineous populations.",
        "hallmark": (
            "COG7/CDG-IIe HALLMARKS: (1) WRINKLED/LOOSE SKIN AT BIRTH — PATHOGNOMONIC among Golgi-CDG; "
            "cutis laxa-like redundant skin without true elastin defect; CDG Golgi phenotype marker; "
            "search for CDG IEF type II in any wrinkled-skin neonate; "
            "(2) MOST SEVERE COG-SUBUNIT CDG — death in infancy in majority; "
            "(3) MULTI-ORGAN FAILURE — cardiac + hepatic + neurological simultaneously; "
            "distinguish from isolated congenital heart disease or isolated liver disease; "
            "(4) TYPE II TRANSFERRIN IEF — defective Golgi sialylation and galactosylation; "
            "(5) COG COMPLEX — 8 subunits; COG1-4 (lobe A) + COG5-8 (lobe B); "
            "COG7 in lobe B; lobe B defects tend to be most severe; "
            "(6) JAUNDICE + CHOLESTASIS in neonatal period — hepatic CDG; "
            "(7) CARDIAC MONITORING mandatory — structural heart anomalies common; "
            "(8) BRAIN MRI ABNORMALITIES — pachygyria, simplified gyri, hypomyelination; "
            "(9) NO EFFECTIVE TREATMENT — supportive only; liver/cardiac surgery as needed"
        ),
        "key_ddx": (
            "COG7 DDx: (1) Cutis laxa (ELN, FBLN5, LTBP4) — loose skin; "
            "but NO CDG IEF type II, no multi-organ; elastin defects; "
            "(2) Other COG-CDGs (COG1, COG4, COG5, COG8) — all type II IEF; "
            "COG5 and COG8 generally milder than COG7; distinguish by sequencing; "
            "(3) SLC35C1 (LAD-II) — type II IEF; Bombay blood group; no wrinkled skin; "
            "(4) Neonatal sepsis — jaundice + multi-organ; but CDG IEF abnormal; "
            "(5) Mitochondrial hepatopathy (DGUOK, POLG) — liver + brain; "
            "respiratory chain enzymes abnormal; normal CDG IEF; "
            "(6) CHARGE syndrome — multi-organ anomalies; CHD7 mutation; normal CDG IEF; no wrinkled skin"
        ),
        "diet_treatment": "No specific treatment. Supportive: cardiac surgery (if operable), ursodeoxycholic acid for cholestasis, clotting factor replacement, seizure management, NIV/O2 for respiratory failure.",
        "gene_therapy_status": "No approved or advanced gene therapy. Golgi-CDG research is less advanced than ER-CDG. COG7 AAV experimental only.",
        "critical_ci": (
            "CRITICAL CI: (1) Missing COG7-CDG in wrinkled-skin neonate — do CDG screen (IEF) in ALL wrinkled-skin/cutis-laxa neonates; "
            "(2) Attributing cardiac anomalies to isolated CHD without metabolic workup in dysmorphic infant; "
            "(3) Attributing jaundice to biliary atresia without CDG screen; "
            "(4) Prognosis counselling without confirming COG7 — lobe B COG defects are most severe; "
            "do rapid sequencing for prognosis discussion"
        ),
        "nbs_marker": "Not in standard NBS. Transferrin IEF type II. COG7 sequencing / WES. Skin biopsy (fibroblasts for Golgi morphology by immunofluorescence). Cardiac ECHO + brain MRI in all.",
        "key_biomarker": "Transferrin IEF type II. Golgi morphology by immunofluorescence (fragmented/dispersed Golgi in fibroblasts). Direct COG7 sequencing. Coagulation profile. Cardiac ECHO. Brain MRI.",
        "severity_spectrum": "Most patients: severe neonatal multi-organ failure → death in infancy. Rarely: prolonged survival >2y with major disability. No mild phenotype described for COG7.",
        "founder_variant": "p.Gly549Asp — Nigerian/West African consanguineous families. Various biallelic LOF variants.",
        "key_variants": [
            "p.Gly549Asp — Nigerian/West African founder",
            "Deletion exon 1-3 — complete LOF, severe",
            "p.Arg71Gln — European case reports",
            "p.Arg428Ter — truncating, severe",
            "Various homozygous in consanguineous families",
        ],
        "seed": SEED_BASE + 6,
    },
    # ── SLC35C1 — GDP-Fucose Transporter (CDG-IIc / LAD-II) ─────────────────────
    {
        "gene": "SLC35C1", "alias": "SLC35C1 — GDP-Fucose Transporter (CDG-IIc / LAD-II, Bombay blood group)",
        "aa": "364 aa", "kDa": "40.0 kDa",
        "gene_class": "transporter",
        "cdg_type": "Type II (Golgi fucosylation defect)",
        "locus": "11p11.2", "omim_gene": 605881,
        "phenotype": "CDG-IIc (LAD-II) — Bombay blood group (h/h, PATHOGNOMONIC), recurrent severe infections, elevated WBC, short stature, intellectual disability; fucose supplementation partial benefit",
        "disease": (
            "SLC35C1 biallelic loss → Congenital Disorder of Glycosylation type IIc (CDG-IIc, LAD-II, OMIM #266265). "
            "SLC35C1 (GDP-fucose transporter 1, 364aa, 40.0kDa) transports GDP-fucose from cytoplasm into Golgi lumen "
            "for fucosylation reactions catalysed by fucosyltransferases (FUT1-9). "
            "Deficiency → no Golgi fucosylation → absence of: "
            "(1) H antigen + ABO antigens on RBCs (→ Bombay/Oh blood group, h/h — PATHOGNOMONIC); "
            "(2) sialyl-Lewis X (CD15s) on neutrophils (required for neutrophil rolling via selectins → "
            "Leukocyte Adhesion Deficiency type II / LADII): recurrent bacterial infections, markedly elevated "
            "WBC (30,000-150,000/µL between infections), delayed umbilical cord separation, poor wound healing; "
            "(3) Lewis blood group antigens. "
            "Intellectual disability (moderate) + short stature + facial dysmorphism. "
            "BOMBAY BLOOD GROUP: anti-H antibodies in all patients → incompatible with essentially all standard blood types; "
            "requires H-antigen negative (Bombay) donor blood — CRITICAL for transfusion. "
            "L-fucose supplementation (25mg/kg/day) partially corrects neutrophil fucosylation → "
            "some improvement in infection frequency; H antigen/ABO may partially appear on treatment."
        ),
        "inheritance": "Autosomal recessive. SLC35C1 11p11.2. Very rare. Middle Eastern founder.",
        "hallmark": (
            "SLC35C1/CDG-IIc/LAD-II HALLMARKS: (1) BOMBAY BLOOD GROUP (h/h) — PATHOGNOMONIC; "
            "absent H antigen by blood typing → cannot receive standard O-negative or any ABO blood; "
            "requires Bombay (Oh) donor; ALWAYS check blood group in CDG-IIc patients before any procedure; "
            "(2) LEUKOCYTE ADHESION DEFICIENCY TYPE II — no sialyl-Lewis X (CD15s) on neutrophils; "
            "flow cytometry: absent CD15s; impaired neutrophil rolling → recurrent severe bacterial infections "
            "despite markedly elevated WBC (paradoxical leukocytosis during infections AND between infections); "
            "(3) DELAYED UMBILICAL CORD SEPARATION (>3 weeks) — LAD feature; "
            "(4) L-FUCOSE SUPPLEMENTATION — 25mg/kg/day; partially restores neutrophil migration; "
            "blood group may partially correct; cognitive benefit minimal; "
            "(5) TYPE II TRANSFERRIN IEF — fucose is on N-glycans in serum; "
            "(6) SHORT STATURE + INTELLECTUAL DISABILITY — moderate; "
            "(7) NOT TYPE I IEF — Golgi-level defect → type II pattern; "
            "(8) BLOOD BANK ALERT — patient must have Bombay blood group on file; "
            "Bombay donors extremely rare (<1 in 250,000); "
            "(9) DISTINGUISH FROM LAD-I (ITGB2/CD18 deficiency) — same LAD phenotype "
            "but LAD-I has normal blood group, absent CD18 on flow; gene panel distinguishes"
        ),
        "key_ddx": (
            "SLC35C1 DDx: (1) LAD-I (ITGB2/CD18 deficiency) — same recurrent infections + leukocytosis; "
            "BUT NORMAL blood group (NOT Bombay); absent CD18 on flow cytometry; AR; "
            "(2) LAD-III (FERMT3/kindlin-3) — LAD phenotype + bleeding; normal blood group; no CDG IEF; "
            "(3) Bombay blood group from other causes — HPRT, FUT1 polymorphisms (not CDG); "
            "normal CDG IEF in non-CDG Bombay; "
            "(4) Other Golgi-CDG (COG7) — type II IEF; no Bombay blood group; wrinkled skin; "
            "(5) Neutrophil function disorders (CGD/CYBB, Chediak-Higashi) — recurrent infections; "
            "normal blood group; respiratory burst test distinguishes CGD; "
            "CDG IEF normal in CGD"
        ),
        "diet_treatment": "L-fucose supplementation 25mg/kg/day (divided doses); partially restores neutrophil sialyl-Lewis X. Prophylactic antibiotics (trimethoprim-sulfamethoxazole). Granulocyte transfusions for life-threatening infections. Blood transfusion: BOMBAY-BLOOD ONLY (Oh donors).",
        "gene_therapy_status": "No approved gene therapy. SLC35C1 HSCT studied (mixed results). L-fucose supplementation is current standard. CRISPR/AAV for SLC35C1 in preclinical.",
        "critical_ci": (
            "CRITICAL CI: (1) Giving standard blood transfusion in CDG-IIc — Bombay (h/h) patients will have "
            "fatal haemolytic transfusion reaction with any standard ABO or O-negative blood; "
            "BOMBAY DONORS ONLY; register patient in blood bank immediately; "
            "(2) Missing LAD-II in recurrent infections with leukocytosis — check blood group + CD15s; "
            "(3) Not starting prophylactic antibiotics — recurrent severe infection mortality; "
            "(4) Assuming L-fucose corrects neurological outcome — cognitive improvement minimal on fucose; "
            "(5) Mixing up LAD-I vs LAD-II — different gene, different flow cytometry marker (CD18 vs CD15s)"
        ),
        "nbs_marker": "Not in standard NBS. Transferrin IEF type II (fucosylated glycans absent). Blood group typing (Bombay/h/h). Flow cytometry: CD15s absent on neutrophils; CD18 present (not LAD-I). SLC35C1 sequencing.",
        "key_biomarker": "Transferrin IEF type II. ABO/H blood group (Bombay — absent H/A/B). Flow cytometry CD15s (absent) + CD18 (present). WBC differential (marked leukocytosis). SLC35C1 gene sequencing.",
        "severity_spectrum": "Severe (neonatal LAD + Bombay + significant infections) → Moderate (recurrent infections + Bombay + moderate ID) → Mild (minimal infections on fucose; Bombay without LAD phenotype — rare high-residual)",
        "founder_variant": "p.Arg147Cys — Middle Eastern (Arab) founder. p.Thr308Arg — European. Various biallelic.",
        "key_variants": [
            "p.Arg147Cys — Middle Eastern/Arab founder (most common CDG-IIc allele globally)",
            "p.Thr308Arg — European",
            "p.Arg38Cys — severe LAD phenotype",
            "p.Gly167Glu — reduced transporter stability",
            "Homozygous variants in consanguineous families",
        ],
        "seed": SEED_BASE + 7,
    },
    # ── MGAT2 — GlcNAc-Transferase II (CDG-IIa) ─────────────────────────────────
    {
        "gene": "MGAT2", "alias": "MGAT2 — N-Acetylglucosaminyltransferase II (CDG-IIa, bleeding + dysmorphism)",
        "aa": "447 aa", "kDa": "50.2 kDa",
        "gene_class": "transferase",
        "cdg_type": "Type II (Golgi processing defect)",
        "locus": "14q21.3", "omim_gene": 602616,
        "phenotype": "CDG-IIa — intellectual disability, facial dysmorphism, behavioural problems, bleeding tendency, seizures; all biantennary complex N-glycans absent; type II IEF",
        "disease": (
            "MGAT2 biallelic loss → Congenital Disorder of Glycosylation type IIa (CDG-IIa, MGAT2-CDG, OMIM #212066). "
            "MGAT2 (UDP-GlcNAc:alpha-6-D-mannosyl-glycoprotein beta-1,2-N-acetylglucosaminyltransferase II, "
            "447aa, 50.2kDa) is a cis-Golgi enzyme that adds GlcNAc to the alpha-6-Man branch of the "
            "N-glycan intermediate, forming the second branch of biantennary complex N-glycans. "
            "Deficiency → ALL biantennary (complex) N-glycans are absent (only high-mannose and "
            "hybrid glycans remain) → type II transferrin IEF pattern. "
            "Phenotype: intellectual disability (moderate to severe), facial dysmorphism "
            "(coarse features, widely spaced teeth, flat nasal bridge, large ears), "
            "behavioural problems (stereotypies, autistic-like features), "
            "seizures (focal/multifocal, drug-responsive in most), "
            "BLEEDING TENDENCY (elevated PT/APTT, low factor XI — glycosylation of coagulation factors impaired). "
            "White matter abnormalities on MRI in some. Hepatomegaly variable. "
            "Phenotype less severe than COG7-CDG; most patients survive to adulthood."
        ),
        "inheritance": "Autosomal recessive. MGAT2 14q21.3. Very rare. Pan-ethnic.",
        "hallmark": (
            "MGAT2/CDG-IIa HALLMARKS: (1) ALL BIANTENNARY COMPLEX N-GLYCANS ABSENT — "
            "glycan mass spectrometry shows high-mannose + hybrid only; no complex glycoforms; "
            "used to confirm diagnosis in fibroblasts; "
            "(2) BLEEDING TENDENCY — elevated PT/APTT; low factor XI; "
            "surgical bleeding risk; manage with factor replacement/FFP; "
            "(3) TYPE II TRANSFERRIN IEF — Golgi N-glycan processing defect; "
            "(4) BEHAVIOURAL PROBLEMS + STEREOTYPIES — autistic-like; "
            "may be misdiagnosed as autism spectrum disorder; check transferrin IEF in ASD + coagulopathy; "
            "(5) FACIAL DYSMORPHISM — coarse features, widely spaced teeth, large ears; "
            "(6) SEIZURES — usually drug-responsive (contrast with DPAGT1 drug-resistant IS); "
            "(7) MILD HEPATOMEGALY — transaminases mildly elevated; less severe than COG7; "
            "(8) SURVIVAL INTO ADULTHOOD in most — less severe than COG7-CDG; "
            "(9) NO BOMBAY BLOOD GROUP — unlike SLC35C1; type II IEF shared but distinct condition"
        ),
        "key_ddx": (
            "MGAT2 DDx: (1) COG7-CDG — type II IEF; wrinkled skin; most severe; "
            "(2) SLC35C1 (LAD-II) — type II IEF; Bombay blood group; recurrent infections; "
            "(3) Autism spectrum disorder — behavioural/stereotypies; "
            "check transferrin IEF in unexplained ASD + coagulopathy; "
            "(4) Angelman syndrome — intellectual disability + behavioural; UBE3A deletion; "
            "normal CDG IEF; "
            "(5) Rett syndrome (MECP2) — stereotypies + regression; "
            "no CDG IEF; no coagulopathy; "
            "(6) Factor XI deficiency (isolated) — bleeding; normal CDG IEF; normal intellect"
        ),
        "diet_treatment": "No disease-modifying treatment. Standard AED for seizures. Factor replacement/FFP for surgical procedures (bleeding risk). Behavioural therapy. Physiotherapy.",
        "gene_therapy_status": "No approved or advanced gene therapy. Golgi processing enzyme delivery complex. Research ongoing.",
        "critical_ci": (
            "CRITICAL CI: (1) Missing CDG screen in dysmorphic patient with bleeding + ID + autistic features; "
            "MGAT2-CDG mimics ASD with coagulopathy; "
            "(2) Missing factor replacement for surgery — bleeding tendency real; "
            "(3) Using CDP-choline or GalNAc supplements in MGAT2-CDG — not beneficial; "
            "(4) Misattributing as isolated factor XI deficiency without investigating cognition + dysmorphism"
        ),
        "nbs_marker": "Not in standard NBS. Transferrin IEF type II. Glycan mass spectrometry (fibroblasts — absent biantennary complex glycans). MGAT2 enzyme assay + gene sequencing.",
        "key_biomarker": "Transferrin IEF type II. Glycan mass spectrometry (no complex N-glycans). PT/APTT elevated. Factor XI (low). Brain MRI (white matter + structural). MGAT2 gene sequencing.",
        "severity_spectrum": "Severe (early seizures + profound ID + major bleeding) → Moderate (moderate ID + behavioural + mild bleeding) → Mild (mild ID + dysmorphism only; rare residual MGAT2 activity)",
        "founder_variant": "p.His262Arg — European enriched. Various compound heterozygous. No dominant single founder.",
        "key_variants": [
            "p.His262Arg — European enriched, moderate",
            "p.Arg303Gln — moderate",
            "p.Arg409Ser — severe, near-null Golgi activity",
            "IVS5+1G>A — splice, variable severity",
            "p.Glu179Lys — attenuated",
        ],
        "seed": SEED_BASE + 8,
    },
    # ── TMEM165 — Golgi Ca2+/Mn2+ Transporter (CDG-IIk) ────────────────────────
    {
        "gene": "TMEM165", "alias": "TMEM165 — Golgi Ca2+/Mn2+ Transporter (CDG-IIk, manganese-dependent glycosylation)",
        "aa": "324 aa", "kDa": "35.4 kDa",
        "gene_class": "transporter",
        "cdg_type": "Type I or Mixed (Golgi ion homeostasis defect)",
        "locus": "4q12", "omim_gene": 614726,
        "phenotype": "CDG-IIk — hepatomegaly, coagulopathy, variable neurological (intellectual disability); manganese-dependent Golgi glycosyltransferases fail; manganese supplementation experimental",
        "disease": (
            "TMEM165 biallelic loss → Congenital Disorder of Glycosylation type IIk (CDG-IIk, TMEM165-CDG, OMIM #614727). "
            "TMEM165 (transmembrane protein 165, 324aa, 35.4kDa) is a multi-pass transmembrane protein in the "
            "Golgi membrane belonging to the UspA superfamily. It functions as a Ca2+/H+ antiporter and "
            "also transports Mn2+ into the Golgi lumen. Manganese is the essential cofactor for numerous "
            "Golgi glycosyltransferases including beta-1,4-galactosyltransferase (B4GALT1) and sialyltransferases "
            "(MnST). Without TMEM165, Golgi Mn2+ is depleted → Mn-dependent glycosyltransferases lose activity → "
            "impaired N- and O-glycan processing. Interestingly, TMEM165-CDG transferrin IEF can be variable "
            "(type I or mixed type I+II features) because both ER and Golgi glycosylation steps are affected "
            "when ion homeostasis is perturbed. "
            "Phenotype: hepatomegaly + elevated transaminases, coagulopathy, "
            "neurological features variable (mild to moderate intellectual disability, seizures in subset), "
            "failure to thrive, facial dysmorphism variable. "
            "Hepatopathy may improve with age. "
            "MANGANESE SUPPLEMENTATION: small case series report improvement in Golgi glycosylation "
            "markers with Mn2+ supplementation (oral manganese gluconate)."
        ),
        "inheritance": "Autosomal recessive. TMEM165 4q12. Rare. European enrichment (Belgian origin of gene discovery).",
        "hallmark": (
            "TMEM165/CDG-IIk HALLMARKS: (1) MANGANESE-DEPENDENT MECHANISM — Golgi Ca2+/Mn2+ transporter; "
            "the only CDG where the defect is in Golgi ion homeostasis rather than sugar supply or enzyme directly; "
            "(2) VARIABLE IEF PATTERN — can show type I, type II, or mixed features (unusual); "
            "do not exclude CDG-IIk on pure type I IEF if hepatopathy + coagulopathy + variable neuro; "
            "(3) MANGANESE SUPPLEMENTATION EXPERIMENTAL — oral Mn gluconate; "
            "some normalisation of glycan biomarkers reported in small series; "
            "monitor serum Mn levels (toxicity risk — manganism); "
            "(4) HEPATOPATHY MAY IMPROVE WITH AGE — transaminases can normalise; less severe long-term; "
            "(5) COAGULOPATHY — low clotting factors (glycosylation of liver proteins impaired); "
            "(6) CO-LOCATION WITH SRD5A3 on 4q12 — both at 4q12 but different genes; "
            "(7) INTELLECTUAL DISABILITY VARIABLE — mild in some, moderate in others; "
            "distinguishes from COG7 (severe) and MPI (none); "
            "(8) GOLGI MORPHOLOGY DISRUPTED — can see Golgi fragmentation in fibroblasts as with other Golgi-CDG"
        ),
        "key_ddx": (
            "TMEM165 DDx: (1) PMM2-CDG — type I IEF; cerebellar hypoplasia; no hepatic improvement with age; "
            "(2) MPI-CDG — type I IEF; NO neurological; mannose-treatable; "
            "(3) PGM1-CDG — mixed IEF; bifid uvula; exercise intolerance; galactose-treatable; "
            "(4) COG7-CDG — type II IEF; wrinkled skin; most severe; no hepatic improvement; "
            "(5) Mitochondrial hepatopathy — hepatomegaly + elevated transaminases; "
            "respiratory chain enzyme assay + lactate; normal CDG IEF; "
            "(6) Manganese transporter disorders (SLC30A10, SLC39A14) — "
            "these cause manganese ACCUMULATION (different mechanism); brain MRI hypermanganesemia; "
            "NOT the same as TMEM165 where Golgi Mn2+ is depleted despite normal serum Mn"
        ),
        "diet_treatment": "Manganese supplementation (oral Mn gluconate, dose individualized; monitor serum Mn and MRI brain for manganism). Coagulopathy: factor replacement for procedures. Supportive hepatic management. Seizure management if present.",
        "gene_therapy_status": "No approved gene therapy. TMEM165 AAV preclinical. Manganese supplementation is current experimental treatment approach.",
        "critical_ci": (
            "CRITICAL CI: (1) Over-supplementing manganese — manganism (basal ganglia damage) from excess Mn; "
            "monitor serum Mn + brain MRI T1 (Mn causes T1 hyperintensity); "
            "(2) Missing TMEM165 on variable IEF — do NOT exclude on non-classic IEF type; "
            "(3) Confusing with manganese accumulation disorders (SLC30A10, SLC39A14) — "
            "those need Mn CHELATION not supplementation; opposite management; "
            "(4) Missing hepatic improvement trajectory — may reassure families on hepatic prognosis"
        ),
        "nbs_marker": "Not in standard NBS. Transferrin IEF (variable, type I or mixed). Serum Mn level. TMEM165 gene sequencing / WES. Golgi morphology in fibroblasts.",
        "key_biomarker": "Transferrin IEF (variable). Serum manganese (can be low-normal). Liver function tests (elevated initially). Coagulation profile. Brain MRI (T1 for Mn deposition on supplementation). TMEM165 sequencing.",
        "severity_spectrum": "Severe (neonatal multi-organ + significant ID) → Moderate (hepatopathy + coagulopathy + moderate ID; hepatopathy may improve with age) → Mild (hepatopathy only; near-normal cognition; rare high-residual TMEM165)",
        "founder_variant": "p.Arg126Ter — Belgian/European founder (original discovery cohort). p.Leu176Arg — European. Various compound heterozygous.",
        "key_variants": [
            "p.Arg126Ter — Belgian/European founder, null",
            "p.Leu176Arg — structural disruption, moderate",
            "p.Gly304Arg — Golgi targeting disrupted",
            "p.Val44del — Belgian index case family",
            "IVS4+5G>A — splice variant, moderate",
        ],
        "seed": SEED_BASE + 9,
    },
]

# ── Patient simulation ──────────────────────────────────────────────────────────
def _make_patients_for_gene(g):
    rng = random.Random(g["seed"])
    gene = g["gene"]
    patients = []
    n = 40
    for i in range(n):
        pid = f"CDG-{gene}-{i+1:03d}"
        # Gene-specific simulation
        if gene == "PMM2":
            age_dx = rng.choice(
                [rng.uniform(0, 0.5)] * 20 +   # neonatal/infant
                [rng.uniform(0.5, 3)] * 12 +    # infant ataxia/developmental delay
                [rng.uniform(3, 15)] * 8        # childhood ID
            )
            subtype = rng.choice(["classic_severe"] * 18 + ["moderate"] * 14 + ["mild_ataxia_only"] * 8)
            tx_controlled = rng.random() < 0.55   # symptomatic only; partial control
            has_liver_disease = rng.random() < 0.40
            neurological = True  # all PMM2 have neuro
            deceased = (subtype == "classic_severe") and rng.random() < 0.25
        elif gene == "MPI":
            age_dx = rng.choice(
                [rng.uniform(0, 0.5)] * 22 +   # neonatal hepatopathy + coagulopathy
                [rng.uniform(0.5, 5)] * 15 +    # infant enteropathy/hypoglycaemia
                [rng.uniform(5, 20)] * 3        # older — cyclic vomiting
            )
            subtype = rng.choice(["hepatopathy_coagulopathy"] * 22 + ["enteropathy_predominant"] * 12 + ["hypoglycaemia_predominant"] * 6)
            tx_controlled = rng.random() < 0.88  # mannose highly effective
            has_liver_disease = rng.random() < 0.90
            neurological = False  # NO neurological involvement
            deceased = not tx_controlled and rng.random() < 0.08
        elif gene == "ALG6":
            age_dx = rng.choice(
                [rng.uniform(0, 0.5)] * 18 +   # neonatal/infant
                [rng.uniform(0.5, 3)] * 14 +    # infant seizures/ID
                [rng.uniform(3, 15)] * 8        # childhood seizure onset
            )
            subtype = rng.choice(["seizure_id"] * 22 + ["hypotonia_coagulopathy"] * 12 + ["mild_id_only"] * 6)
            tx_controlled = rng.random() < 0.60  # symptomatic only
            has_liver_disease = rng.random() < 0.35
            neurological = True
            deceased = rng.random() < 0.08
        elif gene == "DPAGT1":
            age_dx = rng.choice(
                [rng.uniform(0, 0.5)] * 20 +   # neonatal/infant hypotonia
                [rng.uniform(0.5, 5)] * 14 +    # infant myasthenic features
                [rng.uniform(5, 20)] * 6        # childhood myasthenia recognised
            )
            subtype = rng.choice(["severe_id_seizures"] * 18 + ["myasthenic_predominant"] * 14 + ["moderate_id"] * 8)
            tx_controlled = subtype == "myasthenic_predominant" and rng.random() < 0.60  # pyridostigmine partial
            has_liver_disease = rng.random() < 0.20
            neurological = True
            deceased = (subtype == "severe_id_seizures") and rng.random() < 0.15
        elif gene == "SRD5A3":
            age_dx = rng.choice(
                [rng.uniform(0, 0.5)] * 15 +   # neonatal eye/skin
                [rng.uniform(0.5, 5)] * 16 +    # infant retinal dystrophy recognised
                [rng.uniform(5, 20)] * 9        # childhood ERG abnormal
            )
            subtype = rng.choice(["ophthalmological_severe"] * 20 + ["ichthyosis_ataxia"] * 14 + ["retinal_predominant"] * 6)
            tx_controlled = rng.random() < 0.30  # no treatment; supportive
            has_liver_disease = rng.random() < 0.15
            neurological = True
            deceased = rng.random() < 0.05
        elif gene == "PGM1":
            age_dx = rng.choice(
                [rng.uniform(0, 1)] * 12 +     # neonatal cleft + coagulopathy
                [rng.uniform(1, 10)] * 18 +     # childhood exercise intolerance
                [rng.uniform(10, 40)] * 10      # adult rhabdomyolysis
            )
            subtype = rng.choice(["bifid_uvula_coagulopathy"] * 18 + ["exercise_rhabdomyolysis"] * 16 + ["hepatopathy_predominant"] * 6)
            tx_controlled = rng.random() < 0.82  # galactose highly effective
            has_liver_disease = rng.random() < 0.60
            neurological = rng.random() < 0.40   # mild ID variable
            deceased = rng.random() < 0.03
        elif gene == "COG7":
            age_dx = rng.choice(
                [rng.uniform(0, 0.2)] * 30 +   # neonatal multi-organ
                [rng.uniform(0.2, 1)] * 8 +     # infant severe
                [rng.uniform(1, 5)] * 2         # occasional survivor
            )
            subtype = rng.choice(["severe_neonatal_multiorg"] * 28 + ["infant_severe"] * 8 + ["prolonged_survival"] * 4)
            tx_controlled = rng.random() < 0.20  # supportive only; very poor
            has_liver_disease = rng.random() < 0.95
            neurological = True
            deceased = (subtype in ("severe_neonatal_multiorg", "infant_severe")) and rng.random() < 0.80
        elif gene == "SLC35C1":
            age_dx = rng.choice(
                [rng.uniform(0, 0.3)] * 20 +   # neonatal LAD + Bombay identified
                [rng.uniform(0.3, 2)] * 14 +    # infant infections
                [rng.uniform(2, 15)] * 6        # later identified
            )
            subtype = rng.choice(["lad_bombay_severe"] * 22 + ["lad_moderate_fucose_partial"] * 14 + ["bombay_only_mild"] * 4)
            tx_controlled = subtype == "lad_moderate_fucose_partial" and rng.random() < 0.55  # fucose partial
            has_liver_disease = rng.random() < 0.20
            neurological = True   # intellectual disability
            deceased = (subtype == "lad_bombay_severe") and rng.random() < 0.30
        elif gene == "MGAT2":
            age_dx = rng.choice(
                [rng.uniform(0, 1)] * 16 +     # infant ID + facial
                [rng.uniform(1, 8)] * 18 +      # childhood seizures + dysmorphism
                [rng.uniform(8, 30)] * 6        # adult behavioural
            )
            subtype = rng.choice(["id_behaviour_bleeding"] * 22 + ["seizures_dysmorphism"] * 12 + ["mild_id_bleeding"] * 6)
            tx_controlled = rng.random() < 0.55  # symptomatic management
            has_liver_disease = rng.random() < 0.35
            neurological = True
            deceased = rng.random() < 0.05
        else:  # TMEM165 — CDG-IIk
            age_dx = rng.choice(
                [rng.uniform(0, 0.5)] * 16 +   # neonatal hepatopathy
                [rng.uniform(0.5, 5)] * 18 +    # infant ID + hepatomegaly
                [rng.uniform(5, 20)] * 6        # childhood hepatopathy improving
            )
            subtype = rng.choice(["hepatopathy_coagulopathy"] * 20 + ["hepatopathy_id"] * 14 + ["mild_hepatopathy"] * 6)
            tx_controlled = rng.random() < 0.60  # some Mn supplementation benefit
            has_liver_disease = rng.random() < 0.88
            neurological = subtype == "hepatopathy_id" and rng.random() < 0.70
            deceased = rng.random() < 0.08

        patients.append({
            "pid": pid,
            "gene": gene,
            "subtype": subtype,
            "age_dx_y": round(max(0.0, age_dx), 2),
            "tx_controlled": tx_controlled,
            "has_liver_disease": has_liver_disease,
            "neurological": neurological,
            "deceased": deceased,
        })
    return patients


ALL_PATIENTS = []
for _g in CDG_GENES:
    _pts = _make_patients_for_gene(_g)
    _g["n_patients"] = len(_pts)
    _g["patients"] = _pts
    ALL_PATIENTS.extend(_pts)


# ─── API: get_overview ───────────────────────────────────────────────────────────
def get_overview():
    total = len(ALL_PATIENTS)
    n_tx = sum(1 for p in ALL_PATIENTS if p["tx_controlled"])
    n_liver = sum(1 for p in ALL_PATIENTS if p["has_liver_disease"])
    n_neuro = sum(1 for p in ALL_PATIENTS if p["neurological"])
    n_deceased = sum(1 for p in ALL_PATIENTS if p["deceased"])

    gene_summary = []
    for g in CDG_GENES:
        pts = g["patients"]
        gene_summary.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "locus": g["locus"],
            "gene_class": g["gene_class"],
            "cdg_type": g["cdg_type"],
            "n_patients": g["n_patients"],
            "diet_treatment": g["diet_treatment"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "pct_tx": round(100 * sum(1 for p in pts if p["tx_controlled"]) / len(pts), 1),
            "pct_liver": round(100 * sum(1 for p in pts if p["has_liver_disease"]) / len(pts), 1),
            "pct_neuro": round(100 * sum(1 for p in pts if p["neurological"]) / len(pts), 1),
            "pct_deceased": round(100 * sum(1 for p in pts if p["deceased"]) / len(pts), 1),
        })

    return {
        "atlas": "CDG-Atlas — Complete 10-Gene Congenital Disorders of Glycosylation Atlas",
        "n_genes": len(CDG_GENES),
        "n_patients": total,
        "seeds": [g["seed"] for g in CDG_GENES],
        "genes_covered": [g["gene"] for g in CDG_GENES],
        "gene_classes": {
            "phosphomutase":    ["PMM2", "PGM1"],
            "isomerase":        ["MPI"],
            "glucosyltransferase": ["ALG6"],
            "transferase":      ["DPAGT1", "MGAT2"],
            "reductase":        ["SRD5A3"],
            "golgi_tethering":  ["COG7"],
            "transporter":      ["SLC35C1", "TMEM165"],
        },
        "cdg_types": {
            "Type I (LLO defect)":            ["PMM2", "MPI", "ALG6", "DPAGT1", "SRD5A3"],
            "Mixed (Type I + II features)":   ["PGM1"],
            "Type II (Golgi processing)":     ["COG7", "SLC35C1", "MGAT2", "TMEM165"],
        },
        "aggregate_clinical": {
            "pct_tx_controlled": round(100 * n_tx / total, 1),
            "pct_liver_disease": round(100 * n_liver / total, 1),
            "pct_neurological":  round(100 * n_neuro / total, 1),
            "pct_deceased":      round(100 * n_deceased / total, 1),
        },
        "gene_summary": gene_summary,
        "critical_clinical_rules": [
            "PMM2 (CDG-Ia): MOST COMMON CDG (~60%); p.Phe119Leu LETHAL in homozygous — NEVER seen in living patients; cerebellar hypoplasia + coagulopathy + inverted nipples + abnormal fat pads PATHOGNOMONIC; NO effective treatment; mannose does NOT work in PMM2 (enzyme is upstream of mannose conversion)",
            "MPI (CDG-Ib): ONLY CDG without neurological involvement — brain MRI NORMAL; TREATABLE with D-mannose 2g/kg/day; reverses coagulopathy + hepatopathy + enteropathy; transferrin IEF may be NORMAL — use erythrocyte MPI enzyme assay; do NOT miss — it is curative",
            "ALG6 (CDG-Ic): 2nd most common N-CDG (~5%); seizures + intellectual disability; type I IEF — distinguish from PMM2 by PMM2 enzyme assay; mannose DOES NOT work in ALG6-CDG; symptomatic management only",
            "DPAGT1 (CDG-Ij): initiates N-glycosylation (FIRST committed step — dolichol-PP-GlcNAc); myasthenic CMS overlap — AChR-antibody NEGATIVE; pyridostigmine partially effective; DO NOT give immunosuppression (not autoimmune); drug-resistant seizures often dominant",
            "SRD5A3 (CDG-Iq): dolichol synthesis (polyprenol → dolichol); ophthalmological involvement PROMINENT (Leber-like retinal dystrophy, nystagmus, coloboma) in virtually all; ichthyosis-like skin; skin + eye + CDG = SRD5A3; mannose and galactose do NOT help (dolichol defect)",
            "PGM1 (CDG-It): BIFID UVULA + CLEFT PALATE PATHOGNOMONIC — examine palate in any rhabdomyolysis + coagulopathy; TREATABLE with D-galactose 1-1.5g/kg/day; exercise intolerance + rhabdomyolysis + elevated CK; mixed IEF pattern; fasting avoidance (frequent meals + cornstarch)",
            "COG7 (CDG-IIe): MOST SEVERE Golgi-CDG; wrinkled/loose skin at birth PATHOGNOMONIC among Golgi-CDG; multi-organ failure (cardiac + hepatic + brain); very poor prognosis (death in infancy in most); type II IEF; Golgi COG complex tethering defect",
            "SLC35C1 (CDG-IIc, LAD-II): BOMBAY BLOOD GROUP (h/h, absent H antigen) PATHOGNOMONIC — FATAL haemolytic transfusion reaction with standard blood; BOMBAY DONORS ONLY; LAD-II (absent CD15s/sialyl-Lewis X — absent neutrophil rolling); L-fucose 25mg/kg/day partial benefit; register Bombay status in blood bank immediately",
            "MGAT2 (CDG-IIa): ALL biantennary complex N-glycans absent; bleeding tendency (PT/APTT elevated, factor XI low); behavioural problems + stereotypies may mimic ASD — check transferrin IEF in ASD + coagulopathy + dysmorphism; type II IEF; survival into adulthood common",
            "TMEM165 (CDG-IIk): Golgi Ca2+/Mn2+ transporter — manganese-dependent glycosyltransferases fail; variable IEF (type I or mixed); manganese supplementation experimental (small series); monitor serum Mn to avoid manganism (opposite of SLC30A10 where Mn accumulates — different management: NO chelation in TMEM165)",
        ],
    }


# ─── API: get_breakdown ──────────────────────────────────────────────────────────
def get_breakdown():
    gene_rows = []
    for g in CDG_GENES:
        pts = g["patients"]
        gene_rows.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "gene_class": g["gene_class"],
            "cdg_type": g["cdg_type"],
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
            "pct_tx": round(100 * sum(1 for p in pts if p["tx_controlled"]) / len(pts), 1),
            "pct_liver": round(100 * sum(1 for p in pts if p["has_liver_disease"]) / len(pts), 1),
            "pct_neuro": round(100 * sum(1 for p in pts if p["neurological"]) / len(pts), 1),
            "pct_deceased": round(100 * sum(1 for p in pts if p["deceased"]) / len(pts), 1),
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
        })
    return {
        "genes": gene_rows,
        "total": len(CDG_GENES),
        "total_patients": len(ALL_PATIENTS),
    }


# ─── API: get_definitions ─────────────────────────────────────────────────────
def get_definitions():
    return {
        "atlas": "CDG-Atlas — Complete 10-Gene Congenital Disorders of Glycosylation Atlas",
        "cdg_overview": {
            "full_name": "Congenital Disorders of Glycosylation — inherited defects in glycan biosynthesis or processing → hypoglycosylated glycoproteins → multi-system disease",
            "genes_in_atlas": 10,
            "total_known_cdg_types": ">150 distinct CDG types identified to date (and growing via WES/WGS)",
            "collective_incidence": "~1 in 50,000-100,000 overall; PMM2-CDG alone ~1 in 20,000 (most common); likely significantly underdiagnosed",
            "inheritance_note": "Most autosomal recessive; ATP6AP1-CDG and SSR4-CDG are X-linked; TMEM165 lobe shared with SRD5A3 on 4q12",
            "nbs_note": "No CDG in standard expanded NBS (no specific acylcarnitine/amino acid marker for most); clinical + transferrin IEF is the primary screen; WES expanding detection",
        },
        "definitions": [
            {
                "term": "Transferrin Isoelectric Focusing (IEF) — CDG Screening Gold Standard",
                "definition": "Serum transferrin normally carries 4 sialic acids (tetrasialotransferrin = major band). CDG disrupts sialylation: Type I pattern (PMM2, MPI, ALG6, DPAGT1, SRD5A3): increased disialo- and asialotransferrin, decreased tetrasialotransferrin; reflects incomplete LLO → fewer glycan chains attached. Type II pattern (COG7, SLC35C1, MGAT2, TMEM165): abnormal distribution of higher oligosaccharide forms; truncated or incorrectly processed glycans. Mixed pattern (PGM1): features of both. False negatives: neonates (<3 weeks), blood transfusion recipients, severe galactosaemia (mimics type II). Not detected by IEF: CDGs affecting O-glycosylation only (EXT1/EXT2, LARGE, POMT1/2, FKTN, FKRP etc.).",
            },
            {
                "term": "Lipid-Linked Oligosaccharide (LLO) — N-Glycosylation Precursor",
                "definition": "The LLO (dolichol-PP-GlcNAc2Man9Glc3) is assembled sequentially on dolichol-phosphate in the ER membrane: first 7 sugars (2GlcNAc + 5Man) added on the cytoplasmic face, then translocated ('flipped') to the ER lumen, where 4Man + 3Glc are added. The complete Glc3Man9GlcNAc2 structure is transferred en bloc by oligosaccharyltransferase (OST) to Asn-X-Ser/Thr sequons in nascent proteins. Type I CDGs arise from defects in LLO assembly (PMM2, MPI, ALG6, DPAGT1, SRD5A3) or transfer, leading to hypoglycosylation. Type II CDGs arise from defects in Golgi trimming/elongation of the N-glycan after transfer.",
            },
            {
                "term": "Dolichol — Lipid Carrier for LLO N-Glycosylation",
                "definition": "Dolichol is a long-chain polyprenyl alcohol (C80-C100) that serves as the lipid anchor for LLO assembly in the ER membrane. Dolichol biosynthesis: polyprenyl-PP synthase (DHDDS, GGPS1) → long-chain polyprenyl-PP → SRD5A3 reduces the alpha-omega double bond → dolichol. Dolichol is then phosphorylated to dolichol-P (DOLK), which accepts GlcNAc-1-P from UDP-GlcNAc (DPAGT1). SRD5A3 deficiency (CDG-Iq): polyprenol accumulates, dolichol depleted → LLO initiation fails. DHDDS-CDG (DHDDS gene) also disrupts dolichol. Dolichol-P is recycled after oligosaccharyl transfer.",
            },
            {
                "term": "Congenital Myasthenic Syndrome (CMS) — DPAGT1 N-Glycosylation Overlap",
                "definition": "Congenital myasthenic syndromes are inherited disorders of neuromuscular junction (NMJ) transmission. NMJ proteins (acetylcholine receptor [AChR], MuSK, rapsyn, agrin, LRP4, collagen Q, DOK7, GFPT1, ALG2, ALG14, DPAGT1) require N-glycosylation for proper folding, trafficking, and clustering at the NMJ. DPAGT1 (CDG-Ij) initiates N-glycosylation of NMJ proteins → DPAGT1 loss → hypoglycosylated AChR + rapsyn + MuSK → impaired NMJ organisation → fatigable weakness, ptosis, ophthalmoplegia. AChR antibodies are NEGATIVE (structural, not autoimmune). Pyridostigmine (AChEI) + albuterol (beta-2 agonist) partially compensate. Compare: GFPT1-CMS (tubular aggregates in muscle biopsy + CDG overlap). CDG-related CMS is seronegative — always check CDG IEF in seronegative myasthenia + intellectual disability.",
            },
            {
                "term": "Bombay Blood Group (Oh) — SLC35C1 Pathognomonic Hallmark",
                "definition": "The Bombay (Oh) blood group is characterised by complete absence of H antigen (the precursor from which A and B antigens are made). In SLC35C1-CDG (LAD-II), GDP-fucose transport into the Golgi is absent → FUT1 cannot add fucose to make H antigen on RBCs → Bombay phenotype. Serologically: RBCs type as O but are incompatible with anti-H antibody (present in all Bombay patients). ALL standard blood types (O-, O+, A, B, AB) are incompatible — patients have anti-A, anti-B, AND anti-H. Only H-antigen-negative (Bombay) donors can be used. Bombay phenotype also occurs rarely from non-CDG causes (FUT1 mutations) but these patients have normal CDG IEF. Testing: anti-H lectin (Ulex europaeus) agglutination fails with Bombay RBCs. Blood bank must register patient and pre-arrange Bombay blood.",
            },
            {
                "term": "Leukocyte Adhesion Deficiency Type II (LAD-II) — SLC35C1",
                "definition": "LAD-II (SLC35C1-CDG) is caused by absent GDP-fucose transport → no sialyl-Lewis X (CD15s, SLeX) on neutrophil surface glycoproteins → impaired neutrophil rolling on vascular endothelium (selectins recognise sialyl-Lewis X) → neutrophils cannot migrate to infection sites. Clinical: recurrent severe bacterial infections, markedly elevated WBC (paradoxical leukocytosis — neutrophils cannot leave blood vessels), delayed umbilical cord separation (>3 weeks), poor wound healing, intellectual disability. Contrast LAD-I (ITGB2/CD18): same clinical syndrome but absent CD18 on flow cytometry, normal blood group, normal CDG IEF. Diagnosis: flow cytometry (absent CD15s) + Bombay blood group + transferrin IEF type II + SLC35C1 sequencing. L-fucose supplementation partially improves neutrophil function.",
            },
            {
                "term": "PMM2 p.Phe119Leu — Lethal in Homozygous",
                "definition": "p.Phe119Leu (c.357C>A) in PMM2 is a near-null missense variant that abolishes phosphomannomutase 2 catalytic activity. In vitro and from animal models, homozygous complete PMM2 loss is embryonically lethal (cannot assemble N-glycans at all — incompatible with development). Therefore, p.Phe119Leu is NEVER found in homozygous state in living CDG patients. All patients with p.Phe119Leu are compound heterozygous with a milder (at least partial residual activity) second allele. If genetic sequencing reports apparent homozygous p.Phe119Leu: (1) verify sequencing quality/coverage; (2) consider consanguinity with coincidental second pathogenic allele missed; (3) re-examine whether the patient truly has PMM2-CDG vs another type I CDG.",
            },
            {
                "term": "Bifid Uvula — PGM1-CDG Pathognomonic Marker",
                "definition": "Bifid uvula (forked or split uvula) is a midline fusion defect of the secondary palate. In PGM1-CDG, bifid uvula ± cleft palate occurs in the majority of patients (>80%). The combination of bifid uvula + exercise intolerance/rhabdomyolysis + coagulopathy + elevated CK is PATHOGNOMONIC for PGM1-CDG and should trigger CDG screen (transferrin IEF mixed pattern). Isolated bifid uvula is common in the general population (~1%), but the combination with metabolic features is unique to PGM1-CDG. Cleft palate (complete or submucosal) also occurs. Mechanism: PGM1 is required for glycosylation of palate structural proteins during embryonic midline fusion. Unlike non-CDG palatal defects, PGM1-CDG bifid uvula cannot be prevented postnatally but the metabolic consequences are treatable with galactose.",
            },
            {
                "term": "Wrinkled/Loose Skin at Birth — COG7 Golgi-CDG Marker",
                "definition": "COG7-CDG presents at birth with striking redundant, wrinkled, loose skin that superficially resembles cutis laxa (elastin deficiency). Unlike true cutis laxa (ELN, FBLN5 mutations), the COG7-CDG skin wrinkles are not caused by elastin defect — elastic fibres are histologically intact (or only mildly abnormal). The skin wrinkles reflect impaired glycosylation of extracellular matrix proteins (collagen, proteoglycans) involved in skin structure. Among the Golgi-CDG phenotypes, COG7 is the only one with consistent neonatal cutis-laxa-like skin. Differentiates COG7 from other CDGs at the bedside. Skin biopsy shows intact elastic fibres (vs true cutis laxa where elastic fibres are absent/fragmented). All wrinkled-skin neonates should receive CDG transferrin IEF screening.",
            },
            {
                "term": "GDP-Mannose — Central Donor for N-Glycan LLO Assembly",
                "definition": "GDP-mannose is the activated mannose donor for 7 of 9 mannose residues in the LLO (Man9GlcNAc2-PP-dolichol). Synthesis: Fructose-6-P → Mannose-6-P (MPI) → Mannose-1-P (PMM2) → GDP-mannose (GMPPB). PMM2 loss → insufficient GDP-mannose → incomplete LLO (Man5GlcNAc2 instead of Man9GlcNAc2) → hypoglycosylated proteins. MPI loss → cannot make mannose-6-P from fructose-6-P → also insufficient GDP-mannose → same LLO defect. TREATMENT DIFFERENCE: oral mannose bypasses MPI (mannose → mannose-6-P via hexokinase) → restores GDP-mannose in MPI-CDG; but oral mannose CANNOT bypass PMM2 defect (PMM2 is DOWNSTREAM of mannose-6-P — mannose still requires PMM2 to convert M6P → M1P → GDP-mannose).",
            },
            {
                "term": "COG Complex — Conserved Oligomeric Golgi Complex",
                "definition": "The COG complex is a hetero-octameric tethering factor (COG1-COG8) essential for intra-Golgi vesicle trafficking. It consists of two lobes: Lobe A (COG1-4) and Lobe B (COG5-8). COG maintains Golgi enzyme localisation by capturing retrograde vesicles returning from TGN/late Golgi to early Golgi. Without COG, glycosyltransferases and glycosidases are mislocalised or degraded → defective N- and O-glycan processing. All 8 COG subunit genes have been implicated in CDG. COG7 (lobe B) is the most severe CDG; COG5 and COG8 (lobe B) are typically milder. COG1 and COG4 (lobe A) have intermediate severity. All COG-CDGs show type II transferrin IEF.",
            },
            {
                "term": "D-Galactose Therapy — PGM1-CDG",
                "definition": "Oral D-galactose (1-1.5g/kg/day in divided doses) is the established treatment for PGM1-CDG. Mechanism: PGM1 converts glucose-1-phosphate ↔ glucose-6-phosphate and is also required for UDP-glucose and UDP-galactose production from galactose-1-phosphate via the Leloir pathway. Exogenous galactose → galactose-1-phosphate (GALK1) → UDP-galactose (GALT) → glucose-1-phosphate (GALE) → bypasses PGM1 for UDP-galactose and glucose-1-phosphate supply → restores glycosylation. Response: transferrin IEF normalises, coagulopathy reverses, rhabdomyolysis frequency decreases, liver enzymes improve. Distinct from galactose-free diet needed in classic galactosaemia (GALT deficiency) — in PGM1-CDG, galactose is the TREATMENT not the toxin. Also different from mannose for MPI-CDG (wrong sugar).",
            },
            {
                "term": "N-Glycosylation vs O-Glycosylation — CDG Spectrum",
                "definition": "N-glycosylation attaches glycans to asparagine (Asn-X-Ser/Thr sequons). Most CDG diseases affect N-glycosylation and are detectable by transferrin IEF (transferrin is N-glycosylated). O-glycosylation attaches glycans to serine or threonine via GalNAc (mucin-type), mannose (POMT1/POMT2 — congenital muscular dystrophies), fucose (POFUT1/2 — Dowling-Degos / Hajdu-Cheney), glucose (POGLUT1 — Dowling-Degos), or GlcNAc (EXT1/2 — hereditary exostoses). Pure O-glycosylation CDGs are NOT detected on transferrin IEF. PGM1-CDG is unusual: mixed N + O glycosylation defect → mixed IEF pattern. Diagnosis of O-glycosylation CDGs requires specific O-glycan analysis or gene sequencing.",
            },
            {
                "term": "Manganese (Mn2+) in Golgi Glycosylation — TMEM165",
                "definition": "Manganese is an essential cofactor for several Golgi glycosyltransferases, most critically: beta-1,4-galactosyltransferases (B4GALTs) and sialyltransferases (STs). These enzymes have Mn2+-dependent catalytic activity (manganese co-crystal structures). TMEM165 (CDG-IIk) transports Mn2+ from cytoplasm into Golgi lumen; TMEM165 loss → Golgi Mn2+ depletion → B4GALT and ST activity falls → impaired N- and O-galactosylation and sialylation. Manganese supplementation: bypasses TMEM165 by providing excess extracellular Mn → elevated plasma Mn → some enters Golgi via alternative pathways. Distinguish from SLC30A10 and SLC39A14 deficiency: those cause manganese ACCUMULATION (hypermanganesemia → basal ganglia damage) and require chelation; TMEM165 is a Golgi-LOCAL defect not associated with hypermanganesemia.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== CDG Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    print(f"Tx controlled: {ov['aggregate_clinical']['pct_tx_controlled']}%")
    print(f"Neurological: {ov['aggregate_clinical']['pct_neurological']}%")
    print(f"Liver disease: {ov['aggregate_clinical']['pct_liver_disease']}%")
    print(f"Deceased: {ov['aggregate_clinical']['pct_deceased']}%")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
