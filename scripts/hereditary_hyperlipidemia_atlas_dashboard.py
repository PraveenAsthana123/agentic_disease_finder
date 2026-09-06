#!/usr/bin/env python3
"""Hereditary-Hyperlipidemia-Atlas — Complete 8-Gene Hereditary Hyperlipidemia Atlas
LDLR   (LDL receptor; 860 aa; 19p13.2; AD;
         Familial Hypercholesterolemia type 1 (FH1) — most common monogenic dyslipidaemia;
         1:300 heterozygous prevalence; LDL >190 mg/dL untreated; statins + ezetimibe first-line;
         PCSK9i (evolocumab/alirocumab) for inadequate control; LDL apheresis for HoFH;
         seed SEED_BASE+0) ·
APOB   (Apolipoprotein B-100; 4563 aa; 2p24.1; AD;
         Familial Defective ApoB-100 (FDB / FH2) — R3527Q impairs LDL-receptor binding;
         milder phenotype than LDLR (LDL 200–300 mg/dL); statins usually sufficient;
         1:1000 prevalence; WES may call as VUS unless functional assay performed;
         seed SEED_BASE+1) ·
PCSK9  (Proprotein convertase subtilisin/kexin type 9; 692 aa; 1p32.3; AD;
         GOF D374Y/S127R → FH3 (most severe, highest LDL); LOF R46L/Y142X → protective 88% CVD reduction;
         evolocumab/alirocumab FDA-approved (FOURIER/ODYSSEY OUTCOMES); PCSK9i lowers LDL 50–60%;
         seed SEED_BASE+2) ·
LDLRAP1 (LDL receptor adaptor protein 1; 308 aa; 1p36.11; AR;
          Autosomal Recessive Hypercholesterolemia (ARH) — clathrin adaptor; biallelic required;
          liver LDL uptake abolished; PCSK9i surprisingly effective (residual extrahepatic LDLR);
          seed SEED_BASE+3) ·
ABCG5  (ATP-binding cassette subfamily G member 5; 651 aa; 2p21; AR;
         Sitosterolemia type 1 (STSL1) — plant sterol accumulation; xanthomas + premature CVD + haemolysis;
         ezetimibe CURATIVE (blocks NPC1L1 intestinal sterol absorption); low plant-sterol diet;
         seed SEED_BASE+4) ·
ABCG8  (ATP-binding cassette subfamily G member 8; 673 aa; 2p21; AR;
         Sitosterolemia type 2 (STSL2) — most common STSL gene; biliary sterol secretion impaired;
         tendon xanthomas at young age; ezetimibe CURATIVE; DO NOT mistake for FH;
         seed SEED_BASE+5) ·
APOE   (Apolipoprotein E; 299 aa; 19q13.32; AR-like (APOE2/E2 homozygous);
         Familial Dysbetalipoproteinemia (FD) / Type III Hyperlipoproteinemia;
         VLDL remnant accumulation → elevated TG AND LDL; palmar xanthomas PATHOGNOMONIC;
         fibrates + statins; APOE2/E2 + second hit (obesity/diabetes/hypothyroidism) required;
         seed SEED_BASE+6) ·
LPA    (Lipoprotein(a); 4529 aa; 6q25–q26; co-dominant (apo(a) size polymorphism);
         Lp(a) elevation >50 mg/dL independent CVD risk factor; apo(a) kringle-IV repeats inversely correlated;
         no FDA-approved Lp(a)-lowering Rx yet (pelacarsen Phase 3 HORIZON trial);
         aspirin controversial; PCSK9i modest 25% reduction; apheresis for very high-risk;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1670–1677)
"""

import random

SEED_BASE = 1670

HYPERLIPIDEMIA_GENES = [
    # ── LDLR — FH1 / Familial Hypercholesterolemia type 1 ───────────────────
    {
        "gene": "LDLR",
        "protein": "LDLR — FH1 AD — LDL Receptor — Most Common Monogenic Dyslipidaemia — 1:300 Heterozygous — Statins+Ezetimibe First-Line — PCSK9i for Inadequate Control — LDL Apheresis HoFH",
        "alias": (
            "LDLR (low-density lipoprotein receptor); OMIM gene 606945; "
            "Familial Hypercholesterolemia type 1 (FH1) OMIM 143890. "
            "19p13.2; 860 aa; ~95 kDa; AD heterozygous LOF (HeFH) / AR biallelic (HoFH). "
            "FUNCTION: LDLR is the hepatic cell-surface receptor that binds apoB-100 on LDL particles and apoE on VLDL/IDL. "
            "LDL-LDLR complex internalised via clathrin-coated pits → lysosomal LDL degradation → "
            "cholesterol released for hepatocyte use → SREBP2 pathway feedback suppresses LDLR expression when cholesterol sufficient. "
            "LOF LDLR: Reduced or absent hepatic LDL uptake → plasma LDL cholesterol accumulation → "
            "atherosclerotic plaque formation → premature ASCVD. "
            "MUTATIONS: >2,500 pathogenic LDLR variants catalogued (UCL LDLR database); "
            "classes: Class 1 (null — no mRNA), Class 2 (transport defect — ER retention), "
            "Class 3 (binding defect — cannot bind apoB/apoE), Class 4 (internalisation defect — clathrin pit), "
            "Class 5 (recycling defect — LDLR not returned to surface). "
            "EPIDEMIOLOGY: Most common monogenic dyslipidaemia; prevalence 1:300 HeFH worldwide "
            "(previously estimated 1:500, now recognised more common via genetic registries); "
            "HoFH (biallelic): 1:160,000–1:300,000 — most severe. "
            "CLINICAL PHENOTYPE — HeFH: "
            "LDL-C 190–400 mg/dL untreated (vs normal <130 mg/dL); "
            "premature ASCVD (MI in males <55 y, females <65 y); "
            "tendon xanthomas (Achilles, dorsum of hand) — PATHOGNOMONIC when present; "
            "xanthelasma (periorbital); corneal arcus before age 45; "
            "family history 50% first-degree relatives affected. "
            "SIMON BROOME CRITERIA (UK): Definite FH = total cholesterol >7.5 mmol/L + tendon xanthomas; "
            "Possible FH = total cholesterol >7.5 mmol/L + family history premature CVD/hypercholesterolaemia. "
            "DUTCH LIPID CLINIC NETWORK (DLCN) SCORE: points for LDL-C level, family history, clinical features, "
            "and genetic confirmation; score ≥6 probable FH, ≥8 definite FH. "
            "TREATMENT — STATINS FIRST-LINE: "
            "High-intensity statin (rosuvastatin 20–40 mg / atorvastatin 40–80 mg) — LDL-C reduction 50–60%; "
            "mechanism: inhibit HMG-CoA reductase → reduced hepatocyte cholesterol → SREBP2 upregulates LDLR expression → "
            "increased LDL clearance (requires at least partially functional LDLR — limited efficacy in HoFH). "
            "EZETIMIBE ADD-ON: Inhibits NPC1L1 intestinal cholesterol transporter → additional LDL-C reduction 15–20%; "
            "IMPROVE-IT trial: statin + ezetimibe reduces MACE 6.4% vs statin alone (HR 0.936; p=0.016). "
            "PCSK9 INHIBITORS (evolocumab/alirocumab) — LDL-C REDUCTION 50–60% ADDITIONAL: "
            "Evolocumab — FOURIER trial (Sabatine 2017 NEJM): 27% relative MACE reduction (HR 0.85; p<0.001) in ASCVD patients on statin; "
            "LDL-C reduced from 92→30 mg/dL median; "
            "alirocumab — ODYSSEY OUTCOMES (Schwartz 2018 NEJM): 15% relative reduction all-cause mortality. "
            "LDL APHERESIS — HoFH STANDARD OF CARE: "
            "Extracorporeal removal of LDL particles every 1–2 weeks; reduces LDL-C 65–75% acutely; "
            "rebound by next session; cumulative benefit on plaque regression; "
            "FDA-approved for HoFH or HeFH with LDL >300 mg/dL + ASCVD on max therapy. "
            "LOMITAPIDE (HoFH only): MTP inhibitor → blocks VLDL/chylomicron assembly → LDL-C reduction 50%; "
            "hepatotoxicity monitoring mandatory; restricted to HoFH. "
            "INCLISIRAN (siRNA PCSK9 inhibitor): Leqvio® — twice-yearly subcutaneous injection; "
            "50% LDL-C reduction; ORION-10 trial (Ray 2020 NEJM). "
            "CASCADE TESTING: FH is autosomal dominant — universal cascade testing (all 1st-degree relatives) is cost-effective and life-saving; "
            "screening starting age 2–5 years for family members of confirmed FH cases."
        ),
        "locus": "19p13.2",
        "aa": 860,
        "kDa": 95,
        "omim_gene": "606945",
        "omim_disease": "Familial Hypercholesterolemia type 1 HeFH (OMIM 143890); Homozygous FH HoFH (OMIM 143890)",
        "inheritance": "AD heterozygous LOF (HeFH 1:300); AR biallelic (HoFH 1:160,000–1:300,000)",
        "gene_class": "LDL receptor — endocytic hepatic LDL-apoB100/apoE clearance — clathrin-coated pit internalisation — SREBP2 feedback regulation",
        "key_alerts": [
            "LDLR-STATINS-MECHANISM-REQUIRES-LDLR: Statins reduce hepatocyte cholesterol → upregulate remaining functional LDLR → increased LDL clearance; statins have LIMITED efficacy in HoFH (biallelic null) because there is no LDLR to upregulate — add PCSK9i + lomitapide + apheresis",
            "LDLR-LDL-APHERESIS-HOFH-STANDARD-CARE: Biallelic LDLR → HoFH with LDL-C 400–1000 mg/dL untreated; LDL apheresis every 1–2 weeks is standard of care; reduces LDL-C 65–75% per session; cumulative plaque regression proven",
            "LDLR-TENDON-XANTHOMAS-PATHOGNOMONIC: Tendon xanthomas (Achilles/dorsum hand) in young patient with high LDL = FH until proven otherwise; Simon Broome definite FH criteria met — start treatment without waiting for genetic confirmation",
            "LDLR-CASCADE-TESTING-MANDATORY: AD inheritance → 50% first-degree relatives affected; universal cascade testing starting age 2–5 years; FH untreated from birth → decades of LDL exposure → premature MI; early detection is life-saving",
            "LDLR-PCSK9I-LDL-50-60PCT-ADDITIONAL: PCSK9 inhibitors (evolocumab/alirocumab) reduce LDL-C additional 50–60% on top of statins; FOURIER trial: 27% MACE reduction; FDA-approved for HeFH or ASCVD with inadequate statin response",
        ],
        "etiologies": [
            "LDLR Class 2 transport defect (most common ~50% of variants) — missense → misfolded LDLR retained in ER → not trafficked to cell surface",
            "LDLR Class 1 null allele — nonsense/frameshift → absent protein → no LDLR at hepatocyte surface → no LDL clearance",
            "LDLR Class 3 binding defect — EGF precursor domain mutations → cannot bind LDL-apoB100 or VLDL-apoE → normal surface expression but non-functional",
            "HoFH (biallelic LOF) — LDL-C 400–1000 mg/dL untreated; cutaneous xanthomas in childhood; MI before age 20 without treatment",
        ],
        "stats": {
            "mean_dx_age": 35,
            "mean_dx_delay_months": 24,
            "ldl_c_untreated_mean_mgdL": 310,
            "statins_ldl_reduction_pct": 55,
            "pcsk9i_additional_ldl_reduction_pct": 55,
            "lifetime_mi_risk_untreated_heFH_pct": 50,
        },
        "dx_delay_distribution": "18–36 months (LDL often attributed to diet; FH missed without genetic/cascade screening)",
    },

    # ── APOB — FDB / Familial Defective ApoB-100 ────────────────────────────
    {
        "gene": "APOB",
        "protein": "APOB — FDB FH2 AD — ApoB-100 R3527Q — Impaired LDL-Receptor Binding — Milder Than LDLR — Statins Usually Sufficient — 1:1000 Prevalence",
        "alias": (
            "APOB (apolipoprotein B-100); OMIM gene 107730; "
            "Familial Defective ApoB-100 (FDB / FH type 2) OMIM 144010. "
            "2p24.1; 4563 aa (APOB-100 full-length); ~512 kDa; AD missense. "
            "FUNCTION: Apolipoprotein B-100 is the sole structural apoprotein of VLDL, IDL, and LDL. "
            "APOB-100 mediates LDL binding to the LDL receptor via its receptor-binding domain (residues ~3359–3369). "
            "APOB-48: Intestinal-specific truncated form (2152 aa, stop codon introduced by APOBEC1 mRNA editing at codon 2153); "
            "forms chylomicrons — NOT affected in FDB. "
            "PATHOGENIC VARIANT — R3527Q (c.10580G>A): "
            "Arg3527Gln in the receptor-binding domain of apoB-100 → impaired LDLR binding affinity (10–50× reduced); "
            "LDL particles carry R3527Q apoB → cannot bind LDLR efficiently → LDL circulates longer → elevated LDL-C. "
            "Other pathogenic APOB variants: R3527W (more severe), c.10708+1G>T (splice). "
            "EPIDEMIOLOGY: 1:500–1:1,000 in Northern Europeans; R3527Q most common; "
            "underdiagnosed — may have normal or only moderately elevated cholesterol; "
            "phenotype: LDL-C 200–350 mg/dL typically (milder than LDLR HeFH); "
            "cardiovascular risk lower than LDLR FH1 but still significantly elevated vs general population. "
            "CLINICAL DISTINCTION FROM LDLR FH: "
            "Tendon xanthomas LESS frequent in FDB vs LDLR FH (apoB mutation → LDL slightly different particle, less macrophage uptake); "
            "LDL-C generally 200–300 mg/dL (vs >300 mg/dL in LDLR); "
            "GENETIC TESTING MANDATORY: phenotype overlap with LDLR FH — cannot distinguish clinically; "
            "WES may call R3527Q as VUS if functional assay not done — use DLCN score + family history to guide interpretation. "
            "TREATMENT: "
            "High-intensity statin usually sufficient to reach LDL-C targets (<70 mg/dL for ASCVD, <100 mg/dL for primary prevention FH); "
            "mechanism: statins upregulate LDLR → residual LDL-LDLR binding still occurs (apoB binds with low affinity, LDLR still functional in FDB); "
            "ezetimibe add-on if target not reached; PCSK9i in high-risk FDB with inadequate statin response; "
            "statins MORE effective in FDB than LDLR HeFH (LDLR is normal in FDB — statins can fully upregulate it). "
            "PREGNANCY: Statins CONTRAINDICATED in pregnancy (Category X — teratogenic in animal models); "
            "bile acid sequestrants (cholestyramine) safe alternative during pregnancy; "
            "LDL-C increases physiologically in pregnancy (VLDL production increases) — monitor closely."
        ),
        "locus": "2p24.1",
        "aa": 4563,
        "kDa": 512,
        "omim_gene": "107730",
        "omim_disease": "Familial Defective ApoB-100 / FH type 2 (OMIM 144010)",
        "inheritance": "AD — heterozygous missense (R3527Q most common in Northern Europeans)",
        "gene_class": "VLDL/LDL structural apoprotein — LDLR receptor-binding domain (residues ~3359–3369) — mRNA editing yields APOB-48 in intestine",
        "key_alerts": [
            "APOB-R3527Q-LDL-RECEPTOR-BINDING-IMPAIRED: R3527Q in apoB-100 receptor-binding domain → 10–50× reduced LDLR affinity → LDL circulates longer → LDL-C 200–350 mg/dL; LDLR is NORMAL in FDB — statins MORE effective than in LDLR FH because statin-upregulated LDLR can still clear some apoB-R3527Q LDL",
            "APOB-WES-VUS-RISK: R3527Q may be called Variant of Uncertain Significance by standard WES pipelines without functional assay; correlate with DLCN score + family history; functional LDL-binding assay confirms pathogenicity",
            "APOB-STATINS-CONTRAINDICATED-PREGNANCY: Statins are Category X in pregnancy — teratogenic; switch to bile acid sequestrant (cholestyramine) during pregnancy and breastfeeding; monitor LDL-C closely as pregnancy physiologically raises VLDL/LDL",
            "APOB-MILDER-THAN-LDLR: FDB clinical phenotype generally milder (LDL 200–300 mg/dL vs >300 mg/dL LDLR FH); tendon xanthomas less common; but ASCVD risk remains significantly elevated — treat aggressively to LDL targets",
            "APOB-CASCADE-TESTING-MANDATORY: AD inheritance — 50% first-degree relatives carry R3527Q; genetic testing cascade; lipid screening from age 10 in affected families",
        ],
        "etiologies": [
            "APOB R3527Q missense — receptor-binding domain Arg→Gln → impaired LDLR docking → reduced LDL clearance → LDL-C 200–350 mg/dL",
            "APOB R3527W missense — more severe variant than R3527Q; reduced binding affinity even more pronounced → higher LDL-C",
            "APOB splice variants (c.10708+1G>T) — aberrant mRNA processing → truncated apoB → impaired LDL formation and receptor binding",
        ],
        "stats": {
            "mean_dx_age": 38,
            "mean_dx_delay_months": 30,
            "ldl_c_untreated_mean_mgdL": 265,
            "statins_ldl_reduction_pct": 60,
            "r3527q_carrier_prevalence_per_1000": 1,
        },
        "dx_delay_distribution": "24–48 months (milder phenotype; often initially managed without genetic testing; LDL attributed to polygenic causes)",
    },

    # ── PCSK9 — FH3 GOF / Protective LOF ────────────────────────────────────
    {
        "gene": "PCSK9",
        "protein": "PCSK9 — FH3 GOF D374Y/S127R AD — Most Severe FH Phenotype — LOF R46L Y142X Protective 88pct CVD Reduction — Evolocumab Alirocumab FDA FOURIER ODYSSEY",
        "alias": (
            "PCSK9 (proprotein convertase subtilisin/kexin type 9); OMIM gene 607786; "
            "Hypercholesterolaemia Autosomal Dominant type 3 / FH3 OMIM 603776 (GOF). "
            "1p32.3; 692 aa; ~74 kDa; AD GOF (FH3) / LOF protective. "
            "FUNCTION: PCSK9 is a serine protease secreted primarily by the liver that binds to LDLR on hepatocyte cell surface. "
            "PCSK9-LDLR complex internalised → lysosomal degradation of LDLR (PCSK9 prevents LDLR recycling to cell surface); "
            "physiological role: PCSK9 suppresses LDLR surface density → limits LDL uptake → modulates LDL-C homeostasis. "
            "GOF PCSK9 → LDLR degraded faster → fewer LDLR at hepatocyte surface → LDL-C rises (FH3). "
            "LOF PCSK9 → LDLR recycled more efficiently → more LDL cleared → LDL-C falls (protective). "
            "GOF VARIANTS (FH3 — MOST SEVERE FH PHENOTYPE): "
            "D374Y (c.1120G>T) — PCSK9 binds LDLR with 5× higher affinity at physiological pH → "
            "LDLR degraded faster → LDL-C 250–500 mg/dL; "
            "S127R (c.381C>A) — autocatalytic cleavage enhancement → increased secreted PCSK9; "
            "GOF FH3 phenotype: highest LDL-C among all FH subtypes; "
            "presents with tendon xanthomas, premature ASCVD, cutaneous xanthomas. "
            "LOF VARIANTS (PROTECTIVE — DRUG TARGET RATIONALE): "
            "R46L (c.137G>T) — 1:50 in Northern Europeans; heterozygous carriers have LDL-C ~15 mg/dL lower; "
            "Y142X (c.426C>A) — predominantly West African populations; "
            "heterozygous LOF: 28% lower LDL-C; 88% reduction in 10-year CVD risk (Cohen 2006, NEJM); "
            "compound heterozygous/homozygous PCSK9 LOF: very low LDL-C (<30 mg/dL) — no adverse effects reported; "
            "this validated PCSK9 as the drug target for monoclonal antibodies (evolocumab, alirocumab). "
            "PCSK9 INHIBITOR MECHANISM: "
            "Evolocumab (Repatha®) / alirocumab (Praluent®): IgG2 / IgG1 monoclonal antibodies; "
            "bind circulating PCSK9 → prevent PCSK9-LDLR binding → LDLR recycled to surface → more LDL cleared; "
            "LDL-C reduction: 50–60% additional on top of statin; "
            "subcutaneous injection every 2–4 weeks (evolocumab) or 2 weeks (alirocumab). "
            "CLINICAL TRIALS: "
            "FOURIER (Sabatine 2017 NEJM): evolocumab + statin vs statin alone in established ASCVD; "
            "LDL-C 92→30 mg/dL; HR 0.85 for primary MACE (p<0.001); 20% reduction MI; "
            "ODYSSEY OUTCOMES (Schwartz 2018 NEJM): alirocumab post-ACS; 15% all-cause mortality reduction at 4 years; "
            "FOURIER-OLE (O'Donoghue 2022 Circulation): 5-year open-label extension — sustained LDL reduction; "
            "progressive MACE benefit with longer duration of treatment; no safety signal. "
            "INCLISIRAN (siRNA): Leqvio® — targets PCSK9 mRNA in hepatocytes → 50% LDL-C reduction; "
            "twice-yearly subcutaneous injection; ORION-10 (Ray 2020 NEJM). "
            "COMBINATION THERAPY: Statin + ezetimibe + PCSK9i → LDL-C <30 mg/dL achievable in FH3; "
            "lomitapide or LDL apheresis reserved for refractory GOF FH3."
        ),
        "locus": "1p32.3",
        "aa": 692,
        "kDa": 74,
        "omim_gene": "607786",
        "omim_disease": "Familial Hypercholesterolaemia type 3 / FH3 GOF (OMIM 603776); PCSK9 LOF variants are protective — NOT a disease state",
        "inheritance": "AD GOF (FH3 D374Y/S127R — heterozygous gain-of-function); LOF variants are autosomal dominant protective alleles",
        "gene_class": "Serine protease — LDLR degradation chaperone — secreted by liver — PCSK9-LDLR complex targets LDLR for lysosomal destruction — drug target for PCSK9 inhibitor antibodies",
        "key_alerts": [
            "PCSK9-GOF-MOST-SEVERE-FH: PCSK9 D374Y GOF binds LDLR with 5× higher affinity → rapid LDLR degradation → LDL-C 250–500 mg/dL — highest LDL among FH subtypes; requires PCSK9i + statin + ezetimibe combination; LDL apheresis if inadequate",
            "PCSK9-LOF-88PCT-CVD-REDUCTION: R46L/Y142X LOF heterozygotes — 28% lower LDL-C → 88% reduction 10-year CVD risk (Cohen 2006 NEJM); these individuals validated PCSK9 as drug target; homozygous LOF LDL <30 mg/dL — no adverse neurological effects despite very low LDL",
            "PCSK9I-LDL-50-60PCT-ADDITIONAL: Evolocumab/alirocumab bind circulating PCSK9 → prevent LDLR degradation → LDLR recycled → 50–60% additional LDL-C reduction ON TOP of statin; FOURIER: HR 0.85 MACE; ODYSSEY: 15% all-cause mortality reduction",
            "PCSK9-INCLISIRAN-TWICE-YEARLY: Inclisiran (siRNA) targets PCSK9 mRNA in hepatocytes → 50% LDL-C reduction with only 2 subcutaneous injections per year — adherence advantage over bimonthly PCSK9i antibodies",
            "PCSK9-VERY-LOW-LDL-SAFE: PCSK9 LOF homozygotes and patients on triple therapy reaching LDL <30 mg/dL show no adverse neurological/cognitive effects — very low LDL is safe; the lower the LDL in high-risk patients, the better the outcomes (FOURIER-OLE)",
        ],
        "etiologies": [
            "PCSK9 GOF D374Y (c.1120G>T) — PCSK9 binds LDLR 5× stronger at pH 5.4 (lysosomal) → rapid degradation → reduced surface LDLR → LDL-C 250–500 mg/dL",
            "PCSK9 GOF S127R — enhanced autocatalytic cleavage → increased secreted PCSK9 quantity → greater LDLR degradation rate",
            "PCSK9 LOF R46L (common European protective variant) — impaired PCSK9 autocatalytic prodomain cleavage → reduced secreted PCSK9 activity → more LDLR surface expression → lower LDL-C",
            "PCSK9 LOF Y142X (West African protective variant) — truncated non-secreted PCSK9 → no LDLR degradation → very low LDL-C in carriers",
        ],
        "stats": {
            "mean_dx_age": 30,
            "mean_dx_delay_months": 18,
            "ldl_c_untreated_gof_mean_mgdL": 380,
            "pcsk9i_ldl_reduction_pct": 55,
            "fourier_mace_hr": 0.85,
            "lof_cvd_risk_reduction_pct": 88,
        },
        "dx_delay_distribution": "12–24 months (GOF FH3 may present with extreme LDL and early CVD; LOF identified on incidental genetic testing or panels)",
    },

    # ── LDLRAP1 — ARH / Autosomal Recessive Hypercholesterolaemia ───────────
    {
        "gene": "LDLRAP1",
        "protein": "LDLRAP1 — ARH AR Biallelic — LDL-Receptor Adaptor Clathrin — Biallelic Required — Liver LDL Uptake Abolished — PCSK9i Surprisingly Effective via Extrahepatic LDLR",
        "alias": (
            "LDLRAP1 (low-density lipoprotein receptor adaptor protein 1 / ARH); OMIM gene 605747; "
            "Hypercholesterolaemia Autosomal Recessive (ARH) OMIM 603813. "
            "1p36.11; 308 aa; ~35 kDa; AR biallelic LOF. "
            "FUNCTION: LDLRAP1 (ARH) is an adaptor protein that links the LDL receptor cytoplasmic tail to clathrin-coated pits. "
            "LDLR has a YWTD EGF-like domain that must interact with clathrin assembly machinery for internalisation; "
            "ARH bridges LDLR FDNPVY motif → AP-2 adaptor complex → clathrin coat → endocytosis. "
            "Without ARH: LDLR cannot enter clathrin-coated pits in HEPATOCYTES → "
            "LDL is not internalised in liver (the primary LDL clearance organ) → LDL-C markedly elevated. "
            "IMPORTANTLY: ARH function is CELL-TYPE-SPECIFIC — "
            "in lymphocytes and fibroblasts, alternative adaptors (Dab2) compensate for absent ARH → "
            "LDLR internalisation still occurs in these cells; "
            "ARH function critical specifically in hepatocytes and macrophages. "
            "EPIDEMIOLOGY: Rare — predominantly consanguineous families; "
            "founder variants in Sardinian (W22X), Lebanese, and Iranian populations; "
            "phenotype: LDL-C 400–1,000 mg/dL untreated (similar to HoFH); "
            "xanthomas, coronary artery disease, aortic stenosis in young adults; "
            "often misdiagnosed as HoFH (clinical phenotype identical, genetic testing required to distinguish). "
            "CLINICAL DISTINCTIONS FROM HoFH: "
            "ARH: LDLR gene NORMAL (fully functional LDLR protein exists); heterozygous carriers (parents) typically UNAFFECTED; "
            "HoFH: LDLR mutant biallelic — parents are HeFH (elevated LDL); "
            "this distinction matters for cascade testing and treatment response prediction. "
            "TREATMENT RESPONSE — PCSK9i PARADOX: "
            "PCSK9 inhibitors SURPRISINGLY effective in ARH despite absent hepatic ARH-dependent LDLR internalisation: "
            "extrahepatic tissues (adipocytes, adrenals, lymphocytes) express Dab2 as ARH alternative → "
            "ARH-independent LDLR internalisation → PCSK9i prevents LDLR degradation in these tissues → "
            "some LDL clearance occurs; also residual hepatic ARH-independent pathway (Dab2 upregulation); "
            "clinical result: PCSK9i + statin reduces LDL-C 40–50% in ARH (less than HeFH but meaningful); "
            "LDL apheresis remains mainstay for ARH (as in HoFH) every 1–2 weeks. "
            "LOMITAPIDE: Reduces hepatic VLDL assembly → less LDL substrate; "
            "FDA-approved for HoFH — used off-label in ARH; hepatotoxicity monitoring mandatory."
        ),
        "locus": "1p36.11",
        "aa": 308,
        "kDa": 35,
        "omim_gene": "605747",
        "omim_disease": "Autosomal Recessive Hypercholesterolaemia (ARH) OMIM 603813",
        "inheritance": "AR — biallelic LOF required; heterozygous carriers (parents) typically unaffected (unlike HoFH parents who have HeFH)",
        "gene_class": "Clathrin adaptor protein — LDLR FDNPVY motif → AP-2 clathrin coat endocytosis — hepatocyte-specific (Dab2 compensates in lymphocytes/fibroblasts)",
        "key_alerts": [
            "LDLRAP1-PARENTS-UNAFFECTED-KEY-DDx: ARH parents are obligate heterozygous carriers but have NORMAL or near-normal LDL-C (unlike HoFH parents who both have HeFH); this is the key clinical clue to distinguish ARH from HoFH — genetic testing mandatory",
            "LDLRAP1-PCSK9I-PARADOX-EXTRAHEPATIC: PCSK9 inhibitors reduce LDL-C 40–50% in ARH despite absent hepatic ARH function — extrahepatic LDLR (using Dab2 adaptor) cleared by PCSK9i protection; clinically meaningful though less than HeFH response",
            "LDLRAP1-LDL-APHERESIS-MAINSTAY: LDL-C 400–1,000 mg/dL untreated (HoFH-equivalent severity); LDL apheresis every 1–2 weeks is standard of care; lomitapide off-label adjunct; treat to lowest achievable LDL-C",
            "LDLRAP1-HEPATOCYTE-SPECIFIC-MECHANISM: LDLR protein is NORMAL in ARH — the defect is LDLR internalisation in hepatocytes (requires ARH adaptor); in other cell types (lymphocytes), Dab2 substitutes; liver transplantation cures ARH (restores hepatic ARH-dependent LDLR function)",
            "LDLRAP1-CONSANGUINITY-CLUE: Predominantly consanguineous families; founder variants in Sardinian (W22X), Lebanese, Iranian populations; recessive inheritance + extreme LDL in child with unaffected parents → ARH must be in differential",
        ],
        "etiologies": [
            "LDLRAP1 W22X (Sardinian founder) — premature stop → absent ARH protein → hepatic LDLR cannot enter clathrin-coated pits → LDL-C 400–800 mg/dL",
            "LDLRAP1 biallelic missense (Lebanese founder variants) — impaired FDNPVY-AP2 interaction → partial loss hepatic LDLR internalisation",
            "LDLRAP1 compound heterozygous (novel population variants) — different mutations on each allele; consanguineous pedigree less likely for compound het",
        ],
        "stats": {
            "mean_dx_age": 18,
            "mean_dx_delay_months": 36,
            "ldl_c_untreated_mean_mgdL": 580,
            "pcsk9i_ldl_reduction_pct": 45,
            "apheresis_ldl_reduction_pct_per_session": 70,
        },
        "dx_delay_distribution": "24–60 months (misdiagnosed as HoFH; extreme LDL in child with unaffected parents often attributed to diet or laboratory error initially)",
    },

    # ── ABCG5 — Sitosterolemia type 1 ───────────────────────────────────────
    {
        "gene": "ABCG5",
        "protein": "ABCG5 — Sitosterolemia1 AR — Sterolin-1 Plant-Sterol Accumulation — Xanthomas Premature CVD Haemolysis — Ezetimibe CURATIVE — Low Plant-Sterol Diet",
        "alias": (
            "ABCG5 (ATP-binding cassette subfamily G member 5 / sterolin-1); OMIM gene 605459; "
            "Sitosterolemia type 1 (STSL1) OMIM 210250. "
            "2p21; 651 aa; ~75 kDa; AR biallelic. "
            "FUNCTION: ABCG5 forms an obligate heterodimer with ABCG8 (sterolin-2) at two critical sites: "
            "(1) Intestinal apical membrane (enterocytes): ABCG5/G8 pumps plant sterols BACK into the intestinal lumen → "
            "limits absorption of dietary plant sterols (sitosterol, campesterol, stigmasterol) and shellfish sterols; "
            "(2) Hepatocyte canalicular membrane (bile): ABCG5/G8 secretes plant sterols into bile → biliary excretion. "
            "WITHOUT ABCG5: Plant sterols absorbed normally (unchecked by ABCG5/G8 pump) → "
            "sterols accumulate in plasma, tissues, macrophages → xanthomas, atherosclerosis, haemolysis. "
            "CHOLESTEROL ABSORPTION: ABCG5/G8 also limits cholesterol absorption (less important effect — "
            "NPC1L1 is the primary cholesterol transporter, target of ezetimibe). "
            "CLINICAL PHENOTYPE: "
            "Tendon xanthomas at young age (first decade) with disproportionately NORMAL or only mildly elevated LDL-C — "
            "KEY distinguishing feature from FH (where LDL-C is always dramatically elevated); "
            "xanthelasma; interdigital xanthomas; elevated plasma sitosterol/campesterol levels (diagnostic); "
            "premature coronary artery disease (even in those with normal LDL-C); "
            "haemolytic anaemia with stomatocytes — plant sterols incorporate into red cell membranes → "
            "altered RBC membrane fluidity → haemolysis; "
            "thrombocytopenia; "
            "elevated liver enzymes; arthritis (plant sterol deposition in joints). "
            "DIAGNOSTIC TESTS: "
            "Plasma sitosterol (normal <1 mg/dL; in sitosterolemia >5 mg/dL and up to 20× normal); "
            "gas chromatography-mass spectrometry plant sterol panel; "
            "ABCG5/ABCG8 genetic testing; "
            "DO NOT MISTAKE for FH — treatment is completely different. "
            "TREATMENT — EZETIMIBE CURATIVE: "
            "Ezetimibe (NPC1L1 intestinal sterol transporter inhibitor): dramatically reduces plant sterol absorption → "
            "plasma sitosterol normalises → xanthomas regress → CVD risk markedly reduced; "
            "ezetimibe is the first-line treatment (and often CURATIVE in terms of normalising plasma sterols); "
            "statins — limited/no efficacy (plant sterols are NOT synthesised de novo like cholesterol — "
            "HMG-CoA reductase inhibition does not address the absorption/excretion defect); "
            "bile acid sequestrants (cholestyramine) second-line; "
            "LOW PLANT-STEROL DIET: avoid vegetable oils (high sitosterol), nuts, avocado, shellfish, plant margarines; "
            "diet alone insufficient but important adjunct."
        ),
        "locus": "2p21",
        "aa": 651,
        "kDa": 75,
        "omim_gene": "605459",
        "omim_disease": "Sitosterolemia type 1 (STSL1) OMIM 210250",
        "inheritance": "AR — biallelic LOF; ABCG5 and ABCG8 both at 2p21 (head-to-head gene pair); heterozygous carriers asymptomatic",
        "gene_class": "ABC half-transporter — obligate ABCG5/G8 heterodimer — intestinal apical + hepatic canalicular plant sterol efflux pump",
        "key_alerts": [
            "ABCG5-EZETIMIBE-CURATIVE: Ezetimibe (NPC1L1 blocker) dramatically reduces plant sterol absorption → plasma sitosterol normalises → xanthomas regress; ezetimibe is the preferred treatment for sitosterolemia — NOT statins (HMG-CoA inhibition does not address absorption/excretion defect)",
            "ABCG5-NORMAL-LDL-WITH-XANTHOMAS: Sitosterolemia xanthomas occur with NORMAL or only mildly elevated LDL-C — KEY diagnostic clue distinguishing from FH where LDL-C is always markedly elevated; measure plasma sitosterol/campesterol in any patient with tendon xanthomas + non-dramatically elevated LDL",
            "ABCG5-HAEMOLYSIS-STOMATOCYTES: Plant sterols incorporate into red cell membranes → haemolytic anaemia with stomatocytes; haemolysis may be the presenting feature in infants — measure plasma sterols in unexplained haemolysis with stomatocytosis",
            "ABCG5-STATINS-NOT-EFFECTIVE: Statins do not address the plant sterol accumulation defect (plant sterols are dietary, not endogenously synthesised) — primary treatment is ezetimibe + low-plant-sterol diet + bile acid sequestrants; statins may be added for any residual LDL-C elevation",
            "ABCG5-ABCG8-HEAD-TO-HEAD-PAIR: ABCG5 and ABCG8 are head-to-head tandem genes at 2p21; mutations in EITHER gene cause sitosterolemia (type 1 vs type 2); obligate heterodimer — loss of either subunit abolishes pump function; test BOTH genes simultaneously",
        ],
        "etiologies": [
            "ABCG5 biallelic nonsense/frameshift → absent sterolin-1 → no ABCG5/G8 heterodimer → unchecked intestinal plant sterol absorption + absent biliary excretion",
            "ABCG5 biallelic missense (Walker A/B ATP-binding domain) → impaired ATPase activity → non-functional efflux pump despite expressed protein",
            "ABCG5/ABCG8 compound heterozygous in trans (one mutation each gene) — only cis compound heterozygous (same gene) causes sitosterolemia; trans heterozygosity across ABCG5/G8 is NOT pathogenic",
        ],
        "stats": {
            "mean_dx_age": 12,
            "mean_dx_delay_months": 48,
            "plasma_sitosterol_normal_mgdL_max": 1.0,
            "plasma_sitosterol_stsl_mgdL_mean": 15,
            "ezetimibe_sitosterol_reduction_pct": 80,
        },
        "dx_delay_distribution": "36–72 months (misdiagnosed as FH; ezetimibe refusal because 'statins are standard'; diagnosis requires plant sterol assay not routinely ordered)",
    },

    # ── ABCG8 — Sitosterolemia type 2 ───────────────────────────────────────
    {
        "gene": "ABCG8",
        "protein": "ABCG8 — Sitosterolemia2 AR — Sterolin-2 More Common STSL Gene — Biliary Sterol Secretion Impaired — Tendon Xanthomas Young — Ezetimibe CURATIVE",
        "alias": (
            "ABCG8 (ATP-binding cassette subfamily G member 8 / sterolin-2); OMIM gene 605460; "
            "Sitosterolemia type 2 (STSL2) OMIM 618666. "
            "2p21; 673 aa; ~75 kDa; AR biallelic. "
            "FUNCTION: ABCG8 is the obligate heterodimeric partner of ABCG5. "
            "ABCG8 provides the second half-transporter for the ABCG5/G8 ABC transporter pump. "
            "Both ABCG5 and ABCG8 must be expressed and correctly folded for the pump to be active at "
            "intestinal apical membranes and hepatic canalicular membranes. "
            "ABCG8 variants: More commonly mutated in sitosterolemia than ABCG5 in some populations; "
            "D19H (c.55G>C) — common Asian variant (Japanese/Chinese) associated with gallstone susceptibility; "
            "T400K (c.1199C>A) — East Asian population variant. "
            "CLINICAL PHENOTYPE: Identical to ABCG5 sitosterolemia — "
            "tendon xanthomas in first decade of life; elevated plasma plant sterols (sitosterol, campesterol); "
            "premature atherosclerosis; haemolytic anaemia with stomatocytes; thrombocytopenia; arthritis; "
            "LDL-C may be NORMAL or mildly elevated (unlike FH where LDL-C is dramatically elevated). "
            "POPULATION GENETICS: "
            "ABCG8 variants responsible for majority of sitosterolemia cases in East Asian populations; "
            "ABCG5 variants more common in Middle Eastern/South Asian consanguineous families; "
            "in European populations: ABCG8 variants slightly more common than ABCG5; "
            "global prevalence of sitosterolemia (combined STSL1+STSL2): ~1:1,000,000 (rare); "
            "may be underdiagnosed as plant sterol assay not routinely performed. "
            "GALLSTONE LINK — ABCG8 D19H: "
            "Heterozygous ABCG8 D19H (common Asian variant) — impaired biliary cholesterol secretion → "
            "increased risk of cholesterol gallstones; NOT associated with sitosterolemia (heterozygous insufficient); "
            "homozygous D19H → sitosterolemia; "
            "ABCG8 is one of several susceptibility genes for gallstone disease (along with ABCG5). "
            "TREATMENT: "
            "EZETIMIBE FIRST-LINE — dramatically reduces plant sterol absorption (ezetimibe blocks NPC1L1 → "
            "NPC1L1 also transports plant sterols in addition to cholesterol → ezetimibe reduces BOTH); "
            "plasma sitosterol normalises within weeks of ezetimibe therapy; xanthomas regress; CVD risk reduces; "
            "bile acid sequestrants (cholestyramine) second-line; "
            "low plant-sterol diet (avoid vegetable oils, nuts, shellfish, plant margarines) — essential adjunct; "
            "liver transplantation cures sitosterolemia (restores ABCG5/G8 biliary excretion) — "
            "reserved for severe refractory cases with hepatic involvement."
        ),
        "locus": "2p21",
        "aa": 673,
        "kDa": 75,
        "omim_gene": "605460",
        "omim_disease": "Sitosterolemia type 2 (STSL2) OMIM 618666",
        "inheritance": "AR — biallelic LOF; head-to-head gene pair with ABCG5 at 2p21; D19H heterozygous = gallstone susceptibility (NOT sitosterolemia)",
        "gene_class": "ABC half-transporter — obligate ABCG5/G8 heterodimer — biliary cholesterol + plant sterol secretion — intestinal plant sterol efflux — NPC1L1-ezetimibe pathway interdependence",
        "key_alerts": [
            "ABCG8-EZETIMIBE-CURATIVE: Ezetimibe blocks NPC1L1 which transports plant sterols (as well as cholesterol) → dramatic reduction in plant sterol absorption → plasma sitosterol normalises → xanthomas regress; ezetimibe is as effective in STSL2 as STSL1",
            "ABCG8-D19H-GALLSTONES-NOT-SITOSTEROLEMIA: Heterozygous ABCG8 D19H (common Asian variant) → impaired biliary cholesterol secretion → gallstone risk; does NOT cause sitosterolemia (two copies needed); distinguish heterozygous D19H (gallstone risk) from homozygous D19H (sitosterolemia)",
            "ABCG8-EAST-ASIAN-POPULATION: ABCG8 mutations responsible for majority of sitosterolemia in East Asian populations; Japanese and Chinese cohort studies identify ABCG8 variants more commonly than ABCG5; sequence BOTH genes in any suspected sitosterolemia",
            "ABCG8-XANTHOMAS-NORMAL-LDL: Same as ABCG5 — tendon xanthomas in childhood with normal/mildly elevated LDL-C; plasma plant sterol assay (GC-MS) is diagnostic; do not attribute childhood xanthomas to 'diet' without investigating sitosterolemia",
            "ABCG8-LIVER-TRANSPLANT-CURATIVE: Liver transplantation restores hepatic ABCG5/G8 biliary plant sterol excretion — cures sitosterolemia; reserved for severe refractory cases with hepatic disease or failure; ezetimibe is curative medically in most cases",
        ],
        "etiologies": [
            "ABCG8 biallelic LOF (East Asian founder variants T400K, A540I) → no sterolin-2 → no ABCG5/G8 heterodimer → plant sterol accumulation",
            "ABCG8 homozygous D19H (c.55G>C) in East Asian populations → impaired biliary sterol secretion → sitosterolemia",
            "ABCG8 compound heterozygous European variants → disrupted ABC transporter → elevated plant sterols",
        ],
        "stats": {
            "mean_dx_age": 14,
            "mean_dx_delay_months": 42,
            "plasma_sitosterol_normal_mgdL_max": 1.0,
            "plasma_sitosterol_stsl_mgdL_mean": 18,
            "d19h_het_gallstone_risk_rr": 2.5,
        },
        "dx_delay_distribution": "30–60 months (often referred to lipid clinic for 'atypical FH'; plant sterol assay diagnostic; misdiagnosed as FH for years)",
    },

    # ── APOE — Familial Dysbetalipoproteinemia / Type III ───────────────────
    {
        "gene": "APOE",
        "protein": "APOE — FD Type-III-Hyperlipoproteinemia AR-like APOE2-E2 Homozygous — VLDL-Remnant Accumulation — Both-LDL-AND-TG-Elevated — Palmar-Xanthomas-PATHOGNOMONIC — Fibrates-Statins",
        "alias": (
            "APOE (apolipoprotein E); OMIM gene 107741; "
            "Familial Dysbetalipoproteinemia (FD) / Type III Hyperlipoproteinemia OMIM 617347. "
            "19q13.32; 299 aa; ~34 kDa; isoform-determined (APOE2/E2 effectively AR; APOE3/E4 dominant modifications). "
            "ISOFORMS: APOE2 (Cys112, Cys158), APOE3 (Cys112, Arg158), APOE4 (Arg112, Arg158). "
            "FUNCTION: Apolipoprotein E is the major apoprotein of VLDL, HDL, and chylomicron remnants. "
            "APOE mediates hepatic receptor-mediated clearance of remnant lipoproteins (IDL, VLDL remnants, chylomicron remnants) "
            "via LDLR and LRP1 (LDLR-related protein 1). "
            "APOE isoform-specific LDLR binding: "
            "APOE3 — normal LDLR binding (reference isoform); "
            "APOE4 — enhanced LDLR binding; "
            "APOE2 — severely impaired LDLR binding (~1% of APOE3 affinity) — Cys158 → Arg158 structural change "
            "disrupts LDLR receptor-binding domain. "
            "APOE2/E2 HOMOZYGOSITY (ESSENTIAL PREREQUISITE): "
            "APOE2/E2 prevalence in European populations: ~1%; "
            "90% of APOE2/E2 individuals are PROTECTED — paradoxically lower LDL-C and CVD risk "
            "(APOE2 impairs VLDL assembly → less LDL substrate); "
            "10% of APOE2/E2 develop FD — requires a SECOND HIT: "
            "obesity, type 2 diabetes, hypothyroidism, alcohol excess, post-menopausal oestrogen deficiency, "
            "renal failure, or another genetic dyslipidaemia (compound with LDLR mutation, APOB variant). "
            "PATHOPHYSIOLOGY: APOE2 → impaired VLDL remnant/IDL clearance via hepatic receptors → "
            "VLDL remnants (β-VLDL) accumulate → BOTH triglycerides AND LDL-C elevated (mixed hyperlipidaemia); "
            "total cholesterol:TG ratio characteristically 1:1 (in most other hyperlipoproteinaemias, one is more elevated). "
            "CLINICAL PHENOTYPE: "
            "Xanthelasma + tendon xanthomas (HDL particles carry APOE2 → macrophage uptake); "
            "PALMAR (TUBERO-ERUPTIVE) XANTHOMAS — yellow deposits in the palmar skin creases — "
            "PATHOGNOMONIC for FD (not seen in other primary dyslipidaemias); "
            "premature peripheral arterial disease (PAD) — disproportionate risk vs other dyslipidaemias; "
            "premature coronary artery disease. "
            "LIPID PROFILE: Total cholesterol elevated; TG elevated (often 300–800 mg/dL); "
            "LDL-C often SPURIOUSLY LOW by Friedewald formula (cannot calculate accurately in hypertriglyceridaemia); "
            "β-VLDL detectable by agarose lipoprotein electrophoresis (broad β band) or VLDL-C:TG >0.69 ratio. "
            "TREATMENT: "
            "TREAT THE SECOND HIT FIRST: weight loss, glycaemic control (T2DM), thyroid replacement (hypothyroidism), "
            "alcohol cessation → often normalises lipids in FD without pharmacotherapy; "
            "FIBRATES FIRST-LINE PHARMACOTHERAPY: Fenofibrate/gemfibrozil — activate PPARα → increased lipoprotein lipase → "
            "VLDL-TG hydrolysis → reduced remnant load → LDL-C and TG both fall; "
            "STATINS: Also effective (LDLR upregulation → remnant clearance via LDLR) — "
            "can be used instead of or in combination with fibrates; "
            "STATIN + FIBRATE COMBINATION: Higher risk of myopathy (gemfibrozil > fenofibrate with statins — "
            "gemfibrozil inhibits glucuronidation of statins → increased statin plasma levels; "
            "prefer fenofibrate if combination needed)."
        ),
        "locus": "19q13.32",
        "aa": 299,
        "kDa": 34,
        "omim_gene": "107741",
        "omim_disease": "Familial Dysbetalipoproteinemia / Type III Hyperlipoproteinemia (OMIM 617347)",
        "inheritance": "AR-like — APOE2/E2 homozygosity required (necessary but not sufficient — second metabolic hit needed in 90% of APOE2/E2 for phenotype expression); APOE4 associated with Alzheimer's risk (separate phenotype)",
        "gene_class": "Remnant lipoprotein apoprotein — LDLR/LRP1 receptor-binding — VLDL/IDL/chylomicron remnant clearance — isoform Cys112/Arg158 determines LDLR binding affinity",
        "key_alerts": [
            "APOE-PALMAR-XANTHOMAS-PATHOGNOMONIC: Yellow deposits in palmar skin creases (tubero-eruptive/palmar xanthomas) are PATHOGNOMONIC for Familial Dysbetalipoproteinemia — not seen in other primary dyslipidaemias; diagnose FD clinically without lipid panel in any patient with these xanthomas",
            "APOE-SECOND-HIT-REQUIRED: 90% of APOE2/E2 homozygotes do NOT develop FD — they paradoxically have lower CVD risk; FD requires a second metabolic hit (obesity, T2DM, hypothyroidism, alcohol, renal failure); treat the second hit first (weight loss, glycaemic control, thyroid replacement) — lipids may normalise without drugs",
            "APOE-MIXED-HYPERLIPIDAEMIA-BOTH-TG-AND-LDL: FD raises BOTH TG AND LDL-C (unlike pure hypercholesterolaemia or pure hypertriglyceridaemia); LDL-C calculated by Friedewald is INACCURATE when TG >400 mg/dL — use direct LDL measurement or beta-quantification",
            "APOE-FIBRATES-FIRST-LINE: Fibrates (fenofibrate) activate PPARα → lipoprotein lipase → VLDL-remnant clearance — first-line for FD; gemfibrozil + statin combination risks myopathy (gemfibrozil inhibits statin glucuronidation → elevated statin levels); prefer fenofibrate if combination needed",
            "APOE-PAD-RISK-DISPROPORTIONATE: FD causes disproportionately high peripheral arterial disease risk vs coronary artery disease compared with other dyslipidaemias; β-VLDL remnants have strong affinity for arterial wall macrophages → peripheral atherosclerosis; assess ABI in all FD patients",
        ],
        "etiologies": [
            "APOE2/E2 homozygous + obesity/insulin resistance (most common second hit) → impaired VLDL remnant clearance + insulin stimulation of hepatic VLDL production → florid FD",
            "APOE2/E2 + hypothyroidism → thyroid hormone regulates LDLR expression; hypothyroidism downregulates LDLR → worsens APOE2-mediated remnant clearance defect",
            "APOE2/E2 + post-menopausal oestrogen deficiency → oestrogen normally upregulates LDLR and LPL → oestrogen loss → FD phenotype unmasked",
            "APOE2/E2 + compound LDLR heterozygous mutation → double hit on remnant clearance pathway → severe FD + FH phenotype",
        ],
        "stats": {
            "mean_dx_age": 45,
            "mean_dx_delay_months": 36,
            "apoe2_e2_prevalence_pct": 1,
            "apoe2_e2_fd_penetrance_pct": 10,
            "fibrate_ldl_reduction_pct": 35,
            "fibrate_tg_reduction_pct": 50,
        },
        "dx_delay_distribution": "24–60 months (mixed hyperlipidaemia treated empirically with statin; palmar xanthomas not recognised; APOE genotyping not routinely done for mixed dyslipidaemia)",
    },

    # ── LPA — Lipoprotein(a) elevation ──────────────────────────────────────
    {
        "gene": "LPA",
        "protein": "LPA — Lp(a) Elevation Co-Dominant Apo(a)-Size-Polymorphism — Independent-CVD-Risk-Factor >50mg/dL — No-FDA-Approved-Rx-Yet Pelacarsen-Phase3 HORIZON — Aspirin-Controversial — PCSK9i-Modest-25pct",
        "alias": (
            "LPA (lipoprotein(a) / apo(a)); OMIM gene 152200; "
            "Elevated Lp(a) — cardiovascular risk OMIM 152200. "
            "6q25–q26; 4529 aa (isoform-dependent, kringle-IV repeats vary); isoform-determined kDa (200–900 kDa range); "
            "co-dominant inheritance — apo(a) isoform size (number of KIV-2 repeats) co-dominantly determines Lp(a) level. "
            "STRUCTURE: Lp(a) = LDL particle + apo(a) disulfide-linked to apoB-100. "
            "Apo(a) contains multiple kringle-IV (KIV) and kringle-V (KV) domains homologous to plasminogen. "
            "KIV-2 COPY NUMBER POLYMORPHISM: Number of KIV-2 repeats inversely determines Lp(a) level: "
            "few KIV-2 repeats → smaller apo(a) → more efficiently secreted → HIGH Lp(a); "
            "many KIV-2 repeats → larger apo(a) → less efficiently secreted → LOW Lp(a). "
            "This inverse relationship is the major genetic determinant of Lp(a) levels; "
            "Lp(a) level ~80% heritable — one of the most heritable cardiovascular risk factors. "
            "EPIDEMIOLOGY: "
            "Lp(a) >50 mg/dL in ~20–25% of the population; "
            "Lp(a) >30 mg/dL associated with increased CVD risk; "
            "Lp(a) >150 mg/dL (>500 nmol/L) in ~5% of population — very high risk; "
            "Lp(a) levels are 2–3× higher in people of African descent than European populations. "
            "CARDIOVASCULAR RISK MECHANISMS: "
            "Lp(a) delivers LDL lipids to vessel wall (like LDL but with additional pathogenic properties); "
            "oxidised phospholipids (OxPL) carried by Lp(a) → inflammatory signalling in arterial wall → plaque growth; "
            "apo(a) KIV-10 domain inhibits plasminogen activation → impaired fibrinolysis → thrombosis; "
            "aortic valve calcification risk (independent of LDL-C); "
            "risk of MI, stroke, and peripheral arterial disease all increased. "
            "RISK THRESHOLDS: "
            "EAS/ACC: Measure Lp(a) once in all adults for lifetime CVD risk stratification; "
            ">50 mg/dL (>125 nmol/L) = elevated risk; "
            ">150 mg/dL (>500 nmol/L) = very high risk equivalent to FH; "
            "Lp(a) levels do NOT change substantially with diet or exercise (unlike LDL-C). "
            "TREATMENT — NO FDA-APPROVED Lp(a)-LOWERING THERAPY YET (2024): "
            "PCSK9 inhibitors: modest ~25% Lp(a) reduction (mechanism unclear — possibly increased hepatic clearance); "
            "NOT sufficient for very high-risk Lp(a) patients; "
            "NIACIN: 20–30% Lp(a) reduction but abandoned due to lack of CV outcomes benefit (HPS2-THRIVE); "
            "LDL APHERESIS: 60–70% Lp(a) reduction per session — FDA-approved for Lp(a) >60 mg/dL + ASCVD; "
            "NOVEL THERAPIES IN TRIALS: "
            "Pelacarsen (TQJ230) — GalNAc-conjugated antisense oligonucleotide targeting LPA mRNA in hepatocytes → "
            "~80% Lp(a) reduction; HORIZON Phase 3 trial (Tsimikas 2020 NEJM Phase 2 data); "
            "Olpasiran (AMG890) — siRNA targeting LPA → >90% Lp(a) reduction (Phase 2 OCEAN trial); "
            "Lepodisiran (LY3819469) — siRNA Phase 2. "
            "ASPIRIN — CONTROVERSIAL: Apo(a) antifibrinolytic effect (KIV-10 domain inhibits plasminogen) → "
            "theoretical benefit of aspirin for thrombotic risk; "
            "no dedicated RCT in elevated Lp(a) population; current guidelines: aspirin not recommended specifically for Lp(a) elevation; "
            "individual ASCVD risk assessment guides aspirin decision. "
            "MANAGEMENT PRINCIPLE: Focus on aggressively reducing OTHER modifiable risk factors (LDL-C, hypertension, diabetes, smoking) "
            "while awaiting approved Lp(a)-lowering therapy."
        ),
        "locus": "6q25-q26",
        "aa": 4529,
        "kDa": 600,
        "omim_gene": "152200",
        "omim_disease": "Elevated Lipoprotein(a) — cardiovascular risk factor (OMIM 152200); not a monogenic disease — polygenic/co-dominant quantitative trait",
        "inheritance": "Co-dominant (apo(a) KIV-2 kringle repeat copy number inversely determines Lp(a) level); ~80% heritable; both alleles contribute additively; African descent populations have systematically higher Lp(a)",
        "gene_class": "Lipoprotein apo(a) — disulfide-linked to apoB-100 on LDL particle — kringle-IV KIV-2 copy number polymorphism — apo(a) homologous to plasminogen (antifibrinolytic) — OxPL carrier — aortic valve calcification risk",
        "key_alerts": [
            "LPA-MEASURE-ONCE-LIFETIME: EAS/ACC recommend Lp(a) measured once in all adults for lifetime CVD risk assessment; Lp(a) does not change with diet or exercise (unlike LDL-C); if >50 mg/dL, intensify management of all OTHER modifiable risk factors aggressively",
            "LPA-NO-FDA-APPROVED-RX-YET: As of 2024, no FDA-approved Lp(a)-lowering therapy; pelacarsen (80% reduction, HORIZON Phase 3) and olpasiran (>90% reduction) in late-stage trials; interim: PCSK9i modest 25% reduction; LDL apheresis FDA-approved for Lp(a) >60 mg/dL + ASCVD",
            "LPA-VERY-HIGH-RISK-THRESHOLD: Lp(a) >150 mg/dL (>500 nmol/L) in ~5% of population = cardiovascular risk equivalent to heterozygous FH; aortic valve calcification risk markedly elevated; refer to lipid specialist; consider LDL apheresis",
            "LPA-AFRICAN-DESCENT-2-3X-HIGHER: Lp(a) levels systematically 2–3× higher in people of African descent vs European populations; African ancestry patients with 'borderline' Lp(a) may actually be at high absolute risk — interpret in population context",
            "LPA-ASPIRIN-NOT-INDICATED-SPECIFICALLY: Despite apo(a) antifibrinolytic KIV-10 domain, aspirin is NOT recommended specifically for Lp(a) elevation without other ASCVD indications; decisions based on individual absolute CVD risk; HPS2-THRIVE showed niacin (which reduces Lp(a)) did not improve outcomes",
        ],
        "etiologies": [
            "Low KIV-2 repeat count (few repeats) → small apo(a) isoform efficiently secreted → high plasma Lp(a); inversely: many repeats → large apo(a) → poor secretion → low Lp(a)",
            "Population-specific LPA regulatory variants (SNPs in LPA promoter region, common rs10455872, rs3798220) → elevated Lp(a) independent of apo(a) size",
            "rs3798220 (I4399M) in kringle-IV type 2 domain — associated with elevated Lp(a) and increased cardiovascular risk; 2–4× odds ratio for MI in carriers",
        ],
        "stats": {
            "mean_dx_age": 42,
            "mean_dx_delay_months": 60,
            "lpa_elevated_pct_population": 22,
            "lpa_very_high_pct_population": 5,
            "pcsk9i_lpa_reduction_pct": 25,
            "ldl_apheresis_lpa_reduction_pct": 65,
            "pelacarsen_lpa_reduction_pct": 80,
        },
        "dx_delay_distribution": "48–84 months (Lp(a) not measured routinely; CVD events attributed only to LDL-C; Lp(a) testing not on standard lipid panel; awareness low among primary care physicians)",
    },
]


def _generate_patients():
    """Generate 40 patients per gene using deterministic RNG (seed = SEED_BASE + gene_idx)."""
    for idx, gene in enumerate(HYPERLIPIDEMIA_GENES):
        rng = random.Random(SEED_BASE + idx)
        patients = []
        for i in range(40):
            age_at_dx = rng.randint(8, 65)
            dx_delay = rng.randint(6, 84)
            # Gene-specific LDL-C ranges
            if gene["gene"] == "LDLR":
                ldl_c = rng.randint(190, 500)
                tg = rng.randint(80, 200)
            elif gene["gene"] == "APOB":
                ldl_c = rng.randint(180, 350)
                tg = rng.randint(100, 250)
            elif gene["gene"] == "PCSK9":
                ldl_c = rng.randint(220, 600)
                tg = rng.randint(100, 300)
            elif gene["gene"] == "LDLRAP1":
                ldl_c = rng.randint(400, 900)
                tg = rng.randint(150, 350)
            elif gene["gene"] in ["ABCG5", "ABCG8"]:
                ldl_c = rng.randint(80, 250)  # often normal or mildly elevated
                tg = rng.randint(80, 200)
            elif gene["gene"] == "APOE":
                ldl_c = rng.randint(150, 400)
                tg = rng.randint(300, 900)  # markedly elevated TG
            else:  # LPA
                ldl_c = rng.randint(90, 180)  # LDL-C normal (Lp(a) is separate)
                tg = rng.randint(80, 200)

            lpa = rng.randint(50, 300) if gene["gene"] == "LPA" else rng.randint(5, 60)
            on_statin = rng.random() > 0.3
            on_pcsk9i = rng.random() > 0.7 if gene["gene"] in ["LDLR", "PCSK9"] else rng.random() > 0.85
            on_ezetimibe = rng.random() > 0.5
            xanthomas = rng.random() > 0.6 if gene["gene"] in ["LDLR", "PCSK9", "LDLRAP1"] else (
                rng.random() > 0.5 if gene["gene"] in ["ABCG5", "ABCG8"] else rng.random() > 0.85)
            ascvd_event = rng.random() > 0.75 if age_at_dx > 40 else rng.random() > 0.9
            patients.append({
                "patient_id": f"{gene['gene']}-{i+1:03d}",
                "age_at_dx": age_at_dx,
                "dx_delay_months": dx_delay,
                "ldl_c_untreated_mgdL": ldl_c,
                "tg_mgdL": tg,
                "lpa_mgdL": lpa,
                "on_statin": on_statin,
                "on_pcsk9i": on_pcsk9i,
                "on_ezetimibe": on_ezetimibe,
                "xanthomas_present": xanthomas,
                "ascvd_event_prior": ascvd_event,
                "gene": gene["gene"],
                "seed": SEED_BASE + idx,
            })
        gene["patients"] = patients


_generate_patients()


def get_overview():
    all_ages = [p["age_at_dx"] for g in HYPERLIPIDEMIA_GENES for p in g["patients"]]
    all_delays = [p["dx_delay_months"] for g in HYPERLIPIDEMIA_GENES for p in g["patients"]]
    genes = []
    for idx, g in enumerate(HYPERLIPIDEMIA_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
        genes.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "mean_dx_age": round(sum(ages) / len(ages), 1),
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
            "key_alerts": g["key_alerts"],
            "n_patients": len(g["patients"]),
        })
    return {
        "atlas": "Hereditary-Hyperlipidemia-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Hyperlipidemia Atlas — Monogenic Dyslipidaemia (FH/FDB/FH3/ARH/Sitosterolemia/FD/Lp(a))",
        "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
        "total_patients": sum(len(g["patients"]) for g in HYPERLIPIDEMIA_GENES),
        "aggregate_stats": {
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "genes_covered": len(HYPERLIPIDEMIA_GENES),
            "patients_per_gene": 40,
        },
        "genes": genes,
        "top_alerts": [
            "LDLR-STATINS-REQUIRE-LDLR: Statins upregulate remaining LDLR to increase LDL clearance — limited efficacy in HoFH biallelic null; HoFH standard of care = LDL apheresis every 1–2 weeks + lomitapide/PCSK9i; cascade test all first-degree relatives from age 2–5 years",
            "PCSK9-LOF-88PCT-CVD-REDUCTION: PCSK9 R46L/Y142X LOF → 88% reduction 10-year CVD risk (Cohen 2006 NEJM); validated drug target; evolocumab FOURIER HR 0.85 MACE; very low LDL-C (<30 mg/dL) is SAFE — lower is better in established ASCVD",
            "ABCG5-ABCG8-EZETIMIBE-CURATIVE: Sitosterolemia (STSL1/STSL2) — ezetimibe is curative (normalises plasma plant sterols); xanthomas with NORMAL or mildly elevated LDL-C = measure plasma sitosterol; statins are ineffective for the plant sterol accumulation defect",
            "APOE-PALMAR-XANTHOMAS-PATHOGNOMONIC: Yellow palmar skin crease deposits = Familial Dysbetalipoproteinemia until proven otherwise; APOE2/E2 + second hit (obesity/T2DM/hypothyroidism); treat second hit first — lipids may normalise; fibrates first-line pharmacotherapy",
            "LPA-MEASURE-ONCE-NO-APPROVED-RX: Lp(a) >50 mg/dL in 20–25% of population; measure once in all adults; no FDA-approved Lp(a)-lowering therapy yet (pelacarsen HORIZON Phase 3); Lp(a) does not respond to diet/exercise — intensify all other CVD risk factors aggressively",
            "LDLRAP1-PARENTS-UNAFFECTED-DDx-HOFD: ARH parents have normal LDL-C (unlike HoFH parents who are HeFH); AR inheritance with severe hypercholesterolaemia in child + unaffected parents = ARH vs HoFH — genetic testing mandatory to distinguish; PCSK9i effective in ARH via extrahepatic LDLR",
            "APOB-STATINS-EFFECTIVE-FDB: Statins MORE effective in FDB than LDLR FH — LDLR is NORMAL in FDB (apoB R3527Q reduces receptor binding but LDLR protein is intact); statin-upregulated LDLR can still bind apoB-R3527Q LDL; statins usually sufficient to reach LDL targets in FDB",
            "LIPID-LOWERING-PREGNANCY: Statins CONTRAINDICATED in pregnancy (Category X teratogenic); bile acid sequestrants (cholestyramine) safe alternative; ezetimibe not recommended in pregnancy (limited safety data); PCSK9i insufficient safety data; FH management in pregnancy requires specialist input",
        ],
    }


def get_breakdown():
    result = []
    for idx, g in enumerate(HYPERLIPIDEMIA_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "key_alerts": g["key_alerts"],
            "etiologies": g["etiologies"],
            "stats": g["stats"],
            "dx_delay_distribution": g["dx_delay_distribution"],
            "computed": {
                "mean_dx_age": round(sum(ages) / len(ages), 1),
                "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def get_definitions():
    return {
        "concepts": {
            "LDL Receptor Pathway — Statin Mechanism and FH Treatment Rationale": (
                "The LDL receptor (LDLR) pathway is the primary mechanism for plasma LDL-C clearance. "
                "HEPATIC LDL UPTAKE: LDL circulates → apoB-100 binds LDLR on hepatocyte surface → "
                "LDLR-LDL complex in clathrin-coated pit → endocytosis → lysosomal LDL degradation → "
                "cholesterol released for bile acid synthesis, VLDL assembly, membranes; "
                "LDLR recycled to surface (PCSK9-independent pathway). "
                "SREBP2 FEEDBACK: When hepatocyte cholesterol is sufficient → SCAP/INSIG retains SREBP2 in ER; "
                "when cholesterol falls (statin inhibits HMG-CoA reductase → less cholesterol made) → "
                "SCAP escorts SREBP2 to Golgi → S1P/S2P cleave SREBP2 → nuclear SREBP2 → "
                "transcribes LDLR gene → more LDLR at hepatocyte surface → MORE LDL cleared. "
                "STATIN MECHANISM IN FH: Statins reduce hepatocyte cholesterol → SREBP2 upregulates residual functional LDLR; "
                "in HeFH (one normal LDLR allele) → residual LDLR upregulated → substantial LDL reduction; "
                "in HoFH biallelic null → no LDLR to upregulate → statins minimally effective on LDL; "
                "in FDB (APOB mutation — LDLR normal) → statin-upregulated normal LDLR can still bind R3527Q apoB (low affinity but still some clearance); "
                "statins MOST effective in FDB > HeFH; statins LEAST effective in HoFH biallelic null. "
                "PCSK9 ROLE IN LDLR DEGRADATION: After LDLR-LDL internalisation → LDLR must be recycled to surface; "
                "PCSK9 binds LDLR in early endosome → targets LDLR for lysosomal degradation (not recycled); "
                "PCSK9 inhibition → more LDLR recycled → more LDL cleared per hepatocyte — "
                "additive to statin (statin makes more LDLR; PCSK9i preserves more LDLR). "
                "COMBINATION THERAPY RATIONALE: Statin (more LDLR) + ezetimibe (less cholesterol absorbed → SREBP2 activated → even more LDLR) + PCSK9i (LDLR recycled rather than degraded) → synergistic LDL reduction."
            ),
            "PCSK9 Biology — From Discovery to Drug Target Validation": (
                "PCSK9 was discovered as a cause of autosomal dominant hypercholesterolaemia (2003, Abifadel HGNC). "
                "STRUCTURE: Prodomain (auto-inhibitory) + catalytic domain (serine protease) + C-terminal domain. "
                "AUTOCATALYTIC CLEAVAGE: PCSK9 cleaves its own prodomain intramolecularly in the ER; "
                "prodomain remains non-covalently associated → inhibits catalytic activity → PCSK9 secreted as inactive complex. "
                "PCSK9-LDLR BINDING: In acidic endosome (pH 5.4): PCSK9-LDLR binding strengthens → "
                "LDLR cannot release PCSK9 → PCSK9 prevents LDLR from adopting the conformation required for recycling → "
                "LDLR sent to lysosome for degradation. "
                "GOF (D374Y): Asp374Tyr in catalytic domain → PCSK9 binds LDLR with 10× increased affinity at neutral pH and 5× more at pH 5.4 → "
                "LDLR degraded faster → surface LDLR reduced → LDL rises. "
                "LOF VALIDATION (Cohen 2006 NEJM — Dallas Heart Study): "
                "African American subjects with LOF PCSK9 variants (Y142X, C679X) → LDL-C 28% lower → "
                "88% reduction in 10-year Framingham CVD risk — "
                "this proved that lifelong low PCSK9/low LDL-C is safe and cardiovascularly protective → "
                "validated PCSK9 as drug target (therapeutic LOF mimicry). "
                "DRUG DEVELOPMENT TIMELINE: 2003 GOF discovery → 2006 LOF validation → 2012 Phase 1 PCSK9 antibodies → "
                "2015 FDA approval evolocumab/alirocumab → 2017 FOURIER outcomes trial → "
                "2020 inclisiran (siRNA, twice-yearly). "
                "FUTURE: Small molecule PCSK9 inhibitors (PCSK9 synthesis inhibitors, oral) in development; "
                "gene editing approaches (CRISPR Cas9 hepatic PCSK9 knockout) — VERVE-101 Phase 1b trial."
            ),
            "Sitosterolemia — Diagnosis, Plant Sterol Biology, and Ezetimibe Mechanism": (
                "Sitosterolemia (phytosterolemia) is caused by biallelic LOF in ABCG5 or ABCG8 — "
                "the intestinal/biliary plant sterol efflux pump. "
                "PLANT STEROL BIOLOGY: Dietary plant sterols (sitosterol, campesterol, stigmasterol, brassicasterol) and shellfish sterols (stanols) "
                "are structurally similar to cholesterol but NOT synthesised by humans. "
                "NORMAL ABSORPTION: Intestinal NPC1L1 absorbs some dietary plant sterols; "
                "ABCG5/G8 at enterocyte apical membrane pumps most absorbed plant sterols BACK into lumen → "
                "net absorption <5% of dietary plant sterols (vs ~50% for cholesterol); "
                "any absorbed plant sterols → hepatocyte ABCG5/G8 secretes them into bile → biliary excretion. "
                "SITOSTEROLEMIA: ABCG5/G8 absent → plant sterol absorption increases to 20–30% → "
                "biliary excretion impaired → sitosterol accumulates in plasma (5–30 mg/dL; normal <1 mg/dL); "
                "plant sterols deposit in tendons (xanthomas), arterial walls (atherosclerosis), "
                "RBC membranes (haemolysis), synovium (arthritis). "
                "EZETIMIBE MECHANISM (relevance to sitosterolemia): "
                "Ezetimibe binds and inhibits NPC1L1 (Niemann-Pick C1-like 1) at enterocyte brush border membrane; "
                "NPC1L1 transports both cholesterol AND plant sterols into enterocytes; "
                "ezetimibe blocks NPC1L1 → reduces plant sterol absorption dramatically → "
                "plasma sitosterol normalises within weeks → xanthomas regress; "
                "this is the mechanism of ezetimibe's CURATIVE effect in sitosterolemia; "
                "ezetimibe also reduces cholesterol absorption (its approved indication for hypercholesterolaemia). "
                "WHY STATINS FAIL IN SITOSTEROLEMIA: "
                "HMG-CoA reductase inhibition reduces ENDOGENOUS cholesterol synthesis; "
                "plant sterols are EXOGENOUS (dietary) — not synthesised by HMG-CoA reductase; "
                "statins do not address the absorption/excretion defect for plant sterols; "
                "statins may slightly reduce LDL-C (if elevated) via LDLR upregulation but do not reduce plant sterol burden."
            ),
            "Lp(a) — Structure, Cardiovascular Risk, and Emerging Therapeutics": (
                "Lp(a) is an LDL-like particle with apo(a) disulfide-linked to apoB-100. "
                "APO(a) STRUCTURE: Domains: KIV-1 (one copy), KIV-2 (variable repeats — 2–40+), "
                "KIV-3 to KIV-10 (one copy each), KV (one copy), protease domain (serine protease inactive). "
                "KIV-2 POLYMORPHISM: The number of KIV-2 tandem repeat copies is inversely correlated with Lp(a) level: "
                "fewer KIV-2 → smaller apo(a) → more efficiently secreted → higher Lp(a); "
                "this is genetically determined and largely invariant within an individual. "
                "CARDIOVASCULAR MECHANISMS: "
                "(1) OxPL (oxidised phospholipids) enriched on Lp(a) → macrophage activation → "
                "inflammatory signalling in vascular wall → accelerated atherosclerosis; "
                "(2) Antifibrinolytic: KIV-10 domain structurally homologous to plasminogen kringle domain → "
                "competes with plasminogen for fibrin binding → impaired clot dissolution → thrombosis; "
                "(3) Aortic valve calcification: Lp(a)-associated OxPL drives valve interstitial cell calcification → "
                "aortic stenosis risk 2–3× elevated in Lp(a) >50 mg/dL; "
                "(4) LDL cholesterol delivery to arterial wall (as with standard LDL). "
                "EMERGING RNA THERAPEUTICS: "
                "Pelacarsen (IONIS-APO(a)-LRx / TQJ230): GalNAc-conjugated antisense oligonucleotide (ASO); "
                "GalNAc targets asialoglycoprotein receptor on hepatocytes → selective hepatic uptake → "
                "blocks LPA mRNA translation → 80% Lp(a) reduction; "
                "monthly subcutaneous injection; HORIZON Phase 3 trial cardiovascular outcomes (Novartis/Ionis); "
                "Olpasiran (AMG890): GalNAc-siRNA targeting LPA mRNA → >90% Lp(a) reduction; "
                "quarterly subcutaneous injection; OCEAN Phase 2 (Amgen, Nicholls 2022 NEJM); "
                "Lepodisiran (LY3819469): siRNA; 6-monthly injection; Phase 2 (Eli Lilly). "
                "MEASUREMENT: Lp(a) measured in mg/dL or nmol/L; "
                "nmol/L preferred (not affected by apo(a) isoform mass differences); "
                "conversion: 1 mg/dL ≈ 2.5 nmol/L (approximate — depends on apo(a) isoform); "
                "Lp(a) not included in standard lipid panel — must be specifically ordered."
            ),
            "Familial Dysbetalipoproteinemia — APOE2/E2 and the Second-Hit Model": (
                "Familial Dysbetalipoproteinemia (FD / Type III Hyperlipoproteinemia) illustrates how a genotype "
                "(APOE2/E2) requires a metabolic second hit for disease expression. "
                "APOE ISOFORMS AND RECEPTOR BINDING: "
                "APOE3 (reference): Arg158 → normal LDLR binding → normal remnant clearance; "
                "APOE2 (Cys158): Arg158→Cys158 → LDLR binding <1% of APOE3 affinity → impaired remnant clearance; "
                "APOE4 (Arg112): Arg112 enhances LDLR binding preference → faster LDLR occupation by VLDL → "
                "less LDLR available for IDL/LDL (paradoxically raises LDL in some) + Alzheimer's risk. "
                "APOE2/E2 PARADOX: "
                "APOE2/E2 homozygotes (~1% of Europeans) have LOWER LDL-C on average than APOE3/E3 → "
                "APOE2 impairs apoB secretion into VLDL → less VLDL → less LDL substrate; "
                "90% of APOE2/E2 homozygotes therefore have NORMAL or LOW LDL-C and are PROTECTED from CVD; "
                "FD only develops in the 10% who have a second metabolic hit. "
                "SECOND HITS THAT UNMASK FD: "
                "Obesity/insulin resistance: hyperinsulinaemia → upregulates VLDL secretion → VLDL overwhelms impaired remnant clearance; "
                "Type 2 diabetes: same VLDL overproduction; "
                "Hypothyroidism: thyroid hormone upregulates LDLR and LPL; "
                "hypothyroidism → downregulates both → double hit on remnant clearance; "
                "Post-menopausal oestrogen deficiency: oestrogen normally upregulates LDLR; "
                "oestrogen loss → FD often first presents at menopause in women; "
                "Renal failure: impairs LPL-mediated VLDL-TG clearance; "
                "Alcohol excess: stimulates hepatic VLDL secretion; "
                "Co-existing dyslipidaemia gene (LDLR, APOB mutation): additional LDL-raising effect. "
                "LIPOPROTEIN ELECTROPHORESIS: "
                "FD shows broad-beta band on agarose gel (β-VLDL — VLDL remnants migrate to β position); "
                "VLDL-C:TG ratio >0.69 is diagnostic for FD; "
                "Fredrickson Type III pattern. "
                "TREATMENT DECISION TREE: "
                "Step 1: Identify and treat second hit (weight loss, thyroid replacement, glycaemic control, alcohol cessation); "
                "Step 2: If still dyslipidaemic → fibrates (fenofibrate first-line); "
                "Step 3: If fibrate insufficient → add statin; "
                "Step 4: Avoid gemfibrozil + statin (myopathy risk); use fenofibrate instead."
            ),
        },
        "pharmacological_distinctions": [
            "Statins vs ezetimibe in sitosterolemia: Statins inhibit HMG-CoA reductase → reduce ENDOGENOUS cholesterol synthesis only → no effect on plant sterol accumulation (dietary, not synthesised); ezetimibe blocks NPC1L1 → reduces plant sterol AND cholesterol intestinal absorption → ezetimibe is curative for plant sterol elevation; both drugs can be used together if any cholesterol elevation coexists",
            "Fibrates (fenofibrate) vs statins in FD: Fibrates activate PPARα → increase lipoprotein lipase expression → VLDL-TG hydrolysis → VLDL remnant clearance improves → both TG and LDL-C fall; statins upregulate LDLR → increase VLDL remnant hepatic uptake → complementary mechanism; prefer fenofibrate over gemfibrozil when combining with statin (gemfibrozil inhibits statin glucuronidation → elevated statin plasma levels → myopathy risk 10–50× higher; fenofibrate does not inhibit statin metabolism)",
            "PCSK9 inhibitors: FH vs Lp(a): In FH (LDLR mutations) → PCSK9i prevent LDLR degradation → more LDLR recycled → 50–60% additional LDL-C reduction; in elevated Lp(a) → PCSK9i reduce Lp(a) only ~25% (mechanism uncertain — possibly increased hepatic Lp(a) clearance); PCSK9i are NOT adequate monotherapy for very high Lp(a) (>150 mg/dL) — LDL apheresis or experimental RNA therapies needed",
            "Lomitapide in HoFH vs ARH: Lomitapide inhibits MTP (microsomal triglyceride transfer protein) → blocks VLDL and chylomicron assembly → less LDL substrate; FDA-approved for HoFH; used off-label in ARH (similar phenotype); hepatotoxicity with steatosis requiring regular LFTs + liver MRI; fatty liver limits dose escalation; teratogenic — mandatory pregnancy prevention",
            "Bile acid sequestrants (cholestyramine/colesevelam) in pregnancy vs statins: Statins are Category X — absolutely contraindicated in pregnancy (affect cholesterol synthesis needed for foetal development); bile acid sequestrants not absorbed systemically → no foetal exposure → safe in pregnancy for FH management; colesevelam also approved for T2DM glycaemic control; downside: binds other drugs (separate by ≥4 hours), GI side effects, hypertriglyceridaemia risk (avoid in FD with elevated TG)",
        ],
        "key_standards": [
            "ESC/EAS 2019 Guidelines for Management of Dyslipidaemias (Mach 2020, EHJ): LDL-C targets by risk category; statin intensity classification; PCSK9i indications; cascade testing FH recommendation; FH diagnosis criteria (DLCN score, Simon Broome)",
            "ACC/AHA 2018 Cholesterol Guidelines (Grundy 2019, Circulation): Risk enhancers including Lp(a) >50 mg/dL; statin intensity; secondary prevention targets; risk discussion framework; PCSK9i criteria (LDL >70 mg/dL on max statin in very high-risk)",
            "FOURIER Trial (Sabatine 2017 NEJM): Evolocumab + statin vs statin alone in established ASCVD; primary endpoint HR 0.85 (95% CI 0.79–0.92); LDL-C median 92→30 mg/dL; MI reduction 27%; validated PCSK9i for secondary prevention",
            "Cohen 2006 (NEJM — Dallas Heart Study): PCSK9 LOF variants (Y142X, C679X) in African Americans → LDL-C 28% lower → 88% 10-year CVD risk reduction; validated PCSK9 LOF as therapeutic target; basis for PCSK9 antibody and siRNA drug development",
            "EAS Lp(a) Consensus Statement (Kronenberg 2022, EAS): Lp(a) measurement once in all adults; risk thresholds (>50 mg/dL elevated; >150 mg/dL very high risk); LDL apheresis indications; guidance on emerging therapies (pelacarsen, olpasiran); cascade testing recommendation",
        ],
    }
