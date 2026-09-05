#!/usr/bin/env python3
"""Hereditary Lipid Disorders Atlas — Complete 8-Gene Hereditary Lipid/Lipoprotein Disorders Atlas
LDLR    (Familial Hypercholesterolemia type 1 — 860 aa; 19p13.2; LDL receptor;
         AD; most common monogenic hypercholesterolemia 1:200–500;
         statins first line; PCSK9 inhibitors dramatically effective; premature CAD) ·
APOB    (Familial Defective ApoB-100 — 4563 aa; 2p24.1; ApoB LDL receptor ligand;
         AD; p.Arg3527Gln founder mutation; milder than FH; statins + ezetimibe) ·
PCSK9   (Familial Hypercholesterolemia type 3 GOF — 692 aa; 1p32.3;
         AD GOF; rare FH3; PCSK9 inhibitors (evolocumab/alirocumab) first specific Tx;
         LOF variants are PROTECTIVE — coronary risk ↓88%) ·
APOE    (Type III Hyperlipoproteinemia / Dysbetalipoproteinemia — 317 aa; 19q13.32;
         codominant ε2/ε2 + metabolic trigger; VLDL remnant accumulation;
         fibrate FIRST LINE for elevated TG + TC; xanthomata pathognomonic) ·
LPL     (Familial Chylomicronemia Syndrome — 475 aa; 8p21.3;
         AR; severe hypertriglyceridemia; pancreatitis risk;
         fat restriction <20g/day mandatory; NO approved lipid-lowering drug) ·
ABCA1   (Tangier Disease / Familial Hypoalphalipoproteinemia — 2261 aa; 9q31.1;
         AR Tangier; AD hypoalpha; near-zero HDL; orange tonsils PATHOGNOMONIC;
         no approved HDL-raising therapy; lifestyle modification; statin for LDL) ·
LIPA    (Wolman Disease / CESD — 399 aa; 10q23.31; lysosomal acid lipase;
         AR; sebelipase alfa ERT FDA2015; infant Wolman lethal;
         CESD: hepatomegaly + hypercholesterolemia; liver failure risk) ·
APOC2   (ApoC-II Deficiency Chylomicronemia — 101 aa; 19q13.32; LPL cofactor;
         AR; severe hypertriglyceridemia without LPL mutation;
         fat restriction; fresh frozen plasma transfusion in pancreatitis crisis)
320-patient aggregate cohort (8 × 40, seeds 1222–1229)
"""

import random

SEED_BASE = 1222

LIPID_GENES = [
    # ── LDLR — Familial Hypercholesterolemia Type 1 ──────────────────────────
    {
        "gene": "LDLR",
        "protein": "Low-Density Lipoprotein Receptor (LDLR)",
        "alias": (
            "LDLR; OMIM gene 606945; FH1; 19p13.2; 860 aa; ~95 kDa; "
            "AD (haploinsufficiency); Class I-VI variants; LDL cholesterol >8 mmol/L heterozygous; "
            "most common monogenic hypercholesterolemia 1 in 200–500"
        ),
        "aa": "860 aa",
        "kDa": "~95 kDa",
        "locus": "19p13.2",
        "omim_gene": 606945,
        "omim_disease": 143890,
        "inheritance": "AD (haploinsufficiency)",
        "gene_class": (
            "Type I transmembrane receptor; ligand-binding domain (1–292 aa) binds ApoB-100 on LDL and ApoE on IDL/VLDL remnants; "
            "EGF-like domain (292–400 aa) releases LDL in endosome at low pH; "
            "LDLR variants classified into 6 functional classes: "
            "Class I — null allele (premature stop, frameshift, large deletion), no protein; "
            "Class II — transport-defective (LDLR retained in ER → cannot reach surface); "
            "Class III — binding-defective (reaches surface but does not bind LDL); "
            "Class IV — internalisation-defective (binds LDL but does not endocytose); "
            "Class V — recycling-defective (LDLR degraded in lysosome, not recycled); "
            "Class VI — expression-defective (post-transcriptional). "
            "Pathophysiology: half-life of LDL particles normally ~2–3 days; "
            "heterozygous FH: LDLR activity ~50% → LDL clearance halved → plasma LDL 2–3× population mean; "
            "homozygous FH: LDLR activity <2% → LDL 4–10× normal → cutaneous xanthomata before age 10; "
            "PCSK9 targets LDLR for lysosomal degradation — PCSK9 inhibitors increase LDLR recycling"
        ),
        "phenotype": (
            "Heterozygous FH: LDL-C >5 mmol/L (often 6–10 mmol/L); tendon xanthomata (Achilles, extensor tendons); "
            "corneal arcus before age 45; premature cardiovascular disease (MI age 35–55 men; 45–65 women); "
            "Homozygous FH: LDL-C >13 mmol/L; cutaneous xanthomata on buttocks/elbows/knees before age 10; "
            "aortic stenosis; coronary artery disease in first decade of life without treatment; "
            "Family history: vertical AD transmission; 50% of first-degree relatives affected"
        ),
        "hallmark": (
            "TENDON XANTHOMATA (Achilles, hand extensors) — PATHOGNOMONIC for FH; "
            "PREMATURE CAD: MI before age 55 (men) or 60 (women) in patient or first-degree relative; "
            "LDL-C >4.9 mmol/L in adult without secondary cause; "
            "CORNEAL ARCUS before age 45 — suggestive of FH; "
            "FAMILIAL PATTERN: autosomal dominant; cascade genetic testing mandatory in all first-degree relatives; "
            "Dutch Lipid Clinic Network score ≥6 = probable/definite FH — use for diagnosis; "
            "GENETIC DIAGNOSIS: LDLR + APOB + PCSK9 panel sequencing + MLPA for large deletions"
        ),
        "treatment_alert": (
            "HIGH-INTENSITY STATIN first line: rosuvastatin 20–40 mg or atorvastatin 40–80 mg; "
            "LDL-C target: <1.8 mmol/L (established CVD) or <2.6 mmol/L (no CVD); "
            "ADD EZETIMIBE: if LDL-C not at target on maximum statin; additive 15–20% LDL-C reduction; "
            "PCSK9 INHIBITORS (evolocumab/alirocumab): add if LDL-C still >2.6 mmol/L despite statin+ezetimibe; "
            "reduces LDL-C 50–60% on top of statin; approved NICE/FDA for FH; "
            "INCLISIRAN: siRNA PCSK9 inhibitor, twice-yearly injection, similar efficacy; "
            "LOMITAPIDE: for homozygous FH (MTP inhibitor, hepatotoxicity monitoring required); "
            "LIPOPROTEIN APHERESIS: every 2 weeks for severe homozygous FH or refractory hetFH + CAD; "
            "STATINS IN CHILDREN: start age 8–10 years in FH (if LDL-C >5 mmol/L despite diet); "
            "PREGNANCY: statins CONTRAINDICATED (teratogenic) — stop preconception; bile acid sequestrants safe"
        ),
        "key_ddx": (
            "FDB/APOB (milder LDL elevation; Arg3527Gln; same phenotype; genotype required); "
            "FH3/PCSK9 GOF (rare; similar phenotype; GOF vs LOF is critical); "
            "Polygenic hypercholesterolemia (no tendon xanthomata; no strong family history; lower LDL-C); "
            "Familial combined hyperlipidemia (elevated TG + LDL-C; mixed; no tendon xanthomata); "
            "Secondary hypercholesterolemia: hypothyroidism, nephrotic syndrome, cholestasis — exclude first"
        ),
        "ldl_pattern": "Isolated severe LDL-C elevation (>5 mmol/L); TG normal; HDL normal or mildly low",
        "tg_pattern": "Normal (<2 mmol/L); TG elevation suggests secondary cause or combined disorder",
        "primary_complication": "Premature atherosclerotic cardiovascular disease (MI, stroke, peripheral arterial disease)",
        "disease_detail": (
            "FH1 — caused by LDLR LOF mutations (>3000 variants); most common autosomal dominant disorder worldwide; "
            "1 in 200–500 adults (hetFH); 1 in 160,000–300,000 (homFH); 80% of FH cases genetically undetected; "
            "risk of premature MI 10× population in untreated hetFH; annual CVD risk on statins reduced 50%"
        ),
        "variants": [
            {"name": "p.Trp87Gly", "frequency": "Class II transport defect; UK founder variant"},
            {"name": "p.Glu228Lys", "frequency": "Class III binding defect; common European variant"},
            {"name": "c.1061-1G>A", "frequency": "Splice defect; Class I null; Ashkenazi Jewish founder"},
            {"name": "Lebanese founder", "frequency": "p.Cys242Tyr; 15% Lebanese FH population"},
        ],
        "drug_ci": [
            "STATINS CONTRAINDICATED in pregnancy and breastfeeding (teratogenic; Category X)",
            "PCSK9 inhibitors: avoid in pregnancy (limited data); discontinue if pregnant",
            "LOMITAPIDE: liver toxicity; alcohol absolutely contraindicated; CYP3A4 inhibitors raise levels 10×",
            "BILE ACID SEQUESTRANTS reduce absorption of warfarin, thyroid hormones, fat-soluble vitamins — administer 4h apart",
        ],
        "rates": {"drug_error": 0.45, "dx_delay": 0.72, "surveillance": 0.58, "sev_mild": 0.45, "sev_mod": 0.40, "sev_sev": 0.15},
    },
    # ── APOB — Familial Defective ApoB-100 ───────────────────────────────────
    {
        "gene": "APOB",
        "protein": "Apolipoprotein B-100 (ApoB-100)",
        "alias": (
            "APOB; OMIM gene 107730; FDB; 2p24.1; 4563 aa; ~512 kDa; "
            "AD; p.Arg3527Gln founder mutation 0.1–0.5% European; "
            "milder hypercholesterolaemia than FH1 (LDLR); LDL-C typically 5–9 mmol/L"
        ),
        "aa": "4563 aa",
        "kDa": "~512 kDa",
        "locus": "2p24.1",
        "omim_gene": 107730,
        "omim_disease": 144010,
        "inheritance": "AD (LOF binding domain)",
        "gene_class": (
            "APOB is the structural protein of LDL, IDL, VLDL, and Lp(a); "
            "ApoB-100 (liver-synthesised, 4563 aa) is the sole protein on LDL particles; "
            "ApoB-48 (intestinal form, 2152 aa, same gene, alternative editing at codon 2153) is on chylomicrons; "
            "The LDLR-binding domain of ApoB-100 is centred around Arg3527 (residues 3359–3369); "
            "p.Arg3527Gln mutation: disrupts positive charge in LDLR-binding domain → LDL binds LDLR poorly; "
            "FDB LDL clears slower → elevated plasma LDL; LDLR itself is functional → statins still very effective; "
            "FDB is phenotypically similar to hetFH but typically milder; "
            "APOB-48 is NOT affected by p.Arg3527Gln (different domain) → chylomicron clearance normal"
        ),
        "phenotype": (
            "Elevated LDL-C: typically 5–9 mmol/L (milder than LDLR FH1); "
            "tendon xanthomata: less common than FH1 (~30% vs 60% in LDLR); "
            "corneal arcus before age 45: present; "
            "premature coronary artery disease: risk intermediate between population and FH1; "
            "family history: autosomal dominant; 50% first-degree relatives affected; "
            "often misclassified as polygenic hypercholesterolaemia without genetic testing"
        ),
        "hallmark": (
            "p.Arg3527Gln FOUNDER MUTATION — accounts for >90% of FDB in Europeans; "
            "FUNCTIONAL LDLR BUT DEFECTIVE APOB LIGAND — statins are more effective than in LDLR FH "
            "(LDLR upregulated by statin → binds other ApoE-bearing lipoproteins better); "
            "MILDER PHENOTYPE than FH1-LDLR: LDL-C 5–9 vs 7–12 mmol/L; fewer xanthomata; "
            "GENOTYPE REQUIRED to distinguish from LDLR-FH1 (treatment intensity same, prognosis slightly better); "
            "STATIN RESPONSE excellent — 40–50% LDL-C reduction (functional LDLR upregulated)"
        ),
        "treatment_alert": (
            "HIGH-INTENSITY STATIN: same as FH1 — rosuvastatin 20–40 mg or atorvastatin 40–80 mg; "
            "STATIN RESPONSE: typically better than FH1 (functional LDLR can clear other ApoE-bearing lipoproteins); "
            "ADD EZETIMIBE if not at target; "
            "PCSK9 inhibitors: effective — reduce APOB-containing LDL particles; "
            "LDL-C target: <2.6 mmol/L no CVD; <1.8 mmol/L established CVD; "
            "CASCADE TESTING: all first-degree relatives (50% affected); genetic test is definitive; "
            "PREGNANCY: statins contraindicated — switch to bile acid sequestrant if clinically warranted"
        ),
        "key_ddx": (
            "LDLR/FH1 (clinically overlapping; genotype required; LDLR more severe; LDLR statin response slightly less); "
            "PCSK9 GOF/FH3 (rare; overlapping phenotype); "
            "Polygenic hypercholesterolaemia (no founder mutation; family history less strong; no xanthomata usually); "
            "Secondary hypercholesterolaemia: hypothyroidism, nephrotic syndrome — exclude"
        ),
        "ldl_pattern": "Elevated LDL-C (5–9 mmol/L); milder than LDLR-FH; TG normal; HDL normal",
        "tg_pattern": "Normal; TG elevation should prompt secondary cause investigation",
        "primary_complication": "Premature coronary artery disease (risk intermediate between population and FH1)",
        "disease_detail": (
            "FDB — APOB p.Arg3527Gln founder mutation; prevalence 1:500–1000 in Europeans; "
            "clinically indistinguishable from mild FH1 without genetic panel; "
            "statin monotherapy achieves target in ~50% (vs ~30% FH1-LDLR) — reflecting functional LDLR"
        ),
        "variants": [
            {"name": "p.Arg3527Gln", "frequency": ">90% of FDB; European founder; disrupts LDLR-binding"},
            {"name": "p.Arg3527Trp", "frequency": "Severe variant; less common; similar phenotype"},
        ],
        "drug_ci": [
            "STATINS CONTRAINDICATED in pregnancy (teratogenic)",
            "FIBRATES: avoid with statins (myopathy risk); if combined, use rosuvastatin + fenofibrate preferably",
        ],
        "rates": {"drug_error": 0.42, "dx_delay": 0.68, "surveillance": 0.60, "sev_mild": 0.55, "sev_mod": 0.35, "sev_sev": 0.10},
    },
    # ── PCSK9 — Familial Hypercholesterolemia Type 3 (GOF) ───────────────────
    {
        "gene": "PCSK9",
        "protein": "Proprotein Convertase Subtilisin/Kexin Type 9 (PCSK9)",
        "alias": (
            "PCSK9; OMIM gene 607786; FH3-GOF; 1p32.3; 692 aa; ~72 kDa; "
            "AD GOF (FH3); LOF variants protective; secreted serine protease; "
            "target of evolocumab/alirocumab mAbs and inclisiran siRNA"
        ),
        "aa": "692 aa",
        "kDa": "~72 kDa",
        "locus": "1p32.3",
        "omim_gene": 607786,
        "omim_disease": 603776,
        "inheritance": "AD (GOF for FH3); LOF variants are protective",
        "gene_class": (
            "PCSK9 is a serine protease synthesised mainly in hepatocytes; "
            "signal peptide (1–30 aa) + prodomain (31–152 aa, acts as chaperone) + catalytic domain + hinge + C-terminal domain; "
            "Mechanism: PCSK9 binds LDLR extracellular domain at cell surface → PCSK9-LDLR complex internalised → "
            "at endosomal pH, LDLR would normally release LDL and recycle to surface; "
            "PCSK9 binding prevents pH-dependent conformational change → LDLR targeted to lysosome → degraded; "
            "Net effect: PCSK9 reduces surface LDLR density → less LDL clearance → higher plasma LDL; "
            "GOF mutations (p.Asp374Tyr most common): increase PCSK9 affinity for LDLR → more LDLR degraded → FH phenotype; "
            "LOF mutations (p.Tyr142Ter, p.Cys679Ter): reduce PCSK9 → more LDLR recycled → LDL-C 28% lower → "
            "88% reduction in coronary events (Dallas Heart Study) — proof of concept for PCSK9 inhibitors; "
            "PCSK9 inhibitors (evolocumab, alirocumab): mAbs bind secreted PCSK9 → prevent LDLR-binding → "
            "LDLR recycled → LDL-C reduced 50–60% on top of statins"
        ),
        "phenotype": (
            "GOF FH3: similar to FH1-LDLR; LDL-C >5 mmol/L; premature CAD; tendon xanthomata; "
            "p.Asp374Tyr: severe GOF; LDL-C up to 14 mmol/L in homozygous; "
            "p.Ser127Arg: moderate GOF; LDL-C typically 6–9 mmol/L; "
            "LOF carriers (heterozygous): LDL-C ~28% below mean; 47% lower CAD risk (PROVE-IT); "
            "Homozygous LOF: very low LDL-C (<0.5 mmol/L) — NO atherosclerosis; no side effects observed"
        ),
        "hallmark": (
            "RARE FH3 (GOF) — clinically same as FH1/FH2 but requires genetic panel to identify; "
            "PCSK9 INHIBITORS are the SPECIFIC THERAPY — if FH3 confirmed, PCSK9 mAb is first targeted choice; "
            "LOF VARIANTS ARE PROTECTIVE — identify LOF family members as low CVD risk (no treatment needed); "
            "p.Asp374Tyr FOUNDER — Norwegian/European; homozygous form has very severe FH; "
            "GENETIC PARADOX: same gene GOF = FH3 (harmful) vs LOF = cardioprotective (extremely beneficial); "
            "INCLISIRAN (siRNA): same mechanism as mAb PCSK9 inhibitors; twice-yearly injection; FDA-approved 2021"
        ),
        "treatment_alert": (
            "PCSK9 INHIBITORS first-line specific treatment for FH3-GOF: "
            "evolocumab 140 mg SC q2wk or 420 mg monthly; alirocumab 75–150 mg q2wk; "
            "reduces LDL-C 50–60% from statin baseline; FOURIER trial: evolocumab reduces MI 27%; "
            "INCLISIRAN 284 mg SC: twice yearly (Day 1, Day 90, then every 6 months); "
            "ADD HIGH-INTENSITY STATIN as foundation: PCSK9 inhibitors work synergistically with statins "
            "(statin upregulates LDLR; PCSK9 inhibitor prevents LDLR degradation → more surface LDLR); "
            "ADD EZETIMIBE (NPC1L1 inhibitor): further 15–20% LDL-C reduction; "
            "LDL-C target: <1.4 mmol/L in very high CVD risk (AHA 2022); "
            "DO NOT use PCSK9 inhibitors for LOF variant carriers (they already have low LDL — no indication)"
        ),
        "key_ddx": (
            "FH1-LDLR (clinically identical; genotype essential; different target therapy emphasis); "
            "FDB-APOB (milder; different gene; same treatment); "
            "Polygenic hypercholesterolaemia (no single gene variant; no tendon xanthomata); "
            "PCSK9 LOF misidentified as FH — critical error: LOF is PROTECTIVE not FH"
        ),
        "ldl_pattern": "Elevated LDL-C (5–14 mmol/L GOF); very low LDL-C (<0.5 mmol/L homozygous LOF)",
        "tg_pattern": "Normal in GOF FH3; unaffected by PCSK9 pathway",
        "primary_complication": "Premature atherosclerotic cardiovascular disease (GOF); absence of atherosclerosis (LOF)",
        "disease_detail": (
            "FH3 — PCSK9 GOF; rare (<5% of FH); p.Asp374Tyr most common GOF variant; "
            "LOF carriers 1 in 50 Black Americans (p.Tyr142Ter); 1 in 30 White Americans (p.Cys679Ter); "
            "PCSK9 inhibitors represent one of the clearest examples of human genetics informing drug discovery"
        ),
        "variants": [
            {"name": "p.Asp374Tyr GOF", "frequency": "Norwegian founder; severe FH3; LDL-C up to 14 mmol/L"},
            {"name": "p.Ser127Arg GOF", "frequency": "French Canadian founder; moderate FH3"},
            {"name": "p.Tyr142Ter LOF", "frequency": "2% Black Americans; LDL-C 28% lower; protective"},
            {"name": "p.Cys679Ter LOF", "frequency": "~3% White Americans; protective; reduces CVD 47%"},
        ],
        "drug_ci": [
            "PCSK9 inhibitors: avoid in pregnancy (limited safety data)",
            "No known drug-drug interactions with PCSK9 mAbs (biologic, not CYP-metabolised)",
            "STATINS remain contraindicated in pregnancy even if PCSK9 inhibitor prescribed",
        ],
        "rates": {"drug_error": 0.50, "dx_delay": 0.80, "surveillance": 0.52, "sev_mild": 0.40, "sev_mod": 0.40, "sev_sev": 0.20},
    },
    # ── APOE — Type III Hyperlipoproteinemia / Dysbetalipoproteinemia ─────────
    {
        "gene": "APOE",
        "protein": "Apolipoprotein E (ApoE)",
        "alias": (
            "APOE; OMIM gene 107741; HLP3; 19q13.32; 317 aa; ~34 kDa; "
            "codominant ε2/ε2 + metabolic trigger; VLDL remnant (IDL) accumulation; "
            "palmar xanthomata PATHOGNOMONIC; fibrate first line for TG + TC elevation"
        ),
        "aa": "317 aa",
        "kDa": "~34 kDa",
        "locus": "19q13.32",
        "omim_gene": 107741,
        "omim_disease": 617347,
        "inheritance": "Codominant (APOE ε2/ε2 necessary but not sufficient — needs metabolic trigger)",
        "gene_class": (
            "APOE exists as 3 major isoforms (ε2, ε3, ε4) determined by two coding SNPs (rs7412 and rs429358); "
            "ε3 is commonest (population 60%); ε2 = Cys112, Cys158; ε4 = Arg112, Arg158; "
            "ApoE is the ligand for LDLR (LDL clearance) and LRP1 (VLDL/IDL remnant clearance); "
            "ApoE ε2 — reduced affinity for LDLR (~1% of ε3); slow clearance of VLDL remnants (IDL); "
            "APOE ε2/ε2 (1% of population): 90% have mildly elevated TG + TC but no overt disease; "
            "10% with ε2/ε2 develop Type III HLP when second hit present: obesity, hypothyroidism, "
            "diabetes, menopause, renal disease, alcohol, excess saturated fat; "
            "APOE ε4: increased LDL-C (promotes LDL overproduction); Alzheimer's disease risk (3.7× hetFH, 12× homE4/E4); "
            "Mechanism of Type III: APOE ε2 → slow IDL/VLDL remnant clearance → IDL accumulates → "
            "β-VLDL (IDL-like particles) → taken up by macrophages → foam cells → atherosclerosis"
        ),
        "phenotype": (
            "Type III HLP (when metabolic trigger present): TC 7–14 mmol/L + TG 5–15 mmol/L; "
            "xanthomata striae palmaris (palmar/tuberous xanthomata) — PATHOGNOMONIC for Type III HLP; "
            "tubo-eruptive xanthomata on elbows, knees; "
            "premature coronary artery disease and peripheral vascular disease; "
            "statin-resistant: fibrate more effective than statin for this mixed dyslipidaemia; "
            "responds dramatically to: weight loss, treat hypothyroidism, fibrate, omega-3"
        ),
        "hallmark": (
            "PALMAR XANTHOMATA (xanthomata striae palmaris) — PATHOGNOMONIC for Type III; "
            "BOTH TC AND TG ELEVATED (mixed dyslipidaemia) — not isolated hypercholesterolaemia; "
            "REQUIRES METABOLIC TRIGGER: rarely manifests without obesity, hypothyroidism, DM, etc.; "
            "APOE GENOTYPING: ε2/ε2 diagnostic; standard lipid panels cannot diagnose Type III; "
            "FIBRATE FIRST LINE (not statin) — targets TG-rich remnant particles more effectively; "
            "DRAMATIC RESPONSE TO TREATMENT: TG and TC both fall 50–70% with fibrate + lifestyle; "
            "VLDL REMNANTS (β-VLDL): revealed on lipoprotein electrophoresis as broad β band (Type III pattern)"
        ),
        "treatment_alert": (
            "FIBRATE FIRST LINE: fenofibrate 145 mg/day or bezafibrate 400 mg SR; "
            "targets VLDL-TG overproduction and enhances LPL activity → clears remnant particles; "
            "LIFESTYLE: weight loss 10% reduces TG + TC dramatically; alcohol reduction; low saturated fat; "
            "TREAT SECONDARY CAUSES: hypothyroidism (thyroxine resolves Type III); DM control; estrogen effects; "
            "STATIN: add for residual LDL-C elevation; statin alone less effective than fibrate for Type III; "
            "OMEGA-3 FATTY ACIDS (EPA/DHA 2–4 g/day): additive TG lowering; "
            "COMBINATION: fibrate + statin + omega-3 for refractory cases; "
            "FIBRATE + STATIN SAFETY: avoid gemfibrozil + statin (myopathy risk); fenofibrate + statin is safer; "
            "APOE ε4/ε4 subgroup: higher Alzheimer risk — no lipid-lowering drug currently reduces AD risk"
        ),
        "key_ddx": (
            "FH1-LDLR (isolated LDL elevation; no palmar xanthomata; TG normal); "
            "Familial combined hyperlipidemia (FCHL) — no palmar xanthomata; genotype required; "
            "Secondary mixed dyslipidaemia (hypothyroidism, nephrotic syndrome, alcohol — check APOE genotype); "
            "Polygenic mixed dyslipidaemia (no ε2/ε2; multiple SNPs)"
        ),
        "ldl_pattern": "Elevated total cholesterol (TC); LDL-C may be variable (β-VLDL counted as LDL on Friedewald)",
        "tg_pattern": "Elevated TG 5–15 mmol/L; VLDL remnant accumulation; pancreatitis risk >10 mmol/L",
        "primary_complication": "Premature peripheral vascular disease and coronary artery disease; pancreatitis",
        "disease_detail": (
            "Type III HLP — APOE ε2/ε2 + metabolic second hit; prevalence ~1:5000; "
            "ε2/ε2 genotype found in 1% population but only 10% develop Type III; "
            "responds best among dyslipidaemias to treatment (dramatic normalisation with fibrate + lifestyle)"
        ),
        "variants": [
            {"name": "ε2/ε2 (Cys112+Cys158)", "frequency": "1% population; 10% develop Type III"},
            {"name": "ε4/ε4 (Arg112+Arg158)", "frequency": "2% population; high Alzheimer risk; higher LDL-C"},
        ],
        "drug_ci": [
            "GEMFIBROZIL + STATIN: increased myopathy/rhabdomyolysis risk — avoid combination; use fenofibrate + statin instead",
            "FIBRATES: avoid in severe renal impairment (eGFR <30); bile acid sequestrants may worsen TG",
            "OMEGA-3 high-dose: avoid in patients on anticoagulants (platelet effects at >3 g/day)",
        ],
        "rates": {"drug_error": 0.48, "dx_delay": 0.75, "surveillance": 0.55, "sev_mild": 0.35, "sev_mod": 0.45, "sev_sev": 0.20},
    },
    # ── LPL — Familial Chylomicronemia Syndrome ───────────────────────────────
    {
        "gene": "LPL",
        "protein": "Lipoprotein Lipase (LPL)",
        "alias": (
            "LPL; OMIM gene 238600; FCS Type I; 8p21.3; 475 aa; ~53 kDa; "
            "AR; severe hypertriglyceridemia (TG >10–100 mmol/L); recurrent acute pancreatitis; "
            "fat restriction <20g/day mandatory; no approved lipid-lowering drug; volanesorsen approved EU 2019"
        ),
        "aa": "475 aa",
        "kDa": "~53 kDa",
        "locus": "8p21.3",
        "omim_gene": 238600,
        "omim_disease": 238600,
        "inheritance": "AR (homozygous or compound heterozygous)",
        "gene_class": (
            "LPL is synthesised by adipocytes and muscle cells; anchored to capillary endothelium via GPIHBP1; "
            "activated by ApoC-II (APOC2) on chylomicron surface → hydrolyses TG in chylomicrons and VLDL; "
            "LPL LOF → chylomicrons cannot be cleared → TG accumulates in plasma → chylomicron retained; "
            "TG >10 mmol/L: risk of acute pancreatitis (TG hydrolysed to toxic free fatty acids in pancreas); "
            "Eruptive xanthomata (TG deposits in skin macrophages) when TG >20 mmol/L; "
            "Lipaemia retinalis (fundus orange appearance) when TG >30–40 mmol/L; "
            "LPL-deficient plasma appears milky white (chylomicrons floating as cream layer); "
            "ApoC-III (encoded by APOC3) inhibits LPL activity — APOC3 antisense: volanesorsen (approved EU 2019)"
        ),
        "phenotype": (
            "Lifelong severe hypertriglyceridaemia: TG often 20–100 mmol/L; "
            "recurrent acute pancreatitis (life-threatening; trigger for diagnosis); "
            "eruptive xanthomata: tiny yellow-white papules on buttocks/trunk/extremities; "
            "hepatosplenomegaly (esterified chylomicrons cleared by Kupffer cells); "
            "lipemia retinalis: salmon-pink retinal vessels on fundoscopy; "
            "abdominal pain crises; NO premature atherosclerosis (LDL-C often LOW); "
            "cognitive impairment ('brain fog') due to hypertriglyceridaemia microvascular effect"
        ),
        "hallmark": (
            "ERUPTIVE XANTHOMATA — tiny yellow papules on trunk/buttocks; PATHOGNOMONIC for chylomicronemia; "
            "LIPEMIA RETINALIS — salmon/orange retinal vessels visible on fundoscopy (TG >30 mmol/L); "
            "CREAM LAYER ON REFRIGERATED PLASMA — chylomicrons float overnight; "
            "TG >10 mmol/L WITHOUT secondary cause → FCS until proven otherwise; "
            "RECURRENT PANCREATITIS — first presentation often acute abdomen; "
            "LOW LDL-C (despite severe hypertriglyceridaemia) — distinguishes from combined hyperlipidaemia; "
            "FAT RESTRICTION <20g/day MANDATORY — first and most effective intervention"
        ),
        "treatment_alert": (
            "FAT RESTRICTION <20 g/day: MOST EFFECTIVE intervention; reduces TG 50–70%; "
            "ALL DIETARY FAT RESTRICTED (saturated, unsaturated, trans — all raise chylomicrons); "
            "medium-chain triglycerides (MCT oil): can be used as fat substitute (absorbed directly, bypass chylomicron path); "
            "VOLANESORSEN (EU 2019; PFIZER EU approval): antisense oligonucleotide targeting APOC3 mRNA; "
            "reduces ApoC-III → reduces chylomicron retention → TG decreases 70–80%; "
            "PLATELET COUNT MONITORING mandatory on volanesorsen (thrombocytopenia risk); "
            "FIBRATES: limited efficacy in FCS (require functional LPL to work — LPL is absent); "
            "OMEGA-3 FATTY ACIDS: some benefit (reduces VLDL-TG production); "
            "ALCOHOL: ABSOLUTELY CONTRAINDICATED — acutely raises TG → pancreatitis; "
            "STATINS: NOT effective (chylomicrons are not LDL-derived); may use for LDL if needed; "
            "ORAL CONTRACEPTIVES (estrogen): CONTRAINDICATED — raise TG dramatically; "
            "ALIPOGENE TIPARVOVEC (gene therapy EU 2012 — withdrawn): historical; new gene therapies in trials"
        ),
        "key_ddx": (
            "APOC2 deficiency (same phenotype; same fat restriction; no LPL activity with APOC2 — add ApoC-II cofactor test); "
            "Multifactorial chylomicronemia (heterozygous LPL variant + secondary TG trigger); "
            "Secondary hypertriglyceridaemia (diabetes, hypothyroidism, alcohol, drugs — EXCLUDE); "
            "Familial partial lipodystrophy (TG elevation + lipodystrophy)"
        ),
        "ldl_pattern": "LDL-C often LOW (<1 mmol/L) despite massive hypertriglyceridaemia — distinguishing feature",
        "tg_pattern": "Severe TG elevation 10–100+ mmol/L; chylomicronemia; pancreatitis risk",
        "primary_complication": "Recurrent acute pancreatitis (potentially fatal); chronic pancreatitis → exocrine insufficiency",
        "disease_detail": (
            "FCS — AR LPL LOF; prevalence ~1:1,000,000; often presents in childhood with pancreatitis; "
            "pancreatitis risk: TG >10 mmol/L (~5%); TG >20 mmol/L (~10-20%); TG >40 mmol/L (~25%); "
            "chronic pancreatitis → diabetes (30%); exocrine insufficiency → PERT needed"
        ),
        "variants": [
            {"name": "p.Pro207Leu", "frequency": "French-Canadian founder; most common FCS in Quebec"},
            {"name": "p.Gly188Glu", "frequency": "Pan-ethnic founder; most common worldwide FCS mutation"},
            {"name": "p.Asp250Asn", "frequency": "European; disrupts catalytic triad"},
        ],
        "drug_ci": [
            "ALCOHOL ABSOLUTELY CONTRAINDICATED — acutely and severely elevates TG; single exposure can trigger pancreatitis",
            "ESTROGEN-CONTAINING CONTRACEPTIVES ABSOLUTELY CONTRAINDICATED — raise TG 50-100%",
            "FIBRATES: limited benefit in true FCS (LPL absent — fibrate works via LPL activation); not first-line",
            "RETINOIC ACID, TAMOXIFEN, CORTICOSTEROIDS: all raise TG — avoid or use with extreme caution",
        ],
        "rates": {"drug_error": 0.35, "dx_delay": 0.45, "surveillance": 0.72, "sev_mild": 0.15, "sev_mod": 0.35, "sev_sev": 0.50},
    },
    # ── ABCA1 — Tangier Disease / Familial Hypoalphalipoproteinemia ───────────
    {
        "gene": "ABCA1",
        "protein": "ATP-Binding Cassette Transporter A1 (ABCA1)",
        "alias": (
            "ABCA1; OMIM gene 600046; Tangier/FHA; 9q31.1; 2261 aa; ~220 kDa; "
            "AR Tangier disease; AD familial hypoalphalipoproteinemia; "
            "near-zero HDL-C; orange tonsils PATHOGNOMONIC Tangier; peripheral neuropathy"
        ),
        "aa": "2261 aa",
        "kDa": "~220 kDa",
        "locus": "9q31.1",
        "omim_gene": 600046,
        "omim_disease": 205400,
        "inheritance": "AR (Tangier disease); AD (familial hypoalphalipoproteinemia FHA)",
        "gene_class": (
            "ABCA1 is a full ABC transporter (two transmembrane domains + two cytoplasmic NBDs) expressed in macrophages, liver, intestine; "
            "ABCA1 transports phospholipid and cholesterol from intracellular membranes to lipid-poor ApoA-I → "
            "forms nascent pre-β HDL → matures to HDL-C via LCAT; "
            "ABCA1 LOF → ApoA-I and ApoA-II cannot be lipidated → HDL formation fails → "
            "pre-β HDL particles are rapidly catabolised → near-zero HDL-C (Tangier) or low HDL-C (FHA); "
            "Cholesterol accumulates in macrophages (foam cells) → orange tonsils, hepatosplenomegaly, neuropathy; "
            "ABCA1 is regulated by oxysterols via LXR (liver X receptor) — key reverse cholesterol transport pathway; "
            "ABCA1 LOF: impaired efflux from macrophages → foam cell accumulation → atherosclerosis despite low LDL"
        ),
        "phenotype": (
            "Tangier disease (homozygous): HDL-C virtually absent (<0.05 mmol/L); "
            "ORANGE TONSILS: pathognomonic — cholesterol ester deposits stain tonsillar tissue bright orange; "
            "hepatosplenomegaly; peripheral neuropathy (mononeuropathy multiplex pattern); "
            "corneal infiltrates; recurrent thrombocytopenia; "
            "premature atherosclerosis (despite low LDL) from impaired reverse cholesterol transport; "
            "FHA (heterozygous): HDL-C <0.9 mmol/L; low ApoA-I; premature CAD; no orange tonsils"
        ),
        "hallmark": (
            "ORANGE TONSILS — foam cell cholesterol ester deposits; PATHOGNOMONIC Tangier disease; "
            "NEAR-ZERO HDL-C — Tangier; HDL <0.05 mmol/L without other cause → ABCA1 testing mandatory; "
            "PERIPHERAL NEUROPATHY — mononeuropathy multiplex; sensorimotor; can precede other features; "
            "HEPATOSPLENOMEGALY — foam cells in macrophages; "
            "PREMATURE CAD DESPITE LOW LDL-C — reverse cholesterol transport failure; "
            "GENETIC TESTING: ABCA1 sequencing + MLPA (large deletions); "
            "CORNEAL INFILTRATES: diffuse, dotlike opacities on slit-lamp"
        ),
        "treatment_alert": (
            "NO APPROVED HDL-RAISING THERAPY (niacin withdrawn from guidelines; CETP inhibitors failed in RCTs); "
            "FOCUS ON LDL-C AND CVD RISK REDUCTION: "
            "statin + ezetimibe for any elevated LDL-C; PCSK9 inhibitors if LDL-C still high; "
            "LIFESTYLE: aerobic exercise (raises HDL-C 5–10%); Mediterranean diet; smoking cessation; "
            "WEIGHT LOSS: 1 kg loss raises HDL-C ~0.4%; "
            "PERIPHERAL NEUROPATHY MANAGEMENT: neurological monitoring; neuropathic pain treatment; "
            "OPHTHALMOLOGY: annual slit-lamp examination (corneal infiltrates); "
            "SPLENECTOMY: only for symptomatic hypersplenism (risk of sepsis from encapsulated organisms — vaccinate); "
            "GENETIC COUNSELLING: cascade testing (AR); partner testing if family-planning; "
            "AVOID: gemfibrozil-statin combination (myopathy); "
            "EXPERIMENTAL: LXR agonists (in trials for macrophage ABCA1 upregulation)"
        ),
        "key_ddx": (
            "Secondary low HDL: smoking, obesity, diabetes, sedentary lifestyle — far more common; "
            "Other genetic low-HDL: APOA1 mutations (near-zero HDL, neuropathy without orange tonsils); "
            "ApoC-III elevation (inhibits HDL formation — APOC3 variants); "
            "Fish-eye disease (LCAT partial deficiency — low HDL + corneal opacity)"
        ),
        "ldl_pattern": "LDL-C often LOW (VLDL remnants cleared via LPL; low LDL despite severe atherosclerosis risk)",
        "tg_pattern": "Mildly elevated TG possible; VLDL metabolism relatively preserved",
        "primary_complication": "Premature atherosclerosis; peripheral neuropathy; corneal infiltrates; hepatosplenomegaly",
        "disease_detail": (
            "Tangier disease — named after Tangier Island (Virginia, USA) where first families identified in 1961; "
            "prevalence <100 cases worldwide (AR homozygous); FHA (AD, heterozygous) more common; "
            "no curative treatment; prognosis determined by CAD and neuropathy severity"
        ),
        "variants": [
            {"name": "p.Cys2177Tyr", "frequency": "Belgian founder; Tangier disease"},
            {"name": "p.Arg587Trp", "frequency": "FHA; common European AD variant"},
        ],
        "drug_ci": [
            "NIACIN: withdrawn from CVD guidelines (ACCELERATE/AIM-HIGH: no benefit over statin; side effects)",
            "GEMFIBROZIL + STATIN: myopathy risk; avoid; use fenofibrate + statin if fibrate needed",
            "SPLENECTOMY: increased sepsis risk from encapsulated organisms — ensure full vaccination first",
        ],
        "rates": {"drug_error": 0.38, "dx_delay": 0.65, "surveillance": 0.62, "sev_mild": 0.30, "sev_mod": 0.48, "sev_sev": 0.22},
    },
    # ── LIPA — Lysosomal Acid Lipase Deficiency (Wolman / CESD) ──────────────
    {
        "gene": "LIPA",
        "protein": "Lysosomal Acid Lipase (LAL / LIPA)",
        "alias": (
            "LIPA; OMIM gene 613497; LAL-D; 10q23.31; 399 aa; ~46 kDa; "
            "AR; Wolman disease (infant, lethal without ERT) vs CESD (adult, hepatic disease); "
            "sebelipase alfa (ERT) FDA/EMA 2015 — life-saving in Wolman"
        ),
        "aa": "399 aa",
        "kDa": "~46 kDa",
        "locus": "10q23.31",
        "omim_gene": 613497,
        "omim_disease": 278000,
        "inheritance": "AR (homozygous or compound heterozygous)",
        "gene_class": (
            "LAL (lysosomal acid lipase) hydrolyses cholesteryl esters and TG in lysosomes; "
            "specifically, LDL-derived cholesteryl esters after receptor-mediated endocytosis + lysosomal delivery; "
            "LAL activity releases free cholesterol in lysosome → transported to ER (ABCA1, NPC1) → "
            "free cholesterol suppresses HMGCR (→ reduces endogenous cholesterol synthesis) and "
            "LDLR expression (→ reduces LDL uptake); "
            "LAL LOF: cholesteryl esters and TG accumulate in lysosomes of liver, adrenal, spleen, intestine; "
            "Two phenotypic extremes by residual LAL activity: "
            "Wolman disease (near-zero LAL; e4 splice variant p.Glu8_Gln99del; infant onset): "
            "adrenal calcifications, vomiting, failure to thrive, hepatosplenomegaly → death by 6 months without ERT; "
            "CESD (cholesteryl ester storage disease; ~1–5% residual LAL; c.894G>A splice): "
            "adult presentation; hepatomegaly; microvesicular steatosis → cirrhosis; hypercholesterolaemia"
        ),
        "phenotype": (
            "Wolman disease (near-zero LAL): adrenal gland calcification on X-ray (PATHOGNOMONIC); "
            "severe vomiting, diarrhoea, malnutrition, hepatosplenomegaly in first weeks of life; "
            "death before age 6 months without enzyme replacement therapy; "
            "CESD (cholesteryl ester storage disease): hepatomegaly + elevated transaminases; "
            "elevated LDL-C + TG; low HDL-C; slow progression to liver fibrosis/cirrhosis; "
            "premature atherosclerosis; adrenal insufficiency may occur; "
            "CESD: often misdiagnosed as non-alcoholic fatty liver disease (NAFLD)"
        ),
        "hallmark": (
            "ADRENAL CALCIFICATION — bilateral adrenal gland calcium on X-ray/CT in Wolman; PATHOGNOMONIC; "
            "WOLMAN = INFANT EMERGENCY: hepatosplenomegaly + adrenal calcification + failure to thrive → ERT IMMEDIATELY; "
            "CESD = MASQUERADES AS NAFLD: hepatomegaly + microvesicular steatosis + elevated LDL-C in child/adult; "
            "LAL ACTIVITY ASSAY: DBS (dried blood spot) LAL activity — most reliable diagnostic test; "
            "LIVER BIOPSY: microvesicular steatosis + Maltese cross birefringent crystals under polarised light; "
            "SEBELIPASE ALFA (ERT): weekly infusion; FDA/EMA 2015; life-saving in Wolman; stabilises CESD"
        ),
        "treatment_alert": (
            "SEBELIPASE ALFA (KANUMA, Alexion): enzyme replacement therapy; "
            "Wolman disease: 1 mg/kg weekly IV (can escalate to 3 mg/kg); start IMMEDIATELY on diagnosis; "
            "delays death; long-term survival possible with early treatment; "
            "CESD: 1 mg/kg every 2 weeks; reduces LDL-C 30–50%; reduces liver fat; slows fibrosis; "
            "LIVER TRANSPLANTATION: historical fallback for CESD with cirrhosis (ERT preferred); "
            "HEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): used in Wolman as bridge/cure in some centres; "
            "STATINS: adjunctive in CESD for residual LDL-C elevation; "
            "EZETIMIBE: additive cholesterol lowering in CESD; "
            "ADRENAL MONITORING: cortisol levels; supplement if adrenal insufficiency; "
            "DO NOT DIAGNOSE AS NAFLD: LAL-D is treatable — check LAL activity in any child with fatty liver + elevated LDL"
        ),
        "key_ddx": (
            "NAFLD/NASH (CESD mimics — check LAL activity in child with fatty liver + elevated LDL-C); "
            "Niemann-Pick type C (NPC1/NPC2 — lysosomal cholesterol transport; filipin staining distinguishes); "
            "Gaucher disease (GBA — glucocerebrosidase; hepatosplenomegaly; but different lipid pattern); "
            "Wolman DDx: other causes of neonatal hepatosplenomegaly (sepsis, Niemann-Pick A, galactosialidosis)"
        ),
        "ldl_pattern": "Elevated LDL-C (CESD); very high LDL-C from impaired cholesterol feedback to HMGCR/LDLR",
        "tg_pattern": "Elevated TG (CESD); lysosomal TG accumulation; mixed dyslipidaemia pattern",
        "primary_complication": "Wolman: death in infancy without ERT; CESD: hepatic cirrhosis + premature atherosclerosis",
        "disease_detail": (
            "LAL-D — AR; Wolman prevalence 1:350,000–1:500,000; CESD 1:50,000–1:300,000 (underdiagnosed); "
            "CESD underdiagnosed as NAFLD; treatable with sebelipase alfa since 2015; "
            "genetic testing: LIPA sequencing; c.894G>A (exon 8 splice) common CESD variant"
        ),
        "variants": [
            {"name": "c.894G>A (exon 8 splice)", "frequency": "Most common CESD variant; residual 3–5% LAL activity"},
            {"name": "p.Glu8_Gln99del (exon 4 splice)", "frequency": "Wolman; near-zero activity; lethal without ERT"},
        ],
        "drug_ci": [
            "SEBELIPASE ALFA: infusion-related reactions — premedicate with antihistamine/acetaminophen; anaphylaxis protocols required",
            "STATINS + CYCLOSPORIN: myopathy risk elevated in CESD patients on immunosuppression (post-liver transplant)",
        ],
        "rates": {"drug_error": 0.30, "dx_delay": 0.65, "surveillance": 0.70, "sev_mild": 0.25, "sev_mod": 0.40, "sev_sev": 0.35},
    },
    # ── APOC2 — ApoC-II Deficiency Chylomicronemia ────────────────────────────
    {
        "gene": "APOC2",
        "protein": "Apolipoprotein C-II (ApoC-II)",
        "alias": (
            "APOC2; OMIM gene 608083; ApoC-II deficiency; 19q13.32; 101 aa; ~11 kDa; "
            "AR; severe hypertriglyceridaemia without LPL mutation; LPL cofactor absent; "
            "fresh frozen plasma transfusion restores LPL activity (diagnostic + acute therapy)"
        ),
        "aa": "101 aa",
        "kDa": "~11 kDa",
        "locus": "19q13.32",
        "omim_gene": 608083,
        "omim_disease": 207750,
        "inheritance": "AR (homozygous or compound heterozygous)",
        "gene_class": (
            "ApoC-II is a small apolipoprotein (101 aa; ~11 kDa) synthesised in liver; "
            "located on VLDL and chylomicron surface; "
            "Mechanism: ApoC-II is the obligate cofactor for LPL activation; "
            "LPL has negligible activity without ApoC-II → without ApoC-II, LPL enzyme is present but non-functional; "
            "ApoC-II LOF → chylomicrons and VLDL-TG cannot be cleared (identical phenotype to LPL LOF); "
            "Distinction from LPL-FCS: in vitro LPL activity assay: "
            "LPL-FCS: no LPL activity WITH or WITHOUT ApoC-II; "
            "ApoC-II deficiency: no LPL activity WITHOUT ApoC-II → RESTORED WITH exogenous ApoC-II (unique); "
            "FFP (fresh frozen plasma) provides exogenous ApoC-II → restores LPL activity → clears TG; "
            "ApoC-II is gene on 19q13.32 (same cluster as APOE, APOC1); "
            "ApoC-III (APOC3) is a natural LPL inhibitor — its reduction (by volanesorsen) used in FCS treatment"
        ),
        "phenotype": (
            "Severe hypertriglyceridaemia: TG often 20–100 mmol/L (identical to LPL-FCS); "
            "recurrent acute pancreatitis (same pancreatitis risk as LPL-FCS); "
            "eruptive xanthomata; hepatosplenomegaly; lipaemia retinalis; "
            "usually presents in childhood (similar to LPL-FCS); "
            "ApoC-II deficiency can occasionally present in adulthood if partial ApoC-II function retained; "
            "DISTINGUISHING: in vitro LPL activity test with vs without exogenous ApoC-II"
        ),
        "hallmark": (
            "PHENOTYPICALLY IDENTICAL TO LPL-FCS — cannot be distinguished clinically; "
            "DIAGNOSIS: LPL activity assay + ApoC-II assay + APOC2 genetic sequencing; "
            "FFP TRANSFUSION RESTORES LPL ACTIVITY — DIAGNOSTIC AND THERAPEUTIC; "
            "ERUPTIVE XANTHOMATA + PANCREATITIS in child → FCS (LPL or ApoC-II) until proven otherwise; "
            "FAT RESTRICTION <20 g/day: same first-line management as LPL-FCS; "
            "RARE: <100 cases described worldwide; "
            "GENETIC SEQUENCING: APOC2 gene (4 exons); known founder mutations in specific populations"
        ),
        "treatment_alert": (
            "FAT RESTRICTION <20 g/day: FIRST AND MOST EFFECTIVE — reduces chylomicron substrate; "
            "MCT OIL: can substitute for some fat (bypasses chylomicron pathway); "
            "FRESH FROZEN PLASMA (FFP): in acute pancreatitis — provides exogenous ApoC-II; "
            "lowers TG acutely (within hours); used both diagnostically and therapeutically; "
            "VOLANESORSEN (EU 2019): approved for FCS (LPL or ApoC-II deficiency); "
            "antisense oligonucleotide targeting APOC3 mRNA → reduces ApoC-III → reduces chylomicron retention; "
            "PLATELET MONITORING on volanesorsen (thrombocytopenia risk — mandatory weekly CBC); "
            "GEMCABENE (in trials): APOC3 inhibitor; alternative approach; "
            "APOC2 GENE THERAPY: in trials; mimetic peptides (ApoC-II-Milano); "
            "ALCOHOL ABSOLUTELY CONTRAINDICATED (same as LPL-FCS); "
            "ESTROGEN CONTRAINDICATED (raises TG); "
            "FIBRATES: limited efficacy (require functional LPL pathway)"
        ),
        "key_ddx": (
            "LPL-FCS (clinically identical; LPL activity assay + ApoC-II supplementation test distinguishes; genotype); "
            "Multifactorial chylomicronemia (heterozygous variants in multiple genes + metabolic triggers); "
            "Familial combined hyperlipidemia (mixed TG + LDL; no chylomicronemia usually); "
            "Secondary hypertriglyceridaemia (hypothyroidism, diabetes, drugs — less extreme)"
        ),
        "ldl_pattern": "LDL-C often LOW (chylomicrons dominate; LDL production may be secondarily reduced)",
        "tg_pattern": "Severe TG elevation 20–100+ mmol/L; chylomicronemia; pancreatitis risk",
        "primary_complication": "Recurrent acute pancreatitis; chronic pancreatitis; exocrine insufficiency",
        "disease_detail": (
            "ApoC-II deficiency — AR APOC2 LOF; <100 cases worldwide; "
            "presentation identical to LPL-FCS (chylomicronemia); "
            "distinguished by LPL activity assay with/without ApoC-II; FFP restores activity; "
            "management same as LPL-FCS; volanesorsen applies to both"
        ),
        "variants": [
            {"name": "p.Gln66Pro", "frequency": "Common founder mutation in ApoC-II deficiency"},
            {"name": "c.IVS2+1G>A", "frequency": "Splice donor; common European APOC2 deficiency variant"},
        ],
        "drug_ci": [
            "ALCOHOL ABSOLUTELY CONTRAINDICATED — acute TG surge → pancreatitis",
            "ESTROGEN-CONTAINING ORAL CONTRACEPTIVES CONTRAINDICATED — raise TG severely",
            "FIBRATES: limited efficacy (LPL is present but not activated — fibrate cannot compensate for missing ApoC-II)",
            "VOLANESORSEN: thrombocytopenia risk — weekly platelet monitoring mandatory; hold if <50×10⁹/L",
        ],
        "rates": {"drug_error": 0.32, "dx_delay": 0.60, "surveillance": 0.68, "sev_mild": 0.18, "sev_mod": 0.37, "sev_sev": 0.45},
    },
]


def _simulate_gene(gene_entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    patients = []
    g = gene_entry["gene"]
    r = gene_entry.get("rates", {"drug_error": 0.38, "dx_delay": 0.60, "surveillance": 0.62,
                                  "sev_mild": 0.35, "sev_mod": 0.45, "sev_sev": 0.20})
    for i in range(n):
        age = rng.randint(5, 70)
        sex = rng.choice(["M", "F"])
        # LDL in mmol/L (primary metric for LDLR/APOB/PCSK9/APOE; low for LPL/APOC2)
        if g in ("LPL", "APOC2"):
            ldl_c = round(rng.uniform(0.4, 2.5), 1)
            tg = round(rng.uniform(12.0, 80.0), 1)
        elif g == "APOE":
            ldl_c = round(rng.uniform(4.5, 10.0), 1)
            tg = round(rng.uniform(4.0, 15.0), 1)
        elif g == "ABCA1":
            ldl_c = round(rng.uniform(1.5, 4.0), 1)
            tg = round(rng.uniform(1.2, 4.5), 1)
        elif g == "LIPA":
            ldl_c = round(rng.uniform(4.0, 9.5), 1)
            tg = round(rng.uniform(2.5, 8.0), 1)
        else:
            ldl_c = round(rng.uniform(5.0, 12.5), 1)
            tg = round(rng.uniform(0.8, 2.5), 1)
        drug_error = rng.random() < r["drug_error"]
        dx_delay = rng.random() < r["dx_delay"]
        sev_roll = rng.random()
        if sev_roll < r["sev_mild"]:
            severity = "Mild"
        elif sev_roll < r["sev_mild"] + r["sev_mod"]:
            severity = "Moderate"
        else:
            severity = "Severe"
        sur_adh = rng.random() < r["surveillance"]
        patients.append({
            "patient_id": f"{g}-{seed}-{i+1:03d}",
            "gene": g,
            "age": age,
            "sex": sex,
            "ldl_c": ldl_c,
            "tg": tg,
            "drug_error": drug_error,
            "dx_delayed": dx_delay,
            "severity": severity,
            "surveillance_adherent": sur_adh,
        })
    return patients


def _cohort_stats(patients: list) -> dict:
    n = len(patients)
    if n == 0:
        return {}
    return {
        "n": n,
        "drug_error_pct": round(100 * sum(p["drug_error"] for p in patients) / n, 1),
        "dx_delayed_pct": round(100 * sum(p["dx_delayed"] for p in patients) / n, 1),
        "surveillance_adherent_pct": round(100 * sum(p["surveillance_adherent"] for p in patients) / n, 1),
        "severity_mild_pct": round(100 * sum(p["severity"] == "Mild" for p in patients) / n, 1),
        "severity_moderate_pct": round(100 * sum(p["severity"] == "Moderate" for p in patients) / n, 1),
        "severity_severe_pct": round(100 * sum(p["severity"] == "Severe" for p in patients) / n, 1),
        "mean_ldl_c": round(sum(p["ldl_c"] for p in patients) / n, 1),
        "mean_tg": round(sum(p["tg"] for p in patients) / n, 1),
    }


def _all_patients() -> list:
    all_pts = []
    for i, ge in enumerate(LIPID_GENES):
        seed = SEED_BASE + i
        pts = _simulate_gene(ge, seed, 40)
        all_pts.extend(pts)
    return all_pts


# ─── Public API functions ──────────────────────────────────────────────────────

def get_overview() -> dict:
    all_pts = _all_patients()
    agg = _cohort_stats(all_pts)
    return {
        "atlas_name": "Hereditary Lipid Disorders Atlas",
        "atlas_subtitle": "Complete 8-Gene Hereditary Lipid/Lipoprotein Disorders Atlas",
        "n_genes": 8,
        "n_patients": len(all_pts),
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
        "genes": [g["gene"] for g in LIPID_GENES],
        "description": (
            "The Hereditary Lipid Disorders Atlas covers 8 clinically actionable genes across the full spectrum of "
            "inherited dyslipidaemia: LDLR and APOB (the two most common familial hypercholesterolaemia genes, "
            "causing isolated LDL-C elevation and premature coronary artery disease), PCSK9 (GOF → FH3 / LOF → protective, "
            "the pharmacological target of evolocumab/alirocumab/inclisiran), APOE (codominant ε2/ε2 → "
            "Type III hyperlipoproteinemia with palmar xanthomata), LPL and APOC2 (the two chylomicronemia "
            "genes causing severe hypertriglyceridaemia and recurrent pancreatitis), ABCA1 (Tangier disease "
            "— near-zero HDL, orange tonsils, reverse cholesterol transport failure), and LIPA (lysosomal "
            "acid lipase deficiency — Wolman disease in infants, CESD in adults, both treatable with sebelipase alfa). "
            "Together these disorders span LDL-C elevation, TG-mediated pancreatitis, near-zero HDL, and "
            "lysosomal lipid storage. Each has specific treatment algorithms that diverge from generic dyslipidaemia "
            "management. 320 patients (8 × 40, seeds 1222–1229)."
        ),
        "aggregate_clinical": agg,
        "drug_alerts": [
            {
                "title": "FH (LDLR/APOB/PCSK9-GOF): STATINS ARE FIRST LINE — Do NOT withhold over 'young age'",
                "body": (
                    "FH children should start statins from age 8–10 years if LDL-C >5 mmol/L on diet alone. "
                    "Withholding statins over 'coronary risk not yet present' is a clinical error — atherosclerosis "
                    "begins in childhood in untreated FH. Add ezetimibe if target not met. "
                    "PCSK9 inhibitors if still above target (FOURIER trial: 27% reduction in MI)."
                ),
                "type": "danger",
            },
            {
                "title": "STATINS CONTRAINDICATED in PREGNANCY — Switch to bile acid sequestrant preconception",
                "body": (
                    "All statins are Category X in pregnancy (teratogenic). Women with FH planning pregnancy must "
                    "switch to bile acid sequestrant (cholestyramine, colestipol) before conception. "
                    "PCSK9 inhibitors: discontinue if pregnant (limited safety data). "
                    "Do not continue statins into the first trimester — congenital defects risk."
                ),
                "type": "danger",
            },
            {
                "title": "LPL/APOC2 Chylomicronemia: ALCOHOL and ESTROGEN ABSOLUTELY CONTRAINDICATED",
                "body": (
                    "A single alcohol binge can raise TG by 50–100% in LPL/ApoC-II deficient patients and trigger "
                    "acute pancreatitis. All estrogen-containing oral contraceptives are absolutely contraindicated "
                    "(raise TG dramatically). Fat restriction <20 g/day is the cornerstone of management — "
                    "no approved lipid-lowering drug (fibrates, statins) is effective in true LPL-FCS."
                ),
                "type": "danger",
            },
            {
                "title": "APOE ε2/ε2 (Type III): FIBRATE FIRST LINE — Statin alone inadequate for mixed dyslipidaemia",
                "body": (
                    "Type III HLP (dysbetalipoproteinemia) is a MIXED dyslipidaemia: BOTH TC and TG elevated. "
                    "Fibrate (fenofibrate) is first line — targets VLDL remnant overproduction and enhances LPL. "
                    "Statin alone addresses LDL-C but not remnant particles. "
                    "GEMFIBROZIL + STATIN is contraindicated (myopathy). Use fenofibrate + statin if combined needed."
                ),
                "type": "warning",
            },
            {
                "title": "LIPA/Wolman Disease: SEBELIPASE ALFA — Start IMMEDIATELY on infant diagnosis",
                "body": (
                    "Wolman disease is lethal by 6 months without enzyme replacement therapy. "
                    "Sebelipase alfa (Kanuma, 1–3 mg/kg weekly IV) should be started as soon as diagnosis confirmed. "
                    "Do NOT wait for LAL activity confirmation if adrenal calcification + hepatosplenomegaly + failure to thrive — "
                    "begin empirical ERT while testing. CESD (adult) is often misdiagnosed as NAFLD — check LAL DBS activity."
                ),
                "type": "danger",
            },
            {
                "title": "ABCA1/Tangier: NO APPROVED HDL-RAISING DRUG — Focus LDL-C and lifestyle",
                "body": (
                    "Niacin is no longer recommended (AIM-HIGH/ACCELERATE: no CVD benefit over statin + side effects). "
                    "CETP inhibitors failed in trials. Focus on cardiovascular risk reduction via LDL-C targeting, "
                    "aerobic exercise (5–10% HDL increase), and smoking cessation. "
                    "Peripheral neuropathy requires neurological monitoring."
                ),
                "type": "warning",
            },
            {
                "title": "PCSK9: LOF variants are PROTECTIVE — Do NOT misclassify as FH and treat unnecessarily",
                "body": (
                    "PCSK9 LOF variants (p.Tyr142Ter, p.Cys679Ter) reduce LDL-C 28% and coronary events 47%. "
                    "These are found in 2–3% of the population. They are NEVER a disease gene and never require treatment. "
                    "Misidentification of a PCSK9 LOF variant as pathogenic FH3 → unnecessary statin/PCSK9 therapy."
                ),
                "type": "warning",
            },
            {
                "title": "CASCADE GENETIC TESTING is MANDATORY for all index FH cases",
                "body": (
                    "Every newly diagnosed FH index case (LDLR/APOB/PCSK9-GOF) requires immediate family cascade testing. "
                    "50% of first-degree relatives carry the same variant. Each undetected relative has 10× population MI risk. "
                    "Cascade testing has the highest cost-effectiveness of any cardiovascular screening strategy."
                ),
                "type": "warning",
            },
        ],
        "clinical_pearls": [
            "LDLR-FH: tendon xanthomata (Achilles/extensors) are pathognomonic — check in every young MI patient",
            "APOE ε2/ε2: palmar xanthomata = Type III HLP until proven otherwise — treat the metabolic trigger first",
            "LPL/APOC2-FCS: refrigerated plasma cream layer = chylomicronemia — eruptive xanthomata = emergency fat restriction",
            "PCSK9-LOF: low LDL-C is PROTECTIVE, not a disease — do not treat; identify LOF for family risk stratification",
            "LIPA/CESD: child with hepatomegaly + elevated LDL-C = check LAL DBS activity — not all fatty liver is NAFLD",
            "ABCA1/Tangier: orange tonsils + near-zero HDL = Tangier disease — biopsy shows Maltese cross crystals",
            "FH cascade: every index FH case → test all first-degree relatives; Dutch score ≥6 = probable definite FH",
            "Pregnancy + FH: statins absolutely contraindicated — switch to bile acid sequestrant preconception; "
            "high CVD risk pregnancies need obstetric cardiology co-management",
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for i, ge in enumerate(LIPID_GENES):
        seed = SEED_BASE + i
        pts = _simulate_gene(ge, seed, 40)
        stats = _cohort_stats(pts)
        result[ge["gene"]] = {
            "gene": ge["gene"],
            "protein": ge["protein"],
            "alias": ge["alias"],
            "aa": ge["aa"],
            "kDa": ge["kDa"],
            "locus": ge["locus"],
            "omim_gene": ge["omim_gene"],
            "omim_disease": ge["omim_disease"],
            "inheritance": ge["inheritance"],
            "gene_class": ge["gene_class"],
            "phenotype": ge["phenotype"],
            "hallmark": ge["hallmark"],
            "treatment_alert": ge["treatment_alert"],
            "key_ddx": ge["key_ddx"],
            "ldl_pattern": ge["ldl_pattern"],
            "tg_pattern": ge["tg_pattern"],
            "primary_complication": ge["primary_complication"],
            "disease_detail": ge["disease_detail"],
            "variants": ge.get("variants", []),
            "drug_ci": ge.get("drug_ci", []),
            "stats": stats,
        }
    return result


def get_definitions() -> dict:
    return {
        "atlas": (
            "Hereditary Lipid Disorders Atlas — 8 genes: LDLR(FH1)·APOB(FDB)·PCSK9(FH3-GOF/LOF)·APOE(TypeIII)·"
            "LPL(FCS)·ABCA1(Tangier/FHA)·LIPA(Wolman/CESD)·APOC2(ApoC-II-Deficiency). "
            "320 patients, 8×40, seeds 1222–1229."
        ),
        "terms": {
            "familial_hypercholesterolaemia_fh": (
                "Autosomal dominant disorder causing lifelong elevated LDL-C due to impaired LDL receptor pathway; "
                "three causative genes: LDLR (FH1; most common), APOB (FDB/FH2), PCSK9-GOF (FH3); "
                "prevalence: hetFH 1:200–500; homFH 1:160,000–300,000; "
                "untreated hetFH: 10× population coronary risk; treatment dramatically reduces risk"
            ),
            "ldl_receptor_pathway": (
                "LDL-C circulates as LDL particles coated with ApoB-100; "
                "LDLR on hepatocyte surface binds ApoB-100 → clathrin-coated pit → endosome; "
                "endosomal acidification → ApoB-100 released → LDLR recycled to surface; "
                "cycle repeats ~150× per receptor; "
                "PCSK9 intercepts LDLR in endosome → routes LDLR to lysosome for degradation; "
                "statins upregulate LDLR expression (HMGCR inhibition → reduced intrahepatic cholesterol → LXR/SREBP signalling)"
            ),
            "pcsk9_inhibitors": (
                "Monoclonal antibodies (evolocumab, alirocumab) or siRNA (inclisiran) targeting PCSK9 protein; "
                "prevent PCSK9-mediated LDLR degradation → more LDLR recycled to hepatocyte surface → more LDL cleared; "
                "LDL-C reduction: 50–60% on top of statin; FOURIER (evolocumab): 27% reduction in MI; "
                "ODYSSEY OUTCOMES (alirocumab): 15% reduction in major events; "
                "approved for FH and high CVD-risk patients not at LDL-C target on maximal statin+ezetimibe"
            ),
            "familial_chylomicronemia_syndrome_fcs": (
                "Severe hypertriglyceridaemia (TG >10 mmol/L) caused by biallelic LOF in LPL or APOC2; "
                "chylomicrons cannot be cleared → accumulate in plasma; "
                "presentation: recurrent acute pancreatitis, eruptive xanthomata, lipaemia retinalis; "
                "management: fat restriction <20 g/day (cornerstone); volanesorsen (EU 2019); "
                "LDL-C often low (chylomicrons are TG-rich, not LDL particles)"
            ),
            "chylomicronemia": (
                "Presence of chylomicrons in fasting plasma (normally absent after 12h fast); "
                "caused by LPL LOF (FCS) or ApoC-II deficiency or massive secondary TG elevation; "
                "plasma appears milky; refrigerated overnight: cream layer floats on top; "
                "TG typically >10 mmol/L; pancreatitis risk; eruptive xanthomata"
            ),
            "eruptive_xanthomata": (
                "Tiny yellow-white papules (1–4 mm) on buttocks, trunk, extremities; "
                "occur when TG >20 mmol/L; chylomicrons taken up by dermal macrophages → lipid deposits; "
                "pathognomonic for severe hypertriglyceridaemia (LPL/ApoC-II deficiency); "
                "resolve when TG normalised"
            ),
            "lipaemia_retinalis": (
                "Fundoscopic finding: retinal vessels appear salmon-pink/orange when TG >30–40 mmol/L; "
                "caused by lipid-laden chylomicrons in retinal vessels; "
                "a clinical sign of extreme hypertriglyceridaemia; resolves with TG normalisation; "
                "vision usually preserved"
            ),
            "apoe_isoforms": (
                "ApoE exists as ε2, ε3, ε4 isoforms (2 coding SNPs rs7412 + rs429358); "
                "ε3 most common (60% allele frequency); ε2 = Cys112/Cys158 (12%); ε4 = Arg112/Arg158 (14%); "
                "ε2/ε2: reduced LDLR affinity → VLDL remnant accumulation → Type III HLP risk; "
                "ε4: higher LDL-C; 3.7× Alzheimer risk (hetE4); 12× Alzheimer risk (homE4)"
            ),
            "type_iii_hyperlipoproteinemia": (
                "Also: dysbetalipoproteinemia, familial broad-beta disease; "
                "caused by APOE ε2/ε2 + metabolic trigger (obesity, DM, hypothyroidism, etc.); "
                "mixed TC + TG elevation; β-VLDL (remnant particles) accumulate; "
                "palmar xanthomata (xanthomata striae palmaris) pathognomonic; "
                "responds dramatically to fibrate + treatment of metabolic trigger"
            ),
            "dutch_lipid_clinic_network_score": (
                "Clinical scoring system for FH diagnosis (without genetic testing); "
                "scores family history (+1–6), personal CVD history (+2), LDL-C level (+1–8), "
                "physical signs (+4 tendon xanthomata / +2 arcus before 45); "
                "score ≥8: definite FH; 6–7: probable FH; 3–5: possible FH; <3: unlikely FH; "
                "genetic testing recommended for score ≥6"
            ),
            "tangier_disease": (
                "AR ABCA1 LOF; near-zero HDL-C (<0.05 mmol/L); named after Tangier Island, Virginia; "
                "orange tonsils (pathognomonic — cholesterol ester deposits); peripheral neuropathy; "
                "hepatosplenomegaly; corneal infiltrates; premature atherosclerosis despite low LDL; "
                "fewer than 100 cases worldwide; no approved HDL-raising therapy"
            ),
            "lal_deficiency": (
                "Lysosomal acid lipase deficiency — AR LIPA LOF; spectrum from Wolman (infantile, near-zero LAL, lethal) "
                "to CESD (adult, 1–5% residual LAL, hepatic disease); "
                "adrenal calcification pathognomonic for Wolman; "
                "CESD mimics NAFLD in children; "
                "sebelipase alfa (ERT) FDA/EMA 2015 — life-saving in Wolman, slows CESD"
            ),
            "sebelipase_alfa": (
                "Recombinant human lysosomal acid lipase (ERT; marketed as Kanuma, Alexion); "
                "approved FDA/EMA 2015 for LAL deficiency; "
                "Wolman: 1–3 mg/kg weekly IV; CESD: 1 mg/kg every 2 weeks; "
                "reduces hepatic cholesteryl ester accumulation; normalises LDL-C; reduces liver fat; "
                "infusion-related reactions: premedication with antihistamine + acetaminophen"
            ),
            "volanesorsen": (
                "Antisense oligonucleotide targeting APOC3 mRNA (ApoC-III antisense); "
                "ApoC-III is a natural LPL inhibitor; volanesorsen reduces ApoC-III → enhances LPL pathway → "
                "reduces chylomicron TG; EU approved 2019 for FCS (LPL or ApoC-II deficiency); "
                "significant TG reduction (70–80%); major side effect: thrombocytopenia (weekly CBC mandatory)"
            ),
            "cascade_testing": (
                "Systematic genetic testing of relatives of an index case (proband) with a confirmed hereditary condition; "
                "for FH: test all first-degree relatives (parent, siblings, children); 50% carry same variant; "
                "cost-effectiveness: cascade FH testing saves 3.7 life-years per case detected; "
                "most efficient CVD screening strategy known; supported by all lipid guidelines (ESC/ACC/AHA)"
            ),
            "lpl_activity_assay": (
                "In vitro test measuring lipolytic activity of plasma in presence/absence of exogenous ApoC-II; "
                "distinguishes LPL-FCS (no activity WITH or WITHOUT ApoC-II) from ApoC-II deficiency "
                "(no activity without ApoC-II → RESTORED with added ApoC-II); "
                "essential step when LPL and APOC2 genetic testing is pending or inconclusive"
            ),
        },
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(f"Patients: {ov['n_patients']}, Genes: {ov['n_genes']}, Seeds: {ov['seeds']}")
    bd = get_breakdown()
    print(f"Breakdown genes: {list(bd.keys())}")
    df = get_definitions()
    print(f"Definitions terms: {list(df['terms'].keys())}")
    print("OK")
