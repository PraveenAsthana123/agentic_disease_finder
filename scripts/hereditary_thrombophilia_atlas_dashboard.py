#!/usr/bin/env python3
"""Hereditary-Thrombophilia-Atlas — Complete 8-Gene Hereditary Thrombophilia Atlas
F5       (Factor V / Factor V Leiden; 2224 aa; ~270 kDa; 1q24.2; AD;
          OMIM gene 612309; Thrombophilia-3 OMIM 188050;
          p.Arg534Gln [R506Q in mature protein] — APC resistance;
          most common hereditary thrombophilia 3-8% Europeans;
          heterozygous 5-7x VTE risk; homozygous 50-80x;
          OCP combined = 35x risk; seed SEED_BASE+0) ·
F2       (Prothrombin/Thrombin; 622 aa; ~70 kDa; 11p11.2; AD;
          OMIM gene 176930; Thrombophilia-1 OMIM 188050;
          G20210A 3'UTR — increases mRNA stability → 30% elevated prothrombin;
          2-3% Europeans; 2-5x VTE risk; seed SEED_BASE+1) ·
SERPINC1 (Antithrombin III; 464 aa; ~58 kDa; 1q25.1; AD;
          OMIM gene 107300; Antithrombin-III-Deficiency OMIM 613118;
          highest single-gene thrombophilia risk 10-50x;
          Type I (quantitative) and Type II (qualitative);
          heparin cofactor — AT-III inactivates thrombin/Xa;
          heparin resistance if Type II HBS; AT-III concentrate perioperatively;
          seed SEED_BASE+2) ·
PROC     (Protein C; 461 aa; ~62 kDa; 2q14.3; AD;
          OMIM gene 612283; Thrombophilia-3 OMIM 176860;
          vitamin K-dependent serine protease;
          warfarin skin necrosis without heparin bridge — MANDATORY overlap;
          neonatal purpura fulminans homozygous;
          3-5x VTE risk heterozygous; seed SEED_BASE+3) ·
PROS1    (Protein S alpha; 676 aa; ~70 kDa; 3q11.2; AD;
          OMIM gene 176880; Thrombophilia-5 OMIM 612336;
          free protein S is APC cofactor;
          Type I/II/III classification; OCP reduces PS → test off OCP;
          pregnancy reduces PS → test 3 months postpartum; seed SEED_BASE+4) ·
MTHFR    (Methylenetetrahydrofolate reductase; 656 aa; ~74 kDa; 1p36.22; AR;
          OMIM gene 607093; MTHFRD OMIM 236250;
          C677T homozygous 5-15% prevalence; thermolabile variant;
          hyperhomocysteinaemia; folate supplementation reduces homocysteine;
          NICE does NOT recommend routine thrombophilia screen;
          seed SEED_BASE+5) ·
THBD     (Thrombomodulin; 575 aa; ~60 kDa; 20p11.21; AD;
          OMIM gene 188040; Thrombophilia-12 OMIM 614486;
          rare; endothelial TM binds thrombin → TM-thrombin complex activates PC 1000x faster;
          TM mutations impair PC activation; complement-associated TMA;
          recombinant TM ART-123; aHUS overlap; seed SEED_BASE+6) ·
SERPINE1 (PAI-1 / Plasminogen Activator Inhibitor 1; 402 aa; ~45 kDa; 7q22.1; AR;
          OMIM gene 173360; PAI-1-Excess/PAI-1-Deficiency OMIM 173360;
          4G/4G homozygous → elevated PAI-1 → impaired fibrinolysis → thrombosis;
          BUT complete PAI-1 deficiency causes severe bleeding — OPPOSITE phenotype;
          seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1510–1517)
"""

import random

SEED_BASE = 1510

THROMBOPHILIA_GENES = [
    # ── F5 — Factor V Leiden / Most Common Hereditary Thrombophilia ──
    {
        "gene": "F5",
        "protein": "Factor V Leiden — Most Common Hereditary Thrombophilia, APC Resistance, OCP 35x Risk",
        "alias": (
            "F5; OMIM gene 612309; Thrombophilia-3 OMIM 188050; 1q24.2; 2224 aa; ~270 kDa; "
            "F5 encodes coagulation Factor V, a large procofactor that, when activated to FVa by "
            "thrombin or FXa, dramatically accelerates prothrombinase complex assembly (FXa-FVa on "
            "platelet phospholipid surface → prothrombin → thrombin). Activated protein C (APC) "
            "normally inactivates FVa by cleaving at Arg506 (R506), Arg306, and Arg679. "
            "p.Arg534Gln (historically designated R506Q in the mature protein) substitutes Gln for "
            "Arg at the primary APC cleavage site, preventing FVa inactivation — 'APC resistance.' "
            "Factor V Leiden is the most common hereditary thrombophilia: prevalence 3-8% in "
            "European populations, 1-2% in Hispanic populations, <1% in East Asian and African "
            "populations. Heterozygous FVL confers 5-7x increased VTE risk; homozygous FVL confers "
            "50-80x increased VTE risk. Combined with estrogen-containing OCP: risk amplifies to "
            "~35x — an absolute contraindication. Heterozygous FVL + pregnancy: mandatory LMWH "
            "prophylaxis antepartum and 6 weeks postpartum. APC resistance assay (APTT-based) is "
            "the initial functional screen; DNA testing confirms p.Arg534Gln. DOACs (rivaroxaban, "
            "apixaban) are non-inferior to warfarin in FVL VTE management and are preferred. "
            "Cascade testing of first-degree relatives mandatory before OCP initiation, pregnancy, "
            "or elective surgery."
        ),
        "aa": "2224 aa",
        "kDa": "~270 kDa",
        "locus": "1q24.2",
        "omim_gene": 612309,
        "omim_disease": 188050,
        "inheritance": "AD — APC resistance; incomplete clinical penetrance; OCP amplifies risk 35x",
        "gene_class": (
            "Factor V is a 2224-amino acid procofactor synthesised in hepatocytes and circulating "
            "in plasma and stored in platelet alpha-granules. Thrombin cleaves FV at Arg709, Arg1018, "
            "and Arg1545 to generate FVa (heavy and light chains linked by Ca2+), which assembles "
            "with FXa on phosphatidylserine-exposing platelet surfaces to form prothrombinase — "
            "accelerating prothrombin activation by ~300,000-fold. APC (activated protein C), "
            "bound to its cofactor protein S, inactivates FVa by sequential cleavage at Arg506 "
            "(first, rapid), Arg306 (slower), and Arg679. The Leiden variant (p.Arg534Gln, "
            "historically R506Q) eliminates the primary APC cleavage site, producing a 'partially "
            "APC-resistant' FVa that is inactivated 10x more slowly than wild-type FVa. "
            "Paradoxically, FV Leiden also has an anticoagulant function: in its uncleaved "
            "procofactor form FV acts as a cofactor for APC inactivation of FVIIIa on the protein "
            "S-FV-APC complex (TFPI pathway); Leiden FV retains this anticoagulant role, partially "
            "mitigating its procoagulant effect — explaining why penetrance is incomplete. "
            "The net result is a prothrombotic imbalance most manifest in venous beds (deep vein "
            "thrombosis, pulmonary embolism) and in high-risk situations (immobilisation, surgery, "
            "OCP, pregnancy). APC resistance ratio by APTT test: normal ratio >2.0; FVL heterozygous "
            "typically 1.4-2.0; FVL homozygous <1.4. DNA confirmation of p.Arg534Gln by PCR."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("p.R534Q Leiden AD heterozygous — APC resistance, 5-7x VTE risk", 0.85),
            ("p.R534Q Leiden AD homozygous — 50-80x VTE risk, lifelong anticoagulation", 0.10),
            ("p.W1586R HARG (HR2) haplotype variant — mild APC resistance", 0.03),
            ("other missense APC-binding domain — rare atypical APC resistance", 0.02),
        ],
        "key_alerts": [
            "F5-LEIDEN-OCP-COMBINED-35x-VTE-Risk-AVOID-ESTROGEN-CONTAINING-Contraceptives: Factor V Leiden + OCP = 35x VTE risk — estrogen-containing OCP absolutely contraindicated; use progesterone-only or non-hormonal",
            "F5-LEIDEN-HOMOZYGOUS-50-80x-Risk-Anticoagulation-Lifelong: Homozygous F5 Leiden = 50-80x VTE risk — lifelong anticoagulation mandatory after first event",
            "F5-LEIDEN-PREGNANCY-LMWH-Prophylaxis: Pregnancy + F5 Leiden = LMWH prophylaxis mandatory antepartum and postpartum; DOAC contraindicated in pregnancy",
            "F5-LEIDEN-APC-RESISTANCE-Assay-Diagnostic: APC resistance ratio (APTT-based) is the initial functional test; DNA testing confirms F5 Leiden — APC resistance without F5 Leiden = rare type II APC resistance; test DNA mandatorily",
            "F5-LEIDEN-CASCADE-Testing-First-Degree: First-degree relatives have 50% carrier risk — cascade testing before OCP/pregnancy/surgery",
            "F5-LEIDEN-DOAC-PREFERRED-Warfarin-Alternative: DOAC (rivaroxaban/apixaban) non-inferior to warfarin in F5 Leiden — preferred for long-term anticoagulation",
        ],
    },
    # ── F2 — Prothrombin G20210A ──
    {
        "gene": "F2",
        "protein": "Prothrombin G20210A — 3'UTR mRNA Stability Mechanism, WES May Miss, 2-5x VTE Risk",
        "alias": (
            "F2; OMIM gene 176930; Thrombophilia-1 OMIM 188050; 11p11.2; 622 aa; ~70 kDa; "
            "F2 encodes prothrombin (coagulation Factor II), the zymogen precursor of thrombin. "
            "Thrombin is the central effector of coagulation — it cleaves fibrinogen to fibrin, "
            "activates Factor XIII (fibrin crosslinking), activates factors V and VIII "
            "(positive feedback amplification), and activates thrombin-activatable fibrinolysis "
            "inhibitor (TAFI) to suppress fibrinolysis. G20210A (c.*97G>A) is a substitution in "
            "the 3' untranslated region (3'UTR) of the F2 gene — 20 nucleotides downstream of the "
            "stop codon — within the hexanucleotide signal sequence for polyadenylation. "
            "This variant increases mRNA stability and efficiency of 3'-end processing, resulting "
            "in approximately 30% higher circulating prothrombin levels in heterozygous carriers. "
            "Elevated prothrombin provides more substrate for thrombin generation and also impairs "
            "APC-mediated anticoagulation. Prevalence: 2-3% Europeans; 0.4-0.6% non-European "
            "populations; essentially absent in East Asian and sub-Saharan African populations. "
            "VTE risk: heterozygous 2-5x (lower than FVL); homozygous: higher risk, consider "
            "extended anticoagulation. CRITICAL: G20210A is in the 3'UTR — standard WES exome "
            "capture typically ends at the stop codon and may not capture/report 3'UTR variants. "
            "Targeted F2 G20210A assay or comprehensive panel including UTR coverage is required."
        ),
        "aa": "622 aa",
        "kDa": "~70 kDa",
        "locus": "11p11.2",
        "omim_gene": 176930,
        "omim_disease": 188050,
        "inheritance": "AD — 3'UTR gain-of-function mRNA stability; elevated prothrombin 30%",
        "gene_class": (
            "Prothrombin is a 622-amino acid vitamin K-dependent serine protease zymogen, synthesised "
            "by hepatocytes and secreted into plasma at a concentration of ~100 μg/mL (1.4 μM). "
            "The mature protein contains an N-terminal Gla domain (gamma-carboxyglutamate — vitamin K "
            "dependent), two kringle domains, and a C-terminal serine protease domain. "
            "Prothrombinase complex (FXa-FVa-phospholipid-Ca2+) cleaves prothrombin at Arg271 "
            "and Arg320 in sequence to generate alpha-thrombin — the most potent procoagulant enzyme. "
            "The G20210A 3'UTR variant lies within the polyadenylation signal hexanucleotide "
            "(AATAAA → AATAGA in the complement sense), enhancing RNA 3'-end cleavage efficiency "
            "and poly(A) tail addition. The resulting mRNA is more efficiently processed and more "
            "stable, leading to ~130% of normal prothrombin mRNA and protein levels. "
            "Elevated plasma prothrombin amplifies thrombin generation in two ways: "
            "(1) more substrate for prothrombinase → proportionally more thrombin; "
            "(2) prothrombin competes with APC substrate binding on the prothrombinase complex "
            "surface, reducing APC's anticoagulant efficiency. "
            "Unlike FVL (which directly blocks APC cleavage), G20210A acts quantitatively — "
            "its effect is titratable, explaining the lower risk (2-5x vs 5-7x for FVL). "
            "Combined FVL + G20210A double heterozygotes have approximately 20x increased VTE risk."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("G20210A 3'UTR AD heterozygous — elevated prothrombin 30%, 2-5x VTE risk", 0.88),
            ("G20210A homozygous — higher risk, extended anticoagulation considered", 0.08),
            ("other 3'UTR or coding F2 variant — very rare", 0.04),
        ],
        "key_alerts": [
            "F2-G20210A-OCP-Risk-AVOID-ESTROGEN: Prothrombin G20210A + OCP — significant VTE amplification; switch to progesterone-only contraception",
            "F2-G20210A-Elevated-Prothrombin-30pct-mRNA-Stability: G20210A increases prothrombin mRNA stability → elevated plasma prothrombin level → prothrombotic state — mechanism explanation mandatory",
            "F2-G20210A-3UTR-Variant-NOT-Coding-WES-May-Miss: G20210A is in the 3'UTR — standard WES may not capture/report it; targeted F2 G20210A assay or comprehensive panel required",
            "F2-G20210A-HOMOZYGOUS-Higher-Risk-Extended-Anticoagulation: Homozygous G20210A — extended or lifelong anticoagulation post-VTE recommended",
            "F2-G20210A-CASCADE-Testing-First-Degree: First-degree relatives — 50% carrier risk; test before hormonal contraception or pregnancy",
        ],
    },
    # ── SERPINC1 — Antithrombin III Deficiency ──
    {
        "gene": "SERPINC1",
        "protein": "Antithrombin III — Highest Single-Gene VTE Risk 10-50x, Heparin Resistance, AT-III Concentrate",
        "alias": (
            "SERPINC1; OMIM gene 107300; Antithrombin-III-Deficiency OMIM 613118; 1q25.1; 464 aa; ~58 kDa; "
            "SERPINC1 encodes antithrombin III (AT-III), the primary physiological inhibitor of "
            "thrombin and activated factor Xa (and, to a lesser extent, IXa, XIa, XIIa). "
            "AT-III is a serine protease inhibitor (serpin) that operates by presenting a 'bait' "
            "reactive site loop (RSL) that mimics the target protease's substrate, trapping the "
            "protease in a covalent acyl-enzyme complex and inhibiting it irreversibly. "
            "Heparin (unfractionated and LMWH) accelerates AT-III inhibition of thrombin and FXa "
            "by approximately 1000-fold by acting as a conformational template. "
            "Two types of AT-III deficiency: Type I — quantitative deficiency (low AT-III antigen "
            "AND activity, caused by null alleles, frameshift, splice-site, large deletions); "
            "Type II — qualitative deficiency (normal AT-III antigen, low activity), subdivided "
            "into Type II RS (reactive site variants — impaired protease inhibition) and Type II "
            "HBS (heparin-binding site variants — impaired heparin cofactor function). "
            "Type II HBS is clinically critical: AT-III activity is moderately reduced, BUT "
            "heparin cannot accelerate this dysfunctional AT-III → UFH and LMWH fail to "
            "anticoagulate effectively → heparin resistance. AT-III concentrate (Thrombate III, "
            "ATryn recombinant) is required. AT-III deficiency has the highest single-gene VTE "
            "risk of all hereditary thrombophilias: 10-50x lifetime risk."
        ),
        "aa": "464 aa",
        "kDa": "~58 kDa",
        "locus": "1q25.1",
        "omim_gene": 107300,
        "omim_disease": 613118,
        "inheritance": "AD — haploinsufficiency (Type I) or dysfunctional AT-III (Type II RS/HBS); 10-50x VTE risk",
        "gene_class": (
            "Antithrombin III is the archetypal member of the serpin (serine protease inhibitor) "
            "superfamily. Like all serpins, AT-III adopts a metastable conformation with its RSL "
            "exposed as 'bait.' When the target protease (thrombin, FXa) attacks the RSL Arg393-Ser394 "
            "bond, forming an acyl-enzyme intermediate, the RSL inserts into AT-III's central beta-"
            "sheet A, dragging the covalently attached protease across the molecule (60 Å translocation) "
            "— denaturing the protease's active site into a distorted, inactive conformation. "
            "This 'suicide substrate' mechanism creates a permanent 1:1 stoichiometric complex that "
            "is cleared by the liver. Heparin binds a specific basic patch on AT-III (Lys114, "
            "Arg129, Arg132, Lys133, Lys136 of helix D and strand S3C) via electrostatic contacts "
            "with its pentasaccharide sequence, inducing an allosteric conformational change that "
            "exposes the RSL and dramatically accelerates inhibition rates. Type II HBS variants "
            "cluster in this heparin-binding region, reducing heparin affinity — explaining why "
            "heparin therapy fails in these patients. ALWAYS perform functional (chromogenic) "
            "AT-III assay as the primary test: it detects both Type I (low antigen and function) "
            "and Type II (normal antigen, low function). Antigenic AT-III assay alone MISSES Type II. "
            "Acquired AT-III deficiency (sepsis, liver failure, DIC, nephrotic syndrome, heparin use) "
            "must be excluded before diagnosing hereditary deficiency."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("Type I null/frameshift heterozygous — quantitative deficiency AT-III <80%", 0.55),
            ("Type II RS reactive-site variant — impaired thrombin/Xa inhibition", 0.25),
            ("Type II HBS heparin-binding site variant — heparin resistance", 0.12),
            ("large deletion heterozygous — null allele Type I", 0.08),
        ],
        "key_alerts": [
            "SERPINC1-HEPARIN-RESISTANCE-Type2-HBS-AT-III-Concentrate-Required: Type II HBS (heparin-binding site) variant = profound heparin resistance — UFH/LMWH CANNOT anticoagulate effectively; AT-III concentrate replacement MANDATORY peri-operatively and for acute VTE",
            "SERPINC1-HIGHEST-SINGLE-GENE-VTE-RISK-10-50x: AT-III deficiency = highest absolute VTE risk of all hereditary thrombophilias (10-50x); lifelong anticoagulation post-first-event mandatory",
            "SERPINC1-ASSAY-TYPE-I-vs-TYPE-II-Functional-Test-First: Functional (chromogenic) AT-III assay detects both Type I and Type II; antigenic AT-III assay misses Type II — ALWAYS perform functional assay first",
            "SERPINC1-HEREDITARY-Not-Acquired-Rule-Out: Acquired AT-III deficiency (sepsis, liver disease, DIC, heparin use) must be excluded before hereditary diagnosis; repeat testing off-heparin, off-acute-illness",
            "SERPINC1-PERIOPERATIVE-AT-CONCENTRATE-Plan: Pre-operative AT-III concentrate infusion — Thrombate III/ATryn — to AT >80% activity; UFH sensitivity normalises with replacement; plan with haematology",
        ],
    },
    # ── PROC — Protein C Deficiency ──
    {
        "gene": "PROC",
        "protein": "Protein C — Warfarin Skin Necrosis Heparin-Bridge MANDATORY, Neonatal Purpura Fulminans Homozygous",
        "alias": (
            "PROC; OMIM gene 612283; Thrombophilia-3 OMIM 176860; 2q14.3; 461 aa; ~62 kDa; "
            "PROC encodes Protein C, a vitamin K-dependent serine protease zymogen. Protein C is "
            "activated by the thrombin-thrombomodulin (TM) complex on endothelial surfaces to "
            "activated Protein C (APC). APC, with its cofactor Protein S, proteolytically inactivates "
            "FVa (at Arg506, Arg306, Arg679) and FVIIIa — the two procoagulant cofactors of the "
            "amplification phase. This anticoagulant negative feedback is essential for limiting "
            "clot propagation to the site of injury. Protein C deficiency (heterozygous) confers "
            "3-5x increased VTE risk. Homozygous Protein C deficiency is a neonatal emergency: "
            "infants present within hours of birth with purpura fulminans (widespread microvascular "
            "thrombosis, skin necrosis, DIC) — EMERGENCY requiring fresh-frozen plasma or Protein C "
            "concentrate (Ceprotin) STAT. WARFARIN SKIN NECROSIS: this is the defining clinical "
            "hazard of Protein C deficiency. Warfarin suppresses vitamin K-dependent factors "
            "II, VII, IX, X, Protein C, AND Protein S. Protein C has a short half-life (~6-8 h) "
            "— much shorter than prothrombin (~60 h) and Factor X (~40 h). When warfarin is "
            "initiated, Protein C drops first (before procoagulant factors decline sufficiently), "
            "creating a transient hypercoagulable window — causing skin necrosis at fatty areas "
            "(breast, abdomen, thighs). MANDATORY RULE: ALWAYS overlap LMWH or UFH for ≥5 days "
            "when initiating warfarin, until INR is therapeutic on two consecutive days."
        ),
        "aa": "461 aa",
        "kDa": "~62 kDa",
        "locus": "2q14.3",
        "omim_gene": 612283,
        "omim_disease": 176860,
        "inheritance": "AD — haploinsufficiency (Type I) or dysfunctional PC (Type II); homozygous = neonatal purpura fulminans",
        "gene_class": (
            "Protein C is synthesised in hepatocytes as a single-chain zymogen and processed "
            "into a disulfide-linked two-chain molecule (heavy and light chains) after proteolytic "
            "removal of the activation dipeptide. The protein contains a Gla domain (requires "
            "vitamin K for functional gamma-carboxylation), two EGF-like domains, and a serine "
            "protease domain. Protein C activation is dramatically amplified by the thrombomodulin "
            "receptor on endothelial cells: thrombomodulin binds thrombin with high affinity, "
            "and the thrombin-TM complex activates Protein C approximately 1000-fold more "
            "efficiently than free thrombin alone. This ensures Protein C activation occurs "
            "preferentially at the intact endothelium (not at the fibrin clot), providing "
            "spatially regulated anticoagulant feedback. The Gla domain of Protein C is "
            "vitamin K-dependent: warfarin blocks VKORC1 (vitamin K epoxide reductase complex 1), "
            "preventing recycling of vitamin K from the epoxide form, reducing gamma-carboxylation "
            "of Glu residues in the Gla domain, and impairing membrane-phospholipid binding "
            "essential for Protein C activity. Protein C deficiency Type I (low antigen AND "
            "activity) is more common; Type II (normal antigen, reduced activity — detected by "
            "functional/chromogenic assay not clot-based) can be missed if only antigenic testing "
            "is performed. Chromogenic Protein C assay is the preferred diagnostic test."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("Type I missense heterozygous <50% activity — quantitative PC deficiency", 0.60),
            ("Type II heterozygous functional variant — normal antigen, reduced function", 0.25),
            ("homozygous — neonatal purpura fulminans, complete PC absence", 0.08),
            ("compound heterozygous — severe PC deficiency", 0.07),
        ],
        "key_alerts": [
            "PROC-WARFARIN-SKIN-NECROSIS-Heparin-Bridge-MANDATORY: Warfarin initiation WITHOUT heparin bridge causes Protein C drop before factors II/VII/IX/X → hypercoagulable → skin necrosis; ALWAYS overlap LMWH/UFH for ≥5 days and until INR therapeutic",
            "PROC-NEONATAL-PURPURA-FULMINANS-Homozygous-EMERGENCY: Homozygous Protein C deficiency = neonatal purpura fulminans (hours after birth) — EMERGENCY; fresh-frozen plasma/Protein C concentrate STAT; consider liver transplant",
            "PROC-FUNCTIONAL-ASSAY-Chromogenic-Not-Clot-Based: Chromogenic Protein C assay required — clot-based assays miss Type II variants with normal antigen but absent function; send functional AND antigenic",
            "PROC-PROTEIN-C-CONCENTRATE-Ceprotin-Severe-Deficiency: Ceprotin (plasma-derived protein C concentrate) for severe/homozygous PC deficiency and purpura fulminans — compassionate use/licensed in EU",
            "PROC-OCP-Raises-PC-Test-Off-OCP-Borderline: OCP raises Protein C slightly — borderline results should be repeated off OCP; pregnancy reduces PC, repeat 3 months postpartum",
        ],
    },
    # ── PROS1 — Protein S Deficiency ──
    {
        "gene": "PROS1",
        "protein": "Protein S — OCP Confound Test-Off-OCP Mandatory, Pregnancy Test 3M Postpartum, Type I/II/III Classification",
        "alias": (
            "PROS1; OMIM gene 176880; Thrombophilia-5 OMIM 612336; 3q11.2; 676 aa; ~70 kDa; "
            "PROS1 encodes Protein S alpha, a vitamin K-dependent plasma glycoprotein that serves "
            "as the critical cofactor for activated Protein C (APC). Free Protein S (not bound to "
            "C4b-binding protein, C4BP) is the biologically active form. In plasma, approximately "
            "60-70% of Protein S is bound to C4BP (an acute phase reactant) and is inactive; "
            "30-40% circulates as free Protein S. Free Protein S binds APC and enhances its "
            "proteolytic inactivation of FVa and FVIIIa on phospholipid surfaces. "
            "Protein S deficiency classification: Type I — both total and free PS low (and activity "
            "low); most common, caused by large deletions, frameshift, or severe missense. "
            "Type II — functional defect: normal total and free PS antigen, but reduced APC "
            "cofactor activity in functional assay; rare variant type. "
            "Type III — free PS low only (total PS normal or near-normal); caused by missense "
            "variants affecting C4BP binding or free PS equilibrium; most common type after Type I. "
            "CRITICAL TESTING CONFOUNDS: OCP dramatically reduces free Protein S (estrogen "
            "increases C4BP → more PS bound → less free) — thrombophilia testing ON OCP gives "
            "false-positive PS deficiency. Test ≥3 months after stopping OCP. Pregnancy also "
            "reduces free Protein S (to levels that meet Type III criteria in normal pregnancy) "
            "— do NOT test during pregnancy; retest ≥3 months postpartum. Large PROS1 deletions "
            "overlapping the pseudogene PROS2 region are common and require MLPA for detection."
        ),
        "aa": "676 aa",
        "kDa": "~70 kDa",
        "locus": "3q11.2",
        "omim_gene": 176880,
        "omim_disease": 612336,
        "inheritance": "AD — haploinsufficiency (Type I/III) or dysfunctional PS (Type II); free PS is APC cofactor",
        "gene_class": (
            "Protein S is a multidomain vitamin K-dependent glycoprotein: Gla domain (membrane "
            "binding), TSR (thrombin-sensitive region — the 'thrombin-cleavage' loop, though NOT "
            "cleaved by thrombin in normal physiology), four EGF-like domains, and a SHBG-like "
            "globular domain. The SHBG-like domain mediates binding to C4b-binding protein (C4BP), "
            "which sequesters the majority of plasma Protein S. C4BP is an acute-phase protein "
            "that rises during inflammation, infection, and pregnancy, reducing free PS — this is "
            "why PS levels fall in inflammatory states and pregnancy and why PS-based thrombophilia "
            "testing in these contexts gives misleading results. "
            "Free Protein S enhances APC activity by positioning APC optimally on the platelet "
            "or endothelial membrane phospholipid surface for FVa and FVIIIa substrate engagement. "
            "Warfarin reduces Protein S (vitamin K-dependent Gla carboxylation) — warfarin "
            "WITHOUT heparin bridge can cause PS and PC to drop together before procoagulant "
            "factors fall, creating a hypercoagulable window → skin necrosis (same mechanism as "
            "PROC deficiency). MLPA is mandatory for PROS1 if sequencing is non-diagnostic in "
            "a family with confirmed Protein S deficiency: large deletions spanning PROS1 "
            "and the PROS2 pseudogene region account for a significant fraction of PROS1 alleles. "
            "Laboratory diagnosis: measure free PS, total PS, and APC cofactor activity together."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("Type I large deletion/frameshift heterozygous — low total and free PS <50%", 0.45),
            ("Type III missense — low free PS only, normal total PS", 0.30),
            ("Type II functional defect — normal antigen, reduced APC cofactor activity", 0.20),
            ("large deletion PROS1-PROS2 region — MLPA required", 0.05),
        ],
        "key_alerts": [
            "PROS1-FREE-PROTEIN-S-Low-OCP-Confound-Test-Off-OCP: OCP dramatically reduces free Protein S — thrombophilia testing on OCP gives false-positive PS deficiency; ALWAYS test ≥3 months after stopping OCP",
            "PROS1-PREGNANCY-Reduces-Free-PS-Test-3M-Postpartum: Pregnancy physiologically reduces free Protein S — do NOT test during pregnancy; repeat ≥3 months postpartum for accurate result",
            "PROS1-TYPE-III-Low-Free-Only-Total-NORMAL-Test-Both: Type III PS deficiency — low free PS with normal total; Type III is commonest; measure both free and total PS; functional PS assay for Type II",
            "PROS1-WARFARIN-NECROSIS-Risk-Heparin-Bridge: Similar to Protein C deficiency — warfarin without heparin bridge risks PS drop → skin necrosis; overlap heparin mandatory when initiating warfarin",
            "PROS1-LARGE-DELETION-PROS1-PROS2-MLPA-Mandatory: Large deletions spanning PROS1 and pseudogene PROS2 region common — MLPA mandatory if sequencing non-diagnostic in confirmed PS deficiency family",
        ],
    },
    # ── MTHFR — Methylenetetrahydrofolate Reductase ──
    {
        "gene": "MTHFR",
        "protein": "MTHFR C677T — Hyperhomocysteinaemia, Folate Therapy, NOT Routine VTE Screen per NICE",
        "alias": (
            "MTHFR; OMIM gene 607093; MTHFRD OMIM 236250; 1p36.22; 656 aa; ~74 kDa; "
            "MTHFR encodes methylenetetrahydrofolate reductase, the enzyme that irreversibly "
            "reduces 5,10-methylenetetrahydrofolate (5,10-MTHF) to 5-methyltetrahydrofolate "
            "(5-MTHF), the predominant circulating folate form. 5-MTHF donates its methyl group "
            "to homocysteine (via methionine synthase/MS and its cofactor vitamin B12) to regenerate "
            "methionine — the methyl donor for hundreds of cellular methylation reactions. "
            "MTHFR C677T (p.Ala222Val, c.665C>T): the pyrimidine substitution renders the enzyme "
            "thermolabile — it dissociates from its FAD cofactor at 37°C more readily than wild-type, "
            "reducing enzymatic activity to ~30% of normal in TT homozygotes. "
            "Homozygous TT: prevalence 5-15% in European and Hispanic populations, 10-15% in "
            "Mediterranean and Asian populations. "
            "Elevated plasma homocysteine (>15 μmol/L) is the putative thrombophilic effector — "
            "it damages endothelium, promotes oxidative stress, and activates coagulation. "
            "HOWEVER: the independent VTE risk of MTHFR C677T is CONTROVERSIAL. "
            "NICE (2012) and BCSH guidelines explicitly state that MTHFR testing should NOT be "
            "included in routine thrombophilia screens because the attributable VTE risk is modest "
            "and inconsistent across studies. Measure plasma homocysteine, not just genotype. "
            "A1298C (p.Glu429Ala): reduces MTHFR activity ~60% of normal when homozygous; "
            "C677T/A1298C compound heterozygous can significantly elevate homocysteine. "
            "Treatment: 5-MTHF (methylfolate, 400-800 μg/day) bypasses the impaired step."
        ),
        "aa": "656 aa",
        "kDa": "~74 kDa",
        "locus": "1p36.22",
        "omim_gene": 607093,
        "omim_disease": 236250,
        "inheritance": "AR — thermolabile TT homozygous reduces MTHFR to 30%; hyperhomocysteinaemia; controversial VTE risk",
        "gene_class": (
            "MTHFR is a flavoprotein (FAD-dependent) oxidoreductase that functions as a homodimer, "
            "with each subunit containing a catalytic N-terminal TIM barrel domain and a regulatory "
            "C-terminal domain. The C677T substitution (Ala222Val) is located in the FAD-binding "
            "motif of the catalytic domain; the Val222 side chain is slightly larger and less "
            "flexible than Ala, reducing FAD binding affinity at physiological temperature — "
            "hence 'thermolabile.' At 46°C (in vitro thermostability test), TT homozygous MTHFR "
            "loses activity approximately twice as fast as the CC genotype. In vivo at 37°C, "
            "reduced FAD affinity translates to approximately 70% reduction in enzyme activity "
            "in TT homozygotes under folate-replete conditions and up to 30% residual activity "
            "under folate-deficient conditions (folate stabilises FAD binding). "
            "The homocysteine → methionine remethylation pathway critically requires "
            "5-MTHF (MTHFR product), methionine synthase (MS/MTR), and cobalamin (vitamin B12) "
            "as the MS cofactor. B12 deficiency mimics and amplifies MTHFR C677T hyperhomocysteinemia. "
            "5-MTHF supplementation bypasses the MTHFR enzyme — restoring homocysteine remethylation "
            "flux without requiring functional MTHFR. Betaine (trimethylglycine) provides an "
            "alternative remethylation pathway via BHMT enzyme, used in severe MTHFR deficiency. "
            "For VTE risk stratification, measure plasma homocysteine; genotype alone is insufficient."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("C677T homozygous TT — thermolabile MTHFR 30% activity, moderate hyperhomocysteinemia", 0.55),
            ("C677T/A1298C compound heterozygous — significant homocysteine elevation", 0.35),
            ("A1298C homozygous CC — mild-moderate MTHFR reduction", 0.10),
        ],
        "key_alerts": [
            "MTHFR-C677T-FOLATE-SUPPLEMENTATION-Reduces-Homocysteine: C677T homozygous — 5-methyltetrahydrofolate (5-MTHF) supplementation 400-800 mcg/day reduces plasma homocysteine; supplement before confirming thrombophilia impact",
            "MTHFR-NOT-ROUTINE-THROMBOPHILIA-SCREEN-NICE-Guideline: NICE/BCSH guidelines do NOT recommend MTHFR testing as standard thrombophilia screen — independent VTE risk uncertain; measure plasma homocysteine instead if concerned",
            "MTHFR-MEASURE-PLASMA-HOMOCYSTEINE-Not-Genotype-Alone: Elevated homocysteine (>15 μmol/L) is the risk factor — MTHFR C677T genotype without elevated homocysteine is LOW RISK; test plasma homocysteine, not just genotype",
            "MTHFR-COMPOUND-HETEROZYGOUS-C677T-A1298C-Significant: C677T/A1298C compound heterozygous may elevate homocysteine significantly — treat with 5-MTHF; check B12 and folate levels",
            "MTHFR-B12-B6-DEFICIENCY-Worsens-Hyperhomocysteinaemia: B12 and B6 deficiency amplify MTHFR hyperhomocysteinaemia — check and supplement if deficient",
        ],
    },
    # ── THBD — Thrombomodulin ──
    {
        "gene": "THBD",
        "protein": "Thrombomodulin — Rare, Protein C Activation Impaired, aHUS Complement Overlap",
        "alias": (
            "THBD; OMIM gene 188040; Thrombophilia-12 OMIM 614486; 20p11.21; 575 aa; ~60 kDa; "
            "THBD encodes thrombomodulin (TM), a multidomain transmembrane glycoprotein expressed "
            "on the luminal surface of all vascular endothelial cells. TM is the critical "
            "anticoagulant receptor that switches thrombin from its procoagulant (fibrinogen-cleaving) "
            "function to an anticoagulant function (Protein C activation). "
            "Thrombin bound to TM cannot cleave fibrinogen or PAR-1 (platelet receptor), but "
            "instead activates Protein C with ~1000x greater efficiency than free thrombin alone. "
            "The thrombin-TM complex also activates TAFI (thrombin-activatable fibrinolysis "
            "inhibitor) — but this is a procoagulant effect (suppresses fibrinolysis). "
            "TM also binds and activates the complement pathway regulator thrombin-TM-EPCR complex "
            "interacts with complement C3b — explaining the aHUS (atypical haemolytic uremic "
            "syndrome) overlap: THBD mutations can cause thrombotic microangiopathy (TMA) via "
            "complement-mediated endothelial damage, indistinguishable from complement pathway "
            "mutations (CFH, CFI, MCP/CD46, C3). THBD-associated VTE and TMA are RARE; "
            "pathogenicity of THBD variants must be confirmed by specialist before attributing "
            "VTE to THBD. Recombinant thrombomodulin (ART-123/thrombomodulin alpha) received "
            "conditional approval in Japan for DIC management."
        ),
        "aa": "575 aa",
        "kDa": "~60 kDa",
        "locus": "20p11.21",
        "omim_gene": 188040,
        "omim_disease": 614486,
        "inheritance": "AD — rare dominant-negative or haploinsufficient; impaired PC activation; aHUS overlap",
        "gene_class": (
            "Thrombomodulin is a type I transmembrane glycoprotein with five structural domains: "
            "(1) lectin-like domain (domain 1): binds HIGH-mobility group box protein 1 (HMGB1) "
            "— anti-inflammatory function; (2) EGF-like repeat domain (domains 2-6): six tandem "
            "EGF-like repeats; the thrombin-binding site is EGF4-6, and EGF5-6 is specifically "
            "required for Protein C activation; (3) serine/threonine-rich domain (domain 3): "
            "O-glycosylation and chondroitin sulphate modification site; "
            "(4) transmembrane domain; (5) cytoplasmic tail. "
            "When thrombin binds EGF4-6 of TM, its active site is repositioned and partially "
            "occluded for fibrinogen/PAR-1 substrates but optimally aligned for Protein C "
            "(which binds to EPCR/endothelial Protein C receptor in the activation complex). "
            "Pathogenic THBD variants in the EGF-like domain impair either thrombin binding "
            "affinity or Protein C presentation — reducing PC activation. "
            "For aHUS/TMA workup in a THBD variant carrier: complement panel (C3, C4, CH50, "
            "factor H, anti-factor H antibodies, factor I, CD46/MCP expression on cells, "
            "ADAMTS13 to exclude TTP) is mandatory. Eculizumab (anti-C5 monoclonal antibody) "
            "is the treatment for complement-mediated TMA in THBD-aHUS if complement activation "
            "is confirmed. Genetic testing of the complement pathway genes should be co-ordered "
            "in any THBD patient presenting with TMA."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("p.G455E variant — impaired PC activation EGF domain", 0.35),
            ("p.A473T rare EGF domain variant — reduced thrombin-TM-PC activation", 0.25),
            ("p.C537R disulfide bond disruption — EGF domain misfolding", 0.20),
            ("splice-site loss-of-function — reduced TM expression", 0.20),
        ],
        "key_alerts": [
            "THBD-RARE-CONFIRM-WITH-SPECIALIST: THBD variants are rare; confirm pathogenicity with specialist haematologist/coagulation centre before attributing VTE to THBD — low prior probability",
            "THBD-PROTEIN-C-ACTIVATION-Impaired-PC-Level-May-Be-Normal: THBD dysfunction impairs PC activation by thrombin-TM complex — plasma PC level may be normal; functional TM-mediated PC activation assay required",
            "THBD-aHUS-OVERLAP-Complement-Workup: THBD-associated aHUS — complement pathway activation; send complement panel (C3, C4, CH50, factor H, anti-factor H Ab) in THBD with TMA phenotype; eculizumab if complement-mediated",
            "THBD-ART-123-Recombinant-TM-Research: Recombinant thrombomodulin (ART-123) — conditional approval Japan for DIC; research tool for understanding TM biology; enrol eligible patients in TM-pathway trials",
        ],
    },
    # ── SERPINE1 — PAI-1 / Plasminogen Activator Inhibitor 1 ──
    {
        "gene": "SERPINE1",
        "protein": "PAI-1 (SERPINE1) — 4G/4G Elevated Impaired Fibrinolysis vs Complete Deficiency Bleeding OPPOSITE Phenotype",
        "alias": (
            "SERPINE1; OMIM gene 173360; PAI-1-Excess/PAI-1-Deficiency OMIM 173360; 7q22.1; 402 aa; ~45 kDa; "
            "SERPINE1 encodes plasminogen activator inhibitor type 1 (PAI-1), the primary physiological "
            "inhibitor of tissue plasminogen activator (tPA) and urokinase plasminogen activator "
            "(uPA). PAI-1 is a serine protease inhibitor (serpin) — it inactivates tPA and uPA by "
            "the same 'suicide substrate' RSL insertion mechanism as AT-III inhibiting thrombin. "
            "By inhibiting tPA/uPA, PAI-1 limits fibrinolysis (plasminogen activation → plasmin → "
            "fibrin degradation), preserving clot stability. "
            "The 4G/5G promoter polymorphism (c.-675 4G/5G, rs1799889): a single-nucleotide "
            "insertion/deletion in the PAI-1 promoter. The 4G allele (deletion) lacks a binding "
            "site for a transcriptional repressor — resulting in higher PAI-1 transcription and "
            "elevated plasma PAI-1 activity. The 5G allele (insertion) has the repressor-binding "
            "site — lower PAI-1 levels. Genotypes: 4G/4G homozygous → highest PAI-1 levels → "
            "impaired fibrinolysis → increased VTE risk; 4G/5G heterozygous → intermediate; "
            "5G/5G homozygous → lowest PAI-1 → most efficient fibrinolysis. "
            "CRITICAL OPPOSITE PHENOTYPE: complete PAI-1 deficiency (rare biallelic frameshift/null) "
            "causes severe bleeding — not thrombosis. Without PAI-1 to regulate fibrinolysis, "
            "clots lyse too rapidly → delayed wound healing, post-surgical bleeding, "
            "menorrhagia, haemarthrosis. Treatment: antifibrinolytics (tranexamic acid). "
            "The 4G/5G polymorphism is COMMON (not rare) and its independent VTE risk is "
            "controversial — not a routine thrombophilia screen."
        ),
        "aa": "402 aa",
        "kDa": "~45 kDa",
        "locus": "7q22.1",
        "omim_gene": 173360,
        "omim_disease": 173360,
        "inheritance": "AR complete deficiency = severe bleeding; 4G/4G common promoter polymorphism = mild thrombotic tendency",
        "gene_class": (
            "PAI-1 is a 402-amino acid serpin and the fastest-acting serine protease inhibitor "
            "known — with a second-order rate constant for tPA inhibition of ~10^7 M-1 s-1. "
            "Like AT-III, PAI-1 presents a reactive site loop (RSL) with Met-Arg358 (P1-P1') "
            "as the pseudo-substrate for tPA/uPA. The metastable active (open) conformation of "
            "PAI-1 has a very short half-life (~2 hours in plasma at 37°C) before spontaneous "
            "conversion to a latent (closed) inactive form — the RSL inserts into the beta-sheet "
            "A without protease attack, locking the inhibitor in an inactive conformation. "
            "Vitronectin (in plasma and extracellular matrix) stabilises PAI-1 in its active form "
            "by binding to a specific exosite, prolonging half-life. The 4G/5G polymorphism alters "
            "a protein binding site for a transcriptional repressor (Sp1-like factor). In the "
            "4G allele, the repressor cannot bind → PAI-1 gene is transcribed at higher rates → "
            "elevated PAI-1 plasma activity (measured as PAI-1 activity in IU/mL). "
            "Elevated PAI-1 is also observed in metabolic syndrome, obesity, type 2 diabetes, "
            "and after myocardial infarction (where it contributes to impaired coronary artery "
            "thrombolysis). For plasma PAI-1 testing: measure functional PAI-1 activity "
            "(chromogenic assay), not antigen alone, as latent PAI-1 has normal antigen but no "
            "activity. In thrombolysis for massive PE, the 4G/4G genotype may theoretically "
            "require higher or prolonged tPA dosing to overcome PAI-1 inhibition — though "
            "evidence for dose adjustment is limited."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("4G/4G homozygous — elevated PAI-1, impaired fibrinolysis, thrombophilic tendency", 0.65),
            ("4G/5G heterozygous — intermediate PAI-1, moderate fibrinolysis impairment", 0.25),
            ("frameshift biallelic — complete PAI-1 deficiency BLEEDING phenotype", 0.10),
        ],
        "key_alerts": [
            "SERPINE1-4G-4G-ELEVATED-PAI1-Impaired-Fibrinolysis-VTE-Risk: 4G/4G homozygous → elevated plasma PAI-1 → impaired tPA-mediated fibrinolysis → clot persistence → VTE risk; check plasma PAI-1 activity level",
            "SERPINE1-COMPLETE-PAI1-DEFICIENCY-CAUSES-BLEEDING-NOT-Thrombosis: Rare biallelic frameshift causing complete PAI-1 deficiency → OPPOSITE phenotype: severe bleeding (not thrombosis); treat bleeds with antifibrinolytics (tranexamic acid)",
            "SERPINE1-4G5G-POLYMORPHISM-Controversial-Screen-Not-Routine: 4G/5G is a common polymorphism with modest VTE risk association — not a routine thrombophilia screen; clinical utility in VTE risk stratification remains debated",
            "SERPINE1-TPA-ELEVATED-EFFECTIVENESS-4G4G: tPA-based thrombolysis may be relatively less effective in 4G/4G due to PAI-1 inhibition — ensure adequate tPA dose if thrombolysis required for massive PE",
        ],
    },
]


def _make_cohort(gene_data: dict) -> list:
    r = random.Random(gene_data["seed"])
    gene = gene_data["gene"]
    etiologies = gene_data["etiologies"]
    pts = []

    for i in range(gene_data["n_patients"]):
        # Draw etiology
        roll = r.random()
        cumul = 0.0
        etiol = etiologies[-1][0]
        for et, prob in etiologies:
            cumul += prob
            if roll < cumul:
                etiol = et
                break

        # Sex distribution — VTE generally slightly female-predominant due to OCP/pregnancy
        if gene in ("F5", "F2", "PROS1"):
            sex = "F" if r.random() < 0.58 else "M"  # female excess due to OCP/pregnancy triggers
        elif gene in ("SERPINC1", "PROC"):
            sex = "F" if r.random() < 0.54 else "M"
        elif gene == "MTHFR":
            sex = "M" if r.random() < 0.52 else "F"
        else:
            sex = "M" if r.random() < 0.50 else "F"

        # Onset age for first VTE event (years)
        onset_ranges = {
            "F5":       (20, 55),    # wide range, OCP/pregnancy triggers in young women
            "F2":       (22, 55),    # similar to FVL
            "SERPINC1": (18, 45),    # high risk, often first event young
            "PROC":     (15, 50),    # first event young, neonatal in homozygous
            "PROS1":    (18, 50),    # similar to PROC
            "MTHFR":    (30, 65),    # older, often metabolic co-factors
            "THBD":     (20, 55),    # variable
            "SERPINE1": (25, 60),    # older, metabolic syndrome association
        }
        lo, hi = onset_ranges[gene]
        onset_y = round(lo + r.random() * (hi - lo), 1)
        dx_delay_m = round(r.gauss(18, 14))  # months to genetic diagnosis after VTE
        if dx_delay_m < 1:
            dx_delay_m = 1

        # VTE event type
        vte_types = ["DVT", "PE", "Combined DVT+PE", "cerebral", "splanchnic", "upper limb"]
        vte_weights = [0.42, 0.28, 0.18, 0.05, 0.04, 0.03]
        vte_roll = r.random()
        vte_cumul = 0.0
        vte_event = vte_types[-1]
        for vt, wt in zip(vte_types, vte_weights):
            vte_cumul += wt
            if vte_roll < vte_cumul:
                vte_event = vt
                break

        # Anticoagulation type
        ac_types = ["DOAC", "warfarin", "LMWH", "none"]
        ac_weights = [0.50, 0.28, 0.15, 0.07]
        ac_roll = r.random()
        ac_cumul = 0.0
        ac_type = ac_types[0]
        for at, wt in zip(ac_types, ac_weights):
            ac_cumul += wt
            if ac_roll < ac_cumul:
                ac_type = at
                break

        flags = {
            "vte_event_type": vte_event,
            "anticoagulation_type": ac_type,
        }

        if gene == "F5":
            flags["ocp_associated"] = sex == "F" and r.random() < 0.45
            flags["pregnancy_loss"] = sex == "F" and r.random() < 0.22
            flags["warfarin_skin_necrosis"] = r.random() < 0.04
            flags["heparin_resistance"] = False
            flags["cascade_tested"] = r.random() < 0.60
            flags["homocysteine_elevated"] = r.random() < 0.12
            flags["folate_supplemented"] = False
            flags["homozygous_leiden"] = "homozygous" in etiol.lower()
            flags["apc_resistance_ratio_tested"] = r.random() < 0.78
            flags["doac_prescribed"] = ac_type == "DOAC"

        elif gene == "F2":
            flags["ocp_associated"] = sex == "F" and r.random() < 0.40
            flags["pregnancy_loss"] = sex == "F" and r.random() < 0.18
            flags["warfarin_skin_necrosis"] = r.random() < 0.02
            flags["heparin_resistance"] = False
            flags["cascade_tested"] = r.random() < 0.58
            flags["homocysteine_elevated"] = r.random() < 0.15
            flags["folate_supplemented"] = False
            flags["wes_missed_variant"] = r.random() < 0.22  # 3'UTR miss rate
            flags["targeted_f2_assay_done"] = r.random() < 0.68

        elif gene == "SERPINC1":
            flags["ocp_associated"] = sex == "F" and r.random() < 0.28
            flags["pregnancy_loss"] = sex == "F" and r.random() < 0.30
            flags["warfarin_skin_necrosis"] = r.random() < 0.03
            flags["heparin_resistance"] = "HBS" in etiol
            flags["cascade_tested"] = r.random() < 0.64
            flags["homocysteine_elevated"] = r.random() < 0.10
            flags["folate_supplemented"] = False
            flags["at_concentrate_used"] = flags["heparin_resistance"] or r.random() < 0.28
            flags["functional_assay_performed"] = r.random() < 0.72
            flags["acquired_deficiency_excluded"] = r.random() < 0.68
            flags["type_i"] = "Type I" in etiol
            flags["type_ii_hbs"] = "HBS" in etiol

        elif gene == "PROC":
            flags["ocp_associated"] = sex == "F" and r.random() < 0.25
            flags["pregnancy_loss"] = sex == "F" and r.random() < 0.35
            flags["warfarin_skin_necrosis"] = r.random() < 0.14
            flags["heparin_bridge_given"] = r.random() < 0.62
            flags["heparin_resistance"] = False
            flags["cascade_tested"] = r.random() < 0.58
            flags["homocysteine_elevated"] = r.random() < 0.10
            flags["folate_supplemented"] = False
            flags["neonatal_purpura_fulminans"] = "neonatal" in etiol.lower() or "homozygous" in etiol.lower()
            flags["chromogenic_assay_done"] = r.random() < 0.70
            flags["pc_concentrate_used"] = flags["neonatal_purpura_fulminans"] or r.random() < 0.10

        elif gene == "PROS1":
            flags["ocp_associated"] = sex == "F" and r.random() < 0.50
            flags["pregnancy_loss"] = sex == "F" and r.random() < 0.32
            flags["warfarin_skin_necrosis"] = r.random() < 0.10
            flags["heparin_bridge_given"] = r.random() < 0.60
            flags["heparin_resistance"] = False
            flags["cascade_tested"] = r.random() < 0.56
            flags["homocysteine_elevated"] = r.random() < 0.10
            flags["folate_supplemented"] = False
            flags["tested_off_ocp"] = not flags["ocp_associated"] or r.random() < 0.52
            flags["free_ps_tested"] = r.random() < 0.80
            flags["total_ps_tested"] = r.random() < 0.78
            flags["mlpa_performed"] = r.random() < 0.38
            flags["type_iii"] = "Type III" in etiol

        elif gene == "MTHFR":
            flags["ocp_associated"] = sex == "F" and r.random() < 0.20
            flags["pregnancy_loss"] = sex == "F" and r.random() < 0.15
            flags["warfarin_skin_necrosis"] = False
            flags["heparin_resistance"] = False
            flags["cascade_tested"] = r.random() < 0.40
            flags["homocysteine_elevated"] = r.random() < 0.62  # key finding in MTHFR
            flags["folate_supplemented"] = flags["homocysteine_elevated"] and r.random() < 0.72
            flags["b12_checked"] = r.random() < 0.68
            flags["b12_deficient"] = flags["b12_checked"] and r.random() < 0.22
            flags["five_mthf_prescribed"] = flags["folate_supplemented"] and r.random() < 0.58
            flags["nice_guideline_discussed"] = r.random() < 0.44

        elif gene == "THBD":
            flags["ocp_associated"] = sex == "F" and r.random() < 0.20
            flags["pregnancy_loss"] = sex == "F" and r.random() < 0.20
            flags["warfarin_skin_necrosis"] = r.random() < 0.04
            flags["heparin_resistance"] = False
            flags["cascade_tested"] = r.random() < 0.42
            flags["homocysteine_elevated"] = r.random() < 0.10
            flags["folate_supplemented"] = False
            flags["ahus_overlap"] = r.random() < 0.28
            flags["complement_workup_done"] = flags["ahus_overlap"] and r.random() < 0.72
            flags["eculizumab_eligible"] = flags["ahus_overlap"] and r.random() < 0.45
            flags["specialist_confirmed"] = r.random() < 0.60
            flags["pc_activation_assay"] = r.random() < 0.35

        elif gene == "SERPINE1":
            flags["ocp_associated"] = sex == "F" and r.random() < 0.22
            flags["pregnancy_loss"] = sex == "F" and r.random() < 0.15
            flags["warfarin_skin_necrosis"] = False
            flags["heparin_resistance"] = False
            flags["cascade_tested"] = r.random() < 0.35
            flags["homocysteine_elevated"] = r.random() < 0.15
            flags["folate_supplemented"] = False
            flags["bleeding_phenotype"] = "deficiency" in etiol.lower() and "bleeding" in etiol.lower()
            flags["four_g_four_g"] = "4G/4G" in etiol
            flags["pai1_activity_tested"] = r.random() < 0.50
            flags["metabolic_syndrome"] = r.random() < 0.38
            flags["antifibrinolytic_used"] = flags["bleeding_phenotype"] and r.random() < 0.78

        pts.append({
            "pid": f"{gene}-{i+1:03d}",
            "gene": gene,
            "etiology": etiol,
            "sex": sex,
            "age_onset_years": onset_y,
            "dx_delay_months": dx_delay_m,
            **flags,
        })
    return pts


def get_overview():
    all_patients = []
    gene_summaries = []

    for gd in THROMBOPHILIA_GENES:
        pts = _make_cohort(gd)
        all_patients.extend(pts)

        gene_summaries.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "etiologies": [e[0] for e in gd["etiologies"]],
            "key_alerts": gd["key_alerts"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
        })

    n = len(all_patients)

    def g_pts(gene):
        return [p for p in all_patients if p["gene"] == gene]

    def pct(lst, key, val=True):
        if not lst:
            return 0.0
        return round(100 * sum(1 for p in lst if p.get(key) == val) / len(lst), 1)

    f5       = g_pts("F5")
    f2       = g_pts("F2")
    serpc1   = g_pts("SERPINC1")
    proc     = g_pts("PROC")
    pros1    = g_pts("PROS1")
    mthfr    = g_pts("MTHFR")
    thbd     = g_pts("THBD")
    serpe1   = g_pts("SERPINE1")

    mean_delay = round(sum(p["dx_delay_months"] for p in all_patients) / n, 1)

    # VTE event type counts
    pe_pct = round(100 * sum(1 for p in all_patients if "PE" in p.get("vte_event_type", "")) / n, 1)
    dvt_pct = round(100 * sum(1 for p in all_patients if p.get("vte_event_type") == "DVT") / n, 1)

    # Anticoagulation
    doac_pct = round(100 * sum(1 for p in all_patients if p.get("anticoagulation_type") == "DOAC") / n, 1)

    # Warfarin skin necrosis across all
    warfarin_necrosis_pct = round(100 * sum(1 for p in all_patients if p.get("warfarin_skin_necrosis")) / n, 1)

    # OCP-associated (females with OCP trigger)
    ocp_pct = round(100 * sum(1 for p in all_patients if p.get("ocp_associated")) / n, 1)

    # Pregnancy loss
    preg_loss_pct = round(100 * sum(1 for p in all_patients if p.get("pregnancy_loss")) / n, 1)

    # Heparin resistance (SERPINC1 only)
    heparin_resistance_pct = pct(serpc1, "heparin_resistance")

    # Cascade tested
    cascade_pct = round(100 * sum(1 for p in all_patients if p.get("cascade_tested")) / n, 1)

    # Homocysteine elevated
    homocys_pct = round(100 * sum(1 for p in all_patients if p.get("homocysteine_elevated")) / n, 1)

    # Folate supplemented
    folate_pct = pct(mthfr, "folate_supplemented")

    all_alerts = []
    for gd in THROMBOPHILIA_GENES:
        all_alerts.extend(gd["key_alerts"])

    agg = {
        "total_patients": n,
        "total_genes": 8,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_dx_delay_months": mean_delay,
        "pe_event_pct": pe_pct,
        "dvt_event_pct": dvt_pct,
        "anticoagulation_on_doac_pct": doac_pct,
        "warfarin_necrosis_pct": warfarin_necrosis_pct,
        "ocp_associated_pct": ocp_pct,
        "pregnancy_loss_pct": preg_loss_pct,
        "heparin_resistance_pct": heparin_resistance_pct,
        "cascade_tested_pct": cascade_pct,
        "homocysteine_elevated_pct": homocys_pct,
        "mthfr_folate_supplemented_pct": folate_pct,
        # F5
        "f5_ocp_associated_pct": pct(f5, "ocp_associated"),
        "f5_homozygous_leiden_pct": pct(f5, "homozygous_leiden"),
        "f5_apc_resistance_tested_pct": pct(f5, "apc_resistance_ratio_tested"),
        "f5_cascade_tested_pct": pct(f5, "cascade_tested"),
        "f5_doac_prescribed_pct": pct(f5, "doac_prescribed"),
        # F2
        "f2_ocp_associated_pct": pct(f2, "ocp_associated"),
        "f2_wes_missed_pct": pct(f2, "wes_missed_variant"),
        "f2_targeted_assay_pct": pct(f2, "targeted_f2_assay_done"),
        "f2_cascade_tested_pct": pct(f2, "cascade_tested"),
        # SERPINC1
        "serpinc1_heparin_resistance_pct": pct(serpc1, "heparin_resistance"),
        "serpinc1_at_concentrate_used_pct": pct(serpc1, "at_concentrate_used"),
        "serpinc1_functional_assay_pct": pct(serpc1, "functional_assay_performed"),
        "serpinc1_acquired_excluded_pct": pct(serpc1, "acquired_deficiency_excluded"),
        "serpinc1_type_i_pct": pct(serpc1, "type_i"),
        "serpinc1_type_ii_hbs_pct": pct(serpc1, "type_ii_hbs"),
        # PROC
        "proc_warfarin_necrosis_pct": pct(proc, "warfarin_skin_necrosis"),
        "proc_heparin_bridge_given_pct": pct(proc, "heparin_bridge_given"),
        "proc_neonatal_purpura_pct": pct(proc, "neonatal_purpura_fulminans"),
        "proc_chromogenic_assay_pct": pct(proc, "chromogenic_assay_done"),
        "proc_pregnancy_loss_pct": pct(proc, "pregnancy_loss"),
        # PROS1
        "pros1_ocp_confound_pct": pct(pros1, "ocp_associated"),
        "pros1_tested_off_ocp_pct": pct(pros1, "tested_off_ocp"),
        "pros1_free_ps_tested_pct": pct(pros1, "free_ps_tested"),
        "pros1_total_ps_tested_pct": pct(pros1, "total_ps_tested"),
        "pros1_mlpa_performed_pct": pct(pros1, "mlpa_performed"),
        "pros1_type_iii_pct": pct(pros1, "type_iii"),
        # MTHFR
        "mthfr_homocysteine_elevated_pct": pct(mthfr, "homocysteine_elevated"),
        "mthfr_folate_supplemented_pct": pct(mthfr, "folate_supplemented"),
        "mthfr_b12_checked_pct": pct(mthfr, "b12_checked"),
        "mthfr_b12_deficient_pct": pct(mthfr, "b12_deficient"),
        "mthfr_five_mthf_prescribed_pct": pct(mthfr, "five_mthf_prescribed"),
        "mthfr_nice_discussed_pct": pct(mthfr, "nice_guideline_discussed"),
        # THBD
        "thbd_ahus_overlap_pct": pct(thbd, "ahus_overlap"),
        "thbd_complement_workup_pct": pct(thbd, "complement_workup_done"),
        "thbd_eculizumab_eligible_pct": pct(thbd, "eculizumab_eligible"),
        "thbd_specialist_confirmed_pct": pct(thbd, "specialist_confirmed"),
        # SERPINE1
        "serpine1_four_g_four_g_pct": pct(serpe1, "four_g_four_g"),
        "serpine1_bleeding_phenotype_pct": pct(serpe1, "bleeding_phenotype"),
        "serpine1_pai1_activity_tested_pct": pct(serpe1, "pai1_activity_tested"),
        "serpine1_metabolic_syndrome_pct": pct(serpe1, "metabolic_syndrome"),
    }

    return {
        "title": "Hereditary-Thrombophilia-Atlas — Complete 8-Gene Hereditary Thrombophilia Reference",
        "subtitle": (
            "F5 · F2 · SERPINC1 · PROC · PROS1 · MTHFR · THBD · SERPINE1 — "
            "320 patients (8×40, seeds 1510–1517) — Factor V Leiden APC Resistance, "
            "Prothrombin G20210A 3'UTR, AT-III Heparin Resistance, Protein C Warfarin Necrosis, "
            "Protein S OCP Confound, MTHFR Homocysteine, Thrombomodulin aHUS, PAI-1 Fibrinolysis"
        ),
        "genes": gene_summaries,
        "aggregate_stats": agg,
        "top_alerts": all_alerts[:12],
    }


def get_breakdown():
    breakdown = []
    for gd in THROMBOPHILIA_GENES:
        pts = _make_cohort(gd)
        sex_dist = {"M": sum(1 for p in pts if p["sex"] == "M"),
                    "F": sum(1 for p in pts if p["sex"] == "F")}
        mean_onset = round(sum(p["age_onset_years"] for p in pts) / len(pts), 1)
        mean_delay = round(sum(p["dx_delay_months"] for p in pts) / len(pts), 1)
        etiol_counts = {}
        for p in pts:
            etiol_counts[p["etiology"]] = etiol_counts.get(p["etiology"], 0) + 1

        breakdown.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "mean_onset_years": mean_onset,
            "mean_dx_delay_months": mean_delay,
            "sex_distribution": sex_dist,
            "etiology_counts": etiol_counts,
            "key_alerts": gd["key_alerts"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
            "patients": pts,
        })
    return {"breakdown": breakdown}


def get_definitions():
    return {
        "atlas": "Hereditary-Thrombophilia-Atlas — Complete 8-Gene Hereditary Thrombophilia Reference",
        "genes": [gd["gene"] for gd in THROMBOPHILIA_GENES],
        "clinical_definitions": [
            {
                "term": "Factor V Leiden APC Resistance — OCP Multiplied Risk and Homozygous Management",
                "definition": (
                    "Factor V Leiden (p.Arg534Gln) is the most prevalent hereditary thrombophilia "
                    "in European populations, present in 3-8% of unselected individuals and 20-50% "
                    "of selected VTE patients. The Arg534 residue is the primary cleavage site for "
                    "activated Protein C (APC) within FVa; substitution with Gln renders FVa "
                    "approximately 10-fold more resistant to APC-mediated inactivation, prolonging "
                    "FVa lifespan on platelet and endothelial surfaces and amplifying thrombin "
                    "generation. Heterozygous FVL confers a 5-7x increased VTE risk; homozygous "
                    "FVL confers 50-80x risk — a threshold at which lifelong anticoagulation after "
                    "a first unprovoked event is universally recommended. The most clinically "
                    "dangerous interaction is FVL with estrogen-containing oral contraceptive pills: "
                    "estrogens upregulate FV, FVIII, and fibrinogen, and downregulate Protein S — "
                    "compounding the existing APC resistance. The combined risk rises to approximately "
                    "35x baseline VTE risk — an absolute contraindication. APC resistance is "
                    "measured by an APTT-based clotting ratio (ratio of APTT with and without added "
                    "APC): values <2.0 are abnormal. DNA confirmation is mandatory as rare 'Type II' "
                    "APC resistance exists without the Leiden variant (e.g., FVR2 haplotype, FV Hong "
                    "Kong). DOAC agents (rivaroxaban, apixaban, edoxaban) perform equivalently to "
                    "warfarin in FVL-associated VTE and are preferred for long-term management due "
                    "to superior safety profile and lack of INR monitoring requirement."
                ),
            },
            {
                "term": "Prothrombin G20210A — 3'UTR Mechanism and WES Detection Gap",
                "definition": (
                    "The G20210A variant is located 20 nucleotides downstream of the F2 stop codon "
                    "in the 3' untranslated region — not in the coding sequence. This position "
                    "overlaps with the polyadenylation signal hexanucleotide (AATAAA sequence), and "
                    "the A allele creates a slightly more efficient polyadenylation signal, increasing "
                    "3'-end processing efficiency and mRNA stability. The net effect is approximately "
                    "30% higher circulating prothrombin (Factor II) concentration in heterozygous "
                    "carriers, translating to approximately 30% more substrate available for "
                    "prothrombinase-mediated thrombin generation. The prothrombotic mechanism is "
                    "quantitative (more substrate) rather than qualitative (structural defect in "
                    "coagulation control) — explaining the lower per-allele VTE risk (2-5x) compared "
                    "to FVL (5-7x) or AT-III deficiency (10-50x). The WES detection gap is a critical "
                    "clinical problem: most exome capture kits target protein-coding exons and "
                    "immediately flanking splice regions; the F2 3'UTR, being non-coding, is often "
                    "outside the capture boundaries or excluded from variant calling pipelines. "
                    "A patient undergoing WES for recurrent VTE may receive a report stating 'no "
                    "F2 pathogenic variants detected' — which is technically accurate (no coding "
                    "variants) but fails to detect G20210A. Targeted F2 G20210A PCR assay, or "
                    "a thrombophilia-specific panel with explicit 3'UTR coverage, is required. "
                    "Double heterozygosity for FVL and G20210A approximately multiplies risks "
                    "(estimated 20x combined), making accurate detection of both variants essential."
                ),
            },
            {
                "term": "Antithrombin III Deficiency — Heparin Resistance and Perioperative AT-III Concentrate",
                "definition": (
                    "Antithrombin III deficiency has the highest per-allele VTE risk of all "
                    "hereditary thrombophilias: 10-50x baseline risk, with first events often "
                    "occurring in young adults without additional provocative factors. The clinical "
                    "urgency of Type II HBS (heparin-binding site) AT-III deficiency is unmatched "
                    "in thrombophilia medicine: because the HBS variant AT-III cannot bind heparin, "
                    "standard anticoagulation with UFH or LMWH is essentially ineffective. "
                    "The laboratory finding of heparin resistance (requiring >35,000 units UFH/day "
                    "to achieve therapeutic aPTT, or failure to achieve therapeutic anti-Xa on "
                    "standard LMWH dosing) in a patient with low AT-III activity should immediately "
                    "prompt Type II HBS subtyping and AT-III concentrate preparation. "
                    "Perioperative management requires pre-infusion of AT-III concentrate "
                    "(Thrombate III, plasma-derived; or ATryn, recombinant) to raise AT-III activity "
                    "above 80% before surgery, restoring UFH sensitivity for intraoperative "
                    "anticoagulation. Functional (chromogenic) AT-III assay is the only reliable "
                    "diagnostic test: antigenic AT-III assay detects Type I (low quantity) but "
                    "MISSES Type II (normal quantity, dysfunctional), leading to under-diagnosis. "
                    "Acquired AT-III deficiency (from heparin use, liver disease, sepsis/DIC, "
                    "nephrotic syndrome) must be systematically excluded by repeat testing in a "
                    "stable clinical state, off heparin, before a hereditary diagnosis is confirmed."
                ),
            },
            {
                "term": "Protein C Deficiency — Warfarin Skin Necrosis Mechanism and Heparin Bridge Mandatory",
                "definition": (
                    "Protein C deficiency creates one of the few absolute mandatory rules in "
                    "coagulation medicine: warfarin MUST be initiated with a heparin bridge. "
                    "The mechanism of warfarin skin necrosis illuminates why: warfarin blocks "
                    "vitamin K-dependent gamma-carboxylation of coagulation factors II, VII, IX, "
                    "X, Protein C, and Protein S. The critical difference is kinetics. Protein C "
                    "has a plasma half-life of approximately 6-8 hours — the shortest of all "
                    "vitamin K-dependent coagulation proteins. Factor X has a half-life of ~40 h "
                    "and prothrombin ~60 h. When warfarin is started without anticoagulant coverage, "
                    "Protein C levels fall rapidly in the first 24-48 hours while procoagulant "
                    "factors remain near-normal — creating a transient period of profound "
                    "anticoagulant deficiency superimposed on an already low-baseline PC state. "
                    "The resulting unopposed procoagulant drive causes thrombosis in dermal "
                    "venules (particularly at fatty tissue sites — breast, abdomen, thighs, "
                    "buttocks) — clinically presenting as painful skin erythema progressing to "
                    "haemorrhagic bullae and necrosis. Mandatory rule: LMWH or UFH overlap for "
                    "minimum 5 days when initiating warfarin, continued until INR is ≥2.0 on "
                    "two consecutive measurements. DOAC agents bypass this risk entirely and are "
                    "now preferred. Homozygous Protein C deficiency is a neonatal emergency "
                    "requiring immediate Protein C concentrate (Ceprotin) and long-term replacement."
                ),
            },
            {
                "term": "Protein S Deficiency — OCP Confound, Pregnancy Testing, and Type I/II/III Classification",
                "definition": (
                    "Protein S deficiency diagnosis requires navigating several confounds that "
                    "systematically elevate false-positive rates if not addressed. Estrogen-containing "
                    "OCP increases C4b-binding protein (C4BP), which sequesters Protein S, reducing "
                    "free PS to levels diagnostic of Type III deficiency in up to 50% of women on "
                    "OCP who have no PROS1 mutation. Testing must be delayed ≥3 months after "
                    "stopping OCP. Pregnancy reduces free PS to near-Type-III-deficiency levels "
                    "as a physiological adaptation — testing during pregnancy is invariably "
                    "unreliable and should never be used as evidence of hereditary deficiency. "
                    "Test ≥3 months postpartum. The three-type classification is clinically "
                    "important for family cascade testing and laboratory interpretation: "
                    "Type I — both total and free PS antigen low, and APC cofactor activity low; "
                    "most severe, usually caused by frameshift, splice-site, or large deletion; "
                    "Type II — normal total and free PS antigen, but APC cofactor activity "
                    "reduced (functionally defective PS); rare; only detected by functional assay; "
                    "Type III — free PS low but total PS normal or near-normal; the commonest "
                    "variant type; caused by missense variants affecting C4BP-binding equilibrium "
                    "or membrane interaction. Large deletions spanning PROS1 and the adjacent "
                    "pseudogene PROS2 account for a significant proportion of Type I PROS1 "
                    "alleles and require MLPA for detection — sequencing will appear normal. "
                    "Both free PS and total PS must be measured; a functional APC cofactor assay "
                    "is required to detect Type II. Warfarin reduces Protein S and warfarin "
                    "skin necrosis risk parallels that of Protein C deficiency."
                ),
            },
            {
                "term": "MTHFR C677T — Hyperhomocysteinaemia, Folate Therapy, and Guideline Controversy",
                "definition": (
                    "The MTHFR C677T polymorphism is the most prevalent hereditary metabolic "
                    "variant in the thrombophilia literature, yet also the most over-interpreted. "
                    "Homozygous TT genotype (5-15% of Europeans) reduces MTHFR enzyme activity "
                    "to approximately 30% of wild-type CC, impairing 5,10-MTHF reduction to "
                    "5-MTHF — the methyl donor for homocysteine remethylation to methionine. "
                    "The result under conditions of low folate or low B12 intake is hyperhomocysteinaemia "
                    "(elevated plasma total homocysteine >15 μmol/L), which damages endothelium, "
                    "promotes oxidative stress, activates platelets, and may impair fibrinolysis. "
                    "However, multiple large randomised controlled trials of homocysteine-lowering "
                    "therapy (B vitamins, folate) have FAILED to consistently reduce VTE recurrence "
                    "rates, casting serious doubt on homocysteine as an independent VTE mediator "
                    "rather than a biomarker. NICE (2012) and BCSH thrombophilia guidelines "
                    "explicitly recommend against MTHFR testing as a routine thrombophilia screen "
                    "because the clinical utility (change in management based on genotype) is "
                    "undemonstrated. The recommended alternative: measure plasma homocysteine "
                    "directly; if elevated (>15 μmol/L), supplement with 5-MTHF (methylfolate, "
                    "400-800 μg/day) and vitamin B12; recheck homocysteine in 3 months. "
                    "B12 and B6 deficiency significantly amplify hyperhomocysteinaemia in MTHFR "
                    "TT carriers and should always be checked and corrected. C677T/A1298C compound "
                    "heterozygotes may have greater homocysteine elevation than either alone."
                ),
            },
            {
                "term": "Thrombomodulin Deficiency — Protein C Activation Impairment and aHUS Overlap",
                "definition": (
                    "Thrombomodulin (TM, THBD) represents the molecular switch that converts "
                    "thrombin from its procoagulant to anticoagulant function on the endothelial "
                    "surface. TM binds thrombin at EGF domains 4-6 with high affinity, and the "
                    "resulting thrombin-TM complex: (1) is sterically blocked from cleaving "
                    "fibrinogen or activating PAR-1; (2) activates Protein C approximately "
                    "1000-fold more efficiently than free thrombin; and (3) activates TAFI "
                    "(thrombin-activatable fibrinolysis inhibitor) to regulate fibrinolysis. "
                    "THBD pathogenic variants (predominantly missense in the EGF-like domain) "
                    "are rare causes of hereditary thrombophilia, often not captured in standard "
                    "panels. Pathogenicity assessment is challenging because many THBD variants "
                    "of uncertain significance exist. Functional TM-mediated PC activation assay "
                    "(not routine plasma PC level) is required — THBD variants impair the "
                    "thrombin-TM complex kinetics rather than PC expression itself, so plasma PC "
                    "antigen can be normal. The aHUS (atypical haemolytic uraemic syndrome) "
                    "overlap is a critical recognition point: THBD mutations can cause "
                    "complement-mediated thrombotic microangiopathy (TMA) indistinguishable from "
                    "mutations in CFH, CFI, or C3. Any THBD patient presenting with TMA features "
                    "(microangiopathic haemolytic anaemia, thrombocytopenia, renal impairment) "
                    "requires full complement pathway workup and consideration of eculizumab "
                    "(anti-C5 inhibitor). Recombinant thrombomodulin (ART-123) has conditional "
                    "approval in Japan for DIC and is in clinical investigation for aHUS/TMA."
                ),
            },
            {
                "term": "PAI-1 (SERPINE1) — Fibrinolytic Defect vs Complete Deficiency Bleeding Phenotype",
                "definition": (
                    "PAI-1 (plasminogen activator inhibitor type 1) illustrates the spectrum of "
                    "phenotypic consequences possible from variants in a single coagulation gene: "
                    "elevated PAI-1 causes thrombosis, while absent PAI-1 causes severe bleeding. "
                    "The 4G/5G promoter polymorphism is the most clinically encountered variant: "
                    "4G/4G homozygous individuals have elevated plasma PAI-1 activity, impairing "
                    "tPA- and uPA-mediated fibrinolysis. Fibrin clots persist longer, increasing "
                    "VTE risk by a modest margin (estimated OR 1.5-2.0 in some studies). Plasma "
                    "PAI-1 activity also rises in metabolic syndrome, obesity, insulin resistance, "
                    "and cardiovascular disease — so 4G/4G thrombotic risk is substantially "
                    "amplified by metabolic comorbidities. However, the 4G/5G polymorphism is a "
                    "common genetic variant (not a rare pathogenic mutation) and its independent "
                    "VTE risk contribution in the absence of other thrombophilic or metabolic risk "
                    "factors is debated; it is NOT part of routine thrombophilia screening panels. "
                    "Complete PAI-1 deficiency, caused by rare biallelic frameshift or nonsense "
                    "variants, produces the precisely opposite phenotype: severe bleeding, delayed "
                    "wound healing, post-surgical haemorrhage, menorrhagia, haemarthrosis. "
                    "Without PAI-1 to regulate tPA, fibrinolysis is uncontrolled — clots dissolve "
                    "before achieving haemostasis. Treatment is antifibrinolytic: tranexamic acid "
                    "prevents plasminogen binding to lysine residues on fibrin, blocking fibrinolysis. "
                    "In massive PE requiring thrombolysis, the 4G/4G genotype may theoretically "
                    "require consideration of PAI-1-adjusted tPA dosing, though this is not yet "
                    "part of formal treatment guidelines."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(json.dumps(ov["aggregate_stats"], indent=2))
    print(f"\nTop alerts ({len(ov['top_alerts'])}):")
    for a in ov["top_alerts"]:
        print(f"  • {a}")
