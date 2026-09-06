#!/usr/bin/env python3
"""Hereditary-Angioedema-Atlas — Complete 8-Gene Hereditary Angioedema Atlas
SERPING1 (C1-inhibitor / C1-INH; 478 aa; 11q12.1; AD;
          HAE type I (low C1-INH) / type II (dysfunctional C1-INH) — most common ~85%;
          seed SEED_BASE+0) ·
F12      (Coagulation factor XII / Hageman factor; 596 aa; 5q35.3; AD GOF;
          HAE-FXII / HAE type III — estrogen-driven, GOF p.Thr309Lys/Arg;
          seed SEED_BASE+1) ·
PLG      (Plasminogen; 810 aa; 6q26; AD GOF;
          HAE-PLG — Lys330Glu plasminogen variant amplifies contact activation;
          seed SEED_BASE+2) ·
ANGPT1   (Angiopoietin-1; 498 aa; 8q23.1; AD;
          HAE-ANGPT1 — Tie2-mediated endothelial barrier LOF;
          seed SEED_BASE+3) ·
MYOF     (Myoferlin; 2057 aa; 10q24.11; AD;
          HAE-MYOF — membrane fusion/repair deficit → increased endothelial permeability;
          seed SEED_BASE+4) ·
KNG1     (High-molecular-weight kininogen; 644 aa; 3q27.3; AD;
          HAE-KNG1 — HMWK variant enhances bradykinin release;
          seed SEED_BASE+5) ·
HS3ST6   (Heparan sulfate glucosamine 3-O-sulfotransferase 6; 333 aa; 16p13.3; AD;
          HAE-HS3ST6 — contact-activation surface modification → excess FXIIa;
          seed SEED_BASE+6) ·
KLKB1    (Plasma kallikrein / prekallikrein; 638 aa; 4q35.2; AR;
          Prekallikrein deficiency-HAE — prolonged APTT; episodic angioedema;
          seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1614–1621)
"""

import random

SEED_BASE = 1614

HAE_GENES = [
    # ── SERPING1 — HAE type I / II ────────────────────────────────────────
    {
        "gene": "SERPING1",
        "protein": "SERPING1 — HAE Type I/II AD — C1-Inhibitor LOF — Bradykinin-Mediated — C4 Low Between Attacks PATHOGNOMONIC",
        "alias": (
            "SERPING1; OMIM gene 606860; Hereditary Angioedema type I OMIM 106100; "
            "Hereditary Angioedema type II OMIM 106100 (allelic); "
            "11q12.1; 478 aa; ~66 kDa (mature, heavily glycosylated); AD haploinsufficiency. "
            "SERPING1 encodes C1-inhibitor (C1-INH, C1-esterase inhibitor), a serine protease inhibitor "
            "(serpin) and the PRIMARY brake on the plasma contact activation system. "
            "TARGETS INHIBITED: C1r, C1s (classical complement), Factor XIIa (Hageman), "
            "Factor XIa (coagulation), plasma kallikrein, MBL-associated serine proteases (MASPs). "
            "LOF MECHANISM: haploinsufficiency (LOF mutations in one allele) → insufficient C1-INH → "
            "unchecked Factor XII activation → kallikrein activation → cleavage of HMWK (KNG1) → "
            "BRADYKININ generation → bradykinin B2 receptor activation on endothelium → "
            "vascular hyperpermeability → plasma leak into tissue → ANGIOEDEMA. "
            "HAE TYPE I (85% of SERPING1 cases): truncating/frameshift/deletion → LOW C1-INH level + LOW function. "
            "HAE TYPE II (15%): missense → NORMAL or HIGH C1-INH antigen but DYSFUNCTIONAL; "
            "Type II most confused with acquired C1-INH deficiency — functional assay MANDATORY. "
            "CLINICAL ATTACKS: non-pitting, non-pruritic subcutaneous edema (face, extremities, genitalia); "
            "ABDOMINAL ATTACKS (90%): colic, vomiting, diarrhea — mimics acute abdomen; "
            "unnecessary laparotomies historically performed; "
            "LARYNGEAL ATTACKS (40%): life-threatening airway obstruction → death if untreated; "
            "laryngeal HAE is the leading cause of death in untreated families (25–30% mortality historically). "
            "LABORATORY: C4 LOW between attacks (consumption by unregulated C1r/C1s) — "
            "C4 is the single best SCREENING test (sensitivity ~95% between attacks); "
            "C3 NORMAL (C3 convertase not excessively activated); "
            "C1-INH antigen LOW (type I) or NORMAL/HIGH (type II); "
            "C1-INH function REDUCED in both types (functional assay mandatory); "
            "C1q: NORMAL in hereditary HAE (LOW in acquired C1-INH deficiency — key DDx). "
            "TRIGGERS: oestrogens (OCP, HRT, pregnancy), ACE inhibitors (block bradykinin degradation — "
            "ABSOLUTELY CONTRAINDICATED in HAE), physical trauma, surgical/dental procedures, "
            "emotional stress, infections. "
            "DIAGNOSIS LADDER: (1) Clinical suspicion (recurrent non-pruritic, non-pitting edema); "
            "(2) C4 level (if low, continue); (3) C1-INH antigen; (4) C1-INH function; (5) Genetic testing. "
            "ACUTE TREATMENT (on-demand): "
            "C1-INH concentrate IV (Berinert 20 U/kg; Cinryze; Haegarda SC) — plasma-derived; "
            "Icatibant SC (Firazyr) — selective bradykinin B2 receptor antagonist — onset <1 hour; "
            "Ecallantide SC (Kalbitor) — plasma kallikrein inhibitor — physician-administered; "
            "FFPP (fresh-frozen plasma) — if no specific therapy available (risk viral transmission). "
            "DO NOT give antihistamines or corticosteroids for acute HAE — they do not work "
            "(bradykinin-mediated, NOT histamine-mediated). "
            "PROPHYLAXIS: "
            "Long-term: Lanadelumab SC every 2 weeks (Takhzyro) — anti-kallikrein monoclonal Ab; "
            "Berotralstat oral daily (Orladeyo) — kallikrein inhibitor; "
            "C1-INH SC (Haegarda 60 U/kg twice weekly); "
            "Short-term (pre-procedure): C1-INH IV 1000 U 1–6 hours before procedure; "
            "Tranexamic acid — antifibrinolytic, reduces attack frequency (third-line, less effective)."
        ),
        "aa": "478 aa",
        "kDa": "~66 kDa",
        "locus": "11q12.1",
        "omim_gene": 606860,
        "omim_disease": 106100,
        "inheritance": "AD haploinsufficiency; >700 known pathogenic variants; de novo in 25%; penetrance ~100% but variable expressivity",
        "gene_class": (
            "SERPING1 encodes C1 esterase inhibitor (C1-INH), a member of the serpin superfamily. "
            "DOMAIN ARCHITECTURE: N-terminal non-serpin domain (heavily glycosylated, ~40% carbohydrate) + "
            "C-terminal serpin domain containing the reactive centre loop (RCL). "
            "MECHANISM OF INHIBITION: C1-INH acts as a 'suicide substrate' — target protease cleaves "
            "the P1-P1' bond in the RCL → rapid conformational change (stressed → relaxed) → "
            "covalent acyl-enzyme complex → protease permanently inactivated. "
            "KEY INHIBITION: C1r + C1s (complement initiation) → prevents classical pathway overactivation; "
            "Factor XIIa + XIIf → prevents contact activation amplification; "
            "Plasma kallikrein → prevents HMWK cleavage to bradykinin; "
            "Factor XIa → mild anticoagulant effect. "
            "SERPING1 also inhibits tPA and uPA (fibrinolysis) and MASP1/2 (lectin complement). "
            ">700 pathogenic variants span the gene; large deletions (including Alu-mediated recombinations) "
            "account for 15–20% of type I alleles; functional assays (chromogenic or ELISA) are mandatory "
            "because antigen level can be normal in type II."
        ),
        "n_patients": 40,
        "key_alerts": [
            "SERPING1-C4-SCREENING-LOW-BETWEEN-ATTACKS: C4 is consumed by unregulated C1r/C1s activation BETWEEN attacks (not just during); a single C4 level below normal has ~95% sensitivity for SERPING1-HAE; always measure C4 first when HAE is suspected — a normal C4 makes SERPING1-HAE very unlikely; note C4 may transiently normalise in convalescence",
            "SERPING1-ACE-INHIBITOR-ABSOLUTELY-CONTRAINDICATED: ACE inhibitors (ramipril, lisinopril, enalapril, perindopril) degrade bradykinin; in HAE patients, ACE inhibitors remove the final brake on bradykinin accumulation → can precipitate life-threatening laryngeal attacks; ACE inhibitors are ABSOLUTELY CONTRAINDICATED in all HAE patients; use ARBs as alternative antihypertensive if needed",
            "SERPING1-LARYNGEAL-ATTACK-EMERGENCY: Laryngeal HAE = immediate life threat; mortality 25-30% in untreated historical series; every HAE patient must carry home emergency medication (icatibant autoinjector or SC C1-INH) and a medical alert card; teach self-administration; if laryngeal symptoms → immediate home treatment + emergency services simultaneously; do NOT wait to see if it resolves",
            "SERPING1-ABDOMINAL-ATTACKS-NOT-SURGICAL: Abdominal HAE attacks (pain, vomiting, diarrhea) mimic acute abdomen so convincingly that unnecessary laparotomies historically occurred; C4 and HAE history before surgery; abdominal attacks self-resolve in 48-72 hours; opiates worsen attacks (stress response); treat with specific HAE therapy not surgical exploration",
            "SERPING1-TYPE-II-FUNCTIONAL-ASSAY-MANDATORY: HAE type II has NORMAL or HIGH C1-INH antigen on nephelometry/ELISA but DYSFUNCTIONAL protein; relying on antigen alone MISSES type II entirely; ALWAYS request C1-INH FUNCTIONAL assay (chromogenic or clot-based); type II accounts for 15% of SERPING1-HAE and is misdiagnosed as 'histaminergic' until a functional assay is done",
            "SERPING1-PREGNANCY-SPECIALIST-MANAGEMENT: C1-INH levels may rise in pregnancy (oestrogen effect on protein production) paradoxically; however attacks can still occur especially peri-partum; safe acute treatments in pregnancy: C1-INH concentrate (IV/SC); icatibant (data limited but used); ecallantide CONTRAINDICATED (insufficient data); prophylaxis: C1-INH SC preferred; tranexamic acid acceptable; lanadelumab safety data limited",
            "SERPING1-ANTIHISTAMINES-INEFFECTIVE: HAE attacks are bradykinin-mediated NOT histamine-mediated; antihistamines (cetirizine, loratadine, fexofenadine) and corticosteroids are COMPLETELY INEFFECTIVE for acute HAE; this is the most common error when HAE is misdiagnosed as allergic angioedema; rapid recognition of non-pruritic, non-urticarial edema should prompt HAE pathway not allergy pathway",
            "SERPING1-FAMILY-CASCADE-TESTING: SERPING1-HAE is AD; 50% risk to first-degree relatives; de novo mutations occur in ~25% so family history may be negative; test ALL first-degree relatives with C4 + C1-INH antigen + C1-INH function; genetic testing confirms the pathogenic variant; children should be tested at birth or early childhood before first potential attack; negative family history does NOT rule out HAE",
        ],
        "etiologies": {
            "Frameshift/nonsense — truncation, type I HAE (C1-INH absent)": 14,
            "Missense in serpin domain — dysfunctional C1-INH, type II HAE": 8,
            "Splice-site variant — exon skipping, type I or II HAE": 7,
            "Large deletion (Alu-mediated recombination), type I HAE": 6,
            "De novo mutation (no family history)": 4,
            "Promoter variant — reduced expression, type I HAE": 1,
        },
        "stats": {
            "mean_dx_age_y": 25.3,
            "mean_dx_delay_months": 96.0,
            "pct_abdominal_attacks": 90,
            "pct_laryngeal_attacks": 40,
            "pct_c4_low_between_attacks": 95,
            "pct_misdiagnosed_allergy": 70,
            "pct_unnecessary_surgery": 28,
            "pct_ocp_triggered": 65,
            "pct_ace_inhibitor_history": 18,
        },
        "dx_delay_distribution": {"<1 y": 4, "1–5 y": 9, "5–15 y": 17, ">15 y": 10},
    },
    # ── F12 — HAE-FXII / HAE type III ────────────────────────────────────
    {
        "gene": "F12",
        "protein": "F12 — HAE-FXII / HAE Type III AD GOF — Estrogen-Driven — C1-INH Normal — OCP/Pregnancy Trigger",
        "alias": (
            "F12; OMIM gene 234000; Hereditary Angioedema with normal C1-INH (HAE-FXII) OMIM 610618; "
            "5q35.3; 596 aa; ~80 kDa (single chain zymogen); AD gain-of-function. "
            "F12 encodes coagulation Factor XII (Hageman factor), a multi-domain serine protease "
            "and the initiator of the contact activation (kallikrein-kinin) system. "
            "GOF VARIANTS: p.Thr309Lys (c.926C>A) and p.Thr309Arg (c.926C>G) — most frequent; "
            "also p.Glu173Lys (c.517G>A) — located in fibronectin type II domain. "
            "Thr309 is a key glycosylation site in the proline-rich region; its loss removes an "
            "N-linked glycan that normally sterically hinders auto-activation; "
            "GOF variants → spontaneous FXII autoactivation → excess kallikrein → excess bradykinin. "
            "PHENOTYPE: clinically identical to SERPING1-HAE — recurrent non-pitting, non-pruritic "
            "angioedema of skin, abdomen, larynx; ESTROGEN-DEPENDENT — attacks cluster around: "
            "OCP initiation, first/second trimester of pregnancy, HRT; "
            "PREDOMINANTLY AFFECTS FEMALES (males with F12 GOF variant may be asymptomatic carriers); "
            "NORMAL C1-INH antigen and function; NORMAL C4 (key distinguishing feature from SERPING1-HAE). "
            "LABORATORY: C4 NORMAL; C1-INH antigen NORMAL; C1-INH function NORMAL — "
            "diagnosis is CLINICAL + GENETIC; no current validated biomarker; "
            "APTT: prolonged (Factor XII procoagulant role) but no bleeding tendency (FXII deficiency → "
            "prolonged APTT without bleeding — contact pathway activates coagulation in vitro not in vivo). "
            "TRIGGERS: oestrogens (OCP, HRT, tamoxifen), pregnancy, psychological stress; "
            "male carriers may be entirely asymptomatic or rarely symptomatic. "
            "TREATMENT: same as SERPING1-HAE in principle — "
            "Acute: icatibant (bradykinin B2 receptor antagonist — FIRST CHOICE, works regardless of C1-INH level); "
            "C1-INH concentrate (works if attacks are kallikrein-driven, generally effective); "
            "Ecallantide (kallikrein inhibitor — effective). "
            "PROPHYLAXIS: Lanadelumab (anti-kallikrein monoclonal); Berotralstat; "
            "Tranexamic acid (antifibrinolytic — reduces plasmin-mediated FXII activation); "
            "AVOID OESTROGENS — OCP is absolutely contraindicated; "
            "use progestogen-only contraception or non-hormonal methods; "
            "if pregnancy: plan pre-partum with HAE specialist; delivery with icatibant available."
        ),
        "aa": "596 aa",
        "kDa": "~80 kDa",
        "locus": "5q35.3",
        "omim_gene": 234000,
        "omim_disease": 610618,
        "inheritance": "AD gain-of-function; 3 major variants (Thr309Lys, Thr309Arg, Glu173Lys); predominantly females symptomatic",
        "gene_class": (
            "F12 encodes Factor XII (Hageman factor), a multi-domain serine protease zymogen. "
            "DOMAIN ARCHITECTURE (N→C): fibronectin type II → EGF-like → fibronectin type I → "
            "EGF-like-2 → kringle → proline-rich region (contains Thr309 glycosylation site) → "
            "serine protease catalytic domain. "
            "ACTIVATION: surface contact (negative charge — glass, collagen, polyphosphates, nucleic acids) → "
            "conformational change → auto-activation (reciprocal activation with prekallikrein). "
            "CONTACT SYSTEM: FXIIa cleaves prekallikrein (KLKB1) → kallikrein; "
            "kallikrein cleaves HMWK (KNG1) → bradykinin (9aa peptide) + kinin-free kininogen. "
            "GOF variants remove glycosylation site → loss of steric protection → "
            "enhanced FXII autoactivation even WITHOUT surface contact → "
            "amplified contact system activity → bradykinin excess. "
            "NOTE: FXII deficiency (LOF, AR) causes prolonged APTT but NO bleeding "
            "(contact activation is dispensable for in vivo haemostasis) — the inverse phenotype "
            "from the GOF-HAE described here."
        ),
        "n_patients": 40,
        "key_alerts": [
            "F12-HAE-NORMAL-C4-C1INH: In HAE-FXII, C4 is NORMAL and C1-INH antigen/function are NORMAL — this is a C1-INH-independent HAE; diagnosis is clinical (recurrent non-pruritic angioedema) + genetic (F12 GOF variant); do NOT dismiss HAE diagnosis because complement levels are normal; always genotype suspected HAE with normal complement studies",
            "F12-OCP-ABSOLUTELY-CONTRAINDICATED: Oestrogen-containing oral contraceptives are the most common trigger for first and recurrent attacks in F12-HAE; oestrogen upregulates FXII expression and lowers the threshold for contact activation; OCP is ABSOLUTELY CONTRAINDICATED; prescribe progestogen-only pills or non-hormonal contraception; inform gynaecologist of diagnosis before any hormonal prescription",
            "F12-PREDOMINANTLY-FEMALES: Males carrying F12 GOF variants are often entirely asymptomatic or have rare mild attacks; virtually all symptomatic index cases are female (oestrogen dependence); when a female is diagnosed with F12-HAE, test male relatives — even if they are asymptomatic carriers they must avoid prescribing OCP to their daughters",
            "F12-APTT-PROLONGED-NOT-BLEEDING: F12 GOF variants can cause a modestly prolonged APTT in vitro (increased activation shortens APTT by consuming substrate) — OR a prolonged APTT if FXII levels are altered; FXII deficiency causes prolonged APTT without bleeding; do not treat prolonged APTT in HAE-FXII context with fresh-frozen plasma for clotting correction — this is not a bleeding disorder",
            "F12-PREGNANCY-FIRST-TRIMESTER-PEAK: Attacks in F12-HAE cluster in first/second trimester when oestrogen rises rapidly; register with HAE specialist in early pregnancy; icatibant is used off-label in acute attacks during pregnancy (limited but reassuring data); C1-INH concentrate is safe; discuss planned delivery with available acute medication",
            "F12-ICATIBANT-FIRST-CHOICE-ACUTE: For acute attacks in F12-HAE, icatibant (bradykinin B2 receptor antagonist, 30 mg SC) is highly effective — it blocks bradykinin action regardless of C1-INH levels; C1-INH concentrate also works (by reducing kallikrein activity); antihistamines and corticosteroids are ineffective",
            "F12-GENETIC-TESTING-THREE-VARIANTS: 90% of F12-HAE is caused by three variants: p.Thr309Lys, p.Thr309Arg, p.Glu173Lys; targeted variant testing covers most cases; full F12 sequencing for atypical presentations; genetic testing is the gold standard for F12-HAE diagnosis when clinical picture fits but complement labs are normal",
        ],
        "etiologies": {
            "p.Thr309Lys (c.926C>A) GOF — loss of glycosylation, most common": 22,
            "p.Thr309Arg (c.926C>G) GOF — loss of glycosylation, second most common": 11,
            "p.Glu173Lys (c.517G>A) GOF — fibronectin type II domain": 5,
            "Other rare F12 GOF missense": 2,
        },
        "stats": {
            "mean_dx_age_y": 31.8,
            "mean_dx_delay_months": 84.0,
            "pct_female_symptomatic": 92,
            "pct_ocp_triggered": 85,
            "pct_pregnancy_triggered": 72,
            "pct_c4_normal": 100,
            "pct_c1inh_normal": 100,
            "pct_misdiagnosed_allergy": 68,
        },
        "dx_delay_distribution": {"<1 y": 3, "1–5 y": 8, "5–15 y": 18, ">15 y": 11},
    },
    # ── PLG — HAE-PLG ─────────────────────────────────────────────────────
    {
        "gene": "PLG",
        "protein": "PLG — HAE-PLG AD GOF — Plasminogen Lys330Glu — Amplifies Contact Activation — Estrogen-Sensitive",
        "alias": (
            "PLG; OMIM gene 173350; HAE-PLG / Hereditary Angioedema with Plasminogen mutation OMIM 619366; "
            "6q26; 810 aa; ~92 kDa (mature plasminogen with signal peptide cleaved); AD gain-of-function. "
            "PLG encodes plasminogen, the zymogen precursor of plasmin, primarily synthesised in the liver. "
            "CANONICAL FUNCTION: tissue-type plasminogen activator (tPA) and urokinase (uPA) convert "
            "plasminogen to plasmin → fibrin degradation (fibrinolysis). "
            "HAE-LINK: plasmin also ACTIVATES single-chain Factor XII (sc-FXII → αFXIIa) in a "
            "positive feedback loop; GOF PLG variant p.Lys330Glu (c.988A>G) creates a neo-plasminogen "
            "with enhanced affinity for lysine-binding and enhanced activation → amplified FXII activation "
            "→ excess kallikrein → excess bradykinin → angioedema. "
            "p.Lys330Glu: Lys330 is located in kringle 4 of plasminogen; Glu330 disrupts the lysine-binding "
            "capacity of kringle 4, paradoxically enhancing activation kinetics in the contact pathway context. "
            "PHENOTYPE: recurrent non-pitting, non-pruritic subcutaneous/abdominal/laryngeal angioedema; "
            "ESTROGEN-SENSITIVE (OCP and pregnancy worsen — similar pattern to F12-HAE); "
            "NORMAL C4, C1-INH antigen, C1-INH function — complement normal (HAE-PLG is non-C1-INH-dependent); "
            "predominantly but not exclusively females. "
            "DIAGNOSIS: clinical HAE + normal complement + negative F12 GOF screening → PLG genetic testing. "
            "MANAGEMENT: icatibant (bradykinin B2 antagonist — acute first choice); "
            "C1-INH concentrate (acute second choice); "
            "tranexamic acid (antifibrinolytic — specifically targets PLG activation, rational for HAE-PLG; "
            "reduces attack frequency); lanadelumab prophylaxis (off-label but case series show efficacy); "
            "avoid oestrogens."
        ),
        "aa": "810 aa",
        "kDa": "~92 kDa",
        "locus": "6q26",
        "omim_gene": 173350,
        "omim_disease": 619366,
        "inheritance": "AD gain-of-function; p.Lys330Glu (c.988A>G) — single founder-like variant; most families European origin",
        "gene_class": (
            "PLG encodes plasminogen, a multi-domain serine protease zymogen. "
            "DOMAIN ARCHITECTURE: Pan-apple domain → activation peptide → 5 kringle domains "
            "(KR1–KR5, lysine-binding critical for fibrin targeting) → serine protease domain. "
            "Plasminogen is abundant in plasma (~200 µg/mL); activation by tPA/uPA cleaves Arg561-Val562 → plasmin. "
            "HAE-PLG MECHANISM: "
            "HAE-causing p.Lys330Glu lies in kringle 4 — a domain that normally binds lysine residues "
            "on fibrin and cell surfaces. The Glu substitution at Lys330 alters the lysine-binding "
            "pocket conformation, enhancing plasmin activity in the context of contact activation. "
            "Plasmin-mediated FXII activation creates a positive feedback: "
            "FXII → kallikrein → bradykinin (primary HAE pathway) + "
            "plasmin → FXII activation (secondary amplification). "
            "Tranexamic acid (a lysine analogue) blocks plasminogen-fibrin binding sites, "
            "reducing plasmin formation specifically from Glu330 plasminogen → rationale for "
            "tranexamic acid being particularly useful in HAE-PLG prophylaxis."
        ),
        "n_patients": 40,
        "key_alerts": [
            "PLG-SINGLE-VARIANT-p.Lys330Glu: Virtually all HAE-PLG cases carry the same variant p.Lys330Glu (c.988A>G); targeted single-variant testing is sufficient for diagnostic screening; full PLG sequencing rarely needed; this founder effect makes population screening feasible in high-risk families",
            "PLG-TRANEXAMIC-ACID-RATIONAL-CHOICE: Tranexamic acid (antifibrinolytic lysine analogue) is mechanistically targeted in HAE-PLG — it blocks plasminogen binding to lysine sites, reducing plasmin activation of FXII; use tranexamic acid for short-term prophylaxis (1-1.5 g oral TID) and as acute adjunct; more effective in PLG-HAE than in SERPING1-HAE",
            "PLG-C4-NORMAL-DIAGNOSIS-PATHWAY: HAE-PLG has NORMAL C4, C1-INH antigen, and C1-INH function; the diagnostic pathway for suspected HAE with normal complement is: (1) F12 GOF variants; (2) PLG p.Lys330Glu; (3) ANGPT1; (4) MYOF; (5) KNG1; (6) HS3ST6; genetic panel covering all HAE-nC1INH genes is the efficient approach",
            "PLG-ESTROGEN-DEPENDENCE: Like F12-HAE, PLG-HAE attacks are triggered and worsened by oestrogen; avoid oestrogen-containing contraceptives; pregnancy requires specialist monitoring; attacks increase with rising oestrogen in first trimester; progesterone-only or non-hormonal contraception is the safe alternative",
        ],
        "etiologies": {
            "p.Lys330Glu (c.988A>G) — kringle 4 GOF, near all cases": 38,
            "Other PLG variant (rare, role uncertain)": 2,
        },
        "stats": {
            "mean_dx_age_y": 34.2,
            "mean_dx_delay_months": 78.0,
            "pct_female_symptomatic": 88,
            "pct_ocp_triggered": 80,
            "pct_c4_normal": 100,
            "pct_tranexamic_effective": 72,
            "pct_misdiagnosed_allergy": 60,
        },
        "dx_delay_distribution": {"<1 y": 2, "1–5 y": 9, "5–15 y": 20, ">15 y": 9},
    },
    # ── ANGPT1 — HAE-ANGPT1 ──────────────────────────────────────────────
    {
        "gene": "ANGPT1",
        "protein": "ANGPT1 — HAE-ANGPT1 AD — Angiopoietin-1 LOF — Tie2 Endothelial Barrier Loss — Non-Bradykinin Component",
        "alias": (
            "ANGPT1; OMIM gene 601667; HAE-ANGPT1 OMIM 619387; "
            "8q23.1; 498 aa; ~57 kDa (monomer; functional homotrimer/tetramer); AD haploinsufficiency. "
            "ANGPT1 encodes Angiopoietin-1, a secreted glycoprotein ligand for the endothelial receptor "
            "tyrosine kinase Tie2 (TEK). "
            "CANONICAL FUNCTION: Angiopoietin-1/Tie2 signalling is the primary maintenance signal "
            "for endothelial barrier integrity and vessel quiescence: "
            "Angiopoietin-1 (agonist) → Tie2 clustering → PI3K/Akt signalling → cortical actin organisation "
            "→ tight junctions maintained → low permeability, anti-inflammatory endothelium. "
            "Angiopoietin-2 (contextual antagonist/agonist) competes with Ang-1 at Tie2. "
            "HAE MECHANISM: ANGPT1 LOF variants → haploinsufficiency → reduced Tie2 signalling → "
            "endothelial barrier loosened → enhanced vascular permeability → angioedema; "
            "bradykinin may ALSO play a role (some patients respond to icatibant) but the primary "
            "driver is Tie2-pathway insufficiency rather than pure kallikrein pathway. "
            "PHENOTYPE: recurrent non-histaminergic angioedema — subcutaneous (often facial/periorbital), "
            "abdominal, laryngeal; NORMAL C4, C1-INH antigen/function; NORMAL F12 (no GOF variants); "
            "clinically distinguishable only by genetic testing. "
            "MANAGEMENT: limited data; icatibant is used for acute attacks (partial response in some); "
            "C1-INH concentrate may help; theoretically plasma-derived C1-INH works by limiting "
            "kallikrein-mediated permeability even if primary driver differs; "
            "no specific Ang-1/Tie2-targeting therapy in clinical use for HAE-ANGPT1 yet."
        ),
        "aa": "498 aa",
        "kDa": "~57 kDa",
        "locus": "8q23.1",
        "omim_gene": 601667,
        "omim_disease": 619387,
        "inheritance": "AD haploinsufficiency; rare — fewer than 10 families described at most centres; multiple loss-of-function variants",
        "gene_class": (
            "ANGPT1 encodes Angiopoietin-1, a member of the angiopoietin family (Ang-1/2/3/4). "
            "DOMAIN STRUCTURE: N-terminal signal peptide → coiled-coil domain (oligomerisation) → "
            "linker → C-terminal fibrinogen-like domain (Tie2 binding). "
            "Ang-1 forms dimers and higher-order oligomers via the coiled-coil domain; "
            "tetramer/cluster form is the most potent Tie2 activator. "
            "TIE2 SIGNALLING: Ang-1 → Tie2 dimerisation → autophosphorylation → "
            "PI3K → Akt → FOXO1 nuclear exclusion → survival genes, eNOS activation; "
            "Rac1/Rap1 → cortical actin → endothelial cell-cell junctions (VE-cadherin stability). "
            "HAE-ANGPT1 LOF: insufficient Ang-1 → Tie2 baseline signalling reduced → "
            "endothelial barrier constitutively loosened → attacks triggered by minor stimuli."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ANGPT1-RARE-GENETIC-PANEL-NEEDED: HAE-ANGPT1 is one of the rarest HAE subtypes; it is ONLY diagnosable by genetic testing; complement studies and F12/PLG testing will all be normal; a 'HAE-nC1INH' gene panel covering SERPING1, F12, PLG, ANGPT1, MYOF, KNG1, HS3ST6 is required for complete workup of HAE with normal complement",
            "ANGPT1-TIE2-PATHWAY-DISTINCT: The endothelial barrier dysfunction in HAE-ANGPT1 is Tie2-pathway-mediated rather than purely kallikrein-bradykinin; this means standard HAE biomarkers (C4, C1-INH) are useless for monitoring; icatibant and C1-INH have variable efficacy; future targeted Tie2-pathway therapies may be specifically needed",
            "ANGPT1-ACUTE-TREATMENT-ICATIBANT: Despite non-bradykinin primary mechanism, icatibant (bradykinin B2 antagonist) is tried first for acute attacks as bradykinin amplification contributes secondarily; C1-INH concentrate is second option; fresh frozen plasma if neither available",
        ],
        "etiologies": {
            "Missense in fibrinogen-like Tie2-binding domain — LOF": 18,
            "Frameshift/nonsense — truncation, haploinsufficiency": 12,
            "Splice-site — exon skipping, reduced protein": 7,
            "Large deletion (gene-level)": 3,
        },
        "stats": {
            "mean_dx_age_y": 28.7,
            "mean_dx_delay_months": 102.0,
            "pct_female_predominance": 65,
            "pct_c4_normal": 100,
            "pct_c1inh_normal": 100,
            "pct_icatibant_partial_response": 60,
        },
        "dx_delay_distribution": {"<1 y": 2, "1–5 y": 7, "5–15 y": 19, ">15 y": 12},
    },
    # ── MYOF — HAE-MYOF ──────────────────────────────────────────────────
    {
        "gene": "MYOF",
        "protein": "MYOF — HAE-MYOF AD — Myoferlin LOF — Membrane Fusion/Repair Deficit — Endothelial Permeability",
        "alias": (
            "MYOF; OMIM gene 604603; HAE-MYOF OMIM 619388; "
            "10q24.11; 2057 aa; ~237 kDa; AD haploinsufficiency. "
            "MYOF encodes Myoferlin, a member of the ferlin family of C2 domain-containing proteins "
            "involved in membrane fusion, membrane repair, and vesicle trafficking. "
            "EXPRESSION: highly expressed in endothelial cells, muscle, and cardiac tissue. "
            "ENDOTHELIAL FUNCTION: Myoferlin mediates: "
            "(1) membrane repair after injury (Ca²⁺-dependent membrane resealing); "
            "(2) vesicle fusion and exocytosis (Weibel-Palade body secretion of VWF/P-selectin); "
            "(3) endothelial cell-cell junction maintenance; "
            "(4) VEGFR2 internalisation and signalling modulation. "
            "HAE MECHANISM: MYOF LOF → impaired membrane repair and vesicle fusion → "
            "increased baseline endothelial permeability → bradykinin-amplified leak during triggers. "
            "PHENOTYPE: recurrent non-histaminergic angioedema; NORMAL complement (C4, C1-INH); "
            "NORMAL F12, PLG, KNG1 — diagnosis requires genetic panel. "
            "MANAGEMENT: same pragmatic approach as other HAE-nC1INH subtypes — "
            "acute: icatibant or C1-INH concentrate; prophylaxis: tranexamic acid, lanadelumab (off-label); "
            "disease-specific MYOF-targeting therapies are not yet available."
        ),
        "aa": "2057 aa",
        "kDa": "~237 kDa",
        "locus": "10q24.11",
        "omim_gene": 604603,
        "omim_disease": 619388,
        "inheritance": "AD haploinsufficiency; very rare — only a few dozen cases worldwide; LOF variants across gene",
        "gene_class": (
            "MYOF encodes Myoferlin, a 2057 aa ferlin family member. "
            "DOMAIN ARCHITECTURE: 6 C2 domains (C2A–C2F) — Ca²⁺-dependent phospholipid-binding; "
            "FERM domain — membrane anchoring; DysF domains — protein-protein interactions; "
            "C-terminal transmembrane domain — membrane anchoring. "
            "FERLINS: Myoferlin, Dysferlin (DYSF, limb-girdle muscular dystrophy 2B), "
            "Otoferlin (OTOF, non-syndromic hearing loss) share ferlin architecture. "
            "MYOF IN ENDOTHELIUM: Weibel-Palade body (WPB) exocytosis of VWF requires Myoferlin; "
            "VEGFR2 recycling to cell surface for angiogenic signalling requires Myoferlin; "
            "membrane repair at sites of injury requires Ca²⁺-sensing via C2 domains and Myoferlin-mediated "
            "membrane patch fusion. LOF: endothelial fragility, impaired repair → permeability attacks."
        ),
        "n_patients": 40,
        "key_alerts": [
            "MYOF-LARGEST-HAE-GENE: MYOF encodes a 2057 aa protein — gene sequencing panels must include this large gene fully (all 56 exons); MLPA or dosage analysis needed for large deletions; next-generation sequencing panels covering MYOF are essential for complete HAE-nC1INH workup",
            "MYOF-WEIBEL-PALADE-BODY-LINK: Myoferlin participates in Weibel-Palade body (WPB) exocytosis; WPBs release VWF, P-selectin, and angiopoietin-2 on endothelial activation; MYOF LOF may paradoxically increase vascular permeability by impairing the regulated release of endothelial barrier-protecting factors",
            "MYOF-DYSFERLIN-DIFFERENTIAL: Myoferlin is structurally homologous to Dysferlin (DYSF) — loss of DYSF causes Limb-Girdle Muscular Dystrophy 2B; MYOF LOF does NOT cause muscular dystrophy (different expression pattern); do not confuse phenotypes when interpreting ferlin gene panel results",
        ],
        "etiologies": {
            "Missense in C2 domain — Ca²⁺-binding disrupted, LOF": 16,
            "Frameshift/nonsense — truncation, haploinsufficiency": 13,
            "Splice-site — exon skipping": 7,
            "Large intragenic deletion (MLPA)": 4,
        },
        "stats": {
            "mean_dx_age_y": 27.4,
            "mean_dx_delay_months": 108.0,
            "pct_c4_normal": 100,
            "pct_c1inh_normal": 100,
            "pct_icatibant_response": 58,
        },
        "dx_delay_distribution": {"<1 y": 1, "1–5 y": 6, "5–15 y": 20, ">15 y": 13},
    },
    # ── KNG1 — HAE-KNG1 ──────────────────────────────────────────────────
    {
        "gene": "KNG1",
        "protein": "KNG1 — HAE-KNG1 AD — High-MW Kininogen LOF/GOF — Bradykinin Precursor — Contact System",
        "alias": (
            "KNG1; OMIM gene 612358; HAE-KNG1 OMIM 619360; "
            "3q27.3; 644 aa (high-molecular-weight kininogen HMWK isoform); ~120 kDa (HMWK, glycosylated); "
            "AD. "
            "KNG1 encodes kininogen-1 by alternative splicing generating two functional isoforms: "
            "High-molecular-weight kininogen (HMWK/HK) — 644 aa — cofactor in contact activation; "
            "Low-molecular-weight kininogen (LMWK) — shared N-terminal domains, different C-terminal. "
            "HMWK IN CONTACT SYSTEM: HMWK forms a trimolecular complex with prekallikrein (KLKB1) and "
            "Factor XI on activated surfaces (polyphosphates, collagen, misfolded proteins); "
            "FXIIa cleaves HMWK between His-362 and Thr-363, releasing bradykinin (domain D4, residues 362–371). "
            "BRADYKININ RELEASE: cleavage of HMWK liberates bradykinin (Arg-Pro-Pro-Gly-Phe-Ser-Pro-Phe-Arg, 9aa); "
            "bradykinin acts on endothelial B2 receptors → NO + PGI2 + eNOS → vascular dilation + permeability. "
            "HAE-KNG1 MECHANISM: KNG1 gain/alteration-of-function variant → enhanced bradykinin release "
            "from HMWK (possibly by increasing cleavage susceptibility or reducing degradation). "
            "PHENOTYPE: recurrent non-histaminergic angioedema; normal complement; normal F12/PLG/ANGPT1. "
            "LABORATORY: C4, C1-INH, F12 — all normal; bradykinin assay (technically demanding) may show "
            "elevated bradykinin during attacks; HMWK cleavage products may be detectable. "
            "MANAGEMENT: icatibant (B2 antagonist — directly targets bradykinin effect → FIRST LINE); "
            "C1-INH concentrate (reduces kallikrein activity → reduces HMWK cleavage); "
            "lanadelumab prophylaxis (inhibits kallikrein → reduces HMWK cleavage → less bradykinin)."
        ),
        "aa": "644 aa",
        "kDa": "~120 kDa",
        "locus": "3q27.3",
        "omim_gene": 612358,
        "omim_disease": 619360,
        "inheritance": "AD; few families described; variants in bradykinin-containing domain 4 suspected",
        "gene_class": (
            "KNG1 encodes kininogen-1 through alternative splicing. "
            "DOMAIN STRUCTURE of HMWK: signal peptide → domain 1 (D1, binds cysteine protease inhibitors) → "
            "domain 2 (D2, cystatin-like) → domain 3 (D3, cystatin-like) → "
            "domain 4 (D4, bradykinin RYPPGFSPFR9aa) → domain 5 (D5H, histidine-rich, surface binding, "
            "prekallikrein + FXI binding) → domain 6 (D6, LMWK C-term in LMWK isoform). "
            "COFACTOR ROLE: HMWK binds negatively charged surfaces via D5; recruits prekallikrein and FXI "
            "to these surfaces (template function), facilitating FXIIa-mediated activation. "
            "BRADYKININ SEQUENCE: the 9aa bradykinin (His-362–Thr-363 cleavage) and kallidin "
            "(Lys-bradykinin, 10aa; tissue kallikrein cleaves LMWK at D4) are both derived from KNG1. "
            "INHIBITION: C1-INH inhibits kallikrein → reduces HMWK cleavage; "
            "carboxypeptidase N degrades bradykinin in plasma; "
            "ACE (kininase II) is the primary bradykinin-degrading enzyme — its inhibition by ACE-I "
            "doubles circulating bradykinin concentration → explains ACE-inhibitor-induced angioedema."
        ),
        "n_patients": 40,
        "key_alerts": [
            "KNG1-BRADYKININ-DIRECTLY: KNG1 is the direct bradykinin precursor; HAE-KNG1 variants enhance bradykinin liberation from HMWK; icatibant (bradykinin B2 receptor antagonist) is therefore the most mechanistically targeted acute therapy — superior rationale compared to indirect therapies (C1-INH, kallikrein inhibitors) though all can work",
            "KNG1-ACE-INHIBITOR-INTERACTION: ACE (kininase II) degrades bradykinin; ACE inhibitors block this degradation, doubling plasma bradykinin; in KNG1-HAE with already-enhanced bradykinin release, ACE inhibitors are ESPECIALLY dangerous — even patients on ACE-inhibitors for hypertension without known HAE can have KNG1 variant contributing to ACE-inhibitor-induced angioedema; test KNG1 in ACE-I angioedema that does not completely resolve after ACE-I discontinuation",
            "KNG1-BRADYKININ-ASSAY-DURING-ATTACK: Plasma bradykinin rises 8-20-fold during HAE attacks; measurement during an attack provides biological confirmation; samples must be taken into chilled plasma-kallikrein-inhibitor containing tubes and processed immediately; normal bradykinin between attacks does not exclude diagnosis",
        ],
        "etiologies": {
            "Missense in domain 4 (bradykinin region) — enhanced cleavage susceptibility": 18,
            "Missense in D5H (surface-binding) — enhanced HMWK complex assembly": 11,
            "Frameshift/truncation — altered bradykinin release kinetics": 7,
            "Splice-site — alternate exon inclusion": 4,
        },
        "stats": {
            "mean_dx_age_y": 29.5,
            "mean_dx_delay_months": 90.0,
            "pct_c4_normal": 100,
            "pct_icatibant_first_choice": 100,
            "pct_ace_inhibitor_exacerbation": 25,
            "pct_bradykinin_elevated_attack": 90,
        },
        "dx_delay_distribution": {"<1 y": 2, "1–5 y": 8, "5–15 y": 19, ">15 y": 11},
    },
    # ── HS3ST6 — HAE-HS3ST6 ──────────────────────────────────────────────
    {
        "gene": "HS3ST6",
        "protein": "HS3ST6 — HAE-HS3ST6 AD — Heparan Sulfate 3-O-Sulfotransferase 6 — Contact Activation Surface Modification",
        "alias": (
            "HS3ST6; OMIM gene 609680; HAE-HS3ST6 OMIM 619389; "
            "16p13.3; 333 aa; ~38 kDa; AD. "
            "HS3ST6 encodes heparan sulfate glucosamine 3-O-sulfotransferase 6 (HS3OST6), one of seven "
            "3-O-sulfotransferase isoforms in humans. "
            "FUNCTION: HS3ST6 catalyses the 3-O-sulfation of specific glucosamine residues within "
            "heparan sulfate proteoglycan (HSPG) chains — a rare modification that occurs late in "
            "the biosynthesis of HSPGs in the Golgi. "
            "HSPG AND CONTACT ACTIVATION: HSPGs on the endothelial surface provide a highly "
            "anionic charged template that nucleates contact activation of Factor XII; "
            "3-O-sulfation of specific HS sites modulates FXII binding affinity and autoproteolysis. "
            "HAE-HS3ST6 MECHANISM: HS3ST6 variant → altered HSPG sulfation pattern → "
            "enhanced FXII contact activation on endothelial surface → amplified kallikrein → "
            "excess bradykinin → angioedema. "
            "PHENOTYPE: recurrent non-histaminergic angioedema; normal complement; normal F12, PLG, KNG1. "
            "MANAGEMENT: standard HAE-nC1INH approach; icatibant acute; lanadelumab prophylaxis. "
            "LABORATORY BIOMARKER: None; diagnosis entirely genetic. "
            "NOTE: HS3ST6 is functionally distinct from HS3ST1–HS3ST3 isoforms which modify "
            "antithrombin-binding HS sites (heparin pharmacology target) — do not confuse."
        ),
        "aa": "333 aa",
        "kDa": "~38 kDa",
        "locus": "16p13.3",
        "omim_gene": 609680,
        "omim_disease": 619389,
        "inheritance": "AD; very rare; limited kindreds; role fully confirmed in recent HAE-nC1INH panels",
        "gene_class": (
            "HS3ST6 encodes a member of the heparan sulfate 3-O-sulfotransferase family. "
            "ENZYME STRUCTURE: single-pass type II transmembrane Golgi enzyme; "
            "cytoplasmic N-term → transmembrane → lumenal catalytic domain. "
            "REACTION: 3'-phosphoadenosine-5'-phosphosulfate (PAPS) + "
            "GlcNHSO3 in HS → GlcNHSO3(3S) + adenosine-3',5'-bisphosphate. "
            "SUBSTRATE SPECIFICITY: HS3ST6 acts on specific N-sulfated glucosamine residues adjacent "
            "to L-iduronic acid 2-O-sulfate in HS chains — a rare target site generating "
            '3-O-sulfated HS which affects specific protein-binding interactions. '
            "DIFFERENCE FROM HS3ST1: HS3ST1 generates the antithrombin-binding HS site "
            "(clinical heparin); HS3ST6 generates a distinct sulfation pattern affecting FXII "
            "contact activation rather than antithrombin binding. "
            "HAE-HS3ST6 is the most recently characterised subtype, requiring full HAE panel coverage."
        ),
        "n_patients": 40,
        "key_alerts": [
            "HS3ST6-DIAGNOSIS-GENETIC-ONLY: No biomarker identifies HAE-HS3ST6; diagnosis requires genetic panel; must be suspected in HAE-nC1INH (normal C4, C1-INH, F12, PLG) that remains genetically unexplained; a comprehensive panel covering HS3ST6 is essential",
            "HS3ST6-NOT-ANTITHROMBIN-HEPARIN-PATHWAY: HS3ST6 is NOT the enzyme that generates the antithrombin-binding heparin site (that is HS3ST1); there is NO associated bleeding or thrombotic phenotype in HAE-HS3ST6; do not confuse HS3ST6 findings with heparin sensitivity or antithrombin deficiency on genetic reports",
            "HS3ST6-CONTACT-ACTIVATION-SURFACE: The primary mechanism is altered endothelial surface charge/sulfation increasing FXII autoactivation efficiency; mechanistically, kallikrein inhibitors (lanadelumab, berotralstat) should be effective by cutting the pathway downstream of FXII activation; clinical data accumulating",
        ],
        "etiologies": {
            "Missense in catalytic domain — altered PAPS sulfotransfer activity": 19,
            "Frameshift/nonsense — loss of enzyme activity": 11,
            "Splice-site — truncated enzyme": 7,
            "Large deletion": 3,
        },
        "stats": {
            "mean_dx_age_y": 26.8,
            "mean_dx_delay_months": 114.0,
            "pct_c4_normal": 100,
            "pct_c1inh_normal": 100,
            "pct_f12_normal": 100,
            "pct_genetically_unexplained_before_panel": 100,
        },
        "dx_delay_distribution": {"<1 y": 1, "1–5 y": 5, "5–15 y": 21, ">15 y": 13},
    },
    # ── KLKB1 — Prekallikrein deficiency / HAE-KLKB1 ─────────────────────
    {
        "gene": "KLKB1",
        "protein": "KLKB1 — Prekallikrein Deficiency AR — Prolonged APTT Without Bleeding — Episodic Angioedema Subset",
        "alias": (
            "KLKB1; OMIM gene 229000; Prekallikrein (Fletcher factor) deficiency OMIM 612423; "
            "4q35.2; 638 aa; ~88 kDa (mature, processed); AR. "
            "KLKB1 encodes plasma prekallikrein (Fletcher factor), the plasma zymogen of kallikrein. "
            "FUNCTION: prekallikrein circulates in plasma bound to HMWK (KNG1) as a bimolecular complex; "
            "FXIIa cleaves prekallikrein → kallikrein; kallikrein in turn cleaves more FXII "
            "(positive feedback amplification loop) and cleaves HMWK to release bradykinin. "
            "PREKALLIKREIN DEFICIENCY (AR, KLKB1 LOF): "
            "Paradox: KLKB1 LOF → absent kallikrein → LESS bradykinin; yet some patients develop "
            "episodic angioedema-like attacks. "
            "MECHANISMS PROPOSED: (1) absent prekallikrein → compensatory FXII overactivation → "
            "alternative bradykinin-generating pathway; (2) reduced HMWK-binding capacity on surface → "
            "altered contact activation kinetics; (3) coincidental HAE phenotype in families with "
            "dual rare variants; the HAE association of KLKB1 LOF is less firmly established than "
            "the other HAE-nC1INH genes and is currently categorised as 'HAE-unknown' in some guidelines. "
            "LABORATORY: APTT MARKEDLY PROLONGED (contact activation severely impaired in vitro → "
            "no clot initiation via FXII-FXI pathway in APTT tube); "
            "PT, thrombin time: NORMAL; NO BLEEDING (contact pathway is dispensable in vivo haemostasis); "
            "prekallikrein activity: low/absent; HMWK: normal or low. "
            "APTT CORRECTION: mixes 50:50 with normal plasma → CORRECTS (factor deficiency, not inhibitor); "
            "characteristic 'incubation correction' phenomenon — prolonged APTT shortens with prolonged "
            "incubation of APTT tube (FXII activates spontaneously during long incubation). "
            "MANAGEMENT OF ANGIOEDEMA: icatibant if bradykinin-mediated component; "
            "C1-INH concentrate; no specific prekallikrein replacement therapy available. "
            "NO TREATMENT NEEDED FOR APTT: KLKB1 deficiency DOES NOT REQUIRE TREATMENT for the APTT "
            "prolongation itself — no haemostatic support before surgery needed solely because of this."
        ),
        "aa": "638 aa",
        "kDa": "~88 kDa",
        "locus": "4q35.2",
        "omim_gene": 229000,
        "omim_disease": 612423,
        "inheritance": "AR biallelic LOF; extremely rare; consanguineous families; APTT prolongation incidentally found in anaesthetic workup",
        "gene_class": (
            "KLKB1 encodes plasma prekallikrein (Fletcher factor). "
            "DOMAIN STRUCTURE: signal peptide → 4 apple domains (A1–A4, each ~90 aa, β-strand-dominated; "
            "A1 binds C1-INH; A2 contains FXIIa cleavage site; A3 binds HK/HMWK; A4 binds FXI) → "
            "serine protease domain (catalytic triad His-Asp-Ser). "
            "ACTIVATION: FXIIa cleaves Arg-371–Ile-372 in apple domain A2 → kallikrein "
            "(disulfide-linked heavy chain + light chain/serine protease). "
            "KALLIKREIN SUBSTRATES: HMWK (→ bradykinin), FXII (→ FXIIa, positive feedback), "
            "pro-urokinase (→ uPA), pro-HGF (→ hepatocyte growth factor). "
            "Apple domains A1–A4 are shared with FXI (F11) — KLKB1 and F11 share evolutionary origin "
            "from gene duplication; F11 deficiency (Haemophilia C) is a related but distinct contact "
            "factor deficiency with mild bleeding tendency."
        ),
        "n_patients": 40,
        "key_alerts": [
            "KLKB1-PROLONGED-APTT-NO-BLEEDING: Prekallikrein deficiency causes one of the most severely prolonged APTTs in clinical practice (often >120 seconds) WITHOUT ANY BLEEDING — do not transfuse FFP or withhold surgery solely because of APTT prolongation; contact the haematology team; establish KLKB1 deficiency as the cause; no haemostatic replacement needed perioperatively",
            "KLKB1-APTT-INCUBATION-CORRECTION: A characteristic laboratory finding of KLKB1 deficiency is 'incubation correction' — the prolonged APTT progressively shortens when the reaction mixture is incubated longer before adding CaCl2; this happens because prolonged incubation allows spontaneous FXII activation without kallikrein; this feature distinguishes KLKB1 deficiency from most coagulation factor deficiencies",
            "KLKB1-ANGIOEDEMA-LINK-DISPUTED: Not all KLKB1-deficient patients develop angioedema; the HAE-KLKB1 association is less firmly established than HAE-SERPING1 or HAE-FXII; when angioedema occurs in KLKB1-deficient patients, it may represent coincident genetic HAE or a secondary mechanism; icatibant is tried empirically for attacks",
            "KLKB1-DO-NOT-TREAT-APTT: A key teaching point — KLKB1 deficiency, FXII deficiency (F12 LOF — different from HAE-FXII GOF), and HMWK deficiency (KNG1 complete LOF) all cause prolonged APTT without bleeding; the in vivo haemostasis is INTACT; do NOT administer FFP to 'correct' the APTT — this exposes patients to transfusion risks with no haemostatic benefit",
        ],
        "etiologies": {
            "Missense in serine protease domain — reduced/absent catalytic activity": 16,
            "Frameshift/nonsense — truncation, absent protein (AR)": 13,
            "Splice-site — exon skipping, absent or truncated prekallikrein": 8,
            "Large deletion": 3,
        },
        "stats": {
            "mean_dx_age_y": 38.5,
            "mean_dx_delay_months": 120.0,
            "pct_incidental_aptt_discovery": 75,
            "pct_aptt_greater_120s": 82,
            "pct_bleeding_absent": 100,
            "pct_angioedema_subset": 30,
            "pct_misdiagnosed_coagulopathy": 68,
        },
        "dx_delay_distribution": {"<1 y": 8, "1–5 y": 14, "5–15 y": 12, ">15 y": 6},
    },
]


# ─── Patient cohort generation ────────────────────────────────────────────────

def _make_cohort():
    cohort = {}
    for i, gene_info in enumerate(HAE_GENES):
        seed = SEED_BASE + i
        rng = random.Random(seed)
        gene = gene_info["gene"]
        n = gene_info["n_patients"]
        patients = []
        for p in range(n):
            age_dx = round(rng.gauss(gene_info["stats"].get("mean_dx_age_y", 28), 9), 1)
            age_dx = max(0.5, min(80, age_dx))
            dx_delay = round(rng.gauss(gene_info["stats"].get("mean_dx_delay_months", 90), 30), 1)
            dx_delay = max(1.0, min(240, dx_delay))
            patients.append({
                "patient_id": f"{gene}-{seed}-{p+1:03d}",
                "gene": gene,
                "age_at_diagnosis": age_dx,
                "diagnosis_delay_months": dx_delay,
                "seed": seed,
            })
        cohort[gene] = {
            **gene_info,
            "patients": patients,
        }
    return cohort


_COHORT = _make_cohort()


# ─── API functions ─────────────────────────────────────────────────────────────

def get_overview():
    total = sum(v["n_patients"] for v in _COHORT.values())
    mean_dx_age = round(
        sum(p["age_at_diagnosis"] for v in _COHORT.values() for p in v["patients"]) / total, 1
    )
    mean_dx_delay = round(
        sum(p["diagnosis_delay_months"] for v in _COHORT.values() for p in v["patients"]) / total, 1
    )

    top_alerts = []
    for v in _COHORT.values():
        top_alerts.extend(v["key_alerts"][:2])

    genes_summary = []
    for g, v in _COHORT.items():
        pts = v["patients"]
        mean_age = round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1)
        genes_summary.append({
            "gene": g,
            "protein_short": v["protein"][:80],
            "locus": v["locus"],
            "inheritance": v["inheritance"].split(";")[0],
            "omim_disease": v["omim_disease"],
            "mean_dx_age": mean_age,
            "n_patients": v["n_patients"],
        })

    serping1 = _COHORT["SERPING1"]["stats"]
    f12      = _COHORT["F12"]["stats"]
    plg      = _COHORT["PLG"]["stats"]
    klkb1    = _COHORT["KLKB1"]["stats"]

    return {
        "atlas": "Hereditary-Angioedema-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Angioedema Reference",
        "genes": genes_summary,
        "aggregate_stats": {
            "total_patients": total,
            "mean_dx_age_years": mean_dx_age,
            "mean_dx_delay_months": mean_dx_delay,
            "serping1_c4_low_pct": serping1["pct_c4_low_between_attacks"],
            "serping1_laryngeal_pct": serping1["pct_laryngeal_attacks"],
            "serping1_abdominal_pct": serping1["pct_abdominal_attacks"],
            "serping1_misdiagnosed_allergy_pct": serping1["pct_misdiagnosed_allergy"],
            "serping1_unnecessary_surgery_pct": serping1["pct_unnecessary_surgery"],
            "f12_female_pct": f12["pct_female_symptomatic"],
            "f12_ocp_triggered_pct": f12["pct_ocp_triggered"],
            "plg_tranexamic_effective_pct": plg["pct_tranexamic_effective"],
            "klkb1_aptt_no_bleeding_pct": klkb1["pct_bleeding_absent"],
            "klkb1_misdiagnosed_coagulopathy_pct": klkb1["pct_misdiagnosed_coagulopathy"],
            "cascade_tested_pct": 62,
        },
        "top_alerts": top_alerts,
        "seeds": list(range(SEED_BASE, SEED_BASE + 8)),
    }


def get_breakdown():
    result = {}
    for gene, info in _COHORT.items():
        pts = info["patients"]
        result[gene] = {
            "gene": gene,
            "n_patients": info["n_patients"],
            "alias": info["alias"],
            "gene_class": info["gene_class"],
            "locus": info["locus"],
            "aa": info["aa"],
            "kDa": info["kDa"],
            "omim_gene": info["omim_gene"],
            "omim_disease": info["omim_disease"],
            "inheritance": info["inheritance"],
            "key_alerts": info["key_alerts"],
            "etiologies": info["etiologies"],
            "stats": info["stats"],
            "dx_delay_distribution": info["dx_delay_distribution"],
            "patients": pts[:10],
        }
    return result


def get_definitions():
    return {
        "atlas": "Hereditary-Angioedema-Atlas",
        "concepts": {
            "HAE Diagnostic Framework — C1-INH-Dependent vs. Normal-C1-INH Subtypes": (
                "Hereditary Angioedema divides into two broad categories based on complement levels: "
                "(1) C1-INH-DEPENDENT HAE (SERPING1 mutations): "
                "C4 LOW between attacks (hallmark screening test, ~95% sensitive); "
                "C1-INH antigen LOW (type I) or NORMAL/HIGH but DYSFUNCTIONAL (type II); "
                "C3 NORMAL (C3 convertase not excessively activated); "
                "C1q NORMAL (distinguishes from acquired C1-INH deficiency where C1q is LOW). "
                "(2) HAE WITH NORMAL C1-INH (HAE-nC1INH): "
                "All complement studies NORMAL (C4, C1-INH antigen, C1-INH function, C3, C1q); "
                "Diagnosis requires genetic testing — F12 GOF variants (most common), PLG Lys330Glu, "
                "ANGPT1, MYOF, KNG1, HS3ST6, KLKB1; "
                "combined HAE-nC1INH genetic panel is the standard investigation. "
                "BRADYKININ vs. HISTAMINE DISTINCTION (most critical clinical differentiation): "
                "HAE: non-pruritic, non-urticarial, non-pitting edema; NOT responsive to antihistamines/corticosteroids; "
                "Allergic/histaminergic angioedema: pruritic, urticarial, responds to antihistamines; "
                "ACE-inhibitor angioedema: bradykinin-mediated (ACE degrades bradykinin); resolves within "
                "days-weeks after ACE-I discontinuation; may also respond to icatibant. "
                "DEATH RISK: laryngeal HAE is the highest-risk acute presentation; "
                "historical mortality 25-30% in untreated families; modern mortality near zero with "
                "home icatibant/C1-INH available."
            ),
            "Bradykinin Pathway — Contact Activation Cascade": (
                "The bradykinin pathway (contact activation / kallikrein-kinin system) generates "
                "bradykinin in plasma: "
                "TRIGGER: negatively charged surface (polyphosphates, collagen, nucleic acids, misfolded proteins) "
                "activates Factor XII (FXII, Hageman factor). "
                "AMPLIFICATION: FXIIa cleaves prekallikrein (KLKB1) → plasma kallikrein; "
                "kallikrein cleaves more FXII → positive feedback amplification; "
                "kallikrein cleaves HMWK (KNG1) → bradykinin (9aa) + kinin-free kininogen. "
                "BRADYKININ EFFECTS: bradykinin B2 receptor (BDKRB2) on endothelium → "
                "eNOS activation → NO → vasodilatation; "
                "phospholipase C → IP3 → cytoplasmic Ca²⁺ release → "
                "VE-cadherin internalisation → gap formation → plasma extravasation. "
                "DEGRADATION: ACE (kininase II, endothelium) converts bradykinin → inactive fragments "
                "(explains ACE-inhibitor angioedema — removing the brake on bradykinin). "
                "C1-INH CONTROL: C1-INH inhibits FXIIa (major) and kallikrein (major) → "
                "two points of control; haploinsufficiency removes both brakes. "
                "HAE-nC1INH GENES: F12 (GOF → more FXIIa), PLG (plasmin activates FXII), "
                "KNG1 (bradykinin precursor enhancement), HS3ST6 (surface activation modulation), "
                "ANGPT1/MYOF (endothelial barrier — amplifies permeability at lower bradykinin threshold)."
            ),
            "Acute HAE Treatment — Mechanism-Based Hierarchy": (
                "FIRST-LINE SPECIFIC THERAPY (all should be available at home for HAE patients): "
                "1. Icatibant (Firazyr) — selective bradykinin B2 receptor antagonist; "
                "   30 mg SC injection; onset within 30–60 minutes; duration 6–8 hours; "
                "   self-administered SC injection (abdominal); approved for adults; "
                "   safe in HAE-nC1INH subtypes as well (bradykinin common final mediator). "
                "2. C1-INH concentrate IV — plasma-derived (Berinert 20 U/kg; Cinryze) or "
                "   recombinant (Ruconest/Conestat alfa 50 U/kg); "
                "   onset within 30–60 min; restores C1-INH activity directly; "
                "   preferred in pregnancy (safest track record). "
                "3. C1-INH SC (Haegarda) — high-dose SC twice weekly for PROPHYLAXIS; "
                "   also used on-demand for milder attacks. "
                "4. Ecallantide (Kalbitor) — plasma kallikrein inhibitor (anti-kallikrein antibody); "
                "   30 mg SC; physician-administered only (anaphylaxis risk 3–4%); "
                "   approved USA only (not EU). "
                "LAST-RESORT IF SPECIFIC THERAPY UNAVAILABLE: "
                "Fresh frozen plasma (FFP) 2 units IV — contains C1-INH, HMWK, FXII; "
                "transiently normalises the contact system; risk of bloodborne infections + "
                "paradoxical worsening (FFP also contains substrates HMWK + FXII). "
                "DO NOT USE: antihistamines, corticosteroids, adrenaline — "
                "these are for histaminergic anaphylaxis NOT bradykinin-mediated HAE; "
                "adrenaline may provide very transient relief but does NOT treat underlying mechanism. "
                "LARYNGEAL ATTACKS: home treatment immediately + activate emergency services simultaneously; "
                "if intubation needed, use video laryngoscopy (markedly oedematous airway); "
                "surgical airway as backup plan."
            ),
            "Long-Term HAE Prophylaxis — Modern Options": (
                "CURRENT PROPHYLACTIC AGENTS (approved or widely used): "
                "1. Lanadelumab (Takhzyro) — anti-kallikrein monoclonal antibody; "
                "   300 mg SC every 2 weeks (or every 4 weeks in well-controlled patients); "
                "   reduces attack rate by 87% vs placebo; approved for SERPING1-HAE; "
                "   data accumulating for HAE-nC1INH (F12, PLG). "
                "2. Berotralstat (Orladeyo) — oral plasma kallikrein inhibitor; "
                "   150 mg oral once daily; reduces attack rate ~44%; "
                "   convenient oral route; drug interactions via CYP3A4 inhibition. "
                "3. C1-INH SC (Haegarda) — 60 U/kg twice weekly SC; "
                "   ~90% reduction in attack rate; particularly suitable for SERPING1-HAE; "
                "   preferred in pregnancy. "
                "4. C1-INH IV (Cinryze) — 1000 U IV every 3–4 days; "
                "   venous access burden; used in children and those not tolerating SC. "
                "OLDER PROPHYLAXIS (falling out of use for long-term): "
                "Tranexamic acid — antifibrinolytic; 1–1.5 g TID oral; "
                "reduces attack frequency in SERPING1-HAE and HAE-PLG (rationale: PLG pathway); "
                "risk: deep venous thrombosis with prolonged use; less effective than modern agents. "
                "Danazol (attenuated androgen) — increases C1-INH production; "
                "dose 100–200 mg daily; highly effective; "
                "contraindicated in children, pregnancy, androgen-sensitive tumours; "
                "side effects: virilisation, hepatotoxicity, dyslipidaemia; now largely replaced. "
                "SHORT-TERM PROPHYLAXIS (before surgery/dental/procedure): "
                "C1-INH IV 1000 U 1–6 hours before procedure → prevents attack in peri-procedural period; "
                "icatibant 30 mg SC 2 hours before as alternative."
            ),
        },
        "pharmacological_distinctions": [
            "Icatibant vs C1-INH concentrate: Icatibant blocks bradykinin ACTION (B2 receptor); C1-INH replaces the missing inhibitor upstream; both work but icatibant acts faster (30 min) and is self-administered SC; C1-INH is preferred in pregnancy and paediatric HAE",
            "Lanadelumab vs Berotralstat: Both prevent attacks by reducing kallikrein; Lanadelumab (SC every 2–4 weeks, monoclonal antibody) achieves ~87% attack reduction; Berotralstat (oral daily, small molecule) achieves ~44% reduction; patient preference and CYP3A4 drug interactions guide choice",
            "HAE vs Allergic Angioedema treatment: HAE (bradykinin-mediated) does NOT respond to antihistamines, corticosteroids, or adrenaline; allergic/histaminergic angioedema DOES; misapplying allergy treatment to HAE is a life-threatening error — especially for laryngeal HAE where delays in specific therapy cause death",
            "Tranexamic acid in HAE-PLG: Tranexamic acid (lysine analogue, antifibrinolytic) is specifically rational in HAE-PLG — it blocks plasminogen lysine-binding sites, reducing plasmin formation from the Lys330Glu GOF plasminogen → less FXII activation → less bradykinin; more effective in PLG-HAE than SERPING1-HAE",
            "ACE inhibitors and HAE: ACE (kininase II) normally degrades bradykinin; ACE inhibitors remove this degradation → bradykinin accumulates → angioedema (ACE-inhibitor-induced angioedema, AEI-ACE); AEI-ACE can be treated with icatibant; ACE inhibitors are ABSOLUTELY CONTRAINDICATED in all HAE genotypes",
        ],
        "key_standards": [
            "WAO/EAACI HAE International Guidelines 2021: icatibant or C1-INH concentrate as first-line acute treatment; every HAE patient should have home emergency medication; laryngeal attacks require immediate treatment + EMS activation simultaneously",
            "HAE-nC1INH Consensus 2022: diagnosis requires exclusion of SERPING1 mutation + genetic panel for F12, PLG, ANGPT1, MYOF, KNG1, HS3ST6 variants; normal complement does NOT exclude HAE",
            "CASCADE TESTING: all first-degree relatives of any HAE patient (all subtypes) require genetic counselling and genetic testing; children should be tested early given risk of first attack at school age",
            "C4 MONITORING: C4 is the best SCREENING test for SERPING1-HAE (95% sensitive between attacks); functional C1-INH assay is mandatory to distinguish type I from type II and from acquired C1-INH deficiency",
            "AVOID OESTROGENS: All HAE patients (especially F12, PLG subtypes) — no oestrogen-containing OCP, no oestrogen HRT; use progestogen-only or non-hormonal contraception; confirm with prescribing physician",
        ],
    }
