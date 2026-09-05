#!/usr/bin/env python3
"""Coagulopathy-Atlas — Complete 8-Gene Hereditary Bleeding Disorders Atlas
F8      (Factor VIII; ~2351 aa; Xq28; serine protease cofactor; XLR; 1:5000 males; Hemophilia A; emicizumab FDA 2017; FVIII inhibitors in 30% severe) ·
F9      (Factor IX; ~461 aa; Xq27.1; serine protease; XLR; 1:30000 males; Hemophilia B/Christmas; etranacogene dezaparvovec FDA 2022 gene therapy) ·
VWF     (von Willebrand Factor; ~2813 aa; 12p13.31; platelet adhesion + FVIII carrier; AD/AR; most common hereditary bleeding; DDAVP Type1/2A; CI in Type2B) ·
F11     (Factor XI; ~625 aa; 4q35.2; contact/intrinsic pathway; AR; Ashkenazi founder E117X/F283L; Hemophilia C; poor bleed-level correlation) ·
F7      (Factor VII; ~466 aa; 13q34; extrinsic/TF pathway; AR; most common rare factor deficiency 1:500000; rFVIIa/NovoSeven) ·
F13A1   (Factor XIII A-subunit; ~732 aa; 6p25.1; fibrin cross-linking; AR; delayed bleeding 24-48h PATHOGNOMONIC; umbilical stump; normal PT/aPTT/TT — MISSES) ·
ITGA2B  (Integrin αIIb/GpIIb; ~1039 aa; 17q21.31; platelet aggregation fibrinogen receptor; AR; Glanzmann thrombasthenia; absent ADP/collagen/AA aggregation; normal count) ·
GP1BA   (Glycoprotein Ibα; ~626 aa; 17p13.2; VWF receptor; AR; Bernard-Soulier syndrome; giant platelets + thrombocytopenia PATHOGNOMONIC; absent ristocetin agglutination)
320-patient aggregate cohort (8 × 40, seeds 1142–1149)
"""

import random

SEED_BASE = 1142

COAGULOPATHY_GENES = [
    # ── F8 — Factor VIII, Hemophilia A ──────────────────────────────────────────
    {
        "gene": "F8",
        "protein": "Factor VIII (FVIII)",
        "alias": (
            "F8; OMIM gene 300841; Xq28; ~2351 aa; Hemophilia A (OMIM #306700); "
            "XLR; prevalence 1:5000 males (most common severe hereditary bleeding disorder); "
            "serine protease cofactor of FIXa in intrinsic tenase complex (FIXa-FVIIIa); "
            "emicizumab (ACE910) bispecific antibody FDA 2017/2018 — bridges FIXa and FX; "
            "FVIII inhibitors develop in 25-30% of severe HA patients (FVIII neutralising Abs)"
        ),
        "aa": "~2351 aa",
        "kDa": "~267 kDa",
        "gene_class": "Serine protease cofactor; intrinsic tenase complex; coagulation cascade amplifier",
        "locus": "Xq28",
        "omim_gene": 300841,
        "omim_disease": 306700,
        "phenotype": "Hemophilia A — most common severe hereditary bleeding disorder in males; haemarthroses, muscle bleeds, intracranial haemorrhage; FVIII inhibitors 25-30% severe",
        "disease": (
            "F8 encodes the non-enzymatic cofactor Factor VIII (FVIII), which is activated by "
            "thrombin to FVIIIa and functions as the essential cofactor for Factor IXa (FIXa) "
            "in the intrinsic tenase complex (FIXa·FVIIIa·phospholipid·Ca²⁺), which converts "
            "Factor X (FX) to FXa — a critical step in propagation-phase thrombin generation. "
            "NORMAL FUNCTION: FVIIIa dramatically (>10⁵-fold) accelerates the rate at which "
            "FIXa activates FX, enabling sufficient thrombin generation for a stable fibrin clot. "
            "FVIII circulates in complex with VWF (which protects FVIII from premature clearance; "
            "VWF LOF → secondary FVIII reduction). "
            "PATHOMECHANISM: Hemophilia A is caused by LOF variants in F8 (missense, nonsense, "
            "frameshift, large deletions, inversion — Intron 22 inversion accounts for 45% of severe HA). "
            "Severe HA: FVIII <1 IU/dL (<1% normal) — haemarthroses (target joints: ankles, knees, elbows), "
            "deep muscle haematomas, intracranial haemorrhage (ICH — 2-8% lifetime risk, leading cause of death). "
            "Moderate HA: FVIII 1-5 IU/dL — bleeds with minor trauma. "
            "Mild HA: FVIII 5-40 IU/dL — bleeds with surgery or major trauma; often detected incidentally. "
            "SEVERITY CLASSIFICATION: Severe (<1% FVIII), Moderate (1-5%), Mild (5-40%). "
            "TREATMENT: FVIII concentrate replacement (plasma-derived or recombinant); "
            "prophylaxis (2-3×/week IV) vs on-demand. "
            "EMICIZUMAB (Hemlibra): bispecific antibody mimicking FVIIIa function (binds FIXa + FX simultaneously); "
            "subcutaneous weekly/biweekly/monthly — FDA 2017 for inhibitors, 2018 for all severe HA; "
            "game-changer: no IV needed, effective regardless of inhibitor status. "
            "INHIBITORS (neutralising anti-FVIII antibodies): develop in 25-30% of severe HA — "
            "most critical complication; renders standard FVIII replacement ineffective; "
            "bypass therapy: aPCC (FEIBA) or rFVIIa (NovoSeven) until immune tolerance induction (ITI). "
            "GENE THERAPY: multiple products in trials/approval; fitusiran (antithrombin siRNA) FDA 2024 "
            "for HA + HB with/without inhibitors. "
            "DESMOPRESSIN (DDAVP): useful in mild HA (releases stored FVIII from endothelium); "
            "NOT effective in moderate/severe HA; test with DDAVP trial before use."
        ),
        "inheritance": "X-Linked Recessive (XLR); males affected; females carriers (may have mild bleeding due to lyonisation — check FVIII level in all carrier females); 30% de novo",
        "hallmark": "Haemarthroses (spontaneous, target joints), deep muscle haematomas, prolonged aPTT, normal PT/TT/platelet count/bleeding time; FVIII <1% (severe); Intron 22 inversion in 45% severe HA",
        "key_ddx": "Hemophilia B (F9, identical clinical picture, prolonged aPTT, FVIII normal), vWD Type 3 (secondary FVIII reduction via VWF loss, but VWF antigen also low), Combined FV+FVIII deficiency (LMAN1/MCFD2, milder, both factors reduced)",
        "treatment_alert": "Emicizumab SC (weekly/biweekly/monthly) FDA 2017/2018 — game-changer for all HA including inhibitors; FVIII concentrate (recombinant preferred); DDAVP mild HA only; inhibitors → rFVIIa/aPCC bypass + ITI; gene therapy fitusiran FDA 2024",
        "seed": 1142,
        "cohort_n": 40,
        # Clinical rates
        "severe_bleed_rate": 0.75,        # haemarthrosis/muscle bleed in severe HA
        "inhibitor_rate": 0.28,           # inhibitor development
        "icb_rate": 0.05,                 # intracranial bleed
        "joint_damage_rate": 0.55,        # target joint arthropathy
        "on_prophylaxis_rate": 0.80,
        "emicizumab_rate": 0.60,
        "drug_error_rate": 0.07,
        "severity_weights": {"Severe": 0.45, "Moderate": 0.25, "Mild": 0.30},
        "primary_complication": "Haemarthrosis / FVIII inhibitor",
        "platelet_count": "Normal",
        "pt_ptt": "aPTT prolonged; PT normal",
    },

    # ── F9 — Factor IX, Hemophilia B ──────────────────────────────────────────
    {
        "gene": "F9",
        "protein": "Factor IX (FIX)",
        "alias": (
            "F9; OMIM gene 300746; Xq27.1; ~461 aa; Hemophilia B / Christmas disease (OMIM #306900); "
            "XLR; prevalence 1:30000 males; serine protease; intrinsic tenase complex; "
            "etranacogene dezaparvovec (Hemgenix) FDA 2022 — first haemophilia B gene therapy approved; "
            "Leyden phenotype: severe→mild at puberty (testosterone-dependent F9 promoter)"
        ),
        "aa": "~461 aa",
        "kDa": "~55 kDa",
        "gene_class": "Vitamin K-dependent serine protease; intrinsic tenase FIXa; coagulation factor",
        "locus": "Xq27.1",
        "omim_gene": 300746,
        "omim_disease": 306900,
        "phenotype": "Hemophilia B (Christmas disease) — identical clinical phenotype to HA; haemarthroses, muscle bleeds, ICH; FIX <1% severe; inhibitors rarer than HA (3-5%)",
        "disease": (
            "F9 encodes Factor IX (FIX), a Vitamin K-dependent serine protease that is activated "
            "by FXIa (intrinsic pathway) and by TF-FVIIa complex (extrinsic pathway). "
            "NORMAL FUNCTION: Activated FIXa forms the intrinsic tenase complex (FIXa + FVIIIa + "
            "phospholipid + Ca²⁺) which activates FX → FXa, leading to thrombin generation. "
            "Vitamin K carboxylates N-terminal glutamic acid residues (Gla domain) — essential for "
            "membrane binding; warfarin blocks this → FIX activity falls. "
            "PATHOMECHANISM: Hemophilia B is caused by LOF variants in F9 (missense most common; "
            "nonsense; frameshift; large deletions; promoter variants — Leyden type). "
            "CLINICAL: Clinically IDENTICAL to Hemophilia A — cannot be distinguished on clinical grounds; "
            "must differentiate by coagulation factor assay (FVIII normal in HB; FIX reduced). "
            "Laboratory: prolonged aPTT + normal PT + normal platelet count. "
            "Specific factor assay: FIX activity (chromogenic or one-stage clot). "
            "Severity: Severe (<1% FIX), Moderate (1-5%), Mild (5-40%). "
            "TREATMENT: FIX concentrate (recombinant preferred: nonacog alfa, eftrenonacog alfa — "
            "extended half-life fusion proteins allowing less frequent infusion). "
            "INHIBITORS: rare in HB (3-5% severe) — more often anaphylaxis to FIX products. "
            "ETRANACOGENE DEZAPARVOVEC (Hemgenix, AAV5-FIX-Padua): first approved HB gene therapy "
            "(FDA Nov 2022, EMA Feb 2023); single IV infusion → sustained FIX expression (>15×); "
            "FIX-Padua (R338L) variant: 8× higher FIX activity than wild-type → normal range with "
            "lower vector dose; results: 54% median bleed rate reduction, 96% prophylaxis-free at 18m. "
            "LEYDEN PHENOTYPE: Specific promoter variants (e.g., positions +13, -6, -5) cause severe HB "
            "in childhood → spontaneous improvement at puberty (testosterone activates F9 transcription "
            "via promoter androgen response element) → mild HB in adult males. "
            "Critical DDx: HA vs HB — identical clinical, aPTT prolonged; FVIII assay first if aPTT prolonged "
            "(FVIII low in HA, normal in HB); if FVIII normal → FIX assay."
        ),
        "inheritance": "X-Linked Recessive (XLR); males severely affected; female carriers may have reduced FIX (25% symptomatic); 30% de novo; Leyden variant: severe childhood → mild adult (testosterone-dependent transcription)",
        "hallmark": "Haemarthroses, muscle bleeds, prolonged aPTT, normal PT, FVIII NORMAL (key distinguishing feature from HA); FIX <1% severe; Leyden phenotype: spontaneous improvement at puberty",
        "key_ddx": "Hemophilia A (F8 low, aPTT prolonged, identical phenotype — factor assay mandatory), FXI deficiency (Hemophilia C, AR, mild-moderate, poor bleed correlation), FXII deficiency (prolonged aPTT but NO bleeding), Warfarin effect (vitamin K-dependent factors all low — ask drug history)",
        "treatment_alert": "FIX concentrate (recombinant preferred; EHL products: eftrenonacog alfa, albutrepenonacog alfa); gene therapy etranacogene dezaparvovec FDA 2022 (single infusion, Padua variant 8× activity); fitusiran (antithrombin RNAi) FDA 2024 for all haemophilias; inhibitor → rFVIIa bypass",
        "seed": 1143,
        "cohort_n": 40,
        "severe_bleed_rate": 0.72,
        "inhibitor_rate": 0.04,
        "icb_rate": 0.04,
        "joint_damage_rate": 0.52,
        "on_prophylaxis_rate": 0.78,
        "emicizumab_rate": 0.00,          # emicizumab NOT for HB (FIXa only bridges with FX, HB lacks FIX)
        "drug_error_rate": 0.06,
        "severity_weights": {"Severe": 0.40, "Moderate": 0.25, "Mild": 0.35},
        "primary_complication": "Haemarthrosis / joint arthropathy",
        "platelet_count": "Normal",
        "pt_ptt": "aPTT prolonged; PT normal",
    },

    # ── VWF — von Willebrand Disease ──────────────────────────────────────────
    {
        "gene": "VWF",
        "protein": "von Willebrand Factor (VWF)",
        "alias": (
            "VWF; OMIM gene 613160; 12p13.31; ~2813 aa; von Willebrand Disease (OMIM #193400 Type1, "
            "#613554 Type2, #277480 Type3); AD Type1/most Type2; AR Type3; most common hereditary "
            "bleeding disorder 1:100 (Type1); VWF = platelet adhesion (GPIbα) + FVIII carrier protein; "
            "DDAVP (desmopressin) effective in Type1/2A; ABSOLUTELY CI in Type2B — thrombocytopenia"
        ),
        "aa": "~2813 aa (multimer)",
        "kDa": "~250 kDa monomer (multimers up to 20,000 kDa)",
        "gene_class": "Multimeric glycoprotein; platelet adhesion (A1 domain – GPIbα); FVIII carrier (D3 domain); VWF propeptide (D1D2) directs multimerisation; collagen binding (A3 domain)",
        "locus": "12p13.31",
        "omim_gene": 613160,
        "omim_disease": 193400,
        "phenotype": "von Willebrand Disease — most common hereditary bleeding disorder; mucocutaneous bleeding (epistaxis, menorrhagia, gum bleeds); Type 3 = severe; secondary FVIII deficiency in all types",
        "disease": (
            "VWF encodes von Willebrand Factor, the largest multimeric glycoprotein in plasma, "
            "which serves two critical haemostatic functions: "
            "(1) PLATELET ADHESION BRIDGE: VWF A1 domain binds platelet GPIbα under high shear, "
            "mediating platelet tethering and rolling on subendothelial collagen (essential at high-flow "
            "arterial sites — arteries, microcirculation); A3 domain binds collagen directly. "
            "(2) FVIII CARRIER: D3 domain binds and protects FVIII from premature clearance — "
            "VWF LOF → secondary FVIII reduction (explains prolonged aPTT in severe VWD). "
            "CLASSIFICATION (5 types): "
            "Type 1 (75%): partial quantitative deficiency (VWF:Ag 30-50 IU/dL); AD; mild mucocutaneous; "
            "most respond to DDAVP (releases endothelial VWF stores). "
            "Type 2A: qualitative — loss of large VWF multimers; reduced platelet adhesion; "
            "RIPA (ristocetin-induced platelet agglutination) reduced; DDAVP VARIABLY useful. "
            "Type 2B: qualitative — GAIN-OF-FUNCTION (A1 domain) → spontaneous VWF-GPIbα binding → "
            "platelet clumping → thrombocytopenia; DDAVP ABSOLUTELY CONTRAINDICATED (releases "
            "mutant VWF → acute platelet clumping/thrombocytopenia — potentially fatal). "
            "Type 2M: qualitative — decreased platelet binding (A1 domain LOF) but large multimers present; "
            "similar to 2A clinically. "
            "Type 2N (Normandy): qualitative — D3 domain mutations → decreased FVIII binding → "
            "secondary FVIII reduction; mimics mild Hemophilia A (aPTT prolonged, low FVIII, "
            "normal VWF:Ag + RCo); VWF:FVIIIB assay diagnostic — CRITICAL to distinguish from HA. "
            "Type 3: severe quantitative (VWF:Ag <3 IU/dL); AR; near-absent VWF; mucocutaneous + "
            "haemarthroses (due to severe FVIII reduction <10%); VWF concentrate NOT DDAVP. "
            "DIAGNOSIS: VWF:Ag (quantitative), VWF:RCo/VWF:GPIbM (functional), FVIII:C, multimer analysis. "
            "DDAVP TRIAL essential for Type 1 management planning: 0.3 mcg/kg IV → check VWF at 1h; "
            "response = ≥3-fold rise + VWF >50 IU/dL. "
            "TREATMENT: DDAVP (Type 1/2A responders); VWF concentrate (Humate-P, Wilate) for Type 3, "
            "2B, non-responders; tranexamic acid for mucocutaneous bleeding."
        ),
        "inheritance": "AD Type1 (60-70% penetrance, variable expressivity); AD Type2A/2B/2M (dominant-negative or GOF); AR Type3 (compound heterozygous/homozygous); 2N may appear recessive (compound het with Type1 allele)",
        "hallmark": "Mucocutaneous bleeding (epistaxis, menorrhagia, gum bleeds), prolonged bleeding time, VWF:Ag reduced, VWF:RCo reduced, FVIII reduced (secondary); Type2B: thrombocytopenia (PATHOGNOMONIC) + DDAVP CI; Type3: severe haemarthroses",
        "key_ddx": "Hemophilia A (VWF:Ag NORMAL, no mucocutaneous features), Type2N (mimics mild HA — VWF:FVIIIB assay diagnostic), Platelet-type VWD (GPIbα GOF, not VWF GOF — phenotypically identical to Type2B), Acquired vWD (myeloproliferative neoplasms, LVAD, hypothyroidism — antibody mediated or shear-induced cleavage)",
        "treatment_alert": "DDAVP: Type1/2A — test with trial; ABSOLUTELY CI in Type2B (acute thrombocytopenia); VWF concentrate for Type3 and DDAVP failures; tranexamic acid adjunct; Lenacapavir VWF stabiliser (emerging); blood group O has 25% lower VWF (not VWD — lab awareness)",
        "seed": 1144,
        "cohort_n": 40,
        "severe_bleed_rate": 0.55,
        "type2b_rate": 0.12,             # Type 2B (DDAVP CI)
        "type3_rate": 0.08,              # Type 3 (severe AR)
        "ddavp_error_rate": 0.09,        # DDAVP given to Type2B error
        "joint_damage_rate": 0.10,
        "on_prophylaxis_rate": 0.40,
        "drug_error_rate": 0.09,
        "severity_weights": {"Mild": 0.60, "Moderate": 0.32, "Severe": 0.08},
        "primary_complication": "Mucocutaneous bleeding / DDAVP error in Type2B",
        "platelet_count": "Normal (Type1/2A/3); Thrombocytopenic (Type2B)",
        "pt_ptt": "aPTT prolonged (if FVIII <40%); PT normal; PFA-100 prolonged",
    },

    # ── F11 — Factor XI, Hemophilia C ─────────────────────────────────────────
    {
        "gene": "F11",
        "protein": "Factor XI (FXI)",
        "alias": (
            "F11; OMIM gene 264900; 4q35.2; ~625 aa; Factor XI Deficiency / Hemophilia C (OMIM #612416); "
            "AR; Ashkenazi Jewish founder variants E117X (exon 5) + F283L (exon 9) — prevalence 8% carriers AJ; "
            "CRITICAL: bleeding does NOT correlate with FXI level (opposite of FVIII/FIX); "
            "surgery/trauma-triggered bleeds; fibrinolysis-rich tissues (oropharynx, GU) bleed most"
        ),
        "aa": "~625 aa",
        "kDa": "~80 kDa (dimer in plasma)",
        "gene_class": "Contact/intrinsic pathway serine protease; activates FIX; amplification phase; cross-talk with TF pathway",
        "locus": "4q35.2",
        "omim_gene": 264900,
        "omim_disease": 612416,
        "phenotype": "Factor XI Deficiency (Hemophilia C) — AR; Ashkenazi Jewish founder; surgery/trauma-triggered bleeding; poor bleed-level correlation; fibrinolysis-rich site preference (tonsil, prostate, uterus)",
        "disease": (
            "F11 encodes Factor XI (FXI), a contact/intrinsic pathway serine protease that is "
            "activated by FXIIa (contact activation) and by thrombin (amplification loop — "
            "thrombin activates FXI on platelet surfaces, creating a positive feedback). "
            "NORMAL FUNCTION: FXIa activates FIX (intrinsic pathway), feeding into the intrinsic "
            "tenase complex → FX activation → thrombin generation. FXI also activates thrombin-activatable "
            "fibrinolysis inhibitor (TAFI) — critical in fibrinolysis-rich tissues. "
            "PATHOMECHANISM: LOF variants in F11 → reduced FXIa → reduced intrinsic tenase activation "
            "of FIX → impaired thrombin generation amplification. "
            "UNIQUE CLINICAL FEATURE — POOR BLEED-LEVEL CORRELATION: Unlike HA and HB, bleeding risk "
            "in FXI deficiency correlates POORLY with FXI level. Patients with severe deficiency (<10%) "
            "may have few/no bleeds; patients with moderate deficiency may bleed severely. "
            "Bleeding is TRAUMA/SURGERY-TRIGGERED (rarely spontaneous haemarthroses — key DDx from HA/HB). "
            "FIBRINOLYSIS-RICH TISSUE PREFERENCE: Oropharynx, dental, nasal, GU tract (prostate, "
            "uterus) bleed disproportionately because: these tissues have high tPA activity → "
            "FXI (via TAFI activation) normally counteracts local fibrinolysis; FXI deficiency → "
            "excessive local fibrinolysis → prolonged/severe bleeding in these sites. "
            "ASHKENAZI JEWISH FOUNDER MUTATIONS: E117X (exon 5) and F283L (exon 9) account for "
            "~99% of FXI deficiency in Ashkenazi Jews (AJ); combined carrier frequency 8% AJ population. "
            "E117X/F283L compound heterozygotes: moderate-severe FXI deficiency. "
            "LABORATORY: Prolonged aPTT, PT normal, platelet count normal, FVIII/FIX normal. "
            "Specific FXI activity assay (one-stage clot). "
            "TREATMENT: Fresh frozen plasma (FFP) for surgeries/major bleeding; FXI concentrate "
            "(available in UK/Europe — Hemoleven); tranexamic acid HIGHLY effective (blocks the "
            "fibrinolysis that drives bleeding in these patients) — adjunct for dental/nasal/GU bleeds; "
            "rFVIIa for inhibitor cases. Anti-FXI antisense oligonucleotides (abelacimab, IONIS-FXI) "
            "in trials as anticoagulants (reducing FXI → antithrombotic without excess bleeding)."
        ),
        "inheritance": "Autosomal Recessive (AR); homozygous or compound heterozygous; high carrier frequency in Ashkenazi Jews (8%); E117X + F283L founder variants; heterozygotes generally asymptomatic (FXI 50-70%)",
        "hallmark": "Prolonged aPTT with NORMAL PT, platelet count, and PT — FXI assay diagnostic; POOR bleed-level correlation (key distinguishing feature from HA/HB); surgery-triggered bleeding; fibrinolysis-rich tissues bleed most; Ashkenazi Jewish ancestry",
        "key_ddx": "Hemophilia A/B (X-linked, spontaneous haemarthroses, level correlates with bleed), FXII deficiency (prolonged aPTT but NO bleeding — FXII has no haemostatic role), Lupus anticoagulant (prolonged aPTT, mixing study shows inhibitor not correction), Heparin effect (aPTT prolonged, drug history)",
        "treatment_alert": "Tranexamic acid HIGHLY effective for dental/ENT/GU bleeds (blocks fibrinolysis which drives FXI deficiency bleeding); FFP for surgery; FXI concentrate (Hemoleven) in Europe; AVOID anti-fibrinolytics with haematuria (clot retention risk); rFVIIa for inhibitors; note: poor bleed/level correlation — manage by site/context not FXI level",
        "seed": 1145,
        "cohort_n": 40,
        "severe_bleed_rate": 0.35,        # surgery/trauma triggers
        "inhibitor_rate": 0.03,
        "fibrinolysis_site_bleed_rate": 0.55,   # oropharynx/GU/nasal bleeds
        "ashkenazi_rate": 0.70,           # proportion of AJ ancestry in cohort
        "on_prophylaxis_rate": 0.20,
        "drug_error_rate": 0.08,
        "severity_weights": {"Severe": 0.20, "Moderate": 0.45, "Mild": 0.35},
        "primary_complication": "Post-surgical/dental bleeding at fibrinolysis-rich sites",
        "platelet_count": "Normal",
        "pt_ptt": "aPTT prolonged; PT normal; FXI level POOR correlate of bleed risk",
    },

    # ── F7 — Factor VII Deficiency ─────────────────────────────────────────────
    {
        "gene": "F7",
        "protein": "Factor VII (FVII)",
        "alias": (
            "F7; OMIM gene 227500; 13q34; ~466 aa; Factor VII Deficiency (OMIM #227500); "
            "AR; most common rare coagulation factor deficiency 1:500000; "
            "extrinsic/TF-initiated pathway serine protease; "
            "rFVIIa (NovoSeven) / plasma-derived FVII concentrate treatment; "
            "PT markedly prolonged with NORMAL aPTT — unique laboratory signature"
        ),
        "aa": "~466 aa",
        "kDa": "~50 kDa",
        "gene_class": "Vitamin K-dependent serine protease; TF-FVIIa complex; extrinsic coagulation pathway initiator",
        "locus": "13q34",
        "omim_gene": 227500,
        "omim_disease": 227500,
        "phenotype": "Factor VII Deficiency — most common rare coagulation factor deficiency; AR; PT prolonged with NORMAL aPTT PATHOGNOMONIC; mucocutaneous + haemarthroses in severe; heterogeneous phenotype",
        "disease": (
            "F7 encodes Factor VII (FVII), a Vitamin K-dependent serine protease that circulates "
            "as a zymogen and is activated by Tissue Factor (TF, exposed at vessel injury sites) → "
            "TF-FVIIa complex — the PRIMARY INITIATOR of coagulation in vivo. "
            "NORMAL FUNCTION: TF-FVIIa complex activates FX (→ prothrombinase → thrombin) and FIX "
            "(→ intrinsic tenase → FX → thrombin). FVII is the ONLY factor exclusively in the "
            "extrinsic pathway; it is the gatekeeper of coagulation initiation. "
            "PATHOMECHANISM: LOF variants in F7 (missense, nonsense, frameshift) → reduced TF-FVIIa → "
            "impaired coagulation initiation → bleeding (haemostatic plug fails to generate adequate thrombin). "
            "LABORATORY HALLMARK: Isolated prolonged PT + NORMAL aPTT — PATHOGNOMONIC for FVII deficiency "
            "(or extrinsic pathway defects only, e.g. VKD factors, DIC if early). "
            "Most coagulation disorders prolong aPTT; FVII deficiency prolongs ONLY PT. "
            "aPTT normal because FIX, FXI, FXII, FVIII (intrinsic), FXII → all intact. "
            "CLINICAL: Heterogeneous — does NOT correlate with FVII level as expected; "
            "severe deficiency (<1%) can have mild bleeding; moderate deficiency may have ICH. "
            "MANAGEMENT: rFVIIa (NovoSeven) — short half-life (2-4h) → frequent dosing; "
            "plasma-derived FVII concentrate (Coagil-VII, ProSeven); FFP as alternative. "
            "PROPHYLAXIS: rFVIIa or FVII concentrate for severe deficiency. "
            "VITAMIN K: low doses help if dietary/drug-related; not useful for genetic FVII deficiency. "
            "PERIOPERATIVE: FVII level >30% adequate for most surgeries; >50% for neurosurgery. "
            "THROMBOSIS RISK: rFVIIa carries thrombosis risk (TF-FVIIa activates coagulation systemically "
            "at supraphysiological doses) — use lowest effective dose."
        ),
        "inheritance": "Autosomal Recessive (AR); compound heterozygous or homozygous; heterozygotes: FVII 40-60% (generally asymptomatic); most common rare factor deficiency globally",
        "hallmark": "Isolated prolonged PT + NORMAL aPTT PATHOGNOMONIC; FVII-specific assay confirms; heterogeneous phenotype (no strict level-bleed correlation); mucocutaneous + haemarthroses severe; ICH risk in severe",
        "key_ddx": "Vitamin K deficiency (ALL VKD factors low: II, VII, IX, X — also PT prolonged; dietary/drug/malabsorption), Warfarin effect (VKD factors all low + drug history), Early DIC (PT prolonged first; check fibrinogen/d-dimer), Liver disease (multiple factor synthesis impaired)",
        "treatment_alert": "rFVIIa (NovoSeven) — short T½ 2-4h, frequent dosing; FVII concentrate (pd-FVII) preferred if available; target FVII >30% for minor surgery, >50% neurosurgery/severe bleed; FFP alternative; AVOID over-dosing rFVIIa (thrombosis risk); Vitamin K NOT effective for genetic FVII deficiency",
        "seed": 1146,
        "cohort_n": 40,
        "severe_bleed_rate": 0.45,
        "icb_rate": 0.08,
        "on_prophylaxis_rate": 0.45,
        "drug_error_rate": 0.07,
        "severity_weights": {"Severe": 0.30, "Moderate": 0.35, "Mild": 0.35},
        "primary_complication": "ICH in severe / PT-isolated prolongation missed",
        "platelet_count": "Normal",
        "pt_ptt": "PT markedly prolonged; aPTT NORMAL — PATHOGNOMONIC",
    },

    # ── F13A1 — Factor XIII A-Subunit Deficiency ───────────────────────────────
    {
        "gene": "F13A1",
        "protein": "Factor XIII A-subunit (FXIII-A)",
        "alias": (
            "F13A1; OMIM gene 134570; 6p25.1; ~732 aa; Factor XIII Deficiency (OMIM #613235); "
            "AR; rarest autosomal recessive bleeding disorder 1:2,000,000; "
            "fibrin cross-linking transglutaminase; PATHOGNOMONIC: delayed bleeding 24-48h post-injury "
            "(clot forms but dissolves — fibrin not cross-linked); PT/aPTT/TT ALL NORMAL — "
            "standard screening tests COMPLETELY MISS FXIII deficiency; urea clot solubility test"
        ),
        "aa": "~732 aa (A-subunit; FXIII = A₂B₂ tetramer in plasma)",
        "kDa": "~83 kDa (A-subunit)",
        "gene_class": "Transglutaminase; fibrin cross-linking enzyme; cross-links fibrin chains (α-chain Lys-Gln bonds + γ-dimer bonds → fibrin polymer stabilisation); activates TAFI; crosslinks α2-antiplasmin to fibrin",
        "locus": "6p25.1",
        "omim_gene": 134570,
        "omim_disease": 613235,
        "phenotype": "Factor XIII Deficiency — rarest severe hereditary bleeding disorder; AR; normal PT/aPTT/TT (SCREENING TESTS MISS); delayed 24-48h post-injury bleeding PATHOGNOMONIC; umbilical stump bleeding; intracranial haemorrhage; recurrent miscarriage",
        "disease": (
            "F13A1 encodes the catalytic A-subunit of Factor XIII (FXIII), a plasma transglutaminase "
            "that circulates as a heterotetramer (A₂B₂) and is activated by thrombin (+Ca²⁺) → FXIIIa. "
            "NORMAL FUNCTION: FXIIIa cross-links fibrin chains via isopeptide bonds (γ-glutamyl-ε-lysine): "
            "(1) γ-chain dimer bonds: two γ-chains cross-linked → γ-γ dimer (20-fold faster); "
            "(2) α-chain polymer bonds: α-chain cross-links → fibrin polymer; "
            "(3) Cross-links α2-antiplasmin to fibrin → fibrin clot protected from plasmin dissolution; "
            "(4) Cross-links TAFI (thrombin-activatable fibrinolysis inhibitor) → reduced fibrinolysis. "
            "WITHOUT FXIIIa: fibrin clot forms NORMALLY (all other factors intact) but remains "
            "loosely polymerised and dissolves rapidly under fibrinolysis → clot disintegration. "
            "PATHOMECHANISM: FXIII-A LOF → FXIIIa absent → fibrin not cross-linked → clot forms "
            "initially (haemostasis appears to occur) but dissolves 24-48 hours later — DELAYED BLEEDING. "
            "LABORATORY HALLMARK: PT/aPTT/TT/fibrinogen ALL COMPLETELY NORMAL — standard clot-based "
            "coagulation tests assess fibrin formation, NOT stabilisation → FXIII deficiency is "
            "INVISIBLE to routine screening. "
            "Diagnosis requires SPECIFIC tests: (1) Urea clot solubility test (5M urea): normal clot "
            "resists 24h dissolution; FXIII-deficient clot dissolves within 2h (>95% sensitivity); "
            "(2) Quantitative FXIII antigen/activity assay (FXIII:A immunoassay or ammonia-release "
            "activity assay) — definitive. "
            "CLINICAL HALLMARKS: "
            "(1) UMBILICAL STUMP BLEEDING: delayed separation + oozing — the most characteristic "
            "neonatal presentation; present in 80% of FXIII deficiency; virtually PATHOGNOMONIC "
            "(uncommon in any other bleeding disorder). "
            "(2) DELAYED BLEEDING: post-injury/post-surgical bleeding onset 24-48h (clot initially "
            "forms → dissolves) — clinically misleading. "
            "(3) INTRACRANIAL HAEMORRHAGE (ICH): spontaneous ICH in ~25% — leading cause of death/disability; "
            "occurs even with minimal trauma. "
            "(4) RECURRENT MISCARRIAGE (females): FXIII required for fibrin-fibronectin matrix "
            "in placenta implantation → recurrent early pregnancy loss. "
            "(5) Wound dehiscence: poor wound healing (impaired fibrin scaffold). "
            "TREATMENT: FXIII concentrate (Corifact/Fibrogammin) prophylaxis every 4-6 weeks "
            "(FXIII half-life ~11 days) — prevents ICH; target FXIII activity >10%. "
            "FFP or cryoprecipitate if concentrate unavailable."
        ),
        "inheritance": "Autosomal Recessive (AR); FXIII-A subunit (most common, F13A1); FXIII-B subunit (F13B, carries A-subunit in plasma) rarely; 1:2,000,000 general population; higher in consanguineous populations",
        "hallmark": "Normal PT/aPTT/TT (screening misses completely); urea clot solubility diagnostic; delayed 24-48h bleeding PATHOGNOMONIC; umbilical stump bleeding (80%); ICH 25% spontaneous; recurrent miscarriage females; FXIII activity <1% severe",
        "key_ddx": "Dysfibrinogenaemia (abnormal fibrin polymerisation — TT prolonged, reptilase prolonged; fibrinogen level may be normal but dysfunctional), alpha2-antiplasmin deficiency (fibrinolysis excess, clot dissolves rapidly, FXIII assay normal), PAI-1 deficiency (fibrinolysis excess, similar delayed bleeding), Ehlers-Danlos (connective tissue not coagulation)",
        "treatment_alert": "FXIII concentrate (Corifact/Fibrogammin) prophylaxis MANDATORY every 4-6 weeks for all severe FXIII deficiency — prevents spontaneous ICH; target FXIII >10%; standard coagulation tests (PT/aPTT/TT) ALWAYS NORMAL — never rule out FXIII deficiency by normal routine tests; urea clot solubility test if clinical suspicion; umbilical stump bleeding = FXIII deficiency until proven otherwise",
        "seed": 1147,
        "cohort_n": 40,
        "severe_bleed_rate": 0.80,
        "icb_rate": 0.25,
        "umbilical_bleed_rate": 0.80,
        "miscarriage_rate": 0.60,         # in females with FXIII deficiency
        "on_prophylaxis_rate": 0.70,
        "drug_error_rate": 0.15,          # missed by normal coagulation tests
        "severity_weights": {"Severe": 0.80, "Moderate": 0.15, "Mild": 0.05},
        "primary_complication": "ICH / missed by routine coag tests",
        "platelet_count": "Normal",
        "pt_ptt": "PT NORMAL; aPTT NORMAL; TT NORMAL — all routine tests MISS; urea clot solubility POSITIVE",
    },

    # ── ITGA2B — Glanzmann Thrombasthenia ──────────────────────────────────────
    {
        "gene": "ITGA2B",
        "protein": "Integrin αIIb (GpIIb, CD41)",
        "alias": (
            "ITGA2B; OMIM gene 607759; 17q21.31; ~1039 aa; Glanzmann Thrombasthenia (OMIM #273800); "
            "AR; platelet-type bleeding disorder; absent platelet aggregation (ADP/collagen/AA/thrombin); "
            "NORMAL platelet count + morphology; absent αIIbβ3 → no fibrinogen receptor → no aggregation; "
            "ITGB3 (β3-subunit) mutations cause identical phenotype (GT Type II/III)"
        ),
        "aa": "~1039 aa (αIIb); forms heterodimer with β3 (ITGB3, ~788 aa) → αIIbβ3 (GpIIb-IIIa) complex",
        "kDa": "~140 kDa",
        "gene_class": "Integrin alpha subunit; αIIbβ3 (GpIIb-IIIa) fibrinogen receptor; platelet aggregation; clot retraction; outside-in signalling",
        "locus": "17q21.31",
        "omim_gene": 607759,
        "omim_disease": 273800,
        "phenotype": "Glanzmann Thrombasthenia — AR platelet function disorder; absent platelet aggregation with ALL agonists; NORMAL platelet count/morphology; mucocutaneous bleeding; recurrent epistaxis; menorrhagia",
        "disease": (
            "ITGA2B encodes Integrin αIIb (GpIIb), which forms an obligate heterodimer with β3 "
            "(encoded by ITGB3) → the αIIbβ3 complex (GpIIb-IIIa). "
            "NORMAL FUNCTION: αIIbβ3 is the most abundant platelet surface glycoprotein (~80,000 copies/platelet). "
            "Platelet activation (by ADP, thrombin, collagen, TXA2) → inside-out signalling → "
            "αIIbβ3 undergoes conformational change from low- to high-affinity state → binds fibrinogen "
            "(and VWF at high shear) → fibrinogen bridges adjacent platelets → PLATELET AGGREGATION. "
            "αIIbβ3 also mediates clot retraction (tightening the fibrin/platelet clot) via "
            "cytoskeletal connections (actin/myosin). "
            "PATHOMECHANISM: LOF variants in ITGA2B (60%) or ITGB3 (40%) → absent or non-functional "
            "αIIbβ3 → platelet activation occurs normally but FIBRINOGEN BRIDGE FORMATION FAILS → "
            "absent platelet aggregation (ALL agonists: ADP, collagen, arachidonic acid, thrombin, "
            "epinephrine) — complete aggregation failure. "
            "PLATELET COUNT: NORMAL (thrombocytopenia is ABSENT in Glanzmann — key DDx from BSS). "
            "PLATELET MORPHOLOGY: NORMAL by light microscopy. "
            "RISTOCETIN AGGLUTINATION: NORMAL (ristocetin-VWF-GPIbα axis intact in GT — key DDx from BSS). "
            "FLOW CYTOMETRY: absent CD41 (αIIb) and/or CD61 (β3) on platelets — diagnostic. "
            "LABORATORY: Prolonged PFA-100, markedly prolonged bleeding time; aPTT/PT NORMAL. "
            "CLASSIFICATION: Type I (<5% αIIbβ3, most severe), Type II (5-20%), Type III (variant, "
            "normal quantity but dysfunctional). "
            "CLINICAL: severe mucocutaneous bleeding (epistaxis recurrent, gum bleeds, menorrhagia "
            "severe, GI bleeding), haematuria; haemarthroses RARE (not a feature of platelet disorders). "
            "TREATMENT: Platelet transfusion (definitive for acute bleeding); risk of alloimmunisation "
            "to αIIbβ3 → refractoriness to transfusion; rFVIIa (NovoSeven) effective — bypasses "
            "need for αIIbβ3 by generating direct fibrin clot at injury site; "
            "HSCT curative (rare cases, typically children with refractory disease); "
            "AVOIDANCE: NSAIDs/aspirin (inhibit residual platelet function), G2bIIIa inhibitors "
            "(tirofiban, eptifibatide, abciximab — ABSOLUTELY CONTRAINDICATED in GT)."
        ),
        "inheritance": "Autosomal Recessive (AR); ITGA2B (αIIb, 60%) or ITGB3 (β3, 40%); both genes on 17q21; consanguinity enriched; Israeli/Arab/Romany/South Indian founder variants (specific ethnic founder mutations)",
        "hallmark": "Absent platelet aggregation to ALL agonists (ADP/collagen/AA/thrombin); NORMAL platelet count; NORMAL ristocetin agglutination (GPIbα intact); absent CD41/CD61 flow cytometry; prolonged PFA-100/bleeding time; NO haemarthroses",
        "key_ddx": "Bernard-Soulier syndrome (BSS: large platelets + thrombocytopenia + absent ristocetin — GPIbα absent; GT has NORMAL ristocetin), Storage pool disease (platelet aggregation reduced but NOT absent — agonist-selective), Aspirin effect (AA response absent but ADP/collagen intact), Uraemic platelet dysfunction (acquired; secondary to uraemia)",
        "treatment_alert": "Platelet transfusion for acute bleeds (avoid alloimmunisation — use HLA-matched/leukoreduced when possible); rFVIIa (NovoSeven) 90 mcg/kg effective alternative; ABSOLUTELY CONTRAINDICATE: NSAIDs/aspirin (reduce residual platelet function) AND GP IIb/IIIa inhibitors (eptifibatide, tirofiban, abciximab — catastrophic in GT); HSCT curative option for refractory severe cases; menorrhagia — OCP + tranexamic acid + platelet support peri-menstrually",
        "seed": 1148,
        "cohort_n": 40,
        "severe_bleed_rate": 0.70,
        "alloimmunisation_rate": 0.30,    # anti-αIIbβ3 after transfusion
        "epistaxis_rate": 0.85,
        "menorrhagia_rate": 0.75,         # in females
        "drug_error_rate": 0.10,          # NSAID/GP2b3a given
        "on_prophylaxis_rate": 0.30,
        "severity_weights": {"Severe": 0.50, "Moderate": 0.35, "Mild": 0.15},
        "primary_complication": "Alloimmunisation / recurrent epistaxis",
        "platelet_count": "Normal",
        "pt_ptt": "PT normal; aPTT normal; PFA-100 prolonged; aggregation absent ALL agonists",
    },

    # ── GP1BA — Bernard-Soulier Syndrome ──────────────────────────────────────
    {
        "gene": "GP1BA",
        "protein": "Glycoprotein Ibα (GPIbα, CD42b)",
        "alias": (
            "GP1BA; OMIM gene 606672; 17p13.2; ~626 aa; Bernard-Soulier Syndrome (OMIM #231200); "
            "AR (rarely AD with milder phenotype); giant platelets + thrombocytopenia PATHOGNOMONIC; "
            "absent ristocetin-induced platelet agglutination (RIPA); GPIbα = VWF receptor under shear; "
            "GP1BB (GPIbβ) + GP9 + GP5 complete the GpIb-IX-V complex — all can cause BSS"
        ),
        "aa": "~626 aa (GPIbα); GpIb-IX-V complex: GPIbα (GP1BA) + GPIbβ (GP1BB) + GPIX (GP9) + GPV (GP5)",
        "kDa": "~135 kDa (extracellular shedding produces soluble GPIbα = VWF:RCo binding site)",
        "gene_class": "Platelet surface glycoprotein; GpIb-IX-V complex; VWF A1 domain receptor; platelet tethering under high shear; thrombin receptor (high-affinity); cytoskeletal anchor (filamin A)",
        "locus": "17p13.2",
        "omim_gene": 606672,
        "omim_disease": 231200,
        "phenotype": "Bernard-Soulier Syndrome — AR; GIANT PLATELETS + THROMBOCYTOPENIA PATHOGNOMONIC; absent ristocetin platelet agglutination; absent GpIb-IX-V complex; mucocutaneous bleeding; high transfusion refractoriness risk",
        "disease": (
            "GP1BA encodes Glycoprotein Ibα (GPIbα), the ligand-binding α-subunit of the "
            "GpIb-IX-V complex (GPIbα + GPIbβ + GPIX + GPV), the major platelet adhesion receptor. "
            "NORMAL FUNCTION: Under high shear (arteries, microcirculation), GPIbα A1 domain "
            "binds VWF A1 domain (exposed after VWF unfolds under shear) → platelet tethering and "
            "rolling on subendothelium (GPIbα-VWF interaction is the FIRST step of platelet adhesion). "
            "GPIbα is also a HIGH-AFFINITY THROMBIN RECEPTOR (thrombin exosite I binds GPIbα LRR "
            "domain — contributes to thrombin-induced platelet activation). "
            "GpIb-IX-V complex anchors platelet cytoskeleton via filamin A — essential for maintaining "
            "platelet discoid shape; absent GpIb → platelet membrane destabilisation → GIANT PLATELETS. "
            "PATHOMECHANISM: LOF variants in GP1BA (most common), GP1BB, GP9, or GP5 → absent or "
            "dysfunctional GpIb-IX-V complex → "
            "(1) Absent platelet adhesion to subendothelium (under shear) → bleeding. "
            "(2) Abnormal platelet production: megakaryocytes without GpIb cannot bud off normal "
            "platelets → platelet membrane instability → giant granular platelets (approaching lymphocyte "
            "size, 5-10 μm vs normal 2-3 μm) released; premature platelet destruction → THROMBOCYTOPENIA. "
            "LABORATORY HALLMARKS: "
            "(1) GIANT PLATELETS on blood film — approach lymphocyte size; may be under-counted "
            "by automated analysers (counted as WBCs or not detected) → ALWAYS manually count "
            "platelets in BSS (automated count UNRELIABLE). "
            "(2) ABSENT RISTOCETIN AGGLUTINATION (RIPA): ristocetin induces VWF-GPIbα binding; "
            "absent GPIbα → absent RIPA even with added normal VWF; PATHOGNOMONIC. "
            "Contrast with Glanzmann: RIPA NORMAL in GT (GPIbα intact). "
            "(3) Platelet aggregation: ADP/collagen/AA responses INTACT (αIIbβ3 normal in BSS) — "
            "key DDx from Glanzmann (ALL aggregation absent in GT). "
            "(4) Flow cytometry: absent CD42b (GPIbα) — diagnostic. "
            "TREATMENT: Platelet transfusion (main therapy) — high alloimmunisation risk (anti-GPIb); "
            "rFVIIa effective; DDAVP rarely helpful (GPIbα absent — VWF release ineffective); "
            "antifibrinolytics adjunct; HSCT curative option (rare cases). "
            "PLATELET-TYPE VWD: GPIbα GOF mutations (not LOF) → spontaneous GPIbα-VWF binding → "
            "similar to VWD Type2B (thrombocytopenia + absent large VWF multimers) — distinguished "
            "by mixing studies and VWF gene sequencing."
        ),
        "inheritance": "Autosomal Recessive (AR) typical; rarely AD with haploinsufficiency (milder); GP1BA, GP1BB, GP9, or GP5 mutations (same complex — compound molecular heterogeneity); consanguinity enriched; Italian/Arab/French founder variants",
        "hallmark": "GIANT PLATELETS (approaching lymphocyte size, >3.6 μm mean platelet diameter) + THROMBOCYTOPENIA PATHOGNOMONIC; absent ristocetin agglutination (RIPA); INTACT ADP/collagen aggregation (DDx Glanzmann); absent CD42b flow cytometry; automated platelet count UNRELIABLE (manual count mandatory)",
        "key_ddx": "Glanzmann Thrombasthenia (normal platelet count/morphology, absent ALL aggregation, NORMAL ristocetin — GPIbα intact), Platelet-type VWD (GPIbα GOF — similar phenotype, treat with VWF concentrate NOT platelet transfusion), MYH9-related disorders (giant platelets + thrombocytopenia + deafness + cataracts + nephritis — MYH9 mutation, autosomal dominant), Grey platelet syndrome (large agranular platelets, different morphology)",
        "treatment_alert": "Platelet transfusion (main therapy for bleeding episodes); use HLA-matched/anti-GPIb-negative donors when possible (alloimmunisation common); rFVIIa (NovoSeven) 90 mcg/kg effective; DDAVP generally NOT helpful (VWF release has no effect without GPIbα); antifibrinolytics (tranexamic acid) adjunct; manual platelet count MANDATORY (automated analyser counts giant platelets as leukocytes → falsely low count); HSCT curative for severe cases; menorrhagia: OCP + tranexamic + platelet support",
        "seed": 1149,
        "cohort_n": 40,
        "severe_bleed_rate": 0.65,
        "alloimmunisation_rate": 0.35,
        "giant_platelet_rate": 1.00,        # diagnostic feature
        "thrombocytopenia_rate": 0.90,
        "drug_error_rate": 0.08,
        "on_prophylaxis_rate": 0.25,
        "severity_weights": {"Severe": 0.45, "Moderate": 0.40, "Mild": 0.15},
        "primary_complication": "Alloimmunisation to platelet transfusions / missed by automated platelet count",
        "platelet_count": "Thrombocytopenic + giant platelets PATHOGNOMONIC",
        "pt_ptt": "PT normal; aPTT normal; PFA-100 prolonged; RIPA absent; aggregation to ADP/collagen INTACT",
    },
]


def _gen_patients_for_gene(gene_data: dict, seed: int) -> list:
    """Generate 40 deterministic synthetic patients for one bleeding disorder gene."""
    rng = random.Random(seed)
    patients = []
    for i in range(gene_data["cohort_n"]):
        sev_choices = list(gene_data["severity_weights"].keys())
        sev_weights = list(gene_data["severity_weights"].values())
        sev = rng.choices(sev_choices, weights=sev_weights, k=1)[0]

        sex = rng.choice(["M", "F"])

        # Core clinical features
        severe_bleed = rng.random() < gene_data["severe_bleed_rate"]
        inhibitor = rng.random() < gene_data.get("inhibitor_rate", 0.0)
        icb = rng.random() < gene_data.get("icb_rate", 0.0)
        on_prophylaxis = rng.random() < gene_data["on_prophylaxis_rate"]
        drug_error = rng.random() < gene_data["drug_error_rate"]
        age_at_dx_years = round(rng.uniform(0.1, 45.0), 1)
        surveillance_adherent = rng.random() < 0.70

        # Gene-specific features
        type2b_error = False
        if gene_data["gene"] == "VWF":
            type2b = rng.random() < gene_data.get("type2b_rate", 0.0)
            if type2b and drug_error:
                type2b_error = True   # DDAVP given to Type2B patient (error)

        umbilical_bleed = False
        miscarriage = False
        if gene_data["gene"] == "F13A1":
            umbilical_bleed = rng.random() < gene_data.get("umbilical_bleed_rate", 0.0)
            if sex == "F":
                miscarriage = rng.random() < gene_data.get("miscarriage_rate", 0.0)

        alloimmunised = False
        giant_platelets = False
        if gene_data["gene"] in ("ITGA2B", "GP1BA"):
            alloimmunised = rng.random() < gene_data.get("alloimmunisation_rate", 0.0)
        if gene_data["gene"] == "GP1BA":
            giant_platelets = rng.random() < gene_data.get("giant_platelet_rate", 1.0)

        ashkenazi = False
        if gene_data["gene"] == "F11":
            ashkenazi = rng.random() < gene_data.get("ashkenazi_rate", 0.0)

        patients.append({
            "patient_id": f"{gene_data['gene']}-{i+1:03d}",
            "gene": gene_data["gene"],
            "seed": seed,
            "severity": sev,
            "sex": sex,
            "severe_bleed": severe_bleed,
            "inhibitor": inhibitor,
            "icb": icb,
            "on_prophylaxis": on_prophylaxis,
            "drug_error": drug_error,
            "type2b_ddavp_error": type2b_error,
            "umbilical_bleed": umbilical_bleed,
            "miscarriage": miscarriage,
            "alloimmunised": alloimmunised,
            "giant_platelets": giant_platelets,
            "ashkenazi_ancestry": ashkenazi,
            "age_at_dx_years": age_at_dx_years,
            "surveillance_adherent": surveillance_adherent,
            "platelet_count": gene_data["platelet_count"],
            "pt_ptt": gene_data["pt_ptt"],
            "primary_complication": gene_data["primary_complication"],
        })
    return patients


def _gen_cohort() -> list:
    """Generate all 320 patients (8 genes × 40) deterministically."""
    all_pts = []
    for idx, gd in enumerate(COAGULOPATHY_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients_for_gene(gd, seed))
    return all_pts


# ── API functions ──────────────────────────────────────────────────────────────

def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] = sev.get(p["severity"], 0) + 1

    severe_bleed_n = sum(1 for p in patients if p["severe_bleed"])
    inhibitor_n = sum(1 for p in patients if p["inhibitor"])
    icb_n = sum(1 for p in patients if p["icb"])
    prophylaxis_n = sum(1 for p in patients if p["on_prophylaxis"])
    drug_error_n = sum(1 for p in patients if p["drug_error"])
    alloimmunised_n = sum(1 for p in patients if p["alloimmunised"])
    type2b_err_n = sum(1 for p in patients if p["type2b_ddavp_error"])
    umbilical_n = sum(1 for p in patients if p["umbilical_bleed"])
    miscarriage_n = sum(1 for p in patients if p["miscarriage"])
    giant_platelet_n = sum(1 for p in patients if p["giant_platelets"])
    ashkenazi_n = sum(1 for p in patients if p["ashkenazi_ancestry"])
    surveillance_n = sum(1 for p in patients if p["surveillance_adherent"])

    gene_stats = {}
    for gd in COAGULOPATHY_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_stats[gd["gene"]] = {
            "severe_bleed_pct": round(100 * sum(1 for p in gpts if p["severe_bleed"]) / len(gpts), 1),
            "inhibitor_pct": round(100 * sum(1 for p in gpts if p["inhibitor"]) / len(gpts), 1),
            "icb_pct": round(100 * sum(1 for p in gpts if p["icb"]) / len(gpts), 1),
            "drug_error_pct": round(100 * sum(1 for p in gpts if p["drug_error"]) / len(gpts), 1),
        }

    disease_cat = {
        "Hemophilia A (F8)": round(100 * 40 / n, 1),
        "Hemophilia B (F9)": round(100 * 40 / n, 1),
        "von Willebrand Disease (VWF)": round(100 * 40 / n, 1),
        "Factor XI Deficiency / Hemophilia C (F11)": round(100 * 40 / n, 1),
        "Factor VII Deficiency (F7)": round(100 * 40 / n, 1),
        "Factor XIII Deficiency (F13A1)": round(100 * 40 / n, 1),
        "Glanzmann Thrombasthenia (ITGA2B)": round(100 * 40 / n, 1),
        "Bernard-Soulier Syndrome (GP1BA)": round(100 * 40 / n, 1),
    }

    kpis = [
        {"label": "Total Patients", "value": str(n)},
        {"label": "Genes Covered", "value": "8"},
        {"label": "Severe Bleed", "value": f"{round(100*severe_bleed_n/n,1)}%"},
        {"label": "Inhibitor", "value": f"{round(100*inhibitor_n/n,1)}%"},
        {"label": "ICH", "value": f"{round(100*icb_n/n,1)}%"},
        {"label": "On Prophylaxis", "value": f"{round(100*prophylaxis_n/n,1)}%"},
        {"label": "Drug Error", "value": f"{round(100*drug_error_n/n,1)}%"},
        {"label": "Alloimmunised", "value": f"{round(100*alloimmunised_n/n,1)}%"},
    ]

    return {
        "atlas_name": "Coagulopathy-Atlas",
        "atlas_subtitle": (
            "F8·F9·VWF·F11·F7·F13A1·ITGA2B·GP1BA — "
            "320 patients (8×40, seeds 1142–1149)"
        ),
        "n_genes": 8,
        "n_patients": n,
        "seeds": "1142–1149",
        "description": (
            "Comprehensive atlas of 8 major hereditary bleeding disorders: "
            "F8/Hemophilia A (XLR; FVIII <1% severe; haemarthroses; inhibitors 25-30%; "
            "emicizumab FDA 2017 — bispecific bridges FIXa+FX regardless of inhibitor status; "
            "fitusiran FDA 2024); "
            "F9/Hemophilia B (XLR; FIX <1% severe; clinically IDENTICAL to HA — factor assay mandatory; "
            "etranacogene dezaparvovec FDA 2022 gene therapy — FIX-Padua 8× activity; Leyden: severe→mild at puberty); "
            "VWF/vWD (most common hereditary bleeding; Type1 AD partial quantitative; "
            "DDAVP effective Type1/2A; ABSOLUTELY CI in Type2B — thrombocytopenia; "
            "Type3 AR severe — VWF concentrate not DDAVP); "
            "F11/Hemophilia C (AR; Ashkenazi founder E117X/F283L 8%; "
            "poor bleed-level correlation UNIQUE; fibrinolysis-rich sites bleed most; "
            "tranexamic acid highly effective); "
            "F7/Factor VII deficiency (AR; most common rare factor deficiency 1:500000; "
            "isolated prolonged PT + NORMAL aPTT PATHOGNOMONIC; rFVIIa treatment); "
            "F13A1/FXIII deficiency (AR; rarest 1:2M; PT/aPTT/TT ALL NORMAL — standard tests miss completely; "
            "urea clot solubility diagnostic; delayed 24-48h bleeding PATHOGNOMONIC; "
            "umbilical stump bleeding 80%; ICH 25%; FXIII concentrate prophylaxis mandatory); "
            "ITGA2B/Glanzmann thrombasthenia (AR; absent platelet aggregation ALL agonists; "
            "NORMAL platelet count and ristocetin — key DDx from BSS; absent CD41 flow; "
            "rFVIIa effective; NSAID/GP2b3a inhibitors absolutely CI); "
            "GP1BA/Bernard-Soulier (AR; GIANT PLATELETS + thrombocytopenia PATHOGNOMONIC; "
            "absent ristocetin agglutination; intact ADP/collagen aggregation — key DDx from Glanzmann; "
            "automated platelet count UNRELIABLE — manual mandatory)."
        ),
        "aggregate_clinical": {
            "severe_bleed_pct": round(100 * severe_bleed_n / n, 1),
            "inhibitor_pct": round(100 * inhibitor_n / n, 1),
            "icb_pct": round(100 * icb_n / n, 1),
            "on_prophylaxis_pct": round(100 * prophylaxis_n / n, 1),
            "drug_error_pct": round(100 * drug_error_n / n, 1),
            "alloimmunised_pct": round(100 * alloimmunised_n / n, 1),
            "type2b_ddavp_error_pct": round(100 * type2b_err_n / n, 1),
            "umbilical_bleed_pct": round(100 * umbilical_n / n, 1),
            "miscarriage_pct": round(100 * miscarriage_n / n, 1),
            "giant_platelet_pct": round(100 * giant_platelet_n / n, 1),
            "ashkenazi_pct": round(100 * ashkenazi_n / n, 1),
            "surveillance_adherent_pct": round(100 * surveillance_n / n, 1),
        },
        "drug_alerts": [
            {
                "type": "danger",
                "title": "VWF TYPE 2B: DDAVP (DESMOPRESSIN) ABSOLUTELY CONTRAINDICATED",
                "body": (
                    "In Type 2B vWD, the VWF A1 domain GOF mutation causes spontaneous high-affinity "
                    "binding of mutant VWF to platelet GPIbα even without shear stress, leading to "
                    "constitutive platelet clumping and thrombocytopenia at baseline. "
                    "DDAVP releases endothelial stores of mutant VWF in large amounts → acute massive "
                    "platelet clumping → acute severe thrombocytopenia (platelet count can fall to "
                    "<10,000/µL within hours) → paradoxical WORSENING of bleeding and risk of "
                    "thrombotic microangiopathy. DDAVP is the standard treatment for Type 1 vWD and "
                    "must NEVER be given in Type 2B without subtype determination. "
                    "Correct treatment for Type 2B: VWF concentrate (containing both VWF and FVIII — "
                    "Humate-P, Wilate) for bleeding; platelet transfusion if severe thrombocytopenia. "
                    "TYPE DETERMINATION BEFORE TREATMENT IS MANDATORY: VWF:Ag, VWF:RCo, FVIII:C, "
                    "multimer analysis, and VWF gene sequencing for subtyping."
                ),
            },
            {
                "type": "danger",
                "title": "F13A1 (FXIII DEFICIENCY): PT/aPTT/TT ALWAYS NORMAL — SCREENING TESTS MISS COMPLETELY",
                "body": (
                    "Factor XIII deficiency is the only severe hereditary bleeding disorder where ALL "
                    "standard coagulation screening tests (PT, aPTT, thrombin time, fibrinogen level) "
                    "are COMPLETELY NORMAL. This is because FXIII acts AFTER clot formation (cross-links "
                    "fibrin) — tests that measure clotting time will not detect the deficiency. "
                    "A normal coagulation screen DOES NOT exclude FXIII deficiency. "
                    "SUSPECT FXIII DEFICIENCY when: (1) umbilical stump bleeding (>80% patients); "
                    "(2) delayed post-operative/post-injury bleeding onset 24-48 hours; "
                    "(3) spontaneous intracranial haemorrhage (25%); "
                    "(4) recurrent miscarriage in females; (5) poor wound healing. "
                    "DIAGNOSIS: Urea clot solubility test (5M urea — clot dissolves in <2h if FXIII deficient; "
                    "sensitive but not quantitative) + specific FXIII activity assay (ammonia-release "
                    "fluorometric assay). "
                    "TREATMENT: FXIII concentrate (Corifact/Fibrogammin) prophylaxis every 4-6 weeks "
                    "MANDATORY (half-life ~11 days; target activity >10% to prevent spontaneous ICH). "
                    "FFP/cryoprecipitate if concentrate unavailable."
                ),
            },
            {
                "type": "danger",
                "title": "GLANZMANN (ITGA2B/ITGB3): GP IIb/IIIa INHIBITORS ABSOLUTELY CONTRAINDICATED",
                "body": (
                    "GP IIb/IIIa inhibitors (abciximab/ReoPro, eptifibatide/Integrilin, tirofiban/Aggrastat) "
                    "block the αIIbβ3 fibrinogen receptor on platelets. In Glanzmann thrombasthenia, "
                    "αIIbβ3 is already absent/non-functional — these drugs have no additional platelet "
                    "effect BUT severely impair any residual fibrinogen-bridging function that may exist "
                    "in variant/partial forms. Critically: eptifibatide and tirofiban are widely "
                    "administered empirically in ACS; if a GT patient presents with ACS, administering "
                    "these drugs cannot further platelet function and removes the only treatment option "
                    "(platelet transfusion becomes potentially less effective). "
                    "ADDITIONALLY: NSAIDs/aspirin inhibit thromboxane A2 (TXA2)-driven platelet activation "
                    "— removing even this minor residual activation pathway worsens bleeding in GT. "
                    "For GT patients requiring cardiac procedures: rFVIIa + platelet transfusion + "
                    "haematology consultation mandatory. NSAID-free analgesia protocols essential."
                ),
            },
            {
                "type": "warning",
                "title": "F8/F9 INHIBITORS: FVIII/FIX REPLACEMENT INEFFECTIVE — BYPASS THERAPY REQUIRED",
                "body": (
                    "Inhibitors (neutralising IgG4 antibodies against FVIII or FIX) develop in "
                    "25-30% of severe Hemophilia A patients (and 3-5% of HB). Once inhibitors develop, "
                    "standard FVIII/FIX concentrate replacement is INEFFECTIVE (antibody neutralises the "
                    "infused factor immediately). BYPASS THERAPIES bypass the inhibited factor: "
                    "(1) rFVIIa (NovoSeven): activates FX via TF-independent mechanism at high pharmacological doses; "
                    "(2) aPCC (FEIBA): prothrombin complex concentrate with activated factors (FVIIa/Xa/IIa) "
                    "— bypasses FVIII/FIX. "
                    "EMICIZUMAB (Hemlibra): bispecific antibody bridging FIXa and FX (mimics FVIIIa "
                    "function) — effective REGARDLESS of inhibitor status for HA (FDA 2017); "
                    "subcutaneous; game-changer for inhibitor management. "
                    "IMMUNE TOLERANCE INDUCTION (ITI): high-dose daily FVIII infusions → antibody "
                    "elimination in 60-70% — first-line strategy for eradication of inhibitors. "
                    "CAUTION: Do NOT give aPCC within 24h of emicizumab (thromboembolic risk — "
                    "thrombotic microangiopathy reported)."
                ),
            },
            {
                "type": "warning",
                "title": "BERNARD-SOULIER (GP1BA): AUTOMATED PLATELET COUNT UNRELIABLE — MANUAL COUNT MANDATORY",
                "body": (
                    "In Bernard-Soulier syndrome (BSS), giant platelets (5-10 µm diameter, approaching "
                    "lymphocyte size of 8-10 µm) are pathognomonic but cause a critical diagnostic pitfall: "
                    "automated haematology analysers (which size-discriminate cells by impedance) "
                    "count platelets in the 2-20 fL range; giant BSS platelets exceed this threshold "
                    "→ they are counted as small lymphocytes (white cell volume overlap) and MISSED "
                    "from the platelet count → FALSELY LOW automated platelet count. "
                    "The true platelet count, when manually counted on a blood film, is often higher "
                    "than the automated count suggests. "
                    "CONSEQUENCE: Patients with BSS may be misdiagnosed with immune thrombocytopenia "
                    "(ITP) and inappropriately treated with IVIG, steroids, or rituximab — which are "
                    "ineffective and delay correct diagnosis. "
                    "MANDATORY: Blood film review by experienced haematologist for any thrombocytopenia "
                    "with mucocutaneous bleeding; manual platelet count; PAS staining for platelet "
                    "granules; ristocetin agglutination; flow cytometry for CD42b (GPIbα)."
                ),
            },
            {
                "type": "warning",
                "title": "HA vs HB vs vWD TYPE 2N: FACTOR ASSAY SEQUENCE MANDATORY — CLINICAL PICTURE IDENTICAL",
                "body": (
                    "Hemophilia A (F8 deficiency) and Hemophilia B (F9 deficiency) are clinically "
                    "INDISTINGUISHABLE — both present with prolonged aPTT, normal PT, spontaneous "
                    "haemarthroses, and deep muscle bleeds in males. They MUST be differentiated by "
                    "factor assay: measure FVIII first (low → HA), then FIX if FVIII normal (low → HB). "
                    "vWD Type 2N (Normandy): VWF D3 domain mutation → impaired FVIII binding → secondary "
                    "FVIII reduction (aPTT prolonged, low FVIII) → MIMICS mild HA even though VWF:Ag "
                    "and VWF:RCo may be normal or only mildly reduced. "
                    "VWF:FVIIIB binding assay (VWF:FVIIIB) is diagnostic for Type 2N. "
                    "CONSEQUENCES OF MISSED DIFFERENTIATION: treating Type 2N with FVIII concentrate "
                    "fails (FVIII is rapidly cleared because VWF carrier protein is dysfunctional → "
                    "infused FVIII has shortened half-life); correct treatment is VWF concentrate "
                    "(which also raises FVIII by providing functional VWF carrier). "
                    "RULE: Always measure VWF:Ag + VWF:RCo alongside FVIII in any male with "
                    "isolated prolonged aPTT and low FVIII — do not assume HA until VWF excluded."
                ),
            },
        ],
        "critical_rules": [
            "VWF Type2B: DDAVP ABSOLUTELY CI — acute thrombocytopenia; subtype determination mandatory before DDAVP",
            "F13A1: PT/aPTT/TT ALL NORMAL — screening tests MISS completely; urea clot solubility + FXIII assay mandatory if clinical suspicion",
            "F13A1: umbilical stump bleeding = FXIII deficiency until proven otherwise; FXIII concentrate prophylaxis every 4-6 weeks prevents ICH",
            "ITGA2B/GP1BA: GP IIb/IIIa inhibitors (abciximab, eptifibatide, tirofiban) ABSOLUTELY CI in Glanzmann; NSAIDs worsen bleeding",
            "GP1BA: giant platelets miscount on automated analyser — manual blood film count MANDATORY; do not treat as ITP",
            "F8/F9 inhibitors: FVIII/FIX ineffective — emicizumab (HA) or rFVIIa/aPCC bypass; aPCC CI within 24h of emicizumab",
            "F9 vs F8: clinically identical — FVIII assay first (HA if low), then FIX (HB if FVIII normal); VWF:FVIIIB to exclude Type2N",
            "F11: bleed-level correlation POOR — manage by bleeding site/context not FXI level; tranexamic acid highly effective for fibrinolysis-rich sites",
        ],
        "pathway_targets": {
            "F8_F9": "Intrinsic tenase complex (FIXa·FVIIIa) — FVIII/FIX replacement; emicizumab bispecific; gene therapy (etranacogene dezaparvovec HB FDA 2022; fitusiran antithrombin RNAi)",
            "VWF": "VWF-GPIbα adhesion axis — DDAVP (Type1/2A); VWF concentrate (Type3/2B); VWF:Ag/RCo monitoring",
            "F7": "TF-FVIIa extrinsic initiator — rFVIIa (NovoSeven); plasma-derived FVII concentrate; prophylaxis target FVII >30%",
            "F13A1": "Fibrin cross-linking (transglutaminase) — FXIII concentrate (Corifact/Fibrogammin) prophylaxis; target FXIII >10%",
            "ITGA2B_GP1BA": "Platelet surface glycoproteins — platelet transfusion; rFVIIa bypass; HSCT curative; alloimmunisation prevention with HLA-matched platelets",
            "F11": "Intrinsic pathway amplification + TAFI activation — tranexamic acid (anti-fibrinolysis); FFP; FXI concentrate (Hemoleven); anti-FXI ASO trials (antithrombotic application)",
        },
        "severity": sev,
        "disease_category_breakdown": disease_cat,
        "gene_stats": gene_stats,
        "kpis": kpis,
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    genes_out = []
    for gd in COAGULOPATHY_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        n = len(gpts)

        severe_bleed_pct = round(100 * sum(1 for p in gpts if p["severe_bleed"]) / n, 1)
        inhibitor_pct = round(100 * sum(1 for p in gpts if p["inhibitor"]) / n, 1)
        icb_pct = round(100 * sum(1 for p in gpts if p["icb"]) / n, 1)
        prophylaxis_pct = round(100 * sum(1 for p in gpts if p["on_prophylaxis"]) / n, 1)
        drug_error_pct = round(100 * sum(1 for p in gpts if p["drug_error"]) / n, 1)
        alloimmunised_pct = round(100 * sum(1 for p in gpts if p["alloimmunised"]) / n, 1)
        umbilical_pct = round(100 * sum(1 for p in gpts if p["umbilical_bleed"]) / n, 1)
        miscarriage_pct = round(100 * sum(1 for p in gpts if p["miscarriage"]) / n, 1)
        surveillance_pct = round(100 * sum(1 for p in gpts if p["surveillance_adherent"]) / n, 1)
        mean_age_dx = round(sum(p["age_at_dx_years"] for p in gpts) / n, 1)

        genes_out.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "gene_class": gd["gene_class"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "phenotype": gd["phenotype"],
            "disease": gd["disease"],
            "inheritance": gd["inheritance"],
            "hallmark": gd["hallmark"],
            "key_ddx": gd["key_ddx"],
            "treatment_alert": gd["treatment_alert"],
            "platelet_count": gd["platelet_count"],
            "pt_ptt": gd["pt_ptt"],
            "primary_complication": gd["primary_complication"],
            "seed": gd["seed"],
            "cohort_n": n,
            "mean_age_at_dx_years": mean_age_dx,
            "severe_bleed_pct": severe_bleed_pct,
            "inhibitor_pct": inhibitor_pct,
            "icb_pct": icb_pct,
            "prophylaxis_pct": prophylaxis_pct,
            "drug_error_pct": drug_error_pct,
            "alloimmunised_pct": alloimmunised_pct,
            "umbilical_bleed_pct": umbilical_pct,
            "miscarriage_pct": miscarriage_pct,
            "surveillance_adherent_pct": surveillance_pct,
            "severity_weights": gd["severity_weights"],
        })

    return {"genes": genes_out}


def get_definitions() -> list:
    return [
        {
            "term": "Haemostasis Cascade — Primary vs Secondary",
            "definition": (
                "HAEMOSTASIS occurs in two overlapping phases: "
                "PRIMARY HAEMOSTASIS: platelet plug formation — (1) adhesion (GPIbα-VWF under shear → "
                "platelet tethering); (2) activation (ADP, collagen, TXA2 → inside-out αIIbβ3 activation); "
                "(3) aggregation (αIIbβ3-fibrinogen bridges → platelet-platelet cross-linking). "
                "Disorders: vWD (adhesion), Glanzmann (aggregation), Bernard-Soulier (adhesion/GPIbα). "
                "SECONDARY HAEMOSTASIS: coagulation cascade → fibrin clot — Intrinsic (FVIII, FIX, FXI, FXII) "
                "+ Extrinsic (FVII, TF) → Common (FX, FV, prothrombin, fibrinogen) → fibrin polymer. "
                "Disorders: HA (FVIII), HB (FIX), Hemophilia C (FXI), FVII deficiency (extrinsic only). "
                "TERTIARY HAEMOSTASIS: fibrin cross-linking (FXIII) + fibrinolysis regulation (TAFI, PAI-1, α2-AP). "
                "Disorder: FXIII deficiency (cross-linking absent — delayed clot dissolution). "
                "Laboratory tests: PT (extrinsic + common), aPTT (intrinsic + common), TT (fibrinogen → fibrin). "
                "FXIII deficiency: ALL three tests NORMAL — requires dedicated testing."
            ),
        },
        {
            "term": "FVIII Inhibitors (Anti-FVIII Alloantibodies)",
            "definition": (
                "Neutralising IgG4 antibodies against exogenous (transfused) FVIII, developing in "
                "25-30% of severe Hemophilia A patients (usually within first 20 exposure days). "
                "Risk factors: severe HA (F8 null mutations — intron 22 inversion, large deletions), "
                "young age at first exposure, African/Hispanic ancestry, immune response genes (HLA, CTLA-4). "
                "BETHESDA UNIT (BU): assay for inhibitor titre — 1 BU = inhibitor concentration that "
                "inactivates 50% of FVIII in a standard mixture. "
                "LOW-TITRE: <5 BU — standard FVIII (high dose) may still work. "
                "HIGH-TITRE: ≥5 BU — FVIII completely neutralised → bypass therapy mandatory. "
                "IMMUNE TOLERANCE INDUCTION (ITI): daily high-dose FVIII infusions (50 IU/kg QD or "
                "200 IU/kg QD) for months-years → 60-70% success rate (inhibitor titre → undetectable). "
                "EMICIZUMAB: non-factor replacement; bispecific antibody bridging FIXa and FX → "
                "bypasses FVIII entirely; effective regardless of inhibitor titre (FDA 2017 for inhibitors). "
                "AVOID aPCC within 24h of emicizumab — TMA reported (thromboembolic microangiopathy)."
            ),
        },
        {
            "term": "Intrinsic Tenase Complex",
            "definition": (
                "The intrinsic tenase complex is the central amplifier of coagulation and the "
                "molecular target of Hemophilia A and B: "
                "COMPOSITION: FIXa (serine protease) + FVIIIa (cofactor) + phospholipid membrane + Ca²⁺. "
                "ASSEMBLY: Thrombin (from initial TF-FVIIa initiation) cleaves and activates FVIII → FVIIIa; "
                "FXIa (or TF-FVIIa) activates FIX → FIXa; FIXa and FVIIIa co-assemble on platelet "
                "phospholipid membrane. "
                "FUNCTION: Catalyses FX → FXa at ~10⁵-fold accelerated rate vs FIXa alone "
                "(FVIIIa is the obligate cofactor — Hemophilia A = complete loss of this acceleration). "
                "FXa → prothrombinase complex (FXa + FVa + phospholipid + Ca²⁺) → prothrombin → thrombin. "
                "IN HEMOPHILIA A: FIX (FIXa) is present but cannot form functional tenase → FX poorly activated → "
                "insufficient thrombin for stable fibrin clot → bleeding. "
                "IN HEMOPHILIA B: FIXa absent → identical tenase failure. "
                "EMICIZUMAB: bispecific antibody that physically bridges FIXa and FX on platelet surface, "
                "substituting for the FVIIIa cofactor function — works even when FVIIIa is neutralised "
                "by inhibitor antibodies."
            ),
        },
        {
            "term": "von Willebrand Factor (VWF) — Multimer Structure and Function",
            "definition": (
                "VWF is the largest multimeric plasma glycoprotein (monomer ~250 kDa; multimers up to "
                "20,000 kDa). It serves two distinct haemostatic functions: "
                "(1) PLATELET ADHESION: VWF A1 domain binds platelet GPIbα under high shear (arteries). "
                "At low shear (veins), GPIbα-VWF binding is insufficient → VWD primarily affects "
                "high-flow arterial sites (mucocutaneous bleeding, not haemarthroses). "
                "Largest multimers are most active in platelet adhesion (largest have most GPIbα binding sites). "
                "(2) FVIII CARRIER: VWF D3 domain binds FVIII → protects FVIII from premature proteolytic "
                "degradation; VWF concentrates FVIII at injury sites. Loss of VWF → secondary FVIII "
                "reduction → prolonged aPTT (Type 3 vWD can mimic mild HA). "
                "ADAMTS13: metalloprotease that cleaves ultra-large VWF multimers (ULVWFs) released from "
                "endothelium → maintains appropriate multimer size. ADAMTS13 deficiency → ULVWFs accumulate "
                "→ platelet clumping → TTP (thrombotic thrombocytopenic purpura — opposite of vWD). "
                "Type 2B vWD: GOF mutation in A1 domain → VWF binds GPIbα spontaneously (without shear) "
                "→ platelet clumping → thrombocytopenia + loss of large multimers (cleaved by ADAMTS13 "
                "after platelet binding) → DDAVP releases more abnormal VWF → worsening."
            ),
        },
        {
            "term": "Urea Clot Solubility Test (Factor XIII Deficiency Screening)",
            "definition": (
                "The 5M urea clot solubility test is the classic screening test for Factor XIII (FXIII) "
                "deficiency, exploiting the biochemical function of FXIIIa: "
                "PRINCIPLE: FXIIIa cross-links fibrin chains via isopeptide bonds (γ-glutamyl-ε-lysine), "
                "creating a covalent polymer that resists denaturing agents. "
                "PROCEDURE: Patient plasma is clotted (adding thrombin + Ca²⁺) → clot formed → "
                "immerse in 5M urea (a denaturing agent) or 1% monochloroacetic acid → incubate 24h at 37°C. "
                "NORMAL (FXIII present): clot resists dissolution — remains intact at 24h. "
                "ABNORMAL (FXIII absent/deficient): clot dissolves within 2 hours — non-cross-linked "
                "fibrin polymer cannot withstand denaturing conditions. "
                "SENSITIVITY: detects FXIII <5% (severe deficiency) with >95% sensitivity. "
                "LIMITATION: Does NOT detect moderate deficiency (FXIII 5-30%) — quantitative "
                "FXIII activity assay mandatory for complete evaluation. "
                "CLINICAL APPLICATION: Any patient with (1) umbilical stump bleeding; (2) delayed "
                "post-surgical bleeding onset 24-48h; (3) spontaneous ICH with normal routine coagulation; "
                "(4) recurrent miscarriage — should be tested. NORMAL PT/aPTT/TT does NOT exclude FXIII deficiency."
            ),
        },
        {
            "term": "Ristocetin Agglutination (RIPA) and its Diagnostic Role",
            "definition": (
                "Ristocetin is an antibiotic that induces VWF A1 domain conformational change → "
                "VWF binds platelet GPIbα → platelet agglutination (not aggregation — no activation, "
                "only VWF-GPIbα cross-linking). "
                "NORMAL RIPA: ristocetin → VWF conformational change → VWF-A1 binds GPIbα → "
                "platelets agglutinate (clump) in proportion to VWF concentration. "
                "DIAGNOSTIC PATTERNS: "
                "(1) Type 1/2A vWD: reduced RIPA (fewer VWF multimers → less agglutination). "
                "(2) Type 2B vWD: INCREASED RIPA at LOW ristocetin concentrations (A1 GOF → "
                "VWF binds GPIbα with lower ristocetin requirement — low-dose RIPA positive). "
                "(3) Type 3 vWD: absent RIPA (no VWF). "
                "(4) Glanzmann thrombasthenia: NORMAL RIPA (GPIbα intact — agglutination proceeds; "
                "αIIbβ3 is NOT required for ristocetin agglutination). KEY DDx from BSS. "
                "(5) Bernard-Soulier syndrome: ABSENT RIPA even with added normal VWF (GPIbα absent — "
                "no receptor for VWF-A1 to bridge to). KEY DDx from Glanzmann. "
                "(6) Platelet-type vWD: RIPA increased at low ristocetin (GPIbα GOF — "
                "identical to Type 2B; distinguished by mixing studies)."
            ),
        },
        {
            "term": "Emicizumab (Hemlibra) — Bispecific Antibody Mechanism",
            "definition": (
                "Emicizumab (ACE910, Hemlibra) is a bispecific humanised IgG4 antibody that "
                "simultaneously binds FIXa and FX, physically bridging them on the platelet membrane "
                "surface to substitute for the FVIIIa cofactor function — enabling FX activation "
                "WITHOUT requiring FVIII. "
                "MECHANISM: FVIIIa normally serves as a bridge/scaffold presenting FIXa to FX on "
                "the phospholipid surface and allosterically activating FIXa catalysis. "
                "Emicizumab mimics this bridging function: one arm binds FIXa, the other binds FX → "
                "FX is presented to FIXa in the correct orientation for cleavage. "
                "KEY PROPERTY: Works regardless of FVIII inhibitor status — antibodies against FVIII "
                "have NO effect on emicizumab (different structure, non-FVIII epitope). "
                "FDA APPROVALS: 2017 (HA + inhibitors, all ages); 2018 (HA without inhibitors, all ages). "
                "DOSING: Subcutaneous injection weekly/biweekly/monthly (4-week loading then maintenance). "
                "CLINICAL IMPACT: Annual bleed rate reduced ~80-90% vs on-demand FVIII; prophylaxis "
                "maintained with once-monthly SC injection vs 2-3× weekly IV. "
                "SAFETY: aPCC (FEIBA) given within 24h of emicizumab → thrombotic microangiopathy (TMA) "
                "and thromboembolism — contraindicated. rFVIIa safe when used with emicizumab. "
                "NOT for Hemophilia B (targets FIX-FX, which is downstream of the F9 defect)."
            ),
        },
        {
            "term": "Platelet Glycoprotein Nomenclature",
            "definition": (
                "Platelet glycoproteins (GPs) are surface membrane proteins critical for haemostasis: "
                "GPIbα (CD42b, encoded by GP1BA): α-subunit of GpIb-IX-V complex; VWF receptor; "
                "thrombin high-affinity receptor; absent/LOF → Bernard-Soulier syndrome; "
                "RIPA absent (VWF cannot bridge to platelet). "
                "GPIbβ (CD42c, GP1BB): β-subunit of GpIb-IX-V; stabilises complex; BSS if mutated. "
                "GPIX (CD42a, GP9): GpIb-IX-V subunit; most commonly mutated in BSS after GP1BA. "
                "GPIIb (αIIb, CD41, ITGA2B): α-subunit of αIIbβ3 complex; fibrinogen receptor; "
                "absent/LOF → Glanzmann thrombasthenia; aggregation absent; RIPA NORMAL. "
                "GPIIIa (β3, CD61, ITGB3): β-subunit of αIIbβ3; Glanzmann Type II if mutated. "
                "GPVI (CD36-related): collagen receptor; GPVI deficiency → reduced collagen-induced aggregation. "
                "GPIa-IIa (α2β1, VLA-2): secondary collagen receptor; mild bleeding if absent. "
                "FLOW CYTOMETRY PANEL: CD41 (GPIIb), CD42b (GPIbα), CD61 (GPIIIa) — "
                "absent CD41/CD61 with present CD42b → Glanzmann; "
                "absent CD42b with present CD41/CD61 → Bernard-Soulier."
            ),
        },
    ]


if __name__ == "__main__":
    ov = get_overview()
    print(f"Atlas: {ov['atlas_name']}")
    print(f"Patients: {ov['n_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Severe bleed: {ov['aggregate_clinical']['severe_bleed_pct']}%")
    print(f"Inhibitor: {ov['aggregate_clinical']['inhibitor_pct']}%")
    print(f"ICH: {ov['aggregate_clinical']['icb_pct']}%")
    print(f"Drug error: {ov['aggregate_clinical']['drug_error_pct']}%")
    bk = get_breakdown()
    for g in bk["genes"]:
        print(f"  {g['gene']}: n={g['cohort_n']} severe_bleed={g['severe_bleed_pct']}% icb={g['icb_pct']}% inhibitor={g['inhibitor_pct']}%")
    defs = get_definitions()
    print(f"Definitions: {len(defs)}")
