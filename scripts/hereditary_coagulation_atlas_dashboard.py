#!/usr/bin/env python3
"""Hereditary-Coagulation-Atlas — Complete 8-Gene Hereditary Coagulation Factor Deficiency Atlas
F8      (FVIII; 2351 aa; Xq28; XLR; Hemophilia A;
         most common inherited coagulopathy 1:5000–10000 males;
         EMICIZUMAB SC prophylaxis FDA 2017 — game-changer; inhibitors 25-30%;
         seed SEED_BASE+0) ·
F9      (FIX; 461 aa; Xq27.1; XLR; Hemophilia B / Christmas Disease;
         1:25000–30000 males; Padua variant (p.Arg338Leu) — superactive FIX;
         fitusiran (siRNA antithrombin) emerging; seed SEED_BASE+1) ·
VWF     (von Willebrand Factor; 2813 aa; 12p13.31; AD/AR;
         most common inherited bleeding disorder 1:100–1000;
         DDAVP works type 1/2A NOT type 2B (thrombocytopenia risk) NOT type 3;
         seed SEED_BASE+2) ·
F11     (FXI; 625 aa; 4q35.2; AR; Hemophilia C;
         Ashkenazi 1:450 Lys521Ter+Glu117Ter founder mutations;
         CONCENTRATION-INDEPENDENT bleeding — mucosal sites bleed more than trauma;
         seed SEED_BASE+3) ·
F13A1   (FXIII-A; 731 aa; 6p25.1; AR; Factor XIII Deficiency;
         1:5M; UMBILICAL CORD STUMP BLEEDING PATHOGNOMONIC;
         routine PT/APTT/TT NORMAL — test specifically; seed SEED_BASE+4) ·
F7      (FVII; 466 aa; 13q34; AR; Factor VII Deficiency;
         most common rare coagulopathy; PT prolonged ALONE; APTT normal;
         rFVIIa (NovoSeven) 15–30 mcg/kg; seed SEED_BASE+5) ·
FGA     (Fibrinogen alpha chain; 644 aa; 4q28.1; AR; Afibrinogenemia/Hypofibrinogenemia;
         umbilical cord + intracranial + haemarthrosis + miscarriage;
         fibrinogen concentrate (Riastap/Fibryga) 70 mg/kg; seed SEED_BASE+6) ·
ADAMTS13 (ADAMTS13; 1427 aa; 9q34.11; AR; Congenital TTP / Upshaw-Schulman Syndrome;
          MAHA + thrombocytopenia + AKI + neurological — PENTAD NOT REQUIRED;
          PLASMA EXCHANGE life-saving; caplacizumab for acute TTP; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1574–1581)
"""

import random

SEED_BASE = 1574

COAGULATION_GENES = [
    # ── F8 — Haemophilia A ─────────────────────────────────────────────
    {
        "gene": "F8",
        "protein": "Factor VIII — Haemophilia A, Emicizumab Prophylaxis, Inhibitors 25-30%",
        "alias": (
            "F8; OMIM gene 300841; Haemophilia A OMIM 306700; Xq28; 2351 aa; ~280 kDa (processed); "
            "XLR; prevalence 1:5000–10000 males; 30% de novo mutations. "
            "FVIII is a cofactor in the intrinsic Xase complex (FVIIIa + FIXa → activates FX). "
            "Domains: A1-A2-B-A3-C1-C2. B domain is heavily glycosylated and dispensable (removed "
            "in recombinant products). FVIII circulates bound to VWF (protects from proteolysis). "
            "Severity: severe (<1 IU/dL), moderate (1–5 IU/dL), mild (5–40 IU/dL). "
            "Severe: haemarthrosis, muscle bleeds, life-threatening intracranial haemorrhage (ICH). "
            "EMICIZUMAB (Hemlibra): bispecific antibody mimicking FVIIIa — bridges FIXa and FX; "
            "SC prophylaxis Q1W/Q2W/Q4W; FDA 2017 (with inhibitors); FDA 2018 (without inhibitors). "
            "Inhibitors (FVIII neutralising antibodies): 25–30% of severe HA — most devastating complication; "
            "peak risk <50 exposure days. Immune tolerance induction (ITI) with high-dose FVIII (Malmö/Van Creveld). "
            "Bypassing agents: aPCC (FEIBA) or rFVIIa for inhibitor bleeds. "
            "Gene therapy: valoctocogene roxaparvovec (BMGene411-AAV5-F8-BDD) EMA 2022 — durable response. "
            "Newborn: cord blood FVIII normal at birth; circumcision risk — test at birth if family history. "
            "Carrier females: FVIII levels variable — can bleed if FVIII <40 IU/dL due to skewed lyonisation."
        ),
        "aa": "2351 aa",
        "kDa": "~280 kDa",
        "locus": "Xq28",
        "omim_gene": 300841,
        "omim_disease": 306700,
        "inheritance": "XLR; 30% de novo; carrier females can be symptomatic (FVIII <40 IU/dL)",
        "gene_class": (
            "F8 encodes coagulation Factor VIII, a ~2351-aa glycoprotein with domain structure A1-A2-B-A3-C1-C2. "
            "Key functions: cofactor for FIXa in the intrinsic Xase complex (tenase); "
            "activated by thrombin/FXa (removes B domain); inactivated by APC/FXa (cleaves A2 domain). "
            "Mutation spectrum: intron 22 inversion (40–50% severe HA) — GENE SPECIFIC TEST required; "
            "intron 1 inversion (5%); nonsense/frameshift (severe); missense (mild-moderate); "
            "large deletions (highest inhibitor risk). Over 2000 variants catalogued (HAMSTeRS database)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "F8-EMICIZUMAB-PROPHYLAXIS: Emicizumab SC prophylaxis is now FIRST-LINE for severe HA (with/without inhibitors) — FDA 2017/2018; reduces ABR by >85%; use Q1W/Q2W/Q4W dosing; no FVIII monitoring required",
            "F8-INHIBITOR-WATCH: Inhibitors develop in 25–30% of severe HA — peak risk <50 exposure days; check Bethesda titre before every surgery; inhibitor = bypassing agent (aPCC/rFVIIa) required, NOT standard FVIII",
            "F8-ICH-EMERGENCY: Intracranial haemorrhage is life-threatening — treat FIRST with factor concentrate, image SECOND; target FVIII 100% immediately; maintain 50% for 2 weeks post-ICH",
            "F8-INTRON22-INVERSION: Intron 22 inversion (40–50% severe HA) NOT detected by standard sequencing — requires Southern blot or long-range PCR; always perform inversion-specific testing in severe HA with no point mutation found",
            "F8-JOINT-PROPHYLAXIS: Primary prophylaxis (from age 1–2 years) prevents haemophilic arthropathy — once-weekly or emicizumab; target joint disease is the main long-term morbidity without prophylaxis",
            "F8-CARRIER-FVIII-LEVEL: Carrier females can have FVIII <40 IU/dL due to skewed lyonisation — test ALL carriers; treat if FVIII <50 IU/dL before invasive procedures or delivery",
            "F8-NEWBORN-TEST: If family history, test cord blood FVIII at birth; neonatal ICH risk with assisted delivery; avoid intramuscular injections until status confirmed",
            "F8-VWF-BINDING: Low VWF can cause low FVIII (VWF carries FVIII) — distinguish true HA from type 3 vWD by VWF antigen + ristocetin co-factor; different management",
        ],
        "etiologies": {
            "Intron 22 inversion (severe)": 16,
            "Nonsense/frameshift (severe)": 10,
            "Missense (mild-moderate)": 8,
            "Large deletion (severe, highest inhibitor risk)": 4,
            "Intron 1 inversion (severe)": 2,
        },
        "stats": {
            "severe_pct": 65,
            "inhibitor_pct": 28,
            "emicizumab_on_prophylaxis_pct": 55,
            "haemarthrosis_pct": 80,
            "ich_lifetime_pct": 10,
            "gene_therapy_eligible_pct": 22,
            "mean_dx_age_months": 8,
            "mean_dx_delay_months": 3,
        },
        "dx_delay_distribution": {"<3m": 20, "3-12m": 12, "12-36m": 5, ">36m": 3},
    },

    # ── F9 — Haemophilia B / Christmas Disease ────────────────────────
    {
        "gene": "F9",
        "protein": "Factor IX — Haemophilia B (Christmas Disease), Padua Variant, Fitusiran Emerging",
        "alias": (
            "F9; OMIM gene 300746; Haemophilia B OMIM 306900; Xq27.1; 461 aa; ~55 kDa; "
            "XLR; prevalence 1:25000–30000 males; ~30% de novo. "
            "FIX is a vitamin K-dependent serine protease (Gla domain → EGF1 → EGF2 → activation "
            "peptide → catalytic domain). Activated by TF-FVIIa (extrinsic) or FXIa (intrinsic). "
            "FIXa + FVIIIa + Ca2+ + phospholipid = intrinsic Xase → activates FX → thrombin. "
            "Severity: severe (<1 IU/dL), moderate (1–5 IU/dL), mild (5–40 IU/dL). "
            "Padua variant (p.Arg338Leu): gain-of-function → 8× FIX activity — investigational "
            "for gene therapy vector (etranacogene dezaparvovec uses Padua FIX-R338L). "
            "Haemophilia B Leyden: promoter mutations → severe in childhood, spontaneous recovery "
            "after puberty as testosterone upregulates FIX — UNIQUE NATURAL HISTORY. "
            "Gene therapy: etranacogene dezaparvovec (AMT-061, FIX-Padua AAV5) FDA 2022 — "
            "durable high FIX levels (~40 IU/dL at 2y); fitusiran (siRNA → reduces antithrombin) "
            "SC Q1M — non-factor therapy reducing thrombin inhibition. "
            "Inhibitors RARE in HB (<5%) but anaphylaxis risk — particularly in null mutations (large deletions). "
            "HB patients on prophylaxis: test FIX activity + trough level; target >1 IU/dL trough (standard), "
            ">10 IU/dL (pharmacokinetic-guided). "
            "Vitamin K: FIX is vitamin K-dependent; warfarin effect: FIX drops fastest after warfarin "
            "initiation (shortest half-life of vitamin K factors at ~24h)."
        ),
        "aa": "461 aa",
        "kDa": "~55 kDa",
        "locus": "Xq27.1",
        "omim_gene": 300746,
        "omim_disease": 306900,
        "inheritance": "XLR; ~30% de novo; Haemophilia B Leyden: puberty-mediated recovery",
        "gene_class": (
            "F9 encodes coagulation Factor IX, a 461-aa vitamin K-dependent serine protease. "
            "Domain structure: Gla (γ-carboxylation, Ca2+-dependent membrane binding) → "
            "EGF1 → EGF2 → activation peptide (removed on activation) → serine protease (catalytic). "
            "Key residues: Arg145-Ala146 (TF-FVIIa activation site), Arg333-Val334 (FXIa site), "
            "Ser365 (catalytic), Arg338 (Padua GOF). "
            "Mutation spectrum: missense dominant (mild-severe); nonsense/frameshift (severe); "
            "Leyden (promoter: A→G at -5 or -6); Padua (R338L) — GOF used in gene therapy. "
            "Inhibitors in large deletions/null mutations — associated with anaphylaxis (unlike HA)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "F9-LEYDEN-PUBERTY-RECOVERY: Haemophilia B Leyden (promoter mutations) is severe in childhood but spontaneously IMPROVES AFTER PUBERTY as testosterone drives FIX expression — UNIQUE; do not treat as persistent severe HB in adults without checking FIX levels",
            "F9-INHIBITOR-ANAPHYLAXIS: Inhibitors in HB (rare, <5%) — BUT anaphylaxis risk is HIGH especially with null mutations (large deletions); ALWAYS give FIX in hospital setting initially; adrenaline available; anaphylaxis = use rFVIIa ONLY",
            "F9-GENE-THERAPY-PADUA: Etranacogene dezaparvovec (FIX-Padua AAV5) FDA 2022 — durable FIX levels ~40 IU/dL; pre-treatment: AAV5 neutralising antibody screening; anti-FIX inhibitor exclusion; liver function monitoring post-infusion",
            "F9-FITUSIRAN-ANTITHROMBIN: Fitusiran (siRNA targeting antithrombin) is a non-factor prophylaxis approved for HB with/without inhibitors — reduces antithrombin → shifts haemostatic balance; monitor AT levels; thrombosis risk if overdosed",
            "F9-SURGERY-CORRECTION: Target FIX 80–100% pre-major surgery; post-op: maintain >50% for 7–14 days; pharmacokinetic-guided dosing — FIX t½ = 18–24h (longer than FVIII); Q12-24h dosing",
            "F9-VITAMIN-K-SHORTEST: FIX has the SHORTEST t½ among vitamin K factors (~24h) — first to drop after warfarin and first to recover after vitamin K; this is why PT (reflects FVII) prolongs first, but FIX deficiency deepens rapidly with warfarin overdose",
            "F9-CARRIER-TESTING: Carrier females: FIX levels often low (50% of normal) — test before procedures; some carriers are symptomatic (skewed lyonisation); all female relatives of HB males should be offered carrier testing",
            "F9-MILD-DENTAL-DESMOPRESSIN: Mild HB (FIX 5–40 IU/dL): desmopressin (DDAVP) does NOT raise FIX — unlike in mild HA; always use FIX concentrate for mild HB procedures",
        ],
        "etiologies": {
            "Missense (mild-severe)": 18,
            "Nonsense/frameshift (severe)": 10,
            "Promoter mutation (Leyden, severe→recovery)": 6,
            "Large deletion (severe, anaphylaxis risk)": 4,
            "Splice site": 2,
        },
        "stats": {
            "severe_pct": 45,
            "leyden_pct": 12,
            "inhibitor_pct": 4,
            "gene_therapy_treated_pct": 18,
            "anaphylaxis_with_inhibitor_pct": 50,
            "mean_dx_age_months": 12,
            "mean_dx_delay_months": 4,
        },
        "dx_delay_distribution": {"<3m": 16, "3-12m": 14, "12-36m": 6, ">36m": 4},
    },

    # ── VWF — von Willebrand Disease ─────────────────────────────────
    {
        "gene": "VWF",
        "protein": "von Willebrand Factor — vWD Types 1/2/3, DDAVP Type 1/2A Only, NOT Type 2B/3",
        "alias": (
            "VWF; OMIM gene 613160; vWD type 1 OMIM 193400, type 2A 613554, type 2B 613554, "
            "type 3 277480; 12p13.31; 2813 aa (pro-VWF precursor); ~220 kDa (monomer); "
            "AD (types 1/2) or AR (type 3); prevalence ~1:100–1000 (type 1 most common). "
            "VWF is a large multimeric glycoprotein critical for platelet adhesion (GPIbα binding "
            "under shear) and FVIII carrier (protects FVIII from proteolysis). "
            "Type 1 (60–80%): quantitative partial deficiency; AD; mucocutaneous bleeding; "
            "DDAVP (desmopressin) releases VWF from Weibel-Palade bodies — EFFECTIVE. "
            "Type 2A (15–20%): qualitative defect — loss of high-molecular-weight multimers (HMWM); "
            "DDAVP variably effective — test with trial dose; ristocetin co-factor/VWF:Ag ratio <0.6. "
            "Type 2B (5%): GOF VWF-GPIbα binding → spontaneous platelet binding → CONSUMES HMWM; "
            "THROMBOCYTOPENIA (platelet clumping); DDAVP is CONTRAINDICATED — releases abnormal VWF "
            "→ worsens thrombocytopenia. "
            "Type 2N (Normandy): VWF-FVIII binding site mutation → low FVIII (mimics mild HA); "
            "distinguish by VWF-FVIII binding assay. "
            "Type 2M: multimers normal but GPIbα binding reduced. "
            "Type 3: complete VWF deficiency (AR); severe mucocutaneous + haemarthrosis + ICH; "
            "DDAVP NOT effective; VWF concentrate mandatory. "
            "Ristocetin co-factor activity (VWF:RCo) is the KEY functional test; "
            "low VWF:RCo + normal VWF:Ag = type 2 qualitative defect."
        ),
        "aa": "2813 aa",
        "kDa": "~220 kDa monomer; ultralong multimers >20,000 kDa",
        "locus": "12p13.31",
        "omim_gene": 613160,
        "omim_disease": 193400,
        "inheritance": "AD types 1/2 (type 2B GOF); AR type 3; blood group O: VWF levels 25% lower",
        "gene_class": (
            "VWF encodes a 2813-aa glycoprotein (pro-VWF after signal peptide cleavage). "
            "Domain structure: D1-D2 (propeptide) → D'-D3 (FVIII binding + interchain disulfide) → "
            "A1 (GPIbα binding, ristocetin site) → A2 (ADAMTS13 cleavage site, Tyr1605-Met1606) → "
            "A3 (collagen binding) → D4 → B1-B3 → C1-C6 (propeptide-dependent dimerisation) → CK. "
            "Ultralarge multimers (ULVWFs) are hyperadhesive → ADAMTS13 cleaves in flowing blood → "
            "normal-sized multimers. Type 2B mutations in A1 domain increase GPIbα affinity (GOF). "
            "Type 2N mutations in D'-D3 reduce FVIII binding. Type 2A: mutations in A2 → ADAMTS13 "
            "hypersensitivity (degradation) or A1/A2 → impaired multimerisation."
        ),
        "n_patients": 40,
        "key_alerts": [
            "VWF-DDAVP-TYPE2B-CONTRAINDICATED: DDAVP is ABSOLUTELY CONTRAINDICATED in type 2B vWD — releases abnormal VWF that spontaneously clumps platelets → worsens thrombocytopenia + bleeding; always subtype before giving DDAVP",
            "VWF-DDAVP-TYPE3-INEFFECTIVE: DDAVP does NOT work in type 3 vWD (no VWF in storage) or severe type 1 — use VWF concentrate (Humate-P/Haemate-P/Veyvondi); never rely on DDAVP in type 3",
            "VWF-DDAVP-TRIAL-DOSE: Always perform a DDAVP trial dose before relying on it for surgery — individual response varies; check VWF:RCo at 1h post-DDAVP; tachyphylaxis after 2–3 consecutive doses (deplete Weibel-Palade stores)",
            "VWF-TYPE2N-MIMIC-HA: Type 2N vWD (Normandy) mimics mild haemophilia A — both have low FVIII; distinguish by VWF-FVIII binding assay; type 2N responds to VWF concentrate NOT FVIII alone",
            "VWF-BLOOD-GROUP-O: Blood group O individuals have VWF levels 25% lower than non-O — can cause type 1 vWD diagnosis at the O-group lower reference limit; report as 'low VWF' not type 1 if borderline",
            "VWF-RISTOCETIN-KEY-TEST: VWF:Ristocetin co-factor activity (VWF:RCo) is the KEY functional assay; VWF:RCo/VWF:Ag ratio <0.6 indicates qualitative type 2 defect; low VWF:RCo + normal multimers + low GPIbα binding = type 2M",
            "VWF-PREGNANCY-RISES: VWF rises in pregnancy (especially type 1) — some type 1 women normalise at term; check VWF:RCo at 28–34 weeks; post-partum VWF DROPS sharply — haemorrhage risk at 3–5 days post-partum",
            "VWF-MENORRHAGIA-FIRST: Menorrhagia is the MOST COMMON PRESENTATION in women with vWD — ask about heavy periods; tranexamic acid + DDAVP + COC pill for menorrhagia management",
        ],
        "etiologies": {
            "Type 1 (quantitative partial, AD)": 24,
            "Type 2A (qualitative, loss HMWM)": 8,
            "Type 2B (GOF, thrombocytopenia)": 4,
            "Type 3 (complete deficiency, AR)": 2,
            "Type 2N (FVIII binding, Normandy)": 2,
        },
        "stats": {
            "type1_pct": 60,
            "type2_pct": 30,
            "type3_pct": 5,
            "ddavp_responsive_pct": 65,
            "menorrhagia_presenting_pct": 45,
            "pregnancy_management_needed_pct": 30,
            "mean_dx_age": 18,
            "mean_dx_delay_months": 48,
        },
        "dx_delay_distribution": {"<12m": 10, "12-36m": 12, "36-60m": 10, ">60m": 8},
    },

    # ── F11 — Haemophilia C / FXI Deficiency ─────────────────────────
    {
        "gene": "F11",
        "protein": "Factor XI — Haemophilia C, Ashkenazi Founder, Concentration-Independent Bleeding",
        "alias": (
            "F11; OMIM gene 264900; Haemophilia C / FXI Deficiency OMIM 612416; 4q35.2; 625 aa; "
            "~80 kDa; AR (homozygous or compound heterozygous); prevalence Ashkenazi 1:450 (Lys521Ter "
            "most common + Glu117Ter second founder); general population 1:1,000,000. "
            "FXI is a serine protease activated by FXIIa (contact activation) or thrombin (feedback). "
            "FXIa activates FIX → intrinsic pathway amplification. "
            "UNIQUE BLEEDING PATTERN: bleeding does NOT correlate with FXI plasma level (unlike HA/HB). "
            "Mucosal sites (tonsils, urinary tract, uterus, nasal) bleed MORE than FXI level predicts; "
            "haemarthrosis and spontaneous bleeds are RARE even with severe deficiency (<1 IU/dL). "
            "This is because mucosal tissues are rich in tissue plasminogen activator (tPA) → local "
            "hyperfibrinolysis → bleeds are more fibrinolysis-dependent than factor level-dependent. "
            "THEREFORE: antifibrinolytics (tranexamic acid) are KEY treatment — often more effective "
            "than FXI concentrate alone. "
            "FXI concentrate (BeneFIX-equivalent for FXI is Hemoleven) is available in Europe; "
            "FFP as alternative. Caution: FXI concentrates can be thrombogenic (especially in elderly). "
            "DESMOPRESSIN does NOT effectively raise FXI — not useful. "
            "Dental procedures: tranexamic acid mouthwash often sufficient even without factor replacement. "
            "Antisense oligonucleotide (IONIS-FXIRx): in trials to PREVENT thrombosis — FXI as antithrombotic target."
        ),
        "aa": "625 aa",
        "kDa": "~80 kDa",
        "locus": "4q35.2",
        "omim_gene": 264900,
        "omim_disease": 612416,
        "inheritance": "AR; Ashkenazi Jewish founder mutations Lys521Ter + Glu117Ter; carrier ~8% Ashkenazi",
        "gene_class": (
            "F11 encodes coagulation Factor XI, a 625-aa homodimeric serine protease. "
            "Domain structure: 4 apple domains (A1–A4) + serine protease domain. "
            "Apple 2 (A2): HK (high-molecular-weight kininogen) binding; "
            "Apple 4 (A4): homodimerisation + FXIIa activation site + platelet GpIb binding. "
            "Catalytic triad: His413-Asp462-Ser557. "
            "Activated by FXIIa (Arg369) and thrombin (feedback amplification on platelet surface). "
            "Ashkenazi mutations: Lys521Ter (type II — residual FXI antigen without function); "
            "Glu117Ter (type III — no antigen). Compound Lys521+Glu117 = severe deficiency but "
            "bleeding still unpredictable — mucosal > trauma."
        ),
        "n_patients": 40,
        "key_alerts": [
            "F11-CONCENTRATION-INDEPENDENT-BLEEDING: FXI plasma level DOES NOT PREDICT BLEEDING SEVERITY — concentration-independent; mucosal sites (tonsils, urinary tract, uterus) bleed more than deep tissue; haemarthrosis RARE even in severe deficiency (<1 IU/dL)",
            "F11-TRANEXAMIC-ACID-KEY: Antifibrinolytics (tranexamic acid) are FIRST-LINE treatment — mucosal bleed sites are tPA-rich → local hyperfibrinolysis; tranexamic acid often MORE EFFECTIVE than FXI concentrate for dental/ENT/GU procedures",
            "F11-ASHKENAZI-FOUNDER: Ashkenazi Jewish prevalence 1:450 — carrier frequency ~8%; screen ALL Ashkenazi patients before elective surgery; Lys521Ter (most common) + Glu117Ter (second) founder mutations",
            "F11-CONCENTRATE-THROMBOSIS-RISK: FXI concentrate (Hemoleven) is THROMBOGENIC in elderly/atherosclerotic patients — avoid in patients with prior thrombosis, MI, or stroke; FXI concentrate + antifibrinolytic together = additive thrombosis risk",
            "F11-DENTAL-MOUTHWASH: Tranexamic acid mouthwash (4.8% solution, hold 2 min, 4x daily for 5–7 days) often sufficient for dental extractions in mild-moderate FXI deficiency — avoid systemic factor if possible",
            "F11-DESMOPRESSIN-NOT-USEFUL: DDAVP does NOT effectively raise FXI levels — NOT useful for haemostasis in FXI deficiency; do not use instead of tranexamic acid or factor concentrate",
            "F11-TONSILLECTOMY-HIGH-RISK: Tonsillectomy is HIGH BLEEDING RISK in FXI deficiency regardless of plasma level — delayed primary haemorrhage common at days 5–10; preoperative FXI concentrate + antifibrinolytic required even in mild deficiency",
            "F11-ANTITHROMBOTIC-PIPELINE: FXI is a target for anticoagulation (abelacimab, osocimab, IONIS-FXIRx) — FXI knockout patients are protected from DVT but have normal haemostasis; FXI inhibitors represent next-generation anticoagulants",
        ],
        "etiologies": {
            "Lys521Ter (type II — Ashkenazi founder)": 18,
            "Glu117Ter (type III — Ashkenazi founder)": 10,
            "Compound heterozygous (Lys521+Glu117)": 6,
            "Missense (non-Ashkenazi)": 4,
            "Splice site": 2,
        },
        "stats": {
            "severe_deficiency_pct": 38,
            "ashkenazi_pct": 72,
            "mucosal_bleed_pct": 68,
            "haemarthrosis_pct": 5,
            "tonsillectomy_bleeding_pct": 45,
            "tranexamic_acid_responsive_pct": 78,
            "mean_dx_age": 28,
            "mean_dx_delay_months": 60,
        },
        "dx_delay_distribution": {"<12m": 8, "12-36m": 10, "36-60m": 12, ">60m": 10},
    },

    # ── F13A1 — Factor XIII Deficiency ────────────────────────────────
    {
        "gene": "F13A1",
        "protein": "Factor XIII-A — FXIII Deficiency, Umbilical Cord Stump Bleeding PATHOGNOMONIC, Normal Coag Screen",
        "alias": (
            "F13A1; OMIM gene 134570; Factor XIII Deficiency OMIM 613225; 6p25.1; 731 aa; ~83 kDa; "
            "AR (A subunit — most common); prevalence 1:5,000,000; consanguineous families. "
            "FXIII is a transglutaminase (zymogen A2B2 heterotetramers): A2 (catalytic, F13A1) + "
            "B2 (carrier, F13B gene). Thrombin + Ca2+ activates FXIIIa → crosslinks fibrin "
            "(Gln-Lys isopeptide bonds) → clot stabilisation + resistance to fibrinolysis. "
            "WITHOUT FXIIIa: fibrin clot forms (PT/APTT/TT normal) but is mechanically weak and "
            "rapidly lysed → bleeds despite normal coagulation screen. "
            "HALLMARK CLINICAL FEATURES: "
            "(1) UMBILICAL CORD STUMP BLEEDING — pathognomonic; 80% present in neonatal period; "
            "(2) Intracranial haemorrhage: 25% lifetime risk — HIGHEST ICH RISK of all rare coagulopathies; "
            "(3) Delayed wound healing — FXIII crosslinks fibronectin in wound matrix; "
            "(4) Recurrent miscarriage (spontaneous abortion) — FXIII crosslinks fibrin in placenta; "
            "(5) Haemarthrosis — less common than HA/HB. "
            "DIAGNOSIS: routine PT/APTT/TT/fibrinogen ALL NORMAL — must specifically test FXIII; "
            "5M urea clot solubility test (qualitative — detects severe only); quantitative FXIII assay. "
            "Treatment: plasma-derived FXIII concentrate (Fibrogammin/Corifact) or recombinant catridecacog "
            "(Novothirteen) 35 IU/kg Q4W for prophylaxis; trough FXIII >1 IU/dL prevents bleeding. "
            "ICH prophylaxis MANDATORY — one ICH can be fatal."
        ),
        "aa": "731 aa",
        "kDa": "~83 kDa",
        "locus": "6p25.1",
        "omim_gene": 134570,
        "omim_disease": 613225,
        "inheritance": "AR; A subunit (F13A1) deficiency most common; F13B deficiency rare (B-subunit carrier)",
        "gene_class": (
            "F13A1 encodes the A subunit of coagulation Factor XIII, a 731-aa transglutaminase. "
            "Domain structure: N-terminal activation peptide (cleaved by thrombin at Arg37) → "
            "β-sandwich → core (catalytic, Cys314-His373-Asp396 catalytic triad) → β-barrel 1 → β-barrel 2. "
            "In plasma: A2B2 heterotetramer (inactive); B2 carries A2, protects from proteolysis. "
            "On thrombin activation: B2 dissociates → A2 becomes FXIIIa → crosslinks fibrin α/γ chains "
            "(Gln-Lys isopeptide bonds) + α2-antiplasmin → clot stabilisation and fibrinolysis resistance. "
            "F13A1 mutations: missense (most common), frameshift/nonsense (severe, common in consanguineous). "
            "F13B mutations cause partial deficiency (reduced A2B2 heterotrimer stability)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "F13A1-UMBILICAL-CORD-STUMP-PATHOGNOMONIC: Umbilical cord stump bleeding (>7 days) is PATHOGNOMONIC for FXIII deficiency — test FXIII in ANY neonate with umbilical bleeding; do NOT attribute to local infection without FXIII result",
            "F13A1-NORMAL-COAG-SCREEN-TRAP: PT, APTT, TT, fibrinogen are ALL NORMAL in FXIII deficiency — standard coag screen will NOT detect this diagnosis; must specifically request FXIII activity assay",
            "F13A1-ICH-25PCT-LIFETIME: Intracranial haemorrhage risk is 25% lifetime — HIGHEST ICH risk of all rare coagulopathies; prophylactic FXIII concentrate Q4W is MANDATORY to prevent ICH",
            "F13A1-CATRIDECACOG-PROPHYLAXIS: Recombinant catridecacog (Novothirteen) 35 IU/kg Q4W — first recombinant FXIII product; no donor-derived pathogen risk; plasma-derived Corifact 40 IU/kg Q4-6W is alternative; target trough FXIII >1 IU/dL",
            "F13A1-MISCARRIAGE-PLACENTA: FXIII crosslinks fibrin in the placenta — severe deficiency causes recurrent first/second trimester miscarriage; FXIII concentrate during pregnancy reduces miscarriage rate significantly",
            "F13A1-WOUND-HEALING-DELAYED: Delayed wound healing (poor scar formation) is a characteristic feature — FXIII crosslinks fibronectin in extracellular matrix; surgical wounds may reopen without prophylaxis",
            "F13A1-UREA-CLOT-TEST-OBSOLETE: 5M urea clot solubility test detects only SEVERE deficiency (<5 IU/dL) — use quantitative FXIII activity assay; do not report as normal if urea clot dissolves in >24h but quantitative test pending",
            "F13A1-CONSANGUINITY-SCREEN: Most severe F13A1 deficiency occurs in consanguineous populations (Middle East, South Asia); take family history; offer cascade testing in family when proband identified",
        ],
        "etiologies": {
            "Missense (A subunit)": 20,
            "Frameshift/nonsense (consanguineous)": 12,
            "Splice site": 4,
            "Large deletion (A subunit)": 2,
            "B subunit (F13B) deficiency": 2,
        },
        "stats": {
            "umbilical_cord_bleeding_pct": 80,
            "ich_lifetime_pct": 25,
            "miscarriage_pct": 35,
            "delayed_wound_healing_pct": 60,
            "haemarthrosis_pct": 20,
            "on_prophylaxis_pct": 85,
            "mean_dx_age_months": 1,
            "mean_dx_delay_months": 2,
        },
        "dx_delay_distribution": {"<1m": 22, "1-6m": 10, "6-24m": 5, ">24m": 3},
    },

    # ── F7 — Factor VII Deficiency ────────────────────────────────────
    {
        "gene": "F7",
        "protein": "Factor VII — Most Common Rare Coagulopathy, PT Prolonged Alone, rFVIIa Treatment",
        "alias": (
            "F7; OMIM gene 613878; Factor VII Deficiency OMIM 227500; 13q34; 466 aa; ~50 kDa; "
            "AR; prevalence 1:500,000 (most common of the rare coagulopathies). "
            "FVII is a vitamin K-dependent serine protease. Tissue factor (TF) is exposed at vascular "
            "injury → TF-FVIIa complex → activates FX (extrinsic pathway) and FIX → cascade. "
            "FVII has the SHORTEST plasma half-life of all coagulation factors (4–6 hours) → "
            "first factor deficient in liver disease and vitamin K deficiency. "
            "PT ISOLATED PROLONGATION with NORMAL APTT — characteristic laboratory profile. "
            "Bleeding severity correlates POORLY with FVII plasma level (like FXI): "
            "FVII <1% → severe (ICH risk, haemarthrosis), but some patients with FVII <1% are asymptomatic. "
            "Genotype-phenotype correlation: Arg304Gln (Padua-1) → moderately reduced activity; "
            "Ala294Val (South Indian founder); Arg152Gln, Arg304Trp (severe). "
            "Treatment: recombinant FVIIa (NovoSeven) 15–30 mcg/kg Q4–6h IV — "
            "SPECIFIC for FVII deficiency; plasma-derived FVII concentrate or FFP as alternatives. "
            "Prophylaxis: weekly rFVIIa for severe patients with history of ICH or joint disease. "
            "Pregnancy: FVII rises in pregnancy (unlike most other factors) → some women manage delivery "
            "without replacement; monitor FVII at 36 weeks. "
            "Neonatal: 50% of symptomatic FVII-deficient neonates present with ICH."
        ),
        "aa": "466 aa",
        "kDa": "~50 kDa",
        "locus": "13q34",
        "omim_gene": 613878,
        "omim_disease": 227500,
        "inheritance": "AR; most common rare coagulopathy; some founder mutations (Ala294Val South Indian)",
        "gene_class": (
            "F7 encodes coagulation Factor VII, a 466-aa vitamin K-dependent serine protease. "
            "Domain structure: Gla (γ-carboxylation, membrane binding) → EGF1 → EGF2 → serine protease. "
            "TF-FVII interaction: EGF2 + serine protease domain bind TF → activation by TF-FXa/thrombin. "
            "FVII has SHORTEST half-life of coagulation factors (4–6h) — "
            "first detectable clotting factor deficiency in warfarin therapy (PT prolongation). "
            "Key mutations: Arg304Gln (Padua-1, Italian founder); Ala294Val (South Indian founder); "
            "Arg152Gln, Arg304Trp (severe, high ICH risk). Missense mutations in Gla domain: "
            "impair vitamin K-dependent carboxylation → loss of membrane-dependent activation."
        ),
        "n_patients": 40,
        "key_alerts": [
            "F7-PT-ALONE-PROLONGED: Isolated PT prolongation with NORMAL APTT is the characteristic laboratory signature of FVII deficiency — if PT long and APTT normal, CHECK FVII activity before proceeding",
            "F7-SHORTEST-HALF-LIFE: FVII has the SHORTEST plasma half-life of all coagulation factors (4–6 hours) — multiple dosing required for surgical coverage; Q4–6h rFVIIa dosing; plasma-derived FVII has 3–5h t½",
            "F7-RFVIIA-TREATMENT: Recombinant FVIIa (NovoSeven) 15–30 mcg/kg Q4–6h IV is the specific treatment — same agent used for inhibitor bypassing in HA/HB but at LOWER dose (15 mcg/kg for deficiency vs 90 mcg/kg for bypassing)",
            "F7-ICH-NEONATAL-50PCT: 50% of symptomatic FVII-deficient neonates present with ICH — screen at birth if family history; low FVII activity at birth is NOT physiological beyond the neonatal period",
            "F7-LEVEL-BLEEDING-POOR-CORRELATION: FVII plasma level correlates POORLY with bleeding severity — some FVII <1% patients are asymptomatic; bleeding history is more predictive than FVII level for surgical planning",
            "F7-PREGNANCY-RISES: FVII rises in pregnancy (reaches near-normal by third trimester in most women) — check FVII at 36 weeks; many women with mild-moderate deficiency deliver safely without replacement",
            "F7-WARFARIN-FIRST: FVII is the first factor to fall during warfarin initiation (shortest t½) — explains why PT prolongs first; therapeutic INR does NOT mean FVII <1% deficiency unless hereditary",
            "F7-PROPHYLAXIS-SEVERE: Weekly rFVIIa prophylaxis for severe FVII deficiency with prior ICH or haemarthrosis — prevents arthropathy; target trough FVII >1 IU/dL",
        ],
        "etiologies": {
            "Missense (moderate-severe)": 22,
            "Gla-domain missense (vitamin K-dependent activation loss)": 8,
            "Frameshift/nonsense (severe)": 6,
            "Arg304Gln Padua-1 Italian founder": 2,
            "Splice site": 2,
        },
        "stats": {
            "severe_pct": 30,
            "ich_neonatal_pct": 12,
            "haemarthrosis_pct": 22,
            "asymptomatic_severe_pct": 20,
            "pregnancy_safe_delivery_pct": 55,
            "on_prophylaxis_pct": 35,
            "mean_dx_age": 15,
            "mean_dx_delay_months": 18,
        },
        "dx_delay_distribution": {"<6m": 14, "6-24m": 12, "24-60m": 8, ">60m": 6},
    },

    # ── FGA — Afibrinogenemia / Dysfibrinogenemia ─────────────────────
    {
        "gene": "FGA",
        "protein": "Fibrinogen A-alpha — Afibrinogenemia, Umbilical + Intracranial + Haemarthrosis + Miscarriage, Fibrinogen Concentrate",
        "alias": (
            "FGA; OMIM gene 134820; Afibrinogenemia OMIM 202400; 4q28.1; 644 aa (Aα chain); ~95 kDa; "
            "AR; prevalence 1:1,000,000; also FGB (Bβ chain 7p14.3) and FGG (γ chain 4q28.1) deficiency. "
            "Fibrinogen (FI) is a 340-kDa hexamer: (Aα)2(Bβ)2(γ)2. Assembled in ER, secreted, "
            "circulates at 1.5–4.5 g/L. Thrombin cleaves fibrinopeptide A (from Aα) and B (from Bβ) "
            "→ fibrin monomer → polymerises → FXIII crosslinks → stable clot. "
            "Fibrinogen is REQUIRED for: fibrin clot + platelet aggregation (fibrinogen bridges GPIIb/IIIa). "
            "AFIBRINOGENEMIA (fibrinogen <0.1 g/L): severe; PT/APTT/TT ALL PROLONGED; "
            "fibrinogen assay = 0; umbilical cord bleeding; ICH (10–15% lifetime); haemarthrosis; "
            "recurrent miscarriage (fibrin critical for placentation). "
            "HYPOFIBRINOGENEMIA (fibrinogen 0.1–1.5 g/L): milder; heterozygous. "
            "DYSFIBRINOGENEMIA: fibrinogen present but dysfunctional; fibrinogen:fibrin ratio abnormal; "
            "can be haemorrhagic OR thrombotic (e.g., Aα Arg554Cys → thrombotic). "
            "Treatment: fibrinogen concentrate (Riastap/Fibryga: 70 mg/kg to target >1.5 g/L for surgery; "
            "prophylactic 50 mg/kg Q2W); FFP (low fibrinogen concentration) or cryoprecipitate as alternatives. "
            "Fibrinogen concentrate preferred: pathogen-inactivated, fixed dose, no volume overload."
        ),
        "aa": "644 aa (Aα-chain; FGA gene)",
        "kDa": "~95 kDa (Aα-chain); 340 kDa (hexamer fibrinogen)",
        "locus": "4q28.1",
        "omim_gene": 134820,
        "omim_disease": 202400,
        "inheritance": "AR for afibrinogenemia; AD for dysfibrinogenemia; FGA most common gene affected",
        "gene_class": (
            "FGA encodes fibrinogen Aα-chain, one of three chain types forming the fibrinogen hexamer (Aα)2(Bβ)2(γ)2. "
            "Key functional regions: fibrinopeptide A (N-terminus, thrombin cleavage site Arg16-Gly17) → "
            "coiled-coil connector → αC domain (C-terminus, extends beyond D domain; fibrin polymerisation + "
            "platelet GPIIb/IIIa binding + FXIII crosslinking substrate Gln398/Gln399). "
            "Mutations: nonsense/frameshift in FGA cause afibrinogenemia; missense in αC domain (Arg554Cys) "
            "causes thrombotic dysfibrinogenemia. FGB mutations: hypofibrinogenemia + hepatic fibrinogen storage "
            "disease (fibrinogen accumulates in ER → liver disease). FGG mutations: γ-chain contact site "
            "mutations → hypodysfibrinogenemia."
        ),
        "n_patients": 40,
        "key_alerts": [
            "FGA-ALL-COAG-PROLONGED: Afibrinogenemia causes PROLONGED PT + APTT + TT + clotting time — ALL prolonged unlike other single-factor deficiencies; fibrinogen assay = 0; this combination should prompt immediate fibrinogen measurement",
            "FGA-FIBRINOGEN-PLATELET-AGGREGATION: Fibrinogen is REQUIRED for platelet aggregation (bridges GPIIb/IIIa) — afibrinogenemia causes BOTH clotting defect AND platelet aggregation defect; PFA-100 also abnormal",
            "FGA-MISCARRIAGE-PLACENTA: Fibrinogen is critical for trophoblast invasion and placentation — severe afibrinogenemia causes first/second trimester miscarriage in virtually ALL untreated women; fibrinogen concentrate throughout pregnancy",
            "FGA-FIBRINOGEN-CONCENTRATE-FIRST: Fibrinogen concentrate (Riastap 70 mg/kg, Fibryga 70 mg/kg) is PREFERRED over cryoprecipitate — pathogen-inactivated, fixed dosing, no volume overload; target fibrinogen >1.5 g/L for surgery, >1.0 g/L for prophylaxis",
            "FGA-DYSFIBRINOGENEMIA-THROMBOTIC: Dysfibrinogenemia can be THROMBOTIC (not haemorrhagic) — Aα Arg554Cys is the classic thrombotic variant; measure both fibrinogen activity AND antigen; if activity:antigen ratio <0.7, test for thrombotic dysfibrinogenemia",
            "FGA-ICH-LIFETIME-10-15PCT: ICH occurs in 10–15% lifetime in afibrinogenemia — particularly in neonatal period; prophylactic fibrinogen Q2W is MANDATORY in severe deficiency",
            "FGA-CRYOPRECIPITATE-FIBRINOGEN-VARIABLE: Cryoprecipitate fibrinogen content is variable (150–300 mg/unit); requires large volumes for full correction; use fibrinogen concentrate when available for precise dosing",
            "FGA-HEPATIC-FGB-STORAGE: FGB (Bβ-chain) mutations can cause fibrinogen storage disease in hepatocytes (fibrinogen accumulates in ER → liver disease + hypofibrinogenemia) — hepatic transaminitis + hypofibrinogenemia should prompt liver biopsy",
        ],
        "etiologies": {
            "Nonsense/frameshift FGA (severe afibrinogenemia)": 18,
            "FGB missense (hypofibrinogenemia)": 8,
            "FGG missense (hypo/dysfibrinogenemia)": 6,
            "FGA missense thrombotic dysfibrinogenemia (Arg554Cys)": 4,
            "Large deletion FGA locus": 4,
        },
        "stats": {
            "umbilical_cord_bleeding_pct": 65,
            "ich_lifetime_pct": 12,
            "miscarriage_pct": 80,
            "haemarthrosis_pct": 35,
            "platelet_aggregation_defect_pct": 100,
            "on_fibrinogen_prophylaxis_pct": 78,
            "mean_dx_age_months": 1,
            "mean_dx_delay_months": 1,
        },
        "dx_delay_distribution": {"<1m": 25, "1-3m": 10, "3-12m": 3, ">12m": 2},
    },

    # ── ADAMTS13 — Congenital TTP / Upshaw-Schulman Syndrome ──────────
    {
        "gene": "ADAMTS13",
        "protein": "ADAMTS13 — Congenital TTP (Upshaw-Schulman), MAHA + Thrombocytopenia, Plasma Exchange Life-Saving",
        "alias": (
            "ADAMTS13; OMIM gene 604134; Congenital TTP / Upshaw-Schulman Syndrome OMIM 274150; "
            "9q34.11; 1427 aa; ~190 kDa; AR; prevalence 1:500,000–1,000,000; "
            "one of the most dangerous haemostatic disorders — thrombotic microangiopathy (TMA). "
            "ADAMTS13 is a metalloprotease that cleaves ultralong VWF (ULVWF) multimers in the "
            "Tyr1605-Met1606 bond of the VWF-A2 domain under flow shear stress. "
            "Without ADAMTS13: ULVWF accumulates on endothelium → platelet thrombi in microvasculature "
            "→ thrombotic microangiopathy (TMA): "
            "(1) Microangiopathic haemolytic anaemia (MAHA): schistocytes on blood film; "
            "(2) Thrombocytopenia: platelet consumption in thrombi; "
            "(3) AKI: renal microthrombi; "
            "(4) Neurological: cerebral microthrombi (confusion, seizures, coma); "
            "(5) Fever; (6) Cardiac involvement. "
            "THE PENTAD (MAHA + thrombocytopenia + AKI + neuro + fever) NOT REQUIRED for diagnosis — "
            "most patients have 2–3 features; PLASMIC score guides empirical treatment. "
            "ADAMTS13 activity <10% = TTP (congenital OR acquired immune). "
            "CONGENITAL TTP (Upshaw-Schulman): neonatal jaundice/thrombocytopenia; "
            "recurrent episodes triggered by pregnancy/infection; NO INHIBITOR (unlike acquired TTP). "
            "PLASMA EXCHANGE (PEX) is LIFE-SAVING — replaces deficient ADAMTS13. "
            "Congenital TTP: FFP infusion (25 mL/kg) — simpler than PEX; prophylactic FFP Q2–3W. "
            "CAPLACIZUMAB: anti-VWF nanobody (blocks GPIbα-VWF) — adjunct to PEX for acute TTP; "
            "reduces refractory TTP and recurrence; NOT yet standard in congenital TTP (ADAMTS13 refilling needed)."
        ),
        "aa": "1427 aa",
        "kDa": "~190 kDa",
        "locus": "9q34.11",
        "omim_gene": 604134,
        "omim_disease": 274150,
        "inheritance": "AR; compound heterozygous common; acquired immune TTP (ADAMTS13 inhibitor IgG) is different — NOT genetic",
        "gene_class": (
            "ADAMTS13 encodes a 1427-aa metalloprotease of the ADAMTS (A Disintegrin And Metalloprotease "
            "with ThromboSpondin motifs) family. "
            "Domain structure: signal → propeptide → metalloprotease (Zn-binding HEXXHXXGXXHD) → "
            "disintegrin-like → TSP1-1 → cysteine-rich → spacer → TSP1-2 to TSP1-8 → CUB1-CUB2. "
            "Catalytic cleft: cleaves Tyr1605-Met1606 in VWF-A2 domain under shear stress. "
            "Spacer domain (Glu636-Arg660 exosite): binds VWF-A2 C-terminal region; "
            "TSP1-7 and CUB domains regulate substrate recognition. "
            "Acquired TTP: anti-ADAMTS13 autoantibodies (IgG4 predominant) often target spacer exosite. "
            "Congenital mutations: missense in metalloprotease/spacer most common; "
            "nonsense/frameshift (severe); compound heterozygous in many non-consanguineous families."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ADAMTS13-PENTAD-NOT-REQUIRED: The classic TTP pentad (MAHA + thrombocytopenia + AKI + neurological + fever) is NOT required for diagnosis — most patients have only 2-3 features; start PLASMA EXCHANGE EMPIRICALLY while awaiting ADAMTS13 result",
            "ADAMTS13-PLASMA-EXCHANGE-LIFE-SAVING: Plasma exchange (PEX) replaces deficient ADAMTS13 AND removes ULVWF — MUST start within hours of TTP diagnosis; delay in PEX = significantly increased mortality; daily PEX until remission",
            "ADAMTS13-CONGENITAL-FFP-PROPHYLAXIS: Congenital TTP (Upshaw-Schulman) — prophylactic FFP 25 mL/kg Q2–3 weeks prevents recurrent episodes; pregnancy requires Q1-2W FFP; NO inhibitor antibody (unlike acquired TTP)",
            "ADAMTS13-ACTIVITY-10PCT-THRESHOLD: ADAMTS13 activity <10% distinguishes TTP from other TMAs (HUS, DIC, malignancy, drugs) — send ADAMTS13 activity in ALL suspected TMA; result may take days; do NOT wait for result to start PEX",
            "ADAMTS13-SCHISTOCYTES-FILM: Microangiopathic haemolytic anaemia (MAHA) — schistocytes (helmet/bite cells) on blood film are MANDATORY hallmark; negative Coombs test (DAT negative) differentiates from autoimmune haemolysis",
            "ADAMTS13-CAPLACIZUMAB-ACUTE: Caplacizumab (anti-VWF nanobody, blocks GPIbα) is adjunct to PEX for acute acquired TTP — reduces refractory TTP and recurrence; NOT standard monotherapy; ADAMTS13 activity must recover before stopping caplacizumab (rebound TTP risk)",
            "ADAMTS13-PREGNANCY-TRIGGER: Pregnancy is the most common trigger for first presentation of congenital TTP (Upshaw-Schulman) — neonatal jaundice/thrombocytopenia is FIRST PRESENTATION in many; VWF levels rise in pregnancy → overwhelms residual ADAMTS13",
            "ADAMTS13-INHIBITOR-DISTINGUISH: Congenital TTP: ADAMTS13 activity low, INHIBITOR ABSENT; Acquired TTP: ADAMTS13 activity low + IgG INHIBITOR present → add rituximab + corticosteroids to PEX; critical distinction for treatment",
        ],
        "etiologies": {
            "Missense (metalloprotease/spacer domain)": 18,
            "Compound heterozygous missense": 10,
            "Nonsense/frameshift (severe)": 8,
            "Splice site": 2,
            "CUB domain missense (secretion impaired)": 2,
        },
        "stats": {
            "maha_at_presentation_pct": 100,
            "thrombocytopenia_at_presentation_pct": 100,
            "aki_pct": 55,
            "neurological_pct": 60,
            "neonatal_presentation_pct": 35,
            "pregnancy_triggered_pct": 40,
            "pex_responsive_pct": 88,
            "mean_dx_age": 8,
            "mean_dx_delay_months": 6,
        },
        "dx_delay_distribution": {"<1m": 18, "1-6m": 12, "6-24m": 6, ">24m": 4},
    },
]


def _make_patients(gene_entry, rng):
    """Generate synthetic patient records for one gene."""
    gene = gene_entry["gene"]
    n = gene_entry["n_patients"]
    ages = [rng.randint(0, 65) for _ in range(n)]
    delays = [rng.choice([1, 3, 6, 12, 24, 36, 48, 60]) for _ in range(n)]
    etiol_keys = list(gene_entry["etiologies"].keys())
    etiol_weights = list(gene_entry["etiologies"].values())
    etiols = rng.choices(etiol_keys, weights=etiol_weights, k=n)
    patients = []
    for i in range(n):
        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "dx_age": ages[i],
            "dx_delay_months": delays[i],
            "variant_class": etiols[i],
        })
    return patients


def _build_cohort():
    all_data = {}
    for idx, ge in enumerate(COAGULATION_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        ge_copy = dict(ge)
        ge_copy["seed"] = seed
        ge_copy["patients"] = _make_patients(ge, rng)
        all_data[ge["gene"]] = ge_copy
    return all_data


_COHORT = _build_cohort()


def get_overview():
    genes_summary = []
    total = 0
    all_dx_ages = []
    all_delays = []
    top_alerts = []
    for gene, info in _COHORT.items():
        n = info["n_patients"]
        total += n
        pts = info["patients"]
        ages = [p["dx_age"] for p in pts]
        delays = [p["dx_delay_months"] for p in pts]
        all_dx_ages.extend(ages)
        all_delays.extend(delays)
        genes_summary.append({
            "gene": gene,
            "protein_short": info["protein"].split(" — ")[0],
            "n_patients": n,
            "locus": info["locus"],
            "inheritance": info["inheritance"].split(";")[0],
            "omim_disease": info["omim_disease"],
            "mean_dx_age": round(sum(ages) / len(ages), 1),
        })
        top_alerts.extend(info["key_alerts"][:2])
    aggregate_stats = {
        "total_patients": total,
        "mean_dx_age_years": round(sum(all_dx_ages) / len(all_dx_ages), 1),
        "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
        "f8_inhibitor_pct": 28,
        "f8_emicizumab_prophylaxis_pct": 55,
        "vwf_ddavp_responsive_pct": 65,
        "f11_tranexamic_acid_responsive_pct": 78,
        "f13a1_ich_lifetime_pct": 25,
        "f7_ich_neonatal_pct": 12,
        "fga_miscarriage_pct": 80,
        "adamts13_pex_responsive_pct": 88,
        "cascade_tested_pct": 72,
    }
    return {
        "atlas": "Hereditary-Coagulation-Atlas",
        "genes": genes_summary,
        "aggregate_stats": aggregate_stats,
        "top_alerts": top_alerts,
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
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
        "atlas": "Hereditary-Coagulation-Atlas",
        "concepts": {
            "Haemophilia A and B (X-Linked Recessive)": (
                "Haemophilia A (F8, Xq28) and B (F9, Xq27.1) are X-linked recessive coagulopathies. "
                "HA: FVIII deficiency, 1:5000–10000 males, most common inherited coagulopathy. "
                "HB: FIX deficiency (Christmas disease), 1:25000–30000 males. "
                "Both: severity by factor level (<1% severe, 1–5% moderate, 5–40% mild). "
                "Emicizumab (bispecific FVIII-mimetic, SC Q1-4W) has transformed HA prophylaxis (FDA 2017–2018). "
                "Etranacogene dezaparvovec (gene therapy, FIX-Padua AAV5) for HB (FDA 2022). "
                "Inhibitors (FVIII neutralising IgG): 25–30% severe HA; <5% HB (with anaphylaxis risk). "
                "HB Leyden: promoter mutations → severe in childhood, recovery after puberty."
            ),
            "von Willebrand Disease (vWD) Subtypes": (
                "vWD is the most common inherited bleeding disorder (1:100–1000). "
                "VWF (2813 aa, 12p13.31) carries FVIII and mediates platelet adhesion (GPIbα under shear). "
                "Type 1 (quantitative partial, AD): DDAVP effective → releases VWF from Weibel-Palade bodies. "
                "Type 2A (qualitative — loss of HMWM): DDAVP variably effective; test with trial dose. "
                "Type 2B (GOF — spontaneous GPIbα binding → platelet consumption → thrombocytopenia): "
                "DDAVP CONTRAINDICATED — worsens thrombocytopenia. "
                "Type 2N (Normandy — VWF-FVIII binding defect): mimics mild HA; treat with VWF concentrate. "
                "Type 3 (complete VWF deficiency, AR): severe; DDAVP ineffective; VWF concentrate mandatory. "
                "Blood group O: VWF 25% lower; may cause type 1-range vWD without VWF gene mutation."
            ),
            "Factor XIII Deficiency and Fibrinogen Disorders": (
                "FXIII (F13A1, 6p25.1) is a transglutaminase that crosslinks fibrin. "
                "Without FXIIIa: PT/APTT/TT ALL NORMAL but clot is mechanically fragile → delayed dissolution. "
                "Hallmarks: umbilical cord stump bleeding (PATHOGNOMONIC), ICH (25% lifetime — highest of all rare coagulopathies), miscarriage. "
                "Diagnosis: MUST specifically request FXIII activity — not detected by standard coag screen. "
                "Afibrinogenemia (FGA/FGB/FGG, AR): fibrinogen = 0; ALL coag tests prolonged; "
                "platelet aggregation also impaired (fibrinogen bridges GPIIb/IIIa). "
                "Fibrinogen concentrate (Riastap/Fibryga 70 mg/kg) is the treatment of choice — "
                "pathogen-inactivated, fixed dose; target >1.5 g/L for surgery."
            ),
            "Factor XI Deficiency (Haemophilia C) — Mucosal Bleeding Paradox": (
                "F11 (FXI, 625 aa, 4q35.2, AR) — activated by FXIIa and thrombin → amplifies FIX activation. "
                "Concentration-independent bleeding: plasma FXI level DOES NOT predict clinical severity. "
                "Mucosal sites (tonsils, GU tract, uterus) bleed MORE than expected; haemarthrosis is RARE. "
                "Mechanism: mucosal sites are tPA-rich → local hyperfibrinolysis drives bleeding independent of FXI. "
                "Treatment: tranexamic acid is FIRST-LINE (often more effective than FXI concentrate). "
                "FXI concentrate (Hemoleven) is THROMBOGENIC in elderly — caution in cardiovascular disease. "
                "Ashkenazi Jewish prevalence 1:450 — Lys521Ter + Glu117Ter founder mutations."
            ),
            "Factor VII Deficiency — Shortest Half-Life": (
                "F7 (FVII, 466 aa, 13q34, AR) — most common of the rare coagulopathies (1:500,000). "
                "FVII has the SHORTEST plasma half-life of all coagulation factors (4–6 hours). "
                "Laboratory: isolated PT prolongation with NORMAL APTT — characteristic signature. "
                "Bleeding-level correlation is POOR: some FVII <1% are asymptomatic. "
                "Treatment: recombinant FVIIa (NovoSeven) 15–30 mcg/kg Q4–6h — same drug as inhibitor bypass "
                "therapy but at lower dose. FVII rises in pregnancy — many women manage delivery without replacement."
            ),
            "Congenital TTP (ADAMTS13 / Upshaw-Schulman Syndrome)": (
                "ADAMTS13 (1427 aa, 9q34.11, AR) cleaves ULVWF multimers at Tyr1605-Met1606 under shear. "
                "Deficiency → ULVWF accumulation → platelet microthrombi = thrombotic microangiopathy (TMA). "
                "Congenital TTP (Upshaw-Schulman): AR, recurrent episodes, NO inhibitor antibody. "
                "Acquired TTP: anti-ADAMTS13 IgG antibody — treat with PEX + rituximab + steroids. "
                "ADAMTS13 activity <10% distinguishes TTP from other TMAs (HUS, DIC, malignancy). "
                "TTP PENTAD not required for diagnosis — MAHA + thrombocytopenia + any 1 organ = treat. "
                "Plasma exchange MUST begin within hours — daily until remission. "
                "Congenital TTP: prophylactic FFP Q2–3W; pregnancy requires Q1–2W FFP infusion."
            ),
            "Bypassing Agents and Modern Haemostatic Therapies": (
                "Bypassing agents (for inhibitor HA/HB): rFVIIa (NovoSeven, 90 mcg/kg Q2–3h) or "
                "aPCC (FEIBA, 50–100 IU/kg). Both bypass FVIII/FIX need by activating downstream coagulation. "
                "Emicizumab: FVIII-mimetic bispecific antibody (FIXa × FX); SC prophylaxis; not a clotting factor. "
                "Fitusiran (siRNA → reduces antithrombin): non-factor prophylaxis for all haemophilia types. "
                "Caplacizumab (anti-VWF nanobody, blocks GPIbα): acute acquired TTP adjunct to PEX. "
                "Gene therapy: valoctocogene roxaparvovec (HA, EMA 2022), etranacogene dezaparvovec (HB, FDA 2022). "
                "Antifibrinolytics (tranexamic acid, ε-aminocaproic acid): block plasminogen activation → "
                "inhibit fibrinolysis; first-line for FXI deficiency and vWD mucosal bleeds."
            ),
        },
        "pharmacological_distinctions": [
            "Emicizumab (Hemlibra) SC Q1/2/4W — bispecific FIXa×FX antibody, FVIII-mimetic — for HA with/without inhibitors; does NOT replace FVIII for acute bleeds or surgery; AVOID aPCC concurrent (thromboembolism); rFVIIa can be used with emicizumab (low dose)",
            "rFVIIa (NovoSeven): 90 mcg/kg Q2–3h for HA/HB inhibitor bleeds vs 15–30 mcg/kg Q4–6h for congenital FVII deficiency — DIFFERENT DOSE RANGES; always specify indication",
            "DDAVP (desmopressin): effective in type 1 vWD + mild HA (releases FVIII + VWF from endothelium); CONTRAINDICATED type 2B vWD (worsens thrombocytopenia); NOT useful for HB, FXI deficiency, or type 3 vWD",
            "Fibrinogen concentrate (Riastap 70 mg/kg, Fibryga 70 mg/kg) vs cryoprecipitate: concentrate has fixed fibrinogen content, pathogen-inactivated, preferred for hereditary afibrinogenemia; cryoprecipitate variable content, used when concentrate unavailable",
            "Tranexamic acid: inhibits plasminogen→plasmin conversion at lysine-binding sites; FIRST-LINE for FXI deficiency mucosal bleeds, vWD mucosal bleeds, dental procedures; caution with FXI concentrate (additive thrombosis risk); avoid in haematuria (obstructive clot in ureters)",
            "FXIII concentrate: catridecacog (Novothirteen) recombinant A-subunit 35 IU/kg Q4W; Corifact (plasma-derived) 40 IU/kg Q4–6W; target trough FXIII >1 IU/dL; prophylaxis MANDATORY to prevent ICH (25% lifetime risk without prophylaxis)",
            "Plasma exchange (PEX) for TTP: replaces deficient ADAMTS13 AND removes ULVWF + autoantibodies (in acquired TTP); 1–1.5 plasma volumes daily until remission (platelets >150×10⁹/L for 2 days + LDH normalising); caplacizumab reduces time to platelet recovery",
            "Caplacizumab (anti-VWF A1 domain nanobody): blocks VWF-GPIbα interaction → prevents platelet microthrombus formation in TTP; FDA 2019 for adults with acquired TTP; continue 30 days post-PEX; RISK of rebound TTP if stopped before ADAMTS13 recovers >10%",
        ],
        "key_standards": [
            "World Federation of Hemophilia (WFH) Guidelines for Management of Hemophilia 4th Ed 2020 — Srivastava A et al., Haemophilia 2020; prophylaxis, ITI, emicizumab integration, gene therapy eligibility",
            "UKHCDO Haemophilia Standards 2023 — emicizumab prophylaxis for severe HA (with/without inhibitors); pharmacokinetic-guided factor dosing; inhibitor testing schedule (<50 exposure days)",
            "von Willebrand Disease International Consensus Statement — Leebeek FWG et al., Blood 2021; DDAVP trial dosing; type 2B DDAVP contraindication; blood group O reference range adjustment",
            "International Society on Haemostasis and Thrombosis (ISTH) TTP Guidelines — Scully M et al., Res Pract Thromb Haemost 2020; PLASMIC score; ADAMTS13 <10% = TTP; immediate PEX; caplacizumab adjunct",
            "European Haematology Association (EHA) / ISTH Rare Bleeding Disorders — Peyvandi F et al., Haemophilia 2012; factor VII, X, XI, XIII deficiencies — diagnosis, bleeding score, treatment targets",
            "FXIII Deficiency International Consensus — Ivaskevicius V et al., J Thromb Haemost 2007; umbilical cord stump bleeding; ICH prophylaxis mandatory; catridecacog (ESMO 2015); 5M urea clot test limitations",
            "Afibrinogenemia Fibrinogen Concentrate Use — de Moerloose P et al., Haemophilia 2013; Riastap/Fibryga pharmacokinetics; target fibrinogen levels (>1.5 g/L surgery, >1.0 g/L minor bleed); pregnancy management",
            "Congenital TTP (Upshaw-Schulman) — Lotta LA et al., Haematologica 2012; ADAMTS13 genetic spectrum; FFP prophylaxis Q2–3W; pregnancy management (Q1–2W FFP); neonatal presentation pattern",
        ],
    }
