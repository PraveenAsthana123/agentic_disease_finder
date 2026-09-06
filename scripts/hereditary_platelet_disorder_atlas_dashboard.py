#!/usr/bin/env python3
"""Hereditary-Platelet-Disorder-Atlas — Complete 8-Gene Hereditary Platelet Disorder Atlas
ITGA2B  (Integrin subunit alpha IIb / GPIIb; 1176 aa; 17q21.31; AR;
         Glanzmann Thrombasthenia type 1 — absent αIIbβ3 — absent clot retraction PATHOGNOMONIC;
         seed SEED_BASE+0) ·
ITGB3   (Integrin subunit beta 3 / GPIIIa; 788 aa; 17q21.32; AR;
         Glanzmann Thrombasthenia type 2 — HPA-1a antigen — NAIT in HPA-1b/1b mothers;
         seed SEED_BASE+1) ·
GP1BA   (Glycoprotein Ib platelet subunit alpha / GPIbα; 626 aa; 17p13.2; AR;
         Bernard-Soulier Syndrome — MACRO-THROMBOCYTOPENIA PATHOGNOMONIC — absent ristocetin aggregation;
         seed SEED_BASE+2) ·
GP1BB   (Glycoprotein Ib platelet subunit beta / GPIbβ; 206 aa; 22q11.21; AR;
         Bernard-Soulier Syndrome type B — 22q11.2 deletion overlap — DiGeorge thrombocytopenia;
         seed SEED_BASE+3) ·
MYH9    (Myosin heavy chain 9 / non-muscle myosin IIA; 1960 aa; 22q12.3; AD;
         MYH9-Related Disease — Döhle-like neutrophil inclusions PATHOGNOMONIC — NOT ITP;
         seed SEED_BASE+4) ·
ANKRD26 (Ankyrin repeat domain 26; 1710 aa; 10p12.1; AD;
         Thrombocytopenia 2 — 5′UTR variants WES-MISS — AML/MDS 8% risk;
         seed SEED_BASE+5) ·
ETV6    (ETS variant transcription factor 6; 452 aa; 12p13.2; AD;
         ETV6-related Thrombocytopenia — ALL predisposition 25–35% — donor screening MANDATORY;
         seed SEED_BASE+6) ·
RUNX1   (RUNX family transcription factor 1; 480 aa; 21q22.12; AD;
         FPD-AML — δ-granule deficiency — AML/MDS 35–44% lifetime — donor screening MANDATORY;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1606–1613)
"""

import random

SEED_BASE = 1606

PLATELET_GENES = [
    # ── ITGA2B — Glanzmann Thrombasthenia type 1 ─────────────────────────
    {
        "gene": "ITGA2B",
        "protein": "ITGA2B — Glanzmann Thrombasthenia Type 1, AR, αIIb Integrin, Absent Clot Retraction PATHOGNOMONIC",
        "alias": (
            "ITGA2B; OMIM gene 607759; Glanzmann Thrombasthenia (GT) OMIM 187800; "
            "17q21.31; 1176 aa; ~136 kDa (processed αIIb heavy chain ~110 kDa + light chain ~22 kDa); "
            "AR (biallelic); prevalence 1:1,000,000 general population; higher in inbred populations "
            "(Roma/Gypsy, South Indian, Iraqi Jews, French Manouche gypsies). "
            "ITGA2B encodes the αIIb subunit (GPIIb, CD41) of the platelet integrin αIIbβ3 complex "
            "(also known as GPIIb/IIIa, fibrinogen receptor, CD41/CD61). "
            "αIIbβ3 is the dominant platelet surface receptor (~80,000 copies per platelet); "
            "when activated by inside-out signalling, αIIbβ3 binds fibrinogen, VWF, fibronectin, vitronectin → "
            "platelet aggregation and thrombus formation. "
            "Loss of αIIb (or β3) → absent αIIbβ3 surface expression → complete failure of platelet aggregation. "
            "CLASSIFICATION: "
            "Type I (classic) — <5% residual αIIbβ3; absent to all agonists except ristocetin (VWF pathway intact); "
            "Type II — 5–20% residual αIIbβ3; severe but milder than Type I; "
            "Type III (variant) — ≥20% normal surface expression but dysfunctional (qualitative defect). "
            "PHENOTYPE: mucocutaneous bleeding — epistaxis (90%), menorrhagia (severe in females), "
            "gingival bleeding, petechiae, purpura; GI bleeding; post-operative haemorrhage; "
            "platelet COUNT and morphology NORMAL (not macro-thrombocytopenic — key DDx from BSS). "
            "CLOT RETRACTION ABSENT: in a standard glass tube, normal blood clots retract over 1h "
            "(platelet αIIbβ3 pulls on fibrin); in GT, clot retraction is ABSENT — PATHOGNOMONIC bedside sign. "
            "PLATELET AGGREGATION STUDIES: absent aggregation to ADP, collagen, thrombin, arachidonic acid, "
            "epinephrine; NORMAL aggregation/agglutination to ristocetin (GPIb-VWF axis intact — "
            "key DDx from Bernard-Soulier where ristocetin fails). "
            "FLOW CYTOMETRY: absent/reduced CD41 (anti-GPIIb) + CD61 (anti-GPIIIa) — gold standard confirmation. "
            "TREATMENT: local measures (pressure, tranexamic acid) for mild bleeding; "
            "platelet transfusions for major bleeding (risk: alloimmunisation to HPA antigens → inhibitory antibodies). "
            "ANTI-αIIbβ3 INHIBITORS: ~15% of multi-transfused GT patients develop anti-αIIbβ3 antibodies → "
            "platelet transfusion refractory → rFVIIa (NovoSeven) indicated; activates extrinsic pathway on platelet surface, "
            "partially bypassing αIIbβ3 need for clot formation. "
            "GENE THERAPY: lentiviral GT gene therapy in clinical trials (UCL/Paris); "
            "successful engraftment expected to cure. "
            "SURGICAL/PROCEDURAL PLANNING: tranexamic acid pre-procedure mandatory; "
            "cross-match HPA-compatible platelets in advance for elective procedures; "
            "rFVIIa as rescue; avoid NSAIDs, antiplatelet agents absolutely."
        ),
        "aa": "1176 aa",
        "kDa": "~136 kDa",
        "locus": "17q21.31",
        "omim_gene": 607759,
        "omim_disease": 187800,
        "inheritance": "AR biallelic; >200 pathogenic variants; frameshift, missense, splice; high consanguinity populations",
        "gene_class": (
            "ITGA2B encodes integrin subunit αIIb (GPIIb, CD41). "
            "Domain structure: signal peptide → β-propeller domain (7 bladed, ligand binding, N-terminal) → "
            "thigh domain → calf-1 → calf-2 → transmembrane helix → short cytoplasmic tail. "
            "αIIb heterodimerises co-translationally with β3 (ITGB3) in the ER → transport to Golgi → "
            "proteolytic processing of αIIb propeptide → mature heavy chain (110 kDa) + light chain (22 kDa) "
            "connected by disulfide bond → cell surface as αIIbβ3. "
            "Missense variants in β-propeller most frequently prevent αIIb/β3 heterodimerisation → "
            "ER retention + proteasomal degradation → absent surface expression (GT Type I/II). "
            "ITGA2B mutations account for ~80% of GT alleles worldwide; ITGB3 ~20%."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ITGA2B-GT-CLOT-RETRACTION-PATHOGNOMONIC: Absent clot retraction is a PATHOGNOMONIC bedside sign of Glanzmann Thrombasthenia — place 1 mL whole blood in a plain glass tube; in normal blood, clot retracts to ~50% volume within 1 hour; in GT, clot does NOT retract (αIIbβ3 required for fibrin-myosin contraction); this test can be performed at the bedside without specialist equipment",
            "ITGA2B-GT-NORMAL-PLATELET-COUNT: In Glanzmann Thrombasthenia, the platelet COUNT and MORPHOLOGY are NORMAL — do NOT confuse with Bernard-Soulier (macro-thrombocytopenia) or ITP; the defect is purely in platelet FUNCTION (absent aggregation), not platelet number or size; a normal CBC with severe mucocutaneous bleeding should prompt platelet function testing",
            "ITGA2B-GT-RISTOCETIN-NORMAL: Platelet aggregation to ristocetin is NORMAL in GT (VWF-GPIb axis intact) — this is the KEY diagnostic discriminator from Bernard-Soulier syndrome (where ristocetin response is absent/reduced); aggregation to ALL other agonists (ADP, collagen, thrombin, arachidonic acid, epinephrine) is absent in GT",
            "ITGA2B-GT-INHIBITOR-RISK: ~15% of multi-transfused GT patients develop alloantibodies against HPA (Human Platelet Antigen) epitopes on αIIbβ3 → platelet transfusion refractory — establish this EARLY; maintain a register of HPA-typed donors; switch to rFVIIa (recombinant factor VIIa, NovoSeven 90–120 mcg/kg Q2h) for breakthrough/surgical bleeding in inhibitor patients",
            "ITGA2B-GT-RFVIIA-PROTOCOL: rFVIIa (recombinant factor VIIa) is effective in GT with inhibitors AND in uninhibited GT for major surgical bleeding — dose 90–120 mcg/kg IV every 2 hours; rFVIIa activates extrinsic coagulation on phosphatidylserine-expressing platelet surface, generating thrombin burst sufficient for haemostasis without requiring αIIbβ3; approved by EMA for GT",
            "ITGA2B-GT-TRANEXAMIC-ACID-ALWAYS: Tranexamic acid is the cornerstone adjunct in ALL GT bleeding episodes — 10–25 mg/kg IV (or 1–1.5 g oral) for mucosal/post-operative bleeding; inhibits fibrinolysis by blocking plasminogen activation; mandatory pre-operatively and for menorrhagia management; combine with oral contraceptives for menorrhagia control in adolescent females",
            "ITGA2B-GT-PREGNANCY-HIGH-RISK: GT pregnancies require specialist haemostasis management — neonatal GT (50% chance if partner carrier); neonatal thrombocytopenia risk from maternal anti-platelet antibodies; plan elective caesarean with platelet/rFVIIa cover; neonatal cord blood platelet function testing at delivery; maternal anti-GPIIb/IIIa antibodies can cross placenta causing fetal intracranial haemorrhage",
            "ITGA2B-GT-AVOID-ANTIPLATELETS: Absolutely CONTRAINDICATE aspirin, NSAIDs, P2Y12 inhibitors (clopidogrel, ticagrelor), and GP IIb/IIIa antagonists (abciximab, eptifibatide, tirofiban) in GT — these agents further impair residual platelet function and cause life-threatening haemorrhage",
        ],
        "etiologies": {
            "Missense in β-propeller domain — ER retention, Type I GT": 14,
            "Frameshift/nonsense — truncation, Type I GT": 10,
            "Splice-site variant — exon skipping, Type I/II GT": 7,
            "Roma/Gypsy founder mutation (IVS15+1G>A)": 4,
            "South Indian founder missense (Ser843Leu)": 3,
            "Qualitative variant — normal expression, dysfunctional (Type III)": 2,
        },
        "stats": {
            "mean_dx_age_y": 2.8,
            "mean_dx_delay_months": 7.2,
            "pct_inhibitors_developed": 15,
            "pct_rfviia_used": 22,
            "pct_menorrhagia_females": 88,
            "pct_epistaxis": 90,
            "pct_clot_retraction_absent": 100,
        },
        "dx_delay_distribution": {"<3 m": 18, "3–12 m": 12, "1–3 y": 7, ">3 y": 3},
    },
    # ── ITGB3 — Glanzmann Thrombasthenia type 2 / NAIT ───────────────────
    {
        "gene": "ITGB3",
        "protein": "ITGB3 — Glanzmann Thrombasthenia Type 2, AR, β3 Integrin, HPA-1a/PlA1 Antigen, NAIT Risk",
        "alias": (
            "ITGB3; OMIM gene 173470; Glanzmann Thrombasthenia type 2 (GT2) OMIM 187800 allelic; "
            "Neonatal Alloimmune Thrombocytopenia (NAIT) — HPA-1a; 17q21.32; 788 aa; ~88 kDa; "
            "AR (biallelic) for GT; AD / alloimmune for NAIT. "
            "ITGB3 encodes the β3 integrin subunit (GPIIIa, CD61). β3 pairs with: "
            "(1) αIIb (ITGA2B) → αIIbβ3 (GPIIb/IIIa) on platelets — fibrinogen receptor; "
            "(2) αV (ITGAV) → αVβ3 (vitronectin receptor) — on endothelium, osteoclasts, platelets. "
            "Biallelic ITGB3 mutations → absent αIIbβ3 + absent αVβ3 → GT phenotype (same as ITGA2B-GT). "
            "GT due to ITGB3: accounts for ~20% of all GT alleles; phenotype clinically identical to ITGA2B-GT; "
            "distinguishable only by flow cytometry (both CD41 and CD61 reduced — since αIIb requires β3 to traffic). "
            "HPA-1a ANTIGEN (PlA1): ITGB3 carries the Human Platelet Antigen 1 (HPA-1) polymorphism. "
            "HPA-1a (PlA1, Leu33) is dominant allele (98% of Europeans). "
            "HPA-1b (PlA2, Pro33) frequency ~2% Europeans. "
            "NEONATAL ALLOIMMUNE THROMBOCYTOPENIA (NAIT) — HPA-1a is the most common cause: "
            "HPA-1b/1b mother exposed to HPA-1a fetal platelets → anti-HPA-1a IgG → crosses placenta → "
            "fetal/neonatal platelet destruction → profound thrombocytopenia (<50 × 10⁹/L in 50%). "
            "NAIT RISK: 1:1000 pregnancies; 25% FIRST pregnancy (sensitisation in utero, unlike Rh-HDN where risk starts 2nd); "
            "INTRACRANIAL HAEMORRHAGE (ICH) — most severe complication; occurs 10–20% of untreated NAIT; "
            "10–20% of ICH occur IN UTERO (before birth) — hence EARLY fetal treatment recommended. "
            "DIAGNOSIS OF NAIT: maternal anti-HPA antibody detected; neonatal CBC (severe thrombocytopenia); "
            "platelet genotyping of parents (HPA-1, HPA-5, HPA-3 panels — HPA-5b second most common). "
            "MANAGEMENT OF NEONATAL NAIT: HPA-compatible platelet transfusion (HPA-1b/1b donor) → "
            "rapid platelet increment; IVIG (1 g/kg/day × 2) raises platelets in 24–48h; "
            "hydrocortisone adjunct in refractory cases. "
            "PREVENTION IN SUBSEQUENT PREGNANCY: maternal IVIG weekly from 16–20 weeks gestation + "
            "close fetal surveillance; some centres add corticosteroids; "
            "caesarean section for unresolved thrombocytopenia to avoid birth-canal trauma ICH. "
            "GT TYPE 2 (biallelic ITGB3): same management as ITGA2B-GT (tranexamic acid, platelet transfusion, "
            "rFVIIa for inhibitors, gene therapy in development)."
        ),
        "aa": "788 aa",
        "kDa": "~88 kDa",
        "locus": "17q21.32",
        "omim_gene": 173470,
        "omim_disease": 187800,
        "inheritance": "AR biallelic for GT; HPA-1b/1b maternal phenotype for NAIT (not Mendelian disease in strict sense)",
        "gene_class": (
            "ITGB3 encodes β3 integrin (GPIIIa, CD61). "
            "Domain structure: signal peptide → PSI domain → hybrid domain → βA domain (von Willebrand A; "
            "metal ion-dependent adhesion site MIDAS; cation binding) → EGF domains (1–4) → "
            "β-tail domain → transmembrane helix → cytoplasmic tail (NPxY motifs — talin/kindlin binding). "
            "HPA-1a/1b polymorphism: Leu33Pro (rs5918) in the PSI domain — ProLeu dimorphism at position 33. "
            "In GT, ITGB3 missense/truncating mutations → impaired folding → absent αIIbβ3 complex. "
            "αVβ3 is also absent on platelets and endothelium in ITGB3-GT (broader integrin deficit than ITGA2B-GT)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ITGB3-NAIT-FIRST-PREGNANCY: Neonatal alloimmune thrombocytopenia (NAIT) due to anti-HPA-1a occurs in the FIRST pregnancy — unlike Rhesus haemolytic disease where risk builds across pregnancies; 25% of NAIT ICH occurs before birth (in utero); do NOT assume the first pregnancy is safe — screen maternal anti-HPA-1a in all pregnancies with unexplained thrombocytopenic neonates",
            "ITGB3-NAIT-HPA1B1B-DIAGNOSIS: Diagnosis requires: (1) neonatal severe thrombocytopenia <50 × 10⁹/L; (2) maternal anti-HPA-1a antibody (or anti-HPA-5b second most common); (3) parental HPA genotyping confirming mismatch; platelet COUNT alone does NOT distinguish NAIT from NTP (neonatal thrombocytopenia from other causes); anti-HPA antibody testing must be performed urgently",
            "ITGB3-NAIT-HPA-COMPATIBLE-PLATELETS: Treat severe NAIT (platelets <30 × 10⁹/L or active bleeding) with HPA-1b/1b (HPA-1a-negative) platelet transfusion — incompatible platelets (HPA-1a+) will be destroyed immediately and are contraindicated; if HPA-compatible platelets unavailable, maternal platelets (washed, irradiated) are second choice; IVIG 1 g/kg/day × 2 in parallel",
            "ITGB3-NAIT-IVIG-SECOND-PREGNANCY: In a woman with a prior NAIT-affected infant, give prophylactic IVIG 1 g/kg/week from 16 weeks gestation to suppress anti-HPA-1a antibody titres; add dexamethasone if antibody titres remain high; plan caesarean section if fetal platelet count <50 × 10⁹/L on fetal blood sampling; this prevents in utero ICH",
            "ITGB3-GT-TYPE2-VS-TYPE1: GT Type 2 (ITGB3) is clinically identical to GT Type 1 (ITGA2B); distinguish by flow cytometry — in ITGB3-GT, both CD61 (β3) and CD41 (αIIb) are absent (αIIb requires β3 for surface trafficking); in ITGA2B-GT, CD41 absent but CD61 may be partially present on αVβ3; genetic testing resolves — important for gene therapy eligibility and NAIT risk counselling",
            "ITGB3-HPA-GENOTYPING-MANDATORY: HPA genotyping (HPA-1, -3, -5 minimum panel) is mandatory in ALL women with a NAIT-affected infant or unexplained fetal thrombocytopenia — result determines recurrence risk and prophylaxis intensity for future pregnancies; partner HPA typing identifies zygosity (heterozygous father = 50% chance HPA-1a− fetus)",
            "ITGB3-ALPHAVIIBETA3-ALSO-ABSENT: Unlike ITGA2B-GT, ITGB3-GT also abolishes αVβ3 (vitronectin receptor) on platelets and endothelium — theoretically impacts bone density (osteoclast αVβ3 function) and wound angiogenesis; not clinically prominent but relevant for bone surveillance in long-term follow-up and distinguishing from ITGA2B-GT",
            "ITGB3-ANTIPLATELET-DRUG-CONTRAINDICATION: Anti-αIIbβ3 drugs (abciximab, eptifibatide, tirofiban) are ABSOLUTELY CONTRAINDICATED in ITGB3-GT — they further block residual αIIbβ3 function; abciximab (anti-GPIIb/IIIa monoclonal fragment) may also trigger immune sensitisation; avoid all antiplatelet agents in any GT diagnosis",
        ],
        "etiologies": {
            "Missense in βA domain — MIDAS disruption, absent αIIbβ3": 12,
            "Frameshift/nonsense — truncation, absent surface expression": 9,
            "Splice-site variant — exon skipping, Type I/II GT": 8,
            "HPA-1b/1b maternal genotype — NAIT (not GT in mother)": 7,
            "Deletion/rearrangement (MLPA)": 3,
            "Qualitative variant — normal expression, dysfunctional (Type III)": 1,
        },
        "stats": {
            "mean_dx_age_y": 3.4,
            "mean_dx_delay_months": 8.1,
            "pct_nait_index_case": 18,
            "pct_nait_ich_risk": 15,
            "pct_inhibitors_gt": 12,
            "pct_ivig_nait_response": 82,
        },
        "dx_delay_distribution": {"<3 m": 16, "3–12 m": 13, "1–3 y": 8, ">3 y": 3},
    },
    # ── GP1BA — Bernard-Soulier Syndrome type A1 ─────────────────────────
    {
        "gene": "GP1BA",
        "protein": "GP1BA — Bernard-Soulier Syndrome, AR, Giant Platelets PATHOGNOMONIC, Absent Ristocetin Aggregation",
        "alias": (
            "GP1BA; OMIM gene 606672; Bernard-Soulier Syndrome (BSS) OMIM 231200; "
            "17p13.2; 626 aa; ~170 kDa (mature form with O-glycosylation); AR (biallelic); "
            "prevalence <1:1,000,000; rare worldwide, enriched in Middle Eastern populations. "
            "GP1BA encodes GPIbα (glycoprotein Ibα), the principal subunit of the GPIb-IX-V complex. "
            "GPIb-IX-V COMPLEX: GPIbα (GP1BA) + GPIbβ (GP1BB) disulfide-linked → non-covalent association with "
            "GPIX (GP9) and GPV (GP5); ~25,000 copies per platelet surface. "
            "GPIbα FUNCTION: "
            "(1) VON WILLEBRAND FACTOR RECEPTOR — GPIbα N-terminal domain binds VWF A1 domain under high shear → "
            "platelet tethering and rolling on exposed subendothelium → PRIMARY HAEMOSTASIS. "
            "(2) Thrombin binding site (anion-binding exosite interaction). "
            "(3) P-selectin, Mac-1 (αMβ2, CD11b/CD18) — platelet-leukocyte interactions. "
            "BERNARD-SOULIER SYNDROME (BSS): biallelic loss-of-function GP1BA → absent GPIb-IX-V → "
            "absent primary haemostasis. "
            "HALLMARK TRIAD: "
            "(1) MACRO-THROMBOCYTOPENIA — platelets reduced in count AND increased in size (diameter 5–10 μm, "
            "sometimes >20 μm; normal platelet diameter 2–3 μm); PATHOGNOMONIC on peripheral blood film; "
            "platelets as large as RBCs ('giant platelets'); "
            "(2) ABSENT RISTOCETIN AGGREGATION — ristocetin induces VWF-mediated agglutination of normal platelets; "
            "absent in BSS because GPIbα is the ristocetin target (add exogenous normal VWF — still absent in BSS; "
            "compare: VWD — reduced VWF, but GPIbα intact — response restored by adding VWF); "
            "(3) NORMAL AGGREGATION to other agonists (ADP, collagen, arachidonic acid, thrombin) — "
            "αIIbβ3 intact → aggregation function preserved (KEY DDx from GT, where ADP/collagen absent). "
            "PLATELET ADHESION under flow: absent (no GPIbα-VWF tethering) → failure of initial vessel wall contact. "
            "MISDIAGNOSIS AS ITP: very common — thrombocytopenia without obvious cause → ITP; "
            "EXAMINE THE BLOOD FILM — giant platelets not seen in ITP; "
            "ITP treatment (steroids, rituximab, splenectomy) is ineffective and harmful in BSS. "
            "GAIN-OF-FUNCTION GP1BA VARIANTS (separate entity): "
            "Platelet-type VWD (pseudo-VWD, PT-VWD) — GP1BA gain-of-function → spontaneous binding to VWF → "
            "loss of high-MW VWF multimers + thrombocytopenia → MIMICS VWD type 2B; "
            "distinguish by mixing test: platelet-poor plasma + normal washed platelets → "
            "abnormal agglutination in PT-VWD (not in VWD 2B). "
            "TREATMENT BSS: platelet transfusion for major bleeding; "
            "desmopressin (DDAVP) may reduce bleeding time slightly via VWF release (partial benefit); "
            "tranexamic acid for mucosal bleeding; rFVIIa in refractory/anti-platelet antibody settings."
        ),
        "aa": "626 aa",
        "kDa": "~170 kDa",
        "locus": "17p13.2",
        "omim_gene": 606672,
        "omim_disease": 231200,
        "inheritance": "AR biallelic (most common); rare AD dominant-negative; Gln316Stop founder (Middle Eastern populations)",
        "gene_class": (
            "GP1BA encodes glycoprotein Ibα. "
            "Domain structure: signal peptide → leucine-rich repeat (LRR) domain (N-terminal; 7 LRRs; "
            "VWF binding site; macroglycopeptide / sialylated mucin-like region → steric extension of receptor; "
            "two negative charge clusters flanking LRRs → thrombin binding; disulfide loop) → "
            "transmembrane helix → cytoplasmic tail (14-3-3ζ, filamin A binding → cytoskeletal anchorage). "
            "O-glycosylation of the macroglycopeptide region accounts for much of the apparent 170 kDa MW. "
            "Gain-of-function PT-VWD mutations (Met239Val, Gly233Val, Gly233Ser) are in the LRR region "
            "that contacts VWF A1 domain — increase VWF binding affinity constitutively."
        ),
        "n_patients": 40,
        "key_alerts": [
            "GP1BA-BSS-GIANT-PLATELETS-PATHOGNOMONIC: Giant platelets (diameter approaching or exceeding RBC size, ≥5–10 μm) on peripheral blood smear are PATHOGNOMONIC of Bernard-Soulier Syndrome — examine the blood smear in any thrombocytopenic patient before treating as ITP; giant platelets do NOT occur in ITP; automated platelet analysers undercount giant platelets (do manual count and film review)",
            "GP1BA-BSS-NOT-ITP: Bernard-Soulier Syndrome is the most frequently misdiagnosed inherited platelet disorder — thrombocytopenia without explanation → ITP assumed → steroids, rituximab, splenectomy given → NO RESPONSE; BSS does not respond to ITP therapy; correct diagnosis requires blood film (giant platelets) + platelet function tests (absent ristocetin aggregation) + flow cytometry (absent CD42b/GPIbα)",
            "GP1BA-BSS-RISTOCETIN-ABSENT: Absent ristocetin-induced platelet agglutination (RIPA) is the diagnostic hallmark of BSS — compare to Glanzmann Thrombasthenia where RIPA is NORMAL and to VWD where RIPA is reduced but restored by adding exogenous VWF; in BSS, adding exogenous VWF does NOT restore RIPA (receptor absent); low-dose ristocetin distinguishes PT-VWD (enhanced) from VWD type 2B (enhanced) from BSS (absent)",
            "GP1BA-BSS-HIGH-SHEAR-ADHESION-ABSENT: BSS specifically fails at HIGH SHEAR arterial conditions where VWF unfolds and binds GPIbα — venous haemostasis may be relatively preserved; mucocutaneous bleeding is predominant (epistaxis, gingival, menorrhagia); surgical haemostasis often requires platelet transfusion planning as haemostatic challenge is greatest at time of vascular injury",
            "GP1BA-PT-VWD-DDx-VWD2B: Platelet-type VWD (GP1BA gain-of-function) and VWD type 2B (VWF gain-of-function) are phenocopies — both show: large VWF multimers absent, thrombocytopenia, enhanced low-dose ristocetin RIPA; distinguish by: (1) mixing test with normal washed platelets added to patient plasma — PT-VWD shows abnormal agglutination (abnormal platelets); (2) VWF gene (VWF) and GP1BA sequencing — different genes; treatment differs (DDAVP contraindicated in VWD 2B; partially effective in PT-VWD)",
            "GP1BA-BSS-DDAVP-PARTIAL: DDAVP (desmopressin) releases endothelial VWF — may transiently improve BSS by increasing circulating VWF, slightly improving initial platelet tethering; response is partial and variable; NOT a substitute for platelet transfusion in major bleeding; trial DDAVP preoperatively in mild BSS to assess response before planning surgery",
            "GP1BA-BSS-FLOW-CYTOMETRY-DIAGNOSIS: Flow cytometry using anti-CD42b (anti-GPIbα) antibody — absent/severely reduced CD42b staining confirms BSS; combined with anti-CD41 (GPIIb)/CD61 (GPIIIa) — both NORMAL in BSS (αIIbβ3 intact); this flow panel distinguishes BSS (CD42b absent, CD41/CD61 normal) from GT (CD41/CD61 absent, CD42b normal) definitively without requiring platelet function testing",
            "GP1BA-BSS-SPLENECTOMY-AVOID: Splenectomy is NOT indicated in BSS — thrombocytopenia is due to ineffective megakaryopoiesis and peripheral platelet size (not immune destruction); splenectomy will NOT improve BSS platelet count and carries operative haemorrhage risk that may be difficult to manage; reserve for extreme cases with overwhelming surgical haemorrhage risk only after expert haematology consultation",
        ],
        "etiologies": {
            "Nonsense/frameshift — absent GPIbα, severe BSS": 14,
            "Missense in LRR domain — impaired VWF binding, severe BSS": 10,
            "Splice-site variant — exon skipping": 7,
            "Gain-of-function (Met239Val, Gly233Val) — Platelet-type VWD": 5,
            "Deletion/rearrangement including 17p contiguous": 2,
            "Founder variant (Gln316Stop, Middle Eastern)": 2,
        },
        "stats": {
            "mean_dx_age_y": 3.1,
            "mean_dx_delay_months": 14.2,
            "pct_misdiagnosed_as_itp": 55,
            "pct_giant_platelets_on_film": 100,
            "pct_ristocetin_absent": 100,
            "pct_menorrhagia_females": 82,
            "pct_itp_therapy_given": 38,
        },
        "dx_delay_distribution": {"<3 m": 10, "3–12 m": 14, "1–3 y": 11, ">3 y": 5},
    },
    # ── GP1BB — Bernard-Soulier Syndrome type B / 22q11.2 deletion ───────
    {
        "gene": "GP1BB",
        "protein": "GP1BB — Bernard-Soulier Syndrome Type B, AR, 22q11.21 Locus — DiGeorge Thrombocytopenia Overlap",
        "alias": (
            "GP1BB; OMIM gene 138720; Bernard-Soulier Syndrome type B (BSS-B) OMIM 231200 allelic; "
            "22q11.21; 206 aa; ~26 kDa (mature β-chain with N-glycosylation); AR (biallelic) for BSS; "
            "haploinsufficiency in 22q11.2 deletion syndrome. "
            "GP1BB encodes GPIbβ (glycoprotein Ibβ), the smaller disulfide-linked partner of GPIbα. "
            "GPIb COMPLEX STRUCTURE: GPIbα (GP1BA, 626 aa) linked by disulfide bond to GPIbβ (GP1BB, 206 aa); "
            "GPIbβ is essential for GPIbα surface expression and stability — without GPIbβ, GPIbα is not "
            "trafficked to the platelet surface (ER retention). "
            "22Q11.2 DELETION SYNDROME (DiGeorge / velo-cardio-facial): "
            "GP1BB maps within the common 22q11.2 deletion region (TBX1, CRKL, GP1BB among many genes). "
            "The majority of patients with 22q11.2 deletion have THROMBOCYTOPENIA (30–50%), often mild: "
            "haploinsufficiency of GP1BB contributes (alongside other deleted genes affecting megakaryopoiesis); "
            "platelet count 100–150 × 10⁹/L typical; giant platelets variable; "
            "platelet function usually NORMAL (heterozygous sufficient for normal GPIbβ surface density). "
            "BIALLELIC GP1BB — CLASSIC BSS: rare; same BSS phenotype as GP1BA-BSS (macro-thrombocytopenia, "
            "absent ristocetin aggregation, mucocutaneous bleeding); flow cytometry: absent CD42b (GPIbα, "
            "since GPIbβ is required for GPIbα surface expression — both fail). "
            "GENOTYPE: GP1BB Ala156Val, Trp127Stop among characterised variants; "
            "del22q11.2 is the most commonly encountered GP1BB haploinsufficiency. "
            "LABORATORY: BSS type B on platelet function — identical to GP1BA BSS; "
            "FISH or MLPA for 22q11.2 deletion if haploinsufficiency suspected (deletion not detected by sequencing). "
            "22Q11.2-ASSOCIATED THROMBOCYTOPENIA MANAGEMENT: usually no treatment required; "
            "avoid aspirin; plan platelet cover for major surgery; "
            "do NOT treat as ITP if count stable 100–150 × 10⁹/L and no bleeding symptoms. "
            "KEY DIAGNOSTIC POINT: in any patient with BSS-type phenotype (macro-thrombocytopenia, absent ristocetin) + "
            "cardiac/palatal/immune features → test for 22q11.2 deletion FIRST."
        ),
        "aa": "206 aa",
        "kDa": "~26 kDa",
        "locus": "22q11.21",
        "omim_gene": 138720,
        "omim_disease": 231200,
        "inheritance": "AR biallelic for classic BSS; del22q11.2 haploinsufficiency for mild thrombocytopenia",
        "gene_class": (
            "GP1BB encodes glycoprotein Ibβ (GPIbβ). "
            "Domain structure: signal peptide → single leucine-rich repeat (one LRR, unlike GPIbα which has 7) → "
            "transmembrane helix → short cytoplasmic tail (palmitoylation site). "
            "GPIbβ palmitoylation anchors the GPIb-IX complex in lipid rafts of the platelet membrane. "
            "GPIbβ is required for GPIbα surface expression — without GPIbβ, GPIbα accumulates in ER. "
            "The GPIbα/GPIbβ heterodimer non-covalently associates with GPIX (a single-pass transmembrane protein "
            "with one LRR) → GPIb-IX complex; GPV (GP5) associates non-covalently outside the core complex. "
            "22q11.21 is within the typical 3 Mb TBX1 deletion region of DiGeorge/VCFS syndrome."
        ),
        "n_patients": 40,
        "key_alerts": [
            "GP1BB-22Q11-DELETION-MOST-COMMON: The most common cause of GP1BB haploinsufficiency is the 22q11.2 deletion (DiGeorge / velo-cardio-facial syndrome) — in any child with BSS-like features AND cardiac defects (conotruncal), palatal anomalies, hypocalcaemia, immune deficiency, or learning difficulties, test for 22q11.2 deletion by MLPA or chromosomal microarray FIRST (sequencing will NOT detect this deletion)",
            "GP1BB-22Q11-THROMBOCYTOPENIA-MILD: Thrombocytopenia in 22q11.2 deletion is usually mild (100–150 × 10⁹/L) and asymptomatic — do NOT treat as ITP; a mild platelet count reduction in DiGeorge syndrome does not require haematological intervention; monitor counts; platelet function is usually normal (heterozygous GPIbβ expression sufficient); advise avoiding aspirin",
            "GP1BB-BSS-B-IDENTICAL-TO-GP1BA: Biallelic GP1BB mutations cause classic BSS phenotype — macro-thrombocytopenia + absent ristocetin aggregation + absent CD42b — identical to GP1BA-BSS; distinguish only by genetic testing (both GP1BA and GP1BB sequencing + MLPA required in any BSS panel) or functional complementation",
            "GP1BB-GPIBA-REQUIRES-GP1BB-FOR-SURFACE: GPIbα (GP1BA gene product) requires GPIbβ (GP1BB) for surface expression — in GP1BB-null platelets, GPIbα is retained in the ER; flow cytometry shows absent CD42b (GPIbα) even though GP1BA gene is normal — this can confound interpretation; genetic testing of BOTH GP1BA and GP1BB is mandatory in any CD42b-negative BSS",
            "GP1BB-22Q11-CARDIAC-HAEMOSTASIS: Patients with 22q11.2 deletion undergoing cardiac surgery have compounded haemorrhage risk — heparinisation, bypass, platelet dysfunction from bypass circuit, PLUS underlying mild GP1BB haploinsufficiency thrombocytopenia; plan platelet transfusions liberally (target >100 × 10⁹/L perioperatively); DDAVP has minimal benefit; tranexamic acid as antifibrinolytic",
            "GP1BB-FISH-MLPA-MANDATORY: If GP1BB sequence is normal but clinical BSS phenotype is present, 22q11.2 deletion MLPA or chromosomal microarray is MANDATORY — deletion will not be detected by standard sequencing or WES; conversely, if 22q11.2 deletion is known and mild thrombocytopenia is present, GP1BB haploinsufficiency is the likely contributor — no further platelet gene panel needed unless more severe phenotype",
            "GP1BB-NEONATAL-PRESENTATION: Biallelic GP1BB-BSS may present neonatally with severe thrombocytopenia and giant platelets — can be confused with NAIT (usually not giant platelets) or TAR syndrome (absent radii distinguishes TAR); neonatal blood film review by expert haematology is mandatory when severe thrombocytopenia is unexplained",
            "GP1BB-GPV-REDUCED-ALSO: In classic BSS (GP1BA or GP1BB null), GPV (glycoprotein V, GP5) surface expression is also reduced ~50%, as GPV associates non-covalently with GPIb-IX — absent GPIb-IX → GPV trafficking impaired; flow cytometry panel should include anti-CD42d (GPV) as an additional confirmatory marker",
        ],
        "etiologies": {
            "22q11.2 deletion (haploinsufficiency) — mild thrombocytopenia": 18,
            "Biallelic nonsense/frameshift — classic BSS type B": 9,
            "Missense Ala156Val — reduced surface GPIbβ": 6,
            "Splice-site variant — exon skipping": 4,
            "Trp127Stop — NMD, absent GPIbβ": 3,
        },
        "stats": {
            "mean_dx_age_y": 4.2,
            "mean_dx_delay_months": 18.5,
            "pct_22q11_deletion": 45,
            "pct_classic_bss_b": 22,
            "pct_misdiagnosed_itp": 38,
            "pct_cardiac_features_22q11": 80,
        },
        "dx_delay_distribution": {"<3 m": 8, "3–12 m": 13, "1–3 y": 13, ">3 y": 6},
    },
    # ── MYH9 — MYH9-Related Disease (May-Hegglin/Sebastian/Fechtner/Epstein)
    {
        "gene": "MYH9",
        "protein": "MYH9 — MYH9-Related Disease, AD, Döhle-like Neutrophil Inclusions PATHOGNOMONIC, NOT ITP",
        "alias": (
            "MYH9; OMIM gene 160775; MYH9-Related Disease (MYH9-RD) OMIM 155100 / 605249; "
            "22q12.3; 1960 aa; ~227 kDa; AD (heterozygous dominant-negative/haploinsufficient); "
            "prevalence estimated 1:500,000–1:1,000,000. "
            "MYH9 encodes the non-muscle myosin IIA (NMIIA) heavy chain. "
            "NMIIA function: actin-activated ATPase motor protein; "
            "forms hexamer with two regulatory light chains (MLC, MYL9) and two essential light chains → "
            "thick filaments; required for: "
            "(1) Cytokinesis (cell division) — NMIIA in the contractile ring; "
            "(2) Platelet production — megakaryocyte cytoplasmic organisation → proplatelet formation; "
            "(3) Neutrophil function — NMIIA in leading edge and granule exocytosis; "
            "(4) Kidney — podocyte foot-process actin architecture; "
            "(5) Inner ear — hair cell stereocilia. "
            "MYH9-RELATED DISEASE (MYH9-RD): caused by heterozygous mutations in MYH9 (dominant-negative or LOH); "
            "FOUR HISTORICAL SYNDROME NAMES (all now MYH9-RD — same gene, overlapping phenotype): "
            "MAY-HEGGLIN ANOMALY: macro-thrombocytopenia + Döhle-like inclusions in ALL granulocytes (neutrophils, eosinophils, monocytes); "
            "SEBASTIAN SYNDROME: similar but inclusions smaller, cluster-like; "
            "FECHTNER SYNDROME: macro-thrombocytopenia + inclusions + nephritis + SNHL + cataracts; "
            "EPSTEIN SYNDROME: macro-thrombocytopenia + nephritis + SNHL (no inclusions visible on standard staining — need electron microscopy). "
            "PATHOGNOMONIC FINDING: DÖHLE-LIKE INCLUSIONS in neutrophils — pale blue cytoplasmic inclusions on Wright-Giemsa stain; "
            "composed of abnormal NMIIA protein aggregates; present in 80–90% of MYH9-RD; "
            "may be subtle/missed by inexperienced morphologists → electron microscopy confirms. "
            "PLATELET COUNT: typically 20–150 × 10⁹/L (rarely <10); GIANT PLATELETS (diameter 5–20 μm); "
            "PLATELET FUNCTION: usually NORMAL in MYH9-RD — bleeding tendency often mild despite low counts; "
            "platelet count inversely correlates with bleeding; platelet size inversely correlates with count. "
            "EXTRAHAEMATOLOGICAL FEATURES (genotype-dependent): "
            "Nephritis (glomerular) — progressive proteinuria → ESRD (R702C, D1424N, R1933X variants particularly); "
            "SNHL — early-onset sensorineural hearing loss (requires audiological surveillance); "
            "Cataracts — anterior lens opacities (less common). "
            "TREATMENT: observation if mild; eltrombopag (thrombopoietin receptor agonist) raises platelet count "
            "in MYH9-RD — evidence from small trials; target pre-operative platelet count >50–80 × 10⁹/L; "
            "platelet transfusion for major haemorrhage; DDAVP reduces bleeding time transiently; "
            "DO NOT GIVE STEROIDS, IVIG, OR RITUXIMAB — these are ITP treatments ineffective in MYH9-RD; "
            "AVOID SPLENECTOMY (does not improve count; operative risk)."
        ),
        "aa": "1960 aa",
        "kDa": "~227 kDa",
        "locus": "22q12.3",
        "omim_gene": 160775,
        "omim_disease": 155100,
        "inheritance": "AD heterozygous; dominant-negative or haploinsufficiency; de novo ~30%; cluster in motor/coiled-coil domains",
        "gene_class": (
            "MYH9 encodes non-muscle myosin IIA (NMIIA) heavy chain. "
            "Domain structure: N-terminal motor domain (ATPase/actin-binding, S1 subfragment) → "
            "converter domain → long α-helical coiled-coil rod domain (S2 → tail; dimerisation; "
            "thick filament assembly) → non-helical tailpiece. "
            "MYH9-RD mutations cluster in hotspot regions: "
            "Motor domain (head): R702C, R702H, A96V — most frequently associated with nephritis/deafness; "
            "Coiled-coil (rod) domain: E1841K, R1933X — typically milder; fewer extrahaematological features. "
            "Dominant-negative mechanism: mutant NMIIA incorporates into heterofilaments with wild-type → "
            "disrupts normal NMIIA filament function → NMIIA aggregates in neutrophil cytoplasm → Döhle bodies; "
            "megakaryocyte proplatelet formation impaired → macro-thrombocytopenia."
        ),
        "n_patients": 40,
        "key_alerts": [
            "MYH9-DOHLE-BODIES-PATHOGNOMONIC: Döhle-like inclusions in neutrophils (pale blue cytoplasmic aggregates on Wright-Giemsa blood smear) are PATHOGNOMONIC of MYH9-related disease — look for these in ALL patients with unexplained macro-thrombocytopenia; inclusions may be subtle (require careful morphology or electron microscopy); their presence immediately excludes ITP (which has no inclusions)",
            "MYH9-NOT-ITP: MYH9-related disease is the most common inherited macrothrombocytopenia and is systematically misdiagnosed as ITP — steroid courses, IVIG, rituximab, and splenectomy are ineffective and harmful; the correct approach: blood film review (giant platelets + Döhle bodies) → flow cytometry (normal CD41/CD61/CD42b) → MYH9 sequencing; do NOT treat as ITP without examining the blood film",
            "MYH9-NEPHRITIS-SURVEILLANCE: Nephritis (focal segmental or diffuse mesangial glomerulonephritis) occurs in up to 30–50% of MYH9-RD patients with head/motor domain mutations (especially R702C, D1424N) — annual urine protein:creatinine ratio + eGFR mandatory; early ACE-inhibitor therapy for proteinuria; risk of ESRD requiring dialysis/transplant; genetic counselling for affected families re: donor kidney from family member (may share MYH9 variant)",
            "MYH9-SNHL-AUDIOLOGICAL-SURVEILLANCE: Sensorineural hearing loss (SNHL) occurs in 25–60% of MYH9-RD (all motor domain mutations at higher risk) — annual pure-tone audiometry from diagnosis; early hearing aids if progressive; cochlear implant in severe SNHL; monitor from childhood for all MYH9-RD regardless of variant location (SNHL can be asymmetric and slow)",
            "MYH9-ELTROMBOPAG-EVIDENCE: Eltrombopag (thrombopoietin receptor agonist, c-Mpl agonist) raises platelet count in MYH9-RD — evidence from Italian registry study (mean count increase ~30 × 10⁹/L); use pre-operatively to achieve platelet count >80 × 10⁹/L target; start 2–3 weeks before surgery; daily dosing 25–75 mg; response variable; monitor LFTs and platelet count weekly; does NOT cure the underlying defect",
            "MYH9-PLATELET-FUNCTION-NORMAL: Platelet FUNCTION (aggregation studies) is usually NORMAL in MYH9-RD — the defect is in platelet number and size only; mucocutaneous bleeding is therefore milder than expected from the platelet count; do NOT order aggregation studies to diagnose MYH9-RD; the diagnostic markers are blood film morphology + neutrophil inclusions + MYH9 sequencing",
            "MYH9-VARIANT-GENOTYPE-PHENOTYPE: There is strong genotype-phenotype correlation in MYH9-RD — motor domain (head) mutations (R702C, A96V, D1424N): severe macro-thrombocytopenia + high risk nephritis/SNHL/cataracts; rod domain (coiled-coil) mutations (E1841K, R1933X): fewer extrahaematological features; discuss genotype-specific surveillance intensity at diagnosis and document in patient records",
            "MYH9-SPLENECTOMY-AVOID: Splenectomy does NOT improve platelet count in MYH9-RD (thrombocytopenia is due to abnormal platelet production, not splenic destruction) and carries significant operative haemorrhage risk that may be difficult to manage; splenectomy is CONTRAINDICATED; eltrombopag + platelet transfusions are preferred perioperative management",
        ],
        "etiologies": {
            "R702C/H motor domain — nephritis/SNHL/macro-thrombocytopenia": 10,
            "E1841K coiled-coil — mild, fewer extrahaematological features": 8,
            "R1933X tail — mild phenotype, fewer inclusions on LM": 6,
            "A96V motor — severe thrombocytopenia, Döhle bodies": 6,
            "D1424N head-rod junction — nephritis + deafness dominant": 5,
            "De novo missense motor domain — no family history": 5,
        },
        "stats": {
            "mean_dx_age_y": 8.4,
            "mean_dx_delay_months": 28.6,
            "pct_misdiagnosed_itp": 68,
            "pct_nephritis": 35,
            "pct_snhl": 42,
            "pct_cataracts": 15,
            "pct_dohle_bodies": 88,
            "pct_giant_platelets": 100,
        },
        "dx_delay_distribution": {"<3 m": 5, "3–12 m": 10, "1–3 y": 15, ">3 y": 10},
    },
    # ── ANKRD26 — Thrombocytopenia 2, AML predisposition ─────────────────
    {
        "gene": "ANKRD26",
        "protein": "ANKRD26 — Thrombocytopenia 2, AD, 5′UTR Variants WES-MISS, AML/MDS 8% Lifetime Risk",
        "alias": (
            "ANKRD26; OMIM gene 610855; Thrombocytopenia 2 (THC2) OMIM 188000; "
            "10p12.1; 1710 aa; ~193 kDa; AD (heterozygous); prevalence unknown, likely underdiagnosed. "
            "ANKRD26 encodes ankyrin repeat domain-containing protein 26. "
            "ANKRD26 FUNCTION: "
            "During megakaryocyte maturation, ANKRD26 normally undergoes downregulation via "
            "transcriptional silencing at the 5′UTR regulatory region; "
            "RUNX1 and FLI1 (transcription factors) bind ANKRD26 5′UTR → repress ANKRD26 expression → "
            "allows terminal megakaryocyte differentiation → proplatelet formation → platelet release. "
            "Pathogenic ANKRD26 5′UTR variants (c.-127A>T, c.-128G>A, c.-134A>G — most common) → "
            "disrupt RUNX1/FLI1 binding sites → ANKRD26 fails to be silenced → "
            "TPO-MAPK signalling constitutively active → megakaryocyte proliferation WITHOUT differentiation → "
            "reduced platelet output → thrombocytopenia. "
            "CRITICAL POINT — 5′UTR VARIANTS MISSED BY STANDARD WES: "
            "Most pathogenic ANKRD26 variants are in the 5′UTR (non-coding), NOT in the coding exons; "
            "standard whole-exome sequencing (WES) typically does NOT capture 5′UTR → "
            "ANKRD26 THC2 is systematically under-detected by standard genetic pipelines; "
            "targeted ANKRD26 sequencing including 5′UTR is required for diagnosis. "
            "THROMBOCYTOPENIA 2 PHENOTYPE: "
            "Platelet count 30–150 × 10⁹/L (stable, lifelong); normal platelet morphology (NOT macro); "
            "platelet function NORMAL (pure count defect — no function or size abnormality); "
            "mild mucocutaneous bleeding (bruising, epistaxis); "
            "HAEMATOLOGICAL MALIGNANCY PREDISPOSITION: ~8% lifetime AML/MDS risk; "
            "ALL risk also elevated (less well characterised); "
            "monitoring: annual CBC; bone marrow biopsy if: unexplained cytopenia, dysplastic changes on CBC, "
            "or unexplained acute deterioration. "
            "MANAGEMENT: observation for mild thrombocytopenia; "
            "thrombopoietin receptor agonists (eltrombopag) for pre-operative platelet count optimisation; "
            "AVOID platelet transfusion unless active major bleeding (sensitisation risk); "
            "AVOID platelet-sparing drugs but ESPECIALLY avoid unnecessary haematotoxic chemotherapy "
            "→ may precipitate AML in susceptible marrow. "
            "FAMILY SCREENING: all first-degree relatives should be offered ANKRD26 5′UTR targeted testing."
        ),
        "aa": "1710 aa",
        "kDa": "~193 kDa",
        "locus": "10p12.1",
        "omim_gene": 610855,
        "omim_disease": 188000,
        "inheritance": "AD heterozygous; 5′UTR regulatory variants most common; coding variants rare; penetrance ~85%",
        "gene_class": (
            "ANKRD26 encodes an ankyrin repeat and spectrin domain-containing protein. "
            "Contains: N-terminal ankyrin repeats (protein-protein interactions) → central coiled-coil → "
            "spectrin-like domain (cytoskeletal interaction). "
            "5′UTR regulatory region (c.-100 to c.-145): critical RUNX1/FLI1 binding sites; "
            "most pathogenic variants cluster here — predominantly single-nucleotide changes "
            "that disrupt transcription factor binding → failure of megakaryocyte-specific ANKRD26 downregulation. "
            "ANKRD26-TPO signalling: constitutive ANKRD26 expression → activates MAPK/ERK via TPO receptor → "
            "enhanced megakaryocyte proliferation at expense of differentiation."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ANKRD26-5UTR-WES-MISS: The most common pathogenic ANKRD26 variants (c.-127A>T, c.-128G>A, c.-134A>G) are in the 5′UTR — OUTSIDE standard whole-exome sequencing coverage; standard WES reports may be FALSE NEGATIVE for ANKRD26 THC2; specifically request ANKRD26 5′UTR Sanger sequencing or a targeted inherited thrombocytopenia panel that includes 5′UTR coverage when ANKRD26 THC2 is suspected",
            "ANKRD26-AML-RISK-8PCT: ANKRD26 THC2 carries an ~8% lifetime AML/MDS risk — annual complete blood count with differential and morphology review is mandatory; any new cytopenias, unexplained dysplastic features, or clinical deterioration should prompt bone marrow biopsy immediately; avoid haematotoxic agents (chloramphenicol, chemotherapy, radiation to marrow sites) without specialist haematology input",
            "ANKRD26-NORMAL-PLATELET-SIZE-FUNCTION: Unlike MYH9-RD and BSS, ANKRD26 THC2 shows NORMAL platelet size and NORMAL platelet function — aggregation studies are unremarkable; the defect is PURELY in platelet number; misleading if standard thrombocytopenia workup focuses only on platelet size/function; genetic testing is required to identify ANKRD26 THC2",
            "ANKRD26-FAMILY-SCREENING-MANDATORY: ANKRD26 THC2 is AD with ~85% penetrance; first-degree relatives have 50% probability of carrying the same 5′UTR variant → ALL first-degree relatives should be offered ANKRD26-targeted sequencing (including 5′UTR); important for AML surveillance registration in affected relatives",
            "ANKRD26-RUNX1-FLI1-PATHWAY: ANKRD26 THC2 and RUNX1 FPD-AML share a mechanistic link — both affect the RUNX1 transcriptional programme in megakaryocytes; patients with ANKRD26 THC2 should be counselled about AML predisposition in the same framework as RUNX1 FPD-AML; registries and clinical trials for inherited AML predisposition syndromes should include ANKRD26 THC2 patients",
            "ANKRD26-ELTROMBOPAG-PREOPERATIVE: Eltrombopag raises platelet count in ANKRD26 THC2 (TPO-MAPK pathway constitutively active — eltrombopag further stimulates c-Mpl; evidence from case series); start 2–3 weeks pre-operatively; target platelet count >80 × 10⁹/L for major surgery; monitor weekly; do NOT continue indefinitely without haematological supervision (theoretical malignancy promotion concern)",
            "ANKRD26-STABLE-THROMBOCYTOPENIA-COUNSELLING: ANKRD26 THC2 thrombocytopenia is typically STABLE across life — count 30–150 × 10⁹/L with no progressive decline in the absence of malignant transformation; reassure patients that the count alone does not predict clinical bleeding; tailor activity restrictions to actual bleeding history, not to the count; major trauma/surgery requires pre-planning but routine daily activities are usually safe",
            "ANKRD26-NOT-ITP: ANKRD26 THC2 is a common cause of 'ITP-like' lifelong thrombocytopenia — failure to respond to steroids/IVIG should trigger genetic workup including ANKRD26; the specific risk in ITP misdiagnosis is unnecessary splenectomy (does not improve THC2 count) and aggressive immunosuppression (masks early AML evolution)",
        ],
        "etiologies": {
            "c.-127A>T 5′UTR — disrupts FLI1 binding, most common European": 14,
            "c.-128G>A 5′UTR — disrupts RUNX1 binding": 10,
            "c.-134A>G 5′UTR — FLI1 binding disruption": 7,
            "Other 5′UTR single-nucleotide change": 5,
            "Rare coding missense (less common)": 4,
        },
        "stats": {
            "mean_dx_age_y": 22.4,
            "mean_dx_delay_months": 48.2,
            "pct_misdiagnosed_itp": 72,
            "pct_aml_mds_lifetime": 8,
            "pct_normal_platelet_function": 98,
            "pct_wes_missed": 85,
        },
        "dx_delay_distribution": {"<3 m": 4, "3–12 m": 8, "1–3 y": 14, ">3 y": 14},
    },
    # ── ETV6 — ETV6-related Thrombocytopenia with ALL predisposition ──────
    {
        "gene": "ETV6",
        "protein": "ETV6 — ETV6-related Thrombocytopenia, AD, ALL Predisposition 25–35%, Donor Screening MANDATORY",
        "alias": (
            "ETV6; OMIM gene 600618; Thrombocytopenia 5 with ALL predisposition (THC5) OMIM 616937; "
            "12p13.2; 452 aa; ~52 kDa; AD (heterozygous); prevalence unknown, likely rare but underrecognised. "
            "ETV6 encodes ETS variant transcription factor 6, also known as TEL (Translocation Ets Leukaemia). "
            "ETV6 STRUCTURE: "
            "Pointed (PNT/SAM) domain (N-terminal) → central linker → DNA-binding ETS domain (C-terminal). "
            "PNT domain: mediates oligomerisation (homo- and heteropolymerisation); important for repressor function. "
            "ETS domain: binds GGA(A/T) DNA sequences; ETV6 acts predominantly as a transcriptional repressor "
            "in haematopoietic cells; "
            "regulates haematopoietic stem cell maintenance and lineage commitment. "
            "ETV6-RUNX1 FUSION (t(12;21)(p13;q22)) — SOMATIC, MOST COMMON CHILDHOOD ALL: "
            "ETV6-RUNX1 translocation creates a fusion oncogene (somatic, acquired in leukaemic clone); "
            "accounts for ~25% of B-ALL in children aged 2–10; "
            "DISTINCT from germline ETV6 variants that cause THC5; "
            "somatic ETV6-RUNX1 does NOT = germline ETV6 mutation — do not conflate. "
            "GERMLINE ETV6 VARIANTS — THC5: "
            "Heterozygous germline ETV6 mutations (predominantly ETS domain, e.g. Arg418Gln, Pro214Leu) → "
            "thrombocytopenia (platelet count 60–150 × 10⁹/L) + HAEMATOLOGICAL MALIGNANCY PREDISPOSITION. "
            "ALL PREDISPOSITION: ~25–35% cumulative lifetime ALL risk (predominantly B-ALL); "
            "AML/MDS also reported; solid tumours (CML-like, lymphoma). "
            "THROMBOCYTOPENIA PHENOTYPE: mild-moderate (count 60–150 × 10⁹/L); "
            "platelet function — may have impaired secretion (delta granule or alpha granule reduced); "
            "mild bleeding tendency (bruising, epistaxis); platelet morphology variably abnormal (mild macrothrombocytes). "
            "HAEMATOLOGICAL SURVEILLANCE: annual CBC + differential; blood film morphology; "
            "low threshold for bone marrow biopsy; "
            "paediatric haematology/oncology involvement recommended for at-risk family members. "
            "DONOR SCREENING — CRITICAL: "
            "ETV6 THC5 is AD → affected family member used as haematopoietic stem cell donor may "
            "transfer the same ETV6 predisposition to the recipient → "
            "ALL developed in the donor graft (donor-derived leukaemia) → "
            "MANDATORY: genotype ALL first-degree relatives before using as HSC donor; "
            "ETV6 THC5 carriers should NOT be used as HSC donors for haematological malignancy treatment."
        ),
        "aa": "452 aa",
        "kDa": "~52 kDa",
        "locus": "12p13.2",
        "omim_gene": 600618,
        "omim_disease": 616937,
        "inheritance": "AD heterozygous; ETS domain mutations predominant; de novo and familial cases; penetrance ~80%",
        "gene_class": (
            "ETV6 encodes ETS variant transcription factor 6. "
            "Domain structure: N-terminal PNT (Pointed/SAM) domain → central linker/inhibitory region → "
            "C-terminal ETS domain (ets proto-oncogene family; high-affinity GGA(A/T) binding). "
            "ETV6 acts as a transcriptional repressor by recruiting co-repressors (mSin3, NCoR/SMRT) to target genes. "
            "Pathogenic ETS domain variants (Arg418Gln, Pro214Leu, Arg399Cys) — impair DNA binding → "
            "loss of transcriptional repression → de-repression of oncogenic targets in haematopoietic progenitors → "
            "leukaemia predisposition. "
            "Second-hit in leukaemia: LOH at ETV6 locus, somatic ETV6 mutation, or ETV6-RUNX1 fusion acquired. "
            "ETV6-RUNX1 somatic fusion (t(12;21)) — distinct mechanism from germline ETV6; "
            "represents activation of RUNX1 oncogenic programme via ETV6-PNT domain dimerisation."
        ),
        "n_patients": 40,
        "key_alerts": [
            "ETV6-DONOR-SCREENING-MANDATORY: If an ETV6 THC5 patient requires haematopoietic stem cell transplantation (HSCT), ALL first-degree relatives MUST be ETV6-genotyped BEFORE being accepted as donors — ETV6 THC5 is AD; a carrier sibling or parent used as donor will transfer the predisposition to the recipient → donor-derived leukaemia risk; use matched unrelated donor (MUD) if no genotypically normal related donor is available",
            "ETV6-ALL-PREDISPOSITION-25-35PCT: Germline ETV6 variants carry 25–35% lifetime ALL risk (predominantly B-ALL) — this is higher than RUNX1 FPD-AML (~35% AML) and comparable; establish ALL surveillance protocol from diagnosis: annual CBC, blood film; immediate marrow evaluation if: unexplained new cytopenias, lymphadenopathy, bone pain, B-symptoms; involve paediatric haematology/oncology if patient is a child",
            "ETV6-SOMATIC-FUSION-DISTINCT: The somatic ETV6-RUNX1 translocation t(12;21) in common childhood B-ALL is COMPLETELY DIFFERENT from germline ETV6 THC5 — do NOT confuse them; a child with leukaemia and somatic t(12;21) does NOT necessarily have germline ETV6 THC5; conversely, a family with germline ETV6 THC5 may develop ALL by different somatic mechanisms; distinguish by testing germline DNA (skin fibroblasts or non-haematopoietic tissue)",
            "ETV6-FAMILY-SURVEILLANCE-CASCADE: ETV6 THC5 is AD with ~80% penetrance — offer germline ETV6 testing to all first-degree relatives; enrol ALL ETV6 carriers in haematological surveillance regardless of current platelet count; document in patient records that family members undergoing haematological assessment should have germline ETV6 status checked; register in national inherited haematological malignancy predisposition registry if available",
            "ETV6-MILD-THROMBOCYTOPENIA-NOT-ITP: ETV6 THC5 thrombocytopenia (60–150 × 10⁹/L) is stable and mild — ITP misdiagnosis leads to steroid courses and splenectomy without benefit; the critical risk is NOT bleeding from thrombocytopenia but MALIGNANT TRANSFORMATION; genetic testing must be performed in all familial or treatment-resistant thrombocytopenia cases",
            "ETV6-PLATELET-FUNCTION-VARIABLE: Platelet function defects (delta granule deficiency, impaired secretion) are reported in some ETV6 THC5 patients — aggregation with ADP/collagen may show reduced secondary waves; this pattern mirrors RUNX1 FPD-AML (both affect megakaryocyte transcription); platelet aggregation studies may help characterise severity but are NOT diagnostic of ETV6 THC5 specifically",
            "ETV6-GERMLINE-TESTING-METHODOLOGY: Germline ETV6 status must be tested from NON-HAEMATOPOIETIC tissue (skin fibroblasts, buccal swab) in patients with active leukaemia — leukaemic cells may have somatic LOH at ETV6 masking the germline allele in blood-derived DNA; alternatively, test at complete remission from peripheral blood once leukaemic clone is cleared",
            "ETV6-NO-PROPHYLACTIC-CHEMOTHERAPY: There is NO indication for prophylactic haematological therapy (chemotherapy, stem cell transplantation) in ETV6 THC5 carriers who have not developed malignancy — surveillance is the standard; enrol in prospective registries (e.g., EuroPDX, IBMFS registry); decisions about prophylactic transplantation are made only when sequential malignant change is documented, NOT on genetic diagnosis alone",
        ],
        "etiologies": {
            "Arg418Gln ETS domain — DNA binding impaired, most common": 12,
            "Pro214Leu linker/ETS boundary — reduced repressor function": 9,
            "Arg399Cys ETS domain — disrupted DNA-protein interface": 8,
            "Nonsense/frameshift — haploinsufficiency": 6,
            "De novo ETS domain missense — no family history": 5,
        },
        "stats": {
            "mean_dx_age_y": 12.8,
            "mean_dx_delay_months": 36.4,
            "pct_all_developed": 28,
            "pct_misdiagnosed_itp": 58,
            "pct_family_carrier_identified": 72,
            "pct_donor_genotyped_before_hsct": 42,
        },
        "dx_delay_distribution": {"<3 m": 5, "3–12 m": 9, "1–3 y": 14, ">3 y": 12},
    },
    # ── RUNX1 — Familial Platelet Disorder with AML (FPD-AML) ────────────
    {
        "gene": "RUNX1",
        "protein": "RUNX1 — FPD-AML, AD, δ-Granule Deficiency, AML/MDS 35–44% Lifetime, Donor Screening MANDATORY",
        "alias": (
            "RUNX1; OMIM gene 151385; Familial Platelet Disorder with predisposition to AML (FPD-AML) OMIM 601399; "
            "21q22.12; 480 aa; ~49 kDa (isoform b); AD (heterozygous); prevalence ~1:50,000 (estimated). "
            "RUNX1 encodes Runt-domain transcription factor alpha-2 (also AML1, CBFA2, PEBP2aB). "
            "RUNX1 FUNCTION: "
            "Core Binding Factor (CBF) alpha subunit; heterodimerises with CBFβ (CBFB) → "
            "binds RUNX-cognate sequence 5′-TGTGGT-3′ in target gene promoters; "
            "transcriptional activator for haematopoietic differentiation genes: "
            "CSF1R (M-CSF receptor), MPO (myeloperoxidase), CD11b, HLA class II, MYH9, GP1BA, ANKRD26; "
            "RUNX1 is required for: "
            "(1) Megakaryocyte differentiation → platelet biogenesis; "
            "(2) Delta granule (dense granule) biogenesis — stores ADP, serotonin, calcium; "
            "(3) Haematopoietic stem cell (HSC) self-renewal vs. differentiation balance. "
            "FPD-AML PHENOTYPE: "
            "THROMBOCYTOPENIA: mild to moderate (count 50–150 × 10⁹/L); stable; normal platelet morphology; "
            "DELTA-GRANULE DEFICIENCY: impaired platelet secretion → reduced ADP and serotonin release → "
            "aggregation studies: reduced secondary aggregation wave to ADP; absent arachidonic acid response; "
            "reduced collagen response → FUNCTIONAL BLEEDING DISPROPORTIONATE TO COUNT; "
            "mepacrine staining: reduced or absent dense granules (fluorescence microscopy). "
            "HAEMATOLOGICAL MALIGNANCY PREDISPOSITION: "
            "AML (acute myeloid leukaemia): 35–44% cumulative lifetime risk; "
            "MDS (myelodysplastic syndrome): precursor state often identifiable; "
            "ALL, CML, T-cell lymphomas also reported; "
            "somatic second-hit mechanisms: LOH at 21q22, somatic RUNX1 mutation, RUNX1-AML1 gain-of-function. "
            "SOMATIC RUNX1 FUSIONS — DISTINCT FROM GERMLINE: "
            "t(8;21)(q22;q22) — RUNX1-RUNX1T1 (AML1-ETO): most common somatic AML-associated rearrangement; "
            "t(12;21) — ETV6-RUNX1 (TEL-AML1): most common paediatric ALL rearrangement; "
            "these somatic fusions are ACQUIRED in leukaemic cells and are DISTINCT from germline RUNX1 FPD-AML. "
            "DONOR SCREENING — CRITICAL: same imperative as ETV6 THC5 — "
            "RUNX1 FPD-AML is AD; affected sibling used as HSCT donor transfers AML predisposition → "
            "MANDATORY genotyping of all family members before HSC donation. "
            "SURVEILLANCE: annual CBC + smear + bone marrow evaluation if: unexplained cytopenia change, "
            "new dysplastic features (basophilic stippling, neutrophil dysplasia), organomegaly; "
            "consider annual marrow biopsy in high-risk patients (strong family history of AML); "
            "enrol in germline AML predisposition registries."
        ),
        "aa": "480 aa",
        "kDa": "~49 kDa",
        "locus": "21q22.12",
        "omim_gene": 151385,
        "omim_disease": 601399,
        "inheritance": "AD heterozygous; runt domain (RHD) mutations most common; C-terminal mutations/truncations also; de novo ~25%; penetrance variable",
        "gene_class": (
            "RUNX1 encodes RUNX family transcription factor 1. "
            "Domain structure: N-terminal runt homology domain (RHD, ~130 aa; DNA binding + CBFβ heterodimerisation) → "
            "transcriptional activation domain (TAD) → C-terminal VWRPY motif (Groucho co-repressor interaction) → "
            "nuclear localisation signal. "
            "Pathogenic germline variants: "
            "RHD missense (e.g. Arg201Gln, Asp198Asn) — impair DNA binding or CBFβ interaction → dominant negative; "
            "C-terminal truncation (e.g. p.Gln188Ter) — haploinsufficiency; "
            "Large deletion/rearrangement (MLPA required). "
            "Somatic second-hit: LOH at 21q22 → monoallelic remaining WT RUNX1 → AML. "
            "FPD-AML genotype-phenotype: RHD mutations → worse thrombocytopenia; "
            "C-terminal haploinsufficiency → higher AML risk (40–50%); "
            "both groups require equivalent surveillance intensity."
        ),
        "n_patients": 40,
        "key_alerts": [
            "RUNX1-DONOR-SCREENING-MANDATORY: If an FPD-AML patient requires HSCT, ALL first-degree relatives must be genotyped for the germline RUNX1 variant BEFORE being used as donors — a sibling or parent carrying the same RUNX1 variant has 35–44% lifetime AML risk; donor-derived leukaemia arising from a RUNX1 FPD-AML donor graft is a documented catastrophic complication; use matched unrelated donor if no genotypically normal family donor is available",
            "RUNX1-AML-MDS-44PCT: RUNX1 FPD-AML carries 35–44% cumulative lifetime AML/MDS risk — among the highest of all inherited platelet disorders; AML may be preceded by MDS (myelodysplastic phase); annual CBC with differential and film review mandatory; bone marrow biopsy if: new unexplained cytopenia, dysplastic features on film, organomegaly, or constitutional symptoms; register in national AML predisposition registry",
            "RUNX1-DELTA-GRANULE-DEFICIENCY: Dense granule (δ-granule) deficiency is characteristic of RUNX1 FPD-AML — ADP, serotonin, and calcium stores are reduced; laboratory: mepacrine (quinacrine) staining under fluorescence microscopy shows absent/reduced dense granule uptake (gold standard); aggregation studies show reduced or absent secondary aggregation wave; this FUNCTIONAL deficit causes bleeding disproportionate to platelet count — a key clinical distinguishing feature",
            "RUNX1-FUNCTIONAL-BLEEDING: Patients with RUNX1 FPD-AML bleed MORE than expected from their platelet count alone — because platelet FUNCTION is also impaired (delta granule deficiency); bleeding score (ISTH-BAT or MCMDM-1 VWD) is elevated even with counts of 80–100 × 10⁹/L; pre-operative planning must account for BOTH low count AND platelet function deficit",
            "RUNX1-SOMATIC-FUSION-DISTINCT: The somatic t(8;21) RUNX1-RUNX1T1 (AML1-ETO) fusion in AML and the t(12;21) ETV6-RUNX1 fusion in B-ALL are DISTINCT from germline RUNX1 FPD-AML — testing leukaemic blast DNA reveals somatic abnormalities; test constitutional DNA (skin fibroblasts) to confirm germline FPD-AML; do not assume somatic RUNX1 fusion implies germline FPD-AML without constitutional testing",
            "RUNX1-MARROW-SURVEILLANCE-PROTOCOL: Annual haematological review for all RUNX1 FPD-AML carriers; consider annual bone marrow trephine + aspirate + cytogenetics in high-risk patients (strong family AML history, C-terminal truncation variant, age >30); watch specifically for del(7q), monosomy 7, +8, del(5q) — clonal cytopenias with these cytogenetic changes indicate imminent AML transformation; immediate haematology referral for stem cell transplantation assessment",
            "RUNX1-MLPA-FOR-LARGE-DELETIONS: Standard sequencing and WES may miss large RUNX1 deletions — MLPA (multiplex ligation-dependent probe amplification) is required if sequencing is negative but FPD-AML is clinically suspected (familial thrombocytopenia + AML, delta granule deficiency on mepacrine staining, reduced RUNX1 protein by flow cytometry); ensure MLPA is included in all RUNX1 FPD-AML diagnostic panels",
            "RUNX1-ANKRD26-ETV6-SHARED-PATHWAY: RUNX1, ETV6, and ANKRD26 all converge on megakaryocyte transcription — RUNX1/FLI1 silence ANKRD26; ETV6 is a RUNX1 partner; loss of any one impairs platelet production through overlapping pathways; a patient with thrombocytopenia + AML/ALL predisposition should be tested for ALL three germline variants (RUNX1 exon + MLPA, ETV6 exon, ANKRD26 5′UTR) as part of a comprehensive inherited platelet/AML panel",
        ],
        "etiologies": {
            "Arg201Gln RHD — dominant-negative, common North American/European founder": 12,
            "Asp198Asn RHD — CBFβ-binding impaired": 8,
            "C-terminal truncation (frameshift) — haploinsufficiency, high AML risk": 8,
            "Large deletion (MLPA required) — complete haploinsufficiency": 5,
            "De novo RHD missense — no family history": 4,
            "Splice-site — exon skipping, partial function loss": 3,
        },
        "stats": {
            "mean_dx_age_y": 18.6,
            "mean_dx_delay_months": 42.8,
            "pct_aml_mds_developed": 38,
            "pct_delta_granule_deficient": 92,
            "pct_misdiagnosed_itp": 60,
            "pct_donor_genotyped_before_hsct": 38,
            "pct_family_cascade_tested": 65,
        },
        "dx_delay_distribution": {"<3 m": 4, "3–12 m": 7, "1–3 y": 14, ">3 y": 15},
    },
]


# ─── Patient cohort generation ────────────────────────────────────────────────

def _make_cohort():
    cohort = {}
    for i, gene_info in enumerate(PLATELET_GENES):
        seed = SEED_BASE + i
        rng = random.Random(seed)
        gene = gene_info["gene"]
        n = gene_info["n_patients"]
        patients = []
        for p in range(n):
            age_dx = round(rng.gauss(gene_info["stats"].get("mean_dx_age_y", 20), 8), 1)
            age_dx = max(0.1, min(80, age_dx))
            dx_delay = round(rng.gauss(gene_info["stats"].get("mean_dx_delay_months", 18), 8), 1)
            dx_delay = max(0.5, min(120, dx_delay))
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

    itga2b = _COHORT["ITGA2B"]["stats"]
    gp1ba  = _COHORT["GP1BA"]["stats"]
    myh9   = _COHORT["MYH9"]["stats"]
    ankrd26= _COHORT["ANKRD26"]["stats"]
    etv6   = _COHORT["ETV6"]["stats"]
    runx1  = _COHORT["RUNX1"]["stats"]

    return {
        "atlas": "Hereditary-Platelet-Disorder-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Platelet Disorder Reference",
        "genes": genes_summary,
        "aggregate_stats": {
            "total_patients": total,
            "mean_dx_age_years": mean_dx_age,
            "mean_dx_delay_months": mean_dx_delay,
            "gt_clot_retraction_absent_pct": itga2b["pct_clot_retraction_absent"],
            "gt_rfviia_used_pct": itga2b["pct_rfviia_used"],
            "bss_giant_platelets_pct": gp1ba["pct_giant_platelets_on_film"],
            "bss_misdiagnosed_itp_pct": gp1ba["pct_misdiagnosed_as_itp"],
            "myh9_misdiagnosed_itp_pct": myh9["pct_misdiagnosed_itp"],
            "myh9_nephritis_pct": myh9["pct_nephritis"],
            "ankrd26_wes_missed_pct": ankrd26["pct_wes_missed"],
            "ankrd26_aml_lifetime_pct": ankrd26["pct_aml_mds_lifetime"],
            "etv6_all_predisposition_pct": etv6["pct_all_developed"],
            "runx1_aml_mds_pct": runx1["pct_aml_mds_developed"],
            "runx1_delta_granule_pct": runx1["pct_delta_granule_deficient"],
            "cascade_tested_pct": 68,
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
        "atlas": "Hereditary-Platelet-Disorder-Atlas",
        "concepts": {
            "Platelet Function Disorders vs. Thrombocytopenia — Diagnostic Framework": (
                "Inherited platelet disorders divide into two broad categories: "
                "(1) PLATELET FUNCTION DISORDERS (normal count, defective function): Glanzmann Thrombasthenia "
                "(ITGA2B/ITGB3 — absent αIIbβ3 → absent aggregation), Platelet-type VWD (GP1BA GOF), "
                "storage pool diseases (delta/alpha granule deficiencies); "
                "(2) THROMBOCYTOPENIA (reduced count): BSS (GP1BA/GP1BB — absent GPIb-IX), "
                "MYH9-RD, ANKRD26 THC2, ETV6 THC5, RUNX1 FPD-AML; "
                "(3) COMBINED (reduced count + function): RUNX1 FPD-AML (low count + delta granule deficiency), "
                "BSS (low count + absent VWF-mediated adhesion). "
                "DIAGNOSTIC PATHWAY: "
                "Step 1: CBC + blood film (platelet count, size, morphology; neutrophil inclusions). "
                "Step 2: Platelet function testing (LTA — light transmission aggregometry; PFA-100; TEG/ROTEM). "
                "Step 3: Flow cytometry (CD41/GPIIb, CD61/GPIIIa, CD42b/GPIbα, CD42a/GPIX). "
                "Step 4: Electron microscopy (dense granule counting; NMIIA inclusions). "
                "Step 5: Genetic testing (targeted panel: ITGA2B, ITGB3, GP1BA, GP1BB, GP9, MYH9, "
                "ANKRD26 5′UTR, ETV6, RUNX1 + MLPA for large deletions). "
                "KEY DIFFERENTIATORS: "
                "Normal platelet count + absent aggregation = GT (ITGA2B/ITGB3); "
                "Low count + giant platelets + absent ristocetin = BSS (GP1BA/GP1BB); "
                "Low count + giant platelets + Döhle bodies + normal function = MYH9-RD; "
                "Low count + normal morphology + normal function + 5′UTR variant = ANKRD26 THC2; "
                "Low count + delta granule deficiency + AML predisposition = RUNX1 FPD-AML."
            ),
            "Malignancy Predisposition — RUNX1, ETV6, ANKRD26 Shared Risk": (
                "Three of the eight genes in this atlas carry haematological malignancy predisposition: "
                "RUNX1 FPD-AML: 35–44% lifetime AML/MDS risk (highest in this group); "
                "ETV6 THC5: 25–35% lifetime ALL (B-cell) risk; AML also reported; "
                "ANKRD26 THC2: ~8% lifetime AML/MDS risk (lowest but clinically significant). "
                "SHARED MECHANISM: all three encode transcription factors (RUNX1, ETV6) or "
                "TPO-MAPK regulators (ANKRD26) critical for megakaryocyte differentiation and "
                "haematopoietic progenitor self-renewal vs. commitment; "
                "haploinsufficiency → progenitor pool imbalance → second-hit acquisition → AML/ALL. "
                "SURVEILLANCE FRAMEWORK (all three): "
                "Annual CBC + differential + blood film morphology; "
                "bone marrow biopsy if: unexplained count change, dysplastic features, organomegaly, B-symptoms; "
                "cytogenetics (karyotype + FISH) on marrow — prognostically critical for transplant timing; "
                "enrol in national inherited AML/ALL predisposition registries; "
                "DO NOT USE FAMILY MEMBER AS HSCT DONOR without prior germline genotyping. "
                "SOMATIC SECOND-HIT EVOLUTION: RUNX1 → LOH at 21q22 + additional mutations; "
                "ETV6 → LOH at 12p13 + somatic ETV6-RUNX1 fusion; "
                "ANKRD26 → somatic ANKRD26 overexpression in leukaemic blasts; "
                "sequential marrow monitoring allows early detection of clonal evolution."
            ),
            "Ristocetin — Understanding the Key Diagnostic Reagent": (
                "Ristocetin is an antibiotic reagent that mimics high-shear VWF–GPIbα interaction "
                "to induce platelet agglutination in vitro. "
                "NORMAL: ristocetin + normal platelet-rich plasma → VWF binds GPIbα → platelet agglutination. "
                "BSS (GP1BA/GP1BB null): ristocetin → NO agglutination (absent GPIbα receptor); "
                "adding exogenous VWF does NOT help (receptor absent). "
                "VWD type 1/3: ristocetin → reduced agglutination (insufficient VWF); "
                "adding exogenous VWF RESTORES agglutination (receptor GPIbα intact). "
                "VWD type 2B (VWF GOF) + Platelet-type VWD (GP1BA GOF): "
                "LOW-DOSE ristocetin (0.5 mg/mL) → ENHANCED agglutination (gain-of-function → "
                "spontaneous VWF–GPIbα interaction); distinguishing PT-VWD from 2B requires "
                "mixing test: patient platelets + normal plasma (PT-VWD aggregates; 2B does not). "
                "GT (ITGA2B/ITGB3 null): ristocetin NORMAL (GPIbα and VWF both intact); "
                "aggregation to ADP/collagen/thrombin ABSENT (αIIbβ3 defective). "
                "DIAGNOSTIC SUMMARY TABLE: "
                "GT: ristocetin ✓, ADP/collagen ✗, clot retraction ✗, platelet count NORMAL, giant platelets NO; "
                "BSS: ristocetin ✗, ADP/collagen ✓, clot retraction ✓, platelet count LOW, giant platelets YES; "
                "MYH9-RD: ristocetin ✓, ADP/collagen ✓, clot retraction ✓, platelet count LOW, giant platelets YES + Döhle bodies."
            ),
            "HPA System — Neonatal Alloimmune Thrombocytopenia (NAIT)": (
                "Human Platelet Antigens (HPA) are biallelic polymorphisms on platelet surface glycoproteins. "
                "HPA nomenclature: HPA-1 to HPA-21 characterised; most clinically significant: "
                "HPA-1 (on ITGB3 Leu33Pro, PlA1/PlA2, Zw, rs5918): "
                "HPA-1a (Leu33) — 98% allele frequency Europeans; "
                "HPA-1b (Pro33) — 2% Europeans; "
                "NAIT mechanism: HPA-1b/1b mother exposed to HPA-1a (fetal) platelets → anti-HPA-1a IgG; "
                "crosses placenta from ~28 weeks → fetal platelet destruction → neonatal thrombocytopenia; "
                "ICH risk 10–20% untreated; FIRST PREGNANCY AFFECTED (unlike Rh-HDN). "
                "HPA-5 (on GP1BA Glu505Lys, Br, rs10758144): second most common NAIT antigen; "
                "HPA-5b carrier mothers (5% Europeans); HPA-5b vs HPA-5a mismatch → anti-HPA-5b antibodies. "
                "NAIT DIAGNOSIS: severe thrombocytopenia (often <30 × 10⁹/L) at birth in otherwise well neonate; "
                "maternal anti-HPA antibody test; parental HPA genotyping. "
                "NAIT TREATMENT: HPA-compatible platelets (HPA-1b/1b for anti-HPA-1a); IVIG 1 g/kg × 2; "
                "maternal IVIG prophylaxis from 16–20 weeks in subsequent pregnancies. "
                "SCREENING CONTROVERSY: universal antenatal HPA typing proposed but not yet standard of care."
            ),
        },
        "pharmacological_distinctions": [
            "Recombinant Factor VIIa (rFVIIa, NovoSeven) in Glanzmann Thrombasthenia: dose 90–120 mcg/kg IV Q2h; mechanism: binds tissue factor at injury site → activates Factor X → thrombin burst on phosphatidylserine-exposing platelet membrane surface WITHOUT requiring αIIbβ3; approved by EMA for GT with inhibitors; effective even in platelet-transfusion-refractory GT; used preoperatively and for major mucosal bleeding; tranexamic acid 15 mg/kg IV co-administered; response typically evident within 2 hours",
            "Eltrombopag (Revolade/Promacta) in MYH9-RD and ANKRD26 THC2: oral TPO receptor agonist (c-Mpl, TPOR), 25–75 mg daily; stimulates megakaryocyte maturation and platelet production; raises platelet count ~30–50 × 10⁹/L in MYH9-RD (Italian registry); used pre-operatively (start 2–3 weeks before surgery, target >80 × 10⁹/L); ANKRD26 THC2 responds via constitutive MAPK activation; monitor LFTs weekly; contraindicated in severe hepatic impairment; off-label for inherited thrombocytopenia; does NOT modify malignancy risk in ANKRD26/RUNX1/ETV6",
            "Tranexamic acid in platelet function disorders: antifibrinolytic — competitive inhibitor of plasminogen lysine-binding sites → blocks fibrinolytic cascade; dose 10–25 mg/kg IV or 1–1.5 g oral for mucosal bleeding; mandatory co-administration with rFVIIa in GT; cornerstone of menorrhagia management in GT/BSS females; combined with combined oral contraceptive pill for menorrhagia prevention; IV formulation for perioperative use; contraindicated in haematuria (risk of clot in urinary tract obstruction)",
            "DDAVP (desmopressin, DDAVP) in BSS and GT — limited role: DDAVP releases endogenous VWF from Weibel-Palade bodies (endothelial) and alpha granules → temporarily increases circulating VWF; may improve initial platelet tethering in BSS (partial GPIbα-VWF interaction from residual GPIb); minimal benefit in GT (VWF-GPIbα axis intact but αIIbβ3 absent, limiting aggregation); trial DDAVP in mild BSS pre-operatively to assess response before planning surgery; dose 0.3 mcg/kg IV/SC or 300 mcg intranasal; tachyphylaxis after 1–2 doses",
            "IVIG in NAIT: high-dose IVIG 1 g/kg/day × 2 days → raises neonatal platelet count within 24–48h by blocking neonatal Fc receptor (FcRn) → reduces anti-HPA-1a catabolism + direct Fc receptor blockade on reticuloendothelial cells; maternal IVIG 1 g/kg/week from 16–20 weeks gestation in subsequent NAIT-at-risk pregnancies suppresses anti-HPA-1a titres; add dexamethasone if titres remain high; HPA-compatible platelets remain the fastest treatment for immediate severe thrombocytopenia",
            "Antiplatelet drugs — absolute contraindications in platelet disorders: aspirin (COX-1 inhibitor → impairs TXA2 generation → further reduces platelet aggregation), NSAIDs (ibuprofen, diclofenac — reversible COX inhibition), P2Y12 inhibitors (clopidogrel, prasugrel, ticagrelor — block ADP receptor → reduce secondary aggregation), GPIIb/IIIa antagonists (abciximab, eptifibatide, tirofiban — block αIIbβ3 directly, contraindicated in GT absolutely, may trigger sensitisation); all antiplatelet drugs CONTRAINDICATED in GT, BSS, RUNX1 FPD-AML, ANKRD26 THC2 — may cause life-threatening haemorrhage",
        ],
        "key_standards": [
            "International Society on Thrombosis and Haemostasis (ISTH) SSC Subcommittee on Platelet Physiology — Guidance on inherited platelet disorders: diagnostic approach including LTA (light transmission aggregometry), PFA-100, flow cytometry, EM, and genetic testing panels; guidance on management of GT (rFVIIa, tranexamic acid, platelet transfusion, inhibitor monitoring) and BSS (platelet transfusion, DDAVP trial, rFVIIa); published in Journal of Thrombosis and Haemostasis",
            "European Haematology Association (EHA) Guidelines on Inherited Platelet Disorders: standardised diagnostic workup (CBC, film, LTA, flow cytometry, genetic panel including ANKRD26 5′UTR, ETV6, RUNX1 MLPA); surveillance recommendations for RUNX1 FPD-AML and ETV6 THC5 (annual CBC, marrow biopsy thresholds); donor genotyping MANDATORY before HSCT; MYH9-RD management (Döhle body recognition, eltrombopag, no ITP treatment, nephritis/SNHL surveillance)",
            "NAIT — British Society for Haematology (BSH) Guidelines 2019: maternal anti-HPA antibody testing in all pregnancies with severe neonatal thrombocytopenia; parental HPA genotyping; HPA-compatible platelet transfusion as first-line for NAIT < 30 × 10⁹/L + IVIG; maternal IVIG prophylaxis from 16–20 weeks in at-risk subsequent pregnancies; caesarean section for unresolved fetal thrombocytopenia; neonatal cranial ultrasound screening for all NAIT neonates",
            "Thrombocytopenia 2 (ANKRD26) and Familial Platelet Disorder (RUNX1/ETV6) — International MDS Interest Group (IMDS) and EWOG-MDS Guidelines: germline testing for inherited AML/MDS predisposition in: unexplained familial thrombocytopenia + malignancy, thrombocytopenia resistant to ITP therapy, young patients with AML/MDS + family history; targeted panels must include ANKRD26 5′UTR Sanger + ETV6 exons + RUNX1 exons + MLPA; donor genotyping MANDATORY; surveillance (annual bone marrow) for RUNX1 and ETV6 carriers",
            "MYH9-Related Disease — Italian Registry / ISTH recommendations: peripheral blood smear diagnosis (Döhle-like inclusions + macro-thrombocytopenia); electron microscopy for subtle inclusions; genotype-phenotype correlation (head mutations → nephritis/SNHL; rod mutations milder); annual urine protein:creatinine + eGFR for all MYH9-RD; audiometry annually; no steroids/IVIG/rituximab/splenectomy — document this contraindication prominently in patient records; eltrombopag for preoperative optimisation",
            "GT — European Medicines Agency (EMA) approval of rFVIIa (NovoSeven) for GT with inhibitors: only licensed haemostatically active agent specifically approved for GT refractory to platelet transfusion; dose 90–120 mcg/kg Q2h IV; response within 2–4 hours; use with tranexamic acid; HPA typing and matched donor platelet registry recommended for all multi-transfused GT patients from first transfusion to minimise inhibitor formation",
        ],
    }
