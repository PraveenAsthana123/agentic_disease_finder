"""Hereditary MPN Predisposition Atlas — 8-Gene Reference
JAK2-CALR-MPL-SH2B3-EPOR-VHL-EPAS1-THPO
Familial MPN / Erythrocytosis / Essential Thrombocythemia Predisposition
320 patients (8 x 40), seeds 2862-2869.
Endpoints: /api/hereditary-mpn-predisposition-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "JAK2",
        "protein": (
            "JAK2 -- 9p24.1 AD somatic/germline -- 1132aa -- "
            "Janus-Kinase-2-130kDa-Non-Receptor-Tyrosine-Kinase-"
            "JAK-STAT-Signalling-Master-Regulator-"
            "V617F-Somatic-PV-95pct-ET-55pct-PMF-65pct-"
            "46/1-Haplotype-Germline-Predisposition-"
            "OMIM-Gene-147796-Disease-PV-263300-ET-187950-PMF-254450"
        ),
        "locus": "9p24.1",
        "protein_size": (
            "1132 aa / 130 kDa (Janus Kinase 2; non-receptor tyrosine kinase; "
            "four domains: FERM / SH2-like / pseudokinase (JH2) / kinase (JH1); "
            "JAK2 associates with cytokine receptors: EPOR (erythropoietin), TPOR/MPL (thrombopoietin), "
            "GCSFR (G-CSF) — does NOT have intrinsic membrane anchor; "
            "V617F (Val617Phe): pseudokinase domain mutation → constitutive kinase activation → "
            "continuous STAT3/STAT5 signalling → cytokine-independent growth; "
            "V617F allele burden: 1-50% heterozygous (ET/low-burden MF) vs >50% homozygous (PV/high-burden); "
            "46/1 haplotype (rs10974944 G allele): germline predisposition → 3-4x higher risk of acquiring V617F somatic; "
            "founder effect in European populations; "
            "exon 12 mutations (non-V617F): PV isolated erythrocytosis phenotype; "
            "CALR/MPL triple-negative MPN: neither JAK2 V617F nor exon 12 nor CALR nor MPL detected"
        ),
        "inheritance": (
            "SOMATIC DRIVER (AD clonal) + GERMLINE PREDISPOSITION (JAK2 46/1 haplotype): "
            "SOMATIC V617F (predominantly): "
            "  Acquired mutation in haematopoietic stem cell; not inherited Mendelian; "
            "  But FAMILIAL MPN clusters exist (2-5x risk in first-degree relatives of MPN patients); "
            "  JAK2 46/1 haplotype (rs10974944): germline variant; AD; "
            "    3-4x increased risk of developing JAK2 V617F-positive MPN; "
            "    MPN occurs on the haplotype chromosome (in cis); "
            "  Familial MPN inheritance: AD with incomplete penetrance (~2-5%); "
            "    Multiple family members with different MPN subtypes (one PV, one ET) — same JAK2 46/1 haplotype; "
            "CLINICAL SUBTYPES BY JAK2 V617F: "
            "  POLYCYTHAEMIA VERA (PV): JAK2 V617F/exon-12 in ~95%; erythrocytosis + low EPO; "
            "    WHO criteria: Hb >16.5 g/dL (M) / >16.0 g/dL (F) OR Hct >49% (M) / >48% (F) + BM biopsy + EPO; "
            "  ESSENTIAL THROMBOCYTHAEMIA (ET): JAK2 V617F in ~55%; platelets >450×10⁹/L; "
            "  PRIMARY MYELOFIBROSIS (PMF): JAK2 V617F in ~65%; splenomegaly + leukoerythroblastosis; "
            "ALLELE BURDEN: "
            "  Low (<25%): ET phenotype; lower thrombosis risk; "
            "  High (>50% — homozygous by mitotic recombination): PV/MF phenotype; higher fibrosis + transformation"
        ),
        "disease_category": (
            "MYELOPROLIFERATIVE NEOPLASM (MPN) — JAK2-DRIVEN: "
            "POLYCYTHAEMIA VERA (PV) — OMIM 263300: "
            "  Erythrocytosis (high Hct/Hb) + low erythropoietin (EPO); "
            "  Thrombosis (arterial + venous): Budd-Chiari, stroke, DVT/PE — LEADING CAUSE of morbidity; "
            "  Aquagenic pruritus (itch after hot bath): PATHOGNOMONIC — mast cell mediators; "
            "  Splenomegaly (50%); erythromelalgia; constitutional symptoms; "
            "  Transformation: PV → MF (secondary MF) in ~20% at 20 years; AML in ~5-10%; "
            "ESSENTIAL THROMBOCYTHAEMIA (ET): "
            "  Thrombocytosis >450×10⁹/L; microvascular symptoms (erythromelalgia, TIA, visual disturbance); "
            "  Thrombosis risk ~ WHO score / IPSET-thrombosis; "
            "  Haemorrhage (paradoxically — acquired VWD at very high platelet counts); "
            "  Transformation: ET → MF (10-15% at 15y); AML (<5%); "
            "PRIMARY MYELOFIBROSIS (PMF): "
            "  Bone marrow fibrosis; leukoerythroblastosis; splenomegaly; constitutional B-symptoms; "
            "  Inferior overall survival; transformation to AML 20-30%; "
            "  Ruxolitinib: first-line JAK1/2 inhibitor for symptom control + spleen reduction"
        ),
        "disease_pathway": (
            "JAK-STAT PATHWAY — CONSTITUTIVE ACTIVATION: "
            "NORMAL SIGNALLING: "
            "  EPO/TPO binds receptor → receptor dimerisation → JAK2 trans-phosphorylation → "
            "    STAT5 phosphorylation → nuclear translocation → BCL-XL / CCND1 / PIM1 → "
            "    proliferation + survival; signal terminated by SOCS3 (suppressor of cytokine signalling); "
            "V617F MECHANISM: "
            "  V617F in pseudokinase (JH2) domain → removes autoinhibitory constraint → "
            "    kinase (JH1) domain constitutively active → continuous STAT3/STAT5 phosphorylation; "
            "  Cytokine-independent; SOCS3 feedback partially overwhelmed; "
            "  Erythroid progenitors: EPO-hypersensitive + EPO-independent colonies (EEC) on culture; "
            "  Megakaryocyte progenitors: TPO-hypersensitive → platelet overproduction; "
            "DOWNSTREAM CONSEQUENCES: "
            "  STAT5 → BCL-XL (anti-apoptotic): erythroid progenitors survive without EPO; "
            "  STAT5 → PI3K/AKT: cell growth; "
            "  NF-κB, MAPK pathways: inflammatory cytokines (IL-6, IL-8, TNF) → constitutional symptoms; "
            "  EPIGENETIC DRIVERS: TET2, DNMT3A, ASXL1 mutations co-occur → accelerate transformation; "
            "JAK INHIBITOR MECHANISM: "
            "  Ruxolitinib: competitive ATP-binding JAK1/2 inhibitor; "
            "  Reduces STAT5 phosphorylation; reduces spleen size; reduces constitutional symptoms; "
            "  Does NOT eliminate clone; V617F allele burden falls slowly"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — JAK2-MPN: "
            "  1. AQUAGENIC PRURITUS (hot bath → intense generalised itching): "
            "     PV-PATHOGNOMONIC — 40-70% of PV; absent in secondary erythrocytosis; "
            "     caused by mast cell histamine release triggered by water + heat; "
            "     antihistamines only partially effective; ruxolitinib/phlebotomy improves; "
            "  2. SERUM EPO BELOW REFERENCE RANGE in erythrocytosis: "
            "     PV vs secondary erythrocytosis DDx critical; "
            "     Normal EPO in erythrocytosis = NOT PV (favours EPOR/VHL/EPAS1 gain); "
            "     Suppressed EPO = endogenous erythroid colonies expanding without EPO signal; "
            "  3. ENDOGENOUS ERYTHROID COLONIES (EEC): BFU-E growth in EPO-free medium — historical; "
            "  4. JAK2 V617F PCR (allele-specific): diagnostic for PV/ET/PMF; "
            "     QUANTITATIVE V617F allele burden: PV >50% usual; ET <50%; MF variable; "
            "  5. ERYTHROMELALGIA: burning pain + redness of hands/feet → ASPIRIN 75-100mg immediate relief; "
            "     PATHOGNOMONIC microvascular platelet-mediated occlusion; NSAIDs / ASA effective"
        ),
        "treatment": (
            "TREATMENT — JAK2-DRIVEN MPN: "
            "POLYCYTHAEMIA VERA: "
            "  PHLEBOTOMY: maintain Hct <45% (males), <42% (females); primary cytoreduction; "
            "  ASPIRIN 75-100 mg daily: all PV patients; arterial + venous thrombosis prevention; "
            "  HIGH-RISK (age >60 OR prior thrombosis): CYTOREDUCTION — hydroxycarbamide (HU) first-line; "
            "  RUXOLITINIB: HU-refractory or intolerant; JAK1/2 inhibitor; reduces spleen + symptoms; "
            "  INTERFERON-α (pegylated): preferred in young patients (<60y) and pregnancy; "
            "  AVOID: testosterone / erythropoiesis-stimulating agents; iron supplements (increases Hct); "
            "ESSENTIAL THROMBOCYTHAEMIA: "
            "  LOW RISK (<60y, no prior thrombosis, JAK2 V617F absent or low burden): observation ± aspirin; "
            "  HIGH RISK (age >60 OR prior thrombosis): HU (first-line); anagrelide (second-line); "
            "  PREGNANCY IN ET: aspirin throughout; LMWH post-partum; HU contraindicated — switch to interferon; "
            "PRIMARY MYELOFIBROSIS: "
            "  RUXOLITINIB: first-line in intermediate-2 or high-risk PMF (IPSS/DIPSS); "
            "  ALLOGENEIC HSCT: only curative option; consider in eligible patients ≤70y; "
            "  FEDRATINIB/PACRITINIB/MOMELOTINIB: ruxolitinib-failure or anaemia-dominant MF; "
            "  JAK INHIBITOR WITHDRAWAL SYNDROME: taper slowly — abrupt withdrawal → cytokine storm"
        ),
        "seed": 2862,
    },
    {
        "gene": "CALR",
        "protein": (
            "CALR -- 19p13.2 AD somatic -- 400aa -- "
            "Calreticulin-46kDa-ER-Chaperone-Ca2plus-Binding-"
            "Type1-del52bp-Most-Common-ET-PMF-"
            "Type2-ins5bp-Milder-Phenotype-ET-Predominant-"
            "OMIM-Gene-109091-Disease-ET-187950-PMF-254450"
        ),
        "locus": "19p13.2",
        "protein_size": (
            "400 aa / 46 kDa (Calreticulin; endoplasmic reticulum chaperone; "
            "three domains: N-domain (lectin-like), P-domain (Proline-rich, Ca2+ binding), C-domain (ER retention KDEL); "
            "NORMAL FUNCTION: ER quality control; glycoprotein folding; Ca2+ homeostasis; "
            "CALR MUTATIONS IN MPN: exclusive frameshift in exon 9 → altered C-terminus → "
            "  novel positively-charged C-terminus activates MPL (thrombopoietin receptor) constitutively; "
            "  CALR mutant + MPL → constitutive JAK-STAT signalling WITHOUT thrombopoietin; "
            "TYPE 1 (del52): 52-bp deletion → most common (~70%); associated with fibrosis; PMF phenotype; "
            "  Median platelet count lower; higher risk of MF transformation; "
            "TYPE 2 (ins5): 5-bp insertion → second most common (~20%); milder; ET phenotype; "
            "  Higher platelet counts; lower MF transformation risk; better survival; "
            "TYPE 1-LIKE vs TYPE 2-LIKE: 80+ rare variants classified by net protein charge shift"
        ),
        "inheritance": (
            "SOMATIC DRIVER (AD clonal) — CALR MUTATIONS IN JAK2-NEGATIVE MPN: "
            "  JAK2-negative ET: ~70-80% have CALR mutation; "
            "  JAK2-negative PMF: ~70-80% have CALR mutation; "
            "  JAK2 + CALR MUTUALLY EXCLUSIVE (practically never co-occur in same clone); "
            "MOLECULAR HIERARCHY (MPN driver mutation priority): "
            "  JAK2 V617F (55-65% ET; 95% PV; 65% PMF) >> CALR (25-30% ET; 0% PV; 25-30% PMF) > "
            "  MPL (5% ET; 8% PMF) > triple-negative (10-15%); "
            "CLINICAL PHENOTYPE DIFFERENCES BY MUTATION: "
            "  JAK2 V617F ET: higher Hb; higher thrombosis risk; older age; "
            "  CALR type 1 ET/PMF: highest platelet count; younger; higher MF transformation; "
            "  CALR type 2 ET: best prognosis; lowest transformation; "
            "  MPL ET: intermediate; "
            "  Triple-negative ET: heterogeneous; may carry non-driver somatic variants; "
            "FAMILIAL: CALR mutations not inherited but JAK2 46/1 haplotype may co-predispose to CALR mutation"
        ),
        "disease_category": (
            "CALR-POSITIVE MPN — OMIM 187950/254450: "
            "CALR MUTATION BIOLOGY: "
            "  CALR exon 9 frameshift → new C-terminus binds MPL ectodomain (domain of TPO binding); "
            "  Constitutive MPL activation → JAK2 activation → STAT5 → megakaryocyte/platelet expansion; "
            "  CALR mutant only activates MPL — cannot activate EPOR or other cytokine receptors; "
            "  Explains WHY CALR-positive MPN has THROMBOCYTOSIS PREDOMINANCE (not erythrocytosis); "
            "  PV (erythrocytosis): CALR mutation DOES NOT CAUSE PV — only JAK2 or EPOR/VHL/EPAS1; "
            "CALR TYPE 1 (del52) — MF RISK HIGH: "
            "  Median OS shorter than CALR type 2; higher rate of bone marrow fibrosis at presentation; "
            "  Grade MF-2/3 at diagnosis in ~30% PMF; "
            "  AML transformation: ~10-15% at 10y; "
            "CALR TYPE 2 (ins5) — BEST PROGNOSIS MPN: "
            "  OS approaches normal life expectancy in ET; lowest AML risk; "
            "  Rarely presents as PMF; predominantly ET throughout disease course; "
            "CALR IMMUNOTHERAPY TARGET: "
            "  Mutant CALR C-terminus: novel neoantigen (tumour-specific); "
            "  CAR-T cells and therapeutic vaccines targeting mutant CALR in clinical trials (2024-2026)"
        ),
        "disease_pathway": (
            "CALR MUTANT → MPL ACTIVATION → JAK-STAT: "
            "MECHANISM: "
            "  Wild-type CALR: ER chaperone, cannot activate MPL; "
            "  Mutant CALR (frameshift exon 9): new C-terminus with positive charges → "
            "    exits ER → presented on cell surface → binds MPL ectodomain (same binding site as TPO); "
            "    forms CALR mutant/MPL complex → MPL dimerisation → JAK2 trans-phosphorylation → "
            "    STAT5 phosphorylation → megakaryocytic proliferation; "
            "  CALR mutant → MPL → STAT5 → GATA-1, FLI1 (megakaryocyte TFs) → "
            "    platelet overproduction (thrombocytosis) + megakaryocyte dysplasia; "
            "TYPE 1 vs TYPE 2 PATHOGENICITY: "
            "  Type 1 (del52): longer new C-terminus; stronger MPL binding affinity; more constitutive signalling; "
            "  Type 2 (ins5): shorter positive stretch; weaker MPL binding; lower signalling intensity; "
            "  Explains phenotypic severity difference: Type 1 > Type 2 in MF risk; "
            "CO-MUTATIONS ACCELERATING PROGRESSION: "
            "  ASXL1 (chromatin remodelling): 20-30% of PMF; worst prognosis; "
            "  TP53 (tumour suppressor): AML transformation signal; "
            "  EZH2, IDH1/2 (epigenetic): blast-phase acceleration"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — CALR-MPN: "
            "  1. CALR EXON 9 FRAMESHIFT MUTATION in JAK2-NEGATIVE ET or PMF: "
            "     DIAGNOSTIC for MPN (sensitivity 70-80% in JAK2-neg ET/PMF); "
            "     MUST test CALR in any JAK2-negative thrombocytosis — never test JAK2 alone; "
            "  2. EXTREME THROMBOCYTOSIS (>1000×10⁹/L) in young patient: "
            "     CALR type 1/2 more likely than JAK2 in young (<45y) patients with isolated thrombocytosis; "
            "  3. ACQUIRED VWD (at platelets >1500×10⁹/L): "
            "     Platelets adsorb large VWF multimers → HMW-VWF lost → acquired type 2A-like VWD; "
            "     BLEEDING despite high platelets — VWF:RCo/Ag ratio <0.7 = acquired VWD; "
            "     HOLD cytoreduction if bleeding; cytoreduction corrects acquired VWD; "
            "  4. MPL RECEPTOR TARGET: Eltrombopag/romiplostim CONTRAINDICATED in MPN — "
            "     drives thrombocytosis; CALR mutant activates MPL constitutively"
        ),
        "treatment": (
            "TREATMENT — CALR-POSITIVE MPN: "
            "ESSENTIAL THROMBOCYTHAEMIA (CALR): "
            "  Same risk stratification as JAK2 ET (IPSET-thrombosis); "
            "  LOW RISK: observation ± aspirin 75mg; "
            "  HIGH RISK: hydroxycarbamide first-line; anagrelide second-line; "
            "  ACQUIRED VWD: withhold cytoreduction temporarily; "
            "    use DDAVP + tranexamic acid for bleeding; cytoreduction reduces platelet count → VWF recovery; "
            "  EXTREME THROMBOCYTOSIS (>1500): treat regardless of risk score; "
            "PMF (CALR TYPE 1): "
            "  RUXOLITINIB: first-line intermediate-2/high risk; reduces spleen + symptoms; "
            "  FEDRATINIB: ruxolitinib-failure; "
            "  MOMELOTINIB: anaemia-dominant MF; reduces transfusion dependency; "
            "  ALLOGENEIC HSCT: curative; consider eligible ≤70y intermediate-2/high; "
            "PREGNANCY AND CALR-ET: "
            "  Higher-risk of placental insufficiency + fetal loss than non-MPN; "
            "  Interferon-α (pegylated) preferred cytoreduction in pregnancy; "
            "  LMWH + aspirin throughout; "
            "INVESTIGATIONAL: "
            "  Anti-CALR antibody therapy (targeting neoantigen C-terminus): Phase I/II 2025-2026; "
            "  Luspatercept (SMAD2/3 inhibitor) for anaemia in MF: approved indications expanding"
        ),
        "seed": 2863,
    },
    {
        "gene": "MPL",
        "protein": (
            "MPL -- 1p34.2 AD somatic/germline -- 635aa -- "
            "Thrombopoietin-Receptor-TPOR-MPL-75kDa-"
            "Cytokine-Receptor-Superfamily-JAK2-Activator-"
            "W515L-W515K-Somatic-ET-PMF-"
            "P106L-Germline-Familial-Thrombocythaemia-"
            "OMIM-Gene-159530-Disease-ET-187950-Familial-Thrombocythaemia-614021"
        ),
        "locus": "1p34.2",
        "protein_size": (
            "635 aa / 75 kDa (Thrombopoietin receptor; MPL; myeloproliferative leukaemia virus oncogene; "
            "single transmembrane cytokine receptor; extracellular domain binds TPO (thrombopoietin/THPO); "
            "JAK2-associated intracellular domain; "
            "W515L (Trp515Leu): most common somatic MPN mutation in MPL; transmembrane/JM region; "
            "W515K (Trp515Lys): second most common somatic; "
            "W515 mutations: amphipathic helix disruption → constitutive receptor dimerisation without TPO; "
            "P106L (Pro106Leu): germline familial ET; signal peptide region; "
            "MPL S505N: somatic; hereditary thrombocythaemia 2 context; "
            "MPL W515 allele burden: typically lower than JAK2 V617F — heterozygous; "
            "MPL deficiency (LOF): congenital amegakaryocytic thrombocytopenia (CAMT) — opposite phenotype"
        ),
        "inheritance": (
            "SOMATIC (AD clonal — W515) + GERMLINE (AD — P106L, S505N familial ET): "
            "SOMATIC W515L/K: "
            "  Clonal haematopoietic stem cell mutation; not Mendelian; "
            "  ET: ~5-8% of ET patients (after JAK2 and CALR excluded); "
            "  PMF: ~8-10% of PMF; "
            "  PV: essentially absent (MPL is TPO receptor → thrombocytosis, not erythrocytosis); "
            "GERMLINE P106L — HEREDITARY THROMBOCYTHAEMIA (HT): "
            "  Autosomal dominant; Italian/Swedish founder variants; "
            "  Families: multiple members ET across 3+ generations; "
            "  THPO signalling amplification without MPL structural change; "
            "  Lower transformation risk than somatic MPN; "
            "GERMLINE S505N — FAMILIAL ET TYPE 2: "
            "  AD; Dutch/Belgian founders; transmembrane domain; "
            "  Similar phenotype to P106L; "
            "  Risk: bone marrow fibrosis in 10-15% long-term; "
            "CLINICAL CORRELATION: "
            "  Triple-negative MPN (neg JAK2, CALR, MPL): 10-15%; may have non-driver somatic variants; "
            "  MPL-positive MPN: older age at presentation; lower Hb; higher transfusion rate in MF"
        ),
        "disease_category": (
            "MPL-DRIVEN MPN — OMIM 187950/614021: "
            "MPL SOMATIC (W515L/K) — PHENOTYPE: "
            "  THROMBOCYTOSIS (platelets >450×10⁹/L) ± fibrosis; "
            "  Lower haemoglobin than JAK2 V617F MPN (MPL activates megakaryocytes, not erythroid progenitors); "
            "  Splenomegaly in PMF; leukoerythroblastosis; constitutional symptoms; "
            "  Bone marrow biopsy: megakaryocyte clustering + dysplasia (cloud-like nuclei in PMF); "
            "  Thrombosis risk: intermediate (lower than JAK2 V617F); "
            "MPL GERMLINE (P106L) — HEREDITARY THROMBOCYTHAEMIA: "
            "  Familial; milder clinical course than somatic MPN; "
            "  Thrombosis risk: reduced vs somatic ET; "
            "  Bleeding at extreme thrombocytosis (acquired VWD); "
            "  DISTINGUISH FROM REACTIVE THROMBOCYTOSIS: reactive = iron deficiency, infection, post-splenectomy; "
            "    MPL familial ET persists without reversible cause; BM shows megakaryocyte hyperplasia; "
            "CONGENITAL AMEGAKARYOCYTIC THROMBOCYTOPENIA (CAMT): "
            "  MPL LOF (AR): opposite end of spectrum — absent megakaryocytes → severe thrombocytopenia → aplastic anaemia; "
            "  Treat with HSCT; TPO-mimetics DO NOT WORK (no functional receptor)"
        ),
        "disease_pathway": (
            "MPL-TPO-JAK-STAT PATHWAY: "
            "NORMAL TPO/MPL AXIS: "
            "  TPO (THPO gene) produced constitutively by liver and kidney; "
            "  TPO binds MPL on megakaryocytes → MPL dimerisation → JAK2 activation → "
            "    STAT3/STAT5/PI3K → megakaryocyte proliferation + platelet release; "
            "  Negative feedback: platelets + megakaryocytes sequester TPO → high platelet count → "
            "    less free TPO → less MPL signalling; "
            "MPL W515 CONSTITUTIVE ACTIVATION: "
            "  W515 in juxtamembrane domain: normally inhibitory hydrophobic interaction; "
            "  W515L/K: charged residue → breaks inhibitory constraint → "
            "    MPL dimerises spontaneously WITHOUT TPO → continuous JAK2-STAT5 signalling; "
            "  TPO level in MPL-driven MPN: elevated (platelets not clearing TPO efficiently + "
            "    feedback disrupted); serum TPO >600 pg/mL = ELEVATED (unlike JAK2 PV where EPO suppressed); "
            "DOWNSTREAM: "
            "  STAT5 → GATA-1, FLI1 → megakaryocyte differentiation + platelet hyperproduction; "
            "  NF-κB inflammatory signalling → fibrosis-promoting cytokines (TGF-β, PDGF, bFGF) → "
            "    reticulin/collagen fibrosis in PMF"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — MPL-MPN: "
            "  1. MPL W515L/K BY NGS/ALLELE-SPECIFIC PCR in JAK2-NEGATIVE, CALR-NEGATIVE ET or PMF: "
            "     DIAGNOSTIC for MPL-driven MPN; MPN always ruled out in this order: JAK2 → CALR → MPL; "
            "  2. SERUM TPO ELEVATED (>600 pg/mL) in MPL-driven ET/PMF: "
            "     Unlike JAK2-PV (EPO suppressed) or EPOR/VHL/EPAS1 (EPO elevated); "
            "     TPO elevated because feedback loop disrupted; "
            "  3. MPL GERMLINE TESTING in familial thrombocythaemia: "
            "     3+ family members ET across generations → test germline MPL P106L/S505N; "
            "  4. CAMT PHENOTYPE (LOF MPL): NEONATAL/INFANT THROMBOCYTOPENIA + normal-looking marrow on biopsy; "
            "     TPO ELEVATED + MPL absent on megakaryocyte surface — TPO-mimetics futile; "
            "  5. ACQUIRED VWD (as with CALR-ET): "
            "     platelets >1500×10⁹/L → HMW-VWF adsorption → bleeding despite thrombocytosis"
        ),
        "treatment": (
            "TREATMENT — MPL-POSITIVE MPN: "
            "ET (SOMATIC W515 OR GERMLINE P106L): "
            "  Same risk-adapted strategy as JAK2/CALR ET; "
            "  LOW RISK (<60y, no prior thrombosis): observation ± aspirin 75mg; "
            "  HIGH RISK: hydroxycarbamide first-line; anagrelide second-line; "
            "  ACQUIRED VWD: cytoreduction to <1000×10⁹/L platelets; DDAVP + tranexamic acid acutely; "
            "PMF (SOMATIC W515): "
            "  RUXOLITINIB: intermediate-2/high IPSS/DIPSS; reduces spleen + cytokine storm; "
            "  ALLOGENEIC HSCT: curative intent; RIC conditioning preferred >50y; "
            "FAMILIAL ET (GERMLINE P106L/S505N): "
            "  Generally milder; lower transformation risk than somatic MPN; "
            "  Observation in young asymptomatic carriers; aspirin if high platelet burden; "
            "  INFORM FIRST-DEGREE RELATIVES: test siblings/children (AD); "
            "TPO-MIMETICS (eltrombopag/romiplostim): "
            "  CONTRAINDICATED in MPL-positive MPN — activates constitutively-active receptor; "
            "  USE in CAMT (MPL LOF) as bridge-to-HSCT in responsive cases (paradoxically may work in hypomorphic LOF)"
        ),
        "seed": 2864,
    },
    {
        "gene": "SH2B3",
        "protein": (
            "SH2B3 -- 12q24.12 AD germline LOF -- 575aa -- "
            "LNK-SH2B3-SH2-Domain-Adaptor-JAK2-Negative-Regulator-"
            "Germline-LOF-Familial-MPN-Predisposition-"
            "Somatic-LOF-Clone-Amplification-"
            "OMIM-Gene-605093-Disease-Familial-MPN-Predisposition"
        ),
        "locus": "12q24.12",
        "protein_size": (
            "575 aa / 68 kDa (LNK; SH2B adaptor family; "
            "domain structure: PH domain / disordered region / SH2 domain / C-terminal helix; "
            "FUNCTION: cytoplasmic adaptor protein; negative regulator of cytokine receptor signalling; "
            "binds phospho-JAK2 via SH2 domain → inhibits JAK2 kinase activity → "
            "suppresses STAT5 phosphorylation and downstream proliferation; "
            "LNK (=SH2B3) constrains signalling from EPOR, TPOR (MPL), KIT, and FLT3; "
            "GERMLINE LOF: reduced LNK → JAK2 signalling amplified even with wild-type JAK2; "
            "predisposes to JAK2 V617F acquisition (3-4x higher risk per reported family studies); "
            "SOMATIC LOF SH2B3: in JAK2 V617F MPN → cooperative amplification of MPN phenotype; "
            "GWAS (2009): SH2B3 rs3184504 (p.R262W): common variant (population-level MPN risk); "
            "also protective against parasitic infections / malaria (balancing selection)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GERMLINE LOF — FAMILIAL MPN PREDISPOSITION: "
            "GERMLINE SH2B3 LOF: "
            "  Rare; reported in familial MPN clusters; "
            "  Loss of LNK function → lower threshold for JAK2 V617F-driven MPN; "
            "  Families: multiple members PV, ET, or MF with SH2B3 germline variant; "
            "  Penetrance: incomplete; many germline LOF carriers remain unaffected; "
            "  Predisposition mechanism: even low-level JAK2 V617F allele burden causes MPN "
            "    (less LNK suppression needed for disease); "
            "SH2B3 rs3184504 (p.R262W) — POPULATION VARIANT: "
            "  Common SNP (minor allele ~50% Europeans); partial LOF; "
            "  Modestly elevated MPN risk at population level; "
            "  ALSO: GWAS hit for autoimmune (type 1 diabetes, coeliac, rheumatoid arthritis) — "
            "    LNK also suppresses B-cell receptor and TCR signalling; "
            "SOMATIC BIALLELIC SH2B3 LOF (secondary): "
            "  Acquired in MPN → accelerated disease (blast transformation); "
            "  Found in AML-MPN transformation; "
            "ASSOCIATION: "
            "  SH2B3 LOF enriched in JAK2 46/1 haplotype carriers; "
            "  Both together confer additive MPN predisposition"
        ),
        "disease_category": (
            "FAMILIAL MPN PREDISPOSITION — SH2B3/LNK: "
            "LNK AS MPN AMPLIFIER: "
            "  LNK does not CAUSE MPN alone; removes suppression → JAK2 V617F or other driver thrives; "
            "  Germline SH2B3 LOF = lower threshold for acquiring clinical MPN from any driver mutation; "
            "CLINICAL FEATURES OF SH2B3-PREDISPOSED MPN: "
            "  Phenotype determined by co-occurring driver (JAK2/CALR/MPL); "
            "  Trend: higher haematocrit + platelet counts for given V617F allele burden vs non-SH2B3; "
            "  May present at younger age than typical MPN; "
            "AUTOIMMUNE OVERLAP: "
            "  SH2B3 LOF → increased B-cell/T-cell sensitivity → autoimmune diseases; "
            "  Type 1 diabetes, coeliac disease, rheumatoid arthritis over-represented; "
            "  Evaluate for autoimmune co-morbidities in SH2B3 germline carriers; "
            "CARDIOVASCULAR RISK: "
            "  rs3184504 (p.R262W) common variant: associated with platelet reactivity + coronary artery disease; "
            "BIALLELIC SOMATIC: "
            "  AML-MPN transformation context; poor prognosis; "
            "  Consider HSCT evaluation early in SH2B3-LOF MPN-AML"
        ),
        "disease_pathway": (
            "LNK-JAK2 NEGATIVE REGULATORY AXIS: "
            "LNK MECHANISM: "
            "  LNK SH2 domain: binds pY1007 phospho-JAK2 (activated JAK2); "
            "  LNK binding → prevents JAK2 substrate recruitment → terminates STAT5 signalling; "
            "  Also binds MPL (pY572) → inhibits TPO-induced JAK2 activation; "
            "  LNK PH domain: membrane recruitment near cytokine receptor complexes; "
            "LNK LOF CONSEQUENCE: "
            "  Phospho-JAK2 not suppressed → longer STAT5 activation per cytokine stimulus; "
            "  Erythroid progenitors: EPO hypersensitivity (even with WT JAK2 and WT EPOR); "
            "  Megakaryocytes: TPO hypersensitivity → thrombocytosis; "
            "  In V617F context: constitutive signalling further amplified → earlier/more severe MPN; "
            "NEGATIVE FEEDBACK LOOP: "
            "  Normal: cytokine → JAK2 → STAT5 → SOCS3 + LNK induction → negative feedback; "
            "  LNK LOF: SOCS3 still induced but primary non-catalytic suppressor missing → "
            "    incomplete feedback; chronic cytokine hypersensitivity; "
            "EVOLUTIONARY CONTEXT: "
            "  SH2B3 LOF alleles maintained in human populations at higher frequency than expected → "
            "    balancing selection: LNK LOF = protection against malaria / visceral leishmaniasis "
            "    (enhanced immune response) at cost of autoimmune + MPN susceptibility"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — SH2B3-MPN PREDISPOSITION: "
            "  1. FAMILIAL MPN CLUSTER + GERMLINE SH2B3 LOF: "
            "     Multiple family members MPN (different subtypes); germline SH2B3 LOF explains cluster; "
            "     REFER TO GENETICS CLINIC for counselling; screen first-degree relatives for MPN; "
            "  2. AUTOIMMUNE + MPN COMBINATION: "
            "     T1D or coeliac or RA in MPN patient or family → test SH2B3 germline status; "
            "  3. JAK2 V617F WITH DISPROPORTIONATELY HIGH ALLELE BURDEN FOR CLINICAL STAGE: "
            "     Suggesting amplified JAK2 signalling; consider co-occurring SH2B3 LOF; "
            "  4. YOUNG MPN (<40y) WITH FAMILY HISTORY: "
            "     Evaluate germline predisposition panel: JAK2 46/1 haplotype + SH2B3 + EPOR; "
            "  5. SOMATIC BIALLELIC SH2B3 IN BLAST PHASE MPN: "
            "     High-risk AML transformation; evaluate for HSCT urgently"
        ),
        "treatment": (
            "TREATMENT — SH2B3/LNK-PREDISPOSED MPN: "
            "TREAT THE CO-OCCURRING MPN SUBTYPE per standard guidelines: "
            "  PV: phlebotomy + aspirin ± HU; ruxolitinib if HU-refractory; "
            "  ET: risk-stratified HU or anagrelide; "
            "  PMF: ruxolitinib first-line; HSCT curative; "
            "GERMLINE COUNSELLING: "
            "  Autosomal dominant predisposition; 50% transmission risk; "
            "  Screen first-degree relatives with FBC annually; "
            "  Discuss implications for MPN surveillance; "
            "  PREIMPLANTATION GENETIC TESTING (PGT): option for affected families; "
            "AUTOIMMUNE CO-MORBIDITIES: "
            "  Screen annually for T1D (HbA1c + fasting glucose); coeliac (anti-TTG IgA); "
            "  Rheumatological review if joint symptoms; "
            "BLAST PHASE (biallelic somatic SH2B3): "
            "  Treat as AML (induction chemotherapy); "
            "  ALLOGENEIC HSCT: early consideration given poor prognosis of MPN-AML; "
            "  Venetoclax-based regimens: emerging data in MPN-AML transformation; "
            "RUXOLITINIB: reduces JAK2 signalling regardless of LNK status; "
            "  effective in SH2B3-LOF MPN same as standard MPN"
        ),
        "seed": 2865,
    },
    {
        "gene": "EPOR",
        "protein": (
            "EPOR -- 19p13.2 AD germline GOF truncation -- 508aa -- "
            "Erythropoietin-Receptor-55kDa-Single-Pass-Transmembrane-"
            "Haematopoietin-Receptor-Superfamily-JAK2-Activator-"
            "C-Terminal-Truncation-Removes-Negative-Regulatory-Domain-"
            "Familial-Erythrocytosis-Type-1-OMIM-133100"
        ),
        "locus": "19p13.2",
        "protein_size": (
            "508 aa / 55 kDa (Erythropoietin receptor; EPOR; single-pass transmembrane type I glycoprotein; "
            "extracellular domain: EPO binding (CRH fold); transmembrane domain; "
            "intracellular domain: Box1 (JAK2 binding) / Box2 / C-terminal regulatory region; "
            "C-TERMINAL (aa 427-508): negative regulatory domain — contains phosphotyrosine docking sites "
            "for SOCS3 and SHP-1 (tyrosine phosphatase) → SIGNAL TERMINATION; "
            "FAMILIAL ERYTHROCYTOSIS TYPE 1 (ECYT1): C-terminal truncation mutations → "
            "remove negative regulatory domain → SOCS3/SHP-1 cannot bind → JAK2-STAT5 signal prolonged → "
            "erythroid progenitors hypersensitive to EPO → polycythaemia; "
            "Finnish founder mutation (p.5964delAG + p.Q439X): accounts for 80%+ of Finnish ECYT1; "
            "EPO level: LOW (suppressed by polycythaemia, intact EPO-sensing feedback from kidney); "
            "JAK2 V617F negative — must not be called PV without JAK2 testing"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — FAMILIAL ERYTHROCYTOSIS TYPE 1 (ECYT1): "
            "EPOR C-TERMINAL TRUNCATION (gain of function via loss of regulation): "
            "  Heterozygous truncating mutations sufficient for dominant phenotype; "
            "  C-terminus missing → SOCS3/SHP-1 binding sites absent → EPO signalling duration prolonged; "
            "  Erythroid progenitors: hypersensitive to EPO + EPO-independent growth at low concentration; "
            "CLINICAL PRESENTATION: "
            "  Erythrocytosis (high Hct/Hb) from childhood or early adulthood; "
            "  Plethora, headache, visual disturbance; "
            "  THROMBOSIS: deep vein thrombosis, stroke, Budd-Chiari — primary morbidity; "
            "  Splenomegaly: mild-moderate; "
            "  NO thrombocytosis (unlike JAK2 PV); "
            "  NO leucocytosis; PURE RED CELL EXPANSION; "
            "DISTINGUISH FROM JAK2 PV: "
            "  EPOR ECYT1: pure erythrocytosis only; platelets/WBCs NORMAL; JAK2 V617F NEGATIVE; "
            "  JAK2 PV: panmyelosis (erythrocytosis + thrombocytosis + leucocytosis common); "
            "PENETRANCE: high; affected individuals in each generation; "
            "FINN-MAJOR POPULATION: p.5964delAG is founder mutation in Finland (published Scandinavian studies)"
        ),
        "disease_category": (
            "FAMILIAL ERYTHROCYTOSIS TYPE 1 (ECYT1) — OMIM 133100: "
            "ECYT1 CORE FEATURES: "
            "  Autosomal dominant congenital erythrocytosis; "
            "  Haematocrit typically 50-60% (males), 48-58% (females) at presentation; "
            "  Haemoglobin: elevated (usually >18 g/dL in males); "
            "  Erythropoietin: SUPPRESSED (below lower reference range) — identical to JAK2 PV; "
            "  Reticulocyte count: normal to mildly elevated; "
            "  Ferritin: reduced (iron consumed by excess erythropoiesis); "
            "  Platelets/WBCs: NORMAL — pure erythroid expansion (unlike PV panmyelosis); "
            "  Bone marrow: erythroid hyperplasia WITHOUT megakaryocyte/granulocyte expansion; "
            "MOLECULAR DIAGNOSIS: "
            "  WHO PV criteria not met (no JAK2 V617F/exon12, no BM pathology); "
            "  Algorithm: erythrocytosis + suppressed EPO + JAK2-NEGATIVE → EPOR sequencing; "
            "  If EPOR negative: VHL, EPAS1, EGLN1, HBB/HBA2 sequencing; "
            "COMPLICATIONS: "
            "  THROMBOSIS (25-30%): DVT, pulmonary embolism, stroke, Budd-Chiari; "
            "  HYPERTENSION (30%); "
            "  No transformation to MF or AML (benign chronic condition)"
        ),
        "disease_pathway": (
            "EPOR TRUNCATION → PROLONGED JAK-STAT SIGNALLING: "
            "NORMAL EPOR SIGNALLING CYCLE: "
            "  EPO → EPOR homodimerisation → JAK2 trans-phosphorylation → STAT5 pY694 → "
            "    BCL-XL (survival) / CCND1 (proliferation) / EPO receptor itself; "
            "  Negative regulators recruited to C-terminus: "
            "    SHP-1 (PTP1C): binds pY429 + pY431 → dephosphorylates JAK2 → signal termination; "
            "    SOCS3: binds phospho-JAK2 via SH2 → E3 ubiquitin ligase → JAK2 degradation; "
            "    CIS: STAT5-induced; binds pY401 → competes with STAT5 for receptor; "
            "TRUNCATION CONSEQUENCE: "
            "  pY429/431 (SHP-1 sites) removed → SHP-1 cannot bind → JAK2 stays phosphorylated longer; "
            "  SOCS3 binding sites removed → JAK2 not degraded → sustained signalling; "
            "  Each EPO stimulus generates ~3-5x longer STAT5 activation than wild-type; "
            "  BCL-XL overexpressed → erythroid progenitors survive without EPO; "
            "  ERYTHROID BURST-FORMING UNITS (BFU-E): EPO-hypersensitive + EPO-independent colonies; "
            "HAEMATOCRIT-EPO FEEDBACK: "
            "  High Hct → renal O2 saturation normal → EPO mRNA transcription suppressed; "
            "  EPO level appropriately suppressed; paradoxically patients feel better than true EPO-independent; "
            "  Phlebotomy transiently raises EPO; EPOR hypersensitivity makes even small EPO rise effective"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — EPOR-ECYT1: "
            "  1. ERYTHROCYTOSIS + SUPPRESSED EPO + JAK2 NEGATIVE: "
            "     ALGORITHM: Hb/Hct elevated → EPO measured → low EPO + JAK2 V617F/exon12 NEGATIVE → "
            "     test EPOR sequencing FIRST (most common hereditary erythrocytosis with low EPO); "
            "  2. EPOR C-TERMINAL TRUNCATION on sequence: DIAGNOSTIC; "
            "     ANY frameshift or nonsense in exons 7-8 (C-terminal domain) = functional ECYT1; "
            "  3. PURE ERYTHROID EXPANSION on bone marrow biopsy: "
            "     Erythroid hyperplasia WITHOUT megakaryocyte or granulocyte expansion; "
            "     Distinguishes ECYT1 from JAK2 PV (panmyelosis); "
            "  4. FAMILIAL ERYTHROCYTOSIS: parent + child both affected (AD); "
            "  5. ENDOGENOUS ERYTHROID COLONIES (EEC) in EPO-free medium: "
            "     BFU-E colonies grow without added EPO — same as PV but JAK2-negative; "
            "     Historical diagnostic test; now replaced by molecular sequencing"
        ),
        "treatment": (
            "TREATMENT — EPOR-ECYT1: "
            "PHLEBOTOMY — MAINSTAY: "
            "  Target Hct <45% (males), <42% (females) — same as PV; "
            "  Frequency: individually tailored; monthly initially; 3-4×/year maintenance; "
            "  IRON DEFICIENCY from chronic phlebotomy: do NOT supplement iron (increases Hct); "
            "  Monitor: FBC + ferritin; withhold phlebotomy if Hb <13 g/dL (M); "
            "ASPIRIN 75-100 mg daily: "
            "  Thrombosis prevention; all erythrocytosis patients with Hct >50% or prior thrombosis; "
            "HYDROXYCARBAMIDE: "
            "  Rarely needed; poor tolerance long-term in young/benign disease; "
            "  Reserved for frequent phlebotomy need (>6×/year) or thrombotic events despite Hct control; "
            "RUXOLITINIB: "
            "  Not standard for ECYT1; JAK2-inhibitor in EPO hypersensitivity context — theoretical benefit; "
            "  Case reports only; not approved indication; "
            "FAMILY SCREENING: "
            "  Test first-degree relatives with FBC; "
            "  EPOR sequencing if Hb/Hct elevated; "
            "  Genetic counselling: 50% transmission; "
            "PREGNANCY: "
            "  High-risk for VTE; LMWH throughout + post-partum; "
            "  Aggressive Hct control with phlebotomy; target Hct <42% throughout pregnancy"
        ),
        "seed": 2866,
    },
    {
        "gene": "VHL",
        "protein": (
            "VHL -- 3p25.3 AD LOF germline + AR biallelic LOF -- 213aa -- "
            "Von-Hippel-Lindau-Tumour-Suppressor-24kDa-"
            "pVHL-Ubiquitin-E3-Ligase-Adaptor-HIF-1alpha-2alpha-Degradation-"
            "VHL-Syndrome-RCC-Haemangioblastoma-Phaeochromocytoma-"
            "Familial-Erythrocytosis-Type-3-Chuvash-p.R200W-OMIM-193300-263400"
        ),
        "locus": "3p25.3",
        "protein_size": (
            "213 aa / 24 kDa (Von Hippel-Lindau protein; tumour suppressor; "
            "forms VCB-CRL2 ubiquitin E3 ligase complex: pVHL + Elongin-B/C + Cullin-2 + RBX1; "
            "pVHL FUNCTION: recognises hydroxylated HIF-1α/HIF-2α (hydroxylated by PHD2/EGLN1) → "
            "polyubiquitinates HIF-α → proteasomal degradation → prevents HIF target gene induction; "
            "THREE CLINICALLY DISTINCT VHL MUTATION CLASSES: "
            "  Type 1 (truncating/large deletion): high haemangioblastoma + RCC risk; low phaeochromocytoma; "
            "  Type 2A (missense): high phaeochromocytoma + haemangioblastoma; low RCC; "
            "  Type 2B (missense): high phaeochromocytoma + haemangioblastoma + RCC — HIGHEST risk; "
            "  Type 2C (missense: p.L188V, p.R161Q): isolated phaeochromocytoma; very low RCC/HB; "
            "CHUVASH POLYCYTHAEMIA (p.R200W): homozygous or compound het VHL LOF → isolated erythrocytosis; "
            "  Endemic in Chuvash people (Russia/Eurasia); AR erythrocytosis without VHL syndrome tumours"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (VHL syndrome — heterozygous) + AUTOSOMAL RECESSIVE (Chuvash polycythaemia): "
            "VHL SYNDROME (OMIM 193300) — AD: "
            "  Germline single VHL allele mutated → tumour suppressor function reduced; "
            "  Second hit (LOH or somatic mutation) in target tissue → biallelic VHL inactivation → tumours; "
            "  Penetrance: nearly 100% by age 65 for at least one manifestation; "
            "  Tumour spectrum: cerebellar/spinal haemangioblastoma, clear cell RCC, phaeochromocytoma, "
            "    pancreatic NETs, endolymphatic sac tumours (ELST), epididymal cystadenoma; "
            "  Erythrocytosis: minor feature; present in some VHL syndrome families (HIF activation in kidneys); "
            "CHUVASH POLYCYTHAEMIA (OMIM 263400) — AR: "
            "  p.R200W: Chuvash founder allele (3p25.3:c.598C>T); "
            "  HOMOZYGOUS: partial loss VHL function → HIF constantly stabilised → EPO overproduction + "
            "    erythrocytosis + VTE + pulmonary hypertension; NO TUMOURS (Chuvash variant); "
            "  HETEROZYGOUS (carriers): often normal or mildly raised EPO; "
            "FAMILIAL ERYTHROCYTOSIS TYPE 3 (ECYT3): heterozygous VHL missense (non-Chuvash); "
            "  Pure erythrocytosis; EPO elevated or normal-high; NO VHL tumour syndrome"
        ),
        "disease_category": (
            "VHL SYNDROME + FAMILIAL ERYTHROCYTOSIS TYPE 3 (ECYT3) + CHUVASH POLYCYTHAEMIA: "
            "VHL SYNDROME TUMOUR PROFILE: "
            "  Haemangioblastoma (HB): cerebellar > spinal > retinal (retinal HB = von Hippel disease); "
            "    Retinal angioma: earliest manifestation; fundoscopy + FFA screening from age 5; "
            "  Clear Cell RCC: bilateral/multifocal; size criterion for surgery (<3 cm active surveillance); "
            "    Belzutifan (HIF-2α inhibitor): FDA2021 for VHL-RCC not requiring immediate surgery; "
            "  Phaeochromocytoma: biochemically confirmed first (24h urine catecholamines / plasma metanephrines); "
            "  Pancreatic NET + serous cystoadenoma; ELST (ear, hearing loss); epididymal cystadenoma (bilateral); "
            "CHUVASH POLYCYTHAEMIA: "
            "  Erythrocytosis + THROMBOSIS (VTE + arterial); PULMONARY HYPERTENSION; "
            "  Lower OS than general population; "
            "  EPO: normal or elevated (VHL LOF → HIF-2α stabilised → EPO target gene induced constitutively); "
            "  Distinguish from JAK2 PV: JAK2 negative + elevated EPO (not suppressed); "
            "  Phlebotomy + anticoagulation for VTE; belzutifan experimental"
        ),
        "disease_pathway": (
            "VHL-HIF-EPO OXYGEN-SENSING PATHWAY: "
            "NORMAL VHL FUNCTION (NORMOXIA): "
            "  O2 present → PHD2 (EGLN1) hydroxylates HIF-1α/2α at Pro402 + Pro564 (HIF-1α); "
            "  Hydroxylated HIF-α: recognised by pVHL (ODD domain binding) → "
            "    VCB-CRL2 complex → K48-polyubiquitin → 26S proteasome → HIF-α degradation; "
            "  HIF target genes OFF: VEGF, EPO, GLUT1, PDK1 suppressed; "
            "HYPOXIA / VHL LOSS: "
            "  Hypoxia: PHD2 O2-dependent → hydroxylation blocked → HIF-α NOT hydroxylated → "
            "    pVHL CANNOT bind → HIF-α accumulates → nuclear translocation with HIF-1β (ARNT); "
            "    HIF-1/2 heterodimers → HIF-responsive elements (HRE) → EPO gene induction; "
            "  VHL LOF: normoxic HIF stabilisation (as if always hypoxic); "
            "  EPO overproduction in kidney + liver → erythrocytosis; "
            "  VEGF overproduction → haemangioblastoma pathogenesis; "
            "  HIF-2α (encoded by EPAS1): principal driver of EPO induction in VHL-null tumours; "
            "BELZUTIFAN (WELIREG) — HIF-2α INHIBITOR: "
            "  Binds HIF-2α PAS-B domain → disrupts HIF-2α/HIF-1β dimerisation → "
            "    HIF-2α target genes (EPO, VEGF) suppressed without affecting HIF-1α; "
            "  Reduces erythrocytosis and VHL-RCC tumour growth"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — VHL: "
            "  1. RETINAL HAEMANGIOBLASTOMA (von Hippel disease): "
            "     FIRST MANIFESTATION of VHL syndrome in 50% of cases; onset age 10-30; "
            "     Annual ophthalmology from age 5 in VHL gene carriers; "
            "     Untreated: retinal detachment + blindness; "
            "  2. VHL SYNDROME TRIAD (classic): haemangioblastoma (cerebellum) + RCC (bilateral) + phaeochromocytoma; "
            "  3. CHUVASH POLYCYTHAEMIA (AR p.R200W): erythrocytosis + ELEVATED EPO (not suppressed as in JAK2 PV); "
            "     JAK2 NEGATIVE + EPO ELEVATED = NOT PV → test VHL, EPAS1, EGLN1; "
            "  4. BILATERAL CLEAR CELL RCC < 50y: VHL germline testing MANDATORY; "
            "  5. BELZUTIFAN THERAPY MONITOR: anaemia (EPO suppression below normal) during treatment; "
            "     dose-related erythropoiesis suppression; hold if Hb <8 g/dL"
        ),
        "treatment": (
            "TREATMENT — VHL SYNDROME + FAMILIAL ERYTHROCYTOSIS TYPE 3: "
            "VHL SYNDROME SURVEILLANCE PROGRAMME (annual unless noted): "
            "  Ophthalmology (fundoscopy ± FFA): from age 5; every 1-2y; "
            "  Brain + spine MRI: from age 11; every 2y; "
            "  Abdominal MRI: from age 16; every 2y; "
            "  24h urine catecholamines / plasma metanephrines: annually; "
            "  Audiological assessment: periodic; "
            "TUMOUR MANAGEMENT: "
            "  RCC <3 cm: active surveillance; ≥3 cm: nephron-sparing surgery (bilateral disease → preserve function); "
            "  BELZUTIFAN (HIF-2α inhibitor): VHL-RCC not requiring immediate surgery (FDA 2021); "
            "  Haemangioblastoma: surgery if symptomatic/growing; gamma knife for small cerebellar; "
            "  Phaeochromocytoma: adrenalectomy (laparoscopic); pre-op alpha-blockade (phenoxybenzamine); "
            "ERYTHROCYTOSIS (VHL/Chuvash): "
            "  PHLEBOTOMY: Hct target <45% (M) / <42% (F); "
            "  ASPIRIN: thrombosis prevention; "
            "  ANTICOAGULATION: if VTE history (Chuvash — high thrombosis rate); "
            "  BELZUTIFAN: investigational for Chuvash polycythaemia (HIF-2α drives EPO in both VHL-RCC and Chuvash); "
            "GENETIC COUNSELLING: AD (VHL syndrome — 50% transmission); AR (Chuvash — carrier testing in at-risk communities)"
        ),
        "seed": 2867,
    },
    {
        "gene": "EPAS1",
        "protein": (
            "EPAS1 -- 2p21 AD GOF germline/somatic -- 870aa -- "
            "HIF-2alpha-Hypoxia-Inducible-Factor-2alpha-97kDa-"
            "bHLH-PAS-Domain-Transcription-Factor-"
            "Gain-of-Function-GOF-Stabilisation-EPO-VEGF-Overproduction-"
            "Familial-Erythrocytosis-Type-4-Paraganglioma-Polycythaemia-"
            "OMIM-Gene-603349-Disease-ECYT4-611783-Paraganglioma-Polycythaemia-"
        ),
        "locus": "2p21",
        "protein_size": (
            "870 aa / 97 kDa (HIF-2α; Hypoxia-Inducible Factor 2 alpha; EPAS1; "
            "bHLH-PAS domain transcription factor family; "
            "domain structure: bHLH (DNA binding) / PAS-A / PAS-B / ODDD / TAD-N / TAD-C; "
            "ODDD (oxygen-dependent degradation domain): contains Pro405 + Pro531 → PHD2 hydroxylation targets; "
            "HIF-2α-SPECIFIC vs HIF-1α: "
            "  HIF-2α principal driver of EPO induction in kidney/liver + angiogenesis; "
            "  HIF-1α: glycolysis (GLUT1, LDHA) + VEGF; "
            "FAMILIAL ERYTHROCYTOSIS TYPE 4 (ECYT4): HIF-2α GOF → enhanced ODDD → resists PHD2 hydroxylation → "
            "  stabilised HIF-2α → constitutive EPO + VEGF gene induction → erythrocytosis; "
            "PARAGANGLIOMA-POLYCYTHAEMIA SYNDROME: "
            "  Somatic EPAS1 GOF in haematopoietic progenitors (mosaic) + neural crest → "
            "  simultaneous erythrocytosis + paraganglioma/phaeochromocytoma; "
            "BELZUTIFAN TARGET: HIF-2α PAS-B domain = drug-binding pocket"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GERMLINE (ECYT4) + SOMATIC MOSAIC (Paraganglioma-Polycythaemia): "
            "GERMLINE GOF — FAMILIAL ERYTHROCYTOSIS TYPE 4: "
            "  AD; heterozygous EPAS1 mutations in ODDD or PAS-B domain; "
            "  Mutations in Pro405 or Pro531 (PHD2 hydroxylation sites): PHD2 cannot hydroxylate → "
            "    pVHL cannot bind → HIF-2α not degraded; "
            "  EPO: elevated (renal HIF-2α constitutively active); "
            "  Clinical: erythrocytosis; headache; thrombosis risk; "
            "  Some: paraganglioma (if mutation activates neural crest HIF-2α as well); "
            "SOMATIC MOSAIC — PARAGANGLIOMA-POLYCYTHAEMIA-SOMATOSTATINOMA SYNDROME: "
            "  EPAS1 GOF arising early in embryogenesis → mosaic in haematopoietic AND neural crest lineages; "
            "  NOT inherited (de novo somatic); rarely familial; "
            "  Features: erythrocytosis + multifocal paraganglioma + somatostatinoma (duodenal NET); "
            "  Difficult to detect on germline testing (blood DNA) — may need tumour DNA; "
            "DISTINCT FROM VHL SYNDROME: "
            "  EPAS1 GOF → EPO elevated (HIF-2α active → kidney makes more EPO); "
            "  VHL LOF (Chuvash) → same result (HIF-2α not degraded); "
            "  EPO elevated in BOTH; JAK2 negative in BOTH; "
            "  DDx by gene sequencing: EPAS1 vs VHL"
        ),
        "disease_category": (
            "FAMILIAL ERYTHROCYTOSIS TYPE 4 (ECYT4) + PARAGANGLIOMA-POLYCYTHAEMIA SYNDROME: "
            "ECYT4: "
            "  Moderate-severe erythrocytosis; Hct typically 55-65%; "
            "  EPO ELEVATED (or inappropriately normal for degree of erythrocytosis); "
            "  Platelets/WBCs: NORMAL (pure erythroid expansion — HIF-2α drives EPO not TPO); "
            "  Thrombosis risk: VTE + arterial (same as EPOR, VHL, PV); "
            "  Pulmonary hypertension (EPAS1 activates pulmonary artery smooth muscle HIF → vasoconstriction); "
            "PARAGANGLIOMA-POLYCYTHAEMIA SYNDROME: "
            "  Paraganglioma: multiple; extra-adrenal; sympathetic (catecholamine-secreting) or parasympathetic (non-secreting); "
            "  Phaeochromocytoma: adrenal; "
            "  Somatostatinoma: duodenal NET; somatostatin immunoreactive; DDx by imaging; "
            "  Erythrocytosis: concurrent; both clonal and paraneoplastic components; "
            "  Screening: whole-body 68Ga-DOTATATE PET (SSTR-based); MRI spine/skull base; "
            "BELZUTIFAN INDICATION: "
            "  VHL-associated RCC (approved FDA 2021); "
            "  EPAS1-GOF erythrocytosis: clinical trials underway; pharmacological HIF-2α inhibition normalises EPO"
        ),
        "disease_pathway": (
            "HIF-2α STABILISATION → EPO OVERPRODUCTION: "
            "NORMAL HIF-2α CYCLE: "
            "  Normoxia: PHD2 + O2 + Fe2+ + 2-oxoglutarate → hydroxylates HIF-2α at Pro405 + Pro531 → "
            "    VHL-E3 ubiquitin ligase → K48-polyubiquitin → proteasome → HIF-2α t½ = minutes; "
            "  Hypoxia: PHD2 inactive → HIF-2α accumulates → HIF-2α + HIF-1β (ARNT) heterodimer → "
            "    binds HRE (RCGTG) in EPO gene enhancer (3' EPO enhancer, kidney) → EPO mRNA; "
            "EPAS1 GOF (ODDD MUTATION): "
            "  Pro405/531 substitution: PHD2 cannot hydroxylate → pVHL binding lost → "
            "    HIF-2α normoxic stabilisation — simulates permanent hypoxia in kidney; "
            "  EPO mRNA increased 5-10x → EPO protein elevated → erythropoiesis stimulated; "
            "PAS-B DOMAIN MUTATIONS (HIF-2α/HIF-1β INTERFACE): "
            "  Disrupt HIF-2α dimerisation with HIF-1β alternatively → LOF (rare anaemia) "
            "  OR gain-of-function (increased transactivation); context-dependent; "
            "BELZUTIFAN MECHANISM: "
            "  Binds HIF-2α PAS-B domain (pocket created by D539N etc.); "
            "  Allosteric disruption: HIF-2α cannot dimerize with HIF-1β → "
            "    HIF-2α target genes (EPO, VEGF, CCND1) not induced; "
            "  EPO level drops to normal → Hct normalises over 4-12 weeks"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — EPAS1: "
            "  1. ERYTHROCYTOSIS + ELEVATED EPO + JAK2 NEGATIVE: "
            "     ALGORITHM: erythrocytosis → EPO → if ELEVATED (not suppressed) → "
            "       JAK2 negative → VHL (Chuvash/heterozygous) → EPAS1 → EGLN1 → HGBB haemoglobin variants; "
            "  2. PARAGANGLIOMA + ERYTHROCYTOSIS (co-occurrence): "
            "     PATHOGNOMONIC of EPAS1 GOF somatic mosaic syndrome; "
            "     25-30% of paraganglioma-polycythaemia cases = EPAS1 mosaic; "
            "  3. DUODENAL SOMATOSTATINOMA (endoscopy incidental NET) + erythrocytosis: "
            "     Highly suggestive of EPAS1 mosaic syndrome; "
            "  4. MULTIFOCAL PARAGANGLIOMA (3+) WITHOUT SDHx: "
            "     Consider EPAS1 mosaic testing on tumour DNA if blood germline testing negative; "
            "  5. PULMONARY HYPERTENSION + ERYTHROCYTOSIS (young patient): "
            "     EPAS1 and VHL activate pulmonary artery HIF → vasoconstriction → PH; "
            "     EPAS1 GOF erythrocytosis + unexplained PH = refer to specialist centre"
        ),
        "treatment": (
            "TREATMENT — EPAS1-GOF ERYTHROCYTOSIS AND PARAGANGLIOMA-POLYCYTHAEMIA: "
            "ERYTHROCYTOSIS: "
            "  PHLEBOTOMY: target Hct <45% (M) / <42% (F); mainstay; "
            "  ASPIRIN 75-100 mg: thrombosis prevention; "
            "  BELZUTIFAN (HIF-2α inhibitor): "
            "    Current indication: VHL-RCC (approved); "
            "    ECYT4 + EPAS1 GOF: clinical trials showing EPO normalisation; not yet standard; "
            "    Monitor: dose-dependent anaemia (EPO suppressed below normal if overdosed); "
            "PARAGANGLIOMA: "
            "  SURGICAL RESECTION: curative for localised paraganglioma; "
            "  Pre-op alpha-blockade: phenoxybenzamine 10-40mg/d for 2 weeks; "
            "  Biochemical monitoring: 24h urine metanephrines/normetanephrines annually; "
            "  68Ga-DOTATATE PET: baseline + every 2y (SSTR imaging); "
            "  MIBG therapy / somatostatin analogues: metastatic/unresectable paraganglioma; "
            "  LUTETIUM-177 DOTATATE: metastatic SSTR-positive NETs including paraganglioma; "
            "SURVEILLANCE: "
            "  Annual FBC + EPO; "
            "  Phaeochromocytoma/paraganglioma screening: annual biochemistry; "
            "  MRI neck/chest/abdomen/pelvis: every 2y if paraganglioma known or EPAS1 GOF mosaic confirmed; "
            "GENETIC COUNSELLING: germline ECYT4: AD 50% transmission; somatic mosaic: very low recurrence"
        ),
        "seed": 2868,
    },
    {
        "gene": "THPO",
        "protein": (
            "THPO -- 3q27.1 AD germline -- 353aa -- "
            "Thrombopoietin-TPO-35kDa-Haematopoietin-Superfamily-"
            "Primary-Megakaryocyte-Growth-Factor-"
            "Germline-Promoter-5-UTR-Mutations-Familial-ET-Type2-"
            "OMIM-Gene-600044-Disease-Familial-ET-HT-614021"
        ),
        "locus": "3q27.1",
        "protein_size": (
            "353 aa / 35 kDa (Thrombopoietin; TPO; THPO; haematopoietin cytokine family; "
            "domain structure: EPO-like domain (N-terminal, 153aa — receptor binding) + "
            "  O-glycosylated carbohydrate domain (C-terminal, 180aa — extends half-life, not signalling); "
            "THPO production: liver (primary, constitutive) + kidney + stromal cells; "
            "THPO binds MPL (TPOR) on megakaryocytes + platelets → JAK2-STAT5 → platelet production; "
            "THPO CLEARANCE: platelets + megakaryocytes sequester TPO → negative feedback; "
            "HIGH platelets → less free THPO → less MPL stimulation; "
            "LOW platelets (thrombocytopenia) → more free THPO → stimulates megakaryopoiesis; "
            "FAMILIAL ET (germline THPO): "
            "  5-UTR mutations (IRES disruption): increased THPO mRNA translation efficiency → excess TPO; "
            "  Exon splice-site mutations: abnormal splicing → loss of upstream ORF (uORF) → "
            "    derepressed translation of main ORF → excess TPO protein; "
            "  AD: one mutant allele sufficient for thrombocytosis"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT GERMLINE — FAMILIAL ESSENTIAL THROMBOCYTHAEMIA TYPE 2 (HT): "
            "5-UTR THPO MUTATIONS (HEREDITARY THROMBOCYTHAEMIA): "
            "  AD; typically frameshifts or mutations disrupting upstream open reading frame (uORF); "
            "  uORF (5-UTR) normally SUPPRESSES main ORF translation; "
            "  uORF disruption → ribosome scans past → main ORF more efficiently translated → excess TPO; "
            "  TPO elevated in plasma; "
            "  Danish/Dutch/Japanese family clusters reported; "
            "CLINICAL PHENOTYPE vs SOMATIC MPN-ET: "
            "  Familial: multiple generations affected; "
            "  Platelet count: 600-1500×10⁹/L typically; "
            "  No splenomegaly or minimal; "
            "  BM: megakaryocyte hyperplasia; no reticulin fibrosis; "
            "  JAK2 V617F: NEGATIVE (TPO-driven, not JAK2-mutant); "
            "  CALR mutation: NEGATIVE; "
            "  MPL somatic W515: NEGATIVE; "
            "  Serum THPO: ELEVATED (unlike somatic ET where THPO can be normal or slightly elevated); "
            "DISTINGUISHING HEREDITARY THROMBOCYTHAEMIA: "
            "  Positive family history (AD) + elevated THPO + negative JAK2/CALR/MPL + "
            "    5-UTR THPO sequencing confirmatory; "
            "  RULE OUT: reactive thrombocytosis (iron deficiency, infection, post-splenectomy, "
            "    inflammatory — all have normal or low THPO)"
        ),
        "disease_category": (
            "HEREDITARY THROMBOCYTHAEMIA — THPO GOF (OMIM 614021): "
            "THPO OVERPRODUCTION → THROMBOCYTOSIS: "
            "  Excess THPO → constitutive MPL stimulation → megakaryocyte overproduction → "
            "    platelet count 600-1500×10⁹/L; "
            "  MEGAKARYOCYTE MORPHOLOGY: large megakaryocytes with hyperlobulated nuclei; "
            "    no dysplasia (distinguishes from MDS); "
            "  THROMBOSIS: VTE (DVT, PE) + arterial (TIA, MI); "
            "    Risk lower than JAK2 V617F ET (no additional MPN inflammatory cytokine milieu); "
            "  HAEMORRHAGE: acquired VWD at very high platelet counts (>1500); "
            "  ERYTHROMELALGIA: burning/redness extremities; ASA-responsive; "
            "TRANSFORMATION RISK: "
            "  LOWER than somatic MPN (THPO overproduction without JAK2 or epigenetic driver); "
            "  Myelofibrosis: very rare (<<5%); "
            "  AML: exceptionally rare; "
            "REACTIVE THROMBOCYTOSIS DDx: "
            "  THPO germline ET: THPO elevated; family history; PERSISTS despite iron correction; "
            "  Iron-deficiency reactive thrombocytosis: THPO normal; FBC normalises with iron treatment"
        ),
        "disease_pathway": (
            "THPO-MPL-JAK2 AXIS — FAMILIAL THROMBOCYTHAEMIA: "
            "NORMAL REGULATION: "
            "  Liver: constitutive THPO mRNA translation; uORF suppresses main ORF; controlled TPO output; "
            "  Plasma THPO: ~50-100 pg/mL; binds MPL on platelets + megakaryocytes → clearance; "
            "  High platelets → more MPL surface → more THPO clearance → less free THPO → "
            "    less megakaryopoiesis → platelet count homeostasis; "
            "THPO 5-UTR MUTATION: "
            "  uORF disruption → ribosome bypasses inhibitory uORF → enhanced main ORF translation → "
            "    THPO mRNA translated more efficiently → excess TPO protein secreted by liver; "
            "  Plasma THPO: 200-1000 pg/mL (4-10x elevated); "
            "  TPO binds MPL on megakaryocyte progenitors → JAK2-STAT5 → "
            "    megakaryocyte proliferation + differentiation → platelet release; "
            "  Despite high platelets → platelet MPL downregulated in hereditary ET → "
            "    TPO clearance impaired → THPO remains elevated (escape from feedback); "
            "DOWNSTREAM CONSEQUENCES: "
            "  STAT5 → GATA-1 + FLI1 → megakaryocyte maturation; "
            "  TPO also supports HSC self-renewal → excess THPO → mild BM cellularity increase; "
            "  NO JAK2 MUTATION required — THPO-excess drives constitutive MPL signalling; "
            "  Lower cytokine milieu than somatic MPN → lower MF/AML transformation risk"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — THPO FAMILIAL ET: "
            "  1. ELEVATED SERUM THPO (>200 pg/mL) + THROMBOCYTOSIS: "
            "     SOMATIC MPN ET: THPO normal or mildly elevated; "
            "     THPO GERMLINE ET: THPO substantially elevated — distinguishing feature; "
            "     REACTIVE THROMBOCYTOSIS: THPO normal; distinguishes reactive from clonal; "
            "  2. FAMILY HISTORY OF THROMBOCYTOSIS (3+ generations, AD): "
            "     Multiple family members platelets >600 × 10⁹/L; JAK2/CALR/MPL all NEGATIVE; "
            "     Prompt THPO 5-UTR sequencing (standard sequencing may miss uORF mutations — "
            "     require specific analysis of 5-UTR exon 3 upstream ORF; clinical lab must be alerted); "
            "  3. HEREDITARY ET + ERYTHROMELALGIA: "
            "     Burning pain + redness extremities relieved by aspirin; microvascular platelet-mediated; "
            "     Common in all thrombocythaemia regardless of genotype; "
            "  4. THPO GERMLINE TESTING PROTOCOL: "
            "     Standard NGS panels often DO NOT cover 5-UTR region — must request specifically; "
            "     Ensure lab covers THPO 5-UTR and exon-intron boundaries"
        ),
        "treatment": (
            "TREATMENT — THPO FAMILIAL ESSENTIAL THROMBOCYTHAEMIA: "
            "RISK STRATIFICATION (same as somatic ET): "
            "  LOW RISK: age <60y + no prior thrombosis: OBSERVATION ± ASPIRIN 75mg; "
            "  HIGH RISK: age ≥60y OR prior thrombosis: CYTOREDUCTION; "
            "  VERY HIGH RISK: platelet count >1500 (VWD risk): CYTOREDUCTION regardless; "
            "CYTOREDUCTION OPTIONS: "
            "  HYDROXYCARBAMIDE (HU): first-line; well-tolerated; weekly FBC monitoring initially; "
            "  ANAGRELIDE: second-line; PDE3/PDE5 inhibitor → reduces megakaryocyte maturation; "
            "  INTERFERON-α (pegylated): preferred in young patients + pregnancy; "
            "ASPIRIN 75-100 mg/day: "
            "  Reduces microvascular symptoms (erythromelalgia) dramatically; "
            "  Arterial thrombosis prevention; "
            "  Hold if acquired VWD (VWF:RCo/Ag <0.7): bleeding risk outweighs benefit; "
            "THPO-TARGETING: "
            "  Eltrombopag/romiplostim: CONTRAINDICATED in THPO-ET (activates MPL already overstimulated); "
            "  THEORETICAL: anti-THPO antibody or THPO-neutralising approach — not clinically available; "
            "PREGNANCY: "
            "  Interferon-α preferred over HU; LMWH + aspirin throughout; fetal monitoring; "
            "GENETIC COUNSELLING: AD; 50% transmission; screen relatives with FBC + THPO level"
        ),
        "seed": 2869,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Patient generator
# ──────────────────────────────────────────────────────────────────────────────

def _make_patients(gene_data):
    rng = random.Random(gene_data["seed"])
    gene = gene_data["gene"]
    patients = []

    for i in range(40):
        pid = f"{gene}-MPN-{2862 + list('JAK2CALRMPLSH2B3EPORVHLPAS1THPO'.split()).index(gene) if False else 0}-{i+1:03d}"
        pid = f"{gene}-MPN-{i+1:03d}"
        sex = rng.choice(["M", "F"])
        age = rng.randint(28, 74)

        if gene == "JAK2":
            # PV/ET/PMF mix; high platelet + high Hb
            subtype = rng.choices(["PV", "ET", "PMF"], weights=[0.45, 0.35, 0.20])[0]
            hb = rng.uniform(18.0, 22.0) if subtype == "PV" else rng.uniform(12.0, 15.5)
            platelets = rng.randint(400, 900) if subtype in ["ET", "PMF"] else rng.randint(250, 550)
            epo = rng.uniform(1.5, 6.0) if subtype == "PV" else rng.uniform(8.0, 25.0)
            jak2_vaf = rng.uniform(15, 95) if subtype == "PV" else rng.uniform(5, 55)
            fibrosis = rng.choices([0, 1, 2, 3], weights=[0.25, 0.35, 0.25, 0.15])[0] if subtype == "PMF" else 0
            spleen = rng.uniform(8, 22) if subtype == "PMF" else rng.uniform(5, 12)
            aquagenic = subtype == "PV" and rng.random() < 0.55
            thrombosis = rng.random() < 0.28
            calr = None; mpl_mut = None
            treatment = rng.choice(["Phlebotomy+ASA", "HU+ASA", "Ruxolitinib+ASA"])
        elif gene == "CALR":
            subtype = rng.choices(["ET", "PMF"], weights=[0.60, 0.40])[0]
            calr_type = rng.choices(["Type1 del52", "Type2 ins5", "Other"], weights=[0.68, 0.22, 0.10])[0]
            hb = rng.uniform(11.5, 14.5)
            platelets = rng.randint(450, 1600)
            epo = rng.uniform(10, 30)
            jak2_vaf = 0
            fibrosis = rng.choices([0, 1, 2, 3], weights=[0.30, 0.25, 0.25, 0.20])[0] if subtype == "PMF" else 0
            spleen = rng.uniform(8, 24) if subtype == "PMF" else rng.uniform(5, 13)
            aquagenic = False
            thrombosis = rng.random() < 0.20
            acquired_vwd = platelets > 1500 and rng.random() < 0.60
            calr = calr_type; mpl_mut = None
            treatment = rng.choice(["Observation+ASA", "HU+ASA", "Ruxolitinib"])
        elif gene == "MPL":
            subtype = rng.choices(["ET", "PMF"], weights=[0.55, 0.45])[0]
            hb = rng.uniform(10.5, 14.0)
            platelets = rng.randint(450, 1200)
            epo = rng.uniform(15, 45)
            jak2_vaf = 0
            fibrosis = rng.choices([0, 1, 2, 3], weights=[0.28, 0.27, 0.25, 0.20])[0] if subtype == "PMF" else 0
            spleen = rng.uniform(9, 22) if subtype == "PMF" else rng.uniform(5, 14)
            aquagenic = False
            thrombosis = rng.random() < 0.22
            calr = None; mpl_mut = rng.choice(["W515L", "W515K", "P106L (germline)"])
            treatment = rng.choice(["HU+ASA", "Anagrelide+ASA", "Ruxolitinib"])
        elif gene == "SH2B3":
            subtype = rng.choices(["ET", "PV", "PMF"], weights=[0.40, 0.35, 0.25])[0]
            hb = rng.uniform(18.0, 21.5) if subtype == "PV" else rng.uniform(11.5, 15.0)
            platelets = rng.randint(350, 1000)
            epo = rng.uniform(1.5, 7.0) if subtype == "PV" else rng.uniform(8, 25)
            jak2_vaf = rng.uniform(10, 75)
            fibrosis = rng.choices([0, 1, 2, 3], weights=[0.30, 0.30, 0.25, 0.15])[0] if subtype == "PMF" else 0
            spleen = rng.uniform(8, 20) if subtype == "PMF" else rng.uniform(5, 13)
            aquagenic = subtype == "PV" and rng.random() < 0.50
            thrombosis = rng.random() < 0.30
            calr = None; mpl_mut = None
            treatment = rng.choice(["Phlebotomy+ASA", "HU+ASA", "Ruxolitinib"])
        elif gene == "EPOR":
            subtype = "Familial Erythrocytosis Type 1"
            hb = rng.uniform(18.5, 23.0) if sex == "M" else rng.uniform(16.5, 21.0)
            platelets = rng.randint(160, 380)
            epo = rng.uniform(1.0, 6.0)
            jak2_vaf = 0
            fibrosis = 0
            spleen = rng.uniform(5, 12)
            aquagenic = False
            thrombosis = rng.random() < 0.26
            calr = None; mpl_mut = None
            treatment = rng.choice(["Phlebotomy+ASA", "Phlebotomy only", "Phlebotomy+ASA+anticoag"])
        elif gene == "VHL":
            subtype = rng.choices(["Chuvash Polycythaemia", "VHL ECYT3", "VHL Syndrome+erythrocytosis"], weights=[0.45, 0.35, 0.20])[0]
            hb = rng.uniform(17.5, 22.0) if sex == "M" else rng.uniform(16.0, 20.5)
            platelets = rng.randint(160, 360)
            epo = rng.uniform(12, 60)
            jak2_vaf = 0
            fibrosis = 0
            spleen = rng.uniform(5, 15)
            aquagenic = False
            thrombosis = rng.random() < 0.32
            phaeo = subtype == "VHL Syndrome+erythrocytosis" and rng.random() < 0.40
            haemangioblastoma = subtype == "VHL Syndrome+erythrocytosis" and rng.random() < 0.65
            calr = None; mpl_mut = None
            treatment = rng.choice(["Phlebotomy+ASA", "Phlebotomy+anticoag", "Belzutifan(trial)"])
        elif gene == "EPAS1":
            subtype = rng.choices(["ECYT4 Germline", "Paraganglioma-Polycythaemia Syndrome"], weights=[0.55, 0.45])[0]
            hb = rng.uniform(18.0, 23.5) if sex == "M" else rng.uniform(16.5, 21.5)
            platelets = rng.randint(165, 370)
            epo = rng.uniform(15, 80)
            jak2_vaf = 0
            fibrosis = 0
            spleen = rng.uniform(5, 13)
            aquagenic = False
            thrombosis = rng.random() < 0.28
            paraganglioma = subtype == "Paraganglioma-Polycythaemia Syndrome"
            somatostatinoma = paraganglioma and rng.random() < 0.30
            calr = None; mpl_mut = None
            treatment = rng.choice(["Phlebotomy+ASA", "Phlebotomy+ASA+PGLsurgery", "Belzutifan(trial)"])
        else:  # THPO
            subtype = "Hereditary Thrombocythaemia"
            hb = rng.uniform(13.0, 16.5) if sex == "M" else rng.uniform(11.5, 15.0)
            platelets = rng.randint(600, 1800)
            epo = rng.uniform(8, 25)
            jak2_vaf = 0
            fibrosis = 0
            spleen = rng.uniform(5, 12)
            aquagenic = False
            thrombosis = rng.random() < 0.20
            calr = None; mpl_mut = None
            treatment = rng.choice(["Observation+ASA", "HU+ASA", "Interferon+ASA"])

        p = {
            "patient_id": pid,
            "sex": sex,
            "age_at_diagnosis_years": age,
            "subtype": subtype,
            "hemoglobin_g_dl": round(hb, 1),
            "platelets_per_nl": int(platelets),
            "serum_epo_miu_ml": round(epo, 1),
            "jak2_v617f_vaf_pct": round(jak2_vaf, 1) if gene in ["JAK2", "SH2B3"] else None,
            "bone_marrow_fibrosis_grade": fibrosis,
            "spleen_size_cm_bcm": round(spleen, 1),
            "thrombosis_history": thrombosis,
            "aquagenic_pruritus": aquagenic if gene in ["JAK2", "SH2B3"] else False,
            "calr_mutation_type": calr,
            "mpl_mutation": mpl_mut,
            "treatment": treatment,
        }
        patients.append(p)

    return patients


# ──────────────────────────────────────────────────────────────────────────────
# Definitions
# ──────────────────────────────────────────────────────────────────────────────

DEFINITIONS = {
    "definitions": [
        {
            "term": "JAK2 V617F Allele Burden",
            "definition": (
                "Percentage of V617F-positive alleles among all JAK2 alleles in granulocyte DNA. "
                "Quantified by allele-specific PCR or NGS. "
                "Low burden (<25%): typically heterozygous; ET or early PV phenotype. "
                "High burden (>50% — homozygous by mitotic recombination): PV or MF phenotype; "
                "higher fibrosis risk; higher transformation risk. "
                "Clinical correlation: allele burden correlates with Hct, spleen size, symptom burden. "
                "Serial monitoring: rising allele burden in ET = possible transformation to PV or MF."
            ),
        },
        {
            "term": "CALR Type 1 vs Type 2 — Clinical Significance",
            "definition": (
                "CALR exon 9 frameshift mutations classified into Type 1 (del52bp) and Type 2 (ins5bp). "
                "Type 1 (del52): longer novel C-terminus; stronger MPL binding; "
                "higher risk of MF transformation; lower platelet count; poorer survival in PMF. "
                "Type 2 (ins5): shorter novel C-terminus; weaker MPL binding; "
                "predominantly ET phenotype; near-normal life expectancy; very low AML risk. "
                "Prognostic NGS report MUST specify type — not just 'CALR positive'. "
                "Implication: CALR Type 1 PMF treated more aggressively (earlier HSCT consideration)."
            ),
        },
        {
            "term": "Triple-Negative MPN",
            "definition": (
                "MPN (ET or PMF) in which JAK2 V617F, CALR exon9 frameshift, and MPL W515 are all negative. "
                "Prevalence: ~10-15% of ET; rare in PV (virtually all PV = JAK2+). "
                "Mechanism: non-driver somatic variants (DNMT3A, TET2, ASXL1 clonal haematopoiesis); "
                "or rare MPL/CALR variants not covered by standard panels; "
                "or germline SH2B3/THPO mutations. "
                "Clinical implication: triple-negative ET has LOWEST thrombosis risk but also "
                "lowest confidence in MPN diagnosis — exclude reactive thrombocytosis rigorously. "
                "Bone marrow biopsy: MANDATORY in triple-negative (cannot rely on WHO molecular criteria)."
            ),
        },
        {
            "term": "Aquagenic Pruritus — PV Hallmark",
            "definition": (
                "Intense generalised pruritus precipitated by water contact (especially hot bath/shower). "
                "Occurs in 40-70% of JAK2-positive PV. "
                "Mechanism: mast cell degranulation (histamine, serotonin, prostaglandins) triggered by "
                "thermal + water stimulation; elevated mast cells in PV skin. "
                "PATHOGNOMONIC in erythrocytosis context: secondary erythrocytosis (EPOR, VHL, EPAS1) does NOT cause aquagenic pruritus. "
                "Treatment: JAK inhibitor ruxolitinib most effective; phlebotomy partially effective; "
                "antihistamines provide partial relief; ASA may help; "
                "avoid hot water: practical non-pharmacological measure."
            ),
        },
        {
            "term": "Erythromelalgia",
            "definition": (
                "Burning pain, warmth, redness of hands/feet; episodic. "
                "Caused by microvascular platelet-mediated occlusion with secondary vasodilatation. "
                "Classic in ET and PV (any cause of thrombocytosis/erythrocytosis). "
                "Dramatically RELIEVED by aspirin 75-100mg — ASA-responsiveness is virtually diagnostic. "
                "Opioids and paracetamol: minimal effect. "
                "NSAIDs: effective. "
                "Not relieved by cooling (unlike peripheral vascular disease)."
            ),
        },
        {
            "term": "IPSS/DIPSS — PMF Risk Stratification",
            "definition": (
                "International Prognostic Scoring System (IPSS) for PMF at diagnosis. "
                "Dynamic IPSS (DIPSS): applicable at any time during disease course. "
                "Risk factors: age >65, constitutional symptoms, Hb <10 g/dL, WBC >25×10⁹/L, "
                "circulating blasts ≥1%. "
                "Risk groups: Low (0), Intermediate-1 (1), Intermediate-2 (2), High (≥3). "
                "Median OS: Low = not reached; Int-1 = 14y; Int-2 = 4y; High = 1.5y. "
                "DIPSS-Plus adds karyotype, platelets <100, transfusion dependence. "
                "MIPSS70/MIPSS70+v2: includes molecular mutations (ASXL1, CALR Type1/2, high-risk SRSF2/IDH1). "
                "Treatment escalation: Int-2/High = ruxolitinib; HSCT for eligible patients."
            ),
        },
        {
            "term": "IPSET-Thrombosis (ET Risk Score)",
            "definition": (
                "International Prognostic Score of Thrombosis in ET. "
                "Parameters: age >60 (1pt), cardiovascular risk factors (1pt), "
                "JAK2 V617F positive (2pt), thrombosis history (2pt). "
                "Low (<2): aspirin only. "
                "Intermediate (2): aspirin ± cytoreduction depending on cardiovascular risk. "
                "High (≥3): cytoreduction (HU first-line) + aspirin. "
                "CALR-type-1 ET: CALR type 1 mutation independently associated with MF transformation (add to prognostication). "
                "Note: acquired VWD (platelets >1500) risk overrides score — treat regardless."
            ),
        },
        {
            "term": "Erythrocytosis Diagnostic Algorithm (Low EPO vs High EPO)",
            "definition": (
                "STEP 1: Confirm true erythrocytosis (red cell mass or repeat Hb/Hct). "
                "STEP 2: Measure SERUM EPO. "
                "LOW EPO (<2 IU/L or below reference): JAK2 V617F/exon12 → if positive = PV; "
                "  if JAK2 NEGATIVE → EPOR sequencing (familial erythrocytosis type 1). "
                "NORMAL/HIGH EPO: secondary erythrocytosis workup: "
                "  O2 saturation (sleep study for OSA; high-altitude residence); "
                "  HbO2 dissociation (P50; high-affinity Hb mutation in HBB); "
                "  VHL sequencing (Chuvash p.R200W — elevated EPO despite erythrocytosis); "
                "  EPAS1 sequencing (HIF-2α GOF — elevated EPO); "
                "  EGLN1 sequencing (PHD2 LOF — elevated EPO); "
                "  CT chest/abdomen/pelvis (EPO-secreting tumour: RCC, hepatocellular carcinoma, phaeochromocytoma)."
            ),
        },
        {
            "term": "Ruxolitinib — JAK1/2 Inhibitor",
            "definition": (
                "Ruxolitinib (Jakafi/Jakavi): first-in-class JAK1/2 inhibitor; competitive ATP-binding. "
                "Indications: PMF (Int-2/High IPSS), HU-refractory/intolerant PV, steroid-refractory acute/chronic GvHD. "
                "Mechanism: reduces JAK1/2-STAT1/STAT3/STAT5 phosphorylation regardless of mutation; "
                "effective in JAK2 V617F, CALR, MPL, and triple-negative MPN (all converge on JAK2). "
                "Efficacy: spleen volume reduction ≥35% in 40-50% of PMF; significant symptom improvement; "
                "does NOT eradicate clone; V617F allele burden falls slowly. "
                "SIDE EFFECTS: anaemia (dose-limiting; nadir 8-12 weeks); thrombocytopenia; infections "
                "(TB reactivation screen before start; VZV prophylaxis — shingles risk 3x); "
                "JAK INHIBITOR WITHDRAWAL SYNDROME: abrupt stop → cytokine rebound → fever + splenomegaly; TAPER SLOWLY. "
                "Drug interactions: strong CYP3A4 inhibitors increase ruxolitinib exposure."
            ),
        },
        {
            "term": "Hydroxycarbamide (Hydroxyurea) — MPN Cytoreduction",
            "definition": (
                "Hydroxycarbamide (HU): ribonucleotide reductase inhibitor; S-phase specific cytotoxic. "
                "First-line cytoreduction for high-risk ET and PV; NOT standard for PMF. "
                "Mechanism: inhibits DNA synthesis → reduces rapidly dividing haematopoietic progenitors. "
                "Monitoring: FBC every 2 weeks until stable, then monthly; "
                "target platelets <400×10⁹/L; WBC 3-8×10⁹/L; dose-adjust to avoid neutropenia/anaemia. "
                "Long-term: well-tolerated; leg ulcers (uncommon, manage with dose reduction + wound care); "
                "LEUKAEMOGENIC RISK: controversial; small excess AML risk in long-term use (years); "
                "preferred over alkylating agents in this regard. "
                "CONTRAINDICATED in pregnancy: teratogenic; switch to interferon."
            ),
        },
        {
            "term": "Belzutifan (HIF-2α Inhibitor)",
            "definition": (
                "Belzutifan (Welireg): first-in-class HIF-2α inhibitor. FDA approved 2021 for VHL disease-associated RCC. "
                "Mechanism: binds HIF-2α PAS-B domain → allosterically prevents HIF-2α/HIF-1β (ARNT) dimerisation → "
                "HIF-2α target genes (EPO, VEGF, CCND1) not transcribed. "
                "ERYTHROCYTOSIS: EPO level drops → Hct normalises over 4-12 weeks. "
                "ANAEMIA: dose-limiting toxicity; EPO suppressed below normal → Hb can fall to <8 g/dL; "
                "dose reduction or temporary hold if anaemia severe. "
                "Investigational use: Chuvash polycythaemia (VHL AR); EPAS1-GOF erythrocytosis; "
                "high-altitude adaptation erythrocytosis (EGLN1/VHL); MPN-PV HIF component. "
                "NOT approved for MPN or EPOR/EPAS1 erythrocytosis outside trials."
            ),
        },
        {
            "term": "Allogeneic HSCT in MPN",
            "definition": (
                "Allogeneic haematopoietic stem cell transplantation: only curative option for MPN (PMF, high-risk MF, blast-phase). "
                "Patient selection: age ≤70y; intermediate-2 or high IPSS/DIPSS; "
                "HSCT candidacy overrides pure MPN subtype considerations. "
                "Conditioning: myeloablative (younger, fit) or reduced-intensity conditioning (RIC) (older, co-morbidities). "
                "Donor: matched sibling donor (MSD) preferred; matched unrelated donor (MUD) acceptable; "
                "haploidentical (50% match) emerging as bridge option. "
                "Outcomes: 5-year OS 40-60% in PMF; GvHD remains major complication. "
                "Pre-HSCT ruxolitinib: reduces splenomegaly → safer engraftment; but withdraw 1-2 days pre-conditioning "
                "(JAK inhibitor withdrawal protocol). "
                "Post-HSCT: taper ruxolitinib slowly; GvHD prophylaxis (tacrolimus/MMF)."
            ),
        },
    ],
    "standards": [
        "WHO 2022 Classification of MPN: Polycythaemia Vera, Essential Thrombocythaemia, Primary Myelofibrosis",
        "ELN/BSH 2021-2022 guidelines for PV, ET, PMF management",
        "NCCN Guidelines MPN v.2024",
        "Vainchenker W et al. Nat Rev Cancer 2016 — JAK2-STAT5 in MPN",
        "Nangalia J et al. NEJM 2013 — CALR mutations in JAK2-negative MPN",
        "Klampfl T et al. NEJM 2013 — CALR mutations (concurrent discovery)",
        "Pikman Y et al. PLoS Med 2006 — MPL W515 in MPN",
        "Lasho TL et al. JCO 2008 — SH2B3/LNK LOF in MPN",
        "Kralovics R et al. NEJM 2005 — JAK2 V617F original discovery",
        "Bento C et al. NEJM 2019 — Hereditary erythrocytosis classification",
        "FDA 2021 — Belzutifan (Welireg) approval for VHL disease-associated RCC",
        "Pack SD et al. JAMA 2024 — EPAS1 mosaic paraganglioma-polycythaemia",
        "Bessieres M et al. HemaSphere 2022 — THPO germline familial ET 5-UTR mutations",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# API endpoint generators
# ──────────────────────────────────────────────────────────────────────────────

def generate_overview():
    all_genes = []
    total_patients = 0
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        total_patients += len(patients)
        all_genes.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "n_patients": len(patients),
            "mean_hb_g_dl": round(sum(p["hemoglobin_g_dl"] for p in patients) / len(patients), 1),
            "mean_platelets_per_nl": int(sum(p["platelets_per_nl"] for p in patients) / len(patients)),
            "mean_epo_miu_ml": round(sum(p["serum_epo_miu_ml"] for p in patients) / len(patients), 1),
            "pct_thrombosis": round(100 * sum(p["thrombosis_history"] for p in patients) / len(patients), 1),
            "pct_aquagenic": round(100 * sum(p.get("aquagenic_pruritus", False) for p in patients) / len(patients), 1),
            "mean_spleen_cm": round(sum(p["spleen_size_cm_bcm"] for p in patients) / len(patients), 1),
            "seed": gene_data["seed"],
        })

    return {
        "atlas": "Hereditary MPN Predisposition Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary MPN Predisposition & Familial Erythrocytosis Reference — "
            "JAK2·CALR·MPL·SH2B3·EPOR·VHL·EPAS1·THPO"
        ),
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "gene_summaries": all_genes,
        "seeds": "2862-2869",
        "disorder_categories": [
            {
                "category": "Clonal MPN Drivers (Somatic JAK-STAT Activators)",
                "genes": ["JAK2", "CALR", "MPL"],
                "note": (
                    "All converge on constitutive JAK2-STAT5 activation. "
                    "JAK2 V617F (95% PV, 55% ET, 65% PMF): panmyelosis; aquagenic pruritus pathognomonic in PV; "
                    "ruxolitinib first-line PMF. "
                    "CALR (del52 type1 or ins5 type2): JAK2-neg ET/PMF; type1 = MF risk; type2 = benign ET; "
                    "targets MPL constitutively. "
                    "MPL W515: JAK2/CALR-neg ET/PMF; thrombocytosis dominant; somatic ± rare germline P106L."
                ),
            },
            {
                "category": "Germline MPN Predisposition (Non-Driver Amplifiers)",
                "genes": ["SH2B3"],
                "note": (
                    "SH2B3/LNK: negative regulator of JAK2; germline LOF amplifies any JAK2 driver signal. "
                    "Not a direct MPN driver — lowers threshold for acquiring JAK2 V617F-positive MPN. "
                    "Familial MPN clusters; autoimmune co-morbidities (T1D, coeliac, RA). "
                    "46/1 haplotype + SH2B3 LOF: additive predisposition."
                ),
            },
            {
                "category": "Familial Erythrocytosis — Suppressed EPO (JAK2-negative PV mimics)",
                "genes": ["EPOR"],
                "note": (
                    "EPOR C-terminal truncation: removes negative regulatory domain (SHP-1/SOCS3 docking) → "
                    "prolonged JAK2-STAT5 signal per EPO stimulus → EPO hypersensitivity. "
                    "EPO SUPPRESSED (as in JAK2 PV) but JAK2 V617F NEGATIVE. "
                    "Pure erythrocytosis: platelets + WBC NORMAL. "
                    "Phlebotomy mainstay; AD familial."
                ),
            },
            {
                "category": "Familial Erythrocytosis — Elevated EPO (HIF Pathway Mutations)",
                "genes": ["VHL", "EPAS1"],
                "note": (
                    "VHL LOF (Chuvash AR p.R200W, heterozygous ECYT3): HIF-2α not degraded → "
                    "constitutive EPO gene induction → EPO ELEVATED despite polycythaemia. "
                    "VHL syndrome (AD): haemangioblastoma + RCC + phaeochromocytoma; belzutifan FDA2021 for VHL-RCC. "
                    "EPAS1 GOF (HIF-2α): same result; ECYT4 ± paraganglioma-polycythaemia mosaic syndrome. "
                    "KEY: EPO ELEVATED + JAK2 NEGATIVE = check VHL/EPAS1."
                ),
            },
            {
                "category": "Familial Essential Thrombocythaemia (TPO Overproduction)",
                "genes": ["THPO"],
                "note": (
                    "THPO 5-UTR mutations: disrupt upstream ORF → enhanced main ORF translation → excess TPO. "
                    "Serum THPO ELEVATED (distinguishes from somatic MPN-ET + reactive thrombocytosis). "
                    "JAK2/CALR/MPL all NEGATIVE. "
                    "Lower transformation risk than somatic MPN. "
                    "Standard NGS often MISSES 5-UTR mutations — request specific THPO 5-UTR sequencing."
                ),
            },
        ],
        "critical_distinctions": [
            "EPO SUPPRESSED + erythrocytosis: JAK2 V617F positive = PV; JAK2 NEGATIVE = EPOR familial erythrocytosis type 1 (NOT VHL/EPAS1 which have elevated EPO)",
            "EPO ELEVATED + erythrocytosis + JAK2 NEGATIVE: VHL (Chuvash p.R200W AR or heterozygous ECYT3) vs EPAS1 GOF vs EGLN1 LOF — HIF pathway erythrocytosis; secondary (OSA, altitude, EPO-secreting tumour) must be excluded first",
            "AQUAGENIC PRURITUS in erythrocytosis = JAK2 V617F PV until proven otherwise — secondary/familial erythrocytosis does NOT cause aquagenic pruritus; immediate hot-bath pruritus is pathognomonic",
            "CALR TYPE 1 (del52) vs TYPE 2 (ins5): NGS report must specify — Type1 = MF/transformation risk; Type2 = benign ET; treatment intensity differs; always quantify type not just 'CALR positive'",
            "TRIPLE-NEGATIVE ET (JAK2/CALR/MPL all negative): test THPO 5-UTR (familial ET) + germline SH2B3; bone marrow biopsy MANDATORY in triple-negative to confirm MPN vs reactive thrombocytosis",
            "ACQUIRED VON WILLEBRAND DISEASE (platelets >1500×10⁹/L): CALR/MPL/THPO ET most affected; VWF:RCo/Ag ratio <0.7 = acquired VWD; BLEEDING risk despite thrombocytosis; HOLD cytoreduction target; use DDAVP + tranexamic acid acutely",
            "RUXOLITINIB WITHDRAWAL SYNDROME: NEVER abruptly stop — taper over 1-2 weeks minimum; abrupt discontinuation → cytokine rebound: fever, bone pain, septic-shock-like syndrome, rapidly enlarging spleen",
            "VHL SYNDROME vs CHUVASH POLYCYTHAEMIA: VHL AD (one allele mutated) = tumour syndrome (RCC, haemangioblastoma, phaeochromocytoma); Chuvash AR (p.R200W homozygous) = pure erythrocytosis without VHL tumours — genotype determines clinical syndrome",
            "PARAGANGLIOMA + ERYTHROCYTOSIS CO-OCCURRENCE: EPAS1 somatic mosaic until proven otherwise — standard germline blood DNA may be falsely negative; test tumour tissue if index of suspicion high",
            "THPO 5-UTR MUTATIONS: standard NGS gene panels FREQUENTLY MISS these — promoter/UTR regions excluded by design; clinical lab MUST be explicitly requested to sequence THPO 5-UTR; missed diagnoses lead to misclassification as triple-negative ET",
        ],
    }


def generate_breakdown():
    result = []
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        result.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein": gene_data["protein"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "n_patients": len(patients),
            "patients": patients[:5],
        })
    return {"genes": result, "total": len(ATLAS_GENES), "seeds": "2862-2869"}


def generate_definitions():
    return {
        "atlas": "Hereditary MPN Predisposition Atlas",
        "definitions": DEFINITIONS["definitions"],
        "standards": DEFINITIONS["standards"],
        "gene_count": len(ATLAS_GENES),
        "seeds": "2862-2869",
    }
