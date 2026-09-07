#!/usr/bin/env python3
"""Hereditary-LSD-Atlas — Complete 8-Gene Lysosomal Storage Disorder Atlas
GBA     (glucocerebrosidase; 497 aa; 1q22; AR;
         Gaucher disease Type 1/2/3; most common LSD; ERT imiglucerase/velaglucerase;
         SRT miglustat/eliglustat; Parkinson risk 5-10x; seed SEED_BASE+0) .
GLA     (alpha-galactosidase A; 429 aa; Xq22.1; XL;
         Fabry disease; kidney/heart/stroke; ERT agalsidase alfa/beta;
         chaperone migalastat amenable mutations only; seed SEED_BASE+1) .
GAA     (acid alpha-glucosidase; 952 aa; 17q25.3; AR;
         Pompe / GSD-II; alglucosidase alfa / avalglucosidase alfa;
         infantile: cardiomegaly PATHOGNOMONIC; late-onset: proximal myopathy; seed SEED_BASE+2) .
HEXA    (hex A alpha-subunit; 529 aa; 15q23; AR;
         Tay-Sachs; Ashkenazi Jewish 1:27; cherry-red spot PATHOGNOMONIC;
         no approved ERT; HSCT ineffective established disease; seed SEED_BASE+3) .
IDUA    (alpha-L-iduronidase; 653 aa; 4p16.3; AR;
         MPS I — Hurler/Scheie; HSCT before age 2.5 MANDATORY for CNS;
         laronidase ERT; seed SEED_BASE+4) .
IDS     (iduronate-2-sulfatase; 550 aa; Xq28; XL;
         MPS II — Hunter; idursulfase ERT; intrathecal for CNS;
         no HSCT recommendation unlike MPS I; seed SEED_BASE+5) .
GALC    (galactocerebrosidase; 669 aa; 14q31.3; AR;
         Krabbe disease; HSCT in pre-symptomatic MANDATORY curative;
         infantile: severe, symptomatic HSCT futile; seed SEED_BASE+6) .
ARSA    (arylsulfatase A; 507 aa; 22q13.33; AR;
         Metachromatic Leukodystrophy MLD; Libmeldy gene therapy EMA2020;
         nerve conduction slowed; ARSA activity low but pseudo-deficiency common;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 1790-1797)
"""

import random

SEED_BASE = 1790

LSD_GENES = [
    # -- GBA -- Gaucher Disease -----------------------------------------------
    {
        "gene": "GBA",
        "protein": (
            "GBA -- 1q22 AR -- Glucocerebrosidase-497aa -- "
            "Gaucher-Disease-Type-1-2-3 -- Most-Common-LSD-1in40000 -- "
            "Glucocerebroside-Accumulates-Macrophages -- "
            "ERT-Imiglucerase-Velaglucerase-Taliglucerase -- "
            "SRT-Miglustat-Eliglustat-Type1-Only -- "
            "Parkinson-Risk-5-10x-HETEROZYGOUS-CARRIERS-Also -- "
            "Splenomegaly-Bone-Pain-Anaemia-Thrombocytopenia"
        ),
        "alias": (
            "GBA (glucocerebrosidase; glucosylceramidase beta); OMIM gene 606463; "
            "Gaucher disease Type 1 (non-neuronopathic) OMIM 230800; "
            "Type 2 (acute neuronopathic) OMIM 230900; "
            "Type 3 (chronic neuronopathic) OMIM 231000. "
            "1q22; 497 aa; ~59 kDa; autosomal recessive. "
            "FUNCTION: GBA encodes lysosomal acid beta-glucocerebrosidase (glucosylceramidase), "
            "which cleaves glucocerebroside (glucosylceramide) into glucose + ceramide within lysosomes. "
            "GBA LOF -> glucocerebroside accumulates in macrophage lysosomes (Kupffer cells, splenic macrophages, "
            "bone marrow macrophages) -> Gaucher cells (macrophages distended with substrate, "
            "crinkled tissue paper cytoplasm on biopsy -- PATHOGNOMONIC appearance). "
            "CLINICAL TYPES: "
            "TYPE 1 (non-neuronopathic, most common -- 95% in Western countries): "
            "Splenomegaly: massive -- spleen can be 50-100x normal size; "
            "Hepatomegaly: 2-3x normal; "
            "Bone disease: Erlenmeyer flask deformity (distal femur flask-shaped on X-ray -- PATHOGNOMONIC); "
            "bone pain crises, osteonecrosis (avascular necrosis), pathological fractures; "
            "Anaemia: normocytic, normochromic from splenic sequestration; "
            "Thrombocytopenia: from hypersplenism -> bleeding risk; "
            "NO primary CNS involvement in Type 1; "
            "Pulmonary involvement: rare, severe when present. "
            "TYPE 2 (acute neuronopathic): "
            "Onset <2 years; rapidly fatal (death <2 years); "
            "Severe neurological: bulbar palsy, oculomotor abnormalities, trismus, opisthotonus; "
            "ERT does NOT cross blood-brain barrier -- neurological symptoms NOT treated by ERT. "
            "TYPE 3 (chronic neuronopathic): "
            "Less severe CNS involvement; longer survival; saccadic eye movements abnormal; "
            "Norrbottnian form (Sweden): p.Leu483Pro (L444P) homozygous. "
            "PARKINSON RISK -- CRITICAL: "
            "GBA heterozygous carriers (including parents/children of Type 1 patients): "
            "5-10x increased Parkinson disease risk; "
            "GBA variants are the MOST COMMON known genetic risk factor for Parkinson disease; "
            "1q22 LOF variants increase alpha-synuclein aggregation; "
            "Homozygous Gaucher patients also at significantly increased PD risk; "
            "Counsel carriers about PD risk -- NOT just Gaucher disease. "
            "DIAGNOSIS: "
            "GBA enzyme activity in dried blood spot (DBS) or leukocytes -- low; "
            "Biomarkers: plasma chitotriosidase elevated (not specific); "
            "glucosylsphingosine (lyso-Gb1) -- highly sensitive and specific biomarker for monitoring; "
            "GBA gene sequencing: N370S (most common in Ashkenazi, Type 1 only), "
            "L444P (Type 3, neuronopathic risk), F213I, 84GG, IVS2+1G>A. "
            "TREATMENT: "
            "ERT (enzyme replacement therapy): imiglucerase (Cerezyme), "
            "velaglucerase alfa (VPRIV), taliglucerase alfa (Elelyso) -- "
            "IV infusion every 2 weeks; targets visceral disease (spleen, liver, bone, haematology); "
            "Does NOT cross blood-brain barrier -- CNS disease not treated; "
            "SRT (substrate reduction therapy): eliglustat (Cerdelga -- Type 1 adults, CYP2D6 metaboliser status required), "
            "miglustat (Zavesca -- less preferred, GI side effects); "
            "Eliglustat: check CYP2D6 status MANDATORY (poor metabolisers: different dosing); "
            "Splenectomy: largely abandoned -- worsens bone disease; avoid unless life-threatening hypersplenism."
        ),
        "locus": "1q22",
        "aa": 497,
        "kDa": 59,
        "omim_gene": "606463",
        "omim_disease": "230800",
        "inheritance": "AR -- biallelic loss-of-function",
        "gene_class": "Lysosomal acid hydrolase — glucosylceramide beta-glucosidase GH30 family",
        "key_alerts": [
            "GBA-PARKINSON-RISK-HETEROZYGOUS-CARRIERS: GBA heterozygous carriers (parents/siblings of Gaucher patients) have 5-10x increased Parkinson disease risk — most common known genetic risk factor for PD; counsel ALL family members about PD risk at time of Gaucher diagnosis; annual neurological review for carriers aged >40",
            "GBA-ERT-NO-CNS-PENETRATION: ERT (imiglucerase, velaglucerase) does NOT cross the blood-brain barrier — neurological symptoms in Type 2 and Type 3 Gaucher are NOT treated by ERT; Type 2 infantile Gaucher is fatal regardless of ERT; HSCT not routinely recommended for Gaucher",
            "GBA-ELIGLUSTAT-CYP2D6-MANDATORY: Eliglustat (SRT for Type 1 adults) dosing depends on CYP2D6 metaboliser status — check CYP2D6 genotype BEFORE prescribing; ultra-rapid metabolisers: eliglustat not recommended; poor metabolisers: specific dose; drug interactions (fluoxetine, paroxetine) alter CYP2D6 activity",
            "GBA-SPLENECTOMY-AVOID: Splenectomy largely abandoned in Gaucher management — worsens bone disease (spleen acts as storage depot; removal accelerates bone infiltration); ERT/SRT is the treatment; only consider splenectomy for life-threatening splenic rupture or refractory hypersplenism unresponsive to ERT",
            "GBA-ERLENMEYER-FLASK-PATHOGNOMONIC: Erlenmeyer flask deformity of distal femur on plain X-ray is pathognomonic for Gaucher disease — loss of normal constriction at distal femur metaphysis; seen in 50-60% of patients; also: bone marrow MRI quantification (Dixon method) for infiltration monitoring",
        ],
        "etiologies": [
            {"variant": "p.Asn409Ser (N370S; c.1226A>G)", "type": "missense LOF — most common Ashkenazi", "frequency": "~70% Ashkenazi Jewish alleles", "severity": "Type 1 only — N370S never causes neuronopathic disease"},
            {"variant": "p.Leu483Pro (L444P; c.1448T>C)", "type": "missense LOF", "frequency": "~15% alleles worldwide", "severity": "Type 3 (L444P homozygous) or Type 2 (with severe allele) — neuronopathic risk"},
            {"variant": "84GG (c.84dupG) frameshift", "type": "LOF frameshift", "frequency": "~5% Ashkenazi", "severity": "severe — Type 2/3 risk when compound het with L444P"},
            {"variant": "IVS2+1G>A (splice site)", "type": "LOF splice", "frequency": "~4% Ashkenazi", "severity": "severe"},
            {"variant": "p.Phe213Ile (F213I)", "type": "missense LOF", "frequency": "common non-Ashkenazi", "severity": "Type 1 — mild to moderate"},
        ],
        "stats": {
            "prevalence_general": "1:40,000-60,000 general population",
            "prevalence_ashkenazi": "1:450-900 Ashkenazi Jewish (carrier rate 1:15-18)",
            "erlenmeyer_flask": "50-60% of Type 1 patients",
            "parkinson_risk_carrier": "5-10x versus general population",
            "splenomegaly": ">95% of untreated Type 1",
        },
        "dx_delay_distribution": {
            "child_1_5y": 20,
            "child_5_18y": 40,
            "adult_18_40y": 30,
            "late_40y_plus": 10,
        },
    },

    # -- GLA -- Fabry Disease -------------------------------------------------
    {
        "gene": "GLA",
        "protein": (
            "GLA -- Xq22.1 XL -- Alpha-Galactosidase-A-429aa -- "
            "Fabry-Disease -- Globotriaosylceramide-Gb3-Accumulates -- "
            "Kidney-Proteinuria-ESRD-Heart-LVH-Stroke-Angiokeratoma -- "
            "ERT-Agalsidase-Alfa-Replagal-Agalsidase-Beta-Fabrazyme -- "
            "Migalastat-Only-Amenable-Mutations-NOT-All-Variants -- "
            "Female-Carriers-Affected-X-Inactivation-Skewing"
        ),
        "alias": (
            "GLA (galactosidase alpha); OMIM gene 300644; "
            "Fabry disease (Anderson-Fabry disease) OMIM 301500. "
            "Xq22.1; 429 aa; ~48.7 kDa; X-linked (hemizygous males affected; "
            "females may be affected due to skewed X-inactivation). "
            "FUNCTION: GLA encodes lysosomal alpha-galactosidase A, "
            "which cleaves the terminal galactose from globotriaosylceramide (Gb3; GL-3). "
            "GLA LOF -> Gb3 accumulates in endothelium, smooth muscle, cardiomyocytes, "
            "renal podocytes and tubular cells, dorsal root ganglion neurons. "
            "CLINICAL PHENOTYPE (classic — males): "
            "NEUROPATHIC PAIN: burning, lancinating pain in hands and feet (acroparesthesiae); "
            "onset in childhood/adolescence; triggered by fever, exercise, heat; "
            "treatment: carbamazepine, gabapentin, phenytoin for pain; "
            "ANGIOKERATOMA: clusters of dark red telangiectatic skin lesions, "
            "bathing-suit distribution (umbilicus to knees), scrotum; PATHOGNOMONIC in classic Fabry; "
            "CORNEA VERTICILLATA: whorl-like corneal deposits on slit-lamp; "
            "PATHOGNOMONIC — seen in virtually all hemizygous males and most carrier females; "
            "no visual impairment; "
            "RENAL: proteinuria (earliest sign), progressive CKD -> ESRD (3rd-4th decade untreated); "
            "CARDIAC: LVH (left ventricular hypertrophy), hypertrophic cardiomyopathy, "
            "arrhythmias (short PR, WPW pattern), cardiac fibrosis; leading cause of death; "
            "CEREBROVASCULAR: premature stroke/TIA (3rd-5th decade), posterior circulation predominant; "
            "white matter lesions on MRI; "
            "LATE-ONSET VARIANTS: cardiac predominant (p.Ala143Thr, IVS4+919G>A -- p.N215S); "
            "no neuropathic pain, no angiokeratoma -- MISSED by classic screening. "
            "FEMALE CARRIERS: "
            "Historically considered asymptomatic carriers -- now recognised affected; "
            "X-inactivation skewing -> variable disease severity in females; "
            "cornea verticillata present in >80% of carrier females; "
            "renal, cardiac, and CNS involvement described; treatment indicated if symptomatic. "
            "DIAGNOSIS: "
            "Alpha-Gal A enzyme activity (DBS or leukocytes): low/absent in males; "
            "UNRELIABLE in females (carrier females have normal or near-normal activity); "
            "GLA gene sequencing: essential, especially in females; "
            "Biomarker: plasma Gb3 and lyso-Gb3 (globotriaosylsphingosine) elevated; "
            "lyso-Gb3 best for females and monitoring. "
            "TREATMENT: "
            "ERT: agalsidase alfa (Replagal 0.2 mg/kg IV q2w -- Europe) or "
            "agalsidase beta (Fabrazyme 1.0 mg/kg IV q2w -- US); "
            "Chaperone: migalastat (Galafold 123 mg oral q48h): "
            "only for AMENABLE GLA variants (confirmed by GLP amenability assay); "
            "migalastat amenability must be verified for EACH variant individually -- "
            "NOT for all GLA variants; list at galafoldamenabilitytable.com; "
            "adjunctive: ACEi/ARB for proteinuria; antiplatelet/anticoagulation for stroke; "
            "carbamazepine for neuropathic pain; pacemaker/ICD for arrhythmia."
        ),
        "locus": "Xq22.1",
        "aa": 429,
        "kDa": 48,
        "omim_gene": "300644",
        "omim_disease": "301500",
        "inheritance": "X-linked — hemizygous males fully affected; females variably affected (X-inactivation)",
        "gene_class": "Lysosomal acid hydrolase — alpha-galactosidase GH27 family",
        "key_alerts": [
            "GLA-CORNEA-VERTICILLATA-PATHOGNOMONIC: Whorl-like corneal deposits (cornea verticillata) on slit-lamp examination are pathognomonic for Fabry disease — present in virtually all hemizygous males and >80% of carrier females; causes no visual impairment; slit-lamp examination mandatory at diagnosis of any suspected Fabry patient",
            "GLA-MIGALASTAT-AMENABILITY-MANDATORY: Migalastat (oral chaperone therapy) is ONLY effective for amenable GLA variants — amenability must be confirmed via GLP cell-based assay for each specific variant; NOT all GLA variants are amenable; prescribing migalastat for a non-amenable variant = ineffective treatment; verify at galafoldamenabilitytable.com",
            "GLA-FEMALE-CARRIERS-UNDERDIAGNOSED: Female GLA carriers are NOT asymptomatic — X-inactivation skewing causes variable but often significant disease; cornea verticillata, cardiac LVH, renal involvement, and stroke all occur in females; enzyme activity is UNRELIABLE in females (often normal); GLA sequencing mandatory for all female relatives of affected males",
            "GLA-LATE-ONSET-CARDIAC-VARIANTS-MISSED: p.Ala143Thr and IVS4+919G>A (p.Asn215Ser) cause late-onset cardiac Fabry (LVH/HCM only; no pain or angiokeratoma); frequently missed as idiopathic HCM; Fabry screen (enzyme + lyso-Gb3 + GLA sequencing) indicated in all unexplained HCM patients >40 years",
            "GLA-LYSO-GB3-MONITORING-SUPERIOR: Plasma lyso-Gb3 (globotriaosylsphingosine) is superior to Gb3 for diagnosis and monitoring — elevated in females and late-onset variants where enzyme activity may be near-normal; lyso-Gb3 should be measured at baseline and every 6 months on ERT/migalastat to assess treatment response",
        ],
        "etiologies": [
            {"variant": "p.Asn215Ser (IVS4+919G>A pseudoexon)", "type": "intronic — late-onset cardiac only", "frequency": "most common late-onset cardiac variant", "severity": "LVH — no neuropathic pain or angiokeratoma"},
            {"variant": "p.Ala143Thr (c.427G>A)", "type": "missense LOF — late-onset cardiac", "frequency": "common in Taiwan population", "severity": "cardiac Fabry — late-onset"},
            {"variant": "p.Arg112His (c.335G>A)", "type": "missense LOF — classic", "frequency": "moderate", "severity": "classic Fabry — all manifestations"},
            {"variant": "p.Cys56Gly (c.166T>G)", "type": "missense LOF — classic", "frequency": "moderate", "severity": "classic Fabry — severe"},
            {"variant": "Large exon deletions / nonsense", "type": "LOF — classic severe", "frequency": "various", "severity": "classic Fabry — severe, all organ systems"},
        ],
        "stats": {
            "prevalence": "1:40,000-117,000 (NBS studies suggest higher: 1:3,100 late-onset included)",
            "esrd_untreated_males": "~50% by age 50",
            "lvh_males": "~50% of adult males",
            "stroke_risk": "12x general population for premature stroke",
            "cornea_verticillata": ">99% hemizygous males; >80% carrier females",
        },
        "dx_delay_distribution": {
            "child_1_10y": 15,
            "teen_10_18y": 25,
            "adult_18_35y": 40,
            "late_35y_plus": 20,
        },
    },

    # -- GAA -- Pompe Disease (GSD-II) -----------------------------------------
    {
        "gene": "GAA",
        "protein": (
            "GAA -- 17q25.3 AR -- Acid-Alpha-Glucosidase-952aa -- "
            "Pompe-Disease-GSD-II-Glycogen-Storage-Type-2 -- "
            "Glycogen-Accumulates-Lysosomes-Muscle-Heart -- "
            "INFANTILE-ONSET-Cardiomegaly-PATHOGNOMONIC-ECG-Short-PR -- "
            "Alglucosidase-Alfa-Myozyme-Avalglucosidase-Alfa-Nexviazyme-ERT -- "
            "Cipaglucosidase-Avalmanase-Pombiliti-Next-Gen-ERT -- "
            "LATE-ONSET-Proximal-Myopathy-No-Cardiomegaly-Respiratory-Failure"
        ),
        "alias": (
            "GAA (glucosidase alpha acid); OMIM gene 606800; "
            "Pompe disease (glycogen storage disease type II; acid maltase deficiency) OMIM 232300. "
            "17q25.3; 952 aa; ~110 kDa; autosomal recessive. "
            "FUNCTION: GAA encodes lysosomal acid alpha-glucosidase (acid maltase), "
            "which degrades glycogen by cleaving both alpha-1,4 and alpha-1,6 glucosidic linkages "
            "within the lysosome. "
            "GAA LOF -> glycogen accumulates in lysosomal vacuoles in all tissues, "
            "particularly muscle (cardiac and skeletal) and motor neurons. "
            "CLINICAL TYPES: "
            "INFANTILE-ONSET POMPE (IOPD -- most severe): "
            "Onset: <12 months (typically 2-6 months); "
            "MASSIVE CARDIOMEGALY: cardiomegaly + cardiomyopathy -- PATHOGNOMONIC in IOPD; "
            "ECG: high-voltage QRS complexes + shortened PR interval -- PATHOGNOMONIC; "
            "PR interval short because glycogen vacuoles disrupt AV conduction; "
            "Hypotonia ('floppy baby'), progressive muscle weakness, respiratory insufficiency; "
            "Macroglossia in ~25%; "
            "Hepatomegaly present (glycogen accumulation); "
            "Death from cardiorespiratory failure by 1-2 years without treatment. "
            "LATE-ONSET POMPE (LOPD): "
            "Onset: childhood to adult (any age); "
            "Proximal limb-girdle muscle weakness: difficulty climbing stairs, rising from chair; "
            "Respiratory: diaphragm involvement -> orthopnoea, nocturnal hypoventilation; "
            "Respiratory failure may precede limb weakness; "
            "NO or minimal cardiomegaly (unlike IOPD) -- KEY DDx from IOPD; "
            "CK elevated (2-10x); "
            "Ptosis, scapular winging in some; "
            "EMG: myopathic changes; muscle biopsy: vacuolar myopathy with glycogen inclusions. "
            "DIAGNOSIS: "
            "GAA enzyme activity in DBS (neonatal screen) or leukocytes/fibroblasts: low/absent; "
            "Urine glucose tetrasaccharide (Glc4): elevated -- useful screening biomarker; "
            "GAA gene sequencing; "
            "Muscle biopsy: vacuolar myopathy, PAS-positive inclusions, acid phosphatase-positive vacuoles; "
            "ECG: short PR + high voltage in IOPD. "
            "TREATMENT: "
            "ERT: alglucosidase alfa (Myozyme/Lumizyme 20 mg/kg IV q2w) -- first approved; "
            "Avalglucosidase alfa (Nexviazyme; Nexviadyme): higher mannose-6-phosphate receptor binding -- "
            "superior uptake, superior clinical outcomes, now preferred over alglucosidase alfa; "
            "Cipaglucosidase alfa + miglustat (Pombiliti+Opfolda): next-gen ERT + SRT combination; "
            "ERT most effective when started early (NBS pre-symptomatic IOPD): "
            "IOPD started pre-symptomatically: survival >5 years with cardiac recovery; "
            "Respiratory support: CPAP/BiPAP for nocturnal hypoventilation in LOPD; "
            "Physical therapy: respiratory physiotherapy, limb strengthening."
        ),
        "locus": "17q25.3",
        "aa": 952,
        "kDa": 110,
        "omim_gene": "606800",
        "omim_disease": "232300",
        "inheritance": "AR -- biallelic loss-of-function",
        "gene_class": "Lysosomal acid hydrolase — acid alpha-glucosidase GH31 family",
        "key_alerts": [
            "GAA-IOPD-CARDIOMEGALY-SHORT-PR-PATHOGNOMONIC: Infantile Pompe presents with massive cardiomegaly + short PR interval on ECG — both findings together are pathognomonic for IOPD; any infant with unexplained cardiomegaly + hypotonia must have GAA enzyme activity measured immediately; DBS GAA assay is the fastest test",
            "GAA-AVALGLUCOSIDASE-SUPERIOR-PREFERRED: Avalglucosidase alfa (Nexviazyme/Nexviadyme) has superior mannose-6-phosphate receptor affinity vs first-generation alglucosidase alfa — superior muscle uptake and superior motor outcomes in trials; avalglucosidase is now preferred first-line ERT for both IOPD and LOPD where available",
            "GAA-NBS-PRE-SYMPTOMATIC-TRANSFORMATIVE: Newborn screening for Pompe (DBS GAA enzyme assay) identifies IOPD pre-symptomatically; starting ERT before cardiac failure develops dramatically improves outcomes (survival and cardiac function); IOPD detected by NBS and treated early: >90% 5-year survival vs <10% untreated",
            "GAA-LOPD-RESPIRATORY-FIRST: In late-onset Pompe, respiratory failure from diaphragm weakness may precede limb weakness and be the presenting symptom; all LOPD patients need FVC supine + FVC upright (supine FVC drop >10% indicates diaphragm involvement); nocturnal BiPAP initiated when FVC <80% or symptomatic nocturnal hypoventilation",
            "GAA-IVS1-SPLICE-LATE-ONSET: c.-32-13T>G (IVS1) is the most common LOPD variant in Caucasians — residual enzyme activity; compound heterozygous with severe allele = LOPD; homozygous IVS1 = milder LOPD; IVS1 homozygous less severe than IVS1 + null allele",
        ],
        "etiologies": [
            {"variant": "c.-32-13T>G (IVS1; intron 1 splice)", "type": "hypomorphic splice — residual activity", "frequency": "most common LOPD allele in Caucasians (~70%)", "severity": "late-onset Pompe — mild to moderate"},
            {"variant": "p.Asp645Glu (c.1935C>A)", "type": "missense LOF — IOPD allele", "frequency": "common in Chinese/Taiwanese IOPD", "severity": "IOPD — severe"},
            {"variant": "p.Gly648Ser (c.1942G>A)", "type": "missense LOF", "frequency": "moderate", "severity": "LOPD — variable"},
            {"variant": "Exon 18 deletion", "type": "LOF deletion", "frequency": "rare — IOPD", "severity": "IOPD — severe"},
            {"variant": "p.Trp746Ter (c.2237G>A)", "type": "nonsense LOF", "frequency": "rare", "severity": "IOPD — severe"},
        ],
        "stats": {
            "prevalence": "1:40,000 overall; IOPD 1:100,000-200,000; LOPD 1:57,000",
            "iopd_cardiomegaly": ">95% of IOPD",
            "iopd_short_pr": "~70% of IOPD",
            "lopd_respiratory_at_diagnosis": "30-40% already have FVC <80%",
            "ivs1_frequency_caucasian_lopd": "~65-70% of LOPD Caucasian alleles",
        },
        "dx_delay_distribution": {
            "neonatal_0_1m": 25,
            "infant_1_12m": 35,
            "child_1_18y": 20,
            "adult_18y_plus": 20,
        },
    },

    # -- HEXA -- Tay-Sachs Disease -------------------------------------------
    {
        "gene": "HEXA",
        "protein": (
            "HEXA -- 15q23 AR -- Hex-A-Alpha-Subunit-529aa -- "
            "Tay-Sachs-Disease-GM2-Gangliosidosis -- "
            "GM2-Ganglioside-Accumulates-Neurons -- "
            "Cherry-Red-Spot-Macula-PATHOGNOMONIC -- "
            "Ashkenazi-Jewish-1in27-Carrier -- "
            "NO-Approved-ERT -- HSCT-Ineffective-Established-Disease -- "
            "NBS-Carrier-Screening-Reduces-Incidence-Dramatically"
        ),
        "alias": (
            "HEXA (hexosaminidase subunit alpha); OMIM gene 606869; "
            "Tay-Sachs disease (GM2 gangliosidosis type 1) OMIM 272800. "
            "15q23; 529 aa; ~60 kDa; autosomal recessive. "
            "FUNCTION: Hex A (hexosaminidase A) is a heterodimer of alpha (HEXA) + beta (HEXB) subunits. "
            "Hex A cleaves GM2 ganglioside (adding the HEXA-encoded GM2 activator binding step) "
            "in lysosomes of neurons. "
            "HEXA LOF -> Hex A activity absent -> GM2 ganglioside accumulates in neuronal lysosomes -> "
            "progressive neuronal death, particularly cerebral cortex, cerebellum, brainstem. "
            "HEXB LOF -> Sandhoff disease (GM2 gangliosidosis type 2): "
            "both Hex A and Hex B absent; similar but more severe CNS + visceral involvement. "
            "CLINICAL PHENOTYPE (infantile/classic, most severe): "
            "Normal development to age 3-6 months; "
            "STARTLE RESPONSE: exaggerated acoustic startle (hyperekplexia) -- early sign; "
            "Progressive loss of motor milestones: cannot sit/roll -> hypotonia -> spasticity; "
            "CHERRY-RED SPOT: fundoscopy reveals cherry-red macula surrounded by grey-white ring "
            "(lipid-laden ganglion cells unable to transmit light); "
            "PATHOGNOMONIC for Tay-Sachs (and other gangliosidoses/Niemann-Pick type A); "
            "Seizures: myoclonic, generalised; "
            "Blindness: progressive; "
            "Macrocephaly (cerebral storage); "
            "Death: typically by age 4-5 years. "
            "JUVENILE FORM: onset 2-10 years; slower progression; "
            "ADULT/LATE-ONSET: psychiatric symptoms, spinocerebellar ataxia, motor neuron disease mimicry; "
            "cherry-red spot absent in adult form; "
            "often misdiagnosed as psychiatric disorder for years. "
            "NO APPROVED TREATMENT -- CRITICAL: "
            "No approved ERT for Tay-Sachs: Hex A does not reach neurons efficiently; "
            "HSCT is ineffective for established disease -- does not halt neurodegeneration; "
            "HSCT only considered for pre-symptomatic cases (controversial); "
            "Substrate reduction therapy (miglustat): some stabilisation in late-onset, not approved; "
            "Management is supportive: anticonvulsants, gastrostomy, palliative care. "
            "PREVENTION -- CARRIER SCREENING: "
            "Ashkenazi Jewish carrier frequency: 1 in 27; "
            "Carrier screening programs (Dor Yeshorim) have reduced Tay-Sachs incidence by >90% "
            "in Ashkenazi Jewish communities; "
            "Prenatal diagnosis: CVS or amniocentesis for at-risk couples. "
            "PSEUDO-DEFICIENCY: "
            "HEXA p.Arg247Trp and p.Arg249Trp: artificially low enzyme activity with synthetic substrate "
            "in vitro, but normal in vivo activity -- NOT Tay-Sachs; "
            "Differentiate by natural substrate (GM2) activity assay or gene sequencing."
        ),
        "locus": "15q23",
        "aa": 529,
        "kDa": 60,
        "omim_gene": "606869",
        "omim_disease": "272800",
        "inheritance": "AR -- biallelic loss-of-function",
        "gene_class": "Lysosomal acid hydrolase — hexosaminidase alpha subunit GH20 family",
        "key_alerts": [
            "HEXA-CHERRY-RED-SPOT-PATHOGNOMONIC: Cherry-red macula on fundoscopy is pathognomonic for Tay-Sachs (and other GM2 gangliosidoses, Niemann-Pick type A) — white/grey surrounding ring (lipid-laden ganglion cells) with intact foveal blood supply appearing red; every infant with developmental regression must have fundoscopy performed urgently",
            "HEXA-NO-ERT-NO-CURATIVE-HSCT: No approved enzyme replacement therapy for Tay-Sachs — Hex A does not efficiently reach brain neurons even if given IV; HSCT is ineffective for established infantile Tay-Sachs and should NOT be offered; management is entirely palliative; prenatal diagnosis and carrier screening are the only effective interventions",
            "HEXA-ADULT-ONSET-PSYCHIATRIC-MIMICRY: Late-onset Tay-Sachs (LOTS) presents as progressive psychiatric disorder + cerebellar ataxia + motor neuron disease in adults — frequently misdiagnosed as schizophrenia or bipolar disorder for years; no cherry-red spot in adults; HEXA enzyme activity + sequencing mandatory in any young adult with unexplained ataxia + psychosis",
            "HEXA-PSEUDO-DEFICIENCY-TRAP: p.Arg247Trp and p.Arg249Trp cause pseudo-deficiency — low activity with synthetic substrate in vitro but NORMAL in vivo GM2 cleavage; these are NOT pathogenic; mis-identification as Tay-Sachs variants causes unnecessary anxiety; differentiate by natural substrate assay or specifically test for pseudo-deficiency alleles in carrier screening",
            "HEXA-EXAGGERATED-STARTLE-EARLY-SIGN: Exaggerated acoustic startle response (hyperekplexia) is an early and distinctive sign of infantile Tay-Sachs — present from 3-6 months, before cherry-red spot may be obvious; any infant with hyperekplexia + developmental plateau → urgent Hex A enzyme assay",
        ],
        "etiologies": [
            {"variant": "c.1278insTATC (1278+TATC frameshift)", "type": "LOF frameshift — Ashkenazi Jewish founder", "frequency": "~80% of Ashkenazi Jewish alleles", "severity": "IOPD — infantile, fatal"},
            {"variant": "IVS12+1G>C (splice donor intron 12)", "type": "LOF splice — Ashkenazi Jewish founder", "frequency": "~15% of Ashkenazi Jewish alleles", "severity": "IOPD — infantile, fatal"},
            {"variant": "p.Gly269Ser (c.805G>A)", "type": "missense — French Canadian founder", "frequency": "major French Canadian allele", "severity": "IOPD — infantile"},
            {"variant": "p.Arg247Trp (c.739C>T)", "type": "PSEUDO-DEFICIENCY — NOT pathogenic", "frequency": "common — non-Ashkenazi", "severity": "None — pseudo-deficiency; differentiate by natural substrate assay"},
            {"variant": "p.Ile207Val (late-onset)", "type": "hypomorphic missense", "frequency": "late-onset adult form", "severity": "Adult-onset LOTS — psychiatric + ataxia"},
        ],
        "stats": {
            "carrier_rate_ashkenazi": "1 in 27",
            "prevalence_infantile_ashkenazi": "1:3,500 births (pre-screening)",
            "incidence_reduction_post_screening": ">90% reduction in Ashkenazi Jewish communities",
            "age_death_infantile": "4-5 years without treatment",
            "adult_onset_misdiagnosis_delay": "5-15 years (psychiatric misdiagnosis)",
        },
        "dx_delay_distribution": {
            "infant_0_6m": 35,
            "infant_6_18m": 35,
            "child_1_10y": 15,
            "adult_18y_plus": 15,
        },
    },

    # -- IDUA -- MPS I (Hurler/Scheie) ----------------------------------------
    {
        "gene": "IDUA",
        "protein": (
            "IDUA -- 4p16.3 AR -- Alpha-L-Iduronidase-653aa -- "
            "MPS-I-Hurler-Scheie-Spectrum -- "
            "Heparan-Dermatan-Sulphate-Accumulates -- "
            "HSCT-Before-Age-2.5-MANDATORY-Hurler-CNS -- "
            "Laronidase-ERT-Aldurazyme-Somatic-Not-CNS -- "
            "Corneal-Clouding-Coarse-Facies-Gibbus-Macroglossia -- "
            "Cardiac-Valve-Disease-Carpal-Tunnel"
        ),
        "alias": (
            "IDUA (iduronidase alpha-L); OMIM gene 252800; "
            "Mucopolysaccharidosis type I (Hurler syndrome / Scheie syndrome / Hurler-Scheie) OMIM 607014. "
            "4p16.3; 653 aa; ~82.7 kDa; autosomal recessive. "
            "FUNCTION: IDUA encodes lysosomal alpha-L-iduronidase, "
            "which cleaves terminal alpha-L-iduronic acid residues from heparan sulphate and dermatan sulphate "
            "(glycosaminoglycans, GAGs). "
            "IDUA LOF -> GAG accumulation in lysosomes of all cell types (connective tissue, brain, heart, bone). "
            "MPS I SPECTRUM: "
            "HURLER (severe end; MPS IH): "
            "Normal at birth; features emerge 6-12 months; "
            "Coarse facies: wide nose, full lips, frontal bossing, macrocephaly; "
            "Macroglossia; short neck; hirsutism; "
            "Corneal clouding: progressive, 2-4 years onset, slit-lamp essential; "
            "GIBBUS (thoracolumbar kyphosis): pathognomonic in combination with other features; "
            "Dysostosis multiplex: short broad ribs, J-shaped sella, bullet-shaped vertebrae; "
            "Hepatosplenomegaly; "
            "Cardiac: valve disease (mitral/aortic regurgitation/stenosis), cardiomyopathy; "
            "Airway: large tonsils/adenoids, obstructive sleep apnoea, difficult airway; "
            "COGNITIVE DECLINE: severe intellectual disability by age 2-3 years; "
            "Death: typically 5-10 years without treatment (cardiac/respiratory). "
            "SCHEIE (mild end; MPS IS): "
            "Corneal clouding (isolated or with joint stiffness); "
            "Normal intelligence; "
            "Carpal tunnel syndrome, cardiac valve disease; "
            "Long survival. "
            "HURLER-SCHEIE INTERMEDIATE: intermediate severity; "
            "Some cognitive involvement. "
            "TREATMENT -- CRITICAL TIMING PRINCIPLE: "
            "HSCT (haematopoietic stem cell transplantation): "
            "CURATIVE for CNS disease IF performed before 2.5 years of age AND "
            "before cognitive decline: "
            "transplant before age 2.5 years + MLD score (development quotient) >70: "
            "preserves cognitive function; "
            "HSCT AFTER age 2.5 or with established cognitive decline: does NOT restore lost cognition; "
            "ERT (laronidase, Aldurazyme 0.58 mg/kg IV q1w): "
            "Excellent for somatic disease (liver, spleen, joint mobility, respiratory); "
            "Does NOT cross blood-brain barrier -- no CNS benefit; "
            "ERT used as BRIDGE to HSCT, and long-term post-HSCT for somatic disease. "
            "DIAGNOSIS: "
            "Urine GAGs: elevated (dermatan sulphate + heparan sulphate); "
            "IDUA enzyme activity in leukocytes/DBS; "
            "IDUA gene sequencing (W402X and Q70X are common Hurler alleles)."
        ),
        "locus": "4p16.3",
        "aa": 653,
        "kDa": 83,
        "omim_gene": "252800",
        "omim_disease": "607014",
        "inheritance": "AR -- biallelic loss-of-function",
        "gene_class": "Lysosomal acid hydrolase — alpha-L-iduronidase GH39 family",
        "key_alerts": [
            "IDUA-HSCT-BEFORE-AGE-2.5-MANDATORY: HSCT for Hurler (MPS IH) MUST be performed before age 2.5 years to preserve cognitive function — HSCT after this window does not reverse existing cognitive decline; development quotient (DQ) >70 at transplant correlates with cognitive preservation; NBS enables pre-symptomatic HSCT before cognitive loss",
            "IDUA-ERT-NO-CNS-PENETRATION: Laronidase (ERT) does NOT cross the blood-brain barrier — somatic disease responds well (liver, spleen, joint mobility, pulmonary function) but CNS disease is unaffected by ERT alone; ERT is used as bridge to HSCT and long-term post-HSCT for somatic control",
            "IDUA-DIFFICULT-AIRWAY-ALWAYS: MPS I patients always have a potentially difficult airway — macroglossia, large tonsils, short neck, restricted mouth opening, atlantoaxial instability; airway management plan must be established BEFORE any general anaesthesia; warn anaesthetic team explicitly; have senior anaesthetist and videolaryngoscope available",
            "IDUA-CORNEAL-CLOUDING-SLIT-LAMP: Corneal clouding in MPS I requires slit-lamp examination for detection — not always visible to naked eye; leads to progressive visual loss; does NOT respond to ERT or HSCT; corneal transplantation considered in severe cases; regular ophthalmological review mandatory",
            "IDUA-CARDIAC-VALVE-SURGERY: MPS I cardiac valve disease (mitral/aortic regurgitation/stenosis) can be severe and require surgical repair — high perioperative risk in MPS patients; antibiotic prophylaxis for dental procedures; echo every 1-2 years; valve replacement sometimes needed in Scheie/Hurler-Scheie adults",
        ],
        "etiologies": [
            {"variant": "p.Trp402Ter (W402X; c.1205G>A)", "type": "nonsense LOF — Hurler", "frequency": "~50% Hurler alleles in Northern Europe", "severity": "severe — Hurler; no residual activity"},
            {"variant": "p.Gln70Ter (Q70X; c.208C>T)", "type": "nonsense LOF — Hurler", "frequency": "~15% Hurler alleles", "severity": "severe — Hurler"},
            {"variant": "p.Arg89Gln (c.266G>A)", "type": "missense — Hurler-Scheie/Scheie", "frequency": "common mild allele", "severity": "Scheie or Hurler-Scheie — milder"},
            {"variant": "p.Ala300Thr (c.898G>A)", "type": "missense — milder", "frequency": "moderate frequency", "severity": "Scheie spectrum"},
            {"variant": "IVS5-2A>G (splice acceptor)", "type": "LOF splice", "frequency": "rare", "severity": "Hurler — severe"},
        ],
        "stats": {
            "prevalence_mps1": "1:100,000",
            "proportion_hurler": "~60% of MPS I",
            "hsct_cognitive_preservation": ">85% if performed before age 2.5 with DQ>70",
            "corneal_clouding_mps1": ">95% of MPS I at some point",
            "cardiac_valve_disease": "~70% of MPS IH",
        },
        "dx_delay_distribution": {
            "neonatal_0_6m": 15,
            "infant_6_18m": 50,
            "child_1_4y": 30,
            "late_4y_plus": 5,
        },
    },

    # -- IDS -- MPS II (Hunter Syndrome) -------------------------------------
    {
        "gene": "IDS",
        "protein": (
            "IDS -- Xq28 XL -- Iduronate-2-Sulfatase-550aa -- "
            "MPS-II-Hunter-Syndrome -- "
            "Heparan-Dermatan-Sulphate-Accumulates-Males-Affected -- "
            "Idursulfase-ERT-Elaprase-Somatic-Not-CNS -- "
            "Intrathecal-Idursulfase-Pabinafusp-CNS-Penetrant -- "
            "NO-Routine-HSCT-Unlike-MPS-I -- "
            "Pebbly-Ivory-Skin-Lesion-PATHOGNOMONIC -- "
            "Severe-vs-Attenuated-Phenotype"
        ),
        "alias": (
            "IDS (iduronate 2-sulfatase); OMIM gene 300823; "
            "Mucopolysaccharidosis type II (Hunter syndrome) OMIM 309900. "
            "Xq28; 550 aa; ~76.4 kDa; X-linked recessive. "
            "FUNCTION: IDS encodes iduronate-2-sulfatase (I2S), "
            "a lysosomal enzyme that removes the 2-O-sulphate group from L-iduronic acid residues "
            "in heparan sulphate and dermatan sulphate. "
            "IDS LOF -> GAG accumulation (heparan sulphate + dermatan sulphate) in all tissues. "
            "CLINICAL PHENOTYPE: "
            "X-linked: hemizygous males affected; female carriers usually unaffected "
            "(rare symptomatic females from extremely skewed X-inactivation). "
            "SEVERE FORM (MPS IIA -- ~60%): "
            "Coarse facies (similar to Hurler but no corneal clouding in MPS II -- KEY DDx from MPS I); "
            "Progressive intellectual disability; aggressive behaviour; "
            "NO CORNEAL CLOUDING: fundamental DDx from MPS I (Hurler); "
            "Hearing loss (conductive + sensorineural); "
            "Hepatosplenomegaly; "
            "Joint stiffness; dysostosis multiplex; "
            "PEBBLY IVORY-COLOURED SKIN LESIONS (dermatan sulphate deposits in dermis): "
            "Pathognomonic -- whitish/ivory irregular papules on upper back/scapular area; "
            "present in ~20-30% of MPS II patients; "
            "Cardiac valve disease; airway obstruction; "
            "Death: typically 10-15 years without treatment. "
            "ATTENUATED FORM (MPS IIB -- ~40%): "
            "Normal or near-normal intelligence; "
            "Carpal tunnel syndrome; joint stiffness; short stature; "
            "Long survival (6th-7th decade possible). "
            "NO ROUTINE HSCT FOR MPS II: "
            "Unlike MPS I, HSCT has NOT been shown to improve neurological outcomes in MPS II -- "
            "not routinely recommended; "
            "ERT + symptomatic management is standard. "
            "TREATMENT: "
            "ERT: idursulfase (Elaprase 0.5 mg/kg IV q1w): "
            "Improves somatic disease (6MWT, liver/spleen); "
            "Does NOT cross blood-brain barrier -- no CNS benefit; "
            "CNS-PENETRANT ERT: "
            "Intrathecal idursulfase (IT-idursulfase): direct CNS delivery via IT infusion; "
            "Pabinafusp alfa (JR-141 -- Japan approved): I2S fused to anti-transferrin receptor antibody "
            "-- crosses BBB via transferrin receptor-mediated transcytosis; "
            "First CNS-penetrant ERT for MPS II -- significant milestone."
        ),
        "locus": "Xq28",
        "aa": 550,
        "kDa": 76,
        "omim_gene": "300823",
        "omim_disease": "309900",
        "inheritance": "X-linked recessive — hemizygous males affected",
        "gene_class": "Lysosomal sulfatase — iduronate-2-sulfatase sulfatase family",
        "key_alerts": [
            "IDS-NO-CORNEAL-CLOUDING-KEY-DDX-MPS1: MPS II (Hunter) has NO corneal clouding — this is the critical clinical distinguishing feature from MPS I (Hurler) which has prominent corneal clouding; any male with MPS-like features (coarse facies, hepatosplenomegaly, dysostosis multiplex) WITHOUT corneal clouding: test IDS enzyme activity first",
            "IDS-NO-ROUTINE-HSCT: Unlike MPS I (Hurler), HSCT is NOT routinely recommended for MPS II — HSCT has not been shown to improve neurological outcomes in MPS II; standard treatment is ERT; intrathecal or CNS-penetrant ERT approaches are under investigation for CNS disease",
            "IDS-PEBBLY-IVORY-SKIN-PATHOGNOMONIC: Pebbly ivory-coloured skin lesions on the upper back/scapular area are pathognomonic for Hunter syndrome — present in 20-30%; whitish irregular papules from dermatan sulphate deposits in dermis; if seen in a male with coarse features → IDS enzyme assay immediately",
            "IDS-FEMALE-CARRIERS-RARELY-SYMPTOMATIC: Female IDS carriers are almost never symptomatic (X-linked) — but rarely, extreme skewed X-inactivation causes full disease expression in females; any female with MPS II phenotype needs IDS enzyme + genetic testing to confirm; somatic mosaicism also possible",
            "IDS-BBB-PENETRANT-ERT-EMERGING: Pabinafusp alfa (JR-141) — IDS fused to anti-TfR1 antibody — crosses the BBB via transferrin receptor-mediated transcytosis; approved in Japan; first ERT with demonstrated CNS penetration for MPS II; intrathecal idursulfase (IT injection) is an alternative CNS delivery method",
        ],
        "etiologies": [
            {"variant": "Large deletions / inversions (IDS/IDSP1 recombination)", "type": "LOF large rearrangement", "frequency": "~20% of severe MPS II", "severity": "severe MPS IIA"},
            {"variant": "Frameshift insertions/deletions", "type": "LOF frameshift", "frequency": "~30% of IDS variants", "severity": "severe MPS IIA"},
            {"variant": "Nonsense mutations (various)", "type": "LOF nonsense", "frequency": "~25%", "severity": "typically severe"},
            {"variant": "Missense mutations (partial LOF)", "type": "partial LOF missense", "frequency": "~25%", "severity": "attenuated MPS IIB when residual activity retained"},
            {"variant": "p.Arg468Gln (c.1403G>A)", "type": "missense — attenuated", "frequency": "moderate", "severity": "attenuated MPS IIB"},
        ],
        "stats": {
            "prevalence": "1:100,000-170,000 males",
            "severe_form": "~60% of MPS II",
            "attenuated_form": "~40% of MPS II",
            "pebbly_skin_prevalence": "20-30%",
            "cardiac_valve_disease": "~60% of MPS II",
        },
        "dx_delay_distribution": {
            "infant_0_2y": 25,
            "child_2_6y": 55,
            "child_6_10y": 15,
            "late_10y_plus": 5,
        },
    },

    # -- GALC -- Krabbe Disease -----------------------------------------------
    {
        "gene": "GALC",
        "protein": (
            "GALC -- 14q31.3 AR -- Galactocerebrosidase-669aa -- "
            "Krabbe-Disease-Globoid-Cell-Leukodystrophy -- "
            "Psychosine-Galactosylsphingosine-Neurotoxic -- "
            "HSCT-Pre-Symptomatic-Newborn-Screen-MANDATORY-Curative -- "
            "Symptomatic-HSCT-Futile-Does-NOT-Halt-Decline -- "
            "NBS-Detects-Before-Onset-CRITICAL-Window -- "
            "Globoid-Cells-Multinucleated-Macrophages-PATHOGNOMONIC-Biopsy"
        ),
        "alias": (
            "GALC (galactosylceramidase); OMIM gene 606890; "
            "Krabbe disease (globoid cell leukodystrophy, GLD) OMIM 245200. "
            "14q31.3; 669 aa; ~80.6 kDa; autosomal recessive. "
            "FUNCTION: GALC encodes galactocerebrosidase (galactosylceramidase), "
            "a lysosomal enzyme that cleaves galactose from galactosylceramide and "
            "galactosylsphingosine (psychosine). "
            "GALC LOF -> "
            "(1) Psychosine (galactosylsphingosine) accumulates: "
            "DIRECT NEUROTOXIN -- psychosine kills oligodendrocytes (the cells that make myelin); "
            "oligodendrocyte death -> progressive demyelination of CNS and PNS; "
            "(2) Galactosylceramide accumulates in macrophages: "
            "multinucleated globoid cells (macrophages engorged with substrate) -- "
            "PATHOGNOMONIC on brain biopsy. "
            "CLINICAL PHENOTYPE: "
            "INFANTILE (most common -- ~85-90%): "
            "Onset: 3-6 months; normal at birth; "
            "STAGE 1: irritability, feeding difficulties, excessive crying; "
            "STAGE 2: opisthotonos, hypertonicity, seizures, developmental regression; "
            "STAGE 3: decerebrate posturing, blindness, deafness, vegetative state; "
            "Death: typically by 2 years without intervention; "
            "MRI: T2 hyperintensity in posterior periventricular white matter + basal ganglia; "
            "NCV: markedly slowed (demyelinating peripheral neuropathy). "
            "LATE-ONSET KRABBE: "
            "Childhood, juvenile, or adult onset; slowly progressive; "
            "Spastic paraparesis, ataxia, visual loss, neuropathy; "
            "Less fulminant; longer survival. "
            "TREATMENT -- THE PRE-SYMPTOMATIC WINDOW IS EVERYTHING: "
            "HSCT (haematopoietic stem cell transplantation): "
            "PRE-SYMPTOMATIC infantile Krabbe (NBS detected): HSCT is effective and transforms outcome; "
            "Donor microglia replace patient microglia, providing functional GALC enzyme; "
            "Pre-symptomatic HSCT: most children achieve ambulatory status and normal/near-normal development; "
            "SYMPTOMATIC infantile Krabbe: HSCT does NOT halt progression -- transplant futile; "
            "This distinction is critical: the window is narrow (weeks-months); "
            "Late-onset Krabbe: HSCT stabilises neurological function if performed before severe deficit. "
            "NBS: "
            "GALC enzyme activity by DBS -- low/absent; "
            "Newborn screening for Krabbe implemented in New York, other states, several countries; "
            "NBS detects before symptom onset -> HSCT while pre-symptomatic -> curative."
        ),
        "locus": "14q31.3",
        "aa": 669,
        "kDa": 81,
        "omim_gene": "606890",
        "omim_disease": "245200",
        "inheritance": "AR -- biallelic loss-of-function",
        "gene_class": "Lysosomal acid hydrolase — galactosylceramidase GH59 family",
        "key_alerts": [
            "GALC-PRE-SYMPTOMATIC-HSCT-CURATIVE-SYMPTOMATIC-FUTILE: HSCT for infantile Krabbe is curative ONLY if performed before symptoms develop — newborn screening identifies patients in the pre-symptomatic window; once symptomatic (hypertonicity, seizures), HSCT does NOT halt progression; this is the single most critical clinical decision in Krabbe management",
            "GALC-NBS-MANDATORY-TIME-CRITICAL: Newborn screening for Krabbe disease (GALC enzyme DBS) is time-critical — infantile Krabbe has a narrow pre-symptomatic window of weeks to months; refer any DBS-positive newborn urgently to metabolic centre; do NOT wait for symptoms before referral; confirmatory testing + HSCT evaluation must happen in days-to-weeks",
            "GALC-PSYCHOSINE-DIRECT-NEUROTOXIN: Psychosine (galactosylsphingosine) is the primary neurotoxin in Krabbe disease — directly kills oligodendrocytes (not just storage accumulation); psychosine levels in CSF and DBS can be measured as biomarkers; high psychosine = poor prognosis; declining psychosine post-HSCT = treatment response",
            "GALC-MRI-POSTERIOR-WHITE-MATTER: Krabbe MRI shows T2 hyperintensity in posterior periventricular white matter and basal ganglia (posterior predominant) — distinct from other leukodystrophies; MRI involvement at NBS referral indicates early symptomatic stage; normal MRI at referral = pre-symptomatic = HSCT urgently",
            "GALC-LATE-ONSET-SPASTIC-PARAPARESIS: Late-onset Krabbe presents as spastic paraparesis + peripheral neuropathy + slow NCV — frequently misdiagnosed as hereditary spastic paraplegia or CIDP; GALC enzyme assay should be performed in any unexplained demyelinating peripheral neuropathy + CNS white matter disease combination",
        ],
        "etiologies": [
            {"variant": "c.857del18/IVS (30-kb deletion including c.502 region)", "type": "LOF large deletion — most common infantile", "frequency": "~45% of European infantile Krabbe alleles", "severity": "infantile — severe"},
            {"variant": "p.Gly270Asp (c.809G>A)", "type": "missense LOF", "frequency": "common", "severity": "infantile — severe"},
            {"variant": "p.Ile583Ser (c.1748T>G)", "type": "missense LOF", "frequency": "late-onset allele", "severity": "late-onset Krabbe"},
            {"variant": "p.Asp544Asn (c.1630G>A)", "type": "missense — late-onset", "frequency": "late-onset", "severity": "juvenile/adult Krabbe"},
            {"variant": "p.Leu634Ser (c.1901T>C) pseudo-deficiency", "type": "pseudo-deficiency", "frequency": "common in general population", "severity": "None — pseudo-deficiency; NOT Krabbe disease"},
        ],
        "stats": {
            "prevalence": "1:100,000-250,000",
            "infantile_proportion": "~85-90% of Krabbe cases",
            "pre_symptomatic_hsct_ambulation": ">80% ambulatory by age 5",
            "symptomatic_hsct_benefit": "minimal — does not halt progression",
            "psychosine_as_biomarker": "DBS psychosine currently validated",
        },
        "dx_delay_distribution": {
            "neonatal_NBS": 40,
            "infant_3_12m": 40,
            "child_1_10y": 10,
            "adult_18y_plus": 10,
        },
    },

    # -- ARSA -- Metachromatic Leukodystrophy (MLD) ----------------------------
    {
        "gene": "ARSA",
        "protein": (
            "ARSA -- 22q13.33 AR -- Arylsulfatase-A-507aa -- "
            "Metachromatic-Leukodystrophy-MLD -- "
            "Sulphatide-Accumulates-Oligodendrocytes-Schwann-Cells -- "
            "Libmeldy-Gene-Therapy-EMA-2020-Pre-Symptomatic -- "
            "Nerve-Conduction-Slowed-NCV-Mandatory -- "
            "ARSA-Pseudo-Deficiency-Common-Must-Exclude -- "
            "MRI-Tigroid-Leopard-Skin-Pattern-PATHOGNOMONIC"
        ),
        "alias": (
            "ARSA (arylsulfatase A); OMIM gene 607574; "
            "Metachromatic leukodystrophy (MLD) OMIM 250100. "
            "22q13.33; 507 aa; ~62 kDa; autosomal recessive. "
            "FUNCTION: ARSA encodes lysosomal arylsulfatase A, "
            "which cleaves sulphate from sulphatide (3-sulphogalactosylceramide, "
            "a major myelin lipid) and other sulpholipids. "
            "ARSA LOF -> sulphatide accumulates in myelin-forming cells: "
            "oligodendrocytes (CNS myelin) and Schwann cells (PNS myelin) -> "
            "demyelination of both CNS and PNS. "
            "The name 'metachromatic' refers to the staining property of sulphatide: "
            "sulphatide-laden macrophages stain metachomatically (brown-yellow rather than blue) "
            "with cresyl violet stain. "
            "CLINICAL FORMS: "
            "LATE INFANTILE (most common -- ~50%): "
            "Onset: 6 months to 3 years; "
            "Gait disturbance, followed by regression; "
            "NCV (nerve conduction velocity): severely SLOWED -- demyelinating peripheral neuropathy; "
            "NCV testing mandatory for diagnosis and monitoring; "
            "Progressive to tetraplegia, loss of communication, vegetative state; "
            "Death: 5-10 years after onset. "
            "JUVENILE (20-30%): "
            "Onset: 4-16 years; "
            "Behavioural changes, learning difficulties, clumsiness BEFORE motor decline; "
            "Both psychiatric and neurological features. "
            "ADULT (20-30%): "
            "Onset: >16 years; "
            "Psychiatric symptoms (psychosis, behavioural change) may PRECEDE neurological symptoms "
            "by YEARS -- frequently misdiagnosed as psychiatric disorder; "
            "MRI leukodystrophy eventually evident. "
            "MRI PATTERN: "
            "T2 hyperintensity in periventricular white matter with SPARING of U-fibres; "
            "TIGROID/LEOPARD-SKIN PATTERN: "
            "Radiating stripes of relatively normal signal interspersed with T2-high signal "
            "in the posterior deep white matter -- PATHOGNOMONIC for MLD on MRI. "
            "PSEUDO-DEFICIENCY -- CRITICAL: "
            "ARSA pseudo-deficiency alleles (p.Asn350Ser, p.Ile179Ser): "
            "Low ARSA enzyme activity with synthetic substrate in vitro BUT normal sulphatide catabolism -- "
            "NOT MLD; carrier frequency ~1:6 general population; "
            "MUST EXCLUDE pseudo-deficiency before diagnosing MLD: "
            "measure urinary sulphatides (elevated in MLD; normal in pseudo-deficiency); "
            "test for known pseudo-deficiency alleles (N350S, I179Ser). "
            "TREATMENT: "
            "Libmeldy (atidarsagene autotemcel; OTL-200): EMA approved 2020 -- "
            "first approved gene therapy for late-infantile and early-juvenile MLD; "
            "Ex vivo autologous HSC transduction with ARSA cDNA (lentiviral); "
            "Must be given pre-symptomatically (late-infantile) or early-symptomatic (early-juvenile); "
            "HSCT (allogeneic): can stabilise juvenile/adult MLD if pre-symptomatic; "
            "less effective than Libmeldy; "
            "ERT (metacholine alfa -- itepekimab): in trials; does not cross BBB well."
        ),
        "locus": "22q13.33",
        "aa": 507,
        "kDa": 62,
        "omim_gene": "607574",
        "omim_disease": "250100",
        "inheritance": "AR -- biallelic loss-of-function",
        "gene_class": "Lysosomal sulfatase — arylsulfatase A sulfatase family",
        "key_alerts": [
            "ARSA-PSEUDO-DEFICIENCY-MUST-EXCLUDE: ARSA pseudo-deficiency alleles (p.Asn350Ser, p.Ile179Ser) cause low enzyme activity with synthetic substrate but NORMAL sulphatide metabolism — NOT MLD; carrier frequency ~1:6; always measure urinary sulphatides (elevated in MLD, normal in pseudo-deficiency) AND test for pseudo-deficiency alleles before diagnosing MLD",
            "ARSA-LIBMELDY-EMA-2020-PRE-SYMPTOMATIC: Libmeldy (OTL-200 gene therapy) EMA approved 2020 for pre-symptomatic late-infantile and early-symptomatic juvenile MLD — autologous HSC gene therapy; most effective before significant neurological decline; NBS for MLD enables pre-symptomatic treatment; once symptomatic late-infantile onset, Libmeldy benefit is reduced",
            "ARSA-TIGROID-MRI-PATHOGNOMONIC: Tigroid/leopard-skin MRI pattern (stripes of relatively normal signal interspersed with T2 hyperintensity in deep periventricular white matter) is pathognomonic for MLD — seen in late-infantile and juvenile forms; U-fibre sparing distinguishes MLD from many other leukodystrophies; any child with leukodystrophy on MRI needs ARSA enzyme + sulphatide testing",
            "ARSA-NCV-MANDATORY: Nerve conduction velocity (NCV) testing is mandatory for MLD diagnosis and monitoring — severely slowed NCV (demyelinating pattern) reflects peripheral neuropathy; NCV can be abnormal before MRI changes appear in late-infantile MLD; NCV monitoring tracks disease progression and treatment response post-gene-therapy",
            "ARSA-ADULT-PSYCHIATRIC-MISDIAGNOSIS: Adult-onset MLD frequently presents with psychiatric symptoms (psychosis, personality change, behavioural problems) BEFORE neurological symptoms — commonly misdiagnosed as schizophrenia or bipolar disorder for years; MRI leukodystrophy appears later; ARSA enzyme + urinary sulphatides mandatory in any adult with unexplained psychiatric disease + subsequent neurological deterioration",
        ],
        "etiologies": [
            {"variant": "p.Pro426Leu (c.1277C>T)", "type": "missense LOF — late-infantile", "frequency": "most common late-infantile MLD allele in Europeans", "severity": "late-infantile — severe"},
            {"variant": "p.Ile179Ser (c.536T>G) PSEUDO-DEFICIENCY", "type": "pseudo-deficiency allele — NOT pathogenic", "frequency": "carrier frequency ~1:6 general population", "severity": "None — pseudo-deficiency; NOT MLD"},
            {"variant": "p.Asn350Ser (c.1049A>G) PSEUDO-DEFICIENCY", "type": "pseudo-deficiency allele — NOT pathogenic", "frequency": "common", "severity": "None — pseudo-deficiency; NOT MLD"},
            {"variant": "p.Ala212Val (c.635C>T)", "type": "missense — juvenile/adult onset", "frequency": "common juvenile allele", "severity": "juvenile or adult MLD — milder"},
            {"variant": "p.Gly99Val (c.296G>T)", "type": "missense LOF — severe", "frequency": "moderate frequency", "severity": "late-infantile — severe"},
        ],
        "stats": {
            "prevalence": "1:40,000-100,000",
            "late_infantile_proportion": "~50% of MLD",
            "juvenile_proportion": "~20-30%",
            "adult_proportion": "~20-30%",
            "pseudo_deficiency_carrier_rate": "~1:6 general population",
        },
        "dx_delay_distribution": {
            "infant_0_3y": 35,
            "child_3_10y": 25,
            "child_10_16y": 15,
            "adult_16y_plus": 25,
        },
    },
]


def _generate_patients():
    for idx, gene_data in enumerate(LSD_GENES):
        gene = gene_data["gene"]
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        patients = []
        for i in range(40):
            if gene == "GBA":
                sex = rng.choices(["M", "F"], weights=[50, 50])[0]
                gaucher_type = rng.choices(["T1", "T2", "T3"], weights=[70, 10, 20])[0]
                age_dx = rng.randint(1, 55) if gaucher_type == "T1" else rng.randint(0, 2) if gaucher_type == "T2" else rng.randint(1, 20)
                splenomegaly = rng.random() < 0.95
                erlenmeyer_flask = rng.random() < 0.55
                treatment = rng.choices(
                    ["ert_imiglucerase", "ert_velaglucerase", "srt_eliglustat", "srt_miglustat", "none"],
                    weights=[40, 20, 25, 5, 10]
                )[0] if gaucher_type == "T1" else "ert_imiglucerase"
                parkinson_counsel = rng.random() < 0.80
                splenectomy = rng.random() < 0.05
                patients.append({
                    "patient_id": f"GBA-{i+1:03d}", "sex": sex, "gaucher_type": gaucher_type,
                    "age_dx": age_dx, "splenomegaly": splenomegaly,
                    "erlenmeyer_flask_xray": erlenmeyer_flask, "treatment": treatment,
                    "parkinson_counselled": parkinson_counsel,
                    "splenectomy_performed": splenectomy,
                    "gene": gene, "seed": seed,
                })

            elif gene == "GLA":
                sex = rng.choices(["M", "F"], weights=[55, 45])[0]
                fabry_form = rng.choices(["classic", "late_onset_cardiac"], weights=[65, 35])[0]
                age_dx = rng.randint(1, 65)
                cornea_verticillata = rng.random() < 0.99 if sex == "M" else rng.random() < 0.82
                angiokeratoma = (rng.random() < 0.70) if fabry_form == "classic" else False
                treatment = rng.choices(
                    ["ert_agalsidase_alfa", "ert_agalsidase_beta", "migalastat"],
                    weights=[35, 40, 25]
                )[0]
                migalastat_amenable = rng.random() < 0.50 if treatment == "migalastat" else None
                lyso_gb3_elevated = rng.random() < 0.95
                patients.append({
                    "patient_id": f"GLA-{i+1:03d}", "sex": sex, "fabry_form": fabry_form,
                    "age_dx": age_dx, "cornea_verticillata": cornea_verticillata,
                    "angiokeratoma": angiokeratoma, "treatment": treatment,
                    "migalastat_amenability_confirmed": migalastat_amenable,
                    "lyso_gb3_elevated": lyso_gb3_elevated,
                    "gene": gene, "seed": seed,
                })

            elif gene == "GAA":
                sex = rng.choices(["M", "F"], weights=[50, 50])[0]
                pompe_type = rng.choices(["IOPD", "LOPD"], weights=[35, 65])[0]
                age_dx_months = rng.randint(0, 6) if pompe_type == "IOPD" else rng.randint(12, 600)
                cardiomegaly = (rng.random() < 0.95) if pompe_type == "IOPD" else False
                short_pr = (rng.random() < 0.70) if pompe_type == "IOPD" else False
                nbs_detected = (rng.random() < 0.60) if pompe_type == "IOPD" else (rng.random() < 0.05)
                ert = rng.choices(
                    ["avalglucosidase_alfa", "alglucosidase_alfa", "cipaglucosidase_miglustat"],
                    weights=[45, 40, 15]
                )[0]
                respiratory_support = (rng.random() < 0.70) if pompe_type == "LOPD" else (rng.random() < 0.40)
                patients.append({
                    "patient_id": f"GAA-{i+1:03d}", "sex": sex, "pompe_type": pompe_type,
                    "age_dx_months": age_dx_months, "cardiomegaly": cardiomegaly,
                    "short_pr_ecg": short_pr, "nbs_detected": nbs_detected,
                    "ert": ert, "respiratory_support": respiratory_support,
                    "gene": gene, "seed": seed,
                })

            elif gene == "HEXA":
                sex = rng.choices(["M", "F"], weights=[50, 50])[0]
                ts_form = rng.choices(["infantile", "juvenile", "adult_lots"], weights=[65, 15, 20])[0]
                age_dx_months = rng.randint(3, 18) if ts_form == "infantile" else (
                    rng.randint(24, 120) if ts_form == "juvenile" else rng.randint(144, 480))
                cherry_red_spot = (rng.random() < 0.98) if ts_form == "infantile" else (rng.random() < 0.10)
                startle_response = (rng.random() < 0.85) if ts_form == "infantile" else False
                ashkenazi = rng.random() < 0.50
                pseudo_deficiency_excluded = rng.random() < 0.90
                psychiatric_misdiagnosis = (rng.random() < 0.70) if ts_form == "adult_lots" else False
                patients.append({
                    "patient_id": f"HEXA-{i+1:03d}", "sex": sex, "ts_form": ts_form,
                    "age_dx_months": age_dx_months, "cherry_red_spot": cherry_red_spot,
                    "exaggerated_startle": startle_response, "ashkenazi_jewish": ashkenazi,
                    "pseudo_deficiency_excluded": pseudo_deficiency_excluded,
                    "psychiatric_misdiagnosis_first": psychiatric_misdiagnosis,
                    "no_approved_ert": True, "palliative_care": True,
                    "gene": gene, "seed": seed,
                })

            elif gene == "IDUA":
                sex = rng.choices(["M", "F"], weights=[50, 50])[0]
                mps1_form = rng.choices(["hurler", "hurler_scheie", "scheie"], weights=[55, 25, 20])[0]
                age_dx_months = rng.randint(6, 18) if mps1_form == "hurler" else (
                    rng.randint(12, 60) if mps1_form == "hurler_scheie" else rng.randint(24, 240))
                corneal_clouding = rng.random() < 0.95
                gibbus = (rng.random() < 0.80) if mps1_form == "hurler" else (rng.random() < 0.30)
                hsct_performed = (rng.random() < 0.85) if mps1_form == "hurler" else (rng.random() < 0.10)
                hsct_before_25 = (rng.random() < 0.70) if hsct_performed else False
                ert_laronidase = rng.random() < 0.90
                cardiac_valve = rng.random() < 0.70
                patients.append({
                    "patient_id": f"IDUA-{i+1:03d}", "sex": sex, "mps1_form": mps1_form,
                    "age_dx_months": age_dx_months, "corneal_clouding": corneal_clouding,
                    "gibbus": gibbus, "hsct_performed": hsct_performed,
                    "hsct_before_age_2_5": hsct_before_25,
                    "ert_laronidase": ert_laronidase, "cardiac_valve_disease": cardiac_valve,
                    "gene": gene, "seed": seed,
                })

            elif gene == "IDS":
                sex = rng.choices(["M", "F"], weights=[97, 3])[0]
                mps2_form = rng.choices(["severe", "attenuated"], weights=[60, 40])[0]
                age_dx_months = rng.randint(18, 48) if mps2_form == "severe" else rng.randint(36, 120)
                no_corneal_clouding = True
                pebbly_skin = rng.random() < 0.25
                hearing_loss = rng.random() < 0.80
                ert_idursulfase = rng.random() < 0.92
                intrathecal_ert = (rng.random() < 0.20) if mps2_form == "severe" else False
                hsct = False  # not routinely indicated
                patients.append({
                    "patient_id": f"IDS-{i+1:03d}", "sex": sex, "mps2_form": mps2_form,
                    "age_dx_months": age_dx_months,
                    "no_corneal_clouding": no_corneal_clouding,
                    "pebbly_ivory_skin": pebbly_skin, "hearing_loss": hearing_loss,
                    "ert_idursulfase": ert_idursulfase, "intrathecal_ert": intrathecal_ert,
                    "hsct_routinely_indicated": hsct,
                    "gene": gene, "seed": seed,
                })

            elif gene == "GALC":
                sex = rng.choices(["M", "F"], weights=[50, 50])[0]
                krabbe_form = rng.choices(["infantile", "late_onset"], weights=[87, 13])[0]
                nbs_detected = (rng.random() < 0.50) if krabbe_form == "infantile" else False
                pre_symptomatic = (rng.random() < 0.65) if nbs_detected else False
                age_dx_months = rng.randint(0, 3) if pre_symptomatic else (
                    rng.randint(3, 10) if krabbe_form == "infantile" else rng.randint(24, 480))
                hsct_performed = (rng.random() < 0.90) if pre_symptomatic else (rng.random() < 0.15)
                hsct_outcome = rng.choices(
                    ["ambulatory_good_outcome", "developmental_delay", "decline_despite_hsct"],
                    weights=[70, 20, 10]
                )[0] if (hsct_performed and pre_symptomatic) else (
                    "futile_symptomatic" if (hsct_performed and not pre_symptomatic) else "no_hsct")
                psychosine_measured = rng.random() < 0.65
                patients.append({
                    "patient_id": f"GALC-{i+1:03d}", "sex": sex, "krabbe_form": krabbe_form,
                    "nbs_detected": nbs_detected, "pre_symptomatic_at_hsct": pre_symptomatic,
                    "age_dx_months": age_dx_months, "hsct_performed": hsct_performed,
                    "hsct_outcome": hsct_outcome, "psychosine_measured": psychosine_measured,
                    "gene": gene, "seed": seed,
                })

            else:  # ARSA
                sex = rng.choices(["M", "F"], weights=[50, 50])[0]
                mld_form = rng.choices(["late_infantile", "juvenile", "adult"], weights=[50, 25, 25])[0]
                age_dx_months = rng.randint(6, 36) if mld_form == "late_infantile" else (
                    rng.randint(48, 180) if mld_form == "juvenile" else rng.randint(192, 600))
                pseudo_deficiency_excluded = rng.random() < 0.95
                urinary_sulphatides_elevated = rng.random() < 0.95
                ncv_slowed = rng.random() < 0.92
                tigroid_mri = rng.random() < 0.75
                libmeldy_eligible = (mld_form in ["late_infantile", "juvenile"]) and (rng.random() < 0.45)
                psychiatric_misdiagnosis = (rng.random() < 0.65) if mld_form == "adult" else False
                patients.append({
                    "patient_id": f"ARSA-{i+1:03d}", "sex": sex, "mld_form": mld_form,
                    "age_dx_months": age_dx_months,
                    "pseudo_deficiency_excluded": pseudo_deficiency_excluded,
                    "urinary_sulphatides_elevated": urinary_sulphatides_elevated,
                    "ncv_slowed": ncv_slowed, "tigroid_mri": tigroid_mri,
                    "libmeldy_gene_therapy": libmeldy_eligible,
                    "psychiatric_misdiagnosis_first": psychiatric_misdiagnosis,
                    "gene": gene, "seed": seed,
                })
        gene_data["patients"] = patients


_generate_patients()


def overview():
    all_genes_info = [
        {
            "gene": g["gene"],
            "locus": g["locus"],
            "aa": g["aa"],
            "n_patients": len(g["patients"]),
            "inheritance": g["inheritance"],
        }
        for g in LSD_GENES
    ]
    total = sum(len(g["patients"]) for g in LSD_GENES)
    pts = {g["gene"]: g["patients"] for g in LSD_GENES}

    def pct(lst, key, val=True):
        if not lst:
            return 0
        return round(100 * sum(1 for p in lst if p.get(key) == val) / len(lst), 1)

    def pct_true(lst, key):
        if not lst:
            return 0
        return round(100 * sum(1 for p in lst if p.get(key)) / len(lst), 1)

    return {
        "atlas": "Hereditary LSD Atlas — Complete 8-Gene Lysosomal Storage Disorder Atlas",
        "subtitle": (
            "GBA (Gaucher-ERT-SRT-Parkinson-Risk) . GLA (Fabry-Kidney-Heart-Stroke-Migalastat-Amenable) . "
            "GAA (Pompe-IOPD-Cardiomegaly-Short-PR-Avalglucosidase) . "
            "HEXA (Tay-Sachs-Cherry-Red-Spot-No-ERT-NBS-Carrier-Screen) . "
            "IDUA (MPS-I-Hurler-HSCT-Before-2.5-Laronidase) . "
            "IDS (MPS-II-Hunter-No-Corneal-Clouding-Idursulfase-No-Routine-HSCT) . "
            "GALC (Krabbe-Pre-Symptomatic-HSCT-NBS-Time-Critical) . "
            "ARSA (MLD-Libmeldy-EMA2020-Pseudo-Deficiency-Exclude-NCV-Slowed) -- "
            "320 Patients (8x40, Seeds 1790-1797)"
        ),
        "total_patients": total,
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "genes_covered": 8,
            "patients_per_gene": 40,
            "ar_genes": 7,
            "x_linked_genes": 1,
            # GBA
            "gba_ert_or_srt_on_treatment": pct_true(pts["GBA"], "treatment") and round(
                100 * sum(1 for p in pts["GBA"] if p["treatment"] != "none") / 40, 1),
            "gba_erlenmeyer_flask_pct": pct_true(pts["GBA"], "erlenmeyer_flask_xray"),
            "gba_parkinson_counselled_pct": pct_true(pts["GBA"], "parkinson_counselled"),
            "gba_splenectomy_pct": pct_true(pts["GBA"], "splenectomy_performed"),
            # GLA
            "gla_cornea_verticillata_pct": pct_true(pts["GLA"], "cornea_verticillata"),
            "gla_lyso_gb3_elevated_pct": pct_true(pts["GLA"], "lyso_gb3_elevated"),
            "gla_late_onset_cardiac_pct": pct(pts["GLA"], "fabry_form", "late_onset_cardiac"),
            # GAA
            "gaa_iopd_cardiomegaly_pct": pct_true(pts["GAA"], "cardiomegaly"),
            "gaa_nbs_detected_pct": pct_true(pts["GAA"], "nbs_detected"),
            "gaa_avalglucosidase_pct": pct(pts["GAA"], "ert", "avalglucosidase_alfa"),
            # HEXA
            "hexa_cherry_red_spot_infantile_pct": pct_true(pts["HEXA"], "cherry_red_spot"),
            "hexa_no_approved_ert_pct": 100.0,
            "hexa_pseudo_deficiency_excluded_pct": pct_true(pts["HEXA"], "pseudo_deficiency_excluded"),
            # IDUA
            "idua_hsct_performed_pct": pct_true(pts["IDUA"], "hsct_performed"),
            "idua_hsct_before_25_pct": pct_true(pts["IDUA"], "hsct_before_age_2_5"),
            "idua_corneal_clouding_pct": pct_true(pts["IDUA"], "corneal_clouding"),
            # IDS
            "ids_no_corneal_clouding_pct": 100.0,
            "ids_pebbly_skin_pct": pct_true(pts["IDS"], "pebbly_ivory_skin"),
            "ids_ert_idursulfase_pct": pct_true(pts["IDS"], "ert_idursulfase"),
            "ids_hsct_routinely_indicated_pct": 0.0,
            # GALC
            "galc_nbs_detected_pct": pct_true(pts["GALC"], "nbs_detected"),
            "galc_pre_symptomatic_hsct_pct": pct_true(pts["GALC"], "pre_symptomatic_at_hsct"),
            "galc_psychosine_measured_pct": pct_true(pts["GALC"], "psychosine_measured"),
            # ARSA
            "arsa_pseudo_deficiency_excluded_pct": pct_true(pts["ARSA"], "pseudo_deficiency_excluded"),
            "arsa_ncv_slowed_pct": pct_true(pts["ARSA"], "ncv_slowed"),
            "arsa_tigroid_mri_pct": pct_true(pts["ARSA"], "tigroid_mri"),
            "arsa_libmeldy_eligible_pct": pct_true(pts["ARSA"], "libmeldy_gene_therapy"),
        },
        "genes": all_genes_info,
        "top_alerts": [
            "GBA-PARKINSON-RISK-HETEROZYGOUS: GBA heterozygous carriers (parents/children of Gaucher patients) have 5-10x increased Parkinson risk — GBA is the MOST COMMON known genetic risk factor for PD; counsel ALL family members; GBA homozygous patients also at high PD risk; annual neurological review for carriers >40 years",
            "GBA-ELIGLUSTAT-CYP2D6-MANDATORY: Eliglustat (SRT Type 1 adults) dosing requires CYP2D6 metaboliser status BEFORE prescribing — ultra-rapid metabolisers: eliglustat not recommended; drug interactions (fluoxetine, paroxetine) alter CYP2D6; splenectomy AVOID — worsens bone disease",
            "GLA-CORNEA-VERTICILLATA-PATHOGNOMONIC-FEMALES-AFFECTED: Cornea verticillata (whorl-like corneal deposits) is pathognomonic for Fabry — present in >99% males and >80% carrier females; females are NOT just carriers — X-inactivation skewing causes renal/cardiac/CNS disease; enzyme activity UNRELIABLE in females — use GLA sequencing + lyso-Gb3",
            "GLA-MIGALASTAT-AMENABILITY-VERIFY: Migalastat (oral chaperone) is ONLY for amenable GLA variants — amenability must be confirmed by GLP assay for each specific variant; prescribing for non-amenable variant = ineffective; verify at galafoldamenabilitytable.com",
            "GAA-IOPD-CARDIOMEGALY-SHORT-PR-PATHOGNOMONIC: Infantile Pompe: massive cardiomegaly + short PR interval on ECG = PATHOGNOMONIC; any hypotonic infant with cardiomegaly needs urgent GAA DBS assay; avalglucosidase alfa (Nexviazyme) is now preferred ERT over first-generation alglucosidase",
            "HEXA-NO-ERT-NO-CURATIVE-HSCT: No approved ERT for Tay-Sachs — Hex A does not reach brain neurons; HSCT is ineffective for established disease; management is palliative; Ashkenazi Jewish carrier screening (1:27) is the ONLY effective intervention — reduces incidence >90%",
            "IDUA-HSCT-BEFORE-AGE-2.5-MANDATORY: Hurler HSCT MUST be before age 2.5 years to preserve cognition — HSCT after this window does NOT restore lost cognitive function; DQ >70 at transplant required for good cognitive outcome; always establish difficult airway plan for MPS I anaesthesia",
            "IDS-NO-CORNEAL-CLOUDING-KEY-DDX: MPS II (Hunter) has NO corneal clouding — this distinguishes it from MPS I (Hurler); no routine HSCT for MPS II unlike MPS I; pebbly ivory skin lesions (20-30%) are pathognomonic; pabinafusp alfa (Japan) is first BBB-penetrant ERT",
            "GALC-PRE-SYMPTOMATIC-HSCT-CURATIVE-SYMPTOMATIC-FUTILE: Krabbe HSCT is curative ONLY if pre-symptomatic (NBS detected); once symptomatic, HSCT does NOT halt progression; NBS referral must trigger evaluation within days — the window is weeks; psychosine DBS is now a validated biomarker",
            "ARSA-PSEUDO-DEFICIENCY-MUST-EXCLUDE: ARSA pseudo-deficiency (p.Ile179Ser, p.Asn350Ser) causes low enzyme activity with synthetic substrate but is NOT MLD — carrier rate ~1:6; always measure urinary sulphatides AND test pseudo-deficiency alleles; Libmeldy (EMA 2020) is gene therapy for pre-symptomatic/early-symptomatic MLD",
        ],
    }


def breakdown():
    result = []
    for idx, g in enumerate(LSD_GENES):
        pts = g["patients"]
        ec = {}
        for p in pts:
            et = p.get("treatment") or p.get("ts_form") or p.get("mps1_form") or p.get("mps2_form") or p.get("krabbe_form") or p.get("mld_form") or "unknown"
            ec[et] = ec.get(et, 0) + 1
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
            "etiology_counts": ec,
            "computed": {
                "n_patients": len(pts),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": pts[:10],
        })
    return result


def definitions():
    return {
        "concepts": {
            "Lysosomal Storage Disorders — Classification, Pathophysiology and the LSD Spectrum": (
                "Lysosomal storage disorders (LSDs) are a group of ~70 inherited metabolic diseases "
                "caused by deficiency of lysosomal enzymes (or transport proteins, activator proteins, "
                "or membrane proteins) required for the degradation of macromolecules within lysosomes. "
                "CLASSIFICATION BY SUBSTRATE: "
                "(1) Sphingolipidoses: substrate = sphingolipids; "
                "Gaucher (GBA — glucocerebroside), Fabry (GLA — Gb3), "
                "Niemann-Pick A/B (SMPD1 — sphingomyelin), Krabbe (GALC — galactosylceramide), "
                "MLD (ARSA — sulphatide), GM1 (GLB1), Tay-Sachs/GM2 (HEXA/HEXB); "
                "(2) Mucopolysaccharidoses (MPS): substrate = glycosaminoglycans (GAGs); "
                "MPS I Hurler (IDUA), MPS II Hunter (IDS), MPS III Sanfilippo (SGSH, NAGLU, HGSNAT, GNS), "
                "MPS IV Morquio (GALNS, GLB1), MPS VI Maroteaux-Lamy (ARSB), MPS VII Sly (GUSB); "
                "(3) Glycogen storage disease II (GSD-II / Pompe): substrate = glycogen; GAA; "
                "(4) Oligosaccharidoses: mannosidosis (MAN2B1), fucosidosis; "
                "(5) Neuronal ceroid lipofuscinoses (NCLs): CLN1-CLN14; "
                "(6) Mucolipidoses: ML I-IV. "
                "COMMON PATHOPHYSIOLOGY: "
                "Substrate accumulates in lysosomes -> lysosomal enlargement -> cell dysfunction -> "
                "organ/tissue failure depending on which cells express the enzyme and which substrate; "
                "CNS involvement: substrate accumulates in neurons -> neurodegeneration; "
                "Visceral involvement: substrate in macrophages, hepatocytes -> organomegaly; "
                "Skeletal involvement: substrate in bone cells -> dysostosis multiplex. "
                "INCIDENCE: Each individual LSD is rare (1:50,000-1:200,000) but collectively ~1:5,000-7,500 live births. "
                "DIAGNOSIS PATHWAY: "
                "(1) Clinical suspicion (coarse facies, organomegaly, developmental regression, bone disease); "
                "(2) Urine biochemistry: GAGs, sulphatides, oligosaccharides; "
                "(3) Enzyme activity assay (DBS or leukocytes); "
                "(4) Genetic confirmation (gene sequencing); "
                "(5) Biomarkers (lyso-Gb1 Gaucher, lyso-Gb3 Fabry, Glc4 Pompe, psychosine Krabbe). "
                "NEWBORN SCREENING: "
                "NBS programmes detect LSDs pre-symptomatically; "
                "critical for conditions where early treatment is transformative: "
                "Pompe (IOPD: ERT before cardiac failure), Krabbe (HSCT before symptoms), "
                "Fabry, Gaucher -- DBS multiplex enzyme assay panels now available."
            ),
            "Enzyme Replacement Therapy (ERT) in Lysosomal Storage Disorders — Principles and Limitations": (
                "ERT delivers functional recombinant enzyme intravenously to correct the lysosomal enzyme deficiency. "
                "MECHANISM: "
                "Recombinant enzyme -> IV infusion -> binds mannose-6-phosphate (M6P) receptors "
                "on cell surface -> internalised -> delivered to lysosomes -> cleaves accumulated substrate. "
                "KEY PRINCIPLE -- BBB IMPERMEABILITY: "
                "ERT does NOT cross the blood-brain barrier in clinically meaningful quantities; "
                "therefore ERT does NOT treat CNS disease; "
                "conditions with CNS disease (MPS I severe, MPS II severe, MLD, Tay-Sachs, Krabbe): "
                "ERT alone is insufficient; HSCT or gene therapy required for CNS benefit. "
                "APPROVED ERTs IN THIS ATLAS: "
                "Imiglucerase/velaglucerase/taliglucerase (Gaucher -- GBA); "
                "Agalsidase alfa/beta (Fabry -- GLA); "
                "Alglucosidase alfa / avalglucosidase alfa / cipaglucosidase+miglustat (Pompe -- GAA); "
                "Laronidase (MPS I -- IDUA); "
                "Idursulfase / pabinafusp alfa (MPS II -- IDS); "
                "NOTE: No approved ERT for Tay-Sachs (HEXA), Krabbe (GALC), or MLD (ARSA). "
                "LIMITATIONS: "
                "Infusion reactions (10-30%): pre-medicate with antihistamine/paracetamol; "
                "Antibody formation: anti-drug antibodies reduce efficacy (especially in CRIM-negative patients); "
                "CRIM (cross-reactive immunological material): "
                "CRIM-negative IOPD patients (null GAA mutations): no residual protein -> "
                "immune response to ERT -> poor outcome without immunotolerance induction; "
                "High-dose immunotolerance induction (rituximab + methotrexate) before ERT in CRIM-negative IOPD; "
                "ERT is lifelong (q1-2 week infusions) -- significant patient burden; "
                "Skeletal disease responds poorly to ERT in MPS (does not correct bone architecture). "
                "MONITORING: "
                "Biomarkers: chitotriosidase + lyso-Gb1 (Gaucher), lyso-Gb3 (Fabry), Glc4 (Pompe), "
                "urine GAGs (MPS I/II), urinary sulphatides (MLD); "
                "Organ-specific: echo (Fabry/Pompe cardiac), GFR/albumin:creatinine (Fabry renal), "
                "6MWT (Gaucher bone/MPS II motor), pulmonary function (Pompe, MPS)."
            ),
            "HSCT in Lysosomal Storage Disorders — When It Works and When It Does Not": (
                "Haematopoietic stem cell transplantation corrects lysosomal enzyme deficiency "
                "by replacing patient's haematopoietic cells (including brain-resident microglia) "
                "with donor cells expressing normal enzyme activity. "
                "HOW HSCT WORKS IN LSDs: "
                "Donor-derived monocytes -> cross BBB -> differentiate into microglia -> "
                "provide functional enzyme in the CNS; "
                "Process is slow (months-years for full microglial replacement); "
                "Therefore: HSCT effective ONLY if performed before significant neuronal loss; "
                "once neurons are dead, replacing microglia cannot restore function. "
                "CONDITIONS WHERE HSCT IS EFFECTIVE (THIS ATLAS): "
                "MPS I Hurler (IDUA): HSCT before age 2.5 years -- "
                "preserves cognitive function if performed early; standard of care; "
                "Krabbe (GALC): HSCT in pre-symptomatic infants (NBS) -- curative; "
                "MLD (ARSA): HSCT in pre-symptomatic/early juvenile -- "
                "stabilises (but gene therapy now preferred); "
                "CONDITIONS WHERE HSCT IS NOT ROUTINE OR INEFFECTIVE: "
                "Gaucher (GBA): HSCT not recommended -- ERT/SRT effective; "
                "Fabry (GLA): HSCT not recommended -- ERT/migalastat effective; "
                "Pompe (GAA): HSCT not effective -- ERT the treatment; "
                "Tay-Sachs (HEXA): HSCT ineffective for established infantile disease; "
                "MPS II Hunter (IDS): HSCT NOT routinely indicated (unlike MPS I); "
                "evidence insufficient for neurocognitive benefit. "
                "TIMING PRINCIPLE (CRITICAL): "
                "The earlier the HSCT relative to neurological involvement, the better the outcome; "
                "NBS programs for Krabbe (NY state) and MLD (some centres) enable pre-symptomatic HSCT; "
                "HSCT after symptom onset in conditions where it is indicated: "
                "substantially worse outcomes. "
                "CONDITIONING: "
                "Myeloablative conditioning (busulfan-based) required for engraftment in LSDs; "
                "reduced-intensity conditioning associated with graft failure or insufficient chimaerism."
            ),
            "Gene Therapy in Lysosomal Storage Disorders — Approved and Emerging": (
                "Gene therapy in LSDs aims to restore permanent enzyme expression by delivering "
                "a functional copy of the deficient gene. "
                "APPROVED IN THIS ATLAS: "
                "Libmeldy (atidarsagene autotemcel / OTL-200; EMA 2020) for MLD: "
                "Ex vivo autologous HSC gene therapy; lentiviral ARSA cDNA; "
                "First approved gene therapy for a neurological LSD; "
                "Indicated for pre-symptomatic late-infantile MLD and early-symptomatic juvenile MLD; "
                "dramatically superior to HSCT or ERT for MLD; "
                "Available only at specialist centres (Italy, Germany, UK). "
                "EMERGING APPROACHES: "
                "In vivo AAV gene therapy: "
                "GBA/Gaucher: AAV9-GBA intrathecal (CNS Gaucher/PD prevention -- trials); "
                "GAA/Pompe: AAV8-GAA hepatic (reduces GAG antibody formation -- trials); "
                "HEXA/Tay-Sachs: bilateral intracranial AAV9-HEXA -- Phase 1/2 trials; "
                "GALC/Krabbe: CNS-directed AAV (intrathecal + intracerebral -- preclinical to Phase 1); "
                "IDUA/MPS I: AAV9-IDUA intrathecal -- trials ongoing; "
                "GLA/Fabry: AAV gene therapy trials for sustained enzyme expression. "
                "ADVANTAGES OVER ERT: "
                "Single administration (no q2-week infusions); "
                "Potentially CNS-penetrant (intrathecal/intracerebral delivery); "
                "No need for immunotolerance induction (autologous = no immune rejection); "
                "Sustained expression reduces monitoring burden. "
                "CHALLENGES: "
                "Pre-existing AAV antibodies (especially AAV9): may need pre-screening; "
                "Long-term durability unknown; "
                "Manufacturing complexity and cost; "
                "Immunosuppression needed peri-administration."
            ),
            "Pseudo-Deficiency in Lysosomal Enzyme Assays — Avoiding Misdiagnosis": (
                "Pseudo-deficiency refers to variants that reduce enzyme activity with ARTIFICIAL "
                "(synthetic fluorescent) substrates in vitro but do NOT affect the enzyme's ability "
                "to cleave the NATURAL substrate in vivo -- therefore NOT causing disease. "
                "CLINICALLY IMPORTANT PSEUDO-DEFICIENCY VARIANTS IN THIS ATLAS: "
                "HEXA (Tay-Sachs): "
                "p.Arg247Trp (c.739C>T) and p.Arg249Trp: "
                "reduce Hex A activity with synthetic substrate; "
                "normal GM2 cleavage in vivo; NOT Tay-Sachs; "
                "common in non-Ashkenazi populations; detected in carrier screening -- "
                "causes ENORMOUS anxiety if not identified as pseudo-deficiency; "
                "Resolve by: natural substrate (GM2) assay or specific genotyping. "
                "GALC (Krabbe): "
                "p.Leu634Ser (c.1901T>C): "
                "reduces galactocerebrosidase activity in DBS assay; "
                "NOT Krabbe disease; carrier rate ~1:150 in some populations; "
                "creates false positives in NBS programs; "
                "Resolve by: GALC sequencing + psychosine DBS. "
                "ARSA (MLD): "
                "p.Asn350Ser (c.1049A>G) and p.Ile179Ser (c.536T>G): "
                "reduce ARSA activity; NOT MLD; "
                "ARSA pseudo-deficiency carrier rate ~1:6 in general population; "
                "Resolve by: urinary sulphatides (normal in pseudo-deficiency, elevated in MLD) "
                "+ ARSA gene sequencing. "
                "GBA (Gaucher): "
                "No classic HEXA/GALC/ARSA pseudo-deficiency alleles; "
                "however, GBAP1 (GBA pseudogene) amplification during PCR can give misleading sequencing results; "
                "Long-range PCR required to distinguish GBA from GBAP1. "
                "GENERAL PRINCIPLE: "
                "For every LSD with known pseudo-deficiency alleles: "
                "low enzyme activity + suspected pseudo-deficiency variant -> "
                "ALWAYS confirm with: (1) biomarker (natural substrate / metabolite) and "
                "(2) full gene sequencing before informing patient of LSD diagnosis; "
                "failing to exclude pseudo-deficiency has caused significant psychological harm. "
            ),
        },
        "pharmacological_distinctions": [
            "Imiglucerase (Cerezyme 60 U/kg IV q2w) vs velaglucerase alfa (VPRIV 60 U/kg IV q2w) vs taliglucerase alfa (Elelyso 60 U/kg IV q2w) — all target Gaucher Type 1 visceral disease (spleen/liver/bone/haematology); imiglucerase Chinese hamster ovary-derived; velaglucerase human fibroblast-derived; equivalent efficacy; dose can be reduced to 30-45 U/kg q2w after stabilisation",
            "Eliglustat (Cerdelga 84 mg oral BD or 84 mg QD in poor metabolisers) vs miglustat (Zavesca 100 mg TID) — both are SRT for Gaucher Type 1 adults; eliglustat: ceramide analogue, requires CYP2D6 metaboliser testing MANDATORY, significant drug interactions, DO NOT use in poor metabolisers without dose adjustment, not for CNS Gaucher; miglustat: imino sugar, GI side effects (diarrhoea, weight loss), also approved for Niemann-Pick C (NPC) neurological progression",
            "Agalsidase alfa (Replagal 0.2 mg/kg IV q2w — European approval) vs agalsidase beta (Fabrazyme 1.0 mg/kg IV q2w — US/global approval) — both ERT for Fabry (GLA); 5-fold dose difference; clinical equivalence debated; supply shortage of agalsidase beta in 2009-2012 required dose reduction; infusion reactions more common with higher dose agalsidase beta",
            "Migalastat (Galafold 123 mg oral q48h alternating days) — oral pharmacological chaperone for Fabry disease; stabilises misfolded alpha-galactosidase A to allow correct lysosomal trafficking; ONLY effective for amenable GLA variants (confirmed by GLP cell-based assay); NOT for all GLA variants; check galafoldamenabilitytable.com; significant drug interactions on alternating-day schedule",
            "Alglucosidase alfa (Myozyme/Lumizyme 20 mg/kg IV q2w) vs avalglucosidase alfa (Nexviazyme/Nexviadyme 20 mg/kg IV q2w) — both ERT for Pompe (GAA); avalglucosidase has ~15x higher M6P receptor affinity due to engineered high M6P content; COMET trial: avalglucosidase superior to alglucosidase for motor and respiratory outcomes; avalglucosidase now preferred first-line",
            "Cipaglucosidase alfa + miglustat (Pombiliti + Opfolda; q2w IV cipaglucosidase + daily oral miglustat) — next-generation Pompe combination ERT+SRT; miglustat stabilises cipaglucosidase during trafficking to lysosome; used in patients who have switched from alglucosidase or avalglucosidase; approved in patients already on ERT with inadequate response",
            "Laronidase (Aldurazyme 0.58 mg/kg IV q1w) — ERT for MPS I (IDUA); improves liver, spleen, joint mobility, 6MWT, respiratory; does NOT cross BBB; used as bridge to HSCT in Hurler and long-term for MPS IS/IHS; infusion time 4 hours; pre-medicate with antihistamine/paracetamol; check for anaphylaxis risk in patients with high antibody titres",
            "Idursulfase (Elaprase 0.5 mg/kg IV q1w) for MPS II somatic disease — improves liver/spleen, 6MWT, pulmonary; NO CNS penetration; intrathecal idursulfase (IT; Hunterase) monthly intrathecal injection for CNS disease; pabinafusp alfa (JR-141 IV q2w — Japan approved): IDS fused to anti-TfR1 antibody; first BBB-penetrant ERT for MPS II — crosses BBB via transferrin receptor-mediated transcytosis",
            "Libmeldy (atidarsagene autotemcel; OTL-200) — ex vivo autologous HSC lentiviral gene therapy delivering ARSA cDNA; EMA approved 2020 for MLD; one-time administration after myeloablative conditioning; must be given pre-symptomatic (late-infantile) or early-symptomatic (early-juvenile); manufacturing time ~3-6 months from HSC collection; available specialist centres only",
        ],
        "key_standards": [
            "LSD Newborn Screening Protocol: DBS multiplex enzyme assay (GAA, GLA, GBA, GALC, IDUA, ARSA) increasingly used; positive result requires urgent specialist referral within days (especially IOPD and Krabbe where time window is critical); confirmatory enzyme assay + genetic sequencing + biomarker panel before treatment initiation; pseudo-deficiency variants must be excluded",
            "Gaucher Disease Monitoring (GBA): baseline and q6-monthly: Hb, platelets, chitotriosidase, lyso-Gb1, liver/spleen volumes by MRI; bone X-ray (Erlenmeyer flask monitoring); DEXA annually; bone marrow MRI q1-2 years; neurological assessment annually (Type 3); Parkinson surveillance in all patients and first-degree relatives (carriers) from age 40",
            "Fabry Disease Monitoring (GLA): baseline and annually: GFR/albumin-creatinine ratio (renal), echo/cardiac MRI (LVH/fibrosis), 24h Holter (arrhythmia), neurological review, lyso-Gb3; brain MRI at baseline (white matter lesions/stroke); ophthalmological review (cornea verticillata, macular involvement); audiometry annually; genetic testing of all first-degree relatives",
            "Pompe Disease Ventilation Protocol (GAA): FVC upright and supine at every visit; if FVC supine drops >10% vs upright = diaphragm involvement; nocturnal BiPAP initiated when FVC <80% or significant nocturnal desaturation on pulse oximetry; IOPD: echo every 3-6 months on ERT (track LV mass regression); ECG for short PR monitoring; CK and Glc4 q6 monthly",
            "MPS I (IDUA) HSCT Protocol: refer urgently to transplant centre as soon as Hurler diagnosis confirmed; target HSCT before age 2.5 years AND DQ>70; pre-HSCT: laronidase ERT (bridge); donor selection: 10/10 matched unrelated donor or matched sibling preferred; myeloablative busulfan conditioning; post-HSCT: continue laronidase for somatic disease; cognitive monitoring (DQ/IQ) annually; airway plan for all anaesthetics",
            "Krabbe Disease (GALC) NBS-HSCT Pathway: DBS-positive Krabbe NBS -> refer specialist centre within 24-48 hours; confirmatory enzyme assay + GALC sequencing + psychosine DBS; MRI and NCV to assess symptomatic status; pre-symptomatic: HSCT urgently (aim <30 days if possible); psychosine DBS guides prognosis; post-HSCT: psychosine monitoring, neurological assessment q6 monthly, MRI annually",
            "MLD (ARSA) Libmeldy Eligibility Criteria: EMA approval: pre-symptomatic late-infantile MLD (siblings of affected child, NBS detected) OR early-symptomatic juvenile MLD (IQ>70, GMFM>40); confirm by ARSA enzyme + urinary sulphatides + exclude pseudo-deficiency; manufacturing takes 3-6 months — refer immediately on diagnosis; NCV monitoring pre/post-Libmeldy; available specialist centres in Italy, Germany, UK",
            "Pseudo-Deficiency Exclusion Standard: any LSD enzyme assay showing low activity MUST have pseudo-deficiency excluded before clinical diagnosis; HEXA: test p.Arg247Trp + p.Arg249Trp or natural substrate assay; GALC: test p.Leu634Ser + psychosine DBS; ARSA: urinary sulphatides + test p.Asn350Ser + p.Ile179Ser; GBA: long-range PCR to distinguish from GBAP1 pseudogene; never inform patient of LSD diagnosis based on enzyme activity alone without pseudo-deficiency exclusion",
        ],
    }
