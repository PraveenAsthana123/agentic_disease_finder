#!/usr/bin/env python3
"""Hereditary-BMF-Atlas — Complete 8-Gene Hereditary Bone Marrow Failure Syndromes Atlas
FANCA   (Fanconi Anaemia Complementation Group A; 1455 aa; 16q24.3; AR;
          Most common FA group — 60–70% of all FA; FANCA-FANCG heterodimer initiates FA core complex;
          Chromosome fragility test (DEB/MMC) diagnostic — mandatory before gene panel;
          BMF median onset age 7; AML risk 30–40% lifetime; solid tumours (head/neck SCC);
          HSCT curative for BMF; reduced-intensity conditioning MANDATORY — avoid alkylators/radiation;
          Androgen therapy (oxymetholone) bridge to HSCT; avoid tumour-toxic exposures) ·
FANCD2  (Fanconi Anaemia Complementation Group D2; 1451 aa; 3p25.3; AR;
          FANCD2 monoubiquitinated at K561 by FA core complex — central FA pathway node;
          Severe phenotype: early-onset BMF, high solid tumour risk, VACTERL association common;
          DEB/MMC fragility test strongly positive; multiple congenital anomalies;
          FANCD2 foci on replication fork stalling — DNA interstrand crosslink repair;
          Early HSCT recommended; complementation testing classifies FA group) ·
DKC1    (Dyskerin; 514 aa; Xq28; X-linked recessive;
          Dyskeratosis congenita — X-linked form; nail dystrophy + oral leukoplakia + skin reticulate pigmentation CLASSIC TRIAD;
          Dyskerin stabilises H/ACA snoRNAs and is integral to telomerase (TERC component);
          Telomeres very short — below 1st percentile on flow-FISH; DKC1 is more severe than TERC/TERT;
          Progressive BMF; pulmonary fibrosis 20% — NO HSCT if pulmonary fibrosis established;
          Head/neck SCC risk; androgen therapy partially responsive) ·
TERC    (Telomerase RNA Component; ~451 nt RNA; 3q26.2; AD with anticipation;
          TERC mutations — autosomal dominant DC/aplastic anaemia;
          Telomeres shorten each generation — anticipation: grandchildren more severely affected;
          Androgen/danazol therapy responsive — first-line before HSCT in older patients;
          Liver fibrosis + cirrhosis common; pulmonary fibrosis occurs;
          Milder phenotype than DKC1; TERC loss → TERT enzyme unstable;
          Flow-FISH telomere length mandatory — identifies cryptic telomere biology disorders) ·
TERT    (Telomerase Reverse Transcriptase; 1132 aa; 5p15.33; AD/AR;
          Allelic to TERC spectrum — TERT LOF → telomerase insufficient;
          Liver cirrhosis + idiopathic pulmonary fibrosis WITHOUT BMF frequent — tissue-specific;
          BMF variable — aplastic anaemia or isolated cytopenia;
          Homozygous/compound heterozygous → severe DC-like phenotype;
          Heterozygous → variable penetrance — liver > lung > marrow axis;
          Telomere length flow-FISH guides management decisions) ·
ELANE   (Neutrophil Elastase; 267 aa; 19p13.3; AD;
          Severe congenital neutropenia (SCN1/Kostmann type 2) + cyclic neutropenia;
          ANC persistently <200 (SCN) or cyclically 14–21 day cycle (cyclic neutropenia);
          G-CSF (filgrastim) standard first-line — dose 5–10 mcg/kg/day; escalate if ANC target not reached;
          10–15% AML/MDS risk — annual bone marrow surveillance mandatory;
          HSCT for G-CSF failure, AML/MDS, or intolerance;
          ELANE misfolds → ER stress → apoptosis of neutrophil precursors) ·
SBDS    (Shwachman-Bodian-Diamond Syndrome; 250 aa; 7q11.21; AR;
          Shwachman-Diamond syndrome — exocrine pancreatic insufficiency + BMF + skeletal dysplasia (metaphyseal);
          Neutropenia most common cytopenia; thrombocytopenia and aplastic anaemia occur;
          15–30% AML/MDS lifetime risk — annual marrow surveillance mandatory;
          SBDS required for ribosome biogenesis — GTPase EFL1 ejects eIF6 from 60S subunit;
          Pancreatic enzyme replacement therapy (PERT) + fat-soluble vitamins A/D/E/K mandatory;
          HSCT curative for BMF — does NOT correct pancreatic/skeletal disease) ·
GATA2   (GATA Binding Protein 2; 480 aa; 3q21.3; AD;
          GATA2 deficiency syndrome — MonoMAC syndrome / Emberger syndrome;
          Monocytopenia (monocytes <10) + NK depletion + B cell depletion + dendritic cell depletion;
          Atypical mycobacterial infections, HPV-related warts, viral infections (CMV/EBV/HSV);
          Lymphoedema in Emberger variant — primary lymphoedema + MDS;
          MDS/AML in >80% — earliest malignant transformation of any inherited BMF syndrome;
          HSCT CURATIVE — corrects immunodeficiency AND haematological malignancy risk;
          GATA2 monocytopenia panel + flow cytometry essential for diagnosis)
320-patient aggregate cohort (8 × 40, seeds 1438–1445)
"""

import random

SEED_BASE = 1438

BMF_GENES = [
    # ── FANCA — Most common FA group ──
    {
        "gene": "FANCA",
        "protein": "Fanconi Anaemia Complementation Group A Protein",
        "alias": (
            "FANCA; OMIM gene 607139; Fanconi Anaemia OMIM 227650; 16q24.3; 1455 aa; ~163 kDa; "
            "Most common FA complementation group — 60–70% of all FA cases; "
            "FANCA-FANCG heterodimer initiates FA core complex assembly; "
            "DEB/mitomycin C chromosome fragility test: diagnostic — MANDATORY before NGS panel; "
            "Biallelic LOF → defective DNA interstrand crosslink repair → replication fork stalling → BMF; "
            "Radial chromosomes on DEB test pathognomonic; "
            "IVS4+4A>T Portuguese founder; exon 17 c.1FA compound splicing Brazilian; "
            "HSCT curative for BMF — reduced-intensity conditioning MANDATORY — standard myeloablative LETHAL"
        ),
        "aa": "1455 aa",
        "kDa": "~163 kDa",
        "locus": "16q24.3",
        "omim_gene": 607139,
        "omim_disease": 227650,
        "inheritance": "AR — biallelic LOF; rarely mosaic revertants (spontaneous somatic correction — DEB test may be misleadingly negative)",
        "gene_class": (
            "FANCA encodes the largest subunit of the Fanconi anaemia core complex, an E3 ubiquitin "
            "ligase that monoubiquitinates FANCD2 and FANCI at stalled replication forks. FANCA-FANCG "
            "heterodimer nucleates assembly of the 8-subunit FA core complex (FANCA/B/C/E/F/G/L/M). "
            "Biallelic FANCA LOF prevents FANCD2 monoubiquitination → DNA interstrand crosslinks are "
            "not repaired → replication fork collapse → haematopoietic cell apoptosis → bone marrow "
            "failure. Chromosome fragility (DEB/MMC) testing is the gold-standard diagnostic test — "
            "radial chromosomes are pathognomonic. Somatic mosaicism (reversion) can suppress DEB "
            "positivity in blood — skin fibroblasts may be needed. FANCA is heterogeneous: >2000 "
            "distinct pathogenic variants. Tumour predisposition (AML, SCC head/neck) requires "
            "life-long surveillance. HSCT cures BMF but does NOT prevent solid tumours."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("Biallelic frameshift (exon 11 hotspot) — pan-ethnic", 0.22),
            ("Biallelic splice-site variants", 0.18),
            ("IVS4+4A>T Portuguese/Brazilian founder (compound heterozygous)", 0.15),
            ("Large exonic deletion (MLPA mandatory) + point variant", 0.20),
            ("Biallelic nonsense variants", 0.15),
            ("Mosaic revertant + novel pathogenic — DEB borderline", 0.10),
        ],
        "age_onset_years_range": (2, 12),
        "sex_ratio_M": 0.52,
        "rates": {
            "bone_marrow_failure":              0.90,
            "aml_mds_lifetime":                 0.35,
            "head_neck_scc_lifetime":           0.28,
            "gynaecological_scc":               0.20,
            "hsct_performed":                   0.65,
            "androgen_therapy_bridge":          0.45,
            "congenital_anomalies_cafe_au_lait": 0.75,
            "radial_thumb_aplasia":             0.30,
            "short_stature":                    0.55,
            "microcephaly":                     0.25,
            "renal_anomalies":                  0.35,
            "deb_test_positive":                0.92,
            "mosaic_deb_borderline":            0.08,
        },
        "critical_alerts": [
            "ALKYLATING-AGENTS-ABSOLUTE-CI: Cyclophosphamide, busulfan, melphalan in standard doses lethal — reduced-intensity conditioning ONLY",
            "RADIATION-REDUCED: Standard-dose radiation causes severe toxicity — dose-reduce by ≥30% minimum",
            "DEB-TEST-BEFORE-NGS: Chromosome fragility (DEB/MMC) MANDATORY first — mosaic revertants can cause false-negative blood DEB",
            "ANDROGEN-THERAPY-BRIDGE: Oxymetholone/danazol improves counts while awaiting HSCT — NOT curative, monitor liver",
            "ANNUAL-MARROW-SURVEILLANCE: Bone marrow biopsy + cytogenetics annually — AML/MDS in 35%",
            "SOLID-TUMOUR-SURVEILLANCE: Annual head/neck + gynaecological examination from age 15 — SCC risk high",
            "HSCT-CURATIVE-FOR-BMF: Matched sibling or MUD HSCT curative for BMF — does NOT prevent solid tumours",
            "SOMATIC-MOSAICISM: Skin fibroblast DEB if blood DEB negative but clinical suspicion high",
        ],
        "key_ddx_rules": [
            "DEB positive → FA confirmed — no other inherited BMF causes DEB/MMC-fragility",
            "DEB borderline → test skin fibroblasts (somatic mosaicism) and test FANCA/FANCC/FANCG in parallel",
            "Complementation testing with patient cells classifies FA group — guides gene-specific therapy trials",
        ],
    },
    # ── FANCD2 — Central FA Pathway Node ──
    {
        "gene": "FANCD2",
        "protein": "Fanconi Anaemia Complementation Group D2 Protein",
        "alias": (
            "FANCD2; OMIM gene 613984; Fanconi Anaemia D2 OMIM 227646; 3p25.3; 1451 aa; ~162 kDa; "
            "FANCD2 monoubiquitinated at K561 by FA core complex — central node activating HR repair; "
            "FANCD2-FANCI heterodimer (ID2 complex) loads onto stalled replication forks; "
            "Severe phenotype: earlier BMF onset, more congenital anomalies, higher tumour risk; "
            "VACTERL association (vertebral, anal, cardiac, tracheo-oesophageal, renal, limb) frequent; "
            "DEB/MMC fragility positive — strongly positive relative to FANCA"
        ),
        "aa": "1451 aa",
        "kDa": "~162 kDa",
        "locus": "3p25.3",
        "omim_gene": 613984,
        "omim_disease": 227646,
        "inheritance": "AR — biallelic LOF; both sexes equally affected",
        "gene_class": (
            "FANCD2 is the central effector of the Fanconi anaemia pathway. Following DNA interstrand "
            "crosslink detection at stalled replication forks, the FA core complex monoubiquitinates "
            "FANCD2 at K561 and FANCI at K523. Monoubiquitinated FANCD2-FANCI (ID2 complex) recruits "
            "SLX4 (FANCP) nuclease complex and BRCA2 (FANCD1) to coordinate ICL unhooking, translesion "
            "synthesis, and homologous recombination repair. FANCD2-deficient cells show severe "
            "chromosome instability. Clinically, FANCD2-FA patients tend to have earlier BMF onset "
            "(median age 4), more frequent VACTERL associations, higher rates of brain tumours, and "
            "more severe solid tumour predisposition than FANCA-FA. Complementation with FANCA cells "
            "does not correct DEB sensitivity, distinguishing groups A and D2."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("Biallelic frameshift FANCD2 — severe phenotype", 0.30),
            ("Biallelic splice-site FANCD2 — partial residual protein", 0.25),
            ("Biallelic nonsense — null — neonatal/infantile BMF", 0.20),
            ("Missense + frameshift compound heterozygous", 0.15),
            ("Exonic deletion detected by MLPA", 0.10),
        ],
        "age_onset_years_range": (1, 8),
        "sex_ratio_M": 0.50,
        "rates": {
            "bone_marrow_failure":              0.95,
            "aml_mds_lifetime":                 0.45,
            "brain_tumour_medulloblastoma":     0.12,
            "head_neck_scc":                    0.22,
            "vacterl_association":              0.38,
            "short_stature":                    0.70,
            "microcephaly":                     0.40,
            "thumb_anomaly":                    0.35,
            "renal_horseshoe":                  0.28,
            "deb_test_strongly_positive":       0.97,
            "early_hsct_required":              0.78,
        },
        "critical_alerts": [
            "ALKYLATING-AGENTS-ABSOLUTE-CI: Reduced-intensity conditioning MANDATORY — standard alkylation lethal",
            "VACTERL-SCREEN: Cardiac echo + renal USS + spinal X-ray at diagnosis — VACTERL in 38%",
            "BRAIN-TUMOUR-MRI: Annual brain MRI from age 2 — medulloblastoma/brain SCC risk higher than FANCA",
            "EARLY-HSCT: BMF onset earlier than FANCA — HSCT window narrower — refer haematology immediately",
            "DEB-STRONGLY-POSITIVE: FANCD2 DEB score higher than FANCA — diagnostic certainty high",
            "COMPLEMENTATION-TESTING: Classify FA group — FANCD2 patients eligible for group-specific trials",
        ],
        "key_ddx_rules": [
            "FANCD2 DEB score typically higher than FANCA — but group distinction requires complementation assay",
            "VACTERL + DEB positive → strongly suspect FANCD2 (vs FANCA more likely without VACTERL)",
            "Brain tumour in FA context → FANCD2 or FANCD1 (BRCA2) — test both",
        ],
    },
    # ── DKC1 — X-linked Dyskeratosis Congenita ──
    {
        "gene": "DKC1",
        "protein": "Dyskerin (H/ACA Ribonucleoprotein Complex Component DKC1)",
        "alias": (
            "DKC1; OMIM gene 300126; Dyskeratosis Congenita X-linked OMIM 305000; Xq28; 514 aa; ~57.7 kDa; "
            "Dyskerin pseudouridine synthase — H/ACA snoRNP complex; TERC (telomerase RNA) stability; "
            "Classic triad: nail dystrophy + oral leukoplakia + skin reticulate pigmentation — appears age 5-15; "
            "Telomeres <1st percentile on flow-FISH — most severe telomere shortening of all DC genes; "
            "X-linked — males severely affected; females carrier (mild/unaffected usually); "
            "Pulmonary fibrosis in 20% — ABSOLUTE CONTRAINDICATION to HSCT if established PF"
        ),
        "aa": "514 aa",
        "kDa": "~57.7 kDa",
        "locus": "Xq28",
        "omim_gene": 300126,
        "omim_disease": 305000,
        "inheritance": "X-linked recessive — males affected; carrier females rarely manifesting",
        "gene_class": (
            "DKC1 encodes dyskerin, a pseudouridine synthase that is an integral component of H/ACA "
            "ribonucleoprotein (RNP) complexes. In addition to its role in ribosomal RNA pseudouridylation, "
            "dyskerin is essential for the stability and function of the telomerase RNA component (TERC). "
            "DKC1 mutations destabilise TERC, causing progressive telomere shortening that is more severe "
            "than AD TERC/TERT mutations. The resulting telomere biology disorder affects the most "
            "proliferative tissues: haematopoiesis (BMF), mucous membranes (oral leukoplakia), skin "
            "(reticulate pigmentation), and nails (dystrophy). Pulmonary fibrosis complicates 20% of cases "
            "and occurs independently of BMF — HSCT cannot be safely performed once PF is established "
            "because conditioning causes fatal pulmonary toxicity. Flow-FISH telomere length analysis "
            "is the diagnostic cornerstone — telomeres <1st percentile in lymphocytes."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("A353V (c.1058C>T) — most common DKC1 pathogenic variant, X-linked", 0.22),
            ("T49M (c.146C>T) — severe early BMF, < age 5", 0.15),
            ("G402E (c.1205G>A) — intermediate", 0.12),
            ("Splice-site DKC1 — Hoyeraal-Hreidarsson severe form", 0.18),
            ("Missense within pseudouridine synthase domain", 0.20),
            ("Novel hemizygous missense — no family history", 0.13),
        ],
        "age_onset_years_range": (5, 15),
        "sex_ratio_M": 0.92,
        "rates": {
            "bone_marrow_failure":              0.85,
            "nail_dystrophy":                   0.90,
            "oral_leukoplakia":                 0.80,
            "skin_reticulate_pigmentation":     0.75,
            "pulmonary_fibrosis":               0.20,
            "aml_mds":                          0.12,
            "head_neck_scc":                    0.15,
            "telomere_below_1st_percentile":    0.98,
            "androgen_responsive":              0.55,
            "hoyeraal_hreidarsson_severe":      0.10,
            "hsct_performed":                   0.50,
        },
        "critical_alerts": [
            "PULMONARY-FIBROSIS-ABSOLUTE-CI-HSCT: Established pulmonary fibrosis → HSCT ABSOLUTELY CONTRAINDICATED — conditioning pulmonary toxicity lethal",
            "FLOW-FISH-MANDATORY: Telomere length <1st percentile on flow-FISH confirms diagnosis — DKC1 most severe shortening",
            "TRIAD-SEQUENCE: Nail dystrophy → oral leukoplakia → skin pigmentation sequence — CLASSIC appearance age 5-15",
            "ANDROGEN-THERAPY: Oxymetholone/danazol improves counts in 55% — first-line for mild-moderate BMF",
            "ANNUAL-PULMONARY-SCREEN: Pulmonary function tests + DLCO annually — PF onset can be rapid",
            "HEAD-NECK-SCC-SURVEILLANCE: Annual ENT + dermatology from age 15 — leukoplakia can transform",
            "CARRIER-FEMALES: Usually unaffected — flow-FISH telomere length in mother before counselling",
        ],
        "key_ddx_rules": [
            "DEB test NEGATIVE in DC — distinguishes from FA (DEB positive)",
            "Triad + telomere below 1st percentile + X-linked → DKC1 before TERC/TERT",
            "Hoyeraal-Hreidarsson (severe DC with cerebellar hypoplasia) → DKC1 most common cause",
        ],
    },
    # ── TERC — AD Dyskeratosis Congenita with Anticipation ──
    {
        "gene": "TERC",
        "protein": "Telomerase RNA Component (H/ACA RNA, 451 nt)",
        "alias": (
            "TERC; OMIM gene 602322; Aplastic Anaemia/DC OMIM 127550; 3q26.2; 451 nt RNA gene; "
            "Telomerase RNA template strand — provides template for TERT to add TTAGGG repeats; "
            "AD inheritance with anticipation — telomeres shorter each generation; "
            "Grandparents: mild aplastic anaemia; parents: moderate BMF; grandchildren: DC ± Hoyeraal-Hreidarsson; "
            "Androgen/danazol therapy responsive — first-line for older patients or those unfit for HSCT; "
            "Liver fibrosis + pulmonary fibrosis occur — screen annually; "
            "Heterozygous LOF → TERT enzyme destabilised → progressive telomere erosion across generations"
        ),
        "aa": "451 nt RNA",
        "kDa": "RNA component",
        "locus": "3q26.2",
        "omim_gene": 602322,
        "omim_disease": 127550,
        "inheritance": "AD with anticipation — each generation more severely affected due to shorter inherited telomeres",
        "gene_class": (
            "TERC encodes the RNA component of telomerase, which provides the template sequence "
            "(5'-AACCCC-3') for TERT to synthesise telomere repeats (TTAGGG). TERC is also a structural "
            "scaffold of the H/ACA RNP complex, stabilised by dyskerin (DKC1), GAR1, NHP2, and NOP10. "
            "Heterozygous TERC mutations reduce telomerase activity — telomeres shorten progressively. "
            "Since shortened telomeres are transmitted to the next generation, each subsequent generation "
            "inherits a shorter telomere baseline and crosses the critical telomere threshold earlier in life. "
            "This 'anticipation' phenomenon means grandparents may have only mild haematological changes "
            "while grandchildren present with severe DC or even Hoyeraal-Hreidarsson. "
            "Androgen therapy (danazol, oxymetholone) upregulates TERT expression and slows telomere "
            "shortening — partially effective in ~70% of TERC patients."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("TERC template domain mutation (CR4-CR5) — functional telomerase loss", 0.30),
            ("TERC 3' H/ACA box mutation — TERC degraded", 0.25),
            ("Large TERC deletion — NGS misses unless CNV analysis included", 0.15),
            ("Pseudoknot domain mutation — TERT binding site disrupted", 0.20),
            ("Novel TERC variant — de novo (no family history of anticipation)", 0.10),
        ],
        "age_onset_years_range": (15, 45),
        "sex_ratio_M": 0.55,
        "rates": {
            "aplastic_anaemia_moderate_severe":  0.70,
            "liver_fibrosis_cirrhosis":         0.25,
            "pulmonary_fibrosis":               0.18,
            "oral_leukoplakia":                 0.45,
            "nail_dystrophy":                   0.40,
            "skin_pigmentation":                0.35,
            "aml_mds":                          0.10,
            "androgen_responsive":              0.70,
            "telomere_below_1st_percentile":    0.95,
            "family_history_anticipation":      0.65,
            "de_novo_terc":                     0.15,
        },
        "critical_alerts": [
            "ANTICIPATION-FAMILY-SCREEN: Test first-degree relatives — parents may have mild aplastic anaemia, grandchildren may be severely affected",
            "ANDROGEN-THERAPY-FIRST-LINE: Danazol 400–800 mg/day or oxymetholone — 70% response — start before HSCT in older patients",
            "LIVER-FIBROSIS-SCREEN: Annual liver function tests + USS — cirrhosis occurs independently of BMF",
            "PULMONARY-FIBROSIS-ABSOLUTE-CI-HSCT: Established PF → HSCT ABSOLUTELY CONTRAINDICATED",
            "FLOW-FISH-TELOMERE: Telomere length <1st percentile in lymphocytes — diagnoses telomere biology disorder",
            "NGS-CNV-MANDATORY: Large TERC deletions missed by point-mutation NGS — copy-number analysis required",
            "HSCT-REDUCED-CONDITIONING: If HSCT performed — non-myeloablative conditioning only — avoid busulfan",
        ],
        "key_ddx_rules": [
            "Telomere <1st percentile + AD inheritance + anticipation pattern → TERC before TERT",
            "DEB negative (no FA) + telomere very short + family with worsening disease each generation → TERC pathognomonic pattern",
            "Liver cirrhosis + aplastic anaemia in same patient/family → telomere biology (TERC/TERT) very likely",
        ],
    },
    # ── TERT — AD/AR Telomere Biology, Liver/Lung Dominant ──
    {
        "gene": "TERT",
        "protein": "Telomerase Reverse Transcriptase",
        "alias": (
            "TERT; OMIM gene 187270; Aplastic anaemia/DC OMIM 127550, 613989; 5p15.33; 1132 aa; ~127 kDa; "
            "Catalytic subunit of telomerase — reverse transcriptase adds TTAGGG using TERC template; "
            "AD heterozygous LOF: variable penetrance — liver cirrhosis > pulmonary fibrosis > BMF; "
            "AR biallelic: severe DC-like phenotype; "
            "TERT-related liver disease: cryptogenic cirrhosis — TERT often missed in liver workup; "
            "Heterozygous TERT variants found in 5–8% of 'idiopathic' pulmonary fibrosis; "
            "Androgen/danazol therapy partially effective; liver transplant for cirrhosis"
        ),
        "aa": "1132 aa",
        "kDa": "~127 kDa",
        "locus": "5p15.33",
        "omim_gene": 187270,
        "omim_disease": 127550,
        "inheritance": "AD with variable penetrance (heterozygous LOF) or AR (biallelic LOF — severe)",
        "gene_class": (
            "TERT encodes the reverse transcriptase catalytic subunit of human telomerase. It binds "
            "TERC, forms the active telomerase ribonucleoprotein, and adds TTAGGG repeats to chromosome "
            "ends. Heterozygous TERT LOF reduces telomerase activity by ~50%, causing progressive "
            "telomere shortening with tissue-specific manifestations. Unlike TERC, TERT haploinsufficiency "
            "shows strong tissue-specific penetrance: liver and lung are more sensitive than marrow, "
            "so many TERT heterozygotes present with cryptogenic cirrhosis or idiopathic pulmonary fibrosis "
            "without overt BMF. Biallelic TERT mutations cause severe disease indistinguishable from "
            "DKC1-type DC. Importantly, TERT is found in 5–8% of apparently idiopathic pulmonary fibrosis "
            "— genetic testing of all IPF patients/families is now recommended. Androgen therapy upregulates "
            "endogenous TERT expression and is partially effective in haematological manifestations."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("Heterozygous TERT RT domain missense — liver-dominant phenotype", 0.28),
            ("Heterozygous TERT TRBD domain — pulmonary fibrosis dominant", 0.22),
            ("Heterozygous TERT — aplastic anaemia without triad", 0.20),
            ("Biallelic TERT — severe DC-like with triad", 0.15),
            ("Novel heterozygous TERT — de novo — variable penetrance", 0.15),
        ],
        "age_onset_years_range": (20, 60),
        "sex_ratio_M": 0.58,
        "rates": {
            "liver_cirrhosis_fibrosis":         0.45,
            "idiopathic_pulmonary_fibrosis":    0.35,
            "aplastic_anaemia":                 0.40,
            "oral_leukoplakia":                 0.25,
            "nail_dystrophy":                   0.20,
            "telomere_below_1st_percentile":    0.90,
            "aml_mds":                          0.08,
            "androgen_responsive":              0.55,
            "liver_transplant_considered":      0.15,
            "biallelic_severe_phenotype":       0.15,
        },
        "critical_alerts": [
            "LIVER-TRANSPLANT-TERT-SCREEN: All cryptogenic cirrhosis patients — screen TERT/TERC before listing for transplant",
            "IPF-GENETIC-SCREEN: Idiopathic pulmonary fibrosis in families — TERT in 5-8% — genetic counselling mandatory",
            "ANDROGEN-THERAPY: Danazol/oxymetholone for haematological manifestations — does NOT treat lung/liver",
            "PULMONARY-FIBROSIS-ABSOLUTE-CI-HSCT: Established PF → HSCT ABSOLUTELY CONTRAINDICATED",
            "BIALLELIC-SEVERE: Biallelic TERT → DC triad + severe BMF — manage as DKC1-equivalent",
            "FLOW-FISH-TELOMERE: <1st percentile confirms telomere biology disorder — order before management decisions",
            "LIVER-LUNG-SCREEN: Annual LFTs + USS + DLCO even in patients presenting primarily with BMF",
        ],
        "key_ddx_rules": [
            "Cryptogenic cirrhosis + family history of BMF or PF → TERT/TERC first",
            "IPF family cluster → TERT heterozygous LOF in 5-8% — confirm with flow-FISH",
            "TERT vs TERC: TERT more liver/lung dominant, less anticipation — but overlap significant",
        ],
    },
    # ── ELANE — Congenital and Cyclic Neutropenia ──
    {
        "gene": "ELANE",
        "protein": "Neutrophil Elastase (Leukocyte Elastase; ELA2)",
        "alias": (
            "ELANE; OMIM gene 130130; Severe Congenital Neutropenia OMIM 202700; Cyclic Neutropenia OMIM 162800; 19p13.3; 267 aa; ~29.8 kDa; "
            "Serine protease of azurophil granules — ER stress mechanism in neutropenia; "
            "SCN (Kostmann type 2): persistent ANC <200; Cyclic neutropenia: 14–21 day cycles ANC nadir <200; "
            "G-CSF (filgrastim) 5–10 mcg/kg/day standard first-line — ANC target >1000; "
            "10–15% AML/MDS lifetime risk in SCN — annual marrow surveillance MANDATORY; "
            "HSCT for G-CSF failure, AML/MDS, or intolerance"
        ),
        "aa": "267 aa",
        "kDa": "~29.8 kDa",
        "locus": "19p13.3",
        "omim_gene": 130130,
        "omim_disease": 202700,
        "inheritance": "AD — heterozygous gain-of-misfolding — dominant negative; de novo variants common",
        "gene_class": (
            "ELANE encodes neutrophil elastase (NE), a serine protease stored in azurophilic granules "
            "of mature neutrophils. Pathogenic ELANE variants cause misfolding of neutrophil elastase "
            "within the endoplasmic reticulum of myeloid precursors, triggering the unfolded protein "
            "response and accelerated apoptosis of neutrophil precursors at the promyelocyte stage. "
            "Two distinct clinical phenotypes arise from different ELANE variants: severe congenital "
            "neutropenia (SCN1), with persistent ANC <200 and recurrent life-threatening infections "
            "from birth; and cyclic neutropenia (CN), with regular 14–21 day oscillations of ANC. "
            "G-CSF therapy is highly effective for both phenotypes, with >95% responding. AML/MDS "
            "transformation occurs in 10–15% of SCN (not CN) and is associated with CSF3R (G-CSF "
            "receptor) and RUNX1 mutations — annual marrow surveillance is mandatory for SCN."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("P110L (c.329C>T) — most common SCN variant, severe phenotype", 0.18),
            ("G214R (c.640G>A) — severe SCN", 0.12),
            ("Cyclic neutropenia variant (e.g., V181M) — 21-day cycle", 0.22),
            ("De novo missense ELANE — no family history, SCN phenotype", 0.28),
            ("ELANE frameshift — severe SCN, early infectious deaths", 0.10),
            ("Mild ELANE missense — moderate neutropenia, cyclic-like pattern", 0.10),
        ],
        "age_onset_years_range": (0, 2),
        "sex_ratio_M": 0.50,
        "rates": {
            "scn_anc_below_200_persistent":     0.60,
            "cyclic_neutropenia_pattern":       0.40,
            "recurrent_bacterial_infections":   0.95,
            "oral_ulcers_gingivitis":           0.80,
            "cellulitis_pneumonia_sepsis":      0.75,
            "gcsf_responsive":                  0.97,
            "aml_mds_lifetime_scn":             0.13,
            "csf3r_mutation_aml_risk":          0.08,
            "hsct_required_gcsf_failure":       0.10,
            "neutrophil_precursor_arrest_bm":   0.95,
            "de_novo_variant":                  0.35,
        },
        "critical_alerts": [
            "G-CSF-STANDARD-FIRST-LINE: Filgrastim 5 mcg/kg/day — titrate to ANC >1000 — >97% respond — start immediately at diagnosis",
            "ANNUAL-MARROW-SCN: Bone marrow biopsy + cytogenetics annually in SCN — AML/MDS in 13%",
            "CSF3R-MUTATION-SURVEILLANCE: G-CSF receptor mutations (CSF3R) acquired → high AML risk — monitor",
            "CYCLIC-vs-SCN: Document ANC twice-weekly for 6–8 weeks to classify — management differs",
            "GCSF-FAILURE-HSCT: G-CSF dose >40 mcg/kg/day without ANC response → HSCT referral",
            "AZITHROMYCIN-PROPHYLAXIS: Not standard — but infection prophylaxis during low-ANC windows in cyclic neutropenia",
            "DE-NOVO-ELANE: 35% de novo — no family history — genetic counselling important for reproductive decisions",
        ],
        "key_ddx_rules": [
            "SCN without DEB positivity → ELANE first (most common SCN gene in Europeans)",
            "Cyclic pattern 14–21 days → ELANE (G214R and similar) — HAX1 is persistent (Kostmann type 1)",
            "HAX1 vs ELANE: HAX1 has neurological complications (seizures, intellectual disability) — ELANE does not",
        ],
    },
    # ── SBDS — Shwachman-Diamond Syndrome ──
    {
        "gene": "SBDS",
        "protein": "Shwachman-Bodian-Diamond Syndrome Protein (SBDS Ribosome Biogenesis Factor)",
        "alias": (
            "SBDS; OMIM gene 607444; Shwachman-Diamond Syndrome OMIM 260400; 7q11.21; 250 aa; ~28.8 kDa; "
            "SBDS cooperates with EFL1 GTPase to evict eIF6 from 60S pre-ribosomal subunit — last step 60S maturation; "
            "Exocrine pancreatic insufficiency (EPI) — fat-soluble vitamin malabsorption; "
            "BMF: neutropenia most common, then thrombocytopenia, then aplastic anaemia; "
            "Metaphyseal dysostosis — skeletal findings on X-ray; "
            "15–30% AML/MDS lifetime risk — highest malignant transformation of non-FA inherited BMF; "
            "HSCT curative for BMF — does NOT correct EPI or skeletal disease"
        ),
        "aa": "250 aa",
        "kDa": "~28.8 kDa",
        "locus": "7q11.21",
        "omim_gene": 607444,
        "omim_disease": 260400,
        "inheritance": "AR — biallelic LOF (gene conversion from SBDSP1 pseudogene common — standard NGS may miss)",
        "gene_class": (
            "SBDS encodes a ribosome biogenesis factor that, together with the GTPase EFL1, catalyses "
            "the removal of eIF6 from the 60S pre-ribosomal subunit — a critical late step in cytoplasmic "
            "60S ribosomal subunit maturation. Without functional SBDS, 60S subunits cannot enter the "
            "translationally active pool, impairing global protein synthesis in rapidly proliferating cells "
            "including haematopoietic progenitors and pancreatic acinar cells. "
            "The SBDS locus has a nearby pseudogene (SBDSP1) that shares 97% sequence identity — "
            "gene conversion events between SBDS and SBDSP1 account for the majority (~75%) of pathogenic "
            "SBDS alleles, with c.183_184TA>CT (K62X) and c.258+2T>C being the most common. "
            "Standard NGS panels may not distinguish SBDS from SBDSP1 — dedicated SBDS sequencing with "
            "gene conversion analysis is required. Clonal haematopoiesis with TP53 and EIF6 mutations "
            "predicts imminent AML/MDS transformation."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("c.183_184TA>CT (p.Lys62Xaafs*6) gene conversion — most common ~60%", 0.35),
            ("c.258+2T>C gene conversion + c.183_184TA>CT compound het", 0.25),
            ("c.258+2T>C + novel missense compound heterozygous", 0.15),
            ("Biallelic missense (non-gene-conversion region)", 0.10),
            ("Deep intronic gene conversion variant — missed by standard NGS", 0.10),
            ("SBDS deletion — MLPA required", 0.05),
        ],
        "age_onset_years_range": (0, 3),
        "sex_ratio_M": 0.52,
        "rates": {
            "exocrine_pancreatic_insufficiency": 0.90,
            "neutropenia_anc_below_1500":        0.95,
            "thrombocytopenia":                  0.65,
            "aplastic_anaemia":                  0.25,
            "metaphyseal_dysostosis":            0.75,
            "short_stature":                     0.70,
            "aml_mds_lifetime":                  0.25,
            "tp53_clonal_haematopoiesis":        0.20,
            "fat_soluble_vitamin_deficiency":    0.85,
            "fatty_liver_infancy":               0.60,
            "cognitive_impairment":              0.30,
            "gene_conversion_sbdsp1":            0.75,
        },
        "critical_alerts": [
            "PERT-MANDATORY: Pancreatic enzyme replacement therapy + fat-soluble vitamins A/D/E/K — start at diagnosis regardless of BMF status",
            "ANNUAL-MARROW-SURVEILLANCE: Bone marrow biopsy + cytogenetics + TP53 IHC annually — AML/MDS in 25%",
            "TP53-CLONAL-HAEMATOPOIESIS: TP53 acquisition in marrow → imminent AML/MDS — HSCT referral immediately",
            "GENE-CONVERSION-NGS: Standard NGS misses SBDSP1 gene conversion — order dedicated SBDS gene conversion analysis",
            "HSCT-CURES-BMF-NOT-EPI: HSCT corrects haematological disease — EPI persists — PERT continues post-HSCT",
            "SKELETAL-DYSPLASIA: Metaphyseal dysostosis — X-ray at diagnosis — orthopaedic review for progressive changes",
            "SBDS-REDUCED-CONDITIONING: Reduced-intensity HSCT preferred — standard myeloablative poorly tolerated",
        ],
        "key_ddx_rules": [
            "Exocrine pancreatic insufficiency + neutropenia in infancy → SBDS (Shwachman-Diamond) — DEB negative",
            "DEB positive → FA; DEB negative + EPI → SBDS; DEB negative + no EPI → ELANE/GATA2/telomere",
            "TP53 in marrow + SDS → immediate HSCT referral — TP53+ SDS has >50% short-term AML risk",
        ],
    },
    # ── GATA2 — GATA2 Deficiency / MonoMAC / Emberger ──
    {
        "gene": "GATA2",
        "protein": "GATA Binding Protein 2 (Zinc Finger Transcription Factor)",
        "alias": (
            "GATA2; OMIM gene 137295; GATA2 Deficiency OMIM 614738; Emberger OMIM 614038; 3q21.3; 480 aa; ~52.7 kDa; "
            "GATA2 deficiency syndrome (MonoMAC/Emberger) — monocytopenia + NK depletion + B-cell depletion + dendritic cell depletion; "
            "Monocytes <10 cells/μL in peripheral blood — pathognomonic finding (monocytopenia); "
            "Atypical mycobacterial infections, disseminated viral (CMV/EBV/HPV/molluscum), fungal infections; "
            "Lymphoedema (Emberger variant) + MDS/AML; "
            "HSCT CURATIVE — corrects all immune defects and haematological malignancy risk; "
            "MDS/AML in >80% of untreated patients — earliest malignant transformation of any inherited BMF"
        ),
        "aa": "480 aa",
        "kDa": "~52.7 kDa",
        "locus": "3q21.3",
        "omim_gene": 137295,
        "omim_disease": 614738,
        "inheritance": "AD — heterozygous LOF (haploinsufficiency); de novo variants ~30%",
        "gene_class": (
            "GATA2 encodes a zinc finger transcription factor critical for haematopoietic stem and "
            "progenitor cell maintenance and specification of the lymphoid and myeloid lineages. "
            "GATA2 haploinsufficiency causes a primary immunodeficiency-myeloid neoplasm overlap "
            "syndrome characterised by depletion of monocytes, natural killer cells, B lymphocytes, "
            "and plasmacytoid dendritic cells — collectively termed the 'GATA2 immunodeficiency'. "
            "These immune deficiencies cause susceptibility to atypical mycobacteria (MAC/MAI), "
            "disseminated HPV-related warts and dysplasia, chronic viral infections (EBV, CMV, HSV), "
            "and opportunistic fungi. Simultaneously, GATA2 haploinsufficiency predisposes to MDS "
            "and AML with monosomy 7 and inv(3)/t(3;3). The Emberger syndrome variant adds primary "
            "lymphoedema (lymphatic aplasia) to the syndrome. HSCT restores GATA2 expression and "
            "is the only curative treatment — corrects all immune and haematological complications."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("Enhancer region GATA2 variant (9.5 kb downstream) — common AD", 0.25),
            ("Exon 5 frameshift GATA2 — haploinsufficiency, severe", 0.20),
            ("Zinc finger 2 missense GATA2 (C373R, T354M) — dominant negative", 0.20),
            ("De novo GATA2 missense — no family history, variable phenotype", 0.20),
            ("GATA2 intron 5 splice variant — Emberger phenotype", 0.10),
            ("Large GATA2 deletion — MLPA required", 0.05),
        ],
        "age_onset_years_range": (8, 35),
        "sex_ratio_M": 0.45,
        "rates": {
            "monocytopenia_below_10":           0.98,
            "nk_cell_depletion":                0.95,
            "b_cell_depletion":                 0.85,
            "mds_aml_lifetime":                 0.82,
            "monosomy_7_mds":                   0.45,
            "atypical_mycobacteria_infections": 0.60,
            "hpv_warts_dysplasia":              0.55,
            "cmv_ebv_disseminated":             0.40,
            "pulmonary_alveolar_proteinosis":   0.20,
            "lymphoedema_emberger":             0.25,
            "hsct_curative":                    0.88,
            "de_novo_gata2":                    0.30,
        },
        "critical_alerts": [
            "MONOCYTOPENIA-PATHOGNOMONIC: Monocytes <10/μL — order GATA2 panel + flow cytometry (NK, B-cell, dendritic cell depletion) immediately",
            "HSCT-CURATIVE-MANDATORY: HSCT MUST be offered — MDS/AML in 82% untreated — do not delay for haematological 'stability'",
            "MAC-PROPHYLAXIS: Azithromycin prophylaxis for atypical mycobacteria while awaiting HSCT",
            "HPV-SURVEILLANCE: Annual gynaecological (females) + ENT review for HPV-related dysplasia/cancer",
            "ENHANCER-VARIANT: 9.5 kb downstream GATA2 enhancer variants — require specific testing — missed by exome-only NGS",
            "PULMONARY-ALVEOLAR-PROTEINOSIS: Chest CT + BAL if respiratory symptoms — PAP corrects post-HSCT",
            "MONOSOMY-7-URGENT: MDS with monosomy 7 detected → HSCT within 3 months — rapid AML progression",
        ],
        "key_ddx_rules": [
            "Monocytopenia <10/μL + atypical mycobacteria or HPV warts → GATA2 deficiency first",
            "MDS + monosomy 7 in young patient without prior chemotherapy → GATA2, SBDS, FA",
            "Emberger (lymphoedema + MDS) → GATA2 — dedicated enhancer region sequencing mandatory",
        ],
    },
]


def _make_patients(gene_entry):
    rng = random.Random(gene_entry["seed"])
    patients = []
    for i in range(gene_entry["n_patients"]):
        etiol_labels, etiol_weights = zip(*gene_entry["etiologies"])
        etiology = rng.choices(etiol_labels, weights=etiol_weights)[0]
        a_lo, a_hi = gene_entry["age_onset_years_range"]
        age_onset = round(rng.uniform(a_lo, a_hi), 1)
        sex = "M" if rng.random() < gene_entry["sex_ratio_M"] else "F"

        # Build phenotype flags from rates
        flags = {}
        for key, rate in gene_entry["rates"].items():
            flags[key] = rng.random() < rate

        patients.append({
            "patient_id": f"{gene_entry['gene']}-{i+1:03d}",
            "gene": gene_entry["gene"],
            "etiology": etiology,
            "age_onset_years": age_onset,
            "sex": sex,
            "phenotype_flags": flags,
        })
    return patients


def _agg(patients, key):
    vals = [p["phenotype_flags"].get(key, False) for p in patients]
    return round(100 * sum(vals) / len(vals), 1) if vals else 0.0


def get_overview():
    all_patients = []
    for g in BMF_GENES:
        all_patients.extend(_make_patients(g))

    agg = {
        "total_patients":               len(all_patients),
        "n_genes":                      len(BMF_GENES),
        "seeds":                        f"{SEED_BASE}–{SEED_BASE + len(BMF_GENES) - 1}",
        # cross-gene aggregate stats
        "bone_marrow_failure_pct":      _agg(all_patients, "bone_marrow_failure"),
        "aml_mds_lifetime_pct":         _agg(all_patients, "aml_mds_lifetime") or _agg(all_patients, "aml_mds"),
        "hsct_performed_pct":           _agg(all_patients, "hsct_performed"),
        "androgen_responsive_pct":      _agg(all_patients, "androgen_responsive"),
        "telomere_below_1pct_pct":      _agg(all_patients, "telomere_below_1st_percentile"),
        "pulmonary_fibrosis_pct":       _agg(all_patients, "pulmonary_fibrosis"),
        "neutropenia_pct":              _agg(all_patients, "neutropenia_anc_below_1500") or _agg(all_patients, "scn_anc_below_200_persistent"),
        "exocrine_pancreatic_insuff_pct": _agg(all_patients, "exocrine_pancreatic_insufficiency"),
        "monocytopenia_pct":            _agg(all_patients, "monocytopenia_below_10"),
        "deb_fragility_pct":            _agg(all_patients, "deb_test_positive") or _agg(all_patients, "deb_test_strongly_positive"),
    }

    top_alerts = [
        "ALKYLATING-AGENTS-ABSOLUTE-CI-FA: Standard myeloablative conditioning lethal in FANCA/FANCD2 — reduced-intensity MANDATORY",
        "PULMONARY-FIBROSIS-ABSOLUTE-CI-HSCT: Established PF (DKC1/TERC/TERT) → HSCT ABSOLUTELY CONTRAINDICATED — conditioning fatal",
        "DEB-TEST-BEFORE-NGS-FA: Chromosome fragility (DEB/MMC) MANDATORY FIRST in suspected FA — mosaic reversion causes false-negative blood test",
        "GATA2-MONOCYTOPENIA-PATHOGNOMONIC: Monocytes <10/μL → GATA2 panel immediately — MDS/AML in 82% untreated",
        "SBDS-GENE-CONVERSION: Standard NGS misses SBDSP1 gene conversion (75% of SBDS alleles) — dedicated gene conversion analysis mandatory",
        "ANNUAL-MARROW-ALL-INHERITED-BMF: Bone marrow biopsy + cytogenetics annually — AML/MDS risk varies 10–82% by gene",
        "FLOW-FISH-TELOMERE-DC: Telomere length <1st percentile in lymphocytes confirms telomere biology disorder (DKC1/TERC/TERT)",
        "GCSF-FIRST-LINE-ELANE: G-CSF (filgrastim) 5 mcg/kg/day first-line for ELANE-SCN — >97% respond",
    ]

    return {
        "atlas": "Hereditary-BMF-Atlas",
        "title": "Complete 8-Gene Hereditary Bone Marrow Failure Syndromes Atlas",
        "genes": [g["gene"] for g in BMF_GENES],
        "diseases": [g["omim_disease"] for g in BMF_GENES],
        "aggregate_stats": agg,
        "top_alerts": top_alerts,
        "registered": "2026-09-05",
    }


def get_breakdown():
    result = {}
    for g in BMF_GENES:
        patients = _make_patients(g)
        result[g["gene"]] = {
            "gene":        g["gene"],
            "protein":     g["protein"],
            "locus":       g["locus"],
            "aa":          g["aa"],
            "inheritance": g["inheritance"],
            "omim_gene":   g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "n_patients":  g["n_patients"],
            "seed":        g["seed"],
            "alias":       g["alias"],
            "gene_class":  g["gene_class"],
            "critical_alerts": g["critical_alerts"],
            "key_ddx_rules":   g["key_ddx_rules"],
            "phenotype_rates": {k: round(v * 100, 1) for k, v in g["rates"].items()},
            "etiologies":  [{"label": l, "pct": round(w * 100, 1)} for l, w in g["etiologies"]],
            "aggregate":   {k: _agg(patients, k) for k in g["rates"]},
        }
    return result


def get_definitions():
    return {
        "atlas": "Hereditary-BMF-Atlas",
        "definitions": [
            {
                "term": "FA (Fanconi Anaemia)",
                "definition": (
                    "Autosomal recessive (rarely X-linked) inherited BMF syndrome caused by biallelic LOF in any of ≥22 FA complementation genes. "
                    "All FA genes encode components of the FA-BRCA DNA repair pathway for interstrand crosslinks. "
                    "Diagnosis: DEB/MMC chromosome fragility test (radial chromosomes). "
                    "Management: reduced-intensity HSCT; avoid alkylators and radiation."
                )
            },
            {
                "term": "DEB/MMC Chromosome Fragility Test",
                "definition": (
                    "Gold-standard test for Fanconi anaemia. Lymphocytes exposed to diepoxybutane (DEB) or mitomycin C (MMC) in vitro. "
                    "FA cells show increased chromosomal breaks, radial figures, and quadriradials — not seen in other BMF syndromes. "
                    "MUST be performed before NGS panel — mosaic somatic reversion can cause false-negative blood DEB."
                )
            },
            {
                "term": "Telomere Biology Disorder (TBD)",
                "definition": (
                    "Group of disorders caused by short telomeres or impaired telomere maintenance: DC (DKC1, TERC, TERT, NHP2, NOP10, WRAP53), "
                    "Hoyeraal-Hreidarsson, Revesz syndrome, cryptogenic cirrhosis, idiopathic pulmonary fibrosis. "
                    "Diagnosed by flow-FISH telomere length <1st percentile in lymphocytes. "
                    "DEB test NEGATIVE — distinguishes from FA."
                )
            },
            {
                "term": "Flow-FISH Telomere Length",
                "definition": (
                    "Fluorescence in situ hybridisation combined with flow cytometry measuring telomere length in specific lymphocyte subsets. "
                    "Lymphocyte telomere length <1st percentile for age is diagnostic of a telomere biology disorder. "
                    "Essential for diagnosing DKC1, TERC, TERT, and other TBD genes."
                )
            },
            {
                "term": "Anticipation in TERC",
                "definition": (
                    "Progressive worsening of disease severity across generations in TERC-related DC. "
                    "Each generation inherits shorter telomeres (already shortened by parent's TERC haploinsufficiency), "
                    "so the critical telomere threshold is crossed earlier — grandparents: aplastic anaemia only; "
                    "parents: moderate DC; grandchildren: severe DC or Hoyeraal-Hreidarsson. "
                    "Family history must be taken with awareness of anticipation."
                )
            },
            {
                "term": "MonoMAC / GATA2 Deficiency",
                "definition": (
                    "GATA2 deficiency syndrome — monocytopenia + NK depletion + B-cell depletion + plasmacytoid dendritic cell depletion. "
                    "Classic presenting features: atypical mycobacteria (MAC), disseminated HPV (warts, dysplasia), CMV/EBV. "
                    "High risk MDS/AML (monosomy 7). HSCT curative — should not be withheld pending 'haematological stability'."
                )
            },
            {
                "term": "Pulmonary Fibrosis — Absolute CI to HSCT",
                "definition": (
                    "In telomere biology disorders (DKC1, TERC, TERT), established pulmonary fibrosis is an ABSOLUTE CONTRAINDICATION "
                    "to HSCT. Conditioning regimens (especially busulfan, cyclophosphamide, radiation) cause fatal pulmonary toxicity "
                    "in patients with pre-existing PF. Annual pulmonary function testing (DLCO) mandatory — HSCT must be performed "
                    "before PF is established."
                )
            },
            {
                "term": "Reduced-Intensity Conditioning (RIC) — FA Mandatory",
                "definition": (
                    "Standard myeloablative conditioning (busulfan, cyclophosphamide, TBI) is LETHAL in FA patients due to "
                    "defective DNA repair — alkylator and radiation sensitivity extreme. RIC regimens (fludarabine-based) are "
                    "MANDATORY. Even with RIC, outcomes are poorer with mismatched donors — matched sibling preferred."
                )
            },
            {
                "term": "SBDS Gene Conversion",
                "definition": (
                    "75% of SBDS pathogenic alleles result from gene conversion between SBDS and its nearby pseudogene SBDSP1 "
                    "(97% sequence identity). The most common alleles are c.183_184TA>CT (K62X) and c.258+2T>C. "
                    "Standard clinical NGS panels may not reliably distinguish SBDS from SBDSP1 — "
                    "dedicated SBDS gene conversion analysis (long-range PCR or specific allele sequencing) is MANDATORY."
                )
            },
            {
                "term": "ELANE SCN vs Cyclic Neutropenia",
                "definition": (
                    "ELANE mutations cause two distinct phenotypes: SCN (persistent ANC <200, severe infections) and "
                    "cyclic neutropenia (ANC nadir <200 every 14–21 days, milder infections). "
                    "Classification requires twice-weekly ANC measurements for 6–8 weeks. "
                    "AML/MDS risk (10–15%) exists in SCN but NOT in CN — annual marrow surveillance mandatory for SCN only."
                )
            },
            {
                "term": "Complementation Group Testing (FA)",
                "definition": (
                    "Functional assay where patient FA cells are corrected (complemented) by transducing individual FA gene cDNAs. "
                    "The gene that restores DEB resistance identifies the patient's FA complementation group. "
                    "Essential for: (1) identifying gene for mutation analysis; (2) clinical trial eligibility; "
                    "(3) prognosis (FANCD2 = more severe than FANCA)."
                )
            },
            {
                "term": "Cascade Testing — Inherited BMF",
                "definition": (
                    "All first-degree relatives of an inherited BMF proband should be offered appropriate testing: "
                    "FA: DEB test + gene testing; DKC1: telomere flow-FISH; TERC/TERT: gene testing + flow-FISH; "
                    "GATA2: monocyte count + GATA2 sequencing; SBDS: SBDS gene conversion analysis; ELANE: gene testing. "
                    "Carrier identification important for reproductive planning and early disease detection."
                )
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== BREAKDOWN (first gene) ===")
    bd = get_breakdown()
    print(json.dumps(bd[BMF_GENES[0]["gene"]], indent=2))
    print("\n=== DEFINITIONS (first 2) ===")
    defs = get_definitions()
    print(json.dumps(defs["definitions"][:2], indent=2))
