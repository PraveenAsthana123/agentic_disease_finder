#!/usr/bin/env python3
"""Hereditary-SCN-Granulocyte-Cytotoxic-Granule-Defect-Atlas -- Complete 8-Gene Reference
Severe Congenital Neutropenia, Wiskott-Aldrich, WHIM, Chediak-Higashi, Griscelli-2, HPS-2
(ELANE/SCN-I-Cyclic-Neutropenia-3wk-PATHOGNOMONIC-AML-Risk-15pct-G-CSF-HSCT,
 HAX1/SCN-III-Kostmann-Neurodegeneration-RS-HAX1-G-CSF-HSCT,
 G6PC3/SCN-IV-Venous-Angiectasias-PATHOGNOMONIC-Congenital-Heart,
 WASP/Wiskott-Aldrich-Microthrombocytes-PATHOGNOMONIC-XLN-AD-GOF,
 CXCR4/WHIM-Myelokathexis-PATHOGNOMONIC-Plerixafor-AMD3100,
 LYST/Chediak-Higashi-Giant-Granules-PBS-PATHOGNOMONIC-NK-Absent-HLH-Accelerated,
 RAB27A/Griscelli-2-Silvery-Hair-PATHOGNOMONIC-HLH-No-Neurological,
 AP3B1/HPS-2-Oculocutaneous-Albinism-Neutropenia-HLH-Bleeding-Diathesis):

ELANE  (Neutrophil elastase; 218aa; 19p13.3; AD-LOF;
         ELANE encodes neutrophil elastase, a serine protease stored in primary (azurophil) granules;
         AD-LOF mutations cause misfolded elastase that triggers unfolded protein response (UPR) in
           maturing myeloid cells → apoptosis at promyelocyte stage → profound neutropenia;
         TWO DISTINCT PHENOTYPES from same gene:
           (1) SEVERE CONGENITAL NEUTROPENIA TYPE 1 (SCN-I; ~60% of SCN): constant severe
               neutropenia (ANC <200/µL); recurrent bacterial infections from birth; omphalitis,
               umbilical infection; skin/mucous membrane infections; AML/MDS risk 15-20% lifetime;
               G-CSF (filgrastim) first-line; HSCT for G-CSF failure or transformation;
           (2) CYCLIC NEUTROPENIA (CyN): ANC oscillates on a 3-WEEK CYCLE — troughs to <200/µL
               for 3-6 days then recovers to near-normal; synchronised cycling;
               3-WEEK CYCLING PERIOD PATHOGNOMONIC — distinguishes from SCN (constant neutropenia);
               clinical periodicity: mouth ulcers, pharyngitis, lymphadenopathy cycling every 21 days;
               G-CSF low-dose shortens cycling period and raises nadir; HSCT rarely needed for CyN;
         KEY DISTINCTION: SCN1 (constant neutropenia, high AML risk) vs CyN (cycling, AML rare);
         GENOTYPE HELPS: p.G185R → SCN; p.S97L → cyclic; V72M → both; but overlap exists;
         G-CSF RESPONSE: ~90% of SCN-I respond to filgrastim but response requires high doses;
           AML SURVEILLANCE: bone marrow annually; clonal cytogenetics; CSF3R mutation monitoring;
         seed SEED_BASE+0).

HAX1   (HCLS1-associated protein X-1; 279aa; 1q21.3; AR-LOF;
         HAX1 encodes a mitochondrial anti-apoptotic protein maintaining mitochondrial membrane
           potential; HAX1 LOF → increased apoptosis of myeloid progenitors → neutropenia;
         ORIGINAL KOSTMANN DISEASE (SCN-III): classic AD autosomal recessive pedigree originally
           described by Kostmann in 1956 in a Swedish consanguineous family; HAX1 identified 2007;
         TWO VARIANTS:
           (1) HAX1 isoform 1-only mutations: neutropenia ONLY; no neurological features; milder;
           (2) HAX1 RS VARIANT (affects isoforms 1+2): severe neutropenia + NEURODEGENERATION —
               epilepsy, intellectual disability, psychomotor regression;
               RS VARIANT PATHOGNOMONIC DISTINCTION: HAX1 biallelic + neurological = RS variant;
               implies brain HAX1 isoform-2 function beyond myeloid apoptosis;
         AML RISK: similar to SCN-I (~15-20% lifetime); G-CSF responsive; HSCT indicated for
           G-CSF failure, transformation, or severe RS-variant neurodegeneration;
         DIAGNOSIS: ANC <500/µL; bone marrow = maturation arrest at promyelocyte; HAX1 sequencing;
         seed SEED_BASE+1).

G6PC3  (Glucose-6-phosphatase catalytic subunit 3; 346aa; 17q21.31; AR-LOF;
         G6PC3 encodes glucose-6-phosphatase subunit 3, localised to endoplasmic reticulum (ER);
         G6PC3 LOF → ER stress → excessive apoptosis of myeloid cells → neutropenia;
         SEVERE CONGENITAL NEUTROPENIA TYPE 4 (SCN-IV; Dürr-Boztug syndrome):
         PATHOGNOMONIC CLINICAL FEATURE: PROMINENT SUPERFICIAL VENOUS ANGIECTASIAS of trunk/
           extremities in majority of patients — subcutaneous venous malformations visible on skin;
           PATHOGNOMONIC COMBINATION: SCN + visible venous angiectasias = G6PC3 mutation until proven;
         CARDIAC: congenital heart defects in ~50% (VSD, ASD, pulmonary valve stenosis);
         UROGENITAL: urogenital malformations (cryptorchidism, horseshoe kidney);
         METABOLIC: intermittent hypoglycaemia (G6Pase-related), inflammatory bowel disease;
         AML RISK: similar to ELANE/HAX1 (~15%); G-CSF responsive; HSCT for transformation;
         DIAGNOSIS: ANC <500/µL; venous angiectasias on skin examination; G6PC3 sequencing;
           distinguish from SCN types 1-3 by skin vascular findings + cardiac malformations;
         seed SEED_BASE+2).

WASP   (Wiskott-Aldrich syndrome protein; 502aa; Xp11.22; XLR-LOF / AD-GOF;
         WASP encodes a regulator of actin polymerisation in haematopoietic cells;
         TWO DISTINCT PHENOTYPES from same gene (LOF vs GOF):
           (1) WISKOTT-ALDRICH SYNDROME (WAS; XLR-LOF): classic triad of:
               a) THROMBOCYTOPAENIA with MICROPLATELET (mean platelet volume MPV <7 fL);
                  MICROTHROMBOPAENIA PATHOGNOMONIC: platelet count 20,000-50,000 + SMALL PLATELETS
                  (normal MPV 7.5-12 fL) — the COMBINATION of low count + tiny size is hallmark;
               b) ECZEMA (atopic dermatitis); c) RECURRENT INFECTIONS (encapsulated bacteria,
                  pneumocystis, herpes, CMV); autoimmune complications (haemolytic anaemia, nephritis);
                  EBV-associated lymphoma; WISCOTT: Wiskott = IgM low but IgA/IgE elevated;
               WASP protein: absent in severe WAS; reduced in X-linked thrombocytopaenia (XLT);
               HSCT CURATIVE for WAS: matched sibling BEST; gene therapy clinical trials;
               Splenectomy raises platelet count but increases infective risk — USE WITH CAUTION;
           (2) X-LINKED NEUTROPENIA (XLN; AD-GOF): gain-of-function WASP (L270P, I294T, etc.)
               → constitutive actin polymerisation → myeloid maturation block → neutropenia;
               XLN: isolated neutropenia (NO thrombocytopaenia, NO eczema);
               monocytopenia + NK-cell lymphopenia also in XLN;
               DISTINCTION: WAS=LOF=thrombocytopaenia+eczema; XLN=GOF=neutropenia alone;
         DIAGNOSIS: WAS protein expression by flow cytometry (absent in WAS, reduced in XLT,
           normal or abnormal isoform in XLN); WASP sequencing;
         seed SEED_BASE+3).

CXCR4  (C-X-C motif chemokine receptor 4; 360aa; 2q22.1; AD-GOF;
         CXCR4 encodes the receptor for CXCL12/SDF-1; regulates haematopoietic progenitor retention
           in bone marrow; AD-GOF mutations truncate C-terminal regulatory tail → impaired receptor
           internalisation → constitutive prolonged signalling → neutrophils trapped in bone marrow;
         WHIM SYNDROME (Warts-Hypogammaglobulinaemia-Infections-Myelokathexis):
         PATHOGNOMONIC TETRAD (4 'W's):
           (1) WARTS: severe, treatment-refractory HPV warts (common, plantar, genital) — often
               extensive; reflects impaired anti-viral immunity despite normal neutrophil numbers;
           (2) HYPOGAMMAGLOBULINAEMIA: impaired B-cell egress from bone marrow;
           (3) INFECTIONS: recurrent sinopulmonary infections;
           (4) MYELOKATHEXIS: PATHOGNOMONIC bone marrow finding — neutrophils have hypersegmented
               nuclei with thin strands connecting lobes (caterpillar nuclei) + hyperpycnotic nuclei
               + cellular retention in marrow despite peripheral neutropenia;
         PLERIXAFOR (AMD3100): CXCR4 antagonist — CURATIVE mechanism; mobilises neutrophils and
           lymphocytes from bone marrow → normalises counts; FDA-approved for WBC mobilisation;
           used continuously in WHIM; dramatically improves wart clearance and infection frequency;
         GENOTYPIC HALLMARK: AD-GOF typically truncation of last 10-19 aa of C-terminus;
           most common mutation: p.R334X (European); haploinsufficiency model of GOF;
         DIAGNOSIS: WHIM tetrad; bone marrow showing myelokathexis; CXCR4 sequencing;
         seed SEED_BASE+4).

LYST   (Lysosomal trafficking regulator; 3801aa; 1q42.3; AR-LOF;
         LYST encodes a large scaffold protein regulating lysosome-related organelle size and fission;
         LYST LOF → GIANT LYSOSOME-RELATED ORGANELLES — primary granules in neutrophils, melanosomes
           in melanocytes, dense granules in platelets → all appear enlarged;
         CHEDIAK-HIGASHI SYNDROME (CHS):
         PATHOGNOMONIC FEATURE: GIANT AZUROPHIL GRANULES IN NEUTROPHILS ON PERIPHERAL BLOOD SMEAR —
           the most visible haematological pathognomonic sign; giant granules also in eosinophils,
           basophils, monocytes, lymphocytes;
         CLINICAL MANIFESTATIONS:
           (1) PARTIAL OCULOCUTANEOUS ALBINISM: diluted hair (silvery/steel-grey sheen), hypopigmented
               skin, photophobia, nystagmus; HAIR SILVER SHEEN pathognomonic combined with giant granules;
           (2) RECURRENT BACTERIAL INFECTIONS: neutrophil function impaired despite adequate numbers;
               neutrophil granule defect → impaired killing even with normal respiratory burst;
           (3) NK CELL ABSENT/NON-FUNCTIONAL: absent cytotoxic granule exocytosis → susceptibility
               to Epstein-Barr virus (EBV) → ACCELERATED PHASE;
         ACCELERATED PHASE (HLH-like): EBV-triggered haemophagocytic lymphohistiocytosis (HLH);
           fever, cytopaenias, haemophagocytosis in BM, hyperferritinaemia; FATAL without treatment;
           NK CELL ABSENT + ACCELERATED PHASE = LYST until proven otherwise;
         TREATMENT: HSCT before accelerated phase (only cure); HLH-directed therapy for accelerated phase;
         DIAGNOSIS: PBS giant granules (PATHOGNOMONIC); electron microscopy granules; LYST sequencing;
         seed SEED_BASE+5).

RAB27A (Ras-related protein Rab-27A; 221aa; 15q21.3; AR-LOF;
         RAB27A encodes a small GTPase regulating vesicle docking and exocytosis of secretory lysosomes;
         RAB27A LOF → melanosome transport defect in melanocytes + cytotoxic granule exocytosis defect
           in NK/CTL cells;
         GRISCELLI SYNDROME TYPE 2 (GS2; immunological variant):
         PATHOGNOMONIC SILVERY HAIR: irregular large melanin clumps visible on light microscopy of hair
           shaft — SILVERY/GREY HAIR WITH LARGE IRREGULAR CLUMPS = GS2 pathognomonic;
           DISTINCT FROM CHS: hair silvery in both; PBS giant granules in CHS but NOT in GS2;
           hair shaft microscopy: large irregular clumps = GS2/RAB27A; giant granules on PBS = LYST/CHS;
         NK/CTL DEFECT DOMINANT: defective cytotoxic granule secretion → impaired NK and CTL killing;
           HLH in >80% of GS2 patients (EBV-triggered or spontaneous);
         NO NEUROLOGICAL FEATURES (distinguishes from GS1/MYO5A which has neurological features);
           GS2 RULE: silvery hair + HLH (+ NO neurological) = RAB27A;
           GS1 RULE: silvery hair + neurological (NO HLH usually) = MYO5A;
           GS3 RULE: albinism + bleeding diathesis (NO neurological, NO immune) = MLPH;
         TREATMENT: HSCT (curative for immune component; silvery hair persists);
           HLH-directed therapy (dexamethasone + etoposide ± cyclosporine, HLH-2004 protocol);
         DIAGNOSIS: silvery hair + HLH + hair shaft microscopy (irregular clumps) + RAB27A sequencing;
         seed SEED_BASE+6).

AP3B1  (Adaptor-related protein complex 3 subunit beta-1; 1094aa; 5q14.1; AR-LOF;
         AP3B1 encodes the beta subunit of adaptor protein complex 3 (AP-3), required for biogenesis
           of lysosome-related organelles (LROs) including melanosomes, dense granules, Weibel-Palade
           bodies, and lytic granules;
         HERMANSKY-PUDLAK SYNDROME TYPE 2 (HPS-2): the IMMUNODEFICIENCY subtype of HPS;
         TRIAD OF HPS-2:
           (1) OCULOCUTANEOUS ALBINISM (OCA): hypopigmentation of skin, hair, eyes; nystagmus; reduced
               visual acuity; foveal hypoplasia; similar to other HPS types;
           (2) BLEEDING DIATHESIS: absent platelet dense granules → impaired platelet aggregation
               (ADP, collagen response reduced) → prolonged bleeding time; epistaxis, bruising;
               ABSENT PLATELET DENSE GRANULES on electron microscopy PATHOGNOMONIC for ALL HPS;
           (3) IMMUNE DEFECT (UNIQUE TO HPS-2): neutropenia + defective NK cell granule exocytosis
               → SUSCEPTIBILITY TO HLH; this differentiates HPS-2 from HPS-1/HPS-4 which have
               lung fibrosis (not HLH/immune defect);
         HPS-2 vs HPS-1/4: PULMONARY FIBROSIS rare in HPS-2 (unlike HPS-1 commonest cause death);
           instead HPS-2 → neutropenia + HLH risk (like RAB27A/GS2 but ALSO has albinism + bleeding);
         AP3B1 UNIQUE: also affects dendritic cell MHC-II trafficking → adaptive immune defects;
         DIAGNOSIS: OCA + neutropenia + bleeding + absent platelet dense granules on EM;
           AP3B1 sequencing; distinguish from HPS-1 (HSPA8); immunological workup for NK function;
         seed SEED_BASE+7).
"""
import random

SEED_BASE = 4166
GENES = ["ELANE", "HAX1", "G6PC3", "WASP", "CXCR4", "LYST", "RAB27A", "AP3B1"]

GENE_META = {
    "ELANE": {
        "full_name": "Neutrophil elastase (ELANE)",
        "aa": 218, "kda": 25, "locus": "19p13.3", "inheritance": "AD",
        "disease": "Severe Congenital Neutropenia Type 1 / Cyclic Neutropenia (SCN-I / CyN)",
        "omim_gene": "130130", "omim_disease": "202700",
        "nbs": False,
        "pathognomonic": "SCN-I: constant ANC <200/µL from birth; CyN: 21-DAY CYCLING PERIOD PATHOGNOMONIC (ANC oscillates every 3 weeks; aphthous ulcers/pharyngitis cycle synchronised); AML risk 15% in SCN-I",
        "key_drug": "G-CSF (filgrastim) 5 µg/kg/day; escalate to 20-100 µg/kg for SCN-I; HSCT for G-CSF failure or MDS/AML transformation; annual BM surveillance",
        "precision_therapy": "G-CSF first-line (90% response); HSCT for transformation; gene therapy trials (EVI1 genotoxicity risk reduced with SIN lentivirus)",
        "avoid": "Delay in G-CSF initiation (omphalitis sepsis risk); HSCT for CyN (rarely needed); skip annual BM surveillance (misses MDS/AML)",
    },
    "HAX1": {
        "full_name": "HCLS1-associated protein X-1 (HAX1)",
        "aa": 279, "kda": 32, "locus": "1q21.3", "inheritance": "AR",
        "disease": "Severe Congenital Neutropenia Type 3 (SCN-III; Kostmann Disease; HAX1-RS variant with neurodegeneration)",
        "omim_gene": "605998", "omim_disease": "610738",
        "nbs": False,
        "pathognomonic": "HAX1 RS VARIANT: profound neutropenia + PROGRESSIVE NEURODEGENERATION (epilepsy, intellectual disability, psychomotor regression) in patients with mutations affecting BOTH isoforms 1+2; isoform-1-only mutations: neutropenia WITHOUT neurological; AML risk ~15-20%",
        "key_drug": "G-CSF (filgrastim) responsive; HSCT indicated for G-CSF failure, MDS/AML, or severe RS-variant neurodegeneration; antiepileptics for seizures in RS variant",
        "precision_therapy": "G-CSF; HSCT (curative for haematological component; neurodegeneration may not improve post-HSCT in RS variant); gene therapy trials",
        "avoid": "HSCT delay in RS variant (neurodegeneration progresses); assume all HAX1 biallelic have neurological (isoform-1-only do NOT); skip isoform characterisation",
    },
    "G6PC3": {
        "full_name": "Glucose-6-phosphatase catalytic subunit 3 (G6PC3)",
        "aa": 346, "kda": 39, "locus": "17q21.31", "inheritance": "AR",
        "disease": "Severe Congenital Neutropenia Type 4 (SCN-IV; Dürr-Boztug Syndrome)",
        "omim_gene": "611045", "omim_disease": "612541",
        "nbs": False,
        "pathognomonic": "PROMINENT SUPERFICIAL VENOUS ANGIECTASIAS of trunk and extremities PATHOGNOMONIC for G6PC3-SCN; visible subcutaneous venous malformations on skin examination; + congenital heart defects in ~50%; G6PC3-SCN is the only SCN subtype with consistent vascular skin findings",
        "key_drug": "G-CSF (filgrastim) responsive; cardiac surgery for congenital heart defects; HSCT for transformation or G-CSF failure; monitor renal/urogenital malformations",
        "precision_therapy": "G-CSF; HSCT for MDS/AML; cardiac evaluation mandatory (echo at diagnosis); renal imaging (horseshoe kidney); intermittent glucose monitoring",
        "avoid": "Miss venous angiectasias on skin exam (pathognomonic diagnostic clue); skip cardiac echo at diagnosis; delay congenital heart defect management",
    },
    "WASP": {
        "full_name": "Wiskott-Aldrich syndrome protein (WASP)",
        "aa": 502, "kda": 57, "locus": "Xp11.22", "inheritance": "XLR/AD",
        "disease": "Wiskott-Aldrich Syndrome (WAS; XLR-LOF) / X-Linked Neutropenia (XLN; AD-GOF)",
        "omim_gene": "300392", "omim_disease": "301000",
        "nbs": False,
        "pathognomonic": "WAS: THROMBOCYTOPAENIA + MICROPLATELETS (MPV <7 fL) PATHOGNOMONIC — platelet count 20-50k with VERY SMALL platelet size; WAS protein absent on flow cytometry; eczema + recurrent infections complete classic triad; XLN: isolated neutropenia ONLY (GOF)",
        "key_drug": "WAS: HSCT (curative); IVIg for hypogammaglobulinaemia; trimethoprim-sulfamethoxazole PCP prophylaxis; platelet transfusion only for life-threatening bleed (splenectomy last resort); XLN: G-CSF if severe",
        "precision_therapy": "HSCT best curative option for WAS (matched sibling or 10/10 MUD); gene therapy (γ-retroviral: oncogenesis risk; lentiviral SIN vectors: safer); WASP protein expression by flow cytometry predicts severity",
        "avoid": "Splenectomy first-line in WAS (increases severe infective risk — use ONLY if HSCT impossible); live vaccines pre-HSCT; eltrombopag (thrombopoietin receptor agonist — may worsen inflammation)",
    },
    "CXCR4": {
        "full_name": "C-X-C motif chemokine receptor 4 (CXCR4)",
        "aa": 360, "kda": 41, "locus": "2q22.1", "inheritance": "AD",
        "disease": "WHIM Syndrome (Warts-Hypogammaglobulinaemia-Infections-Myelokathexis)",
        "omim_gene": "162643", "omim_disease": "193670",
        "nbs": False,
        "pathognomonic": "MYELOKATHEXIS ON BONE MARROW BIOPSY PATHOGNOMONIC: neutrophils with hypersegmented nuclei connected by thin nuclear strands + hyperpycnotic nuclei + cellular retention in marrow despite peripheral neutropenia; + refractory HPV warts + hypogammaglobulinaemia = WHIM tetrad",
        "key_drug": "Plerixafor (AMD3100) 0.02-0.08 mg/kg SC daily — CXCR4 antagonist mobilises haematopoietic cells from marrow; FDA-approved for WHIM (2024); dramatically improves neutrophil count + wart clearance; G-CSF alternative",
        "precision_therapy": "Plerixafor (AMD3100) — FDA approved for WHIM 2024 as Mozobil; mechanism: CXCR4 antagonism reverses the GOF pathomechanism directly; IVIG for hypogammaglobulinaemia",
        "avoid": "G-CSF monotherapy as sole long-term treatment (plerixafor superior for warts/immunoglobulin); miss bone marrow examination (myelokathexis diagnostic); delay HPV vaccination in unaffected family members",
    },
    "LYST": {
        "full_name": "Lysosomal trafficking regulator (LYST/CHS1)",
        "aa": 3801, "kda": 430, "locus": "1q42.3", "inheritance": "AR",
        "disease": "Chediak-Higashi Syndrome (CHS)",
        "omim_gene": "606897", "omim_disease": "214500",
        "nbs": False,
        "pathognomonic": "GIANT AZUROPHIL GRANULES IN NEUTROPHILS ON PERIPHERAL BLOOD SMEAR PATHOGNOMONIC — visible on standard Giemsa PBS without EM; also in eosinophils/monocytes/lymphocytes; + partial oculocutaneous albinism (silvery-grey hair); NK-CELL ABSENT/DYSFUNCTIONAL; ACCELERATED PHASE (EBV-HLH) in >80% without HSCT",
        "key_drug": "HSCT before accelerated phase (ONLY cure for immune defect; hair colour not restored); Accelerated phase: HLH-2004 protocol (dexamethasone 10 mg/m²/day + etoposide 150 mg/m² biweekly ± cyclosporine); EBV surveillance",
        "precision_therapy": "HSCT (allogeneic, best before accelerated phase develops); gene therapy trials (LYST gene very large — AAV/lentiviral vector challenges)",
        "avoid": "Delay HSCT until accelerated phase (far worse outcome post-HLH); miss PBS giant granules (easiest diagnostic test); assume NK function normal (it is absent in CHS)",
    },
    "RAB27A": {
        "full_name": "Ras-related protein Rab-27A (RAB27A)",
        "aa": 221, "kda": 25, "locus": "15q21.3", "inheritance": "AR",
        "disease": "Griscelli Syndrome Type 2 (GS2; Immunological Variant)",
        "omim_gene": "603868", "omim_disease": "607624",
        "nbs": False,
        "pathognomonic": "SILVERY HAIR WITH LARGE IRREGULAR MELANIN CLUMPS ON HAIR SHAFT MICROSCOPY PATHOGNOMONIC; HLH in >80% (often EBV-triggered); KEY DISTINCTION FROM CHS: NO giant granules on PBS in GS2; KEY DISTINCTION FROM GS1 (MYO5A): NO neurological features in GS2; silvery hair + HLH + no PBS giant granules + no neurological = RAB27A",
        "key_drug": "HSCT curative for immune component (HLH risk abolished); HLH-2004 protocol for active HLH; dexamethasone + etoposide ± cyclosporine; EBV pre-emptive therapy; IVIG",
        "precision_therapy": "HSCT (curative; silvery hair persists post-HSCT as melanocyte defect unaffected by donor marrow); HLH protocol; EBV monitoring and rituximab pre-emptively",
        "avoid": "Confuse GS2 with CHS (both have silvery hair — check PBS for giant granules: absent in GS2); confuse with GS1 (check neurology; GS1 has neurological, GS2 does not); delay HSCT after HLH resolution",
    },
    "AP3B1": {
        "full_name": "Adaptor-related protein complex 3 subunit beta-1 (AP3B1)",
        "aa": 1094, "kda": 121, "locus": "5q14.1", "inheritance": "AR",
        "disease": "Hermansky-Pudlak Syndrome Type 2 (HPS-2)",
        "omim_gene": "603402", "omim_disease": "608233",
        "nbs": False,
        "pathognomonic": "ABSENT PLATELET DENSE GRANULES ON ELECTRON MICROSCOPY PATHOGNOMONIC (universal to all HPS types); HPS-2 UNIQUE: NEUTROPENIA + NK-CELL DEFECT + HLH SUSCEPTIBILITY (other HPS types do NOT have immune defect; HPS-1 has pulmonary fibrosis instead); OCA (oculocutaneous albinism) + bleeding + neutropenia = HPS-2",
        "key_drug": "No curative therapy without HSCT; HSCT for life-threatening HLH or infections; platelet transfusion for surgical bleeding (desmopressin may help); avoid drugs impairing platelet function; lung disease monitoring (less severe than HPS-1)",
        "precision_therapy": "HSCT for immune complications; DDAVP/desmopressin for bleeding; HPS Network multi-disciplinary follow-up; pulmonary function testing annually (HPS-2 lung disease milder than HPS-1)",
        "avoid": "Miss OCA + bleeding + neutropenia combination (should prompt HPS-2 consideration); assume all HPS has pulmonary fibrosis (HPS-2 has immune/HLH instead); confuse with HPS-1 (HPS-1: pulmonary fibrosis, no immune defect)",
    },
}


def _rng(seed):
    return random.Random(seed)


def _aggregate_gene(gene, idx):
    rng = _rng(SEED_BASE + idx)
    meta = GENE_META[gene]
    n = 40
    # sex ratio — WASP is X-linked (males only affected in WAS/XLN LOF)
    if meta["inheritance"] in ("XLR", "XLR/AD"):
        male_n = int(n * 0.92 + rng.uniform(-2, 2))
    elif meta["inheritance"] == "AD":
        male_n = int(n * 0.50 + rng.uniform(-3, 3))
    else:
        male_n = int(n * 0.50 + rng.uniform(-3, 3))
    male_n = max(10, min(n - 5, male_n))
    female_n = n - male_n

    # onset (months) by disease
    onset_map = {
        "ELANE":  (1, 12),
        "HAX1":   (1, 12),
        "G6PC3":  (1, 12),
        "WASP":   (3, 18),
        "CXCR4":  (12, 60),
        "LYST":   (3, 24),
        "RAB27A": (6, 24),
        "AP3B1":  (1, 12),
    }
    lo, hi = onset_map.get(gene, (6, 24))
    ages = [rng.uniform(lo, hi) for _ in range(n)]
    median_onset = sorted(ages)[n // 2]

    # treatment response (G-CSF or equivalent first-line)
    resp_map = {
        "ELANE":  0.88,
        "HAX1":   0.85,
        "G6PC3":  0.82,
        "WASP":   0.70,   # HSCT curative
        "CXCR4":  0.90,   # plerixafor very effective
        "LYST":   0.60,   # HSCT before accelerated phase
        "RAB27A": 0.65,
        "AP3B1":  0.50,
    }
    tx_response_pct = resp_map.get(gene, 0.65) + rng.uniform(-0.05, 0.05)

    # HSCT rate
    hsct_map = {
        "ELANE":  0.20,
        "HAX1":   0.30,
        "G6PC3":  0.20,
        "WASP":   0.60,
        "CXCR4":  0.05,
        "LYST":   0.65,
        "RAB27A": 0.70,
        "AP3B1":  0.15,
    }
    hsct_pct = hsct_map.get(gene, 0.15) + rng.uniform(-0.03, 0.03)

    # AML/HLH transformation events
    transform_map = {
        "ELANE":  0.15,
        "HAX1":   0.18,
        "G6PC3":  0.12,
        "WASP":   0.05,
        "CXCR4":  0.02,
        "LYST":   0.55,   # accelerated phase HLH very common
        "RAB27A": 0.60,
        "AP3B1":  0.20,
    }
    transform_pct = transform_map.get(gene, 0.10) + rng.uniform(-0.03, 0.03)

    # Etiology classes
    etiology_map = {
        "ELANE": [
            "ELANE-SCN1-missense-p.G185R-cyclic",
            "ELANE-SCN1-nonsense-p.V72M-severe",
            "ELANE-CyN-p.S97L-cyclic-phenotype",
            "ELANE-SCN1-splice-site-intron3",
            "ELANE-SCN1-compound-heterozygous-biallelic-AD",
        ],
        "HAX1": [
            "HAX1-isoform1-2-biallelic-RS-variant-neurological",
            "HAX1-isoform1-only-biallelic-neutropenia-only",
            "HAX1-homozygous-c.256C>T-Kostmann-original",
            "HAX1-compound-heterozygous-splice-frameshift",
            "HAX1-large-deletion-exon-rearrangement",
        ],
        "G6PC3": [
            "G6PC3-homozygous-p.L220P-founder",
            "G6PC3-compound-heterozygous-missense",
            "G6PC3-splice-site-exon6",
            "G6PC3-frameshift-truncating",
            "G6PC3-deletion-exon-encompassing",
        ],
        "WASP": [
            "WASP-LOF-nonsense-XLR-classic-WAS-absent-protein",
            "WASP-LOF-missense-XLR-XLT-mild-reduced-protein",
            "WASP-GOF-p.L270P-XLN-neutropenia-only",
            "WASP-GOF-p.I294T-XLN-gain-of-function",
            "WASP-LOF-splice-site-XLR-classic-WAS",
        ],
        "CXCR4": [
            "CXCR4-p.R334X-European-founder-truncation",
            "CXCR4-p.E343X-C-terminal-truncation",
            "CXCR4-p.K337X-truncation-WHIM",
            "CXCR4-p.S338X-WHIM-variant",
            "CXCR4-de-novo-new-truncation-variant",
        ],
        "LYST": [
            "LYST-biallelic-truncating-severe-CHS-accelerated",
            "LYST-biallelic-missense-attenuated-CHS-adult",
            "LYST-compound-heterozygous-truncating-missense",
            "LYST-homozygous-splice-site-classic-CHS",
            "LYST-large-deletion-severe",
        ],
        "RAB27A": [
            "RAB27A-homozygous-p.Trp73X-GS2-classic",
            "RAB27A-compound-heterozygous-missense-severe",
            "RAB27A-splice-site-GS2",
            "RAB27A-frameshift-truncating-GS2",
            "RAB27A-missense-GTPase-domain-impaired-loading",
        ],
        "AP3B1": [
            "AP3B1-biallelic-truncating-HPS2-severe",
            "AP3B1-compound-heterozygous-HPS2",
            "AP3B1-splice-site-HPS2-moderate",
            "AP3B1-missense-beta3A-domain-HPS2",
            "AP3B1-deletion-exon-HPS2-with-HLH",
        ],
    }
    etiology_classes = etiology_map.get(gene, ["class-1", "class-2", "class-3", "class-4", "class-5"])
    etiology_dist = [rng.randint(5, 50) for _ in etiology_classes]
    total_e = sum(etiology_dist)
    etiology_dist = [round(e / total_e * n) for e in etiology_dist]
    etiology_dist[-1] = n - sum(etiology_dist[:-1])

    return {
        "gene": gene,
        "disease": meta["disease"],
        "inheritance": meta["inheritance"],
        "locus": meta["locus"],
        "aa": meta["aa"],
        "kda": meta["kda"],
        "n": n,
        "male_n": male_n,
        "female_n": female_n,
        "median_onset_months": round(median_onset, 1),
        "tx_response_pct": round(tx_response_pct, 3),
        "hsct_pct": round(hsct_pct, 3),
        "transform_pct": round(transform_pct, 3),
        "nbs_detected": 0,
        "pathognomonic": meta["pathognomonic"],
        "key_drug": meta["key_drug"],
        "precision_therapy": meta["precision_therapy"],
        "avoid": meta["avoid"],
        "etiology_classes": [
            {"label": lbl, "count": cnt}
            for lbl, cnt in zip(etiology_classes, etiology_dist)
        ],
    }


def generate_overview():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    return {
        "syndrome": "Hereditary-SCN-Granulocyte-Cytotoxic-Granule-Defect-Atlas",
        "gene": "ELANE-HAX1-G6PC3-WASP-CXCR4-LYST-RAB27A-AP3B1",
        "inheritance": "AD (ELANE, CXCR4) / AR (HAX1, G6PC3, LYST, RAB27A, AP3B1) / XLR+AD-GOF (WASP)",
        "eeg_hallmark": "LYST/HAX1-RS: seizures from neurodegeneration — generalised EEG epileptiform; HLH encephalopathy (LYST/RAB27A/AP3B1): diffuse slow + triphasic waves in crisis; CXCR4/WHIM: EEG normal (HPV-CNS rare); ELANE/SCN: EEG normal unless septic encephalopathy",
        "key_biomarker": "ELANE→ANC<200-cycling-3wk-PATHOGNOMONIC; HAX1→ANC<500-promyelocyte-arrest-BM; G6PC3→ANC<500-venous-angiectasias-skin-PATHOGNOMONIC; WASP→microplatelets-MPV<7fL-absent-WASP-protein-flow-PATHOGNOMONIC; CXCR4→myelokathexis-BM-PATHOGNOMONIC-hypersegmented-thin-strand-nuclei; LYST→giant-granules-PBS-PATHOGNOMONIC-NK-absent; RAB27A→silvery-hair-large-irregular-clumps-hair-shaft-PATHOGNOMONIC-NK-cytotoxicity-absent; AP3B1→absent-platelet-dense-granules-EM-PATHOGNOMONIC-OCA-neutropenia-HLH",
        "key_pathway": "Granule biogenesis and exocytosis (LYST-RAB27A-AP3B1), neutrophil maturation (ELANE-HAX1-G6PC3), actin cytoskeleton-platelet (WASP), chemokine receptor bone marrow retention (CXCR4)",
        "precision_therapy": "ELANE→G-CSF-filgrastim-AML-surveillance; HAX1→G-CSF-HSCT; G6PC3→G-CSF-cardiac-eval; WASP→HSCT-gene-therapy-lentiviral; CXCR4→Plerixafor-AMD3100-FDA2024; LYST→HSCT-before-accelerated-phase; RAB27A→HSCT-HLH-protocol; AP3B1→HSCT-HLH-desmopressin-bleeding",
        "total_patients": 320,
        "genes_count": 8,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_cohorts": cohorts,
        "nbs_genes": [],
        "cascade_testing_anchor": "SCN/Neutropenia workup: ANC <500/µL → CBC serial (cycling 3-week? → ELANE-CyN); bone marrow (myelokathexis → CXCR4; giant granules → LYST; promyelocyte arrest → ELANE/HAX1/G6PC3); skin exam (venous angiectasias → G6PC3); platelet size (microplatelets + eczema → WASP); silvery hair (→ LYST vs RAB27A vs AP3B1); HLH trigger → LYST/RAB27A/AP3B1",
        "pathognomonic_highlights": [
            "ELANE-CyN: 21-DAY CYCLING PERIOD — aphthous ulcers / pharyngitis cycling every 3 weeks synchronised with ANC nadir; no other neutropenia cycles so precisely; if history says 'monthly mouth ulcers' → check CBC every 3 days × 6 weeks for cycling confirmation",
            "G6PC3-SCN-IV: PROMINENT VENOUS ANGIECTASIAS on trunk/limbs skin — visible subcutaneous venous malformations; ONLY SCN subtype with consistent vascular skin findings; examine skin at diagnosis of any SCN",
            "WASP-WAS: MICROPLATELETS (MPV <7 fL) — platelet count and size BOTH reduced; MPV reported on standard CBC; PATHOGNOMONIC COMBINATION of thrombocytopaenia + tiny platelet size in a boy with eczema and infections",
            "CXCR4-WHIM: MYELOKATHEXIS — BM neutrophils with hypersegmented nuclei connected by THIN NUCLEAR STRANDS (caterpillar pattern) + cellular retention; pathognomonic BM finding; Plerixafor FDA-approved 2024 for WHIM; FIRST targeted therapy for a primary immunodeficiency",
            "LYST-CHS: GIANT AZUROPHIL GRANULES ON PBS — single most visible granule abnormality in haematology; visible on Giemsa without electron microscopy; in ALL granule-containing cells (neutrophils, eosinophils, monocytes, lymphocytes); + NK cell absent",
            "RAB27A-GS2: HAIR SHAFT MICROSCOPY — large irregular melanin clumps (vs GS1/MYO5A which also has silvery hair but clumps ARE present but SMALLER and more regular); KEY: PBS absent of giant granules distinguishes GS2 from CHS/LYST",
            "AP3B1-HPS2: ABSENT PLATELET DENSE GRANULES on EM — universal to ALL HPS types (1-10) but IMMUNE DEFECT (neutropenia + NK dysfunction + HLH) specific to HPS-2; PULMONARY FIBROSIS absent or mild in HPS-2 unlike HPS-1 where it is the primary cause of death",
            "HAX1-RS-VARIANT: NEURODEGENERATION in subset with biallelic mutations affecting BOTH isoforms 1+2 — epilepsy + intellectual disability + psychomotor regression; isoform characterisation of HAX1 mutation MANDATORY at diagnosis to predict neurological prognosis",
        ],
        "diagnostic_algorithm": {
            "anc_severe_constant": "ANC <500/µL constant from infancy → BM for maturation arrest → ELANE/HAX1/G6PC3 gene panel; check skin for venous angiectasias (G6PC3); cardiac echo (G6PC3)",
            "anc_cycling": "ANC cycling with ~21-day periodicity → ELANE (cyclic neutropenia); serial CBC every 3 days × 6 weeks confirms cycling; G-CSF shortens cycle",
            "hlh_triggered": "HLH (fever + cytopaenias + hyperferritinaemia + haemophagocytosis) → check hair/skin (silvery hair → LYST vs RAB27A vs AP3B1); PBS for giant granules (present → LYST; absent → RAB27A/AP3B1); NK function assay",
            "silvery_hair": "Silvery hair → hair shaft microscopy (large irregular clumps → GS2/RAB27A; confirm by PBS absent giant granules vs CHS/LYST) vs MYO5A (GS1: neurological features)",
            "microplatelets_male": "Thrombocytopaenia + tiny platelets (MPV <7 fL) + eczema + recurrent infections in male → Wiskott-Aldrich (WASP) → WASP protein flow cytometry → WASP sequencing",
            "warts_neutropenia": "Severe refractory HPV warts + neutropenia/lymphopenia + hypogammaglobulinaemia → WHIM (CXCR4) → BM for myelokathexis → CXCR4 sequencing",
            "oca_neutropenia_bleeding": "OCA (albinism) + neutropenia + prolonged bleeding time → HPS-2 (AP3B1) → platelet EM for absent dense granules → AP3B1 sequencing",
        },
    }


def generate_breakdown():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    neutropenia_scn_group = {
        "genes": ["ELANE", "HAX1", "G6PC3"],
        "mechanism": "Myeloid maturation arrest — neutrophil count low due to apoptosis or ER stress of precursors",
        "shared_feature": "All three cause severe congenital neutropenia (ANC <500/µL); all respond to G-CSF; all carry AML/MDS risk ~12-20% lifetime requiring annual BM surveillance; maturation arrest at promyelocyte on BM biopsy",
        "key_ddx": "ELANE: constant or cycling (21-day) ANC; AD inheritance; highest AML risk. HAX1: AR; RS-variant subgroup with neurodegeneration. G6PC3: AR; PATHOGNOMONIC venous angiectasias + cardiac defects.",
    }

    actin_cytoskeleton_group = {
        "genes": ["WASP"],
        "mechanism": "Actin cytoskeleton regulation in haematopoietic cells — LOF → thrombocytopaenia/immune defect; GOF → myeloid maturation block",
        "shared_feature": "WASP gene: LOF → Wiskott-Aldrich (microplatelets + eczema + infections); GOF → X-linked neutropenia; WASP protein expression differentiates severity of WAS",
        "key_ddx": "WAS-LOF: microplatelets + eczema + immune deficiency → WASP protein absent. XLN-GOF: isolated neutropenia, NO thrombocytopaenia → WASP protein GOF isoform. Platelets are the key differentiator.",
    }

    chemokine_receptor_group = {
        "genes": ["CXCR4"],
        "mechanism": "CXCR4 GOF → prolonged signalling → haematopoietic cells retained in bone marrow → peripheral neutropenia despite marrow hypercellularity",
        "shared_feature": "WHIM: myelokathexis (caterpillar nuclei) on BM is pathognomonic; warts from impaired anti-HPV immunity; plerixafor reverses the GOF mechanism directly",
        "key_ddx": "WHIM: myelokathexis + warts + hypogammaglobulinaemia. SCN types: no myelokathexis; no warts; no hypogammaglobulinaemia. BM morphology is the critical differentiator.",
    }

    granule_biogenesis_group = {
        "genes": ["LYST", "RAB27A", "AP3B1"],
        "mechanism": "Lysosome-related organelle (LRO) biogenesis/exocytosis defects → giant organelles (LYST) or impaired cytotoxic granule secretion (RAB27A, AP3B1)",
        "shared_feature": "All cause partial oculocutaneous albinism (silvery hair); all have NK-cell cytotoxicity defect → HLH susceptibility; all treated with HSCT for immune component",
        "key_ddx": "LYST: giant granules on PBS (PATHOGNOMONIC unique to LYST); accelerated phase HLH + severe neutrophil killing defect. RAB27A: NO giant granules; silvery hair + HLH, NO neurological. AP3B1: NO giant granules; OCA + neutropenia + absent platelet dense granules + HLH; PULMONARY FIBROSIS absent.",
    }

    cascade_testing_algorithm = [
        "Step 1 — ANC <500/µL in infant/child: Serial CBC × 6 weeks (ANC every 3 days to detect cycling); BM biopsy with morphology (look for promyelocyte arrest, myelokathexis, giant granules)",
        "Step 2 — If cycling (~21-day): ELANE sequencing first (cyclic neutropenia most likely)",
        "Step 3 — If constant neutropenia + promyelocyte arrest: ELANE → HAX1 → G6PC3 panel; examine skin for venous angiectasias (G6PC3); cardiac echo; isoform characterisation if HAX1 positive",
        "Step 4 — If myelokathexis on BM: CXCR4 sequencing; look for HPV warts; immunoglobulins",
        "Step 5 — If silvery hair/hypopigmentation + neutropenia/HLH: Hair shaft microscopy first; then PBS for giant granules (→ LYST if present; → RAB27A/AP3B1 if absent); NK cytotoxicity assay",
        "Step 6 — If WAS-like (male + microplatelets + eczema): WASP protein flow cytometry; WASP sequencing (LOF → WAS; GOF → XLN)",
        "Step 7 — If OCA + neutropenia + bleeding + HLH: platelet EM for absent dense granules; AP3B1 sequencing; HPS panel (AP3B1/HPS2 vs HPS1/3/5/6)",
        "Step 8 — HSCT timing: LYST/RAB27A → HSCT before accelerated phase (EBV surveillance monthly); WAS → HSCT at diagnosis if severe; AP3B1 → HSCT for life-threatening HLH",
    ]

    treatment_pathways = [
        {
            "gene": "ELANE",
            "disease": "SCN-I / Cyclic Neutropenia",
            "first_line": "G-CSF (filgrastim) 5-100 µg/kg/day SC; response ~90%; ANC target >1000/µL",
            "precision_therapy": "None (G-CSF is standard); gene therapy trials (SIN lentiviral vector, EVI1 genotoxicity risk reduced vs γ-retroviral)",
            "avoid": "Skip annual BM surveillance; delay G-CSF in omphalitis/neonatal sepsis; HSCT for CyN alone",
            "monitoring": "Annual BM + cytogenetics; CBC monthly; CSF3R mutation testing if AML transformation",
        },
        {
            "gene": "HAX1",
            "disease": "SCN-III Kostmann / RS Variant",
            "first_line": "G-CSF; HSCT for transformation or G-CSF failure; antiepileptics (RS variant)",
            "precision_therapy": "HSCT (haematological cure); isoform-specific genetic counselling; neuropaediatric co-management for RS variant",
            "avoid": "Assume all HAX1 biallelic have neurological (isoform 1-only do NOT); miss RS variant workup (EEG, MRI, developmental assessment)",
            "monitoring": "Annual BM; neurodevelopmental assessment 6-monthly (RS variant); HAX1 isoform characterisation at diagnosis",
        },
        {
            "gene": "G6PC3",
            "disease": "SCN-IV Dürr-Boztug",
            "first_line": "G-CSF; cardiac surgical repair (congenital heart defects); renal follow-up",
            "precision_therapy": "G-CSF; multidisciplinary cardiac/renal/metabolic management; HSCT for transformation",
            "avoid": "Miss venous angiectasias (pathognomonic diagnostic clue); skip cardiac echo at diagnosis; neglect renal imaging",
            "monitoring": "Annual BM; annual cardiac echo; annual renal US; glucose monitoring",
        },
        {
            "gene": "WASP",
            "disease": "Wiskott-Aldrich Syndrome / XLN",
            "first_line": "HSCT (curative for WAS); IVIg; TMP-SMX PCP prophylaxis; platelet transfusion only for life-threatening bleed",
            "precision_therapy": "HSCT (10/10 MUD or matched sibling); lentiviral gene therapy (GT-WAS; Lentiviral SIN safer vs γ-retroviral); WASP protein expression guides prognosis",
            "avoid": "Splenectomy first-line (increases severe infection risk); live vaccines before HSCT; eltrombopag (may worsen inflammation)",
            "monitoring": "WASP protein expression by flow; immunoglobulins 3-monthly; EBV/CMV surveillance",
        },
        {
            "gene": "CXCR4",
            "disease": "WHIM Syndrome",
            "first_line": "Plerixafor (AMD3100) 0.02-0.08 mg/kg SC daily (FDA-approved 2024 for WHIM); G-CSF alternative",
            "precision_therapy": "Plerixafor — directly reverses GOF pathomechanism (CXCR4 antagonist); first precision-targeted therapy for primary immunodeficiency; IVIg for hypogammaglobulinaemia; HPV vaccination of family",
            "avoid": "Miss bone marrow examination for myelokathexis (pathognomonic, diagnostic); neglect HPV vaccination; delay plerixafor in severe warts/neutropenia",
            "monitoring": "ANC + WBC monthly on plerixafor; immunoglobulins 6-monthly; HPV surveillance (genital annual)",
        },
        {
            "gene": "LYST",
            "disease": "Chediak-Higashi Syndrome",
            "first_line": "HSCT before accelerated phase (ONLY curative therapy for immune defect); HLH-2004 protocol for accelerated phase",
            "precision_therapy": "HSCT (allogeneic; best early, before HLH); gene therapy trials (challenging due to LYST gene size 3801aa); EBV prophylaxis/surveillance",
            "avoid": "Delay HSCT until after accelerated phase (outcomes far worse); assume NK function normal (it is absent); miss PBS giant granules",
            "monitoring": "EBV PCR monthly; ferritin monthly; NK function 6-monthly; BM annually",
        },
        {
            "gene": "RAB27A",
            "disease": "Griscelli Syndrome Type 2",
            "first_line": "HSCT curative for immune component; HLH-2004 protocol for active HLH",
            "precision_therapy": "HSCT (curative for HLH risk; silvery hair persists post-HSCT as melanocyte not replaced); rituximab pre-emptively for EBV",
            "avoid": "Confuse with CHS/LYST (both silvery hair — check PBS for giant granules: absent in GS2); confuse with GS1/MYO5A (check neurological features absent in GS2); delay HSCT after HLH",
            "monitoring": "EBV PCR monthly; ferritin monthly; NK cytotoxicity assay 6-monthly; HSCT outcome assessment",
        },
        {
            "gene": "AP3B1",
            "disease": "Hermansky-Pudlak Syndrome Type 2",
            "first_line": "HSCT for life-threatening HLH or severe infections; DDAVP/desmopressin for bleeding episodes; platelet transfusion for surgery",
            "precision_therapy": "HSCT for immune complications; HPS Network multidisciplinary follow-up; pulmonary function annual (HPS-2 lung disease milder than HPS-1/pirfenidone indication)",
            "avoid": "Confuse HPS-2 with HPS-1 (HPS-1: pulmonary fibrosis is primary concern; HPS-2: immune/HLH is primary concern); miss OCA+neutropenia+bleeding combination; delay platelet EM (absent dense granules = universal HPS)",
            "monitoring": "CBC monthly; NK function 6-monthly; pulmonary function annually; ophthalmology annually; platelet function studies 6-monthly",
        },
    ]

    return {
        "syndrome": "Hereditary-SCN-Granulocyte-Cytotoxic-Granule-Defect-Atlas",
        "total_patients": 320,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "pathway_groups": {
            "neutropenia_scn_maturation_arrest": neutropenia_scn_group,
            "actin_cytoskeleton_wasp": actin_cytoskeleton_group,
            "chemokine_receptor_marrow_retention": chemokine_receptor_group,
            "granule_biogenesis_lro": granule_biogenesis_group,
        },
        "cascade_testing_algorithm": cascade_testing_algorithm,
        "treatment_pathways": treatment_pathways,
        "gene_cohorts": cohorts,
        "hlh_risk_table": {
            "LYST": "Accelerated phase (HLH) >80% without HSCT — EBV-triggered; NK absent; giant granules diagnostic",
            "RAB27A": "HLH >80% — EBV-triggered; NK cytotoxicity absent; no giant granules",
            "AP3B1": "HLH susceptibility (lower penetrance than LYST/RAB27A) — neutropenia + NK defect; platelet dense granule EM diagnostic",
            "WASP": "EBV-associated lymphoma in WAS (not classic HLH); EBV-driven lymphoproliferation in ~10%",
            "ELANE": "No HLH risk",
            "HAX1": "AML/MDS transformation risk ~15-20%; not HLH",
            "G6PC3": "AML/MDS transformation risk; not HLH",
            "CXCR4": "No HLH risk; lymphoma risk lower; HPV-associated cancers main concern",
        },
        "plerixafor_spectrum": {
            "CXCR4_whim": "Plerixafor FDA-approved 2024 for WHIM: CXCR4 antagonist → blocks CXCL12/SDF-1 mediated retention → mobilises neutrophils + lymphocytes + B-cells → normalises counts; improves HPV wart clearance; reduces infections; given SC daily",
            "haematopoietic_mobilisation": "Plerixafor also used off-label in other bone marrow failure states with retention; mechanism directly reverses WHIM GOF pathomechanism",
            "distinction": "Plerixafor in WHIM = precision therapy targeting the molecular lesion; G-CSF in SCN = compensatory boost; fundamentally different mechanisms",
        },
    }


def generate_definitions():
    return {
        "syndrome": "Hereditary-SCN-Granulocyte-Cytotoxic-Granule-Defect-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "key_distinguishers": {
            "scn_constant_vs_cycling": "SCN-I (ELANE): constant ANC <200/µL from birth vs Cyclic Neutropenia (ELANE): 21-DAY CYCLING PATHOGNOMONIC — ANC troughs every 21 days synchronised with clinical symptoms (aphthous ulcers); cyclic = fewer AML events; test: serial ANC every 3 days × 6 weeks",
            "scn_subtypes_by_extrahaematological": "G6PC3 is the ONLY SCN subtype with vascular skin findings (venous angiectasias) + cardiac defects; HAX1 RS-variant is the ONLY SCN subtype with neurodegeneration; ELANE has highest AML risk; CXCR4-WHIM has myelokathexis (not maturation arrest)",
            "silvery_hair_ddx": "3 conditions: LYST/CHS (giant PBS granules PRESENT), RAB27A/GS2 (giant PBS granules ABSENT, NO neurological), MYO5A/GS1 (giant PBS granules ABSENT, neurological features); hair shaft microscopy large irregular clumps in all; use PBS + neurology to differentiate",
            "hlh_primary_immunodeficiency_ddx": "HLH in primary immunodeficiency: LYST (giant granules PBS), RAB27A (silvery hair no granules), AP3B1 (OCA + bleeding + neutropenia), WASP-EBV-lymphoma (microplatelets), UNC13D/STX11 (FHL — no albinism); perforin flow cytometry + NK cytotoxicity assay mandatory in all",
            "was_xlt_xln": "WASP LOF severity: absent WASP protein = classic WAS (severe); reduced WASP protein = XLT (mild thrombocytopaenia only); GOF WASP = XLN (neutropenia only, no thrombocytopaenia); WASP protein flow cytometry differentiates WAS vs XLT vs XLN",
            "myelokathexis_vs_promyelocyte_arrest": "Myelokathexis (CXCR4): BM hypercellular, neutrophils PRESENT but with abnormal nuclei (thin strands, hyperpycnotic); Promyelocyte arrest (ELANE/HAX1/G6PC3): BM shows maturation block at promyelocyte; complete distinction on BM morphology",
            "hps2_vs_hps1": "HPS-2 (AP3B1): IMMUNE DEFECT (neutropenia, HLH) + absent platelet dense granules + OCA; PULMONARY FIBROSIS absent/mild; HPS-1 (HSPA8): PULMONARY FIBROSIS (major cause of death by 4th-5th decade) + absent platelet dense granules + OCA; NO immune defect; pirfenidone for HPS-1 lung disease",
        },
        "treatment_matrix": {
            "g_csf_scn": "G-CSF (filgrastim): standard first-line for ELANE-SCN1, HAX1, G6PC3; dose 5-100 µg/kg; 90% response rate; AML surveillance mandatory with long-term G-CSF; PEGFILGRASTIM available for stable patients",
            "hsct_curative": "HSCT curative for: WAS (WASP), CHS (LYST), GS2 (RAB27A), HPS-2 (AP3B1); indicated for G-CSF failure in SCN; best outcomes with matched sibling or 10/10 MUD donor; timing critical (before accelerated phase in LYST/RAB27A)",
            "plerixafor_whim": "Plerixafor (AMD3100) — FIRST FDA-approved targeted therapy for primary immunodeficiency (WHIM, 2024); CXCR4 antagonist; 0.02-0.08 mg/kg SC daily; normalises ANC/WBC; reduces infections; clears HPV warts; transforms WHIM from debilitating to manageable",
            "hlh_protocol": "HLH-2004 protocol: dexamethasone 10 mg/m²/day × 8 weeks (then taper) + etoposide 150 mg/m² biweekly × 8 weeks ± cyclosporine; applies to LYST-accelerated-phase, RAB27A-GS2, AP3B1-HPS2 with HLH; bridge to HSCT",
            "gene_therapy_was": "WAS gene therapy: lentiviral SIN vector (safer than γ-retroviral); WASP cDNA; clinical trials show durable platelet correction and immune reconstitution; approved in some jurisdictions; FIRST gene therapy for immunodeficiency with platelet correction",
        },
        "definitions": [
            {
                "term": "Severe Congenital Neutropenia (SCN)",
                "definition": "ANC persistently <500/µL (severe) or <200/µL (very severe) from birth due to maturation arrest of myeloid precursors; causes omphalitis, recurrent bacterial infections, periodontitis; ELANE (SCN-I), HAX1 (SCN-III), G6PC3 (SCN-IV) account for majority; G-CSF responsive in ~90%; AML/MDS lifetime risk 15-20%; annual bone marrow surveillance mandatory. SCN-II = GFI1; SCN-V = VPS13B."
            },
            {
                "term": "Cyclic Neutropenia",
                "definition": "ELANE mutation subtype with ANC oscillating on a regular 21-day cycle; troughs to <200/µL for 3-6 days then recovers; synchronised clinical symptoms (aphthous ulcers, pharyngitis, lymphadenopathy) cycling every 21 days; 3-WEEK PERIOD PATHOGNOMONIC; AML risk much lower than constant SCN; G-CSF low-dose shortens cycle; serial ANC every 3 days × 6 weeks confirms diagnosis."
            },
            {
                "term": "Myelokathexis",
                "definition": "Bone marrow finding PATHOGNOMONIC for WHIM syndrome (CXCR4 GOF): neutrophils with hypersegmented nuclei connected by THIN CHROMATIN STRANDS (caterpillar pattern) + hyperpycnotic nuclei + degenerative changes; neutrophils PRESENT in marrow (unlike SCN where absent/arrested) but cannot egress peripherally due to CXCR4 over-retention; peripheral blood shows neutropenia despite marrow hypercellularity; marrow appears hypercellular with 'hyperlobed' granulocytes."
            },
            {
                "term": "Chediak-Higashi Giant Granules",
                "definition": "PATHOGNOMONIC for LYST/CHS: giant azurophil (primary) granules visible in ALL granule-containing cells on standard Giemsa-stained peripheral blood smear; neutrophils have large bluish cytoplasmic inclusions; eosinophils have giant eosinophilic granules; lymphocytes have large azurophil granules; caused by LYST protein loss → failure of lysosome fission → unlimited fusion → giant organelles; visible without electron microscopy; single most identifiable granule abnormality in haematology."
            },
            {
                "term": "Accelerated Phase (CHS/Chediak-Higashi)",
                "definition": "EBV-triggered haemophagocytic lymphohistiocytosis (HLH) in LYST-CHS patients; occurs in >80% without HSCT; characterised by: fever, hepatosplenomegaly, pancytopaenia, hyperferritinaemia, haemophagocytosis on BM; caused by absent NK and CTL cytotoxic function → EBV cannot be cleared → uncontrolled macrophage activation; FATAL without aggressive HLH treatment (dexamethasone + etoposide) and subsequent HSCT; HSCT before accelerated phase = best outcomes."
            },
            {
                "term": "Griscelli Syndrome Types (GS1/GS2/GS3)",
                "definition": "Three types all with silvery/grey hair but distinct features: GS1 (MYO5A): silvery hair + NEUROLOGICAL (intellectual disability, seizures, hypotonia); GS2 (RAB27A): silvery hair + HLH (no neurological features); GS3 (MLPH): silvery hair + BLEEDING DIATHESIS (melanophilin; no immune/neurological); diagnosis by hair shaft microscopy (irregular large clumps in all GS types) + phenotype; BM/PBS giant granules ABSENT in all GS types (distinguishes from LYST/CHS)."
            },
            {
                "term": "Microplatelets / Microthrombocytopaenia (Wiskott-Aldrich)",
                "definition": "Pathognomonic for WAS (WASP LOF): mean platelet volume (MPV) <7 fL (normal 7.5-12 fL) on automated CBC; platelet COUNT also low (20,000-50,000/µL); COMBINATION of low count + very small platelets = PATHOGNOMONIC for WAS/XLT; automated analysers report MPV routinely; no other immune thrombocytopaenia has consistently small platelets; ITP has normal/large platelets. Platelet size normalises post-HSCT or gene therapy."
            },
            {
                "term": "WHIM Tetrad (Warts-Hypogammaglobulinaemia-Infections-Myelokathexis)",
                "definition": "Clinical syndrome from CXCR4 AD-GOF: (1) Warts: severe refractory HPV warts from birth/childhood (common, plantar, genital, laryngeal); (2) Hypogammaglobulinaemia: impaired B-cell egress → low IgG/IgM; (3) Infections: recurrent sinopulmonary bacterial infections; (4) Myelokathexis: BM pathognomonic finding; neutropenia (and lymphopaenia) despite marrow hypercellularity. Plerixafor (AMD3100) FDA-approved 2024 — first precision-targeted therapy for primary immunodeficiency."
            },
            {
                "term": "Hermansky-Pudlak Syndrome (HPS) Types",
                "definition": "Family of 10 autosomal recessive disorders (HPS1-10) all sharing: (1) OCA (oculocutaneous albinism); (2) absent platelet dense granules on EM → bleeding diathesis; differing by: HPS-1/HPS-4 → pulmonary fibrosis (MAIN CAUSE DEATH; pirfenidone therapy); HPS-2 (AP3B1) → IMMUNE DEFECT (neutropenia, NK dysfunction, HLH susceptibility; NO severe pulmonary fibrosis); HPS-3/5/6 → milder, no pulmonary fibrosis, no immune defect; KEY: platelet EM (absent dense granules) = diagnosis any HPS; subtype by gene panel."
            },
            {
                "term": "Plerixafor (AMD3100) in WHIM",
                "definition": "CXCR4 antagonist (blocks CXCL12/SDF-1 binding) → mobilises haematopoietic progenitors and mature cells from bone marrow → raises peripheral ANC, lymphocyte count, B-cell count; FDA-approved 2024 specifically for WHIM syndrome; SC injection 0.02-0.08 mg/kg daily; significantly improves neutrophil counts, reduces infections, promotes clearance of HPV warts; first example of targeted molecular therapy correcting the underlying pathomechanism in primary immunodeficiency; prior use: haematopoietic stem cell mobilisation before autologous transplant."
            },
            {
                "term": "HAX1 RS Variant (Neurological Kostmann)",
                "definition": "Subtype of HAX1-SCN (Kostmann disease) where biallelic mutations affect BOTH HAX1 isoforms 1 AND 2 (RS variant): neutropenia + progressive neurodegeneration (epilepsy, intellectual disability, psychomotor regression); isoform-1-only mutations: neutropenia WITHOUT neurological; HAX1 isoform 2 expressed in neurons with anti-apoptotic mitochondrial function in neuronal cells; critical to characterise the mutation at diagnosis for isoform determination; neurological prognosis not improved by HSCT in RS variant."
            },
            {
                "term": "Cascade Testing — SCN/Granule Defect Pathway",
                "definition": "Step 1: ANC <500/µL → serial ANC every 3 days × 6 weeks (cycling?). Step 2: Cycling 21-day → ELANE first. Step 3: Constant + maturation arrest → ELANE/HAX1/G6PC3 panel; skin exam (venous angiectasias → G6PC3); cardiac echo; HAX1 isoform characterisation. Step 4: Myelokathexis on BM → CXCR4. Step 5: Silvery hair + HLH → PBS giant granules (LYST if present; RAB27A if absent); hair shaft microscopy. Step 6: Male + microplatelets + eczema → WASP. Step 7: OCA + neutropenia + bleeding → HPS panel (AP3B1 for HPS-2 immune type)."
            },
        ],
    }
