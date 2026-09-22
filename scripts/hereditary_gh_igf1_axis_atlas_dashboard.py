#!/usr/bin/env python3
"""Hereditary-GH-IGF1-Axis-Atlas — Complete 8-Gene GH/IGF-1 Excess and Deficiency Atlas
AIP     (aryl hydrocarbon receptor-interacting protein; 330 aa; 11q13.2; AD LOF;
         FIPA / AIP-associated pituitary adenoma; young macroadenoma; SSA resistant;
         seed SEED_BASE+0) ·
GHR     (GH receptor; 638 aa; 5p13.1; AR LOF;
         Laron syndrome; absent IGF-1; rhGH INEFFECTIVE; rhIGF-1 treatment;
         cancer protective; seed SEED_BASE+1) ·
IGF1    (IGF-I; 195 aa; 12q23.2; AR LOF;
         absent IGF-1; severe growth failure + SNHL + mild ID; rhIGF-1 treatment;
         seed SEED_BASE+2) ·
IGF1R   (IGF-1 receptor; 1367 aa; 15q26.3; AD/AR LOF;
         IGF-1 resistance; SGA non-catch-up; elevated IGF-1; seed SEED_BASE+3) ·
STAT5B  (STAT5B; 787 aa; 17q21.2; AR LOF;
         GH insensitivity + severe immunodeficiency; no JAK2-STAT5B signalling;
         γc cytokine responses absent; seed SEED_BASE+4) ·
IGFALS  (ALS; 302 aa; 16p13.3; AR LOF;
         acid-labile subunit deficiency; mild growth failure; osteopenia; delayed puberty;
         seed SEED_BASE+5) ·
PAPPA2  (pappalysin-2; 1791 aa; 1q25.2; AR LOF;
         high IGFBP3/ALS; low free IGF-1; growth failure + hypothyroidism + thrombocytopenia;
         seed SEED_BASE+6) ·
GPR101  (GPR101; 536 aa; Xq26.3; AD GOF microduplication;
         X-LAG — X-linked acrogigantism; infant onset; highest GH/IGF-1;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2958-2965)
"""
import random

SEED_BASE = 2958

ATLAS_GENES = [
    {
        "gene": "AIP",
        "protein": (
            "AIP -- 11q13.2 AD LOF -- 330aa -- Aryl-Hydrocarbon-Receptor-"
            "Interacting-Protein-37kDa-HSP90-Co-Chaperone-FIPA-"
            "Young-Macroadenoma-SSA-Resistant-Pegvisomant-Effective-OMIM-605555"
        ),
        "locus": "11q13.2",
        "protein_size": (
            "330 aa / 37 kDa (AIP — aryl hydrocarbon receptor interacting protein; co-chaperone; "
            "FUNCTION: HSP90 co-chaperone; interacts with AhR, HSP90, PDE4A5, PKAR; "
            "  regulates cAMP/PKA signalling in pituitary somatotroph cells; "
            "  tumour suppressor in pituitary; "
            "LOF CONSEQUENCE: "
            "  Loss of AIP → unchecked somatotroph proliferation; "
            "  GH adenoma (somatotropinoma); young onset (typically <30 years); "
            "  Macroadenoma at presentation (>80%); "
            "  Aggressive: cavernous sinus invasion common; "
            "  SSA RESISTANCE: somatostatin analogue (octreotide/lanreotide) poorly effective; "
            "    Mechanism: AIP LOF → altered somatostatin receptor 2 (SSTR2) expression/signalling; "
            "  Pegvisomant (GH receptor antagonist) effective; "
            "  FAMILIAL: FIPA (familial isolated pituitary adenoma); "
            "  Prolactinoma in some AIP families; "
            "encoded 11q13.2"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) LOF — AIP / FIPA: "
            "  PHENOTYPE: "
            "    Somatotropinoma (GH adenoma) — most common AIP tumour type (>70%); "
            "    Young onset: diagnosis typically 15-30 years (vs 40s-50s for sporadic); "
            "    Macroadenoma at diagnosis in >80% (vs <50% sporadic); "
            "    Cavernous sinus invasion common; height >2 m if prepubertal onset (gigantism); "
            "    Adult onset: acromegaly phenotype (coarse features, jaw prognathism, macroglossia); "
            "  KEY CLINICAL RULE — SSA RESISTANT: "
            "    AIP LOF → SSTR2 pathway impaired → octreotide/lanreotide response POOR; "
            "    Pegvisomant (GHR antagonist) normalises IGF-1 in >90%; "
            "    Surgery (transsphenoidal) first line; pegvisomant after; "
            "  FAMILY HISTORY: "
            "    FIPA: ≥2 affected family members; can be same or mixed tumour type; "
            "    Incomplete penetrance (~30%); many obligate carriers unaffected; "
            "    Screen: first-degree relatives at risk → AIP genetic testing; "
            "  SURVEILLANCE: "
            "    MRI pituitary annually if AIP carrier; "
            "    IGF-1 and GH annually from puberty; "
            "  Autoantibody context: not applicable (non-autoimmune)"
        ),
        "disease_category": (
            "AIP-FIPA — YOUNG SOMATOTROPINOMA — SSA RESISTANT — PEGVISOMANT PREFERRED: "
            "  DIAGNOSIS CLUE: young acromegaly/gigantism + macroadenoma + family history; "
            "    IGF-1 and GH elevated; MRI shows macroadenoma often invading cavernous sinus; "
            "  TREATMENT: transsphenoidal surgery (first line); pegvisomant; radiation; "
            "    AVOID relying on SSA as sole medical therapy (poor AIP response); "
            "  GENETIC TESTING INDICATION: "
            "    GH adenoma before age 40 + family history → AIP; "
            "    Macroadenoma in child/adolescent → AIP; "
            "    FIPA syndrome (any pituitary tumour + family history)"
        ),
        "disease_pathway": (
            "AIP LOF → DYSREGULATED cAMP/PKA → UNCHECKED SOMATOTROPH PROLIFERATION: "
            "  Normal AIP function: "
            "    AIP interacts with HSP90 → stabilises SSTR2 at membrane; "
            "    SSTR2 signal transduction intact → SSA (somatostatin) suppresses GH; "
            "    AIP also regulates PDE4A5 → keeps cAMP levels controlled; "
            "    Suppresses cell cycle progression (G1-S checkpoint); "
            "  AIP LOF: "
            "    Reduced SSTR2 stability → SSA resistance; "
            "    PDE4A5 dysregulation → elevated cAMP → protein kinase A activation; "
            "    PKA → CREB phosphorylation → increased GH transcription; "
            "    Unchecked G1-S transition → somatotroph proliferation → adenoma"
        ),
    },
    {
        "gene": "GHR",
        "protein": (
            "GHR -- 5p13.1 AR LOF -- 638aa -- Growth-Hormone-Receptor-"
            "70kDa-JAK2-STAT5B-Signalling-Laron-Syndrome-"
            "rhGH-INEFFECTIVE-rhIGF1-TREATMENT-Cancer-Protective-OMIM-600946"
        ),
        "locus": "5p13.1",
        "protein_size": (
            "638 aa / 70 kDa (GHR — growth hormone receptor; cytokine class I receptor; "
            "FUNCTION: pre-formed homodimer; GH binding → conformational change → "
            "  JAK2 transphosphorylation → STAT5B phosphorylation → nuclear translocation; "
            "  IGF-1 gene activation (primary target); IGFBP3 and ALS production; "
            "  Also signals via MAPK, IRS-1/PI3K; "
            "  Hepatic GHR is primary source of circulating IGF-1 (~75%); "
            "LOF CONSEQUENCE (Laron syndrome, GH insensitivity type 1): "
            "  GHR absent/non-functional → GH cannot signal → IGF-1 absent; "
            "  GH elevated (no feedback inhibition via IGF-1); "
            "  Severe dwarfism (−4 to −10 SDS); "
            "  Obesity, hypoglycaemia (neonatal), normal intelligence; "
            "  UNIQUE FEATURES: "
            "    Very low incidence of cancer (IGF-1 pathway promotes malignancy); "
            "    Low T2DM risk (IGF-1 promotes insulin sensitivity); "
            "    'Protective' against common age-related diseases; "
            "  Ecuadorian Laron cohort: landmark long-term study (Guevara-Aguirre/Longo); "
            "encoded 5p13.1"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — GHR / LARON SYNDROME (GH insensitivity type 1): "
            "  PHENOTYPE: "
            "    Severe short stature: −4 to −10 SDS height; "
            "    Dwarfism with normal to increased body fat; "
            "    Blue sclerae, saddle nose, small hands/feet; "
            "    Neonatal hypoglycaemia (IGF-1 needed for glucose homeostasis); "
            "    Normal intelligence (unlike IGF1 LOF); "
            "    Delayed bone age; small genitalia; "
            "  KEY CLINICAL RULE — rhGH INEFFECTIVE: "
            "    GH receptor absent/non-functional → rhGH cannot signal → NO IGF-1 generated; "
            "    Administering rhGH is futile (will not generate IGF-1); "
            "    TREATMENT: recombinant human IGF-1 (mecasermin / INCRELEX); "
            "    IGF-1 therapy bypasses the blocked receptor; "
            "    Start early (before bone age <9 years for best outcome); "
            "  BIOCHEMICAL FINGERPRINT: "
            "    GH very HIGH (no feedback); IGF-1 very LOW/absent; "
            "    IGFBP3 very LOW; ALS very LOW; "
            "    GH GENERATION TEST: give rhGH 0.1 u/kg ×4 days → IGF-1 does NOT rise (diagnostic); "
            "  CANCER / LONGEVITY: "
            "    Ecuadorian Laron cohort: 0 cancer deaths (vs 17% in controls); "
            "    Near-zero T2DM; "
            "    IGF-1 is growth-promoting → its absence protective against proliferative diseases"
        ),
        "disease_category": (
            "GHR-LARON SYNDROME — SEVERE DWARFISM — rhGH INEFFECTIVE — rhIGF-1 TREATMENT: "
            "  DIAGNOSIS CLUE: severe short stature + very HIGH GH + very LOW IGF-1 + LOW IGFBP3; "
            "    GH generation test: IGF-1 does NOT rise after 4 days rhGH; "
            "  TREATMENT: recombinant human IGF-1 (mecasermin) — subcutaneous BD with meals; "
            "  GENETIC TESTING INDICATION: "
            "    Severe short stature + HIGH GH + low IGF-1 → GHR sequencing (exons 2-10 + splice sites); "
            "    5p13 deletion screen with MLPA if sequencing negative"
        ),
        "disease_pathway": (
            "GHR LOF → NO JAK2-STAT5B SIGNAL → NO IGF-1 PRODUCTION: "
            "  Normal GH-IGF1 axis: "
            "    Hypothalamic GHRH → pituitary GH release; "
            "    Hepatic GHR (homodimer) binds GH → JAK2 (associated with GHR) transphosphorylates; "
            "    STAT5B phosphorylated → dimerises → nuclear translocation → IGF1 gene promoter; "
            "    IGF-1 secreted → systemic effects (growth plate, muscle, brain); "
            "    IGF-1 → negative feedback on GH release (pituitary SSTR, hypothalamic SRIF); "
            "  GHR LOF: "
            "    GH cannot bind functional receptor → JAK2 not activated; "
            "    STAT5B not phosphorylated → IGF1 gene not expressed → serum IGF-1 absent; "
            "    No IGF-1 feedback → GH continues secreting uninhibited → HIGH GH; "
            "    No IGF-1 → linear growth arrest; no IGFBP3 synthesis; ALS absent"
        ),
    },
    {
        "gene": "IGF1",
        "protein": (
            "IGF1 -- 12q23.2 AR LOF -- 195aa -- Insulin-Like-Growth-Factor-I-"
            "7.6kDa-Hepatic-Peripheral-GHR-Downstream-IGF1-ABSOLUTE-ABSENT-"
            "SNHL-Intellectual-Disability-Severe-Growth-Failure-rhIGF1-Treatment-OMIM-147440"
        ),
        "locus": "12q23.2",
        "protein_size": (
            "195 aa / 7.6 kDa (IGF1 — insulin-like growth factor I; "
            "FUNCTION: small secreted peptide; structural homology to insulin; "
            "  binds IGF1R → PI3K/AKT, MAPK/ERK activation; "
            "  Circulates in ternary complex (IGF-1 + IGFBP3 + ALS); "
            "  Sources: liver (75% circulating), peripheral tissues (autocrine/paracrine); "
            "  Required for: linear growth, muscle mass, CNS development (auditory nerve), cognitive function; "
            "LOF CONSEQUENCE (biallelic homozygous deletion described in original case): "
            "  Absolute IGF-1 absence → profound growth failure (≤ −7 SDS); "
            "  GH elevated (no feedback); "
            "  SENSORINEURAL HEARING LOSS: inner ear cochlear development requires IGF-1; "
            "    PATHOGNOMONIC for IGF1 gene LOF (distinguishes from GHR LOF where SNHL absent); "
            "  INTELLECTUAL DISABILITY: mild-moderate; CNS IGF-1 required for cortical development; "
            "  Intrauterine growth restriction (IUGR): IGF-1 needed for in utero growth; "
            "  Microcephaly: head circumference affected; "
            "encoded 12q23.2"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — IGF1 gene deficiency: "
            "  RARE — very few confirmed cases worldwide; "
            "  PHENOTYPE: "
            "    Very severe short stature (≤ −7 SDS); IUGR at birth; "
            "    Microcephaly; "
            "    SNHL: sensorineural hearing loss — cochlear development requires IGF-1; "
            "    Intellectual disability: mild-moderate; "
            "    Delayed bone age; "
            "  BIOCHEMISTRY: "
            "    GH: elevated (no IGF-1 feedback); "
            "    IGF-1: ABSENT (undetectable); "
            "    IGFBP3: low (IGF-1 needed for IGFBP3 stability/production); "
            "    GH generation test: IGF-1 still undetectable after rhGH (IGF1 gene absent); "
            "  DISTINGUISHING FROM GHR LOF (Laron): "
            "    IGF1 LOF has SNHL + intellectual disability; "
            "    GHR-Laron: NO SNHL, NORMAL intelligence; "
            "    Both: HIGH GH + absent IGF-1; "
            "  TREATMENT: recombinant human IGF-1 (mecasermin); "
            "    Auditory rehab (cochlear implant if severe SNHL); "
            "    Early intervention for cognitive development"
        ),
        "disease_category": (
            "IGF1 LOF — SEVERE IUGR + SNHL + ID — rhIGF-1 TREATMENT: "
            "  DIAGNOSIS CLUE: severe short stature + SNHL + intellectual disability + absent IGF-1; "
            "    Distinguishes from Laron (GHR LOF): SNHL + ID = IGF1 gene, not GHR; "
            "  TREATMENT: rhIGF-1 (mecasermin); cochlear implant; cognitive support; "
            "  GENETIC TESTING INDICATION: "
            "    Absent IGF-1 + SNHL + ID + GHR sequencing negative → IGF1 gene"
        ),
        "disease_pathway": (
            "IGF1 LOF → ABSENT CIRCULATING + LOCAL IGF-1 → MULTI-SYSTEM GROWTH FAILURE: "
            "  GH-IGF1 axis breaks at the ligand level: "
            "    GH signals normally → hepatic IGF1 gene promoter active → but NO IGF1 mRNA; "
            "    GH very elevated (no negative feedback); "
            "  LINEAR GROWTH: "
            "    IGF1R in growth plate chondrocytes never activated → columnar elongation absent; "
            "  COCHLEAR: "
            "    IGF-1 essential for spiral ganglion neuron and hair cell survival; "
            "    IGF1 LOF → cochlear apoptosis → SNHL; "
            "    Not a feature of GHR LOF (peripheral GH insensitivity) because local IGF-1 production "
            "    from other tissues may partially compensate cochlea in Laron; "
            "    IGF1 gene LOF → all sources absent → cochlear vulnerable; "
            "  CNS: "
            "    Cortical IGF-1 autocrine/paracrine → dendritic arborisation, synaptogenesis; "
            "    IGF1 LOF → impaired CNS maturation → intellectual disability"
        ),
    },
    {
        "gene": "IGF1R",
        "protein": (
            "IGF1R -- 15q26.3 AD/AR LOF -- 1367aa -- IGF-1-Receptor-"
            "155kDa-RTK-alpha2beta2-Tetrameric-IGF1-Resistance-"
            "SGA-Non-Catch-Up-Elevated-IGF1-Silver-Russell-Like-OMIM-147370"
        ),
        "locus": "15q26.3",
        "protein_size": (
            "1367 aa / 155 kDa (IGF1R — IGF-1 receptor; receptor tyrosine kinase; "
            "FUNCTION: alpha2beta2 disulphide-linked tetramer; IGF-1 binding → autophosphorylation; "
            "  activates IRS-1/2, PI3K/AKT, RAS/MAPK/ERK; "
            "  Ubiquitous expression: critical in prenatal + postnatal growth; "
            "  Also binds IGF-2 (lower affinity) and insulin (very low); "
            "LOF CONSEQUENCE: "
            "  Haploinsufficiency → SGA (small for gestational age) that fails to catch up; "
            "  Elevated IGF-1 (receptor resistance → less negative feedback via IGF-1 → GH remains high → "
            "    more IGF-1 produced — but cannot signal effectively); "
            "  PRENATAL GROWTH FAILURE: IGF1R needed for in utero growth (IGF-2 in placenta also uses IGF1R); "
            "  Dysmorphic features: triangular facies, large ears, body asymmetry (Silver-Russell-like); "
            "  Normal or near-normal height (milder than GHR/IGF1 LOF); "
            "  BIALLELIC LOF: much more severe; "
            "encoded 15q26.3"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) / AUTOSOMAL RECESSIVE (AR) LOF — IGF1R: "
            "  PHENOTYPE: "
            "    Small for gestational age (SGA): birth weight/length ≤ −2 SDS; "
            "    FAILURE TO CATCH UP: unlike constitutional SGA, IGF1R does not catch up; "
            "    Height: −2 to −4 SDS; "
            "    Elevated serum IGF-1 (resistance → reduced negative feedback); "
            "    IGFBP3 normal or elevated; "
            "    Dysmorphic features: Silver-Russell phenotype (body asymmetry, triangular face); "
            "  KEY CLINICAL RULE — ELEVATED IGF-1 IN SHORT PATIENT: "
            "    HIGH IGF-1 + SHORT STATURE → think IGF1R haploinsufficiency; "
            "    CONTRAST with GHR/IGF1 LOF where IGF-1 is LOW; "
            "  TREATMENT: "
            "    rhGH: increases IGF-1 production (more ligand to overcome partial receptor deficit); "
            "    Variable response (depends on residual receptor function); "
            "    rhIGF-1 at pharmacological dose may partially overcome resistance; "
            "  TESTING: "
            "    IGF1R: haploinsufficiency by MLPA (deletions common); sequencing; "
            "    15q26 deletion: check for contiguous gene syndrome"
        ),
        "disease_category": (
            "IGF1R LOF — SGA NON-CATCH-UP + ELEVATED IGF-1 — rhGH VARIABLE RESPONSE: "
            "  DIAGNOSIS CLUE: SGA + failure to catch up + HIGH or normal-high IGF-1 + dysmorphic features; "
            "    Distinct from GHR/IGF1 LOF (where IGF-1 is LOW); "
            "  TREATMENT: rhGH (partial response); MLPA for 15q26 deletions; "
            "  GENETIC TESTING INDICATION: "
            "    SGA + failure to catch up + elevated IGF-1 → IGF1R MLPA + sequencing"
        ),
        "disease_pathway": (
            "IGF1R LOF → REDUCED PI3K/AKT + MAPK SIGNALLING → PRENATAL + POSTNATAL GROWTH FAILURE: "
            "  IGF1R normal function: "
            "    IGF-1 binds α-subunit → β-subunit kinase domain autophosphorylates Tyr residues; "
            "    IRS-1 docking → PI3K → PIP3 → AKT → mTOR → protein synthesis + cell survival; "
            "    Grb2/SOS → RAS → MAPK/ERK → proliferation; "
            "    In placenta: IGF-2 binds IGF1R → trophoblast invasion + nutrient transport; "
            "  IGF1R LOF (haploinsufficiency): "
            "    50% receptor → reduced signal amplitude per unit IGF-1; "
            "    Normal or high IGF-1 levels (more ligand produced to compensate); "
            "    Growth PARTIALLY maintained (distinguishes from complete LOF); "
            "    Prenatal: placenta expresses IGF1R; haploinsufficiency → reduced nutrient transport → SGA"
        ),
    },
    {
        "gene": "STAT5B",
        "protein": (
            "STAT5B -- 17q21.2 AR LOF -- 787aa -- Signal-Transducer-Activator-"
            "Transcription-5B-90kDa-JAK2-Downstream-"
            "GH-Insensitivity-PLUS-Severe-Immunodeficiency-Eosinophilia-OMIM-604260"
        ),
        "locus": "17q21.2",
        "protein_size": (
            "787 aa / 90 kDa (STAT5B — signal transducer and activator of transcription 5B; "
            "FUNCTION: downstream of JAK2 in multiple cytokine pathways: "
            "  GH receptor → JAK2 → STAT5B → IGF1 gene (growth); "
            "  γc cytokine receptors (IL-2, IL-7, IL-15, IL-21) → JAK1/3 → STAT5B (immune); "
            "  Prolactin receptor → JAK2 → STAT5B (lactation); "
            "LOF CONSEQUENCE — DUAL PHENOTYPE: "
            "  1. GH INSENSITIVITY: no STAT5B → IGF1 gene not expressed → short stature like Laron; "
            "     GH high, IGF-1 low, IGFBP3 low (identical to GHR LOF biochemistry); "
            "  2. SEVERE IMMUNODEFICIENCY: "
            "     IL-2/IL-7/IL-15/IL-21 → STAT5B → T-cell development, NK-cell development; "
            "     STAT5B LOF → severely reduced CD4+ T-cells, NK cells; "
            "     Eosinophilia (characteristic — regulatory T-cell function impaired); "
            "     Autoimmune features: autoimmune hepatitis, pneumonitis, eczema; "
            "     Susceptibility to opportunistic infections (Pneumocystis, viruses); "
            "KEY TEACHING: STAT5B LOF = Laron + immunodeficiency — the combination is PATHOGNOMONIC; "
            "encoded 17q21.2"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — STAT5B: "
            "  PHENOTYPE: "
            "    SHORT STATURE: severe (similar to Laron/GHR LOF); GH high; IGF-1 absent; "
            "    IMMUNODEFICIENCY: "
            "      CD4+ T-cell lymphopenia; NK-cell deficiency; "
            "      Eosinophilia (invariant feature — autoimmune); "
            "      Autoimmune hepatitis + lymphoid interstitial pneumonitis; "
            "      Eczema / atopic disease; "
            "      Recurrent infections (viral, fungal, Pneumocystis); "
            "    HYPERPROLACTINAEMIA: mild (prolactin receptor also signals via STAT5B); "
            "  KEY CLINICAL RULE — LARON + IMMUNODEFICIENCY = STAT5B: "
            "    GH high + IGF-1 low (Laron-like) BUT the patient also has "
            "    eosinophilia + autoimmune hepatitis + recurrent infections → STAT5B LOF; "
            "    GHR LOF (true Laron) has NORMAL immune function; "
            "  TREATMENT: "
            "    rhIGF-1 for growth (same as Laron); "
            "    Immunological management: immunoglobulin replacement, prophylactic antimicrobials; "
            "    Caution: immunosuppression for autoimmune hepatitis may worsen infections; "
            "  Autoantibodies: organ-specific (hepatitis, thyroid) can be positive"
        ),
        "disease_category": (
            "STAT5B LOF — LARON PHENOTYPE + SEVERE IMMUNODEFICIENCY — EOSINOPHILIA PATHOGNOMONIC: "
            "  DIAGNOSIS CLUE: Laron-like growth failure + eosinophilia + autoimmune features + infections; "
            "    GHR sequencing negative + immunodeficiency → STAT5B; "
            "  TREATMENT: rhIGF-1 + immunological support; "
            "  GENETIC TESTING INDICATION: "
            "    GH insensitivity (Laron biochemistry) + any immune/autoimmune feature → STAT5B"
        ),
        "disease_pathway": (
            "STAT5B LOF → BOTH GH PATHWAY AND γc CYTOKINE PATHWAY DISRUPTED: "
            "  GH pathway: "
            "    GHR → JAK2 → STAT5B (phosphorylated Y699) → dimerisation → nuclear IGF1 promoter; "
            "    STAT5B LOF → IGF1 gene silent → same as GHR LOF but one step downstream; "
            "  γc cytokine pathway: "
            "    IL-2R (γc + IL-2Rβ) → JAK1/JAK3 → STAT5A/B → IL-2 target genes: "
            "      FoxP3 (Treg differentiation); "
            "      Bcl-2 (T-cell survival); "
            "    IL-7R → JAK1/JAK3 → STAT5B → T-cell development; "
            "    STAT5B LOF → FoxP3 reduced → Treg deficient → autoimmunity; "
            "    STAT5B LOF → Bcl-2 reduced → T-cell apoptosis → lymphopenia; "
            "    Eosinophilia: STAT5B normally promotes eosinophil apoptosis via IL-5 pathway; "
            "    STAT5B LOF → eosinophil accumulation (inverse of eosinophil regulation)"
        ),
    },
    {
        "gene": "IGFALS",
        "protein": (
            "IGFALS -- 16p13.3 AR LOF -- 302aa -- Acid-Labile-Subunit-"
            "35kDa-LRR-Ternary-Complex-Scaffold-"
            "Mild-Growth-Failure-Osteopenia-Delayed-Puberty-OMIM-601489"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "302 aa / 35 kDa (IGFALS — insulin-like growth factor binding protein acid-labile subunit; "
            "FUNCTION: leucine-rich repeat (LRR) protein; scaffolds the ternary complex; "
            "  TERNARY COMPLEX: IGF-1 + IGFBP3 + ALS (ALS = anchor/scaffold); "
            "  ALS prolongs IGF-1 half-life (t1/2 ~15h in ternary complex vs 10 min free); "
            "  ALS also scaffolds IGF-2 + IGFBP3 complexes; "
            "  Liver-specific expression; GH-regulated; "
            "LOF CONSEQUENCE: "
            "  ALS absent → ternary complex cannot form → IGF-1 and IGFBP3 rapidly cleared; "
            "  Serum IGF-1 very low (but present at low levels from non-hepatic sources); "
            "  Serum IGFBP3 very low; GH mildly elevated; "
            "  MILD GROWTH FAILURE: −2 to −3 SDS (much milder than GHR/IGF1 LOF); "
            "  OSTEOPENIA: IGF-1 needed for bone formation → reduced bone density; "
            "  DELAYED PUBERTY: mild; catch-up partial; "
            "  NOT associated with SNHL or intellectual disability; "
            "encoded 16p13.3"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — IGFALS / ALS DEFICIENCY: "
            "  PHENOTYPE: "
            "    Mild-moderate short stature (−2 to −3 SDS); "
            "    Very LOW IGF-1 (proportionally lower than height deficit); "
            "    Very LOW IGFBP3; "
            "    GH mildly elevated; "
            "    OSTEOPENIA: reduced bone mineral density; fracture risk; "
            "    Delayed puberty (by ~1-2 years); "
            "    NORMAL intelligence; NORMAL hearing; "
            "  KEY CLINICAL RULE — MILDER THAN LARON: "
            "    IGFALS LOF milder because some free IGF-1 still reaches tissues (no ALS but IGFBP3 binary "
            "    complex still possible and some free IGF-1 escapes renal clearance); "
            "    CONTRAST: GHR LOF → no IGF-1 at all (absent hepatic production); "
            "  GH GENERATION TEST: "
            "    rhGH → IGF-1 may slightly rise (liver makes some IGFBP3 binary complex); "
            "    IGFBP3 still low (ALS absent); "
            "  TREATMENT: "
            "    rhIGF-1 may help; "
            "    Calcium + Vitamin D for osteopenia; "
            "    Bisphosphonates if osteoporosis-level fractures; "
            "  BIOCHEMISTRY IS PATHOGNOMONIC: "
            "    Extremely low IGF-1 + extremely low IGFBP3 + mild short stature + NO immune deficit"
        ),
        "disease_category": (
            "IGFALS LOF — MILD GROWTH FAILURE + OSTEOPENIA + LOW IGF1 + LOW IGFBP3: "
            "  DIAGNOSIS CLUE: very low IGF-1 + very low IGFBP3 + mild short stature (not severe); "
            "    Normal immune function; normal intelligence; normal hearing; "
            "  TREATMENT: calcium/Vit D; rhIGF-1 consideration; "
            "  GENETIC TESTING INDICATION: "
            "    Low IGF-1 + low IGFBP3 + mild short stature + GHR negative → IGFALS"
        ),
        "disease_pathway": (
            "IGFALS LOF → TERNARY COMPLEX ABSENT → ACCELERATED IGF-1 CLEARANCE: "
            "  Normal ternary complex formation: "
            "    Liver: GH → STAT5B → IGFALS gene + IGFBP3 gene + IGF1 gene all activated; "
            "    IGF-1 secreted → binds IGFBP3 (binary complex) → ALS bridges to binary complex → "
            "    150 kDa ternary complex formed; "
            "    Ternary complex is too large to cross capillary membranes → stays in vascular space; "
            "    Half-life extended enormously (15-20h vs minutes for free IGF-1); "
            "  IGFALS LOF: "
            "    No ALS → IGFBP3 binary complex (60 kDa) forms but rapidly cleared by kidney; "
            "    Free IGF-1 (7.6 kDa) cleared within minutes; "
            "    Net result: very low steady-state IGF-1 and IGFBP3 despite GH signalling intact; "
            "  LIVER STILL MAKES IGF-1: "
            "    GH → STAT5B → IGF1 mRNA → IGF-1 protein produced; "
            "    But it clears so fast that serum levels almost undetectable; "
            "    Local tissue IGF-1 (autocrine) partially compensates → milder than GHR/IGF1 LOF"
        ),
    },
    {
        "gene": "PAPPA2",
        "protein": (
            "PAPPA2 -- 1q25.2 AR LOF -- 1791aa -- Pappalysin-2-PAPP-A2-"
            "200kDa-Metalloprotease-IGFBP3-IGFBP5-Cleaver-"
            "High-IGFBP3-Low-Free-IGF1-Growth-Failure-Hypothyroidism-Thrombocytopenia-OMIM-603247"
        ),
        "locus": "1q25.2",
        "protein_size": (
            "1791 aa / 200 kDa (PAPPA2 — pappalysin-2 / PAPP-A2; metzincin metalloprotease; "
            "FUNCTION: zinc metalloprotease; cleaves IGFBP3 and IGFBP5 → releases IGF-1 from ternary complex; "
            "  Without PAPPA2: IGF-1 trapped in ternary complex (cannot bind IGF1R); "
            "  PAPPA2 makes FREE IGF-1 bioavailable at target tissues; "
            "  Co-expressed with IGF1R in growth plate cartilage, bone, thyroid; "
            "LOF CONSEQUENCE: "
            "  IGFBP3 and IGFBP5 not cleaved → IGF-1 remains BOUND and INACTIVE; "
            "  Total IGF-1: HIGH (all produced, none cleared); "
            "  Free IGF-1: very LOW (none released from binding protein); "
            "  IGFBP3: very HIGH (not degraded); "
            "  GH: HIGH (feedback based on free IGF-1 → none → GH keeps rising); "
            "  PHENOTYPE: "
            "    Short stature despite HIGH total IGF-1; "
            "    Hypothyroidism: thyroid follicle cells require PAPPA2 for local free IGF-1; "
            "    Thrombocytopenia: megakaryocyte maturation requires IGF-1; "
            "encoded 1q25.2"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — PAPPA2: "
            "  RARE — small number of families identified; "
            "  PHENOTYPE: "
            "    SHORT STATURE: −2 to −5 SDS; "
            "    HYPOTHYROIDISM: primary; TSH elevated; free T4 low; "
            "    THROMBOCYTOPENIA: mild-moderate; megakaryocyte maturation requires local free IGF-1; "
            "    Delayed bone age; insulin resistance (free IGF-1 needed for insulin sensitivity); "
            "  KEY BIOCHEMICAL FINGERPRINT — OPPOSITE TO LARON: "
            "    PAPPA2 LOF: TOTAL IGF-1 HIGH + IGFBP3 VERY HIGH + FREE IGF-1 LOW; "
            "    GHR LOF: TOTAL IGF-1 ABSENT + IGFBP3 ABSENT; "
            "  TREATMENT: "
            "    Rhigf-1 (provides free ligand bypassing the binding protein trap); "
            "    Levothyroxine for hypothyroidism; "
            "    Thrombopoietin receptor agonists if thrombocytopenia symptomatic; "
            "  MISDIAGNOSIS RISK: "
            "    Total IGF-1 HIGH → doctor reassured → misses free IGF-1 measurement; "
            "    KEY: always measure FREE IGF-1 if total IGF-1 is high in a short child"
        ),
        "disease_category": (
            "PAPPA2 LOF — HIGH TOTAL IGF-1 + HIGH IGFBP3 + LOW FREE IGF-1 — TRIPLE PHENOTYPE: "
            "  DIAGNOSIS CLUE: short stature + high total IGF-1 + high IGFBP3 + hypothyroidism + "
            "  thrombocytopenia; free IGF-1 very low (must be measured specifically); "
            "  TREATMENT: rhIGF-1 + levothyroxine; "
            "  GENETIC TESTING INDICATION: "
            "    Short stature + high IGF-1 + high IGFBP3 + hypothyroidism → PAPPA2"
        ),
        "disease_pathway": (
            "PAPPA2 LOF → IGF-1 TRAPPED IN TERNARY COMPLEX → TISSUE IGF-1 STARVATION DESPITE HIGH LEVELS: "
            "  Normal PAPPA2 action at growth plate: "
            "    Circulating ternary complex (IGF-1 + IGFBP3 + ALS) arrives at cartilage; "
            "    PAPPA2 (expressed by chondrocytes) cleaves IGFBP3 → free IGF-1 released locally; "
            "    Free IGF-1 binds IGF1R on chondrocyte → columnar proliferation → bone growth; "
            "  PAPPA2 LOF: "
            "    IGFBP3 intact → ternary complex intact → IGF-1 cannot exit complex; "
            "    Free IGF-1 in tissue fluid = near zero; "
            "    Chondrocyte IGF1R never activated despite high total IGF-1 in serum; "
            "    THYROID: follicle cells use PAPPA2 to release local free IGF-1 → TSH sensitivity; "
            "    PAPPA2 LOF → reduced thyroid IGF-1 signalling → TSH resistance → hypothyroidism; "
            "    MEGAKARYOCYTE: requires local free IGF-1 for maturation → thrombocytopenia"
        ),
    },
    {
        "gene": "GPR101",
        "protein": (
            "GPR101 -- Xq26.3 AD GOF microduplication -- 536aa -- G-Protein-"
            "Coupled-Receptor-101-57kDa-Orphan-GPCR-"
            "X-LAG-X-Linked-Acrogigantism-Infant-Onset-HIGHEST-GH-IGF1-OMIM-300393"
        ),
        "locus": "Xq26.3",
        "protein_size": (
            "536 aa / 57 kDa (GPR101 — G-protein-coupled receptor 101; orphan GPCR; "
            "FUNCTION: hypothalamic/pituitary GPCR; activates cAMP pathway; "
            "  unknown endogenous ligand (orphan); "
            "  GOF/overexpression → somatotroph GH release; "
            "  Normally low expression; microduplication → overexpression in somatotroph cells; "
            "GOF CONSEQUENCE (X-LAG — Xq26.3 microduplication): "
            "  GPR101 overexpressed in pituitary → massive GH hypersecretion; "
            "  Onset in infancy/early childhood (earliest gigantism known); "
            "  HIGHEST GH and IGF-1 of any pituitary tumour; "
            "  Mixed GH + prolactin adenoma; "
            "  GIGANTISM: height +4 to +7 SDS before treatment; "
            "  X-LINKED: males more severely affected; females mosaic/milder; "
            "  SOMATIC MOSAIC: many cases have mosaicism (not in all cells); "
            "encoded Xq26.3"
        ),
        "inheritance": (
            "X-LINKED (XL) AD GOF MICRODUPLICATION — GPR101 / X-LAG: "
            "  PHENOTYPE: "
            "    GIGANTISM onset 2-4 years (infancy/early childhood) — earliest of all GH excess forms; "
            "    Height velocity massively accelerated (>12 cm/year); "
            "    Serum GH: extremely high (often >100 mU/L); "
            "    IGF-1: extremely elevated (>3 SDS for age); "
            "    Acral enlargement, coarse features, hyperphagia, polydipsia; "
            "    Mixed GH/prolactin pituitary adenoma; "
            "    Head circumference > 2 SDS; "
            "  X-LINKED PATTERN: "
            "    Males: usually severe (hemizygous duplication → full expression); "
            "    Females: mosaic (somatic duplication; milder); "
            "    GERMLINE: some families with X-LAG across generations; "
            "    DE NOVO: many cases; check parents; "
            "  KEY CLINICAL RULE — EARLIEST GIGANTISM = X-LAG UNTIL PROVEN OTHERWISE: "
            "    Gigantism presenting <5 years → X-LAG + AIP excluded; "
            "    Xq26.3 microduplication: FISH / array CGH / MLPA; "
            "    Standard sequencing MISSES microduplication; "
            "  TREATMENT: "
            "    Surgery (transsphenoidal) first; "
            "    Pasireotide (SSA) + pegvisomant combination (X-LAG often refractory); "
            "    Radiation if residual/recurrent; "
            "    Lanreotide/octreotide: partial response; combination therapy often needed"
        ),
        "disease_category": (
            "GPR101 X-LAG — INFANT-ONSET GIGANTISM — HIGHEST GH/IGF-1 — ARRAY CGH REQUIRED: "
            "  DIAGNOSIS CLUE: gigantism onset <5 years + extremely high GH + mixed GH/prolactin adenoma; "
            "    X-linked pattern; MLPA/array CGH for Xq26.3 microduplication; "
            "    Standard sequencing MISSES — must use copy number methods; "
            "  TREATMENT: surgery + pasireotide + pegvisomant combination; "
            "  GENETIC TESTING INDICATION: "
            "    Gigantism onset <5 years → Xq26.3 MLPA/array CGH → GPR101"
        ),
        "disease_pathway": (
            "GPR101 GOF → ELEVATED cAMP → CONSTITUTIVE GH HYPERSECRETION IN SOMATOTROPHS: "
            "  Normal GPR101 context: "
            "    Low expression in normal pituitary; minimal cAMP contribution; "
            "    Orphan receptor — no known ligand; may respond to a yet-uncharacterised neuropeptide; "
            "  GPR101 microduplication / overexpression: "
            "    Increased copy number → 5-10× more GPR101 protein in somatotroph cells; "
            "    GPR101 constitutively couples to Gs → adenylyl cyclase → cAMP elevated; "
            "    cAMP → PKA → CREB phosphorylation → GH gene transcription + release; "
            "    Unchecked GH secretion → massive IGF-1 production → gigantism; "
            "    SSTR2 may be downregulated (elevated cAMP → SSTR2 internalisation); "
            "    Explains partial SSA resistance in X-LAG (similar to AIP but different mechanism)"
        ),
    },
]


def _make_patients(seed: int, gene: str, n: int = 40) -> list:
    """Generate 40 synthetic GH/IGF-1 axis patients for one gene. Plausible educational simulations."""
    rng = random.Random(seed)

    gene_params = {
        "AIP": {
            "ht_sds_range": (2.5, 6.0), "igf1_x_uln_range": (2.5, 8.0), "gh_range": (20, 200),
            "age_range": (14, 35), "macro_pct": 0.82, "ssa_resist_pct": 0.75,
            "short_stature": False,
        },
        "GHR": {
            "ht_sds_range": (-10.0, -4.0), "igf1_x_uln_range": (0.0, 0.05), "gh_range": (40, 200),
            "age_range": (2, 18), "macro_pct": 0.0, "ssa_resist_pct": 0.0,
            "short_stature": True,
        },
        "IGF1": {
            "ht_sds_range": (-9.0, -5.0), "igf1_x_uln_range": (0.0, 0.04), "gh_range": (30, 180),
            "age_range": (1, 12), "macro_pct": 0.0, "ssa_resist_pct": 0.0,
            "short_stature": True,
        },
        "IGF1R": {
            "ht_sds_range": (-4.0, -1.5), "igf1_x_uln_range": (1.2, 3.5), "gh_range": (5, 25),
            "age_range": (2, 18), "macro_pct": 0.0, "ssa_resist_pct": 0.0,
            "short_stature": True,
        },
        "STAT5B": {
            "ht_sds_range": (-9.0, -4.5), "igf1_x_uln_range": (0.0, 0.06), "gh_range": (35, 190),
            "age_range": (1, 15), "macro_pct": 0.0, "ssa_resist_pct": 0.0,
            "short_stature": True,
        },
        "IGFALS": {
            "ht_sds_range": (-3.0, -1.5), "igf1_x_uln_range": (0.02, 0.10), "gh_range": (8, 40),
            "age_range": (5, 20), "macro_pct": 0.0, "ssa_resist_pct": 0.0,
            "short_stature": True,
        },
        "PAPPA2": {
            "ht_sds_range": (-5.0, -2.0), "igf1_x_uln_range": (1.8, 4.5), "gh_range": (15, 80),
            "age_range": (3, 18), "macro_pct": 0.0, "ssa_resist_pct": 0.0,
            "short_stature": True,
        },
        "GPR101": {
            "ht_sds_range": (4.0, 7.5), "igf1_x_uln_range": (3.0, 10.0), "gh_range": (50, 300),
            "age_range": (1, 8), "macro_pct": 0.90, "ssa_resist_pct": 0.65,
            "short_stature": False,
        },
    }
    p = gene_params.get(gene, gene_params["AIP"])

    def treatment_choice(gene, macro, ssa_res, short_stat):
        r = rng.random()
        if gene == "AIP":
            if r < 0.55: return "Transsphenoidal surgery (first line)"
            if r < 0.75: return "Pegvisomant (GHR antagonist — post-surgery residual)"
            if r < 0.88: return "Surgery + pegvisomant combination"
            if r < 0.96: return "Radiation + pegvisomant"
            return "SSA (partial response — bridge pre-surgery)"
        elif gene == "GHR":
            if r < 0.75: return "Recombinant human IGF-1 (mecasermin / Increlex)"
            if r < 0.90: return "rhIGF-1 + GH monitoring"
            return "Supportive care + nutritional optimisation"
        elif gene == "IGF1":
            if r < 0.70: return "Recombinant human IGF-1 (mecasermin) + cochlear implant"
            if r < 0.88: return "rhIGF-1 + auditory rehabilitation"
            return "rhIGF-1 + special education support"
        elif gene == "IGF1R":
            if r < 0.60: return "Recombinant human GH (rhGH) — partial response"
            if r < 0.80: return "Observation (mild phenotype)"
            return "rhGH + IGF-1 monitoring"
        elif gene == "STAT5B":
            if r < 0.55: return "rhIGF-1 + immunoglobulin replacement"
            if r < 0.75: return "rhIGF-1 + prophylactic antimicrobials"
            if r < 0.88: return "HSCT evaluation (severe immunodeficiency)"
            return "rhIGF-1 + immunosuppression for autoimmune hepatitis (careful)"
        elif gene == "IGFALS":
            if r < 0.55: return "Observation + calcium/Vit D supplementation"
            if r < 0.75: return "rhIGF-1 (modest benefit)"
            return "Supportive care + physiotherapy"
        elif gene == "PAPPA2":
            if r < 0.60: return "Recombinant human IGF-1 + levothyroxine"
            if r < 0.80: return "Levothyroxine + growth monitoring"
            return "rhIGF-1 + thyroid management + thrombopoietin agonist"
        else:  # GPR101
            if r < 0.50: return "Transsphenoidal surgery + pasireotide"
            if r < 0.70: return "Surgery + pasireotide + pegvisomant combination"
            if r < 0.85: return "Lanreotide + pegvisomant (pre/post surgery)"
            return "Radiation + pegvisomant (refractory residual)"

    patients = []
    for i in range(n):
        ht_sds   = round(rng.uniform(*p["ht_sds_range"]), 1)
        igf1_uln = round(rng.uniform(*p["igf1_x_uln_range"]), 2)
        gh_val   = round(rng.uniform(*p["gh_range"]), 1)
        age_dx   = rng.randint(*p["age_range"])
        macro    = rng.random() < p["macro_pct"]
        ssa_res  = rng.random() < p["ssa_resist_pct"]
        short    = p["short_stature"]
        eosinoph = gene == "STAT5B" and rng.random() < 0.90
        snhl     = gene == "IGF1" and rng.random() < 0.80
        hypo_thy = gene == "PAPPA2" and rng.random() < 0.70
        thrombcy = gene == "PAPPA2" and rng.random() < 0.55
        tx = treatment_choice(gene, macro, ssa_res, short)

        patients.append({
            "id":                  f"{gene}-{i+1:02d}",
            "gene":                gene,
            "age_at_dx":           age_dx,
            "height_sds":          ht_sds,
            "igf1_x_uln":          igf1_uln,
            "gh_mU_L":             gh_val,
            "macroadenoma":        macro,
            "ssa_resistant":       ssa_res,
            "short_stature":       short,
            "eosinophilia":        eosinoph,
            "snhl":                snhl,
            "hypothyroidism":      hypo_thy,
            "thrombocytopenia":    thrombcy,
            "treatment":           tx,
        })
    return patients


# ── API surface ───────────────────────────────────────────────────────────────

def generate_overview() -> dict:
    """Atlas overview — aggregate stats across all 8 GH/IGF-1 axis genes."""
    all_patients = []
    for idx, g in enumerate(ATLAS_GENES):
        all_patients.extend(_make_patients(SEED_BASE + idx, g["gene"]))

    n             = len(all_patients)
    n_excess      = sum(1 for p in all_patients if p["height_sds"] > 2.0)
    n_deficiency  = sum(1 for p in all_patients if p["height_sds"] < -2.0)
    n_macro       = sum(1 for p in all_patients if p["macroadenoma"])
    n_ssa_res     = sum(1 for p in all_patients if p["ssa_resistant"])
    n_eosinoph    = sum(1 for p in all_patients if p["eosinophilia"])
    n_snhl        = sum(1 for p in all_patients if p["snhl"])
    n_hypo        = sum(1 for p in all_patients if p["hypothyroidism"])
    mean_ht_sds   = round(sum(p["height_sds"] for p in all_patients) / n, 2)
    mean_igf1     = round(sum(p["igf1_x_uln"] for p in all_patients) / n, 2)

    gene_summary = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        gene_summary.append({
            "gene":            g["gene"],
            "locus":           g["locus"],
            "n_patients":      len(pts),
            "mean_ht_sds":     round(sum(p["height_sds"] for p in pts) / len(pts), 2),
            "mean_igf1_x_uln": round(sum(p["igf1_x_uln"] for p in pts) / len(pts), 2),
            "mean_gh":         round(sum(p["gh_mU_L"] for p in pts) / len(pts), 1),
            "macroadenoma_pct":round(100 * sum(1 for p in pts if p["macroadenoma"]) / len(pts), 1),
            "ssa_resist_pct":  round(100 * sum(1 for p in pts if p["ssa_resistant"]) / len(pts), 1),
            "eosinoph_pct":    round(100 * sum(1 for p in pts if p["eosinophilia"]) / len(pts), 1),
            "snhl_pct":        round(100 * sum(1 for p in pts if p["snhl"]) / len(pts), 1),
            "axis_type": (
                "GH-excess"     if g["gene"] in ("AIP", "GPR101") else
                "GH-insensitivity" if g["gene"] in ("GHR", "STAT5B") else
                "IGF-absence"   if g["gene"] == "IGF1" else
                "IGF-resistance"if g["gene"] == "IGF1R" else
                "IGF-transport" if g["gene"] in ("IGFALS", "PAPPA2") else
                "GH-insensitivity"
            ),
        })

    return {
        "atlas":          "Hereditary-GH-IGF1-Axis-Atlas",
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "n_genes":        len(ATLAS_GENES),
        "n_patients":     n,
        "seeds":          f"{SEED_BASE}\u2013{SEED_BASE + len(ATLAS_GENES) - 1}",
        "axis_disorders": [
            "AIP-FIPA young somatotropinoma SSA-resistant (AIP LOF)",
            "Laron syndrome GH insensitivity rhGH-ineffective (GHR LOF)",
            "IGF-1 deficiency SNHL intellectual disability (IGF1 LOF)",
            "IGF-1 resistance SGA non-catch-up elevated IGF-1 (IGF1R LOF)",
            "GH insensitivity + immunodeficiency eosinophilia (STAT5B LOF)",
            "ALS deficiency mild growth failure osteopenia (IGFALS LOF)",
            "PAPPA2 trapped IGF-1 hypothyroidism thrombocytopenia (PAPPA2 LOF)",
            "X-LAG infant gigantism highest GH/IGF-1 array CGH needed (GPR101 GOF)",
        ],
        "aggregate_metrics": {
            "mean_height_sds":    mean_ht_sds,
            "mean_igf1_x_uln":    mean_igf1,
            "gh_excess_pct":      round(100 * n_excess    / n, 1),
            "gh_deficiency_pct":  round(100 * n_deficiency/ n, 1),
            "macroadenoma_pct":   round(100 * n_macro     / n, 1),
            "ssa_resistant_pct":  round(100 * n_ssa_res   / n, 1),
            "eosinophilia_pct":   round(100 * n_eosinoph  / n, 1),
            "snhl_pct":           round(100 * n_snhl      / n, 1),
            "hypothyroid_pct":    round(100 * n_hypo      / n, 1),
        },
        "gene_summary":   gene_summary,
        "key_clinical_rules": [
            "AIP-FIPA SSA RESISTANT: young GH adenoma + macroadenoma + family history → AIP; SSA (octreotide/lanreotide) POOR response; pegvisomant (GHR antagonist) effective >90%",
            "LARON GH GENERATION TEST: give rhGH ×4 days → IGF-1 does NOT rise → GHR LOF confirmed; NEVER give rhGH as treatment — rhIGF-1 (mecasermin) is correct",
            "IGF1 LOF vs GHR LOF: IGF1 gene LOF has SNHL + intellectual disability; GHR (Laron) has normal intelligence and normal hearing — key distinguisher",
            "IGF1R LOF: ELEVATED total IGF-1 + short stature — always measure free IGF-1; PAPPA2 LOF also has high total IGF-1 but add hypothyroidism + thrombocytopenia",
            "STAT5B LOF = LARON + IMMUNODEFICIENCY: eosinophilia invariant; autoimmune hepatitis + recurrent infections + GH insensitivity → STAT5B (NOT GHR)",
            "X-LAG DETECTION: Xq26.3 microduplication → standard sequencing MISSES; array CGH or MLPA required; infant-onset gigantism (<5 years) → X-LAG excluded by array CGH",
            "PAPPA2 TRAP: total IGF-1 HIGH → do NOT be reassured; free IGF-1 very low → PAPPA2 or IGFALS defect; PAPPA2 adds hypothyroidism + thrombocytopenia",
            "IGFALS MILDEST GH-IGF1 DISORDER: very low IGF-1 + very low IGFBP3 + only mild short stature + normal immunity + normal hearing → IGFALS (not GHR/IGF1 LOF)",
        ],
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for all 8 GH/IGF-1 axis genes."""
    genes_data = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        genes_data.append({
            "gene":              g["gene"],
            "locus":             g["locus"],
            "protein":           g["protein"],
            "protein_size":      g["protein_size"],
            "inheritance":       g["inheritance"],
            "disease_category":  g["disease_category"],
            "disease_pathway":   g["disease_pathway"],
            "n_patients":        len(pts),
            "mean_ht_sds":       round(sum(p["height_sds"] for p in pts) / len(pts), 2),
            "mean_igf1_x_uln":   round(sum(p["igf1_x_uln"] for p in pts) / len(pts), 2),
            "mean_gh":           round(sum(p["gh_mU_L"] for p in pts) / len(pts), 1),
            "macroadenoma_pct":  round(100 * sum(1 for p in pts if p["macroadenoma"]) / len(pts), 1),
            "ssa_resist_pct":    round(100 * sum(1 for p in pts if p["ssa_resistant"]) / len(pts), 1),
            "eosinoph_pct":      round(100 * sum(1 for p in pts if p["eosinophilia"]) / len(pts), 1),
            "snhl_pct":          round(100 * sum(1 for p in pts if p["snhl"]) / len(pts), 1),
            "treatment_breakdown": treatments,
            "patients":          pts,
        })
    return {
        "atlas":    "Hereditary-GH-IGF1-Axis-Atlas",
        "count":    len(genes_data),
        "genes":    genes_data,
    }


def generate_definitions() -> dict:
    """Key clinical terms for Hereditary-GH-IGF1-Axis-Atlas."""
    definitions = [
        {
            "term": "AIP-FIPA — Young Somatotropinoma and SSA Resistance",
            "genes": ["AIP"],
            "definition": (
                "AIP-FIPA: familial isolated pituitary adenoma caused by AIP LOF. "
                "DIAGNOSIS: young acromegaly/gigantism + macroadenoma + incomplete family history "
                "(penetrance ~30%). Presentation age typically 15-30 years (vs 45-50 sporadic). "
                "SSA RESISTANCE: octreotide/lanreotide produce <50% IGF-1 normalisation in >75% of AIP cases. "
                "Mechanism: AIP stabilises SSTR2 at somatotroph membrane; AIP LOF → SSTR2 unstable → "
                "SSA cannot bind effectively. "
                "TREATMENT: "
                "1) Transsphenoidal surgery (first line); "
                "2) Pegvisomant (GHR antagonist) — normalises IGF-1 >90% of cases; "
                "3) Pasireotide (pan-SSTR agonist) — partial benefit; "
                "4) Radiation for residual/recurrent; "
                "SURVEILLANCE: MRI pituitary + IGF-1 annually for all AIP carriers. "
                "FAMILY SCREENING: Offer AIP genetic testing to all first-degree relatives."
            ),
        },
        {
            "term": "Laron Syndrome (GHR LOF) — GH Insensitivity Type 1",
            "genes": ["GHR"],
            "definition": (
                "LARON SYNDROME: AR biallelic GHR LOF → GH cannot signal → absent IGF-1. "
                "BIOCHEMICAL FINGERPRINT: "
                "GH very HIGH (50-200 mU/L or more); IGF-1 ABSENT (<5 ng/mL); "
                "IGFBP3 absent; ALS absent; GH generation test: IGF-1 does NOT rise after 4 days rhGH. "
                "CLINICAL: severe dwarfism −4 to −10 SDS; normal intelligence; normal hearing; obesity; "
                "neonatal hypoglycaemia; blue sclerae; saddle nose; delayed bone age. "
                "CANCER PROTECTIVE OBSERVATION: "
                "Ecuadorian cohort (Guevara-Aguirre/Longo) — 0 cancer deaths vs 17% control relatives; "
                "Linked to absence of IGF-1 signalling (promotes proliferation). "
                "NEVER GIVE rhGH — NO FUNCTIONAL GHR — rhGH IS FUTILE AND POTENTIALLY HARMFUL. "
                "TREATMENT: "
                "Recombinant human IGF-1 (mecasermin/Increlex) SC BD with meals; "
                "Start early (before bone age 9 years for best outcome); "
                "Monitor hypoglycaemia with injections (mecasermin lowers glucose)."
            ),
        },
        {
            "term": "IGF1 Gene Deficiency — SNHL + Intellectual Disability Key Distinguisher",
            "genes": ["IGF1"],
            "definition": (
                "IGF1 GENE LOF (not receptor): ligand absent → no signalling possible. "
                "DISTINGUISHES FROM GHR LOF: "
                "IGF1 LOF has SENSORINEURAL HEARING LOSS (cochlear IGF-1 essential for "
                "spiral ganglion neuron and hair cell development) + INTELLECTUAL DISABILITY "
                "(cortical IGF-1 required for synaptogenesis). "
                "GHR-Laron: NORMAL intelligence, NORMAL hearing — key exam differentiator. "
                "BOTH have: high GH + absent IGF-1 + absent IGFBP3 + dwarfism. "
                "ADDITIONAL FEATURES IGF1 LOF: IUGR + microcephaly. "
                "TREATMENT: rhIGF-1 (mecasermin) + cochlear implant if severe SNHL. "
                "GENETIC TESTING: if GHR sequencing negative + SNHL/ID present → IGF1 gene "
                "(both deletion and point mutations described)."
            ),
        },
        {
            "term": "STAT5B — Laron + Immunodeficiency (Dual Phenotype)",
            "genes": ["STAT5B"],
            "definition": (
                "STAT5B LOF: downstream of both GHR (JAK2→STAT5B) AND γc cytokines (JAK1/3→STAT5B). "
                "GROWTH PHENOTYPE: Laron-like (GH high, IGF-1 absent); GHR sequencing negative. "
                "IMMUNE PHENOTYPE: "
                "Eosinophilia (invariant — regulatory T-cell deficit); "
                "CD4+ T-cell lymphopenia; NK-cell deficiency; "
                "Autoimmune hepatitis (elevated transaminases + liver biopsy); "
                "Lymphoid interstitial pneumonitis; "
                "Eczema + atopy; "
                "Opportunistic infections (Pneumocystis, CMV, EBV). "
                "CLINICAL RULE: any Laron-like patient with eosinophilia + liver disease + "
                "infections = STAT5B until proven otherwise. "
                "TREATMENT: "
                "Growth: rhIGF-1 (mecasermin); "
                "Immune: immunoglobulin replacement; prophylactic TMP-SMX; "
                "Autoimmune hepatitis: careful immunosuppression (balance vs infection risk); "
                "HSCT evaluation if severe combined immunodeficiency."
            ),
        },
        {
            "term": "X-LAG (GPR101) — Earliest Gigantism — Array CGH Required",
            "genes": ["GPR101"],
            "definition": (
                "X-LAG (X-linked acrogigantism): Xq26.3 microduplication → GPR101 overexpression → "
                "massive somatotroph GH hypersecretion. "
                "PATHOGNOMONIC: gigantism onset <5 years (infancy/early childhood) — earliest known. "
                "GH and IGF-1: highest of any pituitary tumour type. "
                "DETECTION: Xq26.3 microduplication — STANDARD SEQUENCING MISSES; "
                "MANDATORY: array CGH or MLPA for Xq26.3. "
                "X-LINKED: males full expression; females mosaic (somatic duplication) — milder. "
                "TREATMENT CHALLENGE: SSA partially resistant (similar mechanism to AIP); "
                "Pasireotide + pegvisomant combination usually needed; "
                "Surgery + multimodal medical therapy. "
                "SCREEN: any child with gigantism onset <5 years → Xq26.3 array CGH + AIP sequencing + "
                "MEN1 panel if family history."
            ),
        },
        {
            "term": "PAPPA2 vs IGFALS — IGF-1 Availability Disorders",
            "genes": ["PAPPA2", "IGFALS"],
            "definition": (
                "IGFALS DEFICIENCY: ALS absent → IGF-1 + IGFBP3 rapidly cleared; "
                "very low total IGF-1 + very low IGFBP3; mild short stature (−2 to −3 SDS); "
                "osteopenia; normal immunity; normal hearing; normal intelligence. "
                "PAPPA2 LOF: IGFBP3/5 not cleaved → IGF-1 trapped in ternary complex; "
                "total IGF-1 HIGH; IGFBP3 VERY HIGH; free IGF-1 very LOW; "
                "short stature + hypothyroidism + thrombocytopenia. "
                "KEY DISCRIMINATOR: "
                "IGFALS: total IGF-1 LOW + IGFBP3 LOW; "
                "PAPPA2: total IGF-1 HIGH + IGFBP3 HIGH + free IGF-1 LOW. "
                "PAPPA2 RULE: high total IGF-1 in short patient → measure FREE IGF-1 and IGFBP3; "
                "if IGFBP3 very high → PAPPA2 LOF. "
                "IGFALS MILDEST GH-IGF1 DISORDER overall. "
                "TREATMENT: IGFALS → calcium/VitD; PAPPA2 → rhIGF-1 + levothyroxine."
            ),
        },
        {
            "term": "GH-IGF1 Axis Biochemical Fingerprint Map",
            "genes": ["AIP", "GHR", "IGF1", "IGF1R", "STAT5B", "IGFALS", "PAPPA2", "GPR101"],
            "definition": (
                "BIOCHEMICAL MAP — DIFFERENTIAL OF 8 GH/IGF-1 AXIS DISORDERS: "
                "AIP (excess): GH ↑↑, IGF-1 ↑↑ (excess), IGFBP3 ↑, pituitary adenoma; "
                "GPR101 (excess): GH ↑↑↑ (highest), IGF-1 ↑↑↑ (highest), infant; "
                "GHR LOF (Laron): GH ↑↑↑, IGF-1 absent, IGFBP3 absent, ALS absent; "
                "STAT5B LOF: GH ↑↑↑, IGF-1 absent, IGFBP3 absent + EOSINOPHILIA + autoimmune; "
                "IGF1 LOF: GH ↑↑↑, IGF-1 absent, IGFBP3 absent + SNHL + ID; "
                "IGF1R LOF: GH slightly ↑, IGF-1 normal-HIGH, IGFBP3 normal-high (resistance); "
                "IGFALS: GH mildly ↑, IGF-1 very LOW, IGFBP3 very LOW (mild height deficit); "
                "PAPPA2: GH ↑, total IGF-1 HIGH, IGFBP3 VERY HIGH, free IGF-1 LOW (trapped). "
                "CLINICAL SHORTCUT: "
                "IGF-1 HIGH + short stature → IGF1R or PAPPA2 (resistance or trapping); "
                "IGF-1 absent + short stature → GHR, IGF1 gene, or STAT5B; "
                "SNHL differentiates IGF1 gene from GHR/STAT5B; "
                "Eosinophilia differentiates STAT5B from GHR."
            ),
        },
    ]
    return {
        "atlas":  "Hereditary-GH-IGF1-Axis-Atlas",
        "count":  len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:600])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Terms: {df['count']}")
