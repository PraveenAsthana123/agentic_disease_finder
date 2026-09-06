#!/usr/bin/env python3
"""Hereditary-Immunodeficiency-Atlas — Complete 8-Gene Hereditary Primary Immunodeficiency Atlas
BTK     (Bruton's Tyrosine Kinase; 659 aa; ~76 kDa; Xq22.1; XL;
         OMIM gene 300300; XLA OMIM 300755;
         X-linked agammaglobulinemia; absent B cells + absent immunoglobulins;
         monthly IVIG LIFELONG; no live vaccines EVER;
         Ibrutinib BTK-inhibitor research in XLA;
         seed SEED_BASE+0) ·
RAG1    (Recombination Activating Gene 1; 1043 aa; ~119 kDa; 11p12; AR;
         OMIM gene 179615; Omenn/SCID OMIM 601457;
         V(D)J recombination arrest → spectrum SCID to Omenn syndrome
         (erythroderma, eosinophilia, elevated IgE) to leaky CID;
         HSCT curative; no live vaccines;
         seed SEED_BASE+1) ·
ADA     (Adenosine Deaminase; 363 aa; ~41 kDa; 20q13.11; AR;
         OMIM gene 608958; ADA-SCID OMIM 102700;
         metabolic SCID — dATP accumulates → lymphotoxic; purine salvage arrest;
         pegademase bovine (PEG-ADA) enzyme replacement;
         Strimvelis ADA gene therapy (EMA 2016 — first approved gene therapy
         for single-gene disorder); HSCT curative;
         seed SEED_BASE+2) ·
CYBB    (Cytochrome b-245 beta chain gp91phox; 570 aa; ~65 kDa; Xp21.1; XL;
         OMIM gene 300481; CGD OMIM 306400;
         chronic granulomatous disease; NADPH oxidase defect → absent respiratory burst;
         catalase-positive organisms (Aspergillus, Staph aureus, Serratia, Nocardia,
         Burkholderia cepacia); prophylactic itraconazole + TMP-SMX lifelong;
         IFN-gamma reduces infections 70%; HSCT curative in young;
         seed SEED_BASE+3) ·
WAS     (Wiskott-Aldrich Syndrome Protein; 502 aa; ~57 kDa; Xp11.23; XL;
         OMIM gene 300392; WAS OMIM 301000;
         classic triad: thrombocytopenia (small platelets) + eczema + immunodeficiency;
         WASP regulates actin polymerisation in haematopoietic cells;
         gene score 1-5 determines phenotype severity;
         HSCT curative; WAS gene therapy trials;
         seed SEED_BASE+4) ·
LRBA    (LPS-Responsive Beige-Like Anchor Protein; 2863 aa; ~321 kDa; 4q31.3; AR;
         OMIM gene 606453; LRBA deficiency OMIM 614700;
         CVID phenotype + autoimmunity + IBD + organomegaly;
         LRBA recycles CTLA4 from endosomes to cell surface;
         abatacept (CTLA4-Ig) restores CTLA4 signalling — dramatic response;
         IVIG + abatacept combination;
         seed SEED_BASE+5) ·
CTLA4   (Cytotoxic T Lymphocyte Antigen 4; 223 aa; ~25 kDa; 2q33.2; AD haploinsufficiency;
         OMIM gene 123890; CTLA4-HI OMIM 616100;
         CTLA4 haploinsufficiency; autoimmunity, CVID-like hypogammaglobulinemia,
         lymphoproliferation, granulomatous disease; Treg dysfunction;
         abatacept (CTLA4-Ig) SPECIFIC treatment — replaces missing CTLA4 function;
         sirolimus for lymphoproliferation; IVIG;
         seed SEED_BASE+6) ·
PIK3CD  (PI3-Kinase Catalytic Delta; 1044 aa; ~119 kDa; 1p36.22; AD GOF;
         OMIM gene 602839; APDS1 OMIM 615513;
         activated PI3K delta syndrome 1;
         GOF → constitutive AKT/mTOR signalling → T cell senescence
         + B cell maturation defect → susceptibility EBV/CMV herpesvirus infections;
         idelalisib PI3Kδ inhibitor clinical trial;
         leniolisib (OMGARD) FDA 2023 — first approved PI3Kδ inhibitor for APDS;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1534–1541)
"""

import random

SEED_BASE = 1534

IMMUNODEFICIENCY_GENES = [
    # ── BTK — X-linked Agammaglobulinemia ──
    {
        "gene": "BTK",
        "protein": "Bruton's Tyrosine Kinase — XLA, Absent B Cells, Monthly IVIG Lifelong, No Live Vaccines",
        "alias": (
            "BTK; OMIM gene 300300; XLA OMIM 300755; Xq22.1; 659 aa; ~76 kDa; "
            "BTK encodes Bruton's tyrosine kinase, a cytoplasmic non-receptor tyrosine "
            "kinase of the Tec family essential for B-cell development beyond the pre-B "
            "cell stage. BTK is activated downstream of the pre-B cell receptor (pre-BCR) "
            "and mature B cell receptor (BCR) signalling cascades via LYN, SYK, and "
            "PI3-kinase-mediated PIP3 production at the membrane. BTK contains five domains: "
            "PH (pleckstrin homology) — recruits BTK to membrane via PIP3; TH (Tec homology); "
            "SH3; SH2; and catalytic kinase domain. Loss-of-function variants (missense, "
            "truncating, splice-site — all causing absent or non-functional BTK protein) "
            "arrest B-cell maturation at the pro-B to pre-B transition in bone marrow. "
            "Result: complete absence of circulating B lymphocytes (CD19/CD20 negative), "
            "all immunoglobulin isotypes absent or markedly reduced (IgG <2 g/L, IgA and "
            "IgM undetectable), and no antigen-specific antibody responses. X-linked "
            "inheritance: almost exclusively affects males; female carriers are clinically "
            "unaffected (random X-inactivation favouring normal X in mature B cells provides "
            "sufficient BTK). Clinical onset: recurrent bacterial infections with encapsulated "
            "organisms (Streptococcus pneumoniae, Haemophilus influenzae, Neisseria "
            "meningitidis) beginning after maternal IgG wanes (age 6-18 months). "
            "Diagnosis: absent CD19+ B cells on flow cytometry + undetectable "
            "immunoglobulins + absent BTK protein on monocyte BTK protein assay + BTK "
            "mutation. Treatment: intravenous immunoglobulin (IVIG) replacement LIFELONG "
            "— every 3-4 weeks to maintain trough IgG >8 g/L; subcutaneous IG (SCIG) "
            "alternative. Critical safety rule: NO LIVE ATTENUATED VACCINES EVER — "
            "OPV (oral polio) has caused paralytic poliomyelitis; live viral vaccines "
            "(MMR, varicella, rotavirus, yellow fever) contraindicated. Echovirus/enterovirus "
            "meningoencephalitis is a late, severe complication. Ibrutinib (BTK kinase inhibitor) "
            "paradoxically used in lymphomas is under research as a BTK-pathway modifier in XLA."
        ),
        "aa": "659 aa",
        "kDa": "~76 kDa",
        "locus": "Xq22.1",
        "omim_gene": 300300,
        "omim_disease": 300755,
        "inheritance": "XL — X-linked recessive; males affected; female carriers unaffected (BTK monocyte protein assay detects carriers)",
        "gene_class": (
            "BTK is a 659-amino acid Tec-family non-receptor tyrosine kinase. Domain architecture: "
            "(1) N-terminal PH domain (aa 1-177) — binds phosphatidylinositol-3,4,5-trisphosphate "
            "(PIP3) generated by PI3Kδ/PI3Kγ downstream of BCR activation, recruiting BTK to "
            "inner plasma membrane leaflet; (2) TH (Tec homology) domain (aa 178-229) — "
            "proline-rich region binds SH3 domains for signalling complex assembly; "
            "(3) SH3 domain (aa 230-285) — protein-protein interaction; (4) SH2 domain "
            "(aa 286-382) — binds phosphotyrosines on LAB, BLNK, and other adaptor proteins; "
            "(5) kinase domain (aa 383-659) — catalytic domain; Thr316 auto-phosphorylation "
            "in PH-TH region and Tyr551 in activation loop of kinase domain are required "
            "for full BTK activation. In BCR signalling: Antigen → BCR crosslinking → "
            "LYN-mediated phosphorylation → CD19/PI3Kδ → PIP3 at membrane → BTK-PH "
            "membrane recruitment → BTK Tyr551 phosphorylation by LYN → BTK "
            "auto-phosphorylation Tyr223 → BLNK scaffold → PLCγ2 activation → "
            "DAG + IP3 → PKC + Ca2+ influx → NF-κB + NFAT transcription → B cell "
            "survival, proliferation, differentiation. Variant distribution in XLA: "
            "~40% missense (kinase domain most common), ~20% splice-site, ~20% nonsense, "
            "~15% small deletions/insertions, ~5% large deletions. BTK protein absent in "
            "monocytes/platelets — Western blot or flow cytometry of BTK protein in "
            "peripheral monocytes is a rapid diagnostic screen and carrier detection test. "
            "Genotype-phenotype: no strict correlation — even hypomorphic variants can cause "
            "full XLA; some PH domain missense → less severe phenotype (partial XLA/CID)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("BTK kinase domain missense XL — absent BTK protein, classic XLA, recurrent bacterial infections", 0.40),
            ("BTK truncating (nonsense/frameshift) XL — absent B cells, severe XLA, early onset <12 months", 0.30),
            ("BTK splice-site XL — reduced/absent BTK, moderate-severe XLA", 0.20),
            ("BTK PH domain missense XL — partial XLA, CID phenotype, some residual B cells", 0.10),
        ],
        "key_alerts": [
            "BTK-IVIG-LIFELONG-MANDATORY: IVIG replacement is LIFELONG in XLA — every 3-4 weeks to maintain trough IgG >8 g/L; NEVER discontinue; failure to maintain trough levels → recurrent pneumonia, bronchiectasis, irreversible lung damage",
            "BTK-NO-LIVE-VACCINES-EVER: Live attenuated vaccines ABSOLUTELY CONTRAINDICATED in XLA — oral poliovirus vaccine (OPV) has caused paralytic poliomyelitis; MMR, varicella, rotavirus, yellow fever, live typhoid all contraindicated; INACTIVATED vaccines safe and recommended",
            "BTK-ECHOVIRUS-MENINGOENCEPHALITIS: Chronic enterovirus/echovirus meningoencephalitis is a late, devastating complication of XLA — presents as progressive neurological decline; diagnose by CSF viral PCR; high-dose IVIG + intrathecal IgG experimental",
            "BTK-MONOCYTE-PROTEIN-ASSAY-CARRIER: BTK protein absent in monocytes of XLA patients — peripheral blood monocyte BTK Western blot / flow cytometry is a rapid screening test AND identifies female carriers (mosaic BTK expression); request before full sequencing when flow shows absent B cells",
            "BTK-B-CELL-ABSENT-FLOW: Absent CD19+/CD20+ B cells on peripheral blood flow cytometry (<2% of lymphocytes) is the cardinal diagnostic finding in XLA; complement with absent serum immunoglobulins; normal or elevated T cells and NK cells",
            "BTK-BRONCHIECTASIS-SURVEILLANCE: Long-term XLA on IVIG → chronic sinopulmonary infections → bronchiectasis in 50% by adulthood; annual HRCT chest from age 10 years; chest physiotherapy; intensive antibiotic treatment of all pulmonary infections",
        ],
    },
    # ── RAG1 — Omenn Syndrome / SCID / Combined Immunodeficiency ──
    {
        "gene": "RAG1",
        "protein": "RAG1 — V(D)J Recombination, Spectrum SCID to Omenn Syndrome to Leaky CID",
        "alias": (
            "RAG1; OMIM gene 179615; Omenn OMIM 603554 / SCID OMIM 601457; 11p12; 1043 aa; ~119 kDa; "
            "RAG1 encodes Recombination-Activating Gene 1, the catalytic endonuclease "
            "component of the RAG1/RAG2 recombinase complex that initiates V(D)J "
            "recombination — the somatic DNA rearrangement process that assembles the "
            "immunoglobulin heavy chain, immunoglobulin light chain, and T cell receptor "
            "alpha/beta/gamma/delta genes from germline V, D, and J gene segments. "
            "RAG1 (plus RAG2) introduces double-strand DNA breaks at recombination signal "
            "sequences (RSS) flanking V, D, and J segments; hairpin intermediates are "
            "resolved by non-homologous end-joining (NHEJ) factors (Ku70/80, DNA-PKcs, "
            "Artemis, XRCC4, LigaseIV). Without RAG1 activity, neither B-cell "
            "immunoglobulin genes nor T-cell receptor genes can be assembled → combined "
            "B-cell and T-cell developmental arrest. Clinical spectrum of RAG1 variants "
            "is uniquely broad: (1) Null variants (frameshift, nonsense, large deletion) → "
            "complete absence of all mature B and T lymphocytes → classic SCID (T-B- NK+ SCID) — "
            "the most severe combined immunodeficiency; presents within weeks of birth with "
            "infections, failure to thrive; (2) Hypomorphic missense variants → partial "
            "RAG1 activity → partial V(D)J recombination → oligoclonal (restricted) T cells "
            "capable of peripheral expansion; few B cells, elevated IgE → Omenn syndrome: "
            "erythroderma/generalised rash (95%), hepatosplenomegaly, lymphadenopathy, "
            "eosinophilia, elevated IgE (despite absent other Ig isotypes), failure to "
            "thrive; oligoclonal activated T cells infiltrate skin and gut; (3) Intermediate "
            "variants → combined immunodeficiency (CID) without full Omenn phenotype. "
            "Diagnosis: absent T and B cells (TREC/KREC NEWBORN SCREENING detects SCID); "
            "Omenn: elevated IgE, eosinophilia, erythroderma, restricted TCR Vbeta spectratype. "
            "Treatment: HSCT is curative for all forms; conditioning required; RAG gene therapy "
            "under development. No live vaccines. IVIG support until HSCT."
        ),
        "aa": "1043 aa",
        "kDa": "~119 kDa",
        "locus": "11p12",
        "omim_gene": 179615,
        "omim_disease": 601457,
        "inheritance": "AR — biallelic loss-of-function (null SCID); biallelic hypomorphic missense → Omenn/CID phenotype",
        "gene_class": (
            "RAG1 is a 1043-amino acid multidomain protein forming a heterotetrameric "
            "RAG1/RAG2/RAG1/RAG2 synaptic complex. Structural domains: (1) N-terminal "
            "ubiquitin-like domain (UBL, aa 1-218) — autoinhibitory; binds histone H3K4me3 "
            "via RAG2 PHD domain for chromatin targeting; (2) central domain (aa 219-383) — "
            "contributes to RAG1 dimerisation and RAG2 interaction; zinc-binding RING domain "
            "(aa 265-380) with E3 ubiquitin ligase activity for histone H3 ubiquitylation; "
            "(3) core RAG1 (aa 384-1008) — essential for catalytic activity; contains the "
            "nonamer-binding domain (NBD, aa 389-464) that contacts the conserved ACAAAAACC "
            "nonamer in RSS; the catalytic RNH (RNase H-like) fold with the DDE triad "
            "(Asp600, Asp708, Glu962) that performs strand cleavage; (4) C-terminal homeodomain "
            "(aa 1009-1040) — contributes to RSS recognition. Mechanism: RAG1/RAG2 "
            "recognises 12-RSS and 23-RSS sequences flanking V, D, J segments (12/23 rule) → "
            "synaptic complex formation → single-strand nicking at RSS border → hairpin "
            "formation on coding end → NHEJ opening and joining → covalently sealed V(D)J "
            "joint. Hypomorphic RAG1 variants: residual ~1-5% recombination activity → "
            "oligoclonal T cells escape thymic selection and cause Omenn syndrome; the same "
            "variants in compound heterozygosity with null alleles produce intermediate CID. "
            "Structural studies show Omenn missense variants concentrate at NBD-RSS interface, "
            "catalytic DDE residues, and RAG2 dimer interface — explaining partial activity. "
            "RAG2 variants produce an indistinguishable phenotype spectrum (RAG2 also null → "
            "SCID; hypomorphic → Omenn)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("RAG1 biallelic null — classic T-B- NK+ SCID, absent T and B cells, neonatal onset", 0.40),
            ("RAG1 biallelic hypomorphic missense — Omenn syndrome, erythroderma, eosinophilia, elevated IgE", 0.30),
            ("RAG1 compound het null/hypomorphic — combined immunodeficiency (CID), partial T/B cells", 0.20),
            ("RAG1 biallelic partial loss — leaky SCID, delayed-onset CID, recurrent sinopulmonary infections", 0.10),
        ],
        "key_alerts": [
            "RAG1-NEWBORN-SCREENING-TREC: SCID caused by RAG1 biallelic null variants is DETECTED by newborn screening via TREC (T-cell receptor excision circles) quantitation — absent TRECs → immediate immunological work-up; TREC screening saves lives by enabling HSCT before infections",
            "RAG1-OMENN-ERYTHRODERMA-DIAGNOSIS: Omenn syndrome presents with generalised erythroderma, hepatosplenomegaly, lymphadenopathy, eosinophilia, elevated IgE — DESPITE absent IgG/IgA/IgM; distinguish from Netherton syndrome, GVHD, and IPEX; RAG1/RAG2 sequencing mandatory",
            "RAG1-HSCT-CURATIVE-URGENT: HSCT is the curative treatment for ALL forms of RAG1 immunodeficiency — SCID and Omenn; pre-HSCT conditioning required (unlike some other SCID forms); refer to PID/HSCT centre IMMEDIATELY on diagnosis; delay increases infection burden and mortality",
            "RAG1-NO-LIVE-VACCINES: Live attenuated vaccines ABSOLUTELY CONTRAINDICATED — BCG (bacille Calmette-Guérin) administered at birth in many countries can cause disseminated BCG disease in SCID; if BCG given before diagnosis → screen for BCG-osis and treat with anti-mycobacterials",
            "RAG1-IVIG-BRIDGE-TO-HSCT: IVIG replacement as a bridge to HSCT — maintains passive immunity; does NOT treat the underlying T-cell immunodeficiency; Omenn patients require immunosuppression (cyclosporin + steroids) pre-HSCT to control oligoclonal T-cell-mediated inflammation",
            "RAG1-GENE-SCORE-SPECTRUM: RAG1 genotype predicts phenotype — null/null → SCID; hypomorphic/hypomorphic → Omenn; null/hypomorphic → CID; request full RAG1 + RAG2 sequencing including large deletions (MLPA) to characterise both alleles and predict severity before HSCT conditioning",
        ],
    },
    # ── ADA — ADA-SCID ──
    {
        "gene": "ADA",
        "protein": "Adenosine Deaminase — ADA-SCID, dATP Lymphotoxicity, Strimvelis Gene Therapy EMA 2016",
        "alias": (
            "ADA; OMIM gene 608958; ADA-SCID OMIM 102700; 20q13.11; 363 aa; ~41 kDa; "
            "ADA encodes adenosine deaminase, a purine salvage pathway enzyme that "
            "catalyses the irreversible deamination of adenosine and deoxyadenosine to "
            "inosine and deoxyinosine respectively. ADA deficiency causes metabolic SCID "
            "through a distinct mechanism from RAG1: accumulation of deoxyadenosine "
            "(dAdo) in cells → phosphorylation by intracellular kinases → dATP "
            "accumulation → severe lymphotoxicity. The key metabolic events: dAdo → "
            "inhibits S-adenosylhomocysteine hydrolase (SAH hydrolase) → accumulation "
            "of S-adenosylhomocysteine → transmethylation inhibition; dAdo phosphorylated "
            "to dATP → dATP accumulates to extraordinarily high levels in lymphocytes "
            "(which uniquely express deoxycytidine kinase with high dAdo affinity) → "
            "dATP-mediated inhibition of ribonucleotide reductase → impaired DNA synthesis "
            "→ lymphocyte apoptosis. T cells are most sensitive, followed by B cells and NK "
            "cells → severe combined immunodeficiency. Clinical: T-B- NK- or T-B- NK+ SCID "
            "depending on residual ADA activity; autosomal recessive; affects both sexes. "
            "Unique features: skeletal abnormalities (chondro-osseous dysplasia — abnormal "
            "costochondral junctions visible on chest X-ray in 50%); elevated dATP in "
            "erythrocytes is a diagnostic marker. Treatment options: (1) HSCT — curative "
            "if matched sibling available; (2) PEG-ADA (pegademase bovine) — intramuscular "
            "enzyme replacement, corrects metabolic toxicity partially, allows interim "
            "immune reconstitution; (3) Gene therapy: Strimvelis (autologous CD34+ "
            "haematopoietic stem cells transduced with ADA-expressing gamma-retroviral vector) "
            "approved by EMA May 2016 — the first conditionally approved gene therapy for a "
            "single-gene primary immunodeficiency; manufactured by Orchard Therapeutics at "
            "a single centre (Milan); superior to PEG-ADA long-term; no graft-vs-host disease risk."
        ),
        "aa": "363 aa",
        "kDa": "~41 kDa",
        "locus": "20q13.11",
        "omim_gene": 608958,
        "omim_disease": 102700,
        "inheritance": "AR — autosomal recessive; biallelic loss-of-function causes complete ADA-SCID; hypomorphic → partial/late-onset",
        "gene_class": (
            "ADA is a 363-amino acid enzyme of the purine salvage pathway. It functions "
            "as a homodimer (each monomer ~41 kDa) of the alpha/beta barrel (TIM barrel) "
            "superfamily. Catalytic mechanism: Zn2+-dependent deamination; the active site "
            "contains a binuclear zinc centre that activates water for nucleophilic attack "
            "on the C6-amino group of adenosine/deoxyadenosine — converting the 6-amino "
            "group to a 6-hydroxyl (keto) group, releasing ammonia and generating inosine/ "
            "deoxyinosine. Tissue expression: highest in lymphocytes (10-15x higher than "
            "red blood cells) — explaining lymphocyte-selective toxicity of ADA deficiency. "
            "ADA is also expressed as a cell surface ecto-enzyme complexed with DPP4/CD26 "
            "(dipeptidyl peptidase IV), where it modulates adenosine signalling in the "
            "extracellular microenvironment. Variant distribution: >60 pathogenic variants "
            "documented; missense variants most common (Arg101Trp, Gly216Arg, Arg211His "
            "among the more prevalent); null variants (nonsense, frameshift) → complete "
            "deficiency → classic neonatal SCID; hypomorphic missense (Glu217Lys) → "
            "residual 1-5% activity → delayed-onset or partial SCID (presenting in late "
            "childhood or adulthood with recurrent infections and declining lymphocytes). "
            "Erythrocyte dATP level is the primary metabolic monitoring marker for "
            "PEG-ADA therapy (target: dATP <0.001 μmol/mL RBC). ADA2 (encoded by CECR1) "
            "is a separate enzyme causing ADA2 deficiency (DADA2) — a vasculitis/autoinflammatory "
            "syndrome with polyarteritis nodosa-like features and stroke, NOT immunodeficiency."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("ADA biallelic null — classic neonatal T-B- NK- SCID, dATP accumulation, absent lymphocytes", 0.40),
            ("ADA biallelic missense (hypomorphic) — late-onset SCID, progressive lymphopenia childhood", 0.25),
            ("ADA compound het null/missense — moderate ADA-SCID, partial immune function, PEG-ADA responsive", 0.25),
            ("ADA biallelic partial — 'leaky' ADA-SCID, adult-onset lymphopenia, recurrent opportunistic infections", 0.10),
        ],
        "key_alerts": [
            "ADA-STRIMVELIS-GENE-THERAPY-EMA-2016: Strimvelis (autologous CD34+ HSC + ADA gamma-retroviral vector) is the EMA-approved gene therapy for ADA-SCID — first approved gene therapy for a PIDs; curative without GvHD risk; manufactured at single centre (Milan); consider when no matched sibling donor available",
            "ADA-PEG-ADA-ENZYME-REPLACEMENT: Pegademase bovine (PEG-ADA) IM weekly/biweekly — corrects metabolic dATP toxicity; allows partial immune reconstitution; use as bridge to HSCT or gene therapy; monitor erythrocyte dATP (target <0.001 μmol/mL RBC) and lymphocyte counts monthly",
            "ADA-DATP-METABOLIC-MONITORING: dATP accumulation in erythrocytes is the metabolic diagnostic marker — measure erythrocyte dATP by HPLC; elevated dATP confirms ADA deficiency; monitoring on PEG-ADA: dATP should normalise; failure to normalise → insufficient PEG-ADA dosing",
            "ADA-SKELETAL-CHONDRO-OSSEOUS: Chondro-osseous dysplasia (abnormal costochondral junctions, metaphyseal irregularities) on chest X-ray in ~50% of ADA-SCID — a diagnostic clue distinguishing ADA-SCID from other SCID forms; radiological finding unique to ADA deficiency",
            "ADA-LATE-ONSET-ADULT-MIMICRY: Hypomorphic ADA variants → delayed-onset ADA deficiency presenting in adulthood with progressive lymphopenia, recurrent infections, and declining immunoglobulins — may mimic CVID; measure ADA enzyme activity in erythrocytes before CVID diagnosis in unexplained progressive lymphopenia",
            "ADA-NO-LIVE-VACCINES: Live vaccines ABSOLUTELY CONTRAINDICATED; BCG-osis risk if BCG given at birth before diagnosis; on PEG-ADA or gene therapy, immune reconstitution should be documented (T cell counts, lymphoproliferative responses, serology) before any vaccine decisions",
        ],
    },
    # ── CYBB — Chronic Granulomatous Disease ──
    {
        "gene": "CYBB",
        "protein": "gp91phox / Cytochrome b-245 Beta — CGD, Absent Respiratory Burst, Catalase-Positive Organisms",
        "alias": (
            "CYBB; OMIM gene 300481; CGD OMIM 306400; Xp21.1; 570 aa; ~65 kDa; "
            "CYBB encodes gp91phox (glycoprotein 91 kDa of phagocyte oxidase), the "
            "beta subunit and catalytic core of the NADPH oxidase complex (NOX2). "
            "CYBB mutations cause the most common form of chronic granulomatous disease "
            "(CGD), accounting for approximately 65-70% of all CGD cases. The NADPH "
            "oxidase complex is the primary mechanism by which phagocytes (neutrophils, "
            "macrophages, monocytes, eosinophils) generate reactive oxygen species (ROS) "
            "to kill ingested pathogens. gp91phox forms a heterodimer with p22phox "
            "(CYBA) in the phagosomal and plasma membranes; upon phagocyte activation, "
            "the cytosolic components p47phox (NCF1), p67phox (NCF2), p40phox (NCF4), "
            "and Rac2 (RHOG2) translocate to the membrane and assemble the active oxidase, "
            "which transfers electrons from cytosolic NADPH to molecular oxygen across the "
            "membrane, producing superoxide (O2-) in the phagosome lumen. Superoxide "
            "is subsequently converted to hydrogen peroxide (H2O2), hydroxyl radical "
            "(OH•), and hypochlorous acid (HOCl, via myeloperoxidase) — collectively "
            "killing engulfed microorganisms. In CGD: absent respiratory burst → inability "
            "to kill CATALASE-POSITIVE organisms (those that destroy their own H2O2, "
            "thereby evading the H2O2-mediated killing that partially compensates for "
            "absent O2- in CGD). Classical susceptibility organisms: Aspergillus species "
            "(most dangerous — pulmonary aspergillosis, invasive aspergillosis), "
            "Staphylococcus aureus (skin abscesses, lymphadenitis, osteomyelitis), "
            "Serratia marcescens, Nocardia, Burkholderia cepacia complex (highly lethal "
            "in CGD — can cause rapidly fatal sepsis), Chromobacterium violaceum "
            "(tropical CGD), Candida (less common). Catalase-negative organisms "
            "(Streptococcus, Haemophilus) rarely cause infections — their own H2O2 "
            "contributes to oxidative killing even in CGD."
        ),
        "aa": "570 aa",
        "kDa": "~65 kDa",
        "locus": "Xp21.1",
        "omim_gene": 300481,
        "omim_disease": 306400,
        "inheritance": "XL — X-linked recessive (CYBB); AR forms: CYBA (p22phox), NCF1 (p47phox), NCF2 (p67phox), NCF4 (p40phox)",
        "gene_class": (
            "gp91phox (CYBB) is a 570-amino acid integral membrane glycoprotein and the "
            "catalytic component of NOX2. Structure: six transmembrane helices (TM1-TM6) "
            "with two heme groups (Fe3+/Fe2+) at fixed potentials coordinated by "
            "histidine residues in TM3 (His101) and TM5 (His209, His222, His281) — forming "
            "a bishistidyl heme bridge at the outer TM3-TM5 interface; a large cytosolic "
            "C-terminal domain (aa 290-570) containing the FAD-binding domain and "
            "NADPH-binding domain. Electron transport chain: NADPH (cytosolic) → FAD "
            "(gp91phox C-terminal) → heme 1 → heme 2 → O2 (phagosome lumen) → O2-. "
            "gp91phox is heavily N-glycosylated (N-glycans on ectodomains at Asn132, "
            "Asn149, Asn240, Asn265, Asn303) — N-glycosylation is required for "
            "membrane targeting and p22phox stabilisation; unglycosylated gp91phox is "
            "retained in ER and rapidly degraded. gp91phox and p22phox (CYBA) are "
            "obligate heterodimers — absence of either causes degradation of the other "
            "(explaining why CYBA mutations phenocopy gp91phox deficiency). DHR "
            "(dihydrorhodamine 123) oxidation assay by flow cytometry is the gold-standard "
            "CGD diagnostic test — neutrophils from CGD patients show absent or markedly "
            "reduced DHR fluorescence after PMA stimulation vs robust oxidative burst in "
            "healthy controls. X-linked CGD: gp91phox protein absent in Western blot; "
            "female carriers show mosaic DHR oxidation (bimodal distribution — "
            "proportion of negative cells reflects X-inactivation skewing). AR CGD "
            "variants (NCF1 most common AR form, ~25% of CGD): DHR absent; gp91phox "
            "protein present but inactive (cytosolic component missing)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("CYBB null/truncating XL — absent gp91phox, classic severe CGD, recurrent Aspergillus/Staph infections", 0.45),
            ("CYBB missense XL — absent/reduced gp91phox, CGD, variable severity based on residual oxidase activity", 0.30),
            ("CYBB splice-site XL — reduced gp91phox, intermediate CGD severity, some residual respiratory burst", 0.15),
            ("NCF1/CYBA AR — absent p47phox/p22phox, CGD phenotype identical to CYBB, AR inheritance", 0.10),
        ],
        "key_alerts": [
            "CYBB-ASPERGILLUS-MOST-DANGEROUS: Aspergillus is the MOST DANGEROUS pathogen in CGD — invasive pulmonary aspergillosis (IPA) has 30-40% mortality in CGD despite treatment; CT chest at first fever >38.5°C in any CGD patient; empiric voriconazole + specialist consultation; lifelong itraconazole prophylaxis MANDATORY",
            "CYBB-BURKHOLDERIA-CEPACIA-LETHAL: Burkholderia cepacia complex causes RAPIDLY FATAL sepsis in CGD — intrinsically resistant to many antibiotics; if isolated from any CGD patient → emergency combination antibiotics (meropenem + TMP-SMX ± minocycline); notify CGD specialist immediately",
            "CYBB-PROPHYLAXIS-LIFELONG-MANDATORY: Lifelong antifungal prophylaxis (itraconazole 100-200 mg/day) AND antibacterial prophylaxis (TMP-SMX 5 mg/kg/day) MANDATORY in ALL CGD patients from diagnosis; prophylaxis reduces infection frequency by >50%; never discontinue prophylaxis",
            "CYBB-IFN-GAMMA-REDUCES-INFECTIONS-70PCT: IFN-gamma (Actimmune) subcutaneous 3x/week reduces serious infections by ~70% in CGD — mechanism involves upregulation of residual NADPH oxidase activity and alternative antimicrobial pathways; recommended as adjunctive therapy particularly in severe CGD",
            "CYBB-DHR-ASSAY-DIAGNOSTIC: Dihydrorhodamine (DHR) oxidation flow cytometry is the gold-standard CGD diagnostic test — PMA-stimulated neutrophils show absent DHR fluorescence shift in CGD (vs bright shift in normal); mosaic DHR in female CYBB carriers (X-inactivation); request DHR before genetic confirmation",
            "CYBB-HSCT-CURATIVE-YOUNG: HSCT is CURATIVE for CGD — consider in young patients with severe/frequently infected CGD, inflammatory complications (colitis, obstructive granulomas), or after life-threatening infections; gene therapy (lentiviral gp91phox) in clinical trials with early promising results",
        ],
    },
    # ── WAS — Wiskott-Aldrich Syndrome ──
    {
        "gene": "WAS",
        "protein": "WASP — Wiskott-Aldrich Syndrome, Triad Thrombocytopenia+Eczema+Immunodeficiency, HSCT Curative",
        "alias": (
            "WAS; OMIM gene 300392; WAS OMIM 301000; Xp11.23; 502 aa; ~57 kDa; "
            "WAS encodes the Wiskott-Aldrich syndrome protein (WASP), a cytoplasmic "
            "scaffolding/actin nucleation-promoting factor expressed exclusively in "
            "haematopoietic cells (lymphocytes, platelets, monocytes, neutrophils, "
            "dendritic cells, NK cells). WASP is an essential regulator of actin "
            "polymerisation downstream of surface receptor signalling in all haematopoietic "
            "lineages. Loss-of-function WAS variants cause Wiskott-Aldrich syndrome (WAS), "
            "with the classic clinical TRIAD: (1) THROMBOCYTOPENIA with SMALL PLATELETS "
            "(microthromobocytopenia) — platelet count typically 20,000-80,000/μL; platelet "
            "volume (MPV) LOW (4-5 fL vs normal 7-11 fL); small platelet size is unique "
            "and near-pathognomonic; autoimmune platelet destruction component (splenic) "
            "also contributes; life-threatening bleeding episodes (intracranial haemorrhage "
            "in 1-10%); (2) ECZEMA — atopic dermatitis-like, often severe, refractory to "
            "standard treatment, begins in infancy; driven by T regulatory cell dysfunction "
            "and Th2 skewing; (3) IMMUNODEFICIENCY — combined B and T cell dysfunction; "
            "progressive decline in lymphocyte numbers and function with age; poor antibody "
            "responses to polysaccharide antigens; recurrent bacterial otitis media, "
            "pneumonia, sinusitis; viral infections (herpesviruses, CMV, EBV); susceptibility "
            "to P. jirovecii pneumonia (PJP); elevated IgA/IgE but low IgM. WAS gene score "
            "1-5 correlates with clinical severity based on variant type and WASP expression: "
            "score 1-2 (missense, partial WASP) → X-linked thrombocytopenia (XLT) — "
            "thrombocytopenia predominant, milder immunodeficiency; score 3-5 (null, absent "
            "WASP) → classic WAS with full triad + autoimmune complications + lymphoma risk. "
            "HSCT is curative for all forms. WAS gene therapy (lentiviral) shows excellent "
            "results in clinical trials."
        ),
        "aa": "502 aa",
        "kDa": "~57 kDa",
        "locus": "Xp11.23",
        "omim_gene": 300392,
        "omim_disease": 301000,
        "inheritance": "XL — X-linked recessive; WAS gene score 1-5 determines severity (XLT vs classic WAS vs severe WAS)",
        "gene_class": (
            "WASP is a 502-amino acid multi-domain scaffold and actin nucleation-promoting "
            "factor (NPF). Domain architecture: (1) N-terminal WASP homology 1 domain "
            "(WH1/EVH1, aa 1-106) — binds WASp-interacting protein (WIP/WIPF1), which "
            "stabilises WASP and prevents its degradation; (2) basic region (BR, aa 107-170) — "
            "binds phosphoinositides (PIP2) and TOCA1 for membrane localisation; contains "
            "GTPase-binding domain (GBD/CRIB, aa 201-321) — binds active Cdc42-GTP, "
            "releasing autoinhibitory conformation; (3) polyproline (PP) region (aa 322-400) — "
            "binds SH3-domain adaptor proteins (NCK, FYN, GRB2, ITK, PSTPIP1) for signalling "
            "complex assembly; (4) VCA (verprolin homology-central-acidic) domain (aa 401-502) — "
            "binds Arp2/3 complex and G-actin; activates Arp2/3 to nucleate branched F-actin "
            "networks. Activation mechanism: resting WASP is autoinhibited (GBD folds back onto "
            "VCA, masking Arp2/3-binding); Cdc42-GTP binding to GBD + PIP2 binding to BR → "
            "conformational opening → VCA exposed → Arp2/3 activation → actin branching. "
            "Cellular functions: immunological synapse formation (TCR and BCR signalling), "
            "platelet cytoskeletal organisation (platelet spreading and activation), NK cell "
            "cytotoxic lytic granule polarisation, dendritic cell migration and podosome "
            "formation. WAS phenotype of platelets: absent WASP → impaired platelet "
            "cytoskeletal dynamics → small, fragile platelets with accelerated splenic "
            "destruction and impaired megakaryocyte proplatelet formation. Small platelet "
            "size (MPV <5 fL) + low platelet count = pathognomonic; normal MPV in ITP "
            "distinguishes ITP from XLT/WAS. WASP protein expression in lymphocytes "
            "by flow cytometry is a rapid functional screen."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("WAS null/absent WASP — score 3-5, classic WAS triad (thrombocytopenia + eczema + immunodeficiency), autoimmunity", 0.40),
            ("WAS missense partial WASP — score 1-2, X-linked thrombocytopenia (XLT), mild-moderate phenotype", 0.30),
            ("WAS truncating/splice-site — score 4-5, severe WAS, lymphoma risk, autoimmune haemolytic anaemia", 0.20),
            ("WAS GOF missense (gain of function) — X-linked neutropenia (XLN), constitutive WASP activation", 0.10),
        ],
        "key_alerts": [
            "WAS-SMALL-PLATELETS-PATHOGNOMONIC: Small platelet size (MPV <5 fL) + thrombocytopenia is near-PATHOGNOMONIC for WAS/XLT — normal or large platelets in ITP; ALWAYS measure MPV in thrombocytopenic male infants; do NOT diagnose ITP in a male infant without ruling out WAS",
            "WAS-INTRACRANIAL-HAEMORRHAGE-RISK: Intracranial haemorrhage (ICH) occurs in 1-10% of WAS patients — leading cause of death in unsupported patients; URGENT platelet transfusion for ICH; low threshold for prophylactic platelet transfusion before procedures; HSCT eliminates ICH risk",
            "WAS-HSCT-CURATIVE-ALL-FORMS: HSCT is CURATIVE for classic WAS AND XLT — corrects thrombocytopenia, eczema, immunodeficiency, and autoimmune complications; best outcomes with matched sibling donor (OS >90%) or MUD before age 5; gene therapy (lentiviral WAS) shows equivalent results in trials",
            "WAS-GENE-SCORE-PREDICTS-THERAPY: WAS gene score 1-2 (XLT) → may consider watchful waiting for mild cases; score 3-5 (classic WAS) → HSCT recommended early; GOF variants (score 5 XLN) → different management; always classify by WASP protein expression + clinical score before therapy decision",
            "WAS-AUTOIMMUNITY-LATE-COMPLICATION: Autoimmune complications (haemolytic anaemia, neutropenia, vasculitis, nephritis, inflammatory bowel disease) develop in ~70% of classic WAS with increasing age — driven by Treg dysfunction; rituximab + IVIG for autoimmune cytopenias; HSCT prevents autoimmune progression",
            "WAS-EBV-LYMPHOMA-SURVEILLANCE: Classic WAS (score 3-5) has 10-22% lifetime risk of EBV-associated B-cell lymphoma — annual EBV PCR in peripheral blood; low threshold for LDH + imaging; rituximab for EBV-driven lymphoproliferation; HSCT before lymphoma development is strongly preferred",
        ],
    },
    # ── LRBA — LRBA Deficiency ──
    {
        "gene": "LRBA",
        "protein": "LRBA — CVID-like + Autoimmunity + IBD + Organomegaly, CTLA4 Recycling Defect, Abatacept Response",
        "alias": (
            "LRBA; OMIM gene 606453; LRBA deficiency OMIM 614700; 4q31.3; 2863 aa; ~321 kDa; "
            "LRBA encodes LPS-responsive beige-like anchor protein, a member of the BEACH "
            "(beige and Chediak-Higashi) domain-containing protein family involved in "
            "vesicular trafficking and endosomal recycling. LRBA is essential for the "
            "recycling of CTLA4 (CD152) from late endosomal compartments back to the cell "
            "surface in regulatory T cells (Tregs) and activated effector T cells. The "
            "mechanistic pathway: CTLA4 is constitutively internalised from the plasma "
            "membrane via clathrin-mediated endocytosis into early endosomes → normally "
            "LRBA recruits the retromer complex (VPS35/VPS26/VPS29) and AP1 to CTLA4-"
            "containing endosomes → CTLA4 is recycled back to the cell surface for "
            "continued B7 ligand downregulation on antigen-presenting cells. Without LRBA: "
            "internalised CTLA4 is shunted to lysosomes for degradation rather than "
            "recycled → net reduction in surface CTLA4 expression on Tregs and activated "
            "T cells despite normal CTLA4 mRNA. The consequence is functional CTLA4 "
            "haploinsufficiency — identical to CTLA4 haploinsufficiency syndrome "
            "(heterozygous CTLA4 mutations). Clinical phenotype of LRBA deficiency is "
            "a primary immunodeficiency WITH prominent autoimmunity: (1) "
            "Hypogammaglobulinaemia (CVID-like) — low IgG, IgA, IgM; recurrent bacterial "
            "infections; (2) Autoimmunity — autoimmune haemolytic anaemia, autoimmune "
            "thrombocytopenia (ITP), type 1 diabetes, thyroiditis, hepatitis; (3) "
            "Inflammatory bowel disease — severe Crohn's-like or UC-like intestinal "
            "inflammation in >50%; (4) Organomegaly — splenomegaly, hepatomegaly, "
            "lymphadenopathy, lymphoproliferation; (5) Granulomatous disease — granulomata "
            "in lung, gut, liver. Critical treatment insight: abatacept (CTLA4-Ig, "
            "Orencia) — a fusion protein of CTLA4 ectodomain and IgG1-Fc — directly "
            "restores CTLA4 signalling by binding and downregulating B7.1/CD80 and "
            "B7.2/CD86 on APCs, bypassing the LRBA recycling defect; dramatic clinical "
            "responses reported, including resolution of IBD, autoimmune cytopenias, "
            "and lymphoproliferation."
        ),
        "aa": "2863 aa",
        "kDa": "~321 kDa",
        "locus": "4q31.3",
        "omim_gene": 606453,
        "omim_disease": 614700,
        "inheritance": "AR — autosomal recessive biallelic loss-of-function; LRBA is one of the largest PIDs genes (2863 aa)",
        "gene_class": (
            "LRBA is a 2863-amino acid protein organised around a central BEACH (beige "
            "and Chediak-Higashi) domain characteristic of the BEACH-WD40 superfamily. "
            "Domain architecture: (1) N-terminal ARM/HEAT repeats (aa 1-800) — predicted "
            "protein-protein interaction scaffold; (2) DUF domain (aa 801-900); (3) "
            "BEACH domain (aa 2347-2507) — the defining structural module of the family; "
            "in Chediak-Higashi protein (LYST), BEACH mediates vesicle fusion; in LRBA, "
            "BEACH mediates retromer/AP1 recruitment to CTLA4-endosomes; (4) WD40 repeats "
            "(aa 2508-2863) — seven-bladed beta-propeller; binds phosphoinositide-enriched "
            "endosomal membranes. LRBA colocalises with CTLA4 in Rab8+/Rab11+ recycling "
            "endosomal compartments; LRBA-deficient T cells retain CTLA4 in LAMP1+ "
            "lysosomal compartments rather than recycling to the surface. Western blot "
            "for LRBA in PBMCs is a rapid functional diagnostic screen (absent band "
            "confirms biallelic LoF). Genetic diagnosis: LRBA is 2863 aa (~60 exons); "
            "missense, truncating, and splice variants distributed across the gene; no "
            "mutational hotspot; large deletions reported. Differential diagnosis: LRBA "
            "deficiency vs CVID — LRBA lacks CVID's predominant B-cell maturation block; "
            "LRBA has early-onset (~3-5 years), more severe autoimmunity, and prominent "
            "IBD vs typical CVID onset in 2nd-3rd decade; abatacept response distinguishes "
            "LRBA/CTLA4-HI from standard CVID which shows little abatacept benefit."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("LRBA biallelic truncating — absent LRBA protein, severe CVID+autoimmunity+IBD, childhood onset", 0.45),
            ("LRBA biallelic missense (BEACH domain) — absent/reduced LRBA, moderate phenotype, IBD+lymphoproliferation", 0.30),
            ("LRBA compound het — CVID-like with autoimmune haemolytic anaemia, ITP, granulomatous lung disease", 0.15),
            ("LRBA splice-site biallelic — reduced LRBA expression, intermediate phenotype, late-onset CVID+autoimmunity", 0.10),
        ],
        "key_alerts": [
            "LRBA-ABATACEPT-DRAMATIC-RESPONSE: Abatacept (CTLA4-Ig, Orencia) produces DRAMATIC clinical responses in LRBA deficiency — resolution of inflammatory bowel disease, autoimmune cytopenias, lymphoproliferation, and granulomata reported; always trial abatacept before more toxic immunosuppression; IV or SC formulation",
            "LRBA-IBD-MIMICS-CROHNS: Intestinal disease in LRBA deficiency (>50% of patients) mimics Crohn's disease or UC — diagnose LRBA BEFORE starting biologics for apparent IBD in a child with recurrent infections + autoimmune cytopenias; standard IBD biologics may be insufficient; abatacept is the preferred intervention",
            "LRBA-WESTERN-BLOT-DIAGNOSTIC: LRBA protein expression in PBMCs by Western blot or intracellular flow cytometry is a rapid diagnostic screen — absent LRBA protein in lymphocytes confirms biallelic LoF; perform before complete sequencing of this large gene (60 exons); reduces turnaround time for critical treatment decisions",
            "LRBA-IVIG-AND-ABATACEPT-COMBINATION: LRBA deficiency management = IVIG lifelong (for hypogammaglobulinaemia) PLUS abatacept (for CTLA4-mediated autoimmunity/IBD/lymphoproliferation); abatacept does NOT replace IVIG — both are required; monitor immunoglobulin troughs and clinical response to each",
            "LRBA-MISDIAGNOSED-AS-CVID: LRBA deficiency is frequently diagnosed as CVID initially — distinguish by: early onset (<10 years), prominent autoimmunity and IBD, splenomegaly, absent LRBA protein, abatacept response; screen for LRBA in all young CVID patients with autoimmune/inflammatory features",
            "LRBA-HSCT-CONSIDERATION: HSCT is curative for LRBA deficiency in severe cases — consider for patients with severe IBD, organomegaly, lymphoproliferation unresponsive to abatacept, or recurrent serious infections; HSCT corrects both immunodeficiency and autoimmunity; timing and conditioning regiment depend on disease activity",
        ],
    },
    # ── CTLA4 — CTLA4 Haploinsufficiency ──
    {
        "gene": "CTLA4",
        "protein": "CTLA4-HI — Haploinsufficiency, Autoimmunity + CVID-like + Lymphoproliferation, Abatacept SPECIFIC",
        "alias": (
            "CTLA4; OMIM gene 123890; CTLA4-HI OMIM 616100; 2q33.2; 223 aa; ~25 kDa; "
            "CTLA4 encodes cytotoxic T-lymphocyte-associated protein 4 (CD152), a "
            "transmembrane immunoreceptor of the CD28 family expressed on activated T cells "
            "and constitutively on regulatory T cells (Tregs). CTLA4 functions as the "
            "primary T-cell immune checkpoint — it competitively outcompetes CD28 for "
            "B7.1/CD80 and B7.2/CD86 ligands on APCs (10-100x higher binding affinity "
            "and avidity than CD28) and trans-endocytoses B7 molecules from the APC surface, "
            "preventing CD28 costimulation and inducing T-cell anergy or tolerance. "
            "CTLA4 haploinsufficiency (CTLA4-HI) is caused by heterozygous loss-of-function "
            "variants in CTLA4 — a dominantly inherited immune dysregulation syndrome. "
            "Reduced CTLA4 expression on Tregs and activated T cells → insufficient B7 "
            "downregulation → CD28-mediated T-cell costimulation escapes checkpoint control "
            "→ lymphoproliferation, autoimmunity, and paradoxically ALSO impaired humoral "
            "immunity. Clinical phenotype: (1) Hypogammaglobulinaemia (CVID-like) — "
            "reduced IgG, IgA, IgM; paradoxical because autoimmune activation is present "
            "but B-cell output is impaired (follicular helper T cell dysregulation); "
            "(2) Autoimmunity — autoimmune cytopenia (AIHA, ITP), thyroiditis, "
            "enteropathy, hepatitis, nephritis, type 1 diabetes; (3) Lymphoproliferation "
            "— splenomegaly, hepatomegaly, lymphadenopathy, T regulatory cell infiltration "
            "of multiple organs (lung, gut, brain); (4) Granulomatous disease — pulmonary "
            "granulomata clinically mimicking sarcoidosis. CTLA4-HI is clinically almost "
            "IDENTICAL to LRBA deficiency because LRBA deficiency causes functional CTLA4 "
            "deficiency via recycling impairment. Treatment: abatacept (CTLA4-Ig) is the "
            "SPECIFIC treatment — by providing exogenous CTLA4 function, abatacept "
            "compensates for haploinsufficiency; mTOR inhibitor sirolimus for "
            "lymphoproliferation; IVIG for hypogammaglobulinaemia."
        ),
        "aa": "223 aa",
        "kDa": "~25 kDa",
        "locus": "2q33.2",
        "omim_gene": 123890,
        "omim_disease": 616100,
        "inheritance": "AD — autosomal dominant haploinsufficiency; heterozygous LoF variants; variable penetrance",
        "gene_class": (
            "CTLA4 is a 223-amino acid type I transmembrane glycoprotein. Domain structure: "
            "(1) N-terminal signal peptide (aa 1-35) — ER translocation; (2) extracellular "
            "domain (aa 36-161) — immunoglobulin V-set domain; contains the B7-binding "
            "MYPPPY motif (aa 99-104) conserved with CD28; two N-glycosylation sites "
            "(Asn78, Asn110); (3) transmembrane domain (aa 162-182); (4) cytoplasmic "
            "tail (aa 183-223) — contains YVKM motif (Tyr182) for PI3K p85 subunit binding, "
            "AP-2 clathrin adaptor binding (endocytosis signal), and Lck-SH2 binding. "
            "CTLA4 is constitutively internalised by AP-2-mediated endocytosis every "
            "10-20 minutes and recycled to the surface via LRBA-dependent recycling "
            "endosomes — this rapid cycling allows CTLA4 to continuously strip B7 "
            "from APC surfaces via trans-endocytosis. CTLA4 binds both B7.1 (CD80) "
            "and B7.2 (CD86) homodimers; CTLA4 homodimerises via disulfide bond (Cys120) "
            "to present bivalent B7-binding surfaces; avidity of bivalent CTLA4 for B7 "
            "dimers exceeds CD28 by 100x. Haploinsufficiency mechanism: heterozygous LoF "
            "reduces CTLA4 expression by ~50% on Tregs — since CTLA4 function at the "
            "immune synapse is critically dependent on competitive kinetics with CD28 "
            "for B7, even a 50% reduction substantially impairs B7 downregulation, "
            "particularly in settings of high antigen/B7 expression. Variable penetrance "
            "(30-70%) is observed in CTLA4-HI families — additional genetic and "
            "environmental modifiers regulate clinical expression. The most common "
            "CTLA4-HI variants are missense affecting the MYPPPY motif, truncating "
            "variants, and splice-site variants."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("CTLA4 truncating heterozygous — haploinsufficiency, CVID-like + lymphoproliferation + autoimmunity", 0.40),
            ("CTLA4 MYPPPY-motif missense heterozygous — impaired B7 binding, immune dysregulation phenotype", 0.30),
            ("CTLA4 splice-site heterozygous — reduced CTLA4 expression, variable penetrance, lymphoproliferation", 0.20),
            ("CTLA4 large deletion heterozygous — complete haploinsufficiency, severe early-onset phenotype", 0.10),
        ],
        "key_alerts": [
            "CTLA4-ABATACEPT-SPECIFIC-TREATMENT: Abatacept (CTLA4-Ig) is the SPECIFIC and rationally targeted treatment for CTLA4-HI — provides exogenous CTLA4 function, restoring B7 downregulation; multiple case series report dramatic responses in lymphoproliferation, autoimmune cytopenia, enteropathy, and lung disease; trial before other immunosuppressants",
            "CTLA4-IDENTICAL-TO-LRBA: CTLA4-HI is clinically indistinguishable from LRBA deficiency — both cause CTLA4 functional deficiency (direct haploinsufficiency vs indirect recycling defect); both respond to abatacept; distinguish by: CTLA4-HI = AD (heterozygous), LRBA = AR (biallelic); CTLA4 protein reduced on flow cytometry in CTLA4-HI; LRBA protein absent in LRBA deficiency",
            "CTLA4-SIROLIMUS-LYMPHOPROLIFERATION: Sirolimus (mTOR inhibitor) reduces lymphoproliferation in CTLA4-HI — lymph node size, splenomegaly, and organ infiltration improve; combine with abatacept for severe lymphoproliferative disease; monitor mTOR toxicities (infections, metabolic, mucositis)",
            "CTLA4-VARIABLE-PENETRANCE-FAMILY: CTLA4-HI has variable penetrance (30-70%) — not all heterozygous family members are clinically affected; always offer genetic testing to first-degree relatives of index cases; clinically unaffected carriers may have subclinical lymphopenia or organ-specific autoimmunity on careful evaluation",
            "CTLA4-PULMONARY-GRANULOMA-MIMICS-SARCOIDOSIS: Pulmonary granulomata in CTLA4-HI are frequently misdiagnosed as sarcoidosis — distinguish by: young age of onset, associated hypogammaglobulinaemia, autoimmune cytopenia, family history, absent ACE elevation in CTLA4-HI; test ACE + CTLA4 gene panel in any young 'sarcoidosis' with combined features",
            "CTLA4-IVIG-REQUIRED-ALONGSIDE-ABATACEPT: IVIG is required alongside abatacept in CTLA4-HI — abatacept corrects immune dysregulation but does not restore immunoglobulin production; maintain IgG trough >7-8 g/L; both treatments required long-term; regular monitoring of immunoglobulins, T/B cell counts, and autoantibodies",
        ],
    },
    # ── PIK3CD — APDS1 / Activated PI3K-Delta Syndrome ──
    {
        "gene": "PIK3CD",
        "protein": "PI3Kδ Catalytic — APDS1, GOF AKT/mTOR, T-cell Senescence, EBV/CMV Susceptibility, Leniolisib FDA 2023",
        "alias": (
            "PIK3CD; OMIM gene 602839; APDS1 OMIM 615513; 1p36.22; 1044 aa; ~119 kDa; "
            "PIK3CD encodes the p110δ catalytic subunit of phosphatidylinositol 3-kinase "
            "delta (PI3Kδ), a class IA PI3K predominantly expressed in haematopoietic cells "
            "(T cells, B cells, NK cells, neutrophils, mast cells, dendritic cells). "
            "Activated PI3K delta syndrome 1 (APDS1) is caused by GAIN-OF-FUNCTION (GOF) "
            "heterozygous variants in PIK3CD that increase PI3Kδ catalytic activity, "
            "producing excessive PIP3 generation and constitutive activation of the "
            "AKT→mTOR→S6K1 signalling axis. The pathological downstream consequences: "
            "(1) T-cell senescence — constitutive mTOR activation drives premature T-cell "
            "differentiation into a senescent, terminally differentiated (CD57+ PD-1+) "
            "phenotype; naive T cells are depleted; T-cell receptor diversity is markedly "
            "reduced; cytotoxic function against viral-infected cells is impaired; "
            "(2) B-cell maturation arrest — PI3Kδ hyperactivation blocks B-cell transition "
            "from naive to memory and class-switched memory B cells; germinal centre "
            "reactions are impaired; immunoglobulin class-switching is defective → "
            "hypogammaglobulinaemia despite normal/elevated IgM (HIGM-like pattern); "
            "(3) Herpesvirus susceptibility — EBV and CMV are incompletely controlled → "
            "recurrent EBV viraemia, EBV-associated lymphoproliferation (EBV+ B cell "
            "lymphoma risk), CMV disease; chronic herpesvirus antigenic drive further "
            "promotes T-cell senescence; (4) Recurrent sinopulmonary bacterial infections "
            "from humoral immunodeficiency. Clinical features: onset in childhood with "
            "recurrent respiratory infections, herpes labialis/zoster, EBV/CMV viraemia, "
            "progressive lymphadenopathy, splenomegaly. Key treatment advance: "
            "leniolisib (OMGARD) received FDA approval March 2023 for APDS1 patients "
            "≥12 years — selective PI3Kδ inhibitor; reduces AKT phosphorylation, "
            "normalises B-cell maturation, reduces lymphoproliferation and infections. "
            "Idelalisib was earlier studied but hepatotoxicity limited its use in children."
        ),
        "aa": "1044 aa",
        "kDa": "~119 kDa",
        "locus": "1p36.22",
        "omim_gene": 602839,
        "omim_disease": 615513,
        "inheritance": "AD — autosomal dominant gain-of-function; heterozygous activating variants; APDS2 from PIK3R1 (p85α regulatory subunit LoF)",
        "gene_class": (
            "PI3Kδ (p110δ) is a 1044-amino acid class IA PI3K catalytic subunit. Domain "
            "architecture: (1) N-terminal adaptor-binding domain (ABD, aa 1-108) — binds "
            "the SH2 domains of the regulatory subunit p85α (PIK3R1) or p85β (PIK3R2) via "
            "the iSH2 coiled-coil domain of p85; maintains PI3K in an autoinhibited "
            "low-basal-activity state; (2) RAS-binding domain (RBD, aa 179-291) — binds "
            "RAS-GTP for allosteric activation; (3) C2 domain (aa 323-480) — membrane "
            "targeting; (4) helical domain (aa 481-686) — connects RBD and kinase "
            "domains; (5) kinase domain (aa 697-1044) — catalytic C-terminal domain; "
            "DFG motif (Asp911, Phe912, Gly913) in activation loop; ATP-binding cleft "
            "with Val828 as the 'gatekeeper' residue (targeted by p110δ-selective "
            "inhibitors via van der Waals contacts); transfers γ-phosphate of ATP to the "
            "3-OH position of phosphatidylinositol-4,5-bisphosphate (PIP2) → "
            "phosphatidylinositol-3,4,5-trisphosphate (PIP3). PIP3 recruits PH-domain-"
            "containing proteins to the inner plasma membrane: AKT (PKB) → phosphorylated "
            "at Thr308 (PDK1) and Ser473 (mTORC2) → activated AKT phosphorylates TSC2 "
            "→ mTORC1 activation → S6K1 → cell growth, protein synthesis, T-cell "
            "differentiation. GOF variants in APDS1: E1021K (Glu1021Lys, the most "
            "common APDS1 variant — located in the kinase domain C-lobe) constitutively "
            "disrupts an autoinhibitory interaction between the kinase C-lobe and the "
            "regulatory p85 nSH2 domain, increasing basal kinase activity 3-5 fold. "
            "Other GOF variants: N334K, C416R (helical domain), and others clustering "
            "in regions that stabilise the open/active kinase conformation. Leniolisib "
            "and idelalisib bind the ATP-binding site within the kinase domain with "
            "high selectivity for p110δ over p110α/β/γ."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("PIK3CD E1021K GOF AD — most common APDS1 variant, constitutive PI3Kδ, T-cell senescence + B-cell maturation arrest", 0.45),
            ("PIK3CD helical domain GOF AD — APDS1, EBV/CMV viraemia, lymphoproliferation, hypogammaglobulinaemia", 0.25),
            ("PIK3CD kinase domain GOF AD (non-E1021K) — APDS1, variable severity, herpesvirus susceptibility", 0.20),
            ("PIK3R1 LoF AD — APDS2, p85α regulatory subunit LoF → PI3Kδ GOF equivalent phenotype, HIGM pattern", 0.10),
        ],
        "key_alerts": [
            "PIK3CD-LENIOLISIB-FDA-2023: Leniolisib (OMGARD) received FDA approval March 2023 for APDS1 (PIK3CD GOF) in patients ≥12 years — first approved PI3Kδ inhibitor for APDS; reduces AKT phosphorylation, normalises B-cell differentiation, reduces lymphoproliferation, infections, and splenomegaly; refer eligible patients to treating centre",
            "PIK3CD-EBV-CMV-HERPEVIRUS-SUSCEPTIBILITY: APDS1 patients have markedly impaired control of EBV and CMV due to senescent cytotoxic T cells — recurrent EBV viraemia, EBV-associated lymphoproliferation (including EBV+ lymphoma), and CMV disease; monitor EBV/CMV DNA PCR quarterly; antiviral prophylaxis (valaciclovir) in high-viraemia periods",
            "PIK3CD-T-CELL-SENESCENCE-IRREVERSIBLE: Constitutive mTOR activation drives irreversible T-cell senescence (CD57+PD-1+CD28-) in APDS1 — senescent T cells cannot be reinvigorated by IVIG; leniolisib may partially restore naive T-cell compartment; T-cell immunological monitoring (TREC, TCR repertoire, naive CD4/CD8 counts) guides therapy",
            "PIK3CD-HIGM-PATTERN-DIAGNOSTIC: APDS1 frequently presents with HIGM-like pattern — elevated or normal IgM with low IgG/IgA due to B-cell class-switching defect; distinguish from CD40L/CD40/AID/UNG-HIGM by: lymphoproliferation, herpesvirus susceptibility, GOF PIK3CD; HIGM pattern + EBV viraemia + lymphadenopathy → order PI3Kδ gene panel",
            "PIK3CD-IVIG-PLUS-LENIOLISIB: IVIG replacement for hypogammaglobulinaemia remains required alongside leniolisib — leniolisib improves B-cell differentiation but does not immediately restore IgG production; maintain IgG trough >7 g/L; reassess need for IVIG after 12+ months of leniolisib therapy",
            "PIK3CD-APDS2-PIK3R1-PANEL: APDS2 (caused by PIK3R1 loss-of-function, p85α regulatory subunit) is clinically identical to APDS1 — SAME phenotype (T-cell senescence, EBV/CMV, HIGM, lymphoproliferation) via equivalent GOF effect on PI3Kδ; ALWAYS sequence BOTH PIK3CD AND PIK3R1 in APDS diagnosis; leniolisib works in APDS2 as well",
        ],
    },
]


def _make_cohort(gd):
    r = random.Random(gd["seed"])
    gene = gd["gene"]
    pts = []
    etiols = gd["etiologies"]
    weights = [e[1] for e in etiols]
    labels = [e[0] for e in etiols]

    for i in range(gd["n_patients"]):
        # Weighted etiology selection
        roll = r.random()
        cumul = 0.0
        etiol = labels[-1]
        for lbl, wt in zip(labels, weights):
            cumul += wt
            if roll < cumul:
                etiol = lbl
                break

        # Sex — XL conditions predominantly affect males
        if gene in ("BTK", "CYBB", "WAS"):
            sex = "M" if r.random() < 0.92 else "F"
        else:
            sex = r.choice(["M", "F"])

        # Age at onset
        if gene == "BTK":
            age_onset = r.gauss(1.2, 0.8)        # onset after maternal IgG wanes ~6-18 months
        elif gene == "RAG1":
            age_onset = r.gauss(0.3, 0.5) if "SCID" in etiol else r.gauss(1.5, 2.0)
        elif gene == "ADA":
            age_onset = r.gauss(0.2, 0.4) if "neonatal" in etiol else r.gauss(8.0, 5.0)
        elif gene == "CYBB":
            age_onset = r.gauss(3.0, 3.0)        # CGD often diagnosed in early childhood
        elif gene == "WAS":
            age_onset = r.gauss(0.5, 0.5)        # thrombocytopenia apparent from birth/infancy
        elif gene == "LRBA":
            age_onset = r.gauss(4.0, 3.0)        # early childhood
        elif gene == "CTLA4":
            age_onset = r.gauss(12.0, 8.0)       # variable, childhood to adulthood
        elif gene == "PIK3CD":
            age_onset = r.gauss(6.0, 4.0)
        else:
            age_onset = r.gauss(5.0, 5.0)
        age_onset = max(0.0, round(age_onset, 1))

        # Dx delay
        if gene in ("BTK", "WAS"):
            dx_delay = r.gauss(18, 12)
        elif gene in ("LRBA", "CTLA4"):
            dx_delay = r.gauss(48, 24)           # frequently misdiagnosed as CVID
        elif gene == "PIK3CD":
            dx_delay = r.gauss(36, 18)
        else:
            dx_delay = r.gauss(24, 18)
        dx_delay = max(0.0, round(dx_delay, 1))

        # ── universal fields ──────────────────────────────────────────
        ivig_given = r.random() < (
            0.98 if gene in ("BTK", "RAG1", "ADA") else
            0.85 if gene in ("CYBB", "WAS") else
            0.90 if gene in ("LRBA", "CTLA4", "PIK3CD") else 0.80
        )
        live_vaccine_avoided = r.random() < (
            0.82 if gene in ("BTK", "RAG1", "ADA", "CYBB", "WAS", "PIK3CD") else 0.70
        )
        hsct_performed = r.random() < (
            0.55 if gene in ("RAG1", "ADA") else
            0.40 if gene in ("WAS", "CYBB") else
            0.10 if gene in ("BTK", "LRBA", "CTLA4", "PIK3CD") else 0.10
        )
        gene_therapy_given = r.random() < (
            0.08 if gene == "ADA" else
            0.04 if gene == "WAS" else 0.0
        )
        prophylaxis_given = r.random() < (
            0.95 if gene == "CYBB" else
            0.70 if gene in ("BTK", "WAS", "RAG1", "ADA") else
            0.50 if gene in ("LRBA", "CTLA4", "PIK3CD") else 0.40
        )

        # ── BTK-specific ──────────────────────────────────────────────
        btk_b_cells_absent = r.random() < 0.96 if gene == "BTK" else False
        btk_monocyte_assay = r.random() < 0.78 if gene == "BTK" else False
        btk_bronchiectasis = r.random() < 0.28 if gene == "BTK" else False
        btk_enterovirus_meningoencephalitis = r.random() < 0.06 if gene == "BTK" else False

        # ── RAG1-specific ─────────────────────────────────────────────
        rag1_omenn_phenotype = ("Omenn" in etiol) if gene == "RAG1" else False
        rag1_trec_detected = r.random() < 0.65 if gene == "RAG1" else False
        rag1_bcg_disease = r.random() < 0.12 if gene == "RAG1" else False

        # ── ADA-specific ──────────────────────────────────────────────
        ada_peg_ada_given = r.random() < 0.52 if gene == "ADA" else False
        ada_skeletal_anomaly = r.random() < 0.45 if gene == "ADA" else False
        ada_datp_elevated = r.random() < 0.90 if gene == "ADA" else False

        # ── CYBB-specific ─────────────────────────────────────────────
        cybb_aspergillus_infection = r.random() < 0.62 if gene == "CYBB" else False
        cybb_dhr_done = r.random() < 0.88 if gene == "CYBB" else False
        cybb_ifn_gamma_given = r.random() < 0.55 if gene == "CYBB" else False
        cybb_burkholderia = r.random() < 0.12 if gene == "CYBB" else False

        # ── WAS-specific ──────────────────────────────────────────────
        was_small_platelets = r.random() < 0.94 if gene == "WAS" else False
        was_eczema = r.random() < 0.82 if gene == "WAS" else False
        was_splenectomy = r.random() < 0.20 if gene == "WAS" else False
        was_ich = r.random() < 0.06 if gene == "WAS" else False  # intracranial haemorrhage
        was_ebv_lymphoma = r.random() < 0.08 if gene == "WAS" else False
        was_autoimmunity = r.random() < 0.45 if gene == "WAS" else False

        # ── LRBA-specific ─────────────────────────────────────────────
        lrba_abatacept_given = r.random() < 0.72 if gene == "LRBA" else False
        lrba_ibd = r.random() < 0.55 if gene == "LRBA" else False
        lrba_western_blot_done = r.random() < 0.68 if gene == "LRBA" else False
        lrba_organomegaly = r.random() < 0.78 if gene == "LRBA" else False
        lrba_autoimmune_cytopenia = r.random() < 0.52 if gene == "LRBA" else False

        # ── CTLA4-specific ────────────────────────────────────────────
        ctla4_abatacept_given = r.random() < 0.75 if gene == "CTLA4" else False
        ctla4_sirolimus_given = r.random() < 0.40 if gene == "CTLA4" else False
        ctla4_lymphoproliferation = r.random() < 0.80 if gene == "CTLA4" else False
        ctla4_granulomata = r.random() < 0.35 if gene == "CTLA4" else False
        ctla4_autoimmune_cytopenia = r.random() < 0.50 if gene == "CTLA4" else False

        # ── PIK3CD-specific ───────────────────────────────────────────
        pik3cd_leniolisib_given = r.random() < 0.38 if gene == "PIK3CD" else False
        pik3cd_ebv_viraemia = r.random() < 0.70 if gene == "PIK3CD" else False
        pik3cd_cmv_disease = r.random() < 0.35 if gene == "PIK3CD" else False
        pik3cd_t_cell_senescence = r.random() < 0.85 if gene == "PIK3CD" else False
        pik3cd_lymphadenopathy = r.random() < 0.72 if gene == "PIK3CD" else False

        pts.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "sex": sex,
            "age_at_onset": age_onset,
            "age_at_dx": max(age_onset, round(age_onset + dx_delay / 12, 1)),
            "dx_delay_months": dx_delay,
            "etiology": etiol,
            "inheritance": gd["inheritance"].split(";")[0].strip(),
            # ── universal treatment/management ──
            "ivig_given": ivig_given,
            "live_vaccine_avoided": live_vaccine_avoided,
            "hsct_performed": hsct_performed,
            "gene_therapy_given": gene_therapy_given,
            "prophylaxis_given": prophylaxis_given,
            # ── BTK ──
            "btk_b_cells_absent": btk_b_cells_absent,
            "btk_monocyte_assay": btk_monocyte_assay,
            "btk_bronchiectasis": btk_bronchiectasis,
            "btk_enterovirus_meningoencephalitis": btk_enterovirus_meningoencephalitis,
            # ── RAG1 ──
            "rag1_omenn_phenotype": rag1_omenn_phenotype,
            "rag1_trec_detected": rag1_trec_detected,
            "rag1_bcg_disease": rag1_bcg_disease,
            # ── ADA ──
            "ada_peg_ada_given": ada_peg_ada_given,
            "ada_skeletal_anomaly": ada_skeletal_anomaly,
            "ada_datp_elevated": ada_datp_elevated,
            # ── CYBB ──
            "cybb_aspergillus_infection": cybb_aspergillus_infection,
            "cybb_dhr_done": cybb_dhr_done,
            "cybb_ifn_gamma_given": cybb_ifn_gamma_given,
            "cybb_burkholderia": cybb_burkholderia,
            # ── WAS ──
            "was_small_platelets": was_small_platelets,
            "was_eczema": was_eczema,
            "was_splenectomy": was_splenectomy,
            "was_ich": was_ich,
            "was_ebv_lymphoma": was_ebv_lymphoma,
            "was_autoimmunity": was_autoimmunity,
            # ── LRBA ──
            "lrba_abatacept_given": lrba_abatacept_given,
            "lrba_ibd": lrba_ibd,
            "lrba_western_blot_done": lrba_western_blot_done,
            "lrba_organomegaly": lrba_organomegaly,
            "lrba_autoimmune_cytopenia": lrba_autoimmune_cytopenia,
            # ── CTLA4 ──
            "ctla4_abatacept_given": ctla4_abatacept_given,
            "ctla4_sirolimus_given": ctla4_sirolimus_given,
            "ctla4_lymphoproliferation": ctla4_lymphoproliferation,
            "ctla4_granulomata": ctla4_granulomata,
            "ctla4_autoimmune_cytopenia": ctla4_autoimmune_cytopenia,
            # ── PIK3CD ──
            "pik3cd_leniolisib_given": pik3cd_leniolisib_given,
            "pik3cd_ebv_viraemia": pik3cd_ebv_viraemia,
            "pik3cd_cmv_disease": pik3cd_cmv_disease,
            "pik3cd_t_cell_senescence": pik3cd_t_cell_senescence,
            "pik3cd_lymphadenopathy": pik3cd_lymphadenopathy,
        })
    return pts


def _pct(pts, key):
    if not pts:
        return 0.0
    return round(100 * sum(1 for p in pts if p.get(key)) / len(pts), 1)


def _make_patients(gene_dict):
    """Public alias for _make_cohort — generates 40 patients for a single gene dict."""
    return _make_cohort(gene_dict)


def get_overview():
    all_pts = []
    gene_summaries = []
    all_alerts = []

    for gd in IMMUNODEFICIENCY_GENES:
        pts = _make_cohort(gd)
        all_pts.extend(pts)
        gene_summaries.append({
            "gene": gd["gene"],
            "protein": gd["protein"][:80],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "mean_onset_years": round(
                sum(p["age_at_onset"] for p in pts) / len(pts), 1
            ),
            "mean_dx_delay_months": round(
                sum(p["dx_delay_months"] for p in pts) / len(pts), 1
            ),
        })
        all_alerts.extend(gd["key_alerts"])

    # Per-gene cohort subsets
    btk     = [p for p in all_pts if p["gene"] == "BTK"]
    rag1    = [p for p in all_pts if p["gene"] == "RAG1"]
    ada     = [p for p in all_pts if p["gene"] == "ADA"]
    cybb    = [p for p in all_pts if p["gene"] == "CYBB"]
    was     = [p for p in all_pts if p["gene"] == "WAS"]
    lrba    = [p for p in all_pts if p["gene"] == "LRBA"]
    ctla4   = [p for p in all_pts if p["gene"] == "CTLA4"]
    pik3cd  = [p for p in all_pts if p["gene"] == "PIK3CD"]

    agg = {
        "total_patients": len(all_pts),
        "mean_dx_delay_months": round(
            sum(p["dx_delay_months"] for p in all_pts) / len(all_pts), 1
        ),
        "hsct_performed_pct": _pct(all_pts, "hsct_performed"),
        "ivig_given_pct": _pct(all_pts, "ivig_given"),
        "live_vaccine_avoided_pct": _pct(all_pts, "live_vaccine_avoided"),
        "gene_therapy_given_pct": _pct(all_pts, "gene_therapy_given"),
        "prophylaxis_given_pct": _pct(all_pts, "prophylaxis_given"),
        # ── BTK stats ──────────────────────────────────────────────────
        "btk_ivig_pct": _pct(btk, "ivig_given"),
        "btk_live_vaccine_avoided_pct": _pct(btk, "live_vaccine_avoided"),
        "btk_b_cells_absent_pct": _pct(btk, "btk_b_cells_absent"),
        "btk_monocyte_assay_pct": _pct(btk, "btk_monocyte_assay"),
        "btk_bronchiectasis_pct": _pct(btk, "btk_bronchiectasis"),
        # ── RAG1 stats ─────────────────────────────────────────────────
        "rag1_hsct_pct": _pct(rag1, "hsct_performed"),
        "rag1_omenn_pct": _pct(rag1, "rag1_omenn_phenotype"),
        "rag1_trec_pct": _pct(rag1, "rag1_trec_detected"),
        "rag1_bcg_disease_pct": _pct(rag1, "rag1_bcg_disease"),
        # ── ADA stats ──────────────────────────────────────────────────
        "ada_gene_therapy_pct": _pct(ada, "gene_therapy_given"),
        "ada_hsct_pct": _pct(ada, "hsct_performed"),
        "ada_peg_ada_pct": _pct(ada, "ada_peg_ada_given"),
        "ada_skeletal_pct": _pct(ada, "ada_skeletal_anomaly"),
        "ada_datp_elevated_pct": _pct(ada, "ada_datp_elevated"),
        # ── CYBB stats ─────────────────────────────────────────────────
        "cybb_prophylaxis_pct": _pct(cybb, "prophylaxis_given"),
        "cybb_ifn_gamma_pct": _pct(cybb, "cybb_ifn_gamma_given"),
        "cybb_aspergillus_pct": _pct(cybb, "cybb_aspergillus_infection"),
        "cybb_dhr_done_pct": _pct(cybb, "cybb_dhr_done"),
        "cybb_hsct_pct": _pct(cybb, "hsct_performed"),
        # ── WAS stats ──────────────────────────────────────────────────
        "was_hsct_pct": _pct(was, "hsct_performed"),
        "was_splenectomy_pct": _pct(was, "was_splenectomy"),
        "was_small_platelets_pct": _pct(was, "was_small_platelets"),
        "was_eczema_pct": _pct(was, "was_eczema"),
        "was_ich_pct": _pct(was, "was_ich"),
        "was_autoimmunity_pct": _pct(was, "was_autoimmunity"),
        # ── LRBA stats ─────────────────────────────────────────────────
        "lrba_abatacept_pct": _pct(lrba, "lrba_abatacept_given"),
        "lrba_ibd_pct": _pct(lrba, "lrba_ibd"),
        "lrba_western_blot_pct": _pct(lrba, "lrba_western_blot_done"),
        "lrba_organomegaly_pct": _pct(lrba, "lrba_organomegaly"),
        "lrba_autoimmune_cytopenia_pct": _pct(lrba, "lrba_autoimmune_cytopenia"),
        # ── CTLA4 stats ────────────────────────────────────────────────
        "ctla4_abatacept_pct": _pct(ctla4, "ctla4_abatacept_given"),
        "ctla4_sirolimus_pct": _pct(ctla4, "ctla4_sirolimus_given"),
        "ctla4_lymphoproliferation_pct": _pct(ctla4, "ctla4_lymphoproliferation"),
        "ctla4_granulomata_pct": _pct(ctla4, "ctla4_granulomata"),
        # ── PIK3CD stats ───────────────────────────────────────────────
        "pik3cd_leniolisib_pct": _pct(pik3cd, "pik3cd_leniolisib_given"),
        "pik3cd_ebv_viraemia_pct": _pct(pik3cd, "pik3cd_ebv_viraemia"),
        "pik3cd_cmv_disease_pct": _pct(pik3cd, "pik3cd_cmv_disease"),
        "pik3cd_t_cell_senescence_pct": _pct(pik3cd, "pik3cd_t_cell_senescence"),
    }

    return {
        "title": (
            "Hereditary-Immunodeficiency-Atlas — Complete 8-Gene Hereditary "
            "Primary Immunodeficiency Reference"
        ),
        "subtitle": (
            "BTK · RAG1 · ADA · CYBB · WAS · LRBA · CTLA4 · PIK3CD — "
            "320 patients (8×40, seeds 1534–1541) — XLA IVIG Lifelong No-Live-Vaccines, "
            "ADA-SCID Strimvelis Gene Therapy EMA 2016, CGD Aspergillus Prophylaxis, "
            "LRBA/CTLA4-HI Abatacept SPECIFIC Treatment, APDS Leniolisib FDA 2023"
        ),
        "genes": gene_summaries,
        "aggregate_stats": agg,
        "top_alerts": all_alerts[:20],
    }


def get_breakdown():
    breakdown = []
    for gd in IMMUNODEFICIENCY_GENES:
        pts = _make_cohort(gd)
        sex_dist = {
            "M": sum(1 for p in pts if p["sex"] == "M"),
            "F": sum(1 for p in pts if p["sex"] == "F"),
        }
        mean_onset = round(sum(p["age_at_onset"] for p in pts) / len(pts), 1)
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
        "atlas": (
            "Hereditary-Immunodeficiency-Atlas — Complete 8-Gene Hereditary "
            "Primary Immunodeficiency Reference"
        ),
        "genes": [gd["gene"] for gd in IMMUNODEFICIENCY_GENES],
        "clinical_definitions": [
            {
                "gene": gd["gene"],
                "full_name": gd["protein"],
                "alias": gd["alias"],
                "aa": gd["aa"],
                "kDa": gd["kDa"],
                "locus": gd["locus"],
                "omim_gene": gd["omim_gene"],
                "omim_disease": gd["omim_disease"],
                "inheritance": gd["inheritance"],
                "gene_class": gd["gene_class"],
                "key_alerts": gd["key_alerts"],
            }
            for gd in IMMUNODEFICIENCY_GENES
        ],
        "cross_cutting_definitions": [
            {
                "term": "Primary Immunodeficiency — Live Vaccine Absolute Contraindication",
                "definition": (
                    "Live attenuated vaccines are ABSOLUTELY CONTRAINDICATED in virtually all "
                    "primary immunodeficiency disorders affecting T-cell or B-cell function. "
                    "Documented risks: oral poliovirus vaccine (OPV) → vaccine-associated "
                    "paralytic poliomyelitis (VAPP) in XLA (BTK deficiency); BCG (Bacille "
                    "Calmette-Guérin) administered at birth in many countries → disseminated "
                    "BCG disease (BCG-osis) in SCID (RAG1, ADA) and CGD (CYBB) patients; "
                    "MMR (measles-mumps-rubella) live vaccine → measles inclusion body "
                    "encephalitis in combined immunodeficiency. Inactivated vaccines (IPV, "
                    "DTaP, Hib, PCV, MenACWY, hepatitis A/B, influenza inactivated) are safe "
                    "and recommended but may produce suboptimal responses depending on the "
                    "underlying immunodeficiency. In IVIG-treated XLA, passively administered "
                    "antibodies in IVIG preparations may interfere with live vaccine "
                    "immunogenicity (additional reason to avoid live vaccines). CRITICAL: "
                    "check BCG status at birth record for any infant newly diagnosed with SCID; "
                    "if BCG given before diagnosis, initiate anti-mycobacterial therapy "
                    "(isoniazid + rifampicin) while awaiting HSCT."
                ),
            },
            {
                "term": "HSCT — Curative Treatment for SCID, CGD, WAS, and Severe Combined PID",
                "definition": (
                    "Haematopoietic stem cell transplantation (HSCT) is curative for multiple "
                    "primary immunodeficiencies. Disease-specific outcomes: RAG1/ADA-SCID — "
                    "overall survival >90% with matched sibling donor (MSD), >80% with MUD "
                    "when performed in infection-free state before 3 months of age; conditioning "
                    "with busulfan/fludarabine or melphalan required for engraftment. CGD (CYBB) "
                    "— myeloablative conditioning + MSD/MUD HSCT; best results in young patients "
                    "(<10 years) before accumulation of fungal infection burden. WAS — HSCT "
                    "corrects all three components of the triad (thrombocytopenia, eczema, "
                    "immunodeficiency) and eliminates lymphoma and autoimmune risk; >90% OS "
                    "with MSD, >80% with MUD. Gene therapy (autologous CD34+ cells) is "
                    "emerging as an alternative for ADA-SCID (Strimvelis, EMA 2016) and WAS "
                    "(lentiviral WASP) — avoids GvHD risk and HLA-barrier. For LRBA, CTLA4-HI, "
                    "and APDS — HSCT is reserved for severe cases unresponsive to medical "
                    "therapy (abatacept for LRBA/CTLA4; leniolisib for APDS)."
                ),
            },
            {
                "term": "IVIG — Immunoglobulin Replacement Therapy in Primary Immunodeficiency",
                "definition": (
                    "Intravenous immunoglobulin (IVIG) replacement is the cornerstone of "
                    "humoral immunodeficiency management in XLA (BTK), CVID-like (LRBA, "
                    "CTLA4-HI, PIK3CD/APDS), and as a bridge to HSCT in SCID (RAG1, ADA). "
                    "Standard dosing: 400-600 mg/kg every 3-4 weeks IV; or 100-200 mg/kg/week "
                    "subcutaneously (SCIG). Target trough IgG: minimum 6-8 g/L for most "
                    "conditions; higher targets (>10 g/L) in patients with chronic lung disease "
                    "or recurrent infections despite standard dosing. IVIG contains pooled IgG "
                    "from ≥1,000 donors — provides broad opsonising and neutralising antibodies. "
                    "IVIG does NOT treat the cellular immunodeficiency component (T cells, NK "
                    "cells, phagocytes) — additional therapies are required. Monitoring: "
                    "trough IgG before each infusion; 6-monthly IgA, IgM; annual renal function "
                    "(sucrose-containing preparations → osmotic nephropathy); thrombotic risk "
                    "in high-dose IVIG (rate-control, hydration). SCIG is preferred for "
                    "home-based management — provides more stable IgG levels, fewer infusion "
                    "reactions, greater patient autonomy."
                ),
            },
            {
                "term": "Abatacept — CTLA4-Ig Fusion for LRBA Deficiency and CTLA4 Haploinsufficiency",
                "definition": (
                    "Abatacept (Orencia; CTLA4-Ig) is a fusion protein of the extracellular "
                    "CTLA4 domain (IgV-set) with human IgG1-Fc. It binds B7.1/CD80 and "
                    "B7.2/CD86 on antigen-presenting cells with high avidity, downregulating "
                    "CD28 costimulatory signals and restoring immune tolerance. In LRBA "
                    "deficiency and CTLA4 haploinsufficiency, abatacept provides exogenous "
                    "CTLA4 function, compensating for either impaired CTLA4 recycling (LRBA) "
                    "or reduced CTLA4 expression (CTLA4-HI). Clinical evidence: multiple case "
                    "series and cohort studies document dramatic responses — resolution of "
                    "inflammatory bowel disease (complete mucosal healing in 60-70%), "
                    "remission of autoimmune cytopenia (AIHA, ITP), regression of "
                    "lymphoproliferation and organomegaly, and improvement in pulmonary "
                    "granulomata. Dosing: IV formulation 10 mg/kg every 2-4 weeks (most used "
                    "in PID); SC formulation (125 mg weekly) used in some centres. Monitoring: "
                    "clinical response at 3-6 months; abatacept infection risk (increased "
                    "susceptibility particularly to intracellular pathogens — TB screening "
                    "before initiation); abatacept does NOT replace IVIG for "
                    "hypogammaglobulinaemia."
                ),
            },
            {
                "term": "Leniolisib (OMGARD) — FDA 2023 Approved PI3Kδ Inhibitor for APDS1",
                "definition": (
                    "Leniolisib (OMGARD; Pharming Group) received FDA approval March 2023 for "
                    "the treatment of adults and adolescents ≥12 years with Activated PI3K Delta "
                    "Syndrome (APDS), including both APDS1 (PIK3CD gain-of-function) and APDS2 "
                    "(PIK3R1 loss-of-function). It is the first FDA-approved drug specifically "
                    "for APDS. Leniolisib is an orally bioavailable, selective PI3Kδ inhibitor "
                    "that blocks the constitutive AKT/mTOR signalling caused by PIK3CD GOF "
                    "variants. Clinical trial results (APDS1/2 phase 3): leniolisib reduced "
                    "lymph node size (primary endpoint), reduced splenomegaly, improved B-cell "
                    "differentiation (increased naive B cells, class-switched memory B cells), "
                    "and reduced EBV viraemia. Dose: 70 mg twice daily orally. Safety: "
                    "hepatotoxicity monitoring (LFTs monthly first 3 months then quarterly); "
                    "serious infection risk (opportunistic infections); diarrhoea. Idelalisib "
                    "(earlier studied) had unacceptable hepatotoxicity and colitis at doses "
                    "required in APDS, limiting its use to adults. IVIG continues alongside "
                    "leniolisib — leniolisib restores B-cell differentiation but does not "
                    "immediately correct immunoglobulin production; assess need for IVIG "
                    "continuation annually."
                ),
            },
            {
                "term": "Strimvelis — EMA 2016 Approved Gene Therapy for ADA-SCID",
                "definition": (
                    "Strimvelis (GlaxoSmithKline/Orchard Therapeutics) received EMA conditional "
                    "approval in May 2016 for ADA-SCID — the first conditionally approved gene "
                    "therapy product for a primary immunodeficiency. It is manufactured by "
                    "ex vivo transduction of autologous patient CD34+ haematopoietic stem cells "
                    "with a gamma-retroviral vector expressing functional human ADA cDNA. "
                    "Corrected CD34+ cells are re-infused after mild myeloablative conditioning "
                    "(busulfan), allowing engraftment of gene-corrected HSCs. Long-term efficacy: "
                    "100% overall survival at 3 years in treated patients; progressive immune "
                    "reconstitution over 12-24 months; most patients achieve T-cell counts >500 "
                    "cells/μL, functional T-cell responses, and reduction of IVIG requirement. "
                    "Advantages over HSCT: no GvHD risk; no allogeneic donor required; "
                    "autologous procedure avoids HLA barriers. Advantages over PEG-ADA: "
                    "potentially curative (stable integration) vs lifelong PEG-ADA injections; "
                    "superior immune reconstitution long-term. Manufactured at a single centre "
                    "(Ospedale San Raffaele, Milan) — patients must travel to Italy for "
                    "treatment. Insertional mutagenesis risk (retroviral vectors) acknowledged "
                    "but not observed in ADA-SCID trials; lentiviral vectors with safer "
                    "integration profiles under development."
                ),
            },
            {
                "term": "DHR Oxidation Assay — Gold-Standard Diagnostic Test for CGD",
                "definition": (
                    "Dihydrorhodamine 123 (DHR) oxidation by flow cytometry is the gold-standard "
                    "functional diagnostic assay for chronic granulomatous disease. Principle: "
                    "neutrophils are loaded with DHR-123 (non-fluorescent), then activated with "
                    "phorbol 12-myristate 13-acetate (PMA), which directly activates PKC → "
                    "NADPH oxidase assembly and activation → superoxide production → conversion "
                    "of DHR-123 to rhodamine 123 (highly fluorescent, measured in FL-1/FITC "
                    "channel). Result interpretation: normal neutrophils show a large rightward "
                    "shift in DHR fluorescence after PMA (positive oxidative burst); CGD "
                    "neutrophils show NO shift (absent oxidative burst) or markedly reduced "
                    "shift (partial CGD, carrier females, attenuated variants). Female CYBB "
                    "carriers: bimodal DHR pattern — two distinct neutrophil populations (one "
                    "DHR-positive [normal allele] and one DHR-negative [CYBB-allele]), reflecting "
                    "X-inactivation mosaicism; proportion of DHR-negative cells correlates "
                    "with clinical severity in carriers. The NBT (nitroblue tetrazolium) slide "
                    "test is an older alternative (reduced NBT → blue formazan in normal "
                    "phagocytes; no colour change in CGD) but less sensitive and less "
                    "quantitative than DHR. DHR should be performed before genetic confirmation "
                    "and is the definitive functional test."
                ),
            },
            {
                "term": "Primary Immunodeficiency Diagnostic Ladder — Flow Cytometry to Genetic Panel",
                "definition": (
                    "A systematic approach to primary immunodeficiency diagnosis: "
                    "(1) Complete blood count with differential — absolute lymphocyte count "
                    "(ALC <3,000/μL in neonates suspect SCID; ALC <1,500/μL in older children); "
                    "thrombocytopenia with small platelets (MPV <5 fL) → WAS; neutropenia; "
                    "(2) Serum immunoglobulins (IgG, IgA, IgM) — absent all isotypes → XLA; "
                    "low IgG/IgA/IgM with elevated IgE → Omenn/RAG1; elevated IgM + low "
                    "IgG/IgA → HIGM syndromes, APDS; (3) Lymphocyte subsets by flow "
                    "cytometry — absent B cells (CD19) → XLA; T-B- → SCID (RAG1, ADA, "
                    "Artemis); T-B+ → SCID (IL2RG/JAK3/IL7R); BTK protein monocytes → XLA "
                    "screen; DHR oxidation → CGD; CTLA4 expression on T cells → CTLA4-HI; "
                    "LRBA Western blot → LRBA deficiency; (4) Functional tests — "
                    "lymphoproliferative responses (PHA, anti-CD3); specific antibody "
                    "responses (post-vaccination titres); oxidative burst (DHR); "
                    "(5) TREC/KREC (newborn screening for SCID); "
                    "(6) Genetic panel — minimum: BTK, RAG1/RAG2, ADA, CYBB, WAS, LRBA, "
                    "CTLA4, PIK3CD, PIK3R1, IL2RG, JAK3, IL7R, DOCK8, CARD11, STAT3, STAT1, "
                    "ITCH, FOXP3, and others; whole exome sequencing for unsolved cases."
                ),
            },
        ],
    }
