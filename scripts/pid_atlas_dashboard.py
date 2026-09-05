#!/usr/bin/env python3
"""PID-Atlas — Complete 8-Gene Primary Immunodeficiency Atlas
IL2RG  (Common gamma chain γc; ~369 aa; Xq13.1; X-SCID T-B+NK-; XLR; gene therapy OTL-101; HSCT curative) ·
ADA    (Adenosine deaminase; ~363 aa; 20q13.12; ADA-SCID; AR; PEG-ADA + Strimvelis gene therapy; HSCT) ·
RAG1   (Recombination Activating Gene 1; ~1043 aa; 11p13; T-B-NK+ SCID / Omenn Syndrome; AR; HSCT curative) ·
BTK    (Bruton's tyrosine kinase; ~659 aa; Xq22.1; XLA agammaglobulinemia; XLR; IgRT lifelong; BTK inhibitors) ·
CYBB   (gp91phox / NOX2; ~570 aa; Xp21.1; X-CGD; XLR; NBT test; prophylaxis + HSCT curative) ·
WAS    (Wiskott-Aldrich Syndrome Protein; ~502 aa; Xp11.23; WAS; XLR; thrombocytopenia+eczema+PID; HSCT/gene therapy) ·
TNFRSF13B (TACI; ~293 aa; 17p11.2; CVID most common symptomatic Ab deficiency adults; AD/AR; IgRT lifelong) ·
STAT3  (Signal transducer and activator of transcription 3; ~770 aa; 17q21.2; Hyper-IgE AD-HIES; AD GOF/LOF)
320-patient aggregate cohort (8 × 40, seeds 1118–1125)
"""

import random

SEED_BASE = 1118

PID_GENES = [
    # ── IL2RG — X-linked Severe Combined Immunodeficiency (X-SCID) ───────────
    {
        "gene": "IL2RG",
        "protein": "Common Gamma Chain (γc; CD132; IL2RG)",
        "alias": "IL2RG; OMIM gene 308380; Xq13.1; ~369 aa; X-SCID (OMIM #300400); XLR; T-B+NK- phenotype; shared γc receptor for IL-2/4/7/9/15/21; gene therapy OTL-101 approved",
        "aa": "~369 aa",
        "kDa": "~42 kDa",
        "mechanism": (
            "IL2RG encodes the common gamma chain (γc, CD132), the shared signal-transducing "
            "subunit of the receptors for interleukins IL-2, IL-4, IL-7, IL-9, IL-15, and IL-21. "
            "NORMAL FUNCTION: γc associates with cytokine-specific α/β chains → activates "
            "JAK1/JAK3 → phosphorylates STAT5 (IL-2/15) or STAT6 (IL-4) → transcription of "
            "survival and proliferation genes for T and NK cells. "
            "IL-7 signalling via γc is NON-REDUNDANT for thymic T-cell development — "
            "without IL-7/γc → no T-cell development. "
            "IL-15 signalling via γc is NON-REDUNDANT for NK-cell development — "
            "without IL-15/γc → no NK cells. "
            "B cells are PRESERVED (B+) because B-cell development uses BLNK/Bruton "
            "pathway (not γc-dependent) — but B cells are non-functional without T-helper cells. "
            "PATHOMECHANISM: hemizygous loss-of-function variants in IL2RG (X-linked) → "
            "absent functional γc → JAK3 cannot signal → "
            "T-LYMPHOCYTES: profound lymphopenia; thymic shadow absent on chest X-ray; "
            "NK-CELLS: absent (NK-); "
            "B-LYMPHOCYTES: present but dysfunctional (B+; maternal antibodies present at birth "
            "masking the diagnosis for first weeks — Newborn Screening (NBS) via TREC detects "
            "T-cell lymphopenia before maternal antibodies wane). "
            "CLINICAL CONSEQUENCE: complete combined immunodeficiency presenting in infancy "
            "with opportunistic infections, failure to thrive, graft-versus-host disease (GvHD) "
            "from maternal T-cell engraftment; death by age 1-2 years without curative treatment."
        ),
        "disease_type": (
            "X-linked Severe Combined Immunodeficiency (X-SCID; OMIM #300400); XLR; "
            "T-B+NK- phenotype; most common SCID in males (~50% all SCID); "
            "curative: HSCT (HLA-matched sibling preferred) OR gene therapy OTL-101 (FDA-approved "
            "2024 for ≥2-year-olds without HLA-matched donor); IgRT post-HSCT if B-cell "
            "dysfunction persists; NBS by TREC essential for pre-symptomatic diagnosis"
        ),
        "locus": "Xq13.1",
        "omim_gene": 308380,
        "omim_disease": 300400,
        "inheritance": (
            "X-LINKED RECESSIVE: hemizygous LOF in males = SCID; "
            "heterozygous females = carriers (usually asymptomatic with normal immunity via "
            "X-inactivation; non-random X-inactivation in lymphocytes is diagnostic of carrier status). "
            "DE NOVO RATE: ~30-40% of cases arise de novo (no family history). "
            "NBS: TREC (T-cell receptor excision circle) assay detects T-cell lymphopenia "
            "on Day 2 Guthrie card; mandated in most high-income countries. "
            "MATERNAL GvHD: transplacentally acquired maternal T-cells engraft in absent thymus → "
            "erythroderma, hepatitis, failure to thrive — pathognomonic of T- SCID in neonate. "
            "INFECTION TRIGGER: live attenuated vaccines (BCG, rotavirus, MMR) are ABSOLUTELY "
            "CONTRAINDICATED until immunity confirmed — BCG-osis (disseminated BCG infection) "
            "is a recognised cause of death if BCG given before SCID diagnosis."
        ),
        "phenotype": (
            "ONSET: infancy (typically 3-6 months as maternal antibodies wane). "
            "INFECTIONS: recurrent, severe, opportunistic — Pneumocystis jirovecii pneumonia (PJP) "
            "50-70%; CMV pneumonitis; candidiasis (mucocutaneous + systemic); viral (RSV, parainfluenza, "
            "adenovirus) — life-threatening lower respiratory infections. "
            "FAILURE TO THRIVE: profound; weight < 3rd centile; feeding difficulties. "
            "ABSENT THYMIC SHADOW: chest X-ray — absent thymus (normally visible in neonate). "
            "MATERNAL ENGRAFTMENT GvHD: erythroderma, transaminitis, diarrhoea — from maternal T-cells "
            "engrafting in immunodeficient host without rejection. "
            "LYMPHOPENIA: CD3+ T-cells virtually absent (<300/µL); NK cells absent; "
            "B cells present but non-functional (hypogammaglobulinaemia develops as maternal Ig wanes). "
            "IMMUNOGLOBULINS: initially normal (maternal IgG); decline after 3-6 months. "
            "LABORATORY: TREC = 0 (absent on NBS); lymphocyte proliferation assay = absent. "
            "NATURAL HISTORY: untreated = death by 12-18 months."
        ),
        "treatment_options": [
            "HSCT (Haematopoietic Stem Cell Transplantation) — CURATIVE: HLA-matched sibling "
            "donor = BEST outcome (>90% survival); HLA-matched unrelated or haploidentical (T-cell "
            "depleted) alternatives; conditioning intensity depends on donor match; "
            "pre-symptomatic HSCT (via NBS) dramatically improves outcome — survival >95% "
            "vs ~75% symptomatic cases; timing: ideally <3.5 months of age",
            "Gene therapy OTL-101 (autologous CD34+ HSC gene-corrected with γc lentiviral vector) — "
            "FDA-approved 2024 (Orchard Therapeutics); for patients ≥2 years without HLA-matched "
            "sibling donor; excellent T and NK cell reconstitution; avoids GvHD; "
            "requires mild myeloablative conditioning (busulfan); European conditional approval "
            "since 2021 (OTL-101); superior NK reconstitution vs retroviral vectors",
            "Protective isolation (positive-pressure laminar flow room) while awaiting HSCT/gene therapy; "
            "NO visitors with viral illness; strict hand hygiene; filtered air",
            "PJP prophylaxis: co-trimoxazole (trimethoprim-sulfamethoxazole) MANDATORY until "
            "immunity reconstituted post-HSCT — PJP most common cause of death in undiagnosed SCID",
            "IgRT (IgG replacement therapy): IV/SC immunoglobulin if B-cell dysfunction persists "
            "post-HSCT; target IgG trough ≥8 g/L",
            "AVOID ABSOLUTELY: live attenuated vaccines (BCG, MMR, rotavirus, varicella, yellow fever, "
            "oral polio) — can cause fatal disseminated infection; irradiated/CMV-negative "
            "blood products until immune reconstitution (prevent transfusion-associated GvHD + CMV)",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "BCG VACCINE — ABSOLUTE CI: Fatal Disseminated BCG-osis",
                "body": (
                    "BCG (live attenuated Mycobacterium bovis) given to a T-cell SCID patient "
                    "causes disseminated BCG infection (BCG-osis): fatal hepatitis, bone marrow "
                    "failure, lymphadenitis. BCG is routinely given at birth in many countries "
                    "BEFORE X-SCID is diagnosed. NBS TREC screening before BCG administration "
                    "is the only prevention. If BCG given before SCID diagnosis: "
                    "anti-mycobacterial triple therapy (isoniazid + rifampicin + ethambutol) "
                    "mandatory; no HSCT until BCG infection cleared."
                ),
            },
            {
                "type": "danger",
                "title": "LIVE VACCINES — ABSOLUTELY CONTRAINDICATED in ALL SCID",
                "body": (
                    "ALL live attenuated vaccines (BCG, MMR, rotavirus, varicella, yellow fever, "
                    "oral polio, LAIV influenza) are absolutely contraindicated until confirmed "
                    "immune reconstitution post-HSCT or gene therapy. "
                    "Rotavirus vaccine: given at 6-8 weeks by schedule — check TREC/immune status "
                    "BEFORE administering. Document CI in chart."
                ),
            },
        ],
        "clinical_rules": [
            "TREC=0 on NBS + absent T-cells → X-SCID or SCID: treat as medical emergency, refer immediately to immunology",
            "BCG AT BIRTH: if NBS not done before BCG → screen URGENTLY for BCG-osis before HSCT",
            "MATERNAL GvHD: erythroderma in neonate with lymphopenia → check maternal T-cell engraftment",
            "ABSENT THYMUS on chest X-ray in infant with infections → SCID until proven otherwise",
            "CARRIER FEMALES: check non-random X-inactivation in lymphocytes if family history",
        ],
        "key_distinguishing": "T-B+NK- phenotype + TREC=0 + Xq13.1 = X-SCID/IL2RG (B+ distinguishes from RAG1 which is T-B-NK+)",
        "severity_weights": {"Severe": 0.70, "Moderate": 0.20, "Mild": 0.10},
        "prevalence_per_100k": 0.5,
        "hsct_rate_pct": 85,
        "gene_therapy_rate_pct": 20,
        "infection_rate_pct": 95,
        "failure_to_thrive_pct": 88,
        "bcg_complication_pct": 18,
        "de_novo_pct": 35,
    },

    # ── ADA — ADA-SCID ───────────────────────────────────────────────────────
    {
        "gene": "ADA",
        "protein": "Adenosine Deaminase (ADA)",
        "alias": "ADA; OMIM gene 608958; 20q13.12; ~363 aa; ADA-SCID (OMIM #102700); AR; T-B-NK- SCID; metabolic SCID; dATP toxicity; PEG-ADA (elapegademase); Strimvelis gene therapy (EMA-approved); HSCT",
        "aa": "~363 aa",
        "kDa": "~41 kDa",
        "mechanism": (
            "ADA encodes adenosine deaminase, a ubiquitous purine salvage enzyme that converts "
            "adenosine → inosine and 2'-deoxyadenosine → 2'-deoxyinosine, preventing toxic "
            "accumulation of deoxyadenosine triphosphate (dATP). "
            "NORMAL FUNCTION: ADA is expressed ubiquitously but is HIGHEST in lymphocytes "
            "(particularly immature thymocytes), where it is essential for processing excess "
            "deoxyadenosine generated during lymphocyte apoptosis. "
            "PATHOMECHANISM: biallelic LOF variants in ADA → absent ADA activity → "
            "accumulation of deoxyadenosine → intracellular phosphorylation → dATP pool expands → "
            "dATP inhibits ribonucleotide reductase → blocks DNA synthesis → "
            "lymphocyte apoptosis in thymus (T-cell lineage) + periphery (B + NK cells). "
            "RESULT: T-B-NK- SCID — profound lymphopenia affecting ALL lymphocyte lineages "
            "(unlike X-SCID where B cells are preserved). "
            "UNIQUE FEATURE: dATP also accumulates in erythrocytes (measured as diagnostic test); "
            "skeletal dysplasia in 50%+ (costochondral flaring, cupped ribs — dATP affects "
            "chondrocyte maturation); neurological features (cognitive, neuromotor) — ADA is "
            "expressed in CNS; some patients with hypomorphic (partial) ADA deficiency present "
            "later in life with milder combined immunodeficiency (delayed-onset ADA-SCID)."
        ),
        "disease_type": (
            "ADA-Severe Combined Immunodeficiency (ADA-SCID; OMIM #102700); AR; "
            "T-B-NK- phenotype (~15% of all SCID); metabolic cause of SCID; "
            "unique skeletal + neurological features; three therapeutic options: "
            "(1) Enzyme Replacement Therapy (ERT): PEG-ADA (elapegademase/Revcovi FDA2018) — "
            "bridge to definitive therapy; (2) Gene therapy: Strimvelis (EMA-approved 2016, "
            "retroviral vector, MRC Milan, curative 80-90%); (3) HSCT (HLA-matched preferred)"
        ),
        "locus": "20q13.12",
        "omim_gene": 608958,
        "omim_disease": 102700,
        "inheritance": (
            "AUTOSOMAL RECESSIVE: biallelic LOF variants (compound heterozygous or homozygous). "
            "DELAYED ONSET: hypomorphic variants with residual ADA activity → late-onset "
            "(childhood, adolescence, even adulthood) milder disease — 'delayed-onset ADA-SCID'; "
            "initial lymphocyte counts may be normal before progressive decline. "
            "NEWBORN SCREENING: TREC assay detects T-lymphopenia; supplementary metabolic NBS "
            "(ADA enzyme activity in dried blood spot) available in some programs. "
            "ERYTHROCYTE dATP: key diagnostic test — dATP/ATP ratio elevated (dATP measured "
            "in erythrocytes because ADA is absent in RBCs normally → dATP accumulates). "
            "GENOTYPE-PHENOTYPE: null/null variants → neonatal SCID; hypomorphic variants "
            "(splice, partial missense) → delayed-onset; "
            "residual ADA activity <1% → typical SCID presentation."
        ),
        "phenotype": (
            "CLASSICAL ADA-SCID (null variants): onset at 3-6 months (overlaps X-SCID clinically). "
            "INFECTIONS: PJP, CMV, candidaemia, recurrent bacterial infections; same opportunistic "
            "infection spectrum as other SCID forms. "
            "SKELETAL DYSPLASIA: costochondral flaring, cupped/flared ribs, platyspondyly "
            "(vertebral flattening), pelvic dysplasia — unique to ADA-SCID, not seen in other SCID. "
            "NEUROLOGICAL: cognitive delay, sensorineural hearing loss, behavioural abnormalities, "
            "hypertonia — partially explained by CNS ADA expression and dATP neurotoxicity; "
            "persists even post-HSCT/gene therapy. "
            "LABORATORY: T-B-NK- (ALL lymphocyte lineages depleted); dATP/ATP ratio elevated "
            "in erythrocytes (diagnostic); ADA enzyme activity <1% in lymphocytes/erythrocytes. "
            "DELAYED ONSET (hypomorphic): recurrent sino-pulmonary infections, autoimmunity "
            "(ITP, autoimmune haemolytic anaemia), partial lymphopenia — mimics CVID or WAS."
        ),
        "treatment_options": [
            "HSCT (Haematopoietic Stem Cell Transplantation) — curative: HLA-matched sibling "
            "preferred; outcomes similar to other SCID forms (>90% survival HLA-matched sibling); "
            "myeloablative conditioning needed for full donor engraftment",
            "Gene therapy Strimvelis (EMA-approved 2016; autologous CD34+ HSC retroviral vector "
            "corrected ex vivo; MRC Milan group; OTL-101 is lentiviral successor FDA-submitted 2024); "
            "curative 80-90%; avoids GvHD; requires mild myeloablative conditioning (busulfan); "
            "preferred for patients without HLA-matched sibling donor",
            "PEG-ADA (elapegademase-lvlr / Revcovi; FDA-approved 2018; polyethylene glycol–modified "
            "bovine ADA): enzyme replacement therapy given IM 2×/week; NOT curative — "
            "bridging therapy to HSCT/gene therapy; maintains partial immunity; "
            "allows vaccinations and some normal immune function; "
            "DISADVANTAGE: lifelong therapy if no transplant; expensive; some patients develop "
            "anti-ADA antibodies; immune reconstitution is partial",
            "PJP prophylaxis: co-trimoxazole MANDATORY until immune reconstitution",
            "IgRT: IgG replacement if B-cell dysfunction; target trough ≥8 g/L",
            "AVOID: live vaccines until immune reconstitution confirmed; irradiated/CMV-negative blood products",
        ],
        "drug_alerts": [
            {
                "type": "warning",
                "title": "PEG-ADA is BRIDGE not CURE — Plan Definitive Therapy",
                "body": (
                    "PEG-ADA (elapegademase) provides partial immune reconstitution and prevents "
                    "opportunistic infections but is NOT curative. Immune responses remain "
                    "suboptimal — patients still at risk of severe infections. "
                    "PLAN HSCT or gene therapy (Strimvelis/OTL-101) — do not allow PEG-ADA "
                    "to become the only long-term strategy without discussion of definitive options. "
                    "Anti-PEG-ADA antibodies develop in ~30% → loss of efficacy."
                ),
            },
            {
                "type": "info",
                "title": "SKELETAL DYSPLASIA: Screen with Skeletal Survey at Diagnosis",
                "body": (
                    "ADA-SCID is unique among SCID forms in causing skeletal dysplasia — "
                    "costochondral flaring (widened costochondral junctions), cupped ribs, "
                    "platyspondyly. A chest X-ray and skeletal survey at diagnosis documents "
                    "extent. These findings improve partially with treatment but neurological "
                    "features may persist."
                ),
            },
        ],
        "clinical_rules": [
            "T-B-NK- phenotype = ADA-SCID or RAG1/2 SCID: measure ADA enzyme activity in RBCs to distinguish",
            "SKELETAL DYSPLASIA (rib flaring) + SCID: ADA-SCID PATHOGNOMONIC combination",
            "DELAYED ONSET ADA: normal TREC at birth is possible — screen all children with recurrent sino-pulmonary infections + lymphopenia for ADA activity",
            "PEG-ADA ANTIBODIES: check anti-ADA IgG titres if immune reconstitution worsens on therapy",
            "NEUROLOGICAL FOLLOW-UP: cognitive + hearing assessment regardless of treatment — CNS dATP toxicity partially irreversible",
        ],
        "key_distinguishing": "T-B-NK- + SKELETAL DYSPLASIA (costochondral flaring) + dATP elevation in RBCs = ADA-SCID (3-lineage lymphopenia distinguishes from X-SCID B+)",
        "severity_weights": {"Severe": 0.60, "Moderate": 0.28, "Mild": 0.12},
        "prevalence_per_100k": 0.3,
        "hsct_rate_pct": 60,
        "gene_therapy_rate_pct": 30,
        "infection_rate_pct": 90,
        "failure_to_thrive_pct": 80,
        "bcg_complication_pct": 10,
        "de_novo_pct": 0,
    },

    # ── RAG1 — T-B-NK+ SCID / Omenn Syndrome ────────────────────────────────
    {
        "gene": "RAG1",
        "protein": "Recombination Activating Gene 1 Protein (RAG1)",
        "alias": "RAG1; OMIM gene 179615; 11p13; ~1043 aa; T-B-NK+ SCID (OMIM #601457) + Omenn Syndrome (OMIM #603554); AR; V(D)J recombination; RAG1/RAG2 complex; HSCT curative",
        "aa": "~1043 aa",
        "kDa": "~119 kDa",
        "mechanism": (
            "RAG1 (with RAG2) forms the RAG recombinase complex that catalyses V(D)J "
            "recombination — the somatic DNA rearrangement required to assemble functional "
            "immunoglobulin (Ig) and T-cell receptor (TCR) genes. "
            "NORMAL FUNCTION: RAG1 binds recombination signal sequences (RSSs) flanking "
            "V, D, and J gene segments; RAG2 co-binds and activates catalytic activity; "
            "RAG complex cleaves DNA at RSSs → hairpin intermediates → NHEJ rejoins the "
            "coding ends → functional Ig/TCR gene assembled. "
            "V(D)J recombination is ESSENTIAL for lymphocyte development: "
            "B cells require IgH + IgL rearrangement to mature; "
            "T cells require TCRβ + TCRα rearrangement to mature. "
            "NK cells do NOT require V(D)J recombination → NK cells are PRESERVED (NK+). "
            "PATHOMECHANISM: "
            "NULL/SEVERE LOF variants → absent RAG recombinase → complete block in "
            "B and T lymphocyte development → T-B-NK+ SCID (T-cell and B-cell absent; NK preserved). "
            "HYPOMORPHIC/PARTIAL LOF variants → residual RAG activity → oligoclonal T cells "
            "escape with limited TCR repertoire → homeostatic expansion → autoreactive T cells "
            "attack host tissues → OMENN SYNDROME (erythroderma, eosinophilia, hepatosplenomegaly, "
            "lymphadenopathy, profound immunodeficiency with paradoxically elevated autoreactive T cells)."
        ),
        "disease_type": (
            "T-B-NK+ SCID (OMIM #601457) and/or Omenn Syndrome (OMIM #603554); AR; "
            "~15-20% of all SCID; V(D)J recombination defect; T-B-NK+ phenotype (NK preserved); "
            "HSCT curative (myeloablative conditioning required for T and B cell engraftment); "
            "Omenn = hypomorphic RAG1 variant with paradoxical erythroderma + eosinophilia"
        ),
        "locus": "11p13",
        "omim_gene": 179615,
        "omim_disease": 601457,
        "inheritance": (
            "AUTOSOMAL RECESSIVE: biallelic LOF or hypomorphic variants. "
            "GENOTYPE-PHENOTYPE CORRELATION: null/null → typical T-B-NK+ SCID; "
            "hypomorphic compound heterozygotes (one null + one partial) → Omenn Syndrome "
            "(or leaky SCID / combined immunodeficiency spectrum). "
            "RAG1 and RAG2 are immediately adjacent at 11p13 — deletions/rearrangements "
            "can affect both genes simultaneously → critical to sequence BOTH genes. "
            "RESIDUAL ACTIVITY: even 1-3% of normal RAG activity produces Omenn-spectrum; "
            "Omenn Syndrome patients are NOT simply 'mild SCID' — they have severe immunopathology "
            "from autoreactive T cells + impaired regulatory T cells."
        ),
        "phenotype": (
            "T-B-NK+ SCID (null/null): classic SCID presentation — "
            "profound lymphopenia (T=0, B=0); NK cells present; "
            "failure to thrive; opportunistic infections (PJP, CMV, candida, adenovirus); "
            "absent IgG (hypogammaglobulinaemia); absent immunological memory. "
            "OMENN SYNDROME (hypomorphic): "
            "ERYTHRODERMA — generalised skin redness/scaling (100%); "
            "EOSINOPHILIA — blood eosinophil count markedly elevated; "
            "HEPATOSPLENOMEGALY — lymphoid infiltration; "
            "LYMPHADENOPATHY — despite profound immunodeficiency; "
            "ELEVATED IgE — despite overall hypogammaglobulinaemia; "
            "AUTOREACTIVE T CELLS: limited TCR repertoire skewed to autoreactive clones "
            "targeting skin, gut, liver; "
            "INFECTIONS: paradoxically variable — some opportunistic infections but some patients "
            "initially appear relatively well despite profound immune dysregulation. "
            "DDx: maternal engraftment GvHD; drug hypersensitivity; congenital Netherton syndrome."
        ),
        "treatment_options": [
            "HSCT — curative for BOTH T-B-NK+ SCID and Omenn Syndrome: "
            "MYELOABLATIVE conditioning required (unlike X-SCID where reduced-intensity may suffice "
            "because host NK cells + residual lymphocytes in RAG1 can resist engraftment); "
            "HLA-matched sibling preferred; outcomes good if performed early pre-symptomatically",
            "Omenn Syndrome pre-HSCT stabilisation: "
            "immunosuppression (corticosteroids, cyclosporin A) to control autoreactive T cells; "
            "protective isolation; infection treatment; "
            "IgRT to provide immunoglobulin until HSCT",
            "PJP prophylaxis: co-trimoxazole MANDATORY until immune reconstitution",
            "Antifungal prophylaxis: fluconazole for candida risk",
            "Antiviral prophylaxis: aciclovir for HSV/VZV risk",
            "IgRT: IgG replacement post-HSCT if B-cell reconstitution incomplete; "
            "lifelong if no B-cell reconstitution achieved",
            "AVOID: live vaccines (absolutely); non-irradiated blood products; "
            "concurrent RAG2 testing mandatory (gene at same locus — both must be sequenced)",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "OMENN SYNDROME ≠ GvHD: Critical Differential Diagnosis",
                "body": (
                    "Omenn Syndrome (RAG1/RAG2 hypomorphic) is frequently misdiagnosed as: "
                    "(1) Maternal T-cell engraftment GvHD; "
                    "(2) Drug reaction with eosinophilia and systemic symptoms (DRESS); "
                    "(3) Netherton Syndrome (SPINK5). "
                    "KEY DISTINGUISHING: Omenn = autologous autoreactive T cells (NOT maternal) "
                    "with limited TCR repertoire; maternal engraftment = maternal T cells by chimerism. "
                    "Send HLA typing + chimerism analysis to distinguish. "
                    "Treatment for Omenn (cyclosporin A) is contraindicated in true GvHD."
                ),
            },
            {
                "type": "warning",
                "title": "SEQUENCE BOTH RAG1 AND RAG2 — Same Locus (11p13)",
                "body": (
                    "RAG1 and RAG2 are immediately adjacent at 11p13. Deletions spanning "
                    "both genes are common. If only RAG1 is sequenced and found negative, "
                    "RAG2 may harbour the pathogenic variant. "
                    "MANDATE: sequence BOTH genes + perform copy number analysis (MLPA) "
                    "when RAG1/RAG2 SCID is suspected."
                ),
            },
        ],
        "clinical_rules": [
            "T-B-NK+ phenotype: NK preserved helps distinguish from X-SCID (T-B+NK-) and ADA-SCID (T-B-NK-)",
            "OMENN SYNDROME: erythroderma + eosinophilia + immunodeficiency → RAG1/2 hypomorphic until proven otherwise",
            "MYELOABLATIVE CONDITIONING for HSCT: required in RAG1 SCID — residual NK cells reject donor cells without conditioning",
            "SEQUENCE RAG2 MANDATORY when RAG1 tested: co-localised at 11p13",
            "MATERNAL CHIMERISM ASSAY: distinguish Omenn from maternal GvHD before starting cyclosporin A",
        ],
        "key_distinguishing": "T-B-NK+ phenotype + Omenn Syndrome (erythroderma+eosinophilia) = RAG1/RAG2 (NK+ distinguishes from X-SCID NK- and ADA-SCID NK-)",
        "severity_weights": {"Severe": 0.55, "Moderate": 0.30, "Mild": 0.15},
        "prevalence_per_100k": 0.4,
        "hsct_rate_pct": 90,
        "gene_therapy_rate_pct": 5,
        "infection_rate_pct": 85,
        "failure_to_thrive_pct": 78,
        "bcg_complication_pct": 12,
        "de_novo_pct": 0,
    },

    # ── BTK — X-linked Agammaglobulinaemia (XLA) ─────────────────────────────
    {
        "gene": "BTK",
        "protein": "Bruton's Tyrosine Kinase (BTK)",
        "alias": "BTK; OMIM gene 300300; Xq22.1; ~659 aa; X-linked Agammaglobulinaemia XLA (OMIM #300755); XLR; absent B cells; profound hypogammaglobulinaemia; IgRT lifelong; ibrutinib structural homology",
        "aa": "~659 aa",
        "kDa": "~76 kDa",
        "mechanism": (
            "BTK (Bruton's Tyrosine Kinase) is a non-receptor cytoplasmic tyrosine kinase of the "
            "Tec kinase family. It is expressed in B-lineage cells (from pro-B to mature B cells "
            "and plasma cells), myeloid cells (monocytes, macrophages, platelets), and "
            "mast cells — but NOT in T cells or NK cells. "
            "NORMAL FUNCTION IN B-CELL DEVELOPMENT: "
            "Pre-BCR (pre-B cell receptor) signalling: successful IgH (immunoglobulin heavy chain) "
            "V(D)J rearrangement → pre-BCR surface expression → BTK is recruited to the receptor → "
            "BTK autophosphorylates (Tyr551 trans; Tyr223 auto) → activates PLCγ2 → IP3 + DAG → "
            "Ca²⁺ mobilisation + PKC activation → survival + proliferation signals → "
            "pre-B cell expands and undergoes IgL (light chain) rearrangement → immature B cell. "
            "Without BTK: pre-BCR signal transduction fails → pre-B cells cannot expand → "
            "B-cell development arrested at pro-B → pre-B transition → "
            "RESULT: absent mature B cells in periphery → profound agammaglobulinaemia. "
            "BTK IS ALSO THE TARGET OF IBRUTINIB (BTK inhibitor used in haematological malignancies) — "
            "ibrutinib causes acquired agammaglobulinaemia similar to XLA as a mechanism-based effect."
        ),
        "disease_type": (
            "X-linked Agammaglobulinaemia (XLA; OMIM #300755); XLR; ~85% of primary agammaglobulinaemia; "
            "absent B cells + profound hypogammaglobulinaemia (IgG <2 g/L, IgA <0.1 g/L, IgM <0.1 g/L); "
            "normal T cells and NK cells; recurrent encapsulated bacterial infections; "
            "lifelong IgRT (subcutaneous or IV immunoglobulin) = mainstay; "
            "BTK inhibitors CAUSE similar phenotype (ibrutinib) — mechanism-based adverse effect"
        ),
        "locus": "Xq22.1",
        "omim_gene": 300300,
        "omim_disease": 300755,
        "inheritance": (
            "X-LINKED RECESSIVE: hemizygous LOF in males; females are usually asymptomatic carriers "
            "(30-50% non-random X-inactivation in B cells — can sometimes be demonstrated diagnostically). "
            "DE NOVO RATE: ~40% — significant proportion without family history. "
            "MATERNAL CARRIERS: may have mildly reduced B-cell counts and Ig levels but usually "
            "clinically unaffected. "
            "GENOTYPE-PHENOTYPE: no strong correlation — same mutation can give variable residual BTK "
            "expression; ~30% of XLA have no detectable BTK protein by flow cytometry of monocytes "
            "(diagnostic test: BTK protein by intracellular flow cytometry). "
            "VARIANT TYPES: missense (~40%), stop/frameshift (~40%), splice (~15%), deletions (~5%); "
            "PH domain missense (N-terminus) tend to abolish membrane targeting."
        ),
        "phenotype": (
            "ONSET: typically 6-18 months (as maternal antibodies wane) — later than SCID forms. "
            "RECURRENT BACTERIAL INFECTIONS: encapsulated organisms particularly affected — "
            "Streptococcus pneumoniae, Haemophilus influenzae, Staphylococcus aureus; "
            "Mycoplasma arthritis (chronic non-pyogenic joint infection — PATHOGNOMONIC for XLA); "
            "enteroviruses: life-threatening in XLA — poliovirus (VAPP from live oral polio vaccine), "
            "echovirus (fatal meningoencephalitis/dermatomyositis). "
            "T CELL IMMUNITY: NORMAL — no opportunistic infections (PJP, CMV) unless on treatment. "
            "LABORATORY: "
            "ALL Ig isotypes profoundly low (IgG <2 g/L; IgA undetectable; IgM undetectable); "
            "B cells virtually absent (<2% CD19+ of lymphocytes; normally 10-20%); "
            "T cells: NORMAL count and function (T-cell normal is KEY DDx from SCID); "
            "BTK protein absent on monocyte/B-cell intracellular flow cytometry; "
            "TREC: normal (T cells present). "
            "PHYSICAL EXAMINATION: absent/hypoplastic tonsils (no B cells → no lymphoid tissue); "
            "absent palpable lymph nodes; splenomegaly uncommon."
        ),
        "treatment_options": [
            "IgRT (Immunoglobulin Replacement Therapy) — LIFELONG MAINSTAY: "
            "IVIG (IV immunoglobulin; Privigen, Gamunex, Flebogamma) every 3-4 weeks "
            "OR SCIG (subcutaneous; Hizentra, Cuvitru) weekly or bi-weekly; "
            "target IgG trough ≥8 g/L (some guidelines ≥10 g/L if recurrent infections); "
            "prevents 80-90% of serious bacterial infections; "
            "ADVANTAGE: avoids HSCT risks; DISADVANTAGE: lifelong infusions",
            "Antibiotic prophylaxis: co-amoxiclav or azithromycin for breakthrough infections; "
            "ciprofloxacin for Mycoplasma arthritis",
            "Aggressive treatment of pulmonary infections: prevents bronchiectasis — "
            "annual PFTs (pulmonary function tests) from age 5 years; "
            "CT chest every 3-5 years to monitor for bronchiectasis progression",
            "ENTEROVIRUS SURVEILLANCE: annual stool surveillance for enterovirus shedding; "
            "if enteroviruses detected → intensify IgRT; pleconaril (compassionate use) "
            "for symptomatic enteroviral disease; no effective vaccine available for XLA patients",
            "AVOID: live oral polio vaccine (VAPP — vaccine-associated paralytic poliomyelitis; "
            "XLA patients cannot clear live poliovirus → disseminated infection); "
            "ibrutinib and other BTK inhibitors exacerbate BTK pathway deficiency",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "ORAL POLIO VACCINE — ABSOLUTE CI: Vaccine-Associated Paralytic Poliomyelitis",
                "body": (
                    "Live oral polio vaccine (OPV/Sabin) given to XLA patients causes "
                    "vaccine-associated paralytic poliomyelitis (VAPP) — inability to clear "
                    "live attenuated poliovirus → progressive CNS infection → paralysis/death. "
                    "Use ONLY inactivated polio vaccine (IPV/Salk) in all XLA patients "
                    "and their household contacts. Document OPV as absolute CI."
                ),
            },
            {
                "type": "warning",
                "title": "MYCOPLASMA ARTHRITIS: Non-Pyogenic, Aspirate Negative — Specific DDx in XLA",
                "body": (
                    "XLA patients develop Mycoplasma (M. pneumoniae, M. hominis, U. urealyticum) "
                    "arthritis — non-purulent, culture-negative on standard media. "
                    "Gram stain negative. SUSPICION: swollen joint + XLA + negative routine cultures "
                    "→ send Mycoplasma/Ureaplasma PCR on joint aspirate. "
                    "Treat with doxycycline (adults) or azithromycin (children). "
                    "JOINT DESTRUCTION occurs rapidly if untreated."
                ),
            },
        ],
        "clinical_rules": [
            "ABSENT B CELLS + PROFOUND HYPOGAMMAGLOBULINAEMIA + NORMAL T CELLS = XLA/BTK (not SCID — T cells normal is key)",
            "BTK PROTEIN by monocyte intracellular flow cytometry: fast, cheap diagnostic screen before gene sequencing",
            "ABSENT TONSILS on physical exam in child with recurrent pneumococcal infections → XLA screen immediately",
            "OPV ABSOLUTELY CI: document in all XLA medical records; household contacts also must avoid OPV",
            "ENTEROVIRUS SHEDDING annual stool PCR: chronic shedding → CNS risk; pleconaril compassionate access",
        ],
        "key_distinguishing": "Absent B cells + ALL Ig isotypes low + NORMAL T cells + absent tonsils = XLA/BTK (normal T cells distinguishes from SCID forms)",
        "severity_weights": {"Mild": 0.35, "Moderate": 0.45, "Severe": 0.20},
        "prevalence_per_100k": 1.0,
        "hsct_rate_pct": 2,
        "gene_therapy_rate_pct": 0,
        "infection_rate_pct": 88,
        "failure_to_thrive_pct": 30,
        "bcg_complication_pct": 2,
        "de_novo_pct": 40,
    },

    # ── CYBB — X-linked Chronic Granulomatous Disease (X-CGD) ────────────────
    {
        "gene": "CYBB",
        "protein": "Cytochrome b-245 Beta Subunit (gp91phox; NOX2)",
        "alias": "CYBB; OMIM gene 300481; Xp21.1; ~570 aa; X-CGD (OMIM #306400); XLR; NADPH oxidase defect; NBT test; nitroblue tetrazolium; DHR flow cytometry; catalase-positive bacteria; itraconazole + TMP-SMX prophylaxis; HSCT curative",
        "aa": "~570 aa",
        "kDa": "~91 kDa (gp91phox, glycoprotein)",
        "mechanism": (
            "CYBB encodes gp91phox (glycoprotein 91 kDa; NOX2), the large subunit of the "
            "phagocyte NADPH oxidase complex (NOX2 complex). "
            "NORMAL FUNCTION — RESPIRATORY BURST: upon phagocytosis of microorganisms, "
            "phagocytes (neutrophils, monocytes, macrophages) assemble the NOX2 complex at "
            "the phagosomal membrane: "
            "gp91phox (CYBB) + p22phox (CYBA) [membrane components; cytochrome b-245 heterodimer] + "
            "p47phox (NCF1) + p67phox (NCF2) + p40phox (NCF4) [cytosolic components] + "
            "Rac2 GTPase → assembled complex transfers electrons from cytoplasmic NADPH "
            "across the membrane to O₂ in the phagosome → superoxide (O₂•⁻) → "
            "H₂O₂ → HOCl (hypochlorous acid via myeloperoxidase) → reactive oxygen species (ROS) "
            "→ ROS kill phagocytosed bacteria and fungi. "
            "PATHOMECHANISM: hemizygous LOF variants in CYBB → absent/non-functional gp91phox → "
            "NOX2 complex cannot assemble → absent respiratory burst → phagocytes engulf bacteria "
            "and fungi normally but CANNOT kill them → intracellular survival of catalase-positive "
            "organisms → granuloma formation (host tissue attempt to wall off organisms) → "
            "CHRONIC GRANULOMATOUS DISEASE. "
            "CRITICAL DISTINCTION: catalase-positive organisms (S. aureus, Aspergillus, "
            "Klebsiella, Serratia, Nocardia, Burkholderia) are particularly dangerous — "
            "they produce catalase that destroys the host's H₂O₂ → "
            "catalase-negative organisms (S. pneumoniae, H. influenzae) provide their own H₂O₂ "
            "to the phagocyte → can be killed despite absent NOX2 (thus these organisms are "
            "NOT typical in CGD — their absence is clinically diagnostically informative)."
        ),
        "disease_type": (
            "X-linked Chronic Granulomatous Disease (X-CGD; OMIM #306400); XLR; "
            "~70% of all CGD (remainder AR CYBA/NCF1/NCF2/NCF4); "
            "absent NADPH oxidase respiratory burst; recurrent life-threatening infections "
            "with CATALASE-POSITIVE organisms (S. aureus, Aspergillus, Nocardia, Serratia, "
            "Burkholderia); colitis (granulomatous); lymphadenitis; "
            "lifelong TMP-SMX + itraconazole prophylaxis; "
            "IFN-γ (FDA-approved adjunct); HSCT curative"
        ),
        "locus": "Xp21.1",
        "omim_gene": 300481,
        "omim_disease": 306400,
        "inheritance": (
            "X-LINKED RECESSIVE: hemizygous LOF in males (gp91phox absent = X-CGD). "
            "CARRIER FEMALES: ~5-10% of female carriers are symptomatic ('carrier CGD') "
            "due to unfavourable lyonisation — low proportion of X-inactivation in phagocytes; "
            "DHR flow cytometry shows bimodal pattern in carriers (some cells normal, some absent). "
            "AUTOSOMAL RECESSIVE CGD (30%): CYBA (p22phox), NCF1 (p47phox), NCF2 (p67phox), "
            "NCF4 (p40phox) — each causes AR-CGD; phenotype similar to X-CGD. "
            "DIAGNOSTIC TEST: DHR (dihydrorhodamine 123) flow cytometry — gold standard; "
            "NBT (nitroblue tetrazolium) test (older, semi-quantitative); "
            "luminol chemiluminescence (research). "
            "VARIANT TYPES in CYBB: missense (~35%), frameshift (~35%), splice (~15%), "
            "large deletions (~15%); large deletions can include McLeod syndrome locus (XK gene) "
            "at Xp21.1 → McLeod syndrome (haemolytic anaemia + acanthocytes) + X-CGD."
        ),
        "phenotype": (
            "ONSET: usually childhood (median age 2-3 years) though diagnosis often delayed. "
            "INFECTIONS: "
            "LUNG: invasive pulmonary aspergillosis (MOST COMMON life-threatening infection — "
            "Aspergillus fumigatus) — recurrent; granulomatous pneumonia; S. aureus pneumonia; "
            "SKIN/LYMPH NODES: suppurative lymphadenitis; S. aureus skin abscesses + cellulitis; "
            "LIVER: S. aureus liver abscesses (PATHOGNOMONIC for CGD — S. aureus liver abscess "
            "is diagnostic); "
            "BONE: osteomyelitis (S. aureus, Serratia, Aspergillus); "
            "GASTROINTESTINAL: CGD colitis (granulomatous, mimics Crohn's — same genetics! "
            "NCF1 p47phox AR-CGD is a Crohn's susceptibility locus); gastric outlet obstruction "
            "(antral granuloma); perirectal abscess; "
            "UNUSUAL ORGANISMS: Burkholderia cepacia (lung); Nocardia (lung, brain); "
            "Serratia marcescens (bone, lung); Chromobacterium violaceum (tropical). "
            "LABORATORY: DHR flow: absent oxidative burst; NBT: no blue dye reduction. "
            "INFLAMMATORY MARKERS: markedly elevated ESR/CRP even when no active infection. "
            "HYPERGAMMAGLOBULINAEMIA: paradoxical IgG elevation (immune dysregulation)."
        ),
        "treatment_options": [
            "TMP-SMX (trimethoprim-sulfamethoxazole) prophylaxis — LIFELONG: "
            "prevents bacterial infections (S. aureus, Nocardia); "
            "give from diagnosis; monitor for adverse effects (cytopenias); "
            "alternative: dicloxacillin for TMP-SMX intolerance",
            "Itraconazole prophylaxis — LIFELONG: prevents invasive aspergillosis; "
            "200 mg/day (adults); monitor liver function; "
            "alternative: voriconazole or posaconazole in high-risk situations; "
            "NO fluconazole (inadequate Aspergillus coverage)",
            "IFN-γ (Actimmune, recombinant human IFN-γ): FDA-approved adjunct for X-CGD; "
            "reduces infection frequency ~70% in trials; "
            "dose: 50 mcg/m² SC three times per week; "
            "mechanism: enhances residual NADPH oxidase activity + upregulates killing pathways; "
            "anti-inflammatory benefit in CGD colitis",
            "HSCT — curative: increasingly offered early in well-conditioned patients; "
            "RIC (reduced intensity conditioning) + HLA-matched unrelated acceptable; "
            ">90% survival modern series; resolves both infections AND CGD colitis; "
            "TIMING: ideally before irreversible organ damage from recurrent infections",
            "Aggressive treatment of established infections: high-dose IV antifungals "
            "(voriconazole for Aspergillus), prolonged IV antibiotics for S. aureus abscesses; "
            "surgical drainage where required (liver abscess, lymph node)",
            "G-CSF adjunct for established infections: enhances neutrophil function transiently",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "S. AUREUS LIVER ABSCESS in CHILD — Think CGD First",
                "body": (
                    "Staphylococcus aureus liver abscess is PATHOGNOMONIC for CGD in children. "
                    "S. aureus is a catalase-positive organism that survives inside CGD phagocytes. "
                    "ANY child with S. aureus liver abscess → URGENT DHR flow cytometry to "
                    "exclude CGD before discharge. Missing CGD = more abscesses, "
                    "more Aspergillus, preventable deaths."
                ),
            },
            {
                "type": "danger",
                "title": "ASPERGILLUS PROPHYLAXIS MANDATORY — Lifelong Itraconazole",
                "body": (
                    "Invasive pulmonary aspergillosis is the leading infectious cause of death "
                    "in X-CGD. Itraconazole (or voriconazole) prophylaxis must continue "
                    "LIFELONG — do NOT stop during remission. "
                    "CGD patients can develop aspergillosis even without obvious neutropenia. "
                    "If breakthrough Aspergillus on itraconazole → switch to voriconazole "
                    "or posaconazole; add inhaled amphotericin for pulmonary disease."
                ),
            },
        ],
        "clinical_rules": [
            "DHR FLOW CYTOMETRY: gold-standard test — absent oxidative burst = CGD; bimodal = carrier female",
            "CATALASE-POSITIVE ORGANISMS ONLY: S. aureus, Aspergillus, Nocardia, Serratia, Burkholderia — CGD spectrum; streptococcal/H. influenzae infections are NOT typical",
            "LIVER ABSCESS IN CHILD: S. aureus liver abscess → screen CGD same admission",
            "CGD COLITIS: mimics Crohn's disease exactly; treat with steroids + IFN-γ (NOT anti-TNF which increases infection risk)",
            "MCLEOD SYNDROME: large CYBB deletions may include XK gene → McLeod haemolytic anaemia + acanthocytes — check blood film",
        ],
        "key_distinguishing": "Absent respiratory burst on DHR/NBT + S. aureus liver abscess + Aspergillus infections = CGD/CYBB (lymphocyte counts NORMAL — distinguishes from SCID forms)",
        "severity_weights": {"Moderate": 0.40, "Severe": 0.40, "Mild": 0.20},
        "prevalence_per_100k": 0.5,
        "hsct_rate_pct": 40,
        "gene_therapy_rate_pct": 5,
        "infection_rate_pct": 95,
        "failure_to_thrive_pct": 45,
        "bcg_complication_pct": 25,
        "de_novo_pct": 25,
    },

    # ── WAS — Wiskott-Aldrich Syndrome ───────────────────────────────────────
    {
        "gene": "WAS",
        "protein": "Wiskott-Aldrich Syndrome Protein (WASP)",
        "alias": "WAS; OMIM gene 300392; Xp11.23; ~502 aa; Wiskott-Aldrich Syndrome (OMIM #301000) + XLT (X-linked thrombocytopenia, OMIM #313900); XLR; classic triad: thrombocytopenia + eczema + immunodeficiency; WASP regulates actin polymerisation; HSCT curative; gene therapy",
        "aa": "~502 aa",
        "kDa": "~53 kDa",
        "mechanism": (
            "WAS encodes WASP (Wiskott-Aldrich Syndrome Protein), an actin nucleation-promoting "
            "factor expressed exclusively in haematopoietic cells. "
            "NORMAL FUNCTION: WASP links upstream signalling (Cdc42 GTPase via WASP's GBD domain; "
            "PIP2; Nck/Grb2 adaptor via proline-rich domain) to actin cytoskeletal "
            "remodelling via its C-terminal VCA domain that activates the Arp2/3 complex → "
            "branched actin polymerisation → essential for: "
            "T cell: immunological synapse formation (T-cell receptor clustering, focal adhesion); "
            "NK cell: cytotoxic granule polarisation; "
            "B cell: antigen receptor signalling amplification; "
            "platelet: proper morphology and activation; "
            "dendritic cell: migration and antigen presentation. "
            "PATHOMECHANISM: LOF variants in WAS → absent/truncated WASP → "
            "PLATELETS: small, dysmorphic microthrombocytes; accelerated splenic destruction; "
            "profound thrombocytopenia with small platelet size (MPV low — KEY diagnostic feature); "
            "T CELLS: impaired immunological synapse → poor TCR signalling → "
            "progressive T-lymphopenia + T-cell dysfunction; "
            "ECZEMA: dysregulated Th2 response (mechanism incompletely understood); "
            "AUTOIMMUNITY: 40-70% of WAS patients develop autoimmune disease (ITP, AIHA, "
            "vasculitis, IBD, nephritis); "
            "LYMPHOMA: 13-22% lifetime risk of EBV-associated lymphoma (B-cell lymphoma)."
        ),
        "disease_type": (
            "Wiskott-Aldrich Syndrome (OMIM #301000); XLR; "
            "CLASSIC TRIAD: thrombocytopenia (SMALL platelets — low MPV PATHOGNOMONIC) + "
            "eczema (atopic-type) + combined immunodeficiency (T + B + NK dysfunction); "
            "40-70% autoimmune complications; 13-22% lymphoma risk; "
            "HSCT curative — OS >90% HLA-matched sibling early HSCT; "
            "gene therapy (OTL-103 lentiviral) in clinical trials"
        ),
        "locus": "Xp11.23",
        "omim_gene": 300392,
        "omim_disease": 301000,
        "inheritance": (
            "X-LINKED RECESSIVE: hemizygous LOF or GOF (gain-of-function) variants. "
            "GENOTYPE-PHENOTYPE SPECTRUM: "
            "NULL variants (no WASP protein) → classic WAS (severe triad + autoimmunity + lymphoma); "
            "MISSENSE (residual protein) → milder phenotype: X-linked thrombocytopenia (XLT) "
            "with mild/absent eczema and mild immunodeficiency; "
            "GOF missense (constitutive WASP activation in GBD domain) → X-linked neutropenia "
            "(XLN; myelodysplasia-like syndrome; different disease!). "
            "WASP protein expression by flow cytometry: absent = null (WAS); "
            "reduced = missense (XLT spectrum); absent in WAS, present (low) in XLT. "
            "DE NOVO RATE: ~30%. "
            "FEMALE CARRIERS: can develop autoimmune thrombocytopenia or mild eczema "
            "if X-inactivation unfavourable."
        ),
        "phenotype": (
            "THROMBOCYTOPENIA: usually severe (20,000-50,000/µL); "
            "platelet size SMALL (MPV low — critical distinguishing feature from ITP where MPV is HIGH); "
            "bleeding manifestations: petechiae, easy bruising, GI bleeding, ICH (intracranial haemorrhage). "
            "ECZEMA: atopic pattern; may be severe and infected. "
            "IMMUNODEFICIENCY: progressive combined T+B deficiency; "
            "T-cell lymphopenia (progressive); hypogammaglobulinaemia (IgG low, IgM low); "
            "poor vaccine responses; recurrent sino-pulmonary infections (H. influenzae, S. pneumoniae); "
            "opportunistic infections (PJP, CMV, HSV) in severe cases. "
            "AUTOIMMUNE COMPLICATIONS (40-70%): "
            "haemolytic anaemia (Coombs-positive); ITP (paradoxically — autoantibodies against platelets); "
            "vasculitis; inflammatory bowel disease; nephritis. "
            "LYMPHOMA RISK (13-22%): EBV-driven B-cell lymphoma; "
            "risk increases with age if no curative treatment."
        ),
        "treatment_options": [
            "HSCT — CURATIVE for classic WAS: "
            "HLA-matched sibling: >90% survival; resolves thrombocytopenia, eczema, immunodeficiency, "
            "and reduces autoimmune + lymphoma risk; "
            "perform EARLY (ideally age <5 years) before EBV-related lymphoma develops; "
            "myeloablative conditioning preferred for full engraftment (prevents mixed chimerism → "
            "incomplete platelet correction); "
            "HLA-matched unrelated: >85% survival in experienced centres",
            "Gene therapy OTL-103 (autologous WASP lentiviral vector): Phase I/II trials; "
            "excellent platelet and immune reconstitution; avoids GvHD; "
            "compassionate access in some centres while awaiting approval",
            "Platelet transfusion: for ICH or major bleeding (irradiated products only); "
            "avoid ITP-like management (IVIg/steroids — partial response only in WAS-thrombocytopenia); "
            "splenectomy increases platelet count but worsens immunodeficiency + increases lymphoma risk",
            "IgRT: IVIg/SCIG if hypogammaglobulinaemia; target trough ≥8 g/L",
            "PJP prophylaxis: co-trimoxazole for T-cell lymphopenic patients",
            "Eczema: topical steroids + tacrolimus; antihistamines; avoid skin infections; "
            "AVOID: high-potency steroids on face; immunosuppressants worsening immunodeficiency",
            "EBV MONITORING: regular EBV-PCR surveillance; EBV-driven lymphoma risk requires "
            "rituximab or HSCT if EBV lymphoproliferative disease develops",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "ICH RISK — Low Platelet + Small Platelet Size: EMERGENCY Protocol",
                "body": (
                    "Wiskott-Aldrich syndrome has platelet counts 20,000-50,000/µL with SMALL "
                    "platelet size (dysfunctional microthrombocytes). Risk of intracranial haemorrhage "
                    "(ICH) is substantially higher than immune thrombocytopenia (ITP) at same count. "
                    "ANY head trauma or neurological symptom → URGENT CT head + neurosurgical review. "
                    "Platelet transfusion (irradiated) indicated for ICH. "
                    "Do NOT manage as ITP — WAS thrombocytopenia does NOT respond to IVIg/steroids adequately."
                ),
            },
            {
                "type": "warning",
                "title": "SPLENECTOMY — Increases Lymphoma Risk: Avoid in Classic WAS",
                "body": (
                    "Splenectomy increases platelet count in WAS but removes the last "
                    "functional lymphoid tissue → worsens combined immunodeficiency → "
                    "dramatically increases EBV-driven B-cell lymphoma risk. "
                    "In classic WAS: proceed to HSCT instead of splenectomy. "
                    "Splenectomy acceptable ONLY in mild XLT variant if HSCT deferred — "
                    "requires lifelong penicillin prophylaxis + pneumococcal/meningococcal vaccination."
                ),
            },
        ],
        "clinical_rules": [
            "SMALL PLATELET (low MPV) + THROMBOCYTOPENIA + ECZEMA IN MALE = WAS: measure WASP protein by flow cytometry immediately",
            "ITP IN MALE INFANT: always check platelet size (MPV) — low MPV = WAS not ITP; IVIg will fail",
            "EBV LYMPHOMA SCREENING: annual EBV-PCR in all non-transplanted WAS patients; lymphoma = poor prognosis without HSCT",
            "HSCT TIMING: early (age <5y) before lymphoma risk escalates; do NOT delay for 'mild' XLT",
            "AUTOIMMUNITY (40-70%): haemolytic anaemia, nephritis, IBD can present BEFORE infections — screen WAS in all young male with autoimmune disease + thrombocytopenia",
        ],
        "key_distinguishing": "Thrombocytopenia with SMALL PLATELETS (low MPV) + eczema + PID in male = WAS (low MPV is PATHOGNOMONIC — distinguishes from ITP where MPV is HIGH)",
        "severity_weights": {"Moderate": 0.35, "Severe": 0.45, "Mild": 0.20},
        "prevalence_per_100k": 0.4,
        "hsct_rate_pct": 70,
        "gene_therapy_rate_pct": 10,
        "infection_rate_pct": 75,
        "failure_to_thrive_pct": 55,
        "bcg_complication_pct": 8,
        "de_novo_pct": 30,
    },

    # ── TNFRSF13B — Common Variable Immunodeficiency (CVID) ──────────────────
    {
        "gene": "TNFRSF13B",
        "protein": "TNF Receptor Superfamily Member 13B (TACI)",
        "alias": "TNFRSF13B; OMIM gene 604907; 17p11.2; ~293 aa; CVID (OMIM #240500) and IgA deficiency; AD/AR; TACI (Transmembrane Activator and CAML Interactor); most common symptomatic primary antibody deficiency in adults; IgRT lifelong",
        "aa": "~293 aa",
        "kDa": "~31 kDa",
        "mechanism": (
            "TNFRSF13B encodes TACI (Transmembrane Activator and Calcium-Modulating Cyclophilin-Ligand "
            "Interactor), a member of the TNF receptor superfamily expressed on mature B cells, "
            "plasmablasts, and plasma cells. "
            "TACI LIGANDS: APRIL (a proliferation-inducing ligand; TNFSF13) and BAFF "
            "(B-cell activating factor; TNFSF13B) — both are survival and differentiation factors "
            "for B cells, particularly for B-cell class switching to IgA and IgG and plasma cell "
            "long-term survival in bone marrow niches. "
            "NORMAL FUNCTION: TACI/APRIL signalling → activation of NF-κB → "
            "class-switch recombination (CSR) to IgA and IgG → plasma cell differentiation → "
            "antibody secretion; TACI also promotes peripheral B-cell deletion (self-tolerance). "
            "PATHOMECHANISM: "
            "HETEROZYGOUS variants (AD, incomplete penetrance ~10%): haploinsufficiency of TACI → "
            "impaired CSR + plasma cell differentiation → reduced IgA, IgG → "
            "Common Variable Immunodeficiency (CVID) or selective IgA deficiency. "
            "HOMOZYGOUS/COMPOUND HETEROZYGOUS (AR): more severe phenotype; "
            "paradoxically, TACI partial GOF variants also cause CVID (gain of autoreactive function "
            "with simultaneous loss of B-cell activation — complex mechanism). "
            "CVID is clinically defined (not genetically) — TNFRSF13B accounts for only ~10% of CVID cases; "
            "majority of CVID remains genetically unexplained."
        ),
        "disease_type": (
            "Common Variable Immunodeficiency (CVID; OMIM #240500) — most common symptomatic "
            "primary antibody deficiency in adults (1:25,000); "
            "also selective IgA deficiency (most common PID overall; 1:500); "
            "TNFRSF13B accounts for ~10% of CVID cases; "
            "AD (incomplete penetrance ~10-20%) or AR; "
            "presentation: sinopulmonary infections + hypogammaglobulinaemia (IgG + IgA ± IgM low) "
            "after age 2 years (usually 2nd-4th decade); "
            "IgRT lifelong; complications: bronchiectasis, autoimmunity, granulomatous disease, lymphoma"
        ),
        "locus": "17p11.2",
        "omim_gene": 604907,
        "omim_disease": 240500,
        "inheritance": (
            "AUTOSOMAL DOMINANT (incomplete penetrance): C104R and A181E variants most common "
            "in European populations; ~10% of TNFRSF13B variant carriers develop CVID (remainder "
            "are subclinical carriers — do NOT immunise all carriers without checking immunoglobulins). "
            "AUTOSOMAL RECESSIVE: biallelic rare — more severe phenotype; "
            "PENETRANCE: extremely variable — same C104R variant causes CVID in some family "
            "members and selective IgA deficiency in others and is asymptomatic in majority. "
            "MOST CVID IS MULTIFACTORIAL — TNFRSF13B variants are risk factors not monogenic causes "
            "in most patients. "
            "GENETIC DIAGNOSIS in CVID: indicated for family counselling; "
            "other CVID genes (NFKB1, NFKB2, ICOS, CD19, CD81, LRBA, CTLA4) "
            "cause more mendelian forms — whole exome/genome increasingly first-line."
        ),
        "phenotype": (
            "CLASSICAL CVID (adults): "
            "RECURRENT SINOPULMONARY INFECTIONS: S. pneumoniae, H. influenzae — "
            "recurrent pneumonia, sinusitis, otitis → bronchiectasis (leading cause of morbidity); "
            "Giardia lamblia GI infection (chronic diarrhoea, malabsorption — unique PID). "
            "HYPOGAMMAGLOBULINAEMIA: IgG <5 g/L; IgA <0.07 g/L (often undetectable); "
            "IgM low or normal; poor vaccine responses (key functional diagnostic criterion). "
            "NON-INFECTIOUS COMPLICATIONS: "
            "AUTOIMMUNITY (20-30%): ITP; AIHA; RA; pernicious anaemia; "
            "GRANULOMATOUS DISEASE (10-20%): sarcoid-like non-caseating granulomas in liver, "
            "spleen, lungs — 'GLILD' (granulomatous-lymphocytic interstitial lung disease); "
            "LYMPHOMA RISK: 5-10 × general population; predominantly B-cell lymphomas; "
            "GI DISEASE: sprue-like enteropathy; nodular lymphoid hyperplasia. "
            "DIAGNOSTIC CRITERIA (ESID 2016): "
            "Age ≥4 years; IgG <5 g/L + IgA or IgM low; poor vaccine responses; "
            "exclusion of other causes."
        ),
        "treatment_options": [
            "IgRT (IgG replacement therapy) — LIFELONG MAINSTAY: "
            "IVIG every 3-4 weeks OR SCIG weekly/bi-weekly; "
            "target trough IgG ≥8 g/L (some guidelines ≥10 g/L for bronchiectasis); "
            "reduces sino-pulmonary infection frequency dramatically; "
            "does NOT resolve autoimmunity, granulomatous disease, or lymphoma risk; "
            "typical dose: 400-600 mg/kg/month IV or equivalent SC",
            "Antibiotic prophylaxis: co-amoxiclav or azithromycin for breakthrough infections "
            "despite IgRT (especially with bronchiectasis)",
            "Respiratory physiotherapy + regular spirometry: for bronchiectasis management; "
            "annual PFT + CT chest every 3-5 years",
            "Giardia: metronidazole × 7 days; tinidazole single dose; retest to confirm eradication",
            "GLILD (granulomatous-lymphocytic interstitial lung disease): "
            "prednisolone ± azathioprine ± rituximab; coordinate with pulmonology; "
            "avoid prolonged high-dose steroids (worsens immunodeficiency)",
            "Autoimmunity (ITP/AIHA): IVIg (double-dose), steroids, rituximab; "
            "splenectomy as last resort (worsens immune function); "
            "immunosuppressants must be balanced against infection risk",
            "Lymphoma surveillance: annual LDH + full blood count; CT if symptoms/lymphadenopathy",
        ],
        "drug_alerts": [
            {
                "type": "warning",
                "title": "LIVE VACCINES IN CVID — Absolute CI Despite 'Late Onset'",
                "body": (
                    "CVID patients present in adulthood and may have received live vaccines "
                    "safely in childhood (when pre-symptomatic). "
                    "Once CVID is diagnosed: live vaccines (yellow fever, MMR, varicella, "
                    "BCG, LAIV) are absolutely contraindicated. "
                    "Also advise: AVOID household contacts receiving live oral polio vaccine (OPV)."
                ),
            },
            {
                "type": "info",
                "title": "GIARDIA is CVID-SPECIFIC: Test Stool PCR if Chronic Diarrhoea",
                "body": (
                    "Giardia lamblia causes chronic malabsorptive diarrhoea in CVID — "
                    "IgA-mediated gut immunity is absent. Standard stool culture may miss it. "
                    "Request stool Giardia PCR (not antigen test alone). "
                    "Treat metronidazole × 7 days (10-14 days in CVID for higher clearance rate). "
                    "Test-of-cure stool PCR 4 weeks post-treatment."
                ),
            },
        ],
        "clinical_rules": [
            "ADULTS with recurrent pneumococcal pneumonia: check serum Ig levels (IgG, IgA, IgM) and vaccine responses — CVID presents in 2nd-4th decade",
            "POOR VACCINE RESPONSE: if titres not reached after standard vaccination → suspect antibody deficiency (CVID/XLA)",
            "GRANULOMA IN CVID: liver/lung granuloma in CVID = GLILD not sarcoidosis — treat with IgRT optimisation before steroids",
            "LYMPHOMA SURVEILLANCE mandatory: 5-10× elevated lymphoma risk — annual LDH + CBC",
            "TNFRSF13B VARIANT CARRIERS: do NOT treat without measuring immunoglobulins — majority are asymptomatic; treat the phenotype not the genotype",
        ],
        "key_distinguishing": "Adults (2nd-4th decade) + ALL Ig isotypes low + poor vaccine responses + recurrent pneumococcal infections = CVID (late onset distinguishes from XLA which presents in infancy)",
        "severity_weights": {"Mild": 0.40, "Moderate": 0.42, "Severe": 0.18},
        "prevalence_per_100k": 4.0,
        "hsct_rate_pct": 1,
        "gene_therapy_rate_pct": 0,
        "infection_rate_pct": 80,
        "failure_to_thrive_pct": 15,
        "bcg_complication_pct": 1,
        "de_novo_pct": 15,
    },

    # ── STAT3 — Hyper-IgE Syndrome (AD-HIES / Job Syndrome) ─────────────────
    {
        "gene": "STAT3",
        "protein": "Signal Transducer and Activator of Transcription 3 (STAT3)",
        "alias": "STAT3; OMIM gene 102582; 17q21.2; ~770 aa; AD-HIES/Job Syndrome (OMIM #147060) from AD LOF; AR-HIES from AR GOF; DOCK8-HIES (OMIM #243700) AR; classic AD-HIES: eczema + staph abscesses + markedly elevated IgE + skeletal/dental anomalies",
        "aa": "~770 aa",
        "kDa": "~92 kDa",
        "mechanism": (
            "STAT3 encodes Signal Transducer and Activator of Transcription 3, a key intracellular "
            "signalling molecule of the JAK-STAT pathway. "
            "STAT3 is activated downstream of many cytokine receptors: "
            "IL-6, IL-10, IL-11, IL-17, IL-21, IL-22, IL-23, IFN-α/β, EGF, LIF, OSM, G-CSF. "
            "NORMAL FUNCTION: cytokine receptor engagement → JAK kinases phosphorylate STAT3 "
            "(Tyr705) → STAT3 dimerises → translocates to nucleus → transcribes target genes "
            "for: Th17 differentiation (IL-17-producing T cells — critical for fungal/bacterial "
            "immunity at epithelial barriers); "
            "acute phase response (IL-6 → STAT3 → CRP, fibrinogen); "
            "anti-inflammatory IL-10 signalling in macrophages; "
            "keratinocyte wound healing; "
            "B-cell differentiation and class switching. "
            "PATHOMECHANISM (AD-HIES / Job Syndrome): "
            "DOMINANT NEGATIVE LOF missense variants in STAT3 (SH2 domain and DNA-binding domain "
            "predominantly) → dominant negative effect — mutant STAT3 dimerises with wild-type "
            "STAT3 and blocks its transcriptional activity → "
            "Th17 deficiency: impaired mucosal antifungal and antibacterial immunity "
            "(Candida, S. aureus); "
            "Absent IL-17 → absent neutrophil recruitment at epithelial surfaces → "
            "'cold abscesses' (no pus formation, no pain, no redness — S. aureus abscesses "
            "without classical inflammatory signs) — PATHOGNOMONIC; "
            "IL-6 signalling failure → incomplete fever response despite severe infection; "
            "SKELETAL/DENTAL: abnormal bone remodelling → delayed primary tooth shedding "
            "(PATHOGNOMONIC — double dentition in adolescents); scoliosis; "
            "MARKEDLY ELEVATED IgE: mechanism incompletely understood; IgE >2000 IU/mL "
            "(often >10,000 IU/mL); highest IgE of any PID."
        ),
        "disease_type": (
            "Autosomal Dominant Hyper-IgE Syndrome (AD-HIES; Job Syndrome; OMIM #147060); "
            "AD; dominant negative STAT3 LOF missense; "
            "triad: recurrent S. aureus 'cold' abscesses + severe eczema + markedly elevated IgE (>2000 IU/mL); "
            "skeletal: double dentition + scoliosis + joint hyperextensibility + recurrent fractures; "
            "NIHPC score ≥40 = diagnosis; TMP-SMX + antifungal prophylaxis; no curative therapy "
            "(HSCT does not correct non-haematopoietic STAT3 defects)"
        ),
        "locus": "17q21.2",
        "omim_gene": 102582,
        "omim_disease": 147060,
        "inheritance": (
            "AUTOSOMAL DOMINANT LOF (dominant negative missense): "
            "~60-70% of AD-HIES are de novo (no family history); "
            "familial cases: 50% offspring affected. "
            "GENOTYPE-PHENOTYPE: SH2 domain variants (most common) = full classic AD-HIES; "
            "DNA-binding domain = similar; "
            "Coiled-coil domain variants = milder. "
            "DIFFERENTIAL: "
            "AR-HIES: biallelic STAT3 GOF (gain-of-function) → opposite mechanism "
            "(constitutive STAT3 activation → autoimmunity-dominant phenotype, NOT classic HIES); "
            "DOCK8 deficiency (OMIM #243700): AR-HIES; eczema + infections + elevated IgE "
            "but + viral susceptibility (molluscum, HPV, HSV) distinguishes from AD-HIES; "
            "elevated IgE alone does NOT = HIES — atopy, parasites, drug reactions all cause elevated IgE. "
            "NIHPC SCORE: National Institutes of Health (NIH) primary care scoring system — "
            "≥40 = AD-HIES; scores eczema, IgE, infections, skeletal, dental, facial features."
        ),
        "phenotype": (
            "CLASSIC AD-HIES TRIAD: "
            "ECZEMA: severe atopic-type eczema from birth; resistant to standard treatments; "
            "typically worse in infancy + improves partially with age. "
            "COLD ABSCESSES: S. aureus skin and deep tissue abscesses WITHOUT classical "
            "inflammatory signs (absent pain, absent erythema, absent warmth, absent fever) — "
            "PATHOGNOMONIC: 'cold abscess'; patient often unaware of abscess until rupture; "
            "candidal infections (oral, oesophageal, vaginal — refractory); "
            "ELEVATED IgE: IgE >2000 IU/mL (commonly >10,000; sometimes >100,000); "
            "eosinophilia (peripheral + tissue). "
            "PULMONARY: recurrent pneumonia (S. aureus, H. influenzae) → "
            "PNEUMATOCELE formation (air-filled cysts in lung — PATHOGNOMONIC for AD-HIES; "
            "secondary Aspergillus colonisation; requires surgical resection). "
            "SKELETAL/DENTAL: "
            "delayed primary tooth shedding (retained deciduous teeth — adolescents with 'double teeth'); "
            "scoliosis; joint hyperextensibility; pathological fractures (minor trauma); "
            "osteopenia. "
            "FACIAL: broad nasal bridge; prominent forehead; coarse facial features (develop in adolescence). "
            "VASCULAR: coronary artery aneurysms; Chiari malformation; cerebral aneurysms."
        ),
        "treatment_options": [
            "TMP-SMX prophylaxis LIFELONG: prevents S. aureus cold abscesses; "
            "alternative: clindamycin; culture + sensitivity for breakthrough infections",
            "Antifungal prophylaxis: fluconazole (covers Candida; does NOT cover Aspergillus); "
            "itraconazole for higher Aspergillus risk (post-pneumatocele); "
            "monitor LFTs during prolonged antifungal use",
            "Pneumatocele management: "
            "surgical resection of large/infected pneumatoceles (risk: Aspergillus colonisation); "
            "lobectomy sometimes required for recurrent infection in same lobe; "
            "bronchoscopy for BAL culture before surgery",
            "DENTAL SURGERY: extraction of retained deciduous teeth that do not shed spontaneously "
            "(avoids double-tooth malocclusion + caries risk); orthodontic monitoring",
            "Eczema: moderate-potency topical steroids; tacrolimus/pimecrolimus; "
            "dupilumab (anti-IL-4Rα) has been used off-label with benefit in some AD-HIES patients "
            "(IL-4/IL-13 pathway dysregulation component); "
            "avoid high-dose systemic steroids (worsens infection risk)",
            "NO CURATIVE THERAPY: HSCT does NOT fully correct AD-HIES — "
            "non-haematopoietic tissues (bone, lung, teeth, brain) retain STAT3 LOF; "
            "HSCT partially improves haematopoietic-dependent immune features but "
            "not skeletal/dental/pulmonary architecture; generally NOT recommended",
        ],
        "drug_alerts": [
            {
                "type": "danger",
                "title": "COLD ABSCESS — NO FEVER, NO PAIN: Do NOT Miss",
                "body": (
                    "AD-HIES cold abscesses present WITHOUT classical inflammatory signs "
                    "because Th17/IL-17 deficiency prevents neutrophil recruitment. "
                    "Patient may be afebrile with a large fluctuant mass they did not notice. "
                    "MANDATE: examine ALL AD-HIES patients for asymptomatic abscesses at "
                    "every visit. Do NOT wait for fever — no fever does NOT mean no infection. "
                    "Incise, drain, and send culture for sensitivity."
                ),
            },
            {
                "type": "warning",
                "title": "PNEUMATOCELES: Surgical Resection to Prevent Aspergillus — Act Early",
                "body": (
                    "Pneumatoceles (post-pneumonia air cysts) in AD-HIES are at high risk of "
                    "secondary Aspergillus fumigatus colonisation → impossible to eradicate "
                    "medically → progressive pulmonary destruction. "
                    "Resect surgically before they enlarge or become colonised. "
                    "If Aspergillus already colonising: voriconazole + surgical referral. "
                    "Annual CT chest mandatory in all AD-HIES patients."
                ),
            },
        ],
        "clinical_rules": [
            "IgE >2000 + COLD ABSCESSES (no fever/pain) + ECZEMA = AD-HIES/STAT3 until proven otherwise",
            "DOUBLE DENTITION in adolescent + eczema + elevated IgE = STAT3 AD-HIES PATHOGNOMONIC",
            "NIHPC SCORE ≥40: calculate at first presentation — guides genetic testing priority",
            "HSCT IS NOT CURATIVE for AD-HIES: do NOT offer HSCT expecting full resolution of skeletal/pulmonary features",
            "DUPILUMAB: off-label for AD-HIES eczema — consider if standard treatment fails; monitor carefully for worsening infections",
        ],
        "key_distinguishing": "IgE >2000 IU/mL + COLD ABSCESSES (no inflammation) + DOUBLE DENTITION + pneumatoceles = AD-HIES/STAT3 (absent fever/pain distinguishes from ordinary abscess)",
        "severity_weights": {"Moderate": 0.45, "Severe": 0.35, "Mild": 0.20},
        "prevalence_per_100k": 0.3,
        "hsct_rate_pct": 5,
        "gene_therapy_rate_pct": 0,
        "infection_rate_pct": 90,
        "failure_to_thrive_pct": 40,
        "bcg_complication_pct": 5,
        "de_novo_pct": 65,
    },
]


def _gen_patients_for_gene(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    patients = []
    weights = gene_data["severity_weights"]
    sev_choices = list(weights.keys())
    sev_probs = list(weights.values())

    for i in range(40):
        sev = rng.choices(sev_choices, sev_probs)[0]
        onset = rng.randint(0, 12) if sev == "Severe" else rng.randint(6, 36)
        dx_delay = rng.randint(1, 36)
        had_hsct = rng.random() < gene_data["hsct_rate_pct"] / 100
        gene_therapy = rng.random() < gene_data["gene_therapy_rate_pct"] / 100
        had_infection = rng.random() < gene_data["infection_rate_pct"] / 100
        ftf = rng.random() < gene_data["failure_to_thrive_pct"] / 100
        bcg_comp = rng.random() < gene_data["bcg_complication_pct"] / 100
        de_novo = rng.random() < gene_data["de_novo_pct"] / 100
        drug_error = rng.random() < 0.10  # 10% management error rate
        sex = rng.choice(["M", "F"])
        # X-linked genes: only males affected (or occasionally symptomatic female carriers)
        if gene_data.get("inheritance", "").startswith("X-LINKED") and sex == "F":
            sev = "Mild"  # carriers rarely severely affected
            had_hsct = False

        patients.append({
            "gene": gene_data["gene"],
            "severity": sev,
            "onset_age_months": onset,
            "diagnosis_age_months": onset + dx_delay,
            "had_hsct": had_hsct,
            "gene_therapy": gene_therapy,
            "opportunistic_infection": had_infection,
            "failure_to_thrive": ftf,
            "bcg_complication": bcg_comp,
            "de_novo": de_novo,
            "drug_error": drug_error,
            "sex": sex,
        })
    return patients


def _gen_cohort() -> list:
    all_pts = []
    for idx, gd in enumerate(PID_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients_for_gene(gd, seed))
    return all_pts


# ── API functions ──────────────────────────────────────────────────────────────

def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] += 1

    hsct_n = sum(1 for p in patients if p["had_hsct"])
    gt_n = sum(1 for p in patients if p["gene_therapy"])
    inf_n = sum(1 for p in patients if p["opportunistic_infection"])
    ftf_n = sum(1 for p in patients if p["failure_to_thrive"])
    bcg_n = sum(1 for p in patients if p["bcg_complication"])
    de_novo_n = sum(1 for p in patients if p["de_novo"])
    error_n = sum(1 for p in patients if p["drug_error"])

    onsets = [p["onset_age_months"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 1)
    mean_dx = round(sum(p["diagnosis_age_months"] for p in patients) / n, 1)

    gene_hsct_pct = {}
    for gd in PID_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_hsct_pct[gd["gene"]] = round(
            100 * sum(1 for p in gpts if p["had_hsct"]) / len(gpts), 1
        )

    # X-linked vs autosomal breakdown
    xlinked_genes = ["IL2RG", "BTK", "CYBB", "WAS"]
    ar_genes = ["ADA", "RAG1"]
    ad_genes = ["TNFRSF13B", "STAT3"]

    disease_cat = {
        "X-linked Combined Immunodeficiency": round(100 * 40 / n, 1),   # IL2RG
        "X-linked Agammaglobulinaemia (XLA)": round(100 * 40 / n, 1),   # BTK
        "X-linked Chronic Granulomatous Disease": round(100 * 40 / n, 1), # CYBB
        "Wiskott-Aldrich Syndrome": round(100 * 40 / n, 1),             # WAS
        "ADA-SCID (Metabolic SCID)": round(100 * 40 / n, 1),            # ADA
        "T-B-NK+ SCID / Omenn Syndrome": round(100 * 40 / n, 1),        # RAG1
        "Common Variable Immunodeficiency": round(100 * 40 / n, 1),      # TNFRSF13B
        "Hyper-IgE Syndrome (AD-HIES)": round(100 * 40 / n, 1),         # STAT3
    }

    clinical_features = {
        "Opportunistic infections": round(100 * inf_n / n, 1),
        "HSCT performed": round(100 * hsct_n / n, 1),
        "Gene therapy received": round(100 * gt_n / n, 1),
        "Failure to thrive": round(100 * ftf_n / n, 1),
        "BCG complication": round(100 * bcg_n / n, 1),
        "De novo variant": round(100 * de_novo_n / n, 1),
    }

    severity_prev = {
        "Severe": round(100 * sev["Severe"] / n, 1),
        "Moderate": round(100 * sev["Moderate"] / n, 1),
        "Mild": round(100 * sev["Mild"] / n, 1),
    }

    kpis = [
        {"label": "Total Patients", "value": str(n)},
        {"label": "Genes Covered", "value": "8"},
        {"label": "HSCT Rate", "value": f"{round(100*hsct_n/n,1)}%"},
        {"label": "Opportunistic Infections", "value": f"{round(100*inf_n/n,1)}%"},
        {"label": "Mean Onset (months)", "value": str(mean_onset)},
        {"label": "Mean Dx Delay (months)", "value": str(round(mean_dx - mean_onset, 1))},
        {"label": "Gene Therapy Rate", "value": f"{round(100*gt_n/n,1)}%"},
        {"label": "De Novo Rate", "value": f"{round(100*de_novo_n/n,1)}%"},
    ]

    return {
        "atlas": "PID-Atlas",
        "full_name": "Complete 8-Gene Primary Immunodeficiency (PID) Atlas",
        "subtitle": (
            "IL2RG·ADA·RAG1·BTK·CYBB·WAS·TNFRSF13B·STAT3 — "
            "320 patients (8×40, seeds 1118–1125)"
        ),
        "description": (
            "Comprehensive atlas of 8 major genetic Primary Immunodeficiency diseases: "
            "IL2RG/X-SCID (XLR; T-B+NK-; γc chain; gene therapy OTL-101 FDA-2024; BCG ABSOLUTELY CI); "
            "ADA/ADA-SCID (AR; T-B-NK-; metabolic SCID; skeletal dysplasia; "
            "PEG-ADA bridge + Strimvelis gene therapy EMA-2016); "
            "RAG1/T-B-NK+ SCID+Omenn (AR; V(D)J recombination; "
            "Omenn=erythroderma+eosinophilia from hypomorphic variants; myeloablative HSCT); "
            "BTK/XLA (XLR; absent B cells; normal T cells; lifelong IgRT; OPV ABSOLUTELY CI; "
            "Mycoplasma arthritis PATHOGNOMONIC); "
            "CYBB/X-CGD (XLR; absent respiratory burst; DHR=diagnostic; "
            "S. aureus liver abscess PATHOGNOMONIC; itraconazole+TMP-SMX lifelong); "
            "WAS/Wiskott-Aldrich (XLR; small platelets MPV-low PATHOGNOMONIC triad+eczema+PID; "
            "EBV lymphoma 13-22%; HSCT curative early); "
            "TNFRSF13B/CVID (AD partial penetrance; adults 2nd-4th decade; "
            "all Ig low + poor vaccines; Giardia; GLILD; bronchiectasis; lifelong IgRT); "
            "STAT3/AD-HIES/Job (AD dominant-negative LOF; IgE>2000 + cold abscesses + "
            "double dentition PATHOGNOMONIC; pneumatoceles; no curative HSCT)"
        ),
        "total_patients": n,
        "genes_covered": 8,
        "patients_per_gene": 40,
        "seed_range": "1118–1125",
        "gene_list": [g["gene"] for g in PID_GENES],
        "severity": sev,
        "severity_prevalence": severity_prev,
        "clinical_features_prevalence": clinical_features,
        "disease_category_breakdown": disease_cat,
        "gene_hsct_pct": gene_hsct_pct,
        "mean_onset_age_months": mean_onset,
        "mean_dx_age_months": mean_dx,
        "kpis": kpis,
        "drug_alerts": [
            {
                "type": "danger",
                "title": "LIVE VACCINES ABSOLUTELY CI IN ALL SCID/PID WITH T-CELL DEFICIENCY",
                "body": (
                    "BCG, MMR, rotavirus, varicella, yellow fever, LAIV, oral polio vaccine: "
                    "ALL absolutely contraindicated in T-cell deficient PID (X-SCID, ADA-SCID, RAG1 SCID). "
                    "BCG at birth (standard in many countries) causes fatal BCG-osis. "
                    "Oral polio vaccine causes VAPP in XLA. "
                    "NBS TREC screening before BCG/rotavirus administration is life-saving."
                ),
            },
            {
                "type": "danger",
                "title": "TREC=0 ON NBS + LYMPHOPENIA → SCID: MEDICAL EMERGENCY",
                "body": (
                    "Newborn Screening TREC=0 (absent T-cell receptor excision circles) "
                    "indicates T-cell lymphopenia. Refer IMMEDIATELY to paediatric immunology. "
                    "Do NOT wait for clinical symptoms. Pre-symptomatic HSCT survival >95% "
                    "vs <75% after symptomatic presentation. NBS is LIFE-SAVING."
                ),
            },
            {
                "type": "warning",
                "title": "IRRADIATED BLOOD PRODUCTS MANDATORY in T-cell Immunodeficiency",
                "body": (
                    "ALL patients with T-cell deficiency (X-SCID, ADA-SCID, RAG1 SCID, WAS) "
                    "must receive CMV-negative, irradiated blood products "
                    "(irradiation prevents transfusion-associated GvHD from donor lymphocytes; "
                    "CMV-negative prevents primary CMV infection). "
                    "Standard non-irradiated transfusions cause fatal GvHD in SCID."
                ),
            },
        ],
        "critical_rules": [
            "TREC=0 → SCID EMERGENCY: refer immunology same day; do NOT observe",
            "BCG BEFORE DIAGNOSIS: treat as BCG-osis (triple anti-mycobacterials) before HSCT",
            "T-CELL PHENOTYPE: T-B+NK- = IL2RG/X-SCID; T-B-NK+ = RAG1/2; T-B-NK- = ADA/RAG1 severe",
            "ABSENT TONSILS + HYPOGAMMAGLOBULINAEMIA in male: XLA until BTK flow cytometry done",
            "SMALL PLATELETS (low MPV) in male infant with eczema: WAS not ITP — WASP flow cytometry",
            "COLD ABSCESS (no fever, no pain) + IgE >2000: AD-HIES/STAT3",
            "S. AUREUS LIVER ABSCESS in child: CGD until DHR flow cytometry negative",
            "GIARDIA in adult with recurrent pneumonia + low IgG: CVID — check vaccine responses",
        ],
    }


def get_breakdown() -> list:
    patients = _gen_cohort()
    result = []
    for gd in PID_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        n = len(gpts)
        hsct_pct = round(100 * sum(1 for p in gpts if p["had_hsct"]) / n, 1)
        gt_pct = round(100 * sum(1 for p in gpts if p["gene_therapy"]) / n, 1)
        inf_pct = round(100 * sum(1 for p in gpts if p["opportunistic_infection"]) / n, 1)
        ftf_pct = round(100 * sum(1 for p in gpts if p["failure_to_thrive"]) / n, 1)
        bcg_pct = round(100 * sum(1 for p in gpts if p["bcg_complication"]) / n, 1)
        de_novo_pct = round(100 * sum(1 for p in gpts if p["de_novo"]) / n, 1)
        onsets = [p["onset_age_months"] for p in gpts]
        mean_onset = round(sum(onsets) / n, 1)

        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "disease_type": gd["disease_type"],
            "inheritance": gd["alias"].split(";")[3].strip() if len(gd["alias"].split(";")) > 3 else "AD/XLR",
            "n_patients": n,
            "mean_onset_age_months": mean_onset,
            "severity_weights": gd["severity_weights"],
            "hsct_pct": hsct_pct,
            "gene_therapy_pct": gt_pct,
            "infection_pct": inf_pct,
            "failure_to_thrive_pct": ftf_pct,
            "bcg_complication_pct": bcg_pct,
            "de_novo_pct": de_novo_pct,
            "prevalence_per_100k": gd["prevalence_per_100k"],
            "key_distinguishing": gd["key_distinguishing"],
            "treatment_options": gd["treatment_options"],
            "drug_alerts": gd["drug_alerts"],
            "clinical_rules": gd["clinical_rules"],
        })
    return result


def get_definitions() -> list:
    return [
        {
            "term": "SCID (Severe Combined Immunodeficiency)",
            "definition": (
                "Profound deficiency of both T-cell and B-cell (and often NK-cell) immunity "
                "resulting in susceptibility to ALL classes of pathogens (bacteria, viruses, fungi, "
                "opportunists). Presents in infancy (3-6 months as maternal antibodies wane). "
                "Fatal without treatment (HSCT or gene therapy) by age 1-2 years. "
                "NBS by TREC assay on Day 2 Guthrie card is life-saving — detects T-lymphopenia "
                "before clinical symptoms. Classified by lymphocyte phenotype: "
                "T-B+NK- (X-SCID/IL2RG), T-B-NK+ (RAG1/2), T-B-NK- (ADA, RAG1 severe)."
            ),
        },
        {
            "term": "TREC (T-Cell Receptor Excision Circles)",
            "definition": (
                "Circular DNA byproducts generated during T-cell receptor V(D)J rearrangement "
                "in the thymus. Measured on Guthrie card dried blood spots as part of Newborn "
                "Screening (NBS). TREC=0 indicates absent T-cell development → SCID until proven "
                "otherwise. TREC screening mandated in many high-income countries. "
                "LIMITATION: does not detect B-cell deficiencies (XLA) or isolated NK deficiencies."
            ),
        },
        {
            "term": "V(D)J Recombination",
            "definition": (
                "Somatic DNA rearrangement of Variable, Diversity, and Joining gene segments "
                "catalysed by the RAG1/RAG2 recombinase complex — generates diverse immunoglobulin "
                "(antibody) and T-cell receptor (TCR) repertoire. "
                "Essential for B-cell and T-cell development (NK cells do not require V(D)J). "
                "RAG1/RAG2 deficiency → absent B and T cells (T-B-NK+ SCID or Omenn Syndrome). "
                "TREC are generated as V(D)J byproducts — basis of NBS TREC assay."
            ),
        },
        {
            "term": "Common Gamma Chain (γc; CD132; IL2RG)",
            "definition": (
                "Shared signalling subunit of the receptors for IL-2, IL-4, IL-7, IL-9, IL-15, "
                "and IL-21. Encoded by IL2RG at Xq13.1. "
                "IL-7/γc = non-redundant for T-cell thymic development; "
                "IL-15/γc = non-redundant for NK-cell development; "
                "B cells do not require γc → B cells PRESERVED in X-SCID (B+ phenotype). "
                "Loss of function = X-SCID (T-B+NK-). "
                "Downstream signalling via JAK1/JAK3 → STAT5."
            ),
        },
        {
            "term": "BCG-osis",
            "definition": (
                "Disseminated infection caused by BCG vaccine (live attenuated Mycobacterium bovis) "
                "given to an infant with T-cell SCID before diagnosis. "
                "Clinical features: hepatitis, lymphadenitis, bone marrow failure, pulmonary infection. "
                "FATAL if untreated. Management: triple anti-mycobacterials (INH + rifampicin + "
                "ethambutol) for minimum 6-12 months; HSCT cannot proceed until BCG cleared. "
                "PREVENTION: NBS TREC before BCG administration."
            ),
        },
        {
            "term": "Omenn Syndrome",
            "definition": (
                "RAG1/RAG2 hypomorphic variant disorder — oligoclonal autoreactive T cells "
                "escape from impaired V(D)J recombination and attack host tissues. "
                "CLINICAL: erythroderma (generalised skin redness) + eosinophilia + "
                "hepatosplenomegaly + lymphadenopathy + elevated IgE + "
                "profound immunodeficiency. "
                "KEY DDx: maternal T-cell GvHD engraftment (chimerism = maternal in GvHD; "
                "host-derived autoreactive T cells in Omenn). "
                "Treatment: immunosuppression (cyclosporin A) + HSCT."
            ),
        },
        {
            "term": "DHR Flow Cytometry (Dihydrorhodamine 123)",
            "definition": (
                "Gold-standard diagnostic test for Chronic Granulomatous Disease (CGD). "
                "Dihydrorhodamine 123 is oxidised by H₂O₂ (generated by NADPH oxidase) "
                "→ fluorescent rhodamine 123, detectable by flow cytometry. "
                "NORMAL: phagocytes show bright fluorescence after PMA stimulation. "
                "X-CGD: absent fluorescence (no NADPH oxidase activity). "
                "CARRIER FEMALES: bimodal pattern (some cells normal, some absent). "
                "ADVANTAGE over NBT test: quantitative, single cells, can identify carriers."
            ),
        },
        {
            "term": "PEG-ADA (Elapegademase-lvlr / Revcovi)",
            "definition": (
                "Polyethylene glycol–modified bovine adenosine deaminase — enzyme replacement "
                "therapy for ADA-SCID. FDA-approved 2018. Given IM 2× per week. "
                "Provides partial immune reconstitution (lymphocytes increase; infections decrease) "
                "but is NOT curative. Bridge to HSCT or gene therapy. "
                "Anti-PEG-ADA antibodies develop in ~30% → loss of efficacy. "
                "Do not accept PEG-ADA as definitive long-term treatment without curative plan."
            ),
        },
        {
            "term": "Strimvelis (OTL-101)",
            "definition": (
                "Gene therapy for ADA-SCID: autologous CD34+ haematopoietic stem cells "
                "transduced ex vivo with retroviral (Strimvelis) or lentiviral (OTL-101) vector "
                "carrying functional ADA cDNA. Strimvelis EMA-approved 2016 (MRC Milan). "
                "OTL-101 lentiviral: FDA approval track 2024. "
                "Efficacy: 80-90% immune reconstitution; curative; avoids GvHD. "
                "Requires mild myeloablative busulfan conditioning. "
                "Most successful gene therapy for primary immunodeficiency to date."
            ),
        },
        {
            "term": "IgRT (Immunoglobulin Replacement Therapy)",
            "definition": (
                "Lifelong treatment for antibody deficiency (XLA, CVID, post-HSCT B-cell failure). "
                "IVIG (intravenous): 400-600 mg/kg every 3-4 weeks. "
                "SCIG (subcutaneous): weekly or bi-weekly; preferred for home use. "
                "TARGET: IgG trough ≥8 g/L (minimum); ≥10 g/L if bronchiectasis or recurrent infections. "
                "DOES NOT replace cell-mediated immunity (T/NK cells) — not adequate for SCID alone."
            ),
        },
        {
            "term": "WASP (Wiskott-Aldrich Syndrome Protein)",
            "definition": (
                "Actin nucleation-promoting factor encoded by WAS at Xp11.23. "
                "Expressed exclusively in haematopoietic cells. "
                "Links upstream signals (Cdc42, PIP2) to Arp2/3-mediated branched actin polymerisation. "
                "Required for: T-cell immunological synapse; NK-cell granule polarisation; "
                "platelet morphology; B-cell signalling. "
                "LOF → WAS (thrombocytopenia + eczema + PID). "
                "GOF (constitutive) → X-linked neutropenia (XLN). "
                "Key diagnostic: WASP protein expression by intracellular flow cytometry."
            ),
        },
        {
            "term": "Cold Abscess (AD-HIES/STAT3)",
            "definition": (
                "Staphylococcus aureus abscess presenting WITHOUT classical inflammatory signs "
                "(absent fever, absent pain, absent warmth, absent erythema). "
                "PATHOGNOMONIC for AD-HIES (Hyper-IgE Syndrome / STAT3 LOF). "
                "Mechanism: Th17 deficiency (STAT3 LOF → impaired IL-17 signalling) → "
                "absent neutrophil recruitment at site → no pus, no inflammation. "
                "Patient often unaware of abscess. Examine ALL AD-HIES patients at every visit."
            ),
        },
        {
            "term": "Pneumatocele",
            "definition": (
                "Thin-walled air-filled cysts in lung parenchyma, typically arising post-pneumonia "
                "(S. aureus pneumonia most common). PATHOGNOMONIC for AD-HIES (STAT3 LOF) when recurrent. "
                "Complication: secondary Aspergillus fumigatus colonisation → "
                "impossible to eradicate medically → progressive pulmonary destruction. "
                "Management: surgical resection before colonisation occurs. "
                "Annual CT chest mandatory in AD-HIES."
            ),
        },
        {
            "term": "Catalase-Positive Organisms (CGD)",
            "definition": (
                "Microorganisms that produce catalase enzyme, which destroys H₂O₂. "
                "In Chronic Granulomatous Disease (CGD), phagocytes cannot generate their own "
                "reactive oxygen species — they depend on microbial H₂O₂ from catalase-NEGATIVE "
                "organisms for killing. Catalase-POSITIVE organisms destroy this H₂O₂ → survive "
                "intracellularly → cause recurrent, severe infections in CGD. "
                "KEY organisms: S. aureus, Aspergillus fumigatus, Nocardia, Serratia marcescens, "
                "Burkholderia cepacia, Chromobacterium violaceum. "
                "S. pneumoniae and H. influenzae (catalase-negative) are NOT typical CGD organisms."
            ),
        },
        {
            "term": "NBT Test (Nitroblue Tetrazolium)",
            "definition": (
                "Functional screening test for Chronic Granulomatous Disease (CGD). "
                "NBT dye (yellow) is reduced to formazan (blue) by NADPH oxidase-generated O₂•⁻. "
                "NORMAL: phagocytes turn blue (positive NBT test). "
                "CGD: phagocytes remain yellow (negative NBT test — absent oxidative burst). "
                "ADVANTAGE: simple, inexpensive, rapid. "
                "DISADVANTAGE: semi-quantitative; less sensitive for carriers than DHR. "
                "DHR flow cytometry is now preferred gold-standard."
            ),
        },
        {
            "term": "CVID (Common Variable Immunodeficiency)",
            "definition": (
                "Most common symptomatic primary antibody deficiency in adults (1:25,000). "
                "DIAGNOSIS (ESID 2016): age ≥4 years; IgG <5 g/L AND low IgA or IgM; "
                "poor vaccine responses (failure to mount fourfold rise to protein vaccines); "
                "exclusion of other causes. "
                "GENES: TNFRSF13B (~10%); NFKB1/2; ICOS; CD19; LRBA; CTLA4; majority unexplained. "
                "COMPLICATIONS: bronchiectasis; autoimmunity 20-30%; GLILD 10-20%; "
                "lymphoma 5-10× risk. "
                "TREATMENT: lifelong IgRT."
            ),
        },
        {
            "term": "Th17 Cells and IL-17 Immunity",
            "definition": (
                "CD4+ T-helper subset producing IL-17A, IL-17F, IL-22. "
                "Critical for mucosal immunity at epithelial barriers (skin, gut, lung). "
                "IL-17 → CXCL8 → neutrophil recruitment → clearance of S. aureus and Candida. "
                "STAT3 LOF (AD-HIES): impaired Th17 differentiation → absent IL-17 → "
                "absent neutrophil recruitment → cold abscesses, Candida mucocutaneous. "
                "IL-12/IL-23 receptor defects: Mendelian Susceptibility to Mycobacterial Disease "
                "(MSMD) — isolated Th1 deficiency for mycobacteria."
            ),
        },
    ]
