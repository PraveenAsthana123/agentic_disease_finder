#!/usr/bin/env python3
"""Hereditary-Primary-Immunodeficiency-Atlas — Complete 8-Gene Atlas (Hereditary Primary Immunodeficiencies)
BTK     (Bruton Tyrosine Kinase; 659 aa; Xq22.2; XL;
         X-linked Agammaglobulinaemia (XLA);
         Absent B cells — virtually ZERO circulating B lymphocytes (CD19+ <1%);
         Recurrent sinopulmonary infections from 6 months (maternal IgG wanes);
         AVOID all live vaccines — oral polio -> vaccine-associated paralytic poliomyelitis (VAPP);
         seed SEED_BASE+0) .
IL2RG   (Common Gamma Chain / Interleukin 2 Receptor Subunit Gamma; 369 aa; Xq13.1; XL;
         X-linked Severe Combined Immunodeficiency (SCID-X1);
         Most common form of SCID; T-B+NK- immunophenotype;
         Maternal T-cell engraftment (non-self T cells) must be excluded before transplant;
         Gene therapy (OTL-101 / GSK2696274) curative -- first gene therapy licensed in EU for SCID;
         seed SEED_BASE+1) .
ADA     (Adenosine Deaminase; 363 aa; 20q13.12; AR;
         ADA-SCID -- first disease treated by gene therapy (1990 trial; Strimvelis EMA approved 2016);
         Toxic accumulation of dATP kills T, B, and NK lymphocytes equally;
         PEG-ADA enzyme replacement bridges to gene therapy or HSCT;
         seed SEED_BASE+2) .
RAG1    (Recombination Activating Gene 1; 1043 aa; 11p13; AR;
         T-B-NK+ SCID; absent V(D)J recombination -> no T or B cells;
         Partial loss-of-function RAG1 -> Omenn syndrome (erythroderma, eosinophilia, elevated IgE);
         seed SEED_BASE+3) .
RAG2    (Recombination Activating Gene 2; 527 aa; 11p13; AR;
         T-B-NK+ SCID or Omenn syndrome when partial; forms heterodimer with RAG1;
         Omenn syndrome = erythroderma + eosinophilia + elevated IgE -- NOT atopic eczema;
         seed SEED_BASE+4) .
DCLRE1C (Artemis / DNA Cross-Link Repair 1C; 692 aa; 10p13; AR;
         Artemis-SCID (T-B-NK+ phenotype); radiation-sensitive SCID;
         Defective DNA double-strand break (DSB) repair -> V(D)J recombination failure;
         Reduced-intensity conditioning MANDATORY -- standard myeloablative chemo is lethal;
         seed SEED_BASE+5) .
JAK3    (Janus Kinase 3; 1124 aa; 19p13.11; AR;
         JAK3-SCID -- T-B+NK- phenotype clinically IDENTICAL to SCID-X1 (IL2RG) but autosomal recessive;
         JAK3 signals downstream of the common gamma chain (IL2RG); loss = same downstream block;
         Test BOTH JAK3 and IL2RG in any patient with T-B+NK- SCID -- females cannot have XLA/SCID-X1;
         seed SEED_BASE+6) .
TNFRSF13B (TACI / TNF Receptor Superfamily Member 13B; 293 aa; 17p11.2; AD/AR;
         Common Variable Immunodeficiency (CVID) -- most common symptomatic PID in adults;
         Low IgG (<4 g/L) + low IgA and/or IgM + absent vaccine responses + age >4 years;
         Autoimmune complications ~25%; non-Hodgkin lymphoma risk 8-fold elevated;
         10% CVID patients have monogenic cause (TNFRSF13B, NFKB1, NFKB2, CTLA4, etc.);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 1974-1981)
"""

import random

SEED_BASE = 1974

IMMUNODEFICIENCY_GENES = [
    # -- BTK -- Bruton Tyrosine Kinase / XLA ----------------------------------------
    {
        "gene": "BTK",
        "alt_name": "BTK (XLA / X-linked Agammaglobulinaemia / Bruton Tyrosine Kinase)",
        "protein": (
            "BTK -- Xq22.2 XL -- BTK-659aa -- "
            "XLA-Absent-B-Cells-CD19-Less-Than-1pct -- "
            "Recurrent-Sinopulmonary-Infections-Onset-6-Months -- "
            "LIVE-VACCINES-ABSOLUTELY-CONTRAINDICATED-OPV-Causes-VAPP -- "
            "IVIG-Replacement-Every-3-4-Weeks-Lifelong -- "
            "Bruton-Tyrosine-Kinase-Pre-B-Cell-Receptor-Signalling"
        ),
        "locus": "Xq22.2",
        "protein_size": "659 aa",
        "inheritance": "XL (X-linked recessive)",
        "age_of_onset": (
            "Onset at 6-12 months: maternal IgG wanes at 6 months, unmasking B-cell defect; "
            "Recurrent otitis media, sinusitis, pneumonia (encapsulated bacteria -- Streptococcus pneumoniae, Haemophilus influenzae); "
            "Chronic enteroviral meningitis/encephalitis: classic late complication in under-treated XLA; "
            "Giardia lamblia diarrhoea: characteristic in XLA (no secretory IgA in gut); "
            "No germinal centres -> no lymphadenopathy, tonsils absent clinically (distinguishes from hypogammaglobulinaemia of prematurity); "
            "Males ONLY (X-linked); female carriers asymptomatic with normal Ig levels due to B-cell X-inactivation skewing"
        ),
        "key_biomarker": (
            "B cells (CD19+): virtually absent (<1% of lymphocytes; normal >5%); "
            "Serum immunoglobulins: absent or severely low IgG (<2 g/L), IgA undetectable, IgM undetectable; "
            "Pre-B cells in bone marrow: absent at pro-B cell stage (arrested before BTK signalling step); "
            "BTK protein expression: absent on monocyte flow cytometry (rapid immunological screen); "
            "molecular: BTK pathogenic variant (hemizygous in males); missense/nonsense/frameshift/splice; "
            "T-cell count: NORMAL (differentiates XLA from SCID -- T cells present, B cells absent); "
            "NK cells: NORMAL in XLA (distinguishes from SCID-X1/JAK3 where NK absent); "
            "ferritin / Hgb: iron deficiency anaemia from chronic GI blood loss (Giardia, inflammation)"
        ),
        "pathognomonic": (
            "Male + absent B cells (CD19+ <1%) + absent Ig (IgG/IgA/IgM) + normal T cells + normal NK cells = XLA (BTK) until proven; "
            "Absent tonsils/lymph nodes clinically: no germinal centres; palpable lymphadenopathy argues AGAINST XLA; "
            "Chronic echovirus meningoencephalitis (dermatomyositis-like syndrome + CNS) = pathognomonic late XLA complication; "
            "DISTINGUISH from transient hypogammaglobulinaemia of infancy: transient = B cells PRESENT; Ig recovers by age 2-3 years; "
            "DISTINGUISH from CVID: CVID onset usually in 2nd-3rd decade; B cells present but non-functional; not male-exclusive; "
            "DISTINGUISH from SCID: SCID = T cells absent; XLA = T cells normal; immunophenotype differentiates immediately"
        ),
        "treatment": (
            "IVIG (or SCIG): 400-600 mg/kg every 3-4 weeks; trough IgG target >8-10 g/L; lifelong; "
            "SCIG (subcutaneous): equivalent efficacy to IVIG; home administration; less anaphylaxis; "
            "Antibiotic prophylaxis: co-trimoxazole or amoxicillin during infection-prone early childhood; "
            "Acute infections: aggressive early antibiotic therapy (avoid encapsulated bacterial sepsis); "
            "Intraventricular Ig: for chronic echoviral encephalitis (compassionate use + antiviral pleconaril); "
            "AVOID: live vaccines ABSOLUTELY (OPV -> VAPP; MMR, rotavirus, varicella, BCG, yellow fever -- all contraindicated); "
            "AVOID: rituximab (depletes residual B cells in rare partial XLA presentations); "
            "Gene therapy: preclinical for XLA; not yet licensed; HSCT rarely indicated (marrow normal except B-cell lineage); "
            "Pulmonary physio: rehabilitation for bronchiectasis from recurrent pneumonias; spirometry annually"
        ),
        "critical_flags": [
            "BTK-LIVE-VACCINES-ABSOLUTELY-CI: oral polio vaccine (OPV) causes vaccine-associated paralytic poliomyelitis (VAPP) in XLA -- do NOT administer OPV or any live-attenuated vaccine (MMR, varicella, rotavirus, BCG, yellow fever, live typhoid) to XLA patients or their household contacts if OPV is used; this is a life-threatening prescribing error that has caused death",
            "BTK-ABSENT-TONSILS-SIGN: clinical absence of tonsils and palpable lymph nodes in a male infant with recurrent sinopulmonary infections is a clinical clue for XLA; germinal centres require B cells to form; order B-cell count (CD19) and serum immunoglobulins urgently -- do not wait for recurrent hospitalisations before testing",
            "BTK-CHRONIC-ENTEROVIRAL-ENCEPHALITIS: echovirus/enterovirus meningoencephalitis is a chronic, progressive, often fatal complication of untreated or under-treated XLA; presents as dermatomyositis-like syndrome + progressive CNS deterioration; prevention = adequate IVIG trough IgG >8 g/L; treatment = intraventricular Ig + pleconaril (compassionate use)",
            "BTK-IVIG-TROUGH-TARGET: trough IgG >8-10 g/L substantially reduces serious bacterial infections compared to historical target of >5 g/L; test trough immediately before next IVIG dose; titrate dose upward if recurrent infections occur even at standard trough; individual variation in IgG catabolism rate is significant and affects dosing interval",
            "BTK-FEMALE-CARRIERS: XLA is X-linked recessive; female carriers have NORMAL immunoglobulins and normal B-cell counts due to X-inactivation skewing in favour of BTK-expressing B cells; carrier females are NOT immunodeficient; offer cascade genetic testing to all maternal relatives of affected males to identify carrier females before they have sons",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- IL2RG -- Common Gamma Chain / SCID-X1 --------------------------------------
    {
        "gene": "IL2RG",
        "alt_name": "IL2RG / Common Gamma Chain (SCID-X1 / X-linked Severe Combined Immunodeficiency)",
        "protein": (
            "IL2RG -- Xq13.1 XL -- IL2RG-369aa -- "
            "SCID-X1-Most-Common-SCID-T-B-Plus-NK-Minus-Phenotype -- "
            "Maternal-T-Cell-Engraftment-Must-Be-Excluded-Before-HSCT -- "
            "Gene-Therapy-OTL-101-Curative-First-EU-Licensed-SCID-Gene-Therapy -- "
            "Gamma-Chain-Shared-IL2-IL4-IL7-IL9-IL15-IL21-Receptors -- "
            "NK-Cells-Absent-Distinguishes-SCID-X1-From-XLA"
        ),
        "locus": "Xq13.1",
        "protein_size": "369 aa",
        "inheritance": "XL (X-linked recessive)",
        "age_of_onset": (
            "Onset within first 3-6 months of life (before maternal Ab wanes; failure-to-thrive + infections begin early); "
            "Profound lymphopenia: absolute lymphocyte count <3,000/uL in neonates (normal >2,500 at birth); "
            "Opportunistic infections: Pneumocystis jirovecii pneumonia (PCP), CMV, Candida, RSV -- life-threatening; "
            "Failure to thrive: weight loss, chronic diarrhoea, persistent oral thrush despite antifungal treatment; "
            "Maternal T-cell engraftment: maternal lymphocytes cross placenta; graft-versus-host disease (GvHD) if unrecognised; "
            "Males ONLY (X-linked); females may have skewed X-inactivation with mild partial T-cell defect (very rare)"
        ),
        "key_biomarker": (
            "T-cell count (CD3+): absent or severely reduced (<300 cells/uL); "
            "NK cells (CD16+CD56+): ABSENT -- key distinguishing feature from XLA and T-B-NK+ SCID types; "
            "B cells (CD19+): PRESENT but non-functional (maternal origin or autologous but non-functional); "
            "TREC (T-cell receptor excision circles): absent on newborn bloodspot screen (NBS); most sensitive early screen; "
            "T-cell proliferation to mitogens: absent (PHA, anti-CD3 stimulation); "
            "molecular: IL2RG pathogenic variant (hemizygous males; heterozygous carrier females); "
            "Maternal engraftment testing: HLA typing + chimerism studies to exclude maternal T-cell engraftment before transplant conditioning"
        ),
        "pathognomonic": (
            "Male + T-B+NK- immunophenotype + profound lymphopenia + absent TREC = SCID-X1 (IL2RG) until JAK3 excluded; "
            "Absent NK cells: T-B+NK- = SCID-X1 (IL2RG) or JAK3-SCID; T-B-NK+ = RAG1/RAG2/Artemis; "
            "Maternal engraftment: HLA typing shows chimerism = maternal T cells present -- CRITICAL identification before HSCT conditioning; "
            "DISTINGUISH from JAK3-SCID: clinically IDENTICAL T-B+NK-; JAK3 is AR (not XL) -- can affect females; test JAK3 in any female with T-B+NK- SCID; "
            "DISTINGUISH from XLA (BTK): XLA = T cells normal; B cells absent; NK normal; no opportunistic infections in infancy"
        ),
        "treatment": (
            "HSCT (haematopoietic stem cell transplantation): treatment of choice; best outcomes <3.5 months of age; HLA-identical sibling = best donor; "
            "MSD (matched sibling donor): >90% survival; T-cell engraftment without conditioning often sufficient; "
            "Haploidentical HSCT: T-cell depleted paternal graft if no MSD; reduced-intensity conditioning; "
            "Gene therapy (OTL-101 / ex-vivo lentiviral vector): EMA licensed 2021; curative for SCID-X1; avoids GvHD risk; "
            "Precautions before HSCT: isolation (protective environment); PCP prophylaxis (co-trimoxazole); antifungal; CMV monitoring; "
            "AVOID: live vaccines ABSOLUTELY; blood transfusions must be irradiated + CMV-negative; "
            "Maternal engraftment: exclude by HLA chimerism before conditioning; "
            "Immunoglobulin replacement: until immune reconstitution post-HSCT or gene therapy; "
            "Newborn screening: TREC-based NBS detects SCID-X1 at birth -- enables pre-symptomatic treatment with >90% survival"
        ),
        "critical_flags": [
            "IL2RG-MATERNAL-ENGRAFTMENT: maternal T lymphocytes cross the placenta in all pregnancies; in SCID-X1 they survive because the baby has no immune response to reject them; these maternal T cells cause severe GvHD if the baby receives HSCT without prior recognition; ALWAYS perform HLA chimerism testing before conditioning for HSCT -- this is non-negotiable",
            "IL2RG-GENE-THERAPY: OTL-101 (lentiviral vector gene therapy for SCID-X1) is EMA-licensed; offers correction without allogeneic donor GvHD risk; refer to specialised gene therapy centre; earlier gammaretroviral vectors caused T-cell leukaemia (insertional mutagenesis in 5/20 patients in early trials); lentiviral self-inactivating vectors have substantially improved safety profile",
            "IL2RG-IRRADIATED-BLOOD-MANDATORY: all blood products for SCID-X1 patients must be irradiated (to prevent transfusion-associated GvHD from donor lymphocytes) AND CMV-negative; a non-irradiated blood transfusion in a SCID patient can be fatal; inform blood bank and theatre staff explicitly on every admission",
            "IL2RG-TREC-NEWBORN-SCREEN: TREC (T-cell receptor excision circles) measured on Guthrie bloodspot detects SCID-X1 at birth before any symptoms; newborn screening for SCID has been adopted in USA, UK, and several EU countries; early HSCT (<3.5 months) achieves >90% survival vs <70% if delayed to symptomatic presentation; advocate for NBS implementation where not yet established",
            "IL2RG-FEMALE-CARRIERS-SYMPTOMATIC-RARE: female carriers of IL2RG variants very rarely develop partial immune deficiency due to non-random X-inactivation in T/NK cells; if a female has T-B+NK- SCID, test JAK3 first (AR) -- but also consider homozygous IL2RG in consanguineous families as an extremely rare possibility",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- ADA -- Adenosine Deaminase / ADA-SCID ----------------------------------------
    {
        "gene": "ADA",
        "alt_name": "ADA (ADA-SCID / Adenosine Deaminase Deficiency -- First Gene Therapy Disease)",
        "protein": (
            "ADA -- 20q13.12 AR -- ADA-363aa -- "
            "ADA-SCID-dATP-Toxic-Accumulation-Kills-T-B-NK-Cells-Equally -- "
            "PEG-ADA-Enzyme-Replacement-Bridges-to-Gene-Therapy -- "
            "Strimvelis-EMA-2016-First-Curative-Gene-Therapy-Licensed-Europe -- "
            "1990-First-Human-Gene-Therapy-Trial-ADA-SCID -- "
            "Adenosine-Deaminase-Purine-Metabolism-Lymphotoxic-dATP-Accumulation"
        ),
        "locus": "20q13.12",
        "protein_size": "363 aa",
        "inheritance": "AR (autosomal recessive)",
        "age_of_onset": (
            "Classic early-onset ADA-SCID: present in first 1-3 months of life (same clinical urgency as other SCID types); "
            "Delayed-onset ADA deficiency: 1-15 years; milder immune deficiency; misdiagnosed as CVID or recurrent chest infections; "
            "Late-onset/partial ADA: adulthood; recurrent respiratory infections; reduced Ig; residual ADA activity 1-5%; "
            "Toxic metabolites: deoxyadenosine and dATP accumulate in all tissues but uniquely toxic to lymphocytes; "
            "Skeletal abnormalities: costochondral flaring and cupping of ribs on X-ray -- distinctive among SCID types; "
            "Neurological: developmental delay, sensorineural hearing loss -- direct dATP toxicity on neurons (not secondary to infections); "
            "Both sexes affected equally (AR): unlike BTK/IL2RG which are X-linked"
        ),
        "key_biomarker": (
            "ADA enzyme activity: absent in red blood cells (RBC lysate assay); diagnostic in classic ADA-SCID; "
            "dATP levels: elevated in erythrocytes; directly correlates with immune deficiency severity and treatment monitoring; "
            "T cells (CD3+): profoundly reduced (B and NK cells also reduced -- pan-lymphopenia distinguishes from other SCID types); "
            "TREC: absent on newborn screen; "
            "Immunoglobulins: low IgG, IgA, IgM (all lineages affected -- unlike XLA where only Ig absent, T cells normal); "
            "molecular: ADA biallelic pathogenic variants; consanguinity increases risk (AR); "
            "Rib X-ray: costochondral cupping + rib flaring -- skeletal finding unique to ADA-SCID among immunodeficiencies; "
            "PEG-ADA monitoring: weekly ADA activity, dATP levels, lymphocyte counts once on ERT to guide dosing"
        ),
        "pathognomonic": (
            "Pan-lymphopenia (T+B+NK all reduced) + absent ADA enzyme activity in RBCs = ADA-SCID until proven; "
            "Costochondral flaring + cupping on chest X-ray in a SCID infant = ADA-SCID -- pathognomonic skeletal finding; "
            "DISTINGUISH from SCID-X1 (IL2RG): SCID-X1 = T-B+NK- (B cells present); ADA-SCID = T-B-NK- (all lineages depleted); "
            "DISTINGUISH from RAG1/RAG2 SCID: RAG = T-B-NK+ (NK cells normal); ADA-SCID = NK cells also depleted; "
            "Late-onset ADA deficiency mimics CVID in adolescents/adults -- measure ADA enzyme activity in any unexplained hypogammaglobulinaemia in patients <20 years especially with consanguinity; "
            "Both sexes equally affected (AR) -- distinguishes from XLA/SCID-X1 which are male-predominant"
        ),
        "treatment": (
            "Gene therapy (Strimvelis): ex-vivo gammaretroviral gene therapy; EMA approved 2016 (first licensed gene therapy for ADA-SCID); "
            "curative; autologous -- no GvHD risk; performed at specialist centres (Milan, London); "
            "Gene therapy (lentiviral vector): newer; improved safety profile over gammaretroviral; clinical trials ongoing globally; "
            "PEG-ADA (polyethylene glycol-conjugated ADA): enzyme replacement therapy; weekly IM injection; "
            "bridges to gene therapy or HSCT; NOT curative; reduces metabolite toxicity; allows partial immune reconstitution; "
            "HSCT: effective but GvHD risk; gene therapy preferred if no HLA-matched sibling; "
            "AVOID: discontinuing PEG-ADA abruptly (rebound dATP toxicity -- lymphocyte depletion recurs within days); "
            "Antibiotic/antifungal/antiviral prophylaxis: PCP co-trimoxazole; antifungal; CMV surveillance; "
            "IVIG: until immune reconstitution post-gene therapy or HSCT; "
            "Irradiated CMV-negative blood: mandatory as for all SCID types; "
            "Neurodevelopmental surveillance: hearing, developmental milestones -- annual even after immune reconstitution"
        ),
        "critical_flags": [
            "ADA-STRIMVELIS-FIRST-LICENSED-GENE-THERAPY: ADA-SCID was the first human gene therapy disease (1990 trial, W. French Anderson) and Strimvelis (2016, EMA) is the first approved gene therapy for a primary immunodeficiency in Europe; gene therapy is now the preferred option over HSCT when no HLA-matched sibling is available; refer early to a gene therapy centre -- do not accept 'HSCT is the only option' without referral",
            "ADA-PEG-ADA-BRIDGING-ONLY: PEG-ADA enzyme replacement is NOT curative but critically buys time for gene therapy or HSCT preparation; do NOT assume PEG-ADA is a long-term solution; antibody formation to PEG-ADA occurs in ~10% and causes ERT failure; monitor ADA activity and dATP levels regularly; never delay referral for definitive therapy",
            "ADA-SKELETAL-CLUE: costochondral cupping and rib flaring on chest X-ray are a distinctive feature of ADA-SCID not seen in any other SCID type; in a SCID infant with unusual rib findings on CXR, always test ADA enzyme activity in RBC lysate before full panel sequencing",
            "ADA-LATE-ONSET-MIMICS-CVID: partial ADA deficiency with residual enzyme activity 1-5% presents in childhood or adulthood with recurrent infections + low immunoglobulins -- clinically indistinguishable from CVID; measure ADA enzyme activity in ALL CVID-like patients under 20 years, especially if consanguinity or refractory to IVIG alone",
            "ADA-NEUROLOGICAL-BURDEN: neurodevelopmental delay, sensorineural hearing loss, and behavioural features occur in ADA-SCID independent of immune reconstitution; these are direct effects of dATP toxicity on neurons; annual audiological and neurodevelopmental assessment mandatory even after successful immune reconstitution with gene therapy or HSCT",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- RAG1 -- Recombination Activating Gene 1 / T-B-NK+ SCID --------------------
    {
        "gene": "RAG1",
        "alt_name": "RAG1 (RAG1-SCID / T-B-NK+ SCID or Omenn Syndrome -- Partial RAG1)",
        "protein": (
            "RAG1 -- 11p13 AR -- RAG1-1043aa -- "
            "T-B-Minus-NK-Plus-SCID-Complete-Loss -- "
            "Omenn-Syndrome-Partial-RAG1-Erythroderma-Eosinophilia-Elevated-IgE-NOT-Atopy -- "
            "VDJ-Recombination-Defect-No-T-or-B-Cell-Receptor-Formation -- "
            "NK-Cells-Present-Distinguishes-RAG-SCID-from-SCID-X1-and-JAK3 -- "
            "RAG1-RAG2-Heterodimer-Required-for-DNA-Cleavage-RSS"
        ),
        "locus": "11p13",
        "protein_size": "1043 aa",
        "inheritance": "AR (autosomal recessive)",
        "age_of_onset": (
            "Complete RAG1 loss: classic SCID presentation in first 3-6 months; "
            "Partial RAG1 (Omenn syndrome): erythroderma, alopecia, eosinophilia, elevated IgE, hepatosplenomegaly in first weeks of life; "
            "Omenn syndrome: oligoclonal autoreactive T cells escape thymus; attack host tissue; resembles severe atopic disease or GvHD; "
            "Hypomorphic RAG1: CID (combined immunodeficiency) with autoimmunity, granulomatous disease in childhood; "
            "Both sexes equally affected (AR); "
            "NK cells present: normal or elevated NK cells in RAG1-SCID (RAG enzymes not required for NK development)"
        ),
        "key_biomarker": (
            "T cells (CD3+): absent in complete RAG1-SCID; oligoclonal activated T cells in Omenn syndrome; "
            "B cells (CD19+): absent in both complete RAG1-SCID and Omenn syndrome; "
            "NK cells (CD16+CD56+): PRESENT and often elevated -- key distinguishing marker from SCID-X1/JAK3; "
            "IgE: markedly elevated in Omenn (>1000 IU/mL; often >5000); NOT expected in typical SCID or atopy alone; "
            "Eosinophilia: peripheral blood eosinophilia in Omenn syndrome; "
            "TREC: absent (no T cells; or oligoclonal TREC in Omenn -- quantitatively reduced); "
            "Skin biopsy in Omenn: lymphocytic infiltrate of activated T cells (differs histologically from atopic eczema); "
            "molecular: RAG1 biallelic variants; genotype-phenotype: null = classic SCID; hypomorphic missense = Omenn or CID"
        ),
        "pathognomonic": (
            "T-B-NK+ phenotype = RAG1, RAG2, or Artemis-SCID; differentiation requires molecular testing; "
            "Omenn syndrome: erythroderma + eosinophilia + elevated IgE + absent B cells + oligoclonal T cells = RAG1 or RAG2 until proven; "
            "DISTINGUISH Omenn from severe atopic eczema: atopic = B cells PRESENT; eosinophilia mild (<1500/uL); Omenn = B cells ABSENT; eosinophilia often >3000/uL; hepatosplenomegaly; "
            "DISTINGUISH Omenn from GvHD: GvHD = prior transfusion or maternal engraftment; Omenn = autologous oligoclonal T cells; HLA chimerism differentiates; "
            "DISTINGUISH from JAK3/SCID-X1: those are T-B+NK- (B cells present, NK absent); RAG1 = T-B-NK+ (B absent, NK present)"
        ),
        "treatment": (
            "Complete RAG1-SCID: HSCT -- treatment of choice; same urgency as SCID-X1; "
            "Conditioning: RAG-SCID patients typically require conditioning for B-cell engraftment (unlike gammachain-SCID); "
            "Omenn syndrome: immunosuppression FIRST before HSCT (cyclosporin A + steroids to control autoreactive T cells); "
            "HSCT after Omenn stabilisation: timing critical -- transplant before organ damage from autoreactive T cells; "
            "PCP prophylaxis, antifungal, antiviral: standard SCID care; "
            "IVIG: while awaiting HSCT; "
            "AVOID: live vaccines; non-irradiated blood products; "
            "Gene therapy: preclinical for RAG1; early-phase trials underway; not yet licensed; "
            "Skin care in Omenn: emollients + topical steroids; systemic cyclosporin for T-cell activation control"
        ),
        "critical_flags": [
            "RAG1-OMENN-NOT-ATOPY: Omenn syndrome (partial RAG1 or RAG2) presents with erythroderma, elevated IgE, and eosinophilia -- easily confused with severe atopic dermatitis; the critical distinction is ABSENT B CELLS in Omenn (B cells present in atopy); always check CD19+ B-cell count in any infant with severe rash + hepatosplenomegaly + failure to thrive",
            "RAG1-NK-CELLS-PRESENT: RAG enzymes are not required for NK-cell development; NK cells are present (or elevated) in RAG1/RAG2-SCID; this T-B-NK+ pattern distinguishes RAG-SCID from SCID-X1 (T-B+NK-) and JAK3-SCID (T-B+NK-) -- immunophenotyping is the first and fastest diagnostic step",
            "RAG1-GENOTYPE-PHENOTYPE: null RAG1 mutations (frameshift, nonsense) produce classic T-B-NK+ SCID; hypomorphic missense mutations with residual RAG1 activity (2-30% of normal) produce Omenn syndrome or combined immunodeficiency with autoimmunity (CID-A); never assume a patient has classic SCID without reading the genetic report for variant type and residual activity prediction",
            "RAG1-OMENN-IMMUNOSUPPRESSION-MANDATORY-BEFORE-HSCT: HSCT without prior immunosuppression in Omenn syndrome carries very high mortality; the autoreactive oligoclonal T cells cause severe GvH-like organ damage post-transplant; cyclosporin + corticosteroids to control autoreactive T cells before conditioning is the standard of care -- do not skip this step",
            "RAG1-HYPOMORPHIC-LATE-PRESENTATION: hypomorphic RAG1 mutations may present in adolescence or adulthood with autoimmunity, granulomatous lung disease, or EBV-driven lymphoproliferation -- mimicking sarcoidosis or CVID; check B-cell subsets, T-cell repertoire diversity, and RAG1 molecular testing in unexplained autoimmune granulomatous disease in young adults",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- RAG2 -- Recombination Activating Gene 2 / T-B-NK+ SCID / Omenn ------------
    {
        "gene": "RAG2",
        "alt_name": "RAG2 (RAG2-SCID / Omenn Syndrome -- Partial RAG2 / T-B-NK+ SCID)",
        "protein": (
            "RAG2 -- 11p13 AR -- RAG2-527aa -- "
            "T-B-Minus-NK-Plus-SCID-Omenn-Syndrome-When-Partial -- "
            "RAG2-Forms-Heterodimer-With-RAG1-For-VDJ-DNA-Cleavage -- "
            "Erythroderma-Eosinophilia-Elevated-IgE-Omenn-NOT-Atopic-Eczema -- "
            "NK-Cells-PRESENT-Distinguishes-T-B-NK-Plus-Pattern -- "
            "RAG1-RAG2-Genes-Adjacent-11p13-Test-Both-Together"
        ),
        "locus": "11p13",
        "protein_size": "527 aa",
        "inheritance": "AR (autosomal recessive)",
        "age_of_onset": (
            "Complete RAG2 null: classic SCID in first 3-6 months; indistinguishable from RAG1-SCID by clinical phenotype alone; "
            "Partial RAG2 (Omenn syndrome): erythroderma + eosinophilia + elevated IgE + hepatosplenomegaly within first weeks; "
            "NK-cell development normal: NK cells present regardless of RAG2 status; "
            "Both sexes equally affected (AR); "
            "RAG1 and RAG2 genes adjacent on chromosome 11p13; compound heterozygosity or homozygosity common; "
            "Hypomorphic RAG2: later childhood or adolescent presentation with CID + autoimmunity possible"
        ),
        "key_biomarker": (
            "T cells (CD3+): absent in complete RAG2-SCID; oligoclonal in Omenn syndrome; "
            "B cells (CD19+): absent in both phenotypes; "
            "NK cells (CD16+CD56+): PRESENT (same as RAG1-SCID) -- NK development independent of RAG enzymes; "
            "IgE: elevated in Omenn (can exceed 10,000 IU/mL -- much higher than typical atopy); "
            "Eosinophil count: elevated in peripheral blood in Omenn syndrome; "
            "RAG2 functional assay: recombination activity of mutant RAG2 on episomal substrate in research setting; "
            "molecular: RAG2 biallelic variants; genotype-phenotype correlation: null = SCID; residual = Omenn or CID; "
            "TREC: absent on newborn screen; "
            "Lymphocyte proliferation: absent to mitogens (PHA) in complete SCID; oligoclonal activation in Omenn"
        ),
        "pathognomonic": (
            "T-B-NK+ phenotype: same as RAG1 SCID; molecular testing differentiates RAG1 from RAG2; "
            "Omenn syndrome RAG2: same features as RAG1 Omenn (erythroderma, eosinophilia, elevated IgE, absent B cells, hepatosplenomegaly); "
            "DISTINGUISH from RAG1: always sequence both RAG1 and RAG2 together -- adjacent genes, test simultaneously; "
            "DISTINGUISH from Artemis-SCID: Artemis = radiation-sensitive (CSI testing); RAG1/2 = NOT radiation-sensitive; "
            "DISTINGUISH from SCID-X1/JAK3: those are T-B+NK- (B present, NK absent); RAG2-SCID is T-B-NK+ (B absent, NK present)"
        ),
        "treatment": (
            "HSCT: same indications and urgency as RAG1-SCID; "
            "Omenn syndrome: immunosuppression (cyclosporin + steroids) before HSCT -- mandatory as for RAG1 Omenn; "
            "Conditioning: myeloablative or reduced-intensity depending on donor; conditioning required for B-cell reconstitution; "
            "Isolation precautions: all SCID precautions (gown/glove/mask, HEPA filtered air, reverse isolation); "
            "PCP prophylaxis (co-trimoxazole), antifungal (fluconazole or itraconazole), CMV surveillance (weekly PCR); "
            "IVIG while awaiting HSCT; "
            "Gene therapy: RAG2 gene therapy clinical trials underway; not yet licensed; "
            "AVOID: live vaccines; non-irradiated blood; IVIG without irradiated labelling confirmed; "
            "Skin: emollients + topical steroids for Omenn erythroderma; systemic cyclosporin"
        ),
        "critical_flags": [
            "RAG2-OMENN-ELEVATED-IGE-NOT-ALLERGY: in RAG2 Omenn syndrome, IgE levels can exceed 10,000 IU/mL -- far higher than most atopic patients; this is NOT IgE-mediated allergy; IgE is driven by the few oligoclonal B-cell precursors activated by autoreactive T cells; allergy investigation is a diagnostic distraction -- perform lymphocyte immunophenotyping immediately",
            "RAG2-PANEL-TESTING-RAG1-AND-RAG2-TOGETHER: RAG1 and RAG2 are adjacent on 11p13 and functionally inseparable; ALWAYS test both genes together on a primary immunodeficiency panel; testing only RAG2 and missing a compound heterozygote with one variant in RAG1 requires re-testing -- use multi-gene PID panels from the outset",
            "RAG2-RADIATION-SENSITIVITY-TEST: Artemis-SCID (DCLRE1C) is also T-B-NK+ but radiation-sensitive; chromosomal sensitivity index (CSI) or colony survival assay with ionising radiation performed on patient fibroblasts distinguishes Artemis from RAG1/2; RAG2-SCID is NOT radiation-sensitive -- this distinction is critical for safe conditioning regimen selection",
            "RAG2-CONDITIONING-DEBATE: for RAG1/RAG2-SCID, optimal conditioning intensity for HSCT is debated; unlike gamma-chain SCID (which engrafts T cells without conditioning), RAG-SCID requires sufficient conditioning for B-cell engraftment to achieve antibody production; reduced-intensity conditioning alone may achieve T-cell but not B-cell reconstitution in some patients",
            "RAG2-HYPOMORPHIC-ADULT-PRESENTATION: hypomorphic RAG2 mutations with 5-15% residual activity may present in adulthood with selective antibody deficiency, recurrent pneumonias, and autoimmune cytopenias -- misdiagnosed as CVID for years; check T-cell repertoire diversity (TCR Vbeta spectratyping) and RAG2 sequencing in young adults with unexplained combined immunodeficiency",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- DCLRE1C -- Artemis / Radiation-Sensitive SCID --------------------------------
    {
        "gene": "DCLRE1C",
        "alt_name": "DCLRE1C / Artemis (Artemis-SCID / Radiation-Sensitive T-B-NK+ SCID)",
        "protein": (
            "DCLRE1C -- 10p13 AR -- DCLRE1C-692aa -- "
            "Artemis-SCID-T-B-NK-Plus-Radiation-Sensitive -- "
            "Defective-DNA-Double-Strand-Break-Repair-VDJ-Recombination-Failure -- "
            "Reduced-Intensity-Conditioning-MANDATORY-Standard-Chemo-Is-Lethal -- "
            "Athabascan-Speaking-Native-American-Founder-Mutation -- "
            "DCLRE1C-Artemis-Endonuclease-DNA-PKcs-Partner-VDJ-Hairpin-Opening"
        ),
        "locus": "10p13",
        "protein_size": "692 aa",
        "inheritance": "AR (autosomal recessive)",
        "age_of_onset": (
            "Classic early SCID: presentation in first 3-6 months identical to other T-B-NK+ SCID types; "
            "Radiation sensitivity: recognised at cellular level; not clinically apparent in infancy but CRITICAL before conditioning; "
            "Athabascan-speaking Native Americans: high frequency founder deletion in DCLRE1C gene; "
            "Post-HSCT with standard conditioning: radiation-sensitive cells -> catastrophic multi-organ toxicity; "
            "Surviving without HSCT: progressive opportunistic infections, failure to thrive, death in first 1-2 years; "
            "Both sexes equally affected (AR)"
        ),
        "key_biomarker": (
            "T cells (CD3+): absent; "
            "B cells (CD19+): absent (same T-B-NK+ pattern as RAG1/2); "
            "NK cells (CD16+CD56+): PRESENT; "
            "Radiation sensitivity testing: chromosomal sensitivity index (CSI) -- cells exposed to ionising radiation show excess chromosomal breaks; "
            "Colony survival assay: fibroblast colony survival after irradiation severely reduced (<10% of normal in Artemis-SCID); "
            "TREC: absent on newborn screen; "
            "molecular: DCLRE1C biallelic pathogenic variants; Athabascan founder deletion identifiable by MLPA; "
            "DNA repair assay: Artemis endonuclease activity absent in lymphocyte or fibroblast cell-free extracts"
        ),
        "pathognomonic": (
            "T-B-NK+ SCID + radiation sensitivity (CSI or colony survival assay) = Artemis-SCID (DCLRE1C) until proven; "
            "Athabascan Native American ancestry + T-B-NK+ SCID = Artemis-SCID founder mutation with very high prior probability; "
            "DISTINGUISH from RAG1/RAG2 SCID: same T-B-NK+ phenotype; radiation sensitivity differentiates; molecular confirms; "
            "DISTINGUISH from DNA-PKcs SCID (PRKDC): also radiation-sensitive; molecular testing required to assign gene; "
            "DISTINGUISH from Omenn (partial RAG): Omenn = elevated IgE + eosinophilia + oligoclonal T cells; Artemis = no T cells at all"
        ),
        "treatment": (
            "HSCT: curative; mandatory; perform as early as possible; "
            "REDUCED-INTENSITY CONDITIONING IS MANDATORY: standard busulfan/cyclophosphamide is lethal in Artemis-SCID due to radiation sensitivity of non-haematopoietic cells; "
            "Reduced-intensity conditioning: fludarabine + low-dose busulfan (or treosulfan) standard; "
            "Athabascan patients: historically very poor outcomes with myeloablative conditioning; substantially improved with fludarabine-based regimens; "
            "Gene therapy: DCLRE1C gene therapy in early-phase clinical trials (early results promising); not yet licensed; "
            "Isolation and prophylaxis: same as all SCID (PCP, antifungal, CMV); reverse isolation; "
            "Irradiated CMV-negative blood: mandatory; "
            "AVOID: live vaccines; high-dose cyclophosphamide; busulfan at myeloablative doses; "
            "CT/radiation minimise: radiation sensitivity persists in non-haematopoietic cells post-HSCT -- prefer MRI lifelong"
        ),
        "critical_flags": [
            "DCLRE1C-RADIATION-SENSITIVE-CONDITIONING-LETHAL: standard myeloablative conditioning (full-dose busulfan + cyclophosphamide) causes lethal multi-organ toxicity in Artemis-SCID due to defective DNA DSB repair in non-haematopoietic cells; ALWAYS use reduced-intensity fludarabine-based conditioning; administering myeloablative conditioning to an undiagnosed Artemis patient is a fatal transplant protocol error",
            "DCLRE1C-ATHABASCAN-FOUNDER: a founder deletion in DCLRE1C causes SCID in Athabascan-speaking Native Americans of Canada and USA (Navajo, Apache ancestral populations) with a carrier frequency of approximately 1:40 in affected communities; screen with targeted MLPA for this deletion before full sequencing in any Athabascan infant with T-B-NK+ SCID",
            "DCLRE1C-CSI-TEST-BEFORE-TRANSPLANT: chromosomal sensitivity index (CSI) testing of patient fibroblasts or lymphocytes with ionising radiation is the clinical test for radiation sensitivity; this test or DCLRE1C molecular exclusion MUST be completed before conditioning for HSCT in any T-B-NK+ SCID patient -- omitting radiation sensitivity testing is a protocol deviation",
            "DCLRE1C-CT-EXPOSURE-MINIMISE: radiation-sensitive non-haematopoietic cells persist even after successful HSCT in Artemis-SCID; minimise CT scans and fluoroscopy lifelong; prefer MRI for diagnostic imaging where feasible; counsel families about radiation exposure in medical imaging and avoid unnecessary screening CT",
            "DCLRE1C-GENE-THERAPY-HORIZON: Artemis gene therapy (DCLRE1C lentiviral vector) is in early-phase clinical trials; initial patients show sustained T and B cell reconstitution without conditioning toxicity; refer affected patients to gene therapy trial centres as an alternative to allogeneic HSCT -- especially if no suitable matched donor is available",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- JAK3 -- Janus Kinase 3 / AR T-B+NK- SCID ------------------------------------
    {
        "gene": "JAK3",
        "alt_name": "JAK3 (JAK3-SCID / T-B+NK- SCID -- AR Phenocopy of SCID-X1 / Janus Kinase 3)",
        "protein": (
            "JAK3 -- 19p13.11 AR -- JAK3-1124aa -- "
            "JAK3-SCID-T-B-Plus-NK-Minus-IDENTICAL-To-SCID-X1-Phenotype -- "
            "Autosomal-Recessive-NOT-X-Linked-Females-Can-Be-Affected -- "
            "JAK3-Signals-Downstream-IL2RG-Common-Gamma-Chain -- "
            "Test-JAK3-AND-IL2RG-Together-In-Any-T-B-Plus-NK-Minus-Patient -- "
            "JAK3-Selective-Kinase-Exclusively-Haematopoietic-Cell-Expression"
        ),
        "locus": "19p13.11",
        "protein_size": "1124 aa",
        "inheritance": "AR (autosomal recessive)",
        "age_of_onset": (
            "Identical clinical presentation to SCID-X1 (IL2RG): first 3-6 months; failure to thrive; opportunistic infections; "
            "T-B+NK- phenotype: B cells present but non-functional; T and NK cells absent; "
            "Unlike SCID-X1: JAK3-SCID is AR -- FEMALES CAN AND DO DEVELOP SCID from JAK3 deficiency; "
            "Consanguinity increases risk (AR); accounts for 5-10% of all SCID diagnoses; "
            "Both sexes affected equally; "
            "Maternal engraftment: same risk as SCID-X1 (all SCID types share this risk); "
            "Clinical: PCP, CMV, Candida, failure to thrive -- identical to SCID-X1 without molecular diagnosis"
        ),
        "key_biomarker": (
            "T cells (CD3+): absent or severely reduced; "
            "NK cells (CD16+CD56+): ABSENT -- same as SCID-X1 (IL2RG); "
            "B cells (CD19+): PRESENT but non-functional; "
            "TREC: absent on newborn screen; "
            "JAK3 protein expression: may be reduced on intracellular flow cytometry; "
            "molecular: JAK3 biallelic pathogenic variants; consanguinity -> homozygous common; "
            "IL2RG sequencing: must be NEGATIVE in males to assign JAK3; females cannot have hemizygous IL2RG so JAK3 is the first test; "
            "STAT5 phosphorylation: absent/reduced in response to IL-2 or IL-15 stimulation (functional downstream pathway assay)"
        ),
        "pathognomonic": (
            "T-B+NK- in a FEMALE infant = JAK3-SCID until proven (cannot be SCID-X1 which requires hemizygous X-linked mutation); "
            "T-B+NK- in a male: test IL2RG FIRST (more common in XL); if IL2RG normal, test JAK3; "
            "DISTINGUISH from IL2RG (SCID-X1): clinically IDENTICAL; differentiated only by molecular testing; inheritance pattern differs; "
            "DISTINGUISH from ZAP70-deficiency: ZAP70 = CD8+ T cells absent but CD4+ T cells present; not the same as JAK3; "
            "DISTINGUISH from ADA-SCID: ADA = T-B-NK- (all lineages); JAK3 = T-B+NK- (B cells present); "
            "JAK3 pathway: JAK3 -> STAT5 -> IL-7 signalling -> T-cell development; same downstream signalling block as IL2RG loss"
        ),
        "treatment": (
            "HSCT: treatment of choice; same as SCID-X1; "
            "Conditioning: not radiation-sensitive; standard or reduced-intensity based on donor; "
            "Gene therapy: JAK3 gene therapy technically challenging (constitutive JAK3 expression risks oncogenic signalling); autologous gene correction approaches under investigation; "
            "Maternal engraftment: exclude by HLA chimerism before conditioning -- same as SCID-X1; "
            "Prophylaxis: PCP (co-trimoxazole), antifungal, CMV surveillance; reverse isolation; "
            "Irradiated CMV-negative blood: mandatory; "
            "IVIG: until immune reconstitution post-HSCT; "
            "AVOID: live vaccines; JAK inhibitors (ruxolitinib, tofacitinib) in immunocompetent patients deplete T/NK -- do not use in JAK3-SCID; "
            "Newborn screen: TREC detects JAK3-SCID same as all other SCID types"
        ),
        "critical_flags": [
            "JAK3-FEMALE-SCID-X1-PHENOTYPE: when a female infant presents with T-B+NK- SCID, the diagnosis CANNOT be SCID-X1 (X-linked hemizygous); JAK3-SCID is the principal AR diagnosis; consanguinity, affected siblings of either sex, and parental origin are clues; gene panel testing covering both JAK3 and IL2RG is mandatory in all T-B+NK- SCID regardless of sex",
            "JAK3-IL2RG-PHENOCOPY: JAK3 signals exclusively downstream of the IL2RG common gamma chain; loss of JAK3 produces identical downstream signalling failure and identical T-B+NK- SCID phenotype; only molecular testing distinguishes the two; this distinction has critical implications for family genetic counselling (XL recurrence risk vs AR 25% recurrence risk) and for gene therapy eligibility",
            "JAK3-INHIBITOR-PARADOX: JAK inhibitors (tofacitinib, ruxolitinib, baricitinib) used clinically for autoimmune diseases block JAK1/JAK3 signalling and CAUSE secondary immunodeficiency resembling JAK3-SCID; patients on JAK inhibitors are at PCP and opportunistic infection risk -- PCP prophylaxis is mandatory; do not confuse iatrogenic JAK inhibition with inherited JAK3 deficiency when interpreting immunological results",
            "JAK3-GENE-THERAPY-CHALLENGE: unlike IL2RG gene therapy where restoring the common gamma chain reconstitutes signalling at physiological levels, JAK3 must be expressed at carefully regulated levels; constitutive JAK3 overexpression risks oncogenic transformation; lentiviral self-inactivating vectors with JAK3 under endogenous promoter control are being developed",
            "JAK3-MATERNAL-ENGRAFTMENT-SAME-RISK: JAK3-SCID carries the same maternal T-cell engraftment risk as SCID-X1; always perform HLA chimerism testing before conditioning; maternal T cells in the infant may initially appear as a partial immune reconstitution -- this is a diagnostic trap; distinguish by HLA typing of the T-cell population",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- TNFRSF13B -- TACI / CVID ---------------------------------------------------
    {
        "gene": "TNFRSF13B",
        "alt_name": "TNFRSF13B / TACI (CVID / Common Variable Immunodeficiency -- Most Common Symptomatic PID in Adults)",
        "protein": (
            "TNFRSF13B -- 17p11.2 AD/AR -- TNFRSF13B-293aa -- "
            "CVID-Most-Common-Symptomatic-PID-Adults-Prevalence-1-in-25000 -- "
            "Low-IgG-Less-Than-4gL-Low-IgA-IgM-Absent-Vaccine-Responses -- "
            "Autoimmune-Complications-25pct-Lymphoma-Risk-8x-Elevated -- "
            "10pct-CVID-Monogenic-TNFRSF13B-NFKB1-CTLA4-LRBA -- "
            "TACI-B-Cell-Differentiation-BAFF-APRIL-Receptor-Isotype-Switching"
        ),
        "locus": "17p11.2",
        "protein_size": "293 aa",
        "inheritance": "AD (heterozygous; reduced penetrance) or AR (homozygous/compound het; more severe)",
        "age_of_onset": (
            "Classic CVID onset: 2nd-3rd decade (adolescence to early adulthood); diagnosis often delayed 5-10 years; "
            "Second peak: age 6-10 years (paediatric CVID -- ~20% of cases); "
            "Recurrent sinopulmonary infections: encapsulated bacterial pneumonia (S. pneumoniae), sinusitis, otitis; "
            "Giardia lamblia: chronic diarrhoea and malabsorption (absent secretory IgA) -- characteristic; "
            "Autoimmune complications: immune thrombocytopenia (ITP), autoimmune haemolytic anaemia (AIHA), inflammatory bowel disease (~25% of CVID patients); "
            "Lymphoma: 8-fold elevated non-Hodgkin lymphoma risk (predominantly B-cell NHL, marginal zone lymphoma); "
            "Bronchiectasis: from recurrent pneumonias if Ig replacement started late (preventable with early diagnosis)"
        ),
        "key_biomarker": (
            "IgG: <4 g/L (or >2 SD below age-adjusted normal); "
            "IgA and/or IgM: low in addition to IgG (at least one of IgA or IgM must also be reduced); "
            "Vaccine responses: absent or severely reduced; test pre/post pneumococcal polysaccharide vaccine (PPV23); antibody response <50% of protected titre = diagnostic criterion; "
            "B cells (CD19+): present but non-functional in most CVID subtypes; "
            "Switched memory B cells (CD27+IgM-IgD-): severely reduced in most CVID subtypes (EUROclass classification system); "
            "TNFRSF13B molecular: heterozygous variants (C104R, A181E most studied) -- penetrance reduced; functional significance must be confirmed in context; "
            "CT chest: bronchiectasis, granulomatous-lymphocytic interstitial lung disease (GLILD), hilar lymphadenopathy, splenomegaly; "
            "Lymph node biopsy if lymphadenopathy: exclude lymphoma; diagnose reactive hyperplasia or GLILD"
        ),
        "pathognomonic": (
            "Low IgG (<4 g/L) ALONE is insufficient for CVID diagnosis -- MUST also have absent vaccine responses + age >4 years + exclusion of secondary causes; "
            "Exclusion of secondary causes mandatory: nephrotic syndrome (IgG loss), protein-losing enteropathy, lymphoma, rituximab therapy, anti-epileptics, thymoma; "
            "TNFRSF13B (TACI) variants in CVID: heterozygous C104R or A181E found in 8-10% CVID (vs 1-2% controls); risk factors NOT fully penetrant causes; homozygous TACI = more severe CVID; "
            "DISTINGUISH from XLA: XLA = absent B cells; male only; onset in infancy; CVID = B cells present; both sexes; adult onset; "
            "DISTINGUISH from transient hypogammaglobulinaemia: spontaneous recovery by age 2-3; CVID = persistent decline without recovery; "
            "DISTINGUISH secondary hypogammaglobulinaemia: rituximab, chemotherapy, nephrotic syndrome, protein-losing enteropathy -- exclude before CVID label"
        ),
        "treatment": (
            "IVIG or SCIG: 400-600 mg/kg every 3-4 weeks; trough IgG target >8 g/L (higher target than historical 5 g/L); lifelong; "
            "SCIG (subcutaneous): equivalent efficacy; lower anaphylaxis rate; home administration; weekly or biweekly; "
            "Antibiotic prophylaxis: azithromycin prophylaxis in bronchiectasis patients (3x/week); "
            "Autoimmune complications: prednisolone-sparing immunosuppressants (mycophenolate, azathioprine; rituximab for ITP/AIHA with caution -- further depletes B cells); "
            "GLILD (granulomatous-lymphocytic interstitial lung disease): prednisolone +/- rituximab +/- azathioprine; "
            "Lymphoma surveillance: annual clinical review; CT if lymphadenopathy or constitutional B symptoms; "
            "AVOID: live vaccines (not absolutely CI as in SCID but avoided in clinical practice -- minimal response anyway); "
            "Monogenic CVID treatment: CTLA4 haploinsufficiency = abatacept; LRBA deficiency = abatacept; PI3K-delta GOF = leniolisib; -- specific targeted therapies change outcomes; "
            "Genetic testing: 10% of CVID patients have identifiable monogenic cause -- multi-gene panel testing changes management"
        ),
        "critical_flags": [
            "TNFRSF13B-LOW-IGG-ALONE-NOT-ENOUGH: a serum IgG below 4 g/L is NECESSARY but NOT SUFFICIENT for CVID diagnosis; the patient must ALSO have documented absent or severely reduced vaccine responses (pre/post polysaccharide pneumococcal vaccine) AND be older than 4 years; failure to document vaccine responses leads to over-diagnosis of CVID and inappropriate lifelong IVIG therapy in patients who may recover",
            "TNFRSF13B-LYMPHOMA-8X-RISK: non-Hodgkin lymphoma (predominantly B-cell NHL; marginal zone lymphoma most common) risk is 8-fold elevated in CVID; unexplained lymphadenopathy, constitutional B symptoms (fever, night sweats, weight loss >10%), or rising LDH in a CVID patient = urgent CT chest/abdomen/pelvis and lymph node biopsy; lymphoma surveillance is mandatory at every annual review",
            "TNFRSF13B-AUTOIMMUNE-25PCT: approximately 25% of CVID patients develop autoimmune complications -- ITP, AIHA, inflammatory bowel disease, vitiligo, alopecia areata; these arise from dysregulated B-cell tolerance mechanisms despite hypogammaglobulinaemia; immunosuppressants may further deplete already-low immunoglobulins; combine IVIG with careful immunosuppression and monitor Ig levels closely",
            "TNFRSF13B-MONOGENIC-10PCT: 10% of CVID patients have an identifiable monogenic cause with specific targeted treatment -- CTLA4 haploinsufficiency (abatacept), LRBA deficiency (abatacept), PI3Kdelta gain-of-function (leniolisib/idelalisib), NFKB1/NFKB2 haploinsufficiency; these patients often have early lymphoproliferation, organomegaly, and autoimmunity as distinguishing features; multi-gene PID panel testing is mandatory in all CVID patients especially those with atypical features",
            "TNFRSF13B-VARIANTS-RISK-NOT-CAUSATION: TNFRSF13B heterozygous variants (C104R, A181E) are found in 8-10% of CVID patients but also in 1-2% of the general population; they are risk alleles with incomplete penetrance; report TACI variants as 'risk allele contributing to CVID phenotype' not as 'the monogenic cause of CVID'; functional B-cell isotype switching studies and family penetrance data are needed before assigning pathogenicity and informing family cascade testing",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _generate_cohort(gene_entry: dict) -> list:
    rng = random.Random(gene_entry["seed"])
    gene = gene_entry["gene"]
    pts = []
    for i in range(40):
        is_xla = gene == "BTK"
        is_scid_x1 = gene == "IL2RG"
        is_ada = gene == "ADA"
        is_rag1 = gene == "RAG1"
        is_rag2 = gene == "RAG2"
        is_artemis = gene == "DCLRE1C"
        is_jak3 = gene == "JAK3"
        is_cvid = gene == "TNFRSF13B"

        # Sex assignment
        if is_xla or is_scid_x1:
            sex = "M"
        elif is_cvid:
            sex = rng.choice(["M", "F", "F"])
        else:
            sex = rng.choice(["M", "F"])

        # Age at diagnosis
        if is_cvid:
            age_dx = rng.randint(14, 55)
            age_unit = "years"
        else:
            age_dx = rng.randint(1, 18)
            age_unit = "months"

        # -- BTK / XLA phenotype --
        sinopulmonary_infections = is_xla and rng.random() < 0.95
        absent_b_cells = is_xla
        igg_level_xla = round(rng.uniform(0.0, 0.8), 2) if is_xla else None
        on_ivig = (is_xla or is_cvid) and rng.random() < 0.90
        live_vaccine_given_error = is_xla and rng.random() < 0.08
        enteroviral_encephalitis = is_xla and rng.random() < 0.07

        # -- SCID-X1 (IL2RG) phenotype --
        pcp_pneumonia = is_scid_x1 and rng.random() < 0.45
        maternal_engraftment = is_scid_x1 and rng.random() < 0.20
        gene_therapy_received = is_scid_x1 and rng.random() < 0.22
        hsct_received_scidx1 = is_scid_x1 and rng.random() < 0.65

        # -- ADA-SCID phenotype --
        on_peg_ada = is_ada and rng.random() < 0.50
        strimvelis_received = is_ada and rng.random() < 0.18
        skeletal_abnormality = is_ada and rng.random() < 0.35
        neurological_features = is_ada and rng.random() < 0.28
        pan_lymphopenia = is_ada

        # -- RAG1 phenotype --
        omenn_syndrome_rag1 = is_rag1 and rng.random() < 0.30
        elevated_ige_rag1 = is_rag1 and omenn_syndrome_rag1
        eosinophilia_rag1 = is_rag1 and omenn_syndrome_rag1
        erythroderma_rag1 = is_rag1 and omenn_syndrome_rag1

        # -- RAG2 phenotype --
        omenn_syndrome_rag2 = is_rag2 and rng.random() < 0.28
        elevated_ige_rag2 = is_rag2 and omenn_syndrome_rag2
        eosinophilia_rag2 = is_rag2 and omenn_syndrome_rag2
        erythroderma_rag2 = is_rag2 and omenn_syndrome_rag2

        # -- Artemis phenotype --
        radiation_sensitive = is_artemis
        athabascan_ancestry = is_artemis and rng.random() < 0.30
        standard_conditioning_error = is_artemis and rng.random() < 0.05

        # -- JAK3 phenotype --
        female_jak3 = is_jak3 and sex == "F"
        misdiagnosed_scidx1 = is_jak3 and rng.random() < 0.25
        jak3_maternal_engraftment = is_jak3 and rng.random() < 0.18

        # -- CVID / TNFRSF13B phenotype --
        igg_level_cvid = round(rng.uniform(1.0, 3.9), 1) if is_cvid else None
        absent_vaccine_response = is_cvid and rng.random() < 0.92
        autoimmune_complication = is_cvid and rng.random() < 0.25
        lymphoma = is_cvid and rng.random() < 0.08
        bronchiectasis = is_cvid and rng.random() < 0.35
        glild = is_cvid and rng.random() < 0.15
        monogenic_cause_found = is_cvid and rng.random() < 0.10
        diagnosis_delay_years = rng.randint(3, 15) if is_cvid else None

        pts.append({
            "patient_id": f"{gene}-{i+1:03d}",
            "age_at_diagnosis": max(0, age_dx),
            "age_unit": age_unit,
            "sex": sex,
            # XLA / BTK fields
            "sinopulmonary_infections": sinopulmonary_infections,
            "absent_b_cells": absent_b_cells,
            "igg_level_xla_g_per_L": igg_level_xla,
            "on_ivig": on_ivig,
            "live_vaccine_error": live_vaccine_given_error,
            "enteroviral_encephalitis": enteroviral_encephalitis,
            # SCID-X1 / IL2RG fields
            "pcp_pneumonia": pcp_pneumonia,
            "maternal_engraftment": maternal_engraftment,
            "gene_therapy_received": gene_therapy_received,
            "hsct_received": hsct_received_scidx1,
            # ADA-SCID fields
            "on_peg_ada": on_peg_ada,
            "strimvelis_received": strimvelis_received,
            "skeletal_abnormality": skeletal_abnormality,
            "neurological_features": neurological_features,
            "pan_lymphopenia": pan_lymphopenia,
            # RAG1 fields
            "omenn_syndrome_rag1": omenn_syndrome_rag1,
            "elevated_ige_rag1": elevated_ige_rag1,
            "eosinophilia_rag1": eosinophilia_rag1,
            "erythroderma_rag1": erythroderma_rag1,
            # RAG2 fields
            "omenn_syndrome_rag2": omenn_syndrome_rag2,
            "elevated_ige_rag2": elevated_ige_rag2,
            "eosinophilia_rag2": eosinophilia_rag2,
            "erythroderma_rag2": erythroderma_rag2,
            # Artemis fields
            "radiation_sensitive": radiation_sensitive,
            "athabascan_ancestry": athabascan_ancestry,
            "standard_conditioning_error": standard_conditioning_error,
            # JAK3 fields
            "female_jak3_patient": female_jak3,
            "misdiagnosed_as_scidx1": misdiagnosed_scidx1,
            "jak3_maternal_engraftment": jak3_maternal_engraftment,
            # CVID / TNFRSF13B fields
            "igg_level_cvid_g_per_L": igg_level_cvid,
            "absent_vaccine_response": absent_vaccine_response,
            "autoimmune_complication": autoimmune_complication,
            "lymphoma": lymphoma,
            "bronchiectasis": bronchiectasis,
            "glild": glild,
            "monogenic_cause_found": monogenic_cause_found,
            "diagnosis_delay_years": diagnosis_delay_years,
        })
    return pts


def overview() -> dict:
    all_cohorts = [_generate_cohort(g) for g in IMMUNODEFICIENCY_GENES]
    total = sum(len(c) for c in all_cohorts)
    all_pts = [p for c in all_cohorts for p in c]

    ivig_n = sum(1 for p in all_pts if p["on_ivig"])
    hsct_n = sum(1 for p in all_pts if p["hsct_received"])
    gene_therapy_n = sum(1 for p in all_pts if p["gene_therapy_received"] or p["strimvelis_received"])
    autoimmune_n = sum(1 for p in all_pts if p["autoimmune_complication"])
    lymphoma_n = sum(1 for p in all_pts if p["lymphoma"])
    pcp_n = sum(1 for p in all_pts if p["pcp_pneumonia"])
    omenn_n = sum(1 for p in all_pts if p["omenn_syndrome_rag1"] or p["omenn_syndrome_rag2"])
    maternal_engraftment_n = sum(
        1 for p in all_pts if p["maternal_engraftment"] or p["jak3_maternal_engraftment"]
    )
    live_vaccine_error_n = sum(1 for p in all_pts if p["live_vaccine_error"])
    radiation_sensitive_n = sum(1 for p in all_pts if p["radiation_sensitive"])
    gene_counts = {g["gene"]: len(_generate_cohort(g)) for g in IMMUNODEFICIENCY_GENES}

    return {
        "atlas": "Hereditary-Primary-Immunodeficiency-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Primary Immunodeficiency Atlas",
        "genes": [g["gene"] for g in IMMUNODEFICIENCY_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "ivig_patients": ivig_n,
        "hsct_patients": hsct_n,
        "gene_therapy_patients": gene_therapy_n,
        "autoimmune_complication_patients": autoimmune_n,
        "lymphoma_patients": lymphoma_n,
        "pcp_pneumonia_patients": pcp_n,
        "omenn_syndrome_patients": omenn_n,
        "maternal_engraftment_patients": maternal_engraftment_n,
        "live_vaccine_error_patients": live_vaccine_error_n,
        "radiation_sensitive_patients": radiation_sensitive_n,
        "gene_patient_counts": gene_counts,
        "pathway": (
            "XLA (BTK): BTK kinase required for pre-B-cell receptor signalling -> B-cell development arrest at pro-B stage. "
            "SCID-X1 (IL2RG) and JAK3-SCID: common gamma chain -> JAK3 -> STAT5 signalling for IL-7 (T/NK development) and IL-15 (NK) -> "
            "T and NK cell developmental arrest; B cells present but non-functional. "
            "ADA-SCID: deoxyadenosine -> dATP toxic accumulation -> lymphocyte apoptosis across all lineages (T+B+NK). "
            "RAG1/RAG2-SCID: V(D)J recombinase absent -> no T-cell receptor or B-cell receptor formation -> T and B lymphopenia; NK unaffected. "
            "Artemis-SCID (DCLRE1C): Artemis endonuclease absent -> hairpin opening in V(D)J recombination fails -> DNA DSB accumulation -> T and B lymphopenia; radiation sensitivity. "
            "CVID (TNFRSF13B/TACI): B-cell maturation and isotype switching defect -> hypogammaglobulinaemia -> absent protective antibody."
        ),
        "key_clinical_insight": (
            "XLA (BTK): NEVER give live vaccines -- OPV causes vaccine-associated paralytic poliomyelitis (VAPP). "
            "SCID-X1 (IL2RG): exclude maternal T-cell engraftment before HSCT; gene therapy OTL-101 EMA-licensed. "
            "ADA-SCID: Strimvelis (EMA 2016) first licensed EU gene therapy; PEG-ADA bridges to definitive cure. "
            "RAG1/RAG2 Omenn: erythroderma + eosinophilia + elevated IgE + absent B cells = NOT atopy -- B-cell count is the diagnostic pivot. "
            "Artemis (DCLRE1C): standard myeloablative conditioning is lethal -- reduced-intensity mandatory. "
            "CVID: low IgG alone is insufficient for diagnosis -- absent vaccine responses are the required second criterion."
        ),
    }


def breakdown() -> dict:
    result = {}
    for gene_entry in IMMUNODEFICIENCY_GENES:
        cohort = _generate_cohort(gene_entry)
        gene = gene_entry["gene"]

        ivig_pct = round(100 * sum(1 for p in cohort if p["on_ivig"]) / len(cohort))
        hsct_pct = round(100 * sum(1 for p in cohort if p["hsct_received"]) / len(cohort))
        gene_therapy_pct = round(
            100 * sum(1 for p in cohort if p["gene_therapy_received"] or p["strimvelis_received"]) / len(cohort)
        )
        pcp_pct = round(100 * sum(1 for p in cohort if p["pcp_pneumonia"]) / len(cohort))
        maternal_engraftment_pct = round(
            100 * sum(1 for p in cohort if p["maternal_engraftment"] or p["jak3_maternal_engraftment"]) / len(cohort)
        )
        autoimmune_pct = round(100 * sum(1 for p in cohort if p["autoimmune_complication"]) / len(cohort))
        lymphoma_pct = round(100 * sum(1 for p in cohort if p["lymphoma"]) / len(cohort))
        omenn_pct = round(
            100 * sum(1 for p in cohort if p["omenn_syndrome_rag1"] or p["omenn_syndrome_rag2"]) / len(cohort)
        )
        bronchiectasis_pct = round(100 * sum(1 for p in cohort if p["bronchiectasis"]) / len(cohort))
        sinopulmonary_pct = round(100 * sum(1 for p in cohort if p["sinopulmonary_infections"]) / len(cohort))

        result[gene] = {
            "gene": gene,
            "alt_name": gene_entry["alt_name"],
            "locus": gene_entry["locus"],
            "protein_size": gene_entry["protein_size"],
            "inheritance": gene_entry["inheritance"],
            "n_patients": len(cohort),
            "ivig_pct": ivig_pct,
            "hsct_pct": hsct_pct,
            "gene_therapy_pct": gene_therapy_pct,
            "pcp_pneumonia_pct": pcp_pct,
            "maternal_engraftment_pct": maternal_engraftment_pct,
            "autoimmune_complication_pct": autoimmune_pct,
            "lymphoma_pct": lymphoma_pct,
            "omenn_syndrome_pct": omenn_pct,
            "bronchiectasis_pct": bronchiectasis_pct,
            "sinopulmonary_infections_pct": sinopulmonary_pct,
            "age_of_onset": gene_entry["age_of_onset"],
            "key_biomarker": gene_entry["key_biomarker"],
            "pathognomonic": gene_entry["pathognomonic"],
            "treatment": gene_entry["treatment"],
            "critical_flags": gene_entry["critical_flags"],
            "seed": gene_entry["seed"],
            "cohort_preview": cohort[:5],
        }
    return result


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Primary-Immunodeficiency-Atlas",
        "gene_definitions": {
            g["gene"]: {
                "protein": g["protein"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
            }
            for g in IMMUNODEFICIENCY_GENES
        },
        "glossary": {
            "X-linked Agammaglobulinaemia (XLA)": (
                "BTK-deficient primary immunodeficiency; male-only (X-linked); absent B cells; absent immunoglobulins; "
                "normal T and NK cells; onset 6-12 months when maternal IgG wanes; IVIG replacement lifelong; "
                "live vaccines absolutely contraindicated -- OPV causes VAPP"
            ),
            "SCID (Severe Combined Immunodeficiency)": (
                "Profound deficiency of T lymphocytes +/- B cells +/- NK cells; opportunistic infections in infancy; "
                "fatal without treatment (HSCT or gene therapy); TREC newborn screening detects presymptomatically; "
                "multiple genetic causes: IL2RG, JAK3, ADA, RAG1, RAG2, DCLRE1C, and others"
            ),
            "SCID-X1 (X-linked SCID)": (
                "IL2RG-deficient SCID; most common SCID type; T-B+NK- immunophenotype; males predominantly; "
                "gene therapy (OTL-101) first EU-licensed gene therapy for SCID (2021); maternal engraftment risk; "
                "irradiated CMV-negative blood products mandatory"
            ),
            "ADA-SCID (Adenosine Deaminase SCID)": (
                "ADA enzyme absent -> dATP accumulates -> T+B+NK all lymphopenic (pan-lymphopenia); "
                "first human gene therapy disease (1990); Strimvelis EMA approved 2016; PEG-ADA bridges to curative therapy; "
                "costochondral rib abnormalities distinctive among SCID types"
            ),
            "T-B-NK+ SCID": (
                "Immunophenotype: T cells absent, B cells absent, NK cells present (NK development independent of RAG/Artemis); "
                "caused by RAG1, RAG2, or Artemis (DCLRE1C) deficiency; "
                "V(D)J recombination failure common mechanism; "
                "Artemis additionally radiation-sensitive -- critical for conditioning selection"
            ),
            "T-B+NK- SCID": (
                "Immunophenotype: T cells absent, NK cells absent, B cells present but non-functional; "
                "caused by IL2RG (SCID-X1, X-linked) or JAK3 (AR); "
                "phenotypically identical -- molecular testing essential; "
                "JAK3 can affect females (AR); SCID-X1 only males"
            ),
            "Omenn Syndrome": (
                "Partial RAG1 or RAG2 deficiency with residual enzyme activity; "
                "oligoclonal autoreactive T cells escape thymus -> erythroderma, hepatosplenomegaly, eosinophilia, elevated IgE, absent B cells; "
                "NOT atopic dermatitis -- B cells are absent in Omenn, present in atopy; "
                "immunosuppression with cyclosporin mandatory before HSCT conditioning"
            ),
            "Artemis-SCID (DCLRE1C)": (
                "Radiation-sensitive SCID; Artemis endonuclease absent -> defective DNA DSB repair + V(D)J recombination failure; "
                "T-B-NK+ phenotype; standard myeloablative HSCT conditioning is lethal; "
                "Athabascan Native American founder deletion common; reduced-intensity conditioning mandatory"
            ),
            "JAK3-SCID": (
                "AR (not XL) SCID; T-B+NK- phenotype identical to SCID-X1 (IL2RG); "
                "JAK3 signals downstream of IL2RG common gamma chain; "
                "females affected (AR); test JAK3 in all females with T-B+NK- SCID -- cannot be SCID-X1"
            ),
            "CVID (Common Variable Immunodeficiency)": (
                "Most common symptomatic PID in adults; prevalence approximately 1:25,000; "
                "diagnostic triad: low IgG (<4 g/L) + absent vaccine responses + age >4 years; "
                "TNFRSF13B, NFKB1, CTLA4, LRBA monogenic causes in 10% of patients; "
                "lymphoma risk 8-fold elevated; autoimmune complications in 25%"
            ),
            "PEG-ADA (Pegylated Adenosine Deaminase)": (
                "Enzyme replacement therapy for ADA-SCID; weekly IM injection; "
                "reduces dATP toxicity; partial immune reconstitution; "
                "NOT curative -- bridges to gene therapy or HSCT; "
                "abrupt discontinuation causes rapid lymphocyte depletion relapse within days"
            ),
            "TREC (T-cell Receptor Excision Circles)": (
                "DNA circles generated during T-cell receptor gene rearrangement in thymus; "
                "absent in T-cell lymphopenia (SCID); measured on neonatal Guthrie bloodspot; "
                "newborn screening for SCID detects before symptoms; early HSCT (<3.5 months) achieves >90% survival"
            ),
            "Maternal T-cell Engraftment": (
                "Maternal lymphocytes cross placenta in all pregnancies; in SCID infants they survive as the baby cannot reject them; "
                "identified by HLA chimerism testing; if unrecognised before HSCT, maternal T cells cause severe GvHD; "
                "exclude in ALL SCID patients by HLA typing before conditioning -- non-negotiable safety step"
            ),
            "V(D)J Recombination": (
                "Process by which RAG1/RAG2 complex generates diverse T-cell receptors and B-cell receptors; "
                "RAG1 and RAG2 introduce DNA double-strand breaks at recombination signal sequences (RSS); "
                "Artemis opens hairpin intermediates (DNA-PKcs-dependent); loss of any component -> absent TCR/BCR -> T and B lymphopenia"
            ),
            "IVIG (Intravenous Immunoglobulin)": (
                "Polyclonal human IgG pooled from >1000 donors; 400-600 mg/kg every 3-4 weeks; "
                "treatment for XLA, CVID, all antibody deficiencies; trough IgG target >8-10 g/L; "
                "SCIG (subcutaneous) equivalent efficacy; home administration; lower anaphylaxis rate"
            ),
            "GLILD (Granulomatous-Lymphocytic Interstitial Lung Disease)": (
                "CVID pulmonary complication; non-caseating granulomas + lymphoid hyperplasia in lung parenchyma; "
                "CT: ground-glass opacification, nodules, hilar/mediastinal lymphadenopathy; BAL + biopsy diagnostic; "
                "treatment: prednisolone +/- rituximab +/- azathioprine; distinct from infection -- BAL culture negative"
            ),
            "Vaccine-Associated Paralytic Poliomyelitis (VAPP)": (
                "Rare but catastrophic complication of oral polio vaccine (OPV) in immunocompromised patients; "
                "XLA patients cannot clear live attenuated poliovirus -> poliovirus replicates -> paralytic polio; "
                "OPV absolutely contraindicated in XLA and all PID; use inactivated polio vaccine (IPV) in patients and household contacts"
            ),
            "Strimvelis": (
                "EMA-approved (2016) ex-vivo gammaretroviral gene therapy for ADA-SCID; "
                "autologous haematopoietic stem cells transduced with ADA cDNA; "
                "first licensed gene therapy for a primary immunodeficiency in Europe; "
                "curative; no GvHD risk (autologous); performed at specialised expert centres"
            ),
        },
        "clinical_pearls": [
            "XLA (BTK): NEVER give OPV or any live vaccine -- VAPP is fatal and entirely preventable; BTK protein staining on monocytes by flow cytometry is a rapid immunological screen before molecular results are available",
            "SCID-X1 (IL2RG): TREC newborn screening saves lives -- early HSCT (<3.5 months) achieves >90% survival; gene therapy OTL-101 is EMA-licensed for SCID-X1 when no HLA-matched sibling donor is available",
            "ADA-SCID: costochondral cupping on CXR in a SCID infant = ADA-SCID until proven; PEG-ADA must never be stopped abruptly; Strimvelis curative gene therapy (EMA 2016) is preferred over HSCT when no matched sibling is available",
            "RAG1 Omenn: absent B cells is the single most important feature that distinguishes Omenn from atopic eczema; elevated IgE in Omenn is NOT IgE-mediated allergy; immunosuppression before HSCT is mandatory -- transplant without it is high-risk",
            "RAG2 Omenn: IgE can exceed 10,000 IU/mL in RAG2 Omenn; always test RAG1 AND RAG2 together on a PID panel; RAG2-SCID is NOT radiation-sensitive unlike Artemis",
            "Artemis (DCLRE1C): radiation-sensitive SCID -- standard busulfan/cyclophosphamide myeloablative conditioning is lethal; reduced-intensity fludarabine-based conditioning is mandatory; exclude Artemis in all T-B-NK+ SCID before conditioning",
            "JAK3-SCID: phenotypically identical to SCID-X1 but AR -- FEMALES CAN AND DO DEVELOP JAK3-SCID; test JAK3 in any female with T-B+NK- SCID -- SCID-X1 is not possible in females",
            "CVID (TNFRSF13B): low IgG alone is insufficient for diagnosis -- document absent vaccine responses; 8-fold lymphoma risk requires annual clinical review; 10% have monogenic cause with targeted treatment options beyond standard IVIG",
            "ALL SCID types: irradiated CMV-negative blood is mandatory -- non-irradiated transfusion can cause fatal transfusion-associated GvHD; inform blood bank and theatre staff explicitly on every admission",
            "ALL SCID types: maternal T-cell engraftment must be excluded by HLA chimerism analysis before any HSCT conditioning -- maternal T cells cause severe GvHD if unrecognised",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str)[:2000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = breakdown()
    first_gene = list(bd.keys())[0]
    print(json.dumps(bd[first_gene], indent=2, default=str)[:1500])
    print("\n=== DEFINITIONS (glossary sample) ===")
    df = definitions()
    print(json.dumps(list(df["glossary"].items())[:4], indent=2))
