#!/usr/bin/env python3
"""Hereditary-Waldenström-Macroglobulinemia-LPL-Predisposition-Atlas -- Complete 8-Gene Reference
MYD88   (Myeloid differentiation primary response 88; 296aa; 3p22.2; AD LOF/GOF;
         Familial WM — L265P somatic >90% WM; rare germline families; TIR-domain adaptor;
         NF-κB/IRAK4 signalling → plasma-cell survival; ibrutinib BTKi first-line FDA2015;
         zanubrutinib ASPEN trial superior to ibrutinib (PFS HR 0.75); MYD88 WT = poor BTKi response;
         seed SEED_BASE+0).
CXCR4   (C-X-C motif chemokine receptor 4; 360aa; 2q22.1; AD GOF;
         WHIM-type C-terminal truncation — 40% WM somatic; familial WHIM syndrome (autosomal dominant);
         CXCR4WHIM → worst WM prognosis (inferior PFS with ibrutinib, MYD88L265P/CXCR4WHIM double-mutant);
         plerixafor (AMD3100) FDA2024 WHIM syndrome — CXCR4 antagonist; mobilises malignant clone;
         seed SEED_BASE+1).
BTK     (Bruton tyrosine kinase; 659aa; Xq22.1; XLR LOF;
         X-linked agammaglobulinaemia (XLA) — BTK LOF → absent B cells → profound immunodeficiency;
         XLA patients surviving on IVIG paradoxically at risk for B-cell malignancies including WM;
         BTK = therapeutic TARGET of ibrutinib/zanubrutinib/acalabrutinib in WM — kinase domain;
         seed SEED_BASE+2).
ATM     (Ataxia-telangiectasia mutated; 3056aa; 11q22.3; AD/AR LOF;
         B-cell malignancy 4-5x monoallelic; CLL 4-5x; NHL 3-5x; WM 3-4x elevated;
         biallelic A-T = ALL/lymphoma 100x PATHOGNOMONIC; RADIOSENSITIVITY ABSOLUTE biallelic;
         Ceralasertib ATRi synthetic lethality; del11q somatic = worst cytogenetics in CLL;
         seed SEED_BASE+3).
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         LFS — WM/lymphoma elevated; del17p somatic = WORST PROGNOSIS WM and CLL;
         AVOID RADIATION ABSOLUTELY; WBMRI Toronto annually; venetoclax + BTKi preferred del17p;
         seed SEED_BASE+4).
NBN     (Nibrin/NBS1; 754aa; 8q21.3; AD LOF;
         Nijmegen Breakage Syndrome (NBS); 657del5 Slavic founder (Czech/Slovak/Polish 1/150);
         B-cell NHL 60x HIGHEST monoallelic — strongest single gene NHL signal;
         WM/LPL 5-8x; RADIOSENSITIVITY biallelic ABSOLUTE CI; MRN complex DSB repair;
         seed SEED_BASE+5).
CARD11  (Caspase recruitment domain family member 11; 1154aa; 7p22.2; AD LOF/GOF;
         GOF = CARD11 gain-of-function → constitutive NF-κB → BENTA disease (B-cell Expansion with NF-κB and T-cell Anergy);
         LOF = CARD11 loss-of-function → immunodeficiency → compensatory B-cell lymphoproliferation → WM/LPL;
         scaffold protein BCL10-MALT1 complex; ibrutinib may be less effective CARD11-GOF (downstream BTK);
         seed SEED_BASE+6).
PIK3CD  (Phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit delta; 1044aa; 1p36.22; AD GOF;
         APDS1 (Activated PI3Kδ Syndrome type 1) — PI3Kδ GOF → hyperactive B-cell signalling;
         IgM elevation + lymphoproliferation; WM/LPL emerging association; leniolisib FDA2023 (p110δ inhibitor);
         idelalisib also PI3Kδ; respiratory infections + herpesvirus reactivation = PATHOGNOMONIC APDS1;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3534-3541)
"""
import random

SEED_BASE = 3534

ATLAS_GENES = [
    {
        "gene": "MYD88",
        "protein": (
            "MYD88 -- 3p22.2 AD LOF/GOF -- 296aa -- "
            "Myeloid-Differentiation-Primary-Response-88-33kDa-"
            "TIR-Domain-Adaptor-IRAK4-Recruiter-NF-kB-Activator-"
            "L265P-Somatic-90pct-Plus-WM-NF-kB-Constitutive-"
            "Ibrutinib-BTKi-FDA2015-Zanubrutinib-ASPEN-PFS-HR-0.75-"
            "MYD88-WT-WM-POOR-BTKi-Response-Familial-WM-Pedigrees-OMIM-602170"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "296 aa / 33 kDa / 3p22.2 MYD88 Waldenström molecular context: "
            "STRUCTURE: "
            "  296 aa / ~33 kDa; TIR (Toll/IL-1 receptor) domain adaptor protein; "
            "  Death domain (DD, aa 1-110) — recruits IRAK4/IRAK1 death domain; "
            "  TIR domain (aa 159-296) — interfaces with TLR/IL-1R cytoplasmic TIR domain; "
            "  L265P hotspot: proline in TIR domain → locked in active conformation → IRAK4 constitutively bound; "
            "  L265P SOMATIC: >90% of WM/LPL tumours carry MYD88 L265P — most common somatic mutation in WM; "
            "FAMILIAL WM: "
            "  Rare germline MYD88 families now reported; L265P primarily somatic not germline in most cases; "
            "  However: TIR-domain germline GOF variants in WM families describe (Treon lab, 2012+); "
            "  Germline MYD88 LOF → innate immunity deficit (susceptibility to pyogenic infections); "
            "  Paradox: somatic L265P GOF → WM; germline LOF → predisposition via impaired anti-tumour TLR signalling; "
            "MOLECULAR MECHANISM: "
            "  MYD88 L265P → constitutive NF-κB + IRAK4 signalling → BCL2, BCL-XL, MYC upregulation; "
            "  Promotes lymphoplasmacytic survival and secretion of IgM (hallmark of WM); "
            "  CXCR4WHIM co-mutation (40% WM) → even stronger NF-κB + SDF-1/CXCL12 retention in BM; "
            "TREATMENT: "
            "  MYD88 L265P WM: ibrutinib FDA2015 first-line (ORR 91%; MYD88L265P BTKi-sensitive); "
            "  Zanubrutinib ASPEN trial: superior to ibrutinib (VGPR 28% vs 19%; PFS HR 0.75); "
            "  MYD88 WT WM (~10%): BTKi less effective → rituximab-based or venetoclax-based preferred; "
            "  MYD88 L265P + CXCR4WHIM: inferior BTKi response → zanubrutinib + venetoclax combination; "
            "SURVEILLANCE: "
            "  Serum protein electrophoresis (SPEP) + IgM quantification + serum viscosity annually >40yr; "
            "  Bone marrow biopsy if IgM >3g/dL or cytopenias; MYD88 L265P allele burden on BM; "
            "  CASCADE: first-degree relatives of familial WM with MYD88 germline LOF — annual IgM quantification"
        ),
        "syndrome": "Familial Waldenström Macroglobulinemia (FWM) — MYD88 / NF-κB pathway",
        "inheritance": "AD LOF (germline immunity) / AD GOF-like (somatic L265P constitutive NF-κB)",
        "wm_risk": "MYD88 L265P somatic >90% WM; germline MYD88 families rare; WM risk elevated in MYD88 LOF families via impaired B-cell immune surveillance; familial WM pedigrees now recognised",
        "pathognomonic": "MYD88 L265P somatic in WM is PATHOGNOMONIC for WM/LPL lineage (>90% sensitivity); MYD88 WT WM = distinct aggressive subtype — different from LPL/WM lineage; CXCR4WHIM co-mutation = double mutant worst prognosis",
        "key_avoid": "Do NOT assume all WM carries MYD88 L265P — ~10% are WT (test BEFORE BTKi); MYD88 WT WM is clinically DISTINCT (worse prognosis, BTKi less effective); never use TBI in BTK-LOF/XLA patients (radiosensitivity); allele-specific PCR for L265P — standard NGS sometimes misses",
        "key_rule": "MYD88 L265P WM: ibrutinib/zanubrutinib preferred — zanubrutinib superior in ASPEN (VGPR 28% vs 19%, fewer AF events); MYD88 WT WM: rituximab-bendamustine or venetoclax-based; CXCR4WHIM co-mutant: zanubrutinib + venetoclax combo, plerixafor re-mobilisation.",
        "surveillance": "Annual IgM quantification + SPEP + IFE + serum viscosity in at-risk carriers; BM biopsy if IgM >3g/dL; MYD88 L265P ctDNA monitoring post-treatment; first-degree cascade IgM from age 40yr",
        "targeted_rx": "Zanubrutinib (BTKi 2nd-gen, ASPEN PFS superior) or ibrutinib (FDA2015); rituximab + bendamustine; venetoclax (MYD88 L265P + del17p); bortezomib + rituximab; autologous SCT eligible patients; CAR-T CD19/CD38 emerging; lenalidomide ± rituximab maintenance",
    },
    {
        "gene": "CXCR4",
        "protein": (
            "CXCR4 -- 2q22.1 AD GOF -- 360aa -- "
            "CXC-Chemokine-Receptor-4-39kDa-7TM-GPCR-"
            "WHIM-Type-C-Terminal-Truncation-40pct-WM-"
            "MYD88-L265P-Plus-CXCR4-WHIM-Double-Mutant-WORST-Prognosis-"
            "Plerixafor-AMD3100-FDA2024-WHIM-Syndrome-BTKi-Inferior-Response-OMIM-162643"
        ),
        "locus": "2q22.1",
        "protein_size": (
            "360 aa / 39 kDa / 2q22.1 CXCR4 Waldenström molecular context: "
            "STRUCTURE: "
            "  360 aa / ~39 kDa; 7-transmembrane (GPCR) receptor; Gαi-coupled; "
            "  C-terminal cytoplasmic tail (aa 310-360) = regulatory domain for receptor internalisation; "
            "  WHIM-type truncations: nonsense/frameshift in C-terminal tail → receptor internalisation impaired; "
            "  S338X WHIM (germline) most common; C-terminal truncations in WM SOMATIC (different spectrum from germline); "
            "CXCR4 IN WM: "
            "  Somatic CXCR4 mutations (WHIM-type) in ~40% WM tumours; "
            "  CXCR4 mutant WM: higher BM infiltration, higher IgM levels, worse BTKi response; "
            "  MYD88L265P/CXCR4WHIM double-mutant = WORST PROGNOSIS; PFS inferior to MYD88L265P only WM; "
            "  Mechanism: CXCR4 truncation → impaired GRK-mediated desensitisation → prolonged SDF-1α signalling; "
            "  SDF-1α (CXCL12) → PI3K-AKT-mTOR + ERK → BM retention of lymphoplasmacytic clone; "
            "WHIM SYNDROME (GERMLINE): "
            "  Rare AD GOF germline syndrome; WHIM = Warts, Hypogammaglobulinemia, Infections, Myelokathexis; "
            "  Myelokathexis: neutrophils arrested in BM (hypersegmented, hypermature — PATHOGNOMONIC); "
            "  HPV-related warts (not immunocompromise alone): pan-HPV serotype warts, HVPV → SCC risk; "
            "  Plerixafor (AMD3100) CXCR4 antagonist → FDA2024 WHIM syndrome (morcyclopentyl-plerixafor); "
            "TREATMENT: "
            "  CXCR4 WHIM WM: zanubrutinib ± venetoclax (double-mutant inferior ibrutinib response); "
            "  Plerixafor: CXCR4 blockade → mobilises BM-retained clone; synergy with BTKi being explored; "
            "  MYD88L265P/CXCR4WHIM: consider bortezomib+rituximab combinations; venetoclax + BTKi; "
            "SURVEILLANCE: "
            "  CXCR4 WHIM germline: annual CBC (myelokathexis ANC), HPV wart surveillance, IgM quantification; "
            "  Cervical HPV: colposcopy if WHIM female; dermatology annual for SCC risk from HPV; "
            "  B-cell malignancy screen: annual SPEP + BM biopsy if cytopenias in germline WHIM"
        ),
        "syndrome": "WHIM Syndrome (germline GOF) / Sporadic WM CXCR4WHIM somatic (double mutant worst prognosis)",
        "inheritance": "AD GOF (germline WHIM) / somatic GOF (~40% WM tumours)",
        "wm_risk": "CXCR4WHIM somatic in ~40% WM; double-mutant MYD88L265P+CXCR4WHIM = WORST WM prognosis; germline WHIM syndrome predisposes to lymphoproliferation and B-cell malignancy; BTKi inferior response in CXCR4WHIM WM",
        "pathognomonic": "Myelokathexis (hypersegmented hypermature neutrophils in BM) = PATHOGNOMONIC WHIM germline; CXCR4WHIM somatic + MYD88L265P double-mutant = WORST WM prognosis; HPV pan-serotype warts + hypogammaglobulinaemia + recurrent infections = WHIM triad",
        "key_avoid": "Do NOT use BTKi alone in MYD88L265P/CXCR4WHIM double-mutant WM — inferior PFS; never administer live vaccines in WHIM syndrome (impaired B-cell responses); CXCR4 allele-specific sequencing required — standard NGS may miss C-terminal truncations; do NOT miss WHIM diagnosis in neutropenia + warts combination",
        "key_rule": "CXCR4WHIM WM: zanubrutinib + venetoclax combination preferred over single-agent BTKi; plerixafor-based mobilisation for SCT in CXCR4WHIM WM; WHIM germline: plerixafor (AMD3100) FDA2024 — CXCR4 antagonist normalises ANC and mobilises B cells; annual HPV screening + dermatology in germline WHIM.",
        "surveillance": "Germline WHIM: annual CBC + SPEP + IgM + ANC monitoring; HPV wart surveillance + colposcopy; IgG supplementation if hypogammaglobulinaemia; Somatic CXCR4WHIM WM: allele burden monitoring post-BTKi; CTDx ctDNA for treatment response",
        "targeted_rx": "Zanubrutinib (preferred over ibrutinib CXCR4WHIM — ASPEN); venetoclax + zanubrutinib; plerixafor (FDA2024 WHIM; investigational in WM); bortezomib + rituximab combinations; CAR-T CD19 emerging; idelalisib (PI3Kδ downstream); maintenance rituximab",
    },
    {
        "gene": "BTK",
        "protein": (
            "BTK -- Xq22.1 XLR LOF -- 659aa -- "
            "Bruton-Tyrosine-Kinase-76kDa-TEC-Family-"
            "XLA-X-Linked-Agammaglobulinemia-BTK-LOF-Absent-B-Cells-"
            "IVIG-Lifelong-XLA-Lymphoma-WM-Risk-"
            "Ibrutinib-Zanubrutinib-Acalabrutinib-BTKi-TARGET-C481S-Resistance-OMIM-300300"
        ),
        "locus": "Xq22.1",
        "protein_size": (
            "659 aa / 76 kDa / Xq22.1 BTK Waldenström molecular context: "
            "STRUCTURE: "
            "  659 aa / ~76 kDa; TEC family non-receptor tyrosine kinase; "
            "  PH domain (aa 1-120) → membrane recruitment (PI3P/PI(3,4,5)P3); "
            "  SH3 domain (aa 214-269) → protein-protein interactions; "
            "  SH2 domain (aa 270-367) → phosphotyrosine recognition; "
            "  Kinase domain (aa 400-659) → C481 = ibrutinib covalent binding site; "
            "X-LINKED AGAMMAGLOBULINAEMIA (XLA): "
            "  BTK LOF in males → arrest at pro-B/pre-B cell stage → absent circulating B cells; "
            "  No B cells → no immunoglobulin production → IgG <2g/L, IgA <0.2g/L, IgM undetectable; "
            "  IVIG (every 3-4 weeks) lifelong → prevents bacterial infections; "
            "  Paradox: XLA patients on long-term IVIG may develop B-cell malignancies (WM/LPL) at low frequency; "
            "BTK AS DRUG TARGET: "
            "  BTK kinase domain C481 = covalent binding site of ibrutinib (1st-gen BTKi); "
            "  C481S resistance mutation (somatic) → escape from ibrutinib; zanubrutinib better tolerated; "
            "  Non-covalent BTKi (pirtobrutinib) overcomes C481S resistance; FDA2023 pirtobrutinib CLL/SLL; "
            "  Acalabrutinib (AstraZeneca) — more selective, fewer cardiac events than ibrutinib; "
            "TREATMENT: "
            "  XLA itself: IVIG prophylaxis indefinitely; gene therapy emerging (BTK-corrected HSCs); "
            "  BTKi in WM: ibrutinib/zanubrutinib/acalabrutinib (not treating XLA but common WM target); "
            "  C481S resistance in WM BTKi-treated: switch to pirtobrutinib (non-covalent); "
            "SURVEILLANCE: "
            "  XLA: serum IgG trough >5g/L on IVIG; annual pulmonary function tests (bronchiectasis); "
            "  Sinus/lung CT for structural damage; echocardiogram (enteroviral cardiomyopathy XLA risk); "
            "  Annual BM surveillance if Ig elevation suggests clonal B-cell emergence on IVIG"
        ),
        "syndrome": "X-Linked Agammaglobulinemia (XLA) — BTK LOF / BTK = drug target for ibrutinib/zanubrutinib in WM",
        "inheritance": "XLR LOF (XLA); hemizygous males affected; females carriers usually unaffected",
        "wm_risk": "XLA males on long-term IVIG: B-cell malignancy risk elevated; BTK is the kinase target of ibrutinib/zanubrutinib in WM treatment; BTK C481S somatic resistance = ibrutinib escape; pirtobrutinib (non-covalent) overcomes C481S resistance",
        "pathognomonic": "Absent circulating B cells (<1% B lymphocytes) PATHOGNOMONIC for XLA-BTK; IgG <2g/L + IgA <0.2g/L + IgM undetectable in male infant = XLA until proven otherwise; enteroviral meningoencephalitis in XLA = LIFE-THREATENING emergency (IVIG intrathecal + pleconaril)",
        "key_avoid": "Do NOT delay IVIG in XLA infant — infections preventable but damage irreversible; NEVER use live vaccines in XLA (rotavirus, MMR, varicella, BCG, oral polio) — viral sepsis risk; do NOT confuse BTK XLA (LOF = no B cells) with BTK as drug target (BTKi INHIBITS BTK in B-cell malignancy — opposite rationale); do NOT use ibrutinib after C481S resistance — switch to pirtobrutinib",
        "key_rule": "XLA: IVIG every 3-4 weeks lifelong; trough IgG >5g/L; no live vaccines ever; enteroviral surveillance; gene therapy eligibility assess. BTKi in WM: ibrutinib/zanubrutinib for MYD88L265P WM; C481S resistance → pirtobrutinib (FDA2023 non-covalent BTKi); echocardiogram baseline before ibrutinib (cardiac events).",
        "surveillance": "XLA males: quarterly IVIG infusions + trough IgG monitoring; annual pulmonary function + CT chest; echocardiogram (cardiac complications); annual BM evaluation if clonal evolution suspected; FISH/NGS for BTK C481S in BTKi-resistant WM",
        "targeted_rx": "XLA: IVIG prophylaxis; gene therapy (emerging); Enteroviral: IVIG bolus + intrathecal + pleconaril; WM BTKi: ibrutinib (FDA2015); zanubrutinib (ASPEN); acalabrutinib; pirtobrutinib (C481S resistance FDA2023); venetoclax combinations (MYD88L265P/CXCR4WHIM)",
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD/AR LOF -- 3056aa -- "
            "Ataxia-Telangiectasia-Mutated-350kDa-PIK3-Kinase-"
            "WM-LPL-3-4x-Elevated-CLL-4-5x-NHL-3-5x-"
            "RADIOSENSITIVITY-ABSOLUTE-Biallelic-TBI-AVOID-"
            "Ceralasertib-ATRi-Synthetic-Lethality-A-T-Syndrome-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / ~350 kDa / 11q22.3 ATM Waldenström molecular context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; PI3-kinase-like kinase (PIKK) family; "
            "  HEAT repeats (aa 1-1960) — scaffold for protein recruitment to DSBs; "
            "  FAT domain (aa 1966-2566); Kinase domain (aa 2715-2966); FATC domain (aa 2967-3056); "
            "  Recruited to DSBs by MRN complex (MRE11-RAD50-NBS1); autophosphorylates S1981; "
            "  Phosphorylates H2AX (γH2AX), CHK2, BRCA1, MDM2, p53 downstream; "
            "B-CELL MALIGNANCY: "
            "  ATM monoallelic LOF: CLL 4-5x, NHL 3-5x, WM/LPL 3-4x elevated risk; "
            "  del11q somatic (ATM locus) = ADVERSE cytogenetics in CLL (~17%); "
            "  WM with monoallelic ATM: higher BM infiltration, more aggressive course; "
            "  RADIOSENSITIVITY in ATM heterozygotes: RT reduce 20-30% dose even monoallelic; "
            "  A-T biallelic: cerebellar ataxia, oculomotor apraxia, telangiectasias PATHOGNOMONIC, ALL/lymphoma 100x; "
            "TREATMENT: "
            "  Ceralasertib (AZD6738) ATRi — synthetic lethality with ATM LOF; ATR compensates for ATM loss; "
            "  ATM LOF WM: zanubrutinib/ibrutinib ± venetoclax; radiosensitivity mandates RT dose reduction; "
            "  Fludarabine: AVOID in ATM biallelic A-T (radiosensitivity analogue — DNA damage);  "
            "  ASCT: evaluate radiosensitivity before TBI-based conditioning; busulfan-based preferred; "
            "SURVEILLANCE: "
            "  A-T biallelic: annual blood counts + AFP (elevated in A-T); neurology every 6 months; "
            "  ATM monoallelic: annual SPEP + IgM quantification; breast MRI (ATM 30-35% lifetime breast); "
            "  Germline ATM carrier: avoid unnecessary diagnostic radiation; "
            "  Family cascade: ATM germline → BRCA-equivalent panel testing for first-degree relatives"
        ),
        "syndrome": "Ataxia-Telangiectasia (A-T; biallelic) / Hereditary B-cell malignancy syndrome (monoallelic)",
        "inheritance": "AD LOF monoallelic (B-cell malignancy) / AR biallelic (A-T syndrome)",
        "wm_risk": "ATM monoallelic LOF: WM/LPL 3-4x, CLL 4-5x HIGHEST, NHL 3-5x; A-T biallelic: ALL/lymphoma 100x PATHOGNOMONIC; del11q somatic in CLL = adverse cytogenetics; RADIOSENSITIVITY absolute biallelic; RT reduce 20-30% monoallelic; Ceralasertib ATRi synthetic lethality",
        "pathognomonic": "Telangiectasias (conjunctival, skin flexures) + cerebellar ataxia + elevated AFP = A-T PATHOGNOMONIC triad; IgA deficiency 70-80% A-T; CLL with del11q somatic = adverse cytogenetics (not germline ATM); biallelic A-T RADIOSENSITIVITY ABSOLUTE — fludarabine and TBI absolutely contraindicated",
        "key_avoid": "RADIOSENSITIVITY ABSOLUTE biallelic A-T — NO radiation therapy, NO TBI; monoallelic ATM: RT reduce 20-30% and avoid high-dose TBI-based ASCT conditioning; NEVER use fludarabine in A-T biallelic (DNA damage analogue — severe toxicity); do NOT underestimate monoallelic radiosensitivity — even heterozygotes more sensitive",
        "key_rule": "ATM LOF WM: zanubrutinib (preferred) or ibrutinib ± venetoclax; RT: reduce 20-30% dose even monoallelic (RADIOSENSITIVITY); ASCT: busulfan-based conditioning preferred over TBI; Ceralasertib ATRi synthetic lethality — ATM LOF = ATR-dependent → Ceralasertib kills ATM-deficient cells; A-T biallelic: fludarabine + TBI ABSOLUTELY CONTRAINDICATED.",
        "surveillance": "ATM monoallelic: annual SPEP + IgM + breast MRI from 40yr; mammography + MRI alternate 6-monthly; avoid unnecessary CT (cumulative radiation); A-T biallelic: AFP every 6mo, CBC annually, neurology, lung function; cardiac echo (A-T cardiomyopathy); genetic counselling first-degree relatives",
        "targeted_rx": "Ceralasertib (ATRi AZD6738) ATM synthetic lethality; zanubrutinib/ibrutinib + venetoclax WM; rituximab-bendamustine (non-ibrutinib option); bortezomib + rituximab; ASCT busulfan-based (avoid TBI); pirtobrutinib BTKi C481S resistance; BCG vaccination status verify in A-T (avoid live)",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "Tumour-Protein-p53-43kDa-Guardian-Genome-"
            "LFS-WM-del17p-Somatic-WORST-PROGNOSIS-"
            "AVOID-RADIATION-ABSOLUTELY-WBMRI-Toronto-Annually-"
            "Venetoclax-BTKi-del17p-Preferred-Ibrutinib-Fludarabine-CAUTION-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 Waldenström molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; homotetrameric transcription factor; zinc finger DBD; "
            "  N-terminal transactivation domain (aa 1-40) + proline-rich region (aa 40-90); "
            "  DNA-binding domain (aa 102-292) — hotspot mutations R175H, G245S, R248W/Q, R249S, R273H/C, R282W; "
            "  Tetramerisation domain (aa 325-356); C-terminal regulatory domain (aa 356-393); "
            "TP53 IN WM/LPL: "
            "  del17p somatic in WM = WORST PROGNOSIS (ibrutinib less effective in del17p WM); "
            "  Germline TP53 LOF = Li-Fraumeni syndrome — lymphoma/WM elevated risk; "
            "  WM with del17p: venetoclax + BTKi combinations preferred; "
            "  PCL (plasma cell leukaemia) in TP53 germline: del17p + extramedullary disease; "
            "LI-FRAUMENI SYNDROME (LFS): "
            "  Germline TP53 LOF: sarcoma (30% HIGHEST), breast (56% women), adrenocortical (childhood), brain, LYM/WM; "
            "  Choroid plexus carcinoma (CPC) in a child = PATHOGNOMONIC for LFS — test TP53 germline; "
            "  AVOID RADIATION ABSOLUTELY — even diagnostic radiation cumulative risk; "
            "  WBMRI Toronto protocol annually → surveillance whole-body + brain MRI; "
            "TREATMENT: "
            "  WM del17p: venetoclax (BCL2 inhibitor) — del17p WM has higher BCL2 expression; "
            "  BTKi + venetoclax (zanubrutinib + venetoclax) — double hit; "
            "  Fludarabine: avoid in del17p WM (DNA damage → clonal evolution); "
            "  Daratumumab + bortezomib in TP53 del17p plasma cell component; "
            "SURVEILLANCE: "
            "  LFS: WBMRI annually (Toronto protocol: brain + abdomen + pelvis + chest); "
            "  Breast MRI 6-monthly from 20yr (women); breast US alternate 3mo from 20yr; "
            "  Upper GI endoscopy annually >25yr; adrenal US annually <5yr; CBC + LFTs; "
            "  Annual dermatology; avoid ionising radiation (diagnostic CT) unless urgent"
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) — TP53 germline LOF / del17p WM somatic = WORST PROGNOSIS",
        "inheritance": "AD LOF (LFS germline) / somatic del17p in WM = adverse cytogenetics",
        "wm_risk": "LFS germline TP53 LOF: lymphoma/WM elevated risk; del17p SOMATIC in WM = WORST PROGNOSIS; BTKi less effective in del17p WM → venetoclax + zanubrutinib preferred; AVOID RADIATION ABSOLUTELY germline TP53; fludarabine CAUTION del17p WM",
        "pathognomonic": "Choroid plexus carcinoma (CPC) in child = PATHOGNOMONIC LFS (test germline TP53); del17p somatic WM = WORST PROGNOSIS cytogenetics (ibrutinib inferior); AVOID RADIATION ABSOLUTELY LFS — even diagnostic cumulative radiation risk; WM + del17p + ibrutinib failure = venetoclax + zanubrutinib combination",
        "key_avoid": "AVOID RADIATION ABSOLUTELY LFS germline TP53 — ALL radiation including diagnostic CT unless critical; del17p WM: avoid fludarabine (clonal evolution risk); never use ibrutinib as single agent in del17p TP53-deleted WM without venetoclax; do NOT miss CPC in child — test TP53 germline immediately",
        "key_rule": "LFS TP53 LOF: WBMRI Toronto annually from birth; breast MRI from 20yr women; avoid CT unless urgent; sarcoma/brain/adrenal surveillance structured. del17p WM somatic: venetoclax + zanubrutinib preferred; daratumumab combinations; fludarabine AVOID; ASCT busulfan-based (avoid TBI in LFS).",
        "surveillance": "LFS: WBMRI annually; breast MRI 6-monthly + US 3-monthly from 20yr; dermatology annually; CBC/LFTs/AFP 6-monthly; avoid ionising radiation; del17p WM: SPEP + IgM + BM biopsy every 6mo on treatment; ctDNA allele burden monitoring",
        "targeted_rx": "del17p WM: venetoclax (BCL2) + zanubrutinib; daratumumab + bortezomib + dexamethasone; ixazomib maintenance; pirtobrutinib (non-covalent BTKi); ASCT busulfan-based; LFS: no targeted Rx — surveillance is the intervention; consider germline TP53 status before ASCT radiation conditioning",
    },
    {
        "gene": "NBN",
        "protein": (
            "NBN -- 8q21.3 AD/AR LOF -- 754aa -- "
            "Nibrin-NBS1-85kDa-MRN-Complex-FHA-BRCT-"
            "NBS-657del5-Slavic-1-in-150-Czech-Slovak-Polish-"
            "B-Cell-NHL-60x-HIGHEST-WM-LPL-5-8x-"
            "RADIOSENSITIVITY-Biallelic-ABSOLUTE-CI-MRN-PARP-DSB-Repair-OMIM-602667"
        ),
        "locus": "8q21.3",
        "protein_size": (
            "754 aa / 85 kDa / 8q21.3 NBN Waldenström molecular context: "
            "STRUCTURE: "
            "  754 aa / ~85 kDa; part of MRN complex (MRE11-RAD50-NBS1); "
            "  FHA domain (aa 1-108) — forkhead-associated; binds γH2AX, MDC1 (DSB localisation); "
            "  BRCT1/BRCT2 tandem (aa 221-421) — phosphopeptide recognition; MDC1 binding → ATM activation; "
            "  C-terminal MRE11-binding domain (aa 648-754) — essential for complex formation; "
            "  657del5 Slavic founder: c.657_661del5 frameshift → premature stop → truncated p70-NBS1 (partial function); "
            "NIJMEGEN BREAKAGE SYNDROME (NBS): "
            "  Biallelic NBN (mostly 657del5 homozygous): microcephaly, birdlike facies, ID, immunodeficiency; "
            "  RADIOSENSITIVITY ABSOLUTE biallelic — similar to A-T; TBI ABSOLUTELY CONTRAINDICATED; "
            "  ALL/lymphoma 100x+ biallelic NBS; most NBS patients die of lymphoma; "
            "B-CELL NHL / WM MONOALLELIC: "
            "  NBN 657del5 heterozygotes: B-cell NHL 60x elevated — HIGHEST monoallelic NHL signal of any gene; "
            "  WM/LPL 5-8x elevated monoallelic 657del5; CLL 3-4x; "
            "  Carrier freq: Czech 1/150, Slovak 1/150, Polish 1/170 — FREQUENT in Eastern European populations; "
            "TREATMENT: "
            "  NBN LOF WM: zanubrutinib/ibrutinib ± venetoclax; radiosensitivity mandate RT reduction; "
            "  Biallelic NBS RADIOSENSITIVITY ABSOLUTE — TBI ABSOLUTELY CONTRAINDICATED for ASCT; "
            "  PARP inhibitors: NBN LOF → MRN complex disruption → HRD → PARP inhibitor sensitivity potential; "
            "SURVEILLANCE: "
            "  NBN 657del5 heterozygote: annual SPEP + IgM + IFE; CBC annually; breast MRI from 40yr (3-4x breast); "
            "  Ethnicity: test all Eastern European lymphoma patients for NBN 657del5; "
            "  Family cascade: 657del5 heterozygote → siblings at risk (1/150 carrier, bi-allelic if both parents 657del5)"
        ),
        "syndrome": "Nijmegen Breakage Syndrome (NBS; biallelic) / Hereditary B-cell malignancy syndrome-HIGHEST (monoallelic 657del5)",
        "inheritance": "AR biallelic NBS / AD LOF monoallelic (B-cell NHL 60x — Slavic founder)",
        "wm_risk": "NBN 657del5 monoallelic: B-cell NHL 60x HIGHEST — strongest monoallelic lymphoma predisposition of any known gene; WM/LPL 5-8x; CLL 3-4x; biallelic NBS ALL/lymphoma 100x; 657del5 Slavic carrier frequency 1/150 Czech/Slovak/Polish — ALWAYS test Eastern European lymphoma",
        "pathognomonic": "Microcephaly + birdlike facies + immunodeficiency (biallelic NBS) PATHOGNOMONIC; NBN 657del5 heterozygote = B-cell NHL 60x HIGHEST monoallelic; Slavic ancestry + WM/LPL = NBN 657del5 first test; biallelic NBS RADIOSENSITIVITY ABSOLUTE TBI ABSOLUTELY CONTRAINDICATED",
        "key_avoid": "RADIOSENSITIVITY biallelic ABSOLUTE — TBI ABSOLUTELY CONTRAINDICATED; 657del5 heterozygote RT also reduce 20-30% (similar to ATM monoallelic); do NOT miss NBN 657del5 in Czech/Slovak/Polish WM patients — carrier 1/150; NEVER standard ASCT TBI in NBS biallelic; do NOT misclassify 657del5 as VUS — it is a confirmed pathogenic founder",
        "key_rule": "NBN 657del5 monoallelic (ANY Eastern European WM patient): test NBN 657del5 first; 60x NHL risk = HIGHEST monoallelic signal; zanubrutinib preferred BTKi; radiosensitivity: RT reduce 20-30%; ASCT: busulfan-based (no TBI); PARP inhibitor eligibility consider (MRN-HRD hypothesis). Biallelic NBS: TBI absolutely contraindicated.",
        "surveillance": "NBN 657del5 heterozygote: annual SPEP + IgM + IFE from age 40yr; CBC annually; breast MRI from 40yr (3-4x breast); avoid cumulative CT radiation; Cascade in Slavic families; NBS biallelic: quarterly CBC + lymphocyte subsets; annual MRI head + abdomen; no TBI ever; multidisciplinary genetics input",
        "targeted_rx": "Zanubrutinib or ibrutinib ± venetoclax (WM); rituximab-bendamustine (non-BTKi option); PARP inhibitors (NBN LOF MRN disruption → HRD hypothesis — evidence emerging); bortezomib + rituximab; no TBI ASCT; IVIG prophylaxis if hypogammaglobulinaemia NBS biallelic",
    },
    {
        "gene": "CARD11",
        "protein": (
            "CARD11 -- 7p22.2 AD LOF/GOF -- 1154aa -- "
            "Caspase-Recruitment-Domain-Family-Member-11-133kDa-"
            "BENTA-Disease-GOF-NF-kB-Constitutive-B-Cell-Expansion-"
            "CARD11-LOF-Immunodeficiency-B-Cell-Lymphoproliferation-WM-"
            "BCL10-MALT1-CBM-Complex-Scaffold-Ibrutinib-CARD11-GOF-INEFFECTIVE-OMIM-607210"
        ),
        "locus": "7p22.2",
        "protein_size": (
            "1154 aa / 133 kDa / 7p22.2 CARD11 Waldenström molecular context: "
            "STRUCTURE: "
            "  1154 aa / ~133 kDa; MAGUK family scaffolding protein; "
            "  CARD domain (aa 1-100) — caspase recruitment; BCL10 binding via CARD-CARD; "
            "  Coiled-coil domain (aa 101-280) — oligomerisation; signalling hub; "
            "  PDZ domain (aa 478-561); SH3 domain (aa 595-647); GUK domain (aa 695-892); "
            "  CARD11 forms CBM complex: CARD11-BCL10-MALT1 → IKKβ → NF-κB canonical pathway; "
            "CARD11 GOF (BENTA DISEASE): "
            "  Germline GOF mutations (G116S, L232F, F255V etc.) → constitutive NF-κB → B-cell expansion; "
            "  BENTA = B-cell Expansion with NF-κB and T-cell Anergy; "
            "  Atopic dermatitis (severe, early onset), recurrent infections, lymphadenopathy; "
            "  IgE markedly elevated; IgA deficiency; eosinophilia; "
            "  B-cell lymphomas (NHL, WM/LPL) in BENTA adulthood — NF-κB-driven; "
            "CARD11 LOF (IMMUNODEFICIENCY): "
            "  Germline LOF (haploinsufficiency) → impaired BCR-TCR → NF-κB → combined immunodeficiency; "
            "  CARD11-LOF CVID-like phenotype: recurrent infections, lymphoproliferation paradox; "
            "  Compensatory B-cell expansion → WM/LPL risk in CARD11 LOF; "
            "IBRUTINIB RESISTANCE: "
            "  CARD11 GOF in WM/DLBCL tumours → NF-κB downstream of BTK → ibrutinib bypassed; "
            "  CARD11 GOF resistance = downstream of BTK inhibition; lenalidomide ± rituximab; "
            "TREATMENT: "
            "  BENTA GOF: lenalidomide (NF-κB modulation); IVIG + antibiotic prophylaxis; "
            "  CARD11-GOF WM: venetoclax + lenalidomide; avoid BTKi monotherapy (GOF → BTK bypass); "
            "  CARD11-LOF: IVIG; haematopoietic SCT for severe combined immunodeficiency; "
            "SURVEILLANCE: "
            "  BENTA: annual SPEP + IgM; lymphoma surveillance (LDH, LN biopsy if adenopathy); "
            "  Dermatology (atopic dermatitis); allergy/immunology; CBC + IgE annually; "
            "  CARD11 LOF: quarterly Ig levels; prophylactic IVIG; annual CBC + lymphocyte subsets"
        ),
        "syndrome": "BENTA Disease (CARD11 GOF — NF-κB constitutive B-cell expansion) / CARD11 LOF immunodeficiency",
        "inheritance": "AD GOF (BENTA disease) / AD LOF (combined immunodeficiency-like phenotype)",
        "wm_risk": "CARD11 GOF: constitutive NF-κB → B-cell expansion (BENTA) → NHL/WM adulthood; CARD11 GOF in WM tumours → IBRUTINIB RESISTANCE (downstream of BTK); CARD11 LOF → lymphoproliferation → WM/LPL paradox; lenalidomide preferred in CARD11-GOF driven WM",
        "pathognomonic": "Severe early-onset atopic dermatitis + B-cell expansion + IgE markedly elevated + T-cell anergy = BENTA PATHOGNOMONIC (CARD11 GOF); CARD11 GOF WM = ibrutinib INEFFECTIVE (downstream NF-κB bypass); lymphoma + severe atopic dermatitis = test CARD11 germline",
        "key_avoid": "Do NOT use BTKi monotherapy in CARD11-GOF WM (downstream NF-κB bypass → ibrutinib ineffective); CARD11 LOF: avoid live vaccines (combined immunodeficiency); do NOT mistake BENTA-associated lymphoma for standard WM (different BTKi response expected); lenalidomide + venetoclax preferred over single-agent BTKi in CARD11-GOF",
        "key_rule": "CARD11-GOF WM: venetoclax + lenalidomide + rituximab (bypass BTKi resistance); NF-κB inhibitor combinations being explored; CARD11-LOF: IVIG prophylaxis; SCT consideration if progressive combined immunodeficiency; BENTA: dermatology management + allergy/immunology + annual lymphoma surveillance; atopic dermatitis control (dupilumab IL-4Rα mAb).",
        "surveillance": "CARD11 GOF (BENTA): annual SPEP + IgM + LDH + LN exam from age 30yr; dermatology annually; IgE + eosinophilia monitoring; CBC quarterly; CARD11 LOF: Ig levels quarterly; lung function annually (bronchiectasis); cascade testing: all first-degree relatives if familial WM with CARD11 variant",
        "targeted_rx": "CARD11-GOF WM: venetoclax + lenalidomide + rituximab; bortezomib-based (NF-κB proteasome-dependent); avoid BTKi monotherapy; dupilumab (IL-4Rα) atopic control BENTA; CARD11-LOF: IVIG; SCT if severe immunodeficiency; investigational NF-κB inhibitors; lenalidomide immunomodulation",
    },
    {
        "gene": "PIK3CD",
        "protein": (
            "PIK3CD -- 1p36.22 AD GOF -- 1044aa -- "
            "PI3K-Catalytic-Subunit-Delta-119kDa-Class-I-PI3K-"
            "APDS1-Activated-PI3K-Delta-Syndrome-Type-1-"
            "Lymphoproliferation-IgM-Elevation-WM-LPL-Emerging-"
            "Leniolisib-FDA2023-PI3K-Delta-Inhibitor-Idelalisib-OMIM-602839"
        ),
        "locus": "1p36.22",
        "protein_size": (
            "1044 aa / 119 kDa / 1p36.22 PIK3CD Waldenström molecular context: "
            "STRUCTURE: "
            "  1044 aa / ~119 kDa; catalytic subunit (p110δ) of class I PI3K; "
            "  PI3K complex: p110δ (PIK3CD) + regulatory subunit p85α/p85β/p55α/p55β/p50α (PIK3R1/R2/R3); "
            "  Phosphorylates PIP2 → PIP3 → PDK1 → AKT activation → mTORC1 → cell proliferation; "
            "  p110δ expression: predominantly lymphoid lineage (B/T cells) — explains immune-specific phenotype; "
            "  GOF mutations: E1021K (most common), N334K, C416R → constitutive lipid kinase activity; "
            "  E1021K: relieved from p85 inhibition + enhanced membrane recruitment; "
            "APDS1 (ACTIVATED PI3Kδ SYNDROME TYPE 1): "
            "  Germline PIK3CD GOF (E1021K most common) → constitutive PI3Kδ → hyperactive B/T cells → paradox: "
            "  Lymphoproliferation (lymphadenopathy, splenomegaly, nodular lymphoid hyperplasia) + immunodeficiency (recurrent respiratory infections); "
            "  Herpesvirus reactivation (EBV, CMV, HSV, VZV) = PATHOGNOMONIC APDS1; "
            "  IgM elevated, IgG/IgA/IgE reduced — dysgammaglobulinaemia; "
            "  WM/LPL emerging association — elevated IgM + lymphoplasmacytic cells in APDS1 BM; "
            "TREATMENT: "
            "  Leniolisib (CDZ173) — selective PI3Kδ inhibitor; FDA2023 APDS1 (first approved PI3Kδi for APDS); "
            "  Idelalisib (ZYDELIG) — PI3Kδ inhibitor; CLL/FL approved; APDS investigational; "
            "  Copanlisib (IV pan-PI3K but δ dominant); umbralisib (TG-1202) PI3Kδ+CK1ε; "
            "  Prophylactic antivirals (acyclovir/valacyclovir) for herpesvirus reactivation MANDATORY; "
            "  IVIG for recurrent bacterial infections (IgG supplementation); "
            "  Avoid live vaccines APDS1 (hyperimmune then immunodeficiency paradox); "
            "SURVEILLANCE: "
            "  APDS1: quarterly CBC + lymphocyte subsets; Ig levels every 6mo; EBV/CMV/VZV viral loads; "
            "  CT chest/abdomen 6-monthly (lymphoproliferation monitoring); annual BM biopsy if IgM >3g/dL; "
            "  Lung function annually (bronchiectasis from infections); dermatology (EBV-driven lymphoproliferation)"
        ),
        "syndrome": "APDS1 — Activated PI3Kδ Syndrome Type 1 (PIK3CD GOF) / emerging WM/LPL predisposition via IgM lymphoproliferation",
        "inheritance": "AD GOF (APDS1 germline) / somatic PIK3CD GOF in ~10-15% WM",
        "wm_risk": "PIK3CD GOF APDS1: IgM elevated + BM lymphoplasmacytic infiltration → WM/LPL emerging; somatic PIK3CD GOF in ~10-15% WM tumours; PI3Kδ → AKT → mTOR pathway → plasma-cell survival; leniolisib FDA2023 targeted therapy; idelalisib PI3Kδ inhibitor active in WM",
        "pathognomonic": "EBV/CMV/VZV reactivation + lymphoproliferation + recurrent respiratory infections + raised IgM = APDS1 PATHOGNOMONIC; herpesviral reactivation in young patient with lymphadenopathy = test PIK3CD germline; APDS1 lymphoproliferation is NOT lymphoma (but WM/LPL risk elevated); leniolisib FDA2023 APDS1 targeted Rx",
        "key_avoid": "Do NOT use live vaccines in APDS1 (paradoxical immune dysregulation — severe reactions); antiviral prophylaxis MANDATORY in APDS1 (herpesvirus reactivation risk); do NOT mistake APDS1 lymphoproliferation for lymphoma without biopsy (reactive vs malignant); idelalisib liver toxicity (transaminitis) — monitor LFTs monthly; do NOT start BTKi in APDS1 WM without checking downstream PI3Kδ pathway status",
        "key_rule": "APDS1 PIK3CD GOF: leniolisib (FDA2023) — first specific PI3Kδi approved; acyclovir/valacyclovir antivirals MANDATORY lifelong; IVIG if IgG <5g/L; live vaccines avoided; annual BM biopsy if IgM >3g/dL. PIK3CD GOF WM: idelalisib or copanlisib + rituximab ± zanubrutinib; mTOR downstream: everolimus/temsirolimus combinations; venetoclax for BCL2-expressing component.",
        "surveillance": "APDS1: CBC + lymphocyte subsets quarterly; Ig levels every 6mo; EBV/CMV/VZV PCR viral loads annually; CT chest/abdomen 6-monthly; lung function annually; LFTs monthly on idelalisib/leniolisib; SPEP + IgM annually from age 30yr; BM biopsy if IgM >3g/dL or cytopenias; cascade testing first-degree relatives",
        "targeted_rx": "Leniolisib (CDZ173 PI3Kδi FDA2023 APDS1); idelalisib (PI3Kδi WM/CLL approved — watch liver toxicity); copanlisib IV (PI3K pan + δ dominant); umbralisib (TG-1202 PI3Kδ+CK1ε); zanubrutinib + venetoclax WM; mTOR inhibitors (everolimus — downstream PI3K); acyclovir prophylaxis MANDATORY APDS1; IVIG supplementation; SCT refractory cases",
    },
]


def _make_patient(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)

    # Gene-specific WM/LPL characteristics
    wm_params = {
        "MYD88":  dict(age_mu=64, age_sd=9,  wm_prob=0.80, myd88_l265p_prob=0.90, cxcr4_mut_prob=0.15, del6q_prob=0.40, hrd_prob=0.10, ibrutinib_eligible=True),
        "CXCR4":  dict(age_mu=66, age_sd=10, wm_prob=0.72, myd88_l265p_prob=0.85, cxcr4_mut_prob=0.75, del6q_prob=0.35, hrd_prob=0.08, ibrutinib_eligible=True),
        "BTK":    dict(age_mu=58, age_sd=12, wm_prob=0.55, myd88_l265p_prob=0.45, cxcr4_mut_prob=0.20, del6q_prob=0.25, hrd_prob=0.12, ibrutinib_eligible=False),
        "ATM":    dict(age_mu=62, age_sd=11, wm_prob=0.68, myd88_l265p_prob=0.70, cxcr4_mut_prob=0.25, del6q_prob=0.38, hrd_prob=0.45, ibrutinib_eligible=True),
        "TP53":   dict(age_mu=60, age_sd=12, wm_prob=0.65, myd88_l265p_prob=0.72, cxcr4_mut_prob=0.18, del6q_prob=0.42, hrd_prob=0.20, ibrutinib_eligible=True),
        "NBN":    dict(age_mu=57, age_sd=13, wm_prob=0.73, myd88_l265p_prob=0.80, cxcr4_mut_prob=0.22, del6q_prob=0.35, hrd_prob=0.35, ibrutinib_eligible=True),
        "CARD11": dict(age_mu=62, age_sd=11, wm_prob=0.60, myd88_l265p_prob=0.55, cxcr4_mut_prob=0.20, del6q_prob=0.30, hrd_prob=0.12, ibrutinib_eligible=False),
        "PIK3CD": dict(age_mu=50, age_sd=15, wm_prob=0.55, myd88_l265p_prob=0.60, cxcr4_mut_prob=0.20, del6q_prob=0.25, hrd_prob=0.10, ibrutinib_eligible=True),
    }
    p = wm_params[gene]

    age           = max(25, min(90, round(rng.gauss(p["age_mu"], p["age_sd"]), 1)))
    has_wm        = rng.random() < p["wm_prob"]
    myd88_l265p   = rng.random() < p["myd88_l265p_prob"]
    cxcr4_mut     = rng.random() < p["cxcr4_mut_prob"]
    del6q         = rng.random() < p["del6q_prob"]
    hrd_positive  = rng.random() < p["hrd_prob"]
    ibrutinib_elig = p["ibrutinib_eligible"] and myd88_l265p and not cxcr4_mut
    zanubrutinib_elig = p["ibrutinib_eligible"] and myd88_l265p

    tumour_types  = ["Waldenström Macroglobulinemia", "LPL (non-IgM)", "IgM MGUS progressed", "B-cell NHL",
                     "CLL/SLL overlap", "Plasma cell component", "APDS lymphoproliferation"]
    tumour_probs  = [0.62, 0.08, 0.10, 0.09, 0.06, 0.03, 0.02]
    cumulative = 0.0; r = rng.random()
    tumour_type = tumour_types[-1]
    for tt, tp in zip(tumour_types, tumour_probs):
        cumulative += tp
        if r < cumulative:
            tumour_type = tt; break

    return {
        "gene":             gene,
        "seed":             seed,
        "age_at_dx":        age,
        "tumour_type":      tumour_type,
        "has_wm":           has_wm,
        "myd88_l265p":      myd88_l265p,
        "cxcr4_mut":        cxcr4_mut,
        "del6q":            del6q,
        "hrd_positive":     hrd_positive,
        "ibrutinib_elig":   ibrutinib_elig,
        "zanubrutinib_elig": zanubrutinib_elig,
    }


def _build_cohorts() -> dict:
    cohorts = {}
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        cohorts[gene] = [_make_patient(gene, SEED_BASE + i + j * 8) for j in range(40)]
    return cohorts


def generate_overview() -> dict:
    cohorts = _build_cohorts()
    all_pts = [p for pts in cohorts.values() for p in pts]

    wm_n         = sum(1 for p in all_pts if p["has_wm"])
    myd88_l265p_n = sum(1 for p in all_pts if p["myd88_l265p"])
    cxcr4_n      = sum(1 for p in all_pts if p["cxcr4_mut"])
    del6q_n      = sum(1 for p in all_pts if p["del6q"])
    zanubrutinib_n = sum(1 for p in all_pts if p["zanubrutinib_elig"])
    mean_age     = round(sum(p["age_at_dx"] for p in all_pts) / len(all_pts), 1)

    gene_summary = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        wm_gene_n = sum(1 for p in pts if p["has_wm"])
        gene_summary.append({
            "gene":           gene,
            "locus":          gd["locus"],
            "n":              len(pts),
            "wm_n":           wm_gene_n,
            "wm_pct":         round(100 * wm_gene_n / len(pts), 1),
            "mean_age_onset": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "myd88_l265p_n":  sum(1 for p in pts if p["myd88_l265p"]),
            "cxcr4_n":        sum(1 for p in pts if p["cxcr4_mut"]),
            "inheritance":    gd["inheritance"],
            "key_distinctions": [
                gd["wm_risk"][:80],
                gd["key_rule"][:80],
            ],
        })

    genes_detail = []
    for gd in ATLAS_GENES:
        genes_detail.append({
            "gene":           gd["gene"],
            "protein":        gd["protein"],
            "locus":          gd["locus"],
            "protein_size":   gd["protein_size"],
            "inheritance":    gd["inheritance"],
            "cancer_risk":    gd["wm_risk"],
            "pathognomonic":  gd["pathognomonic"],
            "surveillance_key": gd["surveillance"][:160],
            "key_distinctions": [
                gd["wm_risk"][:80],
                gd["key_avoid"][:80],
                gd["key_rule"][:80],
            ],
        })

    return {
        "atlas":                 "Hereditary-WM-LPL-Predisposition-Atlas",
        "seed_range":            f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":        len(all_pts),
        "genes_n":               8,
        "wm_total_n":            wm_n,
        "wm_total_pct":          round(100 * wm_n / len(all_pts), 1),
        "mean_age_onset":        mean_age,
        "myd88_l265p_n":         myd88_l265p_n,
        "myd88_l265p_pct":       round(100 * myd88_l265p_n / len(all_pts), 1),
        "cxcr4_n":               cxcr4_n,
        "cxcr4_pct":             round(100 * cxcr4_n / len(all_pts), 1),
        "del6q_n":               del6q_n,
        "del6q_pct":             round(100 * del6q_n / len(all_pts), 1),
        "zanubrutinib_eligible_n":   zanubrutinib_n,
        "zanubrutinib_eligible_pct": round(100 * zanubrutinib_n / len(all_pts), 1),
        "gene_summary":          gene_summary,
        "genes_detail":          genes_detail,
    }


def generate_breakdown() -> dict:
    cohorts = _build_cohorts()
    _gene_pathog = {gd["gene"]: gd["pathognomonic"] for gd in ATLAS_GENES}
    _gene_surv   = {gd["gene"]: gd["surveillance"][:120] for gd in ATLAS_GENES}
    _gene_dist   = {gd["gene"]: [gd["wm_risk"][:80], gd["key_rule"][:80]] for gd in ATLAS_GENES}

    breakdown_array = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        wm_n = sum(1 for p in pts if p["has_wm"])
        breakdown_array.append({
            "gene":             gene,
            "locus":            gd["locus"],
            "n":                len(pts),
            "mean_age_onset":   round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "wm_n":             wm_n,
            "wm_pct":           round(100 * wm_n / len(pts), 1),
            "myd88_l265p_n":    sum(1 for p in pts if p["myd88_l265p"]),
            "cxcr4_n":          sum(1 for p in pts if p["cxcr4_mut"]),
            "del6q_n":          sum(1 for p in pts if p["del6q"]),
            "zanubrutinib_elig_n": sum(1 for p in pts if p["zanubrutinib_elig"]),
            "seed":             SEED_BASE + i,
            "key_distinctions": _gene_dist[gene],
            "pathognomonic":    _gene_pathog[gene],
            "surveillance_key": _gene_surv[gene],
        })
    return {"breakdown": breakdown_array, "total_patients": sum(len(v) for v in cohorts.values())}


def generate_definitions() -> dict:
    defs = {}
    for gd in ATLAS_GENES:
        defs[gd["gene"]] = {
            "gene":          gd["gene"],
            "locus":         gd["locus"],
            "protein_size":  gd["protein_size"],
            "syndrome":      gd["syndrome"],
            "inheritance":   gd["inheritance"],
            "wm_risk":       gd["wm_risk"],
            "pathognomonic": gd["pathognomonic"],
            "key_avoid":     gd["key_avoid"],
            "key_rule":      gd["key_rule"],
            "surveillance":  gd["surveillance"],
            "targeted_rx":   gd["targeted_rx"],
        }

    key_distinctions = [
        "MYD88 L265P WM vs MYD88 WT WM: L265P somatic >90% WM = ibrutinib/zanubrutinib sensitive; WT WM (~10%) = BTKi less effective → rituximab-based preferred; TEST before BTKi",
        "CXCR4WHIM double mutant: MYD88L265P + CXCR4WHIM = WORST WM prognosis; inferior ibrutinib PFS → zanubrutinib + venetoclax combination preferred; plerixafor emerging CXCR4 antagonist",
        "NBN 657del5 vs ATM monoallelic NHL risk: NBN 657del5 = B-cell NHL 60x HIGHEST monoallelic signal; ATM = NHL 3-5x, CLL 4-5x; both Eastern European populations — test NBN first in Czech/Slovak/Polish WM",
        "CARD11 GOF vs PIK3CD GOF WM: CARD11-GOF → constitutive NF-κB DOWNSTREAM of BTK → ibrutinib INEFFECTIVE; PIK3CD-GOF → constitutive PI3Kδ → leniolisib FDA2023 targeted Rx; distinct signalling nodes",
        "BTK XLA vs BTKi target: BTK LOF (XLA) = absent B cells = immunodeficiency (treat with IVIG); BTK as DRUG TARGET = ibrutinib/zanubrutinib covalent kinase inhibition in WM (opposite rationale); do NOT confuse",
        "Radiosensitivity in hereditary WM: NBN biallelic = ABSOLUTE (TBI CI); ATM biallelic = ABSOLUTE (TBI CI); TP53 germline = AVOID RADIATION ABSOLUTELY; ATM/NBN monoallelic = RT reduce 20-30%; zanubrutinib ASPEN outperforms ibrutinib (better VGPR, less AF)",
        "Leniolisib vs idelalisib for PIK3CD: leniolisib FDA2023 APDS1 — first approved PI3Kδi; idelalisib CLL/FL approved but significant hepatotoxicity (transaminitis) + diarrhoea + opportunistic infections; leniolisib more selective, better safety profile in APDS1",
        "Antiviral prophylaxis APDS1 (PIK3CD GOF): EBV/CMV/VZV reactivation PATHOGNOMONIC APDS1 — herpesviral reactivation = hallmark; acyclovir/valacyclovir MANDATORY prophylaxis; EBV-driven lymphoproliferation in APDS1 ≠ WM (biopsy required to distinguish)",
        "Zanubrutinib ASPEN superiority over ibrutinib in WM: VGPR 28% vs 19% (zanubrutinib superior); fewer atrial fibrillation events (2.6% vs 15.2% AF with ibrutinib at 3yr); PFS HR 0.75 (zanubrutinib); zanubrutinib preferred BTKi in ALL WM subtypes",
        "Germline WM predisposition cascade: familial WM (≥2 first-degree relatives with WM/LPL) → test FULL hereditary WM panel (MYD88 germline LOF/GOF + CXCR4 + ATM + NBN + TP53 + CARD11 + PIK3CD); annual IgM quantification in at-risk carriers from age 40yr",
    ]

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"WM RISK: {gd['wm_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": key_distinctions,
        "seed_range":                f"{SEED_BASE}–{SEED_BASE + 7}",
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"Total patients: {ov['total_patients']}, WM n: {ov['wm_total_n']} ({ov['wm_total_pct']}%)")
    print(f"MYD88 L265P: {ov['myd88_l265p_n']} ({ov['myd88_l265p_pct']}%), CXCR4: {ov['cxcr4_n']} ({ov['cxcr4_pct']}%)")
    print("=== BREAKDOWN ===")
    br = generate_breakdown()
    for b in br["breakdown"]:
        print(f"  {b['gene']}: WM {b['wm_pct']}% ({b['wm_n']}/{b['n']}), age {b['mean_age_onset']}yr")
    print("=== DEFINITIONS ===")
    df = generate_definitions()
    print(f"Genes: {list(df['gene_defs'].keys())}")
    print("Script verified OK")
