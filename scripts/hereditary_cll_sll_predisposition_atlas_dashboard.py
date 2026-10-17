#!/usr/bin/env python3
"""Hereditary-CLL-SLL-Predisposition-Atlas -- Complete 8-Gene Reference
ATM     (Ataxia-telangiectasia mutated; 3056aa; 11q22.3; AD/AR LOF;
         CLL 4-5x monoallelic — HIGHEST validated hereditary CLL gene; del11q somatic ~12% CLL;
         biallelic A-T = ALL/lymphoma/CLL 100x PATHOGNOMONIC; RADIOSENSITIVITY ABSOLUTE biallelic;
         Ceralasertib ATRi synthetic lethality; del11q = unmutated IGHV + high-risk CLL;
         seed SEED_BASE+0).
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         LFS; del17p somatic 5-7% at CLL diagnosis → WORST PROGNOSIS CLL;
         AVOID RADIATION ABSOLUTELY (LFS + del17p RT catastrophic); venetoclax FDA2016 del17p CLL;
         TP53-mutated CLL: venetoclax + BTKi preferred; ibrutinib/acalabrutinib first-line del17p;
         seed SEED_BASE+1).
POT1    (Protection of telomeres 1; 634aa; 7q31.33; AD LOF;
         Familial CLL MOST SPECIFIC single gene — 4-6x CLL lifetime risk; shelterin complex;
         telomere dysfunction → chromosomal instability → CLL; POT1-mutated CLL younger onset;
         germline POT1 LOF = familial CLL/lymphoma — cascade testing mandatory;
         seed SEED_BASE+2).
CHEK2   (Checkpoint kinase 2; 543aa; 22q12.1; AD LOF;
         1100delC European founder allele (1:200 Northern European — Czech/Polish/German/Swedish);
         CLL 2-3x monoallelic; BRCA-like HRD partial; downstream ATM checkpoint; IVS2+1G>A Ashkenazi;
         ATM-CHEK2 axis: both attenuate DNA damage checkpoint in B cells → clonal expansion;
         seed SEED_BASE+3).
NBN     (Nibrin/NBS1; 754aa; 8q21.3; AD/AR LOF;
         Nijmegen Breakage Syndrome (NBS); 657del5 Slavic founder (Czech/Slovak/Polish 1/150);
         B-cell NHL 60x HIGHEST monoallelic — strongest single-gene NHL signal; CLL 3-5x;
         RADIOSENSITIVITY biallelic ABSOLUTE CI; MRN complex DSB repair; ibrutinib/zanubrutinib for CLL;
         seed SEED_BASE+4).
BRCA2   (Breast cancer susceptibility gene 2; 3418aa; 13q12.3; AD LOF;
         CLL 2-3x elevated; FA-D1 biallelic PATHOGNOMONIC; HRD in CLL cells; del13q locus;
         PARP inhibitor sensitivity in HRD-positive CLL; HRD testing recommended BRCA2-CLL;
         Olaparib/rucaparib HRD-CLL investigational; avoid unnecessary alkylating agents;
         seed SEED_BASE+5).
CARD11  (Caspase recruitment domain family member 11; 1154aa; 7p22.2; AD LOF/GOF;
         GOF = BENTA disease: constitutive NF-κB → B-cell expansion → CLL/lymphoma risk;
         LOF = immunodeficiency → compensatory B-cell lymphoproliferation;
         ibrutinib less effective CARD11-GOF CLL (downstream of BTK — bypass signalling);
         BCL10-MALT1 CBM complex; BTK-CARD11-NF-κB axis central CLL survival;
         seed SEED_BASE+6).
CDKN2A  (Cyclin-dependent kinase inhibitor 2A; 156aa; 9p21.3; AD LOF;
         FAMM syndrome — melanoma + pancreatic + lymphoid malignancy; del9p21 somatic CLL 5-10%;
         p16INK4a CDK4/6 inhibitor → CDK4/6 inhibitor palbociclib INACTIVE in CDKN2A-null CLL;
         pancreatic cancer 17-39x (EUS surveillance from 40yr); CDKN2A biallelic → Leiden/E-NHL;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3542-3549)
"""
import random

SEED_BASE = 3542

ATLAS_GENES = [
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD/AR LOF -- 3056aa -- "
            "Ataxia-Telangiectasia-Mutated-350kDa-"
            "PIK3-Kinase-DSB-Master-Sensor-"
            "CLL-4-5x-Monoallelic-del11q-12pct-CLL-"
            "RADIOSENSITIVITY-ABSOLUTE-Biallelic-A-T-"
            "Ceralasertib-ATRi-Synthetic-Lethality-"
            "del11q-IGHV-Unmutated-High-Risk-CLL-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa / 11q22.3 ATM CLL molecular context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; serine/threonine PIK3-related kinase (PIKK family); "
            "  HEAT repeats (aa 1-1960) — scaffold; FAT domain (aa 1960-2566) — regulation; "
            "  Kinase domain (aa 2714-2962) — catalytic; FATC (aa 2963-3056) — essential for function; "
            "  Activated by DSBs → autophosphorylation Ser1981 → monomer → active; "
            "  Substrates: H2AX (γH2AX), CHK2, BRCA1, RAD51, MDM2, p53; "
            "CLL MOLECULAR CONTEXT: "
            "  Germline ATM LOF monoallelic → CLL 4-5x lifetime risk (HIGHEST validated hereditary CLL gene); "
            "  del11q22.3 somatic = most common cytogenetic lesion in ATM-associated CLL (~12% at dx); "
            "  del11q CLL = unmutated IGHV, bulky lymphadenopathy, aggressive course; "
            "  Biallelic A-T: CLL/lymphoma/ALL 100x lifetime — onset childhood/young adult; "
            "RADIOSENSITIVITY: "
            "  Biallelic A-T: RADIOSENSITIVITY ABSOLUTE — no RT, no TBI (conditioning); "
            "  Monoallelic germline ATM: RT reduce 20-30%, busulfan-based non-TBI conditioning for SCT; "
            "TREATMENT: "
            "  del11q CLL: ibrutinib or acalabrutinib BTKi front-line (del11q does NOT = BTKi resistance); "
            "  Ceralasertib (AZD6738 ATRi): synthetic lethality with ATM LOF — Phase 2 active CLL; "
            "  Venetoclax + obinutuzumab ± BTKi — del11q high-risk CLL; "
            "  Avoid fludarabine-based chemo in ATM-biallelic (DNA repair defect → toxicity); "
            "SURVEILLANCE: "
            "  Annual CBC + lymphocyte immunophenotyping from age 35yr; "
            "  CT chest/abdomen 2-yearly; SPEP annually; cascade testing family; "
            "  A-T: WBMRI Toronto protocol annually; NO RT ever; "
            "BIALLELIC A-T RADIOSENSITIVITY RULE: "
            "  ANY RT in A-T = pulmonary fibrosis + myelosuppression + radiation recall catastrophic; "
            "  TBI conditioning for SCT ABSOLUTELY CONTRAINDICATED biallelic ATM"
        ),
        "syndrome": "A-T / CLL 4-5x monoallelic / del11q somatic / RADIOSENSITIVITY ABSOLUTE biallelic",
        "inheritance": "AD/AR LOF",
        "cll_risk": "ATM monoallelic: CLL 4-5x HIGHEST hereditary gene; del11q somatic 12% at dx; biallelic A-T: CLL/lymphoma/ALL 100x; del11q CLL = unmutated IGHV + bulky adenopathy; Ceralasertib ATRi Phase 2 synthetic lethality ATM-LOF CLL",
        "pathognomonic": "Cerebellar ataxia + telangiectasias + immunodeficiency + elevated AFP = A-T PATHOGNOMONIC; gamma-H2AX foci absent/reduced DNA-damage response; del11q22.3 on FISH + unmutated IGHV = ATM-pathway CLL high-risk; radiosensitivity testing (clonogenic survival or chromosomal breakage) confirming cellular ATM deficiency",
        "key_avoid": "NEVER use RT in biallelic A-T (ABSOLUTE CONTRAINDICATION — catastrophic pulmonary fibrosis + myelosuppression); avoid TBI for SCT conditioning biallelic ATM; reduce RT 20-30% monoallelic germline ATM; avoid fludarabine-based chemo in biallelic ATM; no alkylating agents in A-T without haematology review",
        "key_rule": "ATM CLL key rules: (1) del11q = IBRUTINIB/ACALABRUTINIB first-line (del11q is NOT BTKi resistance — ibrutinib overcomes del11q prognosis); (2) Ceralasertib ATRi Phase 2 synthetic lethality = emerging ATM-CLL targeted Rx; (3) RADIOSENSITIVITY ABSOLUTE biallelic — no RT, no TBI; (4) monoallelic: RT reduce 20-30%, busulfan conditioning not TBI; (5) cascade test first-degree relatives for germline ATM.",
        "surveillance": "Annual CBC + CLL immunophenotyping (CD5+CD19+CD23+) from age 35yr; CLL FISH panel (del11q, del17p, del13q, trisomy12) at diagnosis; IGHV mutation status at dx; CT annually if high CLL burden; ceralasertib trial if relapsed/refractory ATM-mutated CLL; WBMRI Toronto if A-T biallelic; cascade testing first-degree relatives germline ATM",
        "targeted_rx": "Ibrutinib BTKi (FDA2013 CLL first-line; del11q responds); acalabrutinib BTKi (less AF — preferred if cardiac); venetoclax (BCL2i; del11q CLL); ceralasertib (AZD6738 ATRi Phase 2 synthetic lethality ATM-LOF); obinutuzumab + venetoclax (front-line fit CLL); zanubrutinib BTKi (fewer off-target); avoid fludarabine biallelic; avoid TBI conditioning",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Transcription-Factor-"
            "Tumour-Suppressor-G1-Arrest-"
            "LFS-del17p-5-7pct-CLL-WORST-PROGNOSIS-"
            "Venetoclax-FDA2016-del17p-CLL-"
            "AVOID-RADIATION-ABSOLUTELY-LFS-"
            "WBMRI-Toronto-Annually-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 CLL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; N-terminal transactivation domain (aa 1-40) — MDM2/p300 interaction; "
            "  Proline-rich region (aa 40-92); DNA-binding domain (aa 100-290) — MOST mutations; "
            "  Tetramerisation domain (aa 325-356); C-terminal regulatory domain (aa 363-393); "
            "  Forms tetramer to bind p53 response elements → transcriptional activation; "
            "  R175H, R248Q, R248W, R249S, R273H, R282W = hotspot missense mutations (GOF + dominant negative); "
            "CLL CONTEXT: "
            "  Germline TP53 LOF → Li-Fraumeni syndrome (LFS); CLL/lymphoma 2-3x LFS; "
            "  del17p13.1 somatic = 5-7% at CLL diagnosis; 20-30% at relapse; 40-50% after chemo-refractory; "
            "  del17p CLL = WORST PROGNOSIS — short response to chemo; BTKi + venetoclax REQUIRED; "
            "  TP53-mutated CLL (without del17p): same poor outcome as del17p — TREAT SAME; "
            "TREATMENT: "
            "  del17p / TP53-mutated CLL: venetoclax (BCL2i) FDA2016 + ibrutinib/acalabrutinib; "
            "  Venetoclax + rituximab (Vnet-R) FDA2018 del17p relapsed CLL; "
            "  ELEVATE CLL TN: acalabrutinib front-line superior to ibrutinib (less AF); "
            "  AVOID RT in LFS (ABSOLUTELY — catastrophic secondary malignancy); "
            "  pirtobrutinib (non-covalent BTKi) C481S-resistant CLL; "
            "  allogeneic SCT del17p high-risk young patients; "
            "SURVEILLANCE: "
            "  LFS: WBMRI Toronto protocol annually (brain/spine/whole body); "
            "  CBC + IgM/SPEP annually from age 30yr; colonoscopy, breast MRI, dermatology; "
            "  CLL: FISH del17p at diagnosis AND at relapse (clonal evolution); "
            "  Annual surveillance EVERY year life-long (LFS lifelong cancer risk)"
        ),
        "syndrome": "Li-Fraumeni Syndrome (LFS) / del17p WORST CLL prognosis / venetoclax first-line del17p",
        "inheritance": "AD LOF",
        "cll_risk": "TP53 LFS: CLL/lymphoma 2-3x; del17p somatic 5-7% at dx → 40-50% chemo-refractory CLL; del17p CLL = WORST prognosis; venetoclax FDA2016 + BTKi preferred; TP53-mutated CLL = SAME prognosis as del17p; AVOID RT ABSOLUTELY in LFS",
        "pathognomonic": "LFS: sarcoma + early-onset breast cancer + brain tumour + adrenocortical carcinoma + CLL TRIAD PATHOGNOMONIC; del17p CLL + short response to chemo + rapid relapse = del17p PATHOGNOMONIC; TP53-mutated CLL same poor outcome; WBMRI Toronto protocol MANDATORY annually LFS",
        "key_avoid": "AVOID RADIATION ABSOLUTELY in LFS (secondary malignancy catastrophic — multiple primary cancers); no RT even for relapsed CLL in del17p/LFS; avoid chlorambucil/fludarabine mono (del17p = chemo-resistant); do NOT use bendamustine-based as first-line del17p (inferior); avoid ibrutinib alone del17p without venetoclax doublet (inferior remission depth)",
        "key_rule": "TP53 del17p CLL key rules: (1) venetoclax + BTKi (ibrutinib/acalabrutinib/zanubrutinib) COMBINATION — not monotherapy; (2) AVOID RADIATION ABSOLUTELY (LFS) — no exceptions; (3) WBMRI Toronto annually LFS patients; (4) test TP53 mutation AND del17p at every relapse (clonal evolution); (5) pirtobrutinib (non-covalent) C481S-resistant BTKi; (6) allogeneic SCT consideration young del17p high-risk.",
        "surveillance": "LFS: WBMRI Toronto annually — whole body + brain/spine; colonoscopy age 25yr + 5-yearly; breast MRI annually from age 20yr; dermatology annually; CBC + CLL panel annually from age 30yr; CLL: FISH del17p at diagnosis AND relapse; NGS TP53 mutation sequencing; venetoclax MRD monitoring; cascade testing first-degree relatives LFS",
        "targeted_rx": "Venetoclax (ABT-199/GDC-0199 BCL2i; FDA2016 del17p CLL; FDA2018 rel/ref CLL); ibrutinib (BTKi; active del17p); acalabrutinib (BTKi; ELEVATE CLL TN — less AF); zanubrutinib (SEQUOIA del17p CLL); pirtobrutinib (LOXO-305 non-covalent BTKi C481S-resistant); obinutuzumab + venetoclax (front-line); allogeneic SCT young patients; avoid chemo-based (del17p chemo-resistant); avoid RT (LFS ABSOLUTELY)",
    },
    {
        "gene": "POT1",
        "protein": (
            "POT1 -- 7q31.33 AD LOF -- 634aa -- "
            "Protection-Of-Telomeres-1-71kDa-"
            "Shelterin-OB-Fold-ssDNA-Binding-"
            "Familial-CLL-MOST-SPECIFIC-4-6x-"
            "Telomere-Dysfunction-Chromosomal-Instability-"
            "Younger-Onset-CLL-Cascade-Testing-MANDATORY-OMIM-606478"
        ),
        "locus": "7q31.33",
        "protein_size": (
            "634 aa / ~71 kDa / 7q31.33 POT1 CLL molecular context: "
            "STRUCTURE: "
            "  634 aa / ~71 kDa; N-terminal OB-fold 1 (aa 1-87) and OB-fold 2 (aa 88-178) — telomeric ssDNA binding; "
            "  C-terminal TPP1-interacting domain (aa 320-634) — shelterin complex assembly; "
            "  Binds 3'-telomeric overhang ssDNA (TTAGGG repeat) → protects from ATM/ATR activation; "
            "  POT1 LOF → telomere deprotection → ATM/ATR activation → chromosomal instability; "
            "SHELTERIN COMPLEX: "
            "  TRF1-TRF2-RAP1-TIN2-TPP1-POT1 hexamer; "
            "  POT1-TPP1 heterodimer = telomere end-protection complex; "
            "  POT1 LOF → T-loop uncapping → chromosomal fusions → aneuploidy → clonal haematopoiesis; "
            "FAMILIAL CLL: "
            "  Germline POT1 LOF = MOST SPECIFIC single gene for familial CLL (4-6x lifetime risk); "
            "  POT1-mutated CLL: younger onset (~10yr earlier vs sporadic CLL); more aggressive; "
            "  Somatic POT1 mutations (2-3% sporadic CLL) — distinct from germline but same pathway; "
            "  Other shelterin genes: ACD (TPP1), TINF2, TERC, TERT — related familial lymphoma risk; "
            "CLINICAL FEATURES: "
            "  Familial CLL cluster (≥2 first-degree relatives with CLL) → test POT1 germline first; "
            "  Younger onset, IGHV may be unmutated, del17p or complex karyotype on FISH; "
            "TREATMENT: "
            "  POT1-mutated CLL: standard CLL protocols (ibrutinib/venetoclax); "
            "  No targeted POT1 therapy (2026) — telomerase inhibitors investigational; "
            "  Cascade testing MANDATORY: first-degree relatives CLL panel from age 30yr"
        ),
        "syndrome": "Familial CLL MOST SPECIFIC gene / shelterin telomere / younger-onset CLL / cascade testing mandatory",
        "inheritance": "AD LOF",
        "cll_risk": "POT1 AD LOF: familial CLL 4-6x MOST SPECIFIC single hereditary CLL gene; younger onset (~10yr); shelterin telomere deprotection → chromosomal instability → CLL; somatic POT1 mutations 2-3% sporadic CLL; cascade testing MANDATORY first-degree relatives",
        "pathognomonic": "Familial CLL cluster (≥2 first-degree relatives) + germline POT1 LOF = PATHOGNOMONIC familial CLL; younger onset CLL + telomere dysfunction markers + family history = test POT1 first; somatic POT1 CLL + germline POT1 = dual-hit telomere instability; other shelterin gene mutations (ACD/TINF2) in differential",
        "key_avoid": "Do NOT assume younger-onset CLL is sporadic — test POT1 germline; do NOT use RT without assessing telomere repair capacity (POT1-LOF CLL may have increased radiosensitivity — caution); do NOT skip cascade testing (familial CLL carries 4-6x risk for relatives); do NOT confuse somatic POT1 (2-3% sporadic) with germline POT1 (familial CLL) — NGS panel required",
        "key_rule": "POT1 familial CLL key rules: (1) familial CLL cluster → germline POT1 FIRST; (2) cascade testing MANDATORY first-degree relatives (50% inheritance AD); (3) standard CLL treatment (ibrutinib/venetoclax) — no specific POT1 targeted therapy; (4) younger-onset CLL → full hereditary CLL panel (ATM + TP53 + POT1 + CHEK2 + NBN); (5) telomerase inhibitors (imetelstat) investigational — watch trials.",
        "surveillance": "Annual CBC + CLL immunophenotyping from age 30yr (10yr earlier than sporadic); CLL FISH at diagnosis; IGHV mutation status; telomere length assay (research); cascade testing first-degree relatives; SPEP annually; CT chest/abdomen if symptomatic; familial CLL registry enrollment (NIH CLL Family Study)",
        "targeted_rx": "Ibrutinib BTKi (CLL first-line); acalabrutinib BTKi (ELEVATE CLL TN); venetoclax BCL2i (del17p/TP53 or refactory CLL); obinutuzumab+venetoclax front-line; imetelstat (telomerase inhibitor — investigational CLL); standard CLL protocols — no POT1-specific FDA-approved therapy (2026); cascade testing MANDATORY; clinical trial enrollment recommended",
    },
    {
        "gene": "CHEK2",
        "protein": (
            "CHEK2 -- 22q12.1 AD LOF -- 543aa -- "
            "CHK2-Checkpoint-Kinase-2-61kDa-"
            "ATM-Downstream-DNA-Damage-Checkpoint-"
            "1100delC-European-Founder-1-in-200-"
            "CLL-2-3x-Monoallelic-"
            "IVS2+1G>A-Ashkenazi-Founder-OMIM-604373"
        ),
        "locus": "22q12.1",
        "protein_size": (
            "543 aa / ~61 kDa / 22q12.1 CHEK2 CLL molecular context: "
            "STRUCTURE: "
            "  543 aa / ~61 kDa; N-terminal SQ/TQ cluster (aa 1-90) — ATM phosphorylation sites Thr68 key; "
            "  FHA (forkhead-associated) domain (aa 92-175) — phosphopeptide binding; "
            "  Kinase domain (aa 220-489) — Ser/Thr kinase; "
            "  ATM activates CHEK2 at Thr68 → CHEK2 dimerises → activated → phosphorylates CDC25A/B/C, BRCA1, p53, MDM2; "
            "FOUNDER ALLELES: "
            "  1100delC: frameshift exon 10; Central/Northern European founder; ~1:200 Poles/Czechs/Germans/Swedes; "
            "  IVS2+1G>A: splice site; Ashkenazi Jewish founder (~1:100 carrier); "
            "  I157T: missense kinase domain; more moderate function impairment; common European; "
            "CLL RISK: "
            "  1100delC monoallelic: CLL 2-3x lifetime risk; B-cell NHL 2-3x; prostate 1.5-2x; "
            "  CHEK2 LOF → attenuated ATM-CHEK2 checkpoint in B cells → clonal B-cell expansion; "
            "  CHEK2 + ATM doubly heterozygous: HIGHEST combined CLL risk (near additive); "
            "  I157T: moderate CLL risk (1.5-2x); less severe than 1100delC; "
            "HRD: "
            "  CHEK2 LOF → partial HRD phenotype; PARP inhibitor sensitivity CLL cells (preclinical); "
            "  Platinum-based therapy relatively effective CHEK2-LOF CLL (enhanced DNA damage sensitivity); "
            "TREATMENT: "
            "  CHEK2-CLL: standard ibrutinib/venetoclax protocols; platinum sensitivity may guide later lines; "
            "  No FDA-approved CHEK2-specific therapy (2026); "
            "  Checkpoint kinase inhibitors (prexasertib) investigational (CHK1/CHK2 dual)"
        ),
        "syndrome": "CLL 2-3x / 1100delC European founder (1:200) / IVS2+1G>A Ashkenazi / ATM-CHEK2 checkpoint axis",
        "inheritance": "AD LOF",
        "cll_risk": "CHEK2 1100delC monoallelic: CLL 2-3x; B-cell NHL 2-3x; IVS2+1G>A Ashkenazi 1:100 carrier; ATM-CHEK2 doubly heterozygous = highest combined CLL risk; I157T moderate (1.5-2x); CHEK2 + ATM = additive checkpoint loss → B-cell clonal expansion",
        "pathognomonic": "1100delC on BRCA/Lynch panel or CLL hereditary panel + CLL = CHEK2-associated CLL; IVS2+1G>A in Ashkenazi family with CLL = CHEK2 Ashkenazi founder; double heterozygosity (CHEK2 1100delC + ATM) = near-PATHOGNOMONIC for hereditary CLL/lymphoma cluster; CHEK2 with elevated AFP (A-T vs CHEK2) — test ATM biallelic to differentiate",
        "key_avoid": "Do NOT assume CHEK2 CLL is sporadic — screen first-degree relatives; do NOT use CHEK2 1100delC as 'low-risk variant' — CLL 2-3x is clinically significant in families; do NOT miss CHEK2 + ATM double heterozygosity (additive risk); do NOT prescribe RT without considering partial HRD in CHEK2-CLL; I157T = intermediate risk — intermediate vigilance",
        "key_rule": "CHEK2 CLL key rules: (1) 1100delC European founder — COMMON in Central/Northern European CLL families; (2) CLL 2-3x monoallelic; (3) ATM-CHEK2 combined testing in CLL families (additive risk); (4) standard CLL protocols (ibrutinib/venetoclax) — no CHEK2-specific FDA therapy; (5) partial HRD sensitivity — platinum or PARP inhibitors in later lines; (6) cascade test first-degree relatives.",
        "surveillance": "Annual CBC + CLL immunophenotyping from age 40yr; CLL FISH at diagnosis; CHEK2 + ATM combined panel; IGHV mutation status; cascade testing first-degree relatives; CT chest/abdomen if symptomatic; SPEP annually; consider clinical trial enrollment CHEK2-associated CLL; prexasertib trials watch",
        "targeted_rx": "Ibrutinib BTKi (CLL first-line); acalabrutinib BTKi; venetoclax BCL2i; obinutuzumab + venetoclax; platinum-based (partial HRD — enhanced sensitivity CHEK2-LOF); PARP inhibitors (investigational HRD-CLL); prexasertib CHK1/CHK2 dual inhibitor (investigational); standard CLL protocols — no CHEK2-specific FDA-approved therapy (2026); cascade testing MANDATORY",
    },
    {
        "gene": "NBN",
        "protein": (
            "NBN -- 8q21.3 AD/AR LOF -- 754aa -- "
            "NBS1-Nibrin-85kDa-MRN-Complex-"
            "DSB-Sensor-ATM-Recruiter-"
            "657del5-Slavic-Founder-1-in-150-"
            "B-Cell-NHL-60x-HIGHEST-Monoallelic-"
            "CLL-3-5x-RADIOSENSITIVITY-Biallelic-ABSOLUTE-OMIM-602667"
        ),
        "locus": "8q21.3",
        "protein_size": (
            "754 aa / ~85 kDa / 8q21.3 NBN CLL molecular context: "
            "STRUCTURE: "
            "  754 aa / ~85 kDa; N-terminal FHA domain (aa 1-108) + BRCT1 (aa 114-197) + BRCT2 (aa 221-332) — phosphopeptide docking; "
            "  MRE11-binding domain (aa 665-693); ATM-interacting domain (aa 734-754); "
            "  MRN complex: MRE11-RAD50-NBS1 trimer; DSB sensor and ATM recruitment; "
            "  657del5 = exon 6 frameshift → p.K219fs; founder mutation in Czech/Slovak/Polish population; "
            "NBS1 (NIJMEGEN BREAKAGE SYNDROME): "
            "  NBS = biallelic NBN LOF → microcephaly + immunodeficiency + radiosensitivity + lymphoma 100x; "
            "  Carrier frequency 657del5 = 1:150 in Czech/Slovak/Polish (HIGHEST founder allele frequency); "
            "  NBS clinical triad: microcephaly + café au lait macules + immunodeficiency; "
            "B-CELL NHL CLL RISK: "
            "  NBN 657del5 monoallelic: B-cell NHL 60x HIGHEST single-gene NHL signal; CLL 3-5x; "
            "  STRONGEST germline NHL predisposition gene (per allele lifetime risk); "
            "  Most NBN monoallelic cancers = non-Hodgkin B-cell lymphoma/CLL; "
            "RADIOSENSITIVITY: "
            "  NBS biallelic: RADIOSENSITIVITY ABSOLUTE — no RT, no TBI (same as biallelic ATM); "
            "  Monoallelic NBN: RT reduce 20-30% (partial MRN complex dysfunction); "
            "TREATMENT: "
            "  NBN-CLL: ibrutinib/zanubrutinib preferred BTKi; zanubrutinib less cardiac AE; "
            "  PARP inhibitors: MRN complex LOF → partial HRD → olaparib/rucaparib sensitivity; "
            "  Avoid RT biallelic NBS; reduce RT monoallelic; busulfan conditioning not TBI"
        ),
        "syndrome": "NBS / 657del5 Slavic founder 1:150 / B-cell NHL 60x HIGHEST monoallelic / CLL 3-5x / RADIOSENSITIVITY ABSOLUTE biallelic",
        "inheritance": "AD/AR LOF",
        "cll_risk": "NBN 657del5 monoallelic: B-cell NHL 60x HIGHEST single-gene NHL signal; CLL 3-5x; NBS biallelic: lymphoma 100x ABSOLUTE CI RT; strongest per-allele germline NHL predisposition; 1:150 Czech/Slovak/Polish — COMMON Eastern European hereditary CLL gene",
        "pathognomonic": "Microcephaly + immunodeficiency + 657del5 = NBS PATHOGNOMONIC; B-cell NHL/CLL in Czech/Slovak/Polish patient + family history = test NBN 657del5 FIRST; NBN monoallelic CLL in Eastern European = founder allele; NBS biallelic CLL lymphoma = 100x risk; radiosensitivity testing (chromosomal breakage/clonogenic survival) confirms NBS biallelic",
        "key_avoid": "NEVER use RT in biallelic NBS (ABSOLUTE CONTRAINDICATION — same severity as biallelic ATM); no TBI conditioning biallelic NBS; avoid RT in monoallelic NBN without reduction 20-30%; do NOT miss 657del5 in Eastern European CLL (COMMON founder allele — 1:150 carrier); do NOT confuse NBS (biallelic) with 657del5 monoallelic (very different clinical risk)",
        "key_rule": "NBN 657del5 CLL key rules: (1) test FIRST in Czech/Slovak/Polish CLL patients (1:150 carrier freq); (2) monoallelic NBN = B-cell NHL 60x HIGHEST — CLL 3-5x; (3) RADIOSENSITIVITY ABSOLUTE biallelic NBS — no RT, no TBI; (4) RT reduce 20-30% monoallelic; (5) zanubrutinib preferred BTKi (less AF than ibrutinib); (6) PARP inhibitor sensitivity — MRN partial HRD; (7) cascade test family.",
        "surveillance": "Annual CBC + CLL immunophenotyping from age 35yr; CLL FISH at diagnosis; NBN 657del5 founder test in Eastern European CLL families; WBMRI Toronto biallelic NBS annually; cascade testing first-degree relatives; PARP sensitivity testing at relapse; SPEP annually; Ig levels (immunodeficiency NBS); lymphocyte subsets biallelic NBS",
        "targeted_rx": "Ibrutinib BTKi (CLL first-line); zanubrutinib BTKi PREFERRED (less AF — ASPEN trial); venetoclax BCL2i; PARP inhibitors (olaparib/rucaparib — partial HRD MRN-LOF CLL); obinutuzumab combination; avoid RT biallelic; busulfan non-TBI conditioning; clinical trial enrollment; cascade testing MANDATORY Eastern European families",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 AD LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-"
            "FANCD1-DNA-Binding-DSB-Repair-"
            "CLL-2-3x-Elevated-"
            "HRD-PARP-Olaparib-Sensitivity-"
            "FA-D1-Biallelic-PATHOGNOMONIC-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / ~384 kDa / 13q12.3 BRCA2 CLL molecular context: "
            "STRUCTURE: "
            "  3418 aa / ~384 kDa; N-terminal transactivation domain (BRCA2-PALB2 interaction aa 10-40); "
            "  BRC repeats ×8 (aa 1002-2085) — RAD51 monomer binding (pre-synaptic filament loading); "
            "  DNA-binding domain DBD (aa 2402-3186) — ssDNA/dsDNA junction binding; "
            "  C-terminal domain (aa 3187-3418) — CTP1, BRCA1-BRCA2-NBS1 complex; "
            "  BRCA2 escorts RAD51 to DSBs → strand invasion → homologous recombination; "
            "FANCONI ANAEMIA: "
            "  BRCA2 = FANCD1 (Fanconi anaemia complementation group D1); "
            "  Biallelic BRCA2 LOF = FA-D1 = PATHOGNOMONIC: medulloblastoma + Wilms + AML + developmental anomalies; "
            "  FA-D1 biallelic = extremely severe FA phenotype (most severe complementation group); "
            "CLL RISK: "
            "  Germline BRCA2 LOF monoallelic: CLL 2-3x elevated; del13q13 locus proximate; "
            "  BRCA2 CLL = HRD (homologous recombination deficient) phenotype in CLL cells; "
            "  HRD-positive CLL: enhanced sensitivity to PARP inhibitors; platinum chemotherapy active; "
            "  BRCA2 monoallelic: PARP inhibitor sensitivity (olaparib/rucaparib) in CLL cells; "
            "TREATMENT: "
            "  BRCA2-CLL: standard ibrutinib/venetoclax ± olaparib investigational; "
            "  HRD testing at CLL diagnosis recommended in germline BRCA2 carriers; "
            "  Avoid unnecessary alkylating agents (FA BRCA2 biallelic); "
            "  Platinum-based second-line: BRCA2-LOF CLL enhanced sensitivity"
        ),
        "syndrome": "HBOC / FA-D1 biallelic PATHOGNOMONIC / CLL 2-3x / HRD PARP sensitivity / del13q proximate locus",
        "inheritance": "AD LOF",
        "cll_risk": "BRCA2 AD LOF monoallelic: CLL 2-3x elevated; del13q13 proximate locus; HRD-positive CLL phenotype → PARP inhibitor (olaparib/rucaparib) sensitivity; FA-D1 biallelic: PATHOGNOMONIC severe FA — medulloblastoma + Wilms + AML; platinum chemotherapy active BRCA2-CLL",
        "pathognomonic": "FA-D1 biallelic BRCA2: medulloblastoma + Wilms tumour + AML + congenital abnormalities + DEB/MMC breakage test PATHOGNOMONIC; BRCA2 germline + CLL in HBOC family = hereditary CLL elevated risk; HRD testing positive (genomic scar/LOH) in BRCA2-CLL cells = PARP inhibitor eligible; del13q14 on FISH (RB1/miR-15a locus — proximate to BRCA2 13q12.3)",
        "key_avoid": "Do NOT use alkylating agents without haematological review in BRCA2 biallelic (FA-D1 marrow damage); avoid DNA-damaging agents in FA (profound sensitivity — aplasia); do NOT miss BRCA2 germline in CLL families presenting with HBOC pattern; do NOT withhold PARP inhibitors in HRD-positive BRCA2-CLL (investigational but meaningful activity); do NOT mistake del13q14 (RB1 locus) for BRCA2 (13q12.3) — adjacent but distinct",
        "key_rule": "BRCA2 CLL key rules: (1) HRD testing at CLL diagnosis in BRCA2 germline carriers; (2) olaparib/rucaparib PARP inhibitor — HRD-positive BRCA2-CLL investigational; (3) platinum-based second-line enhanced sensitivity; (4) FA-D1 biallelic = PATHOGNOMONIC severe FA phenotype — DEB/MMC test mandatory; (5) del13q14 (most common CLL cytogenetics 50%) distinct from del13q12.3 (BRCA2); (6) cascade test HBOC family CLL panel.",
        "surveillance": "Annual CBC + CLL panel from age 35yr BRCA2 carriers; HRD testing at CLL diagnosis; CLL FISH (del13q, del11q, del17p, trisomy12) at dx; IGHV mutation status; SPEP annually; PARP inhibitor eligibility assessment at relapse; HBOC surveillance (breast MRI annually from 25yr, prostate from 40yr, pancreatic EUS from 50yr); cascade testing family",
        "targeted_rx": "Ibrutinib BTKi (CLL first-line); venetoclax BCL2i; PARP inhibitors (olaparib/rucaparib investigational HRD-CLL); platinum-based second-line (BRCA2-enhanced sensitivity); obinutuzumab combinations; POLO trial analogue for CLL (maintenance olaparib investigational); avoid alkylating biallelic FA-D1; cascade testing MANDATORY HBOC families",
    },
    {
        "gene": "CARD11",
        "protein": (
            "CARD11 -- 7p22.2 AD LOF/GOF -- 1154aa -- "
            "CARD11-133kDa-CBM-Complex-"
            "BCL10-MALT1-Scaffold-NF-kB-"
            "BENTA-GOF-B-Cell-Expansion-"
            "CLL-NF-kB-Survival-BTK-Downstream-"
            "Ibrutinib-Less-Effective-CARD11-GOF-OMIM-607210"
        ),
        "locus": "7p22.2",
        "protein_size": (
            "1154 aa / ~133 kDa / 7p22.2 CARD11 CLL molecular context: "
            "STRUCTURE: "
            "  1154 aa / ~133 kDa; CARD domain (aa 1-100) — CARMA1 caspase recruitment; "
            "  Coiled-coil domain (aa 101-300) — BCL10 interaction; "
            "  PDZ domain; SH3 domain; GUK domain — scaffold architecture; "
            "  MAGUK (membrane-associated guanylate kinase) scaffold protein; "
            "  CARD11-BCL10-MALT1 (CBM) signalosome: downstream of BCR/TCR activation → IKK complex → NF-κB; "
            "BENTA DISEASE (GOF): "
            "  CARD11 GOF (L231F, L225LI, M243I) → constitutive NF-κB → BENTA disease; "
            "  BENTA: B-cell Expansion with NF-κB and T-cell Anergy; "
            "  Splenomegaly + lymphadenopathy + IgE elevation + atopic disease + recurrent infections; "
            "  CLL/B-cell lymphoma risk elevated in BENTA — constitutive NF-κB B-cell survival; "
            "CARD11 LOF: "
            "  CARD11 LOF → combined immunodeficiency → impaired B+T cell activation; "
            "  Compensatory B-cell lymphoproliferation → CLL/lymphoma predisposition; "
            "CLL TREATMENT: "
            "  CARD11-GOF CLL: ibrutinib LESS EFFECTIVE (CARD11 downstream of BTK — bypass signalling); "
            "  CARD11-GOF → NF-κB constitutive DOWNSTREAM of BTK → BTKi does NOT block NF-κB; "
            "  Venetoclax (BCL2i) + R2 (lenalidomide-rituximab) — CARD11-GOF CLL preferred; "
            "  MALT1 inhibitors (MI-2/JNJ-67856633) investigational — direct CBM target; "
            "  BCL2 inhibitors venetoclax + obinutuzumab standard backbone"
        ),
        "syndrome": "BENTA disease GOF (B-cell Expansion NF-κB) / LOF immunodeficiency / CLL NF-κB survival / ibrutinib less effective GOF",
        "inheritance": "AD LOF/GOF",
        "cll_risk": "CARD11 GOF: BENTA disease → constitutive NF-κB → CLL/B-lymphoma risk; CARD11 LOF: immunodeficiency → B-cell lymphoproliferation → CLL; CARD11-GOF CLL: ibrutinib LESS EFFECTIVE (downstream BTK bypass); venetoclax + MALT1 inhibitors preferred CARD11-GOF; BTK-CARD11-NF-κB survival axis central CLL",
        "pathognomonic": "BENTA: splenomegaly + lymphadenopathy + IgE elevation + atopy + recurrent infections = CARD11-GOF PATHOGNOMONIC; NF-κB constitutive activity in B-cells without BCR stimulation; CLL resistant to ibrutinib with confirmed CARD11-GOF variant = downstream NF-κB bypass; MALT1 cleavage substrates elevated (BCL10/CYLD) in CBM-activated CLL",
        "key_avoid": "Do NOT use ibrutinib as monotherapy in confirmed CARD11-GOF CLL (downstream NF-κB bypass — ibrutinib WON'T block NF-κB in CARD11-GOF); do NOT mistake CARD11-GOF CLL ibrutinib resistance for BTK C481S mutation (different mechanism — check CARD11 before assuming C481S); do NOT skip MALT1 inhibitor trials in CARD11-GOF CLL (specific downstream target); do NOT use corticosteroids long-term BENTA (immunosuppression + risk)",
        "key_rule": "CARD11 CLL key rules: (1) CARD11-GOF CLL = ibrutinib LESS EFFECTIVE (downstream bypass) — switch to venetoclax doublet; (2) BENTA disease GOF = NF-κB constitutive → CLL risk monitored; (3) MALT1 inhibitors (JNJ-67856633) Phase 2 — CARD11-GOF specific target; (4) CARD11 LOF = immunodeficiency + B-lymphoproliferation; (5) test CARD11 at CLL ibrutinib failure (exclude CARD11-GOF bypass); (6) venetoclax + obinutuzumab STANDARD backbone CARD11-GOF CLL.",
        "surveillance": "Annual CBC + CLL immunophenotyping from age 35yr; CARD11 mutation (germline + somatic) at CLL diagnosis and ibrutinib failure; BENTA: quarterly CBC + immunophenotyping + Ig levels from childhood; CT chest/abdomen 2-yearly BENTA; cascade testing first-degree relatives; MALT1 inhibitor trial eligibility (CARD11-GOF); NGS panel (CARD11 + BTK C481S + PLCG2) at relapse",
        "targeted_rx": "Venetoclax BCL2i + obinutuzumab (preferred CARD11-GOF — bypasses BTK-NF-κB); MALT1 inhibitors (JNJ-67856633 / MI-2 — Phase 2; direct CBM target CARD11-GOF); lenalidomide + rituximab R2 (NF-κB modulation); ibrutinib (front-line; test CARD11 first — less effective GOF); acalabrutinib/zanubrutinib (BTKi; same caveat CARD11-GOF); pirtobrutinib (non-covalent BTKi); avoid ibrutinib monotherapy confirmed CARD11-GOF",
    },
    {
        "gene": "CDKN2A",
        "protein": (
            "CDKN2A -- 9p21.3 AD LOF -- 156aa -- "
            "p16-INK4A-17kDa-CDK4-6-Inhibitor-"
            "Cell-Cycle-G1-Arrest-"
            "FAMM-Melanoma-Pancreatic-Lymphoid-"
            "del9p21-Somatic-CLL-5-10pct-"
            "Palbociclib-INACTIVE-p16-Null-OMIM-600160"
        ),
        "locus": "9p21.3",
        "protein_size": (
            "156 aa / ~17 kDa / 9p21.3 CDKN2A CLL molecular context: "
            "STRUCTURE: "
            "  156 aa / ~17 kDa (p16INK4a); CDKN2A locus also encodes p14ARF (alternate reading frame; MDM2 inhibitor → p53 stabilisation); "
            "  4 ankyrin repeats — CDK4/CDK6 binding surface; "
            "  p16INK4a competes with D-cyclins for CDK4/CDK6 → inhibits pRB phosphorylation → G1 arrest; "
            "  p16INK4a LOF → unchecked CDK4/CDK6 → pRB hyperphosphorylation → E2F release → S-phase entry; "
            "FAMM SYNDROME: "
            "  Familial Atypical Multiple Mole Melanoma (FAMM) + pancreatic cancer; "
            "  Germline CDKN2A LOF: melanoma 58-92% lifetime (highest), pancreatic 17-39x HIGHEST; "
            "  Lymphoid malignancy (CLL/NHL) risk elevated in FAMM families; del9p21 somatic CLL 5-10%; "
            "CLL MOLECULAR CONTEXT: "
            "  del9p21 somatic: 5-10% CLL at diagnosis; complex karyotype marker; "
            "  Germline CDKN2A LOF + somatic del9p21 CLL = dual-hit CDKN2A (biallelic); "
            "  p16INK4a-null CLL: CDK4/6 inhibitor palbociclib INACTIVE (no p16 → CDK4/CDK6 already unrestrained = no additional inhibition); "
            "  CDK4/6 inhibitors INEFFECTIVE in p16-null CLL (unlike del13q-RB1-null where CDK4/6i may work); "
            "TREATMENT: "
            "  CDKN2A-CLL: standard ibrutinib/venetoclax; palbociclib INACTIVE if p16-null; "
            "  FAMM: EUS pancreatic surveillance from age 40yr; dermatology annually; "
            "  Venetoclax preferred (BCL2i; active regardless CDKN2A status)"
        ),
        "syndrome": "FAMM syndrome / del9p21 somatic CLL 5-10% / p16INK4a CDK4/6 inhibitor inactive p16-null / pancreatic 17-39x EUS",
        "inheritance": "AD LOF",
        "cll_risk": "CDKN2A FAMM LOF: lymphoid malignancy (CLL/NHL) elevated; del9p21 somatic 5-10% CLL; p16-null CLL → CDK4/6 inhibitor palbociclib INACTIVE; pancreatic 17-39x (EUS from 40yr MANDATORY); complex karyotype CLL marker; venetoclax preferred regardless CDKN2A status",
        "pathognomonic": "FAMM: atypical melanoma >50 moles + family melanoma + pancreatic cancer = CDKN2A PATHOGNOMONIC; del9p21 on CLL FISH + complex karyotype = high-risk CLL; p16 IHC loss in CLL cells = CDKN2A inactivation; palbociclib failure in p16-null CLL (expected — confirmed by IHC); EUS pancreatic surveillance from 40yr MANDATORY in FAMM",
        "key_avoid": "Do NOT use CDK4/6 inhibitors (palbociclib/ribociclib/abemaciclib) in p16INK4a-null CLL (INACTIVE — no therapeutic benefit, avoidable toxicity); do NOT miss pancreatic surveillance in CDKN2A FAMM (17-39x risk — EUS from 40yr MANDATORY); do NOT use CDKN2A IHC alone without FISH/sequencing (p14ARF vs p16INK4a — different coding sequences, same locus); do NOT overestimate RT tolerance (CDKN2A-null cells — checkpoint dysfunction)",
        "key_rule": "CDKN2A CLL key rules: (1) del9p21 CLL → p16 IHC → if p16-null: CDK4/6 inhibitor INACTIVE (palbociclib won't work); (2) venetoclax ACTIVE regardless CDKN2A status; (3) FAMM syndrome: pancreatic EUS from 40yr MANDATORY (17-39x risk); (4) melanoma surveillance annually from 25yr FAMM; (5) FAMM + del9p21 CLL = dual-hit CDKN2A CLL high-risk; (6) cascade test FAMM family CLL panel; ibrutinib/venetoclax standard backbone CDKN2A-CLL.",
        "surveillance": "FAMM: dermoscopy annually from 25yr; pancreatic EUS from 40yr (17-39x risk — 6-monthly if family pancreatic); CBC + CLL panel annually from age 40yr; CLL FISH (del9p21 included) at dx; p16 IHC on CLL cells; NGS panel at diagnosis; cascade testing first-degree relatives; CDKN2A + BRCA2 combined (proximity on 13q vs 9p — unlinked); abdominal MRI pancreas if EUS unavailable",
        "targeted_rx": "Ibrutinib BTKi (CLL first-line); acalabrutinib BTKi; venetoclax BCL2i (ACTIVE regardless CDKN2A — preferred); obinutuzumab + venetoclax; AVOID CDK4/6 inhibitors (palbociclib/ribociclib) p16-null CLL (INACTIVE); EUS pancreatic surveillance + pancreatectomy if warranted; wide-excision melanoma FAMM; aspirin CAPP2 analogue (observational evidence); clinical trial enrollment complex karyotype CLL",
    },
]


def _make_patient(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)

    # Gene-specific CLL/SLL characteristics
    cll_params = {
        "ATM":    dict(age_mu=60, age_sd=10, cll_prob=0.78, del11q_prob=0.65, del17p_prob=0.05,
                       ighv_unmutated_prob=0.65, venetoclax_eligible=True,  hrd_prob=0.08),
        "TP53":   dict(age_mu=57, age_sd=11, cll_prob=0.72, del11q_prob=0.08, del17p_prob=0.80,
                       ighv_unmutated_prob=0.70, venetoclax_eligible=True,  hrd_prob=0.12),
        "POT1":   dict(age_mu=52, age_sd=12, cll_prob=0.68, del11q_prob=0.18, del17p_prob=0.08,
                       ighv_unmutated_prob=0.58, venetoclax_eligible=True,  hrd_prob=0.10),
        "CHEK2":  dict(age_mu=63, age_sd=10, cll_prob=0.62, del11q_prob=0.20, del17p_prob=0.07,
                       ighv_unmutated_prob=0.50, venetoclax_eligible=True,  hrd_prob=0.15),
        "NBN":    dict(age_mu=56, age_sd=12, cll_prob=0.75, del11q_prob=0.25, del17p_prob=0.06,
                       ighv_unmutated_prob=0.60, venetoclax_eligible=True,  hrd_prob=0.20),
        "BRCA2":  dict(age_mu=62, age_sd=11, cll_prob=0.60, del11q_prob=0.15, del17p_prob=0.07,
                       ighv_unmutated_prob=0.48, venetoclax_eligible=True,  hrd_prob=0.55),
        "CARD11": dict(age_mu=55, age_sd=13, cll_prob=0.65, del11q_prob=0.15, del17p_prob=0.05,
                       ighv_unmutated_prob=0.55, venetoclax_eligible=True,  hrd_prob=0.08),
        "CDKN2A": dict(age_mu=61, age_sd=11, cll_prob=0.58, del11q_prob=0.10, del17p_prob=0.08,
                       ighv_unmutated_prob=0.50, venetoclax_eligible=True,  hrd_prob=0.07),
    }
    p = cll_params[gene]

    age               = max(25, min(90, round(rng.gauss(p["age_mu"], p["age_sd"]), 1)))
    has_cll           = rng.random() < p["cll_prob"]
    del11q            = rng.random() < p["del11q_prob"]
    del17p            = rng.random() < p["del17p_prob"]
    ighv_unmutated    = rng.random() < p["ighv_unmutated_prob"]
    hrd_positive      = rng.random() < p["hrd_prob"]
    venetoclax_elig   = p["venetoclax_eligible"] and (del17p or has_cll)
    ibrutinib_elig    = has_cll and gene != "CARD11"  # CARD11-GOF less effective

    tumour_types  = ["CLL", "SLL", "Richter transformation (DLBCL)", "B-cell NHL",
                     "MBL (monoclonal B-cell lymphocytosis)", "WM overlap", "Follicular lymphoma"]
    tumour_probs  = [0.60, 0.12, 0.08, 0.08, 0.06, 0.03, 0.03]
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
        "has_cll":          has_cll,
        "del11q":           del11q,
        "del17p":           del17p,
        "ighv_unmutated":   ighv_unmutated,
        "hrd_positive":     hrd_positive,
        "ibrutinib_elig":   ibrutinib_elig,
        "venetoclax_elig":  venetoclax_elig,
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

    cll_n             = sum(1 for p in all_pts if p["has_cll"])
    del11q_n          = sum(1 for p in all_pts if p["del11q"])
    del17p_n          = sum(1 for p in all_pts if p["del17p"])
    ighv_unmutated_n  = sum(1 for p in all_pts if p["ighv_unmutated"])
    venetoclax_n      = sum(1 for p in all_pts if p["venetoclax_elig"])
    mean_age          = round(sum(p["age_at_dx"] for p in all_pts) / len(all_pts), 1)

    gene_summary = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        cll_gene_n = sum(1 for p in pts if p["has_cll"])
        gene_summary.append({
            "gene":           gene,
            "locus":          gd["locus"],
            "n":              len(pts),
            "cll_n":          cll_gene_n,
            "cll_pct":        round(100 * cll_gene_n / len(pts), 1),
            "mean_age_onset": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "del11q_n":       sum(1 for p in pts if p["del11q"]),
            "del17p_n":       sum(1 for p in pts if p["del17p"]),
            "inheritance":    gd["inheritance"],
            "key_distinctions": [
                gd["cll_risk"][:80],
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
            "cancer_risk":    gd["cll_risk"],
            "syndrome":       gd["syndrome"],
            "pathognomonic":  gd["pathognomonic"],
            "surveillance_key": gd["surveillance"][:160],
            "key_distinctions": [
                gd["cll_risk"][:80],
                gd["key_avoid"][:80],
                gd["key_rule"][:80],
            ],
        })

    return {
        "atlas":                    "Hereditary-CLL-SLL-Predisposition-Atlas",
        "seed_range":               f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":           len(all_pts),
        "genes_n":                  8,
        "cll_total_n":              cll_n,
        "cll_total_pct":            round(100 * cll_n / len(all_pts), 1),
        "mean_age_onset":           mean_age,
        "del11q_n":                 del11q_n,
        "del11q_pct":               round(100 * del11q_n / len(all_pts), 1),
        "del17p_n":                 del17p_n,
        "del17p_pct":               round(100 * del17p_n / len(all_pts), 1),
        "ighv_unmutated_n":         ighv_unmutated_n,
        "ighv_unmutated_pct":       round(100 * ighv_unmutated_n / len(all_pts), 1),
        "venetoclax_eligible_n":    venetoclax_n,
        "venetoclax_eligible_pct":  round(100 * venetoclax_n / len(all_pts), 1),
        "gene_summary":             gene_summary,
        "genes_detail":             genes_detail,
    }


def generate_breakdown() -> dict:
    cohorts = _build_cohorts()
    _gene_pathog = {gd["gene"]: gd["pathognomonic"] for gd in ATLAS_GENES}
    _gene_surv   = {gd["gene"]: gd["surveillance"][:120] for gd in ATLAS_GENES}
    _gene_dist   = {gd["gene"]: [gd["cll_risk"][:80], gd["key_rule"][:80]] for gd in ATLAS_GENES}
    _gene_rule   = {gd["gene"]: gd["key_rule"] for gd in ATLAS_GENES}

    breakdown_array = []
    for i, gd in enumerate(ATLAS_GENES):
        gene = gd["gene"]
        pts  = cohorts[gene]
        cll_n = sum(1 for p in pts if p["has_cll"])
        breakdown_array.append({
            "gene":              gene,
            "locus":             gd["locus"],
            "n":                 len(pts),
            "mean_age_onset":    round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "cll_n":             cll_n,
            "cll_pct":           round(100 * cll_n / len(pts), 1),
            "del11q_n":          sum(1 for p in pts if p["del11q"]),
            "del17p_n":          sum(1 for p in pts if p["del17p"]),
            "ighv_unmutated_n":  sum(1 for p in pts if p["ighv_unmutated"]),
            "hrd_positive_n":    sum(1 for p in pts if p["hrd_positive"]),
            "venetoclax_elig_n": sum(1 for p in pts if p["venetoclax_elig"]),
            "seed":              SEED_BASE + i,
            "key_distinctions":  _gene_dist[gene],
            "pathognomonic":     _gene_pathog[gene],
            "surveillance_key":  _gene_surv[gene],
            "key_rule":          _gene_rule[gene],
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
            "cll_risk":      gd["cll_risk"],
            "pathognomonic": gd["pathognomonic"],
            "key_avoid":     gd["key_avoid"],
            "key_rule":      gd["key_rule"],
            "surveillance":  gd["surveillance"],
            "targeted_rx":   gd["targeted_rx"],
        }

    key_distinctions = [
        "ATM del11q vs TP53 del17p CLL: del11q = high-risk CLL (unmutated IGHV, bulky adenopathy) BUT responds to ibrutinib (BTKi overcomes del11q); del17p = WORST prognosis CLL, chemo-resistant → venetoclax + BTKi REQUIRED; test FISH del11q AND del17p at diagnosis AND relapse",
        "POT1 familial CLL vs sporadic CLL: germline POT1 LOF = MOST SPECIFIC hereditary CLL gene; familial CLL cluster (≥2 first-degree relatives) → test POT1 first; younger onset (~10yr earlier); somatic POT1 (2-3% sporadic) ≠ germline POT1; full germline CLL panel (ATM+TP53+POT1+CHEK2+NBN+BRCA2) mandatory in familial CLL",
        "NBN 657del5 vs ATM CLL risk: NBN 657del5 = B-cell NHL 60x HIGHEST monoallelic (strongest per-allele signal); ATM = CLL 4-5x monoallelic; BOTH Eastern European founders — test NBN 657del5 FIRST in Czech/Slovak/Polish CLL; NBN biallelic (NBS) = RADIOSENSITIVITY ABSOLUTE (same as biallelic ATM)",
        "CHEK2 1100delC vs IVS2+1G>A: 1100delC = Central/Northern European founder (~1:200 Polish/Czech/German); IVS2+1G>A = Ashkenazi Jewish founder (~1:100); both CLL 2-3x monoallelic; CHEK2 + ATM doubly heterozygous = ADDITIVE CLL risk (near-combined); I157T = intermediate (1.5-2x); test both alleles in European CLL families",
        "CARD11-GOF CLL vs BTK C481S ibrutinib resistance: CARD11-GOF CLL = ibrutinib less effective (NF-κB constitutive DOWNSTREAM of BTK — bypass); BTK C481S = ibrutinib covalent bond disrupted (same BTK node); different mechanisms — test CARD11 AND BTK C481S at ibrutinib failure; CARD11-GOF → switch venetoclax + MALT1 inhibitor; C481S → switch pirtobrutinib (non-covalent)",
        "BRCA2 HRD CLL vs CDKN2A p16-null CLL: BRCA2-LOF CLL = HRD positive → PARP inhibitor (olaparib) sensitivity; CDKN2A-null CLL = p16INK4a absent → CDK4/6 inhibitor (palbociclib) INACTIVE; opposite CDK4/6i effect; test HRD (BRCA2) and p16 IHC (CDKN2A) before targeted therapy; venetoclax ACTIVE in both regardless",
        "del13q14 (RB1/miR-15a) vs BRCA2 (13q12.3): del13q14 = MOST COMMON CLL cytogenetic (50% somatic, GOOD prognosis if isolated); BRCA2 at 13q12.3 = ADJACENT but DISTINCT (1.3Mb apart); del13q14 does NOT mean BRCA2 inactivated; BRCA2 germline separate hereditary risk; do NOT conflate del13q14 (somatic CLL) with BRCA2 germline predisposition",
        "Radiosensitivity in hereditary CLL: NBN biallelic (NBS) = RADIOSENSITIVITY ABSOLUTE; ATM biallelic (A-T) = RADIOSENSITIVITY ABSOLUTE; TP53 germline (LFS) = AVOID RADIATION ABSOLUTELY (secondary malignancy); CHEK2/BRCA2 monoallelic = standard RT (mild partial HRD); POT1 monoallelic = caution (telomere dysfunction — increased chromosomal instability with RT); NBN/ATM monoallelic = RT reduce 20-30%",
        "Venetoclax eligibility in hereditary CLL: ALL 8 genes = venetoclax-eligible (BCL2 overexpressed CLL universally); del17p (TP53) = venetoclax PRIORITY (FDA2016 first-line del17p); ibrutinib/venetoclax combination preferred del17p + CARD11-GOF; CDKN2A-null CLL: CDK4/6i INACTIVE but venetoclax ACTIVE; HRD (BRCA2/NBN) + venetoclax = deepest MRD-negative responses",
        "Germline hereditary CLL panel cascade testing: familial CLL → germline panel: ATM (del11q), TP53 (LFS/del17p), POT1 (familial CLL specific), CHEK2 (1100delC + IVS2+1G>A), NBN (657del5 Eastern European), BRCA2 (HBOC + HRD-CLL), CARD11 (BENTA GOF), CDKN2A (FAMM + del9p21); annual CBC from age 30-40yr in at-risk carriers; early CLL diagnosis = treatment advantage",
    ]

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"CLL RISK: {gd['cll_risk']}\n"
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
    print(f"Total patients: {ov['total_patients']}, CLL n: {ov['cll_total_n']} ({ov['cll_total_pct']}%)")
    print(f"del11q: {ov['del11q_n']} ({ov['del11q_pct']}%), del17p: {ov['del17p_n']} ({ov['del17p_pct']}%)")
    print(f"IGHV unmutated: {ov['ighv_unmutated_n']} ({ov['ighv_unmutated_pct']}%), venetoclax-eligible: {ov['venetoclax_eligible_n']} ({ov['venetoclax_eligible_pct']}%)")
    print("=== BREAKDOWN ===")
    br = generate_breakdown()
    for b in br["breakdown"]:
        print(f"  {b['gene']}: CLL {b['cll_pct']}% ({b['cll_n']}/{b['n']}), age {b['mean_age_onset']}yr, del11q {b['del11q_n']}, del17p {b['del17p_n']}")
    print("=== DEFINITIONS ===")
    df = generate_definitions()
    print(f"Genes: {list(df['gene_defs'].keys())}")
