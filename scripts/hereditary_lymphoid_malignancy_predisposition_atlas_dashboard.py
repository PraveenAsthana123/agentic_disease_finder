"""Hereditary Lymphoid Malignancy Predisposition Atlas — 8-Gene Reference
ATM-TP53-CHEK2-PAX5-IKZF1-POT1-RUNX1-STAT3
Germline Predisposition to CLL / ALL / NHL / T-LGL / Li-Fraumeni Haematologic Malignancies
320 patients (8 x 40), seeds 2878-2885.
Endpoints: /api/hereditary-lymphoid-malignancy-predisposition-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD -- 3056aa -- "
            "Ataxia-Telangiectasia-Mutated-Kinase-350kDa-PI3K-Like-Serine-Threonine-Kinase-"
            "DNA-Double-Strand-Break-Sensor-MRN-Complex-Activator-"
            "Homozygous-Ataxia-Telangiectasia-Heterozygous-CLL-MCL-Lymphoma-Predisposition-"
            "OMIM-Gene-607585-Disease-AT-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 350 kDa (Ataxia-Telangiectasia Mutated; PI3K-like serine/threonine kinase (PIKK family); "
            "FAT domain, kinase domain, FATC domain; "
            "activated by DNA double-strand breaks (DSBs) via MRN complex (MRE11-RAD50-NBS1); "
            "phosphorylates H2AX (γH2AX) — DSB foci; p53 Ser15; BRCA1 Ser1387; CHK2 Thr68; "
            "homozygous LOF = Ataxia-Telangiectasia (A-T): progressive cerebellar ataxia, telangiectasias, "
            "combined immunodeficiency (low IgA/IgG/IgE), radiosensitivity, lymphoma/leukaemia; "
            "heterozygous carriers: 2-4× increased CLL risk; mantle cell lymphoma; DLBCL; "
            "del(11q22) (ATM deletion) = most common CLL cytogenetic abnormality (25-35%)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (heterozygous predisposition) / AUTOSOMAL RECESSIVE (homozygous A-T): "
            "  HETEROZYGOUS: ~1-2% population; 2-4× lifetime CLL risk; MCL (high frequency somatic ATM del); "
            "  ~25-35% of CLL show somatic 11q22 deletion (ATM locus); "
            "  Germline heterozygous: moderate cancer predisposition (CLL, lymphoma, breast); "
            "  HOMOZYGOUS: A-T syndrome — cerebellar ataxia onset age 1-2; telangiectasias 3-5y; "
            "    IgA deficiency in 80%; combined immunodeficiency; radiosensitivity (AVOID RT); "
            "    lymphoma/leukaemia risk 100× general population (T-cell lymphoma, ALL, NHL); "
            "    elevated AFP (>2 SD) in >90% A-T patients — useful biomarker; "
            "  SOMATIC ATM del(11q22): confers poor-risk CLL; ibrutinib/acalabrutinib preferred"
        ),
        "disease_category": (
            "ATM-ASSOCIATED LYMPHOID MALIGNANCY SPECTRUM: "
            "CLL: germline het = 2-4× risk; somatic del(11q22) = 25-35% CLL; "
            "  del(11q22) CLL: aggressive; bulky lymphadenopathy; shorter TFS; ibrutinib era improved OS; "
            "MCL: somatic ATM biallelic in ~50% MCL; germline het MCL reported; "
            "DLBCL: germline het modest risk elevation; "
            "T-ALL/NHL: A-T homozygotes (childhood/adolescence); T-cell clonal proliferation; "
            "B-ALL: less common in A-T than T-cell; "
            "HODGKIN LYMPHOMA: 3-4× risk in het carriers (observational data); "
            "BREAST CANCER: het carriers 3× risk (BRCA-type — germline testing offered); "
            "CLINICAL ALERT: A-T patients RADIOSENSITIVE — RT contraindicated unless no alternative; "
            "  Even het carriers: moderate RT sensitivity; reduce RT dose if germline ATM confirmed"
        ),
        "disease_pathway": (
            "ATM — DNA DAMAGE RESPONSE (DDR) PATHWAY: "
            "NORMAL: DSB → MRN complex recruits ATM → ATM autophosphorylation (Ser1981) → kinase activated; "
            "  ATM → γH2AX (chromatin marking); ATM → CHK2 → CDC25 inhibition → G2/M arrest; "
            "  ATM → p53 → CDKN1A (p21) → G1 arrest / apoptosis; "
            "  ATM → BRCA1 → homologous recombination repair (HRR); "
            "  ATM → NF-κB in lymphoid cells; "
            "HETEROZYGOUS LOF: "
            "  50% ATM activity — sufficient for most DSBs; "
            "  Impaired response to replication stress in B-lymphocyte VDJ recombination/CSR; "
            "  CSR generates DSBs; ATM haploinsufficiency → incomplete DSB repair → translocation risk; "
            "  t(14;18), t(11;14) translocation rate elevated; "
            "HOMOZYGOUS LOF (A-T): "
            "  Absent DSB signalling → replication catastrophe; chromosomal instability; "
            "  Purkinje cell hypersensitivity to DSBs (high DDR demand) → progressive cerebellar ataxia; "
            "  Thymic T-cell TCR recombination fails → T-cell immunodeficiency; "
            "  B-cell CSR failure → IgA/IgG/IgE deficiency; "
            "  Chromosomal instability → T-cell clone expansion → T-lymphoma/ALL"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC: "
            "  1. PROGRESSIVE CEREBELLAR ATAXIA onset age 1-2 + TELANGIECTASIAS (ears, conjunctiva, skin) "
            "     + IgA DEFICIENCY + ELEVATED AFP: PATHOGNOMONIC A-T; "
            "     no other condition produces all four simultaneously; "
            "  2. del(11q22) on CLL FISH: ATM locus deletion; most common CLL chromosomal abnormality; "
            "     associated with bulky adenopathy + short time-to-first-treatment; "
            "  3. EXTREME RADIOSENSITIVITY in cancer patient: A-T or ATM het — "
            "     routine RT → severe skin/mucosa toxicity; chromosomal breakage assay + germline ATM; "
            "  4. FAMILY HISTORY CLL + YOUNG-ONSET: germline ATM testing warranted; "
            "DDx homozygous: Friedreich ataxia (frataxin, no telangiectasias, no immunodeficiency), "
            "NBS (11q23 NBS1 — no ataxia, microcephaly), ATLD (MRE11 — milder A-T); "
            "DDx somatic del(11q22): del(13q14) CLL (favourable), del(17p13) TP53 (ultra-poor risk)"
        ),
        "treatment": (
            "ATM-ASSOCIATED LYMPHOID MALIGNANCY TREATMENT: "
            "CLL WITH del(11q22): ibrutinib, acalabrutinib (BTK inhibitors) — preferred over chemo; "
            "  FCR (fludarabine-cyclophosphamide-rituximab): avoid in del(11q22) — poor outcome; "
            "  Venetoclax + obinutuzumab: alternative; "
            "  PARP inhibitor sensitivity (preclinical): olaparib — DSB repair dependency; "
            "A-T MANAGEMENT: "
            "  No disease-modifying therapy; supportive care; "
            "  Immunoglobulin replacement if IgG < 400 mg/dL or recurrent infections; "
            "  AVOID RT (radiosensitivity): use with extreme caution; modified protocols only; "
            "  Chemotherapy: reduce alkylating agent dose (crosslink sensitivity); "
            "  Lymphoma/ALL in A-T: reduced-intensity protocols; HSCT explored; "
            "  Physical therapy: maintain cerebellar function (cannot halt); "
            "  AFP monitoring: progressive rise suggests cerebellar degeneration; "
            "GERMLINE HET: cascade family testing; colonoscopy/mammography; no proven prevention"
        ),
        "seed": 2878,
        "n_patients": 40,
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD -- 393aa -- "
            "Tumour-Protein-p53-43kDa-Transcription-Factor-Guardian-of-the-Genome-"
            "Li-Fraumeni-Syndrome-LFS-Germline-ALL-NHL-Hodgkin-Predisposition-"
            "OMIM-Gene-191170-Disease-LFS-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa (tumour protein p53; tetrameric transcription factor; "
            "domains: N-terminal transactivation (TAD1 1-40, TAD2 40-67), "
            "proline-rich region, sequence-specific DNA-binding domain (DBD 102-292), "
            "tetramerisation domain, C-terminal regulatory domain; "
            "DBD: hotspot mutations R175H, G245S, R248Q/W, R249S, R273C/H, R282W (gain-of-function GOF); "
            "activates CDKN1A (p21), PUMA, NOXA, BAX, MDM2; "
            "mediates G1 arrest, apoptosis, senescence after DNA damage; "
            "LOF → loss of cell-cycle checkpoints → oncogenesis; "
            "GOF mutations: oncogenic (TP53 GOF AML = ultra-poor risk; TP53-mutated AML/MDS ivosidenib? — no; venetoclax/azacitidine + etoposide)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — HAPLOINSUFFICIENCY + DOMINANT NEGATIVE / GOF: "
            "  LI-FRAUMENI SYNDROME (LFS): germline TP53 pathogenic variant; "
            "  Lifetime cancer risk >90% by age 60 (males <female); "
            "  HEMATOLOGIC: ALL (15-20% of LFS cancers in children); NHL; Hodgkin; AML; "
            "  Core cancers: early-onset breast cancer, osteosarcoma, adrenocortical carcinoma, brain tumours; "
            "  Chompret criteria (2015): germline TP53 testing indicated; "
            "  Very early onset: breast <31y, sarcoma <46y, adrenocortical carcinoma <18y; "
            "  RT AVOIDANCE: germline TP53 mutation = severe radiosensitivity (similar to A-T); "
            "    RT → secondary malignancies in LFS; avoid if possible; "
            "  TP53 del(17p13) in CLL/MDS: somatic — ultra-poor risk; venetoclax or ibrutinib; "
            "  TP53 biallelic in AML: CR rates <30%; HSCT only potentially curative"
        ),
        "disease_category": (
            "TP53-ASSOCIATED HEMATOLOGIC MALIGNANCY: "
            "ALL: most common hematologic malignancy in LFS children; B-ALL > T-ALL; "
            "  TP53 LOF ALL: poor prognosis; hypodiploidy (<44 chromosomes) in 90% TP53-mutated ALL; "
            "  Hypodiploid ALL: deep hypodiploid (<44) near-haploid / low-hypodiploid (32-39); "
            "    Deep hypodiploid ALL = essentially diagnostic of germline TP53 in children; "
            "NHL / DLBCL: somatic TP53 mutations in 20% DLBCL; germline carriers elevated; "
            "CLL: del(17p13) somatic = 5-8% CLL; ultra-poor risk; BTK inhibitors preferred; "
            "  Somatic TP53 + del(17p13) biallelic: chemo-refractory; ibrutinib/venetoclax; "
            "AML: TP53-mutated AML (complex karyotype, therapy-related): venetoclax/azacitidine; "
            "  APR-246 (eprenetapopt) — refolding p53 — clinical trials; "
            "HODGKIN: reported in LFS families"
        ),
        "disease_pathway": (
            "TP53 — GUARDIAN OF GENOME / STRESS RESPONSE: "
            "NORMAL STRESS RESPONSE: DNA damage/replication stress → ATM/ATR → MDM2 phosphorylation → "
            "  p53 stabilisation (escapes MDM2 degradation) → tetramer binds RE (5'-RRRCWWGYYY-3'); "
            "  Target genes: CDKN1A (p21, G1 arrest), GADD45 (G2 arrest), BAX/PUMA/NOXA (apoptosis), "
            "  MDM2 (negative feedback), p21 (senescence), LKB1/PTEN (tumour suppression); "
            "  Ribosomal stress: RPL5/RPL11 free ribosomal proteins inhibit MDM2 → p53 activation; "
            "LOF CONSEQUENCES: "
            "  No G1 checkpoint → cells with DSBs replicate → chromosomal instability; "
            "  No apoptosis induction → malignant clones survive; "
            "  Haploinsufficiency in LFS: one allele sufficient for residual p53 activity; "
            "  Second hit (LOH, somatic mutation) in tumour → complete p53 loss; "
            "GOF MUTATIONS: "
            "  Gain-of-function TP53 (R175H, R248W): bind and inhibit p63/p73; "
            "  Transcription of pro-survival genes (MCL1, MDR1); "
            "  GOF → worse prognosis than simple LOF; dominant negative effect"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — TP53/LFS: "
            "  1. HYPODIPLOID ALL (<44 chromosomes) IN A CHILD: germline TP53 probability >90%; "
            "     near-haploid (23-29) or low-hypodiploid (30-39) ALL → reflexly test germline TP53; "
            "  2. ADRENOCORTICAL CARCINOMA IN CHILD + family history: LFS = germline TP53; "
            "     adrenocortical carcinoma in child <18y = ~50-80% LFS in developed world; "
            "  3. MULTIPLE PRIMARY CANCERS (sarcoma + breast + brain + adrenocortical): LFS pedigree; "
            "  4. del(17p13) in CLL + TP53 mutation: biallelic = chemo-refractory; "
            "     FISH + sequencing together for TP53 status in all CLL (ELN mandatory); "
            "  5. RT-associated second malignancy in LFS field: avoid RT in germline TP53; "
            "DDx LFS: Li-Fraumeni-like (CHEK2, BRCA1) — overlapping cancer spectrum but TP53 more severe; "
            "DDx somatic TP53 CLL: del(17p) ± mutation; ultra-poor regardless of source"
        ),
        "treatment": (
            "TP53-ASSOCIATED HEMATOLOGIC MALIGNANCY TREATMENT: "
            "ALL WITH GERMLINE TP53: standard chemotherapy regimens (COG/BFM); "
            "  Avoid RT if possible; radiosensitivity risk; "
            "  MRD-guided intensification; HSCT for high-risk TP53-mutated ALL; "
            "CLL WITH del(17p)/TP53: ibrutinib or acalabrutinib (BTK inhibitor) — FIRST LINE; "
            "  Venetoclax + obinutuzumab: alternative; "
            "  Chemoimmunotherapy (FCR, BR): AVOID; ineffective and toxic; "
            "AML WITH TP53: venetoclax + azacitidine ± etoporide; "
            "  APR-246 (eprenetapopt) refolding trials ongoing; "
            "  Magrolimab (anti-CD47) + azacitidine: promising in TP53-mutated MDS/AML; "
            "  HSCT in CR1: consider; poor outcomes but only curative option; "
            "LFS SURVEILLANCE: whole-body MRI annually (Toronto Protocol); "
            "  Breast: annual MRI from age 20-25; avoid mammography radiation; "
            "  Colonoscopy; brain MRI; CBC annually; "
            "GENETIC COUNSELLING: autosomal dominant; 50% offspring risk; "
            "  Pre-implantation genetic testing (PGT) available"
        ),
        "seed": 2879,
        "n_patients": 40,
    },
    {
        "gene": "CHEK2",
        "protein": (
            "CHEK2 -- 22q12.1 AD -- 543aa -- "
            "Checkpoint-Kinase-2-CHK2-60kDa-Serine-Threonine-Kinase-"
            "ATM-Substrate-DDR-Amplifier-"
            "I157T-Eastern-European-Founder-1100delC-NW-European-Founder-CLL-Lymphoma-Breast-Predisposition-"
            "OMIM-Gene-604373-Disease-CLL-Predisposition"
        ),
        "locus": "22q12.1",
        "protein_size": (
            "543 aa / 60 kDa (CHK2 checkpoint serine/threonine kinase; "
            "domains: SQ/TQ cluster domain (SCD), forkhead-associated (FHA) domain, kinase domain; "
            "activated by ATM phosphorylation at Thr68 (FHA domain) → CHK2 homodimerisation → autophosphorylation → active; "
            "key substrates: p53 Ser20 (stabilisation), CDC25A/C (cell-cycle arrest), BRCA1; "
            "founder variants: p.I157T (missense, common in Eastern Europe/Poland ~5% population), "
            "p.1100delC (frameshift, NW Europe 1-2%), p.IVS2+1G>A (splice); "
            "I157T: moderate risk (~2× CLL/lymphoma); 1100delC: high risk (~4× breast cancer); "
            "CHEK2 biallelic: more severe cancer predisposition (Li-Fraumeni-like)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — HAPLOINSUFFICIENCY: "
            "  Common moderate-penetrance cancer predisposition gene; "
            "  CHEK2 I157T: most common variant worldwide; ~2× CLL and ~4× colorectal cancer; "
            "  CHEK2 1100delC: ~4-5× breast cancer; ~2-3× CLL/NHL; "
            "  CLL: CHEK2 germline variants in ~4-5% CLL patients (enriched vs. population); "
            "  NHL (follicular, DLBCL): modest risk elevation; "
            "  PROSTATE CANCER: 2× in 1100delC carriers; "
            "  BREAST: 1100delC = 28-37% lifetime breast cancer risk; "
            "  COLORECTAL: I157T = 4× risk; "
            "  BIALLELIC CHEK2: Li-Fraumeni-like — multiple primaries; "
            "  Somatic CHEK2 mutations: in CLL (~10%); DLBCL (~5%)"
        ),
        "disease_category": (
            "CHEK2-ASSOCIATED LYMPHOID MALIGNANCY SPECTRUM: "
            "CLL: CHEK2 germline = 2-3× risk; somatic CHEK2 mutations in ~10% CLL; "
            "  CHEK2-mutated CLL: often IgHV unmutated; intermediate prognosis; "
            "NHL: follicular lymphoma, DLBCL — modest risk; "
            "B-ALL: rare germline CHEK2 association (indirect); "
            "MANAGEMENT OF CHEK2 VARIANT CARRIERS: "
            "  Moderate-risk counselling (not high-risk BRCA-equivalent); "
            "  Breast: annual mammography from 40y; consider MRI if additional risk factors; "
            "  Colonoscopy: from 40y; "
            "  No proven hematologic surveillance protocol for CLL risk; "
            "  Cascade testing offered to first-degree relatives; "
            "CLINICAL NOTE: CHEK2 I157T + BRCA2/ATM: compound risk; "
            "  Many CLL panel tests now include CHEK2 routinely"
        ),
        "disease_pathway": (
            "CHEK2 — DDR AMPLIFICATION: "
            "NORMAL: DSB → ATM activates CHK2 (Thr68 phosphorylation) → CHK2 phosphorylates: "
            "  p53 Ser20 (MDM2 dissociation → p53 stabilisation → apoptosis/arrest); "
            "  CDC25A/C (proteasomal degradation → S/G2 arrest); "
            "  BRCA1 (Ser988 → HRR promotion); "
            "  PML (tumour suppressor localisation); "
            "  CHK2 amplifies ATM signal for full DDR response; "
            "HAPLOINSUFFICIENCY (CHEK2 het): "
            "  Attenuated CHK2 activity → p53 less efficiently stabilised; "
            "  G2/M checkpoint partially impaired; "
            "  DNA-damaged cells more likely to escape apoptosis; "
            "  B-lymphocytes: VDJ + CSR generate DSBs physiologically; "
            "  CHK2 het → incomplete repair → translocation accumulation; "
            "I157T MECHANISM: "
            "  Missense in FHA domain — impairs CHK2 dimerisation; "
            "  Kinase activity reduced ~60%; "
            "  Cell cycle arrest competence reduced; "
            "1100delC MECHANISM: "
            "  Frameshift → truncated unstable protein; "
            "  Near-complete LOF; stronger cancer predisposition"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — CHEK2: "
            "  1. CHEK2 not truly 'pathognomonic' for lymphoid malignancy (moderate-risk, common): "
            "     most CHEK2 carriers never develop CLL/lymphoma; "
            "  2. CLL IN PATIENT WITH EASTERN EUROPEAN ANCESTRY + FAMILY HISTORY COLORECTAL/BREAST: "
            "     I157T screen warranted; compound CHEK2 + ATM/BRCA2 → very high CLL risk; "
            "  3. BILATERAL BREAST CANCER <50y + CLL: CHEK2 1100delC carrier screen; "
            "  4. MULTIPLE CHEK2-SPECTRUM CANCERS (breast + colorectal + prostate + CLL) in pedigree: "
            "     CHEK2 most likely explanation; panels include CHEK2 routinely; "
            "DDx: BRCA2 (higher penetrance breast/ovarian/pancreatic); ATM het (CLL + breast overlap); "
            "  PALB2 (breast predominant); "
            "  CHEK2 uniquely common in Eastern European populations (I157T)"
        ),
        "treatment": (
            "CHEK2-ASSOCIATED MALIGNANCY TREATMENT: "
            "CLL WITH CHEK2 GERMLINE/SOMATIC: treat CLL by standard risk stratification; "
            "  CHEK2 alone does not mandate specific CLL therapy divergence; "
            "  If co-present del(11q22) ATM: ibrutinib/venetoclax preferred; "
            "BREAST CANCER IN CHEK2 CARRIERS: "
            "  CHEK2 1100delC: surveillance MRI from age 35-40; bilateral mastectomy option; "
            "  CHEK2 I157T: moderate risk — enhanced imaging; individualize; "
            "COLORECTAL: colonoscopy from age 35-40 in I157T carriers; "
            "SURVEILLANCE PROTOCOL: "
            "  Clinical genetics review; "
            "  Breast: annual mammography ± MRI; "
            "  Colorectal: colonoscopy every 3-5y; "
            "  No proven benefit from hematologic surveillance specifically; "
            "FAMILY CASCADE: first-degree relatives offered CHEK2 testing; "
            "  Population frequency of I157T in Poland/Baltic: ~5% — cascade high volume"
        ),
        "seed": 2880,
        "n_patients": 40,
    },
    {
        "gene": "PAX5",
        "protein": (
            "PAX5 -- 9p13.2 AD -- 391aa -- "
            "Paired-Box-5-Transcription-Factor-47kDa-B-Cell-Identity-Master-TF-"
            "B-Lymphocyte-Commitment-Maintenance-"
            "Familial-B-ALL-Predisposition-Childhood-ALL-Germline-"
            "OMIM-Gene-167414-Disease-ALL-Predisposition-615545"
        ),
        "locus": "9p13.2",
        "protein_size": (
            "391 aa / 47 kDa (Paired box transcription factor 5 / BSAP (B-cell-specific activator protein); "
            "domains: N-terminal paired box (PD, sequence-specific DNA binding), octapeptide, "
            "partial homeodomain, transactivation domain, C-terminal inhibitory domain; "
            "master regulator of B-cell identity — activates B-cell programme at pro-B stage; "
            "maintains B-cell identity by suppressing non-B lineage genes; "
            "represses NOTCH1, FLT3, and myeloid-specific promoters; "
            "targets: CD19, CD79a, BLNK, FCRG2B, RAG1/2, mb-1; "
            "LOF germline variants: p.G183S (most recurrent familial B-ALL), p.R38H, p.G183E; "
            "dominant negative mechanism — mutant PAX5 blocks normal PAX5 target gene activation; "
            "haploinsufficiency at pro-B stage → arrested differentiation susceptible to second hit"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — HAPLOINSUFFICIENCY / DOMINANT NEGATIVE: "
            "  FAMILIAL B-ALL: PAX5 germline variants (G183S most recurrent) → "
            "  multiple family members with B-ALL; penetrance incomplete (~20-30% lifetime ALL); "
            "  Onset: predominantly childhood/adolescence; some adult cases; "
            "  PAX5 somatic mutations: most common somatic alteration in B-ALL (~30%); "
            "  Germline: rare (~1-2% familial ALL); distinguished from somatic by buccal DNA testing; "
            "  Genomic context: often co-occurs with deletion 9p13 (PAX5 locus deletion) in sporadic ALL; "
            "  Second hit in tumour: frequently somatic PAX5 point mutation on second allele; "
            "    OR deletion 9p13.2 → biallelic PAX5 loss in leukaemia; "
            "  Haploinsufficiency model: germline LOF → impaired pro-B commitment → "
            "    cells stalled at pro-B, unable to complete VDJ recombination correctly → "
            "    AID error → secondary chromosomal lesion → ALL"
        ),
        "disease_category": (
            "PAX5-ASSOCIATED B-ALL PREDISPOSITION: "
            "B-ALL: germline PAX5 predisposes to B-cell precursor ALL (B-ALL); "
            "  PAX5 G183S familial ALL: multiple affected family members; "
            "  Age: childhood to young adult; "
            "  Cytogenetics: high hyperdiploid or ETV6-RUNX1 in some; others normal karyotype; "
            "  Response: generally standard ALL protocol; "
            "CLL: rare association (PAX5 expressed in CLL cells; somatic PAX5 mutations in CLL); "
            "  Germline PAX5 CLL: occasional case reports; not well quantified; "
            "NHL: marginal zone lymphoma, follicular lymphoma — PAX5 somatic mutations common; "
            "  Germline contribution: uncertain; "
            "MANAGEMENT: "
            "  Germline PAX5 confirmed → cascade testing; siblings offered testing before HSCT donation; "
            "  No proven prophylactic strategy; "
            "  Bone marrow surveillance interval not established — no guideline-level evidence; "
            "  Treat ALL as per standard risk stratification"
        ),
        "disease_pathway": (
            "PAX5 — B-CELL COMMITMENT AND IDENTITY: "
            "NORMAL HAEMATOPOIESIS: CLP (common lymphoid progenitor) → pro-B: PAX5 expression initiates; "
            "  PAX5 activates: EBF1, CD19, BLNK, mb-1 (Ig alpha), RAG1/RAG2; "
            "  PAX5 represses: C/EBP alpha (myeloid), NOTCH1 (T-cell), FLT3 (myeloid); "
            "  PAX5 establishes B-cell commitment: pro-B → pre-B → immature B → mature B; "
            "  Mature B maintains PAX5 expression; PAX5 loss → plasma cell differentiation; "
            "HAPLOINSUFFICIENCY MECHANISM: "
            "  PAX5 het → 50% PAX5 protein → marginally impaired pro-B commitment; "
            "  Pro-B cells: inefficient VDJ recombination → unusual junctions; "
            "  Increased AID (activation-induced cytidine deaminase) activity → off-target mutations; "
            "DOMINANT NEGATIVE (G183S): "
            "  G183S PAX5 binds target DNA but fails to activate transcription; "
            "  Wild-type PAX5 titrated: effective haploinsufficiency even in het; "
            "  Second hit: somatic deletion 9p13.2 or PAX5 somatic mutation → biallelic loss → ALL"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — PAX5 B-ALL: "
            "  1. MULTIPLE FAMILY MEMBERS WITH B-ALL ACROSS GENERATIONS: "
            "     familial ALL pedigree → germline PAX5 (or IKZF1) sequencing mandatory; "
            "  2. PAX5 G183S ON BUCCAL/GERMLINE TESTING (somatic-origin excluded): "
            "     confirmed germline → familial B-ALL syndrome; "
            "  3. B-ALL PATIENT WITH BUCCAL PAX5 MUTATION = somatic testing: "
            "     tumour PAX5 deletion 9p13 is very common (somatic); distinguish germline from somatic; "
            "     buccal DNA removes tumour-derived DNA; "
            "  4. SIBLING OF PAX5-GERMLINE B-ALL PATIENT: "
            "     donor suitability assessment — test sibling before HSCT donation; "
            "DDx: IKZF1 germline (also familial B-ALL; B-lymphopenia); "
            "  ETV6 germline (thrombocytopenia + ALL + AML); "
            "  Sporadic ALL: somatic PAX5 del9p most common but not familial"
        ),
        "treatment": (
            "PAX5 GERMLINE B-ALL TREATMENT: "
            "ALL TREATMENT: per standard COG/BFM ALL protocol (risk-stratified); "
            "  MRD at day 15/EOI determines intensification; "
            "  B-ALL: multi-agent induction (DVPAS or similar); "
            "  COG AALL0434/0434 or similar for risk assignment; "
            "  High-risk: HSCT considered in CR1; "
            "DONOR ASSESSMENT: "
            "  Siblings must be tested for germline PAX5 before donation; "
            "  PAX5 germline carrier sibling: exclude as donor (carry predisposition); "
            "  MUD preferred if sibling untested or confirmed carrier; "
            "TARGETED THERAPY: "
            "  Ph-like ALL: may co-occur; tyrosine kinase inhibitor (dasatinib/imatinib) if applicable; "
            "  Blinatumomab (CD3xCD19): applicable if PAX5-mutated ALL (CD19+); "
            "  Inotuzumab ozogamicin (anti-CD22): if CD22+; "
            "FAMILY COUNSELLING: "
            "  Incomplete penetrance — not all carriers develop ALL; "
            "  Cascade testing siblings/parents; vigilance for cytopenias; "
            "  Annual CBC in known germline carriers — no guideline-level protocol yet"
        ),
        "seed": 2881,
        "n_patients": 40,
    },
    {
        "gene": "IKZF1",
        "protein": (
            "IKZF1 -- 7p12.2 AD/AR -- 519aa -- "
            "Ikaros-Zinc-Finger-1-58kDa-Hematopoietic-Transcription-Factor-"
            "B-Lymphocyte-Development-Regulator-C2H2-Zinc-Fingers-"
            "Germline-B-ALL-Predisposition-IKZF1-Haploinsufficiency-"
            "OMIM-Gene-603023-Disease-ALL-Predisposition"
        ),
        "locus": "7p12.2",
        "protein_size": (
            "519 aa / 58 kDa (Ikaros zinc finger protein 1; IKZF1/Ikaros; C2H2 zinc finger TF family; "
            "four N-terminal DNA-binding zinc fingers (ZF1-4) binding GGGAA core motif; "
            "two C-terminal dimerisation zinc fingers (ZF5-6); "
            "isoforms: IK1 (full length, tumour suppressor); IK6 (dominant negative, lacks ZF1-4, lacks DNA binding); "
            "IK6 = loss of DNA binding → dominant negative (sequesters IK1 via dimerisation); "
            "IKZF1 activates: VPREB1, IGLL1, RAG2, TdT, CD2, CD3, CD4, CD8; "
            "represses proliferation-associated genes (CDK4, CDC25); "
            "chromatin remodelling at pericentromeric heterochromatin"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (heterozygous) — HAPLOINSUFFICIENCY + DOMINANT NEGATIVE: "
            "  GERMLINE IKZF1 DELETIONS/VARIANTS: "
            "    Heterozygous germline deletion of IKZF1 (del7p12.2) → B-lymphopenia; "
            "    Heterozygous LOF point mutations → B-ALL predisposition; "
            "    SOMATIC: most IKZF1 alterations are somatic in B-ALL (~15-30% ALL); "
            "    Germline: rarer; familial B-ALL; "
            "  B-LYMPHOPENIA PHENOTYPE: "
            "    Germline IKZF1 het in children → low B-cell counts; recurrent sinopulmonary infections; "
            "    IgG deficiency; "
            "    Autosomal dominant B-cell immunodeficiency (CVID-like in some families); "
            "  Ph-like ALL: IKZF1 deletion very common somatic event in Ph-like ALL (ABL-class fusions); "
            "  ALL predisposition: familial cases; incomplete penetrance"
        ),
        "disease_category": (
            "IKZF1-ASSOCIATED B-CELL IMMUNODEFICIENCY AND B-ALL: "
            "B-LYMPHOPENIA / CVID-LIKE IMMUNODEFICIENCY: "
            "  Germline het IKZF1 → B-cell differentiation block at pro-B / common precursor; "
            "  Recurrent sinopulmonary infections (otitis, sinusitis, pneumonia); "
            "  Reduced IgG, IgA, IgM; "
            "  Splenomegaly; organomegaly; "
            "  Immunoglobulin replacement therapy; "
            "B-ALL: germline het + somatic second hit (deletion of remaining IKZF1 allele in tumour); "
            "  Familial B-ALL with IKZF1 germline; "
            "  Somatic IKZF1 deletion in ALL: 15% B-ALL; very common in Ph+ ALL and Ph-like ALL; "
            "  IK6 dominant-negative isoform: somatic; constitutively nuclear; "
            "PROGNOSIS IMPACT: "
            "  Somatic IKZF1 deletion in ALL: associated with poor prognosis (IKZF1plus profile); "
            "  'IKZF1plus': IKZF1 del + CDKN2A/B del + PAX5 del/mut + no ERG del → ultra-high risk"
        ),
        "disease_pathway": (
            "IKZF1 — LYMPHOID COMMITMENT AND CELL-CYCLE CONTROL: "
            "NORMAL HAEMATOPOIESIS: CLP → B-lymphoid: IKZF1 activates B-lineage programme; "
            "  Pro-B → pre-B: IKZF1 regulates VDJ recombination (RAG1/2 activation); "
            "  IKZF1 localises to pericentromeric heterochromatin (HP1-alpha complex); "
            "  IK1: tumour suppressor — represses CDK4 (G1 arrest); promotes p27 Kip1 expression; "
            "  IKZF1 required for T-cell development (thymic selection); "
            "LOF / HAPLOINSUFFICIENCY: "
            "  Reduced IKZF1 → pro-B to pre-B transition impaired; "
            "  Reduced VDJ recombination efficiency; "
            "  CDK4 upregulation → G1/S checkpoint bypass → accelerated proliferation; "
            "IK6 DOMINANT NEGATIVE: "
            "  IK6 lacks DNA-binding ZF1-4; "
            "  IK6 sequesters IK1 (and IK2/IK3) via C-terminal dimerisation → nuclear localisation lost; "
            "  Full loss of IKZF1 target gene activation; "
            "  IK6 = near-complete functional knock-out despite het genomic status"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — IKZF1: "
            "  1. PERSISTENT B-LYMPHOPENIA IN CHILD WITH RECURRENT SINOPULMONARY INFECTIONS + IKZF1 DELETION: "
            "     IKZF1-associated immunodeficiency; distinguish from CVID (BTK, IGHM, CD79A); "
            "  2. FAMILIAL B-ALL WITH B-LYMPHOPENIA IN NON-AFFECTED CARRIERS: "
            "     germline IKZF1 het; low B-cell count in family members without ALL; "
            "  3. B-ALL + IK6 ISOFORM ON RT-PCR / WESTERN: "
            "     dominant negative IK6 = IKZF1 deletion exons 4-7; "
            "     RT-PCR or splice-specific sequencing confirms IK6; "
            "  4. Ph-like ALL + IKZF1 deletion: not diagnostic of germline; somatic; "
            "     Ph-like: ABL1/2/PDGFRB fusions; dasatinib responsive; IKZF1 del co-occurring; "
            "DDx: PAX5 germline (B-ALL but no B-lymphopenia in carrier); "
            "  BTK (XLA — complete B absence, not B-lymphopenia); "
            "  CVID (late-onset B-lymphopenia, normal B-cells childhood)"
        ),
        "treatment": (
            "IKZF1-ASSOCIATED B-CELL IMMUNODEFICIENCY / B-ALL TREATMENT: "
            "B-LYMPHOPENIA / IMMUNODEFICIENCY: "
            "  Immunoglobulin replacement (IVIg/SCIg) if IgG <400 mg/dL or recurrent infections; "
            "  Monthly IVIg or biweekly SCIg; "
            "  Antibiotic prophylaxis for recurrent sinopulmonary disease; "
            "  Avoid live vaccines; "
            "B-ALL TREATMENT: "
            "  Standard ALL chemotherapy (COG/BFM); risk-adapted intensification; "
            "  IKZF1-deletion ALL: historically poor prognosis; MRD-guided escalation; "
            "  Ph-like ALL: add TKI (dasatinib if ABL-class fusion identified); "
            "    JAK1/2 inhibitor (ruxolitinib) if CRLF2-JAK mutation; "
            "  IKZF1plus profile: HSCT in CR1 recommended by many COG/BFM guidelines; "
            "DONOR SELECTION: "
            "  Germline IKZF1 family — siblings offered testing; low B-cell count in sibling = carrier; "
            "  Carrier siblings: not ideal BM donors (carry predisposition); "
            "HAEMATOPOIETIC STEM CELL TRANSPLANT: "
            "  Consider in CR1 for IKZF1plus profile or germline IKZF1 + high-risk ALL; "
            "  Corrects haematopoietic IKZF1 haploinsufficiency (somatic)"
        ),
        "seed": 2882,
        "n_patients": 40,
    },
    {
        "gene": "POT1",
        "protein": (
            "POT1 -- 7q31.33 AD -- 634aa -- "
            "Protection-of-Telomeres-1-71kDa-Shelterin-Complex-Telomere-Cap-OB-Fold-"
            "Single-Stranded-Telomere-DNA-Binding-"
            "Familial-CLL-Melanoma-Angiosarcoma-Glioma-"
            "OMIM-Gene-606478-Disease-CLL-Predisposition"
        ),
        "locus": "7q31.33",
        "protein_size": (
            "634 aa / 71 kDa (Protection of Telomeres 1; shelterin component; "
            "OB-fold 1 + OB-fold 2 for single-stranded telomeric DNA (TTAGGG 3' overhang) binding; "
            "shelterin complex: TRF1-TRF2-RAP1-TIN2-TPP1-POT1; "
            "POT1 interacts with TPP1 (POT1-TPP1 heterodimer); "
            "functions: (1) cap 3' ss-TTAGGG overhang → prevent RPA/ATR activation; "
            "  (2) suppress ALT (alternative lengthening of telomeres); "
            "  (3) regulate telomerase access (POT1-TPP1 processivity factor); "
            "LOF POT1 → uncapped telomere → RPA-ATR-CHK1 activation → chromosomal instability; "
            "germline variants: missense in OB-fold domains (Y36C, Y89C, A76V, Q94E, I78T); "
            "clonal haematopoiesis of indeterminate potential (CHIP) phenotype at telomere level"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — HAPLOINSUFFICIENCY: "
            "  FAMILIAL CLL: POT1 germline variants in ~3-5% familial CLL families; "
            "  Increased risk (~2-8× general population) depending on variant; "
            "  CLL onset: earlier than sporadic (~5-10 years younger); "
            "  Associated malignancies: melanoma (cutaneous); angiosarcoma; glioma (Li-Fraumeni-like); "
            "  Cardiac angiosarcoma: rare but characteristic association; "
            "  Thyroid cancer: reported; "
            "  Penetrance variable; many carriers do not develop malignancy; "
            "  Male predominance in CLL (as in sporadic CLL); "
            "  SOMATIC POT1: recurrent in cutaneous melanoma (~5%); angiosarcoma; glioma; mantle cell lymphoma; "
            "  Telomere length: POT1 het carriers often have shorter telomeres"
        ),
        "disease_category": (
            "POT1-ASSOCIATED FAMILIAL CLL AND MULTI-CANCER PREDISPOSITION: "
            "CLL: most prevalent cancer in POT1 germline families; "
            "  Earlier onset; classical CLL morphology; IgHV mutated or unmutated; "
            "  FISH: 13q14 deletion (most common), trisomy 12; del(11q22)/del(17p) rarer; "
            "  Prognosis: similar to sporadic CLL once diagnosed; "
            "MELANOMA: cutaneous; acral; uveal (less common); "
            "  Melanoma + CLL in same patient or family = POT1 likely; "
            "ANGIOSARCOMA: rare; cardiac angiosarcoma + CLL → POT1 germline testing; "
            "GLIOMA: astrocytoma, GBM — very rare; "
            "MANAGEMENT: "
            "  Clinical genetics review for CLL + melanoma family history; "
            "  Dermatology surveillance: annual skin check; "
            "  No proven hematologic surveillance protocol (no CLL screening guidelines); "
            "  Ophthalmology: slit-lamp for uveal; "
            "  Treat CLL per standard risk stratification"
        ),
        "disease_pathway": (
            "POT1 — TELOMERE PROTECTION AND GENOME STABILITY: "
            "NORMAL TELOMERE MAINTENANCE: "
            "  Telomeres: TTAGGG repeats; protect chromosomal ends from DNA repair machinery; "
            "  Shelterin assembles at telomere: TRF1/TRF2 (dsDNA), POT1-TPP1 (ssDNA 3' overhang); "
            "  POT1 OB-folds: sequence-specific ss-TTAGGG binding; "
            "  POT1 inhibits RPA (replication protein A) from binding ss-TTAGGG; "
            "  RPA-free ss-TTAGGG: ATR/CHK1 not activated; no false DSB signalling; "
            "  POT1-TPP1 recruits and stimulates telomerase; regulates telomere length; "
            "HAPLOINSUFFICIENCY: "
            "  50% POT1 activity → partial capping failure; "
            "  RPA binds uncapped ss-TTAGGG intermittently → low-level ATR activation; "
            "  Chromosomal instability: telomere fusions; end-to-end chromosomal aberrations; "
            "  B-lymphocytes: high proliferative demand + telomere shortening → "
            "    CLL clone expansion from genomically unstable B cell; "
            "  Melanocytes: high UV-DSB rate; POT1 het → melanoma predisposition; "
            "OB-FOLD MISSENSE (Y36C, Y89C): "
            "  Disrupts ssDNA binding geometry; POT1 fails to cap telomere; "
            "  Telomere uncapping → persistent RPA → ATR-CHK1 → chromosomal fusions"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — POT1 CLL: "
            "  1. CLL + CUTANEOUS MELANOMA IN SAME INDIVIDUAL: "
            "     telomere biology predisposition → germline POT1 (or ACD/TPP1/TERT) sequencing; "
            "  2. CLL + CARDIAC ANGIOSARCOMA: rare but characteristic; "
            "     angiosarcoma + haematologic malignancy = shelterin gene work-up; "
            "  3. FAMILIAL CLL (3+ affected relatives): "
            "     germline CLL panel (POT1, ATM, CHEK2, BRCA2, SF3B1); "
            "  4. EARLY-ONSET CLL (<45y) WITH FAMILY HISTORY MELANOMA: "
            "     POT1 germline likely; "
            "  5. SHORTENED TELOMERES + CLL FAMILY: telomere length assay (qPCR / Flow-FISH); "
            "     extremely short telomeres in carriers → shelterin gene panel; "
            "DDx: DKC1/TERC/TERT (dyskeratosis congenita — mucocutaneous triad + aplastic anaemia); "
            "  ATM het (CLL + breast, not melanoma + angiosarcoma)"
        ),
        "treatment": (
            "POT1-ASSOCIATED CLL TREATMENT: "
            "CLL TREATMENT: standard CLL therapy per FISH/IGHV/TP53 risk stratification; "
            "  POT1 germline does NOT mandate a specific CLL treatment change; "
            "  Ibrutinib/acalabrutinib: preferred for IGHV-unmutated or del(11q22)/del(17p); "
            "  FCR: option for fit patients with IGHV mutated + low-risk FISH + age <65; "
            "  Venetoclax + obinutuzumab: effective; MRD-guided fixed-duration; "
            "MELANOMA: standard management; dermatology annual full-body skin exam; "
            "  Dermoscopy; excision; sentinel lymph node biopsy if >1mm Breslow; "
            "  Stage IV: checkpoint inhibitor (anti-PD1 pembrolizumab/nivolumab); "
            "    BRAF V600E/K: BRAF+MEK inhibitor (dabrafenib + trametinib); "
            "ANGIOSARCOMA: anthracycline-based chemotherapy; taxane; "
            "  Targeted: VEGFR inhibitor (sorafenib, pazopanib) off-label; "
            "GENETIC COUNSELLING: "
            "  AD inheritance — 50% offspring risk; cascade family testing; "
            "  Carriers: annual skin exam; haematology if cytopenias develop; no proven CLL screening"
        ),
        "seed": 2883,
        "n_patients": 40,
    },
    {
        "gene": "RUNX1",
        "protein": (
            "RUNX1 -- 21q22.12 AD -- 453aa -- "
            "Runt-Related-Transcription-Factor-1-AML1-CBF-alpha2-50kDa-"
            "Runt-Domain-Core-Binding-Factor-Haematopoietic-Master-TF-"
            "Familial-Platelet-Disorder-FPD-AML1-ALL-Predisposition-"
            "OMIM-Gene-151385-Disease-FPD-AML-601399"
        ),
        "locus": "21q22.12",
        "protein_size": (
            "453 aa / 50 kDa (RUNX1/AML1/CBFα2; core binding factor subunit alpha 2; "
            "Runt homology domain (RHD): binds TGTGGT core; dimerises with CBFβ (non-DNA-binding partner); "
            "C-terminal transactivation domain (TAD); "
            "RHD-CBFβ interaction stabilises RUNX1-DNA complex 10×; "
            "RUNX1-ETO (t(8;21)) most common AML fusion (AML-M2); somatic; "
            "germline LOF: FPD/AML — Familial Platelet Disorder with predisposition to AML; "
            "haploinsufficiency → thrombocytopenia (plt 50-150k) + platelet functional defect; "
            "RUNX1 activates: CSF1R, CD41, PF4, THPO receptor; "
            "RUNX1 required for: HSC emergence (AGM region); megakaryocyte maturation; myeloid/lymphoid"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — HAPLOINSUFFICIENCY: "
            "  FPD/AML (Familial Platelet Disorder with predisposition to Acute Myeloid/Lymphoid Leukaemia): "
            "  Thrombocytopenia: 50-150k (mild-moderate); lifelong; "
            "  Platelet functional defect: dense granule deficiency; impaired collagen/ADP aggregation; "
            "  Bleeding: easy bruising, epistaxis, menorrhagia; often ITP misdiagnosis; "
            "  MALIGNANCY RISK: 35-44% lifetime risk of myeloid malignancy (MDS/AML); "
            "    20% lifetime risk of lymphoid malignancy (B-ALL, T-ALL, NHL); "
            "  AML onset: 30s-50s; "
            "  ALL onset: childhood/adolescence; B-ALL reported; T-ALL rare; "
            "  Progression: thrombocytopenia (stable, years) → CHIP → MDS → AML; "
            "  Second hit: somatic RUNX1 mutation on remaining allele (biallelic in tumour); "
            "  RUNX1 amplification (iAMP21): somatic in 2% childhood B-ALL; different from germline"
        ),
        "disease_category": (
            "RUNX1 FPD/AML — MYELOID AND LYMPHOID MALIGNANCY PREDISPOSITION: "
            "THROMBOCYTOPENIA: universal in FPD/AML; plt 50-150k; "
            "  Functional defect: delta (dense) granule deficiency; impaired platelet activation; "
            "  ITP MISDIAGNOSIS: frequent; IVIg and steroids ineffective and unnecessary; "
            "MDS/AML RISK: 35-44% lifetime; median transformation age 30-40y; "
            "  MDS: RCMD, RAEB common pre-AML; cytogenetics often normal initially; "
            "  AML: de novo myeloid leukaemia; median survival without HSCT poor; "
            "B-ALL RISK: 5-20% (varies by series); paediatric or young adult; "
            "T-ALL: occasional; "
            "NHL: reported; "
            "CLONAL EVOLUTION: ASXL1, BCOR, PHF6, CDC25C mutations common second hits; "
            "CLINICAL ALERT: RUNX1 germline = AVOID SIBLING DONOR without RUNX1 testing; "
            "  ~50% siblings carry same variant → cannot donate marrow (like DDX41)"
        ),
        "disease_pathway": (
            "RUNX1 — HAEMATOPOIETIC TRANSCRIPTION FACTOR PATHWAY: "
            "NORMAL: RUNX1/CBFβ → activates haematopoietic gene programme: "
            "  HSC emergence at AGM (aorta-gonad-mesonephros); "
            "  Megakaryocyte maturation: platelet biogenesis (GP1b, ITGA2B/CD41); "
            "  Myeloid differentiation: CSF1R, CEBPA targets; "
            "  Lymphoid: TCRβ enhancer, IL3 gene regulation; "
            "  RUNX1-ETO fusion (somatic t(8;21)): dominant negative → myeloid block → AML-M2; "
            "HAPLOINSUFFICIENCY (GERMLINE): "
            "  50% RUNX1 → impaired megakaryocyte maturation → thrombocytopenia + plt function defect; "
            "  Delta granule biogenesis: RUNX1 activates NBEAL2, RAB27B; "
            "  50% reduction → fewer dense granules; ADP/collagen response impaired; "
            "CLONAL EVOLUTION TO MDS/AML: "
            "  Germline RUNX1 LOF → HSC replicative stress; "
            "  Somatic second hit: RUNX1 point mutation (second allele) OR RUNX1 deletion; "
            "  ASXL1, SRSF2, IDH1/2 co-mutations → MDS → AML; "
            "LYMPHOID PREDISPOSITION: "
            "  RUNX1 required for T-cell receptor β-chain (TCRβ) enhancer → T-cell development; "
            "  Haploinsufficiency → impaired B/T progenitor pool → clone susceptible to second hit"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — RUNX1 FPD/AML: "
            "  1. THROMBOCYTOPENIA + PLATELET FUNCTIONAL DEFECT IN MULTIPLE FAMILY MEMBERS: "
            "     autosomal dominant; ITP repeatedly failing steroids/IVIg → RUNX1 sequencing; "
            "  2. THROMBOCYTOPENIA (50-150k) → MDS/AML IN YOUNG ADULT (30-40y): "
            "     FPD/AML; thrombocytopenia years before malignancy; "
            "  3. DELTA GRANULE DEFICIENCY ON PLATELET ELECTRON MICROSCOPY: "
            "     FPD/AML specific (vs. GBS/Hermansky-Pudlak — different genes); "
            "  4. BIALLELIC RUNX1 ON TUMOUR SEQUENCING (germline + somatic second hit): "
            "     confirms FPD/AML; buccal DNA for germline origin; "
            "  5. B-ALL WITH FAMILY HISTORY THROMBOCYTOPENIA + MDS: "
            "     RUNX1 germline likely; "
            "DDx: ETV6 (thrombocytopenia + ALL, dense granule defect, different gene); "
            "  ANKRD26 (thrombocytopenia, elevated TPO, MDS risk, TPO R); "
            "  DDX41 (AML not thrombocytopenia at presentation; adult onset)"
        ),
        "treatment": (
            "RUNX1 FPD/AML TREATMENT: "
            "THROMBOCYTOPENIA (STABLE): watchful waiting if plt >50k, no bleeding; "
            "  Avoid antiplatelet drugs (aspirin, NSAIDs, clopidogrel); "
            "  Platelet transfusion for surgery/bleeding; "
            "  AVOID ITP TREATMENTS (IVIg, steroids, rituximab): ineffective, unnecessary; "
            "  Eltrombopag / romiplostim: not recommended (may accelerate MDS); "
            "MDS/AML: HSCT IS TREATMENT OF CHOICE (only cure); "
            "  Indications: blast >5%; progressive cytopenias; transformation signs; "
            "  HSCT timing: before AML transformation; "
            "  Conditioning: myeloablative (younger), RIC (older/comorbidities); "
            "  DONOR SELECTION: siblings must be RUNX1-tested; ~50% carry variant; exclude carriers; "
            "  MUD preferred over untested sibling; "
            "B-ALL: standard ALL chemotherapy (COG/BFM); HSCT in CR1 for high-risk; "
            "  Donor exclusion applies to ALL as well; "
            "SURVEILLANCE: annual CBC in all RUNX1 germline carriers; "
            "  BM biopsy if cytopenias progressive or blasts >2%; "
            "  FAMILY CASCADE: all first-degree relatives offered RUNX1 germline testing"
        ),
        "seed": 2884,
        "n_patients": 40,
    },
    {
        "gene": "STAT3",
        "protein": (
            "STAT3 -- 17q21.2 AD -- 770aa -- "
            "Signal-Transducer-and-Activator-of-Transcription-3-92kDa-SH2-Domain-Transcription-Factor-"
            "JAK-STAT3-Signalling-Cytokine-Receptor-Transducer-"
            "GOF-T-LGL-Leukaemia-Lymphoma-Autoimmune-Cytopenias-LOF-Hyper-IgE-"
            "OMIM-Gene-102582-Disease-LGL-Predisposition"
        ),
        "locus": "17q21.2",
        "protein_size": (
            "770 aa / 92 kDa (Signal Transducer and Activator of Transcription 3; "
            "domains: N-terminal coiled-coil (CCD), DNA-binding domain (DBD), linker, SH2 domain, "
            "transactivation domain (TAD), C-terminal regulatory domain; "
            "SH2 domain: binds phospho-Tyr of cytokine receptors (IL-6R, IL-10R, IL-21R, LIF-R, gp130); "
            "JAK1/2/TYK2 phosphorylates STAT3 Tyr705 → STAT3 dimerisation → nuclear translocation; "
            "STAT3 activates: BCL2, BCL-XL, MCL1, MYC, VEGF, SOCS3 (feedback); "
            "GOF mutations: SH2 domain (Y640F most common; D661Y; K658N; R688W) — constitutive activation; "
            "LOF mutations: DBD/CCD — Hyper-IgE syndrome (HIES); elevated IgE; eczema; recurrent infections"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT — GAIN-OF-FUNCTION (GOF) for LGL/Lymphoma: "
            "  STAT3 GOF mutations: SH2 domain; Y640F (tyrosine 640 phenylalanine) most recurrent; "
            "  GOF → constitutive STAT3 Tyr705 phosphorylation without cytokine stimulus; "
            "  STAT3 GOF CLINICAL SPECTRUM: "
            "    T-cell large granular lymphocyte (T-LGL) leukaemia; "
            "    NK-LGL leukaemia; "
            "    Cytopenias: autoimmune neutropenia, AIHA, ITP (all antibody-mediated); "
            "    Lymphoma: peripheral T-cell lymphoma (PTCL); hepatosplenic T-cell lymphoma; "
            "    Cytokine storm; HLH (haemophagocytic lymphohistiocytosis); "
            "  SOMATIC STAT3 GOF: same mutations clonally acquired; drive LGL expansion; "
            "    ~40% T-LGL leukaemia: somatic STAT3 GOF; "
            "  GERMLINE STAT3 GOF (rare): immunodysregulation-polyendocrinopathy phenotype; "
            "    multi-organ autoimmunity; lymphoproliferation; enteropathy; growth failure"
        ),
        "disease_category": (
            "STAT3 GOF-ASSOCIATED LYMPHOID PROLIFERATIONS AND AUTOIMMUNE CYTOPENIAS: "
            "T-LGL LEUKAEMIA: "
            "  Clonal CD3+/CD8+/CD57+ large granular lymphocytes (LGLs); "
            "  Peripheral blood: LGL count >2×10⁹/L sustained; splenomegaly; "
            "  Cytopenias: neutropenia (most common, ANC <1.0) → recurrent infections; "
            "    AIHA (warm antibody); ITP; pure red cell aplasia (PRCA); "
            "  Rheumatoid arthritis co-occurring in 25-30% T-LGL (Felty syndrome overlap); "
            "  Smouldering / slowly progressive; rarely transforms to overt lymphoma; "
            "NK-LGL: similar but CD3−/CD56+/CD16+; "
            "AGGRESSIVE NK CELL LEUKAEMIA / EXTRA-NODAL NK/T-CELL LYMPHOMA: "
            "  Rare; high-grade; EBER+; HLH risk; "
            "PERIPHERAL T-CELL LYMPHOMA (PTCL): STAT3 mutations enriched; "
            "HLH: STAT3 GOF drives macrophage activation + hypercytokinaemia; "
            "  Germline STAT3 GOF children: recurrent HLH episodes"
        ),
        "disease_pathway": (
            "STAT3 GOF — CONSTITUTIVE JAK-STAT PATHWAY ACTIVATION: "
            "NORMAL: cytokine (IL-6, IL-10, IL-21) → receptor dimerisation → JAK1/JAK2/TYK2 activation → "
            "  STAT3 Tyr705 phosphorylation → pSTAT3 dimerisation → nuclear import → "
            "  STAT3 RE binding (TTCN2-4GAA) → target gene transcription; "
            "  SOCS3 (suppressor of cytokine signalling 3) inhibits JAK → negative feedback; "
            "GOF MECHANISM (Y640F in SH2 domain): "
            "  Y640F in SH2 domain → STAT3 activated at lower cytokine levels OR spontaneously; "
            "  Constitutive pSTAT3 → continuous BCL2/BCL-XL/MCL1 → apoptosis resistance; "
            "  MYC activation → proliferation; VEGF → angiogenesis; "
            "  T-LGL EXPANSION: cytokine-independent survival of T-LGL clone; "
            "  Cytokine milieu: STAT3 GOF cells produce IL-6 and TNF-α → "
            "    autocrine loop → marrow suppression → neutropenia, AIHA, PRCA; "
            "IMMUNE DYSREGULATION: "
            "  Treg function impaired (FOXP3 suppressed by constitutive STAT3); "
            "  Auto-antibodies → AIHA, ITP, anti-neutrophil antibodies; "
            "  NK cell exhaustion → reduced anti-viral surveillance → "
            "    EBV-driven lymphoproliferation in some STAT3 GOF patients"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — STAT3 LGL: "
            "  1. CHRONIC NEUTROPENIA + CLONAL CD8+/CD57+ LGLs >2×10⁹/L IN PERIPHERAL BLOOD: "
            "     T-LGL LEUKAEMIA; next-generation sequencing for STAT3 SH2 mutations; "
            "  2. STAT3 Y640F (OR D661Y/K658N) ON LGL CLONE SEQUENCING: "
            "     PATHOGNOMONIC clonal STAT3 GOF in T-LGL; "
            "  3. RHEUMATOID ARTHRITIS + NEUTROPENIA + LGLs: "
            "     Felty syndrome-like (RA + splenomegaly + neutropenia) → STAT3 GOF T-LGL; "
            "  4. PURE RED CELL APLASIA (PRCA) + THYMOMA + LGL EXCESS: "
            "     STAT3 PRCA-LGL syndrome; "
            "  5. GERMLINE STAT3 GOF IN INFANT: recurrent HLH + enteropathy + lymphoproliferation: "
            "     STAT3 GOF immunodysregulation — distinct from somatic adult LGL; "
            "DDx: NK-LGL (CD3−); EATL (enteropathy-associated T-cell, CD3+/CD4−/CD8+/EBER−); "
            "  PTCL-NOS (not circulating LGLs); T-PLL (CD4+ or CD4+CD8+, not CD57+); "
            "  STAT3 LOF: Hyper-IgE (eczema, elevated IgE, staph infections — opposite phenotype)"
        ),
        "treatment": (
            "STAT3 GOF LGL LEUKAEMIA / ASSOCIATED CYTOPENIAS TREATMENT: "
            "ASYMPTOMATIC T-LGL: watchful waiting (many never require treatment); "
            "  ANC > 0.5 × 10⁹/L, no infections, no transfusion dependence: observe; "
            "FIRST-LINE IMMUNOSUPPRESSION: "
            "  Methotrexate (MTX) 10 mg/m2 weekly: response in 50-60% (neutropenia/PRCA/AIHA); "
            "  Cyclosporin A (CsA): alternative first-line; response 40-50%; "
            "  Cyclophosphamide: less used; effective for refractory cytopenias; "
            "G-CSF: for severe neutropenia + infections (adjunct, not curative); "
            "PRCA: CsA or MTX + erythropoietin; cyclosporin maintenance; "
            "AIHA/ITP: steroid initially; if persistent → MTX/CsA; "
            "JAK1/2 INHIBITORS: "
            "  Ruxolitinib (JAK1/2): early evidence in LGL leukaemia; case series responses; "
            "  Rationale: STAT3 GOF requires JAK for initial activation; JAKi blocks pSTAT3; "
            "GERMLINE STAT3 GOF IMMUNODYSREGULATION (paediatric): "
            "  JAKi (baricitinib, ruxolitinib): promising; early series; "
            "  HSCT: reported curative (corrects haematopoietic STAT3 GOF); "
            "LYMPHOMA (PTCL): CHOP-based chemotherapy ± HSCT; "
            "  Romidepsin, belinostat (HDAC inhibitors): PTCL approval"
        ),
        "seed": 2885,
        "n_patients": 40,
    },
]

# ── Patient simulation ──────────────────────────────────────────────────────


def _simulate_patients(gene_entry: dict) -> list:
    rng = random.Random(gene_entry["seed"])
    g = gene_entry["gene"]
    n = gene_entry["n_patients"]
    patients = []

    # Gene-specific parameter presets
    PARAMS = {
        "ATM": dict(
            age_mu=62, age_sd=12, female_frac=0.38,
            hb_mu=11.8, hb_sd=1.6, plt_mu=145, plt_sd=50,
            mono_mu=0.42, mono_sd=0.22,
            diagnoses=["CLL", "CLL (del11q22)", "MCL", "DLBCL", "NHL", "A-T + T-cell lymphoma"],
            dx_weights=[0.42, 0.28, 0.10, 0.08, 0.07, 0.05],
            treatments=["Ibrutinib", "Acalabrutinib", "Venetoclax+Obin", "R-CHOP", "FCR", "Watch+Wait"],
            tx_weights=[0.28, 0.22, 0.18, 0.14, 0.10, 0.08],
            hsct_frac=0.12, remission_frac=0.64,
        ),
        "TP53": dict(
            age_mu=28, age_sd=18, female_frac=0.52,
            hb_mu=9.4, hb_sd=2.2, plt_mu=88, plt_sd=65,
            mono_mu=0.30, mono_sd=0.20,
            diagnoses=["B-ALL (hypodiploid)", "CLL del(17p)", "AML TP53", "DLBCL", "NHL", "Hodgkin"],
            dx_weights=[0.30, 0.25, 0.18, 0.12, 0.10, 0.05],
            treatments=["COG ALL protocol", "Ibrutinib", "Ven+Aza", "R-CHOP", "ABVD", "HSCT"],
            tx_weights=[0.28, 0.22, 0.18, 0.14, 0.10, 0.08],
            hsct_frac=0.38, remission_frac=0.48,
        ),
        "CHEK2": dict(
            age_mu=58, age_sd=11, female_frac=0.46,
            hb_mu=12.2, hb_sd=1.5, plt_mu=175, plt_sd=45,
            mono_mu=0.48, mono_sd=0.24,
            diagnoses=["CLL", "NHL", "Follicular Lymphoma", "DLBCL", "Breast Cancer", "Colorectal Ca"],
            dx_weights=[0.38, 0.18, 0.16, 0.12, 0.10, 0.06],
            treatments=["Ibrutinib", "Watch+Wait", "R-CHOP", "R-Bendamustine", "ABVD", "Surveillance"],
            tx_weights=[0.28, 0.24, 0.18, 0.14, 0.08, 0.08],
            hsct_frac=0.08, remission_frac=0.68,
        ),
        "PAX5": dict(
            age_mu=12, age_sd=8, female_frac=0.44,
            hb_mu=8.6, hb_sd=2.4, plt_mu=78, plt_sd=52,
            mono_mu=0.28, mono_sd=0.18,
            diagnoses=["B-ALL", "B-ALL (Ph-like)", "Pre-B ALL", "B-ALL relapsed", "NHL", "B-ALL CR"],
            dx_weights=[0.38, 0.22, 0.18, 0.10, 0.07, 0.05],
            treatments=["COG AALL0434", "VXLD+TKI", "Blinatumomab", "Inotuzumab+chemo", "R-CHOP", "HSCT CR1"],
            tx_weights=[0.35, 0.22, 0.16, 0.12, 0.07, 0.08],
            hsct_frac=0.28, remission_frac=0.66,
        ),
        "IKZF1": dict(
            age_mu=10, age_sd=7, female_frac=0.42,
            hb_mu=8.2, hb_sd=2.6, plt_mu=72, plt_sd=58,
            mono_mu=0.25, mono_sd=0.15,
            diagnoses=["B-ALL (IKZF1-del)", "B-ALL Ph-like", "Pre-B ALL", "B-lymphopenia+infection", "ALL relapse", "B-ALL CR"],
            dx_weights=[0.34, 0.22, 0.18, 0.12, 0.08, 0.06],
            treatments=["COG AALL1231", "VXLD+Ruxolitinib", "Blinatumomab", "IVIg+prophylaxis", "HSCT CR1", "Chemotherapy"],
            tx_weights=[0.32, 0.22, 0.18, 0.12, 0.10, 0.06],
            hsct_frac=0.32, remission_frac=0.60,
        ),
        "POT1": dict(
            age_mu=54, age_sd=13, female_frac=0.40,
            hb_mu=12.4, hb_sd=1.6, plt_mu=168, plt_sd=52,
            mono_mu=0.44, mono_sd=0.22,
            diagnoses=["CLL", "CLL+Melanoma", "Melanoma", "Angiosarcoma", "Glioma", "NHL"],
            dx_weights=[0.42, 0.24, 0.18, 0.06, 0.04, 0.06],
            treatments=["Watch+Wait", "Ibrutinib", "Venetoclax+Obin", "Pembrolizumab", "Doxorubicin", "R-Benda"],
            tx_weights=[0.28, 0.26, 0.18, 0.12, 0.06, 0.10],
            hsct_frac=0.06, remission_frac=0.70,
        ),
        "RUNX1": dict(
            age_mu=38, age_sd=14, female_frac=0.50,
            hb_mu=10.8, hb_sd=2.0, plt_mu=92, plt_sd=40,
            mono_mu=0.38, mono_sd=0.20,
            diagnoses=["MDS", "AML", "B-ALL", "FPD (thrombocytopenia only)", "MDS→AML", "T-ALL"],
            dx_weights=[0.28, 0.26, 0.18, 0.14, 0.10, 0.04],
            treatments=["HSCT (allo)", "Ven+Aza", "COG ALL protocol", "Watch+Wait", "Azacitidine", "7+3 induction"],
            tx_weights=[0.32, 0.22, 0.18, 0.14, 0.08, 0.06],
            hsct_frac=0.44, remission_frac=0.54,
        ),
        "STAT3": dict(
            age_mu=52, age_sd=16, female_frac=0.54,
            hb_mu=10.2, hb_sd=2.2, plt_mu=112, plt_sd=68,
            mono_mu=0.50, mono_sd=0.28,
            diagnoses=["T-LGL leukaemia", "NK-LGL leukaemia", "T-LGL+AIHA", "T-LGL+neutropenia", "PTCL", "Autoimmune cytopenia"],
            dx_weights=[0.38, 0.14, 0.18, 0.16, 0.08, 0.06],
            treatments=["Methotrexate", "Cyclosporin A", "Ruxolitinib", "Cyclophosphamide", "Watch+Wait", "CHOP"],
            tx_weights=[0.32, 0.26, 0.14, 0.12, 0.10, 0.06],
            hsct_frac=0.06, remission_frac=0.58,
        ),
    }

    p = PARAMS.get(g, PARAMS["ATM"])

    outcomes = ["remission", "stable", "progressive", "deceased"]
    out_weights = [
        p["remission_frac"],
        0.28,
        1.0 - p["remission_frac"] - 0.28 - 0.08,
        0.08,
    ]
    out_weights[2] = max(out_weights[2], 0.04)

    for i in range(n):
        age = max(1, int(rng.gauss(p["age_mu"], p["age_sd"])))
        sex = "F" if rng.random() < p["female_frac"] else "M"
        hb = round(max(3.0, rng.gauss(p["hb_mu"], p["hb_sd"])), 1)
        plt = max(10, int(rng.gauss(p["plt_mu"], p["plt_sd"])))
        mono = round(max(0.00, rng.gauss(p["mono_mu"], p["mono_sd"])), 2)
        dx = rng.choices(p["diagnoses"], weights=p["dx_weights"])[0]
        tx = rng.choices(p["treatments"], weights=p["tx_weights"])[0]
        hsct = "Yes" if rng.random() < p["hsct_frac"] else "No"
        outcome = rng.choices(outcomes, weights=out_weights)[0]
        patients.append({
            "id": f"{g}-{i+1:03d}",
            "gene": g,
            "age_at_dx": age,
            "sex": sex,
            "hb_gdl": hb,
            "platelets_k": plt,
            "monocytes_abs": mono,
            "primary_dx": dx,
            "treatment": tx,
            "hsct": hsct,
            "outcome": outcome,
        })
    return patients


# ── Public API functions ────────────────────────────────────────────────────


def generate_overview() -> dict:
    gene_summaries = []
    all_patients = []
    for ge in ATLAS_GENES:
        pts = _simulate_patients(ge)
        all_patients.extend(pts)
        ages = [p["age_at_dx"] for p in pts]
        hbs = [p["hb_gdl"] for p in pts]
        plts = [p["platelets_k"] for p in pts]
        pct_f = round(sum(1 for p in pts if p["sex"] == "F") / len(pts) * 100)
        pct_hsct = round(sum(1 for p in pts if p["hsct"] == "Yes") / len(pts) * 100)
        pct_rem = round(sum(1 for p in pts if p["outcome"] == "remission") / len(pts) * 100)
        gene_summaries.append({
            "gene": ge["gene"],
            "locus": ge["locus"],
            "protein_summary": ge["protein"].split("--")[-1].strip()[:90],
            "n_patients": len(pts),
            "median_age_dx": sorted(ages)[len(ages) // 2],
            "pct_female": pct_f,
            "mean_hb_gdl": round(sum(hbs) / len(hbs), 1),
            "mean_platelets_k": round(sum(plts) / len(plts)),
            "pct_hsct": pct_hsct,
            "pct_remission": pct_rem,
        })

    return {
        "atlas": "Hereditary Lymphoid Malignancy Predisposition Atlas",
        "total_patients": len(all_patients),
        "n_genes": len(ATLAS_GENES),
        "seeds": "2878-2885",
        "genes": [ge["gene"] for ge in ATLAS_GENES],
        "gene_summaries": gene_summaries,
        "pathway_summary": {
            "DNA_Damage_Response": ["ATM", "TP53", "CHEK2"],
            "B_Cell_Development_TF": ["PAX5", "IKZF1"],
            "Telomere_Protection": ["POT1"],
            "Haematopoietic_Master_TF": ["RUNX1"],
            "JAK_STAT_Signalling": ["STAT3"],
        },
        "key_clinical_alerts": [
            "ATM homozygous (A-T): RADIOSENSITIVE — RT contraindicated; IgG replacement for immunodeficiency",
            "TP53 germline (LFS): Whole-body MRI annually (Toronto Protocol); AVOID RT; hypodiploid ALL → germline TP53",
            "CHEK2 I157T: Most common in Eastern Europeans; 2-3× CLL + 4× colorectal; moderate-risk counselling",
            "PAX5 germline: Familial B-ALL; buccal DNA to distinguish germline vs somatic; sibling donor exclusion",
            "IKZF1 germline: B-lymphopenia + B-ALL; IKZF1plus profile = HSCT in CR1; IVIg if IgG <400 mg/dL",
            "POT1 germline: Familial CLL + melanoma + angiosarcoma; annual dermatology; treat CLL per standard",
            "RUNX1 FPD/AML: AVOID ITP treatment (ineffective); sibling donor testing mandatory; HSCT for MDS/AML",
            "STAT3 GOF Y640F: Clonal T-LGL + neutropenia; MTX/CsA first-line; JAKi (ruxolitinib) emerging; LOF=Hyper-IgE (opposite!)",
        ],
    }


def generate_breakdown() -> dict:
    result = {}
    for ge in ATLAS_GENES:
        pts = _simulate_patients(ge)
        result[ge["gene"]] = {
            "gene": ge["gene"],
            "locus": ge["locus"],
            "protein": ge["protein"],
            "protein_size": ge["protein_size"],
            "inheritance": ge["inheritance"],
            "disease_category": ge["disease_category"],
            "disease_pathway": ge["disease_pathway"],
            "pathognomonic": ge["pathognomonic"],
            "treatment": ge["treatment"],
            "n_patients": len(pts),
            "patients": pts,
        }
    return result


def generate_definitions() -> dict:
    glossary = {
        "ATM-A-T-PATHOGNOMONIC": (
            "Ataxia-Telangiectasia (A-T) PATHOGNOMONIC TETRAD: "
            "(1) progressive cerebellar ataxia onset age 1-2y; "
            "(2) oculocutaneous telangiectasias (conjunctiva, ears, antecubital fossa) age 3-5y; "
            "(3) IgA deficiency + combined immunodeficiency (low IgG/IgM/IgE); "
            "(4) elevated AFP (>2 SD above age-corrected normal) >90% patients. "
            "No other primary immunodeficiency produces all four simultaneously. "
            "Radiosensitivity: DO NOT order RT without extreme caution — chromosomal breakage; severe toxicity. "
            "del(11q22) in CLL: somatic ATM locus deletion — most common CLL cytogenetic abnormality (25-35%); "
            "ibrutinib/venetoclax preferred over FCR for del(11q22) CLL."
        ),
        "TP53-LFS-HYPODIPLOID-ALL": (
            "Li-Fraumeni Syndrome (LFS) Haematologic Alert: "
            "Hypodiploid B-ALL (<44 chromosomes) in a child → germline TP53 probability >90%; "
            "near-haploid (23-29) and low-hypodiploid (30-39) ALL subtypes. "
            "LFS whole-body MRI (Toronto Protocol): annual MRI preferred over RT-based surveillance (radiosensitivity). "
            "Somatic del(17p13) TP53 in CLL: ultra-poor risk; chemoimmunotherapy ineffective; "
            "ibrutinib or venetoclax first-line. "
            "APR-246 (eprenetapopt): investigational p53-refolding agent for TP53-mutated MDS/AML. "
            "Magrolimab (anti-CD47) + azacitidine: promising in TP53-mutated AML (CD47 phagocytosis signal)."
        ),
        "CHEK2-I157T-FOUNDER": (
            "CHEK2 I157T: Eastern European/Polish founder variant; population frequency ~5% in Poland/Baltic states. "
            "I157T missense in FHA domain: impairs CHK2 dimerisation; ~60% kinase activity reduction. "
            "Risk spectrum: 2-3× CLL; 4× colorectal cancer; modest NHL. "
            "CHEK2 1100delC: NW European founder; frameshift/truncation; near-complete LOF; "
            "4-5× breast cancer risk (28-37% lifetime). "
            "Compound CHEK2 + ATM/BRCA2: synergistic CLL/lymphoma risk. "
            "Management: moderate-risk (not BRCA-equivalent); enhanced breast/colorectal surveillance; "
            "no proven haematologic screening protocol."
        ),
        "PAX5-FAMILIAL-B-ALL": (
            "PAX5 Familial B-ALL: PAX5 G183S most recurrent germline variant. "
            "Dominant negative mechanism: mutant PAX5 blocks normal PAX5 transcriptional activation. "
            "Second hit: somatic deletion 9p13.2 or PAX5 somatic mutation → biallelic PAX5 loss in leukaemia. "
            "Buccal DNA testing: mandatory to distinguish germline from somatic PAX5 mutations "
            "(somatic 9p13 deletion very common in sporadic ALL). "
            "Sibling donor exclusion: test siblings before HSCT donation (AD inheritance — 50% carrier risk). "
            "Incomplete penetrance: ~20-30% lifetime B-ALL risk in carriers — most never develop ALL."
        ),
        "IKZF1-IK6-DOMINANT-NEGATIVE": (
            "IKZF1 IK6 isoform: dominant negative IKZF1 lacking DNA-binding zinc fingers ZF1-4 (exons 4-7 deleted). "
            "IK6 sequesters wild-type IKZF1 via C-terminal dimerisation (ZF5-6 intact). "
            "IK6 = near-complete functional IKZF1 knock-out even in heterozygous context. "
            "IKZF1plus prognostic profile: IKZF1 deletion + CDKN2A/B deletion + PAX5 deletion/mutation + no ERG deletion "
            "→ ultra-high risk B-ALL; HSCT in CR1 recommended by many European/COG guidelines. "
            "Ph-like ALL: most common co-occurring molecular subtype with IKZF1 deletion; "
            "ABL-class fusions → add dasatinib/imatinib; CRLF2-JAK → add ruxolitinib."
        ),
        "POT1-SHELTERIN-TELOMERE": (
            "POT1 shelterin function: POT1-TPP1 heterodimer binds 3' single-strand TTAGGG overhang. "
            "Prevents RPA binding → prevents ATR/CHK1 false-alarm activation at telomere end. "
            "POT1 OB-fold missense (Y36C, Y89C, A76V): disrupts ssDNA-TTAGGG geometry → uncapped telomere → chromosomal instability. "
            "Familial CLL association: ~3-5% familial CLL families carry germline POT1 variants. "
            "Multi-cancer: CLL + cutaneous melanoma + angiosarcoma + glioma. "
            "CLL + cutaneous melanoma → telomere-biology predisposition panel (POT1, ACD, TERT, TERC). "
            "Cardiac angiosarcoma + CLL → POT1 germline testing mandatory."
        ),
        "RUNX1-FPD-AML-DONOR-EXCLUSION": (
            "RUNX1 Familial Platelet Disorder with AML (FPD/AML): "
            "Thrombocytopenia 50-150k + platelet functional defect (dense granule deficiency) → lifelong. "
            "ITP MISDIAGNOSIS COMMON: IVIg/steroids ineffective and unnecessary — avoid. "
            "Lifetime malignancy risk: 35-44% MDS/AML; 15-20% B-ALL or T-ALL. "
            "Sibling donor exclusion (same as DDX41): ~50% siblings carry RUNX1 variant → test before donation. "
            "Annual CBC surveillance in all germline RUNX1 carriers; BM biopsy if cytopenias progress. "
            "Eltrombopag/romiplostim: not recommended (may accelerate MDS evolution). "
            "HSCT: only curative option for MDS/AML — perform before transformation when possible."
        ),
        "STAT3-GOF-Y640F-LGL": (
            "STAT3 GOF Y640F: most recurrent somatic mutation in T-LGL leukaemia (~40% of cases). "
            "Constitutive STAT3 Tyr705 phosphorylation → BCL2/BCL-XL/MCL1 upregulation → LGL clone survival. "
            "T-LGL diagnostic criteria: CD3+/CD8+/CD57+ LGL count >2×10⁹/L sustained >6 months + clonal TCR. "
            "Felty syndrome overlap: rheumatoid arthritis + splenomegaly + neutropenia → 25-30% T-LGL. "
            "Treatment: methotrexate 10 mg/m2 weekly (response 50-60%); cyclosporin A (40-50%); "
            "ruxolitinib (JAKi) emerging in refractory cases. "
            "STAT3 LOF = OPPOSITE PHENOTYPE: Hyper-IgE syndrome (HIES) — elevated IgE, eczema, staph infections. "
            "Germline STAT3 GOF (paediatric): HLH + enteropathy + lymphoproliferation — baricitinib/HSCT."
        ),
        "CASCADE-Testing-Hereditary-Lymphoid-Malignancy": (
            "Hereditary Lymphoid Malignancy Predisposition — Cascade Testing Framework: "
            "Index case confirmed → offer germline testing to all first-degree relatives. "
            "Donor exclusion: ATM-het (CLL/MCL family), PAX5, IKZF1, RUNX1, DDX41 carriers "
            "→ test ALL potential sibling donors before HSCT donation. "
            "RT avoidance: ATM homozygous (A-T) and TP53 germline (LFS) = radiosensitivity; "
            "AVOID standard-dose RT; discuss alternative approaches with radiation oncology. "
            "Penetrance counselling: most germline variants have INCOMPLETE penetrance; "
            "carriers need surveillance but should not be overtreated or labelled as cancer-certain. "
            "Multi-gene panel testing: CLL germline panel (ATM, CHEK2, BRCA2, POT1, SF3B1, MBL); "
            "ALL panel (PAX5, IKZF1, ETV6, TP53, RUNX1); "
            "LGL panel (STAT3 SH2 mutation sequencing on sorted LGL clone + germline buccal)."
        ),
    }
    return {
        "atlas": "Hereditary Lymphoid Malignancy Predisposition Atlas",
        "genes": [ge["gene"] for ge in ATLAS_GENES],
        "n_definitions": len(glossary),
        "glossary": glossary,
    }
