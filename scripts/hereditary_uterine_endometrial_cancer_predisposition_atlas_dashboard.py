#!/usr/bin/env python3
"""Hereditary-Uterine-Endometrial-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
MLH1    (MutL homolog 1; 756aa; 3p22.2; AD LOF;
         Lynch syndrome type 1 — endometrial 40-60% lifetime, Lynch HIGHEST endometrial gene;
         MSI-H PATHOGNOMONIC; pembrolizumab FDA 2017 dMMR/MSI-H tumour-agnostic;
         aspirin CAPP2 600mg/day 50% risk reduction; H. pylori eradication MANDATORY;
         seed SEED_BASE+0) .
MSH2    (MutS homolog 2; 934aa; 2p21; AD LOF;
         Lynch syndrome type 2 — endometrial 40-60%; Muir-Torre syndrome sebaceous PATHOGNOMONIC;
         EPCAM 3'-deletion upstream silencing — MLPA MANDATORY; dMMR pembrolizumab;
         seed SEED_BASE+1) .
MSH6    (MutS homolog 6; 1360aa; 2p16.3; AD LOF;
         Lynch syndrome type 3 — endometrial 71% ABSOLUTE HIGHEST single MMR gene;
         MSI-L in 30% CRC — false-negative testing; endometrial DOMINANT phenotype;
         seed SEED_BASE+2) .
PMS2    (PMS1 homolog 2; 862aa; 7p22.1; AD LOF;
         Lynch syndrome type 4 — endometrial 15-26% lowest penetrance Lynch;
         4 pseudogenes MLPA MANDATORY; biallelic = CMMRD (constitutional MMR deficiency);
         seed SEED_BASE+3) .
PTEN    (Phosphatase and tensin homolog; 403aa; 10q23.31; AD LOF;
         Cowden syndrome / PHTS — endometrial 28-44% lifetime;
         macrocephaly PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC;
         everolimus+lenvatinib FDA 2019 advanced endometrial;
         seed SEED_BASE+4) .
TP53    (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni syndrome — uterine serous/endometrioid elevated;
         AVOID RADIATION ABSOLUTELY (secondary malignancy risk);
         whole-body MRI Toronto protocol annual;
         seed SEED_BASE+5) .
BRCA1   (BRCA1 DNA repair associated; 1863aa; 17q21.31; AD LOF;
         HBOC — uterine serous-like 2-3x elevated risk; clear cell uterine;
         BSO at 35-40yr reduces uterine serous risk; olaparib PAOLA-1/SOLO2;
         seed SEED_BASE+6) .
STK11   (Serine/threonine kinase 11 / LKB1; 433aa; 19p13.3; AD LOF;
         Peutz-Jeghers Syndrome — endometrial 13%; minimal-deviation adenocarcinoma;
         mucocutaneous macules PATHOGNOMONIC; GI endoscopy from 8yr MANDATORY;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3382-3389)
"""
import random

SEED_BASE = 3382

ATLAS_GENES = [
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MutLalpha-Scaffold-85kDa-MMR-PMS2-Heterodimer-"
            "Endometrial-40-60pct-HIGHEST-Lynch-Endometrial-Gene-"
            "MSI-H-PATHOGNOMONIC-MLH1-PMS2-Loss-IHC-"
            "Pembrolizumab-FDA2017-dMMR-Tumour-Agnostic-"
            "Aspirin-CAPP2-600mg-50pct-Risk-Reduction-"
            "HypermethylationSomatic-ExcludeBefore-GermlineCalling-OMIM-120436"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "756 aa / 85 kDa / 3p22.2 MLH1 endometrial cancer molecular context: "
            "STRUCTURE: "
            "  756 aa / 85 kDa; N-terminal ATPase domain (aa 1-340): MutLα complex with PMS2; "
            "  Linker domain (aa 341-498): flexible connector; "
            "  C-terminal domain (aa 499-756): dimerisation with PMS2 — PMS2 lost on IHC when MLH1 absent; "
            "CANCER RISKS (ENDOMETRIAL FOCUS): "
            "  ENDOMETRIAL: 40-60% lifetime (HIGHEST Lynch gene for endometrial — exceeds CRC in women); "
            "  CRC: 52-82% lifetime (Lynch type 1 classical); "
            "  OVARIAN: 8-13% endometrioid/clear cell NOT HGSOC; "
            "  GASTRIC: 6-13% (H. pylori eradication MANDATORY); "
            "KEY MANAGEMENT (ENDOMETRIAL): "
            "  ANNUAL ENDOMETRIAL SAMPLING + TVUS from 30-35yr; "
            "  RISK-REDUCING HYSTERECTOMY + BSO at completion of family (endometrial 93% risk reduction); "
            "  PEMBROLIZUMAB: FDA 2017 dMMR/MSI-H tumour-agnostic; dostarlimab GARNET trial; "
            "  ASPIRIN CAPP2 600mg/day — 50% CRC/endometrial risk reduction Level A; "
            "  OCP 50% risk reduction for endometrial in Lynch; "
            "SOMATIC MLH1 METHYLATION: promoter hypermethylation mimics Lynch — confirm germline before cascade; "
            "IHC MLH1+PMS2 both lost: 90% somatic methylation (not Lynch) unless young or family history"
        ),
        "syndrome": "Lynch syndrome type 1 (Hereditary Non-Polyposis Colorectal Cancer type 1)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "endometrial_risk": "40-60% lifetime HIGHEST Lynch endometrial gene",
        "pathognomonic": "MSI-H on tumour testing + IHC MLH1+PMS2 loss; Amsterdam II criteria",
        "key_avoid": "Do NOT call Lynch without excluding somatic MLH1 promoter hypermethylation (90% of MLH1-loss = somatic); BRAF V600E somatic mutation in CRC = somatic — not Lynch",
        "key_rule": "MSH6 > MLH1 > MSH2 for endometrial risk; MLH1 HIGHEST for CRC. Rule: annual endometrial biopsy from 30-35yr; RRHE at family completion reduces risk 93%",
        "surveillance": "Annual endometrial sampling + TVUS from 30-35yr; annual colonoscopy from 25yr; annual gastroscopy (H. pylori test-and-treat); biennial urinalysis for urothelial 3%",
        "targeted_rx": "Pembrolizumab FDA2017 dMMR/MSI-H tumour-agnostic; dostarlimab GARNET; lenvatinib+pembrolizumab KEYNOTE-775 MMR-proficient fallback",
    },
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- "
            "MutSalpha-MutSbeta-MSH6-MSH3-Scaffold-105kDa-MMR-"
            "Endometrial-40-60pct-Lynch2-MuirTorre-Sebaceous-PATHOGNOMONIC-"
            "EPCAM-3prime-Deletion-MSH2-Methylation-MLPA-MANDATORY-"
            "Urothelial-14pct-HIGHEST-Lynch-Gene-Urothelial-OMIM-609309"
        ),
        "locus": "2p21",
        "protein_size": (
            "934 aa / 105 kDa / 2p21 MSH2 endometrial cancer molecular context: "
            "STRUCTURE: "
            "  934 aa / 105 kDa; MutSα: MSH2+MSH6 (base substitutions + small indels); "
            "  MutSβ: MSH2+MSH3 (large indels); "
            "  Mismatch binding domain: recognises G-T mispairs and +1/+2 insertions; "
            "CANCER RISKS (ENDOMETRIAL FOCUS): "
            "  ENDOMETRIAL: 40-60% lifetime (MUIR-TORRE sebaceous tumours PATHOGNOMONIC); "
            "  CRC: 25-60% (Lynch type 2); "
            "  UROTHELIAL: 14% HIGHEST among Lynch genes; "
            "  OVARIAN: 11-22%; "
            "EPCAM 3'-DELETION: EPCAM (upstream of MSH2) 3'-deletion → MSH2 silencing; "
            "  NO MSH2 coding mutation found — MLPA of MSH2 promoter is MANDATORY; "
            "KEY MANAGEMENT (ENDOMETRIAL): "
            "  ANNUAL ENDOMETRIAL SAMPLING + TVUS from 30-35yr; "
            "  RISK-REDUCING HYSTERECTOMY + BSO at family completion; "
            "  ANNUAL URINE CYTOLOGY from 30-35yr (urothelial 14% HIGHEST); "
            "  MUIR-TORRE: sebaceous adenoma/carcinoma + visceral Lynch cancer = diagnostic; "
            "PEMBROLIZUMAB dMMR/MSI-H FDA 2017; dostarlimab GARNET"
        ),
        "syndrome": "Lynch syndrome type 2 / Muir-Torre syndrome",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "endometrial_risk": "40-60% lifetime Lynch type 2",
        "pathognomonic": "Sebaceous adenoma/carcinoma (Muir-Torre) + endometrial cancer = Lynch MSH2 pathway",
        "key_avoid": "Do NOT miss EPCAM 3'-deletion — MSH2 coding sequencing NORMAL; only MLPA detects EPCAM deletion causing MSH2 silencing; missing this = missing Lynch diagnosis in family",
        "key_rule": "MSH2 urothelial risk 14% HIGHEST Lynch gene for urothelial — annual urine cytology AND cystoscopy from 30-35yr. Muir-Torre sebaceous tumour = MMR IHC mandatory",
        "surveillance": "Annual endometrial sampling + TVUS 30-35yr; annual colonoscopy 25yr; annual urine cytology 30-35yr; annual skin exam Muir-Torre; EPCAM deletion testing MANDATORY in all MSH2 negative Lynch families",
        "targeted_rx": "Pembrolizumab FDA2017 dMMR/MSI-H; dostarlimab GARNET; carboplatin/paclitaxel standard endometrial; lenvatinib+pembrolizumab KEYNOTE-775",
    },
    {
        "gene": "MSH6",
        "protein": (
            "MSH6 -- 2p16.3 Autosomal-Dominant-LOF -- 1360aa -- "
            "MutSalpha-MSH2-Heterodimer-160kDa-PWWP-PIP-Box-MMR-"
            "Endometrial-71pct-ABSOLUTE-HIGHEST-Single-MMR-Gene-"
            "MSI-L-30pct-CRC-False-Negative-DOMINANT-ENDOMETRIAL-"
            "Reduced-Penetrance-Some-Families-Late-Onset-OMIM-600678"
        ),
        "locus": "2p16.3",
        "protein_size": (
            "1360 aa / 160 kDa / 2p16.3 MSH6 endometrial cancer molecular context: "
            "STRUCTURE: "
            "  1360 aa / 160 kDa; PWWP domain: chromatin association; "
            "  PIP box: PCNA interaction for post-replicative MMR; "
            "  MutSα with MSH2: base substitutions repair; "
            "CANCER RISKS (ENDOMETRIAL FOCUS): "
            "  ENDOMETRIAL: 71% lifetime ABSOLUTE HIGHEST single MMR gene for endometrial; "
            "  CRC: 10-22% (much lower than MLH1/MSH2 — MSH6 = endometrial-dominant Lynch); "
            "  OVARIAN: 11-15%; "
            "MSI-L DIAGNOSTIC PROBLEM: "
            "  MSH6 LOF → small indels only; MSI testing with standard panel MAY show MSI-L (not MSI-H); "
            "  MSI-L does NOT exclude Lynch MSH6 — ALWAYS do IHC MMR panel; "
            "  False-negative tumour testing rate 30% — IHC + MSI in parallel; "
            "KEY MANAGEMENT (ENDOMETRIAL): "
            "  ANNUAL ENDOMETRIAL SAMPLING + TVUS from 30-35yr; "
            "  RISK-REDUCING HYSTERECTOMY + BSO strongly recommended at family completion; "
            "  CRC surveillance: colonoscopy from 35yr (every 3yr — lower CRC risk than Lynch1/2); "
            "PEMBROLIZUMAB: approved dMMR/MSI-H; NOTE: MSH6 tumours occasionally MMR-p (MSI-L) — "
            "  lenvatinib+pembrolizumab KEYNOTE-775 approved regardless of MMR status for advanced endometrial"
        ),
        "syndrome": "Lynch syndrome type 3 (endometrial-dominant Lynch)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "endometrial_risk": "71% lifetime ABSOLUTE HIGHEST single MMR gene",
        "pathognomonic": "MSH6+MSH2 IHC loss in endometrial tumour; endometrial dominant Lynch (minimal CRC family history does NOT exclude MSH6 Lynch)",
        "key_avoid": "Do NOT rely on CRC family history to exclude MSH6 Lynch — CRC risk only 10-22%; MSH6 = endometrial-dominant. Do NOT use MSI only — MSI-L in 30% MSH6 tumours; always add IHC",
        "key_rule": "MSH6 endometrial 71% = HIGHEST Lynch gene for endometrial. Rule: IHC MMR all endometrial cancers regardless of age/family history (universal tumour testing). MSI-L ≠ not Lynch for MSH6",
        "surveillance": "Annual endometrial sampling + TVUS from 30-35yr; colonoscopy from 35yr (3-yearly — lower CRC risk); annual ovarian CA-125+TVUS from 30-35yr; urine cytology (urothelial elevated)",
        "targeted_rx": "Pembrolizumab FDA2017 dMMR/MSI-H; lenvatinib+pembrolizumab KEYNOTE-775 (approved regardless of MMR status for advanced endometrial); dostarlimab GARNET",
    },
    {
        "gene": "PMS2",
        "protein": (
            "PMS2 -- 7p22.1 Autosomal-Dominant-LOF -- 862aa -- "
            "MutLalpha-MLH1-Heterodimer-96kDa-Endonuclease-DQHA-Motif-MMR-"
            "Endometrial-15-26pct-LOWEST-Lynch-Penetrance-"
            "4-Pseudogenes-MLPA-MANDATORY-NGS-Often-Misassigned-"
            "Biallelic-CMMRD-Constitutional-MMR-Deficiency-OMIM-600259"
        ),
        "locus": "7p22.1",
        "protein_size": (
            "862 aa / 96 kDa / 7p22.1 PMS2 endometrial cancer molecular context: "
            "STRUCTURE: "
            "  862 aa / 96 kDa; N-terminal ATPase (MLH1 interaction); "
            "  Endonuclease domain: DQHA motif Mg2+-dependent nicking; "
            "  4 PSEUDOGENES (PMS2CL, PMS2P1-3): NGS reads misassigned — standard NGS MISSES 30%; "
            "  MLPA PMS2-specific probes MANDATORY for all families; "
            "CANCER RISKS (ENDOMETRIAL FOCUS): "
            "  ENDOMETRIAL: 15-26% lifetime (LOWEST penetrance Lynch for endometrial); "
            "  CRC: 15-20% (LOWEST penetrance Lynch for CRC); "
            "  OVARIAN: lower vs other Lynch genes; "
            "BIALLELIC CMMRD: "
            "  Both PMS2 alleles mutated (recessive) = Constitutional MMR Deficiency; "
            "  Childhood: CNS tumours + colorectal polyposis + haematological malignancies; "
            "  Café-au-lait macules (NF1-like) PATHOGNOMONIC CMMRD; "
            "  CMMRD: avoid standard MMR IHC (tumours often MMR-proficient at childhood) — sequencing essential; "
            "KEY MANAGEMENT (ENDOMETRIAL): "
            "  ENDOMETRIAL SAMPLING + TVUS from 35yr (later than MLH1/MSH2/MSH6 due to lower risk); "
            "  RISK-REDUCING HYSTERECTOMY + BSO at family completion (lower absolute benefit vs MSH6); "
            "  COLONOSCOPY from 30yr (every 3yr); "
            "PSEUDOGENE TRAP: PMS2 deletion called as variant in pseudogene = FALSE NEGATIVE"
        ),
        "syndrome": "Lynch syndrome type 4 (lowest penetrance Lynch) / CMMRD (biallelic)",
        "inheritance": "AD LOF monoallelic / AR biallelic CMMRD",
        "endometrial_risk": "15-26% lifetime LOWEST Lynch penetrance",
        "pathognomonic": "PMS2+MLH1 IHC loss (MLH1 absent due to MutLα heterodimerisation); CMMRD: café-au-lait macules + CNS tumour + polyps in child",
        "key_avoid": "NEVER diagnose PMS2 Lynch by standard sequencing alone — 4 pseudogenes cause up to 30% false-negative rate; MLPA MANDATORY. Never underestimate biallelic CMMRD in child with NF1-like macules + malignancy",
        "key_rule": "PMS2 = lowest Lynch penetrance but STILL significant absolute risk. Rule: MLPA PMS2-specific is the ONLY reliable test — confirm ALL PMS2 variants with orthogonal long-range PCR or MLPA",
        "surveillance": "Annual endometrial sampling + TVUS from 35yr; colonoscopy from 30yr (3-yearly); consider upper GI from 35yr; CMMRD: annual brain MRI + GI surveillance from childhood",
        "targeted_rx": "Pembrolizumab FDA2017 dMMR/MSI-H; dostarlimab GARNET; CMMRD: nivolumab+ipilimumab emerging (high TMB); checkpoint inhibitors undergoing study for CMMRD brain tumours",
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 Autosomal-Dominant-LOF -- 403aa -- "
            "PI3K-Phosphatase-47kDa-PIP3-Dephosphorylation-AKT-mTOR-"
            "Endometrial-28-44pct-Cowden-PHTS-Lifetime-"
            "Macrocephaly-PATHOGNOMONIC-Lhermitte-Duclos-PATHOGNOMONIC-"
            "Breast-85pct-Thyroid-35pct-Everolimus-Lenvatinib-FDA2019-OMIM-601728"
        ),
        "locus": "10q23.31",
        "protein_size": (
            "403 aa / 47 kDa / 10q23.31 PTEN endometrial cancer molecular context: "
            "STRUCTURE: "
            "  403 aa / 47 kDa; Phosphatase domain (aa 1-185): lipid + protein phosphatase; "
            "  PIP3 → PIP2: AKT activation suppressed; "
            "  C-terminal domain: PTEN-L isoform nuclear function; "
            "  PTEN C2 domain: membrane binding; "
            "CANCER RISKS (ENDOMETRIAL FOCUS): "
            "  ENDOMETRIAL: 28-44% lifetime (Cowden PHTS — highest non-Lynch endometrial risk); "
            "  BREAST: 85% lifetime — DOMINANT presentation; "
            "  THYROID: 35% (follicular >> papillary; multi-nodular goitre 50%); "
            "  COLON: 9% lifetime; "
            "PATHOGNOMONIC FEATURES: "
            "  MACROCEPHALY (OFC >97th centile): PATHOGNOMONIC Cowden — measure in ALL patients; "
            "  LHERMITTE-DUCLOS DISEASE (dysplastic cerebellar gangliocytoma): PATHOGNOMONIC; "
            "  MUCOCUTANEOUS lesions: trichilemmomas (face), oral papillomas, acral keratoses; "
            "  PENILE FRECKLING: PATHOGNOMONIC in males; "
            "KEY MANAGEMENT (ENDOMETRIAL): "
            "  ANNUAL ENDOMETRIAL SAMPLING + TVUS from 30-35yr; "
            "  RISK-REDUCING HYSTERECTOMY + BSO strongly recommended; "
            "  EVEROLIMUS + LENVATINIB: FDA 2019 advanced endometrial (mTOR pathway activation = mechanism); "
            "  BREAST: annual MRI from 30yr; prophylactic mastectomy reduces 90%+ risk; "
            "SOMATIC PTEN: 80% sporadic endometrial cancers have somatic PTEN loss — germline testing needed when PHTS features present"
        ),
        "syndrome": "Cowden syndrome / PTEN hamartoma tumour syndrome (PHTS)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "endometrial_risk": "28-44% lifetime Cowden/PHTS",
        "pathognomonic": "Macrocephaly PATHOGNOMONIC; Lhermitte-Duclos disease PATHOGNOMONIC; trichilemmomas face; oral papillomas; penile freckling (male)",
        "key_avoid": "Do NOT miss PTEN Cowden as a cause of endometrial cancer — 80% of sporadic endometrial has somatic PTEN loss, but PHTS phenotype (macrocephaly, Lhermitte-Duclos) signals germline. Measure OFC in all endometrial patients under 50yr",
        "key_rule": "PTEN endometrial 28-44% — HIGHEST non-Lynch single-gene endometrial predisposition. Rule: macrocephaly + endometrial cancer at any age → PTEN panel. Everolimus+lenvatinib targets the PTEN-mTOR pathway directly",
        "surveillance": "Annual endometrial sampling + TVUS from 30-35yr; annual breast MRI from 30yr; annual thyroid US; biennial colonoscopy from 35yr; annual dermoscopy; brain MRI if Lhermitte-Duclos suspected",
        "targeted_rx": "Everolimus + lenvatinib FDA 2019 advanced/recurrent endometrial (regardless of MSI status — mTOR mechanism); carboplatin/paclitaxel standard; pembrolizumab dMMR subset",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Sequence-Specific-TF-Guardian-Genome-"
            "Uterine-Serous-Elevated-LFS-De-Novo-20pct-"
            "AVOID-RADIATION-ABSOLUTELY-Secondary-Malignancy-Risk-"
            "WBMRI-Toronto-Protocol-Annual-Surveillance-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 43 kDa / 17p13.1 TP53 uterine cancer molecular context: "
            "STRUCTURE: "
            "  393 aa / 43 kDa; N-terminal transactivation domain (aa 1-67): MDM2 binding; "
            "  Proline-rich region (aa 68-98); "
            "  DNA binding domain (aa 102-292): 97% cancer mutations cluster here; "
            "  Tetramerisation domain (aa 323-356); "
            "CANCER RISKS (UTERINE FOCUS): "
            "  UTERINE SEROUS: elevated 2-5x (de novo 20% TP53 germline in apparent sporadic LFS cancers); "
            "  TP53 MUTATIONS IN 90% SPORADIC UTERINE SEROUS: somatic; germline when LFS criteria; "
            "  BREAST: 49-60% lifetime (LFS dominant presentation); "
            "  SARCOMA: 22-30% (soft tissue sarcoma, osteosarcoma); "
            "  CNS: 6-10% (choroid plexus carcinoma PATHOGNOMONIC in child); "
            "  ADRENOCORTICAL: 10% (child 50x elevated — R337H Brazilian founder); "
            "LI-FRAUMENI SYNDROME: "
            "  Any cancer <45yr in proband + 1st-degree relative <45yr or sarcoma at any age; "
            "  De novo mutations 20% — negative family history does NOT exclude LFS; "
            "RADIATION ABSOLUTELY CONTRAINDICATED: "
            "  TP53 biallelic (tumour) = radiation sensitivity; "
            "  TP53 monoallelic (germline) = 20-fold elevated secondary malignancy risk from radiation; "
            "  WBMRI Toronto: annual screening — detects pre-clinical lesions 7.4x earlier than symptom-driven; "
            "TREATMENT: avoid RT for endometrial TP53 carriers — external beam RT ABSOLUTELY CI"
        ),
        "syndrome": "Li-Fraumeni syndrome (LFS)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function; de novo 20%)",
        "endometrial_risk": "Uterine serous 2-5x elevated; breast 49-60% dominant",
        "pathognomonic": "Choroid plexus carcinoma in child PATHOGNOMONIC LFS; adrenocortical carcinoma under 5yr; multiple childhood cancers",
        "key_avoid": "AVOID RADIATION ABSOLUTELY — monoallelic TP53 germline carriers have 20-fold elevated secondary malignancy risk from radiotherapy; external beam RT for uterine cancer = ABSOLUTELY CONTRAINDICATED in LFS",
        "key_rule": "AVOID RT is the most important clinical rule. Rule: any uterine serous cancer under 45yr → screen for LFS; de novo mutation 20% — ask proband directly (not just family history); WBMRI Toronto protocol annually",
        "surveillance": "Annual WBMRI (Toronto protocol) from diagnosis/25yr; annual breast MRI from 20yr (no mammogram if <30yr — radiation risk); annual colonoscopy from 25yr; annual dermatology; annual abdominal US (adrenocortical)",
        "targeted_rx": "Surgery + platinum-based chemo (NO RT); pembrolizumab if dMMR co-occurrence; APR-246 (eprenetapopt) p53 reactivator — clinical trials; MDM2 inhibitors trials; avoid anthracyclines (cardiotoxicity + RT synergy)",
    },
    {
        "gene": "BRCA1",
        "protein": (
            "BRCA1 -- 17q21.31 Autosomal-Dominant-LOF -- 1863aa -- "
            "RING-BRCT-HR-Repair-Scaffold-208kDa-"
            "Uterine-Serous-2-3x-Elevated-Clear-Cell-Uterine-"
            "BSO-35-40yr-Reduces-Uterine-Serous-Risk-"
            "Olaparib-PAOLA-1-SOLO2-Breast-70-80pct-Ovarian-39-44pct-OMIM-113705"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1863 aa / 208 kDa / 17q21.31 BRCA1 uterine cancer molecular context: "
            "STRUCTURE: "
            "  1863 aa / 208 kDa; RING domain (aa 1-109): E3 ubiquitin ligase with BARD1; "
            "  BRCT domain (aa 1646-1863): phosphopeptide binding post-ATM/CHEK2; "
            "  Central HR scaffold: recruits RAD51 via PALB2-BRCA2; "
            "CANCER RISKS (UTERINE FOCUS): "
            "  UTERINE: 2-3x relative risk; serous subtype predominant (BRCA1-associated); "
            "  Clear cell uterine: elevated vs sporadic; "
            "  ABSOLUTE DOMINANT RISKS: breast 70-80% + ovarian 39-44% (managed as HBOC priority); "
            "  BRCA1 uterine risk does NOT exceed BSO-driven surgical menopause risk in most guidelines; "
            "BSO AT 35-40yr: "
            "  REDUCES ovarian 80-96% + breast ~50% premenopausal; "
            "  ALSO REDUCES uterine serous risk (removes estrogenic drive); "
            "  HYSTERECTOMY AT BSO: guideline-concordant for BRCA1 to remove uterine serous risk; "
            "KEY MANAGEMENT (UTERINE): "
            "  BSO + HYSTERECTOMY at 35-40yr eliminates uterine + ovarian + fallopian tube risk; "
            "  Annual TVUS from 30yr if BSO deferred; "
            "  OLAPARIB: PAOLA-1 (ovarian) + SOLO2 FDA-approved; "
            "  Note: uterine serous = HRD-positive — PARPi sensitivity emerging but not yet label"
        ),
        "syndrome": "HBOC1 (Hereditary Breast and Ovarian Cancer Syndrome type 1)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "endometrial_risk": "Uterine serous 2-3x elevated; clear cell uterine elevated",
        "pathognomonic": "HGSOC or uterine serous in BRCA1 carrier; HRD scar (SBS3) on sequencing",
        "key_avoid": "Do NOT omit hysterectomy at BSO in BRCA1 carriers — uterine serous risk persists after oophorectomy if uterus retained; standard BSO without hysterectomy leaves residual uterine serous risk in BRCA1",
        "key_rule": "BRCA1 primary management = breast/ovarian (70-80%/39-44%) — uterine 2-3x is secondary but real. Rule: add hysterectomy at BSO for complete gynaecological risk reduction in BRCA1. HRD scar on uterine serous = PARPi sensitivity",
        "surveillance": "Annual breast MRI from 25-30yr; RRSO + hysterectomy 35-40yr; annual TVUS if BSO deferred; annual mammogram from 30yr",
        "targeted_rx": "Olaparib PAOLA-1/SOLO2 (ovarian/breast); uterine serous HRD: PARPi off-label; bevacizumab GOG218/ICON7; carboplatin/paclitaxel standard; trastuzumab if HER2+ uterine serous",
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "LKB1-AMPK-Master-Kinase-48kDa-"
            "Endometrial-13pct-PJS-Minimal-Deviation-Adenocarcinoma-"
            "Mucocutaneous-Macules-PATHOGNOMONIC-"
            "GI-Endoscopy-8yr-MANDATORY-SCTAT-PATHOGNOMONIC-OMIM-602216"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 48 kDa / 19p13.3 STK11 endometrial cancer molecular context: "
            "STRUCTURE: "
            "  433 aa / 48 kDa; N-terminal regulatory domain; "
            "  Kinase domain (aa 49-309): serine/threonine kinase; "
            "  STRAD-MO25 complex: LKB1 activation; "
            "  AMPK activation → mTOR suppression → energy sensing; "
            "CANCER RISKS (UTERINE/ENDOMETRIAL FOCUS): "
            "  ENDOMETRIAL: 13% lifetime (Peutz-Jeghers); "
            "  MINIMAL-DEVIATION ADENOCARCINOMA (adenoma malignum cervix): PATHOGNOMONIC PJS; "
            "  CERVICAL: 27% overall (adenoma malignum specifically associated with STK11); "
            "  UTERINE: elevated over baseline; "
            "OTHER PJS RISKS: breast 50%, gastric 29%, pancreatic 36%, ovarian SCTAT 21%; "
            "PATHOGNOMONIC FEATURES: "
            "  MUCOCUTANEOUS MACULES (peri-oral, buccal, fingers, toes, perinasal): PATHOGNOMONIC PJS; "
            "  SCTAT (sex cord tumour with annular tubules): bilateral calcified = BENIGN in PJS; "
            "  MINIMAL-DEVIATION ADENOCARCINOMA CERVIX: PATHOGNOMONIC — different histology to usual HPV adenocarcinoma; "
            "  GI HAMARTOMATOUS POLYPS: small bowel intussusception from age 8yr EMERGENCY; "
            "KEY MANAGEMENT (ENDOMETRIAL/UTERINE): "
            "  ANNUAL CERVICAL SMEAR + MRI PELVIS from 25yr (adenoma malignum detection); "
            "  ANNUAL ENDOMETRIAL SAMPLING + TVUS from 30yr; "
            "  GI ENDOSCOPY FROM 8yr (upper + lower) MANDATORY — polyp surveillance prevents intussusception; "
            "  PANCREATIC EUS/MRI from 30yr (36% pancreatic risk); "
            "  BREAST: annual MRI from 25yr (50% risk)"
        ),
        "syndrome": "Peutz-Jeghers syndrome (PJS)",
        "inheritance": "AD LOF (autosomal dominant loss-of-function)",
        "endometrial_risk": "13% endometrial / 27% cervical (adenoma malignum PATHOGNOMONIC)",
        "pathognomonic": "Peri-oral mucocutaneous macules PATHOGNOMONIC; SCTAT bilateral calcified = BENIGN PJS; adenoma malignum cervix PATHOGNOMONIC; GI hamartomatous polyps",
        "key_avoid": "Do NOT misdiagnose adenoma malignum as benign — it looks well-differentiated on low magnification but is malignant; MIB-1 (Ki-67) and CEA help. Do NOT dismiss SCTAT as malignant in PJS context — bilateral calcified SCTAT in PJS = almost always BENIGN",
        "key_rule": "STK11 adenoma malignum is PATHOGNOMONIC. Rule: any adenoma malignum on cervical biopsy → STK11 germline testing mandatory; mucocutaneous macules clinical diagnosis of PJS allows early intervention before genetic testing results",
        "surveillance": "Annual cervical smear + MRI pelvis from 25yr; annual endometrial sampling + TVUS from 30yr; GI endoscopy (upper + lower) from 8yr (2-3yr intervals); pancreatic EUS/MRI from 30yr; annual breast MRI from 25yr; annual testicular US in males (LCCSCT)",
        "targeted_rx": "Surgery for adenoma malignum; MEK inhibitors (selumetinib) emerging for STK11-mutant cancers; PJS GI: polypectomy to prevent intussusception; PARPi not standard; immunotherapy emerging (STK11 loss = immunotherapy resistance — KRAS co-mutation)",
    },
]

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]

_TUMOUR_TYPES = {
    "MLH1":  ["Endometrioid endometrial grade 1-2", "Endometrioid endometrial grade 3", "Synchronous endometrial+ovarian", "Mixed endometrial"],
    "MSH2":  ["Endometrioid endometrial", "Synchronous endometrial+ovarian", "Urothelial carcinoma", "Sebaceous carcinoma (Muir-Torre)"],
    "MSH6":  ["Endometrioid endometrial grade 1-2 (dominant)", "Endometrioid endometrial grade 3", "Endometrial MSI-L subtype", "Ovarian endometrioid"],
    "PMS2":  ["Endometrioid endometrial grade 1", "Endometrioid endometrial grade 2", "Colorectal (Lynch low-penetrance)", "Endometrial MMR-proficient subtype"],
    "PTEN":  ["Endometrioid endometrial grade 1 (dominant)", "Complex atypical hyperplasia → carcinoma", "Endometrioid endometrial grade 2", "Uterine serous (PTEN-null)"],
    "TP53":  ["Uterine serous carcinoma", "Endometrial carcinosarcoma", "Mixed serous+endometrioid", "Uterine serous clear cell"],
    "BRCA1": ["Uterine serous (BRCA1-associated)", "Clear cell uterine", "High-grade endometrial NOS", "Endometrial with HRD scar"],
    "STK11": ["Endometrioid endometrial PJS", "Adenoma malignum cervix (PATHOGNOMONIC)", "Minimal-deviation adenocarcinoma", "Uterine smooth muscle tumour PJS"],
}

_VARIANTS_BY_GENE = {
    "MLH1":  ["c.677G>T (p.Arg226Ter)", "c.1852_1854delAAG", "c.454+1G>A (splice)", "c.199G>A (p.Val67Met)", "c.117-1G>T (splice)"],
    "MSH2":  ["c.1255delA", "c.942+3A>T (splice)", "c.1787_1788delGA", "c.2131G>T (p.Gly711Ter)", "c.388-1G>C (splice)"],
    "MSH6":  ["c.3959_3962delCAAG", "c.116G>T (p.Cys39Phe)", "c.1444C>T (p.Arg482Ter)", "c.3261dupC", "c.2731C>T (p.Arg911Ter)"],
    "PMS2":  ["c.137G>T (p.Cys46Phe)", "c.736_741delAATTGT", "c.1A>T (p.Met1Leu)", "c.2T>A (p.Met1Lys)", "c.903+3A>T (splice PMS2CL pseudogene)"],
    "PTEN":  ["c.389G>A (p.Arg130Gln)", "c.209+1G>A (splice)", "c.800delA", "c.1003C>T (p.Arg335Ter)", "c.697C>T (p.Arg233Ter)"],
    "TP53":  ["c.817C>T (p.Arg273Cys)", "c.844C>T (p.Arg282Trp)", "c.742C>T (p.Arg248Trp)", "c.1010G>A (p.Arg337His)", "c.524G>A (p.Arg175His)"],
    "BRCA1": ["c.68_69delAG (185delAG)", "c.5266dupC (5382insC)", "c.1687C>T (p.Gln563Ter)", "c.3756_3759delGTCT", "c.5123C>A (p.Ala1708Glu)"],
    "STK11": ["c.863_866delATGT", "c.465del (p.Lys155Asnfs)", "c.290C>T (p.Ala97Val)", "c.920-2A>G (splice)", "c.1062_1063insT"],
}

TREATMENT_PROTOCOLS_BY_GENE = {
    "MLH1":  ["Pembrolizumab FDA2017 dMMR/MSI-H", "Dostarlimab GARNET", "Carboplatin/paclitaxel standard", "Risk-reducing hysterectomy+BSO", "Aspirin CAPP2 600mg/day prevention"],
    "MSH2":  ["Pembrolizumab FDA2017 dMMR/MSI-H", "Lenvatinib+pembrolizumab KEYNOTE-775", "Risk-reducing hysterectomy+BSO", "Annual urine cytology urothelial surveillance", "Dostarlimab GARNET"],
    "MSH6":  ["Lenvatinib+pembrolizumab KEYNOTE-775 (regardless MMR status)", "Pembrolizumab dMMR/MSI-H", "Risk-reducing hysterectomy+BSO", "Carboplatin/paclitaxel", "Dostarlimab GARNET"],
    "PMS2":  ["Pembrolizumab FDA2017 dMMR/MSI-H", "Risk-reducing hysterectomy+BSO (lower risk — individualise)", "Carboplatin/paclitaxel", "CMMRD: nivolumab+ipilimumab emerging", "Dostarlimab GARNET"],
    "PTEN":  ["Everolimus+lenvatinib FDA2019 advanced endometrial", "Lenvatinib+pembrolizumab KEYNOTE-775", "Risk-reducing hysterectomy+BSO", "Carboplatin/paclitaxel standard", "Progestin IUD/oral (conservative if fertility desired)"],
    "TP53":  ["Surgery + platinum-based chemo (NO RT ABSOLUTELY)", "APR-246 (eprenetapopt) p53 reactivator trials", "MDM2 inhibitor trials", "Pembrolizumab if dMMR co-occurrence", "Avoid anthracyclines (RT synergy risk)"],
    "BRCA1": ["Olaparib PAOLA-1/SOLO2 (ovarian-primary data)", "BSO+hysterectomy 35-40yr risk-reducing", "Carboplatin/paclitaxel standard", "Bevacizumab GOG218/ICON7", "Trastuzumab if HER2+ uterine serous"],
    "STK11": ["Surgery adenoma malignum (wide excision)", "MEK inhibitors emerging", "GI polypectomy endoscopic prevention", "Carboplatin/paclitaxel standard", "Immunotherapy resistance (STK11 loss) — consider doublet"],
}

SURVEILLANCE_BY_GENE = {
    "MLH1":  ["Annual endometrial sampling + TVUS from 30-35yr", "Annual colonoscopy from 25yr", "Annual gastroscopy H. pylori test-and-treat", "Biennial urine cytology urothelial 3%", "OCP 50% risk reduction option"],
    "MSH2":  ["Annual endometrial sampling + TVUS from 30-35yr", "Annual urine cytology + cystoscopy from 30-35yr", "Annual colonoscopy from 25yr", "Annual skin exam Muir-Torre sebaceous", "EPCAM deletion MLPA MANDATORY"],
    "MSH6":  ["Annual endometrial sampling + TVUS from 30-35yr", "Colonoscopy from 35yr (3-yr interval — lower CRC)", "IHC+MSI in parallel (MSI-L 30% false-negative)", "Annual CA-125+TVUS ovarian 11-15%", "Universal MMR IHC all endometrial cancers"],
    "PMS2":  ["Annual endometrial sampling + TVUS from 35yr", "Colonoscopy from 30yr (3-yr interval)", "MLPA PMS2-specific MANDATORY", "CMMRD: annual brain MRI from childhood", "Upper GI from 35yr"],
    "PTEN":  ["Annual endometrial sampling + TVUS from 30-35yr", "Annual breast MRI from 30yr", "Annual thyroid US", "Biennial colonoscopy from 35yr", "Annual dermoscopy"],
    "TP53":  ["Annual whole-body MRI (WBMRI) Toronto protocol", "Annual breast MRI from 20yr (NO mammogram <30yr)", "Annual colonoscopy from 25yr", "Annual abdominal US (adrenocortical)", "Annual dermatology"],
    "BRCA1": ["Annual breast MRI from 25-30yr + mammogram from 30yr", "BSO+hysterectomy 35-40yr", "Annual TVUS if BSO deferred", "RRSO timing: 35-40yr MANDATORY", "Annual CA-125 post-BSO (low yield)"],
    "STK11": ["Annual cervical smear + MRI pelvis from 25yr", "Annual endometrial sampling + TVUS from 30yr", "GI endoscopy upper+lower from 8yr (2-3yr intervals)", "Pancreatic EUS/MRI from 30yr", "Annual breast MRI from 25yr"],
}


def _make_patients(gene: str, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)
    tumours = _TUMOUR_TYPES.get(gene, ["Endometrial cancer NOS"])
    variants = _VARIANTS_BY_GENE.get(gene, ["Pathogenic variant"])
    pts = []
    for i in range(n):
        age = rng.randint(38, 75)
        pts.append({
            "patient_id": f"{gene[:3]}-UECA-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_dx": age,
            "tumour_type": rng.choice(tumours),
            "variant": rng.choice(variants),
            "stage": rng.choice(["I", "II", "III", "IV"]),
            "checkpoint_inhibitor": rng.random() < (0.75 if gene in ("MLH1","MSH2","MSH6","PMS2") else 0.25 if gene in ("PTEN","STK11") else 0.30),
            "targeted_therapy": rng.random() < (0.55 if gene == "PTEN" else 0.35 if gene == "BRCA1" else 0.20),
            "rrhe_performed": rng.random() < (0.65 if gene in ("MLH1","MSH2","MSH6") else 0.45 if gene in ("PTEN","BRCA1") else 0.30),
            "relapse": rng.random() < 0.40,
            "radiation_given": rng.random() < (0.05 if gene == "TP53" else 0.45),
        })
    return pts


def generate_overview() -> dict:
    cohorts = {}
    for i, g in enumerate(ATLAS_GENES):
        gene = g["gene"]
        pts = _make_patients(gene, SEED_BASE + i)
        cohorts[gene] = pts

    total = sum(len(v) for v in cohorts.values())
    gene_counts = {g: len(pts) for g, pts in cohorts.items()}
    overall_checkpoint_rate = round(
        100 * sum(p["checkpoint_inhibitor"] for pts in cohorts.values() for p in pts) / total, 1
    )
    overall_targeted_rate = round(
        100 * sum(p["targeted_therapy"] for pts in cohorts.values() for p in pts) / total, 1
    )
    rrhe_rate = round(
        100 * sum(p["rrhe_performed"] for pts in cohorts.values() for p in pts) / total, 1
    )
    mean_age = round(
        sum(p["age_at_dx"] for pts in cohorts.values() for p in pts) / total, 1
    )
    stage_iii_iv = round(
        100 * sum(1 for pts in cohorts.values() for p in pts if p["stage"] in ("III","IV")) / total, 1
    )
    tp53_rt_given = round(
        100 * sum(p["radiation_given"] for p in cohorts.get("TP53",[])) / max(len(cohorts.get("TP53",[])),1), 1
    )

    return {
        "atlas": "Hereditary-Uterine-Endometrial-Cancer-Predisposition-Atlas",
        "genes": _GENE_LIST,
        "total_patients": total,
        "gene_counts": gene_counts,
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "checkpoint_inhibitor_rate_pct": overall_checkpoint_rate,
        "targeted_therapy_rate_pct": overall_targeted_rate,
        "rrhe_rate_pct": rrhe_rate,
        "mean_age_at_dx": mean_age,
        "stage_iii_iv_pct": stage_iii_iv,
        "tp53_rt_avoidance_note": f"TP53 cohort: radiation_given {tp53_rt_given}% — target <10% (AVOID RT ABSOLUTELY in LFS)",
        "key_facts": [
            "MSH6: 71% endometrial ABSOLUTE HIGHEST single MMR gene; MSI-L in 30% — ALWAYS do IHC not MSI alone",
            "MLH1/MSH2: 40-60% endometrial; MLH1 somatic methylation 90% of IHC-loss — confirm germline first",
            "PMS2: LOWEST Lynch penetrance 15-26%; 4 pseudogenes — MLPA MANDATORY for all PMS2 families",
            "PTEN Cowden: 28-44% endometrial HIGHEST non-Lynch; macrocephaly PATHOGNOMONIC; everolimus+lenvatinib FDA2019",
            "TP53 LFS: uterine serous 2-5x; AVOID RADIATION ABSOLUTELY — 20-fold secondary malignancy risk",
            "BRCA1: uterine serous 2-3x; hysterectomy AT BSO (35-40yr) eliminates residual uterine serous risk",
            "STK11 PJS: adenoma malignum cervix PATHOGNOMONIC; endometrial 13%; GI endoscopy from 8yr MANDATORY",
            "CASCADE TESTING: universal MMR IHC on all endometrial tumours regardless of age — identifies 3-5% Lynch",
        ],
    }


def generate_breakdown() -> dict:
    breakdown = {}
    from collections import Counter
    for i, g in enumerate(ATLAS_GENES):
        gene = g["gene"]
        pts = _make_patients(gene, SEED_BASE + i)
        tumour_counts = Counter(p["tumour_type"] for p in pts)
        variant_counts = Counter(p["variant"] for p in pts)
        top_tumours = tumour_counts.most_common(3)
        top_variants = variant_counts.most_common(3)
        breakdown[gene] = {
            "gene_info": {
                "gene": gene,
                "protein": g["protein"],
                "locus": g["locus"],
                "syndrome": g["syndrome"],
                "inheritance": g["inheritance"],
                "endometrial_risk": g["endometrial_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_avoid": g["key_avoid"],
                "key_rule": g["key_rule"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            },
            "n": len(pts),
            "checkpoint_inhibitor_pct": round(100 * sum(1 for p in pts if p["checkpoint_inhibitor"]) / len(pts), 1),
            "targeted_therapy_pct": round(100 * sum(1 for p in pts if p["targeted_therapy"]) / len(pts), 1),
            "rrhe_pct": round(100 * sum(1 for p in pts if p["rrhe_performed"]) / len(pts), 1),
            "relapse_pct": round(100 * sum(1 for p in pts if p["relapse"]) / len(pts), 1),
            "mean_age": round(sum(p["age_at_dx"] for p in pts) / len(pts), 1),
            "top_tumour_types": [{"type": t, "count": c} for t, c in top_tumours],
            "top_variants": [{"variant": v, "count": c} for v, c in top_variants],
            "treatment_protocols": TREATMENT_PROTOCOLS_BY_GENE[gene],
            "surveillance_protocols": SURVEILLANCE_BY_GENE[gene],
        }
    return {"breakdown": breakdown, "genes": _GENE_LIST}


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Uterine-Endometrial-Cancer-Predisposition-Atlas",
        "definitions": {
            "msh6_endometrial_dominant": (
                "MSH6 Lynch type 3: MutSα MSH2 heterodimer — endometrial 71% ABSOLUTE HIGHEST single MMR gene; "
                "ENDOMETRIAL DOMINANT: CRC only 10-22% (unlike MLH1/MSH2 where CRC dominates); "
                "MSI-L 30% false-negative: ALWAYS add IHC MMR panel; IHC shows MSH6+MSH2 loss; "
                "LENVATINIB+PEMBROLIZUMAB KEYNOTE-775: approved regardless of MMR status advanced endometrial"
            ),
            "mlh1_somatic_methylation": (
                "MLH1 Lynch type 1: MutLα scaffold — endometrial 40-60%; IHC MLH1+PMS2 BOTH LOST; "
                "SOMATIC METHYLATION TRAP: 90% of MLH1+PMS2 IHC-loss = somatic promoter methylation (NOT Lynch); "
                "CONFIRM GERMLINE: MLH1 promoter methylation test on tumour FIRST; BRAF V600E = somatic (not Lynch); "
                "PEMBROLIZUMAB FDA2017; ASPIRIN CAPP2 600mg/day 50% risk reduction"
            ),
            "pms2_pseudogene_trap": (
                "PMS2 Lynch type 4: MutLα endonuclease — endometrial 15-26% LOWEST Lynch penetrance; "
                "4 PSEUDOGENES: PMS2CL + PMS2P1-3 cause 30% false-negative by standard NGS; "
                "MLPA PMS2-SPECIFIC IS MANDATORY — no exceptions; long-range PCR or PMS2-specific primers; "
                "BIALLELIC CMMRD: childhood CNS + CRC polyposis + café-au-lait macules (NF1-like)"
            ),
            "pten_cowden_endometrial": (
                "PTEN Cowden/PHTS: PI3K-phosphatase — endometrial 28-44% HIGHEST non-Lynch single gene; "
                "MACROCEPHALY (OFC >97th centile) PATHOGNOMONIC — measure all endometrial patients <50yr; "
                "LHERMITTE-DUCLOS (dysplastic cerebellar gangliocytoma) PATHOGNOMONIC; "
                "EVEROLIMUS+LENVATINIB FDA 2019 advanced endometrial: mTOR pathway activation = mechanism"
            ),
            "tp53_avoid_radiation": (
                "TP53 Li-Fraumeni: p53 tumour suppressor — uterine serous 2-5x elevated; "
                "AVOID RADIATION ABSOLUTELY: monoallelic germline TP53 → 20-fold secondary malignancy risk from RT; "
                "EXTERNAL BEAM RT FOR ENDOMETRIAL = ABSOLUTELY CONTRAINDICATED in LFS; "
                "WBMRI TORONTO PROTOCOL annual; de novo 20% — negative family history does NOT exclude LFS"
            ),
            "brca1_uterine_bso_hysterectomy": (
                "BRCA1 HBOC1: RING-BRCT HR scaffold — uterine serous 2-3x; clear cell uterine elevated; "
                "BSO AT 35-40yr: reduces ovarian 80-96% + breast ~50% + uterine serous; "
                "HYSTERECTOMY AT BSO: guideline-concordant to remove residual uterine serous risk; "
                "OLAPARIB PAOLA-1/SOLO2; HRD scar (SBS3) — PARPi sensitivity uterine serous emerging"
            ),
            "stk11_adenoma_malignum": (
                "STK11/LKB1: AMPK master kinase — endometrial 13%; PJS; "
                "ADENOMA MALIGNUM CERVIX PATHOGNOMONIC: minimal-deviation adenocarcinoma; looks benign histologically; "
                "MIB-1 (Ki-67) + CEA immunostaining helps confirm malignancy; RADICAL HYSTERECTOMY; "
                "MUCOCUTANEOUS MACULES PATHOGNOMONIC PJS; GI endoscopy from 8yr MANDATORY"
            ),
            "universal_mmr_testing": (
                "UNIVERSAL MMR IHC TESTING: all endometrial tumours regardless of age/family history; "
                "Identifies 3-5% Lynch (equivalent to universal CRC testing); "
                "IHC FIRST: cost-effective; MSI as reflex confirmatory; "
                "MLH1 methylation test if MLH1+PMS2 lost (excludes somatic); "
                "Goal: identify ALL Lynch carriers — cascade testing prevents future cancers in family"
            ),
            "cascade_testing": (
                "CASCADE TESTING Hereditary Uterine/Endometrial Cancer Predisposition: "
                "index case → first-degree relatives; "
                "MMR Lynch (MLH1/MSH2/MSH6/PMS2): all first-degree — colonoscopy + endometrial surveillance; "
                "PTEN: all first-degree — macrocephaly measurement + breast MRI + endometrial sampling; "
                "TP53 LFS: all first-degree — WBMRI Toronto protocol annually; cascade critical for childhood surveillance; "
                "BRCA1: first-degree females — BSO + breast MRI; males — prostate annual PSA from 40yr; "
                "STK11: clinical diagnosis possible (mucocutaneous macules) — cascade GI + cervical surveillance"
            ),
        },
        "key_clinical_distinctions": [
            "MSH6 endometrial 71% HIGHEST single MMR gene: endometrial-dominant Lynch — CRC only 10-22%",
            "MLH1 somatic methylation: 90% of MLH1+PMS2 IHC-loss = somatic (not Lynch) — confirm before cascade",
            "PMS2 4 pseudogenes: standard NGS MISSES 30% — MLPA PMS2-specific MANDATORY no exceptions",
            "PTEN 28-44%: HIGHEST non-Lynch endometrial; macrocephaly PATHOGNOMONIC; everolimus+lenvatinib FDA2019",
            "TP53 LFS: AVOID RADIATION ABSOLUTELY — 20-fold secondary malignancy; NO external beam RT for uterine",
            "BRCA1 uterine: hysterectomy AT BSO (not BSO alone) for complete gynaecological risk reduction",
            "STK11 adenoma malignum: looks benign on low-power — Ki-67+CEA IHC confirms malignancy; RADICAL surgery",
            "Universal MMR IHC: all endometrial tumours regardless of age identifies 3-5% Lynch carriers",
        ],
    }
