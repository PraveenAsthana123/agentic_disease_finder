#!/usr/bin/env python3
"""Hereditary-Lung-Cancer-Atlas — Complete 8-Gene Hereditary Lung Cancer Atlas
EGFR   (Epidermal Growth Factor Receptor; 1210aa; 7p11.2; AD GOF;
         Hereditary NSCLC with germline T790M / exon 21 L858R;
         Osimertinib (3rd-gen EGFR TKI) first-line; lung adenocarcinoma;
         seed SEED_BASE+0) ·
STK11  (Serine-Threonine Kinase 11 / LKB1; 433aa; 19p13.3; AD LOF;
         Peutz-Jeghers Syndrome;
         Lung adenocarcinoma highest hereditary risk — 7-17x RR; KRAS-co-mut NSCLC;
         SCLC elevated; STK11-mutant NSCLC: PD-L1 cold — immunotherapy often fails;
         seed SEED_BASE+1) ·
TP53   (Tumour Protein p53; 393aa; 17p13.1; AD LOF;
         Li-Fraumeni Syndrome;
         SCLC / lung cancer component LFS; AVOID RADIATION ABSOLUTELY;
         WBMRI annually Toronto protocol;
         seed SEED_BASE+2) ·
RB1    (Retinoblastoma 1; 928aa; 13q14.2; AD LOF;
         Hereditary Retinoblastoma;
         Secondary SCLC 15-20× elevated in Rb survivors; avoid thoracic RT;
         secondary osteosarcoma + SCLC PATHOGNOMONIC Rb survivor profile;
         seed SEED_BASE+3) ·
BRCA2  (Breast Cancer gene 2; 3418aa; 13q12.3; AD LOF;
         HBOC Syndrome;
         Lung cancer 2-3× elevated — NSCLC adenocarcinoma; platinum sensitivity;
         PARP inhibitor olaparib — HRD pathway; lung is 5th BRCA2-associated cancer;
         seed SEED_BASE+4) ·
FLCN   (Folliculin; 579aa; 17p11.2; AD LOF;
         Birt-Hogg-Dubé Syndrome;
         Pulmonary cysts bilateral basal PATHOGNOMONIC; spontaneous pneumothorax 24-38%;
         lung cancer risk elevated; FLCN-null → mTOR activation;
         seed SEED_BASE+5) ·
ATM    (Ataxia-Telangiectasia Mutated; 3056aa; 11q22.3; AR/AD LOF;
         Ataxia-Telangiectasia (biallelic AR) / AT carrier lung cancer (mono AD);
         Lung cancer ~10% A-T; radiosensitivity ABSOLUTE — standard RT doses lethal;
         ATM mono carriers 2-4× lung cancer RR; olaparib/rucaparib sensitivity;
         seed SEED_BASE+6) ·
BAP1   (BRCA1-Associated Protein 1; 729aa; 3p21.1; AD LOF;
         BAP1 Tumour Predisposition Syndrome;
         Mesothelioma PATHOGNOMONIC — 30-60× RR (asbestos synergy lethal);
         Lung adenocarcinoma elevated; uveal melanoma; ccRCC; cutaneous melanoma;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3142-3149)
"""
import random

SEED_BASE = 3142

ATLAS_GENES = [
    {
        "gene": "EGFR",
        "protein": (
            "EGFR -- 7p11.2 Autosomal-Dominant-GOF -- 1210aa -- "
            "Epidermal-Growth-Factor-Receptor-RTK-ERBB1-HER1-"
            "Hereditary-NSCLC-Germline-T790M-L858R-Osimertinib-3rd-Gen-EGFR-TKI-"
            "Lung-Adenocarcinoma-OMIM-131550"
        ),
        "locus": "7p11.2",
        "protein_size": (
            "1210 aa / 7p11.2 EGFR encodes Epidermal Growth Factor Receptor (EGFR / ERBB1 / HER1): "
            "STRUCTURE: "
            "  Extracellular domain (4 sub-domains I-IV): EGF/TGFα ligand binding via domains I+III; "
            "  Transmembrane domain (single-pass type I RTK); "
            "  Juxtamembrane segment (receptor dimerisation regulation); "
            "  Intracellular kinase domain: N-lobe (ATP-binding, Gly-rich P-loop); C-lobe (catalytic); "
            "  C-terminal regulatory domain (multiple Tyr phosphorylation sites Y1068, Y1173, Y1045); "
            "EGFR SIGNALLING: "
            "  Ligand binding → homo/heterodimerisation (HER2 most common heterodimer) → kinase activation; "
            "  Downstream: RAS-RAF-MEK-ERK (proliferation), PI3K-Akt-mTOR (survival), STAT3 (transcription); "
            "GERMLINE EGFR MUTATIONS AND HEREDITARY NSCLC: "
            "  Germline T790M (p.Thr790Met, exon 20): "
            "    Most common hereditary NSCLC-associated EGFR variant; "
            "    Normally T790M is acquired somatic resistance mutation to 1st/2nd gen EGFR TKI; "
            "    Germline T790M: autosomal dominant; ~50% family members affected; "
            "    Lung adenocarcinoma onset: typically 40-60yr (earlier than sporadic NSCLC); "
            "    Never-smoker or light-former-smoker lung adenocarcinoma + family history → consider germline EGFR; "
            "    Adenomatous hyperplasia + multiple ground-glass opacities bilateral: early germline EGFR presentation; "
            "  Germline L858R (p.Leu858Arg, exon 21): "
            "    Rarer germline GOF variant; high-penetrance for lung adenocarcinoma; "
            "    Constitutive kinase activation; responsive to EGFR TKI; "
            "  Germline exon 19 deletions: rare but documented in familial NSCLC; "
            "EGFR TARGETED THERAPY — OSIMERTINIB: "
            "  1st/2nd gen EGFR TKI (erlotinib/gefitinib/afatinib): EGFR-mutant NSCLC; "
            "    T790M gatekeeper mutation: primary resistance in germline T790M; "
            "  Osimertinib (Tagrisso, AstraZeneca): 3rd-gen irreversible EGFR TKI; "
            "    Mechanism: covalently binds C797 in EGFR kinase domain; "
            "    Active against T790M + activating EGFR mutations (L858R, ex19del); "
            "    FDA 2015 (T790M resistance); FDA 2018 (1st-line EGFR-mutant NSCLC — FLAURA); "
            "    FLAURA2 data: osimertinib + chemotherapy → PFS 25.5mo vs 16.7mo; "
            "    Germline EGFR T790M → osimertinib preferred over earlier-gen TKI; "
            "    CNS penetration: superior to 1st/2nd gen — treats brain metastases; "
            "  C797S resistance: osimertinib acquired resistance; MARIPOSA trial: amivantamab+lazertinib; "
            "SURVEILLANCE FOR GERMLINE EGFR CARRIERS: "
            "  Annual low-dose CT chest: screening from age 30-35 or 10yr before earliest family case; "
            "  No smoking: absolute cessation counselling (smoking × EGFR mutation = synergistic carcinogenesis); "
            "  Awareness of GGO/adenomatous hyperplasia: multiple bilateral ground-glass opacities → LDCT + follow"
        ),
        "inheritance": (
            "AD GOF 7p11.2 — EGFR. Germline EGFR variants rare (estimated prevalence ~1:3,000 in Asian populations with familial NSCLC). "
            "T790M: autosomal dominant, penetrance 20-30% (incomplete) — not all carriers develop NSCLC in lifetime. "
            "De novo germline EGFR: not well characterised. "
            "Family history of lung adenocarcinoma + never-smoker + Asian ancestry → priority germline EGFR testing. "
            "EGFR germline vs somatic T790M: tissue biopsy + blood cfDNA + germline (WBC) — all three sources to distinguish."
        ),
        "surveillance_key": "annual LDCT chest from age 30-35; osimertinib (3rd-gen) preferred for germline T790M carriers; never-smoker lung adenocarcinoma + family history → germline EGFR panel; ground-glass opacities bilateral → close LDCT surveillance",
        "pathognomonic": "multiple bilateral ground-glass nodules / adenomatous hyperplasia in never-smoker with family history PATHOGNOMONIC for germline EGFR; germline T790M confirmed on blood WBC DNA — NOT somatic resistance",
    },
    {
        "gene": "STK11",
        "protein": (
            "STK11 -- 19p13.3 Autosomal-Dominant-LOF -- 433aa -- "
            "LKB1-Liver-Kinase-B1-Serine-Threonine-Kinase-Master-Metabolic-Regulator-AMPK-"
            "Peutz-Jeghers-Syndrome-Lung-Adenocarcinoma-7-17x-RR-"
            "KRAS-co-mutation-NSCLC-Immunotherapy-Resistance-OMIM-602216"
        ),
        "locus": "19p13.3",
        "protein_size": (
            "433 aa / 19p13.3 STK11 encodes Serine/Threonine Kinase 11 (LKB1 / liver kinase B1): "
            "STRUCTURE: "
            "  N-terminal nuclear localisation signal; "
            "  Catalytic kinase domain (residues 44-309): STK11 is serine/threonine kinase; "
            "    ATP binding: Gly-rich loop (G57-G62); catalytic base Asp194; "
            "    Activation loop Thr185 — STK11 is constitutively active (does not require phosphorylation for basal activity); "
            "  C-terminal domain (regulatory, STRAD/MO25 interaction); "
            "  Forms complex with STE20-related adaptor (STRAD) + mouse protein 25 (MO25) → cytoplasmic activation; "
            "MECHANISM — STK11/LKB1 TUMOUR SUPPRESSION: "
            "  Master kinase upstream of AMPK (AMP-activated protein kinase) + 13 AMPK-related kinases; "
            "  STK11 → phosphorylates AMPK (Thr172) → AMPK-ON; "
            "  AMPK-ON: "
            "    mTORC1 OFF (via TSC1/TSC2 and RAPTOR phosphorylation) → cell growth suppressed; "
            "    ACC (acetyl-CoA carboxylase) OFF → fatty acid synthesis suppressed; "
            "    PFK2 → glucose utilisation shifted; "
            "  STK11 LOF: AMPK-OFF → mTORC1 constitutive → metabolic reprogramming + proliferation; "
            "  STK11 controls polarity: asymmetric cell division, cell migration suppression; "
            "PEUTZ-JEGHERS SYNDROME (PJS) — LUNG CANCER: "
            "  LUNG ADENOCARCINOMA RISK: highest hereditary lung cancer syndromic risk; "
            "    Relative Risk: 7-17× above general population (varying cohort estimates); "
            "    Absolute lifetime risk (lung): 15-17% in PJS; "
            "    Histology: adenocarcinoma predominates (KRAS-associated molecular signature in sporadic); "
            "  SCLC: also elevated in PJS (2-3× above population); "
            "  PJS MUCOCUTANEOUS SIGNS: "
            "    Mucocutaneous melanin spots PATHOGNOMONIC: "
            "      Perioral (lips, buccal mucosa), periorbital, hands, feet; "
            "      Present in >95% PJS; may fade in adults but buccal mucosa spots persist; "
            "    GI hamartomatous polyps: small intestine > colon > stomach; "
            "      Intussusception risk (child): abdominal pain + obstruction → emergency surgery; "
            "  CANCER RISKS IN PJS: "
            "    CRC: 39% lifetime; GI cancer most common; "
            "    Pancreatic: 11-36% (highest pancreatic hereditary risk alongside BRCA2 and Lynch); "
            "    Breast: 45-50% in females; "
            "    Lung: 15-17%; ovarian: 21% (SCTAT PATHOGNOMONIC in females — also DICER1); "
            "    Cervical: 10%; testicular: Sertoli cell tumour (LCSCT) PATHOGNOMONIC in males; "
            "STK11 SOMATIC MUTATION IN NSCLC: "
            "  Somatic STK11 loss: 15-30% NSCLC (especially squamous + adenocarcinoma with KRAS); "
            "  STK11 + KRAS co-mutation in NSCLC: aggressive biology, immunotherapy resistance; "
            "    KRAS G12C + STK11 LOF: sotorasib/adagrasib may be less effective (still evaluating); "
            "    PD-L1 cold tumour despite high TMB: STK11 suppresses immune infiltration; "
            "    STK11 NSCLC responds poorly to anti-PD-1/PD-L1 (pembrolizumab, atezolizumab); "
            "TARGETED THERAPY FOR STK11-LOF NSCLC: "
            "  mTOR inhibition: everolimus/temsirolimus — limited single-agent activity; "
            "  KRAS G12C inhibitors: sotorasib/adagrasib regardless of STK11 status; "
            "  Phenformin (biguanide): preclinical LKB1-null sensitisation (AMPK-independent mechanism); "
            "  STING pathway activation: STK11-null impairs innate immunity → STING agonists in trials"
        ),
        "inheritance": (
            "AD LOF 19p13.3 — STK11. PJS prevalence ~1:50,000-1:200,000. "
            "De novo STK11: ~45% PJS (large intragenic deletions + truncating variants most common). "
            "Whole-gene deletion: 30% pathogenic STK11 — MLPA mandatory if sequencing negative in clinical PJS. "
            "PJS diagnostic criteria: ≥2 histologically confirmed PJS-type polyps; OR any PJS polyps + mucocutaneous spots; "
            "OR any PJS polyps + positive family history; OR mucocutaneous spots + family history. "
            "STK11 missense: pathogenicity confirmed by kinase activity assay. "
            "Cascade testing: all first-degree relatives."
        ),
        "surveillance_key": "annual LDCT chest from age 35; GI surveillance annual small bowel imaging from age 8-10; annual breast MRI from age 25; annual pancreatic MRI/EUS from age 30; mucocutaneous spots PATHOGNOMONIC — present before polyps in children",
        "pathognomonic": "mucocutaneous melanin spots PATHOGNOMONIC for PJS — perioral, buccal, acral; SCTAT ovary in females PATHOGNOMONIC; LCSCT testis in males PATHOGNOMONIC; GI hamartomatous polyp histology PATHOGNOMONIC",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-Tumour-Suppressor-Guardian-Genome-Transcription-Factor-"
            "Li-Fraumeni-Syndrome-SCLC-Lung-AVOID-RADIATION-ABSOLUTELY-"
            "WBMRI-Toronto-Protocol-Annual-OMIM-191170"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / 17p13.1 TP53 encodes tumour protein p53: "
            "STRUCTURE: "
            "  N-terminal transactivation domain (TAD1 + TAD2 — MDM2-binding site TAD1); "
            "  Proline-rich region (apoptosis signalling); "
            "  Central DNA-binding domain (DBD; residues 100-290): 90%+ cancer mutations cluster here; "
            "    Hotspot residues: R175, G245, R248, R249, R273, R282 (contact mutations, structural mutations); "
            "  Tetramerisation domain (p53 functions as homotetramer = 2 dimers); "
            "  C-terminal regulatory domain (acetylation K373/K382/K320; ubiquitination K305/K370/K372); "
            "  MDM2 negative-feedback: MDM2 transcribed by p53 → MDM2 ubiquitinates p53 → proteasomal degradation; "
            "LI-FRAUMENI SYNDROME (LFS) — LUNG CANCER COMPONENT: "
            "  LFS CORE TUMOUR SPECTRUM: "
            "    Osteosarcoma/soft tissue sarcoma: 35% (most common); "
            "    Pre-menopausal breast Ca (<35yr): 28-30%; "
            "    Brain tumours (glioma, medulloblastoma, choroid plexus Ca): 15%; "
            "    Adrenocortical carcinoma (ACC) paediatric: PATHOGNOMONIC in child <5yr; "
            "    Lung cancer (adenocarcinoma and SCLC): 6-10% LFS cohort cumulative; "
            "      SCLC in germline TP53: rare (<50yr) — typically aggressive; "
            "      Adenocarcinoma: any LFS family with TP53 + non-smoker lung Ca → germline testing; "
            "  AVOID RADIATION ABSOLUTELY: "
            "    Radiation-field sarcoma documented in LFS patients after therapeutic RT; "
            "    Radiation induces DNA DSB → TP53-deficient cells cannot undergo apoptosis/checkpoint → sarcoma; "
            "    Thoracic RT for lung NSCLC in germline TP53: CONTRAINDICATED; "
            "    Systemic alternatives: chemotherapy/TKI/immunotherapy — NO RADIATION in LFS; "
            "  WBMRI PROTOCOL (Toronto/Villani 2016): "
            "    Annual whole-body MRI + brain MRI (detects internal tumours before symptoms); "
            "    Annual abdominal US (ACC paediatric + adult adrenal surveillance); "
            "    Biennial mammography + annual breast MRI (from age 20-25); "
            "    Colonoscopy every 2-5yr from age 25; "
            "    Lung surveillance in LFS: annual LDCT in smokers or family lung Ca history; "
            "SOMATIC TP53 IN LUNG CANCER: "
            "  Somatic TP53 mutation: ~50% NSCLC, ~90% SCLC — most frequently altered gene in lung Ca; "
            "  TP53 somatic does not indicate germline — only specific variants (R248W, G245S hotspots) "
            "    in young patients without tobacco history warrant germline testing; "
            "  TP53 null / p53 IHC absent OR strong diffuse staining (gain-of-function hotspot): carcinoma phenotype; "
            "SYSTEMIC TREATMENT IN GERMLINE TP53 LFS NSCLC: "
            "  Platinum-based doublet: standard first-line NSCLC (carboplatin/cisplatin + pemetrexed); "
            "  Immunotherapy: pembrolizumab/atezolizumab — usable in LFS (non-radiation modality); "
            "  MDM2 inhibitors: APG-115 + pembrolizumab (Phase 2) — MDM2 amplification NSCLC; "
            "  PARP inhibitors: preclinical basis in TP53/p53 null — HRD co-pathway"
        ),
        "inheritance": (
            "AD LOF 17p13.1 — TP53. LFS prevalence ~1:5,000-1:20,000. "
            "De novo TP53: 7-20% LFS families. "
            "TP53 missense vs truncating: GOF missense (R175H, R273H, R248W) may show dominant-negative + GOF oncogenic activity. "
            "TP53 p.R337H (Brazilian founder): lower-penetrance, ACC predominant. "
            "AVOID ALL RADIATION in germline TP53 carriers — therapeutic RT causes radiation-field sarcoma (documented risk). "
            "Cascade testing all first-degree relatives. Prenatal/preimplantation genetic testing available."
        ),
        "surveillance_key": "AVOID ALL RADIATION absolutely in germline TP53; annual WBMRI + brain MRI Toronto protocol; annual LDCT if smoking or family lung Ca; annual abdominal US ACC; breast MRI from age 20-25; NO THORACIC RT even for lung Ca — use systemic",
        "pathognomonic": "ACC in child <5yr PATHOGNOMONIC for LFS — test TP53 immediately; radiation-field sarcoma after RT documents pre-existing germline TP53; multiple primaries <45yr (lung + sarcoma + brain) PATHOGNOMONIC LFS",
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 Autosomal-Dominant-LOF -- 928aa -- "
            "pRB-Retinoblastoma-Protein-E2F-Transcription-Factor-Repressor-"
            "Hereditary-Retinoblastoma-Secondary-SCLC-15-20x-Elevated-"
            "Bilateral-Rb-Germline-Until-Proven-AVOID-RADIATION-OMIM-614041"
        ),
        "locus": "13q14.2",
        "protein_size": (
            "928 aa / 13q14.2 RB1 encodes Retinoblastoma protein (pRB): "
            "STRUCTURE: "
            "  N-terminal domain (binding NHE1, MDM2); "
            "  Pocket domain A and B (AA 379-792): LXCXE motif-binding cleft (viral oncoproteins E1A, E7, T antigen bind here); "
            "    E2F transcription factor binding in pocket domain; "
            "    Spacer region between A and B (between 582-640); "
            "  C-terminal domain: cyclin-binding + nuclear export signal; "
            "pRB MECHANISM — CELL CYCLE GATEKEEPER: "
            "  Hypophosphorylated pRB: binds and represses E2F transcription factors → S-phase gene OFF → G1 arrest; "
            "  CDK4/6-Cyclin D → phosphorylate pRB (early G1 phosphorylation) → partial E2F release; "
            "  CDK2-Cyclin E → hyperphosphorylate pRB (mid-late G1) → full E2F release → S-phase entry; "
            "  CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib) → keep pRB hypophosphorylated → G1 arrest; "
            "HEREDITARY RETINOBLASTOMA — SECONDARY SCLC: "
            "  Bilateral retinoblastoma = GERMLINE RB1 UNTIL PROVEN OTHERWISE; "
            "    Unilateral Rb: 85% somatic, 15% germline (younger age + family history favour germline); "
            "    Bilateral Rb: >95% germline; "
            "  Trilateral retinoblastoma (pineoblastoma): 5% germline Rb — intracranial PATHOGNOMONIC; "
            "SECONDARY SCLC IN HEREDITARY RETINOBLASTOMA: "
            "  Most common secondary cancer in Rb survivors: osteosarcoma > soft tissue sarcoma > SCLC; "
            "  SCLC (small cell lung cancer): 15-20× elevated RR in germline RB1 carriers; "
            "    RB1 deletion: somatic hallmark of SCLC — 90% SCLC have biallelic RB1 loss; "
            "    Germline RB1 heterozygosity → SCLC risk begins in late adulthood (50-60yr); "
            "    Radiation-field SCLC: additional risk if Rb treated with radiotherapy (now rare in modern Rb management); "
            "  AVOID RADIATION IN-FIELD for secondary cancers: "
            "    Radiation-field osteosarcoma + SCLC documented risk in Rb survivors; "
            "    Use systemic chemotherapy / TKI where possible; "
            "RB1 IN NSCLC SOMATIC BIOLOGY: "
            "  RB1 somatic loss: 25% NSCLC adenocarcinoma, >90% SCLC; "
            "  RB1 loss + TP53 loss = classic SCLC molecular signature; "
            "  SCLC transformation from EGFR-mutant NSCLC (after TKI resistance): "
            "    Acquired RB1 + TP53 biallelic loss → histologic transformation to SCLC; "
            "    Monitor for SCLC transformation on EGFR TKI by serial biopsy; "
            "TARGETED THERAPY FOR RB1-LOSS TUMOURS: "
            "  CDK4/6 inhibitors (palbociclib, ribociclib): inactive in RB1-null (pRB substrate absent — no target); "
            "  SCLC: etoposide + carboplatin + atezolizumab (FDA 2019 — IMpower133); "
            "  Trilateral Rb pineoblastoma: methotrexate + carboplatin + etoposide + autologous SCT; "
            "  Aurora kinase inhibitors: SCLC with RB1 loss — early trials; "
            "Rb SURVEILLANCE IN SURVIVORS (secondary cancer monitoring): "
            "  Annual LDCT chest from age 40 (secondary SCLC + osteosarcoma); "
            "  Annual clinical exam + CBC; "
            "  MRI brain (pineoblastoma follow-up — trilateral Rb survivors)"
        ),
        "inheritance": (
            "AD LOF 13q14.2 — RB1. Hereditary Rb prevalence ~1:17,000. "
            "De novo RB1: ~40-50% germline cases (new family). "
            "Mosaicism: ~2-3% germline Rb — parent may have milder eye phenotype or normal. "
            "Low-penetrance RB1: specific missense + in-frame variants → unilateral late-onset Rb; germline low-penetrance variants. "
            "Anticipation: none documented (unlike trinucleotide repeat disorders). "
            "AVOID RADIATION in germline RB1: radiation-field sarcoma + SCLC documented risk."
        ),
        "surveillance_key": "annual LDCT chest from age 40 in Rb survivors; bilateral Rb = germline RB1 until proven; AVOID RADIATION-FIELD treatments (sarcoma + SCLC risk); trilateral pineoblastoma MRI surveillance; CDK4/6 inhibitors inactive in RB1-null SCLC",
        "pathognomonic": "bilateral retinoblastoma = germline RB1 PATHOGNOMONIC until proven otherwise; trilateral retinoblastoma (pineoblastoma) PATHOGNOMONIC for germline Rb; secondary SCLC in Rb survivor PATHOGNOMONIC RB1 loss",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "Breast-Cancer-Gene-2-HR-Repair-Scaffold-RAD51-Filament-"
            "HBOC-Syndrome-Lung-Cancer-2-3x-Elevated-Platinum-Sensitivity-"
            "PARP-Inhibitor-Olaparib-Rucaparib-OMIM-600185"
        ),
        "locus": "13q12.3",
        "protein_size": (
            "3418 aa / 13q12.3 BRCA2 encodes BRCA2 tumour suppressor / HR repair scaffold: "
            "STRUCTURE: "
            "  N-terminal PALB2-binding domain (PALB2 bridges BRCA1-BRCA2 interaction); "
            "  8 BRC repeats (residues 1002-2085): each binds one RAD51 monomer; "
            "  DNA-binding domain (DBD, residues 2396-3186): "
            "    OB-folds (OB1-OB4): single-stranded and double-stranded DNA binding; "
            "    Tower domain (OB3 insertion): minor groove of dsDNA; "
            "  C-terminal RAD51-binding domain (residues 3265-3330): DNA-damage-specific RAD51 interaction; "
            "  Nuclear export signal (NES); "
            "BRCA2 FUNCTION IN HOMOLOGOUS RECOMBINATION: "
            "  DSB signalling → RPA binds ssDNA of resected end; "
            "  BRCA2 loaded onto ssDNA via PALB2/BRCA1; "
            "  BRC repeats load RAD51 monomers onto ssDNA → nucleoprotein filament; "
            "  RAD51 filament: strand invasion of sister chromatid template → faithful repair; "
            "  BRCA2 LOF → no RAD51 filament → non-homologous end joining (error-prone) → genomic instability; "
            "BRCA2 LUNG CANCER — HBOC SECONDARY RISK: "
            "  Lung cancer: 5th most common BRCA2-associated cancer; "
            "    Meta-analysis RR: 1.8-2.5× above population lifetime risk for lung; "
            "    Absolute 70yr cumulative risk: ~5-7% in BRCA2 carriers; "
            "    Predominantly NSCLC adenocarcinoma; KRAS or EGFR co-occurring variants common; "
            "    Earlier onset: lung Ca before age 50 in BRCA2 carriers more common than sporadic; "
            "  Mechanism: homologous recombination deficiency (HRD) → NSCLC with BRCAness signature; "
            "    BRCA2 NSCLC: platinum hypersensitivity → carboplatin doublet favoured; "
            "    PARP inhibitor activity in BRCA2-mutant NSCLC: "
            "      Olaparib (Lynparza): FDA approved BRCA-mutant NSCLC (companion diagnostic); "
            "      Rucaparib: BRCA2-mutant NSCLC studies ongoing; "
            "      Niraparib: Phase 2 data in HRD+ NSCLC; "
            "BRCA2 DOMINANT CANCER SPECTRUM (HBOC): "
            "  Female breast Ca: 69-83% lifetime (BRCA2 moderate-high risk); "
            "  Ovarian Ca: 11-17% lifetime (BRCA2 < BRCA1 ovarian risk); "
            "  Male breast Ca: 5-7% (BRCA2 > BRCA1 for male breast); "
            "  Pancreatic Ca: 5-7% (BRCA2 highest pancreatic hereditary risk after PALB2/STK11); "
            "  Prostate Ca: 20-34% (aggressive, early-onset); "
            "  Melanoma: 2-5% (cutaneous and uveal); "
            "  Lung Ca: 5-7%; "
            "PARP INHIBITOR MECHANISM: "
            "  PARP1 trapping: olaparib traps PARP1 on DNA → SSB → replication fork collapse → DSB; "
            "  In BRCA2 LOF cell: DSB cannot be repaired by HR → synthetic lethality → cell death; "
            "  BRCAness concept: homologous recombination deficiency (HRD) predicts PARP response"
        ),
        "inheritance": (
            "AD LOF 13q12.3 — BRCA2. BRCA2 pathogenic variant prevalence ~1:300-1:800 general population. "
            "Ashkenazi Jewish founder: c.5946delT (6174delT) — most common (~1:40 Ashkenazi). "
            "Also: c.8765delAG (Ashkenazi), 999del5 (Icelandic founder). "
            "BRCA2 biallelic (AR): Fanconi Anaemia Complementation Group D1 (FANCD1) — medulloblastoma, Wilms, leukaemia in childhood. "
            "PALB2 (16p12.2): intermediate HBOC risk — BRCA2 partner; pathogenic PALB2 ≈ BRCA2 breast risk in some families. "
            "Cascade testing all first-degree relatives of BRCA2-positive index."
        ),
        "surveillance_key": "annual breast MRI from age 25 (BRCA2 carriers); risk-reducing bilateral salpingo-oophorectomy age 40-45 (BRCA2); olaparib/rucaparib for BRCA2-mutant NSCLC; platinum-based doublet preferred (HRD sensitivity); annual LDCT if smoking history or family lung Ca",
        "pathognomonic": "male breast cancer + BRCA2 family history PATHOGNOMONIC for BRCA2 GOF; Fanconi Anaemia FANCD1 (biallelic BRCA2) = medulloblastoma + Wilms in childhood PATHOGNOMONIC; platinum hypersensitivity PATHOGNOMONIC for HRD/BRCA2",
    },
    {
        "gene": "FLCN",
        "protein": (
            "FLCN -- 17p11.2 Autosomal-Dominant-LOF -- 579aa -- "
            "Folliculin-mTOR-AMPK-Regulator-"
            "Birt-Hogg-Dube-Syndrome-Pulmonary-Cysts-PATHOGNOMONIC-"
            "Spontaneous-Pneumothorax-24-38pct-Lung-Cancer-Risk-OMIM-607273"
        ),
        "locus": "17p11.2",
        "protein_size": (
            "579 aa / 17p11.2 FLCN encodes Folliculin: "
            "STRUCTURE: "
            "  N-terminal DENN-like domain (Differentially Expressed in Normal and Neoplastic cells); "
            "  C-terminal DENN-like domain; "
            "  FLCN forms complex with FNIP1/FNIP2 (folliculin-interacting proteins) → "
            "    interacts with AMPK + mTOR pathways; "
            "  FLCN LOF → RagC/D GTPase dysregulation → constitutive mTORC1 activation; "
            "  HIF pathway: FLCN modulates HIF1α/HIF2α signalling; "
            "BIRT-HOGG-DUBÉ SYNDROME (BHD) — LUNG MANIFESTATIONS: "
            "  PULMONARY CYSTS — PATHOGNOMONIC FINDING: "
            "    Multiple bilateral cysts: PATHOGNOMONIC for BHD in appropriate clinical context; "
            "    Cyst morphology: thin-walled, variably sized (1mm-20cm), predominantly basilar + medial; "
            "    Adjacent to pulmonary vessels, subpleural distribution; "
            "    Cysts on CT: bilateral + basal + subpleural distribution = high BHD specificity; "
            "    CT chest: >5 bilateral cysts in BHD criteria; "
            "  SPONTANEOUS PNEUMOTHORAX: "
            "    24-38% of BHD patients develop spontaneous pneumothorax; "
            "    Recurrence rate very high: 75% after first spontaneous pneumothorax in BHD; "
            "    Management: pleurodesis early after first episode (not watchful waiting as for primary PSP); "
            "    Family members with recurrent spontaneous pneumothorax → BHD testing mandatory; "
            "  LUNG CANCER IN BHD: "
            "    Absolute lung cancer risk: elevated but less well quantified vs renal cancer; "
            "    Mechanism: FLCN LOF + mTORC1 → pulmonary tumorigenesis; "
            "    Lung adenocarcinoma reported in BHD cohorts; "
            "BHD RENAL CANCER (DOMINANT MANIFESTATION): "
            "  Renal cancer: 5-34% BHD lifetime; hybrid oncocytic/chromophobe histology PATHOGNOMONIC BHD; "
            "  Also: chromophobe RCC, clear-cell RCC, oncocytoma; "
            "  Bilateral + multifocal renal tumours common in BHD; "
            "  Onset: typically 40-50yr (younger than sporadic chromophobe RCC); "
            "BHD CUTANEOUS MANIFESTATIONS: "
            "  Fibrofolliculomas: PATHOGNOMONIC — white dome-shaped facial/neck/trunk papules; "
            "    Multiple fibrofolliculomas = clinical BHD diagnosis requires genetic confirmation; "
            "  Trichodiscomas: benign hair follicle tumours (same spectrum); "
            "  Acrochordons (skin tags): very common in BHD; "
            "TARGETED THERAPY: "
            "  mTOR inhibition (everolimus): active in BHD-related renal tumours (FLCN-null → mTORC1 up); "
            "  Sunitinib / cabozantinib: used for BHD RCC (multi-kinase antiangiogenic); "
            "  Pleurodesis: for recurrent spontaneous pneumothorax (mechanical/chemical); "
            "SURVEILLANCE: "
            "  Annual abdominal MRI/CT from age 20: renal tumour surveillance; "
            "  CT chest baseline: document cyst burden; repeat only if symptomatic; "
            "  Dermatology annual: skin lesions; "
            "  Pneumothorax safety planning: patients warned to avoid high-altitude/diving activities"
        ),
        "inheritance": (
            "AD LOF 17p11.2 — FLCN. BHD prevalence ~1:200,000 (likely underdiagnosed). "
            "De novo FLCN: ~10-15% BHD. "
            "Most pathogenic: frameshift, nonsense, splice-site truncating variants; "
            "  Hotspot: exon 11 C8 insertion-deletion run (c.1733dupC and c.1285dupC most common). "
            "FLCN mosaic: unilateral/milder cutaneous + pulmonary findings. "
            "Recurrent spontaneous pneumothorax without family history: FLCN sequencing in young patients. "
            "Familial renal cancer + pulmonary cysts: FLCN panel priority."
        ),
        "surveillance_key": "annual abdominal MRI from age 20 (renal); CT chest baseline at diagnosis; pleurodesis after FIRST spontaneous pneumothorax (BHD recurrence rate 75%); fibrofolliculomas PATHOGNOMONIC — dermatology annual; avoid diving/altitude activities",
        "pathognomonic": "bilateral basal subpleural pulmonary cysts PATHOGNOMONIC for BHD on CT; fibrofolliculomas facial/neck/trunk PATHOGNOMONIC; hybrid oncocytic/chromophobe RCC PATHOGNOMONIC BHD histology; spontaneous pneumothorax + pulmonary cysts = BHD until proven",
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 Biallelic-AR-LOF-or-Monoallelic-AD-LOF -- 3056aa -- "
            "Ataxia-Telangiectasia-Mutated-PI3K-Like-Kinase-"
            "Ataxia-Telangiectasia-Biallelic-RADIOSENSITIVITY-ABSOLUTE-"
            "AT-Carrier-Lung-Cancer-2-4x-RR-PARP-Inhibitor-Sensitivity-OMIM-607585"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "3056 aa / 11q22.3 ATM encodes ATM serine/threonine kinase: "
            "STRUCTURE: "
            "  FAT domain (FRAP-ATM-TRRAP): N-terminal structural scaffold; "
            "  Kinase domain: PI3K-like kinase (PIKK family); "
            "    Substrates: H2AX (γH2AX — DSB marker), BRCA1, CHK1, CHK2, p53 (Ser15), MDM2; "
            "  FAT-C domain (C-terminal); "
            "  ATM monomer: ~350kDa; active as dimer (DSB) or monomer (oxidative stress); "
            "ATM MECHANISM — DNA DAMAGE RESPONSE MASTER KINASE: "
            "  Double-strand break (DSB) sensing: Mre11-Rad50-NBS1 (MRN complex) recruits ATM to DSB; "
            "  ATM autophosphorylation Ser1981 → ATM monomerisation → active kinase; "
            "  H2AX phosphorylation (γH2AX): focal signal marks DSB at >50 flanking Mb; "
            "    γH2AX recruits MDC1 → RNF8/RNF168 ubiquitin cascade → 53BP1/BRCA1 recruitment; "
            "  CHK2 phosphorylation: ATM → CHK2 Thr68 → G1/S + G2/M checkpoint arrest; "
            "  p53 Ser15 phosphorylation: ATM activates p53 → apoptosis/senescence; "
            "  HR pathway: ATM phosphorylates BRCA1 → promotes HR over NHEJ in S/G2 phase; "
            "ATAXIA-TELANGIECTASIA (A-T) — BIALLELIC AR LOF: "
            "  RADIOSENSITIVITY — ABSOLUTE CONTRAINDICATION TO STANDARD-DOSE RADIOTHERAPY: "
            "    A-T cells: 3-5× hypersensitive to ionising radiation; "
            "    Standard RT doses (60Gy for lung cancer) = potentially lethal in A-T; "
            "    Radiation sensitivity assay (clonogenic survival) confirms A-T; "
            "    Management: AVOID ALL STANDARD-DOSE RT — use systemic therapy exclusively; "
            "  LUNG CANCER IN A-T: "
            "    Lung cancer ~10% A-T lifetime cumulative risk; "
            "    Predominantly NSCLC adenocarcinoma; "
            "    Biallelic ATM → complete HR deficiency → PARPi + platinum hypersensitivity; "
            "  Other A-T manifestations: "
            "    Cerebellar ataxia onset 1-3yr: progressive → wheelchair by 10yr; "
            "    Oculomotor apraxia: hallmark cerebellar sign; "
            "    Telangiectasias: conjunctival (age 3-6yr), cutaneous; "
            "    Immunodeficiency: IgA, IgE, IgG2 deficiency; "
            "    Lymphoid malignancy: leukaemia + lymphoma major cause of mortality in A-T; "
            "    Elevated AFP: characteristic A-T lab marker; "
            "    Insulin resistance and metabolic syndrome; "
            "ATM MONOALLELIC (HETEROZYGOUS) CARRIER LUNG CANCER RISK: "
            "  ~1:100 general population are ATM carriers; "
            "  Monoallelic ATM LOF: lung cancer RR 2-4× above population (meta-analysis); "
            "    Moderate HRD (haploinsufficiency) → partial DDR deficiency; "
            "    Platinum sensitivity preserved in carrier-origin tumours; "
            "    PARP inhibitor: olaparib approved for ATM-mutant solid tumours (prostate, ovarian FDA); "
            "    Emerging: olaparib / niraparib for ATM-mutant NSCLC; "
            "TARGETED THERAPY FOR ATM-MUTANT NSCLC: "
            "  Platinum-based doublet: ATM LOF → HRD → carboplatin/cisplatin preferred; "
            "  PARP inhibitors: olaparib (FDA approved ATM-mutant prostate — activity in NSCLC ATM); "
            "  ATR inhibitors: ATM LOF → cells become ATR-dependent for S-phase survival; "
            "    Ceralasertib (AZD6738, ATRi): clinical trials ATM-mutant NSCLC (PATRIOT trial); "
            "  WEE1 inhibitor (adavosertib): ATM-null → G2 checkpoint bypass → mitotic catastrophe"
        ),
        "inheritance": (
            "Biallelic AR 11q22.3 → A-T. Monoallelic AD → elevated cancer risk (lung, breast, prostate). "
            "A-T prevalence: ~1:40,000-1:100,000. ATM carrier frequency: ~1:100 general population. "
            "Pathogenic ATM variants: truncating (frameshift, nonsense, splice) + missense (kinase domain). "
            "ATM missense pathogenicity: functional kinase assay; most missense are VUS. "
            "AVOID RADIATION in confirmed biallelic A-T — absolute; "
            "ATM carriers (monoallelic): standard RT doses generally tolerable (haploinsufficiency, not complete loss). "
            "Cascade testing first-degree relatives of biallelic A-T proband (parents obligate carriers)."
        ),
        "surveillance_key": "AVOID ALL STANDARD-DOSE RADIATION in biallelic A-T — lethal radiosensitivity; ATM mono carriers lung cancer 2-4× RR — annual LDCT from age 40-45; olaparib/ceralasertib for ATM-mutant NSCLC; ATM IHC loss + NSCLC → PARPi consideration; AFP elevated = characteristic A-T marker",
        "pathognomonic": "cerebellar ataxia onset 1-3yr + oculomotor apraxia + telangiectasia PATHOGNOMONIC A-T; elevated AFP with cerebellar ataxia PATHOGNOMONIC A-T; radiosensitivity assay 3-5× → A-T confirmed; ATM biallelic = absolute contraindication to standard radiotherapy",
    },
    {
        "gene": "BAP1",
        "protein": (
            "BAP1 -- 3p21.1 Autosomal-Dominant-LOF -- 729aa -- "
            "BRCA1-Associated-Protein-1-Deubiquitinase-H2A-Ub-Polycomb-"
            "BAP1-Tumour-Predisposition-Syndrome-Mesothelioma-PATHOGNOMONIC-30-60x-RR-"
            "Uveal-Melanoma-ccRCC-Lung-Adenocarcinoma-OMIM-603089"
        ),
        "locus": "3p21.1",
        "protein_size": (
            "729 aa / 3p21.1 BAP1 encodes BRCA1-Associated Protein-1 (BAP1): "
            "STRUCTURE: "
            "  N-terminal UCH (ubiquitin carboxy-terminal hydrolase) domain (residues 1-240): "
            "    Catalytic cysteine (C91): active site deubiquitinase; "
            "    Removes mono-ubiquitin from H2A K119 (H2Aub1) → transcriptional activation of Polycomb target genes; "
            "  C-terminal HBM (HCF-1 binding motif) domain: ASXL1/ASXL2 interaction → ASXL-BAP1 complex; "
            "  NLS (nuclear localisation signal): nuclear BAP1; "
            "  BRCA1 binding: C-terminal BRCT domain contacts; "
            "BAP1 FUNCTION — CHROMATIN REMODELLING AND TUMOUR SUPPRESSION: "
            "  PR-DUB complex (Polycomb Repressive Deubiquitinase): "
            "    BAP1 + ASXL1/2 + FOXK1/2 + OGT (O-GlcNAc transferase); "
            "    H2Aub1 removal: activates Polycomb-silenced genes (cell differentiation, apoptosis pathways); "
            "  BAP1 LOF → H2Aub1 accumulates → Polycomb target gene silencing → dedifferentiation; "
            "  DNA damage repair: BAP1 + BARD1 + BRCA1 → HR pathway coordination; "
            "  Regulation of YAP/TAZ, HIF-1α, PI3K-Akt — multiple oncogenic pathways; "
            "BAP1 TUMOUR PREDISPOSITION SYNDROME (TPDS): "
            "  MESOTHELIOMA — PATHOGNOMONIC ASSOCIATION: "
            "    Pleural mesothelioma: 30-60× elevated RR in BAP1 carriers (vs general population); "
            "    Peritoneal mesothelioma: also elevated; "
            "    ASBESTOS SYNERGY LETHAL: BAP1 LOF + asbestos exposure = multiplicative (not additive) carcinogenesis; "
            "      BAP1 mesothelioma latency with asbestos: 15-25yr (shorter than sporadic ~40yr); "
            "      Mesothelioma WITHOUT asbestos exposure: test BAP1 germline; "
            "      Any mesothelioma + family history mesothelioma → BAP1 germline mandatory; "
            "    Somatic BAP1: 60% sporadic mesothelioma biallelic loss; "
            "  UVEAL MELANOMA: "
            "    Uveal (choroidal/ciliary body) melanoma: 35-50% lifetime in TPDS; "
            "    Onset: typically 50-60yr (younger in TPDS vs sporadic); "
            "    BAP1 IHC loss in uveal melanoma: strongly correlates with metastatic potential; "
            "      Class 2 uveal melanoma (chromosome 3 monosomy + BAP1 loss): >75% 5yr metastatic rate; "
            "    Annual ophthalmology exam from diagnosis of TPDS: "
            "      Fundoscopy + ultrasound biomicroscopy; "
            "  LUNG ADENOCARCINOMA: "
            "    Lung Ca: 5-7× elevated in TPDS; predominantly adenocarcinoma; "
            "    Mechanism: BAP1 somatic deletion in 3p21.1 region; "
            "    Annual LDCT from age 40 in BAP1 carriers; "
            "    Avoid asbestos exposure: primary prevention — occupational history mandatory; "
            "  CLEAR CELL RENAL CARCINOMA (ccRCC): "
            "    ccRCC: 5-10% TPDS lifetime; younger onset; "
            "    BAP1 somatic loss in 5-15% ccRCC; "
            "  CUTANEOUS MELANOCYTIC TUMOURS: "
            "    BAP1-inactivated naevus (BIN): benign-appearing melanocytic lesion with BAP1 IHC loss; "
            "    Multiple BINs: virtually diagnostic of TPDS; "
            "    Cutaneous melanoma: 4-5× elevated RR; "
            "BAP1 IHC IN SURGICAL PATHOLOGY: "
            "  Nuclear BAP1 staining: intact (normal tumour suppressor present); "
            "  Nuclear BAP1 loss: biallelic somatic BAP1 inactivation (mesothelioma, uveal melanoma); "
            "  BAP1 IHC panel mandatory in all pleural/peritoneal mesothelioma workup; "
            "TARGETED THERAPY FOR BAP1-MUTANT TUMOURS: "
            "  Mesothelioma first-line: pemetrexed + cisplatin (or carboplatin) ± bevacizumab; "
            "  Nivolumab + ipilimumab (CheckMate 743): FDA 2021 first-line mesothelioma; "
            "  PARP inhibitors: BAP1 LOF → partial HRD → olaparib preclinical activity (trials ongoing); "
            "  EZH2 inhibitors: BAP1 LOF → PRC2/EZH2 hyperactivation → EZH2i active (tazemetostat); "
            "    Tazemetostat (Tazverik): FDA 2020 mesothelioma (EZH2-mutant + wt)"
        ),
        "inheritance": (
            "AD LOF 3p21.1 — BAP1. TPDS prevalence: rare, estimated 1:50,000-1:200,000 (likely underdiagnosed). "
            "De novo BAP1: uncommon — most germline cases have family history if actively sought. "
            "Pathogenic variants: frameshift, nonsense, missense (UCH domain catalytic residues), large deletions. "
            "BAP1 large deletion: MLPA/CGH essential if sequencing negative in clinical TPDS. "
            "Founder variants: not yet well established (population-level data accumulating). "
            "ASBESTOS COUNSELLING: all BAP1 carriers must AVOID all asbestos exposure — occupational hazard. "
            "Cascade testing all first-degree relatives mandatory."
        ),
        "surveillance_key": "AVOID ALL ASBESTOS EXPOSURE (primary prevention — mesothelioma 30-60× RR + asbestos synergy); annual ophthalmology from diagnosis (uveal melanoma 35-50%); annual LDCT chest from age 40; annual abdominal MRI/US (renal); BAP1 IHC on all pleural/peritoneal mesothelioma; tazemetostat + nivolumab+ipilimumab for BAP1-mutant mesothelioma",
        "pathognomonic": "pleural/peritoneal mesothelioma + family history mesothelioma PATHOGNOMONIC for BAP1 germline; uveal melanoma + mesothelioma in same individual PATHOGNOMONIC TPDS; multiple BAP1-inactivated naevi (BIN) PATHOGNOMONIC TPDS; BAP1 IHC nuclear loss in mesothelioma + uveal melanoma PATHOGNOMONIC",
    },
]


def _gene_stats(seed: int, gene_cfg: dict) -> dict:
    rng = random.Random(seed)
    gene = gene_cfg["gene"]

    # Gene-specific base parameters (realistic ranges for each gene/syndrome)
    base = {
        "EGFR":  dict(age_mu=52, age_sd=9,  lung_pct=(70,85), pneumo_pct=(5,12),   smoke_never_pct=(60,80), brain_met_pct=(25,40)),
        "STK11": dict(age_mu=43, age_sd=11, lung_pct=(40,55), gi_polyp_pct=(85,95), pancreas_pct=(15,28),   skin_pct=(90,98)),
        "TP53":  dict(age_mu=40, age_sd=12, lung_pct=(20,35), sarcoma_pct=(30,42),  breast_pct=(25,38),     acc_pct=(8,18)),
        "RB1":   dict(age_mu=55, age_sd=10, sclc_pct=(18,28), bilateral_rb_pct=(90,98), osteosarcoma_pct=(20,32), radiation_avoid_pct=(85,95)),
        "BRCA2": dict(age_mu=54, age_sd=10, lung_pct=(25,38), breast_pct=(65,80),   pancreatic_pct=(8,14),   platinum_sens_pct=(55,75)),
        "FLCN":  dict(age_mu=41, age_sd=11, pneumo_pct=(22,38), cyst_pct=(85,95),   renal_pct=(20,35),      fibrofolliculoma_pct=(70,88)),
        "ATM":   dict(age_mu=48, age_sd=12, lung_pct=(18,28), ataxia_pct=(95,100),  telangiectasia_pct=(90,98), radiosensitive_pct=(85,98)),
        "BAP1":  dict(age_mu=52, age_sd=11, meso_pct=(28,45), uveal_mel_pct=(30,48), renal_pct=(8,16),       lung_pct=(15,25)),
    }
    b = base.get(gene, dict(age_mu=48, age_sd=10))

    def pct(lo, hi): return round(rng.uniform(lo, hi), 1)
    def age(): return round(rng.gauss(b["age_mu"], b["age_sd"]), 1)

    ages = [age() for _ in range(40)]
    mean_age = round(sum(ages) / len(ages), 1)

    # Build stats dict with gene-specific fields
    stats = dict(gene=gene, n=40, mean_age_diagnosis=mean_age)
    for key, (lo, hi) in {k: v for k, v in b.items() if k.endswith("_pct")}.items():
        stats[key] = pct(lo, hi)

    stats["locus"] = gene_cfg["locus"]
    stats["inheritance"] = gene_cfg["inheritance"]
    stats["surveillance_key"] = gene_cfg["surveillance_key"]
    stats["pathognomonic"] = gene_cfg["pathognomonic"]
    return stats


def generate_overview() -> dict:
    return {
        "atlas":          "Hereditary-Lung-Cancer-Atlas",
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_genes":    len(ATLAS_GENES),
        "total_patients": 320,
        "genes":          [g["gene"] for g in ATLAS_GENES],
        "inheritance_modes": {
            "EGFR": (
                "AD GOF 7p11.2 (EGFR RTK; 1210aa; Hereditary NSCLC; "
                "germline T790M — osimertinib 3rd-gen TKI preferred; "
                "multiple bilateral GGOs in never-smoker PATHOGNOMONIC; "
                "annual LDCT from age 30-35)"
            ),
            "STK11": (
                "AD LOF 19p13.3 (LKB1 AMPK-master kinase; 433aa; Peutz-Jeghers Syndrome; "
                "lung adenocarcinoma 7-17× RR — highest hereditary lung risk; "
                "mucocutaneous melanin spots PATHOGNOMONIC; "
                "STK11+KRAS NSCLC: PD-L1 cold — immunotherapy often fails)"
            ),
            "TP53": (
                "AD LOF 17p13.1 (p53 guardian genome; 393aa; Li-Fraumeni Syndrome; "
                "SCLC + lung Ca LFS component; AVOID ALL RADIATION ABSOLUTELY; "
                "WBMRI annually Toronto protocol; ACC <5yr PATHOGNOMONIC)"
            ),
            "RB1": (
                "AD LOF 13q14.2 (pRB E2F repressor; 928aa; Hereditary Retinoblastoma; "
                "secondary SCLC 15-20× elevated in Rb survivors; "
                "bilateral Rb = germline until proven; AVOID radiation-field SCLC; "
                "trilateral pineoblastoma PATHOGNOMONIC)"
            ),
            "BRCA2": (
                "AD LOF 13q12.3 (RAD51 scaffold HR repair; 3418aa; HBOC Syndrome; "
                "lung Ca 2-3× elevated — 5th BRCA2-associated cancer; "
                "platinum hypersensitivity + PARP inhibitor (olaparib) sensitivity; "
                "male breast Ca PATHOGNOMONIC BRCA2)"
            ),
            "FLCN": (
                "AD LOF 17p11.2 (folliculin mTOR regulator; 579aa; Birt-Hogg-Dubé Syndrome; "
                "bilateral basal pulmonary cysts PATHOGNOMONIC; "
                "spontaneous pneumothorax 24-38% — pleurodesis after first episode BHD; "
                "fibrofolliculomas PATHOGNOMONIC; hybrid chromophobe RCC PATHOGNOMONIC)"
            ),
            "ATM": (
                "Biallelic AR / Monoallelic AD 11q22.3 (ATM PI3K-like kinase; 3056aa; "
                "A-T biallelic: cerebellar ataxia + telangiectasia + radiosensitivity ABSOLUTE; "
                "ATM mono carriers lung Ca 2-4× RR; "
                "AVOID STANDARD-DOSE RT in biallelic A-T — lethal; "
                "olaparib + ceralasertib sensitivity)"
            ),
            "BAP1": (
                "AD LOF 3p21.1 (BAP1 H2A deubiquitinase Polycomb; 729aa; "
                "BAP1 Tumour Predisposition Syndrome; "
                "mesothelioma PATHOGNOMONIC 30-60× RR — asbestos synergy lethal; "
                "AVOID ALL ASBESTOS; uveal melanoma 35-50%; "
                "tazemetostat + nivolumab+ipilimumab mesothelioma)"
            ),
        },
        "key_clinical_rules": [
            "EGFR germline T790M: never-smoker lung adenocarcinoma + family history → germline EGFR panel; osimertinib (3rd-gen TKI) preferred over earlier-gen TKIs",
            "EGFR: multiple bilateral ground-glass nodules/adenomatous hyperplasia in never-smoker = PATHOGNOMONIC for germline EGFR — annual LDCT from age 30-35",
            "STK11/PJS: mucocutaneous melanin spots (perioral, buccal, acral) PATHOGNOMONIC — present before polyps in children; lung adenocarcinoma 7-17× RR — highest hereditary lung cancer syndromic risk",
            "STK11 NSCLC: STK11 + KRAS co-mutation → PD-L1 cold tumour — anti-PD-1/PD-L1 immunotherapy often fails despite high TMB; use KRAS G12C inhibitor (sotorasib/adagrasib) if applicable",
            "TP53 germline (LFS): AVOID ALL THORACIC RADIATION — standard-dose RT causes radiation-field sarcoma in germline TP53; use systemic alternatives exclusively",
            "RB1/Hereditary Retinoblastoma: bilateral retinoblastoma = germline RB1 until proven otherwise; secondary SCLC 15-20× elevated — annual LDCT from age 40 in Rb survivors",
            "RB1: CDK4/6 inhibitors (palbociclib/ribociclib/abemaciclib) are INACTIVE in RB1-null SCLC (pRB substrate absent) — do not use in RB1-deficient tumours",
            "BRCA2: lung cancer is the 5th BRCA2-associated cancer (2-3× RR); platinum-based doublet preferred (HRD sensitivity); olaparib FDA-approved BRCA2-mutant solid tumours",
            "FLCN/BHD: bilateral basal subpleural pulmonary cysts on CT PATHOGNOMONIC — pleurodesis after FIRST spontaneous pneumothorax (recurrence rate 75% vs 30% in primary PSP)",
            "FLCN: annual abdominal MRI from age 20 (hybrid chromophobe RCC PATHOGNOMONIC BHD histology); avoid asbestos-like occupational inhalation exposures (cyst + airway sensitivity)",
            "ATM biallelic (A-T): ABSOLUTE CONTRAINDICATION to standard-dose radiotherapy (3-5× radiosensitivity — lethal); AFP elevated is characteristic A-T lab marker; cerebellar ataxia + telangiectasia PATHOGNOMONIC",
            "ATM monoallelic carriers: standard RT doses generally tolerable (haploinsufficiency, not complete loss); lung Ca 2-4× RR — annual LDCT from age 40-45; olaparib + ceralasertib (ATRi) clinical activity",
            "BAP1/TPDS: AVOID ALL ASBESTOS EXPOSURE — mesothelioma 30-60× RR and asbestos synergy is multiplicative, not additive; occupational history mandatory at every visit",
            "BAP1: mesothelioma + family history of mesothelioma → BAP1 germline testing mandatory; uveal melanoma annual ophthalmology exam (35-50% TPDS lifetime risk); BAP1 IHC loss PATHOGNOMONIC on mesothelioma/uveal melanoma histology",
            "BAP1: tazemetostat (EZH2 inhibitor) FDA 2020 for mesothelioma; nivolumab + ipilimumab CheckMate 743 FDA 2021 first-line mesothelioma",
            "ALL HEREDITARY LUNG CANCER PANEL: germline EGFR + STK11 + TP53 + RB1 + BRCA2 + FLCN + ATM + BAP1 — indicated for never-smoker lung Ca <50yr, multiple primaries, family history lung Ca, bilateral pulmonary cysts or spontaneous pneumothorax",
        ],
        "gene_panel_note": (
            "Hereditary Lung Cancer Germline Panel (clinical 2024): "
            "EGFR KINASE GOF (hereditary NSCLC adenocarcinoma — germline T790M): "
            "  EGFR: never-smoker adenocarcinoma; bilateral GGOs; osimertinib; "
            "STK11/LKB1 PATHWAY (PJS — lung adenocarcinoma highest syndromic risk): "
            "  STK11: Peutz-Jeghers; mucocutaneous spots PATHOGNOMONIC; 7-17× lung RR; KRAS-co-mut immunotherapy resistance; "
            "TP53 PATHWAY (LFS — SCLC + lung, AVOID RADIATION ABSOLUTELY): "
            "  TP53: LFS; SCLC; thoracic RT contraindicated; WBMRI Toronto; ACC <5yr PATHOGNOMONIC; "
            "RB1 PATHWAY (Rb survivors — secondary SCLC 15-20×): "
            "  RB1: bilateral Rb = germline; secondary SCLC; CDK4/6i inactive in RB1-null; avoid radiation-field; "
            "HOMOLOGOUS RECOMBINATION (HRD) PATHWAY — BRCA2: "
            "  BRCA2: HBOC; lung Ca 5th cancer; platinum + PARP inhibitor (olaparib); male breast Ca PATHOGNOMONIC; "
            "mTOR/AMPK PATHWAY — FLCN/BHD (pulmonary cysts + pneumothorax): "
            "  FLCN: bilateral basal cysts PATHOGNOMONIC; spontaneous pneumothorax 24-38%; pleurodesis first episode; fibrofolliculomas; "
            "DNA DAMAGE RESPONSE (DDR) KINASE — ATM: "
            "  ATM biallelic (A-T): cerebellar ataxia + telangiectasia + RADIOSENSITIVITY ABSOLUTE; ataxia + AFP; "
            "  ATM monoallelic: lung Ca 2-4×; olaparib + ceralasertib (ATRi); "
            "BAP1 DEUBIQUITINASE / POLYCOMB (mesothelioma PATHOGNOMONIC): "
            "  BAP1: mesothelioma 30-60×; AVOID ASBESTOS; uveal melanoma; tazemetostat + nivo+ipi; BAP1 IHC loss PATHOGNOMONIC; "
            "UNIVERSAL TESTING CRITERIA: "
            "  ALL never-smoker lung adenocarcinoma: EGFR germline panel (exclude somatic T790M via WBC DNA); "
            "  ANY lung Ca <50yr without tobacco history: full 8-gene hereditary lung panel; "
            "  Bilateral pulmonary cysts + spontaneous pneumothorax: FLCN germline testing; "
            "  Mesothelioma without asbestos or <60yr: BAP1 germline testing; "
            "  Bilateral retinoblastoma survivor with lung Ca: RB1 germline + SCLC protocol; "
            "  Multiple primaries (lung + sarcoma/breast/brain): TP53 LFS germline"
        ),
    }


def generate_breakdown() -> dict:
    genes_data = []
    for i, gene_cfg in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        stats = _gene_stats(seed, gene_cfg)
        stats["protein_summary"] = gene_cfg["protein"]
        stats["locus"] = gene_cfg["locus"]
        stats["inheritance"] = gene_cfg["inheritance"]
        stats["surveillance_key"] = gene_cfg["surveillance_key"]
        stats["pathognomonic"] = gene_cfg["pathognomonic"]
        genes_data.append(stats)
    return {
        "atlas":          "Hereditary-Lung-Cancer-Atlas",
        "seed_range":     f"{SEED_BASE}-{SEED_BASE + 7}",
        "n_genes":        len(genes_data),
        "total_patients": 320,
        "genes":          genes_data,
    }


def generate_definitions() -> dict:
    definitions = [
        {
            "term": "EGFR-Germline-T790M-Hereditary-NSCLC-Osimertinib-Protocol",
            "definition": (
                "Germline EGFR T790M — Hereditary NSCLC Diagnosis and Osimertinib Management Protocol: "
                "RECOGNISING GERMLINE vs SOMATIC T790M: "
                "  Somatic T790M: acquired resistance to 1st/2nd-gen EGFR TKI in NSCLC; detected in tumour ctDNA; "
                "  Germline T790M: present in ALL tissues (blood/WBC DNA); heritable; "
                "  Distinction: "
                "    Test blood/WBC DNA separately from tumour/ctDNA; "
                "    If T790M in both germline WBC AND tumour: germline confirmed; "
                "    Germline T790M: family history lung Ca, never/light-smoker, early onset (<60yr); "
                "    Multiple bilateral ground-glass opacities (GGOs) or adenomatous hyperplasia: germline pattern; "
                "OSIMERTINIB FIRST-LINE FOR GERMLINE EGFR T790M: "
                "  3rd-generation irreversible EGFR TKI; "
                "  Mechanism: covalently binds C797 (catalytic cysteine) in EGFR kinase domain; "
                "  Overcomes T790M gatekeeper mutation resistance (1st/2nd gen TKI block by T790M); "
                "  FDA 2015: T790M-acquired resistance NSCLC; FDA 2018: 1st-line EGFR-mutant NSCLC (FLAURA); "
                "  Germline T790M + lung adenocarcinoma: osimertinib preferred over erlotinib/gefitinib/afatinib; "
                "  CNS penetration: superior to earlier-gen TKIs — brain metastases can be treated with osimertinib; "
                "SURVEILLANCE PROTOCOL FOR GERMLINE EGFR CARRIERS (not yet with cancer): "
                "  Annual low-dose CT (LDCT) chest from age 30-35 or 10yr before earliest family member's diagnosis; "
                "  GGO management: "
                "    Pure GGO <6mm: 12-month LDCT follow-up; "
                "    GGO ≥6mm or part-solid: 6-month LDCT follow-up + pulmonary oncology; "
                "    Part-solid nodule with solid component ≥6mm: biopsy; "
                "  Smoking cessation: absolute — smoking × EGFR mutation amplifies carcinogenesis; "
                "C797S RESISTANCE TO OSIMERTINIB: "
                "  C797S mutation (exon 20): most common osimertinib acquired resistance; "
                "  Management: "
                "    MARIPOSA trial (amivantamab + lazertinib): bispecific EGFR+MET × 3rd-gen TKI; "
                "    Chemotherapy: platinum-pemetrexed at progression; "
                "    GEOMETRY mono-1 (capmatinib): if MET amplification on resistance biopsy; "
                "  Liquid biopsy (ctDNA) for resistance mutation monitoring during osimertinib"
            ),
        },
        {
            "term": "STK11-PJS-Lung-Cancer-Immunotherapy-Resistance-KRAS-Protocol",
            "definition": (
                "STK11/LKB1 Peutz-Jeghers Syndrome — Lung Cancer Risk and Immunotherapy Resistance Protocol: "
                "WHY STK11/LKB1 LOSS CAUSES IMMUNOTHERAPY RESISTANCE IN NSCLC: "
                "  STK11 LOF → AMPK-OFF → mTORC1 ON + metabolic reprogramming; "
                "  STK11 loss impairs innate immune signalling: "
                "    Reduced STING (stimulator of interferon genes) pathway activation; "
                "    Suppressed type-I interferon response; "
                "    Decreased MHC-I expression on tumour cells → reduced T-cell recognition; "
                "  Result: PD-L1 cold tumour despite potentially high TMB; "
                "    Anti-PD-1 (pembrolizumab, nivolumab) FAILS in STK11-mutant NSCLC: "
                "      KEYNOTE-042: pembrolizumab 1st-line inferior in STK11-mutant NSCLC; "
                "      IMpower110, IMpower150: atezolizumab less effective in STK11-co-mutated; "
                "  KRAS + STK11 co-mutation in NSCLC: "
                "    Most aggressive NSCLC molecular subtype; "
                "    Combined metabolic (LKB1) + RAS signalling → high proliferation + immune exclusion; "
                "    KRAS G12C inhibitors (sotorasib/adagrasib): active regardless of STK11 status; "
                "    CODEX trial: adagrasib + pembrolizumab — STK11 co-mutation reduces benefit; "
                "TREATMENT DECISION ALGORITHM FOR STK11-MUTANT NSCLC: "
                "  Step 1: test KRAS mutation status; "
                "  Step 2: if KRAS G12C: sotorasib (Lumakras) or adagrasib (Krazati) — STK11 does not negate G12C inhibitor activity; "
                "  Step 3: if KRAS WT or non-G12C: platinum-based doublet ± bevacizumab; "
                "  Step 4: do NOT rely on pembrolizumab monotherapy (PD-L1 TPS) in STK11-mutant — combination if PD-L1 >50%; "
                "  Step 5: STING agonist clinical trials (e.g. STING agonist + anti-PD-1): STK11-mutant NSCLC population; "
                "PJS GI SURVEILLANCE FOR LUNG CANCER PATIENTS: "
                "  Annual small bowel imaging (MRE or VCE) from age 8-10; "
                "  Upper endoscopy + colonoscopy every 2-3yr from age 18; "
                "  Annual pancreatic MRI/EUS from age 30 (pancreatic Ca 11-36%)"
            ),
        },
        {
            "term": "BAP1-TPDS-Mesothelioma-Asbestos-Tazemetostat-Protocol",
            "definition": (
                "BAP1 Tumour Predisposition Syndrome — Mesothelioma and Asbestos Prevention Protocol: "
                "WHY BAP1 + ASBESTOS IS UNIQUELY DANGEROUS: "
                "  Normal mesothelioma latency: 30-50yr after first asbestos exposure; "
                "  BAP1 + asbestos latency: 15-25yr (significantly shorter); "
                "  Mechanism of synergy: "
                "    Asbestos fibres → ROS + DSBs + chromosomal instability in mesothelial cells; "
                "    BAP1 LOF → impaired H2A deubiquitination + reduced HR repair; "
                "    Combined: mesothelial cells cannot repair asbestos-induced DNA damage → mesothelioma; "
                "  Dose-response: even LOW-LEVEL asbestos exposure (not just occupational) can trigger mesothelioma in BAP1 carriers; "
                "  TAKE-HOME: BAP1 carriers must avoid ALL asbestos exposure — no threshold is safe; "
                "OCCUPATIONAL HISTORY AND PREVENTION: "
                "  Mandatory occupational history at EVERY genetics visit: "
                "    Past: construction (pre-1980s buildings), shipbuilding, mining, insulation, brake work; "
                "    Current: building renovation, demolition, firefighting; "
                "  Residential: pre-1980 building with damaged insulation — test + abate; "
                "  Family exposure: 2nd-hand asbestos (family member with occupational exposure); "
                "  Protective measures: N99/P100 respirator, full PPE, avoid dry disturbance; "
                "MESOTHELIOMA SURVEILLANCE IN BAP1 TPDS: "
                "  Annual CT chest: from age 35 (or 10yr before earliest family mesothelioma diagnosis); "
                "  Annual PET-CT: some experts recommend from age 40; "
                "  Serum mesothelin (SMRP): baseline + annual — mesothelioma biomarker; "
                "  Thymidine kinase (TK1): supplementary marker; "
                "TREATMENT — BAP1-MUTANT MESOTHELIOMA: "
                "  First-line: nivolumab + ipilimumab (CheckMate 743 FDA 2021): "
                "    Epithelioid + non-epithelioid mesothelioma — all BAP1-mutant respond; "
                "  Chemotherapy: pemetrexed + cisplatin ± bevacizumab (second-line or surgery-unsuitable); "
                "  Tazemetostat (Tazverik, EZH2 inhibitor) FDA 2020: "
                "    BAP1 LOF → PRC2/EZH2 hyperactivation (epigenetic vulnerability); "
                "    Tazemetostat active in BAP1-null mesothelioma (phase 2 data); "
                "  PARP inhibitors: BAP1 LOF → partial HRD → trials with olaparib/rucaparib; "
                "  Surgical: pleurectomy/decortication (P/D) or EPP: selected fit patients with epithelioid; "
                "BAP1 IHC IN PLEURAL/PERITONEAL BIOPSY: "
                "  Nuclear BAP1 loss on IHC: biallelic somatic inactivation — confirms mesothelioma; "
                "  Normal nuclear staining: does NOT exclude BAP1-mutant — one allele may retain function; "
                "  Combined: germline BAP1 (blood) + IHC + somatic sequencing (tumour) for full picture"
            ),
        },
        {
            "term": "ATM-Radiosensitivity-PARP-Inhibitor-Lung-Cancer-Protocol",
            "definition": (
                "ATM Biallelic vs Monoallelic — Radiosensitivity and PARP Inhibitor Protocol for Lung Cancer: "
                "BIALLELIC ATM (ATAXIA-TELANGIECTASIA) — ABSOLUTE RADIOSENSITIVITY: "
                "  Mechanism: complete ATM kinase loss → no γH2AX signalling → no G1/S or G2/M arrest; "
                "    DSBs accumulate unrepaired → clonogenic cell death at 2-5× lower doses; "
                "    3-5× increased radiation sensitivity vs wild-type; "
                "  Clinical implication: "
                "    Standard-dose EBRT (e.g. 60Gy for NSCLC): POTENTIALLY LETHAL in biallelic A-T; "
                "    Even diagnostic CT: cumulative dose — minimise imaging; "
                "    AVOID: thoracic RT, total-body radiation, nuclear medicine therapy (131I, 177Lu); "
                "  Alternatives in A-T lung Ca: "
                "    Systemic chemotherapy: carboplatin/pemetrexed first-line; "
                "    Immunotherapy: pembrolizumab/atezolizumab (non-radiation modality); "
                "    PARP inhibitors: biallelic ATM → complete HRD → PARPi synthetic lethality; "
                "MONOALLELIC ATM (CARRIER) — INTERMEDIATE RADIOSENSITIVITY: "
                "  Standard RT doses generally tolerable (haploinsufficiency ≠ complete ATM loss); "
                "  Verify with radiosensitivity assay (clonogenic survival) if clinical concern; "
                "  Lung cancer in ATM carriers: earlier onset, often adenocarcinoma; "
                "  Platinum-based doublet: preferred (partial HRD → platinum sensitivity); "
                "  PARP INHIBITOR PROTOCOL FOR ATM-MUTANT NSCLC: "
                "    Olaparib (Lynparza): FDA-approved ATM-mutant prostate cancer — activity in NSCLC ATM emerging; "
                "    PATRIOT trial (ceralasertib/AZD6738 + olaparib): ATM-mutant NSCLC; "
                "      Rationale: ATM LOF → ATR-dependent replication → ATRi → fork collapse + mitotic catastrophe; "
                "      Ceralasertib (AZD6738): ATR inhibitor, Phase 2 clinical activity; "
                "    Adavosertib (AZD1775, WEE1i): ATM-null → G2 checkpoint loss → WEE1i → forced mitosis → death; "
                "TESTING ALGORITHM FOR LUNG CANCER + ATM CONCERN: "
                "  Step 1: cerebellar ataxia + telangiectasia + low AFP → biallelic A-T diagnosis; "
                "  Step 2: germline ATM sequencing (blood); "
                "  Step 3: somatic ATM (tumour); "
                "  Step 4: if germline ATM + lung Ca → classify biallelic (A-T) vs monoallelic; "
                "  Step 5: radiosensitivity assay if uncertain; "
                "  Step 6: plan systemic-only if biallelic; PARP ± ATRi if monoallelic"
            ),
        },
        {
            "term": "FLCN-BHD-Pneumothorax-Pleurodesis-Renal-Surveillance-Protocol",
            "definition": (
                "FLCN/Birt-Hogg-Dubé — Pneumothorax Management and Renal Surveillance Protocol: "
                "WHY BHD PNEUMOTHORAX IS DIFFERENT FROM PRIMARY SPONTANEOUS PNEUMOTHORAX: "
                "  Primary spontaneous pneumothorax (PSP): subpleural bleb rupture; young male; recurrence 30%; "
                "  BHD spontaneous pneumothorax: "
                "    Multiple thin-walled cysts throughout lung (not isolated blebs); "
                "    Recurrence rate: 75-80% after first episode (vs 30% PSP); "
                "    Bilateral involvement: high (40-50%); "
                "    Trigger: any activity causing Valsalva or altitude change; "
                "  Implications: "
                "    First spontaneous pneumothorax in BHD → pleurodesis (do NOT just observe); "
                "    Management ≠ conservative aspiration alone; "
                "    VATS pleurodesis: preferred — pleurodesis + cyst resection if accessible; "
                "    Chemical pleurodesis: talc / doxycycline — used if surgical risk high; "
                "ACTIVITIES TO AVOID IN BHD: "
                "  SCUBA diving: absolute contraindication (pulmonary barotrauma); "
                "  High-altitude environments: >3000m without pressurisation (commercial aircraft = safe); "
                "  Rapid Valsalva activities: competitive breath-holding, forceful weightlifting; "
                "  Flying after pneumothorax: wait 6 weeks after resolution before commercial flight; "
                "PULMONARY CYST MANAGEMENT: "
                "  Baseline CT chest at diagnosis: document cyst burden (number, distribution, size); "
                "  Repeat CT: only if symptomatic (pneumothorax, new dyspnoea, haemoptysis); "
                "  No treatment for stable cysts (cannot prevent pneumothorax); "
                "RENAL SURVEILLANCE PROTOCOL: "
                "  Annual abdominal MRI (preferred) or CT from age 20: "
                "    WHY MRI: no radiation + BHD patients may need decades of surveillance; "
                "    Detect: hybrid oncocytic/chromophobe RCC, chromophobe RCC, oncocytoma; "
                "  3cm threshold: lesions <3cm without aggressive features — active surveillance; "
                "  Surgical: >3cm or solid enhancement or growth on serial imaging; "
                "  Partial nephrectomy preferred (nephron-sparing): bilateral + multifocal in BHD; "
                "  mTOR inhibition (everolimus): FLCN-null → mTORC1 → everolimus active in BHD RCC; "
                "BHD SCREENING IN FAMILY MEMBERS: "
                "  All first-degree relatives: FLCN germline testing; "
                "  Confirmed carriers: CT chest baseline (cyst burden) + abdominal MRI; "
                "  Test for spontaneous pneumothorax recurrence: recurrent PSP in young adults → FLCN testing"
            ),
        },
    ]
    return {
        "atlas":       "Hereditary-Lung-Cancer-Atlas",
        "seed_range":  f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
    print(json.dumps(generate_breakdown(), indent=2))
    print(json.dumps(generate_definitions(), indent=2))
