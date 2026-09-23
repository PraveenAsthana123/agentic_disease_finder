#!/usr/bin/env python3
"""Hereditary-Bladder-Urothelial-Cancer-Predisposition-Atlas -- Complete 8-Gene Reference
MSH2   (MutS homolog 2; 934aa; 2p21; AD LOF;
         Lynch type 2 — urothelial 10-14% HIGHEST Lynch gene;
         Muir-Torre sebaceous neoplasms PATHOGNOMONIC;
         EPCAM 3' deletions MLPA MANDATORY;
         annual cystoscopy + urine cytology MANDATORY;
         seed SEED_BASE+0) .
MLH1   (MutL homolog 1; 756aa; 3p22.2; AD LOF;
         Lynch type 1 — urothelial 2-8% MSI-H;
         pembrolizumab FDA2017 MSI-H/MMR-d urothelial;
         annual OGD + colonoscopy from 25-35yr MANDATORY;
         seed SEED_BASE+1) .
MSH6   (MutS homolog 6; 1360aa; 2p16.3; AD LOF;
         Lynch type 3 — endometrial 71% HIGHEST single-MMR-gene;
         urothelial 5-7%; 4 pseudogenes chromosome 2;
         pembrolizumab FDA2017 MSI-H;
         seed SEED_BASE+2) .
BRCA1  (BRCA1 BRCT domain; 1863aa; 17q21.31; AD LOF;
         HBOC-1 — urothelial 1.5-2x elevated risk;
         HRD cisplatin-sensitive (BRCA1 HR-deficient);
         olaparib FDA2020 BRCA1 HRD urothelial;
         seed SEED_BASE+3) .
BRCA2  (HR scaffold/FANCD1; 3418aa; 13q12.3; AD LOF;
         HBOC-2 — urothelial 2-3x elevated risk;
         cisplatin/carboplatin HRD-sensitive urothelial BRCA2;
         olaparib FDA2020 HRD-positive urothelial BRCA2;
         seed SEED_BASE+4) .
RB1    (pRb E2F regulator; 928aa; 13q14.2; AD LOF;
         bilateral retinoblastoma — transitional cell carcinoma 5x post-RT;
         AVOID RADIATION RB1 germline (secondary TCC at RT field);
         annual cystoscopy bilateral RB survivors MANDATORY;
         seed SEED_BASE+5) .
TP53   (Tumour protein p53; 393aa; 17p13.1; AD LOF;
         LFS — urothelial elevated 2-3x;
         AVOID RADIATION ABSOLUTELY;
         WB-MRI Toronto MANDATORY;
         seed SEED_BASE+6) .
PTEN   (Dual phosphatase; 403aa; 10q23.31; AD LOF;
         Cowden/PHTS — urothelial 5-8% elevated;
         bladder surveillance Cowden protocol;
         PI3K/AKT/mTOR pathway — everolimus/alpelisib emerging;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3446-3453)
"""
import random

SEED_BASE = 3446

ATLAS_GENES = [
    {
        "gene": "MSH2",
        "protein": (
            "MSH2 -- 2p21 Autosomal-Dominant-LOF -- 934aa -- "
            "MutS-Homolog-2-100kDa-MMR-Mismatch-Recognition-"
            "Lynch-Type-2-Muir-Torre-"
            "Urothelial-10-14pct-HIGHEST-Lynch-Gene-"
            "EPCAM-3prime-Deletions-MLPA-MANDATORY-"
            "Cystoscopy-Urine-Cytology-Annual-MANDATORY-"
            "OMIM-609309"
        ),
        "locus": "2p21",
        "syndrome": "Lynch syndrome type 2 / Muir-Torre syndrome — sebaceous neoplasms PATHOGNOMONIC",
        "inheritance": "AD LOF",
        "urothelial_risk": "10-14% lifetime — HIGHEST urothelial cancer risk of all Lynch genes",
        "pathognomonic": "Muir-Torre sebaceous neoplasms (sebaceoma/sebaceous carcinoma/keratoacanthoma) PATHOGNOMONIC / MSH2/MSH6 IHC loss",
        "key_avoid": "DO NOT omit EPCAM MLPA — 30% MSH2 Lynch families have EPCAM 3' deletion NOT detected by sequencing",
        "key_rule": "ANNUAL CYSTOSCOPY + URINE CYTOLOGY MANDATORY MSH2 Lynch — urothelial 10-14% HIGHEST; EPCAM MLPA mandatory MSH2-negative Lynch",
        "surveillance": "Annual cystoscopy + urine cytology from 30-35yr / annual colonoscopy 25yr+ / annual OGD 30-35yr / EPCAM MLPA",
        "targeted_rx": "Pembrolizumab FDA2017 MSI-H/MMR-d urothelial (EV+pembro KEYNOTE-869); erdafitinib FGFR3+ urothelial; cisplatin-based MVAC/GC",
    },
    {
        "gene": "MLH1",
        "protein": (
            "MLH1 -- 3p22.2 Autosomal-Dominant-LOF -- 756aa -- "
            "MutL-Homolog-1-85kDa-MMR-Scaffold-"
            "Lynch-Type-1-CMMRD-Biallelic-"
            "Urothelial-2-8pct-MSI-H-PATHOGNOMONIC-"
            "Pembrolizumab-FDA2017-MSI-H-"
            "Constitutional-Methylation-NOT-Inherited-15pct-"
            "OMIM-120436"
        ),
        "locus": "3p22.2",
        "syndrome": "Lynch syndrome type 1 / CMMRD biallelic — pan-Lynch syndrome",
        "inheritance": "AD LOF",
        "urothelial_risk": "2-8% lifetime; MSI-H urothelial; IHC MLH1/PMS2 loss PATHOGNOMONIC Lynch",
        "pathognomonic": "MLH1/PMS2 IHC nuclear loss PATHOGNOMONIC Lynch urothelial / CMMRD biallelic childhood multi-cancer",
        "key_avoid": "DO NOT assume MLH1-negative Lynch is sporadic methylation — constitutional methylation test MANDATORY (~15% NOT inherited)",
        "key_rule": "PEMBROLIZUMAB FDA2017 MSI-H/MMR-d urothelial APPROVED — IHC loss MLH1/PMS2 triggers pembrolizumab + Lynch germline test",
        "surveillance": "Annual cystoscopy + cytology from 30-35yr / annual colonoscopy 25yr+ / annual OGD / aspirin 600mg CAPP2",
        "targeted_rx": "Pembrolizumab FDA2017 MSI-H urothelial; atezolizumab PD-L1+ urothelial; erdafitinib FGFR3+; cisplatin GC/MVAC",
    },
    {
        "gene": "MSH6",
        "protein": (
            "MSH6 -- 2p16.3 Autosomal-Dominant-LOF -- 1360aa -- "
            "MutS-Homolog-6-160kDa-MMR-IDL-bp-Mismatch-"
            "Lynch-Type-3-"
            "Endometrial-71pct-ABSOLUTE-HIGHEST-Single-MMR-Gene-"
            "Urothelial-5-7pct-"
            "4-Pseudogenes-Chromosome-2-Sequencing-Pitfalls-"
            "OMIM-600678"
        ),
        "locus": "2p16.3",
        "syndrome": "Lynch syndrome type 3 — endometrial dominant / urothelial secondary",
        "inheritance": "AD LOF",
        "urothelial_risk": "5-7% lifetime MSH6 Lynch — secondary after endometrial dominant risk",
        "pathognomonic": "MSH6 IHC nuclear loss (MSH2/MSH6 pair) PATHOGNOMONIC Lynch / endometrial 71% HIGHEST MSH6 single-gene",
        "key_avoid": "DO NOT miss MSH6 pseudogenes (4 pseudogenes chromosome 2) — standard sequencing MSH6 pitfall; MLPA + sequencing required",
        "key_rule": "MSH6 Lynch: endometrial 71% HIGHEST PRIORITY surveillance + urothelial 5-7% secondary; annual cystoscopy from 30-35yr",
        "surveillance": "Annual endometrial sampling + US women MSH6 from 30-35yr / annual cystoscopy + cytology / colonoscopy 25yr+",
        "targeted_rx": "Pembrolizumab FDA2017 MSI-H urothelial/endometrial; lenvatinib+pembrolizumab endometrial Lynch (KEYNOTE-775); erdafitinib FGFR3+",
    },
    {
        "gene": "BRCA1",
        "protein": (
            "BRCA1 -- 17q21.31 Autosomal-Dominant-LOF -- 1863aa -- "
            "BRCA1-210kDa-BRCT-Domains-HR-Nuclear-Scaffold-"
            "HBOC-1-Hereditary-Breast-Ovarian-Cancer-"
            "Urothelial-1.5-2x-Elevated-Risk-"
            "HRD-Cisplatin-Sensitive-"
            "Olaparib-FDA2020-BRCA1-Urothelial-HRD-"
            "OMIM-113705"
        ),
        "locus": "17q21.31",
        "syndrome": "HBOC-1 (Hereditary Breast-Ovarian Cancer syndrome type 1) — urothelial secondary",
        "inheritance": "AD LOF",
        "urothelial_risk": "1.5-2x RR BRCA1 monoallelic — modest elevated risk; HRD-positive urothelial (cisplatin/PARP sensitive)",
        "pathognomonic": "HRD-positive urothelial + BRCA1 germline / early-onset bilateral breast cancer / TNBC PATHOGNOMONIC BRCA1",
        "key_avoid": "DO NOT use carboplatin over cisplatin BRCA1 HRD urothelial — cisplatin SUPERIOR in HRD (stronger DNA crosslink → greater HRD advantage)",
        "key_rule": "CISPLATIN PREFERRED OVER CARBOPLATIN in BRCA1 HRD urothelial — olaparib FDA2020 HRD maintenance post-platinum",
        "surveillance": "Annual breast MRI from 25yr BRCA1 / risk-reducing BSO 35-40yr / annual pelvic US/CA-125 / cystoscopy only if urothelial symptoms or family history",
        "targeted_rx": "Olaparib FDA2020 HRD-positive urothelial (PARP inhibitor maintenance); cisplatin GC first-line BRCA1; pembrolizumab PD-L1+ urothelial",
    },
    {
        "gene": "BRCA2",
        "protein": (
            "BRCA2 -- 13q12.3 Autosomal-Dominant-LOF -- 3418aa -- "
            "BRCA2-384kDa-HR-Scaffold-RAD51-Loader-FANCD1-"
            "HBOC-2-"
            "Urothelial-2-3x-Elevated-Risk-"
            "Cisplatin-HRD-Sensitive-"
            "Olaparib-FDA2020-HRD-Positive-Urothelial-"
            "OMIM-600185"
        ),
        "locus": "13q12.3",
        "syndrome": "HBOC-2 (Hereditary Breast-Ovarian Cancer syndrome type 2) / FA-D1 biallelic",
        "inheritance": "AD LOF (monoallelic) / AR biallelic (FA-D1)",
        "urothelial_risk": "2-3x RR BRCA2 monoallelic — higher than BRCA1 urothelial risk elevation",
        "pathognomonic": "HRD-positive urothelial + BRCA2 germline / BRCA2 HRD highest cisplatin/PARP benefit",
        "key_avoid": "DO NOT IGNORE BRCA2 in urothelial — 2-3x RR higher than BRCA1; always biomarker HRD test BRCA2 urothelial",
        "key_rule": "CISPLATIN + OLAPARIB FDA2020 BRCA2 HRD-positive urothelial — platinum induction then olaparib maintenance (PARP inhibitor licensed HRD urothelial)",
        "surveillance": "Annual breast MRI women BRCA2 25yr+ / BSO 35-40yr women / annual pancreatic MRI/EUS 50yr+ / cystoscopy from 50yr BRCA2 if strong urothelial family history",
        "targeted_rx": "Olaparib FDA2020 HRD-positive urothelial BRCA2; cisplatin GC/MVAC; pembrolizumab EV+pembro first-line; erdafitinib FGFR3+ BRCA2",
    },
    {
        "gene": "RB1",
        "protein": (
            "RB1 -- 13q14.2 Autosomal-Dominant-LOF -- 928aa -- "
            "pRb-105kDa-E2F-G1-S-Checkpoint-Regulator-"
            "Bilateral-Retinoblastoma-PATHOGNOMONIC-"
            "Transitional-Cell-Carcinoma-5x-Post-RT-"
            "AVOID-RADIATION-RB1-Germline-"
            "Annual-Cystoscopy-Bilateral-RB-Survivors-MANDATORY-"
            "OMIM-614041"
        ),
        "locus": "13q14.2",
        "syndrome": "Hereditary retinoblastoma predisposition — secondary malignancy spectrum",
        "inheritance": "AD LOF",
        "urothelial_risk": "5x elevated transitional cell carcinoma post-radiation therapy (secondary TCC at RT field)",
        "pathognomonic": "Bilateral retinoblastoma PATHOGNOMONIC germline RB1 / secondary TCC at prior RT field",
        "key_avoid": "AVOID RADIATION RB1 germline (external beam orbit/pelvis) — secondary TCC 5x + sarcoma 40x at radiation field",
        "key_rule": "AVOID RADIATION RB1 GERMLINE — annual cystoscopy bilateral retinoblastoma survivors MANDATORY (secondary TCC surveillance); CDK4-6i INACTIVE RB1-null tumours",
        "surveillance": "Annual cystoscopy + urine cytology bilateral RB survivors (lifelong) / annual MRI prior radiation fields / annual ophthalmology",
        "targeted_rx": "Cisplatin GC/MVAC urothelial RB1 (RB1-null = CDK4-6i resistant; avoid palbociclib); pembrolizumab MSI-H urothelial",
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 Autosomal-Dominant-LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-Guardian-of-Genome-"
            "LFS-Li-Fraumeni-Syndrome-"
            "Urothelial-2-3x-Elevated-"
            "AVOID-RADIATION-ABSOLUTELY-"
            "WB-MRI-Toronto-Protocol-MANDATORY-"
            "OMIM-151623"
        ),
        "locus": "17p13.1",
        "syndrome": "Li-Fraumeni Syndrome (LFS) — pan-tumour predisposition including urothelial",
        "inheritance": "AD LOF",
        "urothelial_risk": "2-3x elevated LFS — urothelial secondary to dominant sarcoma/breast/brain/ACC spectrum",
        "pathognomonic": "Multi-cancer LFS family / early-onset urothelial transitional cell + sarcoma + breast + brain cluster",
        "key_avoid": "AVOID RADIATION ABSOLUTELY — secondary urothelial TCC + sarcoma at radiation field catastrophic LFS",
        "key_rule": "AVOID RADIATION ABSOLUTELY — WB-MRI Toronto annual MANDATORY; cystoscopy preferred over CT urogram (radiation minimisation LFS)",
        "surveillance": "Annual WB-MRI Toronto protocol / cystoscopy if urothelial symptoms (avoid CT urogram — ionising radiation) / annual brain MRI / annual breast MRI women",
        "targeted_rx": "Cisplatin GC urothelial LFS (no specific TP53-targeted therapy approved); pembrolizumab MSI-H urothelial LFS; WB-MRI for staging vs CT when feasible",
    },
    {
        "gene": "PTEN",
        "protein": (
            "PTEN -- 10q23.31 Autosomal-Dominant-LOF -- 403aa -- "
            "PTEN-47kDa-Dual-Phosphatase-PI3K-AKT-mTOR-"
            "Cowden-PHTS-Lhermitte-Duclos-PATHOGNOMONIC-"
            "Macrocephaly-PATHOGNOMONIC-"
            "Urothelial-5-8pct-Elevated-"
            "Bladder-Surveillance-Cowden-Protocol-"
            "OMIM-601728"
        ),
        "locus": "10q23.31",
        "syndrome": "Cowden syndrome / PTEN Hamartoma Tumour Syndrome (PHTS)",
        "inheritance": "AD LOF",
        "urothelial_risk": "5-8x elevated urothelial cancer (bladder TCC) — Cowden PHTS bladder component",
        "pathognomonic": "Macrocephaly (>97th centile) PATHOGNOMONIC Cowden / Lhermitte-Duclos cerebellar dysplastic gangliocytoma PATHOGNOMONIC / mucocutaneous hamartomas",
        "key_avoid": "DO NOT miss macrocephaly as Cowden indicator — macrocephaly alone indications for PTEN testing; avoid over-reliance on thyroid alone (multi-organ Cowden)",
        "key_rule": "BLADDER SURVEILLANCE Cowden PROTOCOL: annual urine cytology from 30-35yr PTEN/Cowden; macrocephaly + multi-organ hamartomas = PTEN testing; PI3Ki/mTORi therapy PTEN-loss urothelial",
        "surveillance": "Annual cystoscopy + urine cytology Cowden from 30-35yr / annual breast MRI 30yr+ women / annual thyroid US / annual endometrial sampling / annual dermatology",
        "targeted_rx": "Everolimus mTORi PTEN-loss urothelial (PI3K/mTOR hyperactivation); pembrolizumab PD-L1+ urothelial PTEN-null; cisplatin GC; alpelisib PI3Ki emerging",
    },
]

_GENE_LIST = [g["gene"] for g in ATLAS_GENES]


def _make_patients(gene: str, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)

    urothelial_subtypes = {
        "MSH2": [
            ("High-grade urothelial carcinoma — bladder MSI-H Lynch", 0.55),
            ("Upper tract urothelial carcinoma (renal pelvis/ureter) MSH2", 0.30),
            ("Urothelial CIS (carcinoma in situ) MSH2 Lynch", 0.15),
        ],
        "MLH1": [
            ("High-grade urothelial carcinoma — bladder MSI-H Lynch 1", 0.60),
            ("Upper tract urothelial carcinoma MLH1 Lynch", 0.28),
            ("Urothelial carcinoma CIS MMR-deficient", 0.12),
        ],
        "MSH6": [
            ("High-grade urothelial carcinoma — bladder MSH6 Lynch 3", 0.58),
            ("Upper tract urothelial carcinoma MSH6", 0.25),
            ("Endometrial carcinoma + urothelial (MSH6 dual)", 0.17),
        ],
        "BRCA1": [
            ("High-grade urothelial carcinoma HRD BRCA1", 0.55),
            ("Muscle-invasive bladder cancer HRD BRCA1 (cisplatin-sensitive)", 0.35),
            ("TNBC + urothelial (BRCA1 dual primary)", 0.10),
        ],
        "BRCA2": [
            ("High-grade urothelial carcinoma HRD BRCA2", 0.52),
            ("Muscle-invasive bladder cancer HRD BRCA2", 0.38),
            ("Upper tract urothelial HRD BRCA2", 0.10),
        ],
        "RB1": [
            ("Secondary transitional cell carcinoma post-RT (bladder)", 0.45),
            ("Primary muscle-invasive bladder cancer RB1-null", 0.38),
            ("Upper tract urothelial RB1-null post-RT", 0.17),
        ],
        "TP53": [
            ("High-grade urothelial carcinoma LFS TP53-mutant", 0.50),
            ("Muscle-invasive bladder cancer TP53 germline", 0.35),
            ("Secondary urothelial post-LFS radiation field", 0.15),
        ],
        "PTEN": [
            ("Urothelial carcinoma Cowden PTEN-null bladder", 0.52),
            ("Muscle-invasive bladder cancer PTEN-null mTOR-active", 0.35),
            ("High-grade urothelial CIS Cowden PHTS", 0.13),
        ],
    }

    variants_by_gene = {
        "MSH2": [
            "p.Arg524Pro (R524P — clamp domain missense MSH2)",
            "p.Lys675Asn (K675N — MutS homology 3 MSH2)",
            "c.IVS5+3A>T (splice donor exon 5 MSH2)",
            "del exon 1-6 (EPCAM 3' deletion → MSH2 silencing MLPA)",
            "p.Arg359Ter (R359X — truncation clamp MSH2)",
            "p.Tyr694Cys (Y694C — C-terminal helix MSH2)",
        ],
        "MLH1": [
            "p.Arg217Ter (R217X — ATPase domain truncation MLH1)",
            "p.Val384Asp (V384D — common missense MMR-deficient MLH1)",
            "c.IVS12-2A>G (splice acceptor intron 12 MLH1)",
            "del exon 16 (large deletion MLH1 Lynch)",
            "p.Gly244Val (G244V — interaction domain MLH1)",
            "MLH1 promoter methylation constitutional (not inherited)",
        ],
        "MSH6": [
            "p.Thr1219Ile (T1219I — MutS homology 5 MSH6)",
            "p.Arg1087Pro (R1087P — MutS homology 4 MSH6 pseudogene-region)",
            "c.IVS9+1G>A (splice donor exon 9 MSH6)",
            "del exon 4-5 (MLPA large deletion MSH6)",
            "p.Phe1088Ser (F1088S — ATPase C-terminal MSH6)",
            "p.Ser144Ter (early truncation MSH6 N-terminal)",
        ],
        "BRCA1": [
            "p.Glu1694Ter (E1694X — BRCT domain truncation BRCA1)",
            "c.5266dupC (5382insC — Ashkenazi Jewish founder BRCA1)",
            "c.IVS20+1G>A (splice donor BRCA1 exon 20)",
            "del exon 2 (MLPA large deletion BRCA1)",
            "p.Gln1756Pro (Q1756P — BRCT structural missense BRCA1)",
            "p.Ser1655Ter (S1655X — BRCT truncation BRCA1)",
        ],
        "BRCA2": [
            "p.Lys3326Ter (K3326X — C-terminal OB truncation BRCA2)",
            "p.Glu1308Ter (BRC repeat 5 truncation BRCA2)",
            "c.IVS11+1G>A (splice donor intron 11 BRCA2 HBOC)",
            "p.Trp2626Ter (OB-fold truncation BRCA2)",
            "del exons 15-26 (large deletion BRCA2 HBOC)",
            "p.Arg2336His (OB fold structural BRCA2 urothelial)",
        ],
        "RB1": [
            "p.Arg455Ter (R455X — ISCN pocket domain truncation RB1)",
            "p.Glu137Ter (early truncation pocket domain A RB1)",
            "c.IVS7+1G>A (splice donor exon 7 RB1)",
            "del exon 13-14 (large deletion RB1 MLPA required)",
            "p.Trp563Ter (pocket domain B truncation RB1)",
            "p.Leu690Pro (pocket domain B hydrophobic structural RB1)",
        ],
        "TP53": [
            "p.Arg248Trp (R248W — DBD hotspot dominant-negative LFS)",
            "p.Arg175His (R175H — DBD structural hotspot LFS)",
            "p.Arg337His (R337H — tetramerisation Brazilian founder LFS)",
            "c.IVS6+1G>A (splice donor intron 6 TP53)",
            "p.Arg273His (R273H — DNA-contact DBD TP53)",
            "del exon 5-6 (germline deletion TP53 LFS)",
        ],
        "PTEN": [
            "p.Arg130Gln (R130Q — phosphatase active site PTEN Cowden)",
            "p.Cys124Ser (C124S — catalytic Cys PTEN phosphatase inactive)",
            "p.Glu242Ter (E242X — C2 domain truncation PTEN)",
            "del exon 5 (MLPA deletion PTEN Cowden)",
            "p.His123Asp (H123D — phosphatase WPD loop PTEN)",
            "c.IVS8+1G>A (splice donor exon 8 PTEN Cowden)",
        ],
    }

    age_ranges = {
        "MSH2":  (30, 75),
        "MLH1":  (32, 75),
        "MSH6":  (35, 75),
        "BRCA1": (40, 75),
        "BRCA2": (42, 78),
        "RB1":   (25, 70),
        "TP53":  (25, 68),
        "PTEN":  (35, 72),
    }

    subtypes = urothelial_subtypes[gene]
    variants = variants_by_gene[gene]
    age_lo, age_hi = age_ranges[gene]

    patients = []
    for i in range(n):
        age = rng.randint(age_lo, age_hi)
        r = rng.random()
        cumulative = 0.0
        tumour = subtypes[-1][0]
        for t, p in subtypes:
            cumulative += p
            if r < cumulative:
                tumour = t
                break
        variant = rng.choice(variants)
        sex = rng.choice(["M", "F"])
        stage = rng.choices(["Ta/T1", "T2", "T3", "T4/M"], weights=[0.28, 0.30, 0.25, 0.17])[0]
        msi_h = gene in ("MSH2", "MLH1", "MSH6") and rng.random() < 0.88
        hrd_positive = gene in ("BRCA1", "BRCA2") and rng.random() < 0.68
        upper_tract = gene in ("MSH2", "MLH1") and rng.random() < 0.32
        radiation_ci = gene in ("RB1", "TP53") and rng.random() < 0.75
        cdk4_6i_inactive = (gene == "RB1") and rng.random() < 0.85
        pembrolizumab_eligible = msi_h and rng.random() < 0.72
        olaparib_eligible = hrd_positive and rng.random() < 0.48
        epcam_deletion = (gene == "MSH2") and rng.random() < 0.30
        macrocephaly = (gene == "PTEN") and rng.random() < 0.65
        cystoscopy_surveillance = rng.random() < 0.60
        relapse = (stage in ("T3", "T4/M")) and rng.random() < 0.55

        patients.append({
            "patient_id": f"{gene[:5].upper()}-{seed:04d}-{i+1:02d}",
            "gene": gene,
            "age_at_dx": age,
            "sex": sex,
            "tumour_type": tumour,
            "stage": stage,
            "variant": variant,
            "msi_h": msi_h,
            "hrd_positive": hrd_positive,
            "upper_tract": upper_tract,
            "radiation_contraindicated": radiation_ci,
            "cdk4_6i_inactive": cdk4_6i_inactive,
            "pembrolizumab_eligible": pembrolizumab_eligible,
            "olaparib_eligible": olaparib_eligible,
            "epcam_deletion": epcam_deletion,
            "macrocephaly": macrocephaly,
            "cystoscopy_surveillance": cystoscopy_surveillance,
            "relapse": relapse,
        })
    return patients


TREATMENT_PROTOCOLS_BY_GENE = {
    "MSH2": [
        "Pembrolizumab FDA2017 MSI-H/MMR-d urothelial APPROVED (first or any line)",
        "Enfortumab vedotin (EV) + pembrolizumab KEYNOTE-869 MSH2 Lynch urothelial first-line advanced",
        "Cisplatin GC (gemcitabine+cisplatin) fit patients MSH2 urothelial first-line if MSI-H not first tested",
        "Erdafitinib FGFR3+ MSH2 urothelial (FGFR3 mutation overlap possible)",
        "BCG intravesical non-muscle-invasive MSH2 Lynch (standard NMIBC protocol)",
        "Cystectomy muscle-invasive MSH2 urothelial (radical cystectomy curative intent)",
        "Annual cystoscopy + urine cytology MSH2 Lynch MANDATORY lifelong",
        "EPCAM MLPA MSH2-negative Lynch families MANDATORY",
    ],
    "MLH1": [
        "Pembrolizumab FDA2017 MSI-H urothelial FIRST-LINE (MLH1 Lynch = MSI-H = pembrolizumab eligible)",
        "Atezolizumab PD-L1+ advanced urothelial MLH1 Lynch (second checkpoint option)",
        "Cisplatin GC or MVAC muscle-invasive MLH1 Lynch urothelial",
        "BCG NMIBC standard protocol MLH1",
        "Annual cystoscopy + cytology MLH1 Lynch from 30-35yr MANDATORY",
        "Annual colonoscopy from 25yr Lynch MLH1 (CRC dominant cancer)",
        "Aspirin 600mg/day CAPP2 Lynch MLH1 protocol",
        "H. pylori eradication mandatory Lynch MLH1 carriers",
    ],
    "MSH6": [
        "Pembrolizumab FDA2017 MSI-H urothelial MSH6 Lynch APPROVED",
        "Lenvatinib+pembrolizumab MSH6 Lynch endometrial cancer (KEYNOTE-775 — endometrial dominant)",
        "Cisplatin GC muscle-invasive urothelial MSH6 Lynch",
        "Annual cystoscopy + cytology MSH6 Lynch from 30-35yr",
        "Annual endometrial sampling + US women MSH6 from 30-35yr (endometrial 71% dominant)",
        "MSH6 pseudogene panel sequencing MLPA mandatory (4 pseudogenes chromosome 2)",
        "Colonoscopy from 25yr MSH6 Lynch (CRC 25-40% lower penetrance than MLH1/MSH2)",
        "Aspirin 600mg/day CAPP2 Lynch MSH6 protocol",
    ],
    "BRCA1": [
        "Olaparib FDA2020 HRD-positive urothelial BRCA1 (PARP inhibitor maintenance post-platinum)",
        "Cisplatin GC first-line BRCA1 urothelial (CISPLATIN preferred over CARBOPLATIN — HRD advantage stronger)",
        "Enfortumab vedotin (EV) + pembrolizumab BRCA1 advanced urothelial (standard first-line advanced)",
        "BCG NMIBC standard protocol BRCA1",
        "Risk-reducing BSO 35-40yr women BRCA1 (ovarian 39-44% dominant management)",
        "Annual breast MRI women BRCA1 from 25yr",
        "HRD biomarker testing mandatory all BRCA1 urothelial (Myriad myChoice or similar)",
        "Cascade BRCA1 germline first-degree relatives MANDATORY",
    ],
    "BRCA2": [
        "Olaparib FDA2020 HRD-positive urothelial BRCA2 (maintenance PARP inhibitor post-platinum)",
        "Cisplatin GC/MVAC first-line BRCA2 urothelial HRD-positive (platinum-sensitive)",
        "Enfortumab vedotin (EV) + pembrolizumab BRCA2 advanced urothelial",
        "Sacituzumab govitecan BRCA2 platinum-refractory urothelial",
        "Annual pancreatic MRI/EUS from 50yr BRCA2 (5-7% pancreatic cancer separate risk)",
        "Annual breast MRI women BRCA2 from 25yr",
        "BSO 35-40yr women BRCA2 (ovarian 11-17%)",
        "Cascade BRCA2 germline first-degree relatives MANDATORY",
    ],
    "RB1": [
        "AVOID RADIATION RB1 germline (secondary TCC 5x + sarcoma 40x at RT field)",
        "Cisplatin GC/MVAC RB1-null urothelial (CDK4-6i INACTIVE — palbociclib/ribociclib useless RB1-null)",
        "Enfortumab vedotin (EV) + pembrolizumab advanced RB1 urothelial",
        "Radical cystectomy muscle-invasive RB1 urothelial curative intent (avoid RT-based trimodality)",
        "Annual cystoscopy + urine cytology bilateral retinoblastoma survivors LIFELONG MANDATORY",
        "Annual MRI prior radiation fields RB1 survivors (secondary sarcoma surveillance concurrent)",
        "AVOID CDK4-6 inhibitors RB1-null tumours (palbociclib/ribociclib/abemaciclib INACTIVE)",
        "Cascade RB1 germline first-degree relatives + newborn ophthalmology",
    ],
    "TP53": [
        "AVOID RADIATION ABSOLUTELY — secondary TCC + sarcoma at RT field LFS catastrophic",
        "Radical cystectomy TP53/LFS urothelial (trimodality with RT CONTRAINDICATED LFS)",
        "Cisplatin GC/MVAC TP53/LFS urothelial (no TP53-targeted therapy approved)",
        "Pembrolizumab PD-L1+ advanced urothelial TP53 germline (MSI-H overlap possible LFS)",
        "Annual WB-MRI Toronto protocol LFS MANDATORY (all organ surveillance)",
        "Cystoscopy preferred over CT urogram for surveillance (radiation minimisation LFS)",
        "Enfortumab vedotin advanced TP53 urothelial (standard-of-care regardless LFS)",
        "De novo TP53 ~25%: test both parents; pre-surgical germline TP53 young-onset TCC",
    ],
    "PTEN": [
        "Everolimus mTORi PTEN-null urothelial (mTOR hyperactivation PTEN-loss pathway)",
        "Alpelisib PI3Ki PTEN-null urothelial (emerging PI3K inhibitor data Cowden urothelial)",
        "Enfortumab vedotin (EV) + pembrolizumab advanced PTEN-null urothelial (standard first-line)",
        "Cisplatin GC PTEN-null muscle-invasive urothelial",
        "Annual cystoscopy + urine cytology Cowden/PHTS from 30-35yr MANDATORY",
        "Annual breast MRI women PTEN Cowden from 30yr (breast 50-85%)",
        "Annual thyroid US PTEN Cowden (papillary thyroid 35%)",
        "Annual endometrial sampling women PTEN Cowden from 30-35yr (endometrial 28-44%)",
    ],
}

SURVEILLANCE_BY_GENE = {
    "MSH2": [
        "Annual cystoscopy + urine cytology MSH2 Lynch from 30-35yr MANDATORY",
        "Annual colonoscopy from 25yr Lynch MSH2 (CRC 40-75% dominant cancer)",
        "Annual OGD from 30-35yr Lynch MSH2",
        "EPCAM MLPA testing MSH2-negative Lynch families MANDATORY (30% missed)",
        "Sebaceous neoplasm any site → Lynch germline test MANDATORY (Muir-Torre)",
        "Aspirin 600mg/day CAPP2 Lynch protocol",
    ],
    "MLH1": [
        "Annual cystoscopy + cytology MLH1 Lynch from 30-35yr",
        "Annual colonoscopy from 25yr Lynch MLH1",
        "Annual OGD from 30-35yr Lynch MLH1",
        "MLH1 constitutional methylation test before cascade (not inherited ~15%)",
        "Aspirin 600mg CAPP2 Lynch MLH1",
        "H. pylori test-and-treat mandatory Lynch MLH1",
    ],
    "MSH6": [
        "Annual cystoscopy + cytology MSH6 Lynch from 30-35yr",
        "Annual endometrial sampling + US women MSH6 from 30-35yr (endometrial 71% MANDATORY priority)",
        "Annual colonoscopy from 25yr MSH6 Lynch",
        "MSH6 pseudogene panel MLPA mandatory (sequencing alone insufficient)",
        "Aspirin 600mg CAPP2 Lynch MSH6",
    ],
    "BRCA1": [
        "Annual breast MRI from 25yr BRCA1 (breast 72% dominant management)",
        "Risk-reducing BSO 35-40yr BRCA1 women (ovarian 39-44%)",
        "Annual pelvic US/CA-125 pre-BSO BRCA1",
        "HRD biomarker test all BRCA1 urothelial tumours (cisplatin/PARP guidance)",
        "Cystoscopy only if urothelial symptoms or strong family history BRCA1",
    ],
    "BRCA2": [
        "Annual breast MRI from 25yr women BRCA2",
        "BSO 35-40yr women BRCA2 (ovarian 11-17%)",
        "Annual pancreatic MRI/EUS from 50yr BRCA2 (5-7% pancreatic)",
        "HRD biomarker test BRCA2 urothelial tumours (olaparib/cisplatin guidance)",
        "Cystoscopy from 50yr BRCA2 if personal or family urothelial history",
    ],
    "RB1": [
        "Annual cystoscopy + urine cytology bilateral retinoblastoma survivors LIFELONG MANDATORY",
        "Annual MRI prior radiation fields RB1 survivors (secondary sarcoma + TCC surveillance)",
        "Annual ophthalmology bilateral RB survivors",
        "AVOID radiation pelvic/abdominal fields RB1 germline carriers",
        "Newborn ophthalmology relatives RB1 germline",
    ],
    "TP53": [
        "Annual WB-MRI Toronto protocol MANDATORY from LFS diagnosis",
        "Cystoscopy preferred over CT urogram LFS (ionising radiation minimisation)",
        "Annual brain MRI LFS",
        "Annual breast MRI women LFS from 20yr",
        "AVOID CT/urogram-based surveillance — endoscopy + MRI preferred across LFS",
        "De novo TP53 ~25%: test both parents before family cascade",
    ],
    "PTEN": [
        "Annual cystoscopy + urine cytology Cowden/PHTS from 30-35yr",
        "Annual breast MRI women PTEN Cowden from 30yr",
        "Annual thyroid US PTEN from 18yr",
        "Annual endometrial sampling + US women PTEN Cowden from 30-35yr",
        "Annual dermatology PTEN (mucocutaneous hamartomas, trichilemmomas)",
        "Brain MRI if neurological symptoms (Lhermitte-Duclos PATHOGNOMONIC)",
    ],
}


def generate_overview() -> dict:
    from collections import Counter
    cohorts = {}
    for i, g in enumerate(ATLAS_GENES):
        cohorts[g["gene"]] = _make_patients(g["gene"], SEED_BASE + i)
    total = sum(len(pts) for pts in cohorts.values())
    gene_counts = {gene: len(pts) for gene, pts in cohorts.items()}
    msi_h_rate = round(100 * sum(1 for pts in cohorts.values() for p in pts if p["msi_h"]) / total, 1)
    hrd_rate = round(100 * sum(1 for pts in cohorts.values() for p in pts if p["hrd_positive"]) / total, 1)
    upper_tract_rate = round(100 * sum(1 for pts in cohorts.values() for p in pts if p["upper_tract"]) / total, 1)
    radiation_ci_rate = round(100 * sum(1 for pts in cohorts.values() for p in pts if p["radiation_contraindicated"]) / total, 1)
    pembrolizumab_rate = round(100 * sum(1 for pts in cohorts.values() for p in pts if p["pembrolizumab_eligible"]) / total, 1)
    olaparib_rate = round(100 * sum(1 for pts in cohorts.values() for p in pts if p["olaparib_eligible"]) / total, 1)
    mean_age = round(sum(p["age_at_dx"] for pts in cohorts.values() for p in pts) / total, 1)

    return {
        "atlas": "Hereditary-Bladder-Urothelial-Cancer-Predisposition-Atlas",
        "genes": _GENE_LIST,
        "total_patients": total,
        "gene_counts": gene_counts,
        "seed_range": f"{SEED_BASE}-{SEED_BASE+7}",
        "msi_h_rate_pct": msi_h_rate,
        "hrd_positive_rate_pct": hrd_rate,
        "upper_tract_rate_pct": upper_tract_rate,
        "radiation_ci_rate_pct": radiation_ci_rate,
        "pembrolizumab_eligible_rate_pct": pembrolizumab_rate,
        "olaparib_eligible_rate_pct": olaparib_rate,
        "mean_age_at_dx": mean_age,
        "key_facts": [
            "MSH2/Lynch-2: MutS homolog 934aa / 100kDa; 2p21; UROTHELIAL 10-14% HIGHEST Lynch gene (renal pelvis + ureter + bladder); Muir-Torre sebaceous neoplasms PATHOGNOMONIC MSH2; EPCAM 3' deletions MLPA MANDATORY (30% missed by sequencing); annual cystoscopy + cytology from 30-35yr MANDATORY; pembrolizumab FDA2017 MSI-H urothelial",
            "MLH1/Lynch-1: MutL homolog 756aa / 85kDa; 3p22.2; urothelial 2-8% MSI-H; IHC MLH1/PMS2 loss PATHOGNOMONIC Lynch; PEMBROLIZUMAB FDA2017 MSI-H/MMR-d urothelial APPROVED; constitutional methylation ~15% MLH1 NOT inherited — methylation test mandatory before cascade; aspirin 600mg CAPP2",
            "MSH6/Lynch-3: MutS homolog 6 1360aa / 160kDa; 2p16.3; ENDOMETRIAL 71% ABSOLUTE HIGHEST single MMR gene (dominant spectrum); urothelial 5-7% secondary; 4 pseudogenes chromosome 2 sequencing PITFALL — MLPA mandatory MSH6; pembrolizumab FDA2017 MSI-H",
            "BRCA1/HBOC-1: 1863aa / 210kDa; 17q21.31; urothelial 1.5-2x elevated HRD; CISPLATIN PREFERRED OVER CARBOPLATIN BRCA1 HRD (stronger DNA crosslink → HRD advantage greater); olaparib FDA2020 HRD-positive urothelial maintenance; breast 72% dominant management priority",
            "BRCA2/HBOC-2: HR scaffold 3418aa / 384kDa; 13q12.3; urothelial 2-3x elevated HRD (HIGHER than BRCA1 urothelial elevation); olaparib FDA2020 HRD-positive urothelial; cisplatin GC/MVAC platinum-sensitive; pancreatic 5-7% → EUS/MRI 50yr+ MANDATORY separate",
            "RB1: pRb 928aa / 105kDa; 13q14.2; SECONDARY TCC POST-RADIATION 5x — secondary urothelial at pelvic RT field bilateral RB survivors; AVOID RADIATION RB1 GERMLINE (pelvic/abdominal fields); CDK4-6i INACTIVE RB1-null tumours; annual cystoscopy bilateral RB survivors LIFELONG MANDATORY",
            "TP53/LFS: p53 393aa / 43kDa; 17p13.1; urothelial 2-3x elevated LFS; AVOID RADIATION ABSOLUTELY (secondary TCC + sarcoma at RT field); WB-MRI Toronto annual MANDATORY; cystoscopy preferred over CT urogram (radiation minimisation); trimodality RT-based bladder preservation CONTRAINDICATED LFS",
            "PTEN/Cowden: dual phosphatase 403aa / 47kDa; 10q23.31; urothelial 5-8x elevated Cowden PHTS; macrocephaly PATHOGNOMONIC Cowden (>97th centile); Lhermitte-Duclos cerebellar PATHOGNOMONIC; mTOR hyperactivation PTEN-loss → everolimus/alpelisib therapy; annual cystoscopy + cytology Cowden from 30-35yr",
        ],
    }


def generate_breakdown() -> dict:
    from collections import Counter
    breakdown = {}
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
                "urothelial_risk": g["urothelial_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_avoid": g["key_avoid"],
                "key_rule": g["key_rule"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            },
            "n": len(pts),
            "msi_h_pct": round(100 * sum(1 for p in pts if p["msi_h"]) / len(pts), 1),
            "hrd_positive_pct": round(100 * sum(1 for p in pts if p["hrd_positive"]) / len(pts), 1),
            "upper_tract_pct": round(100 * sum(1 for p in pts if p["upper_tract"]) / len(pts), 1),
            "radiation_ci_pct": round(100 * sum(1 for p in pts if p["radiation_contraindicated"]) / len(pts), 1),
            "cdk4_6i_inactive_pct": round(100 * sum(1 for p in pts if p["cdk4_6i_inactive"]) / len(pts), 1),
            "pembrolizumab_eligible_pct": round(100 * sum(1 for p in pts if p["pembrolizumab_eligible"]) / len(pts), 1),
            "olaparib_eligible_pct": round(100 * sum(1 for p in pts if p["olaparib_eligible"]) / len(pts), 1),
            "epcam_deletion_pct": round(100 * sum(1 for p in pts if p["epcam_deletion"]) / len(pts), 1),
            "macrocephaly_pct": round(100 * sum(1 for p in pts if p["macrocephaly"]) / len(pts), 1),
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
        "atlas": "Hereditary-Bladder-Urothelial-Cancer-Predisposition-Atlas",
        "definitions": {
            "msh2_lynch2_urothelial_highest_epcam_mlpa_mandatory": (
                "MSH2: 934aa / 100kDa; 2p21; MutSα (MSH2-MSH6) + MutSβ (MSH2-MSH3) mismatch recognition heterodimers; "
                "MSH2 LOF → MMR deficiency → MSI-H — Lynch syndrome type 2; "
                "UROTHELIAL 10-14% LIFETIME: HIGHEST urothelial cancer risk of all Lynch MMR genes — renal pelvis + ureter + bladder; "
                "MSH2/MSH6 IHC NUCLEAR LOSS PATHOGNOMONIC Lynch urothelial; "
                "MUIR-TORRE SEBACEOUS NEOPLASMS PATHOGNOMONIC MSH2: sebaceoma / sebaceous carcinoma / keratoacanthoma — any sebaceous lesion triggers Lynch germline test; "
                "EPCAM 3' DELETIONS MLPA MANDATORY: EPCAM (17q21.31) 3' truncating deletion → MSH2 promoter methylation via read-through → MSH2 IHC loss undetectable by sequencing; 30% MSH2 Lynch families — MLPA MANDATORY MSH2-negative Lynch families; "
                "ANNUAL CYSTOSCOPY + URINE CYTOLOGY MANDATORY from 30-35yr MSH2 Lynch — upper tract imaging (CT urography) every 1-2yr if family history upper tract; "
                "PEMBROLIZUMAB FDA2017 MSI-H urothelial APPROVED any line."
            ),
            "mlh1_lynch1_msi_h_pembrolizumab_fda2017_urothelial": (
                "MLH1: 756aa / 85kDa; 3p22.2; MutLα (MLH1-PMS2 heterodimer) — MMR endonuclease nicks mismatched DNA strand; "
                "MLH1 LOF → MMR deficiency → MSI-H Lynch type 1; "
                "UROTHELIAL 2-8% LIFETIME: IHC MLH1/PMS2 nuclear loss PATHOGNOMONIC Lynch urothelial; "
                "PEMBROLIZUMAB FDA2017 APPROVED ANY MSI-H/MMR-d urothelial cancer (bladder + upper tract); "
                "MLH1 CONSTITUTIONAL METHYLATION (~15% Lynch MLH1): somatic promoter methylation — NOT inherited; methylation testing mandatory before cascade counselling; "
                "CRC 40-80% Lynch type 1 — dominant cancer; colonoscopy annual from 25yr PRIORITY surveillance; "
                "ASPIRIN 600mg/day CAPP2: 60% CRC reduction Lynch — annual OGD 30-35yr mandatory Lynch MLH1; "
                "H. pylori eradication mandatory Lynch MLH1 (modifiable co-carcinogen all Lynch cancers)."
            ),
            "msh6_lynch3_endometrial_71pct_urothelial_pseudogene_mlpa": (
                "MSH6: 1360aa / 160kDa; 2p16.3; MutSα (MSH2-MSH6) — single-base mismatch + IDL small recognition; "
                "MSH6 LOF → MMR deficiency → MSI-H Lynch type 3 — endometrial dominant spectrum; "
                "ENDOMETRIAL 71% LIFETIME ABSOLUTE HIGHEST SINGLE MMR GENE: MSH6 Lynch = ENDOMETRIAL PRIORITY surveillance; "
                "UROTHELIAL 5-7% LIFETIME: secondary risk after endometrial dominant — annual cystoscopy from 30-35yr; "
                "MSH6/MSH2 IHC NUCLEAR LOSS: MSH2 lost simultaneously (MSH2 requires MSH6 for stability); "
                "4 PSEUDOGENES CHROMOSOME 2 MSH6 SEQUENCING PITFALL: standard sequencing misassigns pseudogene reads to MSH6 — MLPA + comprehensive sequencing MANDATORY MSH6; "
                "PEMBROLIZUMAB FDA2017 MSI-H urothelial/endometrial; lenvatinib+pembrolizumab KEYNOTE-775 endometrial Lynch; "
                "Colonoscopy from 25yr Lynch MSH6 (CRC 25-40% lower penetrance than MLH1/MSH2)."
            ),
            "brca1_hboc1_hrd_cisplatin_preferred_olaparib_urothelial": (
                "BRCA1: 1863aa / 210kDa; 17q21.31; BRCT domain scaffold — HR nuclear coordinator; "
                "BRCA1 LOF → HR deficiency (HRD) → NHEJ error-prone → genome instability; "
                "UROTHELIAL 1.5-2x RR BRCA1 monoallelic — HRD-positive urothelial (cisplatin/PARP sensitive); "
                "CISPLATIN PREFERRED OVER CARBOPLATIN BRCA1 HRD UROTHELIAL: cisplatin forms intrastrand crosslinks → GREATER lethality in HRD cells vs carboplatin; do NOT substitute carboplatin for 'tolerability' in BRCA1 fit patients; "
                "OLAPARIB FDA2020 HRD-POSITIVE UROTHELIAL: PARP inhibitor maintenance post-platinum induction — BRCA1 HRD = PARP sensitive; "
                "BREAST 72% LIFETIME BRCA1 — dominant management; annual breast MRI 25yr+ MANDATORY; "
                "BSO 35-40yr BRCA1 (ovarian 39-44%); "
                "HRD biomarker test (Myriad myChoice or similar) mandatory all BRCA1 urothelial for olaparib/cisplatin guidance."
            ),
            "brca2_hboc2_hrd_olaparib_polo_cisplatin_urothelial": (
                "BRCA2/FANCD1: 3418aa / 384kDa; 13q12.3; HR scaffold RAD51 loader (8 BRC repeats); "
                "BRCA2 LOF → HR deficiency → NHEJ error → genome instability; "
                "UROTHELIAL 2-3x RR BRCA2: HIGHER urothelial elevation than BRCA1 (relative risk) — always biomarker HRD test BRCA2 urothelial; "
                "OLAPARIB FDA2020 HRD-POSITIVE UROTHELIAL BRCA2: PARP inhibitor maintenance licensed HRD urothelial; "
                "CISPLATIN GC/MVAC HRD-SENSITIVE: platinum-sensitive BRCA2 HR-deficient — cisplatin preferred; "
                "PANCREATIC CANCER 5-7% BRCA2: annual pancreatic MRI/EUS from 50yr MANDATORY (separate from urothelial risk); "
                "BREAST 69%, OVARIAN 11-17% BRCA2: dominant management priorities — breast MRI 25yr+; BSO 35-40yr; "
                "FA-D1 biallelic BRCA2: severe Fanconi anemia childhood — separate paediatric management (aplastic anemia/AML)."
            ),
            "rb1_secondary_tcc_post_rt_avoid_radiation_cdk4_6i_inactive": (
                "RB1: 928aa / 105kDa; 13q14.2; pRb E2F G1-S checkpoint regulator; pRb LOF → E2F constitutive → uncontrolled S-phase; "
                "BILATERAL RETINOBLASTOMA PATHOGNOMONIC germline RB1; "
                "SECONDARY TCC POST-RADIATION 5x: transitional cell carcinoma at pelvic/abdominal radiation field — bilateral RB survivors; "
                "AVOID RADIATION RB1 GERMLINE: external beam pelvic/abdominal radiation → secondary TCC 5x + sarcoma 40x at field; "
                "ANNUAL CYSTOSCOPY + URINE CYTOLOGY BILATERAL RB SURVIVORS LIFELONG MANDATORY: secondary TCC detection; "
                "CDK4-6i INACTIVE RB1-NULL TUMOURS: palbociclib/ribociclib/abemaciclib are INACTIVE in RB1-null (pRb absent = CDK4/6 target absent = drug useless + harm risk); "
                "CISPLATIN GC/MVAC RB1 urothelial: avoid RT-based trimodality bladder preservation (RT CONTRAINDICATED germline RB1); "
                "Annual MRI prior radiation fields concurrent (secondary sarcoma 40x at same field — dual surveillance rationale)."
            ),
            "tp53_lfs_avoid_radiation_absolutely_wb_mri_urothelial": (
                "TP53: 393aa / 43kDa; 17p13.1; p53 guardian of the genome — activates CDKN1A/p21, MDM2, PUMA; "
                "TP53 LOF → genome instability → LFS multi-cancer spectrum; "
                "UROTHELIAL 2-3x ELEVATED LFS: secondary to dominant sarcoma/breast/brain/ACC LFS spectrum; "
                "AVOID RADIATION ABSOLUTELY: secondary TCC + sarcoma at radiation field CATASTROPHIC LFS; "
                "TRIMODALITY RT-BASED BLADDER PRESERVATION CONTRAINDICATED LFS: standard-of-care bladder-sparing uses RT — CONTRAINDICATED TP53/LFS; cystectomy surgical approach MANDATORY LFS muscle-invasive; "
                "WB-MRI TORONTO PROTOCOL: annual MANDATORY — no ionising radiation; "
                "CYSTOSCOPY PREFERRED OVER CT UROGRAM LFS: CT urogram requires ionising radiation — endoscopy + MRI preferred for surveillance LFS; "
                "De novo TP53 ~25%: test both parents; pre-surgical germline TP53 all young-onset urothelial cancer."
            ),
            "pten_cowden_phts_macrocephaly_pathognomonic_mtor_urothelial": (
                "PTEN: 403aa / 47kDa; 10q23.31; dual phosphatase (protein/lipid) — dephosphorylates PIP3 → PI3K/AKT/mTOR suppression; "
                "PTEN LOF → PI3K/AKT/mTOR hyperactivation → cell growth/survival → hamartoma/malignancy; "
                "COWDEN SYNDROME / PHTS: macrocephaly (>97th centile) PATHOGNOMONIC; Lhermitte-Duclos cerebellar dysplastic gangliocytoma PATHOGNOMONIC; "
                "UROTHELIAL 5-8x ELEVATED COWDEN PHTS: bladder TCC and upper tract urothelial elevated — annual cystoscopy from 30-35yr MANDATORY; "
                "mTOR HYPERACTIVATION PTEN-NULL: everolimus mTOR inhibitor approved TSC/PTEN pathway; alpelisib PI3Kα inhibitor emerging PTEN-loss data; "
                "MACROCEPHALY ALONE TRIGGERS PTEN TEST: macrocephaly >97th centile without other features = PTEN germline testing; "
                "BREAST 50-85% COWDEN: dominant management — annual breast MRI from 30yr MANDATORY; "
                "ENDOMETRIAL 28-44%, THYROID 35% Cowden: multi-organ surveillance protocol mandatory."
            ),
        },
        "key_clinical_distinctions": [
            "MSH2/Lynch-2: UROTHELIAL 10-14% HIGHEST Lynch gene (renal pelvis+ureter+bladder). Muir-Torre sebaceous neoplasms PATHOGNOMONIC MSH2. EPCAM MLPA MANDATORY (30% missed by sequencing). Annual cystoscopy from 30-35yr. Pembrolizumab FDA2017 MSI-H urothelial APPROVED",
            "MLH1/Lynch-1: Urothelial 2-8% MSI-H. IHC MLH1/PMS2 loss PATHOGNOMONIC. PEMBROLIZUMAB FDA2017 ANY MSI-H urothelial APPROVED. Constitutional methylation ~15% NOT inherited (test before cascade). Aspirin 600mg CAPP2",
            "MSH6/Lynch-3: ENDOMETRIAL 71% ABSOLUTE HIGHEST SINGLE MMR GENE — dominant surveillance priority. Urothelial 5-7% secondary. 4 PSEUDOGENES chromosome 2 SEQUENCING PITFALL — MLPA mandatory MSH6. Pembrolizumab FDA2017 MSI-H",
            "BRCA1/HBOC-1: Urothelial 1.5-2x HRD. CISPLATIN PREFERRED OVER CARBOPLATIN BRCA1 (stronger HRD lethality). Olaparib FDA2020 HRD maintenance. Breast 72% dominant priority. BSO 35-40yr",
            "BRCA2/HBOC-2: Urothelial 2-3x (HIGHER than BRCA1 elevation). OLAPARIB FDA2020 HRD-positive urothelial. Cisplatin platinum-sensitive. Pancreatic 5-7% → EUS/MRI 50yr+ separate MANDATORY",
            "RB1: SECONDARY TCC POST-RADIATION 5x at pelvic RT field. AVOID RADIATION RB1 GERMLINE (TCC 5x + sarcoma 40x). CDK4-6i INACTIVE RB1-null (palbociclib useless + harmful). Annual cystoscopy lifelong bilateral RB survivors MANDATORY. RT-based trimodality CONTRAINDICATED RB1",
            "TP53/LFS: AVOID RADIATION ABSOLUTELY (secondary TCC + sarcoma post-RT catastrophic). Trimodality RT CONTRAINDICATED LFS — cystectomy surgical approach MANDATORY. WB-MRI Toronto annual. Cystoscopy over CT urogram (radiation minimisation)",
            "PTEN/Cowden: Urothelial 5-8x elevated. Macrocephaly PATHOGNOMONIC Cowden (>97th centile). Lhermitte-Duclos cerebellar PATHOGNOMONIC. mTOR hyperactivation → everolimus/alpelisib therapy. Annual cystoscopy + cytology Cowden from 30-35yr. Breast 50-85% dominant",
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_overview(), indent=2))
