#!/usr/bin/env python3
"""CII-Deficiency-Atlas — Complete 6-Gene Nuclear-Encoded Complex II (Succinate Dehydrogenase) Atlas
All nuclear-encoded Succinate Dehydrogenase genes: 4 structural subunits + 2 assembly factors
240-patient aggregate cohort (6 × 40, seeds 703–708)

Complex II (Succinate Dehydrogenase / Succinate:Ubiquinone Oxidoreductase, SQR) is unique:
  - ONLY OXPHOS complex with ALL subunits nuclear-encoded (zero mtDNA-encoded subunits)
  - DUAL function: OXPHOS respiratory chain (electron transfer) AND TCA cycle enzyme
    (SDHA/SDHB catalyse succinate → fumarate in TCA cycle)
  - CII is ALWAYS NORMAL in pure mtDNA disorders → used as "internal reference"
    for biochemical fingerprinting (any complex deficiency with CII NORMAL → nuclear or mtDNA CI/CIII/CIV/CV)
  - SDHx mutations cause CANCER (paraganglioma/pheochromocytoma) more commonly than
    classic mitochondrial Leigh disease — unique among OXPHOS complexes

SUBUNIT COMPOSITION (4 structural):
  Fp (flavoprotein) subunit:  SDHA (70 kDa) — FAD-binding; succinate oxidation site
  Ip (iron-sulfur) subunit:   SDHB (32 kDa) — 3 Fe-S clusters; electron relay
  Membrane anchors:           SDHC (15 kDa), SDHD (17 kDa) — ubiquinone binding; membrane attachment

ASSEMBLY FACTORS (2 genes):
  SDHAF1 (LYRM8) — [2Fe-2S] cluster insertion into SDHB; early assembly
  SDHAF2 (SDH5)  — covalent FAD flavination of SDHA His99; prerequisite for catalysis

IMPRINTING (unique feature — only SDHx genes with parental imprinting):
  SDHD (PGL1): paternal imprinting — ONLY paternal transmission causes disease
  SDHAF2 (PGL2): paternal imprinting — ONLY paternal transmission causes disease
  SDHA, SDHB, SDHC, SDHAF1: standard AD or AR without imprinting

IHC SURROGATE:
  SDHB IHC: universal SDHx surrogate — loss of granular cytoplasmic SDHB = any SDHx mutation
  SDHA IHC: ONLY SDHA mutations lose BOTH SDHA + SDHB staining (SDHB IHC alone insufficient for SDHA)

METASTATIC RISK (PGL/PHEO):
  SDHB: 25-40% metastatic (highest of all SDHx)
  SDHD: 1-4% metastatic (predominantly benign HNPGL)
  SDHC: 2-4% metastatic (predominantly benign HNPGL)
  SDHA: intermediate metastatic risk for PGL5
  SDHAF2: rare PGL2 cases; low metastatic risk

BIOCHEMICAL FINGERPRINT (CII-Leigh):
  SDHA: CII 5-15% (isolated CII deficiency; CI/CIII/CIV NORMAL)
  SDHAF1: CII 5-20% (isolated CII deficiency via failed [2Fe-2S] assembly)
  SDHB/SDHC/SDHD/SDHAF2: CII NORMAL in blood/fibroblasts; tumor-only CII loss

COHORT: 6 × 40 = 240 patient slots (seeds 703–708; gene-specific seeds)
"""

import random

SEED = 709
rng  = random.Random(SEED)

# ── All 6 nuclear-encoded CII-related genes — authoritative table ─────────────
# gene_class: "structural_subunit" | "assembly_factor"
# phenotype: primary phenotype class
CII_GENES = [
    {
        "gene": "SDHA",  "aa": "621 aa",  "kDa": "70 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "Fp",
        "cii_module": "Fp subunit (flavoprotein) — FAD binding, succinate oxidation active site",
        "omim_gene": 600857,  "chromosome": "5p15.33",  "seed": 703,
        "phenotype": "PGL5 + Leigh/CII-Deficiency",
        "disease": "SDHA-related PGL5 / CII Deficiency — Isolated CII Deficiency 5-15% (Leigh) OR Hereditary PGL5",
        "disease_omim_leigh": 252011,  "disease_omim_pgl": 614165,  "inheritance": "AR (Leigh) / AD (PGL5)",
        "hallmark": "ONLY SDHx gene causing BOTH classic mitochondrial Leigh disease AND paraganglioma/PHEO; FAD prosthetic group covalently attached at His544",
        "key_ddx": "vs SDHB/SDHC/SDHD: those cause PGL/PHEO NOT Leigh; SDHA IHC: loss of BOTH SDHA+SDHB (SDHB alone insufficient for SDHA mutations)",
        "founder_variant": "p.Arg31Cys (c.91C>T) — CII assembly disruption, recurrent; p.Pro475Ser (FAD-binding region)",
        "sdhb_ihc": "SDHB IHC LOST + SDHA IHC LOST (diagnostic: only SDHA mutations lose SDHA staining)",
        "metastatic_risk": "Intermediate (PGL5) / N/A (Leigh)",
        "hnpgl": False, "leigh_capable": True, "imprinting": None,
        "cii_activity_mean": 10.0, "cii_activity_sd": 4.0,
    },
    {
        "gene": "SDHB",  "aa": "280 aa",  "kDa": "32 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "Ip",
        "cii_module": "Ip subunit (iron-sulfur protein) — 3 Fe-S clusters [2Fe-2S]/[4Fe-4S]/[3Fe-4S]; electron relay Fp→membrane",
        "omim_gene": 185470,  "chromosome": "1p36.13",  "seed": 704,
        "phenotype": "PGL4 (highest malignancy)",
        "disease": "Hereditary Paraganglioma-Pheochromocytoma type 4 (PGL4) — SDHB highest metastatic rate 25-40%",
        "disease_omim_leigh": None,  "disease_omim_pgl": 115310,  "inheritance": "AD (incomplete penetrance)",
        "hallmark": "HIGHEST METASTATIC RATE of all SDHx genes (25-40%); extra-adrenal PGL > adrenal PHEO predominance; early surveillance mandatory",
        "key_ddx": "vs SDHD: SDHB higher metastatic risk; vs VHL/RET/NF1/MAX: all cause PGL/PHEO — SDHx panel first; SDHB IHC LOST in ALL SDHx mutations",
        "founder_variant": "p.Leu139Pro (hotspot UK/Europe); p.Gly12Val; p.Arg27Ter",
        "sdhb_ihc": "SDHB IHC LOST (universal SDHx surrogate — triggers full SDHx panel including SDHA/C/D IHC)",
        "metastatic_risk": "25-40% (HIGHEST of all SDHx)",
        "hnpgl": True, "leigh_capable": False, "imprinting": None,
        "cii_activity_mean": 95.0, "cii_activity_sd": 8.0,  # normal in non-tumor tissue
    },
    {
        "gene": "SDHC",  "aa": "169 aa",  "kDa": "15 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "membrane",
        "cii_module": "Membrane anchor subunit — ubiquinone (CoQ) binding site; attaches CII heterodimer to IMM",
        "omim_gene": 602413,  "chromosome": "1q23.3",  "seed": 705,
        "phenotype": "PGL3 (predominantly HNPGL)",
        "disease": "Hereditary Paraganglioma type 3 (PGL3) — predominantly head and neck paraganglioma (HNPGL)",
        "disease_omim_leigh": None,  "disease_omim_pgl": 605373,  "inheritance": "AD (highly penetrant for HNPGL)",
        "hallmark": "HNPGL predominance (carotid body, glomus jugulare, glomus tympanicum); lower metastatic rate than SDHB; Carney triad association (SDHC epimutation)",
        "key_ddx": "Carney Triad (GIST + PGL + pulmonary chondroma) — SDHC epimutation/somatic vs germline; vs SDHB (higher metastatic risk)",
        "founder_variant": "p.Arg133Ter (recurrent UK); Carney triad: somatic SDHC promoter methylation",
        "sdhb_ihc": "SDHB IHC LOST (triggers full SDHx workup)",
        "metastatic_risk": "2-4% (predominantly benign HNPGL)",
        "hnpgl": True, "leigh_capable": False, "imprinting": None,
        "cii_activity_mean": 96.0, "cii_activity_sd": 7.0,
    },
    {
        "gene": "SDHD",  "aa": "159 aa",  "kDa": "17 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "membrane",
        "cii_module": "Membrane anchor subunit — works with SDHC to anchor Fp+Ip heterodimer to IMM; ubiquinone reduction site",
        "omim_gene": 602690,  "chromosome": "11q23.1",  "seed": 706,
        "phenotype": "PGL1 (paternal imprinting, most common HNPGL gene)",
        "disease": "Hereditary Paraganglioma type 1 (PGL1) — PATERNAL IMPRINTING: only paternally inherited mutations are disease-causing",
        "disease_omim_leigh": None,  "disease_omim_pgl": 168000,  "inheritance": "AD with PATERNAL IMPRINTING (maternal transmission silent)",
        "hallmark": "PARENTAL IMPRINTING — only PATERNALLY inherited SDHD mutations cause PGL1; maternally inherited SDHD = silent carrier; most common HNPGL gene worldwide",
        "key_ddx": "vs SDHAF2: also paternal imprinting but rarer; MUST check transmission route before diagnosis; vs SDHB: SDHD lower metastatic risk",
        "founder_variant": "p.Asp92Tyr (Dutch founder, common Leiden/Groningen); p.Pro81Leu",
        "sdhb_ihc": "SDHB IHC LOST (triggers full SDHx workup including SDHD-specific imprinting history)",
        "metastatic_risk": "1-4% (predominantly benign, multifocal HNPGL common)",
        "hnpgl": True, "leigh_capable": False, "imprinting": "paternal",
        "cii_activity_mean": 97.0, "cii_activity_sd": 5.0,
    },
    {
        "gene": "SDHAF1",  "aa": "118 aa",  "kDa": "13 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF",
        "cii_module": "Assembly factor 1 (LYRM8) — inserts [2Fe-2S] cluster into SDHB Ip subunit; early CII biogenesis step",
        "omim_gene": 612848,  "chromosome": "19q13.12",  "seed": 707,
        "phenotype": "Infantile Leukoencephalopathy (isolated CII deficiency)",
        "disease": "SDHAF1 Infantile Leukoencephalopathy — Isolated CII Deficiency (OMIM 612080); NO PGL/PHEO",
        "disease_omim_leigh": 612080,  "disease_omim_pgl": None,  "inheritance": "AR",
        "hallmark": "INFANTILE LEUKOENCEPHALOPATHY (white matter disease) — NOT Leigh basal ganglia pattern; ISOLATED CII deficiency 5-20%; NO paraganglioma/PHEO; LYRM motif protein",
        "key_ddx": "vs SDHA: SDHA has Leigh BG pattern; SDHAF1 = white matter (leukoencephalopathy), not Leigh; vs ADAR1-AGS/leukodystrophies: CII activity distinguishes",
        "founder_variant": "p.Thr77Pro (Italian founder); p.Leu102Pro",
        "sdhb_ihc": "SDHB IHC LOST in muscle/fibroblast (CII assembly failure — Ip cannot fold correctly without [2Fe-2S])",
        "metastatic_risk": "N/A (no PGL/PHEO)",
        "hnpgl": False, "leigh_capable": True, "imprinting": None,
        "cii_activity_mean": 12.0, "cii_activity_sd": 5.0,
    },
    {
        "gene": "SDHAF2",  "aa": "166 aa",  "kDa": "18 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF",
        "cii_module": "Assembly factor 2 (SDH5/HIF2AN) — covalent FAD flavination of SDHA His99; required BEFORE Fp-Ip heterodimerisation",
        "omim_gene": 613019,  "chromosome": "11q12.2",  "seed": 708,
        "phenotype": "PGL2 (paternal imprinting, rare HNPGL)",
        "disease": "Hereditary Paraganglioma type 2 (PGL2) — PATERNAL IMPRINTING: only paternally inherited SDHAF2 mutations cause disease",
        "disease_omim_leigh": None,  "disease_omim_pgl": 601650,  "inheritance": "AD with PATERNAL IMPRINTING (maternal transmission silent)",
        "hallmark": "SECOND gene with PATERNAL IMPRINTING (after SDHD); SDHAF2 flavinates SDHA His99 — without FAD attachment SDHA cannot catalyse; predominantly HNPGL; rarest SDHx PGL type",
        "key_ddx": "vs SDHD: both paternally imprinted; SDHAF2 rarer; vs SDHA: SDHAF2 causes PGL not Leigh (FAD flavination lost → unflavinated SDHA → no catalysis, not Leigh in most cases)",
        "founder_variant": "p.Gly78Arg (large Dutch family — Baysal 2000 Science landmark discovery); c.232_233delAG",
        "sdhb_ihc": "SDHB IHC LOST in tumour (secondary to SDHA assembly failure without FAD → whole CII lost)",
        "metastatic_risk": "Low (predominantly benign HNPGL, rare adrenal)",
        "hnpgl": True, "leigh_capable": False, "imprinting": "paternal",
        "cii_activity_mean": 96.0, "cii_activity_sd": 6.0,
    },
]

# ── Patient cohort generation ─────────────────────────────────────────────────
def _gen_patient(gene_info, idx):
    rg = random.Random(gene_info["seed"] * 1000 + idx)
    gene = gene_info["gene"]
    is_leigh = gene_info["leigh_capable"]
    is_pgl   = not is_leigh or gene == "SDHA"

    if is_leigh and gene != "SDHA":
        # Pure CII-Leigh (SDHA Leigh subset or SDHAF1)
        age_onset_months = int(rg.gauss(6, 4)); age_onset_months = max(1, age_onset_months)
        phenotype = "CII-Leigh"
    elif gene == "SDHA":
        # Mixed: Leigh OR PGL5
        if rg.random() < 0.55:
            age_onset_months = int(rg.gauss(8, 5)); age_onset_months = max(1, age_onset_months)
            phenotype = "CII-Leigh"
        else:
            age_onset_months = int(rg.gauss(360, 120))  # adult onset for PGL
            age_onset_months = max(120, age_onset_months)
            phenotype = "PGL5"
    else:
        # PGL/PHEO phenotype — adult onset
        age_onset_months = int(rg.gauss(360, 96))
        age_onset_months = max(120, age_onset_months)
        phenotype = "PGL/PHEO"

    # CII enzyme activity (% of mean normal)
    cii_pct = max(3.0, min(110.0, rg.gauss(gene_info["cii_activity_mean"], gene_info["cii_activity_sd"])))

    # Clinical features depend on phenotype
    leigh_mri  = (phenotype == "CII-Leigh") and rg.random() < (0.75 if gene == "SDHA" else 0.65)
    white_matter = (phenotype == "CII-Leigh" and gene == "SDHAF1") and rg.random() < 0.78
    lactic_ac  = (phenotype == "CII-Leigh") and rg.random() < 0.70
    pgl_tumor  = (phenotype in ("PGL/PHEO", "PGL5")) and rg.random() < 0.82
    pheo       = pgl_tumor and (gene == "SDHB") and rg.random() < 0.35
    hnpgl      = pgl_tumor and gene_info["hnpgl"] and rg.random() < 0.72
    metastatic = pgl_tumor and (gene == "SDHB") and rg.random() < 0.30
    multifocal = pgl_tumor and (gene == "SDHD") and rg.random() < 0.45

    # Imprinting — track transmission
    paternal_tx = None
    if gene_info["imprinting"] == "paternal":
        paternal_tx = rg.random() < 0.50  # 50% chance paternal vs maternal transmission

    sdhb_ihc_loss = True  # all SDHx mutations lose SDHB IHC in tumor
    sdha_ihc_loss = (gene == "SDHA")  # only SDHA loses SDHA IHC

    return {
        "patient_id":      f"{gene}-{idx:03d}",
        "gene":            gene,
        "gene_class":      gene_info["gene_class"],
        "phenotype":       phenotype,
        "age_onset_months":age_onset_months,
        "cii_activity_pct":round(cii_pct, 1),
        "leigh_mri":       leigh_mri,
        "white_matter_leukoencephalopathy": white_matter,
        "lactic_acidosis": lactic_ac,
        "pgl_tumor":       pgl_tumor,
        "pheo":            pheo,
        "hnpgl":           hnpgl,
        "metastatic":      metastatic,
        "multifocal_pgl":  multifocal,
        "paternal_transmission": paternal_tx,
        "sdhb_ihc_loss":   sdhb_ihc_loss if pgl_tumor else None,
        "sdha_ihc_loss":   sdha_ihc_loss if pgl_tumor else None,
    }

COHORT = []
for g in CII_GENES:
    for i in range(40):
        COHORT.append(_gen_patient(g, i))


def get_overview():
    n_genes      = len(CII_GENES)
    n_structural = sum(1 for g in CII_GENES if g["gene_class"] == "structural_subunit")
    n_af         = sum(1 for g in CII_GENES if g["gene_class"] == "assembly_factor")
    n_patients   = len(COHORT)
    leigh_pts    = [p for p in COHORT if p["phenotype"] == "CII-Leigh"]
    pgl_pts      = [p for p in COHORT if p["phenotype"] in ("PGL/PHEO", "PGL5")]
    metastatic_pts = [p for p in pgl_pts if p["metastatic"]]
    hnpgl_pts    = [p for p in pgl_pts if p["hnpgl"]]

    # Aggregate phenotype stats
    pgl_rate     = round(len(pgl_pts) / n_patients * 100, 1)
    leigh_rate   = round(len(leigh_pts) / n_patients * 100, 1)
    met_rate_sdhb = round(sum(1 for p in COHORT if p["gene"]=="SDHB" and p["metastatic"]) /
                          max(1, sum(1 for p in COHORT if p["gene"]=="SDHB" and p["pgl_tumor"])) * 100, 1)

    return {
        "atlas":        "CII-Deficiency-Atlas — Complete 6-Gene Nuclear-Encoded Complex II Reference",
        "complex_ii": {
            "full_name":        "Succinate:Ubiquinone Oxidoreductase (SQR) / Succinate Dehydrogenase (SDH)",
            "subunits_total":   4,
            "subunits_nuclear": 4,
            "subunits_mtDNA":   0,
            "assembly_factors": 2,
            "total_genes":      6,
            "size_kDa":         "~124 kDa heterotetramer",
            "function_oxphos":  "Transfers electrons from succinate to ubiquinone (CoQ); connects TCA cycle to ETC",
            "function_tca":     "Catalyses succinate → fumarate (SDHA/SDHB); Step 6 of TCA cycle",
            "unique_feature":   "ONLY OXPHOS complex with ALL subunits nuclear-encoded — CII ALWAYS NORMAL in mtDNA disorders",
        },
        "cii_always_normal_rule": {
            "rule": "CII ALWAYS NORMAL in pure mtDNA disorders (no mtDNA-encoded CII subunits)",
            "implication": "Normal CII activity = mtDNA or CII-nuclear origin; abnormal CII = SDHA or SDHAF1 nuclear mutation",
            "internal_reference": "CII used as internal enzymatic reference in biochemical OXPHOS panels (validates assay; excludes mtDNA CI/CIII/CIV/CV)",
        },
        "series_breakdown": {
            "structural_subunits": n_structural,
            "assembly_factors":    n_af,
            "total_genes":         n_genes,
        },
        "imprinting_genes": {
            "sdhd": {"gene":"SDHD","mechanism":"Paternal imprinting","rule":"Only PATERNALLY inherited SDHD mutations cause PGL1"},
            "sdhaf2": {"gene":"SDHAF2","mechanism":"Paternal imprinting","rule":"Only PATERNALLY inherited SDHAF2 mutations cause PGL2"},
            "note": "Maternally inherited SDHD or SDHAF2 mutations are SILENT in the carrier's offspring (imprinted gene)",
        },
        "ihc_diagnostic": {
            "sdhb_universal": "SDHB IHC: universal SDHx surrogate — loss of granular cytoplasmic staining in tumor = any SDHx mutation → triggers full panel",
            "sdha_specific":  "SDHA IHC: ONLY SDHA mutations cause SDHA IHC loss; SDHB alone insufficient for SDHA diagnosis — MUST add SDHA IHC",
            "algorithm":      "All neuroendocrine tumors (PGL/PHEO): SDHB IHC first → if lost: SDHA IHC + full SDHx panel (SDHA/B/C/D/AF2 sequencing)",
        },
        "metastatic_risk": {
            "SDHB": "25-40% (HIGHEST of all SDHx — mandatory intensive surveillance)",
            "SDHA": "Intermediate (~10-15% for PGL5)",
            "SDHC": "2-4% (predominantly benign HNPGL)",
            "SDHD": "1-4% (predominantly benign, multifocal HNPGL)",
            "SDHAF2": "Rare/low (mostly benign HNPGL)",
        },
        "cohort": {
            "total_patients": n_patients,
            "genes_covered":  n_genes,
            "patients_per_gene": 40,
            "seeds":          "703–708 (gene-specific)",
            "pgl_pheo_count": len(pgl_pts),
            "leigh_cii_count": len(leigh_pts),
        },
        "aggregate_clinical": {
            "pgl_pheo_rate_pct":        pgl_rate,
            "leigh_cii_rate_pct":       leigh_rate,
            "hnpgl_among_pgl_pct":      round(len(hnpgl_pts) / max(1, len(pgl_pts)) * 100, 1),
            "sdhb_metastatic_rate_pct": met_rate_sdhb,
            "cii_deficiency_mean_leigh_pct": round(
                sum(p["cii_activity_pct"] for p in leigh_pts) / max(1, len(leigh_pts)), 1),
        },
        "drug_considerations": {
            "leigh_phenotype": {
                "absolute_ci": [
                    {"drug": "VPA / Valproate", "mechanism": "CoA sequestration → secondary CII substrate depletion; succinyl-CoA pathway disruption"},
                    {"drug": "Metformin", "mechanism": "CI direct inhibition → NADH accumulates → TCA cycle slows → succinate accumulation backpressure on CII"},
                    {"drug": "Propofol", "mechanism": "PRIS (Propofol Infusion Syndrome) — OXPHOS inhibition including CII"},
                    {"drug": "Linezolid", "mechanism": "Inhibits mitochondrial 23S-like ribosome → depletes all nuclear-encoded OXPHOS subunits"},
                    {"drug": "Chloramphenicol", "mechanism": "Mitoribosome inhibition → secondary OXPHOS deficiency"},
                ],
                "mandatory": [
                    "Thiamine (B1) — empiric BTBGD/SLC19A3 exclusion; PDH/αKGDH TCA cycle cofactor",
                    "Biotin — empiric BTD/BTBGD exclusion before diagnosing CII-Leigh",
                    "GIR 6-8 mg/kg/min — support glucose oxidation; avoid fasting",
                    "BTBGD/SLC19A3 MANDATORY exclusion — Leigh-like white matter/BG mimic; Biotin+Thiamine dramatic response",
                    "Levetiracetam (LEV) preferred AED — renal, no CYP450, no mito toxicity",
                    "Riboflavin (B2) — FAD source; theoretical benefit in SDHA (FAD-binding) and SDHAF2 (FAD flavination) mutations",
                ],
            },
            "pgl_pheo_phenotype": {
                "note": "PGL/PHEO patients have NORMAL mitochondrial function in non-tumor tissues; standard mito drug contraindications apply primarily if CII-Leigh features present",
                "pheo_specific": [
                    "Alpha-blockade (phenoxybenzamine/doxazosin) MANDATORY before beta-blockade (catecholamine surge risk)",
                    "AVOID dopamine/tyramine-rich foods (catecholamine release exacerbation)",
                    "Somatostatin analogues for SDHB-metastatic PGL (functional imaging-directed)",
                ],
            },
        },
        "surveillance_protocols": {
            "SDHB": "Whole-body MRI/CT every 12-24 months (highest metastatic risk); 24h urine catecholamines/metanephrines annually",
            "SDHD": "Head/neck MRI every 24 months (HNPGL), whole-body every 36-48 months; family cascade screening (paternal only)",
            "SDHC": "Head/neck MRI every 24-36 months; Carney triad screen (GIST + pulmonary chondroma)",
            "SDHA": "Annual surveillance; PGL5 whole-body + metabolomics; Leigh: metabolic monitoring",
            "SDHAF1": "Brain MRI (white matter) every 12 months; CII enzyme activity monitoring",
            "SDHAF2": "Head/neck MRI every 24 months; MUST trace paternal inheritance before surveillance cascade",
        },
        "sdh_structure": {
            "Fp_heterodimer":  "SDHA (Fp) + SDHB (Ip) = water-soluble peripheral arm; catalytic and electron-relay subunits",
            "membrane_anchor": "SDHC + SDHD = membrane-integral anchor; binds ubiquinone; tethers Fp+Ip to IMM",
            "FAD_covalent":    "FAD covalently linked to SDHA His99 (by SDHAF2); essential for succinate oxidation",
            "FeS_relay":       "SDHB carries [2Fe-2S] (N1a), [4Fe-4S] (N2-like), [3Fe-4S] (N3-like) clusters; inserted by SDHAF1",
            "ubiquinone_site": "Q-site at SDHC/SDHD interface; ubiquinol (QH₂) formed here; feeds CIII",
        },
        "carney_triad": {
            "definition": "GIST + Paraganglioma + Pulmonary chondroma (rare, non-hereditary)",
            "sdh_link": "SDHC epimutation (somatic promoter methylation, NOT germline) — biallelic SDHC loss in GIST/PGL component",
            "ddx_from_germline_sdhc": "Carney Triad = somatic/epigenetic SDHC → no family history; germline SDHC (PGL3) = family history with HNPGL",
        },
        "wes_utility": {
            "SDHA": "WES detectable — nuclear 5p15.33",
            "SDHB": "WES detectable — nuclear 1p36.13",
            "SDHC": "WES detectable — nuclear 1q23.3",
            "SDHD": "WES detectable — nuclear 11q23.1",
            "SDHAF1": "WES detectable — nuclear 19q13.12",
            "SDHAF2": "WES detectable — nuclear 11q12.2",
            "panel_note": "All 6 CII genes nuclear-encoded → WES detects all; targeted SDHx panels (SDHA/B/C/D/AF2) preferred clinically for PGL/PHEO cascade",
        },
    }


def get_breakdown():
    rows = []
    for g in CII_GENES:
        pts = [p for p in COHORT if p["gene"] == g["gene"]]
        leigh_pts = [p for p in pts if p["phenotype"] == "CII-Leigh"]
        pgl_pts   = [p for p in pts if p["phenotype"] in ("PGL/PHEO", "PGL5")]

        leigh_pct = round(len(leigh_pts) / len(pts) * 100, 1) if pts else 0
        pgl_pct   = round(len(pgl_pts) / len(pts) * 100, 1) if pts else 0
        metastatic_pct = round(sum(1 for p in pts if p["metastatic"]) / max(1, len(pgl_pts)) * 100, 1)
        hnpgl_pct = round(sum(1 for p in pts if p["hnpgl"]) / max(1, len(pgl_pts)) * 100, 1)
        lactic_pct = round(sum(1 for p in pts if p["lactic_acidosis"]) / len(pts) * 100, 1) if pts else 0
        white_m_pct = round(sum(1 for p in pts if p["white_matter_leukoencephalopathy"]) / len(pts) * 100, 1)
        mean_cii  = round(sum(p["cii_activity_pct"] for p in pts) / len(pts), 1) if pts else 0
        median_onset = sorted(p["age_onset_months"] for p in pts)[len(pts)//2] if pts else 0

        rows.append({
            "gene":             g["gene"],
            "gene_class":       g["gene_class"],
            "subunit_series":   g["subunit_series"],
            "cii_module":       g["cii_module"],
            "omim_gene":        g["omim_gene"],
            "chromosome":       g["chromosome"],
            "seed":             g["seed"],
            "n_patients":       len(pts),
            "phenotype":        g["phenotype"],
            "inheritance":      g["inheritance"],
            "imprinting":       g["imprinting"],
            "leigh_capable":    g["leigh_capable"],
            "hnpgl_prone":      g["hnpgl"],
            "metastatic_risk":  g["metastatic_risk"],
            "median_onset_months": median_onset,
            "cii_activity_mean_pct": mean_cii,
            "leigh_mri_pct":    leigh_pct,
            "pgl_pheo_rate_pct": pgl_pct,
            "metastatic_pct_of_pgl": metastatic_pct,
            "hnpgl_pct_of_pgl": hnpgl_pct,
            "lactic_acidosis_pct": lactic_pct,
            "white_matter_pct": white_m_pct,
            "disease_summary":  g["disease"][:90],
            "hallmark":         g["hallmark"][:120],
            "founder_variant":  g["founder_variant"],
            "sdhb_ihc":         g["sdhb_ihc"],
        })
    return {"genes": rows, "total": len(rows), "total_patients": len(COHORT)}


def get_definitions():
    return {
        "atlas":             "CII-Deficiency-Atlas — Complete 6-gene nuclear-encoded Complex II reference (4 subunits + 2 assembly factors)",
        "complex_ii":        "Succinate Dehydrogenase / Succinate:Ubiquinone Oxidoreductase (SQR) — tetrameric ~124 kDa IMM complex; only OXPHOS complex with ALL subunits nuclear-encoded; dual OXPHOS+TCA function",
        "CII_always_normal": "CII enzyme activity is ALWAYS NORMAL in pure mtDNA disorders — no mtDNA-encoded CII subunits exist; CII normal = mtDNA or CI/CIII/CIV/CV nuclear etiology",
        "SDHA_Fp":           "SDHA (70 kDa, 5p15.33) — flavoprotein subunit; FAD covalently attached at His544; oxidises succinate to fumarate; ONLY SDHx gene causing Leigh + PGL5",
        "SDHB_Ip":           "SDHB (32 kDa, 1p36.13) — iron-sulfur protein; 3 Fe-S clusters [2Fe-2S]/[4Fe-4S]/[3Fe-4S] relay electrons Fp→membrane; highest metastatic PGL4 risk 25-40%",
        "SDHC_membrane":     "SDHC (15 kDa, 1q23.3) — membrane anchor; ubiquinone binding; PGL3/HNPGL; Carney Triad (somatic SDHC epimutation — GIST+PGL+pulmonary chondroma)",
        "SDHD_membrane":     "SDHD (17 kDa, 11q23.1) — membrane anchor; PGL1; PATERNAL IMPRINTING (only paternally inherited mutations cause disease); most common HNPGL gene; Dutch founder p.Asp92Tyr",
        "SDHAF1_assembly":   "SDHAF1/LYRM8 (13 kDa, 19q13.12) — LYRM-motif assembly factor; inserts [2Fe-2S] cluster into SDHB; isolated CII deficiency; INFANTILE LEUKOENCEPHALOPATHY (white matter, NOT Leigh BG)",
        "SDHAF2_assembly":   "SDHAF2/SDH5 (18 kDa, 11q12.2) — covalent FAD flavination of SDHA His99 (prerequisite for CII assembly); PGL2; PATERNAL IMPRINTING; Baysal 2000 Science discovery; Dutch founder p.Gly78Arg",
        "paternal_imprinting":"SDHD and SDHAF2 have paternal imprinting: the maternal allele is epigenetically silenced; ONLY a mutation on the PATERNALLY inherited allele causes PGL1/PGL2; maternal transmission = silent carrier",
        "SDHB_IHC":          "SDHB immunohistochemistry: universal surrogate marker for ALL SDHx mutations — loss of granular cytoplasmic staining in tumor = any SDHx (SDHA/B/C/D/AF2) mutation → triggers full panel sequencing",
        "SDHA_IHC":          "SDHA immunohistochemistry: additional IHC required for SDHA mutations — ONLY SDHA mutations lose SDHA staining; SDHB IHC alone insufficient to detect SDHA mutations",
        "PGL4_SDHB":         "PGL4 (SDHB): hereditary paraganglioma/pheochromocytoma type 4; 25-40% metastatic risk (HIGHEST); extra-adrenal PGL > adrenal PHEO; early/intensive surveillance mandatory",
        "PGL1_SDHD":         "PGL1 (SDHD): hereditary paraganglioma type 1; predominantly HNPGL; 1-4% metastatic; multifocal common; Dutch founder p.Asp92Tyr; paternal imprinting (MUST check inheritance)",
        "PGL3_SDHC":         "PGL3 (SDHC): predominantly HNPGL; 2-4% metastatic; Carney triad (somatic SDHC epimutation) separate entity from germline PGL3",
        "PGL5_SDHA":         "PGL5 (SDHA): hereditary paraganglioma type 5; intermediate metastatic risk; ONLY SDHx PGL gene that also causes classic Leigh/CII-Leigh with AR biallelic mutations",
        "PGL2_SDHAF2":       "PGL2 (SDHAF2): rarest SDHx PGL type; paternal imprinting (like SDHD); predominantly HNPGL; low metastatic risk; Baysal 2000 Science discovery (Dutch family)",
        "Carney_Triad":      "GIST + Paraganglioma + Pulmonary chondroma — somatic SDHC epimutation (promoter methylation), NOT germline; biallelic SDHC loss in tumor; sporadic (no family history)",
        "Carney_Dyad":       "GIST + Paraganglioma — SDH-deficient GIST + PGL; SDHA/B/C/D germline OR somatic mutations; some cases = Carney Triad without pulmonary chondroma",
        "FeS_relay_SDHB":    "SDHB Fe-S cluster electron relay: electrons from FAD-SDHA → [2Fe-2S] N1a → [4Fe-4S] N2-like → [3Fe-4S] N3-like → ubiquinone at SDHC/SDHD Q-site",
        "FAD_flavination":   "Covalent FAD attachment to SDHA His99 by SDHAF2 (SDH5): prerequisite for CII catalytic activity; SDHAF2 mutations → unflavinated SDHA → CII assembly failure → PGL2",
        "lyrm_sdhaf1":       "LYRM motif (LYR tripeptide) in SDHAF1: conserved Fe-S carrier chaperone domain; coordinates [2Fe-2S] cluster insertion into SDHB; SDHAF1 mutation → failed [2Fe-2S] → SDHB misfolding → CII loss",
        "alpha_blockade":    "Alpha-adrenergic blockade (phenoxybenzamine/doxazosin) MANDATORY before beta-blockade in PHEO — catecholamine excess causes hypertensive crisis; beta-blockade without alpha = paradoxical vasoconstriction",
        "BTBGD_CII":         "SLC19A3 (BTBGD) MUST be excluded before diagnosing SDHAF1/SDHA CII-Leigh — treatable Leigh-like mimic; Biotin+Thiamine dramatic response; identical white matter/BG MRI pattern",
        "vpa_cii":           "VPA absolute CI in CII-Leigh (SDHA/SDHAF1): CoA sequestration → succinyl-CoA depletion → direct TCA/CII substrate impact; secondary to CI inhibition",
        "riboflavin_cii":    "Riboflavin (B2) → FAD → SDHA prosthetic group and SDHAF2 substrate; theoretical benefit in SDHA (missense alleles with partial FAD binding) and SDHAF2 mutations; Level C evidence",
        "sdh_tca":           "SDHA/SDHB = succinate dehydrogenase enzyme = TCA cycle Step 6 (succinate → fumarate); CII is the ONLY OXPHOS complex that is also a TCA cycle enzyme; FAD-linked (unlike NAD-linked CI/CIII)",
        "WES_CII":           "All 6 CII genes nuclear-encoded → ALL WES-detectable; targeted SDHx panels (SDHA/B/C/D/AF1/AF2) preferred for PGL/PHEO cascade testing; SDHB IHC first for neuroendocrine tumors",
    }
