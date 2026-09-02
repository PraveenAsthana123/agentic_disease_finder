#!/usr/bin/env python3
"""SDHA — Succinate Dehydrogenase Subunit A / Flavoprotein Catalytic Subunit /
Complex II Deficiency (Leigh Syndrome, AR) + Paraganglioma 5 (PGL5, AD) +
Carney-Stratakis Syndrome (GIST + PGL, AD).

SDHA (Succinate Dehydrogenase Subunit A; OMIM *600857) encodes the 586-amino-acid,
~70 kDa flavoprotein catalytic subunit of Complex II (succinate dehydrogenase, SDH) —
the only enzyme shared by both the TCA cycle (succinate → fumarate) and the mitochondrial
electron transport chain (FADH2 → ubiquinone). SDHA contains the covalently-bound FAD
cofactor (at His99, attached by SDHAF2) and the active site for succinate oxidation.

  SDHA gene     OMIM *600857
  Protein       Succinate dehydrogenase subunit A (flavoprotein subunit)
  Size          586 aa, ~70 kDa
  Location      Mitochondrial matrix face (no transmembrane helices)
  Chromosome    5p15.33
  CII role      Catalytic FAD-containing subunit; binds SDHB to form SDHA-SDHB core

Disease 1:  Complex II Deficiency (Leigh syndrome variant) — OMIM #252011 / CI2DN1
  Inheritance   AR (autosomal recessive, biallelic)
  Onset         Infantile (3–18 months)
  MRI           Bilateral basal ganglia + brainstem lesions (Leigh pattern, 80%)
  Biochemistry  Isolated CII deficiency 5–30%; CI, CIII, CIV normal
  Key DDx       SDHAF1 (leukoencephalopathy, not Leigh); SURF1 (Leigh + CIV); POLG (multi-organ)

Disease 2:  Paraganglioma 5 (PGL5) — OMIM #614165
  Inheritance   AD (autosomal dominant), low penetrance ~10–15%
  Presentation  Head-neck PGL, adrenal PCC, SDH-deficient GIST
  NOT imprinted Unlike SDHD (PGL1) and SDHAF2 (PGL2); biparental transmission penetrant
  Key DDx       SDHB-PGL4 (high malignancy 20–50%); SDHD-PGL1 (maternal imprinting, higher penetrance)

Disease 3:  Carney-Stratakis Syndrome — OMIM #606764
  Description   Hereditary dyad: paraganglioma + gastrointestinal stromal tumor (GIST)
  Inheritance   AD, incomplete penetrance
  GIST          SDH-deficient GIST (KIT/PDGFRA wild-type), multifocal gastric GIST
  Key DDx       Sporadic GIST (KIT/PDGFRA mutations, SDH-proficient, single lesion)

Reference: Parfait B et al. (2000) Compound heterozygous mutations in the flavoprotein gene
of the respiratory chain complex II in a patient with Leigh syndrome.
Hum Genet 106(2):236–243.
(First SDHA biallelic mutations causing Leigh syndrome / CII deficiency in a patient)

Reference: Burnichon N et al. (2010) SDHA is a tumor suppressor gene causing paraganglioma.
Hum Mol Genet 19(15):3011–3020.
(First identification of SDHA heterozygous mutations in PGL5; SDHA as tumor suppressor)

Reference: Janeway KA et al. (2011) Defects in succinate dehydrogenase in gastrointestinal
stromal tumors lacking KIT and PDGFRA mutations.
Science 331(6014):220–223.
(SDHA-deficient GIST in Carney-Stratakis; SDH-deficient GIST pathomechanism)

Reference: Dwight T et al. (2013) SDHA germline mutations rare in phaeochromocytoma and
paraganglioma. Clin Endocrinol 78(4):515–519.
(SDHA mutation frequency and penetrance in PGL5 series; ~10% penetrance confirmed)

PATHOPHYSIOLOGY (SDHA — dual role: CII catalysis + tumor suppression):

  SDHA in CII catalysis (normal):
    1. SDHAF2 covalently attaches FAD to SDHA His99 (flavinylation — Step 1 of CII assembly)
    2. SDHA (flavinylated) binds SDHAF1-matured SDHB to form the SDHA-SDHB catalytic core
    3. SDHA-SDHB dimer binds SDHC-SDHD membrane anchor subunits → CII holoenzyme
    4. CII: succinate + FAD → fumarate + FADH2; FADH2 → ubiquinol (ETC)
    5. Normal: succinate oxidized; fumarate produced; FADH2 feeds ETC at ubiquinone

  SDHA loss-of-function (biallelic, AR — Leigh/CII):
    1. No functional SDHA catalytic subunit → CII activity abolished (5–30% residual max)
    2. CII selectively deficient; CI/CIII/CIV normal (biochemical fingerprint)
    3. Succinate cannot be oxidized by CII → succinate accumulates
    4. TCA cycle interrupted (succinate → fumarate blocked); NADH backup via other TCA reactions
    5. FADH2 cannot enter ETC via CII → profound energy deficit in high-demand cells
    6. Basal ganglia / brainstem most vulnerable → bilateral symmetric Leigh lesions
    7. Cardiomyopathy in ~25% (high mitochondrial demand in cardiac tissue)
    8. FADH2 entry exclusively via CII — KD CONTRAINDICATED (fatty acid β-oxidation)

  SDHA loss-of-function (monoallelic, AD — PGL5):
    1. Heterozygous germline SDHA mutation → haploinsufficiency
    2. In susceptible neuroendocrine cells, somatic second-hit at SDHA locus (LOH/mutation)
    3. Complete CII loss in tumor cells → succinate-driven pseudo-hypoxia
    4. Succinate inhibits PHD (prolyl hydroxylase domain) enzymes
    5. HIF1α not hydroxylated → not degraded → stabilized → pro-angiogenic/oncogenic
    6. Paraganglioma, pheochromocytoma, or SDH-deficient GIST development
    7. LOW PENETRANCE (~10%): modifier genes, stochastic second-hit timing
    8. NOT IMPRINTED: biparental transmission; maternal and paternal mutations equally penetrant
       (unlike SDHD/PGL1 and SDHAF2/PGL2 which require paternal transmission)

SDHA UNIQUE FEATURES:
  1. DUAL-DISEASE GENE — UNIQUE among SDH genes: SDHA causes BOTH (a) AR recessive
     infantile metabolic disease (CII deficiency/Leigh) from biallelic loss AND
     (b) AD dominant tumor predisposition (PGL5/Carney-Stratakis) from monoallelic loss.
     No other single SDH subunit gene causes both Leigh syndrome and paraganglioma.
  2. SDHA IS THE TARGET OF SDHAF2 (SDHA His99 flavinylation): SDHAF2 encodes the enzyme
     that flavinylates SDHA His99. SDHA mutations at or near His99 may impair flavinylation
     even when SDHAF2 is normal — causal distinction requires functional assay.
  3. 5p15.33 LOCUS — UNIQUE: SDHA is the only SDH subunit on chromosome 5. SDHB (1p36),
     SDHC (1q23), SDHD (11q23), SDHAF1 (19q13), SDHAF2 (11q13). WES mandatory.
  4. NOT MATERNALLY IMPRINTED (PGL5): Unlike SDHD (PGL1, maternal imprinting) and SDHAF2
     (PGL2, maternal imprinting), SDHA/PGL5 is NOT imprinted. Maternal AND paternal SDHA
     mutations can cause PGL5 (with equal ~10% penetrance). Critical for genetic counselling.
  5. CARNEY-STRATAKIS SYNDROME: SDHA germline mutations cause the Carney-Stratakis dyad
     (paraganglioma + SDH-deficient GIST) — the characteristic SDH-deficient gastric GIST
     pattern (multifocal, KIT/PDGFRA wild-type, young onset) is highly associated with SDHA.
  6. IMMUNOHISTOCHEMISTRY: SDH-deficient tumors (PGL5/Carney-Stratakis) show complete loss
     of both SDHA and SDHB staining on IHC — "SDHA-null" pattern (unique to SDHA loss;
     other SDH subunit losses only show SDHB loss, not SDHA loss).
  7. FAD-BINDING CATALYTIC CORE: SDHA contains the FAD cofactor and active site for succinate
     oxidation. Mutations disrupting the active site or FAD binding abolish catalysis entirely
     (severe, for AR disease). Mutations disrupting tumor-suppressor function cause PGL5.

DISTINGUISHING FEATURES vs OTHER SDH/LEIGH/PGL GENES:
  vs SDHAF1 (19q13.12): SDHAF1 = CII deficiency / infantile leukoencephalopathy (AR).
    SDHA-CII = Leigh syndrome, gray matter, basal ganglia (NOT white matter leukoencephalopathy).
    Both AR, both isolated CII deficiency, both MRS succinate elevation; but MRI pattern differs.
    SDHA: cardiomyopathy 25% vs SDHAF1: rare. Brain MRS: SDHA gray matter vs SDHAF1 white matter.
  vs SDHAF2 (11q13.1): SDHAF2 = PGL2, SDHA flavinylation factor, maternal imprinting.
    SDHA-PGL5 = not imprinted; penetrance 10% vs SDHAF2 90%. SDHA causes Leigh too; SDHAF2 does not.
    Both affect SDHA pathway (SDHAF2 flavinylates SDHA His99); different nodes, different diseases.
  vs SDHB (1p36.13): PGL4 — HIGHEST malignancy risk (20–50%). No imprinting. Most common hereditary PGL.
    SDHA-PGL5: malignancy 5%. SDHB: no CII/Leigh association. SDHA: IHC shows SDHA + SDHB loss.
  vs SDHC (1q23.3): PGL3 — head-neck paraganglioma; no imprinting; no GIST association; lower penetrance.
    SDHA-PGL5 may present with GIST (Carney-Stratakis); SDHC does not.
  vs SDHD (11q23.1): PGL1 — maternal imprinting (like SDHAF2 but NOT SDHA); head-neck PGL predominant.
    SDHD penetrance 80% vs SDHA 10%. SDHD 11q23.1 vs SDHA 5p15.33 — different chromosomes.
    SDHA: both maternal + paternal transmission penetrant; SDHD: paternal only.
  vs SURF1 (9q34.2): Leigh syndrome + CIV (cytochrome c oxidase) deficiency.
    SURF1-Leigh: CIV deficient, hairy mitochondria, elevated lactate, cerebellar + pons preferred.
    SDHA-Leigh: CII deficient, striatum + basal ganglia preferred, cardiomyopathy 25%, no CIV deficiency.
  vs POLG (15q25.3): Alpers syndrome / POLG-Leigh: multi-organ, mtDNA depletion, liver failure,
    COMPLEX I + III + IV pattern (not isolated CII). SDHA: isolated CII, no liver failure.
  vs VHL (3p25.3): Von Hippel-Lindau; hemangioblastoma + ccRCC + PCC; direct HIF1α regulation.
    SDHA: indirect HIF1α via succinate-PHD inhibition. No hemangioblastoma in SDHA. VHL: no GIST.
"""

import random
import math

SEED = 705
rng  = random.Random(SEED)

GENE         = "SDHA"
OMIM_GENE    = "600857"
OMIM_DISEASE_CII  = "252011"    # Complex II Deficiency (AR, Leigh)
OMIM_DISEASE_PGL5 = "614165"    # Paraganglioma 5 (AD)
OMIM_DISEASE_CS   = "606764"    # Carney-Stratakis Syndrome (AD)
DISEASE_NAME = (
    "SDHA Succinate Dehydrogenase Subunit A — Complex II Deficiency Leigh Syndrome (AR, "
    "OMIM #252011) + Paraganglioma 5 PGL5 (AD, OMIM #614165) + Carney-Stratakis Syndrome "
    "(AD, OMIM #606764) — Dual-Disease Gene: Biallelic Loss → Leigh / Monoallelic Loss → PGL5"
)
CHROMOSOME   = "5p15.33"
# Mixed cohort: ~60% AR Leigh/CII biallelic, ~40% AD PGL5/Carney-Stratakis monoallelic
N_PATIENTS   = 40

# ─── Variants (AR Leigh/CII — biallelic loss) ────────────────────────────────
VARIANTS_AR = [
    {
        "hgvs_c":    "c.91C>T",
        "hgvs_p":    "p.Arg31Ter",
        "domain":    "N-terminal / early coding — near-start nonsense; null allele",
        "mechanism": (
            "Most common SDHA variant causing Leigh syndrome (Parfait 2000). Arginine-to-stop at "
            "codon 31 generates a 30-amino-acid truncated peptide that lacks the entire FAD-binding "
            "domain (residues 100–400), the active site (residues 254–399), and the SDHB interface. "
            "Truncated protein non-functional, rapidly degraded by mitochondrial quality control. "
            "Complete SDHA loss → CII abolition → severe succinate accumulation → Leigh syndrome."
        ),
        "severity":  "severe",
        "phenotype": "AR_Leigh",
        "notes": "CpG hotspot. Most common Leigh-causing SDHA allele. Biallelic required (AR). Leigh MRI pattern (basal ganglia/brainstem). Cardiomyopathy risk 25%.",
    },
    {
        "hgvs_c":    "c.1664G>A",
        "hgvs_p":    "p.Gly555Glu",
        "domain":    "C-terminal domain — SDHB interface and CII holoenzyme assembly interface",
        "mechanism": (
            "Glycine-to-glutamate at position 555 in the C-terminal region introduces a charged "
            "glutamate residue into a core hydrophobic interface between SDHA and SDHB. The glycine "
            "at this position is conserved across species and is essential for SDHA-SDHB dimer "
            "formation. The glutamate side chain creates steric and electrostatic clash with the SDHB "
            "interface, preventing SDHA-SDHB core formation even if SDHA is correctly flavinylated. "
            "No SDHA-SDHB dimer → CII cannot assemble → isolated CII deficiency → Leigh."
        ),
        "severity":  "severe",
        "phenotype": "AR_Leigh",
        "notes": "SDHB-interface disruption. CII cannot assemble. Severe isolated CII deficiency. Leigh syndrome biallelic.",
    },
    {
        "hgvs_c":    "c.754G>A",
        "hgvs_p":    "p.Gly252Arg",
        "domain":    "FAD-binding domain — catalytic active site (substrate entry tunnel)",
        "mechanism": (
            "Glycine-to-arginine at position 252 in the FAD-binding domain, at the entry of the "
            "substrate tunnel where succinate accesses the FAD cofactor. The bulky arginine side chain "
            "occludes the succinate binding site and disrupts the FAD microenvironment. Even if SDHA "
            "is correctly flavinylated by SDHAF2 (FAD covalently attached at His99), the mutation "
            "prevents succinate from reaching FAD for oxidation. Catalytically null despite flavinylation. "
            "Severe CII deficiency; Leigh syndrome with prominent MRS succinate peak."
        ),
        "severity":  "severe",
        "phenotype": "AR_Leigh",
        "notes": "Active site occlusion. Catalytically null despite normal FAD attachment. Basal ganglia MRS succinate peak prominent. Leigh syndrome.",
    },
    {
        "hgvs_c":    "c.644C>T",
        "hgvs_p":    "p.Ala215Val",
        "domain":    "Protein core packing — conserved alanine in hydrophobic core",
        "mechanism": (
            "Alanine-to-valine substitution in the hydrophobic core of the FAD-binding domain. "
            "The extra methyl group of valine introduces moderate steric strain, slightly misaligning "
            "the FAD-binding loop and reducing the efficiency of SDHAF2-mediated flavinylation. "
            "Some flavinylated SDHA is produced, giving 10–25% residual CII activity. Partial "
            "CII deficiency causes milder Leigh variant — may present as subacute rather than "
            "acute Leigh, with possible longer survival and some cognitive development."
        ),
        "severity":  "moderate",
        "phenotype": "AR_Leigh",
        "notes": "Hypomorphic allele. Partial CII activity 10–25%. Milder Leigh variant, subacute course, longer survival possible.",
    },
    {
        "hgvs_c":    "c.IVS5+1G>A",
        "hgvs_p":    "p.splice_donor_intron5",
        "domain":    "Splice donor — intron 5; exon 5 contains part of the active site",
        "mechanism": (
            "Canonical splice donor site disruption at IVS5+1. Exon 5 encodes active site residues "
            "critical for succinate binding. Splice site mutation causes exon 5 skipping or activation "
            "of a downstream cryptic site, producing an out-of-frame or in-frame deletion of the "
            "active site. Resulting protein lacks functional succinate-binding capacity. "
            "Near-complete CII loss. Severe Leigh syndrome with rapid progression."
        ),
        "severity":  "severe",
        "phenotype": "AR_Leigh",
        "notes": "Null splice-site allele. Active site exon disrupted. Rapid Leigh progression. Common in compound heterozygosity with missense allele.",
    },
    {
        "hgvs_c":    "c.1608G>A",
        "hgvs_p":    "p.Trp536Ter",
        "domain":    "C-terminal — late nonsense; loss of SDHB interface and membrane-anchor binding",
        "mechanism": (
            "Tryptophan-to-stop at codon 536 produces a truncated protein retaining most of the "
            "FAD-binding domain but lacking the C-terminal SDHB-docking surface (residues 536–586). "
            "Some flavinylation may occur at His99, but the truncated SDHA cannot form the "
            "SDHA-SDHB dimer needed for CII assembly. CII holoenzyme formation blocked. "
            "Null for CII function despite partial domain retention."
        ),
        "severity":  "severe",
        "phenotype": "AR_Leigh",
        "notes": "C-terminal truncation. Partial domain retention but CII assembly-incompetent. Biallelic Leigh. Rapid early-infantile onset.",
    },
    {
        "hgvs_c":    "c.443T>C",
        "hgvs_p":    "p.Leu148Pro",
        "domain":    "FAD-binding helix — helix-breaking proline introduction",
        "mechanism": (
            "Leucine-to-proline at position 148 introduces a helix-breaking proline into an "
            "alpha-helix of the FAD-binding domain. Proline cannot participate in backbone hydrogen "
            "bonding; the helix collapses at this point, severely misfolding the FAD-binding region. "
            "SDHAF2-mediated flavinylation at His99 is compromised because the FAD-binding scaffold "
            "is disrupted. SDHA protein likely partially degraded by mitochondrial proteases. "
            "Near-complete CII loss. Classic Leigh syndrome presentation."
        ),
        "severity":  "severe",
        "phenotype": "AR_Leigh",
        "notes": "Helix-breaking proline. FAD-binding domain misfolded. Near-complete CII loss. Leigh syndrome, infantile, classic presentation.",
    },
]

# ─── Variants (AD PGL5 / Carney-Stratakis — monoallelic loss) ─────────────────
VARIANTS_AD = [
    {
        "hgvs_c":    "c.1232G>A",
        "hgvs_p":    "p.Cys411Tyr",
        "domain":    "SDHB-binding interface / iron-sulfur cluster interaction surface",
        "mechanism": (
            "Cysteine-to-tyrosine at position 411 disrupts a conserved cysteine on the SDHB-binding "
            "surface of SDHA. The cysteine may coordinate with SDHB iron-sulfur residues at the "
            "SDHA-SDHB interface. Germline heterozygous mutation → haploinsufficiency; somatic "
            "second-hit (LOH at 5p15.33) in neuroendocrine or GIST progenitor cells → complete "
            "SDHA loss → CII deficiency → succinate accumulation → PHD inhibition → HIF1α "
            "stabilization → pseudo-hypoxia → paraganglioma or SDH-deficient GIST."
        ),
        "severity":  "pathogenic_dominant",
        "phenotype": "AD_PGL5",
        "penetrance_pct": 12,
        "notes": "PGL5 / Carney-Stratakis. Not imprinted. Both parents transmit equally. IHC: SDHA null + SDHB null in tumor. Surveillance: annual imaging.",
    },
    {
        "hgvs_c":    "c.1765C>T",
        "hgvs_p":    "p.Arg589Trp",
        "domain":    "C-terminal extension — SDHB interface tail region",
        "mechanism": (
            "Arginine-to-tryptophan at the extreme C-terminus of SDHA. The C-terminal tail of "
            "SDHA makes critical contacts with SDHB to stabilize the SDHA-SDHB dimer. Disruption "
            "of the charged arginine contact by the bulky hydrophobic tryptophan reduces SDHA-SDHB "
            "binding affinity. Heterozygous: partial CII reduction; somatic LOH → complete CII "
            "loss. Associated with Carney-Stratakis syndrome (PGL + GIST dyad) in reported families."
        ),
        "severity":  "pathogenic_dominant",
        "phenotype": "AD_PGL5_CS",
        "penetrance_pct": 14,
        "notes": "PGL5 + Carney-Stratakis (GIST association). Not imprinted. SDHA IHC null pattern. Young-onset gastric multifocal GIST + head-neck PGL.",
    },
    {
        "hgvs_c":    "c.232G>C",
        "hgvs_p":    "p.Gly78Arg",
        "domain":    "FAD-binding domain — near His99 flavinylation site",
        "mechanism": (
            "Glycine-to-arginine at position 78, adjacent to the His99 FAD-attachment site. "
            "The arginine side chain may sterically interfere with SDHAF2-mediated flavinylation "
            "at His99 in some cellular contexts, or reduce SDHA stability. Monoallelic in PGL5 "
            "families: heterozygous haploinsufficiency insufficient to cause Leigh (biallelic "
            "needed) but sufficient for tumor-suppressor loss after somatic second-hit in "
            "chromaffin/paraganglionic cells. Different disease from biallelic Gly78 mutations."
        ),
        "severity":  "pathogenic_dominant",
        "phenotype": "AD_PGL5",
        "penetrance_pct": 9,
        "notes": "PGL5. Low penetrance. Near His99 FAD site. Monoallelic: PGL5, not Leigh. Biallelic (if compound het with severe allele): Leigh possible.",
    },
]

# ─── Combined variant pool (used for cohort generation) ─────────────────────
ALL_VARIANTS_AR = VARIANTS_AR   # 7 AR alleles
ALL_VARIANTS_AD = VARIANTS_AD   # 3 AD alleles

# ─── Clinical features by phenotype ──────────────────────────────────────────
LEIGH_FEATURES = [
    {"feature": "Bilateral basal ganglia lesions (MRI T2-bright)", "freq_pct": 80},
    {"feature": "Brainstem lesions (periaqueductal)",              "freq_pct": 55},
    {"feature": "Developmental regression",                        "freq_pct": 90},
    {"feature": "Psychomotor retardation",                         "freq_pct": 88},
    {"feature": "Hypotonia",                                       "freq_pct": 82},
    {"feature": "Lactic acidosis (serum)",                         "freq_pct": 70},
    {"feature": "Brain MRS succinate peak elevated",               "freq_pct": 75},
    {"feature": "Cardiomyopathy (HCM/DCM)",                       "freq_pct": 25},
    {"feature": "Respiratory failure / central apnea",             "freq_pct": 45},
    {"feature": "Seizures (often later in course)",                "freq_pct": 40},
    {"feature": "Optic atrophy",                                   "freq_pct": 22},
    {"feature": "Isolated CII deficiency (CIII/CIV normal)",       "freq_pct": 95},
]

PGL5_FEATURES = [
    {"feature": "Head-neck paraganglioma (HNPGL)",                "freq_pct": 65},
    {"feature": "Carotid body tumor",                             "freq_pct": 50},
    {"feature": "Jugulotympanic paraganglioma",                   "freq_pct": 28},
    {"feature": "Adrenal pheochromocytoma (PCC)",                 "freq_pct": 20},
    {"feature": "SDH-deficient GIST (Carney-Stratakis subset)",   "freq_pct": 15},
    {"feature": "Bilateral / multicentric PGL",                   "freq_pct": 18},
    {"feature": "Malignant transformation",                       "freq_pct":  5},
    {"feature": "Catecholamine excess (PCC/secretory PGL)",       "freq_pct": 18},
    {"feature": "Neck mass / pulsatile tinnitus",                 "freq_pct": 58},
    {"feature": "Cranial nerve palsy (IX-XII)",                   "freq_pct": 22},
    {"feature": "Normotensive (head-neck PGL, non-secretory)",    "freq_pct": 72},
    {"feature": "SDHA null on IHC (tumor)",                       "freq_pct": 95},
]

def _pick_weighted(choices, weights):
    total = sum(weights)
    r = rng.uniform(0, total)
    cumulative = 0
    for c, w in zip(choices, weights):
        cumulative += w
        if r < cumulative:
            return c
    return choices[-1]

def _make_ar_patient(i):
    """Generate one AR Leigh/CII patient (biallelic SDHA)."""
    rng.seed(SEED + i * 7 + 1)
    age_onset_mo = rng.randint(3, 18)
    # Pick two alleles (compound het or homozygous)
    allele1 = rng.choice(ALL_VARIANTS_AR)
    allele2 = rng.choice(ALL_VARIANTS_AR)
    sex = rng.choice(["M", "F"])
    cii_residual = round(rng.uniform(5, 28), 1)
    has_cardio = rng.random() < 0.25
    leigh_mri = rng.random() < 0.80
    mrs_succinate = rng.random() < 0.75
    lactic = round(rng.uniform(2.5, 8.0), 1)
    survived_y = round(rng.uniform(1.2, 7.0), 1)

    return {
        "patient_id": f"SDHA-AR-{i+1:03d}",
        "phenotype": "AR_CII_Leigh",
        "sex": sex,
        "age_onset_months": age_onset_mo,
        "allele_1": allele1["hgvs_p"],
        "allele_2": allele2["hgvs_p"],
        "allele_1_origin": "maternal",
        "allele_2_origin": "paternal",
        "cii_residual_pct": cii_residual,
        "ci_ciii_civ_normal": True,
        "leigh_mri": leigh_mri,
        "mrs_succinate_elevated": mrs_succinate,
        "serum_lactate_mmol_L": lactic,
        "cardiomyopathy": has_cardio,
        "survived_years": survived_y,
        "outcome": "deceased" if survived_y < 3 else "severely_affected",
    }

def _make_ad_patient(i):
    """Generate one AD PGL5 / Carney-Stratakis patient (monoallelic SDHA)."""
    rng.seed(SEED + i * 11 + 500)
    age_dx = rng.randint(22, 58)
    allele = rng.choice(ALL_VARIANTS_AD)
    sex = rng.choice(["M", "F"])
    has_gist = (allele["phenotype"] == "AD_PGL5_CS") or (rng.random() < 0.15)
    pgl_site = _pick_weighted(
        ["Head-neck PGL", "Adrenal PCC", "Retroperitoneal PGL", "Multiple sites"],
        [65, 20, 8, 7]
    )
    malignant = rng.random() < 0.05
    bilateral = rng.random() < 0.18

    return {
        "patient_id": f"SDHA-AD-{i+1:03d}",
        "phenotype": "AD_PGL5",
        "sex": sex,
        "age_at_diagnosis_years": age_dx,
        "germline_variant": allele["hgvs_p"],
        "transmission": "biparental_possible",  # NOT imprinted
        "pgl_site": pgl_site,
        "has_gist": has_gist,
        "malignant": malignant,
        "bilateral_pgl": bilateral,
        "sdha_ihc_tumor": "null",
        "sdhb_ihc_tumor": "null",  # SDHA loss causes secondary SDHB loss on IHC
        "penetrance_cohort_pct": round(allele.get("penetrance_pct", 10), 1),
    }

# ─── get_overview ─────────────────────────────────────────────────────────────
def get_overview() -> dict:
    rng.seed(SEED)

    n_ar = 24   # ~60% AR Leigh/CII
    n_ad = 16   # ~40% AD PGL5

    ar_patients = [_make_ar_patient(i) for i in range(n_ar)]
    ad_patients = [_make_ad_patient(i) for i in range(n_ad)]

    # AR cohort stats
    ar_cii_residuals = [p["cii_residual_pct"] for p in ar_patients]
    ar_onset_months  = [p["age_onset_months"] for p in ar_patients]
    ar_cardio_n      = sum(1 for p in ar_patients if p["cardiomyopathy"])
    ar_leigh_n       = sum(1 for p in ar_patients if p["leigh_mri"])
    ar_mrs_n         = sum(1 for p in ar_patients if p["mrs_succinate_elevated"])

    # AD cohort stats
    ad_gist_n        = sum(1 for p in ad_patients if p["has_gist"])
    ad_malignant_n   = sum(1 for p in ad_patients if p["malignant"])
    ad_bilateral_n   = sum(1 for p in ad_patients if p["bilateral_pgl"])

    return {
        "gene":        GENE,
        "omim_gene":   OMIM_GENE,
        "omim_disease_cii":  OMIM_DISEASE_CII,
        "omim_disease_pgl5": OMIM_DISEASE_PGL5,
        "omim_disease_cs":   OMIM_DISEASE_CS,
        "disease_name": DISEASE_NAME,
        "chromosome":  CHROMOSOME,
        "protein":     "Succinate dehydrogenase subunit A (flavoprotein catalytic subunit)",
        "protein_size": "586 aa, ~70 kDa",
        "location":    "Mitochondrial matrix (no transmembrane helices)",
        "fad_site":    "His99 (covalent FAD attachment by SDHAF2)",
        "n_patients":  N_PATIENTS,
        "seed":        SEED,
        "cohort_note": (
            "Dual-phenotype cohort: 24 AR Leigh/CII deficiency (biallelic SDHA) + "
            "16 AD PGL5/Carney-Stratakis (monoallelic SDHA, somatic second-hit in tumor). "
            "Reflects the dual-disease biology of SDHA as both metabolic enzyme and tumor suppressor."
        ),
        "cohort_summary": f"{N_PATIENTS} patients (24 AR Leigh, 16 AD PGL5), seed {SEED}",

        # AR Leigh summary
        "ar_summary": {
            "n":                    n_ar,
            "inheritance":          "AR (biallelic, compound heterozygous or homozygous)",
            "mean_onset_months":    round(sum(ar_onset_months) / n_ar, 1),
            "range_onset_months":   [min(ar_onset_months), max(ar_onset_months)],
            "mean_cii_residual_pct": round(sum(ar_cii_residuals) / n_ar, 1),
            "leigh_mri_pct":        round(100 * ar_leigh_n / n_ar, 1),
            "mrs_succinate_pct":    round(100 * ar_mrs_n / n_ar, 1),
            "cardiomyopathy_pct":   round(100 * ar_cardio_n / n_ar, 1),
            "ci_ciii_civ_normal":   "100% (isolated CII deficiency fingerprint)",
        },

        # AD PGL5 summary
        "ad_summary": {
            "n":              n_ad,
            "inheritance":    "AD (monoallelic, NOT imprinted — biparental transmission)",
            "penetrance_pct": 10,
            "gist_pct":       round(100 * ad_gist_n / n_ad, 1),
            "malignant_pct":  round(100 * ad_malignant_n / n_ad, 1),
            "bilateral_pct":  round(100 * ad_bilateral_n / n_ad, 1),
            "ihc_pattern":    "SDHA null + SDHB null (unique to SDHA loss; SDHB/C/D loss = SDHB null only)",
        },

        "key_facts": [
            "SDHA is the ONLY SDH gene causing BOTH AR Leigh syndrome (biallelic) AND AD paraganglioma (monoallelic)",
            "5p15.33 — unique chromosome; all other SDH subunits on chromosomes 1, 11, 19",
            "PGL5 is NOT maternally imprinted (unlike SDHD-PGL1 and SDHAF2-PGL2)",
            "IHC: SDHA null + SDHB null — only SDHA loss shows dual-null pattern",
            "FAD attached at His99 by SDHAF2 — SDHA mutations near His99 impair flavinylation",
            "KD ABSOLUTELY CONTRAINDICATED in CII/Leigh: FADH2 enters ETC only via deficient CII",
            "Cardiomyopathy 25% in AR Leigh — distinguishes from SDHAF1 (leukoencephalopathy, rare HCM)",
            "Carney-Stratakis: multifocal gastric GIST (KIT/PDGFRA WT) + PGL in young patients",
            "Annual PGL surveillance mandatory for AD carriers (MRI/CT head-neck, DOTATATE PET-CT)",
        ],

        "ar_patients": ar_patients,
        "ad_patients": ad_patients,
    }


# ─── get_breakdown ────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    rng.seed(SEED + 1000)

    # Variant breakdown (both AR and AD)
    all_variants = []
    for v in ALL_VARIANTS_AR:
        all_variants.append({
            "hgvs_c":        v["hgvs_c"],
            "hgvs_p":        v["hgvs_p"],
            "domain":        v["domain"],
            "severity":      v["severity"],
            "phenotype":     "AR Leigh / CII deficiency (biallelic)",
            "mechanism_short": v["mechanism"][:200] + "…",
            "notes":         v["notes"],
        })
    for v in ALL_VARIANTS_AD:
        all_variants.append({
            "hgvs_c":        v["hgvs_c"],
            "hgvs_p":        v["hgvs_p"],
            "domain":        v["domain"],
            "severity":      v["severity"],
            "phenotype":     f"AD PGL5 (monoallelic, penetrance ~{v.get('penetrance_pct',10)}%)",
            "mechanism_short": v["mechanism"][:200] + "…",
            "notes":         v["notes"],
        })

    # Clinical features by phenotype
    leigh_features_out = [
        {
            "feature": f["feature"],
            "freq_pct": f["freq_pct"],
            "phenotype": "AR_Leigh",
        }
        for f in LEIGH_FEATURES
    ]
    pgl5_features_out = [
        {
            "feature": f["feature"],
            "freq_pct": f["freq_pct"],
            "phenotype": "AD_PGL5",
        }
        for f in PGL5_FEATURES
    ]

    # DDx table
    ddx_table = [
        {
            "gene":        "SDHAF1",
            "locus":       "19q13.12",
            "disease":     "CII deficiency — infantile leukoencephalopathy (not Leigh)",
            "key_ddx":     "SDHAF1: WHITE MATTER leukoencephalopathy; SDHA-AR: GRAY MATTER Leigh. KD CI both. No cardiomyopathy (SDHAF1) vs 25% HCM (SDHA).",
            "malignancy":  "None",
            "imprinting":  "None (AR recessive)",
        },
        {
            "gene":        "SDHAF2",
            "locus":       "11q13.1",
            "disease":     "PGL2 — Paraganglioma 2 (SDHA flavinylation factor)",
            "key_ddx":     "SDHAF2: MATERNAL IMPRINTING (paternal-only, penetrance 85–92%). SDHA-PGL5: NOT imprinted (biparental, penetrance 10%). SDHAF2 does NOT cause Leigh. SDHA DOES cause Leigh.",
            "malignancy":  "5%",
            "imprinting":  "YES (maternal imprinting) — paternal transmission only",
        },
        {
            "gene":        "SDHB",
            "locus":       "1p36.13",
            "disease":     "PGL4 — highest malignancy SDH locus",
            "key_ddx":     "SDHB: malignancy 20–50% (highest). SDHA-PGL5: malignancy 5%. IHC: SDHB loss only (SDHA proficient). SDHA IHC null + SDHB null (SDHA loss abolishes SDHB stability).",
            "malignancy":  "20–50% (highest)",
            "imprinting":  "None",
        },
        {
            "gene":        "SDHC",
            "locus":       "1q23.3",
            "disease":     "PGL3 — head-neck PGL, low malignancy",
            "key_ddx":     "SDHC: no GIST association (Carney-Stratakis rare). SDHA-PGL5: GIST in 15% (Carney-Stratakis). No imprinting either. IHC: SDHB null only (SDHA proficient).",
            "malignancy":  "1–3%",
            "imprinting":  "None",
        },
        {
            "gene":        "SDHD",
            "locus":       "11q23.1",
            "disease":     "PGL1 — maternal imprinting, head-neck predominant",
            "key_ddx":     "SDHD: MATERNAL IMPRINTING (paternal-only), penetrance 80%. SDHA-PGL5: NOT imprinted. SDHD more common hereditary head-neck PGL. Both 11q but 10 Mb apart (SDHAF2 11q13 vs SDHD 11q23).",
            "malignancy":  "3–5%",
            "imprinting":  "YES (maternal) — paternal only",
        },
        {
            "gene":        "SURF1",
            "locus":       "9q34.2",
            "disease":     "Leigh syndrome + COX (CIV) deficiency",
            "key_ddx":     "SURF1-Leigh: CIV (COX) deficiency; hairy mitochondria EM; cerebellar preference; elevated lactate. SDHA-AR-Leigh: CII deficiency; basal ganglia/striatum; succinate MRS peak; cardiomyopathy 25%. CIV vs CII is key biochemical distinguisher.",
            "malignancy":  "None",
            "imprinting":  "None (AR)",
        },
        {
            "gene":        "POLG",
            "locus":       "15q25.3",
            "disease":     "Alpers syndrome / POLG-Leigh; mtDNA depletion",
            "key_ddx":     "POLG: multi-complex deficiency (CI+CIII+CIV pattern, not isolated CII); hepatopathy; mtDNA depletion; VPA CI for both (different reason). SDHA: isolated CII, no liver disease.",
            "malignancy":  "None",
            "imprinting":  "None (AR)",
        },
        {
            "gene":        "VHL",
            "locus":       "3p25.3",
            "disease":     "Von Hippel-Lindau — hemangioblastoma + ccRCC + PCC",
            "key_ddx":     "VHL: hemangioblastoma (cerebellum/spine/retina) — absent in SDHA. VHL: ccRCC. SDHA-PGL5: GIST (not RCC). VHL: direct HIF1α suppressor. SDHA: indirect via succinate-PHD inhibition.",
            "malignancy":  "VHL ccRCC 70%",
            "imprinting":  "None (AD, LOH)",
        },
    ]

    # Treatment protocols (separated by phenotype)
    treatment_cii_leigh = {
        "phenotype": "AR CII Deficiency / Leigh Syndrome",
        "absolute_contraindications": [
            {
                "drug":   "Ketogenic diet (KD)",
                "reason": "ABSOLUTE CONTRAINDICATION. FADH2 enters ETC exclusively via CII (deficient in SDHA-null). KD markedly increases fatty acid β-oxidation → massive FADH2 production → all FADH2 blocked at deficient CII → ETC collapse → severe metabolic crisis. Unique to CII deficiency.",
            },
            {
                "drug":   "Valproate (VPA)",
                "reason": "ABSOLUTE CI. Multiple mechanisms: CoA sequestration (inhibits CoA-dependent β-oxidation); mitochondrial toxicity (ETC inhibition); mtDNA depletion risk. Use LEV/lacosamide instead.",
            },
            {
                "drug":   "Metformin",
                "reason": "ABSOLUTE CI. Direct CII inhibitor (inhibits succinate dehydrogenase activity). In CII deficiency context, further reduces the residual 5–30% CII activity.",
            },
            {
                "drug":   "Linezolid",
                "reason": "ABSOLUTE CI. Inhibits mitochondrial ribosome (23S rRNA); blocks translation of mtDNA-encoded subunits. Exacerbates energy failure.",
            },
            {
                "drug":   "Chloramphenicol",
                "reason": "ABSOLUTE CI. Same 23S rRNA ribosome inhibition mechanism as linezolid.",
            },
            {
                "drug":   "Propofol (PRIS risk)",
                "reason": "AVOID. Propofol infusion syndrome (PRIS): inhibits ETC CII and CIV, disrupts fatty acid oxidation — catastrophic in CII-deficient patients. Use sevoflurane for anesthesia.",
            },
        ],
        "recommended_treatments": [
            {"drug": "Riboflavin (B2)", "dose": "100–300 mg/day", "level": "C",
             "rationale": "SDHA contains FAD — riboflavin supplementation may modestly augment residual CII activity in hypomorphic variants. Level C; monitor response. Unlike SDHAF1 (no FAD domain), SDHA HAS FAD — riboflavin theoretically relevant."},
            {"drug": "CoQ10 (Ubiquinol)", "dose": "10–30 mg/kg/day", "level": "C",
             "rationale": "Antioxidant; supports ETC electron transfer distal to CII blockade."},
            {"drug": "Thiamine (B1)", "dose": "100–300 mg/day", "level": "C — MANDATORY EMPIRIC",
             "rationale": "Empiric SLC19A3 / BTD (thiamine transporter deficiency / biotinidase) — must rule out before attributing to CII deficiency. Low risk. MANDATORY until excluded."},
            {"drug": "Biotin", "dose": "5–20 mg/day", "level": "C — MANDATORY EMPIRIC",
             "rationale": "Empiric BTD / HLCS — biotinidase and holocarboxylase synthetase deficiency can mimic Leigh. MANDATORY empiric treatment until excluded."},
            {"drug": "L-Carnitine", "dose": "50–100 mg/kg/day", "level": "C",
             "rationale": "Supports fatty acid metabolism; replenishes CoA pool (partially depleted in mitochondrial disease)."},
            {"drug": "Levetiracetam (LEV)", "dose": "20–60 mg/kg/day", "level": "A (for seizures)",
             "rationale": "Preferred AED: renal metabolism, no mitochondrial toxicity, no CoA interaction."},
            {"drug": "Dichloroacetate (DCA)", "dose": "12.5–25 mg/kg/day", "level": "C",
             "rationale": "Activates PDH → reduces pyruvate/lactate. Off-label. Peripheral neuropathy risk limits long-term use."},
        ],
        "supportive": [
            "IV dextrose / GIR 6–8 mg/kg/min during acute decompensation — NEVER fast",
            "Sevoflurane (not propofol) for anesthesia",
            "Cardiology surveillance: echocardiography every 6–12 months (HCM/DCM risk 25%)",
            "Cochlear implants: limited role (SDHA-Leigh has hearing loss in ~20%)",
            "Gastrostomy tube for feeding support in severe cases",
        ],
    }

    treatment_pgl5 = {
        "phenotype": "AD PGL5 / Carney-Stratakis",
        "absolute_contraindications": [
            {
                "drug":   "Alpha-blockade OMITTED before surgery",
                "reason": "CRITICAL SEQUENCE: For adrenal PCC or secretory PGL — alpha-blockade (phenoxybenzamine) MUST precede beta-blockade by ≥7–14 days pre-op. Reversing order → unopposed alpha vasoconstriction → hypertensive crisis during surgery.",
            },
        ],
        "recommended_treatments": [
            {"drug": "Surgical resection (PGL/PCC)", "dose": "N/A — first-line",  "level": "A",
             "rationale": "Complete surgical excision curative for localized PGL/PCC. Adrenalectomy for PCC (laparoscopic preferred if feasible)."},
            {"drug": "Phenoxybenzamine (alpha-blocker)", "dose": "10–40 mg/day titrated", "level": "A — pre-op",
             "rationale": "Pre-operative alpha-blockade for PCC/secretory PGL. Start ≥7–14 days pre-op."},
            {"drug": "Belzutifan (PT2977)", "dose": "120 mg/day", "level": "B — emerging",
             "rationale": "HIF2α inhibitor — FDA approved for VHL disease; emerging use in SDH-deficient unresectable PGL/PCC and GIST (succinate-PHD-HIF2α pathway). Clinical trials ongoing for SDHA."},
            {"drug": "177Lu-DOTATATE (PRRT)", "dose": "Somatostatin receptor-guided", "level": "B",
             "rationale": "For SSTR2-positive inoperable/metastatic PGL/PCC. DOTATATE PET-CT first to confirm SSTR expression."},
            {"drug": "Imatinib / sunitinib (GIST)", "dose": "400 mg/day imatinib", "level": "B",
             "rationale": "SDH-deficient GIST (Carney-Stratakis) often LESS responsive to imatinib than KIT-mutant GIST, but partial responses reported. Sunitinib second-line. Discuss with GIST multidisciplinary team."},
        ],
        "surveillance": [
            "Annual MRI/CT head-neck for HNPGL in at-risk carriers",
            "Annual DOTATATE PET-CT for systemic PGL/PCC surveillance",
            "24-hour urine/plasma metanephrines/catecholamines annually",
            "Upper GI endoscopy / CT abdomen every 1–2 years (GIST surveillance in Carney-Stratakis)",
            "IHC SDHA + SDHB on all resected tumors — SDHA-null confirms germline relevance",
            "Cascade genetic testing of first-degree relatives (biparental — both parents transmit)",
        ],
    }

    return {
        "gene":             GENE,
        "omim_gene":        OMIM_GENE,
        "chromosome":       CHROMOSOME,
        "variant_breakdown": all_variants,
        "n_ar_variants":    len(ALL_VARIANTS_AR),
        "n_ad_variants":    len(ALL_VARIANTS_AD),
        "leigh_features":   leigh_features_out,
        "pgl5_features":    pgl5_features_out,
        "ddx_table":        ddx_table,
        "treatment_cii_leigh": treatment_cii_leigh,
        "treatment_pgl5":   treatment_pgl5,
        "pathway_context": {
            "cii_assembly_sequence": [
                "Step 1: SDHAF2 → covalent FAD attachment to SDHA His99 (flavinylation)",
                "Step 2: SDHAF1 → SDHB FeS cluster maturation (via HSC20/HSPA9)",
                "Step 3: Flavinylated SDHA + FeS-matured SDHB → SDHA-SDHB catalytic dimer",
                "Step 4: SDHA-SDHB + SDHC-SDHD membrane anchor → CII holoenzyme",
                "SDHA = the catalytic target of both assembly steps 1-2 (SDHA flavinylated; SDHB delivers electrons to ubiquinone through SDHC/D)",
            ],
            "pseudohypoxia_pathway": (
                "CII loss → succinate accumulates in matrix → succinate exits to cytoplasm → "
                "succinate inhibits PHD enzymes (prolyl hydroxylases) → HIF1α/HIF2α NOT hydroxylated "
                "→ NOT degraded by VHL → stabilized → transcription of VEGF, EPO, angiogenic genes → "
                "pseudo-hypoxic tumor microenvironment → paraganglioma / GIST"
            ),
        },
    }


# ─── get_definitions ─────────────────────────────────────────────────────────
def get_definitions() -> dict:
    return {
        "gene": {
            "name":        GENE,
            "full_name":   "Succinate Dehydrogenase Subunit A (Flavoprotein Catalytic Subunit)",
            "omim_gene":   OMIM_GENE,
            "chromosome":  CHROMOSOME,
            "size_aa":     586,
            "size_kda":    70,
            "domains":     [
                "FAD-binding domain (residues 100–400): covalent FAD at His99; succinate active site (254–399)",
                "SDHB-interface domain (C-terminal 450–586): critical for SDHA-SDHB dimer formation",
                "N-terminal mitochondrial targeting sequence (1–50): cleaved after import",
                "Substrate tunnel: succinate entry from solvent to FAD active site",
            ],
            "cofactor":    "FAD (covalently attached at His99 by SDHAF2)",
            "function":    "Succinate oxidation (succinate + FAD → fumarate + FADH2); ubiquinone reduction via SDHB",
            "assembly":    "SDHA is the first subunit flavinylated; then binds SDHB; then SDHC-SDHD anchors in IMM",
        },
        "diseases": {
            "cii_deficiency_leigh": {
                "omim":        OMIM_DISEASE_CII,
                "name":        "Complex II Deficiency (Leigh syndrome variant) — CI2DN1",
                "inheritance": "AR (autosomal recessive, biallelic loss-of-function)",
                "onset":       "Infantile (3–18 months); rarely congenital or juvenile",
                "mri":         "Bilateral symmetric basal ganglia + brainstem (Leigh pattern, 80%)",
                "biochemistry": "Isolated CII deficiency 5–30%; CI/CIII/CIV strictly NORMAL (biochemical fingerprint)",
                "mrs":         "Elevated succinate peak on brain MRS (~75%) — pathognomonic for CII deficiency",
                "cardiomyopathy": "HCM or DCM in 25% — distinguishes from SDHAF1 (rare HCM)",
                "prognosis":   "Severe; most die in early childhood (1–7 years); rare survivors with milder variants",
                "kd":          "ABSOLUTE CONTRAINDICATION — FADH2 enters ETC only via deficient CII",
            },
            "pgl5": {
                "omim":        OMIM_DISEASE_PGL5,
                "name":        "Paraganglioma 5 (PGL5)",
                "inheritance": "AD (autosomal dominant); NOT imprinted — biparental transmission",
                "penetrance":  "~10–15% (low; much lower than SDHD 80% or SDHAF2 85–92%)",
                "sites":       "Head-neck PGL 65%, adrenal PCC 20%, retroperitoneal PGL 8%, multiple sites 7%",
                "malignancy":  "5% (low — similar to SDHAF2; much lower than SDHB 20–50%)",
                "ihc_pattern": "SDHA null + SDHB null (UNIQUE — only SDHA loss causes dual SDHA/SDHB null IHC)",
                "surveillance": "Annual head-neck MRI + DOTATATE PET-CT; plasma/urine metanephrines",
                "not_imprinted": "CRITICAL: unlike SDHD/PGL1 and SDHAF2/PGL2 — both maternal AND paternal SDHA mutations penetrant",
            },
            "carney_stratakis": {
                "omim":        OMIM_DISEASE_CS,
                "name":        "Carney-Stratakis Syndrome (paraganglioma-GIST dyad)",
                "inheritance": "AD, incomplete penetrance",
                "gist":        "SDH-deficient GIST: multifocal, gastric, KIT/PDGFRA wild-type, young onset",
                "gist_ddx":    "SDH-deficient GIST vs sporadic GIST (KIT/PDGFRA mutant, single lesion, older)",
                "treatment":   "Less imatinib-responsive than KIT-mutant GIST; sunitinib partial response",
                "ihc":         "SDHA null + SDHB null on GIST biopsy confirms SDH-deficient pathology",
            },
        },
        "imprinting_comparison": {
            "sdha_pgl5":  "NOT IMPRINTED — biparental; maternal and paternal mutations equally penetrant (~10%)",
            "sdhd_pgl1":  "MATERNALLY IMPRINTED — paternal transmission only; penetrance ~80%",
            "sdhaf2_pgl2": "MATERNALLY IMPRINTED — paternal transmission only; penetrance ~85–92%",
            "sdhb_pgl4":  "NOT IMPRINTED — biparental; penetrance variable; malignancy 20–50%",
            "sdhc_pgl3":  "NOT IMPRINTED — biparental; penetrance lower; head-neck predominant",
        },
        "ihc_interpretation": {
            "sdha_loss":  "SDHA null + SDHB null on IHC → SDHA mutation (germline or somatic)",
            "sdhb_loss_only": "SDHB null, SDHA proficient → SDHB, SDHC, or SDHD mutation (NOT SDHA)",
            "rationale":  "SDHA protein is required to stabilize SDHB in assembled CII. Loss of SDHA → secondary SDHB loss on IHC. Other subunit losses do not affect SDHA stability.",
            "clinical_use": "Order SDHA + SDHB IHC on all resected PGL/PCC and SDH-deficient GIST. SDHA null → sequence SDHA germline.",
        },
        "pathway": {
            "sdha_sdhaf2_link": (
                "SDHAF2 covalently attaches FAD to SDHA His99 (flavinylation). "
                "Mutations in SDHA near His99 may impair SDHAF2-mediated flavinylation "
                "even when SDHAF2 is wild-type. Functional assay (FAD attachment + CII "
                "activity) distinguishes SDHA-intrinsic from SDHAF2-dependent defects."
            ),
            "sdha_sdhaf1_link": (
                "SDHAF1 delivers FeS clusters to SDHB (not SDHA). Once SDHA is flavinylated "
                "(SDHAF2 step) and SDHB FeS-matured (SDHAF1 step), SDHA-SDHB dimer forms. "
                "SDHA mutations: fail at catalytic step. SDHAF1 mutations: SDHB FeS delivery fails, "
                "but SDHA itself can be correctly flavinylated."
            ),
        },
        "key_references": [
            {
                "citation": "Parfait B et al. (2000) Compound heterozygous mutations in the flavoprotein gene of the respiratory chain complex II in a patient with Leigh syndrome. Hum Genet 106(2):236–243.",
                "relevance": "First SDHA biallelic mutations in Leigh syndrome / CII deficiency (patient with compound heterozygous SDHA mutations)",
            },
            {
                "citation": "Burnichon N et al. (2010) SDHA is a tumor suppressor gene causing paraganglioma. Hum Mol Genet 19(15):3011–3020.",
                "relevance": "First SDHA germline mutations in PGL5; established SDHA as tumor suppressor; Burnichon series",
            },
            {
                "citation": "Janeway KA et al. (2011) Defects in succinate dehydrogenase in gastrointestinal stromal tumors lacking KIT and PDGFRA mutations. Science 331(6014):220–223.",
                "relevance": "SDHA-deficient GIST in Carney-Stratakis syndrome; SDH-deficient GIST pathomechanism; IHC pattern",
            },
            {
                "citation": "Dwight T et al. (2013) SDHA germline mutations rare in phaeochromocytoma and paraganglioma. Clin Endocrinol 78(4):515–519.",
                "relevance": "SDHA mutation frequency and low penetrance (~10%) confirmed in PGL5 series; surveillance implications",
            },
            {
                "citation": "Astuti D et al. (2001) Gene mutations in the succinate dehydrogenase subunit SDHB cause susceptibility to familial phaeochromocytoma and to familial paraganglioma. Am J Hum Genet 69(1):49–54.",
                "relevance": "SDHB-PGL4 first report — critical DDx for SDHA-PGL5; malignancy comparison (SDHB 20–50% vs SDHA 5%)",
            },
        ],
        "monitoring_protocol": {
            "AR_CII_Leigh": {
                "mre_brain":   "Every 6–12 months in active disease; baseline at diagnosis",
                "echo":        "Every 6–12 months (HCM/DCM risk 25%)",
                "metabolic":   "Plasma lactate, amino acids, urine organic acids at each visit",
                "EEG":         "If seizures develop (40%); baseline when stable",
                "ophthalmology": "Annual (optic atrophy 22%)",
                "genetics":    "Cascade testing of parents; prenatal/preimplantation available",
            },
            "AD_PGL5": {
                "biochemical": "Annual plasma/urine metanephrines + normetanephrines + catecholamines",
                "imaging":     "Annual MRI/CT head-neck; DOTATATE PET-CT every 1–2 years",
                "gist":        "Upper GI endoscopy + CT abdomen/pelvis every 1–2 years (Carney-Stratakis risk)",
                "genetics":    "Cascade testing biparental (both parents transmit; maternal AND paternal penetrant)",
                "ihc":         "SDHA + SDHB IHC on all resected tumors",
                "start_age":   "Surveillance begins at age 6–8 years (childhood GIST reported in Carney-Stratakis)",
            },
        },
    }


if __name__ == "__main__":
    import json
    print("=== SDHA OVERVIEW ===")
    ov = get_overview()
    print(f"Gene: {ov['gene']}, OMIM Gene: {ov['omim_gene']}")
    print(f"Patients: {ov['n_patients']}, Seed: {ov['seed']}")
    print(f"AR Leigh: {ov['ar_summary']['n']}, AD PGL5: {ov['ad_summary']['n']}")
    print(f"Cohort: {ov['cohort_summary']}")
    print("\n=== BREAKDOWN ===")
    bd = get_breakdown()
    print(f"Variants: {len(bd['variant_breakdown'])} ({bd['n_ar_variants']} AR + {bd['n_ad_variants']} AD)")
    print(f"DDx entries: {len(bd['ddx_table'])}")
    print("\n=== DEFINITIONS (keys) ===")
    df = get_definitions()
    print(list(df.keys()))
    print("\n✅ SDHA dashboard OK")
