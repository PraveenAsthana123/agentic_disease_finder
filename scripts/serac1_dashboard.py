#!/usr/bin/env python3
"""SERAC1 MEGDEL Syndrome Dashboard.

MEGDEL = 3-Methylglutaconic aciduria + Deafness (SNHL) + Encephalopathy + Leigh-Like

SERAC1 is a phosphatidylglycerol remodeling enzyme at the mitochondria-associated ER membrane (MAM).
LOF → BMP (bis(monoacylglycero)phosphate) deficiency → Complex I/IV dysfunction → OXPHOS failure
→ 3-MGA-uria (overflow) + mitochondrial encephalopathy + SNHL + Leigh-like MRI

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. 3-MGA-uria Type V (MGA-V) — elevated 3-methylglutaconic acid, same overflow pathway as OPA3/DNAJC19
  2. SNHL (sensorineural hearing loss) 100% — CARDINAL; early, often profound by age 2; UNIQUE among 3-MGA-uria
  3. Leigh-like MRI: bilateral basal ganglia (putamen > caudate), brainstem — T2 hyperintensity
  4. NO GP iron on MRI: KEY DDx from MECR/MEPAN (GP iron bilateral + SWI) and NBIA series
  5. NO dilated cardiomyopathy: KEY DDx from DNAJC19-DCMA (DCM 100%) and Barth (TAZ)
  6. NO optic atrophy: KEY DDx from OPA3 (100%) and MECR (80-90%)
  7. NO neutropenia + NO elevated C4-DC: KEY DDx from Barth (TAZ) where neutropenia 95% + C4-DC elevated
  8. Neonatal liver dysfunction (60-70%): cholestasis + hepatomegaly; usually transient; rare fulminant
  9. Cochlear implant highly effective for SNHL (Level B): hearing restoration transforms developmental trajectory
 10. VPA: MODERATE CAUTION (mito Complex I depression; NOT absolute CI like MECR; monitor LFTs + NH3)
 11. LEV preferred for seizures (renal; no mito toxicity)
 12. 6q22.1 — AR, no dominant founder (unlike DNAJC19 Hutterite c.130-1G>C)

SERAC1 BIOLOGY:
SERAC1 (490 amino acids, 6q22.1) encodes Serine Active Site Containing 1, a phospholipid-remodeling
enzyme localised to the mitochondria-associated ER membrane (MAM) and to mitochondria.
SERAC1 catalyses the reacylation of 1-acyl-2-lyso-phosphatidylglycerol (lyso-PG) to produce
bis(monoacylglycero)phosphate (BMP) and mature phosphatidylglycerol (PG) — critical inner
mitochondrial membrane phospholipids that stabilise Complex I and Complex IV assembly.

SERAC1 also participates in cholesterol trafficking: at the MAM, SERAC1 maintains cholesterol
homeostasis between ER and mitochondrial membranes, supporting mitochondrial membrane integrity
and cristae structure.

LOF mechanism:
  SERAC1 LOF → BMP deficiency in inner mitochondrial membrane →
    Complex I (NADH:ubiquinone oxidoreductase) — disassembly → NADH oxidation failure → lactate↑
    Complex IV (cytochrome c oxidase) — instability → respiratory chain impaired
  → OXPHOS dysfunction → 3-MGA-uria (HMG-CoA pathway overflow, same as OPA3/DNAJC19)
  → Mitochondrial energy deficit in basal ganglia neurons → Leigh-like signal on MRI
  → Cochlear hair cell energy failure → SNHL (bilateral, sensorineural, often severe)
  → Hepatocyte mito dysfunction → neonatal cholestasis (often transient)

3-MGA mechanism (shared with Type III diseases):
  OXPHOS dysfunction → acetyl-CoA overflow → HMG-CoA pathway → 3-methylglutaconyl-CoA
  cannot be fully catabolised → excreted as 3-methylglutaconic acid (3-MGA) in urine.
  Same overflow pathway as OPA3 (Type III) and DNAJC19 (Type III), ATPAF2 (Type IV).

PROTEIN STRUCTURE (490 aa, 6q22.1):
  MTS / ER signal (aa 1-30): Targets to ER/MAM; retains in ER-mito contact sites
  Serine active site domain (aa 31-350): Lipid reacylase active site; Ser-His-Asp catalytic triad;
    serine active site-containing beta-hydrolase fold
  Trans-membrane domains (aa 351-420): Multi-pass; anchors to MAM/OMM junction
  C-terminal regulatory domain (aa 421-490): Protein-protein interaction; MAM complex assembly

PATHOGENIC VARIANT DISTRIBUTION (biallelic LOF, n=40 patients, seed-539):
  Missense in serine active site domain (Ser active site, aa 31-350): ~45% of alleles
    p.Leu519Phe equivalent / conserved active site substitution: ~20% (European)
    p.Pro511Leu / p.Arg266Gln / other missense: ~25%
  Null variants (nonsense/frameshift): ~35% of alleles
    c.1037delC (frameshift, European): ~12%
    c.890+1G>A (splice, intronic): ~10%
    Other nonsense/frameshift: ~13%
  Splice site + missense compound het: ~15%
  Large deletions (exon-level): ~5%

CLINICAL PHENOTYPE — MEGDEL SYNDROME:
  SENSORINEURAL HEARING LOSS (100%) — CARDINAL FEATURE:
    Bilateral, sensorineural; onset within first year of life (often detected at newborn hearing screen).
    Severity: moderate-profound in most (60 dB+ loss); complete in ~40%.
    SNHL is the MOST DISTINCTIVE feature among all 3-MGA-uria diseases.
    KEY DDx: NO SNHL in DNAJC19, OPA3, MECR, AUH, Barth — SERAC1 alone has SNHL as cardinal.
    Cochlear implant: highly effective (Level B); transforms language acquisition and development.
    Hearing aid: partial benefit (moderate cases); CI preferred if profound.
  3-METHYLGLUTACONIC ACIDURIA (100%):
    Level: 30-200 mmol/mol creatinine; Type V classification.
    Same overflow pathway as OPA3 (Type III) and DNAJC19 (Type III) — NOT primary metabolite error.
    Acylcarnitine profile: NORMAL (KEY DDx from Barth/TAZ where C4-DC elevated).
    No 3-HMG elevation (DDx AUH where 3-HMG is normal but 3-MGA primary mechanism different).
  ENCEPHALOPATHY / INTELLECTUAL DISABILITY (100%):
    Moderate-severe intellectual disability (IQ typically <50 in classic cases).
    Language: severely impaired or absent; AAC (augmentative communication) often needed.
    Motor delay: universal; most achieve independent ambulation (delayed, by 3-5 yr).
    Progressive encephalopathy: slow progression over years (unlike nonprogressive DNAJC19 ataxia).
  LEIGH-LIKE MRI (80-90%):
    Bilateral putamen and/or caudate T2 hyperintensity; brainstem nuclei involvement.
    NOT classic Leigh (not always symmetric or complete) → "Leigh-like."
    Cerebral atrophy variable (30-50% of adults).
    NO GP iron: key negative vs MECR/NBIA where GP T2*/SWI hypointensity present.
  EPILEPSY (60-70%):
    Multiple seizure types: infantile spasms (25%), focal (30%), generalized tonic-clonic (20%), myoclonic (15%).
    Often drug-resistant; polytherapy common.
    LEV preferred; ACTH/VGB for infantile spasms.
  NEONATAL LIVER DYSFUNCTION (60-70%):
    Neonatal cholestasis, hepatomegaly, elevated transaminases; onset first weeks of life.
    Self-limited in most (~85%): resolves by 6-12 months.
    Rare fulminant hepatic failure (~15% of liver-affected patients) — poor prognosis.
    UDCA (ursodeoxycholic acid) for cholestasis support.
  HYPOTONIA (100%):
    Central and peripheral; severe in neonatal period; partially improves with age.
  DYSTONIA (50-60%):
    Generalized or segmental; basal ganglia origin (Leigh-like signal in putamen).
    Oral feeding difficulty; dysphagia requires tube feeding in ~40%.
  SPASTICITY (40-50%):
    Corticospinal involvement; limb spasticity; baclofen + physiotherapy.
  OPTIC ATROPHY: ABSENT (KEY DDx from OPA3 100%, MECR 80-90%)
  DILATED CARDIOMYOPATHY: ABSENT (KEY DDx from DNAJC19 100%, Barth 100%)
  NEUTROPENIA: ABSENT (KEY DDx from Barth 95%)
  GP IRON on MRI: ABSENT (KEY DDx from MECR, NBIA1-7 — all have GP hypointensity on T2*/SWI)
  CHOREA: ABSENT (KEY DDx from OPA3 85-90%, FTL neuroferritinopathy, DNAJC19 absent)

TREATMENT & PHARMACOGENOMICS:
  Cochlear Implant: SNHL — Level B
    Cochlear CI highly effective in MEGDEL; hearing restoration transforms communication + development.
    Should be offered early (ideal <18 months); speech therapy intensive post-CI.
    Anaesthetic caution: mito disease → avoid prolonged fasting; glucose-containing IV fluids; cautious
      neuromuscular blockade; propofol (PRIS risk in mito patients — some centres avoid).
  Hearing Aids: SNHL (partial) — Level C
    Benefit for moderate loss before CI decision; CI preferred for profound loss.
  LEV (Levetiracetam): Seizures — PREFERRED — Level B
    Renal excretion; no hepatic metabolism; no mitochondrial respiratory chain inhibition.
    Same preference across 3-MGA-uria series (OPA3, DNAJC19, MECR, SERAC1).
  ACTH: Infantile Spasms — Level A (UKISS protocol)
    First-line for infantile spasms if present; SERAC1 not a contraindication to ACTH.
  VGB (Vigabatrin): Infantile Spasms — Level A (UKISS protocol)
    Dual ACTH+VGB per UKISS. Caution: VGB visual field defects long-term; SERAC1 not CI to VGB
    (unlike CP aceruloplasminemia where VGB is absolute CI for additive retinal toxicity).
  VPA (Valproate): MODERATE CAUTION
    Complex I depression by VPA → additive risk in SERAC1 (Complex I already deficient).
    NOT absolute CI like MECR (lipoic acid pathway intact in SERAC1).
    Hepatic risk: pre-existing neonatal liver disease in SERAC1 → start VPA only after liver function
      normalises; monitor LFTs + NH3 closely; POLG-LIKE caution applies.
    Use LEV/CLB first; VPA only if seizures refractory and benefits outweigh risk.
  Baclofen: Spasticity — Level C
    Oral baclofen for spasticity management; ITB (intrathecal baclofen) for severe cases.
    No mito contraindication.
  UDCA (Ursodeoxycholic Acid): Neonatal Cholestasis — Level C
    Reduces bile acid toxicity; supportive for neonatal hepatic dysfunction.
    Discontinue once liver function normalises (typically 6-12 months).
  CoQ10: Mitochondrial cofactor — Level C
    100-300 mg/day; rationale: Complex I/IV support; evidence weak but commonly used in mito disease.
  Riboflavin (B2): Mitochondrial cofactor — Level C
    100-200 mg/day; ETFDH/flavoprotein subunit support; reasonable in mito disease.
  L-Carnitine: Secondary depletion — Level C
    50-100 mg/kg/day; secondary carnitine depletion in OXPHOS disease; supplement if C0 low.
  KD (Ketogenic Diet): Investigational — Level D
    Some benefit in Leigh-like mito encephalopathies; SERAC1 complex I involvement → possible benefit.
    No absolute CI (unlike HMGCL where KD CI; SERAC1 ketogenesis intact).
    Requires careful monitoring: liver disease period may contraindicate; delay until liver stable.
  Propofol: AVOID for anaesthesia (Propofol Infusion Syndrome risk in mitochondrial disease)
    Use alternative anaesthetic agents; alert anaesthesiology team.
  Tetrabenazine/Deutetrabenazine: NOT indicated (no chorea — DDx from OPA3/FTL where chorea present)
  PHT/CBZ: AVOID (Complex I inhibition risk; additive mito dysfunction; CYP induction as secondary concern)
"""

import random
import math
from datetime import date, timedelta

SEED = 539
RNG = random.Random(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────
def _date(n_days_ago: int) -> str:
    return (date.today() - timedelta(days=n_days_ago)).isoformat()

def _rng_ages(n: int, mu: float, sigma: float, lo: float, hi: float) -> list:
    out = []
    for _ in range(n):
        v = RNG.gauss(mu, sigma)
        out.append(round(max(lo, min(hi, v)), 1))
    return out


# ── overview ──────────────────────────────────────────────────────────────────
def get_overview() -> dict:
    """SERAC1 MEGDEL Syndrome — overview payload for /api/serac1/overview."""
    n = 40
    # SNHL onset (typically detected at newborn screen or by 12 months)
    snhl_onset_mo = [round(max(0, min(24, RNG.gauss(4, 4))), 1) for _ in range(n)]
    # 3-MGA levels (mmol/mol creatinine)
    mga_vals = [round(max(30, min(200, RNG.gauss(95, 40))), 1) for _ in range(n)]
    # Seizure onset (months, for 60-70% who have seizures)
    seizure_onset_mo = [round(max(1, min(36, RNG.gauss(9, 6))), 1) for _ in range(28)]

    liver_affected_n = 27  # 67%
    ci_candidates_n = 35   # 88% had CI offered/placed

    return {
        "generated": date.today().isoformat(),
        "cohort": n,
        "seed": SEED,
        "disease": "MEGDEL Syndrome (3-MGA-uria, Deafness, Encephalopathy, Leigh-Like)",
        "gene": "SERAC1; Serine Active Site Containing 1",
        "protein": "SERAC1 — 490 aa, phosphatidylglycerol/BMP remodeling enzyme, mitochondria-associated ER membrane (MAM)",
        "chromosome": "6q22.1",
        "omim_gene": "614725",
        "omim_disease": "614739",
        "inheritance": "Autosomal Recessive; biallelic LOF; AR",
        "prevalence": "~100-150 patients worldwide (2026); no dominant founder mutation; diverse populations",
        "first_described": "Wortmann et al. 2012 (AJHG) — SERAC1 as BMP/PG remodeling enzyme at MAM; MEGDEL cohort",
        "category": "3-MGA-uria Type V / Mitochondrial-MAM / MEGDEL",
        "kpis": {
            "n_patients": n,
            "snhl_pct": 100,
            "encephalopathy_pct": 100,
            "mga_pct": 100,
            "leigh_like_mri_pct": 87,
            "epilepsy_pct": 68,
            "neonatal_liver_pct": 67,
            "dystonia_pct": 55,
            "spasticity_pct": 45,
            "optic_atrophy_pct": 0,
            "dcm_pct": 0,
            "neutropenia_pct": 0,
            "gp_iron_mri_pct": 0,
            "cochlear_implant_placed_pct": round(ci_candidates_n / n * 100),
            "mean_mga_mmol": round(sum(mga_vals) / n, 1),
            "mean_snhl_onset_mo": round(sum(snhl_onset_mo) / n, 1),
        },
        "phenotype_summary": {
            "snhl_100pct": "Sensorineural hearing loss — 100%; CARDINAL; bilateral; early onset (median ~4 months, newborn screen); profound in 60%",
            "encephalopathy_100pct": "Encephalopathy / Intellectual Disability — 100%; moderate-severe; language severely impaired; motor delay universal",
            "mga_100pct": "3-Methylglutaconic aciduria — 100%; Type V classification; 30-200 mmol/mol Cr; normal acylcarnitine profile",
            "leigh_like_87pct": "Leigh-like MRI — 87%; bilateral putamen/caudate T2 hyperintensity; brainstem; NOT GP iron (DDx MECR/NBIA)",
            "epilepsy_68pct": "Epilepsy — 68%; infantile spasms (25%), focal (30%), GTCs (20%), myoclonic (15%); often drug-resistant",
            "liver_67pct": f"Neonatal liver dysfunction — 67%; cholestasis + hepatomegaly; self-limited in ~85% (resolves 6-12 mo); {liver_affected_n} patients",
            "dystonia_55pct": "Dystonia — 55%; basal ganglia (Leigh-like putamen); generalized/segmental; dysphagia 40%",
            "key_negatives": "NO optic atrophy (DDx OPA3/MECR); NO DCM (DDx DNAJC19/Barth); NO neutropenia (DDx Barth); NO GP iron (DDx MECR/NBIA); NO chorea (DDx OPA3/FTL); NORMAL acylcarnitine (DDx Barth C4-DC)",
        },
        "clinical_highlights": [
            "SNHL is the SINGLE MOST DISTINCTIVE feature among all 3-MGA-uria diseases — no other 3-MGA disease has SNHL as cardinal",
            "Cochlear implant is highly effective (Level B) — early CI transforms language acquisition despite severe ID",
            "Leigh-like MRI (bilateral putamen) without GP iron: DDx from MECR (GP iron) and NBIA (pallidal iron)",
            "Neonatal liver dysfunction: transient in 85%; must stabilise liver before starting VPA (hepatic risk doubled)",
            "Normal acylcarnitine profile: KEY negative to distinguish from Barth syndrome (TAZ, C4-DC elevated)",
            "NO dilated cardiomyopathy — DDx DNAJC19 (DCM 100%) and Barth (DCM 100%)",
            "NO optic atrophy — DDx OPA3 (100%) and MECR (80-90%); eyes are spared in SERAC1",
            "VPA moderate caution (Complex I already deficient + potential liver disease history) — LEV first",
            "6q22.1 AR; no dominant founder mutation — WES/panel required (no quick founder test)",
            "Propofol anaesthetic risk: PRIS in mito disease; alert anaesthesia team for all surgical procedures",
        ],
        "contraindications": [
            {"drug": "PHT/Phenytoin", "reason": "AVOID — Complex I inhibition additive to SERAC1 OXPHOS dysfunction; CYP induction secondary concern"},
            {"drug": "CBZ/Carbamazepine", "reason": "AVOID — same Complex I and CYP induction concerns as PHT"},
            {"drug": "VPA (Valproate)", "reason": "MODERATE CAUTION — Complex I depression additive; liver disease history in 67% → monitor LFTs + NH3; NOT absolute CI (lipoic acid intact)"},
            {"drug": "Propofol (anaesthesia)", "reason": "AVOID — Propofol Infusion Syndrome (PRIS) risk in mitochondrial disease; use alternatives (sevoflurane, ketamine)"},
            {"drug": "Tetrabenazine/Deutetrabenazine", "reason": "NOT indicated — no chorea in MEGDEL (OPA3/FTL have chorea; SERAC1 does not)"},
        ],
        "thresholds": [
            {"parameter": "3-MGA urine", "threshold": "> 20 mmol/mol Cr", "action": "Diagnostic of 3-MGA-uria; GC-MS urine OA; add SERAC1 to panel if SNHL present"},
            {"parameter": "SNHL on ABR/OAE", "threshold": "Bilateral sensorineural loss", "action": "Refer cochlear implant centre; CI evaluation by 12 months; hearing aid bridge"},
            {"parameter": "ALT/AST > 3× ULN (neonatal)", "threshold": "Liver dysfunction threshold", "action": "UDCA; postpone VPA; supportive; monitor bilirubin; rarely fulminant"},
            {"parameter": "Infantile spasms on EEG (hypsarrhythmia)", "threshold": "IS diagnosis", "action": "ACTH + VGB per UKISS protocol; add LEV for breakthrough"},
            {"parameter": "Carnitine C0 < 25 µmol/L", "threshold": "Secondary depletion", "action": "L-carnitine supplement 50-100 mg/kg/day"},
        ],
        "gene_biology": {
            "protein_length": 490,
            "domains": [
                {"domain": "ER/MAM targeting signal", "residues": "aa 1-30", "function": "Targets to ER-mitochondria contact sites (MAM); retained in mitochondria-associated membranes"},
                {"domain": "Serine active site (lipase fold)", "residues": "aa 31-350", "function": "Ser-His-Asp catalytic triad; lyso-PG reacylase activity → BMP + PG synthesis; beta-hydrolase fold"},
                {"domain": "Transmembrane domain", "residues": "aa 351-420", "function": "Multi-pass TM; anchors to MAM/OMM junction; required for membrane tethering"},
                {"domain": "C-terminal regulatory", "residues": "aa 421-490", "function": "Protein-protein interactions; MAM complex assembly; SERAC1 oligomerization"},
            ],
            "complex": "MAM (mitochondria-associated ER membrane) lipid remodeling complex",
            "partners": "VDAC1 (OMM); IP3R (ER); PTPIP51 (OMM-ER tether); GRP75 (MAM scaffold)",
            "pathway": "Phosphatidylglycerol/BMP remodeling → inner mitochondrial membrane lipid composition → Complex I/IV assembly stability",
            "lof_consequence": "BMP deficiency → Complex I/IV disassembly → OXPHOS dysfunction → 3-MGA-uria + SNHL + Leigh-like encephalopathy + neonatal cholestasis",
        },
        "ddx_table": [
            {"feature": "3-MGA elevated", "serac1_megdel": "✅ 100%", "dnajc19_dcma": "✅ 100%", "opa3_costeff": "✅ 100%", "barth_taz": "✅ 100%", "mecr_mepan": "✅ 100%"},
            {"feature": "SNHL (sensorineural)", "serac1_megdel": "✅ 100% CARDINAL", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "❌ Absent", "barth_taz": "❌ Absent", "mecr_mepan": "❌ Absent"},
            {"feature": "Dilated Cardiomyopathy", "serac1_megdel": "❌ Absent", "dnajc19_dcma": "✅ 100%", "opa3_costeff": "❌ Absent", "barth_taz": "✅ 100%", "mecr_mepan": "❌ Absent"},
            {"feature": "Optic Atrophy", "serac1_megdel": "❌ Absent", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "✅ 100%", "barth_taz": "❌ Absent", "mecr_mepan": "✅ 80-90%"},
            {"feature": "Chorea", "serac1_megdel": "❌ Absent", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "✅ 85-90%", "barth_taz": "❌ Absent", "mecr_mepan": "❌ Absent (dystonia)"},
            {"feature": "GP iron on MRI (T2*/SWI)", "serac1_megdel": "❌ Absent", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "❌ Absent", "barth_taz": "❌ Absent", "mecr_mepan": "✅ Bilateral GP"},
            {"feature": "Leigh-like MRI", "serac1_megdel": "✅ 87% putamen", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "❌ Absent", "barth_taz": "❌ Absent", "mecr_mepan": "❌ Absent"},
            {"feature": "Neonatal liver dysfn", "serac1_megdel": "✅ 67%", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "❌ Absent", "barth_taz": "❌ Absent", "mecr_mepan": "❌ Absent"},
            {"feature": "Neutropenia", "serac1_megdel": "❌ Absent", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "❌ Absent", "barth_taz": "✅ 95%", "mecr_mepan": "❌ Absent"},
            {"feature": "Elevated C4-DC acylcarnitine", "serac1_megdel": "❌ Normal", "dnajc19_dcma": "❌ Normal", "opa3_costeff": "❌ Normal", "barth_taz": "✅ Elevated", "mecr_mepan": "❌ Normal"},
            {"feature": "Founder mutation", "serac1_megdel": "None dominant (WES required)", "dnajc19_dcma": "Hutterite c.130-1G>C", "opa3_costeff": "Iraqi-Jewish p.Gln105*", "barth_taz": "X-linked (Xq28)", "mecr_mepan": "Bedouin p.Tyr200His"},
        ],
    }


# ── breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    """SERAC1 MEGDEL Syndrome — breakdown payload for /api/serac1/breakdown."""
    n = 40
    RNG2 = random.Random(SEED + 1)

    # Phenotype distribution
    phenotype_groups = [
        ("Classic MEGDEL (SNHL+Leigh-like+epilepsy+liver)", 18),
        ("MEGDEL without liver (SNHL+Leigh-like+epilepsy, liver-spared)", 10),
        ("Severe MEGDEL (SNHL+Leigh-like+spastic quadriplegia, non-ambulant)", 8),
        ("Mild MEGDEL (SNHL+encephalopathy, no Leigh-like, ambulant)", 4),
    ]

    # Variant distribution (allele-level, biallelic AR disease)
    variant_dist = [
        {"variant": "Missense: serine active-site domain (various, ~20 alleles/variant class)", "n_alleles": 36, "pct": 45, "effect": "Active-site disruption; BMP/PG synthesis loss; hypomorphic alleles = milder phenotype"},
        {"variant": "Frameshift: c.1037delC (European)", "n_alleles": 10, "pct": 12, "effect": "Premature stop; complete SERAC1 null; severe neonatal/early-infantile presentation"},
        {"variant": "Splice site: c.890+1G>A (intron 7)", "n_alleles": 8, "pct": 10, "effect": "Exon 7 skipping → frameshift; LOF; combined with missense in compound-het patients"},
        {"variant": "Other frameshift/nonsense (diverse)", "n_alleles": 10, "pct": 13, "effect": "Variable origins; all biallelic null = severe"},
        {"variant": "Splice site + missense (compound heterozygous)", "n_alleles": 12, "pct": 15, "effect": "Compound het; severity depends on missense hypomorphism"},
        {"variant": "Large exon deletion", "n_alleles": 4, "pct": 5, "effect": "Complete null; array CGH / WGS needed for detection"},
    ]

    # Treatment distribution
    treatment_dist = [
        {"treatment": "Cochlear Implant (placed)", "n": 35, "pct": 88, "indication": "SNHL profound/severe — Level B; high effectiveness"},
        {"treatment": "Hearing Aids (pre-CI bridge)", "n": 40, "pct": 100, "indication": "SNHL — Level B (bridge to CI, or ongoing for moderate loss)"},
        {"treatment": "LEV (Levetiracetam)", "n": 28, "pct": 70, "indication": "Epilepsy — Level B preferred; renal; no mito toxicity"},
        {"treatment": "ACTH (infantile spasms)", "n": 10, "pct": 25, "indication": "Infantile spasms — Level A (UKISS)"},
        {"treatment": "VGB (vigabatrin) + ACTH", "n": 10, "pct": 25, "indication": "Infantile spasms — Level A (UKISS dual therapy)"},
        {"treatment": "CLB (Clobazam) add-on", "n": 15, "pct": 38, "indication": "Drug-resistant epilepsy — Level C"},
        {"treatment": "KD (Ketogenic Diet)", "n": 6, "pct": 15, "indication": "Drug-resistant epilepsy/Leigh-like — Level D investigational"},
        {"treatment": "UDCA (ursodeoxycholic acid)", "n": 27, "pct": 68, "indication": "Neonatal cholestasis — Level C supportive"},
        {"treatment": "L-Carnitine", "n": 30, "pct": 75, "indication": "Secondary depletion — Level C"},
        {"treatment": "CoQ10 + Riboflavin", "n": 25, "pct": 63, "indication": "Mitochondrial cofactor cocktail — Level C"},
        {"treatment": "Baclofen (oral/ITB)", "n": 18, "pct": 45, "indication": "Spasticity — Level C"},
        {"treatment": "Gastrostomy / tube feeding", "n": 16, "pct": 40, "indication": "Dysphagia / failure to thrive"},
        {"treatment": "Speech + AAC", "n": 40, "pct": 100, "indication": "Severe language impairment; augmentative communication"},
    ]

    # 3-MGA level by phenotype
    mga_by_pheno = [
        {"phenotype": "Classic MEGDEL", "mean_mga": 105, "range": "55-200", "n": 18},
        {"phenotype": "MEGDEL without liver", "mean_mga": 88, "range": "40-170", "n": 10},
        {"phenotype": "Severe MEGDEL", "mean_mga": 122, "range": "70-200", "n": 8},
        {"phenotype": "Mild MEGDEL", "mean_mga": 52, "range": "30-110", "n": 4},
    ]

    # CI outcomes
    ci_outcomes = [
        {"outcome": "Speech perception improved (open set)", "n": 24, "pct_of_ci": 69, "notes": "Best outcomes with CI < 18 months + intensive post-CI therapy"},
        {"outcome": "Language acquisition (some words/phrases)", "n": 21, "pct_of_ci": 60, "notes": "Despite severe ID, hearing restoration supports language foundation"},
        {"outcome": "AAC + CI combined", "n": 18, "pct_of_ci": 51, "notes": "Combined CI + AAC optimal for MEGDEL communication"},
        {"outcome": "CI complications (device failure/repositioning)", "n": 3, "pct_of_ci": 9, "notes": "Standard CI complication rate; not disease-specific"},
    ]

    return {
        "generated": date.today().isoformat(),
        "cohort": n,
        "seed": SEED,
        "phenotype_groups": [{"group": g, "n": c, "pct": round(c / n * 100)} for g, c in phenotype_groups],
        "variant_distribution": variant_dist,
        "treatment_distribution": treatment_dist,
        "mga_by_phenotype": mga_by_pheno,
        "cochlear_implant_outcomes": ci_outcomes,
        "neurological_outcomes": {
            "leigh_like_mri_pct": 87,
            "epilepsy_pct": 68,
            "drug_resistant_epilepsy_pct": 45,
            "infantile_spasms_pct": 25,
            "dystonia_pct": 55,
            "spasticity_pct": 45,
            "independent_ambulation_pct": 60,
            "nonambulant_pct": 40,
            "moderate_severe_id_pct": 95,
            "nonverbal_pct": 70,
        },
        "liver_outcomes": {
            "liver_affected_pct": 67,
            "self_limited_pct": 85,
            "fulminant_hepatic_failure_pct": 15,
            "median_resolution_months": 9,
        },
        "biomarker_summary": {
            "mga_range_mmol_cr": "30-200",
            "mga_mean": 95,
            "acylcarnitine_normal_pct": 100,
            "c4dc_normal_pct": 100,
            "snhl_bilateral_pct": 100,
            "snhl_profound_pct": 60,
            "c0_carnitine_low_pct": 75,
            "lactate_mild_elevation_pct": 65,
        },
    }


# ── definitions ───────────────────────────────────────────────────────────────
def get_definitions() -> dict:
    """SERAC1 MEGDEL Syndrome — definitions for /api/serac1/definitions."""
    return {
        "generated": date.today().isoformat(),
        "disease": "MEGDEL Syndrome (3-MGA-uria + Deafness + Encephalopathy + Leigh-Like)",
        "gene": "SERAC1",
        "omim_gene": "614725",
        "omim_disease": "614739",
        "definitions": [
            {
                "term": "SERAC1 / Serine Active Site Containing 1",
                "definition": "490-amino acid phospholipid remodeling enzyme at the mitochondria-associated ER membrane (MAM). SERAC1 reacylates 1-acyl-2-lyso-phosphatidylglycerol (lyso-PG) to produce BMP (bis(monoacylglycero)phosphate) and mature phosphatidylglycerol (PG), essential for mitochondrial inner membrane composition and Complex I/IV assembly.",
                "relevance": "SERAC1 LOF → BMP deficiency → Complex I/IV disassembly → OXPHOS dysfunction → 3-MGA-uria overflow + cochlear hair cell energy failure (SNHL) + basal ganglia neuronal death (Leigh-like) + hepatocyte mito dysfunction (neonatal cholestasis).",
            },
            {
                "term": "MEGDEL Syndrome",
                "definition": "3-Methylglutaconic aciduria + Deafness (SNHL) + Encephalopathy + Leigh-Like; caused by biallelic SERAC1 LOF. 3-MGA Type V classification. Key distinguishing feature: SNHL (100%) is cardinal and absent in all other 3-MGA-uria diseases.",
                "relevance": "Name encodes the diagnostic tetrad. If a child has 3-MGA-uria AND bilateral SNHL, SERAC1 should be the first gene tested — no other 3-MGA disease produces SNHL.",
            },
            {
                "term": "BMP (Bis(monoacylglycero)phosphate)",
                "definition": "Lysosomal/late-endosomal phospholipid synthesised by SERAC1. BMP is enriched in the inner leaflet of late endosomes and in the inner mitochondrial membrane. Functionally: BMP stabilises Complex I and Complex IV supercomplexes; required for respiratory chain supercomplex assembly (respirasomes).",
                "relevance": "BMP deficiency in SERAC1 LOF → respiratory chain supercomplex destabilisation → isolated Complex I + IV deficiency on BNA (Blue Native gel) — the biochemical signature of MEGDEL in muscle biopsy.",
            },
            {
                "term": "MAM (Mitochondria-Associated ER Membrane)",
                "definition": "Dynamic ER-mitochondria contact sites where SERAC1 resides. MAM coordinates Ca²⁺ transfer, phospholipid biosynthesis/remodeling, cholesterol trafficking, and autophagosome formation between ER and mitochondria.",
                "relevance": "SERAC1 at the MAM links ER-mito communication to inner mito membrane lipid composition. Loss of SERAC1 disrupts MAM function → cholesterol mis-trafficking + BMP/PG deficiency — explaining the multi-organ phenotype (liver, cochlea, brain).",
            },
            {
                "term": "3-MGA-uria Type V",
                "definition": "Classification of SERAC1 MEGDEL syndrome among the 3-methylglutaconic acidurias. Type V designation (alongside Type I-AUH, Type II-TAZ/Barth, Type III-OPA3/DNAJC19, Type IV-ATPAF2). All types share elevated urinary 3-methylglutaconic acid via OXPHOS dysfunction → HMG-CoA pathway overflow; mechanisms differ by gene.",
                "relevance": "3-MGA-uria type alone does NOT establish the gene diagnosis — complete clinical picture (SNHL? DCM? optic atrophy? chorea? liver?) + acylcarnitine profile + WES/panel required for specific gene diagnosis.",
            },
            {
                "term": "Leigh-Like MRI in SERAC1 MEGDEL",
                "definition": "Bilateral T2 hyperintensity in putamen (most common) and/or caudate nucleus, brainstem nuclei, and rarely thalamus; on MRI. 'Leigh-like' because it resembles classic Leigh syndrome (SURF1, SCO2, PDH) but may be incomplete/asymmetric.",
                "relevance": "KEY DDx: Leigh-like signal without GP iron distinguishes SERAC1 from MECR/MEPAN (bilateral GP iron on SWI) and NBIA diseases (pallidal iron). Serum/CSF lactate usually elevated in Leigh-like periods (energy failure in basal ganglia).",
            },
            {
                "term": "SNHL in SERAC1 and Cochlear Implant",
                "definition": "Bilateral sensorineural hearing loss in 100% of SERAC1/MEGDEL patients; typically profound by age 1-2 years. Mechanism: cochlear hair cell high energy demand → OXPHOS failure → hair cell death → permanent SNHL. Cochlear implant bypasses damaged hair cells → direct electrical stimulation of auditory nerve → sound perception restored.",
                "relevance": "Cochlear implant is the single most impactful specific intervention in MEGDEL. Even with severe ID, CI transforms communication ability. Implant before 18 months yields best language outcomes. Alert anaesthesia team of mito disease for CI surgery (propofol risk).",
            },
            {
                "term": "Neonatal Liver Dysfunction in SERAC1",
                "definition": "Neonatal cholestasis, hepatomegaly, elevated transaminases (ALT/AST up to 10× ULN), elevated direct bilirubin; onset first 2-4 weeks of life in ~67% of SERAC1 patients. Self-limited in ~85% (resolves 6-12 months). Rare fulminant hepatic failure (~15% of liver-affected cases) may require liver transplant.",
                "relevance": "Neonatal liver disease in context of 3-MGA-uria + SNHL = SERAC1 until proven otherwise. Liver disease alters VPA risk (avoid in neonatal period; reintroduce only after liver normalisation). UDCA supportive; standard neonatal liver monitoring.",
            },
            {
                "term": "VPA Moderate Caution (NOT Absolute CI) in SERAC1",
                "definition": "VPA inhibits Complex I (additive to SERAC1 Complex I deficiency) AND carries hepatotoxicity risk — especially significant because SERAC1 patients frequently have neonatal/early liver disease. VPA is NOT absolute CI in SERAC1 (lipoic acid pathway intact — contrast MECR where VPA is absolute CI). Moderate caution: if liver normalised and LEV fails, VPA may be used with close LFT + NH3 monitoring.",
                "relevance": "Prescribing hierarchy in MEGDEL: LEV first (preferred) → CLB/TPM → KD investigational → VPA only if refractory and liver stable. NEVER start VPA during neonatal liver dysfunction period.",
            },
            {
                "term": "Propofol Infusion Syndrome (PRIS) in Mitochondrial Disease",
                "definition": "Rare but life-threatening anaesthetic complication: high-dose prolonged propofol → Complex I inhibition → lactic acidosis, rhabdomyolysis, cardiac failure, death. Risk elevated in patients with pre-existing mitochondrial Complex I deficiency (e.g., SERAC1).",
                "relevance": "SERAC1 patients undergoing cochlear implant surgery or other procedures must have anaesthesia team alerted to mito disease. Avoid propofol; prefer sevoflurane (inhalational) or ketamine for procedural sedation; use 5% glucose IV; minimise fasting time.",
            },
            {
                "term": "SERAC1 vs Barth Syndrome (TAZ) DDx",
                "definition": "Both SERAC1 and Barth (TAZ, Xq28) have 3-MGA-uria. KEY DDx: Barth = X-linked recessive (males only unless carrier female), DCM (100%), skeletal myopathy, neutropenia (95%), C4-DC/C4-OH acylcarnitine elevated; SERAC1 = AR (any sex), SNHL (100%), Leigh-like MRI, neonatal liver dysfunction, NO DCM, NO neutropenia, normal acylcarnitine.",
                "relevance": "The single fastest distinguishing test: acylcarnitine profile. C4-DC or 3-methylglutarylcarnitine elevated → Barth (TAZ). Normal acylcarnitine + SNHL → SERAC1. Karyotype/sex helps: Barth males only (XXY excluded).",
            },
        ],
    }
