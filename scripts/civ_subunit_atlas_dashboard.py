#!/usr/bin/env python3
"""CIV-Subunit-Atlas — Complete 19-Gene Nuclear-Encoded Complex IV (Cytochrome c Oxidase) Atlas
4 structural subunits + 15 assembly factors (all nuclear-encoded)
760-patient aggregate cohort (19 × 40, seeds 725–743)

Complex IV (Cytochrome c Oxidase / COX) facts:
  - 14 nuclear-encoded structural subunits + 3 mtDNA-encoded (MT-CO1, MT-CO2, MT-CO3)
  - NDUFA4 reclassified as 14th nuclear structural subunit of CIV (Balsa 2012, Mol Cell)
  - Terminal electron acceptor of the ETC: transfers electrons from reduced Cyt c → O₂ (→ H₂O)
  - Pumps 4H⁺/2e⁻ across IMM; generates ~40% of mitochondrial proton gradient
  - Copper centres: CuA (binuclear, MT-CO2) + CuB (mononuclear, MT-CO1) — redox-active
  - Heme centres: haem a (MT-CO1) + haem a3-CuB binuclear centre (MT-CO1) — O₂ reduction site
  - 15 nuclear-encoded assembly factors orchestrate ordered CIV biogenesis in IMM
  - CII ALWAYS NORMAL in isolated CIV deficiency (CII = internal biochemical reference)
  - WES MISSES MT-CO1, MT-CO2, MT-CO3 — covered in MT-Genome-Atlas separately

ATLAS SCOPE (19 nuclear genes):
  Structural subunits (4 nuclear, individual dashboards built):
    COX4I1, COX6B1, COX8A, NDUFA4
  Assembly factors (15 nuclear, individual dashboards built):
    SURF1, SCO1, SCO2, COX10, COX15, COX20, COA3, COA5,
    COA6, COA7, TACO1, LRPPRC, PET100, COX14, FASTKD2

PHENOTYPIC SPECTRUM (CIV Deficiency):
  - Leigh syndrome / Leigh-like (SURF1 — most common CIV nuclear gene; 30-50 patients/year globally)
  - Cardioencephalomyopathy + HCM 100% (SCO2 — copper delivery, most cardiac CIV gene)
  - French-Canadian Leigh (LRPPRC — MT-CO1 mRNA stability, p.Ala354Val founder)
  - Charcot-Marie-Tooth + Ataxia (SCO1, COA7 — peripheral neuropathy dominant)
  - Spinocerebellar ataxia + axonal neuropathy (COA7 — ONLY CIV gene with ataxia-neuropathy, mild CIV defect)
  - Heme a biosynthesis failure (COX10 → haem a, COX15 → haem a3)
  - PINDAC triad — EPI + Dyserythropoietic anaemia + Calvarial hyperostosis (COX4I1)
  - Cardiomyopathy-only (COA5, COA6 — early cardiac-specific assembly)
  - Hepatic failure / hypertrophic (COX15, TACO1)

BIOCHEMICAL FINGERPRINT (Isolated CIV Deficiency):
  - CIV (COX activity, measured by cytochrome c oxidase assay) markedly reduced
  - CI, CII, CIII normal in pure structural/AF mutations
  - CII ALWAYS NORMAL — internal reference (no mtDNA-encoded CII subunits)
  - Lactate/pyruvate ratio elevated; lactic acidosis common but variable severity
  - Copper supplements (oral CuHis) restore SCO2/SCO1-related enzyme activity in partial deficiency
  - BN-PAGE: residual COX sub-complexes absent; COX assembly intermediates visible in some AFs

COHORT: 19 × 40 = 760 patient slots (seeds 725–743; gene-specific seeds)
"""

import random

SEED = 744
rng  = random.Random(SEED)

# ── All 19 nuclear-encoded CIV-related genes — authoritative table ────────────
# gene_class: "structural_subunit" | "assembly_factor"
CIV_GENES = [
    # ── Nuclear Structural Subunits ──────────────────────────────────────────
    {
        "gene": "COX4I1", "aa": "169 aa", "kDa": "17.2 kDa",
        "gene_class": "structural_subunit", "subunit_series": "COX4",
        "civ_module": "Matrix-facing, no TM helix; ATP/ADP allosteric regulatory site; allosteric switch: ADP (low energy) → active CIV; ATP (high energy) → product inhibition of CIV",
        "omim_gene": 123864, "chromosome": "16q22.1", "seed": 725,
        "phenotype": "PINDAC Syndrome: Exocrine Pancreatic Insufficiency + Dyserythropoietic Anaemia + Calvarial Hyperostosis + CIV Deficiency (COXPD12)",
        "disease": "COXPD12 / PINDAC Syndrome — COX4I1 homozygous deletion/LOF; PINDAC triad pathognomonic; CIV 15-30% residual; CI/CII/CIII NORMAL",
        "disease_omim": 616501, "inheritance": "AR",
        "hallmark": "PINDAC TRIAD PATHOGNOMONIC: Exocrine pancreatic insufficiency (PERT mandatory) + Dyserythropoietic anaemia + Calvarial hyperostosis; ATP allosteric site unique among CIV subunits; Bedouin founder 16q22 deletion; NO HCM; NO Fanconi; NO ataxia",
        "key_ddx": "vs SURF1-Leigh: no PINDAC features; vs Diamond-Blackfan: pure erythroid vs tri-lineage PINDAC; vs Shwachman-Diamond: EPI without anaemia+calvarial; vs Pearson: mtDNA deletion syndrome vs AR nuclear",
        "founder_variant": "Homozygous 16q22 deletion (Bedouin founder); p.Arg78Stop; p.Glu36Lys",
        "leigh_mri_rate": 0.20, "cardiac_rate": 0.05, "hepatopathy_rate": 0.15,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.60, "neuropathy_rate": 0.05,
        "cox_activity_mean": 20.0, "cox_activity_sd": 6.0,
        "pindac": True, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "COX6B1", "aa": "86 aa", "kDa": "10.2 kDa",
        "gene_class": "structural_subunit", "subunit_series": "COX6B",
        "civ_module": "IMS-facing peripheral subunit; stabilises CIV monomer → CIV₂ dimer interface; dimer promotes SC I+III₂+IV respirasome assembly; two isoforms: COX6B1 (ubiquitous) / COX6B2 (testis-specific)",
        "omim_gene": 124089, "chromosome": "19q13.13", "seed": 726,
        "phenotype": "Isolated CIV Deficiency — Infantile Encephalomyopathy + Axial Hypotonia + Psychomotor Delay",
        "disease": "COXPD — COX6B1 infantile encephalomyopathy; rare; CIV 10-25% residual; COX6B1 = first nuclear CIV structural subunit mutation (Massa 2008, AJHG)",
        "disease_omim": 220110, "inheritance": "AR",
        "hallmark": "COX6B1 = FIRST nuclear CIV structural subunit mutation ever identified (Massa 2008 AJHG landmark); dimer interface → destabilises CIV₂ → impaired SC I+III₂+IV; isolated CIV; psychomotor delay dominant feature",
        "key_ddx": "vs SURF1: SURF1 → Leigh MRI predominant; COX6B1 → encephalomyopathy without strict Leigh pattern; vs SCO2: COX6B1 no HCM; SCO2 HCM 100%; vs COX6B2: testis-specific isoform different gene",
        "founder_variant": "p.Trp48Stop (European); p.Arg20Stop; p.Glu66Lys",
        "leigh_mri_rate": 0.35, "cardiac_rate": 0.05, "hepatopathy_rate": 0.10,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.70, "neuropathy_rate": 0.20,
        "cox_activity_mean": 17.0, "cox_activity_sd": 5.0,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "COX8A", "aa": "70 aa", "kDa": "7.6 kDa",
        "gene_class": "structural_subunit", "subunit_series": "COX8",
        "civ_module": "Single TM helix; anchors CIV to IMM; interacts with MT-CO1/MT-CO2 transmembrane region; smallest CIV structural subunit by mass; three COX8 isoforms (COX8A ubiquitous / COX8B/C tissue-specific)",
        "omim_gene": 123870, "chromosome": "11q13.1", "seed": 727,
        "phenotype": "Isolated CIV Deficiency — Progressive Encephalomyopathy + Leigh-like MRI + Epilepsy",
        "disease": "COXPD — COX8A-related; isolated CIV deficiency; 5-15% residual; Leigh-like basal ganglia changes; refractory epilepsy common; Hallmann 2016 AJHG",
        "disease_omim": 616501, "inheritance": "AR",
        "hallmark": "SMALLEST nuclear CIV structural subunit (70 aa); refractory epilepsy in >50% differentiates from other CIV genes; single TM helix disruption → global CIV mis-assembly; Hallmann 2016 first COX8A disease report",
        "key_ddx": "vs SURF1-Leigh: SURF1 more common; COX8A rarer; vs NDUFA4: reclassified CIV subunit, similar small size; vs mitochondrial epilepsy DDx: Dravet (SCN1A nuclear), POLG-related; COX8A epilepsy refractory unlike LEV-responsive mtDNA",
        "founder_variant": "p.Arg12Stop (Middle Eastern); p.Leu39Pro (helix-breaking); p.Gly52Val",
        "leigh_mri_rate": 0.55, "cardiac_rate": 0.05, "hepatopathy_rate": 0.10,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.75, "neuropathy_rate": 0.10,
        "cox_activity_mean": 12.0, "cox_activity_sd": 4.0,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "NDUFA4", "aa": "81 aa", "kDa": "9.3 kDa",
        "gene_class": "structural_subunit", "subunit_series": "COX-NDUFA4",
        "civ_module": "IMS-facing single TM helix; contacts MT-CO2 CuA domain; reclassified as 14th CIV subunit (Balsa 2012, Mol Cell) — NOT Complex I despite NDUFA naming; NDUFA4 = only protein in the NDUFA family that is a CIV structural component",
        "omim_gene": 602137, "chromosome": "7p21.3", "seed": 728,
        "phenotype": "CIV Deficiency — Leigh Syndrome COXPD20 (Isolated CIV-Leigh; CI/CII/CIII NORMAL)",
        "disease": "COXPD20 / Leigh syndrome — NDUFA4 LOF; isolated CIV 5-25% residual; NDUFA4 = CIV NOT CI despite NDUFA prefix; Balsa 2012 reclassification landmark",
        "disease_omim": 256000, "inheritance": "AR",
        "hallmark": "NDUFA4 IS CIV NOT CI — Balsa 2012 Mol Cell reclassification (NDUFA name is historically misleading); ONLY nuclear-encoded member of the NDUFA family that belongs to CIV; IMS-facing TM helix contacts COX2 CuA; isolated CIV (CI/CII/CIII normal); SCO2 DDx: SCO2 HCM 100% vs NDUFA4 no HCM",
        "key_ddx": "vs CI deficiency (NDUFA series): NDUFA4 = CIV NOT CI — check CIV not CI enzymatically; vs SCO2: SCO2 → HCM 100%, NDUFA4 → no HCM; vs SURF1: SURF1 more common CIV-Leigh; vs NDUFA5 (7q32.1, same chromosome different arm = CI)",
        "founder_variant": "p.Arg52Cys (TM helix core); p.Leu40Pro (helix-breaking proline); p.Gly61Ser (IMS loop)",
        "leigh_mri_rate": 0.75, "cardiac_rate": 0.05, "hepatopathy_rate": 0.10,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.80, "neuropathy_rate": 0.10,
        "cox_activity_mean": 14.0, "cox_activity_sd": 5.0,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    # ── Assembly Factors ─────────────────────────────────────────────────────
    {
        "gene": "SURF1", "aa": "300 aa", "kDa": "33.0 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-SURF1",
        "civ_module": "IMM-integral; required for haem a3-CuB binuclear centre formation in MT-CO1; early CIV assembly step; most commonly mutated nuclear CIV gene worldwide; Leigh syndrome most frequent presentation",
        "omim_gene": 185620, "chromosome": "9q34.2", "seed": 729,
        "phenotype": "Leigh Syndrome — SURF1-related (most common CIV-Leigh; COXPD1) — CIV 5-20% residual; normal Cyt c1 (CIII normal); normal CII",
        "disease": "COXPD1 / Leigh syndrome — SURF1; isolated CIV; most common nuclear CIV gene causing Leigh; CIV 5-20% residual; first described Zhu 1998 NatGenet + Tiranti 1998 NatGenet simultaneously",
        "disease_omim": 256000, "inheritance": "AR",
        "hallmark": "SURF1 = MOST COMMON nuclear CIV gene causing Leigh syndrome worldwide; haem a3-CuB centre formation → MT-CO1 maturation; NO HCM (unlike SCO2); NO French-Canadian phenotype (unlike LRPPRC); NO copper deficiency (unlike SCO1/SCO2/COA6); c.312_321dup10 (European commonest); Zhu+Tiranti 1998 dual discovery",
        "key_ddx": "vs SCO2: HCM 100% + copper-responsive vs SURF1 no HCM; vs LRPPRC: French-Canadian founder vs SURF1 pan-ethnic; vs TACO1: TACO1 = MT-CO1 mRNA activator vs SURF1 = haem a3 formation; vs BTBGD: MANDATORY exclusion for all Leigh",
        "founder_variant": "c.312_321dup10 (European); c.574-2A>G (splice, European); c.868_869ins (Asian)",
        "leigh_mri_rate": 0.85, "cardiac_rate": 0.10, "hepatopathy_rate": 0.10,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.85, "neuropathy_rate": 0.10,
        "cox_activity_mean": 10.0, "cox_activity_sd": 3.5,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "SCO1", "aa": "301 aa", "kDa": "33.5 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-SCO",
        "civ_module": "IMM-anchored IMS-facing copper metallochaperone (CXXC motif); transfers Cu(I) to CuA centre of MT-CO2; works with SCO2 sequentially (SCO2 → SCO1 → CuA); mutations → hepatic failure + encephalopathy (SCO1) vs HCM+encephalopathy (SCO2)",
        "omim_gene": 603644, "chromosome": "17p13.1", "seed": 730,
        "phenotype": "Isolated CIV Deficiency — Neonatal Hepatic Failure + Ketoacidotic Encephalopathy (COXPD6)",
        "disease": "COXPD6 — SCO1 neonatal hepatic failure; CIV 5-15% residual; copper-responsive in partial deficiency; distinct from SCO2 (no HCM in SCO1); oral CuHis tried",
        "disease_omim": 612245, "inheritance": "AR",
        "hallmark": "HEPATIC FAILURE DOMINANT (unlike SCO2 HCM); neonatal ketoacidotic encephalopathy; CuA copper delivery (works with SCO2 in series: SCO2 first → then SCO1 → CuA site); CXXC thioredoxin fold; copper supplementation (oral CuHis) rescues partial deficiency in vitro; p.Pro174Leu Toronto founder",
        "key_ddx": "vs SCO2: SCO2 → HCM 100%, SCO1 → hepatic failure 60%, NO HCM; vs COA6: COA6 copper delivery via CuA (overlapping mechanism); vs Wilson (ATP7B): hepatic copper overload vs SCO1 copper deficiency at CIV; vs DGUOK: hepatic + neurological mtDNA depletion vs SCO1 isolated CIV",
        "founder_variant": "p.Pro174Leu (Toronto/Canadian founder); p.Ala200Val; p.Gly132Ser",
        "leigh_mri_rate": 0.40, "cardiac_rate": 0.15, "hepatopathy_rate": 0.60,
        "hcm_rate": 0.05, "lactic_ac_rate": 0.80, "neuropathy_rate": 0.20,
        "cox_activity_mean": 11.0, "cox_activity_sd": 4.0,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "SCO2", "aa": "266 aa", "kDa": "29.5 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-SCO",
        "civ_module": "IMM-anchored IMS-facing copper metallochaperone (CXXC motif); delivers Cu(I) to CuB centre of MT-CO1 AND assists SCO1 pathway; SCO2 precedes SCO1 in Cu assembly cascade; heart-enriched expression → HCM 100%",
        "omim_gene": 604272, "chromosome": "22q13.33", "seed": 731,
        "phenotype": "Fatal Infantile Cardioencephalomyopathy + HCM 100% — HIGHEST CARDIAC RATE OF ALL CIV GENES (COXPD2)",
        "disease": "COXPD2 / Fatal Infantile Cardioencephalomyopathy — SCO2; HCM 100% (most cardiac CIV gene); CIV 5-15% residual; copper-responsive (oral CuHis, penicillamine tried); Jaksch 1999 NatGenet first report",
        "disease_omim": 604377, "inheritance": "AR",
        "hallmark": "HCM 100% — HIGHEST CARDIAC RATE OF ALL NUCLEAR CIV GENES; cardioencephalomyopathy fatal in infancy; CuB delivery to MT-CO1; p.Glu140Lys most common (pan-ethnic); copper therapy (oral CuHis) partial rescue; NDUFV2 (CI, HCM 80%) is top DDx — check CIV not CI enzymatically; Jaksch 1999 landmark NatGenet",
        "key_ddx": "vs SCO1: SCO1 hepatic failure dominant, SCO2 HCM 100%; vs NDUFV2 (CI): HCM 80% + ISOLATED CI vs SCO2 HCM 100% isolated CIV; vs Pompe/HCM: GSD vs CIV; vs COA6: COA6 cardiac (infants) but not 100% HCM like SCO2",
        "founder_variant": "p.Glu140Lys (commonest worldwide); p.Gly193Ser; p.Arg90His",
        "leigh_mri_rate": 0.50, "cardiac_rate": 1.00, "hepatopathy_rate": 0.15,
        "hcm_rate": 1.00, "lactic_ac_rate": 0.90, "neuropathy_rate": 0.05,
        "cox_activity_mean": 8.0, "cox_activity_sd": 3.0,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "COX10", "aa": "443 aa", "kDa": "49.8 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-HEME",
        "civ_module": "IMM multi-TM; protoheme IX farnesyltransferase — catalyses first step of haem a biosynthesis (protoheme → haem o); haem a = essential redox centre of MT-CO1 (both haem a + haem a3); COX10 acts before COX15 in heme a pathway",
        "omim_gene": 602125, "chromosome": "17p12", "seed": 732,
        "phenotype": "Isolated CIV Deficiency — Tubulopathy + Leigh-like + Leukodystrophy (COXPD3); some patients: peripheral neuropathy",
        "disease": "COXPD3 — COX10 haem a biosynthesis failure; isolated CIV; tubulopathy (proximal renal tubular acidosis) prominent distinguishing feature; Valnot 2000 AJHG first report",
        "disease_omim": 220110, "inheritance": "AR",
        "hallmark": "TUBULOPATHY (proximal RTA) distinguishes COX10 from other CIV-Leigh — renal tubular involvement in >40%; haem a first step (before COX15 haem a3 step); leukodystrophy in some; Valnot 2000 AJHG landmark; COX10-vs-COX15: COX10 haem o→haem a (earlier), COX15 haem a→haem a3 (later)",
        "key_ddx": "vs COX15: COX15 → cardiac/Leigh no tubulopathy; COX10 → tubulopathy 40%; vs LRPPRC: LRPPRC = French-Canadian; COX10 pan-ethnic; vs BCS1L (CIII-GRACILE): CIII not CIV; vs RTA DDx: Fanconi/cystinosis vs isolated CIV with RTA",
        "founder_variant": "p.Arg339Trp (European/South American); p.Gly177Arg; p.Ala341Val",
        "leigh_mri_rate": 0.55, "cardiac_rate": 0.20, "hepatopathy_rate": 0.15,
        "hcm_rate": 0.05, "lactic_ac_rate": 0.75, "neuropathy_rate": 0.30,
        "cox_activity_mean": 13.0, "cox_activity_sd": 4.5,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "COX15", "aa": "412 aa", "kDa": "45.5 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-HEME",
        "civ_module": "IMM multi-TM; haem a synthase — converts haem o → haem a3 (second step in haem a pathway, after COX10); haem a3-CuB = O₂ reduction binuclear centre; COX15 yeast homologue complemented by human gene",
        "omim_gene": 603646, "chromosome": "10q24.2", "seed": 733,
        "phenotype": "Isolated CIV Deficiency — Hypertrophic Cardiomyopathy + Leigh-like (COXPD4); some patients fatal infantile hypertrophic cardiomyopathy",
        "disease": "COXPD4 — COX15 haem a3 failure; isolated CIV; HCM 40-60% (second most cardiac CIV AF after SCO2); Leigh-like MRI; Antonicka 2003 AJHG first report",
        "disease_omim": 220110, "inheritance": "AR",
        "hallmark": "HCM 40-60% (second most cardiac CIV AF — after SCO2 100%); haem a3-CuB O₂ reduction site formation; COX15 acts AFTER COX10 in heme a pathway (COX10 haem o → haem a, COX15 haem a → haem a3); Antonicka 2003 AJHG; cardiac + Leigh co-occurrence common",
        "key_ddx": "vs SCO2: SCO2 HCM 100%, COX15 HCM 40-60%; vs COX10: tubulopathy in COX10, not COX15; vs MT-CO1 mtDNA: maternal vs AR; vs Pompe: GSD II enzyme vs CIV activity",
        "founder_variant": "p.Arg217Trp (European); p.Gly280Glu (North American); p.Ile294Val",
        "leigh_mri_rate": 0.55, "cardiac_rate": 0.55, "hepatopathy_rate": 0.15,
        "hcm_rate": 0.50, "lactic_ac_rate": 0.80, "neuropathy_rate": 0.10,
        "cox_activity_mean": 12.0, "cox_activity_sd": 4.0,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "COX20", "aa": "113 aa", "kDa": "12.5 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-MT-CO2",
        "civ_module": "IMM-integral early assembly factor; stabilises newly synthesised MT-CO2 preprotein; required for MT-CO2 insertion into IMM before SCO1/COA6 act; COX20 mutations → sub-complex S2 (MT-CO2-containing) fails to mature",
        "omim_gene": 614698, "chromosome": "1p33", "seed": 734,
        "phenotype": "Isolated CIV Deficiency — Progressive Ataxia + Muscle Weakness + Neuropathy (COXPD25); adolescent/adult onset distinguishes from infantile CIV-Leigh",
        "disease": "COXPD25 — COX20 MT-CO2 stabilisation failure; isolated CIV; adolescent/adult onset typical; progressive ataxia + muscle weakness; Doss 2014 Brain first report",
        "disease_omim": 616580, "inheritance": "AR",
        "hallmark": "ADOLESCENT/ADULT ONSET (distinguishes COX20 from infantile CIV-Leigh like SURF1/SCO2); progressive ataxia + neuropathy dominant presentation; MT-CO2 early stabilisation; CIV sub-complex S2 fails; Doss 2014 Brain landmark; SCO1/SCO2 act AFTER COX20 in MT-CO2 maturation pathway",
        "key_ddx": "vs SURF1: SURF1 infantile Leigh; COX20 adolescent ataxia; vs Friedrich ataxia (FXN): mitochondrial iron vs CIV copper pathway; vs COA7: COA7 also ataxia+neuropathy (mild CIV) vs COX20 moderate CIV; vs spinocerebellar ataxias (SCA): dominant vs AR CIV",
        "founder_variant": "p.Arg60Cys (European); p.Gly51Asp; p.Trp64Arg",
        "leigh_mri_rate": 0.20, "cardiac_rate": 0.05, "hepatopathy_rate": 0.05,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.55, "neuropathy_rate": 0.60,
        "cox_activity_mean": 25.0, "cox_activity_sd": 8.0,
        "pindac": False, "scl_neuro": True, "french_canadian": False,
    },
    {
        "gene": "COA3", "aa": "103 aa", "kDa": "11.3 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-MT-CO1",
        "civ_module": "IMM-integral; stabilises MT-CO1 translation product at ribosome; promotes early sub-complex S1 (MT-CO1 seed); works with COX14 and MITRAC12 at MT-CO1 synthesis/insertion; COA3/MITRAC12/COX14 form early MT-CO1 stabilisation complex",
        "omim_gene": 614775, "chromosome": "17q24.2", "seed": 735,
        "phenotype": "Isolated CIV Deficiency — Progressive Encephalomyopathy + Myopathy + Lactic Acidosis (COXPD21)",
        "disease": "COXPD21 — COA3 early MT-CO1 assembly failure; isolated CIV; progressive encephalomyopathy; Clemente 2015 HumMolGenet first report",
        "disease_omim": 616003, "inheritance": "AR",
        "hallmark": "COA3 = EARLY MT-CO1 STABILISATION (sub-complex S1); works with COX14 and MITRAC12 at MT-CO1 translation; COA3/MITRAC12 regulate mtoribosome pausing for MT-CO1 synthesis; progressive encephalomyopathy; hyperlactataemia common; COA3 distinct from COA5/COA6 (MT-CO2 pathway)",
        "key_ddx": "vs TACO1: TACO1 = transcriptional activator of MT-CO1 mRNA (French-Canadian Leigh); COA3 = post-translational MT-CO1 stabilisation; vs COX14: COX14 overlapping early S1 complex; vs SURF1: haem a3 vs MT-CO1 stabilisation",
        "founder_variant": "p.Ala45Val (European); p.Pro78Leu; p.Gly91Ser (splice region)",
        "leigh_mri_rate": 0.45, "cardiac_rate": 0.10, "hepatopathy_rate": 0.10,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.80, "neuropathy_rate": 0.25,
        "cox_activity_mean": 15.0, "cox_activity_sd": 5.0,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "COA5", "aa": "120 aa", "kDa": "13.9 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-CARDIAC",
        "civ_module": "IMM-integral; heart-enriched expression; MT-CO2 maturation pathway (early cardiac-specific CIV assembly); interacts with SCO1/SCO2 pathway for CuA delivery; distinct from COA6 (COA5 = cardiac, COA6 = ubiquitous copper delivery)",
        "omim_gene": 613920, "chromosome": "2q11.2", "seed": 736,
        "phenotype": "Isolated CIV Deficiency — Neonatal Fatal Cardiomyopathy + Cardiac Hypertrophy (COXPD23); cardiac dominant phenotype",
        "disease": "COXPD23 — COA5 cardiac CIV assembly failure; isolated CIV; neonatal cardiac failure; distinct from SCO2 (which also causes HCM); Huigsloot 2011 AJHG first report",
        "disease_omim": 614924, "inheritance": "AR",
        "hallmark": "CARDIAC-DOMINANT (not HCM 100% like SCO2 but cardiac failure prominent in neonates); heart-enriched expression explains cardiac selectivity; COA5 (cardiac) vs COA6 (ubiquitous copper delivery); Huigsloot 2011 AJHG first published COA5 disease; CIV 10-25% residual in cardiac tissue",
        "key_ddx": "vs SCO2: SCO2 HCM 100%; COA5 cardiac failure but not always HCM; vs COA6: COA6 = infant cardiomyopathy + encephalopathy vs COA5 = neonatal cardiac; vs TMEM70 (CV assembly): CV-Leigh vs CIV-Leigh; vs DGUOK: hepatic+cardiac mtDNA depletion vs isolated CIV",
        "founder_variant": "p.Gly13Asp (cardiac early; Netherlands); p.Ala55Val; p.Trp59Stop",
        "leigh_mri_rate": 0.25, "cardiac_rate": 0.80, "hepatopathy_rate": 0.10,
        "hcm_rate": 0.60, "lactic_ac_rate": 0.75, "neuropathy_rate": 0.05,
        "cox_activity_mean": 14.0, "cox_activity_sd": 5.0,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "COA6", "aa": "127 aa", "kDa": "14.2 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-COPPER",
        "civ_module": "IMS-facing; twin CX9C motif (MIA pathway import); delivers Cu(I) to CuA of MT-CO2; interacts with SCO1/SCO2 in CuA copper delivery cascade; COA6 defects → copper deficiency at CIV but not tissue copper depletion",
        "omim_gene": 614772, "chromosome": "1q42.2", "seed": 737,
        "phenotype": "Isolated CIV Deficiency — Infantile Cardiomyopathy + Encephalopathy (COXPD26); copper delivery failure",
        "disease": "COXPD26 — COA6 copper delivery to CuA; isolated CIV; infant cardiomyopathy + encephalopathy; Ghosh 2014 AJHG first report; twin CX9C motif = MIA-imported small IMS protein",
        "disease_omim": 616501, "inheritance": "AR",
        "hallmark": "TWIN CX9C MOTIF (MIA pathway IMS import) — small copper metallochaperone; CuA copper delivery to MT-CO2 (works with SCO1/SCO2); infant cardiomyopathy + encephalopathy; COA6 deficiency = copper deficiency AT CuA SITE (not systemic copper deficiency); Ghosh 2014 AJHG landmark; distinct from Menkes (ATP7A) systemic copper",
        "key_ddx": "vs SCO1: SCO1 hepatic failure dominant vs COA6 cardiomyopathy; vs SCO2: SCO2 HCM 100% vs COA6 HCM 60%; vs Menkes (ATP7A): systemic copper deficiency vs localised CIV-copper defect; vs COA5: COA5 neonatal vs COA6 infant",
        "founder_variant": "p.Trp59Cys (CX9C motif; European); p.Arg18Stop; p.Gly93Asp",
        "leigh_mri_rate": 0.40, "cardiac_rate": 0.70, "hepatopathy_rate": 0.15,
        "hcm_rate": 0.60, "lactic_ac_rate": 0.80, "neuropathy_rate": 0.10,
        "cox_activity_mean": 13.0, "cox_activity_sd": 4.5,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "COA7", "aa": "231 aa", "kDa": "26.0 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-LATE",
        "civ_module": "IMS-facing; ARM/SEL1-repeat scaffold protein; no TM helices; late-stage CIV assembly (after sub-complex S3 formed); interacts with assembled CIV monomers; mutations → MILD CIV deficiency (30-60% residual) explains later/milder onset",
        "omim_gene": 615623, "chromosome": "6q25.3", "seed": 738,
        "phenotype": "ONLY CIV Gene Causing Spinocerebellar Ataxia + Axonal Neuropathy WITHOUT Leigh/Encephalopathy (COXPD16)",
        "disease": "COXPD16 — COA7 late-stage CIV assembly; MILD CIV 30-60% residual; unique phenotype: spinocerebellar ataxia + axonal neuropathy dominant; NO Leigh MRI; adolescent/adult onset; Chung 2015 Brain first report",
        "disease_omim": 616838, "inheritance": "AR",
        "hallmark": "ONLY CIV gene causing spinocerebellar ataxia + axonal neuropathy without Leigh/encephalopathy — unique CIV phenotype; mild CIV deficiency (30-60% residual) explains later/milder presentation; adolescent/adult onset; ARM/SEL1 scaffold; NO HCM; Chung 2015 Brain landmark; COA7 vs COX20: both ataxia-neuropathy CIV AF but COA7 = late-stage mild vs COX20 = MT-CO2 stabilisation",
        "key_ddx": "vs Friedrich ataxia: FXN iron-sulfur vs CIV; vs CMT2: axonal CMT2 nuclear vs CIV-COA7; vs COX20: similar ataxia-neuropathy, different CIV step; vs POLG-related: CPEO+ataxia vs CIV; vs MitoChip: require CIV assay + SCA panel simultaneously",
        "founder_variant": "p.Tyr137Cys (ARM core); p.Arg157Trp; p.Ala194Thr",
        "leigh_mri_rate": 0.05, "cardiac_rate": 0.05, "hepatopathy_rate": 0.05,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.40, "neuropathy_rate": 0.85,
        "cox_activity_mean": 42.0, "cox_activity_sd": 10.0,
        "pindac": False, "scl_neuro": True, "french_canadian": False,
    },
    {
        "gene": "TACO1", "aa": "472 aa", "kDa": "52.8 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-MT-CO1",
        "civ_module": "Matrix; translational activator of MT-CO1 mRNA — promotes MT-CO1 ribosome translation initiation/elongation; TACO1 mutations → MT-CO1 mRNA not translated → no S1 sub-complex → complete CIV assembly failure; distinct from LRPPRC (MT-CO1 mRNA stability)",
        "omim_gene": 612958, "chromosome": "17q23.3", "seed": 739,
        "phenotype": "Isolated CIV Deficiency — Leigh Syndrome + Ataxia + Late-Onset (relative to other CIV-Leigh) (COXPD7)",
        "disease": "COXPD7 — TACO1 MT-CO1 translational activator failure; isolated CIV; Leigh + ataxia; Weraarpachai 2009 NatGenet first report; some patients later-onset than SURF1",
        "disease_omim": 613990, "inheritance": "AR",
        "hallmark": "TACO1 = TRANSLATIONAL ACTIVATOR of MT-CO1 (distinct from LRPPRC = mRNA stability); TACO1 loss → no MT-CO1 protein → complete CIV collapse; Leigh + ataxia combination relatively specific; Weraarpachai 2009 NatGenet landmark; COA3 acts POST-TRANSLATIONALLY (stabilises MT-CO1 after synthesis) while TACO1 acts PRE-TRANSLATIONALLY (promotes MT-CO1 mRNA translation)",
        "key_ddx": "vs LRPPRC: LRPPRC = French-Canadian + MT-CO1 mRNA stability (both affect MT-CO1); TACO1 pan-ethnic + translational activation; vs COA3: COA3 post-translational, TACO1 pre-translational; vs SURF1: haem a3 vs MT-CO1 translation; vs BTBGD: MANDATORY exclusion",
        "founder_variant": "p.Arg215Pro (European founder); p.Lys147Arg; p.Ala240Val",
        "leigh_mri_rate": 0.70, "cardiac_rate": 0.10, "hepatopathy_rate": 0.15,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.80, "neuropathy_rate": 0.20,
        "cox_activity_mean": 11.0, "cox_activity_sd": 4.0,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "LRPPRC", "aa": "1394 aa", "kDa": "157.8 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-MT-RNA",
        "civ_module": "Mitochondrial RNA-binding protein (PPR repeats); stabilises MT-CO1 and MT-CO3 mRNA (with SLIRP); promotes polyadenylation of mitochondrial transcripts; LRPPRC mutations → defective MT-CO1/CO3 mRNA polyadenylation → CIV assembly failure",
        "omim_gene": 607544, "chromosome": "2p21", "seed": 740,
        "phenotype": "French-Canadian Leigh Syndrome (Leigh Syndrome, French-Canadian Type / LSFC) — CIV 20-30% residual; p.Ala354Val founder mutation 1:23 carriers in Saguenay-Lac-Saint-Jean region",
        "disease": "LSFC / French-Canadian Leigh Syndrome — LRPPRC; CIV 20-30% (intermediate residual vs 5-10% in SURF1); MT-CO1+CO3 mRNA stabilisation failure; Morin 2003 NatGenet founding paper; >200 patients Saguenay-Lac-Saint-Jean Quebec",
        "disease_omim": 220111, "inheritance": "AR",
        "hallmark": "FRENCH-CANADIAN LEIGH SYNDROME (LSFC) — p.Ala354Val founder mutation; Saguenay-Lac-Saint-Jean Quebec region 1:23 carrier; LRPPRC = mRNA polyadenylation anchor (with SLIRP); affects MT-CO1 + MT-CO3 stability → combined CIV loss; intermediate CIV residual (20-30%) explains episodic metabolic crises with periods of stability; Morin 2003 NatGenet landmark",
        "key_ddx": "vs TACO1: TACO1 = translational activation, LRPPRC = mRNA stability (both affect MT-CO1 mRNA); vs SURF1: SURF1 pan-ethnic Leigh; LRPPRC French-Canadian; vs MT-CO1 mtDNA: maternal vs AR; vs MT-RNR2-related: combined OXPHOS vs isolated CIV",
        "founder_variant": "p.Ala354Val (French-Canadian LSFC founder); p.Arg636Stop; p.Gly878Ala",
        "leigh_mri_rate": 0.80, "cardiac_rate": 0.15, "hepatopathy_rate": 0.20,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.85, "neuropathy_rate": 0.10,
        "cox_activity_mean": 24.0, "cox_activity_sd": 7.0,
        "pindac": False, "scl_neuro": False, "french_canadian": True,
    },
    {
        "gene": "PET100", "aa": "71 aa", "kDa": "8.0 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-LATE",
        "civ_module": "IMM-anchored; very small late-stage CIV assembly factor; interacts with partially assembled CIV intermediates (sub-complex S3); stabilises CIV maturation after sub-complex S2 (MT-CO1+MT-CO2) forms; required for full CIV₄ monomer maturation",
        "omim_gene": 614770, "chromosome": "19p13.3", "seed": 741,
        "phenotype": "Isolated CIV Deficiency — Fatal Infantile Hypertrophic Cardiomyopathy + Leigh-like (COXPD25 subset)",
        "disease": "COXPD — PET100 late CIV assembly; fatal infantile; CIV 5-10% residual; Lim 2014 AJHG first report; overlapping phenotype with SURF1 Leigh but different assembly step",
        "disease_omim": 614771, "inheritance": "AR",
        "hallmark": "SMALLEST CIV ASSEMBLY FACTOR (71 aa); late-stage sub-complex S3 stabilisation; fatal infantile course; Lim 2014 AJHG landmark; PET100 acts AFTER MT-CO1+MT-CO2 sub-complex S2 forms — LATER STEP than COA3/COX14 (early S1) and COX20 (early MT-CO2 stabilisation); severe isolated CIV <10% residual",
        "key_ddx": "vs SURF1: SURF1 = haem a3 (early) vs PET100 = late S3 stabilisation; vs COX8A (smallest structural subunit): structural vs assembly; vs COA3: early vs late CIV stage; vs FASTKD2: RNA processing vs assembly",
        "founder_variant": "p.Leu4Pro (IMM anchor helix; Australian); p.Ser44Arg; p.Gly62Val",
        "leigh_mri_rate": 0.65, "cardiac_rate": 0.65, "hepatopathy_rate": 0.10,
        "hcm_rate": 0.55, "lactic_ac_rate": 0.85, "neuropathy_rate": 0.05,
        "cox_activity_mean": 9.0, "cox_activity_sd": 3.0,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "COX14", "aa": "100 aa", "kDa": "11.2 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-MT-CO1",
        "civ_module": "IMM-integral; earliest MT-CO1 sub-complex S1 stabilisation; works with COA3 and MITRAC12 in the MITRAC (Mitochondrial Translation Regulation Assembly intermediate of Cytochrome c oxidase) complex; regulates nascent MT-CO1 translation product cotranslationally",
        "omim_gene": 614478, "chromosome": "12q13.3", "seed": 742,
        "phenotype": "Isolated CIV Deficiency — Infantile Encephalomyopathy + Lactic Acidosis (COXPD22)",
        "disease": "COXPD22 — COX14 early MT-CO1 MITRAC complex failure; isolated CIV; infantile encephalomyopathy; Weraarpachai 2012 NatGenet first report",
        "disease_omim": 604272, "inheritance": "AR",
        "hallmark": "COX14 = MITRAC COMPLEX COMPONENT (with COA3/MITRAC12) — co-translational MT-CO1 stabilisation; EARLIEST CIV assembly step; COX14 + COA3 = complementary early S1 stabilisers (COX14 slightly earlier than COA3); infantile presentation typical; Weraarpachai 2012 NatGenet landmark",
        "key_ddx": "vs COA3: overlapping early S1 MT-CO1 module; COX14 slightly earlier; both infantile; vs TACO1: COX14 post-translational vs TACO1 translational activation; vs SURF1: haem a3 late vs MITRAC early MT-CO1; vs COA5/COA6: MT-CO2 pathway vs MT-CO1 pathway",
        "founder_variant": "p.Arg50Cys (MITRAC contact; European); p.Trp72Stop; p.Gly88Asp",
        "leigh_mri_rate": 0.50, "cardiac_rate": 0.10, "hepatopathy_rate": 0.10,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.80, "neuropathy_rate": 0.20,
        "cox_activity_mean": 14.0, "cox_activity_sd": 4.5,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
    {
        "gene": "FASTKD2", "aa": "709 aa", "kDa": "79.8 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-MT-RNA",
        "civ_module": "Mitochondrial RNA granule; FAST kinase domain protein; processes mitochondrial polycistronic RNA at MT-CO1/CO2/CO3 junctions; required for mature MT-CO1, MT-CO2, and MT-CO3 mRNA production → isolated CIV deficiency; FASTK family (FASTK/FASTKD1-5)",
        "omim_gene": 612322, "chromosome": "2q33.1", "seed": 743,
        "phenotype": "Isolated CIV Deficiency — Infantile Encephalomyopathy + Sensorineural Hearing Loss (COXPD29); RNA processing pathway",
        "disease": "COXPD29 — FASTKD2 mitochondrial RNA processing failure at MT-CO junctions; isolated CIV; infantile; SNHL 40% unique among CIV genes; Ghezzi 2008 AJHG first report",
        "disease_omim": 616478, "inheritance": "AR",
        "hallmark": "FASTKD2 = MITOCHONDRIAL RNA PROCESSING at MT-CO1/CO2/CO3 junctions; SNHL 40% (unique CIV gene feature); FASTK kinase domain + RAP domain (RNA-binding); FASTK family (FASTK/FASTKD1-5) coordinates mitoribosome assembly and mt-RNA maturation; Ghezzi 2008 AJHG landmark; LRPPRC = mRNA stability vs FASTKD2 = RNA processing/junction cleavage",
        "key_ddx": "vs LRPPRC: LRPPRC = mRNA polyadenylation stability; FASTKD2 = polycistronic RNA junction processing; vs TACO1: translational activation vs RNA processing; vs MT-RNR1 (AISNHL): AISNHL aminoglycoside-triggered vs FASTKD2 SNHL without aminoglycosides; vs GJB2 (DFNB1): AR SNHL genetic DDx",
        "founder_variant": "p.Arg448Stop (European); p.Gly509Val; p.Leu234Pro (FAST kinase domain)",
        "leigh_mri_rate": 0.45, "cardiac_rate": 0.10, "hepatopathy_rate": 0.10,
        "hcm_rate": 0.00, "lactic_ac_rate": 0.70, "neuropathy_rate": 0.20,
        "cox_activity_mean": 15.0, "cox_activity_sd": 5.0,
        "pindac": False, "scl_neuro": False, "french_canadian": False,
    },
]

# ── Cohort generation ─────────────────────────────────────────────────────────
def _make_cohort():
    cohort = []
    for gene_info in CIV_GENES:
        rng_gene = random.Random(gene_info["seed"])
        for pt_idx in range(40):
            # Age of onset
            if gene_info["gene"] in ("COX20", "COA7"):
                onset_months = int(rng_gene.gauss(144, 48))   # adolescent/adult
            elif gene_info["gene"] in ("LRPPRC",):
                onset_months = int(rng_gene.gauss(8, 4))      # infantile but episodic
            elif gene_info["gene"] in ("COX4I1",):
                onset_months = int(rng_gene.gauss(3, 2))      # neonatal-infantile
            else:
                onset_months = int(rng_gene.gauss(6, 4))      # classic infantile
            onset_months = max(0, min(onset_months, 600))

            leigh_mri       = rng_gene.random() < gene_info["leigh_mri_rate"]
            cardiac         = rng_gene.random() < gene_info["cardiac_rate"]
            hcm             = rng_gene.random() < gene_info["hcm_rate"]
            hepatopathy     = rng_gene.random() < gene_info["hepatopathy_rate"]
            neuropathy      = rng_gene.random() < gene_info["neuropathy_rate"]
            lactic_ac       = rng_gene.random() < gene_info["lactic_ac_rate"]
            pindac          = rng_gene.random() < (0.90 if gene_info["pindac"] else 0.00)
            scl_neuro       = rng_gene.random() < (0.70 if gene_info["scl_neuro"] else 0.02)
            french_canadian = rng_gene.random() < (0.85 if gene_info["french_canadian"] else 0.00)

            cox_activity_pct = max(3.0, min(
                rng_gene.gauss(gene_info["cox_activity_mean"], gene_info["cox_activity_sd"]),
                95.0))

            cohort.append({
                "gene":              gene_info["gene"],
                "gene_class":        gene_info["gene_class"],
                "patient_id":        f"{gene_info['gene']}-{pt_idx+1:02d}",
                "seed":              gene_info["seed"],
                "age_onset_months":  onset_months,
                "leigh_mri":         leigh_mri,
                "cardiomyopathy":    cardiac,
                "hcm":               hcm,
                "hepatopathy":       hepatopathy,
                "peripheral_neuropathy": neuropathy,
                "lactic_acidosis":   lactic_ac,
                "pindac_syndrome":   pindac,
                "scl_ataxia_neuropathy": scl_neuro,
                "french_canadian_leigh": french_canadian,
                "cox_activity_pct":  round(cox_activity_pct, 1),
            })
    return cohort

COHORT = _make_cohort()


# ── Public API functions ──────────────────────────────────────────────────────
def get_overview():
    n_patients = len(COHORT)
    n_struct   = sum(1 for g in CIV_GENES if g["gene_class"] == "structural_subunit")
    n_af       = sum(1 for g in CIV_GENES if g["gene_class"] == "assembly_factor")

    leigh_pts   = [p for p in COHORT if p["leigh_mri"]]
    cardiac_pts = [p for p in COHORT if p["cardiomyopathy"]]
    hcm_pts     = [p for p in COHORT if p["hcm"]]
    hepato_pts  = [p for p in COHORT if p["hepatopathy"]]
    neuro_pts   = [p for p in COHORT if p["peripheral_neuropathy"]]
    lactic_pts  = [p for p in COHORT if p["lactic_acidosis"]]
    pindac_pts  = [p for p in COHORT if p["pindac_syndrome"]]
    scl_pts     = [p for p in COHORT if p["scl_ataxia_neuropathy"]]
    fc_pts      = [p for p in COHORT if p["french_canadian_leigh"]]

    return {
        "atlas":            "CIV-Subunit-Atlas",
        "title":            "Complete 19-Gene Nuclear-Encoded Complex IV (Cytochrome c Oxidase) Atlas",
        "complex":          "Complex IV (CIV) — Cytochrome c Oxidase",
        "function":         "Terminal ETC electron acceptor: reduces O₂ → H₂O; pumps 4H⁺/2e⁻ across IMM; ~40% of mitochondrial proton gradient",
        "n_genes":          len(CIV_GENES),
        "n_structural_subunits": n_struct,
        "n_assembly_factors":    n_af,
        "n_patients":       n_patients,
        "cohort_formula":   f"{len(CIV_GENES)} genes × 40 patients = {n_patients} patient slots (seeds 725–743)",
        "mtDNA_subunits":   "MT-CO1 + MT-CO2 + MT-CO3 — WES MISSES all 3; covered in MT-Genome-Atlas + individual dashboards",
        "reclassification": "NDUFA4 — reclassified as 14th nuclear CIV structural subunit (Balsa 2012 Mol Cell); historically misclassified as CI; NDUFA4 = ONLY NDUFA-family protein belonging to CIV",
        "cii_always_normal": "CII ALWAYS NORMAL in isolated CIV deficiency — internal biochemical reference (no mtDNA CII subunits)",
        "gene_list": {
            "structural_subunits_4_nuclear": ["COX4I1", "COX6B1", "COX8A", "NDUFA4"],
            "assembly_factors_15_nuclear": [
                "SURF1", "SCO1", "SCO2", "COX10", "COX15", "COX20",
                "COA3", "COA5", "COA6", "COA7", "TACO1", "LRPPRC",
                "PET100", "COX14", "FASTKD2"
            ],
            "mtDNA_subunits_not_in_atlas": ["MT-CO1", "MT-CO2", "MT-CO3"],
        },
        "copper_centres": {
            "CuA": "Binuclear (2 Cu atoms) in MT-CO2 IMS domain — primary electron acceptor from Cyt c; assembly via SCO1+SCO2+COA6",
            "CuB": "Mononuclear in MT-CO1 — O₂ reduction with haem a3; assembly via SCO2",
        },
        "heme_centres": {
            "haem_a":  "MT-CO1 — electron relay (low potential); synthesised COX10 (haem o → haem a)",
            "haem_a3": "MT-CO1 — O₂ reduction binuclear centre with CuB; synthesised COX15 (haem a → haem a3); SURF1 inserts haem a3 into MT-CO1",
        },
        "hallmark_phenotypes": {
            "Leigh_Syndrome":    {"gene": "SURF1", "note": "Most common nuclear CIV-Leigh; haem a3 formation; pan-ethnic; c.312_321dup10 European commonest; NO HCM"},
            "HCM_100pct":        {"gene": "SCO2",  "note": "Highest cardiac CIV gene; copper metallochaperone; HCM 100%; p.Glu140Lys; Jaksch 1999 NatGenet"},
            "French_Canadian":   {"gene": "LRPPRC","note": "LSFC — p.Ala354Val Saguenay-Lac-Saint-Jean 1:23 carrier; MT-CO1+CO3 mRNA stability; intermediate CIV 20-30%"},
            "PINDAC_Triad":      {"gene": "COX4I1","note": "EPI + Dyserythropoietic anaemia + Calvarial hyperostosis; Bedouin 16q22 deletion; ATP allosteric site subunit"},
            "SCA_Neuropathy":    {"gene": "COA7",  "note": "ONLY CIV gene: spinocerebellar ataxia + axonal neuropathy WITHOUT Leigh; mild CIV 30-60%; adolescent onset"},
            "Hepatic_Dominant":  {"gene": "SCO1",  "note": "Hepatic failure 60% + encephalopathy; copper CuA delivery (with SCO2); NO HCM; p.Pro174Leu Toronto founder"},
            "SNHL_CIV":          {"gene": "FASTKD2","note": "SNHL 40% — unique among CIV genes; mt-RNA processing at MT-CO1/CO2/CO3 junctions"},
            "Tubulopathy":       {"gene": "COX10", "note": "Proximal RTA (renal tubular acidosis) 40% — distinguishes COX10 from other CIV-Leigh; haem a biosynthesis step 1"},
            "COX6B1_First":      {"gene": "COX6B1","note": "First nuclear CIV structural subunit mutation identified (Massa 2008 AJHG landmark)"},
        },
        "aggregate_clinical": {
            "leigh_mri_pct":    round(len(leigh_pts)   / n_patients * 100, 1),
            "cardiac_pct":      round(len(cardiac_pts) / n_patients * 100, 1),
            "hcm_pct":          round(len(hcm_pts)     / n_patients * 100, 1),
            "hepatopathy_pct":  round(len(hepato_pts)  / n_patients * 100, 1),
            "neuropathy_pct":   round(len(neuro_pts)   / n_patients * 100, 1),
            "lactic_ac_pct":    round(len(lactic_pts)  / n_patients * 100, 1),
            "pindac_pct":       round(len(pindac_pts)  / n_patients * 100, 1),
            "scl_ataxia_pct":   round(len(scl_pts)     / n_patients * 100, 1),
            "french_canadian_pct": round(len(fc_pts)   / n_patients * 100, 1),
            "mean_cox_activity_pct": round(
                sum(p["cox_activity_pct"] for p in COHORT) / n_patients, 1),
        },
        "drug_contraindications": {
            "absolute_ci_all_19_genes": [
                {"drug": "Propofol",        "mechanism": "DIRECT CIV INHIBITOR — binds MT-CO1 active site; PRIS (Propofol Infusion Syndrome); most relevant CIV absolute CI; AVOID in ALL CIV disorders"},
                {"drug": "VPA / Valproate", "mechanism": "CoA sequestration → impairs OXPHOS substrate supply; mitochondrial toxicity; secondary CIV uncoupling"},
                {"drug": "Metformin",       "mechanism": "CI inhibition → NADH accumulation → reduced Cyt c availability → secondary CIV substrate starvation"},
                {"drug": "Linezolid",       "mechanism": "Mitoribosome inhibition → depletes MT-CO1/CO2/CO3 → CIV assembly failure"},
                {"drug": "Chloramphenicol", "mechanism": "Mitoribosome inhibition → secondary OXPHOS deficiency; acute CIV collapse risk"},
                {"drug": "KD-CAUTION",      "mechanism": "Ketogenic diet: contraindicated in SURF1 (worsens CIV Leigh); relative CI in most CIV genes; substrate shift impairs residual CIV"},
            ],
            "copper_therapy": [
                "Oral CuHis (copper-histidine, 50-150 μg/kg/day) — partial rescue in SCO1, SCO2, COA6 (copper delivery AFs); CuA/CuB copper restoration",
                "Penicillamine trial (copper chelation/mobilisation) — evidence limited; specialist guidance required",
                "Monitor serum ceruloplasmin + free copper (SCO1/SCO2/COA6 specific); NOT for SURF1/COX10/COX15 (different mechanism)",
            ],
            "mandatory_workup": [
                "BTBGD/SLC19A3 MANDATORY exclusion — Leigh/leukoencephalopathy mimic; Biotin+Thiamine responsive (life-saving)",
                "Biotin + Thiamine empiric before CIV diagnosis confirmed (BTBGD exclusion)",
                "GIR 6-8 mg/kg/min — prevent fasting; mandatory glucose infusion in metabolic crisis",
                "Levetiracetam (LEV) preferred AED — renal clearance, no CYP450, no mito toxicity",
                "PERT (Pancreatic Enzyme Replacement Therapy) MANDATORY in COX4I1-PINDAC (exocrine pancreatic insufficiency)",
                "Fat-soluble vitamins ADEK MANDATORY in COX4I1-PINDAC (malabsorption secondary to EPI)",
                "Annual echocardiogram MANDATORY in SCO2, COX15, COA5, COA6, PET100 — HCM surveillance",
                "Audiology surveillance MANDATORY in FASTKD2 (SNHL 40%) and COX4I1 (check for dyserythropoiesis)",
                "Thiamine B1 + Biotin empiric — universal co-supplementation until BTBGD excluded",
            ],
        },
        "wes_utility": {
            "nuclear_genes_detectable": "All 19 nuclear CIV genes WES-detectable (4 structural + 15 assembly factors)",
            "mtDNA_missed":             "MT-CO1, MT-CO2, MT-CO3 (3 mtDNA CIV subunits) — WES MISSES; require dedicated mtDNA panel or long-read sequencing",
            "panel_note":               "Mitochondrial disease gene panels preferred for clinical diagnosis; WES comprehensive but misses mtDNA and large deletions; copper (SCO1/SCO2/COA6) and haem (COX10/COX15) assays complement WES",
            "enzymatic_distinction":    "CIV (COX) activity assay distinguishes CIV deficiency from CI/CI+IV/combined; critical before attributing NDUFA4 mutations to CI (it is CIV not CI — Balsa 2012)",
        },
    }


def get_breakdown():
    rows = []
    for g in CIV_GENES:
        pts = [p for p in COHORT if p["gene"] == g["gene"]]
        leigh_pct  = round(sum(1 for p in pts if p["leigh_mri"]) / len(pts) * 100, 1)
        cardiac_pct= round(sum(1 for p in pts if p["cardiomyopathy"]) / len(pts) * 100, 1)
        hcm_pct    = round(sum(1 for p in pts if p["hcm"]) / len(pts) * 100, 1)
        hepato_pct = round(sum(1 for p in pts if p["hepatopathy"]) / len(pts) * 100, 1)
        neuro_pct  = round(sum(1 for p in pts if p["peripheral_neuropathy"]) / len(pts) * 100, 1)
        lactic_pct = round(sum(1 for p in pts if p["lactic_acidosis"]) / len(pts) * 100, 1)
        pindac_pct = round(sum(1 for p in pts if p["pindac_syndrome"]) / len(pts) * 100, 1)
        scl_pct    = round(sum(1 for p in pts if p["scl_ataxia_neuropathy"]) / len(pts) * 100, 1)
        fc_pct     = round(sum(1 for p in pts if p["french_canadian_leigh"]) / len(pts) * 100, 1)
        mean_cox   = round(sum(p["cox_activity_pct"] for p in pts) / len(pts), 1)
        median_onset = sorted(p["age_onset_months"] for p in pts)[len(pts)//2]

        rows.append({
            "gene":              g["gene"],
            "gene_class":        g["gene_class"],
            "subunit_series":    g["subunit_series"],
            "civ_module":        g["civ_module"][:120],
            "omim_gene":         g["omim_gene"],
            "disease_omim":      g["disease_omim"],
            "chromosome":        g["chromosome"],
            "seed":              g["seed"],
            "n_patients":        len(pts),
            "phenotype":         g["phenotype"],
            "inheritance":       g["inheritance"],
            "median_onset_months":    median_onset,
            "cox_activity_mean_pct":  mean_cox,
            "leigh_mri_pct":     leigh_pct,
            "cardiac_pct":       cardiac_pct,
            "hcm_pct":           hcm_pct,
            "hepatopathy_pct":   hepato_pct,
            "neuropathy_pct":    neuro_pct,
            "lactic_acidosis_pct": lactic_pct,
            "pindac_pct":        pindac_pct,
            "scl_ataxia_pct":    scl_pct,
            "french_canadian_pct": fc_pct,
            "hallmark":          g["hallmark"][:130],
            "founder_variant":   g["founder_variant"],
        })
    return {"genes": rows, "total": len(rows), "total_patients": len(COHORT)}


def get_definitions():
    return {
        "atlas":              "CIV-Subunit-Atlas — Complete 19-gene nuclear-encoded Complex IV reference (4 structural subunits + 15 assembly factors)",
        "complex_iv":         "Complex IV (CIV) / Cytochrome c Oxidase (COX): terminal ETC enzyme; reduces O₂ → 2H₂O; transfers electrons from 4 reduced Cyt c to O₂; pumps 4H⁺/2e⁻ across IMM; 14 nuclear + 3 mtDNA subunits total; heteromeric complex assembled in ordered sub-complex stages",
        "NDUFA4_reclassified":"NDUFA4 is CIV NOT CI — Balsa 2012 Mol Cell reclassification; IMS-facing TM helix contacts MT-CO2 CuA; only NDUFA-family protein in CIV; enzymatic assay MUST confirm CIV deficiency in NDUFA4 mutations (CI will be NORMAL)",
        "CII_always_normal":  "CII ALWAYS NORMAL in isolated CIV deficiency — no mtDNA-encoded CII subunits; CII = internal biochemical reference for all OXPHOS panels",
        "mtDNA_CIV_missed":   "MT-CO1, MT-CO2, MT-CO3 = 3 mtDNA CIV subunits; WES MISSES all 3; require dedicated mtDNA panel; covered in MT-Genome-Atlas and individual MT-CO1/CO2/CO3 dashboards",
        "copper_centres":     "CuA (binuclear, MT-CO2 IMS) assembled by SCO1+SCO2+COA6 in sequence; CuB (mononuclear, MT-CO1) assembled by SCO2; copper therapy (oral CuHis) rescues partial deficiency in SCO1/SCO2/COA6 mutations",
        "heme_pathway":       "Haem a biosynthesis: protoheme → haem o (COX10, farnesyltransferase) → haem a (COX15, haem a synthase) → haem a3-CuB binuclear centre formation (SURF1 insertion into MT-CO1); COX10 acts before COX15",
        "MT_CO1_pathway":     "MT-CO1 stabilisation cascade: TACO1 (translational activation) → COX14+COA3+MITRAC12 (MITRAC complex, cotranslational stabilisation) → sub-complex S1; COA3 overlaps with COX14 (MITRAC) at early S1 stage",
        "MT_CO2_pathway":     "MT-CO2 maturation: COX20 (early stabilisation, sub-complex pre-S2) → SCO2 (CuB delivery) → SCO1+COA6 (CuA delivery) → sub-complex S2 (MT-CO1+MT-CO2); COA5 (cardiac-specific early MT-CO2)",
        "late_assembly":      "Late CIV assembly: sub-complex S3 formation + PET100 + COA7 (late ARM scaffold) → mature CIV₄ monomer → CIV₂ dimer (COX6B1 at dimer interface) → SC I+III₂+IV respirasome",
        "SURF1_landmark":     "SURF1 (300 aa, 9q34.2) — most common nuclear CIV gene causing Leigh syndrome; identified simultaneously by Zhu (1998, NatGenet) + Tiranti (1998, NatGenet); haem a3-CuB formation; NO HCM; c.312_321dup10 European commonest",
        "SCO2_HCM":           "SCO2 (266 aa, 22q13.33) — HCM 100%; highest cardiac rate of ALL nuclear CIV genes; copper (CuB) delivery to MT-CO1; p.Glu140Lys pan-ethnic; Jaksch 1999 NatGenet; cardioencephalomyopathy fatal infantile without copper therapy",
        "LRPPRC_LSFC":        "LRPPRC (1394 aa, 2p21) — French-Canadian Leigh Syndrome (LSFC); p.Ala354Val founder Saguenay-Lac-Saint-Jean Quebec 1:23 carrier; MT-CO1+CO3 mRNA polyadenylation stability (with SLIRP); intermediate CIV 20-30%; Morin 2003 NatGenet",
        "COX4I1_PINDAC":      "COX4I1 (169 aa, 16q22.1) — PINDAC triad pathognomonic: Exocrine Pancreatic Insufficiency + Dyserythropoietic anaemia + Calvarial hyperostosis; Bedouin founder 16q22 deletion; ATP/ADP allosteric site; PERT + ADEK mandatory; COXPD12",
        "COA7_SCA_neuropathy":"COA7 (231 aa, 6q25.3) — ONLY CIV gene causing spinocerebellar ataxia + axonal neuropathy WITHOUT Leigh; mild CIV 30-60% residual; ARM/SEL1 scaffold; adolescent/adult onset; NO HCM; Chung 2015 Brain; COXPD16",
        "SCO1_hepatic":       "SCO1 (301 aa, 17p13.1) — hepatic failure 60% dominant (unlike SCO2 cardiac); neonatal ketoacidotic encephalopathy; p.Pro174Leu Toronto/Canadian founder; copper CuA delivery (sequential with SCO2); COXPD6",
        "COX10_tubulopathy":  "COX10 (443 aa, 17p12) — proximal RTA (tubulopathy) 40% distinguishing feature; haem o→haem a step 1 of haem a biosynthesis; Valnot 2000 AJHG; tubulopathy distinguishes COX10 from all other CIV-Leigh; COXPD3",
        "COA6_copper":        "COA6 (127 aa, 1q42.2) — twin CX9C motif (MIA-imported IMS protein); CuA copper delivery to MT-CO2 (with SCO1/SCO2); infant cardiomyopathy + encephalopathy; copper deficiency at CuA (not systemic); Ghosh 2014 AJHG; COXPD26",
        "FASTKD2_SNHL":       "FASTKD2 (709 aa, 2q33.1) — SNHL 40% (unique among CIV genes); mt-RNA polycistronic cleavage at MT-CO1/CO2/CO3 junctions; FASTK family RNA granule; Ghezzi 2008 AJHG; COXPD29; LRPPRC (mRNA stability) vs FASTKD2 (RNA processing/cleavage)",
        "COX6B1_first":       "COX6B1 (86 aa, 19q13.13) — FIRST nuclear CIV structural subunit mutation identified in human disease (Massa 2008 AJHG); CIV dimer interface (promotes CIV₂); encephalomyopathy; COXPD",
        "propofol_abs_ci":    "Propofol ABSOLUTE CI in ALL CIV disorders — direct CIV (MT-CO1) inhibitor; PRIS (Propofol Infusion Syndrome); life-threatening; use ketamine or dexmedetomidine as anaesthesia alternatives",
        "vpa_ci":             "VPA absolute CI in ALL CIV disorders — CoA sequestration + mito toxicity; LEV preferred AED",
        "metformin_ci":       "Metformin absolute CI — CI inhibition → impaired Cyt c availability → secondary CIV substrate failure",
        "BTBGD_exclusion":    "SLC19A3 (BTBGD) MANDATORY exclusion before diagnosing any CIV-Leigh or leukoencephalopathy — Leigh mimic treatable with Biotin+Thiamine (life-saving dramatic response)",
        "WES_nuclear_CIV":    "All 19 nuclear CIV genes WES-detectable; MT-CO1/CO2/CO3 (mtDNA) NOT WES-detectable; enzymatic CIV assay essential to distinguish CI (in NDUFA4 mis-attribution) from CIV deficiency",
    }
