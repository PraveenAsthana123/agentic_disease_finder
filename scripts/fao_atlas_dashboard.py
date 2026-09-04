#!/usr/bin/env python3
"""FAO-Atlas — Complete 10-Gene Mitochondrial Fatty Acid Oxidation Disorders Atlas
ACADM · ACADVL · HADHA · HADHB · HADH · ACADS · CPT1A · CPT2 · SLC25A20 · SLC22A5
400-patient aggregate cohort (10 × 40, seeds 813–822)

Mitochondrial Fatty Acid Oxidation (FAO) facts:
  - Mitochondrial FAO provides the primary fuel during fasting, prolonged exercise, and KD
  - Beta-oxidation cycle: acyl-CoA → 2,3-enoyl-CoA → 3-hydroxyacyl-CoA → 3-ketoacyl-CoA → acyl-CoA(n-2) + acetyl-CoA
  - Chain-length specificity: Very Long Chain (≥C14): VLCAD (ACADVL); Long Chain: MTP (HADHA/HADHB);
    Medium Chain (C6–C12): MCAD (ACADM); Short Chain (C4–C6): SCAD (ACADS), SCHAD (HADH)
  - Carnitine cycle: long-chain fatty acids cannot cross IMM as free acids; require carnitine shuttle:
    fatty acyl-CoA + carnitine → acylcarnitine (CPT1A) → translocase (CACT/SLC25A20) → matrix → CPT2 → acyl-CoA
  - FAO defects → hypoketotic hypoglycaemia (no acetyl-CoA for ketogenesis during fasting)
    + energy deficit (heart, muscle, liver) + toxic intermediate accumulation (e.g., C14:1 in VLCAD)
  - NBS acylcarnitine profile detects most FAO disorders: C8 elevation (MCAD), C14:1 (VLCAD),
    C16-OH/C18-OH (LCHAD/MTP), very high C16/C18 ratio (CACT), low free carnitine (OCTN2/CPT1A)

ATLAS SCOPE (10 nuclear-encoded mitochondrial FAO genes):
  Acyl-CoA dehydrogenases:
    ACADM  — MCAD (medium chain acyl-CoA dehydrogenase, C6–C12, 421aa, 1p31.1) — MOST COMMON FAO disorder
    ACADVL — VLCAD (very long chain acyl-CoA dehydrogenase, ≥C14, 655aa, 17p13.1) — cardiac/hepatic
  Mitochondrial trifunctional protein (MTP):
    HADHA  — MTP alpha subunit: LCHAD + long-chain 2-enoyl-CoA hydratase (763aa, 2p23.3)
    HADHB  — MTP beta subunit: long-chain 3-ketoacyl-CoA thiolase (474aa, 2p23.3)
  Short chain hydroxyacyl-CoA dehydrogenase:
    HADH   — SCHAD (short chain 3-hydroxyacyl-CoA dehydrogenase, 314aa, 4q22.1) — hyperinsulinism
  Short chain acyl-CoA dehydrogenase:
    ACADS  — SCAD (short chain acyl-CoA dehydrogenase, C4–C6, 412aa, 12q24.31)
  Carnitine palmitoyltransferases:
    CPT1A  — Carnitine palmitoyltransferase 1A (liver isoform, 773aa, 11q13.3) — Arctic variant
    CPT2   — Carnitine palmitoyltransferase 2 (muscle/multi-organ, 658aa, 1p32.3) — rhabdomyolysis
  Carnitine cycle transporters:
    SLC25A20 — CACT (carnitine-acylcarnitine translocase, 301aa, 3p21.31) — neonatal cardiac emergency
    SLC22A5  — OCTN2 (organic cation/carnitine transporter 2, 557aa, 5q31.1) — primary carnitine deficiency

CRITICAL CLINICAL RULES:
  1. KD ABSOLUTE CI: VLCAD, LCHAD (HADHA), MTP (HADHB), CACT (SLC25A20), CPT2 (severe forms) —
     these disorders CANNOT oxidise long-chain fats; KD = pure long-chain fat load → catastrophic energy failure
  2. KD GENERALLY TOLERATED (with caution): MCAD (ACADM) — medium chain oxidation intact; MCT-based KD safer
  3. MCT (medium chain triglyceride) diet: TREATMENT backbone for VLCAD and LCHAD — MCT bypasses long-chain transport/oxidation block
  4. Fasting ABSOLUTELY FORBIDDEN for all FAO disorders — the metabolic emergency trigger; IV dextrose mandatory
  5. VPA HIGH RISK for ALL FAO disorders — inhibits mitochondrial FAO at acyl-CoA dehydrogenase step;
     carnitine depletion secondary to VPA; especially dangerous in MCAD (C8-VPA adduct formation) and VLCAD
  6. L-Carnitine: TREATMENT for OCTN2/SLC22A5 (primary carnitine deficiency — dramatic response);
     consider in CACT/CPT2 to support acylcarnitine handling; NOT proven beneficial for MCAD/VLCAD as primary Rx
  7. NBS: acylcarnitine profile on filter-paper blood spot detects MCAD, VLCAD, LCHAD/MTP, CACT, CPT1A, SCAD at birth
  8. CPT1A Arctic variant (p.Pro479Leu): common in Inuit/First Nations populations; partial CPT1A deficiency;
     generally benign but risk of acute hypoketotic hypoglycaemia under metabolic stress/illness
  9. HADH (SCHAD): uniquely causes congenital hyperinsulinism via loss of inhibitory HADH-GDH interaction;
     diagnosis: protein-sensitive hypoglycaemia + elevated C4-OH acylcarnitine + elevated SCHAD enzyme
  10. BTBGD mandatory exclusion if any Leigh-like presentation (FAO-Leigh overlap with CACT/CPT2 neonatal)

COHORT: 10 × 40 = 400 patient slots (seeds 813–822; gene-specific seeds)
"""

import random

SEED_BASE = 813

# ── All 10 mitochondrial FAO genes ───────────────────────────────────────────
FAO_GENES = [
    # ── ACADM — MCAD, C6–C12, most common FAO disorder, NBS C8 ──
    {
        "gene": "ACADM", "alias": "MCAD — Medium Chain Acyl-CoA Dehydrogenase (C6–C12)",
        "aa": "421 aa", "kDa": "44.8 kDa",
        "gene_class": "acyl_coa_dehydrogenase_medium_chain",
        "locus": "1p31.1", "omim_gene": 607008,
        "phenotype": "MCAD deficiency (MCADD) — MOST COMMON FAO disorder; NBS C8 elevation; hypoketotic hypoglycaemia; avoid fasting",
        "disease": (
            "ACADM biallelic loss → MCAD deficiency (MCADD, OMIM #201450) — most common FAO disorder in European populations (1:12,000 live births). "
            "ACADM encodes medium-chain acyl-CoA dehydrogenase (421aa, homotetrameric FAD-binding enzyme) catalysing the FAD-dependent dehydrogenation of "
            "C6–C12 acyl-CoA substrates (octanoyl-CoA/C8 most important physiologically). Biallelic loss → inability to oxidise medium-chain fatty acids during fasting → "
            "hypoketotic hypoglycaemia (no acetyl-CoA from medium-chain oxidation for ketogenesis) + accumulation of C8-carnitine (octanoylcarnitine) + C6/C10 species. "
            "NBS: C8 acylcarnitine elevation on filter-paper bloodspot (Guthrie card) — first FAO disorder detected by expanded NBS. Phenotype: normally asymptomatic on regular feeds; "
            "metabolic crisis during fasting/illness (vomiting → cannot feed → fatty acid mobilisation with block → encephalopathy, hepatomegaly, hypoketotic hypoglycaemia, coma); "
            "pre-NBS mortality 25%; post-NBS near-normal life expectancy with dietary management. Founder variant: p.Lys304Glu (c.985A>G, ~80% of European MCADD alleles)."
        ),
        "inheritance": "Autosomal recessive, biallelic; high frequency in Northern European populations; p.Lys304Glu homozygous accounts for ~50% of all MCADD diagnoses.",
        "hallmark": (
            "MCADD HALLMARKS: (1) MOST COMMON mitochondrial FAO disorder (1:12,000 European); "
            "(2) NBS C8 acylcarnitine (octanoylcarnitine) — key diagnostic marker; C8/C10 ratio elevated; "
            "(3) HYPOKETOTIC hypoglycaemia — urine ketones absent/low despite profound hypoglycaemia (opposite of normal fasting response); "
            "(4) FOUNDER VARIANT p.Lys304Glu (c.985A>G) in >80% of European alleles — simple PCR screening used pre-NBS era; "
            "(5) FASTING STRICTLY FORBIDDEN — maximum fast duration: 4 hours neonate, 8 hours child, 12 hours adult; "
            "(6) VPA HIGH RISK — VPA is a medium-chain fatty acid analogue; C8-VPA adduct formation competes with MCAD substrate; "
            "(7) KD: generally tolerated (medium-chain oxidation is the block, not long-chain transport); MCT-based KD safer than LCT-based but caution needed; "
            "(8) Riboflavin (B2): FAD cofactor for acyl-CoA dehydrogenases including MCAD; empiric B2 trial in unconfirmed FAO; "
            "(9) Emergency card mandatory: carry IV dextrose protocol; any illness/vomiting → immediate IV glucose (10% dextrose); "
            "(10) Hepatomegaly during crisis from fat accumulation (hepatic steatosis); MRI generally normal inter-crisis."
        ),
        "key_ddx": (
            "vs VLCAD (ACADVL): VLCAD → C14:1 elevation (not C8); more severe cardiac phenotype; KD ABSOLUTE CI in VLCAD (not MCAD); "
            "vs LCHAD (HADHA): C16-OH elevation (not C8); maternal pregnancy complications (AFLP/HELLP); pigmentary retinopathy; "
            "vs GA2/MADD (ETFA/ETFB): multiple acylcarnitine elevations (C4 + C5 + C8 + C10 + C12 + C14); severe metabolic acidosis; riboflavin responsive form; "
            "vs SCHAD/HADH: C4-OH elevation + hyperinsulinism (not present in MCAD); "
            "vs ketotic hypoglycaemia (non-FAO): low acylcarnitines; normal enzyme activity; responds to glucagon."
        ),
        "founder_variant": "p.Lys304Glu (c.985A>G, European founder, >80% of MCADD alleles in Northern European ancestry; disrupts tetramer interface + FAD binding); p.Tyr42His (less common, milder); p.Arg223Trp; many private variants in non-European populations",
        "onset_pattern": "NBS cohort: asymptomatic until metabolic stress; crisis typically 3-24 months (weaning + longer fasting + illness); pre-NBS: acute presentation during febrile illness, vomiting, prolonged fasting.",
        "mri_pattern": "Usually normal; during acute crisis: diffuse cerebral oedema (reversible with glucose); globus pallidus T2 abnormalities reported in severe delayed cases; no chronic structural abnormality in well-managed NBS cases.",
        "hypoglycaemia_rate": 0.88, "hepatopathy_rate": 0.45, "myopathy_rate": 0.08,
        "encephalopathy_rate": 0.20, "epilepsy_rate": 0.12, "ataxia_rate": 0.06,
        "lactic_ac_rate": 0.15, "hcm_rate": 0.03, "snhl_rate": 0.02,
        "rhabdomyolysis_rate": 0.05, "hyperinsulinism_rate": 0.00, "retinopathy_rate": 0.00,
        "renal_rate": 0.02, "neonatal_crisis_rate": 0.20, "maternal_complication_rate": 0.00,
        "seed": 813,
        "kd_absolute_ci": False, "kd_tolerated": True, "mct_treatment": False,
        "lcarnitine_treatment": False, "lcarnitine_ci": False,
        "fasting_forbidden": True, "nbs_detected": True,
        "nbs_marker": "C8 acylcarnitine (octanoylcarnitine) elevation — highly specific",
        "vpa_risk": "HIGH RISK — VPA is a medium-chain fatty acid analogue; C8-VPA metabolite formation competes at MCAD active site; carnitine sequestration; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "IV 10% dextrose (2ml/kg bolus) immediately for any hypoglycaemic crisis; glucose infusion rate (GIR) 8-12 mg/kg/min; maintain blood glucose >4 mmol/L; avoid fasting >4-12 hours by age",
    },
    # ── ACADVL — VLCAD, C14–C20, cardiac/hepatic, KD ABSOLUTE CI ──
    {
        "gene": "ACADVL", "alias": "VLCAD — Very Long Chain Acyl-CoA Dehydrogenase (≥C14)",
        "aa": "655 aa", "kDa": "70.4 kDa",
        "gene_class": "acyl_coa_dehydrogenase_very_long_chain",
        "locus": "17p13.1", "omim_gene": 609575,
        "phenotype": "VLCAD deficiency — second most common FAO disorder; C14:1 NBS marker; cardiac (HCM/arrhythmia) + hepatic + myopathic forms; KD ABSOLUTE CI",
        "disease": (
            "ACADVL biallelic loss → VLCAD deficiency (OMIM #201475) — second most common FAO disorder (~1:30,000). "
            "ACADVL encodes very long-chain acyl-CoA dehydrogenase (655aa, homodimeric, IMM-associated, FAD-binding) catalysing FAD-dependent dehydrogenation "
            "of very long chain acyl-CoA substrates (C14–C20, especially palmitoyl-CoA C16, linoleoyl-CoA C18:2). "
            "Biallelic loss → inability to oxidise long-chain fats → C14:1-carnitine (tetradecadienoylcarnitine) accumulation + energy deficit in heart/muscle/liver. "
            "THREE PHENOTYPIC FORMS: (1) SEVERE NEONATAL/INFANTILE: HCM (55-70%) + hepatomegaly + lactic acidosis + hypoketotic hypoglycaemia; arrhythmias (VT); high mortality without early MCT diet; "
            "(2) CHILDHOOD HEPATIC: hepatomegaly + recurrent hypoketotic hypoglycaemia, no/mild HCM; "
            "(3) ADULT MYOPATHIC: exercise-induced rhabdomyolysis + muscle pain + myoglobinuria; CK spikes with exertion. "
            "MCT (medium chain triglyceride) diet is the cornerstone treatment — MCT enters beta-oxidation directly without VLCAD; bypasses the block."
        ),
        "inheritance": "Autosomal recessive, biallelic; no founder variant; allelic heterogeneity correlates with phenotype severity (null/null = severe; missense = milder).",
        "hallmark": (
            "VLCAD HALLMARKS: (1) KD ABSOLUTE CONTRAINDICATION — KD is a long-chain fat load → direct toxic substrate accumulation; "
            "(2) MCT DIET = PRIMARY TREATMENT — medium chain triglycerides bypass the VLCAD block completely; "
            "(3) C14:1 acylcarnitine = diagnostic NBS marker (tetradecadienoylcarnitine; C14:1/C2 ratio elevated); "
            "(4) THREE FORMS: neonatal-cardiac (HCM/arrhythmia), hepatic (hypoglycaemia), adult-myopathic (rhabdomyolysis); "
            "(5) HCM in 55-70% severe form — dilated or hypertrophic; echo mandatory at presentation; ECG for arrhythmias; "
            "(6) Rhabdomyolysis: exercise-induced in myopathic form; CK >10,000 IU/L; myoglobinuria; risk of AKI; "
            "(7) VPA ABSOLUTE AVOID — inhibits FAO further; accelerates acylcarnitine accumulation; worsens metabolic crisis; "
            "(8) FASTING ABSOLUTELY FORBIDDEN — fasting triggers mobilisation of long-chain fats → VLCAD block → crisis; "
            "(9) Propofol caution — propofol inhibits mitochondrial FAO (propofol infusion syndrome); "
            "(10) Exercise protocol: avoid sustained aerobic exercise; pre-exercise glucose snack; carnitine supplementation may help rhabdomyolysis."
        ),
        "key_ddx": (
            "vs MCAD (ACADM): MCAD → C8 (not C14:1); MCAD KD generally tolerated; MCAD no cardiac phenotype; "
            "vs LCHAD/MTP (HADHA/HADHB): C16-OH and C18-OH elevated (not primarily C14:1); LCHAD → retinopathy + maternal AFLP; MTP → neuropathy; "
            "vs CPT2 (severe): CPT2 → C16/C18 elevation without hydroxyl; neonatal CPT2 cardiac overlap but VLCAD has C14:1; "
            "vs CACT (SLC25A20): CACT → very high C16/C18/C16:1 WITHOUT C14:1; neonatal cardiac equally severe; "
            "vs Pompe / other HCM: Pompe → elevated CK + muscle biopsy glycogen; urine Hex4; normal acylcarnitines."
        ),
        "founder_variant": "No common founder variant; allelic heterogeneity: c.848T>C (p.Val283Ala, common mild allele); c.1349G>A (p.Arg450His, severe); c.1795G>A (p.Asp599Asn, intermediate); null alleles (frameshifts/nonsense) → severe neonatal cardiac form",
        "onset_pattern": "Severe form: neonatal day 1-3 (cardiac crisis, HCM); hepatic form: 3-18 months during illness; myopathic form: adolescence-adulthood (exercise-induced).",
        "mri_pattern": "Usually normal; during severe neonatal crisis: diffuse injury; myopathic form: normal brain; fatty infiltration of liver on ultrasound/MRI; HCM on echo (concentric or dilated, may resolve with MCT treatment).",
        "hypoglycaemia_rate": 0.75, "hepatopathy_rate": 0.65, "myopathy_rate": 0.55,
        "encephalopathy_rate": 0.30, "epilepsy_rate": 0.10, "ataxia_rate": 0.05,
        "lactic_ac_rate": 0.45, "hcm_rate": 0.62, "snhl_rate": 0.02,
        "rhabdomyolysis_rate": 0.45, "hyperinsulinism_rate": 0.00, "retinopathy_rate": 0.02,
        "renal_rate": 0.15, "neonatal_crisis_rate": 0.45, "maternal_complication_rate": 0.05,
        "seed": 814,
        "kd_absolute_ci": True, "kd_tolerated": False, "mct_treatment": True,
        "lcarnitine_treatment": False, "lcarnitine_ci": False,
        "fasting_forbidden": True, "nbs_detected": True,
        "nbs_marker": "C14:1 acylcarnitine (tetradecadienoylcarnitine) elevation — primary VLCAD NBS marker; C14:1/C2 ratio",
        "vpa_risk": "ABSOLUTE AVOID — worsens FAO block; accelerates acylcarnitine accumulation; can precipitate fatal crisis; prefer LEV/LCM/ZNS",
        "metformin_ci": False,
        "acute_treatment": "IV dextrose (10-25%) to maintain glucose >5 mmol/L; no lipid emulsion in crisis (long-chain fats contraindicated); MCT oil via NGT once tolerating; monitor CK and LFTs; arrhythmia management (avoid propofol); cardiac echo",
    },
    # ── HADHA — MTP alpha, LCHAD + 2-enoyl-CoA hydratase, maternal AFLP/HELLP ──
    {
        "gene": "HADHA", "alias": "MTP-α — LCHAD (long-chain 3-hydroxyacyl-CoA dehydrogenase) + long-chain 2-enoyl-CoA hydratase (MTP alpha subunit)",
        "aa": "763 aa", "kDa": "79.5 kDa",
        "gene_class": "mtp_alpha_lchad_hydratase",
        "locus": "2p23.3", "omim_gene": 600890,
        "phenotype": "LCHAD deficiency / MTP alpha deficiency — C16-OH NBS marker; maternal AFLP/HELLP; pigmentary retinopathy; peripheral neuropathy; MCT diet treatment",
        "disease": (
            "HADHA biallelic loss → LCHAD deficiency or complete MTP deficiency (OMIM #609016). "
            "HADHA encodes the alpha subunit (763aa) of the mitochondrial trifunctional protein (MTP), a heterooctameric complex (α4β4) embedded in the IMM. "
            "The alpha subunit carries TWO catalytic domains: (1) long-chain 2-enoyl-CoA hydratase (LCEH) and (2) long-chain 3-hydroxyacyl-CoA dehydrogenase (LCHAD). "
            "The LCHAD domain (Glu510 catalytic) catalyses NAD⁺-dependent oxidation of long-chain L-3-hydroxyacyl-CoA to 3-ketoacyl-CoA (C12–C18 substrates). "
            "BIALLELIC: LCHAD deficiency (isolated LCHAD loss with partial MTP function, founder p.Glu510Gln) OR complete MTP deficiency (null/null → both LCHAD + LCEH lost). "
            "MATERNAL PHENOMENON: heterozygous carrier mothers of LCHAD-deficient fetuses: 79% risk of AFLP (acute fatty liver of pregnancy) and/or HELLP syndrome — "
            "fetal LCHAD-deficient 3-hydroxyfatty acids cross the placenta → maternal hepatotoxicity; test ALL mothers of LCHAD-affected infants. "
            "NBS: C16-OH + C18-OH acylcarnitines. Treatment: MCT-rich + long-chain fat restricted diet; DHA supplementation for retinopathy prevention."
        ),
        "inheritance": "Autosomal recessive, biallelic; founder variant p.Glu510Gln (c.1528G>C) in Northern European (Finnish) populations with isolated LCHAD.",
        "hallmark": (
            "LCHAD/MTP-ALPHA HALLMARKS: (1) KD ABSOLUTE CI — LCHAD cannot oxidise long-chain 3-hydroxy intermediates; LCT-based KD → toxic 3-OH-acylcarnitine accumulation; "
            "(2) MATERNAL AFLP/HELLP — pathognomonic association; test all mothers of affected infants; UNIQUE among FAO disorders; "
            "(3) PIGMENTARY RETINOPATHY — LCHAD/MTP-specific; progressive, worsens with age; chorioretinal degeneration; ophthalmology follow-up every 6-12 months; "
            "(4) PERIPHERAL NEUROPATHY — especially MTP-alpha deficiency; axonal sensorimotor neuropathy; EMG/NCS monitoring; "
            "(5) C16-OH + C18-OH acylcarnitines = NBS markers (3-hydroxy long-chain species); "
            "(6) MCT DIET = cornerstone treatment (with DHA supplementation for retinopathy); "
            "(7) p.Glu510Gln (LCHAD catalytic residue) = Finnish/Northern European founder variant → isolated LCHAD with partial MTP; "
            "(8) FASTING ABSOLUTELY FORBIDDEN; "
            "(9) Exercise-induced rhabdomyolysis in milder MTP/LCHAD forms; "
            "(10) DHA (docosahexaenoic acid) supplementation: reduces retinopathy progression (DHA synthesis requires LCHAD function)."
        ),
        "key_ddx": (
            "vs VLCAD (ACADVL): C14:1 not C16-OH; no retinopathy; no maternal AFLP; "
            "vs HADHB/MTP-beta: HADHB → same C16-OH/C18-OH pattern; neuropathy similar; HADHB deficiency = complete MTP only (no isolated LCHAD); distinguish by sequencing; "
            "vs AFLP without FAO (other causes): LCHAD deficiency maternal AFLP occurs specifically in heterozygous HADHA carrier mothers; "
            "vs Leber congenital amaurosis (retinopathy DDx): LCA → early-onset severe vision loss, genetic panel; C16-OH normal; no FAO phenotype."
        ),
        "founder_variant": "p.Glu510Gln (c.1528G>C, Finnish/Northern European founder, ~70% of European LCHAD alleles; isolated LCHAD catalytic loss); p.Gly242Ala; p.Arg505Gln; null alleles → complete MTP deficiency (more severe neuropathy/retinopathy)",
        "onset_pattern": "Severe neonatal: first hours-days (HCM + lactic acidosis + hypoglycaemia); hepatic: 3-18 months; later presentation with pigmentary retinopathy/neuropathy in milder forms; maternal AFLP in 3rd trimester of CARRYING a LCHAD-affected fetus.",
        "mri_pattern": "Normal or white matter signal abnormalities; no Leigh pattern; retinal exam shows pigmentary changes; peripheral nerve: axonal loss on MRI neurography/NCS.",
        "hypoglycaemia_rate": 0.78, "hepatopathy_rate": 0.72, "myopathy_rate": 0.45,
        "encephalopathy_rate": 0.28, "epilepsy_rate": 0.12, "ataxia_rate": 0.18,
        "lactic_ac_rate": 0.55, "hcm_rate": 0.40, "snhl_rate": 0.02,
        "rhabdomyolysis_rate": 0.38, "hyperinsulinism_rate": 0.00, "retinopathy_rate": 0.65,
        "renal_rate": 0.08, "neonatal_crisis_rate": 0.42, "maternal_complication_rate": 0.79,
        "seed": 815,
        "kd_absolute_ci": True, "kd_tolerated": False, "mct_treatment": True,
        "lcarnitine_treatment": False, "lcarnitine_ci": False,
        "fasting_forbidden": True, "nbs_detected": True,
        "nbs_marker": "C16-OH + C18-OH acylcarnitines (3-hydroxy long-chain species); C18:1-OH also elevated",
        "vpa_risk": "ABSOLUTE AVOID — worsens long-chain FAO block; accelerates 3-hydroxy-acylcarnitine accumulation; hepatotoxic synergy with LCHAD-related hepatopathy; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "IV dextrose (10-25%); strict no long-chain fat in acute phase; MCT oil as sole fat source; DHA supplementation 30-50mg/kg/day; avoid fasting; ophthalmology + EMG/NCS follow-up annually",
    },
    # ── HADHB — MTP beta, long-chain 3-ketoacyl-CoA thiolase, complete MTP deficiency ──
    {
        "gene": "HADHB", "alias": "MTP-β — Long-chain 3-ketoacyl-CoA thiolase (MTP beta subunit)",
        "aa": "474 aa", "kDa": "51.2 kDa",
        "gene_class": "mtp_beta_thiolase",
        "locus": "2p23.3", "omim_gene": 143450,
        "phenotype": "MTP deficiency (beta subunit) — complete MTP complex loss; C16-OH NBS marker; severe neonatal form + neuropathy/retinopathy; KD ABSOLUTE CI",
        "disease": (
            "HADHB biallelic loss → complete MTP (mitochondrial trifunctional protein) deficiency (OMIM #609015). "
            "HADHB encodes the MTP beta subunit (474aa), which carries the long-chain 3-ketoacyl-CoA thiolase (LCKAT) catalytic domain — "
            "the fourth enzyme of beta-oxidation for long-chain substrates, cleaving 3-ketoacyl-CoA to acetyl-CoA + acyl-CoA(n-2). "
            "The heterooctameric MTP complex requires both HADHA (α4) and HADHB (β4): loss of HADHB → complete MTP complex destabilisation (all three MTP-catalysed reactions fail). "
            "HADHB deficiency ALWAYS causes complete MTP loss (unlike HADHA where p.Glu510Gln → isolated LCHAD with partial MTP). "
            "Complete MTP deficiency → more severe neuropathy + more pronounced retinopathy than isolated LCHAD (all three long-chain beta-oxidation steps lost). "
            "NBS: C16-OH + C18-OH (same as HADHA); distinction requires sequencing. Maternal AFLP risk: also present (HADHB carrier mothers) but less studied than HADHA."
        ),
        "inheritance": "Autosomal recessive, biallelic; no common founder variant; allelic heterogeneity; null alleles → most severe neonatal form.",
        "hallmark": (
            "HADHB/MTP-BETA HALLMARKS: (1) COMPLETE MTP LOSS — all three long-chain beta-oxidation steps lost (hydratase + LCHAD + thiolase); "
            "(2) KD ABSOLUTE CI — same rationale as HADHA/LCHAD; long-chain fat entirely unusable; "
            "(3) SEVERE PROGRESSIVE NEUROPATHY — axonal sensorimotor neuropathy more severe than isolated LCHAD (complete MTP loss); "
            "(4) PROGRESSIVE RETINOPATHY — pigmentary degeneration; DHA supplementation recommended; "
            "(5) Same NBS markers as HADHA: C16-OH + C18-OH; ONLY SEQUENCING distinguishes HADHA vs HADHB; "
            "(6) Maternal AFLP risk in heterozygous mothers (less well-characterised than HADHA); "
            "(7) MCT diet = treatment backbone (same as HADHA); "
            "(8) Rhabdomyolysis: exercise-induced + fasting-triggered; CK monitoring; "
            "(9) FASTING ABSOLUTELY FORBIDDEN; "
            "(10) DHA supplementation: may slow retinopathy and neuropathy progression."
        ),
        "key_ddx": (
            "vs HADHA (LCHAD isolated, p.Glu510Gln): identical NBS pattern; HADHB → complete MTP (all 3 steps lost); HADHA founder → isolated LCHAD (only dehydrogenase lost); "
            "sequencing/enzyme panel distinguishes; HADHB: more severe neuropathy; "
            "vs VLCAD: C14:1 not C16-OH; no retinopathy; "
            "vs CPT2 (severe): CPT2 → C16/C18 without hydroxyl group; no retinopathy."
        ),
        "founder_variant": "No common founder; c.1674+1G>T (splice, severe); c.901A>G (p.Asn301Asp, moderate); c.836T>C (p.Leu279Pro, severe neonatal); many private variants",
        "onset_pattern": "Severe neonatal: first days (HCM, lactic acidosis, hypoglycaemia, fatal without treatment); milder alleles → hepatic form in infancy; neuropathy/retinopathy: progressive from childhood.",
        "mri_pattern": "Generally normal brain; peripheral nerve affected (axonal neuropathy on NCS); retinal pigmentary degeneration on fundoscopy/OCT; normal Leigh pattern absent.",
        "hypoglycaemia_rate": 0.82, "hepatopathy_rate": 0.70, "myopathy_rate": 0.55,
        "encephalopathy_rate": 0.32, "epilepsy_rate": 0.10, "ataxia_rate": 0.22,
        "lactic_ac_rate": 0.60, "hcm_rate": 0.45, "snhl_rate": 0.03,
        "rhabdomyolysis_rate": 0.42, "hyperinsulinism_rate": 0.00, "retinopathy_rate": 0.72,
        "renal_rate": 0.08, "neonatal_crisis_rate": 0.48, "maternal_complication_rate": 0.50,
        "seed": 816,
        "kd_absolute_ci": True, "kd_tolerated": False, "mct_treatment": True,
        "lcarnitine_treatment": False, "lcarnitine_ci": False,
        "fasting_forbidden": True, "nbs_detected": True,
        "nbs_marker": "C16-OH + C18-OH acylcarnitines (identical to HADHA; sequencing required for gene-level distinction)",
        "vpa_risk": "ABSOLUTE AVOID — worsens complete MTP block; accelerates 3-hydroxy-acylcarnitine accumulation; synergistic hepatotoxicity; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "IV dextrose; strict no long-chain fat; MCT as sole fat source; DHA supplementation; neuropathy monitoring (EMG/NCS annually); ophthalmology 6-monthly; avoid fasting and strenuous exercise",
    },
    # ── HADH — SCHAD, hyperinsulinism via GDH interaction, C4-OH ──
    {
        "gene": "HADH", "alias": "SCHAD — Short Chain 3-Hydroxyacyl-CoA Dehydrogenase (C4–C6)",
        "aa": "314 aa", "kDa": "34.3 kDa",
        "gene_class": "short_chain_hydroxyacyl_coa_dehydrogenase",
        "locus": "4q22.1", "omim_gene": 601609,
        "phenotype": "SCHAD deficiency — HYPERINSULINISM (GDH disinhibition); protein-sensitive hypoglycaemia; C4-OH NBS marker; unique FAO-hyperinsulinism bridge",
        "disease": (
            "HADH biallelic loss → SCHAD deficiency (OMIM #231530) — unique among FAO disorders: primary manifestation is congenital hyperinsulinism (HI), not classic FAO crisis. "
            "HADH encodes short-chain L-3-hydroxyacyl-CoA dehydrogenase (314aa, homotetrameric, NAD⁺-dependent, matrix) catalysing oxidation of short-chain L-3-hydroxyacyl-CoA (C4-C6). "
            "MECHANISM OF HYPERINSULINISM (unique): HADH normally inhibits glutamate dehydrogenase (GDH, GLUD1) by direct protein-protein interaction. "
            "HADH loss → GDH disinhibition → constitutive GDH activation → elevated α-KG → ATP/ADP ratio ↑ → KATP channel closure → membrane depolarisation → Ca²⁺ entry → insulin secretion. "
            "This makes SCHAD deficiency the only FAO disorder where HI is the dominant feature. "
            "Metabolic phenotype: protein-sensitive hypoglycaemia (leucine and other amino acids feed GDH) + mild FAO defect (C4-OH-acylcarnitine, L-3-hydroxybutyrylcarnitine). "
            "NBS: C4-OH (L-3-hydroxybutyrylcarnitine) elevation — same as HADHA/B for C4, distinguishable by chain-length and clinical context."
        ),
        "inheritance": "Autosomal recessive, biallelic; rare disorder; founder variant in Saudi Arabia (p.Tyr348Cys); multiple private variants worldwide.",
        "hallmark": (
            "SCHAD/HADH HALLMARKS: (1) CONGENITAL HYPERINSULINISM = PRIMARY PHENOTYPE (NOT hypoglycaemia from FAO failure); "
            "(2) PROTEIN-SENSITIVE HYPOGLYCAEMIA — leucine/amino acids trigger GDH-mediated insulin release; protein load test positive; "
            "(3) C4-OH acylcarnitine (L-3-hydroxybutyrylcarnitine) elevation on NBS; "
            "(4) DIAZOXIDE RESPONSE: SCHAD HI typically responds to diazoxide (KATP channel opener) — unlike focal/diffuse KATP-channel HI; "
            "(5) HADH → GDH INTERACTION: SCHAD is the only short-chain hydroxyacyl-CoA dehydrogenase that DIRECTLY INHIBITS GDH; "
            "(6) KD: GENERALLY SAFE (short-chain oxidation path, carnitine cycle not involved in C4 transport); "
            "(7) Fasting: avoid prolonged fasting but less critical than long-chain FAO disorders; "
            "(8) PLASMA INSULIN ELEVATED during hypoglycaemia (hallmark of hyperinsulinism — distinguishes from classical FAO hypoketotic hypoglycaemia); "
            "(9) VPA: relatively lower risk than VLCAD/LCHAD (short-chain pathway less impacted) but caution still warranted; "
            "(10) Diagnosis: C4-OH acylcarnitine + plasma insulin + clinical diazoxide response; confirm with HADH gene sequencing."
        ),
        "key_ddx": (
            "vs MCAD (C4 from MCAD): MCAD → C8 (not C4-OH specifically); no hyperinsulinism; "
            "vs KATP-channel HI (KCNJ11/ABCC8): SCHAD → C4-OH elevated; KCNJ11/ABCC8 → normal acylcarnitines; SCHAD often diazoxide-responsive; "
            "vs GDH-HI (GLUD1 gain-of-function): GLUD1 activating variants → elevated ammonia (HI-HA syndrome); HADH → normal ammonia; "
            "vs HNF4A HI: transient, responds to diazoxide; normal acylcarnitines; family history of MODY1/HNF4A."
        ),
        "founder_variant": "p.Tyr348Cys (Saudi Arabian founder); p.Arg97Cys; p.Glu241Lys; p.Lys12del; p.Gln279Arg — many private variants",
        "onset_pattern": "Neonatal to infantile hyperinsulinism (first days to weeks); macrosomia at birth (fetal hyperinsulinism); recurrent hypoglycaemia with feeds/protein; diagnosis often delayed as FAO disorder diagnosis not initially suspected.",
        "mri_pattern": "Usually normal; hypoglycaemic brain injury possible if severe prolonged episodes (periventricular white matter, cortical); no specific FAO MRI pattern.",
        "hypoglycaemia_rate": 0.95, "hepatopathy_rate": 0.12, "myopathy_rate": 0.05,
        "encephalopathy_rate": 0.15, "epilepsy_rate": 0.20, "ataxia_rate": 0.03,
        "lactic_ac_rate": 0.08, "hcm_rate": 0.02, "snhl_rate": 0.01,
        "rhabdomyolysis_rate": 0.02, "hyperinsulinism_rate": 0.98, "retinopathy_rate": 0.00,
        "renal_rate": 0.02, "neonatal_crisis_rate": 0.65, "maternal_complication_rate": 0.00,
        "seed": 817,
        "kd_absolute_ci": False, "kd_tolerated": True, "mct_treatment": False,
        "lcarnitine_treatment": False, "lcarnitine_ci": False,
        "fasting_forbidden": True, "nbs_detected": True,
        "nbs_marker": "C4-OH acylcarnitine (L-3-hydroxybutyrylcarnitine); plasma insulin elevated during hypoglycaemia",
        "vpa_risk": "CAUTION — short-chain pathway less severely impacted than long-chain FAO disorders; VPA can cause carnitine depletion; monitor carnitine levels; prefer LEV/LCM if needed",
        "metformin_ci": False,
        "acute_treatment": "IV dextrose for hypoglycaemia; diazoxide (KATP channel opener) 5-15 mg/kg/day in 2-3 divided doses for HI; avoid protein loading; protein restriction diet if protein-sensitive; frequent feeds",
    },
    # ── ACADS — SCAD, C4–C6, debated clinical significance ──
    {
        "gene": "ACADS", "alias": "SCAD — Short Chain Acyl-CoA Dehydrogenase (C4–C6)",
        "aa": "412 aa", "kDa": "44.3 kDa",
        "gene_class": "acyl_coa_dehydrogenase_short_chain",
        "locus": "12q24.31", "omim_gene": 606885,
        "phenotype": "SCAD deficiency — NBS C4 (butyrylcarnitine) elevation; debated clinical significance; most NBS-identified cases asymptomatic; metabolic variant",
        "disease": (
            "ACADS biallelic loss → SCAD deficiency (SCADD, OMIM #201470) — short chain acyl-CoA dehydrogenase (412aa, homotetrameric, FAD-binding, matrix) catalyses "
            "FAD-dependent dehydrogenation of butyryl-CoA (C4) and hexanoyl-CoA (C6). "
            "SCADD was originally considered a disease; current consensus: SCADD is a METABOLIC VARIANT with extremely variable penetrance and uncertain clinical significance. "
            "EPIDEMIOLOGY: ~1:50,000 in population; NBS identifies via C4 (butyrylcarnitine) elevation; many NBS-positive infants remain completely asymptomatic lifelong. "
            "COMMON VARIANT: two susceptibility variants (p.Gly209Ser c.625G>A and p.Arg147Trp c.439C>T) reduce SCAD enzyme thermostability but are present in ~7% of general population; "
            "found in compound heterozygosity with pathogenic variants in most symptomatic cases. "
            "RARE SYMPTOMATIC CASES: reported with hypoglycaemia, hypotonia, failure to thrive, developmental delay — but causality debated (many have additional diagnoses). "
            "Current recommendation: most SCADD patients do not require dietary treatment; follow up annually; dietary management only if symptomatic."
        ),
        "inheritance": "Autosomal recessive, biallelic; high population frequency of susceptibility variants (p.Gly209Ser ~5%, p.Arg147Trp ~2%); true pathogenic variants rare.",
        "hallmark": (
            "SCADD HALLMARKS: (1) DEBATED CLINICAL SIGNIFICANCE — most NBS-identified SCADD is asymptomatic; major disease entity status is questioned; "
            "(2) C4 butyrylcarnitine elevation = NBS marker (also seen in isobutyrylglycinuria/IVD-related, distinguish by acylcarnitine profile); "
            "(3) POPULATION FREQUENCY of susceptibility variants (p.Gly209Ser p.Arg147Trp) in 7% of general population — thermolability, not frank enzyme deficiency; "
            "(4) Riboflavin (B2) trial: riboflavin is FAD precursor; SCAD enzyme activity can improve with riboflavin in some variants; "
            "(5) KD: generally safe (short-chain pathway, carnitine cycle not required for C4); "
            "(6) VPA: relatively safe compared to long-chain FAO disorders; low-risk; "
            "(7) FASTING AVOIDANCE: standard metabolic advice but crisis risk lower than MCAD/VLCAD/LCHAD; "
            "(8) Most experts: no dietary restriction for asymptomatic NBS-identified SCADD; observe + annual review; "
            "(9) Diagnosis: C4 on NBS + ACADS sequencing + enzyme assay + urine organic acids (methylsuccinate, ethylmalonate, butyrylglycine); "
            "(10) DIFFERENTIAL: isobutyrylglycinuria (IBD/IVD), GA2 (ETFA/ETFB) — both can elevate C4; distinguish by profile and enzyme panel."
        ),
        "key_ddx": (
            "vs MCAD (ACADM): MCAD → C8 predominant (not C4); MCAD clinically significant disease; "
            "vs isobutyryl-CoA dehydrogenase deficiency (IBD/ACAD8): C4-acylcarnitine also elevated; distinguish by urine organic acids; "
            "vs GA2/MADD (ETFA/ETFB): multiple acylcarnitines including C4; more severe; riboflavin responsive form; "
            "vs SCHAD/HADH: C4-OH (3-hydroxy-C4) not plain C4; hyperinsulinism; clinically distinct."
        ),
        "founder_variant": "p.Gly209Ser (c.625G>A, thermolability susceptibility, ~5% population); p.Arg147Trp (c.439C>T, thermolability susceptibility, ~2% population); true pathogenic variants: c.310C>T (p.Arg104Cys), frameshift/nonsense alleles (rare)",
        "onset_pattern": "Most: asymptomatic, detected by NBS; rare symptomatic cases: infantile (hypoglycaemia, hypotonia); developmentally normal majority.",
        "mri_pattern": "Usually normal; reported cases: white matter signal changes (possibly coincidental); no specific SCADD MRI pattern established.",
        "hypoglycaemia_rate": 0.12, "hepatopathy_rate": 0.08, "myopathy_rate": 0.10,
        "encephalopathy_rate": 0.08, "epilepsy_rate": 0.06, "ataxia_rate": 0.05,
        "lactic_ac_rate": 0.05, "hcm_rate": 0.01, "snhl_rate": 0.01,
        "rhabdomyolysis_rate": 0.02, "hyperinsulinism_rate": 0.00, "retinopathy_rate": 0.00,
        "renal_rate": 0.01, "neonatal_crisis_rate": 0.05, "maternal_complication_rate": 0.00,
        "seed": 818,
        "kd_absolute_ci": False, "kd_tolerated": True, "mct_treatment": False,
        "lcarnitine_treatment": False, "lcarnitine_ci": False,
        "fasting_forbidden": False, "nbs_detected": True,
        "nbs_marker": "C4 butyrylcarnitine elevation on NBS — DEBATED SIGNIFICANCE; most NBS-positive individuals are asymptomatic",
        "vpa_risk": "LOW RISK — short-chain pathway; VPA generally safe; standard monitoring; no specific CI",
        "metformin_ci": False,
        "acute_treatment": "Most cases: no dietary intervention; symptomatic cases: riboflavin trial (100mg/day); avoid prolonged fasting as precaution; carnitine supplementation if depletion detected; no emergency card needed for most",
    },
    # ── CPT1A — Liver isoform, CPT1 deficiency, Arctic variant, ABSOLUTE KD CI ──
    {
        "gene": "CPT1A", "alias": "CPT1A — Carnitine Palmitoyltransferase 1A (liver isoform)",
        "aa": "773 aa", "kDa": "88.0 kDa",
        "gene_class": "carnitine_palmitoyltransferase_1a_liver",
        "locus": "11q13.3", "omim_gene": 600528,
        "phenotype": "CPT1A deficiency — hypoketotic hypoglycaemia; Reye-like hepatopathy; KD ABSOLUTE CI; Arctic variant (p.Pro479Leu) in Inuit populations; fasting FORBIDDEN",
        "disease": (
            "CPT1A biallelic pathogenic loss → CPT1A deficiency (OMIM #255120). "
            "CPT1A encodes carnitine palmitoyltransferase 1A (773aa, liver/kidney isoform), the rate-limiting enzyme of the carnitine cycle: "
            "converts long-chain acyl-CoA (C12–C18) + free carnitine → acylcarnitine ester on the OMM outer face, enabling IMM transport by CACT. "
            "CPT1A has a regulatory C-terminal domain inhibited by malonyl-CoA (fed state signal) — CPT1A deficiency cannot activate long-chain FAO during fasting. "
            "NBS: LOW free carnitine + elevated C0/C16 + C0/C18 ratios; acylcarnitines may be relatively low (block is UPSTREAM of mitochondrial beta-oxidation). "
            "ARCTIC VARIANT: p.Pro479Leu (c.1436C>T) — found in ~68% of Inuit/First Nations individuals of northern Canada and Alaska; "
            "causes partial CPT1A thermolability (~60% residual activity) — generally benign but triggers acute hypoketotic hypoglycaemia under metabolic stress (illness, prolonged fasting); "
            "associated with 'benign neonatal hypoglycaemia' in Inuit communities; genetic counselling important for this at-risk population. "
            "SEVERE DISEASE: complete CPT1A loss → Reye-like hepatic crisis (hepatomegaly + hypertransaminasaemia + coagulopathy) + profound hypoketotic hypoglycaemia."
        ),
        "inheritance": "Autosomal recessive, biallelic; Arctic variant (p.Pro479Leu) enriched in Inuit/First Nations populations (allele frequency up to 0.68 in some communities); pathogenic variants rare in non-Arctic populations.",
        "hallmark": (
            "CPT1A HALLMARKS: (1) KD ABSOLUTE CONTRAINDICATION — CPT1A is the first step in long-chain fat mitochondrial import; KD = pure long-chain fat → cannot enter beta-oxidation; catastrophic energy failure; "
            "(2) ARCTIC VARIANT p.Pro479Leu: partial deficiency in Inuit/First Nations; generally benign; high allele frequency population-specific; "
            "(3) REYE-LIKE HEPATIC CRISIS: severe disease → hepatomegaly + elevated transaminases + coagulopathy + hypoketotic hypoglycaemia during fasting/illness; "
            "(4) LOW FREE CARNITINE + normal/high C0 ratio — NBS pattern: free carnitine (C0) often low; acylcarnitine profile less dramatic than other FAO disorders; "
            "(5) MALONYL-CoA REGULATION: CPT1A inhibited by malonyl-CoA in fed state; deficiency removes this control — no pathological consequence in fed state but critical during fasting when malonyl-CoA falls and long-chain FAO must activate; "
            "(6) FASTING ABSOLUTELY FORBIDDEN; "
            "(7) Renal involvement: kidney (second major site of CPT1A expression) may show fatty infiltration; "
            "(8) L-Carnitine: typically NOT supplemented as the block is at the first step (supplementing carnitine may partially help handle acylcarnitine efflux but is not primary treatment); "
            "(9) Emergency management: IV dextrose; avoid lipid infusions containing long-chain fats; "
            "(10) Liver transplant: has been used in severe forms refractory to medical management."
        ),
        "key_ddx": (
            "vs CPT2 deficiency: CPT2 → C16/C18 acylcarnitine accumulation (within mitochondria); CPT1A → low free carnitine + low acylcarnitines on NBS; "
            "vs CACT (SLC25A20): CACT → very high C16/C18/C16:1 within mitochondria + severe neonatal cardiac; "
            "vs MCAD/VLCAD: CPT1A → no C8/C14:1 elevation; block is at carnitine conjugation not dehydrogenation; "
            "vs Reye syndrome (non-FAO): CPT1A deficiency is a FAO-mimic of Reye; NBS or CPT1A enzyme assay distinguishes."
        ),
        "founder_variant": "p.Pro479Leu (c.1436C>T, Arctic/Inuit/First Nations founder, allele frequency ~0.68 in some Inuit communities — partial deficiency only); pathogenic null alleles (European): c.1319dupC, c.2243_2244delAG, many private variants",
        "onset_pattern": "Severe: infantile (Reye-like crisis during first febrile illness or prolonged fast); Arctic variant: neonatal-infantile benign hypoglycaemia with illness.",
        "mri_pattern": "Usually normal; severe crisis: diffuse cerebral oedema (reversible); no specific structural abnormality.",
        "hypoglycaemia_rate": 0.90, "hepatopathy_rate": 0.75, "myopathy_rate": 0.10,
        "encephalopathy_rate": 0.20, "epilepsy_rate": 0.08, "ataxia_rate": 0.04,
        "lactic_ac_rate": 0.25, "hcm_rate": 0.05, "snhl_rate": 0.01,
        "rhabdomyolysis_rate": 0.05, "hyperinsulinism_rate": 0.00, "retinopathy_rate": 0.00,
        "renal_rate": 0.15, "neonatal_crisis_rate": 0.35, "maternal_complication_rate": 0.00,
        "seed": 819,
        "kd_absolute_ci": True, "kd_tolerated": False, "mct_treatment": False,
        "lcarnitine_treatment": False, "lcarnitine_ci": False,
        "fasting_forbidden": True, "nbs_detected": True,
        "nbs_marker": "Low free carnitine (C0); elevated C0/C16 + C0/C18 ratios (upstream block — acylcarnitines relatively low vs other FAO disorders)",
        "vpa_risk": "HIGH RISK — VPA inhibits CPT1 directly (VPA-carnitine conjugation) + hepatotoxic synergy with CPT1A hepatopathy; ABSOLUTE AVOID in severe CPT1A; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "IV dextrose (10-25%); no long-chain fat (IV lipid emulsions must be MCT-based or withheld); correction of coagulopathy; N-acetylcysteine for hepatic crisis; liver transplant evaluation in severe refractory cases; avoid fasting",
    },
    # ── CPT2 — Multi-organ; neonatal lethal / infantile hepatocardiomuscular / adult myopathic ──
    {
        "gene": "CPT2", "alias": "CPT2 — Carnitine Palmitoyltransferase 2 (inner mitochondrial membrane)",
        "aa": "658 aa", "kDa": "74.0 kDa",
        "gene_class": "carnitine_palmitoyltransferase_2",
        "locus": "1p32.3", "omim_gene": 600650,
        "phenotype": "CPT2 deficiency — THREE FORMS: neonatal lethal (multiorgan), infantile hepatocardiomuscular, adult myopathic (rhabdomyolysis — most common); C16/C18 NBS; KD CI in severe forms",
        "disease": (
            "CPT2 biallelic loss → CPT2 deficiency (OMIM #255110, #600649). "
            "CPT2 encodes carnitine palmitoyltransferase 2 (658aa, IMM inner face enzyme) — the SECOND step of the carnitine cycle: "
            "reconverts long-chain acylcarnitine (from CACT transport) back to acyl-CoA + free carnitine inside the mitochondrial matrix, making long-chain acyl-CoA available for beta-oxidation. "
            "THREE PHENOTYPIC FORMS based on residual enzyme activity: "
            "(1) NEONATAL LETHAL FORM (<10% residual activity): neonatal cardiac arrhythmias + HCM + respiratory failure + liver failure + brain malformations (periventricular heterotopia, neuronal migration defects, renal cysts); "
            "often fatal in first weeks without rapid diagnosis; "
            "(2) INFANTILE HEPATOCARDIOMUSCULAR: 6-24 months, triggered by illness/fasting; HCM + hepatomegaly + hypoketotic hypoglycaemia + myopathy; "
            "(3) ADULT MYOPATHIC (most common, 30-50% residual activity): exercise-induced rhabdomyolysis + myalgia + myoglobinuria; CK spikes; normal life expectancy with management. "
            "Founder variant p.Ser113Leu (c.338C>T) causes adult myopathic form (~60% of adult CPT2 alleles in European populations)."
        ),
        "inheritance": "Autosomal recessive, biallelic; p.Ser113Leu common in adult myopathic form (European); allelic heterogeneity determines phenotypic severity.",
        "hallmark": (
            "CPT2 HALLMARKS: (1) THREE CLINICAL FORMS — neonatal lethal (brain malformations!), infantile hepatocardiomuscular, adult myopathic (MOST COMMON); "
            "(2) KD: ABSOLUTE CI for neonatal/infantile forms; RELATIVE CI for adult myopathic (avoid long-chain fat loading); "
            "(3) ADULT MYOPATHIC FORM = MOST COMMON CPT2 PRESENTATION — rhabdomyolysis after sustained aerobic exercise; "
            "CK >10,000 IU/L; myoglobinuria → dark urine; risk of AKI; often misdiagnosed as viral myositis; "
            "(4) C16 + C18 acylcarnitine elevation on NBS (long-chain species without 3-OH — distinguishes from HADHA/HADHB); "
            "(5) p.Ser113Leu FOUNDER VARIANT (adult myopathic) — homozygous or compound het; "
            "(6) BRAIN MALFORMATIONS in neonatal form: periventricular heterotopia + dysmorphic lateral ventricles + renal cortical cysts = virtually pathognomonic of neonatal CPT2; "
            "(7) L-Carnitine: supplementation may help acylcarnitine efflux; especially useful in infantile form; "
            "(8) FASTING FORBIDDEN in neonatal/infantile; exercise restriction in adult myopathic; "
            "(9) VPA: HIGH RISK in neonatal/infantile forms; CAUTION in adult (monitor CK, carnitine); "
            "(10) Exercise protocol: adult form — warm-up gradually; high-carbohydrate pre-exercise snack; avoid fasting before exercise; MCT not as effective as in VLCAD."
        ),
        "key_ddx": (
            "vs CACT (SLC25A20): CACT → neonatal form also severe with C16/C18; but CACT: very high C16:1 + no brain malformations; "
            "vs CPT1A: CPT1A → low free carnitine, not high C16/C18; no brain malformations; "
            "vs VLCAD: VLCAD → C14:1; CPT2 → C16/C18 without 3-OH; VLCAD no brain malformations in neonatal; "
            "vs Pompe (HCM DDx): Pompe → muscle biopsy glycogen; elevated urine Hex4; normal acylcarnitines."
        ),
        "founder_variant": "p.Ser113Leu (c.338C>T, adult myopathic, ~60% of adult CPT2 European alleles; reduces enzyme thermostability); p.Arg631Cys (severe); null alleles (neonatal lethal); p.Pro50His (mild)",
        "onset_pattern": "Neonatal lethal: day 1-7; infantile: 6-24 months (crisis during illness); adult myopathic: adolescence-adulthood (exercise-related onset); can present at any age.",
        "mri_pattern": "Neonatal form: periventricular nodular heterotopia + dysmorphic lateral ventricles + simplified gyral pattern (brain malformation group); adult/infantile: usually normal brain; muscle: fatty infiltration on MRI in severe myopathic.",
        "hypoglycaemia_rate": 0.55, "hepatopathy_rate": 0.45, "myopathy_rate": 0.80,
        "encephalopathy_rate": 0.25, "epilepsy_rate": 0.15, "ataxia_rate": 0.10,
        "lactic_ac_rate": 0.35, "hcm_rate": 0.35, "snhl_rate": 0.02,
        "rhabdomyolysis_rate": 0.75, "hyperinsulinism_rate": 0.00, "retinopathy_rate": 0.02,
        "renal_rate": 0.25, "neonatal_crisis_rate": 0.30, "maternal_complication_rate": 0.00,
        "seed": 820,
        "kd_absolute_ci": True, "kd_tolerated": False, "mct_treatment": False,
        "lcarnitine_treatment": True, "lcarnitine_ci": False,
        "fasting_forbidden": True, "nbs_detected": True,
        "nbs_marker": "C16 + C18 acylcarnitines (long-chain, without 3-OH prefix — distinguishes from LCHAD/MTP C16-OH/C18-OH)",
        "vpa_risk": "HIGH RISK in neonatal/infantile forms — further worsens FAO; hepatotoxic synergy; ABSOLUTE AVOID in neonatal; CAUTION in adult myopathic (monitor CK/carnitine); prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "IV dextrose; no long-chain fat IV lipid; L-carnitine supplementation; AKI monitoring in rhabdomyolysis (IV fluid, urine alkalinisation); exercise restriction in adult form; carbohydrate-rich pre-exercise fuel in adult form",
    },
    # ── SLC25A20 (CACT) — Carnitine-acylcarnitine translocase, neonatal cardiac emergency ──
    {
        "gene": "SLC25A20", "alias": "CACT — Carnitine-Acylcarnitine Translocase (SLC25A20)",
        "aa": "301 aa", "kDa": "32.5 kDa",
        "gene_class": "carnitine_acylcarnitine_translocase",
        "locus": "3p21.31", "omim_gene": 613698,
        "phenotype": "CACT deficiency — NEONATAL CARDIAC EMERGENCY (HCM + arrhythmia day 1); very high C16/C18/C16:1 NBS; KD ABSOLUTE CI; most severe carnitine cycle disorder; high neonatal mortality",
        "disease": (
            "SLC25A20 biallelic loss → CACT deficiency (OMIM #212138) — one of the most severe mitochondrial FAO/carnitine cycle disorders. "
            "SLC25A20 encodes the carnitine-acylcarnitine translocase (CACT, 301aa, 6-TM helix antiporter in IMM) — the carrier protein that exchanges: "
            "acylcarnitine (from CPT1A) → matrix; free carnitine → cytoplasm (antiport); CACT is the obligate SHUTTLE for all long-chain fatty acids crossing the IMM. "
            "CACT deficiency → all long-chain acylcarnitines CANNOT enter the mitochondrial matrix for beta-oxidation → "
            "catastrophic energy failure in heart + liver + skeletal muscle + brain on day 1 of life. "
            "NBS: VERY HIGH C16 + C18 + C16:1 acylcarnitines (dramatically elevated, often 10-50x normal); free carnitine very low; "
            "NEONATAL EMERGENCY: HCM (hypertrophic cardiomyopathy day 1-2) + ventricular arrhythmias + liver failure + hypoglycaemia + hyperammonaemia (secondary); "
            "mortality >70% untreated in first weeks. Even with early diagnosis and MCT diet, prognosis is guarded."
        ),
        "inheritance": "Autosomal recessive, biallelic; founder variant p.Arg301Gln in Saudi Arabia; multiple private variants; allelic heterogeneity affects severity.",
        "hallmark": (
            "CACT HALLMARKS: (1) NEONATAL CARDIAC EMERGENCY — most severe carnitine cycle disorder; HCM + arrhythmia DAY 1; "
            "(2) KD ABSOLUTE CI — CACT is the IMM gateway for ALL long-chain fats; KD = lethal; no long-chain fat possible; "
            "(3) VERY HIGH C16 + C18 + C16:1 acylcarnitines (10-50x normal) = most dramatic NBS profile among FAO disorders; "
            "(4) FREE CARNITINE CRITICALLY LOW — all carnitine trapped as acylcarnitines cannot recycle; "
            "(5) SECONDARY HYPERAMMONAEMIA — acylcarnitine accumulation inhibits N-acetylglutamate synthase → urea cycle suppression; "
            "(6) MCT diet is the only viable fat source (MCT enters beta-oxidation without CACT transport); "
            "(7) L-Carnitine supplementation: helps maintain carnitine pool for acylcarnitine efflux; "
            "(8) HEART TRANSPLANT: considered in severe refractory HCM; "
            "(9) FASTING ABSOLUTELY FORBIDDEN; "
            "(10) IV lipid emulsion: ONLY MCT-based permitted; standard TPN lipid (LCT-based) ABSOLUTELY CONTRAINDICATED."
        ),
        "key_ddx": (
            "vs CPT2 (neonatal): CPT2 neonatal → C16/C18 without 3-OH + brain malformations + renal cysts; CACT → no brain malformations; "
            "vs VLCAD (neonatal cardiac): VLCAD → C14:1 not C16/C18; less severely elevated acylcarnitines; "
            "vs CPT1A: CPT1A → low free carnitine but low acylcarnitines (upstream block); CACT → very high acylcarnitines (downstream accumulation); "
            "vs OCTN2 (SLC22A5): OCTN2 → low C0 + normal chain-length pattern; carnitine deficiency without dramatic C16/C18 elevation."
        ),
        "founder_variant": "p.Arg301Gln (Saudi Arabian founder); c.227A>G (p.Asn76Ser, Japanese founder); p.Trp231* (European); many private variants in non-consanguineous populations",
        "onset_pattern": "Neonatal: first 24-48 hours (HCM, arrhythmia, hepatomegaly, hypoglycaemia); rare mild alleles: later infantile presentation.",
        "mri_pattern": "Brain: usually normal (distinguishes from CPT2 neonatal which has malformations); liver: marked steatosis on ultrasound/MRI; heart: HCM on echo (concentric or asymmetric septal hypertrophy).",
        "hypoglycaemia_rate": 0.92, "hepatopathy_rate": 0.85, "myopathy_rate": 0.40,
        "encephalopathy_rate": 0.35, "epilepsy_rate": 0.12, "ataxia_rate": 0.04,
        "lactic_ac_rate": 0.65, "hcm_rate": 0.88, "snhl_rate": 0.01,
        "rhabdomyolysis_rate": 0.30, "hyperinsulinism_rate": 0.00, "retinopathy_rate": 0.00,
        "renal_rate": 0.10, "neonatal_crisis_rate": 0.90, "maternal_complication_rate": 0.02,
        "seed": 821,
        "kd_absolute_ci": True, "kd_tolerated": False, "mct_treatment": True,
        "lcarnitine_treatment": True, "lcarnitine_ci": False,
        "fasting_forbidden": True, "nbs_detected": True,
        "nbs_marker": "VERY HIGH C16 + C18 + C16:1 acylcarnitines (10-50× normal); free carnitine critically low; most dramatic FAO NBS profile",
        "vpa_risk": "ABSOLUTE AVOID — further depletes carnitine; worsens acylcarnitine accumulation; hepatotoxic; avoid in any patient with CACT; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "IV dextrose (no long-chain lipid IV); MCT-based feeds only; L-carnitine supplementation; treat arrhythmia (amiodarone with caution); treat hyperammonaemia (arginine + citrulline + rifaximin); cardiac transplant evaluation; PICU emergency",
    },
    # ── SLC22A5 (OCTN2) — Primary carnitine deficiency, L-carnitine DRAMATICALLY RESPONSIVE ──
    {
        "gene": "SLC22A5", "alias": "OCTN2 — Organic Cation/Carnitine Transporter 2 (SLC22A5)",
        "aa": "557 aa", "kDa": "62.9 kDa",
        "gene_class": "octn2_carnitine_transporter",
        "locus": "5q31.1", "omim_gene": 603377,
        "phenotype": "Primary carnitine deficiency (PCD) — DRAMATIC L-carnitine response; HCM + skeletal myopathy; NBS free carnitine critically low; KD not CI but caution; carrier mothers may be symptomatic",
        "disease": (
            "SLC22A5 biallelic loss → Primary Carnitine Deficiency (PCD, OMIM #212140) — "
            "SLC22A5 encodes OCTN2 (organic cation/carnitine transporter 2, 557aa, 12-TM helix, plasma membrane + renal tubule), "
            "the high-affinity Na⁺-dependent carnitine transporter that reabsorbs carnitine from the glomerular filtrate (kidney) and transports carnitine into cells (intestine, heart, muscle). "
            "OCTN2 loss → massive renal carnitine wasting (up to 95% of filtered load lost in urine instead of reabsorbed) → profound systemic carnitine deficiency. "
            "Carnitine is essential for: (1) long-chain fatty acid transport across IMM (carnitine cycle with CPT1A/CACT/CPT2); "
            "(2) removal of toxic acyl groups via urinary acylcarnitine excretion. "
            "PCD PHENOTYPE without treatment: HCM (myocardial lipid storage) + skeletal myopathy (exercise intolerance) + hypoglycaemia (triggered by fasting) + Reye-like hepatopathy. "
            "DRAMATIC L-CARNITINE RESPONSE: oral/IV L-carnitine (100-400 mg/kg/day) normalises plasma carnitine → complete reversal of cardiomyopathy and myopathy in most patients. "
            "CARRIER MOTHERS: obligate heterozygous mothers of PCD children have 50% reduced plasma carnitine and may develop dilated cardiomyopathy — carrier screening + supplementation recommended."
        ),
        "inheritance": "Autosomal recessive, biallelic; Japanese founder variant p.Arg169Trp (c.506C>T, ~1:40,000 Japan, ~40% of Japanese PCD alleles); Faroe Islands founder p.Arg254* (~1:300 Faroese carriers).",
        "hallmark": (
            "PCD/OCTN2 HALLMARKS: (1) L-CARNITINE TREATMENT = DRAMATIC RESPONSE — cardiomyopathy and myopathy FULLY REVERSIBLE with L-carnitine; "
            "earliest-identified FAO disorder where treatment is essentially curative; "
            "(2) FREE CARNITINE CRITICALLY LOW (C0 <5 µmol/L; normal 25-50 µmol/L) — most dramatic carnitine depletion of all disorders; "
            "(3) HCM REVERSIBLE WITH CARNITINE — NBS-diagnosed PCD patients started on carnitine rarely develop cardiomyopathy; "
            "(4) CARRIER MOTHERS AT RISK — 50% carnitine → cardiomyopathy risk; test maternal carnitine when PCD proband identified; "
            "(5) KD: generally safe (carnitine cycle intact once carnitine supplied; long-chain FAO enzymes normal); but high-dose L-carnitine supplement mandatory during KD; "
            "(6) NBS: free carnitine C0 critically low — most sensitive NBS marker; acylcarnitine profile relatively unremarkable (chain-length enzyme activity normal); "
            "(7) URINE CARNITINE: massive carnitine wasting on 24h urine (opposite to CPT1A/CACT where acylcarnitines spill but free carnitine is trapped); "
            "(8) Renal tubular carnitine wasting: continued oral L-carnitine for life (renal leak persistent); "
            "(9) Faroe Islands cluster: p.Arg254* founder, high carrier frequency 1:10 on Faroe Islands; "
            "(10) Riboflavin (B2): not specifically indicated (the FAO enzymes are normal; carnitine supply is the only defect)."
        ),
        "key_ddx": (
            "vs secondary carnitine deficiency (VPA, pivampicillin, dialysis): PCD → SLC22A5 variants; carnitine normalises with supplementation in secondary too; secondary has underlying cause; "
            "vs CACT (SLC25A20): CACT → very high C16/C18 (accumulation inside); PCD → low C0, relatively normal chain-length distribution; "
            "vs CPT1A: CPT1A → low C0 + elevated C0/C16 ratio; CPT1A: acyl-CoA accumulates proximally; PCD: all acylcarnitines low from carnitine scarcity; "
            "vs Pompe (HCM DDx): Pompe → glycogen on biopsy; PCD → lipid vacuoles; carnitine deficiency biochemistry."
        ),
        "founder_variant": "p.Arg169Trp (c.506C>T, Japanese founder, ~40% of Japanese PCD alleles); p.Arg254* (Faroe Islands founder); p.Arg282Gln; many missense variants in compound het; p.Trp51* (Ashkenazi Jewish)",
        "onset_pattern": "NBS cohort: asymptomatic, L-carnitine started before symptoms; undiagnosed: infantile (HCM, Reye-like) or older childhood (exercise myopathy); carrier mothers: adult cardiomyopathy.",
        "mri_pattern": "Cardiac: HCM on echo (resolves with L-carnitine); brain: usually normal; muscle: lipid storage myopathy on biopsy (Oil Red O positive vacuoles); no neurological structural MRI changes.",
        "hypoglycaemia_rate": 0.45, "hepatopathy_rate": 0.35, "myopathy_rate": 0.70,
        "encephalopathy_rate": 0.12, "epilepsy_rate": 0.06, "ataxia_rate": 0.08,
        "lactic_ac_rate": 0.15, "hcm_rate": 0.65, "snhl_rate": 0.01,
        "rhabdomyolysis_rate": 0.15, "hyperinsulinism_rate": 0.00, "retinopathy_rate": 0.00,
        "renal_rate": 0.05, "neonatal_crisis_rate": 0.25, "maternal_complication_rate": 0.25,
        "seed": 822,
        "kd_absolute_ci": False, "kd_tolerated": True, "mct_treatment": False,
        "lcarnitine_treatment": True, "lcarnitine_ci": False,
        "fasting_forbidden": True, "nbs_detected": True,
        "nbs_marker": "Free carnitine C0 critically low (<5 µmol/L; normal 25-50 µmol/L) — most dramatic carnitine depletion; acylcarnitine profile relatively normal chain-length distribution",
        "vpa_risk": "HIGH RISK — VPA depletes carnitine (secondary carnitine deficiency via VPA-CoA formation); catastrophically dangerous in PCD (baseline carnitine already critically low); ABSOLUTE AVOID without monitoring + aggressive carnitine supplementation; prefer LEV/LCM",
        "metformin_ci": False,
        "acute_treatment": "IV L-carnitine (100-300 mg/kg/day IV loading for cardiac crisis); transition to oral L-carnitine (100-400 mg/kg/day in 3-4 divided doses for life); cardiomyopathy reversal within weeks; carrier mothers: carnitine supplementation; avoid fasting during supplementation adjustment",
    },
]

# ── Cohort generator ──────────────────────────────────────────────────────────
def _generate_cohort(g: dict) -> list:
    rng = random.Random(g["seed"])
    patients = []
    ages = [round(rng.uniform(0.1, 45), 1) for _ in range(40)]
    sexes = [rng.choice(["M", "F"]) for _ in range(40)]
    for i, (age, sex) in enumerate(zip(ages, sexes)):
        pid = f"{g['gene']}-{i+1:03d}"
        patients.append({
            "patient_id": pid,
            "gene": g["gene"],
            "age_at_diagnosis": age,
            "sex": sex,
            "hypoglycaemia": rng.random() < g["hypoglycaemia_rate"],
            "hepatopathy": rng.random() < g["hepatopathy_rate"],
            "myopathy": rng.random() < g["myopathy_rate"],
            "encephalopathy": rng.random() < g["encephalopathy_rate"],
            "epilepsy": rng.random() < g["epilepsy_rate"],
            "ataxia": rng.random() < g["ataxia_rate"],
            "lactic_acidosis": rng.random() < g["lactic_ac_rate"],
            "hcm": rng.random() < g["hcm_rate"],
            "snhl": rng.random() < g["snhl_rate"],
            "rhabdomyolysis": rng.random() < g["rhabdomyolysis_rate"],
            "hyperinsulinism": rng.random() < g["hyperinsulinism_rate"],
            "retinopathy": rng.random() < g["retinopathy_rate"],
            "renal": rng.random() < g["renal_rate"],
            "neonatal_crisis": rng.random() < g["neonatal_crisis_rate"],
            "kd_absolute_ci": g["kd_absolute_ci"],
            "mct_treatment": g["mct_treatment"],
            "nbs_detected": rng.random() < (0.92 if g["nbs_detected"] else 0.10),
        })
    return patients


def _pct(cohort: list, key: str) -> float:
    if not cohort:
        return 0.0
    return round(sum(1 for p in cohort if p.get(key)) / len(cohort) * 100, 1)


# ── Public API functions ──────────────────────────────────────────────────────
def get_overview() -> dict:
    """Return atlas-wide overview and aggregate clinical statistics."""
    all_patients = []
    for g in FAO_GENES:
        all_patients.extend(_generate_cohort(g))

    total = len(all_patients)
    kd_ci_patients = [p for p in all_patients if p["kd_absolute_ci"]]
    mct_patients = [p for p in all_patients if p["mct_treatment"]]

    def agg_pct(key):
        return round(sum(1 for p in all_patients if p.get(key)) / total * 100, 1)

    return {
        "atlas_name": "FAO-Atlas",
        "atlas_subtitle": "Complete 10-Gene Mitochondrial Fatty Acid Oxidation Disorders Reference",
        "n_genes": len(FAO_GENES),
        "n_patients": total,
        "seeds": "813–822",
        "description": (
            "Mitochondrial fatty acid oxidation (FAO) provides the primary energy source during fasting, "
            "prolonged exercise, and illness-induced catabolism. The carnitine cycle (CPT1A → CACT → CPT2) "
            "transports long-chain fatty acids into the mitochondrial matrix; chain-length-specific acyl-CoA "
            "dehydrogenases (VLCAD, MCAD, SCAD) initiate beta-oxidation; the MTP complex (HADHA/HADHB) "
            "completes long-chain beta-oxidation; SCHAD/HADH acts on short-chain substrates. "
            "FAO disorders present with hypoketotic hypoglycaemia, cardiomyopathy, hepatopathy, myopathy, "
            "or rhabdomyolysis. KD is absolutely contraindicated in long-chain FAO disorders (VLCAD, LCHAD, "
            "MTP, CACT, CPT2 severe) but generally tolerated in MCAD and OCTN2 with carnitine."
        ),
        "chain_length_classes": {
            "very_long_chain": "≥C14 — VLCAD (ACADVL); transported by carnitine cycle; KD ABSOLUTE CI",
            "long_chain": "C12–C18 — MTP complex (HADHA/HADHB); carnitine cycle required; KD ABSOLUTE CI",
            "medium_chain": "C6–C12 — MCAD (ACADM); partial carnitine transport; KD generally tolerated",
            "short_chain": "C4–C6 — SCAD (ACADS), SCHAD (HADH); no carnitine cycle required; KD safe",
        },
        "carnitine_cycle": {
            "step1_cpt1a": "CPT1A (OMM outer face): fatty acyl-CoA + carnitine → acylcarnitine [rate-limiting; malonyl-CoA regulated]",
            "step2_cact": "CACT/SLC25A20 (IMM): acylcarnitine (in) ↔ free carnitine (out) [antiport IMM translocase]",
            "step3_cpt2": "CPT2 (IMM inner face): acylcarnitine → acyl-CoA + free carnitine [regenerates substrate for beta-oxidation]",
        },
        "aggregate_clinical": {
            "hypoglycaemia_pct": agg_pct("hypoglycaemia"),
            "hepatopathy_pct": agg_pct("hepatopathy"),
            "myopathy_pct": agg_pct("myopathy"),
            "encephalopathy_pct": agg_pct("encephalopathy"),
            "epilepsy_pct": agg_pct("epilepsy"),
            "hcm_pct": agg_pct("hcm"),
            "lactic_acidosis_pct": agg_pct("lactic_acidosis"),
            "rhabdomyolysis_pct": agg_pct("rhabdomyolysis"),
            "retinopathy_pct": agg_pct("retinopathy"),
            "hyperinsulinism_pct": agg_pct("hyperinsulinism"),
            "neonatal_crisis_pct": agg_pct("neonatal_crisis"),
            "nbs_detected_pct": agg_pct("nbs_detected"),
            "kd_absolute_ci_pct": round(len(kd_ci_patients) / total * 100, 1),
            "mct_treatment_pct": round(len(mct_patients) / total * 100, 1),
        },
        "drug_contraindications": {
            "kd_long_chain_absolute_ci": (
                "ABSOLUTE CI for ALL long-chain FAO disorders (VLCAD/ACADVL, LCHAD/HADHA, MTP-beta/HADHB, CACT/SLC25A20, CPT1A, CPT2 severe) — "
                "KD is a long-chain fat load; in these disorders long-chain fats CANNOT be oxidised; KD triggers catastrophic energy failure in heart/muscle/liver. "
                "Genes: ACADVL, HADHA, HADHB, CPT1A, CPT2, SLC25A20."
            ),
            "kd_generally_tolerated": (
                "GENERALLY TOLERATED (with caution) for MCAD (ACADM), SCHAD (HADH), SCAD (ACADS), OCTN2 (SLC22A5 with carnitine supplementation) — "
                "medium/short-chain oxidation intact; MCAD: MCT-based KD safer than LCT-based; OCTN2: KD safe but requires aggressive L-carnitine supplementation."
            ),
            "vpa_risk": "HIGH RISK for ALL long-chain FAO disorders — VPA inhibits mitochondrial beta-oxidation at the acyl-CoA dehydrogenase step; VPA-carnitine adduct depletes free carnitine; especially dangerous in MCAD (C8-VPA), VLCAD, LCHAD, CPT1A, CACT. ABSOLUTE AVOID in severe neonatal/infantile forms. For MCAD/SCAD: HIGH CAUTION with carnitine monitoring. PREFER LEV/LCM/ZNS for all FAO disorders.",
            "fasting_forbidden": "ABSOLUTELY FORBIDDEN for all FAO disorders with metabolic crisis risk (ACADM, ACADVL, HADHA, HADHB, CPT1A, CPT2, SLC25A20) — fasting mobilises long-chain fat stores → FAO block → hypoketotic hypoglycaemia + organ failure. Emergency card MANDATORY for all.",
            "lct_iv_lipid_ci": "Standard long-chain triglyceride (LCT) IV lipid emulsions CONTRAINDICATED in VLCAD, LCHAD/MTP, CACT, CPT1A, CPT2 — use MCT-based or withhold lipid; essential fatty acid need managed separately.",
            "propofol_caution": "PROPOFOL CAUTION — propofol inhibits mitochondrial FAO (propofol infusion syndrome mechanism); avoid in long-chain FAO disorders intraoperatively; use inhalational agents or alternative IV agents.",
        },
        "mct_diet_rationale": {
            "mechanism": "MCT (medium chain triglycerides, C8–C12) enter the mitochondrial matrix directly without the carnitine cycle (passive diffusion + MCAD/short-chain pathways); bypasses the long-chain FAO block in VLCAD, LCHAD/MTP, CACT, CPT1A",
            "genes_where_primary_treatment": ["ACADVL (VLCAD)", "HADHA (LCHAD)", "HADHB (MTP-beta)", "SLC25A20 (CACT)"],
            "dha_supplement": "DHA (docosahexaenoic acid) supplementation 30-50 mg/kg/day added in LCHAD/MTP deficiency — DHA synthesis impaired by LCHAD deficiency; DHA supplementation reduces retinopathy progression",
        },
        "nbs_markers": {
            "ACADM": "C8 acylcarnitine (octanoylcarnitine) — primary MCAD NBS marker",
            "ACADVL": "C14:1 acylcarnitine (tetradecadienoylcarnitine) — primary VLCAD NBS marker",
            "HADHA": "C16-OH + C18-OH acylcarnitines (3-hydroxy long-chain) — LCHAD/MTP marker",
            "HADHB": "C16-OH + C18-OH acylcarnitines — identical to HADHA; gene-level distinction by sequencing only",
            "HADH": "C4-OH acylcarnitine (L-3-hydroxybutyrylcarnitine) + plasma insulin elevation",
            "ACADS": "C4 butyrylcarnitine — DEBATED significance; most NBS-positive SCADD asymptomatic",
            "CPT1A": "Low free carnitine (C0); elevated C0/C16 + C0/C18 ratios (upstream carnitine shuttle block)",
            "CPT2": "C16 + C18 acylcarnitines (long-chain, without 3-OH; distinguishes from LCHAD C16-OH)",
            "SLC25A20": "VERY HIGH C16 + C18 + C16:1 acylcarnitines (10-50× normal); free carnitine critically low — most dramatic FAO NBS profile",
            "SLC22A5": "Free carnitine C0 critically low (<5 µmol/L; normal 25-50) — most profound carnitine depletion; acylcarnitine chain-length distribution relatively normal",
        },
        "maternal_risk": {
            "HADHA_carrier_mothers": "79% risk of AFLP (acute fatty liver of pregnancy) and/or HELLP syndrome when carrying an LCHAD-affected fetus — fetal 3-hydroxy-acylcarnitines cross placenta → maternal hepatotoxicity; test all mothers of LCHAD-affected infants",
            "HADHB_carrier_mothers": "Similar risk to HADHA carriers (less well-characterised); MTP-beta carrier mothers may develop AFLP",
            "SLC22A5_carrier_mothers": "Obligate heterozygous mothers of PCD infants have 50% carnitine → dilated cardiomyopathy risk; screen + supplement carnitine",
            "others": "No established maternal pregnancy risk for MCAD, VLCAD, SCAD, SCHAD, CPT1A, CPT2, CACT carrier status",
        },
        "wes_utility": {
            "detects_all_10": True,
            "note": "WES detects all 10 nuclear-encoded FAO genes. Confirm with acylcarnitine profile + enzyme assay (lymphocytes/fibroblasts/muscle). NBS acylcarnitine profile is the primary screening tool; WES provides genotype for prognosis and carrier testing.",
        },
        "btbgd_exclusion": "MANDATORY exclusion of SLC19A3 (BTBGD) in any Leigh-like presentation overlapping with FAO — CACT/CPT2 neonatal severe forms may mimic Leigh syndrome; empiric biotin + thiamine before full workup.",
        "key_rules": {
            "fasting_rule": "NO FASTING for any long-chain FAO disorder: maximum 4h neonate / 8h infant / 12h child / 16h adult; IV dextrose protocol mandatory for any illness.",
            "kd_differentiation": "KD is ABSOLUTE CI for long-chain FAO disorders (VLCAD, LCHAD, MTP, CPT1A, CPT2, CACT) but GENERALLY SAFE for MCAD, SCHAD, SCAD (short/medium chain; KD-producing acetyl-CoA via MCT intake safe).",
            "lcarnitine_rule": "L-carnitine TREATMENT in OCTN2/PCD (dramatic HCM reversal) and CACT (pool maintenance); NOT clearly beneficial as primary Rx for MCAD/VLCAD/LCHAD (the primary issue is enzyme block, not carnitine depletion).",
            "vpa_rule": "VPA HIGH RISK for ALL long-chain FAO disorders; ABSOLUTE AVOID in neonatal/infantile severe forms; prefer LEV/LCM/ZNS for epilepsy management in all FAO patients.",
            "mct_rule": "MCT diet bypasses long-chain transport/oxidation block in VLCAD, LCHAD, MTP, CACT; MCT is the backbone of dietary treatment for these disorders.",
        },
    }


def get_breakdown() -> dict:
    """Return per-gene detailed breakdown for all 10 FAO genes."""
    results = []
    for g in FAO_GENES:
        cohort = _generate_cohort(g)
        results.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "gene_class": g["gene_class"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "phenotype": g["phenotype"],
            "disease": g["disease"],
            "inheritance": g["inheritance"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "founder_variant": g["founder_variant"],
            "onset_pattern": g["onset_pattern"],
            "mri_pattern": g["mri_pattern"],
            "seed": g["seed"],
            "cohort_n": len(cohort),
            "hypoglycaemia_pct": _pct(cohort, "hypoglycaemia"),
            "hepatopathy_pct": _pct(cohort, "hepatopathy"),
            "myopathy_pct": _pct(cohort, "myopathy"),
            "encephalopathy_pct": _pct(cohort, "encephalopathy"),
            "epilepsy_pct": _pct(cohort, "epilepsy"),
            "ataxia_pct": _pct(cohort, "ataxia"),
            "lactic_ac_pct": _pct(cohort, "lactic_acidosis"),
            "hcm_pct": _pct(cohort, "hcm"),
            "snhl_pct": _pct(cohort, "snhl"),
            "rhabdomyolysis_pct": _pct(cohort, "rhabdomyolysis"),
            "hyperinsulinism_pct": _pct(cohort, "hyperinsulinism"),
            "retinopathy_pct": _pct(cohort, "retinopathy"),
            "renal_pct": _pct(cohort, "renal"),
            "neonatal_crisis_pct": _pct(cohort, "neonatal_crisis"),
            "nbs_detected_pct": _pct(cohort, "nbs_detected"),
            "kd_absolute_ci": g["kd_absolute_ci"],
            "kd_tolerated": g["kd_tolerated"],
            "mct_treatment": g["mct_treatment"],
            "lcarnitine_treatment": g["lcarnitine_treatment"],
            "lcarnitine_ci": g["lcarnitine_ci"],
            "fasting_forbidden": g["fasting_forbidden"],
            "nbs_detected": g["nbs_detected"],
            "nbs_marker": g["nbs_marker"],
            "vpa_risk": g["vpa_risk"],
            "metformin_ci": g["metformin_ci"],
            "acute_treatment": g["acute_treatment"],
        })
    return {"genes": results}


def get_definitions() -> list:
    """Return key FAO clinical terms and definitions."""
    return [
        {
            "term": "Mitochondrial Beta-Oxidation (FAO)",
            "definition": (
                "Cyclic process in the mitochondrial matrix degrading fatty acyl-CoA by sequential 2-carbon removal: "
                "(1) acyl-CoA dehydrogenase (VLCAD/MCAD/SCAD) — FAD-dependent dehydrogenation → 2,3-enoyl-CoA; "
                "(2) enoyl-CoA hydratase (HADHA or SCEH) — hydration → L-3-hydroxyacyl-CoA; "
                "(3) 3-hydroxyacyl-CoA dehydrogenase (LCHAD/HADH/SCHAD) — NAD⁺-dependent oxidation → 3-ketoacyl-CoA; "
                "(4) thiolase (HADHB or MCKAT) — thiolytic cleavage → acyl-CoA(n-2) + acetyl-CoA. "
                "Each cycle generates: 1 FADH₂ + 1 NADH + 1 acetyl-CoA (→ TCA cycle / ketogenesis). "
                "Chain-length specificity: VLCAD (≥C14), MTP (C12–C16 long-chain steps 2-4), MCAD (C6–C12), SCAD/SCHAD (C4–C6)."
            )
        },
        {
            "term": "Carnitine Cycle",
            "definition": (
                "Three-step transport system for long-chain fatty acids (>C12) across the inner mitochondrial membrane (IMM): "
                "(1) CPT1A (OMM, outer face): fatty acyl-CoA + L-carnitine → acylcarnitine + CoA [rate-limiting, malonyl-CoA inhibited]; "
                "(2) CACT/SLC25A20 (IMM translocase): acylcarnitine (cytoplasm → matrix) exchanged for free carnitine (matrix → cytoplasm) [antiport]; "
                "(3) CPT2 (IMM, inner face): acylcarnitine + CoA → acyl-CoA + free carnitine [regenerates beta-oxidation substrate]. "
                "Short and medium-chain fatty acids (≤C12) do NOT require the carnitine cycle — enter matrix as free acids. "
                "Defects at any step → long-chain FAO block with acylcarnitine accumulation and free carnitine depletion."
            )
        },
        {
            "term": "Hypoketotic Hypoglycaemia (FAO signature)",
            "definition": (
                "Hallmark biochemical finding of FAO disorders: blood glucose falls during fasting but ketone bodies (3-hydroxybutyrate, acetoacetate) remain inappropriately LOW or absent. "
                "Normal fasting: hepatic FAO generates acetyl-CoA → ketogenesis → blood ketones rise (ketotic response). "
                "FAO disorder: cannot oxidise fats → no acetyl-CoA → no ketogenesis → hypoglycaemia WITHOUT ketonaemia ('hypoketotic'). "
                "Diagnostic: simultaneous blood glucose <2.6 mmol/L + blood 3-hydroxybutyrate <1.0 mmol/L = hypoketotic hypoglycaemia (FAO or hyperinsulinism). "
                "Distinguished from ketotic hypoglycaemia (normal fasting response with appropriate ketonaemia but threshold lowered): "
                "ketotic hypoglycaemia → high ketones; FAO disorder → low ketones. NEVER normal during metabolic crisis in FAO."
            )
        },
        {
            "term": "KD Contraindication in Long-Chain FAO Disorders",
            "definition": (
                "Ketogenic diet (KD) is ABSOLUTELY CONTRAINDICATED in VLCAD (ACADVL), LCHAD (HADHA), MTP-beta (HADHB), CACT (SLC25A20), CPT1A, and severe CPT2. "
                "RATIONALE: KD consists predominantly of long-chain triglycerides (LCT) — typically 70-90% of calories from fat, mostly C16/C18. "
                "In long-chain FAO disorders: long-chain fats CANNOT be transported or oxidised → KD = pure long-chain toxic substrate load → acylcarnitine accumulation → energy failure in heart/muscle/liver. "
                "IMPORTANT DISTINCTION from OXPHOS disorders: in OXPHOS disorders (MDDS, PDC deficiency where PDC KD is TREATMENT), KD provides acetyl-CoA via beta-oxidation BYPASSING the block. "
                "In FAO disorders the BLOCK IS IN BETA-OXIDATION ITSELF — KD cannot bypass it. "
                "KD GENERALLY SAFE: MCAD (medium chain oxidation intact), SCHAD (short chain), SCAD (short chain), OCTN2 (with carnitine supplementation). "
                "MCT-based KD using medium-chain fats (C8/C10) can be used in VLCAD/LCHAD as MCT bypasses the long-chain block."
            )
        },
        {
            "term": "MCT (Medium Chain Triglyceride) Diet — Bypass Strategy",
            "definition": (
                "MCT diet uses medium-chain triglycerides (C8/C10) as the primary fat source in long-chain FAO disorders. "
                "MECHANISM: MCT (C8 caprylic, C10 capric) are absorbed from the gut and enter the mitochondrial matrix directly by passive diffusion + MCAD pathway, "
                "bypassing CPT1A, CACT, CPT2, VLCAD, MTP steps entirely. "
                "PRIMARY TREATMENT for VLCAD, LCHAD, MTP, CACT deficiencies — replaces long-chain dietary fat. "
                "FORMULATION: MCT oil (derived from coconut/palm kernel oil); MCT-based formulae for infants; MCT powder for children; "
                "typical MCT intake: 20-35% of calories from MCT; residual essential fatty acids (EFA: linoleic C18:2, alpha-linolenic C18:3) provided as small amounts of LCT. "
                "DHA supplementation added for LCHAD/MTP (30-50 mg/kg/day): DHA synthesis impaired by LCHAD deficiency → retinopathy. "
                "MONITORING: plasma C14:1 (VLCAD), C16-OH (LCHAD) for dietary adequacy; carnitine levels; growth; echo (HCM regression indicator)."
            )
        },
        {
            "term": "Maternal AFLP/HELLP in LCHAD Heterozygous Mothers",
            "definition": (
                "Unique to HADHA (LCHAD) deficiency: obligate heterozygous carrier mothers of LCHAD-deficient fetuses have 79% risk of ACUTE FATTY LIVER OF PREGNANCY (AFLP) or HELLP syndrome during the pregnancy. "
                "MECHANISM: LCHAD-deficient fetus accumulates 3-hydroxy-acylcarnitines (especially 3-OH-C14:1, 3-OH-C16) in amniotic fluid and fetal circulation → cross the placenta → enter maternal liver → heterozygous mother has 50% reduced LCHAD activity → cannot fully oxidise fetal-derived 3-OH-fatty acids → maternal hepatic lipid accumulation → AFLP (right upper quadrant pain, jaundice, coagulopathy, hepatic failure) or HELLP (haemolysis + elevated liver enzymes + low platelets). "
                "CLINICAL RULE: all mothers of LCHAD-affected infants must be tested for HADHA carrier status; AFLP diagnosis in mother should prompt urgent neonatal FAO screening; "
                "maternal AFLP may be the first presentation of fetal LCHAD deficiency before infant NBS results available. "
                "HADHB carrier mothers: similar risk (less well-characterised). Other FAO carrier mothers: no established AFLP risk."
            )
        },
        {
            "term": "SCHAD (HADH) Deficiency — Hyperinsulinism Mechanism",
            "definition": (
                "HADH (SCHAD) deficiency causes CONGENITAL HYPERINSULINISM as the PRIMARY PHENOTYPE — unique among FAO disorders where the main feature is NOT hypoglycaemia from FAO failure. "
                "MECHANISM: HADH (SCHAD, short-chain 3-hydroxyacyl-CoA dehydrogenase) directly binds to and inhibits glutamate dehydrogenase (GDH, GLUD1) in normal pancreatic beta cells. "
                "HADH LOSS → GDH DISINHIBITION → constitutively active GDH → excess alpha-ketoglutarate → elevated ATP/ADP ratio in beta cell → KATP channel closure → "
                "membrane depolarisation → calcium influx → insulin secretion INDEPENDENT OF glucose. "
                "Result: protein-sensitive hyperinsulinism (amino acids feed GDH → amplified effect). "
                "DIAZOXIDE RESPONSE: SCHAD-HI typically responds to diazoxide (opens KATP channels) — unlike focal HI (surgery-dependent). "
                "C4-OH acylcarnitine (L-3-hydroxybutyrylcarnitine) elevated on NBS (distinguishable from MCAD C8 by chain length). "
                "Plasma insulin elevated DURING hypoglycaemia (confirms HI mechanism — insulin should be suppressed during hypoglycaemia in normal physiology)."
            )
        },
        {
            "term": "CPT2 Neonatal Lethal Form — Brain Malformations",
            "definition": (
                "The neonatal lethal form of CPT2 deficiency (complete CPT2 loss, <10% residual activity) is one of the few FAO disorders associated with structural BRAIN MALFORMATIONS — "
                "virtually pathognomonic when present in combination with the cardiac/hepatic phenotype. "
                "BRAIN FINDINGS: periventricular nodular heterotopia (PNH) — ectopic grey matter nodules along the lateral ventricular walls; "
                "dysmorphic lateral ventricles; simplified gyral pattern; cortical dysplasia; agenesis of corpus callosum (less common). "
                "ADDITIONAL FEATURES: renal cortical microcysts (bilateral). "
                "MECHANISM: CPT2 is required for long-chain FAO in all tissues including developing neurons; complete CPT2 absence during neurogenesis impairs neuronal migration energy supply → heterotopia. "
                "DIFFERENTIATION: CACT (SLC25A20) neonatal form equally severe (HCM, cardiac arrest) but WITHOUT brain malformations — "
                "brain malformations + neonatal FAO crisis → CPT2 neonatal lethal; no brain malformations → CACT or other carnitine cycle disorder."
            )
        },
        {
            "term": "Primary Carnitine Deficiency (PCD/OCTN2) — L-Carnitine Treatment",
            "definition": (
                "SLC22A5 (OCTN2) deficiency causes Primary Carnitine Deficiency (PCD) — remarkable for its dramatic and essentially curative response to L-carnitine supplementation. "
                "OCTN2 is a high-affinity Na⁺-dependent carnitine transporter in kidney (tubular reabsorption) and intestine/heart/muscle (carnitine uptake). "
                "OCTN2 loss → 95% of filtered carnitine lost in urine (renal wasting) → systemic carnitine depletion → cannot transport long-chain fats into mitochondria → HCM + myopathy. "
                "L-CARNITINE TREATMENT: oral L-carnitine 100-400 mg/kg/day (in 3-4 doses, lifelong) → normalises plasma carnitine despite renal wasting → "
                "complete reversal of cardiomyopathy within weeks-months → normal cardiac function; skeletal myopathy resolves; hypoglycaemia prevented. "
                "CARRIER MOTHERS at risk: heterozygous mothers (50% carnitine) → dilated cardiomyopathy; supplement carnitine in carrier mothers. "
                "MONITORING: plasma free carnitine target >20 µmol/L; urine carnitine/creatinine ratio; echo for cardiomyopathy regression; growth."
            )
        },
        {
            "term": "Arctic/Inuit CPT1A Variant (p.Pro479Leu)",
            "definition": (
                "p.Pro479Leu (c.1436C>T) is a common CPT1A variant in Inuit, First Nations, and other Arctic/circumpolar indigenous populations — "
                "allele frequency up to 0.68 in some Inuit communities (far exceeding any pathogenic variant in any other genetic disease). "
                "MECHANISM: p.Pro479Leu reduces CPT1A enzyme thermostability at physiological temperatures (37°C) → ~60% reduced CPT1A activity under thermal challenge; "
                "the variant does NOT fully abolish CPT1A function at normal temperatures but does impair long-chain FAO activation during cold/metabolic stress. "
                "CLINICAL: generally benign with regular nutritious diet; triggers hypoketotic hypoglycaemia under metabolic stress (illness, fasting, cold exposure); "
                "associated with 'benign neonatal hypoglycaemia' epidemic characterised in Inuit communities (Greeland, Alaska, Canada). "
                "EVOLUTIONARY HYPOTHESIS: p.Pro479Leu may provide adaptive advantage in Arctic environments (altered fat metabolism for cold adaptation); "
                "modern dietary change (traditional high-fat diet → Western processed diet) may increase clinical risk. "
                "MANAGEMENT: avoid prolonged fasting; standard illness protocol; genetic counselling for Arctic/circumpolar populations."
            )
        },
        {
            "term": "NBS Acylcarnitine Profile for FAO Disorders",
            "definition": (
                "Expanded newborn screening (NBS) using tandem mass spectrometry (MS/MS) detects most FAO disorders from a filter-paper blood spot (Guthrie card) at 24-72 hours: "
                "MCAD: C8 acylcarnitine (octanoylcarnitine) — most specific NBS test; C8/C10 ratio also elevated; "
                "VLCAD: C14:1 acylcarnitine (tetradecadienoylcarnitine); C14:1/C2 ratio; "
                "LCHAD/MTP: C16-OH + C18-OH (3-hydroxy long-chain species; HADHA and HADHB give identical profile); "
                "CACT: very high C16 + C18 + C16:1 (most dramatic elevation 10-50×; free carnitine critically low); "
                "CPT2: C16 + C18 (without 3-OH; elevated but less dramatic than CACT); "
                "CPT1A: low free carnitine C0 + elevated C0/C16 + C0/C18 ratios (upstream block pattern); "
                "SCHAD/HADH: C4-OH (L-3-hydroxybutyrylcarnitine) + plasma insulin; "
                "OCTN2: C0 critically low (<5 µmol/L; normal 25-50); acylcarnitine chain-length relatively normal; "
                "SCAD: C4 butyrylcarnitine — DEBATED significance; "
                "Gaps: false negative possible in early NBS (before fasting) for MCAD; false positive common for SCAD variants."
            )
        },
        {
            "term": "VPA and FAO Disorders — Mechanism of Toxicity",
            "definition": (
                "Valproic acid (VPA) is HIGH RISK for ALL FAO disorders — multiple mechanisms: "
                "(1) DIRECT FAO INHIBITION: VPA metabolites (4-en-VPA, 2-propyl-4-pentenoic acid) inhibit mitochondrial acyl-CoA dehydrogenases (MCAD, VLCAD, SCAD) — "
                "direct competition with natural substrates at FAD-binding active site → worsens the pre-existing enzyme deficiency; "
                "(2) CARNITINE SEQUESTRATION: VPA forms VPA-carnitine conjugates (valproylcarnitine) excreted in urine → "
                "reduces free carnitine pool → impairs long-chain FAO (compounding the enzyme defect); "
                "(3) HEPATOTOXICITY SYNERGY: VPA hepatotoxicity occurs in patients with underlying FAO defects (especially MCAD, CPT1A, CACT) — "
                "VPA impairs residual FAO capacity → hepatic fat accumulation → Reye-like hepatic failure; "
                "(4) C8-VPA ADDUCT FORMATION in MCAD: octanoyl-CoA (C8-CoA, MCAD substrate) + VPA metabolites → adduct that inhibits MCAD active site selectively. "
                "CLINICAL RULE: VPA ABSOLUTE AVOID in neonatal/infantile severe FAO disorders; VPA HIGH RISK/CAUTION in MCAD, VLCAD, LCHAD, CPT1A. "
                "PREFER LEV/LCM/ZNS for epilepsy management in all FAO patients."
            )
        },
    ]


if __name__ == "__main__":
    import json
    print("=== FAO-Atlas Overview ===")
    ov = get_overview()
    print(json.dumps({k: v for k, v in ov.items() if k not in ("description",)}, indent=2)[:2000])
    print("\n=== Breakdown (first gene) ===")
    bd = get_breakdown()
    print(json.dumps(bd["genes"][0], indent=2)[:1000])
    print(f"\n=== Definitions: {len(get_definitions())} terms ===")
    print("OK")
