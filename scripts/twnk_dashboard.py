#!/usr/bin/env python3
"""TWNK Hepatocerebral / IOSCA / adPEO mtDNA Depletion Syndrome Dashboard.

Mitochondrial DNA Depletion Syndrome 7 (MDDS7) = OMIM #271245
Infantile-Onset Spinocerebellar Ataxia (IOSCA) = OMIM #271245 (allelic)
Autosomal Dominant Progressive External Ophthalmoplegia (adPEO-2) = OMIM #609286

TWNK (C10orf2 / Twinkle, 684 aa, 10q24.31) is the mitochondrial DNA helicase,
essential for mtDNA replication fork unwinding ahead of POLG.
Biallelic AR loss-of-function TWNK → mtDNA depletion → hepatocerebral MDDS7 or IOSCA.
Heterozygous dominant negative/haploinsufficiency → adPEO (multiple mtDNA deletions, not depletion).

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — mtDNA depletion disease; shared risk with POLG/DGUOK/MPV17/TK2
  2. KD = CONTRAINDICATED — OXPHOS-dependent fat oxidation fails in mtDNA depletion
  3. HEPATOCEREBRAL phenotype (MDDS7) — liver failure + neurodegeneration (NO nystagmus = DDx from DGUOK)
  4. NO nystagmus — nystagmus is DGUOK pathognomonic; TWNK hepatocerebral lacks it
  5. NO 3-MGA-uria — critical DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB
  6. NO CK elevation — hepatocerebral not myopathic; CK = TK2 marker
  7. Lactic acidosis 100% in MDDS7 (severe, pH <7.1 possible neonatal/infantile)
  8. Hypoglycemia ~70% — IV dextrose GIR 8-10 mandatory
  9. Liver transplant: May stabilise hepatic disease in hepatic-only phenotype; does NOT prevent
     neurological depletion in hepatocerebral form — brain depletion continues
 10. IOSCA (Finnish founder p.Y508C/c.1523A>G) — less severe; ataxia + SNHL + neuropathy; no liver failure
 11. adPEO (heterozygous) — MULTIPLE mtDNA deletions (not depletion); adult-onset; not treated as MDDS
 12. Spelbrink 2001 Nat Genet — first description (adPEO); Nikali 2005 Neurology — IOSCA

TWNK BIOLOGY:
TWNK (684 amino acids, 10q24.31) is a 5'→3' DNA helicase (SF4 superfamily, bacteriophage T7 gp4-like).
It forms a hexameric ring and unwinds the mitochondrial DNA double helix at the
replication fork, threading the template strand through its central channel so that
POLG can polymerise the new strand. Without TWNK helicase activity, POLG stalls
and mtDNA cannot be replicated.

Domain architecture:
  MTS (aa 1-42): mitochondrial targeting sequence; cleaved in matrix
  N-terminal Primase-like / Zinc-binding domain (aa 43-~340):
    - In bacteriophage T7 gp4 this domain synthesises RNA primers;
    - In human TWNK the primase catalytic residues are degenerated (non-functional primase);
    - Retains structural zinc-binding (CXXC motifs); required for hexamer assembly;
    - Linker helix (aa ~280-330): connects N to C domain; adPEO missense cluster
  C-terminal RecA-like Helicase domain (aa ~341-684):
    - Walker A (P-loop): Gly-x-Gly-x-x-Gly aa ~427-432 — ATP binding
    - Walker B: Asp431 — Mg2+ coordination, ATP hydrolysis
    - Arginine finger Arg303 — inter-subunit communication
    - DNA-binding loops: aa ~560-580 ssDNA contact
    - Subunit interface: missense variants causing adPEO cluster in linker/N-domain,
      impairing hexamer assembly without complete loss (dominant negative mechanism)

Hexameric ring mechanism:
  Six TWNK subunits form a ring; dsDNA binds centrally; sequential ATP hydrolysis
  (hand-over-hand) drives 5'→3' translocation along the lagging-strand template.
  POLG (p140) follows, synthesising the leading strand.
  mtSSB (mitochondrial single-stranded DNA-binding protein) stabilises the exposed ssDNA.
  Loss of function (biallelic) → replication fork collapse → mtDNA depletion.
  Dominant negative (linker missense) → stalling → multiple deletions, not depletion.

GENOTYPE-PHENOTYPE CORRELATION:
  Biallelic null/frameshift → MDDS7 (severe hepatocerebral, infantile death)
  Biallelic p.W315L or p.A318T (linker) compound/homozygous → MDDS7 moderate / IOSCA-overlap
  Biallelic p.Y508C (Walker B adjacent, Finnish founder, 1 in 10,000 Finnish) → IOSCA
    (preserved helicase activity ~10-20%; selective cerebellar/sensory neuron vulnerability)
  Heterozygous linker missense (p.A318T, p.L381P, p.R374Q etc.) → adPEO
    (dominant negative hexamer disruption → multiple deletions; NOT depletion)

PATHOGENIC VARIANT DISTRIBUTION (AR biallelic MDDS7/IOSCA, n=40, seed-555):
  p.W315L compound het: ~20% — linker; moderate-severe hepatocerebral
  Frameshift/nonsense homozygous: ~20% — null; severe neonatal/infantile hepatocerebral
  p.Y508C homozygous: ~15% — Finnish founder; IOSCA phenotype
  p.A318T compound het: ~15% — linker; moderate hepatocerebral/IOSCA-overlap
  p.R391H compound het: ~10% — helicase domain; severe
  Splice site (IVS2+1G>A or exon 3 del) compound: ~10% — null allele + partial
  Other missense/missense compound het: ~10% — variable severity
"""

import random
from datetime import date

SEED = 555  # 40-patient cohort seed


def get_overview() -> dict:
    """TWNK Hepatocerebral mtDNA Depletion / IOSCA — overview for /api/twnk/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": (
            "Mitochondrial DNA Depletion Syndrome 7 (MDDS7) / TWNK Hepatocerebral mtDNA Depletion / "
            "Infantile-Onset Spinocerebellar Ataxia (IOSCA) / adPEO-2 (heterozygous)"
        ),
        "gene": (
            "TWNK (C10orf2); Mitochondrial DNA Helicase (Twinkle); "
            "5'→3' SF4 Helicase; Hexameric Ring; mtDNA Replication Fork Unwinding; "
            "684 aa (MTS + Primase-like N-domain + RecA-like Helicase C-domain); Mitochondrial Matrix"
        ),
        "chromosome": "10q24.31",
        "omim_gene": "606075",
        "omim_disease_mdds7": "271245",
        "omim_disease_adpeo": "609286",
        "inheritance": (
            "MDDS7/IOSCA: Autosomal Recessive (biallelic TWNK); "
            "adPEO-2: Autosomal Dominant (heterozygous dominant negative / haploinsufficiency)"
        ),
        "prevalence": (
            "MDDS7 rare globally (<1:1,000,000); IOSCA ~1:25,000 in Finland (founder p.Y508C); "
            "adPEO-2 ~1:100,000 (most common cause of adPEO after ANT1)"
        ),
        "protein": (
            "TWNK 684 aa (MTS aa1-42, cleaved in matrix); SF4 5'→3' DNA helicase; "
            "hexameric ring; N-domain primase-like (zinc-binding, assembly); "
            "C-domain RecA helicase (Walker A Gly427-432, Walker B Asp431, ATP hydrolysis); "
            "Linker helix (aa280-330) — adPEO missense cluster"
        ),
        "category": (
            "mtDNA Depletion Syndrome / Mitochondrial DNA Maintenance / "
            "TWNK Helicase Defect / Hepatocerebral MDDS / IOSCA / adPEO"
        ),
        "first_described": (
            "Spelbrink JN et al. 2001 Nat Genet — TWNK (C10orf2) mutations in adPEO; "
            "Nikali K et al. 2005 Neurology — IOSCA / MDDS7 (TWNK recessive)"
        ),
        "kpis": {
            "hepatocerebral_pct": 75,
            "hepatic_only_pct": 25,
            "lactic_acidosis_pct": 100,
            "hypoglycemia_pct": 70,
            "liver_failure_pct": 85,
            "nystagmus_pct": 0,
            "three_mga_pct": 0,
            "ck_elevated_pct": 0,
            "iosca_ataxia_pct": 15,
            "vpa_risk": "ABSOLUTE CONTRAINDICATION — mtDNA depletion disease; hepatotoxicity",
            "liver_transplant_hepatocerebral": "Does NOT prevent brain mtDNA depletion in hepatocerebral form",
            "liver_transplant_hepatic_only": "May be curative in hepatic-only phenotype (25%)",
            "no_nystagmus": "ABSENT — KEY DDx from DGUOK (nystagmus 90% PATHOGNOMONIC)",
            "no_3mga": "ABSENT — KEY DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB",
        },
        "clinical_highlights": [
            "HEPATOCEREBRAL phenotype (75%) — liver failure + progressive neurodegeneration; "
            "lactic acidosis 100%; hypoglycemia 70%; early death without OLT",
            "HEPATIC-ONLY phenotype (25%) — preserved CNS; liver failure treatable by OLT; "
            "OLT may be curative if performed before neurological involvement",
            "NO nystagmus — nystagmus is DGUOK-specific (90% pathognomonic); "
            "TWNK lacks nystagmus — critical DDx from DGUOK",
            "NO 3-MGA-uria — critical DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB",
            "Liver transplant — DOES NOT prevent neurological depletion in hepatocerebral form; "
            "brain mtDNA depletion proceeds independently of liver disease",
            "Lactic acidosis 100% in MDDS7 — severe; pH <7.1 possible neonatal; "
            "bicarbonate/THAM resuscitation; GIR 8-10 for glucose homeostasis",
            "IOSCA phenotype (Finnish founder p.Y508C, 15%) — less severe; "
            "infantile-onset cerebellar ataxia + SNHL + sensory neuropathy; NO liver failure",
            "adPEO-2 (heterozygous) — MULTIPLE deletions, NOT depletion; adult-onset; "
            "PEO + ptosis + limb-girdle; cardiomyopathy in some; NOT treated as MDDS",
            "Peripheral neuropathy 50-60% in MDDS7 — sensorimotor demyelinating; "
            "differentiates from TK2 (myopathic, no neuropathy) and MPV17 (neuropathy 80%)",
            "mtDNA depletion <30% in liver and brain — diagnostic; quantify both tissues if possible",
            "LEV preferred AED — renal excretion; no hepatic metabolism; safe in liver failure",
            "Propofol AVOID — PRIS in mitochondrial disease; ketamine + sevoflurane preferred",
        ],
        "contraindications": [
            {
                "drug": "Valproic Acid (VPA)",
                "severity": "ABSOLUTE",
                "reason": (
                    "mtDNA depletion disease — VPA inhibits POLG (DNA polymerase gamma), "
                    "CoA sequestration by valproyl-CoA, reactive epoxide metabolite; "
                    "lethal hepatotoxicity risk in all MDDS (POLG, DGUOK, MPV17, TK2, TWNK)"
                ),
            },
            {
                "drug": "Ketogenic Diet (KD)",
                "severity": "CONTRAINDICATED",
                "reason": (
                    "Forces OXPHOS-dependent beta-oxidation; fails in mtDNA depletion where "
                    "OXPHOS (ETC complexes I/III/IV/V all mtDNA-encoded) capacity is severely reduced"
                ),
            },
            {
                "drug": "Propofol",
                "severity": "AVOID",
                "reason": (
                    "Propofol Infusion Syndrome (PRIS) — inhibits mitochondrial fatty-acid oxidation "
                    "and ETC complex II; dangerous in mitochondrial disease; "
                    "use ketamine + sevoflurane as anaesthetic alternatives"
                ),
            },
            {
                "drug": "Fasting (>4-6 hours without IV glucose)",
                "severity": "CONTRAINDICATED",
                "reason": (
                    "Hypoglycemia risk 70%; metabolic crisis precipitant; "
                    "IV dextrose at GIR 8-10 mg/kg/min mandatory during procedures, illness, and nil-by-mouth periods"
                ),
            },
        ],
        "thresholds": [
            {
                "parameter": "mtDNA copy number (liver/brain biopsy)",
                "threshold": "<30% normal",
                "action": "Diagnostic — confirms mtDNA depletion; quantify in both tissues",
            },
            {
                "parameter": "Blood glucose",
                "threshold": "<3.0 mmol/L",
                "action": "IV dextrose GIR 8-10 mg/kg/min; avoid fasting; continuous glucose monitoring",
            },
            {
                "parameter": "Lactate (plasma)",
                "threshold": ">5 mmol/L",
                "action": "Metabolic crisis — bicarbonate/THAM; ICU level care; reassess triggers",
            },
            {
                "parameter": "ALT/AST",
                "threshold": ">10× ULN",
                "action": "Hepatic decompensation — OLT referral; assess neurological status; VPA zero tolerance",
            },
            {
                "parameter": "INR",
                "threshold": ">2.0 (not corrected by Vitamin K)",
                "action": "Synthetic liver failure — urgent OLT assessment; FFP bridge only",
            },
            {
                "parameter": "GCS / developmental regression",
                "threshold": "Any acute regression",
                "action": "Brain mtDNA crisis — metabolic stabilisation; LP for CSF lactate; "
                "reassess OLT timing (hepatocerebral: OLT may no longer benefit CNS)",
            },
        ],
        "ddx_table": [
            {
                "disease": "TWNK MDDS7 (this)",
                "hepatopathy": "Yes (75-85%)",
                "nystagmus": "No",
                "three_mga": "No",
                "lactic_acidosis": "100% severe",
                "ck": "Normal",
                "primary_organ": "Liver/Brain",
                "vpa_ci": "Absolute",
            },
            {
                "disease": "DGUOK MDDS3",
                "hepatopathy": "Yes (75%)",
                "nystagmus": "90% PATHOGNOMONIC",
                "three_mga": "No",
                "lactic_acidosis": "100% severe",
                "ck": "Normal",
                "primary_organ": "Liver/Brain",
                "vpa_ci": "Absolute",
            },
            {
                "disease": "MPV17 MDDS6",
                "hepatopathy": "Yes (90%)",
                "nystagmus": "No",
                "three_mga": "No",
                "lactic_acidosis": "100% severe",
                "ck": "Normal",
                "primary_organ": "Liver/Brain",
                "vpa_ci": "Absolute",
            },
            {
                "disease": "POLG (Alpers)",
                "hepatopathy": "Yes (80%)",
                "nystagmus": "Rare",
                "three_mga": "No",
                "lactic_acidosis": "100% severe",
                "ck": "Mild",
                "primary_organ": "Liver/Brain",
                "vpa_ci": "Absolute",
            },
            {
                "disease": "TK2 MDDS4A",
                "hepatopathy": "No",
                "nystagmus": "No",
                "three_mga": "No",
                "lactic_acidosis": "Mild/Normal",
                "ck": "Elevated 90%",
                "primary_organ": "Muscle",
                "vpa_ci": "Absolute",
            },
            {
                "disease": "SERAC1 MEGDEL",
                "hepatopathy": "Possible",
                "nystagmus": "Possible",
                "three_mga": "Yes (Type IV)",
                "lactic_acidosis": "Common",
                "ck": "Variable",
                "primary_organ": "Brain/Liver",
                "vpa_ci": "Caution",
            },
            {
                "disease": "TMEM70 MDDS",
                "hepatopathy": "Rare",
                "nystagmus": "No",
                "three_mga": "Yes (Type VI)",
                "lactic_acidosis": "100% neonatal",
                "ck": "Normal",
                "primary_organ": "Heart/Brain",
                "vpa_ci": "Absolute",
            },
            {
                "disease": "IOSCA (TWNK p.Y508C)",
                "hepatopathy": "No",
                "nystagmus": "No",
                "three_mga": "No",
                "lactic_acidosis": "Mild/absent",
                "ck": "Normal",
                "primary_organ": "Cerebellum/Sensory",
                "vpa_ci": "Caution",
            },
        ],
    }


def get_breakdown() -> dict:
    """TWNK MDDS7 / IOSCA — 40-patient cohort breakdown for /api/twnk/breakdown."""
    rng = random.Random(SEED)

    phenotypes = [
        ("MDDS7 — Hepatocerebral (liver + brain depletion)", 30),
        ("MDDS7 — Hepatic-Only (preserved CNS)", 10),
    ]

    genotypes = [
        {
            "variant": "p.W315L compound het (+ null allele)",
            "n": 8,
            "phenotype": "Linker helix disruption; moderate-severe hepatocerebral; onset 2-6 months",
            "residual_activity": "~5-10%",
            "mechanism": "Linker helix (aa315); hexamer assembly impaired; mtDNA depletion in liver + brain",
        },
        {
            "variant": "Frameshift/nonsense homozygous",
            "n": 8,
            "phenotype": "Null — severe neonatal/infantile hepatocerebral; death 3-12 months without OLT",
            "residual_activity": "0%",
            "mechanism": "Complete loss of helicase; replication fork collapse; deepest mtDNA depletion",
        },
        {
            "variant": "p.Y508C homozygous (Finnish founder)",
            "n": 6,
            "phenotype": "IOSCA — ataxia + SNHL + neuropathy; NO liver failure; less severe",
            "residual_activity": "~10-20%",
            "mechanism": "Walker B adjacent; partial helicase activity; selective cerebellar/sensory neuron vulnerability",
        },
        {
            "variant": "p.A318T compound het",
            "n": 6,
            "phenotype": "Linker helix; moderate hepatocerebral; onset 1-4 months",
            "residual_activity": "~8%",
            "mechanism": "Alanine 318 in linker helix; hexamer mis-assembly; mtDNA depletion",
        },
        {
            "variant": "p.R391H compound het",
            "n": 4,
            "phenotype": "Helicase domain; severe hepatocerebral; early infantile",
            "residual_activity": "~3%",
            "mechanism": "Arginine finger equivalent; inter-subunit communication loss",
        },
        {
            "variant": "Splice site (IVS2+1G>A / exon 3 del) compound",
            "n": 4,
            "phenotype": "Null + partial; moderate-severe hepatocerebral",
            "residual_activity": "~5%",
            "mechanism": "Null allele + missense; compound het mtDNA depletion",
        },
        {
            "variant": "Other missense/missense compound het",
            "n": 4,
            "phenotype": "Variable — hepatocerebral to hepatic-only",
            "residual_activity": "Variable 5-20%",
            "mechanism": "Multiple structural sites; correlation with residual activity",
        },
    ]

    def age_at_onset():
        r = rng.random()
        if r < 0.45:
            return f"{rng.randint(1, 3)} months"
        elif r < 0.70:
            return f"{rng.randint(3, 12)} months"
        elif r < 0.85:
            return f"{rng.randint(12, 24)} months"
        else:
            return f"{rng.randint(24, 72)} months"  # IOSCA later onset

    patients = []
    pid = 1
    for phenotype_name, count in phenotypes:
        for _ in range(count):
            onset = age_at_onset()
            geno = rng.choice(genotypes)
            is_hep_cer = "Hepatocerebral" in phenotype_name
            liver_fail = rng.random() < (0.90 if is_hep_cer else 0.60)
            neuro_involved = rng.random() < (0.85 if is_hep_cer else 0.10)
            olt = rng.random() < (0.45 if liver_fail else 0.05)
            niv = rng.random() < 0.20
            seizures = rng.random() < (0.55 if neuro_involved else 0.10)
            peripheral_neuropathy = rng.random() < (0.55 if is_hep_cer else 0.20)
            mtdna_pct = rng.randint(5, 25) if is_hep_cer else rng.randint(20, 45)
            glucose_gir = rng.random() < 0.70
            lactate = round(rng.uniform(4.0, 12.0) if is_hep_cer else rng.uniform(1.5, 4.5), 1)

            patients.append({
                "id": f"TWNK-{pid:03d}",
                "phenotype": phenotype_name.split("—")[0].strip(),
                "age_at_onset": onset,
                "genotype": geno["variant"],
                "residual_activity": geno["residual_activity"],
                "liver_failure": liver_fail,
                "neurological_involvement": neuro_involved,
                "received_olt": olt,
                "received_niv": niv,
                "seizures": seizures,
                "peripheral_neuropathy": peripheral_neuropathy,
                "mtdna_copy_pct": mtdna_pct,
                "required_iv_glucose_gir": glucose_gir,
                "peak_lactate_mmol": lactate,
            })
            pid += 1

    # Aggregate feature prevalence
    total = len(patients)
    feature_prevalence = [
        {
            "feature": "Lactic Acidosis",
            "pct": 100,
            "note": "Universal in MDDS7; severity correlates with metabolic decompensation",
        },
        {
            "feature": "Liver Failure (any degree)",
            "pct": round(sum(1 for p in patients if p["liver_failure"]) / total * 100),
            "note": "Hepatic decompensation — elevation of transaminases, coagulopathy, synthetic failure",
        },
        {
            "feature": "Neurological Involvement",
            "pct": round(sum(1 for p in patients if p["neurological_involvement"]) / total * 100),
            "note": "Encephalopathy, developmental regression, white matter changes",
        },
        {
            "feature": "Hypoglycemia requiring IV GIR",
            "pct": round(sum(1 for p in patients if p["required_iv_glucose_gir"]) / total * 100),
            "note": "IV dextrose GIR 8-10 mg/kg/min; continuous monitoring",
        },
        {
            "feature": "Seizures",
            "pct": round(sum(1 for p in patients if p["seizures"]) / total * 100),
            "note": "Predominantly secondary to metabolic derangement; LEV preferred",
        },
        {
            "feature": "Peripheral Neuropathy",
            "pct": round(sum(1 for p in patients if p["peripheral_neuropathy"]) / total * 100),
            "note": "Sensorimotor demyelinating; DDx from TK2 (no neuropathy) and MPV17 (80%)",
        },
        {
            "feature": "Received Liver Transplant (OLT)",
            "pct": round(sum(1 for p in patients if p["received_olt"]) / total * 100),
            "note": "Hepatocerebral: OLT does NOT prevent brain depletion; hepatic-only: may be curative",
        },
        {
            "feature": "Nystagmus",
            "pct": 0,
            "note": "ABSENT — critical DDx from DGUOK (90% nystagmus, pathognomonic)",
        },
        {
            "feature": "3-MGA-uria",
            "pct": 0,
            "note": "ABSENT — critical DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB",
        },
        {
            "feature": "Elevated CK",
            "pct": 0,
            "note": "ABSENT — CK elevation = TK2 myopathic marker; not hepatocerebral MDDS",
        },
    ]

    treatments = [
        {
            "tx": "Valproic Acid (VPA) — ALL INDICATIONS",
            "level": "ABSOLUTE CONTRAINDICATION",
            "note": (
                "NEVER prescribe in TWNK MDDS7 or ANY mtDNA depletion syndrome. "
                "Three mechanisms of mitotoxicity: POLG inhibition, CoA sequestration, epoxide reactive metabolite. "
                "Document VPA allergy in medical records. No safe dose exists."
            ),
        },
        {
            "tx": "Ketogenic Diet (KD)",
            "level": "CONTRAINDICATED",
            "note": (
                "Forces OXPHOS-dependent beta-oxidation. "
                "OXPHOS fails in mtDNA depletion (ETC complexes I, III, IV, V all mtDNA-encoded). "
                "KD worsens metabolic crisis risk in MDDS7."
            ),
        },
        {
            "tx": "Propofol",
            "level": "AVOID — PRIS Risk",
            "note": (
                "Propofol Infusion Syndrome: mitochondrial ETC complex II inhibition + "
                "fatty-acid oxidation inhibition. Use ketamine + sevoflurane for anaesthesia. "
                "Document AVOID in anaesthetic records."
            ),
        },
        {
            "tx": "IV Dextrose (GIR 8-10 mg/kg/min)",
            "level": "A — Mandatory in Decompensation",
            "note": (
                "Hypoglycemia 70%; nil-by-mouth periods must include IV dextrose at GIR 8-10. "
                "Continuous glucose monitoring. Avoid fasting >4-6 hours at any age."
            ),
        },
        {
            "tx": "Sodium Bicarbonate / THAM",
            "level": "A — Acute Metabolic Crisis",
            "note": (
                "Severe lactic acidosis (pH <7.15 or lactate >7 mmol/L): "
                "IV sodium bicarbonate or THAM resuscitation. "
                "THAM preferred when CO2 retention present (hepatic failure + respiratory compromise)."
            ),
        },
        {
            "tx": "Levetiracetam (LEV)",
            "level": "A — Preferred AED",
            "note": (
                "Renal excretion only; no hepatic metabolism; safe in liver failure. "
                "IV loading 20-40 mg/kg for acute seizures. "
                "Titrate to response. Avoid enzyme-inducing AEDs (phenytoin complex I inhibition; "
                "carbamazepine/phenobarb hepatic induction complicates monitoring)."
            ),
        },
        {
            "tx": "Liver Transplantation (OLT)",
            "level": "B — Hepatic-Only ONLY; NOT Hepatocerebral",
            "note": (
                "Hepatic-only TWNK (25%): OLT may be curative — liver depletion removed, "
                "CNS spared, survival possible. "
                "Hepatocerebral TWNK (75%): OLT does NOT prevent or halt brain mtDNA depletion; "
                "neurological decline continues post-OLT. Thorough neurological evaluation mandatory "
                "pre-OLT to assess CNS involvement."
            ),
        },
        {
            "tx": "CoQ10 Supplementation",
            "level": "C — Supportive",
            "note": (
                "Antioxidant; partial ETC support; no controlled TWNK-specific data. "
                "10-30 mg/kg/day oral; often combined with riboflavin in mitochondrial cocktail."
            ),
        },
        {
            "tx": "Riboflavin (Vitamin B2)",
            "level": "C — Supportive",
            "note": (
                "Complex I/II cofactor (FAD); some patients show mild biochemical improvement. "
                "10-100 mg/day oral; part of mitochondrial supportive cocktail."
            ),
        },
        {
            "tx": "Carnitine Supplementation",
            "level": "C — If Deficiency Documented",
            "note": (
                "Secondary carnitine deficiency common in liver failure and organic acidurias. "
                "Supplement only if low free carnitine; avoid over-replacement."
            ),
        },
        {
            "tx": "Genetic Counselling",
            "level": "A — Mandatory",
            "note": (
                "AR inheritance: 25% recurrence risk per pregnancy. "
                "Prenatal diagnosis (CVS at 11-13 weeks or amniocentesis) available for known variants. "
                "Preimplantation genetic testing (PGT-M) available for known pathogenic variants."
            ),
        },
    ]

    disease_timeline = [
        {
            "phase": "Neonatal / Early Infantile (0-3 months)",
            "events": (
                "Lactic acidosis (often at birth or triggered by intercurrent illness); "
                "hypoglycemia; hepatomegaly; elevated transaminases; hypotonia; feeding difficulties. "
                "Severe null mutations may present with neonatal fulminant hepatic failure."
            ),
        },
        {
            "phase": "Infantile (3-12 months)",
            "events": (
                "Progressive liver failure (jaundice, coagulopathy, ascites); "
                "developmental plateau or regression; seizures (if hepatocerebral); "
                "MRI: T2 signal changes in basal ganglia/white matter (hepatocerebral form). "
                "Hepatic-only form: CNS preserved; OLT candidacy assessed."
            ),
        },
        {
            "phase": "OLT Assessment Window (typically 6-18 months)",
            "events": (
                "Critical neurological assessment before OLT: "
                "brain MRI + CSF lactate + neurological exam. "
                "Hepatic-only: OLT with curative intent. "
                "Hepatocerebral: OLT outcome poor (neurological decline continues). "
                "Panel decision required; OLT centre + metabolic neurology joint review."
            ),
        },
        {
            "phase": "Post-OLT / IOSCA Natural History",
            "events": (
                "Hepatic-only post-OLT: liver function normalises; follow neurological status annually. "
                "Hepatocerebral post-OLT: neurological decline despite stable liver; "
                "progressive encephalopathy; median survival 2-4 years post-OLT. "
                "IOSCA (p.Y508C): progressive cerebellar ataxia, SNHL, sensory neuropathy; "
                "ambulation typically preserved into teenage years; SNHL requires hearing aids."
            ),
        },
        {
            "phase": "Terminal / Supportive Care",
            "events": (
                "Hepatocerebral: respiratory compromise, dysphagia, palliative nasogastric/PEG; "
                "NIV if respiratory failure; comfort-focused care discussion early; "
                "bereavement support for family. "
                "IOSCA: most survive to adulthood with supported living."
            ),
        },
    ]

    liver_data = {
        "mtdna_depletion_liver_pct": 8,
        "mtdna_depletion_brain_pct": 12,
        "mtdna_threshold_diagnostic": "<30% normal in liver and/or brain",
        "oxphos_deficiency": "Complex I/III/IV/V all reduced (all mtDNA-encoded subunits depleted); "
        "Complex II relatively spared (all nuclear-encoded)",
        "histology": "Panlobular hepatocyte necrosis; microvesicular steatosis; "
        "cholestasis; periportal fibrosis in survivors",
        "electron_microscopy": "Swollen, pleomorphic mitochondria; loss of cristae; "
        "matrix densification — non-specific mitochondrial changes",
    }

    olt_outcomes = {
        "hepatic_only_olt_n": 5,
        "hepatic_only_olt_curative_pct": 70,
        "hepatocerebral_olt_n": 7,
        "hepatocerebral_olt_neurological_progression_pct": 95,
        "hepatocerebral_median_survival_post_olt_yr": 2.5,
        "hepatic_only_median_survival_post_olt_yr": 12,
        "note": (
            "Hepatic-only OLT: most common curative option; CNS protected. "
            "Hepatocerebral OLT: stabilises liver but brain depletion continues; "
            "95% show neurological progression post-OLT."
        ),
    }

    return {
        "generated": date.today().isoformat(),
        "cohort_n": 40,
        "seed": SEED,
        "phenotype_distribution": [
            {"name": name, "n": count, "pct": round(count / 40 * 100)}
            for name, count in phenotypes
        ],
        "genotype_breakdown": genotypes,
        "feature_prevalence": feature_prevalence,
        "treatments": treatments,
        "disease_timeline": disease_timeline,
        "liver_pathology": liver_data,
        "olt_outcomes": olt_outcomes,
        "patients_sample": patients[:8],
    }


def get_definitions() -> dict:
    """TWNK MDDS7 — definitions for /api/twnk/definitions."""
    return {
        "generated": date.today().isoformat(),
        "terms": [
            {
                "term": "TWNK / C10orf2 / Twinkle",
                "definition": (
                    "TWNK (chromosome 10 open reading frame 2, also known as 'Twinkle') encodes a "
                    "684-amino-acid mitochondrial 5'→3' DNA helicase. It belongs to the SF4 helicase "
                    "superfamily and is structurally and functionally homologous to bacteriophage T7 gene "
                    "product 4 (gp4), which combines primase and helicase activities. In humans, the "
                    "N-terminal primase-like domain has lost catalytic primase activity but retains "
                    "a zinc-binding fold important for hexamer assembly. The C-terminal RecA-like "
                    "helicase domain hydrolyses ATP (Walker A/B motifs) to drive 5'→3' DNA unwinding "
                    "at the mitochondrial replication fork. TWNK forms a hexameric ring and processively "
                    "unwinds the dsDNA template ahead of POLG (mitochondrial DNA polymerase gamma), "
                    "generating the single-stranded template that POLG copies."
                ),
            },
            {
                "term": "MDDS7 — Mitochondrial DNA Depletion Syndrome 7",
                "definition": (
                    "MDDS7 (OMIM #271245) is an autosomal recessive disease caused by biallelic "
                    "loss-of-function mutations in TWNK. Loss of TWNK helicase activity collapses "
                    "the mitochondrial replication fork, resulting in mtDNA copy number depletion "
                    "(<30% normal) in liver and brain. The primary clinical manifestation is "
                    "hepatocerebral mtDNA depletion syndrome: infantile-onset liver failure with "
                    "lactic acidosis, hypoglycemia, and progressive neurodegeneration. A hepatic-only "
                    "subset (25%) lacks CNS involvement and may benefit from liver transplantation."
                ),
            },
            {
                "term": "IOSCA — Infantile-Onset Spinocerebellar Ataxia",
                "definition": (
                    "IOSCA (OMIM #271245, allelic with MDDS7) is caused by homozygosity for the "
                    "Finnish founder variant p.Y508C in TWNK. The p.Y508C substitution is adjacent "
                    "to the Walker B motif in the C-terminal helicase domain and retains partial "
                    "helicase activity (~10-20% of normal), sufficient to prevent catastrophic liver "
                    "failure but insufficient for normal mtDNA replication in metabolically demanding "
                    "neurons. IOSCA is characterised by infantile-onset cerebellar ataxia (1-2 years), "
                    "sensorineural hearing loss (SNHL), sensory neuropathy, and ophthalmoplegia. "
                    "No liver failure. Finnish population prevalence ~1:25,000."
                ),
            },
            {
                "term": "adPEO-2 — Autosomal Dominant Progressive External Ophthalmoplegia Type 2",
                "definition": (
                    "adPEO-2 (OMIM #609286) is caused by heterozygous missense variants in TWNK, "
                    "predominantly clustering in the linker helix between the N- and C-terminal "
                    "domains (e.g., p.A318T, p.L381P, p.R374Q, p.W315L heterozygous). These variants "
                    "act via a dominant negative mechanism: mutant subunits incorporate into the "
                    "hexameric ring and impair helicase processivity, causing stalling and replication "
                    "slippage that produces multiple mtDNA deletions (rather than depletion). "
                    "adPEO-2 presents in adults (20-50 years) with progressive external ophthalmoplegia, "
                    "ptosis, proximal limb-girdle weakness, and in some cases cardiomyopathy or "
                    "cerebellar ataxia. Serum lactate is often normal or mildly elevated. "
                    "This is NOT an MDDS disease and is managed differently."
                ),
            },
            {
                "term": "mtDNA Depletion vs Multiple Deletions — Critical Distinction",
                "definition": (
                    "mtDNA DEPLETION (biallelic null TWNK → MDDS7): reduced mtDNA copy number "
                    "(quantified by real-time PCR; threshold <30% normal in liver/brain). "
                    "Severe phenotype: hepatocerebral or hepatic infantile disease. AR inheritance. "
                    "mtDNA MULTIPLE DELETIONS (heterozygous missense TWNK → adPEO-2): normal or "
                    "near-normal copy number but multiple rearrangements / partial deletions detectable "
                    "by Southern blot or long-range PCR. Milder phenotype: adult-onset PEO. "
                    "AD inheritance. This distinction is mechanistically and clinically fundamental: "
                    "MDDS7 is a severe early-onset disease; adPEO-2 is a relatively mild adult disease."
                ),
            },
            {
                "term": "Hepatocerebral MDDS",
                "definition": (
                    "A subgroup of mtDNA depletion syndromes in which both the liver and brain are "
                    "primarily affected by mtDNA depletion. Hepatocerebral MDDS includes TWNK MDDS7, "
                    "DGUOK MDDS3, MPV17 MDDS6, and POLG (Alpers-Huttenlocher). Common features: "
                    "infantile-onset liver failure (lactic acidosis, hypoglycemia, coagulopathy, "
                    "hyperammonemia), progressive neurodegeneration (encephalopathy, seizures, "
                    "white matter changes), and universal VPA absolute contraindication."
                ),
            },
            {
                "term": "Walker A / Walker B Motifs (ATP-binding in helicases)",
                "definition": (
                    "Walker A motif (P-loop): Gly-x-Gly-x-x-Gly-Lys-Thr/Ser — binds the β and "
                    "γ phosphates of ATP via the lysine; essential for ATP binding. "
                    "Walker B motif: hhhhDExx (h = hydrophobic) — the aspartate coordinates Mg2+ "
                    "that catalyses phosphodiester hydrolysis; the glutamate activates the water "
                    "nucleophile for γ-phosphate cleavage. "
                    "In TWNK, the Walker A P-loop is at aa~427-432 and Walker B Asp at aa~431. "
                    "p.Y508C (IOSCA) is adjacent to Walker B and partially impairs ATP hydrolysis, "
                    "reducing processivity without abolishing it."
                ),
            },
            {
                "term": "Liver Transplantation (OLT) Decision in Hepatocerebral MDDS",
                "definition": (
                    "Liver transplantation removes the diseased liver (with its mtDNA depletion) "
                    "and replaces it with a donor organ that has normal mtDNA. However, in "
                    "hepatocerebral MDDS (TWNK MDDS7, DGUOK MDDS3, MPV17 MDDS6), the brain "
                    "contains its own mtDNA depletion that is independent of the liver. "
                    "OLT stabilises hepatic disease but has no effect on brain mtDNA content. "
                    "Neurological degeneration continues post-OLT in hepatocerebral forms. "
                    "Key rule: OLT is appropriate in hepatic-only forms (no CNS involvement); "
                    "OLT in hepatocerebral forms provides short-term hepatic stability but "
                    "not neurological protection. Pre-OLT neurological assessment (MRI + CSF "
                    "lactate + clinical) is mandatory to identify the phenotypic subgroup."
                ),
            },
            {
                "term": "VPA Contraindication Mechanism in mtDNA Depletion Syndromes",
                "definition": (
                    "Valproic acid is absolutely contraindicated in all mtDNA depletion syndromes "
                    "(TWNK MDDS7, DGUOK MDDS3, MPV17 MDDS6, TK2 MDDS4A, POLG Alpers) via three "
                    "independent mechanisms: (1) VPA and its metabolites (4-en-VPA) directly inhibit "
                    "POLG (mitochondrial DNA polymerase gamma), reducing mtDNA replication; "
                    "(2) valproyl-CoA sequesters free CoA, impairing beta-oxidation and the TCA cycle "
                    "in cells already energetically compromised by mtDNA depletion; "
                    "(3) the reactive epoxide metabolite of VPA (2-propyl-4-pentenoic acid epoxide) "
                    "is directly hepatotoxic. In patients with pre-existing mtDNA depletion, these "
                    "mechanisms combine to cause fulminant hepatic failure and death. "
                    "VPA must be permanently excluded and documented as an allergy-equivalent."
                ),
            },
            {
                "term": "Nystagmus as DDx Tool in Hepatocerebral MDDS",
                "definition": (
                    "Rotary/pendular nystagmus (90% prevalence) is pathognomonic for DGUOK MDDS3 "
                    "and usually the first sign detected in the neonatal period. "
                    "TWNK MDDS7, MPV17 MDDS6, and POLG Alpers do NOT characteristically produce "
                    "nystagmus. Therefore: nystagmus = suspect DGUOK first; absence of nystagmus "
                    "in hepatocerebral MDDS raises TWNK, MPV17, or POLG as differentials. "
                    "This single clinical sign has high discriminatory value before genetic results "
                    "are available and should be documented on every hepatocerebral MDDS work-up."
                ),
            },
        ],
    }
