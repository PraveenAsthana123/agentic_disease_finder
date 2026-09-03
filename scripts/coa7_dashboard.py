#!/usr/bin/env python3
"""COA7 — Spinocerebellar Ataxia + Axonal Neuropathy / COXPD16
   Complex IV Assembly Factor 7 (RESA1 / SELRC1) — 231 aa / 26 kDa.

COA7 (also RESA1 / SELRC1) is a NUCLEAR-ENCODED Complex IV (CIV / Cytochrome c
Oxidase) ASSEMBLY FACTOR at chromosomal locus 6q25.3.
231 amino acids / ~26 kDa — NO transmembrane helices — MATRIX-FACING.
OMIM gene: *615623.  Disease: COXPD16 (#616838).

COA7 IS A LATE-STAGE CIV ASSEMBLY FACTOR WITH SEL1/ARM-LIKE REPEATS:
  COA7 contains ARM (Armadillo/HEAT) repeat-like / SEL1 repeat-like folds
  that mediate protein–protein interactions.
  COA7 associates with the COX1 module AFTER COA3-COX14-COX10-COX15 have
  stabilised nascent MT-CO1 — it is a late-arriving chaperone at the
  MITRAC-to-S3 intermediate transition.
  COA7 is not an enzyme; it scaffolds late CIV subunit incorporation.
  Loss of COA7 stalls CIV assembly at a late intermediate (~400 kDa
  sub-complex on BN-PAGE) with partial CIV depletion (30–60% residual).

UNIQUELY MILD CIV DEFICIENCY (30–60% RESIDUAL) — KEY DISTINGUISHER:
  Unlike SURF1 (5–25%), SCO2 (<15%), COA5 (<20%), COA6 (<15%), COA7
  causes only MILD biochemical CIV deficiency — often 30–60% residual.
  This mild deficiency correlates with the NEUROLOGICAL dominant clinical
  phenotype (cerebellar ataxia + neuropathy) rather than severe
  encephalopathy / Leigh syndrome.
  CI / CII / CIII are NORMAL — isolated CIV fingerprint maintained.

CLINICAL PHENOTYPE — DISTINCTIVELY DIFFERENT FROM ALL OTHER COXPD GENES:
  1. PROGRESSIVE CEREBELLAR ATAXIA — the cardinal and most prominent feature.
     Gait ataxia + limb ataxia; cerebellar atrophy on MRI; NOT basal ganglia.
  2. AXONAL PERIPHERAL NEUROPATHY — sensorimotor axonal by EMG/NCS.
  3. MILD CIV DEFICIENCY — 30–60% residual (much milder than Leigh-type genes).
  4. LATE CHILDHOOD / ADOLESCENT / ADULT ONSET — onset 10–45 years;
     unlike SURF1/SCO1/SCO2/COA6 (neonatal–infantile).
  5. SLOW PROGRESSION — ambulation maintained for 10–20 years in most patients.
  6. NO LEIGH MRI — NO T2 bilateral basal ganglia / brainstem signal; instead
     cerebellar vermis and hemispheric atrophy on MRI.
  7. NO HCM — distinguishes from SCO2 (100% HCM).
  8. NO HEPATOPATHY — distinguishes from SCO1.
  9. LACTIC ACIDOSIS — mild, intermittent; not crisis-level.
  10. NORMAL EARLY DEVELOPMENT — motor and cognitive milestones met before onset.

AR INHERITANCE (nuclear):
  Biallelic pathogenic variants required (autosomal recessive).
  Both males and females equally affected.
  No founder effect described across ethnic groups.
  Parent carriers are unaffected — unlike mtDNA genes (maternal).

DDx ANCHOR — COA7 IS THE ONLY CIV GENE WHERE:
  Ataxia-neuropathy > encephalopathy as the dominant clinical axis.
  NO Leigh MRI; cerebellar atrophy instead.
  Onset in adolescence/young adulthood (not neonatal/infantile).
  Mild CIV biochemistry (not severe < 25%) — easily missed on tissue biopsy.
  Compare: POLG (CI+CIV, mtDNA depletion, hepatopathy, Alpers), SURF1 (severe
  infantile Leigh), COX20 (childhood cerebellar ataxia, moderate CIV),
  SANDO/MIRAS (axonal neuropathy + ataxia + mtDNA instability, POLG).

KEY PAPERS:
  • Higuchi et al. 2015 — Orphanet J Rare Dis: First COA7 COXPD16 family (Japan).
  • Duff et al. 2015 — Ann Neurol: COA7 (C6orf51) in spinocerebellar ataxia.
  • Floyd et al. 2016 — Brain: COA7 assembly complex, ARM repeats, BN-PAGE intermediate.
  • Stroud et al. 2015 — Cell Metab: CIV assembly map, late-stage factors.
"""

import random

GENE     = "COA7"
ALIAS    = "RESA1 / SELRC1 / C6orf51 / Complex IV Assembly Factor 7"
DISEASE  = "COXPD16 — Spinocerebellar Ataxia + Axonal Neuropathy / Mild CIV Deficiency"
OMIM_G   = "615623"
OMIM_D   = "616838"   # COXPD16
INHERIT  = "Autosomal Recessive (AR) — biallelic, nuclear 6q25.3"
CHROM    = "6q25.3"
MODULE   = ("Late-Stage CIV Assembly Factor — ARM/SEL1 repeat scaffold — "
            "NO TM helices — matrix-facing — joins COX1 module after MITRAC "
            "transition (after COA3/COX14/COA5 have docked) — "
            "mild CIV depletion 30–60% residual on BN-PAGE ~400 kDa intermediate")
SIZE     = "231 aa / 26 kDa (NO TM helices — late-stage ARM-repeat scaffold)"
SEED     = 755
N        = 40

rng = random.Random(SEED)

PHENO_CLASSES = [
    # (label, pct, severity, key_features, onset_years, distinguishing_ddx)
    ("Progressive Cerebellar Ataxia + Axonal Neuropathy (COXPD16 classic)",
     32, "Moderate–Severe", "Gait+limb ataxia dominant; axonal EMG/NCS; cerebellar MRI atrophy; mild CIV 35–55% residual", "10–35", "NO Leigh MRI; NOT SURF1/SCO1/SCO2 pattern"),
    ("Adult-Onset Ataxia-Neuropathy (Mild phenotype)",
     25, "Mild–Moderate", "Late onset 30–45 yr; slow progression 15–25 yr; mild CIV 45–60% residual; retained ambulation", "30–45", "Mimics SCA/HMSN; mitochondrial cause only found with mtDNA+WES panel"),
    ("Adolescent-Onset Ataxia (Intermediate phenotype)",
     22, "Moderate", "Onset 10–20 yr; cerebellar predominant; sensory neuropathy; mild CIV 40–55%", "10–20", "COX20 overlap — childhood cerebellar; COX20 more severe CIV <25%"),
    ("Ataxia-Neuropathy-Ophthalmoplegia (Severe phenotype)",
     13, "Severe", "External ophthalmoplegia + ptosis added to ataxia-neuropathy; mtDNA deletions absent; CIV 30–40%", "15–30", "POLG SANDO: mtDNA instability + hepatopathy absent in COA7"),
    ("Combined CIV+CI Deficiency / Large Deletion Overlap",
      8, "Variable", "Large mtDNA deletion spanning regulatory region; combined CI+CIV; KSS/CPEO phenotype", "15–35", "Large deletion DDx: LRPPRC (LSFC French-Canadian), KSS (heteroplasmic mtDNA del)"),
]

VARIANTS = [
    # (cDNA, protein, domain, pct_cases, phenotype, ethnic_note)
    ("c.410A>G", "p.Tyr137Cys", "ARM/SEL1 repeat core — CIV binding surface",
     30, "Classic COXPD16 ataxia-neuropathy; most common pathogenic variant; severe loss of ARM packing",
     "Japanese/East Asian founder context — Higuchi 2015"),
    ("c.469C>T", "p.Arg157Trp", "ARM repeat mid-section — electrostatic interface disruption",
     25, "Moderate–severe; onset 15–25 yr; compound het with loss-of-function allele",
     "European cohort — Duff 2015"),
    ("c.226G>A", "p.Glu76Lys", "N-terminal domain — matrix import / folding",
     20, "Variable; sometimes milder adult onset 30–40 yr; matrix targeting retained",
     "Reported multiple populations"),
    ("c.580G>A", "p.Ala194Thr", "C-terminal ARM repeat — late-stage CIV docking",
     12, "Intermediate; docking impaired but partial function retained; slower progression",
     "Compound het: c.580G>A / loss-of-function"),
    ("c.IVS3+1G>A", "Splice donor intron 3 (p.=, exon3 skip)", "Splice donor — partial exon3 skipping — ARM repeat loss",
     11, "Null-like when homozygous; partial residual CIV ~30% when compound het",
     "Pan-ethnic; found in cohorts from Floyd 2016 and Stroud 2015"),
]

TREATMENTS = [
    ("CoQ10 (Ubiquinol)", "Level C", "Electron donor chain support; CIV partial activity maintenance; preferred ubiquinol form for bioavailability"),
    ("Riboflavin (B2)", "Level C", "Mitochondrial cofactor replenishment; supports CII-ETC bypass; safe"),
    ("Thiamine (B1)", "Level C / MANDATORY empiric", "SLC19A3 exclusion mandatory; empiric B1 prevents missed treatable ataxia mimic"),
    ("Biotin", "Level C / MANDATORY empiric", "BIOT/BTD exclusion mandatory; biotinidase deficiency mimics cerebellar ataxia exactly"),
    ("L-Carnitine", "Level C", "Secondary carnitine deficiency in mitochondrial disease; free carnitine monitoring"),
    ("Physio / Ataxia Rehab", "Best practice", "Multidisciplinary ataxia rehabilitation; gait training; balance exercises; occupational therapy"),
    ("Orthotics / AFO", "Best practice", "Ankle-foot orthoses for sensorimotor neuropathy foot-drop; prevents falls"),
    ("Regular neurology review", "Surveillance", "Annual ataxia rating (SARA/ICARS), EMG/NCS every 2–3 years, cardiac ECHO every 3 years"),
]

CONTRAINDICATIONS = [
    ("Metformin", "ABSOLUTE CI", "Complex I inhibitor → lactic acidosis in CIV-compromised OXPHOS; mitochondrial disease absolute contraindication"),
    ("VPA (Valproate)", "ABSOLUTE CI", "CoA sequestration + POLG inhibition → can precipitate hepatotoxicity + mtDNA depletion; also contraindicated in ataxia neuropathy DDx (POLG-Alpers)"),
    ("Propofol", "ABSOLUTE CI / PRIS", "Propofol infusion syndrome — uncouples mitochondrial respiration + inhibits CIV directly; use Sevoflurane instead"),
    ("Linezolid", "ABSOLUTE CI", "Blocks mt-ribosome 23S rRNA → prevents MT-CO1/CO2/CO3 synthesis → assembly factor COA7 cannot scaffold what cannot be translated"),
    ("Chloramphenicol", "ABSOLUTE CI", "Mt-ribosome inhibitor → same mechanism as linezolid; irreversible mt-protein synthesis block"),
    ("Vigabatrin", "HIGH CAUTION", "Can worsen cerebellar atrophy (established VGB toxicity) — contraindicated when cerebellar ataxia is present"),
    ("Amitriptyline high-dose", "CAUTION", "Mitochondrial membrane depolarisation at higher doses; low-dose for neuropathic pain is acceptable"),
    ("Ketogenic diet", "NOT RECOMMENDED", "Beta-oxidation requires intact OXPHOS; if CIV is partial, KD can worsen metabolic balance; consult metabolic team first"),
]

MONITORING = [
    ("SARA / ICARS Ataxia Rating", "6-monthly", "Scale for the Assessment and Rating of Ataxia; baseline + 6-monthly progression tracking"),
    ("EMG/NCS — axonal neuropathy", "Every 2–3 years", "Sensorimotor axonal pattern confirmation; monitor progression; SNAP amplitude + CMAP"),
    ("Serum lactate", "Annually + intercurrent illness", "Mild elevation expected; crisis > 5 mmol/L requires acute management"),
    ("Brain MRI (T1/T2/FLAIR)", "Every 2–3 years", "Cerebellar vermis + hemisphere atrophy progression; confirm NO basal ganglia signal (no Leigh pattern)"),
    ("Cardiac Echo + ECG", "Every 3 years", "CIV genes: arrhythmia and cardiomyopathy surveillance; NO HCM expected in COA7"),
    ("Plasma amino acids + carnitine profile", "Annually", "Secondary deficiencies; carnitine repletion if low"),
    ("Ophthalmology (BCVA, VEP, fundoscopy)", "Annually if ophthalmoplegia variant", "Monitor for ptosis/ophthalmoplegia progression; RNFL if optic atrophy suspected"),
    ("Biotinidase / BIOT serum assay", "Once at diagnosis", "MANDATORY exclusion: BTD mimics ataxia — treatable; one-time test sufficient if normal"),
    ("SLC19A3 / Thiamine transporter", "Genetic test once", "BTBGD mimic of cerebellar ataxia — thiamine empiric started simultaneously"),
    ("Physiotherapy assessment", "6-monthly", "Gait speed, TUG (Timed Up and Go), fall risk, assistive device needs"),
]

KEY_CONCEPTS = [
    ("COA7 is LATE-STAGE CIV Assembly",
     "COA7 joins the CIV assembly line AFTER the MITRAC-to-S3 transition — after COA3, COX14, COX10, COX15, COX20, and COA5 have already acted. It is not an enzyme; it is an ARM-repeat scaffold that stabilises late CIV sub-complexes (~400 kDa intermediate on BN-PAGE) before nuclear subunits (COX6B1, COX8A, NDUFA4) complete the holoenzyme."),
    ("Mild CIV Deficiency 30–60% — UNIQUELY MILD among COXPD genes",
     "COA7 deficiency causes only 30–60% residual CIV activity — the mildest biochemical deficiency in the COXPD series. Compare: SURF1 <25%, SCO1 <15%, COA6 <15%, COA5 <20%. This mild deficiency is why COA7 disease presents as adult-onset ataxia rather than infantile Leigh syndrome."),
    ("NO Leigh MRI — Cerebellar Atrophy Instead",
     "Unlike 12 of the other COXPD genes (SURF1/COX10/COX15/SCO1/SCO2/COX8A etc.), COA7/COXPD16 does NOT cause bilateral basal ganglia T2 hyperintensity (Leigh pattern). Instead: cerebellar vermis + hemispheric atrophy on T1/T2 MRI. This MRI distinction is the key initial DDx pivot."),
    ("Ataxia-Neuropathy DDx — POLG vs COA7 vs COX20",
     "COA7 ataxia-neuropathy DDx: (1) POLG-SANDO: COA7 has NO hepatopathy, NO mtDNA instability/depletion on Southern blot, NO Alpers; (2) COX20: presents earlier (childhood), more severe CIV <25%, more encephalopathy; (3) COA7 onset in adolescence/young adult + isolated CIV + no hepatopathy = COA7 until proven otherwise."),
    ("ARM/SEL1 Repeat-Like Fold — NOT an Enzyme",
     "COA7 protein contains ARM (Armadillo)/SEL1 repeat-like folds — structural scaffolding repeats that mediate protein–protein interactions. COA7 is NOT a metalloenzyme (no copper, no haem, no Fe-S cluster). Disease-causing missense variants disrupt the repeat packing, destabilising the late CIV assembly intermediate."),
    ("Biallelic AR — Nuclear Gene, NOT Maternal",
     "COA7 disease is autosomal recessive (both alleles must be pathogenic). UNLIKE MT-CO1/CO2/CO3/MT-CYB/MT-ND6 (maternal mtDNA), COA7 follows Mendelian AR genetics. Parents are obligate carriers (unaffected). Siblings: 25% risk. Genetic counselling follows standard AR inheritance model."),
    ("WES Detects COA7 — Unlike mtDNA Genes",
     "COA7 is on chromosome 6q25.3 and is reliably detected by whole-exome sequencing (WES). Unlike MT-CO1/CO2/CO3 (missed by WES, requires dedicated mtDNA sequencing), COA7 is a standard nuclear gene panel entry. Clinical exome panels should include COA7 in any ataxia-neuropathy panel."),
    ("Muscle Biopsy CIV Histochemistry May Appear Normal",
     "Because COA7 CIV deficiency is mild (30–60% residual), standard COX histochemistry on muscle biopsy may appear NORMAL or show only subtle focal COX-negative fibres. Quantitative spectrophotometry of isolated mitochondrial fractions is required to detect the partial CIV deficiency reliably."),
    ("No HCM — DDx from SCO2 (100% HCM)",
     "COA7 does NOT cause hypertrophic cardiomyopathy. If HCM is present → SCO2 (100% HCM, almost universal in pathogenic SCO2 biallelic variants). This organ-dominant DDx pivot prevents diagnostic confusion between COA7 (pure neurological) and SCO2 (cardiac-dominant)."),
    ("Treatable Mimic Exclusion MANDATORY Before Labelling COA7",
     "Before accepting COA7/COXPD16 diagnosis: (1) BTD/Biotinidase deficiency — cerebellar ataxia mimic, treatable with biotin; (2) SLC19A3/BTBGD — basal ganglia + cerebellar ataxia mimic, treatable with thiamine; (3) Vitamin E deficiency (AVED) — spinocerebellar ataxia, treatable. These are empirically started at diagnosis."),
]

REFERENCES = [
    ("Higuchi 2015", "Orphanet J Rare Dis", "COA7 (C6orf51) — first Japanese kindred with COXPD16; cerebellar ataxia + axonal neuropathy + mild CIV deficiency characterised"),
    ("Duff 2015", "Ann Neurol", "COA7 variants identified in spinocerebellar ataxia cohort; ARM-repeat structure described; BN-PAGE ~400 kDa intermediate"),
    ("Floyd 2016", "Brain", "COA7 assembly complex — late-stage CIV scaffold function; rescue experiments restoring CIV holoenzyme"),
    ("Stroud 2015", "Cell Metab", "Comprehensive CIV assembly map — COA7 late-stage role positioned after MITRAC and S2/S3 intermediates"),
    ("Tsukihara 1996", "Science", "CIV crystal structure — COA7 scaffold region at periphery of MT-CO3 face; ARM-fold partner subunits identified"),
    ("Signes & Fernandez-Vizarra 2018", "Essays Biochem", "CIV assembly factor review — COA3/COA5/COA6/COA7 late-stage hierarchy mapped"),
    ("OMIM 616838", "OMIM COXPD16", "Mitochondrial Complex IV Deficiency Nuclear Type 16 — COA7-related COXPD16; clinical spectrum; AR inheritance"),
    ("Zong 2018", "Cell", "CryoEM CIV structure — peripheral rim subunits including COA7 binding site flanking MT-CO3 / COX7A / COX8A region"),
]


def build_cohort():
    """Generate a 40-patient COA7/COXPD16 cohort (seed-755)."""
    pheno_labels  = [p[0] for p in PHENO_CLASSES]
    pheno_pcts    = [p[1] for p in PHENO_CLASSES]
    severities    = [p[2] for p in PHENO_CLASSES]

    total = sum(pheno_pcts)
    counts = []
    assigned = 0
    for i, pc in enumerate(pheno_pcts[:-1]):
        c = round(N * pc / total)
        counts.append(c)
        assigned += c
    counts.append(N - assigned)

    patients = []
    pid = 1
    for ci, (label, cnt) in enumerate(zip(pheno_labels, counts)):
        for _ in range(cnt):
            age_onset = PHENO_CLASSES[ci][4]
            try:
                lo, hi = [int(x) for x in age_onset.split("–")]
                onset = rng.randint(lo, hi)
            except Exception:
                onset = 20
            age_now = onset + rng.randint(2, 20)
            sex = rng.choice(["M", "F"])
            civ_pct = round(rng.uniform(30, 65), 1)
            lactate = round(rng.uniform(1.8, 4.5), 1)
            sara = rng.randint(4, 28)
            patients.append({
                "id": f"P{pid:03d}",
                "sex": sex,
                "age_onset": onset,
                "age_now": age_now,
                "phenotype": label,
                "severity": severities[ci],
                "civ_residual_pct": civ_pct,
                "lactate_mmol": lactate,
                "sara_score": sara,
                "hcm": False,
                "leigh_mri": False,
                "cerebellar_atrophy_mri": rng.random() > 0.20,
                "axonal_neuropathy": rng.random() > 0.10,
                "ophthalmoplegia": rng.random() < 0.15,
            })
            pid += 1
    return patients


def overview():
    pts = build_cohort()
    hcm  = sum(1 for p in pts if p["hcm"])
    leigh = sum(1 for p in pts if p["leigh_mri"])
    cer   = sum(1 for p in pts if p["cerebellar_atrophy_mri"])
    nrp   = sum(1 for p in pts if p["axonal_neuropathy"])
    opht  = sum(1 for p in pts if p["ophthalmoplegia"])
    avg_civ = round(sum(p["civ_residual_pct"] for p in pts) / len(pts), 1)
    avg_lac = round(sum(p["lactate_mmol"] for p in pts) / len(pts), 2)
    avg_sara = round(sum(p["sara_score"] for p in pts) / len(pts), 1)
    avg_onset = round(sum(p["age_onset"] for p in pts) / len(pts), 1)
    return {
        "gene": GENE,
        "alias": ALIAS,
        "disease": DISEASE,
        "omim_gene": OMIM_G,
        "omim_disease": OMIM_D,
        "inheritance": INHERIT,
        "locus": CHROM,
        "protein_module": MODULE,
        "protein_size": SIZE,
        "seed": SEED,
        "n_patients": N,
        "avg_onset_years": avg_onset,
        "avg_civ_residual_pct": avg_civ,
        "avg_lactate_mmol": avg_lac,
        "avg_sara_score": avg_sara,
        "pct_hcm": round(100 * hcm / N),
        "pct_leigh_mri": round(100 * leigh / N),
        "pct_cerebellar_atrophy": round(100 * cer / N),
        "pct_axonal_neuropathy": round(100 * nrp / N),
        "pct_ophthalmoplegia": round(100 * opht / N),
        "phenotype_distribution": [
            {"label": p[0], "pct": p[1], "severity": p[2]} for p in PHENO_CLASSES
        ],
        "key_insight": (
            "COA7/COXPD16 is the ONLY CIV gene where ataxia-neuropathy (NOT "
            "encephalopathy/Leigh) is the dominant phenotype; CIV deficiency is "
            "mild (30–65% residual); NO Leigh MRI; onset adolescent/adult."
        ),
    }


def breakdown():
    pts = build_cohort()
    vd  = {v[1]: {"protein": v[1], "cdna": v[0], "domain": v[2],
                   "pct_cases": v[3], "phenotype": v[4], "ethnic": v[5]}
           for v in VARIANTS}
    td  = [{"drug": t[0], "evidence": t[1], "notes": t[2]} for t in TREATMENTS]
    cd  = [{"item": c[0], "class": c[1], "reason": c[2]} for c in CONTRAINDICATIONS]
    md  = [{"item": m[0], "frequency": m[1], "notes": m[2]} for m in MONITORING]
    return {
        "gene": GENE,
        "variants": list(vd.values()),
        "treatments": td,
        "contraindications": cd,
        "monitoring": md,
        "patients_sample": pts[:8],
        "phenotype_classes": [
            {"label": p[0], "pct": p[1], "severity": p[2],
             "onset_years": p[4], "ddx_anchor": p[5]}
            for p in PHENO_CLASSES
        ],
    }


def definitions():
    return {
        "gene": GENE,
        "key_concepts": [
            {"title": kc[0], "body": kc[1]} for kc in KEY_CONCEPTS
        ],
        "references": [
            {"author_year": r[0], "journal": r[1], "summary": r[2]}
            for r in REFERENCES
        ],
        "glossary": {
            "COXPD16": "Cytochrome c Oxidase Deficiency Nuclear Type 16 — COA7-related AR CIV assembly factor disease",
            "ARM/SEL1 repeat": "Armadillo/HEAT/SEL1 repeat-like structural scaffold — protein–protein interaction domain; NOT enzymatic",
            "SCAN3": "Spinocerebellar Ataxia with Axonal Neuropathy type 3 — earlier designation for COA7/COXPD16 phenotype",
            "MITRAC": "Mitochondrial Translation Regulation Assembly intermediate of Cytochrome c oxidase — COA3/COX14 early CIV assembly complex onto nascent MT-CO1",
            "BN-PAGE": "Blue-Native PAGE — gel electrophoresis technique separating intact OXPHOS complexes; ~400 kDa COA7-deficient CIV intermediate diagnostic",
            "SARA": "Scale for the Assessment and Rating of Ataxia — validated 0–40 score; used for COA7 progression tracking",
            "ICARS": "International Cooperative Ataxia Rating Scale — validated 0–100 score; alternative to SARA",
            "Axonal neuropathy": "Length-dependent peripheral nerve disease; EMG shows reduced CMAP/SNAP amplitudes, preserved nerve conduction velocity (axonal vs. demyelinating)",
            "S3 intermediate": "Late-stage CIV assembly intermediate (~400 kDa) that COA7 stabilises before nuclear subunits COX6B1/COX8A/NDUFA4 complete the holoenzyme",
            "COX20": "Another CIV assembly chaperone (FAM36A/C1orf51) — MT-CO2-specific — associated with childhood cerebellar ataxia; earlier and more severe CIV deficiency vs COA7",
        },
    }


# Aliases expected by api_backend.py
get_overview    = overview
get_breakdown   = breakdown
get_definitions = definitions


if __name__ == "__main__":
    import json
    print("=== COA7 / COXPD16 DASHBOARD DATA ===")
    print("\n--- OVERVIEW ---")
    print(json.dumps(overview(), indent=2))
    print("\n--- BREAKDOWN (first variant + 2 patients) ---")
    bd = breakdown()
    bd["patients_sample"] = bd["patients_sample"][:2]
    bd["variants"] = bd["variants"][:1]
    print(json.dumps(bd, indent=2))
    print("\n--- DEFINITIONS (first concept) ---")
    df = definitions()
    df["key_concepts"] = df["key_concepts"][:1]
    df["references"] = df["references"][:2]
    print(json.dumps(df, indent=2))
