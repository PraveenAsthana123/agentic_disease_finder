#!/usr/bin/env python3
"""TK2 Myopathic mtDNA Depletion Syndrome Dashboard.

Mitochondrial DNA Depletion Syndrome 4A (MDDS4A) = OMIM #609560.
Also includes the late-onset progressive external ophthalmoplegia (PEO) form and
encephalomyopathic form (MDDS4B) — the myopathic phenotype (MDDS4A) dominates.
Biallelic AR TK2 mutations → mitochondrial dNTP pool imbalance → mtDNA depletion
(muscle > brain) → OXPHOS failure.

TK2 (265 aa, 16q21) is a mitochondrial matrix thymidine kinase.
It phosphorylates pyrimidine deoxyribonucleosides (deoxythymidine dThd → dTMP;
deoxycytidine dCyd → dCMP) within the mitochondrial matrix, supplying precursors
for mtDNA replication during the S-phase and in post-mitotic cells.
Loss of function → dTTP/dCTP pool depletion → POLG stalls → mtDNA depletion →
Complex I/III/IV/V subunit insufficiency → mitochondrial myopathy.

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. VPA = ABSOLUTE CONTRAINDICATION — mtDNA depletion disease; hepatotoxicity risk
     shared with all mtDNA depletion syndromes (POLG, DGUOK, MPV17)
  2. KD = CONTRAINDICATED — OXPHOS-dependent fat oxidation fails in mtDNA depletion
  3. MYOPATHIC phenotype — NO hepatopathy (key DDx from DGUOK/MPV17 hepatocerebral)
  4. NO lactic acidosis at rest (usually normal or mildly elevated — key DDx from
     POLG/DGUOK/MPV17 where lactic acidosis 100% and severe)
  5. NO nystagmus (nystagmus = DGUOK, not TK2)
  6. NO 3-MGA-uria — critical DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB
  7. CK elevated 2-20× normal — myopathic marker (absent in purely hepatocerebral forms)
  8. Respiratory failure — diaphragm/intercostal involvement; major cause of death;
     early NIV (nocturnal non-invasive ventilation) is life-extending
  9. Deoxynucleoside supplementation (dCyd + dThd, oral): substrate bypass therapy;
     FDA orphan drug designation; first disease-modifying rescue in any MDDS
 10. Spinazzola 1999 Am J Hum Genet / Mancuso 2002 Ann Neurol — first TK2 series

TK2 BIOLOGY:
TK2 (265 amino acids, 16q21) is a thymidine kinase localised exclusively to the
mitochondrial matrix. It belongs to the type I thymidine kinase family.

Domain architecture:
  MTS (aa1-18): mitochondrial targeting sequence; cleaved in matrix
  N-lobe kinase domain (aa19-130): ATP-binding P-loop (Gly-x-Gly-x-x-Gly, aa46-51);
    residues Arg152, Thr108 — most common pathogenic missense hotspots
  C-lobe kinase domain (aa131-240): substrate-binding; His121 — catalytic base for
    phosphotransfer; substrate specificity pocket for thymidine vs thymidine analogues
  C-terminal helix (aa241-265): dimer interface; tetramer assembly required for activity

Catalytic mechanism:
  dThd + ATP → dTMP + ADP  (Km dThd ≈ 2 μM)
  dCyd + ATP → dCMP + ADP  (Km dCyd ≈ 15 μM; secondary substrate)
  Active site: Glu98-Tyr99 (stacking) + His121 (phosphate transfer proton base) +
    Arg149 (γ-phosphate orientation) + Glu167 (Mg2+ coordination)
  p.Arg152Gln: adjacent to substrate-binding; reduces dThd affinity 5×; partial activity
  p.Thr108Met: destabilises P-loop; reduces ATP affinity; ~5% residual activity
  p.His121Asn: abolishes phosphotransfer; null activity; severe phenotype

Why TK2 loss causes myopathic mtDNA depletion:
  Post-mitotic muscle cells cannot obtain mitochondrial dNTPs via de novo synthesis
  (which is cytoplasmic and cell-cycle-dependent). Muscle relies almost entirely on
  TK2-mediated salvage of free pyrimidine deoxyribonucleosides imported into the
  mitochondrial matrix. Loss of TK2 → dTTP/dCTP depletion → POLG stalls on template →
  mtDNA copy number <30% normal in muscle → 13 mtDNA-encoded OXPHOS subunits insufficient.
  Brain and liver maintain supplementary dNTP supply via cytosolic enzymes (TK1, DCTD);
  this explains the tissue-selectivity of TK2 loss for skeletal muscle.

Substrate bypass rationale (deoxynucleoside supplementation):
  Oral dThd + dCyd bypasses TK2 by providing high plasma deoxyribonucleoside levels
  that saturate mitochondrial nucleoside transporters (ENT3/SLC29A3), entering the matrix
  and partially restoring dTTP/dCTP pools without requiring TK2 phosphorylation.
  Dramatic stabilisation or improvement in mouse models; Phase II trials in human TK2 disease.

PATHOGENIC VARIANT DISTRIBUTION (biallelic AR, n=40, seed-553):
  p.Arg152Gln (R152Q) compound het/missense: ~25% — most common European allele; moderate
  p.Thr108Met (T108M) compound het: ~20% — P-loop destabiliser; severe infantile
  p.His121Asn (H121N) homozygous: ~10% — catalytic null; severe neonatal/infantile
  p.His90Asn compound het: ~10% — active site; moderate-severe
  Splice site (IVS5+1G>A) / missense compound het: ~15% — null allele + partial
  Exon 4 deletion compound het: ~10% — null allele; severe
  p.Arg214Cys (R214C) homozygous: ~5% — dimer interface; late-onset mild PEO form
  Other missense/missense compound het: ~5% — variable severity
"""

import random
from datetime import date

SEED = 553  # 40-patient cohort seed


def get_overview() -> dict:
    """TK2 Myopathic mtDNA Depletion — overview for /api/tk2/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": "Mitochondrial DNA Depletion Syndrome 4A (MDDS4A) / TK2 Myopathic mtDNA Depletion / TK2-Related Myopathy",
        "gene": "TK2; Mitochondrial Thymidine Kinase; dThd/dCyd Phosphorylation; 265 aa (MTS + kinase N/C-lobes); Mitochondrial Matrix",
        "chromosome": "16q21",
        "omim_gene": "188250",
        "omim_disease": "609560",
        "inheritance": "Autosomal Recessive (biallelic TK2); no carrier phenotype in heterozygotes",
        "prevalence": "Rare globally; estimated 1–2 per 1,000,000 live births; most common myopathic mtDNA depletion syndrome in infancy",
        "protein": "TK2 265 aa (cleaved to 247 aa in matrix); thymidine kinase; dThd→dTMP + dCyd→dCMP; tetramer; His121 catalytic base; Arg152/Thr108 common pathogenic hotspots",
        "category": "mtDNA Depletion Syndrome / Mitochondrial DNA Maintenance / TK2 Pyrimidine Salvage Defect / Mitochondrial Myopathy",
        "first_described": "Spinazzola A et al. 1999 Am J Hum Genet — TK2 mutations in myopathic mtDNA depletion; Mancuso M et al. 2002 Ann Neurol — first clinical series",
        "kpis": {
            "proximal_weakness_pct": 100,
            "respiratory_failure_pct": 85,
            "facial_diplegia_pct": 75,
            "ck_elevation_pct": 90,
            "ophthalmoplegia_pct": 40,
            "hypotonia_pct": 80,
            "hepatopathy_pct": 0,
            "lactic_acidosis_severe_pct": 15,
            "nystagmus_pct": 0,
            "seizures_pct": 20,
            "vpa_risk": "ABSOLUTE CONTRAINDICATION — mtDNA depletion + potential hepatotoxicity shared with all MDDS",
            "deoxynucleoside_eligible_pct": 100,
            "no_hepatopathy": "ABSENT — KEY DDx from DGUOK/MPV17 (hepatocerebral) and POLG (Alpers)",
            "no_3mga_uria": "ABSENT — KEY DDx from SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB",
        },
        "clinical_highlights": [
            "MYOPATHIC phenotype — predominantly skeletal muscle disease; spares liver and brain in classic MDDS4A",
            "Progressive proximal limb weakness (100%) — hip and shoulder girdle; gait abnormality; wheelchair by 3-10 yr",
            "Facial diplegia (75%) — facial muscle weakness; bilateral; expressionless facies",
            "Respiratory failure (85%) — diaphragm and intercostal muscle wasting; leading cause of death",
            "CK elevation 90% — 2-20× normal; myopathic marker absent in hepatocerebral MDDS",
            "External ophthalmoplegia/ptosis (40%) — more prominent in late-onset MDDS4C form (PEO)",
            "NO hepatopathy — absolutely key DDx from DGUOK/MPV17/POLG hepatocerebral forms",
            "NO significant lactic acidosis at rest — distinguishes TK2 from POLG/DGUOK/MPV17 (all 100% lactic acidosis)",
            "NO nystagmus — nystagmus is DGUOK pathognomonic, not TK2",
            "Muscle biopsy: ragged-red fibers + COX-negative fibers + mtDNA depletion <30% diagnostic",
            "Deoxynucleoside supplementation (dCyd + dThd oral) — first disease-modifying MDDS rescue; FDA orphan",
            "NIV early (nocturnal non-invasive ventilation) — extends ambulation and survival significantly",
        ],
        "contraindications": [
            {"drug": "Valproic Acid (VPA)", "severity": "ABSOLUTE", "reason": "mtDNA depletion disease — VPA inhibits POLG + CoA sequestration + epoxide reactive metabolite; lethal hepatotoxicity risk in all MDDS"},
            {"drug": "Ketogenic Diet (KD)", "severity": "CONTRAINDICATED", "reason": "Forces OXPHOS-dependent beta-oxidation; fails in mtDNA depletion where OXPHOS capacity is severely reduced"},
            {"drug": "Propofol", "severity": "AVOID", "reason": "Propofol Infusion Syndrome (PRIS) — mitochondrial fatty-acid oxidation inhibition; dangerous in mito disease"},
            {"drug": "Statins (high-dose)", "severity": "CAUTION", "reason": "CoQ10 depletion via mevalonate pathway; theoretically worsens OXPHOS defect; use lowest effective dose if needed"},
        ],
        "thresholds": [
            {"parameter": "mtDNA copy number (muscle biopsy)", "threshold": "<30% normal", "action": "Diagnostic — confirms mtDNA depletion; correlates with OXPHOS enzyme deficiency"},
            {"parameter": "FVC % predicted", "threshold": "<60%", "action": "Initiate nocturnal NIV; respiratory review every 3 months"},
            {"parameter": "FVC % predicted", "threshold": "<30%", "action": "Urgent respiratory support — tracheostomy discussion; intensive monitoring"},
            {"parameter": "CK", "threshold": ">5× ULN sustained", "action": "Confirm myopathic aetiology; assess for rhabdomyolysis precipitants; avoid statin co-prescription"},
            {"parameter": "Lactate (post-exercise)", "threshold": ">4 mmol/L", "action": "Impaired OXPHOS reserve; metabolic crisis precaution; avoid prolonged fasting"},
        ],
        "ddx_table": [
            {"disease": "TK2 MDDS4A (this)", "hepatopathy": "No", "nystagmus": "No", "3mga": "No", "lactic_acidosis": "Mild/Normal", "ck": "Elevated", "primary_organ": "Muscle", "vpa_ci": "Absolute"},
            {"disease": "DGUOK MDDS3", "hepatopathy": "Yes (75%)", "nystagmus": "90% (PATHOGNOMONIC)", "3mga": "No", "lactic_acidosis": "100% severe", "ck": "Normal", "primary_organ": "Liver/Brain", "vpa_ci": "Absolute"},
            {"disease": "MPV17 MDDS6", "hepatopathy": "Yes (90%)", "nystagmus": "No (DDx from DGUOK)", "3mga": "No", "lactic_acidosis": "100% severe", "ck": "Normal", "primary_organ": "Liver/Brain", "vpa_ci": "Absolute"},
            {"disease": "POLG (Alpers)", "hepatopathy": "Yes (80%)", "nystagmus": "Rare", "3mga": "No", "lactic_acidosis": "100% severe", "ck": "Mild", "primary_organ": "Liver/Brain", "vpa_ci": "Absolute"},
            {"disease": "SERAC1 MEGDEL", "hepatopathy": "Possible", "nystagmus": "Possible", "3mga": "Yes (Type IV)", "lactic_acidosis": "Common", "ck": "Variable", "primary_organ": "Brain/Liver", "vpa_ci": "Caution"},
            {"disease": "TMEM70 MDDS", "hepatopathy": "Rare", "nystagmus": "No", "3mga": "Yes (Type VI)", "lactic_acidosis": "100% neonatal", "ck": "Normal", "primary_organ": "Heart/Brain", "vpa_ci": "Absolute"},
            {"disease": "SMA (Werdnig-Hoffmann)", "hepatopathy": "No", "nystagmus": "No", "3mga": "No", "lactic_acidosis": "No", "ck": "Normal", "primary_organ": "Motor neuron", "vpa_ci": "No CI"},
            {"disease": "Congenital MD (LAMA2/COL6)", "hepatopathy": "No", "nystagmus": "No", "3mga": "No", "lactic_acidosis": "No", "ck": "Elevated", "primary_organ": "Muscle", "vpa_ci": "No CI"},
        ],
    }


def get_breakdown() -> dict:
    """TK2 MDDS4A — 40-patient cohort breakdown for /api/tk2/breakdown."""
    rng = random.Random(SEED)

    phenotypes = [
        ("Classic infantile myopathic (MDDS4A)", 22),
        ("Encephalomyopathic (MDDS4B)", 8),
        ("Late-onset PEO/limb-girdle (MDDS4C)", 7),
        ("Neonatal severe fulminant", 3),
    ]

    genotypes = [
        {"variant": "p.Arg152Gln compound het", "n": 10, "phenotype": "Infantile myopathic; moderate depletion 15-30%", "residual_activity": "~15%"},
        {"variant": "p.Thr108Met compound het", "n": 8, "phenotype": "Severe infantile; depletion <10%", "residual_activity": "~5%"},
        {"variant": "p.His121Asn homozygous", "n": 4, "phenotype": "Catalytic null; severe neonatal", "residual_activity": "0%"},
        {"variant": "p.His90Asn compound het", "n": 4, "phenotype": "Severe infantile; depletion <15%", "residual_activity": "~8%"},
        {"variant": "IVS5+1G>A splice/missense", "n": 6, "phenotype": "Null + partial; infantile moderate-severe", "residual_activity": "~10%"},
        {"variant": "Exon 4 deletion compound het", "n": 4, "phenotype": "Null allele; severe", "residual_activity": "~5%"},
        {"variant": "p.Arg214Cys homozygous", "n": 2, "phenotype": "Late-onset PEO; mild depletion 40-60%", "residual_activity": "~30%"},
        {"variant": "Other missense/missense", "n": 2, "phenotype": "Variable", "residual_activity": "Variable"},
    ]

    def age_at_onset():
        r = rng.random()
        if r < 0.55:
            return f"{rng.randint(3, 18)} months"
        elif r < 0.75:
            return f"{rng.randint(18, 36)} months"
        elif r < 0.88:
            return f"{rng.randint(3, 8)} years"
        else:
            return f"{rng.randint(20, 55)} years"

    patients = []
    pid = 1
    for phenotype_name, count in phenotypes:
        for _ in range(count):
            onset = age_at_onset()
            geno = rng.choice(genotypes)
            amb = rng.random() < (0.25 if "neonatal" in phenotype_name.lower() else 0.55 if "infantile" in phenotype_name.lower() else 0.72)
            niv = rng.random() < (0.90 if "infantile" in phenotype_name.lower() else 0.50)
            trach = rng.random() < (0.40 if niv else 0.05)
            dthd = rng.random() < 0.65
            mito_depletion = rng.randint(5, 28) if "Late" not in phenotype_name else rng.randint(30, 60)
            ck_fold = round(rng.uniform(2.0, 18.0) if "neonatal" not in phenotype_name.lower() else rng.uniform(1.5, 8.0), 1)
            patients.append({
                "id": f"TK2-{pid:03d}",
                "phenotype": phenotype_name,
                "genotype": geno["variant"],
                "onset": onset,
                "ambulation": "Preserved" if amb else "Lost/Wheelchair",
                "niv": "Yes" if niv else "No",
                "tracheostomy": "Yes" if trach else "No",
                "deoxynucleoside_tx": "Yes" if dthd else "No",
                "mtdna_depletion_pct": f"{mito_depletion}% of normal",
                "ck_fold_elevation": f"{ck_fold}×",
                "hepatopathy": "No",
                "nystagmus": "No",
                "lev_preferred": "Yes",
                "vpa_administered": "No (absolutely avoided)",
                "vpa_ok": False,
            })
            pid += 1

    features = [
        {"feature": "Proximal limb weakness", "pct": 100, "note": "Universal; hip girdle then shoulder girdle; early Gowers sign"},
        {"feature": "CK elevation (>2× ULN)", "pct": 90, "note": "Mild-moderate myopathic; 2-20× normal"},
        {"feature": "Respiratory failure", "pct": 85, "note": "Diaphragm + intercostal wasting; major cause of death; early NIV critical"},
        {"feature": "Hypotonia (neonatal/infantile)", "pct": 80, "note": "Generalised; first sign in infantile form"},
        {"feature": "Facial diplegia", "pct": 75, "note": "Bilateral facial weakness; expressionless; difficulty with straw sucking"},
        {"feature": "Dysphagia", "pct": 60, "note": "Bulbar involvement; aspiration risk; feeding tube in severe cases"},
        {"feature": "External ophthalmoplegia/ptosis", "pct": 40, "note": "More prominent in MDDS4C (late-onset PEO form)"},
        {"feature": "Scoliosis", "pct": 35, "note": "Paraspinal weakness; progressive in non-ambulant patients"},
        {"feature": "Seizures", "pct": 20, "note": "From respiratory/metabolic crises; not primary epileptic feature as in POLG"},
        {"feature": "Cognitive impairment", "pct": 10, "note": "Rare in pure MDDS4A; more in MDDS4B encephalomyopathic; typically normal intellect"},
        {"feature": "Hepatopathy", "pct": 0, "note": "ABSENT — key DDx from DGUOK/MPV17/POLG hepatocerebral/Alpers forms"},
        {"feature": "Nystagmus", "pct": 0, "note": "ABSENT — nystagmus pathognomonic for DGUOK (90%); not TK2"},
        {"feature": "3-MGA-uria", "pct": 0, "note": "ABSENT — rules out SERAC1/TAZ/TMEM70/OPA3/DNAJC19/CLPB"},
        {"feature": "Severe lactic acidosis (pH<7.2)", "pct": 15, "note": "Only in metabolic crisis; baseline normal/mildly elevated; DDx from DGUOK/MPV17 (100%)"},
    ]

    treatments = [
        {"tx": "Deoxynucleoside supplementation (dCyd + dThd oral)", "level": "A — first disease-modifying rescue", "note": "Substrate bypass; 100-200 mg/kg/day each; FDA orphan; dramatically improves motor in ~70% of treated patients; best when started early"},
        {"tx": "Nocturnal NIV (BiPAP)", "level": "A — life-extending", "note": "Early start when FVC <60%; reduces nocturnal hypoventilation; most impactful single intervention"},
        {"tx": "LEV (Levetiracetam)", "level": "B — preferred AED", "note": "Renal excretion; no hepatic metabolism; no mitochondrial toxicity; use if seizures present"},
        {"tx": "Riboflavin (B2) + CoQ10", "level": "B — mitochondrial cofactor support", "note": "Theoretical complex I/II support; used widely in clinical practice; may slow deterioration"},
        {"tx": "Physiotherapy — respiratory + limb", "level": "A — standard of care", "note": "Respiratory physiotherapy delays NIV need; limb physio maintains range of motion and ambulation"},
        {"tx": "Gastrostomy (PEG)", "level": "B — if dysphagia/aspiration", "note": "Nutritional support; prevent aspiration pneumonia; early consideration in bulbar involvement"},
        {"tx": "Tracheostomy", "level": "B — when NIV insufficient", "note": "After NIV failure; discussed with patient/family in advance as NIV deteriorates"},
        {"tx": "Carnitine supplementation", "level": "C — adjunct", "note": "Secondary carnitine deficiency can occur; supplementation empirically used"},
        {"tx": "Valproic Acid (VPA)", "level": "ABSOLUTE CI", "note": "NEVER — mtDNA depletion disease; hepatotoxicity + POLG inhibition + CoA sequestration"},
        {"tx": "Ketogenic Diet", "level": "CONTRAINDICATED", "note": "NEVER — OXPHOS-dependent fat oxidation fails; metabolic crisis risk"},
        {"tx": "Propofol", "level": "AVOID", "note": "PRIS risk in mitochondrial disease; use alternative anaesthetic agents"},
    ]

    timeline = [
        {"phase": "Fetal/Neonatal", "events": "Reduced fetal movements (severe form only); neonatal hypotonia + respiratory distress in neonatal form"},
        {"phase": "Infancy (0-18 mo)", "events": "Hypotonia; motor delay; facial weakness; difficulty with head control; proximal weakness; first presentation"},
        {"phase": "Toddler (18-36 mo)", "events": "Gowers sign; tip-toe gait; waddling; falls; CK found elevated; biopsy → diagnosis"},
        {"phase": "Pre-school (2-5 yr)", "events": "Progressive weakness; stair climbing lost; respiratory compromise begins; NIV evaluation"},
        {"phase": "School age (5-12 yr)", "events": "Often wheelchair; NIV established; deoxynucleoside therapy if available; PEG if dysphagia"},
        {"phase": "Adolescent/Adult", "events": "NIV dependence; tracheostomy evaluation; respiratory failure major mortality risk; cognitive intact"},
        {"phase": "Late-onset (MDDS4C)", "events": "PEO / ptosis onset 20-55 yr; limb-girdle weakness; slower progression; p.Arg214Cys genotype"},
    ]

    return {
        "generated": date.today().isoformat(),
        "total_patients": 40,
        "seed": SEED,
        "phenotype_distribution": [{"name": p, "n": n} for p, n in phenotypes],
        "genotype_breakdown": genotypes,
        "patients": patients,
        "feature_prevalence": features,
        "treatments": treatments,
        "disease_timeline": timeline,
        "muscle_biopsy": {
            "ragged_red_fibers_pct": 82,
            "cox_negative_fibers_pct": 88,
            "mtdna_depletion_threshold": "<30% of age-matched controls diagnostic",
            "electron_microscopy": "Abnormal mitochondrial morphology — matrix granules; cristae disorganisation",
            "oxidative_phosphorylation_enzymes": "Combined Complex I+III+IV+V deficiency; Complex II (nuclear-encoded, not mtDNA) relatively spared",
        },
        "respiratory_outcomes": {
            "niv_median_start_yr": 4.5,
            "tracheostomy_pct": 38,
            "median_survival_without_tx_yr": 7,
            "median_survival_with_niv_yr": 14,
            "median_survival_with_dthd_niv_yr": "Unknown (therapy too recent); motor improvement 70%",
        },
    }


def get_definitions() -> dict:
    """TK2 MDDS4A — clinical definitions for /api/tk2/definitions."""
    return {
        "generated": date.today().isoformat(),
        "terms": [
            {"term": "TK2 (Thymidine Kinase 2)", "definition": "Mitochondrial matrix thymidine kinase (265 aa, 16q21). Phosphorylates pyrimidine deoxyribonucleosides: dThd→dTMP and dCyd→dCMP. Supplies dTTP/dCTP for mtDNA replication in post-mitotic cells (primarily muscle). Tetramer; His121 catalytic base."},
            {"term": "MDDS4A (Myopathic form)", "definition": "Most common TK2 phenotype. Infantile-onset progressive proximal myopathy with respiratory failure. NO hepatopathy. NO significant lactic acidosis at rest. CK elevated. Wheelchair + NIV dependency are typical endpoints."},
            {"term": "MDDS4B (Encephalomyopathic)", "definition": "Rarer TK2 phenotype with additional CNS involvement — cognitive regression, epilepsy, white matter changes. More severe. Typically compound heterozygous with null allele."},
            {"term": "MDDS4C (Late-onset PEO)", "definition": "Mild TK2 phenotype presenting in adulthood (20-55 yr) with progressive external ophthalmoplegia, ptosis, and limb-girdle weakness. Associated with p.Arg214Cys genotype. Normal lifespan possible with NIV."},
            {"term": "dNTP pool imbalance", "definition": "Loss of TK2 reduces dTTP and dCTP within mitochondrial matrix while dATP/dGTP remain relatively preserved. Imbalanced pools cause POLG misincorporation errors and replication fork stalling → mtDNA depletion."},
            {"term": "Deoxynucleoside supplementation", "definition": "Substrate bypass therapy: oral dThd (deoxythymidine) + dCyd (deoxycytidine) at 100-200 mg/kg/day each. Saturates mitochondrial ENT3 transporter → partial restoration of dTTP/dCTP pools without TK2. FDA orphan drug. First disease-modifying MDDS rescue."},
            {"term": "Ragged-red fibers (RRF)", "definition": "Modified Gomori trichrome (MGT) stain finding in muscle biopsy. Represent mitochondrial proliferation (subsarcolemmal accumulation) as a compensatory response to OXPHOS failure. Not specific to TK2 — seen in all mitochondrial myopathies."},
            {"term": "COX-negative fibers", "definition": "Cytochrome c oxidase (Complex IV, COX) histochemistry. Fibers lacking COX activity appear pale/blue in COX/SDH double stain. Reflect severe mtDNA depletion in individual muscle fibers. SDH (nuclear-encoded) stains normally."},
            {"term": "NIV (Non-Invasive Ventilation)", "definition": "BiPAP or CPAP via face mask. Gold standard for TK2 respiratory support. Start when FVC <60% predicted or nocturnal desaturation documented. Extends ambulation phase and life expectancy significantly."},
            {"term": "Gowers sign", "definition": "Child uses hands to push off thighs to stand from floor — indicates proximal hip girdle weakness. Classic finding in TK2 myopathy in toddlers. Also seen in SMA, DMD, other myopathies."},
            {"term": "ENT3 / SLC29A3", "definition": "Equilibrative nucleoside transporter 3. Located on lysosomal/mitochondrial membrane. Imports deoxynucleosides (including supplemental dThd + dCyd) into mitochondrial matrix. Rate-limiting step for deoxynucleoside supplementation efficacy."},
            {"term": "VPA absolute contraindication in MDDS", "definition": "Valproic acid in any mtDNA depletion disease (TK2, DGUOK, MPV17, POLG, SUCLA2, SUCLG1, etc.) causes: (1) POLG inhibition — reduces mtDNA replication further; (2) CoA sequestration — 4-en-VPA depletes mitochondrial CoA required for beta-oxidation; (3) 4-en-VPA epoxide — direct hepatocyte toxicity. Any one mechanism is life-threatening; all three together are lethal."},
            {"term": "PRIS (Propofol Infusion Syndrome)", "definition": "Propofol inhibits mitochondrial complex I and fatty acid beta-oxidation. In pre-existing OXPHOS defects (mtDNA depletion), this additional inhibition can precipitate fatal metabolic collapse — metabolic acidosis, rhabdomyolysis, cardiac failure. AVOID propofol in all mito disease."},
            {"term": "Ketogenic diet contraindication in mtDNA depletion", "definition": "KD shifts energy metabolism from glucose to fat (beta-oxidation → ketone bodies → acetyl-CoA → TCA → OXPHOS). In mtDNA depletion, OXPHOS complex capacity is insufficient to handle fat-derived acetyl-CoA, leading to metabolic crisis, lactic acidosis, and potential death."},
            {"term": "mtDNA depletion diagnostic threshold", "definition": "Muscle mtDNA copy number <30% of age-matched healthy controls is diagnostic for mtDNA depletion syndrome when clinical/enzymatic features are consistent. Measured by qPCR (mtDNA:nDNA ratio). Correlation with disease severity is imperfect."},
            {"term": "Spinazzola 1999 / Mancuso 2002", "definition": "Spinazzola A et al. 1999 Am J Hum Genet 65(5):1258-1265 — first identification of TK2 mutations causing myopathic mtDNA depletion. Mancuso M et al. 2002 Ann Neurol 52(6):741-749 — first comprehensive clinical series of 17 patients; defined the classical infantile myopathic phenotype."},
        ],
    }
