#!/usr/bin/env python3
"""KSS — Kearns-Sayre Syndrome (mtDNA Single Large Deletion).

Primary mtDNA structural disease — sporadic (de novo deletion):
  Kearns-Sayre Syndrome  OMIM #530000
  Related: Pearson Marrow-Pancreas Syndrome OMIM #557000
  Cause: Large single mtDNA deletion (1.1–10 kb); most common = 4977 bp "common deletion"
         (m.8483_13459del / ΔmtDNA4977); eliminates multiple tRNA genes and structural
         subunit genes → multi-complex OXPHOS failure

PATHOPHYSIOLOGY (large single deletion / multi-tRNA / multi-complex OXPHOS):
The 4977 bp common deletion removes 13 genes between two 13-bp direct repeats (ACCTCCCTCACCA):
  • tRNA-Lys (MT-TK)  • tRNA-Gly (MT-TG)  • tRNA-Arg (MT-TR)  • tRNA-Ser (MT-TS2)
  • tRNA-His (MT-TH)  • tRNA-Leu2 (MT-TL2) (partial)
  • ND3 (MT-ND3)  • ND4L (MT-ND4L)  • ND4 (partial) — Complex I
  • ATP8, ATP6 (partial) — Complex V
  • CO3 (MT-CO3) — Complex IV

Loss of these tRNAs means NO translation of any mtDNA-encoded protein using those codons,
affecting all OXPHOS complexes simultaneously. Deletion heteroplasmy >60% in muscle causes
symptomatic multi-complex OXPHOS failure; blood heteroplasmy is typically LOWER than muscle
(segregation advantage of non-deleted mtDNA in rapidly dividing haematopoietic cells).

MOLECULAR MECHANISM — Direct Repeat Slipped-Strand Mispairing:
The deletion arises by slipped-strand mispairing during mtDNA replication between the two
13-bp direct repeats flanking the 4977-bp segment (Schon 1989 Science). This is the molecular
'hotspot' because the repeats promote replication fork slippage. The deletion is:
  — de novo (sporadic) in >95% of KSS cases
  — Rarely maternally transmitted (very low blood heteroplasmy in mothers; germline bottleneck
    favors non-deleted mtDNA; maternal transmission of KSS deletion is exceptional)
  — Tissue-specific: muscle/brain accumulate deletion (post-mitotic); blood/liver less affected

ALLELIC CONDITIONS (same deletion, different tissue distribution/age):
  1. KSS: PEO + RP + CHB + onset <20y (multi-tissue, high heteroplasmy, systemic)
  2. Pearson Syndrome (PS): Infant onset bone marrow failure + exocrine pancreatic failure
     (haematopoietic stem cells affected; if child survives → KSS develops in adolescence
     as post-mitotic cells accumulate deletion; Pearson→KSS transition ~50% of PS survivors)
  3. CPEO (Chronic Progressive External Ophthalmoplegia) with single deletion: PEO only, no
     systemic features; adult onset; milder; same 4977 deletion or other single deletion;
     NO heart block (distinguishes from KSS)
  4. Maternally inherited PEO / Leigh: rare maternal transmission of deletion (not KSS by triad)

KSS CLINICAL PRESENTATION — DIAGNOSTIC TRIAD (ALL THREE REQUIRED):
  1. Progressive External Ophthalmoplegia (PEO) — 100%: bilateral ptosis + ophthalmoplegia;
     CARDINAL; onset often first presenting sign (ptosis in childhood/adolescence); symmetric;
     NOT painful; orbicularis weakness may coexist; Bell phenomenon MANDATORY before ptosis surgery
  2. Pigmentary Retinopathy — 100%: "salt and pepper" pattern (macular > peripheral);
     CARDINAL; ERG extinguished; NOT same as classic retinitis pigmentosa (rod-cone dystrophy);
     visual symptoms variable (peripheral vision preserved longer than LHON); fundoscopy diagnostic
  3. Onset before age 20 — 100%: CARDINAL CRITERION; onset ≥20 → diagnose CPEO, not KSS

ADDITIONAL FEATURES:
  4. Complete Heart Block (CHB) — 57%: PR prolongation → 2nd/3rd degree AV block → sudden death;
     PACEMAKER IS LIFE-SAVING; monitor ECG/Holter annually; any PR >200ms = refer cardiology NOW
  5. Cerebellar Ataxia — 55%: gait ataxia, limb dysmetria, dysarthria; correlates with
     cerebellar+brainstem mtDNA deletion burden; cerebellar atrophy on MRI
  6. CSF Protein Elevation — 50% above 100 mg/dL; 93% have abnormal CSF protein;
     pathognomonic when ≥100 mg/dL + triad; reflects mitochondrial failure of oligodendrocytes
     and myelin maintenance; CSF folate is also depleted (5-MTHF) → leucovorin supplementation
  7. Sensorineural Hearing Loss (SNHL) — 60%: bilateral high-frequency; cochlear OXPHOS failure;
     cochlear implant effective; AMINOGLYCOSIDES ABSOLUTE CI (amplify cochlear mtDNA dysfunction)
  8. Endocrine:
     a. Diabetes mellitus — 13% (insulin-dependent; NOT metformin — Complex I inhibition CI)
     b. Hypoparathyroidism — 20%: hypocalcaemia, tetany, Chvostek/Trousseau; replace Ca2+ + calcitriol
     c. Growth hormone deficiency / short stature — 38%; GH replacement considered
     d. Hypomagnesaemia (renal wasting) — 15%; supplement magnesium
  9. Proximal Myopathy — 55%: exercise intolerance; ragged red fibers (RRF) on Gomori trichrome;
     COX-negative fibers on histochemistry; SDH-positive (succinate dehydrogenase — Complex II,
     nuclear-encoded, unaffected by mtDNA deletion); PATHOGNOMONIC RRF + SDH-positive + COX-negative
  10. Renal Tubulopathy (Fanconi) — 20%: proximal tubular dysfunction; glucosuria, aminoaciduria,
      phosphaturia; monitor renal function; phosphate replacement if hypophosphataemia
  11. Dementia / Cognitive Decline — 25%: subcortical > cortical; leukoencephalopathy pattern;
      bilateral symmetric T2 signal in subcortical white matter (DIFFERENT from Leigh BG/BS)
  12. Cerebellar-Brainstem MRI — 80%: cerebellar+brainstem T2 signal; NOT bilateral BG (→Leigh);
      subcortical WM lesions (→leukoencephalopathy-like); basal ganglia SPARED (DDx Leigh)

HISTOPATHOLOGY (Muscle Biopsy — DIAGNOSTIC):
  • Ragged Red Fibers (RRF): Modified Gomori Trichrome — mitochondrial accumulation at fibre edge
  • COX-negative fibers: cytochrome c oxidase (Complex IV) deficiency — blue Gomori but no brown COX
  • SDH-positive ("ragged blue"): nuclear-encoded Complex II intact; confirms mitochondrial origin
  • Electron microscopy: abnormal mitochondrial cristae; paracrystalline inclusions
  • Long-range PCR: single large deletion confirmed (NOT multiple deletions — DDx POLG/TWNK)
  • Southern blot: stoichiometric deletion band (single species, NOT heterogeneous smear)

KEY DIFFERENTIAL DIAGNOSIS:
  1. POLG1 Alpers / adPEO (POLG, TWNK, DNA2): MULTIPLE deletions (not single); hepatopathy in POLG;
     southern blot shows heterogeneous ladder NOT single band; longer-range PCR for distinction
  2. Leigh Syndrome (Complex I / Leigh-MT): Bilateral BG + brainstem T2 (NOT cerebellar-dominant);
     onset younger; metabolic crisis; different mutations (ND3, ND5, SURF1, SCO2)
  3. MELAS (MT-TL1): SLE (not PEO-dominant); lactic acidosis crisis; no heart block; m.3243A>G
  4. CPEO (nuclear or single deletion): NO RP; NO CHB; NO onset-<20 criterion met if adult; no
     systemic features; may have same 4977 deletion but distribution confined to muscle/eye
  5. Mitofusin-2 / OPA1-Plus: PEO without RP or CHB; multiple deletions; nuclear gene; AD
  6. MERRF (MT-TK): Myoclonic epilepsy; RRF; m.8344A>G; no RP; no CHB; cerebellar ataxia similar
  7. Chronic progressive ophthalmoplegia + myasthenia: tensilon test positive in MG; no RRF; no RP

ABSOLUTE DRUG CONTRAINDICATIONS:
  VPA (valproate) — ABSOLUTE CI ALL KSS:
    (a) CoA sequestration → impairs TCA cycle flux already stressed by multi-complex OXPHOS failure
    (b) POLG1 inhibition → accelerates mtDNA deletion expansion / depletion
    (c) Hepatotoxicity in mitochondrial disease (risk of acute liver failure)
    Alternative: LEV (renal excretion, no mito toxicity), CLB, perampanel if seizures
  Aminoglycosides — ABSOLUTE CI ALL KSS:
    Cochlear OXPHOS (Complex I→IV) already impaired by deletion; aminoglycosides inhibit
    mitoribosome (12S rRNA) → additive and irreversible cochlear hair cell death → sudden
    profound permanent deafness. NEVER use in any confirmed or suspected KSS patient.
    Alternative: beta-lactams, cephalosporins, carbapenems, daptomycin for serious infections
  Linezolid — ABSOLUTE CI:
    Inhibits mitochondrial 23S rRNA → blocks mtDNA-encoded protein synthesis → pan-OXPHOS depletion
    Causes DION, optic neuropathy, lactic acidosis; additive with existing deletion-mediated OXPHOS failure
  Metformin — ABSOLUTE CI:
    Complex I inhibition → worsens existing Complex I failure from deletion → fatal lactic acidosis
    Substitute: insulin, DPP-4 inhibitor, SGLT-2 inhibitor (renal-safe), sulfonylurea (NOT DM monotherapy)
  Ketogenic Diet — ABSOLUTE CI:
    Forces pure fatty acid beta-oxidation → requires intact OXPHOS (FAO generates FADH2→Complex II,
    NADH→Complex I, but also requires Complex IV/V for re-oxidation); deletion-impaired OXPHOS cannot
    sustain this → metabolic collapse + lactic acidosis

DRUGS TO AVOID:
  Propofol — AVOID (PRIS): propofol infusion syndrome; impaired fatty acid oxidation; avoid for anaesthesia
  Phenobarbital — AVOID (Complex I inhibition)
  Glucocorticoids — CAUTION (hyperglycemia in DM subgroup; caution but not absolute CI)
  NSAIDs — CAUTION (nephrotoxic in renal tubulopathy subset; monitor eGFR)

CRITICAL MONITORING (CARDIAC — life-threatening):
  • Baseline: 12-lead ECG + echocardiogram at DIAGNOSIS
  • Annual: 12-lead ECG + 24h Holter monitor
  • ANY PR >200ms, 2nd-degree block → URGENT cardiology referral + pacemaker evaluation
  • 3rd-degree (complete) heart block → EMERGENCY pacemaker insertion (risk of sudden death)
  • Pacemaker type: dual-chamber (DDD) to preserve AV synchrony; ICD not routinely required

TREATMENT (Level of Evidence):
  • Pacemaker: MANDATORY for CHB — Level A (life-saving intervention)
  • Leucovorin (folinic acid, 5-formyl-THF): Level C — replenishes CSF folate;
    CSF 5-MTHF is depleted in KSS (possibly due to impaired folate transport at CP);
    some case reports show functional improvement; dose 2.5–5 mg/day oral
  • CoQ10 / Ubiquinol: Level C — 300–1200 mg/day; improves mitochondrial electron transport
    (bypass Complex I/III blockade by shuttling electrons); ubiquinol preferred (better bioavailability)
  • Riboflavin B2: Level C — Cofactor for Complex I/II/electron transfer flavoprotein
  • Thiamine B1: Level C — PDH/KGDH cofactor; TCA cycle support
  • L-Carnitine: Level C — support FAO; secondary carnitine deficiency reported in some
  • Ptosis surgery (external levator resection): Level B — requires Bell phenomenon test pre-op
    (if absent, exposes cornea → keratopathy risk; frontalis sling preferred if Bell phenomenon weak)
  • Endocrine replacement:
    - DM: insulin + DPP-4 inhibitor (NEVER metformin)
    - Hypoparathyroidism: calcium carbonate + calcitriol (1,25-OH2-D3)
    - GH deficiency: GH replacement if severe (specialist management)
  • Annual ophthalmology: slit-lamp + fundoscopy + visual field (monitor RP progression)
  • Annual audiology: pure tone audiometry; cochlear implant referral if severe SNHL
  • Annual renal: eGFR, urinalysis for glycosuria/proteinuria, phosphate, magnesium
  • Avoid prolonged fasting (hospital NBM: IV dextrose GIR 6–8 mg/kg/min)
  • Genetic counselling: low maternal recurrence risk (<4%); recommend maternal deletion testing;
    siblings: 1–2% recurrence risk (extremely low vs dominant nuclear gene disorders)

KEY REFERENCES:
  Kearns-Sayre TP, Sayre GP. 1958. Two cases of 'sporadic' progressive external ophthalmoplegia.
  Trans Am Ophthalmol Soc. First clinical description.
  Holt IJ, Harding AE, Morgan-Hughes JA. 1988. Deletions of muscle mitochondrial DNA in patients
  with mitochondrial myopathies. Nature 331:717–719. First pathogenic single mtDNA deletion.
  Zeviani M, et al. 1988. Deletions of mitochondrial DNA in Kearns-Sayre syndrome. Neurology 38:1339.
  Schon EA, et al. 1989. A direct repeat is a hotspot for large-scale deletion of human
  mitochondrial DNA. Science 244:346–349. Common deletion molecular mechanism.
  Pearson HA, et al. 1979. A new syndrome of refractory sideroblastic anemia. J Pediatr 95:976.
  Lestienne P, Ponsot G. 1988. Kearns-Sayre syndrome with muscle mitochondrial DNA deletion.
  Lancet 1:885. Early clinical-molecular correlation.
"""

from __future__ import annotations
import random
from typing import Any

# ── Disease constants ────────────────────────────────────────────────────────
SEED        = 589
DISEASE_ID  = "kss"
DISEASE_NAME = "Kearns-Sayre Syndrome"
GENE        = "mtDNA single large deletion"
OMIM_GENE   = "*mitochondrial chromosome"
OMIM_DISEASE = "#530000"
CHROMOSOME  = "mtDNA (mitochondrial chromosome)"
INHERITANCE = "Sporadic (de novo deletion); maternal transmission <4% (rare)"
ONSET       = "Before age 20 (CARDINAL criterion; onset ≥20 → CPEO, not KSS)"
COHORT_SIZE = 40
COLOR       = "#1565c0"   # deep blue — KSS/mtDNA deletion/cardiac-ophthalmic

# Deletion types
DEL_COMMON   = "common 4977 bp (m.8483_13459del)"
DEL_LARGE    = "large deletion >5 kb (non-standard)"
DEL_SMALL    = "small deletion 1.1–3 kb (atypical)"

DELETION_POOL    = [DEL_COMMON, DEL_LARGE, DEL_SMALL]
DELETION_WEIGHTS = [0.65,        0.20,       0.15]

# ── Seeded RNG ───────────────────────────────────────────────────────────────
def _rng() -> random.Random:
    """Seeded RNG for reproducible 40-patient KSS cohort (seed-589)."""
    return random.Random(SEED)


# ── Cohort generation ────────────────────────────────────────────────────────
def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient single-deletion KSS cohort (seed-589).

    Heteroplasmy = muscle heteroplasmy (blood typically lower).
    All patients meet KSS triad: PEO + RP + onset <20y.
    """
    patients = []
    for i in range(1, COHORT_SIZE + 1):
        # Deletion type
        deletion = rng.choices(DELETION_POOL, weights=DELETION_WEIGHTS)[0]
        # Muscle heteroplasmy (60–98%; blood ~20–60% lower)
        het_muscle = round(rng.uniform(60, 98), 1)
        # Sex: equal (sporadic, no sex bias)
        sex = "F" if rng.random() < 0.50 else "M"
        # Age at diagnosis: <20 by definition (mean 12, SD 5)
        age_dx = max(3, round(rng.gauss(12, 5)))
        # Cap at 19 (CARDINAL criterion)
        age_dx = min(age_dx, 19)

        # Clinical features (heteroplasmy-weighted)
        het_frac = het_muscle / 100.0
        chb          = rng.random() < 0.57           # Complete Heart Block 57%
        pacemaker    = chb and rng.random() < 0.90   # 90% of CHB get pacemaker
        ataxia       = rng.random() < 0.55           # Cerebellar ataxia 55%
        csf_high     = rng.random() < 0.50           # CSF protein >100 mg/dL 50%
        snhl         = rng.random() < 0.60           # SNHL 60%
        diabetes     = rng.random() < 0.13           # DM 13%
        hypopara     = rng.random() < 0.20           # Hypoparathyroidism 20%
        short_stat   = rng.random() < 0.38           # Short stature / GHD 38%
        myopathy     = rng.random() < 0.55           # Proximal myopathy 55%
        renal_fanconi= rng.random() < 0.20           # Fanconi tubular 20%
        dementia     = rng.random() < 0.25           # Cognitive decline 25%
        seizures     = rng.random() < 0.10           # Seizures uncommon 10%

        # Treatment
        txs = ["CoQ10/ubiquinol", "Leucovorin", "Riboflavin B2"]
        if pacemaker:    txs.append("Pacemaker (DDD)")
        if diabetes:     txs.append("Insulin (NOT metformin)")
        if hypopara:     txs.append("Ca2+ + calcitriol")
        if snhl:         txs.append("Audiology/CI referral")
        if seizures:     txs.append("LEV (preferred AED)")
        if short_stat:   txs.append("GH evaluation")

        # Drug safety alerts
        alerts_list = []
        if diabetes: alerts_list.append("⚠ DM: metformin ABSOLUTE CI")
        if snhl:     alerts_list.append("⚠ SNHL: aminoglycosides ABSOLUTE CI")
        if seizures: alerts_list.append("⚠ Sz: VPA ABSOLUTE CI")
        if chb and not pacemaker: alerts_list.append("🚨 CHB: pacemaker NOT yet inserted")
        alerts = "; ".join(alerts_list) if alerts_list else "None"

        # Feature string
        feats = ["PEO", "Pigmentary RP"]
        if chb:          feats.append("CHB")
        if ataxia:       feats.append("Ataxia")
        if csf_high:     feats.append("CSF↑")
        if snhl:         feats.append("SNHL")
        if myopathy:     feats.append("Myopathy/RRF")
        if diabetes:     feats.append("DM")
        if hypopara:     feats.append("HypoPTH")
        if short_stat:   feats.append("Short stat")
        if renal_fanconi: feats.append("Fanconi")
        if dementia:     feats.append("Dementia")
        if seizures:     feats.append("Seizures")

        patients.append({
            "id":        f"KSS-{i:03d}",
            "deletion":  deletion,
            "het":       het_muscle,
            "sex":       sex,
            "age_dx":    age_dx,
            "chb":       chb,
            "pacemaker": pacemaker,
            "features":  ", ".join(feats),
            "treatments": ", ".join(txs),
            "alerts":    alerts,
        })
    return patients


# ── Public API functions ─────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    """KSS overview — gene, disease identity, KPIs, contraindications."""
    rng = _rng()
    cohort = _build_cohort(rng)

    n = len(cohort)
    n_chb       = sum(1 for p in cohort if p["chb"])
    n_pacemaker = sum(1 for p in cohort if p["pacemaker"])
    n_ataxia    = sum(1 for p in cohort if "Ataxia" in p["features"])
    n_snhl      = sum(1 for p in cohort if "SNHL" in p["features"])
    n_dm        = sum(1 for p in cohort if "DM" in p["features"])
    n_myopathy  = sum(1 for p in cohort if "Myopathy" in p["features"])
    n_hypopara  = sum(1 for p in cohort if "HypoPTH" in p["features"])
    mean_age_dx = round(sum(p["age_dx"] for p in cohort) / n, 1)
    mean_het    = round(sum(p["het"] for p in cohort) / n, 1)
    n_common    = sum(1 for p in cohort if "4977" in p["deletion"])
    pct_common  = round(n_common / n * 100)

    return {
        "gene":         "mtDNA single large deletion",
        "protein":      "Multiple OXPHOS subunits lost (ND3, ND4L, CO3, ATP6/8 + tRNAs)",
        "disease":      "Kearns-Sayre Syndrome (KSS)",
        "omim_gene":    "Mitochondrial chromosome — no single OMIM gene entry",
        "omim_disease": "#530000",
        "chromosome":   "mtDNA (16,569 bp circular mitochondrial chromosome)",
        "inheritance":  "Sporadic de novo; maternal recurrence <4%",
        "onset":        "Before age 20 (CARDINAL — onset ≥20 → CPEO, not KSS)",
        "cohort":       f"{n} patients · seed-589 · KSS single mtDNA deletion",
        "mechanism": (
            "Large mtDNA deletion (1.1–10 kb) removes multiple tRNA genes + structural "
            "subunit genes (ND3, ND4L, partial ND4/ND5, CO3, ATP8/6). Loss of tRNA-Lys, "
            "tRNA-Gly, tRNA-Arg, tRNA-Ser, tRNA-His impairs translation of ALL OXPHOS "
            "complexes using those codons → multi-complex I+IV+V failure proportional "
            "to deletion heteroplasmy in post-mitotic cells (muscle, brain, heart conduction "
            "system, photoreceptors). Blood heteroplasmy UNDERESTIMATES muscle burden "
            "due to selective replication of non-deleted mtDNA in dividing haematopoietic cells."
        ),
        "mtdna_pattern": (
            f"Single large deletion — {pct_common}% cohort carry 4977 bp common deletion "
            "(m.8483_13459del; between 13-bp direct repeats ACCTCCCTCACCA). Remaining cases: "
            "other large (>5 kb) or atypical small (1.1–3 kb) single deletions. "
            "Long-range PCR shows ONE deletion band (not multiple ladder → DDx POLG/TWNK). "
            "Southern blot: single stoichiometric band. Tissue distribution: muscle/brain "
            ">>blood (blood underestimates by 20–40 percentage points). "
            "Mutation is SPORADIC (de novo during oogenesis/early embryogenesis); "
            "maternal recurrence risk <4%; sibling recurrence ~1–2%."
        ),
        "kpis": [
            {"label": "PEO + RP", "value": "100%", "color": COLOR},
            {"label": "Onset <20y", "value": "100%", "color": COLOR},
            {"label": "CHB (heart block)", "value": f"{n_chb/n*100:.0f}%", "color": "#c62828"},
            {"label": "Pacemakers", "value": str(n_pacemaker), "color": "#1b5e20"},
            {"label": "Cerebellar Ataxia", "value": f"{n_ataxia/n*100:.0f}%", "color": COLOR},
            {"label": "SNHL", "value": f"{n_snhl/n*100:.0f}%", "color": COLOR},
            {"label": "Myopathy/RRF", "value": f"{n_myopathy/n*100:.0f}%", "color": COLOR},
            {"label": "Diabetes", "value": f"{n_dm/n*100:.0f}%", "color": "#e65100"},
            {"label": "Hypoparathyroid", "value": f"{n_hypopara/n*100:.0f}%", "color": "#6a1b9a"},
            {"label": "Mean Δ-het (muscle)", "value": f"{mean_het}%", "color": COLOR},
            {"label": "Common deletion", "value": f"{pct_common}%", "color": COLOR},
            {"label": "Mean Age Dx", "value": f"{mean_age_dx}y", "color": "#37474f"},
        ],
        "contraindications": [
            {
                "drug":      "VPA / Valproate",
                "severity":  "ABSOLUTE CI — ALL KSS",
                "mechanism": "CoA sequestration + POLG1 inhibition (accelerates deletion expansion) + hepatotoxicity; "
                             "worsens existing multi-complex OXPHOS failure; use LEV/CLB/perampanel instead",
            },
            {
                "drug":      "Aminoglycosides (gentamicin, tobramycin, amikacin)",
                "severity":  "ABSOLUTE CI — ALL KSS (SNHL amplification)",
                "mechanism": "Inhibit mitoribosome 12S rRNA → additive cochlear OXPHOS failure → sudden irreversible "
                             "profound deafness; NEVER use in confirmed or suspected KSS; use cephalosporins/carbapenems",
            },
            {
                "drug":      "Linezolid",
                "severity":  "ABSOLUTE CI — ALL mitochondrial disease",
                "mechanism": "Inhibits mitochondrial 23S rRNA → blocks mtDNA-encoded protein synthesis → "
                             "pan-OXPHOS depletion; causes DION, lactic acidosis; alternative: daptomycin/tigecycline",
            },
            {
                "drug":      "Metformin",
                "severity":  "ABSOLUTE CI — DM in KSS",
                "mechanism": "Complex I inhibition → worsens existing Complex I failure from deletion → "
                             "fatal lactic acidosis; use insulin + DPP-4 inhibitor or SGLT-2 inhibitor instead",
            },
            {
                "drug":      "Ketogenic Diet",
                "severity":  "ABSOLUTE CI — ALL KSS",
                "mechanism": "Forces OXPHOS-dependent beta-oxidation; deletion-impaired OXPHOS cannot sustain "
                             "FAO flux → metabolic collapse; lactic acidosis; NO therapeutic role",
            },
            {
                "drug":      "Propofol (infusion)",
                "severity":  "AVOID — PRIS risk",
                "mechanism": "Propofol infusion syndrome; impaired FAO in mitochondrial disease; "
                             "use for brief induction only if necessary; avoid infusion >1 mg/kg/h",
            },
        ],
    }


def get_breakdown() -> dict[str, Any]:
    """KSS patient cohort table + clinical feature frequencies."""
    rng = _rng()
    cohort = _build_cohort(rng)

    n = len(cohort)

    def pct(feat: str) -> int:
        return round(sum(1 for p in cohort if feat in p["features"]) / n * 100)

    feature_frequencies = {
        "PEO (ptosis + ophthalmoplegia)": 100,
        "Pigmentary Retinopathy (salt+pepper)": 100,
        "Onset < 20 years": 100,
        "Complete Heart Block": pct("CHB"),
        "Pacemaker Inserted": round(sum(1 for p in cohort if p["pacemaker"]) / n * 100),
        "Cerebellar Ataxia": pct("Ataxia"),
        "CSF Protein >100 mg/dL": pct("CSF↑"),
        "Sensorineural Hearing Loss": pct("SNHL"),
        "Proximal Myopathy / RRF": pct("Myopathy"),
        "Diabetes Mellitus": pct("DM"),
        "Hypoparathyroidism": pct("HypoPTH"),
        "Short Stature / GHD": pct("Short stat"),
        "Renal Fanconi Tubular": pct("Fanconi"),
        "Dementia / Cognitive Decline": pct("Dementia"),
        "Seizures (uncommon)": pct("Seizures"),
    }

    return {
        "patients": cohort,
        "feature_frequencies": feature_frequencies,
    }


def get_definitions() -> dict[str, Any]:
    """Extended definitions: pharmacology, molecular concepts, disease concepts, safety."""
    return {
        "pharmacology": [
            {
                "term": "VPA Absolute Contraindication — KSS (Multi-Mechanism)",
                "definition": (
                    "Valproate is ABSOLUTELY CONTRAINDICATED in ALL KSS patients regardless of seizure "
                    "severity. Three independent mechanisms amplify toxicity:\n"
                    "1. CoA SEQUESTRATION: valproate conjugated to valproyl-CoA, depleting free CoA "
                    "required as acetyl-CoA for TCA cycle entry — already impaired by multi-complex OXPHOS "
                    "failure — worsening lactic acidosis.\n"
                    "2. POLG1 INHIBITION: valproate inhibits mitochondrial DNA polymerase gamma → "
                    "accelerates expansion of the existing deletion → mtDNA copy number depletion.\n"
                    "3. DIRECT HEPATOTOXICITY: mitochondrial fatty acid oxidation failure + TCA block "
                    "→ microvesicular steatosis → acute liver failure (Reye-like syndrome).\n"
                    "Alternative AEDs: LEV (levetiracetam) — first-line (renal excretion, no mito "
                    "toxicity); CLB (clobazam) — safe adjunct; perampanel — reasonable option.\n"
                    "Seizures in KSS are UNCOMMON (~10%); risk-benefit strongly favours avoidance."
                ),
            },
            {
                "term": "Aminoglycoside Absolute Contraindication — KSS + SNHL",
                "definition": (
                    "Aminoglycoside antibiotics (gentamicin, tobramycin, amikacin, streptomycin, "
                    "kanamycin, neomycin) are ABSOLUTELY CONTRAINDICATED in ALL KSS patients, "
                    "whether or not SNHL is already present.\n"
                    "MECHANISM: Aminoglycosides bind mitochondrial 12S rRNA (MT-RNR1) → inhibit "
                    "mitoribosome → cochlear hair cell OXPHOS already impaired by mtDNA deletion → "
                    "additive energy failure → immediate, irreversible, profound bilateral hearing loss.\n"
                    "Even a single dose can cause sudden complete deafness in a patient with established "
                    "cochlear OXPHOS compromise. The existing deletion already depletes cochlear OXPHOS; "
                    "aminoglycosides remove the residual margin.\n"
                    "SAFE ALTERNATIVES:\n"
                    "  Gram-negative severe: cefepime, piperacillin-tazobactam, meropenem\n"
                    "  MRSA: daptomycin (NOT linezolid — also CI), vancomycin (monitor trough)\n"
                    "  Synergistic coverage: aztreonam + beta-lactam (not aminoglycoside)\n"
                    "Document allergy in chart as 'mitochondrial disease — aminoglycosides ABSOLUTE CI'."
                ),
            },
            {
                "term": "Metformin Contraindication — DM in KSS",
                "definition": (
                    "Metformin is ABSOLUTELY CONTRAINDICATED in KSS-associated diabetes mellitus.\n"
                    "MECHANISM: Metformin inhibits Complex I (mitochondrial respiratory chain) → "
                    "worsens existing Complex I failure due to ND3/ND4L deletion → impairs hepatic "
                    "lactate gluconeogenesis (already stressed) → fatal lactic acidosis.\n"
                    "KSS patients cannot buffer the lactic acid because oxidative phosphorylation "
                    "is already critically impaired. Even low-dose metformin carries unacceptable risk.\n"
                    "SAFE DM TREATMENT IN KSS:\n"
                    "  First-line: Insulin (basal-bolus)\n"
                    "  Adjuncts: DPP-4 inhibitors (sitagliptin, saxagliptin — renal safe; no mito effect)\n"
                    "           SGLT-2 inhibitors (dapagliflozin) — monitor renal function (Fanconi risk)\n"
                    "           Sulfonylureas — second-line (hypoglycaemia risk)\n"
                    "  NEVER: Metformin, phenformin, or any biguanide."
                ),
            },
            {
                "term": "Leucovorin (Folinic Acid) — CSF Folate Deficiency in KSS",
                "definition": (
                    "CSF 5-methyltetrahydrofolate (5-MTHF) is depleted in KSS — possibly due to impaired "
                    "active folate transport at the choroid plexus (OXPHOS-dependent energy-consuming process).\n"
                    "EVIDENCE: Level C — case series show functional improvement with leucovorin supplementation; "
                    "one open-label study (Pineda 2006, NeuropediatricsDOI) documented clinical and biochemical "
                    "improvement. Not randomised controlled.\n"
                    "DOSE: 2.5–5 mg/day oral leucovorin (folinic acid, 5-formyl-THF); "
                    "avoid plain folic acid (poor CNS penetration; different form).\n"
                    "CSF FOLATE MEASUREMENT: Low (<40 nmol/L) in most KSS patients with CSF protein elevation; "
                    "measure before supplementation to confirm deficiency.\n"
                    "MECHANISM OF BENEFIT: Leucovorin bypasses impaired reduction to 5-MTHF; crosses BBB; "
                    "replenishes methyl groups for myelin synthesis + one-carbon metabolism."
                ),
            },
            {
                "term": "Pacemaker — Mandatory Life-Saving Intervention (KSS CHB)",
                "definition": (
                    "Complete Heart Block (CHB) in KSS is caused by mitochondrial failure of "
                    "the atrioventricular (AV) conduction system (AV node, bundle of His): "
                    "deletion-mediated OXPHOS failure causes progressive AV nodal fibrosis.\n"
                    "RISK: Unpredictable sudden cardiac death from complete AV block → ventricular "
                    "standstill. KSS-related CHB can progress rapidly from 1st-degree to complete "
                    "without warning symptoms.\n"
                    "MONITORING PROTOCOL:\n"
                    "  Baseline: 12-lead ECG + echocardiogram at diagnosis\n"
                    "  Annual: 12-lead ECG + 24h Holter\n"
                    "  Any PR >200ms: urgent cardiology + electrophysiology referral\n"
                    "  2nd-degree (Mobitz I/II): elective pacemaker planning\n"
                    "  3rd-degree (complete): EMERGENCY pacemaker insertion (same day)\n"
                    "PACEMAKER TYPE: Dual-chamber DDD pacemaker to preserve AV synchrony; "
                    "ICD not routinely indicated (CHB → asystole, not VF); "
                    "epicardial lead option if transthoracic access difficult.\n"
                    "NOTE: Even asymptomatic CHB carries sudden death risk; pacemaker is NOT optional."
                ),
            },
            {
                "term": "CoQ10 / Ubiquinol Supplementation — Level C (KSS)",
                "definition": (
                    "CoQ10 (Coenzyme Q10, ubiquinone/ubiquinol) supplementation is Level C evidence "
                    "(case series + open-label; no RCT in KSS specifically).\n"
                    "MECHANISM: CoQ10 shuttles electrons from Complex I and II to Complex III. In KSS, "
                    "the deletion impairs Complex I and IV but Complex II is nuclear-encoded and intact; "
                    "CoQ10 supplementation may partially bypass Complex I deficiency by maintaining "
                    "the I→CoQ10→III→cytochrome c→IV electron flow.\n"
                    "DOSE: 300–1200 mg/day divided doses (typical: 300 mg TID); "
                    "ubiquinol (reduced form) preferred for bioavailability.\n"
                    "MONITORING: Plasma CoQ10 level (target >2.5 μg/mL); "
                    "lipid-soluble: give with meals to improve absorption.\n"
                    "COMBINED THERAPY: Often combined with riboflavin B2 + thiamine B1 + leucovorin "
                    "as the 'mitochondrial cocktail'; no single supplement proven alone."
                ),
            },
        ],
        "gene_concepts": [
            {
                "term": "mtDNA Single Large Deletion — 4977 bp Common Deletion (m.8483_13459del)",
                "definition": (
                    "The 4977 bp 'common deletion' is the most frequent pathogenic single mtDNA deletion "
                    "(~30–40% of all single-deletion KSS/CPEO cases).\n"
                    "MOLECULAR STRUCTURE:\n"
                    "  Deletion endpoints: m.8483 to m.13459 (Cambridge reference sequence)\n"
                    "  Flanking direct repeats: 13 bp (ACCTCCCTCACCA) at m.8470–8482 and m.13446–13459\n"
                    "  Deleted segment: 4977 bp including tRNA-Lys (MT-TK), tRNA-Gly (MT-TG), "
                    "  tRNA-Arg (MT-TR), tRNA-Ser2 (MT-TS2), tRNA-His (MT-TH), partial tRNA-Leu2 "
                    "  (MT-TL2), ND3 (MT-ND3), ND4L (MT-ND4L), partial ND4 (MT-ND4), "
                    "  ATP8 (MT-ATP8), partial ATP6 (MT-ATP6), CO3 (MT-CO3)\n"
                    "MECHANISM OF DELETION (Schon 1989):\n"
                    "  Slipped-strand mispairing during mtDNA replication at direct repeats → "
                    "  replication fork slippage → premature re-annealing → deletion of segment "
                    "  between repeats. This is the 'hotspot' mechanism explaining >65% of KSS.\n"
                    "WHY POST-MITOTIC CELLS ACCUMULATE DELETION:\n"
                    "  Deleted mtDNA has replicative advantage (smaller genome → faster replication) "
                    "  in post-mitotic cells; dividing cells (bone marrow) eliminate deleted mtDNA "
                    "  by selective pressure → blood het LOWER than muscle by 20–40 percentage points."
                ),
            },
            {
                "term": "Multi-Complex OXPHOS Failure — KSS Biochemistry",
                "definition": (
                    "Loss of tRNA-Lys, tRNA-Gly, tRNA-Arg impairs translation of ALL 13 mtDNA-encoded "
                    "OXPHOS subunits that depend on these codons (Lys=AAA/AAG, Gly=GGA/GGG/GGC/GGU, "
                    "Arg=CGG/CGC/CGA/CGU/AGA/AGG):\n"
                    "  Complex I: ND1–ND6, ND4L (7 subunits; 6 use deleted tRNAs)\n"
                    "  Complex III: Cytochrome b (CYB) — uses tRNA-Gly\n"
                    "  Complex IV: CO1, CO2, CO3 (3 subunits; CO3 deleted directly)\n"
                    "  Complex V: ATP8, ATP6 (ATP8 deleted; ATP6 partially deleted)\n"
                    "  Complex II: SDHA-D — ALL nuclear-encoded → INTACT (SDH histochemistry: POSITIVE)\n"
                    "DIAGNOSTIC CONSEQUENCE:\n"
                    "  Muscle biopsy: SDH-positive + COX-negative fibers (SDH Complex II intact; "
                    "  COX Complex IV deleted) = PATHOGNOMONIC pattern for large mtDNA deletion\n"
                    "BIOCHEMISTRY: Severely reduced Complex I, III, IV activity; Complex II normal "
                    "(measured by spectrophotometry on muscle homogenate)"
                ),
            },
            {
                "term": "Tissue Heteroplasmy Gradient — Why Blood Underestimates KSS",
                "definition": (
                    "Blood heteroplasmy in KSS is TYPICALLY 20–40 percentage points LOWER than "
                    "muscle heteroplasmy:\n"
                    "REASON: Rapidly dividing haematopoietic cells undergo selective purifying pressure "
                    "against deleted mtDNA (smaller deleted mitochondrial genomes are replicated faster "
                    "BUT haematopoietic stem cell division dilutes the deleted mtDNA through cell "
                    "divisions → blood ends up with lower deletion load).\n"
                    "POST-MITOTIC TISSUES (muscle, brain, heart): No dilution mechanism → "
                    "deleted mtDNA accumulates preferentially (replicative advantage) over decades.\n"
                    "CLINICAL IMPLICATIONS:\n"
                    "  1. Blood DNA testing may show 20–30% heteroplasmy → appears 'low' → may be "
                    "     misinterpreted as benign when muscle has 60–80% → fully symptomatic\n"
                    "  2. ALWAYS test muscle biopsy (or urine sediment as surrogate for non-muscle-biopsy "
                    "     scenarios) to accurately quantify deletion burden\n"
                    "  3. Normal blood mtDNA does NOT exclude KSS in clinically symptomatic patient\n"
                    "  4. Pearson Syndrome patients who survive: blood heteroplasmy decreases as "
                    "     haematopoiesis recovers → KSS features emerge in muscle/brain"
                ),
            },
        ],
        "disease_concepts": [
            {
                "term": "KSS vs CPEO vs Pearson — The Allelic Spectrum of Single Deletion",
                "definition": (
                    "All three conditions can be caused by the SAME 4977 bp deletion; phenotype depends on "
                    "tissue distribution + age at which deletion manifests:\n\n"
                    "KSS (OMIM #530000):\n"
                    "  • PEO + Pigmentary RP + Onset <20 (CARDINAL TRIAD) + CHB/Ataxia/CSF↑\n"
                    "  • High deletion burden in muscle/brain/heart/retina\n"
                    "  • Sporadic; systemic multi-organ involvement\n\n"
                    "Chronic Progressive External Ophthalmoplegia (CPEO):\n"
                    "  • PEO ONLY — no RP, no CHB, no systemic features\n"
                    "  • May be adult onset (over 20)\n"
                    "  • Same deletion OR nuclear gene mutations (POLG, TWNK, OPA1-Plus, DNA2)\n"
                    "  • DISTINCTION: absence of the systemic KSS features; usually milder course\n\n"
                    "Pearson Marrow-Pancreas Syndrome (OMIM #557000):\n"
                    "  • INFANT onset: bone marrow failure (sideroblastic anemia, vacuolated precursors)\n"
                    "    + exocrine pancreatic dysfunction (steatorrhoea)\n"
                    "  • SAME deletion as KSS — affects haematopoietic stem cells in infant\n"
                    "  • ~50% of Pearson survivors develop KSS in adolescence as blood deletion "
                    "    heteroplasmy drops but muscle/brain accumulate deletion\n"
                    "  • Monitor ALL Pearson survivors for PEO, RP, CHB annually\n\n"
                    "MOLECULAR DISTINCTION FROM POLG/TWNK:\n"
                    "  Single deletion (KSS/CPEO) = ONE band on long-range PCR / Southern blot\n"
                    "  Multiple deletions (POLG/TWNK) = multiple bands (ladder) on Southern blot\n"
                    "  Depletion syndromes (POLG/DGUOK/MPV17) = low copy number on qPCR"
                ),
            },
            {
                "term": "Pigmentary Retinopathy in KSS — NOT Classic RP",
                "definition": (
                    "The retinal disease in KSS is DISTINCT from classic retinitis pigmentosa (RP):\n\n"
                    "KSS RETINOPATHY ('salt and pepper'):\n"
                    "  • Macular > peripheral distribution (opposite of classic RP)\n"
                    "  • Coarse granular RPE changes ('salt and pepper' pattern)\n"
                    "  • ERG: extinguished (scotopic + photopic affected)\n"
                    "  • Visual acuity: variable; central vision at risk (macular involvement)\n"
                    "  • Bone-spicule pigmentation ABSENT or minimal (unlike classic RP)\n"
                    "  • Cause: photoreceptor and RPE OXPHOS failure (very high metabolic demand)\n\n"
                    "CLASSIC RP (rod-cone dystrophy):\n"
                    "  • Peripheral > central; bone-spicule pigmentation; night blindness first\n"
                    "  • Various genes: RPGR, RP2, PRPF31, CRB1, USH2A, RHO, PRPH2\n"
                    "  • ERG: scotopic affected first, then photopic\n\n"
                    "DDx LHON (MT-ND4/ND1/ND6):\n"
                    "  • LHON: central vision loss (cecocentral scotoma); retinal ganglion cell loss\n"
                    "  • Telangiectatic microangiopathy, disc pseudoedema in acute phase\n"
                    "  • NO pigmentary changes in LHON; NO PEO\n\n"
                    "MONITORING: Annual ophthalmology + OCT (macular thickness) + low-vision aids"
                ),
            },
            {
                "term": "KSS Histopathology — Diagnostic Muscle Biopsy",
                "definition": (
                    "Muscle biopsy in KSS is DIAGNOSTIC when combined with clinical features:\n\n"
                    "MODIFIED GOMORI TRICHROME:\n"
                    "  Ragged Red Fibers (RRF): mitochondrial accumulation at subsarcolemmal rim "
                    "(red-staining clumps under light microscopy)\n\n"
                    "CYTOCHROME C OXIDASE (COX) HISTOCHEMISTRY:\n"
                    "  COX-NEGATIVE fibers: blue staining only (no brown COX) — Complex IV depleted\n"
                    "  Complex IV subunits CO3 directly deleted by 4977 bp deletion\n\n"
                    "SDH (SUCCINATE DEHYDROGENASE) HISTOCHEMISTRY:\n"
                    "  SDH-POSITIVE ('ragged blue'): all SDH subunits nuclear-encoded → unaffected\n"
                    "  SDH+/COX- = PATHOGNOMONIC for large mtDNA deletion\n\n"
                    "ELECTRON MICROSCOPY:\n"
                    "  Abnormal mitochondrial cristae; paracrystalline inclusions\n"
                    "  Mitochondrial proliferation (response to OXPHOS failure)\n\n"
                    "LONG-RANGE PCR:\n"
                    "  Single deletion band (4977 bp shorter than wild-type) — NOT multiple bands\n"
                    "  DDx POLG/TWNK: multiple deletion ladder = NOT KSS single-deletion\n\n"
                    "SOUTHERN BLOT:\n"
                    "  Single stoichiometric deletion band at expected size; "
                    "heterogeneous smear → depletion syndrome or multiple deletions (not KSS)"
                ),
            },
        ],
        "prescribing_safety": [
            {
                "term": "Prescribing Safety Summary — KSS (mtDNA Single Large Deletion)",
                "definition": (
                    "ABSOLUTE CONTRAINDICATIONS (NEVER use):\n"
                    "  • VPA (valproate): 3 mechanisms — CoA sequestration + POLG inhibition + hepatotoxicity\n"
                    "  • Aminoglycosides: Cochlear OXPHOS failure → sudden irreversible deafness (additive)\n"
                    "  • Linezolid: Mito 23S rRNA inhibition → pan-OXPHOS depletion; use daptomycin instead\n"
                    "  • Metformin (if DM): Complex I inhibition → fatal lactic acidosis in KSS\n"
                    "  • Ketogenic Diet: Forces OXPHOS-dependent beta-oxidation → metabolic collapse\n\n"
                    "AVOID (high risk; use only if no alternative + specialist review):\n"
                    "  • Propofol infusion: PRIS risk (brief induction OK, not maintenance)\n"
                    "  • Phenobarbital: Complex I inhibition; use LEV instead\n"
                    "  • NSAIDs prolonged: Nephrotoxic in Fanconi subset; use paracetamol\n\n"
                    "CAUTION (monitor carefully):\n"
                    "  • Vancomycin: nephrotoxic; monitor trough + renal in Fanconi subset\n"
                    "  • Glucocorticoids: Hyperglycaemia in DM subset; wean quickly\n"
                    "  • SGLT-2 inhibitors: Monitor for Fanconi-amplified glycosuria/DKA risk\n\n"
                    "PREFERRED/SAFE MEDICATIONS:\n"
                    "  AED (if needed): LEV (levetiracetam) — renal excretion, no mito toxicity\n"
                    "  Antibiotics: beta-lactams, cephalosporins, carbapenems, daptomycin\n"
                    "  DM: Insulin + DPP-4 inhibitor (NEVER metformin)\n"
                    "  Hypoparathyroid: Calcium carbonate + calcitriol\n"
                    "  Mitochondrial support: CoQ10 ubiquinol 300–1200 mg/day + leucovorin 2.5–5 mg/day\n"
                    "                       + riboflavin B2 + thiamine B1\n\n"
                    "CARDIAC (CRITICAL — LIFE-SAVING):\n"
                    "  • Annual 12-lead ECG + Holter → ANY CHB → URGENT pacemaker\n"
                    "  • PR >200ms → immediate cardiology referral\n"
                    "  • 3rd-degree CHB → EMERGENCY dual-chamber pacemaker insertion\n\n"
                    "FASTING PROTOCOL (hospital / procedure):\n"
                    "  • NEVER fast >4 hours without IV dextrose GIR 6–8 mg/kg/min\n"
                    "  • Thiamine 100 mg IV with glucose infusion\n"
                    "  • Monitor lactate; treat >5 mmol/L with bicarbonate + IV dextrose"
                ),
            },
        ],
    }
