#!/usr/bin/env python3
"""MT-RNR1 — Mitochondrially Encoded 12S Ribosomal RNA / Aminoglycoside-Induced Sensorineural
Hearing Loss (AISNHL) + Non-Syndromic Maternally Inherited Hearing Loss.

MT-RNR1 (OMIM *561000) encodes the 12S ribosomal RNA (954 nt), the small subunit (mt-SSU) of
the mitoribosome. Unlike the 13 protein-coding mtDNA genes, MT-RNR1 encodes a structural RNA,
not a protein. Its variants cause HEARING LOSS — NOT OXPHOS deficiency — a completely unique
phenotype among all mtDNA genes.

  MT-RNR1 gene         OMIM *561000
  Gene product         12S rRNA (954 nt) — mt small subunit ribosomal RNA; NOT translated
  Genome               H-strand, rCRS positions 648–1601 (954 bp)
  Inheritance          MATERNAL — near-homoplasmic (homoplasmic in most carriers)
  Primary phenotype    Sensorineural hearing loss (SNHL) — NOT combined OXPHOS deficiency

UNIQUE MOLECULAR POSITION:
  MT-RNR1 is the ONLY mtDNA gene whose common pathogenic variants cause isolated SNHL without
  OXPHOS deficiency. The 12S rRNA forms the small subunit (mt-SSU / 28S subunit) of the
  mitoribosome. Point mutations in MT-RNR1 do NOT prevent ribosome assembly per se, but alter
  the decoding center geometry in a way that increases susceptibility to aminoglycosides.

m.1555A>G — THE MOST IMPORTANT VARIANT:
  Position: rCRS 1555 in the decoding loop (helix 44) of 12S rRNA
  Frequency: ~0.1–0.2% of ALL individuals carry this variant (MOST COMMON pathogenic mtDNA
             variant in the general population; estimated 1 in 500–1000 people)
  Near-homoplasmic in virtually all carriers (transmitted as homoplasmic from mother to children)
  Mechanism: m.1555A>G brings the 12S rRNA decoding center closer to the structure found in
    prokaryotic 16S rRNA — the target of aminoglycoside antibiotics (gentamicin, amikacin,
    tobramycin, streptomycin, neomycin, kanamycin). Normal human mt-12S rRNA differs enough
    from bacterial 16S rRNA to be relatively resistant to aminoglycosides; m.1555A>G collapses
    this protective structural difference → aminoglycosides bind the altered human mt-12S rRNA
    → mt-ribosome stalls → cochlear hair cell mtDNA-dependent protein synthesis fails →
    irreversible hair cell death → PERMANENT SNHL within 24–72 hours of aminoglycoside exposure.
  Penetrance WITHOUT aminoglycosides: ~20–30% develop progressive NSHL over years (modifier
    genes: TRMU/MTO1/GTPBP3; nuclear background; mtDNA copy number)
  Penetrance WITH aminoglycosides: ~100% develop severe-to-profound SNHL — often within 24 h

m.1494C>T — SECOND MOST IMPORTANT VARIANT (Asian/Chinese founder):
  Position: rCRS 1494, adjacent to 1555 in helix 44
  Same mechanism as 1555A>G — also sensitises to aminoglycosides
  Frequency: Rarer than 1555A>G; significant in East Asian (especially Chinese) populations
  Hu et al. 2006 discovered this in a large Chinese pedigree (Zeng 2005 NEJM context)

OTHER MT-RNR1 VARIANTS:
  m.827A>G: "highly conserved" A → G; NSHL in some families; less penetrant; lower frequency
  m.961T>C / delT / ins(C): Helix 44 variants; NSHL; heteroplasmic; intermediate phenotype
  m.1095T>C: Variable penetrance NSHL

PATHOPHYSIOLOGY:
  1. Aminoglycoside pathway (m.1555A>G and m.1494C>T):
     Normal mt-12S rRNA helix 44 → differs from E. coli 16S rRNA A-site → aminoglycosides
     do NOT bind efficiently → safe in normal individuals (cochlear hair cells resist)
     m.1555A>G/1494C>T → mt-12S rRNA helix 44 resembles E. coli A-site →
     aminoglycoside binds → mistranslation → mt-ribosome stalls → cochlear hair cell
     (high OXPHOS demand, ~1000 mitochondria per cell) energy failure → rapid irreversible death
  2. Nuclear modifier pathway (non-aminoglycoside NSHL):
     TRMU (tRNA thiol modification) deficiency → reduces mt-tRNA stability → compounding
     mt-translation impairment in MT-RNR1 carriers → NSHL without aminoglycosides
     MTO1/GTPBP3 modify mt-tUCN wobble position → same compounding effect
  3. Cochlear hair cell vulnerability: Outer hair cells (OHCs) of the basal turn (high-frequency
     region) are the most metabolically active → highest OXPHOS demand → most susceptible to
     mt-translation impairment from aminoglycosides → HIGH-FREQUENCY SNHL first

PHENOTYPE SPECTRUM:
  1. Aminoglycoside-induced SNHL (AISNHL) — the emergency:
     • Severe-to-profound bilateral SNHL within 24–72 h of ANY aminoglycoside dose
     • Even a SINGLE dose of gentamicin/amikacin/tobramycin can cause permanent deafness
     • Cochlear implants required in most; hearing aids insufficient for profound SNHL
     • NO recovery — hair cell death is permanent
     • Affects ALL ages including neonates (neonatal sepsis treated with gentamicin is high risk)
  2. Non-syndromic maternally inherited SNHL (without aminoglycosides):
     • 20–30% of m.1555A>G carriers develop progressive SNHL over decades (without aminoglycosides)
     • High-frequency hearing loss first (4–8 kHz), progressing to all frequencies
     • Onset: childhood to adult (variable due to nuclear modifiers)
     • Severity: mild to profound; worsens with age
  3. No OXPHOS deficiency — CI/CII/CIII/CIV/CV all NORMAL in 12S rRNA variants
  4. No neurological features — NO stroke-like episodes, NO seizures, NO Leigh syndrome,
     NO cardiomyopathy — DISTINGUISHING from tRNA gene variants (MELAS/MERRF/MT-TI/MT-TK)

TREATMENT:
  ABSOLUTE CONTRAINDICATION — aminoglycosides in ALL known MT-RNR1 m.1555A>G/m.1494C>T carriers:
    Gentamicin (including topical/ear drops) — most common cause of AISNHL
    Amikacin — TB/serious GNB treatment; ABSOLUTE CI in MT-RNR1 carriers
    Tobramycin — ABSOLUTE CI (including inhaled tobramycin in CF patients)
    Streptomycin — ABSOLUTE CI (TB first-line in some settings)
    Neomycin — ABSOLUTE CI (including topical preparations, e.g. Neosporin)
    Kanamycin — ABSOLUTE CI
    Spectinomycin — ABSOLUTE CI
  NEVER give ANY aminoglycoside to a carrier without confirming lack of MT-RNR1 1555/1494 variant.
  In emergencies where no mtDNA result is available: use non-aminoglycoside alternatives
    (piperacillin-tazobactam, cefepime, meropenem) for gram-negative coverage.
  OTHER DRUGS TO AVOID:
    Cisplatin — ototoxic by different mechanism (cochlear hair cell ROS); compounding risk in MT-RNR1
    Loop diuretics (high-dose furosemide IV) — ototoxic; avoid concurrent aminoglycoside
    Aspirin/salicylates — ototoxic potential at high dose
  SCREENING:
    Family cascade testing: all maternal relatives of any m.1555A>G or m.1494C>T carrier
    Newborn screening (NBS) for m.1555A>G: implemented in some countries (UK, China, parts of Europe)
    Pre-aminoglycoside genetic screening: MANDATORY in elective settings; rapid test available
  TREATMENT OF ESTABLISHED SNHL:
    Hearing aids: mild-to-moderate SNHL
    Cochlear implants: severe-to-profound SNHL; excellent outcomes in MT-RNR1 SNHL
    Educational support / communication therapy
  LEVEL C COFACTORS (theoretical; no randomised trial):
    Antioxidants: CoQ10, Vitamin C, Vitamin E — may reduce ROS-mediated hair cell death
    No strong evidence; not curative

DIAGNOSTIC PATHWAY:
  1. Audiometry: High-frequency SNHL pattern (4 kHz notch → progression to all frequencies)
  2. mtDNA sequencing (blood sufficient — m.1555A>G is homoplasmic, no muscle needed)
     Blood leukocyte DNA is reliable (homoplasmic = same in all tissues)
  3. Family history: Maternal inheritance pattern (all maternal relatives at risk)
  4. WES MISSES MT-RNR1: WES does NOT include mtDNA reliably; requires dedicated mt-12S rRNA
     sequencing or mitoPanel / NextGeneSens / Sanger m.1555 + m.1494 assay

REFERENCES (key):
  Prezant TR, Agapian JV, Bohlman MC, et al. (1993) Mitochondrial ribosomal RNA mutation
    associated with both antibiotic-induced and non-syndromic deafness. Nat Genet 4(3):289-94
    — DISCOVERY of m.1555A>G; seminal paper.
  Hutchin TP, Haworth I, Higashi K, et al. (1993) A molecular basis for human hypersensitivity
    to aminoglycoside antibiotics. Nucleic Acids Res 21(18):4174-9 — mechanism m.1555A>G.
  Hu DN, Qui WQ, Wu BT, et al. (1991) Genetic aspects of antibiotic induced deafness:
    mitochondrial inheritance. J Med Genet 28:79-83 — Chinese cohort aminoglycoside deafness.
  Zeng FG (2005) Trends in cochlear implants. Trends Amplification 8(1):1-34 — cochlear implant
    outcomes in aminoglycoside deafness including MT-RNR1.
  Ramos A, Guerra-Assuncao JA, Betsou F, et al. (2013) Population variation and mutation
    spectra of the human mt 12S rRNA. Mitochondrion 13(6):822-30 — variant frequency/spectrum.
  Bitner-Glindzicz M, Pembrey M, Duncan A, et al. (2009) Prevalence of mitochondrial 1555A→G
    mutation in European children. N Engl J Med 360(6):640-2 — ~1 in 500 population prevalence.
  Casano RA, Johnson DF, Bykhovskaya Y, et al. (1999) Inherited susceptibility to aminoglycoside
    ototoxicity: genetic heterogeneity and role of the 1555A→G mitochondrial mutation. Am J Med Genet
    89(3):167-71 — genetic heterogeneity; nuclear modifier evidence.
  Zhao H, Li R, Wang Q, et al. (2004) Maternally inherited aminoglycoside-induced and nonsyndromic
    deafness is associated with the novel C1494T mutation in the mitochondrial 12S rRNA gene in a
    large Chinese family. Am J Hum Genet 74(1):139-52 — m.1494C>T discovery.
"""

import random

SEED = 841

# Pathogenic variants in MT-RNR1 causing AISNHL / NSHL
VARIANTS = [
    {
        "change": "m.1555A>G",
        "location": "Helix 44 / A-site decoding loop (rCRS 1555)",
        "type": "Homoplasmic SNV",
        "severity": "Severe–Profound (with aminoglycosides) / Mild–Severe (without)",
        "phenotype": "AISNHL (with aminoglycosides) or progressive NSHL",
        "notes": (
            "MOST COMMON pathogenic mtDNA variant — ~0.1–0.2% population; ~1 in 500–1000 carriers. "
            "Near-homoplasmic (transmitted homoplasmic to all children). "
            "ABSOLUTE CI to aminoglycosides — any dose causes permanent SNHL within 24–72 h. "
            "Without aminoglycosides: 20–30% penetrance for progressive NSHL. "
            "Prezant 1993 Nat Genet (discovery) — Bitner-Glindzicz 2009 NEJM (prevalence study)."
        ),
        "allele_freq_pct": 65,
    },
    {
        "change": "m.1494C>T",
        "location": "Helix 44 / adjacent to A-site decoding loop (rCRS 1494)",
        "type": "Homoplasmic SNV (founder — East Asian)",
        "severity": "Severe–Profound (with aminoglycosides)",
        "phenotype": "AISNHL (aminoglycoside-induced) / progressive NSHL",
        "notes": (
            "SECOND most important MT-RNR1 variant. Same mechanism as m.1555A>G — sensitises "
            "helix 44 to aminoglycosides. East Asian (especially Chinese) founder effect. "
            "Homoplasmic; ABSOLUTE CI to aminoglycosides. "
            "Zhao 2004 AJHG (discovery in large Chinese family); Hu 1991 J Med Genet (clinical context)."
        ),
        "allele_freq_pct": 15,
    },
    {
        "change": "m.827A>G",
        "location": "Helix 21 / central pseudoknot region (rCRS 827)",
        "type": "Homoplasmic SNV",
        "severity": "Mild–Moderate",
        "phenotype": "Non-syndromic SNHL (lower penetrance, slower progression)",
        "notes": (
            "Highly conserved A→G; associated with NSHL in some families. "
            "Lower penetrance than m.1555A>G; slower high-frequency progression. "
            "Aminoglycoside risk is lower than m.1555A>G but avoid as precaution. "
            "Ramos 2013 Mitochondrion — variant spectrum analysis."
        ),
        "allele_freq_pct": 10,
    },
    {
        "change": "m.961T>C / delT / insC",
        "location": "Helix 44 proximal / stem loop (rCRS 961)",
        "type": "Heteroplasmic or homoplasmic SNV / indel",
        "severity": "Mild–Moderate",
        "phenotype": "NSHL (progressive, variable penetrance)",
        "notes": (
            "Cluster of variants at position 961; heteroplasmic variants possible. "
            "Progressive high-frequency SNHL; variable penetrance. "
            "Aminoglycoside risk; ABSOLUTE CI as precaution in m.961 carriers presenting with SNHL. "
            "Casano 1999 Am J Med Genet — heterogeneity context."
        ),
        "allele_freq_pct": 10,
    },
]

_VARIANT_CHOICES = [v["change"] for v in VARIANTS]
_VARIANT_WEIGHTS = [v["allele_freq_pct"] for v in VARIANTS]


def _make_patients():
    rng = random.Random(SEED)
    patients = []

    for i in range(40):
        pid = f"MTRNR1-{i+1:03d}"

        # Draw variant
        variant = rng.choices(_VARIANT_CHOICES, weights=_VARIANT_WEIGHTS, k=1)[0]
        var_obj = next(v for v in VARIANTS if v["change"] == variant)

        # Aminoglycoside exposure history
        is_1555_or_1494 = variant in ("m.1555A>G", "m.1494C>T")
        aminoglycoside_exposed = is_1555_or_1494 and rng.random() < 0.55
        # If exposed with 1555/1494 → severe/profound SNHL
        if aminoglycoside_exposed:
            snhl_severity = rng.choices(
                ["Severe", "Profound"], weights=[35, 65], k=1
            )[0]
            snhl_onset_age_years = round(rng.uniform(0, 45), 1)  # can be any age
            aminoglycoside_agent = rng.choice(
                ["Gentamicin", "Amikacin", "Tobramycin", "Streptomycin"]
            )
            aminoglycoside_indication = rng.choice(
                [
                    "Neonatal sepsis (neonatal intensive care)",
                    "Gram-negative bacteraemia (ICU)",
                    "TB treatment (streptomycin)",
                    "Cystic fibrosis exacerbation (tobramycin)",
                    "Surgical prophylaxis",
                    "Urinary tract infection empiric",
                ]
            )
        else:
            # Non-aminoglycoside SNHL
            aminoglycoside_exposed = False
            aminoglycoside_agent = None
            aminoglycoside_indication = None
            if variant == "m.1555A>G":
                penetrant = rng.random() < 0.25  # 20-30% penetrance without aminoglycosides
            elif variant == "m.1494C>T":
                penetrant = rng.random() < 0.20
            elif variant == "m.827A>G":
                penetrant = rng.random() < 0.40
            else:  # m.961
                penetrant = rng.random() < 0.45
            if penetrant:
                snhl_severity = rng.choices(
                    ["Mild", "Moderate", "Severe"], weights=[40, 40, 20], k=1
                )[0]
                snhl_onset_age_years = round(rng.uniform(5, 60), 1)
            else:
                snhl_severity = "Asymptomatic carrier"
                snhl_onset_age_years = None

        # Laterality
        laterality = rng.choices(
            ["Bilateral", "Bilateral (asymmetric)", "Unilateral"],
            weights=[70, 20, 10], k=1
        )[0]

        # Audiogram pattern
        if snhl_severity not in ("Asymptomatic carrier",):
            audiogram = rng.choices(
                [
                    "High-frequency (4–8 kHz notch, progressive)",
                    "Flat moderately severe (all frequencies)",
                    "Sloping high-frequency (2–8 kHz, moderate to profound)",
                    "Profound (all frequencies, post-aminoglycoside)",
                ],
                weights=[35, 20, 25, 20], k=1
            )[0]
        else:
            audiogram = "Normal audiogram (asymptomatic carrier)"

        # Tinnitus
        tinnitus = snhl_severity not in ("Asymptomatic carrier",) and rng.random() < 0.55

        # Maternal family history of SNHL
        maternal_family_snhl = rng.random() < 0.62  # mtDNA is maternal; homoplasmic → all relatives

        # Nuclear modifiers documented
        trmu_modifier = rng.random() < 0.18  # TRMU hypomethylation nuclear modifier
        mto1_modifier = rng.random() < 0.12

        # Cochlear implant
        cochlear_implant = snhl_severity in ("Severe", "Profound") and rng.random() < 0.72
        hearing_aid = snhl_severity in ("Mild", "Moderate") and rng.random() < 0.80

        # Outcome
        if snhl_severity == "Profound" and cochlear_implant:
            outcome = "Cochlear implant — partial hearing restoration"
        elif snhl_severity == "Profound":
            outcome = "Profound SNHL — deaf (no implant)"
        elif snhl_severity == "Severe" and cochlear_implant:
            outcome = "Cochlear implant — good speech comprehension"
        elif snhl_severity == "Severe":
            outcome = "Severe SNHL — hearing aid partial benefit"
        elif snhl_severity == "Moderate" and hearing_aid:
            outcome = "Moderate SNHL — hearing aid benefit"
        elif snhl_severity == "Mild":
            outcome = "Mild SNHL — hearing aid if needed"
        else:
            outcome = "Asymptomatic carrier — no hearing loss yet"

        sex = rng.choice(["M", "F"])
        age_at_diagnosis = round(rng.uniform(0.5, 65), 1)

        patients.append({
            "patient_id": pid,
            "sex": sex,
            "age_at_diagnosis_years": age_at_diagnosis,
            "variant": variant,
            "aminoglycoside_exposed": aminoglycoside_exposed,
            "aminoglycoside_agent": aminoglycoside_agent,
            "aminoglycoside_indication": aminoglycoside_indication,
            "snhl_severity": snhl_severity,
            "snhl_onset_age_years": snhl_onset_age_years,
            "laterality": laterality,
            "audiogram_pattern": audiogram,
            "tinnitus": tinnitus,
            "maternal_family_snhl": maternal_family_snhl,
            "trmu_modifier": trmu_modifier,
            "mto1_modifier": mto1_modifier,
            "cochlear_implant": cochlear_implant,
            "hearing_aid": hearing_aid,
            "outcome": outcome,
            "inheritance": "Maternal (mtDNA — homoplasmic in m.1555A>G/m.1494C>T)",
            "no_oxphos_deficiency": True,  # key: all OXPHOS complexes normal
        })

    return patients


def _cohort_stats(patients):
    n = len(patients)

    def pct(field, val=True):
        if callable(val):
            return round(100 * sum(1 for p in patients if val(p)) / n, 1)
        return round(100 * sum(1 for p in patients if p.get(field) == val) / n, 1)

    aminoglycoside_exposed_n = sum(1 for p in patients if p["aminoglycoside_exposed"])
    severe_profound_n = sum(1 for p in patients if p["snhl_severity"] in ("Severe", "Profound"))
    ci_n = sum(1 for p in patients if p["cochlear_implant"])
    asymptomatic_n = sum(1 for p in patients if p["snhl_severity"] == "Asymptomatic carrier")
    m1555_n = sum(1 for p in patients if p["variant"] == "m.1555A>G")

    return {
        "n": n,
        "aminoglycoside_exposed_pct": round(100 * aminoglycoside_exposed_n / n, 1),
        "severe_profound_pct": round(100 * severe_profound_n / n, 1),
        "cochlear_implant_pct": round(100 * ci_n / n, 1),
        "asymptomatic_carrier_pct": round(100 * asymptomatic_n / n, 1),
        "tinnitus_pct": round(100 * sum(1 for p in patients if p["tinnitus"]) / n, 1),
        "maternal_family_snhl_pct": round(100 * sum(1 for p in patients if p["maternal_family_snhl"]) / n, 1),
        "hearing_aid_pct": round(100 * sum(1 for p in patients if p["hearing_aid"]) / n, 1),
        "m1555_pct": round(100 * m1555_n / n, 1),
        "bilateral_pct": round(100 * sum(1 for p in patients if "Bilateral" in (p["laterality"] or "")) / n, 1),
        "trmu_modifier_pct": round(100 * sum(1 for p in patients if p["trmu_modifier"]) / n, 1),
        "no_oxphos_deficiency_pct": round(100 * sum(1 for p in patients if p.get("no_oxphos_deficiency")) / n, 1),
    }


def get_overview():
    patients = _make_patients()
    stats = _cohort_stats(patients)

    features = [
        {"feature": "SNHL severity: Severe or Profound", "pct": stats["severe_profound_pct"]},
        {"feature": "Aminoglycoside exposure (precipitating)", "pct": stats["aminoglycoside_exposed_pct"]},
        {"feature": "Cochlear implant required", "pct": stats["cochlear_implant_pct"]},
        {"feature": "Tinnitus present", "pct": stats["tinnitus_pct"]},
        {"feature": "Bilateral SNHL", "pct": stats["bilateral_pct"]},
        {"feature": "Maternal family history of SNHL", "pct": stats["maternal_family_snhl_pct"]},
        {"feature": "Asymptomatic carrier (no SNHL yet)", "pct": stats["asymptomatic_carrier_pct"]},
        {"feature": "Hearing aid benefit (mild-moderate)", "pct": stats["hearing_aid_pct"]},
        {"feature": "NO OXPHOS deficiency (CI/CII/CIII/CIV/CV normal)", "pct": stats["no_oxphos_deficiency_pct"]},
        {"feature": "TRMU nuclear modifier documented", "pct": stats["trmu_modifier_pct"]},
        {"feature": "m.1555A>G (main variant)", "pct": stats["m1555_pct"]},
        {"feature": "Maternal family SNHL (cascade testing needed)", "pct": stats["maternal_family_snhl_pct"]},
    ]

    from collections import Counter
    v_counter = Counter(p["variant"] for p in patients)
    top_variants = [{"variant": k, "count": v} for k, v in v_counter.most_common(4)]

    alerts = [
        "🚨 m.1555A>G / m.1494C>T: ABSOLUTE CONTRAINDICATION to ALL aminoglycosides — gentamicin, amikacin, tobramycin, streptomycin, neomycin, kanamycin, spectinomycin. EVEN A SINGLE DOSE causes permanent severe-to-profound SNHL within 24–72 hours. No recovery possible.",
        "🚨 NEONATAL RISK: Neonates with m.1555A>G treated with gentamicin for sepsis are at IMMEDIATE risk of permanent deafness — genetic result must precede non-emergency aminoglycoside use.",
        "🚫 Cisplatin — cochlear ototoxin by ROS mechanism; compounding risk; AVOID in MT-RNR1 carriers where possible.",
        "🚫 High-dose IV loop diuretics (furosemide) — avoid concurrent aminoglycoside + loop diuretic (additive ototoxicity even in non-MT-RNR1 patients; amplified in carriers).",
        "⚠️ MATERNAL INHERITANCE: All maternal relatives (mother, maternal siblings, maternal aunts) are potentially carriers — homoplasmic transmission; cascade mtDNA testing MANDATORY.",
        "⚠️ WES MISSES MT-RNR1: Standard WES does NOT reliably call mtDNA variants — dedicated m.1555A>G / m.1494C>T PCR-RFLP, Sanger, or mtDNA panel required.",
        "⚠️ Blood DNA sufficient: Unlike protein-coding mtDNA genes (heteroplasmic), m.1555A>G is HOMOPLASMIC — blood leukocyte DNA is reliable; no muscle biopsy needed.",
        "✅ Cochlear implants: EXCELLENT outcomes in MT-RNR1 AISNHL — cochlear nerve is intact; hair cell death only; CI hearing restoration is near-normal in many patients.",
        "✅ Newborn screening (NBS) for m.1555A>G: Implemented in UK, China, and some European countries — identify carriers BEFORE aminoglycoside exposure.",
        "✅ Family cascade testing: All maternal relatives must be tested and counselled about aminoglycoside avoidance — LIFE-SAVING prevention.",
        "✅ In emergencies where m.1555A>G result unknown: Use non-aminoglycoside gram-negative coverage (piperacillin-tazobactam, cefepime, meropenem) as default.",
    ]

    return {
        "gene": "MT-RNR1",
        "full_name": "Mitochondrially Encoded 12S Ribosomal RNA",
        "alias": "12S rRNA / mt-12S / MTRNR1 / OMIM *561000",
        "omim_gene": "561000",
        "omim_disease": "Aminoglycoside-induced sensorineural hearing loss / non-syndromic maternally inherited SNHL",
        "disease_name": (
            "Aminoglycoside-Induced Sensorineural Hearing Loss (AISNHL) + Non-Syndromic "
            "Maternally Inherited Hearing Loss — MT-RNR1 m.1555A>G / m.1494C>T"
        ),
        "chromosome": "Mitochondrial DNA (mtDNA) — H-strand, rCRS 648–1601 (954 nt)",
        "inheritance": "Maternal (mtDNA) — HOMOPLASMIC (m.1555A>G/m.1494C>T; near-universal homoplasmy)",
        "product": "12S ribosomal RNA (954 nt) — mt small subunit (mt-SSU / 28S) — NOT translated into protein",
        "population_frequency": "~0.1–0.2% of all individuals (m.1555A>G) — 1 in 500–1000 people",
        "protein_size": "N/A — RNA gene (954 nt); NOT a protein-coding gene",
        "rna": {
            "length_nt": 954,
            "type": "12S ribosomal RNA (small subunit)",
            "ribosome": "Mitoribosome small subunit (mt-SSU / 28S): 12S rRNA + ~30 mitoribosomal proteins (MRPs)",
            "function": (
                "Structural scaffold of the mt-SSU decoding center; "
                "positions mt-tRNAs at the A-site for codon–anticodon matching; "
                "assembly factors: ERAL1 (stabilises 3′ end), RBFA (rRNA processing / anti-association)"
            ),
            "key_domain": "Helix 44 (rCRS ~1481–1571) — the A-site decoding loop; site of m.1555A>G and m.1494C>T",
        },
        "key_message": (
            "MT-RNR1 is UNIQUE among all mtDNA genes: pathogenic variants cause "
            "ISOLATED SENSORINEURAL HEARING LOSS — NOT combined OXPHOS deficiency. "
            "m.1555A>G (~0.1–0.2% of all people) is the MOST COMMON pathogenic mtDNA variant "
            "in the human population and a COMPLETELY PREVENTABLE cause of deafness. "
            "AMINOGLYCOSIDES ARE ABSOLUTELY CONTRAINDICATED IN ALL CARRIERS."
        ),
        "cohort_n": len(patients),
        "seed": SEED,
        "patients": patients[:10],
        "cohort_statistics": stats,
        "cohort_summary_features": features,
        "key_clinical_alerts": alerts,
        "top_variant_counts": top_variants,
        "phenotype_distribution": {
            "aminoglycoside_induced_snhl_pct": stats["aminoglycoside_exposed_pct"],
            "non_aminoglycoside_snhl_pct": round(
                100 - stats["aminoglycoside_exposed_pct"] - stats["asymptomatic_carrier_pct"], 1
            ),
            "asymptomatic_carrier_pct": stats["asymptomatic_carrier_pct"],
        },
        "contrast_with_oxphos_genes": {
            "OXPHOS_deficiency": "ABSENT — CI/CII/CIII/CIV/CV all NORMAL in MT-RNR1 variants",
            "Leigh_syndrome": "ABSENT — NOT a cause of Leigh syndrome",
            "stroke_like_episodes": "ABSENT — unlike MELAS/MT-TL1",
            "myoclonic_epilepsy": "ABSENT — unlike MERRF/MT-TK",
            "cardiomyopathy": "ABSENT — unlike MT-TI/SCO2",
            "lactic_acidosis": "ABSENT (or mild if very severe SNHL, secondary)",
            "ragged_red_fibres": "ABSENT — no muscle pathology",
        },
    }


def get_breakdown():
    patients = _make_patients()
    stats = _cohort_stats(patients)
    n = len(patients)

    from collections import Counter
    v_counter = Counter(p["variant"] for p in patients)
    variant_dist = [
        {"variant": k, "count": v, "allele_freq_pct": round(100 * v / n, 1)}
        for k, v in v_counter.most_common()
    ]

    # Severity distribution
    sev_counter = Counter(p["snhl_severity"] for p in patients)
    severity_dist = [
        {"severity": k, "count": v, "pct": round(100 * v / n, 1)}
        for k, v in sev_counter.most_common()
    ]

    # Aminoglycoside agents
    ag_list = [p["aminoglycoside_agent"] for p in patients if p["aminoglycoside_agent"]]
    ag_counter = Counter(ag_list)
    ag_dist = [
        {"agent": k, "count": v, "pct": round(100 * v / n, 1)}
        for k, v in ag_counter.most_common()
    ]

    # Audiogram pattern distribution
    aud_counter = Counter(p["audiogram_pattern"] for p in patients)
    audiogram_dist = [
        {"pattern": k, "count": v, "pct": round(100 * v / n, 1)}
        for k, v in aud_counter.most_common()
    ]

    # Outcome distribution
    oc_counter = Counter(p["outcome"] for p in patients)
    outcomes = [
        {"outcome": k, "count": v, "pct": round(100 * v / n, 1)}
        for k, v in oc_counter.most_common()
    ]

    # Aminoglycoside indications
    ind_list = [p["aminoglycoside_indication"] for p in patients if p["aminoglycoside_indication"]]
    ind_counter = Counter(ind_list)
    indication_dist = [
        {"indication": k, "count": v}
        for k, v in ind_counter.most_common()
    ]

    treatment_uptake = {
        "Cochlear implant (severe-profound SNHL)": f"{sum(1 for p in patients if p['cochlear_implant'])} / {n} patients",
        "Hearing aid (mild-moderate SNHL)": f"{sum(1 for p in patients if p['hearing_aid'])} / {n} patients",
        "Aminoglycoside avoidance counselling": f"{n} / {n} patients (MANDATORY all carriers)",
        "Maternal family cascade testing": f"{round(0.82 * n)} / {n} patients (completed)",
        "Antioxidant cocktail (CoQ10 / Vit C — Level C)": f"{round(0.38 * n)} / {n} patients",
        "Communication / speech therapy": f"{round(0.62 * n)} / {n} patients",
    }

    nuclear_modifier_breakdown = {
        "TRMU (tRNA thiol modification — nuclear modifier)": f"{stats['trmu_modifier_pct']}% of cohort",
        "MTO1/GTPBP3 (mt-tRNA modification — nuclear modifier)": f"{stats['trmu_modifier_pct'] * 0.65:.1f}% estimated",
        "ERAL1 (12S rRNA stabilisation factor — nuclear)": "Assembly factor; variants reduce 12S rRNA stability",
        "RBFA (rRNA processing / anti-association — nuclear)": "Maturation factor; MT-RNR1 + RBFA → mt-SSU",
    }

    return {
        "gene": "MT-RNR1",
        "all_variants": VARIANTS,
        "variant_distribution": variant_dist,
        "severity_distribution": severity_dist,
        "audiogram_distribution": audiogram_dist,
        "aminoglycoside_agent_distribution": ag_dist,
        "aminoglycoside_indication_distribution": indication_dist,
        "outcome_distribution": outcomes,
        "cohort_statistics": stats,
        "treatment_uptake": treatment_uptake,
        "nuclear_modifier_breakdown": nuclear_modifier_breakdown,
        "key_contrasts": {
            "vs_MT_tRNA_genes": (
                "MT-RNR1: ISOLATED SNHL only — NO combined OXPHOS deficiency. "
                "MT-TK (MERRF)/MT-TL1 (MELAS)/MT-TI/MT-TE: ALL cause combined OXPHOS deficiency "
                "(CI+CIV or CI+CIII+CIV reduced) with multi-organ involvement. "
                "MT-RNR1: audiometry + mtDNA blood test only; NO muscle biopsy required."
            ),
            "vs_GJB2_DFNB1": (
                "GJB2 (Connexin 26, DFNB1) — AR biallelic, most common genetic SNHL in many populations; "
                "NO aminoglycoside sensitivity; NO maternal inheritance. "
                "MT-RNR1 — maternal; homoplasmic; AMINOGLYCOSIDE SENSITIVITY is pathognomonic distinction."
            ),
            "vs_SLC26A4_Pendred": (
                "SLC26A4 (Pendred/EVA) — AR; enlarged vestibular aqueduct on MRI; goitre; NO aminoglycoside CII. "
                "MT-RNR1 — maternal; NO enlarged vestibular aqueduct; NO goitre; aminoglycoside ABSOLUTE CI."
            ),
            "vs_KCNQ4_DFNA2": (
                "KCNQ4 (DFNA2) — AD; OHC degeneration; high-frequency progressive; NO aminoglycoside CI. "
                "MT-RNR1 — maternal; aminoglycoside sensitivity; blood mtDNA test diagnostic."
            ),
        },
        "aminoglycoside_classes": {
            "ABSOLUTE_CI": [
                "Gentamicin (Garamycin) — most common cause of MT-RNR1 AISNHL (neonatal sepsis / hospital use)",
                "Amikacin — TB/serious GNR; ABSOLUTE CI in all MT-RNR1 carriers",
                "Tobramycin — CF exacerbations (inhaled + IV); ABSOLUTE CI",
                "Streptomycin — TB first-line (some countries); ABSOLUTE CI",
                "Neomycin — topical (Neosporin, bowel prep); ABSOLUTE CI including topical preparations",
                "Kanamycin — MDR-TB; ABSOLUTE CI",
                "Spectinomycin — gonorrhoea; ABSOLUTE CI",
                "Paromomycin — parasitic; ABSOLUTE CI",
            ],
            "Safe_alternatives_for_gram_negatives": [
                "Piperacillin-tazobactam (extended-spectrum beta-lactam)",
                "Cefepime (4th-gen cephalosporin — GNR coverage)",
                "Meropenem / Imipenem (carbapenem — broad GNR)",
                "Aztreonam (monobactam — GNR, safe in aminoglycoside allergy context)",
                "Colistin / Polymyxin B — last resort MDR; different ototoxicity mechanism",
            ],
        },
    }


def get_definitions():
    return {
        "gene": "MT-RNR1",
        "full_name": "Mitochondrially Encoded 12S Ribosomal RNA",
        "alias": "12S rRNA / mt-12S rRNA / MTRNR1 / OMIM *561000",
        "omim_gene": "561000",
        "omim_disease": "Aminoglycoside-induced SNHL / non-syndromic maternally inherited hearing loss",
        "disease_name": (
            "Aminoglycoside-Induced Sensorineural Hearing Loss (AISNHL) + "
            "Non-Syndromic Maternally Inherited Hearing Loss — m.1555A>G / m.1494C>T"
        ),
        "chromosome": "Mitochondrial DNA (mtDNA) H-strand, rCRS 648–1601 (954 nt RNA gene)",
        "inheritance": "Maternal (mtDNA) — HOMOPLASMIC in m.1555A>G/m.1494C>T; all children of carrier mother inherit",
        "product": {
            "type": "12S ribosomal RNA (RNA gene — NOT translated into protein)",
            "length_nt": 954,
            "ribosome_unit": "Mitoribosome small subunit (mt-SSU / 28S subunit)",
            "function": (
                "Structural scaffold of the mt decoding center; "
                "positions aminoacyl-mt-tRNAs at the A-site for codon–anticodon recognition; "
                "helix 44 (rCRS ~1481–1571) is the A-site decoding loop targeted by aminoglycosides"
            ),
        },
        "key_variants": [
            {
                "variant": "m.1555A>G",
                "frequency": "~0.1–0.2% of all people (~1 in 500–1000) — MOST COMMON pathogenic mtDNA variant",
                "mechanism": (
                    "Helix 44 A→G makes mt-12S rRNA decoding loop resemble prokaryotic 16S rRNA A-site → "
                    "aminoglycosides bind → mt-ribosome stalls → cochlear hair cell energy failure → permanent SNHL"
                ),
                "penetrance_with_AG": "~100% — severe-to-profound SNHL within 24–72 h",
                "penetrance_without_AG": "~20–30% — progressive NSHL over years (nuclear modifier dependent)",
            },
            {
                "variant": "m.1494C>T",
                "frequency": "Rarer; East Asian (Chinese) founder variant",
                "mechanism": "Same as m.1555A>G — helix 44 adjacent position; aminoglycoside sensitisation",
                "penetrance_with_AG": "~100%",
                "penetrance_without_AG": "~15–25%",
            },
        ],
        "absolute_contraindications": [
            "Gentamicin — ABSOLUTE CI in ALL MT-RNR1 1555/1494 carriers",
            "Amikacin — ABSOLUTE CI",
            "Tobramycin — ABSOLUTE CI (inhaled + IV)",
            "Streptomycin — ABSOLUTE CI",
            "Neomycin — ABSOLUTE CI (including topical)",
            "Kanamycin — ABSOLUTE CI",
            "Spectinomycin — ABSOLUTE CI",
            "Paromomycin — ABSOLUTE CI",
            "Cisplatin — avoid where possible (ROS ototoxin, compounding cochlear damage)",
        ],
        "recommended_treatments": [
            "Aminoglycoside avoidance counselling — ALL carriers; medical alert bracelet recommended",
            "Cascade maternal family testing — ALL maternal relatives require mtDNA testing (blood is sufficient)",
            "Cochlear implants — preferred for severe-to-profound AISNHL; cochlear nerve intact; excellent outcomes",
            "Hearing aids — mild-to-moderate non-aminoglycoside NSHL",
            "Newborn screening for m.1555A>G — implemented in UK, China; identify carriers pre-aminoglycoside",
            "Pre-aminoglycoside genetic screening — MANDATORY in elective settings",
            "Emergency gram-negative alternatives: piperacillin-tazobactam, cefepime, meropenem (non-aminoglycoside)",
            "CoQ10 / Antioxidants — Level C; theoretical; no randomised trial evidence",
        ],
        "key_ddx": [
            {
                "condition": "GJB2 (Connexin 26 / DFNB1) — AR biallelic",
                "distinguishing": (
                    "GJB2: AR (not maternal); NO aminoglycoside hypersensitivity; most common AR SNHL; "
                    "WES detectable (nuclear); MT-RNR1: maternal inheritance; aminoglycoside sensitivity; "
                    "blood mtDNA test (not WES)"
                ),
            },
            {
                "condition": "SLC26A4 (Pendred syndrome / EVA) — AR biallelic",
                "distinguishing": (
                    "SLC26A4: AR; enlarged vestibular aqueduct (EVA) on CT/MRI; goitre (Pendred); "
                    "NO aminoglycoside CI; WES detectable. "
                    "MT-RNR1: NO EVA; NO goitre; ABSOLUTE aminoglycoside CI; maternal mtDNA"
                ),
            },
            {
                "condition": "MT-tRNA genes (MELAS/MERRF/MT-TI/MT-TK/MT-TS1)",
                "distinguishing": (
                    "MT-tRNA variants: COMBINED OXPHOS deficiency (CI+CIV/CIII+CIV reduced); "
                    "multi-organ (Leigh/stroke-like/myoclonic epilepsy/cardiomyopathy). "
                    "MT-RNR1: ISOLATED SNHL; NO OXPHOS deficiency; CI/CII/CIII/CIV/CV all NORMAL"
                ),
            },
            {
                "condition": "Noise-induced / age-related SNHL",
                "distinguishing": (
                    "Noise/age SNHL: bilateral high-frequency; NO maternal clustering; "
                    "NO aminoglycoside history; NO mtDNA variant. "
                    "MT-RNR1: maternal family history; aminoglycoside trigger or NSHL without noise exposure; "
                    "blood mtDNA test confirms"
                ),
            },
            {
                "condition": "Ototoxicity from drugs (non-aminoglycoside) — cisplatin, loop diuretics",
                "distinguishing": (
                    "Cisplatin/loop diuretic ototoxicity: affects ALL patients at high dose (not genotype-specific); "
                    "MT-RNR1: aminoglycoside sensitivity is GENOTYPE-SPECIFIC (m.1555A>G/m.1494C>T carriers only); "
                    "MT-RNR1 carriers are ADDITIONALLY at risk from non-aminoglycoside ototoxins"
                ),
            },
        ],
        "genetic_counselling": {
            "recurrence_risk": (
                "MATERNAL inheritance — ALL children of an affected/carrier mother inherit the same "
                "homoplasmic mtDNA variant; no skipping generation; both sons and daughters affected equally. "
                "Father-to-child transmission NEVER occurs."
            ),
            "cascade_testing": (
                "ALL maternal relatives — mother, maternal siblings (brothers and sisters), maternal aunts/uncles, "
                "maternal first cousins — should be tested by blood mtDNA sequencing. "
                "Homoplasmic: blood leukocyte DNA is reliable (no muscle needed)."
            ),
            "prenatal_diagnosis": (
                "CVS or amniocentesis: blood-level DNA testing reliable for homoplasmic m.1555A>G/m.1494C>T. "
                "All fetuses of carrier mothers will inherit the variant."
            ),
            "medical_alert": (
                "All carriers should wear a MedicAlert bracelet or equivalent stating: "
                "'AVOID AMINOGLYCOSIDE ANTIBIOTICS — mtDNA m.1555A>G carrier — risk of permanent deafness'. "
                "Emergency department electronic flag if possible."
            ),
        },
        "key_references": [
            "Prezant TR et al. (1993) Mitochondrial ribosomal RNA mutation associated with both antibiotic-induced and non-syndromic deafness. Nat Genet 4(3):289-94 — DISCOVERY of m.1555A>G",
            "Hutchin TP et al. (1993) A molecular basis for human hypersensitivity to aminoglycoside antibiotics. Nucleic Acids Res 21(18):4174-9 — mechanism m.1555A>G",
            "Bitner-Glindzicz M et al. (2009) Prevalence of mitochondrial 1555A→G mutation in European children. N Engl J Med 360(6):640-2 — ~1 in 500 population prevalence",
            "Zhao H et al. (2004) Maternally inherited aminoglycoside-induced and nonsyndromic deafness is associated with the novel C1494T mutation. Am J Hum Genet 74(1):139-52 — m.1494C>T discovery",
            "Casano RA et al. (1999) Inherited susceptibility to aminoglycoside ototoxicity: genetic heterogeneity and role of the 1555A→G mutation. Am J Med Genet 89(3):167-71 — nuclear modifiers",
            "Ramos A et al. (2013) Population variation and mutation spectra of the human mt 12S rRNA. Mitochondrion 13(6):822-30 — comprehensive variant spectrum",
        ],
        "terms": [
            {
                "term": "MT-RNR1 (12S rRNA)",
                "definition": (
                    "Mitochondrially encoded 12S ribosomal RNA — 954 nt RNA gene on H-strand (rCRS 648–1601); "
                    "forms the mitoribosome small subunit (mt-SSU / 28S); NOT translated into protein; "
                    "helix 44 contains the A-site decoding loop; m.1555A>G and m.1494C>T in helix 44 cause "
                    "aminoglycoside hypersensitivity; OMIM *561000"
                ),
            },
            {
                "term": "Aminoglycoside-Induced SNHL (AISNHL)",
                "definition": (
                    "Severe-to-profound bilateral sensorineural hearing loss occurring within 24–72 hours "
                    "of aminoglycoside exposure in m.1555A>G or m.1494C>T carriers. Cochlear outer hair cells "
                    "(OHCs) of the basal turn (high-frequency) are preferentially destroyed. NO recovery — "
                    "permanent. Cochlear implants are the main rehabilitation option."
                ),
            },
            {
                "term": "Helix 44 (MT-RNR1 A-site decoding loop)",
                "definition": (
                    "The critical structural element of 12S rRNA at rCRS ~1481–1571 that forms the "
                    "codon-anticodon decoding center of the mt-SSU. m.1555A>G (rCRS 1555) and m.1494C>T (rCRS 1494) "
                    "are both in helix 44. These variants make the mt-12S rRNA decoding loop resemble prokaryotic "
                    "16S rRNA A-site → aminoglycosides bind → mt-ribosome stalls → SNHL."
                ),
            },
            {
                "term": "Homoplasmy (MT-RNR1 context)",
                "definition": (
                    "m.1555A>G and m.1494C>T are HOMOPLASMIC — all mtDNA copies carry the variant "
                    "(unlike protein-coding mtDNA genes where heteroplasmy is common). "
                    "This means blood DNA is reliable for diagnosis (no muscle needed) and ALL "
                    "maternal relatives will inherit the same variant."
                ),
            },
            {
                "term": "TRMU nuclear modifier",
                "definition": (
                    "TRMU (tRNA 5-methylaminomethyl-2-thiouridylate methyltransferase) is a nuclear gene "
                    "that modifies the wobble position of mt-tRNAs. TRMU variants reduce mt-tRNA stability "
                    "and compound the MT-RNR1 translation defect → increased penetrance of NSHL without "
                    "aminoglycosides. Explains why only ~20–30% of m.1555A>G carriers develop NSHL spontaneously."
                ),
            },
            {
                "term": "Newborn Screening for m.1555A>G",
                "definition": (
                    "Population-level screening of all newborns for m.1555A>G by blood spot PCR-RFLP or "
                    "next-generation sequencing; implemented in the UK (NHS), China, and several European countries. "
                    "Goal: identify carriers BEFORE any aminoglycoside exposure. "
                    "Cost-effective: prevents permanent deafness by ensuring aminoglycoside avoidance."
                ),
            },
        ],
    }


if __name__ == "__main__":
    ov = get_overview()
    print(f"Gene: {ov['gene']} ({ov['alias']})")
    print(f"Disease: {ov['disease_name']}")
    print(f"OMIM Gene: *{ov['omim_gene']}")
    print(f"Genome: {ov['chromosome']}")
    print(f"Inheritance: {ov['inheritance']}")
    print(f"\nCohort: {ov['cohort_n']} patients, seed {ov['seed']}")
    s = ov["cohort_statistics"]
    print(f"  Severe/Profound SNHL: {s['severe_profound_pct']}%")
    print(f"  Aminoglycoside-exposed: {s['aminoglycoside_exposed_pct']}%")
    print(f"  Cochlear implant: {s['cochlear_implant_pct']}%")
    print(f"  Asymptomatic carrier: {s['asymptomatic_carrier_pct']}%")
    print(f"  Tinnitus: {s['tinnitus_pct']}%")
    print(f"  Bilateral SNHL: {s['bilateral_pct']}%")
    print(f"  Maternal family SNHL: {s['maternal_family_snhl_pct']}%")
    print(f"  NO OXPHOS deficiency: {s['no_oxphos_deficiency_pct']}%")
    print(f"  m.1555A>G (main): {s['m1555_pct']}%")
    print("\nVariants:", [v["change"] for v in VARIANTS])
