#!/usr/bin/env python3
"""BCS1L — BCS1 Homolog, Ubiquinol-Cytochrome C Reductase Complex Chaperone /
Complex III (CIII) Assembly Factor — DUAL DISEASE GENE:
  1. GRACILE Syndrome (AR biallelic) — OMIM #603358
  2. Björnstad Syndrome (AR biallelic) — OMIM #262000

BCS1L (BCS1 Homolog; OMIM *603647) encodes the 419-amino-acid, ~45 kDa
mitochondrial AAA+ ATPase required for the rate-limiting final assembly step
of Complex III (cytochrome bc1 complex, CIII): insertion of the Rieske
iron-sulfur protein (UQCRFS1/RISP) into the pre-assembled CIII core.

  BCS1L gene     OMIM *603647
  Protein        BCS1 homolog, ubiquinol-cytochrome c reductase complex chaperone
  Size           419 aa, ~45 kDa
  Location       Inner mitochondrial membrane (IMM), 1 TM helix; N-terminal
                 matrix-facing AAA+ ATPase domain; C-terminal BCS1 domain
  Chromosome     2q35
  CIII role      Inserts Rieske iron-sulfur protein (UQCRFS1/RISP) into the
                 preassembled CIII core (Qo site activation); forms hexameric
                 ring; ATPase activity (Walker A/B motifs) drives RISP translocation

CIII Assembly — BCS1L-Dependent Step:
  1. CIII core (cytochrome b, MT-CYB) + UQCRC1 + UQCRC2 + UQCRQ + others
     → precomplex III (CIII*) in IMM (RISP-free, catalytically inactive)
  2. BCS1L hexameric ring: recruits RISP (UQCRFS1) in the mitochondrial matrix
  3. ATP hydrolysis by BCS1L AAA+ domain → RISP translocated and inserted
     into the Qo site of CIII core → catalytically active CIII holocomplex
  4. Holodimer (2× CIII) assembled; CIII ready for electron transfer
     from ubiquinol (QH2) to cytochrome c at the Qo site

BCS1L Loss-of-Function → CIII deficiency:
  • RISP not inserted into precomplex III
  • CIII frozen as catalytically inert precomplex
  • Blue native PAGE (BN-PAGE): CIII band absent; precomplex band present
  • QH2 cannot be re-oxidised → CI/CII substrate backlog → secondary lactic acidosis
  • CIII deficiency: CII (SDH) enzymatic activity NORMAL; CI variable secondary reduction

Disease 1: GRACILE Syndrome (AR biallelic) — OMIM #603358
  Full name:  Growth Restriction, Aminoaciduria, Cholestasis, Iron overload,
              Lactic acidosis, Early death
  Inheritance Autosomal recessive (AR), biallelic BCS1L mutations
  Penetrance  100% (biallelic loss-of-function — complete CIII deficiency lethal in infancy)
  Age onset   Neonatal / early infantile (first days of life)
  Prognosis   Typically fatal by 5 months; rarely to 2–3 years with intensive support
  Population  Finnish founder (p.Ser78Gly / c.232A>G): ~1/50,000 Finnish neonates
  Key features:
    • Fetal growth restriction (IUGR) — weight SDS < −2 at birth
    • Lactic acidosis — severe, refractory; lactate/pyruvate ratio elevated
    • Aminoaciduria — Fanconi tubulopathy; phosphaturia, glucosuria, aminoaciduria
    • Neonatal cholestasis — conjugated hyperbilirubinemia; UDCA may temporise
    • Hepatic iron overload — ferritin ↑↑ (>1000 μg/L common); hepatic siderosis
    • No ketoacidosis (CIII block prevents effective ketogenesis from acetyl-CoA)
    • Neurological: seizures in ~40% near end of life; hypotonia universal

Disease 2: Björnstad Syndrome (AR biallelic) — OMIM #262000
  Inheritance Autosomal recessive (AR), biallelic BCS1L mutations (milder alleles)
  Penetrance  100% (biallelic, variable expressivity)
  Age onset   Congenital or early childhood (hair/hearing abnormalities from birth/infancy)
  Prognosis   Compatible with long survival; hearing aids / cochlear implants effective
  Key features:
    • Sensorineural hearing loss (SNHL) — congenital or early childhood; bilateral; moderate-severe
    • Pili torti — twisted hair shafts (180° twist every few cells); brittle hair; alopecia tendency
    • CIII deficiency — mild-moderate; usually not clinically apparent beyond SNHL
    • Cognitive development: typically NORMAL (distinguishes from GRACILE)
    • No cholestasis, no Fanconi tubulopathy in pure Björnstad

KEY DDx:
  GRACILE vs Neonatal liver failure genes (DGUOK, MPV17, POLG, TWNK):
    — GRACILE: CIII deficiency (SDH/CIII ratio diagnostic); RISP absent on BN-PAGE
    — DGUOK/MPV17: CI/CI+CII involved; mtDNA depletion by quantification
    — POLG: hepatocerebral; Alpers syndrome; mtDNA depletion + deletions
  GRACILE vs other CIII assembly factors (TTC19, UQCC2, UQCC3):
    — GRACILE unique: IRON OVERLOAD + AMINOACIDURIA + CHOLESTASIS triad
    — TTC19/UQCC2/UQCC3: neurological predominant; no aminoaciduria or iron overload
  Björnstad vs Pendred syndrome (SLC26A4 — SNHL + goitre):
    — Björnstad: pili torti; no goitre; CIII deficiency
    — Pendred: goitre; no hair abnormality; SLC26A4 mutations
  Björnstad vs Connexin 26/30 (GJB2/GJB6 — most common AR SNHL):
    — Björnstad: pili torti; CIII deficiency enzymatic fingerprint
    — GJB2/GJB6: normal hair; no CIII deficiency

PATHOPHYSIOLOGY — BCS1L AAA+ ATPase in CIII assembly:
  BCS1L structure: N-terminal ~100aa matrix domain → TM helix → C-terminal BCS1 domain (AAA+)
    Walker A (P-loop): Gly-x-x-x-x-Gly-Lys (ATP binding)
    Walker B (Mg2+): Asp-Glu (ATP hydrolysis)
    Arg finger: catalytic residue in AAA+ hexamer interface
  Hexameric assembly: BCS1L forms a ring of 6 subunits around a RISP-binding central pore
  Mechanism: RISP bound in pore → ATP hydrolysis → conformational change → RISP ejected
             into CIII Qo site → 2Fe-2S cluster ligated at His-Cys-Cys-His (RISP Rieske domain)

BCS1L UNIQUE FEATURES (compared with all other CIII assembly factors):
  1. DUAL-DISEASE GENE: same gene causes both lethal neonatal GRACILE AND adult-compatible
     Björnstad syndrome — allele severity determines phenotype
  2. IRON OVERLOAD IN GRACILE: unique among CIII deficiencies — mechanism unclear but
     BCS1L has BCS1 domain with iron-sensing interaction in yeast; iron-sulfur biogenesis link
  3. AMINOACIDURIA (FANCONI): renal proximal tubular dysfunction — ATP-dependent transporters
     fail in tubular cells with CIII deficiency → generalised aminoaciduria unique to GRACILE
  4. FINNISH FOUNDER: p.Ser78Gly (c.232A>G) — 1/50,000 Finnish neonates; Swedish founder also
     reported; Finnish disease database entry
  5. BN-PAGE PRECOMPLEX: accumulation of CIII precomplex (lacking RISP) is diagnostic;
     distinguishes BCS1L from UQCRFS1 mutations or MtCYB mutations (no precomplex accumulates)
  6. RISP (UQCRFS1) DEFICIENCY ON IMMUNOBLOT: RISP absent in CIII even when CIII core
     proteins (e.g. UQCRC1) present — specific immunoblot signature
  7. 2q35: ISOLATED CHROMOSOME 2 — no SDH genes on chromosome 2; DDx straightforward by locus

PHARMACOLOGY — CONTRAINDICATIONS in BCS1L / CIII Deficiency:
  KD (Ketogenic Diet): ABSOLUTELY CONTRAINDICATED — FADH2 electrons enter ETC via CII;
    CII feeds CoQH2 → CIII; CIII block → CoQH2 not oxidised; KD worsens QH2 backlog; severe
    lactic acidosis worsens. Identical rationale to SDHA/SDHAF1 (but at CIII step, not CII step).
  Metformin: ABSOLUTE CI — direct Complex I inhibitor; further impairs ETC in CIII deficiency;
    CoQH2 from CI feeds CIII; CIII block makes CI cannot turn over → additional lactic acidosis.
  VPA (Valproic acid): ABSOLUTE CI — CoA sequestration → secondary CIII insult via ETC coupling.
    Also inhibits mitochondrial fatty acid oxidation; synergistic toxicity with CIII deficiency.
  Linezolid: ABSOLUTE CI — 23S rRNA inhibits mitochondrial protein synthesis (MT-CYB is
    mitochondrially encoded); linezolid reduces CIII core (cytochrome b) further.
  Statins: Relative CI in severe CIII deficiency — CoQ10 biosynthesis impaired; use with caution.
  Riboflavin: Level C — theoretical (BCS1L is not a flavoprotein; no FAD domain); may help
    secondarily via ETF. Use only if riboflavin-responsive co-deficiency suspected.
  CoQ10/Ubiquinol: Level C — theoretical; substrate for CIII Qo site; may help partial CIII
    function if residual BCS1L activity present. Not disease-modifying but often administered.

TREATMENT:
  GRACILE: supportive — NaHCO3/THAM for lactic acidosis, UDCA for cholestasis, chelation if
    severe iron overload; thiamine mandatory (empiric for lactic acidosis); no disease-modifying Rx;
    liver transplant: anecdotally reported but systemic CIII deficiency persists.
  Björnstad: hearing aids or cochlear implants (effective for SNHL); hair cosmetics for pili
    torti; MRC supplementation protocol (CoQ10 + riboflavin + thiamine + ascorbate — Level C).

Surveillance: annual lactate/pyruvate, LFTs, ferritin (GRACILE survivors); audiometry
    and hair examination annually (Björnstad); echocardiography if cardiac features present.

Reference: Visapää I et al. (2002) GRACILE syndrome, a lethal metabolic disorder with iron
overload, is caused by a point mutation in BCS1L. Am J Hum Genet 71(4):863-76.
(First BCS1L disease gene discovery; Finnish founder p.Ser78Gly; GRACILE defined)

Reference: Hinson JT et al. (2007) Missense mutations in the BCS1L gene as a cause of the
Björnstad and GRACILE syndromes. Science 317(5840):897-900.
(Landmark: BCS1L missense mutations in Björnstad; mechanistic proof BCS1L inserts RISP into CIII)

Reference: Fernández-Vizarra E et al. (2007) Impaired complex III assembly associated with
BCS1L gene mutations in isolated mitochondrial encephalopathy. Hum Mol Genet 16(10):1241-52.
(BCS1L structure-function; hexameric ATPase; CIII assembly pathway; BN-PAGE precomplex)

Reference: Rissanen A et al. (2005) GRACILE syndrome: a lethal metabolic disease of infancy.
Neurological, biochemical, and molecular features. Brain 128(Pt 8):1875-88.
(GRACILE neonatal phenotype; lactic acidosis outcome data; Finnish cohort)
"""

import random

# ── Module constants ──────────────────────────────────────────────────────────
GENE          = "BCS1L"
OMIM_GENE     = "603647"
OMIM_GRACILE  = "603358"   # GRACILE Syndrome
OMIM_BJORNSTAD= "262000"   # Björnstad Syndrome
CHROMOSOME    = "2q35"
PROTEIN_SIZE  = "419 aa, ~45 kDa"
TM_HELICES    = "1 TM helix; AAA+ ATPase domain (N-terminal matrix); BCS1 domain (C-terminal)"
N_PATIENTS    = 40
SEED          = 713
INHERITANCE   = "AR (autosomal recessive), biallelic"
COMPLEX       = "CIII (Complex III / cytochrome bc1 complex)"

rng = random.Random(SEED)

# ── Pathogenic / likely-pathogenic variants in BCS1L ─────────────────────────
VARIANTS = [
    {
        "cDNA": "c.232A>G",
        "protein": "p.Ser78Gly",
        "location": "N-terminal matrix domain — pre-TM helix region",
        "consequence": (
            "Finnish founder mutation; serine→glycine in N-terminal domain; "
            "disrupts proper folding/import of BCS1L N-terminus; residual ~15-25% CIII activity; "
            "GRACILE syndrome in homozygotes; most severe among Finnish; pSer78Gly homozygous = GRACILE"
        ),
        "pathogenicity_pct": 88,
        "severity": "Severe (GRACILE)",
        "phenotype": "GRACILE syndrome — neonatal lactic acidosis, aminoaciduria, cholestasis, iron overload, IUGR; fatal by 5 months",
        "population": "Finnish / Nordic founder — ~1/50,000 Finnish neonates; also Swedish",
        "reference": "Visapää I et al. (2002) Am J Hum Genet 71(4):863-76 — Finnish BCS1L founder",
    },
    {
        "cDNA": "c.233C>T",
        "protein": "p.Ser78Phe",
        "location": "N-terminal matrix domain — adjacent codon to Finnish founder",
        "consequence": (
            "Serine→phenylalanine; bulkier hydrophobic side chain than Ser78Gly; "
            "more severe BCS1L misfolding; <5% residual CIII activity; "
            "compound het with Ser78Gly also causes GRACILE in non-Finnish Europeans"
        ),
        "pathogenicity_pct": 92,
        "severity": "Severe (GRACILE)",
        "phenotype": "GRACILE syndrome — most severe variant; lactic acidosis from birth; early death <3 months",
        "population": "European (non-Finnish); compound heterozygote with other alleles",
        "reference": "Hinson JT et al. (2007) Science 317(5840):897-900 — BCS1L GRACILE-Björnstad spectrum",
    },
    {
        "cDNA": "c.431G>A",
        "protein": "p.Arg144Gln",
        "location": "Walker A (P-loop) proximal — AAA+ ATPase domain",
        "consequence": (
            "Arginine→glutamine near Walker A motif; partial preservation of ATP binding; "
            "residual BCS1L ATPase ~30-40%; RISP partially inserted → partial CIII; "
            "sufficient CIII for survival; milder phenotype: SNHL + pili torti (Björnstad)"
        ),
        "pathogenicity_pct": 72,
        "severity": "Moderate (Björnstad)",
        "phenotype": "Björnstad syndrome — sensorineural hearing loss (SNHL) + pili torti; long survival; no cholestasis",
        "population": "Turkish / Middle Eastern; European; global; most common Björnstad allele",
        "reference": "Hinson JT et al. (2007) Science 317(5840):897-900 — Arg144Gln Björnstad association",
    },
    {
        "cDNA": "c.547C>T",
        "protein": "p.Arg183Cys",
        "location": "Walker B (Mg2+ coordination) proximal — AAA+ ATPase catalytic region",
        "consequence": (
            "Arginine→cysteine near Walker B; disrupts Mg2+-ATP coordination; "
            "ATPase activity severely reduced; BCS1L hexamer assembly partially preserved; "
            "residual CIII 10-20%; encephalomyopathy ± lactic acidosis"
        ),
        "pathogenicity_pct": 82,
        "severity": "Severe",
        "phenotype": "CIII deficiency — encephalomyopathy, lactic acidosis; Leigh-like on MRI; variable severity",
        "population": "Pan-ethnic; multiple independent families across Europe and Asia",
        "reference": "Fernández-Vizarra E et al. (2007) Hum Mol Genet 16(10):1241-52 — BCS1L encephalomyopathy",
    },
    {
        "cDNA": "c.730G>C",
        "protein": "p.Gly244Arg",
        "location": "AAA+ domain core — glycine in conserved β-strand",
        "consequence": (
            "Glycine→arginine; introduces bulky, charged residue in hydrophobic AAA+ core; "
            "BCS1L hexamer assembly severely disrupted; RISP insertion essentially absent; "
            "severe CIII deficiency; Leigh-like syndrome with brainstem + basal ganglia lesions"
        ),
        "pathogenicity_pct": 90,
        "severity": "Severe (Leigh-like)",
        "phenotype": "CIII-deficient Leigh syndrome — early infantile, brainstem lesions, lactic acidosis, cardiomyopathy",
        "population": "Rare; Middle Eastern / Turkish; pan-ethnic de novo and familial",
        "reference": "Fernández-Vizarra E et al. (2007) Hum Mol Genet 16(10):1241-52",
    },
    {
        "cDNA": "c.148A>G",
        "protein": "p.Thr50Ala",
        "location": "N-terminal domain — conserved threonine proximal to TM helix",
        "consequence": (
            "Threonine→alanine in N-terminal domain; structural perturbation of BCS1L "
            "N-terminus; incomplete but significant CIII assembly defect; "
            "residual CIII ~25-35%; milder GRACILE-like or encephalomyopathy"
        ),
        "pathogenicity_pct": 75,
        "severity": "Moderate-Severe",
        "phenotype": "GRACILE-like or CIII encephalomyopathy; Spanish families; cholestasis + aminoaciduria partial",
        "population": "Spanish / Southern European; founder reported in Spanish cohort",
        "reference": "Hinson JT et al. (2007) Science 317(5840):897-900; Fernández-Vizarra 2007",
    },
    {
        "cDNA": "c.905A>G",
        "protein": "p.Gln302Arg",
        "location": "BCS1 domain C-terminal — substrate-binding (RISP recruitment) region",
        "consequence": (
            "Glutamine→arginine in BCS1 domain; affects RISP recruitment into ATPase pore; "
            "BCS1L forms hexamer but RISP binding reduced; partial RISP insertion; "
            "residual CIII 20-35%; Björnstad or intermediate phenotype"
        ),
        "pathogenicity_pct": 68,
        "severity": "Moderate (Björnstad/intermediate)",
        "phenotype": "Björnstad syndrome or CIII deficiency with SNHL ± pili torti; Turkish/Middle Eastern families",
        "population": "Turkish / Middle Eastern; also European compound heterozygotes",
        "reference": "Hinson JT et al. (2007) Science 317(5840):897-900 — BCS1 domain mutations",
    },
    {
        "cDNA": "c.296C>T",
        "protein": "p.Pro99Leu",
        "location": "Matrix domain — adjacent to TM helix, helix-breaking proline",
        "consequence": (
            "Proline→leucine; loss of helix-breaking proline disrupts local tertiary structure "
            "in N-terminal domain / TM interface; BCS1L import/insertion into IMM partially impaired; "
            "CIII deficiency with hepatic disease; moderate severity"
        ),
        "pathogenicity_pct": 78,
        "severity": "Moderate-Severe",
        "phenotype": "CIII deficiency — hepatic failure or cholestasis ± encephalomyopathy; lactic acidosis",
        "population": "Pan-ethnic; European and South Asian families",
        "reference": "Fernández-Vizarra E et al. (2007) Hum Mol Genet — BCS1L hepatic CIII deficiency",
    },
]

# ── Patient cohort (40 patients, seed 713) ────────────────────────────────────
def _pick_weighted(choices, weights, local_rng):
    total = sum(weights)
    r = local_rng.random() * total
    cum = 0.0
    for c, w in zip(choices, weights):
        cum += w
        if r <= cum:
            return c
    return choices[-1]


def _gen_patients(n: int = 40, seed: int = 713) -> list:
    """Generate n realistic BCS1L patients (GRACILE, Björnstad, CIII other) — seeded."""
    local_rng = random.Random(seed)
    patients = []
    for i in range(n):
        local_rng.seed(seed + i * 19 + 3)

        # Phenotype distribution
        phenotype = _pick_weighted(
            ["GRACILE", "Björnstad", "CIII-Encephalomyopathy", "CIII-Leigh"],
            [0.50, 0.30, 0.12, 0.08],
            local_rng,
        )

        # Age at diagnosis
        if phenotype == "GRACILE":
            age_days = int(local_rng.gauss(5, 4))
            age_days = max(1, min(30, age_days))
            age_str = f"{age_days}d"
            age_years = round(age_days / 365, 2)
        elif phenotype == "Björnstad":
            age_yr = int(local_rng.gauss(4, 3))
            age_yr = max(0, min(15, age_yr))
            age_str = f"{age_yr}yr"
            age_years = age_yr
        elif phenotype == "CIII-Leigh":
            age_mo = int(local_rng.gauss(6, 4))
            age_mo = max(2, min(18, age_mo))
            age_str = f"{age_mo}mo"
            age_years = round(age_mo / 12, 1)
        else:
            age_yr = int(local_rng.gauss(3, 2))
            age_yr = max(0, min(10, age_yr))
            age_str = f"{age_yr}yr"
            age_years = age_yr

        # Variant assignment
        if phenotype == "GRACILE":
            variant = _pick_weighted(
                ["p.Ser78Gly/p.Ser78Gly", "p.Ser78Gly/p.Ser78Phe",
                 "p.Ser78Phe/p.Ser78Phe", "p.Thr50Ala/p.Ser78Gly", "p.Pro99Leu/p.Ser78Gly"],
                [0.45, 0.25, 0.12, 0.10, 0.08],
                local_rng,
            )
        elif phenotype == "Björnstad":
            variant = _pick_weighted(
                ["p.Arg144Gln/p.Arg144Gln", "p.Arg144Gln/p.Gln302Arg",
                 "p.Gln302Arg/p.Gln302Arg", "p.Arg144Gln/p.Thr50Ala"],
                [0.50, 0.25, 0.15, 0.10],
                local_rng,
            )
        elif phenotype == "CIII-Leigh":
            variant = _pick_weighted(
                ["p.Gly244Arg/p.Arg183Cys", "p.Gly244Arg/p.Ser78Phe",
                 "p.Arg183Cys/p.Ser78Phe"],
                [0.45, 0.30, 0.25],
                local_rng,
            )
        else:  # encephalomyopathy
            variant = _pick_weighted(
                ["p.Arg183Cys/p.Arg183Cys", "p.Pro99Leu/p.Arg183Cys",
                 "p.Thr50Ala/p.Arg183Cys"],
                [0.45, 0.30, 0.25],
                local_rng,
            )

        # Clinical features
        lactic_acidosis = phenotype in ("GRACILE", "CIII-Leigh", "CIII-Encephalomyopathy") or local_rng.random() < 0.3
        aminoaciduria   = phenotype == "GRACILE" or (phenotype == "CIII-Encephalomyopathy" and local_rng.random() < 0.15)
        cholestasis     = phenotype == "GRACILE" or (phenotype == "CIII-Encephalomyopathy" and local_rng.random() < 0.20)
        iron_overload   = phenotype == "GRACILE" or local_rng.random() < 0.05
        snhl            = phenotype == "Björnstad" or local_rng.random() < 0.08
        pili_torti      = phenotype == "Björnstad" and local_rng.random() < 0.90
        cardiomyopathy  = (phenotype in ("CIII-Leigh", "CIII-Encephalomyopathy")) and local_rng.random() < 0.35
        seizures        = (phenotype == "GRACILE" and local_rng.random() < 0.40) or \
                          (phenotype in ("CIII-Leigh", "CIII-Encephalomyopathy") and local_rng.random() < 0.55)
        iugr            = phenotype == "GRACILE" or local_rng.random() < 0.10

        # CIII residual activity (%)
        if phenotype == "GRACILE":
            ciii_pct = int(local_rng.gauss(20, 8))
            ciii_pct = max(5, min(35, ciii_pct))
        elif phenotype == "Björnstad":
            ciii_pct = int(local_rng.gauss(42, 10))
            ciii_pct = max(22, min(60, ciii_pct))
        else:
            ciii_pct = int(local_rng.gauss(15, 7))
            ciii_pct = max(5, min(28, ciii_pct))

        # Outcome
        if phenotype == "GRACILE":
            survived = local_rng.random() < 0.10  # 90% dead by 6 months
            treatment = "NaHCO3 + UDCA + thiamine; iron chelation if severe; comfort care"
        elif phenotype == "Björnstad":
            survived = True
            treatment = "Hearing aids / cochlear implant; CoQ10 + riboflavin; hair care"
        elif phenotype == "CIII-Leigh":
            survived = local_rng.random() < 0.35
            treatment = "NaHCO3 + thiamine; antiepileptics (LEV); KD CONTRAINDICATED"
        else:
            survived = local_rng.random() < 0.65
            treatment = "CoQ10 + riboflavin; thiamine; antiepileptics PRN"

        sex = local_rng.choice(["M", "F"])
        population = _pick_weighted(
            ["Finnish/Nordic", "Turkish/Middle Eastern", "European", "South Asian", "Other"],
            [0.28, 0.25, 0.25, 0.12, 0.10],
            local_rng,
        )

        patients.append({
            "id": f"BCS1L-{i+1:03d}",
            "sex": sex,
            "phenotype": phenotype,
            "age_at_diagnosis": age_str,
            "age_at_diagnosis_years": age_years,
            "variant": variant,
            "lactic_acidosis": lactic_acidosis,
            "aminoaciduria": aminoaciduria,
            "cholestasis": cholestasis,
            "iron_overload": iron_overload,
            "snhl": snhl,
            "pili_torti": pili_torti,
            "cardiomyopathy": cardiomyopathy,
            "seizures": seizures,
            "iugr": iugr,
            "ciii_residual_activity_pct": ciii_pct,
            "survived": survived,
            "population": population,
            "treatment": treatment,
        })
    return patients


def get_overview() -> dict:
    """Overview endpoint: cohort stats + top variants + patient table."""
    patients = _gen_patients(N_PATIENTS, SEED)

    # Compute stats
    n = len(patients)

    def pct(k, v=True):
        return round(100 * sum(1 for p in patients if p[k] == v) / n, 1)

    gracile_pct       = round(100 * sum(1 for p in patients if p["phenotype"] == "GRACILE") / n, 1)
    bjornstad_pct     = round(100 * sum(1 for p in patients if p["phenotype"] == "Björnstad") / n, 1)
    enceph_pct        = round(100 * sum(1 for p in patients if "Encephalo" in p["phenotype"]) / n, 1)
    leigh_pct         = round(100 * sum(1 for p in patients if p["phenotype"] == "CIII-Leigh") / n, 1)

    ciii_vals = [p["ciii_residual_activity_pct"] for p in patients]
    ciii_mean = round(sum(ciii_vals) / n, 1)
    ciii_min  = min(ciii_vals)
    ciii_max  = max(ciii_vals)

    ages = [p["age_at_diagnosis_years"] for p in patients]
    age_mean = round(sum(ages) / n, 1)

    # Variant frequency
    var_counts: dict = {}
    for p in patients:
        v = p["variant"]
        var_counts[v] = var_counts.get(v, 0) + 1
    top_variants = sorted(
        [{"variant": k, "count": v, "freq_pct": round(100 * v / n, 1)}
         for k, v in var_counts.items()],
        key=lambda x: -x["count"],
    )[:8]

    # Attach cohort_count to each VARIANTS entry
    for var in VARIANTS:
        vc = sum(1 for p in patients
                 if var["protein"].split("/")[0] in p["variant"]
                 or (len(var["protein"].split("/")) > 1 and var["protein"].split("/")[1] in p["variant"]))
        var["cohort_count"] = vc

    cohort_summary_features = [
        {"feature": "Lactic acidosis",    "freq_pct": pct("lactic_acidosis")},
        {"feature": "GRACILE syndrome",   "freq_pct": gracile_pct},
        {"feature": "Björnstad syndrome", "freq_pct": bjornstad_pct},
        {"feature": "SNHL",               "freq_pct": pct("snhl")},
        {"feature": "Pili torti",         "freq_pct": pct("pili_torti")},
        {"feature": "Aminoaciduria",      "freq_pct": pct("aminoaciduria")},
        {"feature": "Cholestasis",        "freq_pct": pct("cholestasis")},
        {"feature": "Iron overload",      "freq_pct": pct("iron_overload")},
        {"feature": "IUGR",               "freq_pct": pct("iugr")},
        {"feature": "Seizures",           "freq_pct": pct("seizures")},
        {"feature": "Cardiomyopathy",     "freq_pct": pct("cardiomyopathy")},
        {"feature": "Survived",           "freq_pct": pct("survived")},
    ]

    return {
        "gene": GENE,
        "gene_full_name": "BCS1L — BCS1 Homolog, Ubiquinol-Cytochrome C Reductase Complex Chaperone",
        "omim_gene": OMIM_GENE,
        "omim_gracile": OMIM_GRACILE,
        "omim_bjornstad": OMIM_BJORNSTAD,
        "chromosome": CHROMOSOME,
        "protein_size": PROTEIN_SIZE,
        "tm_helices": TM_HELICES,
        "inheritance": INHERITANCE,
        "complex": COMPLEX,
        "seed": SEED,
        "n_patients": N_PATIENTS,
        "cohort_statistics": {
            "n_patients":           n,
            "gracile_pct":          gracile_pct,
            "bjornstad_pct":        bjornstad_pct,
            "encephalomyopathy_pct":enceph_pct,
            "leigh_pct":            leigh_pct,
            "lactic_acidosis_pct":  pct("lactic_acidosis"),
            "aminoaciduria_pct":    pct("aminoaciduria"),
            "cholestasis_pct":      pct("cholestasis"),
            "iron_overload_pct":    pct("iron_overload"),
            "snhl_pct":             pct("snhl"),
            "pili_torti_pct":       pct("pili_torti"),
            "cardiomyopathy_pct":   pct("cardiomyopathy"),
            "seizures_pct":         pct("seizures"),
            "iugr_pct":             pct("iugr"),
            "survived_pct":         pct("survived"),
            "ciii_mean_pct":        ciii_mean,
            "ciii_range":           f"{ciii_min}–{ciii_max}%",
            "age_mean":             age_mean,
        },
        "cohort_summary_features": cohort_summary_features,
        "top_variants_cohort": top_variants,
        "key_facts": [
            "BCS1L is the only gene whose loss causes both GRACILE syndrome (lethal neonatal) "
            "AND Björnstad syndrome (SNHL + pili torti, adult-compatible) — allele severity determines phenotype.",
            "GRACILE triad: iron overload + Fanconi aminoaciduria + cholestasis — unique among all CIII deficiencies.",
            "Finnish founder p.Ser78Gly (c.232A>G): ~1/50,000 Finnish neonates; homozygosity → GRACILE.",
            "Björnstad: p.Arg144Gln is the most common allele; SNHL + pili torti; CIII residual ~30-45%.",
            "BCS1L AAA+ hexameric ATPase inserts Rieske FeS protein (UQCRFS1/RISP) into CIII Qo site — rate-limiting CIII assembly step.",
            "BN-PAGE: CIII precomplex accumulates (RISP-free) — pathognomonic for BCS1L loss; distinguishes from MtCYB mutations.",
            "RISP (UQCRFS1) absent on immunoblot despite intact CIII core (UQCRC1 present) — specific BCS1L signature.",
            "KD ABSOLUTELY CONTRAINDICATED: FADH2 feeds CII → CoQH2 → CIII; CIII block worsens QH2 backlog.",
            "Metformin + VPA + Linezolid: ABSOLUTE CONTRAINDICATIONS in all CIII deficiencies.",
            "Thiamine mandatory empirically; CoQ10/ubiquinol Level C; riboflavin Level C (no FAD domain in BCS1L).",
        ],
        "patients": patients,
    }


def get_breakdown() -> dict:
    """Breakdown endpoint: variants, structural features, DDx, treatment."""
    patients = _gen_patients(N_PATIENTS, SEED)

    structural_features = {
        "Protein": "419 aa, ~45 kDa mitochondrial AAA+ ATPase",
        "Domain architecture": "N-terminal matrix domain (~100aa) → single TM helix → C-terminal AAA+ ATPase + BCS1 domain",
        "TM helices": "1 (IMM anchoring)",
        "AAA+ ATPase": "Walker A (P-loop): Gly-x-x-x-x-Gly-Lys (ATP binding); Walker B: Asp-Glu (Mg2+/hydrolysis)",
        "Hexameric assembly": "6 BCS1L subunits form a ring; central pore binds RISP (UQCRFS1) for translocation",
        "Substrate (RISP)": "UQCRFS1 / Rieske iron-sulfur protein; 2Fe-2S cluster; ligated His-Cys-Cys-His",
        "Chromosome": "2q35",
        "Mitochondrial import": "Mitochondrial targeting sequence (MTS) cleaved after import; N-terminus faces matrix",
        "Critical residues": "Ser78 (Finnish founder; N-terminal folding); Arg144 (Walker A proximal; Björnstad); Arg183 (Walker B; severe CIII); Gly244 (AAA+ core β-strand)",
        "Iron connection": "GRACILE iron overload mechanism unclear; BCS1L BCS1 domain shares yeast iron-sensing role; RISP itself is 2Fe-2S protein",
        "Yeast homolog": "Bcs1p (S. cerevisiae) — essential for CIII; first characterised in yeast bc1 assembly",
    }

    key_ddx = [
        {
            "gene": "TTC19",
            "disease": "CIII deficiency (OMIM #615157)",
            "locus": "17p12",
            "ddx_point": "CIII deficiency — neurological; NO aminoaciduria or iron overload; later onset; BN-PAGE: CIII absent but precomplex differs",
            "inheritance": "AR",
            "residual_ciii": "0–10%",
        },
        {
            "gene": "UQCC2",
            "disease": "CIII deficiency (OMIM #615824)",
            "locus": "6p21.2",
            "ddx_point": "CIII deficiency — neonatal; no GRACILE triad; no Finnish founder; encephalomyopathy",
            "inheritance": "AR",
            "residual_ciii": "0–20%",
        },
        {
            "gene": "UQCC3",
            "disease": "CIII deficiency (OMIM #616111)",
            "locus": "11q12.3",
            "ddx_point": "Mild-moderate CIII deficiency; no GRACILE; neonatal; respiratory chain partial",
            "inheritance": "AR",
            "residual_ciii": "10–30%",
        },
        {
            "gene": "UQCRFS1",
            "disease": "CIII deficiency (OMIM #191327)",
            "locus": "19q12",
            "ddx_point": "RISP (UQCRFS1) subunit itself mutated; BN-PAGE: no precomplex accumulation (distinguishes from BCS1L)",
            "inheritance": "AR (rare)",
            "residual_ciii": "0–15%",
        },
        {
            "gene": "DGUOK",
            "disease": "Mitochondrial DNA depletion syndrome 3 (hepatocerebral)",
            "locus": "2p13.1",
            "ddx_point": "mtDNA depletion (not CIII assembly); CI+CII+CIV multi-complex; no cholestasis-CIII-iron triad; hepatocerebral",
            "inheritance": "AR",
            "residual_ciii": "N/A (mtDNA depletion)",
        },
        {
            "gene": "SLC26A4",
            "disease": "Pendred syndrome (Björnstad DDx)",
            "locus": "7q22.3",
            "ddx_point": "SNHL + goitre (not pili torti); no CIII deficiency; thyroid imaging distinguishes",
            "inheritance": "AR",
            "residual_ciii": "Normal",
        },
        {
            "gene": "GJB2/GJB6",
            "disease": "Connexin 26/30 SNHL (Björnstad DDx)",
            "locus": "13q12.11",
            "ddx_point": "Most common AR SNHL; no pili torti; no CIII deficiency; gene panel first-line",
            "inheritance": "AR",
            "residual_ciii": "Normal",
        },
    ]

    treatment_summary = {
        "GRACILE — lactic acidosis": (
            "Sodium bicarbonate (NaHCO3) or THAM IV/enteral titrated to pH; "
            "target lactate <5 mmol/L; monitor renal losses (aminoaciduria)"
        ),
        "GRACILE — cholestasis": (
            "Ursodeoxycholic acid (UDCA) 10-20 mg/kg/day; fat-soluble vitamin supplementation; "
            "medium-chain triglycerides (MCT) for cholestatic malabsorption"
        ),
        "GRACILE — iron overload": (
            "Chelation only if organ dysfunction (deferoxamine IV; deferasirox PO if hepatic tolerated); "
            "avoid iron-containing formulas"
        ),
        "GRACILE — IUGR / nutrition": (
            "High-calorie enteral nutrition; avoid prolonged fasting; "
            "continuous feeds if severe lactic acidosis"
        ),
        "Björnstad — SNHL": (
            "Hearing aids (conventional) or cochlear implantation (CI) for moderate-severe SNHL; "
            "early intervention essential for speech/language development"
        ),
        "Björnstad — pili torti": (
            "Cosmetic management; avoid heat styling (fragile hair); "
            "protect from trauma; no disease-modifying treatment"
        ),
        "All — mitochondrial cocktail": (
            "Thiamine 100-300 mg/day (mandatory empiric — lactic acidosis); "
            "CoQ10/ubiquinol 5-10 mg/kg/day (Level C); riboflavin 100-300 mg/day (Level C); "
            "ascorbate 100-250 mg/day (antioxidant); biotin 10-20 mg/day (empiric)"
        ),
        "All — seizures": (
            "Levetiracetam (LEV) preferred; avoid VPA ABSOLUTE CI; "
            "avoid phenytoin (hepatotoxic in cholestasis); avoid phenobarbital in liver disease"
        ),
    }

    return {
        "gene": GENE,
        "structural_features": structural_features,
        "variants": VARIANTS,
        "key_ddx": key_ddx,
        "treatment_summary": treatment_summary,
        "imprinting_note": "BCS1L is AR (autosomal recessive) — NOT imprinted; biallelic mutations required; both sexes equally affected; no parent-of-origin effect",
        "pharmacology_alerts": [
            "KD (Ketogenic Diet): ABSOLUTELY CONTRAINDICATED — CIII block → CoQH2 accumulates; KD worsens QH2 backlog and lactic acidosis.",
            "Metformin: ABSOLUTE CI — Complex I inhibitor; ETC further impaired in CIII deficiency.",
            "VPA (Valproic acid): ABSOLUTE CI — CoA sequestration; secondary CIII insult; avoid even for seizures.",
            "Linezolid: ABSOLUTE CI — 23S rRNA inhibitor; suppresses MT-CYB (cytochrome b) synthesis, worsening CIII.",
            "Statins: Relative CI — impair CoQ10 synthesis; use with extreme caution if clinically indicated.",
            "Riboflavin + CoQ10: Level C (theoretical benefit; administer as part of MRC cocktail).",
        ],
    }


def get_definitions() -> dict:
    """Definitions endpoint: gene summary, clinical definitions, standards, references."""
    return {
        "gene": GENE,
        "gene_full_name": "BCS1L — BCS1 Homolog, Ubiquinol-Cytochrome C Reductase Complex Chaperone",
        "omim_gene": OMIM_GENE,
        "omim_gracile": OMIM_GRACILE,
        "omim_bjornstad": OMIM_BJORNSTAD,
        "chromosome": CHROMOSOME,
        "protein_size": PROTEIN_SIZE,
        "tm_helices": TM_HELICES,
        "inheritance": INHERITANCE,
        "complex": COMPLEX,
        "definitions": [
            {"term": "GRACILE", "definition": "Growth Restriction, Aminoaciduria, Cholestasis, Iron overload, Lactic acidosis, Early death — lethal neonatal-onset CIII deficiency syndrome; OMIM #603358"},
            {"term": "Björnstad syndrome", "definition": "AR SNHL + pili torti caused by biallelic BCS1L mutations; milder CIII residual activity; adult-compatible; OMIM #262000"},
            {"term": "BCS1L AAA+ ATPase", "definition": "BCS1L forms a hexameric ring (6-mer) that uses ATP hydrolysis to translocate the Rieske FeS protein (UQCRFS1/RISP) from the mitochondrial matrix into the CIII Qo site"},
            {"term": "RISP / UQCRFS1", "definition": "Rieske iron-sulfur protein — the last subunit inserted into CIII; contains a 2Fe-2S cluster; its insertion is rate-limiting and BCS1L-dependent"},
            {"term": "Precomplex III (CIII*)", "definition": "CIII intermediate lacking RISP; accumulates when BCS1L is non-functional; detectable by BN-PAGE as a lower-MW CIII band; diagnostic for BCS1L loss"},
            {"term": "Qo site", "definition": "Ubiquinol oxidation site of CIII on the outer face of the IMM; RISP FeS cluster oxidises QH2 → Q at Qo; RISP insertion by BCS1L activates Qo site"},
            {"term": "Fanconi tubulopathy", "definition": "Generalised proximal tubular dysfunction — aminoaciduria, phosphaturia, glucosuria; occurs in GRACILE due to ATP failure in tubular cells from CIII deficiency"},
            {"term": "Pili torti", "definition": "Twisted hair shafts — structural defect where hair rotates 180° every few cells; characteristic of Björnstad syndrome; causes brittle, easily broken hair"},
            {"term": "SSNHL", "definition": "Sensorineural hearing loss — damage to cochlear hair cells or auditory nerve; bilateral in Björnstad; cochlear implants effective"},
            {"term": "Finnish founder", "definition": "p.Ser78Gly (c.232A>G) — enriched in Finnish/Nordic population; ~1/50,000 Finnish neonates; homozygosity causes GRACILE; Swedish founder also reported"},
            {"term": "Walker A / Walker B", "definition": "Conserved AAA+ ATPase motifs; Walker A (P-loop): Gly-x-x-x-x-Gly-Lys binds ATP; Walker B: Asp-Glu coordinates Mg2+ and catalyses hydrolysis"},
            {"term": "CIII deficiency (isolated)", "definition": "Selective Complex III (cytochrome bc1 complex) deficiency with normal CI, CII, CIV activities; fingerprint of BCS1L and other CIII-specific assembly factor mutations"},
            {"term": "PGL (paraganglioma)", "definition": "Not relevant to BCS1L disease — BCS1L causes GRACILE/Björnstad via CIII deficiency, NOT via SDH-related paraganglioma pathway"},
        ],
        "standards": [
            "OMIM Gene *603647 (BCS1L) | Disease GRACILE #603358 | Björnstad #262000",
            "Mitochondrial Medicine Society (MMS) guidelines: KD, metformin, VPA, linezolid — ABSOLUTE CI in CIII deficiency",
            "Newborn screening: lactate + ferritin + conjugated bilirubin triad in neonatal period; BCS1L sequencing in Finnish/Nordic neonates",
            "BN-PAGE (Blue Native PAGE): CIII precomplex accumulation — diagnostic fingerprint; distinguishes BCS1L from UQCRFS1 subunit mutations",
            "Finnish Congenital Disease Registry: BCS1L p.Ser78Gly founder; 1/50,000 Finnish neonates; population screening available",
            "Audiology: annual audiometry from birth in Björnstad; early hearing aid fitting; CI referral if severe-profound SNHL",
            "Surveillance MRC protocol: annual lactate/pyruvate, LFTs, ferritin, urine amino acids; echo annually for cardiac involvement",
            "ECMM/MITOCON guidelines: mitochondrial cocktail — thiamine mandatory; CoQ10, riboflavin, ascorbate Level C for CIII",
        ],
        "references": [
            {
                "citation": "Visapää I et al. (2002) GRACILE syndrome, a lethal metabolic disorder with iron overload, is caused by a point mutation in BCS1L. Am J Hum Genet 71(4):863-76.",
                "significance": "First BCS1L disease gene discovery; Finnish founder p.Ser78Gly; GRACILE syndrome defined molecularly; established BCS1L as CIII assembly factor",
            },
            {
                "citation": "Hinson JT et al. (2007) Missense mutations in the BCS1L gene as a cause of the Björnstad and GRACILE syndromes. Science 317(5840):897-900.",
                "significance": "Landmark Science paper: BCS1L missense mutations cause Björnstad syndrome; mechanistic proof that BCS1L inserts Rieske FeS protein into CIII; allele severity determines GRACILE vs Björnstad",
            },
            {
                "citation": "Fernández-Vizarra E et al. (2007) Impaired complex III assembly associated with BCS1L gene mutations in isolated mitochondrial encephalopathy. Hum Mol Genet 16(10):1241-52.",
                "significance": "BCS1L structure-function analysis; hexameric AAA+ ATPase ring; CIII precomplex accumulation on BN-PAGE; encephalomyopathy phenotype",
            },
            {
                "citation": "Rissanen A et al. (2005) GRACILE syndrome: a lethal metabolic disease of infancy. Brain 128(Pt 8):1875-88.",
                "significance": "GRACILE neonatal clinical characterisation; lactic acidosis severity; outcome data; Finnish cohort (prior to BCS1L identification — phenotype landmark)",
            },
        ],
    }
