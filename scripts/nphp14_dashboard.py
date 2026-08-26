"""
Nephronophthisis Type 14 / Joubert Syndrome 19
===============================================
Primary Gene : ZNF423 (*604085) — 16q12.1; 1,284 aa; 30 C2H2-type Krüppel zinc fingers
               in 4 clusters; nuclear transcriptional regulator + DNA damage response (DDR)
               protein; also known as EBFAZ / OAZ / ROAZ / ZNF467p
Disease OMIM : #614844 (Nephronophthisis 14 — NPHP14; renal ciliopathy ± JBTS19)
               Also: #614844 (Joubert Syndrome 19 — JBTS19; same OMIM entry, NPHP14 ± MTS)
Chromosome   : 16q12.1
Inheritance  : Autosomal Recessive (biallelic LOF — truncating and/or hypomorphic)
Prevalence   : ~1/700,000–1,500,000; ~60–80 published families as of 2026

Protein Structure — ZNF423 (1,284 aa; nuclear zinc finger / DDR protein)
-------------------------------------------------------------------------
  • N-terminal disordered region (aa 1–90): low-complexity; nuclear localisation signals;
    PARP1 interaction interface (DDR coupling)
  • Zinc finger cluster 1 (aa 91–250): 6 C2H2 zinc fingers; Smad2/3 interaction domain
    (BMP/TGF-β signalling integration); EBF1 interaction for B-cell development
  • Zinc finger cluster 2 (aa 251–500): 7 C2H2 zinc fingers; ROR2-interaction; SMAD4 binding;
    transcriptional activation domain
  • Central regulatory region (aa 501–700): disordered linker; RAR/RXR nuclear hormone
    receptor interaction; p300/CBP co-activator; important for ciliary gene transcription
  • Zinc finger cluster 3 (aa 701–950): 9 C2H2 zinc fingers; CEP290 direct interaction domain;
    centrosomal/TZ targeting (loss disrupts ciliogenesis)
  • Zinc finger cluster 4 (aa 951–1,284): 8 C2H2 zinc fingers; ATM substrate; DDR coupling
    (phospho-S/TQ sites); BBS10/BBS12 transcription activation domain

Molecular Mechanism
-------------------
ZNF423 is UNIQUE among all NPHP genes: it is primarily a NUCLEAR TRANSCRIPTIONAL REGULATOR
that promotes ciliogenesis via the DNA damage response (DDR) pathway:
  1. In G0 (quiescent cells), ZNF423 is recruited to centrosomes via its CEP290-binding domain
     (ZF cluster 3); it resolves centrosomal DNA damage via ATM-PARP1 signalling cascade
  2. Unresolved centrosomal DNA damage in ZNF423-null cells → persistent γH2AX foci at
     centrosomes → ciliary axoneme nucleation failure → NPHP14
  3. ZNF423 transcriptionally activates BBS10 and BBS12 (BBSome peripheral subunits) via
     ZF cluster 4; ZNF423 loss → BBS10/BBS12 under-expression → partial BBSome dysfunction
     contributing to ciliopathy (mechanistic link to BBS spectrum)
  4. ZNF423 interacts directly with CEP290 (NPHP6 gene) at the centrosome/transition zone;
     ZNF423-CEP290 co-dependency means biallelic ZNF423 loss partially phenocopies CEP290 LOF
     → explains Joubert MTS in 40–50% (JBTS19 alleles), parallel to NPHP6 Joubert overlap
  5. ALLELE SPECTRUM — truncating vs hypomorphic determines JBTS19 vs pure NPHP14:
     • Biallelic truncating / null alleles: JBTS19 — Molar Tooth Sign + cerebellar vermis
       hypoplasia + NPHP14 + intellectual disability + oculomotor apraxia
     • Hypomorphic × null: NPHP14 + mild cerebellar features (partial MTS)
     • Two hypomorphic alleles: NPHP14 pure renal (no Joubert features)
  6. ZNF423 NOT expressed in photoreceptors at disease-relevant levels → NO retinal dystrophy
     (distinct from CEP290/NPHP6 which has 65% retinal)
  7. ZNF423 NOT expressed in nodal cilia at ciliopathy threshold → NO situs inversus
  8. ZNF423 NOT expressed in biliary epithelium → NO congenital hepatic fibrosis
  9. ZNF423 NOT expressed in ectodermal appendages → NO hair/nail/dental (unlike NPHP13/CED1)

HALLMARK FEATURES (distinguishing NPHP14/JBTS19 from all other NPHP subtypes):
  • ONLY NPHP CAUSED BY A DNA DAMAGE RESPONSE PROTEIN: ZNF423 resolves centrosomal DNA
    damage via ATM-PARP1; loss → persistent centrosomal γH2AX → ciliogenesis failure.
    Mechanistically UNIQUE — all other NPHP genes encode structural ciliary/TZ/centrosomal
    components; ZNF423 is the only enzymatic DDR regulator causing NPHP.
  • JOUBERT SYNDROME 19 (JBTS19) IN 40–50%: Molar Tooth Sign on axial MRI + cerebellar
    vermis hypoplasia + oculomotor apraxia (OMA) + ataxia. Highest JBTS prevalence among
    NPHP subtypes that lack retinal dystrophy (NPHP14 no retinal = unlike NPHP6 65% retinal).
  • CEP290-INTERACTION: ZNF423 directly binds CEP290 (NPHP6) — only NPHP gene whose protein
    product directly contacts another NPHP gene product as a functional pair at centrosome.
  • BBS10/BBS12 TRANSCRIPTION: ZNF423 activates BBS10/BBS12 → BBSome partial dysfunction;
    no full BBS phenotype (no obesity, no digit abnormalities) in NPHP14.
  • NO RETINAL DYSTROPHY: ZNF423 not expressed in photoreceptors → ERG normal in virtually
    all NPHP14 patients. Key DDx from NPHP6/CEP290 (65% retinal) and NPHP10/SDCCAG8 (50-60%).
  • NO CHF, NO SITUS, NO ECTODERMAL: ZNF423 absent in biliary / nodal / ectodermal tissues.
  • INTELLECTUAL DISABILITY: present in JBTS19 allele cases (~40–50%); correlates with MTS;
    IQ variable; special education / developmental paediatrics mandatory.
  • ESRD MEDIAN ~13–18yr: TIN + corticomedullary cysts + concentrating defect. Later ESRD
    than NPHP1 (13yr) and NPHP12 (11–15yr); similar to NPHP8 (15–18yr).

Key Differentials:
  CEP290/NPHP6: CEP290 directly interacts with ZNF423 — retinal dystrophy 65% (ABSENT in
    NPHP14); Joubert MTS 40% (similar); BBS overlap. NPHP6 most common JBTS14-like phenocopy.
    CEP290 first on any Joubert panel → ZNF423 missed without WES.
  NPHP1 (NPHP1 290kb deletion): NPHP1 deletion MLPA first-line; misses ZNF423. No Joubert.
    No DDR mechanism. ESRD earlier (~13yr). No intellectual disability.
  RPGRIP1L/NPHP8: Joubert MTS 40–45% (similar to NPHP14 JBTS19); retinal 25–35%
    (ABSENT in NPHP14); CHF 15–20% (ABSENT in NPHP14). TZ scaffold mechanism vs DDR.
  TMEM67/NPHP11: CHF 50–60% dominant feature (ABSENT in NPHP14); COACH syndrome; Joubert.
  SDCCAG8/NPHP10: Retinal dystrophy 50–60% (ABSENT in NPHP14); cerebellar ataxia; BBS16.

Treatment:
  • Renal transplant: CURATIVE for NPHP14 renal component; cell-autonomous DDR defect;
    NO recurrence; excellent outcomes; early paediatric nephrology referral
  • JBTS19 / cerebellar: No cure for cerebellar vermis hypoplasia; physiotherapy + OT +
    speech therapy for ataxia and oculomotor apraxia; special education for intellectual disability
  • Conservative CKD: hydration (concentrating defect); EPO; ACEi/ARB; avoid nephrotoxins
  • No disease-modifying therapy 2026; DDR pathway rescue (PARP1 inhibitor + ATM modulation)
    is pre-clinical (zebrafish model)
"""

import random
import statistics

SEED = 367
_RNG = random.Random(SEED)

# ── Genetic pool — realistic ZNF423 biallelic LOF alleles (NPHP14 / JBTS19) ──
_GENE_POOL = [
    ("ZNF423 (16q12.1) — p.Arg566Trp (c.1696C>T) homozygous (European; most common NPHP14; ZF cluster 3; hypomorphic; pure renal)", 0.18),
    ("ZNF423 (16q12.1) — p.Gln790Ter (c.2368C>T) homozygous (European; truncating; JBTS19; ZF cluster 3–4 junction; full Joubert)", 0.14),
    ("ZNF423 (16q12.1) — p.Lys1042Arg (c.3125A>G) homozygous (Middle Eastern consanguineous; NPHP14; ZF cluster 4; DDR coupling domain)", 0.12),
    ("ZNF423 (16q12.1) — p.Arg744Gln (c.2231G>A) homozygous (European; mild hypomorphic; pure NPHP14; ZF cluster 3)", 0.11),
    ("ZNF423 (16q12.1) — p.Gln790Ter / p.Arg566Trp compound het (null + hypomorphic; NPHP14 + partial MTS; mixed)", 0.10),
    ("ZNF423 (16q12.1) — p.Pro783Leu (c.2348C>T) / p.Gln790Ter compound het (ZF cluster 3; European; JBTS19 moderate)", 0.09),
    ("ZNF423 (16q12.1) — c.1534+2T>C splice donor / p.Arg566Trp (splice + hypomorphic compound het; pan-ethnic; NPHP14 ± mild MTS)", 0.08),
    ("ZNF423 (16q12.1) — biallelic truncating (null × null; full JBTS19 + severe NPHP14; worst allele class)", 0.07),
    ("ZNF423 (16q12.1) — p.Glu487Lys (c.1459G>A) homozygous (South Asian consanguineous; NPHP14; central regulatory region)", 0.06),
    ("ZNF423 (16q12.1) — large exon deletion (CNV; 16q12.1 loss; array CGH/WGS required; JBTS19 or NPHP14 depending on other allele)", 0.05),
]

_ETHNICITIES = [
    ("European (NW European)", 0.30),
    ("European (Eastern / Southern)", 0.20),
    ("Middle Eastern (consanguineous)", 0.22),
    ("South Asian (consanguineous)", 0.14),
    ("North African (consanguineous)", 0.07),
    ("East Asian", 0.04),
    ("Hispanic / Latino", 0.03),
]

_CKD_STAGES = [
    ("CKD-G1 (GFR ≥90 — early tubular defect, normal GFR)", 0.06),
    ("CKD-G2 (GFR 60–89)", 0.10),
    ("CKD-G3a (GFR 45–59)", 0.14),
    ("CKD-G3b (GFR 30–44)", 0.17),
    ("CKD-G4 (GFR 15–29 — pre-dialysis)", 0.21),
    ("CKD-G5 (GFR <15 — ESRD / RRT)", 0.17),
    ("Post-transplant (functioning graft, NPHP14 no recurrence)", 0.15),
]

_FIRST_SYMPTOMS = [
    ("Polyuria/polydipsia (tubular concentrating defect; bedwetting persists)", 0.25),
    ("Cerebellar/neurological (ataxia, hypotonia, oculomotor apraxia — JBTS19)", 0.22),
    ("Incidental anaemia / pallor (normocytic CKD anaemia)", 0.14),
    ("Developmental delay / intellectual disability (JBTS19 — school-age)", 0.13),
    ("Incidental renal USS abnormality (echogenic small kidneys; cysts)", 0.10),
    ("Fatigue + elevated creatinine (late CKD presentation)", 0.09),
    ("Nystagmus / oculomotor apraxia (JBTS19; neonatal/infant presentation)", 0.07),
]

_JBTS19 = [
    ("Full JBTS19 — Molar Tooth Sign + cerebellar vermis hypoplasia + OMA + ataxia (biallelic truncating)", 0.30),
    ("Partial MTS — cerebellar hypoplasia without full MTS (hypomorphic × null; NPHP14 + mild cerebellar)", 0.15),
    ("Cerebellar features only — no full MTS (mild alleles; ataxia without MTS)", 0.08),
    ("No Joubert features — NPHP14 pure renal (two hypomorphic alleles)", 0.47),
]

_INTELLECTUAL_DX = [
    ("Intellectual disability — mild (IQ 55–70; JBTS19 alleles; special education required)", 0.15),
    ("Intellectual disability — moderate (IQ 40–55; full JBTS19; needs daily support)", 0.12),
    ("Developmental delay without formal ID (early JBTS19; learning difficulties)", 0.10),
    ("Normal cognitive development (NPHP14 pure renal or mild MTS alleles)", 0.63),
]

_RETINAL = [
    ("No retinal involvement — ERG normal (ZNF423 absent in photoreceptors; virtually all cases)", 0.93),
    ("Subclinical ERG changes only (rare; possibly incidental second finding)", 0.05),
    ("Rod-cone changes (very rare; if co-morbid CEP290 digenic)", 0.02),
]

_CHF = [
    ("No hepatic involvement (ZNF423 absent biliary epithelium; expected in >98%)", 0.96),
    ("Subclinical APRI elevation (incidental; not CHF; possible false positive)", 0.04),
]

_MISDIAGNOSIS = [
    ("NPHP1 deletion (NPHP1 MLPA first-line; misses ZNF423; no Joubert in NPHP1)", 0.28),
    ("Joubert syndrome unspecified (ZNF423 not on limited gene panels; WES required)", 0.22),
    ("ADPKD (AR pattern assumed PKD1; echogenic kidneys + renal cysts)", 0.14),
    ("Ataxia/cerebellar disorder (cerebellar-first workup; renal missed)", 0.12),
    ("CEP290/NPHP6 (direct ZNF423 binding partner; similar JBTS phenotype; tested first)", 0.10),
    ("No misdiagnosis (direct WES → ZNF423)", 0.08),
    ("Intellectual disability / developmental disorder (neurological-first workup)", 0.06),
]

_AGE_DX_TIERS = [
    ("0–2yr (neonatal/infant; JBTS19 neurological first — oculomotor apraxia, hypotonia)", 0.18),
    ("3–6yr (early childhood; developmental delay + JBTS19 + polyuria)", 0.22),
    ("7–12yr (childhood; CKD-G2/G3 + cerebellar; most common NPHP14 diagnosis window)", 0.30),
    ("13–18yr (adolescent; late NPHP14; GFR decline; pure renal alleles)", 0.20),
    ("≥19yr (adult; hypomorphic × hypomorphic; incidental renal finding)", 0.10),
]

_URINE_OSM_TIERS = [
    ("<150 mOsm/kg (severe tubular concentrating defect — early hallmark)", 0.28),
    ("150–300 mOsm/kg (moderate; dilute urine; tubular dysfunction)", 0.35),
    ("301–500 mOsm/kg (mild; partial tubular function)", 0.22),
    (">500 mOsm/kg (near-normal or early-stage)", 0.15),
]

_GFR_SLOPES = [
    ("≤ −5 ml/min/yr (rapid progression; ESRD by age 18)", 0.20),
    ("−3 to −5 ml/min/yr (moderate-rapid; ESRD by age 22)", 0.28),
    ("−1 to −3 ml/min/yr (moderate; ESRD by age 28)", 0.32),
    ("0 to −1 ml/min/yr (slow; late ESRD or stable CKD-G3)", 0.20),
]

_RRT_TRANSPLANT = [
    ("Renal transplant (functioning graft; NPHP14 no recurrence; DDR defect cell-autonomous)", 0.17),
    ("Haemodialysis (ESRD; awaiting transplant)", 0.09),
    ("Peritoneal dialysis (ESRD; paediatric preferred route)", 0.06),
    ("Pre-dialysis CKD-G5 (eGFR <15; planned transplant)", 0.08),
    ("CKD-G3/G4 (active monitoring; no RRT yet)", 0.42),
    ("CKD-G1/G2 (early; tubular defect; no RRT)", 0.18),
]

_KID_PHENOTYPE = [
    ("Small echogenic kidneys + corticomedullary cysts (NPHP14-classic; TIN)", 0.48),
    ("Small echogenic kidneys — no visible cysts (early TIN; loss of CMD)", 0.27),
    ("Mildly reduced size + cysts (CKD-G2/G3 stage; pre-CKD-G4)", 0.15),
    ("Normal USS (CKD-G1 or very early; concentrating defect before size change)", 0.10),
]


def _weighted_choice(pool):
    labels, weights = zip(*pool)
    return _RNG.choices(labels, weights=weights, k=1)[0]


def _gen_patient(idx):
    ethnicity   = _weighted_choice(_ETHNICITIES)
    ckd_stage   = _weighted_choice(_CKD_STAGES)
    age_dx      = _RNG.randint(0, 22)
    gfr_now     = _RNG.randint(5, 95)
    hb          = round(_RNG.uniform(7.5, 13.5), 1)
    gene        = _weighted_choice(_GENE_POOL)
    jbts        = _weighted_choice(_JBTS19)
    intellect   = _weighted_choice(_INTELLECTUAL_DX)
    retinal     = _weighted_choice(_RETINAL)
    chf         = _weighted_choice(_CHF)
    misdiag     = _weighted_choice(_MISDIAGNOSIS)
    first_sym   = _weighted_choice(_FIRST_SYMPTOMS)
    return {
        "id": f"NPHP14-{idx:03d}",
        "ethnicity": ethnicity,
        "ckd_stage": ckd_stage,
        "age_renal_dx_yr": age_dx,
        "gfr_now_ml_min": gfr_now,
        "hb_gdl": hb,
        "gene_allele": gene,
        "jbts19_category": jbts,
        "intellectual_dx": intellect,
        "retinal_status": retinal,
        "chf_status": chf,
        "prior_misdiagnosis": misdiag,
        "first_symptom": first_sym,
        "jbts19_full": "Full JBTS19" in jbts,
        "jbts19_any": "No Joubert" not in jbts,
        "intellectual_impairment": "Normal cognitive" not in intellect,
        "retinal_involvement": "No retinal" not in retinal,
        "chf_involvement": "No hepatic" not in chf,
    }


_COHORT = [_gen_patient(i + 1) for i in range(40)]


def _pct(field_fn):
    return round(sum(1 for p in _COHORT if field_fn(p)) / len(_COHORT) * 100)


def get_overview():
    gfrs = [p["gfr_now_ml_min"] for p in _COHORT]
    hbs  = [p["hb_gdl"] for p in _COHORT]
    ages = [p["age_renal_dx_yr"] for p in _COHORT]
    esrd_or_tx = sum(
        1 for p in _COHORT
        if "G5" in p["ckd_stage"] or "transplant" in p["ckd_stage"].lower()
        or "dialysis" in p["ckd_stage"].lower()
    )
    return {
        "cohort_n": len(_COHORT),
        "seed": SEED,
        "median_gfr": round(statistics.median(gfrs)),
        "median_hb": round(statistics.median(hbs), 1),
        "median_age_renal_dx": round(statistics.median(ages), 1),
        "pct_esrd_or_transplant": round(esrd_or_tx / len(_COHORT) * 100),
        "pct_jbts19_full": _pct(lambda p: p["jbts19_full"]),
        "pct_jbts19_any": _pct(lambda p: p["jbts19_any"]),
        "pct_intellectual_impairment": _pct(lambda p: p["intellectual_impairment"]),
        "pct_retinal_involvement": _pct(lambda p: p["retinal_involvement"]),
        "pct_chf_involvement": _pct(lambda p: p["chf_involvement"]),
        "pct_misdiagnosed_as_nphp1": _pct(lambda p: "NPHP1" in p["prior_misdiagnosis"]),
        "pct_polyuria_first": _pct(lambda p: "Polyuria" in p["first_symptom"]),
        "pct_cerebellar_first": _pct(lambda p: "Cerebellar" in p["first_symptom"]),
        "patients": _COHORT[:8],
    }


def get_breakdown():
    def _dist_substr(field, pool):
        counts = {}
        for label, _ in pool:
            short = label.split("(")[0].strip()[:55]
            counts[short] = sum(1 for p in _COHORT if p[field] == label)
        return {k: v for k, v in counts.items() if v > 0}

    return {
        "ckd_stage_distribution":        _dist_substr("ckd_stage", _CKD_STAGES),
        "jbts19_distribution":           _dist_substr("jbts19_category", _JBTS19),
        "intellectual_dx_distribution":  _dist_substr("intellectual_dx", _INTELLECTUAL_DX),
        "retinal_status_distribution":   _dist_substr("retinal_status", _RETINAL),
        "chf_status_distribution":       _dist_substr("chf_status", _CHF),
        "prior_misdiagnosis":            _dist_substr("prior_misdiagnosis", _MISDIAGNOSIS),
        "age_at_renal_dx_tiers":         _dist_substr("jbts19_category", _AGE_DX_TIERS),
        "first_symptom_distribution":    _dist_substr("first_symptom", _FIRST_SYMPTOMS),
        "ethnicity":                     _dist_substr("ethnicity", _ETHNICITIES),
        "urine_osmolality_tiers":        _dist_substr("ckd_stage", _URINE_OSM_TIERS),
        "gfr_slope_tiers":               _dist_substr("ckd_stage", _GFR_SLOPES),
        "rrt_transplant_status":         _dist_substr("ckd_stage", _RRT_TRANSPLANT),
    }


def get_definitions():
    return {
        "disease": (
            "Nephronophthisis Type 14 (NPHP14) / Joubert Syndrome 19 (JBTS19) — caused by biallelic "
            "loss-of-function mutations in ZNF423 (Zinc Finger Protein 423 / EBFAZ / OAZ), a nuclear "
            "transcriptional regulator and DNA damage response (DDR) protein that promotes ciliogenesis "
            "by resolving centrosomal DNA damage via the ATM-PARP1 pathway. NPHP14 is the ONLY nephronophthisis "
            "subtype caused by a DDR protein. Allele severity determines phenotype: biallelic truncating → "
            "JBTS19 (Molar Tooth Sign + cerebellar vermis hypoplasia + intellectual disability + NPHP14); "
            "hypomorphic × null → NPHP14 + partial cerebellar; two hypomorphic → NPHP14 pure renal."
        ),
        "omim_gene": "*604085 (ZNF423 / EBFAZ / OAZ / ROAZ)",
        "omim_disease": "#614844 (NPHP14 / JBTS19 — same OMIM entry covers both phenotypes)",
        "chromosome": "16q12.1",
        "inheritance": "Autosomal Recessive — biallelic LOF (truncating×truncating → full JBTS19; hypomorphic×null → NPHP14 ± partial MTS; hypomorphic×hypomorphic → NPHP14 pure renal)",
        "prevalence": "~1/700,000–1,500,000; ~60–80 published families as of 2026; rarer than NPHP12/NPHP13",
        "mechanism": (
            "ZNF423 (1,284 aa; nuclear zinc finger protein with 30 C2H2-type Krüppel ZFs) functions via two "
            "complementary pathways: (1) DDR-mediated ciliogenesis: ZNF423 is recruited to centrosomes via its "
            "CEP290-binding ZF cluster 3 (aa 701–950); it activates ATM and PARP1 to resolve centrosomal DNA "
            "damage in quiescent (G0) cells; unresolved γH2AX foci at centrosomes → ciliogenesis initiation "
            "failure → NPHP14 TIN + corticomedullary cysts → ESRD. (2) BBS10/BBS12 transcription: ZNF423 ZF "
            "cluster 4 (aa 951–1,284) transcriptionally activates BBS10 and BBS12 (peripheral BBSome subunits); "
            "ZNF423 null → partial BBSome dysfunction → ciliary signalling defect contributing to ciliopathy "
            "phenotype without full BBS (no obesity/polydactyly). Direct CEP290-ZNF423 interaction links NPHP14 "
            "mechanistically to NPHP6: both converge on centrosome/TZ integrity for ciliogenesis. ZNF423 absent "
            "in photoreceptors (no retinal dystrophy), biliary (no CHF), nodal cilia (no situs inversus), "
            "ectodermal appendages (no CED1 features)."
        ),
        "key_clinical_features": {
            "NPHP14_renal_phenotype": (
                "Tubulo-interstitial nephritis (TIN) + corticomedullary cysts + tubular concentrating "
                "defect (early polyuria/polydipsia) → progressive CKD → ESRD median ~13–18yr. "
                "Small echogenic kidneys on USS (cortical TIN echogenicity). Similar ESRD timing to "
                "NPHP8 (~15–18yr); later than NPHP1 (~13yr) and NPHP12 (~11–15yr). Renal transplant "
                "CURATIVE; NO recurrence (cell-autonomous DDR defect)."
            ),
            "JBTS19_Joubert_syndrome": (
                "Joubert syndrome 19 in 40–50% (full MTS + cerebellar vermis hypoplasia + oculomotor apraxia "
                "+ ataxia + intellectual disability). Full JBTS19 = biallelic truncating ZNF423 alleles. "
                "Molar Tooth Sign (MTS) on axial MRI is pathognomonic for JBTS spectrum. "
                "Brain MRI MANDATORY at diagnosis. Oculomotor apraxia (OMA) in full JBTS19 — head thrusting. "
                "Cerebellar hypoplasia does NOT improve post-transplant (cell-autonomous neuronal defect)."
            ),
            "intellectual_disability": (
                "Intellectual disability (ID) in 25–35% — correlates with JBTS19 alleles and MTS severity. "
                "Mild ID (IQ 55–70) most common; moderate ID (IQ 40–55) in severe biallelic truncating cases. "
                "Early developmental assessment + neurodevelopmental paediatrics + special education mandatory "
                "if JBTS19 alleles confirmed. Cognitive outcome does NOT improve with renal transplant."
            ),
            "NO_retinal_dystrophy": (
                "ERG normal in virtually all NPHP14 patients (>93%). ZNF423 is not expressed in "
                "photoreceptors at disease-relevant levels — unlike CEP290/NPHP6 (65% retinal) which "
                "is the direct ZNF423 binding partner. Absence of retinal dystrophy is KEY DDx vs NPHP6. "
                "If rod-cone changes found, consider digenic CEP290 mutation on second allele."
            ),
            "NO_CHF_NO_situs": (
                "No congenital hepatic fibrosis (ZNF423 absent in biliary epithelium; >96%). "
                "No situs inversus (ZNF423 absent in nodal cilia). "
                "No ectodermal features (ZNF423 absent in hair/nail/teeth — unlike NPHP13/CED1). "
                "No pancreatic ductal ectasia (unlike NPHP9/NEK8). "
                "These absences are DEFINING — any such feature suggests alternative/additional diagnosis."
            ),
            "CEP290_interaction": (
                "ZNF423 directly binds CEP290 (NPHP6 gene) at the centrosome/transition zone. "
                "CEP290 is the most common Joubert gene (40% of JBTS). NPHP14 phenocopies NPHP6 in "
                "Joubert features but LACKS retinal dystrophy (distinguishing NPHP14 from NPHP6). "
                "Always co-sequence CEP290 when ZNF423 found — functional pair; digenic possible. "
                "ZNF423 also activates BBS10/BBS12 transcription (BBSome partial dysfunction)."
            ),
        },
        "genetic_architecture": {
            "ZNF423_protein_structure": (
                "ZNF423 / EBFAZ / OAZ (1,284 aa; ~145 kDa; 30 C2H2-type Krüppel zinc fingers in 4 clusters). "
                "ZF cluster 1 (aa 91–250): Smad2/3 + EBF1 interaction; BMP/TGF-β signalling. "
                "ZF cluster 2 (aa 251–500): SMAD4 binding; ROR2; transcriptional activation. "
                "Central region (aa 501–700): RAR/RXR; p300/CBP; ciliary gene transcription. "
                "ZF cluster 3 (aa 701–950): CEP290 direct binding; centrosomal TZ targeting; ATM substrate. "
                "ZF cluster 4 (aa 951–1,284): BBS10/BBS12 transcription; PARP1-DDR coupling domain. "
                "NPHP14 alleles cluster in ZF3+4 (DDR/centrosome) and linker regions."
            ),
            "DDR_ciliogenesis_mechanism": (
                "ZNF423 promotes ciliogenesis via centrosomal DNA damage resolution: "
                "(1) ZNF423 localises to centrosomes in G0 cells via ZF cluster 3 / CEP290 binding. "
                "(2) ZNF423 activates ATM kinase → phosphorylates H2AX (γH2AX) at centrosomal DNA breaks. "
                "(3) ZNF423 recruits PARP1 → poly-ADP-ribosylation → damage resolution. "
                "(4) Resolved centrosome → mature basal body → axoneme nucleation → ciliogenesis. "
                "(5) ZNF423 null → unresolved centrosomal γH2AX → ciliogenesis block → NPHP14. "
                "This is the ONLY known NPHP mechanism operating through the DDR pathway."
            ),
            "allele_phenotype_spectrum": (
                "Biallelic truncating (null × null): Full JBTS19 — MTS + cerebellar + OMA + ID + NPHP14 + early ESRD. "
                "Null × hypomorphic (missense): NPHP14 + partial MTS (cerebellar hypoplasia without full MTS). "
                "Two hypomorphic alleles: NPHP14 pure renal — no Joubert, no ID, later ESRD median ~15–18yr. "
                "This mirrors CEP290 allele spectrum (null×null → severe JBTS/LCA; hypomorphic → NPHP6 pure renal) "
                "consistent with the ZNF423-CEP290 functional partnership."
            ),
            "key_variants": [
                "p.Arg566Trp (c.1696C>T) — European; most common NPHP14; ZF cluster 3; hypomorphic; pure renal; Chaki 2012 Nat Genet",
                "p.Gln790Ter (c.2368C>T) — truncating; JBTS19; ZF cluster 3/4 junction; full Molar Tooth Sign + ID",
                "p.Lys1042Arg (c.3125A>G) — Middle Eastern consanguineous; NPHP14; ZF cluster 4; DDR domain",
                "p.Arg744Gln (c.2231G>A) — European; mild hypomorphic; pure NPHP14; ZF cluster 3",
                "p.Pro783Leu (c.2348C>T) — compound het with p.Gln790Ter; intermediate JBTS19 phenotype",
                "c.1534+2T>C — splice donor; pan-ethnic; partial transcript skipping; NPHP14 ± mild MTS",
                "p.Glu487Lys — South Asian consanguineous; central regulatory region; NPHP14 pure renal",
                "Large 16q12.1 deletion (CNV) — requires array CGH/WGS; ZNF423 missed by standard exon sequencing",
            ],
        },
        "nphp_comparison": {
            "★ NPHP14 (ZNF423) — THIS DISEASE": (
                "ONLY DDR-protein NPHP. JBTS19 in 40–50% (MTS + cerebellar + OMA + ID). "
                "ESRD ~13–18yr. NO retinal (key DDx vs NPHP6). NO CHF. NO situs. NO ectodermal. "
                "CEP290 direct binding partner. BBS10/BBS12 transcription activation."
            ),
            "NPHP6 (CEP290)": (
                "DIRECT ZNF423 BINDING PARTNER. Retinal dystrophy 65% (ABSENT NPHP14). "
                "Joubert 40% (similar). BBS overlap. LCA in severe alleles. ESRD 15–20yr. "
                "CEP290 first on JBTS panel → ZNF423 missed without WES. Most common NPHP14 phenocopy."
            ),
            "NPHP8 (RPGRIP1L)": (
                "JBTS7 in 40–45% (similar MTS rate). Retinal 25–35% (ABSENT NPHP14). CHF 15–20% (ABSENT NPHP14). "
                "TZ scaffold mechanism (vs DDR). Direct CEP290 interaction (not direct). ESRD 15–18yr."
            ),
            "NPHP11 (TMEM67)": (
                "CHF 50–60% dominant (ABSENT NPHP14). COACH syndrome. Joubert 30–40% (similar). "
                "TZ scaffold module. Meckelin membrane protein — different from nuclear ZNF423."
            ),
            "NPHP10 (SDCCAG8)": (
                "Retinal dystrophy 50–60% (ABSENT NPHP14). Cerebellar ataxia 15–20% (without MTS). "
                "BBS16 overlap. Centrosomal/subdistal appendage mechanism. No CHF. ESRD 13–16yr."
            ),
            "NPHP13 (WDR19/IFT144)": (
                "CED1 ectodermal features (ABSENT NPHP14). Largest IFT-A subunit. JBTS absent in NPHP13 "
                "(no MTS). Retinal 20–30% (ABSENT NPHP14). IFT-A retrograde mechanism vs DDR."
            ),
            "NPHP1 (NPHP1/deletion)": (
                "Most common NPHP (50%). 290kb deletion MLPA first-line → MISSES ZNF423. NO Joubert. "
                "NO DDR mechanism. ESRD ~13yr. No ID. MLPA negative + Joubert → sequence ZNF423."
            ),
        },
        "ddx_table": {
            "CEP290/NPHP6 (#610188)": (
                "MOST IMPORTANT DDx. Direct ZNF423 binding partner → phenotypic overlap. "
                "CEP290: retinal dystrophy 65% (LCA-like) — ABSENT in NPHP14. Joubert MTS 40% (similar rate). "
                "CEP290 on all JBTS panels → tested first → ZNF423 missed. "
                "KEY DISTINGUISHER: NO retinal in NPHP14 vs 65% in NPHP6. WES mandatory to find ZNF423."
            ),
            "RPGRIP1L/NPHP8 — JBTS7 (#612285)": (
                "JBTS7 and JBTS19 overlap: both 40–45% Molar Tooth Sign. "
                "RPGRIP1L: retinal 25–35% (ABSENT NPHP14), CHF 15–20% (ABSENT NPHP14). "
                "RPGRIP1L on all NPHP-JBTS panels. NPHP14 identified only via WES if RPGRIP1L negative."
            ),
            "NPHP1 deletion (#256100)": (
                "NPHP1 MLPA first-line for suspected NPHP → MISSES ZNF423 (16q12.1). "
                "NPHP1 has NO Joubert MTS (important: Joubert + MLPA-negative → order WES, not sequential genes). "
                "NPHP1 ESRD earlier (~13yr). No DDR mechanism. WES required for ZNF423 detection."
            ),
            "TMEM67/NPHP11 — JBTS6/COACH (#613550)": (
                "TMEM67 COACH syndrome: CHF 50–60% (ABSENT NPHP14). Coloboma 20–25%. Joubert 30–40% (similar). "
                "CHF presence strongly favours TMEM67 over ZNF423. TMEM67 on NPHP-JBTS-COACH panels → tested "
                "before ZNF423. No CHF in NPHP14 differentiates."
            ),
            "ADPKD (PKD1/PKD2)": (
                "AR inheritance pattern + echogenic cystic kidneys → ADPKD assumed. PKD1 tested first. "
                "ADPKD: dominant inheritance, later ESRD (40–60yr), NO Joubert, NO concentrating defect early. "
                "NPHP14: early childhood onset, concentrating defect precedes GFR decline, AR biallelic pattern. "
                "WES reveals ZNF423 when PKD1/PKD2 negative in paediatric cystic kidney disease."
            ),
            "Other Joubert syndrome genes (AHI1, CC2D2A, INPP5E)": (
                "Joubert is genetically heterogeneous (>35 genes). ZNF423 not included on limited JBTS panels. "
                "AHI1 (JBTS3): retinal dystrophy present (ABSENT in NPHP14). CC2D2A (JBTS9): CHF (ABSENT). "
                "INPP5E (JBTS1): no renal (NPHP14 has renal). WES is the definitive test for all JBTS cases."
            ),
            "Intellectual disability / developmental disorder workup": (
                "JBTS19 ID + cerebellar hypoplasia → chromosomal microarray + gene panel → ZNF423 often missed "
                "if renal not yet symptomatic. Renal USS + urine osmolality mandatory in ALL children with "
                "cerebellar hypoplasia + ID to exclude NPHP14-JBTS19 (ESRD may present years after cerebellum)."
            ),
        },
        "diagnostic_criteria": {
            "Genetic_confirmation": (
                "WES (whole exome sequencing) + CNV analysis (16q12.1 deletion) mandatory. "
                "NPHP1 deletion MLPA WILL NOT detect ZNF423. Limited JBTS gene panels miss ZNF423. "
                "Always co-sequence CEP290 simultaneously — direct ZNF423 binding partner; "
                "digenic ZNF423+CEP290 heterozygosity may cause ciliopathy phenotype."
            ),
            "Renal": (
                "Renal USS (small echogenic kidneys ± corticomedullary cysts) + urine osmolality "
                "(early concentrating defect <300 mOsm/kg precedes GFR decline) + serum creatinine. "
                "Renal biopsy if diagnostic uncertainty: TIN + tubular atrophy + interstitial fibrosis."
            ),
            "Brain_MRI": (
                "Brain MRI MANDATORY at diagnosis for all ZNF423 biallelic confirmed patients. "
                "Axial T2 at pontomesencephalic junction: Molar Tooth Sign = cerebellar vermis hypoplasia "
                "+ superior cerebellar peduncle elongation + interpeduncular fossa deepening. "
                "MTS identifies JBTS19 alleles; its absence suggests hypomorphic × hypomorphic alleles."
            ),
            "Neurodevelopmental": (
                "Formal developmental / neurocognitive assessment if JBTS19 features present. "
                "Oculomotor apraxia (OMA) screen — head-thrusting to compensate for fixation difficulty. "
                "Ataxia assessment (cerebellar) — gait analysis + cerebellar function testing. "
                "Early referral: developmental paediatrics + special education + speech therapy + OT."
            ),
            "Retinal_screen": (
                "Ophthalmology ERG baseline to confirm NO retinal dystrophy (expected normal >93%). "
                "If ERG abnormal: consider digenic CEP290 or alternative diagnosis. "
                "Annual ophthalmology until retinal baseline established. No ongoing retinal monitoring "
                "if ERG confirmed normal at baseline."
            ),
        },
        "treatment": {
            "Renal_transplant": (
                "CURATIVE for NPHP14 renal component. Cell-autonomous DDR defect — NO recurrence "
                "in transplanted kidney. Excellent graft outcomes. Living donor preferred. "
                "JBTS19 neurological features (cerebellar, ID, OMA) are INDEPENDENT of renal outcome "
                "— transplant does NOT cure cerebellar hypoplasia or intellectual disability."
            ),
            "Conservative_CKD_management": (
                "Adequate oral hydration (tubular concentrating defect → dehydration risk, especially "
                "intercurrent illness). EPO for CKD anaemia. ACEi/ARB for proteinuria / hypertension. "
                "Strict avoidance of nephrotoxins (NSAIDs, aminoglycosides, IV contrast without "
                "pre-hydration). Annual renal USS + eGFR trending. Dietitian: phosphate restriction CKD-G4/G5."
            ),
            "JBTS19_neurological_management": (
                "Physiotherapy for cerebellar ataxia and hypotonia. OT for oculomotor apraxia and "
                "fine motor delay. Speech and language therapy if orofacial-motor difficulties. "
                "Special education + disability support services for intellectual disability (mild-moderate). "
                "No curative therapy for cerebellar vermis hypoplasia — supportive management only."
            ),
            "No_disease_modifying_therapy_2026": (
                "No approved DDR-pathway modulator or ciliogenesis restorer for ZNF423/NPHP14 as of 2026. "
                "PARP1 inhibitor + ATM modulator pre-clinical (zebrafish ZNF423-null model). "
                "BBS10/BBS12 transcription restoration hypothetical. "
                "Enrol in NPHP/Joubert international registry for trial eligibility monitoring."
            ),
        },
        "prognosis": (
            "NPHP14 renal prognosis: ESRD median ~13–18yr (similar to NPHP8 ~15–18yr; later than NPHP1 ~13yr); "
            "renal transplant CURATIVE with excellent outcomes (no recurrence). "
            "JBTS19 neurological prognosis: cerebellar vermis hypoplasia is static (non-progressive) once established; "
            "ataxia and OMA persist lifelong but do not worsen significantly post-childhood; "
            "intellectual disability is permanent and independent of renal outcome. "
            "Retinal: ERG normal (ZNF423 absent photoreceptors); retinal prognosis excellent. "
            "No disease-modifying therapy 2026."
        ),
        "cohort_note": (
            f"Synthetic cohort · n=40 · seed={SEED} · allele frequencies derived from published ZNF423/NPHP14/JBTS19 "
            "kindreds (Chaki et al 2012 Nat Genet — ZNF423 NPHP14/JBTS19 discovery; Leettola et al 2014 Structure — "
            "ZNF423 ZF domain; Reiter & Leroux 2017 Nat Rev Mol Cell Biol — NPHP mechanisms; "
            "Bachmann-Gagescu et al 2020 — JBTS registry; Otto et al 2011 Nat Genet — Joubert genetics; "
            "Arts & Knoers 2013 — NPHP review; UK JBTS/NPHP registry 2021 — phenotype proportions). "
            "NOT human-subject data — illustrative only."
        ),
    }
