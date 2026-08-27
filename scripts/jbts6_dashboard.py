"""
TMEM67 Joubert Syndrome Type 6 (JBTS6) — Meckelin / MKS3 / NPHP11 / COACH
=============================================================================
Primary Gene : TMEM67 (*609884) — 8q22.1; 995 aa; Transmembrane Protein 67 (Meckelin)
               TMEM67 (Meckelin/MKS3) is a TRANSITION ZONE MEMBRANE PROTEIN.
               TMEM67 contains: large N-terminal extracellular fibronectin-III-like (FN-III)
               domain (aa 1–750), a single transmembrane domain (aa 750–770), and a short
               C-terminal cytoplasmic tail (aa 770–995). The FN-III domain is proposed to
               sense extracellular cues (Wnt/Frizzled); the cytoplasmic tail interacts with
               NPHP4 and CC2D2A at the TZ. TMEM67 anchors TZ membrane protein gate integrity;
               loss → gate leaky → GPCR/Smoothened ciliary entry fails → Hedgehog impaired
               → Molar Tooth Sign (MTS), cerebellar vermis hypoplasia.
               TMEM67 is unique: ONLY TZ gene causing COACH (Cerebellar-Oculo-Renal +
               Coloboma + Hepatic fibrosis) along with CC2D2A. ~5-10% of all JBTS.
Disease OMIM : #610688 — Joubert Syndrome 6 (JBTS6)
               Also: #607361 = MKS3 (Meckel-Gruber Syndrome 3) · lethal, biallelic null
               Also: #613550 = NPHP11 (Nephronophthisis 11) · mild, biallelic hypomorphic
               Also: #216360 = COACH Syndrome (Cerebellar-Oculo-renal + Coloboma + Hepatic fibrosis)
Chromosome   : 8q22.1
Inheritance  : Autosomal Recessive — biallelic LOF; ALLELE CLASS GOVERNS DISEASE TIER
Prevalence   : ~5-10% of all Joubert syndrome cases
               ~1/300,000–500,000 worldwide (JBTS6 phenotype)
               COACH: ~30% of JBTS6 cases develop clinically significant hepatic fibrosis

⚠ KEY DIAGNOSTIC PEARL — COACH SYNDROME (Hepatic Fibrosis):
TMEM67 (and CC2D2A) are the ONLY two JBTS genes that cause COACH syndrome —
Cerebellar vermis hypoplasia, Oligophrenia, Ataxia, Coloboma, Hepatic fibrosis.
Hepatic fibrosis (ductal plate malformation, portal fibrosis) occurs in ~30% of JBTS6 and
may lead to portal hypertension, oesophageal varices, and hepatic failure. Liver biopsy shows
DUCTAL PLATE MALFORMATION (congenital hepatic fibrosis pattern). Combined liver-kidney
transplant is indicated in severe COACH with ESRD. All TMEM67 patients require annual liver
function tests, ultrasound, and hepatology referral.

⚠ NORTH AFRICAN FOUNDER ALLELE — p.Cys615Arg:
p.Cys615Arg (c.1843T>C) is the most common TMEM67 allele in North African (Maghreb) and
Middle Eastern populations, often found homozygous in consanguineous families. This allele
disrupts TM-adjacent extracellular folding and leads to JBTS6 + COACH (30-35% hepatic
fibrosis risk when homozygous p.Cys615Arg). It is ENRICHED in consanguineous pedigrees
and should be the first allele checked in Moroccan, Algerian, and Tunisian patients with JBTS.

Protein Structure — TMEM67 / Meckelin (995 aa; TZ membrane protein / FN-III receptor-like)
-------------------------------------------------------------------------------------------
Domain 1: Signal peptide (aa 1–25)               — ER targeting; co-translational cleavage
Domain 2: FN-III extracellular (aa 25–750)       — Wnt/Frizzled ligand sensing; NPHP11 alleles
           Multiple FN-III repeats; Ca²⁺ binding; extracellular cue transduction
           Tyr78Cys — South Asian; N-terminal FN-III fold disruption
           Gln376Ter — truncating null; biallelic → COACH/MKS3
Domain 3: Transmembrane domain (aa 750–770)      — TZ membrane anchoring; single-pass
           Leu736Pro — adjacent; TM insertion failure; MENA allele (JBTS6)
           Cys615Arg — TM-proximal extracellular; North African founder; COACH-enriched
           Trp628Cys — TM/cytoplasmic linker; MENA hypomorphic
Domain 4: C-terminal cytoplasmic tail (aa 770–995) — NPHP4/CC2D2A binding; TZ scaffold
           Arg941Gln — CC domain at cytoplasmic tail; TZ-membrane interface; European allele

Key pathogenic variant classes (TMEM67):
1. p.Arg941Gln (c.2822G>A): cytoplasmic CC domain; NPHP4 interface disrupted; European
2. p.Leu736Pro (c.2207T>C): TM-adjacent; TM insertion failure; MENA most common
3. p.Gln376Ter (c.1126C>T): FN-III truncating null; biallelic → COACH/MKS3
4. p.Tyr78Cys (c.233A>G): N-terminal FN-III; South Asian; JBTS6 mild
5. c.1864+1G>A (splice donor intron 17): null allele; European
6. p.Trp628Cys (c.1884G>T): TM/extracellular linker; MENA hypomorphic; JBTS6 + COACH
7. p.Cys615Arg (c.1843T>C): TM-proximal extracellular; North African founder; COACH-enriched

JBTS6 Allele-Phenotype Tier Rule:
  Biallelic NULL (two truncating): MKS3 (Meckel-Gruber, lethal — encephalocele + polydactyly
                                   + polycystic kidneys + CHF; incompatible with extrauterine life)
  One NULL + one HYPOMORPHIC:      JBTS6 (Molar Tooth Sign + COACH variable + renal variable)
  Biallelic HYPOMORPHIC (missense): JBTS6 ± COACH (30%) or NPHP11 (renal dominant) milder
  Homozygous p.Cys615Arg (founder): JBTS6 + COACH enriched (Maghreb/MENA)
"""

import random
import math

SEED = 419
N    = 40   # 40-patient educational cohort

rng  = random.Random(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────
def _pct(n, total=N):
    return round(n / total * 100)

def _split(total, *fractions):
    buckets = [round(total * f) for f in fractions]
    diff = total - sum(buckets)
    buckets[0] += diff
    return buckets

# ── patient-level data (fixed seed) ──────────────────────────────────────────
patients = []
ethnicities = [
    ('North African / Maghreb',        0.28),  # Cys615Arg founder enriched
    ('European',                       0.25),  # Arg941Gln / splice donor
    ('Middle Eastern',                 0.22),  # Leu736Pro / Trp628Cys
    ('South Asian',                    0.12),  # Tyr78Cys
    ('Other / Unknown',                0.08),
    ('Ashkenazi Jewish',               0.05),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Compound het: null + hypomorphic missense (JBTS6 tier)',
    'Homozygous p.Cys615Arg (North African founder — JBTS6 + COACH)',
    'Compound het: hypomorphic missense + missense (JBTS6 + COACH variable)',
    'Compound het: splice donor c.1864+1G>A + missense (JBTS6)',
    'Compound het: null + truncating (JBTS6 / MKS3 borderline)',
]
allele_fracs = [0.30, 0.27, 0.23, 0.13, 0.07]
allele_pool = []
for ac, frac in zip(allele_classes, allele_fracs):
    allele_pool.extend([ac] * round(frac * N))
while len(allele_pool) < N:
    allele_pool.append(allele_classes[0])
rng.shuffle(allele_pool)

age_dx_pool = (
    [rng.randint(0, 1) for _ in range(20)] +  # neonatal/infantile (MTS on MRI)
    [rng.randint(2, 8) for _ in range(14)] +   # early childhood
    [rng.randint(9, 22) for _ in range(6)]     # later (hepatic/renal-first)
)
rng.shuffle(age_dx_pool)

for i in range(N):
    eth = eth_pool[i]
    allele = allele_pool[i]
    age_dx = age_dx_pool[i]
    has_cys615 = 'Cys615Arg' in allele
    # phenotype flags
    mts          = rng.random() < 0.88
    ataxia       = rng.random() < 0.85
    hypotonia    = rng.random() < 0.85
    oma          = rng.random() < 0.60
    retinal      = rng.random() < 0.35
    hepatic      = rng.random() < (0.42 if has_cys615 else 0.26)
    renal        = rng.random() < 0.38
    polydactyly  = rng.random() < 0.20
    id_          = rng.random() < 0.68
    breathing    = rng.random() < 0.58
    # renal ESRD age (if renal)
    esrd_age = rng.randint(13, 28) if renal else None
    # hepatic portal HTN (subset of hepatic)
    portal_htn = (hepatic and rng.random() < 0.40)
    patients.append({
        "id": f"JBTS6-{i+1:03d}",
        "ethnicity": eth,
        "allele_class": allele,
        "age_dx": age_dx,
        "mts": mts,
        "ataxia": ataxia,
        "hypotonia": hypotonia,
        "oculomotor_apraxia": oma,
        "retinal": retinal,
        "hepatic_fibrosis": hepatic,
        "portal_hypertension": portal_htn,
        "renal_nphp11": renal,
        "esrd_age": esrd_age,
        "polydactyly": polydactyly,
        "intellectual_disability": id_,
        "breathing_dysregulation": breathing,
    })

# ── aggregate counts ──────────────────────────────────────────────────────────
n_mts         = sum(1 for p in patients if p['mts'])
n_ataxia      = sum(1 for p in patients if p['ataxia'])
n_hypotonia   = sum(1 for p in patients if p['hypotonia'])
n_oma         = sum(1 for p in patients if p['oculomotor_apraxia'])
n_retinal     = sum(1 for p in patients if p['retinal'])
n_hepatic     = sum(1 for p in patients if p['hepatic_fibrosis'])
n_portal_htn  = sum(1 for p in patients if p['portal_hypertension'])
n_renal       = sum(1 for p in patients if p['renal_nphp11'])
n_poly        = sum(1 for p in patients if p['polydactyly'])
n_id          = sum(1 for p in patients if p['intellectual_disability'])
n_breathing   = sum(1 for p in patients if p['breathing_dysregulation'])
n_cys615      = sum(1 for p in patients if 'Cys615Arg' in p['allele_class'])


def get_overview():
    return {
        "kpis": [
            {"label": "Cohort (n)", "value": str(N), "color": "#1a237e"},
            {"label": "Molar Tooth Sign", "value": f"{n_mts}/{N}", "color": "#1a237e"},
            {"label": "Cerebellar Ataxia", "value": f"{n_ataxia}/{N}", "color": "#4a148c"},
            {"label": "Hepatic Fibrosis (COACH)", "value": f"{n_hepatic}/{N}", "color": "#b71c1c"},
            {"label": "Renal NPHP11", "value": f"{n_renal}/{N}", "color": "#006064"},
            {"label": "North African Cys615Arg", "value": f"{n_cys615}/{N}", "color": "#e65100"},
        ],
        "hallmark": (
            "TMEM67 (Meckelin/MKS3) is a transition zone membrane protein with a large "
            "fibronectin-III-like extracellular domain and a single transmembrane anchor. "
            "Loss of TMEM67 → TZ membrane gate fails → GPCR/Smo ciliary entry impaired "
            "→ Hedgehog & Wnt impairment → Molar Tooth Sign + COACH hepatic fibrosis. "
            "TMEM67 accounts for ~5–10% of all Joubert syndrome cases worldwide."
        ),
        "critical_diagnostic_pearl": (
            "COACH SYNDROME (Hepatic Fibrosis): TMEM67 and CC2D2A are the ONLY two JBTS "
            "genes causing COACH (Cerebellar-Oculo-renal + Coloboma + Hepatic fibrosis). "
            f"Hepatic fibrosis (ductal plate malformation) occurs in ~{_pct(n_hepatic)}% "
            f"of this JBTS6 cohort. All TMEM67 patients require annual LFTs, liver ultrasound, "
            "and hepatology review from diagnosis. Portal hypertension and oesophageal varices "
            "may develop. Combined liver-kidney transplant indicated for COACH + ESRD."
        ),
        "north_african_founder_pearl": (
            "p.Cys615Arg (c.1843T>C) is the MOST COMMON TMEM67 allele in North African "
            "(Moroccan, Algerian, Tunisian) and Maghreb populations — found homozygous in "
            f"consanguineous families. {n_cys615}/{N} patients carry this allele. "
            "COACH hepatic fibrosis risk is enriched (~40% when homozygous p.Cys615Arg). "
            "This allele is ABSENT from most European JBTS gene panels; targeted TMEM67 "
            "sequencing or WES is mandatory in North African patients with Molar Tooth Sign."
        ),
        "allele_phenotype_rule": (
            "Biallelic NULL (two truncating): MKS3 — Meckel-Gruber lethal (encephalocele + "
            "polydactyly + polycystic kidneys; incompatible with extrauterine life). "
            "One NULL + one HYPOMORPHIC: JBTS6 (Molar Tooth Sign ± COACH ± NPHP11). "
            "Biallelic HYPOMORPHIC (missense/missense): JBTS6 milder ± COACH or NPHP11. "
            "Homozygous p.Cys615Arg (founder): JBTS6 + COACH enriched (Maghreb)."
        ),
        "prevalence": "~1/300,000–500,000 worldwide (JBTS6); ~5–10% of all Joubert syndrome cases",
        "first_description": (
            "Baala et al., 2007 (Nat Genet 39:875) — TMEM67/MKS3 identified as JBTS6 gene "
            "in consanguineous Moroccan families with Joubert syndrome + hepatic fibrosis. "
            "Smith et al., 2006 (Nat Genet 38:191) — MKS3 gene identified for Meckel-Gruber."
        ),
        "gene_summary": {
            "symbol": "TMEM67",
            "alias": "MECKELIN / MKS3",
            "omim_gene": "*609884",
            "omim_disease_jbts6": "#610688",
            "omim_disease_mks3": "#607361",
            "omim_disease_nphp11": "#613550",
            "omim_disease_coach": "#216360",
            "chromosome": "8q22.1",
            "protein_length": "995 aa",
            "protein_class": "Transmembrane protein; FN-III extracellular + single TM + cytoplasmic",
            "function": "TZ membrane gate integrity; Wnt/Frizzled extracellular sensing; NPHP4/CC2D2A TZ anchoring",
        },
        "phenotype_summary": {
            "mts_pct": _pct(n_mts),
            "ataxia_pct": _pct(n_ataxia),
            "hypotonia_pct": _pct(n_hypotonia),
            "oma_pct": _pct(n_oma),
            "retinal_pct": _pct(n_retinal),
            "hepatic_pct": _pct(n_hepatic),
            "portal_htn_pct": _pct(n_portal_htn),
            "renal_pct": _pct(n_renal),
            "id_pct": _pct(n_id),
            "breathing_pct": _pct(n_breathing),
            "polydactyly_pct": _pct(n_poly),
        },
    }


def get_breakdown():
    # Ethnicity distribution
    eth_counts = {}
    for p in patients:
        eth_counts[p['ethnicity']] = eth_counts.get(p['ethnicity'], 0) + 1

    # Allele class distribution
    allele_counts = {}
    for p in patients:
        allele_counts[p['allele_class']] = allele_counts.get(p['allele_class'], 0) + 1

    # Age at diagnosis distribution
    age_buckets = {"0-1y (neonatal)": 0, "2-8y": 0, "9-22y": 0}
    for p in patients:
        a = p['age_dx']
        if a <= 1:
            age_buckets["0-1y (neonatal)"] += 1
        elif a <= 8:
            age_buckets["2-8y"] += 1
        else:
            age_buckets["9-22y"] += 1

    # COACH breakdown: hepatic vs no hepatic
    coach_eth = {}
    for p in patients:
        if p['hepatic_fibrosis']:
            eth = p['ethnicity']
            coach_eth[eth] = coach_eth.get(eth, 0) + 1

    # Renal outcomes
    esrd_ages = [p['esrd_age'] for p in patients if p['renal_nphp11'] and p['esrd_age']]
    avg_esrd_age = round(sum(esrd_ages) / len(esrd_ages), 1) if esrd_ages else None

    # Transplant projections
    n_renal_tx = round(n_renal * 0.72)   # ~72% of NPHP11 reach ESRD needing Tx
    n_liver_tx  = round(n_portal_htn * 0.50)  # ~50% of portal HTN need liver Tx
    n_combined  = round(min(n_renal_tx, n_liver_tx) * 0.30)  # combined Tx subset

    return {
        "ethnicity": [{"ethnicity": k, "n": v, "pct": _pct(v)} for k, v in eth_counts.items()],
        "allele_class": [{"class": k, "n": v, "pct": _pct(v)} for k, v in allele_counts.items()],
        "age_at_diagnosis": [{"bucket": k, "n": v, "pct": _pct(v)} for k, v in age_buckets.items()],
        "coach_by_ethnicity": [{"ethnicity": k, "n_hepatic": v, "pct_of_group": _pct(v, eth_counts.get(k, 1))} for k, v in coach_eth.items()],
        "phenotype_matrix": [
            {"feature": "Molar Tooth Sign (MTS)", "n": n_mts, "pct": _pct(n_mts), "note": "Pathognomonic; brain MRI mandatory"},
            {"feature": "Cerebellar Ataxia", "n": n_ataxia, "pct": _pct(n_ataxia), "note": "Gait ataxia; truncal instability"},
            {"feature": "Neonatal Hypotonia", "n": n_hypotonia, "pct": _pct(n_hypotonia), "note": "Universal early feature"},
            {"feature": "Oculomotor Apraxia (OMA)", "n": n_oma, "pct": _pct(n_oma), "note": "Horizontal gaze initiation failure"},
            {"feature": "Intellectual Disability", "n": n_id, "pct": _pct(n_id), "note": "Moderate > severe; variable"},
            {"feature": "Breathing Dysregulation", "n": n_breathing, "pct": _pct(n_breathing), "note": "Neonatal episodic apnea/hyperpnea; self-resolves"},
            {"feature": "Retinal Dystrophy", "n": n_retinal, "pct": _pct(n_retinal), "note": "Rod-cone; annual ERG mandatory"},
            {"feature": "Hepatic Fibrosis (COACH)", "n": n_hepatic, "pct": _pct(n_hepatic), "note": "Ductal plate malformation; portal fibrosis — DISTINCTIVE"},
            {"feature": "Portal Hypertension", "n": n_portal_htn, "pct": _pct(n_portal_htn), "note": "Subset of hepatic fibrosis; varices risk"},
            {"feature": "Renal NPHP11", "n": n_renal, "pct": _pct(n_renal), "note": "TIN; ESRD risk median ~18yr"},
            {"feature": "Polydactyly", "n": n_poly, "pct": _pct(n_poly), "note": "Postaxial; less common than SRTD/MKS3"},
            {"feature": "North African p.Cys615Arg", "n": n_cys615, "pct": _pct(n_cys615), "note": "Founder allele; COACH enriched"},
        ],
        "transplant_outcomes": {
            "n_renal_transplant_needed": n_renal_tx,
            "n_liver_transplant_needed": n_liver_tx,
            "n_combined_liver_kidney": n_combined,
            "renal_tx_outcome": "CURATIVE (cell-autonomous, no recurrence post-Tx)",
            "hepatic_tx_outcome": "CURATIVE for CHF/varices; neurological/retinal NOT corrected",
            "combined_tx_note": "Combined liver-kidney Tx indicated when both ESRD + portal HTN present",
            "avg_esrd_age": avg_esrd_age,
        },
        "allele_protein_table": [
            {"variant": "p.Arg941Gln", "cdna": "c.2822G>A", "domain": "C-tail CC / NPHP4-interface", "ethnic": "European", "tier": "JBTS6 (hypomorphic)"},
            {"variant": "p.Leu736Pro", "cdna": "c.2207T>C", "domain": "TM-adjacent extracellular", "ethnic": "MENA", "tier": "JBTS6 / NPHP11"},
            {"variant": "p.Gln376Ter", "cdna": "c.1126C>T", "domain": "FN-III truncating null", "ethnic": "Pan-ethnic", "tier": "JBTS6 het / biallelic→MKS3"},
            {"variant": "p.Tyr78Cys",  "cdna": "c.233A>G",  "domain": "N-terminal FN-III", "ethnic": "South Asian", "tier": "JBTS6 mild"},
            {"variant": "c.1864+1G>A", "cdna": "Splice donor intron 17", "domain": "FN-III null (splice)", "ethnic": "European", "tier": "Null allele; JBTS6 het"},
            {"variant": "p.Trp628Cys", "cdna": "c.1884G>T", "domain": "TM/extracellular linker", "ethnic": "MENA", "tier": "JBTS6 + COACH hypomorphic"},
            {"variant": "p.Cys615Arg", "cdna": "c.1843T>C", "domain": "TM-proximal extracellular", "ethnic": "North African founder", "tier": "JBTS6 + COACH enriched"},
        ],
    }


def get_definitions():
    return {
        "gene": "TMEM67 (Transmembrane Protein 67; also MECKELIN, MKS3)",
        "omim_gene": "*609884",
        "omim_jbts6": "#610688",
        "omim_mks3": "#607361",
        "omim_nphp11": "#613550",
        "omim_coach": "#216360",
        "chromosome": "8q22.1",
        "protein": "995 aa; N-terminal FN-III extracellular domain + single transmembrane domain + short cytoplasmic tail (NPHP4/CC2D2A binding)",
        "pathway": "Transition Zone (TZ) membrane protein; part of NPHP-MKS-JBTS module; TZ gate integrity + Wnt/FZD extracellular sensing",
        "allele_tier_rule": (
            "Biallelic NULL → MKS3 lethal (encephalocele + polydactyly + PKD). "
            "One NULL + one HYPOMORPHIC → JBTS6 (MTS ± COACH ± NPHP11). "
            "Biallelic HYPOMORPHIC → JBTS6 mild / COACH / NPHP11. "
            "Homozygous p.Cys615Arg (North African founder) → JBTS6 + COACH enriched."
        ),
        "coach_syndrome": (
            "Cerebellar-Oculo-renal syndrome + Coloboma + Hepatic fibrosis. "
            "Caused by TMEM67 (~50%) and CC2D2A (~50%). Ductal plate malformation "
            "→ congenital hepatic fibrosis → portal hypertension → varices risk. "
            "Annual LFTs, liver ultrasound, endoscopy (varices screening) mandatory."
        ),
        "mks3": "Meckel-Gruber Syndrome Type 3 — biallelic null TMEM67; lethal; encephalocele, polydactyly, polycystic kidneys, oligohydramnios",
        "nphp11": "Nephronophthisis Type 11 — biallelic hypomorphic TMEM67; renal TIN; ESRD median ~18yr; No MTS in isolation",
        "mts": "Molar Tooth Sign — pathognomonic brain MRI finding; elongated superior cerebellar peduncles + cerebellar vermis hypoplasia/aplasia",
        "oma": "Oculomotor Apraxia — horizontal gaze initiation failure; compensatory head thrusting; JBTS6 ~60%",
        "fn_iii": "Fibronectin Type III domain — extracellular module on TMEM67; proposed Wnt/Frizzled ligand-binding; Ca²⁺-binding",
        "tm_domain": "Single transmembrane domain (aa 750-770) — anchors TMEM67 in TZ membrane; TM-adjacent alleles disrupt insertion",
        "tz_module": "NPHP-MKS-JBTS module — MKS1, TMEM67, CC2D2A, TMEM216, B9D1, B9D2; TZ gate integrity complex",
        "ductal_plate_malformation": "Congenital hepatic fibrosis pattern; abnormal bile duct proliferation; portal fibrosis → portal hypertension",
        "cys615arg_founder": "p.Cys615Arg (c.1843T>C) — most common TMEM67 allele in North African/Maghreb populations; COACH fibrosis enriched",
        "coach_tx": "Combined liver-kidney transplant — indicated in severe COACH + ESRD; both CURATIVE; neurological/retinal NOT corrected post-Tx",
        "therapy_status": "No disease-modifying therapy 2026; renal and liver transplant CURATIVE for organ endpoints; gene therapy pre-clinical",
        "inheritance": "Autosomal Recessive — biallelic loss-of-function; allele class governs disease tier (MKS3 → JBTS6 → NPHP11)",
        "frequency": "~5-10% of all Joubert syndrome; ~1/300,000-500,000 worldwide",
        "coach_frequency": "~30% of JBTS6 develop clinically significant hepatic fibrosis (COACH spectrum); ~40% if homozygous p.Cys615Arg",
        "north_african_enrichment": "p.Cys615Arg is the dominant TMEM67 allele in Moroccan, Algerian, and Tunisian patients with JBTS/COACH",
        "related_genes": "CC2D2A (JBTS9/COACH — only other JBTS gene causing COACH); MKS1 (MKS1/BBS13); TMEM216 (JBTS2); NPHP1 (JBTS4)",
    }
