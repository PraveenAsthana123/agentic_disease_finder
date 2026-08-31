"""
TCTN2 Joubert Syndrome Type 13 (JBTS13) — Autosomal Recessive / Tectonic-2 / Tectonic Complex / MKS8 Severe Tier
====================================================================================================================
Primary Gene : TCTN2 (*613846) — 12q24.31; 1424 aa; Tectonic-2.
               TCTN2 is one of three tectonic paralogues (TCTN1, TCTN2, TCTN3) that assemble as
               heterotrimers in the Tectonic complex at the ciliary transition zone (TZ).
               The Tectonic complex creates a lipid-enriched gate (cholesterol/sphingolipid) at the
               TZ membrane, controlling entry and exit of signalling proteins including SMO and GPCRs
               required for Hedgehog activation.
               TCTN2 protein domains:
               - Signal peptide (aa 1–22): ER targeting; secretory pathway entry
               - TCTN dimerisation domain (aa 23–300): heterodimer formation with TCTN1 and TCTN3;
                 TCTN1/TCTN2 heterodimer is the primary functional unit within the complex
               - Tectonic domain core (aa 300–950): TZ membrane scaffold; lipid gate organisation;
                 interaction surface with TMEM67, CC2D2A, MKS1 (MKS module bridge)
               - MKS module C-terminal (aa 950–1424): direct TMEM67/CC2D2A/MKS1 binding;
                 B9D1/TMEM231 interface; anchors Tectonic complex to MKS-module proteins
               TCTN2 LOF → Tectonic complex disassembled → lipid gate fails → SMO excluded from
               cilia → Hedgehog signalling failure → Molar Tooth Sign (MTS).

⚠ MKS8 TIER — TCTN2-SPECIFIC RULE:
   TCTN2 BIALLELIC NULL → MKS8 (Meckel-Gruber Syndrome 8, #615990) — PERINATAL LETHAL.
   This is the CRITICAL distinction from TCTN1 (JBTS11): TCTN1 biallelic null → JBTS11
   (live birth), but TCTN2 biallelic null → MKS8 (perinatal lethal, encephalocele + PKD).
   JBTS13 patients therefore require at least ONE non-null (hypomorphic) allele to survive
   to live birth — the allele-class tier rule is absolute for TCTN2.
   Key diagnostic rule: if WES identifies biallelic TCTN2 loss-of-function, neonatal
   survival indicates a hypomorphic component was missed — re-analyse for splicing/promoter
   effects creating residual function.

⚠ TCTN2 vs TCTN1 — SAME CHROMOSOME DISTINCTION:
   TCTN1 (12q24.11) and TCTN2 (12q24.31) are on the SAME chromosome 12 arm — 20 Mb apart.
   WES panels MUST distinguish them: TCTN1 biallelic null → JBTS11 (live birth, no MKS tier),
   TCTN2 biallelic null → MKS8 (perinatal lethal). Gene-panel report must name the specific
   TCTN paralogue. "TCTN" without number is diagnostically insufficient.

⚠ TECTONIC COMPLEX — HETEROTRIMER SCAFFOLD:
   TCTN1–TCTN2–TCTN3 assemble as a heterotrimer at the TZ. Each paralogue occupies a
   distinct structural position. TCTN2 provides the primary MKS module bridge (C-terminal
   interaction with TMEM67/CC2D2A/MKS1). Loss of TCTN2 → entire TZ MKS module anchor
   collapses → more severe ciliopathy phenotype than TCTN1 or TCTN3 loss alone.
   TCTN3 LOF → OFD4 (Oral-Facial-Digital Type 4) — distinct from JBTS13.

Disease OMIM : #614173 — Joubert Syndrome Type 13 (JBTS13)
               #615990 — Meckel-Gruber Syndrome 8 (MKS8) — severe/null allele tier
Chromosome   : 12q24.31
Inheritance  : Autosomal recessive — biallelic LOF; null/null → MKS8 lethal;
               null/hypomorphic or biallelic hypomorphic → JBTS13 (live birth)
Cohort size  : 40-patient educational cohort (seed 433)
"""

import random
import math

SEED = 433
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
    ('European',                       0.30),
    ('Middle Eastern / MENA',          0.28),   # Arg438Trp founder
    ('South Asian',                    0.20),   # Leu566Pro prevalent
    ('North African',                  0.12),   # Ala318Val founder (mild/NPHP-only)
    ('East Asian',                     0.05),
    ('Other / Unknown',                0.05),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('European')
eth_pool = eth_pool[:N]
rng.shuffle(eth_pool)

# Allele classes (TCTN2 MKS8 tier rule: biallelic null NOT in live-birth cohort):
#   null/strong hypomorphic (~30% — one truncating null + strong missense → JBTS13 severe)
#   null/mild hypomorphic (~25% — one null + mild missense → JBTS13 moderate)
#   missense/missense (~25% — biallelic moderate missense → JBTS13 moderate)
#   hypomorphic/hypomorphic (~20% — biallelic hypomorphic → JBTS13 mild/NPHP-only)
allele_classes = (
    ['null/strong-hypomorphic (severe JBTS13)'] * 12 +
    ['null/mild-hypomorphic (moderate JBTS13)'] * 10 +
    ['missense/missense (moderate JBTS13)']     * 10 +
    ['hypomorphic/hypomorphic (mild/NPHP-only)']* 8
)
rng.shuffle(allele_classes)

for i in range(N):
    eth  = eth_pool[i]
    acls = allele_classes[i]
    age_dx = rng.randint(1, 30) if 'severe' in acls or 'moderate' in acls else rng.randint(6, 90)

    # Phenotype probabilities by allele class
    # TCTN2 distinctive: hepatic CHF present, MKS8 null tier, polydactyly lower than KIF7
    if acls == 'null/strong-hypomorphic (severe JBTS13)':
        p_ataxia  = 0.95; p_hypo = 0.92; p_oma = 0.68; p_breath = 0.70
        p_ret = 0.48; p_poly = 0.18; p_renal = 0.42; p_hepatic = 0.28; p_id = 0.88
    elif acls == 'null/mild-hypomorphic (moderate JBTS13)':
        p_ataxia  = 0.88; p_hypo = 0.82; p_oma = 0.55; p_breath = 0.60
        p_ret = 0.38; p_poly = 0.12; p_renal = 0.30; p_hepatic = 0.18; p_id = 0.70
    elif acls == 'missense/missense (moderate JBTS13)':
        p_ataxia  = 0.78; p_hypo = 0.72; p_oma = 0.45; p_breath = 0.48
        p_ret = 0.28; p_poly = 0.08; p_renal = 0.22; p_hepatic = 0.12; p_id = 0.60
    else:  # hypomorphic/hypomorphic
        p_ataxia  = 0.55; p_hypo = 0.50; p_oma = 0.30; p_breath = 0.30
        p_ret = 0.15; p_poly = 0.04; p_renal = 0.18; p_hepatic = 0.06; p_id = 0.42

    # Specific allele by ethnicity + class
    if acls == 'null/strong-hypomorphic (severe JBTS13)':
        if eth == 'European':
            allele = 'Arg729*/Gly447Arg or c.1556+1G>A/Gly447Arg'
        elif eth in ('Middle Eastern / MENA',):
            allele = 'Arg729*/Arg438Trp or Arg876*/Arg438Trp'
        elif eth == 'South Asian':
            allele = 'Arg729*/Leu566Pro or c.1556+1G>A/Leu566Pro'
        elif eth == 'North African':
            allele = 'Arg876*/Arg438Trp'
        else:
            allele = 'Arg729*/Gly447Arg (pan-ethnic null)'
    elif acls == 'null/mild-hypomorphic (moderate JBTS13)':
        if eth in ('Middle Eastern / MENA',):
            allele = 'c.1556+1G>A/Arg438Trp'
        elif eth == 'South Asian':
            allele = 'Arg729*/Leu566Pro (milder) or Arg876*/Ala318Val'
        elif eth == 'North African':
            allele = 'Arg729*/Ala318Val'
        elif eth == 'East Asian':
            allele = 'c.1556+1G>A/Tyr1192Cys'
        else:
            allele = 'Arg729*/Gly447Arg (European moderate)'
    elif acls == 'missense/missense (moderate JBTS13)':
        if eth in ('Middle Eastern / MENA',):
            allele = 'Arg438Trp/Arg438Trp (MENA founder homozygous)'
        elif eth == 'South Asian':
            allele = 'Leu566Pro/Gly447Arg or Leu566Pro/Leu566Pro'
        elif eth == 'North African':
            allele = 'Arg438Trp/Ala318Val or Gly447Arg/Ala318Val'
        elif eth == 'East Asian':
            allele = 'Tyr1192Cys/Gly447Arg'
        else:
            allele = 'Gly447Arg/Arg438Trp (European compound het missense)'
    else:  # hypomorphic
        if eth == 'North African':
            allele = 'Ala318Val/Ala318Val (North African mild founder homozygous)'
        elif eth == 'East Asian':
            allele = 'Tyr1192Cys/Tyr1192Cys or Tyr1192Cys/Ala318Val'
        elif eth in ('Middle Eastern / MENA',):
            allele = 'Arg438Trp/Ala318Val (mild hypomorphic compound)'
        else:
            allele = 'Gly447Arg/Ala318Val (hypomorphic — mild/NPHP-only)'

    esrd_age = rng.randint(18, 38) if rng.random() < p_renal else None

    patients.append({
        "id":           f"JBTS13-{i+1:03d}",
        "sex":          rng.choice(["M", "M", "F"]),   # equal sex ratio (AR)
        "ethnicity":    eth,
        "allele_class": acls,
        "allele":       allele,
        "age_dx_yr":    round(age_dx / 12, 1),
        "ataxia":       rng.random() < p_ataxia,
        "hypotonia":    rng.random() < p_hypo,
        "oma":          rng.random() < p_oma,
        "retinal":      rng.random() < p_ret,
        "polydactyly":  rng.random() < p_poly,
        "renal":        rng.random() < p_renal,
        "esrd_age":     esrd_age,
        "hepatic":      rng.random() < p_hepatic,
        "id_":          rng.random() < p_id,
        "breathing":    rng.random() < p_breath,
    })

# ── aggregate phenotype counts ────────────────────────────────────────────────
n_mts       = N   # 100% — MTS pathognomonic for all JBTS
n_ataxia    = sum(1 for p in patients if p["ataxia"])
n_hypotonia = sum(1 for p in patients if p["hypotonia"])
n_oma       = sum(1 for p in patients if p["oma"])
n_retinal   = sum(1 for p in patients if p["retinal"])
n_polydactyly = sum(1 for p in patients if p["polydactyly"])
n_renal     = sum(1 for p in patients if p["renal"])
n_hepatic   = sum(1 for p in patients if p["hepatic"])
n_id        = sum(1 for p in patients if p["id_"])
n_breathing = sum(1 for p in patients if p["breathing"])


# ── API builders ──────────────────────────────────────────────────────────────
def get_overview():
    allele_dist = {}
    for p in patients:
        allele_dist[p["allele_class"]] = allele_dist.get(p["allele_class"], 0) + 1

    return {
        "gene":              "TCTN2",
        "disease":           "Joubert Syndrome Type 13 / Meckel-Gruber Syndrome 8 (JBTS13/MKS8) — Autosomal Recessive",
        "omim_gene":         "613846",
        "omim_disease_jbts13": "614173",
        "omim_disease_mks8": "615990",
        "chromosome":        "12q24.31",
        "protein":           "1424 aa — Signal peptide (aa 1-22) / TCTN dimerisation domain (aa 23-300; TCTN1/TCTN3 heterotrimer) / Tectonic domain core (aa 300-950; TZ lipid gate scaffold; TMEM67/CC2D2A/MKS1 bridge) / MKS module C-terminal (aa 950-1424; B9D1/TMEM231 interface)",
        "inheritance":       "Autosomal recessive — biallelic LOF; null/null → MKS8 (perinatal lethal); null/hypomorphic or biallelic hypomorphic → JBTS13 (live birth)",
        "prevalence":        "~2–3% of all Joubert syndrome; ~1/600,000–1,200,000 worldwide",
        "hallmark":          "Molar Tooth Sign (MTS) — 100%; TCTN2 Tectonic complex TZ lipid-gate failure; MKS8 lethal tier (biallelic null); hepatic CHF ~18%; retinal ~38%; NO corpus callosum anomaly; lower polydactyly than KIF7/JBTS12",
        "tctn2_function_pearl": (
            "TCTN2 is one of three tectonic paralogues (TCTN1, TCTN2, TCTN3) that assemble as a "
            "heterotrimer in the Tectonic complex at the ciliary transition zone (TZ). The Tectonic "
            "complex creates a cholesterol- and sphingolipid-enriched lipid gate at the TZ membrane, "
            "controlling ciliary entry/exit of signalling proteins. TCTN2 provides the primary MKS "
            "module bridge: its C-terminal domain (aa 950-1424) directly contacts TMEM67, CC2D2A, "
            "and MKS1 — anchoring the Tectonic heterotrimer to the broader MKS-module scaffold. "
            "TCTN2 LOF → Tectonic complex disassembled → lipid gate fails → SMO excluded from cilia "
            "→ Hedgehog signalling failure → cerebellar vermis hypoplasia → Molar Tooth Sign (MTS). "
            "The stronger MKS bridge role of TCTN2 versus TCTN1 explains why TCTN2 null is MKS8 "
            "(lethal) while TCTN1 null is JBTS11 (live birth) — a critical paralogue-specific distinction."
        ),
        "mks8_tier_pearl": (
            "TCTN2 biallelic null → MKS8 (Meckel-Gruber Syndrome 8, #615990) — perinatal lethal. "
            "This is the defining distinction from TCTN1 (JBTS11): TCTN1 null → JBTS11 (live birth), "
            "TCTN2 null → MKS8 (lethal). JBTS13 patients ALWAYS have at least one hypomorphic "
            "(partially functional) TCTN2 allele. The allele-class tier rule is absolute: "
            "biallelic TCTN2 truncating nulls → MKS8, not JBTS13. "
            "If WES reports biallelic TCTN2 loss-of-function in a live-birth patient, a "
            "hypomorphic element (deep intronic/promoter variant) was missed — mandatory re-analysis. "
            "This MKS8 tier places TCTN2 alongside CEP290 (MKS4), TMEM67 (MKS3), RPGRIP1L (MKS5), "
            "CC2D2A (MKS6) in the MKS-tier JBTS gene set."
        ),
        "tctn_paralogue_pearl": (
            "TCTN1 (12q24.11) and TCTN2 (12q24.31) are 20 Mb apart on chromosome 12q — same arm, "
            "different loci. Gene panels MUST report the specific TCTN paralogue; 'TCTN mutation' "
            "is diagnostically insufficient. TCTN3 (OMIM 613847) causes OFD4 (Oral-Facial-Digital "
            "Syndrome Type 4) — a distinct condition, NOT Joubert syndrome. The three TCTN "
            "paralogues have identical disease-tier differentiation: TCTN1 → JBTS11 only, "
            "TCTN2 → JBTS13/MKS8, TCTN3 → OFD4. This heterotrimer specificity makes TCTN2 "
            "the most severe tectonic subunit for reproductive counselling."
        ),
        "first_description": "Garcia-Gonzalo FR et al., Nat Genet 2011 — Tectonic complex identified; TCTN1/TCTN2/TCTN3 as TZ lipid-gate scaffold; TCTN2/3 mutations in Joubert syndrome patients",
        "gene_summary": (
            "TCTN2 (Tectonic-2, 1424 aa) is a TZ membrane scaffold protein forming a heterotrimer "
            "with TCTN1 and TCTN3 in the Tectonic complex. The Tectonic complex creates a "
            "cholesterol/sphingolipid-enriched lipid gate at the TZ, controlling ciliary import/export "
            "of signalling proteins. TCTN2 provides the primary MKS module bridge via its C-terminal "
            "domain, directly binding TMEM67, CC2D2A, and MKS1. TCTN2 LOF → complex collapse → "
            "lipid gate failure → SMO exclusion → Hedgehog failure → MTS. The MKS8 severe tier "
            "(biallelic null → lethal) is unique to TCTN2 among tectonic paralogues. JBTS13 patients "
            "require at least one hypomorphic allele. Clinical features include MTS (100%), "
            "cerebellar ataxia (~87%), neonatal hypotonia (~82%), hepatic CHF (~18%), retinal "
            "rod-cone dystrophy (~38%), and NPHP-like renal TIN (~30%). Polydactyly is less "
            "prominent than KIF7/JBTS12 (~12%); no corpus callosum anomaly (unlike KIF7)."
        ),
        "cohort_size": N,
        "kpis": [
            {"label": "Molar Tooth Sign",         "value": f"{n_mts}/{N} (100%)",                    "color": "#1a237e"},
            {"label": "Cerebellar Ataxia",         "value": f"{n_ataxia}/{N} ({_pct(n_ataxia)}%)",    "color": "#1565c0"},
            {"label": "Neonatal Hypotonia",        "value": f"{n_hypotonia}/{N} ({_pct(n_hypotonia)}%)", "color": "#283593"},
            {"label": "Oculomotor Apraxia",        "value": f"{n_oma}/{N} ({_pct(n_oma)}%)",           "color": "#4527a0"},
            {"label": "Intellectual Disability",   "value": f"{n_id}/{N} ({_pct(n_id)}%)",             "color": "#6a1b9a"},
            {"label": "Breathing Dysreg.",         "value": f"{n_breathing}/{N} ({_pct(n_breathing)}%)", "color": "#880e4f"},
            {"label": "Retinal Dystrophy",         "value": f"{n_retinal}/{N} ({_pct(n_retinal)}%)",   "color": "#b71c1c"},
            {"label": "Renal (NPHP-like TIN)",     "value": f"{n_renal}/{N} ({_pct(n_renal)}%)",       "color": "#00695c"},
            {"label": "Hepatic CHF",               "value": f"{n_hepatic}/{N} ({_pct(n_hepatic)}%)",   "color": "#f57f17"},
            {"label": "Polydactyly",               "value": f"{n_polydactyly}/{N} ({_pct(n_polydactyly)}%)", "color": "#e65100"},
            {"label": "MKS8 Tier (biallelic null)","value": "MKS8 lethal; JBTS13 = null+hypomorphic", "color": "#4a148c"},
            {"label": "No Corpus Callosum Anomaly","value": "TCTN2 distinctive (vs KIF7/JBTS12)",     "color": "#37474f"},
        ],
        "allele_class_distribution": [
            {"allele_class": k, "count": v, "pct": _pct(v)} for k, v in allele_dist.items()
        ],
        "phenotype_summary": {
            "mts_pct":         100,
            "ataxia_pct":      _pct(n_ataxia),
            "hypotonia_pct":   _pct(n_hypotonia),
            "oma_pct":         _pct(n_oma),
            "retinal_pct":     _pct(n_retinal),
            "polydactyly_pct": _pct(n_polydactyly),
            "renal_pct":       _pct(n_renal),
            "hepatic_pct":     _pct(n_hepatic),
            "id_pct":          _pct(n_id),
            "breathing_pct":   _pct(n_breathing),
        },
    }


def get_breakdown():
    allele_dist = {}
    for p in patients:
        allele_dist[p["allele_class"]] = allele_dist.get(p["allele_class"], 0) + 1

    eth_counts = {}
    for p in patients:
        eth_counts[p["ethnicity"]] = eth_counts.get(p["ethnicity"], 0) + 1

    key_variants = [
        {
            "variant":      "p.Arg438Trp (c.1312C>T)",
            "domain":       "Tectonic domain — entry boundary",
            "effect":       "Arg→Trp at tectonic domain entry; disrupts TCTN1/TCTN2 heterodimer interface; reduced TZ membrane targeting; partial complex assembly",
            "population":   "Middle Eastern / MENA — founder allele",
            "allele_class": "hypomorphic or null/hypomorphic",
            "severity":     "Moderate JBTS13 (homozygous: mild–moderate; compound with null: moderate–severe)",
            "renal_risk":   "~30%",
            "hepatic_risk": "~18%",
            "omim_note":    "MENA founder; most common TCTN2 allele in Middle Eastern populations; variable expressivity",
        },
        {
            "variant":      "p.Gly447Arg (c.1339G>A)",
            "domain":       "Tectonic domain — entry",
            "effect":       "Gly→Arg disrupts critical Gly-kink in tectonic domain; beta-strand geometry altered; TCTN1 binding interface reduced",
            "population":   "Pan-ethnic",
            "allele_class": "null/missense or missense/missense",
            "severity":     "Moderate–severe JBTS13",
            "renal_risk":   "~38%",
            "hepatic_risk": "~22%",
            "omim_note":    "Tectonic domain beta-strand disruption; pan-ethnic moderate/severe JBTS13",
        },
        {
            "variant":      "p.Leu566Pro (c.1697T>C)",
            "domain":       "Tectonic domain core",
            "effect":       "Leu→Pro in tectonic core alpha-helix; Pro breaks alpha-helix → domain unfolding; lipid-gate interaction surface disrupted",
            "population":   "South Asian",
            "allele_class": "null/strong-hypomorphic or missense/missense",
            "severity":     "Moderate–severe JBTS13",
            "renal_risk":   "~42%",
            "hepatic_risk": "~25%",
            "omim_note":    "South Asian tectonic core allele; alpha-helix disruption → moderate–severe JBTS13",
        },
        {
            "variant":      "p.Arg729Ter (c.2185C>T)",
            "domain":       "Tectonic domain C-terminal — truncating null",
            "effect":       "Premature stop; NMD → null; entire MKS module C-terminal (aa 730–1424) lost; TMEM67/CC2D2A/MKS1 interaction abolished",
            "population":   "European",
            "allele_class": "null/hypomorphic (one null allele → JBTS13 if partnered with hypomorph; null/null → MKS8 lethal)",
            "severity":     "MKS8 (biallelic) / JBTS13 severe (with hypomorphic partner)",
            "renal_risk":   "~50%",
            "hepatic_risk": "~30%",
            "omim_note":    "European truncating null; critical MKS8/JBTS13 distinction; MKS module C-tail completely lost",
        },
        {
            "variant":      "c.1556+1G>A (splice donor intron 13)",
            "domain":       "Tectonic domain — splice null",
            "effect":       "Splice donor disruption; exon 13 skipping → frameshift → NMD; null allele; tectonic core ablated",
            "population":   "European / pan-ethnic",
            "allele_class": "null (→ MKS8 if biallelic; JBTS13 severe if + hypomorphic partner)",
            "severity":     "MKS8 (biallelic null) / JBTS13 severe",
            "renal_risk":   "~50%",
            "hepatic_risk": "~28%",
            "omim_note":    "European splice null; most common European TCTN2 null allele; MKS8 when biallelic",
        },
        {
            "variant":      "p.Ala318Val (c.953C>T)",
            "domain":       "N-terminal tectonic domain boundary",
            "effect":       "Ala→Val at N-terminal tectonic domain boundary; mildly reduces TCTN1/TCTN2 interface affinity; partial complex assembly; NPHP-predominant",
            "population":   "North African — founder allele",
            "allele_class": "hypomorphic/hypomorphic → mild JBTS13 / NPHP-only",
            "severity":     "Mild JBTS13 or NPHP-only (biallelic Ala318Val: renal-predominant)",
            "renal_risk":   "~22%",
            "hepatic_risk": "~8%",
            "omim_note":    "North African founder; mildest TCTN2 allele; biallelic → mild JBTS13 or isolated NPHP phenotype",
        },
        {
            "variant":      "p.Tyr1192Cys (c.3575A>G)",
            "domain":       "C-terminal MKS module — TCTN2/TCTN3 interface",
            "effect":       "Tyr→Cys at TCTN2/TCTN3 interaction interface within MKS C-terminal; aberrant disulfide bridge; TCTN3 binding reduced",
            "population":   "East Asian",
            "allele_class": "missense/missense or null/mild-hypomorphic",
            "severity":     "Moderate JBTS13",
            "renal_risk":   "~28%",
            "hepatic_risk": "~15%",
            "omim_note":    "East Asian C-terminal interface; TCTN3 binding partner disruption; moderate JBTS13",
        },
        {
            "variant":      "p.Arg876Ter (c.2626C>T)",
            "domain":       "MKS module C-terminal — truncating null",
            "effect":       "Premature stop; NMD; MKS C-terminal lost (aa 877–1424); TMEM67/CC2D2A/MKS1 interaction abolished; MKS8 when biallelic",
            "population":   "Pan-ethnic",
            "allele_class": "null (→ MKS8 biallelic; → JBTS13 severe with hypomorphic partner)",
            "severity":     "MKS8 (biallelic) / JBTS13 severe",
            "renal_risk":   "~48%",
            "hepatic_risk": "~28%",
            "omim_note":    "Pan-ethnic MKS C-terminal null; biallelic → MKS8 lethal; compound with hypomorphic → severe JBTS13",
        },
    ]

    domain_matrix = [
        {
            "domain":   "Signal peptide (aa 1-22)",
            "function": "ER targeting; secretory pathway entry",
            "variants": "—",
            "phenotype":"No JBTS13 variants reported (signal peptide mutations → protein absent from ER)",
            "severity": "Expected lethal (MKS8 tier if biallelic null)",
        },
        {
            "domain":   "TCTN dimerisation (aa 23-300)",
            "function": "TCTN1/TCTN2/TCTN3 heterotrimer assembly; N-terminal TCTN1 interface",
            "variants": "Arg438Trp (boundary); Ala318Val (N-term, hypomorphic)",
            "phenotype":"Heterodimer affinity reduction; partial complex assembly; variable JBTS13",
            "severity": "Mild–moderate (partial function retained)",
        },
        {
            "domain":   "Tectonic domain core (aa 300-950)",
            "function": "TZ membrane scaffold; cholesterol/sphingolipid lipid gate organisation; TMEM67/CC2D2A/MKS1 primary interaction surface",
            "variants": "Gly447Arg; Leu566Pro; c.1556+1G>A (splice); Arg729* (truncating null)",
            "phenotype":"Lipid gate disassembly; SMO exclusion; Hedgehog failure; MTS; retinal + renal + hepatic features",
            "severity": "Moderate–severe; truncating → MKS8 (null tier)",
        },
        {
            "domain":   "MKS module C-terminal (aa 950-1424)",
            "function": "TMEM67/CC2D2A/MKS1 binding; B9D1/TMEM231 interface; anchors Tectonic complex to MKS-module proteins",
            "variants": "Tyr1192Cys; Arg876* (null); full C-terminal truncations",
            "phenotype":"MKS module anchor lost; TZ gate fully collapses; hepatic + renal + retinal full penetrance",
            "severity": "Truncating null → MKS8 lethal; missense → moderate JBTS13",
        },
    ]

    pathway_steps = [
        {
            "step":    "1. Tectonic complex assembly",
            "normal":  "TCTN1 + TCTN2 + TCTN3 form heterotrimer at TZ; TCTN2 provides MKS bridge (C-terminal → TMEM67/CC2D2A/MKS1)",
            "loss":    "TCTN2 LOF → heterotrimer cannot form; TCTN1 and TCTN3 cannot independently scaffold TZ gate",
            "outcome": "TZ lipid gate disassembled",
        },
        {
            "step":    "2. TZ lipid gate formation",
            "normal":  "Cholesterol + sphingolipid enrichment at TZ membrane; diffusion barrier created for SMO, GPCRs, IFT",
            "loss":    "Lipid gate absent → non-ciliary membrane proteins freely enter cilia; SMO accumulation pattern disrupted",
            "outcome": "Ciliary compartment identity lost",
        },
        {
            "step":    "3. SMO ciliary entry",
            "normal":  "SMO translocates into cilia upon Shh binding PTCH1; TZ lipid gate controls SMO entry timing",
            "loss":    "Lipid gate absent → SMO excluded constitutively or unable to localise correctly → Hedgehog signal not transmitted",
            "outcome": "Hedgehog activation failure",
        },
        {
            "step":    "4. GLI processing failure",
            "normal":  "SMO in cilia → GLI2/3 activator processing at ciliary tip → Hedgehog transcriptional response",
            "loss":    "SMO absent from cilia → GLI activators not processed → Hedgehog pathway silent",
            "outcome": "Cerebellar vermis hypoplasia → Molar Tooth Sign (MTS)",
        },
        {
            "step":    "5. Multi-organ cilia dysfunction",
            "normal":  "Photoreceptor connecting cilium; kidney tubular cilia; cholangiocyte bile duct cilia all require TZ integrity",
            "loss":    "TCTN2 LOF in all ciliated cells → retinal rod-cone dystrophy; NPHP-like TIN; ductal plate malformation (CHF)",
            "outcome": "Multi-organ ciliopathy (retinal + renal + hepatic)",
        },
    ]

    management = [
        {
            "intervention":    "Brain MRI — Molar Tooth Sign",
            "timing":          "At first clinical suspicion; repeat if incomplete",
            "rationale":       "MTS pathognomonic (100%); MRI essential for JBTS13 vs MKS8 distinction (post-neonatal survival implies hypomorphic allele); cerebellar vermis hypoplasia grading",
            "level":           "Level A (diagnostic — mandatory)",
        },
        {
            "intervention":    "TCTN2 molecular sequencing + del/dup analysis",
            "timing":          "At MTS diagnosis; gene panel including TCTN1/TCTN2/TCTN3",
            "rationale":       "TCTN2 vs TCTN1 (same chr 12 arm) must be distinguished; MKS8 tier risk counselling; allele-class tier (null/hypomorphic vs biallelic null) critical for prognosis",
            "level":           "Level A (molecular — mandatory)",
        },
        {
            "intervention":    "Ophthalmology — ERG + fundus",
            "timing":          "Within first year; annually thereafter",
            "rationale":       "Retinal rod-cone dystrophy in ~38%; higher than KIF7/JBTS12 (~18%); photoreceptor connecting cilium TCTN2-dependent; annual ERG for progression tracking",
            "level":           "Level A (annual surveillance)",
        },
        {
            "intervention":    "Renal ultrasound + eGFR + urinalysis",
            "timing":          "Annual from diagnosis",
            "rationale":       "NPHP-like TIN in ~30%; ESRD median ~22yr; ACE-I for proteinuria; renal transplant curative (cell-autonomous AR ciliopathy); close monitoring essential",
            "level":           "Level A (annual)",
        },
        {
            "intervention":    "Liver function tests + hepatic ultrasound",
            "timing":          "At diagnosis; annually; GI referral if CHF features",
            "rationale":       "Hepatic CHF in ~18% (ductal plate malformation; portal HTN; varices risk); higher than KIF7/JBTS12; cholangiocyte cilia TZ TCTN2-dependent; liver transplant if decompensated CHF",
            "level":           "Level A (hepatic surveillance)",
        },
        {
            "intervention":    "Neurodevelopmental + cognitive assessment",
            "timing":          "From 12 months; annually",
            "rationale":       "Intellectual disability in ~70%; early intervention maximises outcomes; cerebellar ataxia ~87%; OMA ~55%",
            "level":           "Level A",
        },
        {
            "intervention":    "Physiotherapy — cerebellar ataxia",
            "timing":          "From diagnosis, ongoing",
            "rationale":       "Cerebellar ataxia in ~87%; balance + core + gait training; SARA tracking",
            "level":           "Level A",
        },
        {
            "intervention":    "Polysomnography / sleep study",
            "timing":          "At diagnosis; repeat if breathing concerns",
            "rationale":       "Breathing dysregulation ~60%; brainstem TZ cilia involvement; episodic apnea/hyperpnea monitoring",
            "level":           "Level B",
        },
        {
            "intervention":    "Genetic counselling — JBTS13 vs MKS8 tier",
            "timing":          "At molecular diagnosis",
            "rationale":       "AR biallelic — 25% sibling recurrence; TCTN2 null/null → MKS8 lethal (critical distinction from TCTN1/JBTS11); families must understand MKS8 lethal risk when both parents carry TCTN2 null alleles; PGT-M/prenatal diagnosis available",
            "level":           "Mandatory counselling",
        },
        {
            "intervention":    "TCTN2 vs TCTN1 paralogue distinction",
            "timing":          "At molecular diagnosis",
            "rationale":       "TCTN1 (12q24.11) null → JBTS11 (no MKS tier); TCTN2 (12q24.31) null → MKS8 (lethal); same chromosome arm, different disease tier. Gene report MUST specify TCTN2 (not generic 'TCTN'). Reproductive counselling entirely different depending on paralogue.",
            "level":           "Level A (molecular diagnostic)",
        },
    ]

    patient_rows = []
    for p in patients:
        patient_rows.append([
            p["id"], p["sex"], p["ethnicity"],
            f"{p['age_dx_yr']} yr",
            p["allele_class"].split("(")[0].strip(),
            p["allele"],
            "Yes" if p["ataxia"]    else "No",
            "Yes" if p["oma"]       else "No",
            "Yes" if p["retinal"]   else "No",
            "Yes" if p["renal"]     else "No",
            f"ESRD ~{p['esrd_age']}yr" if p["esrd_age"] else "—",
            "Yes" if p["hepatic"]   else "No",
            "Yes" if p["polydactyly"] else "No",
        ])

    return {
        "cohort_size":            N,
        "ethnicity_distribution": [{"ethnicity": k, "count": v, "pct": _pct(v)}
                                   for k, v in sorted(eth_counts.items(), key=lambda x: -x[1])],
        "allele_distribution":    [{"allele_class": k, "count": v, "pct": _pct(v)}
                                   for k, v in sorted(allele_dist.items(), key=lambda x: -x[1])],
        "key_variants":           key_variants,
        "domain_phenotype_matrix":domain_matrix,
        "pathway_steps":          pathway_steps,
        "management":             management,
        "patient_table":          patient_rows,
        "phenotype_counts": {
            "mts":         n_mts,
            "ataxia":      n_ataxia,
            "hypotonia":   n_hypotonia,
            "oma":         n_oma,
            "retinal":     n_retinal,
            "polydactyly": n_polydactyly,
            "renal":       n_renal,
            "hepatic":     n_hepatic,
            "id":          n_id,
            "breathing":   n_breathing,
        },
    }


def get_definitions():
    return {
        "tctn2_gene":          "TCTN2 gene (12q24.31; *613846) — 1424 aa Tectonic-2; signal peptide / TCTN dimerisation domain (TCTN1/TCTN3 heterotrimer) / tectonic domain core (TZ lipid gate; TMEM67/CC2D2A/MKS1 bridge) / MKS module C-terminal (B9D1/TMEM231 interface)",
        "jbts13":              "Joubert Syndrome Type 13 (OMIM #614173) — TCTN2 null/hypomorphic compound; AR; MTS + cerebellar ataxia + retinal + renal + hepatic CHF; lower polydactyly than KIF7; MKS8 severe tier (biallelic null → lethal)",
        "mks8":                "Meckel-Gruber Syndrome 8 (OMIM #615990) — TCTN2 biallelic null; perinatal lethal; encephalocele + PKD + polydactyly; all null alleles; JBTS13 patients ALWAYS carry ≥1 hypomorphic allele",
        "tectonic_complex":    "TCTN1 + TCTN2 + TCTN3 heterotrimer assembled at the ciliary transition zone (TZ); creates a cholesterol/sphingolipid-enriched lipid gate controlling protein entry/exit; TCTN2 provides primary MKS module bridge (C-terminal → TMEM67/CC2D2A/MKS1)",
        "tz_lipid_gate":       "Transition Zone lipid gate — cholesterol- and sphingolipid-enriched membrane domain at the TZ base of cilia; acts as a diffusion barrier; Tectonic complex, B9D1/TMEM231, and NPHP-module are co-assembled; SMO and other signalling GPCRs regulated by this gate",
        "mks_module_bridge":   "TCTN2 C-terminal domain (aa 950-1424) directly contacts TMEM67 (MKS3), CC2D2A (MKS6), and MKS1 — anchoring the Tectonic complex to the MKS-module proteins at the TZ; this bridge function is why TCTN2 null is more severe (MKS8) than TCTN1 null (JBTS11)",
        "tctn1_distinction":   "TCTN1 (12q24.11) vs TCTN2 (12q24.31) — same chromosome arm, 20 Mb apart; TCTN1 null → JBTS11 (live birth, no MKS tier); TCTN2 null → MKS8 (perinatal lethal); gene panels MUST name the specific paralogue; TCTN3 → OFD4 (distinct)",
        "tctn3_distinction":   "TCTN3 (OMIM 613847) — third tectonic paralogue; LOF → OFD4 (Oral-Facial-Digital Syndrome Type 4), NOT Joubert syndrome; each tectonic gene has a distinct primary disease",
        "heterotrimer_rule":   "The TCTN1/TCTN2/TCTN3 heterotrimer requires all three subunits for assembly; TCTN2 LOF is most severe because TCTN2 provides the MKS module bridge; without TCTN2, TCTN1 and TCTN3 cannot independently scaffold the TZ gate even if individually present",
        "mts":                 "Molar Tooth Sign — pathognomonic brain MRI finding; elongated superior cerebellar peduncles + cerebellar vermis hypoplasia; 100% in JBTS13",
        "retinal_tctn2":       "Rod-cone dystrophy in ~38% of JBTS13 — higher than KIF7/JBTS12 (~18%); TCTN2 expression in photoreceptor connecting cilium; TZ lipid gate failure → outer segment formation impaired; annual ERG required; lower than MKS-tier CEP290 (~50%)",
        "renal_tctn2":         "NPHP-like tubulointerstitial nephritis (TIN) in ~30% of JBTS13; ESRD median ~22yr; renal transplant curative (cell-autonomous AR ciliopathy); eGFR/urinalysis annual surveillance",
        "hepatic_chf":         "Congenital Hepatic Fibrosis (CHF) in ~18% of JBTS13 — ductal plate malformation; portal hypertension; varices risk; cholangiocyte bile duct cilia require TZ integrity; liver transplant for decompensated CHF; higher than KIF7/JBTS12 (~5%)",
        "polydactyly_tctn2":   "Post-axial polydactyly in ~12% of JBTS13 — lower than KIF7/JBTS12 (~35-45%); TCTN2 has modest Hedgehog/digit patterning role; polydactyly when present is usually unilateral hand",
        "no_cc_anomaly":       "TCTN2/JBTS13 has NO corpus callosum anomaly — distinguishes from KIF7/JBTS12 (20-25% CC anomaly); brain MRI normal CC morphology in JBTS13",
        "allele_tier_rule":    "TCTN2 allele-class tier: biallelic null → MKS8 (perinatal lethal, NOT in live-birth cohort); null + strong hypomorphic → JBTS13 severe; null + mild hypomorphic → JBTS13 moderate; biallelic missense → JBTS13 moderate; biallelic hypomorphic → JBTS13 mild/NPHP-only",
        "arg438trp":           "p.Arg438Trp (c.1312C>T) — MENA founder allele; tectonic domain entry; most common TCTN2 allele in Middle Eastern populations; hypomorphic (partial complex assembly); variable expressivity from mild to moderate JBTS13",
        "garcia_gonzalo_2011": "Garcia-Gonzalo FR et al., Nat Genet 2011 — Tectonic complex identified at TZ; TCTN2/TCTN3 mutations found in Joubert syndrome families; established Tectonic complex as lipid gate scaffold essential for ciliary compartment identity",
        "inheritance":         "Autosomal recessive — biallelic LOF; 25% sibling recurrence; carriers phenotypically normal; MKS8 risk when BOTH parents carry null TCTN2 alleles (25% MKS8 lethal, 50% JBTS13, 25% unaffected); PGT-M and prenatal diagnosis (CVS/amniocentesis) available",
        "frequency":           "~2–3% of all Joubert syndrome; ~1/600,000–1,200,000 worldwide",
        "related_genes":       "Tectonic complex: TCTN1 (JBTS11 — no MKS tier) · TCTN3 (OFD4 — not Joubert) · MKS-module partners: TMEM67 (JBTS6/MKS3) · CC2D2A (JBTS9/MKS6) · MKS1 (MKS1) · B9D1/TMEM231 · NPHP-module: NPHP1/4/8 · Other MKS-tier JBTS: CEP290 (JBTS5/MKS4) · RPGRIP1L (JBTS7/MKS5)",
        "therapy_status":      "No disease-modifying therapy 2026 for TCTN2/JBTS13; renal transplant curative for ESRD (cell-autonomous); liver transplant for decompensated CHF; retinal — symptomatic support; no Hedgehog-pathway agonist trials yet for TCTN2/TZ lipid gate; gene therapy (AAV-TCTN2) conceptual",
    }
