"""
RPGRIP1L Joubert Syndrome Type 7 (JBTS7) — FTM / NPHP8 / MKS5
=============================================================================
Primary Gene : RPGRIP1L (*610937) — 16q12.2; 1315 aa; Retinitis Pigmentosa GTPase
               Regulator-Interacting Protein 1-Like (also FTM [fantom] in mouse; NPHP8)
               RPGRIP1L is a TRANSITION ZONE Y-LINK SCAFFOLD PROTEIN.
               RPGRIP1L contains: N-terminal coiled-coil (CC) domain (aa 1–450),
               central RPGRIP1 Homology (RH) domain (aa 450–960; NPHP4-binding),
               and C-terminal C2 domain (aa 960–1315; membrane association, RPGR binding).
               RPGRIP1L anchors TZ Y-link spokes to the ciliary axoneme; its loss disrupts
               Y-link integrity → GPCR/Smo exclusion from cilium → Hedgehog failure
               → Molar Tooth Sign (MTS), cerebellar vermis hypoplasia.
               Unique: RPGRIP1L is the only TZ Y-link protein with a dual NPHP4-RPGR
               binding capacity (RH + C2), making retinal involvement variable but possible.
               ~2–3% of all Joubert syndrome cases worldwide.
Disease OMIM : #611560 — Joubert Syndrome 7 (JBTS7)
               Also: #611561 = MKS5 (Meckel-Gruber Syndrome 5) · lethal, biallelic null
               Also: #613568 = NPHP8 (Nephronophthisis 8) · mild, biallelic hypomorphic
Chromosome   : 16q12.2
Inheritance  : Autosomal Recessive — biallelic LOF; ALLELE CLASS GOVERNS DISEASE TIER
Prevalence   : ~2–3% of all Joubert syndrome cases
               ~1/1,000,000–2,000,000 worldwide (JBTS7 phenotype)

⚠ KEY DIAGNOSTIC PEARL — EUROPEAN Ala229Thr FOUNDER ALLELE:
p.Ala229Thr (c.685G>A) is the MOST COMMON RPGRIP1L allele in European populations —
found homozygous or as compound het with a second hypomorphic allele. It is the most
frequent cause of JBTS7 in European cohorts (Northern European enrichment). Biallelic
p.Ala229Thr causes mild JBTS7 WITHOUT MKS5 (pure hypomorphic). This allele is present
at low frequency in gnomAD and can be mistaken for a benign variant when found in trans
with another hypomorphic allele; clinical context (MTS on MRI) is essential for correct
interpretation. Never call biallelic p.Ala229Thr as MKS5-tier: allele class governs.

⚠ ALLELE-CLASS TIER RULE (RPGRIP1L):
Biallelic NULL (two truncating/splice-null): MKS5 (Meckel-Gruber lethal — encephalocele +
   polydactyly + polycystic kidneys + oligohydramnios; perinatal lethal)
One NULL + one HYPOMORPHIC (e.g. Ala229Thr): JBTS7 (Molar Tooth Sign ± renal ± retinal)
Biallelic HYPOMORPHIC (Ala229Thr/Ala229Thr or missense/missense): JBTS7 mild or NPHP8 (renal dominant)
p.Asn694Ser compound het: NPHP8 (renal-dominant, South Asian; no or mild MTS)

Protein Structure — RPGRIP1L / FTM (1315 aa; TZ Y-link scaffold)
------------------------------------------------------------------
Domain 1: N-terminal coiled-coil CC (aa 1–450)       — NPHP1-like CC; PCARE binding
           Ala229Thr — European founder hypomorphic; CC fold partial destabilisation
           Trp519Ter — truncating null; biallelic → MKS5 (boundary CC/RH)
Domain 2: RPGRIP1 Homology (RH) domain (aa 450–960)  — NPHP4 binding; TZ Y-link anchoring
           Arg1174Gln — RH C-terminal region; NPHP4 interface; MENA; JBTS7/NPHP8
           Asn694Ser  — RH core; South Asian; hypomorphic; NPHP8 renal-dominant
           Leu821Pro  — RH-C2 junction; MENA; JBTS7 moderate
           c.2407+2T>A — splice donor exon 15; null; European; JBTS7 compound het
Domain 3: C2 domain (aa 960–1315)                    — Membrane association; RPGR binding
           Lys1326Ter — C2 truncating null; biallelic → MKS5; pan-ethnic

Key pathogenic variant classes (RPGRIP1L):
1. p.Ala229Thr (c.685G>A): N-terminal CC; European founder; JBTS7 hypomorphic
2. p.Arg1174Gln (c.3521G>A): RH domain C-term; NPHP4 interface; MENA; JBTS7/NPHP8
3. p.Lys1326Ter (c.3976A>T): C2 truncating null; pan-ethnic; biallelic → MKS5
4. p.Trp519Ter (c.1557G>A): CC/RH boundary null; pan-ethnic; biallelic → MKS5
5. c.2407+2T>A (splice donor exon 15): null; European; JBTS7 compound het
6. p.Asn694Ser (c.2081A>G): RH domain; South Asian; hypomorphic; NPHP8 mild
7. p.Leu821Pro (c.2462T>C): RH-C2 junction; MENA; JBTS7 moderate

JBTS7 Allele-Phenotype Tier Rule:
  Biallelic NULL (two truncating): MKS5 (Meckel-Gruber, lethal — encephalocele + polydactyly
                                   + polycystic kidneys + oligohydramnios; incompatible with
                                   extrauterine life)
  One NULL + one HYPOMORPHIC:      JBTS7 (Molar Tooth Sign ± NPHP8 renal ± retinal)
  Biallelic HYPOMORPHIC (Ala229Thr/Ala229Thr or missense/missense): JBTS7 mild or NPHP8
  p.Asn694Ser compound het (South Asian): NPHP8 renal-dominant ± mild JBTS7
"""

import random
import math

SEED = 421
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
    ('European',                       0.38),  # Ala229Thr founder enriched
    ('Middle Eastern / MENA',          0.25),  # Arg1174Gln / Leu821Pro
    ('South Asian',                    0.18),  # Asn694Ser / NPHP8 spectrum
    ('North African',                  0.10),  # Leu821Pro / Trp519Ter
    ('Other / Unknown',                0.06),
    ('East Asian',                     0.03),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Compound het: null + p.Ala229Thr (JBTS7 — European)',
    'Homozygous p.Ala229Thr (European founder — JBTS7 mild)',
    'Compound het: hypomorphic missense + missense (JBTS7 / NPHP8)',
    'Compound het: splice donor c.2407+2T>A + missense (JBTS7)',
    'Compound het: null + truncating (JBTS7 / MKS5 borderline)',
]
allele_fracs = [0.32, 0.25, 0.22, 0.13, 0.08]
allele_pool = []
for ac, frac in zip(allele_classes, allele_fracs):
    allele_pool.extend([ac] * round(frac * N))
while len(allele_pool) < N:
    allele_pool.append(allele_classes[0])
rng.shuffle(allele_pool)

age_dx_pool = (
    [rng.randint(0, 1) for _ in range(18)] +  # neonatal/infantile (MTS on MRI)
    [rng.randint(2, 8) for _ in range(16)] +   # early childhood
    [rng.randint(9, 24) for _ in range(6)]     # later (renal-first / NPHP8)
)
rng.shuffle(age_dx_pool)

for i in range(N):
    eth = eth_pool[i]
    allele = allele_pool[i]
    age_dx = age_dx_pool[i]
    has_ala229 = 'Ala229Thr' in allele
    # phenotype flags
    mts          = rng.random() < 0.87
    ataxia       = rng.random() < 0.87
    hypotonia    = rng.random() < 0.83
    oma          = rng.random() < 0.45
    retinal      = rng.random() < 0.25
    hepatic      = rng.random() < 0.10   # mild CHF possible (not COACH enriched)
    renal        = rng.random() < 0.30
    polydactyly  = rng.random() < 0.08
    id_          = rng.random() < 0.70
    breathing    = rng.random() < 0.55
    # renal ESRD age (if renal) — NPHP8 median ~22yr
    esrd_age = rng.randint(16, 32) if renal else None
    # hepatic: mild CHF subset (no portal HTN tier in JBTS7)
    mild_chf = hepatic
    patients.append({
        "id": f"JBTS7-{i+1:03d}",
        "ethnicity": eth,
        "allele_class": allele,
        "age_dx": age_dx,
        "mts": mts,
        "ataxia": ataxia,
        "hypotonia": hypotonia,
        "oculomotor_apraxia": oma,
        "retinal": retinal,
        "hepatic_mild_chf": mild_chf,
        "renal_nphp8": renal,
        "esrd_age": esrd_age,
        "polydactyly": polydactyly,
        "intellectual_disability": id_,
        "breathing_dysregulation": breathing,
        "ala229_carrier": has_ala229,
    })

# ── aggregate counts ──────────────────────────────────────────────────────────
n_mts         = sum(1 for p in patients if p['mts'])
n_ataxia      = sum(1 for p in patients if p['ataxia'])
n_hypotonia   = sum(1 for p in patients if p['hypotonia'])
n_oma         = sum(1 for p in patients if p['oculomotor_apraxia'])
n_retinal     = sum(1 for p in patients if p['retinal'])
n_hepatic     = sum(1 for p in patients if p['hepatic_mild_chf'])
n_renal       = sum(1 for p in patients if p['renal_nphp8'])
n_poly        = sum(1 for p in patients if p['polydactyly'])
n_id          = sum(1 for p in patients if p['intellectual_disability'])
n_breathing   = sum(1 for p in patients if p['breathing_dysregulation'])
n_ala229      = sum(1 for p in patients if p['ala229_carrier'])


def get_overview():
    return {
        "kpis": [
            {"label": "Cohort (n)", "value": str(N), "color": "#0d47a1"},
            {"label": "Molar Tooth Sign", "value": f"{n_mts}/{N}", "color": "#0d47a1"},
            {"label": "Cerebellar Ataxia", "value": f"{n_ataxia}/{N}", "color": "#4a148c"},
            {"label": "Renal NPHP8", "value": f"{n_renal}/{N}", "color": "#00695c"},
            {"label": "Retinal Dystrophy", "value": f"{n_retinal}/{N}", "color": "#e65100"},
            {"label": "European Ala229Thr", "value": f"{n_ala229}/{N}", "color": "#880e4f"},
        ],
        "hallmark": (
            "RPGRIP1L (FTM/NPHP8) is a transition zone Y-link scaffold protein with an "
            "N-terminal coiled-coil, a central RPGRIP1 Homology (RH) domain that binds NPHP4, "
            "and a C-terminal C2 domain that anchors membrane association and binds RPGR. "
            "Loss of RPGRIP1L → TZ Y-link spokes detach from axoneme → GPCR/Smo exclusion "
            "→ Hedgehog failure → Molar Tooth Sign + cerebellar vermis hypoplasia. "
            "RPGRIP1L accounts for ~2–3% of all Joubert syndrome cases worldwide."
        ),
        "critical_diagnostic_pearl": (
            "EUROPEAN Ala229Thr FOUNDER ALLELE: p.Ala229Thr (c.685G>A) is the MOST COMMON "
            "RPGRIP1L allele in European populations — found homozygous or compound het with "
            f"a second hypomorphic allele. {n_ala229}/{N} patients in this cohort carry Ala229Thr. "
            "Biallelic p.Ala229Thr causes mild JBTS7 WITHOUT MKS5 (pure hypomorphic — allele "
            "class governs). This allele is present at low frequency in gnomAD and can be "
            "mistaken for benign when found in trans; Molar Tooth Sign on brain MRI confirms "
            "pathogenicity. NEVER assign MKS5-tier to biallelic Ala229Thr patients."
        ),
        "allele_phenotype_rule": (
            "Biallelic NULL (two truncating/splice-null): MKS5 — Meckel-Gruber lethal "
            "(encephalocele + polydactyly + polycystic kidneys; perinatal lethal). "
            "One NULL + one HYPOMORPHIC (e.g. Ala229Thr): JBTS7 (MTS ± NPHP8 renal ± retinal). "
            "Biallelic HYPOMORPHIC (Ala229Thr/Ala229Thr): JBTS7 mild or NPHP8 renal-dominant. "
            "p.Asn694Ser compound het (South Asian): NPHP8 renal-dominant ± mild JBTS7."
        ),
        "prevalence": "~1/1,000,000–2,000,000 worldwide (JBTS7); ~2–3% of all Joubert syndrome cases",
        "first_description": (
            "Delous et al., 2007 (Nat Genet 39:875) — RPGRIP1L identified as MKS5 gene. "
            "Baala et al., 2007 (Nat Genet 39:875) — RPGRIP1L/FTM as JBTS7 gene in "
            "consanguineous families with Joubert syndrome and variable renal phenotype. "
            "Arts et al., 2007 — NPHP8 alleles defined the renal-dominant spectrum."
        ),
        "gene_summary": {
            "symbol": "RPGRIP1L",
            "alias": "FTM (fantom) / NPHP8",
            "omim_gene": "*610937",
            "omim_disease_jbts7": "#611560",
            "omim_disease_mks5": "#611561",
            "omim_disease_nphp8": "#613568",
            "chromosome": "16q12.2",
            "protein_length": "1315 aa",
            "protein_class": "TZ Y-link scaffold; CC + RH domain + C2 domain",
            "function": "TZ Y-link anchoring; NPHP4 binding (RH domain); RPGR binding (C2 domain); Y-link–axoneme junction",
        },
        "phenotype_summary": {
            "mts_pct": _pct(n_mts),
            "ataxia_pct": _pct(n_ataxia),
            "hypotonia_pct": _pct(n_hypotonia),
            "oma_pct": _pct(n_oma),
            "retinal_pct": _pct(n_retinal),
            "hepatic_pct": _pct(n_hepatic),
            "renal_pct": _pct(n_renal),
            "id_pct": _pct(n_id),
            "breathing_pct": _pct(n_breathing),
            "polydactyly_pct": _pct(n_poly),
            "ala229_pct": _pct(n_ala229),
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
    age_buckets = {"0-1y (neonatal)": 0, "2-8y": 0, "9-24y": 0}
    for p in patients:
        a = p['age_dx']
        if a <= 1:
            age_buckets["0-1y (neonatal)"] += 1
        elif a <= 8:
            age_buckets["2-8y"] += 1
        else:
            age_buckets["9-24y"] += 1

    # Renal outcomes
    esrd_ages = [p['esrd_age'] for p in patients if p['renal_nphp8'] and p['esrd_age']]
    avg_esrd_age = round(sum(esrd_ages) / len(esrd_ages), 1) if esrd_ages else None

    # Transplant projections
    n_renal_tx = round(n_renal * 0.65)   # ~65% of NPHP8 reach ESRD needing Tx

    return {
        "ethnicity": [{"ethnicity": k, "n": v, "pct": _pct(v)} for k, v in eth_counts.items()],
        "allele_class": [{"class": k, "n": v, "pct": _pct(v)} for k, v in allele_counts.items()],
        "age_at_diagnosis": [{"bucket": k, "n": v, "pct": _pct(v)} for k, v in age_buckets.items()],
        "phenotype_matrix": [
            {"feature": "Molar Tooth Sign (MTS)", "n": n_mts, "pct": _pct(n_mts), "note": "Pathognomonic; brain MRI mandatory"},
            {"feature": "Cerebellar Ataxia", "n": n_ataxia, "pct": _pct(n_ataxia), "note": "Gait ataxia; truncal instability"},
            {"feature": "Neonatal Hypotonia", "n": n_hypotonia, "pct": _pct(n_hypotonia), "note": "Universal early feature"},
            {"feature": "Oculomotor Apraxia (OMA)", "n": n_oma, "pct": _pct(n_oma), "note": "Horizontal gaze initiation failure; ~45%"},
            {"feature": "Intellectual Disability", "n": n_id, "pct": _pct(n_id), "note": "Moderate > severe; variable"},
            {"feature": "Breathing Dysregulation", "n": n_breathing, "pct": _pct(n_breathing), "note": "Neonatal episodic apnea/hyperpnea; self-resolves"},
            {"feature": "Retinal Dystrophy", "n": n_retinal, "pct": _pct(n_retinal), "note": "Rod-cone; RPGR-binding C2 implicated; annual ERG"},
            {"feature": "Renal NPHP8", "n": n_renal, "pct": _pct(n_renal), "note": "TIN; ESRD median ~22yr (later than JBTS3/NPHP1)"},
            {"feature": "Hepatic (mild CHF)", "n": n_hepatic, "pct": _pct(n_hepatic), "note": "Mild CHF only; NO COACH — not a TMEM67/CC2D2A gene"},
            {"feature": "Polydactyly", "n": n_poly, "pct": _pct(n_poly), "note": "Postaxial; rare in JBTS7 (~8%)"},
            {"feature": "European p.Ala229Thr", "n": n_ala229, "pct": _pct(n_ala229), "note": "Most common RPGRIP1L allele; European enriched"},
        ],
        "transplant_outcomes": {
            "n_renal_transplant_needed": n_renal_tx,
            "renal_tx_outcome": "CURATIVE (cell-autonomous; no recurrence post-Tx)",
            "neurological_corrected": "NOT CORRECTED by renal transplant",
            "retinal_corrected": "NOT CORRECTED by renal transplant",
            "avg_esrd_age": avg_esrd_age,
            "note": "No liver transplant indicated in JBTS7 (not COACH gene; mild CHF only)",
        },
        "allele_protein_table": [
            {"variant": "p.Ala229Thr", "cdna": "c.685G>A", "domain": "N-terminal CC (aa 229)", "ethnic": "European founder", "tier": "JBTS7 hypomorphic (biallelic → mild JBTS7)"},
            {"variant": "p.Arg1174Gln", "cdna": "c.3521G>A", "domain": "RH domain C-term / NPHP4 interface", "ethnic": "MENA", "tier": "JBTS7 / NPHP8"},
            {"variant": "p.Lys1326Ter", "cdna": "c.3976A>T", "domain": "C2 domain truncating null", "ethnic": "Pan-ethnic", "tier": "Null; biallelic → MKS5"},
            {"variant": "p.Trp519Ter",  "cdna": "c.1557G>A", "domain": "CC/RH boundary truncating null", "ethnic": "Pan-ethnic", "tier": "Null; biallelic → MKS5"},
            {"variant": "c.2407+2T>A",  "cdna": "Splice donor exon 15", "domain": "RH domain (splice-null)", "ethnic": "European", "tier": "Null; JBTS7 compound het"},
            {"variant": "p.Asn694Ser",  "cdna": "c.2081A>G", "domain": "RH domain core", "ethnic": "South Asian", "tier": "Hypomorphic; NPHP8 renal-dominant"},
            {"variant": "p.Leu821Pro",  "cdna": "c.2462T>C", "domain": "RH-C2 junction", "ethnic": "MENA", "tier": "JBTS7 moderate"},
        ],
        "jbts_comparison": [
            {"type": "JBTS3", "gene": "AHI1", "distinctive": "OMA ~75% (highest); Ashkenazi Arg830Trp founder", "coach": "No", "hepatic": "No"},
            {"type": "JBTS4", "gene": "NPHP1", "distinctive": "High renal ~45%; MLPA mandatory (610kb del)", "coach": "No", "hepatic": "No"},
            {"type": "JBTS5", "gene": "CEP290", "distinctive": "Most common JBTS gene; IVS26 invisible to WES; highest retinal (57%)", "coach": "No", "hepatic": "No"},
            {"type": "JBTS6", "gene": "TMEM67", "distinctive": "COACH hepatic fibrosis ~30% DISTINCTIVE; North African Cys615Arg", "coach": "Yes", "hepatic": "Yes ~30%"},
            {"type": "JBTS7", "gene": "RPGRIP1L", "distinctive": "European Ala229Thr founder; Y-link scaffold; MKS5 biallelic null", "coach": "No", "hepatic": "Mild only"},
        ],
    }


def get_definitions():
    return {
        "gene": "RPGRIP1L (Retinitis Pigmentosa GTPase Regulator-Interacting Protein 1-Like; also FTM [fantom]; NPHP8)",
        "omim_gene": "*610937",
        "omim_jbts7": "#611560",
        "omim_mks5": "#611561",
        "omim_nphp8": "#613568",
        "chromosome": "16q12.2",
        "protein": "1315 aa; N-terminal coiled-coil (CC, aa 1-450) + RPGRIP1 Homology (RH) domain (aa 450-960; NPHP4 binding) + C2 domain (aa 960-1315; RPGR binding, membrane)",
        "pathway": "Transition Zone (TZ) Y-link scaffold; part of NPHP-MKS-JBTS module; Y-link anchoring to ciliary axoneme",
        "allele_tier_rule": (
            "Biallelic NULL → MKS5 lethal (encephalocele + polydactyly + PKD). "
            "One NULL + one HYPOMORPHIC → JBTS7 (MTS ± NPHP8 ± retinal). "
            "Biallelic HYPOMORPHIC (Ala229Thr/Ala229Thr) → JBTS7 mild or NPHP8. "
            "p.Asn694Ser compound het (South Asian) → NPHP8 renal-dominant."
        ),
        "mks5": "Meckel-Gruber Syndrome Type 5 — biallelic null RPGRIP1L; lethal; encephalocele, polydactyly, polycystic kidneys, oligohydramnios",
        "nphp8": "Nephronophthisis Type 8 — biallelic hypomorphic RPGRIP1L; renal TIN; ESRD median ~22yr; mild or no MTS in isolation",
        "mts": "Molar Tooth Sign — pathognomonic brain MRI; elongated superior cerebellar peduncles + cerebellar vermis hypoplasia/aplasia",
        "oma": "Oculomotor Apraxia — horizontal gaze initiation failure; compensatory head thrusting; JBTS7 ~45% (less than JBTS3 ~75%)",
        "ala229thr_pearl": (
            "p.Ala229Thr (c.685G>A) — most common RPGRIP1L allele in European populations. "
            "Hypomorphic: biallelic causes mild JBTS7 (NOT MKS5). Present at low gnomAD frequency; "
            "clinical MTS on MRI confirms pathogenicity. Do NOT call biallelic Ala229Thr as MKS5-tier."
        ),
        "rh_domain": "RPGRIP1 Homology (RH) domain (aa 450-960) — interacts with NPHP4 at TZ Y-link; most disease alleles cluster here",
        "c2_domain": "C2 domain (aa 960-1315) — membrane association; binds RPGR (retinal GTPase regulator); retinal variability via this interaction",
        "y_link_scaffold": "TZ Y-link structure — connects axonemal doublet microtubules to ciliary membrane via Y-shaped electron-dense links; RPGRIP1L anchors the axonemal spoke",
        "tz_module": "NPHP-MKS-JBTS module — RPGRIP1L, NPHP4, CEP290, AHI1, CC2D2A, TMEM67; TZ gate integrity complex",
        "therapy_status": "No disease-modifying therapy 2026; renal transplant CURATIVE for NPHP8 renal endpoint; neurological/retinal NOT corrected",
        "inheritance": "Autosomal Recessive — biallelic loss-of-function; allele class governs disease tier (MKS5 → JBTS7 → NPHP8)",
        "frequency": "~2-3% of all Joubert syndrome; ~1/1,000,000-2,000,000 worldwide",
        "no_coach": "RPGRIP1L is NOT a COACH gene — mild CHF may occur but hepatic fibrosis is NOT a distinctive feature (unlike TMEM67/CC2D2A)",
        "related_genes": "NPHP4 (NPHP4 — RPGRIP1L binding partner); AHI1 (JBTS3); TMEM67 (JBTS6/COACH); CC2D2A (JBTS9/COACH); RPGR (retinal; C2 binding)",
    }
