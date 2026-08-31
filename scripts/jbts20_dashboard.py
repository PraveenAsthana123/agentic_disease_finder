"""
TMEM231 Joubert Syndrome Type 20 (JBTS20) — Autosomal Recessive / TMEM231 (Transmembrane Protein 231) / TZ Membrane Bridge / B9-Tectonic Interface / No MKS Tier
===================================================================================================================================================================
Primary Gene : TMEM231 (*614949) — 16q23.1; ~669 aa; Transmembrane protein 231.
               TMEM231 is a multi-pass transmembrane protein embedded in the membrane of
               the ciliary transition zone (TZ). It functions as the structural bridge between
               the B9 complex inner-leaflet anchoring layer (B9D1, B9D2, MKS1) and the Tectonic
               complex lipid gate (TCTN1, TCTN2, TCTN3), working through direct contacts with
               B9D1 at the inner leaflet and TMEM67/TMEM138 at the TZ membrane.
               TMEM231 protein domain architecture:
               - N-terminal cytoplasmic tail (aa 1–75): TZ targeting; direct TMEM138 interaction
                 surface; B9D1 docking interface; ciliary import signal
               - Transmembrane segments 1–4 (aa 76–380): four TM helices; TZ membrane embedding;
                 B9D1 extracellular contact; B9D2 interface; forms TZ membrane channel scaffold
               - Extracellular loops EL1–EL3 (aa 381–550): TMEM67 docking; MKS1 interaction;
                 extracellular TZ gate reinforcement; Tectonic module interface
               - C-terminal intracellular region (aa 551–669): NPHP4 contact; IFT-A docking site;
                 RPGRIP1L interaction; cytoplasmic TZ gate stabilisation
               TMEM231 LOF → B9-Tectonic interface disrupted → TZ gate partially destabilised →
               SMO excluded → Hedgehog failure → Molar Tooth Sign (MTS).

⚠ NO MKS TIER — TMEM231-SPECIFIC RULE:
   Biallelic TMEM231 null alleles (null/null genotype) → JBTS20 live birth, NOT Meckel-Gruber
   Syndrome. Unlike B9D1 (JBTS19/MKS9), B9D2 (JBTS34/MKS10), and MKS1 (JBTS28), TMEM231 LOF
   disrupts the TZ bridge without collapsing the entire B9 complex inner-leaflet anchor. The
   B9D1-B9D2-MKS1 core remains partially functional, providing sufficient TZ scaffolding to
   prevent perinatal lethality. This is the critical counselling distinction: JBTS20 families
   carry NO MKS perinatal-lethal risk, regardless of allele class. The absence of MKS risk must
   be stated explicitly when counselling JBTS20 families who may conflate their diagnosis with the
   MKS9-tier JBTS19 (B9D1) — the two involve adjacent B9 complex members but have fundamentally
   different lethality risk profiles.

⚠ B9-TECTONIC BRIDGE — TMEM231-SPECIFIC MOLECULAR DISTINCTION:
   TMEM231 is uniquely positioned at the interface between the B9 complex (inner-leaflet anchor)
   and the Tectonic complex (lipid gate). It contacts B9D1 on the inner leaflet side and TMEM67/
   TMEM138 at the TZ membrane. This makes TMEM231 LOF phenotypically intermediate between pure B9
   module loss (JBTS19/B9D1 — more severe, MKS9 tier) and pure Tectonic module loss (JBTS18/TCTN3
   — TZ gate disrupted but inner leaflet intact). DDx: TMEM138/JBTS16 is at 11q12.2 (adjacent to
   TMEM216), while TMEM231 is at 16q23.1 — different chromosomes, require WES to distinguish.

⚠ RENAL PENETRANCE (~25%): Annual NPHP-like protocol mandatory. ESRD median ~25 yr.
   TMEM231 LOF disrupts TZ gate in renal tubular primary cilia → NPHP-like tubulointerstitial
   nephritis. Intermediate renal penetrance between B9D1/JBTS19 (~35%) and TCTN3/JBTS18 (~20%),
   consistent with partial TZ gate disruption (B9-Tectonic bridge lost, but B9 core retained).
   Annual surveillance (creatinine, cystatin C, urine osmolality, microalbuminuria) from diagnosis.
   Renal transplant curative; no allograft recurrence (cell-autonomous AR ciliopathy).

Disease OMIM : #614990 — Joubert Syndrome Type 20 (JBTS20)
Chromosome   : 16q23.1
Inheritance  : Autosomal recessive — biallelic LOF; NO MKS lethal tier
Cohort size  : 40-patient educational cohort (seed 447)
"""

import random
import math

SEED = 447
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
    ('European',               0.32),
    ('Middle Eastern / MENA',  0.24),
    ('South Asian',            0.22),
    ('North African',          0.12),
    ('East Asian',             0.06),
    ('Other / Unknown',        0.04),
]

# Allele classes (NO MKS tier — all result in live birth)
allele_classes = [
    ('Biallelic Missense',      0.38),   # moderate phenotype
    ('Null / Hypomorphic',      0.32),   # JBTS20 live birth
    ('Splice / Null Compound',  0.18),   # splice + null compound het
    ('Biallelic Splice',        0.12),   # biallelic splice — variable severity
]

variants = [
    'Arg185Gln/Arg185Gln',
    'Arg185Gln/Gly484Arg',
    'Tyr249Cys/Arg185Gln',
    'Leu356Pro/Tyr249Cys',
    'Gln53Ter/Arg185Gln',
    'c.312+1G>A/Arg185Gln',
    'Trp411Ter/Ala203Val',
    'Gly484Arg/Tyr249Cys',
    'Arg185Gln/Leu356Pro',
    'Tyr249Cys/Gly484Arg',
    'Ala203Val/Ala203Val',
]

_rng_p = random.Random(SEED + 1)
for i in range(N):
    eth = _rng_p.choices([e[0] for e in ethnicities], weights=[e[1] for e in ethnicities])[0]
    ac  = _rng_p.choices([a[0] for a in allele_classes], weights=[a[1] for a in allele_classes])[0]
    var = _rng_p.choice(variants)
    age = _rng_p.randint(2, 42)
    sex = _rng_p.choice(['M', 'F'])

    ataxia   = _rng_p.random() < 0.85
    hypotonia= _rng_p.random() < 0.80
    oma      = _rng_p.random() < 0.50
    breath   = _rng_p.random() < 0.52
    retinal  = _rng_p.random() < 0.22
    renal    = _rng_p.random() < 0.25
    hepatic  = _rng_p.random() < 0.10
    poly     = _rng_p.random() < 0.12
    id_flag  = _rng_p.random() < 0.68
    cc       = _rng_p.random() < 0.08   # corpus callosum anomaly

    patients.append({
        'id':           f'JBTS20-{i+1:03d}',
        'age':          age,
        'sex':          sex,
        'ethnicity':    eth,
        'allele_class': ac,
        'variant':      var,
        'mts':          True,   # MTS is diagnostic criterion — 100%
        'ataxia':       ataxia,
        'hypotonia':    hypotonia,
        'oma':          oma,
        'breathing':    breath,
        'retinal':      retinal,
        'renal':        renal,
        'hepatic':      hepatic,
        'poly':         poly,
        'id_flag':      id_flag,
        'cc':           cc,
    })

# ── aggregate counts ─────────────────────────────────────────────────────────
n_mts      = N
n_ataxia   = sum(1 for p in patients if p['ataxia'])
n_hypotonia= sum(1 for p in patients if p['hypotonia'])
n_oma      = sum(1 for p in patients if p['oma'])
n_breath   = sum(1 for p in patients if p['breathing'])
n_retinal  = sum(1 for p in patients if p['retinal'])
n_renal    = sum(1 for p in patients if p['renal'])
n_hepatic  = sum(1 for p in patients if p['hepatic'])
n_poly     = sum(1 for p in patients if p['poly'])
n_id       = sum(1 for p in patients if p['id_flag'])
n_cc       = sum(1 for p in patients if p['cc'])

_eth_counts = {}
for p in patients:
    _eth_counts[p['ethnicity']] = _eth_counts.get(p['ethnicity'], 0) + 1

_ac_counts = {}
for p in patients:
    _ac_counts[p['allele_class']] = _ac_counts.get(p['allele_class'], 0) + 1


# ── API functions ─────────────────────────────────────────────────────────────
def get_overview():
    return {
        "disease_id": "jbts20",

        "kpis": {
            "total_patients":   N,
            "mts_pct":          100,
            "ataxia_pct":       _pct(n_ataxia),
            "hypotonia_pct":    _pct(n_hypotonia),
            "oma_pct":          _pct(n_oma),
            "breathing_pct":    _pct(n_breath),
            "retinal_pct":      _pct(n_retinal),
            "renal_pct":        _pct(n_renal),
            "hepatic_pct":      _pct(n_hepatic),
            "poly_pct":         _pct(n_poly),
            "id_pct":           _pct(n_id),
            "cc_pct":           _pct(n_cc),
            "no_mks_tier":      True,
        },

        "alerts": {
            "no_mks_tier": (
                "NO MKS TIER — TMEM231 biallelic null → JBTS20 LIVE BIRTH, NOT Meckel-Gruber Syndrome. "
                "Unlike B9D1/JBTS19 (MKS9 tier, ~22% null/null risk) or B9D2/JBTS34 (MKS10 tier), "
                "TMEM231 LOF disrupts only the B9-Tectonic bridge; the B9D1-B9D2-MKS1 inner-leaflet "
                "anchor is retained, preventing perinatal lethality. No MKS counselling needed. "
                "Critical counselling distinction for families and clinicians familiar with JBTS19."
            ),
            "b9_tectonic_bridge": (
                "B9-TECTONIC BRIDGE — TMEM231 bridges B9 complex (inner leaflet: B9D1/B9D2/MKS1) and "
                "Tectonic complex (lipid gate: TCTN1/TCTN2/TCTN3) at the TZ. Contacts B9D1, TMEM138, "
                "TMEM67. DDx: TMEM138/JBTS16 (11q12.2, adjacent to TMEM216) vs TMEM231/JBTS20 (16q23.1) "
                "— different chromosomes, require WES to distinguish. TMEM138 has TMEM216 co-dependency; "
                "TMEM231 does not."
            ),
            "renal_25pct": (
                "RENAL 25% — Intermediate between B9D1/JBTS19 (~35%) and TCTN3/JBTS18 (~20%). Consistent "
                "with partial TZ gate disruption (bridge lost, B9 core retained). Annual NPHP-like "
                "surveillance mandatory. ESRD median ~25 yr. Renal transplant curative."
            ),
        },

        "key_facts": [
            "TMEM231 (669 aa) bridges B9 complex inner leaflet (B9D1/B9D2/MKS1) and Tectonic lipid gate (TCTN1/2/3)",
            "N-tail (aa 1–75): TMEM138 contact; B9D1 docking; TZ targeting",
            "TM1-4 (aa 76–380): TZ membrane embedding; B9D1/B9D2 extracellular interface",
            "Extracellular loops EL1-3 (aa 381–550): TMEM67/MKS1 docking; Tectonic module interface",
            "C-terminal (aa 551–669): NPHP4 contact; IFT-A docking; RPGRIP1L interaction",
            "NO MKS tier — biallelic null → JBTS20 live birth; no Meckel-Gruber risk",
            "Renal penetrance 25% (intermediate: B9D1/35% > TMEM231/25% > TCTN3/20%)",
            "Retinal penetrance 22% (rod-cone dystrophy) — annual ERG from age 3",
            "Hepatic CHF 10% — 2-yr surveillance LFTs + hepatic ultrasound",
            "Frequency ~1% of all Joubert syndrome (~1/4–8 million worldwide)",
        ],

        "patients": [
            {
                "id":           p['id'],
                "age":          p['age'],
                "sex":          p['sex'],
                "ethnicity":    p['ethnicity'],
                "allele_class": p['allele_class'],
                "variant":      p['variant'],
                "mts":          p['mts'],
                "ataxia":       p['ataxia'],
                "hypotonia":    p['hypotonia'],
                "oma":          p['oma'],
                "breathing":    p['breathing'],
                "retinal":      p['retinal'],
                "renal":        p['renal'],
                "hepatic":      p['hepatic'],
                "poly":         p['poly'],
                "id_flag":      p['id_flag'],
                "cc":           p['cc'],
            }
            for p in patients
        ],
    }


def get_breakdown():
    return {
        "disease_id": "jbts20",

        "ethnicity_distribution": [
            {"ethnicity": eth, "count": cnt, "pct": _pct(cnt)}
            for eth, cnt in sorted(_eth_counts.items(), key=lambda x: -x[1])
        ],

        "allele_class_distribution": [
            {"allele_class": ac, "count": cnt, "pct": _pct(cnt)}
            for ac, cnt in sorted(_ac_counts.items(), key=lambda x: -x[1])
        ],

        "phenotype_summary": {
            "mts":       {"n": n_mts,      "pct": _pct(n_mts)},
            "ataxia":    {"n": n_ataxia,   "pct": _pct(n_ataxia)},
            "hypotonia": {"n": n_hypotonia,"pct": _pct(n_hypotonia)},
            "oma":       {"n": n_oma,      "pct": _pct(n_oma)},
            "breathing": {"n": n_breath,   "pct": _pct(n_breath)},
            "retinal":   {"n": n_retinal,  "pct": _pct(n_retinal)},
            "renal":     {"n": n_renal,    "pct": _pct(n_renal)},
            "hepatic":   {"n": n_hepatic,  "pct": _pct(n_hepatic)},
            "poly":      {"n": n_poly,     "pct": _pct(n_poly)},
            "id":        {"n": n_id,       "pct": _pct(n_id)},
            "cc":        {"n": n_cc,       "pct": _pct(n_cc)},
        },

        "notable_variants": [
            {
                "name":       "Arg185Gln",
                "cdna":       "c.554G>A",
                "domain":     "TM2 proximal cytoplasmic face — B9D1 extracellular contact surface",
                "population": "Pan-ethnic (commonest JBTS20 missense)",
                "severity":   "Moderate",
                "mechanism":  "Partial B9D1 contact impairment; TZ bridge partially destabilised; JBTS20 moderate phenotype; most commonly seen missense allele",
            },
            {
                "name":       "Tyr249Cys",
                "cdna":       "c.746A>G",
                "domain":     "TM3 — TZ membrane lipid-facing surface",
                "population": "South Asian",
                "severity":   "Moderate–Severe",
                "mechanism":  "TM3 helix destabilisation; disrupts TZ membrane embedding; stronger TZ gate loss than Arg185Gln; renal/retinal penetrance elevated",
            },
            {
                "name":       "Leu356Pro",
                "cdna":       "c.1067T>C",
                "domain":     "TM4 region — extracellular loop EL1 entry",
                "population": "South Asian",
                "severity":   "Moderate–Severe",
                "mechanism":  "TM4/EL1 junction proline substitution kinks helix; disrupts TMEM67 docking at EL1; intermediate TZ gate destabilisation",
            },
            {
                "name":       "Gln53Ter",
                "cdna":       "c.157C>T",
                "domain":     "N-terminal cytoplasmic tail — premature stop; truncates before TM1",
                "population": "European",
                "severity":   "Severe (Null)",
                "mechanism":  "Complete loss of TMEM231 — no TZ targeting, no B9D1/TMEM138 contact, no TZ bridge. Biallelic null/null genotype → JBTS20 live birth (no MKS risk)",
            },
            {
                "name":       "Gly484Arg",
                "cdna":       "c.1450G>A",
                "domain":     "Extracellular loop EL3 — MKS1 interaction surface",
                "population": "Middle Eastern / MENA",
                "severity":   "Moderate",
                "mechanism":  "EL3 Gly-to-Arg disrupts MKS1 docking; TZ extracellular reinforcement impaired; JBTS20 moderate — renal penetrance in range",
            },
            {
                "name":       "c.312+1G>A",
                "cdna":       "c.312+1G>A",
                "domain":     "Splice donor intron 3 — TM1-TM2 boundary",
                "population": "European",
                "severity":   "Severe (Null)",
                "mechanism":  "Splice donor abolition → exon 3 skip → frameshift → NMD. Full null. Compound het with missense → JBTS20 live birth",
            },
            {
                "name":       "Ala203Val",
                "cdna":       "c.608C>T",
                "domain":     "TM2 — hydrophobic core",
                "population": "North African founder",
                "severity":   "Mild (Hypomorphic)",
                "mechanism":  "Conservative Val substitution retains partial TZ membrane embedding; mild JBTS20; important hypomorphic allele for compound het counselling",
            },
            {
                "name":       "Trp411Ter",
                "cdna":       "c.1233G>A",
                "domain":     "Extracellular loop EL2 — TMEM67 docking region",
                "population": "Pan-ethnic",
                "severity":   "Severe (Null)",
                "mechanism":  "Mid-protein truncating nonsense — loss of EL2/EL3/C-terminal NPHP4 interface. Full null. Biallelic → JBTS20 (no MKS risk)",
            },
        ],
    }


def get_definitions():
    return {
        "disease_id":    "jbts20",
        "gene_full_name":"Transmembrane Protein 231 (TMEM231) — B9-Tectonic TZ Bridge; No MKS Tier; B9D1/TMEM138/TMEM67 Contact Network",
        "omim_gene":     "614949",
        "omim_jbts20":   "614990",
        "chromosome":    "16q23.1",
        "protein_size":  (
            "~669 aa — N-terminal cytoplasmic tail / TMEM138 interface / B9D1 docking (aa 1–75); "
            "Transmembrane segments TM1–4 / TZ membrane / B9D1-B9D2 extracellular interface (aa 76–380); "
            "Extracellular loops EL1–3 / TMEM67 / MKS1 docking (aa 381–550); "
            "C-terminal intracellular / NPHP4 / IFT-A / RPGRIP1L contact (aa 551–669)"
        ),
        "inheritance":   "Autosomal recessive — biallelic LOF; NO MKS lethal tier (biallelic null → JBTS20 live birth)",

        "no_mks_tier_rule": (
            "TMEM231 biallelic null (null/null genotype, e.g. Gln53Ter/Gln53Ter, c.312+1G>A/Trp411Ter) "
            "→ JBTS20 LIVE BIRTH, NOT Meckel-Gruber Syndrome. Unlike B9D1 (JBTS19/MKS9), B9D2 (JBTS34/"
            "MKS10), and MKS1 (JBTS28/MKS1) — which all carry null/null perinatal-lethal MKS risk — "
            "TMEM231 LOF disrupts only the B9-Tectonic bridge while the core B9D1-B9D2-MKS1 inner-leaflet "
            "anchor remains partially functional. This provides sufficient TZ scaffolding to prevent the "
            "complete TZ gate collapse that causes perinatal lethality in MKS. Counsellors MUST state "
            "explicitly that JBTS20 carries NO MKS perinatal-lethal risk, particularly for families "
            "referred after an initial JBTS19 (B9D1) evaluation where MKS9 risk was discussed."
        ),

        "glossary": [
            {
                "term": "TMEM231",
                "definition": (
                    "Transmembrane Protein 231 (OMIM *614949). Multi-pass TZ membrane protein (669 aa, 16q23.1). "
                    "Functions as the bridge between the B9 complex inner-leaflet anchor (B9D1, B9D2, MKS1) "
                    "and the Tectonic complex lipid gate (TCTN1, TCTN2, TCTN3). Contacts B9D1 at the inner "
                    "leaflet and TMEM67/TMEM138 at the TZ membrane. LOF → JBTS20 (no MKS tier)."
                ),
            },
            {
                "term": "B9-Tectonic bridge",
                "definition": (
                    "Structural concept for TMEM231's unique TZ position: it connects the B9 complex "
                    "(inner-leaflet anchor — B9D1, B9D2, MKS1) to the Tectonic complex (lipid gate — "
                    "TCTN1, TCTN2, TCTN3). Loss of TMEM231 disrupts this bridge, partially destabilising "
                    "both modules, but the B9 core is retained, explaining the intermediate severity of "
                    "JBTS20 versus pure B9 module loss (JBTS19) or pure Tectonic module loss (JBTS18)."
                ),
            },
            {
                "term": "No MKS tier (TMEM231)",
                "definition": (
                    "TMEM231 biallelic null → JBTS20 live birth. Critical counselling distinction vs "
                    "B9D1/JBTS19 (MKS9), B9D2/JBTS34 (MKS10), MKS1/JBTS28 (MKS1) — all three B9 complex "
                    "members carry null/null perinatal-lethal MKS risk. TMEM231 does not, because it is a "
                    "bridge protein rather than a core B9 complex structural member."
                ),
            },
            {
                "term": "Transition zone (TZ)",
                "definition": (
                    "Compartment at the base of the ciliary axoneme between the basal body and the ciliary "
                    "shaft. Acts as a diffusion barrier ('ciliary gate') controlling protein composition of "
                    "the ciliary membrane. Requires three cooperative modules: B9 complex (inner leaflet "
                    "anchor — B9D1/B9D2/MKS1), NPHP module (Y-link scaffold — NPHP1/4), and Tectonic "
                    "complex (lipid gate — TCTN1/2/3). TMEM231 bridges B9 and Tectonic modules."
                ),
            },
            {
                "term": "TMEM231 vs TMEM138 DDx",
                "definition": (
                    "TMEM138 (JBTS16, 11q12.2) and TMEM231 (JBTS20, 16q23.1) are both TZ membrane proteins "
                    "that interact with B9D1. Key distinctions: TMEM138 is at 11q12.2 adjacent to TMEM216 "
                    "(mutual stabilisation co-dependency); TMEM231 is at 16q23.1 (no adjacent co-dependency). "
                    "TMEM138 LOF specifically destabilises TMEM216 (JBTS9 allelic); TMEM231 LOF disrupts the "
                    "B9-Tectonic bridge without destabilising TMEM216. WES must confirm gene identity by "
                    "chromosomal locus — 11q vs 16q."
                ),
            },
            {
                "term": "Molar Tooth Sign (MTS)",
                "definition": (
                    "Pathognomonic MRI appearance in Joubert syndrome: elongated superior cerebellar "
                    "peduncles + vermis hypoplasia form a 'molar tooth' shape on axial brain MRI. "
                    "Present in 100% of JBTS20 cases (diagnostic criterion)."
                ),
            },
            {
                "term": "NPHP-like TIN (JBTS20)",
                "definition": (
                    "Nephronophthisis-like tubulointerstitial nephritis. In JBTS20: affects ~25% of patients "
                    "— intermediate penetrance between B9D1/JBTS19 (~35%) and TCTN3/JBTS18 (~20%). Annual "
                    "surveillance mandatory from diagnosis. ESRD median ~25 yr; renal transplant curative, "
                    "no allograft recurrence (cell-autonomous AR ciliopathy)."
                ),
            },
            {
                "term": "B9 complex vs TMEM231 distinction",
                "definition": (
                    "B9 complex (B9D1/B9D2/MKS1) forms the inner-leaflet TZ anchor; all three members carry "
                    "MKS null/null perinatal-lethal risk. TMEM231 bridges B9 and Tectonic complexes but is "
                    "NOT a core B9 complex member — TMEM231 LOF leaves the B9D1-B9D2-MKS1 core partially "
                    "intact, preventing MKS lethality. This is the mechanistic basis for JBTS20 being MKS-tier-free."
                ),
            },
        ],

        "domain_matrix": [
            {
                "domain":          "N-terminal cytoplasmic tail / TMEM138 interface / B9D1 docking (aa 1–75)",
                "location":        "N-terminus — disordered cytoplasmic; primary protein interaction hub",
                "function":        "TZ targeting; TMEM138 protein–protein contact (B9D1-TMEM138-TMEM231 triad); B9D1 inner-leaflet docking; ciliary import signal; Gln53Ter (European null) truncates before TM1",
                "variant_examples":"Gln53Ter (European null, severe — truncates before TM1); Ala203Val in TM2 (North African founder, hypomorphic/mild)",
            },
            {
                "domain":          "Transmembrane segments TM1–4 / TZ membrane / B9D1-B9D2 extracellular interface (aa 76–380)",
                "location":        "Central — four TM helices span the TZ membrane bilayer",
                "function":        "TZ membrane embedding; B9D1 extracellular contact; B9D2 interface; TZ membrane channel scaffold; Arg185Gln (TM2, pan-ethnic) and Tyr249Cys (TM3, South Asian) disrupt TZ membrane embedding",
                "variant_examples":"Arg185Gln (pan-ethnic, moderate); Tyr249Cys (South Asian, moderate-severe); Leu356Pro (South Asian, moderate-severe); Ala203Val (North African founder, hypomorphic/mild); c.312+1G>A (splice null, European)",
            },
            {
                "domain":          "Extracellular loops EL1–3 / TMEM67 docking / MKS1 interface (aa 381–550)",
                "location":        "Extracellular TZ — three loops projecting into the extracellular TZ space",
                "function":        "TMEM67 docking (EL1/EL2); MKS1 interaction (EL3); extracellular TZ gate reinforcement; Tectonic module interface; Gly484Arg (EL3, MENA) disrupts MKS1 docking; Trp411Ter (EL2, pan-ethnic null)",
                "variant_examples":"Gly484Arg (MENA, moderate — EL3/MKS1 contact); Trp411Ter (pan-ethnic null — truncates EL2/EL3/C-term)",
            },
            {
                "domain":          "C-terminal intracellular region / NPHP4 / IFT-A / RPGRIP1L contact (aa 551–669)",
                "location":        "C-terminus — intracellular; cytoplasmic TZ gate stabilisation",
                "function":        "NPHP4 contact (NPHP module bridge); IFT-A docking site; RPGRIP1L interaction; cytoplasmic TZ gate stabilisation; Trp411Ter truncates this region entirely",
                "variant_examples":"Trp411Ter removes entire C-terminal NPHP4/IFT-A interface (pan-ethnic null, severe)",
            },
        ],

        "clinical_pearls": [
            {
                "title": "TMEM231 — B9-Tectonic TZ Bridge: No MKS Tier (Biallelic Null → JBTS20 Live Birth)",
                "detail": (
                    "TMEM231 uniquely bridges the B9 complex inner-leaflet anchor (B9D1/B9D2/MKS1) and the "
                    "Tectonic complex lipid gate (TCTN1/TCTN2/TCTN3) at the TZ. Unlike all three B9 complex "
                    "members — B9D1 (JBTS19/MKS9), B9D2 (JBTS34/MKS10), and MKS1 (JBTS28/MKS1) — which "
                    "carry biallelic null/null perinatal-lethal MKS risk, TMEM231 LOF disrupts only the "
                    "bridge while the B9D1-B9D2-MKS1 core inner-leaflet anchor is retained. This provides "
                    "sufficient TZ scaffolding to prevent MKS perinatal lethality. JBTS20 families carry "
                    "NO MKS risk. Counsellors must state this explicitly, particularly for families "
                    "referred after JBTS19 evaluation where MKS9 risk was discussed."
                ),
            },
            {
                "title": "TMEM231 vs TMEM138 (JBTS16): Two TZ Membrane Proteins — Different Chromosomes, Different Mechanisms, Different DDx",
                "detail": (
                    "Both TMEM231 (JBTS20, 16q23.1) and TMEM138 (JBTS16, 11q12.2) are TZ membrane proteins "
                    "that interact with B9D1. Key clinical DDx: TMEM138 is at chromosome 11q12.2, physically "
                    "adjacent to TMEM216 (JBTS2, 11q12.2) — TMEM138 LOF specifically destabilises TMEM216 "
                    "(co-dependency effect). TMEM231 is at chromosome 16q23.1 — no adjacent co-dependency. "
                    "TMEM138/JBTS16 has TMEM216 co-dependency as a mandatory counselling point; TMEM231/"
                    "JBTS20 does not. WES must distinguish by chromosomal locus: 11q vs 16q. Phenotypic "
                    "penetrance rates are also distinct: TMEM138/JBTS16 renal ~22%; TMEM231/JBTS20 renal ~25%."
                ),
            },
            {
                "title": "Intermediate Severity Position: JBTS20 Between B9 Module (JBTS19) and Tectonic Module (JBTS18)",
                "detail": (
                    "TMEM231/JBTS20 occupies an intermediate severity position consistent with its bridge role. "
                    "Renal penetrance: B9D1/JBTS19 35% > TMEM231/JBTS20 25% > TCTN3/JBTS18 20%. This gradient "
                    "reflects degree of TZ gate disruption: complete B9 complex collapse (JBTS19, maximum) > "
                    "B9-Tectonic bridge loss (JBTS20, intermediate) > partial Tectonic complex disruption "
                    "(JBTS18, minimum). Clinicians should not apply JBTS19 penetrance rates to JBTS20 patients — "
                    "renal and retinal surveillance intensity should be calibrated to JBTS20's intermediate risk, "
                    "not the higher-risk JBTS19 profile."
                ),
            },
            {
                "title": "Renal Penetrance 25%: Annual NPHP Protocol Mandatory — Intermediate Risk Profile",
                "detail": (
                    "JBTS20/TMEM231 has ~25% renal penetrance (NPHP-like TIN) — intermediate between B9D1/"
                    "JBTS19 (35%) and TCTN3/JBTS18 (20%). Annual surveillance mandatory from diagnosis: "
                    "creatinine, cystatin C, urine osmolality, microalbuminuria (NPHP-like TIN is a "
                    "concentrating defect — not a proteinuric disease; monitoring must begin before "
                    "proteinuria). ESRD median ~25 yr. Renal transplant curative, no allograft recurrence "
                    "(cell-autonomous AR ciliopathy). Do NOT apply JBTS19 (35% renal) surveillance intensity "
                    "to JBTS20 — over-surveillance burdens families; do not apply JBTS18 (20%) risk — "
                    "under-surveillance misses the 25% penetrance."
                ),
            },
            {
                "title": "Retinal Penetrance 22% and Polydactyly 12%: Annual ERG from Age 3 Mandatory",
                "detail": (
                    "JBTS20/TMEM231 retinal penetrance (22%) is slightly lower than TMEM138/JBTS16 (25%) and "
                    "TCTN3/JBTS18 (25%). Rod-cone dystrophy pattern (ERG: rod-dominated early, cone dysfunction "
                    "progressive). Annual ERG from age 3 is mandatory regardless of visual symptoms — ERG "
                    "detects subclinical rod-cone dysfunction years before fundus changes or visual acuity loss. "
                    "Post-axial polydactyly (12%) is present in a minority; skeletal survey mandatory in "
                    "polydactyly cases to exclude broader ciliopathy skeletal involvement."
                ),
            },
        ],

        "literature_highlights": [
            "Shi X et al. (2017) Super-resolution microscopy reveals that disruption of ciliary transition-zone architecture causes Joubert syndrome. Nat Cell Biol 19(10):1178–88. [TMEM231 TZ localisation and architecture].",
            "Sang L et al. (2011) Mapping the NPHP-JBTS-MKS protein network reveals ciliopathy disease genes and pathways. Cell 145(4):513–28. [TMEM231 in ciliopathy network].",
            "Huang L et al. (2011) TMEM231, mutated in orofaciodigital and Joubert syndromes, is required for the ciliary transition zone. J Cell Biol 194(4):491–505. [TMEM231 primary JBTS20 discovery paper].",
            "Parisi MA (2019) The molecular genetics of Joubert syndrome and related ciliopathies. Transl Sci Rare Dis 4(1-2):25–49.",
            "Bachmann-Gagescu R et al. (2020) JBTS disease gene landscape across 460 families. Hum Mutat 41(4):e1–e45.",
        ],

        "phenotype_frequencies": {
            "mts_pathognomonic":       "100% (MTS is the diagnostic criterion)",
            "cerebellar_ataxia":       f"{_pct(n_ataxia)}%",
            "neonatal_hypotonia":      f"{_pct(n_hypotonia)}%",
            "oculomotor_apraxia":      f"{_pct(n_oma)}%",
            "breathing_dysregulation": f"{_pct(n_breath)}%",
            "intellectual_disability": f"{_pct(n_id)}%",
            "retinal_rod_cone":        f"{_pct(n_retinal)}%",
            "renal_nphp_tin":          f"{_pct(n_renal)}%",
            "hepatic_chf":             f"{_pct(n_hepatic)}%",
            "polydactyly_post_axial":  f"{_pct(n_poly)}%",
            "corpus_callosum_anomaly": f"{_pct(n_cc)}%",
            "no_mks_tier":             "Confirmed — biallelic null/null → JBTS20 live birth (NO Meckel-Gruber risk)",
            "b9_tectonic_bridge":      "Confirmed — TMEM231 bridges B9 complex (inner leaflet) and Tectonic complex (lipid gate)",
        },
    }
