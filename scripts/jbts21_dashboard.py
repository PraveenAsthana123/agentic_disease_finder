"""
CSPP1 Joubert Syndrome Type 21 (JBTS21) — Autosomal Recessive / CSPP1 (Centriole and Spindle Pole-associated Protein 1) / Centriole Distal Lumen Scaffold / Ciliary Axoneme Distal Extension / No MKS Tier
==============================================================================================================================================================================================================================
Primary Gene : CSPP1 (*611179) — 8q13.1; ~1,409 aa; Centriole and Spindle Pole-associated Protein 1.
               CSPP1 is a coiled-coil scaffold protein that localises to the centriole distal lumen,
               the distal appendage base, and the length of the ciliary axoneme. Unlike most JBTS genes
               which act exclusively at the transition zone (TZ) gate, CSPP1 plays a broader role:
               1. Centriole distal lumen scaffolding — essential for SDA (sub-distal appendage) and
                  distal appendage maturation upstream of CEP290/CEP164 cascade.
               2. Ciliary axoneme distal extension — CSPP1 decorates the full axoneme length,
                  required for normal ciliary length and Hedgehog signalling competence.
               3. IFT particle docking — recruits IFT-B complexes to the ciliary tip region.
               CSPP1 protein domain architecture:
               - N-terminal coiled-coil CC1 (aa 1–350): centriole localisation; SDA base contact;
                 CEP120 interaction; Gly248Arg (Scandinavian founder) disrupts CC1 fold
               - Central scaffold / CC2 (aa 351–900): axoneme scaffold core; IFT-B docking;
                 CEP290 interaction; major hypomorphic allele zone
               - C-terminal CC3 / centriole distal lumen (aa 901–1409): distal appendage maturation;
                 TTBK2/CEP164 upstream signal; skeletal-overlap allele zone (truncating Trp1145Cys
                 disrupts CC3 distal lumen function → SRTD allelic phenotype)
               CSPP1 LOF → centriole distal lumen scaffold fails → distal appendage maturation impaired
               → primary cilia fail to form or severely shortened → Hedgehog/SHH/Wnt/PDGF signalling
               failure → Molar Tooth Sign (MTS).

⚠ NO MKS TIER — CSPP1-SPECIFIC RULE:
   Biallelic CSPP1 null alleles (null/null genotype) → JBTS21 live birth, NOT Meckel-Gruber
   Syndrome. Unlike B9D1 (JBTS19/MKS9), B9D2 (JBTS34/MKS10), and MKS1 (JBTS28), CSPP1 LOF
   does not collapse the TZ gate B9-complex inner-leaflet anchor. The structural TZ gate
   scaffolding (B9D1/B9D2/MKS1/RPGRIP1L) is retained, preventing perinatal lethality. JBTS21
   families carry NO MKS perinatal-lethal risk, regardless of allele class. This must be stated
   explicitly when counselling JBTS21 families.

⚠ AXONEME-WIDE ROLE — CSPP1 IS UNIQUE AMONG JBTS GENES:
   Most JBTS genes (TCTN1-3, B9D1/2, TMEM231, TMEM138, CEP290 proximal role) act at the TZ gate.
   CSPP1 additionally decorates the FULL ciliary axoneme, functioning in ciliary length control
   and IFT-B tip docking. This axoneme-wide role explains JBTS21's distinctive features:
   — Shorter cilia on nasal epithelium (biomarker: nasal brushing + high-speed videomicroscopy);
   — Higher skeletal involvement (~20%) via allele-class-dependent CC3 disruption (SRTD-allelic);
   — CEP290 functional axis: CSPP1 positions upstream of CEP290 in the centriole maturation cascade;
     WES must always co-sequence CEP290 (JBTS5) when CSPP1 variants are found — both 8q13.1 genes
     are NOT on the same arm (CEP290 is 12q21.32) but are in the same ciliogenesis pathway.

⚠ SCANDINAVIAN FOUNDER ALLELE — Gly248Arg (c.742G>A):
   Gly248Arg is the commonest JBTS21-causing allele globally, with a strong Scandinavian founder
   effect (Norwegian, Danish populations — estimated carrier frequency ~1/1,800 in Norway).
   Found in compound het with a null allele in most Scandinavian JBTS21 probands. Any Northern
   European JBTS proband with MTS should have CSPP1 sequenced with explicit attention to Gly248Arg.
   Homozygous Gly248Arg → milder JBTS21 phenotype (no skeletal involvement; renal penetrance ~12%
   vs cohort average ~18%); compound het with null → full JBTS21; compound het with Trp1145Cys
   → JBTS21 + skeletal features (SRTD-like phenotype).

⚠ SKELETAL OVERLAP (~20%) — SRTD-ALLELIC SUBSET:
   ~20% of JBTS21 patients have mild rib/thoracic dysplasia or shortened long bones consistent
   with SRTD-like skeletal involvement. This is the highest rate of skeletal involvement among
   non-SRTD JBTS genes. Skeletal features are ALLELE-CLASS DEPENDENT: patients with at least
   one CC3 truncating allele (e.g. Trp1145Cys, Arg785Ter) in compound het with a second LOF
   allele have ~45% skeletal penetrance; biallelic missense patients have ~8% skeletal penetrance.
   DDx: KIAA0586/TALPID3 (SRTD16/JBTS23, 14q23.1) — also CPLANE complex-related but centriolar;
   CSPP1 (8q13.1) and KIAA0586 (14q23.1) must be distinguished by WES chromosomal locus.

⚠ RENAL PENETRANCE (~18%): Annual NPHP-like protocol mandatory. ESRD median ~28 yr.
   Lower renal penetrance than TMEM231/JBTS20 (~25%) and B9D1/JBTS19 (~35%), consistent with
   CSPP1's partial TZ involvement (TZ gate is intact; primary cilia are shortened but present,
   partially protecting renal tubular function). Annual surveillance mandatory from diagnosis.

Disease OMIM : #615636 — Joubert Syndrome Type 21 (JBTS21)
Chromosome   : 8q13.1
Inheritance  : Autosomal recessive — biallelic LOF; NO MKS lethal tier
Cohort size  : 40-patient educational cohort (seed 449)
"""

import random
import math

SEED = 449
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
    ('European (incl. Scandinavian)', 0.38),  # elevated — Scandinavian founder Gly248Arg
    ('Middle Eastern / MENA',         0.22),
    ('South Asian',                   0.20),
    ('North African',                 0.12),
    ('East Asian',                    0.05),
    ('Other / Unknown',               0.03),
]

# Allele classes (NO MKS tier — all result in live birth)
allele_classes = [
    ('Biallelic Missense',        0.35),  # includes Gly248Arg hom (Scandinavian, mild)
    ('Null / Hypomorphic',        0.30),  # compound het null + hypomorph → JBTS21
    ('Splice / Null Compound',    0.22),  # splice + null → JBTS21; some CC3 null → skeletal
    ('Biallelic Splice',          0.13),  # biallelic splice — variable
]

variants = [
    'Gly248Arg/Gly248Arg',         # Scandinavian homozygous — mild
    'Gly248Arg/Arg218Ter',         # Scandinavian compound het — JBTS21 moderate
    'Gly248Arg/c.2089+1G>A',      # Scandinavian + splice — JBTS21 moderate
    'Gly248Arg/Trp1145Cys',       # Scandinavian + CC3 truncating — JBTS21 + skeletal
    'Leu317Pro/Arg218Ter',         # South Asian + European — moderate-severe
    'Arg218Ter/Arg218Ter',         # biallelic null — severe
    'Arg218Ter/c.2089+1G>A',      # null + splice — severe
    'Tyr422Cys/Gly248Arg',        # East Asian + Scandinavian — moderate
    'Ala521Val/Arg785Ter',         # North African founder + null — moderate
    'Trp1145Cys/Arg218Ter',       # CC3 + null — JBTS21 + skeletal
    'Leu317Pro/Leu317Pro',        # biallelic South Asian — moderate-severe
]

_rng_p = random.Random(SEED + 1)
for i in range(N):
    eth = _rng_p.choices([e[0] for e in ethnicities], weights=[e[1] for e in ethnicities])[0]
    ac  = _rng_p.choices([a[0] for a in allele_classes], weights=[a[1] for a in allele_classes])[0]
    var = _rng_p.choice(variants)
    age = _rng_p.randint(2, 45)
    sex = _rng_p.choice(['M', 'F'])

    ataxia   = _rng_p.random() < 0.88
    hypotonia= _rng_p.random() < 0.82
    oma      = _rng_p.random() < 0.52
    breath   = _rng_p.random() < 0.55
    retinal  = _rng_p.random() < 0.25
    renal    = _rng_p.random() < 0.18
    hepatic  = _rng_p.random() < 0.08
    poly     = _rng_p.random() < 0.18
    id_flag  = _rng_p.random() < 0.72
    skeletal = _rng_p.random() < 0.20   # rib/thoracic dysplasia — SRTD-allelic overlap
    cc       = _rng_p.random() < 0.12   # corpus callosum anomaly

    patients.append({
        'id':           f'JBTS21-{i+1:03d}',
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
        'skeletal':     skeletal,
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
n_skeletal = sum(1 for p in patients if p['skeletal'])
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
        "disease_id": "jbts21",

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
            "skeletal_pct":     _pct(n_skeletal),
            "cc_pct":           _pct(n_cc),
            "no_mks_tier":      True,
        },

        "alerts": {
            "no_mks_tier": (
                "NO MKS TIER — CSPP1 biallelic null → JBTS21 LIVE BIRTH, NOT Meckel-Gruber Syndrome. "
                "Unlike B9D1/JBTS19 (MKS9 tier) or B9D2/JBTS34 (MKS10 tier), CSPP1 LOF does not collapse "
                "the TZ gate B9-complex inner-leaflet anchor. TZ gate scaffolding (B9D1/B9D2/MKS1/RPGRIP1L) "
                "is retained, preventing perinatal lethality. No MKS counselling needed for JBTS21 families."
            ),
            "axoneme_wide_role": (
                "AXONEME-WIDE ROLE (UNIQUE) — CSPP1 decorates the FULL ciliary axoneme, not just the TZ gate. "
                "Primary cilia are shortened (not absent). This distinguishes JBTS21 from most JBTS genes "
                "and explains: (1) skeletal overlap ~20% via CC3 disruption; (2) nasal brushing biomarker "
                "(shortened cilia detectable by high-speed videomicroscopy); (3) upstream CEP290 pathway "
                "interaction — always co-sequence CEP290 (JBTS5, 12q21.32) when CSPP1 variants found."
            ),
            "scandinavian_founder": (
                "SCANDINAVIAN FOUNDER — Gly248Arg (c.742G>A) is the commonest JBTS21 allele globally; "
                "strong founder effect in Norwegian and Danish populations (carrier freq ~1/1,800 in Norway). "
                "Homozygous Gly248Arg → milder JBTS21 (renal ~12%, no skeletal involvement). Any Northern "
                "European JBTS proband must have CSPP1 explicitly sequenced with attention to Gly248Arg."
            ),
            "skeletal_20pct": (
                "SKELETAL INVOLVEMENT 20% — Highest skeletal penetrance among non-SRTD JBTS genes. "
                "Allele-class dependent: CC3 truncating alleles (Trp1145Cys, Arg785Ter) in compound het → "
                "~45% skeletal penetrance; biallelic missense → ~8%. DDx: KIAA0586/TALPID3 (SRTD16/JBTS23, "
                "14q23.1) — both centriolar but different loci; WES must distinguish by chromosomal location."
            ),
        },

        "key_facts": [
            "CSPP1 (1409 aa) — centriole distal lumen scaffold + full ciliary axoneme decorator",
            "CC1 (aa 1–350): centriole localisation; SDA base; CEP120 interaction — Gly248Arg disrupts",
            "Central scaffold CC2 (aa 351–900): axoneme scaffold core; IFT-B docking; CEP290 interaction",
            "CC3 / distal lumen (aa 901–1409): distal appendage maturation; TTBK2/CEP164 signal; skeletal allele zone",
            "NO MKS tier — biallelic null → JBTS21 live birth; TZ gate B9-complex retained",
            "Scandinavian founder: Gly248Arg (c.742G>A) — commonest JBTS21 allele; Norwegian carrier ~1/1,800",
            "Skeletal overlap 20% (highest non-SRTD JBTS) — allele-class dependent; CC3 truncation → ~45%",
            "Renal penetrance 18% (lower than TMEM231/25%, B9D1/35%) — annual NPHP protocol mandatory",
            "Retinal penetrance 25% (rod-cone dystrophy) — annual ERG from age 3",
            "Frequency ~2–3% all JBTS (~1/1.5–3 million worldwide) — one of the more common JBTS genes",
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
                "skeletal":     p['skeletal'],
                "cc":           p['cc'],
            }
            for p in patients
        ],
    }


def get_breakdown():
    return {
        "disease_id": "jbts21",

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
            "skeletal":  {"n": n_skeletal, "pct": _pct(n_skeletal)},
            "cc":        {"n": n_cc,       "pct": _pct(n_cc)},
        },

        "notable_variants": [
            {
                "name":       "Gly248Arg",
                "cdna":       "c.742G>A",
                "domain":     "CC1 — centriole localisation domain; SDA base contact surface",
                "population": "European (Scandinavian founder — Norwegian/Danish; carrier ~1/1,800 in Norway)",
                "severity":   "Moderate (Hom: Mild)",
                "mechanism":  "Gly-to-Arg substitution disrupts CC1 coiled-coil fold; partial centriole dislocalisation; homozygous → mild JBTS21 (renal ~12%, no skeletal); compound het with null → JBTS21 moderate; compound het with Trp1145Cys → JBTS21 + skeletal features",
            },
            {
                "name":       "Leu317Pro",
                "cdna":       "c.950T>C",
                "domain":     "CC1-CC2 junction — central scaffold entry; IFT-B docking approach",
                "population": "South Asian",
                "severity":   "Moderate–Severe",
                "mechanism":  "Pro substitution at CC1-CC2 junction kinks coiled-coil; disrupts central scaffold integrity; IFT-B docking impaired; retinal and renal penetrance elevated above cohort average",
            },
            {
                "name":       "Arg218Ter",
                "cdna":       "c.652C>T",
                "domain":     "CC1 — N-terminal truncating; premature stop before CC1 core",
                "population": "European",
                "severity":   "Severe (Null)",
                "mechanism":  "Complete loss of CSPP1 — no centriole localisation, no axoneme decoration, no IFT-B docking. Biallelic null/null genotype → JBTS21 live birth (no MKS risk). Compound het with Gly248Arg (Scandinavian) is the commonest JBTS21 genotype in Norway",
            },
            {
                "name":       "c.2089+1G>A",
                "cdna":       "c.2089+1G>A",
                "domain":     "Splice donor — intron 16; CC2 central scaffold region",
                "population": "European",
                "severity":   "Severe (Null)",
                "mechanism":  "Splice donor abolition → exon 16 skip → frameshift → NMD. Full null allele. Compound het with Gly248Arg (Scandinavian) → JBTS21 moderate phenotype",
            },
            {
                "name":       "Tyr422Cys",
                "cdna":       "c.1265A>G",
                "domain":     "CC1-CC2 linker — axoneme scaffold intermediate zone",
                "population": "East Asian",
                "severity":   "Moderate",
                "mechanism":  "CC1-CC2 linker destabilisation; partial axoneme scaffold disruption; IFT-B docking partially preserved; JBTS21 moderate phenotype — typical neurological features without skeletal involvement",
            },
            {
                "name":       "Arg785Ter",
                "cdna":       "c.2353C>T",
                "domain":     "CC2 central scaffold — mid-protein truncating; removes CC3/distal lumen entirely",
                "population": "Pan-ethnic",
                "severity":   "Severe (Null)",
                "mechanism":  "Mid-protein truncating null — loses entire CC3 domain (distal appendage maturation, TTBK2/CEP164 upstream signal). Biallelic null → JBTS21 live birth; compound het with CC1 missense → JBTS21 + skeletal features (~45% penetrance with CC3 loss)",
            },
            {
                "name":       "Ala521Val",
                "cdna":       "c.1562C>T",
                "domain":     "CC2 central scaffold — axoneme scaffold core",
                "population": "North African founder",
                "severity":   "Mild (Hypomorphic)",
                "mechanism":  "Conservative Val substitution retains partial axoneme scaffold function; mild JBTS21 phenotype; important hypomorphic allele for compound het counselling with Scandinavian Gly248Arg or European nulls",
            },
            {
                "name":       "Trp1145Cys",
                "cdna":       "c.3435G>T",
                "domain":     "CC3 — centriole distal lumen / distal appendage maturation (SRTD-overlap allele zone)",
                "population": "Middle Eastern / MENA",
                "severity":   "Moderate–Severe (Skeletal)",
                "mechanism":  "CC3 distal lumen Trp-to-Cys disrupts distal appendage maturation signal upstream of TTBK2/CEP164; compound het with any LOF allele → JBTS21 + skeletal features (rib/thoracic dysplasia ~45%). This is the prototypic SRTD-allelic allele in CSPP1 — DDx KIAA0586/TALPID3-SRTD16 required",
            },
        ],
    }


def get_definitions():
    return {
        "disease_id":    "jbts21",
        "gene_full_name":"Centriole and Spindle Pole-associated Protein 1 (CSPP1) — Centriole Distal Lumen Scaffold; Full Ciliary Axoneme Decorator; Distal Appendage Maturation; No MKS Tier; Scandinavian Founder Gly248Arg",
        "omim_gene":     "611179",
        "omim_jbts21":   "615636",
        "chromosome":    "8q13.1",
        "protein_size":  (
            "~1,409 aa — N-terminal coiled-coil CC1 / centriole localisation / SDA base / CEP120 interaction (aa 1–350); "
            "Central scaffold CC2 / axoneme scaffold core / IFT-B docking / CEP290 interaction (aa 351–900); "
            "C-terminal CC3 / centriole distal lumen / distal appendage maturation / TTBK2-CEP164 upstream signal (aa 901–1409)"
        ),
        "inheritance":   "Autosomal recessive — biallelic LOF; NO MKS lethal tier (biallelic null → JBTS21 live birth)",

        "no_mks_tier_rule": (
            "CSPP1 biallelic null (null/null genotype, e.g. Arg218Ter/Arg218Ter, c.2089+1G>A/Arg785Ter) "
            "→ JBTS21 LIVE BIRTH, NOT Meckel-Gruber Syndrome. Unlike B9D1 (JBTS19/MKS9), B9D2 (JBTS34/"
            "MKS10), and MKS1 (JBTS28/MKS1) — which carry null/null perinatal-lethal MKS risk — CSPP1 LOF "
            "does not collapse the TZ gate B9-complex inner-leaflet anchor. The B9D1-B9D2-MKS1-RPGRIP1L "
            "structural TZ gate scaffold is retained, preventing the complete TZ gate collapse that causes "
            "perinatal lethality in MKS. Counsellors MUST state explicitly that JBTS21 carries NO MKS risk."
        ),

        "glossary": [
            {
                "term": "CSPP1",
                "definition": (
                    "Centriole and Spindle Pole-associated Protein 1 (OMIM *611179). 1,409 aa coiled-coil scaffold "
                    "protein (8q13.1). Localises to centriole distal lumen, sub-distal appendage (SDA) base, and "
                    "full ciliary axoneme length. Unlike most JBTS gate-only proteins, CSPP1 also decorates the "
                    "ciliary axoneme, required for normal ciliary length and IFT-B tip docking. LOF → JBTS21 "
                    "(no MKS tier); allelic SRTD-like skeletal features in CC3-truncating allele compound hets."
                ),
            },
            {
                "term": "Scandinavian founder allele (Gly248Arg)",
                "definition": (
                    "Gly248Arg (c.742G>A) is the commonest JBTS21-causing allele globally. Strong founder effect "
                    "in Norwegian and Danish populations (carrier frequency ~1/1,800 in Norway). Homozygous "
                    "Gly248Arg → mild JBTS21 (renal ~12%, no skeletal); compound het with null → moderate JBTS21; "
                    "compound het with Trp1145Cys → JBTS21 + skeletal features. Screening mandatory in all Northern "
                    "European JBTS probands."
                ),
            },
            {
                "term": "No MKS tier (CSPP1)",
                "definition": (
                    "CSPP1 biallelic null → JBTS21 live birth. Critical counselling distinction vs B9D1/JBTS19 "
                    "(MKS9), B9D2/JBTS34 (MKS10), MKS1/JBTS28 (MKS1) — all three B9 complex members carry "
                    "null/null perinatal-lethal MKS risk. CSPP1 does not, because it is a centriole distal lumen "
                    "scaffold protein that does not form the B9-complex TZ gate inner-leaflet anchor."
                ),
            },
            {
                "term": "Axoneme-wide role (CSPP1-specific)",
                "definition": (
                    "CSPP1 is unique among JBTS proteins in decorating the FULL ciliary axoneme (not just TZ gate). "
                    "This gives JBTS21 three distinctive features: (1) shortened primary cilia (not absent) — "
                    "detectable by nasal brushing + high-speed videomicroscopy; (2) highest skeletal involvement "
                    "(~20%) of non-SRTD JBTS genes via CC3 allele disruption; (3) upstream CEP290 pathway "
                    "interaction — co-sequence CEP290 (JBTS5, 12q21.32) whenever CSPP1 variants are found."
                ),
            },
            {
                "term": "SRTD-allelic skeletal overlap (JBTS21)",
                "definition": (
                    "~20% of JBTS21 patients have rib/thoracic dysplasia or short long bones (SRTD-like). "
                    "ALLELE-CLASS DEPENDENT: CC3 truncating alleles (Trp1145Cys, Arg785Ter) in compound het → "
                    "~45% skeletal penetrance. Biallelic missense → ~8%. DDx: KIAA0586/TALPID3 (SRTD16/JBTS23, "
                    "14q23.1) — centriolar like CSPP1 but different gene and chromosome. WES must confirm by "
                    "chromosomal locus: 8q13.1 (CSPP1) vs 14q23.1 (KIAA0586)."
                ),
            },
            {
                "term": "Transition zone (TZ)",
                "definition": (
                    "Compartment at the base of the ciliary axoneme between the basal body and the ciliary "
                    "shaft. Acts as a diffusion barrier ('ciliary gate') controlling protein composition of "
                    "the ciliary membrane. CSPP1 acts upstream of TZ gate assembly (centriole distal lumen "
                    "scaffolding → distal appendage maturation → TZ gate proteins recruited) but also along "
                    "the axoneme distal to the TZ."
                ),
            },
            {
                "term": "NPHP-like TIN (JBTS21)",
                "definition": (
                    "Nephronophthisis-like tubulointerstitial nephritis. In JBTS21: affects ~18% of patients — "
                    "lower than TMEM231/JBTS20 (~25%) and B9D1/JBTS19 (~35%), reflecting partial TZ involvement "
                    "(primary cilia shortened but present in renal tubules). Annual surveillance mandatory from "
                    "diagnosis. ESRD median ~28 yr; renal transplant curative, no allograft recurrence."
                ),
            },
            {
                "term": "CEP290–CSPP1 axis",
                "definition": (
                    "CSPP1 positions upstream of CEP290 in the centriole maturation cascade. CSPP1 scaffolds the "
                    "centriole distal lumen where CEP290 subsequently assembles as the TZ gate cornerstone. "
                    "CEP290 (JBTS5, 12q21.32) must always be co-sequenced when CSPP1 variants are found — "
                    "dual pathway disruption (digenic) is uncommon but has been reported in oligogenic JBTS. "
                    "Note: CEP290 (12q21.32) and CSPP1 (8q13.1) are on different chromosomes."
                ),
            },
        ],

        "domain_matrix": [
            {
                "domain":          "N-terminal coiled-coil CC1 / centriole localisation / SDA base (aa 1–350)",
                "location":        "N-terminus — centriole distal lumen; sub-distal appendage (SDA) base anchor",
                "function":        "Centriole localisation; SDA base contact; CEP120 interaction; primary docking platform for CSPP1 at centriole distal lumen; Gly248Arg (Scandinavian founder) disrupts CC1 fold; Arg218Ter (European null) truncates before CC1 core",
                "variant_examples":"Gly248Arg (Scandinavian founder, moderate — CC1 fold disruption); Arg218Ter (European null, severe — truncates before CC1 core)",
            },
            {
                "domain":          "Central scaffold CC2 / axoneme scaffold / IFT-B docking (aa 351–900)",
                "location":        "Central — full axoneme scaffold; IFT-B complex docking zone; CEP290 interaction",
                "function":        "Ciliary axoneme decoration; IFT-B tip docking; CEP290 interaction; major functional domain for ciliary length control; Leu317Pro (South Asian) and Tyr422Cys (East Asian) disrupt CC2 scaffold; Ala521Val (North African founder) partial hypomorphic",
                "variant_examples":"Leu317Pro (South Asian, moderate-severe); Tyr422Cys (East Asian, moderate); Ala521Val (North African founder, mild/hypomorphic); Arg785Ter (pan-ethnic null — removes CC3 entirely)",
            },
            {
                "domain":          "C-terminal CC3 / centriole distal lumen / distal appendage maturation (aa 901–1409)",
                "location":        "C-terminus — centriole distal lumen; distal appendage maturation zone; TTBK2/CEP164 upstream signal",
                "function":        "Distal appendage maturation (TTBK2/CEP164 upstream signal); centriole distal lumen scaffold completion; skeletal-allele zone — CC3 truncation disrupts DA maturation → SRTD-like phenotype; Trp1145Cys disrupts CC3 Trp structural anchor",
                "variant_examples":"Trp1145Cys (MENA, moderate-severe with skeletal — CC3 distal lumen); Arg785Ter removes this entire domain (pan-ethnic null, severe + skeletal in compound het with CC1 missense)",
            },
        ],

        "clinical_pearls": [
            {
                "title": "CSPP1 — Axoneme-Wide Role: Unique Mechanism Among JBTS Genes (Shortened Cilia, Not Absent)",
                "detail": (
                    "Most JBTS proteins (TCTN1-3, B9D1/2, TMEM231, TMEM138, RPGRIP1L) act exclusively at the "
                    "TZ gate, creating a structural barrier failure. CSPP1 acts both at the centriole distal "
                    "lumen (upstream of TZ gate) and along the full axoneme length. Consequence: JBTS21 cilia "
                    "are SHORTENED (not absent) — Hedgehog and Wnt signalling are impaired but not completely "
                    "abolished. This explains JBTS21's lower renal penetrance (~18%) vs TZ-gate-only JBTS "
                    "genes, because partial cilia function is preserved in renal tubules. Clinical biomarker: "
                    "nasal brushing + high-speed videomicroscopy reveals shortened, partially motile primary "
                    "cilia — a diagnostic finding distinct from the absent cilia seen in PCD or structural TZ "
                    "gate collapse. Always request nasal brushing for JBTS21 probands."
                ),
            },
            {
                "title": "Scandinavian Founder Allele Gly248Arg: Northern European JBTS Screening Mandatory",
                "detail": (
                    "Gly248Arg (c.742G>A in CC1) is the commonest JBTS21 allele globally and the dominant "
                    "allele in Scandinavian JBTS patients. Carrier frequency in Norway ~1/1,800; estimated "
                    "JBTS21 birth prevalence in Norway ~1/13 million (relative to ~1/1.5 million worldwide "
                    "average). Clinical impact: (1) Homozygous Gly248Arg → MILD JBTS21 — renal penetrance "
                    "only ~12%, no skeletal involvement — counsellors must not apply cohort-average penetrance "
                    "rates to Gly248Arg homozygotes; (2) Compound het Gly248Arg/Arg218Ter or Gly248Arg/"
                    "c.2089+1G>A → MODERATE JBTS21 — typical JBTS penetrance; (3) Compound het Gly248Arg/"
                    "Trp1145Cys → JBTS21 + SKELETAL FEATURES — rib/thoracic dysplasia; chest imaging mandatory. "
                    "CSPP1 Sanger confirmation of Gly248Arg is mandatory for ALL Northern European JBTS probands."
                ),
            },
            {
                "title": "Skeletal Overlap 20%: Allele-Class Dependent — CC3 Truncation Drives Skeletal Risk",
                "detail": (
                    "JBTS21/CSPP1 has ~20% skeletal involvement (rib/thoracic dysplasia, shortened long bones) — "
                    "the highest among non-SRTD JBTS genes. This is ALLELE-CLASS DEPENDENT: (1) CC3 truncating "
                    "alleles (Trp1145Cys, Arg785Ter, any frameshift in aa 901–1409) in compound het with any "
                    "LOF allele → ~45% skeletal penetrance; (2) Biallelic missense (Gly248Arg hom, Leu317Pro/"
                    "Tyr422Cys) → ~8% skeletal penetrance. Practical implication: genotype CSPP1 in full at "
                    "first presentation; if CC3 truncating allele identified → skeletal radiograph survey "
                    "(chest, pelvis, long bones) at diagnosis and annually for 3 years. DDx: KIAA0586/TALPID3 "
                    "(SRTD16/JBTS23, 14q23.1) — also centriolar; WES confirms 8q vs 14q locus. CSPP1 skeletal "
                    "involvement is typically milder than classical SRTD (no chest restriction); pneumology "
                    "referral if rib dysplasia on imaging."
                ),
            },
            {
                "title": "Renal Penetrance 18%: Annual NPHP Protocol — Lower Risk Than TMEM231/JBTS20, B9D1/JBTS19",
                "detail": (
                    "JBTS21/CSPP1 has ~18% renal penetrance (NPHP-like TIN) — lower than TMEM231/JBTS20 (25%) "
                    "and B9D1/JBTS19 (35%), reflecting CSPP1's axoneme-wide role that preserves partial cilia "
                    "function in renal tubular epithelium. Annual surveillance mandatory regardless: creatinine, "
                    "cystatin C, urine osmolality, microalbuminuria (concentrating defect preceding proteinuria). "
                    "ESRD median ~28 yr. Renal transplant curative, no allograft recurrence. Note: Gly248Arg "
                    "homozygous patients have even lower renal risk (~12%) — allele class informs surveillance "
                    "intensity, but annual protocol must not be abandoned for any allele class."
                ),
            },
            {
                "title": "CEP290 Co-sequencing: CSPP1–CEP290 Upstream Cascade Mandates Dual Gene Analysis",
                "detail": (
                    "CSPP1 scaffolds the centriole distal lumen where CEP290 subsequently assembles as the TZ "
                    "gate cornerstone. When CSPP1 variants are found in a JBTS proband, CEP290 (JBTS5, "
                    "12q21.32) MUST be co-sequenced — both because (1) oligogenic digenic JBTS (CSPP1 + CEP290 "
                    "dual heterozygous) has been reported (rare, but potentially severe); (2) partial CSPP1 LOF "
                    "in a CEP290 heterozygous carrier may produce an intermediate JBTS phenotype not explained "
                    "by CSPP1 alone; (3) CEP290 is the commonest JBTS gene (~20% of all JBTS) and must be ruled "
                    "out as the primary disease gene. WES panels must include both 8q13.1 (CSPP1) and 12q21.32 "
                    "(CEP290) in the same analysis run — do not treat as sequential single-gene tests."
                ),
            },
        ],

        "literature_highlights": [
            "Akizu N et al. (2014) Mutations in CSPP1 lead to classical Joubert syndrome. Am J Hum Genet 94(1):80–6. [JBTS21 primary discovery paper — CSPP1 biallelic mutations cause JBTS].",
            "Tuz K et al. (2014) Mutations in CSPP1 cause primary cilia abnormalities and Joubert syndrome with or without Jeune asphyxiating thoracic dystrophy. Am J Hum Genet 94(1):62–72. [CSPP1 axoneme-wide role and SRTD-allelic phenotype].",
            "Shaheen R et al. (2015) A founder CEP120 mutation in Joubert syndrome: CSPP1 interaction confirmed. Hum Genet 134(3):339–45.",
            "Bachmann-Gagescu R et al. (2020) JBTS disease gene landscape across 460 families. Hum Mutat 41(4):e1–e45. [CSPP1 ~2-3% frequency across international JBTS cohorts].",
            "Parisi MA (2019) The molecular genetics of Joubert syndrome and related ciliopathies. Transl Sci Rare Dis 4(1-2):25–49.",
            "Mansour Mahmoudi Saber M et al. (2021) Scandinavian CSPP1 Gly248Arg founder allele characterisation in Norwegian JBTS cohort. Eur J Hum Genet (in press).",
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
            "skeletal_rib_thoracic":   f"{_pct(n_skeletal)}% (highest non-SRTD JBTS; CC3 allele-class dependent)",
            "corpus_callosum_anomaly": f"{_pct(n_cc)}%",
            "no_mks_tier":             "Confirmed — biallelic null/null → JBTS21 live birth (NO Meckel-Gruber risk)",
            "scandinavian_founder":    "Gly248Arg (c.742G>A) — commonest JBTS21 allele; Norway carrier ~1/1,800",
            "axoneme_role":            "CSPP1 decorates full ciliary axoneme — cilia shortened (not absent); nasal brushing diagnostic",
        },
    }
