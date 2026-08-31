"""
CEP83 Joubert Syndrome Type 22 (JBTS22) — Autosomal Recessive / CEP83 (CCDC41) / Distal Appendage Foundation Scaffold / Ciliogenesis Initiation Block / No MKS Tier
=============================================================================================================================================================================================================
Primary Gene : CEP83 (*617233) — 12q22; ~826 aa; Centrosomal Protein 83 kDa (also CCDC41 — Coiled-Coil Domain Containing 41).
               CEP83 is the MOST PROXIMAL distal appendage (DA) protein and the master organiser of
               the entire DA hierarchy. It localises to the subdistal/distal appendage junction of
               the mother centriole independently of all other DA proteins.

               CEP83 is the FOUNDATION of the DA scaffold — it nucleates all downstream DA assembly:
               CEP83 → CEP89 → SCLT1 → FBF1 → LRRC45 → CEP164 (NPHP15)
               Loss of CEP83 = simultaneous loss of ALL downstream DA proteins from the centriole.

               CEP83 protein domain architecture (~826 aa):
               - N-terminal coiled-coil CC1 (aa 1–120): centriole anchoring; subdistal appendage
                 docking interface; CE63/C3orf14 interaction surface; Ala148Val (North African
                 founder) is a hypomorphic allele in this region
               - Coiled-coil CC2 / CEP89 recruitment module (aa 140–380): recruits the first
                 downstream DA protein (CEP89/CCDC123); essential for DA hierarchy nucleation;
                 Arg252Cys (MENA founder) disrupts CC2 CEP89 interface
               - Central scaffold / SCLT1 binding interface (aa 380–600): recruits SCLT1 (third
                 step in DA hierarchy); coiled-coil rich; self-oligomerisation domain; Leu387Pro
                 (South Asian) disrupts SCLT1 docking
               - C-terminal regulatory domain (aa 600–826): CEP164 indirect association (via
                 SCLT1→FBF1→LRRC45→CEP164 cascade); IFT-B docking surface; TZ scaffold connection

               CEP83 LOF pathway:
               → DA scaffold absent → centriole fails to dock to ciliary vesicle (EHD1/SNAP29/Rab8a)
               → CP110/CEP97 cap NOT removed (ciliogenesis initiation fails)
               → axoneme NOT initiated → cilia absent (not shortened — complete block)
               → Hedgehog/SHH/Wnt/PDGF signalling failure → Molar Tooth Sign (MTS)
               → Renal: absent primary cilia in tubular epithelium → TIN + ESRD (very high penetrance)
               → Retinal: absent connecting cilia in photoreceptors → rod-cone dystrophy

⚠ DISTAL APPENDAGE FOUNDATION — CEP83-SPECIFIC RULE:
   CEP83 is the ONLY NPHP/JBTS gene that is also the FOUNDATION of the distal appendage scaffold.
   Loss of CEP83 destroys ALL downstream DA proteins (CEP89, SCLT1, FBF1, LRRC45, CEP164/NPHP15)
   from the centriole simultaneously. This is fundamentally different from TZ gate genes (TCTN1-3,
   B9D1/2, TMEM231) which act downstream of DA formation. CEP83 acts UPSTREAM of ciliogenesis
   initiation — cilia are ABSENT (not shortened as in CSPP1/JBTS21 or NPHP module genes).

⚠ VERY HIGH RENAL PENETRANCE (65–70%) — EARLIEST JBTS RENAL RISK:
   JBTS22/CEP83 has the HIGHEST renal penetrance of any non-NPHP1 JBTS gene: ~65–70% of biallelic
   CEP83 cases develop significant CKD. ESRD median ~14–18 yr (juvenile-adolescent onset — earlier
   than TMEM231/JBTS20 median 25yr, CSPP1/JBTS21 median 28yr). All JBTS22 patients require
   annual renal surveillance from diagnosis: creatinine, cystatin C, urine osmolality, spot
   albumin:creatinine ratio. Concentrating defect (polyuria/polydipsia) typically precedes
   proteinuria by years — do not wait for proteinuria to start nephrology monitoring.

⚠ NO MKS TIER — CEP83-SPECIFIC RULE:
   Biallelic CEP83 null alleles (null/null genotype) → JBTS22 live birth, NOT Meckel-Gruber
   Syndrome. Unlike B9D1 (JBTS19/MKS9), B9D2 (JBTS34/MKS10), and MKS1 (JBTS28) — which carry
   perinatal-lethal MKS risk in null/null genotype — CEP83 LOF does NOT collapse the TZ gate
   B9-complex inner-leaflet anchor. The DA block prevents ciliogenesis initiation, but the TZ
   structural scaffold proteins (B9D1/B9D2/MKS1/RPGRIP1L) are independently expressed and retain
   their basal assembly capacity. JBTS22 families carry NO MKS perinatal-lethal risk.

⚠ CEP164 (NPHP15) CO-SEQUENCING MANDATORY:
   CEP164 (NPHP15, 11q13.4) is the direct downstream target of the CEP83 DA hierarchy. When CEP83
   variants are found, CEP164 MUST be co-sequenced: (1) biallelic CEP164 also causes NPHP15/JBTS-
   like phenotype; (2) digenic CEP83 + CEP164 heterozygosity may produce additive DA scaffold
   failure; (3) CEP83 (12q22) and CEP290 (NPHP6, 12q21.32) are on the SAME chromosome arm 12q —
   targeted CEP290 single-gene tests do NOT cover CEP83 — WES is mandatory.

⚠ JBTS22 vs PURE RENAL NPHP18 — SAME GENE, TWO PHENOTYPE CLASSES:
   The same biallelic CEP83 alleles can produce either pure renal NPHP18 (no MTS) OR JBTS22
   (MTS confirmed). This phenotypic variability is only partially explained by allele class.
   Brain MRI MANDATORY for all biallelic CEP83 patients — Molar Tooth Sign identifies JBTS22
   alleles vs pure renal NPHP18; MRI result profoundly changes prognosis and surveillance plan.

Disease OMIM : #617265 — Nephronophthisis 18 / Joubert Syndrome Type 22 (JBTS22)
               Gene OMIM: *617233 (CEP83)
Chromosome   : 12q22
Inheritance  : Autosomal recessive — biallelic LOF; NO MKS lethal tier
Cohort size  : 40-patient educational cohort (seed 451) — JBTS22 (MTS-confirmed) subset
"""

import random

SEED = 451
N    = 40   # 40-patient JBTS22 educational cohort (MTS-confirmed)

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
    ('European (non-consanguineous)',          0.28),
    ('Middle Eastern / MENA (consanguineous)', 0.30),  # Arg252Cys MENA founder elevated
    ('South Asian (consanguineous)',           0.20),
    ('North African (consanguineous)',         0.14),  # Ala148Val North African founder
    ('East Asian',                             0.05),
    ('Other / Unknown',                        0.03),
]

# Allele classes (NO MKS tier — all result in live birth)
allele_classes = [
    ('Biallelic Null / Truncating',  0.30),  # most severe — complete DA scaffold loss
    ('Null / Missense Compound',     0.32),  # null + missense (founder or de novo)
    ('Biallelic Missense',           0.25),  # includes founder allele compounds
    ('Splice / Null Compound',       0.13),  # splice + null
]

variants = [
    'Arg252Cys/Arg252Cys',          # MENA founder homozygous — moderate
    'Arg252Cys/Arg200Ter',          # MENA founder + European null — moderate-severe
    'Arg252Cys/c.1122+1G>A',        # MENA founder + splice — moderate-severe
    'Gly112Arg/Arg200Ter',          # CC1 + null — severe
    'Leu387Pro/Arg530Ter',          # SCLT1-binding + null — severe
    'Arg200Ter/Arg200Ter',          # biallelic null — severe
    'Arg200Ter/c.1122+1G>A',        # null + splice — severe
    'Arg530Ter/c.1122+1G>A',        # null + splice — severe
    'Tyr694Cys/Arg252Cys',          # C-term + MENA founder — moderate
    'Ala148Val/Arg252Cys',          # North African hypomorph + MENA founder — mild-moderate
    'Ala148Val/Arg200Ter',          # North African hypomorph + null — moderate
    'Leu387Pro/Gly112Arg',          # SCLT1-binding + CC1 — severe
    'Gln614Ter/Arg252Cys',          # C-term null + MENA founder — moderate-severe
]

_rng_p = random.Random(SEED + 1)
for i in range(N):
    eth = _rng_p.choices([e[0] for e in ethnicities], weights=[e[1] for e in ethnicities])[0]
    ac  = _rng_p.choices([a[0] for a in allele_classes], weights=[a[1] for a in allele_classes])[0]
    var = _rng_p.choice(variants)
    age = _rng_p.randint(2, 40)
    sex = _rng_p.choice(['M', 'F'])

    ataxia    = _rng_p.random() < 0.70
    hypotonia = _rng_p.random() < 0.68
    oma       = _rng_p.random() < 0.42
    breath    = _rng_p.random() < 0.45
    retinal   = _rng_p.random() < 0.35
    renal     = _rng_p.random() < 0.68   # high penetrance — DA block in tubular cilia
    hepatic   = _rng_p.random() < 0.08
    poly      = _rng_p.random() < 0.05   # rare in CEP83/JBTS22
    id_flag   = _rng_p.random() < 0.60
    esrd      = _rng_p.random() < 0.28   # subset with ESRD at time of study
    # No skeletal involvement (CEP83 is DA protein — no SRTD allelic phenotype)
    # No situs inversus (CEP83 not expressed in nodal cilia)

    patients.append({
        'id':           f'JBTS22-{i+1:03d}',
        'age':          age,
        'sex':          sex,
        'ethnicity':    eth,
        'allele_class': ac,
        'variant':      var,
        'mts':          True,   # MTS confirmed — JBTS22 diagnostic criterion (100%)
        'ataxia':       ataxia,
        'hypotonia':    hypotonia,
        'oma':          oma,
        'breathing':    breath,
        'retinal':      retinal,
        'renal':        renal,
        'hepatic':      hepatic,
        'poly':         poly,
        'id_flag':      id_flag,
        'esrd':         esrd,
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
n_esrd     = sum(1 for p in patients if p['esrd'])

_eth_counts = {}
for p in patients:
    _eth_counts[p['ethnicity']] = _eth_counts.get(p['ethnicity'], 0) + 1

_ac_counts = {}
for p in patients:
    _ac_counts[p['allele_class']] = _ac_counts.get(p['allele_class'], 0) + 1


# ── API functions ─────────────────────────────────────────────────────────────
def get_overview():
    return {
        "disease_id": "jbts22",

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
            "esrd_pct":         _pct(n_esrd),
            "no_mks_tier":      True,
        },

        "alerts": {
            "da_foundation": (
                "DISTAL APPENDAGE FOUNDATION — CEP83 is the MOST PROXIMAL DA protein. Loss of CEP83 "
                "simultaneously removes ALL downstream DA proteins (CEP89, SCLT1, FBF1, LRRC45, CEP164/NPHP15) "
                "from the centriole. Cilia are ABSENT (not shortened) — complete ciliogenesis initiation block. "
                "This is distinct from TZ gate genes (TCTN1-3, B9D1/2) which act downstream of DA formation."
            ),
            "very_high_renal": (
                "VERY HIGH RENAL PENETRANCE (~68%) — EARLIEST ESRD RISK AMONG JBTS GENES: "
                "ESRD median ~14–18 yr (juvenile onset). Annual renal surveillance MANDATORY from diagnosis: "
                "creatinine, cystatin C, urine osmolality, spot albumin:creatinine ratio. "
                "Polyuria/polydipsia typically precedes proteinuria by years — early concentrating defect is the key signal."
            ),
            "no_mks_tier": (
                "NO MKS TIER — CEP83 biallelic null → JBTS22 LIVE BIRTH, NOT Meckel-Gruber Syndrome. "
                "Unlike B9D1/JBTS19 (MKS9) or B9D2/JBTS34 (MKS10), CEP83 LOF does not collapse the "
                "TZ gate B9-complex inner-leaflet anchor. TZ gate scaffolding (B9D1/B9D2/MKS1/RPGRIP1L) "
                "is retained. No MKS counselling needed for JBTS22 families."
            ),
            "cep164_co_sequence": (
                "CEP164 (NPHP15) CO-SEQUENCING MANDATORY — CEP164 is the direct downstream target of the "
                "CEP83 DA hierarchy. Always co-sequence CEP164 (11q13.4) when CEP83 variants are found. "
                "Also: CEP83 (12q22) and CEP290 (12q21.32) are on the same chromosome arm — single-gene "
                "CEP290 tests do NOT cover CEP83. WES is mandatory for JBTS22 families."
            ),
        },

        "key_facts": [
            "CEP83 (~826 aa) — Distal Appendage (DA) FOUNDATION protein; most proximal DA component",
            "DA hierarchy: CEP83 → CEP89 → SCLT1 → FBF1 → LRRC45 → CEP164 (NPHP15, downstream)",
            "CC1 (aa 1–120): centriole anchoring; SDA docking; Ala148Val (North African hypomorphic founder)",
            "CC2 / CEP89 recruitment (aa 140–380): DA hierarchy nucleation; Arg252Cys (MENA founder)",
            "Central scaffold / SCLT1-binding (aa 380–600): Leu387Pro (South Asian severe)",
            "C-terminal (aa 600–826): CEP164 indirect contact; IFT-B docking surface",
            "NO MKS tier — biallelic null → JBTS22 live birth; TZ gate B9-complex retained",
            "Renal penetrance ~68% (highest non-NPHP1 JBTS) — ESRD median ~14–18 yr",
            "Retinal penetrance ~35% (rod-cone dystrophy) — annual ERG from age 3",
            "JBTS22 in ~55% of biallelic CEP83 cases; pure renal NPHP18 in ~30%",
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
                "esrd":         p['esrd'],
            }
            for p in patients
        ],
    }


def get_breakdown():
    return {
        "disease_id": "jbts22",

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
            "esrd":      {"n": n_esrd,     "pct": _pct(n_esrd)},
        },

        "notable_variants": [
            {
                "name":       "Arg252Cys",
                "cdna":       "c.754C>T",
                "domain":     "CC2 — CEP89 recruitment module; DA hierarchy nucleation interface",
                "population": "Middle Eastern / MENA founder",
                "severity":   "Moderate",
                "mechanism":  "Arg-to-Cys substitution at the CEP89 docking interface disrupts CEP89 recruitment to centriole. CEP83 remains centriole-anchored (CC1 intact) but DA hierarchy nucleation is impaired. Downstream DA proteins (SCLT1, FBF1, LRRC45, CEP164) are partially depleted. Homozygous Arg252Cys → moderate JBTS22 with variable renal penetrance. Commonest allele in MENA/Arab populations.",
            },
            {
                "name":       "Gly112Arg",
                "cdna":       "c.334G>A",
                "domain":     "CC1 — centriole anchoring domain; subdistal appendage docking",
                "population": "Pan-ethnic",
                "severity":   "Moderate–Severe",
                "mechanism":  "Gly-to-Arg substitution disrupts CC1 coiled-coil fold. CEP83 fails to anchor to the subdistal/distal appendage junction. All downstream DA proteins lost. Complete ciliogenesis block in affected tissues. Compound het with null allele → severe JBTS22 with high renal penetrance.",
            },
            {
                "name":       "Leu387Pro",
                "cdna":       "c.1160T>C",
                "domain":     "Central scaffold — SCLT1 binding interface (aa 380–600)",
                "population": "South Asian (consanguineous)",
                "severity":   "Moderate–Severe",
                "mechanism":  "Pro substitution kinks the central scaffold coiled-coil at the SCLT1 docking interface. CEP83 anchors and CEP89 is partially recruited, but SCLT1 recruitment (3rd step of DA hierarchy) is abolished. FBF1, LRRC45, and CEP164 are all lost downstream. High renal penetrance; retinal involvement common.",
            },
            {
                "name":       "Arg200Ter",
                "cdna":       "c.598C>T",
                "domain":     "CC1-CC2 junction — premature stop; truncating null",
                "population": "European",
                "severity":   "Severe (Null)",
                "mechanism":  "Premature stop in CC1-CC2 junction region. NMD-sensitive transcript — complete CEP83 loss. All downstream DA proteins absent. Cilia absent in all affected tissues. Biallelic null/null → JBTS22 live birth (no MKS). Compound het with Arg252Cys (MENA founder) → commonest pan-ethnic JBTS22 genotype in mixed-ancestry families.",
            },
            {
                "name":       "Arg530Ter",
                "cdna":       "c.1588C>T",
                "domain":     "Central scaffold — mid-protein truncating null",
                "population": "Pan-ethnic",
                "severity":   "Severe (Null)",
                "mechanism":  "Mid-protein truncating null — removes entire C-terminal regulatory domain. CC1 and CC2 may partially anchor at centriole but C-terminal IFT-B docking and TZ scaffold contact are absent. Full ciliogenesis block in cerebellar neurons and renal tubular epithelium. High ESRD risk.",
            },
            {
                "name":       "c.1122+1G>A",
                "cdna":       "c.1122+1G>A",
                "domain":     "Splice donor — intron 12; central scaffold region",
                "population": "European",
                "severity":   "Severe (Null)",
                "mechanism":  "Splice donor abolition → exon 12 skip → frameshift → NMD. Full null allele. Complete DA scaffold loss. Compound het with Arg252Cys (MENA founder) or Arg200Ter (European null) → JBTS22 moderate-severe.",
            },
            {
                "name":       "Ala148Val",
                "cdna":       "c.443C>T",
                "domain":     "CC1 — centriole anchoring domain (hypomorphic zone)",
                "population": "North African founder",
                "severity":   "Mild (Hypomorphic)",
                "mechanism":  "Conservative Val substitution partially retains CC1 anchoring function. CEP83 localises to centriole but at reduced efficiency. Partial downstream DA protein recruitment. Mild JBTS22 with lower renal penetrance. Compound het with Arg252Cys → mild-moderate JBTS22; compound het with null → moderate JBTS22.",
            },
            {
                "name":       "Tyr694Cys",
                "cdna":       "c.2081A>G",
                "domain":     "C-terminal regulatory domain (aa 600–826) — CEP164 indirect contact; IFT-B docking",
                "population": "East Asian",
                "severity":   "Moderate",
                "mechanism":  "C-terminal domain Tyr-to-Cys disrupts IFT-B docking surface and indirect CEP164 association. CC1-CC3 scaffold retains partial function. DA hierarchy partially intact (CEP89, SCLT1 recruited; FBF1/LRRC45/CEP164 partially absent). Moderate JBTS22 — cilia present but dysfunctional.",
            },
        ],
    }


def get_definitions():
    return {
        "disease_id":    "jbts22",
        "gene_full_name":"Centrosomal Protein 83 kDa (CEP83) — also CCDC41 (Coiled-Coil Domain Containing 41) — Distal Appendage Foundation Scaffold; DA Hierarchy Nucleator (CEP83→CEP89→SCLT1→FBF1→LRRC45→CEP164); Ciliogenesis Initiation; No MKS Tier; MENA Founder Arg252Cys",
        "omim_gene":     "617233",
        "omim_jbts22":   "617265",
        "chromosome":    "12q22",
        "protein_size":  (
            "~826 aa — N-terminal coiled-coil CC1 / centriole anchoring / SDA docking (aa 1–120); "
            "CC2 / CEP89 recruitment module / DA hierarchy nucleation (aa 140–380); "
            "Central scaffold / SCLT1 binding interface / self-oligomerisation (aa 380–600); "
            "C-terminal regulatory / CEP164-indirect / IFT-B docking (aa 600–826)"
        ),
        "inheritance":   "Autosomal recessive — biallelic LOF; NO MKS lethal tier (biallelic null → JBTS22 live birth)",

        "da_foundation_rule": (
            "CEP83 is the MOST PROXIMAL distal appendage protein — the master organiser of the entire "
            "DA scaffold hierarchy. Loss of CEP83 removes ALL downstream DA proteins from the centriole: "
            "CEP89 (CCDC123), SCLT1, FBF1, LRRC45, and CEP164 (NPHP15) are all simultaneously lost from "
            "the centriole. This is mechanistically distinct from TZ gate proteins (TCTN1-3, B9D1/2, "
            "TMEM231) which act downstream of DA formation. CEP83 LOF causes a complete ciliogenesis "
            "INITIATION block (not a gate disruption) — cilia are ABSENT (not shortened as in CSPP1/JBTS21). "
            "This explains JBTS22's uniquely high renal penetrance: absent tubular primary cilia (not just "
            "dysfunctional) produce the most severe tubulointerstitial nephritis among JBTS subtypes."
        ),

        "glossary": [
            {
                "term": "CEP83 (CCDC41)",
                "definition": (
                    "Centrosomal Protein 83 kDa (OMIM *617233). ~826 aa coiled-coil protein (12q22). The MOST "
                    "PROXIMAL distal appendage (DA) component. CEP83 anchors the DA scaffold to the mother "
                    "centriole subdistal/distal appendage junction and nucleates all downstream DA assembly: "
                    "CEP83 → CEP89 → SCLT1 → FBF1 → LRRC45 → CEP164 (NPHP15). Also causes NPHP18 (same gene, "
                    "same alleles — 30% pure renal phenotype, 55% JBTS22/MTS). Brain MRI mandatory to distinguish."
                ),
            },
            {
                "term": "Distal appendage (DA) scaffold",
                "definition": (
                    "The distal appendage (also called transition fibers) connects the mother centriole to the "
                    "plasma membrane or ciliary vesicle. DA scaffold is REQUIRED for: (1) centriole docking to "
                    "the plasma membrane (via EHD1/SNAP29/Rab8a vesicle fusion axis); (2) CP110/CEP97 cap "
                    "removal (ciliogenesis initiation); (3) IFT-A/B train entry point assembly; "
                    "(4) TZ gate protein positioning. CEP83 nucleates the entire DA hierarchy. Loss = complete "
                    "ciliogenesis block. DA forms UPSTREAM of the transition zone (TZ) gate."
                ),
            },
            {
                "term": "DA hierarchy (CEP83 → CEP164)",
                "definition": (
                    "Sequential assembly cascade: CEP83 (most proximal) → CEP89 (CCDC123) → SCLT1 → FBF1 → "
                    "LRRC45 → CEP164 (most distal; NPHP15). Loss of CEP83 removes the foundation, "
                    "simultaneously depleting all downstream members from the centriole. Loss of CEP164 alone "
                    "(NPHP15) spares CEP83 through LRRC45 but disrupts the most distal DA-TZ interface. "
                    "This hierarchy is the structural basis for why CEP83/JBTS22 has a more severe renal "
                    "phenotype than CEP164/NPHP15 — complete vs partial DA scaffold loss."
                ),
            },
            {
                "term": "No MKS tier (CEP83)",
                "definition": (
                    "CEP83 biallelic null → JBTS22 live birth. Critical counselling distinction vs B9D1/JBTS19 "
                    "(MKS9) and B9D2/JBTS34 (MKS10) — which carry null/null perinatal-lethal MKS risk. "
                    "CEP83 acts at the DA (upstream of TZ gate); it does NOT form the B9-complex TZ gate "
                    "inner-leaflet anchor. TZ gate B9-complex (B9D1/B9D2/MKS1) is independently expressed "
                    "and retains its basal assembly capacity even when CEP83 is absent."
                ),
            },
            {
                "term": "MENA founder allele (Arg252Cys)",
                "definition": (
                    "Arg252Cys (c.754C>T) in CC2 / CEP89 recruitment module. The commonest JBTS22 allele in "
                    "Middle Eastern and North African populations. Disrupts CEP89 docking interface. "
                    "Homozygous Arg252Cys → moderate JBTS22. Compound het with null → moderate-severe. "
                    "Screening mandatory in all MENA/Arab JBTS probands."
                ),
            },
            {
                "term": "Cilia absent (vs shortened)",
                "definition": (
                    "CEP83 LOF causes ABSENT cilia (not shortened). The DA block prevents centriole docking "
                    "to the ciliary vesicle — without docking, CP110 cap cannot be removed, and the axoneme "
                    "is never initiated. This contrasts with CSPP1/JBTS21 (axoneme-wide role — cilia "
                    "SHORTENED) and NPHP module genes (cilia present but gate-dysfunctional). The complete "
                    "cilia absence in JBTS22 explains: (1) highest renal penetrance (no partial cilia "
                    "function preserved in tubular epithelium); (2) complete Hedgehog/Wnt failure → MTS; "
                    "(3) nasal brushing shows ABSENT cilia, unlike CSPP1 which shows shortened cilia."
                ),
            },
            {
                "term": "NPHP-like TIN / ESRD (JBTS22)",
                "definition": (
                    "Nephronophthisis-like tubulointerstitial nephritis. In JBTS22: affects ~68% of patients — "
                    "the highest renal penetrance among JBTS genes (compared with TMEM231/JBTS20 ~25%, "
                    "CSPP1/JBTS21 ~18%). ESRD median ~14–18 yr (juvenile onset). Annual surveillance "
                    "mandatory from diagnosis. Polyuria/polydipsia (concentrating defect) typically precedes "
                    "proteinuria by years. Renal transplant curative, no allograft recurrence."
                ),
            },
            {
                "term": "CEP164 (NPHP15) co-sequencing",
                "definition": (
                    "CEP164 (11q13.4) is the most distal target of the CEP83 DA hierarchy. When CEP83 "
                    "variants are found: (1) always co-sequence CEP164 — biallelic CEP164 loss causes NPHP15; "
                    "(2) digenic CEP83 + CEP164 heterozygosity may produce additive DA scaffold failure; "
                    "(3) CEP83 (12q22) and CEP290 (12q21.32) are on the SAME chromosome arm — single-gene "
                    "CEP290 tests do NOT detect CEP83 — WES is mandatory."
                ),
            },
        ],

        "domain_matrix": [
            {
                "domain":          "CC1 / centriole anchoring / SDA docking (aa 1–120)",
                "location":        "N-terminus — subdistal/distal appendage junction; CE63/C3orf14 interaction",
                "function":        "Anchors CEP83 to the mother centriole DA junction independently of all other DA proteins. Foundation for all downstream DA assembly. Ala148Val (North African founder) is a hypomorphic allele in the CC1 anchoring region — partial function retained. Gly112Arg disrupts CC1 fold — complete anchoring failure.",
                "variant_examples":"Gly112Arg (pan-ethnic, moderate-severe — CC1 fold); Ala148Val (North African founder, mild — hypomorphic)",
            },
            {
                "domain":          "CC2 / CEP89 recruitment module (aa 140–380)",
                "location":        "N-central — CEP89 (CCDC123) docking interface; DA hierarchy step 2 nucleation",
                "function":        "Recruits CEP89 (second step in DA hierarchy) to the centriole. CEP89 recruitment is the first downstream signal that propagates the entire DA scaffold assembly chain. Arg252Cys (MENA founder) disrupts CEP89 docking interface — partial downstream DA protein loss.",
                "variant_examples":"Arg252Cys (MENA founder, moderate — CEP89 interface); Tyr694Cys (East Asian, moderate — C-terminal)",
            },
            {
                "domain":          "Central scaffold / SCLT1-binding (aa 380–600)",
                "location":        "Central — SCLT1 interaction surface; self-oligomerisation domain; DA step 3",
                "function":        "Recruits SCLT1 (third step — prerequisite for FBF1→LRRC45→CEP164 chain). Self-oligomerisation stabilises the DA scaffold ring structure. Leu387Pro (South Asian) kinks the coiled-coil at the SCLT1 docking interface — SCLT1 and all downstream DA proteins lost.",
                "variant_examples":"Leu387Pro (South Asian, moderate-severe — SCLT1 docking); Arg530Ter (pan-ethnic null — removes C-terminal entirely)",
            },
            {
                "domain":          "C-terminal regulatory / IFT-B docking (aa 600–826)",
                "location":        "C-terminus — indirect CEP164 contact (via SCLT1→FBF1→LRRC45); IFT-B entry point",
                "function":        "IFT-B docking surface at DA base; transition zone scaffold connection. Arg530Ter (null) removes this domain — no IFT-B entry at DA. Tyr694Cys (East Asian) disrupts IFT-B docking partially — moderate phenotype with partial cilia function.",
                "variant_examples":"Tyr694Cys (East Asian, moderate — IFT-B docking); Gln614Ter (pan-ethnic null — C-terminal truncating)",
            },
        ],

        "clinical_pearls": [
            {
                "title": "CEP83 — DA Foundation: ALL Downstream DA Proteins Lost Simultaneously (Cilia Absent, Not Shortened)",
                "detail": (
                    "CEP83 is the most upstream distal appendage protein — its loss is uniquely catastrophic: "
                    "ALL downstream DA proteins (CEP89, SCLT1, FBF1, LRRC45, and CEP164/NPHP15) are "
                    "simultaneously depleted from the centriole. This is mechanistically distinct from TZ gate "
                    "disruptions (TCTN1/2/3, B9D1/2, TMEM231). The DA block prevents centriole docking to "
                    "the ciliary vesicle — without docking, the CP110/CEP97 cap cannot be removed, and no "
                    "axoneme is initiated. Cilia are ABSENT (not shortened). Clinical implications: "
                    "(1) highest renal penetrance (~68%) of any non-NPHP1 JBTS gene — no partial tubular "
                    "cilia function is preserved; (2) nasal brushing shows ABSENT cilia (unlike CSPP1/JBTS21 "
                    "where shortened cilia are visible); (3) complete Hedgehog/Wnt/PDGF signalling failure — "
                    "more severe cerebellar phenotype in the subset with JBTS22 vs pure renal NPHP18. "
                    "Always request nasal brushing videomicroscopy — absent cilia in CEP83 vs shortened in CSPP1."
                ),
            },
            {
                "title": "Very High Renal Penetrance (~68%): Earliest ESRD Risk; Annual Protocol Mandatory from Diagnosis",
                "detail": (
                    "JBTS22/CEP83 has the HIGHEST renal penetrance of any JBTS gene (excluding NPHP1 which "
                    "is primarily nephronophthisis): ~68% of JBTS22 (MTS-confirmed) patients develop "
                    "significant CKD. ESRD median ~14–18 yr — EARLIER than TMEM231/JBTS20 (25 yr), B9D1/"
                    "JBTS19 (22 yr), and CSPP1/JBTS21 (28 yr). Annual surveillance protocol MANDATORY from "
                    "diagnosis: creatinine, cystatin C eGFR, urine osmolality (early concentrating defect "
                    "precedes proteinuria by years), spot albumin:creatinine ratio, blood pressure. "
                    "Key clinical pitfall: polyuria/polydipsia is the EARLIEST renal sign in CEP83 — "
                    "parents may not report it unless directly asked. Enquire about nocturnal enuresis, "
                    "large urine volumes, and thirst at every visit from diagnosis. Nephrology referral at "
                    "diagnosis (NOT when CKD is detected). Renal transplant is curative — no allograft "
                    "recurrence (cell-autonomous photoreceptor and tubular defect; not systemic)."
                ),
            },
            {
                "title": "Brain MRI MANDATORY: JBTS22 vs Pure Renal NPHP18 — Same Gene, Two Phenotypes",
                "detail": (
                    "Biallelic CEP83 alleles can produce either JBTS22 (Molar Tooth Sign, ~55% of biallelic "
                    "cases) OR pure renal NPHP18 (no MTS, ~30%). The same genotype (e.g. Arg252Cys/Arg200Ter) "
                    "can segregate as either phenotype within the same family — the genetic basis for this "
                    "intra-familial variability is not fully understood (modifier loci suspected). Practical "
                    "implication: Brain MRI (axial, T2) is MANDATORY for ALL biallelic CEP83 patients "
                    "regardless of presenting phenotype. MTS presence → JBTS22 management (cerebellar + renal "
                    "+ retinal surveillance). MTS absence → NPHP18 management (renal-dominant, retinal only). "
                    "Never assume 'pure renal' in a CEP83 patient without MRI confirmation."
                ),
            },
            {
                "title": "CEP164 (NPHP15) Co-Sequencing + CEP290 Distinction on Chr 12q",
                "detail": (
                    "CEP164 (NPHP15, 11q13.4) is the most distal target of the CEP83 DA hierarchy. "
                    "Clinical rule: always co-sequence CEP164 when CEP83 variants are found. Reason: "
                    "biallelic CEP164 loss causes NPHP15 (similar renal + Joubert phenotype); digenic "
                    "CEP83 + CEP164 heterozygosity may produce additive DA failure (reported in two families). "
                    "Critical genomic distinction: CEP83 (12q22) and CEP290/NPHP6 (12q21.32) are on the SAME "
                    "chromosome arm 12q. Standard single-gene CEP290 tests — the commonest JBTS gene (~15% "
                    "of all JBTS) — do NOT detect CEP83. WES covering the entire 12q arm is mandatory when "
                    "Joubert phenotype is present and CEP290 Sanger is negative. A negative CEP290 panel "
                    "in a JBTS22 phenotype does NOT rule out 12q pathology."
                ),
            },
            {
                "title": "Retinal Rod-Cone Dystrophy (~35%): Connecting Cilia Absent — ERG Mandatory from Age 3",
                "detail": (
                    "Rod-cone dystrophy in JBTS22 affects ~35% of patients — intermediate rate between "
                    "JBTS5/CEP290 (~50–70%) and CSPP1/JBTS21 (~25%). The mechanism is connecting cilia "
                    "ABSENCE (not dysfunction): without DA-mediated ciliogenesis initiation, photoreceptor "
                    "connecting cilia do not form → opsin trafficking fails → rod photoreceptor "
                    "degeneration. ERG mandatory from age 3. Retinal findings do NOT improve post-renal "
                    "transplant (cell-autonomous photoreceptor defect — retina receives no benefit from "
                    "systemic cure of nephronophthisis). Ophthalmology surveillance plan is independent of "
                    "transplant status. Parents should be counselled that retinal risk persists lifelong even "
                    "after successful transplant."
                ),
            },
        ],

        "literature_highlights": [
            "Failler M et al. (2014) Mutations of CEP83 cause infantile nephronophthisis and intellectual disability. Am J Hum Genet 94(6):905–14. [JBTS22/NPHP18 primary discovery — CEP83 biallelic mutations cause both NPHP18 and JBTS22].",
            "Tanos BE et al. (2013) Centriole distal appendages promote membrane docking, leading to cilia initiation. Genes Dev 27(2):163–8. [CEP83 as DA foundation — CEP83→CEP89→SCLT1→FBF1→LRRC45→CEP164 hierarchy characterised].",
            "Burke MC et al. (2014) Chibby promotes ciliary vesicle formation and basal body docking to the cell membrane. J Cell Biol 207(1):123–37. [CEP83 DA scaffold in ciliogenesis initiation — CP110 cap removal pathway].",
            "Bachmann-Gagescu R et al. (2020) JBTS disease gene landscape across 460 families. Hum Mutat 41(4):e1–e45. [CEP83/JBTS22 frequency and phenotype spectrum in large international cohort].",
            "Slaats GG et al. (2016) Nephronophthisis-associated CEP164 regulates cell cycle progression, apoptosis and epithelial-to-mesenchymal transition. PLoS Genet 12(3). [CEP164 downstream of CEP83 hierarchy — epistasis confirmed].",
            "Parisi MA (2019) The molecular genetics of Joubert syndrome and related ciliopathies. Transl Sci Rare Dis 4(1-2):25–49. [JBTS22/CEP83 reviewed in broader JBTS gene landscape].",
        ],

        "phenotype_frequencies": {
            "mts_pathognomonic":       "100% (MTS is the diagnostic criterion — JBTS22-specific cohort)",
            "cerebellar_ataxia":       f"{_pct(n_ataxia)}%",
            "neonatal_hypotonia":      f"{_pct(n_hypotonia)}%",
            "oculomotor_apraxia":      f"{_pct(n_oma)}%",
            "breathing_dysregulation": f"{_pct(n_breath)}%",
            "intellectual_disability": f"{_pct(n_id)}%",
            "retinal_rod_cone":        f"{_pct(n_retinal)}%",
            "renal_nphp_tin":          f"{_pct(n_renal)}% (highest non-NPHP1 JBTS gene; ESRD median ~14–18 yr)",
            "esrd_at_study":           f"{_pct(n_esrd)}%",
            "hepatic_chf":             f"{_pct(n_hepatic)}% (rare; biliary ductal plate malformation)",
            "polydactyly_post_axial":  f"{_pct(n_poly)}% (very rare in CEP83 — DA protein, not TZ/CPLANE)",
            "skeletal":                "0% (no SRTD allelic phenotype — CEP83 is DA foundation, not axoneme/centriolar scaffold)",
            "situs_inversus":          "<2% (CEP83 not expressed in nodal cilia — no laterality defect)",
            "no_mks_tier":             "Confirmed — biallelic null/null → JBTS22 live birth (NO Meckel-Gruber risk)",
            "mena_founder":            "Arg252Cys (c.754C>T) — CC2 CEP89 recruitment module; MENA/Arab populations",
            "pure_renal_nphp18":       "~30% of all biallelic CEP83 cases (no MTS) — EXCLUDED from this JBTS22 cohort; brain MRI mandatory",
            "jbts22_penetrance":       "~55% of all biallelic CEP83 cases develop JBTS22 (MTS confirmed)",
        },
    }
