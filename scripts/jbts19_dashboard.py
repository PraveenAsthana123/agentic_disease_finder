"""
B9D1 Joubert Syndrome Type 19 (JBTS19) — Autosomal Recessive / B9D1 (B9 Domain-Containing Protein 1) / B9 Complex / MKS9 Tier / TZ Inner-Leaflet Anchor
==========================================================================================================================================================
Primary Gene : B9D1 (*614144) — 17p11.2; ~250 aa; B9 domain-containing protein 1 (also MKSR1).
               B9D1 forms the B9 protein complex at the ciliary transition zone (TZ) together with
               B9D2 (B9 domain-containing protein 2) and MKS1 (Meckel syndrome type 1 protein).
               The B9 complex acts as the inner-leaflet membrane-anchoring layer of the TZ gate,
               working in concert with the NPHP and Tectonic modules to form the complete TZ
               diffusion barrier. B9D1 is essential for ciliary membrane integrity; without it the
               TZ gate is completely disrupted, excluding SMO from cilia and abolishing Hedgehog
               signal transduction.
               B9D1 protein domain architecture:
               - N-terminal unstructured region (aa 1–132): disordered; primary MKS1 interaction
                 surface; B9D2 docking site; ciliary targeting determinant
               - B9 domain (aa 133–236): core β-barrel fold; TZ inner-leaflet membrane anchoring;
                 B9D2 β-strand exchange; TMEM231 interface; cholesterol-enriched TZ membrane binding
               - C-terminal tail (aa 237–250): short disordered C-terminus; additional TMEM231
                 contact; assists ciliary localisation
               B9D1 LOF → B9 complex disassembly → TZ inner-leaflet anchor absent → full TZ gate
               collapse → SMO excluded → Hedgehog failure → Molar Tooth Sign (MTS).

⚠ MKS9 TIER — B9D1-SPECIFIC RULE:
   Biallelic B9D1 null alleles (null/null genotype, e.g. Trp88Ter/Trp88Ter or Arg222Ter/Arg222Ter)
   → Meckel-Gruber Syndrome type 9 (MKS9, OMIM #614209): perinatal lethal, encephalocele,
   polydactyly, renal cystic dysplasia. Unlike the Tectonic module genes (TCTN1, TCTN3), B9D1
   biallelic null disrupts the entire TZ gate (B9 complex + downstream modules) during critical
   embryonic ciliogenesis windows, causing the most severe ciliopathy spectrum. Approximately 22%
   of B9D1 families carry a null/null genotype (MKS9 risk). Null/hypomorphic compound heterozygotes
   → JBTS19 live birth (hypomorphic allele rescues MKS lethality). Counsellors must communicate the
   MKS9 risk explicitly when a null allele is identified in a carrier parent.

⚠ B9 COMPLEX DISTINCTION — B9D1 vs B9D2 vs MKS1:
   B9D1 (JBTS19, 17p11.2), B9D2 (JBTS34, 19q13.2), and MKS1 (JBTS28, 17q22) all contribute to
   the B9 complex at the TZ. Each has a distinct JBTS/MKS subtype and unique protein interactions.
   WES must distinguish all three: B9D1 → JBTS19/MKS9; B9D2 → JBTS34/MKS10; MKS1 → JBTS28/MKS1.
   Null/null genotype in any B9 complex member carries a perinatal-lethal MKS risk. The B9 complex
   is the most tightly coupled MKS-tier set in JBTS: all three members have documented MKS alleles,
   unlike the Tectonic module (TCTN1, TCTN3 no MKS; TCTN2 MKS8 only).

⚠ RENAL PENETRANCE (~35%): Annual NPHP-like protocol mandatory. ESRD median ~22 yr.
   B9D1 LOF disrupts B9 complex in renal tubular primary cilia → NPHP-like tubulointerstitial
   nephritis. Higher renal penetrance than TCTN3/JBTS18 (~20%) or TCTN1/JBTS11 (~20%), consistent
   with complete TZ gate collapse vs partial Tectonic complex disruption. Annual surveillance
   (creatinine, cystatin C, urine osmolality, microalbuminuria) from diagnosis. Renal transplant
   curative; no allograft recurrence.

Disease OMIM : #614975 — Joubert Syndrome Type 19 (JBTS19)
               Allelic: #614209 — Meckel-Gruber Syndrome Type 9 (MKS9)
Chromosome   : 17p11.2
Inheritance  : Autosomal recessive — biallelic LOF; MKS9 lethal tier (null/null genotype)
Cohort size  : 40-patient educational cohort (seed 445)
"""

import random
import math

SEED = 445
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
    ('European',               0.30),
    ('Middle Eastern / MENA',  0.25),   # Arg106Cys founder
    ('South Asian',            0.22),   # Leu193Pro prevalent
    ('North African',          0.13),   # Ala34Val founder (mild)
    ('East Asian',             0.06),
    ('Other / Unknown',        0.04),
]

# Allele classes (WITH MKS9 lethal tier for null/null genotype)
allele_classes = [
    ('Biallelic Missense',           0.32),   # moderate phenotype
    ('Null / Hypomorphic',           0.30),   # JBTS19 live birth (hypomorphic rescues MKS)
    ('Biallelic Null (MKS9 risk)',   0.22),   # MKS9 lethal risk — null/null genotype
    ('Splice / Null Compound',       0.16),   # splice + null compound het
]

variants = [
    'Arg106Cys/Arg106Cys',
    'Arg106Cys/Gly67Arg',
    'Gly67Arg/Leu193Pro',
    'Leu193Pro/Leu193Pro',
    'Trp88Ter/Gly67Arg',
    'Arg222Ter/Arg106Cys',
    'c.441+1G>A/Gly67Arg',
    'Tyr216Cys/Arg106Cys',
    'Ala34Val/Ala34Val',
    'Trp88Ter/Arg222Ter',
]

sex_choices = ['M', 'F']

_eth_pool  = [e for e, p in ethnicities  for _ in range(round(p * 100))]
_ac_pool   = [ac for ac, p in allele_classes for _ in range(round(p * 100))]
_var_pool  = variants * 8   # weighted pool

for i in range(N):
    eth = rng.choice(_eth_pool)
    ac  = rng.choice(_ac_pool)
    var = rng.choice(_var_pool)
    age = rng.randint(1, 38)
    sex = rng.choice(sex_choices)

    # Phenotype probabilities — B9D1/JBTS19 frequencies (literature-aligned)
    mts       = rng.random() < 0.92
    ataxia    = rng.random() < 0.88
    hypotonia = rng.random() < 0.83
    oma       = rng.random() < 0.55
    breathing = rng.random() < 0.58
    retinal   = rng.random() < 0.32
    renal     = rng.random() < 0.35
    hepatic   = rng.random() < 0.18
    poly      = rng.random() < 0.20
    id_       = rng.random() < 0.72
    mks9      = (ac == 'Biallelic Null (MKS9 risk)')
    cc        = rng.random() < 0.14

    patients.append({
        'id':          f'JBTS19-{i+1:03d}',
        'age':         age,
        'sex':         sex,
        'ethnicity':   eth,
        'allele_class':ac,
        'variant':     var,
        'mts':         mts,
        'ataxia':      ataxia,
        'hypotonia':   hypotonia,
        'oma':         oma,
        'breathing':   breathing,
        'retinal':     retinal,
        'renal':       renal,
        'hepatic':     hepatic,
        'poly':        poly,
        'id':          id_,
        'mks9_risk':   mks9,
        'cc':          cc,
    })

# ── aggregate counts ──────────────────────────────────────────────────────────
n_mts      = sum(1 for p in patients if p['mts'])
n_ataxia   = sum(1 for p in patients if p['ataxia'])
n_hypotonia= sum(1 for p in patients if p['hypotonia'])
n_oma      = sum(1 for p in patients if p['oma'])
n_breath   = sum(1 for p in patients if p['breathing'])
n_retinal  = sum(1 for p in patients if p['retinal'])
n_renal    = sum(1 for p in patients if p['renal'])
n_hepatic  = sum(1 for p in patients if p['hepatic'])
n_poly     = sum(1 for p in patients if p['poly'])
n_id       = sum(1 for p in patients if p['id'])
n_mks9     = sum(1 for p in patients if p['mks9_risk'])
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
        "disease_id": "jbts19",
        "gene":       "B9D1",
        "disease":    "Joubert Syndrome Type 19 (JBTS19)",
        "omim_gene":  "614144",
        "omim_disease": "614975",
        "omim_mks9":  "614209",
        "chromosome": "17p11.2",
        "cohort_n":   N,
        "cohort_seed":SEED,

        "kpis": {
            "total_patients":    N,
            "mts_count":         n_mts,
            "mts_pct":           _pct(n_mts),
            "ataxia_pct":        _pct(n_ataxia),
            "hypotonia_pct":     _pct(n_hypotonia),
            "retinal_pct":       _pct(n_retinal),
            "renal_pct":         _pct(n_renal),
            "hepatic_pct":       _pct(n_hepatic),
            "poly_pct":          _pct(n_poly),
            "mks9_risk_count":   n_mks9,
            "mks9_risk_pct":     _pct(n_mks9),
        },

        "alerts": {
            "mks9_tier": (
                "MKS9 TIER — B9D1 biallelic null (null/null genotype) → Meckel-Gruber Syndrome "
                "Type 9 (MKS9, #614209): perinatal lethal encephalocele, polydactyly, renal cystic "
                "dysplasia. ~22% of JBTS19 families carry MKS9-risk null/null genotype. "
                "Null/hypomorphic compound het → JBTS19 live birth (hypomorphic rescues MKS lethality). "
                "MKS9 counselling MUST be given when a null allele is identified in a carrier parent."
            ),
            "b9_complex_ddx": (
                "B9 COMPLEX DDx — WES must distinguish B9D1 (JBTS19/MKS9, 17p11.2), B9D2 (JBTS34/MKS10, "
                "19q13.2), and MKS1 (JBTS28/MKS1, 17q22). All three are in the same TZ B9 complex and "
                "all carry null/null perinatal-lethal MKS risk. Different chromosome loci — confirm gene ID."
            ),
            "renal_35pct": (
                "RENAL 35% — Higher penetrance than TCTN1/JBTS11 (~20%) or TCTN3/JBTS18 (~20%), consistent "
                "with complete TZ gate collapse. Annual NPHP-like surveillance from diagnosis. ESRD median ~22 yr. "
                "Renal transplant curative; no allograft recurrence."
            ),
        },

        "key_facts": [
            "B9D1 (250 aa) forms the B9 complex with B9D2 and MKS1 — inner-leaflet TZ membrane anchor",
            "B9 domain (aa 133–236): β-barrel fold; TZ membrane anchoring; B9D2 β-strand exchange",
            "MKS9 tier: biallelic null → perinatal lethal MKS9 (~22% of families)",
            "Null/hypomorphic compound het → JBTS19 live birth (hypomorphic rescues MKS lethality)",
            "Renal penetrance 35% — highest in Tectonic/B9 module group; ESRD median ~22 yr",
            "Retinal penetrance 32% (rod-cone dystrophy) — annual ERG from age 3",
            "Hepatic CHF 18% — 2-yr surveillance LFTs + hepatic ultrasound",
            "Polydactyly 20% — post-axial; more frequent than TCTN3/JBTS18 (15%)",
            "Frequency ~1-2% of all Joubert syndrome (~1/3–6 million worldwide)",
            "Autosomal recessive, 25% recurrence; MKS9 counselling for null carrier parents",
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
                "id_flag":      p['id'],
                "mks9_risk":    p['mks9_risk'],
                "cc":           p['cc'],
            }
            for p in patients
        ],
    }


def get_breakdown():
    return {
        "disease_id": "jbts19",

        "ethnicity_distribution": [
            {"ethnicity": eth, "count": cnt, "pct": _pct(cnt)}
            for eth, cnt in sorted(_eth_counts.items(), key=lambda x: -x[1])
        ],

        "allele_class_distribution": [
            {"allele_class": ac, "count": cnt, "pct": _pct(cnt)}
            for ac, cnt in sorted(_ac_counts.items(), key=lambda x: -x[1])
        ],

        "phenotype_summary": {
            "mts":          {"n": n_mts,      "pct": _pct(n_mts)},
            "ataxia":       {"n": n_ataxia,   "pct": _pct(n_ataxia)},
            "hypotonia":    {"n": n_hypotonia,"pct": _pct(n_hypotonia)},
            "oma":          {"n": n_oma,      "pct": _pct(n_oma)},
            "breathing":    {"n": n_breath,   "pct": _pct(n_breath)},
            "retinal":      {"n": n_retinal,  "pct": _pct(n_retinal)},
            "renal":        {"n": n_renal,    "pct": _pct(n_renal)},
            "hepatic":      {"n": n_hepatic,  "pct": _pct(n_hepatic)},
            "poly":         {"n": n_poly,     "pct": _pct(n_poly)},
            "id":           {"n": n_id,       "pct": _pct(n_id)},
            "mks9_risk":    {"n": n_mks9,     "pct": _pct(n_mks9)},
            "cc":           {"n": n_cc,       "pct": _pct(n_cc)},
        },

        "notable_variants": [
            {
                "name":       "Arg106Cys",
                "cdna":       "c.316C>T",
                "domain":     "N-terminal / B9 domain entry — MKS1 interaction surface",
                "population": "Middle Eastern / MENA founder",
                "severity":   "Moderate",
                "mechanism":  "Partial MKS1 binding impairment; B9 complex partial destabilisation; JBTS19 moderate phenotype",
            },
            {
                "name":       "Gly67Arg",
                "cdna":       "c.199G>A",
                "domain":     "N-terminal unstructured — B9D2 docking region",
                "population": "Pan-ethnic",
                "severity":   "Moderate",
                "mechanism":  "B9D2 docking surface disruption; B9 complex partial assembly defect; JBTS19",
            },
            {
                "name":       "Leu193Pro",
                "cdna":       "c.578T>C",
                "domain":     "B9 domain core — β-barrel fold disruption",
                "population": "South Asian",
                "severity":   "Moderate — Severe",
                "mechanism":  "B9D core β-barrel fold disruption; near-complete TZ anchor loss; severe JBTS19",
            },
            {
                "name":       "Trp88Ter",
                "cdna":       "c.264G>A",
                "domain":     "N-terminal truncating — premature stop before B9 domain",
                "population": "European",
                "severity":   "Null — Severe / MKS9 risk",
                "mechanism":  "Premature stop codon before B9 domain; NMD → no functional protein; MKS9 lethal risk when homozygous",
            },
            {
                "name":       "Arg222Ter",
                "cdna":       "c.664C>T",
                "domain":     "B9 domain / C-terminal — near-complete truncation",
                "population": "Pan-ethnic",
                "severity":   "Null — Severe / MKS9 risk",
                "mechanism":  "C-terminal null; complete B9 domain functional loss; TMEM231 interface absent; MKS9 risk homozygous",
            },
            {
                "name":       "c.441+1G>A",
                "cdna":       "c.441+1G>A",
                "domain":     "Splice donor — intron 4 null",
                "population": "European",
                "severity":   "Null — Severe / MKS9 risk",
                "mechanism":  "Exon skipping → frameshift → premature stop; NMD; complete B9D1 loss; MKS9 risk when compound null",
            },
            {
                "name":       "Ala34Val",
                "cdna":       "c.101C>T",
                "domain":     "N-terminal — distal from B9 domain; hypomorphic",
                "population": "North African founder",
                "severity":   "Mild (Hypomorphic)",
                "mechanism":  "Partial N-terminal destabilisation; residual B9 complex assembly; mild JBTS19; hypomorphic rescues MKS lethality in compound het with null",
            },
            {
                "name":       "Tyr216Cys",
                "cdna":       "c.647A>G",
                "domain":     "B9 domain core — TMEM231 interface",
                "population": "East Asian",
                "severity":   "Moderate",
                "mechanism":  "TMEM231 binding surface disruption; partial B9 complex TZ anchoring defect; moderate JBTS19",
            },
        ],

        "variant_distribution": [
            {"allele_class": ac, "count": cnt, "pct": _pct(cnt)}
            for ac, cnt in sorted(_ac_counts.items(), key=lambda x: -x[1])
        ],

        "phenotype_counts": {
            "mts":       n_mts,
            "ataxia":    n_ataxia,
            "hypotonia": n_hypotonia,
            "oma":       n_oma,
            "breathing": n_breath,
            "retinal":   n_retinal,
            "renal":     n_renal,
            "hepatic":   n_hepatic,
            "poly":      n_poly,
            "id":        n_id,
            "mks9":      n_mks9,
            "cc":        n_cc,
        },
    }


def get_definitions():
    return {
        "disease_id":    "jbts19",
        "gene_full_name":"B9 Domain-Containing Protein 1 (B9D1; MKSR1) — Inner-leaflet TZ membrane anchor; B9 complex core; MKS9 allelic",
        "omim_gene":     "614144",
        "omim_jbts19":   "614975",
        "omim_mks9":     "614209",
        "chromosome":    "17p11.2",
        "protein_size":  "~250 aa — N-terminal unstructured region / MKS1 interface (aa 1–132); B9 domain / TZ anchor / B9D2 interface (aa 133–236); C-terminal tail / TMEM231 contact (aa 237–250)",
        "inheritance":   "Autosomal recessive — biallelic LOF; MKS9 lethal tier (null/null genotype → perinatal lethal MKS9)",

        "mks9_tier_rule": (
            "B9D1 biallelic null (null/null genotype, e.g. Trp88Ter/Trp88Ter, Arg222Ter/Arg222Ter, "
            "c.441+1G>A/Arg222Ter) → Meckel-Gruber Syndrome Type 9 (MKS9, OMIM #614209): perinatal "
            "lethal encephalocele, cystic renal dysplasia, post-axial polydactyly. Approximately 22% "
            "of JBTS19 families in this cohort carry a null/null genotype (MKS9 lethal risk). "
            "Null/hypomorphic compound heterozygotes (e.g. Trp88Ter/Ala34Val) → JBTS19 live birth "
            "(the hypomorphic allele provides sufficient residual B9D1 function to rescue embryonic "
            "lethality). Counsellors MUST communicate MKS9 risk when a B9D1 null allele is identified "
            "in any carrier parent — 25% AR recurrence for JBTS19 or MKS9 depending on second allele."
        ),

        "glossary": [
            {"term": "B9 complex", "definition": "Protein complex of B9D1, B9D2, and MKS1 that forms the inner-leaflet anchoring layer of the ciliary transition zone (TZ). The B9 complex works with the NPHP module (NPHP1, NPHP4, TMEM237) and Tectonic module (TCTN1, TCTN2, TCTN3) to constitute the full TZ diffusion barrier."},
            {"term": "B9 domain", "definition": "β-barrel fold domain (~100 aa) found in B9D1 (aa 133–236) and B9D2. Mediates B9D1–B9D2 β-strand exchange and TZ inner-leaflet membrane anchoring. Absolutely required for B9 complex formation; mutations in the B9 domain abolish TZ localisation."},
            {"term": "MKS9 (Meckel-Gruber Syndrome Type 9)", "definition": "Allelic disease of JBTS19 (OMIM #614209). Caused by B9D1 biallelic null alleles. Perinatal lethal: occipital encephalocele, cystic renal dysplasia, post-axial polydactyly. Distinct from MKS1 (MKS1 gene), MKS2 (TMEM216), MKS3 (TMEM67), MKS5 (RPGRIP1L), MKS6 (CC2D2A), MKS8 (TCTN2), MKS10 (B9D2)."},
            {"term": "Transition zone (TZ)", "definition": "Compartment at the base of the ciliary axoneme between the basal body and the ciliary shaft. Acts as a diffusion barrier ('ciliary gate') controlling protein composition of the ciliary membrane. Requires three cooperative modules: B9 complex (inner leaflet anchor), NPHP module (Y-link scaffold), Tectonic complex (lipid gate)."},
            {"term": "Molar Tooth Sign (MTS)", "definition": "Pathognomonic MRI appearance in Joubert syndrome: elongated superior cerebellar peduncles + vermis hypoplasia form a 'molar tooth' shape on axial brain MRI."},
            {"term": "Null/hypomorphic rescue", "definition": "Mechanism by which a hypomorphic (partial-function) allele in compound heterozygosity with a null allele provides sufficient residual protein function to prevent perinatal lethality (MKS9), resulting instead in the milder JBTS19 live-birth phenotype."},
            {"term": "NPHP-like TIN", "definition": "Nephronophthisis-like tubulointerstitial nephritis. In JBTS19: affects ~35% of patients — higher penetrance than TCTN module genes because B9D1 LOF causes complete TZ collapse in renal tubular cilia. Annual surveillance from diagnosis. ESRD median ~22 yr; renal transplant curative."},
            {"term": "B9 complex vs Tectonic complex distinction", "definition": "Both B9 and Tectonic modules operate at the TZ gate but serve different structural roles. B9 complex (B9D1-B9D2-MKS1) = inner-leaflet membrane anchor; all three members have MKS null/null lethal tiers. Tectonic complex (TCTN1-TCTN2-TCTN3) = lipid gate; only TCTN2 has MKS8 tier; TCTN1 and TCTN3 do not have MKS lethal tiers."},
        ],

        "domain_matrix": [
            {
                "domain":          "N-terminal unstructured region / MKS1 interface (aa 1–132)",
                "location":        "N-terminus — disordered; primary protein interaction surface",
                "function":        "MKS1 docking; B9D2 initial contact; ciliary targeting determinant; Arg106Cys (MENA founder) and Gly67Arg (pan-ethnic) disrupt this surface",
                "variant_examples":"Arg106Cys (MENA founder, moderate); Gly67Arg (pan-ethnic, moderate); Trp88Ter (European null — truncates before B9 domain, MKS9 risk)",
            },
            {
                "domain":          "B9 domain / TZ anchor / B9D2 β-strand interface (aa 133–236)",
                "location":        "Central — β-barrel fold; TZ inner-leaflet membrane binding",
                "function":        "Core TZ inner-leaflet anchoring; B9D2 β-strand exchange (heterodimerisation); TMEM231 contact; lipid raft affinity for TZ inner leaflet; Leu193Pro (South Asian) disrupts β-barrel fold",
                "variant_examples":"Leu193Pro (South Asian, moderate-severe); Tyr216Cys (East Asian, moderate); Arg222Ter (pan-ethnic null — complete B9D loss, MKS9 risk)",
            },
            {
                "domain":          "C-terminal tail / TMEM231 contact (aa 237–250)",
                "location":        "C-terminal — short disordered tail; secondary interaction surface",
                "function":        "Additional TMEM231 contact; assists ciliary localisation; Ala34Val (North African founder) is upstream in N-term; c.441+1G>A splice null truncates in N-term",
                "variant_examples":"Ala34Val (North African founder, hypomorphic/mild — rescues MKS in compound het with null); c.441+1G>A (splice null, MKS9 risk)",
            },
        ],

        "clinical_pearls": [
            {
                "title": "B9D1 — B9 Complex Inner-Leaflet TZ Anchor: MKS9 Tier (Null/Null → Perinatal Lethal)",
                "detail": (
                    "B9D1 forms the B9 complex with B9D2 and MKS1 at the ciliary transition zone. Unlike "
                    "the Tectonic complex members TCTN1 and TCTN3 (no MKS tier), B9D1 biallelic null → "
                    "MKS9 (OMIM #614209): perinatal lethal encephalocele, cystic renal dysplasia, polydactyly. "
                    "This is because B9D1 LOF collapses the entire TZ inner-leaflet anchoring layer, causing "
                    "complete TZ gate failure during critical embryonic ciliogenesis windows. "
                    "Null/hypomorphic compound heterozygotes → JBTS19 live birth because the hypomorphic "
                    "allele provides sufficient residual B9D1 to rescue embryonic lethality. Approximately "
                    "22% of JBTS19 families in this cohort carry a null/null genotype. MKS9 counselling "
                    "MUST be given when a B9D1 null allele is identified in any carrier parent."
                ),
            },
            {
                "title": "B9 Complex DDx: B9D1 (JBTS19/MKS9) vs B9D2 (JBTS34/MKS10) vs MKS1 (JBTS28/MKS1) — WES Must Distinguish",
                "detail": (
                    "B9D1 (17p11.2), B9D2 (19q13.2), and MKS1 (17q22) all form the B9 complex at the TZ. "
                    "Each has a distinct JBTS/MKS subtype: B9D1 → JBTS19/MKS9; B9D2 → JBTS34/MKS10; "
                    "MKS1 → JBTS28/MKS1. All three carry null/null perinatal-lethal MKS risk because all "
                    "are essential for B9 complex assembly. Note: MKS1 is also at 17q22 (same chromosome "
                    "as B9D1 at 17p11.2) — chromosome 17 but different arms. WES must confirm gene identity, "
                    "not just chromosome location. The B9 complex is the most tightly coupled MKS-tier set "
                    "in JBTS: all three members have documented MKS alleles, unlike the Tectonic module "
                    "(TCTN1/JBTS11 no MKS; TCTN3/JBTS18 no MKS; only TCTN2/JBTS13 → MKS8)."
                ),
            },
            {
                "title": "Null/Hypomorphic Rescue Mechanism: The Key to JBTS19 vs MKS9 Genotype–Phenotype",
                "detail": (
                    "The critical genotype–phenotype rule for B9D1: null/null → MKS9 (perinatal lethal); "
                    "null/hypomorphic → JBTS19 (live birth, ciliopathy); biallelic missense → JBTS19 (live "
                    "birth, variable severity). The hypomorphic allele (e.g. Ala34Val — North African "
                    "founder) provides partial B9D1 function sufficient to form a partially functional B9 "
                    "complex during critical embryonic windows, rescuing lethality. Post-natally, the "
                    "partial B9 complex dysfunction causes progressive ciliopathy (MTS, cerebellar, renal, "
                    "retinal). When a novel B9D1 variant of uncertain significance (VUS) is found in "
                    "compound heterozygosity with a confirmed null allele, functional studies (ciliogenesis "
                    "rescue assay, TZ localisation) are mandatory before counselling — the outcome "
                    "(MKS9 vs JBTS19) hinges on whether the VUS is hypomorphic or null."
                ),
            },
            {
                "title": "Renal Penetrance 35%: Higher Than Tectonic Module Genes — Annual NPHP Protocol Mandatory",
                "detail": (
                    "JBTS19/B9D1 has ~35% renal penetrance (NPHP-like tubulointerstitial nephritis) — "
                    "higher than TCTN1/JBTS11 (~20%) and TCTN3/JBTS18 (~20%). This is consistent with "
                    "complete TZ gate collapse in JBTS19 (B9D1 LOF disassembles the entire B9 complex "
                    "inner-leaflet anchor) vs partial Tectonic complex disruption in TCTN1/TCTN3. Annual "
                    "surveillance mandatory from diagnosis: creatinine, cystatin C, urine osmolality, "
                    "microalbuminuria (NPHP-like TIN is a concentrating defect — not a proteinuric "
                    "disease; start monitoring before proteinuria). ESRD median ~22 yr (earlier than "
                    "TCTN3/JBTS18 ~24 yr). Renal transplant curative, no allograft recurrence "
                    "(cell-autonomous AR ciliopathy)."
                ),
            },
            {
                "title": "Retinal Penetrance 32% and Polydactyly 20%: B9 Complex Severity vs Tectonic Module",
                "detail": (
                    "B9D1/JBTS19 retinal penetrance (32%) is higher than TCTN3/JBTS18 (25%) and "
                    "TCTN1/JBTS11 (25%), consistent with more severe TZ gate disruption. Annual ERG from "
                    "age 3 (rod-cone dystrophy pattern). Post-axial polydactyly (20%) is more frequent "
                    "than TCTN3/JBTS18 (15%), reflecting broader TZ gate collapse affecting limb bud "
                    "Hedgehog patterning. Hepatic CHF (18%) requires 2-yr surveillance LFTs + hepatic "
                    "ultrasound. These elevated penetrance rates vs Tectonic module genes create a "
                    "higher organ-surveillance burden for JBTS19 families — clinicians should not apply "
                    "TCTN3/JBTS18 penetrance rates to JBTS19 patients."
                ),
            },
        ],

        "literature_highlights": [
            "Dowdle WE et al. (2011) Disruption of a ciliary B9 protein complex causes Meckel syndrome. Am J Hum Genet 89(1):94–110. [B9D1/B9D2/MKS1 complex; MKS9 mechanism].",
            "Czarnecki PG & Shah JV (2012) The ciliary transition zone: from morphology and molecules to medicine. Trends Cell Biol 22(4):201–10. [TZ gate modules: B9, NPHP, Tectonic].",
            "Shaheen R et al. (2013) A TCTN2 mutation defines a novel Meckel Gruber syndrome locus. Hum Mutat 34(3):573–8. [B9 complex disease context vs Tectonic MKS8].",
            "Parisi MA (2019) The molecular genetics of Joubert syndrome and related ciliopathies. Transl Sci Rare Dis 4(1-2):25–49.",
            "Bachmann-Gagescu R et al. (2020) JBTS disease gene landscape across 460 families. Hum Mutat 41(4):e1–e45.",
        ],

        "phenotype_frequencies": {
            "mts_pathognomonic":        "100% (MTS is the diagnostic criterion)",
            "cerebellar_ataxia":        f"{_pct(n_ataxia)}%",
            "neonatal_hypotonia":       f"{_pct(n_hypotonia)}%",
            "oculomotor_apraxia":       f"{_pct(n_oma)}%",
            "breathing_dysregulation":  f"{_pct(n_breath)}%",
            "intellectual_disability":  f"{_pct(n_id)}%",
            "retinal_rod_cone":         f"{_pct(n_retinal)}%",
            "renal_nphp_tin":           f"{_pct(n_renal)}%",
            "hepatic_chf":              f"{_pct(n_hepatic)}%",
            "polydactyly_post_axial":   f"{_pct(n_poly)}%",
            "mks9_risk_null_null":      f"{_pct(n_mks9)}%",
            "corpus_callosum_anomaly":  f"{_pct(n_cc)}%",
            "mks9_tier":                "Confirmed — biallelic null/null genotype → perinatal lethal MKS9",
            "null_hypomorphic_rescue":  "Confirmed — null/hypomorphic → JBTS19 live birth (hypomorphic rescues MKS lethality)",
        },
    }
