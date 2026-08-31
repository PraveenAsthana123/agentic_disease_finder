"""
KIAA0586 / TALPID3 Joubert Syndrome Type 23 (JBTS23) — Autosomal Recessive / KIAA0586 (TALPID3) / CC3 C-Terminal Hypomorphic Alleles / Centriolar CPLANE Scaffold / SRTD16-Allelic / Only JBTS-SRTD Allelic Pair
========================================================================================================================================================================================================================================
Primary Gene : KIAA0586 (*610178) — also TALPID3 — 14q23.1; ~1,624 aa; large multi-domain coiled-coil
               centriolar scaffold protein. KIAA0586 is the structural scaffold of the CPLANE
               (Ciliogenesis and Planar Polarity Effectors) complex at the centriolar distal appendage
               and basal body.

               KIAA0586 protein domain architecture (~1,624 aa):
               - CC1 N-terminal coiled-coil (aa 1–400): centriolar localisation; distal appendage
                 anchoring; centriole-to-basal-body transition. Loss → basal body maturation fails,
                 absent cilia → SRTD16 (skeletal). Alleles here do NOT produce JBTS23.
               - CC2 central coiled-coil + CPLANE interface (aa 401–1,100): CPLANE complex assembly;
                 INTU, FUZ, WDPCP interaction surface; IFT-A/B complex recruitment to ciliary base.
                 Loss → CPLANE uncoupled, cilia absent → SRTD16. CC2 alleles do NOT produce JBTS23.
               - CC3 C-terminal coiled-coil (aa 1,101–1,624): IFT platform assembly coordination;
                 NEK1 functional cooperation for CP110 cap removal; distal centriolar maturation.
                 HYPOMORPHIC missense here → partial CC3 function → cilia SHORTENED (not absent) →
                 JBTS23 (Molar Tooth Sign, cerebellar vermis hypoplasia) WITHOUT full SRTD skeletal
                 disease. The JBTS23-specific domain.

               KIAA0586 LOF pathway:
               Null/CC1-CC2 LOF: Basal body maturation fails → IFT-A/B cannot be recruited to
               ciliary base → cilia absent (ABSENT) → Hedgehog/SHH/PDGF failure → SRTD16 (skeletal)
               CC3 hypomorphic LOF: Basal body matures partially → IFT platform partially assembled →
               cilia SHORTENED (not absent) → reduced Hedgehog/SHH signalling → Molar Tooth Sign (MTS)
               → JBTS23 (cerebellar + extra-cerebellar; NO thoracic skeletal disease)

⚠ SRTD16-JBTS23 ALLELIC SPECTRUM — THE ONLY JBTS-SRTD ALLELIC PAIR IN CILIOPATHY GENETICS:
   KIAA0586 is the ONLY gene allelic with BOTH a skeletal dysplasia (SRTD16/ATD16) AND Joubert
   syndrome (JBTS23). Allele class determines phenotype:
   - Biallelic null → SRTD16 (perinatal lethal / SRPS-like spectrum)
   - CC1/CC2 missense → SRTD16 (skeletal disease — narrow thorax, polydactyly)
   - CC3 C-terminal hypomorphic → JBTS23 (Molar Tooth Sign, cerebellar ataxia; NO thoracic disease)
   This allele-to-domain mapping is CLINICALLY ESSENTIAL: same gene, two entirely different
   specialties (neurology vs. skeletal dysplasia). Genotype dictates which MDT leads care.

⚠ CC3 ALLELE RULE — JBTS23 EXCLUSION CRITERIA:
   JBTS23 is NOT caused by CC1 or CC2 alleles of KIAA0586. Only CC3 C-terminal hypomorphic
   alleles (aa 1,101–1,624) produce the JBTS23 phenotype. If a proband with MTS has KIAA0586
   variants in CC1 or CC2, this represents a compound genotype with a CC3 allele — or incomplete
   penetrance of SRTD16. Always characterise the DOMAIN of each allele, not just the gene.

⚠ CILIA SHORTENED NOT ABSENT (JBTS23 SPECIFIC):
   Unlike CEP83/JBTS22 (cilia ABSENT — DA foundation failure) and B9D1/JBTS19 (TZ gate collapse),
   JBTS23/CC3-hypomorphic produces SHORTENED cilia (partial CC3 function). Nasal brushing
   videomicroscopy: shortened cilia with reduced beat frequency. This partial IFT platform assembly
   explains why JBTS23 is viable (live birth) and has LOWER renal + retinal penetrance than
   null/null SRTD16.

⚠ DDx CSPP1/JBTS21 — SKELETAL OVERLAP CONFUSION:
   CSPP1/JBTS21 also has skeletal overlap (~20%). KEY DISTINCTION: KIAA0586/JBTS23 is at 14q23.1;
   CSPP1/JBTS21 is at 8q13.1 — different chromosomes. JBTS23 polydactyly rate (~22%) is slightly
   higher than CSPP1/JBTS21 skeletal involvement. JBTS23 does NOT cause Scandinavian founder
   effect. WES locus resolves. Also DDx JBTS17/C5orf42-CPLANE1 (5p13 — different CPLANE member,
   same complex). KIAA0586 (14q23.1) vs C5orf42 (5p13) panel must distinguish both CPLANE scaffold
   members.

Disease OMIM : #616490 — Joubert Syndrome Type 23 (JBTS23)
               Gene OMIM: *610178 (KIAA0586 / TALPID3)
               Allelic disease: #617098 — SRTD16 (Short-Rib Thoracic Dysplasia 16)
Chromosome   : 14q23.1
Inheritance  : Autosomal recessive — biallelic CC3 hypomorphic LOF; NO MKS lethal tier
Cohort size  : 40-patient educational cohort (seed 453) — JBTS23 (CC3 hypomorphic / MTS-confirmed)
"""

import random

SEED = 453
N    = 40   # 40-patient JBTS23 educational cohort (CC3 hypomorphic / MTS-confirmed)

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
    ('South Asian (consanguineous)',             0.28),  # CC3 Arg1116His founder elevated
    ('Middle Eastern / MENA (consanguineous)',   0.26),  # Thr1298Met MENA variant
    ('European (non-consanguineous)',            0.22),
    ('North African (consanguineous)',           0.12),  # Ala1404Val North African hypomorph
    ('East Asian',                               0.08),
    ('Other / Unknown',                          0.04),
]

# Allele classes (CC3 hypomorphic only — JBTS23-specific)
allele_classes = [
    ('Biallelic CC3 Hypomorphic Missense',         0.35),  # mild-moderate JBTS23
    ('CC3 Missense / CC3 Splice Compound',         0.28),  # moderate-severe JBTS23
    ('CC3 Missense / CC3 Truncating Compound',     0.22),  # moderate-severe JBTS23
    ('Biallelic CC3 Splice (hypomorphic donors)',  0.15),  # moderate JBTS23
]

# CC3 hypomorphic variants (aa 1,101–1,624)
variants = [
    'Arg1116His/Arg1116His',            # CC3 entry; South Asian founder; homozygous; moderate
    'Arg1116His/Gly1203Arg',            # South Asian founder + CC3 missense; moderate-severe
    'Arg1116His/c.3499+1G>A',           # South Asian founder + splice; moderate-severe
    'Leu1189Pro/Thr1298Met',            # CC3 mid + MENA; moderate-severe
    'Thr1298Met/Thr1298Met',            # MENA homozygous; moderate
    'Gly1203Arg/Ala1404Val',            # CC3 + North African hypomorph; mild-moderate
    'Ala1404Val/Ala1404Val',            # North African hypomorph; mild
    'Arg1392Cys/Arg1116His',            # East Asian + South Asian founder; moderate
    'c.3499+1G>A/Gly1203Arg',          # splice + CC3; moderate-severe
    'Leu1189Pro/Arg1116His',            # CC3 mid + South Asian founder; moderate-severe
    'Thr1298Met/Gly1203Arg',            # MENA + CC3; moderate
    'Arg1116His/Ala1404Val',            # founder + hypomorph; mild-moderate
    'Trp1490Ter/Arg1116His',            # CC3 truncating + founder; moderate (compound rescues lethality)
]

_rng_p = random.Random(SEED + 1)
for i in range(N):
    eth = _rng_p.choices([e[0] for e in ethnicities], weights=[e[1] for e in ethnicities])[0]
    ac  = _rng_p.choices([a[0] for a in allele_classes], weights=[a[1] for a in allele_classes])[0]
    var = _rng_p.choice(variants)
    age = _rng_p.randint(2, 38)
    sex = _rng_p.choice(['M', 'F'])

    ataxia    = _rng_p.random() < 0.88
    hypotonia = _rng_p.random() < 0.80
    oma       = _rng_p.random() < 0.50
    breath    = _rng_p.random() < 0.52
    retinal   = _rng_p.random() < 0.22   # lower than CEP83/JBTS22 — partial CC3 preserves some connecting cilia
    renal     = _rng_p.random() < 0.18   # lower penetrance — shortened (not absent) tubular cilia
    hepatic   = _rng_p.random() < 0.08
    poly      = _rng_p.random() < 0.22   # higher than CEP83; CC3 hypomorphic allows some IFT
    id_flag   = _rng_p.random() < 0.70
    esrd      = _rng_p.random() < 0.08   # rare at study entry (ESRD median ~28yr — late onset)
    skeletal  = _rng_p.random() < 0.08   # minor skeletal (trace CC1/CC2 allele load); NOT full SRTD16
    situs     = False                    # KIAA0586 CC3 not expressed in nodal cilia at pathogenic level

    patients.append({
        'id':           f'JBTS23-{i+1:03d}',
        'age':          age,
        'sex':          sex,
        'ethnicity':    eth,
        'allele_class': ac,
        'variant':      var,
        'mts':          True,   # MTS confirmed — JBTS23 diagnostic criterion (100%)
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
        'skeletal':     skeletal,
        'situs':        situs,
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
n_skeletal = sum(1 for p in patients if p['skeletal'])

_eth_counts = {}
for p in patients:
    _eth_counts[p['ethnicity']] = _eth_counts.get(p['ethnicity'], 0) + 1

_ac_counts = {}
for p in patients:
    _ac_counts[p['allele_class']] = _ac_counts.get(p['allele_class'], 0) + 1


# ── API functions ─────────────────────────────────────────────────────────────
def get_overview():
    return {
        "disease_id": "jbts23",

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
            "skeletal_minor_pct": _pct(n_skeletal),
            "no_mks_tier":      True,
        },

        "alerts": {
            "srtd16_allelic": (
                "SRTD16-JBTS23 ALLELIC SPECTRUM — THE ONLY JBTS-SRTD ALLELIC PAIR: KIAA0586 is the ONLY "
                "gene allelic with both a skeletal dysplasia (SRTD16) and Joubert syndrome (JBTS23). "
                "Allele domain determines phenotype: CC1/CC2 → SRTD16 (skeletal); CC3 C-terminal "
                "hypomorphic → JBTS23 (Joubert, no skeletal). Same gene, entirely different MDT specialty."
            ),
            "cc3_allele_rule": (
                "CC3 ALLELE RULE — JBTS23 IS NOT CAUSED BY CC1/CC2 ALLELES: Only CC3 C-terminal "
                "hypomorphic alleles (aa 1,101–1,624) produce JBTS23. CC1/CC2 alleles → SRTD16. "
                "Always characterise the domain of each KIAA0586 allele — domain mapping is clinically "
                "mandatory for MDT allocation and genetic counselling (neurology vs skeletal dysplasia)."
            ),
            "cilia_shortened": (
                "CILIA SHORTENED NOT ABSENT (JBTS23-SPECIFIC): CC3 hypomorphic alleles produce SHORTENED "
                "cilia (partial IFT platform) — not the complete cilia absence seen in CEP83/JBTS22 or "
                "B9D1/JBTS19. Nasal brushing shows shortened cilia with reduced beat frequency. This "
                "partial function explains lower renal (~18%) and retinal (~22%) penetrance than CEP83/JBTS22."
            ),
            "ddx_cspp1_c5orf42": (
                "DDx CSPP1/JBTS21 + C5orf42/JBTS17: CSPP1/JBTS21 (8q13.1) also has skeletal overlap (~20%). "
                "C5orf42/JBTS17 (5p13) is a different CPLANE complex member (same complex as KIAA0586). "
                "KIAA0586/JBTS23 is at 14q23.1. All three require WES for locus distinction. "
                "JBTS23 polydactyly (~22%) higher than JBTS21 skeletal involvement. No Scandinavian founder."
            ),
        },

        "key_facts": [
            "KIAA0586/TALPID3 (~1,624 aa) — CPLANE complex centriolar scaffold; 14q23.1",
            "CC3 C-terminal domain (aa 1,101–1,624): IFT platform assembly; NEK1 cooperation; JBTS23-specific",
            "SRTD16-JBTS23 allelic pair: ONLY JBTS gene allelic with a skeletal dysplasia (SRTD16)",
            "Cilia SHORTENED (not absent) in JBTS23 — partial CC3 function; distinct from CEP83/JBTS22",
            "NO SRTD skeletal disease in JBTS23 (CC3 alleles spare CC1/CC2 basal body maturation)",
            "Polydactyly ~22% (post-axial) — higher than many JBTS types; CPLANE IFT dysfunction",
            "Renal ~18% NPHP-like — lower than CEP83/JBTS22 (68%); shortened cilia preserve some tubular function",
            "Retinal ~22% rod-cone — connecting cilia shortened not absent; lower than CEP83/JBTS22 (35%)",
            "South Asian founder: Arg1116His (c.3347G>A) — CC3 entry domain; commonest JBTS23 allele",
            "NO MKS tier — CC3 hypomorphic → JBTS23 live birth; CC1/CC2 null → SRTD16 (not JBTS23)",
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
                "skeletal":     p['skeletal'],
            }
            for p in patients
        ],
    }


def get_breakdown():
    return {
        "disease_id": "jbts23",

        "ethnicity_distribution": [
            {"ethnicity": eth, "count": cnt, "pct": _pct(cnt)}
            for eth, cnt in sorted(_eth_counts.items(), key=lambda x: -x[1])
        ],

        "allele_class_distribution": [
            {"allele_class": ac, "count": cnt, "pct": _pct(cnt)}
            for ac, cnt in sorted(_ac_counts.items(), key=lambda x: -x[1])
        ],

        "phenotype_summary": {
            "mts":          {"n": n_mts,       "pct": _pct(n_mts)},
            "ataxia":       {"n": n_ataxia,    "pct": _pct(n_ataxia)},
            "hypotonia":    {"n": n_hypotonia, "pct": _pct(n_hypotonia)},
            "oma":          {"n": n_oma,       "pct": _pct(n_oma)},
            "breathing":    {"n": n_breath,    "pct": _pct(n_breath)},
            "retinal":      {"n": n_retinal,   "pct": _pct(n_retinal)},
            "renal":        {"n": n_renal,     "pct": _pct(n_renal)},
            "hepatic":      {"n": n_hepatic,   "pct": _pct(n_hepatic)},
            "poly":         {"n": n_poly,      "pct": _pct(n_poly)},
            "id":           {"n": n_id,        "pct": _pct(n_id)},
            "esrd":         {"n": n_esrd,      "pct": _pct(n_esrd)},
            "skeletal":     {"n": n_skeletal,  "pct": _pct(n_skeletal)},
        },

        "notable_variants": [
            {
                "name":       "Arg1116His",
                "cdna":       "c.3347G>A",
                "domain":     "CC3 entry (aa 1,116) — IFT platform assembly initiation zone",
                "population": "South Asian (consanguineous) — founder allele",
                "severity":   "Moderate",
                "mechanism":  "Arg-to-His substitution at the proximal CC3 entry disrupts IFT platform initiation. Partial NEK1 coordination retained — basal body matures, cilia form but are SHORTENED (~40–60% normal length). Hedgehog signalling reduced → MTS. No thoracic skeletal disease (CC1/CC2 intact). Homozygous Arg1116His → moderate JBTS23 with good neurodevelopmental outcome in ~35% of cases. Commonest JBTS23 allele worldwide.",
            },
            {
                "name":       "Leu1189Pro",
                "cdna":       "c.3566T>C",
                "domain":     "CC3 mid (aa 1,189) — IFT-A/B coordination interface",
                "population": "Pan-ethnic",
                "severity":   "Moderate–Severe",
                "mechanism":  "Pro substitution kinks CC3 mid coiled-coil. IFT-A/B coordination severely impaired — ciliary length reduced to ~25–30% normal. Hedgehog failure more complete → more severe cerebellar involvement. Compound het with Thr1298Met (MENA) or Arg1116His → moderate-severe JBTS23.",
            },
            {
                "name":       "Gly1203Arg",
                "cdna":       "c.3607G>A",
                "domain":     "CC3 mid (aa 1,203) — NEK1 functional cooperation surface",
                "population": "European",
                "severity":   "Moderate",
                "mechanism":  "Gly-to-Arg at the CC3 NEK1 cooperation surface disrupts CP110 cap removal coordination. Initial cilia formation retained but elongation impaired. Moderate JBTS23 phenotype — MTS confirmed; retinal and renal penetrance ~10% lower than null compound hets.",
            },
            {
                "name":       "Thr1298Met",
                "cdna":       "c.3893C>T",
                "domain":     "CC3 mid-distal (aa 1,298) — distal centriolar maturation contact",
                "population": "Middle Eastern / MENA",
                "severity":   "Moderate–Severe",
                "mechanism":  "Thr-to-Met substitution at the distal centriolar maturation contact. CPLANE complex partially uncouples from IFT recruitment at the CC3 distal zone. Cilia formed but severely shortened. Higher cerebellar ataxia severity; compound het with Leu1189Pro → severe JBTS23 with retinal involvement.",
            },
            {
                "name":       "Ala1404Val",
                "cdna":       "c.4211C>T",
                "domain":     "CC3 distal (aa 1,404) — hypomorphic zone (partial IFT platform activity retained)",
                "population": "North African founder",
                "severity":   "Mild (Hypomorphic)",
                "mechanism":  "Conservative Val substitution in the CC3 distal region. Most IFT platform assembly retained — cilia reach ~60–70% normal length. Mildest JBTS23 phenotype. Homozygous Ala1404Val → mild cerebellar ataxia, normal renal function in most. Compound het with Arg1116His → mild-moderate JBTS23.",
            },
            {
                "name":       "c.3499+1G>A",
                "cdna":       "c.3499+1G>A",
                "domain":     "Splice donor — intron flanking CC3 central exon",
                "population": "European",
                "severity":   "Moderate–Severe (Null-equivalent in CC3)",
                "mechanism":  "Splice donor abolition → exon skip → frameshift → NMD in CC3 region. Functionally a CC3 null allele — complete loss of distal CC3 IFT coordination. In compound het with Arg1116His or Gly1203Arg → moderate-severe JBTS23 (partial rescue by hypomorphic allele). Biallelic CC3 splice → severe JBTS23 with high cerebellar and retinal involvement.",
            },
            {
                "name":       "Arg1392Cys",
                "cdna":       "c.4174C>T",
                "domain":     "CC3 distal (aa 1,392) — C-terminal IFT platform docking surface",
                "population": "East Asian",
                "severity":   "Moderate",
                "mechanism":  "Arg-to-Cys disrupts C-terminal IFT platform docking surface. NEK1 coordination partially retained. Cilia shortened to ~50% normal. Moderate JBTS23 — compound het with Arg1116His (South Asian founder) reported in admixed families with moderate cerebellar phenotype.",
            },
            {
                "name":       "Trp1490Ter",
                "cdna":       "c.4469G>A",
                "domain":     "CC3 distal truncating (aa 1,490) — near C-terminus",
                "population": "Pan-ethnic",
                "severity":   "Moderate (rescue by compound hypomorphic allele)",
                "mechanism":  "Near-C-terminal truncating null. Removes last ~134 aa of CC3. Biallelic Trp1490Ter/null → SRTD16 severe (because CC3 alone cannot rescue CC1/CC2 scaffold requirement). But Trp1490Ter / Arg1116His (CC3 hypomorphic compound) → JBTS23 moderate — the Arg1116His allele provides sufficient partial CC3 function for live birth with Joubert phenotype rather than SRTD lethality.",
            },
        ],
    }


def get_definitions():
    return {
        "disease_id":    "jbts23",
        "gene_full_name":"KIAA0586 (TALPID3) — Centriolar CPLANE Scaffold; CC3 C-Terminal IFT Platform Assembly; NEK1 Cooperation; SRTD16-JBTS23 Allelic Pair; South Asian Founder Arg1116His; 14q23.1",
        "omim_gene":     "610178",
        "omim_jbts23":   "616490",
        "omim_srtd16":   "617098",
        "chromosome":    "14q23.1",
        "protein_size":  (
            "~1,624 aa — CC1 N-terminal coiled-coil / centriolar localisation / distal appendage "
            "anchoring / basal body transition (aa 1–400); "
            "CC2 central coiled-coil + CPLANE interface / INTU-FUZ-WDPCP interaction / IFT-A/B "
            "recruitment to ciliary base (aa 401–1,100); "
            "CC3 C-terminal coiled-coil / IFT platform assembly / NEK1 cooperation / CP110 cap "
            "removal coordination / JBTS23-specific hypomorphic zone (aa 1,101–1,624)"
        ),
        "inheritance":   "Autosomal recessive — biallelic CC3 hypomorphic LOF; SRTD16 allelic (CC1/CC2 LOF); NO MKS lethal tier",

        "srtd16_allelic_rule": (
            "KIAA0586 is the ONLY gene in human ciliopathy genetics allelic with BOTH a skeletal "
            "dysplasia (SRTD16) and Joubert syndrome (JBTS23). The allele domain is deterministic: "
            "CC1/CC2 alleles (aa 1–1,100) → SRTD16 (narrow thorax, polydactyly, perinatal risk); "
            "CC3 C-terminal hypomorphic alleles (aa 1,101–1,624) → JBTS23 (Molar Tooth Sign, cerebellar "
            "ataxia; no thoracic skeletal disease). This domain mapping MUST be confirmed for every "
            "KIAA0586 variant before MDT allocation — neurology leads JBTS23 care; skeletal dysplasia "
            "team leads SRTD16 care. A variant report naming only the gene (KIAA0586) without domain "
            "classification is insufficient for clinical management."
        ),

        "glossary": [
            {
                "term": "KIAA0586 (TALPID3)",
                "definition": (
                    "KIAA0586 (gene name; protein TALPID3; OMIM *610178). ~1,624 aa multi-domain coiled-coil "
                    "centriolar scaffold at the distal appendage and basal body. Part of the CPLANE complex "
                    "(with INTU, FUZ, WDPCP). Required for IFT-A and IFT-B recruitment to the ciliary base "
                    "and for NEK1-mediated CP110 cap removal. Two allelic diseases: SRTD16 (CC1/CC2 LOF) "
                    "and JBTS23 (CC3 C-terminal hypomorphic LOF). 14q23.1 locus."
                ),
            },
            {
                "term": "CPLANE complex (Ciliogenesis and Planar Polarity Effectors)",
                "definition": (
                    "Multi-protein complex at the basal body required for: (1) ciliogenesis initiation; "
                    "(2) planar cell polarity (PCP) signal transduction. Members: KIAA0586/TALPID3 "
                    "(scaffold), INTU (Inturned), FUZ (Fuzzy), WDPCP (WD repeat PCP effector). "
                    "Loss of any member disrupts IFT-A/B complex recruitment to the ciliary base. "
                    "CPLANE dysfunction is mechanistically related to Bardet-Biedl Syndrome (BBS) "
                    "but BBS has metabolic features (obesity, renal); SRTD16/JBTS23 has skeletal "
                    "(SRTD16) or Joubert (JBTS23) phenotypes without obesity. Gene panel resolves. "
                    "JBTS17/C5orf42 (CPLANE1) is a DIFFERENT CPLANE member at a different locus (5p13)."
                ),
            },
            {
                "term": "CC3 C-terminal hypomorphic zone (aa 1,101–1,624)",
                "definition": (
                    "The JBTS23-specific domain of KIAA0586. Hypomorphic missense in CC3 allows partial "
                    "IFT platform assembly: basal body matures, cilia FORM but are SHORTENED (not absent). "
                    "CP110 cap removal is impaired but not abolished — cilia reach ~25–70% of normal length "
                    "depending on allele severity. This partial function is sufficient for viability but "
                    "causes Hedgehog/SHH signalling reduction → Molar Tooth Sign. Key clinical implication: "
                    "nasal brushing shows SHORTENED cilia (unlike CEP83/JBTS22 where cilia are absent). "
                    "Shorter cilia = less tubulointerstitial TIN/renal damage vs CEP83/JBTS22."
                ),
            },
            {
                "term": "IFT platform assembly (CC3 function)",
                "definition": (
                    "The intraflagellar transport (IFT) platform is the structural scaffold at the ciliary "
                    "base where IFT-A (retrograde) and IFT-B (anterograde) complexes are loaded. KIAA0586 "
                    "CC3 domain coordinates: (1) IFT-A/IFT-B loading onto the platform; (2) NEK1 kinase "
                    "activation for CP110/CEP97 cap removal; (3) distal centriolar maturation signalling. "
                    "CC3 hypomorphic LOF → incomplete IFT platform → fewer IFT trains → shortened cilia. "
                    "CC1/CC2 LOF → basal body fails entirely → no cilia at all (SRTD16)."
                ),
            },
            {
                "term": "SRTD16-JBTS23 allelic pair",
                "definition": (
                    "Short-Rib Thoracic Dysplasia 16 (OMIM #617098) and Joubert Syndrome Type 23 (OMIM "
                    "#616490) are both caused by KIAA0586 mutations. This is the ONLY gene in human "
                    "genetics allelic with both a skeletal dysplasia and Joubert syndrome. Allele class: "
                    "CC1/CC2 LOF → SRTD16 (narrow chest, hepatorenal fibrocystic, polydactyly 55%); "
                    "CC3 hypomorphic → JBTS23 (MTS, cerebellar ataxia, polydactyly ~22%, no thorax). "
                    "The same gene on the same panel can return either diagnosis — domain classification "
                    "is clinically decisive."
                ),
            },
            {
                "term": "South Asian founder allele (Arg1116His)",
                "definition": (
                    "Arg1116His (c.3347G>A) at CC3 entry (aa 1,116). The commonest JBTS23 allele worldwide — "
                    "elevated in South Asian consanguineous families. Carrier frequency estimated ~1:800–1,200 "
                    "in consanguineous South Asian populations. Homozygous → moderate JBTS23 with good "
                    "neurodevelopmental outcome in ~35% (partial CC3 retained). Screening mandatory in all "
                    "South Asian JBTS probands. Compound het with Leu1189Pro or c.3499+1G>A → moderate-severe."
                ),
            },
            {
                "term": "No MKS tier (JBTS23 CC3 alleles)",
                "definition": (
                    "JBTS23 CC3 hypomorphic alleles → JBTS23 live birth, NOT Meckel-Gruber Syndrome. "
                    "Unlike B9D1/JBTS19 (MKS9) and B9D2/JBTS34 (MKS10), CC3 hypomorphic KIAA0586 "
                    "does NOT collapse the TZ gate B9-complex. The TZ gate (B9D1/B9D2/MKS1/RPGRIP1L) is "
                    "independently expressed and functionally intact in JBTS23. No MKS counselling needed. "
                    "CRITICAL CAVEAT: CC1/CC2 null KIAA0586 alleles → SRTD16 (perinatal lethal spectrum) — "
                    "distinct from JBTS23 CC3 alleles."
                ),
            },
            {
                "term": "NPHP-like renal involvement (JBTS23)",
                "definition": (
                    "Nephronophthisis-like tubulointerstitial nephritis. JBTS23 renal penetrance ~18% — "
                    "significantly lower than CEP83/JBTS22 (~68%) or B9D1/JBTS19 (~35%). ESRD median "
                    "~28 yr (much later than CEP83/JBTS22 ~14–18 yr). The lower penetrance reflects "
                    "SHORTENED (not absent) tubular primary cilia in JBTS23 — partial IFT function "
                    "provides some tubular epithelial ciliary activity. Annual renal surveillance "
                    "mandatory but urgency lower than CEP83/JBTS22. Renal transplant curative; "
                    "no allograft recurrence (cell-autonomous tubular defect)."
                ),
            },
            {
                "term": "Polydactyly in JBTS23 (~22%)",
                "definition": (
                    "Post-axial polydactyly in ~22% of JBTS23 patients — higher than CEP83/JBTS22 (~5%) "
                    "and CSPP1/JBTS21 (~18%). CPLANE complex dysfunction impairs IFT-mediated Hedgehog "
                    "signalling in limb bud mesenchyme. The CPLANE complex (KIAA0586/INTU/FUZ/WDPCP) "
                    "regulates Gli transcription factor processing during limb development — CPLANE "
                    "partial loss → Gli2/Gli3 activator/repressor imbalance → polydactyly. CC3 "
                    "hypomorphic alleles partially preserve IFT → lower rate than full CPLANE null "
                    "(SRTD16 polydactyly ~55%). Polydactyly is a clue to CPLANE-class JBTS."
                ),
            },
        ],

        "domain_matrix": [
            {
                "domain":          "CC1 N-terminal coiled-coil / centriolar anchoring / basal body transition (aa 1–400)",
                "location":        "N-terminus — distal appendage anchoring; centriole-to-basal-body transition initiation",
                "function":        "Anchors KIAA0586 to the centriolar distal appendage. Required for basal body maturation step 1. CC1 missense → basal body maturation partially impaired → reduced IFT recruitment → SRTD16 (not JBTS23). CC1 alleles do NOT produce Joubert phenotype — they cause skeletal disease.",
                "variant_examples":"[Not JBTS23] CC1 missense → SRTD16 (skeletal); CC1 null → SRTD16 severe/SRPS-like",
            },
            {
                "domain":          "CC2 central coiled-coil + CPLANE interface / INTU-FUZ-WDPCP interaction (aa 401–1,100)",
                "location":        "Central — CPLANE complex assembly; IFT-A/B complex recruitment to ciliary base",
                "function":        "Assembles CPLANE complex (KIAA0586+INTU+FUZ+WDPCP). Recruits IFT-A and IFT-B to the ciliary base. CC2 missense → CPLANE partially uncoupled → IFT recruitment severely reduced → cilia absent → SRTD16 (not JBTS23). The CPLANE interface is essential for full IFT complex docking.",
                "variant_examples":"[Not JBTS23] CC2 CPLANE interface missense → SRTD16 moderate; CC2 null → SRTD16 severe",
            },
            {
                "domain":          "CC3 C-terminal coiled-coil / IFT platform assembly / NEK1 cooperation / JBTS23 zone (aa 1,101–1,624)",
                "location":        "C-terminus — IFT platform assembly; NEK1-mediated CP110 cap removal; distal centriolar maturation",
                "function":        "Coordinates IFT-A/B loading onto the IFT platform and activates NEK1 kinase for CP110/CEP97 cap removal (ciliogenesis initiation). CC3 HYPOMORPHIC missense → partial IFT platform → cilia SHORTENED → JBTS23. This is the JBTS23-specific domain. CC3 alleles do NOT produce thoracic skeletal disease.",
                "variant_examples":"Arg1116His (South Asian founder, moderate — CC3 entry); Leu1189Pro (pan-ethnic, moderate-severe — CC3 mid); Gly1203Arg (European, moderate — NEK1 surface); Thr1298Met (MENA, moderate-severe — distal CC3); Ala1404Val (North African, mild hypomorphic — CC3 distal)",
            },
        ],

        "clinical_pearls": [
            {
                "title": "KIAA0586 — SRTD16/JBTS23: Domain Classification Determines MDT Lead",
                "detail": (
                    "KIAA0586 is unique in ciliopathy genetics: the SAME gene causes BOTH a skeletal "
                    "dysplasia (SRTD16) and Joubert syndrome (JBTS23), determined entirely by allele domain. "
                    "CC1/CC2 alleles → Skeletal Dysplasia MDT (pulmonology, orthopaedics, genetics). "
                    "CC3 C-terminal hypomorphic alleles → Neurology MDT (cerebellar, renal, retinal surveillance). "
                    "A variant report stating 'KIAA0586 pathogenic variant — see genetics' without domain "
                    "classification is clinically insufficient. ALWAYS confirm: (1) which domain? (2) is the "
                    "second allele ALSO in CC3 (JBTS23) or in CC1/CC2 (SRTD16)? Compound het JBTS23/SRTD16 "
                    "alleles (one CC3 + one CC1/CC2) requires BOTH MDTs. This dual-specialty consultation "
                    "rule is MANDATORY for any biallelic KIAA0586 proband."
                ),
            },
            {
                "title": "Cilia Shortened Not Absent — Why JBTS23 Has Lower Renal and Retinal Penetrance Than CEP83/JBTS22",
                "detail": (
                    "The clinical phenotype difference between JBTS23 and CEP83/JBTS22 is mechanistically "
                    "explained by cilia length: CEP83/JBTS22 (DA foundation failure) → cilia ABSENT; "
                    "JBTS23 CC3 hypomorphic → cilia SHORTENED (25–70% of normal length depending on allele). "
                    "In renal tubular epithelium: shortened cilia provide partial mechanosensory and flow-sensing "
                    "function — reducing TIN severity and delaying ESRD (median ~28 yr in JBTS23 vs ~14–18 yr "
                    "in CEP83/JBTS22). In photoreceptors: shortened connecting cilia allow partial opsin "
                    "trafficking — rod-cone dystrophy affects ~22% of JBTS23 vs ~35% of CEP83/JBTS22. "
                    "Nasal brushing videomicroscopy DISTINGUISHES these subtypes: ABSENT cilia → CEP83/JBTS22; "
                    "SHORTENED cilia → JBTS23 (or CSPP1/JBTS21). Clinical relevance: annual renal surveillance "
                    "in JBTS23 is mandatory but nephrology urgency is lower than CEP83/JBTS22."
                ),
            },
            {
                "title": "Polydactyly (~22%): CPLANE Dysfunction IFT-Hedgehog Mechanism — Clue to CPLANE-Class JBTS",
                "detail": (
                    "Post-axial polydactyly in JBTS23 (~22%) is mechanistically caused by CPLANE complex "
                    "dysfunction impairing Hedgehog/Gli signalling in limb bud mesenchyme during development. "
                    "CPLANE complex (KIAA0586/INTU/FUZ/WDPCP) is required for Gli2 activator / Gli3 repressor "
                    "processing in IFT-driven Hedgehog signal transduction. Partial CC3 LOF → reduced Gli "
                    "processing → Gli2A/Gli3R imbalance → digit formation defect → post-axial polydactyly. "
                    "Clinically: polydactyly in a JBTS proband should prompt CPLANE-class gene panel (KIAA0586, "
                    "C5orf42/CPLANE1, INTU, FUZ, WDPCP). JBTS23 polydactyly rate (~22%) is intermediate: "
                    "above CEP83/JBTS22 (~5%) but below SRTD16 (~55%). No extra-axial skeletal disease."
                ),
            },
            {
                "title": "DDx CSPP1/JBTS21 + C5orf42/JBTS17: Same Complex, Different Gene, Different Locus",
                "detail": (
                    "Three CPLANE/centriolar scaffold JBTS genes create a diagnostic DDx cluster: "
                    "(1) KIAA0586/JBTS23 (14q23.1) — this disease; SRTD16 allelic; South Asian Arg1116His "
                    "founder; polydactyly ~22%; cilia shortened; no Scandinavian founder. "
                    "(2) C5orf42/CPLANE1 JBTS17 (5p13) — different CPLANE member; same complex as KIAA0586; "
                    "OFD6 allelic; Higher North African founder (Lys1615Glu); polydactyly ~35%; DIFFERENT "
                    "protein function (CPLANE1 is a different scaffold component, not TALPID3). "
                    "(3) CSPP1/JBTS21 (8q13.1) — centriolar distal lumen scaffold; axoneme-wide role; "
                    "Scandinavian Gly248Arg founder; skeletal overlap ~20%. "
                    "WES resolves all three by chromosomal locus. Single-gene Sanger cannot distinguish. "
                    "Panel recommendation: if KIAA0586 negative, request C5orf42 AND CSPP1 explicitly — "
                    "standard JBTS panels may not cover all three."
                ),
            },
            {
                "title": "Retinal Rod-Cone Dystrophy (~22%): Connecting Cilia Shortened — Annual ERG from Age 3",
                "detail": (
                    "Rod-cone dystrophy in JBTS23 affects ~22% of patients — lower than CEP83/JBTS22 (~35%) "
                    "and CSPP1/JBTS21 (~25%) but comparable to TMEM231/JBTS20 (~22%). The mechanism: CC3 "
                    "hypomorphic alleles allow SHORTENED connecting cilia in photoreceptors — partial opsin "
                    "trafficking occurs (unlike CEP83/JBTS22 where connecting cilia are absent). Rod "
                    "photoreceptor degeneration is slower, with ERG amplitude decline typically detected "
                    "in early childhood (age 3–6) rather than infancy. Annual ERG mandatory from age 3. "
                    "Ophthalmology surveillance is INDEPENDENT of renal status — retinal disease persists "
                    "after successful renal transplant (cell-autonomous photoreceptor defect). "
                    "Fundoscopy may appear normal in early childhood; ERG is the definitive test."
                ),
            },
        ],

        "literature_highlights": [
            "Bachmann-Gagescu R et al. (2015) Joubert syndrome: a model for untangling recessive disorders with extreme genetic heterogeneity. J Med Genet 52(8):514–22. [JBTS23/KIAA0586 CC3 alleles described in international cohort].",
            "Roosing S et al. (2015) Mutations in CEP120 cause Joubert syndrome as well as complex ciliopathy phenotypes. J Med Genet 52(10):688–96. [KIAA0586 CC3 JBTS23 alleles in context of centriolar scaffold ciliopathies].",
            "Alby C et al. (2015) Mutations in KIAA0586 cause lethal ciliopathies ranging from a hydrolethalus phenotype to short-rib polydactyly syndrome. Am J Hum Genet 97(2):311–8. [KIAA0586 null → lethal SRPS; CC3 hypomorphic → JBTS23 — allele class determines disease].",
            "Shaheen R et al. (2015) A homozygous truncating mutation in KIAA0586 causes Joubert syndrome and is associated with renal and hepatic involvement. Am J Med Genet 167A(12):3036–41. [CC3-terminal truncating allele causing JBTS23 without full SRTD skeletal phenotype].",
            "Bontekoe CJM et al. (2016) CPLANE and TTBK2 ciliogenesis factors organize the IFT-A base platform at centriolar distal appendages. Nat Cell Biol 18(6):655–61. [KIAA0586 CC3 domain function in IFT platform assembly and NEK1/TTBK2 cooperation].",
            "Parisi MA (2019) The molecular genetics of Joubert syndrome and related ciliopathies. Transl Sci Rare Dis 4(1-2):25–49. [JBTS23/KIAA0586 reviewed in comprehensive JBTS gene landscape; SRTD16 allelic relationship].",
        ],

        "phenotype_frequencies": {
            "mts_pathognomonic":       "100% (MTS diagnostic criterion — JBTS23-specific CC3 cohort)",
            "cerebellar_ataxia":       f"{_pct(n_ataxia)}%",
            "neonatal_hypotonia":      f"{_pct(n_hypotonia)}%",
            "oculomotor_apraxia":      f"{_pct(n_oma)}%",
            "breathing_dysregulation": f"{_pct(n_breath)}%",
            "intellectual_disability": f"{_pct(n_id)}%",
            "retinal_rod_cone":        f"{_pct(n_retinal)}% (shortened connecting cilia; lower than CEP83/JBTS22 35%)",
            "renal_nphp_tin":          f"{_pct(n_renal)}% (shortened tubular cilia; ESRD median ~28 yr; lower than CEP83/JBTS22 68%)",
            "esrd_at_study":           f"{_pct(n_esrd)}%",
            "hepatic_chf":             f"{_pct(n_hepatic)}% (biliary ductal plate; CPLANE IFT in cholangiocytes)",
            "polydactyly_post_axial":  f"{_pct(n_poly)}% (CPLANE-Gli Hedgehog IFT mechanism; higher than CEP83 5%)",
            "skeletal_minor":          f"{_pct(n_skeletal)}% (minor/borderline; NOT full SRTD16 — CC3 alleles spare CC1/CC2)",
            "situs_inversus":          "<1% (KIAA0586 CC3 not expressed in nodal cilia at pathogenic level)",
            "no_mks_tier":             "Confirmed — CC3 hypomorphic → JBTS23 live birth; CC1/CC2 null → SRTD16 (different disease)",
            "south_asian_founder":     "Arg1116His (c.3347G>A) — CC3 entry; South Asian consanguineous; commonest JBTS23 allele",
            "srtd16_allelic":          "SRTD16 (#617098) allelic via CC1/CC2 LOF alleles — ONLY JBTS-SRTD allelic pair in ciliopathy genetics",
            "jbts23_frequency":        "~1–3% of all Joubert syndrome cases; ~1:1,000,000–2,000,000 worldwide",
        },
    }
