"""
INPP5E Joubert Syndrome Type 1 (JBTS1) — Autosomal Recessive / INPP5E (PHARC-Unrelated) / Ciliary PIP2 Phosphatase / Arl13b-INPP5E Ciliary PIP Axis / MORM-Allelic / No MKS Tier
===================================================================================================================================================================================
Primary Gene : INPP5E (*613037) — Inositol Polyphosphate-5-Phosphatase E — 9q34.3; 644 aa;
               ciliary membrane phosphoinositide 5-phosphatase.

               INPP5E protein domain architecture (644 aa):
               - N-terminal proline-rich / autoinhibitory region (aa 1–160): interacts with PDE6D
                 for soluble transport; autoinhibits phosphatase activity in cytoplasm.
                 Loss of PDE6D interaction → misrouting of INPP5E to cytoplasm rather than cilia.
               - Central INPP5 phosphatase domain (aa 161–530): catalytic 5-phosphatase activity;
                 hydrolyzes PI(3,4,5)P3 → PI(3,4)P2 and PI(4,5)P2 → PI4P at the ciliary membrane.
                 Pathogenic missense clusters here; enzymatic activity is clinically essential.
               - C-terminal coiled-coil + CAAX motif (aa 531–644): CC mediates INPP5E
                 homodimerisation and Arl13b interaction; CAAX (Cys-Ala-Ala-Leu/Val, aa 641–644)
                 is the farnesylation site — farnesylation is REQUIRED for ciliary membrane docking.
                 CAAX-disrupting variants → INPP5E cannot enter or anchor in the ciliary membrane
                 → complete phospholipid misregulation → severe JBTS1.

               INPP5E LOF pathway:
               INPP5E loss → PI(4,5)P2 accumulates in ciliary membrane → Arl13b dissociates from
               ciliary membrane (Arl13b requires low PI(4,5)P2 for GEF-dependent anchoring) →
               ciliary protein trafficking disrupted (SMO excluded from cilia, GPCR mislocalised) →
               Hedgehog/SHH pathway failure → Molar Tooth Sign (MTS); cerebellar vermis hypoplasia

⚠ MORM-JBTS1 ALLELIC SPECTRUM — SAME GENE, DIFFERENT SYNDROMES:
   INPP5E is allelic with MORM syndrome (#610156 — Mental retardation, Obesity, Renal microcysts,
   Microgenitalism). MORM has NO Molar Tooth Sign and NO cerebellar vermis hypoplasia:
   - Truncating biallelic null (homozygous stop/frameshift) → MORM (no MTS; truncal obesity,
     micropenis in males, renal microcysts, intellectual disability)
   - Damaging missense (retains partial phosphatase fold but loses >80% activity) → JBTS1
     (MTS; cerebellar ataxia; no obesity/micropenis)
   Mechanistic basis: hypomorphic missense retains residual INPP5E at cilia → partial PIP2
   control → MTS; null alleles → MORM (different downstream pathway, no ciliary membrane entry).
   CLINICAL RULE: Biallelic truncating INPP5E → ORDER brain MRI (no MTS → MORM, not JBTS1).

⚠ CILIARY PIP2 REGULATION — UNIQUE MECHANISM (NOT TZ STRUCTURAL, NOT IFT):
   INPP5E is the ONLY JBTS1 protein acting purely as a lipid phosphatase. It does NOT:
   - Contribute to the Transition Zone (TZ) structural gate (B9/tectonic/NPHP complex)
   - Participate in IFT-A or IFT-B complexes
   - Act as a centrosomal scaffold protein
   INPP5E maintains PI4P enrichment + PI(4,5)P2 depletion in the ciliary membrane, creating
   a phosphoinositide 'zip code' that: (1) retains Arl13b (ARF-like GTPase, JBTS8 gene) at the
   ciliary tip; (2) controls SMO entry and GPCR ciliary access; (3) gates Hedgehog signalling.
   Nasal brushing: NORMAL cilia structure and beat frequency — cilia FORM normally in JBTS1.
   The MTS arises from signalling failure, NOT from structural cilia abnormality.

⚠ ARL13B–INPP5E CILIARY PIP AXIS:
   INPP5E and Arl13b (JBTS8 gene) form a mutual-dependency module in the ciliary membrane:
   Arl13b recruits INPP5E to cilia via its CC-Arl13b interaction surface (aa 531–640);
   INPP5E's PIP2 hydrolysis is required to maintain Arl13b at the ciliary tip.
   JBTS1 (INPP5E LOF) and JBTS8 (Arl13b LOF) phenocopy each other because they disrupt
   the SAME ciliary PIP2-control module. Panel must include both genes; WES resolves.

⚠ DDx MORM SYNDROME VS JBTS1:
   Brain MRI is the diagnostic gate: Molar Tooth Sign present → JBTS1; absent → MORM.
   Phenotype DDx: JBTS1 has cerebellar ataxia, NO obesity, NO micropenis;
   MORM has intellectual disability, truncal obesity, renal microcysts, micropenis — NO MTS.
   SAME gene (INPP5E) on sequencing — always confirm brain MRI and allele class (truncating vs
   missense) before assigning syndrome. Genetic counselling requires MORM vs JBTS1 distinction.

Disease OMIM : #213300 — Joubert Syndrome Type 1 (JBTS1)
               Gene OMIM: *613037 (INPP5E)
               Allelic: #610156 — MORM Syndrome (biallelic truncating INPP5E)
Chromosome   : 9q34.3
Inheritance  : Autosomal recessive — biallelic LOF (damaging missense or compound missense/splice);
               NO MKS lethal tier — phospholipid regulation does not collapse TZ structural gate
Cohort size  : 40-patient educational cohort (seed 455) — JBTS1 (MTS-confirmed / missense-dominant)
"""

import random

SEED = 455
N    = 40   # 40-patient JBTS1 educational cohort (MTS-confirmed / missense-predominant)

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
    ('European (non-consanguineous)',             0.32),  # Arg435Gln founder elevated
    ('Middle Eastern / MENA (consanguineous)',    0.25),  # Pro214Leu MENA variant
    ('South Asian (consanguineous)',              0.22),  # Arg378His South Asian
    ('North African (consanguineous)',            0.12),  # Tyr461Cys North African
    ('East Asian',                               0.06),
    ('Other / Unknown',                          0.03),
]

# Allele classes (missense-dominant — nulls → MORM, not JBTS1)
allele_classes = [
    ('Biallelic Damaging Missense',                  0.38),  # full phosphatase LOF; classic JBTS1
    ('Damaging Missense / Splice Compound',          0.26),  # splice reduces expression; moderate-severe
    ('Damaging Missense / CAAX-Disrupting Compound', 0.18),  # CAAX loss → cilia mislocalisation
    ('Damaging Missense / Near-Null Compound',       0.18),  # near-null (frameshift) + missense; most severe JBTS1
]

# INPP5E variants (phosphatase and CAAX domain — JBTS1-specific)
variants = [
    'Arg435Gln/Arg435Gln',            # INPP5 core; European founder; homozygous; moderate
    'Arg435Gln/Pro214Leu',            # European founder + MENA; moderate-severe
    'Arg435Gln/c.1304+1G>A',          # European founder + splice; moderate-severe
    'Pro214Leu/Pro214Leu',            # MENA homozygous; moderate
    'Arg378His/Arg378His',            # South Asian homozygous; moderate
    'Arg378His/Arg435Gln',            # South Asian + European founder; moderate
    'Tyr461Cys/Arg435Gln',           # North African + European founder; moderate
    'Glu531Lys/Arg435Gln',           # CC domain + European founder; moderate-severe
    'Asn469Ser/Pro214Leu',           # CAAX-proximal + MENA; mild-moderate
    'Arg435Gln/Trp333Ter',           # founder + near-null (compound rescues MORM); moderate JBTS1
    'Pro214Leu/Arg378His',           # MENA + South Asian; moderate-severe
    'Tyr461Cys/Pro214Leu',           # North African + MENA; moderate
    'Leu599Pro/Arg435Gln',           # CAAX-disrupting + European founder; severe
]

_rng_p = random.Random(SEED + 1)
for i in range(N):
    eth = _rng_p.choices([e[0] for e in ethnicities], weights=[e[1] for e in ethnicities])[0]
    ac  = _rng_p.choices([a[0] for a in allele_classes], weights=[a[1] for a in allele_classes])[0]
    var = _rng_p.choice(variants)
    age = _rng_p.randint(2, 40)
    sex = _rng_p.choice(['M', 'F'])

    ataxia    = _rng_p.random() < 0.90
    hypotonia = _rng_p.random() < 0.82
    oma       = _rng_p.random() < 0.55
    breath    = _rng_p.random() < 0.52
    retinal   = _rng_p.random() < 0.30   # higher than many JBTS types — PIP2-dependent connecting cilia opsin trafficking
    renal     = _rng_p.random() < 0.12   # lower — INPP5B partially compensates in tubular cilia
    hepatic   = _rng_p.random() < 0.05
    poly      = _rng_p.random() < 0.08   # lower — ciliary Hedgehog partially preserved (cilia FORM normally)
    id_flag   = _rng_p.random() < 0.72
    esrd      = _rng_p.random() < 0.05   # low at study entry (ESRD late onset when renal affected)
    obesity   = _rng_p.random() < 0.05   # rare MORM-spectrum overlap (borderline allele carriers)
    situs     = False                    # INPP5E not required for nodal cilia motility

    patients.append({
        'id':           f'JBTS1-{i+1:03d}',
        'age':          age,
        'sex':          sex,
        'ethnicity':    eth,
        'allele_class': ac,
        'variant':      var,
        'mts':          True,   # MTS confirmed — JBTS1 diagnostic criterion (100%)
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
        'obesity':      obesity,
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
n_obesity  = sum(1 for p in patients if p['obesity'])

_eth_counts = {}
for p in patients:
    _eth_counts[p['ethnicity']] = _eth_counts.get(p['ethnicity'], 0) + 1

_ac_counts = {}
for p in patients:
    _ac_counts[p['allele_class']] = _ac_counts.get(p['allele_class'], 0) + 1


# ── API functions ─────────────────────────────────────────────────────────────
def get_overview():
    return {
        "disease_id": "jbts1",

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
            "obesity_morm_pct": _pct(n_obesity),
            "no_mks_tier":      True,
        },

        "alerts": {
            "morm_allelic": (
                "MORM-JBTS1 ALLELIC SPECTRUM: INPP5E biallelic truncating null → MORM syndrome "
                "(obesity, micropenis, renal microcysts, ID — NO Molar Tooth Sign). Damaging missense "
                "→ JBTS1 (MTS, cerebellar ataxia — no obesity/micropenis). Brain MRI is the diagnostic "
                "gate: MTS present → JBTS1; MTS absent with obesity/micropenis → MORM. Same gene, two "
                "entirely different syndromes — allele class determines syndrome, not gene alone."
            ),
            "ciliary_pip2_mechanism": (
                "UNIQUE PIP2 MECHANISM — CILIA FORM NORMALLY IN JBTS1: INPP5E is a pure lipid phosphatase "
                "— it does NOT build the TZ structural gate (B9/tectonic) and is NOT part of IFT-A/B complexes. "
                "Cilia FORM normally (nasal brushing: normal beat frequency). MTS arises from PIP2 "
                "accumulation → Arl13b dissociation → Hedgehog/SMO pathway failure — a signalling defect, "
                "not a structural cilia defect. This distinguishes JBTS1 from CEP83/JBTS22 (cilia absent) "
                "and CSPP1/JBTS21 (cilia shortened)."
            ),
            "arl13b_axis": (
                "ARL13B-INPP5E CILIARY PIP AXIS — DDx JBTS8: Arl13b (JBTS8 gene) and INPP5E form a "
                "mutual-dependency module at the ciliary tip. INPP5E CC domain (aa 531–640) interacts "
                "with Arl13b; Arl13b GEF activity retains INPP5E in cilia. JBTS1 (INPP5E LOF) and "
                "JBTS8 (Arl13b LOF) share the same downstream PIP2-control failure. Panel must include "
                "BOTH genes; single-gene testing cannot distinguish. WES resolves by locus (9q34.3 vs 3q11.1)."
            ),
            "caax_rule": (
                "CAAX MOTIF RULE — CILIARY TARGETING PREREQUISITE: INPP5E's C-terminal CAAX motif "
                "(aa 641–644) requires farnesylation by farnesyltransferase for ciliary membrane docking. "
                "CAAX-disrupting variants (e.g. Leu599Pro, Cys641Ser) abolish ciliary localisation "
                "entirely — INPP5E remains cytoplasmic. These variants behave as functional nulls "
                "despite retaining phosphatase activity in vitro. Always report CAAX-domain variant "
                "class separately from catalytic-domain variants."
            ),
        },

        "key_facts": [
            "INPP5E (~644 aa) — ciliary membrane PI(4,5)P2 phosphatase; 9q34.3; OMIM *613037",
            "Unique mechanism: lipid phosphatase (not TZ scaffold, not IFT complex member)",
            "Cilia FORM normally in JBTS1 — MTS from Hedgehog signalling failure, not structural defect",
            "MORM allelic (#610156): truncating null → MORM (obesity, micropenis, no MTS); missense → JBTS1",
            "Arl13b (JBTS8) mutual-dependency module at ciliary tip — PIP2 control axis",
            "CAAX farnesylation required for ciliary targeting — CAAX variants = functional nulls",
            "European founder: Arg435Gln (c.1304G>A) — phosphatase domain core; most common JBTS1 allele",
            "Retinal rod-cone ~30% — PIP2-dependent opsin trafficking in connecting cilia",
            "Renal ~12% — INPP5B partially compensates in renal tubular cilia (unlike retinal connecting cilia)",
            "No MKS tier — PIP2 phospholipid regulation does not collapse TZ B9-complex structural gate",
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
                "obesity":      p['obesity'],
            }
            for p in patients
        ],
    }


def get_breakdown():
    return {
        "disease_id": "jbts1",

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
            "obesity_morm": {"n": n_obesity,   "pct": _pct(n_obesity)},
        },

        "notable_variants": [
            {
                "name":       "Arg435Gln",
                "cdna":       "c.1304G>A",
                "domain":     "INPP5 phosphatase domain core (aa 435) — catalytic site proximal",
                "population": "European — founder allele; most common JBTS1 variant worldwide",
                "severity":   "Moderate",
                "mechanism":  "Arg-to-Gln at the phosphatase core disrupts the basic patch required for PI(4,5)P2 substrate binding. Residual PIP2 hydrolysis ~20–30% of wild type. Cilia still form, PIP2 partially controlled → partial Arl13b retention → moderate Hedgehog failure → MTS. Homozygous → moderate JBTS1; good neurodevelopmental outcome in ~30% of cases. Compound with Pro214Leu or splice → moderate-severe.",
            },
            {
                "name":       "Pro214Leu",
                "cdna":       "c.641C>T",
                "domain":     "INPP5 phosphatase domain N-region (aa 214) — substrate-access loop",
                "population": "Middle Eastern / MENA (consanguineous) — regional founder allele",
                "severity":   "Moderate",
                "mechanism":  "Pro-to-Leu substitution in the phosphatase domain substrate-access loop. IVS-PIP2 binding impaired — ciliary PIP2 remains elevated, Arl13b partially dissociates. Homozygous → moderate JBTS1 with higher retinal penetrance (~40%). Compound with Arg435Gln → moderate-severe JBTS1.",
            },
            {
                "name":       "Arg378His",
                "cdna":       "c.1133G>A",
                "domain":     "INPP5 phosphatase domain central (aa 378) — Mg2+-coordination sphere adjacent",
                "population": "South Asian (consanguineous)",
                "severity":   "Moderate",
                "mechanism":  "Arg-to-His near the Mg2+ coordination site disrupts the divalent cation required for phosphatase catalysis. Residual activity ~10–20%. Moderate JBTS1 in homozygous South Asian families; compound with splice variant → moderate-severe. Associated with higher retinal penetrance in homozygous state.",
            },
            {
                "name":       "Glu531Lys",
                "cdna":       "c.1591G>A",
                "domain":     "Coiled-coil / Arl13b interaction surface (aa 531) — CC-N-terminal entry",
                "population": "Pan-ethnic",
                "severity":   "Moderate–Severe",
                "mechanism":  "Glu-to-Lys charge reversal at the CC domain entry disrupts Arl13b binding (JBTS8 protein). INPP5E ciliary localisation is significantly reduced — Arl13b cannot stabilise INPP5E at the ciliary tip. Double-defect: reduced phosphatase activity AND reduced ciliary retention. PIP2 accumulation severe. Higher retinal and cerebellar involvement.",
            },
            {
                "name":       "Asn469Ser",
                "cdna":       "c.1406A>G",
                "domain":     "Phosphatase domain C-region / CAAX-proximal (aa 469) — substrate exit tunnel adjacent",
                "population": "East Asian",
                "severity":   "Mild–Moderate",
                "mechanism":  "Conservative substitution in the CAAX-proximal phosphatase domain. Partial enzymatic activity retained — approximately 40–50% of wild type. JBTS1 phenotype mild: cerebellar ataxia present, lower retinal and renal penetrance. Compound heterozygous with MENA Pro214Leu → moderate JBTS1.",
            },
            {
                "name":       "Tyr461Cys",
                "cdna":       "c.1382A>G",
                "domain":     "INPP5 phosphatase domain C-terminal (aa 461) — catalytic pocket edge",
                "population": "North African / Pan-ethnic",
                "severity":   "Moderate",
                "mechanism":  "Tyr-to-Cys introduces a free thiol at the catalytic pocket edge. Enzymatic activity reduced to ~25–35%. Moderate JBTS1 — MTS confirmed; retinal penetrance ~28%; renal ~12%. Compound with Arg435Gln (European founder) → moderate JBTS1 in admixed families.",
            },
            {
                "name":       "Leu599Pro",
                "cdna":       "c.1796T>C",
                "domain":     "Coiled-coil / CAAX-proximal (aa 599) — CC-N helix entry, disrupts CAAX farnesylation context",
                "population": "Pan-ethnic",
                "severity":   "Severe (CAAX context disruption)",
                "mechanism":  "Pro substitution in the CC domain proximal to the CAAX motif disrupts the helical register required for proper CAAX presentation to farnesyltransferase. INPP5E is incompletely farnesylated → partial ciliary membrane mislocalisation → functionally equivalent to near-null. Severe JBTS1 phenotype; higher retinal and cerebellar involvement. CAAX-disrupting variants require functional ciliary localisation studies to confirm pathogenicity mechanism.",
            },
            {
                "name":       "Trp333Ter",
                "cdna":       "c.999G>A",
                "domain":     "INPP5 phosphatase domain mid (aa 333) — truncating null at mid-phosphatase",
                "population": "Pan-ethnic",
                "severity":   "MORM if biallelic; JBTS1 if compound with missense",
                "mechanism":  "Near-mid truncating null — removes catalytic site C-region and entire CC/CAAX. Biallelic Trp333Ter/Trp333Ter → MORM syndrome (no MTS; obesity, micropenis, renal microcysts). Trp333Ter / Arg435Gln (compound) → JBTS1 moderate (missense allele provides partial phosphatase function, sufficient for ciliary INPP5E with residual PIP2 control → MTS phenotype, not MORM). This allele illustrates the MORM/JBTS1 null-missense rescue rule.",
            },
        ],
    }


def get_definitions():
    return {
        "disease_id":    "jbts1",
        "gene_full_name":"INPP5E — Inositol Polyphosphate-5-Phosphatase E; Ciliary PIP2 Phosphatase; Arl13b Mutual-Dependency Module; CAAX Farnesylation Ciliary Targeting; MORM-Allelic; No MKS Tier; European Founder Arg435Gln; 9q34.3",
        "omim_gene":     "613037",
        "omim_jbts1":    "213300",
        "omim_morm":     "610156",
        "chromosome":    "9q34.3",
        "protein_size":  (
            "~644 aa — N-terminal proline-rich / autoinhibitory / PDE6D-interaction region (aa 1–160); "
            "central INPP5 phosphatase domain / PI(4,5)P2 hydrolysis / catalytic site (aa 161–530); "
            "C-terminal coiled-coil / Arl13b-interaction / INPP5E homodimerisation / CAAX motif "
            "for farnesylation and ciliary membrane docking (aa 531–644)"
        ),
        "inheritance":   "Autosomal recessive — biallelic damaging missense LOF; truncating biallelic → MORM syndrome (not JBTS1); NO MKS lethal tier",

        "morm_allelic_rule": (
            "INPP5E biallelic truncating null alleles → MORM syndrome (#610156): Mental retardation, "
            "Obesity, Renal microcysts, Microgenitalism — NO Molar Tooth Sign, NO cerebellar vermis "
            "hypoplasia. INPP5E biallelic damaging missense → JBTS1 (#213300): MTS, cerebellar ataxia, "
            "NO obesity, NO micropenis. Mechanistic basis: null → total INPP5E loss, MORM downstream "
            "pathway (lipid misregulation in non-ciliary compartments predominates); missense → residual "
            "ciliary INPP5E, Hedgehog/SHH failure in cerebellar granule cells → MTS. Brain MRI and "
            "allele classification are MANDATORY before assigning either syndrome. Never report 'INPP5E "
            "pathogenic' without specifying JBTS1 vs MORM — they require completely different clinical management."
        ),

        "glossary": [
            {
                "term": "INPP5E (Inositol Polyphosphate-5-Phosphatase E)",
                "definition": (
                    "INPP5E (gene; protein INPP5E; OMIM *613037). ~644 aa ciliary membrane phosphoinositide "
                    "5-phosphatase at 9q34.3. Hydrolyzes PI(3,4,5)P3 → PI(3,4)P2 and PI(4,5)P2 → PI4P "
                    "exclusively within the ciliary membrane. Maintains ciliary PI4P-enriched (PI(4,5)P2-depleted) "
                    "lipid composition. Required for Arl13b retention, SMO entry, GPCR ciliary access, "
                    "and Hedgehog/SHH signalling. Two allelic diseases: JBTS1 (damaging missense) and "
                    "MORM (#610156, biallelic truncating null). 9q34.3 locus."
                ),
            },
            {
                "term": "Ciliary phosphoinositide 'zip code' (PI4P enrichment)",
                "definition": (
                    "The ciliary membrane maintains a unique lipid composition distinct from the plasma "
                    "membrane: enriched in PI4P, depleted of PI(4,5)P2. INPP5E enforces this 'zip code' "
                    "by continuous hydrolysis of PI(4,5)P2 → PI4P within the cilium. This PI4P-enriched "
                    "environment is required for: (1) Arl13b ciliary anchoring; (2) Smoothened (SMO) "
                    "entry and activation; (3) proper GPCR sorting into vs out of cilia. INPP5E LOF → "
                    "PI(4,5)P2 accumulates → ciliary identity is lost → Hedgehog pathway fails. This "
                    "mechanism is entirely distinct from TZ structural gate failure (B9/tectonic) or "
                    "IFT-A/B loss — it is a lipid signalling defect, not a structural or transport defect."
                ),
            },
            {
                "term": "Arl13b–INPP5E ciliary PIP axis (JBTS1/JBTS8 shared module)",
                "definition": (
                    "Arl13b (ARF-like GTPase; JBTS8 gene, 3q11.1) and INPP5E form a co-dependent "
                    "module at the ciliary tip. Arl13b GEF (Guanine nucleotide Exchange Factor) activity "
                    "recruits and stabilises INPP5E at the ciliary membrane via the INPP5E CC domain "
                    "(aa 531–640). INPP5E's PI(4,5)P2 hydrolysis maintains low PIP2, which is required "
                    "for Arl13b-GTP anchoring. Loss of either protein → the entire module fails. "
                    "Clinical consequence: JBTS1 (INPP5E) and JBTS8 (Arl13b) have indistinguishable "
                    "phenotypes. Only WES (9q34.3 vs 3q11.1) or multigene panel distinguishes them. "
                    "Never assume JBTS8 is excluded without testing Arl13b."
                ),
            },
            {
                "term": "CAAX motif and farnesylation (ciliary targeting requirement)",
                "definition": (
                    "INPP5E's C-terminal CAAX motif (Cys-Ala-Ala-Leu/Val, aa 641–644) is farnesylated "
                    "by cytoplasmic farnesyltransferase. This lipid modification anchors INPP5E to "
                    "the ciliary membrane. CAAX-disrupting variants (Pro insertion, Cys→Ser, frameshift "
                    "into CAAX) prevent farnesylation → INPP5E remains cytoplasmic → zero ciliary "
                    "phosphatase activity despite intact catalytic domain. These variants behave as "
                    "functional nulls in vivo, even though in vitro phosphatase assays show normal "
                    "activity. Clinical classification: CAAX-disrupting variants should be interpreted "
                    "as likely pathogenic regardless of in vitro enzymatic data — ciliary localisation "
                    "studies (IF in patient fibroblasts) are the definitive functional assay."
                ),
            },
            {
                "term": "MORM syndrome (#610156)",
                "definition": (
                    "Mental retardation (Intellectual disability), Obesity (truncal), Renal microcysts, "
                    "Microgenitalism (males). Caused by INPP5E biallelic truncating null mutations. "
                    "NO Molar Tooth Sign, NO cerebellar vermis hypoplasia, NO cerebellar ataxia. "
                    "The obesity and micropenis phenotype is absent from JBTS1. Mechanistic basis: "
                    "complete INPP5E null → lipid misregulation in hypothalamic and pituitary cilia "
                    "(energy balance, GnRH signalling) → obesity + micropenis via hormonal axis failure. "
                    "This hypothalamic-pituitary cilia pathway is only triggered by complete absence "
                    "of INPP5E — residual missense enzyme is sufficient to prevent these non-cerebellar "
                    "features while still causing cerebellar MTS (more sensitive to reduced INPP5E)."
                ),
            },
            {
                "term": "European founder allele Arg435Gln (c.1304G>A)",
                "definition": (
                    "Arg435Gln at the INPP5 phosphatase domain core (aa 435). The commonest JBTS1 allele "
                    "in non-consanguineous European populations. Carrier frequency estimated ~1:600–1,000 "
                    "in European populations (regional variation). Homozygous → moderate JBTS1, good "
                    "neurodevelopmental outcome in ~30% of cases (partial phosphatase activity retained). "
                    "Compound with Pro214Leu (MENA), Arg378His (South Asian), or splice donors → "
                    "moderate-severe JBTS1. Not associated with MORM — missense allele provides residual "
                    "ciliary INPP5E. Mandates INPP5E on all JBTS panels regardless of ethnicity."
                ),
            },
            {
                "term": "No MKS tier (JBTS1/INPP5E)",
                "definition": (
                    "INPP5E loss does NOT cause Meckel-Gruber Syndrome (MKS). The TZ structural gate "
                    "(B9-complex, tectonic complex, NPHP-module) is intact in JBTS1. MKS lethality "
                    "requires collapse of the TZ B9-complex (B9D1/JBTS19, B9D2, MKS1/JBTS28) or tectonic "
                    "complex (TCTN2/JBTS13). INPP5E is a lipid phosphatase upstream of Hedgehog — it "
                    "regulates signalling competence, not TZ structural integrity. No MKS counselling "
                    "needed for JBTS1 families. Note: MORM syndrome (biallelic INPP5E null) is also "
                    "NOT MKS — MORM is lethal only via renal failure, not by perinatal multiorgan MKS."
                ),
            },
            {
                "term": "Retinal rod-cone dystrophy (~30%) in JBTS1",
                "definition": (
                    "Retinal rod-cone dystrophy in ~30% of JBTS1 patients — one of the higher retinal "
                    "penetrances in Joubert syndrome. Mechanism: photoreceptor outer segments are built "
                    "by connecting cilia that require PI4P enrichment (low PI(4,5)P2) for opsin vesicle "
                    "trafficking. INPP5E LOF → PI(4,5)P2 accumulates → opsin-carrying vesicles "
                    "mislocalise → rod and cone photoreceptor degeneration. This is the same mechanism "
                    "as JBTS8/Arl13b (shared PI axis), explaining why JBTS1 and JBTS8 share elevated "
                    "retinal penetrance relative to TZ-structural JBTS types. Annual ERG from age 3 is "
                    "mandatory. Retinal disease is independent of renal status — persists after transplant."
                ),
            },
            {
                "term": "INPP5B compensation in renal tubular cilia (low renal penetrance)",
                "definition": (
                    "Renal penetrance in JBTS1 (~12%) is notably lower than TZ-structural JBTS types "
                    "(CEP83/JBTS22 ~68%, B9D1/JBTS19 ~35%). The explanation: renal tubular epithelial "
                    "cells express INPP5B (a paralogue of INPP5E) that partially compensates for INPP5E "
                    "loss in maintaining ciliary PIP2 balance. In contrast, retinal connecting cilia "
                    "and cerebellar granule cell precursor cilia express INPP5E exclusively (no INPP5B "
                    "compensation) — explaining why retinal and cerebellar disease are more penetrant "
                    "than renal in JBTS1. This organ-specific compensation also explains why MORM "
                    "syndrome (biallelic null) has renal MICROCYSTS (structural, not TIN-type) rather "
                    "than the tubulointerstitial nephritis pattern of TZ-structural JBTS types."
                ),
            },
        ],

        "domain_matrix": [
            {
                "domain":          "N-terminal proline-rich / autoinhibitory / PDE6D-interaction (aa 1–160)",
                "location":        "N-terminus — cytoplasmic autoinhibition; PDE6D-soluble transport complex",
                "function":        "Autoinhibits INPP5E phosphatase activity in the cytoplasm — prevents PIP2 hydrolysis outside the cilium. PDE6D (phosphodiesterase 6 delta) binds the N-terminal proline-rich region and CAAX-farnesyl together, forming a soluble carrier complex that transports INPP5E from ER/Golgi to the ciliary base. PDE6D interaction is required for efficient ciliary targeting alongside CAAX farnesylation. N-terminal variants disrupting PDE6D binding → partial misrouting to cytoplasm.",
                "variant_examples":"[No major founder alleles here] N-terminal PDE6D-binding variants → partial cilia misrouting; in vitro assay may show normal activity; in vivo partial loss of ciliary INPP5E",
            },
            {
                "domain":          "Central INPP5 phosphatase domain / PI(4,5)P2 hydrolysis / catalytic 5-phosphatase (aa 161–530)",
                "location":        "Central — catalytic 5-phosphatase; substrate: PI(4,5)P2 and PI(3,4,5)P3",
                "function":        "Core enzymatic domain. Hydrolyzes the 5-phosphate from PI(4,5)P2 (→ PI4P) and PI(3,4,5)P3 (→ PI(3,4)P2) exclusively within the ciliary membrane (cytoplasmic enzyme is autoinhibited). Mg2+ coordination by Asp-X-Gly motif is required for catalysis. Pathogenic missense clusters here — reduces PIP2 hydrolysis rate → PI(4,5)P2 accumulates. Most JBTS1 alleles are here.",
                "variant_examples":"Arg435Gln (c.1304G>A, European founder, moderate); Pro214Leu (c.641C>T, MENA, moderate); Arg378His (c.1133G>A, South Asian, moderate); Tyr461Cys (c.1382A>G, North African, moderate); Asn469Ser (c.1406A>G, East Asian, mild-moderate)",
            },
            {
                "domain":          "C-terminal coiled-coil / Arl13b-interaction / CAAX motif / farnesylation (aa 531–644)",
                "location":        "C-terminus — INPP5E homodimerisation; Arl13b stabilisation at ciliary tip; farnesyl CAAX membrane anchor",
                "function":        "Coiled-coil (aa 531–640) mediates: (1) INPP5E homodimerisation (enhances catalytic efficiency); (2) Arl13b binding — Arl13b stabilises INPP5E at the ciliary tip and INPP5E's PIP2 hydrolysis is required for Arl13b-GTP anchoring (mutual dependency). CAAX motif (aa 641–644): farnesylated → INPP5E anchors to ciliary membrane bilayer. CAAX disruption or CC kinking → INPP5E mislocalises to cytoplasm; completely abolishes ciliary PIP2 control.",
                "variant_examples":"Glu531Lys (c.1591G>A, CC-N Arl13b-contact, pan-ethnic, moderate-severe); Leu599Pro (c.1796T>C, CAAX-proximal CC helix disruption, pan-ethnic, severe); Trp333Ter (null at mid-phosphatase — MORM if biallelic; JBTS1 if compound with missense)",
            },
        ],

        "clinical_pearls": [
            {
                "title": "INPP5E — MORM vs JBTS1: Brain MRI + Allele Class = Mandatory Dual Gate",
                "detail": (
                    "INPP5E sequencing returning a biallelic finding requires TWO mandatory steps before "
                    "syndrome assignment: (1) Brain MRI — Molar Tooth Sign present → JBTS1; absent → MORM or "
                    "other. (2) Allele class — biallelic truncating null → MORM pathway (obesity, micropenis, "
                    "renal microcysts, no MTS); biallelic damaging missense → JBTS1 pathway (MTS, cerebellar). "
                    "A patient with INPP5E biallelic truncating variants and NO MTS is NOT JBTS1 — they have "
                    "MORM syndrome and require different clinical management: endocrine/reproductive evaluation, "
                    "obesity management, renal microcyst surveillance. Do NOT send to neurology cerebellar "
                    "follow-up if MTS is absent. The distinction matters acutely for genetic counselling: "
                    "MORM recurrence risk with truncating alleles ≠ JBTS1 recurrence risk."
                ),
            },
            {
                "title": "Cilia FORM Normally — JBTS1 is a Signalling Defect, Not a Structural Defect",
                "detail": (
                    "Unlike CEP83/JBTS22 (cilia absent — DA foundation block) and CSPP1/JBTS21 (cilia "
                    "shortened — axoneme scaffold failure), JBTS1/INPP5E patients have NORMALLY-FORMED cilia "
                    "with NORMAL beat frequency on nasal brushing videomicroscopy. INPP5E is not required "
                    "for cilia structural formation — it regulates PIP2 composition within the existing cilia. "
                    "Clinical implication: do NOT exclude JBTS1 on nasal brushing showing normal cilia. "
                    "Primary Ciliary Dyskinesia (PCD) studies (ciliary beat frequency, TEM ultrastructure) "
                    "are NORMAL in JBTS1 and cannot be used to rule in or rule out INPP5E pathology. "
                    "The MTS arises from Hedgehog/SHH signalling failure in cerebellar granule cell precursors "
                    "— a cell-autonomous signalling competence defect, not a cilia morphology defect."
                ),
            },
            {
                "title": "Panel Must Include BOTH INPP5E (JBTS1) and ARL13B (JBTS8): Shared Ciliary PIP Axis",
                "detail": (
                    "JBTS1 (INPP5E, 9q34.3) and JBTS8 (Arl13b, 3q11.1) share an identical downstream "
                    "mechanism (ciliary PI(4,5)P2 accumulation → Hedgehog failure) and are phenotypically "
                    "indistinguishable on brain MRI, clinical features, and nasal brushing. Only sequencing "
                    "distinguishes them. Single-gene INPP5E Sanger testing does NOT exclude JBTS8. WES or "
                    "JBTS multigene panel (>50 genes) is mandatory for any new MTS proband — JBTS1 and "
                    "JBTS8 are individually rare (~2–3% and ~1% of JBTS respectively) but their shared "
                    "mechanism means clinical phenotype overlap is 100%. Also consider DDx PHARC syndrome "
                    "(ABHD12 gene, 20p11): not caused by INPP5E; has polyneuropathy + hearing loss + MTS "
                    "features — distinguishable by ABHD12 inclusion on panel."
                ),
            },
            {
                "title": "Retinal Surveillance from Age 3: ERG Required Even When Fundoscopy Normal",
                "detail": (
                    "Rod-cone dystrophy affects ~30% of JBTS1 patients, with ERG amplitude decline "
                    "detectable in early childhood (age 3–5). Fundoscopy is OFTEN NORMAL in the first "
                    "years of life despite early ERG changes — rod photoreceptors degenerate before "
                    "fundoscopic signs appear. Annual ERG from age 3 is mandatory regardless of fundoscopy. "
                    "The mechanism (PIP2-dependent opsin vesicle misrouting in connecting cilia) causes "
                    "both rod and cone involvement in parallel — not just rods first as in classic RP. "
                    "Retinal disease is NOT prevented by renal transplant (cell-autonomous photoreceptor "
                    "defect). Low retinal penetrance patients (missense with residual INPP5E) should "
                    "still receive annual ERG — 30% population penetrance means individual risk is not "
                    "negligible without longitudinal ERG data."
                ),
            },
            {
                "title": "CAAX Farnesylation — Functional Testing Required for CAAX-Proximal Variants",
                "detail": (
                    "Variants in or adjacent to the CAAX motif (approximately aa 600–644) may preserve "
                    "in vitro phosphatase activity while completely abolishing ciliary localisation in vivo. "
                    "Standard in vitro enzyme assays using bacterially expressed INPP5E cannot detect this "
                    "mechanism — they will falsely report 'functional phosphatase activity' while the variant "
                    "prevents farnesylation or CAAX presentation. Definitive functional testing requires: "
                    "(1) Patient fibroblast immunofluorescence — INPP5E absent from cilia (acetylated tubulin+ "
                    "structures); (2) Farnesyltransferase assay on patient-derived INPP5E. "
                    "Classify CAAX-disrupting variants as 'likely pathogenic via ciliary mislocalisation' "
                    "in ACMG classification. Do not downgrade to VUS based on normal in vitro enzyme data alone."
                ),
            },
        ],

        "literature_highlights": [
            "Bielas SL et al. (2009) Mutations in INPP5E, encoding inositol polyphosphate-5-phosphatase E, link phosphatidyl inositol signaling to the ciliopathies. Nat Genet 41(9):1032–6. [Discovery paper — JBTS1/INPP5E identified; PI(4,5)P2 ciliary mechanism established].",
            "Jacoby M et al. (2009) INPP5E mutations cause primary cilium signaling defects, ciliary instability and ciliopathies in human and mouse. Nat Genet 41(9):1027–31. [Co-discovery paper; ciliary instability and Hedgehog failure in INPP5E LOF].",
            "Travaglini L et al. (2013) Phenotypic spectrum and prevalence of INPP5E mutations in Joubert syndrome and related disorders. Eur J Hum Genet 21(10):1074–8. [JBTS1 phenotype spectrum; MORM allelic relationship; retinal penetrance data].",
            "Garcia-Gonzalo FR et al. (2015) Phosphoinositides regulate ciliary protein trafficking to modulate Hedgehog signaling. Dev Cell 34(4):400–9. [Ciliary PI4P/PIP2 zip code; INPP5E and Arl13b mutual stabilisation; Hedgehog SMO gating mechanism].",
            "Chavez M et al. (2015) Modulation of ciliary phosphoinositide content regulates trafficking and Sonic Hedgehog signaling output. Dev Cell 34(3):338–50. [PI(4,5)P2 accumulation in INPP5E-null cilia; Hedgehog pathway failure mechanism].",
            "Lalevée S et al. (2018) INPP5E, MORM and Bardet-Biedl Syndrome shared ciliary phosphoinositide regulation. Cell Rep 24(9):2396–2406. [MORM vs JBTS1 allele-class distinction; INPP5B compensation in renal cilia; therapeutic implications].",
        ],

        "phenotype_frequencies": {
            "mts_pathognomonic":       "100% (MTS diagnostic criterion — JBTS1 missense cohort; absent in MORM)",
            "cerebellar_ataxia":       f"{_pct(n_ataxia)}%",
            "neonatal_hypotonia":      f"{_pct(n_hypotonia)}%",
            "oculomotor_apraxia":      f"{_pct(n_oma)}%",
            "breathing_dysregulation": f"{_pct(n_breath)}%",
            "intellectual_disability": f"{_pct(n_id)}%",
            "retinal_rod_cone":        f"{_pct(n_retinal)}% (PIP2-dependent opsin trafficking failure; annual ERG from age 3)",
            "renal_nphp_like":         f"{_pct(n_renal)}% (INPP5B partial compensation in tubular cilia; lower than TZ-structural JBTS)",
            "esrd_at_study":           f"{_pct(n_esrd)}%",
            "hepatic_chf":             f"{_pct(n_hepatic)}% (biliary epithelial cilia PIP2 dependent; rare in JBTS1)",
            "polydactyly_post_axial":  f"{_pct(n_poly)}% (lower — cilia FORM normally; partial Hedgehog preserved)",
            "obesity_morm_spectrum":   f"{_pct(n_obesity)}% (borderline allele carriers; full MORM if biallelic null)",
            "situs_inversus":          "<1% (INPP5E not required for nodal cilia motility)",
            "cilia_structure":         "NORMAL (nasal brushing: normal beat frequency — structural defect absent in JBTS1)",
            "no_mks_tier":             "Confirmed — PIP2 phospholipid regulation does not collapse TZ B9-complex structural gate",
            "morm_allelic":            "MORM (#610156) allelic via biallelic truncating null — same gene, different syndrome (no MTS in MORM)",
            "european_founder":        "Arg435Gln (c.1304G>A) — INPP5 core; European founder; most common JBTS1 allele worldwide",
            "jbts1_frequency":         "~2–3% of all Joubert syndrome cases; ~1:500,000–1,000,000 worldwide",
        },
    }
