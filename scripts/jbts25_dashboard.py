"""
CEP104 Joubert Syndrome Type 25 (JBTS25) — Autosomal Recessive / CEP104 / Ciliary Tip TOG Scaffold / TTBK2 Co-Scaffold / No MKS Tier
======================================================================================================================================
Primary Gene : CEP104 (*616078) — Centrosomal Protein 104kDa (also KIAA0562, FAP256 orthologue) — 1p36.32; ~1338 aa;
               centriolar satellite protein with a TOG (Tumor OVerexpressed Gene) microtubule-binding domain;
               ciliary tip scaffold that couples tubulin delivery to axonemal elongation checkpointing.

               CEP104 protein domain architecture (~1338 aa):
               - N-terminal TOG domain (aa 1–280): Tumor OVerexpressed Gene domain; HEAT-repeat β-propeller fold;
                 binds free αβ-tubulin dimers non-polymerised; delivers tubulin dimers to growing axonemal plus-ends
                 at the ciliary tip; structurally homologous to XMAP215/ch-TOG N-terminal TOG1.
                 TOG inner surface (aa 1–140): tubulin-contacting HEAT repeats; αβ-tubulin β-subunit contact patch;
                 pathogenic missense hot-zone (Gly178Asp, Arg210His in HEAT-HEAT linker).
                 TOG outer surface (aa 141–280): CLUAP1 co-binding interface; IFT-B1 scaffold docking;
                 couples tubulin delivery to IFT-B1 anterograde transport at the ciliary tip.
               - Central linker / TTBK2-scaffold (aa 281–600): structurally flexible linker connecting TOG to CC;
                 TTBK2 (Tau Tubulin Kinase 2) interaction motifs (aa 310–380 and aa 490–555);
                 CEP104 acts as a TTBK2 co-scaffold at the ciliary tip, sustaining kinase activity for axonemal
                 elongation checkpointing; TTBK2 is recruited from the transition fiber (via CEP164/NPHP15) and
                 phosphorylates MPP9 (CP110 cap lock) and KIF2A (depolymerising kinesin) at the ciliary tip.
                 CEP104 central linker LOF → TTBK2 de-stabilised at the tip → MPP9 hypophosphorylation →
                 CP110 cap re-engagement → cilia tip retraction → shortened primary cilia.
               - Coiled-coil domain (aa 601–950): homo-oligomerisation; anchors CEP104 dimers to the distal
                 end of central pair microtubules; ciliary tip identity signal; FOP/FGFR1OP interaction surface
                 for centriolar satellite tethering; centriolar satellite CEP104 pool required for pre-assembly.
                 CC N-zone (aa 601–750): primary homo-dimerisation; disruption → monomeric CEP104 → loss of
                 ciliary tip avidity; Thr544Met MENA founder maps here (aa 544, central linker-CC junction).
                 CC C-zone (aa 751–950): satellite-to-cilia transfer signal; Leu680Pro severe misfolding allele
                 (aa 680 within CC N-zone; numbering based on full protein, Pro helix-breaker).
               - C-terminal extension (aa 951–1338): CLUAP1/Qilin-interaction module; couples CEP104 to IFT-B1
                 retrograde return from the ciliary tip; distal axonemal cap stabilisation checkpoint;
                 C-terminal LOF (Glu862Ter and beyond) → loss of IFT-B1 tip coupling → retrograde retrieval
                 failure → tip tubulin pool depletion → shortened, unstable cilia.
                 CTE inner (aa 951–1150): IFT-B1 interface (CLUAP1/IFT88/IFT52); Arg1080Trp South Asian cluster.
                 CTE outer (aa 1151–1338): microtubule polymerisation checkpoint; distal cap stabilisation.

               CEP104 LOF pathway (JBTS25):
               CEP104 biallelic LOF → (1) defective TOG-mediated tubulin dimer delivery to ciliary tip axoneme AND
               (2) TTBK2 co-scaffold failure → MPP9 hypophosphorylation → reduced CP110 cap removal speed →
               combined tip polymerisation deficit + CP110 re-engagement → primary cilia SHORTENED (not absent) →
               Hedgehog/SMO ciliary trafficking partially impaired → cerebellar vermis hypoplasia → Molar Tooth Sign.
               Renal tubular cilia shortened → tubulo-interstitial stress → NPHP-like nephropathy (~18%).
               Retinal connecting cilia partially affected → rod-cone dystrophy (~28%).
               Cilia FORM but are SHORT — morphologically distinct from cilia-absent (CEP83-JBTS22, distal appendage
               foundation absent) and cilia-ultrastructurally normal (ZNF423-JBTS24, transcriptional mechanism).

⚠ CILIARY TIP SCAFFOLD MECHANISM — UNIQUE IN JBTS SPECTRUM:
   CEP104 is the ONLY confirmed JBTS gene encoding a TOG microtubule-binding scaffold specifically at the
   CILIARY TIP (distal axonemal plus-end). This is distinct from:
   - IFT-A (retrograde motor loading, NPHP12/13): transport, not tip scaffolding
   - IFT-B (anterograde cargo delivery, NPHP19/IFT81): cargo, not tip polymerisation
   - TZ structural (B9D1/JBTS19, TMEM216/JBTS2): transition zone gate, not tip
   - Centriolar distal appendage (CEP83/JBTS22, CEP164/NPHP15): ciliogenesis initiation, not tip elongation
   CEP104 acts AFTER ciliation is initiated — it controls CILIARY LENGTH and TIP STABILITY.
   Primary cilia in CEP104 LOF: FORM normally, ARE SHORT (~30–50% of wild-type length in fibroblasts).
   Nasal brushing PCD studies: cilia present, beat frequency reduced (tip instability), normal ultrastructure.

⚠ TTBK2 — CEP104 AXIS (NOT ALLELIC — DIFFERENT DISEASE):
   CEP104 and TTBK2 (Tau Tubulin Kinase 2) are functionally coupled at the ciliary tip. TTBK2 biallelic LOF
   → Spinocerebellar Ataxia Type 11 (SCA11, OMIM #604432): autosomal dominant gain-of-function in non-ciliary
   TTBK2 function (microtubule dynamics, neurodegeneration). CEP104 LOF → JBTS25 (ciliary tip scaffolding).
   These are NOT allelic — different genes, different modes of inheritance, different diseases.
   CLINICAL NOTE: TTBK2 dominant-negative alleles mimic some JBTS25 features in the cerebellar phenotype; WES
   mandatory to distinguish — gene panel must include both CEP104 and TTBK2 for cerebellar ataxia DDx.

⚠ CEP104 — CLUAP1 / IFT-B1 AXIS:
   CEP104 C-terminal extension interfaces with CLUAP1 (IFT38, IFT-B1 core subunit). CEP104 LOF → partial IFT-B1
   tip coupling failure → IFT-B1 retrograde dissociation at the tip impaired → IFT particle accumulation at the
   tip (unlike IFT-A LOF which causes tip accumulation via retrograde failure). CEP104 patients may show enlarged
   ciliary tips on EM — a radiological/EM diagnostic clue. CEP104 and CLUAP1 share an interaction surface; CLUAP1
   biallelic LOF → JBTS9/CC2D2A modifier phenotype (not allelic with JBTS25, but synergistic in compound families).

⚠ NO MKS TIER — CEP104 IS NOT A TZ GATE/B9-COMPLEX PROTEIN:
   CEP104 localises to the CILIARY TIP (distal axonemal plus-end) and centriolar satellites. It does NOT:
   - Contribute to the transition zone (TZ) B9-domain complex (B9D1/JBTS19, B9D2/JBTS34)
   - Form part of the tectonic module (TCTN1/JBTS11, TCTN2/JBTS13, TCTN3/JBTS18)
   - Localise to distal appendages (CEP83/JBTS22, CEP164/NPHP15)
   Biallelic null CEP104 → JBTS25, live birth, no perinatal lethal/Meckel phenotype.
   MKS tier requires TZ structural collapse (B9-complex or TMEM-module failure). CEP104 biallelic null:
   TZ gate B9-complex INTACT → diffusion barrier functional → selective lipid composition preserved.

Disease OMIM : #616778 — Joubert Syndrome Type 25 (JBTS25)
               Gene OMIM: *616078 (CEP104 / KIAA0562)
               No known allelic syndrome with different phenotype from allele-class threshold
Chromosome   : 1p36.32
Inheritance  : Autosomal recessive — biallelic LOF (truncating or damaging missense)
               NO MKS lethal tier — ciliary tip scaffold; TZ gate B9-complex remains intact
Cohort size  : 40-patient educational cohort (seed 463) — JBTS25 (MTS-confirmed)
"""

import random

SEED = 463
N    = 40   # 40-patient JBTS25 educational cohort (MTS-confirmed)

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
    ('Middle Eastern / MENA (consanguineous)',    0.30),  # Thr544Met MENA founder elevated
    ('European (non-consanguineous)',             0.25),  # Arg244Cys European cluster
    ('South Asian (consanguineous)',              0.22),  # Arg1080Trp South Asian cluster
    ('North African (consanguineous)',            0.12),  # Gly178Asp North African cluster
    ('East Asian',                               0.07),
    ('Other / Unknown',                          0.04),
]

# Allele classes (truncating LOF + missense)
allele_classes = [
    ('Biallelic Loss-of-Function (frameshift/stop)',     0.38),  # truncating nulls; most severe
    ('Missense / Splice Compound Heterozygous',         0.30),  # compound; moderate-severe
    ('Biallelic Damaging Missense',                    0.22),  # biallelic missense; moderate
    ('Missense / Near-null Compound',                  0.10),  # near-null rescued to JBTS25
]

# CEP104 variants (pathogenic / likely pathogenic — JBTS25-specific alleles)
variants = [
    'Thr544Met/Thr544Met',            # CC N-zone entry; MENA homozygous; moderate
    'Thr544Met/Arg244Cys',            # MENA founder + European linker; moderate-severe
    'Thr544Met/c.1462+1G>A',          # MENA founder + splice; moderate-severe
    'Arg244Cys/Arg244Cys',            # European linker homozygous; moderate-severe
    'Arg244Cys/Glu862Ter',            # European + truncating; compound
    'Gly178Asp/Gly178Asp',           # North African TOG homozygous; moderate
    'Gly178Asp/Thr544Met',            # North African + MENA; moderate
    'Arg1080Trp/Arg1080Trp',         # South Asian CTE homozygous; moderate-severe
    'Arg1080Trp/Thr544Met',          # South Asian + MENA; moderate-severe
    'Leu680Pro/Thr544Met',           # Severe misfolding + MENA founder; severe
    'Ala298Val/Arg244Cys',           # Linker mild + European; moderate
    'Ala298Val/Ala298Val',           # Central linker biallelic mild; mild-moderate
    'Glu862Ter/Arg244Cys',           # Truncating + European linker; compound
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
    oma       = _rng_p.random() < 0.52
    breath    = _rng_p.random() < 0.55
    retinal   = _rng_p.random() < 0.28   # rod-cone; shorter connecting cilia
    renal     = _rng_p.random() < 0.18   # NPHP-like; shortened tubular cilia
    hepatic   = _rng_p.random() < 0.08
    poly      = _rng_p.random() < 0.15   # post-axial; Hedgehog digit patterning impaired
    id_flag   = _rng_p.random() < 0.72
    esrd      = _rng_p.random() < 0.06   # lower than high-renal JBTS subtypes (CEP83, JBTS22)
    situs     = False                    # CEP104 not required for nodal cilia motility

    patients.append({
        'id':           f'JBTS25-{i+1:03d}',
        'age':          age,
        'sex':          sex,
        'ethnicity':    eth,
        'allele_class': ac,
        'variant':      var,
        'mts':          True,   # MTS confirmed — JBTS25 diagnostic criterion (100%)
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

_eth_counts = {}
for p in patients:
    _eth_counts[p['ethnicity']] = _eth_counts.get(p['ethnicity'], 0) + 1

_ac_counts = {}
for p in patients:
    _ac_counts[p['allele_class']] = _ac_counts.get(p['allele_class'], 0) + 1


# ── API functions ─────────────────────────────────────────────────────────────
def get_overview():
    return {
        "disease_id": "jbts25",

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
            "ciliary_tip_mechanism": (
                "CILIARY TIP TOG SCAFFOLD — UNIQUE CEP104 MECHANISM: CEP104 is the ONLY confirmed JBTS gene "
                "encoding a TOG microtubule-binding domain scaffold at the ciliary distal tip. CEP104 LOF → "
                "defective tubulin delivery to axonemal plus-ends (TOG domain failure) AND TTBK2 co-scaffold "
                "instability at the tip → cilia form but are SHORT (~30–50% WT length) — NOT absent. "
                "Nasal brushing PCD: cilia present, beat frequency reduced (tip instability), ultrastructure normal. "
                "Do NOT exclude JBTS25 on basis of ciliary presence — tip length assay or IFT-B tip accumulation EM required."
            ),
            "ttbk2_distinction": (
                "TTBK2 — CEP104 AXIS (NOT ALLELIC): CEP104 and TTBK2 are functionally coupled at the ciliary "
                "tip but are NOT allelic. TTBK2 dominant LOF → Spinocerebellar Ataxia Type 11 (SCA11, #604432): "
                "autosomal DOMINANT, non-ciliary neurodegeneration. CEP104 biallelic LOF → JBTS25: autosomal "
                "RECESSIVE, ciliopathy. Both present with cerebellar ataxia — WES mandatory to distinguish. "
                "Gene panel must include CEP104 and TTBK2 for cerebellar ataxia differential diagnosis."
            ),
            "cluap1_ift_b1_axis": (
                "CEP104 — CLUAP1/IFT-B1 COUPLING: CEP104 C-terminal extension interfaces with CLUAP1 (IFT38, "
                "IFT-B1 core). CEP104 LOF → IFT-B1 tip coupling failure → IFT particle accumulation at ciliary "
                "tip (opposite to IFT-A retrograde failure). Enlarged ciliary tips on EM: diagnostic clue in "
                "JBTS25. CEP104 and CLUAP1 share an interaction surface — CLUAP1 co-immunoprecipitation confirms "
                "CEP104 pathogenicity for VUS interpretation. Panel mandatory with IFT81/NPHP19 (IFT-B1 core)."
            ),
            "mena_founder": (
                "MENA FOUNDER — Thr544Met (c.1631C>T) CC N-zone entry: most prevalent JBTS25 allele in "
                "consanguineous Middle Eastern/MENA families. Disrupts the coiled-coil homo-oligomerisation "
                "interface at the CC N-zone entry; monomeric CEP104 loses ciliary tip avidity. Moderate JBTS25: "
                "MTS confirmed, cerebellar ataxia, ~25% renal. Carrier screening recommended in MENA populations."
            ),
        },

        "key_facts": [
            "CEP104 (~1338 aa) — TOG microtubule-binding scaffold at ciliary tip; 1p36.32; OMIM *616078",
            "Unique mechanism: TOG domain delivers tubulin dimers to axonemal plus-end AND TTBK2 co-scaffold",
            "Cilia FORM but are SHORT (~30–50% WT length) — NOT absent; tip length assay diagnostic",
            "IFT particle accumulation at ciliary tip (IFT-B1 retrograde coupling failure) — EM diagnostic clue",
            "No MKS tier — ciliary tip scaffold; TZ B9-complex gate INTACT in CEP104 biallelic null",
            "No known allelic syndrome (no allele-class threshold to distinct disease, unlike ZNF423-JBTS24/NPHP10)",
            "MENA founder: Thr544Met (c.1631C>T) — CC N-zone entry; homo-oligomerisation failure",
            "European cluster: Arg244Cys (c.730C>T) — central linker / TTBK2-scaffold interface",
            "Retinal rod-cone ~28% — connecting cilia shortened; annual ERG from age 3",
            "Renal ~18% NPHP-like — tubular cilia shortened; annual US + creatinine protocol",
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
        "disease_id": "jbts25",

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
        },

        "notable_variants": [
            {
                "name":       "Thr544Met",
                "cdna":       "c.1631C>T",
                "domain":     "CC N-zone entry / coiled-coil homo-oligomerisation interface (aa 544)",
                "population": "Middle Eastern / MENA (consanguineous) — regional founder allele",
                "severity":   "Moderate",
                "mechanism":  "Thr-to-Met substitution at the CC N-zone entry residue disrupts the hydrophilic surface required for CEP104 homo-dimerisation. Monomeric CEP104 loses ciliary tip avidity — cannot form the stable dimer platform needed for TTBK2 co-scaffolding at the axonemal plus-end. Residual partial TTBK2 recruitment via IFT-B1 scaffold (~30–40% WT). Cilia shortened ~40–55% WT length. Moderate JBTS25: MTS confirmed, cerebellar ataxia, ~25% renal penetrance. Homozygous Thr544Met → moderate phenotype; compound with truncating or Arg244Cys → moderate-severe.",
            },
            {
                "name":       "Arg244Cys",
                "cdna":       "c.730C>T",
                "domain":     "Central linker / TTBK2-scaffold interface (aa 244) — TTBK2 co-scaffold motif 1",
                "population": "European (non-consanguineous)",
                "severity":   "Moderate–Severe",
                "mechanism":  "Arg-to-Cys substitution introduces a free thiol in the first TTBK2 co-scaffold interaction motif (aa 310–380 region proximity; aa 244 is in the N-linker region upstream). The thiol can form aberrant disulfide bonds, locking the linker in a non-productive conformation that sterically occludes TTBK2 binding. TTBK2 tip recruitment reduced ~55% WT → MPP9 phosphorylation failure → CP110 cap re-engagement → cilia retraction. More severe tip phenotype than Thr544Met. Moderate-severe JBTS25: MTS, cerebellar ataxia, ~35% renal, ~32% retinal. European non-consanguineous families carry one copy; second allele often splice or frameshift.",
            },
            {
                "name":       "Gly178Asp",
                "cdna":       "c.533G>A",
                "domain":     "N-terminal TOG domain HEAT repeat inner surface (aa 178) — αβ-tubulin β-subunit contact patch",
                "population": "North African (consanguineous)",
                "severity":   "Moderate",
                "mechanism":  "Gly-to-Asp substitution introduces a bulky charged residue in the flexible HEAT repeat inner surface glycine loop required for tubulin β-subunit accommodation in the TOG binding groove. αβ-tubulin dimer binding affinity reduced to ~25–35% WT — partial tubulin delivery to the ciliary tip preserved. Cilia shortened ~30–45% WT length. Moderate JBTS25: MTS confirmed, cerebellar ataxia, ~22% renal, ~25% retinal. Homozygous Gly178Asp in North African consanguineous families → moderate, good early motor outcome in ~25% of cases. Include CEP104 on North African JBTS panels.",
            },
            {
                "name":       "Arg1080Trp",
                "cdna":       "c.3238C>T",
                "domain":     "C-terminal extension / CLUAP1-interaction module (aa 1080) — IFT-B1 tip coupling surface",
                "population": "South Asian (consanguineous)",
                "severity":   "Moderate–Severe",
                "mechanism":  "Arg-to-Trp substitution at the CLUAP1 binding surface disrupts the basic patch required for CLUAP1 acidic patch interaction (IFT-B1 core coupling). CEP104 cannot stably interact with IFT-B1 at the ciliary tip — IFT-B1 retrograde dissociation impaired → IFT particle accumulation at ciliary tip on EM (diagnostic clue). TTBK2 co-scaffolding partially intact (linker unaffected), but IFT-B1 tip coupling failure → tubulin delivery to tip reduced secondary to IFT-B1 stalling. Moderate-severe JBTS25: cerebellar ataxia, ~38% renal, ~30% retinal. South Asian mandatory panel inclusion.",
            },
            {
                "name":       "Glu862Ter",
                "cdna":       "c.2584G>T",
                "domain":     "CC C-zone truncating null (aa 862) — removes entire C-terminal extension",
                "population": "Pan-ethnic",
                "severity":   "Severe",
                "mechanism":  "Truncating stop at aa 862 (mid-CC C-zone) removes the entire C-terminal extension (aa 863–1338), eliminating the CLUAP1/IFT-B1 coupling module entirely. CEP104 protein truncated to TOG + linker + CC N-zone only. Loss of IFT-B1 tip coupling → severe tip tubulin pool depletion → cilia shortened ~55–65% WT length (most severe among JBTS25 alleles). Biallelic Glu862Ter → most severe JBTS25: profoundly truncated cerebellar vermis, early ESRD (~age 12–16 when renal affected), severe intellectual disability. Biallelic Glu862Ter homozygous is phenotypically consistent with JBTS25 live birth (not perinatal lethal, TZ gate INTACT).",
            },
            {
                "name":       "Leu680Pro",
                "cdna":       "c.2039T>C",
                "domain":     "CC N-zone / homo-oligomerisation helix (aa 680) — domain misfolding",
                "population": "Pan-ethnic",
                "severity":   "Severe (domain misfolding)",
                "mechanism":  "Pro substitution in the CC N-zone primary helix introduces a rigid helix-breaker disrupting CC homo-dimerisation and causing global misfolding of the CC domain. CEP104 protein aggregates in the cytosol — accelerated proteasomal degradation of the CC-CTE fragments. Functional null at the ciliary tip for both TTBK2 scaffolding and IFT-B1 coupling. Compound Leu680Pro/Thr544Met (founder) → severe JBTS25: early ESRD (~age 14–16 when renal affected), retinal rod-cone severe onset, profoundly shortened cilia on EM. Never observed homozygous in the literature (potentially selected against in consanguineous populations).",
            },
            {
                "name":       "c.1462+1G>A",
                "cdna":       "c.1462+1G>A",
                "domain":     "Splice donor — CC N-zone / C-zone junction exon (intron 14)",
                "population": "European (non-consanguineous)",
                "severity":   "Moderate–Severe",
                "mechanism":  "G>A transversion at the canonical splice donor site of intron 14 (CC N-zone/C-zone junction exon) → cryptic splice activation → 48-aa deletion within the CC domain (Δaa 450–497 from mature mRNA). Deletion disrupts CC N-zone homo-dimerisation and N-zone/C-zone transition, preventing ciliary tip targeting. CEP104 truncated variant mis-localises to centriolar satellites but cannot transfer to the ciliary tip. TTBK2 tip recruitment severely impaired. Compound Thr544Met/c.1462+1G>A → moderate-severe JBTS25. European compound-het families: include intron 14 splice on sequencing panels.",
            },
            {
                "name":       "Ala298Val",
                "cdna":       "c.893C>T",
                "domain":     "Central linker (aa 298) — mild TTBK2-scaffold impairment",
                "population": "Pan-ethnic",
                "severity":   "Mild–Moderate",
                "mechanism":  "Conservative Ala-to-Val substitution in the central linker (upstream of the primary TTBK2 co-scaffold motif 1). Val side-chain bulk slightly restricts linker flexibility; TTBK2 binding affinity reduced ~15–20% WT. Cilia shortened ~15–25% WT length — milder tip phenotype. Mild-moderate JBTS25: MTS present but may be subtle on MRI, cerebellar ataxia mild, renal ~12%, retinal ~18%. Biallelic Ala298Val → mild JBTS25 with better neurodevelopmental outcome. Compound Ala298Val/Arg244Cys → moderate JBTS25.",
            },
        ],
    }


def get_definitions():
    return {
        "disease_id":    "jbts25",
        "gene_full_name":"CEP104 — Centrosomal Protein 104kDa; Ciliary Tip TOG Microtubule-Binding Scaffold; TTBK2 Co-Scaffold; IFT-B1/CLUAP1 Tip Coupler; No MKS Tier; MENA Founder Thr544Met; 1p36.32",
        "omim_gene":     "616078",
        "omim_jbts25":   "616778",
        "chromosome":    "1p36.32",
        "protein_size":  (
            "~1338 aa — N-terminal TOG domain (aa 1–280) | Central linker / TTBK2-scaffold (aa 281–600) | "
            "Coiled-coil domain (aa 601–950) | C-terminal extension / CLUAP1 module (aa 951–1338)"
        ),
        "inheritance":   "Autosomal recessive — biallelic LOF (truncating null or damaging missense)",
        "mks_tier":      False,
        "mechanism_class": "Ciliary Tip Scaffold / TOG Microtubule Polymerisation",
        "mechanism_detail": (
            "CEP104 is a centriolar satellite → ciliary tip scaffold containing a TOG (Tumor OVerexpressed Gene) "
            "microtubule-binding domain. TOG domain binds free αβ-tubulin dimers and delivers them to axonemal "
            "plus-ends for ciliary elongation. Central linker acts as a TTBK2 co-scaffold at the ciliary tip "
            "(TTBK2 phosphorylates MPP9 CP110-lock and KIF2A depolymerising kinesin). C-terminal extension couples "
            "CEP104 to IFT-B1 core via CLUAP1. CEP104 LOF → cilia FORM but are SHORT (~30–50% WT length). "
            "IFT-B1 tip accumulation on EM (diagnostic). TZ B9-complex gate INTACT."
        ),
        "cilia_phenotype": "Shortened (NOT absent) — cilia form normally, axonemal elongation impaired; ~30–50% WT length",
        "hedgehog_impact": "Partial SMO/Hedgehog impairment (shorter cilia reduce but do not abolish SMO trafficking)",
        "mts_mechanism":  (
            "Shortened cilia → partial SMO exclusion → reduced Hedgehog transduction → cerebellar granule cell "
            "precursor proliferation partially impaired → cerebellar vermis hypoplasia → Molar Tooth Sign"
        ),
        "allelic_diseases": [],
        "key_ddx": [
            "JBTS22 CEP83: cilia ABSENT (DA foundation absent) vs JBTS25 CEP104: cilia SHORT (tip scaffold) — "
            "nasal brushing distinguishes (absent vs short beats); EM shows enlarged tips in CEP104",
            "JBTS24 ZNF423: cilia NORMAL (transcriptional) vs JBTS25 CEP104: cilia SHORT (tip scaffold) — "
            "PCD beat frequency reduced in CEP104, normal in ZNF423; WES mandatory",
            "JBTS19 B9D1: MKS-tier (TZ gate collapsed, perinatal lethal null) vs JBTS25 CEP104: no MKS tier "
            "(ciliary tip, TZ gate intact) — allele-class threshold different; CEP104 biallelic null = live birth",
            "SCA11 TTBK2: dominant cerebellar ataxia (neurodegeneration) vs JBTS25 CEP104: recessive ciliopathy — "
            "MTS absent in SCA11; autosomal dominant vs recessive; WES + brain MRI mandatory",
        ],
        "surveillance_protocol": {
            "renal":    "Annual renal US + creatinine/eGFR from diagnosis; ESRD median age ~22–28yr when renal affected",
            "retinal":  "ERG + fundus photography from age 3; annual review; rod-cone dystrophy progressive",
            "hepatic":  "LFTs + hepatic US at diagnosis; repeat 2-yearly if abnormal",
            "neuro":    "Brain MRI at diagnosis (MTS confirmation); annual physiotherapy assessment; cochlear screen",
            "cilia":    "Nasal brushing videomicroscopy + EM at diagnosis (tip enlargement, short cilia, reduced beat frequency)",
        },
        "treatment": {
            "renal":    "Renal transplant curative — no recurrence (cell-autonomous defect, transplanted kidney has WT CEP104)",
            "retinal":  "No established treatment; gene therapy trials recruiting; retinal degeneration monitoring",
            "general":  "Physiotherapy, occupational therapy, speech therapy; seizure management if epilepsy develops",
            "ttbk2_note": "TTBK2 activators under investigation (pre-clinical); CEP104 tip scaffold restoration via gene therapy is the target",
        },
        "frequency":     "~1–3% of all molecularly confirmed Joubert syndrome; ~1 in 500,000–1,000,000 (JBTS25-specific)",
        "founder_variants": [
            {
                "variant":      "Thr544Met (c.1631C>T)",
                "population":   "Middle Eastern / MENA (consanguineous)",
                "frequency":    "Most prevalent JBTS25 allele in MENA families",
                "domain":       "CC N-zone entry / homo-oligomerisation interface",
                "severity":     "Moderate",
            },
            {
                "variant":      "Arg244Cys (c.730C>T)",
                "population":   "European (non-consanguineous)",
                "frequency":    "European cluster; commonly compound heterozygous",
                "domain":       "Central linker / TTBK2-scaffold",
                "severity":     "Moderate–Severe",
            },
            {
                "variant":      "Gly178Asp (c.533G>A)",
                "population":   "North African (consanguineous)",
                "frequency":    "North African regional cluster",
                "domain":       "TOG HEAT repeat — tubulin contact patch",
                "severity":     "Moderate",
            },
            {
                "variant":      "Arg1080Trp (c.3238C>T)",
                "population":   "South Asian (consanguineous)",
                "frequency":    "South Asian cluster",
                "domain":       "C-terminal extension / CLUAP1 surface",
                "severity":     "Moderate–Severe",
            },
        ],
    }
