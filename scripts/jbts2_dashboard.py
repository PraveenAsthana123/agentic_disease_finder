"""
TMEM216 Joubert Syndrome Type 2 (JBTS2) — Autosomal Recessive / TMEM216 / 4-Pass TZ Membrane Scaffold / Y-Link TZ Gate / MKS2-Allelic / MKS Tier
=====================================================================================================================================================
Primary Gene : TMEM216 (*613277) — Transmembrane Protein 216 — 11q13.2; 148 aa;
               4-pass transmembrane protein anchored in the transition zone (TZ) membrane.

               TMEM216 protein domain architecture (148 aa):
               - N-terminal cytoplasmic tail (aa 1–15): RPGRIP1L interaction; TZ matrix anchoring;
                 connects to NPHP-module Y-link scaffold.
               - TM1 (aa 16–38): first transmembrane helix; spans TZ membrane bilayer.
               - EL1 (extracellular loop 1, aa 39–45): periaxonemal space; TMEM67/meckelin docking
                 surface; MKS-module scaffold contact.
               - TM2 (aa 46–70): second transmembrane helix; coaxial with TM1 in TZ membrane.
               - ICL1 (intracellular loop 1, aa 71–78): cytoplasmic; anchors to NPHP4-B9D1
                 interaction network; Arg73 (founder allele site) sits here.
               - TM3 (aa 79–101): third transmembrane helix.
               - EL2 (extracellular loop 2, aa 102–110): tectonic complex (TCTN1-3) contact; outer
                 TZ gate periaxonemal scaffold.
               - TM4 (aa 111–134): fourth transmembrane helix; MKS1-RPGRIP1L docking surface in
                 cytoplasmic leaflet.
               - C-terminal cytoplasmic tail (aa 135–148): NPHP4-IFT-A contact platform.

               TMEM216 LOF pathway:
               TMEM216 loss → Y-link TZ scaffold destabilised → TZ gate diffusion barrier leaky →
               SMO fails to enter cilia (despite leaky gate, bulk protein diffusion non-selective) →
               Hedgehog/SHH pathway failure → Molar Tooth Sign (MTS); cerebellar vermis hypoplasia.
               Simultaneously: B9-complex (B9D1-B9D2-MKS1) gate co-destabilised → MKS grade
               phenotype in null alleles → Meckel syndrome (perinatal lethal, renal+CNS+liver).

⚠ MKS2-JBTS2 ALLELIC SPECTRUM — SAME GENE, DIFFERENT TIERS:
   TMEM216 is allelic with Meckel syndrome type 2 (MKS2, #603194). Allele class controls tier:
   - Biallelic null (truncating stop/frameshift) → MKS2 (Meckel-Gruber syndrome: perinatal
     lethal; large occipital encephalocele, bilateral polycystic kidneys, postaxial polydactyly,
     absent olfactory bulbs). MKS2 is universally lethal by neonatal period.
   - Biallelic hypomorphic missense (retains partial TZ membrane scaffold function) → JBTS2
     (Joubert syndrome type 2: live birth, MTS, cerebellar ataxia, neurodevelopmental disability).
   CLINICAL RULE: Biallelic truncating TMEM216 in a pregnancy → HIGH MKS2 risk. Brain MRI +
   prenatal ultrasound mandatory. Compound null/missense → JBTS2 phenotype more likely but
   variable (phenotype-genotype imperfect). TMEM216 is a MANDATORY MKS panel gene.

⚠ TZ GATE — DIFFUSION BARRIER MECHANISM (STRUCTURAL):
   TMEM216 maintains the transition zone (TZ) diffusion barrier — the molecular 'fence' between
   the ciliary membrane and the bulk plasma membrane. Unlike INPP5E (JBTS1, lipid signalling),
   TMEM216 is a structural TZ membrane scaffold. TZ gate failure → non-selective protein
   diffusion into cilium → signalling pathway dysfunction (SMO excluded by unknown compensatory
   mechanism; Hedgehog fails). Cilia FORM normally in JBTS2 (structural gate fails, not
   axoneme assembly). Nasal brushing: normal cilia structure, POSSIBLY reduced beat amplitude.

⚠ ASHKENAZI JEWISH FOUNDER — Arg73Leu (c.218G>T):
   Arg73Leu in ICL1 is the commonest JBTS2 allele worldwide. Carrier frequency ~1:92–100 in
   Ashkenazi Jewish populations (one of the highest carrier frequencies for any ciliopathy
   founder allele). Homozygous Arg73Leu → JBTS2 (moderate; live birth; MTS confirmed).
   Carrier frequency mandates TMEM216 on all Ashkenazi Jewish reproductive carrier panels.
   R73L preserves partial TZ membrane scaffold function → JBTS2 not MKS2.

Disease OMIM : #608091 — Joubert Syndrome Type 2 (JBTS2)
               Gene OMIM: *613277 (TMEM216)
               Allelic: #603194 — Meckel Syndrome Type 2 (MKS2) — biallelic null TMEM216
Chromosome   : 11q13.2
Inheritance  : Autosomal recessive — biallelic LOF (hypomorphic missense → JBTS2; null → MKS2)
               MKS TIER — biallelic truncating null → Meckel syndrome (perinatal lethal)
Cohort size  : 40-patient educational cohort (seed 457) — JBTS2 (MTS-confirmed / missense-dominant)
"""

import random

SEED = 457
N    = 40   # 40-patient JBTS2 educational cohort (MTS-confirmed / missense-predominant)

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
    ('Ashkenazi Jewish (non-consanguineous)',      0.30),  # Arg73Leu founder elevated
    ('Middle Eastern / MENA (consanguineous)',     0.25),  # Glu66Lys MENA variant
    ('South Asian (consanguineous)',               0.20),  # Arg123Gln South Asian
    ('North African (consanguineous)',             0.14),  # Tyr109Cys North African
    ('European (non-consanguineous)',              0.08),  # Tyr29Cys European
    ('East Asian',                                0.03),
]

# Allele classes (missense-dominant — nulls → MKS2, not JBTS2 if biallelic)
allele_classes = [
    ('Biallelic Hypomorphic Missense',             0.40),  # partial TZ scaffold; classic JBTS2
    ('Hypomorphic Missense / Splice Compound',     0.28),  # splice reduces expression; moderate-severe
    ('Hypomorphic Missense / Near-Null Compound',  0.20),  # near-null (frameshift) + missense; severe JBTS2
    ('Biallelic Moderate Missense',                0.12),  # two moderate missense; moderate-severe
]

# TMEM216 variants (ICL1, EL2, TM domains — JBTS2-specific hypomorphic alleles)
variants = [
    'Arg73Leu/Arg73Leu',                # ICL1; Ashkenazi Jewish founder; homozygous; moderate
    'Arg73Leu/Glu66Lys',               # Founder + MENA; moderate-severe
    'Arg73Leu/c.218+1G>A',             # Founder + splice; moderate-severe
    'Arg73Leu/Tyr109Cys',              # Founder + North African; moderate-severe
    'Glu66Lys/Glu66Lys',              # MENA homozygous; moderate
    'Tyr109Cys/Tyr109Cys',            # North African homozygous; moderate
    'Arg123Gln/Arg73Leu',             # South Asian + founder; moderate
    'Arg123Gln/Glu66Lys',             # South Asian + MENA; moderate-severe
    'Phe81Ser/Arg73Leu',              # TM3 core + founder; moderate-severe
    'Tyr29Cys/Arg73Leu',              # TM1 European + founder; moderate
    'Thr92Met/Arg73Gln',              # ICL1/TM3 East Asian + mild; mild-moderate
    'Arg65Ter/Arg73Leu',              # Truncating null + founder; compound: JBTS2 (not MKS2)
    'Tyr135Cys/Glu66Lys',            # C-tail + MENA; moderate-severe
    'Tyr109Cys/Arg123Gln',           # North African + South Asian; moderate
]

_rng_p = random.Random(SEED + 1)
for i in range(N):
    eth = _rng_p.choices([e[0] for e in ethnicities], weights=[e[1] for e in ethnicities])[0]
    ac  = _rng_p.choices([a[0] for a in allele_classes], weights=[a[1] for a in allele_classes])[0]
    var = _rng_p.choice(variants)
    age = _rng_p.randint(2, 42)
    sex = _rng_p.choice(['M', 'F'])

    ataxia    = _rng_p.random() < 0.88
    hypotonia = _rng_p.random() < 0.82
    oma       = _rng_p.random() < 0.55
    breath    = _rng_p.random() < 0.58
    retinal   = _rng_p.random() < 0.35   # TZ gate leakiness → photoreceptor connecting cilia affected
    renal     = _rng_p.random() < 0.28   # NPHP-like; higher than JBTS1 (TZ structural defect)
    hepatic   = _rng_p.random() < 0.20   # biliary epithelial cilia TZ gate failure; MKS overlap
    poly      = _rng_p.random() < 0.20   # post-axial; Gli2/3 Hedgehog failure
    id_flag   = _rng_p.random() < 0.72
    esrd      = _rng_p.random() < 0.08   # ESRD at study entry (median ~25yr when affected)
    situs     = _rng_p.random() < 0.02   # rare — TZ gate integrity not required for nodal motility

    mts_confirmed = True  # cohort is MTS-confirmed (excludes MKS2 nulls)

    patients.append({
        'id':           f'JBTS2-{i+1:03d}',
        'age':          age,
        'sex':          sex,
        'ethnicity':    eth,
        'allele_class': ac,
        'variant':      var,
        'ataxia':       ataxia,
        'hypotonia':    hypotonia,
        'oma':          oma,
        'breath':       breath,
        'retinal':      retinal,
        'renal':        renal,
        'hepatic':      hepatic,
        'poly':         poly,
        'id_flag':      id_flag,
        'esrd':         esrd,
        'situs':        situs,
        'mts':          mts_confirmed,
    })

# ── aggregate counts for endpoints ───────────────────────────────────────────
n_ataxia   = sum(1 for p in patients if p['ataxia'])
n_hypotonia= sum(1 for p in patients if p['hypotonia'])
n_oma      = sum(1 for p in patients if p['oma'])
n_breath   = sum(1 for p in patients if p['breath'])
n_retinal  = sum(1 for p in patients if p['retinal'])
n_renal    = sum(1 for p in patients if p['renal'])
n_hepatic  = sum(1 for p in patients if p['hepatic'])
n_poly     = sum(1 for p in patients if p['poly'])
n_id       = sum(1 for p in patients if p['id_flag'])
n_esrd     = sum(1 for p in patients if p['esrd'])
n_situs    = sum(1 for p in patients if p['situs'])
n_mts      = sum(1 for p in patients if p['mts'])

n_f        = sum(1 for p in patients if p['sex'] == 'F')
n_m        = N - n_f

age_list   = [p['age'] for p in patients]
age_mean   = round(sum(age_list) / len(age_list), 1)
age_min    = min(age_list)
age_max    = max(age_list)

eth_counts = {}
for p in patients:
    eth_counts[p['ethnicity']] = eth_counts.get(p['ethnicity'], 0) + 1

ac_counts = {}
for p in patients:
    ac_counts[p['allele_class']] = ac_counts.get(p['allele_class'], 0) + 1

var_counts = {}
for p in patients:
    v = p['variant'].split('/')[0]
    var_counts[v] = var_counts.get(v, 0) + 1


# ── public API ────────────────────────────────────────────────────────────────
def get_overview():
    return {
        "disease_id":    "jbts2",
        "gene":          "TMEM216",
        "omim_gene":     "613277",
        "omim_disease":  "608091",
        "omim_allelic":  "603194 (MKS2 — biallelic null)",
        "chromosome":    "11q13.2",
        "protein":       "148 aa — 4-pass TZ membrane scaffold; Y-link TZ gate; MKS2-allelic",
        "inheritance":   "Autosomal recessive — hypomorphic missense → JBTS2; null → MKS2 (lethal)",
        "mks_tier":      True,
        "cohort_n":      N,
        "cohort_seed":   SEED,
        "cohort_label":  "JBTS2 MTS-confirmed cohort (missense/hypomorphic alleles; MKS2 nulls excluded)",

        "kpis": {
            "n_patients":            N,
            "mts_confirmed":         n_mts,
            "pct_cerebellar_ataxia": _pct(n_ataxia),
            "pct_hypotonia":         _pct(n_hypotonia),
            "pct_oma":               _pct(n_oma),
            "pct_retinal":           _pct(n_retinal),
            "pct_renal":             _pct(n_renal),
            "pct_hepatic":           _pct(n_hepatic),
            "pct_poly":              _pct(n_poly),
            "age_mean":              age_mean,
            "age_range":             f"{age_min}–{age_max}",
            "sex_f":                 n_f,
            "sex_m":                 n_m,
        },

        "mks2_allelic_rule": (
            "TMEM216 biallelic null → MKS2 (Meckel syndrome type 2, #603194): perinatal lethal. "
            "Hypomorphic missense → JBTS2 (#608091): live birth, MTS. Allele class + prenatal "
            "ultrasound + MRI mandatory before counselling. Ashkenazi Jewish Arg73Leu founder "
            "allele (~1:92 carrier frequency) is hypomorphic → JBTS2, not MKS2."
        ),

        "ashkenazi_founder": (
            "Arg73Leu (c.218G>T, ICL1): ~1:92–100 Ashkenazi Jewish carrier frequency. "
            "Homozygous → moderate JBTS2 (live birth, MTS confirmed). Mandatory on Ashkenazi "
            "Jewish reproductive carrier screening panels."
        ),

        "frequency":   "~2–3% of all Joubert syndrome; ~1:500,000–1,000,000 worldwide",
        "tz_mechanism": (
            "TMEM216 is a structural TZ (transition zone) membrane scaffold. "
            "LOF destabilises the Y-link TZ gate diffusion barrier → non-selective ciliary "
            "membrane protein trafficking failure → SMO excluded → Hedgehog failure → MTS. "
            "Distinct from INPP5E (JBTS1, lipid signalling) — TMEM216 is structural, not enzymatic."
        ),
    }


def get_breakdown():
    return {
        "disease_id":    "jbts2",
        "cohort_n":      N,

        "ethnicity_distribution": [
            {"ethnicity": k, "n": v, "pct": _pct(v)} for k, v in sorted(eth_counts.items(), key=lambda x: -x[1])
        ],

        "allele_class_distribution": [
            {"class": k, "n": v, "pct": _pct(v)} for k, v in sorted(ac_counts.items(), key=lambda x: -x[1])
        ],

        "top_variants": [
            {"variant": k, "n": v, "pct": _pct(v)} for k, v in sorted(var_counts.items(), key=lambda x: -x[1])[:8]
        ],

        "phenotype_bars": [
            {"feature": "Cerebellar Ataxia",         "n": n_ataxia,   "pct": _pct(n_ataxia)},
            {"feature": "Neonatal Hypotonia",         "n": n_hypotonia,"pct": _pct(n_hypotonia)},
            {"feature": "Oculomotor Apraxia",         "n": n_oma,      "pct": _pct(n_oma)},
            {"feature": "Breathing Dysregulation",    "n": n_breath,   "pct": _pct(n_breath)},
            {"feature": "Intellectual Disability",    "n": n_id,       "pct": _pct(n_id)},
            {"feature": "Retinal Rod-Cone Dystrophy", "n": n_retinal,  "pct": _pct(n_retinal)},
            {"feature": "Renal NPHP-like",            "n": n_renal,    "pct": _pct(n_renal)},
            {"feature": "Hepatic CHF/DPM",            "n": n_hepatic,  "pct": _pct(n_hepatic)},
            {"feature": "Polydactyly (post-axial)",   "n": n_poly,     "pct": _pct(n_poly)},
            {"feature": "ESRD at Study Entry",        "n": n_esrd,     "pct": _pct(n_esrd)},
            {"feature": "Situs Inversus",             "n": n_situs,    "pct": _pct(n_situs)},
        ],

        "clinical_pearls": [
            {
                "title": "MKS2-JBTS2 Allelic Gate — Allele Class Determines Lethal vs Live-Birth Tier",
                "detail": (
                    "Biallelic truncating TMEM216 → MKS2 (perinatal lethal); biallelic hypomorphic missense → JBTS2 "
                    "(live birth). Arg65Ter/missense compound → JBTS2 (missense allele provides residual TZ scaffold). "
                    "Prenatal: occipital encephalocele + large polycystic kidneys on USS → MKS2; MTS on fetal MRI → JBTS2. "
                    "Both require urgent TMEM216 sequencing in the proband + parental carrier testing."
                ),
            },
            {
                "title": "Ashkenazi Jewish Arg73Leu: Carrier Frequency ~1:92 — Reproductive Panel Mandatory",
                "detail": (
                    "Arg73Leu (c.218G>T) at ICL1 is the commonest JBTS2 allele worldwide. Carrier frequency ~1:92–100 "
                    "in Ashkenazi Jewish populations — one of the highest ciliopathy founder allele carrier frequencies. "
                    "Homozygous → moderate JBTS2 (MTS, cerebellar ataxia, neurodevelopmental delay). TMEM216 must be "
                    "on all Ashkenazi Jewish carrier panels. Arg73Leu is hypomorphic — partial ICL1 scaffold preserved "
                    "→ JBTS2, not MKS2."
                ),
            },
            {
                "title": "Hepatic CHF ~20%: Higher Than Most JBTS Types (MKS Module Overlap)",
                "detail": (
                    "Hepatic involvement (congenital hepatic fibrosis / ductal plate malformation) affects ~20% of JBTS2 "
                    "patients — significantly higher than INPP5E/JBTS1 (~5%) or AHI1/JBTS3 (~8%). Mechanism: biliary "
                    "epithelial cilia require TMEM216-dependent TZ gate integrity for cholangiocyte polarity signalling. "
                    "MKS2-overlap module: TMEM216 is in the same Y-link complex as TMEM67/MKS3 (which causes COACH "
                    "syndrome with hepatic fibrosis). Annual LFTs + hepatic USS from diagnosis mandatory in JBTS2."
                ),
            },
            {
                "title": "Cilia Form Normally — TZ Gate Is Structural Not Axonemal (Different From JBTS22)",
                "detail": (
                    "JBTS2/TMEM216: cilia FORM normally — axoneme is intact. Nasal brushing: normal cilia morphology on TEM. "
                    "Contrast: JBTS22/CEP83 (cilia ABSENT — distal appendage block). JBTS21/CSPP1 (cilia SHORTENED — "
                    "axoneme scaffold failure). JBTS2 TZ gate failure causes signalling incompetence without ablating "
                    "ciliogenesis. Clinical: do NOT exclude JBTS2 based on normal nasal brushing TEM or normal ciliary "
                    "beat frequency. PCD studies are uninformative for TMEM216/TZ gate ciliopathies."
                ),
            },
            {
                "title": "DDx TMEM216 (JBTS2/MKS2) vs TMEM67 (JBTS6/MKS3/COACH): Same Y-Link Module",
                "detail": (
                    "TMEM216 (11q13.2) and TMEM67 (8q22.2) are in the same Y-link TZ membrane scaffold complex. "
                    "TMEM67/MKS3 causes JBTS6 (mild retinal, high hepatic), COACH syndrome (Cerebellar vermis hypoplasia, "
                    "Oligophrenia, Ataxia, Coloboma, Hepatic fibrosis), and MKS3. "
                    "DDx: COACH (hepatic fibrosis + coloboma, TMEM67) vs JBTS2 (higher OMA, no coloboma, TMEM216). "
                    "Both require pan-TZ multigene panel — single-gene Sanger cannot distinguish without full clinical."
                ),
            },
        ],

        "renal_hepatic_note": (
            "TMEM216 JBTS2: Renal NPHP-like ~28% (tubulointerstitial nephritis, concentrating defect, "
            "proteinuria; ESRD median ~25yr when affected). Hepatic CHF/DPM ~20% — among highest in JBTS "
            "spectrum (Y-link TZ module overlap with TMEM67/MKS3-COACH syndrome). Annual LFTs + renal "
            "protocol USS from diagnosis; nephrology + hepatology co-management mandatory if organ involved."
        ),
    }


def get_definitions():
    return {
        "disease_id":    "jbts2",
        "gene_full_name":"TMEM216 — Transmembrane Protein 216; 4-Pass TZ Membrane Scaffold; Y-Link TZ Gate; MKS2-Allelic; Ashkenazi Jewish Founder Arg73Leu; 11q13.2",
        "omim_gene":     "613277",
        "omim_jbts2":    "608091",
        "omim_mks2":     "603194",
        "chromosome":    "11q13.2",
        "protein_size":  (
            "~148 aa — 4-pass transmembrane protein; N-terminal cytoplasmic tail RPGRIP1L interaction (aa 1–15); "
            "TM1 (aa 16–38); EL1 TMEM67-docking (aa 39–45); TM2 (aa 46–70); ICL1 NPHP4-B9D1 contact / "
            "Arg73 founder site (aa 71–78); TM3 (aa 79–101); EL2 tectonic complex contact (aa 102–110); "
            "TM4 MKS1-RPGRIP1L docking (aa 111–134); C-tail NPHP4-IFT-A platform (aa 135–148)"
        ),
        "inheritance":   "Autosomal recessive — biallelic hypomorphic missense LOF → JBTS2; biallelic truncating null → MKS2 (Meckel syndrome, perinatal lethal); MKS TIER confirmed",

        "mks2_allelic_rule": (
            "TMEM216 biallelic truncating null alleles → MKS2 (Meckel syndrome type 2, #603194): "
            "perinatal lethal; large occipital encephalocele, bilateral polycystic kidneys, polydactyly, "
            "absent olfactory bulbs, hepatic ductal plate malformation. NO live birth expected in biallelic null. "
            "TMEM216 hypomorphic missense (partial TZ scaffold preserved) → JBTS2 (#608091): live birth, "
            "MTS, cerebellar ataxia, neurodevelopmental disability, variable renal/hepatic/retinal. "
            "Compound null/missense alleles: phenotype depends on residual scaffold function from missense allele. "
            "Brain MRI (MTS vs encephalocele), prenatal ultrasound, and allele class are MANDATORY before "
            "counselling. Never assign JBTS2 without excluding MKS2 prenatal risk."
        ),

        "glossary": [
            {
                "term": "TMEM216 (Transmembrane Protein 216)",
                "definition": (
                    "TMEM216 (gene; protein TMEM216; OMIM *613277). ~148 aa 4-pass transmembrane protein at 11q13.2. "
                    "Structural component of the transition zone (TZ) Y-link scaffold — the 'fence posts' that connect "
                    "the ciliary axoneme to the ciliary membrane at the base of the cilium. Maintains the TZ diffusion "
                    "barrier (molecular gate) that restricts non-ciliary proteins from entering the ciliary compartment. "
                    "Two allelic diseases: JBTS2 (hypomorphic missense, live birth, MTS) and MKS2 (#603194, biallelic "
                    "null, perinatal lethal, Meckel syndrome). 11q13.2 locus."
                ),
            },
            {
                "term": "Transition Zone (TZ) Y-link scaffold and diffusion barrier",
                "definition": (
                    "The transition zone (TZ) is the basal segment of the cilium immediately distal to the basal body. "
                    "TZ Y-links are protein bridges that connect the ciliary axoneme microtubules to the ciliary membrane "
                    "— appearing as 'Y'-shaped structures on cilium cross-section TEM. TMEM216 is embedded in the TZ "
                    "membrane, anchoring Y-links via interactions with RPGRIP1L (N-tail), NPHP4-B9D1-B9D2 (ICL1), "
                    "TMEM67/MKS3 (EL1), and TCTN1-3/tectonic complex (EL2). Intact TZ Y-links maintain the TZ "
                    "diffusion barrier — a size and lipid-based molecular 'gate' preventing non-ciliary proteins from "
                    "entering the ciliary compartment. TMEM216 LOF → Y-link gaps → diffusion barrier leaky → SMO "
                    "excluded → Hedgehog pathway fails."
                ),
            },
            {
                "term": "MKS2 allelic tier and prenatal risk",
                "definition": (
                    "Meckel syndrome type 2 (MKS2, #603194): lethal ciliopathy caused by biallelic null TMEM216. "
                    "Perinatal features: large occipital encephalocele (100%), bilateral enlarged polycystic kidneys "
                    "(100%), postaxial hexadactyly (~75%), absent olfactory bulbs, hepatic ductal plate malformation, "
                    "pulmonary hypoplasia. Universally lethal by neonatal period. Prenatal USS: enlarged echogenic "
                    "kidneys + posterior fossa abnormality at 11–14 weeks. Fetal MRI: encephalocele. Distinguish from "
                    "JBTS2: MTS (no encephalocele) = JBTS2; encephalocele (no MTS) = MKS2. Compound null/missense: "
                    "phenotype depends on missense allele's residual TZ scaffold function — cannot predict from "
                    "sequence alone without functional data."
                ),
            },
            {
                "term": "Ashkenazi Jewish founder allele Arg73Leu (c.218G>T)",
                "definition": (
                    "Arg73Leu at ICL1 (intracellular loop 1, aa 73). The commonest JBTS2 allele worldwide. Carrier "
                    "frequency ~1:92–100 in Ashkenazi Jewish populations — one of the highest carrier frequencies for "
                    "any autosomal recessive ciliopathy founder allele globally (comparable to Connexin-26 for DFNB1 "
                    "hearing loss in some populations). Homozygous Arg73Leu → moderate JBTS2: live birth, MTS "
                    "confirmed, cerebellar ataxia ~88%, neurodevelopmental disability, retinal ~35%, renal ~28%, "
                    "hepatic ~20%. Arg73Leu is hypomorphic — ICL1 salt bridge disrupted but partial Y-link scaffold "
                    "contacts preserved → JBTS2 not MKS2. Mandatory on all Ashkenazi Jewish reproductive panels. "
                    "Carrier-carrier couples: 1:4 risk of homozygous offspring → JBTS2 (not MKS2 if Arg73Leu/Arg73Leu)."
                ),
            },
            {
                "term": "Hepatic CHF and ductal plate malformation in JBTS2",
                "definition": (
                    "Congenital hepatic fibrosis (CHF) and ductal plate malformation (DPM) occur in ~20% of JBTS2 "
                    "patients — among the highest hepatic penetrance in the JBTS spectrum. Mechanism: biliary "
                    "epithelial cells (cholangiocytes) require intact TZ Y-link scaffold (TMEM216-TMEM67 complex) "
                    "for polarised cilia signalling that controls ductal plate remodelling during fetal liver "
                    "development. TMEM216 LOF → cholangiocyte cilia signalling failure → ductal plate fails to "
                    "remodel → CHF + DPM. This mechanism overlaps with MKS3/TMEM67 (COACH syndrome) because both "
                    "proteins are in the same Y-link complex. CHF is progressive — annual LFTs, GGT, ultrasound "
                    "with portal Doppler. Varices / portal hypertension require gastroenterology co-management."
                ),
            },
            {
                "term": "DDx TMEM216 (JBTS2) vs TMEM67 (JBTS6) vs RPGRIP1L (JBTS7): Y-Link Module",
                "definition": (
                    "TMEM216 (11q13.2), TMEM67/MKS3 (8q22.2), and RPGRIP1L/NPHP8 (16q12.2) all encode proteins in "
                    "the same Y-link TZ module. Each has a different allelic disease spectrum: TMEM216 → JBTS2/MKS2; "
                    "TMEM67 → JBTS6/MKS3/COACH syndrome; RPGRIP1L → JBTS7/NPHP8/MKS5/COACH. "
                    "Clinical DDx: COACH syndrome (TMEM67 or RPGRIP1L) has coloboma + high hepatic; "
                    "JBTS2 has no coloboma, high hepatic but less than COACH; JBTS6 has higher retinal and "
                    "lower renal than JBTS2. Phenotype overlap mandates pan-TZ multigene panel. "
                    "Single-gene testing of any one Y-link gene is insufficient for JBTS diagnosis."
                ),
            },
            {
                "term": "No situs inversus (<2%) in JBTS2 — TZ gate integrity ≠ nodal cilia motility",
                "definition": (
                    "Situs inversus (organ laterality reversal) is extremely rare in JBTS2 (<2%). "
                    "The embryonic node cilia that determine left-right asymmetry are MOTILE cilia — their "
                    "rotational beating generates leftward nodal flow. TZ Y-link scaffold integrity (TMEM216) "
                    "is required for signalling competence of PRIMARY non-motile cilia, but nodal cilia motility "
                    "is driven by axonemal dyneins (DNAI1, DNAI2) and is largely independent of TZ gate function. "
                    "This distinguishes JBTS2 from primary ciliary dyskinesia (PCD) where situs inversus / "
                    "bronchiectasis are cardinal features. Situs inversus in a JBTS2 patient should prompt "
                    "co-sequencing for PCD-associated dyneins (compound disease)."
                ),
            },
            {
                "term": "Retinal rod-cone dystrophy (~35%) in JBTS2",
                "definition": (
                    "Retinal rod-cone dystrophy in ~35% of JBTS2 patients — similar to JBTS1 (INPP5E). "
                    "Mechanism: photoreceptor outer segment connecting cilia require intact TZ Y-link gate "
                    "(TMEM216) for opsin protein trafficking fidelity. TZ gate failure → non-selective "
                    "protein diffusion in connecting cilia → opsin mislocalisation → photoreceptor "
                    "degeneration. Annual ERG from age 3 mandatory. Ophthalmology involvement from "
                    "diagnosis regardless of fundoscopy appearance (ERG abnormal before fundoscopic "
                    "changes). Distinct from Leber Congenital Amaurosis (LCA) — later-onset, "
                    "milder in JBTS2 than CEP290-related LCA."
                ),
            },
        ],

        "domain_matrix": [
            {
                "domain":          "N-terminal cytoplasmic tail / RPGRIP1L interaction (aa 1–15)",
                "location":        "N-terminus — cytoplasmic; TZ matrix anchoring; NPHP-module Y-link base",
                "function":        "Short cytoplasmic N-tail anchors TMEM216 to the TZ matrix via RPGRIP1L (NPHP8/JBTS7) interaction. RPGRIP1L provides the vertical cytoplasmic pillar of the Y-link; TMEM216 N-tail connects TM scaffold to this pillar. Loss of N-tail contacts → partial TZ matrix uncoupling without complete TM scaffold loss.",
                "variant_examples":"N-tail variants: rare; partial RPGRIP1L uncoupling; mild-moderate JBTS2 phenotype",
            },
            {
                "domain":          "EL1 / TMEM67-MKS module docking (aa 39–45)",
                "location":        "Extracellular loop 1 — periaxonemal space; TMEM67/MKS3 and MKS1 interaction surface",
                "function":        "EL1 protrudes into the periaxonemal space (between axoneme and ciliary membrane) and docks onto TMEM67 (MKS3/JBTS6 protein) and MKS1. This interaction links TMEM216 into the broader MKS-module Y-link scaffold. EL1 disruption → MKS-module uncoupling → Y-link gate failure → in severe cases, MKS2 risk even with partial TM scaffold.",
                "variant_examples":"EL1 variants associated with MKS2-JBTS2 overlap phenotypes; hepatic penetrance increases with EL1 disruption",
            },
            {
                "domain":          "ICL1 / NPHP4-B9D1 contact / Arg73 founder site (aa 71–78)",
                "location":        "Intracellular loop 1 — cytoplasmic; NPHP4-B9D1-B9D2 scaffold contact; Arg73 salt bridge",
                "function":        "ICL1 is the primary pathogenic hotspot for JBTS2 missense alleles. Arg73 forms an ICL1 salt bridge critical for NPHP4 and B9D1 docking to the cytoplasmic face of TM scaffold. Arg73Leu (Ashkenazi founder) disrupts this salt bridge → partial NPHP4-B9D1 uncoupling → attenuated Y-link gate → hypomorphic phenotype (JBTS2, not MKS2). Full ICL1 deletion or truncation (Arg65Ter) → MKS2-equivalent gate collapse.",
                "variant_examples":"Arg73Leu (c.218G>T, Ashkenazi founder, hypomorphic, moderate JBTS2); Glu66Lys (c.196G>A, MENA, moderate-severe); Phe81Ser (c.242T>C, TM3-ICL1 boundary, pan-ethnic, moderate-severe); Arg65Ter (c.193C>T, truncating null — MKS2 if biallelic, JBTS2 if compound with missense)",
            },
            {
                "domain":          "EL2 / tectonic complex contact (aa 102–110)",
                "location":        "Extracellular loop 2 — periaxonemal space; TCTN1-TCTN2-TCTN3 tectonic complex docking",
                "function":        "EL2 docks TMEM216 onto the tectonic complex (TCTN1/JBTS9, TCTN2/JBTS13, TCTN3/JBTS18 proteins). This positions TMEM216 within the outer TZ gate scaffold (tectonic-MKS interface). EL2 variants disrupt tectonic docking → outer TZ gate leakier → Hedgehog signalling failure; hepatic penetrance increases (biliary cholangiocytes highly dependent on tectonic-TMEM gate).",
                "variant_examples":"Tyr109Cys (c.326A>G, EL2, North African founder, moderate-severe); Trp105Ter (c.315G>A, EL2 truncating null, MKS2 if biallelic)",
            },
            {
                "domain":          "TM4 / MKS1-RPGRIP1L cytoplasmic docking (aa 111–134)",
                "location":        "Fourth transmembrane helix — cytoplasmic leaflet face docks MKS1 and RPGRIP1L",
                "function":        "TM4 cytoplasmic face coordinates MKS1 (B9-complex) and RPGRIP1L binding. MKS1 connection links TMEM216 TM scaffold to the B9-complex gate (B9D1-B9D2-MKS1). TM4 missense alleles → partial B9-complex uncoupling → moderate-severe JBTS2; renal and hepatic penetrance elevated (B9 complex required for tubular and biliary cilia).",
                "variant_examples":"Arg123Gln (c.368G>A, TM4, South Asian, moderate); Tyr135Cys (c.404A>G, C-tail/TM4 boundary, MENA, moderate-severe)",
            },
        ],

        "clinical_pearls": [
            {
                "title": "TMEM216 — MKS2-JBTS2: Allele Class + Brain MRI + Prenatal USS = Mandatory Triple Gate",
                "detail": (
                    "TMEM216 biallelic sequencing returning any result requires THREE mandatory steps before syndrome "
                    "assignment: (1) Allele class — truncating null/null → MKS2 counselling; null/missense or missense/missense "
                    "→ JBTS2 pathway. (2) Brain MRI — MTS present → JBTS2; occipital encephalocele → MKS2 grade. "
                    "(3) Prenatal/neonatal: renal USS — bilateral enlarged polycystic kidneys → MKS2. Never report "
                    "'TMEM216 pathogenic biallelic' without the tier assignment. Genetic counselling must address "
                    "both JBTS2 (25% sibling risk of MTS, live birth) and MKS2 (25% risk of lethal Meckel if biallelic null)."
                ),
            },
            {
                "title": "Panel Must Include TMEM67 (JBTS6/COACH), RPGRIP1L (JBTS7), TCTN2 (JBTS13): Same Y-Link Module",
                "detail": (
                    "TMEM216 (11q13.2) interacts directly with TMEM67 (8q22.2), RPGRIP1L (16q12.2), and TCTN1-3 in the Y-link "
                    "TZ scaffold. All of these are individually rare (<3% each of JBTS) but collectively form the most common "
                    "TZ module causing JBTS. Single-gene TMEM216 testing does NOT exclude the module. WES or JBTS "
                    "multigene panel (>50 genes) is mandatory. Phenotypic DDx: COACH syndrome (TMEM67/RPGRIP1L = "
                    "coloboma + high CHF); JBTS2 = high OMA (55%) + high CHF (20%) + no coloboma. Coloboma in JBTS2 "
                    "should prompt urgent TMEM67 / RPGRIP1L testing."
                ),
            },
            {
                "title": "Hepatic Fibrosis Annual Surveillance from Diagnosis — Not Optional in JBTS2",
                "detail": (
                    "JBTS2 has ~20% hepatic penetrance (CHF/DPM) — significantly higher than most JBTS types. "
                    "Hepatic fibrosis is clinically silent in early childhood (normal LFTs, normal liver size) but "
                    "progressive. Portal hypertension and varices can develop by age 10–15. Annual protocol: "
                    "LFTs (ALT, GGT, bilirubin), hepatic USS with portal Doppler, elastography if available. "
                    "MKS-module overlap (same Y-link complex as TMEM67/COACH) explains disproportionate hepatic "
                    "involvement vs other JBTS types. GI / hepatology co-management mandatory at first hepatic finding. "
                    "Liver transplant has been curative for end-stage CHF — no recurrence (cell-autonomous cilia defect "
                    "in donor organ corrected)."
                ),
            },
            {
                "title": "Renal Surveillance — ESRD Median ~25yr, Younger than INPP5E/JBTS1 (~35yr)",
                "detail": (
                    "Renal NPHP-like nephropathy affects ~28% of JBTS2 patients (higher than INPP5E/JBTS1 ~12%). "
                    "ESRD median ~25yr when renal disease present (earlier than JBTS1 but later than CEP83/JBTS22 "
                    "~14-18yr). Annual renal protocol: spot urine protein/creatinine ratio, osmolality (concentrating "
                    "defect sentinel), GFR estimation (cystatin C preferred over creatinine in young ciliopathy patients), "
                    "renal USS (echogenicity, cyst detection). Renal transplant is curative — no recurrence in "
                    "allograft (cell-autonomous tubular cilia defect). Preemptive transplant evaluation recommended "
                    "at GFR <30 ml/min/1.73m²."
                ),
            },
            {
                "title": "DDx Nasal Brushing: Normal Cilia Structure Confirms TZ Gate Defect, Not PCD",
                "detail": (
                    "JBTS2/TMEM216: nasal brushing electron microscopy shows NORMAL ciliary ultrastructure (normal "
                    "outer dynein arms, nexin links, central pair). Beat frequency may be mildly reduced but not "
                    "absent. Primary Ciliary Dyskinesia (PCD) shows absent/reduced ODAs, nexin defects, transposition — "
                    "absent in TMEM216. If nasal brushing shows PCD-pattern in a suspected JBTS2 patient → compound "
                    "disease (JBTS2 + PCD) or alternative diagnosis (not JBTS2 alone). Conversely: normal nasal "
                    "brushing does NOT exclude JBTS2 — TZ gate defects do not disrupt axonemal dynein assembly."
                ),
            },
        ],

        "literature_highlights": [
            "Valente EM et al. (2010) Mutations in TMEM216 perturb ciliogenesis and cause Joubert, Meckel and related syndromes. Nat Genet 42(7):619–25. [JBTS2/MKS2 TMEM216 discovery; allelic spectrum; TZ gate mechanism established].",
            "Edvardson S et al. (2010) Joubert syndrome 2 (JBTS2) in Ashkenazi Jews and its DNA-based prenatal diagnosis. Ann Neurol 67(4):524–31. [Arg73Leu Ashkenazi founder allele; ~1:92 carrier frequency; prenatal diagnosis].",
            "Shi X et al. (2017) Super-resolution microscopy reveals that disruption of ciliary transition-zone architecture causes Joubert syndrome. Nat Cell Biol 19(10):1178–88. [TMEM216 Y-link ultrastructure; TZ gate diffusion barrier mechanism; super-resolution imaging of Y-links].",
            "Huang L et al. (2011) TMEM216 protein and its role in ciliogenesis and human ciliopathies. J Cell Biol 193(7):1325–39. [TMEM216 domain architecture; RPGRIP1L-TMEM67 interaction network; Y-link scaffold assembly].",
            "Garcia-Gonzalo FR et al. (2011) A transition zone complex regulates mammalian ciliogenesis and ciliary-based sensory functions. Nat Genet 43(8):776–84. [TZ gate complex: TMEM216-TMEM67-RPGRIP1L-TCTN1 module; hepatic + renal penetrance mechanism].",
            "Barker AR et al. (2014) The CDK5 and Abl enzyme substrate CABLES1 controls cisplatin sensitivity and the TZ complex in mouse models of Joubert syndrome. Cell Rep 8(6):1600–13. [TZ gate diffusion barrier molecular mechanism; JBTS2 ciliary signalling model].",
        ],

        "phenotype_frequencies": {
            "mts_pathognomonic":       "100% (MTS diagnostic criterion — JBTS2 missense/hypomorphic cohort; MKS2 null excluded)",
            "cerebellar_ataxia":       f"{_pct(n_ataxia)}%",
            "neonatal_hypotonia":      f"{_pct(n_hypotonia)}%",
            "oculomotor_apraxia":      f"{_pct(n_oma)}%",
            "breathing_dysregulation": f"{_pct(n_breath)}%",
            "intellectual_disability": f"{_pct(n_id)}%",
            "retinal_rod_cone":        f"{_pct(n_retinal)}% (TZ gate leakiness → connecting cilia opsin trafficking failure; annual ERG from age 3)",
            "renal_nphp_like":         f"{_pct(n_renal)}% (tubulointerstitial nephritis; ESRD median ~25yr when affected; NPHP4-B9D1 contact disrupted)",
            "hepatic_chf_dpm":         f"{_pct(n_hepatic)}% (biliary cholangiocyte TZ gate / MKS-module overlap with TMEM67; annual LFT + hepatic USS mandatory)",
            "esrd_at_study":           f"{_pct(n_esrd)}%",
            "polydactyly_post_axial":  f"{_pct(n_poly)}% (Gli2/3 Hedgehog failure; Y-link TZ gate disruption)",
            "situs_inversus":          f"{_pct(n_situs)}% (<2% — nodal cilia motility largely independent of TZ gate integrity)",
            "cilia_structure":         "NORMAL (nasal brushing: normal TEM ultrastructure — TZ gate defect does not ablate axoneme; unlike PCD)",
            "mks2_tier":               "MKS TIER CONFIRMED — biallelic truncating null TMEM216 → MKS2 (#603194) perinatal lethal",
            "ashkenazi_founder":       "Arg73Leu (c.218G>T) — ICL1 salt bridge; Ashkenazi Jewish founder; ~1:92 carrier; commonest JBTS2 allele worldwide",
            "jbts2_frequency":         "~2–3% of all Joubert syndrome; ~1:500,000–1,000,000 worldwide",
        },
    }
