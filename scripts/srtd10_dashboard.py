"""
IFT172 Short-Rib Thoracic Dysplasia 10 (SRTD10) — Jeune Asphyxiating Thoracic Dystrophy Type 10
==================================================================================================
Primary Gene : IFT172 (*607386) — 2p23.3; ~1749 aa; Intraflagellar Transport Protein 172
               IFT172 is the LARGEST subunit of the IFT-B2 subcomplex (the distal, tip-proximal
               half of the IFT-B anterograde transport complex).
               IFT-B2 contains: IFT80 (SRTD1), IFT172 (SRTD10), IFT88, IFT57, IFT38/CLUAP1
               (SRTD13), IFT25, IFT27.
               IFT172 has a large N-terminal WD40 multi-repeat domain and a C-terminal TPR domain.
               IFT172 functions as the distal tip-anchoring scaffold — it contacts kinesin-2
               (KIF3A/KIF3B/KAP) at the ciliary tip, stabilising IFT-B2 at the anterograde end
               of the IFT train. Loss → IFT-B2 disassembles at the ciliary tip → anterograde IFT
               truncated → SHORTENED CILIA (EM; same class as SRTD1/IFT80) → Hedgehog (Ihh/Shh)
               components cannot reach tip → GLI3R accumulates in chondrocytes → NARROW THORAX.
               IFT172 is also expressed in the photoreceptor connecting cilium (rod-cone) and
               renal tubular primary cilia.

Disease OMIM : #615490 — Asphyxiating Thoracic Dystrophy 10 (ATD10 / SRTD10)
               Same ciliopathy spectrum as SRTD1; IFT172 mutations discovered in 2015
               (Halbritter et al., Am J Hum Genet; Bujakowska et al., J Clin Invest).
               Rare biallelic null alleles → Short-Rib Polydactyly Syndrome (SRPS) spectrum.
               No separate renal-only (NPHP) or Joubert phenotype described for IFT172.
               Note: IFT172 was also first described as a Bardet-Biedl-like gene (retinal
               phenotype) before full SRTD10 skeletal phenotype was delineated.
Chromosome   : 2p23.3
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or homozygous consanguineous)
Prevalence   : ~1/200,000–500,000; ~1–2% of molecularly confirmed SRTD cases (2026);
               Rarer than DYNC2H1/SRTD3 (~50%) and WDR19/SRTD5 (~1%), but slightly more
               common than IFT80/SRTD1 (~0.5%); ~30–50 families reported worldwide.

Protein Structure — IFT172 (1749 aa; IFT-B2 largest subunit)
--------------------------------------------------------------
Domain 1: N-terminal WD40 multi-repeat region (aa 1–980)
           Large multi-blade WD40 beta-propeller; forms the central scaffold of IFT-B2;
           key contacts with IFT80 (SRTD1), IFT88, and IFT57 within IFT-B2;
           Pathogenic missense hotspot: blades 6–12 (aa 400–800); moderate phenotype cluster
Domain 2: WD40-TPR hinge / interdomain linker (aa 981–1100)
           Flexible interdomain region; connects WD40 to C-terminal TPR;
           missense here → severe-moderate phenotype (kinesin-2 anchoring partially lost)
Domain 3: C-terminal TPR domain (aa 1101–1749)
           Multiple TPR repeats; responsible for kinesin-2 (KIF3A) docking at ciliary tip;
           IFT-B2 tip-anchoring function; hypomorphic missense here → milder phenotype
           (partial kinesin-2 contact maintained)

Key pathogenic variant classes (IFT172):
1. WD40 blade 6–12 missense (aa 400–800): compound het or homozygous; moderate SRTD10
2. Compound het: truncating + missense (any domain): moderate to severe SRTD10
3. Biallelic null (truncating + truncating): SRPS spectrum; perinatal lethal
4. Hypomorphic C-terminal TPR missense (aa 1101–1749): milder SRTD10; partial kinesin-2 docking
5. WD40-TPR hinge missense (aa 981–1100): severe-moderate; kinesin-2 anchoring disrupted
"""

import random
import math

SEED = 401
N    = 40   # 40-patient educational cohort

rng  = random.Random(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────
def _pct(n, total=N):
    return round(n / total * 100)

def _split(total, *fractions):
    """Distribute 'total' into len(fractions) buckets deterministically."""
    buckets = [round(total * f) for f in fractions]
    diff = total - sum(buckets)
    buckets[0] += diff
    return buckets

# ── patient-level data (fixed seed) ──────────────────────────────────────────
patients = []
ethnicities = [
    ('Middle Eastern / North African', 0.32),   # consanguineous homozygous — common
    ('European',                        0.28),   # compound het most common
    ('South Asian',                     0.20),   # hypomorphic C-terminal TPR
    ('East Asian',                      0.10),
    ('Latin American',                  0.06),
    ('Other / Unknown',                 0.04),
]
eth_pool = []
for eth, frac in ethnicities:
    eth_pool.extend([eth] * round(frac * N))
while len(eth_pool) < N:
    eth_pool.append('Other / Unknown')
rng.shuffle(eth_pool)

allele_classes = [
    'Compound het missense + truncating',
    'Homozygous missense (WD40 blade 6–12)',
    'Compound het two missense',
    'Homozygous truncating (null / SRPS)',
    'Hypomorphic C-terminal TPR missense',
]
allele_weights = [0.32, 0.30, 0.18, 0.12, 0.08]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null   = 'null' in allele.lower() or 'SRPS' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.24, 0.46, 0.30])[0]

    poly_status = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'Preaxial (rare)', 'None'],
        weights=[0.20, 0.22, 0.04, 0.54])[0]

    renal = rng.choices(
        ['TIN + corticomedullary cysts → ESRD', 'TIN (non-dialysis)', 'None'],
        weights=[0.12, 0.14, 0.74])[0]

    ckd_stage = (
        rng.choices(['CKD 4', 'CKD 5 / ESRD'], weights=[0.38, 0.62])[0]
        if 'ESRD' in renal or 'TIN' in renal else 'None')

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Reduced ERG only', 'None'],
        weights=[0.08, 0.06, 0.86])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Elevated LFTs only', 'None'],
        weights=[0.06, 0.04, 0.90])[0]

    veptr = rng.choices(
        ['VEPTR (growing rod)', 'MAGEC rod', 'Observation only'],
        weights=[0.34, 0.10, 0.56])[0]

    tx_renal = (
        rng.choices(['Renal transplant (done)', 'Dialysis (awaiting Tx)', 'Conservative management'],
                    weights=[0.50, 0.22, 0.28])[0]
        if 'ESRD' in renal else 'None')

    resp_mgmt = rng.choices(
        ['Mechanical ventilation (neonatal)', 'Non-invasive ventilation', 'Observation / physio'],
        weights=[0.22, 0.28, 0.50])[0]

    presentation = rng.choices(
        ['Prenatal (polyhydramnios / short limbs)', 'Neonatal respiratory failure',
         'Infant thorax + limb anomaly', 'Childhood chronic renal + skeletal'],
        weights=[0.24, 0.20, 0.36, 0.20])[0]

    misdiagnosis = rng.choices(
        ['SRTD3/DYNC2H1 (most common SRTD — gene panel corrected)',
         'SRTD1/IFT80 (same IFT-B2 class — gene panel corrected)',
         'EVC (Ellis-van Creveld)',
         'Bardet-Biedl (retinal phenotype prominent)',
         'SRTD5/WDR19 (IFT-A mistaken — gene panel corrected)',
         'None'],
        weights=[0.14, 0.10, 0.06, 0.06, 0.04, 0.60])[0]

    age_dx = rng.choices(
        ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.38, 0.28, 0.20, 0.14])[0]

    sex = rng.choice(['M', 'F'])

    patients.append(dict(
        ethnicity    = eth,
        allele       = allele,
        null_allele  = null,
        thorax_sev   = thorax_sev,
        poly_status  = poly_status,
        renal        = renal,
        ckd_stage    = ckd_stage,
        retinal      = retinal,
        hepatic      = hepatic,
        veptr        = veptr,
        tx_renal     = tx_renal,
        resp_mgmt    = resp_mgmt,
        presentation = presentation,
        misdiagnosis = misdiagnosis,
        age_dx       = age_dx,
        sex          = sex,
    ))

# ── aggregate counts ──────────────────────────────────────────────────────────
def _cnt(field, value):
    return sum(1 for p in patients if p[field] == value)

def _cnt_partial(field, value):
    return sum(1 for p in patients if value in p[field])

thorax_severe_n   = _cnt_partial('thorax_sev', 'Severe')
polydactyly_n     = sum(1 for p in patients if p['poly_status'] != 'None')
renal_any_n       = sum(1 for p in patients if p['renal'] != 'None')
retinal_any_n     = sum(1 for p in patients if p['retinal'] != 'None')
hepatic_chf_n     = sum(1 for p in patients if p['hepatic'] != 'None')
veptr_any_n       = sum(1 for p in patients if p['veptr'] != 'Observation only')
transplant_done_n = _cnt('tx_renal', 'Renal transplant (done)')
misdiagnosis_n    = sum(1 for p in patients if p['misdiagnosis'] != 'None')

sex_M = sum(1 for p in patients if p['sex'] == 'M')
sex_F = N - sex_M

age_dx_0_1   = _cnt('age_dx', '0–1 yr (neonatal)')
age_dx_2_5   = _cnt('age_dx', '2–5 yr (infant)')
age_dx_6_10  = _cnt('age_dx', '6–10 yr (child)')
age_dx_11_16 = _cnt('age_dx', '11–16 yr (teen)')

# ── public API helpers ────────────────────────────────────────────────────────

def get_overview():
    return {
        "disease":    "IFT172 Short-Rib Thoracic Dysplasia 10 (SRTD10 / ATD10 / Jeune Syndrome)",
        "gene":       "IFT172 (*607386) — 2p23.3 — 1749 aa — IFT-B2 LARGEST Subunit (WD40 + TPR; anterograde IFT tip-anchor)",
        "omim_gene":  "607386",
        "omim_disease": "615490",
        "chromosome": "2p23.3",
        "cohort_n":   N,
        "seed":       SEED,
        "kpis": {
            "thorax_severe_n":    thorax_severe_n,
            "thorax_severe_pct":  _pct(thorax_severe_n),
            "polydactyly_n":      polydactyly_n,
            "polydactyly_pct":    _pct(polydactyly_n),
            "renal_any_n":        renal_any_n,
            "renal_any_pct":      _pct(renal_any_n),
            "retinal_any_n":      retinal_any_n,
            "retinal_any_pct":    _pct(retinal_any_n),
            "hepatic_chf_n":      hepatic_chf_n,
            "hepatic_chf_pct":    _pct(hepatic_chf_n),
            "veptr_any_n":        veptr_any_n,
            "veptr_any_pct":      _pct(veptr_any_n),
            "transplant_done_n":  transplant_done_n,
            "misdiagnosis_n":     misdiagnosis_n,
            "misdiagnosis_pct":   _pct(misdiagnosis_n),
        },
        "sex_split":  {"M": sex_M, "F": sex_F},
        "age_distribution": {
            "dx_0_1yr":   age_dx_0_1,
            "dx_2_5yr":   age_dx_2_5,
            "dx_6_10yr":  age_dx_6_10,
            "dx_11_16yr": age_dx_11_16,
        },
        "mechanism": (
            "IFT172 loss disables the IFT-B2 subcomplex at the ciliary tip. IFT172 is the "
            "LARGEST IFT-B2 subunit (1749 aa) and functions as the tip-anchoring scaffold — "
            "its C-terminal TPR domain docks kinesin-2 (KIF3A/KIF3B/KAP) at the anterograde "
            "tip of the IFT train. Without IFT172, IFT-B2 loses tip stability → anterograde "
            "IFT-B train cannot complete ciliary extension → SHORTENED CILIA (EM; same IFT-B2 "
            "class as SRTD1/IFT80) → Hedgehog signal components (SMO, Gli2/Gli3, Patched) "
            "cannot be transported to the ciliary tip for processing → GLI3R accumulates in "
            "chondrocytes → NARROW THORAX (primary), short ribs, short limbs. Secondary: renal "
            "TIN/ESRD (~20–30% survivors), rod-cone dystrophy (~10–15%), CHF (~8%). Notable: "
            "IFT172 retinal phenotype was identified before the full SRTD10 skeletal phenotype "
            "(Bujakowska et al., 2015, J Clin Invest) — atypical BBS-like retinal presentation "
            "is a recognised IFT172 phenotypic spectrum."
        ),
        "key_distinction": (
            "IFT172 (SRTD10) is the LARGEST IFT-B2 subunit and the SECOND IFT-B2 SRTD gene "
            "after IFT80 (SRTD1, 2007). IFT172 is unique among SRTD genes for its dual "
            "phenotypic presentation: SRTD10 (full SRTD thoracic + skeletal) and retinal-only "
            "BBS-like phenotype (severe allele effect at retinal connecting cilium with milder "
            "systemic effect). Ciliary EM: SHORTENED cilia — same IFT-B2 class as SRTD1/IFT80, "
            "distinct from IFT-A short stubby (SRTD4/5/7/9) and dynein-2 club-tip (SRTD3/8/11/15/17). "
            "NO NPHP phenotype (unlike TTC21B/SRTD4). NO ectodermal features (unique to WDR19/SRTD5). "
            "NO Joubert MTS. More common than IFT80/SRTD1: ~1–2% of SRTD vs ~0.5%."
        ),
        "ift_b2_subunit_table": [
            {"subunit": "IFT80",      "role": "WD40 β-propeller; IFT38/CLUAP1 + IFT88 contact; IFT-B2 assembly scaffold",      "srtd": "SRTD1",  "omim_gene": "611229", "chr": "3q25.33",  "freq": "~0.5–1% SRTD (founding gene, 2007)"},
            {"subunit": "IFT172",     "role": "LARGEST IFT-B2 subunit; WD40 + TPR; kinesin-2 tip-anchoring; IFT-B2 scaffold",   "srtd": "SRTD10", "omim_gene": "607386", "chr": "2p23.3",   "freq": "~1–2% SRTD (2nd IFT-B2 gene, 2015)"},
            {"subunit": "CLUAP1/IFT38", "role": "IFT-B2 hub; directly contacts IFT80 N-terminal WD40; IFT-B1/B2 linchpin",     "srtd": "SRTD13", "omim_gene": "616470", "chr": "16q23.1",  "freq": "Very rare (<10 families, 2016)"},
            {"subunit": "IFT88",      "role": "IFT-B2 structural scaffold; contacts IFT80 C-terminal; Hedgehog signalling",     "srtd": "—",      "omim_gene": "600595", "chr": "13q12.11", "freq": "No SRTD reported (may be lethal)"},
            {"subunit": "IFT57",      "role": "IFT-B2 coiled-coil; contacts IFT-B1 via IFT52 bridge",                          "srtd": "—",      "omim_gene": "605502", "chr": "3p21.31",  "freq": "No SRTD reported"},
            {"subunit": "IFT25",      "role": "Small IFT-B2 subunit; Hedgehog signalling; binds IFT27 (Rab-like GTPase)",       "srtd": "—",      "omim_gene": "613286", "chr": "17q21.2",  "freq": "Retinal only (not SRTD)"},
            {"subunit": "IFT27",      "role": "Rab-like GTPase; IFT-B2 membrane-signalling; pairs with IFT25",                  "srtd": "—",      "omim_gene": "615504", "chr": "22q12.3",  "freq": "Bardet-Biedl-like (not SRTD)"},
        ],
    }


def get_breakdown():
    def _dist(field):
        counts = {}
        for p in patients:
            v = p[field]
            counts[v] = counts.get(v, 0) + 1
        return [{"label": k, "n": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]

    def _eth_dist():
        counts = {}
        for p in patients:
            e = p['ethnicity']
            counts[e] = counts.get(e, 0) + 1
        return [{"ethnicity": k, "n": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]

    return {
        "thorax_distribution": _dist('thorax_sev'),
        "polydactyly_distribution": [
            {"label": "Postaxial (bilateral)",  "n": _cnt('poly_status', 'Postaxial (bilateral)')},
            {"label": "Postaxial (unilateral)", "n": _cnt('poly_status', 'Postaxial (unilateral)')},
            {"label": "Preaxial (rare)",        "n": _cnt('poly_status', 'Preaxial (rare)')},
            {"label": "None",                   "n": _cnt('poly_status', 'None')},
        ],
        "renal_distribution": _dist('renal'),
        "ckd_stage_distribution": [
            {"label": "CKD 4",         "n": _cnt('ckd_stage', 'CKD 4')},
            {"label": "CKD 5 / ESRD",  "n": _cnt('ckd_stage', 'CKD 5 / ESRD')},
            {"label": "None / not applicable", "n": _cnt('ckd_stage', 'None')},
        ],
        "retinal_distribution": _dist('retinal'),
        "hepatic_distribution": _dist('hepatic'),
        "veptr_distribution": _dist('veptr'),
        "presentation_distribution": _dist('presentation'),
        "misdiagnosis_distribution": _dist('misdiagnosis'),
        "allele_class_summary": _dist('allele'),
        "ethnicity_distribution": _eth_dist(),
        "treatment_renal": _dist('tx_renal'),
        "respiratory_management": _dist('resp_mgmt'),
        "top_variants": [
            {"variant": "p.Pro740Ser (c.2218C>T) — WD40 blade 9 (IFT80-contact surface, aa 720–760) — European compound het — moderate SRTD10", "n": rng.randint(4, 6)},
            {"variant": "p.Arg880Cys (c.2638C>T) — WD40-TPR hinge (aa 870–910; kinesin-2 anchoring region) — Middle Eastern homozygous — moderate-severe SRTD10", "n": rng.randint(3, 5)},
            {"variant": "p.Gln992Ter (c.2974C>T) — truncating null (hinge/TPR boundary) — SRPS spectrum — pan-ethnic", "n": rng.randint(2, 4)},
            {"variant": "p.Ala1146Val (c.3437C>T) — C-terminal TPR (kinesin-2 KIF3A-docking repeat, aa 1130–1170) — South Asian hypomorphic — mild SRTD10", "n": rng.randint(2, 3)},
            {"variant": "p.Lys505Glu (c.1513A>G) — WD40 blade 6 (IFT88-contact zone, aa 490–530) — MENA homozygous — moderate SRTD10", "n": rng.randint(1, 3)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":             "IFT172 — Intraflagellar Transport Protein 172",
            "also_known_as":    "osm-1 (C. elegans ortholog); KIAA0222 (historical); SRTD10 gene; largest IFT-B2 subunit",
            "omim_gene":        "*607386",
            "chromosome":       "2p23.3",
            "protein_size":     "1749 aa",
            "protein_class":    "IFT-B2 subcomplex — LARGEST subunit; WD40 + TPR dual-domain scaffold; anterograde IFT tip-anchor",
            "domain_structure": "N-terminal WD40 multi-repeat domain (aa 1–980; large multi-blade propeller; IFT80/IFT88/IFT57 contacts; pathogenic hotspot blades 6–12 aa 400–800) + WD40-TPR hinge (aa 981–1100; flexible; severe-moderate missense phenotype; kinesin-2 docking partially lost) + C-terminal TPR domain (aa 1101–1749; multiple TPR repeats; kinesin-2 KIF3A tip-docking; hypomorphic missense → mild SRTD10; partial kinesin-2 contact maintained)",
            "function":         "Largest IFT-B2 subunit; scaffolds IFT-B2 assembly via WD40 domain (contacts IFT80/SRTD1, IFT88, IFT57 within IFT-B2); anchors kinesin-2 (KIF3A) at ciliary tip via C-terminal TPR domain; loss → IFT-B2 tip instability → anterograde IFT train truncated → shortened cilia → Hedgehog processing failure",
            "inheritance":      "Autosomal Recessive — biallelic LOF",
            "key_contacts":     "IFT80 (SRTD1; WD40 domain contact within IFT-B2); IFT88 (WD40 blade 6 zone); IFT57 (WD40 domain structural partner); kinesin-2/KIF3A (C-terminal TPR domain; tip-anchoring)",
            "expression":       "Primary non-motile cilia (9+0); chondrocytes (growth plate); renal tubular epithelium; retinal photoreceptor connecting cilium (prominent — BBS-like retinal phenotype)",
            "phenotypic_note":  "DUAL PHENOTYPE: SRTD10 (full skeletal + renal + retinal) with severe alleles, AND retinal-predominant BBS-like phenotype (hypomorphic alleles with tissue-specific sensitivity of photoreceptor connecting cilium). IFT172 is the only IFT-B2 SRTD gene with a recognised retinal-predominant allele series.",
            "discovery_note":   "IFT172/SRTD10 identified in 2015 by two simultaneous reports — Halbritter et al. (Am J Hum Genet 2015) for SRTD10 skeletal phenotype, and Bujakowska et al. (J Clin Invest 2015) for retinal phenotype — establishing IFT172 as both an SRTD and retinal dystrophy gene.",
            "ift_b_complex":    "IFT-B anterograde complex has two sub-complexes: IFT-B1 (core, kinesin-docking base) and IFT-B2 (distal, tip-proximal). IFT172 is exclusively IFT-B2, and is the LARGEST IFT-B2 component.",
        },
        "disease_card": {
            "omim_disease":     "#615490 — Asphyxiating Thoracic Dystrophy 10 (ATD10 / SRTD10)",
            "full_name":        "Short-Rib Thoracic Dysplasia 10 With or Without Polydactyly",
            "alternative_names": "SRTD10; Asphyxiating Thoracic Dystrophy 10 (ATD10); Jeune syndrome type 10",
            "primary_finding":  "NARROW THORAX — pathognomonic; neonatal respiratory failure in severe (null) alleles",
            "polydactyly":      "~40–50% (postaxial most common; preaxial rare ~4%)",
            "renal":            "TIN + corticomedullary cysts → ESRD in ~20–30% survivors; NO NPHP phenotype",
            "retinal":          "Rod-cone dystrophy in ~10–15% (connecting cilium IFT-B2 failure; more frequent than SRTD1/IFT80); BBS-like hypomorphic allele series recognised",
            "hepatic_chf":      "Ductal plate malformation → CHF in ~8%",
            "situs_inversus":   "ABSENT — primary non-motile cilia (9+0); not nodal motile cilia",
            "joubert_mts":      "ABSENT — no Molar Tooth Sign described for IFT172",
            "ectodermal":       "ABSENT — no sparse hair, hypodontia, or dystrophic nails (ectodermal is unique to WDR19/SRTD5)",
            "srps_spectrum":    "Biallelic null alleles → SRPS (perinatal lethal)",
            "renal_transplant": "CURATIVE — cell-autonomous IFT defect; no post-Tx recurrence",
            "surgical_treatment": "VEPTR / MAGEC growing rods (primary thoracic intervention)",
            "prevalence":       "~1/200,000–500,000; ~30–50 families worldwide (2026); ~1–2% of SRTD (rare; slightly more common than SRTD1/IFT80)",
            "dual_phenotype_note": "UNIQUE among IFT-B2 SRTD genes: IFT172 alleles can cause either full SRTD10 (biallelic LOF, skeletal dominant) or retinal-only BBS-like phenotype (hypomorphic alleles; photoreceptor connecting cilium most sensitive). Gene panel mandatory to distinguish from other SRTDs.",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, polyhydramnios → suspect SRTD",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened tubular bones; SHORTENED cilia (EM) — same IFT-B2 class as IFT80/SRTD1; distinct from IFT-A short stubby and dynein-2 club tip",
            "3. GENE PANEL (mandatory first-line): IFT172 must be included alongside DYNC2H1, WDR19, IFT140, WDR35, WDR60, WDR34, TTC21B, TCTEX1D2, DYNC2LI1, IFT80, CLUAP1 — SRTD10 clinically identical to all other SRTDs; gene panel is the ONLY way to distinguish",
            "4. WES + CNV if panel negative — IFT172 2p23.3; CNV deletion analysis mandatory for large gene deletions",
            "5. Ciliopathy EM (skin/fibroblast biopsy): SHORTENED cilia — IFT-B2 failure (same class as SRTD1/IFT80); cannot distinguish SRTD1 from SRTD10 by EM alone; gene panel always required",
            "6. Renal USS: corticomedullary cysts; creatinine + GFR annually; NO NPHP variant expected for IFT172",
            "7. ERG annually from age 6 (rod-cone dystrophy ~10–15%; note BBS-like retinal phenotype possible with hypomorphic alleles); ophthalmology referral — retinal involvement more frequent than SRTD1",
            "8. Liver USS + APRI if hepatic signs suspected (CHF ~8%); liver biopsy if APRI elevated",
            "9. Brain MRI: NO Molar Tooth Sign expected; normal posterior fossa",
            "10. MLPA/CNV for IFT172 2p23.3 deletions if sequencing negative but phenotype compelling",
            "11. Consider retinal-only phenotype workup if atypical BBS presentation without thoracic: IFT172 hypomorphic alleles may cause isolated retinal dystrophy — ophthalmology gene panel includes IFT172",
        ],
        "mechanism_glossary": [
            {"term": "IFT172",           "definition": "Intraflagellar Transport Protein 172 — 1749 aa; LARGEST IFT-B2 subunit; WD40 multi-repeat domain (N-terminal; IFT80/IFT88/IFT57 contacts) + C-terminal TPR domain (kinesin-2/KIF3A tip-anchoring). Loss → IFT-B2 tip instability → anterograde IFT truncated → shortened cilia."},
            {"term": "IFT-B2 subcomplex", "definition": "The distal (tip-proximal) half of the IFT-B anterograde complex: IFT80 (SRTD1) + IFT172 (SRTD10) + IFT88 + IFT57 + IFT38/CLUAP1 (SRTD13) + IFT25 + IFT27. IFT-B2 is the 'tip-anchored' sub-complex that positions Hedgehog components for ciliary tip processing."},
            {"term": "IFT-B complex",   "definition": "The anterograde IFT complex, driven by kinesin-2 (KIF3A/KIF3B/KAP) from the ciliary base to the tip. Consists of IFT-B1 (core/proximal) and IFT-B2 (distal/tip-proximal). IFT172 is the key tip-anchoring scaffold within IFT-B2."},
            {"term": "Shortened cilia", "definition": "EM finding in IFT-B2 SRTDs (SRTD1/IFT80, SRTD10/IFT172, SRTD13/CLUAP1): cilia are shortened because IFT-B anterograde delivery to the tip is impaired — axonemal building blocks cannot be delivered. Distinct from IFT-A short stubby (IFT-B cannot enter at base) and dynein-2 club/bulging tip (retrograde failure)."},
            {"term": "Kinesin-2 tip-anchoring", "definition": "The C-terminal TPR domain of IFT172 docks kinesin-2 (KIF3A) at the distal tip of the anterograde IFT train, stabilising the motor–cargo complex during tip-directed transport. Without IFT172 TPR, kinesin-2 loses tip-anchorage → IFT-B2 cargo (Hedgehog components) falls short of the ciliary tip."},
            {"term": "Anterograde IFT", "definition": "Kinesin-2-driven transport from the ciliary base to the tip. IFT172 is the largest scaffold in this process within IFT-B2, ensuring Hedgehog signal components (SMO, Gli2/3, Patched) reach the tip for processing."},
            {"term": "Hedgehog (Ihh/Shh)", "definition": "Growth plate signalling requiring ciliary tip delivery. IFT172 loss → Gli components cannot reach the tip → GLI3R accumulates → Ihh/Shh response impaired in chondrocytes → short ribs, narrow thorax, short limbs."},
            {"term": "BBS-like retinal phenotype", "definition": "IFT172 hypomorphic alleles (particularly C-terminal TPR missense) can cause retinal-predominant phenotype resembling Bardet-Biedl syndrome — rod-cone dystrophy without full SRTD10 thoracic skeleton. The photoreceptor connecting cilium appears most sensitive to partial IFT172 loss. Gene panel includes IFT172 in BBS work-up."},
            {"term": "Cell-autonomous",  "definition": "The IFT-B2 defect is intrinsic to the cell. After renal transplantation, the donor kidney (IFT172+) functions normally — CURATIVE for renal disease; no recurrence post-transplant."},
        ],
        "key_variants": [
            {"variant": "p.Pro740Ser",  "domain": "WD40 blade 9 (IFT80-contact surface, aa 720–760)", "consequence": "Disrupts WD40 surface contacting IFT80 (SRTD1) within IFT-B2; IFT-B2 assembly impaired; European compound het; moderate SRTD10",     "ethnicity": "European"},
            {"variant": "p.Arg880Cys",  "domain": "WD40-TPR hinge (aa 870–910; kinesin-2 anchoring)", "consequence": "Hinge missense → kinesin-2 tip-anchoring partially disrupted; IFT-B2 tip unstable; Middle Eastern homozygous; moderate-severe SRTD10", "ethnicity": "Middle Eastern (homozygous)"},
            {"variant": "p.Gln992Ter",  "domain": "Truncating null (hinge/TPR boundary)",             "consequence": "Premature stop; biallelic null → SRPS spectrum; pan-ethnic",                                                                            "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ala1146Val", "domain": "C-terminal TPR (kinesin-2 KIF3A-docking repeat, aa 1130–1170)", "consequence": "Hypomorphic; South Asian; milder SRTD10; partial kinesin-2 tip-docking maintained; retinal involvement prominent",       "ethnicity": "South Asian"},
            {"variant": "p.Lys505Glu",  "domain": "WD40 blade 6 (IFT88-contact zone, aa 490–530)",   "consequence": "IFT88-IFT172 interface disrupted; MENA homozygous; moderate SRTD10",                                                                 "ethnicity": "MENA"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months; same as all SRTDs",
            "2. Respiratory: neonatal mechanical ventilation if severe null alleles; CPAP/NIV for moderate; wean as thorax expands with VEPTR",
            "3. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — no recurrence (cell-autonomous). NOTE: NO NPHP allele — all SRTD10 renal disease is secondary SRTD skeletal phenotype",
            "4. Retinal: annual ERG from age 6; ophthalmology; low vision support — retinal involvement more frequent than SRTD1 (~10–15%); hypomorphic alleles may cause BBS-like isolated retinal dystrophy; no disease-modifying therapy 2026",
            "5. Hepatic: APRI + USS annually if suspected CHF; hepatology referral; avoid hepatotoxic drugs",
            "6. Genetics: cascade testing of parents + siblings; NO NPHP or Joubert allele-severity counselling; standard AR 25% recurrence risk; prenatal/PGT; COUNSEL re: retinal-only allele risk (hypomorphic C-terminal TPR variants may cause isolated retinal phenotype)",
            "7. MDT: paediatric orthopaedics (VEPTR), nephrology, respiratory, ophthalmology (prominent retinal risk), genetics, hepatology",
            "8. BBS-like phenotype counselling: families with IFT172 hypomorphic alleles may present with isolated retinal dystrophy mimicking Bardet-Biedl — gene panel (including IFT172) in BBS workup; ophthalmology referral mandatory for all SRTD10 patients regardless of thoracic severity",
        ],
        "ddx_table": [
            {"disease": "SRTD1 (IFT80)",      "key_difference": "SAME IFT-B2 COMPLEX — IFT80 is the FOUNDING SRTD gene (2007); shortened cilia (same EM class); gene panel is the ONLY way to distinguish SRTD10 from SRTD1; IFT80 rarer (~0.5% vs ~1–2%)"},
            {"disease": "SRTD13 (CLUAP1)",    "key_difference": "SAME IFT-B2 COMPLEX — CLUAP1/IFT38 directly contacts IFT80; very rare (<10 families); shortened cilia; gene panel mandatory"},
            {"disease": "SRTD3 (DYNC2H1)",    "key_difference": "Most common SRTD (~50%); DYNEIN-2 (not IFT-B2); CLUB cilia on EM (IFT-B stranded at tip) — distinct from SRTD10 shortened cilia; gene panel resolves"},
            {"disease": "SRTD5 (WDR19)",      "key_difference": "Most common IFT-A SRTD; IFT-A complex (not IFT-B2); short STUBBY cilia; ectodermal features UNIQUE to WDR19 (absent in SRTD10); gene panel required"},
            {"disease": "SRTD9 (IFT140)",     "key_difference": "IFT-A core scaffold (not IFT-B2); short stubby cilia (IFT-A); no ectodermal; gene panel required"},
            {"disease": "SRTD4 (TTC21B)",     "key_difference": "IFT-A TPR adaptor (not IFT-B2); short stubby cilia; THREE phenotypes (SRTD4/NPHP12/JBTS12); IFT172 has NO NPHP/Joubert allele"},
            {"disease": "Bardet-Biedl (BBS)", "key_difference": "IFT172 can MIMIC BBS with hypomorphic retinal-predominant alleles (BBS-like phenotype); no narrow thorax in BBS; obesity + hypogonadism absent in SRTD10; gene panel differentiates — IFT172 is now on BBS gene panels"},
            {"disease": "EVC (Ellis-van Creveld)", "key_difference": "CHD present in ~60% EVC; EVC ectodermal features (teeth/nails/hair); EVC is hedgehog-cilia pathway but not IFT-B2"},
            {"disease": "Thanatophoric dysplasia", "key_difference": "Dominant FGFR3 gain-of-function (not AR IFT-B2); thanatophoric has curved femurs, macrocephaly, cloverleaf skull — absent in SRTD10"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== SRTD10 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("Thorax:", b["thorax_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
