"""
CLUAP1 Short-Rib Thoracic Dysplasia 13 (SRTD13) — Jeune Asphyxiating Thoracic Dystrophy Type 13
===================================================================================================
Primary Gene : CLUAP1 (*616470) — also known as IFT38 — 16q23.1; ~435 aa;
               Clusterin Associated Protein 1 / Intraflagellar Transport Protein 38.
               CLUAP1/IFT38 is the HUB subunit of the IFT-B2 subcomplex that directly contacts
               IFT80 (SRTD1) via its N-terminal region, and also bridges IFT-B1 and IFT-B2
               sub-complexes (IFT-B1/B2 linchpin function).
               IFT-B2 contains: IFT80 (SRTD1), IFT172 (SRTD10), IFT88, IFT57, IFT38/CLUAP1
               (SRTD13), IFT25, IFT27.
               CLUAP1 has a central domain contacting IFT80 N-terminal WD40, a linchpin region
               bridging IFT-B1 (via IFT52) and IFT-B2, and a C-terminal extension contacting
               IFT57 and IFT88.
               Loss → IFT-B2 hub disrupted → IFT-B1/B2 linkage severed → anterograde IFT
               truncated → SHORTENED CILIA (EM; same class as SRTD1/IFT80 and SRTD10/IFT172)
               → Hedgehog (Ihh/Shh) components cannot reach tip → GLI3R accumulates in
               chondrocytes → NARROW THORAX.

Disease OMIM : #616300 — Asphyxiating Thoracic Dystrophy 13 (ATD13 / SRTD13)
               Identified in 2016 (Roosing et al., J Clin Invest).
               EXTREMELY RARE: fewer than 10 families described worldwide (2016–2026).
               No separate renal-only (NPHP) or Joubert (JBTS) phenotypic allele series.
               No retinal-predominant BBS-like allele (unlike IFT172/SRTD10).
               Ciliary EM: SHORTENED cilia — same IFT-B2 class as SRTD1 (IFT80) and SRTD10 (IFT172).
Chromosome   : 16q23.1
Inheritance  : Autosomal Recessive — biallelic LOF (compound het or homozygous consanguineous)
Prevalence   : Extremely rare — fewer than 10 families worldwide as of 2026; possibly
               under-ascertained on older gene panels that did not include CLUAP1.
               No reliable prevalence estimate; SRTD13 is the rarest of the IFT-B2 SRTDs.

Protein Structure — CLUAP1/IFT38 (435 aa; IFT-B2 hub / IFT-B1-B2 linchpin)
-------------------------------------------------------------------------------
Domain 1: N-terminal region / IFT80-contact surface (aa 1–150)
           Directly contacts IFT80 (SRTD1) N-terminal WD40 domain within IFT-B2.
           Pathogenic missense hotspot in this region → IFT80-CLUAP1 interface lost;
           IFT-B2 scaffold destabilised; moderate SRTD13
Domain 2: Central linchpin region (aa 151–280)
           IFT-B1/IFT-B2 linchpin; bridges IFT-B1 (via IFT52) to IFT-B2 scaffold;
           missense here → severe: IFT-B1 and IFT-B2 separation → complete anterograde failure
Domain 3: C-terminal extension (aa 281–435)
           Contacts IFT57 and IFT88 within IFT-B2;
           hypomorphic missense here → milder phenotype (partial IFT-B2 assembly maintained)

Key pathogenic variant classes (CLUAP1):
1. IFT80-contact surface missense (aa 1–150): compound het or homozygous; moderate SRTD13
2. Central linchpin missense (aa 151–280): severe; IFT-B1/B2 uncoupling
3. Biallelic null (truncating + truncating): SRPS spectrum; perinatal lethal
4. C-terminal extension missense (aa 281–435): mild SRTD13 (hypomorphic; partial IFT57/IFT88 contact)
5. Compound het: truncating + missense: moderate to severe
"""

import random
import math

SEED = 403
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
    ('Middle Eastern / North African', 0.38),   # consanguineous homozygous — most common for rare gene
    ('European',                        0.24),   # compound het
    ('South Asian',                     0.20),   # hypomorphic C-terminal alleles
    ('East Asian',                      0.08),
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
    'Homozygous missense (IFT80-contact surface aa 1–150)',
    'Compound het missense + truncating',
    'Compound het two missense',
    'Central linchpin missense (aa 151–280) — severe',
    'Homozygous truncating (null / SRPS)',
    'Hypomorphic C-terminal extension missense (aa 281–435)',
]
allele_weights = [0.30, 0.25, 0.18, 0.12, 0.10, 0.05]

for i in range(N):
    eth    = eth_pool[i]
    allele = rng.choices(allele_classes, weights=allele_weights)[0]
    null   = 'null' in allele.lower() or 'SRPS' in allele

    thorax_sev = rng.choices(
        ['Severe (perinatal/neonatal respiratory failure)', 'Moderate', 'Mild'],
        weights=[0.20, 0.48, 0.32])[0]

    poly_status = rng.choices(
        ['Postaxial (bilateral)', 'Postaxial (unilateral)', 'Preaxial (rare)', 'None'],
        weights=[0.16, 0.20, 0.04, 0.60])[0]

    renal = rng.choices(
        ['TIN + corticomedullary cysts → ESRD', 'TIN (non-dialysis)', 'None'],
        weights=[0.06, 0.10, 0.84])[0]

    ckd_stage = (
        rng.choices(['CKD 4', 'CKD 5 / ESRD'], weights=[0.38, 0.62])[0]
        if 'ESRD' in renal or 'TIN' in renal else 'None')

    retinal = rng.choices(
        ['Rod-cone dystrophy (ERG confirmed)', 'Reduced ERG only', 'None'],
        weights=[0.05, 0.05, 0.90])[0]

    hepatic = rng.choices(
        ['CHF (ductal plate malformation)', 'Elevated LFTs only', 'None'],
        weights=[0.04, 0.03, 0.93])[0]

    veptr = rng.choices(
        ['VEPTR (growing rod)', 'MAGEC rod', 'Observation only'],
        weights=[0.34, 0.10, 0.56])[0]

    tx_renal = (
        rng.choices(['Renal transplant (done)', 'Dialysis (awaiting Tx)', 'Conservative management'],
                    weights=[0.50, 0.22, 0.28])[0]
        if 'ESRD' in renal else 'None')

    resp_mgmt = rng.choices(
        ['Mechanical ventilation (neonatal)', 'Non-invasive ventilation', 'Observation / physio'],
        weights=[0.18, 0.26, 0.56])[0]

    presentation = rng.choices(
        ['Prenatal (polyhydramnios / short limbs)', 'Neonatal respiratory failure',
         'Infant thorax + limb anomaly', 'Childhood chronic renal + skeletal'],
        weights=[0.22, 0.18, 0.40, 0.20])[0]

    misdiagnosis = rng.choices(
        ['SRTD1/IFT80 (same IFT-B2 class — shortened cilia — gene panel corrected)',
         'SRTD3/DYNC2H1 (most common SRTD — gene panel corrected)',
         'SRTD10/IFT172 (same IFT-B2 class — gene panel corrected)',
         'EVC (Ellis-van Creveld)',
         'SRTD5/WDR19 (IFT-A mistaken — gene panel corrected)',
         'None'],
        weights=[0.16, 0.12, 0.08, 0.06, 0.04, 0.54])[0]

    age_dx = rng.choices(
        ['0–1 yr (neonatal)', '2–5 yr (infant)', '6–10 yr (child)', '11–16 yr (teen)'],
        weights=[0.36, 0.30, 0.20, 0.14])[0]

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
        "disease":    "CLUAP1 Short-Rib Thoracic Dysplasia 13 (SRTD13 / ATD13 / Jeune Syndrome)",
        "gene":       "CLUAP1/IFT38 (*616470) — 16q23.1 — 435 aa — IFT-B2 Hub / IFT-B1/B2 Linchpin (IFT80-contact)",
        "omim_gene":  "616470",
        "omim_disease": "616300",
        "chromosome": "16q23.1",
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
            "CLUAP1/IFT38 loss disrupts the IFT-B2 subcomplex at its hub. CLUAP1 is the "
            "IFT-B2 HUB subunit (435 aa): its N-terminal region directly contacts IFT80 "
            "(SRTD1) N-terminal WD40 domain, and its central linchpin region bridges IFT-B1 "
            "(via IFT52) to IFT-B2. Without CLUAP1: the IFT80-CLUAP1 interface is broken "
            "→ IFT-B2 scaffold destabilised → IFT-B1/B2 linkage severed → anterograde IFT-B "
            "train is truncated → SHORTENED CILIA (EM; same IFT-B2 class as SRTD1/IFT80 and "
            "SRTD10/IFT172) → Hedgehog signal components (SMO, Gli2/Gli3, Patched) cannot be "
            "transported to the ciliary tip for processing → GLI3R accumulates in chondrocytes "
            "→ NARROW THORAX (primary), short ribs, short limbs. Secondary: renal TIN/ESRD "
            "(~10–15% survivors), rod-cone dystrophy (~5–10%), CHF (~5%). SRTD13 is the RAREST "
            "of the IFT-B2 SRTDs (fewer than 10 families worldwide, 2016–2026); likely "
            "under-ascertained on older panels."
        ),
        "key_distinction": (
            "CLUAP1/IFT38 (SRTD13) is the RAREST IFT-B2 SRTD — fewer than 10 families "
            "described worldwide (Roosing et al., J Clin Invest 2016). It is the THIRD IFT-B2 "
            "SRTD gene after IFT80 (SRTD1, 2007) and IFT172 (SRTD10, 2015). CLUAP1's unique "
            "role as the IFT-B1/B2 LINCHPIN (bridging both sub-complexes) means its loss is "
            "mechanistically distinct from SRTD1/SRTD10 (which affect IFT-B2 assembly and "
            "tip-anchoring respectively), yet the EM ciliary phenotype is identical: SHORTENED "
            "cilia (IFT-B2 class). NO NPHP phenotype. NO Joubert MTS. NO ectodermal features. "
            "NO retinal-predominant allele series (unlike IFT172/SRTD10). Gene panel is "
            "MANDATORY — SRTD13 is clinically identical to SRTD1 and SRTD10; only molecular "
            "testing distinguishes them. Likely under-ascertained: CLUAP1 may have been absent "
            "from early SRTD panels. Under-diagnosis means true prevalence is unknown."
        ),
        "ift_b2_subunit_table": [
            {"subunit": "IFT80",       "role": "WD40 β-propeller; IFT38/CLUAP1 + IFT88 contact; IFT-B2 assembly scaffold",     "srtd": "SRTD1",  "omim_gene": "611229", "chr": "3q25.33",  "freq": "~0.5–1% SRTD (founding gene, 2007)"},
            {"subunit": "IFT172",      "role": "LARGEST IFT-B2 subunit; WD40 + TPR; kinesin-2 tip-anchoring; IFT-B2 scaffold",  "srtd": "SRTD10", "omim_gene": "607386", "chr": "2p23.3",   "freq": "~1–2% SRTD (2nd IFT-B2 gene, 2015)"},
            {"subunit": "CLUAP1/IFT38","role": "IFT-B2 HUB; contacts IFT80 N-terminal WD40; IFT-B1/B2 linchpin (bridges IFT52)","srtd": "SRTD13", "omim_gene": "616470", "chr": "16q23.1",  "freq": "RAREST IFT-B2 SRTD (<10 families, 2016)"},
            {"subunit": "IFT88",       "role": "IFT-B2 structural scaffold; contacts IFT80 C-terminal; Hedgehog signalling",    "srtd": "—",      "omim_gene": "600595", "chr": "13q12.11", "freq": "No SRTD reported (may be lethal)"},
            {"subunit": "IFT57",       "role": "IFT-B2 coiled-coil; contacts IFT-B1 via IFT52 bridge; CLUAP1 C-terminal",      "srtd": "—",      "omim_gene": "605502", "chr": "3p21.31",  "freq": "No SRTD reported"},
            {"subunit": "IFT25",       "role": "Small IFT-B2 subunit; Hedgehog signalling; binds IFT27 (Rab-like GTPase)",      "srtd": "—",      "omim_gene": "613286", "chr": "17q21.2",  "freq": "Retinal only (not SRTD)"},
            {"subunit": "IFT27",       "role": "Rab-like GTPase; IFT-B2 membrane-signalling; pairs with IFT25",                 "srtd": "—",      "omim_gene": "615504", "chr": "22q12.3",  "freq": "Bardet-Biedl-like (not SRTD)"},
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
            {"variant": "p.Arg99His (c.296G>A) — IFT80-contact surface (aa 80–120; N-terminal IFT80 WD40 binding zone) — European compound het — moderate SRTD13", "n": rng.randint(3, 5)},
            {"variant": "p.Glu220Lys (c.658G>A) — central linchpin region (aa 200–250; IFT-B1/B2 bridge; IFT52 contact) — Middle Eastern homozygous — moderate-severe SRTD13", "n": rng.randint(3, 5)},
            {"variant": "p.Gln298Ter (c.892C>T) — truncating null (central/C-terminal boundary) — SRPS spectrum — pan-ethnic", "n": rng.randint(2, 4)},
            {"variant": "p.Ala312Val (c.935C>T) — C-terminal extension (IFT57/IFT88 contact zone, aa 290–340) — South Asian hypomorphic — mild SRTD13", "n": rng.randint(2, 3)},
            {"variant": "p.Lys131Glu (c.391A>G) — IFT80-contact surface (aa 110–150; IFT80 N-WD40 interface) — MENA homozygous — moderate SRTD13", "n": rng.randint(1, 3)},
        ],
    }


def get_definitions():
    return {
        "gene_card": {
            "gene":             "CLUAP1 — Clusterin Associated Protein 1 (also IFT38)",
            "also_known_as":    "IFT38 (prior name; same protein); CLUAP1 = Clusterin Associated Protein 1; historical alias qilin (zebrafish ortholog); SRTD13 gene; IFT-B2 hub/linchpin subunit",
            "omim_gene":        "*616470",
            "chromosome":       "16q23.1",
            "protein_size":     "435 aa",
            "protein_class":    "IFT-B2 subcomplex — IFT-B2 HUB / IFT-B1/B2 linchpin; N-terminal IFT80-contact; central IFT-B1/B2 bridge (IFT52 contact); C-terminal IFT57/IFT88 contacts",
            "domain_structure": "N-terminal IFT80-contact surface (aa 1–150; directly contacts IFT80 N-terminal WD40 domain; pathogenic missense hotspot; moderate SRTD13 if disrupted) + Central linchpin region (aa 151–280; IFT-B1/IFT-B2 bridge via IFT52; loss → IFT-B1 + IFT-B2 uncoupling; severe phenotype) + C-terminal extension (aa 281–435; IFT57 and IFT88 contacts within IFT-B2; hypomorphic missense → milder SRTD13; partial IFT-B2 assembly maintained)",
            "function":         "IFT-B2 hub subunit; directly contacts IFT80 (SRTD1) N-terminal WD40 domain; central linchpin role bridging IFT-B1 (core) and IFT-B2 (distal) sub-complexes via IFT52; C-terminal contacts IFT57 and IFT88 within IFT-B2; loss → IFT-B2 hub disrupted → IFT-B1/B2 linkage severed → anterograde IFT train truncated → shortened cilia → Hedgehog processing failure",
            "inheritance":      "Autosomal Recessive — biallelic LOF",
            "key_contacts":     "IFT80 (SRTD1; N-terminal IFT80-contact surface — direct and critical); IFT52 (central linchpin region; IFT-B1/B2 bridge); IFT57 (C-terminal extension; within IFT-B2); IFT88 (C-terminal extension; within IFT-B2)",
            "expression":       "Primary non-motile cilia (9+0); chondrocytes (growth plate); renal tubular epithelium; retinal photoreceptor connecting cilium (lower sensitivity than IFT172); zebrafish qilin ortholog — key cilia model organism gene",
            "rarity_note":      "SRTD13 is the RAREST IFT-B2 SRTD: fewer than 10 families worldwide as of 2026 (Roosing et al. 2016). Under-ascertainment likely on older panels not including CLUAP1. No separate NPHP or Joubert phenotypic allele described. True prevalence unknown.",
            "discovery_note":   "Roosing et al. (J Clin Invest, 2016) — zebrafish qilin/cluap1 mutant model confirmed loss-of-cilia phenotype; human SRTD13 patient cohort identified via whole-exome sequencing; simultaneous zebrafish functional validation is a hallmark of the publication.",
            "ift_b_complex":    "IFT-B anterograde complex has two sub-complexes: IFT-B1 (core; kinesin-docking base) and IFT-B2 (distal; tip-proximal). CLUAP1/IFT38 is the LINCHPIN that physically bridges IFT-B1 and IFT-B2 — uniquely, its loss can uncouple both sub-complexes from each other.",
        },
        "disease_card": {
            "omim_disease":     "#616300 — Asphyxiating Thoracic Dystrophy 13 (ATD13 / SRTD13)",
            "full_name":        "Short-Rib Thoracic Dysplasia 13 With or Without Polydactyly",
            "alternative_names": "SRTD13; Asphyxiating Thoracic Dystrophy 13 (ATD13); Jeune syndrome type 13",
            "primary_finding":  "NARROW THORAX — pathognomonic; neonatal respiratory failure in severe (null) alleles",
            "polydactyly":      "~35–40% (postaxial most common; slightly less frequent than SRTD1/IFT80 ~50%)",
            "renal":            "TIN + corticomedullary cysts → ESRD in ~10–15% survivors; NO NPHP phenotype",
            "retinal":          "Rod-cone dystrophy in ~5–10% (lower frequency than IFT172/SRTD10 and similar to IFT80/SRTD1); NO retinal-predominant allele series",
            "hepatic_chf":      "Ductal plate malformation → CHF in ~5%",
            "situs_inversus":   "ABSENT — primary non-motile cilia (9+0); not nodal motile cilia",
            "joubert_mts":      "ABSENT — no Molar Tooth Sign described for CLUAP1",
            "ectodermal":       "ABSENT — no sparse hair, hypodontia, or dystrophic nails (ectodermal is unique to WDR19/SRTD5)",
            "srps_spectrum":    "Biallelic null alleles → SRPS (perinatal lethal)",
            "renal_transplant": "CURATIVE — cell-autonomous IFT defect; no post-Tx recurrence",
            "surgical_treatment": "VEPTR / MAGEC growing rods (primary thoracic intervention)",
            "prevalence":       "Extremely rare — fewer than 10 families worldwide (2016–2026); rarest IFT-B2 SRTD; likely under-ascertained",
            "under_ascertainment_note": "CLUAP1 was not on early SRTD gene panels. A significant proportion of 'gene-panel-negative' SRTDs may harbour CLUAP1 variants. Updated comprehensive SRTD panels now include CLUAP1.",
        },
        "diagnostic_workup": [
            "1. Prenatal USS: short ribs, narrow thorax, short limbs, polyhydramnios → suspect SRTD",
            "2. Postnatal skeletal survey: horizontal ribs, narrow bell-shaped thorax, shortened tubular bones; SHORTENED cilia (EM) — IFT-B2 class (same as IFT80/SRTD1 and IFT172/SRTD10); distinct from IFT-A short stubby and dynein-2 club/bulging tip",
            "3. GENE PANEL (mandatory and must include CLUAP1): CLUAP1 must be included alongside DYNC2H1, WDR19, IFT140, WDR35, WDR60, WDR34, TTC21B, TCTEX1D2, DYNC2LI1, IFT80, IFT172 — SRTD13 is clinically identical to all other SRTDs; gene panel is the ONLY way to distinguish; CLUAP1 was absent from older panels → under-ascertainment",
            "4. WES + CNV if panel negative — CLUAP1 16q23.1; CNV deletion analysis mandatory for large gene deletions",
            "5. Ciliopathy EM (skin/fibroblast biopsy): SHORTENED cilia — IFT-B2 failure (same class as SRTD1/IFT80 and SRTD10/IFT172); cannot distinguish SRTD1, SRTD10, or SRTD13 by EM alone; gene panel always required",
            "6. Renal USS: corticomedullary cysts; creatinine + GFR annually; NO NPHP allele variant — all renal disease is secondary SRTD13 skeletal phenotype",
            "7. ERG from age 6 (rod-cone dystrophy ~5–10%); ophthalmology referral; lower retinal frequency than IFT172/SRTD10; NO BBS-like isolated retinal allele described",
            "8. Liver USS + APRI if hepatic signs suspected (CHF ~5%); liver biopsy if APRI elevated",
            "9. Brain MRI: NO Molar Tooth Sign expected; normal posterior fossa",
            "10. MLPA/CNV for CLUAP1 16q23.1 deletions if sequencing negative but phenotype compelling",
            "11. Update SRTD gene panel to confirm CLUAP1 is included — older panels may lack it; re-test gene-panel-negative SRTD patients from pre-2016 era with updated panel",
        ],
        "mechanism_glossary": [
            {"term": "CLUAP1/IFT38",  "definition": "Clusterin Associated Protein 1 / IFT38 — 435 aa; IFT-B2 hub subunit; N-terminal region contacts IFT80 (SRTD1) directly; central linchpin region bridges IFT-B1 and IFT-B2 via IFT52; C-terminal contacts IFT57 and IFT88 within IFT-B2. Loss → IFT-B2 hub disrupted → IFT-B1/B2 uncoupling → anterograde IFT truncated → shortened cilia."},
            {"term": "IFT-B1/B2 linchpin", "definition": "CLUAP1's unique function: physically bridges IFT-B1 (the 'core' proximal sub-complex that docks kinesin-2) to IFT-B2 (the distal tip-proximal sub-complex). This IFT-B1/B2 linkage maintains the integrity of the full anterograde IFT train. Unlike IFT80 (SRTD1; IFT-B2 scaffold) and IFT172 (SRTD10; tip-anchor), CLUAP1 loss uncouples the entire B1/B2 assembly."},
            {"term": "IFT-B2 subcomplex", "definition": "The distal (tip-proximal) half of the IFT-B anterograde complex: IFT80 (SRTD1) + IFT172 (SRTD10) + IFT88 + IFT57 + IFT38/CLUAP1 (SRTD13) + IFT25 + IFT27. IFT-B2 is the 'tip-proximal' sub-complex that positions Hedgehog components for ciliary tip processing."},
            {"term": "IFT-B complex",   "definition": "The anterograde IFT complex, driven by kinesin-2 (KIF3A/KIF3B/KAP) from the ciliary base to the tip. Consists of IFT-B1 (core/proximal) and IFT-B2 (distal/tip-proximal). CLUAP1 is the linchpin connecting IFT-B1 to IFT-B2."},
            {"term": "Shortened cilia", "definition": "EM finding in IFT-B2 SRTDs (SRTD1/IFT80, SRTD10/IFT172, SRTD13/CLUAP1): cilia are shortened because IFT-B anterograde delivery to the tip is impaired — axonemal building blocks cannot be delivered. Distinct from IFT-A short stubby (IFT-B cannot enter at base) and dynein-2 club/bulging tip (retrograde failure)."},
            {"term": "Anterograde IFT", "definition": "Kinesin-2-driven transport from the ciliary base to the tip. CLUAP1 links IFT-B1 (kinesin-2 docking) to IFT-B2 (Hedgehog cargo carriers), ensuring the entire IFT-B train can transit from base to tip as a unit."},
            {"term": "Hedgehog (Ihh/Shh)", "definition": "Growth plate signalling requiring ciliary tip delivery. CLUAP1 loss → anterograde IFT failure → Gli components cannot reach the tip → GLI3R accumulates → Ihh/Shh response impaired in chondrocytes → short ribs, narrow thorax, short limbs."},
            {"term": "Under-ascertainment", "definition": "SRTD13/CLUAP1 is the rarest IFT-B2 SRTD (<10 families 2016–2026), partly because CLUAP1 was absent from pre-2016 SRTD gene panels. Some 'panel-negative' SRTD cases from the pre-2016 era may harbour CLUAP1 variants. Updated panels now include CLUAP1."},
            {"term": "Cell-autonomous",  "definition": "The IFT-B2 defect is intrinsic to the cell. After renal transplantation, the donor kidney (CLUAP1+) functions normally — CURATIVE for renal disease; no recurrence post-transplant."},
        ],
        "key_variants": [
            {"variant": "p.Arg99His",   "domain": "IFT80-contact surface (aa 80–120; N-terminal IFT80 WD40 binding zone)", "consequence": "Disrupts CLUAP1-IFT80 interface within IFT-B2; IFT-B2 hub function impaired; European compound het; moderate SRTD13",  "ethnicity": "European"},
            {"variant": "p.Glu220Lys",  "domain": "Central linchpin (aa 200–250; IFT-B1/B2 bridge; IFT52 contact)",       "consequence": "IFT-B1/B2 linkage disrupted; IFT-B train uncoupled; Middle Eastern homozygous; moderate-severe SRTD13",                     "ethnicity": "Middle Eastern (homozygous)"},
            {"variant": "p.Gln298Ter",  "domain": "Truncating null (central/C-terminal boundary)",                          "consequence": "Premature stop; biallelic null → SRPS spectrum; pan-ethnic",                                                                "ethnicity": "Pan-ethnic"},
            {"variant": "p.Ala312Val",  "domain": "C-terminal extension (IFT57/IFT88 contact zone, aa 290–340)",            "consequence": "Hypomorphic; South Asian; milder SRTD13; partial IFT57/IFT88 contacts maintained within IFT-B2",                            "ethnicity": "South Asian"},
            {"variant": "p.Lys131Glu",  "domain": "IFT80-contact surface (aa 110–150; IFT80 N-WD40 interface)",            "consequence": "IFT80 N-terminal WD40 binding surface disrupted; MENA homozygous; moderate SRTD13",                                          "ethnicity": "MENA"},
        ],
        "treatment_summary": [
            "1. NARROW THORAX (primary): VEPTR (vertical expandable prosthetic titanium rib) or MAGEC growing rods — first-line; serial expansion every 6 months; same as all SRTDs",
            "2. Respiratory: neonatal mechanical ventilation if severe null alleles; CPAP/NIV for moderate; wean as thorax expands with VEPTR",
            "3. Renal: annual creatinine + GFR; USS for cyst progression; ACEi/ARB for proteinuria; dialysis bridge; RENAL TRANSPLANT IS CURATIVE — no recurrence (cell-autonomous). NOTE: NO NPHP allele — all SRTD13 renal disease is secondary skeletal phenotype",
            "4. Retinal: annual ERG from age 6; ophthalmology; low vision support (~5–10%); NO BBS-like isolated retinal allele described for CLUAP1; standard SRTD retinal surveillance protocol",
            "5. Hepatic: APRI + USS annually if suspected CHF; hepatology referral; avoid hepatotoxic drugs",
            "6. Genetics: cascade testing of parents + siblings; NO NPHP or Joubert allele-severity counselling; standard AR 25% recurrence risk; prenatal/PGT; COUNSEL re: under-ascertainment — SRTD13 may appear as 'panel-negative SRTD' on older panels not including CLUAP1",
            "7. MDT: paediatric orthopaedics (VEPTR), nephrology, respiratory, ophthalmology, genetics, hepatology",
            "8. Panel update: ensure current SRTD gene panel includes CLUAP1/IFT38 — previously absent from older panels; re-testing gene-panel-negative SRTD index cases from pre-2016 era is recommended",
        ],
        "ddx_table": [
            {"disease": "SRTD1 (IFT80)",      "key_difference": "SAME IFT-B2 COMPLEX — IFT80 is the direct CLUAP1 binding partner (N-terminal WD40 contact); shortened cilia (same EM class); gene panel is the ONLY way to distinguish SRTD13 from SRTD1; IFT80 rarer (~0.5%) but more common than CLUAP1"},
            {"disease": "SRTD10 (IFT172)",    "key_difference": "SAME IFT-B2 COMPLEX — IFT172 is the LARGEST IFT-B2 subunit (tip-anchor); both give shortened cilia (same EM class); IFT172 has dual phenotype (BBS-like retinal) absent in SRTD13; gene panel mandatory"},
            {"disease": "SRTD3 (DYNC2H1)",    "key_difference": "Most common SRTD (~50%); DYNEIN-2 (not IFT-B2); CLUB cilia on EM (IFT-B stranded at tip) — distinct from SRTD13 shortened cilia; gene panel resolves"},
            {"disease": "SRTD5 (WDR19)",      "key_difference": "Most common IFT-A SRTD; IFT-A complex (not IFT-B2); short STUBBY cilia; ectodermal features UNIQUE to WDR19 (absent in SRTD13); gene panel required"},
            {"disease": "SRTD9 (IFT140)",     "key_difference": "IFT-A core scaffold (not IFT-B2); short stubby cilia (IFT-A); no ectodermal; gene panel required"},
            {"disease": "SRTD4 (TTC21B)",     "key_difference": "IFT-A TPR adaptor (not IFT-B2); short stubby cilia; THREE phenotypes (SRTD4/NPHP12/JBTS12); CLUAP1 has NO NPHP/Joubert allele"},
            {"disease": "EVC (Ellis-van Creveld)", "key_difference": "CHD present in ~60% EVC; EVC ectodermal features (teeth/nails/hair); EVC is hedgehog-cilia pathway but not IFT-B2"},
            {"disease": "Thanatophoric dysplasia", "key_difference": "Dominant FGFR3 gain-of-function (not AR IFT-B2); curved femurs, macrocephaly, cloverleaf skull — absent in SRTD13"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== SRTD13 Overview ===")
    print(json.dumps(get_overview(), indent=2))
    print("\n=== Breakdown (sample) ===")
    b = get_breakdown()
    print("Thorax:", b["thorax_distribution"])
    print("Alleles:", b["allele_class_summary"])
    print("\n=== Definitions (gene card) ===")
    print(json.dumps(get_definitions()["gene_card"], indent=2))
