#!/usr/bin/env python3
"""OPA1 — Autosomal Dominant Optic Atrophy (ADOA / Kjer Disease / OPA1-Plus).

OPA1-Related Mitochondrial Disease — AD haploinsufficiency (optic nerve fusion/cristae):
  Autosomal Dominant Optic Atrophy (ADOA / Kjer Disease)
  OMIM Disease #165500 (ADOA) / #125250 (OPA1-Plus / Behr-allelic)
  OMIM Gene OPA1 *605290

OPA1 (OPA1 Mitochondrial Dynamin-Like GTPase; 3q29; 960 aa) encodes the only IMM-anchored
dynamin-like GTPase in human cells. OPA1 is essential for two processes:
  (1) Inner mitochondrial membrane (IMM) fusion — OPA1 tethers and fuses IMMs of adjacent
      mitochondria; haploinsufficiency → fragmented mitochondria → OXPHOS supercomplex
      destabilisation → increased superoxide → retinal ganglion cell (RGC) axon degeneration
  (2) Cristae remodelling — OPA1 oligomers tighten cristae junctions → compartmentalises
      cytochrome c within cristae → limits apoptosis; OPA1 LOF → open cristae → cytochrome c
      release → amplified apoptosis under bioenergetic stress

Most common hereditary optic neuropathy worldwide (prevalence 1:30,000–1:50,000).

TWO CLINICAL TIERS OF OPA1 DISEASE:
  ADOA (pure optic atrophy) — ~80% of OPA1 patients:
    Autosomal dominant, haploinsufficiency; early childhood visual loss (onset ~6 yr);
    insidious, slowly progressive; temporal optic disc pallor → full optic atrophy;
    bilateral central/centrocaecal scotoma; tritanopia (blue-yellow dyschromatopsia);
    final VA often 6/60 to counting fingers; NON-arteritic (vs NAION).
  OPA1-Plus (~20% of OPA1 patients):
    Multisystem mitochondrial disease beyond optic nerve; mtDNA multiple deletions
    accumulate in post-mitotic muscle (same molecular pattern as adPEO series);
    SNHL ≥50%, ataxia ~40%, peripheral neuropathy ~35%, myopathy ~35%, PEO ~25%,
    spastic paraplegia ~20%, cognitive decline ~20%, parkinsonism ~15%;
    Amati-Bonneau 2008 Brain first described OPA1-Plus mechanism.

DOMINANT NEGATIVE vs HAPLOINSUFFICIENCY:
  The vast majority of ADOA-causing OPA1 mutations are LOF (frameshift, nonsense, splice,
  large deletion) causing haploinsufficiency — 50% residual OPA1 is insufficient for
  normal RGC mitochondrial network maintenance. Missense mutations in the GTPase domain
  (e.g. p.R445H; OPA1-Plus hotspot) cause dominant negative with more severe multisystem
  disease. Genotype-phenotype: GTPase domain missense → OPA1-Plus; LOF frameshift/nonsense
  → pure ADOA. Haploinsufficiency = enough OPA1 for most tissues (high fusion reserve)
  but NOT retinal ganglion cell axons (highest OXPHOS demand density per unit length).

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. ETHAMBUTOL = ABSOLUTE CONTRAINDICATION — ethambutol directly inhibits copper-
     dependent mitochondrial Complex IV in optic nerve axons; OPA1 RGCs already operating
     at OXPHOS threshold; ethambutol → acute-on-chronic optic neuropathy → rapid vision
     loss; NEVER use ethambutol in any OPA1 patient (including asymptomatic carriers)
  2. LINEZOLID = ABSOLUTE CONTRAINDICATION — linezolid inhibits mitochondrial 16S rRNA
     (28S rRNA equivalent) → blocks mitochondrial protein synthesis → acute optic
     neuropathy; 100% reversible if caught early but permanent if prolonged; NEVER use
     in OPA1; use alternative antibiotics; if unavoidable, ophthalmology weekly
  3. CHLORAMPHENICOL = AVOID — same mitochondrial ribosome inhibition mechanism as
     linezolid; systemic (not topical) chloramphenicol only; topical eye drops safe
  4. TOBACCO = ABSOLUTE CONTRAINDICATION (environmental) — tobacco toxins (cyanide,
     acrolein) cause mitochondrial ETC inhibition + OXPHOS uncoupling in optic nerve;
     OPA1 + tobacco = dramatically accelerated visual loss (synergistic); smoking
     cessation MANDATORY; passive smoke exposure also risk factor
  5. ALCOHOL = AVOID — ethanol metabolite acetaldehyde inhibits Complex I + depletes
     mitochondrial glutathione; OPA1 RGCs vulnerable; binge drinking → acute worsening
     episodes; counsel strongly against alcohol
  6. OPA1-PLUS detection — CRITICAL: re-examine all OPA1 patients every 2 years for
     multisystem features; SNHL (audiogram annually), myopathy (CK), neuropathy (NCS),
     ataxia (SARA scale), PEO (Hess chart); OPA1-Plus diagnosis changes management
     (add AED safety considerations, VPA caution, propofol caution)
  7. mtDNA MULTIPLE DELETIONS in OPA1-Plus — long-range PCR on muscle shows same
     deletion pattern as adPEO series (POLG2/DNA2/SLC25A4/TWNK-PEOA3); mechanism:
     OPA1 LOF → mitochondrial fragmentation → altered replication fork geometry →
     deletion accumulation; links OPA1-Plus mechanistically to adPEO spectrum
  8. IDEBENONE (Raxone) — Level C evidence in ADOA (unlike LHON where Level B approved
     EU); short-chain CoQ10 analogue; bypasses Complex I dysfunction; trials ongoing;
     best evidence for recovery if started early in the visual loss episode
  9. AMIODARONE = CAUTION — amiodarone causes drug-induced optic neuropathy in the
     general population; amplified in OPA1; use only if cardiology compelling indication;
     ophthalmology pre-treatment and quarterly monitoring
 10. TRITANOPIA (blue-yellow dyschromatopsia) — CHARACTERISTIC colour vision defect;
     contrast with red-green defect of LHON (often mistaken for ADOA); formal colour
     vision (Farnsworth-Munsell 100-hue test) distinguishes ADOA from LHON
 11. LENADOGENE NOLPARVOVEC (Lumevoq) — intravitreal AAV2 MT-ND4 gene therapy;
     approved EU 2021 for LHON (NOT OPA1 ADOA); OPA1-targeted intravitreal gene therapy
     in clinical trials (NCT02418389); NOT currently standard of care for OPA1 ADOA
 12. DELETTRE 2000 / ALEXANDER 2000 — dual landmark papers (NatGenet) simultaneously
     identified OPA1 as the ADOA gene in European families; Delettre used French Breton
     families (p.E364K-French-founder); Alexander used British families
 13. AMIODARONE, VPA, LINEZOLID, ETHAMBUTOL, CHLORAMPHENICOL, TOBACCO, ALCOHOL —
     the "OPA1 Danger Seven"; must be on every OPA1 patient's medical alert card
 14. GENETIC COUNSELLING — AD, 50% offspring risk; penetrance ~98%; variable
     expressivity (vision loss 6/6 to count fingers within family); asymptomatic
     carriers UNCOMMON; genetic testing of at-risk family members important to
     pre-empt prescription of danger drugs
 15. VPA CAUTION (not absolute CI unless OPA1-Plus with myopathy) — VPA CoA
     sequestration is mainly a risk when there is skeletal muscle OXPHOS deficiency
     (COX-negative fibres in OPA1-Plus); pure ADOA patients: VPA use requires specialist
     review but not absolute CI; OPA1-Plus patients with myopathy: VPA AVOID

OPA1 MOLECULAR BIOLOGY:
OPA1 (960 aa; 3q29; UniProt O60313) is the human homologue of yeast Mgm1.
Eight isoforms from alternative splicing of exons 4, 4b, and 5b; all are IMM-anchored.
Two functional pools:
  Long OPA1 (L-OPA1): IMM-tethered via N-terminal TM anchor; drives IMM tethering
  Short OPA1 (S-OPA1): generated by cleavage at S1 (OMA1/YME1L) + S2 (YME1L) sites;
    required for IMM fusion trans-complementation; L + S together needed for fusion
Key domains:
  N-terminal TM (aa ~1–60): IMM anchor; import + topology determinant
  Coiled-coil domain 1 (aa ~100–200): self-assembly nucleus
  GTPase domain (aa ~360–600): GTP hydrolysis drives membrane curvature; OPA1-Plus
    missense hotspot (p.R445H, p.R445C in GTPase domain)
  Middle domain (aa ~600–700): dynamin-like switch
  GED (GTPase effector domain, aa ~800–960): GTPase stimulation + assembly
Cristae function:
  OPA1 oligomers (ring/filament) tighten cristae junctions (nm-scale IMM invagination
  openings); normal cristae: cytochrome c sequestered inside; OPA1 LOF: open cristae →
  cytochrome c escapes IMS → lowered apoptotic threshold → RGC loss amplified under
  physiological stress
"""

from __future__ import annotations

import random
from typing import Any

SEED = 581          # reproducible 40-patient cohort (OPA1-ADOA/OPA1-Plus)
N_PATIENTS = 40


def _rng() -> random.Random:
    return random.Random(SEED)


# ── cohort builder ──────────────────────────────────────────────────────────

def _build_cohort(rng: random.Random) -> list[dict]:
    """Generate a 40-patient OPA1-ADOA / OPA1-Plus cohort (seed-581)."""
    VARIANT_POOL = [
        "p.R445H-GTPase-OPA1Plus-hotspot",
        "p.R290Q-GTPase-ADOA-pure",
        "c.2708-2711del-frameshift-LOF-ADOA",
        "p.I382M-GTPase-ADOA-mild",
        "Exon-11-15-deletion-LOF",
        "p.G488R-GTPase-OPA1Plus",
        "p.E364K-French-Breton-founder-ADOA",
        "p.K468R-GTPase-mild",
        "Other-LOF-frameshift-ADOA",
        "Other-GTPase-missense-OPA1Plus",
    ]
    VARIANT_WEIGHTS = [0.18, 0.14, 0.13, 0.10, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05]

    TIER_POOL = ["ADOA-pure", "OPA1-Plus"]
    TIER_WEIGHTS = [0.80, 0.20]

    patients = []
    for i in range(N_PATIENTS):
        tier = rng.choices(TIER_POOL, weights=TIER_WEIGHTS, k=1)[0]
        is_plus = (tier == "OPA1-Plus")

        # Optic atrophy: 100% in ADOA
        optic_atrophy = True
        # SNHL: 50% in OPA1-Plus, 5% in pure ADOA
        snhl = rng.random() < (0.55 if is_plus else 0.06)
        # Ataxia: 40% in OPA1-Plus
        ataxia = rng.random() < (0.42 if is_plus else 0.04)
        # Myopathy: 35% in OPA1-Plus
        myopathy = rng.random() < (0.38 if is_plus else 0.03)
        # Peripheral neuropathy: 35% in OPA1-Plus
        neuropathy = rng.random() < (0.36 if is_plus else 0.03)
        # PEO: 25% in OPA1-Plus
        peo = rng.random() < (0.28 if is_plus else 0.0)
        # Spastic paraplegia: 20% OPA1-Plus
        paraplegia = rng.random() < (0.22 if is_plus else 0.02)
        # Cognitive: 20% OPA1-Plus
        cognitive = rng.random() < (0.20 if is_plus else 0.02)
        # Parkinsonism: 15% OPA1-Plus
        parkinsonism = rng.random() < (0.16 if is_plus else 0.01)
        # COX-negative fibres (muscle mtDNA deletions): 90% OPA1-Plus
        cox_negative = rng.random() < (0.90 if is_plus else 0.0)

        onset = round(rng.gauss(7, 3.5), 1) if not is_plus else round(rng.gauss(6, 3.0), 1)
        onset = max(1.0, min(35.0, onset))
        dx_delay = round(rng.gauss(8, 4.0), 1)
        dx_delay = max(1.0, min(20.0, dx_delay))

        va_right = round(rng.gauss(0.18, 0.12), 2)
        va_right = max(0.02, min(0.7, va_right))
        va_left = round(rng.gauss(0.17, 0.11), 2)
        va_left = max(0.02, min(0.7, va_left))

        ck_uln = round(rng.gauss(1.3, 0.5), 1) if is_plus else round(rng.gauss(0.9, 0.2), 1)
        ck_uln = max(0.5, min(5.0, ck_uln))

        variant = rng.choices(VARIANT_POOL, weights=VARIANT_WEIGHTS, k=1)[0]

        patients.append({
            "id": f"OPA1-P{i+1:02d}",
            "tier": tier,
            "age_onset_years": onset,
            "dx_delay_yr": dx_delay,
            "optic_atrophy": optic_atrophy,
            "snhl": snhl,
            "ataxia": ataxia,
            "myopathy": myopathy,
            "peripheral_neuropathy": neuropathy,
            "peo": peo,
            "spastic_paraplegia": paraplegia,
            "cognitive_decline": cognitive,
            "parkinsonism": parkinsonism,
            "cox_negative_fibres": cox_negative,
            "va_right": va_right,
            "va_left": va_left,
            "ck_x_uln": ck_uln,
            "variant": variant,
        })
    return patients


def get_overview() -> dict[str, Any]:
    rng = _rng()
    patients = _build_cohort(rng)

    n_optic = sum(1 for p in patients if p["optic_atrophy"])
    n_snhl = sum(1 for p in patients if p["snhl"])
    n_ataxia = sum(1 for p in patients if p["ataxia"])
    n_myopathy = sum(1 for p in patients if p["myopathy"])
    n_neuropathy = sum(1 for p in patients if p["peripheral_neuropathy"])
    n_peo = sum(1 for p in patients if p["peo"])
    n_para = sum(1 for p in patients if p["spastic_paraplegia"])
    n_cognitive = sum(1 for p in patients if p["cognitive_decline"])
    n_parkinson = sum(1 for p in patients if p["parkinsonism"])
    n_plus = sum(1 for p in patients if p["tier"] == "OPA1-Plus")

    avg_onset = round(sum(p["age_onset_years"] for p in patients) / N_PATIENTS, 1)

    return {
        "gene": "OPA1",
        "protein": "OPA1 Mitochondrial Dynamin-Like GTPase — 960 aa",
        "disease": (
            "OPA1-Related Mitochondrial Disease — "
            "Autosomal Dominant Optic Atrophy (ADOA / Kjer Disease) "
            "and OPA1-Plus Multisystem Syndrome"
        ),
        "omim_gene": "*605290",
        "omim_disease": (
            "#165500 (ADOA — autosomal dominant optic atrophy / Kjer disease; "
            "Delettre 2000 NatGenet + Alexander 2000 NatGenet); "
            "#125250 (OPA1-Plus / Behr-allelic multisystem; Amati-Bonneau 2008 Brain)"
        ),
        "chromosome": "3q29",
        "inheritance": (
            "Autosomal Dominant (AD) — haploinsufficiency (frameshift/nonsense/large deletion = pure ADOA); "
            "dominant-negative GTPase missense (p.R445H, p.G488R) = OPA1-Plus multisystem; "
            "penetrance ~98%; variable expressivity (6/6 to CF within same family); "
            "50% offspring risk; NO maternal inheritance (nuclear gene; CRITICAL DDx from LHON)"
        ),
        "onset": (
            f"Childhood — mean {avg_onset} years (range 1–35 years); "
            "insidious onset of visual loss; often detected at school vision screening; "
            "contrast with LHON (subacute adult onset) and Kjer disease historical description"
        ),
        "mechanism": (
            "Heterozygous OPA1 LOF (frameshift/nonsense/splice = ADOA pure) or dominant-negative "
            "GTPase missense (= OPA1-Plus) → 50% or less residual OPA1 → (1) insufficient IMM "
            "fusion → fragmented mitochondrial network in retinal ganglion cells (RGCs) → "
            "disrupted OXPHOS supercomplexes (Complex I/III/IV) → increased ROS → RGC axon "
            "degeneration → temporal optic disc pallor → bilateral optic atrophy; "
            "(2) open cristae junctions (OPA1 oligomers normally tighten cristae) → "
            "cytochrome c escape → lowered apoptotic threshold → amplified RGC death; "
            "(3) OPA1-Plus: GTPase missense → additional OXPHOS complex instability → "
            "mtDNA multiple deletions accumulate in post-mitotic skeletal muscle → "
            "COX-negative fibres → multisystem disease (SAME molecular mechanism as adPEO series)"
        ),
        "mtdna_pattern": (
            "OPA1-Plus: mtDNA multiple deletions (long-range PCR, muscle required) — "
            "IDENTICAL molecular fingerprint to adPEO series (POLG2/DNA2/SLC25A4/TWNK-PEOA3); "
            "normal copy number (deletion NOT depletion); blood unreliable (tissue-specific). "
            "Pure ADOA: NO deletions, NO depletion; nuclear gene haploinsufficiency mechanism only. "
            "This links OPA1-Plus mechanistically to the adPEO series despite a primary optic nerve phenotype."
        ),
        "key_labs": [
            "Full-field visual acuity (best-corrected, Snellen; VA often 6/36–6/60; record both eyes separately)",
            "Colour vision (Farnsworth-Munsell 100-hue or Ishihara): tritanopia (blue-yellow axis — DDx from LHON red-green)",
            "Humphrey visual fields: central/centrocaecal scotoma (bilateral); document MD and PSD trends",
            "OCT (optical coherence tomography): temporal RNFL thinning (temporal sector first); macular ganglion cell layer",
            "Fundus photography: temporal optic disc pallor → full optic atrophy; cup/disc ratio typically normal (<0.7)",
            "OPA1 molecular panel: NGS (point mutations) + del/dup analysis (MLPA/array-CGH for exonic deletions ~10%)",
            "Audiogram annually (SNHL in OPA1-Plus; progressive sensorineural, bilateral)",
            "Muscle biopsy + mtDNA long-range PCR (OPA1-Plus screen): COX-negative fibres + multiple deletions if plus-form",
            "CK: mildly elevated in OPA1-Plus myopathy (<3–5× ULN); normal in pure ADOA",
            "NCS/EMG: axonal sensory neuropathy in OPA1-Plus peripheral neuropathy",
            "Brain MRI: cerebellar atrophy if ataxia (OPA1-Plus); optic nerve signal change on fat-suppressed T2",
            "VEP (pattern-reversal): prolonged P100 latency + reduced amplitude (optic nerve conduction delay)",
        ],
        "kpis": [
            {"label": "Optic Atrophy (CARDINAL)", "value": f"{n_optic}/{N_PATIENTS} ({round(n_optic/N_PATIENTS*100)}%)", "color": "#1b5e20"},
            {"label": "OPA1-Plus (multisystem)", "value": f"{n_plus}/{N_PATIENTS} ({round(n_plus/N_PATIENTS*100)}%)", "color": "#2e7d32"},
            {"label": "SNHL (OPA1-Plus)", "value": f"{n_snhl}/{N_PATIENTS} ({round(n_snhl/N_PATIENTS*100)}%)", "color": "#388e3c"},
            {"label": "Ataxia (OPA1-Plus)", "value": f"{n_ataxia}/{N_PATIENTS} ({round(n_ataxia/N_PATIENTS*100)}%)", "color": "#43a047"},
            {"label": "Myopathy + COX-neg", "value": f"{n_myopathy}/{N_PATIENTS} ({round(n_myopathy/N_PATIENTS*100)}%)", "color": "#4caf50"},
            {"label": "Neuropathy (OPA1-Plus)", "value": f"{n_neuropathy}/{N_PATIENTS} ({round(n_neuropathy/N_PATIENTS*100)}%)", "color": "#66bb6a"},
            {"label": "PEO (OPA1-Plus)", "value": f"{n_peo}/{N_PATIENTS} ({round(n_peo/N_PATIENTS*100)}%)", "color": "#81c784"},
            {"label": "Parkinsonism (OPA1-Plus)", "value": f"{n_parkinson}/{N_PATIENTS} ({round(n_parkinson/N_PATIENTS*100)}%)", "color": "#a5d6a7"},
        ],
        "feature_bars": [
            {"label": "Optic Atrophy (bilateral, CARDINAL — 100%)", "pct": round(n_optic / N_PATIENTS * 100)},
            {"label": "OPA1-Plus (multisystem tier — GTPase missense)", "pct": round(n_plus / N_PATIENTS * 100)},
            {"label": "SNHL (predominantly in OPA1-Plus)", "pct": round(n_snhl / N_PATIENTS * 100)},
            {"label": "Cerebellar Ataxia (OPA1-Plus)", "pct": round(n_ataxia / N_PATIENTS * 100)},
            {"label": "Proximal Myopathy (OPA1-Plus)", "pct": round(n_myopathy / N_PATIENTS * 100)},
            {"label": "Peripheral Neuropathy (OPA1-Plus)", "pct": round(n_neuropathy / N_PATIENTS * 100)},
            {"label": "PEO — bilateral ophthalmoplegia (OPA1-Plus)", "pct": round(n_peo / N_PATIENTS * 100)},
            {"label": "Spastic Paraplegia (OPA1-Plus)", "pct": round(n_para / N_PATIENTS * 100)},
            {"label": "Cognitive Decline (OPA1-Plus)", "pct": round(n_cognitive / N_PATIENTS * 100)},
            {"label": "Parkinsonism (partial L-DOPA, OPA1-Plus)", "pct": round(n_parkinson / N_PATIENTS * 100)},
        ],
        "contraindications": [
            {
                "drug": "Ethambutol",
                "severity": "ABSOLUTE",
                "reason": (
                    "Ethambutol chelates copper → inhibits cytochrome c oxidase (Complex IV) "
                    "in optic nerve axons (highest OXPHOS demand); OPA1 haploinsufficiency means "
                    "RGCs already operating at bioenergetic threshold; ethambutol → acute-on-chronic "
                    "optic neuropathy → rapid irreversible vision loss; NEVER use ethambutol in any "
                    "OPA1 patient — including pre-symptomatic carriers; use alternative TB therapy "
                    "(rifampicin + isoniazid + pyrazinamide ± streptomycin for 2 months then RIF+INH)"
                ),
            },
            {
                "drug": "Linezolid (systemic)",
                "severity": "ABSOLUTE",
                "reason": (
                    "Linezolid inhibits mitochondrial 23S/16S rRNA (mitochondrial ribosome inhibition) "
                    "→ blocks synthesis of Complex I, III, IV, V subunits → mitochondrial protein "
                    "synthesis failure in optic nerve → acute drug-induced optic neuropathy (DION); "
                    "reversible only if drug stopped within 2–4 weeks; permanent blindness if prolonged; "
                    "amplified in OPA1 haploinsufficiency (synergistic Complex I/IV impairment); "
                    "use alternatives: daptomycin, tedizolid (shorter course), co-trimoxazole"
                ),
            },
            {
                "drug": "Tobacco (all forms)",
                "severity": "ABSOLUTE (environmental — STOP IMMEDIATELY)",
                "reason": (
                    "Tobacco combustion products include cyanide (Complex IV inhibitor), acrolein "
                    "(Complex II/III inhibitor), and superoxide generators; all amplify OXPHOS "
                    "failure in OPA1-deficient RGCs; OPA1 + active smoking = dramatically accelerated "
                    "visual loss and higher risk of reaching severe visual impairment (< 6/60); "
                    "passive smoke also hazardous; smoking cessation = single most important "
                    "modifiable factor in OPA1 management; e-cigarettes also contain acrolein — AVOID"
                ),
            },
            {
                "drug": "Chloramphenicol (systemic only)",
                "severity": "AVOID",
                "reason": (
                    "Systemic chloramphenicol inhibits mitochondrial 70S ribosome → blocks mtDNA-"
                    "encoded subunit synthesis → same mechanism as linezolid; optic neuropathy risk; "
                    "TOPICAL chloramphenicol (eye drops) is safe (negligible systemic absorption); "
                    "alternative IV antibiotics: carbapenems, piperacillin-tazobactam, aminoglycosides"
                ),
            },
            {
                "drug": "Amiodarone",
                "severity": "CAUTION (cardiology must weigh benefit vs DION risk)",
                "reason": (
                    "Amiodarone causes drug-induced optic neuropathy (DION) in ~1–2% of the general "
                    "population via mitochondrial Complex I inhibition + lipid deposition in optic nerve; "
                    "risk amplified in OPA1 patients (pre-existing RGC bioenergetic vulnerability); "
                    "ophthalmology baseline exam + quarterly monitoring mandatory if amiodarone unavoidable; "
                    "alternative anti-arrhythmics: flecainide (atrial only), sotalol, dronedarone"
                ),
            },
            {
                "drug": "Alcohol (heavy/binge)",
                "severity": "AVOID",
                "reason": (
                    "Acetaldehyde (ethanol metabolite) inhibits Complex I + depletes mitochondrial "
                    "glutathione (GSH) → oxidative stress in optic nerve; binge drinking associated "
                    "with acute visual deterioration episodes in LHON and ADOA; chronic moderate "
                    "alcohol: advise minimise (< 1 unit/day); binge drinking: absolutely prohibited"
                ),
            },
        ],
        "ddx_highlights": [
            "OPA1 ADOA vs LHON (Leber Hereditary Optic Neuropathy): CRITICAL DDx — ADOA is nuclear AD (50% offspring risk) vs LHON is mitochondrial maternal inheritance (mtDNA m.11778G>A / m.3460G>A / m.14484T>C); ADOA = childhood slow progressive vs LHON = subacute adult-onset (15–35 yr); ADOA tritanopia (blue-yellow axis) vs LHON telangiectatic microangiopathy + red-green defect; ADOA bilateral symmetric vs LHON sequential",
            "OPA1 ADOA vs Normal-Tension Glaucoma (NTG): bilateral optic disc pallor in ADOA mimics NTG; DDx: ADOA family history (AD), childhood onset, central scotoma, normal IOP throughout, normal nerve fibre layer thinning pattern (temporal > inferior in ADOA vs inferotemporal in NTG), OPA1 panel positive",
            "OPA1 ADOA vs NAION (Non-Arteritic Anterior Ischaemic Optic Neuropathy): NAION = acute unilateral with segmental disc oedema, altitudinal field loss, no family history; ADOA = bilateral symmetric, childhood onset, central/centrocaecal scotoma, AD family history, no disc oedema at onset",
            "OPA1-Plus vs adPEO series (POLG2/DNA2/SLC25A4-PEOA1/TWNK-PEOA3): all share mtDNA multiple deletions + COX-negative fibres in muscle; OPA1-Plus is the ONLY adPEO-spectrum disease that presents primarily with optic atrophy (not PEO) as the leading feature; PEO is secondary/mild in OPA1-Plus; OPA1 GTPase domain variant is the molecular flag for OPA1-Plus",
            "OPA1 ADOA vs Wolfram Syndrome (WFS1): both cause progressive optic atrophy starting in childhood; Wolfram = AR, adds DM (type 1 DM), diabetes insipidus, hearing loss, psychiatric; WFS1 OA onset ~ 6 yr similar to OPA1; key DDx: DM at onset in Wolfram (rare in OPA1), AR family history, WFS1 gene panel",
            "OPA1 vs TMEM126A/OPA7 and other rare ADOA genes: TMEM126A causes AR optic atrophy (AROA7); ACO2 causes infantile cerebellar-retinal degeneration; AFG3L2 causes dominant SCA28 + optic neuropathy; gene panel mandatory if OPA1 negative in classical ADOA presentation",
        ],
        "references": [
            {
                "author": "Delettre C et al.",
                "year": 2000,
                "journal": "Nature Genetics",
                "title": "Nuclear gene OPA1, encoding a mitochondrial dynamin-related protein, is mutated in dominant optic atrophy",
                "note": "Volume 26, pp 207–210 — French discovery paper identifying OPA1 as ADOA gene; Breton families; p.E364K French founder mutation; established dynamin-GTPase as optic atrophy mechanism",
            },
            {
                "author": "Alexander C et al.",
                "year": 2000,
                "journal": "Nature Genetics",
                "title": "OPA1, encoding a dynamin-related GTPase, is mutated in autosomal dominant optic atrophy linked to chromosome 3q28",
                "note": "Volume 26, pp 211–215 — Simultaneous British discovery paper; same issue as Delettre; established OPA1 independently; confirmed dynamin-like GTPase mechanism of optic atrophy",
            },
            {
                "author": "Amati-Bonneau P et al.",
                "year": 2008,
                "journal": "Brain",
                "title": "OPA1 mutations induce mitochondrial DNA instability and optic atrophy plus phenotypes",
                "note": "Volume 131, pp 338–351 — Landmark OPA1-Plus paper: identified that OPA1 GTPase missense mutations cause mtDNA multiple deletions + multisystem disease beyond optic atrophy; established OPA1-Plus spectrum; p.R445H hotspot characterised",
            },
            {
                "author": "Yu-Wai-Man P et al.",
                "year": 2010,
                "journal": "Brain",
                "title": "Multi-system neurological disease is common in patients with OPA1 mutations",
                "note": "Volume 133, pp 771–786 — Large OPA1 cohort study; quantified OPA1-Plus prevalence (~20% of OPA1 patients); characterised multisystem features including SNHL, ataxia, neuropathy, PEO, myopathy; recommended annual multisystem review for all OPA1 patients",
            },
            {
                "author": "Lenaers G et al.",
                "year": 2012,
                "journal": "Orphanet Journal of Rare Diseases",
                "title": "Dominant optic atrophy",
                "note": "Comprehensive clinical review of OPA1 ADOA spectrum; natural history; management; genotype-phenotype; ethambutol / linezolid contraindications; UK + French practice guidelines basis",
            },
        ],
    }


def get_breakdown() -> dict[str, Any]:
    rng = _rng()
    patients = _build_cohort(rng)

    n_optic = sum(1 for p in patients if p["optic_atrophy"])
    n_snhl = sum(1 for p in patients if p["snhl"])
    n_ataxia = sum(1 for p in patients if p["ataxia"])
    n_myopathy = sum(1 for p in patients if p["myopathy"])
    n_neuropathy = sum(1 for p in patients if p["peripheral_neuropathy"])
    n_peo = sum(1 for p in patients if p["peo"])
    n_para = sum(1 for p in patients if p["spastic_paraplegia"])
    n_cognitive = sum(1 for p in patients if p["cognitive_decline"])
    n_parkinson = sum(1 for p in patients if p["parkinsonism"])
    n_plus = sum(1 for p in patients if p["tier"] == "OPA1-Plus")

    avg_onset = round(sum(p["age_onset_years"] for p in patients) / N_PATIENTS, 1)
    avg_dx_delay = round(sum(p["dx_delay_yr"] for p in patients) / N_PATIENTS, 1)
    avg_va_r = round(sum(p["va_right"] for p in patients) / N_PATIENTS, 2)
    avg_va_l = round(sum(p["va_left"] for p in patients) / N_PATIENTS, 2)

    from collections import Counter
    variant_counts = Counter(p["variant"] for p in patients)
    tier_counts = Counter(p["tier"] for p in patients)
    misdiag_counts = Counter(
        rng.choice([
            "Normal-Tension-Glaucoma",
            "LHON-Mimicry",
            "NAION-Unilateral-Presentation",
            "Optic-Neuritis-MS-Workup",
            "Idiopathic-Optic-Atrophy",
            "Wolfram-Syndrome",
        ])
        for _ in range(N_PATIENTS)
    )

    summary = {
        "n_patients": N_PATIENTS,
        "avg_onset_years": avg_onset,
        "avg_dx_delay_years": avg_dx_delay,
        "avg_va_right": avg_va_r,
        "avg_va_left": avg_va_l,
        "optic_atrophy_pct": round(n_optic / N_PATIENTS * 100),
        "opa1_plus_pct": round(n_plus / N_PATIENTS * 100),
        "snhl_pct": round(n_snhl / N_PATIENTS * 100),
        "ataxia_pct": round(n_ataxia / N_PATIENTS * 100),
        "myopathy_pct": round(n_myopathy / N_PATIENTS * 100),
        "neuropathy_pct": round(n_neuropathy / N_PATIENTS * 100),
        "peo_pct": round(n_peo / N_PATIENTS * 100),
        "cognitive_pct": round(n_cognitive / N_PATIENTS * 100),
        "parkinsonism_pct": round(n_parkinson / N_PATIENTS * 100),
    }

    etiology_distribution = [
        {"label": label, "pct": round(count / N_PATIENTS * 100)}
        for label, count in variant_counts.most_common()
    ]

    tier_distribution = [
        {"label": label, "pct": round(count / N_PATIENTS * 100)}
        for label, count in tier_counts.most_common()
    ]

    misdiagnosis_distribution = [
        {"label": label, "pct": round(count / N_PATIENTS * 100)}
        for label, count in misdiag_counts.most_common()
    ]

    patient_rows = [
        {
            "id": p["id"],
            "tier": p["tier"],
            "age_onset": p["age_onset_years"],
            "snhl": p["snhl"],
            "ataxia": p["ataxia"],
            "myopathy": p["myopathy"],
            "neuropathy": p["peripheral_neuropathy"],
            "peo": p["peo"],
            "parkinsonism": p["parkinsonism"],
            "va_right": p["va_right"],
            "va_left": p["va_left"],
            "ck_x_uln": p["ck_x_uln"],
            "dx_delay_yr": p["dx_delay_yr"],
            "variant": p["variant"],
        }
        for p in patients
    ]

    treatments = [
        {
            "name": "Tobacco Cessation (MANDATORY first-line intervention)",
            "tier": "First-line — modifiable risk factor",
            "evidence": "Level A (expert consensus + mechanistic)",
            "mechanism": (
                "Tobacco combustion products (cyanide, acrolein, nitric oxide, superoxide) inhibit "
                "Complex IV and III respectively and deplete mitochondrial GSH → amplify OXPHOS "
                "failure in OPA1-deficient RGCs → accelerated visual loss; smoking cessation removes "
                "the largest modifiable stressor on already-compromised bioenergetic reserve in RGCs"
            ),
            "dose": "Complete cessation of all tobacco + nicotine products (including vaping — acrolein); "
                    "pharmacotherapy: varenicline (Champix) first-line — no known mito toxicity; "
                    "NRT (patch/gum) acceptable; avoid bupropion (lowers seizure threshold in OPA1-Plus CNS disease)",
            "monitoring": "Carbon monoxide breath test at each visit; cotinine urine if compliance uncertain; "
                          "repeat visual acuity + OCT at 6 months to objectively document benefit of cessation",
        },
        {
            "name": "Alcohol Minimisation",
            "tier": "Lifestyle modification",
            "evidence": "Level B (observational cohort + mechanistic)",
            "mechanism": (
                "Acetaldehyde depletes mitochondrial NADH + GSH → ROS generation in optic nerve; "
                "binge episodes cause acute IOP spikes and optic disc ischaemia; "
                "total abstinence recommended for active symptomatic OPA1; moderate reduction for "
                "asymptomatic carriers (≤7 units/week; no binge)"
            ),
            "dose": "Active ADOA: complete alcohol abstinence recommended; "
                    "pre-symptomatic OPA1 carriers: max 7 units/week, no binge (>6 units in single session)",
            "monitoring": "Symptom diary; VA log; MCV as alcohol biomarker at annual review",
        },
        {
            "name": "Idebenone (Raxone)",
            "tier": "Investigational/off-label in ADOA (approved for LHON in EU)",
            "evidence": "Level C in ADOA (Leber-approved; ADOA trials ongoing)",
            "mechanism": (
                "Idebenone is a short-chain CoQ10 analogue that bypasses Complex I dysfunction "
                "by donating electrons directly to Complex III; in RGCs with disrupted OXPHOS "
                "supercomplexes (OPA1 deficiency), idebenone may partially restore electron "
                "transfer efficiency and ATP production; antioxidant role reduces superoxide-driven "
                "apoptosis in surviving RGCs; RESCUE trial (LHON) showed recovery in patients who "
                "could still see; ADOA data from small series only"
            ),
            "dose": "900 mg/day in 3 divided doses with meals (same as LHON adult dose); "
                    "children: 20 mg/kg/day up to 900 mg; idebenone capsules opened on food if swallowing difficulty",
            "monitoring": "VA (monthly for 6 months then 3-monthly); OCT RNFL; liver function at 3 months; "
                          "generally well tolerated; GI side effects commonest; discontinue if no response at 12 months",
        },
        {
            "name": "Coenzyme Q10 (CoQ10) + B-vitamin supplements",
            "tier": "Standard supplementation",
            "evidence": "Level C (empirical; no RCT in OPA1)",
            "mechanism": (
                "CoQ10 supports residual OXPHOS function; Complex I/III/IV electron transfer; "
                "B2 (riboflavin) = FAD/FMN precursor for Complex I/II; "
                "B12 = cofactor for methylmalonyl-CoA mutase + methionine synthase; "
                "used empirically across mitochondrial optic nerve diseases"
            ),
            "dose": "CoQ10 400–1200 mg/day (ubiquinol preferred); riboflavin 100–400 mg/day; "
                    "B12 1 mg/day IM monthly or 1000 µg sublingual daily",
            "monitoring": "Plasma CoQ10 level (target >2.5 µg/mL); annual LFTs; annual B12 level",
        },
        {
            "name": "Low Vision Rehabilitation + Adaptive Technology",
            "tier": "Standard of care — visual rehabilitation",
            "evidence": "Standard of care",
            "mechanism": (
                "Maximise residual visual function: low vision aids (magnifiers, telescopes, "
                "CCTV systems), screen reader software, large-print support, contrast enhancement; "
                "occupational therapy for daily living adaptations; driving cessation counselling "
                "when VA falls below legal standard (≥6/12 UK / 20/40 US)"
            ),
            "dose": "Low vision assessment (low vision optometrist + RNIB/equivalent); "
                    "adaptive technology assessment; driving DVLA/DVSA notification mandatory if VA < legal standard",
            "monitoring": "Annual review; functional vision assessment (reading, navigation, screen use); "
                          "psychological support referral (adjustment to vision loss; prevalence of depression high)",
        },
        {
            "name": "Gene Therapy — Intravitreal AAV2-OPA1 (clinical trials)",
            "tier": "Investigational (trial phase; NOT standard of care)",
            "evidence": "Preclinical evidence + Phase I/II trials ongoing",
            "mechanism": (
                "AAV2 vector transduces retinal ganglion cells via intravitreal injection → "
                "delivers wild-type OPA1 cDNA → restores OPA1 expression in RGCs → "
                "rescues mitochondrial fusion + OXPHOS + cristae; "
                "mouse OPA1 model: gene therapy rescues ~60% of RGC survival; "
                "clinical trial NCT02418389 (PoEM trial, UK/France); "
                "contra: severe strabismus (PEO) may affect injection precision"
            ),
            "dose": "Intravitreal injection 1–3 × 10^10 vg/eye (trial dose range); "
                    "single unilateral injection first; second eye if response confirmed; "
                    "avoid propofol-free anaesthesia NOT required (optic not skeletal muscle)",
            "monitoring": "Monthly VA + OCT post-injection for 12 months (trial protocol); "
                          "intraocular inflammation monitoring (uveitis protocol); "
                          "systemically: hepatic function (AAV2 liver transduction negligible)",
        },
        {
            "name": "Levetiracetam (LEV) — preferred AED if seizures in OPA1-Plus",
            "tier": "Preferred AED (OPA1-Plus with CNS involvement)",
            "evidence": "Expert consensus + mechanistic safety",
            "mechanism": (
                "LEV has no known mitochondrial toxicity; renal excretion as inactive hydrolysate "
                "(no hepatic CYP metabolism); does not affect CoA pool; SV2A mechanism; "
                "safe in OPA1-Plus with multisystem disease including CNS involvement (ataxia/cognitive)"
            ),
            "dose": "500–3000 mg/day in two divided doses; renal dose-adjustment if eGFR <60",
            "monitoring": "Mood/behavioural side effects (irritability); EEG response; renal function annually",
            "caution": (
                "VPA: use with caution in OPA1-Plus with myopathy (COX-negative fibres + skeletal mtDNA deletions) "
                "— VPA CoA sequestration risk in OXPHOS-deficient muscle; pure ADOA: VPA not absolute CI but "
                "specialist review recommended. AVOID ethambutol and linezolid regardless of indication."
            ),
        },
    ]

    systemic_features = [
        {
            "label": "Bilateral Optic Atrophy (CARDINAL — 100%)",
            "pct": round(n_optic / N_PATIENTS * 100),
            "note": (
                "Bilateral temporal optic disc pallor → full atrophy; central/centrocaecal scotoma; "
                "tritanopia (blue-yellow dyschromatopsia on FM-100 hue test); "
                "VA range 6/6 to hand motion within same family (variable expressivity); "
                "OCT temporal RNFL thinning earliest finding; VEP prolonged P100 latency; "
                "onset: mean ~7 yr (range infancy to 35 yr)"
            ),
        },
        {
            "label": "OPA1-Plus tier (~20% of OPA1)",
            "pct": round(n_plus / N_PATIENTS * 100),
            "note": (
                "Subset with GTPase domain missense (p.R445H, p.G488R); adds mtDNA multiple deletions "
                "in skeletal muscle (same molecular pattern as adPEO series); multisystem features "
                "appear in 2nd–4th decade, decades after optic atrophy onset; annual multisystem "
                "review mandatory in all OPA1 patients (genotype alone cannot exclude OPA1-Plus)"
            ),
        },
        {
            "label": "SNHL (predominantly OPA1-Plus)",
            "pct": round(n_snhl / N_PATIENTS * 100),
            "note": (
                "Bilateral sensorineural hearing loss; progressive; high-frequency predominantly; "
                "audiogram annually from diagnosis; cochlear implant if severe (safe — no propofol "
                "restriction in pure OPA1 unlike adPEO series with COX-negative fibres)"
            ),
        },
        {
            "label": "Cerebellar Ataxia (OPA1-Plus)",
            "pct": round(n_ataxia / N_PATIENTS * 100),
            "note": (
                "Progressive cerebellar ataxia; SARA scale monitoring; cerebellar atrophy on MRI "
                "in severe cases; gait aid if needed; physiotherapy for balance and coordination"
            ),
        },
        {
            "label": "Proximal Myopathy (OPA1-Plus — COX-negative fibres)",
            "pct": round(n_myopathy / N_PATIENTS * 100),
            "note": (
                "Skeletal muscle COX-negative fibres + mtDNA multiple deletions (long-range PCR); "
                "same molecular pattern as adPEO series; proximal > distal; CK mild elevation; "
                "VPA caution in OPA1-Plus myopathy (OXPHOS-deficient muscle vulnerable to CoA depletion)"
            ),
        },
        {
            "label": "Peripheral Neuropathy (axonal, OPA1-Plus)",
            "pct": round(n_neuropathy / N_PATIENTS * 100),
            "note": "Axonal sensory > motor; NCS/EMG for characterisation; vibration proprioception; overlap with ataxia worsens balance",
        },
        {
            "label": "PEO — bilateral ophthalmoplegia (OPA1-Plus)",
            "pct": round(n_peo / N_PATIENTS * 100),
            "note": (
                "External ophthalmoplegia in OPA1-Plus (secondary feature unlike primary PEO in adPEO series); "
                "same COX-negative fibre mechanism in extraocular muscle; Hess chart annually if present; "
                "Bell's phenomenon assessment before any ptosis surgery"
            ),
        },
        {
            "label": "Spastic Paraplegia (OPA1-Plus)",
            "pct": round(n_para / N_PATIENTS * 100),
            "note": "Corticospinal tract involvement; hyperreflexia; spasticity; baclofen (intrathecal if severe); physiotherapy; walking aids",
        },
        {
            "label": "Cognitive Decline (OPA1-Plus)",
            "pct": round(n_cognitive / N_PATIENTS * 100),
            "note": "MoCA/MMSE annual monitoring in OPA1-Plus with CNS involvement; memory + executive function domains; neuropsychology referral",
        },
        {
            "label": "Parkinsonism (OPA1-Plus — partial L-DOPA)",
            "pct": round(n_parkinson / N_PATIENTS * 100),
            "note": "Partial L-DOPA response; DaT-SPECT if uncertain; SNPC deletion burden (COX-negative nigrostriatal neurons); dopaminergic therapy trial if functional",
        },
    ]

    return {
        "summary": summary,
        "patients": patient_rows,
        "etiology_distribution": etiology_distribution,
        "tier_distribution": tier_distribution,
        "misdiagnosis_distribution": misdiagnosis_distribution,
        "systemic_features": systemic_features,
        "treatments": treatments,
    }


def get_definitions() -> dict[str, Any]:
    return {
        "gene_biology": [
            {
                "term": "OPA1 — Mitochondrial Dynamin-Like GTPase",
                "definition": (
                    "OPA1 (960 aa; 3q29; UniProt O60313) is the human homologue of yeast Mgm1. "
                    "Eight isoforms from alternative splicing of exons 4, 4b, and 5b; all are "
                    "IMM-anchored via an N-terminal transmembrane domain. Two functional pools: "
                    "Long OPA1 (L-OPA1): IMM-tethered, drives tethering; "
                    "Short OPA1 (S-OPA1): generated by OMA1/YME1L cleavage at S1/S2 sites; "
                    "L + S together required for inner membrane fusion. Domains: N-terminal TM "
                    "(aa 1–60), coiled-coil 1 (aa 100–200), GTPase domain (aa 360–600, OPA1-Plus "
                    "hotspot), middle domain, GED (aa 800–960). Functions: IMM fusion + cristae "
                    "junction tightening."
                ),
            },
            {
                "term": "OPA1 Haploinsufficiency vs Dominant-Negative Mechanism",
                "definition": (
                    "Most ADOA mutations = LOF (frameshift, nonsense, splice, large deletion) → "
                    "haploinsufficiency: 50% residual OPA1 insufficient for RGC mitochondrial network "
                    "maintenance (pure ADOA). GTPase domain missense (p.R445H, p.G488R) = dominant "
                    "negative: mutant OPA1 inhibits WT OPA1 oligomerisation → additional OXPHOS "
                    "instability → mtDNA multiple deletions in post-mitotic muscle → OPA1-Plus "
                    "multisystem disease. Genotype predicts tier: LOF → ADOA; GTPase missense → OPA1-Plus "
                    "(but not absolute — modifier genes + stochastic factors play a role)."
                ),
            },
            {
                "term": "OPA1 Cristae Remodelling — Apoptotic Threshold",
                "definition": (
                    "OPA1 oligomers form ring/filament structures at cristae junctions (nm-scale IMM "
                    "invaginations) → mechanically tighten the junction → compartmentalises cytochrome c "
                    "within cristae lumen. Normal: cytochrome c inside cristae → apoptotic signal "
                    "requires cristae remodelling before cyt c releases. OPA1 LOF → open/loose cristae "
                    "junctions → cytochrome c redistributes to IMS at baseline → minimal additional "
                    "stimulus (ischaemia, ROS, OXPHOS stress) sufficient to trigger caspase cascade → "
                    "RGC death amplified → OPA1 RGCs have a lowered apoptotic threshold."
                ),
            },
            {
                "term": "OPA1-Plus — mtDNA Multiple Deletions Mechanism",
                "definition": (
                    "In OPA1-Plus (GTPase missense), mitochondrial fragmentation + OXPHOS instability "
                    "alters mtDNA replication fork geometry in post-mitotic skeletal muscle → deletions "
                    "accumulate via same mechanism as adPEO series (POLG2/DNA2/SLC25A4/TWNK-PEOA3). "
                    "Long-range PCR shows identical multiple-deletion pattern; copy number NORMAL "
                    "(deletion, not depletion). Blood unreliable (tissue-specific; muscle required). "
                    "This links OPA1-Plus molecularly to the adPEO spectrum despite optic-first phenotype."
                ),
            },
        ],
        "disease_concepts": [
            {
                "term": "ADOA — Autosomal Dominant Optic Atrophy (Kjer Disease)",
                "definition": (
                    "ADOA (OMIM #165500) is the most common hereditary optic neuropathy (prevalence "
                    "1:30,000–50,000). Caused by AD OPA1 haploinsufficiency. Bilateral progressive "
                    "optic atrophy with childhood onset (mean ~7 yr). Cardinal features: temporal disc "
                    "pallor, central/centrocaecal scotoma, tritanopia, slowly progressive visual loss. "
                    "Penetrance ~98%, expressivity highly variable. Two landmark papers simultaneously "
                    "identified OPA1: Delettre et al. 2000 NatGenet (France) + Alexander et al. 2000 "
                    "NatGenet (UK)."
                ),
            },
            {
                "term": "OPA1-Plus — Multisystem OPA1 Disease",
                "definition": (
                    "OPA1-Plus refers to the ~20% of OPA1 patients who develop features beyond "
                    "optic atrophy: SNHL, ataxia, myopathy, neuropathy, PEO, spastic paraplegia, "
                    "cognitive decline, parkinsonism. Typically caused by OPA1 GTPase domain missense "
                    "mutations (dominant-negative) rather than pure LOF. Features appear in 2nd–4th "
                    "decade, years to decades after optic atrophy onset. Key molecular finding: mtDNA "
                    "multiple deletions in skeletal muscle (long-range PCR). First described by "
                    "Amati-Bonneau et al. 2008 Brain. Yu-Wai-Man 2010 Brain quantified prevalence."
                ),
            },
            {
                "term": "Tritanopia — Characteristic OPA1 Colour Vision Defect",
                "definition": (
                    "Tritanopia = blue-yellow axis colour vision defect (loss of S-cone discrimination). "
                    "Caused by preferential damage to RGCs subserving short-wavelength (blue) colour "
                    "vision (small bistratified cells — most vulnerable to OXPHOS failure). "
                    "Clinical test: Farnsworth-Munsell 100-hue test shows tritanopic axis error; "
                    "Ishihara plates primarily test red-green (not OPA1-specific). "
                    "DDx: LHON patients more commonly show red-green + non-specific colour defect; "
                    "tritanopia strongly suggests ADOA in a child with bilateral optic atrophy."
                ),
            },
            {
                "term": "Ethambutol Optic Neuropathy — ABSOLUTE CI in OPA1",
                "definition": (
                    "Ethambutol (EMB) chelates copper → inhibits cytochrome c oxidase (Complex IV) "
                    "in optic nerve via cupric ion sequestration from the binuclear CuA/CuB centres. "
                    "Drug-induced optic neuropathy (DION) from EMB typically reversible in immunocompetent "
                    "patients if caught early. In OPA1 haploinsufficiency: RGCs already at OXPHOS "
                    "threshold; EMB → acute-on-chronic Complex IV failure → irreversible vision loss. "
                    "Alternative TB regimens: HRZE → HRZ (drop E for Z in OPA1); or "
                    "rifampicin-isoniazid-pyrazinamide-amikacin (if DS-TB and EMB needed — use amikacin "
                    "with monthly VA monitoring + immediate stop if VA deteriorates)."
                ),
            },
            {
                "term": "OPA1 vs LHON — Critical Diagnostic Distinction",
                "definition": (
                    "ADOA (OPA1, nuclear gene) vs LHON (mtDNA m.11778G>A/m.3460G>A/m.14484T>C): "
                    "ADOA = AD nuclear (50% offspring risk) vs LHON = mitochondrial maternal (maternal "
                    "transmission; male bias 80–90% penetrant). ADOA = slow progressive childhood onset "
                    "vs LHON = subacute adult onset (15–35 yr) with acute phase. ADOA = tritanopia vs "
                    "LHON = red-green + telangiectatic microangiopathy on fundus. ADOA bilateral "
                    "symmetric simultaneously vs LHON sequential (fellow eye months later). "
                    "Both: central scotoma, optic atrophy. Gene panel or mtDNA testing required."
                ),
            },
        ],
        "prescribing_safety": [
            {
                "term": "Ethambutol + Linezolid — ABSOLUTE CONTRAINDICATIONS in OPA1",
                "definition": (
                    "Two drugs require absolute prohibition in all OPA1 patients: "
                    "(1) Ethambutol: Cu chelation → Complex IV inhibition → acute-on-chronic optic "
                    "neuropathy; irreversible in OPA1; TB alternative: RIF+INH+PZA ± streptomycin. "
                    "(2) Linezolid: mitochondrial 23S rRNA inhibitor → blocks mtDNA-encoded subunit "
                    "synthesis → acute DION; reversible if stopped within 2–4 weeks; permanent if "
                    "prolonged; alternative: daptomycin, tedizolid, co-trimoxazole, rifampicin (for "
                    "MRSA if appropriate). ALWAYS document OPA1 diagnosis on drug allergy/alert screen."
                ),
            },
            {
                "term": "Tobacco — ABSOLUTE Environmental Contraindication",
                "definition": (
                    "Tobacco smoke is the most important modifiable risk factor in OPA1 ADOA. "
                    "Cyanide (CN) directly inhibits Complex IV (same mechanism as ethambutol); "
                    "acrolein inhibits Complex II/III; carbon monoxide blocks cytochrome c oxidase; "
                    "superoxide generators accelerate RGC oxidative stress. OPA1 haploinsufficiency "
                    "means RGCs have no OXPHOS reserve to buffer the tobacco toxic load. "
                    "Smoking OPA1 patients progress 3–5× faster than non-smoking OPA1. "
                    "Varenicline (Champix/Chantix) is the preferred pharmacotherapy for cessation — "
                    "no known mitochondrial toxicity; no drug interactions with OPA1 management."
                ),
            },
            {
                "term": "VPA (Valproic Acid) — CAUTION in OPA1-Plus; NOT Absolute CI in Pure ADOA",
                "definition": (
                    "VPA sequesters mitochondrial CoA (as valproyl-CoA + propionyl-CoA) → disrupts "
                    "beta-oxidation + CoA-dependent OXPHOS reactions. In OPA1-Plus patients with "
                    "COX-negative skeletal muscle fibres (OXPHOS-deficient): VPA = SIGNIFICANT "
                    "RISK; avoid or use only after specialist mitochondrial medicine review. "
                    "In pure ADOA patients with no skeletal muscle involvement: VPA not absolute CI "
                    "(optic nerve does not depend on beta-oxidation in the same way as skeletal muscle) "
                    "but neurologist review recommended. If VPA required in pure ADOA: ensure LFTs, "
                    "CoA supplementation, and close monitoring; AED alternatives: LEV, lamotrigine, "
                    "lacosamide."
                ),
            },
            {
                "term": "Levetiracetam (LEV) — Preferred AED in OPA1",
                "definition": (
                    "LEV is the first-line AED choice across the OPA1 spectrum (ADOA + OPA1-Plus). "
                    "Renal excretion as inactive SV2A enzyme hydrolysate; no hepatic CYP450 metabolism; "
                    "no interference with CoA or mitochondrial metabolism; no enzyme induction; "
                    "no drug-drug interaction with idebenone or CoQ10. Dose: 500–3000 mg/day in 2 "
                    "divided doses. Monitor behavioural/mood side effects (irritability is common); "
                    "renal dose-adjustment if eGFR <60 mL/min. Safe in OPA1-Plus multisystem "
                    "disease including CNS (ataxia, cognitive) and skeletal muscle (myopathy)."
                ),
            },
        ],
    }
