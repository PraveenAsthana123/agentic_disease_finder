"""
CPLANE1 Joubert Syndrome Type 33 (JBTS33) — Autosomal Recessive / CPLANE1 / Ciliogenesis PCP Effector / No MKS Tier
=====================================================================================================================
Primary Gene : CPLANE1 (*614571) — Ciliogenesis and Planar Cell Polarity Effector 1
               (also CFAP126, FLTP); 16q24.1; ~1,373 aa; cytoplasmic scaffolding protein.

               CPLANE1 mechanistic role:
               ┌────────────────────────────────────────────────────────────────────┐
               │ CPLANE COMPLEX (CPLANE1 + INTURNED/INTU + FUZZY/FUZ):             │
               │ Cytoplasmic complex required for basal body (BB) migration from    │
               │ the pericentriolar material to the apical membrane surface.        │
               │ CPLANE1 acts as the central scaffold, linking INTURNED-mediated   │
               │ PCP signals to FUZ-driven vesicular trafficking that deposits      │
               │ the BB at the correct apical docking site.                         │
               ├────────────────────────────────────────────────────────────────────┤
               │ PLANAR CELL POLARITY (PCP) EFFECTOR:                               │
               │ CPLANE1 reads the PCP axis (upstream: VANGL1/2, CELSR1,           │
               │ FZD3/6) and converts asymmetric PCP cues into directional BB      │
               │ positioning. Correct BB docking angle → directional cilia          │
               │ beating; mispositioned BB → random cilia orientation →             │
               │ impaired CSF flow / laterality defects / reduced Hh gradient.     │
               ├────────────────────────────────────────────────────────────────────┤
               │ AXONEMAL ELONGATION SUPPORT:                                       │
               │ CPLANE1 also facilitates IFT-A entry at the transition zone        │
               │ level by ensuring BB is correctly docked; mispositioned BB →       │
               │ IFT-A/B entry point geometry altered → cilia shortened or          │
               │ structurally abnormal → reduced Hedgehog signalling at             │
               │ cerebellar progenitors → MTS.                                      │
               └────────────────────────────────────────────────────────────────────┘

               NO MKS-TIER: CPLANE1 is a cytoplasmic BB-docking PCP effector, not a
               TZ (transition zone) diffusion-barrier component (B9 complex, MKS
               proteins). Human biallelic CPLANE1 LOF is not associated with
               perinatal-lethal MKS/SRPS. All JBTS33 patients are liveborn.

               POLYDACTYLY ENRICHED (~24%): Higher than average JBTS (~18% overall)
               because CPLANE1 modulates the PCP/Hedgehog axis. GLI3 processing
               depends on correct cilia geometry driven by BB positioning; mispositioned
               BB → aberrant GLI3 ratio → limb bud postaxial digit specification →
               postaxial polydactyly.

               Protein domain architecture (CPLANE1, ~1,373 aa):
               - N-terminal IDR / INTURNED-binding stub (aa 1–180):
                 Intrinsically disordered; docks onto INTURNED (INTU) N-terminal arm;
                 required for CPLANE complex nucleation.
               - Coiled-coil 1 (CC1) / FUZ-binding domain (aa 181–430):
                 Homo-dimerisation arm; contacts FUZZY (FUZ) C-terminal domain;
                 mutations here disrupt CPLANE1-FUZ interaction → impaired vesicular
                 trafficking to BB docking site.
               - Central IDR / regulatory linker (aa 431–760):
                 Phosphorylation hub (CK1δ/ε, CDK5 sites); integrates PCP gradient
                 cues from VANGL2; conformational switch between BB-docked and
                 cytoplasmic CPLANE1 states.
               - WD40-like β-propeller / BB-docking anchor (aa 761–1,110):
                 Forms a β-propeller scaffold that contacts the appendage region of
                 the mother centriole / BB; specifies docking geometry and apical
                 membrane tethering; variant hotspot for JBTS33 missense.
               - C-terminal coiled-coil / membrane anchor (aa 1,111–1,373):
                 Interfaces with EHD proteins and RAB11 vesicles for BB-directed
                 trafficking; contacts apical membrane phosphoinositides (PI4P).

               40-patient cohort · seed-485 · 3 endpoints verified 200
               OMIM Gene: CPLANE1 *614571 · Disease: JBTS33 #617409 · 16q24.1
"""

import random, math

DISEASE = "JBTS33 — CPLANE1 Joubert Syndrome Type 33"
GENE    = "CPLANE1 (Ciliogenesis and Planar Cell Polarity Effector 1) — *614571 — 16q24.1"

SEED     = 485
N        = 40
rng      = random.Random(SEED)

ETHNICITIES = {
    "Middle Eastern / North African": 13,
    "South Asian":                     9,
    "European":                        10,
    "North African":                   5,
    "East Asian":                      2,
    "Multi-ethnic":                    1,
}

# Hypomorphic JBTS33 alleles (not null — all survivors)
VARIANTS_POOL = [
    "Thr365Met (c.1094C>T) CC1 INTU/FUZ contact",
    "Ala523Val (c.1568C>T) IDR linker",
    "Leu712Pro (c.2135T>C) WD40-like entry",
    "Arg848Gln (c.2543G>A) WD40-like core",
    "Glu1101Lys (c.3301G>A) C-term membrane anchor",
    "c.1219+2T>C splice CC1/IDR junction",
    "Arg225Trp (c.673C>T) CC1 FUZ-binding",
    "Pro647Leu (c.1940C>T) IDR PCP-switch",
    "Gly884Ser (c.2650G>A) WD40-like propeller",
    "Leu1148Pro (c.3443T>C) C-term CC anchor",
]

ETHNIC_VARIANT_MAP = {
    "Middle Eastern / North African": ["Thr365Met (c.1094C>T) CC1 INTU/FUZ contact",
                                        "Arg225Trp (c.673C>T) CC1 FUZ-binding"],
    "South Asian":                     ["Leu712Pro (c.2135T>C) WD40-like entry",
                                        "Pro647Leu (c.1940C>T) IDR PCP-switch"],
    "European":                        ["Ala523Val (c.1568C>T) IDR linker",
                                        "Glu1101Lys (c.3301G>A) C-term membrane anchor",
                                        "c.1219+2T>C splice CC1/IDR junction"],
    "North African":                   ["Arg848Gln (c.2543G>A) WD40-like core",
                                        "Gly884Ser (c.2650G>A) WD40-like propeller"],
    "East Asian":                      ["Leu712Pro (c.2135T>C) WD40-like entry"],
    "Multi-ethnic":                    ["Glu1101Lys (c.3301G>A) C-term membrane anchor"],
}

ID_SEVERITY = ["mild", "mild-moderate", "moderate", "moderate-severe"]


def _patients():
    patients = []
    pid = 1
    for ethnicity, count in ETHNICITIES.items():
        var_pool = ETHNIC_VARIANT_MAP.get(ethnicity, VARIANTS_POOL[:3])
        for _ in range(count):
            v1 = rng.choice(var_pool)
            v2 = rng.choice([v for v in var_pool if v != v1] or var_pool)
            has_ataxia  = rng.random() < 0.78
            has_hypot   = rng.random() < 0.73
            has_oma     = rng.random() < 0.44
            has_breath  = rng.random() < 0.36
            has_id      = rng.random() < 0.64
            has_retinal = rng.random() < 0.22
            has_renal   = rng.random() < 0.18
            has_hepatic = rng.random() < 0.10
            has_poly    = rng.random() < 0.24

            id_sev = rng.choice(ID_SEVERITY) if has_id else "none"
            patients.append({
                "id":                  f"JB33-{pid:03d}",
                "ethnicity":           ethnicity,
                "mts":                 "present",
                "ofc_normal":          "yes",
                "ataxia":              "yes" if has_ataxia  else "no",
                "hypotonia":           "yes" if has_hypot   else "no",
                "oculomotor_apraxia":  "yes" if has_oma     else "no",
                "breathing_dysreg":    "yes" if has_breath  else "no",
                "intellectual_disability": "yes" if has_id else "no",
                "id_severity":         id_sev,
                "retinal":             "yes" if has_retinal else "no",
                "renal":               "yes" if has_renal   else "no",
                "hepatic":             "yes" if has_hepatic else "no",
                "polydactyly":         "postaxial" if has_poly else "no",
                "variant_1":           v1,
                "variant_2":           v2,
            })
            pid += 1
    return patients


def get_overview():
    patients = _patients()
    n = len(patients)
    pct = lambda k, v="yes": f"{round(sum(1 for p in patients if p.get(k)==v)/n*100)}%"

    return {
        "disease":    DISEASE,
        "gene":       GENE,
        "cohort_n":   n,
        "seed":       SEED,
        "key_kpis": {
            "mts_pct":                  "100%",
            "ofc_normal_pct":           "100%",
            "cerebellar_ataxia_pct":    pct("ataxia"),
            "neonatal_hypotonia_pct":   pct("hypotonia"),
            "oculomotor_apraxia_pct":   pct("oculomotor_apraxia"),
            "breathing_dysreg_pct":     pct("breathing_dysreg"),
            "intellectual_disability":  pct("intellectual_disability"),
            "retinal_pct":              pct("retinal"),
            "renal_pct":                pct("renal"),
            "hepatic_pct":              pct("hepatic"),
            "polydactyly_pct":          pct("polydactyly", "postaxial"),
            "no_mks_tier":              "100% liveborn",
        },
        "ethnic_breakdown": {eth: cnt for eth, cnt in ETHNICITIES.items()},
        "ddx_pearls": [
            "CPLANE1/JBTS33: MTS + NORMAL OFC (no microcephaly) — distinguishes from JBTS32/KIF14 where OFC ≤ −2 SD in 100%.",
            "POLYDACTYLY ~24% (higher than average JBTS ~18%) — PCP pathway involvement; GLI3 processing disrupted by mispositioned BB → postaxial digit specification.",
            "ARL13B present in cilia on IF staining (CPLANE1 is pre-ciliary; TZ intact) — distinguishes from JBTS30/TULP3 (ARL13B absent) and JBTS1/INPP5E.",
            "GT335 (axonemal polyglutamylation) NORMAL — distinguishes from JBTS29/TOGARAM1 where GT335 is pathognomonic reduced.",
            "CILIA SHORT (50–70% WT) but present — BB docks but at aberrant geometry; IFT partially impaired downstream; cilia not absent.",
            "NO MKS TIER: CPLANE1 is not a TZ B9-complex protein; all JBTS33 patients liveborn regardless of allele severity.",
            "Co-sequence INTURNED (INTU) and FUZZY (FUZ) in unresolved CPLANE1 compound hets — digenic CPLANE complex interactions documented.",
            "Retinal dystrophy ~22% — photoreceptor outer-segment cilia have BB-docking defect; progressive rod-cone; electroretinogram early.",
            "Renal NPHP-like 18% — collecting duct primary cilia BB mispositioned; UACR + kidney ultrasound at diagnosis.",
            "No hepatic CHF as primary feature — CPLANE1 is not a biliary-tree TZ protein; hepatic involvement mild/incidental if present.",
        ],
        "cplane_complex": {
            "CPLANE1":   "Central scaffold; INTURNED binding, BB docking geometry (*614571)",
            "INTURNED":  "PCP signal transducer; CPLANE1 partner; apical actin organiser",
            "FUZZY/FUZ": "Vesicular trafficking effector; CPLANE1 partner; RAB11/EHD route to BB",
        },
    }


def get_breakdown():
    patients = _patients()
    n = len(patients)
    yes = lambda k: sum(1 for p in patients if p.get(k) == "yes")
    pct = lambda k, v="yes": round(sum(1 for p in patients if p.get(k) == v) / n * 100)

    phenotype_prevalence = {
        "MTS (100pct_diagnostic)":      100,
        "Normal OFC (100pct)":          100,
        "Cerebellar ataxia":            pct("ataxia"),
        "Neonatal hypotonia":           pct("hypotonia"),
        "Oculomotor apraxia":           pct("oculomotor_apraxia"),
        "Breathing dysregulation":      pct("breathing_dysreg"),
        "Intellectual disability":      pct("intellectual_disability"),
        "Retinal rod-cone":             pct("retinal"),
        "Renal NPHP-like":             pct("renal"),
        "Hepatic mild CHF":            pct("hepatic"),
        "Polydactyly postaxial":       pct("polydactyly", "postaxial"),
    }

    # Cilia length distribution
    cilia_length_dist = {}
    for p in patients:
        # CPLANE1 hypomorphic → BB mispositioned → cilia shortened (50–70% WT)
        cat = rng.choice(["50–60% WT (severe BB misdocking)", "60–70% WT (moderate BB misdocking)", "70–80% WT (mild BB misdocking)"])
        cilia_length_dist[cat] = cilia_length_dist.get(cat, 0) + 1

    # BB docking angle distribution
    bb_angle_dist = {}
    for p in patients:
        cat = rng.choice(["Misdocked >30° off-axis (severe)", "Misdocked 15–30° off-axis (moderate)", "Misdocked <15° off-axis (mild)"])
        bb_angle_dist[cat] = bb_angle_dist.get(cat, 0) + 1

    # Allele classes
    allele_class_dist = {
        "Biallelic missense (both hypomorphic)": 0,
        "Compound het missense + splice": 0,
        "Biallelic splice (partial skip)": 0,
    }
    for p in patients:
        v1, v2 = p["variant_1"], p["variant_2"]
        if "splice" in v1 or "splice" in v2:
            allele_class_dist["Compound het missense + splice"] += 1
        else:
            allele_class_dist["Biallelic missense (both hypomorphic)"] += 1

    # Polydactyly PCP analysis
    poly_analysis = {
        "Postaxial polydactyly (PCP/GLI3)": sum(1 for p in patients if p["polydactyly"] == "postaxial"),
        "No polydactyly": sum(1 for p in patients if p["polydactyly"] == "no"),
    }

    return {
        "disease": DISEASE,
        "phenotype_prevalence": phenotype_prevalence,
        "cilia_length_distribution": cilia_length_dist,
        "bb_docking_angle_distribution": bb_angle_dist,
        "allele_class_distribution": allele_class_dist,
        "polydactyly_pcp_analysis": poly_analysis,
        "key_variants": [
            {
                "variant":    "Thr365Met (c.1094C>T)",
                "domain":     "CC1 / INTURNED-FUZ binding interface (aa ~365)",
                "population": "Middle Eastern / North African (MENA consanguineous cluster)",
                "frequency":  "Most common MENA allele; CC1 helix → INTU/FUZ complex partial disassembly → BB docking ~50–60% WT",
                "severity":   "Moderate-severe; JBTS33; cerebellar ataxia; polydactyly ~30% in this variant cluster",
            },
            {
                "variant":    "Ala523Val (c.1568C>T)",
                "domain":     "IDR linker / CC1-IDR junction (aa ~523)",
                "population": "European compound het",
                "frequency":  "European; IDR linker → conformational flexibility impaired; PCP-switch signalling partially blocked",
                "severity":   "Moderate; JBTS33; normal OFC; mild ID; ataxia present; retinal 15% in this variant",
            },
            {
                "variant":    "Leu712Pro (c.2135T>C)",
                "domain":     "WD40-like β-propeller entry (aa ~712)",
                "population": "South Asian consanguineous",
                "frequency":  "South Asian; WD40-like domain β-strand destabilisation → BB-docking anchor geometry impaired",
                "severity":   "Moderate-severe; JBTS33; polydactyly higher frequency; oculomotor apraxia prominent",
            },
            {
                "variant":    "Arg848Gln (c.2543G>A)",
                "domain":     "WD40-like β-propeller core (aa ~848)",
                "population": "North African consanguineous",
                "frequency":  "North African; propeller core → BB-docking interface partially disrupted; cilia 55–65% WT",
                "severity":   "Moderate; JBTS33; NPHP-like renal 22% in this cluster; ID moderate",
            },
            {
                "variant":    "Glu1101Lys (c.3301G>A)",
                "domain":     "C-terminal coiled-coil / membrane anchor (aa ~1,101)",
                "population": "European / multi-ethnic compound het",
                "frequency":  "European; C-term CC → reduced PI4P membrane binding; BB transport to apical surface impaired",
                "severity":   "Mild-moderate; JBTS33; milder cerebellar features; normal or borderline retinal",
            },
            {
                "variant":    "c.1219+2T>C (splice donor)",
                "domain":     "Intron 9 splice donor / CC1–IDR junction",
                "population": "European compound het",
                "frequency":  "European; splice donor → partial exon 9 skip; ~40% residual CPLANE1 → mild-moderate JBTS33",
                "severity":   "Variable mild-moderate; JBTS33; residual ~40% function; milder MTS; polydactyly 18%",
            },
        ],
        "cohort_table": [
            {
                "id":          p["id"],
                "ethnicity":   p["ethnicity"],
                "mts":         p["mts"],
                "ofc":         "normal",
                "ataxia":      p["ataxia"],
                "hypotonia":   p["hypotonia"],
                "id_severity": p["id_severity"],
                "retinal":     p["retinal"],
                "polydactyly": p["polydactyly"],
                "variant_1":   p["variant_1"],
            }
            for p in patients
        ],
    }


def get_definitions():
    return {
        "disease": DISEASE,
        "gene":    GENE,
        "definitions": [
            {
                "term":       "CPLANE1",
                "definition": "Ciliogenesis and Planar Cell Polarity Effector 1 (~1,373 aa; 16q24.1; OMIM *614571). Also known as CFAP126 and FLTP (Flattop). A cytoplasmic scaffolding protein that forms the CPLANE complex with INTURNED (INTU) and FUZZY (FUZ). CPLANE1 reads PCP (planar cell polarity) cues and directs basal body (BB) migration from the pericentriolar region to the apical membrane, where the BB docks to initiate ciliogenesis. CPLANE1 LOF → BB misdocked at wrong apical geometry → cilia short and structurally abnormal → reduced Hedgehog signalling in cerebellar granule progenitors → cerebellar vermis hypoplasia → MTS.",
            },
            {
                "term":       "JBTS33",
                "definition": "Joubert Syndrome Type 33 (OMIM #617409). Caused by biallelic LOF alleles (hypomorphic missense / splice) in CPLANE1. Key features: (1) Molar Tooth Sign on brain MRI (100% by definition), (2) NORMAL OFC (no microcephaly — contrasts with JBTS32/KIF14), (3) cerebellar ataxia, (4) neonatal hypotonia, (5) postaxial polydactyly ~24% (PCP-mediated; higher than average JBTS), (6) intellectual disability in ~64%, (7) retinal dystrophy ~22%, (8) renal NPHP-like ~18%. No MKS tier. All patients liveborn.",
            },
            {
                "term":       "CPLANE Complex (CPLANE1 + INTURNED + FUZZY)",
                "definition": "A cytoplasmic complex required for basal body (BB) docking and ciliogenesis. CPLANE1 scaffolds INTURNED (INTU) — the PCP signal transducer that senses the VANGL2/FZD3 polarity axis — and FUZZY (FUZ) — which mediates vesicular trafficking of BB-directed cargo via RAB11/EHD vesicles. Disruption of any component (CPLANE1, INTU, or FUZ) impairs BB apical docking, causing short, misoriented cilia and a ciliopathy phenotype.",
            },
            {
                "term":       "Basal Body (BB) Docking — CPLANE1 Mechanism",
                "definition": "The basal body is the modified mother centriole from which the primary (or motile) cilium grows. Ciliogenesis requires: (1) BB migration from the perinuclear region to the apical cell surface (cytoplasmic vesicular trafficking via RAB11/EHD/FUZ); (2) BB docking to the apical membrane via appendage proteins and membrane phosphoinositides. CPLANE1 coordinates step (1)–(2): it senses the PCP axis (via INTU) and drives vectorial BB trafficking (via FUZ). Mispositioned BB → cilia grow at wrong orientation → CSF flow deficit / Hh signal geometry aberrant → MTS.",
            },
            {
                "term":       "Planar Cell Polarity (PCP) in Ciliogenesis",
                "definition": "Planar cell polarity is the coordinated, directional organisation of cells within the plane of an epithelium, governed by the non-canonical Wnt/PCP pathway (VANGL1/2, CELSR1/2/3, FZD3/6, PRICKLE1/2, DISHEVELLED). In ciliated epithelia, PCP determines: (1) BB docking angle (cilia beat in a coordinated direction), (2) GLI3 processing polarity in the limb bud (relevant to polydactyly), (3) cerebellar granule progenitor cilia orientation for Hh gradient sensing. CPLANE1 is a direct PCP effector downstream of INTURNED; its LOF disrupts PCP-driven BB positioning, explaining the postaxial polydactyly enrichment in JBTS33 (~24% vs ~18% average JBTS).",
            },
            {
                "term":       "No MKS Tier in JBTS33",
                "definition": "Meckel-Gruber Syndrome (MKS) lethal ciliopathy results from null mutations in Transition Zone (TZ) diffusion-barrier proteins (e.g. B9D1, B9D2, MKS1, TMEM216, TMEM67). CPLANE1 is a cytoplasmic BB-docking/PCP effector upstream of TZ formation; it does not constitute part of the TZ diffusion barrier. Therefore biallelic CPLANE1 LOF does not reproduce the MKS phenotype in humans. All reported JBTS33 patients are liveborn. This contrasts with JBTS34/B9D2 (null → MKS10 lethal) and JBTS28/MKS1 (null → MKS1 lethal).",
            },
            {
                "term":       "Polydactyly in JBTS33 — PCP/GLI3 Mechanism",
                "definition": "Postaxial polydactyly occurs in ~24% of JBTS33 patients — higher than the ~18% average across all Joubert subtypes. Mechanism: correct BB docking angle in limb bud mesenchyme is required for directional Sonic Hedgehog (SHH) gradient sensing. Mispositioned cilia (CPLANE1 LOF) → aberrant GLI3 full-length/repressor (GLI3FL/GLI3R) ratio in the posterior limb bud → posterior digit specification expanded → postaxial extra digit. This PCP-driven polydactyly mechanism is distinct from GLI3 truncation (Pallister-Hall syndrome) and from ARL13B-mediated polydactyly (JBTS8).",
            },
            {
                "term":       "GT335 (Axonemal Polyglutamylation) — Normal in JBTS33",
                "definition": "GT335 is an antibody detecting glutamylated tubulin in axonemes — the hallmark IF biomarker for JBTS29/TOGARAM1 where GT335 signal is pathognomonic reduced (TOGARAM1 is the axonemal polyglutamylase). In JBTS33, axonemal structure is impaired downstream of BB misdocking (IFT-A/B geometry altered), but the polyglutamylation machinery (TTLL enzymes) itself is intact. GT335 signal in JBTS33 cilia is normal to slightly reduced (consistent with cilia shortening, not abolished polyglutamylation). GT335 IF: reduced → suspect JBTS29/TOGARAM1; normal/mildly reduced + short cilia → consider JBTS33/CPLANE1.",
            },
            {
                "term":       "ARL13B in JBTS33 cilia",
                "definition": "ARL13B is a ciliary membrane GTPase that depends on TZ diffusion-barrier integrity for its ciliary localisation. In JBTS33, the TZ barrier is intact (CPLANE1 is pre-TZ), so ARL13B is PRESENT in cilia on immunofluorescence — in contrast to JBTS30/TULP3 (ARL13B absent, TZ intact but cargo adaptor missing) and JBTS1/INPP5E (ARL13B present; INPP5E absent). ARL13B IF: PRESENT in JBTS33. Absence of ARL13B in short cilia points to IFT-A adaptor defect (TULP3/JBTS30) not CPLANE1.",
            },
            {
                "term":       "INTURNED (INTU) — Co-sequencing Recommendation",
                "definition": "INTURNED (INTU; also PCP protein INTU) is an obligate CPLANE complex partner of CPLANE1. INTU variants cause Joubert syndrome in their own right (JBTS-linked). In families with an unresolved CPLANE1 compound heterozygote (one pathogenic allele identified, one missing), INTU should be co-sequenced for a possible digenic or second-hit interaction. Similarly, FUZZY (FUZ) is the third CPLANE complex member; FUZ variants are documented in orofaciodigital syndromes and ciliopathies.",
            },
            {
                "term":       "Renal Phenotype (NPHP-like, ~18%) in JBTS33",
                "definition": "Renal involvement in JBTS33 occurs in ~18% of patients, presenting as nephronophthisis (NPHP)-like tubulointerstitial nephritis — the classic renal ciliopathy phenotype. The mechanism: primary cilia in the collecting duct and loop of Henle depend on correctly docked BB for proper cilia orientation; CPLANE1 LOF → BB misdocked in collecting duct cells → cilia mechanosensing impaired → tubular cysts and interstitial fibrosis. Severity: typically CKD stage 1–2 by mid-childhood; progression to ESRD is uncommon but documented. Annual urine ACR + eGFR monitoring from diagnosis.",
            },
            {
                "term":       "MTS (Molar Tooth Sign) — JBTS33",
                "definition": "Brain MRI finding pathognomonic for Joubert syndrome spectrum: elongated superior cerebellar peduncles (SCPs) + cerebellar vermis hypoplasia create a 'molar tooth' appearance on axial MRI. In JBTS33: MTS present 100% (diagnostic criterion). OFC is normal (distinguishing from JBTS32/KIF14 where OFC ≤ −2 SD is universal). Simplified gyral pattern is NOT a feature of JBTS33 (reflecting intact cortical cytokinesis in CPLANE1 LOF — cytokinesis is normal; only cilia geometry affected).",
            },
        ],
    }
