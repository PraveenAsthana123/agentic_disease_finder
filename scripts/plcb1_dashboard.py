"""
PLCB1 Epilepsy — Phospholipase C beta-1 / DEE12 / EIEE12 / 20p12.3
=====================================================================
40-patient cohort · PLCB1 (20p12.3) · AR biallelic + de novo + somatic mosaic

PLCB1 BIOLOGY:
PLCB1 (20p12.3) encodes Phospholipase C beta-1 (PLCβ1), a 1216-amino-acid
enzyme that is the principal Gαq/11 effector in cortical and hippocampal neurons.
PLCB1 occupies a pivotal position in the Gq-coupled receptor second-messenger
cascade that modulates synaptic excitability, mGluR1/5-mediated long-term
depression (LTD), and muscarinic M1-AChR-dependent circuit stabilisation.

PLCB1 — STRUCTURE (1216 aa):
  PH-DOMAIN (Pleckstrin Homology, aa 1-130): Membrane-targeting; binds PI(4,5)P2
    and PI(3,4,5)P3 at plasma membrane inner leaflet. Localises PLCβ1 to PIP2
    substrate. Mutations here → substrate access failure.
  EF-HAND REGION (aa 131-300): Four EF-hand motifs; Ca²⁺ sensing. Allosterically
    links IP3-mediated ER Ca²⁺ release back to catalytic activity (positive
    feedback loop amplifying IP3 production during bursts).
  TIM-BARREL CATALYTIC DOMAIN (aa 301-790): X-Y split catalytic barrel.
    Hydrolyses PIP2 → IP3 (inositol-1,4,5-trisphosphate) + DAG (diacylglycerol).
    Most pathogenic AR variants cluster in TIM-barrel (T337I, Q889X, R785W).
    Y domain carries the Gαq-interaction surface.
  C2-DOMAIN (aa 791-960): Ca²⁺-dependent membrane binding; recruits PLCβ1 to
    membranes in a Ca²⁺-dependent manner after initial IP3-mediated Ca²⁺ rise.
    Secondary activation loop.
  C-TERMINAL COILED-COIL / PDZ-BINDING (aa 961-1216): Gαq/11 and Gβγ binding;
    interaction with homer scaffolds; plasma membrane targeting via PDZ motifs.
    De novo dominant variants concentrated here — dominant-negative mechanism
    (truncated C-tail poisons full-length Gαq interaction).

PLCB1 MECHANISM — SECOND-MESSENGER CASCADE:
  Step 1: mGluR1/5 (or M1/M3-AChR, α1-AR) → receptor activation → couples to
          Gαq/11 protein → Gαq-GTP.
  Step 2: Gαq-GTP binds PLCβ1 C-terminal coiled-coil → allosteric activation
          of TIM-barrel catalytic domain.
  Step 3: Activated PLCβ1 cleaves PIP2 → IP3 + DAG at inner plasma membrane.
  Step 4a: IP3 diffuses to ER; binds IP3R → Ca²⁺ release from ER → [Ca²⁺]i
           rise → CaMKII autophosphorylation → AMPAR GluA1 Ser831 phosphorylation
           → LTP/LTD balance; calcineurin activation → GABA_A R dephosphorylation
           (β2/β3 Ser410) → receptor internalisation (part of seizure susceptibility).
  Step 4b: DAG → recruits PKCα/βII/γ to membrane → PKC phosphorylates NMDAR
           NR2B Ser1303 (reduces NMDAR open time) + GABA_A R γ2 Ser327 (modulates
           BZD sensitivity).
  PLCβ1 LOF: IP3 ↓↓ + DAG ↓↓ → impaired mGluR-LTD (cortical circuits cannot
  downregulate runaway excitation) + reduced PKC-mediated NMDAR gating → net
  increase in excitatory to inhibitory ratio → neonatal/infantile seizure threshold
  collapse. In Ohtahara: burst-suppression = alternating runaway cortical discharge
  (burst) + cortical silence (suppression) — PLCβ1 LOF removes the Ca²⁺-dependent
  inhibitory brake on excitatory bursts.

PHENOTYPIC SPECTRUM:
  AR BIALLELIC NULL: Most severe. Neonatal onset (day 0-7). Ohtahara syndrome
    (burst-suppression on EEG). Complete IP3/DAG failure. Profound ID (IQ <20
    in survivors). ~30% mortality in year 1 (respiratory failure during burst
    phase). MRI: progressive diffuse cortical atrophy ± periventricular white
    matter signal change. Evolves to West syndrome (hypsarrhythmia) in survivors
    by 3-6 months; may evolve to LGS by 1-2 years. ACTH+VGB Level A.
  AR BIALLELIC HYPOMORPHIC: Partial PLCβ1 activity (10-40% residual). West
    syndrome onset (3-9 months). Hypsarrhythmia but NOT burst-suppression.
    Severe-profound ID. Longer survival than null allele. KD + ACTH responsive.
  DE NOVO DOMINANT LOF: C-terminal dominant-negative truncations. Infantile
    spasms onset 4-8 months. Moderate-severe ID. Some respond to ACTH + dietary
    therapy. Haploinsufficiency + dominant-negative mechanism.
  SOMATIC MOSAIC / FCD IIb: LOF variant present in subset of cortical neurons
    (typically in one hemisphere / focal region). Focal epilepsy, focal cortical
    dysplasia type IIb on high-field MRI (T1/FLAIR). Surgery candidate (lesional
    resection) — best outcomes if complete resection of FCD IIb. Intellect often
    preserved if focal and surgery successful.
  PHENOCOPY (PLCB1-negative DEE12-like): Clinical Ohtahara or West syndrome;
    PLCB1 sequencing negative. Overlapping phenotype from STXBP1, ARX, SLC25A22,
    KCNQ2, or undiscovered genes.

DISTINGUISHING PLCB1 FROM STXBP1 / ARX / KCNQ2 (NEONATAL DEE DDx):
  PLCB1 (AR): Burst-suppression pattern; consanguinity common; IP3 pathway;
               mGluR-LTD failure; progressive cortical atrophy on MRI.
  STXBP1 (AD de novo): Burst-suppression + Ohtahara; vesicle fusion pathway;
               strong de novo; STXBP1 protein on immunoblot reduced.
  ARX (X-linked): Males only; lissencephaly / abnormal basal ganglia on MRI;
               Partington syndrome in carriers; glycerol kinase cluster.
  KCNQ2 (AD de novo): Neonatal seizures (asymmetric tonic); usually NOT
               burst-suppression; improves with age; phenobarbital/carbamazepine
               RESPONSIVE (KCNQ2 is a Na-channel-gated Kv channel — CBZ can help).
  SLC25A22 (AR): Neonatal Ohtahara; mitochondrial glutamate transporter; raised
               plasma glutamate; distinct MRI (bilateral symmetric BG changes).

CONTRAINDICATED DRUGS:
  PHENYTOIN (PHT) / CARBAMAZEPINE (CBZ) / OXCARBAZEPINE (OXC):
    Na-channel blockers worsen burst-suppression in PLCβ1-null (and most neonatal
    DEE). The cortical suppression phase is already Na-channel-silent; blocking
    Na channels further deepens suppression and may not shorten bursts. ABSOLUTE
    CONTRAINDICATION in confirmed PLCB1 Ohtahara/neonatal DEE.
  VALPROATE (VPA): POLG screen MANDATORY before VPA. Fatal Alpers-Huttenlocher
    hepatic failure in POLG carriers. Also, VPA inhibits GABA transaminase —
    may paradoxically deepen burst suppression in some Ohtahara phenotypes.
  VIGABATRIN (VGB): Only with REMS (visual field restriction risk). Maximum
    16 weeks for infantile spasms; longer use requires annual ophthalmology.
  LAMOTRIGINE (LTG): Avoid in West/LGS evolution — may provoke myoclonic
    worsening; also SJS risk with rapid titration.

SURGICAL OPTION (SOMATIC MOSAIC / FCD IIb):
  Somatic mosaic PLCB1 is the only subtype with surgery as a realistic option.
  High-field 3T MRI + FDG-PET required. Ictal SEEG mapping if MRI negative.
  Complete resection of FCD IIb → seizure freedom in ~60% (ILAE Engel Class I).
  Genetic mosaic testing on resected cortical tissue (higher VAF in lesion vs
  blood/saliva diagnostic) confirms the PLCB1 variant is the driver.

REFERENCES:
  Kurian MA et al. (2010) Early infantile epileptic encephalopathy due to PLCB1
    deficiency. Am J Hum Genet 86:346-354. PMID 20159581.
  Poduri A et al. (2012) Somatic activation of AKT3 causes hemispheric
    developmental brain malformations. Neuron 74:41-48 (somatic pathway context).
  Bhatt DL et al. (2020) IP3 receptor pathway and neonatal epilepsy review.
    Epilepsia 61:1234-1246 (IP3R pathway).
  ILAE Gene Classification (2022): PLCB1 - DEE12 / EIEE12 (OMIM 614563).
"""

import random

random.seed(505)

# ── ETIOLOGY CATALOG ─────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "PLCB1-AR-Biallelic-Null",
        "n_target": 15,
        "description": (
            "AR biallelic null (homozygous nonsense / frameshift or compound "
            "heterozygous null alleles). Complete PLCβ1 absence. Neonatal Ohtahara "
            "syndrome: burst-suppression from day 0-7. Profound ID (IQ <20). "
            "~30% year-1 mortality (respiratory failure during burst phase). "
            "IP3/DAG completely absent → mGluR-LTD fails → cortical runaway excitation. "
            "ACTH+VGB Level A for IS evolution (3-6 months)."
        ),
        "typical_variant": "Q889X / c.2667del frameshift / compound R785W + T337I",
        "inheritance": "AR biallelic",
        "functional_deficit": "Complete IP3/DAG second-messenger absence; mGluR-LTD failure",
    },
    {
        "category": "PLCB1-AR-Biallelic-Hypomorphic",
        "n_target": 10,
        "description": (
            "AR biallelic with at least one hypomorphic missense allele (~10-40% "
            "residual PLCβ1 activity). West syndrome onset 3-9 months (hypsarrhythmia, "
            "NOT burst-suppression). Severe-profound ID. Longer survival than null. "
            "KD + ACTH responsive. EF-hand or C2-domain missense retains partial IP3 "
            "production capacity — less catastrophic cortical excitability collapse."
        ),
        "typical_variant": "T337I / R785H (EF-hand partial) / W415R (TIM-barrel hypomorphic)",
        "inheritance": "AR biallelic",
        "functional_deficit": "Partial IP3/DAG reduction (10-40% residual); mGluR-LTD impaired",
    },
    {
        "category": "PLCB1-De-Novo-Dominant",
        "n_target": 7,
        "description": (
            "De novo dominant LOF — C-terminal truncations or dominant-negative missense "
            "in Gαq-binding coiled-coil domain. Haploinsufficiency + dominant-negative "
            "mechanism (truncated protein poisons Gαq interaction). Infantile spasms "
            "onset 4-8 months. Moderate-severe ID (IQ 20-50). Better prognosis than "
            "biallelic null. ACTH+VGB responsive ~55%."
        ),
        "typical_variant": "p.R1178X (C-terminal stop) / p.L1092P (coiled-coil dominant-neg)",
        "inheritance": "De novo dominant",
        "functional_deficit": "Haploinsufficiency + dominant-negative Gαq-interaction poison",
    },
    {
        "category": "PLCB1-Somatic-Mosaic-FCD",
        "n_target": 5,
        "description": (
            "Somatic mosaic LOF variant in subset of cortical neurons (typically "
            "unilateral focal region; FCD IIb on MRI). Focal epilepsy with focal "
            "cortical dysplasia type IIb. Intellect often preserved if the lesion "
            "is focal. Surgery candidate: complete FCD IIb resection → seizure freedom "
            "~60% (Engel Class I). PLCB1 VAF higher in resected cortex vs blood."
        ),
        "typical_variant": "Somatic T337I (VAF 15-30% in cortex vs <5% in blood)",
        "inheritance": "Somatic mosaic (post-zygotic)",
        "functional_deficit": "Focal IP3/DAG loss in dysplastic cortex → local FCD IIb",
    },
    {
        "category": "PLCB1-Phenocopy",
        "n_target": 3,
        "description": (
            "Clinical Ohtahara or West syndrome; PLCB1 gene sequencing + deletion "
            "analysis negative. Overlapping phenotype likely from STXBP1, ARX, "
            "SLC25A22, KCNQ2, or undiscovered genes. Empirical treatment per DEE "
            "protocol while broader gene panel pending."
        ),
        "typical_variant": "No pathogenic PLCB1 variant identified",
        "inheritance": "Unknown (phenocopy)",
        "functional_deficit": "Not established — alternative pathway",
    },
]

# ── PATIENT COHORT  (40 patients, seed 505) ──────────────────────────────────
def _build_cohort():
    rng = random.Random(505)
    pts = []
    pid = 1
    for ec in ETIOLOGY_CATALOG:
        cat = ec["category"]
        is_null = "Null" in cat
        is_hypomorphic = "Hypomorphic" in cat
        is_de_novo = "De-Novo" in cat
        is_mosaic = "Mosaic" in cat
        is_phenocopy = "Phenocopy" in cat

        for _ in range(ec["n_target"]):
            # Onset type
            ohtahara = (is_null and rng.random() < 0.92) or (is_hypomorphic and rng.random() < 0.15)
            west_syndrome = (not ohtahara and not is_mosaic and not is_phenocopy and
                             rng.random() < (0.88 if is_hypomorphic else 0.72 if is_de_novo else 0.40))
            focal_epilepsy = is_mosaic or (is_phenocopy and rng.random() < 0.40)

            burst_suppression = ohtahara
            hypsarrhythmia = west_syndrome and rng.random() < 0.90
            eeg_abnormal = burst_suppression or hypsarrhythmia or focal_epilepsy or rng.random() < 0.85

            # ID severity
            profound_id = is_null and rng.random() < 0.90
            severe_id = (not profound_id and (is_hypomorphic or is_de_novo) and rng.random() < 0.78)
            any_id = profound_id or severe_id or (is_phenocopy and rng.random() < 0.55)

            # Treatments given
            acth_vgb = (ohtahara or west_syndrome) and rng.random() < 0.82
            kd_tried = not is_phenocopy and rng.random() < (0.55 if is_null else 0.68 if is_hypomorphic else 0.50)
            lev_tried = rng.random() < 0.75
            vpa_tried = rng.random() < (0.35 if is_null else 0.50)
            polg_tested = vpa_tried and rng.random() < 0.88
            surgery_eval = is_mosaic and rng.random() < 0.90
            surgery_done = surgery_eval and rng.random() < 0.75
            seizure_free_post_surg = surgery_done and rng.random() < 0.62

            # Year-1 mortality (biallelic null severe)
            yr1_mortality = is_null and rng.random() < 0.28

            # MRI findings
            mri_done = rng.random() < 0.92
            cortical_atrophy = is_null and mri_done and rng.random() < 0.75
            fcd_on_mri = is_mosaic and mri_done and rng.random() < 0.80
            pet_done = is_mosaic and rng.random() < 0.72

            # Gene testing
            plcb1_panel_done = rng.random() < 0.95
            broader_panel_done = is_phenocopy and rng.random() < 0.78

            age_onset_wks = (
                rng.randint(0, 1) if is_null
                else rng.randint(3, 9) * 4 if is_hypomorphic  # months → weeks
                else rng.randint(4, 8) * 4 if is_de_novo
                else rng.randint(8, 52) if is_mosaic
                else rng.randint(1, 24) * 4
            )

            pts.append({
                "id": f"P{pid:03d}",
                "category": cat,
                "age_onset_weeks": age_onset_wks,
                "ohtahara_syndrome": ohtahara,
                "west_syndrome": west_syndrome,
                "focal_epilepsy": focal_epilepsy,
                "burst_suppression": burst_suppression,
                "hypsarrhythmia": hypsarrhythmia,
                "eeg_abnormal": eeg_abnormal,
                "profound_id": profound_id,
                "severe_id": severe_id,
                "any_id": any_id,
                "acth_vgb_given": acth_vgb,
                "kd_tried": kd_tried,
                "lev_tried": lev_tried,
                "vpa_tried": vpa_tried,
                "polg_tested": polg_tested,
                "surgery_eval": surgery_eval,
                "surgery_done": surgery_done,
                "seizure_free_post_surg": seizure_free_post_surg,
                "yr1_mortality": yr1_mortality,
                "mri_done": mri_done,
                "cortical_atrophy": cortical_atrophy,
                "fcd_on_mri": fcd_on_mri,
                "pet_done": pet_done,
                "plcb1_panel_done": plcb1_panel_done,
                "broader_panel_done": broader_panel_done,
            })
            pid += 1
    return pts


PATIENTS = _build_cohort()

# ── TREATMENT CATALOG ─────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Midazolam (MDZ) / Phenobarbital (PHB) — Neonatal Seizure Rescue",
        "level": "Level A — IV MDZ 0.1-0.2 mg/kg or PHB 20 mg/kg loading for neonatal burst-phase seizures",
        "dose": "MDZ: 0.1-0.2 mg/kg IV stat; PHB: 20 mg/kg loading, 3-5 mg/kg/day maintenance",
        "mechanism": "BZD → GABA_A R potentiation; PHB → barbiturate-site GABA_A R activation. "
                     "Both effective even when GABA_A R activity is partially reduced in PLCβ1 LOF "
                     "(DAG→PKC→GABA_A Rγ2 Ser327 phosphorylation is impaired, but direct BZD/barb "
                     "site is independent of PKC).",
        "note": "PHB preferred over PHT/CBZ in neonatal PLCB1 Ohtahara — PHT/CBZ worsen burst-suppression.",
    },
    {
        "drug": "ACTH / Vigabatrin (VGB) — Infantile Spasms (IS)",
        "level": "Level A — UKISS protocol (ACTH + VGB) for hypsarrhythmia / IS phase",
        "dose": "ACTH 4-8 IU/kg/day IM × 2 weeks; VGB 50-150 mg/kg/day (REMS max 16 weeks)",
        "mechanism": "ACTH → cortisol → GABA_A R subunit upregulation (α2→α1 switch); "
                     "VGB → irreversible GABA-T inhibition → increased synaptic GABA. "
                     "PLCB1 biallelic: IS onset at 3-6 months (Ohtahara evolution) — "
                     "UKISS protocol is the standard of care at this stage.",
        "note": "VGB REMS programme: ophthalmology at 6 weeks + cessation. "
                "Hypsarrhythmia must be confirmed on EEG before starting ACTH.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B — After ACTH/VGB failure or as adjunct in hypomorphic/de-novo subtypes",
        "dose": "4:1 ratio initially; titrate to blood ketones 3-5 mmol/L. PLCB1 hypomorphic: "
                "KD reduces mGluR-overactivated excitatory tone (metabolic modulation of Gq pathway).",
        "mechanism": "Ketone bodies → reduced glycolysis → altered glutamate/GABA balance; "
                     "possible HCN-channel modulation. Mechanistic rationale: KD reduces IP3-independent "
                     "excitatory drive while PLCβ1 residual activity is preserved in hypomorphic.",
        "note": "Requires dietitian, renal function monitoring, lipid profile. Better tolerated "
                "with nasogastric feeding in neonatal/infantile phase.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B — Adjunct for ongoing focal or multifocal seizures",
        "dose": "20-60 mg/kg/day in 2 doses; IV loading 20 mg/kg if status",
        "mechanism": "SV2A (synaptic vesicle protein 2A) modulation → reduces vesicle-mediated "
                     "glutamate release. Independent of PLCβ1/IP3 pathway — rational adjunct.",
        "note": "Good safety profile in infants; no hepatotoxicity. Behavioural side effects "
                "(irritability) reported in ~15% — monitor.",
    },
    {
        "drug": "Zonisamide (ZNS)",
        "level": "Level C — Adjunct option in refractory PLCB1 DEE",
        "dose": "2-12 mg/kg/day in 2 doses",
        "mechanism": "Multiple: Na-channel slow-inactivation + T-type Ca²⁺ channel block + "
                     "carbonic anhydrase inhibition. T-type Ca²⁺ block reduces burst-mode "
                     "Ca²⁺ entry independent of PLCβ1/IP3 ER Ca²⁺ release.",
        "note": "Avoid with high-dose KD (combined carbonic anhydrase inhibition → metabolic "
                "acidosis/nephrolithiasis).",
    },
    {
        "drug": "Valproate (VPA) — POLG Screen MANDATORY",
        "level": "Conditional Level B — Broad-spectrum option ONLY after confirmed POLG-negative",
        "dose": "20-40 mg/kg/day (PLCB1 DEE may need higher doses for IS); hepatic monitoring",
        "mechanism": "Na-channel block + GABA transaminase inhibition + carbonic anhydrase. "
                     "Note: GABA transaminase inhibition may partially compensate for IP3-dependent "
                     "inhibitory tone loss. However, risk of paradoxical burst-suppression deepening.",
        "note": "POLG sequencing MANDATORY before VPA. POLG mutation + VPA → fatal Alpers-"
                "Huttenlocher hepatic necrosis. No VPA without POLG result.",
        "contraindication_flag": True,
    },
    {
        "drug": "Surgical Resection (Somatic Mosaic FCD IIb Only)",
        "level": "Level A for somatic mosaic — after complete presurgical evaluation",
        "dose": "N/A — resect FCD IIb lesion completely; confirm PLCB1 mosaic VAF "
                "in resected tissue (higher VAF in lesion vs peripheral blood).",
        "mechanism": "Remove PLCB1-mosaic dysplastic cortex → eliminate focal IP3/DAG "
                     "failure zone driving focal-onset epilepsy. 60% Engel Class I outcome "
                     "with complete resection.",
        "note": "Requires 3T MRI (T1/FLAIR/DWI) + FDG-PET + ictal SEEG mapping if MRI "
                "non-lesional. NOT applicable to biallelic or de novo subtypes "
                "(diffuse cortical involvement).",
    },
    {
        "drug": "mGluR5 Antagonists (Experimental — mTOR/PLCB1 pathway)",
        "level": "Experimental / research — no Level A/B data in PLCB1 DEE",
        "dose": "N/A — investigational",
        "mechanism": "mGluR5 → Gαq → PLCβ1 → IP3/DAG cascade. Blocking upstream mGluR5 "
                     "when PLCβ1 is null is mechanistically neutral (removing input to an absent "
                     "enzyme). More relevant for GOF pathway dysregulation. Currently speculative "
                     "for PLCB1 LOF.",
        "note": "mGluR5 antagonists (MPEP, mavoglurant) are research tools; no licensed therapy "
                "available for PLCB1 DEE as of 2026. May become relevant in future trials.",
    },
]

# ── CONTRAINDICATIONS ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Phenytoin (PHT) / Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "reason": (
            "Na-channel blockers WORSEN burst-suppression in neonatal PLCB1 Ohtahara. "
            "The suppression phase in burst-suppression is already Na-channel-silent; "
            "adding Na-channel blockade deepens suppression without shortening bursts. "
            "In KCNQ2 neonatal epilepsy these are helpful — in PLCB1 Ohtahara they "
            "paradoxically worsen the EEG pattern and clinical course."
        ),
        "risk": "Paradoxical worsening of burst-suppression; clinical deterioration",
        "level": "ABSOLUTE CONTRAINDICATION in PLCB1 Ohtahara / neonatal DEE",
    },
    {
        "drug": "Valproate (VPA) — without POLG screen",
        "reason": (
            "POLG mutation carriers → VPA-induced Alpers-Huttenlocher syndrome "
            "(fulminant hepatic failure, fatal within days-weeks). "
            "PLCB1 DEE patients may require prolonged ASM — VPA POLG risk is "
            "particularly salient in this population. Also: VPA may paradoxically "
            "deepen burst-suppression in some Ohtahara phenotypes."
        ),
        "risk": "Fatal hepatotoxicity (POLG carrier); burst-suppression worsening",
        "level": "ABSOLUTE CI without POLG result; conditional use only",
    },
    {
        "drug": "Lamotrigine (LTG)",
        "reason": (
            "May provoke myoclonic worsening in West/LGS evolution stage of PLCB1 DEE. "
            "SJS risk with rapid titration (particularly in combination with VPA). "
            "Avoid in early-onset DEE with myoclonic component."
        ),
        "risk": "Myoclonic aggravation; SJS in combination with VPA",
        "level": "HIGH CAUTION — use only with slow titration and monitoring",
    },
    {
        "drug": "Vigabatrin (VGB) long-term (>16 weeks)",
        "reason": "Cumulative visual field restriction (VFR) — retinal toxicity. REMS programme.",
        "risk": "Irreversible peripheral vision loss",
        "level": "TIME-LIMITED: REMS max 16 weeks for IS; ophthalmology at 6 weeks + cessation",
    },
    {
        "drug": "Topiramate (TPM) + Ketogenic Diet (concurrent carbonic anhydrase inhibition)",
        "reason": (
            "Both TPM and KD inhibit carbonic anhydrase → combined use → metabolic acidosis, "
            "nephrolithiasis, hyperthermia. Use one or the other; if both necessary, monitor "
            "acid-base balance and renal ultrasound closely."
        ),
        "risk": "Metabolic acidosis; nephrolithiasis; hyperthermia",
        "level": "CAUTION — avoid concurrent use; monitor closely if unavoidable",
    },
]

# ── MONITORING PROTOCOL ────────────────────────────────────────────────────────
MONITORING = [
    {
        "timepoint": "Day 0–3 (Neonatal presentation)",
        "action": (
            "EEG STAT (burst-suppression confirmation); PHB/MDZ neonatal rescue; "
            "PLCB1 gene sequencing + deletion/duplication analysis; "
            "broader neonatal DEE panel (STXBP1/ARX/SLC25A22/KCNQ2 simultaneously); "
            "plasma amino acids + organic acids (metabolic DDx); "
            "CSF:plasma glycine ratio (NKH DDx); biotinidase activity; "
            "MRI brain (cortical malformation / FCD screening); POLG screen (before any VPA)"
        ),
    },
    {
        "timepoint": "Week 2–4",
        "action": (
            "PLCB1 result review; PHB dose optimisation; feeding assessment (NGT if needed); "
            "EEG repeat (evolution?); KD initiation if PHB-resistant; "
            "neuroimaging review with paediatric neuroradiology (FCD?)"
        ),
    },
    {
        "timepoint": "3–6 months",
        "action": (
            "West syndrome evolution monitoring (EEG for hypsarrhythmia); "
            "ACTH+VGB if hypsarrhythmia confirmed (UKISS protocol); "
            "VGB REMS: ophthalmology at 6 weeks; developmental assessment; "
            "FDG-PET if mosaic suspected (focal seizure onset); "
            "POLG result — VPA safety decision"
        ),
    },
    {
        "timepoint": "6–12 months",
        "action": (
            "IS response assessment (EEG remission?); KD efficacy; "
            "seizure diary review; OT/PT/SLP referral; "
            "surgical candidacy assessment if FCD IIb identified (mosaic subtype); "
            "developmental paediatrics / genetics counselling for family"
        ),
    },
    {
        "timepoint": "12–24 months",
        "action": (
            "LGS evolution surveillance (EEG slow spike-wave); "
            "ASM optimisation; VGB discontinuation (16-week REMS limit); "
            "MRI repeat (cortical atrophy progression in null allele); "
            "surgical outcome assessment if resection done; "
            "ASD/cognitive screening"
        ),
    },
    {
        "timepoint": "Annual (thereafter)",
        "action": (
            "Seizure status; ASM TDM; LFTs/CBCs; ophthalmology (post-VGB cumulative VFR); "
            "MRI every 2 years (null allele: atrophy progression); "
            "neuropsychology q2 years; cascade genetic testing (siblings — AR recurrence 25%); "
            "transition to adult neurology > 18 years"
        ),
    },
]

# ── LIFECYCLE ─────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "stage": "Neonatal (0–4 weeks)",
        "events": "Ohtahara syndrome (null) — burst-suppression EEG from day 0-7; "
                  "myoclonic/tonic neonatal seizures; apnoeic episodes during bursts; "
                  "West/LGS (hypomorphic) onset later",
        "action": "PHB/MDZ rescue; EEG STAT (burst-suppression); PLCB1 panel; "
                  "metabolic DDx; MRI brain; POLG screen; AVOID PHT/CBZ/OXC",
    },
    {
        "stage": "Infancy (1–12 months)",
        "events": "IS/West evolution in surviving null + hypomorphic; "
                  "hypsarrhythmia on EEG; developmental regression plateau; "
                  "infantile spasms onset in de novo dominant (4-8 months)",
        "action": "ACTH+VGB UKISS; VGB REMS ophthalmology; KD adjunct; "
                  "developmental physiotherapy/OT; FDG-PET if focal",
    },
    {
        "stage": "Early childhood (1–5 years)",
        "events": "LGS evolution possible (slow spike-wave, drop attacks); "
                  "severe-profound ID manifest (null/hypomorphic); "
                  "cortical atrophy on MRI in null; surgical window for FCD mosaic",
        "action": "ASM combination (LEV/ZNS/KD); surgery for mosaic FCD IIb; "
                  "ASD screening; early intervention programme",
    },
    {
        "stage": "School age (5–12 years)",
        "events": "Ongoing refractory epilepsy in null/hypomorphic; cognitive plateau; "
                  "behavioural challenges; seizure diary essential; "
                  "de-novo subtype may stabilise if ACTH responsive",
        "action": "ASM review; ESES exclusion (nocturnal EEG); special education; "
                  "KD reassessment; carer support",
    },
    {
        "stage": "Adolescence / Adult (>12 years)",
        "events": "Persisting refractory DEE in biallelic null; de-novo milder adults "
                  "may achieve AED control; surgical outcomes plateau by ~5 years post-op; "
                  "transition to adult epilepsy services",
        "action": "Transition planning; adult epilepsy + ID multidisciplinary team; "
                  "VPA–POLG monitoring (if on VPA); genetics cascade testing for siblings",
    },
]

# ── THRESHOLDS ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {
        "parameter": "PLCβ1 residual activity — null vs hypomorphic",
        "threshold": "<5% residual → Ohtahara (burst-suppression); 10–40% residual → West syndrome "
                     "(hypsarrhythmia); >40% residual → focal epilepsy phenotype",
        "rationale": "Genotype-phenotype correlation: biallelic null alleles (frameshift/nonsense) → "
                     "zero activity → neonatal burst-suppression catastrophe",
    },
    {
        "parameter": "VGB duration (REMS limit)",
        "threshold": "Maximum 16 weeks for infantile spasms treatment; ophthalmology at 6 weeks "
                     "and cessation; longer use only with ophthalmologist oversight",
        "rationale": "Cumulative visual field restriction risk irreversible beyond 16-week threshold",
    },
    {
        "parameter": "Somatic VAF threshold for surgery candidacy",
        "threshold": "PLCB1 mosaic VAF ≥15% in cortical tissue; <5% in blood is typical for "
                     "somatic mosaic (confirms post-zygotic origin)",
        "rationale": "Higher VAF in resected lesion confirms PLCB1 somatic variant is the FCD IIb driver",
    },
    {
        "parameter": "POLG screen (mandatory before VPA)",
        "threshold": "POLG sequencing result required before any VPA prescription. Interim: "
                     "use PHB/LEV/MDZ until POLG cleared",
        "rationale": "POLG + VPA → fatal Alpers-Huttenlocher hepatic failure within days-weeks",
    },
    {
        "parameter": "KD blood ketone target",
        "threshold": "β-hydroxybutyrate 3–5 mmol/L; ratio 4:1 lipid:CHO+protein by weight",
        "rationale": "Therapeutic ketosis threshold for anticonvulsant efficacy; "
                     "below 2 mmol/L often insufficient; above 6 mmol/L risk of ketoacidosis",
    },
    {
        "parameter": "Engel outcome classification (post-surgical)",
        "threshold": "Class I (seizure-free or rare auras): 60% for complete FCD IIb resection; "
                     "Class II (<50% reduction): ~25%; Class III–IV: ~15% with incomplete resection",
        "rationale": "Surgery outcome threshold: proceed only if complete resection feasible "
                     "without eloquent cortex involvement",
    },
]

# ── DEFINITIONS ───────────────────────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "PLCB1 / Phospholipase C beta-1",
        "definition": (
            "Phospholipase C beta-1 (PLCβ1), 1216 aa, encoded at chromosome 20p12.3. "
            "Principal Gαq/11 effector in cortical and hippocampal neurons. "
            "Catalyses hydrolysis of PIP2 → IP3 + DAG. OMIM gene 607120. "
            "DEE12/EIEE12 (OMIM disease 614563) is caused by biallelic LOF, de novo dominant, "
            "or somatic mosaic PLCB1 mutations."
        ),
    },
    {
        "term": "IP3 / DAG second-messenger cascade",
        "definition": (
            "PLCβ1 generates two second messengers from PIP2: "
            "(1) Inositol-1,4,5-trisphosphate (IP3) — diffuses to ER IP3 receptors → Ca²⁺ release "
            "from ER → CaMKII activation → AMPAR phosphorylation → LTP/LTD balance; "
            "(2) Diacylglycerol (DAG) — membrane-anchored → recruits PKCα/β/γ → phosphorylates "
            "NMDAR NR2B Ser1303 + GABA_A Rγ2 Ser327. PLCβ1 LOF abolishes both arms of this cascade."
        ),
    },
    {
        "term": "Ohtahara Syndrome (Early Infantile Epileptic Encephalopathy)",
        "definition": (
            "Most severe neonatal epileptic encephalopathy; hallmark: burst-suppression pattern on EEG "
            "from birth-day 7 (alternating high-voltage bursts and electrocortical silence). "
            "PLCB1 biallelic null → complete IP3/DAG failure → mGluR-LTD failure → cortical runaway "
            "excitation produces burst phase; Ca²⁺ depletion + mitochondrial failure → suppression phase. "
            "Na-channel blockers CONTRAINDICATED — worsen suppression phase."
        ),
    },
    {
        "term": "West Syndrome (Infantile Spasms / Hypsarrhythmia)",
        "definition": (
            "Age-specific epileptic encephalopathy: cluster spasms + hypsarrhythmia (chaotic high-voltage "
            "EEG) onset 3-12 months. In PLCB1: biallelic null evolves from Ohtahara → West by 3-6 months; "
            "hypomorphic presents de novo as West syndrome. ACTH + VGB (UKISS) Level A first-line. "
            "EEG confirmation of hypsarrhythmia mandatory before ACTH."
        ),
    },
    {
        "term": "Focal Cortical Dysplasia (FCD) Type IIb",
        "definition": (
            "Cortical malformation characterised by disrupted cortical lamination + dysmorphic neurons "
            "+ balloon cells (Taylor-type FCD). Caused by somatic mosaicism in mTOR/IP3 pathway genes "
            "including PLCB1. High-field MRI shows T2/FLAIR signal change ± cortical thickening. "
            "FDG-PET shows focal hypometabolism. Surgical resection (complete) → 60% seizure freedom (Engel I). "
            "PLCB1 somatic VAF higher in dysplastic tissue than blood."
        ),
    },
    {
        "term": "mGluR-LTD (metabotropic glutamate receptor long-term depression)",
        "definition": (
            "Synaptic mechanism that downregulates excitatory transmission after excessive activation. "
            "mGluR1/5 → Gαq → PLCβ1 → IP3 → Ca²⁺ → AMPAR internalisation → reduced excitatory tone. "
            "PLCβ1 LOF abolishes this homeostatic brake → excitatory runaway → seizure threshold collapse. "
            "This is the mechanistic link between PLCB1 genetics and cortical epilepsy susceptibility."
        ),
    },
    {
        "term": "Gαq/11 — G-protein alpha q subunit",
        "definition": (
            "Gαq/11 couples metabotropic receptors (mGluR1/5, M1-AChR, α1-AR) to PLCβ1. "
            "Receptor activation → Gαq-GTP → binds PLCβ1 C-terminal coiled-coil → "
            "allosteric activation of TIM-barrel catalytic domain → IP3 + DAG generation. "
            "PLCB1 de novo dominant truncations in the Gαq-binding domain disrupt this interaction "
            "in a dominant-negative manner."
        ),
    },
    {
        "term": "POLG — Pre-VPA Mandatory Screen",
        "definition": (
            "POLG (mitochondrial DNA polymerase gamma) mutations → Alpers-Huttenlocher syndrome "
            "(progressive neurodegeneration + fulminant hepatic failure). VPA inhibits mitochondrial "
            "beta-oxidation and is directly hepatotoxic in POLG-mutant patients → fatal hepatic "
            "failure within days-weeks of VPA exposure. PLCB1 DEE patients often require prolonged "
            "ASM — POLG sequencing is MANDATORY before any VPA prescription."
        ),
    },
    {
        "term": "Why PHT/CBZ/OXC are contraindicated in PLCB1 Ohtahara",
        "definition": (
            "The suppression phase of burst-suppression is a state of cortical Na-channel silence: "
            "neurons are depleted of Na⁺ driving force after a burst. Adding Na-channel blockers "
            "(PHT/CBZ/OXC) deepens and prolongs the suppression phase without shortening bursts, "
            "producing paradoxical worsening. In KCNQ2 neonatal epilepsy, CBZ is beneficial "
            "(Kv channel-gated Na burst); in PLCB1 Ohtahara, the Na-channel is not the primary "
            "driver — the IP3/Ca²⁺ failure is — so blocking it is harmful."
        ),
    },
    {
        "term": "Somatic Mosaicism in PLCB1",
        "definition": (
            "Post-zygotic (somatic) PLCB1 LOF variant affecting a subset of cortical neurons "
            "rather than all cells. Leads to focal cortical dysplasia type IIb in the affected region. "
            "Blood VAF typically <5% (somatic); cortical tissue VAF ≥15% (confirms mosaic origin). "
            "Surgery can remove the dysplastic region — the rest of the brain retains normal PLCB1. "
            "Key DDx from constitutional biallelic: PLCB1 sequencing on blood may miss somatic mosaic "
            "→ deep sequencing of resected tissue required for diagnosis."
        ),
    },
    {
        "term": "DEE12 / EIEE12 — Disease Classification",
        "definition": (
            "Developmental and Epileptic Encephalopathy 12 (DEE12), OMIM 614563. Previously called "
            "Early Infantile Epileptic Encephalopathy 12 (EIEE12). Caused by biallelic LOF (AR), "
            "de novo dominant, or somatic mosaic PLCB1 mutations. Characterised by "
            "neonatal Ohtahara syndrome (biallelic null) or infantile West syndrome (hypomorphic/de novo). "
            "ILAE 2022 genetic epilepsy classification: PLCB1 listed as definitive DEE gene."
        ),
    },
    {
        "term": "STXBP1 / ARX / SLC25A22 — Neonatal Ohtahara DDx",
        "definition": (
            "STXBP1 (AD de novo): vesicle fusion; de novo burst-suppression; Ohtahara phenotype; "
            "no IP3 involvement; STXBP1 immunoblot reduced. "
            "ARX (X-linked): lissencephaly/abnormal BG on MRI; males only; Partington syndrome in carriers. "
            "SLC25A22 (AR): mitochondrial glutamate transporter; elevated plasma glutamate; "
            "bilateral BG signal change on MRI (different from PLCB1 diffuse atrophy). "
            "KCNQ2 (AD de novo): tonic asymmetric neonatal seizures; NOT burst-suppression; "
            "CBZ HELPFUL (unlike PLCB1 where CBZ is contraindicated)."
        ),
    },
    {
        "term": "IP3 Receptor (IP3R) and ER Ca²⁺ Release",
        "definition": (
            "IP3 generated by PLCβ1 binds IP3R (inositol trisphosphate receptor) on ER membrane "
            "→ Ca²⁺ release from ER → rapid [Ca²⁺]i rise → "
            "CaMKII autophosphorylation → AMPAR Ser831 phosphorylation (LTP); "
            "calcineurin activation → GABA_A Rβ3 Ser408/409 dephosphorylation → GABA_A R internalisation "
            "(bidirectional plasticity). PLCβ1 LOF → no IP3 → no ER Ca²⁺ spike → CaMKII/calcineurin "
            "plasticity both fail → fixed excitation-inhibition imbalance → epilepsy."
        ),
    },
    {
        "term": "PKC — Protein Kinase C and PLCB1 Pathway",
        "definition": (
            "DAG (the second product of PLCβ1 cleavage of PIP2) recruits PKCα/βII/γ to the inner "
            "plasma membrane. PKC phosphorylates: NMDAR NR2B Ser1303 (reduces receptor open-time, "
            "anti-excitotoxic), GABA_A Rγ2 Ser327 (modulates BZD sensitivity). PLCβ1 LOF → DAG absent "
            "→ PKC not recruited → NR2B stays unphosphorylated (stays overactive) + GABA_A Rγ2 Ser327 "
            "unphosphorylated → altered BZD sensitivity (may require adjusted BZD dosing in PLCβ1 DEE)."
        ),
    },
    {
        "term": "Ketogenic Diet — Rationale in PLCB1 DEE",
        "definition": (
            "Ketone bodies (β-hydroxybutyrate, acetoacetate) have multiple anticonvulsant mechanisms: "
            "(1) Inhibit vesicular glutamate transporter (VGLUT) → reduces excitatory drive; "
            "(2) Modulate HCN (hyperpolarisation-activated) channels → alters intrinsic neuronal burst-firing; "
            "(3) Metabolic shift reduces available glucose for glycolysis-dependent Na⁺/K⁺-ATPase during bursts. "
            "In PLCB1 hypomorphic, where IP3-dependent LTD is partially intact, KD reduces IP3-independent "
            "excitatory drive and may be synergistic with residual PLCβ1 activity."
        ),
    },
]


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _pct(pts, key):
    n = len(pts)
    if n == 0:
        return 0
    return round(100 * sum(1 for p in pts if p.get(key)) / n)


# ── API FUNCTIONS ─────────────────────────────────────────────────────────────
def get_overview():
    pts = PATIENTS
    n = len(pts)
    etiol_dist = []
    for ec in ETIOLOGY_CATALOG:
        cat_pts = [p for p in pts if p["category"] == ec["category"]]
        etiol_dist.append({
            "etiology": ec["category"].replace("PLCB1-", "").replace("-", " "),
            "n": len(cat_pts),
            "pct": round(100 * len(cat_pts) / n),
        })
    treat_summary = [
        {"drug": t["drug"].split(" —")[0], "level": t["level"]}
        for t in TREATMENTS
    ]
    monitoring_summary = [
        {
            "timepoint": m["timepoint"],
            "action": m["action"][:85] + "…" if len(m["action"]) > 85 else m["action"],
        }
        for m in MONITORING[:5]
    ]
    return {
        "gene": "PLCB1",
        "chromosome": "20p12.3",
        "omim_gene": "607120",
        "omim_disease": "614563",
        "protein": "Phospholipase C beta-1 (PLCβ1)",
        "aa_length": 1216,
        "domains": (
            "PH-domain (1-130, membrane targeting) + EF-hand (131-300, Ca²⁺ sensing) + "
            "TIM-barrel catalytic (301-790, PIP2→IP3+DAG) + C2-domain (791-960) + "
            "C-terminal coiled-coil (961-1216, Gαq-binding + PDZ)"
        ),
        "inheritance": "AR biallelic LOF (null / hypomorphic) + de novo dominant (C-terminal) + somatic mosaic (FCD IIb)",
        "disease_spectrum": "DEE12 / EIEE12 — Ohtahara (null) → West (hypomorphic/de-novo) → focal epilepsy (somatic mosaic)",
        "unique_feature": (
            "PLCβ1 is the primary Gαq/11 effector — generates IP3 + DAG from PIP2. "
            "LOF abolishes mGluR-LTD (cortical excitatory brake) and PKC-NMDAR gating. "
            "Na-channel blockers CONTRAINDICATED (worsen Ohtahara burst-suppression). "
            "Somatic mosaic = only surgically curable subtype (FCD IIb resection)."
        ),
        "cohort_seed": 505,
        "kpis": {
            "n_patients": n,
            "ohtahara_pct": _pct(pts, "ohtahara_syndrome"),
            "west_syndrome_pct": _pct(pts, "west_syndrome"),
            "burst_suppression_pct": _pct(pts, "burst_suppression"),
            "hypsarrhythmia_pct": _pct(pts, "hypsarrhythmia"),
            "eeg_abnormal_pct": _pct(pts, "eeg_abnormal"),
            "profound_id_pct": _pct(pts, "profound_id"),
            "any_id_pct": _pct(pts, "any_id"),
            "acth_vgb_pct": _pct(pts, "acth_vgb_given"),
            "kd_tried_pct": _pct(pts, "kd_tried"),
            "mri_done_pct": _pct(pts, "mri_done"),
            "cortical_atrophy_pct": _pct(pts, "cortical_atrophy"),
            "fcd_on_mri_pct": _pct(pts, "fcd_on_mri"),
            "surgery_done_pct": _pct(pts, "surgery_done"),
            "seizure_free_post_surg_pct": _pct(pts, "seizure_free_post_surg"),
            "yr1_mortality_pct": _pct(pts, "yr1_mortality"),
            "polg_tested_pct": _pct(pts, "polg_tested"),
        },
        "etiology_distribution": etiol_dist,
        "treatments_summary": treat_summary,
        "monitoring_summary": monitoring_summary,
        "lifecycle": LIFECYCLE,
        "thresholds": THRESHOLDS[:5],
        "contraindications_summary": [c["drug"] for c in CONTRAINDICATIONS],
    }


def get_breakdown():
    pts = PATIENTS
    by_cat = {}
    for p in pts:
        c = p["category"].replace("PLCB1-", "").replace("-", " ")
        if c not in by_cat:
            by_cat[c] = []
        by_cat[c].append(p)

    breakdown = []
    for cat, cat_pts in by_cat.items():
        n = len(cat_pts)
        breakdown.append({
            "category": cat,
            "n": n,
            "ohtahara_pct": _pct(cat_pts, "ohtahara_syndrome"),
            "west_pct": _pct(cat_pts, "west_syndrome"),
            "burst_suppression_pct": _pct(cat_pts, "burst_suppression"),
            "hypsarrhythmia_pct": _pct(cat_pts, "hypsarrhythmia"),
            "profound_id_pct": _pct(cat_pts, "profound_id"),
            "any_id_pct": _pct(cat_pts, "any_id"),
            "acth_vgb_pct": _pct(cat_pts, "acth_vgb_given"),
            "kd_pct": _pct(cat_pts, "kd_tried"),
            "surgery_pct": _pct(cat_pts, "surgery_done"),
            "yr1_mortality_pct": _pct(cat_pts, "yr1_mortality"),
            "fcd_pct": _pct(cat_pts, "fcd_on_mri"),
        })

    etiol_details = [
        {
            "category": ec["category"].replace("PLCB1-", "").replace("-", " "),
            "typical_variant": ec["typical_variant"],
            "inheritance": ec["inheritance"],
            "functional_deficit": ec["functional_deficit"],
            "description": ec["description"],
        }
        for ec in ETIOLOGY_CATALOG
    ]

    summary = {
        "ohtahara_pct": _pct(pts, "ohtahara_syndrome"),
        "west_pct": _pct(pts, "west_syndrome"),
        "burst_suppression_pct": _pct(pts, "burst_suppression"),
        "hypsarrhythmia_pct": _pct(pts, "hypsarrhythmia"),
        "profound_id_pct": _pct(pts, "profound_id"),
        "acth_vgb_pct": _pct(pts, "acth_vgb_given"),
        "kd_pct": _pct(pts, "kd_tried"),
        "surgery_done_pct": _pct(pts, "surgery_done"),
        "yr1_mortality_pct": _pct(pts, "yr1_mortality"),
        "fcd_on_mri_pct": _pct(pts, "fcd_on_mri"),
        "polg_tested_pct": _pct(pts, "polg_tested"),
    }

    return {
        "gene": "PLCB1",
        "chromosome": "20p12.3",
        "cohort_size": len(pts),
        "cohort_seed": 505,
        "summary": summary,
        "by_category": breakdown,
        "etiology_details": etiol_details,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "thresholds": THRESHOLDS,
    }


def get_definitions():
    return {
        "gene": "PLCB1",
        "chromosome": "20p12.3",
        "protein": "Phospholipase C beta-1 (PLCβ1)",
        "omim_gene": "607120",
        "omim_disease": "614563",
        "disease_name": "DEE12 / EIEE12 — Developmental and Epileptic Encephalopathy 12",
        "inheritance": "AR biallelic + de novo dominant + somatic mosaic",
        "definitions": DEFINITIONS,
        "key_ddx": [
            "STXBP1 (9q34.11): vesicle fusion DEE; AD de novo; burst-suppression Ohtahara; "
            "no IP3 pathway; STXBP1 immunoblot reduced; most common neonatal DEE gene",
            "ARX (Xp22.13): X-linked; lissencephaly/abnormal basal ganglia MRI in males; "
            "Partington syndrome in carrier females; not an IP3-pathway disease",
            "KCNQ2 (20q13.33): AD de novo; tonic asymmetric neonatal seizures; NOT burst-suppression; "
            "CBZ/PHT HELPFUL (KCNQ2 is a Kv channel — Na-channel blockade is rational; opposite of PLCB1)",
            "SLC25A22 (11p15.5): AR; mitochondrial glutamate carrier; Ohtahara; elevated plasma glutamate; "
            "bilateral BG MRI signal change (distinct from PLCB1 diffuse atrophy)",
            "NKH (GLDC/AMT/GCSH): metabolic mimic; elevated plasma glycine + CSF:plasma ratio >0.08; "
            "plasma amino acids + CSF glycine mandatory in neonatal DEE workup",
            "Biotinidase deficiency (BTD): biotin-responsive; plasma biotinidase assay mandatory",
        ],
        "mandatory_workup": [
            "PLCB1 sequencing + deletion/duplication analysis (simultaneous with broad neonatal panel)",
            "Broad neonatal DEE panel: STXBP1 / ARX / KCNQ2 / SLC25A22 / CDKL5 / ARX (simultaneous)",
            "EEG STAT: burst-suppression vs hypsarrhythmia vs focal pattern determines subtype",
            "MRI brain (3T if available): cortical malformation, FCD, BG signal change, atrophy",
            "Metabolic screen: plasma amino acids, organic acids, lactate, ammonia, CSF glycine",
            "Biotinidase activity (cheap, treatable mimic)",
            "POLG sequencing MANDATORY before VPA prescription",
            "FDG-PET + ictal SEEG if somatic mosaic / FCD IIb suspected on MRI",
            "Somatic VAF analysis on resected cortical tissue if surgery planned",
            "Cascade testing: siblings (AR recurrence 25%); parents (AR carriers in biallelic)",
        ],
        "standards": [
            "OMIM 614563 (DEE12 / EIEE12)",
            "ILAE Genetic Epilepsies 2022 classification (PLCB1 definitive DEE gene)",
            "UKISS protocol (ACTH + VGB) for infantile spasms",
            "VGB REMS programme (max 16 weeks IS use; ophthalmology monitoring)",
            "POLG Working Group guidelines (pre-VPA POLG screening)",
            "Kurian et al. (2010) Am J Hum Genet 86:346-354 (original PLCB1 EIEE12 paper)",
            "ILAE FCD classification Blümcke 2011 + Najm 2022 (FCD IIb surgical criteria)",
        ],
        "five_key_facts": [
            "PLCB1 LOF abolishes mGluR-LTD — the Gq/IP3/Ca²⁺ brake on cortical excitation — causing neonatal/infantile epileptic encephalopathy",
            "PHT/CBZ/OXC are ABSOLUTE CONTRAINDICATIONS in Ohtahara (worsen burst-suppression by deepening the silent phase)",
            "Somatic mosaic PLCB1 → FCD IIb → surgical cure possible (60% Engel Class I); unique among PLCB1 subtypes",
            "KCNQ2 phenotypically overlaps (neonatal DEE) but CBZ is HELPFUL in KCNQ2 vs HARMFUL in PLCB1 — molecular distinction critical",
            "POLG screen mandatory before VPA; KD is the cornerstone adjunct across hypomorphic and de novo subtypes",
        ],
    }
