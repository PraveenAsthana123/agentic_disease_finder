#!/usr/bin/env python3
"""MCOLN1 / Mucolipidosis IV (ML-IV) Epilepsy Dashboard — seed data module.

Mucolipidosis type IV (ML-IV): caused by biallelic LOF in MCOLN1 gene (19p13.2), encoding
mucolipin-1 (TRPML1), a lysosomal Ca2+/Na+ channel of the TRP superfamily. Deficiency impairs
lysosomal membrane fusion, lysosomal exocytosis, and Ca2+-triggered lysosomal biogenesis →
accumulation of lipids AND water-soluble substances in lysosomes (mixed substrate storage).

KEY DISTINGUISHING FEATURES:
  1. CORNEAL CLOUDING — PRESENT (bilateral, from infancy) — directly OPPOSITE to GNPTAB (ML-II/III
     where corneal clouding is ABSENT); corneal clouding in ML-IV is one of the first clinical signs;
     progressive; may reduce visual acuity over time (combined with retinal degeneration);
     slit-lamp exam mandatory in any LSD workup with psychomotor retardation.
  2. ELEVATED SERUM GASTRIN + ACHLORHYDRIA — PATHOGNOMONIC for ML-IV (100% of patients);
     MCOLN1 required for V-ATPase-mediated H+ secretion via lysosomal membrane recycling in
     gastric parietal cells → achlorhydria (no gastric acid) → gastrin hyperproduction by negative
     feedback failure; serum gastrin markedly elevated (>10x normal); easily measured blood test
     that is SPECIFIC to ML-IV among all LSDs; plasma enzyme panel NORMAL (not elevated — unlike ML-II).
  3. NORMAL PLASMA LYSOSOMAL ENZYMES — unlike GNPTAB (ML-II/III, 10-50x plasma elevation);
     MCOLN1 deficiency does NOT affect M6P targeting (phosphotransferase machinery intact);
     lysosomal enzymes trafficked normally to lysosomes → NORMAL plasma enzyme levels;
     this NORMAL plasma panel in a patient with corneal clouding + psychomotor retardation →
     narrows differential to ML-IV (plasma enzyme elevation would suggest ML-II or other LSD).
  4. ASHKENAZI JEWISH FOUNDER MUTATIONS — IVS3-2A>G (c.406-2A>G) ~67-73% Ashkenazi alleles;
     p.Arg403Cys ~22-25%; carrier frequency ~1:100 Ashkenazi Jews; disease prevalence ~1:40,000;
     highest prevalence in Israel (Ashkenazi Jewish population); non-Ashkenazi variants private,
     scattered globally with >50 different pathogenic variants reported; Sephardic founder alleles
     rare compared to Ashkenazi; carrier screening available in Ashkenazi Jewish carrier programs.
  5. RETINAL DEGENERATION — ERG (electroretinogram) abnormal in 100% of ML-IV patients;
     progressive rod+cone dystrophy; often precedes symptomatic visual loss by years;
     ERG is the single most sensitive diagnostic test beyond MCOLN1 sequencing;
     CRITICAL for VGB decision: VGB causes peripheral visual field loss → RELATIVE CI in ML-IV
     (already compromised retina; monitoring impossible in severe ID); baseline ERG mandatory.
  6. NO BONE DISEASE / NO SKELETAL CHANGES — unlike MPS diseases (MPS-I, MPS-VI, MPS-VII, MPS-IVA)
     which have dysostosis multiplex, atlantoaxial instability, joint stiffness, carpal tunnel;
     ML-IV has NO bone disease, NO GAG accumulation (not a proteoglycan disorder);
     urine GAG screen NORMAL (critical differential point: GAG-negative LSD with corneal clouding);
     no atlantoaxial instability (no anesthesia concern for AAI positioning).
  7. IRON DEFICIENCY ANEMIA — secondary to achlorhydria (impaired gastric acid → impaired Fe3+→Fe2+
     conversion → reduced duodenal iron absorption); responds to supplemental iron (oral or IV);
     anemia may worsen neurological symptoms if untreated; serum gastrin elevated simultaneously
     provides dual diagnostic confirmation (gastrin + anemia in psychomotor retardation child = ML-IV).
  8. PSYCHOMOTOR RETARDATION — ALL patients have intellectual disability and motor delay from infancy;
     Ashkenazi severe type I: profound ID, non-ambulatory; late-onset type II: milder ID, longer
     survival; NO neurological regression in early childhood (static developmental delay, not
     progressive like Tay-Sachs or CLN); slow progression from adult years onward.
  9. EPILEPSY 15-25% — LOWER prevalence than most LSDs; predominantly focal onset or GTCS;
     NO infantile spasms pattern (unlike ML-II GNPTAB where IS is dominant); DRE 5-15%;
     myoclonic seizures rare (unlike Tay-Sachs HEXA or CLNs); seizure burden relatively low
     compared to storage/metabolic diseases with cortical involvement.
  10. NO ERT APPROVED (2026) — unlike MPS diseases with multiple ERTs; MCOLN1 is a membrane
      channel protein (not a soluble enzyme) → cannot be delivered via IV ERT mechanism;
      gene therapy strategies (AAV9-MCOLN1 intrathecal/IV) in preclinical and early Phase I;
      mucolipin-specific Ca2+ channel agonists (ML-SA1, MK6-83) under research as pharmacochaperones.

Enzyme biology:
  MCOLN1 encodes mucolipin-1 (TRPML1), a voltage-gated Ca2+/Na+ channel predominantly located
  on the lysosomal membrane. TRPML1 mediates Ca2+ release from lysosomes, which is required for:
  1) lysosomal membrane fusion with autophagosomes and late endosomes (impaired → accumulation);
  2) lysosomal exocytosis (TRPML1-Ca2+ triggers plasma membrane fusion of lysosomes);
  3) TFEB nuclear translocation (lysosomal biogenesis transcription factor activation);
  4) mTORC1 sensing at the lysosomal surface (TRPML1 modulates mTOR complex 1 activity).
  Deficiency causes: progressive lysosomal storage of lipids (gangliosides, phospholipids) AND
  water-soluble substances (mucopolysaccharide-like materials) → "mucolipidosis" (mixed storage).
  Storage detected by: EM (lamellar bodies + amorphous inclusions in lysosomes); fibroblast storage
  on EM is diagnostically distinctive. Unlike ML-II/III (M6P targeting defect), ML-IV substrate
  accumulation reflects impaired lysosomal membrane dynamics, not enzyme targeting failure.

Pharmacological safety profile:
  CORNEAL CI: VGB (vigabatrin) RELATIVE CI — bilateral concentric visual field constriction from
    VGB compounds already-progressive retinal degeneration; if VGB used, ERG baseline mandatory;
    informed consent for irreversible visual field loss in already-compromised vision.
  RETINAL MONITORING: Any AED requiring periodic visual field testing (VGB, vigabatrin) carries
    INCREASED RISK in ML-IV given combined retinal degeneration + intellectual disability making
    formal perimetry impossible or unreliable.
  POLG1: MANDATORY before VPA (standard for all LSDs — CPIC Grade A);
  CARDIAC: No cardiac involvement in ML-IV → PHT not contraindicated (contrast with GNPTAB ML-II).
  ANESTHESIA: Moderate risk — no major airway concern (no gingival hyperplasia, no AAI) but
    corneal clouding may affect eye protection during anesthesia; standard precautions.
"""

import random

GENE = "MCOLN1"
LOCUS = "19p13.2"
OMIM_GENE = "605248"
OMIM_DISEASE = "252650"
INHERITANCE = "AR (Autosomal Recessive, Biallelic LOF)"
COHORT_SIZE = 40
PROTEIN = "Mucolipin-1 (TRPML1 — Lysosomal Ca2+/Na+ TRP Channel)"
DISEASE = "Mucolipidosis type IV (ML-IV)"

DISEASE_MECHANISM = (
    "MCOLN1 (19p13.2) encodes mucolipin-1 (TRPML1), a lysosomal Ca2+/Na+ channel (TRP superfamily). "
    "Biallelic LOF → impaired lysosomal Ca2+ release → defective lysosomal membrane fusion, "
    "lysosomal exocytosis, and TFEB/mTORC1 lysosomal signaling → mixed lysosomal storage "
    "(lipids + water-soluble substances). Clinically: corneal clouding + psychomotor retardation + "
    "retinal degeneration + achlorhydria/elevated gastrin. NO bone disease, NO GAG accumulation. "
    "Ashkenazi Jewish founder mutations (IVS3-2A>G ~70%; p.Arg403Cys ~24%). No ERT approved 2026. "
    "PATHOGNOMONIC: serum gastrin markedly elevated + corneal clouding + NORMAL plasma lysosomal enzymes."
)

# Five etiology classes (Ashkenazi severe, Ashkenazi late-onset, compound het, homozygous missense, private variant)
ETIOLOGIES = [
    {
        "name": "Ashkenazi Severe (IVS3-2A>G Homozygous or IVS3-2A>G/Arg403Cys) — Type I",
        "pct": 35,
        "n": 14,
        "ml_type": "ML-IV Type I",
        "seizure_risk": "Epilepsy 20-30%; focal onset + GTCS; no IS pattern; ERG always abnormal; corneal clouding from year 1",
        "eeg": "Diffuse background slowing; focal discharges (temporal/occipital); rarely hypsarrhythmia; progressive EEG deterioration",
        "variant_detail": (
            "IVS3-2A>G homozygous (c.406-2A>G / NC_000019.10:g.7597167A>G): founder splice mutation, "
            "~67-73% of Ashkenazi alleles; abolishes canonical 3' splice site of intron 3 → exon 4 skipping "
            "→ frameshift p.Ser136SerfsTer15 → NMD → complete null allele; "
            "compound heterozygote IVS3-2A>G + p.Arg403Cys (second Ashkenazi founder, ~22-25% alleles) → Type I severe; "
            "carrier frequency 1:100 Ashkenazi Jews; disease frequency ~1:40,000 Ashkenazi."
        ),
    },
    {
        "name": "Ashkenazi Late-Onset (p.Arg403Cys Homozygous) — Type II",
        "pct": 20,
        "n": 8,
        "ml_type": "ML-IV Type II",
        "seizure_risk": "Epilepsy 10-20%; milder seizure burden; late onset; longer survival; cognitive better than Type I",
        "eeg": "Mild diffuse slowing; focal temporal discharges; ERG abnormal but slower progression than Type I",
        "variant_detail": (
            "p.Arg403Cys (c.1207C>T) homozygous: second Ashkenazi founder missense mutation; "
            "Arg403 is a critical pore-lining residue in the TRP channel transmembrane domain 5 region; "
            "c.1207C>T → missense in the intramembrane channel-forming region → partial loss of channel function "
            "(residual Ca2+ flux ~5-15%); milder disease trajectory than IVS3-2A>G null; "
            "Type II: slower neurodegeneration, surviving to adulthood; motor function retained longer; "
            "corneal clouding milder, achlorhydria present, gastrin elevated."
        ),
    },
    {
        "name": "Compound Heterozygous — Non-Ashkenazi or Mixed Ancestry",
        "pct": 25,
        "n": 10,
        "ml_type": "ML-IV Type I/Intermediate",
        "seizure_risk": "Epilepsy 15-25%; intermediate severity depends on allele severity combination; variable",
        "eeg": "Variable based on allele severity; focal > generalized; progressive slowing over years",
        "variant_detail": (
            "Compound heterozygous: one Ashkenazi founder (IVS3-2A>G or p.Arg403Cys) + one private variant "
            "(missense, nonsense, splice-site, indel); >50 different non-Ashkenazi private pathogenic variants "
            "reported in MCOLN1 globally; severity depends on allele combination: "
            "null + hypomorphic = intermediate; null + null = severe; hypomorphic + hypomorphic = mild; "
            "MCOLN1 sequencing essential (enzyme panel normal; urine GAG normal; serum gastrin diagnostic "
            "biomarker available before sequencing complete)."
        ),
    },
    {
        "name": "Homozygous Missense — Hypomorphic Variants (Partial Function)",
        "pct": 12,
        "n": 5,
        "ml_type": "ML-IV Type II/Attenuated",
        "seizure_risk": "Epilepsy 5-15%; lowest epilepsy burden; best cognitive outcomes in ML-IV spectrum; ambulatory",
        "eeg": "Mild background changes; focal discharges uncommon; ERG mildly abnormal at baseline",
        "variant_detail": (
            "Homozygous hypomorphic missense: retains 10-30% TRPML1 channel function; "
            "examples: p.Val446Leu (pore-forming domain), p.Ala419Pro (TM6 helix) — partial function retained; "
            "attenuated phenotype: mild to moderate ID, corneal clouding mild (photophobia only initially), "
            "achlorhydria + gastrin still elevated (channel-independent gastric parietal cell function), "
            "ERG mildly abnormal; retinal progression very slow; adults functional with support; "
            "no gait deterioration until late adulthood."
        ),
    },
    {
        "name": "Private / Novel Variants (Including Large Deletions, Complex Rearrangements)",
        "pct": 8,
        "n": 3,
        "ml_type": "ML-IV Severe/Variable",
        "seizure_risk": "Epilepsy 20-35%; severity depends on variant class; genomic deletions → null → severe",
        "eeg": "Variable; large deletion null alleles behave as Type I severe; high-resolution EEG monitoring",
        "variant_detail": (
            "Private variants: whole-gene deletions (del 19p13.2), complex rearrangements, deep intronic "
            "splice variants (cryptic exon activation), VUS requiring functional assay (TRPML1 Ca2+ flux "
            "in patient fibroblasts using GCaMP6 Ca2+ indicator + ML-SA1 mucolipin agonist stimulation); "
            "MCOLN1 full gene sequencing + CNV analysis (array CGH or WGS) required; "
            "functional validation by mucolipin Ca2+ release assay in fibroblasts when VUS identified; "
            "serum gastrin biomarker helpful in atypical presentations regardless of variant class."
        ),
    },
]


def get_overview():
    random.seed(42)
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "inheritance": INHERITANCE,
        "cohort_size": COHORT_SIZE,
        "disease_mechanism": DISEASE_MECHANISM,
        "color_scheme": {
            "ACCENT":  "#1a237e",   # deep indigo — ML-IV corneal clouding, primary
            "ACCENT2": "#880e4f",   # deep magenta — VGB RELATIVE CI, retinal degeneration
            "ACCENT3": "#e65100",   # deep orange — elevated gastrin, achlorhydria, caution
            "ACCENT4": "#4a148c",   # deep purple — no ERT, mucolipin channel biology
            "ACCENT5": "#006064",   # teal — Ashkenazi founder, iron anemia, safe
            "ACCENT6": "#1565c0",   # blue — LEV first-line, safe AEDs, supportive care
        },
        "kpis": {
            "corneal_clouding":         "PRESENT — bilateral from early infancy; slit-lamp mandatory; OPPOSITE of GNPTAB (ML-II absent)",
            "gastrin_pathognomonic":    "Serum gastrin markedly elevated (100%) — PATHOGNOMONIC; achlorhydria; iron anemia",
            "plasma_enzymes":           "NORMAL plasma lysosomal enzyme panel — distinguishes ML-IV from ML-II (10-50x elevated)",
            "retinal_degeneration":     "ERG abnormal in 100% — rod+cone dystrophy; progressive visual field loss; VGB RELATIVE CI",
            "no_bone_disease":          "NO bone disease, NO GAG, NO dysostosis multiplex, NO atlantoaxial instability",
            "ashkenazi_founder":        "IVS3-2A>G ~70% + p.Arg403Cys ~24% Ashkenazi alleles; carrier 1:100 AJ; disease 1:40,000 AJ",
            "iron_deficiency_anemia":   "Secondary to achlorhydria (impaired Fe absorption); responds to iron supplementation",
            "epilepsy_prevalence":      "15-25% overall — LOW for LSD; focal + GTCS; NO infantile spasms (vs ML-II IS 40-55%)",
            "drug_resistance":          "DRE 5-15% — lower than most LSDs",
            "no_ert_approved":          "No ERT approved (2026) — channel protein, not soluble enzyme; gene therapy Phase I research",
            "no_hsct_evidence":         "No HSCT evidence (unlike MPS-I Hurler Level A HSCT before age 2.5yr)",
            "polg1_mandatory":          "MANDATORY before VPA (CPIC Grade A)",
            "vgb_relative_ci":          "VGB RELATIVE CI — retinal degeneration + visual monitoring impossible in severe ID",
            "pht_safety":               "PHT/Fosphenytoin NOT contraindicated (no cardiac CI unlike GNPTAB ML-II)",
            "anesthesia_moderate_risk": "Moderate anesthesia risk — no gingival hyperplasia, no AAI; corneal eye protection needed",
            "psychomotor_retardation":  "ALL patients — severe Type I (IVS3-2A>G), milder Type II (p.Arg403Cys homozygous)",
            "urine_gag_normal":         "Urine GAG NORMAL — critical: NOT a proteoglycan disorder; GAG screen false-negative",
            "storage_mixed":            "Mixed storage: lipids + water-soluble (not pure lipid like Tay-Sachs or pure GAG like MPS)",
            "mcoln1_trpml1_channel":    "TRPML1 = lysosomal Ca2+ channel; modulates TFEB, mTORC1, lysosomal exocytosis",
            "gene_therapy_research":    "AAV9-MCOLN1 (intrathecal/IV) preclinical + early Phase I; ML-SA1/MK6-83 Ca2+ agonists research",
        },
        "etiologies": ETIOLOGIES,
    }


def get_breakdown():
    random.seed(42)
    patients = []

    # Seizure type pools by ML-IV type
    type1_seizure_types = [
        "Focal Onset Aware",
        "Focal to Bilateral GTCS",
        "GTCS",
        "Focal + GTCS",
        "GTCS + Focal",
    ]
    type2_seizure_types = [
        "GTCS",
        "Focal Onset Aware",
        "GTCS",
        "Focal Onset Impaired",
    ]
    intermediate_seizure_types = [
        "Focal Onset Aware",
        "GTCS",
        "Focal to Bilateral GTCS",
        "GTCS",
    ]

    treatment_map = {
        "ML-IV Type I":              ["Levetiracetam", "Lamotrigine", "Valproate", "Lacosamide"],
        "ML-IV Type II":             ["Levetiracetam", "Lamotrigine", "Oxcarbazepine"],
        "ML-IV Type I/Intermediate": ["Levetiracetam", "Valproate", "Lamotrigine", "Lacosamide"],
        "ML-IV Type II/Attenuated":  ["Levetiracetam", "Lamotrigine"],
        "ML-IV Severe/Variable":     ["Levetiracetam", "Valproate", "Lacosamide"],
    }

    for eth in ETIOLOGIES:
        n = eth["n"]
        ml_type = eth["ml_type"]
        is_type1 = ml_type in ("ML-IV Type I", "ML-IV Severe/Variable")
        is_type2 = ml_type == "ML-IV Type II"
        is_attenuated = ml_type == "ML-IV Type II/Attenuated"

        for i in range(n):
            # Age of onset
            if is_type1:
                age_onset = round(random.uniform(0.5, 3.0), 1)
            elif is_type2:
                age_onset = round(random.uniform(1.5, 6.0), 1)
            elif is_attenuated:
                age_onset = round(random.uniform(3.0, 12.0), 1)
            else:
                age_onset = round(random.uniform(0.8, 5.0), 1)

            # Seizure type
            if is_type1:
                seizure_type = random.choice(type1_seizure_types)
            elif is_type2:
                seizure_type = random.choice(type2_seizure_types)
            else:
                seizure_type = random.choice(intermediate_seizure_types)

            # Clinical features
            has_corneal_clouding = True  # 100% of ML-IV
            has_retinal_degeneration = True  # ERG abnormal 100%
            has_iron_anemia = random.random() < 0.80  # 80% iron deficiency
            has_achlorhydria = True  # 100%
            gastrin_level = f"{random.randint(8, 30)}x ULN elevated" if random.random() < 0.90 else "5-8x ULN"

            # Treatment
            tx_pool = treatment_map.get(ml_type, ["Levetiracetam"])
            treatment_1 = tx_pool[0]
            treatment_2 = tx_pool[1] if len(tx_pool) > 1 else "None"

            # CI avoided
            ci_pool = ["VGB (retinal CI)", "Typical APs (RELATIVE)", "VGB + Perampanel"]
            ci_avoided = random.choice(ci_pool)

            patients.append({
                "id":                      f"ML4-{i+1:03d}",
                "etiology":                eth["name"][:50],
                "ml_type":                 ml_type,
                "age_onset_yr":            age_onset,
                "seizure_type":            seizure_type,
                "corneal_clouding":        has_corneal_clouding,
                "retinal_degeneration":    has_retinal_degeneration,
                "iron_anemia":             has_iron_anemia,
                "achlorhydria":            has_achlorhydria,
                "gastrin_level":           gastrin_level,
                "treatment_1":             treatment_1,
                "treatment_2":             treatment_2,
                "ci_avoided":              ci_avoided,
            })

    # Summary tables
    seizure_summary = [
        {"type": "Focal Onset Aware",           "pct": 40, "note": "Most common in ML-IV; temporal and occipital foci; reflects lysosomal storage in cortex"},
        {"type": "Focal to Bilateral GTCS",      "pct": 35, "note": "Secondary generalization from focal onset; LEV first-line (no PHT CI in ML-IV)"},
        {"type": "GTCS (Primarily Generalized)", "pct": 20, "note": "Generalized tonic-clonic; less common than focal; lamotrigine or LEV backbone"},
        {"type": "Myoclonic",                    "pct": 8,  "note": "Rare in ML-IV — unlike Tay-Sachs (HEXA) or CLN diseases; myoclonus suggests alternative diagnosis"},
        {"type": "Status Epilepticus",           "pct": 5,  "note": "Rare; no specific IV AED CI (unlike GNPTAB ML-II where fosphenytoin ABSOLUTE CI)"},
    ]

    trigger_summary = [
        {"trigger": "Febrile illness / infection",     "pct": 55, "note": "Most common trigger; fever threshold lowered; prophylactic clobazam for febrile episodes"},
        {"trigger": "Sleep deprivation",               "pct": 40, "note": "Common trigger; structured sleep schedule critical; melatonin for sleep-wake cycle support"},
        {"trigger": "Missed AED doses",                "pct": 35, "note": "Caregiver burden high in ML-IV (severe ID); electronic reminders + blister packs reduce misses"},
        {"trigger": "Emotional or physiological stress","pct": 30, "note": "Hyperventilation during distress; non-epileptic movement episodes may be confused with seizures"},
        {"trigger": "Photic stimulation",              "pct": 18, "note": "Rare; less photosensitive than CLN or Tay-Sachs; ERG abnormality does not correlate with photosensitivity"},
        {"trigger": "Iron deficiency / anemia",        "pct": 15, "note": "Untreated iron deficiency anemia (from achlorhydria) lowers seizure threshold; iron repletion reduces seizure frequency"},
        {"trigger": "Visual sensory overload",         "pct": 12, "note": "Rare ML-IV-specific: retinal degeneration + cortical visual processing dysfunction may trigger occipital seizures"},
    ]

    treatment_summary = [
        {"drug": "Levetiracetam (LEV)", "level": "Level B", "indication": "Focal and generalized seizures; first-line backbone in ML-IV; no specific drug interactions; safe cardiac profile; no POLG1 interaction; well tolerated", "ci": None},
        {"drug": "Lamotrigine (LTG)",  "level": "Level B", "indication": "Focal onset seizures; useful in Type II/attenuated; titrate slowly (rash risk); reduces focal discharge burden; good long-term tolerability in adults", "ci": None},
        {"drug": "Valproate (VPA)",    "level": "Level B — POLG1 mandatory", "indication": "Broad-spectrum; GTCS + focal; POLG1/POLG2 sequencing mandatory before prescribing; hepatic monitoring; carnitine supplementation considered", "ci": "POLG1 carrier → Alpers risk → ABSOLUTE CI if POLG1+"},
        {"drug": "Lacosamide",         "level": "Level C", "indication": "Adjunctive for focal seizures; sodium channel slow-inactivation; no cardiac CI in ML-IV (unlike GNPTAB ML-II where cardiac monitoring needed); useful when LEV+LTG insufficient", "ci": None},
        {"drug": "Oxcarbazepine (OXC)","level": "Level C", "indication": "Focal epilepsy; safe in ML-IV (no cardiac CI); CAUTION if cognitive side effects in already-impaired patients; useful in Type II milder phenotype", "ci": None},
        {"drug": "Clobazam (CLB)",     "level": "Level C", "indication": "Adjunctive; febrile seizure prophylaxis; acute cluster management; useful in intellectual disability with compliance issues (once/twice daily dosing)", "ci": None},
        {"drug": "Iron supplementation","level": "Level A — achlorhydria CI prevention", "indication": "MANDATORY for all ML-IV patients: achlorhydria → impaired Fe absorption → iron deficiency anemia; oral iron (ferrous sulfate) or IV iron if oral inadequate; prevents anemia-driven seizure threshold reduction", "ci": None},
        {"drug": "Melatonin",          "level": "Level B (sleep)", "indication": "Sleep-wake cycle dysfunction in ML-IV; structured sleep routine; melatonin 0.5-3mg at bedtime; reduces febrile trigger risk (adequate sleep maintains immune threshold)", "ci": None},
        {"drug": "Lubricating eye drops","level": "Level A (symptomatic)", "indication": "Corneal clouding management; prevents corneal desiccation; preservative-free artificial tears; ophthalmology review every 6 months; progressive corneal clouding may require contact lens fitting", "ci": None},
    ]

    contraindication_summary = [
        {"drug": "Vigabatrin (VGB)",          "severity": "RELATIVE CI",  "reason": "VGB causes permanent bilateral concentric visual field constriction in 30-40% of patients; ML-IV has progressive retinal degeneration (ERG abnormal 100%) → VGB accelerates functional visual loss AND visual field monitoring via standard perimetry IMPOSSIBLE in severe intellectual disability; if VGB attempted despite CI, ERG baseline mandatory (ERG more reliable than perimetry in ID); informed consent for irreversible vision loss"},
        {"drug": "Typical Antipsychotics",    "severity": "HIGH RISK",    "reason": "Behavioral problems in ML-IV (from ID + frustration) may prompt AP prescriptions; haloperidol/fluphenazine → EPS in lysosomal storage disease (GABAergic/dopaminergic system involvement); use atypical APs (quetiapine, risperidone low dose) if behavioral management required; document specific behavioral target before prescribing"},
        {"drug": "Valproate (POLG1+)",        "severity": "ABSOLUTE CI if POLG1 carrier", "reason": "POLG1/POLG2 sequencing MANDATORY before VPA in ALL ML-IV patients (CPIC Grade A evidence); POLG1 pathogenic variants → VPA → Alpers-Huttenlocher syndrome (acute liver failure + irreversible neurological deterioration); never start VPA before POLG1 result available; LEV is safe alternative backbone"},
        {"drug": "Perampanel (PER)",          "severity": "CAUTION",      "reason": "AMPA receptor antagonist; behavioral side effects (aggression, irritability) may be unacceptable in ML-IV patients with pre-existing ID + behavioral challenges; if used, start at 2mg/day and titrate very slowly; caregiver-reported behavioral monitoring essential; alternative to PER: clobazam for adjunctive focal coverage"},
        {"drug": "Topiramate high-dose",      "severity": "CAUTION",      "reason": "Cognitive side effects (word-finding difficulty, memory impairment) compound already-severe intellectual disability; low-dose topiramate (<100mg/day) may be used with close monitoring if other options exhausted; cognitive assessment at every dose change"},
        {"drug": "Phenobarbital chronic",     "severity": "CAUTION",      "reason": "Sedation + cognitive blunting in severe ID; long-term cognitive impairment; use only as last resort or for acute SE; prefer LEV/LTG/lacosamide for chronic management"},
    ]

    return {
        "patients":                  patients,
        "etiology_breakdown":        [{"name": e["name"], "pct": e["pct"], "n": e["n"]} for e in ETIOLOGIES],
        "etiologies":                ETIOLOGIES,
        "seizure_summary":           seizure_summary,
        "trigger_summary":           trigger_summary,
        "treatment_summary":         treatment_summary,
        "contraindication_summary":  contraindication_summary,
    }


def get_definitions():
    random.seed(42)
    return {
        "glossary": [
            {
                "term": "MCOLN1 / Mucolipin-1 (TRPML1)",
                "definition": (
                    "MCOLN1 (19p13.2) encodes mucolipin-1 (TRPML1), a ~580-amino-acid lysosomal cation channel "
                    "of the transient receptor potential (TRP) superfamily; bimodal topology with two transmembrane "
                    "domains flanking an intraluminal loop; calcium/sodium channel activated by phosphoinositides "
                    "and channel agonists (ML-SA1, MK6-83); required for lysosomal membrane fusion, Ca2+-triggered "
                    "lysosomal exocytosis, TFEB nuclear translocation (lysosomal biogenesis), and mTORC1 lysosomal "
                    "signaling; TRPML1 belongs to the TRPML subfamily (also TRPML2/MCOLN2 and TRPML3/MCOLN3); "
                    "biallelic LOF → Mucolipidosis type IV."
                ),
            },
            {
                "term": "Mucolipidosis type IV (ML-IV) — Disease Definition",
                "definition": (
                    "ML-IV: autosomal recessive lysosomal storage disorder caused by biallelic LOF in MCOLN1; "
                    "lysosomal accumulation of MIXED substrates (lipids: gangliosides, phospholipids + water-soluble "
                    "materials: mucopolysaccharide-like); EM: lamellar bodies + amorphous material in lysosomes; "
                    "name 'mucolipidosis' reflects mixed lipid + polysaccharide storage; "
                    "phenotype: corneal clouding + severe psychomotor retardation + retinal degeneration + achlorhydria; "
                    "NOT a proteoglycan disorder (urine GAG normal — distinguishes from MPS); "
                    "NOT an M6P targeting defect (plasma lysosomal enzymes NORMAL — distinguishes from ML-II/III)."
                ),
            },
            {
                "term": "Corneal Clouding (ML-IV — PRESENT, Bilateral)",
                "definition": (
                    "Corneal clouding is PRESENT in ALL ML-IV patients (bilateral, progressive); "
                    "begins in the first year of life; results from lysosomal storage in corneal keratocytes + "
                    "endothelial dysfunction; photophobia + mild visual impairment early; advanced clouding reduces "
                    "visual acuity significantly (combined with retinal degeneration → compounding visual disability); "
                    "CRITICAL DIFFERENTIAL: corneal clouding PRESENT in ML-IV but ABSENT in ML-II/III (GNPTAB); "
                    "also present in: MPS-I Hurler, MPS-VI Maroteaux-Lamy, MPS-VII Sly, LCAT deficiency; "
                    "slit-lamp examination mandatory in all LSD evaluations — corneal clouding pattern guides diagnosis."
                ),
            },
            {
                "term": "Achlorhydria + Elevated Serum Gastrin (PATHOGNOMONIC for ML-IV)",
                "definition": (
                    "MCOLN1 is required for lysosomal membrane recycling in gastric parietal cells that drives "
                    "vacuolar H+-ATPase (V-ATPase) H+ secretion → MCOLN1 LOF → impaired lysosomal exocytosis → "
                    "V-ATPase membrane cycling failure → achlorhydria (absence of gastric acid); "
                    "parietal cell acid secretion failure → gastrin hyperproduction by G cells (negative feedback "
                    "loss) → serum gastrin markedly elevated (>10x ULN in 100% of patients); "
                    "DIAGNOSTIC SIGNIFICANCE: elevated fasting serum gastrin + achlorhydria (pH>4 after maximal "
                    "histamine stimulation) in a child with psychomotor retardation + corneal clouding = ML-IV "
                    "until proven otherwise; PATHOGNOMONIC combination unique to ML-IV among all LSDs; "
                    "achlorhydria → iron malabsorption → iron deficiency anemia (secondary consequence)."
                ),
            },
            {
                "term": "Normal Plasma Lysosomal Enzyme Panel (ML-IV — Critical Distinction from ML-II)",
                "definition": (
                    "In ML-IV, plasma lysosomal enzyme levels are NORMAL (HEX-A, HEX-B, arylsulfatase, "
                    "beta-galactosidase, beta-glucuronidase all within normal range); "
                    "this is because MCOLN1 deficiency does NOT affect the M6P targeting machinery — "
                    "lysosomal enzymes are correctly tagged with M6P and trafficked to lysosomes via normal pathway; "
                    "CRITICAL DISTINCTION from ML-II/III (GNPTAB): plasma enzymes 10-50x elevated in ML-II/III; "
                    "NORMAL plasma panel + corneal clouding + psychomotor retardation + elevated gastrin = ML-IV; "
                    "ELEVATED plasma panel + coarse facies + NO corneal clouding = ML-II/III (GNPTAB); "
                    "this single lab distinction (plasma enzyme panel) differentiates the two major mucolipidoses."
                ),
            },
            {
                "term": "Retinal Degeneration + ERG (100% Abnormal in ML-IV)",
                "definition": (
                    "Progressive rod-cone dystrophy in 100% of ML-IV patients; "
                    "ERG (electroretinogram) abnormal from early life — often first objective diagnostic test; "
                    "scotopic ERG (rod response): reduced b-wave amplitude first; photopic ERG (cone response): "
                    "progressive deterioration over years; combined with corneal clouding → compounding visual loss; "
                    "CLINICAL IMPLICATIONS: vigabatrin (VGB) RELATIVE CI in ML-IV — VGB causes irreversible "
                    "peripheral visual field constriction → compounds already-progressive retinal degeneration; "
                    "formal perimetry impossible in severe ID → rely on ERG for monitoring; "
                    "annual ophthalmology: ERG + slit-lamp + visual acuity assessment (behavioural/preferential looking)."
                ),
            },
            {
                "term": "Iron Deficiency Anemia (Secondary to Achlorhydria in ML-IV)",
                "definition": (
                    "Achlorhydria in ML-IV → impaired gastric acidification → impaired Fe3+→Fe2+ conversion "
                    "(gastric acid required for duodenal iron solubilization and reduction to absorbable Fe2+); "
                    "iron deficiency anemia develops in ~80% of ML-IV patients over time; "
                    "anemia worsens neurological symptoms (cognitive decline, fatigue, seizure threshold reduction); "
                    "treatment: oral ferrous sulfate (200-300mg elemental Fe daily) or IV iron if oral insufficient; "
                    "MANAGEMENT NOTE: iron absorption may remain suboptimal despite supplementation due to ongoing "
                    "achlorhydria; ascorbic acid co-administration enhances Fe2+ availability; "
                    "serum ferritin + TIBC monitoring every 6 months; IV iron (ferric gluconate, iron sucrose) "
                    "bypasses GI absorption limitation in refractory cases."
                ),
            },
            {
                "term": "Ashkenazi Jewish Founder Mutations (IVS3-2A>G + p.Arg403Cys)",
                "definition": (
                    "IVS3-2A>G (c.406-2A>G, splice): ~67-73% of Ashkenazi Jewish MCOLN1 alleles; "
                    "abrogates canonical 3' splice site of intron 3 → exon 4 skipping → frameshift Ser136SerfsTer15 "
                    "→ NMD → complete null; founder mutation tracing to 15th-18th century Ashkenazi Jewish communities; "
                    "p.Arg403Cys (c.1207C>T, missense): ~22-25% of Ashkenazi alleles; pore-lining residue "
                    "in TM5 → partial channel dysfunction (~5-15% residual Ca2+ flux); "
                    "IVS3-2A>G homozygous or IVS3-2A>G/Arg403Cys = Type I severe ML-IV; "
                    "Arg403Cys homozygous = Type II milder late-onset ML-IV; "
                    "Ashkenazi Jewish carrier frequency ~1:100; disease prevalence ~1:40,000 AJ population; "
                    "carrier screening panels available and recommended pre-pregnancy in AJ couples."
                ),
            },
            {
                "term": "No ERT Approved (MCOLN1, 2026) — Membrane Channel vs Soluble Enzyme",
                "definition": (
                    "Enzyme replacement therapy (ERT) is not applicable to ML-IV: "
                    "MCOLN1 encodes a transmembrane channel protein (not a soluble lysosomal hydrolase); "
                    "standard ERT mechanism delivers M6P-tagged recombinant enzymes via IV infusion → "
                    "M6P receptor uptake into lysosomes; this mechanism requires a SOLUBLE enzyme that can be "
                    "M6P-tagged and endocytosed → CANNOT be used for a 7-transmembrane domain channel; "
                    "STRATEGIES under research: AAV9-MCOLN1 (gene therapy — intrathecal/IV, preclinical 2020-2025, "
                    "early Phase I); mucolipin-specific chemical chaperones; ML-SA1/MK6-83 (TRPML1 agonists — "
                    "pharmacological activation of residual channel function in hypomorphic variants); "
                    "no approved disease-modifying therapy as of 2026."
                ),
            },
            {
                "term": "VGB RELATIVE CI (Retinal Degeneration + Visual Field Monitoring Failure)",
                "definition": (
                    "Vigabatrin (VGB): GABA transaminase irreversible inhibitor; causes permanent bilateral "
                    "concentric visual field constriction (nasal > temporal) in 30-40% of patients; "
                    "in ML-IV: VGB is RELATIVE CI because: "
                    "1) ML-IV has pre-existing progressive retinal degeneration (ERG abnormal 100%) → "
                    "VGB adds additional retinal insult to already-failing photoreceptors; "
                    "2) Standard VGB monitoring (formal perimetry every 6 months) requires patient cooperation "
                    "→ IMPOSSIBLE in severe intellectual disability (ML-IV Type I/II); "
                    "ERG substitution monitoring possible but does not capture full visual field scope; "
                    "classification: RELATIVE CI (not absolute) because epilepsy benefit may outweigh risk "
                    "in refractory IS IF ERG baseline documented + informed consent obtained; "
                    "NOTE: ML-IV does not have IS (infantile spasms) pattern — VGB rarely indicated anyway."
                ),
            },
            {
                "term": "POLG1 Mandatory (MCOLN1, CPIC Grade A)",
                "definition": (
                    "POLG1/POLG2 sequencing MANDATORY before valproate in all ML-IV patients; "
                    "POLG1 pathogenic variants → VPA → Alpers-Huttenlocher syndrome (acute hepatic failure + "
                    "neurological deterioration + death); CPIC Grade A evidence: do NOT start VPA before "
                    "POLG1 result confirmed negative; order POLG1 simultaneously with MCOLN1 diagnostic workup; "
                    "if POLG1+ → LEV as backbone AED (no POLG1 interaction); LTG second-line; "
                    "lacosamide for focal adjunct; VPA contraindicated for POLG1 carriers regardless of "
                    "ML-IV disease benefit; this rule applies to ALL LSD patients, not just ML-IV."
                ),
            },
            {
                "term": "AAV9-MCOLN1 Gene Therapy (Research Phase 2026)",
                "definition": (
                    "Adeno-associated virus serotype 9 (AAV9) carrying functional MCOLN1 cDNA: "
                    "AAV9 has broad CNS tropism (neurons + astrocytes + oligodendrocytes via intrathecal/IV route); "
                    "MCOLN1 cDNA under CAG or synapsin promoter → mucolipin-1 re-expression in neurons; "
                    "intrathecal administration bypasses blood-brain barrier more efficiently than IV; "
                    "preclinical efficacy shown in Mcoln1-/- mouse model (2020-2024): reduced lysosomal storage, "
                    "improved motor function, extended survival; "
                    "as of 2026: early Phase I dose-finding in non-Ashkenazi severe ML-IV (type I); "
                    "MCOLN1 agonists (ML-SA1, MK6-83): small-molecule TRPML1 Ca2+ channel activators → "
                    "pharmacological approach for hypomorphic variants retaining partial channel function."
                ),
            },
            {
                "term": "Urine GAG Screen — NORMAL in ML-IV (Critical Negative Finding)",
                "definition": (
                    "ML-IV is NOT a proteoglycan (GAG) storage disorder — MCOLN1 deficiency does not cause "
                    "heparan sulfate, dermatan sulfate, keratan sulfate, or chondroitin sulfate accumulation; "
                    "urine GAG screen (quantitative GAG / spot urine GAG/creatinine + TLC fractionation) "
                    "is NORMAL or only minimally elevated in ML-IV; "
                    "CLINICAL TRAP: if a patient has psychomotor retardation + corneal clouding and the urine "
                    "GAG screen is ordered, NORMAL result does NOT exclude an LSD — it excludes MPS disorders "
                    "but ML-IV must still be considered; order serum gastrin + MCOLN1 sequencing when urine GAG "
                    "is normal in this clinical picture; MPS diseases (elevated GAG): MPS-I, MPS-II, MPS-VI, MPS-VII."
                ),
            },
            {
                "term": "No Bone Disease in ML-IV (vs MPS Diseases — Critical Negative)",
                "definition": (
                    "ML-IV has NO skeletal abnormalities, NO dysostosis multiplex, NO atlantoaxial instability "
                    "(unlike MPS-IVA/GALNS where AAI is life-threatening), NO joint contractures, NO carpal tunnel; "
                    "CRITICAL DIFFERENTIAL: all MPS diseases have varying degrees of skeletal involvement from "
                    "GAG-driven bone destruction + lysosomal overload; ML-IV patients walk, move joints normally "
                    "(until late neurological progression); "
                    "ANESTHESIA IMPLICATION: no AAI → no cervical spine precautions required (unlike MPS-IVA/MPS-I); "
                    "no gingival hyperplasia (unlike GNPTAB ML-II) → no airway concern from oral tissue; "
                    "ML-IV anesthesia risk: moderate (corneal eye protection + standard precautions)."
                ),
            },
            {
                "term": "Differential Diagnosis Summary — ML-IV vs Other Corneal Clouding LSDs",
                "definition": (
                    "Corneal clouding differentials in LSD: "
                    "1) MPS-I Hurler (IDUA): HS+DS urine elevated; alpha-L-iduronidase low; HSCT Level A; "
                    "2) MPS-VI Maroteaux-Lamy (ARSB): DS+C4S urine elevated; ARSB low; normal intelligence; "
                    "3) MPS-VII Sly (GUSB): HS+DS+CS triple elevated — PATHOGNOMONIC; vestronidase alfa ERT; "
                    "4) ML-IV (MCOLN1): urine GAG NORMAL; plasma enzymes NORMAL; gastrin ELEVATED — pathognomonic; "
                    "5) GNPTAB (ML-II): NO corneal clouding (opposite of ML-IV!); plasma enzymes 10-50x elevated; "
                    "6) LCAT deficiency (lecithin-cholesterol acyltransferase): corneal clouding + fish-eye; "
                    "KEY RULE: corneal clouding + normal urine GAG + normal plasma enzymes + elevated gastrin = ML-IV; "
                    "confirm: MCOLN1 sequencing + ERG + fasting serum gastrin + iron studies."
                ),
            },
        ],
        "diagnostic_algorithm": [
            "Step 1: Suspect ML-IV — infant or toddler with bilateral corneal clouding + psychomotor retardation "
            "(developmental delay from early infancy, not regression); NO skeletal abnormalities, NO dysostosis, "
            "NO coarse facies (facies typically normal in ML-IV unlike MPS); photophobia and watering eyes from corneal clouding",
            "Step 2: Ophthalmology URGENT — slit-lamp examination confirms bilateral corneal clouding; "
            "ERG (electroretinogram): abnormal rod + cone responses in 100% of ML-IV even at early stage; "
            "ERG abnormality + corneal clouding in infant with psychomotor retardation = ML-IV workup mandatory",
            "Step 3: Serum gastrin (fasting) — markedly elevated in 100% of ML-IV patients (>10x ULN); "
            "paired with gastric pH testing (achlorhydria confirmed: pH>4 after histamine stimulation); "
            "PATHOGNOMONIC combination: corneal clouding + elevated gastrin + psychomotor retardation",
            "Step 4: Plasma lysosomal enzyme panel — HEX-A, HEX-B, arylsulfatase, beta-galactosidase, "
            "beta-glucuronidase: NORMAL in ML-IV (not elevated); if elevated 10-50x → consider ML-II/III (GNPTAB); "
            "normal panel + corneal clouding + elevated gastrin narrows diagnosis to ML-IV before sequencing",
            "Step 5: Urine GAG screen (quantitative + TLC) — NORMAL in ML-IV; "
            "normal urine GAG excludes MPS diseases (MPS-I, MPS-II, MPS-VI, MPS-VII); "
            "ML-IV is the key diagnosis when urine GAG normal + corneal clouding + psychomotor retardation",
            "Step 6: MCOLN1 gene sequencing (19p13.2) — biallelic pathogenic variants; "
            "Ashkenazi Jewish patients: targeted IVS3-2A>G + p.Arg403Cys panel first (covers ~95% AJ alleles); "
            "non-Ashkenazi: full gene sequencing ± RNA analysis for splice variants; "
            "CNV analysis (array CGH or WGS) for suspected deletions/rearrangements",
            "Step 7: Iron studies — ferritin, TIBC, serum iron, CBC for iron deficiency anemia; "
            "achlorhydria → impaired iron absorption → iron deficiency in ~80% of ML-IV patients; "
            "initiate iron supplementation (ferrous sulfate or IV iron) regardless of sequencing result once "
            "clinical diagnosis confirmed; prevents anemia-driven seizure threshold reduction",
            "Step 8: EEG MANDATORY — baseline EEG before any AED; classify seizure type (focal vs generalized); "
            "background slowing reflects neurological involvement; focal discharges temporal or occipital; "
            "NO hypsarrhythmia (infantile spasms NOT a feature of ML-IV — if IS present, reconsider diagnosis)",
            "Step 9: POLG1/POLG2 sequencing — MANDATORY before any valproate prescription; "
            "order simultaneously with MCOLN1 workup; CPIC Grade A; "
            "if POLG1 positive → LEV + LTG backbone; VPA absolutely contraindicated",
            "Step 10: Brain MRI — white matter changes (periventricular leukoencephalopathy) on T2/FLAIR "
            "common in ML-IV Type I; cerebellar atrophy in advanced disease; "
            "MRI patterns support ML-IV but not pathognomonic; correlate with clinical + biochemical findings",
            "Step 11: Multi-specialist team coordination — Metabolic/LSD specialist (diagnosis + monitoring); "
            "Neurology (seizure management, EEG); Ophthalmology (corneal clouding + ERG); Gastroenterology "
            "(achlorhydria management, iron monitoring); Hematology (anemia workup if refractory); "
            "Genetics (variant interpretation, family counseling, carrier testing)",
            "Step 12: Register in ML-IV natural history registry + consider gene therapy trial eligibility "
            "(AAV9-MCOLN1 early Phase I for severe Type I); Ashkenazi Jewish family members: offer carrier "
            "testing (IVS3-2A>G + p.Arg403Cys targeted panel); patient/caregiver counseling on progressive "
            "retinal degeneration trajectory and visual support planning",
        ],
        "pharmacological_distinctions": [
            "VGB vs LEV for focal seizures in ML-IV: LEV PREFERRED (no retinal CI, no visual monitoring burden); "
            "VGB RELATIVE CI due to ML-IV retinal degeneration (ERG abnormal 100%) — VGB irreversible visual "
            "field constriction compounds already-failing photoreceptors; if focal seizures are VGB-responsive "
            "and all alternatives exhausted, document ERG baseline, obtain informed consent, monitor ERG "
            "(not perimetry — impossible in severe ID); this VGB rule is ML-IV-SPECIFIC vs GNPTAB (ML-II) "
            "where VGB is ABSOLUTE CI for different reason (cardiac toxicity in ML-II cardiomyopathy)",
            "PHT/Fosphenytoin in ML-IV vs ML-II (GNPTAB): PHT is NOT contraindicated in ML-IV "
            "(no cardiac involvement in ML-IV — no cardiomyopathy, no QTc prolongation); contrast with "
            "GNPTAB ML-II where PHT/fosphenytoin ABSOLUTE CI (cardiac); in ML-IV SE: IV fosphenytoin IS an "
            "option (unlike ML-II where IV LEV replaces it); important not to apply GNPTAB ML-II CI rules "
            "to ML-IV — different disease, different pharmacological safety profile",
            "Iron supplementation as seizure threshold modifier: achlorhydria-driven iron deficiency anemia "
            "in ML-IV lowers seizure threshold; treating anemia with iron supplementation can reduce seizure "
            "frequency (Level A recommendation for anemia treatment; Level C for seizure threshold effect); "
            "ascorbic acid 250mg with each iron dose enhances Fe2+ availability in achlorhydric gut; "
            "IV iron (ferric gluconate 125mg IV) bypasses GI limitation in refractory cases",
            "Lamotrigine vs carbamazepine for focal seizures in ML-IV: both effective; LTG preferred in "
            "adults surviving to adulthood (Type II/attenuated) for long-term tolerability; "
            "CBZ/OXC reasonable (no cardiac CI in ML-IV unlike GNPTAB ML-II); titrate slowly in severe ID "
            "(behavioral change assessment challenging); OXC preferred in females of reproductive age "
            "(less teratogenic than CBZ, though fertility reduced in severe ML-IV)",
            "POLG1 rule: applies universally to ALL LSD patients including ML-IV; "
            "do NOT use VPA before POLG1 negative confirmed; if seizure emergency requires rapid initiation, "
            "use LEV IV loading (60mg/kg over 15 minutes) as bridge while POLG1 result pending; "
            "never start VPA empirically in an LSD patient without POLG1 screening",
            "No disease-modifying drug therapy (2026): no approved ERT, no HSCT; all pharmacological "
            "management is symptomatic (AEDs + iron + ophthalmic lubricants + sleep support); "
            "clinical trial referral is appropriate for all newly-diagnosed ML-IV patients "
            "(AAV9 gene therapy registries); contrast with MPS diseases where ERT is disease-modifying "
            "(laronidase for MPS-I, vestronidase alfa for MPS-VII, elosulfase alfa for MPS-IVA)",
        ],
        "differential_diagnosis": [
            "GNPTAB (Mucolipidosis II/III — I-Cell Disease / Pseudo-Hurler): "
            "DISTINGUISHES from ML-IV: GNPTAB has NO corneal clouding (ML-IV corneal clouding PRESENT); "
            "GNPTAB plasma enzymes 10-50x ELEVATED (ML-IV NORMAL); GNPTAB gingival hyperplasia "
            "pathognomonic from birth (ML-IV normal gingiva); GNPTAB cardiomyopathy 80% (ML-IV no cardiac); "
            "GNPTAB IS seizures dominant (ML-IV no IS); both have no ERT, no HSCT; "
            "diagnosis: if plasma enzymes ELEVATED + no corneal clouding → GNPTAB; "
            "if plasma enzymes NORMAL + corneal clouding + elevated gastrin → ML-IV",
            "MPS-I Hurler (IDUA, 4p16.3): coarse facies + dysostosis + corneal clouding; "
            "DISTINGUISHES: urine HS+DS markedly elevated (ML-IV urine GAG NORMAL); "
            "alpha-L-iduronidase very low in leukocytes; dysostosis multiplex + atlantoaxial instability; "
            "HSCT before age 2.5yr Level A (ML-IV no HSCT); laronidase ERT available; "
            "ML-IV: normal urine GAG + normal plasma enzymes + elevated gastrin — rules out MPS-I",
            "MPS-VII Sly (GUSB, 7q11.21): HS+DS+CS triple elevated in urine (PATHOGNOMONIC of MPS-VII); "
            "corneal clouding present (overlap with ML-IV); non-immune hydrops fetalis in severe MPS-VII; "
            "DISTINGUISHES: urine GAG elevated in MPS-VII (ML-IV NORMAL); GUSB enzyme low; "
            "vestronidase alfa ERT for MPS-VII (ML-IV no ERT); "
            "triple GAG elevation distinguishes MPS-VII from any form of mucolipidosis",
            "Tay-Sachs / GM2 Gangliosidosis (HEXA, 15q23): Ashkenazi founder mutations (overlap); "
            "DISTINGUISHES: Tay-Sachs has CHERRY-RED SPOT on fundoscopy (ML-IV has retinal degeneration "
            "but NO cherry-red spot); Tay-Sachs more rapid neurodegeneration with regression (ML-IV: "
            "psychomotor retardation without early regression); HEX-A very low; no corneal clouding "
            "in Tay-Sachs; cherry-red spot is the fundoscopic hallmark that separates GM2 from ML-IV",
            "Cystinosis (CTNS, 17p13.2): corneal crystals (NOT diffuse clouding — crystal deposition "
            "vs ML-IV diffuse stromal clouding); photophobia in both but crystal pattern different on "
            "slit-lamp; cystinosis: Fanconi syndrome, growth failure, renal failure (ML-IV: no renal); "
            "CTNS enzyme assay leukocyte cystine; cysteamine therapy available for cystinosis; "
            "ML-IV: normal renal function, elevated gastrin, retinal degeneration not crystalline",
        ],
    }
