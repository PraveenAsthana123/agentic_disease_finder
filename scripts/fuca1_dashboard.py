#!/usr/bin/env python3
"""FUCA1 / Fucosidosis Epilepsy Dashboard — seed data module.

Fucosidosis: alpha-L-fucosidase (FUCA1) deficiency (FUCA1, 1p36.11, AR).
Oligosaccharidosis group (not MPS): fucosylated glycoproteins, glycolipids, and
oligosaccharides accumulate in lysosomes → progressive neurodegeneration.
KEY DISTINGUISHING FEATURES:
  1. ANGIOKERATOMA CORPORIS DIFFUSUM — present in ~50% of type 2 (attenuated);
     only other oligosaccharidosis with angiokeratoma; clinically resembles Fabry
     BUT Fucosidosis is AR (both sexes), Fabry is X-linked (hemizygous males dominant).
  2. PROGRESSIVE INTELLECTUAL DISABILITY — severe in type 1, milder in type 2.
  3. NO APPROVED ERT OR SRT (2026) — supportive care + seizure management only.
  4. EPILEPSY 70-80% — predominantly generalized; spasms in infantile type 1.
Italian founder mutation: p.Arg178Ter (c.532C>T) — enriched in Southern Italy/Calabria,
accounts for ~25% of Italian alleles. Second common Italian founder: p.Glu375Ter.
Two clinical types:
  Type 1 (Severe/Infantile): onset 3-18 months, rapid neurodegeneration, death by age 5-6
    (historically; survival extended with supportive care).
  Type 2 (Attenuated/Juvenile): onset 1-3 years, slower progression, angiokeratoma
    prominent, survival into adulthood.
Epilepsy mechanism: cortical glycan accumulation → cortical hyperexcitability +
inflammatory glial activation + progressive white matter loss.
No HSCT evidence: CNS decline not halted; HSCT not standard (unlike MPS-I Hurler).
"""
import random

GENE = "FUCA1"
LOCUS = "1p36.11"
OMIM = "230000"
INHERITANCE = "Autosomal Recessive (AR) — biallelic FUCA1 LOF; both males AND females equally affected"
COHORT_SIZE = 40
DISEASE_MECHANISM = (
    "Alpha-L-fucosidase (FUCA1) deficiency → lysosomal accumulation of fucosylated "
    "glycoproteins, glycolipids, and oligosaccharides. Fucose residues normally cleaved "
    "from N- and O-linked glycans by FUCA1; without the enzyme, fucosylated substrates "
    "accumulate progressively in neurons, glial cells, hepatocytes, and vascular endothelium. "
    "CNS accumulation drives cortical glycan storage → progressive neurodegeneration, "
    "white matter loss, cortical atrophy, and cerebellar atrophy. Epilepsy arises from "
    "cortical hyperexcitability due to glycan-mediated glial activation, excitatory/inhibitory "
    "imbalance, and structural cortical damage. Angiokeratoma corporis diffusum (type 2) "
    "reflects glycan accumulation in dermal capillary endothelium — the same mechanism as "
    "Fabry disease (GLA) but FUCA1 is AR, not X-linked, and the accumulated substrate is "
    "fucosylated (not globotriaosylceramide). No lysosomal enzyme replacement therapy "
    "approved (2026); gene therapy and substrate reduction strategies remain investigational. "
    "Urine oligosaccharide screening (thin-layer chromatography or mass spectrometry) shows "
    "elevated fucosylated oligosaccharides — the biochemical diagnosis; confirmed by FUCA1 "
    "enzyme assay in leukocytes/fibroblasts (markedly reduced, <5% control)."
)

# 5 variant classes (etiologies) — deterministic percentages summing to 100
ETIOLOGIES = [
    {
        "name": "Type 1 / Severe (Italian Founder — p.Arg178Ter / Null-Null)",
        "pct": 30,
        "n": 12,
        "seizure_risk": "80-90% (cortical glycan storage dominant; epileptic spasms in infancy)",
        "eeg": "Hypsarrhythmia in infantile spasms; diffuse high-amplitude slowing; "
               "multifocal discharges; burst-suppression in end-stage; progressive "
               "electrodecrease with cortical atrophy; myoclonic jerks on EEG from age 2-3yr",
        "variant_detail": (
            "Biallelic nonsense p.Arg178Ter (Italian Calabrian founder, c.532C>T) — enzyme "
            "absent (<1% control); severe type 1 phenotype: onset 3-18 months, rapid neurodegeneration, "
            "severe intellectual disability, spastic quadriplegia, epileptic spasms progressing to "
            "GTCS + myoclonus; coarse facies, hepatosplenomegaly; historically died age 5-6 (survival "
            "extended to teens-20s with supportive care); no angiokeratoma (type 1 hallmark absent); "
            "MRI: progressive cerebral + cerebellar atrophy, white matter T2 hyperintensity"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "Type 2 / Attenuated (Angiokeratoma — Compound-Het Missense)",
        "pct": 28,
        "n": 11,
        "seizure_risk": "60-70% (slower cortical accumulation; focal and GTCS predominant)",
        "eeg": "Focal temporal or frontal discharges; mild diffuse slowing initially; "
               "background alpha preserved early; later progressive slowing; myoclonic "
               "activity at later stage; sleep EEG: increased interictal discharges; "
               "angiokeratoma does NOT produce EEG changes directly",
        "variant_detail": (
            "Compound-het: one severe + one missense allele — residual enzyme 3-10% control; "
            "attenuated type 2 phenotype: onset 1-3 years; angiokeratoma corporis diffusum "
            "prominent (50% of type 2, lower trunk/genitalia/thighs — resembles Fabry but AR); "
            "moderate-severe intellectual disability; spastic features; survival into adulthood "
            "with supportive care; second Italian founder p.Glu375Ter enriched in this group; "
            "MRI: slower cortical atrophy; diagnosis often delayed due to milder phenotype"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "Type 1 / Severe (Non-Italian Biallelic Nonsense — Private Founders)",
        "pct": 22,
        "n": 9,
        "seizure_risk": "80-85% (severe infantile; spasms + GTCS + myoclonus)",
        "eeg": "Hypsarrhythmia early; multifocal discharges; high-amplitude irregular slow "
               "background; progressive deterioration of EEG background; burst-suppression late; "
               "epileptic spasms 70% of type 1 severe group; myoclonic-astatic from year 2-3; "
               "EEG mandatory to guide ACTH therapy for spasms",
        "variant_detail": (
            "Non-Italian biallelic truncating variants (nonsense/frameshift from multiple "
            "populations — Turkish, Japanese, Spanish, Algerian founder variants); enzyme absent; "
            "same severe type 1 phenotype; no population-specific founder advantage; WGS + enzyme "
            "assay required in non-Italian populations; MRI: progressive cortical + subcortical atrophy, "
            "T2 white matter changes basal ganglia and thalami; no ERT available; ACTH Level A "
            "for infantile spasms; LEV first-line ongoing seizures"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "Type 2 / Attenuated (Biallelic Missense — Extended Survival)",
        "pct": 13,
        "n": 5,
        "seizure_risk": "50-60% (mild-moderate; focal + GTCS; late onset seizures in some)",
        "eeg": "Near-normal background early; focal slowing temporal/frontal; focal discharges "
               "without generalization early; later mild diffuse slowing; myoclonic features "
               "appear at advanced stage; EEG may be normal at diagnosis in attenuated type 2; "
               "sleep study mandatory (OSA risk from macroglossia in advanced cases)",
        "variant_detail": (
            "Biallelic missense — residual enzyme 8-18% control; attenuated phenotype: intellectual "
            "disability mild-moderate; angiokeratoma in 40-50%; prolonged survival adulthood-elderly; "
            "diagnosis often delayed 5-15 years (mild phenotype); urine oligosaccharide screening key; "
            "vacuolated lymphocytes on peripheral blood smear — diagnostic clue; MRI: mild cortical "
            "atrophy; functional ability preserved relatively longer; reproductive counseling required; "
            "no ERT but metabolic monitoring essential; genetic counseling AR"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "Rare/Private (Deep Intronic / Novel Biallelic)",
        "pct": 7,
        "n": 3,
        "seizure_risk": "65-75% (variable spectrum; diagnostic delay compounds severity)",
        "eeg": "Variable; WGS + RNA-seq required; FUCA1 enzyme assay low confirms regardless "
               "of molecular result; urine oligosaccharide chromatography elevated fucosylated "
               "bands (diagnostic); EEG: spectrum from focal to diffuse depending on phenotypic severity; "
               "treat seizures empirically by type; review EEG 3-monthly if progressive",
        "variant_detail": (
            "Deep intronic splicing or novel biallelic private mutations; WGS + RNA-seq + enzyme assay "
            "required if panel-negative with oligosaccharidosis biochemistry; vacuolated lymphocytes "
            "on blood smear is rapid clue; skin/rectal biopsy electron microscopy confirms glycan storage; "
            "phenotype variable based on residual enzyme; no ERT — supportive management; ophthalmology "
            "(lens opacities described in rare cases); audiology; dysphagia management in advanced forms"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
]

# 5 seizure types
SEIZURE_TYPES = [
    {
        "type": "Epileptic Spasms / Infantile Spasms (IS)",
        "pct": 55,
        "eeg": "Hypsarrhythmia — high-amplitude chaotic multifocal slow-and-spike; "
               "modified hypsarrhythmia with asymmetry; electrodecrement on ictal EEG; "
               "ACTH Level A (preferred over VGB — visual field monitoring impossible in "
               "severe ID); vigabatrin RELATIVE CI (irreversible visual fields — same reasoning "
               "as CLN1, Krabbe — severe ID precludes monitoring)",
    },
    {
        "type": "Generalized Tonic-Clonic (GTCS)",
        "pct": 72,
        "eeg": "Generalized spike-wave 2.5-4 Hz; bifrontal predominance; tonic phase EMG "
               "artifact then clonic; postictal generalized EEG suppression; multiple spike-wave "
               "bursts interictally; LEV Level B first-line; VPA Level B (POLG1 exclusion mandatory — "
               "unlike MPS, FUCA1 is lysosomal not mitochondrial but POLG1 exclusion best practice); "
               "PHT/fosphenytoin AVOID — aggravates myoclonus if present",
    },
    {
        "type": "Myoclonic Seizures",
        "pct": 40,
        "eeg": "Generalized polyspike-wave 4-6 Hz; cortical myoclonus with EEG back-averaging "
               "shows cortical correlate; myoclonus progresses with disease; PHT/CBZ/OXC worsen "
               "myoclonus (ABSOLUTE CI if myoclonus present); VPA Level B + levetiracetam Level B "
               "combination effective; perampanel Level C adjunct; zonisamide Level C adjunct; "
               "clonazepam Level C (tolerance limits long-term use)",
    },
    {
        "type": "Focal Seizures (Temporal / Frontal)",
        "pct": 35,
        "eeg": "Focal temporal or frontal theta-delta slow with superimposed spikes; "
               "focal onset with secondary generalization common; ictal EEG lateralized; "
               "lesional epilepsy pattern from focal cortical glycan accumulation; "
               "lacosamide Level C adjunct; LEV Level B first-line; oxcarbazepine CAUTION "
               "if myoclonus present — worsen; focal spikes may precede clinical seizures",
    },
    {
        "type": "Myoclonic-Atonic / Drop Attacks",
        "pct": 22,
        "eeg": "Slow spike-wave 1.5-2.5 Hz; atonic component with EEG generalized burst; "
               "falls with head drop; helmet mandatory; valproate + clobazam effective; "
               "felbamate Level C (hepatotoxicity risk); rufinamide Level B (Lennox-Gastaut "
               "overlap pattern); avoid CBZ/OXC/PHT (aggravate drop attacks); "
               "ketogenic diet Level B adjunct",
    },
]

# 7 seizure triggers
TRIGGERS = [
    {
        "trigger": "Febrile Illness / Intercurrent Infection",
        "pct": 68,
        "note": "Most common trigger in types 1 and 2; glycan burden increases metabolic "
                "demand; fever lowers seizure threshold; aggressive fever control mandatory "
                "(paracetamol/NSAIDs); rescue protocol required (buccal midazolam / rectal "
                "diazepam) — written action plan for caregivers; respiratory infections "
                "most dangerous (aspiration risk + fever + seizures)",
    },
    {
        "trigger": "Sleep Deprivation / Disrupted Sleep Architecture",
        "pct": 55,
        "note": "Progressive CNS involvement disrupts sleep architecture; REM reduction; "
                "increased slow-wave; nighttime seizures cluster; melatonin Level B for "
                "sleep regulation; polysomnography if OSA suspected (macroglossia rare in "
                "FUCA1 but progressive dysphagia/bulbar dysfunction in advanced disease); "
                "sleep-related EEG monitoring important in advanced cases",
    },
    {
        "trigger": "Metabolic / Nutritional Stress",
        "pct": 42,
        "note": "Inadequate nutrition (dysphagia, poor PO intake) → hypoglycemia risk → "
                "seizure exacerbation; gastrostomy (PEG) placement recommended in type 1 "
                "by age 2-3 years; dietitian-led ketogenic diet evaluation (Level B adjunct "
                "for DRE); nutritional metabolic monitoring quarterly; avoid prolonged fasting",
    },
    {
        "trigger": "General Anaesthesia / Sedation",
        "pct": 35,
        "note": "Anaesthesia risk: glycan accumulation in airway (macroglossia rare in FUCA1 "
                "vs MPS-I, but progressive bulbar dysfunction increases aspiration risk); "
                "succinylcholine CI (myopathic changes); volatile agents reduce seizure threshold "
                "in glycoprotein storage; pre-GA lysosomal enzyme screen required; post-GA "
                "seizure rescue protocol mandatory; ICU monitoring 24h post-GA in type 1",
    },
    {
        "trigger": "Missed / Sub-therapeutic AED Dosing",
        "pct": 30,
        "note": "Adherence complex in severe ID (swallowing difficulties, behavioral resistance); "
                "liquid formulations preferred; gastrostomy-based drug delivery in advanced cases; "
                "therapeutic drug monitoring (TDM) — VPA, LEV, CLB levels quarterly; caregiver "
                "education critical; rescue protocol at home for breakthrough seizures",
    },
    {
        "trigger": "Emotional / Sensory Stimulation (Startle)",
        "pct": 22,
        "note": "Startle-induced myoclonus common in advanced cortical involvement; "
                "sudden noise, touch, or visual stimuli provoke myoclonic jerks or GTCS; "
                "modify environment (sensory desensitization, low-stimulation room); "
                "clonazepam or levetiracetam reduces startle myoclonus; assess for "
                "cortical myoclonus with EEG back-averaging",
    },
    {
        "trigger": "Photosensitivity (Photic)",
        "pct": 15,
        "note": "Photoparoxysmal response on EEG in minority (~15%); screen with IPS during "
                "routine EEG; tinted lenses for photosensitive individuals; avoid flicker "
                "sources (screens, strobe); valproate or levetiracetam reduces photosensitivity; "
                "VGB RELATIVE CI (visual field monitoring impossible in severe ID — prefer ACTH "
                "for spasms, VPA/LEV for photosensitive GTCS)",
    },
]

# 7 treatments
TREATMENTS = [
    {
        "name": "Levetiracetam (LEV)",
        "level": "Level B — First-Line",
        "role": "GTCS, focal, myoclonic seizures; broad-spectrum; renal dosing; liquid "
                "formulation available (gastrostomy compatible); behavioral side-effects "
                "(agitation, aggression) monitored — may overlap with baseline ID behavior; "
                "TDM if response uncertain",
        "ci": None,
    },
    {
        "name": "Valproic Acid (VPA)",
        "level": "Level B — First-Line (POLG1 Exclusion Mandatory)",
        "role": "GTCS, myoclonic, myoclonic-atonic; broad-spectrum; POLG1 exclusion before "
                "initiation (best practice; FUCA1 lysosomal not mitochondrial, but CPIC A "
                "recommendation applies); LFTs 3-monthly; ammonium levels if encephalopathic; "
                "hyperammonemia risk; carnitine supplementation if VPA-related",
        "ci": "Contraindicated if POLG1 pathogenic variant identified",
    },
    {
        "name": "ACTH / Corticotropin",
        "level": "Level A — Infantile Spasms (IS)",
        "role": "Preferred over VGB for spasms in FUCA1 (severe ID precludes visual field "
                "monitoring — VGB irreversible visual loss); high-dose ACTH (150 IU/m²/day) "
                "14 days then taper; EEG resolution of hypsarrhythmia endpoint; monitor BP, "
                "glucose, infection risk; equivalent to prednisolone in some protocols",
        "ci": "Systemic infection (relative CI); hypertension monitoring required",
    },
    {
        "name": "Clobazam (CLB)",
        "level": "Level B — Adjunct",
        "role": "GTCS adjunct, myoclonic-atonic, focal; 1,5-benzodiazepine (less sedating "
                "than clonazepam); liquid formulation; tolerance develops; intermittent "
                "use (cluster prevention, febrile protocol); drug interactions (VPA elevates "
                "active metabolite N-desmethylclobazam)",
        "ci": None,
    },
    {
        "name": "Rufinamide",
        "level": "Level B — Myoclonic-Atonic / Drop Attacks",
        "role": "Lennox-Gastaut overlap pattern (myoclonic-atonic drop attacks); reduces "
                "atonic components; VPA co-administration increases rufinamide levels (dose "
                "reduce rufinamide 50% if VPA co-prescribed); QTc monitoring; gastrostomy "
                "compatible (tablet crushed or suspension)",
        "ci": None,
    },
    {
        "name": "Ketogenic Diet (KD)",
        "level": "Level B — DRE Adjunct",
        "role": "Drug-resistant epilepsy (GTCS, myoclonic, spasms refractory to 2+ AEDs); "
                "classical 4:1 KD or MCT oil protocol; dietitian-led implementation; "
                "gastrostomy-compatible KD formula available; monitor lipids, renal function, "
                "growth; particularly useful in type 1 severe with multiple seizure types; "
                "anticonvulsant mechanism via GABA-A, HCN1, and mTOR pathways",
        "ci": "Fatty acid oxidation defects must be excluded before initiation",
    },
    {
        "name": "Buccal Midazolam / Rectal Diazepam (Rescue)",
        "level": "Level A — Status Epilepticus / Rescue",
        "role": "Acute cluster or prolonged seizure protocol; caregiver-administered at home "
                "(5-min rule: seizure >5 min OR 2+ seizures without return to baseline → "
                "rescue dose); written emergency protocol mandatory; school/residential care "
                "staff trained; IV lorazepam if rescue fails (hospital setting); IV LEV "
                "second-line SE (preferred over fosphenytoin — avoid PHT in myoclonic disease)",
        "ci": None,
    },
]

# 5 contraindications
CONTRAINDICATIONS = [
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin",
        "risk": "ABSOLUTE CI — Myoclonus Worsening",
        "reason": "Phenytoin aggravates cortical myoclonus and myoclonic seizures (blocks "
                  "Na⁺ channels → paradoxical increase in cortical excitability in myoclonic "
                  "epilepsy pattern); NEVER use for status epilepticus if myoclonus present — "
                  "IV LEV or IV valproate replaces PHT/fosphenytoin in FUCA1 SE protocol; "
                  "myoclonus present in 40% of FUCA1 — assume present if uncharacterized",
        "alternative": "IV Levetiracetam (60 mg/kg over 10 min) for SE; VPA IV for SE if POLG1-cleared",
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "risk": "RELATIVE CI — Myoclonus / HIGH RISK",
        "reason": "CBZ/OXC aggravate myoclonic and myoclonic-atonic seizures (Na⁺ channel "
                  "blockade worsens cortical myoclonus pattern); RELATIVE CI (not absolute) "
                  "because focal seizures without myoclonus may benefit — use ONLY if EEG "
                  "confirms pure focal seizures without myoclonic features; reassess at each "
                  "review with EEG; peripheral neuropathy in advanced FUCA1 makes CBZ-related "
                  "neuropathy confounded",
        "alternative": "Lacosamide (focal, no myoclonus worsening); LEV or LCM for focal seizures",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "risk": "RELATIVE CI — Irreversible Visual Field Loss",
        "reason": "VGB causes irreversible bilateral concentric visual field constriction "
                  "(30-50% of long-term users); visual field perimetry impossible in severe "
                  "ID — defect undetectable until advanced; RELATIVE CI (not absolute): "
                  "acceptable ONLY if ACTH fails for spasms and no alternative exists; "
                  "OCT retinal monitoring is alternative but requires cooperation; "
                  "prefer ACTH Level A for IS in FUCA1",
        "alternative": "ACTH Level A for infantile spasms; VPA + LEV for other seizure types",
    },
    {
        "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine)",
        "risk": "HIGH RISK — Glycoprotein Accumulation / EPS",
        "reason": "Glycoprotein accumulation in basal ganglia from FUCA1 deficiency → "
                  "increased dopamine receptor sensitivity; typical antipsychotics block D2 "
                  "receptors → dystonia, rigidity, tardive dyskinesia at lower doses than "
                  "neurotypical; behavioral issues common in ID — atypical antipsychotics "
                  "(aripiprazole, quetiapine low-dose) preferred if absolutely needed; "
                  "NMS risk elevated in glycoprotein storage disorders",
        "alternative": "Atypical antipsychotics (aripiprazole first; low-dose quetiapine); behavioral therapy",
    },
    {
        "drug": "POLG1 — VPA Decision",
        "risk": "MANDATORY EXCLUSION — CPIC A Best Practice",
        "reason": "FUCA1 deficiency is lysosomal (not mitochondrial), so POLG1 hepatotoxicity "
                  "risk is not directly amplified by FUCA1 enzyme deficiency; however, CPIC Level A "
                  "mandates POLG1 testing before VPA in ANY patient with neurological disease + "
                  "progressive features — Fucosidosis qualifies due to progressive neurodegeneration; "
                  "if POLG1 pathogenic variant found: valproate ABSOLUTELY contraindicated "
                  "(fatal hepatotoxicity); if POLG1 wild-type: VPA safe to use with LFT monitoring",
        "alternative": "LEV or CLB if POLG1 positive; VPA if POLG1 confirmed wild-type",
    },
]

# Definitions / glossary
DEFINITIONS = {
    "gene": GENE,
    "full_name": "Alpha-L-Fucosidase 1 (FUCA1)",
    "disease": "Fucosidosis (Alpha-L-Fucosidase Deficiency / Oligosaccharidosis)",
    "omim": OMIM,
    "locus": LOCUS,
    "inheritance": INHERITANCE,
    "enzyme_defect": "Alpha-L-fucosidase deficiency → lysosomal accumulation of fucosylated glycoproteins, glycolipids, oligosaccharides",
    "storage_material": "Fucosylated oligosaccharides and glycolipids (NOT GAGs — differs from MPS)",
    "ert": "None approved (2026) — investigational gene therapy (AAV9/AAVrh10-FUCA1)",
    "hsct": "NOT standard — CNS decline not halted by HSCT; supportive care only",
    "epilepsy_pct": "70-80% (type 1: 80-90%; type 2: 60-70%)",
    "dre_pct": "30-40% drug-resistant (type 1 dominant)",
    "key_distinguishing": (
        "Angiokeratoma corporis diffusum in TYPE 2 (50%) — only oligosaccharidosis with angiokeratoma; "
        "resembles Fabry (GLA) BUT Fucosidosis is AR (both sexes), Fabry is XL (hemizygous males); "
        "NO approved ERT unlike MPS types (Hurler-laronidase, Hunter-idursulfase, Maroteaux-Lamy-galsulfase); "
        "urine oligosaccharide chromatography elevated fucosylated bands (vs GAG-specific MPS); "
        "vacuolated lymphocytes on blood smear (diagnostic clue); FUCA1 enzyme assay confirming"
    ),
    "founder_mutation": "p.Arg178Ter (c.532C>T) — Italian Calabrian founder ~25% of Italian alleles; p.Glu375Ter — second Italian founder",
    "polg1_mandatory": True,
    "differential": (
        "Fabry disease (GLA): angiokeratoma XL vs AR; globotriaosylceramide vs fucosylated glycans; "
        "ERT available (agalsidase); males predominantly; "
        "MPS-I/II/III: GAG elevation vs oligosaccharides; ERT available for some; "
        "GM1-gangliosidosis (GLB1): ganglioside accumulation; cherry-red spot; "
        "Aspartylglucosaminuria (AGA): aspartylglucosamine accumulation; Finnish founder; "
        "Alpha-mannosidosis (MAN2B1): mannose-rich oligosaccharides; hearing loss; "
        "Sialidosis (NEU1): sialylated oligosaccharides; cherry-red spot + myoclonus"
    ),
}

ABBREVIATIONS = {
    "FUCA1": "Alpha-L-Fucosidase 1 gene (1p36.11)",
    "AR": "Autosomal Recessive — both males and females equally affected",
    "IS": "Infantile Spasms (Epileptic Spasms) — ACTH Level A preferred (VGB relative CI)",
    "GTCS": "Generalized Tonic-Clonic Seizures",
    "DRE": "Drug-Resistant Epilepsy (failure of 2+ appropriate AEDs)",
    "ERT": "Enzyme Replacement Therapy — NONE APPROVED for Fucosidosis (2026)",
    "HSCT": "Hematopoietic Stem Cell Transplant — NOT standard in Fucosidosis",
    "VGB": "Vigabatrin — RELATIVE CI (irreversible visual field loss; monitoring impossible severe ID)",
    "PHT": "Phenytoin — ABSOLUTE CI if myoclonus (aggravates cortical myoclonus)",
    "CBZ/OXC": "Carbamazepine/Oxcarbazepine — RELATIVE CI (myoclonus worsening)",
    "LEV": "Levetiracetam — Level B first-line (GTCS, focal, myoclonic)",
    "VPA": "Valproic Acid — Level B (POLG1 exclusion mandatory; LFTs 3-monthly)",
    "ACTH": "Adrenocorticotrophic Hormone — Level A for infantile spasms",
    "KD": "Ketogenic Diet — Level B DRE adjunct",
    "TDM": "Therapeutic Drug Monitoring",
    "POLG1": "Polymerase Gamma 1 — mitochondrial; exclusion mandatory before VPA (CPIC A)",
    "Oligosaccharidosis": "Group of LSDs with oligosaccharide/glycoprotein accumulation (not GAG) — Fucosidosis, Mannosidosis, Sialidosis, Aspartylglucosaminuria",
    "Angiokeratoma": "Dark-red/blue skin lesions from glycan accumulation in dermal capillaries — type 2 Fucosidosis + Fabry disease",
    "CPIC A": "Clinical Pharmacogenomics Implementation Consortium Grade A — strongest recommendation",
}

CLINICAL_PEARLS = [
    {
        "pearl": "Angiokeratoma in type 2 FUCA1 — always check pedigree: AR vs XL",
        "detail": (
            "Angiokeratoma corporis diffusum: when seen in epilepsy patient, differential is "
            "Fabry disease (XL — males, GLA gene, ERT available) vs Fucosidosis type 2 (AR — "
            "both sexes, FUCA1 gene, NO ERT). Key differentiation: sex inheritance pattern "
            "(Fabry: predominantly hemizygous males; Fucosidosis: both sexes equally); "
            "enzyme assay: alpha-galactosidase A (GLA) vs alpha-L-fucosidase (FUCA1); "
            "urine substrate: globotriaosylceramide (Fabry) vs fucosylated oligosaccharides (Fucosidosis). "
            "ERROR to assume angiokeratoma = Fabry in a female with epilepsy + ID — check FUCA1 first."
        ),
    },
    {
        "pearl": "PHT absolute CI if myoclonus — use IV LEV for status epilepticus",
        "detail": (
            "In Fucosidosis SE with myoclonic features: PHT/fosphenytoin ABSOLUTE CI (worsens "
            "cortical myoclonus). Emergency protocol: buccal midazolam → IV lorazepam → IV LEV "
            "(60 mg/kg) → IV valproate (if POLG1-cleared) → anaesthetic options. "
            "NEVER use PHT/fosphenytoin as second-line for SE in any myoclonic epilepsy disorder. "
            "Train ED staff: Fucosidosis ID bracelet should indicate PHT CI."
        ),
    },
    {
        "pearl": "Urine oligosaccharide chromatography — not routine GAG screen",
        "detail": (
            "Standard urine GAG screen (heparan, dermatan, keratan sulfate) is NORMAL in Fucosidosis. "
            "Specific oligosaccharide screen (TLC or mass spec) required — elevates fucosylated "
            "oligosaccharide bands. Clinicians ordering 'urine MPS screen' will MISS Fucosidosis. "
            "Vacuolated lymphocytes on blood smear + urine oligosaccharide chromatography + FUCA1 "
            "enzyme assay = complete diagnostic workup for oligosaccharidosis workup. "
            "Differentiate from MPS by substrate specificity."
        ),
    },
    {
        "pearl": "ACTH preferred over VGB for infantile spasms — visual monitoring impossible",
        "detail": (
            "Vigabatrin causes irreversible concentric visual field loss in 30-50% of long-term users. "
            "In Fucosidosis type 1 severe ID: visual field Goldman perimetry impossible; OCT requires "
            "cooperation (difficult in infant + severe ID); VGB visual loss undetectable until advanced. "
            "ACTH Level A (150 IU/m²/day × 14 days) is the preferred first-line for IS in Fucosidosis. "
            "VGB reserved as last resort only if ACTH fails AND no alternative; document informed consent "
            "regarding irreversible visual risk."
        ),
    },
]

MONITORING_PARAMETERS = [
    "EEG 6-monthly (type 1); annually (type 2) — document progression, guide AED titration",
    "MRI brain 12-monthly in type 1 (cortical + cerebellar atrophy progression); 2-yearly type 2",
    "LFTs 3-monthly if on VPA (POLG1 exclusion mandatory before initiation)",
    "Vacuolated lymphocytes on blood smear — at diagnosis and annually (marker of storage burden)",
    "Urine oligosaccharide chromatography — at diagnosis; annually to monitor biochemical progression",
    "FUCA1 enzyme level — leukocytes/fibroblasts at diagnosis; family cascade screening (AR counseling)",
    "Ophthalmology annually — lens opacities; angiokeratoma skin mapping in type 2",
    "Dermatology 6-monthly — angiokeratoma extent in type 2 (lower trunk, genitalia, thighs)",
    "Swallowing assessment (FEES or videofluoroscopy) 6-monthly in type 1 advanced — PEG timing",
    "Neuropsychological battery annually — adaptive function, developmental regression monitoring",
    "Echocardiogram if progressive dyspnea (cardiac glycan involvement rare but reported)",
    "POLG1 genotyping before VPA initiation — report turnaround 2-4 weeks; interim LEV",
    "TDM: VPA trough, LEV trough, CLB + N-desmethylclobazam quarterly",
    "Bone density (DEXA) 2-yearly in ambulatory patients (immobilization + anticonvulsant effect)",
    "Genetic counseling for parents (recurrence risk 25% AR) + extended family cascade",
]

REFERENCES = [
    "Willems PJ, et al. Fucosidosis revisited. Am J Med Genet. 1991;38(1):111-131.",
    "Kousseff BG, et al. Fucosidosis type 2. Pediatrics. 1976;57(2):205-213.",
    "Stepien KM, et al. Lysosomal storage disorders: pitfalls in diagnosis. J Inherit Metab Dis. 2022.",
    "Orphanet: Fucosidosis (ORPHA:349). www.orpha.net",
    "OMIM #230000 — FUCOSIDOSIS. www.omim.org/entry/230000",
    "CPIC Guidelines — Valproate/POLG1 interaction. cpicpgx.org",
    "Barone R, et al. Biochemical diagnosis of oligosaccharidoses. Clin Biochem. 2010.",
]


def get_overview():
    random.seed(42)
    kpis = [
        {"label": "Gene", "value": GENE},
        {"label": "Locus", "value": LOCUS},
        {"label": "Inheritance", "value": "AR"},
        {"label": "Cohort", "value": f"{COHORT_SIZE} pts"},
        {"label": "Epilepsy", "value": "70-80%"},
        {"label": "DRE Rate", "value": "30-40%"},
        {"label": "ERT", "value": "None (2026)"},
        {"label": "HSCT", "value": "Not standard"},
        {"label": "Italian Founder", "value": "p.Arg178Ter"},
        {"label": "POLG1 Excl.", "value": "Mandatory"},
    ]
    return {
        "kpis": kpis,
        "disease_mechanism": DISEASE_MECHANISM,
        "epilepsy_prevalence_pct": 75,
        "drug_resistance_pct": 35,
        "angiokeratoma_pct_type2": 50,
        "type1_severe_pct": 52,
        "type2_attenuated_pct": 41,
        "progressive_neurodegeneration": True,
        "clinical_pearls": CLINICAL_PEARLS,
        "monitoring_parameters": MONITORING_PARAMETERS,
    }


def get_breakdown():
    random.seed(42)
    return {
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "cohort_size": COHORT_SIZE,
        "gene": GENE,
        "disease": "Fucosidosis",
    }


def get_definitions():
    return {
        **DEFINITIONS,
        "abbreviations": ABBREVIATIONS,
        "references": REFERENCES,
        "clinical_pearls": CLINICAL_PEARLS,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2, default=str)[:500])
    print("\n=== BREAKDOWN ===")
    print(json.dumps(get_breakdown(), indent=2, default=str)[:500])
    print("\n=== DEFINITIONS ===")
    print(json.dumps(get_definitions(), indent=2, default=str)[:500])
