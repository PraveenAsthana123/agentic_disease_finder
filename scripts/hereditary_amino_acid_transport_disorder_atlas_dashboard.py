#!/usr/bin/env python3
"""Hereditary-Amino-Acid-Transport-Disorder-Atlas -- Complete 8-Gene Reference
Hereditary Defects of Amino Acid and Small Molecule Transporters:
SLC6A19-SLC3A1-SLC7A9-SLC7A7-SLC17A5-SLC6A8-SLC6A5-SLC25A22
(SLC6A19/Hartnup-634aa-5p15.33-AR-B0AT1-NeutralAminoAcid-PellagraLike-Tryptophan-Malabsorption-Nicotinamide-CURATIVE,
 SLC3A1/Cystinuria-TypeA-685aa-2p21-AR-rBAT-DibasicAminoAcid-Cystine-Stones-PATHOGNOMONIC-Tiopronin,
 SLC7A9/Cystinuria-TypeB-487aa-19q13.11-AR-b0+AT-DibasicAminoAcid-Cystine-Stones-SamePhenotype-TypeA,
 SLC7A7/LPI-511aa-14q11.2-AR-y+LAT1-DibasicAminoAcid-Efflux-Hyperammonemia-AlveolarProteinosis-PATHOGNOMONIC,
 SLC17A5/SallaDisease-ISSD-495aa-6q13-AR-Sialin-LysosomalSialicAcid-Finnish-Founder,
 SLC6A8/CreatineTransporterDeficiency-636aa-Xq28-XLR-CrT-NoBrainCreatine-MRS-PATHOGNOMONIC,
 SLC6A5/Hyperekplexia3-799aa-11p15.1-AR-GlyT2-GlycineTransporter-ExaggeratedStartle-PATHOGNOMONIC-Clonazepam,
 SLC25A22/EarlyEpilepticEncephalopathy-323aa-11p15.5-AR-GC1-MitochondrialGlutamateCarrier-Hypsarrhythmia-KD):

SLC6A19 (B0 neutral amino acid transporter 1 / B0AT1; 634aa; 5p15.33; AR-LOF;
     SLC6A19 encodes B0AT1, the apical brush-border neutral amino acid transporter in intestine and kidney;
     B0AT1 transports all neutral (monoamino-monocarboxylic) amino acids — especially tryptophan, leucine, phenylalanine;
     requires ancillary protein ACE2 (intestine) or collectrin (kidney) for membrane trafficking;
     SLC6A19 DEFICIENCY -- HARTNUP DISEASE -- OMIM 234500;
     MECHANISM: B0AT1 LOF → intestinal malabsorption of neutral amino acids → tryptophan deficiency →
       tryptophan → nicotinic acid (niacin/NAD) pathway impaired → pellagra-like syndrome;
       renal reabsorption failure: massive neutral aminoaciduria (DIAGNOSTIC);
     PATHOGNOMONIC FEATURES:
       (1) Pellagra-like photosensitive rash: sun-exposed areas; redness, scaling, hyperpigmentation;
           differentiates from true pellagra only by aminoaciduria;
       (2) Intermittent cerebellar ataxia: triggered by sun, fever, stress, nutritional deficiency;
       (3) Neutral aminoaciduria on urine AA chromatography: ALL neutral AAs elevated (tryptophan, alanine,
           serine, threonine, asparagine, glutamine, valine, leucine, isoleucine, phenylalanine, tyrosine);
     INCOMPLETE PENETRANCE: many SLC6A19 homozygotes are ASYMPTOMATIC (require dietary tryptophan deficiency trigger);
     DIAGNOSIS: urine amino acid chromatography — isolated neutral aminoaciduria (NOT dibasic, NOT dicarboxylic);
     TREATMENT: nicotinamide (niacinamide) 50-100 mg/day — CURATIVE for rash and ataxia;
       high-protein diet; sun protection; avoid triggers;
     seed SEED_BASE+0).

SLC3A1 (Neutral and basic amino acid transport protein rBAT; 685aa; 2p21; AR-LOF;
     SLC3A1 encodes rBAT (related to b0,+ amino acid transporter), the heavy subunit of the b0,+ transporter;
     rBAT forms a disulfide-linked heterodimer with SLC7A9 (b0,+AT light chain);
     the complex mediates apical reabsorption of dibasic amino acids (lysine, arginine, ornithine) AND cystine
     in renal proximal tubule and intestinal epithelium;
     SLC3A1 DEFICIENCY -- CYSTINURIA TYPE A -- OMIM 220100;
     MECHANISM: rBAT LOF → cystine accumulates in renal tubular lumen → cystine stones (cystine insoluble at pH <7.0);
     PATHOGNOMONIC FEATURE: CYSTINE UROLITHIASIS;
       Cystine stones: radiopaque (poorly, less than calcium), staghorn morphology, early onset (childhood/adolescence);
       Recurrent nephrolithiasis; renal colic; urinary tract obstruction; risk of chronic kidney disease;
       CYANIDE-NITROPRUSSIDE TEST: positive — rapid screening (turns purple-red);
       Urine amino acids: cystine + lysine + arginine + ornithine elevated (ALL dibasic AAs);
       Stone analysis: cystine by infrared spectroscopy PATHOGNOMONIC;
     GENOTYPE-PHENOTYPE:
       Type A (SLC3A1 biallelic): homozygous → complete defect; heterozygous: ASYMPTOMATIC (renal dibasic AA mild);
       Type B (SLC7A9 biallelic): see below;
       Type AB (mixed): SLC3A1 + SLC7A9 one allele each — some have intermediate phenotype;
     TREATMENT:
       High fluid intake (>3L/day): first-line to dilute urine cystine;
       Urine alkalinization: potassium citrate or sodium bicarbonate → pH >7.5 (cystine more soluble);
       Tiopronin (THIOLA): chelating agent — forms mixed disulfide with cystine (more soluble); preferred over D-penicillamine;
       D-Penicillamine: effective but more side effects; second-line;
       Urologic: lithotripsy, ureteroscopy for established stones;
     seed SEED_BASE+1).

SLC7A9 (B(0,+)-type amino acid transporter 1 / b0,+AT; 487aa; 19q13.11; AR-LOF;
     SLC7A9 encodes the light chain (b0,+AT) of the b0,+ amino acid transporter heterodimer;
     SLC7A9 forms the catalytic subunit; SLC3A1 (rBAT) is the chaperone/heavy chain;
     SLC7A9 DEFICIENCY -- CYSTINURIA TYPE B -- OMIM 600918;
     PHENOTYPE: IDENTICAL to Type A cystinuria (SLC3A1);
       Cystine urolithiasis; recurrent nephrolithiasis; staghorn calculi; CKD risk;
       Urine: cystine + dibasic AAs elevated;
     IMPORTANT DISTINCTION FROM TYPE A:
       Heterozygous SLC7A9 carriers: may have INTERMEDIATE aminoaciduria (incomplete dominant);
         some heterozygotes develop stones (incomplete penetrance dominance);
         in contrast: SLC3A1 heterozygotes are completely asymptomatic;
       This heterozygote effect is clinically significant for carrier counseling;
     GENOTYPE-PHENOTYPE: p.A182T (Mediterranean founder); p.G105R;
     TREATMENT: identical to type A — hydration, alkalinization, tiopronin; urologic management;
     COMBINED TYPE AB: one SLC3A1 + one SLC7A9 pathogenic variant → phenotype similar to types A and B;
     seed SEED_BASE+2).

SLC7A7 (Y+L amino acid transporter 1 / y+LAT1; 511aa; 14q11.2; AR-LOF;
     SLC7A7 encodes y+LAT1, the light chain of the y+L amino acid transporter;
     this transporter mediates EFFLUX of dibasic amino acids (lysine, arginine, ornithine) from cells
     and small intestinal absorption of dibasic amino acids;
     SLC7A7 DEFICIENCY -- LYSINURIC PROTEIN INTOLERANCE (LPI) -- OMIM 222700;
     MECHANISM: y+LAT1 LOF → dibasic AAs (lysine, arginine, ornithine) trapped INSIDE epithelial cells;
       intestinal absorption of lysine/arginine/ornithine SEVERELY impaired;
       ornithine and arginine deficiency → urea cycle partial impairment → hyperammonemia;
       lysine deficiency → growth failure; poor protein tolerance;
     PATHOGNOMONIC CLINICAL FEATURES:
       (1) PULMONARY ALVEOLAR PROTEINOSIS (PAP): present in 60-70%; alveolar proteinaceous material;
           recurrent pulmonary infections; dyspnea; ground-glass opacities on CT; BAL shows foamy macrophages;
           PAP in LPI distinguishes from cystinuria (no lung disease);
       (2) HYPERAMMONEMIA after protein-rich meals: post-prandial hyperammonemia; encephalopathy risk;
           plasma NH3 elevated especially post-protein load;
       (3) HEMOPHAGOCYTIC LYMPHOHISTIOCYTOSIS (HLH): life-threatening complication (~20%); trigger any infection;
           recurrent macrophage activation;
       (4) Urine: dibasic aminoaciduria (lysine, arginine, ornithine ELEVATED);
           plasma: lysine, arginine, ornithine paradoxically LOW (trapped intracellularly);
       (5) Orotic acid elevated (ornithinemia diverts → orotic acid synthesis);
       (6) OSTEOPOROSIS: lysine deficiency impairs collagen cross-linking; vertebral fractures in young adults;
     TREATMENT:
       Low-protein diet + protein restriction: reduces post-prandial hyperammonemia;
       L-citrulline supplementation: citrulline bypasses y+LAT1 → enters cells directly → replenishes ornithine/arginine;
         citrulline supplement reduces hyperammonemia DRAMATICALLY; first-line supplement;
       Lysine supplement: prevents growth failure;
       Hydroxychloroquine: for PAP (reduces macrophage activation); second-line;
       HLH: cyclosporine, etoposide per HLH-94/2004 protocol if triggered;
     seed SEED_BASE+3).

SLC17A5 (Sialin; 495aa; 6q13; AR-LOF;
     SLC17A5 encodes sialin, a lysosomal H+/sialic acid cotransporter;
     sialin exports free sialic acid (N-acetylneuraminic acid, Neu5Ac) from lysosomes to cytoplasm;
     SLC17A5 DEFICIENCY -- TWO PHENOTYPES SAME GENE:
       (1) INFANTILE SIALIC ACID STORAGE DISEASE (ISSD) -- OMIM 269920: severe, lethal in infancy;
       (2) SALLA DISEASE -- OMIM 604369: milder, adult survival; FINNISH FOUNDER DISEASE (p.R39C);
     MECHANISM: sialin LOF → free sialic acid accumulates in lysosomes (cannot be exported);
       lysosomal engorgement; cellular dysfunction; brain predominantly affected;
     CLINICAL FEATURES — ISSD (severe):
       Hydrops fetalis or severe fetal ascites;
       Neonatal: severe hypotonia, hepatosplenomegaly, course facies, severe ID;
       Cardiomegaly; bone abnormalities (severe dysostosis multiplex);
       Death in infancy (months to 2 years); profound neurological impairment;
     CLINICAL FEATURES — SALLA DISEASE (mild):
       Finnish founder (p.R39C homozygous): most common;
       Intellectual disability: mild-to-severe (Finnish mean IQ 30-50);
       Ataxia: cerebellar; nystagmus; hypotonia;
       Spasticity develops; epilepsy in 50%;
       White matter abnormalities: dysmyelination on MRI;
       Survival into adulthood;
     PATHOGNOMONIC DIAGNOSIS:
       Urine free sialic acid: MARKEDLY ELEVATED (100-1000x normal) — PATHOGNOMONIC;
       Lysosomal free sialic acid (cultured fibroblasts): elevated;
       SLC17A5 sequencing: p.R39C = Salla; severe mutations = ISSD;
     TREATMENT: no approved disease-modifying therapy; supportive;
       Salla disease: physiotherapy, communication aids, epilepsy management;
       ISSD: palliative;
     seed SEED_BASE+4).

SLC6A8 (Sodium- and chloride-dependent creatine transporter 1 / CrT1; 636aa; Xq28; XLR;
     SLC6A8 encodes CrT1, the Na+/Cl−-dependent creatine transporter at the blood-brain barrier and neurons;
     creatine synthesis occurs in liver/kidney (AGAT + GAMT enzymes) → creatine exported → enters brain via CrT1;
     SLC6A8 DEFICIENCY -- CREATINE TRANSPORTER DEFICIENCY (CTD) / CEREBRAL CREATINE DEFICIENCY SYNDROME 1 -- OMIM 300352;
     X-LINKED: males predominantly and severely affected (hemizygous);
       females: variable (heterozygous carriers — may be mildly affected or asymptomatic);
     MECHANISM: CrT1 LOF → brain cannot import creatine → brain creatine/phosphocreatine ABSENT;
       peripheral creatine (liver/kidney, muscle) NORMAL or elevated (unable to enter brain);
     PATHOGNOMONIC BRAIN MRS:
       (1) ABSENT brain creatine peak at 3.03 ppm on 1H-MRS PATHOGNOMONIC;
       (2) Urine creatine/creatinine ratio: ELEVATED (creatine not taken up by brain → spills in urine);
       (3) Plasma creatine: normal or elevated; urine creatine elevated;
     CLINICAL FEATURES:
       Intellectual disability: mild-to-severe; most males have severe ID;
       Speech/language delay: often disproportionate (expressive language most affected);
       Behavioral abnormalities: autistic features; self-injurious behavior; hyperactivity (30-40%);
       Epilepsy: variable severity; may be refractory;
       Hypotonia;
       Females: most are mild/unaffected but some have learning difficulties, speech delay (X-inactivation dependent);
     TREATMENT:
       Creatine supplementation: INCREASES urine creatine but does NOT improve brain MRS (cannot cross BBB);
       L-arginine + glycine (creatine synthesis precursors): NOT effective if transporter absent;
       Current trials: prodrugs designed to bypass CrT1; cyclocreatine; guanidinoacetate not helpful;
       Gene therapy: investigational;
       Management: antiepileptics; behavioral therapy; speech therapy;
     seed SEED_BASE+5).

SLC6A5 (Sodium- and chloride-dependent glycine transporter 2 / GlyT2; 799aa; 11p15.1; AR-LOF;
     SLC6A5 encodes GlyT2, the presynaptic glycine transporter at inhibitory glycinergic synapses;
     GlyT2 is responsible for reuptake of glycine from the synapse into the presynaptic terminal
     — required to maintain glycine vesicular stores for inhibitory neurotransmission;
     inhibitory glycinergic synapses mediate reciprocal inhibition in spinal cord reflex circuits;
     SLC6A5 DEFICIENCY -- HYPEREKPLEXIA TYPE 3 (HKPG3) / GLYCINE TRANSPORTER 2 DEFICIENCY -- OMIM 149400 (hyperekplexia);
     MECHANISM: GlyT2 LOF → impaired glycine recycling at inhibitory synapses → depleted glycine vesicular stores →
       DISINHIBITION of spinal cord interneurons → EXAGGERATED STARTLE RESPONSE;
     PATHOGNOMONIC CLINICAL FEATURES:
       (1) EXAGGERATED STARTLE REFLEX (hyperekplexia): sudden loud noise, touch, or unexpected stimulus →
           massive, non-habituating whole-body startle with generalized stiffness;
           PATHOGNOMONIC: NON-HABITUATING (normal startle habituates after 3-5 stimuli; hyperekplexia does NOT);
       (2) NEONATAL HYPERTONIA: excessive rigidity in neonates; may look like opisthotonus;
           NOSE TAP REFLEX: tapping nose → non-habituating startle = bedside test;
           TONIC SPASMS: apnea episodes from startle-induced rigidity (life-threatening in neonates);
       (3) STIFFNESS: episodic generalized stiffness triggered by startle;
       (4) EEG: usually NORMAL (not epilepsy — distinguish from neonatal seizures);
       (5) EMG: normal; spinal cord interneuron pathophysiology;
     DIFFERENTIAL: GLYCINE RECEPTOR SUBUNIT GENES (GLRA1, GLRB) cause hyperekplexia types 1+2;
       SLC6A5 = type 3; presynaptic (transport) vs. postsynaptic (receptor);
     TREATMENT:
       Clonazepam: first-line; GABA-A potentiation → increases inhibitory tone; highly effective;
       Glycine supplement: may help — increases substrate availability;
       HEIMLICH MANEUVER: CRITICAL for neonates with startle-induced apnea (stimulates swallowing reflex
         to abort tonic episode); parents must learn LIFE-SAVING TECHNIQUE;
     seed SEED_BASE+6).

SLC25A22 (Mitochondrial glutamate carrier 1 / GC1; 323aa; 11p15.5; AR-LOF;
     SLC25A22 encodes the mitochondrial glutamate carrier 1 (GC1), mediating glutamate import into mitochondria
     in exchange for hydroxyl ions (H−/Glu exchange) — critical for neuronal energy metabolism and
     aspartate-malate shuttle;
     neuronal mitochondria depend on glutamate transport for TCA anaplerosis and GABA synthesis;
     SLC25A22 DEFICIENCY -- EARLY MYOCLONIC ENCEPHALOPATHY (EME) / NEONATAL EPILEPTIC ENCEPHALOPATHY -- OMIM 609304;
     MECHANISM: GC1 LOF → neuronal glutamate entry into mitochondria impaired →
       GABA synthesis impaired (glutamate → GABA via GAD) + energy deficit + anaplerosis failure →
       severe epileptic encephalopathy;
     PATHOGNOMONIC CLINICAL FEATURES:
       (1) NEONATAL ONSET REFRACTORY MYOCLONIC EPILEPSY: myoclonic jerks from FIRST DAYS of life;
           erratic myoclonus (asynchronous, multifocal) = characteristic;
           seizures begin day 0-3 of life;
       (2) EEG: BURST SUPPRESSION in neonatal period → HYPSARRHYTHMIA later (first months);
           burst-suppression → hypsarrhythmia transition is nearly PATHOGNOMONIC for SLC25A22-EME;
       (3) Profound global developmental delay: no motor or cognitive development;
       (4) Hypotonia: severe; virtually no voluntary movement;
       (5) Severe microcephaly: progressive after birth;
       (6) ABSENCE of dysmorphic features: normal external appearance, no organomegaly;
     BIOCHEMISTRY:
       Plasma + CSF amino acids: NO diagnostic pattern (unlike GLDC/nonketotic hyperglycinemia);
       CSF glutamate may be low (but not consistently);
       Organic acids: NORMAL (distinguishes from organic acidemias);
     TREATMENT:
       KETOGENIC DIET: partial response in some patients; reduces seizure frequency;
         ketone bodies provide alternative anaplerotic substrate bypassing GC1;
       Antiepileptics: largely refractory; phenobarbital, clonazepam, vigabatrin tried;
       Pyridoxine trial: MANDATORY (rule out B6-responsive epilepsy) — SLC25A22 does NOT respond;
       Prognosis: uniformly severe; no patient achieves normal development;
     seed SEED_BASE+7)."""

import random, hashlib, json
from typing import List, Dict, Any

ATLAS_ID  = "hereditary-amino-acid-transport-disorder-atlas"
SEED_BASE = 4254
GENES     = ["SLC6A19", "SLC3A1", "SLC7A9", "SLC7A7", "SLC17A5", "SLC6A8", "SLC6A5", "SLC25A22"]

RENAL_INTESTINAL_GROUP  = ["SLC6A19", "SLC3A1", "SLC7A9", "SLC7A7"]
LYSOSOMAL_TRANSPORT_GROUP = ["SLC17A5"]
CNS_TRANSPORT_GROUP     = ["SLC6A8", "SLC6A5", "SLC25A22"]

GENE_META = {
    "SLC6A19":  {"aa": 634,  "locus": "5p15.33",  "inh": "AR",   "size_kda": 72,  "mechanism": "B0AT1-neutral-AA-intestine-kidney-tryptophan-malabsorption-pellagra"},
    "SLC3A1":   {"aa": 685,  "locus": "2p21",      "inh": "AR",   "size_kda": 79,  "mechanism": "rBAT-dibasic-AA-cystine-renal-reabsorption-cystinuria-TypeA"},
    "SLC7A9":   {"aa": 487,  "locus": "19q13.11",  "inh": "AR",   "size_kda": 55,  "mechanism": "b0+AT-light-chain-cystinuria-TypeB-heterozygote-intermediate-aminoaciduria"},
    "SLC7A7":   {"aa": 511,  "locus": "14q11.2",   "inh": "AR",   "size_kda": 57,  "mechanism": "y+LAT1-dibasic-AA-efflux-LPI-hyperammonemia-PAP-citrulline-responsive"},
    "SLC17A5":  {"aa": 495,  "locus": "6q13",      "inh": "AR",   "size_kda": 54,  "mechanism": "sialin-lysosomal-free-sialic-acid-ISSD-Salla-Finnish-founder"},
    "SLC6A8":   {"aa": 636,  "locus": "Xq28",      "inh": "XLR",  "size_kda": 70,  "mechanism": "CrT1-brain-creatine-transporter-absent-brain-creatine-MRS-ID-XLR"},
    "SLC6A5":   {"aa": 799,  "locus": "11p15.1",   "inh": "AR",   "size_kda": 87,  "mechanism": "GlyT2-presynaptic-glycine-reuptake-hyperekplexia-nonhabituating-startle"},
    "SLC25A22": {"aa": 323,  "locus": "11p15.5",   "inh": "AR",   "size_kda": 33,  "mechanism": "GC1-mitochondrial-glutamate-carrier-neonatal-myoclonic-EE-hypsarrhythmia"},
}

PATHOGNOMONIC = {
    "SLC6A19":  "Pellagra-like photosensitive rash + intermittent cerebellar ataxia + URINE NEUTRAL AMINOACIDURIA PATHOGNOMONIC; nicotinamide supplement CURATIVE; incomplete penetrance",
    "SLC3A1":   "CYSTINE UROLITHIASIS PATHOGNOMONIC; urine dibasic aminoaciduria (cystine+lysine+arginine+ornithine); cyanide-nitroprusside test positive; tiopronin chelation; Type A heterozygotes ASYMPTOMATIC",
    "SLC7A9":   "Identical cystinuria phenotype to SLC3A1; KEY DISTINCTION: heterozygous SLC7A9 may show intermediate aminoaciduria and stones (incomplete dominance); Type B carrier not fully asymptomatic",
    "SLC7A7":   "ALVEOLAR PROTEINOSIS (PAP) in 60-70% PATHOGNOMONIC for LPI; post-protein hyperammonemia; urine dibasic AAs HIGH, plasma LOW; HLH risk; citrulline supplement first-line",
    "SLC17A5":  "URINE FREE SIALIC ACID 100-1000x elevated PATHOGNOMONIC; Finnish founder p.R39C = Salla (mild); severe mutations = ISSD (lethal infancy); white matter dysmyelination MRI",
    "SLC6A8":   "ABSENT BRAIN CREATINE at 3.03ppm on 1H-MRS PATHOGNOMONIC; urine creatine/creatinine ratio elevated; XLR — males severely affected; creatine supplement DOES NOT correct brain MRS",
    "SLC6A5":   "NON-HABITUATING EXAGGERATED STARTLE REFLEX PATHOGNOMONIC; neonatal hypertonia; nose-tap reflex; apnea from tonic spasms; EEG NORMAL (not epilepsy); clonazepam EFFECTIVE; Heimlich LIFE-SAVING",
    "SLC25A22": "NEONATAL MYOCLONIC EPILEPSY + BURST-SUPPRESSION → HYPSARRHYTHMIA TRANSITION PATHOGNOMONIC; onset day 0-3; uniformly refractory; no organomegaly; ketogenic diet partial response",
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _make_cohort(gene: str, seed: int) -> Dict[str, Any]:
    rng = _rng(seed)
    n = 40
    gene_meta = GENE_META[gene]

    if gene in ("SLC6A19",):
        pathogenic_frac = rng.uniform(0.75, 0.88)   # incomplete penetrance
    elif gene in ("SLC3A1", "SLC7A9"):
        pathogenic_frac = rng.uniform(0.82, 0.93)
    elif gene in ("SLC7A7", "SLC17A5"):
        pathogenic_frac = rng.uniform(0.85, 0.95)
    elif gene in ("SLC6A8",):
        pathogenic_frac = rng.uniform(0.88, 0.96)   # XLR — clear in hemizygous males
    elif gene in ("SLC6A5",):
        pathogenic_frac = rng.uniform(0.80, 0.93)
    else:  # SLC25A22
        pathogenic_frac = rng.uniform(0.86, 0.95)

    pathogenic = int(n * pathogenic_frac)
    vus = n - pathogenic

    # Treatable genes
    if gene == "SLC6A19":
        treatable = True    # nicotinamide curative
    elif gene in ("SLC3A1", "SLC7A9"):
        treatable = True    # tiopronin + hydration
    elif gene == "SLC7A7":
        treatable = True    # citrulline supplement, low protein
    elif gene == "SLC17A5":
        treatable = False   # no disease-modifying therapy
    elif gene == "SLC6A8":
        treatable = False   # creatine does not cross BBB
    elif gene == "SLC6A5":
        treatable = True    # clonazepam effective
    else:  # SLC25A22
        treatable = False   # largely refractory

    # Neurological involvement
    if gene == "SLC6A19":
        neurological_frac = rng.uniform(0.20, 0.45)  # intermittent ataxia only in symptomatic
    elif gene in ("SLC3A1", "SLC7A9"):
        neurological_frac = 0.0   # pure renal/metabolic
    elif gene == "SLC7A7":
        neurological_frac = rng.uniform(0.55, 0.80)  # encephalopathy risk
    elif gene == "SLC17A5":
        neurological_frac = rng.uniform(0.90, 0.99)  # profound ND
    elif gene == "SLC6A8":
        neurological_frac = rng.uniform(0.90, 0.99)  # ID universal in males
    elif gene == "SLC6A5":
        neurological_frac = rng.uniform(0.40, 0.65)  # hyperekplexia + possible ID
    else:  # SLC25A22
        neurological_frac = rng.uniform(0.95, 1.00)  # severe universal

    return {
        "gene":              gene,
        "n":                 n,
        "pathogenic_n":      pathogenic,
        "vus_n":             vus,
        "locus":             gene_meta["locus"],
        "inheritance":       gene_meta["inh"],
        "aa":                gene_meta["aa"],
        "size_kda":          gene_meta["size_kda"],
        "mechanism_tag":     gene_meta["mechanism"],
        "pathognomonic":     PATHOGNOMONIC[gene],
        "treatable":         treatable,
        "neurological_frac": neurological_frac,
    }


def generate_overview() -> Dict[str, Any]:
    cohorts = []
    for i, gene in enumerate(GENES):
        cohorts.append(_make_cohort(gene, SEED_BASE + i))

    summary_table = []
    for c in cohorts:
        summary_table.append({
            "gene":          c["gene"],
            "locus":         c["locus"],
            "aa":            c["aa"],
            "kda":           c["size_kda"],
            "inheritance":   c["inheritance"],
            "mechanism":     c["mechanism_tag"],
            "pathognomonic": c["pathognomonic"],
            "n":             c["n"],
            "treatable":     c["treatable"],
        })

    transport_groups = {
        "renal_intestinal": {
            "genes":       ["SLC6A19", "SLC3A1", "SLC7A9", "SLC7A7"],
            "description": "Intestinal + renal tubular amino acid transport defects — urine amino acid chromatography diagnostic",
            "notes":       "CRITICAL: SLC3A1/SLC7A9 both cause cystinuria (dibasic AAs); SLC6A19 causes neutral aminoaciduria; SLC7A7 dibasic with lung complications",
        },
        "lysosomal": {
            "genes":       ["SLC17A5"],
            "description": "Lysosomal free sialic acid transport defect — urine free sialic acid 100-1000x elevated",
            "notes":       "ISSD (severe neonatal) vs Salla disease (mild, Finnish) — same gene different alleles",
        },
        "cns_transport": {
            "genes":       ["SLC6A8", "SLC6A5", "SLC25A22"],
            "description": "CNS-specific transport defects — brain MRS (SLC6A8), startle reflex (SLC6A5), neonatal EEG (SLC25A22)",
            "notes":       "SLC6A8: X-linked; brain MRS absent creatine PATHOGNOMONIC; SLC6A5: hyperekplexia, not epilepsy; SLC25A22: neonatal burst-suppression",
        },
    }

    total_patients = sum(c["n"] for c in cohorts)
    treatable_genes = [c["gene"] for c in cohorts if c["treatable"]]

    return {
        "atlas_id":          ATLAS_ID,
        "title":             "Hereditary-Amino-Acid-Transport-Disorder-Atlas — 8-Gene Reference",
        "genes":             GENES,
        "total_patients":    total_patients,
        "summary_table":     summary_table,
        "transport_groups":  transport_groups,
        "treatable_genes":   treatable_genes,
        "seeds":             f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
        "diagnostic_triage": (
            "STEP 1: Urine amino acid chromatography — first-line triage. "
            "Neutral aminoaciduria (Ala/Ser/Thr/Gln/Asn/Trp/Val/Leu/Ile/Phe/Tyr) → SLC6A19 (Hartnup). "
            "Dibasic aminoaciduria (Cys/Lys/Arg/Orn elevated) → SLC3A1 or SLC7A9 (cystinuria). "
            "Dibasic elevated IN URINE but LOW in plasma → SLC7A7 (LPI; check PAP, NH3). "
            "Urine free sialic acid markedly elevated → SLC17A5 (Salla/ISSD). "
            "STEP 2 (CNS): Brain MRS 1H absent creatine → SLC6A8 (creatine transporter; check sex-linked). "
            "Neonatal non-habituating startle + normal EEG → SLC6A5 (hyperekplexia; clonazepam urgently). "
            "Neonatal myoclonus + burst-suppression EEG → SLC25A22 (ketogenic diet trial)."
        ),
    }


def generate_breakdown() -> Dict[str, Any]:
    cohorts = []
    for i, gene in enumerate(GENES):
        cohorts.append(_make_cohort(gene, SEED_BASE + i))

    gene_profiles = []
    for c in cohorts:
        rng = _rng(SEED_BASE + GENES.index(c["gene"]))
        gene = c["gene"]

        if gene == "SLC6A19":
            treatment_response = {
                "nicotinamide": "50-100mg/day CURATIVE for rash and ataxia; high-protein diet; sun protection",
                "dietary":      "High-protein diet to compensate intestinal neutral AA malabsorption",
                "monitoring":   "Urine amino acids annually; nutritional status; tryptophan levels",
            }
            key_labs = "Urine AA: neutral aminoaciduria (ALL monoamino-monocarboxylic AAs elevated); plasma AA: normal or low neutral AAs"
            profile_extra = {
                "symptomatic_penetrance_pct": round(rng.uniform(0.20, 0.50) * 100, 1),
                "rash_pct": round(rng.uniform(0.55, 0.85) * 100, 1),
                "ataxia_pct": round(rng.uniform(0.30, 0.60) * 100, 1),
            }
        elif gene == "SLC3A1":
            treatment_response = {
                "hydration":        "3+ L/day fluid intake first-line; urine SG <1.010",
                "alkalinization":   "Potassium citrate to maintain urine pH >7.5; cystine solubility pH-dependent",
                "tiopronin":        "THIOLA: chelation → mixed disulfide; preferred over D-penicillamine",
                "urologic":         "Lithotripsy / ureteroscopy for established stones; urological follow-up",
            }
            key_labs = "Urine AA: cystine + lysine + arginine + ornithine elevated; cyanide-nitroprusside test positive; stone infrared spectroscopy cystine"
            profile_extra = {
                "nephrolithiasis_pct": round(rng.uniform(0.80, 0.98) * 100, 1),
                "staghorn_pct": round(rng.uniform(0.25, 0.55) * 100, 1),
                "ckd_risk_pct": round(rng.uniform(0.20, 0.40) * 100, 1),
            }
        elif gene == "SLC7A9":
            treatment_response = {
                "hydration":        "Same as SLC3A1 — 3+ L/day",
                "alkalinization":   "Same as SLC3A1 — potassium citrate",
                "tiopronin":        "Same — chelation; consider in heterozygous carriers with stones",
            }
            key_labs = "Same as SLC3A1; check heterozygous carriers (SLC7A9 het may have intermediate aminoaciduria)"
            profile_extra = {
                "nephrolithiasis_pct": round(rng.uniform(0.78, 0.96) * 100, 1),
                "het_aminoaciduria_pct": round(rng.uniform(0.40, 0.70) * 100, 1),  # carrier effect
            }
        elif gene == "SLC7A7":
            treatment_response = {
                "citrulline":       "L-citrulline 100-200mg/kg/day — FIRST-LINE; bypasses y+LAT1; replenishes ornithine/arginine; reduces hyperammonemia",
                "low_protein":      "Protein restriction; avoid high-protein meals (trigger hyperammonemia)",
                "lysine":           "L-lysine supplement to prevent growth failure and osteoporosis",
                "hcq":              "Hydroxychloroquine for PAP (reduces macrophage activation)",
                "hlh_protocol":     "HLH-94/2004 cyclosporine + etoposide if HLH triggered",
            }
            key_labs = "Urine AA: lysine/arginine/ornithine elevated; plasma: LOW lysine/arginine/ornithine; plasma NH3 elevated post-protein; orotic acid elevated; CT chest for PAP"
            profile_extra = {
                "pap_pct": round(rng.uniform(0.55, 0.75) * 100, 1),
                "hlh_pct": round(rng.uniform(0.15, 0.30) * 100, 1),
                "osteoporosis_pct": round(rng.uniform(0.40, 0.65) * 100, 1),
            }
        elif gene == "SLC17A5":
            treatment_response = {
                "supportive":   "No disease-modifying therapy; physio, communication, AEDs",
                "salla":        "Mild variant (Finnish p.R39C): PT/OT; communication AAC; AED for epilepsy",
                "issd":         "Palliative care; severe — death infancy",
            }
            key_labs = "Urine free sialic acid (Neu5Ac): >10 mmol/mol creatinine (normal <1); fibroblast lysosomal free sialic acid; SLC17A5 sequencing (p.R39C = Salla)"
            profile_extra = {
                "issd_severe_pct": round(rng.uniform(0.20, 0.40) * 100, 1),  # proportion with severe ISSD
                "salla_mild_pct": round(rng.uniform(0.55, 0.75) * 100, 1),
                "epilepsy_pct": round(rng.uniform(0.40, 0.65) * 100, 1),
            }
        elif gene == "SLC6A8":
            treatment_response = {
                "creatine":     "Creatine monohydrate: increases URINE creatine but does NOT improve brain MRS — BBB impermeant",
                "experimental": "Cyclocreatine prodrugs (phase 2 trials); creatine ethyl ester (limited data)",
                "aed":          "Antiepileptics for seizure management",
                "behavioral":   "Applied behavior analysis, speech therapy for ID and behavioral issues",
            }
            key_labs = "1H-MRS brain: absent creatine peak at 3.03 ppm PATHOGNOMONIC; urine creatine/creatinine ratio markedly elevated; plasma creatine normal or elevated"
            profile_extra = {
                "severe_id_males_pct": round(rng.uniform(0.75, 0.92) * 100, 1),
                "epilepsy_pct": round(rng.uniform(0.45, 0.70) * 100, 1),
                "behavioral_asd_pct": round(rng.uniform(0.30, 0.55) * 100, 1),
                "female_affected_pct": round(rng.uniform(0.20, 0.45) * 100, 1),  # mildly affected female carriers
            }
        elif gene == "SLC6A5":
            treatment_response = {
                "clonazepam":   "FIRST-LINE; highly effective; reduces startle severity and apnea episodes; start urgently in neonates",
                "glycine":      "Glycine supplement: may increase synaptic glycine reserve; adjunctive",
                "heimlich":     "LIFE-SAVING: Heimlich maneuver for neonatal startle-induced apnea — parents MUST learn technique",
                "monitoring":   "Respiratory monitoring in neonatal period; apnea alarm; reduce startling stimuli in NICU",
            }
            key_labs = "Video-EEG: normal background, no epileptiform activity (excludes epilepsy); nose-tap reflex non-habituating; EMG: normal; SLC6A5 panel with GLRA1/GLRB hyperekplexia DDx"
            profile_extra = {
                "neonatal_apnea_pct": round(rng.uniform(0.45, 0.70) * 100, 1),
                "clonazepam_response_pct": round(rng.uniform(0.75, 0.92) * 100, 1),
                "id_pct": round(rng.uniform(0.25, 0.50) * 100, 1),
            }
        else:  # SLC25A22
            treatment_response = {
                "ketogenic_diet":   "Partial response in 30-50%; reduces seizure frequency; improves EEG in some",
                "pyridoxine_trial": "MANDATORY — 100mg IV to rule out PDE (SLC25A22 does not respond)",
                "aed":              "Phenobarbital first-line; clonazepam; vigabatrin; largely refractory",
                "prognosis":        "Uniformly severe — no patient achieves independent function",
            }
            key_labs = "EEG: burst-suppression → hypsarrhythmia (KEY transition); urine organic acids NORMAL; plasma AA: no diagnostic pattern; CSF amino acids: may show low glutamate"
            profile_extra = {
                "burst_suppression_onset_days": round(rng.uniform(0, 3), 0),
                "keto_partial_response_pct": round(rng.uniform(0.30, 0.50) * 100, 1),
                "hypsarrhythmia_pct": round(rng.uniform(0.70, 0.90) * 100, 1),
            }

        gene_profiles.append({
            "gene":             gene,
            "locus":            c["locus"],
            "aa":               c["aa"],
            "inheritance":      c["inheritance"],
            "n":                c["n"],
            "pathognomonic":    c["pathognomonic"],
            "treatable":        c["treatable"],
            "key_labs":         key_labs,
            "treatment":        treatment_response,
            "profile_extra":    profile_extra,
            "neurological_pct": round(c["neurological_frac"] * 100, 1),
        })

    return {
        "atlas_id":      ATLAS_ID,
        "gene_profiles": gene_profiles,
        "seeds":         f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",
    }


def generate_definitions() -> Dict[str, Any]:
    return {
        "atlas_id": ATLAS_ID,
        "seeds":    f"{SEED_BASE}-{SEED_BASE + len(GENES) - 1}",

        "primary_cascade_split": (
            "STEP 1: URINE AMINO ACID CHROMATOGRAPHY — first-line for all suspected AA transport defects. "
            "Neutral aminoaciduria (Trp/Ala/Ser/Thr elevated) → SLC6A19 (Hartnup); nicotinamide trial. "
            "Dibasic aminoaciduria (Cys/Lys/Arg/Orn elevated) → cystinuria panel (SLC3A1/SLC7A9). "
            "Dibasic AA elevated urine but LOW plasma + NH3 elevated → SLC7A7 (LPI); check PAP on CT chest. "
            "Urine free sialic acid markedly elevated → SLC17A5 (ISSD/Salla). "
            "STEP 2 — CNS-SPECIFIC: Brain 1H-MRS absent creatine peak → SLC6A8 (XLR — check sex). "
            "Neonatal hypertonia + non-habituating startle + normal EEG → SLC6A5; clonazepam URGENTLY. "
            "Neonatal myoclonus + burst-suppression → SLC25A22; pyridoxine 100mg IV first, then KD."
        ),

        "key_distinguishers": [
            {
                "feature":      "Neutral aminoaciduria vs dibasic aminoaciduria",
                "significance": "Fundamental triage split on urine AA chromatography",
                "neutral_aa":   "SLC6A19 (Hartnup): Trp, Ala, Ser, Thr, Gln, Asn, Val, Leu, Ile, Phe, Tyr ALL elevated",
                "dibasic_aa":   "SLC3A1/SLC7A9 (cystinuria): cystine, Lys, Arg, Orn elevated",
                "dibasic_lpi":  "SLC7A7 (LPI): dibasic AA HIGH in urine, paradoxically LOW in plasma (intracellular trap)",
            },
            {
                "feature":      "SLC3A1 vs SLC7A9 cystinuria distinction",
                "significance": "Same phenotype but carrier status differs clinically",
                "type_a_slc3a1": "SLC3A1 heterozygotes: COMPLETELY ASYMPTOMATIC; purely autosomal recessive",
                "type_b_slc7a9": "SLC7A9 heterozygotes: may show INTERMEDIATE aminoaciduria; ~10-20% develop stones (incomplete dominance)",
                "clinical_implication": "Carrier testing: SLC7A9 carriers may need urological follow-up; SLC3A1 carriers do not",
            },
            {
                "feature":      "LPI pulmonary alveolar proteinosis (PAP) — PATHOGNOMONIC",
                "significance": "Separates LPI from cystinuria and Hartnup; defines pulmonary surveillance mandatory",
                "prevalence":   "60-70% of LPI patients develop PAP; may precede other features",
                "mechanism":    "Macrophage dysfunction from dibasic AA imbalance → surfactant accumulation",
                "management":   "Hydroxychloroquine for PAP; BAL if severe; HLH watch (20% risk)",
                "workup":       "CT chest MANDATORY at diagnosis; annual pulmonary function tests",
            },
            {
                "feature":      "Salla vs ISSD — same SLC17A5 gene, different severity",
                "significance": "Genotype determines prognosis: Finnish founder p.R39C = Salla (mild); severe mutations = ISSD (lethal)",
                "salla":        "p.R39C homozygous: mild ID, ambulatory, adult survival; Finnish population founder",
                "issd":         "Severe LOF mutations (p.R39C absent): hydrops, lethal infancy; profound multisystem",
                "diagnosis":    "Urine free sialic acid elevated in BOTH; genotype determines phenotype prediction",
            },
            {
                "feature":      "SLC6A8 creatine supplement DOES NOT WORK — critical clinical trap",
                "significance": "Creatine supplement widely prescribed; for CTD it is ineffective for brain",
                "mechanism":    "CrT1 absent → creatine cannot enter brain regardless of how much is given peripherally",
                "urine_result": "Creatine supplement INCREASES urine creatine further (no brain uptake); MRS unchanged",
                "what_works":   "No approved therapy; prodrug trials ongoing; gene therapy investigational",
                "clinical_tip": "Brain MRS showing absent creatine on creatine supplement confirms SLC6A8 (GAMT/AGAT deficiency would respond)",
            },
            {
                "feature":      "Hyperekplexia vs neonatal epilepsy — critical distinction (SLC6A5)",
                "significance": "Misdiagnosis as epilepsy leads to antiepileptic drugs that don't work; clonazepam is correct",
                "hyperekplexia": "EEG NORMAL; non-habituating startle (>5 stimuli, still full response); whole-body extension-abduction",
                "neonatal_seizures": "EEG shows ictal discharges; epileptiform activity; habituates normally",
                "test":         "NOSE TAP TEST: tap tip of nose → full body startle (non-habituating) = hyperekplexia",
                "treatment":    "Clonazepam immediate — reduces apnea risk; NOT regular antiepileptics",
                "emergency":    "HEIMLICH maneuver for apnea episode — parents must demonstrate competency at discharge",
            },
            {
                "feature":      "SLC25A22 vs GLDC non-ketotic hyperglycinemia (NKH) — neonatal myoclonus DDx",
                "significance": "Both present neonatal myoclonus; different metabolic profile and management",
                "slc25a22":     "CSF amino acids NORMAL (glutamate may be low); organic acids NORMAL; burst-suppression → hypsarrhythmia",
                "nkh_gldc":     "CSF glycine MARKEDLY elevated; CSF/plasma glycine ratio >0.08 PATHOGNOMONIC; organic acids NORMAL",
                "management":   "GLDC-NKH: sodium benzoate + dextromethorphan; SLC25A22: ketogenic diet",
                "shared":       "Both require pyridoxine 100mg IV urgently to exclude PDE",
            },
        ],

        "treatment_matrix": [
            {
                "intervention": "Nicotinamide (niacinamide) — SLC6A19 Hartnup",
                "gene":         "SLC6A19",
                "dose":         "50-100 mg/day",
                "mechanism":    "Replenishes NAD+ pathway bypassing intestinal tryptophan absorption",
                "response":     "CURATIVE for rash and ataxia; sun protection needed alongside",
                "evidence":     "Well-established clinical response; incompletely penetrant — only symptomatic patients need treatment",
            },
            {
                "intervention": "Tiopronin (THIOLA) + alkalinization — Cystinuria SLC3A1/SLC7A9",
                "gene":         "SLC3A1 + SLC7A9",
                "dose":         "Tiopronin 100-300 mg/day; potassium citrate to pH >7.5; 3L+/day fluid",
                "mechanism":    "Tiopronin forms mixed disulfide with cysteine (soluble); alkalinization increases solubility",
                "response":     "Reduces stone formation; does not cure aminoaciduria",
                "evidence":     "Urological monitoring + genetic counseling for carriers",
            },
            {
                "intervention": "L-Citrulline supplement — LPI SLC7A7",
                "gene":         "SLC7A7",
                "dose":         "100-200 mg/kg/day in 4 doses; adjust by ammonia levels",
                "mechanism":    "Citrulline bypasses y+LAT1 → enters cells directly → replenishes ornithine for urea cycle",
                "response":     "SIGNIFICANTLY reduces hyperammonemia; PAP slower to respond",
                "evidence":     "Standard of care; lysine also needed to prevent growth failure",
            },
            {
                "intervention": "Clonazepam — Hyperekplexia SLC6A5",
                "gene":         "SLC6A5",
                "dose":         "0.1-0.3 mg/kg/day; titrate to effect",
                "mechanism":    "GABA-A potentiation → increases chloride conductance → inhibitory tone supplemented",
                "response":     "HIGHLY EFFECTIVE for startle severity and apnea episodes; first-line",
                "evidence":     "Neonatal hyperekplexia is a medical emergency — start within hours of diagnosis",
            },
            {
                "intervention": "Ketogenic diet — SLC25A22 mitochondrial glutamate carrier",
                "gene":         "SLC25A22",
                "dose":         "Classic 4:1 ratio; MCT oil also used",
                "mechanism":    "Ketone bodies provide alternative anaplerotic substrate to mitochondria bypassing GC1",
                "response":     "Partial in 30-50%; reduces seizure frequency; does not restore development",
                "evidence":     "Best available option given refractory nature; earlier start correlates with better partial response",
            },
        ],

        "urine_aa_cascade": {
            "neutral":         "SLC6A19 — Trp/Ala/Ser/Thr/Gln/Val/Leu/Ile/Phe/Tyr all elevated; dibasic NORMAL",
            "dibasic_renal":   "SLC3A1 or SLC7A9 — Cys/Lys/Arg/Orn elevated; cyanide-nitroprusside positive",
            "dibasic_lpi":     "SLC7A7 — dibasic HIGH urine + LOW plasma → efflux defect + NH3 elevated",
            "free_sialic":     "SLC17A5 — urine free sialic acid 100-1000x elevated (must request specifically)",
            "creatine_ratio":  "SLC6A8 — urine creatine/creatinine elevated; brain MRS confirms absent peak",
        },
    }
