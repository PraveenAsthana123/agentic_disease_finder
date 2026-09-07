#!/usr/bin/env python3
"""Hereditary-Mitochondrial-RC-Atlas — Complete 8-Gene Hereditary Mitochondrial Respiratory Chain Atlas
POLG    (Alpers-Huttenlocher / SANDO / PEO3; 1240 aa; 15q25.1; AR;
         mtDNA Polymerase gamma — replication + repair of mitochondrial DNA;
         VPA ABSOLUTE CI — depletes residual mtDNA;
         CSF lactate elevated; mtDNA depletion in liver/muscle;
         seed SEED_BASE+0) .
TWNK    (Progressive External Ophthalmoplegia 1 / IOSCA; 684 aa; 10q24.31; AD/AR;
         Twinkle mtDNA helicase — unwinds mtDNA ahead of polymerase;
         AD: ptosis + ophthalmoplegia + multiple mtDNA deletions in muscle;
         AR: infantile-onset spinocerebellar ataxia (IOSCA);
         seed SEED_BASE+1) .
SURF1   (Leigh Syndrome — CIV subtype; 309 aa; 9q34.2; AR;
         COX (Cytochrome c Oxidase) assembly factor — inserts copper and haem a3;
         MOST COMMON nuclear cause of Leigh syndrome in Europeans;
         COX-deficient ragged red fibres ABSENT — assembly factor not structural;
         seed SEED_BASE+2) .
BCS1L   (GRACILE Syndrome / Björnstad Syndrome; 419 aa; 2q35; AR;
         Complex III (CIII) assembly factor — inserts Rieske iron-sulphur subunit;
         GRACILE = Growth restriction Aminoaciduria Cholestasis Iron overload Lactic acidosis Early death;
         p.Ser78Gly Finno-Ugric founder; Björnstad = pili torti + SNHL milder;
         seed SEED_BASE+3) .
SCO2    (Fatal Infantile Cardioencephalomyopathy / CIV-SCO2; 266 aa; 22q13.33; AR;
         CIV copper scaffold — delivers copper to COX2 subunit;
         hypertrophic cardiomyopathy + lactic acidosis + hypotonia TRIAD;
         copper supplementation emerging; p.Glu140Lys most common variant;
         seed SEED_BASE+4) .
PDHA1   (Pyruvate Dehydrogenase Deficiency; 390 aa; Xp22.12; X-linked;
         PDH E1-alpha subunit — thiamine-dependent pyruvate → acetyl-CoA;
         KETOGENIC DIET THERAPEUTIC (unique — bypasses pyruvate via fat);
         thiamine B1 trial MANDATORY — PDH is B1-dependent;
         seed SEED_BASE+5) .
ACAD9   (Complex I Deficiency — ACAD9; 621 aa; 3q21.3; AR;
         CI assembly factor — dual role as acyl-CoA dehydrogenase AND CI assembly;
         RIBOFLAVIN B2 RESPONSIVE ~50% — CRITICAL treatment;
         exercise intolerance + lactic acidosis + cardiomyopathy;
         seed SEED_BASE+6) .
NDUFS1  (Complex I Deficiency — Core Subunit; 727 aa; 2q33.3; AR;
         NADH:ubiquinone reductase core subunit — Fe-S cluster N5/N6a/N6b in 75kDa subunit;
         most common NUCLEAR-encoded CI gene causing Leigh/leukoencephalopathy;
         BN-PAGE CI band absent; brain MRI T2-FLAIR white matter + basal ganglia;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1846–1853)
"""

import random

SEED_BASE = 1846

MITO_GENES = [
    # -- POLG -- DNA Polymerase Gamma -------------------------------------------
    {
        "gene": "POLG",
        "protein": (
            "POLG -- 15q25.1 AR -- mtDNA-Polymerase-Gamma-1240aa -- "
            "Alpers-Huttenlocher-SANDO-PEO3-Myocerebrohepatopathy -- "
            "VPA-ABSOLUTE-CI-Depletes-Residual-mtDNA-Hepatic-Failure -- "
            "mtDNA-Depletion-Liver-Muscle-PATHOGNOMONIC -- "
            "CSF-Lactate-Elevated-Brain-Occipital-T2-Lesions -- "
            "p.Ala467Thr-Most-Common-European-Variant"
        ),
        "alias": (
            "POLG (DNA polymerase gamma, catalytic subunit); OMIM gene 174763. "
            "Alpers-Huttenlocher syndrome OMIM 203700; SANDO OMIM 607459; PEO3 OMIM 609286. "
            "15q25.1; 1240 aa; ~140 kDa; mitochondrial matrix; forms heterotrimer with "
            "two POLG2 (accessory) subunits; autosomal recessive (biallelic) for severe phenotypes, "
            "autosomal dominant (monoallelic) for milder adult PEO. "
            "FUNCTION: POLG is the sole DNA polymerase responsible for mitochondrial DNA "
            "(mtDNA) replication and repair. mtDNA encodes 13 OXPHOS subunits + 22 tRNAs + 2 rRNAs. "
            "Without functional POLG: mtDNA cannot be replicated → mtDNA DEPLETION "
            "(reduced copy number per cell) or accumulation of somatic mtDNA mutations/deletions. "
            "PHENOTYPIC SPECTRUM (genotype determines severity): "
            "1. ALPERS-HUTTENLOCHER SYNDROME (severe biallelic, childhood onset): "
            "triad — intractable seizures (refractory status epilepticus) + "
            "progressive hepatic failure (ultimately fatal) + "
            "psychomotor regression; "
            "occipital lobe involvement on MRI (T2 high signal posterior cortex); "
            "seizures often first presenting feature in infancy-early childhood; "
            "hepatic failure: may be precipitated by VPA → ABSOLUTE CONTRAINDICATION; "
            "death in 2-10 years typically; "
            "2. SANDO (Sensory Ataxic Neuropathy, Dysarthria, and Ophthalmoplegia): "
            "adult onset (3rd-5th decade); "
            "sensory axonal neuropathy (ataxia from sensory loss) + "
            "cerebellar ataxia + dysarthria + ophthalmoplegia/ptosis; "
            "PNS and CNS both affected; "
            "3. MELAS/Leigh overlap: some POLG mutations cause Leigh syndrome or MELAS-like; "
            "4. PEO3 (Progressive External Ophthalmoplegia): "
            "adult-onset ptosis + ophthalmoplegia; milder; "
            "muscle biopsy: ragged red fibres (COX-negative), multiple mtDNA deletions. "
            "VPA ABSOLUTE CONTRAINDICATION: "
            "Valproate directly inhibits POLG catalytic activity; "
            "in POLG-deficient liver (already low mtDNA copy number): "
            "VPA precipitates fatal hepatic failure; "
            "mechanism: VPA + POLG inhibition → CATASTROPHIC mtDNA depletion in liver; "
            "even a single dose of VPA can be lethal in Alpers; "
            "VPA is also CI in any patient with unexplained hepatopathy + seizures "
            "(until POLG excluded); "
            "alternative AEDs: levetiracetam, phenobarbital, benzodiazepines. "
            "NUCLEOSIDE ANALOGUE ANTIRETROVIRALS (AZT, d4T, ddI): "
            "also inhibit POLG → worsen mtDNA depletion; "
            "avoid in POLG patients receiving HIV treatment; "
            "DIAGNOSIS: "
            "CSF lactate: elevated (often >3 mmol/L); "
            "plasma lactate; liver transaminases; "
            "muscle biopsy (adults): ragged red fibres, multiple mtDNA deletions on Southern blot/long-range PCR; "
            "liver biopsy (Alpers): mtDNA depletion quantification; "
            "POLG gene sequencing (exons 2-23, ~1400 bp coding); "
            "fibroblast or muscle mtDNA copy number. "
            "COMMON VARIANTS: "
            "p.Ala467Thr: MOST COMMON European POLG variant "
            "(~68% of SANDO alleles in Europeans; compound heterozygous with p.Trp748Ser); "
            "p.Trp748Ser: second most common; usually compound het with p.Ala467Thr; "
            "p.Gly848Ser: severe; often homozygous in Alpers. "
            "TREATMENT: "
            "No disease-modifying therapy established; "
            "mitochondrial cocktail: coenzyme Q10, riboflavin, thiamine, antioxidants "
            "(theoretical benefit, no RCT evidence for POLG); "
            "seizures: levetiracetam (first-line for POLG); benzodiazepines for status; "
            "AVOID: valproate, lamotrigine (may exacerbate), nucleoside analogues; "
            "liver transplant: does NOT correct extrahepatic disease; "
            "generally not recommended for Alpers (hepatic failure is terminal but neurological disease "
            "continues and worsens). "
            "KEY CLINICAL FACTS: "
            "Alpers = refractory seizures + hepatic failure in child = POLG until proven otherwise; "
            "VPA rule: ALL patients with unexplained seizures + liver disease → exclude POLG before VPA; "
            "SANDO: adult onset; sensory > cerebellar ataxia; check EMG (sensory axonal neuropathy); "
            "muscle biopsy: ragged red fibres + multiple mtDNA deletions diagnostic; "
            "liver transplant is not standard for Alpers (neurological disease continues)."
        ),
        "age_of_onset": "Alpers: 2-4 years; SANDO: 3rd-5th decade; PEO3: 4th+ decade",
        "inheritance": "AR (biallelic severe); AD (monoallelic PEO)",
        "locus": "15q25.1",
        "protein_size": "1240 aa",
        "key_biomarker": "CSF lactate >3 mmol/L; muscle mtDNA deletions (multiple); liver mtDNA depletion; POLG sequencing",
        "pathognomonic": "Refractory seizures + hepatic failure in child (Alpers); sensory ataxia + ophthalmoplegia in adult (SANDO)",
        "treatment": "VPA ABSOLUTE CI; levetiracetam for seizures; avoid nucleoside analogues; mitochondrial cocktail supportive",
        "critical_flags": [
            "VPA-ABSOLUTE-CI-POLG — depletes mtDNA in liver → catastrophic hepatic failure; even single dose lethal in Alpers",
            "NUCLEOSIDE-ANALOGUES-CI-AZT-D4T-DDI — also inhibit POLG → worsen mtDNA depletion",
            "REFRACTORY-SEIZURES-HEPATOPATHY-CHILD-EXCLUDE-POLG-BEFORE-VPA — clinical rule",
            "LIVER-TRANSPLANT-NOT-RECOMMENDED-ALPERS — neurological disease continues; hepatic only",
            "p.Ala467Thr-MOST-COMMON-EUROPEAN — 68% SANDO alleles; compound het with p.Trp748Ser",
            "MUSCLE-BIOPSY-MULTIPLE-mtDNA-DELETIONS — ragged red fibres COX-neg; adults diagnostic",
            "CSF-LACTATE->3mmol/L-CLASSIC — check in any child with refractory seizures + hepatopathy",
            "OCCIPITAL-T2-LESIONS-MRI-ALPERS — posterior cortex affected early; seizure focus",
        ],
    },
    # -- TWNK -- Twinkle Helicase -----------------------------------------------
    {
        "gene": "TWNK",
        "protein": (
            "TWNK -- 10q24.31 AD-AR -- mtDNA-Helicase-Twinkle-684aa -- "
            "PEO1-Ptosis-Ophthalmoplegia-Multiple-mtDNA-Deletions-Muscle-AD -- "
            "IOSCA-Infantile-Onset-Spinocerebellar-Ataxia-AR-Severe -- "
            "Ragged-Red-Fibres-Multiple-Deletions-Muscle-Biopsy -- "
            "Mitochondrial-DNA-Depletion-Syndrome-7-MNGIE-Overlap"
        ),
        "alias": (
            "TWNK (twinkle mtDNA helicase; also called C10orf2 or PEO1 gene); OMIM gene 606075. "
            "PEO1 OMIM 157640; IOSCA (infantile-onset spinocerebellar ataxia) OMIM 271245; "
            "MSCAE (mtDNA depletion syndrome type 7, hepatocerebral) OMIM 271245. "
            "10q24.31; 684 aa; ~75 kDa; mitochondrial matrix; forms hexameric ring; "
            "autosomal dominant (missense gain-of-dysfunction) or autosomal recessive (biallelic null). "
            "FUNCTION: TWNK (Twinkle helicase) unwinds the double-stranded mtDNA ahead of "
            "POLG (the replication polymerase) during mtDNA replication. "
            "Without functional Twinkle: mtDNA replication stalls → "
            "AD mutations (dominant negative): cause mtDNA replication errors → "
            "multiple somatic mtDNA DELETIONS accumulate over decades → "
            "RRFs (ragged red fibres) + COX-negative fibres in muscle; "
            "AR mutations (biallelic null): cause mtDNA DEPLETION in severe infantile forms. "
            "AUTOSOMAL DOMINANT PEO1: "
            "Most common presentation of TWNK mutations; "
            "adult onset (typically 3rd-5th decade); "
            "progressive bilateral ptosis (drooping eyelids) followed by "
            "ophthalmoplegia (restricted eye movements); "
            "mild proximal limb weakness; "
            "exercise intolerance, fatigue; "
            "generally slowly progressive; "
            "muscle biopsy: ragged red fibres + multiple mtDNA deletions (Southern blot/array CGH); "
            "SPINOCEREBELLAR ATAXIA NEUROPATHY (SCAN): "
            "some TWNK-AD patients also develop cerebellar ataxia + sensory neuropathy; "
            "similar to SANDO phenotype but from TWNK not POLG. "
            "AUTOSOMAL RECESSIVE IOSCA (Infantile Onset Spinocerebellar Ataxia): "
            "Finnish founder (TWNK p.Tyr508Cys); "
            "presents at 9-18 months with cerebellar ataxia; "
            "progressive: sensory neuropathy, ophthalmoplegia, hearing loss, "
            "cognitive decline, epilepsy (especially in females — hormone-dependent); "
            "severe form specific to Finnish population; "
            "mtDNA depletion in muscle (not deletions — different from AD form). "
            "AUTOSOMAL RECESSIVE MSCAE / Hepatocerebral depletion: "
            "severe infantile presentation; "
            "liver failure + encephalopathy; "
            "mtDNA depletion in liver; "
            "similar severity to POLG Alpers. "
            "DIAGNOSIS: "
            "Plasma/CSF lactate: elevated; "
            "muscle biopsy: RRFs, COX-negative fibres, multiple mtDNA deletions (AD); "
            "mtDNA quantification: depletion in AR severe forms; "
            "long-range PCR / Southern blot: multiple mtDNA deletions; "
            "TWNK sequencing; "
            "mitochondrial respiratory chain enzyme assay (CI, CIII, CIV combined deficiency from mtDNA deletions). "
            "TREATMENT: "
            "No disease-modifying therapy; "
            "supportive: CoQ10, riboflavin, antioxidants; "
            "ptosis: ptosis surgery (frontalis sling) when severe; "
            "AVOID: valproate (inhibits POLG in same pathway); "
            "exercise: low-impact encouraged; avoid intense anaerobic exercise; "
            "genetic counselling: AD PEO1 — 50% risk to offspring; AR IOSCA — 25% risk. "
            "KEY CLINICAL FACTS: "
            "PEO1 (TWNK) vs CPEO from POLG: both cause PEO + RRFs; "
            "TWNK-AD: age-related multiple deletions on muscle biopsy; "
            "POLG-AR: may be younger; also hepatic involvement; "
            "IOSCA (AR-TWNK): Finnish; infantile ataxia → progressive multisystem; "
            "distinguished from other spinocerebellar ataxias by mtDNA depletion + mitochondrial enzymology; "
            "hearing impairment develops later in IOSCA: audiological surveillance required."
        ),
        "age_of_onset": "PEO1 (AD): 3rd-5th decade; IOSCA (AR): 9-18 months",
        "inheritance": "AD (PEO1); AR (IOSCA/MSCAE)",
        "locus": "10q24.31",
        "protein_size": "684 aa",
        "key_biomarker": "Multiple mtDNA deletions on muscle biopsy (AD); mtDNA depletion (AR); elevated lactate",
        "pathognomonic": "PEO1: adult progressive bilateral ptosis + ophthalmoplegia + RRFs/mtDNA deletions in muscle",
        "treatment": "Supportive; avoid VPA; ptosis surgery; 50% offspring risk (AD counselling)",
        "critical_flags": [
            "VPA-AVOID-TWNK-SAME-PATHWAY-AS-POLG — not absolute CI like Alpers but avoid in all TWNK-mito disease",
            "AD-PEO1-50pct-OFFSPRING-RISK — autosomal dominant; genetic counselling mandatory",
            "MUSCLE-BIOPSY-MULTIPLE-DELETIONS-PATHOGNOMONIC-AD-FORM — long-range PCR/Southern blot",
            "AR-IOSCA-FINNISH-FOUNDER-p.Tyr508Cys — population-specific; distinguish from other infantile ataxias",
            "HEARING-SURVEILLANCE-MANDATORY-IOSCA — audiology at onset + annually",
            "RRF-COX-NEGATIVE-FIBRES-MUSCLE — shared with POLG, mtDNA maintenance gene defects",
            "COMBINED-OXPHOS-DEFICIENCY — CI+CIII+CIV all reduced from mtDNA deletion pool",
            "MSCAE-AR-FATAL-INFANTILE-LIVER — biallelic null → hepatocerebral depletion similar to Alpers",
        ],
    },
    # -- SURF1 -- Leigh Syndrome CIV Assembly ------------------------------------
    {
        "gene": "SURF1",
        "protein": (
            "SURF1 -- 9q34.2 AR -- COX-Assembly-Factor-309aa -- "
            "Leigh-Syndrome-CIV-Most-Common-European-Nuclear-Cause -- "
            "COX-Deficiency-Histochemistry-ABSENT-KEY-Diagnostic -- "
            "Brainstem-Bilaterally-Symmetric-T2-Lesions-PATHOGNOMONIC-Leigh -- "
            "NO-Ragged-Red-Fibres-ASSEMBLY-FACTOR-Not-Structural-Subunit -- "
            "p.845delCT-Most-Common-European-Founder"
        ),
        "alias": (
            "SURF1 (surfeit gene 1; COX assembly factor); OMIM gene 185620. "
            "Leigh syndrome due to COX deficiency OMIM 256000. "
            "9q34.2; 309 aa; ~35 kDa; inner mitochondrial membrane; autosomal recessive. "
            "FUNCTION: SURF1 is an assembly factor for cytochrome c oxidase (CIV / COX). "
            "SURF1 facilitates the early steps of COX assembly, specifically "
            "the insertion of haem a and haem a3 (the oxygen-binding cofactor) "
            "into the COX1 (MT-CO1) subunit. "
            "Without SURF1: COX assembly stalls → CIV enzyme markedly deficient → "
            "impaired oxidative phosphorylation → lactic acidosis + energy failure. "
            "LEIGH SYNDROME (subacute necrotising encephalopathy): "
            "bilateral, symmetrical, T2-FLAIR hyperintense lesions on brain MRI "
            "in: basal ganglia (putamen, caudate), thalamus, brainstem (periaqueductal grey, "
            "substantia nigra, inferior olivary nucleus, tegmentum); "
            "pathologically: foci of spongiosis, vascular proliferation, demyelination, "
            "neuronal loss in these regions; "
            "clinical: progressive psychomotor regression, "
            "hypotonia → hypertonia, bulbar dysfunction "
            "(feeding difficulties, respiratory problems), "
            "episodic deterioration triggered by intercurrent illness, "
            "nystagmus, ophthalmoplegia, ataxia; "
            "SURF1-Leigh: often slightly later onset and slower progression than mtDNA-encoded Leigh; "
            "EUROPEAN GENETICS: "
            "SURF1 is the MOST COMMON identifiable nuclear gene cause of Leigh syndrome "
            "in European populations; "
            "p.845delCT (c.845_846delCT): most common European pathogenic variant "
            "(truncating deletion causing protein loss); "
            "other common: p.Ala444Val, nonsense, splice-site variants. "
            "COX HISTOCHEMISTRY (MUSCLE/BRAIN BIOPSY): "
            "COX (CIV) staining: MARKEDLY REDUCED or absent — key diagnostic finding; "
            "SDH (succinate dehydrogenase, CII) staining: PRESERVED or increased (compensatory) — "
            "COX/SDH double staining: COX-negative fibres appear blue (SDH positive); "
            "CRITICAL POINT: "
            "SURF1 is an ASSEMBLY FACTOR (not a structural subunit of CIV) → "
            "RAGGED RED FIBRES (RRF) ARE ABSENT (RRF from mtDNA deletion/depletion, "
            "not nuclear assembly factor defects); "
            "this is a key distinguishing feature from POLG/TWNK (which can cause RRF "
            "through mtDNA-level damage). "
            "BIOCHEMISTRY: "
            "CIV (COX) enzyme activity: markedly reduced in muscle, fibroblasts, liver; "
            "CI, CII, CIII activities: usually normal or mildly reduced (isolated CIV defect); "
            "plasma and CSF lactate: elevated, often >4 mmol/L; "
            "lactate:pyruvate ratio: elevated (>20) — reflects OXPHOS block. "
            "DIAGNOSIS: "
            "MRI brain: bilateral symmetric T2-FLAIR lesions in basal ganglia/brainstem; "
            "muscle biopsy: COX-deficient fibres (COX/SDH staining); "
            "OXPHOS enzymology: isolated CIV deficiency; "
            "SURF1 gene sequencing. "
            "TREATMENT: "
            "No proven disease-modifying therapy; "
            "mitochondrial cocktail: CoQ10, riboflavin (B2), thiamine (B1), antioxidants; "
            "carbohydrate load worsens lactic acidosis → moderate fat-carb balance; "
            "avoid triggers of metabolic decompensation: "
            "illness, fasting, sedatives, high-dose sodium bicarbonate (paradoxical); "
            "VALPROATE AVOID (POLG inhibition risk; plus CI-to-OXPHOS enzymes generally); "
            "dichloroacetate (DCA): reduces lactic acidosis (theoretical, some trials); "
            "respiratory support: often needed in late disease; "
            "genetic counselling: AR 25% sibling risk. "
            "KEY CLINICAL FACTS: "
            "SURF1 = most common nuclear Leigh in Europeans; "
            "COX/SDH staining on muscle biopsy shows COX-negative fibres "
            "(RRFs absent — no mtDNA-level pathology); "
            "Leigh MRI: bilateral symmetric basal ganglia + brainstem T2 bright; "
            "SURF1-Leigh: slightly milder than mtDNA-encoded Leigh (e.g., MT-ATP6 m.8993T>G); "
            "intercurrent illness = greatest risk for acute decompensation (hospitalise early, IV glucose)."
        ),
        "age_of_onset": "Infancy to early childhood (usually 3 months to 2 years)",
        "inheritance": "AR",
        "locus": "9q34.2",
        "protein_size": "309 aa",
        "key_biomarker": "COX-deficient fibres on muscle biopsy; isolated CIV deficiency on OXPHOS enzymology; CSF/plasma lactate elevated",
        "pathognomonic": "Bilateral symmetric T2-bright basal ganglia/brainstem lesions (Leigh) + COX-deficient fibres + absent RRFs",
        "treatment": "No specific therapy; avoid VPA/fasting/intercurrent illness; IV glucose in crisis; CoQ10/riboflavin supportive",
        "critical_flags": [
            "COX-DEFICIENT-FIBRES-COX-SDH-STAINING-MANDATORY — blue fibres on double stain; diagnostic",
            "NO-RAGGED-RED-FIBRES-ASSEMBLY-FACTOR — SURF1 is assembly factor; RRF absent unlike mtDNA-mutation disease",
            "LEIGH-MRI-BILATERAL-SYMMETRIC-BG-BRAINSTEM-T2-BRIGHT — pathognomonic radiological pattern",
            "MOST-COMMON-NUCLEAR-LEIGH-EUROPE-SURF1 — p.845delCT most common European truncating variant",
            "INTERCURRENT-ILLNESS-METABOLIC-DECOMPENSATION-EMERGENCY-IV-GLUCOSE — hospitalise at first sign",
            "VALPROATE-AVOID-MITO-DISEASE — general OXPHOS CI; not CIV-specific but avoid all mito patients",
            "ISOLATED-CIV-DEFICIENCY — CI/CII/CIII relatively spared; CIV selectively reduced",
            "LACTATE-PYRUVATE-RATIO->20-OXPHOS-BLOCK — confirms mitochondrial (not cytoplasmic) lactic acidosis",
        ],
    },
    # -- BCS1L -- GRACILE Syndrome / Bjornstad ----------------------------------
    {
        "gene": "BCS1L",
        "protein": (
            "BCS1L -- 2q35 AR -- CIII-Assembly-AAA-Plus-ATPase-419aa -- "
            "GRACILE-Syndrome-Neonatal-Lethal-MOST-SEVERE-CIII-Defect -- "
            "Bjornstad-Syndrome-Pili-Torti-SNHL-Milder -- "
            "p.Ser78Gly-Finno-Ugric-Founder-GRACILE -- "
            "BN-PAGE-CIII-Pre-Complex-Accumulates-KEY-DDx-TTC19 -- "
            "Rieske-Fe-S-Cluster-Subunit-Insertion-Defect"
        ),
        "alias": (
            "BCS1L (BCS1 homolog, ubiquinol-cytochrome c reductase complex chaperone); OMIM gene 603647. "
            "GRACILE syndrome OMIM 603358; Björnstad syndrome OMIM 262000. "
            "2q35; 419 aa; ~48 kDa; inner mitochondrial membrane; "
            "forms homohexameric ring; AAA+ ATPase family; autosomal recessive. "
            "FUNCTION: BCS1L inserts the Rieske iron-sulphur protein (UQCRFS1/RISP) "
            "into the pre-assembled Complex III (CIII) core. "
            "RISP is the last subunit to be incorporated — BCS1L uses ATP hydrolysis "
            "to displace a placeholder protein and insert RISP with its 2Fe-2S cluster. "
            "Without BCS1L: RISP cannot be inserted → "
            "CIII-pre-complex accumulates (contains all subunits EXCEPT RISP) → "
            "catalytically dead CIII-pre-complex on BN-PAGE (a diagnostic signature distinguishing "
            "BCS1L-CIII deficiency from TTC19-CIII deficiency, where the pre-complex is absent). "
            "GRACILE SYNDROME (most severe BCS1L allele — p.Ser78Gly): "
            "G — Growth restriction (intrauterine and postnatal); "
            "R — Renal tubular dysfunction (aminoaciduria, glycosuria, phosphaturia — Fanconi); "
            "A — Aminoaciduria (from Fanconi); "
            "C — Cholestasis (hepatocellular and canalicular cholestasis); "
            "I — Iron overload (paradoxical siderosis in liver/heart/endocrine); "
            "L — Lactic acidosis (severe, early, neonatal); "
            "E — Early death (neonatal to infantile; rarely survive beyond 4-6 months); "
            "caused by homozygous p.Ser78Gly (c.232A>G), which is a Finno-Ugric founder variant "
            "(Finnish/Estonian/Sámi population; carrier frequency ~1:80 in Finland). "
            "BJÖRNSTAD SYNDROME (milder BCS1L alleles): "
            "Pili torti (twisted brittle hair — DDx Menkes! but BCS1L NOT copper-related); "
            "sensorineural hearing loss (bilateral, usually moderate-to-severe); "
            "NO neurological or metabolic crises in classic Björnstad; "
            "CIII deficiency biochemically demonstrable but clinically mild; "
            "important DDx: Menkes (ATP7A) also causes pili torti but has low copper + "
            "neurodegeneration + connective tissue — distinguish by copper studies. "
            "BN-PAGE HALLMARK: "
            "Blue native PAGE (BN-PAGE) of isolated mitochondria or fibroblasts: "
            "CIII pre-complex ACCUMULATES (band lower than mature CIII); "
            "mature CIII absent or markedly reduced; "
            "CONTRAST WITH TTC19 (another CIII assembly factor): "
            "TTC19 deficiency: pre-complex does NOT accumulate (absent) — key DDx on BN-PAGE; "
            "BIOCHEMISTRY: "
            "CIII (ubiquinol:cytochrome c reductase) enzyme activity: markedly reduced in muscle, fibroblasts; "
            "CI activity may also be secondarily reduced (CI-CIII supercomplex); "
            "CIV (COX) activity: usually normal. "
            "DIAGNOSIS: "
            "Neonatal lactic acidosis (GRACILE) + Fanconi + cholestasis + siderosis → BCS1L; "
            "pili torti + SNHL (Björnstad) → BCS1L (check copper to exclude Menkes); "
            "BN-PAGE: CIII pre-complex accumulation; "
            "CIII enzymology; fibroblast or liver mitochondrial function; "
            "BCS1L sequencing. "
            "TREATMENT: "
            "GRACILE: intensive supportive (NICU); acidosis management; iron chelation; "
            "no disease-modifying treatment established; prognosis very poor; "
            "Björnstad: hearing aids; cochlear implant if profound SNHL; hair cosmetic care; "
            "coenzyme Q10 (theoretical CIII electron carrier supplement); "
            "no CIII-specific pharmacotherapy approved. "
            "KEY CLINICAL FACTS: "
            "GRACILE = neonatal lactic acidosis + Fanconi + cholestasis + siderosis → BCS1L; "
            "siderosis (iron overload) in GRACILE is NOT from iron ingestion but from impaired "
            "mitochondrial iron utilization + haem synthesis defect; "
            "Björnstad: milder BCS1L; pili torti + SNHL; distinguish from Menkes by copper studies; "
            "BN-PAGE pre-complex = BCS1L; no pre-complex = TTC19."
        ),
        "age_of_onset": "GRACILE: neonatal (lethal neonatal/infantile); Björnstad: childhood",
        "inheritance": "AR",
        "locus": "2q35",
        "protein_size": "419 aa",
        "key_biomarker": "BN-PAGE CIII pre-complex accumulation; CIII enzymology deficiency; neonatal lactic acidosis + Fanconi (GRACILE)",
        "pathognomonic": "GRACILE triad (growth restrict+Fanconi+cholestasis+siderosis+lactic acidosis, neonatal lethal); pili torti+SNHL (Björnstad)",
        "treatment": "GRACILE: supportive NICU; iron chelation; Björnstad: hearing aids/CI; CoQ10; no curative therapy",
        "critical_flags": [
            "GRACILE-p.Ser78Gly-FINNO-UGRIC-FOUNDER — Finnish/Estonian; carrier 1:80 Finland; neonatal lethal",
            "BN-PAGE-CIII-PRECOMPLEX-ACCUMULATES-DDx-TTC19 — pre-complex band = BCS1L; absent = TTC19",
            "SIDEROSIS-PARADOX-NOT-IRON-INGESTION — iron accumulation from CIII/haem synthesis defect",
            "PILI-TORTI-SNHL-BJORNSTAD-DDx-MENKES — check copper (Cu normal in BCS1L; low in ATP7A/Menkes)",
            "CIII-PRECOMPLEX-DEAD-ENZYMATICALLY — catalytically inactive; cannot transfer electrons",
            "FANCONI-RENAL-TUBULAR-DYSFUNCTION-GRACILE — aminoaciduria + glycosuria + phosphaturia",
            "CHOLESTASIS-NEONATAL-GRACILE — canalicular + hepatocellular; may resemble biliary atresia",
            "CI-ALSO-REDUCED-SECONDARY — CI-CIII supercomplex disrupted by absent functional CIII",
        ],
    },
    # -- SCO2 -- Fatal Infantile Cardioencephalomyopathy ------------------------
    {
        "gene": "SCO2",
        "protein": (
            "SCO2 -- 22q13.33 AR -- CIV-Copper-Scaffold-266aa -- "
            "Fatal-Infantile-Cardioencephalomyopathy-CIV-SCO2 -- "
            "Hypertrophic-Cardiomyopathy-HCM-EARLIEST-LEADING-KILLER -- "
            "Copper-Supplementation-Emerging-Rationale -- "
            "p.Glu140Lys-Most-Common-Variant-Compound-Het -- "
            "COX2-Subunit-Cu-Delivery-Defect-CuA-Centre"
        ),
        "alias": (
            "SCO2 (synthesis of cytochrome c oxidase 2; copper chaperone); OMIM gene 604272. "
            "Cardioencephalomyopathy, fatal infantile, due to cytochrome c oxidase deficiency 1 "
            "OMIM 604377. "
            "22q13.33; 266 aa; ~32 kDa; inner mitochondrial membrane (IMS face); autosomal recessive. "
            "FUNCTION: SCO2 is a copper chaperone specific to the assembly of "
            "cytochrome c oxidase (COX/CIV), the terminal enzyme of the OXPHOS chain. "
            "SCO2 delivers copper to the CuA metallocentre located in the COX2 (MT-CO2) subunit. "
            "The CuA centre receives electrons from cytochrome c and passes them to "
            "the COX1 binuclear haem a3/CuB active site (oxygen reduction centre). "
            "Without SCO2: CuA centre cannot be assembled → COX2 inactive → "
            "CIV assembly blocked → profound CIV deficiency → OXPHOS failure → "
            "energy deficit in high-demand tissues (heart, brain, muscle). "
            "CLINICAL PRESENTATION: "
            "Fatal infantile cardioencephalomyopathy — onset typically neonatal to 4 months; "
            "TRIAD: "
            "1. Hypertrophic cardiomyopathy (HCM) — EARLIEST manifestation; "
            "often causes death before full neurological syndrome develops; "
            "biventricular or concentric HCM; cardiac failure + arrhythmias; "
            "2. Lactic acidosis — severe, neonatal; "
            "lactate often >10 mmol/L; "
            "3. Hypotonia and encephalopathy — floppy, feeds poorly, "
            "reduced consciousness; "
            "COX-deficient fibres on muscle biopsy (similar to SURF1); "
            "liver involvement in some patients (hepatic failure); "
            "brain: Leigh-like MRI in some cases; "
            "PROGNOSIS: "
            "Without treatment: death usually within weeks to months; "
            "most severe OXPHOS cardiomyopathy presentation in neonates. "
            "p.Glu140Lys (E140K): "
            "most common SCO2 pathogenic variant; "
            "located in a conserved Cys-X3-Cys motif responsible for Cu2+ coordination; "
            "usually found in COMPOUND HETEROZYGOSITY with a null allele; "
            "homozygous Glu140Lys rare (reduces but does not abolish SCO2 copper transfer). "
            "COPPER SUPPLEMENTATION RATIONALE: "
            "In cell lines and some animal models: "
            "exogenous copper (as CuCl2 or copper histidinate) can partially restore "
            "CIV assembly in SCO2-deficient cells; "
            "MECHANISM: residual SCO2 protein with some copper-coordinating capacity "
            "may be enhanced by increasing copper substrate concentration; "
            "CLINICAL STATUS (2026): emerging/investigational; "
            "no large RCT completed; compassionate use in some centres; "
            "copper histidinate IV/SQ: dose and schedule under investigation; "
            "consider alongside mitochondrial cocktail. "
            "DIAGNOSIS: "
            "Neonatal HCM + lactic acidosis + hypotonia → CIV deficiency panel; "
            "COX histochemistry (muscle biopsy): COX-deficient fibres; "
            "CIV enzymology: markedly reduced in muscle, fibroblasts; "
            "SCO2 gene sequencing (small gene, 3 exons, easy to sequence); "
            "copper studies: serum copper normal (systemic copper NOT deficient — "
            "problem is intramitochondrial copper delivery to CIV). "
            "TREATMENT: "
            "Cardiac support: diuretics, ACE inhibitors, antiarrhythmics; "
            "AVOID beta-blockers (negative inotropy can worsen HCM with outflow obstruction); "
            "lactic acidosis management: IV glucose (GIR 8-12 mg/kg/min); "
            "avoid fasting; "
            "sodium bicarbonate: cautious use (pH correction); "
            "copper supplementation: emerging investigational; "
            "mitochondrial cocktail: CoQ10, carnitine, riboflavin; "
            "heart transplant: reported in some cases; does not correct muscle/brain; "
            "prognosis poor; most lethal within first year. "
            "KEY CLINICAL FACTS: "
            "SCO2 = neonatal HCM + lactic acidosis = think CIV-copper defect; "
            "SCO2 is the copper delivery arm of CIV — "
            "serum copper normal (systemic copper not deficient — problem is mitochondrial copper routing); "
            "p.Glu140Lys most common; usually compound het with null allele; "
            "COX-deficient on muscle biopsy (like SURF1) but CARDIAC is MORE PROMINENT than in SURF1; "
            "copper supplementation: rationale sound; emerging evidence; monitor closely."
        ),
        "age_of_onset": "Neonatal to 4 months (fatal infantile)",
        "inheritance": "AR",
        "locus": "22q13.33",
        "protein_size": "266 aa",
        "key_biomarker": "COX-deficient fibres; CIV enzymology markedly reduced; HCM on echocardiography; neonatal lactic acidosis",
        "pathognomonic": "Neonatal HCM + lactic acidosis + hypotonia + COX-deficient fibres = SCO2/CIV-copper defect",
        "treatment": "Cardiac support; avoid beta-blockers HCM-outflow; IV glucose crisis; copper supplementation emerging; heart Tx palliative",
        "critical_flags": [
            "NEONATAL-HCM-EARLIEST-FEATURE-LEADING-CAUSE-OF-DEATH — cardiac failure before neurology in SCO2",
            "COPPER-SUPPLEMENTATION-EMERGING-INVESTIGATIONAL — partial CIV rescue in cell models; no RCT yet",
            "p.Glu140Lys-MOST-COMMON-COMPOUND-HET — Cys-X3-Cys Cu-coordination motif; paired with null allele",
            "SERUM-COPPER-NORMAL — systemic Cu NOT deficient; problem is mitochondrial CuA delivery to COX2",
            "COX-DEFICIENT-FIBRES-MUSCLE-LIKE-SURF1 — isolated CIV; CARDIAC more prominent than in SURF1-Leigh",
            "AVOID-BETA-BLOCKERS-HCM-OUTFLOW — negative inotropy + obstruction; use diuretics + ACE-I instead",
            "IV-GLUCOSE-GIR-8-12-CRISIS — avoid fasting; continuous high-rate dextrose neonatal ICU",
            "HEART-TRANSPLANT-PALLIATIVE-NOT-CURATIVE — corrects cardiac disease; muscle/brain unaffected",
        ],
    },
    # -- PDHA1 -- Pyruvate Dehydrogenase Deficiency ----------------------------
    {
        "gene": "PDHA1",
        "protein": (
            "PDHA1 -- Xp22.12 X-linked -- PDH-E1-Alpha-Subunit-390aa -- "
            "Pyruvate-Dehydrogenase-Deficiency-Leigh-Syndrome -- "
            "KETOGENIC-DIET-THERAPEUTIC-UNIQUE-Bypasses-Pyruvate -- "
            "Thiamine-B1-Trial-MANDATORY-PDH-Thiamine-Dependent -- "
            "X-linked-Males-Severe-Females-Variable-X-Inactivation -- "
            "Lactate-Pyruvate-Ratio-NORMAL-Unlike-Other-Mito-Disease"
        ),
        "alias": (
            "PDHA1 (pyruvate dehydrogenase E1 alpha 1 subunit); OMIM gene 300502. "
            "Pyruvate dehydrogenase E1-alpha deficiency OMIM 312170. "
            "Xp22.12; 390 aa; ~43 kDa; mitochondrial matrix; "
            "E1 component (alpha subunit) of the pyruvate dehydrogenase complex (PDC); "
            "X-linked (gene on X chromosome). "
            "FUNCTION: The pyruvate dehydrogenase complex (PDC) catalyses the "
            "oxidative decarboxylation of PYRUVATE to ACETYL-CoA: "
            "Pyruvate + CoA + NAD+ → Acetyl-CoA + CO2 + NADH "
            "This is the IRREVERSIBLE gateway from glycolysis to the TCA cycle (Krebs cycle). "
            "PDC is a massive multienzyme complex (~10 MDa): "
            "E1 (PDHA1+PDHB): thiamine pyrophosphate (TPP) — decarboxylation; "
            "E2 (DLAT): dihydrolipoamide acetyltransferase; "
            "E3 (DLD): dihydrolipoamide dehydrogenase (shared with BCKDH, KGD); "
            "E3BP (PDHX): E3 binding protein; "
            "Regulatory: PDK1-4 (kinase, inhibits) + PDP1-2 (phosphatase, activates). "
            "Without PDHA1: PDC inactive → "
            "PYRUVATE CANNOT enter TCA cycle; "
            "pyruvate diverts to: "
            "(a) lactate (by LDH) → LACTIC ACIDOSIS; "
            "(b) alanine (transamination) → elevated alanine; "
            "brain energy crisis: normal brain uses glucose → pyruvate → acetyl-CoA; "
            "PDHA1 deficiency = brain pyruvate cannot be oxidised. "
            "KETOGENIC DIET — UNIQUE FEATURE (THE KEY CLINICAL PEARL): "
            "Unlike most mitochondrial diseases (where KD may be harmful or neutral): "
            "KD is SPECIFICALLY BENEFICIAL in PDHA1 deficiency because: "
            "fat → fatty acids → beta-oxidation → ACETYL-CoA (bypassing pyruvate step); "
            "ketone bodies (beta-hydroxybutyrate, acetoacetate) cross BBB and provide "
            "acetyl-CoA directly to brain TCA cycle, bypassing the defective PDC; "
            "KD is the PRIMARY metabolic intervention — start as soon as diagnosis confirmed. "
            "THIAMINE (VITAMIN B1) TRIAL — MANDATORY: "
            "PDC requires thiamine pyrophosphate (TPP) as cofactor for the E1 component; "
            "SOME PDHA1 MUTATIONS are thiamine-responsive "
            "(mutations near TPP binding site may be partially rescued by high-dose thiamine); "
            "thiamine 100-600 mg/day trial: give to ALL PDHA1 patients for minimum 6-12 months; "
            "assess response by PDH enzyme activity in fibroblasts + clinical/biochemical improvement; "
            "positive response → maintain on high-dose thiamine. "
            "LACTATE:PYRUVATE RATIO — DISTINCTIVE: "
            "NORMAL or near-normal L:P ratio (unlike other mitochondrial diseases); "
            "REASON: pyruvate accumulates (cannot enter TCA), so BOTH lactate AND pyruvate rise; "
            "L = LDH × Pyruvate, P elevated → L:P ratio relatively normal despite lactic acidosis; "
            "compare to OXPHOS defects (elevated L:P ratio >20 because pyruvate is normal but "
            "NADH:NAD ratio is high, driving LDH towards lactate). "
            "X-LINKED GENETICS: "
            "PDHA1 is on X chromosome; "
            "males: hemizygous → severe phenotype (one copy, all cells affected); "
            "females: heterozygous → VARIABLE severity depending on "
            "X-inactivation pattern in brain cells; "
            "some females severely affected (skewed X-inactivation); "
            "phenotypic range: neonatal lactic acidosis + Leigh syndrome + death → "
            "mild intellectual disability in females. "
            "CLINICAL PHENOTYPE: "
            "Leigh syndrome (most common severe presentation); "
            "neonatal lactic acidosis; "
            "episodic ataxia; "
            "carbohydrate-sensitive worsening (after glucose load → more pyruvate → more crisis); "
            "AVOID GLUCOSE INFUSIONS: glucose load worsens (drives glycolysis → more pyruvate); "
            "use lipid infusions in crisis instead. "
            "DIAGNOSIS: "
            "Lactic acidosis with NORMAL L:P ratio (key distinguishing feature); "
            "elevated alanine (transamination of accumulated pyruvate); "
            "PDH enzyme activity: markedly reduced in fibroblasts/white blood cells; "
            "PDHA1 gene sequencing; "
            "brain MRI: Leigh pattern or agenesis of corpus callosum in some. "
            "KEY CLINICAL FACTS: "
            "KD is beneficial (unique among mito diseases); "
            "Thiamine trial mandatory; "
            "glucose infusions WORSEN (use lipids in crisis); "
            "L:P ratio relatively normal (both rise together); "
            "X-linked: males severely affected; check females carefully."
        ),
        "age_of_onset": "Neonatal to infantile (severe males); variable in females",
        "inheritance": "X-linked (PDHA1 on Xp22.12)",
        "locus": "Xp22.12",
        "protein_size": "390 aa",
        "key_biomarker": "Lactic acidosis + NORMAL lactate:pyruvate ratio; elevated alanine; PDH enzyme activity low in fibroblasts",
        "pathognomonic": "Lactic acidosis + normal L:P ratio + Leigh syndrome + response to KD (X-linked)",
        "treatment": "Ketogenic diet THERAPEUTIC (bypasses pyruvate); thiamine B1 trial mandatory; avoid glucose infusions in crisis",
        "critical_flags": [
            "KETOGENIC-DIET-THERAPEUTIC-UNIQUE-PDHA1 — KD beneficial; fat→acetyl-CoA bypasses defective PDC",
            "THIAMINE-B1-TRIAL-MANDATORY-ALL-PDHA1 — some mutations TPP-binding site; 100-600mg/day trial",
            "GLUCOSE-INFUSIONS-WORSEN-CRISIS — drives more pyruvate; use LIPID infusions in acute decompensation",
            "LACTATE-PYRUVATE-RATIO-NORMAL-DISTINGUISHING — both rise proportionally; unlike OXPHOS L:P>20",
            "X-LINKED-MALES-HEMIZYGOUS-SEVERE — females variable; skewed X-inactivation determines severity",
            "ELEVATED-ALANINE-PYRUVATE-TRANSAMINATION — plasma amino acids: alanine disproportionately elevated",
            "CARBOHYDRATE-LOAD-WORSENS-AVOID — high carb → pyruvate → crisis; minimise carbs; KD is treatment",
            "CORPUS-CALLOSUM-AGENESIS-SOME-CASES — MRI structural brain malformation in severe neonatal forms",
        ],
    },
    # -- ACAD9 -- Complex I Assembly Factor -------------------------------------
    {
        "gene": "ACAD9",
        "protein": (
            "ACAD9 -- 3q21.3 AR -- CI-Assembly-Factor-621aa -- "
            "Complex-I-Deficiency-Riboflavin-B2-RESPONSIVE-50pct-CRITICAL -- "
            "Exercise-Intolerance-Lactic-Acidosis-Cardiomyopathy-TRIAD -- "
            "BN-PAGE-CI-Band-Absent-CIII-Also-Affected -- "
            "Dual-Role-Acyl-CoA-Dehydrogenase-AND-CI-Assembly -- "
            "Riboflavin-100-300mg-Day-Trial-MANDATORY-ACAD9"
        ),
        "alias": (
            "ACAD9 (acyl-CoA dehydrogenase family member 9); OMIM gene 611103. "
            "Mitochondrial complex I deficiency, nuclear type 20 (MC1DN20) OMIM 611126. "
            "3q21.3; 621 aa; ~70 kDa; mitochondrial matrix; "
            "forms homodimer; member of ACAD (acyl-CoA dehydrogenase) superfamily; "
            "autosomal recessive. "
            "FUNCTION — DUAL ROLE: "
            "1. CLASSIC ACAD ENZYMATIC ROLE (minor physiological role): "
            "ACAD9 can function as a long-chain acyl-CoA dehydrogenase in beta-oxidation "
            "(oxidises C16-C20 acyl-CoAs), but this is not its PRIMARY physiological function; "
            "in vivo, VLCAD (ACADVL) is the dominant long-chain FAO enzyme; "
            "ACAD9 long-chain FAO activity is DISPENSABLE for mitochondrial beta-oxidation. "
            "2. PRIMARY PHYSIOLOGICAL ROLE — CI ASSEMBLY: "
            "ACAD9 is an essential assembly factor for NADH:ubiquinone oxidoreductase (Complex I/CI); "
            "ACAD9 functions within the CI assembly module (MCIA complex) in the early "
            "assembly of the membrane arm of CI; "
            "ACAD9 interacts with TMEM126B, ECSIT, and TIMMDC1 "
            "to form the ACAD9-MCIA intermediate; "
            "without ACAD9: CI assembly cannot proceed past early membrane module → "
            "CI subassembly on BN-PAGE shows band at ~400 kDa (partial intermediate); "
            "mature CI band absent. "
            "RIBOFLAVIN (VITAMIN B2) RESPONSIVENESS: "
            "~50% of ACAD9 patients show DRAMATIC IMPROVEMENT with high-dose riboflavin; "
            "MECHANISM: riboflavin is the precursor of FAD (flavin adenine dinucleotide); "
            "ACAD9 protein contains an FAD-binding domain (from its ACAD structure); "
            "high FAD concentrations (from riboflavin supplementation) may "
            "stabilise the mutant ACAD9 protein → restore CI assembly function "
            "even when the enzyme's FAO catalytic activity is not restored; "
            "DOSE: riboflavin 100-300 mg/day (doses up to 400 mg used); "
            "RESPONSE TIMELINE: months to years; reassess with CI enzymology in muscle; "
            "riboflavin is safe, inexpensive, well-tolerated → TRIAL MANDATORY IN ALL ACAD9 PATIENTS. "
            "CLINICAL PRESENTATION: "
            "Onset: neonatal to early childhood; "
            "CARDINAL TRIAD: "
            "1. Exercise intolerance (lactic acidosis on exertion); "
            "2. Lactic acidosis (at rest in severe; on exertion in milder); "
            "3. Cardiomyopathy (HCM or DCM — present in ~50-70% of ACAD9 cases); "
            "additional: hypotonia, encephalopathy (less common than in SURF1/SCO2); "
            "some patients have later-onset presentation with exercise-induced myalgia + "
            "rhabdomyolysis (milder presentation, riboflavin-responsive); "
            "ACYLCARNITINE: may show mild C14-C20 acylcarnitine elevation "
            "(from residual FAO role of ACAD9 protein) — but NOT as dramatic as VLCAD or LCHAD. "
            "BN-PAGE: "
            "CI band absent (major ~980 kDa band); "
            "CIII may also be reduced (CI-CIII supercomplex requires functional CI); "
            "CI sub-assembly at ~400 kDa may be visible (partial assembly intermediate). "
            "DIAGNOSIS: "
            "CSF/plasma lactate elevated; "
            "muscle biopsy: CI enzymology markedly reduced; "
            "BN-PAGE: CI band absent + CIII reduction; "
            "ACAD9 sequencing; "
            "acylcarnitine profile: mildly elevated C16/C18 (subtle, less than VLCAD); "
            "riboflavin trial: demonstrate biochemical (CI activity) + clinical response. "
            "TREATMENT: "
            "RIBOFLAVIN: first-line, 100-300 mg/day; response rate ~50%; "
            "reassess CI enzymology at 3-6 months; "
            "mitochondrial cocktail: CoQ10, carnitine (if secondary depletion); "
            "avoid fasting; avoid intense exercise; "
            "cardiac: monitor echocardiography; HCM management; avoid beta-blockers if LVOTO; "
            "riboflavin non-responders: supportive only. "
            "KEY CLINICAL FACTS: "
            "ACAD9 = CI deficiency + riboflavin-responsive; "
            "do NOT diagnose as VLCAD based on mild C14/18 acylcarnitine elevation — "
            "check CI enzymology and ACAD9 sequencing; "
            "riboflavin is safe → give to ALL; "
            "riboflavin-responsive patients can lead near-normal lives; "
            "cardiomyopathy is important cause of morbidity; echocardiography surveillance."
        ),
        "age_of_onset": "Neonatal to early childhood (majority); some late-onset/exercise-induced",
        "inheritance": "AR",
        "locus": "3q21.3",
        "protein_size": "621 aa",
        "key_biomarker": "CI enzymology markedly reduced; BN-PAGE CI absent; lactate elevated; mild C14-C18 acylcarnitine",
        "pathognomonic": "CI deficiency + riboflavin responsive (50%) + exercise intolerance + cardiomyopathy",
        "treatment": "Riboflavin 100-300mg/day MANDATORY TRIAL (50% respond); CoQ10; avoid fasting; cardiac surveillance",
        "critical_flags": [
            "RIBOFLAVIN-B2-TRIAL-MANDATORY-ALL-ACAD9 — 50% response rate; safe; 100-300mg/day; check CI enzymology",
            "CI-DEFICIENCY-NOT-FAO-DISORDER — ACAD9 is CI assembly factor; mild acylcarnitine is not VLCAD",
            "CARDIOMYOPATHY-50-70pct-ECHOCARDIOGRAPHY-MANDATORY — HCM or DCM; leading cause of morbidity",
            "BN-PAGE-CI-ABSENT-CIII-REDUCED — CI-CIII supercomplex disruption; CIII secondary reduction",
            "RIBOFLAVIN-STABILISES-MUTANT-ACAD9-FAD-DOMAIN — mechanism via FAD-binding domain stabilisation",
            "EXERCISE-INDUCED-LACTIC-ACIDOSIS-MILDER-FORM — some adult patients; riboflavin-responsive",
            "DUAL-ROLE-FAO-AND-CI-ASSEMBLY — do not confuse with VLCAD; CI enzymology distinguishes",
            "NON-RESPONDERS-SUPPORTIVE-ONLY — 50% do not respond; CI enzyme defect persists",
        ],
    },
    # -- NDUFS1 -- Complex I Core Subunit (75kDa) -------------------------------
    {
        "gene": "NDUFS1",
        "protein": (
            "NDUFS1 -- 2q33.3 AR -- CI-Core-75kDa-Subunit-727aa -- "
            "Complex-I-Deficiency-Most-Common-Nuclear-CI-Gene-Leigh-Leukoencephalopathy -- "
            "Fe-S-Clusters-N5-N6a-N6b-Electron-Transport-Chain -- "
            "BN-PAGE-CI-Band-Absent-No-Precomplex -- "
            "T2-FLAIR-White-Matter-Basal-Ganglia-MRI -- "
            "No-Disease-Modifying-Therapy-Riboflavin-Not-Responsive"
        ),
        "alias": (
            "NDUFS1 (NADH:ubiquinone oxidoreductase core subunit S1; 75 kDa subunit); OMIM gene 157655. "
            "Mitochondrial complex I deficiency, nuclear type 5 (MC1DN5) OMIM 618226. "
            "2q33.3; 727 aa; ~80 kDa (75 kDa subunit by SDS-PAGE); mitochondrial matrix; "
            "the largest nuclear-encoded subunit of Complex I; autosomal recessive. "
            "FUNCTION: NDUFS1 is the core catalytic subunit of the "
            "NADH:ubiquinone oxidoreductase (Complex I / NADH dehydrogenase) — "
            "the largest OXPHOS complex (~980 kDa, 45 subunits, 7 mtDNA-encoded). "
            "Complex I function: "
            "NADH (from TCA cycle) → CI → transfers 2e- via 8 iron-sulphur (Fe-S) clusters → "
            "ubiquinone (CoQ) → reduced to ubiquinol → "
            "simultaneously pumps 4H+ across IMM → "
            "creates proton motive force (PMF) driving ATP synthase. "
            "NDUFS1 (75kDa subunit) contains: "
            "N5, N6a, N6b Fe-S clusters — these are the electron transfer relay stations "
            "in the matrix arm of CI; "
            "the N6b cluster is the electron exit point to ubiquinone; "
            "NDUFS1 is the last subunit from which electrons pass before reducing ubiquinone. "
            "Without NDUFS1: CI cannot be assembled correctly → "
            "BN-PAGE: CI band absent at ~980 kDa → "
            "no intermediate CI sub-assembly (contrast with ACAD9 where partial intermediate exists); "
            "CI is the most common OXPHOS deficiency; "
            "NDUFS1 is the most commonly mutated nuclear CI-subunit gene. "
            "CLINICAL PHENOTYPE: "
            "Onset: infancy to early childhood (most severe: neonatal); "
            "Two main presentations: "
            "1. LEIGH SYNDROME: "
            "bilateral symmetric T2-bright basal ganglia + brainstem (indistinguishable "
            "from SURF1-Leigh or mtDNA-encoded Leigh radiologically); "
            "developmental regression, hypotonia, ataxia, respiratory compromise; "
            "episodic decompensation with intercurrent illness; "
            "2. LEUKOENCEPHALOPATHY: "
            "diffuse T2-FLAIR white matter abnormality; "
            "can be isolated (no basal ganglia) or combined; "
            "MACROCEPHALY reported in some (white matter expansion); "
            "additional: subacute necrotising encephalopathy; "
            "CARDIOMYOPATHY: less common than in ACAD9/SCO2 (CI disorder, not CIV); "
            "SEVERITY: "
            "depends on variant — null alleles → neonatal fatal; "
            "hypomorphic missense → later onset, slower progression. "
            "BN-PAGE: "
            "CI band absent (complete assembly defect); "
            "NO CI pre-complex visible (contrast with ACAD9 where partial intermediate at ~400 kDa); "
            "CIII may be secondarily reduced (CI-CIII supercomplex assembly requires functional CI). "
            "BIOCHEMISTRY: "
            "CI enzymology: markedly reduced in muscle, fibroblasts, liver; "
            "plasma/CSF lactate: elevated; "
            "spectrophotometric CI activity: <20% of controls typically; "
            "OXPHOS enzymology panel: combined CI+CIII reduction common; CIV relatively spared. "
            "RIBOFLAVIN: "
            "NOT expected to be responsive in NDUFS1 (no FAD-binding domain in NDUFS1); "
            "riboflavin responsive CI is mainly ACAD9 (FAD domain) and ACAD9-like assembly factors; "
            "CoQ10 supplementation: theoretical basis (CI generates NADH → CoQ), "
            "supportive evidence only. "
            "DIAGNOSIS: "
            "Brain MRI: Leigh or leukoencephalopathy pattern; "
            "muscle biopsy + CI enzymology: markedly reduced; "
            "BN-PAGE: CI absent (no intermediate); "
            "NDUFS1 gene sequencing "
            "(comprehensive CI gene panel preferable given genetic heterogeneity — "
            "45 CI subunit genes + multiple CI assembly factor genes); "
            "mtDNA sequencing to exclude MT-ND1-6 variants (mt-encoded CI subunits). "
            "TREATMENT: "
            "No disease-modifying therapy; "
            "mitochondrial cocktail: CoQ10, riboflavin (limited rationale for NDUFS1 specifically), "
            "idebenone (short-chain CoQ analogue), thiamine; "
            "avoid intercurrent illness triggers (hospitalise early, IV glucose); "
            "avoid valproate; "
            "Leigh-specific: carbohydrate load worsens; "
            "palliative/supportive in severe forms; "
            "respiratory support in advanced disease. "
            "KEY CLINICAL FACTS: "
            "NDUFS1 = most commonly mutated nuclear CI-subunit gene; "
            "Leigh or leukoencephalopathy on MRI + CI deficiency = comprehensive CI gene panel; "
            "riboflavin NOT expected responsive (unlike ACAD9 which IS responsive — "
            "crucial distinction when choosing which CI-deficient patient gets riboflavin trial); "
            "BN-PAGE: CI absent, no intermediate = structural subunit defect (not assembly factor); "
            "genetic heterogeneity of CI deficiency: 20+ nuclear genes → panel testing essential."
        ),
        "age_of_onset": "Neonatal to early childhood; severity dependent on variant",
        "inheritance": "AR",
        "locus": "2q33.3",
        "protein_size": "727 aa",
        "key_biomarker": "CI enzymology markedly reduced; BN-PAGE CI absent (no pre-complex); Leigh/leukoencephalopathy MRI; CSF lactate elevated",
        "pathognomonic": "CI deficiency + Leigh syndrome or leukoencephalopathy + BN-PAGE CI absent with no intermediate",
        "treatment": "Supportive; avoid VPA/fasting/illness; CoQ10/idebenone; CI gene panel mandatory; riboflavin NOT specific response",
        "critical_flags": [
            "MOST-COMMON-NUCLEAR-CI-SUBUNIT-GENE-CAUSING-CI-DEFICIENCY — largest nuclear CI subunit; Fe-S clusters",
            "RIBOFLAVIN-NOT-RESPONSIVE-UNLIKE-ACAD9 — no FAD domain; distinguish from ACAD9 CI deficiency",
            "BN-PAGE-CI-ABSENT-NO-INTERMEDIATE — structural subunit defect; vs ACAD9 (partial intermediate visible)",
            "LEIGH-AND-LEUKOENCEPHALOPATHY-BOTH-PRESENTATIONS — MRI white matter OR basal ganglia or both",
            "COMPREHENSIVE-CI-GENE-PANEL-MANDATORY — 20+ nuclear genes; mtDNA MT-ND1-6 also exclude",
            "CIII-ALSO-REDUCED-CI-CIII-SUPERCOMPLEX — secondary CIII loss when CI absent; supercomplex disruption",
            "INTERCURRENT-ILLNESS-DECOMPENSATION-EMERGENCY — same as SURF1 Leigh; hospitalise early; IV glucose",
            "MACROCEPHALY-SOME-LEUKOENCEPHALOPATHY-FORMS — white matter expansion; head circumference monitor",
        ],
    },
]

# ---------------------------------------------------------------------------
# Patient cohort generation
# ---------------------------------------------------------------------------
def _make_patients(gene_data, seed):
    rng = random.Random(seed)
    gene = gene_data["gene"]
    inheritance = gene_data["inheritance"]
    n = 40
    patients = []

    severity_weights = {
        "POLG":   ("mild", "moderate", "severe", [0.20, 0.35, 0.45]),
        "TWNK":   ("mild", "moderate", "severe", [0.30, 0.45, 0.25]),
        "SURF1":  ("mild", "moderate", "severe", [0.10, 0.30, 0.60]),
        "BCS1L":  ("mild", "moderate", "severe", [0.15, 0.25, 0.60]),  # GRACILE severe, Bjornstad mild
        "SCO2":   ("mild", "moderate", "severe", [0.05, 0.15, 0.80]),
        "PDHA1":  ("mild", "moderate", "severe", [0.20, 0.30, 0.50]),
        "ACAD9":  ("mild", "moderate", "severe", [0.25, 0.40, 0.35]),
        "NDUFS1": ("mild", "moderate", "severe", [0.10, 0.30, 0.60]),
    }
    sevs = severity_weights.get(gene, ("mild","moderate","severe",[0.20,0.40,0.40]))
    sev_pool = []
    for s, w in zip(sevs[:3], sevs[3]):
        sev_pool.extend([s] * int(w * n))
    while len(sev_pool) < n:
        sev_pool.append(sevs[int(n/2) % 3])
    rng.shuffle(sev_pool)

    for i in range(n):
        age = rng.choices(
            [rng.uniform(0, 0.5), rng.uniform(0.5, 5), rng.uniform(5, 20), rng.uniform(20, 60)],
            weights=[0.35, 0.40, 0.15, 0.10]
        )[0]
        if gene == "TWNK" and inheritance.startswith("AD"):
            age = rng.uniform(15, 65)
        elif gene == "POLG":
            sev = sev_pool[i]
            if sev == "severe":
                age = rng.uniform(0.2, 5)
            else:
                age = rng.uniform(20, 55)
        elif gene == "PDHA1":
            age = rng.uniform(0, 3) if sev_pool[i] == "severe" else rng.uniform(1, 15)

        sev = sev_pool[i]
        on_treatment = rng.random() < 0.70
        riboflavin_response = (gene == "ACAD9") and rng.random() < 0.50
        kd_therapy = (gene == "PDHA1") and rng.random() < 0.75
        thiamine_response = (gene == "PDHA1") and rng.random() < 0.30
        cardiac = rng.random() < (0.70 if gene == "SCO2" else 0.50 if gene == "ACAD9" else 0.25 if gene in ("SURF1","NDUFS1","BCS1L") else 0.15)
        neuro = rng.random() < (0.90 if gene in ("POLG","SURF1","NDUFS1","PDHA1") else 0.65)
        leigh_mri = (gene in ("SURF1","SCO2","NDUFS1","PDHA1")) and rng.random() < 0.70
        mtdna_depletion = (gene in ("POLG","TWNK")) and rng.random() < 0.65
        vpa_exposure = (gene == "POLG") and rng.random() < 0.12  # ~12% had inadvertent VPA

        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "age_at_presentation": round(age, 2),
            "severity": sev,
            "on_treatment": on_treatment,
            "riboflavin_responsive": riboflavin_response,
            "kd_therapy": kd_therapy,
            "thiamine_responsive": thiamine_response,
            "cardiac_involvement": cardiac,
            "neurological_involvement": neuro,
            "leigh_mri": leigh_mri,
            "mtdna_depletion": mtdna_depletion,
            "vpa_inadvertent_exposure": vpa_exposure,
            "family_cascade": rng.random() < 0.45,
        })
    return patients


COHORT = []
for _idx, _g in enumerate(MITO_GENES):
    COHORT.extend(_make_patients(_g, SEED_BASE + _idx))


# ---------------------------------------------------------------------------
# API functions
# ---------------------------------------------------------------------------
def overview():
    total = len(COHORT)
    severe_n = sum(1 for p in COHORT if p["severity"] == "severe")
    riboflavin_resp = sum(1 for p in COHORT if p.get("riboflavin_responsive"))
    kd_pts = sum(1 for p in COHORT if p.get("kd_therapy"))
    cardiac_n = sum(1 for p in COHORT if p.get("cardiac_involvement"))
    neuro_n = sum(1 for p in COHORT if p.get("neurological_involvement"))
    leigh_n = sum(1 for p in COHORT if p.get("leigh_mri"))
    mtdna_dep = sum(1 for p in COHORT if p.get("mtdna_depletion"))
    vpa_exp = sum(1 for p in COHORT if p.get("vpa_inadvertent_exposure"))
    cascade = sum(1 for p in COHORT if p.get("family_cascade"))

    gene_stats = {}
    for g in MITO_GENES:
        gene = g["gene"]
        pts = [p for p in COHORT if p["gene"] == gene]
        gene_stats[gene] = {
            "n": len(pts),
            "inheritance": g["inheritance"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "severe_pct": round(sum(1 for p in pts if p["severity"] == "severe") / len(pts) * 100, 1),
            "cardiac_pct": round(sum(1 for p in pts if p.get("cardiac_involvement")) / len(pts) * 100, 1),
            "treatment": g["treatment"],
        }

    return {
        "atlas": (
            "Hereditary-Mitochondrial-RC-Atlas — Complete 8-Gene Hereditary Mitochondrial "
            "Respiratory Chain (Nuclear-Encoded) Defects Atlas"
        ),
        "subtitle": (
            "POLG-1240aa-15q25.1-AR-Alpers-SANDO-PEO3-VPA-ABSOLUTE-CI | "
            "TWNK-684aa-10q24.31-AD-PEO1-IOSCA-AR-Multiple-mtDNA-Deletions | "
            "SURF1-309aa-9q34.2-AR-Leigh-CIV-Most-Common-European-COX-Deficient-RRF-Absent | "
            "BCS1L-419aa-2q35-AR-GRACILE-Neonatal-Lethal-Bjornstad-Pili-Torti-SNHL | "
            "SCO2-266aa-22q13.33-AR-Fatal-HCM-CIV-Copper-Emerging-Therapy | "
            "PDHA1-390aa-Xp22.12-XL-KD-THERAPEUTIC-Thiamine-B1-Mandatory | "
            "ACAD9-621aa-3q21.3-AR-CI-Assembly-Riboflavin-50pct-RESPONSIVE | "
            "NDUFS1-727aa-2q33.3-AR-CI-Subunit-Leigh-Leuko-Most-Common-Nuclear-CI-Gene"
        ),
        "total_patients": total,
        "total_genes": len(MITO_GENES),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "severe_n": severe_n,
        "severe_pct": round(severe_n / total * 100, 1),
        "cardiac_n": cardiac_n,
        "cardiac_pct": round(cardiac_n / total * 100, 1),
        "neurological_n": neuro_n,
        "neurological_pct": round(neuro_n / total * 100, 1),
        "leigh_mri_n": leigh_n,
        "riboflavin_responsive_n": riboflavin_resp,
        "riboflavin_responsive_pct": round(riboflavin_resp / total * 100, 1),
        "kd_therapy_n": kd_pts,
        "mtdna_depletion_n": mtdna_dep,
        "inadvertent_vpa_exposure_n": vpa_exp,
        "family_cascade_n": cascade,
        "gene_stats": gene_stats,
        "key_contraindications": [
            "VPA-ABSOLUTE-CI-POLG-mtDNA-Depletion-Hepatic-Failure",
            "NUCLEOSIDE-ANALOGUES-CI-POLG-AZT-D4T-DDI",
            "GLUCOSE-INFUSIONS-WORSEN-PDHA1-Use-Lipids-Crisis",
            "BETA-BLOCKERS-AVOID-HCM-OUTFLOW-OBSTRUCTION-SCO2",
            "METFORMIN-ABSOLUTE-CI-ALL-MITO-DISEASE-Lactic-Acidosis",
            "STATINS-CAUTION-MITO-DISEASE-CoQ10-Depletion",
            "FASTING-EXTREME-HAZARD-ALL-MITO-DISEASE",
        ],
        "key_treatment_pearls": [
            "RIBOFLAVIN-B2-MANDATORY-TRIAL-ACAD9-50pct-RESPONSE-100-300mg-Day",
            "KETOGENIC-DIET-THERAPEUTIC-PDHA1-UNIQUE-Fat-Bypasses-Pyruvate",
            "THIAMINE-B1-MANDATORY-PDHA1-PDH-Thiamine-Dependent-Cofactor",
            "COPPER-SUPPLEMENTATION-EMERGING-SCO2-CIV-Copper-Rescue",
            "CoQ10-ALL-MITO-DISEASE-Supportive-Electron-Carrier",
            "IV-GLUCOSE-GIR-8-12-CRISIS-ALL-Except-PDHA1-Use-Lipids-Instead",
        ],
        "critical_diagnosis_tools": [
            "BN-PAGE-CI-Absent-ACAD9-vs-NDUFS1-Distinguish-By-Intermediate-Band",
            "COX-SDH-Double-Staining-SURF1-SCO2-COX-Deficient-Fibres",
            "MTDNA-QUANTIFICATION-POLG-TWNK-Depletion-or-Deletions",
            "LONG-RANGE-PCR-MULTIPLE-DELETIONS-TWNK-POLG-Adult-PEO",
            "OXPHOS-ENZYMOLOGY-Panel-All-5-Complexes-CI-II-III-IV-V",
            "LACTATE-PYRUVATE-RATIO-PDHA1-NORMAL-vs-OXPHOS->20",
        ],
        "all_critical_flags": [
            flag
            for g in MITO_GENES
            for flag in (g["critical_flags"][:2])
        ],
        "critical_treatment_alerts": [
            flag
            for g in MITO_GENES
            for flag in g["critical_flags"]
        ],
    }


def breakdown():
    per_gene = {}
    for g in MITO_GENES:
        gene = g["gene"]
        pts = [p for p in COHORT if p["gene"] == gene]
        mild     = sum(1 for p in pts if p["severity"] == "mild")
        moderate = sum(1 for p in pts if p["severity"] == "moderate")
        severe   = sum(1 for p in pts if p["severity"] == "severe")
        on_treatment = sum(1 for p in pts if p.get("on_treatment"))
        cardiac  = sum(1 for p in pts if p.get("cardiac_involvement"))
        neuro    = sum(1 for p in pts if p.get("neurological_involvement"))
        leigh    = sum(1 for p in pts if p.get("leigh_mri"))
        mtdna    = sum(1 for p in pts if p.get("mtdna_depletion"))
        ribo     = sum(1 for p in pts if p.get("riboflavin_responsive"))
        kd       = sum(1 for p in pts if p.get("kd_therapy"))
        per_gene[gene] = {
            "gene": gene,
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "n_patients": len(pts),
            "severity": {"mild": mild, "moderate": moderate, "severe": severe},
            "on_treatment": on_treatment,
            "on_treatment_pct": round(on_treatment / len(pts) * 100, 1) if pts else 0,
            "cardiac_involvement": cardiac,
            "cardiac_pct": round(cardiac / len(pts) * 100, 1) if pts else 0,
            "neurological_involvement": neuro,
            "leigh_mri": leigh,
            "mtdna_depletion": mtdna,
            "riboflavin_responsive": ribo,
            "kd_therapy": kd,
            "family_cascade": sum(1 for p in pts if p.get("family_cascade")),
            "protein_description": g["protein"],
            "age_of_onset": g["age_of_onset"],
        }
    return {
        "atlas": "Hereditary-Mitochondrial-RC-Atlas — Per-Gene Breakdown",
        "genes": per_gene,
        "aggregate": {
            "total_patients": len(COHORT),
            "total_genes": len(MITO_GENES),
            "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
            "all_inheritance": list({g["inheritance"] for g in MITO_GENES}),
        },
    }


def definitions():
    defs = {}
    for g in MITO_GENES:
        defs[g["gene"]] = g["alias"]

    defs["Mitochondrial Respiratory Chain — OXPHOS Overview and Complex Hierarchy"] = (
        "The mitochondrial respiratory chain (MRC / OXPHOS) comprises five multi-subunit complexes "
        "embedded in the inner mitochondrial membrane (IMM), plus the mobile electron carriers "
        "ubiquinone (CoQ10) and cytochrome c. "
        "COMPLEX I (NADH:ubiquinone oxidoreductase, ~980 kDa, 45 subunits): "
        "accepts 2 electrons from NADH → passes via 8 Fe-S clusters → reduces CoQ to CoQH2; "
        "pumps 4H+ across IMM; largest OXPHOS complex; "
        "7 core subunits mtDNA-encoded (MT-ND1 to MT-ND6, MT-ND4L); "
        "most common OXPHOS deficiency; "
        "nuclear CI genes: 38 structural subunits + 15+ assembly factors. "
        "COMPLEX II (succinate:ubiquinone oxidoreductase, ~140 kDa, 4 subunits): "
        "accepts 2e- from succinate (TCA) → FAD → 3 Fe-S clusters → CoQ; "
        "only OXPHOS complex entirely nuclear-encoded; "
        "SDH — fully preserved in most mitochondrial diseases → used as internal control on histochemistry. "
        "COMPLEX III (ubiquinol:cytochrome c reductase, ~480 kDa, 11 subunits/monomer): "
        "accepts 2e- from CoQH2 → Rieske Fe-S (UQCRFS1) → cytochrome b (MT-CYB) → cytochrome c1 → cyt c; "
        "pumps 4H+ via Q-cycle; "
        "1 subunit mtDNA-encoded (MT-CYB); assembly factors include BCS1L (inserts RISP/Rieske). "
        "COMPLEX IV (cytochrome c oxidase, ~200 kDa, 14 subunits): "
        "accepts 4e- from 4× cyt c → haem a → haem a3/CuB (binuclear centre) → O2 → 2H2O; "
        "pumps 4H+ across IMM; "
        "3 core subunits mtDNA-encoded (MT-CO1, MT-CO2, MT-CO3); "
        "requires copper cofactors (SCO1/SCO2 deliver Cu); SURF1 inserts haem a/a3. "
        "COMPLEX V (ATP synthase, ~600 kDa): "
        "uses PMF (H+ gradient) to phosphorylate ADP → ATP; "
        "F0 (membrane) + F1 (catalytic); 2 subunits mtDNA-encoded (MT-ATP6, MT-ATP8). "
        "ELECTRON FLOW: NADH → CI → CoQ ← CII → CIII → cyt c → CIV → O2; "
        "proton gradient drives CV → ATP. "
        "SUPERCOMPLEXES ('respirasomes'): "
        "CI+CIII+CIV can assemble into supercomplexes; "
        "disruption of one complex can secondarily reduce others "
        "(CI absent → CIII also reduced on BN-PAGE; seen in NDUFS1, ACAD9). "
        "COMBINED OXPHOS DEFICIENCY: "
        "Nuclear-encoded mtDNA maintenance genes (POLG, TWNK) → "
        "mtDNA depletion/deletions → deficiency of ALL complexes with mtDNA-encoded subunits (CI, CIII, CIV, CV) "
        "→ combined OXPHOS deficiency; CII relatively spared (fully nuclear-encoded). "
        "DIAGNOSIS HIERARCHY: "
        "1. Plasma/CSF lactate + L:P ratio; "
        "2. Urine organic acids (lactate, TCA intermediates); "
        "3. Plasma amino acids (alanine elevated); "
        "4. Muscle biopsy: histochemistry (COX/SDH) + OXPHOS enzymology + mtDNA studies + BN-PAGE; "
        "5. Fibroblast OXPHOS enzymology; "
        "6. Gene panel (nuclear OXPHOS genes + mtDNA sequencing + deletion/depletion analysis)."
    )

    defs["VPA and Mitochondrial Disease — Absolute Contraindications"] = (
        "Valproate (valproic acid, sodium valproate) is CONTRAINDICATED in several "
        "mitochondrial and metabolic diseases. "
        "POLG (most critical): "
        "VPA inhibits POLG catalytic activity directly AND through VPA-CoA metabolites; "
        "in POLG-deficient patients (already reduced mtDNA copy number): "
        "VPA → catastrophic mtDNA depletion in liver → fulminant hepatic failure; "
        "ABSOLUTE CONTRAINDICATION — even a single dose can precipitate fatal liver failure; "
        "mechanism: VPA is metabolised to valproyl-CoA which is a direct POLG inhibitor; "
        "ALL MITO DISEASE (general): "
        "VPA inhibits CI and CIV enzyme activities at high concentrations; "
        "worsens energy deficit in any OXPHOS disease; "
        "best practice: AVOID VPA in any proven or suspected mitochondrial disease; "
        "ALTERNATIVES: "
        "levetiracetam (LEV): preferred for epilepsy in mito disease; "
        "phenobarbital: second line; "
        "benzodiazepines: for acute status; "
        "RULE: 'Refractory seizures + unexplained hepatopathy in child → EXCLUDE POLG before VPA'; "
        "DLD DISEASE: VPA CI — inhibits E3 (dihydrolipoamide dehydrogenase) shared with BCKDH/PDH; "
        "ORGANIC ACIDEMIAS (PA, MMA, IVA): VPA CI — worsens CoA sequestration; "
        "UREA CYCLE DISORDERS (OTC): VPA CI — hyperammonaemia; "
        "ALPERS-SPECIFIC: VPA hepatotoxicity in Alpers is not idiosyncratic — it is a direct, "
        "mechanism-based drug interaction; warn all prescribers; wear medic alert bracelet."
    )

    defs["Leigh Syndrome — DDx of Nuclear vs mtDNA Forms"] = (
        "Leigh syndrome (subacute necrotising encephalopathy) is defined by: "
        "symmetric T2-bright lesions in basal ganglia (putamen, caudate) + brainstem (periaqueductal grey, "
        "substantia nigra, inferior olive, tegmentum), with progressive neurological regression. "
        "COMMON GENETIC CAUSES: "
        "NUCLEAR-ENCODED (from this atlas): "
        "SURF1 (CIV assembly) — most common European nuclear Leigh; isolated CIV deficiency; "
        "NDUFS1 (CI subunit) — CI deficiency Leigh or leukoencephalopathy; "
        "PDHA1 (PDH E1α) — X-linked; L:P normal; KD beneficial; "
        "ACAD9 (CI assembly) — CI deficiency; riboflavin-responsive 50%; "
        "SCO2 (CIV copper) — Leigh-like + HCM dominant; "
        "SDHA (CII subunit) — SDHA mutations cause Leigh; only CI+CII combined deficiency; "
        "MT-DNA-ENCODED: "
        "MT-ATP6 m.8993T>G or m.8993T>C — NARP/Leigh syndrome; CIV (ATP6) mutation; "
        "most severe mtDNA-Leigh; maternal inheritance; "
        "MT-ND genes (MT-ND1-6, MT-ND4L) — CI subunit mutations causing Leigh-like; "
        "MT-TL1 m.3243A>G (MELAS) — can cause Leigh-like features; "
        "DISTINGUISHING FEATURES: "
        "Maternal inheritance → mtDNA (pedigree); "
        "COX-deficient fibres on biopsy → CIV gene (SURF1, SCO2); "
        "RRFs present → mtDNA-level pathology (POLG, TWNK, mtDNA-encoded mutation); "
        "RRFs absent → nuclear-encoded assembly factor (SURF1, ACAD9, NDUFS1); "
        "isolated CI deficiency → nuclear CI gene (NDUFS1, ACAD9, NDUFV1, etc.); "
        "isolated CIV deficiency → SURF1, SCO2, COX10, COX15; "
        "combined CI+CIII+CIV → mtDNA maintenance gene (POLG, TWNK, mtDNA-encoded); "
        "L:P RATIO: "
        "ELEVATED (>20): OXPHOS complex defect (impaired NADH re-oxidation → NADH/NAD ratio high); "
        "NORMAL: PDH deficiency (PDHA1) — both lactate and pyruvate rise proportionally; "
        "BIOTIN TRIAL: if biotin-thiamine responsive basal ganglia disease suspected "
        "(SLC19A3 BTBGD mutation — treatment-emergent cause of 'Leigh-like' MRI + dramatic biotin response). "
        "WORKUP SEQUENCE: "
        "MRI → lactate/L:P → muscle biopsy (enzymology + histochemistry + BN-PAGE + mtDNA) "
        "→ comprehensive gene panel (nuclear OXPHOS + mtDNA). "
        "KEY DDx non-mitochondrial: "
        "Biotin-thiamine-responsive basal ganglia disease (SLC19A3) — biotin CURATIVE; "
        "Wernicke encephalopathy — thiamine B1 IV CURATIVE; "
        "Wilson disease — copper accumulation; "
        "methylmalonic/propionic acidemia (basal ganglia stroke-like); "
        "Canavan disease (T2 subcortical white matter; N-acetylaspartate elevated on MRS)."
    )

    defs["BN-PAGE (Blue Native PAGE) — Interpreting CI/CIII Assembly Patterns"] = (
        "Blue native polyacrylamide gel electrophoresis (BN-PAGE) separates intact "
        "mitochondrial complexes and their assembly intermediates by mass. "
        "Used to distinguish: "
        "(a) Complete absence of a mature complex; "
        "(b) Accumulation of an intermediate (partial assembly); "
        "(c) Supercomplex disruption. "
        "NORMAL BN-PAGE BANDS: "
        "CI: ~980 kDa (fully assembled, most prominent); "
        "CIII: ~500 kDa (dimer); ~250 kDa (monomer); "
        "CIV: ~200 kDa; "
        "CI-CIII supercomplex: ~1300-2000 kDa. "
        "PATHOLOGICAL PATTERNS IN THIS ATLAS: "
        "NDUFS1 (CI structural subunit): "
        "CI band ABSENT; no sub-assembly intermediate visible; "
        "CIII secondarily reduced (CI-CIII supercomplex disrupted). "
        "ACAD9 (CI assembly factor): "
        "CI band ABSENT; "
        "~400 kDa CI sub-assembly intermediate VISIBLE "
        "(ACAD9-MCIA intermediate containing partial CI membrane module); "
        "CIII also reduced (supercomplex). "
        "BCS1L (CIII assembly factor — RISP insertion): "
        "CIII pre-complex PRESENT at ~500 kDa (without RISP, lower density, slightly different migration); "
        "mature CIII band absent or very reduced; "
        "RISP-free pre-complex is catalytically dead. "
        "TTC19 (CIII assembly factor — late step): "
        "CIII pre-complex ABSENT; "
        "CONTRAST WITH BCS1L where pre-complex accumulates. "
        "SURF1 / SCO2 (CIV assembly / copper scaffold): "
        "CIV band absent or very reduced; "
        "CI and CIII relatively preserved (isolated CIV defect). "
        "POLG / TWNK (mtDNA maintenance): "
        "Combined CI+CIII+CIV reduction (all mtDNA-subunit-containing complexes); "
        "CII preserved (fully nuclear-encoded). "
        "CLINICAL UTILITY: "
        "BN-PAGE on fibroblasts (most accessible) or muscle; "
        "helps distinguish assembly factor defect vs structural subunit defect; "
        "identifies which complex is deficient before gene panel result; "
        "guides prioritisation of OXPHOS gene panel. "
        "LIMITATIONS: "
        "Some complexes not well resolved in all tissue types; "
        "fibroblasts may underestimate OXPHOS defect if defect not expressed in skin cells "
        "(brain-enriched gene expression); "
        "muscle biopsy gives most reliable OXPHOS enzymology + BN-PAGE."
    )

    defs["Mitochondrial Disease — Key Drug Contraindications (Complete List)"] = (
        "ABSOLUTE CONTRAINDICATIONS in mitochondrial disease (nuclear or mtDNA): "
        "VALPROATE (VPA): "
        "CI in all suspected/proven mitochondrial disease; "
        "ABSOLUTE CI in POLG disease (direct POLG inhibitor → mtDNA depletion → fatal hepatic failure); "
        "mechanism: valproyl-CoA metabolites inhibit POLG and CI/CIV; "
        "alternative AEDs: levetiracetam, phenobarbital, benzodiazepines. "
        "METFORMIN: "
        "ABSOLUTE CI in all mitochondrial disease; "
        "mechanism: inhibits Complex I → worsens lactic acidosis; "
        "CI inhibition → ↑NADH:NAD → ↑lactate; risk of fatal lactic acidosis; "
        "diabetes in mito disease patient: insulin preferred; DPP4-I or GLP-1 agonist caution. "
        "NUCLEOSIDE ANALOGUES (AZT, d4T, ddI, stavudine): "
        "inhibit POLG (mtDNA depletion risk); "
        "CI in POLG disease specifically; caution in ALL mito disease; "
        "alternative ART regimens: tenofovir alafenamide (TAF) + integrase inhibitor preferred. "
        "AMINOGLYCOSIDES: "
        "CI in LHON (MT-ND4/MT-ND1/MT-ND6 mutations) — risk of blindness; "
        "general caution in mito patients — cochleotoxicity risk if mitochondrial hearing impairment. "
        "STATINS: "
        "Not absolutely contraindicated; use with caution; "
        "mechanism: statins deplete mevalonate pathway → CoQ10 depletion; "
        "CoQ10 is essential electron carrier in OXPHOS; "
        "supplement CoQ10 100-600 mg/day if statin needed. "
        "SODIUM BICARBONATE (high dose): "
        "can paradoxically worsen lactic acidosis; "
        "not a routine treatment for mitochondrial lactic acidosis; "
        "use only in severe life-threatening acidosis (pH <7.0). "
        "GENERAL CAUTION: "
        "Any drug that inhibits mitochondrial function → use with extreme caution; "
        "LINEZOLID: inhibits mitochondrial protein synthesis → lactic acidosis; "
        "CHLORAMPHENICOL: CI in mito disease; "
        "PROPOFOL (PRIS): propofol infusion syndrome = acquired mito disease; "
        "triggers fatal lactic acidosis in undiagnosed mito patients; "
        "alternative anaesthetic: sevoflurane/isoflurane preferred; "
        "ANAESTHESIA in mito disease: avoid propofol infusions >5 mg/kg/hr; "
        "use volatile anaesthetics; monitor pH/lactate throughout; maintain normoglycaemia."
    )

    return {
        "atlas": "Hereditary-Mitochondrial-RC-Atlas — Clinical Definitions",
        "definitions": defs,
        "total_genes": len(MITO_GENES),
        "total_definition_entries": len(defs),
    }
