#!/usr/bin/env python3
"""NCL-Atlas — Complete 8-Gene Neuronal Ceroid Lipofuscinosis (Batten Disease) Atlas
CLN1 (PPT1) · CLN2 (TPP1) · CLN3 · CLN4B (DNAJC5) · CLN5 · CLN6 · CLN7 (MFSD8) · CLN10 (CTSD)
320-patient aggregate cohort (8 × 40, seeds 934–941)

Neuronal Ceroid Lipofuscinoses (NCL / Batten disease) facts:
  - NCL = the most common group of inherited progressive neurodegenerative storage disorders
    of childhood; characterised by accumulation of autofluorescent ceroid lipopigment in
    neurons and other cell types; lysosomal storage diseases.
  - Shared clinical features: progressive cognitive and motor decline, seizures, visual
    failure (except CLN4B adult form), and premature death.
  - KEY TEACHING POINTS:
      CLN2/TPP1 — ONLY NCL gene with FDA/EMA-APPROVED ERT: cerliponase alfa (Brineura)
        300 mg ICV every 2 weeks via Ommaya reservoir; slows motor decline; start BEFORE
        symptom onset if sibling diagnosed.
      CLN3 — MOST COMMON NCL; 1.02 kb deletion in 73-85% of alleles; vacuolated
        lymphocytes on peripheral blood smear are DIAGNOSTIC (quick cheap test).
      CLN4B (DNAJC5) — ONLY AUTOSOMAL DOMINANT NCL; no vision loss (visual system
        spared); adult onset; family history may span generations.
      CLN1/PPT1 — GRODs (granular osmiophilic deposits) on EM PATHOGNOMONIC;
        isoelectric EEG by age 3y; most severe/earliest onset of all NCL.
      CLN10/CTSD — Congenital NCL (earliest of all); HCM unique to CLN10 among NCL;
        GRODs on EM (same as CLN1); cathepsin D deficiency.
  - EM ultrastructure is gene-specific and remains diagnostically valuable:
      GRODs = CLN1, CLN10 | Curvilinear bodies = CLN2 | Fingerprint profiles = CLN3, CLN4B
      Mixed CB + FP = CLN5, CLN7 | Rectilinear profiles = CLN6
  - PPT1 enzyme assay (DBS/leukocytes) + TPP1 enzyme assay: first-line biochemical
    confirmation before WES for CLN1 and CLN2 respectively.
  - Ophthalmology (ERG + fundus): mandatory — NCL causes progressive retinal dystrophy in
    ALL types EXCEPT CLN4B adult form (visual system spared in CLN4B).

COHORT: 8 × 40 = 320 patient slots (seeds 934–941; gene-specific seeds)
"""

import random

SEED_BASE = 934

# ── All 8 NCL Genes ───────────────────────────────────────────────────────────────
NCL_GENES = [
    # ── CLN1 (PPT1) — Infantile NCL / Santavuori-Haltia ─────────────────────────
    {
        "gene": "CLN1", "protein": "PPT1",
        "alias": "CLN1/PPT1 — Infantile NCL / Santavuori-Haltia disease (OMIM #256730)",
        "aa": "306 aa", "kDa": "34 kDa",
        "gene_class": "Lysosomal thioesterase: palmitoyl-protein thioesterase 1 — removes fatty acid thioesters (palmitoyl groups) from S-acylated proteins during lysosomal degradation",
        "ncl_subgroup": "Infantile NCL (CLN1)",
        "locus": "1p34.2", "omim_gene": 600722,
        "inheritance": "AR. 1p34.2. Both sexes equally. Finnish founder p.Arg122Trp (~40% of Finnish CLN1 alleles).",
        "em_finding": "GRODs (Granular Osmiophilic Deposits) — electron-dense granular material in lysosomes; PATHOGNOMONIC for CLN1 (and CLN10)",
        "onset_range_y": (0.5, 2.0),
        "phenotype": (
            "Infantile NCL (Santavuori-Haltia): most severe and earliest NCL; onset 6-24 months; "
            "rapid neurodegeneration; isoelectric (flat) EEG by age 3y; visual failure (optic atrophy, "
            "macular degeneration); seizures; myoclonus; death typically 8-13y"
        ),
        "disease": (
            "CLN1 encodes palmitoyl-protein thioesterase 1 (PPT1, 306aa, 34kDa), a lysosomal "
            "thioesterase that removes long-chain fatty acid thioesters (palmitoyl groups) from "
            "S-acylated (palmitoylated) proteins during lysosomal proteolysis. "
            "PPT1 is highly expressed in neurons. Loss of PPT1 → failure to depalmitoylate "
            "substrate proteins → accumulation of fatty acid-modified proteins and ceroid "
            "lipopigment (autofluorescent storage material) within neuronal lysosomes. "
            "EM: GRODS (granular osmiophilic deposits) — electron-dense granular material "
            "filling lysosomes; PATHOGNOMONIC for CLN1. GRODs are also seen in CLN10/CTSD. "
            "Clinical: onset 6-24 months (most commonly 12-18 months); initial hypotonia and "
            "developmental plateau followed by rapid regression; visual failure early (macular "
            "degeneration, optic atrophy); myoclonus; generalised seizures; hyperekplexia; "
            "isoelectric (flat) EEG by age 3 years (profoundly abnormal); rapid progression "
            "to vegetative state; death typically 8-13 years. "
            "Finnish founder variant p.Arg122Trp (c.364C>T): ~40% of Finnish CLN1 alleles; "
            "Finnish incidence ~1/20,000. Global incidence ~1/100,000-200,000. "
            "PPT1 enzyme assay (leukocytes or DBS): severely reduced activity; first-line "
            "biochemical test before WES. "
            "No FDA/EMA-approved ERT for CLN1. Gene therapy (AAV-PPT1 intrathecal/ICV) "
            "in clinical trials; shown to extend survival in PPT1-null mice."
        ),
        "hallmark": (
            "CLN1/PPT1 HALLMARKS: "
            "(1) GRODS ON EM PATHOGNOMONIC: granular osmiophilic deposits in lysosomes; "
            "distinguishes CLN1 from other NCL types (except CLN10 which also has GRODs); "
            "(2) ISOELECTRIC EEG BY AGE 3Y: flat EEG (absence of recordable brain electrical "
            "activity) by 3 years is a clinical hallmark of CLN1; "
            "(3) EARLIEST/MOST SEVERE NCL: onset 6-24m; fastest progression of all NCL forms; "
            "(4) PPT1 ENZYME ASSAY: first-line biochemical test; severely reduced DBS/leukocyte PPT1 activity; "
            "(5) FINNISH FOUNDER p.Arg122Trp: Finnish enrichment; "
            "(6) NO APPROVED ERT: only gene therapy trials (AAV-PPT1); "
            "(7) VISUAL FAILURE EARLY: macular degeneration + optic atrophy; "
            "(8) OPHTHALMOLOGY MANDATORY: ERG + fundus mandatory at diagnosis"
        ),
        "key_ddx": (
            "CLN1 DDx: "
            "(1) CLN2/TPP1: onset 2-4y (later than CLN1); CB+FP on EM (not GRODs); APPROVED ERT available; "
            "(2) CLN10/CTSD: GRODs also present; but congenital/neonatal onset + HCM unique to CLN10; "
            "(3) Rett syndrome (MECP2): progressive neurological disorder in girls; no GRODs; "
            "(4) GM2 gangliosidosis (Tay-Sachs): cherry-red spot; enzyme assay HexA; "
            "(5) Krabbe disease: fast-conducting nerve; psychosine; GALC enzyme; "
            "(6) Canavan disease: macrocephaly; NAA on MRS"
        ),
        "diet_treatment": (
            "No approved ERT for CLN1. Supportive: antiepileptics (LEV preferred, avoid VPA in "
            "NCL as may worsen; carbamazepine, clonazepam for myoclonus); "
            "feeding support (NG/PEG tube as swallowing fails); physiotherapy; "
            "visual aids until vision lost; palliative care. "
            "Gene therapy trials: AAV-PPT1 ICV — Phase I/II (NCT01414985 and follow-on trials). "
            "Cysteamine + NAC combination (antioxidant, thioester substrate): preclinical benefit; "
            "clinical trial data awaited."
        ),
        "gene_therapy_status": (
            "AAV-PPT1 (AAV2 or AAVrh10 serotypes, ICV delivery): preclinical survival extension in "
            "PPT1-null mice; Phase I/II clinical trials ongoing. PPT1 cDNA (306aa) fits easily in "
            "AAV capsid. Intrathecal/ICV route preferred for CNS coverage. "
            "Cysteamine + N-acetylcysteine (thioester bond cleaving + antioxidant): clinical trial "
            "NCT02227771; modest benefit signal."
        ),
        "critical_ci": (
            "CRITICAL: (1) Missing PPT1 enzyme assay — always test DBS/leukocyte PPT1 before WES; "
            "(2) Assuming GRODs = CLN10 only — GRODs occur in BOTH CLN1 and CLN10; "
            "(3) Missing ophthalmology at diagnosis — ERG + fundus mandatory; "
            "(4) Using VPA for seizures — VPA is generally avoided in NCL (may worsen); "
            "(5) Delaying diagnosis in Finnish patients — p.Arg122Trp targeted testing first"
        ),
        "nbs_marker": (
            "No universal NBS. PPT1 enzyme activity (DBS cards): very low/absent → CLN1. "
            "PPT1 included in expanded NBS programs in some countries. "
            "Molecular: CLN1 sequencing; targeted p.Arg122Trp in Finnish patients."
        ),
        "key_biomarker": (
            "PPT1 enzyme assay (DBS/leukocytes): severely reduced. EM: GRODs (lysosomal granular "
            "osmiophilic deposits). EEG: isoelectric by age 3y. ERG: absent/severely attenuated "
            "(retinal dystrophy). CLN1 sequencing. Autofluorescent storage material (UV microscopy)."
        ),
        "severity_spectrum": (
            "Classic infantile CLN1: onset 6-18m; death 8-13y. Late-infantile CLN1: onset 2-4y; "
            "slower progression. Juvenile CLN1: rare; missense variants; onset 4-10y. "
            "Adult CLN1: very rare; missense variants; onset >20y. "
            "Genotype correlates: null → infantile; missense → later onset."
        ),
        "founder_variant": "p.Arg122Trp (c.364C>T) — Finnish founder; ~40% of Finnish CLN1 alleles; elevated incidence in Finland (~1/20,000 vs global 1/100,000-200,000).",
        "key_variants": ["p.Arg122Trp (Finnish founder)", "p.Thr75Pro", "p.Leu10X (nonsense)", "c.223A>G p.Arg75Gly", "Exon 4 deletion"],
        "seed": SEED_BASE + 0,
    },

    # ── CLN2 (TPP1) — Late-Infantile NCL / Janský-Bielschowsky ──────────────────
    {
        "gene": "CLN2", "protein": "TPP1",
        "alias": "CLN2/TPP1 — Late-Infantile NCL / Janský-Bielschowsky disease (OMIM #204500)",
        "aa": "563 aa", "kDa": "66 kDa (precursor); 46 kDa (mature processed form)",
        "gene_class": "Lysosomal serine protease: tripeptidyl peptidase I (TPP1) — cleaves tripeptides from N-terminus of proteins in acidic lysosomal environment",
        "ncl_subgroup": "Late-Infantile NCL (CLN2) — ONLY NCL with approved ERT",
        "locus": "11p15.4", "omim_gene": 607998,
        "inheritance": "AR. 11p15.4. Both sexes equally. p.Arg208X + p.Gln422X are the two most common variants.",
        "em_finding": "Curvilinear bodies (CB) + fingerprint profiles (FP) — curvilinear profiles are MOST CHARACTERISTIC; fingerprint profiles also seen",
        "onset_range_y": (2.0, 4.0),
        "phenotype": (
            "Late-Infantile NCL (Janský-Bielschowsky): onset 2-4y with seizures (often first symptom); "
            "followed by ataxia, myoclonus, cognitive decline, visual failure; death typically 8-12y. "
            "ONLY NCL gene with FDA/EMA-approved ERT (cerliponase alfa/Brineura)"
        ),
        "disease": (
            "CLN2 encodes tripeptidyl peptidase I (TPP1, 563aa precursor; 46kDa mature), a lysosomal "
            "serine protease that cleaves tripeptides from the N-terminus of proteins. TPP1 is an "
            "aspartyl endopeptidase-like proenzyme activated by autocatalytic cleavage in the acidic "
            "lysosomal environment. "
            "TPP1 deficiency → failure to degrade lysosomal proteins → accumulation of subunit c of "
            "mitochondrial ATP synthase (the main component of NCL storage material) and other proteins "
            "→ ceroid lipopigment accumulation. "
            "EM: curvilinear bodies (CB) — parallel curved stacks of membranes; fingerprint profiles "
            "(FP) — concentric whorled patterns; CB+FP mixed is most characteristic for CLN2. "
            "Clinical: seizures typically the FIRST symptom at age 2-4y (language delay + seizure onset "
            "is a characteristic combination); rapid progression to ataxia, myoclonus, cognitive decline; "
            "visual failure (macular degeneration, optic atrophy); loss of ambulation by 4-6y after "
            "symptom onset; death typically 8-12y without ERT. "
            "TPP1 enzyme assay (DBS/leukocytes): severely reduced; FIRST-LINE test. "
            "Common variants: p.Arg208X (c.622C>T nonsense) and p.Gln422X (c.1264C>T nonsense) "
            "together account for ~40-50% of CLN2 alleles in European patients. "
            "CERLIPONASE ALFA (Brineura, BioMarin): first and only approved ERT for NCL; "
            "FDA approval 2017; EMA approval 2017; 300 mg ICV (intracerebroventricular) every 2 weeks "
            "via Ommaya reservoir; slows motor decline significantly; start BEFORE symptom onset "
            "if sibling diagnosed (pre-symptomatic treatment ideal). "
            "Incidence: ~1/100,000-500,000."
        ),
        "hallmark": (
            "CLN2/TPP1 HALLMARKS: "
            "(1) ONLY NCL WITH FDA/EMA APPROVED ERT: cerliponase alfa (Brineura) 300mg ICV q2 weeks; "
            "Ommaya reservoir required; slows motor decline; treat pre-symptomatically if sibling dx; "
            "(2) SEIZURES AS FIRST SYMPTOM: onset 2-4y; seizures (often febrile convulsions or myoclonic) "
            "before other neurological symptoms; language delay + early seizures = test TPP1; "
            "(3) CURVILINEAR BODIES + FINGERPRINT PROFILES ON EM: mixed CB+FP characteristic; "
            "(4) TPP1 ENZYME ASSAY FIRST-LINE: DBS/leukocyte TPP1 activity severely reduced; "
            "(5) p.Arg208X + p.Gln422X COMMON VARIANTS: ~40-50% European alleles; targeted testing; "
            "(6) VISUAL FAILURE: macular degeneration + optic atrophy; ERG mandatory; "
            "(7) PRE-SYMPTOMATIC TREATMENT: start Brineura before symptoms in at-risk siblings; "
            "(8) ICV RESERVOIR: Ommaya port required; biweekly administration; specialist centre"
        ),
        "key_ddx": (
            "CLN2 DDx: "
            "(1) CLN1/PPT1: earlier onset (6-18m); GRODs (not CB) on EM; no approved ERT; "
            "(2) CLN3: older onset (4-10y); fingerprint profiles; vacuolated lymphocytes on smear; "
            "(3) Dravet syndrome (SCN1A): early febrile seizures; no storage material on EM; normal vision; "
            "(4) GLUT1 deficiency: seizures + intellectual disability; CSF glucose low; normal EM; "
            "(5) GM2 gangliosidosis (Tay-Sachs): cherry-red spot; normal ERG; HexA assay; "
            "(6) Myoclonic epilepsy of Lafora: Lafora bodies (PAS+ intracellular); EPM2A/EPM2B"
        ),
        "diet_treatment": (
            "Cerliponase alfa (Brineura) 300mg ICV every 2 weeks via Ommaya reservoir: "
            "FDA-approved (2017) + EMA-approved (2017); slows motor decline; administer at "
            "specialist centre; reservoir requires surgical placement; pre-treat with "
            "antihistamine + corticosteroid to reduce infusion reactions. "
            "Antiepileptics: LEV preferred; clonazepam/clobazam for myoclonus; avoid VPA. "
            "Supportive: PEG tube, physiotherapy, visual aids, palliative care."
        ),
        "gene_therapy_status": (
            "AAV-TPP1 gene therapy (AAVrh10-CLN2 or similar): Phase I/II trials; "
            "preclinical data positive in CLN2 dog model (natural model — Dachshunds). "
            "Combined ERT + gene therapy approaches under investigation. "
            "TPP1 cDNA (563aa) at limit of standard AAV capacity; high-capacity AAV vectors used."
        ),
        "critical_ci": (
            "CRITICAL: (1) Delaying ERT — cerliponase alfa must start BEFORE advanced motor decline; "
            "pre-symptomatic treatment in at-risk siblings is ideal; "
            "(2) Missing TPP1 enzyme assay — test DBS/leukocytes before WES in late-infantile NCL; "
            "(3) Not placing Ommaya reservoir — ICV delivery requires surgical Ommaya; cannot give IV; "
            "(4) Using VPA for seizures — avoid in NCL; "
            "(5) Missing ophthalmology — ERG + fundus at diagnosis mandatory"
        ),
        "nbs_marker": (
            "TPP1 enzyme assay (DBS): included in expanded NBS in some jurisdictions. "
            "NBS detection allows pre-symptomatic treatment initiation. "
            "CLN2 sequencing: targeted p.Arg208X + p.Gln422X first in European patients."
        ),
        "key_biomarker": (
            "TPP1 enzyme assay (DBS/leukocytes): severely reduced. EM: curvilinear bodies + fingerprint "
            "profiles. EEG: high-amplitude occipital spikes (early); progressive deterioration. "
            "ERG: attenuated (retinal dystrophy). CLN2 sequencing. CSF neurofilament light chain (elevated)."
        ),
        "severity_spectrum": (
            "Classic late-infantile: onset 2-4y; death 8-12y without ERT. With cerliponase alfa (Brineura): "
            "significantly slowed motor decline; extended ambulatory period. "
            "Variant late-infantile CLN2: onset 4-7y; slower progression; missense variants. "
            "Juvenile CLN2: rare; onset >6y."
        ),
        "founder_variant": "p.Arg208X (c.622C>T) and p.Gln422X (c.1264C>T) — most common variants, together ~40-50% of European CLN2 alleles. No single ethnic-specific founder comparable to Finnish CLN1.",
        "key_variants": ["p.Arg208X (c.622C>T)", "p.Gln422X (c.1264C>T)", "c.509-1G>C (splice)", "p.Gly284Val", "c.887-10A>G (splice)"],
        "seed": SEED_BASE + 1,
    },

    # ── CLN3 — Juvenile NCL / Batten Disease (JNCL) ─────────────────────────────
    {
        "gene": "CLN3", "protein": "CLN3",
        "alias": "CLN3 — Juvenile NCL / Batten disease (JNCL) (OMIM #204200)",
        "aa": "438 aa", "kDa": "48 kDa",
        "gene_class": "Transmembrane lysosomal/late endosomal protein — precise function uncertain; roles proposed in lysosomal pH regulation, arginine transport, autophagic flux, membrane trafficking",
        "ncl_subgroup": "Juvenile NCL (CLN3) — MOST COMMON NCL",
        "locus": "16p12.1", "omim_gene": 607042,
        "inheritance": "AR. 16p12.1. Both sexes equally. 1.02 kb genomic deletion (chr16:28,532,191-28,533,199) in 73-85% of CLN3 disease alleles.",
        "em_finding": "Fingerprint profiles (FP) — concentric whorled lamellar profiles; vacuolated lymphocytes on blood smear DIAGNOSTIC",
        "onset_range_y": (4.0, 10.0),
        "phenotype": (
            "Juvenile NCL (Batten disease / JNCL): MOST COMMON NCL; onset 4-10y with visual failure "
            "(first symptom); followed by seizures, cognitive decline, motor deterioration; "
            "vacuolated lymphocytes on blood smear DIAGNOSTIC; death typically 15-35y"
        ),
        "disease": (
            "CLN3 encodes a 438aa (48kDa) transmembrane protein localised to lysosomes and late "
            "endosomes. The precise function of CLN3 remains incompletely understood despite being "
            "the most common NCL gene. Proposed roles include: lysosomal pH regulation, lysosomal "
            "arginine transport (CLN3 as lysosomal arginine transporter), autophagic flux, and "
            "membrane trafficking (endosomal recycling). "
            "CLN3 deficiency → subunit c of mitochondrial ATP synthase accumulation (same storage "
            "material as CLN2) + fingerprint profile ultrastructure on EM. "
            "MOST COMMON NCL GENE: CLN3 accounts for the majority of NCL cases worldwide. "
            "1.02 kb genomic deletion (chr16:28,532,191-28,533,199): a specific intragenic deletion "
            "removing exons 7-8 of CLN3; detected by targeted PCR (NOT by WES — genomic deletion); "
            "found in 73-85% of CLN3 disease alleles; homozygous in ~40% of JNCL patients; "
            "compound heterozygous (deletion + point mutation) in ~45%. "
            "CLINICAL PEARL: Do NOT skip to WES if CLN3 is suspected — targeted PCR for the 1.02kb "
            "deletion is the first step; deletion-negative alleles then go to sequencing. "
            "VACUOLATED LYMPHOCYTES: vacuoles visible on routine peripheral blood smear (Giemsa stain) "
            "in CLN3 patients — DIAGNOSTIC (pathognomonic); quick and cheap test. "
            "Clinical: visual failure (macular degeneration, rod-cone dystrophy) is FIRST symptom "
            "at age 4-7y; seizures follow (myoclonic + generalised); cognitive decline; "
            "behavioural/psychiatric features; progressive motor deterioration; parkinsonian features "
            "late; death typically 15-35y (range 10-50y+). "
            "No approved ERT or disease-modifying therapy. Supportive treatment only."
        ),
        "hallmark": (
            "CLN3 HALLMARKS: "
            "(1) MOST COMMON NCL: Juvenile NCL / Batten disease; CLN3 accounts for majority of NCL; "
            "(2) 1.02 kb DELETION — PCR FIRST, NOT WES: 73-85% of CLN3 alleles carry this deletion; "
            "targeted PCR detects deletion; WES will MISS it; always PCR first; "
            "(3) VACUOLATED LYMPHOCYTES ON BLOOD SMEAR — DIAGNOSTIC: order peripheral blood smear; "
            "Giemsa stain; clear cytoplasmic vacuoles in lymphocytes; quick cheap test; "
            "(4) VISUAL FAILURE AS FIRST SYMPTOM: rod-cone dystrophy + macular degeneration; "
            "ERG abnormal early; ophthalmologist often first doctor to suspect NCL; "
            "(5) FINGERPRINT PROFILES ON EM: concentric whorled lamellar structure; "
            "(6) NO APPROVED ERT: unlike CLN2, no disease-modifying therapy for CLN3; "
            "(7) PSYCHIATRIC FEATURES: behavioural/psychiatric symptoms common in JNCL; "
            "(8) ARGININE TRANSPORT: CLN3 may be lysosomal arginine transporter — therapeutic target"
        ),
        "key_ddx": (
            "CLN3 DDx: "
            "(1) CLN2/TPP1: earlier onset (2-4y); seizures first (not vision); CB+FP on EM; TPP1 enzyme reduced; "
            "(2) CLN4B/DNAJC5: adult onset (20-50y); dominant; no vision loss; "
            "(3) Stargardt disease (ABCA4): isolated macular degeneration; no seizures; no vacuolated lymphocytes; "
            "(4) Leber congenital amaurosis: early blindness; no seizures; no storage material; "
            "(5) Juvenile Huntington: CAG repeat; chorea; no retinal dystrophy"
        ),
        "diet_treatment": (
            "No approved disease-modifying therapy for CLN3. Supportive: "
            "antiepileptics (LEV preferred; clonazepam for myoclonus; avoid VPA); "
            "visual aids (low vision support); behavioural/psychiatric management; "
            "physiotherapy; PEG tube for dysphagia; palliative care. "
            "Gene therapy: AAV-CLN3 trials ongoing (intrathecal/ICV delivery). "
            "Cysteamine: clinical trial NCT02274987; modest benefit signal."
        ),
        "gene_therapy_status": (
            "AAV9-CLN3 gene therapy: Phase I/II clinical trials (NCT03770572); intrathecal delivery; "
            "preclinical benefit in Cln3-knockout mice. CLN3 cDNA (438aa) fits AAV; "
            "challenge is transmembrane protein — membrane insertion must be preserved. "
            "RNA therapy (antisense oligonucleotides to skip mutant exon) under study."
        ),
        "critical_ci": (
            "CRITICAL: (1) Ordering WES without PCR first — 1.02kb deletion is missed by WES; "
            "targeted PCR for the deletion is step 1 if CLN3 suspected; "
            "(2) Not ordering blood smear — vacuolated lymphocytes are DIAGNOSTIC and FREE; "
            "(3) Missing retinal dystrophy — vision loss is CLN3's first symptom; "
            "(4) Misdiagnosis as primary psychiatric disorder — behavioural features early in JNCL; "
            "(5) Assuming CLN3 = CLN2 management — no ERT available for CLN3"
        ),
        "nbs_marker": (
            "No standard NBS for CLN3. Targeted PCR for 1.02kb deletion in suspected cases. "
            "Peripheral blood smear: vacuolated lymphocytes (diagnostic). "
            "CLN3 sequencing (after deletion PCR)."
        ),
        "key_biomarker": (
            "Peripheral blood smear: vacuolated lymphocytes DIAGNOSTIC. Targeted PCR: 1.02kb deletion "
            "(73-85% of alleles). EM: fingerprint profiles. ERG: rod-cone dystrophy pattern. "
            "CLN3 sequencing. Autofluorescent storage material (UV microscopy of lymphocytes)."
        ),
        "severity_spectrum": (
            "Classic JNCL: onset 4-7y visual failure; death 15-35y. "
            "Variant JNCL (non-deletion alleles): variable course. "
            "Genotype-phenotype: homozygous 1.02kb deletion = classic JNCL; compound heterozygotes vary."
        ),
        "founder_variant": "1.02 kb genomic deletion (chr16:28,532,191-28,533,199, exons 7-8): found in 73-85% of all CLN3 disease alleles worldwide; not ethnically restricted; most common NCL mutation globally.",
        "key_variants": ["1.02kb deletion exons 7-8 (73-85% of alleles)", "p.Glu295Lys", "p.Val330Leu", "p.Asp366Asn", "c.1056del (frameshift)"],
        "seed": SEED_BASE + 2,
    },

    # ── CLN4B (DNAJC5) — Adult NCL / Parry disease / Kufs type A ────────────────
    {
        "gene": "CLN4B", "protein": "DNAJC5",
        "alias": "CLN4B/DNAJC5 — Adult NCL / Parry disease / Kufs disease type A (OMIM #162350)",
        "aa": "198 aa", "kDa": "22 kDa",
        "gene_class": "HSP40 co-chaperone (CSPα / cysteine-string protein alpha) — J-domain co-chaperone; assists HSP70 in synaptic vesicle protein refolding; prevents aggregation of misfolded proteins at presynaptic terminals",
        "ncl_subgroup": "Adult NCL (CLN4B) — ONLY AUTOSOMAL DOMINANT NCL",
        "locus": "20q13.33", "omim_gene": 611203,
        "inheritance": "AUTOSOMAL DOMINANT. 20q13.33. Gain-of-toxic-function mechanism. p.Leu115Arg + p.Phe116Ile are the two known pathogenic dominant variants. Family history across generations.",
        "em_finding": "Fingerprint profiles (FP) — concentric whorled profiles; same as CLN3 but in adult-onset context",
        "onset_range_y": (20.0, 50.0),
        "phenotype": (
            "Adult NCL / Kufs disease type A / Parry disease: ONLY AUTOSOMAL DOMINANT NCL; onset 20-50y; "
            "progressive myoclonic epilepsy + dementia + cerebellar ataxia; "
            "VISUAL SYSTEM SPARED (no vision loss — KEY DDx from all AR NCL types); "
            "family history across generations"
        ),
        "disease": (
            "CLN4B (DNAJC5) encodes cysteine-string protein alpha (CSPα), an HSP40 co-chaperone "
            "of 198aa (22kDa). CSPα localises to synaptic vesicle membranes via palmitoylation of "
            "its cysteine-string region. It recruits cytosolic HSC70 (HSPA8) + SGT to form a "
            "trimeric chaperone complex at presynaptic terminals, preventing aggregation of misfolded "
            "synaptic proteins (particularly SNARE complex proteins and SNAP25). "
            "CSPα also facilitates membrane insertion of client proteins. "
            "CLN4B pathogenic variants (p.Leu115Arg and p.Phe116Ile) act by GAIN OF TOXIC FUNCTION: "
            "mutant CSPα misfolds → self-aggregates → sequesters wild-type CSPα + other chaperones "
            "→ failure of synaptic protein quality control → neuronal death. "
            "ONLY AUTOSOMAL DOMINANT NCL GENE: unique among all NCL forms (CLN1-10 are AR except CLN4B). "
            "Only two known pathogenic variants: p.Leu115Arg (more common; Parry family original) "
            "and p.Phe116Ile (fewer reported families). Both in the cysteine-string region. "
            "Clinical features: adult onset (20-50y); progressive myoclonic epilepsy; "
            "cerebellar ataxia; dementia; behavioural changes; parkinsonian features. "
            "KEY: NO VISION LOSS in CLN4B — visual system spared; distinguishes CLN4B from all "
            "other NCL types (CLN1-3, CLN5-10 all cause visual failure). "
            "EM: fingerprint profiles (same as CLN3); lymphocyte vacuoles ABSENT (unlike CLN3). "
            "No approved therapy. Prognosis: progressive over 5-15y after symptom onset. "
            "Incidence: very rare; only ~40-50 families reported worldwide."
        ),
        "hallmark": (
            "CLN4B/DNAJC5 HALLMARKS: "
            "(1) ONLY AUTOSOMAL DOMINANT NCL: unique inheritance among all NCL genes; "
            "family history across multiple generations; "
            "(2) NO VISION LOSS — KEY DDx: visual system is SPARED in CLN4B; "
            "all other NCL types (CLN1-3, CLN5-10) cause visual failure; "
            "adult NCL without visual failure = test DNAJC5; "
            "(3) ADULT ONSET 20-50y: clinically resembles adult-onset progressive myoclonic epilepsy; "
            "(4) ONLY TWO KNOWN PATHOGENIC VARIANTS: p.Leu115Arg + p.Phe116Ile in cysteine-string region; "
            "targeted testing sufficient; "
            "(5) GAIN-OF-TOXIC-FUNCTION: dominant negative/toxic gain mechanism; CSPα aggregates; "
            "(6) FINGERPRINT PROFILES ON EM: same as CLN3 but in adult; no lymphocyte vacuoles; "
            "(7) SYNAPTIC VESICLE CO-CHAPERONE: CSPα function at presynaptic terminals; "
            "(8) RARE: ~40-50 families worldwide; seek specialist NCL centre"
        ),
        "key_ddx": (
            "CLN4B DDx: "
            "(1) CLN6 (Kufs type A also): AR; also adult NCL; rectilinear profiles (not FP); "
            "(2) Unverricht-Lundborg (EPM1, CSTB): progressive myoclonic epilepsy; AR; earlier onset (6-15y); no dementia early; "
            "(3) MERRF (mitochondrial, MT-TK): ragged-red fibres; elevated lactate; mtDNA; "
            "(4) Lafora disease (EPM2A/EPM2B): Lafora bodies; AR; younger onset; "
            "(5) GSS/CJD (prion): no dominant family history matching CLN4B; rapid prion course"
        ),
        "diet_treatment": (
            "No approved disease-modifying therapy for CLN4B. Symptomatic: "
            "antiepileptics (LEV, clonazepam, piracetam for myoclonus); "
            "behavioural/psychiatric management; physiotherapy; "
            "palliative care. Chaperone-targeted therapies under investigation (e.g., HSP70 "
            "inducers — geldanamycin analogues)."
        ),
        "gene_therapy_status": (
            "No gene therapy approved or in clinical trials. DNAJC5 (198aa) is small and AAV-amenable. "
            "Challenge: gain-of-toxic-function mechanism means gene replacement alone insufficient; "
            "must also silence the mutant allele (allele-specific siRNA or ASO). "
            "HSP70 overexpression strategy (compensate for lost chaperone capacity): preclinical study."
        ),
        "critical_ci": (
            "CRITICAL: (1) Missing dominant inheritance — CLN4B is AD; check family history carefully; "
            "(2) Missing the 'no vision loss' clue — adult NCL without visual failure = test DNAJC5; "
            "(3) Assuming only CLN1-3 exist — adult NCL is underdiagnosed; CLN4B/CLN6 in adults; "
            "(4) Ordering full NCL panel when only two variants exist — targeted p.Leu115Arg/p.Phe116Ile first; "
            "(5) Missing CLN6 in DDx — Kufs type A phenotype also seen in CLN6 (AR)"
        ),
        "nbs_marker": (
            "No NBS. Diagnosis: adult-onset myoclonic epilepsy + dementia + family history (dominant) "
            "+ EM fingerprint profiles + DNAJC5 targeted sequencing (p.Leu115Arg or p.Phe116Ile). "
            "No enzyme assay available."
        ),
        "key_biomarker": (
            "DNAJC5 sequencing: targeted p.Leu115Arg + p.Phe116Ile. EM: fingerprint profiles. "
            "Absence of lymphocyte vacuoles (distinguishes from CLN3). Absence of visual failure "
            "(key DDx). MRI brain: cerebellar/cortical atrophy. EEG: myoclonic epilepsy pattern."
        ),
        "severity_spectrum": (
            "Adult onset 20-50y; progressive myoclonic epilepsy + ataxia + dementia; "
            "progression over 5-15y after symptom onset; wheelchair by 10-15y from onset; "
            "death typically in 50-70y range depending on onset age."
        ),
        "founder_variant": "p.Leu115Arg (original Parry family, first described in USA) and p.Phe116Ile — only two known pathogenic DNAJC5 variants. Both in the cysteine-string region (palmitoylation domain). No ethnic founder enrichment.",
        "key_variants": ["p.Leu115Arg (Parry disease, most common)", "p.Phe116Ile (second variant)"],
        "seed": SEED_BASE + 3,
    },

    # ── CLN5 — Finnish Late-Infantile Variant NCL ────────────────────────────────
    {
        "gene": "CLN5", "protein": "CLN5",
        "alias": "CLN5 — Finnish late-infantile NCL variant (OMIM #256731)",
        "aa": "407 aa", "kDa": "60 kDa (glycosylated), 47 kDa (unglycosylated); soluble form also secreted",
        "gene_class": "Lysosomal transmembrane protein (also exists as soluble secreted form) — function not fully characterised; roles in lysosomal sorting and membrane trafficking",
        "ncl_subgroup": "Late-Infantile NCL variant (CLN5)",
        "locus": "13q22.3", "omim_gene": 608102,
        "inheritance": "AR. 13q22.3. Both sexes equally. Finnish founder p.Tyr392X (c.1175G>A), accounting for ~94% of Finnish CLN5 alleles.",
        "em_finding": "Mixed curvilinear bodies (CB) + fingerprint profiles (FP) + rectilinear profiles — heterogeneous mixture on EM",
        "onset_range_y": (4.0, 7.0),
        "phenotype": (
            "Finnish late-infantile NCL variant: onset 4-7y; visual failure + cognitive decline + "
            "seizures + ataxia + motor deterioration; Finnish founder p.Tyr392X (94% Finnish alleles); "
            "mixed CB+FP+rectilinear on EM; death typically 14-25y"
        ),
        "disease": (
            "CLN5 encodes a 407aa protein that exists in two forms: a transmembrane lysosomal "
            "membrane protein and a soluble secreted form (signal peptide cleavage). Both forms "
            "are glycosylated. CLN5 localises to lysosomes and is partially secreted. "
            "The precise molecular function of CLN5 remains incompletely understood. "
            "Proposed roles: lysosomal protein sorting (interacts with sortilin/GGA proteins), "
            "membrane trafficking (late endosome-lysosome fusion), and lysosomal enzyme delivery. "
            "CLN5 deficiency → accumulation of ceroid lipopigment with mixed ultrastructure. "
            "EM: heterogeneous — curvilinear bodies (CB) + fingerprint profiles (FP) + rectilinear "
            "profiles; the mixed pattern is characteristic of CLN5. "
            "FINNISH VARIANT NCL: CLN5 mutations were first identified in Finnish patients, hence "
            "'Finnish variant late-infantile NCL.' Finnish founder variant p.Tyr392X (c.1175G>A) "
            "accounts for ~94% of CLN5 alleles in Finnish patients. Outside Finland, CLN5 occurs "
            "globally with diverse variants. "
            "Clinical: onset 4-7y; visual failure early (similar to CLN3 but earlier); "
            "seizures (myoclonic, generalised); cerebellar ataxia; cognitive decline; "
            "slower progression than CLN1/CLN2; death typically 14-25y. "
            "No approved ERT or gene therapy. Incidence: highest in Finland (~1/100,000); "
            "rare globally."
        ),
        "hallmark": (
            "CLN5 HALLMARKS: "
            "(1) FINNISH VARIANT LATE-INFANTILE NCL: historically called 'Finnish variant'; "
            "Finnish founder p.Tyr392X in 94% of Finnish CLN5 alleles; "
            "(2) MIXED CB+FP+RECTILINEAR EM: heterogeneous ultrastructure; "
            "(3) ONSET 4-7y: slightly later than CLN2 (2-4y); similar to CLN3 but with more "
            "severe early course; "
            "(4) SOLUBLE SECRETED FORM: CLN5 is partially secreted — therapeutic implication "
            "(enzyme replacement might be feasible via secreted protein); "
            "(5) VISUAL FAILURE EARLY: macular degeneration + retinal dystrophy; "
            "(6) CEREBELLAR ATAXIA: prominent in CLN5 vs CLN3; "
            "(7) SLOWER THAN CLN1/CLN2: survival to 14-25y"
        ),
        "key_ddx": (
            "CLN5 DDx: "
            "(1) CLN2/TPP1: earlier onset (2-4y); CB+FP (not rectilinear component); approved ERT; TPP1 enzyme; "
            "(2) CLN3: similar onset (4-10y); pure fingerprint profiles; vacuolated lymphocytes; 1.02kb deletion; "
            "(3) CLN6: similar onset + rectilinear profiles; CLN6 EM = rectilinear predominantly; AR; "
            "(4) CLN7/MFSD8: Turkish variant; similar age; mixed CB+FP; Turkish founder p.Tyr319Cys"
        ),
        "diet_treatment": (
            "No approved disease-modifying therapy. Supportive: antiepileptics (LEV preferred), "
            "visual aids, physiotherapy, cognitive/speech therapy, PEG feeding, palliative care. "
            "Gene therapy: AAV9-CLN5 trials underway."
        ),
        "gene_therapy_status": (
            "AAV9-CLN5 gene therapy: preclinical benefit in CLN5-null sheep model (natural CLN5 model). "
            "Phase I/II trial initiated. The secreted form of CLN5 offers potential cross-correction "
            "(secreted enzyme taken up by neighbouring cells — similar ERT rationale to lysosomal enzymes)."
        ),
        "critical_ci": (
            "CRITICAL: (1) Missing Finnish founder — test p.Tyr392X first in Finnish patients; "
            "(2) Assuming CLN5 = CLN2 — no ERT available for CLN5; "
            "(3) Missing mixed EM ultrastructure — CLN5 has CB+FP+rectilinear (not pure CB or FP); "
            "(4) Not checking opthalmology — visual failure is early in CLN5"
        ),
        "nbs_marker": (
            "No standard NBS for CLN5. Diagnosis: clinical + EM (mixed ultrastructure) + CLN5 sequencing. "
            "Finnish patients: targeted p.Tyr392X (c.1175G>A) first. "
            "No enzyme assay available (CLN5 function unknown)."
        ),
        "key_biomarker": (
            "CLN5 sequencing (targeted p.Tyr392X in Finnish). EM: mixed CB+FP+rectilinear profiles. "
            "ERG: retinal dystrophy. Autofluorescent storage material. No enzyme assay available."
        ),
        "severity_spectrum": (
            "Onset 4-7y; progressive neurodegeneration; loss of ambulation typically by 10-15y; "
            "death 14-25y. Slower progression than CLN1/CLN2; somewhat similar to CLN3."
        ),
        "founder_variant": "p.Tyr392X (c.1175G>A) — Finnish founder; ~94% of Finnish CLN5 alleles; incidence in Finland ~1/100,000.",
        "key_variants": ["p.Tyr392X (Finnish founder, 94% Finnish)", "p.Trp75X", "p.Ile247Asn", "p.Arg516X", "c.835-1G>A (splice)"],
        "seed": SEED_BASE + 4,
    },

    # ── CLN6 — Late-Infantile/Adult NCL variant ──────────────────────────────────
    {
        "gene": "CLN6", "protein": "CLN6",
        "alias": "CLN6 — Late-infantile/adult NCL variant; Kufs disease type A (adult CLN6) (OMIM #601780)",
        "aa": "311 aa", "kDa": "36 kDa",
        "gene_class": "ER-resident type I transmembrane protein — function uncertain; localised to endoplasmic reticulum (ER), NOT lysosomes; roles proposed in lysosomal enzyme sorting and ER quality control",
        "ncl_subgroup": "Late-Infantile NCL variant / Adult NCL Kufs (CLN6)",
        "locus": "15q23", "omim_gene": 606725,
        "inheritance": "AR. 15q23. Both sexes equally. p.Trp263Cys enriched in Newfoundland/Portuguese populations.",
        "em_finding": "Rectilinear profiles (RL) — straight parallel membrane stacks; characteristic of CLN6",
        "onset_range_y": (1.5, 8.0),
        "phenotype": (
            "Late-infantile/adult NCL variant: childhood onset (18m-8y) in late-infantile form; "
            "adult onset (20-50y) for Kufs type A form (CLN6); ER-resident protein (unique among NCL); "
            "rectilinear profiles on EM; p.Trp263Cys Newfoundland/Portuguese enrichment; "
            "visual failure + seizures + ataxia"
        ),
        "disease": (
            "CLN6 encodes a 311aa (36kDa) ER-resident type I transmembrane protein. "
            "UNIQUE AMONG NCL GENES: CLN6 localises to the endoplasmic reticulum (ER), NOT to lysosomes. "
            "This is mechanistically puzzling because NCL storage material accumulates in lysosomes — "
            "CLN6 must affect lysosomal function indirectly (e.g., via lysosomal enzyme sorting at "
            "the ER or ER-Golgi trafficking). Proposed function: ER quality control of lysosomal "
            "hydrolases before their Golgi-dependent mannose-6-phosphate tagging and lysosomal delivery. "
            "CLN6 deficiency → ceroid lipopigment accumulation (subunit c of ATP synthase + sphingolipid "
            "activator proteins). "
            "EM: rectilinear profiles (RL) — parallel straight stacks of membranes; characteristic. "
            "CLN6 causes TWO DISTINCT PHENOTYPES: "
            "1. Late-infantile NCL (childhood): onset 18m-8y; visual failure + seizures + ataxia + "
            "cognitive decline; death 5-12y after onset. "
            "2. Kufs disease type A (adult form): onset 20-50y; progressive myoclonic epilepsy + "
            "dementia (similar to CLN4B Kufs type A, but CLN6 is AR); no vision loss in adult form. "
            "p.Trp263Cys: enriched in Newfoundland (Canada) and Portuguese populations. "
            "Incidence: rare; estimated 1/200,000-1,000,000."
        ),
        "hallmark": (
            "CLN6 HALLMARKS: "
            "(1) ER-RESIDENT PROTEIN — UNIQUE AMONG NCL: CLN6 is in the ER, not lysosomes; "
            "the only ER-localised NCL gene; mechanistically distinct from other NCL genes; "
            "(2) RECTILINEAR PROFILES ON EM: straight parallel membrane stacks; characteristic for CLN6; "
            "(3) KUFS TYPE A ADULT FORM (AR): CLN6 causes adult NCL (Kufs type A) with AR inheritance; "
            "distinguish from CLN4B (Kufs type A, dominant); EM rectilinear vs fingerprint; "
            "(4) TWO PHENOTYPES FROM ONE GENE: late-infantile (childhood) + adult (Kufs); "
            "(5) p.Trp263Cys NEWFOUNDLAND/PORTUGUESE: regional enrichment; targeted testing; "
            "(6) VISUAL FAILURE IN CHILDHOOD FORM: retinal dystrophy + optic atrophy; "
            "adult Kufs CLN6 may spare vision"
        ),
        "key_ddx": (
            "CLN6 DDx: "
            "(1) CLN5: similar age of onset; mixed CB+FP+rectilinear (not purely rectilinear); Finnish p.Tyr392X; "
            "(2) CLN7/MFSD8: similar onset; mixed CB+FP; Turkish founder; MFS transporter (different from ER protein); "
            "(3) CLN4B: adult NCL (Kufs A); AD inheritance (not AR); fingerprint profiles (not rectilinear); p.Leu115Arg/p.Phe116Ile only; "
            "(4) CLN2/TPP1: earlier onset (2-4y); approved ERT; CB+FP; TPP1 enzyme reduced"
        ),
        "diet_treatment": (
            "No approved disease-modifying therapy. Supportive: antiepileptics (LEV preferred), "
            "visual aids, physiotherapy, speech therapy, PEG feeding, palliative care. "
            "Gene therapy (AAV-CLN6): preclinical benefit; Phase I/II trial for late-infantile CLN6 "
            "(NCT02725580 — completed; ongoing follow-up)."
        ),
        "gene_therapy_status": (
            "AAV9-CLN6 (intrathecal/ICV): Phase I/II trial NCT02725580 at Nationwide Children's Hospital — "
            "first intrathecal AAV9 NCL trial; safety signal positive; efficacy data published. "
            "CLN6 cDNA (311aa) fits AAV. Challenge: ER-resident protein delivery."
        ),
        "critical_ci": (
            "CRITICAL: (1) Confusing CLN6 Kufs-A with CLN4B Kufs-A — both cause adult NCL Kufs type A; "
            "CLN6 is AR, CLN4B is AD; rectilinear (CLN6) vs fingerprint (CLN4B) on EM; "
            "(2) Assuming CLN6 = lysosomal — CLN6 is ER-resident; "
            "(3) Missing p.Trp263Cys in Newfoundland/Portuguese patients; "
            "(4) No enzyme assay for CLN6 — molecular diagnosis only"
        ),
        "nbs_marker": (
            "No NBS for CLN6. Diagnosis: EM (rectilinear profiles) + CLN6 sequencing. "
            "No enzyme assay available. Targeted p.Trp263Cys in Newfoundland/Portuguese patients."
        ),
        "key_biomarker": (
            "CLN6 sequencing (targeted p.Trp263Cys in Newfoundland/Portuguese). "
            "EM: rectilinear profiles. ERG: retinal dystrophy (childhood form). "
            "Autofluorescent storage material. No enzyme assay."
        ),
        "severity_spectrum": (
            "Late-infantile CLN6: onset 18m-8y; death 5-12y after onset. "
            "Adult Kufs CLN6: onset 20-50y; slower progression than childhood form; "
            "survival 10-20y after symptom onset."
        ),
        "founder_variant": "p.Trp263Cys — Newfoundland (Canada) and Portuguese population enrichment. No single dominant global founder.",
        "key_variants": ["p.Trp263Cys (Newfoundland/Portuguese)", "p.Tyr198X", "p.Arg252X", "c.316insC (frameshift)", "p.Gly123Glu"],
        "seed": SEED_BASE + 5,
    },

    # ── CLN7 (MFSD8) — Late-Infantile Turkish Variant NCL ───────────────────────
    {
        "gene": "CLN7", "protein": "MFSD8",
        "alias": "CLN7/MFSD8 — Turkish late-infantile NCL variant (TCLN) (OMIM #610951)",
        "aa": "518 aa", "kDa": "58 kDa",
        "gene_class": "Major facilitator superfamily domain-containing protein 8 (MFSD8) — lysosomal membrane transporter; MFS superfamily; substrate unknown; roles in lysosomal membrane integrity and ion/metabolite transport",
        "ncl_subgroup": "Late-Infantile NCL Turkish variant (CLN7)",
        "locus": "4q28.2", "omim_gene": 611124,
        "inheritance": "AR. 4q28.2. Both sexes equally. p.Tyr319Cys Turkish founder — prevalent in Turkey and surrounding region.",
        "em_finding": "Mixed curvilinear bodies (CB) + fingerprint profiles (FP) — similar to CLN5 but in Turkish clinical context",
        "onset_range_y": (2.0, 7.0),
        "phenotype": (
            "Turkish late-infantile NCL (TCLN): onset 2-7y; seizures + visual failure + cognitive "
            "decline + ataxia; Turkish founder p.Tyr319Cys; lysosomal MFS transporter; "
            "mixed CB+FP on EM; death typically 8-20y"
        ),
        "disease": (
            "CLN7 encodes MFSD8 (major facilitator superfamily domain-containing protein 8), "
            "a 518aa (58kDa) lysosomal membrane transporter belonging to the major facilitator "
            "superfamily (MFS) — a large family of secondary active transporters. "
            "MFSD8 localises to the lysosomal membrane. Its substrate is unknown, but it likely "
            "transports a lysosomal metabolite or ion across the lysosomal membrane. "
            "MFSD8 deficiency → accumulation of ceroid lipopigment; lysosomal storage. "
            "EM: mixed CB (curvilinear bodies) + FP (fingerprint profiles); similar pattern to CLN5. "
            "FIRST IDENTIFIED IN TURKISH PATIENTS: CLN7 mutations were first identified in Turkish "
            "patients with late-infantile NCL — hence 'Turkish variant late-infantile NCL (TCLN).' "
            "Turkish founder variant p.Tyr319Cys: enriched in Turkish population; accounts for "
            "a significant proportion of CLN7 alleles in Turkey. "
            "Clinical: onset 2-7y; seizures (myoclonic, generalised); visual failure (retinal "
            "dystrophy, optic atrophy); cerebellar ataxia; cognitive decline; motor deterioration; "
            "death typically 8-20y. Progression is intermediate between CLN2 and CLN3. "
            "CLN7/MFSD8 mutations now identified in many non-Turkish populations globally. "
            "Incidence: rare globally; highest in Turkey."
        ),
        "hallmark": (
            "CLN7/MFSD8 HALLMARKS: "
            "(1) TURKISH LATE-INFANTILE VARIANT: originally described in Turkish patients; "
            "Turkish founder p.Tyr319Cys; now found globally; "
            "(2) LYSOSOMAL MFS TRANSPORTER: MFSD8 is a transporter family protein in lysosomal membrane; "
            "substrate unknown — active research; "
            "(3) MIXED CB+FP ON EM: similar to CLN5; distinguishing between CLN5 and CLN7 requires "
            "molecular diagnosis; "
            "(4) VISUAL FAILURE + SEIZURES: onset 2-7y; similar clinical course to other late-infantile NCL; "
            "(5) NO APPROVED ERT: supportive treatment only; gene therapy preclinical"
        ),
        "key_ddx": (
            "CLN7 DDx: "
            "(1) CLN2/TPP1: similar onset (2-4y); CB+FP on EM; APPROVED ERT (cerliponase alfa); TPP1 enzyme reduced; "
            "(2) CLN5: mixed CB+FP+rectilinear; Finnish founder p.Tyr392X; similar EM but different gene; "
            "(3) CLN6: rectilinear profiles (not mixed CB+FP); ER-resident protein; "
            "(4) CLN3/JNCL: fingerprint only; older onset (4-10y); vacuolated lymphocytes"
        ),
        "diet_treatment": (
            "No approved disease-modifying therapy. Supportive: antiepileptics (LEV preferred), "
            "visual aids, physiotherapy, speech therapy, PEG feeding, palliative care. "
            "Gene therapy (AAV-MFSD8): preclinical studies ongoing."
        ),
        "gene_therapy_status": (
            "AAV9-MFSD8 gene therapy: preclinical studies in MFSD8-knockout mice; "
            "ICV delivery. MFSD8 cDNA (518aa) fits AAV. No clinical trials yet. "
            "The unknown substrate of MFSD8 complicates the development of alternative strategies "
            "(substrate reduction, enzyme replacement not applicable to membrane transporters)."
        ),
        "critical_ci": (
            "CRITICAL: (1) Missing CLN7 in non-Turkish patients — now found globally; "
            "(2) Assuming ERT available — no approved therapy for CLN7; "
            "(3) Confusing CLN7 with CLN2 EM — both have CB+FP; molecular diagnosis essential; "
            "(4) Missing p.Tyr319Cys targeted test in Turkish patients first"
        ),
        "nbs_marker": (
            "No NBS for CLN7. Diagnosis: EM (mixed CB+FP) + CLN7/MFSD8 sequencing. "
            "Turkish patients: targeted p.Tyr319Cys first. No enzyme assay."
        ),
        "key_biomarker": (
            "CLN7/MFSD8 sequencing (targeted p.Tyr319Cys in Turkish patients). "
            "EM: mixed CB+FP profiles. ERG: retinal dystrophy. "
            "Autofluorescent storage material. No enzyme assay."
        ),
        "severity_spectrum": (
            "Onset 2-7y; progressive neurodegeneration; intermediate severity; "
            "death 8-20y depending on onset age and variant. "
            "p.Tyr319Cys homozygotes: classic Turkish late-infantile course."
        ),
        "founder_variant": "p.Tyr319Cys — Turkish founder variant; prevalent in Turkish patients with late-infantile NCL; also reported in surrounding regions (Middle East).",
        "key_variants": ["p.Tyr319Cys (Turkish founder)", "p.Glu336X", "p.Arg436X", "c.103C>T p.Arg35X", "c.881+1G>A (splice)"],
        "seed": SEED_BASE + 6,
    },

    # ── CLN10 (CTSD) — Congenital NCL ───────────────────────────────────────────
    {
        "gene": "CLN10", "protein": "CTSD",
        "alias": "CLN10/CTSD — Congenital NCL; cathepsin D deficiency (OMIM #610127)",
        "aa": "412 aa", "kDa": "52 kDa (precursor); 34+14 kDa (mature two-chain form)",
        "gene_class": "Lysosomal aspartyl protease: cathepsin D — major lysosomal endopeptidase; ubiquitous expression; degrades denatured proteins in the lysosomal lumen; activates other lysosomal enzymes",
        "ncl_subgroup": "Congenital NCL (CLN10) — earliest/most severe NCL",
        "locus": "11p15.5", "omim_gene": 116840,
        "inheritance": "AR (biallelic loss-of-function or severe missense). 11p15.5. Both sexes equally. p.Trp383Cys and frameshift nulls reported.",
        "em_finding": "GRODs (Granular Osmiophilic Deposits) — same as CLN1/PPT1; electron-dense granular deposits in lysosomes",
        "onset_range_y": (0.0, 0.5),
        "phenotype": (
            "Congenital NCL: EARLIEST onset of all NCL types (congenital or early neonatal); "
            "HYPERTROPHIC CARDIOMYOPATHY (HCM) unique among all NCL genes; "
            "GRODs on EM (same as CLN1); cathepsin D deficiency; extremely rare; "
            "rapidly fatal"
        ),
        "disease": (
            "CLN10 encodes cathepsin D (CTSD), a 412aa lysosomal aspartyl protease. "
            "Cathepsin D is synthesised as a 52kDa precursor, processed in the Golgi and lysosomes "
            "to a mature two-chain form (34kDa + 14kDa linked by disulfide). It is the major "
            "lysosomal endopeptidase, degrading denatured proteins and activating other lysosomal "
            "enzymes (e.g., procathepsin B, procathepsin L). "
            "Cathepsin D deficiency → failure of lysosomal protein degradation → accumulation of "
            "ceroid lipopigment. EM: GRODs (granular osmiophilic deposits) — SAME as CLN1/PPT1. "
            "GRODs in congenital-onset NCL = test both PPT1 (CLN1) and CTSD (CLN10). "
            "CLN10 is the CONGENITAL FORM of NCL — the earliest and most severe: "
            "presentation may be in utero (fetal brain malformations detected on ultrasound), "
            "at birth, or in the first weeks of life. "
            "HYPERTROPHIC CARDIOMYOPATHY (HCM): unique to CLN10 among all NCL genes; "
            "HCM is present in the majority of CLN10 patients and is not a feature of any other NCL gene; "
            "HCM may be the presenting feature or detected on echocardiography. "
            "Clinical: seizures (typically within hours to days of birth); hypotonia; rapid neurological "
            "deterioration; HCM (cardiomyopathy); brain malformations (lissencephaly-like, simplified gyri); "
            "death typically within weeks to months. "
            "Extremely rare: fewer than 30 cases reported globally. "
            "Pathogenic variants: p.Trp383Cys (missense — reduces but does not abolish CTSD activity); "
            "frameshift/nonsense nulls cause most severe congenital phenotype. "
            "Note: CTSD heterozygous loss has also been associated with late-onset NCL and non-NCL disorders."
        ),
        "hallmark": (
            "CLN10/CTSD HALLMARKS: "
            "(1) CONGENITAL NCL — EARLIEST ONSET OF ALL NCL: in utero or neonatal presentation; "
            "most severe NCL; distinguishable by congenital onset; "
            "(2) HCM UNIQUE TO CLN10 AMONG NCL GENES: hypertrophic cardiomyopathy is seen in CLN10; "
            "NOT a feature of CLN1-7; HCM in neonatal NCL = test CTSD; echocardiogram mandatory; "
            "(3) GRODs ON EM (same as CLN1): granular osmiophilic deposits; "
            "GRODs in congenital setting = both CLN1 enzyme assay and CTSD sequencing; "
            "(4) BRAIN MALFORMATIONS: lissencephaly-like or simplified gyri on MRI/pathology; "
            "(5) CATHEPSIN D DEFICIENCY: CTSD enzyme activity (DBS or fibroblasts) severely reduced; "
            "(6) EXTREMELY RARE: <30 cases worldwide; "
            "(7) RAPIDLY FATAL: death within weeks to months of birth in classic congenital form"
        ),
        "key_ddx": (
            "CLN10 DDx: "
            "(1) CLN1/PPT1: GRODs also; but onset 6-24m (not congenital); no HCM; PPT1 enzyme reduced; "
            "(2) Pompe disease (GAA): HCM; but storage of glycogen (not ceroid); enzyme assay (alpha-glucosidase); "
            "(3) Danon disease (LAMP2): HCM + skeletal myopathy + intellectual disability; LAMP2 gene; "
            "(4) Barth syndrome (TAZ): HCM + cardiomyopathy; mitochondrial; acylcarnitines; no ceroid; "
            "(5) Neonatal-onset metabolic disorders (organic acidemias): metabolic acidosis; "
            "specific enzyme/metabolite abnormalities; no GRODs"
        ),
        "diet_treatment": (
            "No approved therapy. Supportive: neonatal intensive care; cardiac management for HCM; "
            "antiepileptics (phenobarbital/LEV for neonatal seizures); palliative care. "
            "Given rapid lethality, treatment is largely supportive. "
            "Experimental: recombinant CTSD (cathepsin D enzyme replacement) under development; "
            "gene therapy AAV-CTSD in neonatal mouse models."
        ),
        "gene_therapy_status": (
            "Recombinant CTSD (enzyme replacement): under investigation; CTSD is naturally secreted "
            "and can be taken up by cells via M6P receptors — ERT approach feasible in principle. "
            "AAV9-CTSD: neonatal mouse studies. No clinical trials due to extreme rarity and rapid lethality."
        ),
        "critical_ci": (
            "CRITICAL: (1) Missing HCM in neonatal NCL workup — echocardiogram mandatory in any "
            "neonatal presentation with suspected NCL; HCM is specific to CLN10; "
            "(2) Confusing GRODs with CLN1 only — GRODs in congenital setting = both PPT1 and CTSD; "
            "(3) Missing CTSD enzyme assay — perform alongside PPT1 in neonatal GRODs; "
            "(4) Not recognising extreme rarity — <30 cases worldwide; specialist referral mandatory"
        ),
        "nbs_marker": (
            "No NBS for CLN10. Diagnosis: neonatal seizures + HCM + GRODs on EM + CTSD enzyme activity "
            "(severely reduced) + CTSD sequencing. Prenatal: fetal brain malformations on ultrasound "
            "may prompt testing if family history."
        ),
        "key_biomarker": (
            "CTSD enzyme activity (DBS/fibroblasts/leukocytes): severely reduced. EM: GRODs. "
            "Echocardiography: HCM (unique to CLN10). MRI brain: lissencephaly/simplified gyri. "
            "CTSD sequencing (p.Trp383Cys and frameshift/null variants). "
            "Autofluorescent storage material."
        ),
        "severity_spectrum": (
            "Classic congenital CLN10: in utero / birth / neonatal onset; death weeks to months. "
            "Rarer late-infantile CLN10: onset 1-4y (partial CTSD activity); slower course. "
            "Genotype-phenotype: null alleles = congenital; p.Trp383Cys (partial activity) = variable."
        ),
        "founder_variant": "p.Trp383Cys — reduces (but does not abolish) CTSD activity; some cases with this variant have later onset. Frameshift/nonsense nulls cause most severe congenital phenotype. No ethnic founder.",
        "key_variants": ["p.Trp383Cys (missense, partial activity)", "c.764_765delAT (frameshift)", "p.Gln148X", "p.Ile231Ser", "c.895+1G>A (splice)"],
        "seed": SEED_BASE + 7,
    },
]


# ── Patient Generation ────────────────────────────────────────────────────────────
def _make_patients(gene_dict):
    """Generate 40 synthetic patient records for a given NCL gene."""
    rng = random.Random(gene_dict["seed"])
    gene = gene_dict["gene"]

    PHENO_PROBS = {
        "CLN1":   [0.70, 0.20, 0.10],   # Classic infantile / Late-infantile CLN1 / Adult CLN1
        "CLN2":   [0.75, 0.20, 0.05],   # Classic / Variant late-infantile / Juvenile CLN2
        "CLN3":   [0.80, 0.15, 0.05],   # Classic JNCL / Variant JNCL / Atypical
        "CLN4B":  [0.70, 0.20, 0.10],   # Classic Parry / Mild adult / Atypical
        "CLN5":   [0.75, 0.20, 0.05],   # Classic / Variant / Mild
        "CLN6":   [0.65, 0.25, 0.10],   # Late-infantile / Kufs-adult / Mild
        "CLN7":   [0.70, 0.25, 0.05],   # Classic Turkish variant / Variant / Mild
        "CLN10":  [0.65, 0.25, 0.10],   # Congenital / Late-infantile CLN10 / Partial
    }
    CLASS_NAMES = {
        "CLN1":   ["Classic Infantile CLN1", "Late-Infantile CLN1", "Juvenile/Adult CLN1"],
        "CLN2":   ["Classic Late-Infantile CLN2", "Variant Late-Infantile CLN2", "Juvenile CLN2"],
        "CLN3":   ["Classic JNCL (1.02kb del)", "Variant JNCL", "Atypical JNCL"],
        "CLN4B":  ["Classic Parry Disease", "Mild Adult CLN4B", "Atypical CLN4B"],
        "CLN5":   ["Classic Finnish Late-Infantile", "Variant CLN5", "Mild CLN5"],
        "CLN6":   ["Classic Late-Infantile CLN6", "Adult Kufs CLN6", "Mild CLN6"],
        "CLN7":   ["Classic Turkish Late-Infantile", "Variant CLN7", "Mild CLN7"],
        "CLN10":  ["Classic Congenital CLN10", "Late-Infantile CLN10", "Partial CTSD Deficiency"],
    }

    AGE_RANGES = {
        "CLN1":   [(0.5, 2.0), (2.0, 4.0), (4.0, 10.0)],
        "CLN2":   [(2.0, 4.0), (3.0, 7.0), (5.0, 10.0)],
        "CLN3":   [(4.0, 10.0), (4.0, 12.0), (6.0, 14.0)],
        "CLN4B":  [(20.0, 50.0), (25.0, 55.0), (30.0, 60.0)],
        "CLN5":   [(4.0, 7.0), (5.0, 9.0), (6.0, 12.0)],
        "CLN6":   [(1.5, 8.0), (18.0, 50.0), (3.0, 12.0)],
        "CLN7":   [(2.0, 7.0), (3.0, 8.0), (4.0, 10.0)],
        "CLN10":  [(0.0, 0.3), (0.5, 4.0), (1.0, 6.0)],
    }

    probs = PHENO_PROBS.get(gene, [0.60, 0.30, 0.10])
    classes = CLASS_NAMES.get(gene, ["Severe", "Moderate", "Mild"])

    patients = []
    for i in range(40):
        r = rng.random()
        if r < probs[0]:
            pheno_idx = 0
        elif r < probs[0] + probs[1]:
            pheno_idx = 1
        else:
            pheno_idx = 2
        pheno = classes[pheno_idx]

        age_range = AGE_RANGES.get(gene, [(2.0, 8.0), (5.0, 15.0), (10.0, 30.0)])[pheno_idx]
        age_dx = round(rng.uniform(*age_range), 1)

        sex = rng.choice(["M", "F"])

        # EM finding per gene
        em_map = {
            "CLN1": "GRODs",
            "CLN2": rng.choice(["Curvilinear bodies + fingerprint", "Curvilinear bodies"]),
            "CLN3": "Fingerprint profiles",
            "CLN4B": "Fingerprint profiles",
            "CLN5": rng.choice(["Mixed CB+FP+rectilinear", "Mixed CB+FP"]),
            "CLN6": "Rectilinear profiles",
            "CLN7": rng.choice(["Mixed CB+FP", "Curvilinear + fingerprint"]),
            "CLN10": "GRODs",
        }
        em = em_map.get(gene, "Mixed")

        # Visual failure (all except CLN4B)
        visual_failure = (gene != "CLN4B") and (rng.random() < 0.90 if pheno_idx == 0 else rng.random() < 0.75)

        # Seizures
        seizures = rng.random() < 0.92

        # Gene-specific features
        if gene == "CLN1":
            enzyme_reduced = True
            erp = rng.choice(["PPT1 severely reduced (DBS)", "PPT1 undetectable (leukocytes)", "PPT1 <2% normal"])
            presenting = rng.choice(["Developmental regression", "Visual failure + hypotonia", "Myoclonus onset", "Seizures + regression"])
            outcome = rng.choice(["Vegetative state by 3y", "Isoelectric EEG by 3y", "Death age 8-13y", "Progressive deterioration"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "protein": "PPT1",
                "sex": sex, "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting, "em_finding": em,
                "visual_failure": visual_failure, "seizures": seizures,
                "enzyme_assay": erp, "outcome": outcome,
                "gene_therapy_trial": rng.random() < 0.15,
            })
        elif gene == "CLN2":
            enzyme_reduced = True
            erp = rng.choice(["TPP1 severely reduced (DBS)", "TPP1 undetectable (leukocytes)", "TPP1 <3% normal"])
            presenting = rng.choice(["Seizures (first symptom)", "Language delay + seizures", "Myoclonic seizures 2-4y", "Febrile convulsions then epilepsy"])
            on_ert = rng.random() < 0.55   # Brineura availability
            outcome = rng.choice(["On cerliponase alfa (Brineura)", "Declined ERT (family)", "Motor decline slowed on ERT", "Pre-symptomatic ERT sibling"]) if on_ert else rng.choice(["Progressive motor decline", "Loss of ambulation 6y", "Death age 8-12y"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "protein": "TPP1",
                "sex": sex, "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting, "em_finding": em,
                "visual_failure": visual_failure, "seizures": seizures,
                "enzyme_assay": erp, "on_cerliponase_alfa": on_ert,
                "outcome": outcome,
            })
        elif gene == "CLN3":
            has_deletion = rng.random() < 0.83   # 73-85% alleles → ~83% of patients have at least one deletion
            vacuolated_lymphocytes = rng.random() < 0.95  # diagnostic
            presenting = rng.choice(["Visual failure (first symptom)", "Progressive visual loss", "Macular degeneration noted", "Myopic changes + regression"])
            outcome = rng.choice(["Progressive visual + cognitive decline", "Wheelchair by 15y", "Seizure-free on LEV", "Late-stage behavioural features"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "protein": "CLN3",
                "sex": sex, "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting, "em_finding": em,
                "visual_failure": visual_failure, "seizures": seizures,
                "has_1_02kb_deletion": has_deletion,
                "vacuolated_lymphocytes_smear": vacuolated_lymphocytes,
                "outcome": outcome,
            })
        elif gene == "CLN4B":
            variant = rng.choice(["p.Leu115Arg", "p.Phe116Ile"]) if rng.random() < 0.80 else "p.Leu115Arg"
            family_history = rng.random() < 0.80   # dominant; family history
            presenting = rng.choice(["Progressive myoclonic epilepsy", "Cognitive decline + myoclonus", "Cerebellar ataxia + dementia", "Dementia onset 30s-40s"])
            outcome = rng.choice(["Progressive myoclonic epilepsy", "Dementia + ataxia", "Wheelchair decade after onset", "Institutionalised care"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "protein": "DNAJC5/CSPα",
                "sex": sex, "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting, "em_finding": em,
                "visual_failure": False,   # CLN4B: visual system spared
                "seizures": seizures,
                "pathogenic_variant": variant,
                "family_history_dominant": family_history,
                "outcome": outcome,
            })
        elif gene == "CLN5":
            is_finnish = rng.random() < 0.35  # subset of CLN5 patients are Finnish
            founder = "p.Tyr392X (Finnish)" if is_finnish else rng.choice(["Other CLN5 variant", "p.Trp75X", "p.Arg516X"])
            presenting = rng.choice(["Visual failure + cognitive decline", "Seizures 4-7y", "Cerebellar ataxia onset", "Regression + visual loss"])
            outcome = rng.choice(["Progressive neurodegeneration", "Loss of ambulation 12y", "Death age 14-25y", "Late-stage wheelchair"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "protein": "CLN5",
                "sex": sex, "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting, "em_finding": em,
                "visual_failure": visual_failure, "seizures": seizures,
                "finnish_founder": is_finnish, "variant_note": founder,
                "outcome": outcome,
            })
        elif gene == "CLN6":
            kufs_adult = (pheno_idx == 1)  # adult Kufs form
            newf_portuguese = rng.random() < 0.20
            presenting = (
                rng.choice(["Progressive myoclonic epilepsy (adult)", "Dementia onset 30-40y", "Ataxia + cognitive decline"])
                if kufs_adult else
                rng.choice(["Seizures onset", "Visual failure + ataxia", "Cognitive regression"])
            )
            outcome = rng.choice(["Progressive course", "Adult Kufs progression", "Wheelchair decade from onset"]) if kufs_adult else rng.choice(["Death 5-12y after onset", "Progressive neurodegeneration", "Late-stage palliative"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "protein": "CLN6",
                "sex": sex, "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting, "em_finding": em,
                "visual_failure": visual_failure if not kufs_adult else rng.random() < 0.30,
                "seizures": seizures, "kufs_adult_form": kufs_adult,
                "newfoundland_portuguese": newf_portuguese,
                "outcome": outcome,
            })
        elif gene == "CLN7":
            turkish_patient = rng.random() < 0.55
            founder = "p.Tyr319Cys (Turkish)" if turkish_patient else rng.choice(["Other MFSD8 variant", "p.Glu336X", "p.Arg436X"])
            presenting = rng.choice(["Seizures 2-7y", "Visual failure + seizures", "Cerebellar ataxia onset", "Cognitive decline + myoclonus"])
            outcome = rng.choice(["Progressive neurodegeneration", "Loss of ambulation", "Death age 8-20y", "Late-stage wheelchair"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "protein": "MFSD8",
                "sex": sex, "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting, "em_finding": em,
                "visual_failure": visual_failure, "seizures": seizures,
                "turkish_founder": turkish_patient, "variant_note": founder,
                "outcome": outcome,
            })
        elif gene == "CLN10":
            hcm = rng.random() < 0.80  # HCM in most CLN10 patients
            congenital = (pheno_idx == 0)
            presenting = (
                rng.choice(["Neonatal seizures + HCM", "Birth seizures", "Congenital brain malformations", "Neonatal hypotonia + HCM"])
                if congenital else
                rng.choice(["Early infantile seizures", "Developmental delay + HCM", "Visual failure + seizures", "Hypotonia onset"])
            )
            outcome = rng.choice(["Death weeks-months", "Palliative neonatal care", "Rapidly fatal"]) if congenital else rng.choice(["Progressive decline", "Death age 2-8y", "Prolonged survival (partial CTSD)"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "protein": "CTSD/CathepsinD",
                "sex": sex, "phenotypic_class": pheno, "age_dx_y": age_dx,
                "presenting_feature": presenting, "em_finding": em,
                "visual_failure": visual_failure, "seizures": seizures,
                "hypertrophic_cardiomyopathy": hcm, "congenital_onset": congenital,
                "outcome": outcome,
            })
        else:
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene,
                "sex": sex, "phenotypic_class": pheno, "age_dx_y": age_dx,
                "em_finding": em, "visual_failure": visual_failure, "seizures": seizures,
            })
    return patients


# ── Populate cohorts at module load (constants — no side-effectful execution) ─────
for _g in NCL_GENES:
    _g["patients"] = _make_patients(_g)
    _g["n_patients"] = len(_g["patients"])

ALL_PATIENTS = [p for g in NCL_GENES for p in g["patients"]]


# ─── API: get_overview ────────────────────────────────────────────────────────────
def get_overview():
    """Return high-level NCL Atlas summary."""
    total = len(ALL_PATIENTS)

    gene_summary = []
    for g in NCL_GENES:
        pts = g["patients"]
        gene_summary.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "gene_class": g["gene_class"],
            "ncl_subgroup": g["ncl_subgroup"],
            "n_patients": g["n_patients"],
            "inheritance": g["inheritance"],
            "em_finding": g["em_finding"],
            "phenotype": g["phenotype"],
            "diet_treatment": g["diet_treatment"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
        })

    n_visual_failure   = sum(1 for p in ALL_PATIENTS if p.get("visual_failure", False))
    n_seizures         = sum(1 for p in ALL_PATIENTS if p.get("seizures", False))
    n_on_ert           = sum(1 for p in ALL_PATIENTS if p.get("on_cerliponase_alfa", False))
    n_hcm              = sum(1 for p in ALL_PATIENTS if p.get("hypertrophic_cardiomyopathy", False))
    n_vacuolated       = sum(1 for p in ALL_PATIENTS if p.get("vacuolated_lymphocytes_smear", False))
    n_deletion         = sum(1 for p in ALL_PATIENTS if p.get("has_1_02kb_deletion", False))

    return {
        "atlas": "NCL-Atlas — Complete 8-Gene Neuronal Ceroid Lipofuscinosis (Batten Disease) Atlas",
        "n_genes": len(NCL_GENES),
        "n_patients": total,
        "seeds": [g["seed"] for g in NCL_GENES],
        "genes_covered": [g["gene"] for g in NCL_GENES],
        "gene_subgroups": {
            "Infantile NCL (CLN1/PPT1)": ["CLN1"],
            "Late-Infantile NCL — ERT available (CLN2/TPP1)": ["CLN2"],
            "Juvenile NCL / JNCL — Most Common (CLN3)": ["CLN3"],
            "Adult NCL — Dominant (CLN4B/DNAJC5)": ["CLN4B"],
            "Late-Infantile Finnish Variant (CLN5)": ["CLN5"],
            "Late-Infantile/Adult NCL — ER-Resident (CLN6)": ["CLN6"],
            "Late-Infantile Turkish Variant (CLN7/MFSD8)": ["CLN7"],
            "Congenital NCL — HCM (CLN10/CTSD)": ["CLN10"],
        },
        "n_visual_failure": n_visual_failure,
        "n_seizures": n_seizures,
        "n_on_ert_cerliponase": n_on_ert,
        "n_hcm_cln10": n_hcm,
        "n_vacuolated_lymphocytes_cln3": n_vacuolated,
        "n_cln3_deletion": n_deletion,
        "em_key": {
            "GRODs": ["CLN1", "CLN10"],
            "Curvilinear bodies (CB)": ["CLN2"],
            "Fingerprint profiles (FP)": ["CLN3", "CLN4B"],
            "Mixed CB+FP": ["CLN5", "CLN7"],
            "Rectilinear profiles": ["CLN6"],
        },
        "critical_clinical_rules": [
            "CLN2/TPP1 CERLIPONASE ALFA (BRINEURA) — ONLY APPROVED NCL ERT: 300 mg ICV every 2 weeks via Ommaya reservoir; FDA and EMA approved 2017; slows motor decline in late-infantile NCL (CLN2/TPP1); start pre-symptomatically if sibling diagnosed; Ommaya port requires surgical placement; specialist centre delivery; TPP1 enzyme assay (DBS/leukocytes) is FIRST-LINE confirmation before WES",
            "CLN3 VACUOLATED LYMPHOCYTES ON BLOOD SMEAR — DIAGNOSTIC: order peripheral blood smear (Giemsa stain) in any child with visual failure + seizures; clear cytoplasmic vacuoles in lymphocytes are DIAGNOSTIC for CLN3; quick, cheap, widely available test; do this BEFORE EM or WES",
            "CLN3 1.02 kb DELETION — PCR FIRST, NOT WES: the 1.02 kb intragenic deletion (exons 7-8, chr16:28,532,191-28,533,199) accounts for 73-85% of all CLN3 alleles; this deletion is NOT detected by standard WES; targeted PCR for the deletion must be the FIRST molecular test when CLN3 is suspected",
            "CLN4B (DNAJC5) — ONLY AUTOSOMAL DOMINANT NCL, NO VISION LOSS: CLN4B is the sole AD NCL gene; adult onset 20-50y; all other NCL types (CLN1-3, CLN5-10) are AR; visual system is SPARED in CLN4B (no retinal dystrophy); adult NCL without visual failure in a patient with family history (dominant) = test DNAJC5 targeted p.Leu115Arg + p.Phe116Ile",
            "CLN10 (CTSD) — CONGENITAL NCL WITH HCM UNIQUE AMONG NCL: CLN10 is the only NCL gene causing congenital or neonatal-onset disease AND hypertrophic cardiomyopathy (HCM); HCM is absent in all other NCL genes; neonatal seizures + HCM + GRODs on EM = test CTSD; GRODs on EM in congenital setting = both PPT1 (CLN1) and CTSD (CLN10) enzyme assays",
            "EM ULTRASTRUCTURE IS GENE-SPECIFIC: GRODs = CLN1 + CLN10 | Curvilinear bodies (CB) = CLN2 (most characteristic) | Fingerprint profiles (FP) = CLN3 + CLN4B | Mixed CB+FP = CLN5 + CLN7 | Rectilinear profiles = CLN6; EM distinguishes NCL from other storage disorders and narrows the gene diagnosis; perform EM on skin biopsy, conjunctival biopsy, or buffy coat",
            "PPT1 AND TPP1 ENZYME ASSAYS ARE FIRST-LINE FOR CLN1/CLN2: PPT1 enzyme activity (DBS or leukocytes) is first-line for CLN1; TPP1 enzyme activity is first-line for CLN2; both are rapid, cheap, and available before WES; severely reduced activity confirms the diagnosis; normal activity in suspected NCL drives WES for CLN3-CLN10",
            "OPHTHALMOLOGY (ERG + FUNDUS) MANDATORY IN ALL NCL: progressive retinal dystrophy (rod-cone dystrophy) occurs in ALL NCL types EXCEPT CLN4B adult form; ERG (electroretinogram) is abnormal early; fundus shows macular degeneration and optic atrophy; ophthalmology referral at diagnosis is mandatory for all NCL; CLN4B exception: visual system spared",
            "VPA AVOIDANCE IN NCL: valproate (VPA) is generally avoided in NCL — NCL involves lysosomal dysfunction and VPA may worsen mitochondrial and lysosomal function; LEV (levetiracetam) is the preferred antiepileptic; clonazepam/clobazam for myoclonus; carbamazepine for focal seizures if needed; discuss with specialist",
            "SIBLING TESTING AND PRE-SYMPTOMATIC ERT (CLN2): if a proband is diagnosed with CLN2/TPP1, siblings should be tested immediately; if mutation confirmed in a sibling, start cerliponase alfa (Brineura) PRE-SYMPTOMATICALLY — pre-symptomatic ERT is the ideal scenario given the devastating natural history; confirm TPP1 deficiency by enzyme assay before starting ERT",
        ],
        "gene_summary": gene_summary,
        "mri_note": "Brain MRI in NCL: early may show cerebellar atrophy (CLN1 early), cortical atrophy (CLN3 late), or be near-normal; iron accumulation NOT a feature of NCL (unlike NBIA); white matter changes can occur; MRI less specific than EM + enzyme assay; CLN10 may show lissencephaly-like gyral simplification congenitally.",
    }


# ─── API: get_breakdown ───────────────────────────────────────────────────────────
def get_breakdown():
    """Return per-gene detailed breakdown for the gene table and clinical atlas tabs."""
    gene_rows = []
    for g in NCL_GENES:
        pts = g["patients"]
        n_visual = sum(1 for p in pts if p.get("visual_failure", False))
        n_seiz   = sum(1 for p in pts if p.get("seizures", False))
        gene_rows.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "gene_class": g["gene_class"],
            "ncl_subgroup": g["ncl_subgroup"],
            "inheritance": g["inheritance"],
            "em_finding": g["em_finding"],
            "onset_range_y": g["onset_range_y"],
            "n_patients": g["n_patients"],
            "seed": g["seed"],
            "phenotype": g["phenotype"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "diet_treatment": g["diet_treatment"],
            "gene_therapy_status": g["gene_therapy_status"],
            "critical_ci": g["critical_ci"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "key_variants": g["key_variants"],
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
            "n_visual_failure": n_visual,
            "n_seizures": n_seiz,
        })
    return {
        "genes": gene_rows,
        "total": len(NCL_GENES),
        "total_patients": len(ALL_PATIENTS),
    }


# ─── API: get_definitions ─────────────────────────────────────────────────────────
def get_definitions():
    """Return NCL clinical term definitions."""
    return {
        "atlas": "NCL-Atlas — Complete 8-Gene Neuronal Ceroid Lipofuscinosis (Batten Disease) Atlas",
        "ncl_overview": {
            "full_name": "Neuronal Ceroid Lipofuscinoses (NCL / Batten disease) — the most common group of inherited progressive neurodegenerative storage disorders of childhood; characterised by lysosomal accumulation of autofluorescent ceroid lipopigment in neurons and many other cell types; clinical triad: progressive cognitive and motor decline, epilepsy, and visual failure (except CLN4B)",
            "genes_in_atlas": 8,
            "only_approved_ert": "CLN2/TPP1: cerliponase alfa (Brineura) 300mg ICV q2 weeks — ONLY approved NCL ERT",
            "most_common_ncl": "CLN3 (Juvenile NCL / Batten disease) — most common worldwide",
            "em_key": "GRODs=CLN1/CLN10 | CB=CLN2 | FP=CLN3/CLN4B | CB+FP=CLN5/CLN7 | Rectilinear=CLN6",
        },
        "definitions": [
            {
                "term": "NCL — Neuronal Ceroid Lipofuscinoses: Definition, Classification, and Pathology",
                "definition": (
                    "Neuronal Ceroid Lipofuscinoses (NCL) are a group of >13 inherited lysosomal storage disorders "
                    "characterised by: (1) progressive neurodegeneration (cognitive decline, motor deterioration, "
                    "seizures), (2) visual failure (retinal dystrophy/optic atrophy) in most types, and (3) "
                    "lysosomal accumulation of autofluorescent ceroid lipopigment — a mixture of subunit c of "
                    "mitochondrial ATP synthase, sphingolipid activator proteins (SAPs A and D), and oxidised "
                    "phospholipids.\n\n"
                    "Pathological hallmark: autofluorescent storage material visible under UV/fluorescence microscopy "
                    "in neurons, skin, conjunctiva, and blood cells. EM ultrastructure is gene-specific (GRODs, CB, FP, "
                    "rectilinear profiles) and guides molecular diagnosis.\n\n"
                    "Inheritance: all AR except CLN4B (DNAJC5), which is AD.\n\n"
                    "Treatment: only CLN2/TPP1 has approved ERT (cerliponase alfa/Brineura ICV). All others: supportive.\n\n"
                    "Incidence: collectively ~1/100,000 live births; CLN3 (Juvenile) is most common."
                ),
            },
            {
                "term": "Cerliponase Alfa (Brineura) — ONLY Approved NCL ERT (CLN2/TPP1)",
                "definition": (
                    "Cerliponase alfa (Brineura, BioMarin) is a recombinant human tripeptidyl peptidase 1 (rhTPP1) "
                    "enzyme replacement therapy — the FIRST and ONLY approved disease-modifying therapy for any NCL.\n\n"
                    "Approval: FDA 2017 (accelerated approval); EMA 2017. Indication: CLN2 disease (late-infantile NCL).\n\n"
                    "Dose: 300 mg ICV (intracerebroventricular) every 2 weeks.\n\n"
                    "Route: Ommaya reservoir (surgically implanted subcutaneous reservoir connected to a ventricular "
                    "catheter); cannot be given intravenously (does not cross BBB in therapeutic amounts).\n\n"
                    "Mechanism: replaces deficient TPP1 enzyme in the CNS lysosomes; ICV delivery bypasses BBB; "
                    "enzyme taken up by neurons via mannose-6-phosphate (M6P) receptor-mediated endocytosis.\n\n"
                    "Efficacy: significantly slows motor decline (assessed by CLN2 Clinical Rating Scale motor domain); "
                    "does not fully stop progression but dramatically extends ambulatory period vs natural history.\n\n"
                    "Key clinical pearl: START BEFORE SYMPTOMS in at-risk siblings. If a sibling is diagnosed with "
                    "CLN2, test other children immediately; if TPP1 deficiency confirmed, start cerliponase alfa "
                    "PRE-SYMPTOMATICALLY — pre-symptomatic treatment is the best outcome scenario.\n\n"
                    "Pre-infusion: antihistamine + corticosteroid to prevent infusion reactions (CSF inflammatory response).\n\n"
                    "Monitoring: CLN2 Clinical Rating Scale (language + motor domains) quarterly; MRI brain annually."
                ),
            },
            {
                "term": "CLN3 1.02 kb Deletion — Most Common NCL Mutation, Missed by WES",
                "definition": (
                    "The 1.02 kb genomic deletion in CLN3 (chr16:28,532,191-28,533,199, removing exons 7 and 8) is the "
                    "single most common NCL mutation globally, found in 73-85% of all CLN3 disease alleles.\n\n"
                    "CRITICAL: This deletion is NOT detected by standard exome sequencing (WES) or gene panel "
                    "sequencing. It requires targeted PCR or chromosomal microarray/CNV analysis.\n\n"
                    "Clinical testing algorithm for suspected CLN3:\n"
                    "  Step 1: Targeted PCR for 1.02kb deletion (both alleles)\n"
                    "  Step 2: If one allele has deletion, sequence CLN3 coding region for second allele\n"
                    "  Step 3: If deletion negative on both alleles, full CLN3 sequencing\n\n"
                    "Genotyping results:\n"
                    "  ~40% of JNCL patients: homozygous 1.02kb deletion\n"
                    "  ~45% of JNCL patients: compound heterozygous (deletion + point mutation)\n"
                    "  ~15% of JNCL patients: two point mutations (no deletion)\n\n"
                    "Note: vacuolated lymphocytes on blood smear should be checked FIRST — it is faster, "
                    "cheaper, and immediately available while PCR is arranged."
                ),
            },
            {
                "term": "Vacuolated Lymphocytes — Diagnostic Test for CLN3 (Blood Smear)",
                "definition": (
                    "Vacuolated lymphocytes are lymphocytes with multiple clear cytoplasmic vacuoles visible on a "
                    "peripheral blood smear (Wright-Giemsa stain). In the context of NCL, they are DIAGNOSTIC for CLN3.\n\n"
                    "Test: order a routine peripheral blood smear and request specific examination for vacuolated "
                    "lymphocytes. Standard blood film preparation. No special handling required.\n\n"
                    "Interpretation: presence of vacuolated lymphocytes in a child with progressive visual loss + "
                    "seizures is DIAGNOSTIC for CLN3 Batten disease — no other commonly encountered neurological "
                    "condition produces this combination.\n\n"
                    "Why vacuoles occur: CLN3 protein is absent from lysosomal membranes in lymphocytes → ceroid "
                    "storage material accumulates → vacuole-like appearance in lymphocyte cytoplasm.\n\n"
                    "Other storage disorders with vacuolated lymphocytes: some other LSDs (Niemann-Pick, GM1 "
                    "gangliosidosis) — context (age, clinical features) distinguishes.\n\n"
                    "Key point: This is a FREE test available in any hospital laboratory. Order it FIRST when "
                    "CLN3 is suspected, before EM or molecular testing."
                ),
            },
            {
                "term": "NCL EM Ultrastructure — Gene-Specific Diagnostic Patterns",
                "definition": (
                    "Electron microscopy (EM) of storage material in NCL shows gene-specific ultrastructural patterns "
                    "that narrow the molecular diagnosis before sequencing:\n\n"
                    "GRODs (Granular Osmiophilic Deposits): electron-dense granular material filling lysosomes; "
                    "= CLN1 (PPT1) and CLN10 (CTSD); fingerprint-like at high magnification.\n\n"
                    "Curvilinear Bodies (CB): curved parallel stacks of membrane; most characteristic of CLN2 (TPP1).\n\n"
                    "Fingerprint Profiles (FP): concentric whorled lamellar membranes (resembles fingerprint); "
                    "= CLN3 and CLN4B; also mixed with CB in CLN2, CLN5, CLN7.\n\n"
                    "Rectilinear Profiles (RL): straight parallel membrane stacks; = CLN6 (most specific).\n\n"
                    "Mixed CB + FP: CLN5 (also rectilinear component) and CLN7.\n\n"
                    "EM biopsy sources: skin biopsy (eccrine sweat glands most informative), conjunctival biopsy, "
                    "muscle biopsy, or buffy coat from blood. Skin biopsy is preferred (least invasive, "
                    "reliable eccrine gland harvest).\n\n"
                    "GRODs in a NEONATE or INFANT = test BOTH PPT1 (CLN1) and CTSD (CLN10)."
                ),
            },
            {
                "term": "CLN4B/DNAJC5 — Only Dominant NCL; CSPα Chaperone; No Vision Loss",
                "definition": (
                    "CLN4B (DNAJC5) is unique among NCL genes in two critical ways:\n"
                    "1. ONLY AUTOSOMAL DOMINANT NCL gene — all others are AR.\n"
                    "2. VISUAL SYSTEM IS SPARED — no retinal dystrophy, no ERG abnormality.\n\n"
                    "Protein: CSPα (cysteine-string protein alpha, 198aa, 22kDa) — a synaptic vesicle-associated "
                    "HSP40 co-chaperone. CSPα palmitoylation targets it to synaptic vesicle membranes. It recruits "
                    "HSC70 + SGT to form a trimeric chaperone preventing misfolding of synaptic proteins.\n\n"
                    "Only two known pathogenic variants:\n"
                    "  p.Leu115Arg (more common) — Parry disease (original description)\n"
                    "  p.Phe116Ile (fewer reported families)\n"
                    "Both in the cysteine-string region; gain of toxic function (dominant negative).\n\n"
                    "Clinical: adult onset 20-50y; progressive myoclonic epilepsy + cerebellar ataxia + dementia; "
                    "NO VISION LOSS; family history spanning generations (dominant); fingerprint profiles on EM; "
                    "no vacuolated lymphocytes (unlike CLN3).\n\n"
                    "Kufs disease type A naming: CLN4B (dominant) and CLN6 (recessive) both historically called "
                    "'Kufs type A' — distinct genes but same phenotypic label; differentiate by inheritance and EM."
                ),
            },
            {
                "term": "CLN10/CTSD — Congenital NCL and Hypertrophic Cardiomyopathy",
                "definition": (
                    "CLN10 (cathepsin D, CTSD) is the CONGENITAL form of NCL — the earliest and most severe.\n\n"
                    "CTSD (412aa): major lysosomal aspartyl protease; ubiquitous expression; processes lysosomal "
                    "proteins; activates other lysosomal hydrolases; secreted form (takes up by M6P receptors).\n\n"
                    "Congenital NCL hallmarks:\n"
                    "  - Onset: in utero (fetal brain malformations on ultrasound), at birth, or first weeks of life\n"
                    "  - HYPERTROPHIC CARDIOMYOPATHY (HCM): unique to CLN10 among all NCL genes; present in majority\n"
                    "  - GRODs on EM: same as CLN1 — distinguish by onset age and presence of HCM\n"
                    "  - Brain malformations: lissencephaly-like, simplified gyri\n"
                    "  - Prognosis: extremely poor; typically fatal within weeks to months\n\n"
                    "Genotype: frameshift/nonsense nulls = most severe congenital phenotype; p.Trp383Cys (partial "
                    "activity) may give later, less severe onset.\n\n"
                    "CTSD enzyme assay: severely reduced activity in DBS, fibroblasts, leukocytes — confirms diagnosis.\n\n"
                    "HCM + neonatal seizures + GRODs = CTSD + CLN10 diagnosis; "
                    "echocardiogram is mandatory in any suspected NCL with congenital onset."
                ),
            },
            {
                "term": "CLN6 — ER-Resident NCL Protein and Kufs Type A Adult Disease",
                "definition": (
                    "CLN6 is unique among NCL genes because its protein localises to the ENDOPLASMIC RETICULUM (ER), "
                    "not to lysosomes. All other NCL proteins (CLN1-5, CLN7-10) are lysosomal.\n\n"
                    "CLN6 protein (311aa, 36kDa): ER-resident type I transmembrane protein; function uncertain; "
                    "proposed role in ER quality control of lysosomal hydrolases before Golgi M6P tagging and "
                    "lysosomal delivery. CLN6 loss → indirect lysosomal dysfunction.\n\n"
                    "EM: rectilinear profiles (straight parallel membrane stacks) — most characteristic for CLN6.\n\n"
                    "Two phenotypes:\n"
                    "  1. Late-infantile CLN6: onset 18m-8y; visual failure + seizures + ataxia; death 5-12y after onset.\n"
                    "  2. Adult Kufs disease type A (CLN6 form): onset 20-50y; progressive myoclonic epilepsy + dementia; "
                    "     AR inheritance (unlike CLN4B Kufs which is AD).\n\n"
                    "Founder: p.Trp263Cys enriched in Newfoundland (Canada) and Portuguese populations.\n\n"
                    "CLN6 gene therapy (AAV9-CLN6 intrathecal): Phase I/II trial at Nationwide Children's Hospital "
                    "(NCT02725580) — first intrathecal AAV9 trial for NCL; safety positive.\n\n"
                    "No enzyme assay for CLN6 (function unknown); molecular diagnosis only."
                ),
            },
            {
                "term": "NCL Gene Therapy — Status Across All 8 Genes",
                "definition": (
                    "Gene therapy is the primary therapeutic research strategy for NCL (except CLN2 which already "
                    "has approved ERT). All NCL genes are AAV-compatible in size (<2.5 kb cDNA):\n\n"
                    "CLN1 (PPT1): AAV-PPT1 ICV/intrathecal — Phase I/II trials ongoing; cysteamine+NAC adjunct.\n"
                    "CLN2 (TPP1): AAV-TPP1 — preclinical positive (Dachshund natural model); combined ERT+GT studied.\n"
                    "CLN3: AAV9-CLN3 intrathecal — Phase I/II NCT03770572; ASO approach also studied.\n"
                    "CLN4B (DNAJC5): allele-specific silencing required (dominant toxic gain); no trials.\n"
                    "CLN5: AAV9-CLN5 ICV — Phase I/II; sheep natural model; soluble form cross-correction possible.\n"
                    "CLN6: AAV9-CLN6 intrathecal — Phase I/II NCT02725580 completed; data published.\n"
                    "CLN7 (MFSD8): AAV9-MFSD8 — preclinical; unknown substrate complicates alternatives.\n"
                    "CLN10 (CTSD): neonatal AAV; CTSD ERT (secreted enzyme) also feasible; no trials.\n\n"
                    "Key challenge: timing. Gene therapy must be given early — before major neuronal loss. "
                    "Pre-symptomatic delivery (identified via NBS or sibling diagnosis) is the ideal scenario. "
                    "Natural disease models: CLN2 Dachshunds, CLN5 sheep, CLN6 sheep — invaluable for clinical translation."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== NCL Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    print(f"EM key: {ov['em_key']}")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
    print("OK")
