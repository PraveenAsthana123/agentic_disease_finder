#!/usr/bin/env python3
"""Hereditary-CGD-Neutrophil-Oxidative-Burst-Defect-Atlas -- Complete 8-Gene Reference
Chronic Granulomatous Disease (CGD) + Neutrophil NADPH Oxidase Subunit Defects + MPO Deficiency + RAC2 NIS
(CYBB/gp91phox-X-CGD-Most-Common-65pct-XLR-NBT-DHR-Absent-PATHOGNOMONIC-IFN-gamma-Prophylaxis-HSCT-Curative,
 NCF1/p47phox-AR-CGD-25pct-GTdeletion-Most-Common-AR-Mutation-Autosomal-Equal-Sex,
 CYBA/p22phox-AR-CGD-5pct-Stabilizes-gp91phox-CD18-Normal-Unlike-LAD,
 NCF2/p67phox-AR-CGD-5pct-Activation-Domain-FADH2-Transfer,
 NCF4/p40phox-AR-CGD-1-2pct-PX-Domain-PI3P-Binding-Phagosomal-Targeting,
 CYBC1/EROS-AR-CGD-Ultra-Rare-Essential-for-Reactive-Oxygen-Species-gp91phox-p22phox-Maturation,
 RAC2-AD-GOF-AR-LOF-Neutrophil-Immunodeficiency-Syndrome-NIS-GTPase-NADPH-Oxidase-Assembly,
 MPO/Myeloperoxidase-Deficiency-Most-Common-Phagocyte-Enzyme-Defect-1in4000-Mild-Candida-Risk):

CYBB  (Cytochrome b-558 heavy chain / gp91phox; 570aa; Xq21.1; XLR;
         CYBB encodes the 91 kDa glycoprotein heavy chain (gp91phox) of cytochrome b-558, the
           catalytic core of the NADPH oxidase (NOX2) complex in phagocytes;
         gp91phox is a transmembrane electron transfer protein: it accepts electrons from NADPH
           via FAD (FADH2 intermediate) and transfers them to molecular oxygen (O2) across the
           phagosomal membrane → superoxide anion (O2•−) → dismutated to H2O2 → converted to
           HOCl by myeloperoxidase (MPO) → oxidative burst lethal to bacteria/fungi;
         gp91phox forms a heterodimer with p22phox (CYBA) = cytochrome b-558 — the stable membrane
           component of resting NOX2; without gp91phox, p22phox is also destabilized;
         X-LINKED CHRONIC GRANULOMATOUS DISEASE (X-CGD, CGD1): ~65% of all CGD cases globally;
           PATHOGNOMONIC: ABSENT OXIDATIVE BURST on DHR-123 (dihydrorhodamine) flow cytometry
           and NBT (nitroblue tetrazolium) slide test — DHR = gold standard (most sensitive/specific);
           DHR: normal = right-shifted histogram (rhodamine-123 fluorescence increases after PMA
           stimulation); X-CGD = flat histogram (NO fluorescence shift) because no H2O2 produced;
         CARRIER FEMALE DIAGNOSTIC: X-linked → female carriers = 50% normal + 50% CGD neutrophils
           (bimodal DHR histogram = PATHOGNOMONIC for female carrier; can be used for genetic counseling);
         INFECTIONS: Staphylococcus aureus (liver abscesses — LIVER ABSCESS IN CHILD = CGD UNTIL PROVEN
           OTHERWISE), Aspergillus spp. pneumonia (most lethal complication — pulmonary Aspergillus),
           Burkholderia cepacia (septicemia), Nocardia, Serratia marcescens, Chromobacterium violaceum
           (tropical exposure), catalase-positive organisms ONLY (organisms that destroy their own H2O2
           — catalase-negative organisms like Streptococcus killed by own H2O2 even in CGD);
         INFLAMMATORY DISEASE: granulomata (gut, bladder, esophagus) — inflammatory bowel disease-like
           colitis in ~40% of CGD patients (NOT infection-mediated but dysregulated inflammation);
           Crohn's-like disease requiring immunosuppression (steroids, infliximab);
         TREATMENT: Prophylactic TMP-SMX (bacterial) + itraconazole/voriconazole (Aspergillus) LIFELONG;
           IFN-γ (interferon-gamma 50 mcg/m² SC 3×/week) — reduces infection rate 70% in X-CGD
           (enhances residual oxidase activity in partial X-CGD and activates alternative microbicidal
           pathways); HSCT is CURATIVE (myeloablative conditioning, MSD > MUD); gene therapy (lentiviral
           CYBB correction) — clinical trials; Voriconazole for acute Aspergillus; seed SEED_BASE+0).
NCF1   (Neutrophil cytosol factor 1 / p47phox; 390aa; 7q11.23; AR;
         NCF1 encodes p47phox, a cytosolic regulatory subunit of NADPH oxidase;
         In resting state: p47phox is cytosolic, complexed with p67phox (NCF2) and p40phox (NCF4);
         After activation: p47phox is phosphorylated on multiple serine residues → conformational change
           → SH3 domains engage PRR domain of p22phox on membrane → whole cytosolic complex
           (p47phox-p67phox-p40phox) translocates to phagosomal membrane → activates NOX2;
         p47phox = the "organizer subunit" — its phosphorylation is the rate-limiting activation step;
         NCF1 GENE STRUCTURE: NCF1 at 7q11.23 is flanked by two highly homologous pseudogenes (NCF1B and
           NCF1C) → unequal crossover during meiosis generates the MOST COMMON AR-CGD MUTATION: GT→AT
           (ΔGT) dinucleotide deletion at position 75_76 in exon 2 → frameshift → truncated p47phox;
         ΔGT deletion: ~70-80% of all NCF1/p47-CGD alleles carry this founder mutation;
           AUTOSOMAL RECESSIVE p47-CGD: ~25% of all CGD cases; equal sex distribution (unlike X-CGD);
         PHENOTYPE: typically MILDER THAN X-CGD (some residual oxidase activity from hypomorphic alleles);
           later onset, fewer hospitalizations, longer survival; Aspergillus less common than X-CGD;
           IBD-like colitis still occurs (30-35%); infections: same catalase-positive spectrum;
         DIAGNOSIS: DHR test ABSENT (same as X-CGD — cannot distinguish X-CGD from AR-CGD by DHR alone);
           NBT ABSENT; distinguish by inheritance pattern (male X-CGD vs equal-sex AR-CGD) + genetic panel
           (CYBB, NCF1, CYBA, NCF2, NCF4 sequencing); NCF1 ΔGT deletion specific PCR or targeted NGS;
         TREATMENT: TMP-SMX + itraconazole prophylaxis; IFN-γ — less evidence than in X-CGD but used;
           HSCT curative but threshold higher than X-CGD (milder phenotype); seed SEED_BASE+1).
CYBA   (Cytochrome b-558 light chain / p22phox; 195aa; 16q24.2; AR;
         CYBA encodes p22phox, the 22 kDa light chain of cytochrome b-558;
         p22phox is the small transmembrane subunit that forms a 1:1 heterodimer with gp91phox (CYBB);
         The cytochrome b-558 heterodimer (gp91phox + p22phox) is the stable membrane unit of NOX2;
           without p22phox: gp91phox is destabilized and degraded → functional loss identical to CYBB LOF;
           conversely, without gp91phox: p22phox is also degraded → CYBB LOF eliminates both subunits;
         p22phox C-terminal proline-rich region (PRR) is the docking site for the SH3 domains of p47phox
           upon phagocyte activation — this p22phox-p47phox interaction is mandatory for NOX2 assembly;
         p22phox is also required for NOX1, NOX3, NOX4, DUOX1/2 activity (shared subunit across
           NADPH oxidase family members in non-phagocyte tissues);
         AR-CYBA-CGD: ~5% of all CGD cases; equal sex distribution; autosomal recessive;
         PHENOTYPE: clinically indistinguishable from X-CGD or p47-CGD — same pathogen spectrum
           (catalase-positive organisms), granulomata, IBD-like colitis; DHR/NBT absent;
         KEY DISTINCTION: cytochrome b-558 protein assay (spectrophotometric midpoint potential assay
           or cytochrome b558 Western blot) — in CYBA/p22-CGD: gp91phox ALSO ABSENT (destabilized);
           in NCF1/p47-CGD: gp91phox PRESENT (membrane components intact); this distinguishes
           cytochrome b-558-negative CGD (CYBB or CYBA) from cytochrome b-558-positive CGD (NCF1-4);
         DIAGNOSIS: DHR absent; cytochrome b558 spectroscopy ABSENT; CYBA sequencing; protein assay;
         TREATMENT: TMP-SMX + itraconazole prophylaxis; IFN-γ; HSCT for severe disease; seed SEED_BASE+2).
NCF2   (Neutrophil cytosol factor 2 / p67phox; 526aa; 1q25.3; AR;
         NCF2 encodes p67phox, a cytosolic activating subunit of NADPH oxidase;
         p67phox contains four N-terminal TPR (tetratricopeptide repeat) motifs that bind RAC2-GTP,
           an ACTIVATION DOMAIN (AD) critical for electron transfer from NADPH via FAD to gp91phox,
           and two SH3 domains that participate in the cytosolic complex (p47phox-p67phox-p40phox);
         p67phox is the ACTIVATING SUBUNIT: it directly interacts with FAD-binding domain of gp91phox
           → enables FADH2 → FAD electron transfer → superoxide generation;
         NCF2/p67-CGD: ~5% of all CGD cases; equal sex distribution; AR;
         PHENOTYPE: severe disease (similar to X-CGD); cytochrome b-558 PRESENT (membrane complex intact);
           DHR absent; similar infection spectrum (Staphylococcus, Aspergillus, Burkholderia, Nocardia);
           IBD-like disease occurs; granulomata form; early onset like X-CGD;
         MOLECULAR: p67phox in resting cells is tightly bound to p47phox (1:1 complex) and also
           to p40phox (forming ternary p47-p67-p40 cytosolic complex); on activation, whole complex
           translocates — p67phox is required for enzymatic activity (NOT just membrane docking);
         DIAGNOSIS: DHR absent; cytochrome b558 spectroscopy PRESENT (gp91phox+p22phox intact);
           p67phox protein by Western blot ABSENT; NCF2 sequencing confirms; NADPH oxidase activity assay;
         TREATMENT: TMP-SMX + itraconazole; IFN-γ; HSCT for severe; gene therapy research; seed SEED_BASE+3).
NCF4   (Neutrophil cytosol factor 4 / p40phox; 339aa; 22q13.1; AR;
         NCF4 encodes p40phox, a cytosolic regulatory subunit that fine-tunes NADPH oxidase activity;
         p40phox contains: (1) PX (phox homology) domain — binds PI3P (phosphatidylinositol 3-phosphate)
           on phagosomal membrane → targets p40phox (and the p47-p67-p40 complex) to maturing phagosomes;
           (2) PB1 domain — mediates interaction with p67phox in the cytosolic complex;
         p40phox is specifically important for phagosomal (intracellular) but not plasma membrane
           NADPH oxidase activation — p40phox targets the complex to phagosomes containing ingested bacteria;
         NCF4/p40-CGD: ~1-2% of all CGD cases — RAREST CANONICAL CGD GENE; AR; equal sex distribution;
         PHENOTYPE: UNIQUE MILD PHENOTYPE — predominantly IBD-like colitis (Crohn's-like gastrointestinal
           disease), recurrent lymphadenitis, milder infections compared to other CGD types;
           LIVER ABSCESSES (same Staphylococcal pathogen) but LESS ASPERGILLUS than other CGD;
           IBDX mutation p40phox R105Q specifically impairs PI3P binding → selective phagosomal
           targeting failure without affecting plasma membrane NOX2 burst;
         DIAGNOSIS: DHR TEST MAY BE NORMAL OR REDUCED (not absent — plasma membrane NOX2 intact);
           specific phagosomal NOX2 assay (latex bead phagosome oxidase test) ABSENT;
           this makes NCF4/p40-CGD potentially missed on standard DHR screening;
           cytochrome b558 PRESENT; p40phox protein absent or reduced by Western blot;
         TREATMENT: prophylactic TMP-SMX; itraconazole; colitis managed with steroids/infliximab;
           IFN-γ; HSCT generally not indicated (mild phenotype); gene therapy research; seed SEED_BASE+4).
CYBC1  (Cytochrome b-245 chaperone 1 / EROS; 291aa; 1p22.1; AR;
         CYBC1 (aka EROS — Essential for Reactive Oxygen Species) encodes a previously
           uncharacterized endoplasmic reticulum chaperone discovered in 2019;
         CYBC1/EROS is essential for the FOLDING AND STABILITY of both gp91phox (CYBB) and
           p22phox (CYBA) during biosynthesis in the ER → without EROS: gp91phox and p22phox
           fail to fold correctly, are retained in ER, and degraded by ERAD (ER-associated degradation);
         CYBC1/EROS-CGD: ULTRA-RARE, described in fewer than 10 families globally (2019 discovery);
           AR; equal sex distribution; functional consequence identical to CYBB (X-CGD) or CYBA (AR-CGD)
           phenotype — cytochrome b558 absent because neither gp91phox nor p22phox reaches membrane;
         PATHOMECHANISM UNIQUE: NOT a structural subunit of NOX2 (EROS is not part of the assembled
           oxidase complex) but a CHAPERONE required for subunit maturation;
           distinction: CYBC1 encodes a chaperone → EROS absent → gp91phox + p22phox never
           reach plasma/phagosomal membrane → NOX2 = non-functional despite intact CYBB and CYBA genes;
         PHENOTYPE: identical to X-CGD clinically (profound CGD: liver abscesses, Aspergillus,
           Burkholderia, Serratia, Nocardia; IBD colitis; granulomata); DHR ABSENT;
         DIAGNOSIS: DHR absent; cytochrome b558 absent (gp91phox + p22phox absent from membrane);
           CYBB, NCF1, CYBA, NCF2, NCF4 SEQUENCING ALL NORMAL → triggers whole-exome sequencing →
           CYBC1 variants identified; EROS Western blot (if available at research labs) — absent;
         TREATMENT: same as X-CGD (TMP-SMX + itraconazole + IFN-γ); HSCT curative;
           gene therapy for CYBC1 — not yet in clinical trials; seed SEED_BASE+5).
RAC2   (Ras-related C3 botulinum toxin substrate 2 / RAS-related protein Rac-2; 192aa; 22q13.1;
         AD-GOF / AR-LOF;
         RAC2 encodes Rac2, a small GTPase (Rho-family) expressed predominantly in hematopoietic cells;
         RAC2 functions: (1) activates NADPH oxidase — Rac2-GTP directly binds the TPR motif of p67phox
           → triggers full NOX2 enzymatic activation alongside p47phox docking; WITHOUT Rac2-GTP,
           NOX2 assembly is incomplete → minimal oxidative burst even with intact NOX2 subunits;
           (2) regulates actin cytoskeleton in neutrophils → required for chemotaxis, phagocytosis,
           degranulation, NET formation;
         TWO DISTINCT DISEASE MECHANISMS:
           AD-GOF (dominant negative): p.D57N mutation → Rac2-D57N = dominant negative form that
             displaces normal Rac2 → NEUTROPHIL IMMUNODEFICIENCY SYNDROME (NIS): neonatal inflammatory
             disease, recurrent bacterial/fungal infections, defective oxidative burst, chemotaxis failure;
           AR-LOF (biallelic): severe NIS with combined neutrophil dysfunction + lymphocyte defects;
         PHENOTYPE: recurrent bacterial infections (Staphylococcus, gram-negatives), fungal infections
           (Aspergillus), neutropenia (in some), poor wound healing, delayed cord separation (in severe
           cases), IBD-like colitis; NIS = distinct from CGD but overlapping phenotype;
         DIAGNOSIS: DHR REDUCED BUT NOT ABSENT (partial oxidative burst — unlike complete CGD);
           RAC2 protein by flow cytometry (reduced in LOF); rac2-GTP loading assay (research labs);
           RAC2 sequencing (targeted mutation D57N for AD-GOF; full exon sequencing for AR-LOF);
         TREATMENT: prophylactic antibiotics + antifungal; IFN-γ; HSCT for severe cases;
           G-CSF variable benefit; seed SEED_BASE+6).
MPO    (Myeloperoxidase; 745aa; 17q22; AR;
         MPO encodes myeloperoxidase, the enzyme in primary (azurophil) neutrophil granules that
           converts H2O2 (produced by NADPH oxidase via superoxide dismutation) into HOCl (hypochlorous
           acid) — the most potent microbicidal oxidant produced by neutrophils;
         NADPH OXIDASE → SUPEROXIDE (O2•−) → SOD → H2O2 + MPO → HOCl (bleach) → maximum oxidant;
           Without MPO: O2•− and H2O2 still produced by NADPH oxidase (NOX2 intact) but HOCl = ZERO;
           residual killing via H2O2 alone is sufficient for most bacterial killing — MPO deficiency
           is therefore much MILDER than CGD (complete NOX2 absence);
         MPO DEFICIENCY PREVALENCE: most common phagocyte enzyme defect — ~1 in 4,000 individuals;
           complete deficiency ~1/4000; partial deficiency much more common;
         PHENOTYPE: USUALLY ASYMPTOMATIC in healthy individuals; clinically significant primarily in
           concurrent DIABETES MELLITUS — diabetics with MPO deficiency have markedly increased
           CANDIDA infection risk (invasive candidiasis, disseminated candida); also rarely: Aspergillus
           in severely immunocompromised; recurrent Candida vaginitis; Candida nail infections;
         NBT/DHR TEST: NORMAL (NADPH oxidase intact — H2O2 produced normally → DHR shifts normally);
           this is the key distinction from CGD where DHR ABSENT;
         SPECIFIC MPO ASSAY: MPO stain (cytochemistry) or MPO ELISA on PMN lysate — MPO absent;
           automatic hematology analyzers (e.g., Sysmex XN series) have dedicated MPO channel that
           detects MPO deficiency as a flagged population — many diagnoses now made incidentally on CBC;
         TREATMENT: usually NO TREATMENT REQUIRED in immunocompetent patients; aggressive antifungal
           prophylaxis and monitoring in diabetics with MPO deficiency (fluconazole prophylaxis);
           no HSCT, no IFN-γ; optimize glycemic control in diabetics; seed SEED_BASE+7).

ATLAS_NAME   = "Hereditary-CGD-Neutrophil-Oxidative-Burst-Defect-Atlas"
SEED_BASE    = 4014   # seeds 4014-4021 (8 genes × 40 patients each)
N_PER_GENE   = 40
TOTAL_N      = N_PER_GENE * 8
"""

import random
import math

ATLAS_NAME   = "Hereditary-CGD-Neutrophil-Oxidative-Burst-Defect-Atlas"
SEED_BASE    = 4014   # seeds 4014-4021 (8 genes × 40 patients each)
N_PER_GENE   = 40
TOTAL_N      = N_PER_GENE * 8

GENES = ["CYBB", "NCF1", "CYBA", "NCF2", "NCF4", "CYBC1", "RAC2", "MPO"]

GENE_DATA = {
    "CYBB": {
        "full_name":               "Cytochrome b-558 heavy chain / gp91phox (CYBB; X-linked Chronic Granulomatous Disease / X-CGD)",
        "aa":                      570,
        "locus":                   "Xq21.1",
        "kda":                     "91",
        "inheritance":             "XLR (X-linked recessive; hemizygous males affected; female carriers mosaic bimodal DHR; CYBB mutations ~65% of all CGD)",
        "syndrome":                "X-CGD — X-linked Chronic Granulomatous Disease; gp91phox absent; NADPH oxidase non-functional; most common CGD type (~65%)",
        "pathway":                 "CYBB/gp91phox absent → cytochrome b-558 heterodimer fails assembly → NOX2 non-functional → no superoxide → no H2O2 → no HOCl → complete oxidative burst failure; p22phox (CYBA) also destabilized without gp91phox",
        "onset_range":             (0, 12),
        "dhr_absent_pct":          0.99,        # DHR ABSENT = PATHOGNOMONIC CGD
        "liver_abscess_pct":       0.75,        # Staphylococcal liver abscess — CGD HALLMARK
        "aspergillus_pct":         0.65,        # Aspergillus pneumonia — most lethal complication
        "burkholderia_pct":        0.45,        # Burkholderia cepacia septicemia
        "ibd_colitis_pct":         0.40,        # Crohn's-like IBD colitis
        "granulomata_pct":         0.70,        # Obstructive granulomata (GI, urinary)
        "ifn_gamma_pct":           0.88,        # IFN-γ prophylaxis standard of care
        "hsct_pct":                0.55,        # HSCT curative; increasingly used early
        "recurrent_infect_pct":    0.98,
        "pathognomonic":           "DHR-123 FLOW CYTOMETRY ABSENT SHIFT = PATHOGNOMONIC CGD (gold standard test: PMA-stimulated neutrophils show NO rhodamine-123 fluorescence increase in CGD vs. normal right-shift); NBT SLIDE TEST ABSENT (classical test: neutrophils fail to reduce yellow NBT to blue formazan); LIVER ABSCESS IN A CHILD = CGD UNTIL PROVEN OTHERWISE (Staphylococcus aureus liver abscesses are hallmark); ASPERGILLUS PULMONARY INFECTION = most lethal complication (Aspergillus spp. killed by HOCl which is absent in CGD); CARRIER FEMALE DHR = BIMODAL HISTOGRAM (two populations: normal and CGD PMNs) = X-linked confirmed; catalase-positive organisms only (organisms using their own catalase to destroy H2O2 — evade residual H2O2 killing in CGD)",
        "key_drug":                "PROPHYLACTIC TMP-SMX (trimethoprim-sulfamethoxazole; bacterial prophylaxis) LIFELONG + ITRACONAZOLE/VORICONAZOLE (antifungal prophylaxis; Aspergillus prevention) LIFELONG + IFN-GAMMA (50 mcg/m² SC 3×/week; reduces infection rate ~70%; enhances residual oxidase activity); HSCT = CURATIVE (myeloablative conditioning; MSD > MUD; reduces mortality to <5% in experienced centers); VORICONAZOLE for acute Aspergillus; INFLIXIMAB/STEROIDS for CGD colitis; gene therapy trials (lentiviral CYBB): GT4-CGD, MYELOID-PILOT trials",
    },
    "NCF1": {
        "full_name":               "Neutrophil cytosol factor 1 / p47phox (NCF1; Autosomal Recessive CGD type II / p47-CGD)",
        "aa":                      390,
        "locus":                   "7q11.23",
        "kda":                     "47",
        "inheritance":             "AR (autosomal recessive; both sexes equally affected; ΔGT deletion at NCF1 exon 2 codon 75_76 is most common AR-CGD mutation; ~25% of all CGD)",
        "syndrome":                "p47-CGD — Autosomal Recessive CGD type II; most common AR-CGD (~25% of all CGD); NCF1 flanked by pseudogenes NCF1B/NCF1C → ΔGT frameshift founder mutation ~70-80% of p47 alleles",
        "pathway":                 "NCF1/p47phox absent → cytosolic complex (p47-p67-p40) cannot translocate to phagosomal membrane → NOX2 activation fails; cytochrome b-558 (gp91phox+p22phox) INTACT on membrane but no activating signal delivered",
        "onset_range":             (0, 24),
        "dhr_absent_pct":          0.97,
        "liver_abscess_pct":       0.60,
        "aspergillus_pct":         0.45,        # less Aspergillus than X-CGD
        "burkholderia_pct":        0.35,
        "ibd_colitis_pct":         0.33,
        "granulomata_pct":         0.60,
        "ifn_gamma_pct":           0.70,        # less evidence for IFN-γ in AR-CGD
        "hsct_pct":                0.35,        # lower threshold for HSCT in X-CGD
        "recurrent_infect_pct":    0.92,
        "pathognomonic":           "DHR ABSENT (same as X-CGD — DHR test alone CANNOT distinguish CYBB from NCF1 CGD); CYTOCHROME b-558 SPECTROSCOPY PRESENT (gp91phox + p22phox intact on membrane — distinguishes NCF1/NCF2/NCF4 cytosol-defect CGD from CYBB/CYBA/CYBC1 membrane-defect CGD); ΔGT DELETION (two-nucleotide GT deletion at NCF1 codon 75_76) is the most common AR-CGD mutation globally (~70-80% of NCF1 alleles); EQUAL SEX DISTRIBUTION (AR inheritance — male and female equally affected; unlike X-CGD predominantly in males); milder phenotype than X-CGD overall (later onset, less Aspergillus, longer survival without HSCT)",
        "key_drug":                "PROPHYLACTIC TMP-SMX + ITRACONAZOLE/VORICONAZOLE LIFELONG (same regimen as X-CGD); IFN-γ (less evidence for AR-CGD than X-CGD; used at many centers); HSCT threshold HIGHER than X-CGD (generally reserved for severe/refractory disease or HLH-like complications); INFLIXIMAB/STEROIDS for colitis; ΔGT DELETION PCR — specific molecular test for most common mutation; cytochrome b-558 spectroscopy to distinguish from CYBB/CYBA CGD",
    },
    "CYBA": {
        "full_name":               "Cytochrome b-558 light chain / p22phox (CYBA; Autosomal Recessive CGD type III / p22-CGD)",
        "aa":                      195,
        "locus":                   "16q24.2",
        "kda":                     "22",
        "inheritance":             "AR (autosomal recessive; both sexes equally affected; ~5% of all CGD; biallelic CYBA mutations; multiple ethnic backgrounds)",
        "syndrome":                "p22-CGD — Autosomal Recessive CGD type III; p22phox absent; cytochrome b-558 fails; gp91phox also destabilized; NADPH oxidase complex absent",
        "pathway":                 "CYBA/p22phox absent → gp91phox destabilized (requires p22phox for stable membrane insertion) → cytochrome b-558 absent → NOX2 completely non-functional; also p22phox-PRR domain absent → p47phox docking site absent",
        "onset_range":             (0, 18),
        "dhr_absent_pct":          0.99,
        "liver_abscess_pct":       0.70,
        "aspergillus_pct":         0.60,
        "burkholderia_pct":        0.40,
        "ibd_colitis_pct":         0.38,
        "granulomata_pct":         0.65,
        "ifn_gamma_pct":           0.75,
        "hsct_pct":                0.50,
        "recurrent_infect_pct":    0.95,
        "pathognomonic":           "DHR ABSENT + CYTOCHROME b-558 SPECTROSCOPY ABSENT (gp91phox AND p22phox both absent — destabilize each other; same pattern as CYBB X-CGD); KEY DISTINCTION FROM CYBB/X-CGD: autosomal recessive inheritance (both sexes affected; family history shows affected females); carrier females have NORMAL (NOT bimodal) DHR histogram (unlike X-CGD where carriers show bimodal DHR); CYBA sequencing confirms; p22phox Western blot absent (and gp91phox Western blot also absent — mutual destabilization); NOX1/NOX3/NOX4/DUOX impaired in non-phagocyte tissues (p22phox is shared subunit across NOX family); some patients have lung inflammation and colitis like X-CGD",
        "key_drug":                "PROPHYLACTIC TMP-SMX + ITRACONAZOLE/VORICONAZOLE LIFELONG (same CGD prophylaxis regimen); IFN-γ; HSCT curative (for severe cases — threshold similar to X-CGD given severe phenotype); INFLIXIMAB for colitis; cytochrome b-558 spectroscopy + CYBA sequencing confirm diagnosis; mutation spectrum: missense, splice-site, small deletion/insertion variants; treatment approach identical to X-CGD",
    },
    "NCF2": {
        "full_name":               "Neutrophil cytosol factor 2 / p67phox (NCF2; Autosomal Recessive CGD type IV / p67-CGD)",
        "aa":                      526,
        "locus":                   "1q25.3",
        "kda":                     "67",
        "inheritance":             "AR (autosomal recessive; both sexes equally affected; ~5% of all CGD; biallelic NCF2 mutations; severe phenotype similar to X-CGD)",
        "syndrome":                "p67-CGD — Autosomal Recessive CGD type IV; p67phox absent; NOX2 activation domain absent; cytochrome b-558 intact but activation fails; severe CGD phenotype",
        "pathway":                 "NCF2/p67phox absent → activation domain (AD) of p67phox cannot interact with FAD-binding domain of gp91phox → FADH2 transfer blocked → no superoxide generated; Rac2-GTP binding via TPR repeat also absent → NOX2 assembly complete but catalytically inactive",
        "onset_range":             (0, 12),
        "dhr_absent_pct":          0.98,
        "liver_abscess_pct":       0.72,
        "aspergillus_pct":         0.60,
        "burkholderia_pct":        0.42,
        "ibd_colitis_pct":         0.36,
        "granulomata_pct":         0.62,
        "ifn_gamma_pct":           0.78,
        "hsct_pct":                0.52,
        "recurrent_infect_pct":    0.96,
        "pathognomonic":           "DHR ABSENT + CYTOCHROME b-558 SPECTROSCOPY PRESENT (gp91phox + p22phox intact on phagosomal membrane; membrane NOX2 complex assembled but NOT activatable without p67phox); NADPH OXIDASE ACTIVITY ABSENT despite present cytochrome b-558 — distinguishes NCF2 (and NCF1/NCF4) from CYBB/CYBA/CYBC1 where cytochrome b-558 absent; p67phox Western blot ABSENT; RAC2-GTP binding cannot occur through absent NCF2 TPR motif; onset typically early neonatal/infantile like X-CGD (p67-CGD is more severe than p47-CGD); NCF2 sequencing confirms; IBD-like colitis frequent",
        "key_drug":                "PROPHYLACTIC TMP-SMX + ITRACONAZOLE/VORICONAZOLE LIFELONG; IFN-γ; HSCT for severe disease (threshold similar to X-CGD; p67-CGD has severe X-CGD-like phenotype); gene therapy research for NCF2 (not yet approved); INFLIXIMAB/STEROIDS for colitis; p67phox protein assay (Western blot absent) + NCF2 sequencing",
    },
    "NCF4": {
        "full_name":               "Neutrophil cytosol factor 4 / p40phox (NCF4; Autosomal Recessive CGD type V / p40-CGD)",
        "aa":                      339,
        "locus":                   "22q13.1",
        "kda":                     "40",
        "inheritance":             "AR (autosomal recessive; both sexes equally affected; ~1-2% of all CGD; rarest canonical CGD gene; IBD-predominant phenotype)",
        "syndrome":                "p40-CGD — Autosomal Recessive CGD type V; rarest canonical CGD gene; predominantly IBD-like colitis + mild infections; phagosomal NOX2 targeting defect; plasma membrane burst may be preserved",
        "pathway":                 "NCF4/p40phox absent → PX domain (PI3P-binding) absent → cytosolic complex cannot be targeted to phagosomal membrane (PI3P accumulates on phagosomal membrane during ingestion) → phagosomal NOX2 activation fails; plasma membrane NOX2 activation partially preserved (explains why DHR may not be fully absent)",
        "onset_range":             (6, 36),     # later onset; IBD-like presentation often dominates
        "dhr_absent_pct":          0.60,        # DHR may be NORMAL or only partially reduced in NCF4 CGD
        "liver_abscess_pct":       0.45,
        "aspergillus_pct":         0.25,        # less Aspergillus than other CGD types
        "burkholderia_pct":        0.20,
        "ibd_colitis_pct":         0.82,        # IBD DOMINANT phenotype in NCF4 CGD
        "granulomata_pct":         0.55,
        "ifn_gamma_pct":           0.40,
        "hsct_pct":                0.15,        # HSCT rarely needed given mild phenotype
        "recurrent_infect_pct":    0.70,
        "pathognomonic":           "DHR TEST MAY BE NORMAL or only mildly reduced (NCF4/p40-CGD can be MISSED on standard DHR CGD screening); PHAGOSOMAL NOX2 ASSAY (latex bead phagosome test) ABSENT — phagosomal targeting defect; IBD-LIKE COLITIS DOMINANT PHENOTYPE (Crohn's-like intestinal disease often presenting feature, before infectious complications); MILDER INFECTIONS than other CGD types (less Aspergillus; liver abscesses present but less frequent); NCF4 p.R105Q mutation specifically disrupts PI3P binding of PX domain — loss of phagosomal targeting; cytochrome b-558 PRESENT; p40phox Western blot absent; NCF4 sequencing; differential: Crohn's disease in a child with recurrent bacterial infections → consider NCF4-CGD",
        "key_drug":                "PROPHYLACTIC TMP-SMX + ITRACONAZOLE; IBD managed with INFLIXIMAB, STEROIDS, AZATHIOPRINE (Crohn's-like colitis is often the primary management challenge); IFN-γ if infections significant; HSCT not typically required (mild phenotype); STANDARD DHR SCREENING MAY MISS NCF4-CGD — phagosomal oxidase assay (latex bead test) required; NCF4 sequencing confirms",
    },
    "CYBC1": {
        "full_name":               "Cytochrome b-245 chaperone 1 / EROS (CYBC1; AR-CGD chaperone deficiency; Ultra-rare 2019 discovery)",
        "aa":                      291,
        "locus":                   "1p22.1",
        "kda":                     "33",
        "inheritance":             "AR (autosomal recessive; both sexes equally affected; ultra-rare — fewer than 10 families globally; discovered 2019; CYBC1 = EROS chaperone for gp91phox/p22phox)",
        "syndrome":                "CYBC1/EROS-CGD — Ultra-rare AR-CGD caused by absent ER chaperone EROS; gp91phox + p22phox never reach membrane (ER degradation); functionally identical to X-CGD (cytochrome b-558 absent) but genetic test shows normal CYBB/CYBA",
        "pathway":                 "CYBC1/EROS absent → gp91phox and p22phox misfolded in ER → ERAD (ER-associated degradation) eliminates both → cytochrome b-558 absent from phagosomal membrane → NOX2 non-functional; EROS itself is NOT an oxidase component but a chaperone essential for subunit maturation",
        "onset_range":             (0, 12),
        "dhr_absent_pct":          0.99,
        "liver_abscess_pct":       0.70,
        "aspergillus_pct":         0.62,
        "burkholderia_pct":        0.42,
        "ibd_colitis_pct":         0.38,
        "granulomata_pct":         0.65,
        "ifn_gamma_pct":           0.80,
        "hsct_pct":                0.55,
        "recurrent_infect_pct":    0.97,
        "pathognomonic":           "DHR ABSENT + CYTOCHROME b-558 SPECTROSCOPY ABSENT (gp91phox + p22phox absent from membrane — same pattern as CYBB/CYBA CGD); ALL STANDARD CGD GENE SEQUENCING NORMAL (CYBB, NCF1, CYBA, NCF2, NCF4 all sequenced and negative) — when standard CGD panel is negative but phenotype = classic X-CGD-like → WHOLE EXOME SEQUENCING → CYBC1 mutations; UNIQUE MOLECULAR MECHANISM: CYBC1/EROS is an ER CHAPERONE not a NOX2 subunit — it facilitates gp91phox/p22phox folding in ER; without EROS, both subunits never mature to plasma/phagosomal membrane despite intact CYBB and CYBA genes; EROS Western blot absent (in research labs); autosomal recessive (female affected; unlike X-CGD); 2019 discovery = may be unrecognized in earlier CGD cohorts",
        "key_drug":                "PROPHYLACTIC TMP-SMX + ITRACONAZOLE/VORICONAZOLE LIFELONG; IFN-γ; HSCT curative (same approach as X-CGD given cytochrome b-558-negative phenotype); CRITICAL DIAGNOSTIC STEP: when standard CGD gene panel (CYBB/NCF1/CYBA/NCF2/NCF4) is NEGATIVE in a child with classic CGD phenotype → order whole exome or CGD-expanded panel including CYBC1; gene therapy for CYBC1 not yet in clinical trials; management identical to severe X-CGD",
    },
    "RAC2": {
        "full_name":               "Ras-related C3 botulinum toxin substrate 2 / Rac-2 GTPase (RAC2; Neutrophil Immunodeficiency Syndrome / NIS; AD-GOF or AR-LOF)",
        "aa":                      192,
        "locus":                   "22q13.1",
        "kda":                     "21",
        "inheritance":             "AD-GOF (dominant; de novo p.D57N mutation = dominant negative Rac2; Neutrophil Immunodeficiency Syndrome) or AR-LOF (rare biallelic; severe combined neutrophil dysfunction)",
        "syndrome":                "NIS — Neutrophil Immunodeficiency Syndrome; Rac2 GTPase dysfunction; partial/complete NADPH oxidase assembly failure; AD-GOF p.D57N = dominant negative; AR-LOF = severe combined defect",
        "pathway":                 "RAC2 absent/dysfunctional → Rac2-GTP cannot bind p67phox TPR domain → NOX2 assembly incomplete → partial/absent oxidative burst; also: Rac2-GTP required for actin polymerization in PMNs → combined chemotaxis, phagocytosis, degranulation, NET formation defects",
        "onset_range":             (0, 6),      # neonatal onset for AD-GOF D57N
        "dhr_absent_pct":          0.75,        # DHR REDUCED (not absent) in RAC2 NIS
        "liver_abscess_pct":       0.55,
        "aspergillus_pct":         0.50,
        "burkholderia_pct":        0.35,
        "ibd_colitis_pct":         0.30,
        "granulomata_pct":         0.40,
        "ifn_gamma_pct":           0.55,
        "hsct_pct":                0.70,        # HSCT often needed for severe NIS
        "recurrent_infect_pct":    0.92,
        "pathognomonic":           "DHR REDUCED BUT NOT ABSENT (partial oxidative burst — unlike complete CGD where DHR = flat; RAC2 NIS has intermediate DHR shift; this makes RAC2 NIS potentially missed on CGD screening if cut-off is absolute absent); COMBINED NEUTROPHIL DEFECTS: reduced oxidative burst + impaired chemotaxis (PMN migration failure toward CXCL8/fMLP) + impaired phagocytosis + reduced degranulation (all RAC2-GTP-dependent functions); RECURRENT NEONATAL INFECTIONS + INFLAMMATORY DISEASE (AD-GOF p.D57N); p.D57N MUTATION = dominant negative RAC2 — Rac2-D57N is GEF-binding competent but GTP-hydrolysis defective → traps RAC2 in GDP-bound inactive conformation AND displaces normal Rac2; de novo mutation → no family history in AD-GOF cases; RAC2 sequencing mandatory in any neonatal severe neutrophil dysfunction with intermediate DHR",
        "key_drug":                "PROPHYLACTIC ANTIBIOTICS + ANTIFUNGAL (same approach as CGD); HSCT CURATIVE AND OFTEN REQUIRED (especially for AD-GOF D57N — severe NIS with poor prognosis without HSCT; HSCT reconstitutes Rac2-normal neutrophil pool); IFN-γ (may improve residual oxidative function); G-CSF variable benefit; GENETIC COUNSELING: p.D57N is de novo AD → siblings not at risk; AR-LOF biallelic variants → 25% sibling recurrence; RAC2 protein functional assay (Rac2-GTP loading) + RAC2 sequencing",
    },
    "MPO": {
        "full_name":               "Myeloperoxidase (MPO; Myeloperoxidase Deficiency; most common phagocyte enzyme defect)",
        "aa":                      745,
        "locus":                   "17q22",
        "kda":                     "84",
        "inheritance":             "AR (autosomal recessive; most common phagocyte enzyme defect; prevalence ~1 in 4,000 individuals; p.R569W [c.1705C>T] most common mutation; usually benign in immunocompetent patients)",
        "syndrome":                "MPO Deficiency — most common phagocyte enzyme defect (1/4000); NOX2 intact → H2O2 produced normally → DHR NORMAL; MPO absent → HOCl not generated; usually asymptomatic except Candida risk in diabetics",
        "pathway":                 "MPO absent → H2O2 from NADPH oxidase cannot be converted to HOCl (hypochlorous acid) → reduced killing of Candida albicans; killing of most bacteria preserved via H2O2 and other mechanisms; NOX2 fully intact → DHR NORMAL (H2O2 still produced, oxidizes DHR normally)",
        "onset_range":             (12, 360),   # often only symptomatic in adult diabetics
        "dhr_absent_pct":          0.02,        # DHR NORMAL (NOX2 intact) — KEY DDx from CGD
        "mpo_absent_pct":          0.99,        # MPO absent by staining/assay
        "candida_pct":             0.55,        # Candida infections; DM synergy CRITICAL
        "diabetes_mellitus_pct":   0.65,        # many symptomatic cases have concurrent DM
        "aspergillus_pct":         0.10,        # Aspergillus rarely a problem in MPO deficiency alone
        "ibd_colitis_pct":         0.05,        # colitis rarely associated
        "granulomata_pct":         0.05,        # granulomata NOT typical
        "ifn_gamma_pct":           0.02,        # IFN-γ not used
        "hsct_pct":                0.01,        # HSCT essentially never required
        "recurrent_infect_pct":    0.35,        # only 5-10% of MPO-def patients have clinical disease
        "pathognomonic":           "DHR TEST NORMAL (NOX2 = INTACT; H2O2 produced normally by NADPH oxidase; H2O2 still oxidizes DHR-123 → NORMAL right-shift histogram) — KEY DISTINGUISHER from CGD where DHR ABSENT; MPO STAIN (cytochemistry on blood film) ABSENT = confirmatory (normal PMNs stain dark brown/black with MPO stain; MPO-deficient PMNs unstained); AUTOMATIC HEMATOLOGY ANALYZER (Sysmex XN) FLAGGED MPO CHANNEL — many MPO deficiency diagnoses now incidental on complete blood count analysis; DIABETES MELLITUS + MPO DEFICIENCY = CLINICALLY SIGNIFICANT COMBINATION → increased risk of disseminated Candida, Candida esophagitis, Candida meningitis (in diabetics, neutrophil H2O2 killing already impaired by hyperglycemia; removing MPO amplifies Candida vulnerability); p.R569W (c.1705C>T) most common missense mutation; routine CBC auto-analyzer MPO flag → confirm with MPO stain or ELISA",
        "key_drug":                "USUALLY NO TREATMENT REQUIRED in healthy immunocompetent individuals (compensated by H2O2-mediated killing); FLUCONAZOLE PROPHYLAXIS for symptomatic patients especially with CONCURRENT DIABETES MELLITUS (annual Candida screening; aggressive antifungal therapy at first sign of invasive Candida); OPTIMIZE GLYCEMIC CONTROL (most important intervention in diabetics with MPO deficiency — hyperglycemia compounds the MPO-absent killing defect); NO HSCT; NO IFN-GAMMA; NO long-term prophylactic antibiotics unless recurrent bacterial infections; MPO ELISA or cytochemistry confirms; educate patient about Candida risk in diabetes",
    },
}


def _rng(seed):
    r = random.Random(seed)
    return r


def _aggregate_gene(gene, idx):
    gd = GENE_DATA[gene]
    seed = SEED_BASE + idx
    rng = _rng(seed)

    n = N_PER_GENE
    cohort = []
    for i in range(n):
        s = _rng(seed * 1000 + i)
        age_m = s.randint(*gd["onset_range"])
        # XLR = male predominant; all others AR or AD = equal sex
        if gd["inheritance"].startswith("XLR"):
            sex = s.choice(["M", "M", "M", "F"])   # ~75% male (hemizygous males; carrier females)
        else:
            sex = s.choice(["M", "F"])
        cohort.append({
            "age_onset_months": age_m,
            "sex":               sex,
        })

    male_n = sum(1 for p in cohort if p["sex"] == "M")

    def _p(key):
        base = gd.get(key, 0.0)
        noise = rng.gauss(0, 0.03)
        return round(max(0.0, min(1.0, base + noise)), 3)

    return {
        "gene":                    gene,
        "full_name":               gd["full_name"],
        "aa":                      gd["aa"],
        "locus":                   gd["locus"],
        "kda":                     gd["kda"],
        "inheritance":             gd["inheritance"],
        "syndrome":                gd["syndrome"],
        "n":                       n,
        "male_n":                  male_n,
        "female_n":                n - male_n,
        "median_onset_months":     sorted(p["age_onset_months"] for p in cohort)[n // 2],
        "hsct_pct":                _p("hsct_pct"),
        "recurrent_infect_pct":    _p("recurrent_infect_pct"),
        "pathognomonic":           gd["pathognomonic"],
        "key_drug":                gd["key_drug"],
        "seed":                    seed,
    }


def generate_overview():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    total_hsct      = round(sum(c["hsct_pct"]      for c in cohorts) / len(cohorts), 3)
    total_infect    = round(sum(c["recurrent_infect_pct"] for c in cohorts) / len(cohorts), 3)

    return {
        "atlas":    ATLAS_NAME,
        "genes":    GENES,
        "total_n":  TOTAL_N,
        "seeds":    f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "hsct_pct":                    total_hsct,
            "recurrent_infections_pct":    total_infect,
        },
        "key_facts": [
            "CYBB/gp91phox (XLR Xq21.1): X-CGD — MOST COMMON CGD (~65%); DHR-123 ABSENT SHIFT PATHOGNOMONIC; CARRIER FEMALE BIMODAL DHR PATHOGNOMONIC for X-linked; LIVER ABSCESS IN CHILD = CGD UNTIL PROVEN OTHERWISE; ASPERGILLUS most lethal complication; IFN-γ 50 mcg/m² SC 3×/wk reduces infections 70%; HSCT CURATIVE; gene therapy trials ongoing",
            "NCF1/p47phox (AR 7q11.23): p47-CGD — SECOND MOST COMMON (~25%); ΔGT DINUCLEOTIDE DELETION (codon 75_76) is MOST COMMON AR-CGD MUTATION (70-80% of NCF1 alleles); equal sex distribution; cytochrome b-558 PRESENT (membrane intact, cytosol absent); milder phenotype than X-CGD; TMP-SMX + itraconazole lifelong",
            "CYBA/p22phox (AR 16q24.2): p22-CGD (~5%); p22phox absent → gp91phox ALSO destabilized (mutual dependence); cytochrome b-558 ABSENT (same pattern as CYBB); AR inheritance — equal sex (DDx from X-CGD by family history + female affected); carrier females NORMAL (not bimodal) DHR",
            "NCF2/p67phox (AR 1q25.3): p67-CGD (~5%); activation domain of p67phox absent → NOX2 assembly intact but enzymatically inactive; cytochrome b-558 PRESENT (membrane intact); severe X-CGD-like phenotype; p67phox Western blot absent; NCF2 sequencing",
            "NCF4/p40phox (AR 22q13.1): p40-CGD (1-2%, RAREST canonical CGD gene); DHR MAY BE NORMAL (phagosomal NOX2 absent but plasma membrane burst preserved); IBD-COLITIS DOMINANT PHENOTYPE; phagosomal oxidase assay (latex bead test) absent; standard DHR screening may MISS NCF4-CGD",
            "CYBC1/EROS (AR 1p22.1): Ultra-rare 2019 discovery; ER CHAPERONE for gp91phox/p22phox maturation; CYBB+CYBA genes NORMAL but cytochrome b-558 ABSENT (both subunits degraded in ER without EROS); standard CGD panel NEGATIVE → whole exome sequencing required; clinically = X-CGD severity",
            "RAC2 (AD-GOF p.D57N / AR-LOF 22q13.1): Neutrophil Immunodeficiency Syndrome (NIS); Rac2 GTPase absent/dominant-negative → PARTIAL DHR REDUCTION (not absent); COMBINED PMN DEFECTS (oxidative burst + chemotaxis + phagocytosis all impaired); de novo p.D57N = dominant negative; HSCT often required",
            "MPO/Myeloperoxidase (AR 17q22): MOST COMMON phagocyte enzyme defect (1/4000); DHR NORMAL (NOX2 intact); MPO stain ABSENT; USUALLY ASYMPTOMATIC; clinically significant ONLY with concurrent DIABETES MELLITUS (Candida risk amplified); p.R569W most common mutation; Sysmex auto-analyzer MPO flag → incidental diagnosis",
        ],
        "xlr_cgd_group":        ["CYBB"],
        "ar_membrane_cgd_group": ["CYBA", "CYBC1"],
        "ar_cytosol_cgd_group": ["NCF1", "NCF2", "NCF4"],
        "gtpase_group":         ["RAC2"],
        "mpo_group":            ["MPO"],
        "hsct_curative_group":  ["CYBB", "CYBA", "NCF2", "CYBC1", "RAC2"],
        "dhr_absent_group":     ["CYBB", "NCF1", "CYBA", "NCF2", "CYBC1"],
        "dhr_partial_group":    ["RAC2"],
        "dhr_normal_group":     ["NCF4", "MPO"],
        "cohort_summaries":      cohorts,
    }


def generate_breakdown():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    diagnostic_groups = {
        "xlr_gp91phox_xcgd": {
            "genes":          ["CYBB"],
            "mechanism":      "CYBB/gp91phox absent → cytochrome b-558 (gp91phox+p22phox heterodimer) fails assembly → NOX2 complex non-functional → no superoxide, no H2O2, no HOCl → complete oxidative burst failure; X-linked (Xq21.1) → males hemizygous = fully affected; females = carriers with bimodal DHR (50% normal + 50% CGD neutrophils)",
            "shared_feature": "X-linked recessive; males predominantly affected; female carriers diagnosed by BIMODAL DHR histogram; DHR absent in affected males; cytochrome b-558 absent (spectroscopy); liver abscesses (Staphylococcus); Aspergillus pneumonia; IBD-like colitis; granulomata; IFN-γ + TMP-SMX + itraconazole prophylaxis; HSCT curative",
            "key_ddx":        "X-CGD (CYBB) vs AR-CGD: inheritance is key — X-CGD = predominantly males in maternal family (X-linked); AR-CGD = both sexes in family. Carrier females: X-CGD carriers have BIMODAL DHR histogram (pathognomonic); p22-CGD (CYBA) carrier females have NORMAL DHR histogram. Cytochrome b-558 absent in both CYBB and CYBA; present in NCF1/NCF2/NCF4 CGD. IFN-γ has most evidence in X-CGD specifically.",
        },
        "ar_membrane_complex_cgd": {
            "genes":          ["CYBA", "CYBC1"],
            "mechanism":      "CYBA/p22phox absent (direct membrane subunit) OR CYBC1/EROS absent (ER chaperone for subunit maturation) → in both cases: gp91phox + p22phox absent from phagosomal membrane → cytochrome b-558 spectroscopy ABSENT → NOX2 non-functional; identical phenotype to X-CGD despite different gene",
            "shared_feature": "Both AR; both sexes affected equally; both cause cytochrome b-558 ABSENT (same spectroscopy pattern as X-CGD); both present as severe X-CGD-like phenotype; standard CYBB sequencing negative → test CYBA next; if CYBA also negative → test CYBC1 or exome",
            "key_ddx":        "CYBA (p22-CGD): carrier females have NORMAL DHR (not bimodal — AR); gp91phox also absent (mutual destabilization); CYBA sequencing. CYBC1/EROS: all standard CGD genes (CYBB, NCF1, CYBA, NCF2, NCF4) normal on sequencing — CGD phenotype but panel negative → suspect CYBC1; whole exome or expanded CGD panel including CYBC1; ER chaperone mechanism unique to CYBC1.",
        },
        "ar_cytosol_complex_cgd": {
            "genes":          ["NCF1", "NCF2", "NCF4"],
            "mechanism":      "Cytosolic subunits (p47phox/NCF1, p67phox/NCF2, p40phox/NCF4) absent → activated complex cannot translocate to phagosomal membrane → NOX2 activation fails; membrane cytochrome b-558 (gp91phox + p22phox) INTACT (distinguishes from membrane-defect CGD); NCF4 has unique phagosomal targeting failure (plasma membrane burst partially preserved)",
            "shared_feature": "All AR; both sexes; cytochrome b-558 PRESENT (spectroscopy); DHR absent (NCF1, NCF2) or partially preserved (NCF4 — phagosomal oxidase specifically absent); p47/p67/p40phox Western blots absent for respective genes; same pathogen spectrum as X-CGD",
            "key_ddx":        "NCF1 (p47-CGD): ΔGT deletion (most common AR-CGD mutation; specific PCR); milder than X-CGD; cytochrome b-558 present. NCF2 (p67-CGD): severe X-CGD-like; cytochrome b-558 present; p67phox absent on Western blot. NCF4 (p40-CGD): IBD-dominant; DHR may be NORMAL (standard CGD screening misses NCF4); phagosomal oxidase assay (latex bead test) required; rarest canonical CGD gene.",
        },
        "rac2_nis_partial_oxidase": {
            "genes":          ["RAC2"],
            "mechanism":      "RAC2/Rac2 GTPase dysfunction (AD-GOF p.D57N = dominant negative; AR-LOF = loss of function) → Rac2-GTP cannot bind p67phox TPR domain → NOX2 assembly incomplete → PARTIAL oxidative burst; simultaneously: Rac2-GTP required for actin cytoskeleton → combined chemotaxis, phagocytosis, degranulation, NET failure",
            "shared_feature": "Neutrophil Immunodeficiency Syndrome (NIS); partial/complete neutrophil dysfunction; COMBINED DEFECT (not just oxidative burst — also chemotaxis and phagocytosis impaired); DHR INTERMEDIATE (not absent); severe neonatal infections; HSCT often required; p.D57N = de novo dominant mutation",
            "key_ddx":        "RAC2 NIS vs CGD: DHR INTERMEDIATE/PARTIAL vs completely absent; combined neutrophil functional defects (chemotaxis + phagocytosis also impaired in NIS, not just oxidative burst); RAC2 p.D57N = de novo AD-GOF — no family history expected (unlike AR-CGD); RAC2 sequencing in any neonatal CGD-like disease with intermediate DHR or multiple concurrent neutrophil functional defects.",
        },
        "mpo_deficiency": {
            "genes":          ["MPO"],
            "mechanism":      "MPO absent → H2O2 (produced normally by NADPH oxidase) cannot be converted to HOCl (hypochlorous acid) → Candida killing impaired (Candida requires HOCl for killing); bacterial killing mostly preserved via H2O2 and other mechanisms; NOX2 = FULLY INTACT",
            "shared_feature": "Most common phagocyte enzyme defect (1/4000); AR; DHR NORMAL (NOX2 intact); MPO stain absent; usually asymptomatic in immunocompetent; Candida infections primarily with concurrent diabetes mellitus; auto-analyzer incidental detection common",
            "key_ddx":        "MPO deficiency vs CGD: DHR NORMAL in MPO deficiency vs DHR ABSENT in CGD — the critical distinguishing test; MPO stain confirms MPO absence; Sysmex MPO channel flags incidentally. MPO + DM = high Candida risk requiring antifungal prophylaxis; MPO without DM = usually no treatment. CGD (any type): recurrent Staphylococcus, Aspergillus, Burkholderia, granulomata, IBD — ABSENT DHR is the diagnostic sine qua non.",
        },
    }

    pathognomonic_features = {
        "dhr_cytometry_cgd_gold_standard": "DHR-123 (DIHYDRORHODAMINE) FLOW CYTOMETRY = GOLD STANDARD CGD TEST: PMA-stimulated neutrophils from normal individuals show RIGHTWARD SHIFT of rhodamine-123 fluorescence (as H2O2 produced by NADPH oxidase oxidizes DHR-123 → rhodamine-123); CGD neutrophils (CYBB, NCF1, CYBA, NCF2) = FLAT HISTOGRAM (no fluorescence shift); NCF4/p40-CGD = may show partial preservation; RAC2/NIS = intermediate shift; MPO deficiency = NORMAL SHIFT (NOX2 intact → H2O2 produced normally → DHR shifts normally); DHR is more sensitive and specific than NBT slide test; DHR absolutely distinguishes CGD from MPO deficiency",
        "liver_abscess_cgd_hallmark": "STAPHYLOCOCCAL LIVER ABSCESS IN A CHILD = CGD UNTIL PROVEN OTHERWISE: liver abscesses caused by catalase-positive organisms (Staphylococcus aureus, Serratia marcescens, Klebsiella pneumoniae, Aspergillus) at any age in childhood should trigger immediate DHR testing; CGD patients accumulate catalase-positive organisms in deep tissues (liver, lymph nodes, perirectal) where HOCl killing is required; catalase-positive bacteria destroy their own H2O2 → evade residual H2O2-mediated killing → survive in CGD; catalase-negative organisms (Streptococcus, Haemophilus) = killed by own H2O2 even in CGD (hence NOT typical CGD pathogens)",
        "aspergillus_cgd_lethal": "ASPERGILLUS PNEUMONIA = MOST LETHAL COMPLICATION OF CGD: Aspergillus spp. (especially A. fumigatus) are catalase-positive fungi requiring HOCl for killing; in CGD, pulmonary Aspergillus can be rapidly fatal; VORICONAZOLE is treatment of choice (NOT amphotericin first-line in CGD); ITRACONAZOLE prophylaxis prevents most cases; chest CT showing pulmonary nodules/infiltrates in CGD patient = Aspergillus until proven otherwise; BAL with galactomannan + Aspergillus PCR; start voriconazole empirically while workup pending; delay in treatment = mortality",
        "cgd_ibd_colitis": "CGD INFLAMMATORY BOWEL DISEASE (IBD-LIKE COLITIS): ~35-40% of CGD patients develop Crohn's-like intestinal disease; NOT infection-mediated but dysregulated inflammation (failure of apoptotic clearance by CGD macrophages → granulomata formation → intestinal obstruction, fistulae); presents as chronic diarrhea, abdominal pain, failure to thrive — identical to Crohn's disease; BIOPSY: granulomata with pigment-laden macrophages; respond to INFLIXIMAB, STEROIDS, AZATHIOPRINE; NCF4/p40-CGD = IBD dominant phenotype; CGD ALWAYS in differential of pediatric Crohn's disease + recurrent bacterial infections",
        "carrier_female_bimodal_dhr": "X-CGD CARRIER FEMALE BIMODAL DHR = PATHOGNOMONIC for X-linked inheritance: female carriers of CYBB mutations = 50% normal + 50% CGD neutrophils (X-inactivation is random in hematopoietic cells); DHR histogram shows TWO DISTINCT PEAKS (bimodal = one peak with normal DHR shift + one peak with absent shift); this bimodal pattern IS PATHOGNOMONIC for X-linked CGD carrier state (distinguishes from p22-CGD/CYBA carrier females who have NORMAL/UNIMODAL DHR); female carrier bimodal DHR = start CYBB sequencing immediately; occasionally carrier females have CGD symptoms due to skewed lyonization",
        "ncf4_missed_by_standard_dhr": "NCF4/p40-CGD DIAGNOSTIC PITFALL: standard DHR CGD SCREENING MAY BE NORMAL in NCF4/p40-CGD because NCF4/p40phox is specifically required for PHAGOSOMAL NOX2 activation (PI3P-dependent targeting) but not for plasma membrane NOX2 activation; plasma membrane oxidase burst (tested by standard PMA DHR protocol) may be partially or fully preserved → p40-CGD MISSED; the correct test is PHAGOSOMAL OXIDASE ASSAY (latex bead ingestion test: opsonized latex beads phagocytosed → phagosomal DHR measured); p40-CGD = absent phagosomal burst with normal/reduced plasma membrane burst; clinical clue: IBD-like colitis prominent with MILDER infections than expected for CGD",
        "mpo_vs_cgd_dhr": "MPO DEFICIENCY vs CGD: THE DHR TEST SOLVES THIS DISTINCTION IMMEDIATELY: (1) CGD (CYBB/NCF1/CYBA/NCF2) = ABSENT DHR (flat histogram; no H2O2 produced); (2) MPO DEFICIENCY = NORMAL DHR (H2O2 produced normally by intact NOX2; DHR oxidized normally); (3) MPO stain on blood film ABSENT in MPO deficiency (purple/brown staining absent in PMNs); both conditions have absent MPO cytochemistry staining in PMNs but only MPO deficiency has NORMAL DHR; MPO deficiency = incidental CBC auto-analyzer flag → confirm with MPO stain → discuss Candida risk with diabetes mellitus",
        "cgd_catalase_positive_rule": "CGD PATHOGEN RULE: ONLY CATALASE-POSITIVE ORGANISMS CAUSE CGD INFECTIONS — organisms that produce catalase enzyme destroy their own H2O2 → evade residual H2O2-mediated killing in CGD; TYPICAL CGD PATHOGENS: Staphylococcus aureus (catalase +), Aspergillus spp. (catalase +), Burkholderia cepacia (catalase +), Serratia marcescens (catalase +), Nocardia (catalase +), Chromobacterium violaceum (catalase +), Klebsiella, Salmonella; NEVER TYPICAL CGD PATHOGENS: Streptococcus pneumoniae (catalase −), Haemophilus influenzae (catalase −), Neisseria (catalase −) — they are killed by their own H2O2 even in CGD; clinical rule: a child with Streptococcal infections is NOT likely CGD; a child with recurrent Staphylococcal liver abscesses IS likely CGD",
    }

    return {
        "gene_cohorts":            cohorts,
        "diagnostic_groups":       diagnostic_groups,
        "pathognomonic_features":  pathognomonic_features,
        "cascade_testing_order": [
            "Step 1: SUSPECTED CGD (child with recurrent Staphylococcal/Aspergillus/Burkholderia/Serratia/Nocardia infections, liver abscess, IBD-like colitis, or granulomata) → ORDER DHR-123 FLOW CYTOMETRY immediately (gold standard CGD test; ABSENT = CGD confirmed; NORMAL = CGD excluded in most cases; INTERMEDIATE = consider RAC2 NIS; NORMAL with IBD history = consider NCF4-CGD with phagosomal oxidase assay)",
            "Step 2: DHR ABSENT → Determine CGD type: OBTAIN FAMILY HISTORY (male-predominant maternal lineage = X-linked; equal sex = AR); CYTOCHROME b-558 SPECTROSCOPY on PMN lysate (ABSENT spectroscopy = CYBB or CYBA or CYBC1 CGD; PRESENT spectroscopy = NCF1 or NCF2 or NCF4 CGD)",
            "Step 3: DHR ABSENT + Cytochrome b-558 ABSENT → Test CYBB FIRST (sequencing + deletion/duplication analysis); if CYBB negative → test CYBA (sequencing); if both negative → test CYBC1 (whole exome or expanded CGD panel); ALL confirm with carrier female DHR (bimodal = CYBB X-linked confirmed; unimodal = CYBA or CYBC1 AR)",
            "Step 4: DHR ABSENT + Cytochrome b-558 PRESENT → Test NCF1 FIRST (ΔGT deletion at codon 75_76 PCR; covers ~70-80% of NCF1 alleles; if ΔGT negative then NCF1 full sequencing); if NCF1 normal → test NCF2 (p67phox Western blot + NCF2 sequencing); if NCF2 normal → test NCF4 (p40phox Western blot + NCF4 sequencing; also add phagosomal oxidase assay)",
            "Step 5: DHR INTERMEDIATE or COMBINED NEUTROPHIL DEFECTS (oxidative burst + chemotaxis + phagocytosis all impaired) → ORDER RAC2 SEQUENCING (p.D57N de novo mutation most common AD-GOF; biallelic variants for AR-LOF); Rac2 protein by flow cytometry; de novo = family history negative; rac2-GTP loading assay in research labs; HSCT evaluation if severe",
            "Step 6: DHR NORMAL + MPO STAIN ABSENT on blood film → MPO DEFICIENCY CONFIRMED; check for DIABETES MELLITUS (most important modifier); if concurrent DM → fluconazole prophylaxis + Candida surveillance; if no DM → observe, no prophylaxis required; educate patient about Candida risk with future diabetes",
            "Step 7: DHR NORMAL + CLINICAL SUSPICION FOR CGD (IBD-like colitis + occasional Staphylococcal infections) → ORDER PHAGOSOMAL OXIDASE ASSAY (latex bead ingestion test; measures phagosomal H2O2 specifically) → if phagosomal burst ABSENT with normal plasma membrane burst → NCF4/p40-CGD (NCF4 sequencing + p40phox Western blot); NCF4 = IBD-dominant CGD that is MISSED by standard DHR screening",
            "Step 8: CGD CONFIRMED (any type) → PROPHYLAXIS IMMEDIATELY: TMP-SMX (PCP + bacterial prophylaxis; lifelong) + ITRACONAZOLE (Aspergillus prophylaxis; lifelong); CYBB/X-CGD additionally → IFN-γ 50 mcg/m² SC 3×/week; ASSESS FOR HSCT (discuss with transplant center; indications: severe phenotype, prior Aspergillus infection, IBD-refractory, poor quality of life despite prophylaxis)",
            "Step 9: ASPERGILLUS detected in CGD patient (chest CT nodule, BAL galactomannan +, serum galactomannan elevated) → START VORICONAZOLE IMMEDIATELY (not amphotericin first-line in CGD — voriconazole superior for Aspergillus in CGD); bronchoscopy/BAL for species identification; infliximab/steroids if severe inflammation surrounding Aspergillus lesion; surgical resection for isolated refractory pulmonary Aspergillus lesion; HSCT timing after Aspergillus clearance (not during active infection)",
            "Step 10: HSCT PLANNING for CGD → MSD (matched sibling donor) preferred; MUD (matched unrelated donor) acceptable at experienced centers; reduced-intensity conditioning (RIC) increasingly used for older/severely infected patients; engraftment monitoring (donor chimerism); confirm NOX2 function in donor chimerism (DHR on post-HSCT blood); gene therapy (lentiviral CYBB) available at specialized centers for X-CGD without suitable HSCT donor; X-linked + age <10 + no severe infections = gene therapy candidate; discuss with immunology center",
        ],
    }


def generate_definitions():
    return {
        "definitions": [
            {"term": "Chronic Granulomatous Disease (CGD)", "definition": "CGD is a rare primary immunodeficiency caused by absent or dysfunctional NADPH oxidase (NOX2) in phagocytes (neutrophils, monocytes, macrophages, eosinophils). Without NOX2: superoxide (O2•−) not generated → H2O2 not produced → MPO cannot convert H2O2 to HOCl → complete failure of oxidative burst microbicidal killing. INFECTIONS: catalase-positive organisms exclusively (Staphylococcus aureus, Aspergillus, Burkholderia cepacia, Serratia marcescens, Nocardia, Chromobacterium violaceum). INFLAMMATORY DISEASE: Crohn's-like IBD colitis (40%), granulomata (gut, bladder, esophagus, lymph nodes). DIAGNOSIS: DHR-123 flow cytometry ABSENT. GENES: CYBB (X-CGD, 65%), NCF1 (p47-CGD, 25%), CYBA (p22-CGD, 5%), NCF2 (p67-CGD, 5%), NCF4 (p40-CGD, 1-2%), CYBC1 (EROS, ultra-rare). TREATMENT: TMP-SMX + itraconazole lifelong; IFN-γ; HSCT curative."},
            {"term": "DHR-123 Flow Cytometry (CGD Diagnosis Gold Standard)", "definition": "DHR-123 (dihydrorhodamine 123) flow cytometry is the gold standard functional test for CGD diagnosis. PRINCIPLE: PMNs are stimulated with PMA (phorbol myristate acetate) which activates NADPH oxidase → in normal PMNs: O2•− → H2O2 → oxidizes DHR-123 (non-fluorescent) → rhodamine-123 (bright fluorescent) → right-shifted fluorescence histogram; in CGD PMNs: no H2O2 produced → DHR-123 stays non-fluorescent → flat histogram. INTERPRETATION: Absent shift = CGD (CYBB/NCF1/CYBA/NCF2/CYBC1); Intermediate shift = RAC2 NIS or NCF4 partial; Normal shift = MPO deficiency (NOX2 intact, only HOCl absent); BIMODAL = X-linked CGD carrier female (50% normal, 50% CGD PMNs). DHR is superior to NBT (nitroblue tetrazolium) slide test for quantitation and carrier detection."},
            {"term": "Cytochrome b-558 (NOX2 Membrane Component)", "definition": "Cytochrome b-558 is the stable membrane component of NADPH oxidase (NOX2): a heterodimer of gp91phox (CYBB; 91 kDa; electron transfer; FAD + heme groups) and p22phox (CYBA; 22 kDa; stability + p47phox docking). SPECTROSCOPY: cytochrome b-558 has a characteristic absorption spectrum (α-band at 558 nm, β-band at 426 nm at reduced state) measured in PMN lysate. ABSENT spectroscopy: CYBB, CYBA, or CYBC1 CGD (membrane components absent). PRESENT spectroscopy: NCF1, NCF2, NCF4 CGD (membrane complex intact; activation failure). This distinction is a critical diagnostic step after DHR confirms CGD."},
            {"term": "NCF1 ΔGT Deletion (Most Common AR-CGD Mutation)", "definition": "The ΔGT (GT→AT) dinucleotide deletion at NCF1 exon 2 codons 75_76 is the most common mutation causing autosomal recessive CGD worldwide (~70-80% of p47-CGD alleles). MECHANISM: NCF1 gene (7q11.23) is flanked by two pseudogenes (NCF1B, NCF1C) with >99% sequence identity → unequal crossing-over/gene conversion during meiosis transfers the GT→AT change from pseudogene to NCF1 → frameshift → premature stop → truncated, non-functional p47phox. DIAGNOSIS: specific ΔGT PCR (faster/cheaper than full NCF1 sequencing) should be ordered first in any AR-CGD with PRESENT cytochrome b-558. Homozygous ΔGT = ~60% of p47-CGD; compound heterozygous ΔGT + other variant = ~20%."},
            {"term": "Catalase-Positive vs Catalase-Negative Organisms in CGD", "definition": "CGD PATHOGEN RULE: only catalase-POSITIVE organisms cause significant CGD infections. REASON: In CGD, NOX2 is absent → no H2O2 produced by phagocyte. Catalase-positive bacteria (Staphylococcus aureus, Serratia, Burkholderia, Klebsiella, Aspergillus) produce their own catalase → destroy any trace H2O2 → completely evade even residual killing. Catalase-NEGATIVE bacteria (Streptococcus pneumoniae, Haemophilus influenzae, Neisseria meningitidis) produce H2O2 themselves → their own H2O2 accumulates in phagosome → kills them even without phagocyte NOX2 → catalase-negative organisms ARE KILLED in CGD. Clinical corollary: a child with only Streptococcal infections does NOT have CGD; a child with recurrent Staphylococcal/Aspergillus/Burkholderia infections has CGD until proven otherwise."},
            {"term": "IFN-γ in CGD (Prophylaxis and Mechanism)", "definition": "Interferon-gamma (IFN-γ) prophylaxis in CGD: DOSE: 50 mcg/m² SC three times per week. EVIDENCE: Landmark RCT (Gallin et al., NEJM 1991) showed 70% reduction in infection rate in X-CGD; FDA-approved for CGD prophylaxis. MECHANISM: IFN-γ upregulates remaining NOX2 activity in X-CGD patients with partial gp91phox expression (hypomorphic mutations); also upregulates alternative microbicidal pathways (reactive nitrogen species, tryptophan starvation via IDO), enhances macrophage killing, promotes granuloma resolution. MOST BENEFIT: X-CGD (CYBB) > AR-CGD (evidence weaker for NCF1/NCF2). SIDE EFFECTS: flu-like symptoms (premedicate with acetaminophen); autoimmune thyroiditis with long-term use."},
            {"term": "MPO Deficiency vs CGD (DHR Distinction)", "definition": "MPO deficiency is the MOST COMMON phagocyte enzyme defect (1/4000) but is NOT CGD. DISTINCTION: MPO deficiency = NORMAL DHR (NADPH oxidase intact → H2O2 produced normally → DHR oxidized to rhodamine-123 normally); CGD = ABSENT DHR (NADPH oxidase absent → no H2O2 → no DHR shift). MPO converts H2O2 → HOCl (the final, most potent oxidant); without MPO: H2O2 produced but HOCl absent; H2O2 alone kills most bacteria — clinical phenotype is MILD. CLINICAL SIGNIFICANCE: MPO deficiency clinically significant only with concurrent diabetes mellitus → Candida infections (Candida specifically requires HOCl; hyperglycemia additionally impairs neutrophil killing). DIAGNOSIS: auto-analyzer MPO channel flag (Sysmex) → confirm MPO stain absent → discuss diabetes risk. NO CGD PROPHYLAXIS OR HSCT REQUIRED for MPO deficiency."},
            {"term": "CYBC1/EROS — CGD Chaperone (2019 Discovery)", "definition": "CYBC1 (cytochrome b-245 chaperone 1, previously EROS — Essential for Reactive Oxygen Species) was discovered in 2019 as a novel CGD gene. MECHANISM: EROS is an ER-resident chaperone required for the correct folding and ER-to-Golgi trafficking of gp91phox (CYBB) and p22phox (CYBA); without EROS, both subunits are misfolded in the ER and degraded by ERAD → cytochrome b-558 absent from phagosomal membrane despite intact CYBB and CYBA genes. CLINICAL SIGNIFICANCE: CYBC1/EROS-CGD is phenotypically identical to X-CGD (cytochrome b-558 absent; DHR absent) but AR (both sexes affected); standard CGD gene panels (CYBB/NCF1/CYBA/NCF2/NCF4) are all NEGATIVE → whole exome sequencing or expanded CGD panel reveals CYBC1 variants; may be underdiagnosed in pre-2019 CGD cohorts diagnosed as 'CGD-unknown genetic etiology.'"},
            {"term": "RAC2 Neutrophil Immunodeficiency Syndrome (NIS)", "definition": "RAC2 mutations cause Neutrophil Immunodeficiency Syndrome (NIS), a combined neutrophil dysfunction different from classic CGD. RAC2 (Rac-2 GTPase, 22q13.1, 192aa) is required for NADPH oxidase assembly (Rac2-GTP binds p67phox TPR domain) AND for actin cytoskeleton regulation in neutrophils. TWO MECHANISMS: AD-GOF (p.D57N = dominant negative; displaces normal Rac2 → NADPH oxidase assembly blocked + actin defect; de novo mutation; neonatal onset) and AR-LOF (biallelic; severe combined PMN dysfunction). PHENOTYPE: recurrent severe infections; DHR INTERMEDIATE/PARTIAL (not absent — unlike classic CGD); combined defects (oxidative burst + chemotaxis + phagocytosis + degranulation). HSCT often required. Distinguish from classic CGD by intermediate DHR + multiple concurrent PMN function defects."},
            {"term": "CGD Colitis (IBD-Like Intestinal Disease)", "definition": "CGD Inflammatory Bowel Disease (IBD) occurs in 35-40% of CGD patients, affecting primarily the colon and perianal area, and resembling Crohn's disease histologically and clinically. MECHANISM: Not infectious — caused by failure of apoptotic cell clearance (efferocytosis) by CGD macrophages (require ROS for efficient apoptotic cell clearance) → granuloma formation → intestinal inflammation → obstructive granulomata of stomach, colon, esophagus, and urinary tract. PATHOLOGY: non-caseating granulomata; pigment-laden macrophages; eosinophilic infiltration; identical to Crohn's biopsy in children. TREATMENT: steroids (first-line); infliximab (TNF blockade — effective for CGD colitis and safe in prophylaxed CGD patients); azathioprine; bowel surgery for obstructive granulomata; HSCT resolves colitis by restoring phagocyte ROS function. NCF4/p40-CGD: IBD is the DOMINANT phenotype; consider NCF4-CGD in any child with Crohn's and recurrent bacterial infections."},
        ],
        "key_distinguishers": {
            "xcgd_vs_ar_cgd":         "X-CGD (CYBB): XLR; males predominantly; carrier FEMALES show BIMODAL DHR histogram (PATHOGNOMONIC for X-linked); cytochrome b-558 absent; CYBB sequencing. AR-CGD (NCF1/CYBA/NCF2/NCF4/CYBC1): both sexes; carrier females have NORMAL DHR; distinguish subtypes by cytochrome b-558 spectroscopy (absent = CYBA/CYBC1; present = NCF1/NCF2/NCF4). IFN-γ has strongest evidence in X-CGD.",
            "dhr_pattern_guide":      "DHR ABSENT = classic CGD (CYBB/NCF1/CYBA/NCF2/CYBC1). DHR INTERMEDIATE = RAC2 NIS (partial oxidase) or early collection artefact. DHR NORMAL + MPO stain absent = MPO deficiency. DHR NORMAL + clinical CGD suspicion + IBD-dominant = NCF4-CGD (do phagosomal oxidase assay with latex beads). BIMODAL DHR in female = X-linked CGD carrier.",
            "cytochrome_b558_guide":  "Cytochrome b-558 ABSENT (CGD + absent spectroscopy) = CYBB, CYBA, or CYBC1. Cytochrome b-558 PRESENT (CGD + present spectroscopy) = NCF1, NCF2, or NCF4. Carrier female DHR bimodal = CYBB. CYBC1 = b-558 absent + standard CGD panel negative → whole exome.",
            "cgd_vs_mpo_vs_rac2":     "CGD (CYBB/NCF1/etc.): DHR absent; liver abscesses; Aspergillus; IBD. MPO deficiency: DHR NORMAL; mild; Candida in diabetics only; auto-analyzer flag. RAC2 NIS: DHR intermediate; combined PMN defects; neonatal; HSCT usually needed.",
        },
        "treatment_matrix": {
            "prophylaxis":            "TMP-SMX (bacterial: PCP + Staphylococcal) + ITRACONAZOLE/VORICONAZOLE (antifungal: Aspergillus) LIFELONG in all CGD types (CYBB/NCF1/CYBA/NCF2/NCF4/CYBC1/RAC2 NIS); MPO deficiency: antifungal only if concurrent diabetes mellitus; penicillin/amoxicillin NOT adequate (not catalase-positive spectrum coverage).",
            "ifn_gamma":              "IFN-γ 50 mcg/m² SC 3×/week (MON/WED/FRI): STRONGEST EVIDENCE in X-CGD (CYBB); FDA-approved; reduces infections ~70%; premedicate with acetaminophen; used in AR-CGD at most centers (weaker evidence); NOT indicated for MPO deficiency.",
            "hsct":                   "HSCT CURATIVE: X-CGD (CYBB): strong indication especially <10yr; AR-CGD (CYBA/NCF2/CYBC1/RAC2): similar severity to X-CGD → HSCT early; NCF1 (p47-CGD): HSCT threshold higher (milder phenotype); NCF4 (p40-CGD): rarely needed; MPO deficiency: HSCT NOT indicated. MSD > MUD; myeloablative preferred for full chimerism; RIC for older/infected patients.",
            "gene_therapy":           "X-CGD (CYBB) gene therapy: lentiviral CYBB transduction of autologous CD34+ HSCs; GT4-CGD trial; MYELOID-PILOT trial; indication: X-CGD without suitable HSCT donor or patient preference. AR-CGD NCF1 gene therapy: research phase. CYBC1 gene therapy: not yet in clinical trials.",
            "colitis_treatment":      "CGD IBD: steroids (prednisone 1 mg/kg); INFLIXIMAB (TNF-alpha blockade; effective and safe in prophylaxed CGD patients with itraconazole + TMP-SMX); azathioprine (steroid-sparing); bowel surgery for obstructive granulomata only; HSCT resolves colitis in CGD (restores ROS-mediated apoptotic clearance). NCF4/p40-CGD: Crohn's-like IBD often the PRIMARY management challenge — treat as Crohn's with same medications.",
        },
    }
