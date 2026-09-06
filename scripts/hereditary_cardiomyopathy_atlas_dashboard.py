#!/usr/bin/env python3
"""Hereditary-Cardiomyopathy-Atlas — Complete 8-Gene Hereditary Cardiomyopathy Atlas
MYBPC3  (Cardiac Myosin-Binding Protein C; 1274 aa; ~150 kDa; 11p11.2; AD;
         OMIM gene 600958; FHC4 OMIM 115197;
         most common HCM gene 35-40%; haploinsufficiency mechanism;
         c.2373insG Portuguese founder; nonsense/frameshift 30%;
         Mavacamten (Camzyos FDA 2022) for obstructive HCM;
         seed SEED_BASE+0) ·
MYH7    (Beta-myosin heavy chain; 1935 aa; ~223 kDa; 14q11.2; AD;
         OMIM gene 160760; FHC1 OMIM 192600;
         2nd most common HCM gene 25-35%; missense dominant negative;
         Arg403Gln first identified mutation — malignant phenotype;
         Mavacamten targets MYH7-driven hyperdynamic state;
         seed SEED_BASE+1) ·
TNNT2   (Cardiac Troponin T; 288 aa; ~36 kDa; 1q32.1; AD;
         OMIM gene 191045; FHC2 OMIM 115195;
         HCM + DCM; 5% HCM; Arg92Trp — high SCD risk despite mild LVH;
         TNNT2-DCM: thin filament dysfunction; sarcomere haploinsufficiency;
         seed SEED_BASE+2) ·
LMNA    (Lamin A/C; 664 aa; ~74 kDa; 1q22; AD;
         OMIM gene 150330; DCM1A OMIM 115200;
         DCM + cardiac conduction disease (CCD); LMNA-DCM most common genetic DCM;
         ICD MANDATORY before conventional LVEF threshold;
         Padua score ≥4 → ICD independent of LVEF;
         seed SEED_BASE+3) ·
SCN5A   (Cardiac sodium channel Nav1.5; 2016 aa; ~227 kDa; 3p22.2; AD;
         OMIM gene 600163; Brugada1 OMIM 601144; DCM1E OMIM 601154;
         Brugada syndrome, LQTS3, DCM, sick sinus syndrome;
         Flecainide ABSOLUTELY CONTRAINDICATED in Brugada;
         Quinidine — Brugada GOF-like type 1 pattern;
         seed SEED_BASE+4) ·
DSP     (Desmoplakin; 2871 aa; ~332 kDa; 6p24.3; AD/AR;
         OMIM gene 125647; ARVC8 OMIM 607450;
         ARVC left-dominant; biallelic → Carvajal (woolly hair + keratoderma + DCM);
         woolly hair + keratoderma PATHOGNOMONIC for DSP;
         fibrofatty replacement left > right ventricle;
         seed SEED_BASE+5) ·
PKP2    (Plakophilin 2; 837 aa; ~97 kDa; 12p11.21; AD;
         OMIM gene 602861; ARVC9 OMIM 609040;
         most common ARVC gene 40%; desmosomal intercalated disc;
         epsilon wave + right ventricular fibrofatty replacement;
         Task Force Criteria 2010; exercise restriction MANDATORY;
         seed SEED_BASE+6) ·
PLN     (Phospholamban; 52 aa; ~6 kDa; 6q22.31; AD;
         OMIM gene 172405; DCM1P OMIM 609909;
         SERCA2a inhibitor; Arg14del Dutch/Belgian founder — 15x DCM risk;
         homozygous → lethal infantile DCM; ICD mandatory;
         last-resort cardiac transplantation; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1518–1525)
"""

import random

SEED_BASE = 1518

CARDIOMYOPATHY_GENES = [
    # ── MYBPC3 — Most Common HCM Gene ──
    {
        "gene": "MYBPC3",
        "protein": "Cardiac MyBP-C — Most Common HCM Gene 35-40%, Haploinsufficiency, Mavacamten FDA 2022",
        "alias": (
            "MYBPC3; OMIM gene 600958; FHC4 OMIM 115197; 11p11.2; 1274 aa; ~150 kDa; "
            "MYBPC3 encodes cardiac myosin-binding protein C (cMyBP-C), a thick-filament "
            "accessory protein located at the C-zone of the sarcomere A-band. cMyBP-C "
            "regulates cross-bridge cycling kinetics: its N-terminal domains (C0-C2) interact "
            "with both myosin S2 and actin, providing a mechanosensitive brake on actomyosin "
            "interaction. MYBPC3 is the most common HCM gene, responsible for 35-40% of "
            "genotype-positive HCM. The predominant mechanism is haploinsufficiency: truncating "
            "variants (nonsense, frameshift, splice-site) produce mRNA subject to nonsense-mediated "
            "decay, reducing cMyBP-C stoichiometry in the sarcomere. Approximately 25-30% of "
            "MYBPC3 variants are missense. c.2373insG (p.Glu791LysfsTer40) is a Portuguese "
            "founder mutation. MYBPC3 haploinsufficiency causes incomplete penetrance — "
            "many carriers remain asymptomatic until the 5th-6th decade. Wall thickness in "
            "MYBPC3-HCM tends to be moderate (18-22mm); prognosis is generally more benign "
            "than MYH7. Mavacamten (Camzyos; FDA April 2022) is a selective cardiac myosin "
            "inhibitor approved for symptomatic obstructive HCM (NYHA II-III + LVOTO "
            ">30 mmHg): it reduces cross-bridge formation and LVOTO, improving symptoms and "
            "exercise capacity (EXPLORER-HCM, VALOR-HCM trials). Echocardiographic monitoring "
            "mandatory — EF <50% requires dose reduction or discontinuation. ICD implantation "
            "per HCM SCD risk score (ESC) or Mayo criteria (AHA/ACC)."
        ),
        "aa": "1274 aa",
        "kDa": "~150 kDa",
        "locus": "11p11.2",
        "omim_gene": 600958,
        "omim_disease": 115197,
        "inheritance": "AD — haploinsufficiency; incomplete penetrance; phenotype age-dependent",
        "gene_class": (
            "MYBPC3 is a 1274-amino acid modular protein organised into C0-C10 immunoglobulin "
            "and fibronectin type III domains, plus a unique MYBPC motif. At rest, the N-terminal "
            "S2-binding domain tethers cMyBP-C to myosin, suppressing cross-bridge formation. "
            "Upon phosphorylation (PKA, PKC, CaMKII sites at Ser273, Ser282, Ser302) during "
            "adrenergic stimulation, this interaction is released, accelerating cross-bridge "
            "cycling and enhancing contractility — a key cardiac reserve mechanism. "
            "Haploinsufficiency (50% reduction in cMyBP-C) shifts the sarcomere toward "
            "hypercontractility: excess cross-bridge formation → increased ATP consumption → "
            "myocyte disarray → hypertrophic remodelling. Mavacamten restores the suppressive "
            "brake by allosterically binding myosin S1/S2 junction, stabilising the "
            "super-relaxed state (SRX) and reducing basal ATP turnover. MYBPC3 penetrance "
            "is influenced by modifier genes (ACTN2, MYOM1) and environmental factors; "
            "40% of elderly MYBPC3 carriers may have subclinical hypertrophy only identifiable "
            "by cardiac MRI late gadolinium enhancement (LGE). Pre-clinical sarcomere gene "
            "replacement therapy (AAV9-MYBPC3) is in Phase I/II trials."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("MYBPC3 truncating AD heterozygous — haploinsufficiency, obstructive HCM", 0.55),
            ("MYBPC3 missense AD heterozygous — dominant negative, moderate HCM", 0.25),
            ("MYBPC3 c.2373insG Portuguese founder — haploinsufficiency", 0.10),
            ("MYBPC3 splice-site AD heterozygous — exon skipping, mild-moderate HCM", 0.10),
        ],
        "key_alerts": [
            "MYBPC3-MAVACAMTEN-FDA-2022-Obstructive-HCM-LVOTO-30mmHg: Mavacamten (Camzyos) approved obstructive HCM NYHA II-III + LVOTO >30 mmHg — reduces cross-bridge cycling; ECHO EF mandatory monitoring; stop if EF <50%",
            "MYBPC3-ICD-SCD-Risk-Score-NOT-Just-Wall-Thickness: ICD implantation uses HCM SCD risk score (ESC 5yr) or Mayo criteria — wall thickness alone is insufficient; LGE >15% LV mass = high risk",
            "MYBPC3-INCOMPLETE-PENETRANCE-Elderly-Onset: MYBPC3 HCM often presents 5th-6th decade; a normal echo at age 30 does NOT exclude future HCM — rescreen every 3-5 years lifelong",
            "MYBPC3-BETA-BLOCKER-FIRST-LINE: Beta-blockers (metoprolol/bisoprolol) FIRST LINE for symptomatic HCM; verapamil alternative if beta-blocker contraindicated; disopyramide for refractory LVOTO",
            "MYBPC3-EXERCISE-RESTRICTION-Competitive: Competitive sports PROHIBITED in HCM — moderate recreational exercise permitted; shared decision-making with sports cardiologist",
            "MYBPC3-CASCADE-GENETIC-Testing: First-degree relatives require cardiac evaluation + genetic testing — 50% inheritance risk; echo annually if gene-positive phenotype-negative",
        ],
    },
    # ── MYH7 — Sarcomeric Dominant Negative HCM ──
    {
        "gene": "MYH7",
        "protein": "Beta-Myosin Heavy Chain — 2nd Most Common HCM Gene, Dominant Negative, Arg403Gln Malignant",
        "alias": (
            "MYH7; OMIM gene 160760; FHC1 OMIM 192600; 14q11.2; 1935 aa; ~223 kDa; "
            "MYH7 encodes the cardiac beta-myosin heavy chain (beta-MHC), the predominant "
            "motor protein of the adult ventricular sarcomere. MYH7 is the second most common "
            "HCM gene (25-35% of genotype-positive cases) and the first HCM gene identified "
            "(Geisterfer-Lowrance 1990 Cell). Unlike MYBPC3 haploinsufficiency, MYH7 HCM "
            "is predominantly caused by missense mutations acting via a dominant-negative "
            "poison-polypeptide mechanism — the mutant myosin is incorporated into the "
            "sarcomere but disrupts cross-bridge kinetics. Arg403Gln (p.Arg453Gln in the "
            "pre-protein) was the first HCM mutation identified; it impairs myosin motor "
            "function and is associated with severe HCM, high rates of sudden cardiac death "
            "(SCD), and a malignant phenotype. MYH7-HCM typically presents earlier (2nd-4th "
            "decade), with greater wall thickness (>25 mm not uncommon), more extensive LGE, "
            "and higher SCD risk than MYBPC3. Other high-risk MYH7 mutations: Arg719Trp, "
            "Arg663His, Arg453Cys, Val606Met. Mavacamten is effective in MYH7-driven "
            "obstructive HCM. Septal reduction therapy (myectomy or alcohol ablation) for "
            "drug-refractory obstructive HCM. Myectomy preferred in young patients."
        ),
        "aa": "1935 aa",
        "kDa": "~223 kDa",
        "locus": "14q11.2",
        "omim_gene": 160760,
        "omim_disease": 192600,
        "inheritance": "AD — dominant negative poison polypeptide; full penetrance; earlier onset than MYBPC3",
        "gene_class": (
            "MYH7 is a 1935-amino acid molecular motor protein with an N-terminal motor domain "
            "(ATPase catalytic core + actin-binding interface), a lever arm (essential light "
            "chain/regulatory light chain binding), and a long alpha-helical coiled-coil rod "
            "forming the thick filament backbone. The cross-bridge cycle: (1) myosin-ADP-Pi "
            "binds actin; (2) Pi release triggers power stroke (5-10 nm displacement, ~4 pN "
            "force); (3) ADP release; (4) ATP binding dissociates myosin from actin; "
            "(5) ATP hydrolysis re-cocks the lever arm — completing one cycle (~20-50 per "
            "second per head). MYH7 HCM mutations cluster in the motor domain (Arg403, Arg453, "
            "Arg663, Arg719) and the converter domain (Asp778, Gly741). Dominant-negative "
            "mechanism: mutant myosin heads with impaired ATPase kinetics (slower ATP "
            "hydrolysis, altered force generation) are incorporated into thick filaments "
            "alongside normal heads, disrupting cooperative cross-bridge cycling. The result "
            "is an overall hypercontractile state with inefficient ATP utilisation — identical "
            "end-organ effect (myocyte hypertrophy, fibrosis, disarray) despite lower per-head "
            "force generation. Energy depletion is a key HCM mechanism: the chronically "
            "energy-starved cardiomyocyte upregulates hypertrophic signalling pathways. "
            "Mavacamten targets MYH7 directly: allosteric binding to the myosin S1 domain "
            "stabilises the super-relaxed inhibited state, reducing basal cross-bridge number "
            "and ATP consumption."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("MYH7 Arg403Gln AD — malignant phenotype, early onset, high SCD", 0.20),
            ("MYH7 Arg719Trp AD — severe HCM, high SCD risk", 0.15),
            ("MYH7 other motor domain missense AD — moderate-severe HCM", 0.45),
            ("MYH7 rod domain missense AD — milder phenotype, DCM overlap", 0.20),
        ],
        "key_alerts": [
            "MYH7-ARG403GLN-MALIGNANT-High-SCD-Risk: MYH7 Arg403Gln (and Arg719Trp, Arg663His) = malignant phenotype — high SCD risk, ICD threshold lower, early specialist referral mandatory",
            "MYH7-MAVACAMTEN-Obstructive-HCM: Mavacamten approved obstructive HCM — MYH7 hypercontractile state responsive; ECHO EF monitoring mandatory; QTc monitoring required",
            "MYH7-MYECTOMY-Preferred-Young-Patients: Surgical septal myectomy preferred over alcohol ablation for drug-refractory LVOTO in young patients (<50yr) — better long-term outcomes, lower arrhythmia risk",
            "MYH7-DIGOXIN-AVOIDED: Digoxin contraindicated in obstructive HCM — positive inotropy worsens LVOTO; dihydropyridine calcium channel blockers also contraindicated in obstructive HCM",
            "MYH7-EXERCISE-RESTRICTION: Competitive sports absolutely prohibited; moderate exercise with cardiologist guidance; annual cardiac MRI + SCD risk assessment mandatory",
            "MYH7-DCM-OVERLAP-Rod-Domain: MYH7 rod domain variants can cause DCM (not HCM) — full phenotype characterisation mandatory before treatment; ICD criteria differ between HCM and DCM",
        ],
    },
    # ── TNNT2 — Troponin T HCM + DCM ──
    {
        "gene": "TNNT2",
        "protein": "Cardiac Troponin T — HCM Arg92Trp High SCD Low LVH, DCM Thin Filament, Dual Phenotype",
        "alias": (
            "TNNT2; OMIM gene 191045; FHC2 OMIM 115195 / CMD1D OMIM 601494; 1q32.1; 288 aa; ~36 kDa; "
            "TNNT2 encodes cardiac troponin T (cTnT), the tropomyosin-binding subunit of the "
            "troponin complex that anchors the regulatory unit to the thin filament. TNNT2 "
            "mutations cause both HCM (~5% of genotype-positive cases) and DCM (~5-10% of "
            "genetic DCM). TNNT2-HCM is clinically distinctive: Arg92Trp (p.Arg94Trp) causes "
            "HCM with MILD or absent LVH (<15 mm) but DISPROPORTIONATELY HIGH sudden cardiac "
            "death risk — a critically important phenotype for clinicians: echocardiography can "
            "appear near-normal while the patient is at high SCD risk. Cardiac MRI with LGE "
            "frequently reveals extensive fibrosis despite mild hypertrophy. The mechanistic "
            "basis: Arg92Trp reduces cTnT's ability to inhibit actomyosin ATPase at low Ca2+ "
            "concentrations — the regulatory OFF state is destabilised, causing basal "
            "hypercontractility and fibrosis without compensatory hypertrophy. "
            "TNNT2-DCM mutations (e.g., Arg141Trp, deletion of Lys210) impair thin filament "
            "activation — loss-of-function reduces contractile force, causing DCM. "
            "Troponin T is also the serum biomarker for myocardial injury (high-sensitivity "
            "cTnT assay); TNNT2 mutations do NOT directly affect serum cTnT interpretation."
        ),
        "aa": "288 aa",
        "kDa": "~36 kDa",
        "locus": "1q32.1",
        "omim_gene": 191045,
        "omim_disease": 115195,
        "inheritance": "AD — HCM dominant negative thin filament; DCM haploinsufficiency/dominant negative",
        "gene_class": (
            "Cardiac troponin T is a 288-amino acid natively unstructured protein that forms "
            "the structural backbone of the troponin complex (TnT-TnI-TnC). cTnT has three "
            "functional regions: (1) N-terminal tail (T1) — intrinsically disordered, binds "
            "tropomyosin on the thin filament coiled-coil overlap region with high affinity; "
            "(2) central hypervariable region — isoform-specific regulation; (3) C-terminal "
            "globular region (T2) — interacts with troponin I (TnI) and troponin C (TnC) to "
            "form the core troponin regulatory switch. The thin filament regulatory cycle: at "
            "low [Ca2+], TnI inhibitory region binds actin, blocking myosin access (OFF state); "
            "Ca2+ binds TnC → conformational change releases TnI from actin → tropomyosin "
            "moves to expose myosin-binding sites → cross-bridge formation (ON state). "
            "Arg92 in the T1-T2 linker region contacts tropomyosin; Arg92Trp disrupts this "
            "interface, destabilising the OFF state and causing constitutive partial thin "
            "filament activation even at diastolic Ca2+ concentrations — the sarcomere cannot "
            "fully relax. This ongoing low-level activation increases resting ATP consumption "
            "and cardiomyocyte stress without triggering the full hypertrophic response — "
            "explaining why LVH is mild but fibrosis (and SCD risk) is high. Omecamtiv "
            "mecarbil (cardiac myosin activator) is under investigation for TNNT2-DCM."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("TNNT2 Arg92Trp AD HCM — mild LVH, high SCD risk, extensive LGE", 0.40),
            ("TNNT2 other HCM missense AD — moderate HCM, thin filament hypercontractility", 0.25),
            ("TNNT2 DCM missense AD — thin filament hypocontractility, systolic dysfunction", 0.25),
            ("TNNT2 deletion/truncation AD DCM — haploinsufficiency, severe DCM, transplant", 0.10),
        ],
        "key_alerts": [
            "TNNT2-ARG92TRP-HIGH-SCD-MILD-LVH: TNNT2 Arg92Trp = mild/absent LVH but HIGH SCD risk — DO NOT use LV wall thickness alone to assess SCD risk; cardiac MRI LGE mandatory",
            "TNNT2-CARDIAC-MRI-MANDATORY: Cardiac MRI with LGE is ESSENTIAL in TNNT2-HCM — identifies extensive fibrosis not visible on echo; LGE >15% LV mass = ICD indication",
            "TNNT2-DUAL-PHENOTYPE-HCM-DCM: Same gene causes both HCM (gain-of-function thin filament) and DCM (loss-of-function thin filament) — variant functional classification mandatory",
            "TNNT2-ICD-LOW-THRESHOLD: ICD implantation at lower threshold than general population in TNNT2-HCM given disproportionate SCD risk; shared decision-making with inherited cardiac conditions specialist",
            "TNNT2-EXERCISE-RESTRICTION: Competitive sport prohibition applies equally to TNNT2-HCM regardless of mild phenotype — SCD risk is mutation-determined not LVH-determined",
            "TNNT2-DCM-SACUBITRIL-VALSARTAN: TNNT2-DCM treated per standard DCM protocol — sacubitril/valsartan + beta-blocker + MRA + SGLT2i per HFrEF guidelines; ICD if LVEF <35% despite OMT",
        ],
    },
    # ── LMNA — DCM + Cardiac Conduction Disease ──
    {
        "gene": "LMNA",
        "protein": "Lamin A/C — DCM + Cardiac Conduction Disease, ICD MANDATORY Before Conventional Threshold, Padua Score",
        "alias": (
            "LMNA; OMIM gene 150330; DCM1A OMIM 115200; 1q22; 664 aa; ~74 kDa; "
            "LMNA encodes Lamin A and Lamin C (alternative splicing; exons 1-10 shared; "
            "lamin A has exon 11-12 unique C-terminus), type V intermediate filament proteins "
            "that form the nuclear lamina — a meshwork underlying the inner nuclear membrane "
            "providing nuclear structural support, chromatin organisation, and mechanical "
            "transduction. LMNA-DCM (DCM1A) is the most common genetic cause of DCM with "
            "cardiac conduction disease (CCD), accounting for 5-8% of familial DCM. "
            "Clinical triad: (1) progressive DCM (typically age 20-50); (2) early CCD "
            "(AV block, sinus node dysfunction, bundle branch block — often preceding DCM); "
            "(3) supraventricular and ventricular arrhythmias (AF, VT, VF). "
            "CRITICAL: LMNA-DCM carries high risk of malignant ventricular arrhythmia and "
            "sudden death at LVEF above the conventional ICD threshold (35%). The Padua "
            "risk score (2019) — male sex, non-missense variant, LVEF <45%, non-sustained VT, "
            "LGE on CMR — identifies patients needing ICD at LVEF 35-50%. "
            "ICD should be implanted at Padua score ≥4 regardless of LVEF."
        ),
        "aa": "664 aa",
        "kDa": "~74 kDa",
        "locus": "1q22",
        "omim_gene": 150330,
        "omim_disease": 115200,
        "inheritance": "AD — nuclear envelope dysfunction; high penetrance; progressive CCD + DCM",
        "gene_class": (
            "Lamin A/C are type V intermediate filament proteins sharing N-terminal head, "
            "central coiled-coil rod, and C-terminal Ig-fold tail (CaaX motif in lamin A "
            "for farnesylation). Lamins A/C polymerise into the inner nuclear lamina meshwork "
            "with B-type lamins. Functions: (1) nuclear mechanical stability — resist "
            "deformation from cytoskeletal forces transmitted via LINC complex (nesprin/SUN "
            "proteins); (2) chromatin organisation — heterochromatin anchoring, gene "
            "repression through lamin-associated domains (LADs); (3) mechanosensing — "
            "convert cytoskeletal strain into gene expression changes; (4) DNA damage "
            "response — repair factor recruitment. In cardiomyocytes, the nucleus experiences "
            "large cyclic deformations with each contraction. LMNA mutations (missense, "
            "nonsense, frameshift, splice-site — missense ~50%) impair nuclear lamina "
            "integrity, making nuclei fragile and susceptible to mechanical rupture during "
            "contractile cycles. This triggers DNA damage signalling → p53/apoptosis → "
            "cardiomyocyte loss → DCM. Concurrently, aberrant LINC signalling activates "
            "ERK1/2 and mTOR pathways; EMERIN mislocalisation disrupts conduction system "
            "development → CCD. Temsirolimus (mTOR inhibitor) is in clinical trial (LMNA-DCM). "
            "Non-missense variants (truncating) carry higher Padua risk than missense."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("LMNA truncating AD (nonsense/frameshift) — high Padua score, DCM + severe CCD", 0.40),
            ("LMNA missense AD — DCM + CCD, variable penetrance, moderate Padua risk", 0.45),
            ("LMNA splice-site AD — exon skipping, truncated lamin, severe DCM", 0.10),
            ("LMNA homozygous/compound het — severe early-onset DCM, progeroid overlap", 0.05),
        ],
        "key_alerts": [
            "LMNA-ICD-MANDATORY-Padua-Score-4-Regardless-LVEF: LMNA-DCM + Padua score ≥4 → ICD MANDATORY regardless of LVEF; do NOT wait for LVEF <35%; male sex + non-missense + NSVT + LGE each add score",
            "LMNA-CCD-PRECEDES-DCM: Cardiac conduction disease (AV block/BBB/sick sinus) often precedes DCM in LMNA by years — annual ECG mandatory from diagnosis; pacing may be needed before DCM develops",
            "LMNA-AF-HIGH-STROKE-RISK: Atrial fibrillation in LMNA-DCM = high stroke risk; anticoagulation mandatory; rhythm control preferred (AF → VT/VF risk higher in LMNA)",
            "LMNA-NON-MISSENSE-HIGHEST-RISK: Truncating LMNA variants (nonsense, frameshift, splice-site) carry higher malignant arrhythmia risk than missense — lower ICD threshold in truncating variants",
            "LMNA-CARDIAC-MRI-MANDATORY-LGE: LGE on cardiac MRI (mid-wall/septal pattern) is an independent Padua risk factor; CMR mandatory in all LMNA carriers before ICD decision",
            "LMNA-CASCADE-TESTING-Urgent: First-degree relatives require urgent cascade testing + ECG — CCD can cause sudden death before DCM becomes apparent; 50% inheritance risk",
        ],
    },
    # ── SCN5A — Nav1.5 Brugada/DCM/LQTS3 ──
    {
        "gene": "SCN5A",
        "protein": "Nav1.5 Cardiac Sodium Channel — Brugada Flecainide ABSOLUTE CI, LQTS3, DCM, Sick Sinus",
        "alias": (
            "SCN5A; OMIM gene 600163; Brugada1 OMIM 601144; LQTS3 OMIM 603830; DCM1E OMIM 601154; "
            "3p22.2; 2016 aa; ~227 kDa; "
            "SCN5A encodes the alpha-subunit of the cardiac voltage-gated sodium channel "
            "(Nav1.5), which underlies the rapid upstroke (Phase 0) of the cardiac action "
            "potential in working myocardium and the His-Purkinje system. SCN5A has remarkably "
            "pleiotropic effects: different mutation classes cause distinct arrhythmia syndromes "
            "via opposite mechanisms. Loss-of-function (LOF) variants (haploinsufficiency, "
            "trafficking defects, gating shifts) → reduced INa → Brugada syndrome (BrS), "
            "sick sinus syndrome, conduction disease, progressive cardiac conduction defect. "
            "Gain-of-function (GOF) variants → increased persistent INa (late current) → "
            "prolonged APD → LQTS3 (third most common LQTS). Certain SCN5A variants cause "
            "DCM through cardiomyocyte apoptosis triggered by Na+/Ca2+ overload. "
            "CRITICAL: Sodium channel blockers (flecainide, propafenone, ajmaline) are "
            "ABSOLUTELY CONTRAINDICATED in Brugada syndrome — they exacerbate conduction "
            "failure and can precipitate ventricular fibrillation. The ajmaline/flecainide "
            "challenge is the DIAGNOSTIC test for Brugada but must be performed in a setting "
            "with defibrillation capability and immediately reversed."
        ),
        "aa": "2016 aa",
        "kDa": "~227 kDa",
        "locus": "3p22.2",
        "omim_gene": 600163,
        "omim_disease": 601144,
        "inheritance": "AD — LOF → Brugada/CCD; GOF → LQTS3; pleiotropy; incomplete penetrance",
        "gene_class": (
            "Nav1.5 is a 2016-amino acid alpha-subunit forming a 24-transmembrane segment "
            "protein with four homologous domains (I-IV), each with six segments (S1-S6). "
            "The S4 segments (positively charged arginine/lysine residues) serve as voltage "
            "sensors — depolarisation causes outward S4 movement, opening the channel pore "
            "formed by S5-S6 loops. Selectivity filter (DEKA ring: Asp-Glu-Lys-Ala) confers "
            "sodium selectivity. Fast inactivation: the III-IV linker 'IFM motif' (Ile-Phe-Met) "
            "blocks the pore within milliseconds. Slow inactivation: a separate mechanism "
            "operating over seconds. Nav1.5 current (INa) underlies: (1) rapid action "
            "potential upstroke (Phase 0: +250 V/s in working myocardium); (2) conduction "
            "velocity in His-Purkinje; (3) minimum persistent INa contributing to Phase 3. "
            "Brugada LOF mutations reduce INa by: trafficking defects (truncations), altered "
            "voltage dependence (right-shifted V½ activation), faster inactivation, or "
            "haploinsufficiency. The right ventricular epicardium has intrinsically less "
            "reserve INa density due to Ito (transient outward current) predominance — "
            "LOF Nav1.5 causes Phase 2 re-entry uniquely in the RV epicardium → Brugada ST "
            "elevation. LQTS3 GOF mutations slow fast inactivation (IFM motif disruption), "
            "creating persistent inward late INa throughout the plateau → QT prolongation → "
            "early afterdepolarisations → torsades de pointes. Quinidine reduces Brugada "
            "arrhythmias by blocking Ito, rebalancing Phase 2; mexiletine reduces late INa "
            "in LQTS3."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("SCN5A LOF AD Brugada — type 1 pattern, fever-unmasked, ICD/quinidine", 0.40),
            ("SCN5A GOF AD LQTS3 — persistent late INa, QTc >500ms, mexiletine", 0.25),
            ("SCN5A LOF AD DCM+CCD — Na/Ca overload, progressive conduction disease, ICD", 0.20),
            ("SCN5A LOF AD sick-sinus + overlap — sinus arrest, brady-tachycardia syndrome", 0.15),
        ],
        "key_alerts": [
            "SCN5A-BRUGADA-FLECAINIDE-PROPAFENONE-ABSOLUTE-CI: Sodium channel blockers (flecainide, propafenone, ajmaline) ABSOLUTELY CONTRAINDICATED in Brugada syndrome — worsen conduction failure, precipitate VF; ajmaline challenge is DIAGNOSTIC only in monitored setting with defibrillator",
            "SCN5A-BRUGADA-FEVER-TRIGGER: Fever is the most common Brugada VF trigger — antipyretics (paracetamol) IMMEDIATELY for any fever; sodium channel blocker-containing medications must be reviewed",
            "SCN5A-LQTS3-MEXILETINE: Mexiletine (INa late blocker) reduces QTc in LQTS3; betablocker partially effective; avoid drugs prolonging QT (see CredibleMeds); ICD for QTc >500 or recurrent syncope",
            "SCN5A-QUINIDINE-BRUGADA: Quinidine (Ito blocker) reduces Brugada arrhythmia burden — ICD preferred for high-risk; quinidine bridge or for asymptomatic type-1 pattern with inducible VF",
            "SCN5A-DRUG-CHECK-MANDATORY: Brugadrugs.org list mandatory before prescribing ANY drug to SCN5A-Brugada patient — many common medications (psychotropics, antifungals, prokinetics) can unmask Brugada",
            "SCN5A-ICD-HIGH-RISK: ICD mandatory for aborted SCD or syncope in Brugada; asymptomatic type-1 pattern: electrophysiology study + shared decision-making",
        ],
    },
    # ── DSP — Left-Dominant ARVC / Carvajal ──
    {
        "gene": "DSP",
        "protein": "Desmoplakin — ARVC Left-Dominant, Woolly Hair + Keratoderma PATHOGNOMONIC, Biallelic Carvajal",
        "alias": (
            "DSP; OMIM gene 125647; ARVC8 OMIM 607450; Carvajal OMIM 605676; 6p24.3; 2871 aa; ~332 kDa; "
            "DSP encodes desmoplakin, the most abundant protein of the cardiac desmosome. "
            "Desmosomes are specialised cell-cell adhesion junctions at the intercalated disc "
            "that mechanically couple adjacent cardiomyocytes during contraction. Desmoplakin "
            "forms the bridge between desmosomal cadherins (desmoglein/desmocollin) and the "
            "intermediate filament cytoskeleton (desmin). DSP mutations cause ARVC — a "
            "progressive replacement of ventricular myocardium with fibrofatty tissue. "
            "DSP-ARVC has a distinctly left-dominant or biventricular phenotype (unlike PKP2 "
            "which is predominantly right-ventricular): LV involvement is common (LV wall "
            "motion abnormalities, LV fibrosis on LGE) and can even be an isolated LMNA-like "
            "DCM phenotype with predominantly LV involvement. "
            "CRITICAL CLINICAL CLUE: Woolly hair (tightly coiled, curly hair texture change) "
            "AND palmoplantar keratoderma (thickening of skin on palms/soles) are "
            "PATHOGNOMONIC for DSP-ARVC heterozygous; biallelic DSP = Carvajal syndrome "
            "(woolly hair + keratoderma + DCM — not ARVC). When a young patient with "
            "unexplained DCM/ARVC has woolly hair, DSP must be excluded."
        ),
        "aa": "2871 aa",
        "kDa": "~332 kDa",
        "locus": "6p24.3",
        "omim_gene": 125647,
        "omim_disease": 607450,
        "inheritance": "AD — haploinsufficiency ARVC (left-dominant); AR biallelic → Carvajal (DCM + woolly hair + keratoderma)",
        "gene_class": (
            "Desmoplakin is a 2871-amino acid obligate desmosomal linker protein organised into "
            "three domains: (1) N-terminal head (NTP) — binds the desmosomal plaque proteins "
            "plakophilin (PKP2) and plakoglobin (JUP), anchoring DSP to the desmosomal outer "
            "dense plaque; (2) central coiled-coil rod — antiparallel homodimerisation, "
            "forming the desmoplakin dimer backbone; (3) C-terminal plakin repeat domains "
            "(DPCT): three subdomains (A, B, C) bind intermediate filaments — specifically "
            "desmin (in cardiomyocytes), keratins (in epithelial cells). The desmosome "
            "assembly: desmoglein/desmocollin cadherin ectodomains form Ca2+-dependent "
            "trans-homophilic bonds at the cell surface; intracellularly, plakoglobin and "
            "PKP2 recruit DSP; DSP C-terminus then captures desmin intermediate filaments, "
            "creating a transcellular mechanical continuum. Under mechanical load during "
            "systole, this chain distributes contractile forces across the intercalated disc "
            "preventing membrane tearing. DSP haploinsufficiency → desmosome disruption → "
            "mechanical stress → cardiomyocyte detachment and death → inflammatory response "
            "(lymphocytic myocarditis in early ARVC) → adipogenesis (WNT signalling "
            "suppression by nuclear plakoglobin → adipocytes replace dead myocytes). "
            "DSP C-terminal variants (binding desmin) show higher LV involvement; N-terminal "
            "variants more RV-dominant. Biallelic DSP truncation → complete desmoplakin "
            "absence → Carvajal: severe DCM + woolly hair + keratoderma (distinct from ARVC)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("DSP truncating AD heterozygous — left-dominant ARVC, LGE LV, woolly hair", 0.50),
            ("DSP missense AD heterozygous — ARVC/DCM overlap, variable phenotype", 0.25),
            ("DSP biallelic (Carvajal) — AR, woolly hair + keratoderma + DCM, childhood", 0.10),
            ("DSP N-terminal missense AD — predominantly RV-ARVC, Task Force positive", 0.15),
        ],
        "key_alerts": [
            "DSP-WOOLLY-HAIR-KERATODERMA-PATHOGNOMONIC: Woolly hair + palmoplantar keratoderma in DCM/ARVC patient = DSP until proven otherwise; biallelic DSP = Carvajal syndrome (DCM, NOT ARVC)",
            "DSP-LEFT-DOMINANT-ARVC-LV-INVOLVEMENT: DSP-ARVC frequently involves left ventricle — do NOT exclude ARVC diagnosis because RV appears normal; LV LGE on CMR is characteristic DSP pattern",
            "DSP-EXERCISE-RESTRICTION-MANDATORY: Exercise is the strongest environmental trigger for ARVC progression — competitive sport ABSOLUTELY prohibited; DSP gene-positive phenotype-negative individuals should also restrict exercise",
            "DSP-ICD-LV-LGE-TRIGGER: LGE of LV in DSP-ARVC is an independent SCD risk factor — ICD implantation indicated when LV LGE + reduced LVEF, NSVT, or syncope",
            "DSP-FLECAINIDE-CI-ARVC: Flecainide and other sodium channel blockers are CONTRAINDICATED in ARVC — pro-arrhythmic in fibrofatty myocardium; amiodarone if AAD needed",
            "DSP-CARDIAC-MRI-TASK-FORCE: CMR is essential for Task Force Criteria assessment in DSP — LV LGE + focal wall motion abnormalities may be the ONLY imaging findings; echo can be normal",
        ],
    },
    # ── PKP2 — Most Common ARVC Gene ──
    {
        "gene": "PKP2",
        "protein": "Plakophilin 2 — Most Common ARVC Gene 40%, RV Fibrofatty, Epsilon Wave, Task Force Criteria",
        "alias": (
            "PKP2; OMIM gene 602861; ARVC9 OMIM 609040; 12p11.21; 837 aa; ~97 kDa; "
            "PKP2 encodes plakophilin 2, a desmosomal armadillo repeat protein that is the "
            "most common genetic cause of arrhythmogenic right ventricular cardiomyopathy "
            "(ARVC), accounting for 40% of genotype-positive cases. PKP2 localises to "
            "the desmosomal outer dense plaque and the nucleus. At the desmosome, PKP2 "
            "bridges desmosomal cadherins (desmoglein-2/desmocollin-2) to desmoplakin (DSP) "
            "and plakoglobin (JUP). PKP2 also activates beta-catenin/Wnt signalling via "
            "nuclear localisation, maintaining cardiomyocyte identity and suppressing adipogenesis. "
            "PKP2-ARVC is predominantly RIGHT ventricular: the 2010 Task Force Criteria define "
            "major and minor criteria for RV structural (echo wall motion, CMR), electrical "
            "(epsilon wave, terminal activation duration, late potentials, VT with LBBB "
            "morphology — RV origin), histological (fibrofatty replacement >3 adipocytes "
            "within 3 high-power fields), and genetic (pathogenic variant in one of 5 "
            "desmosomal genes) categories. Epsilon wave (E-wave: small deflection after "
            "QRS on signal-averaged ECG and in V1-V3) represents delayed activation of "
            "fibrofatty tissue replacing the RV free wall — a major electrical criterion. "
            "VT in PKP2-ARVC has LBBB morphology (RV origin) and inferior axis."
        ),
        "aa": "837 aa",
        "kDa": "~97 kDa",
        "locus": "12p11.21",
        "omim_gene": 602861,
        "omim_disease": 609040,
        "inheritance": "AD — desmosomal haploinsufficiency; RV-predominant fibrofatty replacement; exercise-accelerated",
        "gene_class": (
            "PKP2 is an 837-amino acid armadillo repeat protein (Arm-RPT motifs 1-9) with a "
            "unique N-terminal head domain (NTP) and C-terminal arm domain. The nine armadillo "
            "repeats form a right-handed superhelix that mediates protein-protein interactions "
            "with the desmosomal cadherin cytoplasmic tails (desmoglein-2 NTP-binding site), "
            "desmoplakin N-terminus (via the arm domain), and plakoglobin. PKP2 is also "
            "found in the nucleus where it modulates RNA polymerase III transcription "
            "and interacts with beta-catenin/TCF — nuclear PKP2 loss may contribute to "
            "adipogenic transformation in ARVC. The ARVC pathogenesis sequence: PKP2 "
            "haploinsufficiency → reduced desmosomal density at intercalated disc → "
            "mechanical uncoupling under high-load conditions (exercise) → cardiomyocyte "
            "apoptosis → inflammatory response (early myocarditis phase, troponin release, "
            "oedema on CMR) → Wnt/beta-catenin suppression → adipogenic and fibrotic "
            "remodelling (late fibrofatty replacement) → conduction slowing in fibrofatty "
            "tissue → re-entrant VT → SCD. The RV is preferentially affected because: "
            "thinner wall, higher wall stress per unit area, and lower intercalated disc "
            "density compared to LV. PKP2 c.2013delC (p.Pro672ArgfsTer5) is one of the "
            "most common ARVC truncating variants."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("PKP2 truncating AD — most common ARVC, RV fibrofatty, epsilon wave", 0.55),
            ("PKP2 missense AD — ARVC, variable penetrance, later onset", 0.25),
            ("PKP2 large deletion AD (MLPA) — severe ARVC, early SCD risk", 0.10),
            ("PKP2 compound heterozygous — biallelic, severe early ARVC, neonatal onset", 0.10),
        ],
        "key_alerts": [
            "PKP2-EXERCISE-ARVC-ACCELERATES: Exercise is the STRONGEST modifiable ARVC trigger — competitive sport ABSOLUTELY prohibited; even moderate exercise accelerates fibrofatty progression; exercise restriction = disease modification",
            "PKP2-LBBB-VT-RV-ORIGIN: VT in PKP2-ARVC has LBBB morphology with inferior or superior axis (RV origin) — distinguish from fascicular VT (RBBB morphology); EP study mandatory for VT substrate characterisation",
            "PKP2-EPSILON-WAVE-MAJOR-CRITERION: Epsilon wave (small deflection after QRS in V1-V3 on standard or signal-averaged ECG) is a MAJOR Task Force Criterion — specifically request signal-averaged ECG",
            "PKP2-MLPA-MANDATORY-Negative-Sequencing: ARVC panel-negative patients with strong Task Force criteria — MLPA mandatory to exclude PKP2 large exon deletions/duplications missed by sequencing",
            "PKP2-FLECAINIDE-SOTALOL-CI: Class IC antiarrhythmics (flecainide) CONTRAINDICATED in ARVC; sotalol may worsen; amiodarone preferred if AAD required; catheter ablation for recurrent VT",
            "PKP2-ICD-PRIMARY-PREVENTION: ICD indicated for PKP2-ARVC with NSVT + LVEF <45% + RV dysfunction — primary prevention SCD; ICD reduces VF mortality; shared decision with ARVC specialist",
        ],
    },
    # ── PLN — Dutch Founder DCM ──
    {
        "gene": "PLN",
        "protein": "Phospholamban — SERCA2a Inhibitor, Arg14del Dutch/Belgian Founder, ICD Mandatory, Transplant",
        "alias": (
            "PLN; OMIM gene 172405; DCM1P OMIM 609909; 6q22.31; 52 aa; ~6 kDa; "
            "PLN encodes phospholamban, a 52-amino acid single-pass transmembrane "
            "micropeptide that is the master regulator of cardiac SERCA2a (sarcoplasmic "
            "reticulum calcium ATPase 2a) in the cardiac sarcoplasmic reticulum (SR). "
            "PLN inhibits SERCA2a in its unphosphorylated state by direct protein-protein "
            "interaction, reducing SR calcium reuptake rate and decreasing SR calcium "
            "content — this 'braking' mechanism is the basis for cardiac beta-adrenergic "
            "reserve (PKA phosphorylation of Ser16 and CaMKII phosphorylation of Thr17 "
            "releases PLN from SERCA2a, enabling the positive lusitropic and inotropic "
            "effects of catecholamines). PLN p.Arg14del (c.40_42delAGA, deletion of "
            "arginine at position 14 in the cytoplasmic domain) is a Dutch/Belgian founder "
            "mutation with a carrier frequency of 1 in 400-800 in the Netherlands, "
            "conferring ~15-fold increased DCM risk. Arg14del impairs the PKA phosphorylation "
            "site, making PLN constitutively active — SERCA2a is chronically inhibited, "
            "SR calcium depleted, leading to systolic dysfunction and DCM. Homozygous "
            "Arg14del → lethal infantile DCM (neonatal heart failure). ICD mandatory — "
            "high risk of malignant VT/VF. Many patients ultimately require cardiac "
            "transplantation."
        ),
        "aa": "52 aa",
        "kDa": "~6 kDa",
        "locus": "6q22.31",
        "omim_gene": 172405,
        "omim_disease": 609909,
        "inheritance": "AD — constitutive SERCA2a inhibition; Arg14del Dutch/Belgian founder 1/400-800 NL; AR homozygous = lethal neonatal DCM",
        "gene_class": (
            "Phospholamban (PLN) is a 52-amino acid type II transmembrane protein forming "
            "pentameric channels in the SR membrane. PLN has three domains: "
            "(1) N-terminal cytoplasmic domain Ia (residues 1-22) — disordered, contains "
            "Ser16 (PKA phosphorylation site) and Thr17 (CaMKII site) and Arg14 (required "
            "for phosphorylation-dependent regulation); (2) amphipathic helix Ib "
            "(residues 23-30) — membrane-water interface; (3) C-terminal transmembrane "
            "helix II (residues 31-52) — anchors PLN in SR membrane and mediates "
            "SERCA2a interaction within the transmembrane domain. PLN-SERCA2a interaction: "
            "unphosphorylated PLN contacts SERCA2a transmembrane helices M2, M4, M6, M9, "
            "reducing apparent Ca2+ affinity (Km from ~0.2 to 0.8 μM) and slowing "
            "conformational cycling from E2 (Ca2+-free) to E1 (Ca2+-bound) state — net "
            "result is slower SR reuptake and reduced SR Ca2+ load. Beta-adrenergic "
            "stimulation → cAMP → PKA → phospho-Ser16-PLN → PLN dissociates from SERCA2a "
            "→ SERCA2a activity doubles → SR Ca2+ rapidly restored → positive lusitropy "
            "(faster relaxation) and positive inotropy (larger Ca2+ transient). Arg14del: "
            "Arg14 is required for PKA substrate recognition; deletion eliminates "
            "phosphorylation, locking PLN in the inhibitory state — even under maximal "
            "adrenergic stimulation, SERCA2a remains inhibited. SR Ca2+ depletion → "
            "reduced peak [Ca2+] transient → reduced systolic force → DCM progression. "
            "Gene therapy (AAV-SERCA2a; CUPID trial) failed to improve outcomes in general "
            "DCM; targeted PLN/SERCA2a approaches are in pre-clinical investigation."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("PLN Arg14del AD heterozygous — Dutch/Belgian founder, DCM, 15x risk", 0.70),
            ("PLN Arg14del AD homozygous — lethal neonatal DCM, transplant", 0.05),
            ("PLN other missense AD — SERCA2a binding domain, DCM, arrhythmia", 0.15),
            ("PLN frameshift/truncating AD — SERCA2a regulatory loss, DCM", 0.10),
        ],
        "key_alerts": [
            "PLN-ARG14DEL-DUTCH-FOUNDER-15x-DCM-Risk: PLN Arg14del is a Dutch/Belgian founder mutation (1/400-800 Netherlands) — cascade testing of all first-degree relatives mandatory; 50% risk per first-degree relative",
            "PLN-ICD-MANDATORY-Malignant-VT: PLN-DCM has high malignant VT/VF risk — ICD mandatory when LVEF <35% or NSVT present; recurrent ICD shocks may indicate need for transplant listing",
            "PLN-HOMOZYGOUS-NEONATAL-LETHAL-DCM: Homozygous PLN Arg14del = lethal infantile DCM — emergency cardiac transplantation; both parents are heterozygous carriers (autosomal recessive homozygous risk if both parents carry Arg14del)",
            "PLN-SACUBITRIL-VALSARTAN-MAXIMUM-DOSE: PLN-DCM responds to maximally tolerated sacubitril/valsartan + carvedilol + eplerenone + SGLT2i — guideline-directed medical therapy mandatory before transplant evaluation",
            "PLN-CARDIAC-TRANSPLANT-LAST-RESORT: PLN-DCM frequently progresses to end-stage heart failure — transplant listing should be discussed early; support with LVAD as bridge to transplant if needed",
            "PLN-WEARABLE-DEFIBRILLATOR: Wearable cardioverter defibrillator (LifeVest) as bridge during waiting period for ICD implantation if acutely decompensated PLN-DCM",
        ],
    },
]


def _make_cohort(gd):
    r = random.Random(gd["seed"])
    gene = gd["gene"]
    pts = []
    etiols = gd["etiologies"]
    weights = [e[1] for e in etiols]
    labels = [e[0] for e in etiols]

    for i in range(gd["n_patients"]):
        # Assign etiology
        roll = r.random()
        cumul = 0.0
        etiol = labels[-1]
        for lbl, wt in zip(labels, weights):
            cumul += wt
            if roll < cumul:
                etiol = lbl
                break

        sex = "M" if r.random() < 0.55 else "F"

        # Age of onset — varies by gene
        if gene in ("MYBPC3",):
            onset_y = max(18, r.gauss(48, 14))
        elif gene in ("MYH7",):
            onset_y = max(12, r.gauss(36, 14))
        elif gene in ("TNNT2",):
            onset_y = max(14, r.gauss(38, 12))
        elif gene in ("LMNA",):
            onset_y = max(20, r.gauss(38, 10))
        elif gene in ("SCN5A",):
            onset_y = max(14, r.gauss(34, 14))
        elif gene in ("DSP",):
            onset_y = max(16, r.gauss(32, 12))
        elif gene in ("PKP2",):
            onset_y = max(14, r.gauss(30, 12))
        elif gene in ("PLN",):
            onset_y = max(10, r.gauss(32, 10))
        else:
            onset_y = max(10, r.gauss(35, 12))

        onset_y = round(onset_y, 1)

        # Diagnostic delay
        dx_delay_m = max(1, round(r.gauss(18, 14)))

        flags = {}

        if gene == "MYBPC3":
            flags["obstructive_hcm"] = r.random() < 0.65
            flags["lvoto_30mmhg"] = flags["obstructive_hcm"] and r.random() < 0.72
            flags["mavacamten_prescribed"] = flags["lvoto_30mmhg"] and r.random() < 0.42
            flags["icd_implanted"] = r.random() < 0.38
            flags["beta_blocker"] = r.random() < 0.80
            flags["exercise_restricted"] = r.random() < 0.70
            flags["cascade_tested"] = r.random() < 0.62
            flags["cardiac_mri_done"] = r.random() < 0.78
            flags["lge_extensive"] = flags["cardiac_mri_done"] and r.random() < 0.32
            flags["septal_reduction"] = flags["obstructive_hcm"] and r.random() < 0.12

        elif gene == "MYH7":
            flags["arg403gln"] = "Arg403Gln" in etiol
            flags["obstructive_hcm"] = r.random() < 0.58
            flags["lvoto_30mmhg"] = flags["obstructive_hcm"] and r.random() < 0.68
            flags["mavacamten_prescribed"] = flags["lvoto_30mmhg"] and r.random() < 0.38
            flags["icd_implanted"] = r.random() < 0.50
            flags["beta_blocker"] = r.random() < 0.82
            flags["exercise_restricted"] = r.random() < 0.78
            flags["cascade_tested"] = r.random() < 0.60
            flags["cardiac_mri_done"] = r.random() < 0.80
            flags["lge_extensive"] = flags["cardiac_mri_done"] and r.random() < 0.48
            flags["septal_myectomy"] = flags["obstructive_hcm"] and r.random() < 0.15

        elif gene == "TNNT2":
            flags["hcm_phenotype"] = "HCM" in etiol or "Arg92Trp" in etiol
            flags["dcm_phenotype"] = "DCM" in etiol
            flags["mild_lvh"] = flags["hcm_phenotype"] and r.random() < 0.65
            flags["high_scd_risk"] = "Arg92Trp" in etiol or r.random() < 0.25
            flags["icd_implanted"] = r.random() < 0.48
            flags["cardiac_mri_done"] = r.random() < 0.75
            flags["lge_extensive"] = flags["cardiac_mri_done"] and r.random() < 0.52
            flags["exercise_restricted"] = r.random() < 0.72
            flags["cascade_tested"] = r.random() < 0.55
            flags["sacubitril_valsartan"] = flags["dcm_phenotype"] and r.random() < 0.62

        elif gene == "LMNA":
            flags["ccd_precedes_dcm"] = r.random() < 0.55
            flags["av_block"] = r.random() < 0.58
            flags["pacemaker_implanted"] = flags["av_block"] and r.random() < 0.72
            flags["icd_implanted"] = r.random() < 0.62
            flags["padua_score_4plus"] = r.random() < 0.50
            flags["lvef_below_45"] = r.random() < 0.48
            flags["nsvt"] = r.random() < 0.42
            flags["af"] = r.random() < 0.38
            flags["anticoagulated"] = flags["af"] and r.random() < 0.82
            flags["cardiac_mri_done"] = r.random() < 0.70
            flags["lge_midwall"] = flags["cardiac_mri_done"] and r.random() < 0.60
            flags["cascade_tested"] = r.random() < 0.58
            flags["non_missense"] = "truncating" in etiol.lower() or "nonsense" in etiol.lower() or "frameshift" in etiol.lower() or "splice" in etiol.lower()

        elif gene == "SCN5A":
            flags["brugada"] = "Brugada" in etiol
            flags["lqts3"] = "LQTS3" in etiol
            flags["dcm_scn5a"] = "DCM" in etiol
            flags["flecainide_received"] = flags["brugada"] and r.random() < 0.08  # should never happen
            flags["fever_triggered"] = flags["brugada"] and r.random() < 0.45
            flags["quinidine_prescribed"] = flags["brugada"] and r.random() < 0.28
            flags["mexiletine_prescribed"] = flags["lqts3"] and r.random() < 0.52
            flags["icd_implanted"] = r.random() < 0.55
            flags["qtc_prolonged"] = flags["lqts3"] and r.random() < 0.72
            flags["drug_check_done"] = r.random() < 0.60
            flags["cascade_tested"] = r.random() < 0.52
            flags["ajmaline_challenge"] = flags["brugada"] and r.random() < 0.42

        elif gene == "DSP":
            flags["woolly_hair"] = r.random() < 0.72
            flags["keratoderma"] = r.random() < 0.58
            flags["carvajal"] = "Carvajal" in etiol or "biallelic" in etiol.lower()
            flags["lv_involvement"] = r.random() < 0.68
            flags["lv_lge"] = flags["lv_involvement"] and r.random() < 0.72
            flags["rv_involvement"] = r.random() < 0.52
            flags["icd_implanted"] = r.random() < 0.50
            flags["exercise_restricted"] = r.random() < 0.78
            flags["cardiac_mri_done"] = r.random() < 0.78
            flags["myocarditis_episode"] = r.random() < 0.28
            flags["cascade_tested"] = r.random() < 0.55
            flags["task_force_positive"] = r.random() < 0.62

        elif gene == "PKP2":
            flags["epsilon_wave"] = r.random() < 0.45
            flags["rv_dysfunction"] = r.random() < 0.62
            flags["lbbb_vt"] = r.random() < 0.48
            flags["icd_implanted"] = r.random() < 0.55
            flags["exercise_restricted"] = r.random() < 0.82
            flags["mlpa_done"] = r.random() < 0.48
            flags["task_force_positive"] = r.random() < 0.70
            flags["cardiac_mri_done"] = r.random() < 0.75
            flags["rv_fibrofatty_lge"] = flags["cardiac_mri_done"] and r.random() < 0.58
            flags["cascade_tested"] = r.random() < 0.58
            flags["myocarditis_episode"] = r.random() < 0.22
            flags["flecainide_received"] = r.random() < 0.05  # should NOT happen in ARVC

        elif gene == "PLN":
            flags["arg14del"] = "Arg14del" in etiol
            flags["homozygous"] = "homozygous" in etiol.lower()
            flags["icd_implanted"] = r.random() < 0.65
            flags["sacubitril_valsartan"] = r.random() < 0.68
            flags["beta_blocker"] = r.random() < 0.85
            flags["mra"] = r.random() < 0.72
            flags["sglt2i"] = r.random() < 0.55
            flags["transplant_listed"] = r.random() < 0.22
            flags["transplanted"] = flags["transplant_listed"] and r.random() < 0.45
            flags["lvad_bridge"] = flags["transplant_listed"] and r.random() < 0.30
            flags["nsvt"] = r.random() < 0.52
            flags["cascade_tested"] = r.random() < 0.62

        pts.append({
            "pid": f"{gene}-{i+1:03d}",
            "gene": gene,
            "etiology": etiol,
            "sex": sex,
            "age_onset_years": onset_y,
            "dx_delay_months": dx_delay_m,
            **flags,
        })
    return pts


def get_overview():
    all_patients = []
    gene_summaries = []

    for gd in CARDIOMYOPATHY_GENES:
        pts = _make_cohort(gd)
        all_patients.extend(pts)

        gene_summaries.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "etiologies": [e[0] for e in gd["etiologies"]],
            "key_alerts": gd["key_alerts"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
        })

    n = len(all_patients)

    def g_pts(gene):
        return [p for p in all_patients if p["gene"] == gene]

    def pct(lst, key, val=True):
        if not lst:
            return 0.0
        return round(100 * sum(1 for p in lst if p.get(key) == val) / len(lst), 1)

    mybpc3 = g_pts("MYBPC3")
    myh7   = g_pts("MYH7")
    tnnt2  = g_pts("TNNT2")
    lmna   = g_pts("LMNA")
    scn5a  = g_pts("SCN5A")
    dsp    = g_pts("DSP")
    pkp2   = g_pts("PKP2")
    pln    = g_pts("PLN")

    mean_delay = round(sum(p["dx_delay_months"] for p in all_patients) / n, 1)
    icd_pct = round(100 * sum(1 for p in all_patients if p.get("icd_implanted")) / n, 1)
    exercise_restricted_pct = round(100 * sum(1 for p in all_patients if p.get("exercise_restricted")) / n, 1)
    cascade_pct = round(100 * sum(1 for p in all_patients if p.get("cascade_tested")) / n, 1)
    cardiac_mri_pct = round(100 * sum(1 for p in all_patients if p.get("cardiac_mri_done")) / n, 1)
    lge_pct = round(100 * sum(1 for p in all_patients if p.get("lge_extensive") or p.get("lge_midwall") or p.get("lv_lge") or p.get("rv_fibrofatty_lge")) / n, 1)

    all_alerts = []
    for gd in CARDIOMYOPATHY_GENES:
        all_alerts.extend(gd["key_alerts"])

    agg = {
        "total_patients": n,
        "total_genes": 8,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_dx_delay_months": mean_delay,
        "icd_implanted_pct": icd_pct,
        "exercise_restricted_pct": exercise_restricted_pct,
        "cascade_tested_pct": cascade_pct,
        "cardiac_mri_done_pct": cardiac_mri_pct,
        "lge_detected_pct": lge_pct,
        # MYBPC3
        "mybpc3_obstructive_pct": pct(mybpc3, "obstructive_hcm"),
        "mybpc3_mavacamten_pct": pct(mybpc3, "mavacamten_prescribed"),
        "mybpc3_icd_pct": pct(mybpc3, "icd_implanted"),
        "mybpc3_beta_blocker_pct": pct(mybpc3, "beta_blocker"),
        "mybpc3_septal_reduction_pct": pct(mybpc3, "septal_reduction"),
        # MYH7
        "myh7_arg403gln_pct": pct(myh7, "arg403gln"),
        "myh7_obstructive_pct": pct(myh7, "obstructive_hcm"),
        "myh7_mavacamten_pct": pct(myh7, "mavacamten_prescribed"),
        "myh7_icd_pct": pct(myh7, "icd_implanted"),
        "myh7_lge_extensive_pct": pct(myh7, "lge_extensive"),
        # TNNT2
        "tnnt2_mild_lvh_pct": pct(tnnt2, "mild_lvh"),
        "tnnt2_high_scd_pct": pct(tnnt2, "high_scd_risk"),
        "tnnt2_lge_extensive_pct": pct(tnnt2, "lge_extensive"),
        "tnnt2_icd_pct": pct(tnnt2, "icd_implanted"),
        "tnnt2_dcm_sacubitril_pct": pct(tnnt2, "sacubitril_valsartan"),
        # LMNA
        "lmna_ccd_precedes_dcm_pct": pct(lmna, "ccd_precedes_dcm"),
        "lmna_av_block_pct": pct(lmna, "av_block"),
        "lmna_pacemaker_pct": pct(lmna, "pacemaker_implanted"),
        "lmna_icd_pct": pct(lmna, "icd_implanted"),
        "lmna_padua_score_4plus_pct": pct(lmna, "padua_score_4plus"),
        "lmna_af_pct": pct(lmna, "af"),
        "lmna_lge_midwall_pct": pct(lmna, "lge_midwall"),
        "lmna_non_missense_pct": pct(lmna, "non_missense"),
        # SCN5A
        "scn5a_brugada_pct": pct(scn5a, "brugada"),
        "scn5a_lqts3_pct": pct(scn5a, "lqts3"),
        "scn5a_flecainide_received_pct": pct(scn5a, "flecainide_received"),
        "scn5a_fever_triggered_pct": pct(scn5a, "fever_triggered"),
        "scn5a_quinidine_pct": pct(scn5a, "quinidine_prescribed"),
        "scn5a_mexiletine_pct": pct(scn5a, "mexiletine_prescribed"),
        # DSP
        "dsp_woolly_hair_pct": pct(dsp, "woolly_hair"),
        "dsp_keratoderma_pct": pct(dsp, "keratoderma"),
        "dsp_lv_involvement_pct": pct(dsp, "lv_involvement"),
        "dsp_lv_lge_pct": pct(dsp, "lv_lge"),
        "dsp_carvajal_pct": pct(dsp, "carvajal"),
        "dsp_exercise_restricted_pct": pct(dsp, "exercise_restricted"),
        # PKP2
        "pkp2_epsilon_wave_pct": pct(pkp2, "epsilon_wave"),
        "pkp2_rv_dysfunction_pct": pct(pkp2, "rv_dysfunction"),
        "pkp2_lbbb_vt_pct": pct(pkp2, "lbbb_vt"),
        "pkp2_task_force_positive_pct": pct(pkp2, "task_force_positive"),
        "pkp2_exercise_restricted_pct": pct(pkp2, "exercise_restricted"),
        "pkp2_mlpa_done_pct": pct(pkp2, "mlpa_done"),
        # PLN
        "pln_arg14del_pct": pct(pln, "arg14del"),
        "pln_icd_pct": pct(pln, "icd_implanted"),
        "pln_transplant_listed_pct": pct(pln, "transplant_listed"),
        "pln_transplanted_pct": pct(pln, "transplanted"),
        "pln_sacubitril_pct": pct(pln, "sacubitril_valsartan"),
        "pln_nsvt_pct": pct(pln, "nsvt"),
    }

    return {
        "title": "Hereditary-Cardiomyopathy-Atlas — Complete 8-Gene Hereditary Cardiomyopathy Reference",
        "subtitle": (
            "MYBPC3 · MYH7 · TNNT2 · LMNA · SCN5A · DSP · PKP2 · PLN — "
            "320 patients (8×40, seeds 1518–1525) — HCM Mavacamten, "
            "LMNA ICD Padua Score, Brugada Flecainide ABSOLUTE CI, "
            "DSP Woolly Hair Pathognomonic, ARVC Exercise Restriction, PLN Transplant"
        ),
        "genes": gene_summaries,
        "aggregate_stats": agg,
        "top_alerts": all_alerts[:12],
    }


def get_breakdown():
    breakdown = []
    for gd in CARDIOMYOPATHY_GENES:
        pts = _make_cohort(gd)
        sex_dist = {"M": sum(1 for p in pts if p["sex"] == "M"),
                    "F": sum(1 for p in pts if p["sex"] == "F")}
        mean_onset = round(sum(p["age_onset_years"] for p in pts) / len(pts), 1)
        mean_delay = round(sum(p["dx_delay_months"] for p in pts) / len(pts), 1)
        etiol_counts = {}
        for p in pts:
            etiol_counts[p["etiology"]] = etiol_counts.get(p["etiology"], 0) + 1

        breakdown.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "mean_onset_years": mean_onset,
            "mean_dx_delay_months": mean_delay,
            "sex_distribution": sex_dist,
            "etiology_counts": etiol_counts,
            "key_alerts": gd["key_alerts"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
            "patients": pts,
        })
    return {"breakdown": breakdown}


def get_definitions():
    return {
        "atlas": "Hereditary-Cardiomyopathy-Atlas — Complete 8-Gene Hereditary Cardiomyopathy Reference",
        "genes": [gd["gene"] for gd in CARDIOMYOPATHY_GENES],
        "clinical_definitions": [
            {
                "term": "MYBPC3 Haploinsufficiency HCM — Mavacamten and Incomplete Penetrance",
                "definition": (
                    "MYBPC3 is the most common HCM gene (35-40%), predominantly causing disease "
                    "via haploinsufficiency — truncating variants reduce cMyBP-C to ~50% of "
                    "normal stoichiometry in the sarcomere, releasing the cross-bridge 'brake' "
                    "and causing hypercontractility. Penetrance is incomplete and age-related: "
                    "many carriers first develop hypertrophy in the 5th-6th decade. Mavacamten "
                    "(Camzyos, FDA April 2022) is a first-in-class cardiac myosin inhibitor "
                    "for symptomatic obstructive HCM (NYHA II-III, LVOTO >30 mmHg): it "
                    "allosterically stabilises the super-relaxed myosin state, reducing "
                    "cross-bridge number and LVOTO. Echocardiographic EF monitoring is mandatory "
                    "during mavacamten therapy — dose reduction or discontinuation required "
                    "if EF <50%. Annual cascade screening of gene-positive phenotype-negative "
                    "family members is mandatory (50% per-generation risk)."
                ),
            },
            {
                "term": "MYH7 Arg403Gln Malignant HCM — Dominant Negative Poison Polypeptide",
                "definition": (
                    "MYH7 (beta-myosin heavy chain) causes HCM via dominant-negative "
                    "'poison polypeptide' mechanism — the mutant myosin is incorporated into "
                    "thick filaments and disrupts cooperative cross-bridge kinetics. Arg403Gln "
                    "(the first identified HCM mutation, Geisterfer-Lowrance 1990) is associated "
                    "with severe HCM, early onset, high SCD risk, and malignant phenotype. "
                    "MYH7-HCM tends to have greater wall thickness, earlier presentation, and "
                    "more extensive LGE than MYBPC3. Surgical septal myectomy is preferred over "
                    "alcohol ablation for drug-refractory LVOTO in young MYH7 patients (better "
                    "long-term outcomes, avoids conduction block complication of ablation). "
                    "Digoxin is contraindicated in obstructive HCM."
                ),
            },
            {
                "term": "TNNT2 Arg92Trp — High SCD Risk Despite Mild or Absent LVH",
                "definition": (
                    "TNNT2-HCM (cardiac troponin T) presents a critical diagnostic trap: "
                    "Arg92Trp and related TNNT2 HCM mutations cause disproportionately HIGH "
                    "sudden cardiac death risk relative to MILD or ABSENT left ventricular "
                    "hypertrophy. The standard HCM SCD risk tool (wall thickness criterion) "
                    "significantly underestimates risk in TNNT2-HCM. Cardiac MRI with "
                    "late gadolinium enhancement (LGE) is mandatory — extensive fibrosis "
                    "is frequently present despite near-normal wall thickness. LGE >15% "
                    "LV mass is an independent ICD indication. TNNT2 also causes DCM "
                    "(opposite mechanism — hypocontractility) requiring standard HFrEF "
                    "therapy including sacubitril/valsartan."
                ),
            },
            {
                "term": "LMNA-DCM Padua Risk Score — ICD Before Conventional LVEF Threshold",
                "definition": (
                    "LMNA-DCM (lamin A/C) is the most common genetic cause of DCM with cardiac "
                    "conduction disease (CCD) and carries high risk of malignant ventricular "
                    "arrhythmia at LVEF above the conventional ICD threshold (35%). The "
                    "2019 Padua Risk Score quantifies arrhythmic risk: (1) male sex, "
                    "(2) non-missense variant (truncating/splice-site), (3) LVEF <45%, "
                    "(4) non-sustained VT on Holter, (5) LGE on cardiac MRI. Score ≥4 = "
                    "ICD indication REGARDLESS of LVEF. CCD (AV block, bundle branch block) "
                    "frequently PRECEDES DCM by years — annual ECG mandatory from time of "
                    "diagnosis. Atrial fibrillation requires anticoagulation; rhythm control "
                    "preferred due to AF-to-VT risk in LMNA."
                ),
            },
            {
                "term": "SCN5A Brugada Syndrome — Flecainide ABSOLUTE Contraindication",
                "definition": (
                    "SCN5A loss-of-function variants cause Brugada syndrome (BrS) — a channelopathy "
                    "with characteristic coved-type ST elevation in V1-V3 and risk of ventricular "
                    "fibrillation, particularly during fever, sleep, or with sodium channel blockers. "
                    "Flecainide, propafenone, ajmaline, and all Class IC antiarrhythmics are "
                    "ABSOLUTELY CONTRAINDICATED in Brugada — they worsen INa reduction, precipitate "
                    "phase 2 re-entry, and can cause VF. Ajmaline/flecainide challenge IS "
                    "the diagnostic provocation test but must be performed in a monitored setting "
                    "with immediate defibrillation capability. Fever is the most common trigger — "
                    "antipyretics mandatory immediately. Quinidine (Ito blocker) reduces Brugada "
                    "arrhythmias. The Brugadrugs.org database must be checked before prescribing "
                    "any medication to a Brugada patient."
                ),
            },
            {
                "term": "DSP-ARVC Left-Dominant Phenotype — Woolly Hair + Keratoderma PATHOGNOMONIC",
                "definition": (
                    "Desmoplakin (DSP) ARVC is distinct from classical right-ventricular ARVC "
                    "(PKP2): DSP mutations cause LEFT-dominant or biventricular cardiomyopathy "
                    "with fibrofatty replacement predominantly in the LV — LV LGE on CMR is "
                    "the characteristic finding. The diagnosis can be missed if only RV criteria "
                    "are applied. CRITICAL CLINICAL CLUE: Woolly hair (tightly coiled, changed "
                    "hair texture) and palmoplantar keratoderma (thickened palms/soles) are "
                    "PATHOGNOMONIC for heterozygous DSP. Any young patient with DCM/ARVC + "
                    "woolly hair requires DSP gene testing. Biallelic DSP = Carvajal syndrome "
                    "(DCM + woolly hair + keratoderma — this is NOT classical ARVC). Exercise "
                    "restriction is mandatory and is the strongest modifiable disease modifier."
                ),
            },
            {
                "term": "PKP2-ARVC Task Force Criteria — Epsilon Wave and Exercise Prohibition",
                "definition": (
                    "PKP2 (plakophilin 2) is the most common ARVC gene (~40%). PKP2-ARVC "
                    "primarily affects the RIGHT ventricle with fibrofatty replacement, "
                    "wall motion abnormalities, and re-entrant VT with LBBB morphology "
                    "(inferior axis = free wall origin; superior axis = inferior wall). "
                    "The 2010 Task Force Criteria define ARVC by: (1) RV structural/functional "
                    "criteria; (2) tissue characterisation (fatty infiltration on CMR); "
                    "(3) repolarisation abnormalities (T-wave inversion V1-V3 in adults); "
                    "(4) depolarisation abnormalities (epsilon wave = major criterion — "
                    "terminal deflection after QRS on signal-averaged ECG); (5) arrhythmia "
                    "(LBBB-morphology VT/PVCs); (6) family history/genetics. Exercise "
                    "is the STRONGEST arrhythmogenic and disease-progression trigger — "
                    "competitive sport ABSOLUTELY prohibited in PKP2 carriers regardless "
                    "of phenotype status."
                ),
            },
            {
                "term": "PLN Arg14del — Dutch Founder DCM, SERCA2a Dysregulation, Cardiac Transplantation",
                "definition": (
                    "Phospholamban (PLN) p.Arg14del is a Dutch/Belgian founder mutation "
                    "(carrier frequency ~1/400-800 in the Netherlands) that prevents PKA "
                    "phosphorylation of PLN Ser16, locking phospholamban in its SERCA2a- "
                    "inhibitory state even under adrenergic stimulation. SERCA2a is "
                    "chronically suppressed → reduced SR calcium reuptake → reduced SR "
                    "calcium stores → impaired systolic Ca2+ transient → DCM progression. "
                    "Homozygous PLN Arg14del → lethal neonatal DCM requiring emergency "
                    "cardiac transplantation. Heterozygous PLN-DCM carries high malignant "
                    "VT/VF risk — ICD mandatory when LVEF <35% or NSVT present. Maximally "
                    "tolerated GDMT (sacubitril/valsartan, carvedilol, eplerenone, SGLT2i) "
                    "is mandatory. Cardiac transplantation is frequently required as disease "
                    "progresses; LVAD as bridge to transplant when appropriate."
                ),
            },
        ],
    }
