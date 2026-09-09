#!/usr/bin/env python3
"""Hereditary-Non-Syndromic-Hearing-Loss-Atlas — Complete 8-Gene Atlas
(GJB2 · SLC26A4 · OTOF · MYO15A · TECTA · KCNQ4 · LHFPL5 · GJB6).

GJB2     (Connexin 26 / Cx26; 226 aa; ~26 kDa; 13q12.11; AR (biallelic) / AD (rare);
           DFNB1A — MOST COMMON AR NSHL GLOBALLY ~50% of AR NSHL;
           p.35delG European/Mediterranean founder ~70% of GJB2 alleles in Europeans;
           p.167delT Ashkenazi Jewish founder; p.235delC East Asian founder;
           Prelingual SNHL ranging moderate to profound; seed SEED_BASE+0).
SLC26A4  (Pendrin; 780 aa; ~90 kDa; 7q22.3; AR;
           DFNB4 (non-syndromic EVA) / Pendred syndrome (EVA + goiter);
           ENLARGED VESTIBULAR AQUEDUCT (EVA): hallmark on CT/MRI — PATHOGNOMONIC;
           p.H723R East Asian founder; biallelic required for full Pendred syndrome;
           seed SEED_BASE+1).
OTOF     (Otoferlin; 1997 aa; ~227 kDa; 2p23.3; AR;
           DFNB9 — AUDITORY NEUROPATHY SPECTRUM DISORDER (ANSD);
           PATHOGNOMONIC: present OAE + absent/severely abnormal ABR;
           p.Ile515Thr Spanish/Iberian founder; cochlear implant EXCELLENT outcome;
           seed SEED_BASE+2).
MYO15A   (Myosin XVa; 3530 aa; ~395 kDa; 17p11.2; AR;
           DFNB3 — profound prelingual SNHL;
           SHORT OUTER HAIR CELL STEREOCILIA (electron microscopy);
           p.Arg2169Gln South Asian/Iranian founder; DFNB3 original Pakistan/Bali pedigrees;
           seed SEED_BASE+3).
TECTA    (Alpha-tectorin; 2155 aa; ~222 kDa; 11q23.3; AR and AD;
           DFNB21 (AR — prelingual profound) and DFNA8/12 (AD — MID-FREQUENCY U-SHAPED dip);
           TECTORIAL MEMBRANE component; U-shaped mid-frequency audiogram pathognomonic for DFNA8;
           seed SEED_BASE+4).
KCNQ4   (Kv7.4 / KCNQ4; 695 aa; ~77 kDa; 1p34.2; AD;
           DFNA2A — progressive high-frequency SNHL; dominant-negative haploinsufficiency;
           p.Gly285Ser Korean founder; progressive SNHL typically onset 2nd–3rd decade;
           KCNQ channel family — potential pharmacological target;
           seed SEED_BASE+5).
LHFPL5  (TMHS / Lipoma HMGIC Fusion Partner-Like 5; 219 aa; ~25 kDa; 6p21.31; AR;
           DFNB67 — profound congenital SNHL;
           Tetraspan membrane hair-cell protein; part of TMC1/2 mechanotransduction complex;
           Turkish/Pakistani founder; p.Arg83His Turkish consanguineous;
           seed SEED_BASE+6).
GJB6    (Connexin 30 / Cx30; 261 aa; ~30 kDa; 13q12.11; AR / digenic with GJB2;
           DFNB1B — del(GJB6-D13S1830) 342 kb deletion most common GJB6 allele;
           DIGENIC: one GJB2 pathogenic allele + del(GJB6-D13S1830) → DFNB1 hearing loss;
           MLPA/CNV-seq MANDATORY — exome/NGS panel misses large deletions;
           seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2438-2445).
"""

import random

SEED_BASE = 2438

NSHL_GENES = [
    # -- GJB2 -- DFNB1A (Most Common AR NSHL ~50% of AR NSHL Globally) ----------------------------
    {
        "gene": "GJB2",
        "alt_name": (
            "GJB2 (GJB2-226aa-13q12.11 / AR-biallelic -- "
            "DFNB1A-MOST-COMMON-AR-NSHL-GLOBALLY-~50pct-ALL-AR-NSHL -- "
            "p.35delG-EUROPEAN-FOUNDER-~70pct-GJB2-ALLELES-EUROPEANS -- "
            "p.167delT-ASHKENAZI-FOUNDER-p.235delC-EAST-ASIAN-FOUNDER -- "
            "PRELINGUAL-SNHL-MODERATE-TO-PROFOUND-FLAT-OR-SLOPING-AUDIOGRAM -- "
            "COCHLEAR-IMPLANT-EXCELLENT-OUTCOMES-GJB2-BILATERAL-CI-RECOMMENDED)"
        ),
        "protein": (
            "GJB2 -- 13q12.11 AR (biallelic) -- GJB2-226aa -- "
            "Connexin-26-Cx26-26kDa-Gap-Junction-Beta-2-4-TM-Helices -- "
            "Cochlear-Supporting-Cell-Gap-Junction-Potassium-Recycling -- "
            "OMIM-Gene-121011-Disease-DFNB1-220290-DFNA3-601544"
        ),
        "locus": "13q12.11",
        "protein_size": "226 aa / 26 kDa",
        "inheritance": (
            "AR (biallelic for DFNB1A) / AD (rare dominant variants for DFNA3); "
            "GJB2 encodes Connexin 26 (Cx26), a gap junction protein forming hemichannels and channels between cochlear supporting cells. "
            "FUNCTION: Cx26 channels recycle K+ from hair cells back to endolymph via supporting cell network; "
            "GJB2 LOF → K+ recycling failure → endocochlear potential collapse → mechanotransduction failure → SNHL. "
            "MOST COMMON AR NSHL GENE: ~50% of AR NSHL globally; most common in Europeans (p.35delG), Ashkenazi (p.167delT), East Asians (p.235delC); "
            "FOUNDER MUTATIONS: p.35delG (c.35delG) — European/Mediterranean, ~70% GJB2 alleles in Europeans; "
            "p.167delT — Ashkenazi Jewish; p.235delC — East Asian; p.W77X — South Indian; "
            "PHENOTYPE: prelingual SNHL, moderate to profound; audiogram varies by genotype — null/null → profound; "
            "CI OUTCOMES: cochlear implant outcomes EXCELLENT for GJB2 — spiralling cisternae intact, SGN preserved; "
            "GJB2 is located adjacent to GJB6 (Cx30) on 13q12.11 — compound heterozygosity common"
        ),
        "disease_category": (
            "DFNB1A (AR) — most common AR non-syndromic hearing loss globally; "
            "prelingual SNHL ranging moderate to profound; no vestibular or retinal involvement; "
            "cochlear implantation highly effective"
        ),
        "disease_pathway": (
            "GJB2/CX26 IN K+ RECYCLING: gap junction channels formed by Cx26 (and Cx30) connect cochlear supporting cells "
            "(Deiters cells → inner sulcus cells → spiral limbus → fibrocytes). "
            "K+ flow pathway: hair cells → perilymph → fibrocyte network → lateral wall → stria vascularis → endolymph. "
            "GJB2 LOF → gap junction dysfunction → K+ accumulates near hair cells → endocochlear potential collapses → "
            "mechanotransduction channels (TMC1/TMC2) cannot function → SNHL. "
            "HISTOPATHOLOGY: spiral ganglion neurons (SGN) relatively preserved; organ of Corti degeneration secondary. "
            "CI MECHANISM: cochlear implant bypasses hair cells entirely; stimulates SGNs directly; "
            "preserved SGN explains EXCELLENT CI outcomes in GJB2 SNHL. "
            "GENOTYPE-PHENOTYPE: null/null (e.g., p.35delG homozygous) → profound SNHL; "
            "null/hypomorphic (e.g., p.35delG/p.V37I) → mild-moderate SNHL; "
            "p.V37I homozygous → mild SNHL, common East Asian 'mild' variant. "
            "GENETIC EPIDEMIOLOGY: carrier frequency p.35delG 1/33 in Europeans; 1/1000 births affected."
        ),
        "pathognomonic": (
            "GJB2-DFNB1A CLINICAL FEATURES: "
            "1. PRELINGUAL SNHL: congenital or very early onset; typically identified on newborn hearing screen; "
            "audiogram: flat or gently sloping moderate-to-profound loss; "
            "SNHL is bilateral, symmetric; rarely asymmetric; "
            "2. NO RETINAL INVOLVEMENT: ophthalmology normal — key DDx from Usher syndrome; "
            "3. NO VESTIBULAR INVOLVEMENT: vestibular testing normal — GJB2 pure cochlear; "
            "4. COCHLEAR ANATOMY: CT/MRI cochlea normal — no EVA, no cochlear malformation (DDx SLC26A4); "
            "5. FAMILY HISTORY: AR pattern — carrier parents, 25% risk per sibling; "
            "GJB2 IS PURE COCHLEAR NON-SYNDROMIC: no skin, vision, thyroid, cardiac, renal, or neurological features. "
            "CI OUTCOMES: among best CI outcomes of any genetic SNHL — SGN preservation; "
            "implant both ears in profound loss."
        ),
        "treatment": (
            "HEARING AIDS: all degrees; digital programmable aids; early fitting critical for speech development; "
            "COCHLEAR IMPLANT: FIRST-LINE for profound/severe-profound SNHL; "
            "GJB2 outcomes EXCELLENT — among the best genetic SNHL CI results; "
            "implant early (before 12 months in profound congenital loss) for optimal speech-language; "
            "bilateral CI recommended; "
            "SPEECH-LANGUAGE THERAPY: mandatory; auditory-verbal or total communication approach; "
            "GENETIC COUNSELLING: AR 25% risk per sibling; carrier testing both parents; "
            "test siblings of affected children even if clinically normal; "
            "EXPANDED CARRIER SCREENING: GJB2/GJB6 carrier testing available preconception; "
            "NEWBORN HEARING SCREEN: GJB2 is leading cause of failed newborn screen — "
            "early identification enables early amplification/CI referral."
        ),
        "key_features": [
            "Most common AR NSHL globally (~50% of AR NSHL); p.35delG European founder",
            "Prelingual SNHL moderate to profound; flat/sloping bilateral audiogram",
            "No retinal involvement (DDx Usher), no vestibular (DDx SLC26A4), normal CT (DDx SLC26A4)",
            "Cochlear implant EXCELLENT outcomes — spiral ganglion neurons preserved",
            "Carrier frequency 1/33 Europeans; 1/1000 births affected",
            "p.167delT Ashkenazi; p.235delC East Asian; p.W77X South Indian founders",
            "Located 13q12.11 adjacent to GJB6 — compound het GJB2/GJB6 common",
            "Genotype-phenotype: null/null → profound; null/hypomorphic → mild-moderate",
        ],
        "key_ddx": (
            "SLC26A4 (DFNB4/Pendred): EVA on CT/MRI — key DDx; thyroid goiter; GJB2 has NORMAL CT; "
            "OTOF (DFNB9): ANSD pattern — present OAE + absent ABR; GJB2 has typical SNHL pattern; "
            "Usher syndrome (MYO7A, USH2A, etc.): RP + SNHL; GJB2 has NO retinal involvement; "
            "Connexin-related: GJB6 large deletion — digenic GJB2/GJB6; test GJB6 copy number (MLPA) with GJB2; "
            "Environmental SNHL (CMV, aminoglycosides, meningitis): history, unilateral, asymmetric; "
            "A1555G mitochondrial: aminoglycoside-precipitated SNHL; maternal inheritance."
        ),
        "systemic_involvement": (
            "NO systemic involvement in GJB2-related DFNB1A. Pure non-syndromic cochlear SNHL. "
            "No skin (DDx KID syndrome — GJB2 dominant gain-of-function variants, different disease). "
            "No thyroid (DDx Pendred/SLC26A4). No retinal (DDx Usher). No vestibular (DDx SLC26A4). "
            "Note: dominant GJB2 gain-of-function mutations → palmoplantar keratoderma + SNHL (KID/HID syndrome) — "
            "completely different from biallelic LOF DFNB1A."
        ),
        "onset_age": "Congenital or early postnatal prelingual SNHL; identified newborn hearing screen",
        "surgical_urgency": "CI urgency: implant before 12 months in profound congenital SNHL for best language outcomes",
        "gene_family": "Connexin / gap junction protein family; Cx26 subunit; forms hexameric hemichannels",
        "morphology": (
            "AUDIOGRAM: flat or gently sloping moderate-to-profound bilateral SNHL; "
            "ABR: present but threshold-elevated (type 1 SNHL pattern — DDx OTOF-ANSD); "
            "OAE: absent in moderate-profound loss (DDx OTOF where OAE present); "
            "CT/MRI: NORMAL cochlear anatomy — no EVA (DDx SLC26A4), no Mondini"
        ),
        "n_patients": 40,
    },

    # -- SLC26A4 -- DFNB4 / Pendred Syndrome (EVA — Enlarged Vestibular Aqueduct) -------------------
    {
        "gene": "SLC26A4",
        "alt_name": (
            "SLC26A4 (SLC26A4-780aa-7q22.3 / AR -- "
            "DFNB4-NON-SYNDROMIC-SNHL-OR-PENDRED-SYNDROME-EVA+GOITER -- "
            "ENLARGED-VESTIBULAR-AQUEDUCT-EVA-CT-MRI-PATHOGNOMONIC-HALLMARK -- "
            "p.H723R-EAST-ASIAN-FOUNDER-~50pct-EAST-ASIAN-SLC26A4-ALLELES -- "
            "THYROID-GOITER-EUTHYROID-50-60pct-PENDRED-BIALLELIC -- "
            "SNHL-FLUCTUATING-PROGRESSIVE-HEAD-TRAUMA-TRIGGERS)"
        ),
        "protein": (
            "SLC26A4 -- 7q22.3 AR -- SLC26A4-780aa -- "
            "Pendrin-SLC26-Family-Anion-Transporter-Cl-/I-/HCO3--90kDa -- "
            "Cochlear-Endolymphatic-Sac-Thyroid-Follicular-Cell -- "
            "OMIM-Gene-605646-Disease-DFNB4-600791-Pendred-274600"
        ),
        "locus": "7q22.3",
        "protein_size": "780 aa / 90 kDa",
        "inheritance": (
            "AR (biallelic for both DFNB4 and Pendred syndrome); "
            "SLC26A4 encodes pendrin, an apical anion transporter (Cl-/I-/HCO3- exchanger) in cochlear endolymphatic sac, "
            "kidney intercalated cells, and thyroid follicular cells. "
            "COCHLEA: pendrin maintains endolymph pH and Cl- homeostasis in endolymphatic sac; "
            "SLC26A4 LOF → endolymphatic hydrops → enlarged endolymphatic sac and duct → EVA; "
            "THYROID: pendrin in thyroid required for iodide efflux into follicle lumen; "
            "SLC26A4 LOF → impaired organification → goitrogenesis; euthyroid or subclinical hypothyroid; "
            "BIALLELIC: both alleles must be pathogenic for Pendred syndrome; "
            "MONOALLELIC EVA: heterozygous SLC26A4 variants found in ~50% of EVA — second hit unclear (EVA spectrum); "
            "FOUNDER: p.H723R (c.2168A>G) — East Asian, ~50% of East Asian SLC26A4 alleles; "
            "p.L236P, p.T416P — European founders; p.IVS8+1G>A — splice site common"
        ),
        "disease_category": (
            "DFNB4 (AR, non-syndromic SNHL + EVA) or Pendred syndrome (AR, EVA + thyroid goiter); "
            "SNHL can be progressive and fluctuating; EVA on CT/MRI is pathognomonic"
        ),
        "disease_pathway": (
            "PENDRIN FUNCTION: SLC26A4 transports Cl-/HCO3-/I- across apical membranes. "
            "COCHLEAR: pendrin expressed in endolymphatic sac (ES) and duct; regulates endolymph Cl- and HCO3-; "
            "SLC26A4 LOF → endolymph ion imbalance → hydrops → morphological abnormality: enlarged endolymphatic sac + duct (EVA on CT); "
            "Endolymphatic hydrops → mechanical disruption of ion gradients → fluctuating-progressive SNHL. "
            "HEAD TRAUMA / STRAINING: Valsalva manoeuvre, head trauma → sudden SNHL episodes in EVA; "
            "perilymph fistula risk → contact sports are CONTRAINDICATED. "
            "THYROID: pendrin transports iodide across apical membrane of thyroid follicular cells; "
            "SLC26A4 LOF → reduced iodide in follicle lumen → impaired thyroid peroxidase-mediated organification → "
            "goitrogenesis; Pendred syndrome thyroid often euthyroid (compensated) or subclinical hypothyroid; "
            "perchlorate discharge test: >10% iodide release = positive (organification defect). "
            "KIDNEY: pendrin in type B and non-A-non-B intercalated cells — bicarbonaturia in some."
        ),
        "pathognomonic": (
            "SLC26A4 / DFNB4 / PENDRED SYNDROME PATHOGNOMONIC FEATURES: "
            "1. EVA (ENLARGED VESTIBULAR AQUEDUCT): CT/MRI: vestibular aqueduct midpoint width >1.5 mm (Cincinnati/Valvassori criteria); "
            "OR endolymphatic duct+sac enlarged on MRI T2; "
            "EVA IS THE SINGLE MOST IMPORTANT DIAGNOSTIC CLUE — any SNHL child with EVA → test SLC26A4; "
            "2. SNHL FLUCTUATING AND PROGRESSIVE: episodes of sudden SNHL after head trauma, Valsalva, illness; "
            "may recover partially; overall progressive; "
            "3. THYROID GOITER: Pendred syndrome = biallelic SLC26A4 + EVA SNHL + euthyroid or subclinical goiter; "
            "goiter may not develop until teens; perchlorate discharge test positive (organification defect); "
            "4. CONTACT SPORTS CONTRAINDICATED: head trauma → sudden SNHL episode; strict avoidance; "
            "5. VESTIBULAR DYSFUNCTION: 60-80% have some vestibular hypofunction (ddx: unilateral vs bilateral); "
            "Mondini dysplasia co-occurs in some patients (cochlea <2.5 turns + incomplete partition)."
        ),
        "treatment": (
            "AVOID CONTACT SPORTS / HEAD TRAUMA: absolute contraindication — sudden SNHL risk; "
            "AVOID BAROTRAUMA: no diving, caution with air travel / Valsalva; "
            "HEARING AIDS: early fitting; aids useful in moderate-severe range; "
            "COCHLEAR IMPLANT: for severe-profound SNHL; EVA does not preclude CI; "
            "CI outcomes generally good; note perilymph gusher risk during CI surgery — "
            "surgeon must be aware: drill slowly, have perilymph sealant ready; "
            "THYROID MONITORING: annual TSH/FT4; supplement if hypothyroid; "
            "iodine-adequate diet; avoid goitrogens (excessive cruciferous vegetables); "
            "GENETIC COUNSELLING: AR 25% risk; cascade carrier testing; "
            "AUDIOMETRY: 6-monthly until stable, then annually; "
            "VESTIBULAR REHABILITATION: if vestibular hypofunction identified."
        ),
        "key_features": [
            "EVA (enlarged vestibular aqueduct) on CT/MRI — PATHOGNOMONIC; width >1.5 mm",
            "SNHL fluctuating and progressive; head trauma can trigger sudden SNHL episodes",
            "Pendred syndrome = biallelic SLC26A4 + EVA + euthyroid goiter",
            "Contact sports CONTRAINDICATED — head trauma risk",
            "p.H723R East Asian founder ~50% of East Asian SLC26A4 alleles",
            "Cochlear implant feasible but perilymph gusher risk intraoperatively",
            "Perchlorate discharge test: >10% positive for organification defect (Pendred)",
            "Vestibular hypofunction 60-80%; Mondini co-occurs in subset",
        ],
        "key_ddx": (
            "GJB2 (DFNB1A): normal CT — no EVA; GJB2 has normal cochlear anatomy on imaging; "
            "Other EVA causes: IP-II (incomplete partition type II = Mondini + EVA) — POU3F4 XLR; "
            "Isolated EVA without SLC26A4 — ~50% of EVA cases have no SLC26A4 variant; consider FOXI1, KCNJ10; "
            "Waardenburg syndrome (PAX3, MITF, EDNRB, SOX10, EDN3): pigmentation changes, dystopia canthorum; "
            "Branchiooto renal (EYA1, SIX1, SIX5): external ear + renal anomalies; "
            "Autoimmune inner ear disease: rapid bilateral SNHL, elevated ESR/CRP, responds to steroids."
        ),
        "systemic_involvement": (
            "THYROID: euthyroid or subclinical goiter in Pendred syndrome; perchlorate test positive; "
            "TSH annual monitoring mandatory. "
            "KIDNEY: pendrin in intercalated cells — hyperchloraemia/bicarbonaturia in some. "
            "VESTIBULAR: hypofunction 60-80%; variable degree. "
            "NO cardiac, NO retinal, NO neurological features."
        ),
        "onset_age": "Congenital or early childhood prelingual SNHL; fluctuating episodes; thyroid goiter teens onward",
        "surgical_urgency": "CI surgeon must anticipate perilymph gusher (EVA); contact sports absolute CI from diagnosis",
        "gene_family": "SLC26 anion transporter family; pendrin subtype; apical Cl-/HCO3-/I- exchanger",
        "morphology": (
            "CT: EVA (vestibular aqueduct midpoint >1.5 mm); "
            "MRI T2: enlarged endolymphatic sac/duct; possible IP-II Mondini (<2.5 cochlear turns); "
            "AUDIOGRAM: bilateral SNHL, moderate to profound, fluctuating; "
            "VESTIBULAR: caloric hyporesponse in majority"
        ),
        "n_patients": 40,
    },

    # -- OTOF -- DFNB9 (Auditory Neuropathy Spectrum Disorder — ANSD) --------------------------------
    {
        "gene": "OTOF",
        "alt_name": (
            "OTOF (OTOF-1997aa-2p23.3 / AR -- "
            "DFNB9-AUDITORY-NEUROPATHY-SPECTRUM-DISORDER-ANSD -- "
            "PATHOGNOMONIC-PRESENT-OAE-ABSENT-OR-ABNORMAL-ABR -- "
            "p.Ile515Thr-IBERIAN-SPANISH-FOUNDER-pQ829X-WIDESPREAD -- "
            "COCHLEAR-IMPLANT-EXCELLENT-OUTCOME-RESTORES-SYNCHRONY -- "
            "HEARING-AIDS-OFTEN-INEFFECTIVE-CI-PREFERRED)"
        ),
        "protein": (
            "OTOF -- 2p23.3 AR -- OTOF-1997aa -- "
            "Otoferlin-Multiple-C2-Domains-227kDa-Inner-Hair-Cell-Synaptic-Vesicle-Fusion -- "
            "Ca2+-Sensor-for-IHC-Exocytosis-Ribbon-Synapse -- "
            "OMIM-Gene-603681-Disease-DFNB9-601071"
        ),
        "locus": "2p23.3",
        "protein_size": "1997 aa / 227 kDa",
        "inheritance": (
            "AR (biallelic); "
            "OTOF encodes otoferlin, a multi-C2-domain calcium-sensing protein expressed in inner hair cells (IHC). "
            "FUNCTION: otoferlin is the MAIN CALCIUM SENSOR for synaptic vesicle exocytosis at the IHC ribbon synapse; "
            "acts as an SNAREs cofactor (analogous to synaptotagmin at conventional synapses); "
            "OTOF LOF → IHC cannot release glutamate onto type I SGNs → auditory nerve does not fire → "
            "AUDITORY NEUROPATHY: OHC function preserved (OAE present), but IHC→SGN synapse fails (ABR absent). "
            "ANSD AUDIOGRAM PATTERN: present distortion-product or transient OAE; absent/severely abnormal ABR/ASSR; "
            "CI RATIONALE: cochlear implant stimulates SGNs electrically, bypasses IHC synapse → restores synchrony → "
            "EXCELLENT CI outcomes in OTOF-ANSD (SGNs intact, ribbon synapse is the failure point). "
            "FOUNDERS: p.Ile515Thr (c.1544T>C) — Spanish/Iberian ~70% of Spanish DFNB9 alleles; "
            "p.Gln829Stop (c.2485C>T) — more widespread; "
            "TEMPERATURE-SENSITIVE variant p.Ile515Thr: some patients' ANSD worsens with fever (temperature-sensitive ANSD)"
        ),
        "disease_category": (
            "DFNB9 (AR) — auditory neuropathy spectrum disorder (ANSD) due to IHC synaptic vesicle fusion failure; "
            "present OAE, absent ABR; cochlear implant EXCELLENT; hearing aids limited benefit"
        ),
        "disease_pathway": (
            "IHC RIBBON SYNAPSE: inner hair cells form specialised ribbon synapses with type I spiral ganglion neurons (SGN). "
            "Ribbon synapses can sustain very high rates of vesicle release (1000/s) required for temporal coding of sound. "
            "Ca2+ influx (via CaV1.3 channels) → otoferlin senses Ca2+ rise → promotes SNARE complex assembly → "
            "synaptic vesicle fusion → glutamate release → AMPA receptors on SGN dendrites → action potential in auditory nerve. "
            "OTOF LOF: Ca2+ influx normal; OHC electromotility normal (OAE present); "
            "IHC vesicle fusion FAILS → no glutamate release → auditory nerve silent → ABR absent. "
            "CLINICAL: ANSD = present OAE but no synchronised auditory response; "
            "speech perception disproportionately poor vs pure-tone thresholds; "
            "temporal modulation perception severely impaired (temporal fine structure coding fails). "
            "CI OUTCOME: electrical stimulation bypasses IHC→SGN synapse entirely → directly activates SGNs → "
            "CI RESTORES SYNCHRONY → OTOF-ANSD patients achieve EXCELLENT CI outcomes, "
            "better than many non-ANSD SNHL patients."
        ),
        "pathognomonic": (
            "OTOF-DFNB9 / ANSD PATHOGNOMONIC PATTERN: "
            "1. OAE PRESENT: distortion-product (DPOAE) and/or transient (TEOAE) otoacoustic emissions PRESENT; "
            "OHC function intact — cochlear amplifier working; "
            "2. ABR ABSENT OR SEVERELY ABNORMAL: auditory brainstem response: no clear wave I, III, V; "
            "or grossly desynchronised ABR (elevated thresholds >90 dBnHL); "
            "3. ANSD DIAGNOSIS: present OAE + absent/abnormal ABR = ANSD (per Starr 1996 definition); "
            "OTOF is the most common GENETIC cause of ANSD; "
            "4. SPEECH PERCEPTION: severely impaired, disproportionate to pure-tone thresholds; "
            "patient appears to hear but cannot understand speech; "
            "5. HEARING AIDS: often INEFFECTIVE in ANSD — amplifying a desynchronised signal; "
            "HEARING AIDS may even worsen OHC (acoustic OAE may disappear); "
            "6. CI CHOICE: CI is FIRST-LINE for OTOF-ANSD — outcomes excellent; "
            "TEMPERATURE-SENSITIVE SUBTYPE: some p.Ile515Thr patients — SNHL worsens with fever, improves when afebrile."
        ),
        "treatment": (
            "COCHLEAR IMPLANT: FIRST-LINE treatment; outcomes EXCELLENT — among best CI outcomes for any genetic SNHL; "
            "implant early; bilateral CI recommended; "
            "avoid FM systems as sole amplification — synchrony bypass needed; "
            "HEARING AIDS: trial briefly; if no benefit (or OAE disappear) → proceed to CI; "
            "ANSD DIAGNOSIS PROTOCOL: must distinguish OTOF-genetic ANSD from auditory neuropathy due to "
            "hyperbilirubinemia, prematurity, CMV, AIFM1 (X-linked) — genetic testing mandatory; "
            "GENETIC COUNSELLING: AR 25% risk; p.Ile515Thr targeted test first in Iberian ancestry; "
            "AUDIOMETRY: behavioural testing with soundfield (ABR unreliable for threshold in ANSD); "
            "ASSR at multiple frequencies; OAE monitoring for OHC status; "
            "AVOID FEVER-TRIGGERING SITUATIONS: in temperature-sensitive subtype, fever → acute SNHL; "
            "antipyretics promptly."
        ),
        "key_features": [
            "ANSD: present OAE + absent/abnormal ABR — PATHOGNOMONIC of OTOF-DFNB9",
            "Speech perception severely impaired; hearing aids often ineffective",
            "Cochlear implant EXCELLENT outcomes — restores SGN synchrony",
            "p.Ile515Thr Iberian/Spanish founder ~70% Spanish DFNB9 alleles; temperature-sensitive subtype",
            "Otoferlin: Ca2+ sensor for IHC synaptic vesicle exocytosis at ribbon synapse",
            "OHC intact → OAE present despite profound hearing impairment (diagnostic trap)",
            "Most common genetic cause of ANSD",
            "Gene therapy trials in preclinical stages (dual-vector for large coding sequence)",
        ],
        "key_ddx": (
            "Neonatal jaundice ANSD: bilirubin damages SGNs; no OAE loss initially; history of hyperbilirubinemia; "
            "AIFM1 (X-linked ANSD): males; AIFM1 mitochondrial apoptosis; "
            "CMV ANSD: CMV serology/PCR; unilateral or asymmetric; "
            "PJVK (DFNB59): ANSD; absent OAE in some; "
            "GJB2 (DFNB1A): typical SNHL pattern — absent OAE + abnormal ABR (not ANSD); normal OHC/IHC anatomy; "
            "Auditory processing disorder: normal OAE AND ABR; central not peripheral; "
            "Malingering: OAE present + normal ABR at soft levels; inconsistent behavioural thresholds."
        ),
        "systemic_involvement": (
            "NO systemic involvement. Pure cochlear inner-hair-cell synaptic NSHL. "
            "No retinal, no vestibular, no neurological, no cardiac features. "
            "TEMPERATURE-SENSITIVE SUBTYPE (p.Ile515Thr): SNHL fluctuates with body temperature — fever worsens; "
            "not a systemic condition but requires fever management."
        ),
        "onset_age": "Congenital or early prelingual SNHL; detected on newborn ABR screen (OAE passes, ABR fails)",
        "surgical_urgency": "CI urgency: early implantation; ANSD diagnosis must be confirmed genetically before CI",
        "gene_family": "Ferlin family; multi-C2-domain Ca2+-sensing protein; otoferlin is cochlea-specific",
        "morphology": (
            "AUDIOGRAM: moderate to profound SNHL; thresholds variable; "
            "OAE: PRESENT (DPOAE/TEOAE) — OHC intact; "
            "ABR: absent or severely desynchronised wave I-V; "
            "CT/MRI: NORMAL cochlear anatomy (DDx SLC26A4)"
        ),
        "n_patients": 40,
    },

    # -- MYO15A -- DFNB3 (Profound Prelingual SNHL; Short Stereocilia) --------------------------------
    {
        "gene": "MYO15A",
        "alt_name": (
            "MYO15A (MYO15A-3530aa-17p11.2 / AR -- "
            "DFNB3-PROFOUND-PRELINGUAL-SNHL-BILATERAL -- "
            "SHORT-OUTER-HAIR-CELL-STEREOCILIA-ULTRASTRUCTURE-PATHOGNOMONIC -- "
            "p.Arg2169Gln-SOUTH-ASIAN-IRANIAN-FOUNDER-DFNB3-BALI-PEDIGREES -- "
            "COCHLEAR-IMPLANT-GOOD-OUTCOMES-SGN-RELATIVELY-PRESERVED -- "
            "3530aa-LARGEST-MYOSIN-IN-COCHLEA)"
        ),
        "protein": (
            "MYO15A -- 17p11.2 AR -- MYO15A-3530aa -- "
            "Myosin-XVa-Motor-Protein-395kDa-N-Terminal-Extension-SH3-MyTH4-FERM-Domains -- "
            "Stereocilia-Tip-Elongation-Motor-EPS8-Scaffold-Tip-Complex -- "
            "OMIM-Gene-602666-Disease-DFNB3-600316"
        ),
        "locus": "17p11.2",
        "protein_size": "3530 aa / 395 kDa",
        "inheritance": (
            "AR (biallelic); "
            "MYO15A encodes myosin XVa, an unconventional class XV myosin motor protein expressed in cochlear hair cells. "
            "DOMAIN STRUCTURE: long N-terminal extension (unique to MYO15A) → SH3 domain → motor domain → "
            "IQ motifs → coiled-coil → MyTH4 domain → FERM domain → PDZ-ligand C-terminus; "
            "FUNCTION: MYO15A transports EPS8 and WHIRLIN (WHRN) to stereocilia tips; "
            "EPS8 caps the barbed end of actin at tips → controls stereocilia length; "
            "MYO15A LOF → EPS8/WHRN fail to reach tips → actin uncapped → SHORT stereocilia → "
            "mechanotransduction impaired → profound SNHL. "
            "LARGE GENE: MYO15A is 3530 aa — challenges gene therapy (single AAV limited to ~4.7 kb insert); "
            "split-intein dual-vector approaches in development. "
            "ORIGINAL DFNB3 PEDIGREES: Bali, Indonesia (DFNB3) and Pakistani consanguineous families; "
            "FOUNDER: p.Arg2169Gln (FERM domain) — South Asian/Iranian families; "
            "p.Thr1213Ile — Turkish families; many private variants worldwide"
        ),
        "disease_category": (
            "DFNB3 (AR) — profound prelingual non-syndromic SNHL; "
            "hair cell stereocilia tip defect; cochlear implant outcomes generally good"
        ),
        "disease_pathway": (
            "STEREOCILIA ARCHITECTURE: hair cell stereocilia are actin-filled protrusions arranged in staircase rows; "
            "longest stereocilia closest to kinocilium; length gradient critical for deflection mechanics. "
            "MYO15A MOTOR FUNCTION: MYO15A walks along actin filaments in stereocilia shafts toward the plus (barbed) end at tips. "
            "CARGO: MYO15A carries EPS8 (actin barbed-end capper) and WHIRLIN (PDZ scaffold) to stereocilia tips. "
            "EPS8 AT TIPS: caps barbed end of actin → stops polymerisation → sets FINAL stereocilia length; "
            "Without EPS8 at tips → unrestricted actin polymerisation does not occur normally → SHORT, STUNTED stereocilia. "
            "MYO15A LOF: stereocilia fail to elongate to correct length → gradient disrupted → "
            "tip links (CDH23-PCDH15) cannot form correctly → mechanotransduction channels (TMC1/TMC2) not gated → "
            "profound SNHL. "
            "HISTOLOGY: scanning EM shows dramatically short stereocilia in all hair cell rows."
        ),
        "pathognomonic": (
            "MYO15A-DFNB3 CLINICAL FEATURES: "
            "1. PROFOUND PRELINGUAL BILATERAL SNHL: flat audiogram ≥90 dBHL all frequencies; "
            "identified newborn hearing screen; "
            "2. NO VESTIBULAR INVOLVEMENT: vestibular function normal — DDx MYO7A (Usher 1) which has vestibular areflexia; "
            "3. NO RETINAL INVOLVEMENT: ophthalmology normal; "
            "4. NORMAL CT/MRI: cochlear anatomy normal — DDx SLC26A4 (EVA); "
            "5. NON-SYNDROMIC: no systemic features; "
            "ULTRASTRUCTURE (research only): scanning electron microscopy of hair cells shows SHORT stereocilia — "
            "not clinically accessible but pathognomonic on research specimens. "
            "CI OUTCOMES: cochlear implant outcomes generally good to very good; SGNs relatively preserved."
        ),
        "treatment": (
            "COCHLEAR IMPLANT: FIRST-LINE for profound SNHL; early implantation; "
            "CI outcomes generally good (SGNs preserved, cochlear anatomy normal); "
            "HEARING AIDS: trial in early stages; profound loss → limited benefit → CI; "
            "SPEECH-LANGUAGE THERAPY: mandatory; auditory-verbal approach with CI; "
            "GENETIC COUNSELLING: AR 25% risk; consanguinity increases risk (South Asian, Iranian, Turkish pedigrees); "
            "GENE THERAPY: preclinical (split-intein dual-AAV approaches for 3530 aa gene)."
        ),
        "key_features": [
            "DFNB3 — profound prelingual bilateral SNHL; flat audiogram ≥90 dBHL",
            "SHORT hair cell stereocilia (ultrastructure) — MYO15A transports EPS8 to tips",
            "No vestibular, no retinal involvement (DDx MYO7A/Usher 1)",
            "Normal CT/MRI (DDx SLC26A4 EVA)",
            "p.Arg2169Gln South Asian/Iranian founder; Bali Indonesia DFNB3 original pedigree",
            "3530 aa — largest cochlear myosin; gene therapy challenge (dual-vector)",
            "CI outcomes good — SGNs preserved; cochlear anatomy normal",
            "WHRN (DFNB31/USH2D) and EPS8 are downstream cargo — same stereocilia tip pathway",
        ],
        "key_ddx": (
            "MYO7A (USH1B): profound SNHL + VESTIBULAR AREFLEXIA + RP — MYO15A has NO vestibular, NO retinal; "
            "GJB2 (DFNB1A): most common; normal OAE absent; no stereocilia defect; connexin recycling; "
            "OTOF (DFNB9): ANSD — present OAE; MYO15A: absent OAE; "
            "SLC26A4 (DFNB4): EVA on CT — MYO15A has normal CT; "
            "WHRN (DFNB31/USH2D): same stereocilia tip pathway; USH2D has RP; DFNB31 pure HL."
        ),
        "systemic_involvement": (
            "NO systemic involvement. Pure non-syndromic cochlear SNHL. "
            "No vestibular (DDx MYO7A), no retinal (DDx Usher), no skin, no cardiac, no renal features. "
            "Consanguinity pattern common in affected families."
        ),
        "onset_age": "Congenital profound SNHL; identified at newborn hearing screen",
        "surgical_urgency": "CI: early implantation for speech-language; no intraoperative surgical urgency",
        "gene_family": "Unconventional myosin motor protein family (Class XV); stereocilia tip complex motor",
        "morphology": (
            "AUDIOGRAM: flat profound bilateral SNHL (≥90 dBHL all frequencies); "
            "OAE: absent (OHC dysfunction due to short stereocilia); "
            "ABR: absent/threshold elevated (typical SNHL pattern — DDx OTOF-ANSD); "
            "CT/MRI: NORMAL cochlear anatomy"
        ),
        "n_patients": 40,
    },

    # -- TECTA -- DFNB21 (AR Profound) / DFNA8/12 (AD Mid-Frequency U-Shaped) ----------------------
    {
        "gene": "TECTA",
        "alt_name": (
            "TECTA (TECTA-2155aa-11q23.3 / AR-and-AD -- "
            "DFNB21-AR-PROFOUND-PRELINGUAL-SNHL -- "
            "DFNA8-DFNA12-AD-MID-FREQUENCY-U-SHAPED-AUDIOGRAM-PATHOGNOMONIC -- "
            "TECTORIAL-MEMBRANE-COMPONENT-ALPHA-TECTORIN -- "
            "SAME-GENE-DIFFERENT-INHERITANCE-DIFFERENT-PHENOTYPE -- "
            "ZONA-PELLUCIDA-ZP-DOMAINS-ENTACTIN-N-TERMINAL)"
        ),
        "protein": (
            "TECTA -- 11q23.3 AR (DFNB21) / AD (DFNA8/12) -- TECTA-2155aa -- "
            "Alpha-Tectorin-222kDa-Zona-Pellucida-ZP-Domain-Entactin-N-Term-Tectorial-Membrane -- "
            "Non-Collagenous-ECM-Glycoprotein-Tectorial-Membrane-Major-Component -- "
            "OMIM-Gene-602574-Disease-DFNA8-601543-DFNA12-601543-DFNB21-603629"
        ),
        "locus": "11q23.3",
        "protein_size": "2155 aa / 222 kDa",
        "inheritance": (
            "AR (biallelic) → DFNB21 (profound prelingual SNHL); "
            "AD (heterozygous) → DFNA8 or DFNA12 (non-progressive/progressive mid-frequency U-shaped SNHL); "
            "TECTA encodes alpha-tectorin, a major non-collagenous glycoprotein of the tectorial membrane (TM). "
            "DOMAIN STRUCTURE: N-terminal entactin-like domain → von Willebrand factor D domain → "
            "3× zona pellucida (ZP) domains → C-terminal propeptide; "
            "FUNCTION: alpha-tectorin polymerises via ZP domains into the matrix of the tectorial membrane; "
            "TM overlies OHC stereocilia tips — contact at TM–stereocilia interface is required for normal mechanics; "
            "TECTA LOF (AR) → DFNB21 — TM absent or non-functional → profound mechano-coupling failure → severe SNHL; "
            "TECTA dominant missense (ZP domain) → DFNA8/12 — abnormal TM matrix → "
            "selective mid-frequency coupling defect → U-shaped audiogram (mid-frequency dip: 1-2 kHz region). "
            "GENOTYPE-PHENOTYPE: ZP domain missense → AD mid-freq; entactin/vWD domain → AR/AD variable"
        ),
        "disease_category": (
            "DFNB21 (AR, biallelic) — severe-profound prelingual SNHL; "
            "DFNA8/12 (AD, monoallelic) — non-progressive or slowly progressive mid-frequency U-shaped SNHL; "
            "same gene, opposite inheritance → completely different phenotypes"
        ),
        "disease_pathway": (
            "TECTORIAL MEMBRANE (TM): a gel-like acellular matrix overlying the organ of Corti. "
            "OHC stereocilia tips are embedded in the TM — when basilar membrane vibrates, TM-stereocilia "
            "relative motion shears stereocilia → opens mechanotransduction channels. "
            "ALPHA-TECTORIN ROLE: structural component of TM matrix (with beta-tectorin/TECTB, collagen XI); "
            "TM stiffness and geometry are critical for frequency tuning. "
            "DFNB21 (AR): complete TECTA loss → TM fails to form or is severely malformed → "
            "OHC stereocilia never contacted by TM → no mechano-coupling → profound SNHL across all frequencies. "
            "DFNA8/12 (AD missense): mutant alpha-tectorin incorporated into TM → alters TM stiffness at specific regions → "
            "selectively impairs mechano-coupling at mid frequencies (1-2 kHz) → U-shaped audiogram; "
            "high and low frequencies less affected (different TM-basilar membrane coupling)."
        ),
        "pathognomonic": (
            "TECTA CLINICAL FEATURES — INHERITANCE-DEPENDENT: "
            "AR / DFNB21: profound prelingual bilateral SNHL; flat audiogram; no vestibular; no retinal; "
            "AD / DFNA8: MID-FREQUENCY U-SHAPED AUDIOGRAM PATHOGNOMONIC: "
            "audiogram dips at 1-2 kHz with relative sparing of low and high frequencies; "
            "non-progressive (DFNA8) or slowly progressive (DFNA12) in different families; "
            "mid-frequency notch on audiogram → first DDx: noise-induced (notch at 4 kHz) vs TECTA (notch 1-2 kHz); "
            "dominant family history (AD DFNA8/12); progressive across decades; "
            "SAME GENE TEACHING PEARL: TECTA is the CLASSIC EXAMPLE of same gene → "
            "different inheritance → completely different audiometric phenotype; "
            "biallelic null = profound; monoallelic missense = mid-freq selective."
        ),
        "treatment": (
            "DFNB21 (AR profound): COCHLEAR IMPLANT early; outcomes generally good; "
            "DFNA8/12 (AD mid-freq): HEARING AIDS tailored to mid-frequency U-shape; "
            "low-mid frequency amplification; aids in 2nd-3rd decade when dip deepens; "
            "MONITORING: 6-monthly audiometry in DFNA8/12 — progressive variants; "
            "GENETIC COUNSELLING: AR 25% risk (DFNB21); AD 50% risk (DFNA8/12); "
            "NOISE PROTECTION: all TECTA patients — avoid excessive noise; "
            "SCHOOL SUPPORT: FM system for mid-frequency loss (DFNA8/12) — speech intelligibility affected."
        ),
        "key_features": [
            "DFNB21 (AR) — profound prelingual SNHL; DFNA8/12 (AD) — mid-frequency U-shaped SNHL",
            "Mid-frequency U-shaped audiogram (1-2 kHz dip) — PATHOGNOMONIC for DFNA8/12",
            "Same gene, different inheritance → completely different phenotype (classic teaching case)",
            "Alpha-tectorin: major tectorial membrane non-collagenous glycoprotein",
            "ZP domain missense → AD mid-freq; complete LOF → AR profound",
            "No vestibular, no retinal, no systemic involvement (pure NSHL)",
            "11q23.3; zona pellucida domain protein; TM stiffness and frequency tuning",
            "Noise-induced notch at 4 kHz vs TECTA notch at 1-2 kHz — DDx key",
        ],
        "key_ddx": (
            "Noise-induced SNHL: notch at 4 kHz (NOT 1-2 kHz); occupational/recreational noise history; "
            "TECTB (beta-tectorin): also TM component; "
            "KCNQ4 (DFNA2A): progressive HF SNHL (high freq, not mid-freq); "
            "COL11A1/COL11A2: TM collagen components; Stickler syndrome (systemic); "
            "GJB2 DFNB1A (AR): flat profound SNHL; connexin recycling not TM; no mid-freq U-shape."
        ),
        "systemic_involvement": (
            "NO systemic involvement. Pure non-syndromic cochlear SNHL in both AR and AD forms. "
            "No vestibular, no retinal, no neurological, no cardiac, no renal features."
        ),
        "onset_age": "DFNB21: congenital profound; DFNA8: childhood-early adult mid-frequency notch develops",
        "surgical_urgency": "CI for DFNB21 (AR profound); no surgical urgency for DFNA8/12 (mid-freq, managed with aids)",
        "gene_family": "Zona pellucida domain glycoprotein family; tectorial membrane ECM; alpha-tectorin",
        "morphology": (
            "DFNB21 audiogram: flat profound SNHL; "
            "DFNA8/12 audiogram: U-shaped bilateral mid-frequency dip at 1-2 kHz; "
            "OAE: absent in DFNB21; present/reduced in mild DFNA8/12; "
            "CT/MRI: normal cochlear anatomy (no EVA, no Mondini)"
        ),
        "n_patients": 40,
    },

    # -- KCNQ4 -- DFNA2A (Progressive High-Frequency SNHL; Dominant-Negative) ----------------------
    {
        "gene": "KCNQ4",
        "alt_name": (
            "KCNQ4 (KCNQ4-695aa-1p34.2 / AD -- "
            "DFNA2A-PROGRESSIVE-HIGH-FREQUENCY-SNHL-DOMINANT-NEGATIVE -- "
            "ONSET-2ND-3RD-DECADE-BILATERAL-SYMMETRIC-PROGRESSIVE -- "
            "p.Gly285Ser-KOREAN-FOUNDER-MOST-COMMON-KCNQ4-VARIANT -- "
            "KCNQ-CHANNEL-FAMILY-KNCQ-PHARMACOLOGICAL-TARGET -- "
            "DOMINANT-NEGATIVE-MECHANISM-PATHOGNOMONIC)"
        ),
        "protein": (
            "KCNQ4 -- 1p34.2 AD -- KCNQ4-695aa -- "
            "Kv7.4-Voltage-Gated-K-Channel-KCNQ-Family-77kDa -- "
            "Outer-Hair-Cell-Basolateral-Membrane-K-Efflux -- "
            "OMIM-Gene-603537-Disease-DFNA2A-600101"
        ),
        "locus": "1p34.2",
        "protein_size": "695 aa / 77 kDa",
        "inheritance": (
            "AD (haploinsufficiency + dominant-negative mechanisms); "
            "KCNQ4 encodes Kv7.4 (KCNQ4), a voltage-gated K+ channel of the KCNQ family. "
            "KCNQ family: 5 members (KCNQ1-5); KCNQ1 → cardiac IKs; KCNQ2-3 → neuronal M-current; "
            "KCNQ4 → OHC basolateral K+ efflux; KCNQ5 → widespread. "
            "FUNCTION: Kv7.4 channels mediate K+ efflux from OHC cytoplasm; "
            "After mechanotransduction → K+ enters OHC via tip-link channels (TMC1/TMC2) → "
            "KCNQ4 repolarises OHC via basolateral K+ efflux → K+ returned to perilymph → recycling; "
            "OHC electromotility (PRESTIN, SLC26A5) requires proper ionic milieu. "
            "DOMINANT-NEGATIVE: Kv7.4 forms homotetramers; mutant subunit poisons the tetramer → "
            "heterozygous dominant-negative loss of function → OHC K+ recycling fails → OHC degeneration → "
            "progressive SNHL. "
            "FOUNDERS: p.Gly285Ser (c.853G>A) — Korean; p.Trp276Ser — European; p.Gly285Cys — worldwide"
        ),
        "disease_category": (
            "DFNA2A (AD) — progressive bilateral high-frequency SNHL; onset 2nd–3rd decade; "
            "dominant-negative KCNQ channel dysfunction; potential pharmacological target"
        ),
        "disease_pathway": (
            "K+ RECYCLING IN OHC: mechanotransduction channels (TMC1/TMC2) allow K+ in from endolymph → "
            "this depolarises OHC → prestin (SLC26A5) electromotility provides cochlear amplification; "
            "K+ must be efficiently recycled out of OHC basolaterally. "
            "KCNQ4 role: main K+ efflux channel at OHC basolateral membrane; "
            "LOF → K+ accumulates in OHC → chronic depolarisation → oxidative stress → OHC degeneration; "
            "OHC death → loss of cochlear amplification → elevated thresholds → SNHL. "
            "HIGH-FREQUENCY FIRST: OHC at basal (high-freq) turn are metabolically most active and first vulnerable → "
            "progressive loss begins at high frequencies → spreads centrally with age. "
            "PHARMACOLOGICAL TARGET: KCNQ channels are potentiated by open-channel activators (e.g., retigabine/ezogabine); "
            "in vitro and animal models: KCNQ4 activators slow OHC degeneration; "
            "potential future therapeutic approach for DFNA2A patients."
        ),
        "pathognomonic": (
            "KCNQ4-DFNA2A CLINICAL FEATURES: "
            "1. PROGRESSIVE HIGH-FREQUENCY SNHL: bilateral symmetric; starts at high frequencies (4-8 kHz); "
            "onset typically 2nd–3rd decade; "
            "2. AUDIOGRAM SLOPE: downsloping bilateral SNHL — 4 kHz worse than 2 kHz worse than 1 kHz; "
            "progressive slope deepens with age; "
            "3. AD FAMILY HISTORY: vertical transmission; 50% recurrence; "
            "4. TINNITUS: high-frequency tinnitus common as OHC degenerate; "
            "5. PROGRESSION RATE: ~2-3 dB/year at affected frequencies; "
            "by 5th-6th decade: severe-profound SNHL → CI consideration; "
            "6. DOMINANT-NEGATIVE MECHANISM: heterozygous missense → mutant subunit poisons KCNQ4 tetramer; "
            "haploinsufficiency variants (stop, frameshift) have milder/incomplete penetrance."
        ),
        "treatment": (
            "HEARING AIDS: progressive HF SNHL → aids when thresholds ≥25-30 dBHL at speech frequencies; "
            "high-frequency amplification initially; "
            "COCHLEAR IMPLANT: when severe-profound (typically 5th-6th decade); CI outcomes generally good; "
            "NOISE PROTECTION: strict hearing protection — KCNQ4 OHC already metabolically stressed; "
            "noise superimposed → accelerated loss; mandatory ear protection at work/recreation; "
            "AUDIOMETRY: annual monitoring; "
            "TINNITUS MANAGEMENT: masking, CBT, hearing aid tinnitus programs; "
            "GENETIC COUNSELLING: AD 50% risk; predictive testing for at-risk relatives; "
            "PHARMACOLOGICAL (FUTURE): KCNQ channel activators (retigabine analogues) — clinical trials needed; "
            "KCNQ4-specific precision therapy goal."
        ),
        "key_features": [
            "DFNA2A — progressive bilateral high-frequency SNHL; onset 2nd-3rd decade",
            "Dominant-negative mechanism: heterozygous missense poisons Kv7.4 tetramer",
            "Audiogram: bilateral downsloping; 4-8 kHz first; ~2-3 dB/year progression",
            "p.Gly285Ser Korean founder; p.Trp276Ser European; dominant-negative most severe",
            "KCNQ4: OHC basolateral K+ efflux channel; pharmacological target (KCNQ activators)",
            "Tinnitus common; noise protection MANDATORY — accelerates OHC loss",
            "CI outcomes good when severe-profound (5th-6th decade)",
            "KCNQ family: KCNQ1 cardiac; KCNQ2-3 neural M-current; KCNQ4 OHC-specific",
        ],
        "key_ddx": (
            "Noise-induced SNHL: history; notch 4 kHz; no AD family history; "
            "Age-related SNHL (presbycusis): onset later (>60y); no clear AD history; "
            "GJA8 (connexin 50): cataract + SNHL in some; "
            "KCNQ1 (LQT1): cardiac arrhythmia SNHL is via KCNE1 not KCNQ4; "
            "TECTA DFNA8/12: mid-frequency U-shape — KCNQ4 is high-freq slope; "
            "WFS1 (Wolfram): HF SNHL + DM + OA + DI — syndromic."
        ),
        "systemic_involvement": (
            "NO systemic involvement. Pure non-syndromic progressive cochlear SNHL. "
            "No vestibular, no retinal, no cardiac, no neurological features. "
            "Tinnitus is a cochlear feature, not a systemic one."
        ),
        "onset_age": "2nd–3rd decade for high-frequency thresholds; symptomatic typically 3rd–4th decade",
        "surgical_urgency": "No immediate urgency; progressive over decades; CI consideration 5th-6th decade",
        "gene_family": "KCNQ / Kv7 voltage-gated K+ channel family; Kv7.4 isoform; OHC-specific",
        "morphology": (
            "AUDIOGRAM: bilateral symmetric downsloping SNHL; 4-8 kHz worst; progressive; "
            "OAE: reduced/absent at high freq as OHC degenerate; "
            "ABR: elevated threshold; wave I-V present (typical SNHL); "
            "CT/MRI: normal cochlear anatomy"
        ),
        "n_patients": 40,
    },

    # -- LHFPL5 -- DFNB67 (Profound Congenital SNHL; TMC Mechanotransduction Complex) ----------------
    {
        "gene": "LHFPL5",
        "alt_name": (
            "LHFPL5 (LHFPL5-219aa-6p21.31 / AR -- "
            "DFNB67-PROFOUND-CONGENITAL-BILATERAL-SNHL -- "
            "TETRASPAN-MEMBRANE-HAIR-CELL-PROTEIN-TMC1-TMC2-GATING-COMPLEX -- "
            "p.Arg83His-TURKISH-CONSANGUINEOUS-FOUNDER -- "
            "STEREOCILIA-TIP-LINK-MECHANOTRANSDUCTION-ACCESSORY -- "
            "CIB2-LHFPL5-TMC1-TMC2-PCDH15-COMPLEX)"
        ),
        "protein": (
            "LHFPL5 -- 6p21.31 AR -- LHFPL5-219aa -- "
            "TMHS-Tetraspan-Membrane-Protein-Hair-Cell-25kDa-4-TM-Helices -- "
            "Mechanotransduction-Channel-Complex-TMC1-TMC2-Accessory-Subunit -- "
            "OMIM-Gene-609427-Disease-DFNB67-610265"
        ),
        "locus": "6p21.31",
        "protein_size": "219 aa / 25 kDa",
        "inheritance": (
            "AR (biallelic); "
            "LHFPL5 (also called TMHS — Tetraspan Membrane protein of Hair cell Stereocilia) encodes a 4-TM protein "
            "in the mechanotransduction complex at hair cell stereocilia tips. "
            "MECHANOTRANSDUCTION COMPLEX: the stereocilia tip-link lower end (PCDH15) connects to the "
            "mechanotransduction channel complex: PCDH15 → CIB2 → LHFPL5 → TMC1/TMC2 (the channel pore). "
            "LHFPL5 is an auxiliary subunit that stabilises TMC1/TMC2 at the channel site; "
            "LHFPL5 LOF → TMC1/TMC2 mislocalised from stereocilia tips → mechanotransduction channel fails → "
            "profound congenital SNHL. "
            "FOUNDERS: p.Arg83His — Turkish consanguineous; p.Leu117Pro — Pakistani; "
            "c.210+1G>T splice site — worldwide; "
            "CI OUTCOMES: cochlear implant outcomes generally good (SGNs preserved; channel complex failure, not SGN)"
        ),
        "disease_category": (
            "DFNB67 (AR) — profound congenital non-syndromic SNHL; "
            "mechanotransduction channel auxiliary subunit defect; cochlear implant outcomes good"
        ),
        "disease_pathway": (
            "MECHANOTRANSDUCTION CHANNEL COMPLEX: "
            "Sound deflects hair bundles → tip links (CDH23 upper / PCDH15 lower) pull on mechanotransduction channel → "
            "channel opens → K+ and Ca2+ enter → hair cell depolarises → glutamate release. "
            "TMC1/TMC2 are the channel-forming subunits (homologs of MEC-2 in C. elegans); "
            "LHFPL5 (TMHS) is an auxiliary subunit that: (1) physically bridges PCDH15 to TMC1/TMC2; "
            "(2) is required for TMC1/TMC2 trafficking to stereocilia tips; "
            "(3) influences channel conductance properties. "
            "LHFPL5 LOF: TMC1/TMC2 no longer anchored correctly at channel site → "
            "mechanotransduction current absent → no K+ influx → hair cell silent → profound SNHL. "
            "HISTOLOGY: OHC present but mechanotransduction absent; stereocilia structurally intact at early stages; "
            "SGNs preserved initially → good CI candidacy."
        ),
        "pathognomonic": (
            "LHFPL5-DFNB67 CLINICAL FEATURES: "
            "1. PROFOUND CONGENITAL BILATERAL SNHL: flat audiogram ≥90 dBHL; "
            "identified newborn hearing screen; "
            "2. NORMAL OAE initially possible (OHC present, mechanotransduction channel failing — not OHC): "
            "some DFNB67 patients have reduced/absent DPOAE; varies; "
            "3. NO VESTIBULAR INVOLVEMENT: normal vestibular function (DDx MYO7A, SLC26A4); "
            "4. NO RETINAL INVOLVEMENT: ophthalmology normal (DDx Usher); "
            "5. NORMAL CT/MRI: normal cochlear anatomy (DDx SLC26A4 EVA); "
            "6. NON-SYNDROMIC: no systemic features; "
            "CONSANGUINITY: common in Turkish, Pakistani families; "
            "CI OUTCOMES: cochlear implant outcomes generally good."
        ),
        "treatment": (
            "COCHLEAR IMPLANT: FIRST-LINE for profound congenital SNHL; early implantation; "
            "CI outcomes generally good (SGNs preserved); "
            "HEARING AIDS: trial but profound loss → limited benefit → early CI referral; "
            "SPEECH-LANGUAGE THERAPY: mandatory; auditory-verbal approach with CI; "
            "GENETIC COUNSELLING: AR 25% risk; consanguinity assessment; "
            "GENE THERAPY: LHFPL5 small gene (219 aa) — excellent AAV delivery candidate; "
            "preclinical mouse models (Lhfpl5−/−) show gene therapy efficacy."
        ),
        "key_features": [
            "DFNB67 — profound congenital bilateral SNHL; flat audiogram ≥90 dBHL",
            "LHFPL5 (TMHS): auxiliary subunit of TMC1/TMC2 mechanotransduction channel complex",
            "PCDH15 → CIB2 → LHFPL5 → TMC1/TMC2 — mechanotransduction assembly chain",
            "p.Arg83His Turkish consanguineous founder; p.Leu117Pro Pakistani",
            "No vestibular, no retinal, normal CT (pure non-syndromic NSHL)",
            "CI outcomes good — SGNs preserved; channel complex failure only",
            "Small gene (219 aa) — excellent AAV gene therapy candidate; mouse models successful",
            "6p21.31; 4-TM topology; expressed exclusively in hair cell stereocilia",
        ],
        "key_ddx": (
            "TMC1 (DFNB7/11 AR or DFNA36 AD): channel pore subunit; LHFPL5 is the auxiliary subunit; "
            "CIB2 (DFNB48): also part of same complex; profound SNHL; "
            "PCDH15 (USH1F/DFNB23): tip-link lower end; USH1F has RP + vestibular; DFNB23 pure HL; "
            "GJB2 (DFNB1A): most common; connexin recycling, not mechanotransduction channel; "
            "MYO15A (DFNB3): stereocilia tip elongation; similar profound SNHL but different pathway."
        ),
        "systemic_involvement": (
            "NO systemic involvement. Pure non-syndromic cochlear SNHL. "
            "No vestibular, no retinal, no cardiac, no skin, no renal features. "
            "Consanguinity is contextual, not a systemic disease feature."
        ),
        "onset_age": "Congenital profound SNHL; identified at newborn hearing screen",
        "surgical_urgency": "CI: early implantation; no intraoperative surgical urgency",
        "gene_family": "LHFPL/Tetraspan family; TMHS isoform; mechanotransduction auxiliary subunit",
        "morphology": (
            "AUDIOGRAM: flat profound bilateral SNHL; "
            "OAE: variable (may be present early, absent with OHC secondary degeneration); "
            "ABR: absent/threshold elevated; typical SNHL pattern; "
            "CT/MRI: normal cochlear anatomy"
        ),
        "n_patients": 40,
    },

    # -- GJB6 -- DFNB1B (del(GJB6-D13S1830) Large Deletion; Digenic with GJB2) --------------------
    {
        "gene": "GJB6",
        "alt_name": (
            "GJB6 (GJB6-261aa-13q12.11 / AR-digenic-GJB2 -- "
            "DFNB1B-del-GJB6-D13S1830-342kb-LARGE-DELETION-MOST-COMMON-GJB6-ALLELE -- "
            "DIGENIC-DFNB1-ONE-GJB2-PATHOGENIC-ALLELE-PLUS-DEL-GJB6-D13S1830 -- "
            "MLPA-CNV-SEQ-MANDATORY-EXOME-MISSES-LARGE-DELETION -- "
            "SAME-CHROMOSOMAL-REGION-13q12.11-GJB2-GJB6-CO-EXPRESSED)"
        ),
        "protein": (
            "GJB6 -- 13q12.11 AR (biallelic or digenic with GJB2) -- GJB6-261aa -- "
            "Connexin-30-Cx30-30kDa-Gap-Junction-Beta-6-4-TM-Helices -- "
            "Cochlear-Supporting-Cell-Gap-Junction-K+-Recycling-Partner-Cx26 -- "
            "OMIM-Gene-604418-Disease-DFNB1B-612645-DFNA3B"
        ),
        "locus": "13q12.11",
        "protein_size": "261 aa / 30 kDa",
        "inheritance": (
            "AR (biallelic GJB6) or digenic (one GJB2 pathogenic allele + del(GJB6-D13S1830)); "
            "GJB6 encodes Connexin 30 (Cx30), a gap junction protein co-expressed with Cx26 in cochlear supporting cells. "
            "GJB2 and GJB6 are located on the same chromosome (13q12.11), ~35 kb apart; "
            "they share regulatory elements and form heteromeric channels together. "
            "DEL(GJB6-D13S1830): a recurrent ~342 kb deletion removing GJB6 and regulatory region; "
            "this deletion ALSO disrupts the GJB2 enhancer region (sharing regulatory control); "
            "DIGENIC MECHANISM: one GJB2 pathogenic variant (allele 1) + del(GJB6-D13S1830) on opposite chromosome (allele 2) → "
            "DFNB1 non-syndromic hearing loss; "
            "both alleles together reduce cochlear gap junction function below threshold; "
            "SECOND DELETION: del(GJB6-D13S1254) — rarer, also removes GJB6; "
            "MLPA MANDATORY: standard NGS/exome panels do NOT detect large deletions reliably → "
            "MLPA or CNV-seq required when GJB2 single-allele found + no second GJB2 variant."
        ),
        "disease_category": (
            "DFNB1B (digenic or biallelic AR) — non-syndromic hearing loss; "
            "del(GJB6-D13S1830) most common GJB6 allele; MLPA/CNV mandatory for detection; "
            "phenotype similar to GJB2 DFNB1A (moderate to profound prelingual SNHL)"
        ),
        "disease_pathway": (
            "CX26/CX30 HETEROMERIC CHANNELS: Connexin 26 and Connexin 30 co-assemble into gap junction channels in cochlear supporting cells. "
            "Heteromeric Cx26/Cx30 channels have distinct properties from homomeric channels; "
            "cochlear K+ recycling requires both Cx26 and Cx30 to function optimally. "
            "DEL(GJB6-D13S1830) EFFECT: "
            "(1) Removes GJB6 coding sequence → no Cx30 protein on that chromosome; "
            "(2) Also removes a cis-regulatory element controlling GJB2 transcription → "
            "reduces GJB2 expression on that chromosome even if GJB2 coding sequence intact. "
            "DIGENIC: GJB2 pathogenic allele (e.g., p.35delG) on chromosome 1 + del(GJB6-D13S1830) on chromosome 2 → "
            "total cochlear Cx26+Cx30 function reduced below threshold → K+ recycling impaired → SNHL. "
            "PHENOTYPE: similar to biallelic GJB2 (null/null) — moderate to profound SNHL; "
            "exact severity depends on GJB2 residual function."
        ),
        "pathognomonic": (
            "GJB6-DFNB1B / DIGENIC DFNB1 DIAGNOSTIC SCENARIO: "
            "1. SINGLE GJB2 PATHOGENIC VARIANT FOUND, NO SECOND: "
            "genetic lab reports 'heterozygous GJB2 p.35delG — carrier, no diagnosis'; "
            "BUT patient has bilateral moderate-profound SNHL → suspect digenic DFNB1; "
            "2. MLPA OR CNV-SEQ MANDATORY: test for del(GJB6-D13S1830) and del(GJB6-D13S1254); "
            "if deletion found → DFNB1 DIAGNOSIS CONFIRMED (digenic compound het); "
            "3. EXOME/WES MISSES: large deletions not reliably detected by exome sequencing; "
            "standard hearing loss panel reporting only point variants MISSES this diagnosis; "
            "4. PHENOTYPE: bilateral symmetric prelingual SNHL moderate to profound; "
            "flat audiogram; no retinal, no vestibular; "
            "5. CI OUTCOMES: cochlear implant outcomes excellent (same as GJB2 DFNB1A — SGNs preserved)."
        ),
        "treatment": (
            "COCHLEAR IMPLANT: for severe-profound SNHL; EXCELLENT outcomes — same as GJB2; "
            "HEARING AIDS: moderate-severe range; "
            "SPEECH-LANGUAGE THERAPY: early; "
            "GENETIC DIAGNOSIS PROTOCOL: "
            "Step 1: sequence GJB2 (bidirectional Sanger or NGS panel); "
            "Step 2: if single GJB2 variant + SNHL → MLPA for del(GJB6-D13S1830) and del(GJB6-D13S1254); "
            "Step 3: if both identified → digenic DFNB1 confirmed; "
            "GENETIC COUNSELLING: digenic inheritance complicates counselling — "
            "risk to siblings: 25% if both parents carry one allele each; "
            "carrier testing: test both alleles in parents separately; "
            "EXPANDED CARRIER SCREENING: GJB2+GJB6 deletion panel — standard in many commercial carrier screens."
        ),
        "key_features": [
            "DFNB1B — digenic DFNB1: one GJB2 allele + del(GJB6-D13S1830) on opposite chromosome",
            "del(GJB6-D13S1830) ~342 kb deletion — most common GJB6 pathogenic allele",
            "MLPA/CNV-seq MANDATORY — exome/NGS misses large deletions; diagnostic trap",
            "13q12.11: GJB2 and GJB6 on same chromosome ~35 kb apart; shared regulatory elements",
            "Prelingual moderate-profound SNHL; phenotype similar to GJB2 biallelic",
            "Cx26/Cx30 form heteromeric channels — both required for K+ recycling",
            "CI outcomes EXCELLENT — SGNs preserved; same as GJB2",
            "Second deletion: del(GJB6-D13S1254) — also clinically significant, rarer",
        ],
        "key_ddx": (
            "GJB2 biallelic (DFNB1A): same phenotype; biallelic GJB2 — no GJB6 deletion involved; "
            "One GJB2 allele + SNHL WITHOUT del(GJB6): check for other regulatory GJB2 variants (e.g., -23+1G>A splice site); "
            "Dominant GJB6 (DFNA3B): monoallelic GJB6 missense → mild progressive SNHL; "
            "CONNEXIN 31 GJB3 (DFNA2B/DFNB1-related): different connexin, different locus; "
            "Other AR NSHL: MYO15A, LHFPL5 etc. — same phenotype but different gene; "
            "Carrier misdiagnosis: single GJB2 allele ≠ carrier if GJB6 deletion not tested."
        ),
        "systemic_involvement": (
            "NO systemic involvement. Pure non-syndromic cochlear SNHL. "
            "No vestibular, no retinal, no cardiac, no skin features. "
            "KID syndrome (GJB2 dominant gain-of-function) is a completely different entity — not related to GJB6."
        ),
        "onset_age": "Congenital or early prelingual bilateral SNHL; detected newborn hearing screen",
        "surgical_urgency": "CI early for profound SNHL; no intraoperative urgency; same as GJB2 DFNB1A",
        "gene_family": "Connexin / gap junction protein family; Cx30 subunit; heteromeric partner of Cx26",
        "morphology": (
            "AUDIOGRAM: moderate to profound bilateral SNHL; flat or sloping; "
            "OAE: absent in profound loss; "
            "ABR: elevated threshold; typical SNHL pattern; "
            "CT/MRI: NORMAL cochlear anatomy"
        ),
        "n_patients": 40,
    },
]


def _make_patients(gene_entry: dict) -> list[dict]:
    rng = random.Random(SEED_BASE + NSHL_GENES.index(gene_entry))
    gene = gene_entry["gene"]
    patients = []
    for i in range(gene_entry["n_patients"]):
        age = rng.randint(1, 70)
        sex = rng.choice(["M", "F"])
        onset = gene_entry["onset_age"]

        # Phenotype severity varies by gene
        if gene in ("GJB2", "MYO15A", "LHFPL5", "GJB6"):
            if gene == "GJB2":
                severity = rng.choice(["Moderate", "Severe", "Profound", "Profound", "Severe"])
            else:
                severity = "Profound"
        elif gene == "SLC26A4":
            severity = rng.choice(["Moderate-severe", "Severe", "Profound", "Moderate-severe"])
        elif gene == "OTOF":
            severity = rng.choice(["Severe", "Profound", "Severe-profound"])
        elif gene == "TECTA":
            # Depends on AR vs AD — simulate 50/50 mix
            if i < 20:
                severity = "Profound"  # AR DFNB21
            else:
                severity = rng.choice(["Mild-moderate", "Moderate"])  # AD DFNA8/12 mid-freq
        elif gene == "KCNQ4":
            severity = rng.choice(["Mild", "Moderate", "Moderate-severe", "Moderate"])

        ci_status = rng.choice(["Cochlear implant bilateral", "Cochlear implant unilateral", "Hearing aids", "Under evaluation"])
        if gene == "KCNQ4":
            ci_status = rng.choice(["Hearing aids", "Hearing aids", "Hearing aids", "Under evaluation", "Cochlear implant unilateral"])

        speech_outcome = rng.choice(["Excellent", "Good", "Developing", "Fair"])
        if gene == "OTOF":
            speech_outcome = rng.choice(["Excellent", "Excellent", "Good"])  # ANSD CI outcomes excellent

        evs = []
        if gene == "SLC26A4":
            evs.append("EVA on CT confirmed")
            if rng.random() < 0.55:
                evs.append("Thyroid goiter — Pendred syndrome")
        if gene == "GJB2":
            founder = rng.choice(["p.35delG/p.35delG", "p.35delG/p.167delT", "p.235delC compound het", "p.35delG/p.V37I"])
            evs.append(f"Variant: {founder}")
        if gene == "OTOF":
            evs.append("ANSD: OAE present, ABR absent")
            if rng.random() < 0.4:
                evs.append("p.Ile515Thr Iberian founder confirmed")
        if gene == "GJB6":
            evs.append("del(GJB6-D13S1830) + GJB2 p.35delG — digenic DFNB1")
            evs.append("MLPA confirmed deletion")
        if gene == "SLC26A4" and rng.random() < 0.3:
            evs.append("Mondini cochlear malformation on CT")
        if gene == "KCNQ4" and rng.random() < 0.7:
            evs.append("Tinnitus — high frequency")
        if gene == "TECTA" and i >= 20:
            evs.append("U-shaped audiogram 1-2 kHz dip (DFNA8/12)")

        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "age": age,
            "sex": sex,
            "onset": onset[:60],
            "snhl_severity": severity,
            "ci_status": ci_status,
            "speech_outcome": speech_outcome,
            "key_features": "; ".join(gene_entry["key_features"][:3]),
            "extra_events": "; ".join(evs) if evs else "—",
        })
    return patients


def _all_patients() -> list[dict]:
    out = []
    for g in NSHL_GENES:
        out.extend(_make_patients(g))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    """Aggregate summary for the /overview endpoint."""
    patients = _all_patients()
    gene_counts = {}
    severity_counts: dict[str, int] = {}
    ci_counts: dict[str, int] = {}

    for p in patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1
        severity_counts[p["snhl_severity"]] = severity_counts.get(p["snhl_severity"], 0) + 1
        ci_counts[p["ci_status"]] = ci_counts.get(p["ci_status"], 0) + 1

    gene_highlights = {
        g["gene"]: {
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"][:120] + "...",
            "disease_category": g["disease_category"][:120] + "...",
            "pathognomonic_short": g["key_features"][0],
        }
        for g in NSHL_GENES
    }

    return {
        "atlas": "Hereditary-Non-Syndromic-Hearing-Loss-Atlas",
        "subtitle": "Complete 8-Gene NSHL Reference (GJB2 · SLC26A4 · OTOF · MYO15A · TECTA · KCNQ4 · LHFPL5 · GJB6)",
        "total_patients": len(patients),
        "genes_covered": len(NSHL_GENES),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + len(NSHL_GENES) - 1}",
        "patients_per_gene": gene_counts,
        "snhl_severity_distribution": severity_counts,
        "ci_status_distribution": ci_counts,
        "gene_highlights": gene_highlights,
        "clinical_pearls": [
            "GJB2 (DFNB1A): most common AR NSHL globally (~50%); p.35delG European; CI outcomes excellent",
            "SLC26A4 (DFNB4/Pendred): EVA on CT/MRI = pathognomonic; contact sports CONTRAINDICATED",
            "OTOF (DFNB9): ANSD pattern — present OAE + absent ABR; CI first-line (excellent); hearing aids ineffective",
            "MYO15A (DFNB3): profound congenital SNHL; short stereocilia; 3530 aa largest cochlear myosin",
            "TECTA: AR (DFNB21) = profound; AD (DFNA8/12) = mid-frequency U-shaped — same gene, opposite inheritance",
            "KCNQ4 (DFNA2A): progressive HF SNHL; dominant-negative; noise protection mandatory; pharmacological target",
            "LHFPL5 (DFNB67): mechanotransduction auxiliary subunit (TMC1/TMC2 complex); small gene — AAV candidate",
            "GJB6 (DFNB1B): del(GJB6-D13S1830) digenic with GJB2; MLPA/CNV-seq MANDATORY — exome misses deletion",
        ],
    }


def get_breakdown() -> dict:
    """Per-gene breakdown for the /breakdown endpoint."""
    breakdown = {}
    for g in NSHL_GENES:
        patients = _make_patients(g)
        ci_breakdown: dict[str, int] = {}
        severity_breakdown: dict[str, int] = {}
        for p in patients:
            ci_breakdown[p["ci_status"]] = ci_breakdown.get(p["ci_status"], 0) + 1
            severity_breakdown[p["snhl_severity"]] = severity_breakdown.get(p["snhl_severity"], 0) + 1

        breakdown[g["gene"]] = {
            "gene": g["gene"],
            "alt_name": g["alt_name"],
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway": g["disease_pathway"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "key_features": g["key_features"],
            "key_ddx": g["key_ddx"],
            "systemic_involvement": g["systemic_involvement"],
            "onset_age": g["onset_age"],
            "surgical_urgency": g["surgical_urgency"],
            "gene_family": g["gene_family"],
            "morphology": g["morphology"],
            "n_patients": g["n_patients"],
            "snhl_severity_distribution": severity_breakdown,
            "ci_status_distribution": ci_breakdown,
            "sample_patients": patients[:5],
        }
    return {"breakdown_by_gene": breakdown, "total_genes": len(NSHL_GENES)}


def get_definitions() -> dict:
    """Clinical definitions and glossary for the /definitions endpoint."""
    return {
        "atlas": "Hereditary-Non-Syndromic-Hearing-Loss-Atlas",
        "key_clinical_definitions": {
            "NSHL": (
                "Non-Syndromic Hearing Loss: hearing loss without other major clinical features. "
                "Locus nomenclature: DFNB = autosomal recessive; DFNA = autosomal dominant; DFNX = X-linked. "
                "~50% of congenital SNHL is genetic; ~80% of genetic SNHL is autosomal recessive."
            ),
            "ANSD": (
                "Auditory Neuropathy Spectrum Disorder: present otoacoustic emissions (OAE) + "
                "absent or severely abnormal auditory brainstem response (ABR). "
                "Indicates intact OHC function but failed IHC→SGN synaptic transmission. "
                "OTOF is the most common genetic cause of ANSD. "
                "Cochlear implant FIRST-LINE; hearing aids often ineffective."
            ),
            "EVA": (
                "Enlarged Vestibular Aqueduct: vestibular aqueduct midpoint width >1.5 mm on CT (Cincinnati criteria); "
                "hallmark of SLC26A4 (DFNB4/Pendred syndrome). "
                "Contact sports ABSOLUTELY CONTRAINDICATED — head trauma triggers sudden SNHL."
            ),
            "DFNB1": (
                "Most common AR NSHL locus. DFNB1A = biallelic GJB2 (~50% AR NSHL). "
                "DFNB1B = del(GJB6-D13S1830) biallelic or digenic with GJB2. "
                "Both GJB2 and GJB6 are on 13q12.11 ~35 kb apart."
            ),
            "CI_outcomes_ranking": (
                "Cochlear implant outcomes by gene (best → good → variable): "
                "GJB2/GJB6 (EXCELLENT) = OTOF-ANSD (EXCELLENT) > MYO15A (GOOD) > LHFPL5 (GOOD) > "
                "SLC26A4 (GOOD, caution perilymph gusher) > TECTA-DFNB21 (GOOD) > KCNQ4 (GOOD, late-stage)."
            ),
            "Pendred_syndrome": (
                "Autosomal recessive: biallelic SLC26A4 + EVA + euthyroid or subclinical goiter. "
                "Perchlorate discharge test >10% = organification defect. "
                "Annual TSH/FT4 monitoring mandatory."
            ),
            "GJB2_founders": (
                "p.35delG (c.35delG): European/Mediterranean — ~70% of European GJB2 alleles. "
                "p.167delT (c.501delT): Ashkenazi Jewish — ~50% of Ashkenazi GJB2 alleles. "
                "p.235delC (c.235delC): East Asian — ~70% of East Asian GJB2 alleles. "
                "p.W77X: South Indian. p.R143W: Middle Eastern."
            ),
            "GJB6_deletion_MLPA": (
                "del(GJB6-D13S1830): ~342 kb deletion at 13q12.11 removing GJB6 + GJB2 regulatory element. "
                "NOT DETECTED by exome sequencing or standard NGS panels. "
                "MLPA (multiple ligation-dependent probe amplification) or CNV-seq MANDATORY "
                "when single GJB2 variant found + SNHL diagnosis."
            ),
            "KCNQ4_dominant_negative": (
                "KCNQ4 (Kv7.4) forms homotetramers; dominant-negative missense → mutant subunit poisons tetramer. "
                "p.Gly285Ser Korean founder — dominant-negative, moderate-severe phenotype. "
                "KCNQ channel activators (retigabine analogues) — potential precision therapy."
            ),
            "TECTA_dual_inheritance": (
                "TECTA encodes alpha-tectorin (tectorial membrane). "
                "Biallelic LOF → DFNB21 (profound prelingual SNHL). "
                "Monoallelic ZP domain missense → DFNA8/12 (AD mid-frequency U-shaped audiogram). "
                "CLASSIC example: same gene → opposite inheritance → completely different audiometric phenotype."
            ),
            "temperature_sensitive_OTOF": (
                "p.Ile515Thr OTOF variant (Iberian/Spanish founder) causes temperature-sensitive ANSD: "
                "SNHL worsens with fever, partially recovers when afebrile. "
                "Prompt antipyretics in febrile illness. Cochlear implant eliminates temperature sensitivity."
            ),
        },
        "diagnostic_protocol": {
            "step_1": "Newborn hearing screen: ABR + OAE; refer if fail",
            "step_2": "Diagnostic ABR + DPOAE/TEOAE: if OAE present + ABR absent/abnormal → ANSD diagnosis → OTOF first",
            "step_3": "CT temporal bone: if EVA → SLC26A4 first; if cochlear malformation → dedicated workup",
            "step_4": "Genetic panel: GJB2 sequencing (tier 1 — most common)",
            "step_5": "MLPA for del(GJB6-D13S1830)/del(GJB6-D13S1254) if single GJB2 allele found",
            "step_6": "Comprehensive hearing loss gene panel (100+ genes) if steps 1-5 non-diagnostic",
            "step_7": "Thyroid function (TSH/FT4) if EVA or SLC26A4 confirmed",
            "step_8": "Ophthalmology if any suspicion of syndromic HL (Usher, TIMM8A, etc.)",
        },
        "cochlear_implant_criteria": {
            "profound_SNHL": "≥90 dBHL bilateral → CI first-line",
            "severe_SNHL": "70-89 dBHL bilateral with <50% speech discrimination → CI",
            "ANSD": "Present OAE + absent ABR regardless of behavioural thresholds → CI first-line (hearing aids unreliable)",
            "age_target": "Before 12 months for prelingual congenital SNHL — optimal speech-language outcomes",
            "bilateral": "Both ears preferred; bilateral CI better outcomes",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI testing
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    ov = get_overview()
    print(f"Overview — total patients: {ov['total_patients']}, genes: {ov['genes_covered']}")

    bk = get_breakdown()
    print(f"Breakdown — genes returned: {len(bk['breakdown_by_gene'])}")

    df = get_definitions()
    print(f"Definitions — keys: {list(df.keys())}")
