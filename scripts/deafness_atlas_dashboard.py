#!/usr/bin/env python3
"""Deafness-Atlas — Complete 8-Gene Hereditary Deafness & Usher Syndrome Atlas
GJB2   (Connexin 26; ~226 aa; 13q12.11; DFNB1A; AR; most common AR SNHL; p.35delG European founder; CI curative) ·
SLC26A4 (Pendrin; ~780 aa; 7q22.3; DFNB4/Pendred syndrome; AR; EVA pathognomonic; thyroid goiter 70%) ·
OTOF   (Otoferlin; ~1997 aa; 2p23.3; DFNB9/ANSD; AR; Normal OAE + absent ABR = ANSD; CI curative; HA fails) ·
MYO7A  (Myosin VIIA; ~2215 aa; 11q13.5; Usher 1B; AR; deaf + RP + absent vestibular; deafblind adult onset) ·
USH2A  (Usherin; ~5202 aa; 1q41; Usher 2A; AR; most common Usher; normal vestibular; RP mid-childhood) ·
CDH23  (Cadherin-23; ~3354 aa; 10q22.1; Usher 1D/DFNB12; AR; tip-link cadherin; profound SNHL + RP + vestibular) ·
PCDH15 (Protocadherin-15; ~1955 aa; 10q21.1; Usher 1F/DFNB23; AR; ankle-link cadherin; R245X Ashkenazi) ·
CLRN1  (Clarin-1; ~232 aa; 3q25.1; Usher 3A; AR; PROGRESSIVE SNHL + RP + variable vestibular; p.Asn48Lys Finnish)
320-patient aggregate cohort (8 × 40, seeds 1126–1133)
"""

import random

SEED_BASE = 1126

DEAFNESS_GENES = [
    # ── GJB2 — DFNB1A, Connexin 26, most common AR SNHL ────────────────────────
    {
        "gene": "GJB2",
        "protein": "Connexin 26 (Cx26; GJB2)",
        "alias": "GJB2; OMIM gene 121011; 13q12.11; ~226 aa; DFNB1A (OMIM #220290); AR biallelic; most common AR SNHL worldwide; p.35delG European founder (>40% alleles in Europeans); carrier rate 1:33 Europeans; CI curative; no vestibular; no RP",
        "aa": "~226 aa",
        "kDa": "~26 kDa",
        "gene_class": "Gap junction beta-2 protein; potassium recycling in cochlea",
        "locus": "13q12.11",
        "omim_gene": 121011,
        "omim_disease": 220290,
        "phenotype": "DFNB1A — Non-syndromic autosomal recessive sensorineural hearing loss",
        "disease": (
            "GJB2 encodes Connexin 26 (Cx26), a gap junction protein critical for potassium (K+) "
            "recycling in the cochlea. "
            "NORMAL FUNCTION: Cx26 forms hexameric hemichannels (connexons) between cochlear "
            "supporting cells (Deiters, Hensen, inner sulcus), allowing K+ to flow from hair cells "
            "back to the endolymph via the recycling pathway. This maintains endocochlear potential "
            "(EP ~+80 mV), essential for sound transduction by inner hair cells (IHCs). "
            "PATHOMECHANISM: biallelic LOF variants in GJB2 → absent or non-functional Cx26 → "
            "disrupted K+ recycling → K+ accumulates in perilymph → hair cell depolarisation and "
            "degeneration → congenital sensorineural hearing loss. "
            "PHENOTYPE: pure non-syndromic SNHL (profound in most biallelic null/null); "
            "no retinitis pigmentosa; no vestibular dysfunction; no other organ involvement. "
            "FOUNDER VARIANTS: p.35delG (c.35delG, Gly12Val frameshift) — most common in Europeans "
            "(>40% GJB2 alleles in European deaf populations); p.167delT — Ashkenazi Jewish founder; "
            "p.235delC — East Asian founder (Japanese, Korean, Chinese). "
            "Compound heterozygotes (p.35delG + other pathogenic allele) are common. "
            "COCHLEAR IMPLANT: excellent outcomes in GJB2-DFNB1A — cochlear anatomy normal, "
            "auditory nerve intact, no additional syndromic complications."
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic LOF (homozygous or compound heterozygous); carrier rate 1:33 Europeans",
        "hallmark": "Congenital profound SNHL; normal cochlear anatomy on MRI/CT; no vestibular dysfunction; no RP; carrier 1:33 Europeans",
        "key_ddx": "SLC26A4 (EVA on CT), OTOF (ANSD — normal OAE), Usher (RP + vestibular), CMV (maternal SNHL)",
        "founder_variant": "p.35delG (c.35delG) European; p.167delT Ashkenazi; p.235delC East Asian",
        "onset_pattern": "Congenital (prelingual)",
        "seed": 1126,
        "cohort_n": 40,
        # Cohort generation weights
        "severity_weights": {"Profound": 0.70, "Severe": 0.20, "Moderate": 0.10},
        "has_rp_rate": 0.0,
        "has_vestibular_rate": 0.0,
        "ci_rate": 0.75,
        "ha_rate": 0.60,
        "ansd_rate": 0.0,
        "congenital_rate": 1.0,
        "nbs_detected_rate": 0.90,
        "usher_type": None,
        "ci_recommended": True,
        "ansd_gene": False,
        "aminoglycoside_risk": "ABSOLUTE AVOID — additional sensorineural cochlear toxicity on pre-existing GJB2 SNHL; use alternative antibiotics",
        "loop_diuretic_risk": "HIGH RISK ototoxic synergy — furosemide/ethacrynic acid potentiate SNHL; if essential: minimum dose, monitor hearing",
        "vpa_risk": "Not a primary concern for GJB2-DFNB1A; VPA ototoxicity not gene-specific",
    },

    # ── SLC26A4 — DFNB4 / Pendred Syndrome, EVA, thyroid goiter ─────────────────
    {
        "gene": "SLC26A4",
        "protein": "Pendrin (SLC26A4)",
        "alias": "SLC26A4; OMIM gene 605646; 7q22.3; ~780 aa; DFNB4 (OMIM #600791) / Pendred Syndrome (OMIM #274600); AR; 2nd most common AR SNHL; EVA pathognomonic; Pendrin = chloride/iodide/formate antiporter; thyroid goiter 70%; perchlorate discharge test positive; CI effective",
        "aa": "~780 aa",
        "kDa": "~86 kDa",
        "gene_class": "Solute carrier family 26, member 4; chloride/iodide antiporter; expressed in inner ear, thyroid, kidney",
        "locus": "7q22.3",
        "omim_gene": 605646,
        "omim_disease": 274600,
        "phenotype": "DFNB4 (isolated EVA) / Pendred Syndrome (EVA + thyroid goiter)",
        "disease": (
            "SLC26A4 encodes Pendrin, a chloride/iodide/formate antiporter expressed in cochlear "
            "endolymph-producing epithelium, thyroid follicular cells, and renal cortical collecting "
            "ducts. "
            "NORMAL FUNCTION in cochlea: Pendrin localises to the apical membrane of cochlear "
            "epithelial cells bordering the endolymph (type B intercalated cells of stria vascularis). "
            "It transports Cl− out of and HCO3− into endolymph, maintaining endolymphatic fluid "
            "composition and volume. Pendrin also expressed in the vestibular aqueduct epithelium. "
            "NORMAL FUNCTION in thyroid: Pendrin transports iodide across the apical membrane "
            "of thyroid follicular cells into the follicular lumen — essential step for thyroid "
            "hormone synthesis. "
            "PATHOMECHANISM: biallelic LOF variants → absent Pendrin → "
            "COCHLEA: endolymphatic acidification + volume dysregulation → "
            "Enlarged Vestibular Aqueduct (EVA) — a bony canal enlargement visible on CT temporal "
            "bone (EVA: width >1.5 mm at midpoint, or >2 mm at operculum — PATHOGNOMONIC). "
            "SNHL: progressive/fluctuating (with episodes of sudden worsening); "
            "moderate-to-profound; low-frequency preserved initially in some. "
            "THYROID (Pendred syndrome): impaired iodide transport into follicular lumen → "
            "organification defect → partial hypothyroidism; thyroid goiter (70%) develops from "
            "childhood; perchlorate discharge test positive (>10% discharge = organification defect). "
            "CLINICAL SPECTRUM: "
            "DFNB4: EVA + SNHL only (no clinically apparent thyroid disease); "
            "Pendred Syndrome: EVA + SNHL + goiter (± subclinical hypothyroidism). "
            "HEAD TRAUMA AND BAROTRAUMA: physical trauma, Valsalva, diving, high-impact sports → "
            "acute endolymph pressure changes → sudden hearing deterioration (irreversible). "
            "Patients must STRICTLY AVOID head trauma and pressure changes — a critical management rule."
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic LOF (compound heterozygous common); genotype-phenotype correlation imperfect",
        "hallmark": "EVA on CT temporal bone (PATHOGNOMONIC) + progressive/fluctuating SNHL ± thyroid goiter; perchlorate discharge test positive",
        "key_ddx": "GJB2 (no EVA, stable SNHL), OTOF (ANSD), Mondini dysplasia (incomplete partition type II — common with SLC26A4), Waardenburg (pigmentation)",
        "founder_variant": "p.IVS8+1G>A (common in many populations); p.L236P European; p.T416P Asian",
        "onset_pattern": "Congenital or early childhood; progressive + fluctuating",
        "seed": 1127,
        "cohort_n": 40,
        "severity_weights": {"Severe": 0.40, "Profound": 0.35, "Moderate": 0.25},
        "has_rp_rate": 0.0,
        "has_vestibular_rate": 0.30,
        "ci_rate": 0.55,
        "ha_rate": 0.65,
        "ansd_rate": 0.0,
        "congenital_rate": 0.85,
        "nbs_detected_rate": 0.85,
        "usher_type": None,
        "ci_recommended": True,
        "ansd_gene": False,
        "aminoglycoside_risk": "ABSOLUTE AVOID — aminoglycosides accelerate SNHL in SLC26A4 EVA; use alternative antibiotics",
        "loop_diuretic_risk": "HIGH RISK — furosemide + EVA = heightened ototoxicity; potentiates endolymph disruption; avoid if possible",
        "vpa_risk": "Not a primary concern for SLC26A4-DFNB4/Pendred",
    },

    # ── OTOF — DFNB9 / Auditory Neuropathy Spectrum Disorder (ANSD) ──────────────
    {
        "gene": "OTOF",
        "protein": "Otoferlin (OTOF)",
        "alias": "OTOF; OMIM gene 603681; 2p23.3; ~1997 aa; DFNB9 (OMIM #601071) / ANSD; AR; otoferlin = IHC synaptic vesicle exocytosis; Normal OAE + absent/abnormal ABR = PATHOGNOMONIC ANSD; CI curative (HA fails); temperature-sensitive variants",
        "aa": "~1997 aa",
        "kDa": "~227 kDa",
        "gene_class": "Ferlin family; C2-domain calcium-sensor protein; inner hair cell synaptic vesicle exocytosis",
        "locus": "2p23.3",
        "omim_gene": 603681,
        "omim_disease": 601071,
        "phenotype": "DFNB9 — Auditory Neuropathy Spectrum Disorder (ANSD) due to OTOF variants",
        "disease": (
            "OTOF encodes Otoferlin, a large C2-domain calcium-sensor protein expressed specifically "
            "in inner hair cells (IHCs) of the cochlea. "
            "NORMAL FUNCTION: Otoferlin is the primary calcium sensor for synaptic vesicle "
            "exocytosis at the IHC ribbon synapse — when a sound stimulus depolarises the IHC, "
            "calcium influx through CaV1.3 channels triggers otoferlin-dependent vesicle fusion "
            "with the presynaptic membrane → glutamate release → activation of spiral ganglion "
            "neurons → auditory brainstem response (ABR) generated. "
            "Outer hair cells (OHCs) do NOT require otoferlin for their electromotility function. "
            "PATHOMECHANISM: biallelic LOF variants → absent/non-functional otoferlin → "
            "IHC synaptic vesicle release failure → sound TRANSDUCTION by OHCs is intact "
            "(otoacoustic emissions PRESERVED — OAEs present/normal) → "
            "but TRANSMISSION from IHC to spiral ganglion is absent → ABR absent or severely "
            "abnormal → AUDITORY NEUROPATHY SPECTRUM DISORDER (ANSD). "
            "CLINICAL HALLMARK (PATHOGNOMONIC): Normal or near-normal OAEs (cochlear amplifier "
            "intact) + Absent/abnormal ABR (no IHC→nerve synchrony). "
            "HEARING AID FAILURE: conventional hearing aids amplify sound but cannot restore "
            "neural synchrony at the IHC synapse → hearing aid provides little or no benefit in OTOF-ANSD. "
            "COCHLEAR IMPLANT CURATIVE: CI bypasses the spiral ganglion and stimulates the "
            "cochlear nerve electrically → restores neural synchrony → excellent speech outcomes "
            "(CI outcomes in OTOF-ANSD among best of all CI aetiologies). "
            "TEMPERATURE-SENSITIVE VARIANTS: some OTOF variants (e.g. p.Ile515Thr) cause "
            "hearing loss that worsens with fever (temperature-sensitive ANSD) — hearing can "
            "fluctuate dramatically with body temperature, mimicking auditory processing disorder. "
            "GENE THERAPY: AAV-OTOF cochlear gene therapy clinical trials (2023-2024) showing "
            "dramatic hearing restoration — first successful gene therapy for hereditary deafness."
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic LOF; no dominant form; p.Q829X common in Spanish populations",
        "hallmark": "Normal OAEs + absent/abnormal ABR = PATHOGNOMONIC ANSD; hearing aid fails; CI curative; temperature-sensitive variants",
        "key_ddx": "GJB2 (absent OAEs), SLC26A4 (EVA on CT), neonatal hypoxia-ANSD (bilateral, usually mild), auditory synaptopathy from CABP2/OTOF",
        "founder_variant": "p.Q829X Spanish; p.Ile515Thr (temperature-sensitive); biallelic mutations common across populations",
        "onset_pattern": "Congenital (prelingual); temperature-sensitive variant may be fluctuating",
        "seed": 1128,
        "cohort_n": 40,
        "severity_weights": {"Profound": 0.65, "Severe": 0.25, "Moderate": 0.10},
        "has_rp_rate": 0.0,
        "has_vestibular_rate": 0.0,
        "ci_rate": 0.80,
        "ha_rate": 0.15,  # HA generally fails in ANSD
        "ansd_rate": 1.0,
        "congenital_rate": 0.90,
        "nbs_detected_rate": 0.85,
        "usher_type": None,
        "ci_recommended": True,
        "ansd_gene": True,
        "aminoglycoside_risk": "ABSOLUTE AVOID — aminoglycosides destroy OHCs → removes residual cochlear function; CI outcomes worsen with prior aminoglycoside exposure",
        "loop_diuretic_risk": "HIGH RISK — furosemide potentiates aminoglycoside ototoxicity; avoid concurrent use",
        "vpa_risk": "Not a primary concern for OTOF-ANSD",
    },

    # ── MYO7A — Usher Syndrome Type 1B ─────────────────────────────────────────
    {
        "gene": "MYO7A",
        "protein": "Myosin VIIA (MYO7A)",
        "alias": "MYO7A; OMIM gene 276903; 11q13.5; ~2215 aa; Usher syndrome type 1B (OMIM #276900); AR; deaf + RP + absent vestibular (Usher type 1); deafblindness adult onset; profound congenital SNHL; early CI recommended; annual retinal monitoring mandatory",
        "aa": "~2215 aa",
        "kDa": "~254 kDa",
        "gene_class": "Unconventional myosin motor protein; stereocilia development; retinal photoreceptor maintenance",
        "locus": "11q13.5",
        "omim_gene": 276903,
        "omim_disease": 276900,
        "phenotype": "Usher Syndrome Type 1B — congenital profound SNHL + RP + absent vestibular function",
        "disease": (
            "MYO7A encodes Myosin VIIA, an unconventional myosin motor protein expressed in "
            "cochlear hair cell stereocilia and retinal photoreceptor cells. "
            "NORMAL FUNCTION in cochlea: Myosin VIIA localises to the stereocilia of cochlear "
            "hair cells, where it is essential for the formation and maintenance of the stereocilia "
            "bundle. It transports cargo along actin filaments and maintains tip-link tension "
            "required for mechanotransduction. Critical during embryonic hair cell development. "
            "NORMAL FUNCTION in retina: Myosin VIIA is expressed in retinal pigment epithelium "
            "(RPE), where it is required for melanosome transport and phagosome processing — "
            "essential for photoreceptor outer segment (POS) renewal. Also expressed in "
            "photoreceptors themselves. "
            "PATHOMECHANISM: biallelic LOF variants → absent/non-functional Myosin VIIA → "
            "COCHLEA: stereocilia bundle disorganisation → profound congenital SNHL (prelingual); "
            "VESTIBULAR SYSTEM: absent/severely reduced vestibular hair cell function → "
            "profound bilateral vestibular hypofunction → delayed motor milestones (sitting, "
            "standing, walking delayed by months); absent vestibulo-ocular reflex (VOR). "
            "RETINA: progressive RPE failure → photoreceptor degeneration → "
            "Retinitis Pigmentosa (RP): bone-spicule pigmentation, arteriolar attenuation, "
            "disc pallor on fundoscopy; ERG shows reduced/absent rod responses (scotopic) first, "
            "then cone responses (photopic); night blindness (nyctalopia) is typically the FIRST "
            "visual symptom (ages 8-16 years) → progressive tunnel vision → legal blindness. "
            "USHER TYPE 1 HALLMARKS: profound congenital SNHL + absent vestibular function + "
            "early-onset RP → DEAFBLINDNESS in adulthood. "
            "MANAGEMENT: early cochlear implant (recommended before age 2 years) for auditory "
            "rehabilitation; annual ophthalmology from childhood; low-vision aids; orientation "
            "and mobility training; driving must CEASE when visual fields restricted."
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic LOF (compound heterozygous common); MYO7A most common Usher type 1 gene",
        "hallmark": "Profound congenital SNHL + absent vestibular (delayed walking) + RP (night blindness first) = Usher type 1; deafblindness in adulthood",
        "key_ddx": "CDH23-Usher1D (worse prognosis), PCDH15-Usher1F, non-Usher SNHL (no RP), Waardenburg type 4 (pigmentation), Bardet-Biedl (obesity + polydactyly)",
        "founder_variant": "Diverse; R244P common in multiple populations; c.470T>C Spanish",
        "onset_pattern": "Congenital (SNHL + vestibular); RP onset teens (night blindness first)",
        "seed": 1129,
        "cohort_n": 40,
        "severity_weights": {"Profound": 0.92, "Severe": 0.07, "Moderate": 0.01},
        "has_rp_rate": 1.0,
        "has_vestibular_rate": 1.0,
        "ci_rate": 0.85,
        "ha_rate": 0.10,  # profound; HA limited benefit; CI preferred
        "ansd_rate": 0.0,
        "congenital_rate": 1.0,
        "nbs_detected_rate": 0.88,
        "usher_type": "1",
        "ci_recommended": True,
        "ansd_gene": False,
        "aminoglycoside_risk": "ABSOLUTE AVOID — aminoglycosides accelerate outer hair cell loss on top of existing SNHL; also retinal toxicity reported with systemic aminoglycosides",
        "loop_diuretic_risk": "HIGH RISK — potentiates cochlear toxicity; avoid furosemide/ethacrynic acid in all Usher syndrome patients",
        "vpa_risk": "Not a primary ototoxicity concern for MYO7A; VPA retinal toxicity theoretically additive — discuss with ophthalmology",
    },

    # ── USH2A — Usher Syndrome Type 2A ─────────────────────────────────────────
    {
        "gene": "USH2A",
        "protein": "Usherin (USH2A)",
        "alias": "USH2A; OMIM gene 608400; 1q41; ~5202 aa; Usher syndrome type 2A (OMIM #276901); AR; most common Usher syndrome (~50% all Usher); normal vestibular; RP mid-childhood/teens; moderate-severe SNHL (low freq preserved); p.Glu767Ser European founder ~30% alleles; ERG diagnostic; HA + CI considered",
        "aa": "~5202 aa",
        "kDa": "~570 kDa",
        "gene_class": "Extracellular matrix protein; ankle-link complex of stereocilia; photoreceptor calyceal process",
        "locus": "1q41",
        "omim_gene": 608400,
        "omim_disease": 276901,
        "phenotype": "Usher Syndrome Type 2A — moderate-severe SNHL + RP + NORMAL vestibular function",
        "disease": (
            "USH2A encodes Usherin, a large extracellular matrix protein and the most common "
            "Usher syndrome gene, accounting for ~50% of all Usher syndrome cases. "
            "NORMAL FUNCTION in cochlea: Usherin localises to the ankle-link region of stereocilia "
            "(near the base), where it maintains the integrity of the stereocilia bundle through "
            "its interaction with Whirlin and VLGR1 (ADGRV1). It is essential for the development "
            "of a specialised microvilli-based ankle-link complex during hair cell maturation. "
            "NORMAL FUNCTION in retina: Usherin localises to the calyceal processes of "
            "photoreceptors — finger-like projections surrounding the connecting cilium between "
            "inner and outer segments. It anchors the photoreceptor outer segment and supports "
            "ongoing outer segment renewal. "
            "PATHOMECHANISM: biallelic LOF variants → absent/truncated Usherin → "
            "COCHLEA: ankle-link complex disruption → moderate-severe SNHL; "
            "low-frequency hearing is PRESERVED better than high-frequency (sloping audiogram) — "
            "important for hearing aid candidacy; "
            "VESTIBULAR: NORMAL vestibular function (key distinguishing feature from Usher type 1); "
            "patients walk at normal age, no vestibular dysfunction. "
            "RETINA: calyceal process failure → progressive rod photoreceptor degeneration → "
            "bone-spicule RP on fundoscopy (age 10-20 years onset typically); "
            "ERG (electroretinogram) shows reduced scotopic (rod) responses BEFORE fundoscopic "
            "signs appear — ERG is DIAGNOSTIC, not fundoscopy alone; "
            "Night driving MUST CEASE once RP diagnosed. "
            "FOUNDER VARIANT: p.Glu767Ser (c.2299delG) — frameshift; >30% USH2A alleles in "
            "European populations. "
            "USHER TYPE 2 FEATURES: moderate-severe SNHL + normal vestibular + RP onset teens = "
            "Usher type 2; less severe than type 1 in SNHL and motor function."
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic; p.Glu767Ser founder common in Europeans; large gene — CNV/deletion screening needed",
        "hallmark": "Most common Usher syndrome; NORMAL vestibular; RP onset teens; moderate-severe SNHL (low-freq preserved); ERG diagnostic; p.Glu767Ser founder",
        "key_ddx": "Usher 1 (MYO7A/CDH23/PCDH15 — absent vestibular, profound SNHL), Usher 3 (CLRN1 — progressive), DFNB1A (no RP), isolated RP",
        "founder_variant": "p.Glu767Ser (c.2299delG) — >30% European USH2A alleles; p.Cys759Phe also common",
        "onset_pattern": "Congenital SNHL; RP onset mid-childhood to teens; normal vestibular",
        "seed": 1130,
        "cohort_n": 40,
        "severity_weights": {"Severe": 0.45, "Moderate": 0.40, "Profound": 0.15},
        "has_rp_rate": 1.0,
        "has_vestibular_rate": 0.0,  # NORMAL vestibular in Usher 2A
        "ci_rate": 0.35,
        "ha_rate": 0.80,  # moderate-severe → HA effective initially
        "ansd_rate": 0.0,
        "congenital_rate": 1.0,
        "nbs_detected_rate": 0.80,
        "usher_type": "2",
        "ci_recommended": True,  # when severe/profound or HA inadequate
        "ansd_gene": False,
        "aminoglycoside_risk": "ABSOLUTE AVOID — aminoglycosides cause additional cochlear toxicity on pre-existing SNHL; also potential additive retinal toxicity",
        "loop_diuretic_risk": "HIGH RISK — ototoxic in all hereditary SNHL; avoid furosemide unless essential; monitor audiogram",
        "vpa_risk": "VPA retinal toxicity potential additive to RP — discuss with ophthalmology before prescribing; not primary concern but document discussion",
    },

    # ── CDH23 — Usher Syndrome Type 1D / DFNB12 ─────────────────────────────────
    {
        "gene": "CDH23",
        "protein": "Cadherin-23 (CDH23)",
        "alias": "CDH23; OMIM gene 605516; 10q22.1; ~3354 aa; Usher 1D (OMIM #601067) / DFNB12 (OMIM #601386); AR; tip-link cadherin; deafness + RP + severe vestibular dysfunction (Usher type 1); profound congenital SNHL; delayed walking (vestibular); worse visual prognosis than USH2A; CI recommended",
        "aa": "~3354 aa",
        "kDa": "~388 kDa",
        "gene_class": "Cadherin superfamily; tip-link upper strand of hair cell stereocilia; photoreceptor ribbon synapse",
        "locus": "10q22.1",
        "omim_gene": 605516,
        "omim_disease": 601067,
        "phenotype": "Usher Syndrome Type 1D / DFNB12 — profound SNHL + RP + severe vestibular dysfunction",
        "disease": (
            "CDH23 encodes Cadherin-23, a large calcium-dependent cell adhesion molecule essential "
            "for the tip-link of cochlear and vestibular hair cell stereocilia. "
            "NORMAL FUNCTION in cochlea: Cadherin-23 forms the UPPER STRAND of the tip-link — "
            "the fine extracellular filament connecting the tip of a shorter stereocilium to the "
            "side of the adjacent taller stereocilium. Tip-links gate mechanosensory transduction "
            "channels at the stereocilia tip: stereocilia deflection → tip-link tension → "
            "channel opening → K+/Ca2+ influx → hair cell depolarisation → sound transduction. "
            "CDH23 also forms lateral links maintaining stereocilia bundle cohesion. "
            "NORMAL FUNCTION in retina: Cadherin-23 localises to the photoreceptor ribbon "
            "synapse, where it is required for the structural integrity of the synaptic complex "
            "between photoreceptors and bipolar/horizontal cells. "
            "PATHOMECHANISM: biallelic severe LOF variants → absent tip-link → "
            "mechanotransduction failure → profound congenital SNHL + absent vestibular function "
            "(Usher type 1 phenotype); "
            "HYPOMORPHIC variants (residual CDH23 function): DFNB12 — non-syndromic SNHL "
            "without RP, reflecting partial tip-link function preservation. "
            "VESTIBULAR: bilateral absent vestibular function → delayed sitting (>9 months) "
            "and walking (>18 months); absent Moro reflex head righting; "
            "toddlers must hold walls; cannot tandem-walk; Romberg positive. "
            "RETINA: RP with generally WORSE visual prognosis than USH2A — "
            "earlier and more severe photoreceptor degeneration in CDH23-Usher1D. "
            "ERG shows absent/severely reduced rod and cone responses in teens-twenties. "
            "MANAGEMENT: early CI (before age 2 years) mandatory for auditory rehabilitation; "
            "annual ophthalmology; low-vision and orientation-mobility rehabilitation; "
            "gene-panel mandatory at diagnosis."
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic; LOF = Usher 1D; hypomorphic = DFNB12; large gene (~6.5 kb coding)",
        "hallmark": "Usher type 1: profound SNHL + absent vestibular (delayed walking) + early severe RP; tip-link cadherin upper strand; worse visual prognosis than USH2A",
        "key_ddx": "MYO7A-Usher1B (tip-link motor not tip-link itself), PCDH15-Usher1F (ankle-link), USH2A (normal vestibular, later RP), DFNB12 (CDH23 hypomorphic, no RP)",
        "founder_variant": "Diverse; p.R1746Q noted in some populations; no single dominant European founder",
        "onset_pattern": "Congenital (SNHL + vestibular); RP onset teens; earlier/more severe than USH2A",
        "seed": 1131,
        "cohort_n": 40,
        "severity_weights": {"Profound": 0.90, "Severe": 0.08, "Moderate": 0.02},
        "has_rp_rate": 1.0,
        "has_vestibular_rate": 1.0,
        "ci_rate": 0.85,
        "ha_rate": 0.10,
        "ansd_rate": 0.0,
        "congenital_rate": 1.0,
        "nbs_detected_rate": 0.88,
        "usher_type": "1",
        "ci_recommended": True,
        "ansd_gene": False,
        "aminoglycoside_risk": "ABSOLUTE AVOID — aminoglycosides destroy OHCs adding to total cochlear damage; tip-link restoration by CI is compromised by OHC loss",
        "loop_diuretic_risk": "HIGH RISK — ototoxic synergy; avoid ethacrynic acid and furosemide; use thiazide if diuresis required",
        "vpa_risk": "VPA retinal toxicity: theoretically additive to RP; flag in prescribing; not primary concern",
    },

    # ── PCDH15 — Usher Syndrome Type 1F / DFNB23 ────────────────────────────────
    {
        "gene": "PCDH15",
        "protein": "Protocadherin-15 (PCDH15)",
        "alias": "PCDH15; OMIM gene 605514; 10q21.1; ~1955 aa; Usher 1F (OMIM #602083) / DFNB23 (OMIM #609533); AR; ankle-link cadherin; deafness + RP + vestibular dysfunction (Usher type 1); R245X Ashkenazi founder; DFNB23 (no RP) from hypomorphic alleles; CI recommended; profound congenital SNHL",
        "aa": "~1955 aa",
        "kDa": "~224 kDa",
        "gene_class": "Protocadherin superfamily; ankle-link lower strand of hair cell stereocilia tip-link complex; photoreceptor calyceal process",
        "locus": "10q21.1",
        "omim_gene": 605514,
        "omim_disease": 602083,
        "phenotype": "Usher Syndrome Type 1F / DFNB23 — profound SNHL + RP + vestibular dysfunction (Usher type 1)",
        "disease": (
            "PCDH15 encodes Protocadherin-15, which forms the LOWER STRAND of the tip-link "
            "complex (paired with CDH23 upper strand) in cochlear and vestibular hair cells. "
            "NORMAL FUNCTION in cochlea: PCDH15 forms homodimers at the tip of shorter "
            "stereocilia and binds CDH23 at the side of the adjacent taller stereocilium, "
            "assembling the tip-link. Both proteins are required simultaneously — "
            "loss of either PCDH15 or CDH23 abolishes tip-link and mechanotransduction. "
            "PCDH15 is also essential for the ankle-link complex (with USH2A/Usherin and VLGR1), "
            "making it critical at multiple points in stereocilia bundle architecture. "
            "NORMAL FUNCTION in retina: PCDH15 localises to the photoreceptor calyceal processes "
            "and ribbon synapse — required for photoreceptor outer segment structural integrity. "
            "PATHOMECHANISM: biallelic severe LOF → absent PCDH15 → tip-link + ankle-link "
            "failure → profound congenital SNHL + absent vestibular function (Usher type 1); "
            "progressive RP (photoreceptor degeneration) onset teens-twenties. "
            "HYPOMORPHIC ALLELES: some PCDH15 missense alleles retain partial function → "
            "DFNB23: non-syndromic profound SNHL without RP → reveals PCDH15's critical "
            "cochlear role even when retinal disease is absent. "
            "ASHKENAZI FOUNDER VARIANT: p.R245X (c.733C>T) — common in Ashkenazi Jewish "
            "populations; approximately 1:120 Ashkenazi carrier rate for this variant. "
            "Mandatory screening in Ashkenazi Jewish couples undergoing reproductive counselling. "
            "MANAGEMENT: early CI (before age 2) for Usher 1F; annual ophthalmology from "
            "childhood; vestibular rehabilitation; low-vision aids and mobility training."
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic; R245X Ashkenazi founder (1:120 carrier); hypomorphic alleles → DFNB23 (no RP)",
        "hallmark": "Usher type 1: profound SNHL + absent vestibular + RP; ankle-link lower strand; R245X Ashkenazi founder; DFNB23 = hypomorphic alleles without RP",
        "key_ddx": "CDH23-Usher1D (tip-link upper strand, similar phenotype), MYO7A-Usher1B, DFNB23 (PCDH15 hypomorphic, no RP), GJB2 (no RP, no vestibular)",
        "founder_variant": "p.R245X (c.733C>T) Ashkenazi Jewish founder (~1:120 carrier rate); p.L1825P in other populations",
        "onset_pattern": "Congenital (SNHL + vestibular); RP onset teens; DFNB23 (hypomorphic) = congenital SNHL, no RP",
        "seed": 1132,
        "cohort_n": 40,
        "severity_weights": {"Profound": 0.88, "Severe": 0.10, "Moderate": 0.02},
        "has_rp_rate": 0.85,  # ~85% full Usher 1F; 15% DFNB23 hypomorphic (no RP)
        "has_vestibular_rate": 0.85,
        "ci_rate": 0.82,
        "ha_rate": 0.12,
        "ansd_rate": 0.0,
        "congenital_rate": 1.0,
        "nbs_detected_rate": 0.87,
        "usher_type": "1",
        "ci_recommended": True,
        "ansd_gene": False,
        "aminoglycoside_risk": "ABSOLUTE AVOID — cochlear amplifier OHC destruction in PCDH15/Usher worsens irrecoverable SNHL; CI outcomes impaired with prior OHC loss",
        "loop_diuretic_risk": "HIGH RISK — ototoxic synergy with pre-existing SNHL; furosemide contraindicated in Usher if alternatives exist",
        "vpa_risk": "VPA retinal toxicity additive to RP — document discussion with ophthalmology; not primary concern",
    },

    # ── CLRN1 — Usher Syndrome Type 3A ─────────────────────────────────────────
    {
        "gene": "CLRN1",
        "protein": "Clarin-1 (CLRN1)",
        "alias": "CLRN1; OMIM gene 606397; 3q25.1; ~232 aa; Usher syndrome type 3A (OMIM #276902); AR; PROGRESSIVE postlingual SNHL (not congenital-profound) + RP + variable vestibular; p.Asn48Lys Finnish founder (~50% Finnish Usher 3); HA work initially; CI when severe; annual audiology + retinal monitoring mandatory; gene therapy trials",
        "aa": "~232 aa",
        "kDa": "~26 kDa",
        "gene_class": "Clarin family; 4-transmembrane protein; stereocilia and photoreceptor ribbon synapse maintenance",
        "locus": "3q25.1",
        "omim_gene": 606397,
        "omim_disease": 276902,
        "phenotype": "Usher Syndrome Type 3A — PROGRESSIVE SNHL (postlingual) + RP + variable vestibular dysfunction",
        "disease": (
            "CLRN1 encodes Clarin-1, a 4-transmembrane tetraspanin-related protein expressed in "
            "cochlear hair cells and photoreceptors. "
            "NORMAL FUNCTION in cochlea: Clarin-1 localises to the stereocilia bundle and the "
            "hair cell ribbon synapse region. It is required for maintenance of the stereocilia "
            "bundle integrity and the organisation of synaptic scaffolding proteins at the IHC "
            "ribbon synapse. Clarin-1 interacts with harmonin and whirlin (other Usher proteins). "
            "NORMAL FUNCTION in retina: Clarin-1 is expressed in photoreceptors, particularly at "
            "the calyceal processes and ribbon synaptic terminals. Required for long-term "
            "photoreceptor survival. "
            "PATHOMECHANISM: biallelic LOF variants → absent Clarin-1 → "
            "PROGRESSIVE DETERIORATION (not congenital-profound as in Usher types 1 and 2): "
            "HEARING: postlingual progressive SNHL — patients are born with SOME useful hearing "
            "(often in the mild-moderate range); hearing deteriorates progressively through "
            "childhood, adolescence, and adulthood → eventually severe-profound. "
            "Hearing aids work initially and provide significant benefit (unlike Usher type 1 "
            "where profound deafness is congenital). CI is appropriate when HA inadequate. "
            "VESTIBULAR: VARIABLE (not uniformly absent as in Usher 1): "
            "some patients have progressive vestibular dysfunction; some have normal vestibular. "
            "RETINA: RP with progressive rod then cone photoreceptor degeneration; "
            "onset from teens; ERG shows progressive deterioration. "
            "UNIQUE FEATURE: progressive nature means NBS may miss CLRN1 in early childhood "
            "(hearing may pass initial newborn screen); diagnosis often delayed until progressive "
            "pattern recognised. "
            "FINNISH FOUNDER: p.Asn48Lys (c.144T>A) — ~50% of Finnish Usher syndrome type 3 "
            "alleles; elevated frequency in Finnish population (Finno-Ugric founder effect). "
            "GENE THERAPY: CLRN1-targeted AAV gene therapy in preclinical and early clinical "
            "development (2023-2024) — most promising Usher gene therapy target after OTOF. "
            "MONITORING SCHEDULE: audiogram every 6-12 months (progressive — track rate of "
            "decline); annual ERG and visual field testing mandatory."
        ),
        "inheritance": "Autosomal Recessive (AR); biallelic; p.Asn48Lys Finnish founder; Ashkenazi founder p.Y176X also reported",
        "hallmark": "PROGRESSIVE postlingual SNHL (passes NBS initially); RP + variable vestibular; p.Asn48Lys Finnish founder; HA effective initially; CI when severe; gene therapy trials",
        "key_ddx": "USH2A (SNHL congenital, normal vestibular, RP onset similar timing), Usher 1 (congenital profound), age-related HL + coincidental RP, DFNB1A (no RP)",
        "founder_variant": "p.Asn48Lys (c.144T>A) — Finnish founder (~50% Finnish Usher 3 alleles); p.Y176X Ashkenazi",
        "onset_pattern": "Postlingual progressive SNHL (not congenital-profound); RP onset teens-twenties; variable vestibular",
        "seed": 1133,
        "cohort_n": 40,
        "severity_weights": {"Moderate": 0.35, "Severe": 0.40, "Profound": 0.25},
        "has_rp_rate": 1.0,
        "has_vestibular_rate": 0.50,  # variable vestibular involvement
        "ci_rate": 0.45,  # CI when HA insufficient; progressive course
        "ha_rate": 0.75,  # HA effective in early-moderate stages
        "ansd_rate": 0.0,
        "congenital_rate": 0.0,  # NOT congenital-profound; progressive postlingual
        "nbs_detected_rate": 0.40,  # often PASSES NBS due to progressive nature
        "usher_type": "3",
        "ci_recommended": True,  # when HA insufficient
        "ansd_gene": False,
        "aminoglycoside_risk": "ABSOLUTE AVOID — aminoglycosides accelerate progressive cochlear degeneration; hasten need for CI; single course may cause irreversible severe step-down in hearing",
        "loop_diuretic_risk": "HIGH RISK — potentiates cochlear toxicity in progressive SNHL; monitor audiogram closely if furosemide cannot be avoided",
        "vpa_risk": "VPA retinal toxicity: additive concern with RP; discuss with ophthalmology; not primary prescribing contraindication but document risk discussion",
    },
]


# ── Cohort generation ──────────────────────────────────────────────────────────

def _gen_patients_for_gene(gene_data: dict, seed: int) -> list:
    """Generate 40 deterministic synthetic patients for one gene."""
    rng = random.Random(seed)
    patients = []
    for i in range(gene_data["cohort_n"]):
        # Severity
        sev_choices = list(gene_data["severity_weights"].keys())
        sev_weights = list(gene_data["severity_weights"].values())
        sev = rng.choices(sev_choices, weights=sev_weights, k=1)[0]

        # Clinical features
        has_rp = rng.random() < gene_data["has_rp_rate"]
        has_vest = rng.random() < gene_data["has_vestibular_rate"]
        ci_performed = rng.random() < gene_data["ci_rate"]
        hearing_aid = rng.random() < gene_data["ha_rate"]
        ansd = gene_data["ansd_gene"]
        congenital = rng.random() < gene_data["congenital_rate"]
        nbs_detected = rng.random() < gene_data["nbs_detected_rate"]

        # Age at diagnosis (months)
        if congenital:
            onset_months = rng.randint(0, 6)
            dx_delay = rng.randint(0, 18)
        else:
            # Progressive (CLRN1): detected later in childhood
            onset_months = rng.randint(24, 72)
            dx_delay = rng.randint(6, 48)

        sex = rng.choice(["M", "F"])

        # Drug exposure errors (aminoglycoside/loop diuretic prescribed despite CI)
        drug_error = rng.random() < 0.08

        patients.append({
            "gene": gene_data["gene"],
            "severity": sev,
            "has_rp": has_rp,
            "has_vestibular_dysfunction": has_vest,
            "ci_performed": ci_performed,
            "hearing_aid": hearing_aid,
            "ansd": ansd,
            "congenital_onset": congenital,
            "nbs_detected": nbs_detected,
            "onset_age_months": onset_months,
            "diagnosis_age_months": onset_months + dx_delay,
            "sex": sex,
            "drug_error": drug_error,
            "usher_type": gene_data["usher_type"],
        })
    return patients


def _gen_cohort() -> list:
    """Generate all 320 patients (8 genes × 40) deterministically."""
    all_pts = []
    for idx, gd in enumerate(DEAFNESS_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients_for_gene(gd, seed))
    return all_pts


# ── API functions ──────────────────────────────────────────────────────────────

def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    # Severity breakdown
    sev = {"Profound": 0, "Severe": 0, "Moderate": 0, "Mild": 0}
    for p in patients:
        sev[p["severity"]] = sev.get(p["severity"], 0) + 1

    # Aggregate clinical statistics
    rp_n = sum(1 for p in patients if p["has_rp"])
    vest_n = sum(1 for p in patients if p["has_vestibular_dysfunction"])
    ci_n = sum(1 for p in patients if p["ci_performed"])
    ha_n = sum(1 for p in patients if p["hearing_aid"])
    ansd_n = sum(1 for p in patients if p["ansd"])
    congenital_n = sum(1 for p in patients if p["congenital_onset"])
    nbs_n = sum(1 for p in patients if p["nbs_detected"])
    profound_n = sev.get("Profound", 0)
    error_n = sum(1 for p in patients if p["drug_error"])

    mean_onset = round(sum(p["onset_age_months"] for p in patients) / n, 1)
    mean_dx = round(sum(p["diagnosis_age_months"] for p in patients) / n, 1)

    # Per-gene CI rates
    gene_ci_pct = {}
    for gd in DEAFNESS_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_ci_pct[gd["gene"]] = round(
            100 * sum(1 for p in gpts if p["ci_performed"]) / len(gpts), 1
        )

    # Usher type distribution
    usher1_pts = [p for p in patients if p["usher_type"] == "1"]
    usher2_pts = [p for p in patients if p["usher_type"] == "2"]
    usher3_pts = [p for p in patients if p["usher_type"] == "3"]
    non_usher_pts = [p for p in patients if p["usher_type"] is None]

    disease_cat = {
        "Non-syndromic AR SNHL (GJB2/DFNB1A)": round(100 * 40 / n, 1),
        "Pendred Syndrome / DFNB4 (SLC26A4)": round(100 * 40 / n, 1),
        "Auditory Neuropathy ANSD (OTOF/DFNB9)": round(100 * 40 / n, 1),
        "Usher Syndrome Type 1B (MYO7A)": round(100 * 40 / n, 1),
        "Usher Syndrome Type 2A (USH2A)": round(100 * 40 / n, 1),
        "Usher Syndrome Type 1D (CDH23)": round(100 * 40 / n, 1),
        "Usher Syndrome Type 1F (PCDH15)": round(100 * 40 / n, 1),
        "Usher Syndrome Type 3A (CLRN1)": round(100 * 40 / n, 1),
    }

    kpis = [
        {"label": "Total Patients", "value": str(n)},
        {"label": "Genes Covered", "value": "8"},
        {"label": "CI Performed", "value": f"{round(100*ci_n/n,1)}%"},
        {"label": "Has Retinitis Pigmentosa", "value": f"{round(100*rp_n/n,1)}%"},
        {"label": "Usher Syndrome (any type)", "value": f"{round(100*len(usher1_pts+usher2_pts+usher3_pts)/n,1)}%"},
        {"label": "ANSD (OTOF)", "value": f"{round(100*ansd_n/n,1)}%"},
        {"label": "Congenital Onset", "value": f"{round(100*congenital_n/n,1)}%"},
        {"label": "NBS Detected", "value": f"{round(100*nbs_n/n,1)}%"},
    ]

    return {
        "atlas_name": "Deafness-Atlas",
        "atlas_subtitle": (
            "GJB2·SLC26A4·OTOF·MYO7A·USH2A·CDH23·PCDH15·CLRN1 — "
            "320 patients (8×40, seeds 1126–1133)"
        ),
        "n_genes": 8,
        "n_patients": n,
        "seeds": "1126–1133",
        "description": (
            "Comprehensive atlas of 8 major hereditary deafness and Usher syndrome genes: "
            "GJB2/DFNB1A (AR; Connexin 26; most common AR SNHL; p.35delG European founder; "
            "CI curative; carrier 1:33 Europeans; no RP, no vestibular); "
            "SLC26A4/Pendred (AR; EVA PATHOGNOMONIC on CT temporal bone; progressive/fluctuating SNHL; "
            "thyroid goiter 70%; avoid head trauma/barotrauma; CI effective); "
            "OTOF/ANSD (AR; Normal OAE + absent ABR = ANSD PATHOGNOMONIC; "
            "hearing aid FAILS; CI curative; gene therapy trials 2023-24); "
            "MYO7A/Usher1B (AR; profound congenital SNHL + absent vestibular + RP; "
            "night blindness first; early CI; annual retinal monitoring); "
            "USH2A/Usher2A (AR; most common Usher ~50%; normal vestibular; moderate-severe SNHL; "
            "HA initially; ERG diagnostic; p.Glu767Ser European founder); "
            "CDH23/Usher1D (AR; tip-link upper strand; profound SNHL + RP + absent vestibular; "
            "worse visual prognosis than USH2A); "
            "PCDH15/Usher1F (AR; ankle-link lower strand; R245X Ashkenazi founder; "
            "DFNB23 = hypomorphic no RP; CI recommended); "
            "CLRN1/Usher3A (AR; PROGRESSIVE postlingual SNHL + RP + variable vestibular; "
            "p.Asn48Lys Finnish founder; HA initially; CI when severe; audiogram every 6-12 months). "
            "ALL 8 genes: aminoglycosides ABSOLUTELY CONTRAINDICATED — additional cochlear toxicity; "
            "loop diuretics HIGH RISK; Usher syndrome requires gene panel + ERG + vestibular testing "
            "at diagnosis; driving assessment mandatory with RP."
        ),
        "aggregate_clinical": {
            "snhl_pct": 100.0,
            "profound_snhl_pct": round(100 * profound_n / n, 1),
            "rp_pct": round(100 * rp_n / n, 1),
            "vestibular_dysfunction_pct": round(100 * vest_n / n, 1),
            "ci_performed_pct": round(100 * ci_n / n, 1),
            "ansd_pct": round(100 * ansd_n / n, 1),
            "congenital_onset_pct": round(100 * congenital_n / n, 1),
            "nbs_detected_pct": round(100 * nbs_n / n, 1),
            "hearing_aid_pct": round(100 * ha_n / n, 1),
            "drug_error_pct": round(100 * error_n / n, 1),
        },
        "drug_contraindications": {
            "aminoglycosides": (
                "ABSOLUTE AVOID in ALL 8 genes — gentamicin, tobramycin, amikacin, streptomycin, "
                "neomycin cause additional irreversible sensorineural cochlear toxicity targeting "
                "outer hair cells. In patients with pre-existing hereditary SNHL, aminoglycosides "
                "accelerate and worsen hearing loss, often precipitating a step-down to profound. "
                "In OTOF-ANSD: OHC loss worsens CI candidacy and outcomes. "
                "ALWAYS use alternative antibiotics: beta-lactams, cephalosporins, fluoroquinolones "
                "(ear drops topical only — not systemic). If aminoglycoside absolutely essential "
                "(TB, Gram-negative sepsis), once-daily dosing + TDM + serial audiometry mandatory."
            ),
            "loop_diuretics": (
                "HIGH RISK — furosemide, ethacrynic acid (most ototoxic loop diuretic), "
                "bumetanide cause dose-dependent endocochlear potential disruption, potentiating "
                "SNHL. Especially dangerous in GJB2 (potassium recycling disruption) and "
                "SLC26A4 (EVA + endolymph volume dysregulation). If loop diuretic essential: "
                "lowest effective dose; avoid concomitant aminoglycoside; monitor audiogram."
            ),
            "cisplatin_caution": (
                "Cisplatin and carboplatin cause cumulative cochlear toxicity. "
                "In patients with hereditary SNHL, platinum-based chemotherapy accelerates "
                "hearing loss beyond the baseline. Discuss oncology/audiology cochlear monitoring "
                "plan; consider carboplatin over cisplatin where equivalent oncologic efficacy."
            ),
        },
        "usher_types": {
            "type_1": ["MYO7A", "CDH23", "PCDH15"],
            "type_2": ["USH2A"],
            "type_3": ["CLRN1"],
        },
        "key_rules": {
            "ci_rule": (
                "CI is curative/rehabilitative in GJB2, SLC26A4, OTOF (best outcomes in ANSD), "
                "and all Usher type 1 genes (MYO7A, CDH23, PCDH15) — cochlear anatomy usually "
                "normal in all 8 genes (no cochlear aplasia). CI should be offered early "
                "(before age 2 years) in all profound congenital SNHL. "
                "In OTOF-ANSD: CI is the ONLY effective intervention — hearing aids fail. "
                "In GJB2 bilateral profound: CI outcomes are among the best of all aetiologies."
            ),
            "ansd_rule": (
                "ANSD diagnostic pattern: Normal OAEs (outer hair cells intact) + "
                "Absent or severely abnormal ABR (inner hair cell synapse or nerve conduction failure). "
                "This pattern is PATHOGNOMONIC for ANSD. "
                "In OTOF-ANSD: OAEs normal because OHCs do not require otoferlin. "
                "Hearing aid DOES NOT WORK — amplification cannot restore synchrony at the "
                "IHC ribbon synapse. CI restores synchrony by electrical stimulation. "
                "ALWAYS perform both OAE and ABR in newborn hearing screening to detect ANSD."
            ),
            "usher_monitoring": (
                "ALL Usher syndrome patients (MYO7A/CDH23/PCDH15/USH2A/CLRN1): "
                "Gene panel + ERG + vestibular caloric testing at diagnosis. "
                "Annual ophthalmology with ERG and visual field testing from age 5-8 years. "
                "Driving assessment mandatory when RP diagnosed — night driving must CEASE first; "
                "driving licence review when central visual field <120 degrees. "
                "Low-vision rehabilitation + orientation and mobility training. "
                "Social/education support: deafblindness services when visual impairment severe. "
                "CLRN1: audiogram every 6-12 months to track progressive decline rate."
            ),
            "aminoglycoside_rule": (
                "NEVER prescribe gentamicin, tobramycin, amikacin, or streptomycin systemically "
                "to any patient with hereditary SNHL — GJB2, SLC26A4, OTOF, MYO7A, USH2A, "
                "CDH23, PCDH15, or CLRN1. "
                "Document contraindication in the medical record and allergy/alert field. "
                "Alternative antibiotics exist for all common indications. "
                "If unavoidable: once-daily dosing, serum TDM, serial audiometry."
            ),
            "eva_rule": (
                "Enlarged Vestibular Aqueduct (EVA) on CT temporal bone = SLC26A4 until "
                "proven otherwise. EVA width >1.5 mm at midpoint PATHOGNOMONIC. "
                "Management rules for EVA/SLC26A4: "
                "AVOID head trauma — contact sports, diving, high-impact activities PROHIBITED; "
                "AVOID barotrauma — aircraft cabin pressure changes, scuba diving, heavy lifting/Valsalva; "
                "NO cochlear tap/perilymph pressure testing; "
                "Monitor thyroid function (TSH, free T4) annually — 25% develop hypothyroidism. "
                "Progressive hearing loss despite precautions = CI indication."
            ),
        },
        "nbs_utility": (
            "Universal Newborn Hearing Screening (UNHS) by OAE and/or ABR (AABR) is the primary "
            "detection pathway for hereditary SNHL. NBS detects GJB2 (bilateral profound), "
            "SLC26A4, MYO7A/CDH23/PCDH15 (all profound congenital), and USH2A (moderate-severe). "
            "CRITICAL LIMITATION: OTOF-ANSD may PASS OAE-only NBS (OAEs are NORMAL in ANSD) — "
            "ABR or AABR must be included to detect ANSD. Two-tier screening (OAE then AABR) "
            "is mandatory to avoid missing OTOF-ANSD. "
            "CLRN1 (Usher 3A) often PASSES NBS because hearing is not profound at birth — "
            "progressive pattern emerges later; genetic testing indicated in any child with "
            "progressive SNHL diagnosed after initial normal NBS. "
            "Refer all NBS-failed infants to paediatric audiology and genetics within 3 months."
        ),
        "severity": sev,
        "disease_category_breakdown": disease_cat,
        "gene_ci_pct": gene_ci_pct,
        "mean_onset_age_months": mean_onset,
        "mean_dx_age_months": mean_dx,
        "kpis": kpis,
        "drug_alerts": [
            {
                "type": "danger",
                "title": "AMINOGLYCOSIDES ABSOLUTELY CONTRAINDICATED IN ALL 8 HEREDITARY DEAFNESS GENES",
                "body": (
                    "Gentamicin, tobramycin, amikacin, streptomycin, neomycin: ABSOLUTELY "
                    "CONTRAINDICATED in all GJB2, SLC26A4, OTOF, MYO7A, USH2A, CDH23, PCDH15, "
                    "CLRN1 patients. Aminoglycosides cause irreversible outer hair cell loss, "
                    "accelerating baseline SNHL to profound. In OTOF-ANSD: OHC loss worsens CI "
                    "outcomes. In Usher syndrome: retinal aminoglycoside toxicity adds to RP. "
                    "Document contraindication in all medical records. "
                    "Alternative antibiotics must be used for ALL indications."
                ),
            },
            {
                "type": "danger",
                "title": "ANSD — HEARING AID FAILS: Normal OAE + Absent ABR = CI Required (OTOF)",
                "body": (
                    "Auditory Neuropathy Spectrum Disorder (OTOF-ANSD): hearing aid amplifies "
                    "sound but CANNOT restore IHC synaptic synchrony. Prescribing a hearing aid "
                    "as the primary intervention in confirmed ANSD delays cochlear implant and "
                    "impairs speech-language outcomes. Refer for CI assessment immediately when "
                    "OAE normal + ABR absent/severely abnormal (ANSD pattern confirmed). "
                    "NBS: AABR must be performed — OAE-only NBS will MISS OTOF-ANSD."
                ),
            },
            {
                "type": "warning",
                "title": "EVA (SLC26A4/Pendred): AVOID HEAD TRAUMA — Contact Sports PROHIBITED",
                "body": (
                    "Enlarged Vestibular Aqueduct creates vulnerability to sudden SNHL "
                    "worsening from physical trauma, barotrauma, or Valsalva manoeuvres. "
                    "All SLC26A4/Pendred patients: no contact sports (rugby, boxing, football "
                    "headers, martial arts); no diving; no heavy straining; "
                    "advise parents and school clearly. A single head impact can cause sudden "
                    "irreversible step-down from moderate to profound hearing loss."
                ),
            },
            {
                "type": "warning",
                "title": "USHER SYNDROME: Driving Assessment Mandatory — Night Driving MUST CEASE",
                "body": (
                    "All Usher syndrome patients with RP (MYO7A, USH2A, CDH23, PCDH15, CLRN1): "
                    "night blindness (nyctalopia) is the FIRST symptom of RP and prohibits "
                    "night driving immediately. Visual field restriction progresses to tunnel "
                    "vision → legal blindness. "
                    "Formal driving assessment by low-vision specialist is MANDATORY at RP diagnosis. "
                    "Failure to advise cessation of night driving = medico-legal liability."
                ),
            },
        ],
        "critical_rules": [
            "NORMAL OAE + ABSENT ABR = ANSD: refer for CI assessment — hearing aid will NOT work (OTOF)",
            "EVA on CT temporal bone: SLC26A4 until proven otherwise; contact sports PROHIBITED; avoid barotrauma",
            "GJB2 bilateral profound SNHL: CI most effective genetic deafness — refer by age 12 months",
            "USHER TYPE 1 (MYO7A/CDH23/PCDH15): absent vestibular → delayed walking; early CI before age 2; annual ophthalmology",
            "USH2A: ERG diagnostic BEFORE fundoscopy shows RP changes; night driving MUST CEASE at RP diagnosis",
            "CLRN1: audiogram every 6-12 months; PASSES NBS — progressive SNHL pattern → genetic testing mandatory",
            "AMINOGLYCOSIDES ABSOLUTELY CI IN ALL 8 GENES: document in medical record as drug contraindication",
            "ALL USHER: gene-panel + ERG + vestibular caloric testing at diagnosis — do not rely on audiogram alone",
        ],
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    genes_out = []
    for gd in DEAFNESS_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        n = len(gpts)

        snhl_pct = 100.0  # all patients have SNHL by definition
        profound_pct = round(100 * sum(1 for p in gpts if p["severity"] == "Profound") / n, 1)
        rp_pct = round(100 * sum(1 for p in gpts if p["has_rp"]) / n, 1)
        vest_pct = round(100 * sum(1 for p in gpts if p["has_vestibular_dysfunction"]) / n, 1)
        ci_pct = round(100 * sum(1 for p in gpts if p["ci_performed"]) / n, 1)
        ha_pct = round(100 * sum(1 for p in gpts if p["hearing_aid"]) / n, 1)
        ansd_pct = round(100 * sum(1 for p in gpts if p["ansd"]) / n, 1)
        congenital_pct = round(100 * sum(1 for p in gpts if p["congenital_onset"]) / n, 1)
        progressive_pct = round(100 - congenital_pct, 1)
        nbs_pct = round(100 * sum(1 for p in gpts if p["nbs_detected"]) / n, 1)

        mean_onset = round(sum(p["onset_age_months"] for p in gpts) / n, 1)

        genes_out.append({
            "gene": gd["gene"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "gene_class": gd["gene_class"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "phenotype": gd["phenotype"],
            "disease": gd["disease"],
            "inheritance": gd["inheritance"],
            "hallmark": gd["hallmark"],
            "key_ddx": gd["key_ddx"],
            "founder_variant": gd["founder_variant"],
            "onset_pattern": gd["onset_pattern"],
            "seed": gd["seed"],
            "cohort_n": n,
            "mean_onset_age_months": mean_onset,
            "snhl_pct": snhl_pct,
            "profound_snhl_pct": profound_pct,
            "rp_pct": rp_pct,
            "vestibular_dysfunction_pct": vest_pct,
            "ci_performed_pct": ci_pct,
            "hearing_aid_pct": ha_pct,
            "ansd_pct": ansd_pct,
            "congenital_onset_pct": congenital_pct,
            "progressive_pct": progressive_pct,
            "nbs_detected_pct": nbs_pct,
            "ci_recommended": gd["ci_recommended"],
            "ansd_gene": gd["ansd_gene"],
            "usher_type": gd["usher_type"],
            "aminoglycoside_risk": gd["aminoglycoside_risk"],
            "loop_diuretic_risk": gd["loop_diuretic_risk"],
            "vpa_risk": gd["vpa_risk"],
            "severity_weights": gd["severity_weights"],
        })

    return {"genes": genes_out}


def get_definitions() -> list:
    return [
        {
            "term": "SNHL (Sensorineural Hearing Loss)",
            "definition": (
                "Hearing loss due to dysfunction of the cochlear sensory cells (inner and/or "
                "outer hair cells) or the auditory nerve, as opposed to conductive hearing loss "
                "(middle ear or external ear pathology). SNHL is characterised by air-bone gap "
                "absent on pure-tone audiometry. Hereditary SNHL accounts for ~60% of congenital "
                "deafness; GJB2 is the most common AR cause. Severity graded by pure-tone average: "
                "mild (26-40 dB HL), moderate (41-55 dB HL), moderately severe (56-70 dB HL), "
                "severe (71-90 dB HL), profound (>90 dB HL). Profound SNHL = deaf; CI candidacy."
            ),
        },
        {
            "term": "Usher Syndrome",
            "definition": (
                "Autosomal recessive syndrome combining sensorineural hearing loss with "
                "retinitis pigmentosa (RP), the leading genetic cause of deafblindness. "
                "Three clinical types: "
                "Type 1 (MYO7A, CDH23, PCDH15, CIB2, USH1C, SANS): congenital profound SNHL + "
                "absent vestibular function (delayed walking) + RP onset teens → Usher type 1 is most severe. "
                "Type 2 (USH2A, ADGRV1, WHRN): congenital moderate-severe SNHL + NORMAL vestibular + "
                "RP onset teens-twenties; most common Usher type. "
                "Type 3 (CLRN1): PROGRESSIVE postlingual SNHL + RP + variable vestibular; "
                "Finnish founder variant p.Asn48Lys. "
                "Combined prevalence ~1:6,000-25,000. All Usher types: gene panel + ERG + vestibular "
                "function testing mandatory at diagnosis."
            ),
        },
        {
            "term": "ANSD (Auditory Neuropathy Spectrum Disorder)",
            "definition": (
                "Hearing disorder characterised by abnormal auditory nerve synchrony despite "
                "intact outer hair cell (OHC) function. Diagnostic hallmark: "
                "Present/normal OAEs (OHC-mediated cochlear amplification intact) + "
                "absent or severely abnormal ABR (IHC → nerve transmission or synchrony failure). "
                "Causes: IHC synaptic failure (OTOF, CABP2), auditory nerve dysfunction, "
                "neonatal hypoxia (bilateral), hyperbilirubinaemia. "
                "CRITICAL: hearing aids amplify sound but CANNOT restore neural synchrony → "
                "hearing aid provides minimal benefit in most ANSD. "
                "Cochlear implant CURATIVE in OTOF-ANSD — electrical stimulation bypasses "
                "IHC synapse and drives cochlear nerve directly."
            ),
        },
        {
            "term": "OAE (Otoacoustic Emissions)",
            "definition": (
                "Low-level sounds generated by the electromotility of outer hair cells (OHCs) "
                "in the cochlea, detectable in the external ear canal using a sensitive microphone. "
                "Present in normal-hearing ears; absent when OHC function is damaged "
                "(noise, ototoxic drugs, most SNHL). "
                "Used in Newborn Hearing Screening (OAE test): pass = OHCs functional. "
                "CRITICAL LIMITATION: OAEs are PRESENT/NORMAL in ANSD (e.g. OTOF-DFNB9) — "
                "OAE-only NBS will MISS OTOF-ANSD. Must combine with AABR to detect ANSD. "
                "Transient evoked OAE (TEOAE) and distortion product OAE (DPOAE) are standard tests."
            ),
        },
        {
            "term": "ABR (Auditory Brainstem Response)",
            "definition": (
                "Electrophysiological test recording electrical potentials from the auditory nerve "
                "and brainstem in response to click or tone-burst stimuli, using scalp electrodes. "
                "Waves I-V correspond to cochlear nerve, cochlear nucleus, superior olive, "
                "lateral lemniscus, and inferior colliculus. "
                "NORMAL ABR: waves I-V present, latencies and amplitudes within normal limits. "
                "ABSENT ABR + NORMAL OAE = ANSD pattern (PATHOGNOMONIC). "
                "AABR (automated ABR): used in newborn screening — detects ANSD that OAE misses. "
                "ABR also used for threshold estimation in infants who cannot perform behavioural audiometry."
            ),
        },
        {
            "term": "Cochlear Implant (CI)",
            "definition": (
                "Electronic device surgically implanted into the cochlea that directly electrically "
                "stimulates the cochlear (auditory) nerve, bypassing damaged hair cells. "
                "Consists of external processor + internal receiver-stimulator + electrode array "
                "inserted into the scala tympani. "
                "BEST OUTCOMES: GJB2-DFNB1A, OTOF-ANSD, Usher syndrome type 1 (early CI). "
                "CANDIDACY: bilateral profound SNHL (>90 dB HL) with little benefit from HA; "
                "also bilateral severe SNHL with poor speech discrimination despite HA. "
                "TIMING: earlier = better outcomes (critical period for auditory cortex plasticity); "
                "aim <12 months in congenital profound SNHL; best results before age 2. "
                "Contraindication: cochlear aplasia (Michel deformity), absent cochlear nerve."
            ),
        },
        {
            "term": "EVA (Enlarged Vestibular Aqueduct)",
            "definition": (
                "Bony enlargement of the vestibular aqueduct (VA), the bony canal connecting "
                "the inner ear to the posterior cranial fossa, containing the endolymphatic duct. "
                "Diagnostic threshold: VA width >1.5 mm at midpoint or >2 mm at the operculum "
                "on high-resolution CT temporal bone. "
                "PATHOGNOMONIC for SLC26A4/Pendred syndrome/DFNB4 when bilateral. "
                "Clinical consequence: endolymph volume and pressure dysregulation → "
                "progressive/fluctuating SNHL; acute SNHL episodes with head trauma or Valsalva. "
                "Management: avoid all head trauma, contact sports, barotrauma, and Valsalva."
            ),
        },
        {
            "term": "Retinitis Pigmentosa (RP)",
            "definition": (
                "Progressive hereditary retinal degeneration affecting rod photoreceptors first "
                "(peripheral retina), then cones (central retina). "
                "Clinical features: "
                "Night blindness (nyctalopia) — first symptom, often in teens. "
                "Progressive tunnel vision (constricted visual field). "
                "Fundoscopy: bone-spicule pigmentation, arteriolar attenuation, disc pallor. "
                "ERG: reduced/absent scotopic (rod) responses first, then photopic (cone) responses. "
                "In Usher syndrome: RP combined with SNHL → deafblindness. "
                "NO curative treatment (2024); investigational: gene therapy (RPE65 approved for "
                "Leber congenital amaurosis); vitamin A palmitate may slow progression in some RP forms."
            ),
        },
        {
            "term": "ERG (Electroretinogram)",
            "definition": (
                "Electrophysiological test recording the electrical response of the retina to "
                "standardised light stimuli, using a contact lens electrode on the cornea. "
                "Scotopic ERG: measures rod photoreceptor function (dark-adapted). "
                "Photopic ERG: measures cone photoreceptor function (light-adapted). "
                "In Usher syndrome / RP: scotopic ERG reduced/absent BEFORE fundoscopic signs appear — "
                "ERG is the MOST SENSITIVE diagnostic test for early RP. "
                "ERG must be performed at diagnosis in ALL Usher syndrome patients, "
                "even before fundoscopy shows changes. "
                "Electrooculogram (EOG): measures RPE function; abnormal in Best vitelliform macular dystrophy."
            ),
        },
        {
            "term": "Connexin 26 / GJB2",
            "definition": (
                "Gap junction beta-2 protein encoded by GJB2 at 13q12.11. "
                "Forms hexameric hemichannels (connexons) between cochlear supporting cells, "
                "enabling potassium (K+) recycling from perilymph back to endolymph — "
                "critical for maintaining endocochlear potential (+80 mV) required for hair cell "
                "mechanotransduction. "
                "Most common AR cause of non-syndromic SNHL worldwide. "
                "Founder variants: p.35delG (c.35delG) in Europeans (>40% alleles); "
                "p.167delT in Ashkenazi Jews; p.235delC in East Asians. "
                "Carrier rate: 1:33 in Europeans. "
                "Phenotype: bilateral, often profound, non-syndromic SNHL; no vestibular, no RP."
            ),
        },
        {
            "term": "Otoferlin / OTOF",
            "definition": (
                "Large C2-domain calcium-sensor protein encoded by OTOF at 2p23.3 (~1997 aa). "
                "Expressed specifically in inner hair cells (IHCs) of the cochlea. "
                "Essential for calcium-triggered synaptic vesicle exocytosis at the IHC ribbon "
                "synapse — the key step in converting sound-induced IHC depolarisation into "
                "glutamate release → activation of spiral ganglion neurons → ABR generation. "
                "Outer hair cells do NOT require otoferlin → OAEs preserved in OTOF-ANSD. "
                "Biallelic LOF → DFNB9 / Auditory Neuropathy Spectrum Disorder (ANSD). "
                "Hearing aid fails (cannot restore synaptic synchrony). CI curative. "
                "Gene therapy (AAV-OTOF) in clinical trials 2023-24 with dramatic results."
            ),
        },
        {
            "term": "Tip-Link (CDH23 / PCDH15)",
            "definition": (
                "Fine extracellular filament connecting the tip of a shorter stereocilium to the "
                "side of the adjacent taller stereocilium in cochlear and vestibular hair cells. "
                "Composition: upper strand = Cadherin-23 (CDH23) homodimer; "
                "lower strand = Protocadherin-15 (PCDH15) homodimer; "
                "the two strands form a CDH23-PCDH15 heterotetrameric complex. "
                "Function: tip-links gate mechanosensory transduction channels (MET channels, "
                "TMC1/TMC2) at the tips of stereocilia — deflection → tip-link tension → "
                "MET channel opening → K+/Ca2+ influx → hair cell depolarisation → sound transduction. "
                "Loss of CDH23 (upper strand) OR PCDH15 (lower strand) → tip-link absent → "
                "mechanotransduction failure → profound congenital SNHL + absent vestibular (Usher type 1)."
            ),
        },
    ]


# ── Module self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=== Deafness Atlas — Self-test ===")

    overview = get_overview()
    print(f"\nAtlas: {overview['atlas_name']}")
    print(f"Patients: {overview['n_patients']}")
    print(f"Seeds: {overview['seeds']}")
    print(f"Aggregate clinical:")
    for k, v in overview["aggregate_clinical"].items():
        print(f"  {k}: {v}%")

    print(f"\nKPIs:")
    for kpi in overview["kpis"]:
        print(f"  {kpi['label']}: {kpi['value']}")

    breakdown = get_breakdown()
    print(f"\nBreakdown — {len(breakdown['genes'])} genes:")
    for g in breakdown["genes"]:
        print(
            f"  {g['gene']:8s} | n={g['cohort_n']} | profound={g['profound_snhl_pct']}% "
            f"| RP={g['rp_pct']}% | vest={g['vestibular_dysfunction_pct']}% "
            f"| CI={g['ci_performed_pct']}% | ANSD={g['ansd_pct']}% "
            f"| Usher={g['usher_type']}"
        )

    defs = get_definitions()
    print(f"\nDefinitions: {len(defs)} terms")
    for d in defs:
        print(f"  {d['term']}")

    print("\n=== Self-test PASSED ===")
