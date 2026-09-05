#!/usr/bin/env python3
"""Retinal Dystrophy Atlas — Complete 8-Gene Hereditary Retinal Dystrophy Atlas
RPGR    (RP3 / X-linked RP — 815 aa; Xp21.1; cilia TZ axoneme; XL;
         most common X-linked RP (~70% XLRP); ORF15 exon standard NGS MISSES;
         long-read or ORF15-specific PCR MANDATORY; AAV gene therapy MeiraGTx Beacon) ·
USH2A   (Usher syndrome type IIA — 5202 aa; 1q41; scaffolding protein;
         most common Usher gene (>50% Usher, >80% USH2); congenital SNHL + RP + normal vestibular;
         p.Glu767Sfs*21 European founder; cochlear implant EFFECTIVE) ·
ABCA4   (Stargardt disease 1 / STGD1 — 2273 aa; 1p22.1; ABC transporter;
         most common AR juvenile macular dystrophy; bisretinoid/A2E accumulation;
         VITAMIN A SUPPLEMENTS ABSOLUTELY CI — worsens A2E accumulation) ·
RDH12   (LCA13 / EOSRD — 316 aa; 14q24.1; retinal dehydrogenase;
         severe infantile-onset; ERG extinguished by age 2; no dark adaptation from birth;
         vitamin A supplementation SAFE — contrast to ABCA4) ·
PRPF31  (RP11 — 499 aa; 19q13.42; U4/U6-U5 tri-snRNP; AD haploinsufficiency;
         INCOMPLETE PENETRANCE; MLPA MANDATORY — 15% are large deletions) ·
EYS     (RP25 — 3165 aa; 6q12; spacer protein; largest non-dystrophin retinal gene (3 Mb);
         most common AR RP in Japanese (30–35%); MLPA MANDATORY — 15–20% large deletions) ·
CNGB3   (Achromatopsia ACHM3 — 809 aa; 8q21.3; CNG channel beta-3;
         most common achromatopsia gene (~50% ACHM); AAV gene therapy BIIB112/botaretigene Phase III) ·
RS1     (X-linked retinoschisis XLRS — 224 aa; Xp22.13; retinoschisin octamer;
         spoke-wheel foveal schisis PATHOGNOMONIC; electronegative ERG;
         CARBONIC ANHYDRASE INHIBITORS ABSOLUTELY CI — paradoxically worsen retinoschisis)
320-patient aggregate cohort (8 × 40, seeds 1246–1253)
"""

import random

SEED_BASE = 1246

RETINAL_GENES = [
    # ── RPGR — RP3 / X-linked RP ────────────────────────────────────────────
    {
        "gene": "RPGR",
        "protein": "RPGR (Retinitis Pigmentosa GTPase Regulator)",
        "alias": (
            "RPGR; OMIM gene 312610; Retinitis Pigmentosa 3 (RP3) #300029; Xp21.1; 815 aa; ~90 kDa; "
            "X-linked (XL); accounts for ~70% of X-linked RP (XLRP) and ~15–20% of all RP; "
            "localises to connecting cilium / transition zone axoneme; interacts with RPGRIP1; "
            "ORF15 isoform contains highly repetitive purine-rich glutamic acid–glycine region; "
            "standard NGS has high error rate at ORF15 due to sequence complexity"
        ),
        "aa": "815 aa",
        "kDa": "~90 kDa",
        "locus": "Xp21.1",
        "omim_gene": 312610,
        "omim_disease": 300029,
        "inheritance": "X-linked (XL); males affected; 30% of female carriers symptomatic",
        "gene_class": (
            "Retinitis pigmentosa GTPase regulator; two main isoforms: "
            "RPGR-constitutive (exons 1–14 + ORF15); RPGR-ORF15 (exon ORF15 appended); "
            "ORF15 exon: highly repetitive purine-rich sequence encoding glutamic acid–glycine-rich domain; "
            "sequence complexity → polymerase slippage during PCR and sequencing → standard NGS UNRELIABLE; "
            "insertions/deletions in ORF15 account for ~60–70% of all RPGR mutations; "
            "long-read sequencing (PacBio, Nanopore) or dedicated ORF15-specific PCR with Sanger MANDATORY; "
            "RPGR-ORF15 mutations → severe cone-rod involvement (male): night blindness first decade + "
            "central vision loss 4th–5th decade; RPGR-exon 1–14 mutations → cone-rod dystrophy (CRD) variant; "
            "protein localises to connecting cilium / transition zone axoneme; links photoreceptor outer segment "
            "transport to IFT machinery; interacts with RPGRIP1 (14q11.2); "
            "female carriers: 30% show symptomatic tapeto-retinal degeneration (sectoral or generalised); "
            "gene therapy: AAV2/5-RPGR trials (MeiraGTx — XLRP-RPGR; Beacon study Phase I/II); "
            "XLRP-RPGR (AAV5-RPGR) showed photoreceptor sensitivity improvement in treated eyes"
        ),
        "phenotype": (
            "Males (affected): "
            "Onset night blindness — first decade; "
            "Peripheral visual field loss (tunnel vision) — progressive 2nd–3rd decade; "
            "Central vision relatively preserved until 4th–5th decade; "
            "ERG: severely reduced rod and cone components; scotopic (rod) earlier and worse; "
            "Fundus: retinal vessel attenuation, bone-spicule pigmentation peripheral, optic disc pallor (waxy); "
            "RPGR-ORF15: more severe RP (pure rod-cone); "
            "RPGR-exon1-14: cone-rod dystrophy (CRD) — central vision affected EARLIER than in ORF15-RP; "
            "Female carriers (30% symptomatic): "
            "Sectoral retinal degeneration; asymmetric RP possible; "
            "FAF: bilateral zig-zag pattern of autofluorescence along retinal meridians (carrier female FAF sign); "
            "No systemic features (non-syndromic); "
            "Anterior segment: no cataract early; myopia common"
        ),
        "hallmark": (
            "XLRP IN A MALE — RPGR most likely (~70% of XLRP); sequence ORF15 with long-read or dedicated PCR; "
            "ORF15 STANDARD NGS UNRELIABLE — negative standard NGS does NOT rule out RPGR; "
            "FEMALE CARRIER FAF ZIG-ZAG: bilateral meridional autofluorescence pattern is PATHOGNOMONIC for RPGR female carrier; "
            "RPGR-ORF15 vs RPGR-exon1-14: exon1-14 → CRD (central earlier) vs ORF15 → classic RP (peripheral first); "
            "SEVERITY: ORF15 mutations ≈ more severe overall RP; de novo ORF15 mutations are common; "
            "GENE THERAPY ELIGIBILITY: Beacon trial (MeiraGTx AAV5-RPGR): males with RPGR-ORF15 mutation; "
            "AVOID BRIGHT LIGHT: accelerates outer segment damage in RPGR patients; "
            "DDx XLRP: RPGR (70%) vs RP2 (Xp11.23, ~10% XLRP) — RP2 is GTPase activating protein; "
            "vs CHM (choroideremia Xq21.2 — night blindness + choriocapillaris atrophy with SCALLOPED border); "
            "vs RS1 (XLRS — retinoschisis, not RP)"
        ),
        "treatment_alert": (
            "ORF15 SEQUENCING MANDATORY: standard NGS pipeline MISSES ORF15 mutations; "
            "order long-read sequencing (PacBio/Nanopore ORF15 amplicon) or Sanger-based ORF15-specific PCR; "
            "gene panel report negative for RPGR by standard sequencing is INSUFFICIENT — request ORF15-dedicated method; "
            "GENE THERAPY: AAV5-RPGR (MeiraGTx, Beacon study Phase I/II); "
            "subretinal injection; males with confirmed RPGR-ORF15 mutation; "
            "BRIGHT LIGHT AVOIDANCE: UV-blocking lenses; hat brims; avoid bright sunlight exposure; "
            "VITAMIN A: no proven benefit in RPGR-RP specifically; "
            "LOW-VISION REHABILITATION: eccentric viewing; mobility aids; cane training; "
            "GENETIC COUNSELLING: X-linked — carrier testing of female relatives; "
            "cascade testing of brothers (affected) and sisters (carrier risk 50% if mother is carrier); "
            "no male-to-male transmission (X-linked); de novo mutations in ORF15 are common; "
            "CARRIER FEMALES: ophthalmic review + ERG + FAF even if asymptomatic"
        ),
        "key_ddx": (
            "RP2-XLRP: Xp11.23; GTPase activating protein; ~10% XLRP; similar RP phenotype; gene panel differentiates; "
            "CHM-choroideremia: Xq21.2; REP1 protein; scalloped choriocapillaris atrophy border (not bone spicule); "
            "RPGR-CRD (exon1-14) vs STGD1/ABCA4: CRD is macular-first; ABCA4 FAF shows flecks + silent choroid; "
            "RS1-XLRS: retinoschisis (spoke-wheel OCT) not RP; electronegative ERG; different phenotype; "
            "Autosomal RP (PRPF31, EYS, CNGB1, PDE6A/B): no family history sex skew; panel needed"
        ),
        "vision_pattern": "Night blindness first decade; peripheral loss 2nd–3rd decade; central preserved until 4th–5th decade",
        "erg_pattern": "Severely reduced rod and cone components; scotopic (rod) first; photopic reduced later",
        "severity_weights": [0.10, 0.55, 0.35],  # mild/moderate/severe
    },

    # ── USH2A — Usher Syndrome Type IIA ──────────────────────────────────────
    {
        "gene": "USH2A",
        "protein": "Usherin (USH2A Scaffolding Protein)",
        "alias": (
            "USH2A; OMIM gene 608400; Usher Syndrome Type IIA #276901; 1q41; 5202 aa; ~580 kDa; "
            "AR; most common Usher syndrome gene (>50% of all Usher; >80% of Usher type II); "
            "scaffolding protein at photoreceptor calyceal process and hair cell ankle link complex; "
            "p.Glu767Sfs*21 (c.2299delG): European founder mutation (~40% of USH2A alleles in NW European patients)"
        ),
        "aa": "5202 aa",
        "kDa": "~580 kDa",
        "locus": "1q41",
        "omim_gene": 608400,
        "omim_disease": 276901,
        "inheritance": "AR (biallelic); high carrier frequency 1 in 100 in European populations",
        "gene_class": (
            "Type II transmembrane scaffolding protein; "
            "contains fibronectin type III repeats + laminin G + EGF-like domains; "
            "localises to: (1) photoreceptor calyceal process (periciliary membrane complex) — links outer segment to inner segment; "
            "(2) inner ear hair cell ankle link complex — structural link between adjacent stereocilia; "
            "calyceal process localisation in human photoreceptors (absent in mouse) explains why Ush2a mouse model "
            "has milder retinal phenotype than human; "
            "LOF → photoreceptor ciliary trafficking failure → progressive rod and cone degeneration (onset 2nd decade); "
            "LOF → inner ear hair cell ankle link absence → congenital sensorineural hearing loss; "
            "p.Glu767Sfs*21 (c.2299delG): frameshift → NMD → null allele; European founder; "
            "detected in ~40% of USH2A alleles in NW European cohorts; "
            "p.Cys759Phe: second most common allele; "
            "large exon deletions: 5–8% of USH2A cases — MLPA recommended if one allele found; "
            "vestibular function: NORMAL (distinguishes USH2 from USH1 — absent vestibular); "
            "Usher type I: MYO7A (most common), CDH23, PCDH15, SANS, CIB2 — congenital PROFOUND HL + ABSENT vestibular + RP; "
            "Usher type III: CLRN1 — PROGRESSIVE HL (not congenital) + RP + variable vestibular"
        ),
        "phenotype": (
            "TRIAD OF USH2A / USH2: "
            "1. RETINITIS PIGMENTOSA: onset 2nd decade (teenage years); "
            "   night blindness + peripheral field loss; bone-spicule pigmentation; ERG reduced rod first; "
            "   central vision preserved into 4th–5th decade; "
            "2. CONGENITAL SENSORINEURAL HEARING LOSS: "
            "   moderate-to-severe range (not profound); present from birth; "
            "   speech acquisition possible with hearing aids; "
            "   distinguishes USH2 from USH1 (USH1 = profound); "
            "3. NORMAL VESTIBULAR FUNCTION: "
            "   balance intact; no delayed walking; no oscillopsia from vestibular failure; "
            "   distinguishes USH2 from USH1 (USH1 = absent vestibular + delayed walking); "
            "No systemic features (renal/cardiac/pulmonary normal); "
            "Fundus: bone-spicule pigmentation; optic disc pallor; vessel attenuation; "
            "OCT: outer nuclear layer thinning; IS/OS disruption; "
            "Audiogram: mild-to-moderate slope pattern; speech frequencies affected"
        ),
        "hallmark": (
            "RP + CONGENITAL MODERATE SNHL + NORMAL VESTIBULAR = USH2 (USH2A most common); "
            "NORMAL BALANCE distinguishes USH2 from USH1 (USH1 patients have absent vestibular — delayed walking, vestibular problems); "
            "USH1 vs USH2 distinction is CRITICAL for cochlear implant counselling: "
            "USH2 has normal vestibular → CI is safe (no balance risk post-implant); "
            "USH1 also has safe CI but vestibular is already absent; "
            "p.Glu767Sfs*21 (c.2299delG) FOUNDER: screen for this first in NW European patients; "
            "COCHLEAR IMPLANT EFFECTIVE for HL component in USH2A; "
            "TIMING: CI in early childhood optimises speech/language development; "
            "SINGLE ALLELE FOUND: request MLPA (5–8% of USH2A cases have large deletions); "
            "USH3/CLRN1 DDx: progressive HL (not congenital) + later-onset RP + variable vestibular"
        ),
        "treatment_alert": (
            "COCHLEAR IMPLANT: EFFECTIVE and recommended for HL component; "
            "normal vestibular function means NO additional balance risk post-CI; "
            "early childhood CI gives best speech and language outcomes; "
            "HEARING AIDS: use until CI decision; severe HL range responds; "
            "RETINAL: no approved retinal therapy; "
            "natural history trials ongoing; gene therapy in early development; "
            "LOW-VISION AIDS: eccentric viewing training; retinal rehabilitation; "
            "vitamin A: evidence equivocal; some centres use 15,000 IU/day (not in pregnancy); "
            "AVOID excess vitamin A if ABCA4 co-suspected; "
            "p.Glu767Sfs*21 SCREENING: test this allele first in NW European patients before full panel; "
            "SINGLE ALLELE FOUND: order MLPA to detect large deletions (2nd allele); "
            "GENETIC COUNSELLING: AR; sibling risk 25%; partner carrier testing; "
            "educational support for dual sensory (vision + hearing) impairment; "
            "Deafblind rehabilitation services at advanced stages"
        ),
        "key_ddx": (
            "USH1 (MYO7A most common): PROFOUND congenital HL + ABSENT vestibular (delayed walking, vestibular hypofunction) + RP; "
            "USH3 (CLRN1-USH3A): PROGRESSIVE HL (not congenital) + RP + variable vestibular; "
            "non-syndromic RP: no HL; genetic panel; "
            "non-syndromic SNHL: no RP; "
            "Alström syndrome (ALMS1): obesity + SNHL + cone-rod dystrophy + cardiomyopathy + diabetes; "
            "Wolfram syndrome (WFS1): diabetes insipidus + DM + optic atrophy + sensorineural deafness (DIDMOAD)"
        ),
        "vision_pattern": "RP onset 2nd decade; night blindness first; peripheral field loss; central preserved 4th–5th decade",
        "erg_pattern": "Reduced rod and cone ERG; rod-first pattern; progressive scotopic and photopic reduction",
        "severity_weights": [0.20, 0.55, 0.25],  # mild/moderate/severe
    },

    # ── ABCA4 — Stargardt Disease 1 (STGD1) ──────────────────────────────────
    {
        "gene": "ABCA4",
        "protein": "ABCA4 (ATP-Binding Cassette Transporter A4 / Rim Protein / RmP)",
        "alias": (
            "ABCA4; OMIM gene 601691; Stargardt Disease 1 (STGD1) #248200 / Cone-Rod Dystrophy 3 (CORD3) #604116; "
            "1p22.1; 2273 aa; ~250 kDa; "
            "AR; most common AR juvenile macular dystrophy; prevalence 1 in 8000–10000; "
            "ATP-binding cassette transporter; flips N-retinylidene-PE (NRPE) across photoreceptor disc membrane"
        ),
        "aa": "2273 aa",
        "kDa": "~250 kDa",
        "locus": "1p22.1",
        "omim_gene": 601691,
        "omim_disease": 248200,
        "inheritance": "AR (biallelic); rare dominant alleles reported (IVD pattern)",
        "gene_class": (
            "ABC transporter subfamily A; 2 transmembrane domains + 2 nucleotide-binding domains (NBDs); "
            "photoreceptor outer segment disc membrane; "
            "FUNCTION: flips N-retinylidene-phosphatidylethanolamine (NRPE) from lumenal to cytoplasmic leaflet of disc membrane; "
            "NRPE = Schiff base adduct of all-trans retinal + phosphatidylethanolamine; "
            "LOF → NRPE accumulates in disc lumen → condenses to bisretinoid A2PE → "
            "transferred to RPE by phagocytosis → A2PE hydrolysed to A2E (toxic bisretinoid) → "
            "A2E accumulates in RPE lysosomes → lysosomal dysfunction → RPE cell death → "
            "overlying photoreceptor death → central scotoma + bull's-eye maculopathy; "
            "FUNDUS AUTOFLUORESCENCE (FAF): A2E/bisretinoid fluoresces intensely → INCREASED FAF; "
            "as RPE dies → FAF DECREASES (dark/silent areas = dead RPE); "
            "FLECKS: subretinal yellow-white deposits at level of RPE/Bruch's membrane — "
            "pisciform (fish-tail) flecks PATHOGNOMONIC for STGD1; "
            "SILENT CHOROID: on fluorescein angiography — choroidal fluorescence blocked by lipofuscin; "
            "BEATEN BRONZE macula: clinically; "
            "p.Gly1961Glu (p.G1961E): most common mild allele (late-onset, IVD = inter-individual variability pattern); "
            "p.Leu541Pro + p.Ala1038Val: digenic Spanish pair (both on same allele); "
            "VITAMIN A CI: ALL-TRANS RETINAL is the substrate → supplemental vitamin A → "
            "increased all-trans retinal → increased NRPE → MORE bisretinoid → WORSE disease; "
            "dietary vitamin A is safe (normal physiological amounts); mega-dose supplements are NOT"
        ),
        "phenotype": (
            "JUVENILE ONSET (usually age 8–20 years): "
            "Central visual acuity loss (VA 20/50–20/200 range); "
            "central/paracentral scotoma; "
            "reading difficulty; colour vision disturbance; "
            "FUNDUS: pisciform (fish-tail) yellow-white flecks at RPE level (PATHOGNOMONIC); "
            "beaten bronze macular sheen; late — extensive RPE atrophy; "
            "FAF: central decreased AF (RPE atrophy) + surrounding hyper-AF (lipofuscin accumulation); "
            "SILENT CHOROID on FA (lipofuscin blocks choroidal fluorescence); "
            "OCT: outer nuclear layer thinning; IS/OS (ellipsoid zone) disruption; RPE loss; "
            "ERG: variable — often preserved peripherally early; full-field ERG may be normal or mildly reduced early; "
            "macular ERG (multifocal ERG) severely reduced at fovea; "
            "late STGD1: CORD (cone-rod dystrophy) pattern — peripheral involvement too"
        ),
        "hallmark": (
            "PISCIFORM FLECKS IN A TEENAGER WITH CENTRAL VISION LOSS = STGD1/ABCA4 until proven otherwise; "
            "SILENT CHOROID ON FA: lipofuscin blocks choroidal fluorescence — DISTINCTIVE; "
            "VITAMIN A MEGA-DOSE SUPPLEMENTS ABSOLUTELY CONTRAINDICATED: "
            "worsen bisretinoid/A2E accumulation → faster photoreceptor and RPE loss; "
            "dietary vitamin A (in food) is SAFE — only high-dose supplement capsules are CI; "
            "p.Gly1961Glu: mild/late-onset allele (IVD pattern) — may be missed if searching only for severe alleles; "
            "BEATEN BRONZE MACULAR SHEEN: early clinical sign at fundoscopy; "
            "AVOID BRIGHT LIGHT (blue-light avoidance investigational); UV-blocking lenses; "
            "EMIXUSTAT PHASE III: visual cycle modulator targeting RPE65 (reduces 11-cis retinal synthesis → less A2E); "
            "primary endpoint not met in 2022 trial; optimization ongoing"
        ),
        "treatment_alert": (
            "VITAMIN A SUPPLEMENTS: ABSOLUTELY CONTRAINDICATED; "
            "all-trans retinal is the SUBSTRATE for bisretinoid → supplemental vitamin A → more substrate → more A2E; "
            "dietary vitamin A (normal food) is safe; advise patients explicitly; "
            "multi-vitamins containing vitamin A/beta-carotene should be AVOIDED; "
            "DARK ADAPTATION COUNSELLING: avoid prolonged dark adaptation (activates more rhodopsin → more atRAL); "
            "BLUE-LIGHT AVOIDANCE: investigational; blue-blocking lenses (amber-tinted) may reduce bisretinoid phototoxicity; "
            "EMIXUSTAT (ACU-4429): visual cycle modulator (RPE65 inhibitor) Phase III; "
            "insufficient efficacy at primary endpoint 2022; ongoing optimization; "
            "GENE THERAPY: AAV-ABCA4 challenging due to large gene size (need dual AAV or alternative vector); "
            "research ongoing; not yet clinical phase; "
            "RNA ANTISENSE OLIGONUCLEOTIDE (ASO): for deep intronic variants; in development; "
            "GENETIC COUNSELLING: AR; sibling risk 25%; single allele found → MLPA + deep intronic variant testing; "
            "LOW-VISION AIDS: magnification; eccentric viewing; "
            "avoid vitamin A-rich supplements; register with sight loss charity"
        ),
        "key_ddx": (
            "Best disease (BEST1, AD vitelliform dystrophy): vitelliform 'egg-yolk' lesion on OCT; EOG Arden ratio reduced <1.5; "
            "PRPH2-pattern dystrophy (AD): butterfly/annular pattern; older onset; ERG normal or mildly reduced; "
            "AMD (age-related macular degeneration): sporadic; older onset; drusen at RPE-Bruch's interface; no flecks; "
            "cone dystrophy (GUCY2D, CNGA3, CNGB3): photophobia + achromatopsia; cone-selective ERG; "
            "central serous chorioretinopathy (CSR): fluid accumulation; male predominance; corticosteroid association"
        ),
        "vision_pattern": "Central VA loss 20/50–20/200; central/paracentral scotoma; peripheral vision preserved until late",
        "erg_pattern": "Full-field ERG normal to mildly reduced early; multifocal ERG central severely reduced; late CORD pattern",
        "severity_weights": [0.20, 0.55, 0.25],  # mild/moderate/severe
    },

    # ── RDH12 — LCA13 / EOSRD ────────────────────────────────────────────────
    {
        "gene": "RDH12",
        "protein": "RDH12 (Retinal Dehydrogenase 12 / All-Trans Retinol Dehydrogenase)",
        "alias": (
            "RDH12; OMIM gene 608830; Leber Congenital Amaurosis 13 (LCA13) #612712 / EOSRD (Early-Onset Severe Retinal Dystrophy); "
            "14q24.1; 316 aa; ~35 kDa; "
            "AR; severe infantile-onset retinal dystrophy; retinal dehydrogenase 12; "
            "reduces all-trans retinal (atRAL) → all-trans retinol in the visual cycle"
        ),
        "aa": "316 aa",
        "kDa": "~35 kDa",
        "locus": "14q24.1",
        "omim_gene": 608830,
        "omim_disease": 612712,
        "inheritance": "AR (biallelic)",
        "gene_class": (
            "Short-chain dehydrogenase/reductase (SDR) family; NADPH-dependent; "
            "FUNCTION: reduces all-trans retinal (atRAL) to all-trans retinol (atROL) in photoreceptor inner segment; "
            "atRAL is released from opsin after photoactivation → atRAL is toxic at high concentrations; "
            "RDH12 detoxification: atRAL → atROL (non-toxic) → LRAT → all-trans retinyl ester → "
            "exported to RPE for 11-cis retinal recycling via retinoid cycle (RPE65 dependent); "
            "LOF → atRAL accumulates in photoreceptor inner segment → photoreceptor toxicity; "
            "CONSEQUENCE: no dark adaptation from birth (cannot regenerate rhodopsin efficiently); "
            "infantile onset: nystagmus + photophobia from months of age; "
            "CONTRAST WITH ABCA4: RDH12 acts at REDUCTION step (atRAL → atROL in inner segment); "
            "ABCA4 acts at FLIPPING step (NRPE across disc membrane); "
            "vitamin A supplementation is SAFE (even beneficial) in RDH12 — provides substrate (atROL) "
            "for RPE65 cycle; this CONTRASTS with ABCA4 where mega-dose vit A INCREASES atRAL → worsens disease; "
            "p.Leu99Ile: recurrent pathogenic variant; moderate-severe; "
            "p.Arg161Gln: mild-moderate range; "
            "p.Arg293*: null — severe; "
            "ERG: profoundly reduced or extinguished rod AND cone responses by age 2 (panretinal, not macular selective); "
            "this distinguishes LCA13 from STGD1 (macular-selective ERG normal peripherally)"
        ),
        "phenotype": (
            "SEVERE INFANTILE ONSET (LCA pattern): "
            "Nystagmus: onset 2–4 months of life; "
            "Photophobia: severe from infancy; "
            "DIGIT-RUBBING / OCULODIGITAL SIGN (Franceschetti): rubbing eyes → phosphene stimulation; "
            "ERG: profoundly reduced or EXTINGUISHED rod AND cone responses by age 2; "
            "NO dark adaptation from birth; "
            "Fundus: may appear relatively normal early (deceptively); retinal vessel attenuation; "
            "optic disc pallor develops; peripheral pigmentary changes; "
            "OCT: photoreceptor layer thinning from infancy; IS/OS zone disrupted; "
            "Visual acuity: severely reduced (CF to HM) from infancy; "
            "Cognitive: normal (isolated retinal disease, no syndromic associations); "
            "No hearing loss, no kidney/systemic involvement (non-syndromic)"
        ),
        "hallmark": (
            "INFANTILE NYSTAGMUS + PHOTOPHOBIA + EXTINGUISHED ERG = LCA — RDH12 is LCA13; "
            "ERG EXTINGUISHED ROD AND CONE BY AGE 2: pan-retinal (distinguishes from STGD1 — macular only); "
            "DIGIT-RUBBING: child rubs eyes rhythmically (phosphene-seeking oculodigital sign); "
            "RDH12-LCA13 vs CEP290-LCA10: CEP290 has deep intronic IVS26 mutation missed by standard exome; "
            "RDH12-LCA13 vs GUCY2D-LCA1: GUCY2D has minimally reduced cone ERG initially (amaurosis fugax); "
            "RDH12-LCA13 vs RPE65-LCA2: TREATABLE with voretigene neparvovec-rzyl (Luxturna) for RPE65 — "
            "MUST distinguish LCA13 from LCA2 before considering voretigene; "
            "VITAMIN A SAFE in RDH12 (contrast to ABCA4 where supplement vitamin A is CI); "
            "LOW-VISION AIDS ESSENTIAL from infancy: tactile/Braille learning; orientation mobility"
        ),
        "treatment_alert": (
            "VITAMIN A SUPPLEMENTATION: SAFE (and may be beneficial) — provides atROL substrate for residual RDH12 activity; "
            "CONTRAST WITH ABCA4: vitamin A mega-dose is CI in ABCA4 but SAFE in RDH12; "
            "NO APPROVED THERAPY for RDH12-LCA13 specifically; "
            "voretigene neparvovec-rzyl (Luxturna) is approved for RPE65-LCA2 ONLY — "
            "DO NOT USE for RDH12-LCA13 (wrong enzyme in visual cycle); "
            "GENE THERAPY: AAV-RDH12 at preclinical stage; animal model studies ongoing; "
            "PHOTOPHOBIA MANAGEMENT: dark-tint lenses; dark environments; red-filter contacts; "
            "LOW-VISION AIDS: white cane + orientation/mobility training from preschool; "
            "Braille education recommended; "
            "OCULODIGITAL SIGN: reassure parents (not damaging); supervise to prevent corneal abrasion; "
            "GENETIC COUNSELLING: AR; sibling risk 25%; "
            "single allele found → MLPA + deep intronic analysis; "
            "SOCIAL SERVICES: register visual impairment; educational psychology; "
            "SYSTEMIC: no systemic surveillance needed (non-syndromic)"
        ),
        "key_ddx": (
            "CEP290-LCA10 (12q21.32): deep intronic IVS26 c.2991+1655A>G MISSES standard exome — target sequencing mandatory; "
            "GUCY2D-LCA1 (17p13.1): most common LCA gene; minimally reduced cone ERG initially; amaurosis fugax; "
            "RPE65-LCA2 (1p31.3): TREATABLE with voretigene (Luxturna) — CRITICAL TO DISTINGUISH from RDH12; "
            "AIPL1-LCA4 (17p13.2): severe; neonatal ERG extinguished; "
            "CRB1-LCA8/RP12 (1q31.3): preserved para-arteriolar RPE (PPRPE sign); nummular pigmentation; "
            "TULP1-LCA15 (6p21.3): severe early onset; salt-and-pepper fundus"
        ),
        "vision_pattern": "Infantile-onset severely reduced VA (CF to HM) from first months; severe photophobia; no dark adaptation",
        "erg_pattern": "Extinguished rod AND cone ERG by age 2 (pan-retinal; not macular-selective as in STGD1)",
        "severity_weights": [0.05, 0.25, 0.70],  # mild/moderate/severe
    },

    # ── PRPF31 — RP11 ────────────────────────────────────────────────────────
    {
        "gene": "PRPF31",
        "protein": "PRPF31 (Pre-mRNA Processing Factor 31 / U4/U6-U5 tri-snRNP component)",
        "alias": (
            "PRPF31; OMIM gene 606419; Retinitis Pigmentosa 11 (RP11) #600138; 19q13.42; 499 aa; ~55 kDa; "
            "AD (haploinsufficiency); accounts for ~3–8% of AD RP; "
            "INCOMPLETE PENETRANCE (50–80% in families) — explained by expression level of normal allele; "
            "MLPA MANDATORY — 15% of PRPF31 cases are large deletions missed by standard sequencing"
        ),
        "aa": "499 aa",
        "kDa": "~55 kDa",
        "locus": "19q13.42",
        "omim_gene": 606419,
        "omim_disease": 600138,
        "inheritance": "AD (haploinsufficiency); INCOMPLETE PENETRANCE (50–80%)",
        "gene_class": (
            "Pre-mRNA processing factor; spliceosome component; "
            "PRPF31 is an essential component of the U4/U6-U5 tri-snRNP spliceosome complex; "
            "tri-snRNP: PRPF31 + PRPF3 + PRPF6 + PRPF8 + EFTUD2 + other proteins + U4/U6 and U5 snRNAs; "
            "PRPF31 bridges U4 snRNA and 15.5K protein (NHPX/SNRNP15.5) in tri-snRNP assembly; "
            "RP mechanism: photoreceptors have highest transcriptional activity → most spliceosome demand → "
            "haploinsufficiency threshold reached first in photoreceptors; "
            "INCOMPLETE PENETRANCE MECHANISM: "
            "normal allele expression level determines penetrance; "
            "high normal PRPF31 expression: CNOT3 (Ccr4-Not complex subunit 3) promotes transcription + "
            "long non-coding RNA PRPF31-AS1 also modulates; "
            "families with high normal allele expression → lower penetrance (unaffected obligate carriers); "
            "families with lower normal allele expression → higher penetrance; "
            "'unaffected carrier' = has mutation but sufficient normal PRPF31 from other allele expression; "
            "allele-specific silencing by miRNA can also explain penetrance variation; "
            "p.Arg423*: most common null allele; p.Ala351Pro: missense; p.Leu197Pro: missense; "
            "LARGE DELETIONS: 15% of PRPF31 cases — chromosomal region deletions at 19q13.42; "
            "standard sequencing misses these; MLPA (multiplex ligation-dependent probe amplification) detects; "
            "other spliceopathy RP genes: PRPF8 (RP13, 17p13.3), TOPORS (RP31, 9p21.1), PRPF3 (RP18, 1q21.2), "
            "PRPF6 (RP60, 20q13.33) — all cause AD RP via spliceosome dysfunction"
        ),
        "phenotype": (
            "VARIABLE ONSET due to incomplete penetrance: 2nd to 6th decade (wide range); "
            "CLASSIC RP FEATURES: night blindness first; peripheral field loss; tunnel vision; "
            "central vision preserved until late; "
            "Fundus: bone-spicule pigmentation peripheral; retinal vessel attenuation; optic disc pallor (waxy); "
            "INCOMPLETE PENETRANCE: 20–50% of mutation carriers in a family may be UNAFFECTED; "
            "'skipped generation' pattern — AD RP with apparent gaps in family history; "
            "INTRAFAMILY VARIABILITY: affected members range from minimal symptoms (30 years) "
            "to severe RP (20 years); "
            "ERG: reduced rod and cone amplitudes; scotopic first; "
            "OCT: outer nuclear layer thinning; ellipsoid zone loss; "
            "No systemic features (non-syndromic); "
            "Anterior segment: posterior subcapsular cataract in some cases"
        ),
        "hallmark": (
            "AD RP WITH INCOMPLETE PENETRANCE + VARIABLE EXPRESSIVITY — think PRPF31; "
            "UNAFFECTED OBLIGATE CARRIERS in family — parent affected → grandparent unaffected → "
            "grandchild affected: PRPF31 INCOMPLETE PENETRANCE explains this 'skipped generation'; "
            "MLPA MANDATORY: 15% of PRPF31 cases are large deletions → standard sequencing returns NEGATIVE; "
            "negative sequencing does NOT rule out PRPF31 — must order MLPA; "
            "PENETRANCE MODIFIER: CNOT3 expression level on normal allele; "
            "families can have 'high-expression' or 'low-expression' normal allele; "
            "this explains why some families have 80% penetrance and others 50%; "
            "SPLICEOPATHY DDx: PRPF8-RP13 + PRPF3-RP18 + PRPF6-RP60 + TOPORS-RP31 — "
            "all cause AD RP via spliceosome; only molecular panel distinguishes"
        ),
        "treatment_alert": (
            "MLPA MANDATORY: order alongside sequence-based panel for PRPF31; "
            "15% of PRPF31 cases are large deletions — sequencing-only panel will MISS these; "
            "GENETIC COUNSELLING: incomplete penetrance complicates risk estimation; "
            "each child of an affected parent has 50% chance of inheriting the allele; "
            "but penetrance is 50–80% — not all who inherit will develop RP; "
            "CNOT3 modifier testing: research-based; can help estimate penetrance in specific families; "
            "NO APPROVED THERAPY: no RPE65-like specific treatment for PRPF31-RP; "
            "gene therapy approaches (PRPF31 AAV supplementation): in preclinical development; "
            "VITAMIN A: some centres use 15,000 IU/day for AD RP (Berson 1993 data); "
            "evidence specific to PRPF31-RP limited; "
            "DHA omega-3 supplementation: investigational adjunct; "
            "LOW-VISION REHABILITATION: eccentric viewing; magnification; "
            "DRIVING: night driving restrictions when night vision significantly impaired; "
            "GENETIC TESTING FAMILY CASCADE: test all 1st-degree relatives; "
            "unaffected relatives who carry mutation need ophthalmic monitoring"
        ),
        "key_ddx": (
            "PRPF8-RP13 (17p13.3): AD spliceopathy RP; similar RP; gene panel needed; "
            "PRPF3-RP18 (1q21.2): AD spliceopathy RP; similar RP; panel needed; "
            "PRPF6-RP60 (20q13.33): AD spliceopathy RP; similar RP; panel needed; "
            "TOPORS-RP31 (9p21.1): AD RP; nuclear envelope protein + E3 ligase; panel needed; "
            "RHO-RP4 (3q22.1): most common AD RP gene; no incomplete penetrance; rhodopsin mutation; "
            "RPGR-XLRP: X-linked; male predominance; ORF15 NGS issue"
        ),
        "vision_pattern": "Variable onset 2nd–6th decade; classic RP trajectory — night blindness first; peripheral then central loss",
        "erg_pattern": "Reduced rod and cone amplitudes; scotopic first; intrafamily variability in ERG severity",
        "severity_weights": [0.25, 0.50, 0.25],  # mild/moderate/severe
    },

    # ── EYS — RP25 ───────────────────────────────────────────────────────────
    {
        "gene": "EYS",
        "protein": "EYS (Eyes Shut Homolog — Photoreceptor Ciliary Spacer Protein)",
        "alias": (
            "EYS; OMIM gene 612424; Retinitis Pigmentosa 25 (RP25) #602772; 6q12; 3165 aa; ~346 kDa; "
            "AR; largest non-dystrophin photoreceptor-specific gene (3 Mb genomic; 44 exons); "
            "most common AR RP gene in Japanese patients (30–35% of Japanese AR RP); "
            "5–10% of European AR RP; "
            "large genomic deletions: 15–20% of EYS cases; MLPA MANDATORY"
        ),
        "aa": "3165 aa",
        "kDa": "~346 kDa",
        "locus": "6q12",
        "omim_gene": 612424,
        "omim_disease": 602772,
        "inheritance": "AR (biallelic)",
        "gene_class": (
            "Eyes shut homolog (Drosophila); spacer/structural protein; no enzymatic activity; "
            "3 Mb genomic span — largest photoreceptor-specific gene (DYSTROPHIN at 2.4 Mb affects all muscle); "
            "44 exons; 3165 amino acids; "
            "DOMAIN STRUCTURE: EGF-like repeats + laminin G domains; "
            "LOCALISATION: connecting cilium / calyceal process of photoreceptors; "
            "spacer protein that maintains structural integrity of the calyceal process and periciliary space; "
            "calyceal processes are finger-like projections surrounding the photoreceptor outer segment base; "
            "EYS LOF → calyceal process structural failure → outer segment attachment unstable → "
            "rod photoreceptor degeneration → night blindness → progressive peripheral field loss; "
            "GENE SIZE CHALLENGE: 3165 aa cDNA >> 4.7 kb AAV cargo limit → gene therapy DIFFICULT; "
            "dual AAV overlapping split approach feasible in principle; "
            "JAPANESE FOUNDER VARIANTS: p.Tyr2935* and p.Gln2767*; "
            "EUROPEAN VARIANTS: p.Pro1734Leu; c.8648-2A>G (splice); "
            "LARGE DELETIONS: 15–20% of EYS cases — MLPA essential; "
            "standard exome misses large deletions; "
            "vitamin A supplementation: evidence less clear-cut than RPE65/GUCY2D; investigational"
        ),
        "phenotype": (
            "CLASSIC AUTOSOMAL RECESSIVE RP: "
            "Onset: night blindness 1st–2nd decade; "
            "Progressive peripheral field loss; tunnel vision; "
            "Fundus: bone-spicule pigmentation; vessel attenuation; optic disc pallor (waxy); "
            "ERG: markedly reduced rod and cone responses; scotopic >> photopic reduction; "
            "OCT: outer nuclear layer thinning; IS/OS disruption from periphery inward; "
            "central vision preserved longer (10–15 years after night blindness onset typically); "
            "rate of progression variable between patients; "
            "NO SYSTEMIC FEATURES (non-syndromic); no HL, no kidney involvement; "
            "JAPANESE ENRICHMENT: ~30–35% of AR RP in Japanese patients → EYS is first-line consideration"
        ),
        "hallmark": (
            "AR RP IN A JAPANESE PATIENT — EYS most common (30–35% of Japanese AR RP); "
            "EYS LARGEST RETINAL GENE (3 Mb): MLPA MANDATORY — 15–20% large deletions; "
            "standard panel negative does NOT exclude EYS — must add MLPA; "
            "SINGLE ALLELE FOUND: order MLPA before concluding monoallelic; "
            "GENE THERAPY CHALLENGE: 3165 aa >> AAV capacity (4.7 kb); novel vector strategies needed; "
            "VITAMIN A: use cautiously; evidence less clear in EYS than in other RP genes; "
            "p.Tyr2935* and p.Gln2767* are Japanese founder variants — fast-track in Japanese patients; "
            "MLPA for EYS is distinct from PRPF31 MLPA — need EYS-specific MLPA probe set"
        ),
        "treatment_alert": (
            "MLPA MANDATORY: 15–20% of EYS cases are large genomic deletions; "
            "EYS-specific MLPA probe set (not generic); "
            "standard exome/panel sequencing misses large deletions at 6q12; "
            "GENE THERAPY: currently not feasible with standard AAV (gene too large); "
            "research ongoing with dual AAV split constructs; "
            "VITAMIN A SUPPLEMENTATION: investigational for RP; less clear evidence for EYS-RP than RPE65/GUCY2D; "
            "use at clinician discretion (some centres 15,000 IU/day); "
            "DHA supplementation: investigational; "
            "LOW-VISION REHABILITATION: as per other RP; eccentric viewing training; "
            "DRIVING: restrict night driving early; periodic field test for driving suitability; "
            "GENETIC COUNSELLING: AR; sibling 25%; cascade testing of 1st-degree relatives; "
            "Japanese cohort: p.Tyr2935* + p.Gln2767* — targeted screening before full panel; "
            "SOCIAL SERVICES: visual impairment registration; "
            "RETINAL CLINIC FOLLOW-UP: annual ERG + visual field + OCT monitoring"
        ),
        "key_ddx": (
            "USH2A-RP25: check HL status — if SNHL present → USH2A preferred; EYS is non-syndromic; "
            "CNGB1-RP45 (2q11.2): AR RP; rod-specific; slow ERG recovery; "
            "PDE6A-RP43 (5q33.1): AR RP; rod cGMP phosphodiesterase alpha; "
            "PDE6B-RP40 (4p16.3): AR RP; rod cGMP phosphodiesterase beta; "
            "CNGB1-RP45: night blindness + slow ERG ON response (rod-specific); "
            "RPGR-XLRP: X-linked; male predominance; different inheritance"
        ),
        "vision_pattern": "Night blindness 1st–2nd decade; peripheral field loss progressive; central vision preserved longer",
        "erg_pattern": "Markedly reduced rod and cone responses; scotopic predominant; progressive reduction",
        "severity_weights": [0.15, 0.55, 0.30],  # mild/moderate/severe
    },

    # ── CNGB3 — Achromatopsia ACHM3 ──────────────────────────────────────────
    {
        "gene": "CNGB3",
        "protein": "CNGB3 (Cyclic Nucleotide-Gated Channel Beta-3 Subunit)",
        "alias": (
            "CNGB3; OMIM gene 605080; Achromatopsia 3 (ACHM3) #262300; 8q21.3; 809 aa; ~94 kDa; "
            "AR; most common achromatopsia gene (~50% of all achromatopsia); "
            "tetrameric CNG channel (2×CNGA3 + 2×CNGB3) controls cone photoreceptor depolarization; "
            "gene therapy: botaretigene sparoparvovec (BIIB112 / AAV-CNGB3, MeiraGTx/Janssen) Phase III"
        ),
        "aa": "809 aa",
        "kDa": "~94 kDa",
        "locus": "8q21.3",
        "omim_gene": 605080,
        "omim_disease": 262300,
        "inheritance": "AR (biallelic)",
        "gene_class": (
            "Cyclic nucleotide-gated (CNG) channel beta-3 subunit; "
            "CNG channel: heterotetrameric; 2×CNGA3 + 2×CNGB3 in cone inner segment; "
            "in dark-adapted state: cGMP binds CNG channel → channel open → Na+/Ca2+ influx → cone depolarised; "
            "light activation: cone opsins (L/M/S-cone pigments) → Gt (transducin) → PDE6 → cGMP hydrolysis → "
            "cGMP falls → CNG channel closes → cone hyperpolarises → synaptic signal; "
            "CNGB3 role: regulatory beta subunit; modulates channel gating kinetics + CNG channel expression; "
            "CNGB3 LOF → CNG channel non-functional (CNGA3 alone insufficiently assembled) → "
            "cone photoreceptor cannot depolarise → NO cone phototransduction → achromatopsia; "
            "rod CNG channels: CNGA1 + CNGB1 — unaffected → NORMAL rod function; "
            "COMPLETE ACHROMATOPSIA: total loss of cone function; "
            "p.Thr383IlefsX* (c.1148delC): German/Norwegian founder (~50% of CNGB3 alleles in European patients); "
            "p.Arg403Gln: second most common; "
            "GENE THERAPY: AAV8-CNGB3 (botaretigene sparoparvovec, BIIB112): "
            "MeiraGTx (licensed to Janssen); Phase III trial ongoing; "
            "Phase I/II showed partial recovery of cone ERG and objective acuity in treated adults; "
            "HYPOMORPHIC ALLELES: some CNGB3 variants → incomplete achromatopsia (some residual colour); "
            "blue-cone monochromacy (BCM): OPN1LW + OPN1MW deletion Xq28 → "
            "only short-wavelength (blue) cone function; distinguished from ACHM by X-linked pattern"
        ),
        "phenotype": (
            "COMPLETE ACHROMATOPSIA (ACHM): "
            "NYSTAGMUS: pendular, onset 2–4 months of life; "
            "PHOTOPHOBIA: severe; children avoid bright light; head nodding; "
            "REDUCED VISUAL ACUITY: 20/200 range (0.05–0.1 Snellen); "
            "ABSENT COLOUR DISCRIMINATION: complete monochromat; no red-green or blue-yellow colour; "
            "Foveal hypoplasia on OCT: inner segment zone disruption at fovea; "
            "no complete IS/OS (ellipsoid zone) signal at fovea (cone IS/OS absent); "
            "DARK-ADAPTED (SCOTOPIC) ERG: NORMAL rod responses; "
            "PHOTOPIC ERG: EXTINGUISHED (cone-selective defect); "
            "this pattern (normal rod + absent cone ERG) is DIAGNOSTIC for achromatopsia; "
            "fundus: foveal reflex absent; otherwise may appear relatively normal; "
            "no bone-spicule pigmentation (not an RP); "
            "PROGRESSIVE: slowly progressive in some patients; generally considered stationary in childhood"
        ),
        "hallmark": (
            "PENDULAR NYSTAGMUS + PHOTOPHOBIA + NO COLOR VISION + NORMAL ROD ERG = ACHROMATOPSIA; "
            "NORMAL SCOTOPIC (ROD) ERG + EXTINGUISHED PHOTOPIC (CONE) ERG: DIAGNOSTIC pattern; "
            "p.Thr383IlefsX* (GERMAN/NORWEGIAN FOUNDER): screen first in European patients; "
            "GENE THERAPY IMMINENT: botaretigene sparoparvovec (BIIB112, MeiraGTx/Janssen) Phase III; "
            "most promising near-future approved therapy among 8 retinal dystrophy genes; "
            "DARK-ADAPTED ENVIRONMENT MANAGEMENT: children prefer dim light; arrange classroom + home accordingly; "
            "TINTED CONTACT LENSES: red-filter / dark-tinted lenses reduce photophobia; "
            "INCOMPLETE ACHM: CNGB3 hypomorphic alleles → some residual colour → distinguish from complete ACHM; "
            "BLUE-CONE MONOCHROMACY DDx: X-linked (males affected); some colour (blue/yellow discrimination); "
            "OPN1LW + OPN1MW gene deletion at Xq28"
        ),
        "treatment_alert": (
            "GENE THERAPY: botaretigene sparoparvovec (BIIB112/AAV-CNGB3, MeiraGTx/Janssen) Phase III; "
            "subretinal injection; refer patients to trial site if eligible; "
            "Phase I/II results: partial cone ERG recovery + improved light-adapted acuity; "
            "trial now enrolling adults and children; "
            "PHOTOPHOBIA MANAGEMENT: "
            "dark-tinted contact lenses (red-filter); amber-tinted glasses; "
            "brim hats; dark environments at school and home; "
            "avoid bright sunlight without protection; "
            "VISUAL AIDS: magnification; dark backgrounds on screens; high contrast text; "
            "NYSTAGMUS MANAGEMENT: null position; biofeedback approaches; "
            "EDUCATIONAL: classroom front-row seating; large print; "
            "coloured filters for reading; avoid glare; "
            "GENETIC COUNSELLING: AR; sibling 25%; "
            "p.Thr383IlefsX* founder — fast-track test in German/Norwegian heritage; "
            "CNGA3 (25% of ACHM) — test if CNGB3 negative; panel covers both; "
            "FOLLOW-UP: annual photopic ERG + OCT foveal structure monitoring; "
            "counsel family that gene therapy is in Phase III — plan for access when approved"
        ),
        "key_ddx": (
            "CNGA3-ACHM2 (2q11.2): identical achromatopsia phenotype; ~25% of ACHM; gene panel needed; "
            "GNAT2-ACHM4 (1p13.3): identical; ~2% of ACHM; "
            "PDE6C-ACHM5 (10q23.33): identical; rare; "
            "Blue-cone monochromacy (BCM, OPN1LW+OPN1MW deletion Xq28): X-linked; "
            "some colour (blue/violet wavelength discrimination); distinguishable; "
            "Incomplete achromatopsia: residual cone function; hypomorphic CNGB3 alleles; "
            "Congenital stationary night blindness (CSNB): night blindness dominant; normal photopic ERG; "
            "Nystagmus-only workup: FRMD7 (X-linked idiopathic infantile nystagmus) — no visual loss"
        ),
        "vision_pattern": "VA 20/200 (0.05–0.1); complete colour blindness; photophobia; no central field loss (scotoma not typical)",
        "erg_pattern": "Normal scotopic (rod) ERG; extinguished photopic (cone) ERG — diagnostic for ACHM",
        "severity_weights": [0.15, 0.60, 0.25],  # mild/moderate/severe
    },

    # ── RS1 — X-linked Retinoschisis ─────────────────────────────────────────
    {
        "gene": "RS1",
        "protein": "Retinoschisin (RS1 — Octameric Discoidal Cell Adhesion Protein)",
        "alias": (
            "RS1; OMIM gene 300839; X-linked Retinoschisis (XLRS) #312700; Xp22.13; 224 aa; ~24 kDa; "
            "X-linked (XL); males affected; females carriers (rarely symptomatic); "
            "retinoschisin: secreted octameric discoidal complex; "
            "cell adhesion + synaptic transmission in inner nuclear layer (INL); "
            "splitting of INL on OCT = retinoschisis; spoke-wheel foveal pattern PATHOGNOMONIC"
        ),
        "aa": "224 aa",
        "kDa": "~24 kDa",
        "locus": "Xp22.13",
        "omim_gene": 300839,
        "omim_disease": 312700,
        "inheritance": "X-linked (XL); males affected; female carriers (rarely symptomatic with mild schisis)",
        "gene_class": (
            "Discoidin domain protein; "
            "STRUCTURE: 24 aa signal peptide + RS1 domain + discoidin (DS) domain; "
            "secreted as disulphide-linked octamer (dimer of tetramers) into interphotoreceptor matrix; "
            "FUNCTION: cell adhesion and synaptic integrity in retina; "
            "highly expressed in photoreceptors (rods + cones) and bipolar cells; "
            "maintains structural integrity of inner nuclear layer (INL) and outer plexiform layer (OPL); "
            "LOF → loss of cell adhesion between INL layers → SPLITTING of INL → retinoschisis; "
            "FOVEAL RETINOSCHISIS: spoke-wheel pattern on OCT (radial folds of INL at fovea); "
            "PATHOGNOMONIC for XLRS; "
            "ERG CHARACTERISTIC: ELECTRONEGATIVE ERG — b-wave amplitude << a-wave amplitude; "
            "normal ERG: b-wave > a-wave (b reflects INL/bipolar activity; a reflects photoreceptor); "
            "in XLRS: photoreceptor (a-wave) relatively preserved; INL signal (b-wave) severely reduced; "
            "b-wave deficit reflects INL retinoschisin deficiency → bipolar cell synaptic failure; "
            "CARBONIC ANHYDRASE INHIBITORS (CAI) — PARADOXICAL WORSENING: "
            "dorzolamide, brinzolamide eye drops + oral acetazolamide → INCREASE cystic schisis volume in XLRS; "
            "mechanism: CAI shifts fluid dynamics → increases schisis cavities; "
            "OPPOSITE of expected effect in cystoid macular edema (CME) — "
            "CME responds to CAI; XLRS WORSENS with CAI; "
            "CAI ABSOLUTELY CONTRAINDICATED IN XLRS; "
            "p.Arg213Trp: most common pathogenic variant; p.Arg102Trp; p.Cys59Ser; "
            "exon 3–6 missense mutations most common; exon deletions detectable by MLPA"
        ),
        "phenotype": (
            "MALES (affected): "
            "Bilateral symmetric visual acuity reduction from childhood (VA 20/50–20/200); "
            "SPOKE-WHEEL FOVEAL RETINOSCHISIS on OCT: radial cystic folds at fovea; PATHOGNOMONIC; "
            "INL splitting visible as cystic spaces on cross-sectional OCT; "
            "fundus: spoke-wheel or stellate macular pattern; may appear subtle without OCT; "
            "PERIPHERAL RETINOSCHISIS: ~50% have peripheral schisis (different quadrant to foveal); "
            "bullous schisis in periphery → vitreoretinal complications; "
            "VITREOUS HAEMORRHAGE risk from peripheral schisis; "
            "RETINAL DETACHMENT risk (rhegmatogenous from schisis break); "
            "ERG: ELECTRONEGATIVE PATTERN — b-wave selectively reduced; b << a; "
            "FEMALES: carriers; rarely develop mild foveal schisis; usually asymptomatic; "
            "COMPLICATIONS: vitreous haemorrhage; retinal detachment; tractional membranes; "
            "STRABISMUS / AMBLYOPIA in childhood: reduced VA → amblyopia risk"
        ),
        "hallmark": (
            "SPOKE-WHEEL FOVEAL SCHISIS ON OCT IN A MALE = XLRS/RS1; PATHOGNOMONIC; "
            "ELECTRONEGATIVE ERG: b-wave << a-wave — DIAGNOSTIC; "
            "b-wave reflects INL/bipolar transmission; a-wave reflects photoreceptor; "
            "XLRS: photoreceptor (a) normal; INL (b) lost → b << a → electronegative; "
            "CARBONIC ANHYDRASE INHIBITORS (CAI) ABSOLUTELY CONTRAINDICATED: "
            "dorzolamide + brinzolamide eye drops + oral acetazolamide WORSEN retinoschisis; "
            "increases cystic schisis volume (OPPOSITE of CME response); "
            "NEVER prescribe topical CAI (dorzolamide/brinzolamide) or oral acetazolamide in XLRS; "
            "GENE THERAPY: RGX-121 (REGENXBIO, AAV intravitreal) Phase I/II; stabilisation reported; "
            "CSNB DDx: congenital stationary night blindness also has electronegative ERG but NO schisis on OCT; "
            "VITREORETINAL COMPLICATION: vitreous haemorrhage + retinal detachment → ophthalmology emergency"
        ),
        "treatment_alert": (
            "CARBONIC ANHYDRASE INHIBITORS (CAI): ABSOLUTELY CONTRAINDICATED; "
            "dorzolamide (Trusopt) eye drops → WORSEN XLRS schisis (paradoxical); "
            "brinzolamide (Azopt) eye drops → WORSEN XLRS schisis; "
            "oral acetazolamide (Diamox) → WORSEN XLRS schisis; "
            "these drugs are used in CME (other causes) — XLRS is the EXCEPTION where they are CI; "
            "advise all prescribers + patient card; "
            "GENE THERAPY: RGX-121 (REGENXBIO AAV9-RS1 intravitreal, Phase I/II): "
            "stabilisation in some patients; early results promising; not yet Phase III; "
            "BEVACIZUMAB: for neovascular complications; "
            "VITREORETINAL SURGERY: for retinal detachment or non-resolving vitreous haemorrhage; "
            "PERIPHERAL SCHISIS LASER DEMARCATION: controversial; may increase detachment risk; "
            "typically observe peripheral schisis unless bullous or complicated; "
            "AMBLYOPIA THERAPY: early aggressive if detected in childhood; "
            "optical correction + patching for amblyopia; "
            "GENETIC COUNSELLING: X-linked; no male-to-male transmission; "
            "daughters of affected males are OBLIGATE carriers; "
            "sisters and mothers of affected males need carrier testing (50% risk if mother is carrier); "
            "FOLLOW-UP: annual ERG + OCT monitoring; ophthalmology emergency pathway for vitreous haemorrhage"
        ),
        "key_ddx": (
            "CSNB (congenital stationary night blindness): electronegative ERG but NO retinoschisis on OCT; "
            "CSNB is non-progressive; XLRS can be mildly progressive; OCT distinguishes; "
            "Goldman-Favre / NR2E3 enhanced S-cone syndrome: schisis + night blindness + cystoid macular lesions; "
            "NR2E3 gene (15q23); AR; distinctive enhanced S-cone ERG; "
            "vitreoretinal traction with foveal schisis: secondary to epiretinal membrane; no RS1 mutation; OCT + history; "
            "Juvenile X-linked retinoschisis vs exudative vitreoretinopathy (EVR/FZD4): "
            "EVR has different OCT and family history; FZD4 mutation; "
            "ACHM with foveal hypoplasia: no schisis on OCT; different ERG pattern"
        ),
        "vision_pattern": "VA 20/50–20/200 in males from childhood; central scotoma possible; peripheral schisis complications",
        "erg_pattern": "Electronegative ERG (b-wave << a-wave) — selective INL/bipolar transmission defect; pathognomonic",
        "severity_weights": [0.20, 0.55, 0.25],  # mild/moderate/severe
    },
]


# ─── Patient simulation ────────────────────────────────────────────────────────

def _simulate_gene(gene_def: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    patients = []
    sev_w = gene_def.get("severity_weights", [0.20, 0.55, 0.25])
    severities = rng.choices(["Mild", "Moderate", "Severe"], weights=sev_w, k=n)

    # Retinal-specific clinical feature probabilities per gene
    feature_map = {
        "RPGR": {
            "night_blindness": 0.98, "photophobia": 0.35, "nystagmus": 0.10,
            "hearing_loss": 0.00, "color_blind": 0.10, "schisis": 0.00,
            "macular_primary": 0.15, "gene_therapy_eligible": 0.55,
        },
        "USH2A": {
            "night_blindness": 0.95, "photophobia": 0.30, "nystagmus": 0.05,
            "hearing_loss": 0.99, "color_blind": 0.00, "schisis": 0.00,
            "macular_primary": 0.10, "gene_therapy_eligible": 0.10,
        },
        "ABCA4": {
            "night_blindness": 0.40, "photophobia": 0.50, "nystagmus": 0.05,
            "hearing_loss": 0.00, "color_blind": 0.30, "schisis": 0.00,
            "macular_primary": 0.99, "gene_therapy_eligible": 0.20,
        },
        "RDH12": {
            "night_blindness": 0.99, "photophobia": 0.99, "nystagmus": 0.98,
            "hearing_loss": 0.00, "color_blind": 0.50, "schisis": 0.00,
            "macular_primary": 0.30, "gene_therapy_eligible": 0.05,
        },
        "PRPF31": {
            "night_blindness": 0.90, "photophobia": 0.20, "nystagmus": 0.05,
            "hearing_loss": 0.00, "color_blind": 0.05, "schisis": 0.00,
            "macular_primary": 0.10, "gene_therapy_eligible": 0.08,
        },
        "EYS": {
            "night_blindness": 0.97, "photophobia": 0.15, "nystagmus": 0.05,
            "hearing_loss": 0.00, "color_blind": 0.08, "schisis": 0.00,
            "macular_primary": 0.10, "gene_therapy_eligible": 0.05,
        },
        "CNGB3": {
            "night_blindness": 0.05, "photophobia": 0.99, "nystagmus": 0.99,
            "hearing_loss": 0.00, "color_blind": 0.99, "schisis": 0.00,
            "macular_primary": 0.10, "gene_therapy_eligible": 0.80,
        },
        "RS1": {
            "night_blindness": 0.20, "photophobia": 0.20, "nystagmus": 0.15,
            "hearing_loss": 0.00, "color_blind": 0.10, "schisis": 0.99,
            "macular_primary": 0.95, "gene_therapy_eligible": 0.30,
        },
    }

    gene_name = gene_def["gene"]
    fm = feature_map.get(gene_name, {
        "night_blindness": 0.80, "photophobia": 0.30, "nystagmus": 0.10,
        "hearing_loss": 0.00, "color_blind": 0.10, "schisis": 0.00,
        "macular_primary": 0.20, "gene_therapy_eligible": 0.10,
    })

    # Age at diagnosis (years)
    age_ranges = {
        "RPGR": (5, 25), "USH2A": (10, 30), "ABCA4": (8, 25), "RDH12": (0, 3),
        "PRPF31": (15, 55), "EYS": (8, 30), "CNGB3": (0, 3), "RS1": (3, 15),
    }
    age_min, age_max = age_ranges.get(gene_name, (5, 30))

    for j in range(n):
        sev = severities[j]
        age_diag_yr = round(rng.uniform(age_min, age_max), 1)

        night_blindness = rng.random() < fm["night_blindness"]
        photophobia = rng.random() < fm["photophobia"]
        nystagmus = rng.random() < fm["nystagmus"]
        hearing_loss = rng.random() < fm["hearing_loss"]
        color_blind = rng.random() < fm["color_blind"]
        schisis = rng.random() < fm["schisis"]
        macular_primary = rng.random() < fm["macular_primary"]
        gene_therapy_eligible = rng.random() < fm["gene_therapy_eligible"]

        # ERG pattern
        if gene_name == "CNGB3":
            erg_pattern = "Normal scotopic / Extinguished photopic"
        elif gene_name == "RS1":
            erg_pattern = "Electronegative (b<a)"
        elif gene_name == "RDH12":
            erg_pattern = "Extinguished rod+cone"
        elif gene_name in ("RPGR", "EYS", "USH2A", "PRPF31"):
            erg_pattern = "Reduced rod+cone (rod>cone)"
        elif gene_name == "ABCA4":
            erg_pattern = "Mfocal macular reduced; peripheral variable"
        else:
            erg_pattern = "Reduced rod+cone"

        # VA estimate
        if gene_name in ("RDH12", "CNGB3"):
            va_logmar = round(rng.uniform(0.8, 1.3), 2)
        elif gene_name in ("RS1", "ABCA4"):
            va_logmar = round(rng.uniform(0.3, 0.8), 2)
        elif sev == "Severe":
            va_logmar = round(rng.uniform(0.5, 1.2), 2)
        elif sev == "Moderate":
            va_logmar = round(rng.uniform(0.2, 0.6), 2)
        else:
            va_logmar = round(rng.uniform(0.0, 0.3), 2)

        # Vitamin A consideration
        if gene_name == "ABCA4":
            vit_a_status = "ABSOLUTELY CI (mega-dose supplement)"
        elif gene_name == "RDH12":
            vit_a_status = "SAFE (beneficial)"
        else:
            vit_a_status = "Investigational / Use at discretion"

        patients.append({
            "id": f"{gene_name}-{seed}-{j+1:02d}",
            "gene": gene_name,
            "seed": seed,
            "age_diagnosis_yr": age_diag_yr,
            "severity": sev,
            "night_blindness": night_blindness,
            "photophobia": photophobia,
            "nystagmus": nystagmus,
            "hearing_loss": hearing_loss,
            "color_blind": color_blind,
            "schisis": schisis,
            "macular_primary": macular_primary,
            "gene_therapy_eligible": gene_therapy_eligible,
            "erg_pattern": erg_pattern,
            "va_logmar": va_logmar,
            "vit_a_status": vit_a_status,
        })
    return patients


def _cohort_stats(patients: list) -> dict:
    n = len(patients)
    if n == 0:
        return {}

    night_bl_n = sum(p["night_blindness"] for p in patients)
    photophobia_n = sum(p["photophobia"] for p in patients)
    nystagmus_n = sum(p["nystagmus"] for p in patients)
    hearing_loss_n = sum(p["hearing_loss"] for p in patients)
    color_blind_n = sum(p["color_blind"] for p in patients)
    schisis_n = sum(p["schisis"] for p in patients)
    macular_n = sum(p["macular_primary"] for p in patients)
    gt_eligible_n = sum(p["gene_therapy_eligible"] for p in patients)

    mean_age = round(sum(p["age_diagnosis_yr"] for p in patients) / n, 1)
    mean_va = round(sum(p["va_logmar"] for p in patients) / n, 2)

    sev_mild = sum(p["severity"] == "Mild" for p in patients)
    sev_mod = sum(p["severity"] == "Moderate" for p in patients)
    sev_sev = sum(p["severity"] == "Severe" for p in patients)

    return {
        "n": n,
        "night_blindness_pct": round(100 * night_bl_n / n, 1),
        "photophobia_pct": round(100 * photophobia_n / n, 1),
        "nystagmus_pct": round(100 * nystagmus_n / n, 1),
        "hearing_loss_pct": round(100 * hearing_loss_n / n, 1),
        "color_blind_pct": round(100 * color_blind_n / n, 1),
        "schisis_pct": round(100 * schisis_n / n, 1),
        "macular_primary_pct": round(100 * macular_n / n, 1),
        "gene_therapy_eligible_pct": round(100 * gt_eligible_n / n, 1),
        "mean_age_diagnosis_yr": mean_age,
        "mean_va_logmar": mean_va,
        "severity_mild_pct": round(100 * sev_mild / n, 1),
        "severity_moderate_pct": round(100 * sev_mod / n, 1),
        "severity_severe_pct": round(100 * sev_sev / n, 1),
    }


def _all_patients() -> list:
    all_pts = []
    for i, ge in enumerate(RETINAL_GENES):
        seed = SEED_BASE + i
        pts = _simulate_gene(ge, seed, 40)
        all_pts.extend(pts)
    return all_pts


# ─── Public API functions ──────────────────────────────────────────────────────

def get_overview() -> dict:
    all_pts = _all_patients()
    agg = _cohort_stats(all_pts)
    return {
        "atlas_name": "Retinal Dystrophy Atlas",
        "atlas_subtitle": "Complete 8-Gene Hereditary Retinal Dystrophy Atlas",
        "gene_count": 8,
        "n_genes": 8,
        "n_patients": len(all_pts),
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
        "genes": [g["gene"] for g in RETINAL_GENES],
        "description": (
            "The Retinal Dystrophy Atlas covers 8 clinically actionable genes across the full spectrum of hereditary "
            "retinal dystrophies: "
            "RPGR (RP3/XLRP — most common X-linked RP, ~70% of XLRP; ORF15 exon standard NGS MISSES; "
            "long-read sequencing mandatory; AAV5-RPGR gene therapy Phase I/II Beacon study, MeiraGTx), "
            "USH2A (Usher syndrome type IIA — most common Usher gene >50%; RP + congenital SNHL + normal vestibular; "
            "p.Glu767Sfs*21 European founder; cochlear implant EFFECTIVE), "
            "ABCA4 (Stargardt disease STGD1 — most common AR juvenile macular dystrophy; "
            "VITAMIN A SUPPLEMENTS ABSOLUTELY CONTRAINDICATED — worsen bisretinoid A2E accumulation; "
            "pisciform flecks + silent choroid PATHOGNOMONIC), "
            "RDH12 (LCA13 — severe infantile onset; ERG extinguished by age 2; vitamin A supplementation SAFE "
            "— contrast to ABCA4 where it is CI), "
            "PRPF31 (RP11 — AD spliceopathy; INCOMPLETE PENETRANCE; MLPA MANDATORY — 15% are large deletions "
            "missed by standard sequencing; penetrance explained by CNOT3-modulated normal allele expression), "
            "EYS (RP25 — AR; largest non-dystrophin retinal gene 3 Mb; most common AR RP in Japanese 30–35%; "
            "MLPA MANDATORY — 15–20% large deletions; gene therapy challenging due to size), "
            "CNGB3 (achromatopsia ACHM3 — most common achromatopsia gene ~50%; normal rod ERG + extinguished "
            "cone ERG diagnostic; botaretigene sparoparvovec/BIIB112 gene therapy Phase III — most promising "
            "near-future approval among this atlas), "
            "and RS1 (X-linked retinoschisis — spoke-wheel foveal schisis OCT PATHOGNOMONIC; "
            "electronegative ERG b<a; CARBONIC ANHYDRASE INHIBITORS ABSOLUTELY CONTRAINDICATED — "
            "paradoxically worsen retinoschisis, opposite of CME). "
            "320 patients (8 × 40, seeds 1246–1253)."
        ),
        "aggregate_clinical": agg,
        "drug_alerts": [
            {
                "title": "ABCA4-Stargardt: VITAMIN A MEGA-DOSE SUPPLEMENTS ABSOLUTELY CI — worsen bisretinoid/A2E accumulation",
                "body": (
                    "ABCA4 transports N-retinylidene-PE (NRPE) across disc membrane. LOF → A2E bisretinoid accumulates → "
                    "RPE lipofuscin → RPE cell death. "
                    "High-dose vitamin A → more all-trans retinal substrate → more NRPE → MORE A2E → faster degeneration. "
                    "DIETARY vitamin A (from food) is safe. HIGH-DOSE SUPPLEMENT CAPSULES are CONTRAINDICATED. "
                    "Multi-vitamins containing vitamin A or beta-carotene should also be avoided. "
                    "Advise explicitly at every visit. Patient should inform all other prescribers."
                ),
            },
            {
                "title": "RS1-XLRS: CARBONIC ANHYDRASE INHIBITORS (dorzolamide, brinzolamide, acetazolamide) ABSOLUTELY CI — paradoxically worsen retinoschisis",
                "body": (
                    "Carbonic anhydrase inhibitors are used for cystoid macular edema (CME) in many conditions. "
                    "In X-linked retinoschisis (RS1 LOF), CAIs PARADOXICALLY INCREASE cystic schisis volume. "
                    "Dorzolamide eye drops (Trusopt), brinzolamide (Azopt), and oral acetazolamide (Diamox) are "
                    "ALL ABSOLUTELY CONTRAINDICATED in XLRS. "
                    "Mechanism: altered fluid dynamics at INL. "
                    "Never prescribe CAI for 'macular oedema' without first confirming the diagnosis is NOT XLRS. "
                    "Carry patient alert card documenting this contraindication."
                ),
            },
            {
                "title": "RPGR: ORF15 EXON STANDARD NGS MISSES — long-read sequencing or ORF15-specific PCR/Sanger MANDATORY",
                "body": (
                    "RPGR ORF15 exon is a highly repetitive purine-rich sequence encoding a Glu-Gly repeat region. "
                    "Standard NGS has high error rate here → insertions/deletions in ORF15 are MISSED or misreported. "
                    "ORF15 mutations account for ~60–70% of all RPGR pathogenic variants. "
                    "A negative standard RPGR panel report does NOT exclude RPGR. "
                    "ORDER: long-read sequencing (PacBio/Nanopore ORF15 amplicon) OR ORF15-specific PCR + Sanger confirmation. "
                    "All XLRP patients with negative standard panel must have ORF15-dedicated testing."
                ),
            },
            {
                "title": "PRPF31 + EYS: MLPA MANDATORY — large deletions missed by standard sequencing (15–20% of cases)",
                "body": (
                    "PRPF31 (RP11): 15% of pathogenic variants are large chromosomal deletions at 19q13.42. "
                    "EYS (RP25): 15–20% of pathogenic variants are large genomic deletions at 6q12 (largest retinal gene: 3 Mb). "
                    "Standard panel sequencing misses both. MLPA (multiplex ligation-dependent probe amplification) is required. "
                    "Use gene-specific MLPA probe sets (PRPF31-MLPA and EYS-MLPA are distinct). "
                    "A single allele found by sequencing + negative MLPA still requires deep intronic variant testing. "
                    "Never conclude 'monoallelic' without MLPA."
                ),
            },
            {
                "title": "CNGB3-Achromatopsia: Gene therapy (botaretigene sparoparvovec/BIIB112) Phase III — most promising near-future approval",
                "body": (
                    "Botaretigene sparoparvovec (AAV8-CNGB3, BIIB112) by MeiraGTx/Janssen is in Phase III trial for CNGB3-achromatopsia. "
                    "Phase I/II: partial recovery of photopic (cone) ERG + improved light-adapted visual acuity in treated patients. "
                    "Subretinal injection; eligibility criteria: biallelic CNGB3 pathogenic variants + cone ERG extinguished/severely reduced. "
                    "Refer eligible patients to trial sites NOW. "
                    "This is the most advanced gene therapy among the 8 genes in this atlas — "
                    "CNGA3-achromatopsia (ACHM2) is in parallel gene therapy development."
                ),
            },
            {
                "title": "USH2A: Cochlear implant EFFECTIVE for HL component — normal vestibular function means NO balance risk",
                "body": (
                    "Usher syndrome type IIA (USH2A): RP + congenital moderate-severe SNHL + NORMAL vestibular function. "
                    "Normal vestibular function means cochlear implant does NOT add balance risk post-surgery. "
                    "Cochlear implant is recommended and effective for the hearing component. "
                    "Early childhood CI gives best speech and language outcomes. "
                    "CONTRAST with USH1 (MYO7A etc.): profound SNHL + ABSENT vestibular — "
                    "CI still safe but patients already have vestibular deficit; balance counselling needed."
                ),
            },
            {
                "title": "RDH12-LCA13: VITAMIN A SUPPLEMENTATION IS SAFE — contrast to ABCA4 where it is absolutely CI",
                "body": (
                    "RDH12 reduces all-trans retinal (atRAL) to all-trans retinol (atROL) in photoreceptor inner segment. "
                    "LOF → atRAL accumulates → photoreceptor toxicity. "
                    "Vitamin A supplementation provides atROL substrate which bypasses the RDH12 reduction step "
                    "and feeds the retinoid cycle via LRAT → RPE65. SAFE and potentially beneficial. "
                    "CONTRAST WITH ABCA4: in ABCA4 deficiency, extra vitamin A → more atRAL → more A2E bisretinoid → WORSE. "
                    "The visual cycle position of the defect determines vitamin A safety — know your gene."
                ),
            },
            {
                "title": "RS1-XLRS: Electronegative ERG (b-wave < a-wave) is DIAGNOSTIC — do not miss this pattern",
                "body": (
                    "Normal ERG: b-wave amplitude > a-wave. "
                    "a-wave = photoreceptor (outer retina) response; b-wave = inner nuclear layer/bipolar (inner retina) response. "
                    "In XLRS: retinoschisin LOF → INL structural failure → bipolar cell synaptic transmission impaired → "
                    "b-wave severely reduced; photoreceptors relatively preserved → a-wave near-normal. "
                    "Result: b < a = ELECTRONEGATIVE ERG. "
                    "This pattern should trigger: RS1 gene sequencing + OCT foveal schisis look. "
                    "Also seen in CSNB (congenital stationary night blindness) but without schisis on OCT."
                ),
            },
        ],
        "clinical_pearls": [
            "XLRP diagnosis: RPGR (~70% of XLRP) vs RP2 (~10%); always test ORF15 with long-read or dedicated PCR — standard NGS misses ORF15",
            "Usher syndrome hierarchy: USH1 (MYO7A — profound congenital HL + absent vestibular + RP) vs USH2 (USH2A — moderate HL + normal vestibular + RP) vs USH3 (CLRN1 — progressive HL)",
            "ABCA4 vitamin A CI vs RDH12 vitamin A SAFE: know the visual cycle step where each gene acts",
            "Achromatopsia: normal scotopic (rod) ERG + extinguished photopic (cone) ERG + nystagmus + photophobia = CNGB3 or CNGA3; BIIB112/botaretigene Phase III is most advanced retinal gene therapy here",
            "XLRS electronegative ERG (b<a) = look for foveal schisis on OCT; CAI ABSOLUTELY CI in XLRS",
            "MLPA is mandatory for PRPF31 (15% deletions) and EYS (15–20% deletions); sequencing alone is insufficient",
            "PRPF31 incomplete penetrance: 50–80%; unaffected obligate carriers explained by CNOT3-mediated high normal allele expression",
            "RDH12-LCA13: ERG extinguished rod AND cone by age 2 (pan-retinal) distinguishes from ABCA4-STGD1 (macular-selective ERG normal peripherally)",
            "LCA differential: CEP290-LCA10 (IVS26 deep intronic misses exome), GUCY2D-LCA1 (minimally reduced cone ERG initially), RPE65-LCA2 (TREATABLE with Luxturna — must distinguish from RDH12)",
            "EYS is most common AR RP in Japanese (30–35%): always test in Japanese AR RP patients; p.Tyr2935* and p.Gln2767* are Japanese founders"
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for i, ge in enumerate(RETINAL_GENES):
        seed = SEED_BASE + i
        pts = _simulate_gene(ge, seed, 40)
        stats = _cohort_stats(pts)
        result[ge["gene"]] = {
            "gene": ge["gene"],
            "protein": ge["protein"],
            "alias": ge["alias"],
            "aa": ge["aa"],
            "kDa": ge["kDa"],
            "locus": ge["locus"],
            "omim_gene": ge["omim_gene"],
            "omim_disease": ge["omim_disease"],
            "inheritance": ge["inheritance"],
            "gene_class": ge["gene_class"],
            "phenotype": ge["phenotype"],
            "hallmark": ge["hallmark"],
            "treatment_alert": ge["treatment_alert"],
            "key_ddx": ge["key_ddx"],
            "vision_pattern": ge["vision_pattern"],
            "erg_pattern": ge["erg_pattern"],
            "cohort_n": len(pts),
            "seed": seed,
            "stats": stats,
            "patients": pts,
        }
    return result


def get_definitions() -> dict:
    return {
        "atlas_name": "Retinal Dystrophy Atlas",
        "terms": [
            {
                "term": "Retinal dystrophy — classification",
                "definition": (
                    "Hereditary retinal dystrophies are classified by: "
                    "(1) TOPOGRAPHY — panretinal (RP-type: rod-cone dystrophy, peripheral first) vs "
                    "macular (central first, cone-dominant: Stargardt, Best, pattern) vs "
                    "cone/cone-rod (central vision lost first) vs stationary (non-progressive: CSNB, achromatopsia); "
                    "(2) INHERITANCE — AR, AD, X-linked (XL), mitochondrial; "
                    "(3) SYNDROMIC vs NON-SYNDROMIC — USH2A adds HL; Bardet-Biedl adds obesity + renal; "
                    "Kearns-Sayre adds cardiomyopathy + CPEO (mitochondrial); "
                    "(4) FUNCTIONAL CELL TYPE AFFECTED — rods (scotopic, night vision) vs cones "
                    "(photopic, colour, acuity) vs both (cone-rod or rod-cone dystrophy)."
                ),
            },
            {
                "term": "Electroretinogram (ERG) — interpretation",
                "definition": (
                    "Full-field ERG records retinal electrical responses to light flash. "
                    "KEY WAVEFORMS: "
                    "a-wave (negative): photoreceptor hyperpolarisation (outer retina); "
                    "b-wave (positive): Müller cell / bipolar cell depolarisation (inner retina); "
                    "normal b > a. "
                    "SCOTOPIC (dark-adapted): tests rods; "
                    "PHOTOPIC (light-adapted): tests cones; "
                    "ELECTRONEGATIVE ERG: b-wave < a-wave → selective inner retinal dysfunction → XLRS, CSNB; "
                    "EXTINGUISHED ERG: flat line rod+cone → severe panretinal disease → RDH12-LCA; "
                    "SELECTIVE PHOTOPIC EXTINCTION: cone loss with normal rod → achromatopsia (CNGB3); "
                    "MULTIFOCAL ERG (mfERG): maps macular function topographically → macular disease (STGD1)."
                ),
            },
            {
                "term": "Fundus autofluorescence (FAF)",
                "definition": (
                    "FAF uses short-wavelength (SW-FAF, 488 nm excitation) or near-infrared (NIR-FAF) light "
                    "to detect intrinsic fluorophores in RPE lipofuscin. "
                    "Normal FAF: uniform low-level autofluorescence (disc = dark; fovea = slightly dark). "
                    "INCREASED FAF: lipofuscin accumulation (A2E bisretinoid in ABCA4-STGD1); "
                    "DECREASED FAF (dark/hypo-AF): RPE cell loss (atrophy) — dead RPE cannot produce AF; "
                    "STGD1 PATTERN: flecks hyper-AF → central hypo-AF (dead RPE) + peripheral hyper-AF (live RPE + lipofuscin); "
                    "RPGR female carrier: bilateral zig-zag meridional hyper-AF pattern — pathognomonic; "
                    "RP-pattern: peripheral hypo-AF ring (dead peripheral RPE) + central preserved."
                ),
            },
            {
                "term": "Gene therapy vectors in retinal dystrophy",
                "definition": (
                    "ADENO-ASSOCIATED VIRUS (AAV): primary vector for retinal gene therapy; "
                    "serotypes: AAV2 (fovea; classic), AAV5 (photoreceptors; RPGR Beacon), "
                    "AAV8 (photoreceptors + RPE; CNGB3/botaretigene), AAV9 (pan-retinal; RS1/RGX-121); "
                    "CARGO LIMIT: ~4.7 kb insert — limits therapy for large genes (EYS 3165 aa >> limit); "
                    "DELIVERY: subretinal (under RPE; direct photoreceptor contact) vs intravitreal (less invasive; "
                    "larger cargo diffusion challenges); "
                    "APPROVED: voretigene neparvovec-rzyl (Luxturna, AAV2-RPE65, subretinal) for RPE65-LCA2/RP20; "
                    "PHASE III: botaretigene sparoparvovec (BIIB112/CNGB3-ACHM3); "
                    "PHASE I/II: AAV5-RPGR (MeiraGTx Beacon); RGX-121 intravitreal (RS1-XLRS); "
                    "DUAL AAV: split-intein approach for large genes (EYS, ABCA4); research-stage."
                ),
            },
            {
                "term": "Achromatopsia — complete vs incomplete",
                "definition": (
                    "COMPLETE ACHROMATOPSIA: total absence of cone function; "
                    "triad — nystagmus (onset 2–4 months) + photophobia (severe) + VA 20/200 + absent colour discrimination; "
                    "ERG: normal scotopic (rod) + extinguished photopic (cone); "
                    "genes: CNGB3 (~50%), CNGA3 (~25%), GNAT2 (~2%), PDE6C, PDE6H, ATF6; "
                    "INCOMPLETE ACHROMATOPSIA: hypomorphic alleles → residual cone function → some colour; "
                    "CNGB3 hypomorphic variants → mild colour perception; must distinguish from complete; "
                    "BLUE-CONE MONOCHROMACY (BCM): X-linked; deletion of OPN1LW + OPN1MW (Xq28); "
                    "ONLY short-wavelength (blue) cone function preserved; males preferentially affected; "
                    "distinguish from ACHM by X-linked pattern + some colour discrimination (blue-yellow)."
                ),
            },
            {
                "term": "X-linked retinoschisis — retinoschisin and electronegative ERG",
                "definition": (
                    "Retinoschisin (RS1): secreted octameric discoidal protein; discoidin domain; "
                    "maintains INL cell adhesion at OPL (outer plexiform layer) synapse. "
                    "LOF → INL splits → RETINOSCHISIS (spoke-wheel OCT pattern at fovea). "
                    "ELECTRONEGATIVE ERG: b << a; "
                    "b-wave reflects INL/bipolar signal (lost in XLRS); "
                    "a-wave reflects photoreceptor (relatively preserved in XLRS); "
                    "other causes of electronegative ERG: CSNB (congenital stationary night blindness: "
                    "CACNA1A incomplete CSNB; GRM6/TRPM1 complete CSNB — no schisis on OCT). "
                    "CARBONIC ANHYDRASE INHIBITORS: PARADOXICALLY WORSEN XLRS — "
                    "increase schisis cavity volume (opposite of effect in CME)."
                ),
            },
            {
                "term": "Usher syndrome types I / II / III",
                "definition": (
                    "USHER TYPE I: congenital PROFOUND sensorineural HL + ABSENT vestibular function "
                    "(delayed walking, vestibular hypofunction) + RP; "
                    "genes: MYO7A (most common), CDH23, PCDH15, SANS/USH1G, CIB2; "
                    "CI safe but vestibular already absent; "
                    "USHER TYPE II: congenital MODERATE-SEVERE HL + NORMAL vestibular + RP (onset 2nd decade); "
                    "genes: USH2A (>80% of USH2), ADGRV1/VLGR1, WHRN; "
                    "CI recommended + effective; normal vestibular → no balance risk post-CI; "
                    "USHER TYPE III: PROGRESSIVE HL (not congenital, worsens over time) + RP + variable vestibular; "
                    "gene: CLRN1 (DFNB80); rare; "
                    "USH2 is MOST COMMON (USH2A alone is >50% of all Usher syndrome)."
                ),
            },
            {
                "term": "ABCA4 bisretinoid / A2E pathway",
                "definition": (
                    "Visual cycle: 11-cis retinal → photoactivated → all-trans retinal (atRAL) + opsin; "
                    "atRAL reacts with phosphatidylethanolamine (PE) in disc → N-retinylidene-PE (NRPE); "
                    "ABCA4: flips NRPE from disc lumen to cytoplasmic leaflet → hydrolysed → atRAL exported → "
                    "reduced by RDH8/12 → retinol → recycled via RPE; "
                    "ABCA4 LOF → NRPE not flipped → condenses to A2PE → phagocytosed to RPE → "
                    "A2PE hydrolysed to A2E (di-retinal pyridinium bisretinoid); "
                    "A2E accumulates in RPE lysosomes → lysosomal dysfunction → complement activation → "
                    "RPE cell death → photoreceptor death → central vision loss; "
                    "VITAMIN A CI: more vitamin A → more atRAL → more NRPE → MORE A2E → faster degeneration."
                ),
            },
            {
                "term": "RPGR ORF15 exon — why standard NGS fails",
                "definition": (
                    "RPGR ORF15: alternative terminal exon encoding a Glu-Gly-rich repeat domain (purine-rich DNA); "
                    "repetitive purine-rich sequence causes: "
                    "(1) PCR polymerase slippage → insertion/deletion artefacts; "
                    "(2) library preparation artefacts; "
                    "(3) short reads misalign to repeats; "
                    "CONSEQUENCE: standard Illumina short-read NGS gives HIGH ERROR RATE at ORF15; "
                    "pathogenic frameshift/deletion in ORF15 may appear as sequencing artefact → MISSED; "
                    "ORF15 mutations = ~60–70% of all RPGR pathogenic variants; "
                    "SOLUTIONS: long-read sequencing (PacBio SMRT amplicon; Nanopore) OR "
                    "ORF15-dedicated PCR with Sanger sequencing of overlapping amplicons."
                ),
            },
            {
                "term": "Incomplete penetrance in PRPF31-RP11 spliceopathy",
                "definition": (
                    "PRPF31 haploinsufficiency → RP11 (AD RP) with INCOMPLETE PENETRANCE (50–80%); "
                    "MECHANISM: normal allele expression level determines penetrance; "
                    "high expression of normal PRPF31 allele → sufficient spliceosome supply → "
                    "haploinsufficiency threshold NOT reached → unaffected obligate carrier; "
                    "regulators of normal allele expression: CNOT3 (Ccr4-Not deadenylase complex) → "
                    "promotes PRPF31 transcription; lncRNA PRPF31-AS1 → modulates; allele-specific miRNA silencing; "
                    "CLINICAL: 'skipped generation' pedigree pattern; "
                    "unaffected grandparent + affected grandchild with the same allele; "
                    "risk estimation: each child of carrier has 50% of inheriting allele × penetrance (50–80%); "
                    "genetic counselling complexity: unaffected relative carrying allele still at risk."
                ),
            },
            {
                "term": "MLPA for retinal gene large deletions",
                "definition": (
                    "MLPA (multiplex ligation-dependent probe amplification): detects copy number variation (CNV) "
                    "and large exonic deletions/duplications undetectable by sequencing-based methods. "
                    "KEY RETINAL GENES requiring MLPA: "
                    "PRPF31 (RP11, 19q13.42): 15% of cases are large deletions; "
                    "EYS (RP25, 6q12): 15–20% of cases are large deletions (largest retinal gene 3 Mb); "
                    "USH2A: 5–8% large deletions (single allele found → MLPA for second); "
                    "RS1: exon deletions (MLPA for exons 3–6 missense hotspot region); "
                    "standard exome sequencing does NOT reliably detect large intragenic deletions; "
                    "DIAGNOSTIC PITFALL: single allele found by sequencing in AR gene → conclude 'monoallelic' without MLPA → WRONG."
                ),
            },
            {
                "term": "Rod-cone vs cone-rod dystrophy",
                "definition": (
                    "ROD-CONE DYSTROPHY (RP-pattern): rods affected first → night blindness precedes central loss; "
                    "peripheral field lost before central; "
                    "genes: RPGR, USH2A, EYS, PRPF31, CNGB1, PDE6A, PDE6B, RHO; "
                    "CONE-ROD DYSTROPHY (CRD): cones affected first → central vision + colour loss first; "
                    "peripheral field lost later; genes: ABCA4 (late stage), RPGR exon1-14, GUCY2D, PROM1, CDHR1; "
                    "MACULAR DYSTROPHY: selective central/macular involvement; ABCA4-STGD1, BEST1, PRPH2; "
                    "STATIONARY: non-progressive; achromatopsia (CNGB3/CNGA3), CSNB (GRM6/TRPM1/CACNA1A); "
                    "ERG guides classification: scotopic (rod) vs photopic (cone) which is worse."
                ),
            },
            {
                "term": "Foveal hypoplasia — achromatopsia vs other causes",
                "definition": (
                    "Foveal hypoplasia on OCT: absent IS/OS (ellipsoid zone) signal at fovea; "
                    "inner retinal layers extend into fovea (absence of foveal pit); "
                    "CAUSES IN RETINAL DYSTROPHY: "
                    "(1) Achromatopsia (CNGB3/CNGA3): cone IS/OS absent at fovea (cone-specific); "
                    "rod IS/OS normal (pericentral); "
                    "(2) Albinism (OCA/OA): foveal hypoplasia from abnormal decussation; "
                    "misrouted optic nerve fibres (OCT macular + VEP); "
                    "(3) PAX6 mutations (aniridia): foveal hypoplasia + iris agenesis; "
                    "DISTINCTION: achromatopsia OCT shows normal rod IS/OS with absent cone signal at fovea; "
                    "albinism shows undifferentiated foveal pit on structural OCT."
                ),
            },
            {
                "term": "Vitamin A and the visual cycle — gene-specific CI vs SAFE distinction",
                "definition": (
                    "Visual cycle enzyme positions determine vitamin A supplementation safety: "
                    "SAFE (or beneficial): "
                    "RDH12-LCA13 — reduces atRAL→atROL; supplement provides atROL upstream of block; "
                    "RPE65-LCA2 — 11-cis retinol production; supplement feeds substrate (benefit modest); "
                    "ABSOLUTELY CONTRAINDICATED (high-dose supplement): "
                    "ABCA4-STGD1 — supplement → more atRAL → more NRPE → more A2E bisretinoid → faster RPE death; "
                    "GENERAL RP (uncertain): RHO-RP4, PRPF31-RP11, EYS-RP25 — evidence limited; "
                    "Berson 1993 recommended 15,000 IU/day for selected RP; not universally accepted; "
                    "RULE: know your gene's visual cycle position before recommending or prohibiting vitamin A."
                ),
            },
            {
                "term": "Pisciform flecks and silent choroid — ABCA4-Stargardt PATHOGNOMONIC signs",
                "definition": (
                    "PISCIFORM FLECKS: yellow-white deposits at RPE-Bruch's membrane interface; "
                    "fish-tail (pisciform) shape; scattered at posterior pole + mid-periphery; "
                    "represent photoreceptor outer segments shed into RPE full of A2E-laden lipofuscin; "
                    "PATHOGNOMONIC for ABCA4-STGD1 when combined with macular atrophy; "
                    "SILENT CHOROID on FA: lipofuscin-laden RPE blocks choroidal fluorescence → "
                    "choroidal vessels not visible (choroidal fluorescence normally seen through normal RPE); "
                    "BEATEN BRONZE MACULAR SHEEN: clinical fundoscopy — metal-like sheen at macula from dense lipofuscin; "
                    "differential for flecks: fundus flavimaculatus (late-onset ABCA4 variant); "
                    "North Carolina macular dystrophy (MCDR1, AD, chr6 non-coding variant — drusen-like, not flecks); "
                    "PRPH2 pattern dystrophy (butterfly pattern — different shape to pisciform)."
                ),
            },
        ]
    }
