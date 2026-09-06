#!/usr/bin/env python3
"""Inherited Retinal Dystrophy Atlas — Complete 8-Gene IRD Atlas
RPGR   (RP GTPase regulator; ~815 aa; Xp11.23; XLRP RP3 — most common X-linked RP; ORF15 hotspot; gene therapy trials) ·
ABCA4  (ABCA4 transporter; ~2273 aa; 1p22.1; Stargardt STGD1 + CRD + arRP — Vitamin A ABSOLUTELY CONTRAINDICATED) ·
USH2A  (Usherin; ~5202 aa; 1q41; Usher syndrome 2A USH2A — most common AR-RP; 80% of USH2; SNHL + RP) ·
PRPF31 (Pre-mRNA processing 31; ~499 aa; 19q13.42; adRP13 — haploinsufficiency; INCOMPLETE PENETRANCE; modifier gene) ·
CEP290 (Centrosomal protein 290; ~2479 aa; 12q21.32; LCA10 — most common LCA gene; IVS26+1655A>G FOUNDER; EDIT-101) ·
MYO7A  (Myosin VIIA; ~2215 aa; 11q13.5; Usher syndrome 1B USH1B — most common USH1; congenital profound deafness + RP + vestibular areflexia) ·
CRB1   (Crumbs homologue 1; ~1406 aa; 1q31.3; LCA8 + RP12 — thickened retina OCT PATHOGNOMONIC; paravenous RP) ·
BEST1  (Bestrophin 1; ~585 aa; 11q12.3; Best vitelliform macular dystrophy VMD2 — ERG normal EOG Arden ratio <1.5 PATHOGNOMONIC)
320-patient aggregate cohort (8 × 40, seeds 1094–1101)
"""

import random

SEED_BASE = 1094

IRD_GENES = [
    # ── RPGR — X-linked RP ───────────────────────────────────────────────────
    {
        "gene": "RPGR", "protein": "Retinitis Pigmentosa GTPase Regulator (RPGR)",
        "alias": "RPGR; OMIM gene 312610; Xp11.23; ~815 aa; XLRP RP3 (OMIM #300455); XLR; most common X-linked IRD; ORF15 exon hotspot ~70% of RPGR mutations",
        "aa": "~815 aa", "kDa": "~90 kDa",
        "mechanism": (
            "RPGR encodes retinitis pigmentosa GTPase regulator, a ciliary protein essential for "
            "photoreceptor maintenance. It localises to the connecting cilium (transition zone) of "
            "rod and cone photoreceptors, where it acts as a molecular scaffold for intraflagellar "
            "transport and membrane protein trafficking. "
            "NORMAL FUNCTION: RPGR-RCC1-like domain (RLD) interacts with GTPases; "
            "the ORF15 isoform (RPGR-ORF15) is the major retinal isoform and contains a glutamic "
            "acid-rich repeat region encoded by exon ORF15. "
            "XLRP PATHOMECHANISM: hemizygous loss-of-function variants → connecting cilium "
            "dysfunction → failure to transport opsins and membrane proteins to outer segment → "
            "outer segment degeneration → rod cell death followed by cone cell death. "
            "ORF15 HOTSPOT: ~70% of RPGR mutations cluster in the purine-rich ORF15 region; "
            "this region is prone to replication errors and requires dedicated sequencing protocols "
            "(standard NGS often misses ORF15 indels — repeat-enriched sequencing or long-read NGS mandatory). "
            "CARRIER FEMALES: most are unaffected carriers; ~15–25% develop mild-moderate RP "
            "from skewed X-inactivation (Lyon hypothesis) — carriers should have dilated fundus exam."
        ),
        "disease_type": "X-linked Retinitis Pigmentosa RP3 (XLRP; RPGR LOF; ORF15 hotspot ~70%; rod-cone dystrophy; males severe; carrier females ~15-25% symptomatic)",
        "locus": "Xp11.23", "omim_gene": 312610, "omim_disease": 300455,
        "inheritance": (
            "X-LINKED RECESSIVE (XLRP): hemizygous males severely affected; "
            "heterozygous females are carriers (usually unaffected, but 15–25% develop mild-moderate RP "
            "from skewed X-inactivation — all carrier females need dilated fundus examination). "
            "XLRP accounts for ~10–15% of all RP. RPGR mutations account for ~70–75% of XLRP. "
            "FAMILY SCREENING: obligate carrier mothers → 50% of sons affected; 50% of daughters are carriers. "
            "NO MALE-TO-MALE TRANSMISSION (distinguishes X-linked from AD). "
            "GENETIC TESTING CAVEAT: ORF15 mutations missed by standard NGS — "
            "dedicated ORF15 sequencing or long-read NGS MANDATORY. "
            "FEMALE CARRIERS: dilated fundus exam + ERG annually; some develop RP."
        ),
        "phenotype": (
            "X-LINKED RETINITIS PIGMENTOSA 3 (RP3): "
            "ONSET: childhood — typically night blindness (nyctalopia) in first decade. "
            "ROD-CONE PATTERN: rods affected first → night blindness → peripheral visual field loss "
            "(ring scotoma expanding centripetally) → tunnel vision → central vision loss (late). "
            "FUNDUS: bone-spicule pigmentation in mid-periphery PATHOGNOMONIC of RP; "
            "waxy disc pallor; attenuated retinal vessels. "
            "ERG: severely reduced or absent rod responses early; cone responses reduced later. "
            "SEVERITY: XLRP (RPGR) is among the most severe forms of RP — "
            "legal blindness by 3rd–4th decade in most hemizygous males. "
            "CYSTOID MACULAR OEDEMA (CMO): occurs in ~30% — anti-VEGF NOT indicated "
            "(CMO here is inflammatory, not neovascular); "
            "carbonic anhydrase inhibitors (acetazolamide, dorzolamide) for CMO. "
            "GENE THERAPY: clinical trials ongoing (subretinal AAV2/5-RPGR)."
        ),
        "treatment_options": [
            "Low vision rehabilitation: magnifiers, CCTV, screen readers, orientation and mobility training; "
            "MANDATORY referral to low vision specialist at diagnosis — maximise functional vision",
            "UV protection: dark-adapted state is compromised → sunglasses outdoors (UV-blocking wraparounds); "
            "maximum tinted lenses in bright light; dark adaptation training",
            "Carbonic anhydrase inhibitors (CAIs) for CMO: acetazolamide 250–500 mg PO daily OR "
            "dorzolamide eye drops 2% TID for cystoid macular oedema (CMO) — "
            "treat early before CMO causes irreversible macular damage; "
            "monitor renal function with systemic acetazolamide; "
            "topical preferred for long-term use",
            "Gene therapy trials: subretinal AAV-RPGR (multiple Phase I/II/III trials ongoing); "
            "eligibility assessment at IRD tertiary centre — enrol in registry (eyeGENE, ERED); "
            "RPGR gene therapy trials show cone preservation in early disease",
            "Genetic counselling: X-linked pattern — all sons of affected male are unaffected; "
            "all daughters are obligate carriers; maternal side cascade testing; "
            "carrier females need annual dilated fundus exam + ERG",
            "NO vitamin A for XLRP (evidence in ABCA4 strongly NEGATIVE; "
            "benefit data for RP limited — avoid unless specifically advised by IRD specialist); "
            "Omega-3 supplementation: some evidence for slower progression in RP — discuss with IRD team",
        ],
        "critical_avoid": (
            "RPGR/XLRP: DO NOT give high-dose Vitamin A — no proven benefit in XLRP; "
            "Vitamin A CONTRAINDICATED in ABCA4 (accelerates A2E toxin accumulation); "
            "CMO in XLRP is INFLAMMATORY not neovascular — anti-VEGF is INEFFECTIVE; "
            "use carbonic anhydrase inhibitors (acetazolamide/dorzolamide) for CMO. "
            "ORF15 SEQUENCING MANDATORY — standard NGS misses ~70% of pathogenic RPGR variants."
        ),
        "key_ddx": [
            "RP2-XLRP: second most common XLRP gene (RP2); clinically similar to RPGR-RP3; distinguished by sequencing",
            "RPGR-associated cone dystrophy: some RPGR variants cause cone-rod (not rod-cone) dystrophy",
            "Usher syndrome: RP + hearing loss — check audiogram in all RP patients; RPGR is RP-only",
            "X-linked retinoschisis (RS1): X-linked; schisis cavities on OCT; different pattern from RP",
            "PRPF31 adRP: autosomal dominant; often incomplete penetrance; no X-linked pattern",
        ],
        "severity_weights": {"Mild": 0.12, "Moderate": 0.35, "Severe": 0.53},
        "onset_age_range": (3, 12),
        "dx_lag_y": (2, 8),
        "drug_error_rate": 0.10,
        "gene_therapy_eligible_rate": 0.45,
        "hearing_loss_rate": 0.0,
        "vestibular_rate": 0.0,
        "progression_rate": 0.90,
        "cognitive_rate": 0.02,
        "first_line_drug": "Carbonic anhydrase inhibitor (CMO); low vision rehab",
    },
    # ── ABCA4 — Stargardt / STGD1 ──────────────────────────────────────────
    {
        "gene": "ABCA4", "protein": "ATP-Binding Cassette Subfamily A Member 4 (ABCA4)",
        "alias": "ABCA4; OMIM gene 601691; 1p22.1; ~2273 aa; Stargardt disease STGD1 (OMIM #248200); AR; most common inherited macular dystrophy",
        "aa": "~2273 aa", "kDa": "~256 kDa",
        "mechanism": (
            "ABCA4 encodes a photoreceptor-specific ATP-binding cassette transporter (flippase) "
            "localised to the disc membranes of rod and cone outer segments. "
            "NORMAL FUNCTION: ABCA4 transports N-retinylidene-phosphatidylethanolamine (N-Ret-PE) "
            "from the luminal leaflet to the cytoplasmic leaflet of disc membranes following "
            "phototransduction — facilitating retinoid recycling (visual cycle). "
            "STGD1 PATHOMECHANISM: biallelic ABCA4 LOF → N-Ret-PE accumulates in disc lumen → "
            "condenses to all-trans-retinal dimer (atRAL dimer) → phosphatidylethanolamine adduct "
            "A2PE → hydrolysed by RPE lysosomal phosphodiesterase → bisretinoid A2E "
            "(N-retinylidene-N-retinylethanolamine). "
            "A2E TOXICITY: A2E accumulates in RPE lysosomes → impairs lysosomal function → "
            "inhibits RPE65 (retinoid isomerase) → disrupts retinoid recycling → RPE cell death → "
            "cone photoreceptor degeneration → central vision loss. "
            "CRITICAL: Vitamin A supplementation ACCELERATES A2E accumulation (provides substrate) — "
            "ABSOLUTELY CONTRAINDICATED in ABCA4-Stargardt. "
            "LIGHT: all-trans-retinal (atRAL) generation from bleaching → more A2E substrate — "
            "light restriction (UV-blocking sunglasses MANDATORY) is evidence-based."
        ),
        "disease_type": "Stargardt Macular Dystrophy STGD1 (AR biallelic ABCA4; A2E accumulation; macular atrophy; VITAMIN A ABSOLUTELY CONTRAINDICATED; most common inherited macular disease)",
        "locus": "1p22.1", "omim_gene": 601691, "omim_disease": 248200,
        "inheritance": (
            "AUTOSOMAL RECESSIVE: biallelic pathogenic variants. "
            "ABCA4 is the most common AR IRD gene — STGD1 prevalence ~1:8,000–1:10,000. "
            "Most patients compound heterozygotes — two different pathogenic alleles. "
            "COMMON VARIANTS: p.Gly1961Glu (~12% of alleles); c.5461-10T>C (hypomorphic, often missed); "
            "p.Leu541Pro + p.Ala1038Val (common cis allele in European populations). "
            "CARRIER FREQUENCY: ~1:50–1:80 (European populations). "
            "FAMILY SCREENING: siblings 25% risk; parents are obligate carriers. "
            "EXTENDED HAPLOTYPE ANALYSIS: important for hypomorphic alleles; "
            "full ABCA4 gene sequencing including deep intronic variants required."
        ),
        "phenotype": (
            "STARGARDT MACULAR DYSTROPHY (STGD1): "
            "ONSET: typically first–second decade (juvenile form); adult-onset forms with milder alleles. "
            "MACULAR ATROPHY: bilateral central scotoma → reduced central visual acuity (VA); "
            "peripheral vision preserved until advanced disease. "
            "FUNDUS: macular atrophy (beaten-bronze or bull's-eye appearance); "
            "FLECKS — whitish-yellowish fundus flecks (lipofuscin deposits) in posterior pole and mid-periphery — "
            "characteristic of STGD1; absent in best disease. "
            "FAF (Fundus Autofluorescence): increased AF (lipofuscin/A2E) → later decreased AF (RPE atrophy) — "
            "FAF monitoring is standard of care. "
            "ERG: may be normal early; rod and cone involvement in advanced disease (cone > rod affected). "
            "OCT: early foveal thinning; later outer nuclear layer loss + IS/OS disruption + RPE loss. "
            "FLUORESCEIN ANGIOGRAPHY: 'silent' (dark) choroid — choroidal fluorescence blocked by A2E — "
            "PATHOGNOMONIC for STGD1. "
            "VISION OUTCOME: legal blindness (VA <6/60) in ~50% by age 50."
        ),
        "treatment_options": [
            "Vitamin A ABSOLUTELY CONTRAINDICATED — prescribing Vitamin A in ABCA4-Stargardt is a CRITICAL ERROR; "
            "A2E accumulation accelerated; do not prescribe ANY vitamin A or beta-carotene supplements; "
            "counsel patient to avoid multivitamins containing vitamin A; "
            "DARK ADAPTATION SUPPLEMENT: lutein/zeaxanthin reasonable (different pathway)",
            "UV-blocking sunglasses MANDATORY: all-trans-retinal (atRAL) bleaching generates A2E substrate; "
            "light restriction slows disease — evidence-based; "
            "prescribe dark-adapted UV wraparound sunglasses + wide-brim hat; "
            "indoor blue-light filters for screens; "
            "this is the ONLY proven disease-modifying intervention available now",
            "Low vision rehabilitation: magnifiers, CCTV, eccentric viewing training, screen readers; "
            "register with national low vision services; "
            "orientaton and mobility training early — before legal blindness",
            "Gene therapy / pharmacological trials: CPCB-RPE1 (RPE cell transplant); "
            "STG-001 (RNA therapy targeting hypomorphic c.5461-10T>C allele); "
            "emixustat (visual cycle modulator reducing A2E); "
            "fenretinide; STAR trial (subretinal gene therapy); "
            "enrol in eyeGENE and ERED registry — eligibility depends on variant and stage",
            "Genetic counselling: AR — 25% sibling risk; carrier parents unaffected; "
            "avoid vitamin A in ALL biallelic ABCA4 patients regardless of phenotypic severity; "
            "detailed variant characterisation required (hypomorphic alleles change prognosis)",
        ],
        "critical_avoid": (
            "ABCA4-STARGARDT: VITAMIN A ABSOLUTELY CONTRAINDICATED — accelerates A2E bisretinoid "
            "accumulation → faster RPE and photoreceptor degeneration. "
            "DO NOT prescribe retinol, beta-carotene, or multivitamins containing vitamin A. "
            "LIGHT RESTRICTION MANDATORY: UV-blocking wraparound sunglasses outdoors + indoor blue-light filters. "
            "Anti-VEGF ONLY for secondary choroidal neovascularisation (CNV) — NOT for primary STGD1 treatment."
        ),
        "key_ddx": [
            "Best vitelliform macular dystrophy (BEST1): EOG Arden ratio <1.5 PATHOGNOMONIC; ERG normal; no dark choroid",
            "Pattern dystrophy (PRPH2): butterfly-shaped or other patterns; no fundus flecks; AD",
            "Cone dystrophy (KCNV2, CNGA3): central loss + photophobia; ERG shows selective cone loss",
            "Age-related macular degeneration (AMD): onset >50; drusen; may have CNV; no flecks in periphery",
            "Fundus flavimaculatus: ABCA4 adult-onset variant with prominent flecks; same gene different severity",
        ],
        "severity_weights": {"Mild": 0.20, "Moderate": 0.45, "Severe": 0.35},
        "onset_age_range": (8, 20),
        "dx_lag_y": (2, 6),
        "drug_error_rate": 0.25,  # Vitamin A prescribed — critical common error
        "gene_therapy_eligible_rate": 0.35,
        "hearing_loss_rate": 0.0,
        "vestibular_rate": 0.0,
        "progression_rate": 0.85,
        "cognitive_rate": 0.01,
        "first_line_drug": "UV sunglasses (MANDATORY); Vitamin A ABSOLUTELY CI",
    },
    # ── USH2A — Usher syndrome 2A ───────────────────────────────────────────
    {
        "gene": "USH2A", "protein": "Usherin (USH2A)",
        "alias": "USH2A; OMIM gene 608400; 1q41; ~5202 aa; Usher syndrome type 2A (OMIM #276901); AR; most common Usher gene (~80% USH2); arRP without hearing loss",
        "aa": "~5202 aa (isoform b)", "kDa": "~614 kDa",
        "mechanism": (
            "USH2A encodes usherin, a large extracellular matrix protein with multiple laminin EGF "
            "and fibronectin type III domains. It localises to the periciliary membrane complex (PMC) "
            "at the base of the photoreceptor cilium and to the stereocilia ankle-link complex in cochlear hair cells. "
            "NORMAL FUNCTION: usherin forms the 'ankle-link' complex with VLGR1 and whirlin at "
            "the base of stereocilia → essential for stereocilia cohesion and mechanotransduction; "
            "in photoreceptors → structural scaffold at periciliary region → maintains OS integrity. "
            "USH2A PATHOMECHANISM: biallelic loss-of-function → stereocilia ankle-link deficiency → "
            "progressive hair-cell degeneration → sensorineural hearing loss; "
            "simultaneously → periciliary membrane dysfunction → photoreceptor degeneration → RP. "
            "ISOFORM B (retinal): the long isoform (isoform b; ~14.8 kb coding; largest mRNA in eye) "
            "contains the transmembrane domain and is the retina-specific isoform — "
            "mutations throughout cause RP; "
            "SHORT ISOFORM A (cochlea): may explain some genotype-phenotype correlations. "
            "RP WITHOUT DEAFNESS: biallelic missense in isoform-b-specific domains → "
            "non-syndromic arRP (USH2A accounts for ~15–20% of arRP)."
        ),
        "disease_type": "Usher Syndrome Type 2A (AR biallelic USH2A; RP + mild-moderate SNHL + normal vestibular function; 80% of USH2; also arRP without deafness)",
        "locus": "1q41", "omim_gene": 608400, "omim_disease": 276901,
        "inheritance": (
            "AUTOSOMAL RECESSIVE: biallelic pathogenic variants. "
            "USH2A accounts for ~30–40% of ALL Usher syndrome cases and ~80% of Usher type 2. "
            "COMMON VARIANTS: p.Glu767SerfsX21 (c.2299delG, ~15% of USH2A alleles — European founder); "
            "p.Cys759Phe (c.2276G>T); deep intronic variants (c.7595-2144A>G) — "
            "latter creates cryptic exon and requires RNA studies or retinal organoid systems to detect. "
            "FAMILY SCREENING: siblings 25% recurrence; carrier parents unaffected. "
            "NON-SYNDROMIC arRP: ~15–20% of arRP from USH2A — hearing test MANDATORY in all arRP patients. "
            "PRENATAL TESTING: available once both parental variants identified."
        ),
        "phenotype": (
            "USHER SYNDROME TYPE 2A (USH2A): "
            "HEARING: mild-to-moderate bilateral sensorineural hearing loss (SNHL) — congenital but "
            "often not noticed until school age; high-frequency loss predominates; "
            "VESTIBULAR: NORMAL (distinguishes USH2 from USH1 — USH1 has vestibular areflexia). "
            "VISION: retinitis pigmentosa typically onset in late childhood to teenage years — "
            "night blindness → peripheral field loss → central loss; "
            "LATER onset and SLOWER progression than USH1. "
            "CYSTOID MACULAR OEDEMA: occurs in ~30–40%; "
            "carbonic anhydrase inhibitors (acetazolamide) effective — treat early. "
            "NON-SYNDROMIC arRP (USH2A without hearing loss): "
            "~15–20% of USH2A patients have isolated RP without audible hearing loss on standard audiometry — "
            "pure-tone audiometry MANDATORY in all arRP patients (subtle high-frequency loss detectable). "
            "GENE THERAPY: QR-421a (antisense oligonucleotide targeting exon 13 of USH2A) in trials — "
            "eligibility: patients with at least one allele with exon 13 mutation."
        ),
        "treatment_options": [
            "Hearing aids: mild-moderate SNHL in USH2A — hearing aids from diagnosis; "
            "annual audiological review; cochlear implants NOT typically needed in USH2A "
            "(severity insufficient in most); "
            "LOOP systems for classroom/public spaces; "
            "refer to audiologist at genetic diagnosis — do not wait for subjective complaint",
            "Carbonic anhydrase inhibitors (CMO): acetazolamide 250–500 mg PO or dorzolamide 2% TID "
            "for cystoid macular oedema — treat early; "
            "OCT monitoring for CMO at each clinic visit",
            "Low vision rehabilitation: magnifiers, screen readers, orientation and mobility; "
            "MANDATORY blind/VI registration when criteria met",
            "QR-421a (Sepofarsen/antisense oligonucleotide) trial: intravitreal injection targeting "
            "USH2A exon 13 variants — enrol eligible patients; "
            "eligibility requires at least one exon 13 pathogenic allele",
            "UV protection and light restriction: as per all RP — "
            "sunglasses outdoors MANDATORY",
            "Multidisciplinary care: ophthalmologist + audiologist + rehabilitation specialist + "
            "social worker; Usher syndrome-specific support groups (Usher Coalition, Retina UK)",
        ],
        "critical_avoid": (
            "USH2A: ALL arRP patients must have PURE-TONE AUDIOMETRY — "
            "USH2A without hearing loss on clinical exam may have subtle high-frequency SNHL. "
            "VESTIBULAR NORMAL in USH2 — if vestibular dysfunction found, consider USH1 genes (MYO7A, CDH23). "
            "Anti-VEGF NOT for CMO in RP — use carbonic anhydrase inhibitors. "
            "Vitamin A: benefit controversial — no proven benefit in USH2A; avoid unless IRD specialist advises."
        ),
        "key_ddx": [
            "Usher type 1 (MYO7A, CDH23): profound congenital deafness + vestibular areflexia + RP; USH1 more severe",
            "Usher type 3 (CLRN1): progressive deafness (not congenital); variable vestibular; mainly Finnish",
            "Non-syndromic arRP: isolate USH2A RP without hearing loss — audiogram always",
            "Bardet-Biedl syndrome: RP + obesity + polydactyly + renal anomalies — syndromic",
            "RPGR X-linked RP: no hearing loss; males affected; X-linked pattern",
        ],
        "severity_weights": {"Mild": 0.28, "Moderate": 0.42, "Severe": 0.30},
        "onset_age_range": (10, 20),
        "dx_lag_y": (3, 8),
        "drug_error_rate": 0.12,
        "gene_therapy_eligible_rate": 0.30,
        "hearing_loss_rate": 0.92,
        "vestibular_rate": 0.0,
        "progression_rate": 0.82,
        "cognitive_rate": 0.02,
        "first_line_drug": "Hearing aid (MANDATORY at Dx); CAI for CMO; low vision rehab",
    },
    # ── PRPF31 — adRP13 ─────────────────────────────────────────────────────
    {
        "gene": "PRPF31", "protein": "Pre-mRNA Processing Factor 31 (PRPF31)",
        "alias": "PRPF31; OMIM gene 606419; 19q13.42; ~499 aa; autosomal dominant RP13 (OMIM #600138); AD haploinsufficiency; INCOMPLETE PENETRANCE — asymptomatic carriers common",
        "aa": "~499 aa", "kDa": "~55 kDa",
        "mechanism": (
            "PRPF31 encodes pre-mRNA processing factor 31, a component of the U4/U6·U5 tri-snRNP complex "
            "in the spliceosome (a large ribonucleoprotein machine responsible for pre-mRNA splicing). "
            "NORMAL FUNCTION: PRPF31 is essential for spliceosome assembly — it bridges U4 snRNA "
            "and NHP2L1 (15.5K protein) and anchors the tri-snRNP to the pre-mRNA; "
            "PRPF31 expression is particularly high in the retina (photoreceptors have extremely "
            "high metabolic demands → massive splicing requirement for phototransduction proteins). "
            "HAPLOINSUFFICIENCY MECHANISM: heterozygous LOF → 50% PRPF31 → "
            "spliceosome capacity reduced → splicing errors in highly expressed retinal transcripts → "
            "photoreceptor dysfunction. "
            "INCOMPLETE PENETRANCE (unique to PRPF31): "
            "the amount of PRPF31 produced from the wild-type allele is variable and regulated by "
            "a quantitative modifier locus on chromosome 14q (possibly CNOT3). "
            "If WT allele is upregulated → enough PRPF31 → ASYMPTOMATIC CARRIER. "
            "If WT allele is not upregulated → haploinsufficiency manifests → RP. "
            "PENETRANCE: ~50–80% of mutation carriers develop RP; "
            "20–50% are completely asymptomatic despite carrying the mutation."
        ),
        "disease_type": "Autosomal Dominant RP13 (AD PRPF31 haploinsufficiency; spliceosome component; INCOMPLETE PENETRANCE ~20-50% carriers unaffected; modifier locus chromosome 14q)",
        "locus": "19q13.42", "omim_gene": 606419, "omim_disease": 600138,
        "inheritance": (
            "AUTOSOMAL DOMINANT — haploinsufficiency: one pathogenic allele sufficient. "
            "CRITICAL: INCOMPLETE PENETRANCE — 20–50% of heterozygous carriers are ASYMPTOMATIC; "
            "modifier locus on chromosome 14q determines whether WT allele is upregulated sufficiently. "
            "FAMILY SCREENING COMPLEXITY: "
            "an apparently unaffected parent with the mutation is an 'obligate non-penetrant carrier' — "
            "does NOT mean the mutation is benign; "
            "offspring of non-penetrant carriers still have 50% chance of inheriting mutation + "
            "independently 50–80% chance of penetrance. "
            "GENETIC COUNSELLING CHALLENGE: "
            "must explain incomplete penetrance concept; "
            "predictive genetic testing for asymptomatic family members is valuable; "
            "penetrance cannot be predicted reliably from modifier locus genotype alone. "
            "SPORADIC CASES: de novo PRPF31 mutations occur (5–10%)."
        ),
        "phenotype": (
            "AUTOSOMAL DOMINANT RP13 (PRPF31): "
            "ONSET: variable — childhood to adult onset depending on penetrance modifier; "
            "affected individuals typically notice night blindness in first or second decade. "
            "ROD-CONE DYSTROPHY: classic RP pattern — nyctalopia → peripheral field loss → "
            "tunnel vision → central loss (late). "
            "SEVERITY: generally MILDER than XLRP (RPGR) but variable within families "
            "(related to modifier genotype). "
            "FUNDUS: bone-spicule pigmentation + waxy disc pallor + attenuated vessels (classic RP). "
            "CYSTOID MACULAR OEDEMA: occurs in ~20–30% — treat with CAIs. "
            "NON-PENETRANT CARRIERS: completely normal fundus, normal ERG, normal visual fields — "
            "these individuals still carry and transmit the mutation. "
            "ERG: reduced rod responses; cone responses variably affected."
        ),
        "treatment_options": [
            "Low vision rehabilitation: magnifiers, screen readers, orientation and mobility; "
            "register with low vision services; plan for progressive loss",
            "CAIs for CMO: acetazolamide or dorzolamide for cystoid macular oedema",
            "Genetic counselling (complex): explain incomplete penetrance — "
            "asymptomatic family members carrying mutation still transmit 50% risk; "
            "predictive genetic testing for all at-risk relatives; "
            "penetrance cannot be predicted from single gene test alone",
            "UV protection: sunglasses MANDATORY for all RP patients",
            "Gene therapy: AAV-based approaches in development for adRP; "
            "allele-specific RNA silencing + replacement strategies; "
            "clinical trials expected — register in eyeGENE/ERED",
            "Annual monitoring: dilated fundus + OCT + visual fields; "
            "ERG to track progression; "
            "non-penetrant carriers need annual fundus exam (may develop RP at any age)",
        ],
        "critical_avoid": (
            "PRPF31: INCOMPLETE PENETRANCE — do NOT reassure asymptomatic mutation carrier that they are unaffected for life; "
            "onset can occur at any age. "
            "DO NOT assume an unaffected parent has NOT passed on the mutation — "
            "non-penetrant carriers transmit with normal probability. "
            "Vitamin A: no proven benefit in PRPF31-adRP; no evidence for harm (unlike ABCA4) — "
            "generally avoid without specialist guidance."
        ),
        "key_ddx": [
            "Other adRP genes (RHO, PRPF8, PRPF6): clinically similar; distinguished by gene panel",
            "RPGR XLRP: X-linked — males severely affected; no female carriers with normal vision having RP",
            "RHO-adRP: most common adRP gene; p.Pro23His most common; generally more severe than PRPF31",
            "Non-penetrant PRPF31 carrier mimicking sporadic RP: always take three-generation family history",
        ],
        "severity_weights": {"Mild": 0.30, "Moderate": 0.45, "Severe": 0.25},
        "onset_age_range": (8, 30),
        "dx_lag_y": (3, 10),
        "drug_error_rate": 0.15,  # Missing non-penetrant carriers in counselling
        "gene_therapy_eligible_rate": 0.25,
        "hearing_loss_rate": 0.0,
        "vestibular_rate": 0.0,
        "progression_rate": 0.78,
        "cognitive_rate": 0.01,
        "first_line_drug": "Low vision rehab; CAI for CMO; genetic counselling (incomplete penetrance)",
    },
    # ── CEP290 — LCA10 ──────────────────────────────────────────────────────
    {
        "gene": "CEP290", "protein": "Centrosomal Protein 290 (CEP290)",
        "alias": "CEP290; OMIM gene 610142; 12q21.32; ~2479 aa; LCA10 (OMIM #611755) — most common LCA gene; IVS26+1655A>G founder variant; NPHP6; EDIT-101 antisense oligo",
        "aa": "~2479 aa", "kDa": "~290 kDa",
        "mechanism": (
            "CEP290 encodes centrosomal protein 290, a large coiled-coil protein that is an essential "
            "component of the ciliary transition zone (TZ) — the gating module at the base of the cilium. "
            "NORMAL FUNCTION: CEP290 acts as part of the 'ciliary gate' MKS/NPHP module — "
            "regulates protein entry and exit at the transition zone; "
            "maintains ciliary protein composition by preventing diffusion of non-ciliary proteins "
            "into the ciliary compartment; "
            "essential for photoreceptor connecting cilium (primary cilium) integrity and "
            "for olfactory cilium, renal tubular cilium, brain ependymal cilia. "
            "LCA10 PATHOMECHANISM: biallelic LOF → transition zone dysfunction → "
            "failure to traffic opsins and other phototransduction proteins to outer segments → "
            "rod and cone degeneration from birth → congenital blindness. "
            "IVS26+1655A>G FOUNDER VARIANT: "
            "an intronic variant (c.2991+1655A>G) that creates an aberrant splice donor site → "
            "inclusion of a cryptic exon → premature stop codon → truncated protein; "
            "this variant accounts for ~15–20% of all LCA10 alleles in North European/North American populations; "
            "CRITICAL: NOT detected by standard exonic sequencing — "
            "targeted IVS26+1655A>G assay or deep intronic sequencing required. "
            "SYNDROMIC SPECTRUM: severe biallelic null variants → Joubert syndrome (JBTS), "
            "nephronophthisis (NPHP6), Meckel syndrome (MKS4) — "
            "retina-only phenotype in IVS26+1655A>G homozygotes (hypomorphic)."
        ),
        "disease_type": "Leber Congenital Amaurosis 10 LCA10 (AR biallelic CEP290; most common LCA gene; IVS26+1655A>G FOUNDER; congenital blindness + nystagmus; EDIT-101 antisense oligo)",
        "locus": "12q21.32", "omim_gene": 610142, "omim_disease": 611755,
        "inheritance": (
            "AUTOSOMAL RECESSIVE: biallelic pathogenic variants. "
            "CEP290 is the most common genetic cause of LCA (~20–25% of LCA cases). "
            "IVS26+1655A>G VARIANT: most common single pathogenic CEP290 variant worldwide (~1:300 carrier frequency in Europeans); "
            "MISSED BY STANDARD EXONIC SEQUENCING — targeted intronic assay mandatory. "
            "GENOTYPE-PHENOTYPE: IVS26+1655A>G/IVS26+1655A>G (homozygous) → LCA only (retina-specific, no kidney/brain); "
            "biallelic null variants → Joubert/NPHP/Meckel — renal/brain/retina. "
            "FAMILY SCREENING: siblings 25% risk; cascade test parents. "
            "RENAL SCREENING: ALL CEP290 patients → annual renal function + ultrasound "
            "(NPHP spectrum even in 'pure' LCA — subclinical renal involvement in ~10%)."
        ),
        "phenotype": (
            "LEBER CONGENITAL AMAUROSIS 10 (LCA10): "
            "ONSET: congenital — severe visual impairment from birth (nystagmus typically first sign). "
            "NYSTAGMUS: horizontal pendular nystagmus within first weeks of life — "
            "often the presenting feature; DIGITAL EYE PRESSING (oculodigital sign) in infants. "
            "VISUAL ACUITY: severely reduced from birth — often light perception only; "
            "paradoxical: some LCA10 patients retain some central cone function (IVS26+1655A>G hypomorphic) — "
            "measure VA carefully. "
            "ERG: flat or near-flat from infancy (extinguished rod and cone responses). "
            "FUNDUS: may appear normal at birth → progressive pigmentary changes later; "
            "macular coloboma-like lesion in some. "
            "OLFACTION: olfactory dysfunction (common — CEP290 in olfactory cilia) — "
            "TEST olfaction in all LCA10 patients. "
            "RENAL: monitor annually — subclinical NPHP in ~10% even in IVS26+1655A>G cases. "
            "GENE THERAPY TARGET: IVS26+1655A>G — EDIT-101 (antisense oligonucleotide, intravitreal) "
            "corrects aberrant splicing → partial visual function recovery shown in Phase I/II."
        ),
        "treatment_options": [
            "EDIT-101 (antisense oligonucleotide) — eligibility: patients with at least ONE IVS26+1655A>G allele; "
            "intravitreal injection; targets aberrant cryptic exon insertion; "
            "Phase I/II trial (ILLUMINATE) showed visual improvement in some patients; "
            "assess eligibility at IRD tertiary centre; "
            "enrol in ERED/eyeGENE registry",
            "Renal surveillance: annual renal function (creatinine, eGFR) + urine analysis + "
            "renal ultrasound in ALL CEP290-LCA10 patients — "
            "NPHP spectrum even with IVS26+1655A>G hypomorphic allele; "
            "nephrology referral if abnormal",
            "Early visual rehabilitation: specialist education and habilitation for infants — "
            "mobility training, braille, talking books; "
            "NICU referral for infants with suspected LCA (ERG + genetic testing in year 1); "
            "early intervention improves developmental outcomes",
            "Nystagmus management: optometry assessment + prism glasses for null zone; "
            "BOTOX injections (rarely used); tenotomy for large-amplitude nystagmus; "
            "contact lenses to reduce nystagmus amplitude in some patients",
            "Olfactory assessment + ENT referral: olfactory dysfunction common — "
            "important for quality of life and safety (gas leak detection); "
            "smell training protocols (olfactory rehabilitation)",
            "Genetic counselling: explain IVS26+1655A>G — common variant missed by exonic panels; "
            "IVS26+1655A>G compound heterozygotes may have different phenotype (milder) vs biallelic null",
        ],
        "critical_avoid": (
            "CEP290 LCA10: IVS26+1655A>G IS MISSED BY STANDARD EXONIC SEQUENCING — "
            "targeted intronic assay or deep-intronic NGS MANDATORY in all LCA patients. "
            "DO NOT miss renal involvement — annual renal surveillance MANDATORY even in 'pure' LCA10. "
            "Oculodigital sign in infant = LCA until proven otherwise — urgent ERG + genetic testing. "
            "Paradoxical vision: IVS26+1655A>G homozygotes may have residual vision — "
            "do not assume no vision from nystagmus alone; measure VA carefully."
        ),
        "key_ddx": [
            "RPE65-LCA2: congenital blindness; gene therapy Luxturna (FDA 2017) available — distinguish by genetic testing",
            "GUCY2D-LCA1: most severe LCA; completely flat ERG; no treatment yet",
            "AIPL1-LCA4: severe congenital; large gene panel for LCA needed",
            "Achromatopsia (CNGA3/CNGB3): photophobia + nystagmus + loss of color; ERG shows absent cone but present rod",
            "Joubert syndrome: cerebellar vermis hypoplasia ('molar tooth sign') + ataxia — if CEP290 null alleles",
        ],
        "severity_weights": {"Mild": 0.10, "Moderate": 0.25, "Severe": 0.65},
        "onset_age_range": (0, 0.5),
        "dx_lag_y": (1, 4),
        "drug_error_rate": 0.18,  # Missing IVS26 variant; missing renal surveillance
        "gene_therapy_eligible_rate": 0.60,
        "hearing_loss_rate": 0.0,
        "vestibular_rate": 0.0,
        "progression_rate": 0.70,
        "cognitive_rate": 0.08,  # Joubert spectrum in null/null
        "first_line_drug": "EDIT-101 (if IVS26+1655A>G eligible); renal surveillance; early rehab",
    },
    # ── MYO7A — Usher 1B ────────────────────────────────────────────────────
    {
        "gene": "MYO7A", "protein": "Myosin VIIA (MYO7A)",
        "alias": "MYO7A; OMIM gene 276903; 11q13.5; ~2215 aa; Usher syndrome 1B USH1B (OMIM #276900) — most common USH1; DFNB2 autosomal recessive; DFNA11 autosomal dominant",
        "aa": "~2215 aa", "kDa": "~254 kDa",
        "mechanism": (
            "MYO7A encodes myosin VIIA, an unconventional myosin motor protein expressed in cochlear "
            "hair cells and retinal pigment epithelium (RPE). "
            "NORMAL FUNCTION (cochlea): myosin VIIA localises to stereocilia and maintains cohesion of "
            "the stereocilia hair bundle via its motor activity — essential for mechanotransduction; "
            "links stereocilia to kinocilium; "
            "moves along actin filaments using ATP hydrolysis. "
            "NORMAL FUNCTION (retina): in RPE, myosin VIIA transports melanosomes toward the apical "
            "microvilli in response to light + is required for phagocytosis of shed outer segment discs "
            "by RPE — disruption causes outer segment accumulation → photoreceptor toxicity. "
            "USH1B PATHOMECHANISM: biallelic MYO7A LOF → "
            "stereocilia bundle disorganisation → mechanotransduction failure → "
            "PROFOUND CONGENITAL SENSORINEURAL DEAFNESS; "
            "simultaneously → RPE phagocytosis defect + connecting cilium dysfunction → "
            "RP (typically onset late childhood to teenage years). "
            "VESTIBULAR ARREFLEXIA: MYO7A in vestibular hair cells → vestibular system failure → "
            "VESTIBULAR AREFLEXIA = PATHOGNOMONIC of USH1 (all USH1 genes cause vestibular areflexia)."
        ),
        "disease_type": "Usher Syndrome Type 1B USH1B (AR biallelic MYO7A; profound congenital SNHL + RP + VESTIBULAR AREFLEXIA PATHOGNOMONIC; most common USH1; cochlear implants effective for hearing)",
        "locus": "11q13.5", "omim_gene": 276903, "omim_disease": 276900,
        "inheritance": (
            "AUTOSOMAL RECESSIVE: biallelic pathogenic variants. "
            "MYO7A is the most common USH1 gene — accounts for ~35–50% of all Usher type 1. "
            "ALLELIC SERIES: "
            "- Biallelic null → USH1B (deafness + RP + vestibular areflexia — full syndrome); "
            "- DFNB2: some biallelic hypomorphic → non-syndromic AR deafness only (no RP); "
            "- DFNA11: heterozygous dominant → non-syndromic AD progressive deafness (later onset). "
            "FAMILY SCREENING: siblings 25% recurrence; carrier parents unaffected. "
            "COCHLEAR IMPLANT: HIGHLY EFFECTIVE in MYO7A-DFNB2/USH1B — "
            "should be offered early (before 2 years) for maximal speech/language development."
        ),
        "phenotype": (
            "USHER SYNDROME TYPE 1B (MYO7A): "
            "HEARING: PROFOUND CONGENITAL bilateral SNHL — fails newborn hearing screening; "
            "no useful residual hearing without amplification in most. "
            "VESTIBULAR: VESTIBULAR AREFLEXIA — absent caloric responses + absent VEMPs; "
            "PATHOGNOMONIC for USH1 (all USH1 genes: MYO7A, CDH23, PCDH15, SANS, USH1G); "
            "clinical consequences: delayed motor milestones (balance-dependent) — "
            "most USH1 children walk at ~18–24 months (delayed vs 12 months normal); "
            "Romberg positive; fall in dark (uncompensated vestibular loss). "
            "VISION: RP onset typically late childhood to teenage — "
            "night blindness → peripheral field loss → tunnel vision; "
            "more severe than USH2A. "
            "COMMUNICATION: profound deafness from birth → cochlear implant HIGHLY effective; "
            "sign language as primary communication mode if no CI or delayed CI."
        ),
        "treatment_options": [
            "Cochlear implant (CI): HIGHLY EFFECTIVE for MYO7A-USH1B deafness — "
            "MANDATORY referral for CI assessment at diagnosis; "
            "bilateral CI provides best auditory outcomes; "
            "optimal timing: before 2 years of age for maximal speech-language development; "
            "CI does NOT help RP (retina separately affected); "
            "CI in combination with visual communication (sign language) during RP progression",
            "Visual rehabilitation: low vision services; mobility training begins EARLY "
            "BEFORE severe RP — deafblind rehabilitation specialists; "
            "cane training + guide dog when RP advanced; "
            "tactile communication (deafblind manual alphabet, Tadoma)",
            "Vestibular rehabilitation: physiotherapy for balance; "
            "compensatory strategies for vestibular areflexia; "
            "fall prevention (rails, night-lights); "
            "swimming with supervision (no vestibular righting reflex)",
            "CAIs for CMO: dorzolamide or acetazolamide for cystoid macular oedema in RP",
            "Multidisciplinary deafblind team: ophthalmologist + audiologist + CI team + "
            "rehabilitation specialist + orientation-mobility instructor + social worker + "
            "psychologist (adjust to progressive vision loss on background of deafness)",
            "Gene therapy trials: subretinal MYO7A delivery in development (LVMO7A lentiviral vector — "
            "lentivirus preferred over AAV due to large MYO7A cDNA size exceeding AAV capacity); "
            "eligibility assessment at IRD centre",
        ],
        "critical_avoid": (
            "MYO7A USH1B: VESTIBULAR AREFLEXIA — swim supervision MANDATORY; fall prevention critical; "
            "Romberg positive — warn patient/family about dark environment falls. "
            "COCHLEAR IMPLANT TIMING: delay beyond 3 years significantly worsens speech-language outcomes; "
            "REFER URGENTLY on diagnosis. "
            "DO NOT assume 'not deaf' because child is responding to vibration/visual cues — "
            "formal audiological testing (ABR) mandatory in all infants with nystagmus or suspected LCA/USH."
        ),
        "key_ddx": [
            "USH2A Usher 2: mild-moderate SNHL (not profound congenital); normal vestibular; milder RP",
            "CDH23 Usher 1D: clinically identical to USH1B — genetically distinguished",
            "PCDH15 Usher 1F: USH1; distinguished by genotype only",
            "Non-syndromic DFNB2 (MYO7A): deafness only without RP — hypomorphic alleles",
            "CHARGE syndrome: coloboma + choanal atresia + heart defect + growth retardation + ear anomalies (CHD7)",
        ],
        "severity_weights": {"Mild": 0.08, "Moderate": 0.30, "Severe": 0.62},
        "onset_age_range": (0, 0.1),  # congenital deafness; RP onset 10-20y
        "dx_lag_y": (1, 5),
        "drug_error_rate": 0.12,
        "gene_therapy_eligible_rate": 0.20,
        "hearing_loss_rate": 1.0,  # profound congenital
        "vestibular_rate": 1.0,   # vestibular areflexia PATHOGNOMONIC
        "progression_rate": 0.88,
        "cognitive_rate": 0.03,
        "first_line_drug": "Cochlear implant (URGENT, <2y); deafblind rehab; CI for hearing",
    },
    # ── CRB1 — LCA8 / RP12 ──────────────────────────────────────────────────
    {
        "gene": "CRB1", "protein": "Crumbs Homologue 1 (CRB1)",
        "alias": "CRB1; OMIM gene 604210; 1q31.3; ~1406 aa; LCA8 (OMIM #613835) + RP12 (OMIM #600105); AR; thickened retina OCT PATHOGNOMONIC; para-arteriolar RPE preservation (PPRPE)",
        "aa": "~1406 aa", "kDa": "~155 kDa",
        "mechanism": (
            "CRB1 encodes crumbs homologue 1, an apically localised transmembrane protein that is "
            "part of the Crumbs polarity complex (CRB/PALS1/PATJ complex) — essential for "
            "photoreceptor outer limiting membrane (OLM) integrity and apicobasal polarity. "
            "NORMAL FUNCTION: CRB1 maintains the OLM (a junctional complex between Müller glial cells "
            "and photoreceptors at the outer nuclear layer); "
            "localises to the subapical region (SAR) of photoreceptors between OS and ONL; "
            "regulates photoreceptor morphogenesis and Müller cell morphology. "
            "CRB1-LCA/RP PATHOMECHANISM: biallelic CRB1 LOF → OLM disruption → "
            "Müller cell process disorganisation → photoreceptor outer segment misalignment → "
            "THICKENED RETINA on OCT (Müller cell hypertrophy fills outer retina); "
            "photoreceptor degeneration follows. "
            "THICKENED RETINA: CRB1 mutations uniquely cause THICKENED retina on OCT "
            "(most IRD genes cause thinning) — PATHOGNOMONIC finding. "
            "PARA-ARTERIOLAR RPE PRESERVATION (PPRPE): "
            "periarteriolar patches of preserved RPE visible on fundus exam — "
            "PATHOGNOMONIC for CRB1-RP12."
        ),
        "disease_type": "LCA8 (severe congenital) / RP12 (AR CRB1; thickened retina OCT PATHOGNOMONIC; para-arteriolar RPE preservation PATHOGNOMONIC; Müller cell hypertrophy)",
        "locus": "1q31.3", "omim_gene": 604210, "omim_disease": 613835,
        "inheritance": (
            "AUTOSOMAL RECESSIVE: biallelic pathogenic variants. "
            "CRB1 accounts for ~7–10% of LCA cases and ~4–5% of arRP. "
            "COMMON VARIANTS: p.Cys948Tyr (c.2843G>A) — most common pathogenic CRB1 allele in European populations; "
            "p.Lys801fsX — Dutch founder allele. "
            "FAMILY SCREENING: siblings 25% risk; parents are obligate carriers. "
            "ALLELIC HETEROGENEITY: LCA8 (severe, congenital) vs RP12 (milder, juvenile-onset RP) — "
            "both caused by biallelic CRB1 but different allele combinations and modifier effects."
        ),
        "phenotype": (
            "CRB1-LCA8 (SEVERE): congenital nystagmus + profoundly reduced VA from birth; "
            "flat ERG; thickened retina on OCT — PATHOGNOMONIC; fundus pigmentary changes from year 1. "
            "CRB1-RP12 (MILDER): childhood-onset RP — nyctalopia → peripheral field loss → "
            "central loss; slower progression than LCA8. "
            "FUNDUS HALLMARKS: "
            "1. PARA-ARTERIOLAR RPE PRESERVATION (PPRPE) — periarteriolar islands of preserved fundus "
            "along blood vessel paths — PATHOGNOMONIC for CRB1-RP12; "
            "2. THICKENED RETINA on OCT (increased total retinal thickness due to Müller cell hypertrophy) — "
            "opposite of typical RP thinning — PATHOGNOMONIC for CRB1 in any IRD. "
            "MACULAR DYSTROPHY: coloboma-like macular lesion in some LCA8 patients. "
            "COGNITIVE FUNCTION: usually NORMAL (rare Müller gliosis → mild processing issues). "
            "CMO: cystoid macular oedema occurs — treat with CAIs."
        ),
        "treatment_options": [
            "Low vision rehabilitation: specialists, screen readers, orientation and mobility; "
            "register with VI services early — CRB1-LCA8 may have near-zero vision from infancy",
            "CAIs for CMO: acetazolamide or dorzolamide for cystoid macular oedema — "
            "OCT monitoring mandatory (watch for thickened retina pattern)",
            "Gene therapy development: subretinal AAV approaches in preclinical stage; "
            "CRB1 mouse models rescued by AAV-CRB1 delivery; "
            "clinical trials expected — register in eyeGENE/ERED",
            "Annual monitoring: OCT (retinal thickness tracking — UNIQUE to CRB1; expect progressive thinning as disease advances despite initial thickening); "
            "visual fields + ERG annually; renal monitoring not required (unlike CEP290)",
            "Genetic counselling: AR — 25% sibling risk; "
            "explain PPRPE and thickened retina as distinctive features; "
            "LCA8 vs RP12 distinction important for prognosis and educational planning",
        ],
        "critical_avoid": (
            "CRB1: THICKENED retina on OCT — do NOT assume normal/mild disease; thickening is pathological Müller hypertrophy. "
            "PPRPE PATHOGNOMONIC — if para-arteriolar RPE preservation seen, CRB1 is top differential. "
            "Vitamin A: no benefit demonstrated; avoid as in most RP."
        ),
        "key_ddx": [
            "CEP290 LCA10: congenital blindness + nystagmus; flat ERG; but retina may be thin, not thick",
            "Leber congenital amaurosis (other genes): LCA2 (RPE65 — treatable); LCA1 (GUCY2D)",
            "Goldmann-Favre syndrome (NR2E3): vitreoretinal; schisis + RP; enhanced S-cone syndrome",
            "Coats disease: exudative retinal detachment; unilateral; no hereditary pattern",
        ],
        "severity_weights": {"Mild": 0.15, "Moderate": 0.32, "Severe": 0.53},
        "onset_age_range": (0, 5),
        "dx_lag_y": (1, 5),
        "drug_error_rate": 0.10,
        "gene_therapy_eligible_rate": 0.15,
        "hearing_loss_rate": 0.0,
        "vestibular_rate": 0.0,
        "progression_rate": 0.80,
        "cognitive_rate": 0.05,
        "first_line_drug": "CAI for CMO; low vision rehab; OCT monitoring (unique thick pattern)",
    },
    # ── BEST1 — Best vitelliform macular dystrophy ───────────────────────────
    {
        "gene": "BEST1", "protein": "Bestrophin 1 (BEST1)",
        "alias": "BEST1; OMIM gene 607854; 11q12.3; ~585 aa; Best vitelliform macular dystrophy VMD2 (OMIM #153700) — AD; autosomal recessive bestrophinopathy ARB; AVMD; EOG Arden ratio <1.5 PATHOGNOMONIC",
        "aa": "~585 aa", "kDa": "~68 kDa",
        "mechanism": (
            "BEST1 encodes bestrophin 1, a calcium-activated chloride channel (CaCC) in the basolateral "
            "membrane of retinal pigment epithelium (RPE). "
            "NORMAL FUNCTION: bestrophin 1 channels Cl⁻ across RPE basolateral membrane in response "
            "to intracellular Ca²⁺ — regulates fluid balance across RPE; "
            "maintains subretinal space homeostasis; "
            "also modulates voltage-gated calcium channels in RPE. "
            "VMD2 PATHOMECHANISM: heterozygous dominant-negative BEST1 mutation → "
            "mutant protein inhibits WT protein channel function (dominant negative) → "
            "reduced chloride conductance → impaired fluid resorption from subretinal space → "
            "VITELLIFORM MATERIAL (lipofuscin/lipid) accumulates under macula → "
            "characteristic 'egg-yolk' lesion (vitelliform). "
            "EOG (Electro-Oculogram) MECHANISM: the light-evoked change in RPE standing potential "
            "(Arden ratio) depends on BEST1 channel activity — "
            "ARDEN RATIO <1.5 in ALL BEST1 mutation carriers (even pre-symptomatic) — "
            "PATHOGNOMONIC diagnostic test. "
            "ERG IS NORMAL: photoreceptors are not primarily affected early → normal ERG "
            "(distinguishes from RP, where ERG is abnormal). "
            "AUTOSOMAL RECESSIVE BESTROPHINOPATHY (ARB): biallelic BEST1 → "
            "more severe, multifocal vitelliform lesions + angle-closure glaucoma risk."
        ),
        "disease_type": "Best Vitelliform Macular Dystrophy VMD2 (AD BEST1 dominant-negative; 'egg-yolk' vitelliform lesion; ERG NORMAL; EOG Arden ratio <1.5 PATHOGNOMONIC; angle-closure glaucoma risk in ARB)",
        "locus": "11q12.3", "omim_gene": 607854, "omim_disease": 153700,
        "inheritance": (
            "AUTOSOMAL DOMINANT (VMD2): heterozygous pathogenic variants — dominant negative mechanism. "
            "PENETRANCE: ~96–100% (essentially fully penetrant for abnormal EOG); "
            "but visual symptoms may be minimal in some carriers. "
            "AUTOSOMAL RECESSIVE BESTROPHINOPATHY (ARB): biallelic variants → more severe diffuse lesions; "
            "multifocal vitelliform deposits; angle-closure glaucoma risk — "
            "MANDATORY gonioscopy in all ARB patients. "
            "FAMILY SCREENING: all first-degree relatives of VMD2 patients → EOG + genetic testing; "
            "pre-symptomatic carriers identified by EOG (abnormal even before fundus changes). "
            "SPORADIC: de novo BEST1 variants in ~30% (no family history). "
            "GENOTYPE-PHENOTYPE: p.Arg218Cys most common; stage of disease does not correlate with VA."
        ),
        "phenotype": (
            "BEST VITELLIFORM MACULAR DYSTROPHY (VMD2) — STAGES: "
            "Stage 0 (Previtelliform): normal fundus; normal VA; abnormal EOG ONLY — carrier detected. "
            "Stage 1 (Vitelliform): CLASSIC 'EGG-YOLK' lesion — bright yellow-orange subfoveal deposit "
            "on fundus; VA often NORMAL or near-normal at this stage. "
            "Stage 2 (Pseudohypopyon): lesion partially absorbed — layered appearance (fluid/lipofuscin level). "
            "Stage 3 (Vitelliruptive/'Scrambled egg'): lesion fragmented and dispersed. "
            "Stage 4 (Atrophic): macular atrophy → VA DECLINES — often 6/24 or worse. "
            "CHOROIDAL NEOVASCULARISATION (CNV): complication in ~20–25% — anti-VEGF EFFECTIVE for CNV; "
            "sudden VA loss in known Best disease → urgent OCT + FFA to rule out CNV. "
            "ERG: NORMAL (photoreceptors intact early — distinguishes from RP). "
            "EOG ARDEN RATIO: <1.5 in ALL MUTATION CARRIERS — the gold standard diagnostic test."
        ),
        "treatment_options": [
            "Monitoring: annual OCT + FAF + VA — track progression stage by stage; "
            "EOG at diagnosis and in all first-degree relatives (carrier detection); "
            "urgent OCT + FFA if sudden VA loss (rule out CNV)",
            "Anti-VEGF for CNV: intravitreal ranibizumab/bevacizumab/aflibercept — "
            "EFFECTIVE for choroidal neovascularisation complication in Best disease; "
            "~20–25% develop CNV (usually at vitelliruptive/atrophic stage); "
            "PRN or treat-and-extend regimen; "
            "IMPORTANT: anti-VEGF is for CNV COMPLICATION only — NOT for primary VMD2 treatment",
            "Genetic counselling: AD with near-complete penetrance for EOG abnormality; "
            "all first-degree relatives MUST have EOG + genetic testing (pre-symptomatic); "
            "de novo cases: 50% offspring risk",
            "ARB (biallelic BEST1): MANDATORY gonioscopy + IOP monitoring — "
            "angle-closure glaucoma risk (shallow anterior chamber); "
            "laser iridotomy if angle narrow; "
            "multifocal ERG (mfERG) for functional mapping in multifocal lesions",
            "Low vision rehabilitation: when VA declines to stage 4 atrophy — "
            "magnifiers, eccentric viewing training; "
            "PROGNOSIS: most VMD2 patients maintain driving standard vision until 4th–5th decade",
            "Gene therapy research: preclinical stage; BEST1 restoration in iPSC-RPE models; "
            "clinical trials not yet open",
        ],
        "critical_avoid": (
            "BEST1: ERG NORMAL — do NOT exclude bestrophinopathy because ERG is normal; "
            "EOG Arden ratio <1.5 is the diagnostic test (abnormal even in stage 0 carriers). "
            "Anti-VEGF ONLY for CNV complication — NOT for primary vitelliform lesion treatment. "
            "ARB: mandatory gonioscopy — angle-closure glaucoma risk missed if not screened. "
            "DO NOT confuse vitelliform lesion with vitelliform-pattern AMD — different diseases; "
            "BEST1 testing mandatory in any young patient with vitelliform macular lesion."
        ),
        "key_ddx": [
            "Pattern dystrophy (PRPH2): butterfly or other pattern; ERG shows mildly reduced pattern; EOG normal",
            "Vitelliform-like AMD: older onset; drusen elsewhere; BEST1 negative",
            "Cone dystrophy: central loss; photophobia; ERG shows cone dysfunction",
            "Foveomacular vitelliform dystrophy (adult-onset VMD): PRPH2 variant; later onset; EOG normal or mildly reduced",
            "Stargardt (ABCA4): flecks beyond macula; dark choroid on FFA; EOG normal",
        ],
        "severity_weights": {"Mild": 0.45, "Moderate": 0.38, "Severe": 0.17},
        "onset_age_range": (5, 15),
        "dx_lag_y": (2, 8),
        "drug_error_rate": 0.20,  # Using anti-VEGF for primary VMD2; missing ARB angle-closure
        "gene_therapy_eligible_rate": 0.05,
        "hearing_loss_rate": 0.0,
        "vestibular_rate": 0.0,
        "progression_rate": 0.65,
        "cognitive_rate": 0.01,
        "first_line_drug": "Anti-VEGF (CNV only); EOG monitoring; annual OCT; genetic cascade",
    },
]


def _gen_patients_for_gene(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    n = 40
    patients = []
    sw = gene_data["severity_weights"]
    severities = list(sw.keys())
    weights = list(sw.values())

    for i in range(n):
        sev = rng.choices(severities, weights=weights, k=1)[0]

        onset_lo, onset_hi = gene_data["onset_age_range"]
        onset = round(rng.uniform(onset_lo, max(onset_lo + 0.01, onset_hi)), 2)
        lag = round(rng.uniform(*gene_data["dx_lag_y"]), 2)
        dx_age = round(onset + lag, 2)

        prog = rng.random() < gene_data["progression_rate"]
        drug_err = rng.random() < gene_data["drug_error_rate"]
        gt_elig = rng.random() < gene_data["gene_therapy_eligible_rate"]
        hearing = rng.random() < gene_data["hearing_loss_rate"]
        vestib = rng.random() < gene_data["vestibular_rate"]
        cognitive = rng.random() < gene_data["cognitive_rate"]

        # Vision loss correlated with severity
        vision_loss = sev == "Severe" or (sev == "Moderate" and rng.random() < 0.55)

        patients.append({
            "id": f"IRD-{gene_data['gene']}-{seed}-{i + 1:03d}",
            "gene": gene_data["gene"],
            "seed": seed,
            "onset_age_y": onset,
            "diagnosis_age_y": dx_age,
            "severity": sev,
            "vision_loss": vision_loss,
            "hearing_loss": hearing,
            "vestibular_dysfunction": vestib,
            "gene_therapy_eligible": gt_elig,
            "drug_avoid_prescribed_error": drug_err,
            "on_targeted_therapy": gt_elig and rng.random() < 0.30,
            "disease_progression": prog,
            "cognitive_impairment": cognitive,
        })
    return patients


def _gen_cohort() -> list:
    all_patients = []
    for i, gd in enumerate(IRD_GENES):
        all_patients.extend(_gen_patients_for_gene(gd, SEED_BASE + i))
    return all_patients


def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] += 1

    vision_n   = sum(1 for p in patients if p["vision_loss"])
    hearing_n  = sum(1 for p in patients if p["hearing_loss"])
    vestib_n   = sum(1 for p in patients if p["vestibular_dysfunction"])
    gt_n       = sum(1 for p in patients if p["gene_therapy_eligible"])
    targeted_n = sum(1 for p in patients if p["on_targeted_therapy"])
    drug_err_n = sum(1 for p in patients if p["drug_avoid_prescribed_error"])
    prog_n     = sum(1 for p in patients if p["disease_progression"])

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 2)
    mean_dx = round(sum(p["diagnosis_age_y"] for p in patients) / n, 2)

    return {
        "atlas": "IRD-Atlas",
        "full_name": "Complete 8-Gene Inherited Retinal Dystrophy Atlas",
        "subtitle": "RPGR·ABCA4·USH2A·PRPF31·CEP290·MYO7A·CRB1·BEST1 — 320 patients (8×40, seeds 1094–1101)",
        "description": (
            "Comprehensive atlas of 8 major genetic inherited retinal dystrophies encompassing: "
            "X-LINKED RP3 (RPGR — XLR; ORF15 hotspot ~70% of mutations; rod-cone dystrophy; "
            "ORF15 missed by standard NGS; CMO in 30% — use CAIs not anti-VEGF; gene therapy trials); "
            "STARGARDT MACULAR DYSTROPHY STGD1 (ABCA4 — AR; A2E bisretinoid toxin; "
            "VITAMIN A ABSOLUTELY CONTRAINDICATED; light restriction MANDATORY; dark choroid FFA PATHOGNOMONIC; "
            "most common inherited macular dystrophy 1:8000); "
            "USHER SYNDROME TYPE 2A (USH2A — AR; RP + mild-moderate SNHL + normal vestibular; "
            "80% of USH2; 15% of arRP; hearing aids MANDATORY; exon 13 antisense oligo QR-421a trial); "
            "AUTOSOMAL DOMINANT RP13 (PRPF31 — AD; spliceosome component; "
            "INCOMPLETE PENETRANCE ~20-50% asymptomatic carriers; modifier locus chromosome 14q); "
            "LEBER CONGENITAL AMAUROSIS 10 (CEP290 — AR; most common LCA gene 20-25%; "
            "IVS26+1655A>G FOUNDER missed by exonic sequencing; congenital blindness + nystagmus; "
            "EDIT-101 antisense oligo for IVS26 carriers; annual renal surveillance MANDATORY); "
            "USHER SYNDROME TYPE 1B (MYO7A — AR; profound CONGENITAL SNHL + RP + "
            "VESTIBULAR AREFLEXIA PATHOGNOMONIC; cochlear implant HIGHLY EFFECTIVE <2y; "
            "deafblind rehabilitation essential); "
            "LCA8/RP12 (CRB1 — AR; THICKENED retina OCT PATHOGNOMONIC; "
            "para-arteriolar RPE preservation PATHOGNOMONIC; Müller cell hypertrophy); "
            "BEST VITELLIFORM MACULAR DYSTROPHY VMD2 (BEST1 — AD dominant-negative; "
            "ERG NORMAL; EOG ARDEN RATIO <1.5 PATHOGNOMONIC even pre-symptomatic; "
            "anti-VEGF ONLY for CNV complication; ARB biallelic — angle-closure glaucoma risk)."
        ),
        "total_patients": n,
        "genes_covered": len(IRD_GENES),
        "patients_per_gene": 40,
        "seed_range": "1094–1101",
        "gene_list": [g["gene"] for g in IRD_GENES],
        "disease_category_breakdown": {
            "X-linked RP (RPGR XLR; ORF15 hotspot; rod-cone; males severe; CAI for CMO)": ["RPGR"],
            "Stargardt Macular Dystrophy (AR ABCA4; A2E toxin; Vitamin A ABSOLUTELY CI; dark choroid FFA)": ["ABCA4"],
            "Usher syndrome 2A (AR USH2A; RP + mild-mod SNHL + normal vestibular; 80% USH2; QR-421a trial)": ["USH2A"],
            "Autosomal dominant RP13 (AD PRPF31; haploinsufficiency; INCOMPLETE PENETRANCE; spliceosome)": ["PRPF31"],
            "LCA10 (AR CEP290; most common LCA; IVS26+1655A>G FOUNDER; congenital blindness; EDIT-101)": ["CEP290"],
            "Usher syndrome 1B (AR MYO7A; profound CONGENITAL SNHL + RP + VESTIBULAR AREFLEXIA; CI <2y)": ["MYO7A"],
            "LCA8 / RP12 (AR CRB1; THICKENED retina OCT; PPRPE PATHOGNOMONIC; Müller hypertrophy)": ["CRB1"],
            "Best VMD2 (AD BEST1; ERG normal; EOG Arden <1.5 PATHOGNOMONIC; anti-VEGF CNV only)": ["BEST1"],
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "mean_diagnosis_age_y": mean_dx,
        "kpis": [
            {"label": "Total Patients", "value": n, "color": "#37474f"},
            {"label": "Genes Covered", "value": len(IRD_GENES), "color": "#1a237e"},
            {"label": "Patients/Gene", "value": 40, "color": "#4a148c"},
            {"label": "Vision Loss", "value": f"{round(100 * vision_n / n, 1)}%", "color": "#b71c1c"},
            {"label": "Hearing Loss (USH)", "value": f"{round(100 * hearing_n / n, 1)}%", "color": "#e65100"},
            {"label": "Gene Therapy Eligible", "value": f"{round(100 * gt_n / n, 1)}%", "color": "#1b5e20"},
        ],
        "clinical_features_prevalence": {
            "Vision Loss (legal blindness)": round(100 * vision_n / n, 1),
            "Hearing Loss (USH2A + MYO7A)": round(100 * hearing_n / n, 1),
            "Vestibular Areflexia (MYO7A USH1)": round(100 * vestib_n / n, 1),
            "Gene Therapy Trial Eligible": round(100 * gt_n / n, 1),
            "On Targeted Therapy": round(100 * targeted_n / n, 1),
            "Drug-Prescribing Error Detected": round(100 * drug_err_n / n, 1),
            "Disease Progression": round(100 * prog_n / n, 1),
        },
        "drug_alerts": [
            "ABCA4 (Stargardt): VITAMIN A ABSOLUTELY CONTRAINDICATED — accelerates A2E bisretinoid "
            "accumulation → faster RPE degeneration. DO NOT prescribe retinol, beta-carotene, or "
            "any multivitamin containing vitamin A to ABCA4-Stargardt patients. "
            "LIGHT RESTRICTION MANDATORY — UV-blocking wraparound sunglasses outdoors ALWAYS.",
            "RPGR/RP: CMO (cystoid macular oedema) is INFLAMMATORY — anti-VEGF is INEFFECTIVE; "
            "use CARBONIC ANHYDRASE INHIBITORS (acetazolamide 250-500 mg PO or dorzolamide 2% TID). "
            "ORF15 MUTATIONS MISSED BY STANDARD NGS — dedicated ORF15 sequencing MANDATORY in XLRP panels.",
            "CEP290 LCA10: IVS26+1655A>G IS MISSED BY STANDARD EXONIC NGS — "
            "targeted intronic assay MANDATORY in all LCA patients. "
            "RENAL SURVEILLANCE MANDATORY annually — even 'pure' LCA10 has subclinical NPHP risk.",
            "MYO7A USH1B: COCHLEAR IMPLANT MUST be offered BEFORE AGE 2 — delay worsens speech outcomes; "
            "VESTIBULAR AREFLEXIA — swim supervision MANDATORY; fall prevention critical in dark environments.",
            "BEST1 VMD2: EOG Arden ratio <1.5 PATHOGNOMONIC even before fundus changes — "
            "test ALL first-degree relatives. Anti-VEGF ONLY for CNV complication — NOT primary VMD2 therapy. "
            "ARB (biallelic BEST1): MANDATORY gonioscopy — angle-closure glaucoma risk.",
            "PRPF31 adRP: INCOMPLETE PENETRANCE — do NOT reassure asymptomatic carrier; "
            "non-penetrant carriers transmit mutation at normal Mendelian probability. "
            "CRB1: thickened retina on OCT is PATHOGNOMONIC — opposite of typical IRD thinning.",
            "USH2A: ALL arRP patients MUST have AUDIOGRAM — USH2A without subjective hearing loss "
            "may have high-frequency SNHL detectable only on pure-tone audiometry.",
        ],
        "diagnostic_pearls": [
            "RPGR (XLRP): males severely affected; ORF15 hotspot ~70% — standard NGS misses; "
            "CMO treat with CAI (not anti-VEGF); carrier females 15-25% symptomatic",
            "ABCA4 (Stargardt): 'dark' (silent) choroid on fluorescein angiography PATHOGNOMONIC; "
            "fundus flecks + macular atrophy; Vitamin A ABSOLUTELY CONTRAINDICATED",
            "USH2A (Usher 2): RP + mild-moderate SNHL + NORMAL vestibular; hearing aids MANDATORY; "
            "check audiogram in ALL arRP patients to detect USH2A",
            "PRPF31 (adRP13): INCOMPLETE PENETRANCE — asymptomatic mutation carriers exist in every family; "
            "EOG + ERG + genetics for all first-degree relatives regardless of symptoms",
            "CEP290 (LCA10): IVS26+1655A>G — most common variant — MISSED BY EXONIC SEQUENCING; "
            "congenital blindness + nystagmus; EDIT-101 antisense oligo for eligible patients",
            "MYO7A (Usher 1B): VESTIBULAR AREFLEXIA = pathognomonic of USH1; delayed walking milestone; "
            "profound congenital deafness; cochlear implant URGENT <2y",
            "CRB1 (LCA8/RP12): THICKENED retina on OCT (not thinned!) PATHOGNOMONIC; "
            "para-arteriolar RPE preservation (PPRPE) on fundus PATHOGNOMONIC",
            "BEST1 (VMD2): EOG Arden ratio <1.5 in ALL carriers EVEN stage 0; "
            "ERG NORMAL (distinguishes from RP); anti-VEGF only for CNV complication",
        ],
    }


def get_breakdown() -> dict:
    all_patients = _gen_cohort()
    breakdown = {}

    for gd in IRD_GENES:
        gene_pts = [p for p in all_patients if p["gene"] == gd["gene"]]
        n = len(gene_pts)
        sev_counts = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for p in gene_pts:
            sev_counts[p["severity"]] += 1

        breakdown[gd["gene"]] = {
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "disease_type": gd["disease_type"],
            "inheritance": gd["inheritance"],
            "phenotype": gd["phenotype"],
            "mechanism": gd["mechanism"],
            "treatment_options": gd["treatment_options"],
            "key_ddx": gd["key_ddx"],
            "critical_avoid": gd.get("critical_avoid", ""),
            "first_line_drug": gd["first_line_drug"],
            "n_patients": n,
            "mean_onset_age_y": round(sum(p["onset_age_y"] for p in gene_pts) / n, 2),
            "mean_dx_age_y": round(sum(p["diagnosis_age_y"] for p in gene_pts) / n, 2),
            "severity_distribution": {
                "mild_pct": round(100 * sev_counts["Mild"] / n, 1),
                "moderate_pct": round(100 * sev_counts["Moderate"] / n, 1),
                "severe_pct": round(100 * sev_counts["Severe"] / n, 1),
            },
            "vision_loss_pct": round(100 * sum(1 for p in gene_pts if p["vision_loss"]) / n, 1),
            "hearing_loss_pct": round(100 * sum(1 for p in gene_pts if p["hearing_loss"]) / n, 1),
            "vestibular_pct": round(100 * sum(1 for p in gene_pts if p["vestibular_dysfunction"]) / n, 1),
            "gene_therapy_eligible_pct": round(100 * sum(1 for p in gene_pts if p["gene_therapy_eligible"]) / n, 1),
            "on_targeted_therapy_pct": round(100 * sum(1 for p in gene_pts if p["on_targeted_therapy"]) / n, 1),
            "drug_error_pct": round(100 * sum(1 for p in gene_pts if p["drug_avoid_prescribed_error"]) / n, 1),
            "progression_pct": round(100 * sum(1 for p in gene_pts if p["disease_progression"]) / n, 1),
            "cognitive_impairment_pct": round(100 * sum(1 for p in gene_pts if p["cognitive_impairment"]) / n, 1),
        }

    return {
        "atlas": "IRD-Atlas",
        "subtitle": "Per-gene clinical breakdown — 320 patients (8×40, seeds 1094–1101)",
        "genes": breakdown,
        "gene_order": [g["gene"] for g in IRD_GENES],
    }


def get_definitions() -> dict:
    return {
        "atlas": "IRD-Atlas",
        "subtitle": "Clinical and genetic terminology definitions for Inherited Retinal Dystrophy Atlas",
        "definitions": {
            "Inherited Retinal Dystrophy (IRD)": (
                "A heterogeneous group of hereditary disorders causing progressive loss of "
                "photoreceptor function and/or retinal pigment epithelium (RPE) function, "
                "leading to visual impairment or blindness. "
                "Classification: rod-cone dystrophies (RP — peripheral before central); "
                "cone-rod dystrophies (central before peripheral); macular dystrophies (central only); "
                "stationary disorders. "
                "Prevalence: ~1:3,000 combined (most common cause of severe sight impairment "
                "in working-age adults in developed countries). "
                "Genetics: >250 causative genes identified; inheritance: AD, AR, X-linked, mitochondrial. "
                "Key investigations: ERG (retinal function); OCT (retinal structure); "
                "fundus autofluorescence; visual fields; EOG (RPE function); genetic panel."
            ),
            "Electroretinogram (ERG)": (
                "Electrophysiological test measuring retinal function in response to light stimulation. "
                "COMPONENTS: scotopic (dark-adapted, rod-dominated) — a-wave (photoreceptors) + "
                "b-wave (bipolar cells); photopic (light-adapted, cone-dominated). "
                "RP: rod responses severely reduced early; cones later. "
                "LCA (CEP290, CRB1): flat/undetectable from infancy. "
                "BEST DISEASE (BEST1): ERG NORMAL — critical distinguishing feature from RP. "
                "Stargardt (ABCA4): ERG normal early; abnormal in advanced disease. "
                "Standard: ISCEV ERG protocol. "
                "Paediatric: ERG under GA/sedation in young children."
            ),
            "Electro-Oculogram (EOG) / Arden Ratio": (
                "Electrophysiological test of RPE function. "
                "TECHNIQUE: records standing potential across RPE as eyes move from darkness to light; "
                "Arden ratio = light peak / dark trough amplitude. "
                "NORMAL Arden ratio: ≥1.85 (>185%). "
                "PATHOGNOMONIC for BEST1 mutations: Arden ratio <1.5 in ALL carriers, "
                "even pre-symptomatic (stage 0 — normal fundus). "
                "CRITICAL: EOG is abnormal BEFORE any fundus change in VMD2 — "
                "enables carrier detection in at-risk relatives. "
                "EOG normal in Stargardt (ABCA4) — helps distinguish from Best disease. "
                "EOG also abnormal in choroideraemia (CHM) and fundus flavimaculatus."
            ),
            "Optical Coherence Tomography (OCT) in IRD": (
                "Non-invasive cross-sectional retinal imaging using near-infrared interferometry. "
                "NORMAL: distinct retinal layers visible (RPE, ellipsoid zone/IS-OS, outer nuclear layer etc.). "
                "RP: progressive outer nuclear layer thinning; loss of ellipsoid zone; later RPE loss. "
                "CRB1 UNIQUE: THICKENED retina — Müller cell hypertrophy causes increased total retinal thickness; "
                "the OPPOSITE of typical IRD thinning — PATHOGNOMONIC for CRB1. "
                "CMO: cystoid macular oedema — fluid-filled cysts in outer plexiform layer; "
                "treat with carbonic anhydrase inhibitors. "
                "Best disease: vitelliform material between RPE and ellipsoid zone; stage-dependent OCT."
            ),
            "Vitamin A Contraindication in ABCA4-Stargardt": (
                "Vitamin A (all-trans-retinol) is ABSOLUTELY CONTRAINDICATED in ABCA4-Stargardt disease. "
                "MECHANISM: vitamin A is the precursor to all-trans-retinal (atRAL); "
                "ABCA4 deficiency → atRAL accumulates in disc lumen → condenses to A2E bisretinoid; "
                "supplemental vitamin A → MORE atRAL substrate → MORE A2E → faster RPE toxicity. "
                "CONTRAST WITH RP: omega-3 and lutein/zeaxanthin may have some benefit in general RP "
                "(not contraindicated). Vitamin A benefit in X-linked RP claimed by one trial (Berson) "
                "but NOT recommended in ABCA4. "
                "CRITICAL PRESCRIBING ERROR: giving AREDS/AREDS2 vitamins (contain beta-carotene/vitamin A) "
                "to an ABCA4-Stargardt patient — ALWAYS check formulation."
            ),
            "Cystoid Macular Oedema (CMO) in RP": (
                "Accumulation of fluid in cyst-like spaces within the outer plexiform layer of the macula "
                "in retinitis pigmentosa patients — occurs in ~30-40% of RP (all genetic forms). "
                "MECHANISM: Müller cell dysfunction → K⁺ and water transport failure → fluid accumulation. "
                "NOT neovascular — anti-VEGF is INEFFECTIVE for RP-CMO. "
                "TREATMENT: carbonic anhydrase inhibitors (CAIs) — "
                "systemic acetazolamide 250-500 mg/day OR topical dorzolamide 2% TID; "
                "topical preferred for long-term; treat early before irreversible macular damage. "
                "MONITORING: OCT at each visit in all RP patients to detect CMO early."
            ),
            "Usher Syndrome Types 1, 2, 3": (
                "Usher syndrome: combined retinitis pigmentosa + sensorineural hearing loss ± vestibular dysfunction. "
                "TYPE 1 (USH1 — genes: MYO7A, CDH23, PCDH15, SANS, USH1G): "
                "PROFOUND CONGENITAL deafness + RP (onset childhood) + VESTIBULAR AREFLEXIA. "
                "Cochlear implant HIGHLY effective; deafblind rehabilitation essential. "
                "TYPE 2 (USH2 — genes: USH2A, ADGRV1, WHRN): "
                "MILD-MODERATE congenital SNHL + RP (later onset) + NORMAL VESTIBULAR. "
                "Hearing aids; may not need CI; USH2A most common (80% of USH2). "
                "TYPE 3 (USH3 — gene: CLRN1): "
                "PROGRESSIVE hearing loss (not congenital) + RP + variable vestibular. "
                "DISTINGUISHING VESTIBULAR: USH1 = vestibular areflexia (delayed walking); "
                "USH2 = normal vestibular; USH3 = variable."
            ),
            "Leber Congenital Amaurosis (LCA)": (
                "Severe rod-cone dystrophy presenting at birth or within the first year of life. "
                "Features: severe visual impairment from birth; nystagmus; absent or near-absent ERG; "
                "oculodigital sign (eye-pressing in infants); variable fundus appearance. "
                "MOST COMMON LCA GENES: CEP290 (LCA10, ~20-25%); GUCY2D (LCA1); CRB1 (LCA8); "
                "RPE65 (LCA2 — gene therapy Luxturna available!); RPGRIP1 (LCA6). "
                "RPE65-LCA2 TREATMENT: voretigene neparvovec (Luxturna) — FDA approved 2017; "
                "subretinal injection AAV2-RPE65; eligibility: biallelic RPE65 mutations, "
                "sufficient viable retinal cells. "
                "KEY: test ALL LCA patients for RPE65 mutation — treatable cause must not be missed."
            ),
            "Gene Therapy in IRD": (
                "Delivery of a functional copy of the mutated gene (or antisense oligonucleotide to correct splicing) "
                "to restore retinal function. "
                "ROUTES: subretinal injection (under retina, into subretinal space — high local concentration, "
                "immune-privileged) vs intravitreal injection (into vitreous — less invasive, wider distribution). "
                "VECTORS: AAV (adeno-associated virus) most common — AAV2, AAV5, AAV8, AAV9; "
                "lentiviral for large genes (MYO7A too large for AAV — lentiviral preferred). "
                "APPROVED: Luxturna (voretigene neparvovec, RPE65, FDA 2017) — only approved IRD gene therapy. "
                "IN TRIALS: RPGR (AAV-RPGR), CEP290-IVS26 (EDIT-101 antisense), USH2A (QR-421a), "
                "RLBP1, CNGB3. "
                "PRINCIPLE: treat early, when photoreceptors still viable — gene therapy cannot restore "
                "cells that have already degenerated."
            ),
            "IVS26+1655A>G (CEP290 Founder Variant)": (
                "The most common pathogenic CEP290 variant causing LCA10. "
                "Location: deep intronic — 1655 bp downstream of exon 26 (intron 26). "
                "Mechanism: creates an aberrant splice donor site → inclusion of a cryptic exon (128 bp) → "
                "frameshift → premature stop codon → truncated/absent CEP290 protein. "
                "CRITICAL: NOT detected by standard exon-capture NGS panels. "
                "Requires: (1) whole-genome sequencing; or (2) specific intronic assay; or "
                "(3) targeted deep-intronic panel; or (4) Sanger sequencing of intron 26. "
                "Carrier frequency: ~1:290 in Europeans. "
                "TREATMENT TARGET: EDIT-101 antisense oligonucleotide (IONIS/Editas) targets the "
                "aberrant splice site → blocks cryptic exon inclusion → restores normal CEP290 splicing. "
                "Phase I/II ILLUMINATE trial showed visual function improvement."
            ),
            "Vestibular Areflexia (Usher 1)": (
                "Absence of vestibular (labyrinthine) reflex responses — pathognomonic of Usher type 1. "
                "TESTS: ice water caloric irrigation (no eye movement response — ABSENT caloric reflex); "
                "Video Head Impulse Test (vHIT) — absent VOR gain; "
                "VEMPS (vestibular evoked myogenic potentials) — absent. "
                "CLINICAL CONSEQUENCES: "
                "delayed motor milestones — most USH1 children walk at 18-24 months (normal: ~12 months); "
                "Romberg positive (sway with eyes closed); "
                "falls in dark or uneven terrain (no vestibular compensation); "
                "CANNOT swim safely unsupervised (no righting reflex underwater). "
                "REHABILITATION: physiotherapy for balance training; "
                "compensation via visual and proprioceptive pathways; "
                "fall prevention strategies. "
                "ALL USH1 GENES cause vestibular areflexia — MYO7A, CDH23, PCDH15, SANS, USH1G."
            ),
            "Incomplete Penetrance (PRPF31-adRP)": (
                "A genetic phenomenon where not all individuals carrying a disease-causing variant "
                "develop the phenotype. Penetrance = proportion of mutation carriers who show the disease. "
                "PRPF31-adRP has ~50–80% penetrance — 20–50% of carriers are completely asymptomatic "
                "with normal fundus and normal ERG throughout life. "
                "MOLECULAR MECHANISM: the WT PRPF31 allele can be upregulated by a modifier locus "
                "(CNOT3 on chromosome 14q); if WT allele is sufficiently upregulated → enough PRPF31 → "
                "no RP phenotype (non-penetrant carrier). "
                "GENETIC COUNSELLING IMPLICATIONS: "
                "- An unaffected parent who carries the PRPF31 mutation is a NON-PENETRANT CARRIER "
                "(not a genetic escape — they still transmit the mutation to offspring at 50%). "
                "- Offspring of non-penetrant carriers: 50% inherit mutation × 50-80% penetrance = "
                "25-40% risk of RP. "
                "- CANNOT predict penetrance from PRPF31 genotype alone (modifier locus testing "
                "not clinically validated). "
                "- MUST explain this to families — do not reassure 'unaffected parent means safe.'"
            ),
            "Para-Arteriolar RPE Preservation (PPRPE — CRB1)": (
                "A distinctive fundus finding pathognomonic for CRB1-associated RP (RP12). "
                "APPEARANCE: islands or ribbons of preserved retinal pigment epithelium (RPE) "
                "that follow the course of retinal arterioles — creating a characteristic "
                "pattern of preserved tissue adjacent to blood vessels surrounded by degenerated retina. "
                "MECHANISM: retinal arterioles may provide paracrine survival signals to adjacent RPE — "
                "speculative, but the topographic association is consistent. "
                "SIGNIFICANCE: PATHOGNOMONIC for CRB1-RP12; seeing PPRPE on fundus examination "
                "should immediately prompt CRB1 genetic testing. "
                "DISTINGUISH from normal perivascular sparing in other conditions."
            ),
            "Fundus Autofluorescence (FAF) in IRD": (
                "Non-invasive imaging using lipofuscin autofluorescence properties of RPE cells "
                "to assess RPE health and map disease progression. "
                "LIPOFUSCIN: waste products (A2E and bisretinoids) accumulate in RPE lysosomes — "
                "autofluoresce at 488 nm excitation / ~520 nm emission. "
                "RP: perifoveal ring of INCREASED AF (boundary between preserved and degenerated retina); "
                "as RP advances, ring contracts centripetally; DECREASED AF = RPE loss. "
                "STARGARDT (ABCA4): INCREASED AF in flecks (A2E-laden RPE) → "
                "DECREASED AF in areas of RPE atrophy (lost RPE). "
                "FAF is used as primary outcome measure in IRD clinical trials "
                "(rate of AF atrophy zone enlargement)."
            ),
        },
    }
