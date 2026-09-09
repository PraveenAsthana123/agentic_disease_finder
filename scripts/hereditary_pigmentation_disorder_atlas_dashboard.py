#!/usr/bin/env python3
"""Hereditary-Pigmentation-Disorder-Atlas — Complete 8-Gene Hereditary Pigmentation Disorder Atlas
(TYR · OCA2 · TYRP1 · SLC45A2 · HPS1 · LYST · KIT · MC1R).

TYR      (Tyrosinase; 529 aa; 11q14.3; AR;
          Oculocutaneous Albinism Type 1 (OCA1A / OCA1B);
          COMPLETE ABSENCE OF PIGMENT IN SKIN/HAIR/EYES PATHOGNOMONIC (OCA1A);
          Nystagmus + photophobia + foveal hypoplasia = ocular triad mandatory;
          OCA1A: zero tyrosinase activity — white hair + pink eyes lifelong;
          OCA1B: residual activity — yellow-blonde hair accrues with age;
          SPF50 mandatory lifelong (melanoma risk without UV protection);
          seed SEED_BASE+0).
OCA2     (OCA2 P-protein; 838 aa; 15q13.3; AR;
          Oculocutaneous Albinism Type 2 (OCA2);
          Most common OCA worldwide, most common OCA sub-Saharan Africa (prevalence 1:3900);
          Yellow-blonde hair/pale-cream skin in dark-skinned populations PATHOGNOMONIC;
          15q11-q13 deletion: when biallelic hits OCA2 → concurrent Prader-Willi/Angelman;
          seed SEED_BASE+1).
TYRP1    (Tyrosinase-Related Protein 1; 537 aa; 9p23; AR;
          Oculocutaneous Albinism Type 3 (OCA3 / Rufous albinism);
          RED-RUFOUS HAIR + BRONZE SKIN + BROWN EYES in Africans PATHOGNOMONIC;
          Ocular features milder than OCA1/OCA2 — often missed/underdiagnosed;
          Underdiagnosed worldwide — pigmentation reduced but NOT absent;
          seed SEED_BASE+2).
SLC45A2  (Solute Carrier Family 45 Member 2 / MATP; 530 aa; 5p13.2; AR;
          Oculocutaneous Albinism Type 4 (OCA4);
          Most common OCA in Japan (>70% of Japanese OCA);
          MILD-TO-MODERATE CREAM SKIN + NYSTAGMUS PATHOGNOMONIC;
          Variable expressivity: near-normal to OCA1B-like pale;
          seed SEED_BASE+3).
HPS1     (Hermansky-Pudlak Syndrome 1 protein; 700 aa; 10q24.2; AR;
          Hermansky-Pudlak Syndrome Type 1 (HPS1);
          OCA + PLATELET DENSE GRANULE DEFICIENCY + CEROID ACCUMULATION TRIAD PATHOGNOMONIC;
          Puerto Rican founder (p.Gln116fs 16bp del, 1:1800 prevalence northwest Puerto Rico);
          PULMONARY FIBROSIS (interstitial lung disease) lethal 3rd-4th decade PATHOGNOMONIC;
          NO ASPIRIN/NSAIDs EVER — platelet aggregation defect, bleeding risk;
          seed SEED_BASE+4).
LYST     (Lysosomal Trafficking Regulator; 3801 aa; 1q42.3; AR;
          Chédiak-Higashi Syndrome (CHS);
          PARTIAL ALBINISM + GIANT PEROXIDASE-POSITIVE GRANULES IN NEUTROPHILS PATHOGNOMONIC;
          Silver-grey hair + recurrent staphylococcal/streptococcal infections;
          Accelerated Phase = Hemophagocytic Lymphohistiocytosis (HLH) — LETHAL without HSCT;
          HSCT only curative treatment — corrects HLH/infections, NOT neurological progression;
          seed SEED_BASE+5).
KIT      (KIT proto-oncogene receptor tyrosine kinase; 976 aa; 4q12; AD;
          Piebaldism;
          WHITE FORELOCK + STABLE DEPIGMENTED PATCHES (ventral/abdominal/extremity) PATHOGNOMONIC;
          PRESENT FROM BIRTH AND STABLE (non-progressive) = KEY DDx from vitiligo (acquired/progressive);
          White patch on frontal scalp + triangular abdominal patch PATHOGNOMONIC;
          No systemic features — isolated pigmentation defect;
          seed SEED_BASE+6).
MC1R     (Melanocortin 1 Receptor; 317 aa; 16q24.3; AD incomplete penetrance;
          Red Hair Color (RHC) / Melanoma Predisposition Syndrome;
          RED HAIR + FAIR SKIN + EPHELIDES (freckles) + MELANOMA RISK PATHOGNOMONIC;
          R/R variants (Arg151Cys/Arg160Trp/Asp294His) = highest melanoma risk (4-10x);
          SPF50 mandatory lifelong + annual total-body dermoscopy;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2302-2309).
"""

import random

SEED_BASE = 2302

PIGMENT_GENES = [
    # -- TYR — OCA1 -----------------------------------------------------------------------
    {
        "gene": "TYR",
        "alt_name": (
            "TYR (TYR-529aa-11q14.3 / AR — Oculocutaneous-Albinism-Type-1-OCA1A-OCA1B — "
            "COMPLETE-ABSENCE-PIGMENT-SKIN-HAIR-EYES-OCA1A-PATHOGNOMONIC — "
            "NYSTAGMUS+PHOTOPHOBIA+FOVEAL-HYPOPLASIA-OCULAR-TRIAD — "
            "SPF50-Mandatory-Lifelong-Melanoma-Risk)"
        ),
        "protein": (
            "TYR -- 11q14.3 AR -- TYR-529aa -- "
            "Tyrosinase-58kDa-Copper-Containing-Monooxygenase-Melanosome-Membrane -- "
            "OCA1A-OMIM-203100-Zero-Tyrosinase-Activity -- "
            "OCA1B-OMIM-606952-Residual-Activity-Yellow-Blonde-Hair-Accrues-Age -- "
            "TYR-L-DOPA-Oxidase-L-Tyrosine→L-DOPA→Dopaquinone→Eumelanin-Phaeomelanin -- "
            "WHITE-HAIR-PINK-SKIN-PINK-BLUE-EYES-OCA1A-COMPLETE-NO-PIGMENT-LIFETIME -- "
            "YELLOW-BLONDE-HAIR-BY-AGE-1-2yr-OCA1B-RESIDUAL-MELANOSOME-ENZYME -- "
            "NYSTAGMUS-BILATERAL-PENDULAR-JERK-PRESENT-BIRTH-ALL-OCA1 -- "
            "PHOTOPHOBIA-SEVERE-SUN-AVOIDANCE-UV-BURN-MINUTES-WITHOUT-SPF -- "
            "FOVEAL-HYPOPLASIA-REDUCED-VISUAL-ACUITY-20-200-Typical-No-Cure -- "
            "MISROUTED-OPTIC-TRACTS-Chiasmal-Decussation-Defect-VEP-DIAGNOSTIC -- "
            "OMIM-Gene-TYR-606933-Disease-OCA1-203100"
        ),
        "locus": "11q14.3",
        "protein_size": "529 aa / 58 kDa",
        "inheritance": (
            "AR; biallelic TYR mutations; de novo rare; both parents obligate carriers (25% recurrence); "
            "OCA1A: typically biallelic null (frameshift/nonsense/splice) → zero activity; "
            "OCA1B: compound heterozygote null + hypomorphic → residual activity; "
            "carrier testing recommended family members; prenatal diagnosis CVS/amniocentesis"
        ),
        "pigment_category": "OCA — Oculocutaneous Albinism Type 1 (TYR, zero or reduced tyrosinase)",
        "pathognomonic": (
            "OCA1A: COMPLETE ABSENCE OF MELANIN — white hair at birth (stays white lifelong), "
            "translucent pink-white skin (no tan ever), pink/light-blue eyes (iris transilluminates); "
            "OCA1B: yellow-blonde pigment accrues in first 1-2 years (residual enzyme); "
            "NYSTAGMUS (bilateral pendular, present from birth) + PHOTOPHOBIA + FOVEAL HYPOPLASIA = "
            "ocular triad PATHOGNOMONIC for all OCA types; VEP shows chiasmal misrouting"
        ),
        "treatment": (
            "SPF50+ BROAD-SPECTRUM SUNSCREEN mandatory lifelong — no UV protection at all in OCA1A; "
            "Annual dermatology: SCC/BCC/melanoma surveillance from age 10yr; "
            "Low vision rehabilitation: high-power spectacles, magnifiers, tinted lenses (photophobia); "
            "Nystagmus: base-out prisms (null-point head turn correction), botulinum toxin trialled; "
            "Ophthalmology: refraction (high myopia/hyperopia common), amblyopia treatment if asymmetric; "
            "NO curative therapy — gene therapy trials ongoing for OCA1B; "
            "Hat + UPF50 clothing + tinted wrap-around glasses outdoors mandatory; "
            "Ocular motility surgery if severe anomalous head posture (nystagmus null-point surgery)"
        ),
        "key_features": [
            "OCA1A: WHITE hair + PINK-WHITE skin + PINK/BLUE eyes from birth, NO pigment ever — COMPLETE ABSENCE",
            "OCA1B: YELLOW-BLONDE hair accrues age 1-2yr (residual tyrosinase), skin/eyes very pale",
            "NYSTAGMUS — bilateral pendular or jerk, present from birth, improves slightly with age",
            "PHOTOPHOBIA — severe, outdoor activity requires wraparound tinted glasses + hat",
            "FOVEAL HYPOPLASIA — reduced visual acuity (typically 20/100–20/400), nystagmus drives this",
            "OPTIC TRACT MISROUTING — chiasmal decussation defect; VEP asymmetry = diagnostic signature",
        ],
        "monitoring": [
            "Annual total-body dermatology examination: SCC/BCC/melanoma from age 10yr",
            "Ophthalmology 6-monthly to age 5yr (refractive + amblyopia screen); annual thereafter",
            "VEP (visual evoked potential) at diagnosis: confirms chiasmal misrouting (diagnostic)",
            "MRI brain if nystagmus atypical (rule out cerebellar/brainstem pathology)",
            "Vitamin D supplementation: sun avoidance + SPF → vitamin D deficiency common",
            "Annual ophthalmology including motility, refraction, IOP — nystagmus management",
        ],
        "key_ddx": [
            "OCA2 (OCA2/P-protein — most common OCA, yellow-blonde pigment in dark races; some eye color)",
            "OCA4/SLC45A2 (most common in Japan; variable pale-cream; VEP confirms OCA vs CSNB)",
            "Hermansky-Pudlak (HPS1 — albinism + easy bruising + prolonged bleeding; platelet delta granules absent)",
            "Chédiak-Higashi (LYST — silver-grey hair + giant neutrophil granules; infections; HLH accelerated phase)",
        ],
        "melanoma_risk": True,
        "bleeding_risk": False,
        "hlh_risk": False,
        "stable_pigment": False,
        "ocular_emergency": False,
        "severity_options": [
            "OCA1A — complete (no pigment ever, white hair lifelong, highest melanoma risk)",
            "OCA1B — partial (yellow-blonde hair accrues, pale skin, melanoma risk high)",
        ],
        "systemic_options": ["Nystagmus", "Photophobia", "Foveal hypoplasia", "Reduced VA", "Sun sensitivity", "Optic misrouting"],
        "treatments_used": ["SPF50 sunscreen", "Tinted glasses", "Low vision aids", "Hat + UPF clothing", "Prisms", "Annual dermatology"],
    },
    # -- OCA2 — OCA2 -----------------------------------------------------------------------
    {
        "gene": "OCA2",
        "alt_name": (
            "OCA2 (OCA2-838aa-15q13.3 / AR — Oculocutaneous-Albinism-Type-2 — "
            "Most-Common-OCA-Worldwide-Most-Common-Sub-Saharan-Africa-1:3900 — "
            "YELLOW-BLONDE-HAIR-PALE-CREAM-SKIN-DARK-RACES-PATHOGNOMONIC — "
            "15q11-q13-Deletion-Prader-Willi-Angelman-Overlap)"
        ),
        "protein": (
            "OCA2 -- 15q13.3 AR -- OCA2-838aa -- "
            "P-Protein-110kDa-12-Pass-Transmembrane-Melanosomal-Membrane-Transporter -- "
            "OCA2-Most-Common-OCA-Worldwide-Prevalence-1-40000-Europe-1-3900-Africa -- "
            "YELLOW-BLONDE-HAIR-PALE-CREAM-SKIN-IN-DARK-SKINNED-POPULATIONS-PATHOGNOMONIC -- "
            "SKY-BLUE-TO-HAZEL-EYE-COLOR-Variable-Iris-Pigmentation -- "
            "PRADER-WILLI-ANGELMAN-15q11-q13-DELETION-HITS-OCA2-BIALLELIC -- "
            "P-Protein-Regulates-Melanosomal-pH-Tyrosine-Transport-Eumelanin-Synthesis -- "
            "NYSTAGMUS-PHOTOPHOBIA-FOVEAL-HYPOPLASIA-Universal-All-OCA2 -- "
            "p.Arg305Trp-Most-Common-European-OCA2-Allele -- "
            "p.Val443Ile-Plus-IVS17-nt-3-African-Founder-Haplotype -- "
            "OMIM-Gene-OCA2-611409-Disease-OCA2-203200"
        ),
        "locus": "15q13.3",
        "protein_size": "838 aa / 110 kDa",
        "inheritance": (
            "AR; biallelic OCA2 mutations; 15q11-q13 deletion (Prader-Willi/Angelman region) "
            "may simultaneously disrupt OCA2 locus → syndromic albinism + PWS/AS; "
            "founder haplotypes in African populations; p.Arg305Trp most common European allele; "
            "carrier frequency 1:8 in sub-Saharan Africa (high prevalence)"
        ),
        "pigment_category": "OCA — Oculocutaneous Albinism Type 2 (OCA2/P-protein, most common OCA worldwide)",
        "pathognomonic": (
            "YELLOW-BLONDE HAIR + PALE CREAM SKIN in individuals from dark-skinned populations "
            "(sub-Saharan African, Native American, East Asian) PATHOGNOMONIC — striking contrast; "
            "European OCA2: white-to-cream skin, light-blonde to white hair; "
            "Eye color variable: sky-blue to hazel (iris pigment partially present, unlike OCA1A pink); "
            "NYSTAGMUS + PHOTOPHOBIA + FOVEAL HYPOPLASIA — same ocular triad as all OCA types"
        ),
        "treatment": (
            "SPF50 + UPF50 clothing mandatory (melanoma + SCC risk, especially in high-UV countries); "
            "CRITICAL in sub-Saharan Africa where OCA prevalence high and UV exposure extreme — "
            "SPF provision access is a public-health issue; "
            "Annual dermatology: SCC common in equatorial Africa in OCA patients; "
            "Low vision rehabilitation: tinted lenses, magnifiers, large-print aids; "
            "Ophthalmology: refraction + amblyopia; nystagmus management; "
            "Prader-Willi/Angelman workup if 15q deletion suspected (OGD/FISH/methylation PCR); "
            "Vitamin D: sun avoidance paradox — supplementation needed"
        ),
        "key_features": [
            "Most common OCA worldwide and most common single-gene cause of albinism in sub-Saharan Africa (1:3900)",
            "YELLOW-BLONDE hair + PALE CREAM skin in dark-pigmented races — striking contrast = pathognomonic",
            "Eye color VARIABLE (sky-blue to hazel) — NOT always pink/achromatic unlike OCA1A",
            "15q11-q13 deletion overlap: simultaneous OCA2 + Prader-Willi or Angelman syndrome possible",
            "p.Arg305Trp: most common European OCA2 allele; founder haplotypes in African populations",
            "Nystagmus + photophobia + foveal hypoplasia — universal ocular triad (same as OCA1)",
        ],
        "monitoring": [
            "Annual dermatology: SCC/melanoma — especially critical in high-UV (equatorial) settings",
            "Ophthalmology 6-monthly to age 5yr; annual thereafter",
            "15q methylation study / SNP array if developmental delay or hypotonia: rule out Prader-Willi/Angelman",
            "VEP: confirms chiasmal misrouting (diagnostic for OCA type)",
            "Vitamin D: supplementation as sun avoidance + SPF prevents synthesis",
            "Psychosocial: stigma significant in equatorial Africa — counselling/community support essential",
        ],
        "key_ddx": [
            "TYR-OCA1A (zero pigment lifelong — white hair, pink eyes; no yellow hair accrual ever)",
            "TYR-OCA1B (yellow-blonde accrues but pink/blue eyes, European prevalence)",
            "SLC45A2-OCA4 (most common in Japan; variable pale; molecular panel distinguishes)",
            "Prader-Willi/Angelman + OCA2 (15q11-q13 deletion — check for PWS features if OCA2 + hypotonia)",
        ],
        "melanoma_risk": True,
        "bleeding_risk": False,
        "hlh_risk": False,
        "stable_pigment": False,
        "ocular_emergency": False,
        "severity_options": [
            "Moderate (cream skin, light-blonde hair, partial iris pigment — common European OCA2)",
            "Moderate-severe (yellow-blonde hair + pale cream skin in African/dark-skinned patients)",
        ],
        "systemic_options": ["Nystagmus", "Photophobia", "Foveal hypoplasia", "Sun sensitivity", "Reduced VA", "Social stigma"],
        "treatments_used": ["SPF50 sunscreen", "UPF clothing", "Tinted glasses", "Low vision aids", "Annual dermatology", "Vitamin D"],
    },
    # -- TYRP1 — OCA3 -----------------------------------------------------------------------
    {
        "gene": "TYRP1",
        "alt_name": (
            "TYRP1 (TYRP1-537aa-9p23 / AR — Oculocutaneous-Albinism-Type-3-OCA3-Rufous-Albinism — "
            "RED-RUFOUS-HAIR+BRONZE-SKIN+BROWN-EYES-Africans-PATHOGNOMONIC — "
            "MILDER-OCULAR-FEATURES-Than-OCA1-OCA2-Often-Missed-Underdiagnosed)"
        ),
        "protein": (
            "TYRP1 -- 9p23 AR -- TYRP1-537aa -- "
            "Tyrosinase-Related-Protein-1-75kDa-DHICA-Oxidase-Melanosome-Membrane -- "
            "OCA3-Rufous-Albinism-Most-Common-In-Sub-Saharan-Africa-Papua-New-Guinea -- "
            "RED-RUFOUS-HAIR-Reddish-Brown-Orange-Red-Tones-PATHOGNOMONIC-Dark-Races -- "
            "BRONZE-SKIN-Reddish-Hypopigmented-Not-White-Pigmentation-Reduced-Not-Absent -- "
            "BROWN-HAZEL-EYES-Variable-Iris-Pigment-Often-Misidentified-as-Normal -- "
            "TYRP1-Stabilizes-TYR-Protein-Cofactor-DHI-DHICA-Polymerization-Eumelanin -- "
            "OCULAR-FEATURES-MILDER-Than-OCA1-OCA2-Nystagmus-Less-Severe-Or-Absent -- "
            "UNDERDIAGNOSED-Worldwide-Pigmentation-Reduced-Not-Absent -- "
            "OMIM-Gene-TYRP1-115501-Disease-OCA3-203290"
        ),
        "locus": "9p23",
        "protein_size": "537 aa / 75 kDa",
        "inheritance": (
            "AR; biallelic TYRP1 mutations; primarily reported in sub-Saharan African and "
            "Papua New Guinean populations (founder mutations); underdiagnosed globally; "
            "molecular panel essential as clinical recognition often difficult"
        ),
        "pigment_category": "OCA — Oculocutaneous Albinism Type 3 (TYRP1, rufous/red-bronze phenotype — reduced eumelanin)",
        "pathognomonic": (
            "RED-RUFOUS HAIR (reddish-brown to orange-red) + BRONZE SKIN + BROWN/HAZEL EYES "
            "in individuals from sub-Saharan African or Papua New Guinean populations PATHOGNOMONIC; "
            "PIGMENTATION REDUCED BUT NOT ABSENT = KEY DDx from OCA1A (absent); "
            "OCULAR FEATURES MILDER than OCA1/OCA2 — nystagmus may be absent or minimal; "
            "Easily missed or diagnosed as 'red hair' variant rather than OCA subtype"
        ),
        "treatment": (
            "SPF50 + UPF clothing — melanoma/SCC risk present but lower than OCA1/OCA2 (some pigment protection); "
            "Annual dermatology from age 10yr: SCC risk in high-UV equatorial settings; "
            "Low vision assessment: refraction + amblyopia if nystagmus present; "
            "Tinted glasses if photophobia (milder than OCA1 but may still be present); "
            "Molecular diagnosis mandatory — clinical recognition often missed; "
            "Genetic counselling: distinguish from OCA1/OCA2/OCA4 for accurate recurrence risk; "
            "Psychosocial support: appearance-related stigma in African communities"
        ),
        "key_features": [
            "RED-RUFOUS HAIR (reddish-brown to orange-red tones) in Africans — pathognomonic",
            "BRONZE SKIN — reddish-hypopigmented, NOT white; pigment reduced not absent",
            "BROWN/HAZEL EYES — iris pigment partially present; often normal-appearing eye color",
            "OCULAR FEATURES MILDER — nystagmus absent or minimal in some patients (unlike OCA1/2)",
            "UNDERDIAGNOSED — clinical pattern easily missed; molecular panel required for confirmation",
            "TYRP1 stabilizes tyrosinase; DHICA oxidase activity; eumelanin polymerization cofactor",
        ],
        "monitoring": [
            "Annual dermatology: SCC/melanoma — equatorial Africa high UV exposure + reduced melanin",
            "Ophthalmology: VA + refraction + VEP (confirm OCA ocular signature even if mild)",
            "Molecular confirmation mandatory: NGS OCA panel (TYR/OCA2/TYRP1/SLC45A2) to distinguish subtype",
            "Psychosocial assessment: stigma management in community/school settings",
            "Vitamin D: supplementation if SPF use high + sun avoidance",
            "Family cascade: siblings/parents carrier testing once proband molecular confirmed",
        ],
        "key_ddx": [
            "TYR-OCA1B (yellow-blonde hair; European predominance; TYR mutation confirmed; ocular features similar)",
            "OCA2 (yellow-blonde hair; most common African OCA; molecular panel distinguishes TYRP1 vs OCA2)",
            "Red hair (MC1R variants — European; NOT albinism; full pigmentation; no nystagmus/foveal hypoplasia)",
            "SLC45A2-OCA4 (mild-moderate pale skin; Japanese commonest; TYRP1/OCA4 molecular distinction)",
        ],
        "melanoma_risk": True,
        "bleeding_risk": False,
        "hlh_risk": False,
        "stable_pigment": False,
        "ocular_emergency": False,
        "severity_options": [
            "Mild (bronze skin, rufous hair, near-normal eyes — often missed clinically)",
            "Moderate (reddish-orange hair, pale bronze skin, mild nystagmus)",
        ],
        "systemic_options": ["Rufous hair", "Bronze skin hypopigmentation", "Mild nystagmus", "Photophobia", "Sun sensitivity"],
        "treatments_used": ["SPF50 sunscreen", "UPF clothing", "Tinted glasses", "Low vision assessment", "Annual dermatology"],
    },
    # -- SLC45A2 — OCA4 -----------------------------------------------------------------------
    {
        "gene": "SLC45A2",
        "alt_name": (
            "SLC45A2 (SLC45A2-530aa-5p13.2 / AR — Oculocutaneous-Albinism-Type-4-OCA4-MATP — "
            "Most-Common-OCA-Japan->70pct-Japanese-OCA — "
            "MILD-CREAM-SKIN+NYSTAGMUS-PATHOGNOMONIC — "
            "Variable-Expressivity-Near-Normal-To-OCA1B-Like)"
        ),
        "protein": (
            "SLC45A2 -- 5p13.2 AR -- SLC45A2-530aa -- "
            "MATP-AIM1-58kDa-12-Pass-Transmembrane-Melanosomal-H+-Sucrose-Transporter -- "
            "OCA4-Most-Common-OCA-Japan->70pct-Japanese-OCA-Patients -- "
            "CREAM-WHITE-SKIN-VERY-PALE-HAIR-VARIABLE-EXPRESSIVITY -- "
            "MILD-TO-MODERATE-PHENOTYPE-Variable-Residual-Transporter-Activity -- "
            "NYSTAGMUS-PRESENT-PATHOGNOMONIC-Photophobia-Foveal-Hypoplasia-All-OCA4 -- "
            "MATP-Membrane-Associated-Transporter-Protein-Regulates-Melanosomal-pH -- "
            "Near-Normal-Pigmentation-Possible-Mild-End-Spectrum -- "
            "OCA4-Underrecognized-Europe-Americas-vs-Japan -- "
            "OMIM-Gene-SLC45A2-606202-Disease-OCA4-606574"
        ),
        "locus": "5p13.2",
        "protein_size": "530 aa / 58 kDa",
        "inheritance": (
            "AR; biallelic SLC45A2 mutations; highly prevalent in Japan (>70% Japanese OCA); "
            "underrecognized in European and American OCA panels; "
            "variable expressivity (near-normal to OCA1B-like pale); "
            "molecular panel essential to distinguish from OCA1/OCA2"
        ),
        "pigment_category": "OCA — Oculocutaneous Albinism Type 4 (SLC45A2/MATP, most common OCA in Japan)",
        "pathognomonic": (
            "MILD-TO-MODERATE PALE CREAM SKIN + NYSTAGMUS PATHOGNOMONIC; "
            "hair: very pale yellow to near-white; skin: pale-cream, may tan minimally; "
            "OCA4 in Japan: most common form; >70% Japanese OCA patients have SLC45A2 mutations; "
            "VARIABLE EXPRESSIVITY: near-normal pigmentation at mild end to OCA1B-like pale at severe; "
            "ocular triad: nystagmus + photophobia + foveal hypoplasia (as all OCA types)"
        ),
        "treatment": (
            "SPF50 + UPF clothing mandatory (melanoma/SCC risk proportional to hypopigmentation level); "
            "Annual dermatology from age 10yr; "
            "Ophthalmology: refraction + amblyopia + nystagmus management (prisms, motility surgery if severe); "
            "Low vision rehabilitation: magnifiers, large-print, tinted lenses; "
            "Molecular panel critical (OCA4 often missed on TYR/OCA2-only panels); "
            "Vitamin D monitoring: SPF use + sun avoidance; "
            "Genetic counselling: Japan-specific recurrence risk 25% AR; prenatal diagnosis available"
        ),
        "key_features": [
            "Most common OCA in Japan: >70% of Japanese OCA patients carry SLC45A2 mutations",
            "PALE CREAM SKIN — variable from near-normal to very pale; hair pale yellow to near-white",
            "VARIABLE EXPRESSIVITY — near-normal at mild end to OCA1B-like at severe end",
            "NYSTAGMUS + PHOTOPHOBIA + FOVEAL HYPOPLASIA — universal OCA ocular triad",
            "MATP/SLC45A2 regulates melanosomal pH → disrupted tyrosine transport → reduced melanin",
            "Often missed by TYR/OCA2-only panels — full OCA gene panel required",
        ],
        "monitoring": [
            "Annual dermatology: SCC/melanoma — risk proportional to degree of hypopigmentation",
            "Ophthalmology 6-monthly to age 5yr; annually thereafter",
            "Full OCA NGS panel (TYR/OCA2/TYRP1/SLC45A2/HPS1-8/LYST) at diagnosis",
            "VEP: chiasmal misrouting confirmation (all OCA types)",
            "Vitamin D: supplementation monitoring",
            "Psychosocial: appearance concerns, especially in Japan where OCA4 most common",
        ],
        "key_ddx": [
            "TYR-OCA1B (yellow-blonde hair accrual; TYR mutation; European prevalence)",
            "OCA2 (most common worldwide; OCA2/P-protein mutation; yellow-blonde in dark races)",
            "TYRP1-OCA3 (rufous-bronze hair-skin in Africans; TYRP1 mutation; mild ocular features)",
            "Isolated nystagmus (CSNB, INS — no OCA pigmentation; VEP OCA pattern = misrouting vs normal)",
        ],
        "melanoma_risk": True,
        "bleeding_risk": False,
        "hlh_risk": False,
        "stable_pigment": False,
        "ocular_emergency": False,
        "severity_options": [
            "Mild (near-normal pigmentation, mild nystagmus — may be undiagnosed until molecular panel)",
            "Moderate (pale cream skin, very light hair, nystagmus evident)",
        ],
        "systemic_options": ["Nystagmus", "Photophobia", "Foveal hypoplasia", "Pale skin", "Sun sensitivity"],
        "treatments_used": ["SPF50 sunscreen", "UPF clothing", "Tinted glasses", "Low vision aids", "Annual dermatology", "Prisms"],
    },
    # -- HPS1 — Hermansky-Pudlak Syndrome Type 1 -------------------------------------------
    {
        "gene": "HPS1",
        "alt_name": (
            "HPS1 (HPS1-700aa-10q24.2 / AR — Hermansky-Pudlak-Syndrome-Type-1 — "
            "OCA+PLATELET-DENSE-GRANULE-DEFICIENCY+CEROID-ACCUMULATION-TRIAD-PATHOGNOMONIC — "
            "PULMONARY-FIBROSIS-LETHAL-3rd-4th-Decade-PATHOGNOMONIC — "
            "NO-ASPIRIN-NSAIDs-EVER-Platelet-Aggregation-Defect)"
        ),
        "protein": (
            "HPS1 -- 10q24.2 AR -- HPS1-700aa -- "
            "HPS1-Protein-80kDa-Biogenesis-of-Lysosome-Related-Organelles-Complex-3-BLOC-3-Subunit -- "
            "HPS1-BLOC-3-Melanosome-Platelet-Dense-Granule-Biogenesis -- "
            "OCA-ALBINISM-Universal-All-HPS-Types-Melanosome-Biogenesis-Defect -- "
            "PLATELET-DENSE-GRANULE-ABSENT-ADP-Serotonin-Zero-Output-Prolonged-Bleeding -- "
            "CEROID-ACCUMULATION-Lipofuscin-Like-Material-Bowel-Kidney-Lung -- "
            "PULMONARY-FIBROSIS-Usual-Interstitial-Pneumonia-Pattern-Lethal-3rd-4th-Decade -- "
            "PUERTO-RICO-FOUNDER-p.Gln116fs-16bp-DEL-1-1800-Northwest-Puerto-Rico -- "
            "NO-ASPIRIN-NSAIDs-CONTRAINDICATED-ABSOLUTE-Platelet-Function-Defect -- "
            "PIRFENIDONE-Slows-Pulmonary-Fibrosis-Progression -- "
            "OMIM-Gene-HPS1-604982-Disease-HPS1-203300"
        ),
        "locus": "10q24.2",
        "protein_size": "700 aa / 80 kDa",
        "inheritance": (
            "AR; biallelic HPS1 mutations; Puerto Rican founder: p.Gln116fs 16bp deletion "
            "= prevalence 1:1800 northwest Puerto Rico (1:18 carrier); "
            "European/other: compound heterozygotes common; "
            "8 HPS genes (HPS1-3,5-7,HPS9,AP3B1) — HPS1 most severe pulmonary phenotype"
        ),
        "pigment_category": "Hermansky-Pudlak Syndrome — OCA + Platelet Dense Granule Deficiency + Ceroid Accumulation (BLOC-3)",
        "pathognomonic": (
            "OCULOCUTANEOUS ALBINISM (same as OCA1B-like, variable pale) + "
            "EASY BRUISING / PROLONGED BLEEDING (platelet dense granule absent — "
            "no ADP/serotonin release, prolonged bleeding time despite NORMAL platelet count) + "
            "PULMONARY FIBROSIS (UIP pattern, lethal 3rd-4th decade) TRIAD PATHOGNOMONIC for HPS; "
            "PLATELET DENSE GRANULES ABSENT ON ELECTRON MICROSCOPY = DEFINITIVE DIAGNOSTIC; "
            "NO ASPIRIN/NSAIDs EVER — contraindicated absolutely"
        ),
        "treatment": (
            "NO ASPIRIN/NSAIDs ABSOLUTE CONTRAINDICATION — platelet function defect; "
            "Desmopressin (DDAVP) 0.3 mcg/kg IV before surgery/invasive procedures; "
            "Platelet transfusion if major bleeding; "
            "PIRFENIDONE — anti-fibrotic, slows pulmonary fibrosis progression (approved); "
            "Nintedanib — alternative anti-fibrotic; "
            "LUNG TRANSPLANT — for end-stage pulmonary fibrosis if candidate; "
            "Pulmonary function tests (FVC, DLCO) annually from age 20yr; "
            "HRCT chest baseline then 2-yearly (UIP pattern monitoring); "
            "SPF50 + annual dermatology (albinism — melanoma/SCC risk); "
            "Bowel surveillance: colitis occurs in some HPS subtypes; "
            "Ophthalmology: standard OCA low vision management"
        ),
        "key_features": [
            "OCA — albinism (tyrosinase-positive; pale skin/hair/eyes; nystagmus; photophobia)",
            "PLATELET DENSE GRANULE DEFICIENCY — absent ADP/serotonin stores; prolonged bleeding time; NORMAL platelet count",
            "EASY BRUISING + PROLONGED BLEEDING — menorrhagia common; epistaxis; surgical risk high",
            "PULMONARY FIBROSIS (UIP pattern) — lethal 3rd-4th decade PATHOGNOMONIC for HPS1",
            "CEROID ACCUMULATION — lipofuscin-like material; bowel + kidney + lung accumulation",
            "Puerto Rican founder: p.Gln116fs — highest HPS1 prevalence worldwide (1:1800 NW Puerto Rico)",
        ],
        "monitoring": [
            "PFTs (FVC, DLCO, 6-min walk test) annually from age 20yr: monitor pulmonary fibrosis",
            "HRCT chest at diagnosis + 2-yearly: UIP pattern appearance and progression",
            "Platelet electron microscopy at diagnosis: confirm dense granule absence (diagnostic)",
            "Pre-operative bleeding risk: DDAVP protocol + haematology consult mandatory",
            "Annual dermatology: OCA-related SCC/melanoma",
            "Ophthalmology: standard OCA low vision management + photophobia tinted lenses",
        ],
        "key_ddx": [
            "TYR-OCA1/OCA2 (albinism but NO bleeding, NO pulmonary fibrosis — platelet EM normal)",
            "Chédiak-Higashi/LYST (albinism + immune deficiency + HLH; giant neutrophil granules on smear)",
            "Glanzmann thrombasthenia (platelet aggregation defect but NO albinism; different mechanism)",
            "Idiopathic pulmonary fibrosis (IPF — NO albinism; no platelet defect; older onset)",
        ],
        "melanoma_risk": True,
        "bleeding_risk": True,
        "hlh_risk": False,
        "stable_pigment": False,
        "ocular_emergency": False,
        "severity_options": [
            "HPS1 typical (albinism + bleeding tendency + pulmonary fibrosis onset 30-40yr)",
            "HPS1 severe (early pulmonary fibrosis + significant haemorrhagic events)",
        ],
        "systemic_options": ["Albinism (OCA)", "Easy bruising", "Prolonged bleeding", "Pulmonary fibrosis", "Nystagmus", "Photophobia", "Colitis"],
        "treatments_used": ["SPF50 sunscreen", "Pirfenidone", "DDAVP pre-operative", "Tinted glasses", "Annual PFTs/HRCT", "NO aspirin/NSAIDs"],
    },
    # -- LYST — Chédiak-Higashi Syndrome ---------------------------------------------------
    {
        "gene": "LYST",
        "alt_name": (
            "LYST (LYST-3801aa-1q42.3 / AR — Chédiak-Higashi-Syndrome-CHS — "
            "PARTIAL-ALBINISM+GIANT-PEROXIDASE-POSITIVE-GRANULES-NEUTROPHILS-PATHOGNOMONIC — "
            "ACCELERATED-PHASE-HLH-LETHAL-HSCT-Only-Curative — "
            "SILVER-GREY-HAIR-Recurrent-Staphylococcal-Streptococcal-Infections)"
        ),
        "protein": (
            "LYST -- 1q42.3 AR -- LYST-3801aa -- "
            "Lysosomal-Trafficking-Regulator-CHS1-430kDa-Largest-Intracellular-Trafficking-Protein -- "
            "LYST-BEACH-Domain-WD40-Repeats-Lysosome-Biogenesis-Vesicle-Trafficking -- "
            "GIANT-PEROXIDASE-POSITIVE-INTRACYTOPLASMIC-GRANULES-NEUTROPHILS-PATHOGNOMONIC -- "
            "Peripheral-Blood-Smear-Myeloperoxidase-Stain-Shows-Giant-Granules-DIAGNOSTIC -- "
            "SILVER-GREY-HAIR-Distinctive-Metallic-Sheen-Partial-OCA-Not-Complete-Albinism -- "
            "NK-CELL-CYTOTOXICITY-ABSENT-CTL-Cytotoxicity-Absent-Kills-Not-Happening -- "
            "ACCELERATED-Phase-HLH-Hemophagocytic-Lymphohistiocytosis-Fever-Splenomegaly-Fatal -- "
            "HSCT-CURATIVE-HLH-Infections-But-NOT-Neurological-Progression -- "
            "NEUROLOGICAL-PROGRESSION-Post-HSCT-Spinocerebellar-Neuropathy-Late -- "
            "OMIM-Gene-LYST-606897-Disease-CHS-214500"
        ),
        "locus": "1q42.3",
        "protein_size": "3801 aa / 430 kDa",
        "inheritance": (
            "AR; biallelic LYST mutations; consanguinity increases risk; "
            "no known founder mutations (pan-ethnic); "
            "genotype-phenotype correlation: truncating → classic childhood-onset CHS; "
            "missense hypomorphic → attenuated adult-onset CHS (partial NK cytotoxicity preservation)"
        ),
        "pigment_category": "Chédiak-Higashi Syndrome — Partial OCA + Giant Lysosomal Granules + NK/CTL Cytotoxicity Defect",
        "pathognomonic": (
            "PARTIAL ALBINISM (SILVER-GREY HAIR with metallic sheen; pale skin; blue-grey irides) + "
            "GIANT PEROXIDASE-POSITIVE INTRACYTOPLASMIC GRANULES IN NEUTROPHILS "
            "(peripheral blood smear myeloperoxidase stain = PATHOGNOMONIC, DEFINITIVE, IMMEDIATE) + "
            "RECURRENT PYOGENIC INFECTIONS (staph/strep — impaired neutrophil bactericidal function); "
            "ACCELERATED PHASE = HLH (haemophagocytic lymphohistiocytosis): "
            "fever + splenomegaly + cytopenias + hyperferritinaemia = LIFE-THREATENING EMERGENCY"
        ),
        "treatment": (
            "HAEMATOPOIETIC STEM CELL TRANSPLANT (HSCT) — ONLY CURATIVE TREATMENT; "
            "Corrects HLH risk + infections; does NOT prevent neurological progression post-HSCT; "
            "ACCELERATED PHASE: HLH protocol (dexamethasone + etoposide ± ciclosporin = HLH-2004) "
            "as BRIDGE TO HSCT — urgent haematology/oncology; "
            "Pre-HSCT: antibacterial + antifungal prophylaxis (TMP-SMX + azole); "
            "G-CSF: trials show limited benefit in CHS (giant granule formation persists); "
            "Vitamin C: antioxidant, historically used, limited evidence; "
            "NEUROLOGICAL MONITORING post-HSCT: spinocerebellar degeneration progresses despite HSCT; "
            "SPF50: partial OCA skin UV protection; "
            "Ophthalmology: partial OCA nystagmus + photophobia management"
        ),
        "key_features": [
            "SILVER-GREY HAIR — metallic sheen, partial albinism (NOT complete white); distinctive appearance",
            "GIANT PEROXIDASE-POSITIVE GRANULES in neutrophils (peripheral smear, MPO stain) — PATHOGNOMONIC",
            "RECURRENT PYOGENIC INFECTIONS — Staphylococcus/Streptococcus; deep abscesses; impaired killing",
            "NK CELL CYTOTOXICITY ABSENT — unable to kill infected cells; basis for HLH vulnerability",
            "ACCELERATED PHASE = HLH — fever + splenomegaly + pancytopenia + hyperferritinaemia — LETHAL WITHOUT HSCT",
            "HSCT corrects immune defects but NOT late neurological progression (spinocerebellar atrophy)",
        ],
        "monitoring": [
            "Peripheral blood smear (MPO stain) at diagnosis: giant granules confirm CHS",
            "NK cell cytotoxicity assay: absent function (perforin release defect) — diagnostic",
            "Ferritin + triglycerides + fibrinogen + CBC: monthly in young children (HLH surveillance)",
            "Bone marrow aspirate if suspected accelerated phase: haemophagocytosis",
            "Neurological assessment annually (post-HSCT): spinocerebellar signs emerge late",
            "Ophthalmology: partial OCA ocular features (nystagmus/photophobia management)",
        ],
        "key_ddx": [
            "HPS1/HPS (albinism + platelet defect + pulmonary fibrosis; NO giant neutrophil granules; NO HLH)",
            "Griscelli syndrome (GS1/MYO5A, GS2/RAB27A — silver hair + immune defect; NO giant granules; RAB27A → HLH)",
            "Familial HLH (PRF1/UNC13D/STX11 — HLH without albinism; perforin/MUNC pathway mutations)",
            "Neutropenia other causes (SCN/cyclic — no giant granules; no albinism; different gene)",
        ],
        "melanoma_risk": False,
        "bleeding_risk": True,
        "hlh_risk": True,
        "stable_pigment": False,
        "ocular_emergency": False,
        "severity_options": [
            "Classic CHS (childhood onset, severe infections, accelerated phase HLH risk — truncating mutations)",
            "Attenuated CHS (adult onset, milder infections, reduced accelerated phase risk — hypomorphic mutations)",
        ],
        "systemic_options": ["Silver-grey hair", "Partial albinism", "Recurrent infections", "Giant granules", "HLH risk", "Nystagmus", "Bleeding tendency"],
        "treatments_used": ["HSCT (curative)", "HLH-2004 protocol", "TMP-SMX prophylaxis", "Azole prophylaxis", "SPF50 sunscreen", "Ophthalmology"],
    },
    # -- KIT — Piebaldism ------------------------------------------------------------------
    {
        "gene": "KIT",
        "alt_name": (
            "KIT (KIT-976aa-4q12 / AD — Piebaldism — "
            "WHITE-FORELOCK+STABLE-DEPIGMENTED-PATCHES-PATHOGNOMONIC — "
            "PRESENT-FROM-BIRTH-STABLE-NON-PROGRESSIVE-KEY-DDx-From-Vitiligo-Acquired-Progressive — "
            "No-Systemic-Features-Isolated-Pigmentation-Defect)"
        ),
        "protein": (
            "KIT -- 4q12 AD -- KIT-976aa -- "
            "KIT-Proto-Oncogene-Receptor-Tyrosine-Kinase-145kDa-Type-III-RTK-SCF-Receptor -- "
            "Stem-Cell-Factor-SCF-Receptor-Melanocyte-Migration-Mast-Cell-Haematopoiesis -- "
            "KIT-Loss-of-Function-AD-Piebaldism-Melanocyte-Migration-Failure-Embryogenesis -- "
            "WHITE-FORELOCK-100pct-Penetrance-Frontal-Scalp-White-Patch-PATHOGNOMONIC -- "
            "TRIANGULAR-DEPIGMENTED-ABDOMINAL-PATCH-Ventral-Chest-Arms-Legs-Classic -- "
            "STABLE-FROM-BIRTH-NON-PROGRESSIVE-KEY-DDx-Vitiligo-Acquired-Progresses -- "
            "ISLANDS-OF-NORMAL-PIGMENTED-SKIN-Within-White-Patches-PATHOGNOMONIC -- "
            "KIT-p.Asp816Val-Gain-of-Function-Mastocytosis-Not-Piebaldism-Opposite-Direction -- "
            "NO-Hirschsprung-Waardenburg-Features-Isolated-Piebaldism -- "
            "OMIM-Gene-KIT-164920-Disease-Piebaldism-172800"
        ),
        "locus": "4q12",
        "protein_size": "976 aa / 145 kDa",
        "inheritance": (
            "AD; heterozygous KIT loss-of-function mutations; "
            "high penetrance (>95%); complete family history — autosomal dominant, vertical transmission; "
            "de novo mutations ~15%; genotype-phenotype: larger deletions/certain missense → more extensive depigmentation; "
            "KIT gain-of-function (p.Asp816Val) → systemic mastocytosis — different disease, opposite effect"
        ),
        "pigment_category": "Piebaldism — Stable Congenital Depigmented Patches (KIT, melanocyte migration failure)",
        "pathognomonic": (
            "WHITE FORELOCK (frontal scalp triangular depigmented patch) + "
            "STABLE DEPIGMENTED LEUKODERMIC PATCHES (ventral chest, abdomen, arms/legs — "
            "with islands of normal/hyperpigmented skin within white patches) PATHOGNOMONIC; "
            "PRESENT FROM BIRTH AND STABLE (non-progressive) = KEY DDx from vitiligo "
            "(vitiligo is acquired after birth and progressive); "
            "NO SYSTEMIC FEATURES — isolated pigmentation defect; no hearing, cardiac, or bowel abnormalities"
        ),
        "treatment": (
            "NO curative systemic treatment required — isolated cosmetic condition; "
            "SPF50 on depigmented patches — CRITICAL: no melanin, UV burn risk minimal but present; "
            "Cosmetic options: self-tanning lotions on white patches; skin-coloured camouflage cosmetics; "
            "Repigmentation procedures (limited evidence): "
            "split-thickness skin grafting, melanocyte transplantation, suction blister grafting; "
            "Phototherapy (NB-UVB) — limited efficacy in true piebaldism (melanocytes absent); "
            "Excimer laser: may help border zones with residual melanocytes; "
            "Camouflage + psychosocial support — most patients choose conservative management; "
            "Genetic counselling: AD (50% risk to offspring); cosmetic only condition"
        ),
        "key_features": [
            "WHITE FORELOCK — triangular depigmented patch on frontal scalp, present from birth, 100% penetrance",
            "STABLE CONGENITAL LEUKODERMA — depigmented patches present at birth, DO NOT PROGRESS",
            "ISLANDS OF PIGMENTED SKIN within white patches — characteristic heterogeneous pattern",
            "KEY DDx VITILIGO: piebaldism = CONGENITAL STABLE; vitiligo = ACQUIRED PROGRESSIVE",
            "NO systemic features — NO hearing loss, NO cardiac, NO Hirschsprung disease (DDx Waardenburg)",
            "KIT loss-of-function = melanocyte migration failure from neural crest during embryogenesis",
        ],
        "monitoring": [
            "SPF50 application to depigmented patches: no melanin → UV sensitivity (sunburn, SCC risk low)",
            "Annual dermatology: SCC/BCC surveillance in chronically sun-exposed depigmented areas",
            "Ophthalmology: NOT required (no ocular features in piebaldism, unlike Waardenburg)",
            "Audiogram: NOT required (no SNHL in piebaldism — unlike Waardenburg syndrome)",
            "Psychosocial assessment: appearance impact, camouflage support",
            "Genetic counselling: 50% offspring risk; cosmetic-only condition — reassurance important",
        ],
        "key_ddx": [
            "Vitiligo (ACQUIRED progressive depigmentation — autoimmune; starts AFTER birth; NOT present from birth)",
            "Waardenburg syndrome (PAX3/MITF/EDNRB/EDN3 — white forelock + SENSORINEURAL HEARING LOSS + heterochromia)",
            "Tuberous sclerosis (TSC1/TSC2 — ash-leaf macules present birth but oval not forelock; other systemic features)",
            "Nevus depigmentosus (congenital stable but solitary, no forelock; DIFFERENT distribution; NOT KIT)",
        ],
        "melanoma_risk": False,
        "bleeding_risk": False,
        "hlh_risk": False,
        "stable_pigment": True,
        "ocular_emergency": False,
        "severity_options": [
            "Limited (white forelock only, small ventral patches)",
            "Extensive (large bilateral depigmented patches, widespread leukoderma)",
        ],
        "systemic_options": ["White forelock", "Stable leukoderma", "Depigmented patches (ventral)", "Hyperpigmented islands within white patches"],
        "treatments_used": ["SPF50 sunscreen", "Cosmetic camouflage", "Skin grafting (selected)", "Melanocyte transplantation", "Psychosocial support"],
    },
    # -- MC1R — Red Hair / Melanoma Predisposition -----------------------------------------
    {
        "gene": "MC1R",
        "alt_name": (
            "MC1R (MC1R-317aa-16q24.3 / AD-Incomplete-Penetrance — Red-Hair-Color-RHC-Melanoma-Predisposition — "
            "RED-HAIR+FAIR-SKIN+EPHELIDES-FRECKLES-PATHOGNOMONIC — "
            "R-Variants-4-10x-Melanoma-Risk-Compound-Heterozygotes-Highest — "
            "SPF50-Mandatory-Annual-Dermoscopy)"
        ),
        "protein": (
            "MC1R -- 16q24.3 AD-Incomplete -- MC1R-317aa -- "
            "Melanocortin-1-Receptor-35kDa-7-Pass-GPCR-Gs-Coupled-Melanocyte-cAMP-Pigment-Switch -- "
            "Alpha-MSH→MC1R→cAMP→MITF→TYR-TYRP1-TYRP2→Eumelanin-Brown-Black -- "
            "MC1R-LOF-Phaeomelanin-Predominance-Red-Yellow-Pigment-Not-Eumelanin -- "
            "RED-HAIR-Orange-Red-Copper-Strawberry-Blonde-PATHOGNOMONIC -- "
            "FAIR-FRECKLED-SKIN-EPHELIDES-Sun-Reactive-Skin-Type-I-II -- "
            "R-VARIANTS-Arg151Cys-Arg160Trp-Asp294His-HIGHEST-MELANOMA-RISK-4-10x -- "
            "r-VARIANTS-Val60Leu-Val92Met-Asp294His-Moderate-Risk -- "
            "COMPOUND-HETEROZYGOTES-R/R-Highest-Risk-Even-Without-Red-Hair-Phenotype -- "
            "UV-SIGNATURE-MUTATIONS-Higher-in-MC1R-Melanomas -- "
            "OMIM-Gene-MC1R-155555-Disease-RHC-266300"
        ),
        "locus": "16q24.3",
        "protein_size": "317 aa / 35 kDa",
        "inheritance": (
            "AD with incomplete penetrance; codominant/additive allelic effects; "
            "R variants: Arg151Cys, Arg160Trp, Asp294His = highest penetrance red hair + melanoma risk; "
            "r variants: Val60Leu, Val92Met = lower penetrance; "
            "compound heterozygotes (R/R or R/r): additive risk even without overt red hair; "
            "MC1R variants common: ~15% Europeans carry at least one R variant"
        ),
        "pigment_category": "MC1R — Red Hair Color / Melanoma Predisposition (GPCR eumelanin/phaeomelanin switch)",
        "pathognomonic": (
            "RED/AUBURN/STRAWBERRY-BLONDE HAIR + FAIR FRECKLED SKIN (skin phototype I-II) + "
            "EPHELIDES (freckles on sun-exposed sites) PATHOGNOMONIC; "
            "MELANOMA RISK: R/R compound heterozygotes = 4-10x elevated lifetime melanoma risk; "
            "UV-SIGNATURE MUTATIONS more frequent in MC1R-associated melanoma; "
            "PARADOX: compound heterozygotes can have brown hair but still carry full melanoma risk"
        ),
        "treatment": (
            "SPF50 BROAD-SPECTRUM SUNSCREEN MANDATORY LIFELONG from childhood; "
            "ANNUAL TOTAL-BODY DERMOSCOPY — melanoma surveillance from age 18yr (earlier if family history); "
            "Avoid tanning beds ABSOLUTELY — UV-sensitised skin; "
            "Hat + UPF50+ clothing + shade during peak UV hours (10am-4pm); "
            "Vitamin D supplementation: high SPF use + indoor occupation → deficiency; "
            "Dermoscopy by dermatologist: any changing/new pigmented lesion → urgent review; "
            "Wide local excision for melanoma; sentinel lymph node biopsy per AJCC staging; "
            "Genetic counselling: 50% offspring risk per R variant allele; "
            "Shade + sun-safe behaviours from infancy — ephelide (freckle) count predicts UV exposure history"
        ),
        "key_features": [
            "RED/AUBURN/STRAWBERRY-BLONDE HAIR — MC1R LOF → phaeomelanin (red-yellow) predominance over eumelanin",
            "FAIR FRECKLED SKIN — skin phototype I (always burns, never tans) or II (usually burns, rarely tans)",
            "EPHELIDES (freckles) — UV-induced, concentrated on sun-exposed sites (face, shoulders, arms)",
            "MELANOMA RISK — R/R compound heterozygotes: 4-10x lifetime risk; independent of hair colour",
            "UV-SIGNATURE MUTATIONS — C>T transitions at dipyrimidines enriched in MC1R melanomas",
            "COMPOUND HETEROZYGOTES can have BROWN hair but still carry FULL MELANOMA RISK (paradox)",
        ],
        "monitoring": [
            "Annual total-body dermoscopy from age 18yr (earlier if family history melanoma)",
            "Monthly self-examination: ABCDE criteria for any new/changing lesion",
            "Urgent dermatology: any pigmented lesion with new asymmetry/bleeding/satellite",
            "Vitamin D levels annually (SPF + indoor lifestyle → insufficiency common)",
            "Ophthalmology: uveal melanoma risk slightly elevated — no routine surveillance unless other risk factors",
            "Family history: cascade MC1R genotyping for first-degree relatives in high-risk families",
        ],
        "key_ddx": [
            "OCA1B (yellow-blonde hair; albinism + nystagmus + photophobia; TYR mutation; MC1R does NOT cause OCA)",
            "Familial melanoma (CDKN2A/CDK4/BAP1 — melanoma + dysplastic naevi; MC1R modifier not causative alone)",
            "Xeroderma Pigmentosum (XP — extreme UV sensitivity + melanoma + neurodegeneration; different mechanism)",
            "Ephelides (freckles) differential: xeroderma pigmentosum, LEOPARD/Peutz-Jeghers — different distribution/cause",
        ],
        "melanoma_risk": True,
        "bleeding_risk": False,
        "hlh_risk": False,
        "stable_pigment": True,
        "ocular_emergency": False,
        "severity_options": [
            "r/r homozygous (mild red hair tint, moderate melanoma risk)",
            "R/r or R/R (full red hair, high-very high melanoma risk)",
        ],
        "systemic_options": ["Red hair", "Fair freckled skin", "Ephelides", "Sun sensitivity", "Melanoma risk", "Vitamin D deficiency"],
        "treatments_used": ["SPF50 sunscreen", "Annual dermoscopy", "UPF clothing", "Vitamin D supplements", "Shade/sun-safe behaviour", "Wide excision (melanoma)"],
    },
]


def _build_cohort() -> list:
    cohort = []
    for i, entry in enumerate(PIGMENT_GENES):
        seed = SEED_BASE + i
        rng = random.Random(seed)
        for j in range(40):
            gene = entry["gene"]
            severity = rng.choice(entry["severity_options"])
            systemic = rng.sample(entry["systemic_options"], k=min(3, len(entry["systemic_options"])))
            treatment = rng.choice(entry["treatments_used"])
            age_dx = rng.uniform(0.1, 35.0) if gene in ("TYR", "OCA2", "TYRP1", "SLC45A2", "HPS1", "LYST", "KIT") else rng.uniform(15.0, 55.0)
            fu_yrs = rng.uniform(0.5, 20.0)
            melanoma_event = rng.random() < (0.12 if entry["melanoma_risk"] else 0.01)
            bleeding_event = rng.random() < (0.35 if entry["bleeding_risk"] else 0.0)
            hlh_event = rng.random() < (0.45 if entry["hlh_risk"] else 0.0)
            cohort.append({
                "patient_id": f"{gene}-{seed:04d}-{j+1:02d}",
                "gene": gene,
                "severity": severity,
                "systemic_features": systemic,
                "treatment": treatment,
                "age_at_dx_yrs": round(age_dx, 1),
                "follow_up_yrs": round(fu_yrs, 1),
                "melanoma_event": melanoma_event,
                "bleeding_event": bleeding_event,
                "hlh_event": hlh_event,
                "seed": seed,
            })
    return cohort


def generate_overview() -> dict:
    cohort = _build_cohort()
    gene_counts = {}
    type_counts = {}
    melanoma_genes = []
    bleeding_genes = []
    hlh_genes = []
    gene_summary = {}

    for entry in PIGMENT_GENES:
        g = entry["gene"]
        pts = [p for p in cohort if p["gene"] == g]
        if entry["melanoma_risk"]:
            melanoma_genes.append(g)
        if entry["bleeding_risk"]:
            bleeding_genes.append(g)
        if entry["hlh_risk"]:
            hlh_genes.append(g)
        cat = entry["pigment_category"].split("—")[0].strip()
        type_counts[cat] = type_counts.get(cat, 0) + len(pts)
        gene_counts[g] = len(pts)
        gene_summary[g] = {
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "pigment_category": entry["pigment_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "melanoma_risk": entry["melanoma_risk"],
            "bleeding_risk": entry["bleeding_risk"],
            "hlh_risk": entry["hlh_risk"],
            "stable_pigment": entry["stable_pigment"],
        }

    return {
        "title": "Hereditary-Pigmentation-Disorder-Atlas — Complete 8-Gene Hereditary Pigmentation Disorder Atlas",
        "n_genes": len(PIGMENT_GENES),
        "n_patients": len(cohort),
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "pigment_categories": {
            "OCA — Tyrosinase-pathway (TYR)": "TYR (OCA1A zero-activity / OCA1B residual; ANHIDROSIS DDx — sweating NORMAL; UV burn)",
            "OCA — OCA2/P-protein (most common worldwide)": "OCA2 (most common OCA; yellow-blonde dark races; 15q11-q13 overlap PWS/AS)",
            "OCA — Rufous/Bronze (TYRP1)": "TYRP1 (OCA3; rufous-red hair + bronze skin + brown eyes in Africans; milder ocular; underdiagnosed)",
            "OCA — MATP/Japan (SLC45A2)": "SLC45A2 (OCA4; most common in Japan; variable pale-cream; nystagmus; full OCA panel needed)",
            "Syndromic OCA — Hermansky-Pudlak (HPS1)": "HPS1 (OCA + platelet dense granule deficiency + pulmonary fibrosis LETHAL; NO aspirin; Puerto Rico founder)",
            "Syndromic OCA — Chédiak-Higashi (LYST)": "LYST (silver-grey hair + giant neutrophil granules + HLH accelerated phase; HSCT curative)",
            "Stable leukoderma — Piebaldism (KIT)": "KIT (white forelock + stable depigmented patches; CONGENITAL STABLE = KEY DDx vitiligo acquired/progressive)",
            "Melanoma predisposition — Red hair (MC1R)": "MC1R (red hair + fair skin + ephelides; R/R 4-10x melanoma risk; SPF50 + annual dermoscopy)",
        },
        "inheritance_map": {
            "TYR": "AR", "OCA2": "AR", "TYRP1": "AR", "SLC45A2": "AR",
            "HPS1": "AR", "LYST": "AR", "KIT": "AD", "MC1R": "AD-incomplete",
        },
        "key_clinical_pearls": [
            "TYR-OCA1: COMPLETE ABSENCE MELANIN SKIN/HAIR/EYES (OCA1A = zero tyrosinase — white hair lifelong; OCA1B = residual — yellow-blonde accrues yr 1-2); NYSTAGMUS+PHOTOPHOBIA+FOVEAL HYPOPLASIA TRIAD = ALL OCA types; VEP chiasmal misrouting PATHOGNOMONIC; SPF50 mandatory; annual dermatology (melanoma/SCC)",
            "OCA2: MOST COMMON OCA WORLDWIDE (1:40,000 Europe; 1:3,900 sub-Saharan Africa); YELLOW-BLONDE HAIR + PALE CREAM SKIN in dark-skinned populations PATHOGNOMONIC; 15q11-q13 deletion → concurrent Prader-Willi or Angelman + OCA2; p.Arg305Trp most common European allele; SPF provision = public health priority equatorial Africa",
            "TYRP1-OCA3: RUFOUS ALBINISM — RED-RUFOUS HAIR + BRONZE SKIN + BROWN EYES in Africans PATHOGNOMONIC; PIGMENT REDUCED NOT ABSENT = KEY DDx OCA1A; MILDER OCULAR — nystagmus absent/mild; UNDERDIAGNOSED worldwide; molecular panel required; equatorial Africa SCC risk still elevated",
            "SLC45A2-OCA4: MOST COMMON OCA JAPAN (>70% Japanese OCA patients); VARIABLE EXPRESSIVITY (near-normal to OCA1B-like); OCA4 missed by TYR/OCA2-only panels — FULL OCA PANEL MANDATORY; MATP/melanosomal pH regulation; nystagmus + photophobia universal in OCA4",
            "HPS1-Hermansky-Pudlak: OCA + PLATELET DENSE GRANULE DEFICIENCY (absent ADP/serotonin release — NORMAL platelet COUNT but prolonged bleeding time) + PULMONARY FIBROSIS (UIP pattern, lethal 3rd-4th decade) TRIAD PATHOGNOMONIC; PUERTO RICO FOUNDER (1:1800 NW Puerto Rico); NO ASPIRIN/NSAIDs ABSOLUTE CONTRAINDICATION; PIRFENIDONE for pulmonary fibrosis; DDAVP pre-operatively; platelet EM confirms granule absence",
            "LYST-Chédiak-Higashi: PARTIAL ALBINISM (SILVER-GREY HAIR metallic sheen) + GIANT PEROXIDASE-POSITIVE GRANULES IN NEUTROPHILS (peripheral smear MPO stain = PATHOGNOMONIC, IMMEDIATE DIAGNOSTIC) + NK cytotoxicity absent; ACCELERATED PHASE = HLH (fever+splenomegaly+cytopenias+hyperferritinaemia) = LETHAL WITHOUT HSCT; HLH-2004 protocol bridge to HSCT; HSCT curative for HLH/infections but NOT neurological progression",
            "KIT-Piebaldism: WHITE FORELOCK (triangular frontal scalp patch) + STABLE DEPIGMENTED PATCHES (ventral/abdominal/extremity; islands of pigmented skin within white patches) PATHOGNOMONIC; CONGENITAL STABLE NON-PROGRESSIVE = KEY DDx from VITILIGO (acquired/progressive); NO systemic features (no hearing loss distinguishes from Waardenburg); conservative management + SPF; grafting/melanocyte transplant if desired",
            "MC1R-Red Hair: RED/AUBURN/STRAWBERRY-BLONDE HAIR + FAIR FRECKLED SKIN (phototype I-II) + EPHELIDES PATHOGNOMONIC; R-VARIANTS (Arg151Cys/Arg160Trp/Asp294His) = 4-10x melanoma risk; COMPOUND HETEROZYGOTES (R/R) can have BROWN hair yet carry FULL MELANOMA RISK; UV-SIGNATURE MUTATIONS enriched in MC1R melanomas; SPF50 LIFELONG MANDATORY + ANNUAL TOTAL-BODY DERMOSCOPY from age 18yr",
        ],
        "gene_summary": gene_summary,
        "melanoma_risk_genes": melanoma_genes,
        "bleeding_risk_genes": bleeding_genes,
        "hlh_risk_genes": hlh_genes,
        "stable_pigment_genes": [e["gene"] for e in PIGMENT_GENES if e["stable_pigment"]],
        "diagnostic_algorithm": {
            "Step_1": "Classify pigmentation type: (A) COMPLETE ABSENCE (white hair/pink skin/pink eyes from birth) → TYR-OCA1A; (B) PARTIAL ABSENCE (pale cream/variable) → OCA1B/OCA2/TYRP1/SLC45A2; (C) SILVER-GREY metallic hair + recurrent infections → LYST-CHS; (D) STABLE congenital white patch → KIT-piebaldism; (E) RED hair + freckles + melanoma risk → MC1R",
            "Step_2": "Ocular assessment: nystagmus present = all OCA subtypes (TYR/OCA2/TYRP1/SLC45A2) + HPS; nystagmus absent/mild = TYRP1 (milder), KIT (none), MC1R (none)",
            "Step_3": "Systemic screen: BLEEDING tendency + albinism → HPS1 (platelet EM + PFTs); RECURRENT INFECTIONS + silver hair + giant granules on smear → LYST-CHS (MPO stain IMMEDIATE); PULMONARY FIBROSIS + albinism → HPS1; WHITE FORELOCK stable + NO deafness → KIT piebaldism",
            "Step_4": "Ethnicity/geography clue: Sub-Saharan Africa yellow-blonde → OCA2 (most common); Japan pale-cream → SLC45A2 (OCA4 >70%); rufous-red hair bronze skin Africa → TYRP1-OCA3; NW Puerto Rico + albinism + bleeding → HPS1 founder",
            "Step_5": "Melanoma risk stratification: MC1R R-variants → annual dermoscopy; all OCA subtypes → annual dermatology; piebaldism depigmented patches → SPF50 critical; CHS/HPS → SPF50 + dermatology",
            "Step_6": "Molecular panel: full OCA NGS (TYR/OCA2/TYRP1/SLC45A2/HPS1-3,5-7/LYST/MC1R/KIT) + platelet EM + VEP for complete classification",
        },
        "type_distribution": dict(sorted(type_counts.items(), key=lambda x: -x[1])[:10]),
    }


def generate_breakdown() -> dict:
    cohort = _build_cohort()
    by_gene = {}
    for p in cohort:
        by_gene.setdefault(p["gene"], []).append(p)

    gene_breakdown = {}
    for entry in PIGMENT_GENES:
        g = entry["gene"]
        pts = by_gene.get(g, [])

        severity_dist = {}
        treatment_dist = {}
        systemic_dist = {}
        for p in pts:
            sv = p["severity"]
            severity_dist[sv] = severity_dist.get(sv, 0) + 1
            tx = p["treatment"]
            treatment_dist[tx] = treatment_dist.get(tx, 0) + 1
            for sf in p["systemic_features"]:
                systemic_dist[sf] = systemic_dist.get(sf, 0) + 1

        avg_age = round(sum(p["age_at_dx_yrs"] for p in pts) / len(pts), 2) if pts else 0
        avg_fu = round(sum(p["follow_up_yrs"] for p in pts) / len(pts), 1) if pts else 0
        melanoma_n = sum(1 for p in pts if p["melanoma_event"])
        bleeding_n = sum(1 for p in pts if p["bleeding_event"])
        hlh_n = sum(1 for p in pts if p["hlh_event"])

        gene_breakdown[g] = {
            "gene": g,
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "pigment_category": entry["pigment_category"],
            "n_patients": len(pts),
            "avg_age_at_dx_yrs": avg_age,
            "avg_follow_up_yrs": avg_fu,
            "melanoma_event_pct": round(melanoma_n / len(pts) * 100, 1) if pts else 0,
            "bleeding_event_pct": round(bleeding_n / len(pts) * 100, 1) if pts else 0,
            "hlh_event_pct": round(hlh_n / len(pts) * 100, 1) if pts else 0,
            "pathognomonic": entry["pathognomonic"],
            "treatment_highlight": entry["treatment"][:600],
            "key_features": entry["key_features"],
            "treatment_summary": entry["treatment"][:700],
            "monitoring": entry["monitoring"],
            "key_ddx": entry["key_ddx"],
            "melanoma_risk": entry["melanoma_risk"],
            "bleeding_risk": entry["bleeding_risk"],
            "hlh_risk": entry["hlh_risk"],
            "stable_pigment": entry["stable_pigment"],
            "severity_distribution": dict(sorted(severity_dist.items(), key=lambda x: -x[1])),
            "systemic_distribution": dict(sorted(systemic_dist.items(), key=lambda x: -x[1])),
            "treatment_distribution": dict(sorted(treatment_dist.items(), key=lambda x: -x[1])),
            "patients_sample": pts[:5],
        }

    return {
        "title": "Hereditary-Pigmentation-Disorder-Atlas — Per-Gene Breakdown",
        "n_genes": 8,
        "n_patients": len(cohort),
        "gene_breakdown": gene_breakdown,
        "clinical_emergency_flags": [
            "HPS1 NO-ASPIRIN/NSAIDs ABSOLUTE: Platelet dense granule deficiency → ANY aspirin/NSAID = major haemorrhage risk; alternative analgesia (paracetamol) mandatory; pre-operative haematology review + DDAVP (0.3 mcg/kg IV) before ALL procedures; bleeding time prolonged; platelet count NORMAL — do not be falsely reassured by normal CBC",
            "HPS1 PULMONARY FIBROSIS EMERGENCY: Annual PFTs (FVC/DLCO) + HRCT chest from age 20yr; declining DLCO is earliest sign; pirfenidone/nintedanib started at first sign of ILD; lung transplant referral for end-stage; do NOT withhold anti-fibrotic due to albinism diagnosis alone — HPS1 pulmonary fibrosis is the lethal complication",
            "LYST-CHS ACCELERATED PHASE = HLH EMERGENCY: Fever + splenomegaly + pancytopenia + hyperferritinaemia in CHS = ACCELERATED PHASE (HLH) — LIFE-THREATENING; URGENT haematology; HLH-2004 protocol (dexamethasone + etoposide ± ciclosporin) as BRIDGE TO HSCT; HSCT is ONLY CURATIVE THERAPY; delay = death; any CHS patient with fever unresponsive to antibiotics = URGENT HLH workup",
            "TYR/OCA HEAT STROKE PARADOX: OCA patients in tropical/equatorial countries: UV protection mandatory but SPF prevents vitamin D synthesis → vitamin D monitoring + supplementation; OCA skin burns within MINUTES in direct sun without SPF (no UV barrier from melanin); protective hat + UPF clothing + shade from infancy",
            "MC1R MELANOMA SURVEILLANCE: R/R compound heterozygotes carry 4-10x melanoma risk INDEPENDENT of hair colour — brown-haired individuals with R/R genotype require same surveillance as red-haired patients; annual total-body dermoscopy from age 18yr (earlier if family history); tanning bed absolute contraindication; any rapidly changing lesion = URGENT dermoscopy/biopsy",
        ],
    }


def generate_definitions() -> dict:
    return {
        "title": "Hereditary-Pigmentation-Disorder-Atlas — Definitions & Glossary",
        "gene_entries": {
            entry["gene"]: {
                "full_name": entry["gene"],
                "protein_size": entry["protein_size"],
                "locus": entry["locus"],
                "inheritance": entry["inheritance"],
                "disease_name": entry["pigment_category"],
                "pathognomonic": entry["pathognomonic"],
                "key_features": entry["key_features"],
                "treatment": entry["treatment"],
                "key_ddx": entry["key_ddx"],
                "monitoring": entry["monitoring"],
            }
            for entry in PIGMENT_GENES
        },
        "pigment_biology_glossary": {
            "Melanin synthesis pathway": "Melanin is synthesised in melanosomes within melanocytes from tyrosine: L-Tyrosine → (TYR) → L-DOPA → (TYR) → Dopaquinone → (TYR/TYRP2/TYRP1) → Eumelanin (brown-black) or → Phaeomelanin (red-yellow, via cysteine conjugation). OCA: disruption at TYR, OCA2, TYRP1, SLC45A2 → reduced/absent eumelanin. MC1R LOF → reduced α-MSH signalling → phaeomelanin predominance → red hair",
            "Oculocutaneous Albinism (OCA)": "Group of AR disorders characterised by reduced/absent melanin in skin, hair, and eyes. All OCA subtypes share: (1) nystagmus (pendular/jerk, bilateral, present birth); (2) photophobia (iris transillumination, no melanin light absorption); (3) foveal hypoplasia (reduced acuity 20/100-20/400); (4) optic tract misrouting (chiasmal decussation defect, VEP asymmetry — DIAGNOSTIC). Skin/hair pigmentation varies by type: OCA1A (none) → OCA1B, OCA2, OCA4 (variable pale) → OCA3 TYRP1 (rufous-bronze, milder eyes)",
            "VEP chiasmal misrouting (OCA diagnostic signature)": "In OCA, optic fibres from temporal retina (normally ipsilateral) are aberrantly crossed at the optic chiasm → both hemispheres respond primarily to contralateral eye stimulation; VEP (visual evoked potential): monocular stimulation shows hemispheric asymmetry/amplitude difference — PATHOGNOMONIC for OCA; used to distinguish OCA from other causes of nystagmus (CSNB, spasmus nutans, INS) where VEP pattern is symmetric. VEP misrouting detectable from birth",
            "Platelet dense granules (delta granules)": "Dense granules (δ-granules) in platelets contain: ADP, ATP, serotonin, Ca2+, pyrophosphate. On platelet activation, dense granules release ADP → ADP-receptor (P2Y12) activation → secondary wave platelet aggregation amplification. HPS: HPS1/2/5/6/7/9 → BLOC-2/3 deficiency → impaired dense granule biogenesis → ABSENT dense granules on electron microscopy (EM) → impaired secondary aggregation → prolonged bleeding time + haemorrhage despite normal platelet COUNT. EM of platelets = DEFINITIVE diagnostic test for HPS",
            "Hemophagocytic Lymphohistiocytosis (HLH)": "Life-threatening hyperinflammatory syndrome: uncontrolled activation of cytotoxic T-cells and macrophages with impaired NK/CTL cytotoxicity → cytokine storm (IFN-γ/TNF-α/IL-6). In CHS (LYST): NK cell cytotoxicity absent → inability to terminate immune responses → accumulation of activated cytotoxic cells. Diagnostic criteria (HLH-2004): fever + splenomegaly + cytopenias (≥2 lineages) + hyperferritinaemia (>500 µg/L) + hypofibrinogenaemia + haemophagocytosis on BM biopsy + elevated sCD25 + absent NK activity. Treatment: dexamethasone + etoposide (HLH-2004) → bridge to HSCT",
            "BLOC complexes (Biogenesis of Lysosome-Related Organelles)": "Multi-protein complexes required for biogenesis of lysosome-related organelles (melanosomes, platelet dense granules, lamellar bodies). BLOC-1 (HPS7/8/9), BLOC-2 (HPS3/5/6), BLOC-3 (HPS1/4), AP-3 (HPS2). HPS1+HPS4 form BLOC-3 → required for melanosome and dense granule formation. LYST disrupts global lysosomal trafficking (different mechanism: BEACH domain = lysosome-organelle budding). Disruption of any BLOC → HPS spectrum with varying severity",
            "Melanocortin 1 Receptor (MC1R) pigment switch": "MC1R is a Gs-coupled GPCR on melanocyte surface. α-MSH (POMC-derived) binds MC1R → cAMP ↑ → PKA → MITF transcription factor → upregulates TYR/TYRP1/TYRP2 → eumelanin synthesis (brown-black). ASIP (agouti signalling protein) antagonises MC1R → phaeomelanin (red-yellow). MC1R LOF variants: R-variants (Arg151Cys, Arg160Trp, Asp294His) = loss of α-MSH response → constitutive phaeomelanin → red hair + fair skin + increased UV sensitivity + melanoma risk",
            "Piebaldism vs Vitiligo": "PIEBALDISM (KIT): congenital stable leukoderma; white forelock present at birth; KIT LOF → melanocyte migration failure in embryogenesis; melanocytes physically ABSENT from white patches; non-progressive; no autoimmunity. VITILIGO: acquired autoimmune destruction of melanocytes post-birth; depigmentation starts after birth; progressive; Koebner phenomenon; associated thyroid/other autoimmune diseases; Smyth locus (HLA); responds to immunosuppression/phototherapy. DISTINCTION: birth vs acquired; stable vs progressive",
        },
        "treatment_glossary": {
            "SPF50 sunscreen (OCA — all subtypes)": "Broad-spectrum SPF50+ mineral sunscreen (zinc oxide/titanium dioxide — preferred for OCA infants/children) applied to all sun-exposed skin daily from diagnosis; reapply every 2hr during outdoor activity; in equatorial Africa where OCA2/TYRP1 prevalence is high, SPF provision is a public-health intervention; skin cancer (SCC most common) = significant cause of morbidity/mortality in African OCA patients without access to SPF; SPF50 alone insufficient without hat + UPF clothing + shade",
            "Pirfenidone (HPS1 pulmonary fibrosis)": "Anti-fibrotic pyridinone; reduces TGF-β-mediated fibroblast proliferation and collagen deposition; slows decline in FVC in HPS1-associated ILD; started at first sign of pulmonary fibrosis (FVC <80% or DLCO <70%); doses: 267mg TID escalating to 801mg TID; nausea + photosensitivity common; monitor LFTs; approved for IPF (used off-label in HPS1 ILD with strong evidence); nintedanib alternative (PDGFR/VEGFR/FGFR inhibitor); lung transplant for end-stage",
            "HLH-2004 Protocol (Chédiak-Higashi accelerated phase)": "Dexamethasone 10 mg/m² daily × 2wk → taper × 6wk + etoposide 150 mg/m² IV × 1-2 doses (weekly) + ciclosporin A (optional) → bridge to HSCT; intrathecal methotrexate if CNS involvement; HSCT as soon as donor identified; delay in starting HLH-2004 → fatal outcome; etoposide = crucial component; supportive: G-CSF + antibacterial + antifungal + IVIG; HSCT corrects HLH/infections but NOT neurological progression (continues post-HSCT)",
            "DDAVP (HPS1 pre-operative haemostasis)": "Desmopressin (DDAVP) 0.3 mcg/kg IV (max 20 mcg) in 50mL saline over 30min, 30-60 min before procedure; releases vWF from endothelial Weibel-Palade bodies → improves platelet adhesion; haemostatic effect lasts 4-6hr; tachyphylaxis with repeat doses (restrict to 3 doses); avoid: hyponatraemia (monitor Na+); NO evidence platelet dense granule production is restored (mechanism: vWF-mediated adhesion improvement independent of dense granules); haematology-directed use; platelet transfusion available if DDAVP insufficient",
            "HSCT (Chédiak-Higashi — curative)": "Allogeneic haematopoietic stem cell transplantation (HSCT) using matched related/unrelated/haplo donor; myeloablative or RIC conditioning; corrects NK cell cytotoxicity defect + phagocyte function + HLH risk; does NOT correct: (1) neurological progression (late spinocerebellar degeneration persists post-HSCT); (2) albinism (melanocyte-intrinsic, not haematopoietic); best outcomes: HSCT before accelerated phase, age <6yr, matched donor; survival >85% in modern series with HLA-matched donor; unrelated donor HSCT outcomes improving",
            "Melanocyte transplantation (Piebaldism)": "Autologous non-cultured epidermal cell suspension (ReCell) or cultured melanocyte suspension grafted onto dermabraded depigmented patch; efficacy variable (40-70% repigmentation in good studies); success depends on: thin scar formation, dermal architecture, patient compliance; optimal for stable patches with good wound healing; suction blister grafting: simpler; split-thickness skin graft: reliable but donor scar; NB-UVB phototherapy post-grafting stimulates melanocyte proliferation from graft; cosmetically significant for facial/hand patches; most patients choose conservative management",
        },
        "diagnostic_tests": {
            "VEP (Visual Evoked Potential) for OCA": "Gold standard for chiasmal misrouting in OCA; monocular flash/pattern-reversal VEP; in OCA: temporal retinal fibres aberrantly cross → contralateral hemisphere dominates for BOTH eyes; asymmetric hemispheric amplitude response = chiasmal misrouting; PATHOGNOMONIC for OCA; distinguishes OCA from CSNB (congenital stationary night blindness), INS (infantile nystagmus syndrome), spasmus nutans where VEP is symmetric; performed from birth (flash VEP); pattern-reversal from age 3-4yr",
            "Platelet electron microscopy (HPS)": "Transmission electron microscopy (TEM) of washed platelets: normal platelets: 3-8 dense granules per platelet (electron-opaque spots on TEM); HPS: ABSENT dense granules (completely empty — 'ghost' organelles visible in some subtypes); definitive diagnostic test for HPS; distinguishes HPS from other platelet function disorders (storage pool disease partial vs absent); whole blood lumi-aggregometry (ADP/collagen/arachidonic acid + luminescence for ATP release) confirms absent secondary aggregation wave in HPS; confirm with molecular panel for HPS gene subtype",
            "Peripheral blood smear — MPO stain (CHS)": "Peripheral blood smear with peroxidase (myeloperoxidase) staining: in CHS/LYST: giant peroxidase-positive intracytoplasmic granules in neutrophils, eosinophils, monocytes; granules visible on Wright-Giemsa (azurophilic giant granules); MPO stain highlights them dramatically; PATHOGNOMONIC for CHS; performed at diagnosis and during infection to monitor granule size; NK cell cytotoxicity assay (chromium release) confirms absent killing; molecular LYST gene sequencing confirms diagnosis",
            "HRCT chest (HPS1 pulmonary monitoring)": "High-resolution CT chest: OCA1B/HPS1 patients from age 20yr; classic UIP (usual interstitial pneumonia) pattern: subpleural bibasal fibrosis + honeycombing ± traction bronchiectasis; ground-glass may precede honeycombing; bronchoalveolar lavage: ceroid-laden macrophages (periodic acid-Schiff positive) = pathognomonic for HPS; interval HRCT 2-yearly unless accelerating; concurrent PFTs (FVC/DLCO/6-min walk): decline DLCO earliest functional marker; HRCT + PFTs together guide pirfenidone start and lung transplant referral timing",
            "Dermoscopy (MC1R melanoma surveillance)": "Total-body photography at baseline + annual dermoscopy by trained dermatologist for MC1R R-variant carriers; dermoscopic patterns: atypical network, regression, blue-white veil, irregular globules → biopsy; reflectance confocal microscopy (RCM): non-invasive assessment of borderline lesions; ABCDE self-examination monthly; any changing/new lesion → URGENT dermoscopy within 2 weeks; nevus count >50 + MC1R R-variant = very high risk; RCM-guided biopsy reduces unnecessary excision rate",
            "OCA NGS gene panel": "Comprehensive OCA sequencing: TYR, OCA2, TYRP1, SLC45A2 (OCA1-4 core) + HPS1, HPS3-8, AP3B1 (HPS) + LYST (CHS) + MC1R (red hair/melanoma) + KIT (piebaldism) + MITF/PAX3/SOX10/EDN3/EDNRB (Waardenburg) + BLOC1S5 (BLOC-1); identifies subtype for: accurate recurrence risk (AR vs AD); drug contraindications (HPS1/no-aspirin); organ surveillance (HPS1 lung; CHS HSCT); melanoma counselling (MC1R R-variants); avoids misclassification of e.g. OCA3-rufous as 'red hair variant'",
        },
    }


# Aliases for api_backend.py compatibility
def overview() -> dict:
    return generate_overview()


def breakdown() -> dict:
    return generate_breakdown()


def definitions() -> dict:
    return generate_definitions()


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"  Title: {ov['title']}")
    print(f"  Patients: {ov['n_patients']}  |  Genes: {ov['n_genes']}")
    print(f"  Seeds: {ov['seed_range']}")
    print("  Key pearls:")
    for p in ov["key_clinical_pearls"][:4]:
        print(f"    - {p[:120]}")
    print("\n=== BREAKDOWN (gene counts) ===")
    bk = generate_breakdown()
    for g, info in bk["gene_breakdown"].items():
        print(f"  {g}: {info['n_patients']} pts | Type: {info['pigment_category'][:70]}")
    print("\n=== DEFINITIONS (gene count) ===")
    df = generate_definitions()
    print(f"  Genes defined: {len(df['gene_entries'])}")
    print(f"  Biology glossary entries: {len(df['pigment_biology_glossary'])}")
