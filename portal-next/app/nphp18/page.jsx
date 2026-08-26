'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'CEP83 Architecture & DA Hierarchy', 'Definitions'];

// NPHP18 colour scheme — CEP83 / distal appendage foundation / JBTS22 / DA hierarchy
const ACCENT  = '#1a237e';   // deep indigo — CEP83 DA foundation; master organiser; proximal DA
const ACCENT2 = '#880e4f';   // deep pink/magenta — Joubert syndrome (JBTS22); MTS; cerebellar
const ACCENT3 = '#e65100';   // burnt orange — retinal dystrophy; rod-cone; LCA-like
const ACCENT4 = '#1b5e20';   // deep green — DA hierarchy cascade; CEP89→SCLT1→FBF1→CEP164
const ACCENT5 = '#b71c1c';   // deep red — ESRD; CKD progression; tubular loss
const ACCENT6 = '#37474f';   // dark slate — molecular architecture; protein domains
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts; CEP290 confusion; NPHP1 MLPA
const ACCENT8 = '#004d40';   // dark teal — DA scaffold; distal appendage biology

const SEED = 375;
const _COHORT_SIZE = 40;

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Alert({ color, children }) {
  return (
    <div className="alert mb-2" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
      {children}
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function Badge({ text, color }) {
  return <span className="badge me-1" style={{ background: color, fontSize: '0.72em' }}>{text}</span>;
}

function Bar({ label, value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{value}</span>
      </div>
      <div className="progress" style={{ height: 8, borderRadius: 4 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color, borderRadius: 4 }} />
      </div>
    </div>
  );
}

export default function NPHP18Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp18/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp18/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp18/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading NPHP18 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; Nephronophthisis Type 18 / Joubert Syndrome 22 (NPHP18/JBTS22) — CEP83 · DA Foundation
            </h4>
            <div className="small text-muted mb-1">
              <strong>CEP83</strong> (CCDC41) · 12q22 · ~826 aa · most PROXIMAL distal appendage protein ·
              nucleates CEP89→SCLT1→FBF1→LRRC45→CEP164 hierarchy · JBTS22 in ~55% · retinal ~30–40% ·
              ESRD median ~14–18yr
            </div>
            <div className="small">
              <Badge text="OMIM *617233" color={ACCENT} />
              <Badge text="#617265 NPHP18" color={ACCENT} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="12q22" color={ACCENT8} />
              <Badge text="DA foundation" color={ACCENT4} />
              <Badge text="JBTS22 ~55%" color={ACCENT2} />
              <Badge text="Retinal ~30–40%" color={ACCENT3} />
              <Badge text="CEP164 directly downstream" color={ACCENT4} />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT, fontSize: '0.8em' }}>
              ESRD median ~14–18yr
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT2, fontSize: '0.8em' }}>
              JBTS22 {ov.pct_jbts22_confirmed}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              Retinal {ov.pct_retinal_dystrophy}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              ESRD/Tx {ov.pct_esrd_or_transplant}%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT4}>
        <strong>&#x1f9ec; CEP83 IS THE PROXIMAL FOUNDATION OF THE ENTIRE DISTAL APPENDAGE SCAFFOLD:</strong> CEP83 (CCDC41)
        is the most upstream component of the distal appendage (DA) hierarchy at the mother centriole.
        CEP83 nucleates the entire cascade: <strong>CEP83 → CEP89 → SCLT1 → FBF1 → LRRC45 → CEP164 (NPHP15)</strong>.
        Loss of CEP83 removes ALL downstream DA proteins from the centriole simultaneously — unlike loss
        of CEP164 (NPHP15) which removes only the terminal step. DA is required for centriole-to-vesicle
        docking, CP110 cap removal, and axoneme initiation. CEP83 is the only NPHP gene encoding a
        proximal DA foundation protein; loss phenocopies NPHP15 (CEP164) PLUS adds Joubert syndrome.
      </Alert>
      <Alert color={ACCENT2}>
        <strong>&#x1f9e0; JOUBERT SYNDROME 22 (JBTS22) — MOLAR TOOTH SIGN IN ~{ov.pct_jbts22_confirmed}% — BRAIN MRI MANDATORY:</strong> Biallelic
        CEP83 LOF causes Joubert syndrome (Molar Tooth Sign on axial MRI: cerebellar vermis hypoplasia +
        SCP elongation) in ~55–65% of cases. Oculomotor apraxia (OMA), neonatal hypotonia, cerebellar
        ataxia, developmental delay. MTS ABSENT in ~30% (pure renal NPHP18 alleles). Brain MRI
        MANDATORY at diagnosis — distinguishes JBTS22 alleles from pure renal NPHP18 alleles,
        guides developmental paediatrics referral, and determines ophthalmology surveillance intensity.
        CEP83 is only the 2nd DA protein (after CEP164/NPHP15) associated with both JBTS and NPHP.
      </Alert>
      <Alert color={ACCENT3}>
        <strong>&#x1f441;&#xfe0f; RETINAL DYSTROPHY IN ~{ov.pct_retinal_dystrophy}% — ANNUAL ERG MANDATORY:</strong> Rod–cone dystrophy in ~28–35%
        (ERG abnormal, progressive) and LCA-like severe early retinal disease in ~5–7% (ERG flat,
        neonatal nystagmus; null×null alleles). Retinal does NOT improve post-renal transplant
        (cell-autonomous photoreceptor connecting cilium defect). Annual ERG + fundoscopy mandatory
        in ALL biallelic CEP83 patients from diagnosis. Distinguished from NPHP6/CEP290 (65% retinal,
        LCA10 IVS26 most common LCA variant) and NPHP5/IQCB1 (>95% retinal).
      </Alert>
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; DIAGNOSTIC TRAPS — CEP290 ON 12q21.32 vs CEP83 ON 12q22 (SAME ARM, DIFFERENT GENES):</strong> CEP290 (NPHP6)
        at 12q21.32 and CEP83 at 12q22 are on the same chromosome arm — targeted CEP290 single-gene
        panels do NOT cover CEP83. NPHP1 MLPA (290kb deletion test) also misses CEP83. Both genes
        can cause JBTS + retinal + NPHP, making clinical distinction difficult. {ov.pct_misdiagnosed_cep290}% of
        cohort initially underwent CEP290 testing that was negative before CEP83 was tested.
        WES mandatory. CEP164 (NPHP15) must ALWAYS be co-sequenced — same DA hierarchy, directly
        downstream of CEP83.
      </Alert>
      <Alert color={ACCENT}>
        <strong>&#x2705; RENAL TRANSPLANT CURATIVE FOR NEPHRONOPHTHISIS · RETINAL/CEREBELLAR DO NOT IMPROVE:</strong> Donor
        kidney has functional CEP83 → normal DA assembly → intact primary cilia → no TIN recurrence.
        Excellent graft outcomes; pre-emptive transplant preferred. HOWEVER: retinal dystrophy
        and cerebellar hypoplasia (JBTS22) are cell-autonomous and do NOT improve post-transplant.
        Multi-disciplinary team (nephrology + ophthalmology + neurodevelopmental paediatrics) required
        for JBTS22 cases. Pure renal NPHP18 cases: focused nephrology management only post-transplant.
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Cohort (n)" value={ov.cohort_n} color={ACCENT} />
            <KPI label="Median GFR" value={`${ov.median_gfr} ml/min`} color={ACCENT} />
            <KPI label="Median Hb" value={`${ov.median_hb} g/dL`} color={ACCENT6} />
            <KPI label="Median age Dx" value={`${ov.median_age_renal_dx}yr`} color={ACCENT} />
            <KPI label="ESRD/Transplant" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT5} />
            <KPI label="JBTS22 (MTS)" value={`${ov.pct_jbts22_confirmed}%`} color={ACCENT2} />
            <KPI label="Retinal dystrophy" value={`${ov.pct_retinal_dystrophy}%`} color={ACCENT3} />
            <KPI label="LCA-like" value={`${ov.pct_lca_like}%`} color={ACCENT3} />
            <KPI label="Situs inversus" value="<2%" color={ACCENT} />
            <KPI label="NPHP1 MLPA misdiag" value={`${ov.pct_misdiagnosed_nphp1}%`} color={ACCENT7} />
            <KPI label="CEP290 misdiag" value={`${ov.pct_misdiagnosed_cep290}%`} color={ACCENT7} />
            <KPI label="JBTS-unknown misdiag" value={`${ov.pct_misdiagnosed_jbts_unknown}%`} color={ACCENT7} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="&#x1f9ec; CEP83 — Distal Appendage Foundation (12q22)" color={ACCENT6}>
                <div className="small text-muted mb-2">
                  CEP83 (CCDC41; ~826 aa) is the most proximal distal appendage (DA) protein —
                  the foundational scaffold that nucleates the entire DA hierarchy at the mother
                  centriole. Loss of CEP83 removes ALL downstream DA proteins (CEP89, SCLT1,
                  FBF1, LRRC45, CEP164/NPHP15) simultaneously, ablating ciliogenesis.
                </div>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>CEP83 (also CCDC41 — Coiled-Coil Domain Containing 41)</td></tr>
                    <tr><td className="fw-bold">Location</td><td>12q22 (distinct from CEP290 at 12q21.32)</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>~826 aa · most proximal DA component · DA foundation · coiled-coil rich</td></tr>
                    <tr><td className="fw-bold">DA Hierarchy</td><td>CEP83 → CEP89 → SCLT1 → FBF1 → LRRC45 → CEP164 (NPHP15)</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*617233</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#617265 (NPHP18 / JBTS22)</td></tr>
                    <tr><td className="fw-bold">Key partners</td><td>CEP89/CCDC123 (direct; DA step 1); SCLT1; IFT-B; Rab8a/RABIN8; CP110/CEP97 (antagonist); TTBK2 (phosphorylates CEP83)</td></tr>
                    <tr><td className="fw-bold">Phenotypes</td><td>NPHP18 (renal) + JBTS22 (Joubert, ~55%) + retinal (~30–40%) + rare CHF (~5–10%)</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/500,000–1,000,000; ~60–90 published families (2026)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive · biallelic LOF</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="&#x1f6a8; NPHP18 / JBTS22 Hallmark Features" color={ACCENT}>
                {[
                  ['CEP83 is the proximal DA foundation — loss destroys ALL downstream DA proteins (CEP89, SCLT1, FBF1, LRRC45, CEP164)', ACCENT4,
                   'CEP83 is the only NPHP gene encoding the proximal distal appendage foundation. Loss simultaneously removes CEP89, SCLT1, FBF1, LRRC45, and CEP164 (NPHP15) from the centriole. This distinguishes NPHP18 from NPHP15: both target the same DA hierarchy, but CEP83 loss is more upstream and broader in its cascade effect. CEP164 (NPHP15) must ALWAYS be co-sequenced'],
                  ['JBTS22 — Molar Tooth Sign in ~55% — Brain MRI mandatory at diagnosis', ACCENT2,
                   'Cerebellar vermis hypoplasia + SCP elongation (MTS) on axial T1/T2 MRI in ~55–65% of biallelic CEP83 cases. Oculomotor apraxia (OMA), neonatal hypotonia, ataxia. MRI at diagnosis determines: JBTS22 alleles vs pure renal NPHP18 alleles; developmental paediatrics referral; ophthalmology surveillance intensity. Pure renal NPHP18 (~30%): no MTS, no OMA, no cerebellar signs'],
                  [`Retinal dystrophy ~${ov.pct_retinal_dystrophy}% — rod-cone + LCA-like; ERG annual mandatory`, ACCENT3,
                   'Rod-cone dystrophy in ~28–35% (ERG abnormal, progressive from rods outward). LCA-like severe early in ~5–7% (ERG flat, neonatal nystagmus; null×null alleles). Retinal does NOT improve post-transplant (cell-autonomous photoreceptor connecting cilium defect). Annual ERG + fundoscopy from diagnosis. Distinguished from CEP290 (65% retinal, LCA10) and IQCB1/NPHP5 (>95% retinal)'],
                  ['Pure renal NPHP18 (~30%) — ESRD median 14–18yr; no MTS, no retinal', ACCENT,
                   'Missense×missense (hypomorphic) alleles cause pure renal phenotype: TIN + corticomedullary cysts + concentrating defect → ESRD. No cerebellar (MTS absent), no retinal (ERG normal). Onset ~10–14yr. Renal transplant CURATIVE; cell-autonomous DA defect → no recurrence in graft. Pre-emptive transplant preferred. Focused nephrology management only'],
                  ['CEP290 (12q21.32) confusion — same chromosome arm, DIFFERENT gene; WES mandatory', ACCENT7,
                   'CEP290 (NPHP6/LCA10) at 12q21.32 and CEP83 at 12q22 are on the same chromosome arm 12q — targeted CEP290 gene panels do NOT cover CEP83. NPHP1 MLPA (290kb) misses CEP83. Both cause JBTS + retinal + NPHP, making clinical DDx difficult. WES is the ONLY reliable method. ~13% of cohort had negative CEP290 testing before CEP83 identified'],
                  ['CHF rare (~5–10%) — annual liver USS until age 18yr if present', ACCENT8,
                   'Biliary ductal plate malformation in ~5–10% of biallelic CEP83 patients — much less frequent than TMEM67/NPHP11 (40–50% CHF) or RPGRIP1L/NPHP8 (15–20% CHF). Annual liver USS + LFTs if CHF confirmed until age 18yr; portal hypertension surveillance after 10yr. CEP83 not abundant in cholangiocytes; biliary involvement likely due to partial ductal plate ciliation loss'],
                  ['Allele–phenotype rule: null×null → JBTS22+severe+retinal; missense×missense → pure renal mild', ACCENT6,
                   'Truncating×truncating biallelic variants: JBTS22 most likely + LCA-like retinal + ESRD <12yr. Truncating×missense: JBTS22 probable; renal 60–70%; rod-cone retinal variable. Missense×missense (hypomorphic: p.Ala412Val, p.Leu287Pro, p.Glu195Lys): pure renal NPHP18 preferred; MTS absent or mild; ESRD 18–25yr; ERG often normal. Allele type predicts phenotype class'],
                  ['Renal transplant CURATIVE · cerebellar/retinal cell-autonomous — excellent renal outcomes', ACCENT,
                   'Donor kidney has functional CEP83 → normal DA → intact tubular cilia → no TIN. No disease recurrence in graft. Excellent long-term renal outcomes post-transplant. HOWEVER: retinal dystrophy and cerebellar hypoplasia (JBTS22) are cell-autonomous — NOT corrected by renal transplant. Multi-disciplinary team mandatory for JBTS22: nephrology + developmental paediatrics + ophthalmology + physiotherapy + OT + speech'],
                ].map(([title, color, sub], i) => (
                  <div key={i} className="mb-2 p-2 rounded" style={{ background: color + '12', borderLeft: `3px solid ${color}` }}>
                    <div className="fw-bold small" style={{ color }}>{title}</div>
                    <div className="text-muted" style={{ fontSize: '0.75em' }}>{sub}</div>
                  </div>
                ))}
              </Section>
            </div>
          </div>

          {/* Sample patients */}
          <Section title={`&#x1f4cb; Sample Patients (first 8 of ${_COHORT_SIZE} · seed=${SEED})`} color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead>
                  <tr style={{ background: ACCENT + '22' }}>
                    <th>ID</th><th>Ethnicity</th><th>CKD Stage</th>
                    <th>Age Dx (yr)</th><th>GFR</th>
                    <th>JBTS22</th><th>Retinal</th><th>Misdiagnosis</th>
                  </tr>
                </thead>
                <tbody>
                  {ov.patients.map(p => (
                    <tr key={p.id}>
                      <td><span className="badge" style={{ background: ACCENT }}>{p.id}</span></td>
                      <td style={{ fontSize: '0.72em' }}>{p.ethnicity.split('(')[0].trim()}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.ckd_stage.split('(')[0].trim()}</td>
                      <td>{p.age_renal_dx_yr}</td>
                      <td>{p.gfr_now_ml_min}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.jbts22_status.split('(')[0].trim().slice(0, 22)}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.retinal_status.split('(')[0].trim().slice(0, 22)}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.prior_misdiagnosis.split('(')[0].trim().slice(0, 26)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── Tab 1: Diagnostic Breakdown ── */}
      {tab === 1 && bk && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="CKD Stage / Renal Status Distribution" color={ACCENT}>
              {Object.entries(bk.ckd_stage_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="JBTS22 Status (Joubert — Molar Tooth Sign)" color={ACCENT2}>
              {Object.entries(bk.jbts22_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Retinal Involvement" color={ACCENT3}>
              {Object.entries(bk.retinal_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Urine Osmolality Tiers (Tubular Concentrating Defect)" color={ACCENT}>
              {Object.entries(bk.urine_osmolality_tiers).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Prior Misdiagnosis Distribution" color={ACCENT7}>
              {Object.entries(bk.prior_misdiagnosis).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="First Presenting Symptom" color={ACCENT}>
              {Object.entries(bk.first_symptom_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Kidney USS Phenotype" color={ACCENT8}>
              {Object.entries(bk.kidney_phenotype).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
              ))}
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT6}>
              {Object.entries(bk.ethnicity).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 2: CEP83 Architecture & DA Hierarchy ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="&#x1f9ec; CEP83 Architecture, DA Hierarchy & Ciliogenesis" color={ACCENT4}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).filter(([k]) => k !== 'key_variants').map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT4}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{typeof v === 'string' ? v : JSON.stringify(v)}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="&#x1f9ea; Key Variants (CEP83 NPHP18/JBTS22)" color={ACCENT}>
              {df.genetic_architecture && df.genetic_architecture.key_variants && df.genetic_architecture.key_variants.map((v, i) => (
                <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                  <div className="small text-muted">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="&#x1f4ca; NPHP Subtype Comparison" color={ACCENT6}>
              {df.nphp_comparison && Object.entries(df.nphp_comparison).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded"
                  style={{
                    background: k.includes('★') ? ACCENT + '18' : ACCENT6 + '08',
                    borderLeft: `3px solid ${k.includes('★') ? ACCENT : ACCENT6}`
                  }}>
                  <div className="fw-bold small" style={{ color: k.includes('★') ? ACCENT : ACCENT6 }}>{k}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-12">
            <Section title="&#x1fa7a; Differential Diagnosis Table" color={ACCENT7}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead style={{ background: ACCENT7 + '18' }}>
                    <tr><th>Condition</th><th>Key Distinguishing Features from NPHP18/JBTS22</th></tr>
                  </thead>
                  <tbody>
                    {df.ddx_table && Object.entries(df.ddx_table).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT7, minWidth: 200 }}>{k}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 3: Definitions ── */}
      {tab === 3 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Disease Definition" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {['disease','omim_gene','omim_disease','chromosome','inheritance','prevalence'].map(k => (
                    <tr key={k}>
                      <td className="fw-bold text-capitalize" style={{ width: 140 }}>{k.replace(/_/g,' ')}</td>
                      <td>{df[k]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="Mechanism" color={ACCENT2}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.mechanism}
              </div>
            </Section>
            <Section title="&#x1f3e5; Treatment" color={ACCENT}>
              {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Key Clinical Features" color={ACCENT3}>
              {df.key_clinical_features && Object.entries(df.key_clinical_features).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT3 + '08', borderLeft: `3px solid ${ACCENT3}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="Diagnostic Criteria" color={ACCENT6}>
              {df.diagnostic_criteria && Object.entries(df.diagnostic_criteria).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT6 + '08', borderLeft: `3px solid ${ACCENT6}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT6 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="Prognosis" color={ACCENT}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.prognosis}
              </div>
            </Section>
            <Section title="Cohort Note" color={ACCENT6}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT6 + '06', lineHeight: 1.6 }}>
                {df.cohort_note}
              </div>
            </Section>
          </div>
        </div>
      )}

      <div className="mt-4 pt-3 border-top">
        <Link href="/" className="btn btn-sm btn-outline-secondary me-2">&#x2190; Portal Home</Link>
        <Link href="/nphp17" className="btn btn-sm btn-outline-primary">&#x2190; NPHP17 (MAPKBP1/JNK Scaffold)</Link>
      </div>
    </div>
  );
}
