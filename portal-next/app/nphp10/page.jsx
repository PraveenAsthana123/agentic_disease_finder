'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & Allele Spectrum', 'Definitions'];

// NPHP10 colour scheme — centrosomal indigo-violet (SDCCAG8; retinal+renal+cerebellar; BBS16 bridge)
const ACCENT  = '#311b92';   // deep indigo — NPHP10 centrosomal identity; SDCCAG8
const ACCENT2 = '#1b5e20';   // deep green — renal transplant curative / excellent outcome
const ACCENT3 = '#6a1b9a';   // deep purple — retinal dystrophy (rod-cone; most frequent feature)
const ACCENT4 = '#00695c';   // dark teal — cerebellar ataxia (AHI1 interaction; NOT Joubert)
const ACCENT5 = '#e65100';   // deep orange — BBS16 overlap (obesity + cognitive + polydactyly)
const ACCENT6 = '#1a237e';   // deep navy — epidemiology / rare; centrosomal network
const ACCENT7 = '#ad1457';   // dark pink — LCA misdiagnosis (most common error; retinal-only workup)
const ACCENT8 = '#4e342e';   // dark brown — SDCCAG8 interactome (CEP290+IQCB1 direct binding)

const SEED = 359;
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

export default function NPHP10Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp10/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp10/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp10/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading NPHP10 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; Nephronophthisis Type 10 / Bardet-Biedl Syndrome 16 (NPHP10 / BBS16)
            </h4>
            <div className="small text-muted mb-1">
              <strong>SDCCAG8</strong> · 1q44 · 713 aa · Serologically Defined Colon Cancer Antigen 8 (CCCAP) ·
              Centrosomal / basal-body protein · Subdistal appendage complex · NOT a TZ scaffold ·
              CEP290 + IQCB1 direct interactor
            </div>
            <div className="small">
              <Badge text="OMIM *613524" color={ACCENT} />
              <Badge text="#613615 NPHP10" color={ACCENT} />
              <Badge text="#615993 BBS16" color={ACCENT5} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="1q44" color={ACCENT8} />
              <Badge text="Centrosomal NPHP" color={ACCENT4} />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT, fontSize: '0.8em' }}>
              ESRD median ~13–16yr
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              Retinal {ov.pct_retinal_dystrophy}% (rod-cone)
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT4, fontSize: '0.8em' }}>
              Cerebellar {ov.pct_cerebellar_ataxia}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              BBS16-overlap {ov.pct_bbs_overlap}%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; MOST COMMON MISDIAGNOSIS — LCA (retinal only):</strong> Retinal dystrophy
        dominates presentation → LCA gene panel → SDCCAG8 not on standard LCA-18 panel → renal monitoring
        omitted → ESRD at unscheduled presentation. SDCCAG8 MUST be on ALL LCA, NPHP, SLS, and BBS extended panels.
        Renal USS mandatory in all patients with rod-cone dystrophy / LCA.
      </Alert>
      <Alert color={ACCENT3}>
        <strong>&#x1f441;&#xfe0f; RETINAL DYSTROPHY {ov.pct_retinal_dystrophy}% — DOES NOT improve post-transplant:</strong> Rod-cone
        degeneration; may mimic LCA (early-onset flat ERG); cell-autonomous photoreceptor connecting cilium
        defect. Low-vision rehabilitation from diagnosis. ERG mandatory at NPHP10 diagnosis; annual ophthalmology.
      </Alert>
      <Alert color={ACCENT4}>
        <strong>&#x1f9e0; CEREBELLAR ATAXIA {ov.pct_cerebellar_ataxia}% — NOT Joubert (no Molar Tooth Sign):</strong> SDCCAG8
        binds AHI1 (Jouberin/JBTS3) → cerebellar vermis hypoplasia in subset. MRI confirms cerebellar
        hypoplasia WITHOUT molar tooth sign. Physiotherapy + OT improve function; independent of renal transplant.
      </Alert>
      <Alert color={ACCENT5}>
        <strong>&#x1f4c8; BBS16 OVERLAP {ov.pct_bbs_overlap}%:</strong> Biallelic severe truncating SDCCAG8 alleles →
        BBSome disruption → obesity + cognitive impairment ± rare post-axial polydactyly. BBS without
        polydactyly + TIN/NPHP → test SDCCAG8 on extended BBS panel (&gt;20 genes).
      </Alert>
      <Alert color={ACCENT2}>
        <strong>&#x2705; RENAL TRANSPLANT = CURATIVE:</strong> Cell-autonomous centrosomal defect. NO recurrence.
        Excellent graft outcomes. Retinal and cerebellar defects do NOT improve post-transplant (cell-autonomous).
        Living donor preferred.
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
            <KPI label="Median Hb" value={`${ov.median_hb} g/dL`} color="#5d4037" />
            <KPI label="Median age Dx" value={`${ov.median_age_renal_dx}yr`} color={ACCENT} />
            <KPI label="ESRD/Transplant" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT} />
            <KPI label="Retinal dystrophy" value={`${ov.pct_retinal_dystrophy}%`} color={ACCENT3} />
            <KPI label="Cerebellar ataxia" value={`${ov.pct_cerebellar_ataxia}%`} color={ACCENT4} />
            <KPI label="BBS16 overlap" value={`${ov.pct_bbs_overlap}%`} color={ACCENT5} />
            <KPI label="Polyuria first" value={`${ov.pct_polyuria_first_symptom}%`} color={ACCENT8} />
            <KPI label="Visual sx first" value={`${ov.pct_visual_symptoms_first}%`} color={ACCENT3} />
            <KPI label="LCA misdiag" value={`${ov.pct_misdiagnosed_as_lca}%`} color={ACCENT7} />
            <KPI label="Situs inversus" value="0%" color={ACCENT2} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="&#x1f9ec; SDCCAG8 — Centrosomal Architecture" color={ACCENT8}>
                <div className="small text-muted mb-2">
                  SDCCAG8 (CCCAP) localises to centriole subdistal appendages — anchors the basal body to
                  enable ciliogenesis. It bridges the NPHP TZ-network to centrosomal platform via CEP290
                  (NPHP6) and IQCB1 (NPHP5) direct binding. NOT a TZ scaffold (distinct from NPHP1-4-8-9).
                </div>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>SDCCAG8 (Serologically Defined Colon Cancer Antigen 8 / CCCAP)</td></tr>
                    <tr><td className="fw-bold">Location</td><td>1q44</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>713 aa · ~80 kDa · centrosomal basal-body protein</td></tr>
                    <tr><td className="fw-bold">Domains</td><td>CC targeting (1–200) + coiled-coil scaffold (200–550) + CEP290/IQCB1-binding C-term (550–713)</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*613524</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#613615 (NPHP10) · #615993 (BBS16)</td></tr>
                    <tr><td className="fw-bold">Key interactors</td><td>CEP290 (NPHP6), IQCB1 (NPHP5), OFD1, AHI1 (JBTS3), BBSome/ARL6</td></tr>
                    <tr><td className="fw-bold">Mechanism</td><td>Subdistal appendage → basal body anchoring → ciliogenesis initiation failure → TIN</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/200,000–500,000 (~100+ reported cases)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive · biallelic LOF</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="&#x1f6a8; NPHP10 Hallmark Features" color={ACCENT}>
                {[
                  [`Retinal dystrophy ${ov.pct_retinal_dystrophy}% (rod-cone)`, ACCENT3,
                   'Most frequent extra-renal feature; LCA-like in severe; ERG flat; visual loss; cell-autonomous — NO improvement post-Tx'],
                  [`Cerebellar ataxia ${ov.pct_cerebellar_ataxia}% (NOT Joubert)`, ACCENT4,
                   'AHI1 interaction → cerebellar vermis hypoplasia WITHOUT molar tooth sign; MRI distinguishes from Joubert'],
                  [`BBS16 overlap ${ov.pct_bbs_overlap}% (obesity + cognitive)`, ACCENT5,
                   'Severe truncating alleles → BBSome disruption → obesity ± mild cognitive ± rare polydactyly'],
                  ['CENTROSOMAL (not TZ scaffold)', ACCENT,
                   'Basal body anchoring failure at subdistal appendages — distinct from NPHP1-4-8-9 TZ diffusion barrier collapse'],
                  ['NO situs inversus (0%)', ACCENT2,
                   'SDCCAG8 absent from nodal cilia; laterality always normal — differentiates from NPHP2/3/9'],
                  ['NO CHF (0%)', ACCENT2,
                   'SDCCAG8 absent from biliary epithelium; no ductal plate dysfunction; no hepatic fibrosis'],
                  ['NO pancreatic ductal ectasia (0%)', ACCENT2,
                   'Absent from pancreatic ducts — unlike NPHP9 (NEK8); helps differentiate from NPHP9'],
                  [`ESRD median ~${ov.median_age_renal_dx}yr (cohort)`, ACCENT,
                   'Small echogenic kidneys; corticomedullary cysts; TIN — NPHP1/5/6-like pattern; range 4–30yr'],
                  ['Renal transplant CURATIVE', ACCENT2,
                   'Cell-autonomous centrosomal defect; NO recurrence; retinal/cerebellar defects independent'],
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
                    <th>Retinal</th><th>Cerebellar</th><th>BBS16</th>
                    <th>First Symptom</th>
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
                      <td>
                        {p.retinal_dystrophy
                          ? <span className="badge" style={{ background: ACCENT3 }}>Ret ✓</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td>
                        {p.cerebellar_ataxia
                          ? <span className="badge" style={{ background: ACCENT4 }}>Cereb ✓</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td>
                        {p.bbs_overlap
                          ? <span className="badge" style={{ background: ACCENT5 }}>BBS ✓</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td style={{ fontSize: '0.72em' }}>{p.first_symptom.split('(')[0].trim().slice(0, 30)}</td>
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
            <Section title="Kidney Phenotype Distribution" color={ACCENT}>
              {Object.entries(bk.kidney_phenotype_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Retinal Status Distribution" color={ACCENT3}>
              {Object.entries(bk.retinal_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Cerebellar Status Distribution" color={ACCENT4}>
              {Object.entries(bk.cerebellar_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
            </Section>
            <Section title="BBS16-Overlap Distribution" color={ACCENT5}>
              {Object.entries(bk.bbs_overlap_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="CKD Stage Distribution" color={ACCENT6}>
              {Object.entries(bk.ckd_stage_current).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
            <Section title="RRT / Transplant Status" color={ACCENT2}>
              {Object.entries(bk.rrt_transplant_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Prior Misdiagnosis (most common: LCA)" color={ACCENT7}>
              {Object.entries(bk.prior_misdiagnosis).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="Age at Renal Diagnosis — Tiers" color={ACCENT}>
              {Object.entries(bk.age_at_renal_dx_tiers).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
          </div>
          <div className="col-12">
            <div className="row g-3">
              <div className="col-md-4">
                <Section title="Ethnicity Distribution" color={ACCENT8}>
                  {Object.entries(bk.ethnicity).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="Urine Osmolality Tiers (Tubular Defect)" color={ACCENT}>
                  {Object.entries(bk.urine_osmolality_tiers).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="GFR Slope Tiers (Progression Rate)" color={ACCENT6}>
                  {Object.entries(bk.gfr_slope_tiers).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                  ))}
                </Section>
              </div>
            </div>
          </div>
          <div className="col-12">
            <Section title="First Presenting Symptom" color={ACCENT}>
              <div className="row g-2">
                {Object.entries(bk.first_symptom_distribution).map(([k, v]) => (
                  <div key={k} className="col-md-6">
                    <Bar label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  </div>
                ))}
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 2: Genetics & Allele Spectrum ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="&#x1f9ec; Gene Structure & Centrosomal Architecture" color={ACCENT8}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="&#x1f9ea; Key Variants (SDCCAG8 NPHP10 / BBS16)" color={ACCENT3}>
              {df.key_variants && df.key_variants.map((v, i) => (
                <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT3 + '08', borderLeft: `3px solid ${ACCENT3}` }}>
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
                    <tr><th>Condition</th><th>Key Distinguishing Features from NPHP10</th></tr>
                  </thead>
                  <tbody>
                    {df.ddx_table && Object.entries(df.ddx_table).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT7, minWidth: 160 }}>{k}</td>
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
            <Section title="Mechanism" color={ACCENT8}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.mechanism}
              </div>
            </Section>
            <Section title="&#x1f3e5; Treatment" color={ACCENT2}>
              {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT2 + '08', borderLeft: `3px solid ${ACCENT2}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
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
            <Section title="Diagnostic Criteria" color={ACCENT4}>
              {df.diagnostic_criteria && Object.entries(df.diagnostic_criteria).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT4 + '08', borderLeft: `3px solid ${ACCENT4}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}</div>
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
        <Link href="/nphp9" className="btn btn-sm btn-outline-primary">&#x2190; NPHP9 (NEK8)</Link>
      </div>
    </div>
  );
}
