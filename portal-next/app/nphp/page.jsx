'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & Subtype', 'Definitions'];

// NPHP colour scheme — cobalt-teal-burgundy-amber (renal; tubular; cystic ciliopathy)
const ACCENT  = '#0d47a1';   // deep cobalt — nephronophthisis; renal TZ ciliopathy
const ACCENT2 = '#004d40';   // dark teal — ESRD / transplant / renal replacement therapy
const ACCENT3 = '#1b5e20';   // dark green — NPHP1 gene / 2q13 deletion (most common)
const ACCENT4 = '#b71c1c';   // deep red — ESRD progression / anemia / clinical urgency
const ACCENT5 = '#4a148c';   // deep purple — Senior-Løken syndrome / retinal dystrophy
const ACCENT6 = '#37474f';   // dark slate — epidemiology / AR genetics / consanguinity
const ACCENT7 = '#e65100';   // deep orange — misdiagnosis / dehydration risk alert
const ACCENT8 = '#827717';   // dark amber — urine osmolality / tubular defect

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
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function NphpPage() {
  const [tab, setTab] = useState(0);
  const [ov, setOv]   = useState(null);
  const [bk, setBk]   = useState(null);
  const [df, setDf]   = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/nphp/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="container py-4 text-danger">Error: {err}</div>;
  if (!ov)  return <div className="container py-4 text-muted">Loading NPHP dashboard…</div>;

  const kpis = ov.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          🧬 Nephronophthisis Type 1 (NPHP1 — Juvenile NPHP; Senior-Løken Syndrome 1)
        </h4>
        <div className="text-muted small">
          NPHP1 · 2q13 · ~290 kb homozygous deletion · TZ ciliopathy (non-motile) · AR Biallelic LOF · OMIM #256100
          &nbsp;|&nbsp; Cohort: {_COHORT_SIZE} patients (seed-341) · 3 endpoints verified
        </div>
        <div className="mt-1">
          <Badge text="TZ Ciliopathy" color={ACCENT} />
          <Badge text="ESRD Median 13yr" color={ACCENT4} />
          <Badge text="2q13 Deletion 80%" color={ACCENT3} />
          <Badge text="Transplant EXCELLENT" color={ACCENT2} />
          <Badge text="Senior-Løken ~12%" color={ACCENT5} />
          <Badge text="AR" color={ACCENT6} />
          <Badge text="~1/50,000" color={ACCENT6} />
        </div>
      </div>

      {/* Tab Nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row mb-2">
            <KPI label="Cohort (n)" value={kpis.cohort_n ?? _COHORT_SIZE} color={ACCENT} />
            <KPI label="Median Dx Age (yr)" value={kpis.median_age_dx_yr ?? '—'} color={ACCENT2} />
            <KPI label="Median GFR at Dx" value={kpis.median_gfr_at_dx_ml_min ? `${kpis.median_gfr_at_dx_ml_min} mL/min` : '—'} color={ACCENT4} />
            <KPI label="Median U-Osm" value={kpis.median_urine_osmolality ? `${kpis.median_urine_osmolality} mosm/kg` : '—'} color={ACCENT8} />
            <KPI label="% ESRD/RRT" value={kpis.pct_esrd_or_rrt ? `${kpis.pct_esrd_or_rrt}%` : '—'} color={ACCENT4} />
            <KPI label="% Transplanted" value={kpis.pct_transplanted ? `${kpis.pct_transplanted}%` : '—'} color={ACCENT2} />
          </div>
          <div className="row mb-3">
            <KPI label="% 2q13 Deletion" value={kpis.pct_2q13_homoz_deletion ? `${kpis.pct_2q13_homoz_deletion}%` : '—'} color={ACCENT3} />
            <KPI label="% Senior-Løken" value={kpis.pct_senior_loken ? `${kpis.pct_senior_loken}%` : '—'} color={ACCENT5} />
            <KPI label="% Polyuria First" value={kpis.pct_polyuria_presenting ? `${kpis.pct_polyuria_presenting}%` : '—'} color={ACCENT8} />
            <KPI label="Mean Hgb (g/dL)" value={kpis.mean_hgb_g_dl ?? '—'} color={ACCENT4} />
            <KPI label="% Consanguineous" value={kpis.pct_consanguineous ? `${kpis.pct_consanguineous}%` : '—'} color={ACCENT6} />
            <KPI label="% EPO Therapy" value={kpis.pct_epo_therapy ? `${kpis.pct_epo_therapy}%` : '—'} color={ACCENT4} />
          </div>

          {/* Alerts */}
          <Section title="⚠️ Clinical Alerts" color={ACCENT4}>
            {ov.alerts && Object.entries(ov.alerts).map(([k, v]) => (
              <Alert key={k} color={ACCENT4}>
                <span className="fw-bold">{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}:</span>{' '}{v}
              </Alert>
            ))}
          </Section>

          {/* Key Facts */}
          <Section title="🔬 Key Facts — NPHP1 Clinical & Molecular" color={ACCENT}>
            <ul className="mb-0 small">
              {(ov.key_facts || []).map((f, i) => <li key={i} className="mb-1">{f}</li>)}
            </ul>
          </Section>

          {/* Clinical Profile Summary */}
          <Section title="🫘 NPHP1 Clinical Profile Snapshot" color={ACCENT2}>
            <div className="row">
              <div className="col-md-6">
                <table className="table table-sm table-bordered small mb-0">
                  <tbody>
                    <tr><td className="fw-bold" style={{ color: ACCENT }}>Primary Gene (most common)</td><td>{kpis.gene}</td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>{kpis.chromosome}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{kpis.inheritance}</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>{kpis.prevalence}</td></tr>
                    <tr><td className="fw-bold">Cohort Type</td><td>{kpis.cohort_type}</td></tr>
                    <tr><td className="fw-bold">Syndrome</td><td>{kpis.syndrome}</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                <div className="p-2 rounded small" style={{ background: ACCENT + '10', border: `1px solid ${ACCENT}40` }}>
                  <div className="fw-bold mb-1" style={{ color: ACCENT }}>vs. ADPKD (most common misdiagnosis):</div>
                  <ul className="mb-0">
                    <li><b>NPHP: SMALL or normal kidneys</b> — ADPKD: MASSIVELY ENLARGED (key discriminator)</li>
                    <li><b>NPHP: Autosomal RECESSIVE</b> — ADPKD: Autosomal DOMINANT</li>
                    <li><b>NPHP: No haematuria</b> — ADPKD: haematuria common</li>
                    <li><b>NPHP: Concentrating defect (polyuria)</b> — ADPKD: HTN flank pain first</li>
                    <li><b>NPHP: MLPA/array-CGH for 2q13 deletion</b> — ADPKD: PKD1/2 sequencing</li>
                    <li><b>NPHP Transplant: NO recurrence</b> — gene panel essential before living donor</li>
                  </ul>
                </div>
              </div>
            </div>
          </Section>

          {/* Patient Sample */}
          <Section title={`👥 Sample Patients (first 8 of ${_COHORT_SIZE})`} color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-hover small mb-0">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr>
                    <th>ID</th><th>Gene</th><th>Ethnicity</th>
                    <th>Dx Age</th><th>GFR Dx</th><th>GFR Now</th>
                    <th>U-Osm</th><th>Hgb</th><th>SLS</th><th>RRT/Tx</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.patients || []).map(p => (
                    <tr key={p.id}>
                      <td><code style={{ fontSize: '0.75em' }}>{p.id}</code></td>
                      <td style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.gene}>{p.gene?.split('(')[0]?.trim()}</td>
                      <td>{p.ethnicity?.split('(')[0]?.trim()}</td>
                      <td>{p.age_at_diagnosis_yr} yr</td>
                      <td style={{ color: p.gfr_at_dx_ml_min < 30 ? ACCENT4 : ACCENT2 }}>
                        {p.gfr_at_dx_ml_min}
                      </td>
                      <td style={{ color: p.gfr_now_ml_min < 15 ? ACCENT4 : p.gfr_now_ml_min < 30 ? ACCENT7 : ACCENT2 }}>
                        {p.gfr_now_ml_min}
                      </td>
                      <td style={{ color: p.urine_osmolality_mosm < 200 ? ACCENT4 : ACCENT8 }}>
                        {p.urine_osmolality_mosm}
                      </td>
                      <td style={{ color: p.hemoglobin_g_dl < 9 ? ACCENT4 : ACCENT6 }}>
                        {p.hemoglobin_g_dl}
                      </td>
                      <td>{p.senior_loken ? <span style={{ color: ACCENT5 }}>✓ SLS</span> : '—'}</td>
                      <td style={{ maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.rrt_or_transplant}>{p.rrt_or_transplant?.split('—')[0]?.trim().split(',')[0]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 1: Diagnostic Breakdown ── */}
      {tab === 1 && bk && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="🫘 Renal Ultrasound Findings" color={ACCENT}>
                {Object.entries(bk.renal_ultrasound || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                ))}
              </Section>

              <Section title="📊 CKD Stage at Diagnosis" color={ACCENT4}>
                {Object.entries(bk.ckd_stage_at_diagnosis || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>

              <Section title="💧 Urine Osmolality Tiers (mosm/kg)" color={ACCENT8}>
                {Object.entries(bk.urine_osmolality_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                ))}
              </Section>

              <Section title="👁️ Senior-Løken Distribution" color={ACCENT5}>
                {Object.entries(bk.senior_loken_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
                ))}
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="🔬 MLPA 2q13 Deletion Result" color={ACCENT3}>
                {Object.entries(bk.mlpa_2q13_result || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>

              <Section title="💊 RRT / Transplant Status" color={ACCENT2}>
                {Object.entries(bk.rrt_transplant_status || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
                ))}
              </Section>

              <Section title="⏱️ Age at Diagnosis Tiers" color={ACCENT6}>
                {Object.entries(bk.age_at_diagnosis_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                ))}
              </Section>

              <Section title="⚠️ Prior Misdiagnosis" color={ACCENT7}>
                {Object.entries(bk.prior_misdiagnosis || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
                ))}
              </Section>
            </div>
          </div>

          {/* Second row */}
          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 Gene Distribution" color={ACCENT3}>
                {Object.entries(bk.gene_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="🌍 Ethnicity" color={ACCENT6}>
                {Object.entries(bk.ethnicity || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                ))}
              </Section>

              <Section title="🎯 Presenting Symptom" color={ACCENT8}>
                {Object.entries(bk.presenting_symptom || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                ))}
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: Genetics & Subtype ── */}
      {tab === 2 && df && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 Genetic Architecture" color={ACCENT3}>
                {Object.entries(df.genetic_architecture || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="📋 NPHP Subtypes" color={ACCENT}>
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: ACCENT + '15' }}>
                    <tr><th>Subtype</th><th>Details</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(df.nphp_subtypes || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT, whiteSpace: 'nowrap' }}>{k.replace(/_/g, ' ')}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>

              <Section title="🦠 Founder Variants" color={ACCENT3}>
                <ul className="small mb-0">
                  {(df.founder_variants || []).map((v, i) => <li key={i} className="mb-1">{v}</li>)}
                </ul>
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="👁️ Senior-Løken Syndrome" color={ACCENT5}>
                <div className="small p-2 rounded" style={{ background: ACCENT5 + '10', border: `1px solid ${ACCENT5}40` }}>
                  {df.senior_loken_syndrome}
                </div>
              </Section>

              <Section title="🔍 Differential Diagnosis" color={ACCENT7}>
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: ACCENT7 + '15' }}>
                    <tr><th>Condition</th><th>Key Differentiators</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(df.ddx_table || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT7, whiteSpace: 'nowrap' }}>{k.replace(/_/g, ' ')}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && df && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="📖 Disease Overview" color={ACCENT}>
                <table className="table table-sm table-bordered small mb-3">
                  <tbody>
                    <tr><td className="fw-bold">Disease</td><td>{df.disease}</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>{df.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>{df.omim_disease}</td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>{df.chromosome}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{df.inheritance}</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>{df.prevalence}</td></tr>
                  </tbody>
                </table>
                <div className="small p-2 rounded" style={{ background: ACCENT + '08', border: `1px solid ${ACCENT}30` }}>
                  <b>Mechanism:</b> {df.mechanism}
                </div>
              </Section>

              <Section title="🔬 Renal Histology (Biopsy Triad)" color={ACCENT}>
                {Object.entries(df.renal_histology || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="🏥 Diagnostic Criteria" color={ACCENT3}>
                {Object.entries(df.diagnostic_criteria || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="🩺 Key Clinical Features" color={ACCENT2}>
                {Object.entries(df.key_clinical_features || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="💊 Treatment" color={ACCENT4}>
                {Object.entries(df.treatment || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="📈 Prognosis" color={ACCENT2}>
                <div className="small p-2 rounded" style={{ background: ACCENT2 + '10', border: `1px solid ${ACCENT2}40` }}>
                  {df.prognosis}
                </div>
              </Section>

              <div className="text-muted small mt-2">{df.cohort_note}</div>
            </div>
          </div>
        </div>
      )}

      <div className="mt-3 pt-2 border-top">
        <Link href="/" className="text-muted small">← Back to Portal</Link>
      </div>
    </div>
  );
}
