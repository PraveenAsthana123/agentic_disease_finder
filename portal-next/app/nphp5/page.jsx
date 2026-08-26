'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & Subtype', 'Definitions'];

// NPHP5 colour scheme — teal-indigo-amber-red (LCA-like retinal; severe visual; renal; SLS5 most common)
const ACCENT  = '#00695c';   // deep teal — IQCB1/NPHP5 / most common SLS gene
const ACCENT2 = '#1b5e20';   // deep green — transplant curative / excellent renal outcome
const ACCENT3 = '#1a237e';   // dark indigo — genetics / RPGR interaction / NPHP-RC module
const ACCENT4 = '#e65100';   // deep orange — ESRD / renal / age at renal dx
const ACCENT5 = '#004d40';   // very dark teal — renal/tubular component
const ACCENT6 = '#37474f';   // dark slate — epidemiology / AR / no situs / no CHF
const ACCENT7 = '#bf360c';   // deep burnt orange — LCA misdiagnosis (most critical error)
const ACCENT8 = '#4a148c';   // dark violet — visual impairment / nystagmus / LCA-like

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

export default function Nphp5Page() {
  const [tab, setTab] = useState(0);
  const [ov, setOv]   = useState(null);
  const [bk, setBk]   = useState(null);
  const [df, setDf]   = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/nphp5/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp5/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp5/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="container py-4 text-danger">Error: {err}</div>;
  if (!ov)  return <div className="container py-4 text-muted">Loading NPHP5 dashboard…</div>;

  const kpis = ov.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          🧬 Nephronophthisis Type 5 / Senior-Løken Syndrome 5 (IQCB1/NPHP5 — Most Common SLS Gene)
        </h4>
        <div className="text-muted small">
          IQCB1 · 3q21.1 · 590 aa IQ motif-containing B1 (NPHP5) · RPGR interaction · Photoreceptor CC & Renal TZ · AR Biallelic LOF · OMIM #609254
          &nbsp;|&nbsp; Cohort: {_COHORT_SIZE} patients (seed-349) · 3 endpoints verified
        </div>
        <div className="mt-1">
          <Badge text="MOST COMMON Senior-Løken Gene" color={ACCENT} />
          <Badge text="Severe LCA-like Retinal" color={ACCENT8} />
          <Badge text="Retinal >> Renal" color={ACCENT8} />
          <Badge text="ESRD Median ~13yr" color={ACCENT4} />
          <Badge text="Nystagmus ~75-80%" color={ACCENT8} />
          <Badge text="No Situs Inversus" color={ACCENT6} />
          <Badge text="No CHF" color={ACCENT6} />
          <Badge text="Transplant EXCELLENT" color={ACCENT2} />
          <Badge text="AR · ~1/100,000–1/500,000" color={ACCENT6} />
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
            <KPI label="Median Retinal Dx (yr)" value={kpis.median_age_retinal_dx_yr ?? '—'} color={ACCENT8} />
            <KPI label="Median Renal Dx (yr)" value={kpis.median_age_renal_dx_yr ?? '—'} color={ACCENT4} />
            <KPI label="Median GFR at Dx" value={kpis.median_gfr_at_dx_ml_min ? `${kpis.median_gfr_at_dx_ml_min} mL/min` : '—'} color={ACCENT4} />
            <KPI label="Median U-Osm" value={kpis.median_urine_osmolality ? `${kpis.median_urine_osmolality} mosm/kg` : '—'} color={ACCENT5} />
            <KPI label="% ESRD/RRT" value={kpis.pct_esrd_or_rrt ? `${kpis.pct_esrd_or_rrt}%` : '—'} color={ACCENT4} />
          </div>
          <div className="row mb-3">
            <KPI label="% Transplanted" value={kpis.pct_transplanted ? `${kpis.pct_transplanted}%` : '—'} color={ACCENT2} />
            <KPI label="% Senior-Løken (SLS)" value={kpis.pct_senior_loken ? `${kpis.pct_senior_loken}%` : '—'} color={ACCENT} />
            <KPI label="% Nystagmus" value={kpis.pct_nystagmus ? `${kpis.pct_nystagmus}%` : '—'} color={ACCENT8} />
            <KPI label="% Legally Blind" value={kpis.pct_legally_blind_or_worse ? `${kpis.pct_legally_blind_or_worse}%` : '—'} color={ACCENT8} />
            <KPI label="% LCA Misdiagnosis" value={kpis.pct_lca_misdiagnosis ? `${kpis.pct_lca_misdiagnosis}%` : '—'} color={ACCENT7} />
            <KPI label="% Consanguineous" value={kpis.pct_consanguineous ? `${kpis.pct_consanguineous}%` : '—'} color={ACCENT6} />
          </div>

          {/* Alerts */}
          <Section title="⚠️ Clinical Alerts" color={ACCENT4}>
            {ov.alerts && Object.entries(ov.alerts).map(([k, v]) => (
              <Alert key={k} color={
                k === 'most_common_sls_gene' ? ACCENT :
                k === 'retinal_dominates_clinically' ? ACCENT8 :
                k === 'lca_misdiagnosis_most_common' ? ACCENT7 :
                k === 'transplant_curative_retinal_not' ? ACCENT2 : ACCENT6
              }>
                <span className="fw-bold">{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}:</span>{' '}{v}
              </Alert>
            ))}
          </Section>

          {/* Key Facts */}
          <Section title="🔬 Key Facts — NPHP5/IQCB1 Clinical & Molecular" color={ACCENT}>
            <ul className="mb-0 small">
              {(ov.key_facts || []).map((f, i) => <li key={i} className="mb-1">{f}</li>)}
            </ul>
          </Section>

          {/* NPHP Comparison Table */}
          <Section title="🫘 NPHP1 / NPHP4 / NPHP5 / CEP290 — Key Differences (Senior-Løken Focus)" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small mb-0">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr>
                    <th>Feature</th>
                    <th style={{ color: ACCENT3 }}>NPHP1 (Juvenile)</th>
                    <th style={{ color: '#6a1b9a' }}>NPHP4/SLS4</th>
                    <th style={{ color: ACCENT }}>NPHP5/SLS5 — THIS DISEASE</th>
                    <th style={{ color: ACCENT4 }}>CEP290/NPHP6/LCA10</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td className="fw-bold">SLS / Retinal</td>
                    <td>10–15% mild SLS</td>
                    <td style={{ color: '#6a1b9a' }}>~15–20% SLS4 (rod-cone)</td>
                    <td style={{ color: ACCENT }}>≥90% SLS5 — DOMINANT; LCA-like</td>
                    <td style={{ color: ACCENT4 }}>LCA10 (IVS26) — severe</td></tr>
                  <tr><td className="fw-bold">Retinal severity</td>
                    <td>Mild (reduced ERG)</td>
                    <td>Moderate rod-cone</td>
                    <td style={{ color: ACCENT8 }}>SEVERE LCA-like; flat ERG; nystagmus</td>
                    <td style={{ color: ACCENT4 }}>Severe LCA10 or Joubert</td></tr>
                  <tr><td className="fw-bold">Retinal vs Renal timing</td>
                    <td>Renal often first</td>
                    <td>Renal often first</td>
                    <td style={{ color: ACCENT }}>RETINAL FIRST (precedes renal)</td>
                    <td>Retinal first (LCA10)</td></tr>
                  <tr><td className="fw-bold">Nystagmus</td>
                    <td>Rare</td>
                    <td>OMA/nystagmus ~20%</td>
                    <td style={{ color: ACCENT8 }}>~75–80% (sensory nystagmus)</td>
                    <td>Common (LCA10)</td></tr>
                  <tr><td className="fw-bold">ESRD median</td>
                    <td>13 yr</td>
                    <td>17–20 yr</td>
                    <td style={{ color: ACCENT4 }}>~13 yr (similar to NPHP1)</td>
                    <td>Variable; Joubert may survive longer</td></tr>
                  <tr><td className="fw-bold">Molar Tooth Sign (MRI)</td>
                    <td>ABSENT</td>
                    <td>ABSENT</td>
                    <td style={{ color: ACCENT6 }}>ABSENT</td>
                    <td style={{ color: ACCENT4 }}>PRESENT (Joubert); absent LCA10</td></tr>
                  <tr><td className="fw-bold">Most common misdiagnosis</td>
                    <td>ADPKD</td>
                    <td>LCA / ADPKD</td>
                    <td style={{ color: ACCENT7 }}>LCA (omit renal workup)</td>
                    <td>LCA10 / Joubert</td></tr>
                  <tr><td className="fw-bold">Key panel addition</td>
                    <td>MLPA for 2q13 del</td>
                    <td>WES + ciliopathy panel</td>
                    <td style={{ color: ACCENT }}>ADD IQCB1 to ALL LCA panels</td>
                    <td>IVS26 specific allele; WES</td></tr>
                  <tr><td className="fw-bold">RPGR interaction</td>
                    <td>None direct</td>
                    <td>None direct</td>
                    <td style={{ color: ACCENT }}>DIRECT RPGR binding at CC</td>
                    <td>CEP290 at CC; different complex</td></tr>
                  <tr><td className="fw-bold">Transplant</td>
                    <td style={{ color: ACCENT2 }}>EXCELLENT</td>
                    <td style={{ color: ACCENT2 }}>EXCELLENT</td>
                    <td style={{ color: ACCENT2 }}>EXCELLENT (renal); retinal unchanged</td>
                    <td style={{ color: ACCENT2 }}>EXCELLENT (renal); retinal unchanged</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient Sample */}
          <Section title={`👥 Sample Patients (first 8 of ${_COHORT_SIZE})`} color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-hover small mb-0">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr>
                    <th>ID</th><th>Gene/Allele</th><th>Ethnicity</th>
                    <th>Retinal Dx</th><th>Renal Dx</th><th>GFR Now</th>
                    <th>U-Osm</th><th>Visual Acuity</th><th>Nystagmus</th><th>RRT/Tx</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.patients || []).map(p => (
                    <tr key={p.id}>
                      <td><code style={{ fontSize: '0.75em' }}>{p.id}</code></td>
                      <td style={{ maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.gene}>{p.gene?.split('—')?.slice(-1)[0]?.trim()?.split('/')[0]?.trim()}</td>
                      <td>{p.ethnicity?.split('(')[0]?.trim()}</td>
                      <td style={{ color: ACCENT8 }}>{p.age_retinal_dx_yr} yr</td>
                      <td style={{ color: ACCENT4 }}>{p.age_renal_dx_yr} yr</td>
                      <td style={{ color: p.gfr_now_ml_min < 15 ? ACCENT4 : p.gfr_now_ml_min < 30 ? ACCENT7 : ACCENT2 }}>
                        {p.gfr_now_ml_min}
                      </td>
                      <td style={{ color: p.urine_osmolality_mosm < 150 ? ACCENT4 : ACCENT5 }}>
                        {p.urine_osmolality_mosm}
                      </td>
                      <td style={{ color: p.visual_acuity?.includes('Light') || p.visual_acuity?.includes('CF') ? ACCENT8 : ACCENT6,
                                   maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.visual_acuity}>
                        {p.visual_acuity?.split('(')[0]?.trim()}
                      </td>
                      <td style={{ color: p.ocular_motor?.includes('Nystagmus') ? ACCENT8 : ACCENT6 }}>
                        {p.ocular_motor?.includes('Nystagmus') ? '👁 Nys' : '—'}
                      </td>
                      <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.rrt_or_transplant}>{p.rrt_or_transplant?.split('—')[0]?.trim()?.split('(')[0]?.trim()}</td>
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
              <Section title="🫘 Kidney Size Distribution" color={ACCENT5}>
                {Object.entries(bk.kidney_size_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
                ))}
              </Section>

              <Section title="👁 Retinal Involvement (LCA-like / SLS5)" color={ACCENT8}>
                {Object.entries(bk.retinal_involvement || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                ))}
              </Section>

              <Section title="👁 Ocular Motor / Nystagmus" color={ACCENT8}>
                {Object.entries(bk.ocular_motor_nystagmus || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                ))}
              </Section>

              <Section title="🔍 Visual Acuity Distribution" color={ACCENT3}>
                {Object.entries(bk.visual_acuity_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>

              <Section title="📊 CKD Stage (Current Visit)" color={ACCENT4}>
                {Object.entries(bk.ckd_stage_current || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="⏱️ Age at Retinal Dx Tiers" color={ACCENT8}>
                {Object.entries(bk.age_at_retinal_dx_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                ))}
              </Section>

              <Section title="⏱️ Age at Renal Dx Tiers" color={ACCENT4}>
                {Object.entries(bk.age_at_renal_dx_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>

              <Section title="💧 Urine Osmolality Tiers (mosm/kg)" color={ACCENT5}>
                {Object.entries(bk.urine_osmolality_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
                ))}
              </Section>

              <Section title="💊 RRT / Transplant Status" color={ACCENT2}>
                {Object.entries(bk.rrt_transplant_status || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
                ))}
              </Section>

              <Section title="📉 GFR Slope (ml/min/yr)" color={ACCENT3}>
                {Object.entries(bk.gfr_slope_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>

              <Section title="⚠️ Prior Misdiagnosis" color={ACCENT7}>
                {Object.entries(bk.prior_misdiagnosis || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
                ))}
              </Section>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 IQCB1 Allele Distribution" color={ACCENT3}>
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
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: Genetics & Subtype ── */}
      {tab === 2 && df && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 Genetic Architecture (IQCB1/3q21.1)" color={ACCENT3}>
                {Object.entries(df.genetic_architecture || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="🔍 Founder Variants" color={ACCENT3}>
                <ul className="small mb-0">
                  {(df.founder_variants || []).map((v, i) => <li key={i} className="mb-1">{v}</li>)}
                </ul>
              </Section>

              <Section title="🔗 RPGR Interaction Note" color={ACCENT}>
                <div className="small p-2 rounded" style={{ background: ACCENT + '10', border: `1px solid ${ACCENT}40` }}>
                  {df.rpgr_interaction_note}
                </div>
              </Section>

              <Section title="🦠 NPHP Subtype Comparison" color={ACCENT}>
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: ACCENT + '15' }}>
                    <tr><th>Subtype / Gene</th><th>Key Distinguishing Feature</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(df.nphp_comparison || {}).map(([k, v]) => (
                      <tr key={k} style={k.includes('NPHP5') ? { background: ACCENT + '15' } : {}}>
                        <td className="fw-bold" style={{ color: k.includes('NPHP5') ? ACCENT : ACCENT6, whiteSpace: 'nowrap' }}>{k}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>

            <div className="col-md-6">
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
