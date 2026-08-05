'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'patients',    label: 'Per Patient' },
  { id: 'definitions', label: 'Definitions' },
];

const STATUS_COLOR = {
  confirmed:      '#22c55e',
  auto_coded:     '#6366f1',
  pending_review: '#f59e0b',
  rejected:       '#ef4444',
};

const STATUS_LABEL = {
  confirmed:      'Confirmed',
  auto_coded:     'Auto-Coded (AI)',
  pending_review: 'Pending Review',
  rejected:       'Rejected',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className={`h4 fw-bold mb-1 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ pct, color, max }) {
  const w = Math.min(100, Math.max(0, max ? (pct / max) * 100 : pct || 0));
  return (
    <div className="progress" style={{ height: 10, borderRadius: 6 }}>
      <div className="progress-bar" style={{ width: `${w}%`, backgroundColor: color || '#3b82f6', borderRadius: 6 }} />
    </div>
  );
}

function ConfBadge({ conf }) {
  const c = conf || 0;
  const cls = c >= 0.9 ? 'success' : c >= 0.7 ? 'info' : 'warning';
  return <span className={`badge bg-${cls}`}>{(c * 100).toFixed(0)}%</span>;
}

function StatusBadge({ status }) {
  const color = STATUS_COLOR[status] || '#6b7280';
  return (
    <span className="badge" style={{ backgroundColor: color, fontSize: '0.7rem' }}>
      {STATUS_LABEL[status] || status}
    </span>
  );
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;

  const kpis       = ov.kpis               || {};
  const statusDist = ov.status_distribution || [];
  const topCodes   = ov.top_codes           || [];
  const confTiers  = ov.confidence_tiers    || [];
  const coders     = ov.coder_breakdown     || [];
  const aiVsHuman  = ov.ai_vs_human         || [];
  const rejReasons = ov.rejection_reasons   || [];
  const trend      = ov.monthly_trend       || [];

  const statusTotal = statusDist.reduce((s, x) => s + x.value, 0);
  const maxCode     = Math.max(...topCodes.map(c => c.count), 1);
  const maxCoder    = Math.max(...coders.map(c => c.count), 1);

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Total Records"     value={kpis.total_records}                               color="primary"   sub="ICD-10 coding entries" />
        <KPI label="Patients"          value={kpis.total_patients}                               color="info"      sub="with coded encounters" />
        <KPI label="Avg Confidence"    value={`${((kpis.avg_confidence || 0) * 100).toFixed(1)}%`} color="success" sub="AI + human coding" />
        <KPI label="Unique Codes"      value={kpis.unique_codes}                                 color="secondary" sub="distinct ICD-10 codes" />
      </div>

      <div className="row mb-4">
        <KPI label="Confirmed Rate"    value={`${(kpis.confirmed_rate || 0).toFixed(1)}%`}       color="success"   sub="accepted by reviewer" />
        <KPI label="Rejection Rate"    value={`${(kpis.rejection_rate || 0).toFixed(1)}%`}       color="danger"    sub="rejected after review" />
        <KPI label="AI Auto-Coded"     value={kpis.auto_coded_count}                             color="primary"   sub="pending validation" />
        <KPI label="2° Coverage"       value={`${(kpis.secondary_coverage_pct || 0).toFixed(1)}%`} color="info"  sub="records with secondary codes" />
      </div>

      {/* Status Distribution + Top Codes */}
      <div className="row mb-4">
        <div className="col-12 col-md-5 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Status Distribution</div>
            <div className="card-body">
              {statusDist.map(s => (
                <div key={s.name} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span><StatusBadge status={s.name} /></span>
                    <span className="fw-semibold">{s.value} ({statusTotal ? ((s.value / statusTotal) * 100).toFixed(0) : 0}%)</span>
                  </div>
                  <Bar pct={s.value} color={STATUS_COLOR[s.name]} max={statusTotal} />
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-12 col-md-7 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Top ICD-10 Codes</div>
            <div className="card-body p-0">
              <div className="table-responsive" style={{ maxHeight: 260, overflowY: 'auto' }}>
                <table className="table table-sm mb-0">
                  <thead className="table-light sticky-top">
                    <tr><th>Code</th><th>Description</th><th className="text-center">Count</th></tr>
                  </thead>
                  <tbody>
                    {topCodes.map(c => (
                      <tr key={c.code}>
                        <td><code className="small">{c.code}</code></td>
                        <td className="small">{c.description}</td>
                        <td className="text-center">
                          <span className="badge bg-secondary">{c.count}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Confidence Tiers + AI vs Human */}
      <div className="row mb-4">
        <div className="col-12 col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Confidence Tiers</div>
            <div className="card-body">
              {confTiers.map(t => {
                const color = t.name.startsWith('High') ? '#22c55e' : t.name.startsWith('Medium') ? '#06b6d4' : '#f59e0b';
                const total = confTiers.reduce((s, x) => s + x.value, 0);
                return (
                  <div key={t.name} className="mb-3">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{t.name}</span>
                      <span className="fw-semibold">{t.value} ({total ? ((t.value / total) * 100).toFixed(0) : 0}%)</span>
                    </div>
                    <Bar pct={t.value} color={color} max={total} />
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="col-12 col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">AI vs Human Coding</div>
            <div className="card-body">
              {aiVsHuman.map(h => (
                <div key={h.type} className="mb-4 p-2 rounded" style={{ background: '#f8f9fa' }}>
                  <div className="fw-semibold small mb-2">{h.type} Coding</div>
                  <div className="d-flex justify-content-between small mb-1">
                    <span>Records:</span><span className="fw-bold">{h.count}</span>
                  </div>
                  <div className="d-flex justify-content-between small mb-1">
                    <span>Avg Confidence:</span><span><ConfBadge conf={h.avg_confidence} /></span>
                  </div>
                  {h.type === 'Human' && (
                    <>
                      <div className="d-flex justify-content-between small mb-1">
                        <span>Confirmed:</span><span className="text-success fw-bold">{h.confirmed}</span>
                      </div>
                      <div className="d-flex justify-content-between small">
                        <span>Rejected:</span><span className="text-danger fw-bold">{h.rejected}</span>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-12 col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Rejection Reasons</div>
            <div className="card-body">
              {rejReasons.map(r => (
                <div key={r.reason} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{r.reason.replace(/_/g, ' ')}</span>
                    <span className="badge bg-danger">{r.count}</span>
                  </div>
                  <Bar pct={r.count} color="#ef4444" max={Math.max(...rejReasons.map(x => x.count), 1)} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Coder Breakdown + Monthly Trend */}
      <div className="row mb-4">
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Coder Breakdown</div>
            <div className="card-body">
              {coders.map(c => (
                <div key={c.coder} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{c.coder}</span>
                    <span>{c.count} records · <ConfBadge conf={c.avg_confidence} /></span>
                  </div>
                  <Bar pct={c.count} color="#6366f1" max={maxCoder} />
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Monthly Coding Volume</div>
            <div className="card-body">
              {trend.length === 0 ? (
                <div className="text-muted">No trend data.</div>
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 100, minWidth: trend.length * 40 }}>
                    {trend.map((d, i) => {
                      const maxC = Math.max(...trend.map(x => x.count), 1);
                      const h = Math.round((d.count / maxC) * 100);
                      return (
                        <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <div title={`${d.month}: ${d.count} records`}
                            style={{ width: '100%', height: `${h}%`, backgroundColor: '#6366f1', borderRadius: '3px 3px 0 0', opacity: 0.85 }}
                          />
                          <div className="text-muted" style={{ fontSize: '0.65rem', marginTop: 2 }}>{d.month.slice(5)}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function PatientsPanel({ bk }) {
  const [sort, setSort] = useState('total_codes');
  const [dir,  setDir]  = useState(-1);

  if (!bk) return <div className="text-muted p-3">Loading…</div>;
  const patients = [...(bk.patient_summary || [])].sort((a, b) => dir * ((a[sort] || 0) - (b[sort] || 0)));

  const toggle = col => {
    if (sort === col) setDir(d => -d);
    else { setSort(col); setDir(-1); }
  };
  const Hdr = ({ col, label }) => (
    <th className="small" style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => toggle(col)}>
      {sort === col ? (dir === -1 ? '▼ ' : '▲ ') : ''}{label}
    </th>
  );

  return (
    <div>
      <div className="mb-2 text-muted small">{patients.length} patients — click column header to sort</div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-light">
            <tr>
              <th className="small">Patient</th>
              <Hdr col="total_codes"    label="Total Codes" />
              <Hdr col="unique_codes"   label="Unique Codes" />
              <Hdr col="avg_conf"       label="Avg Confidence" />
              <Hdr col="confirmed"      label="Confirmed" />
              <Hdr col="rejected"       label="Rejected" />
              <th className="small">Latest Encounter</th>
            </tr>
          </thead>
          <tbody>
            {patients.map(p => (
              <tr key={p.patient_id}>
                <td><span className="badge bg-secondary">{p.patient_id}</span></td>
                <td className="text-center">{p.total_codes}</td>
                <td className="text-center">{p.unique_codes}</td>
                <td><ConfBadge conf={p.avg_conf} /></td>
                <td className="text-center">
                  {p.confirmed > 0 ? <span className="badge bg-success">{p.confirmed}</span> : <span className="text-muted">0</span>}
                </td>
                <td className="text-center">
                  {p.rejected > 0 ? <span className="badge bg-danger">{p.rejected}</span> : <span className="text-muted">0</span>}
                </td>
                <td className="text-muted small">{p.latest_encounter}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsPanel({ def }) {
  if (!def) return <div className="text-muted p-3">Loading…</div>;

  const statuses   = def.statuses              || {};
  const confThresh = def.confidence_thresholds || {};
  const workflow   = def.coding_workflow       || [];
  const chapters   = def.icd10_chapters        || {};
  const metrics    = def.metrics               || {};

  return (
    <div>
      <div className="row mb-4">
        {/* Status Definitions */}
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Status Definitions</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Status</th><th>Definition</th></tr></thead>
                <tbody>
                  {Object.entries(statuses).map(([k, v]) => (
                    <tr key={k}>
                      <td><StatusBadge status={k} /></td>
                      <td className="small">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Confidence Thresholds */}
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Confidence Thresholds</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Level</th><th>Range</th><th>Action</th></tr></thead>
                <tbody>
                  {Object.entries(confThresh).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-semibold small text-capitalize">{k}</td>
                      <td><code className="small">{v.range}</code></td>
                      <td className="small">{v.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Coding Workflow */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">AI-Assisted Coding Workflow</div>
        <div className="card-body">
          <ol className="mb-0 ps-3">
            {workflow.map((step, i) => (
              <li key={i} className="small mb-1">{step.replace(/^\d+\.\s*/, '')}</li>
            ))}
          </ol>
        </div>
      </div>

      {/* ICD-10 Chapters + Metrics */}
      <div className="row">
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Relevant ICD-10 Chapters</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Code Prefix</th><th>Chapter</th></tr></thead>
                <tbody>
                  {Object.entries(chapters).map(([k, v]) => (
                    <tr key={k}>
                      <td><code className="small">{k}</code></td>
                      <td className="small">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Metric Definitions</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Metric</th><th>Definition</th></tr></thead>
                <tbody>
                  {Object.entries(metrics).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-semibold small">{k.replace(/_/g, ' ')}</td>
                      <td className="small">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ICD10CodingPage() {
  const [tab, setTab] = useState('overview');
  const [ov,  setOv]  = useState(null);
  const [bk,  setBk]  = useState(null);
  const [def, setDef] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/icd10-coding/overview`).then(r => r.json()),
      fetch(`${API}/api/icd10-coding/breakdown`).then(r => r.json()),
      fetch(`${API}/api/icd10-coding/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDef(d); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: '1.6rem' }}>&#x1f3f7;&#xfe0f;</span>
        <div>
          <h4 className="mb-0 fw-bold">ICD-10 AI Coding Dashboard</h4>
          <div className="text-muted small">
            {ov
              ? `${ov.kpis?.total_records} records · ${ov.kpis?.total_patients} patients · ${ov.kpis?.unique_codes} unique codes · avg confidence ${((ov.kpis?.avg_confidence || 0) * 100).toFixed(1)}%`
              : 'Loading…'}
          </div>
        </div>
      </div>

      {err && <div className="alert alert-danger">{err}</div>}

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel    ov={ov} />}
      {tab === 'patients'    && <PatientsPanel     bk={bk} />}
      {tab === 'definitions' && <DefinitionsPanel  def={def} />}
    </div>
  );
}
