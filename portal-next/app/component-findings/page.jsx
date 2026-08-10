'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const AGREE_COLOR = { agree: '#22c55e', partial: '#f59e0b', disagree: '#ef4444' };
const AGREE_BS   = { agree: 'success',  partial: 'warning',  disagree: 'danger'  };

const COMPONENT_ICON = {
  acquisition: '📡',
  artifacts: '🔧',
  background: '🌊',
  epileptiform: '⚡',
  explainability: '🔍',
  video: '🎥',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h4 mb-0 fw-bold" style={{ color }}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function AgreeBadge({ val }) {
  return (
    <span className={`badge bg-${AGREE_BS[val] || 'secondary'}`}>{val}</span>
  );
}

function AgreementBar({ agree, partial, disagree, total }) {
  if (!total) return null;
  const agPct     = (agree   / total) * 100;
  const parPct    = (partial / total) * 100;
  const disPct    = (disagree / total) * 100;
  return (
    <div className="progress" style={{ height: 16, borderRadius: 8 }}>
      <div className="progress-bar bg-success"  style={{ width: `${agPct}%`  }} title={`Agree ${agree}`} />
      <div className="progress-bar bg-warning"  style={{ width: `${parPct}%` }} title={`Partial ${partial}`} />
      <div className="progress-bar bg-danger"   style={{ width: `${disPct}%` }} title={`Disagree ${disagree}`} />
    </div>
  );
}

function OverviewTab({ ov }) {
  const kpi  = ov.kpis || {};
  const dist = ov.agreement_distribution || [];
  const comp = ov.component_agreement   || [];
  const rev  = ov.reviewer_agreement    || [];
  const trend = ov.monthly_trend        || [];

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Total Findings"  value={kpi.total_findings}   color="#3b82f6" />
        <KPI label="Patients"        value={kpi.total_patients}   color="#6366f1" />
        <KPI label="Reviewers"       value={kpi.total_reviewers}  color="#8b5cf6" />
        <KPI label="Components"      value={kpi.total_components} color="#f59e0b" />
        <KPI label="Agreement Rate"  value={`${kpi.agreement_rate}%`}     color="#22c55e" />
        <KPI label="Disagreement Rate" value={`${kpi.disagreement_rate}%`} color="#ef4444" />
      </div>

      <div className="row">
        {/* Agreement Distribution Donut */}
        <div className="col-md-4 mb-4">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2"><strong>Agreement Distribution</strong></div>
            <div className="card-body">
              {dist.map(d => {
                const pct = ((d.value / kpi.total_findings) * 100).toFixed(1);
                return (
                  <div key={d.name} className="mb-3">
                    <div className="d-flex justify-content-between mb-1">
                      <span className={`badge bg-${AGREE_BS[d.name.toLowerCase()] || 'secondary'}`}>{d.name}</span>
                      <small className="text-muted">{d.value} ({pct}%)</small>
                    </div>
                    <div className="progress" style={{ height: 12 }}>
                      <div
                        className={`progress-bar bg-${AGREE_BS[d.name.toLowerCase()] || 'secondary'}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Component breakdown */}
        <div className="col-md-8 mb-4">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2"><strong>Agreement by EEG Component</strong></div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <thead><tr>
                  <th>Component</th>
                  <th className="text-end">Total</th>
                  <th className="text-end text-success">Agree</th>
                  <th className="text-end text-warning">Partial</th>
                  <th className="text-end text-danger">Disagree</th>
                  <th style={{ minWidth: 120 }}>Rate</th>
                </tr></thead>
                <tbody>
                  {comp.map(c => (
                    <tr key={c.component}>
                      <td>{COMPONENT_ICON[c.component] || '🔬'} {c.component}</td>
                      <td className="text-end">{c.total}</td>
                      <td className="text-end text-success">{c.agree}</td>
                      <td className="text-end text-warning">{c.partial}</td>
                      <td className="text-end text-danger">{c.disagree}</td>
                      <td>
                        <AgreementBar agree={c.agree} partial={c.partial} disagree={c.disagree} total={c.total} />
                        <small className="text-muted">{c.agree_pct}%</small>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Monthly Trend */}
      {trend.length > 0 && (
        <div className="card shadow-sm mb-4">
          <div className="card-header py-2"><strong>Monthly Trend</strong></div>
          <div className="card-body">
            <table className="table table-sm mb-0">
              <thead><tr>
                <th>Month</th>
                <th className="text-end text-success">Agree</th>
                <th className="text-end text-warning">Partial</th>
                <th className="text-end text-danger">Disagree</th>
                <th style={{ minWidth: 180 }}>Distribution</th>
              </tr></thead>
              <tbody>
                {trend.map(m => {
                  const tot = m.agree + m.disagree + m.partial;
                  return (
                    <tr key={m.month}>
                      <td>{m.month}</td>
                      <td className="text-end text-success">{m.agree}</td>
                      <td className="text-end text-warning">{m.partial}</td>
                      <td className="text-end text-danger">{m.disagree}</td>
                      <td><AgreementBar agree={m.agree} partial={m.partial} disagree={m.disagree} total={tot} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Reviewer leaderboard */}
      <div className="card shadow-sm">
        <div className="card-header py-2"><strong>Reviewer Agreement Rates</strong></div>
        <div className="card-body">
          <div className="row">
            {rev.map(r => (
              <div key={r.reviewer} className="col-6 col-md-3 mb-3">
                <div className="card border-0 shadow-sm h-100">
                  <div className="card-body text-center py-2">
                    <div className="h5 fw-bold text-primary">{r.agree_pct}%</div>
                    <div className="small fw-bold">{r.reviewer}</div>
                    <div className="text-muted" style={{ fontSize: '0.75rem' }}>{r.total} reviews</div>
                    <div className="mt-1">
                      <AgreementBar agree={r.agree} partial={r.partial} disagree={r.disagree} total={r.total} />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ByComponentTab({ ov }) {
  const comp = ov.component_agreement || [];
  const [sel, setSel] = useState(comp[0]?.component || '');

  const selData = comp.find(c => c.component === sel);

  return (
    <div>
      <div className="mb-3 d-flex flex-wrap gap-2">
        {comp.map(c => (
          <button
            key={c.component}
            className={`btn btn-sm ${sel === c.component ? 'btn-primary' : 'btn-outline-secondary'}`}
            onClick={() => setSel(c.component)}
          >
            {COMPONENT_ICON[c.component] || ''} {c.component}
          </button>
        ))}
      </div>

      {selData && (
        <div className="card shadow-sm">
          <div className="card-header py-2">
            <strong>{COMPONENT_ICON[selData.component] || '🔬'} {selData.component.charAt(0).toUpperCase() + selData.component.slice(1)}</strong>
            <span className="ms-2 text-muted small">— {selData.total} reviews</span>
          </div>
          <div className="card-body">
            <div className="row mb-3">
              <div className="col-4 text-center">
                <div className="h4 text-success fw-bold">{selData.agree}</div>
                <div className="text-muted small">Agree</div>
              </div>
              <div className="col-4 text-center">
                <div className="h4 text-warning fw-bold">{selData.partial}</div>
                <div className="text-muted small">Partial</div>
              </div>
              <div className="col-4 text-center">
                <div className="h4 text-danger fw-bold">{selData.disagree}</div>
                <div className="text-muted small">Disagree</div>
              </div>
            </div>
            <AgreementBar agree={selData.agree} partial={selData.partial} disagree={selData.disagree} total={selData.total} />
            <div className="mt-2 text-muted small text-end">{selData.agree_pct}% agreement rate</div>
          </div>
        </div>
      )}

      <div className="mt-4">
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead className="table-dark"><tr>
              <th>Component</th>
              <th className="text-end">Total</th>
              <th className="text-end">Agree</th>
              <th className="text-end">Partial</th>
              <th className="text-end">Disagree</th>
              <th className="text-end">Agreement %</th>
              <th>Bar</th>
            </tr></thead>
            <tbody>
              {comp.sort((a, b) => b.agree_pct - a.agree_pct).map(c => (
                <tr key={c.component} className={sel === c.component ? 'table-active' : ''}>
                  <td className="fw-bold">{COMPONENT_ICON[c.component] || ''} {c.component}</td>
                  <td className="text-end">{c.total}</td>
                  <td className="text-end text-success">{c.agree}</td>
                  <td className="text-end text-warning">{c.partial}</td>
                  <td className="text-end text-danger">{c.disagree}</td>
                  <td className="text-end fw-bold">{c.agree_pct}%</td>
                  <td style={{ minWidth: 120 }}>
                    <AgreementBar agree={c.agree} partial={c.partial} disagree={c.disagree} total={c.total} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ByReviewerTab({ ov }) {
  const rev = ov.reviewer_agreement || [];
  const sorted = [...rev].sort((a, b) => b.agree_pct - a.agree_pct);

  return (
    <div>
      <div className="row mb-4">
        {sorted.map((r, i) => (
          <div key={r.reviewer} className="col-6 col-md-3 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header py-1 d-flex align-items-center gap-2">
                <span className={`badge bg-${i === 0 ? 'warning text-dark' : 'secondary'}`}>#{i + 1}</span>
                <small className="fw-bold">{r.reviewer}</small>
              </div>
              <div className="card-body text-center py-2">
                <div className="h4 fw-bold text-primary">{r.agree_pct}%</div>
                <div className="text-muted small">agreement</div>
                <div className="small mt-2">
                  <span className="text-success me-1">{r.agree}✓</span>
                  <span className="text-warning me-1">{r.partial}~</span>
                  <span className="text-danger">{r.disagree}✗</span>
                </div>
                <div className="mt-1">
                  <AgreementBar agree={r.agree} partial={r.partial} disagree={r.disagree} total={r.total} />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card shadow-sm">
        <div className="card-header py-2"><strong>Full Reviewer Table</strong></div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-dark"><tr>
                <th>Reviewer</th>
                <th className="text-end">Total</th>
                <th className="text-end text-success">Agree</th>
                <th className="text-end text-warning">Partial</th>
                <th className="text-end text-danger">Disagree</th>
                <th className="text-end">Agreement %</th>
                <th>Distribution</th>
              </tr></thead>
              <tbody>
                {sorted.map(r => (
                  <tr key={r.reviewer}>
                    <td className="fw-bold">👨‍⚕️ {r.reviewer}</td>
                    <td className="text-end">{r.total}</td>
                    <td className="text-end text-success">{r.agree}</td>
                    <td className="text-end text-warning">{r.partial}</td>
                    <td className="text-end text-danger">{r.disagree}</td>
                    <td className="text-end fw-bold">{r.agree_pct}%</td>
                    <td style={{ minWidth: 140 }}>
                      <AgreementBar agree={r.agree} partial={r.partial} disagree={r.disagree} total={r.total} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function AllFindingsTab({ bd }) {
  const [search, setSearch]   = useState('');
  const [compFilter, setComp] = useState('');
  const [agreeFilter, setAgree] = useState('');
  const all = bd.all_findings || [];

  const filtered = all.filter(f => {
    const s = search.toLowerCase();
    const matchText = !s || f.doctor_finding?.toLowerCase().includes(s) ||
                      f.patient_id?.toLowerCase().includes(s) ||
                      f.doctor?.toLowerCase().includes(s);
    const matchComp  = !compFilter  || f.component  === compFilter;
    const matchAgree = !agreeFilter || f.agree_with_ai === agreeFilter;
    return matchText && matchComp && matchAgree;
  });

  const components = [...new Set(all.map(f => f.component))].sort();

  return (
    <div>
      <div className="row mb-3 g-2">
        <div className="col-md-4">
          <input
            className="form-control form-control-sm"
            placeholder="Search findings, patient, doctor…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
        <div className="col-md-3">
          <select className="form-select form-select-sm" value={compFilter} onChange={e => setComp(e.target.value)}>
            <option value="">All Components</option>
            {components.map(c => <option key={c} value={c}>{COMPONENT_ICON[c] || ''} {c}</option>)}
          </select>
        </div>
        <div className="col-md-3">
          <select className="form-select form-select-sm" value={agreeFilter} onChange={e => setAgree(e.target.value)}>
            <option value="">All Verdicts</option>
            <option value="agree">Agree</option>
            <option value="partial">Partial</option>
            <option value="disagree">Disagree</option>
          </select>
        </div>
        <div className="col-md-2 text-muted small d-flex align-items-center">
          {filtered.length} / {all.length} shown
        </div>
      </div>

      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-dark"><tr>
            <th>Patient</th>
            <th>Component</th>
            <th>Doctor Finding</th>
            <th>Reviewer</th>
            <th>Verdict</th>
            <th>Date</th>
          </tr></thead>
          <tbody>
            {filtered.slice(0, 100).map(f => (
              <tr key={f.id}>
                <td><code>{f.patient_id}</code></td>
                <td><span className="badge bg-secondary">{COMPONENT_ICON[f.component] || ''} {f.component}</span></td>
                <td style={{ maxWidth: 300 }} className="text-wrap small">{f.doctor_finding}</td>
                <td className="small">{f.doctor}</td>
                <td><AgreeBadge val={f.agree_with_ai} /></td>
                <td className="text-muted small">{f.created_at?.slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {filtered.length > 100 && (
        <div className="text-muted small text-center">Showing first 100 of {filtered.length} results. Refine search to narrow.</div>
      )}
    </div>
  );
}

function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2"><strong>{defs.title}</strong></div>
        <div className="card-body">
          <p>{defs.description}</p>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6 mb-4">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2"><strong>EEG Components</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <tbody>
                  {Object.entries(defs.components || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-nowrap">{COMPONENT_ICON[k] || ''} {k}</td>
                      <td className="small">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-4">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2"><strong>Agreement Levels</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <tbody>
                  {Object.entries(defs.agreement_levels || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td><AgreeBadge val={k} /></td>
                      <td className="small">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {(defs.clinical_relevance || []).length > 0 && (
        <div className="card shadow-sm mb-4">
          <div className="card-header py-2"><strong>Clinical Relevance</strong></div>
          <div className="card-body">
            <ul className="mb-0">
              {defs.clinical_relevance.map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        </div>
      )}

      <div className="card shadow-sm">
        <div className="card-header py-2"><strong>Glossary</strong></div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <tbody>
              {Object.entries(defs.glossary || {}).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-bold text-nowrap"><code>{k}</code></td>
                  <td className="small">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

const TABS = [
  { id: 'overview',    label: '📊 Overview' },
  { id: 'component',  label: '🔬 By Component' },
  { id: 'reviewer',   label: '👨‍⚕️ By Reviewer' },
  { id: 'findings',   label: '📋 All Findings' },
  { id: 'definitions', label: '📖 Definitions' },
];

export default function ComponentFindingsDashboard() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/component-findings/overview`).then(r => r.json()),
      fetch(`${API}/api/component-findings/breakdown`).then(r => r.json()),
      fetch(`${API}/api/component-findings/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err)  return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-3">Loading component findings data…</div>;

  const kpi = ov.kpis || {};

  return (
    <div className="p-3">
      <h3>🔬 EEG Component Findings — Doctor-AI Agreement</h3>
      <p className="text-muted">
        {kpi.total_findings} findings · {kpi.total_patients} patients · {kpi.total_reviewers} reviewers ·{' '}
        <strong className="text-success">{kpi.agreement_rate}%</strong> overall agreement
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewTab     ov={ov} />}
      {tab === 'component'   && <ByComponentTab   ov={ov} />}
      {tab === 'reviewer'    && <ByReviewerTab     ov={ov} />}
      {tab === 'findings'    && <AllFindingsTab    bd={bd} />}
      {tab === 'definitions' && <DefinitionsTab   defs={defs} />}
    </div>
  );
}
