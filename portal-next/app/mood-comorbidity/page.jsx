'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Per-Patient' },
  { id: 'definitions', label: 'Definitions' },
];

const SEVERITY_COLOR = { severe: 'danger', moderate: 'warning', mild: 'info', minimal: 'success' };
const INTERP_COLOR = {
  Severe: 'danger', 'Moderately Severe': 'danger', Moderate: 'warning', Mild: 'info', Minimal: 'success',
  'Likely major depression (>15)': 'danger', 'Unlikely major depression': 'success',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function SeverityBadge({ severity }) {
  const color = SEVERITY_COLOR[severity?.toLowerCase()] || 'secondary';
  return <span className={`badge bg-${color}`}>{severity}</span>;
}

function InterpBadge({ interp }) {
  const color = INTERP_COLOR[interp] || 'secondary';
  return <span className={`badge bg-${color}`}>{interp || '—'}</span>;
}

function BarChart({ data, labelKey, valueKey, color = 'primary', title }) {
  if (!data || !data.length) return null;
  const max = Math.max(...data.map(d => d[valueKey]));
  return (
    <div className="mb-4">
      {title && <div className="fw-semibold mb-2 small text-muted">{title}</div>}
      {data.map((d, i) => (
        <div key={i} className="mb-2">
          <div className="d-flex justify-content-between small mb-1">
            <span>{d[labelKey]}</span>
            <span className="fw-semibold">{d[valueKey]}</span>
          </div>
          <div className="progress" style={{ height: 10 }}>
            <div
              className={`progress-bar bg-${color}`}
              style={{ width: `${max > 0 ? (d[valueKey] / max) * 100 : 0}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const k = data.kpis || {};
  return (
    <div>
      <div className="row mb-3">
        <KPI label="Screened" value={k.n_screened} color="info" sub="patients" />
        <KPI label="With Comorbidity" value={`${k.comorbidity_rate_pct}%`} color={k.comorbidity_rate_pct >= 60 ? 'danger' : 'warning'} sub={`${k.n_with_comorbidity} patients`} />
        <KPI label="Avg Behavioral Risk" value={k.avg_behavioral_risk_score} color={k.avg_behavioral_risk_score >= 60 ? 'danger' : k.avg_behavioral_risk_score >= 40 ? 'warning' : 'success'} sub="score / 100" />
        <KPI label="Treatment Resistant" value={k.treatment_resistant_n} color="danger" sub="patients" />
      </div>
      <div className="row mb-3">
        <KPI label="PHQ-9 Avg" value={k.phq9_avg} color={k.phq9_avg >= 15 ? 'danger' : k.phq9_avg >= 10 ? 'warning' : 'success'} sub="depression score" />
        <KPI label="GAD-7 Avg" value={k.gad7_avg} color={k.gad7_avg >= 15 ? 'danger' : k.gad7_avg >= 10 ? 'warning' : 'success'} sub="anxiety score" />
        <KPI label="NDDI-E Positive" value={k.nddie_depression_positive} color="danger" sub="likely depression" />
      </div>

      <div className="row">
        <div className="col-md-6">
          <div className="card mb-3">
            <div className="card-header fw-semibold">Top Comorbid Conditions</div>
            <div className="card-body">
              <BarChart data={data.top_conditions} labelKey="condition" valueKey="count" color="danger" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card mb-3">
            <div className="card-header fw-semibold">Behavioral Risk Severity</div>
            <div className="card-body">
              {(data.severity_distribution || []).map((s, i) => (
                <div key={i} className="d-flex justify-content-between align-items-center mb-2">
                  <SeverityBadge severity={s.severity} />
                  <span className="fw-semibold">{s.count} patients</span>
                </div>
              ))}
            </div>
          </div>
          <div className="card mb-3">
            <div className="card-header fw-semibold">Treatment Status</div>
            <div className="card-body">
              <BarChart data={data.treatment_status_distribution} labelKey="status" valueKey="count" color="warning" />
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <div className="card mb-3">
            <div className="card-header fw-semibold">PHQ-9 Depression Severity</div>
            <div className="card-body">
              <BarChart data={data.phq9_distribution} labelKey="category" valueKey="count" color="primary" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card mb-3">
            <div className="card-header fw-semibold">GAD-7 Anxiety Severity</div>
            <div className="card-body">
              <BarChart data={data.gad7_distribution} labelKey="category" valueKey="count" color="info" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const [search, setSearch] = useState('');
  const patients = (data.per_patient || []).filter(p =>
    !search || p.patient_id.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card mb-3">
            <div className="card-header fw-semibold">Condition Co-occurrence (Top 6)</div>
            <div className="card-body">
              {(data.condition_cooccurrence || []).length === 0 && (
                <div className="text-muted small">No co-occurring conditions.</div>
              )}
              {(data.condition_cooccurrence || []).map((c, i) => (
                <div key={i} className="d-flex justify-content-between align-items-start mb-2 small">
                  <div>
                    <span className="badge bg-secondary me-1">{c.condition_a}</span>
                    <span className="text-muted">+</span>
                    <span className="badge bg-secondary ms-1">{c.condition_b}</span>
                  </div>
                  <span className="badge bg-danger ms-2">{c.count} pts</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card mb-3">
            <div className="card-header fw-semibold">PHQ-9 vs GAD-7 Scores</div>
            <div className="card-body">
              <div className="table-responsive" style={{ maxHeight: 200, overflowY: 'auto' }}>
                <table className="table table-sm table-striped mb-0" style={{ fontSize: '0.78rem' }}>
                  <thead><tr><th>Patient</th><th>PHQ-9</th><th>GAD-7</th><th>Risk</th></tr></thead>
                  <tbody>
                    {(data.phq9_vs_gad7_scatter || []).map((s, i) => (
                      <tr key={i}>
                        <td>{s.patient_id}</td>
                        <td><span className={`badge bg-${s.phq9 >= 15 ? 'danger' : s.phq9 >= 10 ? 'warning' : 'success'}`}>{s.phq9}</span></td>
                        <td><span className={`badge bg-${s.gad7 >= 15 ? 'danger' : s.gad7 >= 10 ? 'warning' : 'success'}`}>{s.gad7}</span></td>
                        <td><SeverityBadge severity={s.risk_severity} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header fw-semibold d-flex justify-content-between align-items-center">
          Per-Patient Comorbidity Table
          <input
            className="form-control form-control-sm w-auto"
            placeholder="Filter patient…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{ width: 180 }}
          />
        </div>
        <div className="card-body p-0">
          <div className="table-responsive" style={{ maxHeight: 480, overflowY: 'auto' }}>
            <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.78rem' }}>
              <thead className="table-light sticky-top">
                <tr>
                  <th>Patient</th>
                  <th>Risk Score</th>
                  <th>Severity</th>
                  <th>PHQ-9</th>
                  <th>GAD-7</th>
                  <th>NDDI-E</th>
                  <th>Conditions</th>
                  <th>Treatment</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{p.patient_id}</td>
                    <td>
                      <div className="d-flex align-items-center gap-1">
                        <div className="progress flex-grow-1" style={{ height: 8, minWidth: 60 }}>
                          <div
                            className={`progress-bar bg-${p.behavioral_risk_score >= 60 ? 'danger' : p.behavioral_risk_score >= 40 ? 'warning' : 'success'}`}
                            style={{ width: `${p.behavioral_risk_score || 0}%` }}
                          />
                        </div>
                        <span style={{ minWidth: 30 }}>{p.behavioral_risk_score?.toFixed(0)}</span>
                      </div>
                    </td>
                    <td><SeverityBadge severity={p.risk_severity} /></td>
                    <td>
                      {p.phq9_score != null ? (
                        <><span className="fw-semibold">{p.phq9_score}</span> <InterpBadge interp={p.phq9_interpretation} /></>
                      ) : '—'}
                    </td>
                    <td>
                      {p.gad7_score != null ? (
                        <><span className="fw-semibold">{p.gad7_score}</span> <InterpBadge interp={p.gad7_interpretation} /></>
                      ) : '—'}
                    </td>
                    <td>
                      {p.nddie_score != null ? (
                        <span className="fw-semibold">{p.nddie_score}</span>
                      ) : '—'}
                    </td>
                    <td>
                      {(p.comorbidities || []).length === 0
                        ? <span className="text-muted">None</span>
                        : p.comorbidities.map((c, j) => (
                          <span key={j} className="badge bg-secondary me-1 mb-1" style={{ fontSize: '0.65rem' }}>{c}</span>
                        ))
                      }
                    </td>
                    <td>
                      <span className={`badge bg-${p.treatment_status === 'treatment_resistant' ? 'danger' : p.treatment_status === 'untreated' ? 'warning' : 'success'}`}>
                        {(p.treatment_status || '').replace(/_/g, ' ')}
                      </span>
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

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  return (
    <div>
      <div className="card mb-3">
        <div className="card-header fw-semibold">Screening Instruments</div>
        <div className="card-body">
          {(data.instruments || []).map((inst, i) => (
            <div key={i} className="mb-3 pb-3 border-bottom">
              <div className="fw-bold">{inst.name} — {inst.full_name}</div>
              <div className="text-muted small mb-1">{inst.purpose}</div>
              <div className="small"><strong>Scoring:</strong> {inst.scoring}</div>
              <div className="small text-info"><strong>Epilepsy relevance:</strong> {inst.relevance}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card mb-3">
        <div className="card-header fw-semibold">Risk Severity Tiers</div>
        <div className="card-body">
          <div className="table-responsive">
            <table className="table table-sm">
              <thead><tr><th>Tier</th><th>Score Range</th><th>Action</th></tr></thead>
              <tbody>
                {(data.risk_severity_tiers || []).map((t, i) => (
                  <tr key={i}>
                    <td><SeverityBadge severity={t.tier} /></td>
                    <td>{t.score_range}</td>
                    <td className="small">{t.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card mb-3">
        <div className="card-header fw-semibold">Key Comorbidities in Epilepsy</div>
        <div className="card-body">
          {(data.key_comorbidities || []).map((c, i) => (
            <div key={i} className="mb-3">
              <div className="fw-semibold">{c.condition} <span className="badge bg-secondary">{c.prevalence_epilepsy}</span></div>
              <div className="small text-muted">{c.mechanism}</div>
              <div className="small text-warning"><strong>Governance:</strong> {c.governance_note}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card mb-3">
        <div className="card-header fw-semibold">AI Governance Notes</div>
        <div className="card-body">
          <ul className="mb-0">
            {(data.ai_governance_notes || []).map((n, i) => (
              <li key={i} className="small mb-1">{n}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="card">
        <div className="card-header fw-semibold">References</div>
        <div className="card-body">
          <ul className="mb-0">
            {(data.references || []).map((r, i) => (
              <li key={i} className="small">{r}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default function MoodComorbidityPage() {
  const [tab, setTab] = useState('overview');
  const [data, setData] = useState({});

  useEffect(() => {
    if (!data[tab]) {
      fetch(`${API}/api/mood-comorbidity/${tab}`)
        .then(r => r.json())
        .then(d => setData(prev => ({ ...prev, [tab]: d })))
        .catch(() => setData(prev => ({ ...prev, [tab]: { error: 'Failed to load' } })));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: '1.6rem' }}>🧠</span>
        <div>
          <h4 className="mb-0 fw-bold">Mood-Comorbidity View</h4>
          <div className="text-muted small">Psychiatric comorbidity screening — PHQ-9 · GAD-7 · NDDI-E · Behavioral Risk</div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewPanel data={data.overview} />}
      {tab === 'breakdown' && <BreakdownPanel data={data.breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={data.definitions} />}
    </div>
  );
}
