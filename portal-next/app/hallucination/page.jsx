'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'risk-types',  label: 'Risk Types' },
  { id: 'grounding',   label: 'Grounding' },
  { id: 'mitigations', label: 'Mitigations' },
  { id: 'definitions', label: 'Definitions' },
];

const SEV_COLOR = { critical: 'danger', high: 'warning', medium: 'info', low: 'success' };
const RISK_LEVEL_COLOR = { elevated: 'warning', high: 'danger', critical: 'danger', low: 'success', moderate: 'info' };

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

function RiskGauge({ score, level }) {
  const color = RISK_LEVEL_COLOR[level] || 'warning';
  const pct = Math.min(Math.max(score || 0, 0), 100);
  return (
    <div className="card shadow-sm mb-4">
      <div className="card-header py-2 bg-dark text-white">
        <strong>Overall Hallucination Risk Score</strong>
      </div>
      <div className="card-body text-center">
        <div className={`display-4 fw-bold text-${color} mb-1`}>{pct.toFixed(1)}</div>
        <div className="text-muted small mb-3">out of 100 — lower is better</div>
        <div className="progress mb-2" style={{ height: 24, borderRadius: 12 }}>
          <div
            className={`progress-bar bg-${color}`}
            style={{ width: `${pct}%`, borderRadius: 12, transition: 'width 0.8s ease' }}
          />
        </div>
        <span className={`badge bg-${color} mt-1`} style={{ fontSize: '0.95rem' }}>
          {(level || 'unknown').toUpperCase()} RISK
        </span>
      </div>
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const sum = data.summary || {};
  const dist = data.grounding_distribution || {};
  const types = data.type_coverage || [];
  const conf = data.confidence_stats || {};

  const totalDist = (dist.grounded || 0) + (dist.partially_grounded || 0) + (dist.ungrounded || 0);

  return (
    <div>
      <div className="row mb-4">
        <KPI label="Overall Risk Score"  value={`${sum.overall_risk_score?.toFixed(1)}`}  color={RISK_LEVEL_COLOR[sum.risk_level] || 'warning'} sub={sum.risk_level?.toUpperCase()} />
        <KPI label="Grounding Score"     value={`${sum.grounding_score?.toFixed(1)}%`}     color="info"    sub="RAG embedding quality" />
        <KPI label="Citation Rate"       value={`${sum.citation_rate?.toFixed(1)}%`}        color="success" sub="Analyses with evidence" />
        <KPI label="Faithfulness Rate"   value={`${sum.faithfulness_rate?.toFixed(1)}%`}    color="warning" sub="Operator confirmations" />
      </div>
      <div className="row mb-4">
        <KPI label="Total Embeddings"    value={sum.total_embeddings}   color="primary" sub="ChromaDB vectors" />
        <KPI label="AI Analyses"         value={sum.total_analyses}     color="secondary" sub="Predictions scored" />
        <KPI label="RAG Queries"         value={sum.total_rag_queries}  color="info" sub="Retrieval calls" />
        <KPI label="HITL Reviews"        value={sum.hitl_reviews}       color="dark" sub="Human verifications" />
      </div>

      <RiskGauge score={sum.overall_risk_score} level={sum.risk_level} />

      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header py-2 bg-dark text-white">Grounding Distribution</div>
            <div className="card-body">
              {[
                { key: 'grounded', label: 'Grounded', color: 'success' },
                { key: 'partially_grounded', label: 'Partially Grounded', color: 'warning' },
                { key: 'ungrounded', label: 'Ungrounded', color: 'danger' },
              ].map(({ key, label, color }) => {
                const count = dist[key] || 0;
                const pct = totalDist > 0 ? ((count / totalDist) * 100).toFixed(1) : 0;
                return (
                  <div key={key} className="mb-3">
                    <div className="d-flex justify-content-between mb-1">
                      <span className={`text-${color} fw-semibold`}>{label}</span>
                      <span className="fw-bold">{count} <small className="text-muted">({pct}%)</small></span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header py-2 bg-dark text-white">Confidence Statistics</div>
            <div className="card-body">
              <div className="row text-center">
                <div className="col-4">
                  <div className="h4 text-success fw-bold">{conf.avg?.toFixed(2)}</div>
                  <div className="text-muted small">Avg</div>
                </div>
                <div className="col-4">
                  <div className="h4 text-danger fw-bold">{conf.min?.toFixed(2)}</div>
                  <div className="text-muted small">Min</div>
                </div>
                <div className="col-4">
                  <div className="h4 text-info fw-bold">{conf.max?.toFixed(2)}</div>
                  <div className="text-muted small">Max</div>
                </div>
              </div>
              <div className="mt-3">
                <div className="text-muted small fw-bold mb-1">CONFIDENCE RANGE</div>
                <div className="progress" style={{ height: 14 }}>
                  <div
                    className="progress-bar bg-success"
                    style={{ width: `${((conf.avg || 0) * 100).toFixed(0)}%` }}
                  />
                </div>
                <div className="d-flex justify-content-between mt-1">
                  <span className="text-muted" style={{ fontSize: '0.7rem' }}>0</span>
                  <span className="text-muted" style={{ fontSize: '0.7rem' }}>1.0</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header py-2 bg-dark text-white">Document Type Coverage ({types.length} types)</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Document Type</th>
                <th className="text-end">Count</th>
                <th>Coverage Bar</th>
              </tr>
            </thead>
            <tbody>
              {types.map((t, i) => {
                const maxCount = Math.max(...types.map(x => x.count), 1);
                const pct = ((t.count / maxCount) * 100).toFixed(0);
                return (
                  <tr key={i}>
                    <td className="fw-semibold">{t.type}</td>
                    <td className="text-end fw-bold">{t.count}</td>
                    <td style={{ width: '40%' }}>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-primary" style={{ width: `${pct}%` }} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function RiskTypesPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const risks = data.risk_breakdown || [];

  return (
    <div>
      <p className="text-muted mb-3">6 hallucination risk types identified — sorted by risk score (highest first).</p>
      <div className="row">
        {[...risks].sort((a, b) => b.risk_score - a.risk_score).map((r, i) => {
          const color = SEV_COLOR[r.severity] || 'secondary';
          return (
            <div key={i} className="col-md-6 mb-3">
              <div className={`card h-100 shadow-sm border-${color}`}>
                <div className={`card-header d-flex justify-content-between align-items-center py-2 bg-${color} text-white`}>
                  <strong>{r.label}</strong>
                  <span className="badge bg-light text-dark">{r.severity?.toUpperCase()}</span>
                </div>
                <div className="card-body">
                  <div className="mb-3">
                    <div className="d-flex justify-content-between mb-1">
                      <span className="text-muted small">Risk Score</span>
                      <span className={`fw-bold text-${color}`}>{r.risk_score?.toFixed(1)}</span>
                    </div>
                    <div className="progress" style={{ height: 12 }}>
                      <div className={`progress-bar bg-${color}`} style={{ width: `${r.risk_score?.toFixed(0)}%` }} />
                    </div>
                  </div>
                  <div className="p-2 bg-light rounded small">
                    <strong className="text-muted">Mitigation:</strong>
                    <div className="mt-1">{r.mitigation}</div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function GroundingPanel({ breakdown }) {
  if (!breakdown) return <div className="text-muted">Loading…</div>;
  const patients = breakdown.patient_grounding || [];
  const diseases = breakdown.disease_coverage || [];
  const faith = breakdown.interaction_faithfulness || {};
  const hitl = breakdown.hitl_verification || {};
  const queryPats = breakdown.query_patients || [];

  return (
    <div>
      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header py-2 bg-dark text-white">Interaction Faithfulness</div>
            <div className="card-body">
              <div className="row text-center mb-3">
                <div className="col-4">
                  <div className="h4 fw-bold text-primary">{faith.total_assistant_responses?.toLocaleString()}</div>
                  <div className="text-muted small">AI Responses</div>
                </div>
                <div className="col-4">
                  <div className="h4 fw-bold text-danger">{faith.corrections}</div>
                  <div className="text-muted small">Corrections</div>
                </div>
                <div className="col-4">
                  <div className="h4 fw-bold text-success">{faith.confirmations}</div>
                  <div className="text-muted small">Confirmations</div>
                </div>
              </div>
              <div>
                <div className="d-flex justify-content-between mb-1">
                  <span className="text-muted small">Faithfulness Rate</span>
                  <span className="fw-bold text-warning">{faith.faithfulness_rate?.toFixed(1)}%</span>
                </div>
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar bg-warning" style={{ width: `${faith.faithfulness_rate?.toFixed(0)}%` }} />
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header py-2 bg-dark text-white">HITL Verification</div>
            <div className="card-body">
              <div className="row text-center mb-3">
                <div className="col-4">
                  <div className="h4 fw-bold text-primary">{hitl.total_reviews}</div>
                  <div className="text-muted small">Total Reviews</div>
                </div>
                <div className="col-4">
                  <div className="h4 fw-bold text-success">{hitl.approved}</div>
                  <div className="text-muted small">Approved</div>
                </div>
                <div className="col-4">
                  <div className="h4 fw-bold text-danger">{hitl.rejected}</div>
                  <div className="text-muted small">Rejected</div>
                </div>
              </div>
              {hitl.total_reviews > 0 && (
                <div>
                  <div className="d-flex justify-content-between mb-1">
                    <span className="text-muted small">Approval Rate</span>
                    <span className="fw-bold text-success">{((hitl.approved / hitl.total_reviews) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar bg-success" style={{ width: `${((hitl.approved / hitl.total_reviews) * 100).toFixed(0)}%` }} />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="row mb-4">
        <div className="col-md-5 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">Disease Coverage ({diseases.length} diseases)</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Disease</th><th className="text-end">Docs</th></tr></thead>
                <tbody>
                  {diseases.map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold text-capitalize">{d.disease.replace('_', ' ')}</td>
                      <td className="text-end fw-bold">{d.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-7 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              Per-Patient Grounding ({patients.length} patients)
            </div>
            <div className="card-body p-0" style={{ maxHeight: 280, overflowY: 'auto' }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>Patient</th><th className="text-center">Docs</th><th>Grounding</th></tr>
                </thead>
                <tbody>
                  {patients.map((p, i) => {
                    const score = p.grounding_score || 0;
                    const color = score >= 70 ? 'success' : score >= 40 ? 'warning' : 'danger';
                    return (
                      <tr key={i}>
                        <td><code className="text-info">{p.patient_id}</code></td>
                        <td className="text-center">{p.total_docs}</td>
                        <td style={{ width: '45%' }}>
                          <div className="d-flex align-items-center gap-2">
                            <div className="progress flex-grow-1" style={{ height: 8 }}>
                              <div className={`progress-bar bg-${color}`} style={{ width: `${score}%` }} />
                            </div>
                            <small className={`text-${color} fw-bold`} style={{ minWidth: 36 }}>{score.toFixed(0)}%</small>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {queryPats.length > 0 && (
        <div className="alert alert-secondary">
          <strong>RAG Query Patients:</strong>{' '}
          {queryPats.map(p => <code key={p} className="me-2 text-info">{p}</code>)}
          <span className="text-muted small ms-2">— patients who triggered RAG retrieval</span>
        </div>
      )}
    </div>
  );
}

function MitigationsPanel({ breakdown }) {
  if (!breakdown) return <div className="text-muted">Loading…</div>;
  const mitigations = breakdown.mitigations || [];

  return (
    <div>
      <p className="text-muted mb-4">Active hallucination prevention strategies deployed in this system.</p>
      <div className="row">
        {mitigations.map((m, i) => (
          <div key={i} className="col-md-6 mb-3">
            <div className="card h-100 shadow-sm border-success">
              <div className="card-header d-flex justify-content-between align-items-center py-2 bg-success text-white">
                <strong>{m.strategy}</strong>
                <span className="badge bg-light text-success fw-bold">{m.status?.toUpperCase()}</span>
              </div>
              <div className="card-body">
                <div className="row">
                  <div className="col-6">
                    <div className="text-muted small fw-bold mb-1">COVERAGE</div>
                    <div>{m.coverage}</div>
                  </div>
                  <div className="col-6">
                    <div className="text-muted small fw-bold mb-1">EFFECTIVENESS</div>
                    <div className="fw-bold text-success">{m.effectiveness}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="alert alert-info mt-2">
        <strong>All {mitigations.length} mitigations active.</strong>{' '}
        Hallucination risk is continuously monitored via RAG grounding, HITL reviews, and confidence calibration.
      </div>
    </div>
  );
}

function DefinitionsPanel({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  const metrics = defs.metrics || [];
  return (
    <div className="row">
      {metrics.map((d, i) => (
        <div key={i} className="col-md-6 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header py-2 bg-light">
              <strong>{d.metric}</strong>
            </div>
            <div className="card-body">
              <p className="text-muted small mb-2">{d.definition}</p>
              {d.source && (
                <div className="mt-2 p-2 bg-light rounded">
                  <span className="text-muted" style={{ fontSize: '0.7rem' }}>SOURCE: </span>
                  <code style={{ fontSize: '0.7rem' }}>{d.source}</code>
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function HallucinationPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/hallucination/overview`)
      .then(r => r.json()).then(setOverview).catch(() => setErr('Failed to load overview'));
    fetch(`${API}/api/hallucination/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => setErr('Failed to load breakdown'));
    fetch(`${API}/api/hallucination/definitions`)
      .then(r => r.json()).then(setDefs).catch(() => setErr('Failed to load definitions'));
  }, []);

  const sum = overview?.summary || {};

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-1 gap-2 flex-wrap">
        <h1 className="h4 mb-0">🧠 Hallucination Risk Dashboard</h1>
        <span className="badge bg-danger">AI Governance</span>
        <span className="badge bg-secondary">LLM Ops</span>
        {sum.risk_level && (
          <span className={`badge bg-${RISK_LEVEL_COLOR[sum.risk_level] || 'warning'}`}>
            {sum.risk_level?.toUpperCase()} RISK
          </span>
        )}
      </div>
      <p className="text-muted small mb-3">
        Hallucination detection &amp; grounding analysis — {sum.total_embeddings ?? '—'} embeddings · {sum.total_analyses ?? '—'} analyses · {sum.hitl_reviews ?? '—'} HITL reviews
      </p>

      {err && <div className="alert alert-danger">{err}</div>}

      <ul className="nav nav-tabs mb-4">
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

      {tab === 'overview'    && <OverviewPanel    data={overview} />}
      {tab === 'risk-types'  && <RiskTypesPanel   data={overview} />}
      {tab === 'grounding'   && <GroundingPanel   breakdown={breakdown} />}
      {tab === 'mitigations' && <MitigationsPanel breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel defs={defs} />}
    </div>
  );
}
