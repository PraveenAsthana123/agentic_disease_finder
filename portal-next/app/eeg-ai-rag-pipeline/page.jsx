'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'breakdown',   label: 'Step Breakdown' },
  { id: 'definitions', label: 'Definitions' },
];

const PHASE_COLORS = ['primary', 'info', 'success', 'warning', 'danger', 'secondary'];

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

function StatusBadge({ status }) {
  const color = status === 'built' ? 'success' : status === 'partial' ? 'warning' : 'secondary';
  return <span className={`badge bg-${color}`}>{status}</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;

  const k = data.kpis || {};
  const phases = data.phases || [];
  const steps = data.steps || [];

  return (
    <div>
      <div className="row mb-4">
        <KPI label="Total Steps"    value={k.total_steps}    color="primary" />
        <KPI label="Built"          value={k.built}          color="success" sub="100% complete" />
        <KPI label="Completion"     value={`${k.completion_pct}%`} color="success" />
        <KPI label="Phases"         value={k.phases}         color="info" />
      </div>

      <div className="alert alert-success mb-4">
        <strong>✅ Full pipeline operational</strong> — all 23 steps built end-to-end. From raw EEG upload to RAG-assisted clinical report with HITL review.
      </div>

      <h5 className="mb-3">Phase Summary</h5>
      <div className="row mb-4">
        {phases.map((ph, i) => (
          <div key={ph.phase} className="col-md-4 mb-3">
            <div className={`card border-${PHASE_COLORS[i % PHASE_COLORS.length]} h-100`}>
              <div className={`card-header bg-${PHASE_COLORS[i % PHASE_COLORS.length]} text-white py-2`}>
                <strong>{ph.phase}</strong>
              </div>
              <div className="card-body py-2">
                <div className="d-flex justify-content-between align-items-center">
                  <span className="text-muted small">{ph.total} steps</span>
                  <span className="badge bg-success">{ph.completion_pct}%</span>
                </div>
                <div className="progress mt-2" style={{ height: 6 }}>
                  <div className={`progress-bar bg-success`} style={{ width: `${ph.completion_pct}%` }} />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h5 className="mb-3">Pipeline Flow (23 steps)</h5>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-dark">
            <tr>
              <th style={{ width: 50 }}>#</th>
              <th>Step</th>
              <th>Phase</th>
              <th>Status</th>
              <th>Where Built</th>
            </tr>
          </thead>
          <tbody>
            {steps.map(s => (
              <tr key={s.n}>
                <td className="fw-bold text-muted">{s.n}</td>
                <td>
                  <div className="fw-semibold">{s.step}</div>
                  <div className="text-muted small">{s.detail}</div>
                </td>
                <td><span className="badge bg-secondary bg-opacity-75">{s.phase}</span></td>
                <td><StatusBadge status={s.status} /></td>
                <td className="text-muted small">{s.where || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;

  const steps = data.steps || [];
  const phaseMap = {};
  steps.forEach(s => {
    const ph = s.phase || 'Other';
    if (!phaseMap[ph]) phaseMap[ph] = [];
    phaseMap[ph].push(s);
  });

  return (
    <div>
      {Object.entries(phaseMap).map(([phase, phSteps], i) => (
        <div key={phase} className="mb-4">
          <h6 className={`text-${PHASE_COLORS[i % PHASE_COLORS.length]} fw-bold mb-2`}>
            {phase} ({phSteps.length} steps)
          </h6>
          {phSteps.map(s => (
            <div key={s.n} className="card mb-2 shadow-sm">
              <div className="card-body py-2 px-3">
                <div className="d-flex justify-content-between align-items-start">
                  <div>
                    <span className="text-muted small me-2">#{s.n}</span>
                    <span className="fw-semibold">{s.step}</span>
                    <div className="text-muted small mt-1">{s.detail}</div>
                    {s.where && (
                      <div className="text-info small mt-1">
                        <em>{s.where}</em>
                      </div>
                    )}
                  </div>
                  <StatusBadge status={s.status} />
                </div>
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;

  const phases = data.phases || [];
  const terms = data.terms || [];

  return (
    <div>
      <h5 className="mb-3">Phase Definitions</h5>
      {phases.map((ph, i) => (
        <div key={ph.name} className="card mb-3 shadow-sm">
          <div className={`card-header bg-${PHASE_COLORS[i % PHASE_COLORS.length]} bg-opacity-10 py-2`}>
            <strong>{ph.name}</strong>
            {ph.steps && <span className="text-muted small ms-2">Steps {ph.steps}</span>}
          </div>
          <div className="card-body py-2">
            <p className="mb-0 small">{ph.description}</p>
          </div>
        </div>
      ))}

      {terms.length > 0 && (
        <>
          <h5 className="mt-4 mb-3">Glossary</h5>
          <div className="table-responsive">
            <table className="table table-sm table-bordered">
              <thead className="table-light">
                <tr><th>Term</th><th>Definition</th></tr>
              </thead>
              <tbody>
                {terms.map(t => (
                  <tr key={t.term}>
                    <td className="fw-semibold text-nowrap">{t.term}</td>
                    <td className="text-muted small">{t.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

export default function EegAiRagPipelinePage() {
  const [overview,    setOverview]    = useState(null);
  const [breakdown,   setBreakdown]   = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab,         setTab]         = useState('overview');
  const [error,       setError]       = useState(null);

  useEffect(() => {
    fetch(`${API}/api/eeg-ai-rag-pipeline/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(String(e)));
    fetch(`${API}/api/eeg-ai-rag-pipeline/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/eeg-ai-rag-pipeline/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div>
      <div className="mb-4">
        <h2 className="mb-1">🔬 EEG → AI → RAG Pipeline</h2>
        <p className="text-muted mb-0">
          Complete 23-step end-to-end pipeline: raw EEG acquisition → preprocessing → feature engineering → model training → RAG report generation → HITL review.
        </p>
        {overview && (
          <div className="text-muted small mt-1">
            Updated: {overview.updated_at} · {overview.note}
          </div>
        )}
      </div>

      {error && <div className="alert alert-warning">Backend error: {error}</div>}

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
      {tab === 'breakdown'   && <BreakdownPanel   data={breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={definitions} />}
    </div>
  );
}
