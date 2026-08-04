'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'modalities',  label: 'Modalities' },
  { id: 'ai-streams',  label: 'AI Streams' },
  { id: 'phases',      label: 'Phases' },
  { id: 'checklist',   label: 'Provider Checklist' },
  { id: 'definitions', label: 'Definitions' },
];

const TIER_COLORS = { 1: 'danger', 2: 'warning', 3: 'info' };
const TIER_LABELS = { 1: 'Tier 1 — Core', 2: 'Tier 2 — Extended', 3: 'Tier 3 — Specialized' };
const STATUS_COLOR = { built: 'success', partial: 'warning', planned: 'secondary', scaffold: 'info', unknown: 'light' };

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
  const s = (status || '').toLowerCase();
  const key = Object.keys(STATUS_COLOR).find(k => s.startsWith(k)) || 'unknown';
  return <span className={`badge bg-${STATUS_COLOR[key]}`}>{s || '—'}</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const sum = data.summary || {};
  const scale = data.target_scale || {};
  const tierDist = data.modality_tier_distribution || {};
  const statusDist = data.modality_status_distribution || {};
  const streamDist = data.ai_stream_distribution || {};

  return (
    <div>
      <div className="row mb-4">
        <KPI label="Total Modalities" value={sum.total_modalities}  color="primary" />
        <KPI label="Built"            value={sum.modalities_built}  color="success" sub={`${sum.coverage_pct?.toFixed(0)}% complete`} />
        <KPI label="AI Streams"       value={sum.ai_streams_total}  color="info"    sub={`${sum.ai_streams_built} built`} />
        <KPI label="Phases Built"     value={`${sum.phases_built}/${sum.phases_total}`} color="warning" sub={sum.phases_partial ? `${sum.phases_partial} partial` : ''} />
      </div>

      {data.research_question && (
        <div className="alert alert-info mb-4">
          <strong>Research Question:</strong> {data.research_question}
        </div>
      )}

      <div className="row mb-4">
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header bg-dark text-white py-2">Target Scale</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <tbody>
                  <tr><td className="text-muted ps-3">Patients</td><td className="fw-bold">{scale.patients}</td></tr>
                  <tr><td className="text-muted ps-3">EEG Studies</td><td className="fw-bold">{scale.eeg_studies}</td></tr>
                  <tr><td className="text-muted ps-3">Video-EEG</td><td className="fw-bold">{scale.video_eeg}</td></tr>
                  <tr><td className="text-muted ps-3">Retro Years</td><td className="fw-bold">{scale.retrospective_years}</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header bg-dark text-white py-2">Modality Status</div>
            <div className="card-body">
              {Object.entries(statusDist).map(([status, count]) => (
                <div key={status} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <StatusBadge status={status} />
                    <span className="fw-bold">{count}</span>
                  </div>
                  <div className="progress" style={{ height: 6 }}>
                    <div
                      className={`progress-bar bg-${STATUS_COLOR[status] || 'secondary'}`}
                      style={{ width: `${Math.round((count / (sum.total_modalities || 1)) * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header bg-dark text-white py-2">AI Stream Status</div>
            <div className="card-body">
              {Object.entries(streamDist).map(([status, count]) => (
                <div key={status} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <StatusBadge status={status} />
                    <span className="fw-bold">{count}</span>
                  </div>
                  <div className="progress" style={{ height: 6 }}>
                    <div
                      className={`progress-bar bg-${STATUS_COLOR[status] || 'secondary'}`}
                      style={{ width: `${Math.round((count / (sum.ai_streams_total || 1)) * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <h5 className="mb-3">Tier Distribution</h5>
      <div className="row">
        {Object.entries(TIER_LABELS).map(([tier, label]) => {
          const key = `tier_${tier}`;
          return (
            <div key={tier} className="col-md-4 mb-3">
              <div className={`card border-${TIER_COLORS[tier]}`}>
                <div className="card-body text-center py-3">
                  <div className={`h3 fw-bold text-${TIER_COLORS[tier]}`}>{tierDist[key] || 0}</div>
                  <div className="small text-muted">{label}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ModalitiesPanel({ breakdown }) {
  const [filter, setFilter] = useState('all');
  if (!breakdown) return <div className="text-muted">Loading…</div>;
  const modalities = breakdown.modalities || [];
  const filtered = filter === 'all' ? modalities : modalities.filter(m => {
    const s = (m.status || '').toLowerCase();
    return s === filter || s.startsWith(filter);
  });

  const byTier = {};
  filtered.forEach(m => {
    const t = m.tier || 0;
    if (!byTier[t]) byTier[t] = [];
    byTier[t].push(m);
  });

  return (
    <div>
      <div className="mb-3 d-flex gap-2 flex-wrap">
        {['all', 'built', 'partial', 'scaffold', 'planned'].map(f => (
          <button
            key={f}
            className={`btn btn-sm ${filter === f ? 'btn-dark' : 'btn-outline-secondary'}`}
            onClick={() => setFilter(f)}
          >
            {f === 'all' ? `All (${modalities.length})` : f}
          </button>
        ))}
      </div>

      {Object.entries(byTier).sort(([a],[b]) => parseInt(a)-parseInt(b)).map(([tier, items]) => (
        <div key={tier} className="mb-4">
          <h6 className={`text-${TIER_COLORS[tier] || 'muted'} mb-2`}>{TIER_LABELS[tier] || `Tier ${tier}`}</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover align-middle">
              <thead className="table-dark">
                <tr>
                  <th>Code</th>
                  <th>Modality</th>
                  <th>Measures</th>
                  <th>AI Potential</th>
                  <th>Phase</th>
                  <th>Status</th>
                  <th>Where Built</th>
                </tr>
              </thead>
              <tbody>
                {items.map(m => (
                  <tr key={m.code}>
                    <td><code className="text-info">{m.code}</code></td>
                    <td className="fw-semibold">{m.name}</td>
                    <td className="text-muted small">{m.measures}</td>
                    <td className="text-muted small">{m.ai_potential}</td>
                    <td><span className="badge bg-secondary">{m.phase}</span></td>
                    <td><StatusBadge status={m.status} /></td>
                    <td className="text-muted small" style={{ maxWidth: 240 }}>{m.where || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}

function AIStreamsPanel({ breakdown }) {
  if (!breakdown) return <div className="text-muted">Loading…</div>;
  const streams = breakdown.ai_streams || [];

  return (
    <div>
      <p className="text-muted mb-3">{streams.length} AI analytics streams — EEG, video, multimodal, and autonomic.</p>
      {streams.map((s, i) => {
        const isBuilt = (s.status || '').startsWith('built');
        return (
          <div key={s.id || i} className="card mb-3 shadow-sm">
            <div className={`card-header d-flex justify-content-between align-items-center py-2 bg-${isBuilt ? 'success' : 'secondary'} text-white`}>
              <strong>{(s.id || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || `Stream ${i+1}`}</strong>
              <span className="badge bg-light text-dark">{isBuilt ? '✅ built' : s.status}</span>
            </div>
            <div className="card-body">
              <div className="row">
                <div className="col-md-4 mb-2">
                  <div className="text-muted small fw-bold mb-1">INPUT</div>
                  <div>{s.input}</div>
                </div>
                <div className="col-md-4 mb-2">
                  <div className="text-muted small fw-bold mb-1">OUTPUT</div>
                  <div>{s.output}</div>
                </div>
                <div className="col-md-4 mb-2">
                  <div className="text-muted small fw-bold mb-1">MODELS</div>
                  <div className="d-flex flex-wrap gap-1">
                    {(s.models || []).map(m => (
                      <span key={m} className="badge bg-primary bg-opacity-75">{m}</span>
                    ))}
                  </div>
                </div>
              </div>
              {s.where && (
                <div className="mt-2 p-2 bg-light rounded">
                  <code className="small text-dark">{s.where}</code>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function PhasesPanel({ breakdown }) {
  if (!breakdown) return <div className="text-muted">Loading…</div>;
  const phases = breakdown.phases || [];
  const modalities = breakdown.modalities || [];

  return (
    <div>
      {phases.map(ph => {
        const phaseMods = modalities.filter(m => m.phase === ph.phase);
        return (
          <div key={ph.phase} className="card mb-3 shadow-sm">
            <div className={`card-header d-flex justify-content-between align-items-center py-2 bg-${ph.status === 'built' ? 'success' : ph.status === 'partial' ? 'warning' : 'secondary'} text-white`}>
              <strong>Phase {ph.phase}: {ph.name}</strong>
              <StatusBadge status={ph.status} />
            </div>
            {phaseMods.length > 0 && (
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Code</th><th>Modality</th><th>Tier</th><th>Status</th></tr>
                  </thead>
                  <tbody>
                    {phaseMods.map(m => (
                      <tr key={m.code}>
                        <td><code className="text-info">{m.code}</code></td>
                        <td>{m.name}</td>
                        <td><span className={`badge bg-${TIER_COLORS[m.tier] || 'secondary'}`}>T{m.tier}</span></td>
                        <td><StatusBadge status={m.status} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ChecklistPanel({ breakdown }) {
  const [answers, setAnswers] = useState({});
  if (!breakdown) return <div className="text-muted">Loading…</div>;
  const questions = breakdown.provider_questions || [];
  const answeredCount = Object.values(answers).filter(v => v === 'yes').length;

  return (
    <div>
      <div className="alert alert-secondary mb-3">
        <strong>Provider Data Availability Checklist</strong> — {answeredCount}/{questions.length} confirmed
        <div className="progress mt-2" style={{ height: 8 }}>
          <div className="progress-bar bg-success" style={{ width: `${Math.round(answeredCount / Math.max(questions.length, 1) * 100)}%` }} />
        </div>
      </div>
      <div className="list-group">
        {questions.map((q, i) => (
          <div key={i} className="list-group-item d-flex justify-content-between align-items-center">
            <span>{q}</span>
            <div className="btn-group btn-group-sm">
              {['yes', 'no', 'tbd'].map(v => (
                <button
                  key={v}
                  className={`btn ${answers[i] === v
                    ? v === 'yes' ? 'btn-success' : v === 'no' ? 'btn-danger' : 'btn-warning'
                    : 'btn-outline-secondary'}`}
                  onClick={() => setAnswers(a => ({ ...a, [i]: v }))}
                >
                  {v.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DefinitionsPanel({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  const definitions = defs.definitions || [];
  return (
    <div className="row">
      {definitions.map((d, i) => (
        <div key={i} className="col-md-6 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header py-2 bg-light">
              <strong>{d.term}</strong>
            </div>
            <div className="card-body py-2 text-muted small">{d.definition}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function DatasetCoveragePage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/dataset-coverage/overview`)
      .then(r => r.json()).then(setOverview).catch(() => setErr('Failed to load overview'));
    fetch(`${API}/api/dataset-coverage/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => setErr('Failed to load breakdown'));
    fetch(`${API}/api/dataset-coverage/definitions`)
      .then(r => r.json()).then(setDefs).catch(() => setErr('Failed to load definitions'));
  }, []);

  const sum = overview?.summary || {};

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-1 gap-2 flex-wrap">
        <h1 className="h4 mb-0">🗂 Dataset Coverage</h1>
        <span className="badge bg-primary">Neurophysiology</span>
        {overview?.focus && <span className="badge bg-secondary">{overview.focus}</span>}
        {sum.coverage_pct !== undefined && (
          <span className="badge bg-success">{sum.coverage_pct?.toFixed(0)}% built</span>
        )}
      </div>
      <p className="text-muted small mb-3">Neurophysiology Dataset &amp; Modality Coverage Map — {sum.total_modalities} modalities · {sum.ai_streams_total} AI streams · 4 phases</p>

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

      {tab === 'overview'    && <OverviewPanel   data={overview} />}
      {tab === 'modalities'  && <ModalitiesPanel  breakdown={breakdown} />}
      {tab === 'ai-streams'  && <AIStreamsPanel   breakdown={breakdown} />}
      {tab === 'phases'      && <PhasesPanel      breakdown={breakdown} />}
      {tab === 'checklist'   && <ChecklistPanel   breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel defs={defs} />}
    </div>
  );
}
