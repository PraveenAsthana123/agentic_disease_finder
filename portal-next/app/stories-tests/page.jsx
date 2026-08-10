'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: '📊 Overview' },
  { id: 'stories',     label: '📖 User Stories' },
  { id: 'testing',     label: '🧪 Test Matrix' },
  { id: 'definitions', label: '📚 Definitions' },
];

const STATUS_COLOR = { built: 'success', partial: 'warning', planned: 'info' };
const STATUS_ICON  = { built: '✅', partial: '🔧', planned: '📅' };

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

function BarChart({ data, labelKey, valueKey, color = 'primary', title }) {
  if (!data?.length) return null;
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
            <div className={`progress-bar bg-${color}`}
              style={{ width: `${max > 0 ? (d[valueKey] / max) * 100 : 0}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function StatusBadge({ status }) {
  return (
    <span className={`badge bg-${STATUS_COLOR[status] || 'secondary'}`}>
      {STATUS_ICON[status] || '?'} {status}
    </span>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const s = data.summary || {};
  const statusDist = data.status_distribution || [];
  const dimTable   = data.dimension_table || [];
  const personas   = data.personas || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="User Stories"    value={s.total_user_stories}  color="primary"  sub="persona-driven requirements" />
        <KPI label="Demo Stories"    value={s.total_demo_stories}  color="info"     sub="timed walkthrough scripts" />
        <KPI label="Test Dimensions" value={s.total_test_dimensions} color="warning" sub="9-axis coverage matrix" />
        <KPI label="% Built"         value={`${s.pct_built ?? 0}%`} color={s.pct_built >= 80 ? 'success' : 'warning'} sub="of test dimensions" />
      </div>

      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Test-Dimension Status</div>
            <div className="card-body">
              <BarChart data={statusDist} labelKey="name" valueKey="value" color="success" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Personas Covered</div>
            <div className="card-body">
              <ul className="list-group list-group-flush">
                {personas.map((p, i) => (
                  <li key={i} className="list-group-item d-flex align-items-center gap-2">
                    <span className="badge bg-primary rounded-pill">{i + 1}</span>
                    <span className="small">{p}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold">Test Dimension Summary</div>
        <div className="table-responsive">
          <table className="table table-hover table-sm mb-0">
            <thead className="table-dark">
              <tr>
                <th>Dimension</th>
                <th>What is Tested</th>
                <th>How</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {dimTable.map((row, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{row.dim}</td>
                  <td className="small">{row.tests}</td>
                  <td className="small text-muted">{row.how}</td>
                  <td><StatusBadge status={row.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function StoriesPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const userStories = data.user_stories || [];
  const demoStories = data.demo_stories || [];

  return (
    <div>
      <h6 className="fw-bold mb-3">👤 User Stories</h6>
      <div className="row mb-4">
        {userStories.map((s, i) => (
          <div key={i} className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header d-flex align-items-center gap-2">
                <span className="badge bg-primary">{s.persona}</span>
              </div>
              <div className="card-body">
                <p className="small mb-2 fst-italic">"{s.story}"</p>
                <div className="text-muted" style={{ fontSize: '0.75rem' }}>
                  <strong>Endpoint:</strong> <code>{s.endpoint}</code>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-3">🎬 Demo Stories</h6>
      <div className="row">
        {demoStories.map((d, i) => (
          <div key={i} className="col-md-4 mb-3">
            <div className="card shadow-sm h-100 border-info">
              <div className="card-header bg-info text-white fw-semibold small">{d.title}</div>
              <div className="card-body">
                <p className="small mb-2">{d.script}</p>
                <div className="text-muted" style={{ fontSize: '0.75rem' }}>
                  <strong>Shows:</strong> {d.shows}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TestingPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const testing = data.testing || [];

  return (
    <div>
      <div className="card shadow-sm">
        <div className="card-header fw-semibold">9-Dimension Testing Matrix</div>
        <div className="table-responsive">
          <table className="table table-hover table-sm mb-0">
            <thead className="table-dark">
              <tr>
                <th>#</th>
                <th>Dimension</th>
                <th>Tests</th>
                <th>How to Run</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {testing.map((row, i) => (
                <tr key={i}>
                  <td className="text-muted">{i + 1}</td>
                  <td className="fw-semibold">{row.dim}</td>
                  <td className="small">{row.tests}</td>
                  <td className="small text-muted font-monospace">{row.how}</td>
                  <td><StatusBadge status={row.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const glossary  = data.glossary  || [];
  const notes     = data.notes     || [];
  const refs      = data.references || [];
  const legend    = data.status_legend || [];

  return (
    <div>
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">📖 Glossary</div>
            <ul className="list-group list-group-flush">
              {glossary.map((g, i) => (
                <li key={i} className="list-group-item">
                  <strong className="small">{g.term}</strong>
                  <div className="text-muted" style={{ fontSize: '0.78rem' }}>{g.definition}</div>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">🚦 Status Legend</div>
            <ul className="list-group list-group-flush">
              {legend.map((l, i) => (
                <li key={i} className="list-group-item d-flex align-items-start gap-2">
                  <StatusBadge status={l.status} />
                  <span className="small">{l.meaning}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">📝 Notes</div>
            <ul className="list-group list-group-flush">
              {notes.map((n, i) => (
                <li key={i} className="list-group-item small">{n}</li>
              ))}
            </ul>
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">🔗 References</div>
            <ul className="list-group list-group-flush">
              {refs.map((r, i) => (
                <li key={i} className="list-group-item small text-muted">{r}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function StoriesTestsDashboard() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/stories-tests/overview`).then(r => r.json()),
      fetch(`${API}/api/stories-tests/breakdown`).then(r => r.json()),
      fetch(`${API}/api/stories-tests/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1">📖 User Stories &amp; Tests Dashboard</h4>
        <p className="text-muted small mb-0">
          Persona-driven user stories · timed demo scripts · 9-dimension testing matrix
        </p>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel    data={ov} />}
      {tab === 'stories'     && <StoriesPanel     data={bd} />}
      {tab === 'testing'     && <TestingPanel     data={bd} />}
      {tab === 'definitions' && <DefinitionsPanel data={defs} />}
    </div>
  );
}
