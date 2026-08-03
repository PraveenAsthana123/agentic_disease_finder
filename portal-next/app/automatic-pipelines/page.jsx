'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Breakdown' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '\u2014'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  const color = status === 'automatic' ? 'success' : status === 'semi' ? 'warning' : 'secondary';
  return <span className={`badge bg-${color}`}>{status}</span>;
}

function TriggerBadge({ trigger }) {
  const colors = { upload: 'info', 'on-demand': 'primary', scheduled: 'warning', query: 'success', stream: 'danger' };
  const t = trigger.toLowerCase();
  let cat = 'other';
  if (t.includes('upload')) cat = 'upload';
  else if (t.includes('cron') || t.includes('scheduled')) cat = 'scheduled';
  else if (t.includes('query')) cat = 'query';
  else if (t.includes('stream')) cat = 'stream';
  else if (t.includes('on-demand')) cat = 'on-demand';
  return <span className={`badge bg-${colors[cat] || 'secondary'} bg-opacity-75`}>{trigger}</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (!data.available) return <div className="alert alert-warning">{data.note || 'No data available'}</div>;

  const s = data.summary || {};
  const triggers = data.trigger_distribution || {};
  const endpoints = data.endpoint_type_distribution || {};

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Pipelines" value={s.total_pipelines} color="primary" sub="registered process chains" />
        <KPI label="Fully Automatic" value={s.automatic} color="success" sub="no manual steps" />
        <KPI label="Semi-Automatic" value={s.semi} color="warning" sub="some manual input" />
        <KPI label="Automation Rate" value={`${s.automation_pct}%`} color="info" sub="automatic / total" />
      </div>
      <div className="row mb-3">
        <KPI label="Total Stages" value={s.total_stages} color="secondary" sub="across all pipelines" />
        <KPI label="Avg Stages" value={s.avg_stages_per_pipeline} color="dark" sub="per pipeline" />
        <KPI label="Max Stages" value={s.max_stages} color="danger" sub="longest pipeline" />
        <KPI label="Min Stages" value={s.min_stages} color="success" sub="shortest pipeline" />
      </div>

      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Trigger Distribution</div>
            <div className="card-body">
              {Object.entries(triggers).map(([k, v]) => (
                <div key={k} className="d-flex justify-content-between align-items-center mb-2">
                  <span className="text-capitalize">{k}</span>
                  <div className="d-flex align-items-center gap-2">
                    <div className="progress" style={{ width: 120, height: 10 }}>
                      <div className="progress-bar bg-info" style={{ width: `${(v / s.total_pipelines) * 100}%` }} />
                    </div>
                    <span className="badge bg-primary">{v}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Endpoint Types</div>
            <div className="card-body">
              {Object.entries(endpoints).map(([k, v]) => (
                <div key={k} className="d-flex justify-content-between align-items-center mb-2">
                  <span className="text-uppercase fw-semibold">{k}</span>
                  <div className="d-flex align-items-center gap-2">
                    <div className="progress" style={{ width: 120, height: 10 }}>
                      <div className="progress-bar bg-success" style={{ width: `${(v / s.total_pipelines) * 100}%` }} />
                    </div>
                    <span className="badge bg-secondary">{v}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (!data.available) return <div className="alert alert-warning">No data available</div>;

  const pipelines = data.pipelines || [];
  const meta = data.meta || {};

  return (
    <div>
      {meta.note && <div className="alert alert-info small mb-3">{meta.note}</div>}
      <div className="table-responsive">
        <table className="table table-hover table-bordered align-middle">
          <thead className="table-dark">
            <tr>
              <th style={{ width: 30 }}>#</th>
              <th>Process</th>
              <th>Status</th>
              <th>Trigger</th>
              <th>Endpoint</th>
              <th>Stages</th>
              <th style={{ width: 60 }}>#</th>
            </tr>
          </thead>
          <tbody>
            {pipelines.map((p, i) => (
              <tr key={i}>
                <td className="text-muted">{i + 1}</td>
                <td className="fw-semibold">{p.process}</td>
                <td><StatusBadge status={p.status} /></td>
                <td><TriggerBadge trigger={p.trigger} /></td>
                <td><code className="small">{p.endpoint}</code></td>
                <td>
                  <ol className="mb-0 ps-3 small">
                    {(p.stages || []).map((s, j) => <li key={j}>{s}</li>)}
                  </ol>
                </td>
                <td className="text-center"><span className="badge bg-dark">{p.stage_count}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {meta.updated_at && (
        <div className="text-muted small mt-2">Registry updated: {meta.updated_at}</div>
      )}
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (!data.available) return <div className="alert alert-warning">No data available</div>;

  const defs = data.definitions || [];

  return (
    <div>
      <table className="table table-striped">
        <thead className="table-light">
          <tr><th style={{ width: 200 }}>Term</th><th>Definition</th></tr>
        </thead>
        <tbody>
          {defs.map((d, i) => (
            <tr key={i}>
              <td className="fw-semibold">{d.term}</td>
              <td>{d.definition}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function AutomaticPipelinesPage() {
  const [tab, setTab] = useState('overview');
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch(`${API}/api/automatic-pipelines/${tab}`)
      .then(r => r.json())
      .then(d => setData(prev => ({ ...prev, [tab]: d })))
      .catch(() => setData(prev => ({ ...prev, [tab]: { available: false, note: 'Fetch failed' } })))
      .finally(() => setLoading(false));
  }, [tab]);

  return (
    <div className="container-fluid py-4">
      <h3 className="mb-1">Automatic Pipelines</h3>
      <p className="text-muted mb-3">End-to-end automated process chains — status, triggers, stages, and automation rate</p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {loading && !data[tab] && <div className="text-muted">Loading...</div>}

      {tab === 'overview' && <OverviewPanel data={data.overview} />}
      {tab === 'breakdown' && <BreakdownPanel data={data.breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={data.definitions} />}
    </div>
  );
}
