'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Issue Breakdown' },
  { id: 'definitions', label: 'Definitions' },
];

const SEV_COLORS = { P1: 'danger', P2: 'warning' };
const DET_COLORS = { built: 'success', partial: 'warning', planned: 'secondary' };
const DET_ICONS = { built: '\u2705', partial: '\u26a0\ufe0f', planned: '\u23f3' };

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

function SeverityBadge({ severity }) {
  const color = SEV_COLORS[severity] || 'secondary';
  return <span className={`badge bg-${color}`}>{severity}</span>;
}

function DetectionBadge({ status }) {
  const color = DET_COLORS[status] || 'secondary';
  const icon = DET_ICONS[status] || '';
  return <span className={`badge bg-${color}`}>{icon} {status}</span>;
}

function CoverageBar({ built, partial, planned, total }) {
  const bPct = Math.round((built / Math.max(total, 1)) * 100);
  const pPct = Math.round((partial / Math.max(total, 1)) * 100);
  const plPct = 100 - bPct - pPct;
  return (
    <div className="progress" style={{ height: 18 }}>
      <div className="progress-bar bg-success" style={{ width: `${bPct}%` }} title={`Built: ${built}`}>{bPct > 8 ? `${bPct}%` : ''}</div>
      <div className="progress-bar bg-warning" style={{ width: `${pPct}%` }} title={`Partial: ${partial}`}>{pPct > 8 ? `${pPct}%` : ''}</div>
      <div className="progress-bar bg-secondary" style={{ width: `${plPct}%` }} title={`Planned: ${planned}`}>{plPct > 8 ? `${plPct}%` : ''}</div>
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;
  if (!data.available) return <div className="alert alert-info">Production issues catalog not available.</div>;

  const s = data.summary || {};
  const layers = data.layer_distribution || [];
  const sevDist = data.severity_distribution || [];
  const detDist = data.detection_distribution || [];
  const pareto = data.pareto_issues || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Issues" value={s.total_issues} color="primary" sub={`across ${s.total_layers} layers`} />
        <KPI label="P1 (Critical)" value={s.p1} color={s.p1 > 0 ? 'danger' : 'success'} sub="immediate safety risk" />
        <KPI label="P2 (Major)" value={s.p2} color="warning" sub="degrades quality" />
        <KPI label="Detection Coverage" value={`${s.coverage_pct}%`} color={s.coverage_pct >= 80 ? 'success' : s.coverage_pct >= 50 ? 'warning' : 'danger'} sub={`${s.built} built + ${s.partial} partial`} />
      </div>

      <div className="row mb-3">
        <KPI label="Built" value={s.built} color="success" sub="fully implemented" />
        <KPI label="Partial" value={s.partial} color="warning" sub="incomplete coverage" />
        <KPI label="Planned" value={s.planned} color="secondary" sub="not yet implemented" />
        <KPI label="Layers" value={s.total_layers} color="info" sub="architectural tiers" />
      </div>

      <div className="row mb-4">
        <div className="col-md-8 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Detection Coverage by Layer</div>
            <div className="card-body">
              {layers.map(l => (
                <div key={l.layer} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="fw-semibold">{l.layer}</span>
                    <span className="text-muted">{l.built + l.partial}/{l.total} covered ({l.p1 > 0 ? `${l.p1} P1` : ''}{ l.p1 > 0 && l.p2 > 0 ? ', ' : ''}{l.p2 > 0 ? `${l.p2} P2` : ''})</span>
                  </div>
                  <CoverageBar built={l.built} partial={l.partial} planned={l.planned} total={l.total} />
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Distribution</div>
            <div className="card-body">
              <div className="small fw-semibold mb-2">By Severity</div>
              {sevDist.map(s => (
                <div key={s.name} className="d-flex justify-content-between mb-1">
                  <SeverityBadge severity={s.name} />
                  <span className="fw-bold">{s.value}</span>
                </div>
              ))}
              <hr />
              <div className="small fw-semibold mb-2">By Detection Status</div>
              {detDist.map(d => (
                <div key={d.name} className="d-flex justify-content-between mb-1">
                  <span className={`badge bg-${d.name === 'Built' ? 'success' : d.name === 'Partial' ? 'warning' : 'secondary'}`}>{d.name}</span>
                  <span className="fw-bold">{d.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {pareto.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Pareto Issues (Top 20% cause 80% of incidents)</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead><tr><th>#</th><th>Issue</th></tr></thead>
              <tbody>
                {pareto.map((p, i) => (
                  <tr key={i}><td>{i + 1}</td><td className="small">{typeof p === 'string' ? p : p.issue || JSON.stringify(p)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {data.internal_flow && (
        <div className="alert alert-light border small">
          <strong>Internal Flow:</strong> {data.internal_flow}
        </div>
      )}
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (!data.available) return <div className="alert alert-info">No breakdown data.</div>;

  const [sevFilter, setSevFilter] = useState('all');
  const [detFilter, setDetFilter] = useState('all');
  const [expandedLayer, setExpandedLayer] = useState(null);

  const layers = data.layers || [];

  const allIssues = layers.flatMap(l => l.issues.map(i => ({ ...i, layer: l.layer })));
  const filtered = allIssues.filter(i =>
    (sevFilter === 'all' || i.severity === sevFilter) &&
    (detFilter === 'all' || i.det_status === detFilter)
  );

  return (
    <div>
      {data.meta && (
        <div className="alert alert-light border small mb-3">
          <strong>{data.meta.title}</strong>
          {data.meta.note && <span> &mdash; {data.meta.note}</span>}
          {data.meta.updated_at && <span className="text-muted"> (updated {data.meta.updated_at})</span>}
        </div>
      )}

      <div className="d-flex gap-2 mb-3 flex-wrap">
        <div className="btn-group">
          {['all', 'P1', 'P2'].map(f => (
            <button key={f} className={`btn btn-sm ${sevFilter === f ? 'btn-primary' : 'btn-outline-secondary'}`} onClick={() => setSevFilter(f)}>
              {f === 'all' ? `All Sev (${allIssues.length})` : `${f} (${allIssues.filter(i => i.severity === f).length})`}
            </button>
          ))}
        </div>
        <div className="btn-group">
          {['all', 'built', 'partial', 'planned'].map(f => (
            <button key={f} className={`btn btn-sm ${detFilter === f ? 'btn-info' : 'btn-outline-secondary'}`} onClick={() => setDetFilter(f)}>
              {f === 'all' ? 'All Status' : `${f.charAt(0).toUpperCase() + f.slice(1)} (${allIssues.filter(i => i.det_status === f).length})`}
            </button>
          ))}
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-body p-0">
          <table className="table table-sm table-striped mb-0">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Issue</th>
                <th>Severity</th>
                <th>Root Cause</th>
                <th>Detection</th>
                <th>Solution</th>
                <th>Project Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((issue, i) => (
                <tr key={i}>
                  <td className="small text-muted">{issue.layer}</td>
                  <td className="fw-semibold small">{issue.issue}</td>
                  <td><SeverityBadge severity={issue.severity} /></td>
                  <td className="small text-muted" style={{ maxWidth: 160 }}>{issue.root_cause}</td>
                  <td className="small text-muted" style={{ maxWidth: 160 }}>{issue.detection}</td>
                  <td className="small text-muted" style={{ maxWidth: 180 }}>{issue.solution}</td>
                  <td><DetectionBadge status={issue.det_status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="small text-muted">{filtered.length} of {allIssues.length} issues shown</div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (!data.available) return <div className="alert alert-info">No definitions data.</div>;

  const defs = data.definitions || [];
  const notes = data.clinical_notes || [];
  const refs = data.references || [];

  return (
    <div>
      <div className="card mb-4">
        <div className="card-header fw-semibold">Terminology</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead><tr><th>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {defs.map((d, i) => (
                <tr key={i}>
                  <td className="fw-semibold small" style={{ minWidth: 150 }}>{d.term}</td>
                  <td className="small">{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {notes.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">Clinical Notes</div>
          <div className="card-body">
            <ul className="mb-0">
              {notes.map((n, i) => <li key={i} className="small mb-1">{n}</li>)}
            </ul>
          </div>
        </div>
      )}

      {refs.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">References</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <tbody>
                {refs.map((r, i) => (
                  <tr key={i}><td className="small">{typeof r === 'string' ? r : r.title || JSON.stringify(r)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ProductionIssuesPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/production-issues/overview`).then(r => r.json()).then(setOverview).catch(() => setOverview({ error: 'Failed to load overview' }));
    fetch(`${API}/api/production-issues/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setBreakdown({ error: 'Failed to load breakdown' }));
    fetch(`${API}/api/production-issues/definitions`).then(r => r.json()).then(setDefinitions).catch(() => setDefinitions({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-4">
      <h3 className="mb-1">{'\ud83d\udea8'} Production Issues Monitor</h3>
      <p className="text-muted small mb-3">
        Enterprise agentic AI production issue catalog &mdash; severity scoring, detection coverage, layer-by-layer breakdown, and Pareto analysis across {overview?.summary?.total_layers || '...'} architectural layers.
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewPanel data={overview} />}
      {tab === 'breakdown' && <BreakdownPanel data={breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={definitions} />}
    </div>
  );
}
