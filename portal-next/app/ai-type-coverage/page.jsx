'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: '📊 Overview' },
  { id: 'breakdown', label: '🗂 Coverage Matrix' },
  { id: 'all-types', label: '📋 All Types' },
  { id: 'definitions', label: '📖 Definitions' },
];

const STATUS_COLOR = { built: 'success', 'not-pulled': 'secondary', scaffold: 'warning', planned: 'info' };
const STATUS_ICON = { built: '✅', 'not-pulled': '—', scaffold: '🔧', planned: '📅' };

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

function StatusBadge({ status }) {
  return (
    <span className={`badge bg-${STATUS_COLOR[status] || 'secondary'}`}>
      {STATUS_ICON[status] || '?'} {status}
    </span>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const s = data.summary || {};
  const statusDist = data.status_distribution || [];
  const catDist = data.category_distribution || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total AI Types" value={s.total_types} color="primary" sub="cataloged from insur_project" />
        <KPI label="Built" value={s.built} color="success" sub="in agenticfinder" />
        <KPI label="Coverage" value={`${s.coverage_pct}%`} color={s.coverage_pct >= 50 ? 'success' : s.coverage_pct >= 25 ? 'warning' : 'danger'} sub="of 201 types" />
        <KPI label="Not Pulled" value={s.not_pulled} color="secondary" sub="not applicable / pending" />
      </div>

      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card mb-3">
            <div className="card-header fw-semibold">Status Distribution</div>
            <div className="card-body">
              {statusDist.map((s, i) => (
                <div key={i} className="d-flex justify-content-between align-items-center mb-2">
                  <StatusBadge status={s.name} />
                  <span className="fw-semibold">{s.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card mb-3">
            <div className="card-header fw-semibold">Coverage Progress</div>
            <div className="card-body">
              <div className="mb-3">
                <div className="d-flex justify-content-between small mb-1">
                  <span>Built ({s.built} / {s.total_types})</span>
                  <span>{s.coverage_pct}%</span>
                </div>
                <div className="progress" style={{ height: 20 }}>
                  <div
                    className="progress-bar bg-success"
                    style={{ width: `${s.coverage_pct || 0}%` }}
                  >
                    {s.coverage_pct}%
                  </div>
                </div>
              </div>
              <div className="text-muted small">
                Source: <strong>{data.source}</strong><br />
                Project: <strong>{data.project}</strong><br />
                Updated: {data.updated_at?.slice(0, 10)}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header fw-semibold">Built AI Types by Category</div>
        <div className="card-body">
          <BarChart data={catDist} labelKey="name" valueKey="value" color="success" />
        </div>
      </div>
    </div>
  );
}

function CoverageMatrixPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const perCategory = data.per_category || [];

  return (
    <div>
      <div className="alert alert-success small mb-3">
        <strong>Built:</strong> {data.built_count} AI types implemented with real logic and verified endpoints.
        <strong className="ms-3">Not Pulled:</strong> {data.not_pulled_count} types from source catalog not applicable to epilepsy EEG.
      </div>
      <div className="row">
        {perCategory.map((cat, i) => (
          <div key={i} className="col-md-6 mb-3">
            <div className="card h-100">
              <div className="card-header fw-semibold d-flex justify-content-between align-items-center">
                <span>{cat.category}</span>
                <span className="badge bg-success">{cat.count} built</span>
              </div>
              <div className="card-body p-2">
                {(cat.types || []).map((t, j) => (
                  <div key={j} className="mb-2 p-2 border rounded" style={{ fontSize: '0.78rem' }}>
                    <div className="d-flex align-items-start gap-2">
                      <span className="badge bg-success" style={{ whiteSpace: 'nowrap' }}>✅ built</span>
                      <div>
                        <div className="fw-semibold">{t.type}</div>
                        {t.note && <div className="text-muted">{t.note.slice(0, 120)}{t.note.length > 120 ? '…' : ''}</div>}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AllTypesPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  const perCategory = data.per_category || [];
  const notPulled = data.not_pulled_list || [];
  const scaffold = data.scaffold_list || [];
  const planned = data.planned_list || [];

  // Combine all types into one flat list
  const builtTypes = perCategory.flatMap(cat =>
    (cat.types || []).map(t => ({ ...t, status: 'built', category: cat.category }))
  );
  const allTypes = [
    ...builtTypes,
    ...notPulled.map(t => ({ ...t, status: 'not-pulled', category: '' })),
    ...scaffold.map(t => ({ ...t, status: 'scaffold', category: '' })),
    ...planned.map(t => ({ ...t, status: 'planned', category: '' })),
  ];

  const filtered = allTypes.filter(t => {
    const matchStatus = statusFilter === 'all' || t.status === statusFilter;
    const q = search.toLowerCase();
    const matchSearch = !q || t.type?.toLowerCase().includes(q) || t.category?.toLowerCase().includes(q);
    return matchStatus && matchSearch;
  });

  return (
    <div>
      <div className="d-flex gap-2 mb-3 flex-wrap">
        <input
          className="form-control form-control-sm"
          placeholder="Search AI type…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{ maxWidth: 240 }}
        />
        <select
          className="form-select form-select-sm"
          value={statusFilter}
          onChange={e => setStatusFilter(e.target.value)}
          style={{ maxWidth: 180 }}
        >
          <option value="all">All Statuses</option>
          <option value="built">Built ({data.built_count})</option>
          <option value="not-pulled">Not Pulled ({data.not_pulled_count})</option>
          <option value="scaffold">Scaffold ({data.scaffold_count})</option>
          <option value="planned">Planned ({data.planned_count})</option>
        </select>
        <span className="text-muted small align-self-center">{filtered.length} of {allTypes.length}</span>
      </div>

      <div className="table-responsive" style={{ maxHeight: 600, overflowY: 'auto' }}>
        <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.78rem' }}>
          <thead className="table-light sticky-top">
            <tr>
              <th>AI Type</th>
              <th>Status</th>
              <th>Category</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((t, i) => (
              <tr key={i}>
                <td className="fw-semibold font-monospace">{t.type}</td>
                <td><StatusBadge status={t.status} /></td>
                <td>{t.category || <span className="text-muted">—</span>}</td>
                <td className="text-muted">{t.note?.slice(0, 100) || '—'}{t.note?.length > 100 ? '…' : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const statusLegend = data.status_legend || [];
  const glossary = data.glossary || [];
  const clinicalNotes = data.clinical_notes || [];
  const references = data.references || [];

  return (
    <div>
      <div className="card mb-3">
        <div className="card-header fw-semibold">Status Legend</div>
        <div className="card-body">
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead><tr><th>Status</th><th>Meaning</th></tr></thead>
              <tbody>
                {statusLegend.map((s, i) => (
                  <tr key={i}>
                    <td><StatusBadge status={s.status} /></td>
                    <td className="small">{s.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {glossary.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Glossary</div>
          <div className="card-body">
            {glossary.map((g, i) => (
              <div key={i} className="mb-2">
                <span className="fw-semibold">{g.term}</span>
                <span className="text-muted"> — {g.definition}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {clinicalNotes.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Clinical Notes</div>
          <div className="card-body">
            <ul className="mb-0">
              {clinicalNotes.map((n, i) => <li key={i} className="small mb-1">{n}</li>)}
            </ul>
          </div>
        </div>
      )}

      {references.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">References</div>
          <div className="card-body">
            <ul className="mb-0">
              {references.map((r, i) => <li key={i} className="small mb-1">{r}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default function AITypeCoveragePage() {
  const [tab, setTab] = useState('overview');
  const [data, setData] = useState({});

  useEffect(() => {
    // both 'breakdown' and 'all-types' share the /breakdown endpoint — store under 'breakdown'
    const storeKey = tab === 'all-types' ? 'breakdown' : tab;
    const endpoint = storeKey;
    if (!data[storeKey]) {
      fetch(`${API}/api/ai-type-coverage/${endpoint}`)
        .then(r => r.json())
        .then(d => setData(prev => ({ ...prev, [storeKey]: d })))
        .catch(() => setData(prev => ({ ...prev, [storeKey]: { error: 'Failed to load' } })));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: '1.6rem' }}>🤖</span>
        <div>
          <h4 className="mb-0 fw-bold">AI Type Coverage Dashboard</h4>
          <div className="text-muted small">
            201 AI types cataloged · 46 built in agenticfinder (epilepsy EEG) · 22.9% coverage
          </div>
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
      {tab === 'breakdown' && <CoverageMatrixPanel data={data.breakdown} />}
      {tab === 'all-types' && <AllTypesPanel data={data.breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={data.definitions} />}
    </div>
  );
}
