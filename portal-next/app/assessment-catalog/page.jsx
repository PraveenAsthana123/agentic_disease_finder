'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const PRIORITY_COLORS = {
  'mandatory':       '#ef4444',
  'highly valuable': '#f97316',
  'very valuable':   '#f59e0b',
  'valuable':        '#22c55e',
  'preferred':       '#3b82f6',
};

const CATEGORY_COLORS = [
  '#6366f1','#3b82f6','#10b981','#f59e0b','#ec4899','#8b5cf6',
];

function StatCard({ label, value, sub, color = '#6366f1' }) {
  return (
    <div className="col-6 col-md mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h4 mb-0 fw-bold" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function PriorityBadge({ priority }) {
  const color = PRIORITY_COLORS[priority] || '#6b7280';
  return (
    <span className="badge" style={{ backgroundColor: color, fontSize: '0.7rem' }}>
      {priority}
    </span>
  );
}

function StatusBadge({ status }) {
  const color = status === 'built' ? '#22c55e' : '#f59e0b';
  return (
    <span className="badge" style={{ backgroundColor: color, fontSize: '0.7rem' }}>
      {status}
    </span>
  );
}

function MiniBar({ value, total, color }) {
  const pct = total ? Math.min(100, (value / total) * 100) : 0;
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 8, minWidth: 80 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color || '#6366f1' }} />
      </div>
      <span className="small fw-bold">{value}</span>
    </div>
  );
}

// ---- Tab: Overview ----
function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  if (!data.available) return <div className="alert alert-warning">{data.message || 'Unavailable'}</div>;

  const k = data.kpis || {};
  const cats = data.categories || [];
  const priDist = data.priority_distribution || [];
  const stDist = data.status_distribution || [];
  const totalPri = priDist.reduce((s, x) => s + x.count, 0);

  return (
    <div>
      {/* KPI row */}
      <div className="row g-2 mb-3">
        <StatCard label="Total Instruments" value={k.total_instruments} color="#6366f1" />
        <StatCard label="Built in Platform" value={k.built} sub="all live" color="#22c55e" />
        <StatCard label="Categories"        value={k.categories}         color="#3b82f6" />
        <StatCard label="Top-10 for Thesis" value={k.top10_for_thesis}   color="#f59e0b" />
        <StatCard label="Mandatory"         value={k.mandatory}          color="#ef4444" />
        <StatCard label="Specialists"       value={k.specialists}        color="#8b5cf6" />
      </div>

      <div className="row g-3 mb-3">
        {/* Category breakdown */}
        <div className="col-md-7">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small">Instruments by Category</div>
            <div className="card-body">
              {cats.map((cat, i) => (
                <div key={cat.category} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{cat.category}</span>
                    <span className="small text-muted">{cat.built}/{cat.total}</span>
                  </div>
                  <MiniBar value={cat.built} total={cat.total} color={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Priority distribution */}
        <div className="col-md-5">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small">Priority Distribution</div>
            <div className="card-body">
              {priDist.map(p => (
                <div key={p.priority} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <PriorityBadge priority={p.priority} />
                    <span className="small text-muted">{p.count} instruments</span>
                  </div>
                  <MiniBar value={p.count} total={totalPri} color={PRIORITY_COLORS[p.priority] || '#6b7280'} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Status summary */}
      <div className="card shadow-sm">
        <div className="card-header fw-bold small">Platform Status</div>
        <div className="card-body">
          <div className="row g-3">
            {stDist.map(s => (
              <div key={s.status} className="col-md-3 text-center">
                <div className="h3 fw-bold" style={{ color: s.status === 'built' ? '#22c55e' : '#f59e0b' }}>
                  {s.count}
                </div>
                <StatusBadge status={s.status} />
              </div>
            ))}
            <div className="col-md-9">
              <div className="alert alert-success mb-0 py-2">
                <strong>✅ 100% Coverage:</strong> All {k.total_instruments} instruments are built and live in the platform.
                The epilepsy EEG thesis has complete assessment coverage across {k.categories} clinical specialties
                with {k.research_variables} research variables extracted for AI model features.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---- Tab: Top 10 Thesis ----
function Top10Tab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  const instruments = data.instruments || [];
  const top10 = instruments.filter(i => i.in_top10);

  return (
    <div>
      <div className="alert alert-info mb-3 py-2">
        <strong>Top-10 for Thesis:</strong> These instruments provide the highest clinical signal density
        for the epilepsy EEG AI governance study. All are built and data-complete in the platform.
      </div>
      <div className="row g-3">
        {top10.map((inst, idx) => (
          <div key={inst.name} className="col-md-6">
            <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${PRIORITY_COLORS[inst.priority] || '#6b7280'}` }}>
              <div className="card-body">
                <div className="d-flex justify-content-between align-items-start mb-1">
                  <h6 className="card-title mb-0 fw-bold">
                    <span className="text-muted me-1">#{idx + 1}</span> {inst.name}
                  </h6>
                  <PriorityBadge priority={inst.priority} />
                </div>
                <div className="text-muted small mb-1">{inst.purpose}</div>
                <div className="d-flex gap-2 flex-wrap mt-1">
                  <span className="badge bg-light text-dark" style={{ fontSize: '0.65rem' }}>📤 {inst.output}</span>
                  <span className="badge bg-light text-dark" style={{ fontSize: '0.65rem' }}>👤 {inst.specialist}</span>
                  <span className="badge bg-light text-dark" style={{ fontSize: '0.65rem' }}>📂 {inst.category}</span>
                </div>
                {inst.note && (
                  <div className="mt-2 small text-success">
                    <em>{inst.note.length > 120 ? inst.note.slice(0, 120) + '…' : inst.note}</em>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---- Tab: Full Catalog ----
function CatalogTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  const [search, setSearch] = useState('');
  const [filterCat, setFilterCat] = useState('All');
  const [filterPri, setFilterPri] = useState('All');

  const instruments = data.instruments || [];
  const categories = ['All', ...new Set(instruments.map(i => i.category))];
  const priorities = ['All', ...new Set(instruments.map(i => i.priority))];

  const filtered = instruments.filter(i => {
    const matchSearch = !search || i.name.toLowerCase().includes(search.toLowerCase()) ||
      i.purpose.toLowerCase().includes(search.toLowerCase());
    const matchCat = filterCat === 'All' || i.category === filterCat;
    const matchPri = filterPri === 'All' || i.priority === filterPri;
    return matchSearch && matchCat && matchPri;
  });

  return (
    <div>
      <div className="row g-2 mb-3">
        <div className="col-md-4">
          <input className="form-control form-control-sm" placeholder="Search instrument or purpose…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <div className="col-md-4">
          <select className="form-select form-select-sm" value={filterCat} onChange={e => setFilterCat(e.target.value)}>
            {categories.map(c => <option key={c}>{c}</option>)}
          </select>
        </div>
        <div className="col-md-4">
          <select className="form-select form-select-sm" value={filterPri} onChange={e => setFilterPri(e.target.value)}>
            {priorities.map(p => <option key={p}>{p}</option>)}
          </select>
        </div>
      </div>
      <div className="small text-muted mb-2">Showing {filtered.length} of {instruments.length} instruments</div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-dark">
            <tr>
              <th>Instrument</th>
              <th>Purpose</th>
              <th>Output</th>
              <th>Specialist</th>
              <th>Priority</th>
              <th>Status</th>
              <th>Thesis</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(inst => (
              <tr key={inst.name}>
                <td className="fw-semibold small">{inst.name}</td>
                <td className="small text-muted">{inst.purpose}</td>
                <td className="small">{inst.output}</td>
                <td className="small">{inst.specialist}</td>
                <td><PriorityBadge priority={inst.priority} /></td>
                <td><StatusBadge status={inst.status} /></td>
                <td className="text-center">{inst.in_top10 ? '⭐' : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---- Tab: By Category ----
function ByCategoryTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  const instruments = data.instruments || [];

  // Group by category
  const grouped = {};
  instruments.forEach(inst => {
    if (!grouped[inst.category]) grouped[inst.category] = [];
    grouped[inst.category].push(inst);
  });

  return (
    <div>
      {Object.entries(grouped).map(([cat, items], ci) => (
        <div key={cat} className="card shadow-sm mb-3">
          <div className="card-header d-flex justify-content-between align-items-center"
               style={{ backgroundColor: CATEGORY_COLORS[ci % CATEGORY_COLORS.length] + '22',
                        borderLeft: `4px solid ${CATEGORY_COLORS[ci % CATEGORY_COLORS.length]}` }}>
            <span className="fw-bold small">{cat}</span>
            <span className="badge bg-secondary">{items.length} instruments</span>
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0 align-middle">
                <thead className="table-light">
                  <tr>
                    <th className="ps-3">Instrument</th>
                    <th>Purpose</th>
                    <th>Output</th>
                    <th>Specialist</th>
                    <th>Priority</th>
                    <th>Top-10</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map(inst => (
                    <tr key={inst.name}>
                      <td className="ps-3 fw-semibold small">{inst.name}</td>
                      <td className="small text-muted">{inst.purpose}</td>
                      <td className="small">{inst.output}</td>
                      <td className="small">{inst.specialist}</td>
                      <td><PriorityBadge priority={inst.priority} /></td>
                      <td className="text-center">{inst.in_top10 ? '⭐' : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ---- Tab: Definitions ----
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" /></div>;

  const statusLegend  = data.status_legend     || [];
  const priorityLeg   = data.priority_legend   || [];
  const glossary      = data.glossary          || [];
  const clinicalNotes = data.clinical_notes    || [];
  const references    = data.references        || [];

  return (
    <div>
      <div className="row g-3">
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small">Status Legend</div>
            <div className="card-body">
              {statusLegend.map(s => (
                <div key={s.status} className="mb-2">
                  <StatusBadge status={s.status} />
                  <p className="small text-muted mt-1 mb-0">{s.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-8">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small">Priority Legend</div>
            <div className="card-body">
              <div className="row g-2">
                {priorityLeg.map(p => (
                  <div key={p.priority} className="col-md-6">
                    <div className="d-flex gap-2 align-items-start">
                      <PriorityBadge priority={p.priority} />
                      <p className="small text-muted mb-0">{p.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mt-3">
        <div className="card-header fw-bold small">Instrument Glossary ({glossary.length} terms)</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th className="ps-3" style={{ width: '15%' }}>Abbreviation</th>
                  <th>Definition</th>
                  <th style={{ width: '20%' }}>Score Range</th>
                  <th style={{ width: '15%' }}>Specialist</th>
                </tr>
              </thead>
              <tbody>
                {glossary.map(g => (
                  <tr key={g.term}>
                    <td className="ps-3 fw-bold text-primary small">{g.term}</td>
                    <td className="small">{g.definition}</td>
                    <td className="small text-muted">{g.score_range || '—'}</td>
                    <td className="small">{g.specialist || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {clinicalNotes.length > 0 && (
        <div className="card shadow-sm mt-3">
          <div className="card-header fw-bold small">Clinical Notes</div>
          <div className="card-body">
            <ul className="mb-0">
              {clinicalNotes.map((n, i) => <li key={i} className="small">{n}</li>)}
            </ul>
          </div>
        </div>
      )}

      {references.length > 0 && (
        <div className="card shadow-sm mt-3">
          <div className="card-header fw-bold small">References</div>
          <div className="card-body">
            <ul className="mb-0">
              {references.map((r, i) => <li key={i} className="small text-muted">{r}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

// ---- Main page ----
export default function AssessmentCatalogPage() {
  const [tab, setTab]         = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError]     = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/assessment-catalog/overview`).then(r => r.json()),
      fetch(`${API}/api/assessment-catalog/breakdown`).then(r => r.json()),
      fetch(`${API}/api/assessment-catalog/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  const tabs = [
    { id: 'overview',    label: '📊 Overview'     },
    { id: 'top10',       label: '⭐ Top-10 Thesis' },
    { id: 'catalog',     label: '📋 Full Catalog'  },
    { id: 'by-category', label: '📂 By Category'  },
    { id: 'definitions', label: '📖 Definitions'  },
  ];

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-0">📊 Clinical Assessment Catalog — Epilepsy EEG Thesis</h4>
        <p className="text-muted small mb-0">
          26 clinical assessment instruments across 6 specialties · all built in platform ·
          10 top-priority instruments for thesis · 4 mandatory
        </p>
      </div>

      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active fw-bold' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewTab    data={overview}    />}
      {tab === 'top10'       && <Top10Tab       data={breakdown}   />}
      {tab === 'catalog'     && <CatalogTab     data={breakdown}   />}
      {tab === 'by-category' && <ByCategoryTab  data={breakdown}   />}
      {tab === 'definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
