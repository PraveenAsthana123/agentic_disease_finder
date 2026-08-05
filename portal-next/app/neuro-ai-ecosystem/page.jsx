'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLOR = {
  built:      'success',
  installed:  'primary',
  external:   'warning',
  commercial: 'secondary',
};

const STATUS_ICON = {
  built:      '✅',
  installed:  '📦',
  external:   '🔗',
  commercial: '💼',
};

function KpiCard({ label, value, color }) {
  return (
    <div className="col">
      <div className={`card border-${color} h-100`}>
        <div className="card-body text-center py-2">
          <div className={`fs-4 fw-bold text-${color}`}>{value ?? '—'}</div>
          <div className="small text-muted">{label}</div>
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  const color = STATUS_COLOR[status] || 'secondary';
  const icon  = STATUS_ICON[status]  || '•';
  return (
    <span className={`badge bg-${color} me-1`}>
      {icon} {status}
    </span>
  );
}

function RatingStars({ count }) {
  if (!count) return <span className="text-muted small">—</span>;
  return (
    <span>
      {'★'.repeat(count)}{'☆'.repeat(5 - count)}
    </span>
  );
}

export default function NeuroAiEcosystemPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [err,  setErr]  = useState(null);
  const [catFilter, setCatFilter] = useState('All');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/neuro-ai-ecosystem/overview`).then(r => r.json()),
      fetch(`${API}/api/neuro-ai-ecosystem/breakdown`).then(r => r.json()),
      fetch(`${API}/api/neuro-ai-ecosystem/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err)  return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-4">Loading Neuro AI Ecosystem…</div>;

  const s = ov.summary || {};
  const cats = (bd?.per_category || []);
  const allCatNames = ['All', ...cats.map(c => c.category)];
  const filteredCats = catFilter === 'All' ? cats : cats.filter(c => c.category === catFilter);

  const TABS = [
    { id: 'overview',    label: '📊 Overview' },
    { id: 'by-category', label: '🗂 By Category' },
    { id: 'recommended', label: '⭐ Recommended Stack' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">🧬 Neuro AI Ecosystem</h4>
      <p className="text-muted small mb-3">
        Open-source epilepsy / EEG / cognitive / neuropsychiatry platform catalog —
        64 tools across 10 categories. status: built (live) / installed / external / commercial.
      </p>

      {/* KPI Row */}
      <div className="row row-cols-3 row-cols-md-6 g-2 mb-3">
        <KpiCard label="Total Tools"  value={s.total_tools}  color="primary"   />
        <KpiCard label="Built"        value={s.built}        color="success"   />
        <KpiCard label="Installed"    value={s.installed}    color="primary"   />
        <KpiCard label="External"     value={s.external}     color="warning"   />
        <KpiCard label="Commercial"   value={s.commercial}   color="secondary" />
        <KpiCard label="Categories"   value={s.categories}   color="info"      />
      </div>

      {/* Tabs */}
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

      {/* ── Overview tab ── */}
      {tab === 'overview' && (
        <div>
          {/* Status distribution bar */}
          <div className="card mb-3">
            <div className="card-header fw-semibold">Status Distribution</div>
            <div className="card-body">
              <div className="d-flex gap-2 mb-2 flex-wrap">
                {(ov.status_distribution || []).map(sd => (
                  <span key={sd.name} className={`badge bg-${STATUS_COLOR[sd.name] || 'secondary'} fs-6`}>
                    {STATUS_ICON[sd.name] || '•'} {sd.name}: {sd.value}
                  </span>
                ))}
              </div>
              {/* Progress bar */}
              <div className="progress" style={{ height: 20 }}>
                {(ov.status_distribution || []).map(sd => {
                  const pct = s.total_tools ? ((sd.value / s.total_tools) * 100).toFixed(1) : 0;
                  return (
                    <div
                      key={sd.name}
                      className={`progress-bar bg-${STATUS_COLOR[sd.name] || 'secondary'}`}
                      style={{ width: `${pct}%` }}
                      title={`${sd.name}: ${sd.value} (${pct}%)`}
                    >
                      {pct > 6 ? `${sd.name} ${pct}%` : ''}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Category summary table */}
          <div className="card mb-3">
            <div className="card-header fw-semibold">Category Summary</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Category</th>
                    <th className="text-center">Tools</th>
                    <th>Status breakdown</th>
                  </tr>
                </thead>
                <tbody>
                  {cats.map(cat => (
                    <tr key={cat.category}>
                      <td className="fw-semibold small">{cat.category}</td>
                      <td className="text-center">{cat.tool_count}</td>
                      <td>
                        {Object.entries(cat.status_counts || {}).map(([st, cnt]) => (
                          <span key={st} className={`badge bg-${STATUS_COLOR[st] || 'secondary'} me-1`}>
                            {st}: {cnt}
                          </span>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── By Category tab ── */}
      {tab === 'by-category' && (
        <div>
          {/* Category filter */}
          <div className="d-flex flex-wrap gap-2 mb-3">
            {allCatNames.map(n => (
              <button
                key={n}
                className={`btn btn-sm ${catFilter === n ? 'btn-primary' : 'btn-outline-secondary'}`}
                onClick={() => setCatFilter(n)}
              >
                {n === 'All' ? 'All' : n.split(' ')[0]}
              </button>
            ))}
          </div>

          {filteredCats.map(cat => (
            <div key={cat.category} className="card mb-3">
              <div className="card-header d-flex justify-content-between align-items-center">
                <span className="fw-semibold">{cat.category}</span>
                <span className="badge bg-primary">{cat.tool_count} tools</span>
              </div>
              {cat.note && <div className="card-body pb-0 small text-muted">{cat.note}</div>}
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Tool</th>
                      <th>Purpose</th>
                      <th>Status</th>
                      <th>Rating</th>
                      <th>Domain</th>
                      <th>Endpoints</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(cat.tools || []).map(tool => (
                      <tr key={tool.name}>
                        <td className="fw-semibold small">{tool.name}</td>
                        <td className="small">{tool.purpose}</td>
                        <td><StatusBadge status={tool.status} /></td>
                        <td><RatingStars count={tool.rating} /></td>
                        <td className="small text-muted">{tool.domain || '—'}</td>
                        <td className="small text-info">{tool.endpoints || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Recommended Stack tab ── */}
      {tab === 'recommended' && (
        <div className="card">
          <div className="card-header fw-semibold">⭐ Recommended Open-Source Neuro AI Stack</div>
          <div className="card-body">
            <p className="text-muted small mb-3">
              Minimum viable open-source platform for epilepsy EEG AI research.
              Combines the best tools from each domain into a coherent stack.
            </p>
            <div className="row row-cols-1 row-cols-md-2 g-3">
              {Object.entries(ov.recommended_stack || {}).map(([domain, tool]) => (
                <div key={domain} className="col">
                  <div className="d-flex align-items-start gap-2 p-2 border rounded">
                    <span className="badge bg-info text-dark mt-1 text-wrap" style={{ minWidth: 140 }}>
                      {domain}
                    </span>
                    <span className="fw-semibold small">{tool}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions tab ── */}
      {tab === 'definitions' && defs && (
        <div>
          {/* Status legend */}
          <div className="card mb-3">
            <div className="card-header fw-semibold">Status Legend</div>
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Status</th><th>Description</th></tr>
                </thead>
                <tbody>
                  {(defs.status_legend || []).map(sl => (
                    <tr key={sl.status}>
                      <td><StatusBadge status={sl.status} /></td>
                      <td className="small">{sl.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Glossary */}
          <div className="card mb-3">
            <div className="card-header fw-semibold">Glossary</div>
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Term</th><th>Definition</th></tr>
                </thead>
                <tbody>
                  {(defs.glossary || []).map(g => (
                    <tr key={g.term}>
                      <td className="fw-semibold small">{g.term}</td>
                      <td className="small">{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Clinical notes */}
          {(defs.clinical_notes || []).length > 0 && (
            <div className="card mb-3">
              <div className="card-header fw-semibold">Clinical Notes</div>
              <ul className="list-group list-group-flush">
                {defs.clinical_notes.map((n, i) => (
                  <li key={i} className="list-group-item small">{n}</li>
                ))}
              </ul>
            </div>
          )}

          {/* References */}
          {(defs.references || []).length > 0 && (
            <div className="card">
              <div className="card-header fw-semibold">References</div>
              <ul className="list-group list-group-flush">
                {defs.references.map((r, i) => (
                  <li key={i} className="list-group-item small text-muted">{r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
