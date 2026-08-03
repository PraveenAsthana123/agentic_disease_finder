'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Item Breakdown' },
  { id: 'definitions', label: 'Definitions' },
];

const STATUS_COLORS = { present: 'success', partial: 'warning', missing: 'danger' };
const STATUS_ICONS = { present: '\u2705', partial: '\u26a0\ufe0f', missing: '\u274c' };

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
  const color = STATUS_COLORS[status] || 'secondary';
  return <span className={`badge bg-${color}`}>{status}</span>;
}

function CompletionBar({ present, partial, missing, total }) {
  const pPct = Math.round((present / Math.max(total, 1)) * 100);
  const partPct = Math.round((partial / Math.max(total, 1)) * 100);
  const mPct = 100 - pPct - partPct;
  return (
    <div className="progress" style={{ height: 18 }}>
      <div className="progress-bar bg-success" style={{ width: `${pPct}%` }} title={`Present: ${present}`}>{pPct > 8 ? `${pPct}%` : ''}</div>
      <div className="progress-bar bg-warning" style={{ width: `${partPct}%` }} title={`Partial: ${partial}`}>{partPct > 8 ? `${partPct}%` : ''}</div>
      <div className="progress-bar bg-danger" style={{ width: `${mPct}%` }} title={`Missing: ${missing}`}>{mPct > 8 ? `${mPct}%` : ''}</div>
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const kpis = data.kpis || {};
  const cats = data.category_breakdown || [];
  const dist = data.status_distribution || {};
  const tiers = data.tier_coverage || [];
  const cg = data.control_groups_summary || {};

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Data Items" value={kpis.total_items} color="primary" sub={`across ${kpis.categories} categories`} />
        <KPI label="Present" value={kpis.present} color="success" sub={`${kpis.completeness_pct}% complete`} />
        <KPI label="Partial" value={kpis.partial} color="warning" sub="schema exists, needs data" />
        <KPI label="Missing" value={kpis.missing} color="danger" sub="not yet implemented" />
      </div>

      <div className="row mb-4">
        <div className="col-md-8 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Category Completeness</div>
            <div className="card-body">
              {cats.map(c => (
                <div key={c.category} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="fw-semibold">{c.category}</span>
                    <span className="text-muted">{c.present}/{c.total} present</span>
                  </div>
                  <CompletionBar present={c.present} partial={c.partial} missing={c.missing} total={c.total} />
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Status Distribution</div>
            <div className="card-body">
              <div className="mb-3">
                <div className="d-flex justify-content-between mb-1">
                  <span className="badge bg-success">Present</span>
                  <span className="fw-bold">{dist.present}</span>
                </div>
                <div className="d-flex justify-content-between mb-1">
                  <span className="badge bg-warning text-dark">Partial</span>
                  <span className="fw-bold">{dist.partial}</span>
                </div>
                <div className="d-flex justify-content-between mb-1">
                  <span className="badge bg-danger">Missing</span>
                  <span className="fw-bold">{dist.missing}</span>
                </div>
              </div>

              <hr />
              <div className="small fw-semibold mb-2">Data Tiers</div>
              {tiers.map(t => (
                <div key={t.tier} className="d-flex justify-content-between small mb-1">
                  <span>{t.label}</span>
                  <span className="fw-bold">{t.count} items</span>
                </div>
              ))}

              {cg.most_valuable && cg.most_valuable.length > 0 && (
                <>
                  <hr />
                  <div className="small fw-semibold mb-2">Control Groups</div>
                  <div className="small text-muted mb-1">{cg.note}</div>
                  <div className="small">Min cohorts: {cg.minimum_cohorts} | Ideal: {cg.ideal_cohorts}</div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {data.single_most_important && (
        <div className="alert alert-info small">
          <strong>Single Most Important:</strong> {data.single_most_important}
        </div>
      )}
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  const [filter, setFilter] = useState('all');

  const allItems = data.all_items || [];
  const perCat = data.per_category || [];
  const filtered = filter === 'all' ? allItems : allItems.filter(i => i.status === filter);

  return (
    <div>
      <div className="btn-group mb-3">
        {['all', 'present', 'partial', 'missing'].map(f => (
          <button key={f} className={`btn btn-sm ${filter === f ? 'btn-primary' : 'btn-outline-secondary'}`} onClick={() => setFilter(f)}>
            {f === 'all' ? `All (${allItems.length})` : `${f.charAt(0).toUpperCase() + f.slice(1)} (${allItems.filter(i => i.status === f).length})`}
          </button>
        ))}
      </div>

      <div className="card mb-4">
        <div className="card-body p-0">
          <table className="table table-sm table-striped mb-0">
            <thead><tr><th>Item</th><th>Category</th><th>Status</th><th>Note</th></tr></thead>
            <tbody>
              {filtered.map((item, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{item.name}</td>
                  <td className="small text-muted">{item.category}</td>
                  <td><StatusBadge status={item.status} /></td>
                  <td className="small text-muted" style={{ maxWidth: 300 }}>{item.note || '\u2014'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {data.top10_artifacts && data.top10_artifacts.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Top 10 Artifacts</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead><tr><th>#</th><th>Artifact</th></tr></thead>
              <tbody>
                {data.top10_artifacts.map((a, i) => (
                  <tr key={i}><td>{i + 1}</td><td className="small">{typeof a === 'string' ? a : a.name || JSON.stringify(a)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {data.technician_deliverables && data.technician_deliverables.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Technician Deliverables</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead><tr><th>Deliverable</th></tr></thead>
              <tbody>
                {data.technician_deliverables.map((d, i) => (
                  <tr key={i}><td className="small">{typeof d === 'string' ? d : d.name || JSON.stringify(d)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;

  const levels = data.status_levels || [];
  const tiers = data.data_tiers || [];
  const glossary = data.glossary || [];
  const refs = data.references || [];

  return (
    <div>
      <div className="card mb-4">
        <div className="card-header fw-semibold">Status Levels</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead><tr><th>Status</th><th>Description</th></tr></thead>
            <tbody>
              {levels.map((l, i) => (
                <tr key={i}>
                  <td><span className="badge" style={{ backgroundColor: l.color, color: '#fff' }}>{l.label}</span></td>
                  <td className="small">{l.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-header fw-semibold">Data Tiers</div>
        <div className="card-body">
          {tiers.map((t, i) => (
            <div key={i} className="mb-3">
              <div className="fw-semibold">{t.label}</div>
              <div className="small text-muted mb-1">{t.description}</div>
              <div className="small">{(t.items || []).join(' \u2022 ')}</div>
            </div>
          ))}
        </div>
      </div>

      {glossary.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">Glossary</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead><tr><th>Term</th><th>Definition</th></tr></thead>
              <tbody>
                {glossary.map((g, i) => (
                  <tr key={i}><td className="fw-semibold small">{g.term}</td><td className="small">{g.definition}</td></tr>
                ))}
              </tbody>
            </table>
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

export default function DataRequirementsPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/data-requirements/overview`).then(r => r.json()).then(setOverview).catch(() => setOverview({ error: 'Failed to load overview' }));
    fetch(`${API}/api/data-requirements/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setBreakdown({ error: 'Failed to load breakdown' }));
    fetch(`${API}/api/data-requirements/definitions`).then(r => r.json()).then(setDefinitions).catch(() => setDefinitions({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-4">
      <h3 className="mb-1">{'\ud83d\udcca'} Data Requirements Gap Dashboard</h3>
      <p className="text-muted small mb-3">
        Dataset completeness tracker — what a top-tier epilepsy AI dataset needs vs what the project has.
        {overview?.updated_at && <span> Updated: {overview.updated_at}</span>}
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
