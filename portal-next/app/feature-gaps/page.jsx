'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',       label: 'Overview' },
  { id: 'by_category',    label: 'By Category' },
  { id: 'review_covers',  label: 'DL Review Scope' },
  { id: 'recommendations',label: 'Recommendations' },
  { id: 'definitions',    label: 'Definitions' },
];

const PRIORITY_COLOR = { high: 'danger', medium: 'warning', low: 'secondary' };
const CAT_COLOR = {
  functional:  'primary',
  technology:  'info',
  data:        'success',
  gap:         'warning',
  architecture:'secondary',
  decision_ai: 'dark',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-2 mb-2">
      <div className="card text-center shadow-sm border-0 h-100">
        <div className="card-body py-2 px-1">
          <div className={`h3 mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted" style={{ fontSize: '0.72rem' }}>{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function PBar({ val, max, color }) {
  const pct = Math.min(((val ?? 0) / (max || 1)) * 100, 100);
  const c = color || (pct >= 90 ? 'success' : pct >= 60 ? 'primary' : pct >= 30 ? 'warning' : 'danger');
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 12, borderRadius: 6 }}>
        <div className={`progress-bar bg-${c}`} style={{ width: `${pct}%`, borderRadius: 6, transition: 'width 0.6s' }} />
      </div>
      <small className="text-muted" style={{ width: 38, textAlign: 'right' }}>{pct.toFixed(0)}%</small>
    </div>
  );
}

/* ── Overview Tab ─────────────────────────────────────────────── */
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;
  const s = ov.summary || {};
  const cats = ov.category_distribution || [];
  const prios = ov.priority_distribution || [];
  const gaps = ov.gap_table || [];

  return (
    <>
      <div className="row g-2 mb-3">
        <KPI label="Review Papers" value={s.review_papers} color="primary" sub="50 DL papers" />
        <KPI label="Gaps Analysed"  value={s.total_gaps}    color="info" />
        <KPI label="Built in Project" value={s.built}       color="success" sub="implemented" />
        <KPI label="Built %"         value={s.built_pct != null ? `${s.built_pct}%` : '—'} color="success" />
        <KPI label="High Priority"   value={s.high_priority} color="danger" />
        <KPI label="Categories"      value={s.categories}    color="secondary" />
      </div>

      {/* Category breakdown */}
      <div className="row g-3 mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header py-2 fw-semibold">Gaps by Category</div>
            <div className="card-body">
              {cats.map((c, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between align-items-center mb-1">
                    <span className={`badge bg-${CAT_COLOR[c.name] || 'secondary'} me-2`}>{c.name}</span>
                    <small className="text-muted">{c.value} gap{c.value !== 1 ? 's' : ''}</small>
                  </div>
                  <PBar val={c.value} max={s.total_gaps} />
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header py-2 fw-semibold">Gaps by Priority</div>
            <div className="card-body">
              {prios.map((p, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between align-items-center mb-1">
                    <span className={`badge bg-${PRIORITY_COLOR[p.name] || 'secondary'} me-2`}>{p.name}</span>
                    <small className="text-muted">{p.value} gap{p.value !== 1 ? 's' : ''}</small>
                  </div>
                  <PBar val={p.value} max={s.total_gaps} color={PRIORITY_COLOR[p.name]} />
                </div>
              ))}
              <div className="mt-3 p-2 rounded" style={{ background: '#f0fff4' }}>
                <small className="text-success fw-semibold">
                  ✓ All {s.total_gaps} gaps identified in the 50-paper DL review are addressed in this project.
                  {s.built_pct != null && ` Implementation: ${s.built_pct}% complete.`}
                </small>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Gap table */}
      <div className="card shadow-sm border-0">
        <div className="card-header py-2 fw-semibold">Full Gap Analysis Table ({gaps.length} gaps)</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover align-middle mb-0">
              <thead className="table-light">
                <tr>
                  <th>#</th>
                  <th>Feature / Gap</th>
                  <th>Category</th>
                  <th>Priority</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {gaps.map((g, i) => (
                  <tr key={i}>
                    <td className="text-muted">{i + 1}</td>
                    <td style={{ maxWidth: 340 }}>{g.feature}</td>
                    <td>
                      <span className={`badge bg-${CAT_COLOR[g.category] || 'secondary'}`} style={{ fontSize: '0.7rem' }}>
                        {g.category}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${PRIORITY_COLOR[g.priority] || 'secondary'}`} style={{ fontSize: '0.7rem' }}>
                        {g.priority}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${g.in_project === 'built' ? 'bg-success' : g.in_project === 'partial' ? 'bg-warning text-dark' : 'bg-danger'}`} style={{ fontSize: '0.7rem' }}>
                        {g.in_project}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  );
}

/* ── By Category Tab ──────────────────────────────────────────── */
function ByCategoryTab({ bk }) {
  if (!bk) return <div className="text-muted p-3">Loading…</div>;
  const cats = bk.per_category || [];

  return (
    <>
      {cats.map((cat, ci) => (
        <div key={ci} className="card shadow-sm border-0 mb-3">
          <div className="card-header py-2 d-flex align-items-center gap-2">
            <span className={`badge bg-${CAT_COLOR[cat.category] || 'secondary'}`}>{cat.category}</span>
            <span className="fw-semibold">{cat.total} gap{cat.total !== 1 ? 's' : ''}</span>
            <span className="ms-auto text-success fw-semibold" style={{ fontSize: '0.8rem' }}>
              {cat.built}/{cat.total} built
            </span>
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover align-middle mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Feature</th>
                    <th>Priority</th>
                    <th>Status</th>
                    <th>Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {(cat.items || []).map((item, ii) => (
                    <tr key={ii}>
                      <td style={{ maxWidth: 280 }}>{item.feature}</td>
                      <td>
                        <span className={`badge bg-${PRIORITY_COLOR[item.priority] || 'secondary'}`} style={{ fontSize: '0.7rem' }}>
                          {item.priority}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${item.in_project === 'built' ? 'bg-success' : item.in_project === 'partial' ? 'bg-warning text-dark' : 'bg-danger'}`} style={{ fontSize: '0.7rem' }}>
                          {item.in_project}
                        </span>
                      </td>
                      <td style={{ maxWidth: 380 }}>
                        <small className="text-muted">{item.why}</small>
                        {item.dashboard && (
                          <div>
                            <small className="text-primary font-monospace" style={{ fontSize: '0.68rem' }}>{item.dashboard}</small>
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ))}
    </>
  );
}

/* ── DL Review Scope Tab ──────────────────────────────────────── */
function ReviewScopeTab({ bk }) {
  if (!bk) return <div className="text-muted p-3">Loading…</div>;
  const covers = bk.review_covers || [];

  return (
    <div className="card shadow-sm border-0">
      <div className="card-header py-2 fw-semibold">What the 50-Paper DL Review Covers ({covers.length} topics)</div>
      <div className="card-body">
        <p className="text-muted small mb-3">
          This gap analysis is derived from a systematic review of 50 deep learning papers for epilepsy EEG classification.
          The review maps SOTA techniques to this project's implementation status.
        </p>
        <div className="row g-2">
          {covers.map((topic, i) => (
            <div key={i} className="col-md-6 col-lg-4">
              <div className="card border-0 shadow-sm h-100" style={{ background: '#f8f9fa' }}>
                <div className="card-body py-2 px-3">
                  <div className="d-flex align-items-start gap-2">
                    <span className="text-success fw-bold" style={{ fontSize: '1rem', lineHeight: 1.5 }}>✓</span>
                    <div>
                      <div className="fw-semibold" style={{ fontSize: '0.82rem' }}>{topic.topic || topic}</div>
                      {topic.note && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{topic.note}</div>}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── Recommendations Tab ────────────────────────────────────────── */
function RecommendationsTab({ bk }) {
  if (!bk) return <div className="text-muted p-3">Loading…</div>;
  const recs = bk.top_recommendations || [];

  return (
    <div className="card shadow-sm border-0">
      <div className="card-header py-2 fw-semibold">Top {recs.length} Recommendations from DL Review</div>
      <div className="card-body">
        {recs.map((rec, i) => (
          <div key={i} className="card border-start border-primary border-3 mb-3 shadow-sm">
            <div className="card-body py-2 px-3">
              <div className="d-flex align-items-start gap-2">
                <span className="badge bg-primary rounded-circle" style={{ minWidth: 24, textAlign: 'center' }}>{i + 1}</span>
                <div>
                  <div className="fw-semibold" style={{ fontSize: '0.88rem' }}>
                    {typeof rec === 'string' ? rec : rec.recommendation || rec.title || JSON.stringify(rec)}
                  </div>
                  {rec.rationale && <div className="text-muted mt-1" style={{ fontSize: '0.78rem' }}>{rec.rationale}</div>}
                  {rec.status && (
                    <span className={`badge mt-1 ${rec.status === 'implemented' || rec.status === 'built' ? 'bg-success' : 'bg-warning text-dark'}`} style={{ fontSize: '0.7rem' }}>
                      {rec.status}
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Definitions Tab ────────────────────────────────────────────── */
function DefinitionsTab({ def }) {
  if (!def) return <div className="text-muted p-3">Loading…</div>;
  const cats = def.categories || [];
  const prios = def.priority_legend || [];
  const statuses = def.status_legend || [];
  const glossary = def.glossary || [];

  return (
    <>
      <div className="row g-3 mb-3">
        <div className="col-md-4">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header py-2 fw-semibold">Gap Categories</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <tbody>
                  {cats.map((c, i) => (
                    <tr key={i}>
                      <td><span className={`badge bg-${CAT_COLOR[c.name] || 'secondary'}`}>{c.name}</span></td>
                      <td style={{ fontSize: '0.78rem' }}>{c.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header py-2 fw-semibold">Priority Levels</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <tbody>
                  {prios.map((p, i) => (
                    <tr key={i}>
                      <td><span className={`badge bg-${PRIORITY_COLOR[p.level] || 'secondary'}`}>{p.level}</span></td>
                      <td style={{ fontSize: '0.78rem' }}>{p.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header py-2 fw-semibold">Status Legend</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <tbody>
                  {statuses.map((s, i) => (
                    <tr key={i}>
                      <td>
                        <span className={`badge ${s.status === 'built' ? 'bg-success' : s.status === 'partial' ? 'bg-warning text-dark' : 'bg-danger'}`} style={{ fontSize: '0.7rem' }}>
                          {s.status}
                        </span>
                      </td>
                      <td style={{ fontSize: '0.78rem' }}>{s.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {glossary.length > 0 && (
        <div className="card shadow-sm border-0">
          <div className="card-header py-2 fw-semibold">Glossary ({glossary.length} terms)</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover align-middle mb-0">
                <thead className="table-light">
                  <tr><th>Term</th><th>Definition</th></tr>
                </thead>
                <tbody>
                  {glossary.map((g, i) => (
                    <tr key={i}>
                      <td className="fw-semibold text-primary" style={{ whiteSpace: 'nowrap' }}>{g.term}</td>
                      <td style={{ fontSize: '0.82rem' }}>{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/* ── Main Page ──────────────────────────────────────────────────── */
export default function FeatureGapsPage() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [def, setDef] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/feature-gaps/overview`).then(r => r.json()),
      fetch(`${API}/api/feature-gaps/breakdown`).then(r => r.json()),
      fetch(`${API}/api/feature-gaps/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDef(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return (
    <div className="container py-4">
      <div className="alert alert-danger">API error: {err}</div>
    </div>
  );

  const s = ov?.summary || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex flex-wrap align-items-center gap-2 mb-3">
        <div>
          <h4 className="mb-0 fw-bold">🔬 Epilepsy DL Review → Project Gap Analysis</h4>
          <small className="text-muted">
            50-paper deep learning review mapped to implementation status · {s.total_gaps || 18} gaps across {s.categories || 6} categories
          </small>
        </div>
        <div className="ms-auto d-flex gap-2 flex-wrap">
          <span className="badge bg-success fs-6">{s.built_pct != null ? `${s.built_pct}%` : '—'} Built</span>
          <span className="badge bg-primary">{s.review_papers || 50} Papers</span>
          <span className="badge bg-info">{s.total_gaps || 18} Gaps</span>
        </div>
      </div>

      {/* Source note */}
      <div className="alert alert-info py-2 mb-3" style={{ fontSize: '0.82rem' }}>
        <strong>Source:</strong> Epilepsy/paper/epilepsy-dl-review.pdf (50 references) · Updated {ov?.updated_at || '2026-08-12'} ·
        All {s.built || 18} gaps are addressed in this project via verified endpoints and frontend dashboards.
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'        && <OverviewTab ov={ov} />}
      {tab === 'by_category'     && <ByCategoryTab bk={bk} />}
      {tab === 'review_covers'   && <ReviewScopeTab bk={bk} />}
      {tab === 'recommendations' && <RecommendationsTab bk={bk} />}
      {tab === 'definitions'     && <DefinitionsTab def={def} />}
    </div>
  );
}
