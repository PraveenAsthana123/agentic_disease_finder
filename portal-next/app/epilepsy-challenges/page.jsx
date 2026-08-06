'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const LEVEL_COLORS = { basic: 'success', intermediate: 'warning', high: 'danger' };
const LEVEL_BADGES = { basic: '🟢', intermediate: '🟡', high: '🔴' };

export default function EpilepsyChallengesDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [levelFilter, setLevelFilter] = useState('all');
  const [expanded, setExpanded] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/epilepsy-challenges/overview`).then(r => r.json()),
      fetch(`${API}/api/epilepsy-challenges/breakdown`).then(r => r.json()),
      fetch(`${API}/api/epilepsy-challenges/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading Epilepsy Challenges data…</div>;

  const summary = ov.summary || {};
  const levelDist = ov.level_distribution || [];
  const allChallenges = bd?.all_challenges || [];
  const byLevel = bd?.by_level || {};
  const glossary = defs?.glossary || [];
  const levels = defs?.levels || [];
  const starMethod = defs?.star_method || {};
  const clinicalNotes = defs?.clinical_notes || [];
  const references = defs?.references || [];

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'all', label: '📋 All Challenges' },
    { id: 'star', label: '⭐ STAR Detail' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const filtered = allChallenges.filter(c => {
    const matchLevel = levelFilter === 'all' || c.level === levelFilter;
    const q = search.toLowerCase();
    const matchSearch = !q || (
      (c.challenge || '').toLowerCase().includes(q) ||
      (c.ai_help || '').toLowerCase().includes(q) ||
      (c.level || '').toLowerCase().includes(q)
    );
    return matchLevel && matchSearch;
  });

  const toggleExpand = n => setExpanded(prev => prev === n ? null : n);

  return (
    <div className="p-3">
      <h3>⚡ Epilepsy / EEG Challenges Dashboard</h3>
      <p className="text-muted">
        {summary.total_challenges} challenges · {summary.basic} basic · {summary.intermediate} intermediate · {summary.high} high ·
        {' '}{summary.ai_help_coverage_pct}% AI-help coverage · {summary.star_documented} STAR-documented
      </p>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div>
          {/* KPI cards */}
          <div className="row g-3 mb-4">
            <div className="col-sm-3">
              <div className="card text-center h-100">
                <div className="card-body">
                  <div className="fs-1 fw-bold text-primary">{summary.total_challenges}</div>
                  <div className="text-muted small">Total Challenges</div>
                </div>
              </div>
            </div>
            <div className="col-sm-3">
              <div className="card text-center h-100 border-success">
                <div className="card-body">
                  <div className="fs-1 fw-bold text-success">{summary.basic}</div>
                  <div className="text-muted small">🟢 Basic</div>
                </div>
              </div>
            </div>
            <div className="col-sm-3">
              <div className="card text-center h-100 border-warning">
                <div className="card-body">
                  <div className="fs-1 fw-bold text-warning">{summary.intermediate}</div>
                  <div className="text-muted small">🟡 Intermediate</div>
                </div>
              </div>
            </div>
            <div className="col-sm-3">
              <div className="card text-center h-100 border-danger">
                <div className="card-body">
                  <div className="fs-1 fw-bold text-danger">{summary.high}</div>
                  <div className="text-muted small">🔴 High</div>
                </div>
              </div>
            </div>
          </div>

          {/* Level summary cards */}
          {levelDist.map(ld => {
            const lvlChallenges = allChallenges.filter(c => c.level === ld.name);
            const color = LEVEL_COLORS[ld.name] || 'secondary';
            const badge = LEVEL_BADGES[ld.name] || '⚪';
            return (
              <div key={ld.name} className={`card mb-3 border-${color}`}>
                <div className="card-header d-flex align-items-center">
                  <span className="me-2 fs-5">{badge}</span>
                  <strong className={`text-${color} text-capitalize`}>{ld.name}</strong>
                  <span className="ms-2 text-muted small">({ld.value} challenges)</span>
                </div>
                <div className="card-body p-2">
                  <div className="row g-2">
                    {lvlChallenges.map(c => (
                      <div key={c.n} className="col-md-6">
                        <div className="border rounded p-2 h-100 bg-light">
                          <div className="small fw-semibold">#{c.n} {c.challenge}</div>
                          <div className="small text-muted mt-1">🤖 {c.ai_help}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── ALL CHALLENGES TAB ── */}
      {tab === 'all' && (
        <div>
          <div className="row g-2 mb-3">
            <div className="col-md-6">
              <input
                className="form-control"
                placeholder="Search challenges…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <div className="col-md-3">
              <select className="form-select" value={levelFilter} onChange={e => setLevelFilter(e.target.value)}>
                <option value="all">All Levels</option>
                <option value="basic">🟢 Basic</option>
                <option value="intermediate">🟡 Intermediate</option>
                <option value="high">🔴 High</option>
              </select>
            </div>
            <div className="col-md-3 d-flex align-items-center">
              <span className="text-muted small">{filtered.length} of {allChallenges.length} challenges</span>
            </div>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr>
                  <th>#</th>
                  <th>Level</th>
                  <th>Challenge</th>
                  <th>AI Help</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(c => (
                  <tr key={c.n}>
                    <td className="text-muted">{c.n}</td>
                    <td>
                      <span className={`badge bg-${LEVEL_COLORS[c.level] || 'secondary'}`}>
                        {LEVEL_BADGES[c.level] || ''} {c.level}
                      </span>
                    </td>
                    <td>{c.challenge}</td>
                    <td className="text-muted small">{c.ai_help}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── STAR DETAIL TAB ── */}
      {tab === 'star' && (
        <div>
          <p className="text-muted small mb-3">
            Click any challenge to expand the full STAR (Situation / Task / Action / Result) justification.
          </p>
          {['basic', 'intermediate', 'high'].map(lvl => {
            const lvlItems = (byLevel[lvl] || []);
            const color = LEVEL_COLORS[lvl] || 'secondary';
            const badge = LEVEL_BADGES[lvl] || '⚪';
            return (
              <div key={lvl} className="mb-4">
                <h5 className={`text-${color}`}>{badge} {lvl.charAt(0).toUpperCase() + lvl.slice(1)} Challenges</h5>
                {lvlItems.map(c => (
                  <div key={c.n} className="card mb-2">
                    <div
                      className="card-header d-flex justify-content-between align-items-center"
                      style={{ cursor: 'pointer' }}
                      onClick={() => toggleExpand(`${lvl}-${c.n}`)}
                    >
                      <span>
                        <strong>#{c.n}</strong> {c.challenge}
                      </span>
                      <span className="text-muted small">{expanded === `${lvl}-${c.n}` ? '▲' : '▼'}</span>
                    </div>
                    {expanded === `${lvl}-${c.n}` && (
                      <div className="card-body">
                        <div className="mb-2">
                          <span className="badge bg-info text-dark me-2">🤖 AI Help</span>
                          <span>{c.ai_help}</span>
                        </div>
                        <table className="table table-sm table-bordered mb-0">
                          <tbody>
                            <tr>
                              <th className="bg-light" style={{ width: '15%' }}>S — Situation</th>
                              <td>{c.situation || '—'}</td>
                            </tr>
                            <tr>
                              <th className="bg-light">T — Task</th>
                              <td>{c.task || '—'}</td>
                            </tr>
                            <tr>
                              <th className="bg-light">A — Action</th>
                              <td>{c.action || '—'}</td>
                            </tr>
                            <tr>
                              <th className="bg-light">R — Result</th>
                              <td>{c.result || '—'}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && (
        <div>
          {/* STAR method */}
          <div className="card mb-3">
            <div className="card-header"><strong>⭐ STAR Method</strong></div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  {Object.entries(starMethod).map(([k, v]) => (
                    <tr key={k}>
                      <th style={{ width: '5%' }}>{k.toUpperCase()}</th>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Difficulty levels */}
          <div className="card mb-3">
            <div className="card-header"><strong>📊 Difficulty Levels</strong></div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Level</th><th>Description</th></tr></thead>
                <tbody>
                  {levels.map(l => (
                    <tr key={l.name}>
                      <td>
                        <span className={`badge bg-${LEVEL_COLORS[l.name] || 'secondary'}`}>
                          {LEVEL_BADGES[l.name] || ''} {l.name}
                        </span>
                      </td>
                      <td className="small">{l.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Glossary */}
          <div className="card mb-3">
            <div className="card-header"><strong>📖 Glossary</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Term</th><th>Definition</th></tr></thead>
                <tbody>
                  {glossary.map(g => (
                    <tr key={g.term}>
                      <td><code>{g.term}</code></td>
                      <td className="small">{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Clinical notes */}
          {clinicalNotes.length > 0 && (
            <div className="card mb-3">
              <div className="card-header"><strong>🏥 Clinical Notes</strong></div>
              <div className="card-body">
                <ul className="mb-0">
                  {clinicalNotes.map((n, i) => <li key={i} className="small">{n}</li>)}
                </ul>
              </div>
            </div>
          )}

          {/* References */}
          {references.length > 0 && (
            <div className="card mb-3">
              <div className="card-header"><strong>📚 References</strong></div>
              <div className="card-body">
                <ul className="mb-0">
                  {references.map((r, i) => <li key={i} className="small">{r}</li>)}
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
