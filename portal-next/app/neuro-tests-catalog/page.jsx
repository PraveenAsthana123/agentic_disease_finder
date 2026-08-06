'use client';
import { useEffect, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const EEG_LINK_COLORS = {
  core: '#6366f1',
  'evoked-potential': '#8b5cf6',
  autonomic: '#ef4444',
  motor: '#f59e0b',
  'peripheral/independent': '#10b981',
  other: '#06b6d4',
};

const eegLinkBadge = (link = '') => {
  const key = Object.keys(EEG_LINK_COLORS).find(k => link.toLowerCase().startsWith(k)) || 'other';
  const color = EEG_LINK_COLORS[key] || '#9ca3af';
  return (
    <span className="badge" style={{ background: color, fontSize: 10, whiteSpace: 'normal', textAlign: 'left' }}>
      {link}
    </span>
  );
};

function KpiCard({ label, value, color, sub }) {
  return (
    <div className="card text-center shadow-sm h-100">
      <div className="card-body py-3">
        <div className="fs-3 fw-bold" style={{ color: color || '#6366f1' }}>{value ?? '—'}</div>
        <div className="small fw-semibold text-muted">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
      </div>
    </div>
  );
}

function BarRow({ label, value, max, color }) {
  const pct = max ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small fw-semibold">{label}</span>
        <span className="small text-muted">{value} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color || '#6366f1' }} />
      </div>
    </div>
  );
}

function TestCard({ test }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="card shadow-sm mb-2" style={{ cursor: 'pointer' }} onClick={() => setOpen(o => !o)}>
      <div className="card-body py-2 px-3">
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span className="badge bg-primary" style={{ fontSize: 12 }}>{test.id}</span>
          <span className="fw-semibold small">{test.name}</span>
          {test.has_case_data && <span className="badge bg-success" style={{ fontSize: 10 }}>📁 case data</span>}
          <span className="ms-auto">{eegLinkBadge(test.eeg_link)}</span>
        </div>
        {open && (
          <div className="mt-2 ps-1" style={{ fontSize: 12 }}>
            <div><span className="text-muted">Purpose:</span> {test.purpose}</div>
            <div><span className="text-muted">Role:</span> {test.role}</div>
            <div><span className="text-muted">Output:</span> {test.output}</div>
            {test.note && <div className="text-muted mt-1" style={{ fontSize: 11 }}>{test.note}</div>}
          </div>
        )}
      </div>
    </div>
  );
}

export default function NeuroTestsCatalogPage() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/neuro-tests-catalog/overview`).then(r => r.json()),
      fetch(`${API}/api/neuro-tests-catalog/breakdown`).then(r => r.json()),
      fetch(`${API}/api/neuro-tests-catalog/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov) return (
    <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}>
      <div className="spinner-border text-primary" />
    </div>
  );

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'by-eeg-link', label: '🔗 By EEG Linkage' },
    { id: 'by-role', label: '👤 By Role' },
    { id: 'all-tests', label: '🧪 All Tests' },
    { id: 'definitions', label: '📚 Definitions' },
  ];

  const sm = ov.summary || {};
  const eegLinkDist = ov.eeg_link_distribution || [];
  const roleDist = ov.role_distribution || [];
  const testsTable = ov.tests_table || [];
  const maxEegLink = Math.max(...eegLinkDist.map(e => e.value), 1);
  const maxRole = Math.max(...roleDist.map(r => r.value), 1);

  const byEegLink = bd?.by_eeg_link || {};
  const byRole = bd?.by_role || {};

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1">🧠 Neurophysiology / Electrodiagnostic Test Catalog</h2>
      <p className="text-muted mb-3">
        {sm.total_tests} tests cataloged · {sm.built} built · {sm.eeg_linkage_categories} EEG linkage categories · {sm.unique_roles} roles
      </p>

      {/* KPI row */}
      <div className="row g-2 mb-4">
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Total Tests" value={sm.total_tests} color="#6366f1" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Built" value={sm.built} color="#10b981" sub="all verified" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="EEG Link Categories" value={sm.eeg_linkage_categories} color="#8b5cf6" sub="core/evoked/autonomic/motor" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Unique Roles" value={sm.unique_roles} color="#f59e0b" sub="clinical staff" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Output Types" value={sm.unique_output_types} color="#06b6d4" sub="distinct measurements" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Partial" value={sm.partial ?? 0} color="#9ca3af" sub="none pending" />
        </div>
      </div>

      {/* Tabs */}
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

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div className="row g-4">
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">🔗 EEG Linkage Distribution</div>
              <div className="card-body">
                {eegLinkDist.map(e => (
                  <BarRow key={e.name} label={e.name} value={e.value}
                    max={maxEegLink} color={EEG_LINK_COLORS[e.name] || '#9ca3af'} />
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">👤 Role Distribution</div>
              <div className="card-body">
                {roleDist.map((r, i) => (
                  <BarRow key={i} label={r.name} value={r.value}
                    max={maxRole} color="#6366f1" />
                ))}
              </div>
            </div>
          </div>
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">🧪 Quick Test Index</div>
              <div className="card-body p-0">
                <div style={{ overflowX: 'auto' }}>
                  <table className="table table-sm table-hover table-striped mb-0" style={{ fontSize: 12 }}>
                    <thead className="table-dark">
                      <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Purpose</th>
                        <th>Role</th>
                        <th>Output</th>
                        <th>EEG Linkage</th>
                        <th>Case Data</th>
                      </tr>
                    </thead>
                    <tbody>
                      {testsTable.map((t, i) => (
                        <tr key={i}>
                          <td><span className="badge bg-primary">{t.id}</span></td>
                          <td className="fw-semibold">{t.name}</td>
                          <td className="text-muted">{t.purpose}</td>
                          <td>{t.role}</td>
                          <td>{t.output}</td>
                          <td>{eegLinkBadge(t.eeg_link)}</td>
                          <td>{t.has_case_data ? <span className="badge bg-success">📁 yes</span> : <span className="badge bg-secondary">—</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* BY EEG LINKAGE TAB */}
      {tab === 'by-eeg-link' && (
        <div className="row g-4">
          {Object.entries(byEegLink).map(([cat, tests]) => (
            <div key={cat} className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold d-flex align-items-center gap-2">
                  {eegLinkBadge(cat)}
                  <span>{cat}</span>
                  <span className="badge bg-secondary ms-auto">{tests.length}</span>
                </div>
                <div className="card-body">
                  {tests.map((t, i) => <TestCard key={i} test={t} />)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* BY ROLE TAB */}
      {tab === 'by-role' && (
        <div className="row g-4">
          {Object.entries(byRole).map(([role, tests]) => (
            <div key={role} className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">
                  👤 {role}
                  <span className="badge bg-secondary ms-2">{tests.length}</span>
                </div>
                <div className="card-body">
                  {tests.map((t, i) => <TestCard key={i} test={t} />)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ALL TESTS TAB */}
      {tab === 'all-tests' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">🧪 All {testsTable.length} Tests (click row to expand)</div>
          <div className="card-body">
            {testsTable.map((t, i) => <TestCard key={i} test={{ ...t, note: t.note || '' }} />)}
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div className="row g-4">
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">📌 Status Legend</div>
              <div className="card-body">
                {(defs.status_legend || []).map((s, i) => (
                  <div key={i} className="mb-2">
                    <span className="badge bg-primary me-2">{s.status}</span>
                    <span className="small text-muted">{s.description}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">🔗 EEG Link Types</div>
              <div className="card-body">
                {(defs.eeg_link_types || []).map((e, i) => (
                  <div key={i} className="mb-3">
                    {eegLinkBadge(e.type)}
                    <p className="small text-muted mb-0 mt-1">{e.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">📖 Glossary</div>
              <div className="card-body">
                <div className="row">
                  {(defs.glossary || []).map((g, i) => (
                    <div key={i} className="col-md-6 mb-2">
                      <span className="fw-semibold small text-primary">{g.term}</span>
                      <p className="text-muted small mb-0">{g.definition}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          {defs.clinical_notes && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">🏥 Clinical Notes</div>
                <div className="card-body">
                  <div className="row">
                    {Object.entries(defs.clinical_notes).map(([k, v]) => (
                      <div key={k} className="col-md-6 mb-2">
                        <span className="fw-semibold small">{k}</span>
                        <p className="text-muted small mb-0">{v}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
          {defs.references && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">📚 References</div>
                <div className="card-body">
                  <ul className="mb-0">
                    {defs.references.map((r, i) => (
                      <li key={i} className="small text-muted">{r}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
