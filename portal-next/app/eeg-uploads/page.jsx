'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'breakdown',  label: 'All Uploads' },
  { id: 'definitions', label: 'Definitions' },
];

const FORMAT_COLORS = {
  '.edf': 'primary',
  '.bdf': 'success',
  '.fif': 'info',
  '.csv': 'warning',
  '.set': 'secondary',
};

const DEPT_COLORS = {
  neurology:    'primary',
  neurosurgery: 'danger',
  psychiatry:   'purple',
  sleep_lab:    'info',
  geriatrics:   'warning',
  unassigned:   'secondary',
};

const DISEASE_COLORS = {
  epilepsy:       'danger',
  sleep_disorder: 'info',
  depression:     'warning',
  parkinsons:     'secondary',
  alzheimers:     'dark',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-2 mb-3">
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

function BarChart({ items, keyField, valueField, colorFn, label }) {
  if (!items || !items.length) return <p className="text-muted small">No data.</p>;
  const max = Math.max(...items.map(i => i[valueField]));
  return (
    <div>
      {label && <div className="text-muted small mb-2">{label}</div>}
      <table className="table table-sm mb-0">
        <tbody>
          {items.map((item, i) => {
            const pct = max > 0 ? ((item[valueField] / max) * 100).toFixed(1) : 0;
            const color = colorFn ? colorFn(item[keyField]) : 'primary';
            return (
              <tr key={i}>
                <td className="text-nowrap small fw-semibold" style={{ width: '35%' }}>
                  {item[keyField]}
                </td>
                <td style={{ width: '50%' }}>
                  <div className="progress" style={{ height: 14 }}>
                    <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }}>
                      <span className="small">{item[valueField]}</span>
                    </div>
                  </div>
                </td>
                <td className="small text-end text-muted">{item[valueField]}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function MonthlyTrend({ trend }) {
  if (!trend || !trend.length) return <p className="text-muted small">No trend data.</p>;
  const max = Math.max(...trend.map(t => t.uploads));
  return (
    <div>
      <div className="d-flex align-items-end gap-1" style={{ height: 80 }}>
        {trend.map((t, i) => {
          const h = max > 0 ? Math.round((t.uploads / max) * 70) : 0;
          return (
            <div key={i} className="flex-grow-1 d-flex flex-column align-items-center">
              <div
                className="bg-primary rounded-top"
                style={{ height: h, minHeight: t.uploads > 0 ? 4 : 0, width: '100%' }}
                title={`${t.month}: ${t.uploads} uploads`}
              />
            </div>
          );
        })}
      </div>
      <div className="d-flex gap-1 mt-1">
        {trend.map((t, i) => (
          <div key={i} className="flex-grow-1 text-center text-muted" style={{ fontSize: '0.6rem' }}>
            {t.month.slice(5)}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function EEGUploadsPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [search, setSearch] = useState('');
  const [fmtFilter, setFmtFilter] = useState('all');
  const [deptFilter, setDeptFilter] = useState('all');

  useEffect(() => {
    fetch(`${API}/api/uploads/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/uploads/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/uploads/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return (
    <div className="p-4 text-center">
      <div className="spinner-border text-primary" />
      <div className="mt-2 text-muted">Loading Upload Analytics…</div>
    </div>
  );

  const uploads = bd?.all_uploads || [];
  const formats  = [...new Set(uploads.map(u => u.format))].sort();
  const depts    = [...new Set(uploads.map(u => u.department))].sort();

  const filtered = uploads.filter(u => {
    const q = search.toLowerCase();
    const matchQ = !q || u.patient_id?.toLowerCase().includes(q) || u.file_name?.toLowerCase().includes(q) || u.disease?.toLowerCase().includes(q);
    const matchFmt  = fmtFilter  === 'all' || u.format      === fmtFilter;
    const matchDept = deptFilter === 'all' || u.department  === deptFilter;
    return matchQ && matchFmt && matchDept;
  });

  return (
    <div className="container-fluid py-3">
      <div className="mb-3 d-flex align-items-center gap-2">
        <span style={{ fontSize: '1.4rem' }}>📤</span>
        <div>
          <h4 className="mb-0 fw-bold">EEG Upload Analytics</h4>
          <div className="text-muted small">
            {ov.total_uploads} uploads · {ov.total_patients} patients · {ov.total_diseases} diseases · {ov.total_formats} formats
          </div>
        </div>
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

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          {/* KPIs */}
          <div className="row g-2 mb-4">
            <KPI label="Total Uploads"    value={ov.total_uploads}    color="primary" />
            <KPI label="Patients Covered" value={ov.total_patients}   color="success" />
            <KPI label="Diseases"         value={ov.total_diseases}   color="info" />
            <KPI label="Departments"      value={ov.total_departments} color="warning" />
            <KPI label="File Formats"     value={ov.total_formats}    color="secondary" />
            <KPI
              label="Epilepsy Share"
              value={ov.disease_distribution?.find(d => d.disease === 'epilepsy')
                ? `${Math.round((ov.disease_distribution.find(d => d.disease === 'epilepsy').count / ov.total_uploads) * 100)}%`
                : '—'}
              color="danger"
            />
          </div>

          <div className="row g-3 mb-4">
            {/* Monthly Trend */}
            <div className="col-md-8">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold py-2">📈 Monthly Upload Volume</div>
                <div className="card-body">
                  <MonthlyTrend trend={ov.monthly_trend} />
                  <div className="d-flex flex-wrap gap-2 mt-2">
                    {(ov.monthly_trend || []).map(t => (
                      <span key={t.month} className="badge bg-primary bg-opacity-10 text-primary">
                        {t.month}: {t.uploads}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Top Uploaders */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold py-2">🏆 Top Uploaders</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Patient ID</th><th className="text-end">Uploads</th></tr>
                    </thead>
                    <tbody>
                      {(ov.top_uploaders || []).map((u, i) => (
                        <tr key={i}>
                          <td className="small">{u.patient_id}</td>
                          <td className="text-end small fw-bold">{u.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3 mb-4">
            {/* Disease Distribution */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold py-2">🧠 By Disease</div>
                <div className="card-body">
                  <BarChart
                    items={ov.disease_distribution}
                    keyField="disease"
                    valueField="count"
                    colorFn={k => DISEASE_COLORS[k] || 'secondary'}
                  />
                </div>
              </div>
            </div>

            {/* Department Distribution */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold py-2">🏥 By Department</div>
                <div className="card-body">
                  <BarChart
                    items={ov.department_distribution}
                    keyField="department"
                    valueField="count"
                    colorFn={k => DEPT_COLORS[k] || 'secondary'}
                  />
                </div>
              </div>
            </div>

            {/* Format Distribution */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold py-2">📁 By File Format</div>
                <div className="card-body">
                  <BarChart
                    items={ov.format_distribution}
                    keyField="format"
                    valueField="count"
                    colorFn={k => FORMAT_COLORS[k] || 'secondary'}
                  />
                  <div className="d-flex flex-wrap gap-1 mt-2">
                    {(ov.format_distribution || []).map(f => (
                      <span
                        key={f.format}
                        className={`badge bg-${FORMAT_COLORS[f.format] || 'secondary'}`}
                      >
                        {f.format} {f.count}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── BREAKDOWN ── */}
      {tab === 'breakdown' && (
        <>
          {/* Filters */}
          <div className="row g-2 mb-3">
            <div className="col-md-5">
              <input
                type="text"
                className="form-control form-control-sm"
                placeholder="Search patient ID, filename, disease…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <div className="col-md-3">
              <select
                className="form-select form-select-sm"
                value={fmtFilter}
                onChange={e => setFmtFilter(e.target.value)}
              >
                <option value="all">All formats</option>
                {formats.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
            </div>
            <div className="col-md-3">
              <select
                className="form-select form-select-sm"
                value={deptFilter}
                onChange={e => setDeptFilter(e.target.value)}
              >
                <option value="all">All departments</option>
                {depts.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
            <div className="col-md-1 d-flex align-items-center">
              <span className="text-muted small">{filtered.length} rows</span>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-body p-0">
              <div className="table-responsive" style={{ maxHeight: 600 }}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light sticky-top">
                    <tr>
                      <th>#</th>
                      <th>Patient ID</th>
                      <th>File Name</th>
                      <th>Disease</th>
                      <th>Department</th>
                      <th>Format</th>
                      <th>Uploaded</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((u, i) => (
                      <tr key={u.id}>
                        <td className="text-muted small">{u.id}</td>
                        <td className="small fw-semibold">{u.patient_id}</td>
                        <td className="small font-monospace">{u.file_name}</td>
                        <td>
                          <span className={`badge bg-${DISEASE_COLORS[u.disease] || 'secondary'}`}>
                            {u.disease}
                          </span>
                        </td>
                        <td>
                          <span className={`badge bg-${DEPT_COLORS[u.department] || 'secondary'} bg-opacity-75`}>
                            {u.department}
                          </span>
                        </td>
                        <td>
                          <span className={`badge bg-${FORMAT_COLORS[u.format] || 'secondary'}`}>
                            {u.format}
                          </span>
                        </td>
                        <td className="small text-muted">{u.created_at}</td>
                      </tr>
                    ))}
                    {filtered.length === 0 && (
                      <tr>
                        <td colSpan={7} className="text-center text-muted py-4">No uploads match the filter.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold py-2">📋 Field Definitions</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Field</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(defs.fields || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td className="small font-monospace fw-semibold">{k}</td>
                        <td className="small text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold py-2">📁 File Formats</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Format</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(defs.file_formats || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td>
                          <span className={`badge bg-${FORMAT_COLORS[k] || 'secondary'}`}>{k}</span>
                        </td>
                        <td className="small text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold py-2">🏥 Departments</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(defs.departments || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td className="small fw-semibold">{k}</td>
                        <td className="small text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {defs.clinical_notes?.length > 0 && (
              <div className="card shadow-sm">
                <div className="card-header fw-semibold py-2">📝 Clinical Notes</div>
                <div className="card-body">
                  <ul className="mb-0 ps-3">
                    {defs.clinical_notes.map((n, i) => (
                      <li key={i} className="small text-muted mb-1">{n}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
