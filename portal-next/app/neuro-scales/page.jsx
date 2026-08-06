'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPICard({ label, value, color = 'primary', sub }) {
  return (
    <div className="col-6 col-md-2 mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className={`h5 mb-0 text-${color}`}>{value ?? '—'}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function SeverityBadge({ severity }) {
  const s = (severity || '').toLowerCase();
  let color = 'secondary';
  if (s.includes('severe') || s.includes('high risk') || s.includes('non-adherent')) color = 'danger';
  else if (s.includes('moderate') || s.includes('medium') || s.includes('mild')) color = 'warning';
  else if (s.includes('minimal') || s.includes('low') || s.includes('normal') || s.includes('remission') || s.includes('favorable') || s.includes('high adher')) color = 'success';
  else if (s.includes('free') || s.includes('class 1') || s.includes('class i')) color = 'success';
  return <span className={`badge bg-${color}`} style={{ fontSize: '0.72rem' }}>{severity || '—'}</span>;
}

function OutcomeBadge({ favorable }) {
  return favorable
    ? <span className="badge bg-success">Favorable</span>
    : <span className="badge bg-danger">Unfavorable</span>;
}

function ScaleTable({ patients, cols }) {
  const [filter, setFilter] = useState('');
  const rows = (patients || []).filter(p =>
    !filter || (p.name || '').toLowerCase().includes(filter.toLowerCase()) || (p.patient_id || '').toLowerCase().includes(filter.toLowerCase())
  );
  return (
    <div>
      <input className="form-control form-control-sm mb-2" style={{ maxWidth: 260 }}
        placeholder="Filter by name / ID…" value={filter} onChange={e => setFilter(e.target.value)} />
      <div style={{ overflowX: 'auto' }}>
        <table className="table table-sm table-hover table-striped" style={{ fontSize: '0.82rem' }}>
          <thead className="table-dark">
            <tr>
              <th>Patient</th>
              <th>Age</th>
              {cols.map((c, i) => <th key={i}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && <tr><td colSpan={2 + cols.length} className="text-center text-muted">No results</td></tr>}
            {rows.map((p, i) => (
              <tr key={i}>
                <td><span className="fw-semibold">{p.name || p.patient_id}</span><br /><span className="text-muted" style={{ fontSize: '0.7rem' }}>{p.patient_id}</span></td>
                <td>{p.age ?? '—'}</td>
                {cols.map((c, j) => <td key={j}>{c.render(p)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function NeuroScalesDashboard() {
  const [catalog, setCatalog] = useState(null);
  const [engel, setEngel] = useState(null);
  const [ilae, setIlae] = useState(null);
  const [sudep, setSudep] = useState(null);
  const [laep, setLaep] = useState(null);
  const [mmas, setMmas] = useState(null);
  const [hamd, setHamd] = useState(null);
  const [bdi, setBdi] = useState(null);
  const [panss, setPanss] = useState(null);
  const [gcs, setGcs] = useState(null);
  const [rankin, setRankin] = useState(null);
  const [stroop, setStroop] = useState(null);
  const [tmt, setTmt] = useState(null);
  const [tab, setTab] = useState('catalog');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/neuro-scales/catalog`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/engel`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/ilae`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/sudep`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/laep`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/mmas`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/hamd`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/bdi`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/panss`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/gcs`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/rankin`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/stroop`).then(r => r.json()).catch(() => null),
      fetch(`${API}/api/neuro-scales/tmt`).then(r => r.json()).catch(() => null),
    ]).then(([ca, en, il, su, la, mm, ha, bd, pa, gc, ra, st, tm]) => {
      setCatalog(ca); setEngel(en); setIlae(il); setSudep(su); setLaep(la);
      setMmas(mm); setHamd(ha); setBdi(bd); setPanss(pa); setGcs(gc);
      setRankin(ra); setStroop(st); setTmt(tm);
    }).catch(e => setErr(e.message));
  }, []);

  const tabs = [
    { id: 'catalog', label: '📋 Scale Catalog' },
    { id: 'epilepsy', label: '⚡ Epilepsy Outcomes' },
    { id: 'depression', label: '🧠 Depression & Psych' },
    { id: 'neurostatus', label: '🏥 Neuro Status' },
    { id: 'risk', label: '⚠️ Risk & Adherence' },
    { id: 'cognitive', label: '🔬 Cognitive Tests' },
  ];

  const engelPatients = engel?.patients || [];
  const ilaePatients = ilae?.patients || [];
  const sudepPatients = sudep?.patients || [];
  const laepPatients = laep?.patients || [];
  const mmasPatients = mmas?.patients || [];
  const hamdPatients = hamd?.patients || [];
  const bdiPatients = bdi?.patients || [];
  const panssPatients = panss?.patients || [];
  const gcsPatients = gcs?.patients || [];
  const rankinPatients = rankin?.patients || [];
  const stroopPatients = stroop?.patients || [];
  const tmtPatients = tmt?.patients || [];

  // Catalog category distribution
  const catDist = {};
  (catalog?.scales || []).forEach(s => { catDist[s.category] = (catDist[s.category] || 0) + 1; });
  const catEntries = Object.entries(catDist).sort((a, b) => b[1] - a[1]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-3 mb-3">
        <h4 className="mb-0">🩺 Neuro-Clinical Scales Dashboard</h4>
        <span className="badge bg-primary">{catalog?.total_scales || 23} Scales</span>
        <span className="badge bg-secondary">{catalog?.total_patients || 41} Patients</span>
        <span className="badge bg-info text-dark">23 Domains</span>
      </div>

      {err && <div className="alert alert-danger">Error: {err}</div>}

      {/* KPI row */}
      <div className="row mb-3">
        <KPICard label="Total Scales" value={catalog?.total_scales ?? 23} color="primary" />
        <KPICard label="Scale Categories" value={catalog?.categories?.length ?? 20} color="info" />
        <KPICard label="Patients Scored" value={catalog?.total_patients ?? 41} color="success" />
        <KPICard label="Engel I–II (Favorable)" value={engelPatients.filter(p => p.is_favorable).length} color="success" sub="seizure-free/rare" />
        <KPICard label="SUDEP High Risk" value={sudepPatients.filter(p => (p.severity || '').toLowerCase().includes('high')).length} color="danger" sub="SUDEP-7" />
        <KPICard label="MMAS Low Adherence" value={mmasPatients.filter(p => (p.adherence_level || '').toLowerCase().includes('low')).length} color="warning" sub="AED adherence" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* CATALOG TAB */}
      {tab === 'catalog' && (
        <div className="row">
          <div className="col-md-5 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold bg-primary text-white">Scale Catalog — {catalog?.total_scales ?? 23} Validated Scales</div>
              <div className="card-body p-0" style={{ maxHeight: 520, overflowY: 'auto' }}>
                <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.82rem' }}>
                  <thead className="table-light sticky-top">
                    <tr><th>Scale</th><th>Category</th><th>Range</th><th>Higher=</th></tr>
                  </thead>
                  <tbody>
                    {(catalog?.scales || []).map((s, i) => (
                      <tr key={i}>
                        <td><span className="fw-semibold">{s.name}</span></td>
                        <td><span className="badge bg-light text-dark border">{s.category}</span></td>
                        <td className="text-monospace">{s.range}</td>
                        <td>{s.higher === 'better' ? <span className="text-success">Better</span> : <span className="text-danger">Worse</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-7 mb-3">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">Scales by Category</div>
              <div className="card-body">
                {catEntries.map(([cat, cnt], i) => (
                  <div key={i} className="d-flex align-items-center mb-2">
                    <span className="me-2 small" style={{ minWidth: 200 }}>{cat}</span>
                    <div className="flex-grow-1 me-2">
                      <div className="progress" style={{ height: 18 }}>
                        <div className="progress-bar bg-primary" style={{ width: `${(cnt / Math.max(...catEntries.map(e => e[1]))) * 100}%` }}>
                          {cnt}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Scale Descriptions</div>
              <div className="card-body p-0" style={{ maxHeight: 260, overflowY: 'auto' }}>
                <table className="table table-sm mb-0" style={{ fontSize: '0.78rem' }}>
                  <tbody>
                    {(catalog?.scales || []).map((s, i) => (
                      <tr key={i}><td className="fw-semibold" style={{ minWidth: 120 }}>{s.name}</td><td className="text-muted">{s.description}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* EPILEPSY OUTCOMES TAB */}
      {tab === 'epilepsy' && (
        <div>
          <div className="row mb-3">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold bg-warning text-dark">
                  Engel Classification — {engelPatients.length} patients
                  <span className="ms-2 badge bg-success">{engelPatients.filter(p => p.is_favorable).length} Favorable</span>
                  <span className="ms-1 badge bg-danger">{engelPatients.filter(p => !p.is_favorable).length} Unfavorable</span>
                </div>
                <div className="card-body p-0" style={{ maxHeight: 380, overflowY: 'auto' }}>
                  <ScaleTable
                    patients={engelPatients}
                    cols={[
                      { label: 'Class', render: p => <span className="fw-bold">{p.engel_class}{p.engel_sub ? p.engel_sub.slice(1) : ''}</span> },
                      { label: 'Label', render: p => p.class_label },
                      { label: 'Outcome', render: p => <OutcomeBadge favorable={p.is_favorable} /> },
                      { label: 'Sz/30d', render: p => p.seizure_count_30d },
                    ]}
                  />
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold bg-info text-dark">
                  ILAE Outcome — {ilaePatients.length} patients
                  <span className="ms-2 badge bg-success">{ilaePatients.filter(p => p.is_favorable).length} Favorable</span>
                </div>
                <div className="card-body p-0" style={{ maxHeight: 380, overflowY: 'auto' }}>
                  <ScaleTable
                    patients={ilaePatients}
                    cols={[
                      { label: 'Class', render: p => <span className="fw-bold">{p.ilae_class}</span> },
                      { label: 'Label', render: p => <span style={{ fontSize: '0.73rem' }}>{p.class_label}</span> },
                      { label: 'Outcome', render: p => <OutcomeBadge favorable={p.is_favorable} /> },
                      { label: 'Ann. Sz Days', render: p => p.annualized_seizure_days ?? 0 },
                    ]}
                  />
                </div>
              </div>
            </div>
          </div>
          {/* Engel class distribution */}
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Engel Class Distribution</div>
            <div className="card-body">
              {['I', 'II', 'III', 'IV'].map(cls => {
                const cnt = engelPatients.filter(p => p.engel_class === cls).length;
                const pct = engelPatients.length ? ((cnt / engelPatients.length) * 100).toFixed(0) : 0;
                const colors = { I: 'success', II: 'info', III: 'warning', IV: 'danger' };
                return (
                  <div key={cls} className="d-flex align-items-center mb-2">
                    <span className="me-2 fw-bold" style={{ minWidth: 80 }}>Class {cls}</span>
                    <div className="flex-grow-1 me-2">
                      <div className="progress" style={{ height: 22 }}>
                        <div className={`progress-bar bg-${colors[cls]}`} style={{ width: `${pct}%` }}>{cnt} ({pct}%)</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* DEPRESSION & PSYCH TAB */}
      {tab === 'depression' && (
        <div className="row">
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold bg-info text-dark">HAM-D-17 — {hamdPatients.length} patients</div>
              <div className="card-body p-0" style={{ maxHeight: 400, overflowY: 'auto' }}>
                <ScaleTable
                  patients={hamdPatients}
                  cols={[
                    { label: 'Score', render: p => <span className="fw-bold">{p.total_score}</span> },
                    { label: 'Severity', render: p => <SeverityBadge severity={p.severity} /> },
                    { label: 'Remission', render: p => p.meets_remission ? <span className="badge bg-success">Yes</span> : <span className="badge bg-danger">No</span> },
                  ]}
                />
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold bg-warning text-dark">BDI-II — {bdiPatients.length} patients</div>
              <div className="card-body p-0" style={{ maxHeight: 400, overflowY: 'auto' }}>
                <ScaleTable
                  patients={bdiPatients}
                  cols={[
                    { label: 'Score', render: p => <span className="fw-bold">{p.total_score}</span> },
                    { label: 'Severity', render: p => <SeverityBadge severity={p.severity} /> },
                    { label: 'Remission', render: p => p.meets_remission ? <span className="badge bg-success">Yes</span> : <span className="badge bg-danger">No</span> },
                  ]}
                />
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold bg-secondary text-white">PANSS — {panssPatients.length} patients</div>
              <div className="card-body p-0" style={{ maxHeight: 400, overflowY: 'auto' }}>
                <ScaleTable
                  patients={panssPatients}
                  cols={[
                    { label: 'Score', render: p => <span className="fw-bold">{p.total_score}</span> },
                    { label: 'Severity', render: p => <SeverityBadge severity={p.severity} /> },
                    { label: 'Positive', render: p => p.positive_subscale ?? '—' },
                    { label: 'Negative', render: p => p.negative_subscale ?? '—' },
                  ]}
                />
              </div>
            </div>
          </div>
          {/* Severity distributions */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Depression Severity Breakdown (HAM-D vs BDI-II)</div>
              <div className="card-body">
                <div className="row">
                  {['HAM-D', 'BDI-II'].map((scaleName, si) => {
                    const pts = si === 0 ? hamdPatients : bdiPatients;
                    const sevDist = {};
                    pts.forEach(p => { sevDist[p.severity] = (sevDist[p.severity] || 0) + 1; });
                    const entries = Object.entries(sevDist).sort((a, b) => b[1] - a[1]);
                    return (
                      <div key={si} className="col-md-6">
                        <h6 className="fw-bold">{scaleName}</h6>
                        {entries.map(([sev, cnt], i) => {
                          const pct = pts.length ? ((cnt / pts.length) * 100).toFixed(0) : 0;
                          return (
                            <div key={i} className="d-flex align-items-center mb-2">
                              <span className="me-2 small" style={{ minWidth: 180 }}>{sev}</span>
                              <div className="flex-grow-1 me-1">
                                <div className="progress" style={{ height: 18 }}>
                                  <div className="progress-bar bg-info" style={{ width: `${pct}%` }}>{cnt} ({pct}%)</div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* NEURO STATUS TAB */}
      {tab === 'neurostatus' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold bg-dark text-white">Glasgow Coma Scale — {gcsPatients.length} patients</div>
              <div className="card-body p-0" style={{ maxHeight: 420, overflowY: 'auto' }}>
                <ScaleTable
                  patients={gcsPatients}
                  cols={[
                    { label: 'GCS', render: p => <span className="fw-bold">{p.total_score}/15</span> },
                    { label: 'Severity', render: p => <SeverityBadge severity={p.severity} /> },
                    { label: 'Eye', render: p => p.eye_score ?? '—' },
                    { label: 'Verbal', render: p => p.verbal_score ?? '—' },
                    { label: 'Motor', render: p => p.motor_score ?? '—' },
                  ]}
                />
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold bg-primary text-white">Modified Rankin Scale — {rankinPatients.length} patients</div>
              <div className="card-body p-0" style={{ maxHeight: 420, overflowY: 'auto' }}>
                <ScaleTable
                  patients={rankinPatients}
                  cols={[
                    { label: 'Score', render: p => <span className="fw-bold">{p.rankin_score}</span> },
                    { label: 'Label', render: p => <span style={{ fontSize: '0.73rem' }}>{p.class_label}</span> },
                    { label: 'Favorable', render: p => p.is_favorable ? <span className="badge bg-success">Yes</span> : <span className="badge bg-danger">No</span> },
                  ]}
                />
              </div>
            </div>
          </div>
          {/* GCS distribution */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">GCS Severity Distribution</div>
              <div className="card-body">
                {(() => {
                  const dist = {};
                  gcsPatients.forEach(p => { dist[p.severity] = (dist[p.severity] || 0) + 1; });
                  const colorMap = { 'Mild': 'warning', 'Moderate': 'danger', 'Severe': 'dark', 'Normal': 'success', 'No impairment': 'success' };
                  return Object.entries(dist).sort((a, b) => b[1] - a[1]).map(([sev, cnt], i) => {
                    const pct = gcsPatients.length ? ((cnt / gcsPatients.length) * 100).toFixed(0) : 0;
                    return (
                      <div key={i} className="d-flex align-items-center mb-2">
                        <span className="me-2 small" style={{ minWidth: 180 }}>{sev}</span>
                        <div className="flex-grow-1 me-1">
                          <div className="progress" style={{ height: 22 }}>
                            <div className={`progress-bar bg-${colorMap[sev] || 'secondary'}`} style={{ width: `${pct}%` }}>{cnt} ({pct}%)</div>
                          </div>
                        </div>
                      </div>
                    );
                  });
                })()}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* RISK & ADHERENCE TAB */}
      {tab === 'risk' && (
        <div className="row">
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold bg-danger text-white">SUDEP-7 Risk — {sudepPatients.length} patients</div>
              <div className="card-body p-0" style={{ maxHeight: 420, overflowY: 'auto' }}>
                <ScaleTable
                  patients={sudepPatients}
                  cols={[
                    { label: 'Score', render: p => <span className="fw-bold">{p.total_score}</span> },
                    { label: 'Risk', render: p => <SeverityBadge severity={p.severity} /> },
                    { label: 'GTC #', render: p => p.gtc_count ?? 0 },
                    { label: 'AEDs', render: p => p.aed_count ?? 0 },
                  ]}
                />
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold bg-warning text-dark">LAEP (AED Side Effects) — {laepPatients.length} patients</div>
              <div className="card-body p-0" style={{ maxHeight: 420, overflowY: 'auto' }}>
                <ScaleTable
                  patients={laepPatients}
                  cols={[
                    { label: 'Score', render: p => <span className="fw-bold">{p.total_score}</span> },
                    { label: 'Burden', render: p => <SeverityBadge severity={p.burden_level} /> },
                    { label: 'CNS', render: p => p.cns_score ?? '—' },
                    { label: 'Systemic', render: p => p.systemic_score ?? '—' },
                  ]}
                />
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold bg-success text-white">MMAS-4 Adherence — {mmasPatients.length} patients</div>
              <div className="card-body p-0" style={{ maxHeight: 420, overflowY: 'auto' }}>
                <ScaleTable
                  patients={mmasPatients}
                  cols={[
                    { label: 'Score', render: p => <span className="fw-bold">{p.total_score}/4</span> },
                    { label: 'Level', render: p => <SeverityBadge severity={p.adherence_level} /> },
                    { label: 'Uninten.', render: p => p.domain_scores?.unintentional ?? '—' },
                    { label: 'Inten.', render: p => p.domain_scores?.intentional ?? '—' },
                  ]}
                />
              </div>
            </div>
          </div>
          {/* SUDEP risk distribution */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">SUDEP Risk & AED Adherence Distribution</div>
              <div className="card-body">
                <div className="row">
                  <div className="col-md-6">
                    <h6 className="fw-bold">SUDEP-7 Risk Level</h6>
                    {(() => {
                      const dist = {};
                      sudepPatients.forEach(p => { dist[p.severity] = (dist[p.severity] || 0) + 1; });
                      const cm = { 'Low risk': 'success', 'Moderate risk': 'warning', 'High risk': 'danger' };
                      return Object.entries(dist).sort((a, b) => b[1] - a[1]).map(([sev, cnt], i) => {
                        const pct = sudepPatients.length ? ((cnt / sudepPatients.length) * 100).toFixed(0) : 0;
                        return (
                          <div key={i} className="d-flex align-items-center mb-2">
                            <span className="me-2 small" style={{ minWidth: 130 }}>{sev}</span>
                            <div className="flex-grow-1">
                              <div className="progress" style={{ height: 20 }}>
                                <div className={`progress-bar bg-${cm[sev] || 'secondary'}`} style={{ width: `${pct}%` }}>{cnt} ({pct}%)</div>
                              </div>
                            </div>
                          </div>
                        );
                      });
                    })()}
                  </div>
                  <div className="col-md-6">
                    <h6 className="fw-bold">MMAS-4 Adherence Level</h6>
                    {(() => {
                      const dist = {};
                      mmasPatients.forEach(p => { dist[p.adherence_level] = (dist[p.adherence_level] || 0) + 1; });
                      const cm = { 'High adherence': 'success', 'Medium adherence': 'warning', 'Low adherence': 'danger' };
                      return Object.entries(dist).sort((a, b) => b[1] - a[1]).map(([lvl, cnt], i) => {
                        const pct = mmasPatients.length ? ((cnt / mmasPatients.length) * 100).toFixed(0) : 0;
                        return (
                          <div key={i} className="d-flex align-items-center mb-2">
                            <span className="me-2 small" style={{ minWidth: 160 }}>{lvl}</span>
                            <div className="flex-grow-1">
                              <div className="progress" style={{ height: 20 }}>
                                <div className={`progress-bar bg-${cm[lvl] || 'secondary'}`} style={{ width: `${pct}%` }}>{cnt} ({pct}%)</div>
                              </div>
                            </div>
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* COGNITIVE TESTS TAB */}
      {tab === 'cognitive' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold bg-purple text-white" style={{ background: '#7c3aed' }}>
                Stroop Color-Word Test — {stroopPatients.length} patients
              </div>
              <div className="card-body p-0" style={{ maxHeight: 380, overflowY: 'auto' }}>
                <ScaleTable
                  patients={stroopPatients}
                  cols={[
                    { label: 'Interference', render: p => <span className="fw-bold">{p.interference_score?.toFixed?.(1) ?? p.interference_score ?? '—'}</span> },
                    { label: 'Severity', render: p => <SeverityBadge severity={p.severity} /> },
                    { label: 'Color RT', render: p => p.color_rt ? `${p.color_rt}s` : '—' },
                    { label: 'Word RT', render: p => p.word_rt ? `${p.word_rt}s` : '—' },
                  ]}
                />
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold bg-primary text-white">
                Trail Making Test — {tmtPatients.length} patients
              </div>
              <div className="card-body p-0" style={{ maxHeight: 380, overflowY: 'auto' }}>
                <ScaleTable
                  patients={tmtPatients}
                  cols={[
                    { label: 'TMT-A (s)', render: p => p.tmt_a_seconds ?? '—' },
                    { label: 'TMT-B (s)', render: p => p.tmt_b_seconds ?? '—' },
                    { label: 'B/A Ratio', render: p => p.b_a_ratio?.toFixed?.(2) ?? p.b_a_ratio ?? '—' },
                    { label: 'Severity', render: p => <SeverityBadge severity={p.severity} /> },
                  ]}
                />
              </div>
            </div>
          </div>
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Cognitive Domain Summary</div>
              <div className="card-body">
                <div className="row g-2">
                  {[
                    { domain: 'Attention (Stroop)', api: 'stroop', n: stroopPatients.length },
                    { domain: 'Executive (TMT)', api: 'tmt', n: tmtPatients.length },
                    { domain: 'Working Memory (Digit Span)', api: 'digit-span', n: 41 },
                    { domain: 'Executive (WCST)', api: 'wcst', n: 41 },
                    { domain: 'Working Memory (N-Back)', api: 'nback', n: 41 },
                    { domain: 'Inhibition (Go/No-Go)', api: 'gonogo', n: 41 },
                    { domain: 'Sustained Attention (CPT)', api: 'cpt', n: 41 },
                    { domain: 'Visuospatial (Clock Drawing)', api: 'clock-drawing', n: 41 },
                    { domain: 'Verbal Memory (RAVLT)', api: 'ravlt', n: 41 },
                    { domain: 'Language (Verbal Fluency)', api: 'verbal-fluency', n: 41 },
                    { domain: 'Sleep Quality (PSQI)', api: 'psqi', n: 41 },
                  ].map((d, i) => (
                    <div key={i} className="col-md-4">
                      <div className="card border-0 bg-light">
                        <div className="card-body py-2 px-3">
                          <div className="fw-semibold small">{d.domain}</div>
                          <div className="text-muted" style={{ fontSize: '0.72rem' }}>
                            <span className="badge bg-secondary me-1">n={d.n}</span>
                            <span className="text-muted">/api/neuro-scales/{d.api}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
