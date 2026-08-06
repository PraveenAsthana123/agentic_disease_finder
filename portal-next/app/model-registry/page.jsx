'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const accColor = a => a == null ? 'secondary' : a >= 70 ? 'success' : a >= 55 ? 'warning' : 'danger';
const sizeLabel = kb => kb >= 1024 ? `${(kb/1024).toFixed(1)} MB` : `${kb} KB`;
const diseaseIcon = d => ({
  alzheimer:'🧠', autism:'🔵', depression:'💙', epilepsy:'⚡',
  parkinson:'🤝', schizophrenia:'🌀', stress:'😰', other:'📦',
}[d] || '📦');

export default function ModelRegistryDashboard() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [search, setSearch]   = useState('');
  const [filterDisease, setFilterDisease] = useState('all');
  const [filterAlgo, setFilterAlgo]       = useState('all');
  const [filterFolder, setFilterFolder]   = useState('all');
  const [sortBy, setSortBy]   = useState('mtime');
  const [sortDir, setSortDir] = useState('desc');
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/model-registry/overview`).then(r => r.json()),
      fetch(`${API}/api/model-registry/breakdown`).then(r => r.json()),
      fetch(`${API}/api/model-registry/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err)  return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-3">Loading model registry…</div>;

  const TABS = [
    { id: 'overview',     label: '📊 Overview' },
    { id: 'artifacts',    label: '📦 Artifacts' },
    { id: 'by_disease',   label: '🧠 By Disease' },
    { id: 'by_algo',      label: '⚙️ By Algorithm' },
    { id: 'definitions',  label: '📖 Definitions' },
  ];

  const allArtifacts  = bd?.artifacts || [];
  const diseases      = ['all', ...new Set(allArtifacts.map(a => a.disease))].sort();
  const algos         = ['all', ...new Set(allArtifacts.map(a => a.algo))].sort();
  const folders       = ['all', 'saved_models', 'models'];

  const filtered = allArtifacts
    .filter(a => filterDisease === 'all' || a.disease === filterDisease)
    .filter(a => filterAlgo    === 'all' || a.algo    === filterAlgo)
    .filter(a => filterFolder  === 'all' || a.folder  === filterFolder)
    .filter(a => !search || a.filename.toLowerCase().includes(search.toLowerCase()) ||
                            a.disease.includes(search.toLowerCase()) ||
                            a.algo.toLowerCase().includes(search.toLowerCase()))
    .sort((x, y) => {
      let cmp = 0;
      if (sortBy === 'accuracy_pct') cmp = (x.accuracy_pct ?? -1) - (y.accuracy_pct ?? -1);
      else if (sortBy === 'size_kb') cmp = x.size_kb - y.size_kb;
      else if (sortBy === 'mtime')   cmp = x.mtime.localeCompare(y.mtime);
      else if (sortBy === 'disease') cmp = x.disease.localeCompare(y.disease);
      else if (sortBy === 'algo')    cmp = x.algo.localeCompare(y.algo);
      else if (sortBy === 'folder')  cmp = x.folder.localeCompare(y.folder);
      return sortDir === 'asc' ? cmp : -cmp;
    });

  const toggleSort = col => {
    if (sortBy === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortBy(col); setSortDir('desc'); }
  };
  const sortArrow = col => sortBy === col ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '';

  const kpis = ov.kpis || {};
  const byDisease = ov.by_disease || [];
  const byAlgo    = ov.by_algo    || [];
  const byFolder  = ov.by_folder  || [];

  // Disease detail grouping
  const diseaseGroups = {};
  allArtifacts.forEach(a => {
    if (!diseaseGroups[a.disease]) diseaseGroups[a.disease] = [];
    diseaseGroups[a.disease].push(a);
  });

  return (
    <div className="p-3">
      <h3>📦 Model Registry</h3>
      <p className="text-muted">
        {kpis.total_artifacts} model artifacts · {kpis.diseases} diseases · {kpis.algorithms} algorithms ·{' '}
        avg accuracy {kpis.avg_accuracy_pct ?? 'N/A'}% · {kpis.total_size_mb} MB total storage
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          <div className="row g-3 mb-4">
            {[
              { label: 'Total Artifacts',  val: kpis.total_artifacts,          color: 'primary' },
              { label: 'Diseases Covered', val: kpis.diseases,                  color: 'info' },
              { label: 'Algorithms',       val: kpis.algorithms,                color: 'secondary' },
              { label: 'Avg Accuracy',     val: kpis.avg_accuracy_pct != null ? `${kpis.avg_accuracy_pct}%` : 'N/A', color: accColor(kpis.avg_accuracy_pct) },
              { label: 'Total Storage',    val: `${kpis.total_size_mb} MB`,     color: 'warning' },
            ].map(c => (
              <div key={c.label} className="col-6 col-md-4 col-lg-2">
                <div className={`card border-${c.color} text-center`}>
                  <div className="card-body py-2">
                    <div className={`fs-4 fw-bold text-${c.color}`}>{c.val}</div>
                    <div className="small text-muted">{c.label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row g-3">
            {/* By Disease */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">🧠 By Disease</div>
                <div className="card-body p-2">
                  {byDisease.map(r => (
                    <div key={r.disease} className="d-flex align-items-center mb-2">
                      <span className="me-2">{diseaseIcon(r.disease)}</span>
                      <span className="me-auto text-capitalize small">{r.disease}</span>
                      <span className="badge bg-primary">{r.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* By Algorithm */}
            <div className="col-md-5">
              <div className="card h-100">
                <div className="card-header fw-semibold">⚙️ By Algorithm</div>
                <div className="card-body p-2" style={{maxHeight:260,overflowY:'auto'}}>
                  {byAlgo.map(r => {
                    const maxC = byAlgo[0]?.count || 1;
                    return (
                      <div key={r.algo} className="mb-1">
                        <div className="d-flex justify-content-between small mb-1">
                          <span className="text-truncate" style={{maxWidth:160}}>{r.algo}</span>
                          <span className="text-muted">{r.count}</span>
                        </div>
                        <div className="progress" style={{height:6}}>
                          <div className="progress-bar bg-info" style={{width:`${(r.count/maxC)*100}%`}} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* By Folder */}
            <div className="col-md-3">
              <div className="card h-100">
                <div className="card-header fw-semibold">📁 By Folder</div>
                <div className="card-body p-2">
                  {byFolder.map(r => (
                    <div key={r.folder} className="d-flex align-items-center mb-2">
                      <span className="me-auto small font-monospace">{r.folder}/</span>
                      <span className="badge bg-secondary">{r.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── ARTIFACTS TABLE ── */}
      {tab === 'artifacts' && (
        <div>
          <div className="row g-2 mb-3">
            <div className="col-md-3">
              <input className="form-control form-control-sm" placeholder="Search filename, disease, algo…"
                value={search} onChange={e => setSearch(e.target.value)} />
            </div>
            <div className="col-auto">
              <select className="form-select form-select-sm" value={filterDisease} onChange={e => setFilterDisease(e.target.value)}>
                {diseases.map(d => <option key={d} value={d}>{d === 'all' ? 'All diseases' : d}</option>)}
              </select>
            </div>
            <div className="col-auto">
              <select className="form-select form-select-sm" value={filterAlgo} onChange={e => setFilterAlgo(e.target.value)}>
                {algos.map(a => <option key={a} value={a}>{a === 'all' ? 'All algos' : a}</option>)}
              </select>
            </div>
            <div className="col-auto">
              <select className="form-select form-select-sm" value={filterFolder} onChange={e => setFilterFolder(e.target.value)}>
                {folders.map(f => <option key={f} value={f}>{f === 'all' ? 'All folders' : f + '/'}</option>)}
              </select>
            </div>
            <div className="col-auto ms-auto text-muted small align-self-center">
              {filtered.length} / {allArtifacts.length} artifacts
            </div>
          </div>

          <div className="table-responsive" style={{maxHeight:560,overflowY:'auto'}}>
            <table className="table table-sm table-hover mb-0">
              <thead className="table-dark sticky-top">
                <tr>
                  <th style={{cursor:'pointer'}} onClick={() => toggleSort('folder')}>Folder{sortArrow('folder')}</th>
                  <th style={{cursor:'pointer'}} onClick={() => toggleSort('disease')}>Disease{sortArrow('disease')}</th>
                  <th style={{cursor:'pointer'}} onClick={() => toggleSort('algo')}>Algorithm{sortArrow('algo')}</th>
                  <th>Filename</th>
                  <th style={{cursor:'pointer'}} onClick={() => toggleSort('accuracy_pct')}>Acc%{sortArrow('accuracy_pct')}</th>
                  <th style={{cursor:'pointer'}} onClick={() => toggleSort('size_kb')}>Size{sortArrow('size_kb')}</th>
                  <th style={{cursor:'pointer'}} onClick={() => toggleSort('mtime')}>Date{sortArrow('mtime')}</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((a, i) => (
                  <tr key={i}>
                    <td><span className="badge bg-secondary font-monospace">{a.folder}/</span></td>
                    <td><span>{diseaseIcon(a.disease)}</span> <span className="text-capitalize">{a.disease}</span></td>
                    <td><code className="small">{a.algo}</code></td>
                    <td className="font-monospace small text-truncate" style={{maxWidth:220}} title={a.filename}>{a.filename}</td>
                    <td>
                      {a.accuracy_pct != null
                        ? <span className={`badge bg-${accColor(a.accuracy_pct)}`}>{a.accuracy_pct}%</span>
                        : <span className="text-muted">—</span>}
                    </td>
                    <td className="small">{sizeLabel(a.size_kb)}</td>
                    <td className="small text-muted">{a.mtime}</td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr><td colSpan={7} className="text-center text-muted py-3">No artifacts match filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── BY DISEASE ── */}
      {tab === 'by_disease' && (
        <div className="row g-3">
          {Object.entries(diseaseGroups).sort(([a],[b]) => a.localeCompare(b)).map(([disease, arts]) => {
            const accs = arts.filter(a => a.accuracy_pct != null).map(a => a.accuracy_pct);
            const avgAcc = accs.length ? Math.round(accs.reduce((s,v) => s+v, 0) / accs.length) : null;
            const maxAcc = accs.length ? Math.max(...accs) : null;
            const totalKb = arts.reduce((s,a) => s+a.size_kb, 0);
            const algSet = [...new Set(arts.map(a => a.algo))];
            const prodCount = arts.filter(a => a.folder === 'models').length;
            return (
              <div key={disease} className="col-md-6 col-lg-4">
                <div className="card h-100">
                  <div className="card-header fw-semibold">
                    {diseaseIcon(disease)} <span className="text-capitalize">{disease}</span>
                  </div>
                  <div className="card-body">
                    <div className="row text-center mb-2">
                      <div className="col">
                        <div className="fw-bold">{arts.length}</div>
                        <div className="small text-muted">Artifacts</div>
                      </div>
                      <div className="col">
                        <div className={`fw-bold text-${accColor(avgAcc)}`}>{avgAcc ?? '—'}%</div>
                        <div className="small text-muted">Avg Acc</div>
                      </div>
                      <div className="col">
                        <div className={`fw-bold text-${accColor(maxAcc)}`}>{maxAcc ?? '—'}%</div>
                        <div className="small text-muted">Max Acc</div>
                      </div>
                      <div className="col">
                        <div className="fw-bold">{sizeLabel(totalKb)}</div>
                        <div className="small text-muted">Storage</div>
                      </div>
                    </div>
                    <div className="small text-muted mb-1">
                      <strong>Algorithms ({algSet.length}):</strong> {algSet.slice(0,6).join(', ')}{algSet.length>6?' …':''}
                    </div>
                    {prodCount > 0 && (
                      <span className="badge bg-success">✅ {prodCount} in production</span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── BY ALGORITHM ── */}
      {tab === 'by_algo' && (
        <div>
          <div className="table-responsive">
            <table className="table table-sm table-bordered">
              <thead className="table-dark">
                <tr>
                  <th>Algorithm</th>
                  <th>Total</th>
                  <th>Avg Acc%</th>
                  <th>Max Acc%</th>
                  <th>Total Size</th>
                  <th>Diseases</th>
                  <th>Production</th>
                </tr>
              </thead>
              <tbody>
                {[...new Set(allArtifacts.map(a => a.algo))].sort().map(algo => {
                  const arts = allArtifacts.filter(a => a.algo === algo);
                  const accs = arts.filter(a => a.accuracy_pct != null).map(a => a.accuracy_pct);
                  const avgAcc = accs.length ? Math.round(accs.reduce((s,v)=>s+v,0)/accs.length) : null;
                  const maxAcc = accs.length ? Math.max(...accs) : null;
                  const totalKb = arts.reduce((s,a) => s+a.size_kb, 0);
                  const dss = [...new Set(arts.map(a => a.disease))];
                  const prodCount = arts.filter(a => a.folder === 'models').length;
                  return (
                    <tr key={algo}>
                      <td><code>{algo}</code></td>
                      <td>{arts.length}</td>
                      <td>
                        {avgAcc != null
                          ? <span className={`badge bg-${accColor(avgAcc)}`}>{avgAcc}%</span>
                          : '—'}
                      </td>
                      <td>
                        {maxAcc != null
                          ? <span className={`badge bg-${accColor(maxAcc)}`}>{maxAcc}%</span>
                          : '—'}
                      </td>
                      <td>{sizeLabel(totalKb)}</td>
                      <td className="small">{dss.map(d => `${diseaseIcon(d)} ${d}`).join(', ')}</td>
                      <td>
                        {prodCount > 0
                          ? <span className="badge bg-success">{prodCount}</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div>
          <h5>{defs.title}</h5>
          <div className="row g-3 mb-3">
            <div className="col-md-7">
              <div className="card">
                <div className="card-header fw-semibold">Artifact Types & Naming Convention</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Type</th><th>Pattern</th><th>Example</th></tr>
                    </thead>
                    <tbody>
                      {(defs.artifact_types || []).map((at, i) => (
                        <tr key={i}>
                          <td className="small">{at.name}</td>
                          <td><code className="small">{at.pattern}</code></td>
                          <td className="font-monospace small text-muted">{at.example}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-5">
              <div className="card mb-3">
                <div className="card-header fw-semibold">Storage Folders</div>
                <div className="card-body p-2">
                  {Object.entries(defs.folders || {}).map(([folder, desc]) => (
                    <div key={folder} className="mb-2">
                      <code className="me-1">{folder}/</code>
                      <span className="small text-muted">{desc}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card">
                <div className="card-header fw-semibold">Accuracy Encoding</div>
                <div className="card-body small text-muted">{defs.accuracy_encoding}</div>
              </div>
            </div>
          </div>
          <div className="card mb-3">
            <div className="card-header fw-semibold">Model Lifecycle</div>
            <div className="card-body">
              <div className="d-flex flex-wrap gap-2 align-items-center">
                {(defs.lifecycle || []).map((step, i) => (
                  <span key={i}>
                    <span className="badge bg-primary">{i+1}. {step}</span>
                    {i < defs.lifecycle.length - 1 && <span className="text-muted mx-1">→</span>}
                  </span>
                ))}
              </div>
            </div>
          </div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Diseases</div>
                <div className="card-body">
                  {(defs.diseases || []).map(d => (
                    <span key={d} className="badge bg-secondary me-1 mb-1">{diseaseIcon(d)} {d}</span>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Algorithms</div>
                <div className="card-body">
                  {(defs.algorithms || []).map(a => (
                    <span key={a} className="badge bg-info text-dark me-1 mb-1">{a}</span>
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
