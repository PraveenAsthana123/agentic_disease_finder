'use client';
import { useState, useEffect } from 'react';

const DISEASE_COLOR = {
  epilepsy:      '#7c3aed',
  alzheimer:     '#2563eb',
  parkinson:     '#059669',
  depression:    '#d97706',
  schizophrenia: '#dc2626',
  autism:        '#0891b2',
  stress:        '#6b7280',
  other:         '#94a3b8',
};

const ALGO_COLOR = [
  '#6366f1','#10b981','#f59e0b','#ef4444','#3b82f6',
  '#8b5cf6','#14b8a6','#f97316','#84cc16','#ec4899',
  '#06b6d4','#a855f7',
];

function KPI({ label, value, sub }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 10, padding: '16px 20px', minWidth: 140 }}>
      <div style={{ fontSize: 26, fontWeight: 700, color: '#f1f5f9' }}>{value ?? '—'}</div>
      <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function Bar({ label, count, max, color }) {
  const pct = max ? (count / max) * 100 : 0;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, color: '#cbd5e1', marginBottom: 3 }}>
        <span>{label}</span><span style={{ fontWeight: 600 }}>{count}</span>
      </div>
      <div style={{ background: '#334155', borderRadius: 4, height: 8, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, background: color || '#6366f1', height: '100%', borderRadius: 4, transition: 'width 0.4s' }} />
      </div>
    </div>
  );
}

const TABS = ['Overview', 'By Disease', 'All Artifacts', 'Definitions'];

export default function ModelRegistry() {
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [tab, setTab]             = useState(0);
  const [err, setErr]             = useState(null);
  const [search, setSearch]       = useState('');
  const [filterDis, setFilterDis] = useState('');
  const [filterAlgo, setFilterAlgo] = useState('');
  const [sortKey, setSortKey]     = useState('disease');
  const [sortDir, setSortDir]     = useState(1);

  useEffect(() => {
    Promise.all([
      fetch('/api/model-registry/overview').then(r => r.json()),
      fetch('/api/model-registry/breakdown').then(r => r.json()),
      fetch('/api/model-registry/definitions').then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefs(df);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err)      return <div style={{ padding: 40, color: '#f87171' }}>Error: {err}</div>;
  if (!overview) return <div style={{ padding: 40, color: '#94a3b8' }}>Loading model registry...</div>;

  const { kpis, by_disease, by_algo } = overview;
  const artifacts = breakdown?.artifacts || [];

  const maxDis  = Math.max(...(by_disease || []).map(d => d.count), 1);
  const maxAlgo = Math.max(...(by_algo    || []).map(a => a.count), 1);

  const allDiseases = [...new Set(artifacts.map(a => a.disease))].sort();
  const allAlgos    = [...new Set(artifacts.map(a => a.algo))].sort();

  const filtered = artifacts.filter(a => {
    if (filterDis  && a.disease !== filterDis)  return false;
    if (filterAlgo && a.algo    !== filterAlgo) return false;
    if (search) {
      const q = search.toLowerCase();
      return a.filename.toLowerCase().includes(q) || a.disease.includes(q) || a.algo.toLowerCase().includes(q);
    }
    return true;
  }).sort((a, b) => {
    const va = a[sortKey] ?? '', vb = b[sortKey] ?? '';
    if (typeof va === 'number') return (va - vb) * sortDir;
    return String(va).localeCompare(String(vb)) * sortDir;
  });

  function toggleSort(key) {
    if (sortKey === key) setSortDir(d => -d);
    else { setSortKey(key); setSortDir(1); }
  }
  function th(key, label) {
    const active = sortKey === key;
    return (
      <th onClick={() => toggleSort(key)} style={{ cursor: 'pointer', padding: '8px 10px', color: active ? '#7c3aed' : '#94a3b8', userSelect: 'none', whiteSpace: 'nowrap' }}>
        {label} {active ? (sortDir > 0 ? '▲' : '▼') : ''}
      </th>
    );
  }

  const byDisease = {};
  for (const a of artifacts) {
    if (!byDisease[a.disease]) byDisease[a.disease] = [];
    byDisease[a.disease].push(a);
  }

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', padding: 24, fontFamily: 'system-ui,sans-serif' }}>
      <h2 style={{ marginBottom: 6, fontSize: 22, fontWeight: 700 }}>
        Model Registry
        <span style={{ marginLeft: 12, background: '#1e293b', borderRadius: 6, padding: '3px 10px', fontSize: 13, fontWeight: 400, color: '#94a3b8' }}>
          {kpis.total_artifacts} artifacts · {kpis.diseases} diseases
        </span>
      </h2>
      <p style={{ color: '#64748b', marginBottom: 20, fontSize: 14 }}>
        Real artifact scan — saved_models/ + models/ directories. Filename-derived metadata.
      </p>

      <div style={{ display: 'flex', gap: 6, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={i} onClick={() => setTab(i)} style={{
            padding: '7px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#7c3aed' : '#1e293b',
            color: tab === i ? '#fff' : '#94a3b8', fontSize: 14, fontWeight: 500,
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && (
        <>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 28 }}>
            <KPI label="Total Artifacts" value={kpis.total_artifacts} />
            <KPI label="Diseases" value={kpis.diseases} />
            <KPI label="Algorithms" value={kpis.algorithms} />
            <KPI label="Avg Accuracy" value={kpis.avg_accuracy_pct != null ? `${kpis.avg_accuracy_pct}%` : 'N/A'} sub="filename-encoded" />
            <KPI label="Total Size" value={`${kpis.total_size_mb} MB`} />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, maxWidth: 900 }}>
            <div style={{ background: '#1e293b', borderRadius: 10, padding: 20 }}>
              <h6 style={{ color: '#94a3b8', marginBottom: 14, fontWeight: 600 }}>Artifacts by Disease</h6>
              {(by_disease || []).map(d => (
                <Bar key={d.disease} label={d.disease} count={d.count} max={maxDis}
                     color={DISEASE_COLOR[d.disease] || '#6366f1'} />
              ))}
            </div>

            <div style={{ background: '#1e293b', borderRadius: 10, padding: 20 }}>
              <h6 style={{ color: '#94a3b8', marginBottom: 14, fontWeight: 600 }}>Top Algorithms</h6>
              {(by_algo || []).map((a, i) => (
                <Bar key={a.algo} label={a.algo} count={a.count} max={maxAlgo}
                     color={ALGO_COLOR[i % ALGO_COLOR.length]} />
              ))}
            </div>
          </div>
        </>
      )}

      {tab === 1 && (
        <div>
          {Object.entries(byDisease).sort(([a], [b]) => a.localeCompare(b)).map(([dis, arts]) => {
            const algoGroups = {};
            for (const a of arts) {
              if (!algoGroups[a.algo]) algoGroups[a.algo] = 0;
              algoGroups[a.algo]++;
            }
            return (
              <div key={dis} style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                  <span style={{ width: 12, height: 12, borderRadius: '50%', background: DISEASE_COLOR[dis] || '#6366f1', display: 'inline-block' }} />
                  <strong style={{ fontSize: 16, textTransform: 'capitalize' }}>{dis}</strong>
                  <span style={{ color: '#64748b', fontSize: 13 }}>{arts.length} artifacts</span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {Object.entries(algoGroups).sort(([, a], [, b]) => b - a).map(([algo, cnt]) => (
                    <span key={algo} style={{
                      background: '#0f172a', borderRadius: 6, padding: '3px 10px',
                      fontSize: 12, color: '#cbd5e1', border: '1px solid #334155',
                    }}>{algo} <strong>{cnt}</strong></span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {tab === 2 && (
        <>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
            <input value={search} onChange={e => setSearch(e.target.value)}
              placeholder="Search filename, disease, algo..."
              style={{ padding: '7px 12px', borderRadius: 7, border: '1px solid #334155', background: '#1e293b', color: '#e2e8f0', fontSize: 13, minWidth: 240 }} />
            <select value={filterDis} onChange={e => setFilterDis(e.target.value)}
              style={{ padding: '7px 12px', borderRadius: 7, border: '1px solid #334155', background: '#1e293b', color: '#e2e8f0', fontSize: 13 }}>
              <option value="">All Diseases</option>
              {allDiseases.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
            <select value={filterAlgo} onChange={e => setFilterAlgo(e.target.value)}
              style={{ padding: '7px 12px', borderRadius: 7, border: '1px solid #334155', background: '#1e293b', color: '#e2e8f0', fontSize: 13 }}>
              <option value="">All Algorithms</option>
              {allAlgos.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
            <span style={{ color: '#64748b', fontSize: 13, alignSelf: 'center' }}>{filtered.length} of {artifacts.length}</span>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#1e293b' }}>
                <tr>
                  {th('disease', 'Disease')}
                  {th('algo', 'Algorithm')}
                  {th('filename', 'Filename')}
                  {th('folder', 'Folder')}
                  {th('accuracy_pct', 'Acc %')}
                  {th('size_kb', 'Size KB')}
                  {th('date', 'Train Date')}
                </tr>
              </thead>
              <tbody>
                {filtered.map((a, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#0f172a' : '#1a2535', borderBottom: '1px solid #1e293b' }}>
                    <td style={{ padding: '7px 10px' }}>
                      <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: DISEASE_COLOR[a.disease] || '#6366f1', marginRight: 6 }} />
                      {a.disease}
                    </td>
                    <td style={{ padding: '7px 10px', color: '#a5b4fc' }}>{a.algo}</td>
                    <td style={{ padding: '7px 10px', color: '#94a3b8', fontFamily: 'monospace', fontSize: 11 }}>{a.filename}</td>
                    <td style={{ padding: '7px 10px', color: '#64748b' }}>{a.folder}/</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right', color: a.accuracy_pct != null ? '#4ade80' : '#475569' }}>
                      {a.accuracy_pct != null ? `${a.accuracy_pct}%` : '—'}
                    </td>
                    <td style={{ padding: '7px 10px', textAlign: 'right', color: '#94a3b8' }}>{a.size_kb}</td>
                    <td style={{ padding: '7px 10px', color: '#64748b' }}>{a.date || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {tab === 3 && defs && (
        <div style={{ maxWidth: 800 }}>
          <h5 style={{ color: '#e2e8f0', marginBottom: 16 }}>{defs.title}</h5>

          <div style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 16 }}>
            <h6 style={{ color: '#94a3b8', marginBottom: 12 }}>Artifact Types</h6>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead><tr style={{ color: '#64748b' }}>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Type</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Pattern</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Example</th>
              </tr></thead>
              <tbody>
                {defs.artifact_types.map((t, i) => (
                  <tr key={i} style={{ borderTop: '1px solid #334155' }}>
                    <td style={{ padding: '7px 10px', color: '#c4b5fd' }}>{t.name}</td>
                    <td style={{ padding: '7px 10px', fontFamily: 'monospace', color: '#7dd3fc', fontSize: 11 }}>{t.pattern}</td>
                    <td style={{ padding: '7px 10px', fontFamily: 'monospace', color: '#94a3b8', fontSize: 11 }}>{t.example}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 16 }}>
            <h6 style={{ color: '#94a3b8', marginBottom: 10 }}>Storage Folders</h6>
            {Object.entries(defs.folders || {}).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 10 }}>
                <code style={{ color: '#a5b4fc', fontSize: 13 }}>{k}/</code>
                <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 3 }}>{v}</div>
              </div>
            ))}
          </div>

          <div style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 16 }}>
            <h6 style={{ color: '#94a3b8', marginBottom: 10 }}>Accuracy Encoding</h6>
            <p style={{ color: '#cbd5e1', fontSize: 13, margin: 0 }}>{defs.accuracy_encoding}</p>
          </div>

          <div style={{ background: '#1e293b', borderRadius: 10, padding: 20 }}>
            <h6 style={{ color: '#94a3b8', marginBottom: 10 }}>Model Lifecycle</h6>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
              {(defs.lifecycle || []).map((step, i) => (
                <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ background: '#0f172a', borderRadius: 6, padding: '4px 12px', fontSize: 12, color: '#e2e8f0', border: '1px solid #334155' }}>{step}</span>
                  {i < defs.lifecycle.length - 1 && <span style={{ color: '#475569' }}>→</span>}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
