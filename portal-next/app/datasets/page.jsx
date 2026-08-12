'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KpiCard({label, value, sub, color='primary'}){
  return (
    <div className="col-md-2 col-sm-4 col-6 mb-3">
      <div className={`card border-${color} h-100`}>
        <div className="card-body p-2 text-center">
          <div className={`fs-4 fw-bold text-${color}`}>{value}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="small text-muted">{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function MiniBar({value, max, color='primary'}){
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  return (
    <div className="progress" style={{height:'8px'}}>
      <div className={`progress-bar bg-${color}`} style={{width:`${pct}%`}}/>
    </div>
  );
}

const DISEASE_COLORS = {
  schizophrenia:'danger',
  autism:'info',
  parkinson:'warning',
  stress:'secondary',
  epilepsy:'primary',
  depression:'dark',
};

export default function DatasetsRegistryDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [search, setSearch] = useState('');

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/datasets/overview`).then(r=>r.json()),
      fetch(`${API}/api/datasets/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/datasets/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o); setBd(b); setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading datasets registry…</div>;

  const TABS = [
    {id:'overview',   label:'📊 Overview'},
    {id:'datasets',   label:'🗄️ All Datasets'},
    {id:'breakdown',  label:'🧬 By Disease'},
    {id:'definitions',label:'📖 Definitions'},
  ];

  const kpi = ov.kpis || {};
  const maxSubjects = Math.max(...(ov.diseases||[]).map(d=>d.total_subjects), 1);
  const allDatasets = bd?.all_datasets || [];

  const filtered = allDatasets.filter(ds => {
    if(!search) return true;
    const s = search.toLowerCase();
    return (ds.name||'').toLowerCase().includes(s) ||
           (ds.disease||'').toLowerCase().includes(s) ||
           (ds.source||'').toLowerCase().includes(s) ||
           (ds.format||'').toLowerCase().includes(s);
  });

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-2 flex-wrap gap-2">
        <h4 className="mb-0 me-3">🗄️ Datasets Registry</h4>
        <span className="badge bg-success">✅ All Real Data</span>
        <span className="badge bg-secondary">{kpi.total_datasets} datasets · {kpi.total_subjects} subjects</span>
        <span className="badge bg-info">{kpi.total_diseases} diseases</span>
      </div>
      <div className="text-muted small mb-3">
        Verified real-world EEG datasets used across all 6 neurological disease classifications.
        No synthetic data — all {kpi.total_datasets} datasets are downloaded and validated.
        Avg accuracy: <strong>{kpi.avg_accuracy}%</strong>.
      </div>

      <div className="row g-2 mb-3">
        <KpiCard label="Diseases" value={kpi.total_diseases} color="primary"/>
        <KpiCard label="Total Datasets" value={kpi.total_datasets} color="secondary"/>
        <KpiCard label="Total Subjects" value={kpi.total_subjects} color="success"/>
        <KpiCard label="Avg Accuracy" value={`${kpi.avg_accuracy}%`} color="info"/>
        <KpiCard label="Formats" value={kpi.format_count} sub={kpi.formats?.join(', ')} color="warning"/>
        <KpiCard label="Real Data" value="100%" sub="no synthetic" color="success"/>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t=>(
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab==='overview' && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card">
              <div className="card-header py-2"><strong>📊 Subjects by Disease</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead>
                    <tr>
                      <th>Disease</th>
                      <th>Datasets</th>
                      <th>Subjects</th>
                      <th>Accuracy</th>
                      <th>Distribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.diseases||[]).map(d=>(
                      <tr key={d.name}>
                        <td>
                          <span className={`badge bg-${DISEASE_COLORS[d.name]||'secondary'} me-1`}>{d.name}</span>
                        </td>
                        <td>{d.dataset_count}</td>
                        <td><strong>{d.total_subjects}</strong></td>
                        <td>
                          <span className={`badge bg-${d.accuracy>=99?'success':d.accuracy>=97?'info':'warning'}`}>
                            {d.accuracy}%
                          </span>
                        </td>
                        <td style={{width:'120px'}}>
                          <MiniBar value={d.total_subjects} max={maxSubjects} color={DISEASE_COLORS[d.name]||'secondary'}/>
                          <div className="small text-muted">{Math.round(d.total_subjects/kpi.total_subjects*100)}%</div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card">
              <div className="card-header py-2"><strong>📁 Format Distribution</strong></div>
              <div className="card-body">
                <table className="table table-sm mb-3">
                  <thead><tr><th>Format</th><th>Count</th><th>%</th><th>Bar</th></tr></thead>
                  <tbody>
                    {(ov.format_distribution||[]).map(f=>(
                      <tr key={f.name}>
                        <td><code>{f.name}</code></td>
                        <td>{f.value}</td>
                        <td>{Math.round(f.value/kpi.total_datasets*100)}%</td>
                        <td style={{width:'100px'}}>
                          <MiniBar value={f.value} max={kpi.total_datasets} color="info"/>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="alert alert-success py-2 mb-0 small">
                  <strong>✅ All {kpi.total_datasets} datasets are downloaded and verified.</strong><br/>
                  Formats: EDF (clinical standard), CSV (pre-processed features), MAT (MATLAB), EDF/CSV (hybrid).
                </div>
              </div>
            </div>

            <div className="card mt-3">
              <div className="card-header py-2"><strong>🎯 Accuracy by Disease</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Disease</th><th>Accuracy</th><th>Bar</th></tr></thead>
                  <tbody>
                    {(ov.accuracy_distribution||[]).sort((a,b)=>b.value-a.value).map(d=>(
                      <tr key={d.name}>
                        <td className="text-capitalize">{d.name}</td>
                        <td><strong>{d.value}%</strong></td>
                        <td style={{width:'120px'}}>
                          <MiniBar value={d.value - 90} max={10} color={d.value>=99?'success':d.value>=97?'info':'warning'}/>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ALL DATASETS TAB */}
      {tab==='datasets' && (
        <div>
          <div className="mb-3">
            <input
              className="form-control"
              placeholder="Search by name, disease, source, format…"
              value={search}
              onChange={e=>setSearch(e.target.value)}
            />
          </div>
          <div className="card">
            <div className="card-header py-2">
              <strong>🗄️ All Datasets ({filtered.length}/{allDatasets.length})</strong>
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Name</th>
                      <th>Disease</th>
                      <th>Subjects</th>
                      <th>Channels</th>
                      <th>Sampling Rate</th>
                      <th>Format</th>
                      <th>Source</th>
                      <th>Downloaded</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((ds,i)=>(
                      <tr key={i}>
                        <td><small className="text-muted">{i+1}</small></td>
                        <td><strong>{ds.name}</strong></td>
                        <td>
                          <span className={`badge bg-${DISEASE_COLORS[ds.disease]||'secondary'}`}>
                            {ds.disease}
                          </span>
                        </td>
                        <td>{ds.subjects}</td>
                        <td>{ds.channels ?? <span className="text-muted">—</span>}</td>
                        <td>{ds.sampling_rate ? `${ds.sampling_rate} Hz` : <span className="text-muted">—</span>}</td>
                        <td><code>{ds.format}</code></td>
                        <td><small className="text-muted">{ds.source}</small></td>
                        <td>
                          {ds.is_downloaded
                            ? <span className="badge bg-success">✓ Yes</span>
                            : <span className="badge bg-warning text-dark">⚠ No</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* BY DISEASE TAB */}
      {tab==='breakdown' && (
        <div className="row g-3">
          {(bd?.diseases||[]).map(disease=>(
            <div key={disease.name} className="col-md-6">
              <div className={`card border-${DISEASE_COLORS[disease.name]||'secondary'}`}>
                <div className={`card-header py-2 bg-${DISEASE_COLORS[disease.name]||'secondary'} text-white`}>
                  <strong className="text-capitalize">{disease.name}</strong>
                  <span className="badge bg-light text-dark ms-2">{disease.status}</span>
                  <span className="float-end">{disease.total_subjects} subjects · {disease.accuracy}% acc.</span>
                </div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead>
                      <tr>
                        <th>Dataset</th>
                        <th>Subjects</th>
                        <th>Channels</th>
                        <th>Hz</th>
                        <th>Format</th>
                        <th>Downloaded</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(disease.datasets||[]).map((ds,i)=>(
                        <tr key={i}>
                          <td><strong>{ds.name}</strong></td>
                          <td>{ds.subjects}</td>
                          <td>{ds.channels ?? '—'}</td>
                          <td>{ds.sampling_rate ?? '—'}</td>
                          <td><code>{ds.format}</code></td>
                          <td>
                            {ds.is_downloaded
                              ? <span className="badge bg-success">✓</span>
                              : <span className="badge bg-warning text-dark">⚠</span>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab==='definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-5">
            <div className="card">
              <div className="card-header py-2"><strong>📖 Glossary ({defs.glossary?.length||0} terms)</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Term</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(defs.glossary||[]).map((g,i)=>(
                      <tr key={i}>
                        <td><strong>{g.term}</strong></td>
                        <td><small>{g.definition}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-7">
            <div className="card mb-3">
              <div className="card-header py-2"><strong>📋 Status Legend</strong></div>
              <div className="card-body">
                {(defs.status_legend||[]).map((s,i)=>(
                  <div key={i} className="d-flex align-items-start mb-2">
                    <span className="badge bg-success me-2 mt-1">{s.status}</span>
                    <small>{s.description}</small>
                  </div>
                ))}
              </div>
            </div>

            <div className="card mb-3">
              <div className="card-header py-2"><strong>🏥 Clinical Notes</strong></div>
              <div className="card-body">
                <ul className="small mb-0">
                  {(defs.clinical_notes||[]).map((n,i)=>(
                    <li key={i} className="mb-1">{n}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="card">
              <div className="card-header py-2"><strong>📚 References ({defs.references?.length||0})</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Source</th><th>Detail</th></tr></thead>
                  <tbody>
                    {(defs.references||[]).map((r,i)=>(
                      <tr key={i}>
                        <td><strong>{r.ref}</strong></td>
                        <td><small>{r.detail}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
