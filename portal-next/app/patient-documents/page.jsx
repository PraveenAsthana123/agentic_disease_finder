'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function PatientDocumentsDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/patient-documents/overview`).then(r=>r.json()),
      fetch(`${API}/api/patient-documents/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/patient-documents/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading patient documents data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'types', label:'By Type'},
    {id:'patients', label:'Per Patient'},
    {id:'recent', label:'Recent'},
    {id:'definitions', label:'Definitions'},
  ];

  const kpis = ov.kpis;
  const kpiMap = {};
  kpis.forEach(k=>{ kpiMap[k.label]=k; });

  const totalDocs = kpiMap['Total Documents']?.value || 0;
  const patients = kpiMap['Patients']?.value || 0;
  const docTypes = kpiMap['Document Types']?.value || 0;
  const shared = kpiMap['Shared with Patient']?.value || 0;
  const downloaded = kpiMap['Downloaded']?.value || 0;
  const avgSize = kpiMap['Avg File Size']?.value || '—';
  const totalStorage = kpiMap['Total Storage']?.value || '—';

  const shareRate = kpiMap['Shared with Patient']?.sub || '';
  const downloadRate = kpiMap['Downloaded']?.sub || '';

  const catColors = {clinical:'#3b82f6', administrative:'#f59e0b', educational:'#22c55e'};

  return (<div className="p-3">
    <h3>📄 Patient Documents Dashboard</h3>
    <p className="text-muted">
      Document management &mdash; {totalDocs} documents · {patients} patients · {docTypes} types ·{' '}
      {shareRate} share rate · {downloadRate} download rate
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      {/* KPI row */}
      <div className="row mb-3">
        {[
          ['Total Documents', totalDocs, 'primary'],
          ['Patients', patients, 'info'],
          ['Document Types', docTypes, 'success'],
          ['Shared', shared, 'warning'],
          ['Downloaded', downloaded, 'secondary'],
          ['Total Storage', totalStorage, 'dark'],
        ].map(([label,val,c])=>
          <div key={label} className="col-6 col-md-2 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className={`h5 mb-0 text-${c}`}>{val}</div>
              <div className="text-muted small">{label}</div>
            </div></div>
          </div>
        )}
      </div>

      <div className="row mb-3">
        {/* Type distribution */}
        <div className="col-md-5">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Document Types</h6>
            {ov.type_distribution.map(d=>{
              const pct = Math.round((d.value/totalDocs)*100);
              return <div key={d.name} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'150px'}}>{d.name}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-primary" style={{width:`${pct}%`}}>
                      {d.value} ({pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>

        {/* Category + sharing status */}
        <div className="col-md-7">
          <div className="row">
            <div className="col-12 mb-3">
              <div className="card shadow-sm"><div className="card-body">
                <h6>Category Breakdown</h6>
                <div className="d-flex gap-3 flex-wrap">
                  {ov.category_distribution.map(c=>{
                    const pct = Math.round((c.value/totalDocs)*100);
                    return <div key={c.name} className="text-center flex-fill">
                      <div className="h4 mb-0" style={{color: catColors[c.name]||'#6b7280'}}>{c.value}</div>
                      <div className="small text-muted text-capitalize">{c.name}</div>
                      <div className="small">({pct}%)</div>
                    </div>;
                  })}
                </div>
              </div></div>
            </div>
            <div className="col-12">
              <div className="card shadow-sm"><div className="card-body">
                <h6>Sharing Status</h6>
                {ov.sharing_status.map(s=>{
                  const pct = Math.round((s.value/totalDocs)*100);
                  return <div key={s.name} className="d-flex align-items-center mb-2">
                    <span className="me-2 small" style={{minWidth:'170px'}}>{s.name}</span>
                    <div className="flex-grow-1 me-2">
                      <div className="progress" style={{height:'20px'}}>
                        <div className="progress-bar" style={{width:`${pct}%`, backgroundColor: s.color||'#6b7280'}}>
                          {s.value} ({pct}%)
                        </div>
                      </div>
                    </div>
                  </div>;
                })}
              </div></div>
            </div>
          </div>
        </div>
      </div>

      {/* Monthly trend */}
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Monthly Upload Trend</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Month</th><th>Uploads</th><th>Trend</th></tr></thead>
          <tbody>
            {ov.monthly_trend.map((m,i)=>{
              const prev = i>0 ? ov.monthly_trend[i-1].uploads : m.uploads;
              const diff = m.uploads - prev;
              return <tr key={m.month}>
                <td>{m.month}</td>
                <td className="fw-bold">{m.uploads}</td>
                <td>{i===0?'—':diff>0?<span className="text-success">+{diff}</span>:diff<0?<span className="text-danger">{diff}</span>:<span className="text-muted">—</span>}</td>
              </tr>;
            })}
          </tbody>
        </table>
      </div></div>
    </div>}

    {tab==='types' && bd && <div>
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Type × Category Matrix</h6>
            <div className="table-responsive">
              <table className="table table-sm table-striped table-bordered">
                <thead><tr><th>Document Type</th><th>Category</th><th>Count</th></tr></thead>
                <tbody>
                  {bd.type_category.map((tc,i)=>
                    <tr key={i}>
                      <td>{tc.document_type}</td>
                      <td>
                        <span className="badge text-capitalize"
                          style={{backgroundColor: catColors[tc.category]||'#6b7280'}}>
                          {tc.category}
                        </span>
                      </td>
                      <td>{tc.count}</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>File Size by Document Type</h6>
            <div className="table-responsive">
              <table className="table table-sm table-striped table-bordered">
                <thead><tr>
                  <th>Type</th><th>Count</th><th>Avg KB</th><th>Max KB</th><th>Total MB</th>
                </tr></thead>
                <tbody>
                  {bd.size_by_type.map(s=>
                    <tr key={s.document_type}>
                      <td className="fw-bold">{s.document_type}</td>
                      <td>{s.count}</td>
                      <td>{s.avg_kb.toLocaleString()}</td>
                      <td>{s.max_kb.toLocaleString()}</td>
                      <td>{s.total_mb.toFixed(1)}</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div></div>
        </div>
      </div>
    </div>}

    {tab==='patients' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Per-Patient Document Summary ({bd.per_patient.length} patients)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Patient</th><th>Docs</th><th>Storage (MB)</th><th>Shared</th><th>Downloaded</th>
              <th>Share Rate</th><th>Types</th><th>First Upload</th><th>Latest Upload</th>
            </tr></thead>
            <tbody>
              {bd.per_patient.map(p=>
                <tr key={p.patient_id}>
                  <td className="fw-bold font-monospace">{p.patient_id}</td>
                  <td>{p.doc_count}</td>
                  <td>{p.total_size_mb.toFixed(1)}</td>
                  <td>{p.shared}</td>
                  <td>{p.downloaded}</td>
                  <td>
                    <span className={`badge bg-${p.share_rate>=80?'success':p.share_rate>=50?'warning':'danger'}`}>
                      {p.share_rate.toFixed(0)}%
                    </span>
                  </td>
                  <td>{p.type_count}</td>
                  <td className="text-muted small">{p.first_upload}</td>
                  <td className="text-muted small">{p.last_upload}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='recent' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Recent Documents (latest {bd.recent_documents.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped table-bordered">
            <thead><tr>
              <th>Upload Date</th><th>Patient</th><th>Type</th><th>Category</th>
              <th>Size (KB)</th><th>Shared</th><th>Downloaded</th>
            </tr></thead>
            <tbody>
              {bd.recent_documents.map((doc,i)=>
                <tr key={i}>
                  <td>{doc.upload_date}</td>
                  <td className="font-monospace">{doc.patient_id}</td>
                  <td className="fw-bold">{doc.document_type}</td>
                  <td>
                    <span className="badge text-capitalize"
                      style={{backgroundColor: catColors[doc.category]||'#6b7280'}}>
                      {doc.category}
                    </span>
                  </td>
                  <td>{doc.file_size_kb.toLocaleString()}</td>
                  <td>{doc.shared
                    ? <span className="badge bg-success">Shared</span>
                    : <span className="badge bg-secondary">No</span>}</td>
                  <td>{doc.downloaded
                    ? <span className="badge bg-info">Yes</span>
                    : <span className="badge bg-light text-dark">No</span>}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Document Types</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Type</th><th>Description</th></tr></thead>
          <tbody>
            {defs.document_types.map(d=>
              <tr key={d.type}><td className="fw-bold">{d.type}</td><td>{d.description}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>

      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Categories</h6>
            {defs.categories.map(c=>
              <div key={c.name} className="mb-2">
                <span className="badge me-2 text-capitalize"
                  style={{backgroundColor: c.color||'#6b7280'}}>{c.name}</span>
                <span className="small">{c.description}</span>
              </div>
            )}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Sharing Workflow</h6>
            <p className="small">{defs.sharing_workflow?.description}</p>
            {defs.sharing_workflow?.statuses?.map(s=>
              <div key={s.status} className="mb-1">
                <strong className="small">{s.status}:</strong>{' '}
                <span className="small text-muted">{s.description}</span>
              </div>
            )}
          </div></div>
        </div>
      </div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Glossary</h6>
        <table className="table table-sm table-bordered">
          <thead><tr><th>Term</th><th>Definition</th></tr></thead>
          <tbody>
            {defs.glossary.map(g=>
              <tr key={g.term}><td className="fw-bold">{g.term}</td><td>{g.definition}</td></tr>
            )}
          </tbody>
        </table>
      </div></div>

      {defs.clinical_note && <div className="alert alert-info">
        <strong>Clinical Note:</strong> {defs.clinical_note}
      </div>}
    </div>}
  </div>);
}
