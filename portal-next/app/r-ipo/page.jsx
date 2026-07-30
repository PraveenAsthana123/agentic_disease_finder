'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusColor = s => ({'built':'success','partial':'warning','planned':'secondary'}[s]||'info');
const phaseColor = p => ({'input':'primary','process':'warning','output':'success'}[p]||'secondary');

export default function RoleIPODashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/r-ipo/overview`).then(r=>r.json()),
      fetch(`${API}/api/r-ipo/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/r-ipo/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading role pipeline data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'pipelines', label:'Role Pipelines'},
    {id:'matrix', label:'Cross-Role Matrix'},
    {id:'definitions', label:'Definitions'},
  ];

  return (<div className="p-3">
    <h3>Role IPO Pipeline Dashboard</h3>
    <p className="text-muted">
      Input &rarr; Process &rarr; Output pipelines per clinical role &mdash;
      {' '}{ov.kpis?.find(k=>k.label==='Total Roles')?.value} roles,
      {' '}{ov.kpis?.find(k=>k.label==='Total Pipeline Steps')?.value} total stages
    </p>
    {ov.honest_note && <div className="alert alert-info small py-1 px-2">{ov.honest_note}</div>}

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {(ov.kpis||[]).map(k=>{
          const c = k.color==='green'?'success':k.color==='red'?'danger':k.color==='yellow'?'warning':k.color==='blue'?'primary':k.color==='gray'?'secondary':'info';
          return <div key={k.label} className="col-6 col-md-2 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className={`h5 mb-0 text-${c}`}>{k.value}</div>
              <div className="text-muted small">{k.label}</div>
            </div></div>
          </div>;
        })}
      </div>

      <div className="row mb-3">
        <div className="col-md-5">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Status Distribution</h6>
            {(ov.status_distribution||[]).map(({name,value})=>{
              const total = ov.roles?.length || 1;
              const pct = ((value/total)*100).toFixed(0);
              return <div key={name} className="d-flex align-items-center mb-2">
                <span className="me-2 small" style={{minWidth:'70px'}}>{name}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'20px'}}>
                    <div className={`progress-bar bg-${statusColor(name)}`} style={{width:`${pct}%`}}>
                      {value} ({pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-7">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Role Summary</h6>
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0">
                <thead><tr><th>Role</th><th>Steps</th><th>Priority</th><th>Status</th></tr></thead>
                <tbody>
                  {(ov.roles||[]).map(r=>
                    <tr key={r.name}>
                      <td className="fw-semibold">{r.name}</td>
                      <td>{r.step_count}</td>
                      <td>{r.priority}</td>
                      <td><span className={`badge bg-${statusColor(r.status)}`}>{r.status}</span></td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div></div>
        </div>
      </div>
    </div>}

    {tab==='pipelines' && bd && <div>
      <h5>Per-Role Pipelines</h5>
      {Object.entries(bd.pipelines||{}).map(([role, pdata])=>
        <div key={role} className="card shadow-sm mb-3"><div className="card-body">
          <h6>{role} <span className="text-muted small">({pdata.total} stages)</span></h6>
          <div className="d-flex flex-wrap gap-1 align-items-center mb-3">
            {(pdata.stages||[]).map((s, i)=>
              <span key={i} className="d-flex align-items-center">
                <span className={`badge bg-${phaseColor(s.phase)} px-2 py-1`}>
                  {s.label}
                </span>
                {i < pdata.stages.length - 1 && <span className="mx-1 text-muted">&rarr;</span>}
              </span>
            )}
          </div>
          {pdata.mermaid && <details className="mb-0">
            <summary className="small text-muted" style={{cursor:'pointer'}}>Mermaid diagram source</summary>
            <pre className="bg-light p-2 mt-1 small mb-0" style={{maxHeight:'200px', overflow:'auto'}}>{pdata.mermaid}</pre>
          </details>}
        </div></div>
      )}
    </div>}

    {tab==='matrix' && bd && <div>
      <h5>Cross-Role Step Comparison</h5>
      <p className="text-muted small">Which pipeline steps appear in which roles</p>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <div className="table-responsive">
          <table className="table table-sm table-bordered" style={{fontSize:'0.75rem'}}>
            <thead><tr>
              <th style={{position:'sticky',left:0,background:'#fff',zIndex:1}}>Role</th>
              {(bd.cross_matrix?.step_labels||[]).map(sl=>
                <th key={sl} className="text-center" style={{writingMode:'vertical-rl',transform:'rotate(180deg)',maxWidth:'30px',whiteSpace:'nowrap'}}>{sl}</th>
              )}
            </tr></thead>
            <tbody>
              {(bd.cross_matrix?.rows||[]).map(r=>
                <tr key={r.role}>
                  <td style={{position:'sticky',left:0,background:'#fff',zIndex:1}} className="fw-semibold small">{r.role}</td>
                  {(bd.cross_matrix?.step_labels||[]).map(sl=>
                    <td key={sl} className="text-center">
                      {r[sl] ? <span className="text-success">&#10003;</span> : ''}
                    </td>
                  )}
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      <h5>Definitions &amp; Reference</h5>
      <div className="row">
        <div className="col-md-4">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Pipeline Phases</h6>
            {(defs.phases||[]).map(p=>
              <div key={p.name} className="mb-2">
                <span className={`badge bg-${phaseColor(p.name.toLowerCase())} me-2`}>{p.name}</span>
                <span className="small">{p.description}</span>
              </div>
            )}
          </div></div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Quality Gates</h6>
            {(defs.quality_gates||[]).map(g=>
              <div key={g.gate} className="mb-2">
                <strong className="small">{g.gate}:</strong>{' '}
                <span className="small text-muted">{g.description}</span>
              </div>
            )}
          </div></div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Status Legend</h6>
            {(defs.status_legend||[]).map(s=>
              <div key={s.status} className="mb-2">
                <span className={`badge bg-${statusColor(s.status)} me-2`}>{s.status}</span>
                <span className="small">{s.description}</span>
              </div>
            )}
          </div></div>
        </div>
      </div>
    </div>}
  </div>);
}
