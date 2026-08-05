'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function ConsultantWorkflowsDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [selectedRole, setSelectedRole] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/consultant-workflows/overview`).then(r=>r.json()),
      fetch(`${API}/api/consultant-workflows/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/consultant-workflows/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{
      setOv(o); setBd(b); setDefs(d);
      if(b?.roles?.length) setSelectedRole(b.roles[0].role_id);
    }).catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading consultant workflows...</div>;

  const TABS=[
    {id:'overview', label:'Overview'},
    {id:'workflows', label:'Workflows'},
    {id:'phases', label:'Phase Details'},
    {id:'definitions', label:'Definitions'},
  ];

  const roleColor = i => ['#6366f1','#22c55e','#f59e0b','#ef4444','#06b6d4','#8b5cf6'][i%6];

  const activeRole = bd?.roles?.find(r=>r.role_id===selectedRole);

  return (<div className="p-3">
    <h3>Consultant Process Workflows Dashboard</h3>
    <p className="text-muted">
      Human clinical oversight — {ov.summary.total_roles} consultant roles,{' '}
      {ov.summary.total_phases} phases, {ov.summary.total_steps} process steps,{' '}
      {ov.summary.total_signoffs} sign-off gates
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {[
          {label:'Consultant Roles', value:ov.summary.total_roles, color:'#6366f1'},
          {label:'Process Phases', value:ov.summary.total_phases, color:'#22c55e'},
          {label:'Process Steps', value:ov.summary.total_steps, color:'#f59e0b'},
          {label:'Sign-Off Gates', value:ov.summary.total_signoffs, color:'#ef4444'},
          {label:'Mandatory Roles', value:ov.summary.mandatory_roles, color:'#06b6d4'},
          {label:'Avg Phases/Role', value:ov.summary.avg_phases_per_role, color:'#8b5cf6'},
        ].map(k=><div key={k.label} className="col-6 col-md-2 mb-2">
          <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
            <div className="h4 mb-0 fw-bold" style={{color:k.color}}>{k.value}</div>
            <div className="text-muted small">{k.label}</div>
          </div></div>
        </div>)}
      </div>

      <div className="row mb-3">
        <div className="col-md-5">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Phases per Consultant Role</h6>
            {ov.phase_distribution.map((r,i)=>{
              const max = Math.max(...ov.phase_distribution.map(x=>x.value));
              const pct = max>0 ? ((r.value/max)*100).toFixed(0) : 0;
              return <div key={r.name} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{r.name}</span><span className="fw-bold">{r.value} phases</span>
                </div>
                <div className="progress" style={{height:'16px'}}>
                  <div className="progress-bar" style={{width:`${pct}%`, backgroundColor:roleColor(i)}}/>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Sign-Off Gates per Role</h6>
            {ov.signoff_distribution.map((r,i)=>{
              const max = Math.max(...ov.signoff_distribution.map(x=>x.value));
              const pct = max>0 ? ((r.value/max)*100).toFixed(0) : 0;
              return <div key={r.name} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{r.name}</span><span className="fw-bold">{r.value}</span>
                </div>
                <div className="progress" style={{height:'14px'}}>
                  <div className="progress-bar bg-danger" style={{width:`${pct}%`, opacity:0.8}}/>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-3">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Role Summary</h6>
            {ov.role_summary.map((r,i)=><div key={r.role_id} className="mb-2 pb-2 border-bottom">
              <div className="fw-bold small" style={{color:roleColor(i)}}>{r.name}</div>
              <div className="d-flex gap-2 mt-1">
                <span className="badge" style={{backgroundColor:roleColor(i)}}>{r.phases} phases</span>
                <span className="badge bg-secondary">{r.steps} steps</span>
                <span className="badge bg-danger">{r.signoffs} gates</span>
              </div>
            </div>)}
          </div></div>
        </div>
      </div>
    </div>}

    {tab==='workflows' && bd && <div>
      <div className="row mb-3">
        {bd.roles.map((r,i)=><div key={r.role_id} className="col-md-6 mb-3">
          <div className="card shadow-sm h-100" style={{borderLeft:`4px solid ${roleColor(i)}`}}>
            <div className="card-body">
              <h6 style={{color:roleColor(i)}}>{r.name}</h6>
              <p className="text-muted small mb-2">{r.summary}</p>
              <div className="d-flex gap-2 mb-3">
                <span className="badge" style={{backgroundColor:roleColor(i)}}>{r.phases.length} phases</span>
                <span className="badge bg-secondary">{r.phases.reduce((a,p)=>a+p.step_count,0)} steps</span>
              </div>
              {r.phases.map((ph,pi)=><div key={pi} className="mb-2">
                <div className="fw-bold small text-secondary">{ph.name}</div>
                {ph.steps.map((st,si)=><div key={si} className="ms-2 mb-1 p-1 rounded" style={{backgroundColor:'#f8f9fa'}}>
                  <div className="fw-bold small">{st.step}</div>
                  <div className="text-muted" style={{fontSize:'0.75rem'}}>
                    <span className="text-primary">In:</span> {st.input} &rarr; <span className="text-success">Out:</span> {st.output}
                  </div>
                </div>)}
              </div>)}
            </div>
          </div>
        </div>)}
      </div>
    </div>}

    {tab==='phases' && bd && <div>
      <div className="mb-3">
        <div className="d-flex gap-2 flex-wrap">
          {bd.roles.map((r,i)=><button key={r.role_id}
            className={`btn btn-sm ${selectedRole===r.role_id?'btn-primary':'btn-outline-secondary'}`}
            style={selectedRole===r.role_id?{backgroundColor:roleColor(i),borderColor:roleColor(i)}:{}}
            onClick={()=>setSelectedRole(r.role_id)}>
            {r.name}
          </button>)}
        </div>
      </div>

      {activeRole && <div>
        <h5 style={{color:roleColor(bd.roles.findIndex(r=>r.role_id===selectedRole))}}>{activeRole.name}</h5>
        <p className="text-muted small mb-3">{activeRole.summary}</p>
        {activeRole.phases.map((ph,pi)=><div key={pi} className="card shadow-sm mb-3">
          <div className="card-header py-2 d-flex justify-content-between align-items-center"
            style={{backgroundColor:'#f0f4ff'}}>
            <span className="fw-bold">{ph.name}</span>
            <span className="badge bg-primary">{ph.step_count} steps</span>
          </div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr><th style={{width:'20%'}}>Step</th><th style={{width:'25%'}}>Input</th><th style={{width:'35%'}}>Task</th><th style={{width:'20%'}}>Output</th></tr>
              </thead>
              <tbody>
                {ph.steps.map((st,si)=><tr key={si}>
                  <td className="fw-bold small">{st.step}</td>
                  <td className="text-muted small">{st.input}</td>
                  <td className="small">{st.task}</td>
                  <td><span className="badge bg-success text-wrap" style={{fontSize:'0.7rem'}}>{st.output}</span></td>
                </tr>)}
              </tbody>
            </table>
          </div>
        </div>)}
      </div>}
    </div>}

    {tab==='definitions' && defs && <div>
      <div className="row">
        <div className="col-md-5">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Consultant Roles</h6>
            {defs.roles.map((r,i)=><div key={r.id} className="mb-3 pb-2 border-bottom">
              <div className="fw-bold small" style={{color:roleColor(i)}}>{r.name}</div>
              <div className="text-muted small">{r.summary}</div>
            </div>)}
          </div></div>
        </div>
        <div className="col-md-7">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Glossary</h6>
            <div className="table-responsive">
              <table className="table table-sm">
                <thead><tr><th style={{width:'25%'}}>Term</th><th>Definition</th></tr></thead>
                <tbody>
                  {defs.glossary.map(g=><tr key={g.term}>
                    <td><strong>{g.term}</strong></td>
                    <td className="small">{g.definition}</td>
                  </tr>)}
                </tbody>
              </table>
            </div>
          </div></div>
        </div>
      </div>
    </div>}
  </div>);
}
