'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function BusinessWorkflowsDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/business-workflows/overview`).then(r=>r.json()),
      fetch(`${API}/api/business-workflows/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/business-workflows/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading business workflows data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'workflows', label:'Workflow Types'},
    {id:'active', label:'Active'},
    {id:'failed', label:'Failed'},
    {id:'patients', label:'Per Patient'},
    {id:'definitions', label:'Definitions'},
  ];

  const statusColor = s => ({active:'primary',completed:'success',failed:'danger',paused:'warning',pending:'secondary'}[s]||'info');
  const priorityColor = p => ({critical:'danger',high:'warning',medium:'primary',low:'secondary'}[p]||'info');

  const kpis = [
    {label:'Total Workflows', value:ov.total_workflows, color:'#3b82f6'},
    {label:'Patients', value:ov.total_patients, color:'#8b5cf6'},
    {label:'Active', value:ov.active_workflows, color:'#06b6d4'},
    {label:'Workflow Types', value:ov.total_workflow_types, color:'#64748b'},
    {label:'Completion Rate', value:ov.completion_rate+'%', color:'#10b981'},
    {label:'Failure Rate', value:ov.failure_rate+'%', color:'#ef4444'},
    {label:'Avg Duration', value:Math.round(ov.avg_duration_min)+'m', color:'#f59e0b'},
    {label:'Avg Step %', value:ov.avg_step_completion_pct+'%', color:'#6366f1'},
  ];

  return (<div className="p-3">
    <h3>Business Workflows Dashboard</h3>
    <p className="text-muted">
      Clinical &amp; administrative workflow automation &mdash; {ov.total_workflows} workflows,{' '}
      {ov.total_patients} patients, {ov.total_workflow_types} types,{' '}
      {ov.completion_rate}% completion rate
    </p>

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {tab==='overview' && <div>
      <div className="row mb-3">
        {kpis.map(k=>
          <div key={k.label} className="col-6 col-md-3 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className="h5 mb-0" style={{color:k.color}}>{k.value}</div>
              <div className="text-muted small">{k.label}</div>
            </div></div>
          </div>
        )}
      </div>

      <div className="row mb-3">
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Status Distribution</h6>
            {ov.status_distribution.map(s=>{
              const pct = ((s.count/ov.total_workflows)*100).toFixed(1);
              return <div key={s.status} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span className="text-capitalize">{s.status}</span>
                  <span>{s.count} ({pct}%)</span>
                </div>
                <div className="progress" style={{height:'18px'}}>
                  <div className={`progress-bar bg-${statusColor(s.status)}`} style={{width:`${pct}%`}}>
                    {parseFloat(pct)>10 && s.count}
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>

        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Category Distribution</h6>
            {ov.category_distribution.map(c=>{
              const pct = ((c.count/ov.total_workflows)*100).toFixed(0);
              const colors = {Administrative:'info',Clinical:'success',Technical:'primary',Compliance:'warning'};
              return <div key={c.category} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{c.category}</span><span>{c.count} ({pct}%)</span>
                </div>
                <div className="progress" style={{height:'16px'}}>
                  <div className={`progress-bar bg-${colors[c.category]||'secondary'}`} style={{width:`${pct}%`}}/>
                </div>
              </div>;
            })}
          </div></div>
        </div>

        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Priority &amp; Trigger</h6>
            <div className="mb-3">
              {ov.priority_distribution.map(p=>{
                const pct = ((p.count/ov.total_workflows)*100).toFixed(0);
                return <div key={p.priority} className="mb-1">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="text-capitalize">{p.priority}</span><span>{p.count}</span>
                  </div>
                  <div className="progress" style={{height:'12px'}}>
                    <div className={`progress-bar bg-${priorityColor(p.priority)}`} style={{width:`${pct}%`}}/>
                  </div>
                </div>;
              })}
            </div>
            <h6>Trigger Type</h6>
            {ov.trigger_distribution.map(t=>{
              const pct = ((t.count/ov.total_workflows)*100).toFixed(0);
              return <div key={t.trigger_type} className="mb-1">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{t.trigger_type}</span><span>{t.count}</span>
                </div>
                <div className="progress" style={{height:'12px'}}>
                  <div className="progress-bar bg-dark" style={{width:`${pct}%`}}/>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>

      {bd && <div className="card shadow-sm"><div className="card-body">
        <h6>Owner Workload</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr><th>Owner</th><th>Total</th><th>Completed</th><th>Active</th><th>Failed</th><th>Avg Duration</th></tr></thead>
            <tbody>
              {bd.owner_workload.map(o=><tr key={o.owner}>
                <td><span className="badge bg-secondary">{o.owner}</span></td>
                <td>{o.total}</td>
                <td><span className="text-success fw-bold">{o.completed}</span></td>
                <td><span className="text-primary">{o.active}</span></td>
                <td><span className={o.failed>0?'text-danger fw-bold':''}>{o.failed}</span></td>
                <td>{o.avg_duration_min ? Math.round(o.avg_duration_min)+'m' : '-'}</td>
              </tr>)}
            </tbody>
          </table>
        </div>
      </div></div>}
    </div>}

    {tab==='workflows' && bd && <div>
      <div className="row">
        {bd.workflow_stats.map(w=><div key={w.workflow_name} className="col-md-6 mb-3">
          <div className="card shadow-sm h-100"><div className="card-body">
            <h6>{w.workflow_name}</h6>
            <div className="d-flex gap-2 flex-wrap mb-2">
              <span className="badge bg-secondary">{w.total} runs</span>
              <span className="badge bg-success">{w.completed} done</span>
              {w.failed>0 && <span className="badge bg-danger">{w.failed} failed</span>}
            </div>
            <div className="mb-2">
              <div className="d-flex justify-content-between small mb-1">
                <span>Avg Step Completion</span>
                <span>{w.avg_step_pct}%</span>
              </div>
              <div className="progress" style={{height:'16px'}}>
                <div className={`progress-bar ${w.avg_step_pct>=75?'bg-success':w.avg_step_pct>=50?'bg-warning':'bg-danger'}`}
                  style={{width:`${w.avg_step_pct}%`}}/>
              </div>
            </div>
            <div className="small text-muted">
              Avg duration: {w.avg_duration_min ? Math.round(w.avg_duration_min)+'m' : 'N/A'}
            </div>
          </div></div>
        </div>)}
      </div>

      {bd.category_status_crosstab && <div className="card shadow-sm mt-3"><div className="card-body">
        <h6>Category × Status Matrix</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead><tr>
              <th>Category</th>
              {Object.keys(bd.category_status_crosstab[0]||{}).filter(k=>k!=='category').map(s=>
                <th key={s} className="text-capitalize">{s}</th>
              )}
            </tr></thead>
            <tbody>
              {bd.category_status_crosstab.map(row=><tr key={row.category}>
                <td><strong>{row.category}</strong></td>
                {Object.entries(row).filter(([k])=>k!=='category').map(([s,v])=>
                  <td key={s}><span className={`badge bg-${statusColor(s)}`}>{v||0}</span></td>
                )}
              </tr>)}
            </tbody>
          </table>
        </div>
      </div></div>}
    </div>}

    {tab==='active' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Active Workflows ({bd.active_workflows.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr>
              <th>ID</th><th>Name</th><th>Category</th><th>Priority</th><th>Trigger</th>
              <th>Owner</th><th>Patient</th><th>Steps</th><th>Started</th>
            </tr></thead>
            <tbody>
              {bd.active_workflows.map(w=><tr key={w.workflow_id}>
                <td><code className="small">{w.workflow_id}</code></td>
                <td>{w.workflow_name}</td>
                <td><span className="badge bg-info text-dark">{w.category}</span></td>
                <td><span className={`badge bg-${priorityColor(w.priority)}`}>{w.priority}</span></td>
                <td><span className="small">{w.trigger_type}</span></td>
                <td>{w.owner}</td>
                <td>{w.patient_id||'-'}</td>
                <td>
                  <div className="progress" style={{height:'16px',minWidth:'60px'}}>
                    <div className="progress-bar bg-primary"
                      style={{width:`${(w.steps_completed/w.steps_total)*100}%`}}>
                      {w.steps_completed}/{w.steps_total}
                    </div>
                  </div>
                </td>
                <td className="small">{w.created_at?.replace('T',' ').substring(0,16)}</td>
              </tr>)}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='failed' && bd && <div>
      {bd.failed_workflows.length===0
        ? <div className="alert alert-success">No failed workflows.</div>
        : <div className="card shadow-sm"><div className="card-body">
          <h6>Failed Workflows ({bd.failed_workflows.length})</h6>
          <div className="table-responsive">
            <table className="table table-sm table-danger">
              <thead><tr>
                <th>ID</th><th>Name</th><th>Category</th><th>Owner</th><th>Patient</th>
                <th>Retries</th><th>Error</th><th>Created</th>
              </tr></thead>
              <tbody>
                {bd.failed_workflows.map(w=><tr key={w.workflow_id}>
                  <td><code className="small">{w.workflow_id}</code></td>
                  <td>{w.workflow_name}</td>
                  <td>{w.category}</td>
                  <td>{w.owner}</td>
                  <td>{w.patient_id||'-'}</td>
                  <td>{w.retry_count}</td>
                  <td><span className="small text-danger">{w.error_message||'Unknown error'}</span></td>
                  <td className="small">{w.created_at?.replace('T',' ').substring(0,16)}</td>
                </tr>)}
              </tbody>
            </table>
          </div>
        </div></div>}

      {bd.recent_workflows && <div className="card shadow-sm mt-3"><div className="card-body">
        <h6>Recent Workflows</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr><th>ID</th><th>Name</th><th>Status</th><th>Priority</th><th>Owner</th><th>Patient</th><th>Created</th></tr></thead>
            <tbody>
              {bd.recent_workflows.slice(0,15).map(w=><tr key={w.workflow_id}>
                <td><code className="small">{w.workflow_id}</code></td>
                <td>{w.workflow_name}</td>
                <td><span className={`badge bg-${statusColor(w.status)}`}>{w.status}</span></td>
                <td><span className={`badge bg-${priorityColor(w.priority)}`}>{w.priority}</span></td>
                <td>{w.owner}</td>
                <td>{w.patient_id||'-'}</td>
                <td className="small">{w.created_at?.replace('T',' ').substring(0,16)}</td>
              </tr>)}
            </tbody>
          </table>
        </div>
      </div></div>}
    </div>}

    {tab==='patients' && bd && <div>
      <div className="card shadow-sm"><div className="card-body">
        <h6>Per-Patient Workflows ({bd.per_patient.length} patients)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead><tr><th>Patient</th><th>Total</th><th>Active</th><th>Completed</th><th>Failed</th><th>Workflow Types</th></tr></thead>
            <tbody>
              {bd.per_patient.sort((a,b)=>b.total-a.total).map(p=><tr key={p.patient_id}>
                <td>{p.patient_id}</td>
                <td>{p.total}</td>
                <td><span className="text-primary">{p.active}</span></td>
                <td><span className="text-success">{p.completed}</span></td>
                <td><span className={p.failed>0?'text-danger fw-bold':''}>{p.failed}</span></td>
                <td><span className="small">{p.workflow_types}</span></td>
              </tr>)}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {tab==='definitions' && defs && <div>
      <div className="row">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Workflow Types</h6>
            {Object.entries(defs.workflow_types).map(([name,desc])=><div key={name} className="mb-2">
              <div className="fw-bold small">{name}</div>
              <div className="text-muted small">{desc}</div>
            </div>)}
          </div></div>

          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Categories</h6>
            {Object.entries(defs.categories).map(([cat,desc])=><div key={cat} className="mb-2">
              <div className="fw-bold small">{cat}</div>
              <div className="text-muted small">{desc}</div>
            </div>)}
          </div></div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Statuses</h6>
            {Object.entries(defs.statuses).map(([s,desc])=><div key={s} className="mb-1 d-flex gap-2">
              <span className={`badge bg-${statusColor(s)} text-nowrap`}>{s}</span>
              <span className="small text-muted">{desc}</span>
            </div>)}
          </div></div>

          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Priority Levels</h6>
            {Object.entries(defs.priorities).map(([p,desc])=><div key={p} className="mb-1 d-flex gap-2">
              <span className={`badge bg-${priorityColor(p)} text-nowrap`}>{p}</span>
              <span className="small text-muted">{desc}</span>
            </div>)}
          </div></div>

          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Trigger Types</h6>
            {Object.entries(defs.trigger_types).map(([t,desc])=><div key={t} className="mb-1">
              <span className="badge bg-dark me-2">{t}</span>
              <span className="small text-muted">{desc}</span>
            </div>)}
          </div></div>

          <div className="card shadow-sm"><div className="card-body">
            <h6>Glossary</h6>
            {Object.entries(defs.glossary).map(([term,def])=><div key={term} className="mb-1">
              <span className="fw-bold small">{term}: </span>
              <span className="text-muted small">{def}</span>
            </div>)}
          </div></div>
        </div>
      </div>
    </div>}
  </div>);
}
